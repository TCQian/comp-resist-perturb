# This script applies an adaptive frequency attack to each training image before performing actual JPEG compression.
# The attack is applied in the DCT domain (per channel) using an adaptive mask (based on Sobel gradients).
# After JPEG compression the adaptive perturbation remains more intact.
# The training loop is otherwise identical.
# sd_train_adaptive_perturb.py
import argparse
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    StableDiffusionPipeline,
)
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
from io import BytesIO
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt


def preprocess(example):
    transform = transforms.Compose(
        [
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
        ]
    )
    image = example["image"]
    example["pixel_values"] = transform(image)
    if isinstance(example["caption"], list):
        example["text"] = example["caption"][0]
    else:
        example["text"] = example["caption"]
    return example


def adaptive_frequency_attack(image, steps=5, alpha=0.005):
    """
    Applies an adaptive frequency perturbation in the DCT domain on a per-channel basis.
    """
    perturbed = image.clone().cpu().numpy()
    for _ in range(steps):
        update = np.zeros_like(perturbed)
        for c in range(perturbed.shape[0]):
            channel = perturbed[c]
            dct_channel = cv2.dct(channel.astype(np.float32))
            abs_dct = cv2.convertScaleAbs(dct_channel)
            grad_x = cv2.Sobel(abs_dct, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(abs_dct, cv2.CV_64F, 0, 1, ksize=3)
            mask = np.sqrt(grad_x**2 + grad_y**2)
            mask = mask / (mask.max() + 1e-8)
            dct_perturb = alpha * mask
            dct_channel_updated = dct_channel + dct_perturb
            channel_updated = cv2.idct(dct_channel_updated)
            update[c] = channel_updated - channel
        perturbed = perturbed + update
        perturbed = np.clip(perturbed, 0, 1)
    return torch.tensor(perturbed)


def actual_jpeg_compression(image_tensor, quality=75):
    pil_image = transforms.ToPILImage()(image_tensor.cpu())
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed_image = Image.open(buffer).convert("RGB")
    return transforms.ToTensor()(compressed_image)


def evaluate_model(pipeline, device, prompts, num_samples=4):
    from transformers import CLIPProcessor, CLIPModel

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    scores = []
    for prompt in prompts:
        generated = pipeline(
            prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            num_images_per_prompt=num_samples,
        ).images
        inputs = clip_processor(
            text=[prompt] * num_samples,
            images=generated,
            return_tensors="pt",
            padding=True,
        ).to(device)
        outputs = clip_model(**inputs)
        score = outputs.logits_per_image.diag().mean().item()
        scores.append(score)
        print(f"Prompt: {prompt} - CLIP Score: {score:.4f}")
    avg_score = sum(scores) / len(scores)
    print("Average CLIP Score:", avg_score)
    return avg_score


def visualize_results(
    pipeline, prompts, num_images_per_prompt=4, output_path="visualization_adaptive.png"
):
    rows = len(prompts)
    cols = num_images_per_prompt
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows == 1:
        axes = [axes]
    for i, prompt in enumerate(prompts):
        generated = pipeline(
            prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            num_images_per_prompt=num_images_per_prompt,
        ).images
        for j, image in enumerate(generated):
            ax = axes[i][j] if cols > 1 else axes[i]
            ax.imshow(image)
            ax.axis("off")
        axes[i][0].set_title(prompt, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"Visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Stable Diffusion with adaptive frequency perturbation (surviving JPEG) using mixed precision."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sd_finetuned_adaptive",
        help="Directory to save the fine-tuned model.",
    )
    parser.add_argument(
        "--num_train_steps", type=int, default=1000, help="Number of training steps."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for training."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-6, help="Learning rate."
    )
    parser.add_argument(
        "--eval_prompts",
        nargs="+",
        default=["a futuristic cityscape", "an impressionist painting"],
        help="Evaluation prompts",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = load_dataset("coco_captions", split="train")
    dataset = dataset.map(preprocess, num_proc=4)
    dataset.set_format(type="torch", columns=["pixel_values", "text"])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model_id = "CompVis/stable-diffusion-v1-4"
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(
        device
    )
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float16
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=torch.float16
    ).to(device)
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    scaler = GradScaler(enabled=(device == "cuda"))
    unet.train()
    global_step = 0

    while global_step < args.num_train_steps:
        for batch in dataloader:
            if global_step >= args.num_train_steps:
                break

            original_image = batch["pixel_values"].to(device)
            # Apply the adaptive frequency attack.
            perturbed_image = torch.stack(
                [adaptive_frequency_attack(img) for img in original_image]
            )
            # Compress using actual JPEG compression.
            compressed_image = torch.stack(
                [actual_jpeg_compression(img, quality=75) for img in perturbed_image]
            )
            training_image = compressed_image.to(device)
            prompts = batch["text"]

            with torch.no_grad():
                latents = (
                    vae.encode(training_image).latent_dist.sample()
                    * vae.config.scaling_factor
                )

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, scheduler.num_train_timesteps, (latents.shape[0],), device=device
            ).long()

            with autocast(enabled=(device == "cuda")):
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                text_inputs = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(device)
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(text_input_ids)[0]
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if global_step % 50 == 0:
                mse_diff = torch.nn.functional.mse_loss(
                    perturbed_image, training_image
                ).item()
                print(
                    f"Step {global_step}: Loss {loss.item():.4f}, JPEG MSE Diff (adaptive) {mse_diff:.6f}"
                )
            global_step += 1

    os.makedirs(args.output_dir, exist_ok=True)
    unet.save_pretrained(os.path.join(args.output_dir, "unet"))
    vae.save_pretrained(os.path.join(args.output_dir, "vae"))
    text_encoder.save_pretrained(os.path.join(args.output_dir, "text_encoder"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))
    scheduler.save_pretrained(os.path.join(args.output_dir, "scheduler"))
    print("Model saved to", args.output_dir)

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.output_dir, torch_dtype=torch.float16
    )
    pipeline.to(device)
    pipeline.enable_attention_slicing()
    print("Evaluating model...")
    evaluate_model(pipeline, device, args.eval_prompts)

    # Visualization: generate a grid of images for the evaluation prompts.
    visualize_results(
        pipeline,
        args.eval_prompts,
        num_images_per_prompt=4,
        output_path="sd_train_adaptive_visualization.png",
    )


if __name__ == "__main__":
    main()
