# This script applies a basic (pixel‑space) perturbation to each training image,
# then compresses it as an actual JPEG using PIL’s encoder (via an in‑memory file) before proceeding with training.
# The script logs the MSE difference between the perturbed image and its JPEG‑compressed version.
# sd_train_perturb.py
import argparse
import os
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
from torch.amp import autocast, GradScaler
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
    example["text"] = example["name"]  # Using butterfly name as caption
    return example


def apply_basic_perturbation(image, epsilon=0.01):
    noise = torch.randn_like(image) * epsilon
    return torch.clamp(image + noise, 0, 1)


def perform_jpeg_compression(image_tensor, quality=75):
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
    pipeline, prompts, num_images_per_prompt=4, output_path="visualization.png"
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
        description="Fine-tune Stable Diffusion on Smithsonian Butterflies dataset."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sd_train_perturb",
        help="Directory to save the fine-tuned model.",
    )
    parser.add_argument(
        "--visual_output_dir",
        type=str,
        default="./sd_train_perturb_visual",
        help="Directory to save the visualisation results.",
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
        "--eval", type=bool, default=False, help="Evaluate the model after training."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory to save the dataset.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./model",
        help="Directory to save the models.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="+",
        default=["a butterfly with blue wings"],
        help="Prompts for evaluation and visualization.",
    )
    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.mps.is_available() else "cpu"
    )

    dataset = load_dataset(
        "huggan/smithsonian_butterflies_subset", split="train", cache_dir=args.data_dir
    )
    dataset = dataset.map(preprocess)
    dataset.set_format(type="torch", columns=["pixel_values", "text"])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model_id = "CompVis/stable-diffusion-v1-4"
    tokenizer = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-large-patch14", cache_dir=args.model_dir
    )
    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-large-patch14",
        cache_dir=args.model_dir,
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", cache_dir=args.model_dir
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", cache_dir=args.model_dir
    ).to(device)
    scheduler = DDPMScheduler.from_pretrained(
        model_id, subfolder="scheduler", cache_dir=args.model_dir
    )

    # Training loop
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    scaler = GradScaler(enabled=(device == "cuda"))
    unet.train()
    global_step = 0

    while global_step < args.num_train_steps:
        for batch in dataloader:
            if global_step >= args.num_train_steps:
                break

            images = batch["pixel_values"].to(device)
            images = apply_basic_perturbation(images)
            perturbed_image = images.clone()
            images = torch.stack(
                [perform_jpeg_compression(img, quality=75) for img in images]
            ).to(device)
            prompts = batch["text"]

            with torch.no_grad():
                latents = (
                    vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
                )

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
            ).long()

            with autocast(device, enabled=(device == "cuda")):
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
                mse_diff = torch.nn.functional.mse_loss(perturbed_image, images).item()
                print(
                    f"Step {global_step}: Loss {loss.item():.4f}, JPEG MSE Diff {mse_diff:.6f}"
                )
            global_step += 1

    os.makedirs(args.output_dir, exist_ok=True)
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
    )
    pipeline.save_pretrained(args.output_dir)
    print("Model saved to", args.output_dir)

    # Evaluation: load a diffusion pipeline from the fine-tuned model.
    pipeline = StableDiffusionPipeline.from_pretrained(args.output_dir)
    pipeline.to(device)
    pipeline.enable_attention_slicing()
    print("Evaluating model...")
    evaluate_model(pipeline, device, args.prompt)

    # Visualization step: generate and display a grid of images.
    visualize_results(
        pipeline,
        args.prompt,
        num_images_per_prompt=4,
        output_path=args.visual_output_dir + "/visualization.png",
    )


if __name__ == "__main__":
    main()
