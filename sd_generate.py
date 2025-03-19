import torch
import argparse
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    StableDiffusionPipeline,
)
from transformers import CLIPTokenizer, CLIPTextModel
import matplotlib.pyplot as plt


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
        default="./sd_train",
        help="Directory to save the fine-tuned model.",
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
        default=["a female celebrity with blonde hair"],
        help="Prompts for evaluation and visualization.",
    )
    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.mps.is_available() else "cpu"
    )

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

    # Pre-visualization step: generate and display a grid of images.
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
    )
    visualize_results(
        pipeline,
        args.prompt,
        num_images_per_prompt=4,
        output_path=args.output_dir + "/pre_visualization.png",
    )


if __name__ == "__main__":
    main()
