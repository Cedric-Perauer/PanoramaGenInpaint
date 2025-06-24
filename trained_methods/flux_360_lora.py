import torch
from diffusers import DiffusionPipeline
from transformers import T5EncoderModel
import numpy as np
from PIL import Image
from torchao.quantization import quantize_, int8_weight_only


def load_flux_lora_pipeline():
    # Load base model
    base_pipeline = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)

    # quantize_(base_pipeline.transformer, int8_weight_only())
    # quantize_(base_pipeline.vae, int8_weight_only())

    # Load LoRA weights
    lora_path = "MultiTrickFox/Flux-LoRA-Equirectangular-v3"
    base_pipeline.load_lora_weights(lora_path)

    # Enable model offloading for memory efficiency
    base_pipeline.enable_model_cpu_offload()

    return base_pipeline


def generate_360_image(pipeline, prompt, negative_prompt="", num_inference_steps=50, guidance_scale=7.5):
    """
    Generate a 360-degree equirectangular image using the Flux-LoRA model.

    Args:
        pipeline: The loaded Flux pipeline
        prompt (str): Text prompt for generation
        negative_prompt (str): Negative prompt for generation
        num_inference_steps (int): Number of denoising steps
        guidance_scale (float): Guidance scale for generation

    Returns:
        PIL.Image: Generated 360-degree image
    """
    # Generate the image
    image = pipeline(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=1024,  # Standard equirectangular width
        height=512,  # Standard equirectangular height
        generator=torch.Generator("cpu").manual_seed(42),
    ).images[0]

    return image


if __name__ == "__main__":
    # Example usage
    pipeline = load_flux_lora_pipeline()

    # Generate a 360-degree image
    prompt = "a beautiful panoramic view of a mountain landscape at sunset, equirectangular projection"
    negative_prompt = "distorted, blurry, low quality, bad anatomy"

    image = generate_360_image(
        pipeline, prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=50, guidance_scale=2.5
    )

    # Save the generated image
    image.save("generated_360.jpg")
