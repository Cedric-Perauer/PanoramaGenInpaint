import torch
from diffusers import DiffusionPipeline
from transformers import T5EncoderModel
import numpy as np
from PIL import Image
import cv2
import os
from torchao.quantization import quantize_, int8_weight_only
from peft import PeftModel


def draw_largest_contour(image_path, output_path=None):
    """
    Draw contours only for the largest white region in an image.

    Args:
        image_path (str): Path to the input image
        output_path (str, optional): Path to save the output image. If None, will show the image.

    Returns:
        numpy.ndarray: Image with contours drawn
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to get binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find all contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    if not contours:
        print("No contours found in the image")
        return img

    largest_contour = max(contours, key=cv2.contourArea)

    # Create a copy of the original image
    result = img.copy()

    # Draw only the largest contour
    cv2.drawContours(result, [largest_contour], -1, (0, 255, 0), 2)

    if output_path:
        cv2.imwrite(output_path, result)
    else:
        cv2.imshow("Largest Contour", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result


def load_flux_lora_pipeline():
    # Load base model
    base_pipeline = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)

    # Apply quantization
    quantize_(base_pipeline.transformer, int8_weight_only())
    quantize_(base_pipeline.vae, int8_weight_only())

    # Load LoRA weights using PEFT
    lora_path = "MultiTrickFox/Flux-LoRA-Equirectangular-v3"
    base_pipeline.unet = PeftModel.from_pretrained(base_pipeline.unet, lora_path, torch_dtype=torch.bfloat16)

    # Enable model offloading for memory efficiency
    base_pipeline.enable_model_cpu_offload()

    return base_pipeline


def generate_360_image(pipeline, prompt, negative_prompt="", num_inference_steps=50, guidance_scale=2.5):
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
        negative_prompt=negative_prompt,
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

    # Convert PIL image to OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Create imgs directory if it doesn't exist
    os.makedirs("imgs", exist_ok=True)

    # Save the original image
    cv2.imwrite("imgs/generated_360.jpg", cv_image)

    # Draw contours and save
    contour_image = draw_largest_contour("imgs/generated_360.jpg", "imgs/generated_360_contour.jpg")

    print("Images saved to imgs/ folder:")
    print("- generated_360.jpg (original)")
    print("- generated_360_contour.jpg (with contour)")
