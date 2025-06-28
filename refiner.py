#!/usr/bin/env python3
import torch
import os
import argparse
from diffusers import FluxImg2ImgPipeline
from diffusers.utils import load_image
import gc
from torchao.quantization import quantize_, int8_weight_only


def clear_gpu_memory():
    """Clear CUDA memory to free up GPU resources"""
    # Delete variables that might hold tensors
    try:
        del pipe
    except:
        pass

    # Empty CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Run garbage collection
    gc.collect()

    print("GPU memory cleared")


def refine_image(image_path, output_path="refined_output.png"):
    clear_gpu_memory()
    # Load components
    pipe = FluxImg2ImgPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    quantize_(pipe.transformer, int8_weight_only())
    quantize_(pipe.vae, int8_weight_only())
    pipe.enable_model_cpu_offload()

    # Load input image
    noisy_image = load_image(image_path)

    # Refine with parameters
    refined_image = pipe(
        image=noisy_image,
        # Controls noise reduction (0=ignore input, 1=full denoising)
        strength=0.4,
        num_inference_steps=100,  # More steps → higher quality
        guidance_scale=3.5,  # Text prompt relevance (if using)
        prompt="",
    ).images[0]

    refined_image.save(output_path)
    print(f"Refined image saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Refine an image using FLUX model")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--output_path", type=str, default="refined_output.png", help="Path to save the refined image")

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.image_path):
        print(f"Error: Input file '{args.image_path}' does not exist")
        return

    try:
        refine_image(args.image_path, args.output_path)
    except Exception as e:
        print(f"Error during image refinement: {e}")


if __name__ == "__main__":
    main()
