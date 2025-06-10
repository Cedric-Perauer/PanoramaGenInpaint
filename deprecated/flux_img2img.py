import torch
from diffusers import FluxImg2ImgPipeline
from diffusers.utils import load_image
from utils import pil_to_cv2, show_image_cv2


def run_img2img():
    # Load components
    pipe = FluxImg2ImgPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    # Load noisy input image
    noisy_image = load_image("imgs/render_img_0.png")

    # Refine with parameters
    refined_image = pipe(
        image=noisy_image,
        strength=0.3,
        # Controls noise reduction (0=ignore input, 1=full denoising)
        num_inference_steps=50,  # More steps → higher quality
        guidance_scale=3.5,  # Text prompt relevance (if using)
        prompt="High-quality clean image",  # Optional text guidance
    ).images[0]

    show_image_cv2(pil_to_cv2(refined_image))
    return refined_image


if __name__ == "__main__":
    refined_image = run_img2img()
    refined_image.save("refined_output.png")
