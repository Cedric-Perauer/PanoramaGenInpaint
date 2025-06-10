import torch
from torchvision import transforms

from diffusers import StableDiffusion3InpaintPipeline
from diffusers.utils import load_image


def preprocess_image(image):
    image = image.convert("RGB")
    image = transforms.CenterCrop(
        (image.size[1] // 64 * 64,
         image.size[0] // 64 * 64))(image)
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0).to("cuda")
    return image


def preprocess_mask(mask):
    mask = mask.convert("L")
    mask = transforms.CenterCrop(
        (mask.size[1] // 64 * 64,
         mask.size[0] // 64 * 64))(mask)
    mask = transforms.ToTensor()(mask)
    mask = mask.to("cuda")
    return mask


pipe = StableDiffusion3InpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16,
).to("cuda")

prompt = "A yellow cat with a red collar"

source_image = load_image(
    "https://huggingface.co/alimama-creative/SD3-Controlnet-Inpainting/resolve/main/images/dog.png"
)
mask = load_image(
    "https://huggingface.co/alimama-creative/SD3-Controlnet-Inpainting/resolve/main/images/dog_mask.png")

source = preprocess_image(source_image)
mask = preprocess_mask(mask)

image = pipe(
    prompt=prompt,
    image=source,
    mask_image=mask,
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=7.0,
    strength=0.6,
).images[0]

image.save("overture-creations-5sI6fQgYIuo_output.jpg")
