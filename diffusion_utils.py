from diffusers import DiffusionPipeline
import torch
from transformers import T5EncoderModel
from utils import *
import numpy as np
from image_utils import visualize_all_inpainting_masks
from PIL import Image
from matplotlib import pyplot as plt

IMAGE_SIZE = 1024


def composite_with_mask(
        destination,
        source,
        mask=None,
        resize_source=True,
        resize_mode="bilinear"):
    """
    Composite source onto destination using mask:
    - Where mask=0: keep destination
    - Where mask=1: use source
    """
    device = destination.device
    destination = destination.permute(0, 3, 1, 2)
    source = source.permute(0, 3, 1, 2)
    batch_size, channels, dest_height, dest_width = destination.shape

    result = destination.clone().float()

    source_slice = source.to(device).float()

    if resize_source:
        source_slice = torch.nn.functional.interpolate(
            source_slice, size=(dest_height, dest_width), mode=resize_mode)

    if mask is None:
        mask_slice = torch.ones(
            (batch_size, 1, dest_height, dest_width), device=device)
    else:
        mask_slice = mask.to(device).float()

        if mask_slice.ndim == 3:
            mask_slice = mask_slice.unsqueeze(1)

        if mask_slice.shape[-2:] != (dest_height, dest_width):
            mask_slice = torch.nn.functional.interpolate(
                mask_slice, size=(dest_height, dest_width), mode=resize_mode)

    if mask_slice.max() > 1.0:
        mask_slice = mask_slice / 255.0

    if mask_slice.shape[1] == 1 and source_slice.shape[1] > 1:
        mask_slice = mask_slice.expand(-1, channels, -1, -2)

    print(
        f"Mask min: {mask_slice.min().item()}, max: {mask_slice.max().item()}")

    result = result * (1 - mask_slice) + source_slice * mask_slice

    del source_slice, mask_slice
    torch.cuda.empty_cache()

    return result.to(destination.dtype).permute(0, 2, 3, 1)


def vis_inpaint_strategy(vis=False):
    try:
        initial_pano_pil = Image.open("imgs/initial_pano_with_back.png")
        side_pano = Image.open("imgs/initial_pano_center.png")
        initial_pano_np = np.array(initial_pano_pil)
        side_pano_np = np.array(side_pano)

        # Generate the visualization data
        all_views_data = visualize_all_inpainting_masks(
            initial_panorama=initial_pano_np,
            side_pano=side_pano_np,
            output_size=1024,  # Should match inpainting model input size
        )

        # --- Plot the results ---
        num_views = len(all_views_data)

        fig, axes = plt.subplots(
            num_views, 3, figsize=(
                12, 3 * num_views))  # Width, Height
        fig.suptitle(
            "Individual Inpainting View Masks and Projections",
            fontsize=16)

        if num_views == 1:
            axes = np.array([axes])

        for i, data in enumerate(all_views_data):
            ax_render = axes[i, 0]
            ax_mask = axes[i, 1]
            ax_highlight = axes[i, 2]

            ax_render.imshow(data["render"])
            ax_render.set_title(
                f"{data['label']}\nY={data['yaw']}, P={data['pitch']}, FoV={data['fov']}\nRendered View",
                fontsize=8)
            ax_render.axis("off")

            ax_mask.imshow(data["mask"], cmap="gray")
            ax_mask.set_title("Mask (White=Inpaint)", fontsize=8)
            ax_mask.axis("off")

            # Column 3: Highlighted Footprint on Equirectangular
            ax_highlight.imshow(data["highlighted_pano"])
            ax_highlight.set_title("Mask Footprint on Pano", fontsize=8)
            ax_highlight.axis("off")

        plt.tight_layout(rect=[0, 0.01, 1, 0.98])  # Adjust layout
        plt.show()
        return all_views_data

    except FileNotFoundError:
        print("Error: 'initial_pano_with_back.png' not found.")
        print("Please ensure you have created the initial panorama with front/back duplication first.")
    except Exception as e:
        print(f"An error occurred: {e}")


def fix_inpaint_mask(
        mask,
        contour_color=(
            0,
            255,
            0),
    fill_color=(
            0,
            0,
            0),
        extend_amount=100,
        mode=None,
        side="r"):
    mask_copy = mask.copy()
    if mask_copy.dtype != np.uint8:

        mask_copy = (mask_copy * 255).astype(np.uint8)

    inverted = cv2.bitwise_not(mask_copy)
    contours, _ = cv2.findContours(
        inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(mask_copy, contours, -1, 0, -1)
    if extend_amount > 0:
        kernel = np.ones((extend_amount, extend_amount), np.uint8)
        mask_copy = cv2.dilate(mask_copy, kernel, iterations=1)

    blur_amount = 20
    if blur_amount > 0:
        mask_copy = cv2.GaussianBlur(
            mask_copy, (blur_amount * 2 + 1, blur_amount * 2 + 1), 0)

    if mode is not None:
        if side == "r":
            if mode == "step1":
                mask_copy[:, -150:] = 0
            elif mode == "step2":
                mask_copy = np.zeros_like(mask_copy)
                mask_copy[:, -200:] = 255
            else:
                raise ValueError(f"Invalid mode: {mode}")

        if side == "l":
            if mode == "step1":
                mask_copy[:, :250] = 0
            elif mode == "step2":
                mask_copy = np.zeros_like(mask_copy)
                mask_copy[:, :270] = 1
            else:
                raise ValueError(f"Invalid mode: {mode}")

    return mask_copy


def load_pipeline(four_bit=False):
    from diffusers import FluxTransformer2DModel, FluxFillPipeline

    orig_pipeline = DiffusionPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)

    transformer = FluxTransformer2DModel.from_pretrained(
        "sayakpaul/FLUX.1-Fill-dev-nf4",
        subfolder="transformer",
        torch_dtype=torch.bfloat16)

    text_encoder_2 = T5EncoderModel.from_pretrained(
        "sayakpaul/FLUX.1-Fill-dev-nf4",
        subfolder="text_encoder_2",
        torch_dtype=torch.bfloat16)

    pipeline = FluxFillPipeline.from_pipe(
        orig_pipeline,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        torch_dtype=torch.bfloat16)

    pipeline.enable_model_cpu_offload()
    return pipeline

def load_contolnet_pipeline():
    import sys

    sys.path.append("FLUX-Controlnet-Inpainting/")
    from diffusers.utils import load_image, check_min_version
    from controlnet_flux import FluxControlNetModel
    from transformer_flux import FluxTransformer2DModel
    from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
    from torchao.quantization import quantize_, int8_weight_only

    controlnet = FluxControlNetModel.from_pretrained(
        "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta",
        torch_dtype=torch.bfloat16)
    transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="transformer",
        torch_dtype=torch.bfloat16)

    quantize_(transformer, int8_weight_only())
    quantize_(controlnet, int8_weight_only())
    pipe = FluxControlNetInpaintingPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        controlnet=controlnet,
        transformer=transformer,
        torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.transformer.to(torch.bfloat16)
    pipe.controlnet.to(torch.bfloat16)
    return pipe


def generate_outpaint(
        pipe,
        image,
        mask,
        vis=False,
        use_flux=False,
        num_steps=50,
        prompt="a city town square"):
    if use_flux:
        image = pipe(
            prompt=prompt,
            image=preprocess_image(image),
            mask_image=mask,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            guidance_scale=30,
            num_inference_steps=num_steps,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]
    else:
        image = pipe(
            prompt=prompt,
            image=preprocess_image(image),
            mask_image=mask,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]
    if vis:
        show_image_cv2(pil_to_cv2(image))
    return image


def outpaint_controlnet(
        pipe,
        image,
        mask,
        vis=False,
        num_steps=50,
        prompt="a city town square",
        cond_scale=0.9,
        guidance_scale=3.5):
    generator = torch.Generator(device="cpu").manual_seed(42)
    # Inpaint
    size = (768, 768)
    result = pipe(
        prompt=prompt,
        height=size[1],
        width=size[0],
        control_image=image,
        control_mask=mask,
        num_inference_steps=num_steps,
        generator=generator,
        controlnet_conditioning_scale=cond_scale,
        guidance_scale=guidance_scale,
        negative_prompt="",
        true_guidance_scale=3.5,
    ).images[0]
    if vis:
        print("Inpainting done")
        show_image_cv2(pil_to_cv2(result))

    return result


def fix_mask(
        image,
        source_image,
        fill_shape,
        offset,
        fill_size,
        left,
        pipe,
        vis=False):
    clear_gpu_memory()

    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE + fill_shape), dtype=np.uint8)
    mask_width = 10
    if left:
        mask[:, IMAGE_SIZE +
             offset -
             mask_width: IMAGE_SIZE +
             offset +
             mask_width] = 1
    else:
        mask[:, fill_size - mask_width: mask_width + fill_size] = 1
    blur_amount = 50
    blurred_mask = cv2.GaussianBlur(
        mask * 255, (blur_amount * 2 + 1, blur_amount * 2 + 1), 0) * 255
    blurred_mask = blurred_mask.astype(np.float32)
    image_gen = pipe(
        prompt="",
        image=preprocess_image(cv2_to_pil(image)),
        mask_image=blurred_mask,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE + fill_shape,
        guidance_scale=40,
        num_inference_steps=10,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    return image_gen

