import torch
from diffusers import FluxPipeline
from diffusers.utils import load_image
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from utils import pil_to_cv2, cv2_to_pil, show_image_cv2, clear_gpu_memory, run_with_conda_env
import copy
from diffusion_utils import (
    load_pipeline,
    generate_outpaint,
    fix_mask,
    fix_inpaint_mask,
    vis_inpaint_strategy,
    load_contolnet_pipeline,
    outpaint_controlnet,
)
import torch
from dust3r_infer import get_focals
from image_utils import *
import copy
from tqdm import tqdm

GEN = False
USE_SDXL = False
REFINER = False
COMPOSITE = False
TOP_BOTTOM_VIEWS = False
IMAGE_SIZE = 1024

if GEN:
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    prompt = "a town square in a city with a person standing in the center"
    image = pipe(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    image.save("middle.jpg")
    clear_gpu_memory()
    focals = get_focals("middle.jpg", 512)
    source_image_cv2 = pil_to_cv2(image)
    show_image_cv2(source_image_cv2)

all_views_data = vis_inpaint_strategy()


top_and_bottom_views = [all_views_data[0], all_views_data[3]]
side_views = all_views_data[6:]

right_side_nums = []
left_side_nums = []
clear_gpu_memory()

pipeline = load_contolnet_pipeline()

inital_pano = Image.open("imgs/initial_pano_with_back.png")
initial_pano_np = np.array(inital_pano)


side_view_pano = Image.open("imgs/initial_pano_center.png")
side_view_pano_np = np.array(side_view_pano)


if TOP_BOTTOM_VIEWS:
    for view in top_and_bottom_views:
        mask = view["mask"]
        mask[mask == 255] = 1
        new_mask = fix_inpaint_mask(mask * 255)
        inital_pano = Image.open("initial_pano_with_back.png")
        initial_pano_np = np.array(inital_pano)
        render_img = cv2.cvtColor(view["render"], cv2.COLOR_BGR2RGB)
        render_img = cv2_to_pil(render_img)

        print(f"Render image shape: {render_img.size}{mask.shape}")
        image = generate_outpaint(pipeline, render_img, new_mask, vis=True, prompt="floor of a city town square")
        final_pano = project_perspective_to_equirect(
            cv2.cvtColor(pil_to_cv2(image), cv2.COLOR_BGR2RGB),
            initial_pano_np,
            yaw_deg=-view["yaw"],
            pitch_deg=view["pitch"],
            h_fov_deg=view["fov"],
            v_fov_deg=view["fov"],
            mask=mask,
        )


for idx, view in enumerate(tqdm(side_views, desc="Processing side views")):
    show_image_cv2(cv2.cvtColor(initial_pano_np, cv2.COLOR_BGR2RGB))
    render_img = render_perspective(
        side_view_pano_np, view["yaw"], -view["pitch"], view["fov"], view["vfov"], IMAGE_SIZE
    )
    show_image_cv2(cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB))

    mask = create_mask_from_black(render_img, threshold=10)
    if idx == 0:
        new_mask = fix_inpaint_mask(mask, extend_amount=100)
    else:
        new_mask = fix_inpaint_mask(mask, extend_amount=20)

    save_mask = Image.fromarray(new_mask).convert("L")
    save_mask.save(f"new_mask_{idx}.png")
    render_img = Image.fromarray(render_img).convert("RGB")
    show_image_cv2(pil_to_cv2(new_mask))
    prompt = "a new photorealistic house on a market square in 1800s stylex, without people"

    print(f"Render image shape: {render_img.size}{mask.shape}")
    render_img.save(f"imgs/render_{idx}.png")

    # if idx == 0:
    #    cond_scale = 0.4
    # else:
    cond_scale = 0.9

    if idx not in right_side_nums and idx not in left_side_nums:
        image = outpaint_controlnet(
            pipeline,
            render_img,
            new_mask,
            vis=True,
            prompt=prompt,
            num_steps=50,
            guidance_scale=3.5,
            cond_scale=cond_scale,
        )
    else:
        if idx in right_side_nums:
            print(f"Right side {idx}")
            cur_mask = fix_inpaint_mask(new_mask, mode="step1", side="r")
            render_img.save(f"imgs/input_1_{idx}.png")
            image = outpaint_controlnet(
                pipeline,
                render_img,
                copy.deepcopy(cur_mask),
                vis=True,
                prompt=prompt,
                num_steps=50,
                guidance_scale=3.5,
                cond_scale=cond_scale,
            )
            Image.fromarray(cur_mask).convert("RGB").save(f"imgs/cur_mask_{idx}_1.png")
            image.save(f"imgs/fix_render_{idx}_1.png")

            cur_mask = fix_inpaint_mask(new_mask, mode="step2", side="r")
            image.save(f"imgs/input_2_{idx}.png")
            image = outpaint_controlnet(
                pipeline,
                image,
                copy.deepcopy(cur_mask),
                vis=True,
                prompt=prompt,
                num_steps=50,
                guidance_scale=3.5,
                cond_scale=cond_scale,
            )
            Image.fromarray(cur_mask).convert("RGB").save(f"imgs/cur_mask_{idx}_2.png")
            image.save(f"imgs/fix_render_{idx}_2.png")

        else:
            new_mask = fix_inpaint_mask(new_mask, mode="step1", side="l")

    image = image.resize((1024, 1024), Image.LANCZOS)
    image.save(f"imgs/render_in_{idx}.png")
    if COMPOSITE:
        # Composite the outpainted image with the rendered image using the mask
        mask_array = np.array(new_mask)
        mask_array = mask_array.astype(np.float32) / 255.0  # Normalize to 0-1

        # Convert images to numpy arrays
        outpainted_np = np.array(image)
        rendered_np = np.array(render_img)

        # Resize outpainted image to match mask dimensions
        outpainted_np = cv2.resize(outpainted_np, (mask_array.shape[1], mask_array.shape[0]))
        rendered_np = cv2.resize(rendered_np, (mask_array.shape[1], mask_array.shape[0]))
        mask_array = np.expand_dims(mask_array, axis=2)  # Add channel dimension

        # Composite using the mask
        composite = outpainted_np * mask_array + rendered_np * (1 - mask_array)
        image = Image.fromarray(composite.astype(np.uint8))

        image = image.resize((1024, 1024), Image.LANCZOS)
        image.save(f"imgs/render_in_{idx}.png")

    clear_gpu_memory()
    # if idx == 0 :
    if REFINER or idx == 0:
        run_with_conda_env(
            "diffusers33", f"refiner.py imgs/render_in_{idx}.png --output_path imgs/refined_output_{idx}.png"
        )
        image = Image.open(f"imgs/refined_output_{idx}.png")
    # else :
    #    image = Image.open(f"imgs/render_{idx}.png")

    new_mask = np.array(new_mask)
    cur_mask = new_mask
    if idx == 0:
        cur_mask = None
    initial_pano_np = project_perspective_to_equirect(
        cv2.cvtColor(pil_to_cv2(image), cv2.COLOR_BGR2RGB),
        initial_pano_np,
        yaw_deg=view["yaw"],
        pitch_deg=view["pitch"],
        h_fov_deg=view["fov"],
        v_fov_deg=view["fov"],
        mask=cur_mask,  # Original mask - only project where we outpainted,
        mirror=False,
    )

    side_view_pano_np = project_perspective_to_equirect(
        cv2.cvtColor(pil_to_cv2(image), cv2.COLOR_BGR2RGB),
        side_view_pano_np,
        yaw_deg=view["yaw"],
        pitch_deg=view["pitch"],
        h_fov_deg=view["fov"],
        v_fov_deg=view["fov"],
        mask=cur_mask,  # Original mask - only project where we outpainted,
        mirror=False,
    )

    cur_pano = cv2.cvtColor(initial_pano_np, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"imgs/cur_pano_{idx}.png", cur_pano)
