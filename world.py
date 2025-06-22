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
    fix_mask_region,
    vis_inpaint_strategy,
    load_contolnet_pipeline,
    outpaint_controlnet,
)
import torch
from dust3r_infer import get_focals
from image_utils import create_mask_from_black, project_perspective_to_equirect, render_perspective
import copy
from tqdm import tqdm

GEN = False
USE_SDXL = False
REFINER = False
COMPOSITE = False
TOP_BOTTOM_VIEWS = True
IMAGE_SIZE = 1024
SIDE_VIEWS = True
cond_scale = 0.9
TOP_BOTTOM_FIRST = True
GEN_FIRST = False

GEN_TOP_BOTTOM = False


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


top_and_bottom_views = all_views_data[:4]
side_views = all_views_data[4:]


right_side_nums = []
left_side_nums = []
clear_gpu_memory()

pipeline = load_contolnet_pipeline()

if GEN_FIRST:
    print("Generating first image")
    inital_pano = Image.open("imgs/initial_pano_center.png")
    initial_pano_np = np.array(inital_pano)

    idx = 0
    view = side_views[0]
    render_img = render_perspective(initial_pano_np, view["yaw"], -view["pitch"], view["fov"], view["vfov"], IMAGE_SIZE)

    mask = create_mask_from_black(render_img, threshold=10)
    new_mask = fix_inpaint_mask(mask, extend_amount=100)

    save_mask = Image.fromarray(new_mask).convert("L")
    save_mask.save(f"new_mask_{idx}.png")
    render_img = Image.fromarray(render_img).convert("RGB")

    prompt = "a photorealistic house on a market square in 1800s style, the sky region is blue with clouds"

    print(f"Render image shape: {render_img.size}{mask.shape}")
    render_img.save(f"imgs/render_{idx}.png")

    # if idx == 0:
    #    cond_scale = 0.4
    # else:
    cond_scale = 0.9

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

    image = image.resize((1024, 1024), Image.LANCZOS)
    image.save(f"imgs/render_in_{idx}.png")
    clear_gpu_memory()

    run_with_conda_env(
        "diffusers33", f"refiner.py imgs/render_in_{idx}.png --output_path imgs/refined_output_{idx}.png"
    )
    image = Image.open(f"imgs/refined_output_{idx}.png")

    inital_pano_np = project_perspective_to_equirect(
        cv2.cvtColor(pil_to_cv2(image), cv2.COLOR_BGR2RGB),
        initial_pano_np,
        yaw_deg=view["yaw"],
        pitch_deg=view["pitch"],
        h_fov_deg=view["fov"],
        v_fov_deg=view["fov"],
        mask=None,  # Original mask - only project where we outpainted,
        mirror=False,
    )

    cur_pano = cv2.cvtColor(inital_pano_np, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"imgs/cur_pano_initial.png", cur_pano)
else:
    initial_pano_np = np.array(Image.open("imgs/cur_pano_initial.png"))

if TOP_BOTTOM_FIRST and GEN_TOP_BOTTOM:
    print("Processing top and bottom views")
    cur_pano = Image.open("imgs/cur_pano_initial.png")
    if TOP_BOTTOM_VIEWS:
        for idx, view in tqdm(enumerate(top_and_bottom_views), desc="Processing top and bottom views"):
            if idx <= 1:
                prompt = "floor of a city town square"
            else:
                prompt = "A Blue sky"

            render_img = render_perspective(
                initial_pano_np, view["yaw"], -view["pitch"], view["fov"], view["vfov"], IMAGE_SIZE
            )
            # show_image_cv2(cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB))

            mask = create_mask_from_black(render_img, threshold=10)
            new_mask = mask
            new_mask = fix_inpaint_mask(mask, extend_amount=20)
            cv2_to_pil(new_mask).save(f"imgs/new_mask_{idx}.png")
            render_img = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)
            render_img = cv2_to_pil(render_img)
            print(f"Render image shape: {render_img.size}{mask.shape}")
            render_img.save(f"imgs/render_in_top_bottom_{idx}.png")
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
            image = image.resize((1024, 1024), Image.LANCZOS)
            image.save(f"imgs/render_top_bottom_out_{idx}.png")
            """
            if REFINER or idx == 0:
                run_with_conda_env(
                    "diffusers33",
                    f"refiner.py imgs/render_top_bottom_out_{idx}.png --output_path imgs/refined_top_bottom_{idx}.png",
                )
                image = Image.open(f"imgs/refined_top_bottom_{idx}.png")
            """

            inital_pano_np = project_perspective_to_equirect(
                cv2.cvtColor(pil_to_cv2(image), cv2.COLOR_BGR2RGB),
                initial_pano_np,
                yaw_deg=-view["yaw"],
                pitch_deg=view["pitch"],
                h_fov_deg=view["fov"],
                v_fov_deg=view["fov"],
                mask=new_mask,
            )
            inital_pano = Image.fromarray(initial_pano_np).convert("RGB")
            inital_pano.save(f"imgs/top_bottom_pano_{idx}.png")

idx = len(top_and_bottom_views) - 1

if TOP_BOTTOM_FIRST:
    print("Processing side views from top bottom load")
    side_view_pano = Image.open(f"imgs/top_bottom_pano_{idx}.png")

else:
    side_view_pano = Image.open("imgs/cur_pano_initial.png")

side_view_pano_np = np.array(side_view_pano)
side_view_middle_only = Image.open("imgs/cur_pano_initial.png")
# side_view_middle_only = Image.open("imgs/cur_pano_5.png")
side_view_middle_only_np = np.array(side_view_middle_only)


if SIDE_VIEWS:
    print("Processing side views")
    for idx, view in enumerate(tqdm(side_views[1:], desc="Processing side views")):
        if TOP_BOTTOM_VIEWS:
            mask_img = render_perspective(
                side_view_middle_only_np, view["yaw"], -view["pitch"], view["fov"], view["vfov"], IMAGE_SIZE
            )
            mask = create_mask_from_black(mask_img, threshold=10)
            new_mask = mask
            new_mask[:80, :] = 0
            new_mask[-80:, :] = 0

        render_img = render_perspective(
            side_view_pano_np, view["yaw"], -view["pitch"], view["fov"], view["vfov"], IMAGE_SIZE
        )
        if TOP_BOTTOM_VIEWS == False:
            mask = create_mask_from_black(render_img, threshold=10)
            new_mask = fix_inpaint_mask(mask, extend_amount=20)

        extension = 20
        if idx == 6:
            extension = 100

        new_mask = fix_mask_region(mask, extension=extension)
        save_mask = Image.fromarray(new_mask).convert("L")
        save_mask.save(f"imgs/new_mask_{idx}.png")
        render_img = Image.fromarray(render_img).convert("RGB")

        prompt = "a new photorealistic house on a market square in 1800s stylex, without people"
        print(f"Render image shape: {render_img.size}{mask.shape}")
        render_img.save(f"imgs/render_{idx}.png")

        cond_scale = 0.9

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

        image = image.resize((1024, 1024), Image.LANCZOS)
        image.save(f"imgs/render_in_{idx}.png")
        if COMPOSITE:
            # Composite the outpainted image with the rendered image using the
            # mask
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
        # if REFINER or idx == 0:
        #    run_with_conda_env(
        #        "diffusers33", f"refiner.py imgs/render_in_{idx}.png --output_path imgs/refined_output_{idx}.png"
        #    )
        #    image = Image.open(f"imgs/refined_output_{idx}.png")
        # else :
        #    image = Image.open(f"imgs/render_{idx}.png")

        new_mask = np.array(new_mask)
        cur_mask = new_mask
        # if idx == 0:
        #    cur_mask = None
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

        side_view_middle_only_np = project_perspective_to_equirect(
            cv2.cvtColor(pil_to_cv2(image), cv2.COLOR_BGR2RGB),
            side_view_middle_only_np,
            yaw_deg=view["yaw"],
            pitch_deg=view["pitch"],
            h_fov_deg=view["fov"],
            v_fov_deg=view["fov"],
            mask=cur_mask,  # Original mask - only project where we outpainted,
            mirror=False,
        )

        cur_pano = cv2.cvtColor(side_view_pano_np, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"imgs/cur_pano_{idx}.png", cur_pano)


if TOP_BOTTOM_VIEWS:
    if not TOP_BOTTOM_FIRST:
        cur_pano = Image.open("imgs/cur_pano_6.png")
        initial_pano_np = np.array(cur_pano)
        top_bottom_views = [all_views_data[1], all_views_data[4]]

        for cidx, view in tqdm(enumerate(top_bottom_views[:2]), desc="Processing top and bottom views"):
            idx = cidx + len(side_views) + 3
            print(f"Processing top and bottom views {idx}")
            if idx == 0:
                prompt = "floor of a city town square"
            else:
                prompt = "A Blue sky region"

            render_img = render_perspective(
                initial_pano_np, view["yaw"], -view["pitch"], view["fov"], view["vfov"], IMAGE_SIZE
            )

            render_img_pil = cv2_to_pil(render_img)
            render_img_pil.save(f"imgs/render_in_top_bottom_mask{idx}.png")
            mask = create_mask_from_black(render_img, threshold=10)
            cv2_to_pil(mask).save(f"imgs/new_mask_pre_{idx}.png")
            # new_mask = fix_inpaint_mask(mask, extend_amount=20)
            extend_amount = 20
            kernel = np.ones((extend_amount, extend_amount), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

            new_mask = mask
            cv2_to_pil(new_mask).save(f"imgs/new_mask_{idx}.png")
            render_img = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)
            # render_img = cv2.cvtColor(view["render"], cv2.COLOR_BGR2RGB)
            render_img = cv2_to_pil(render_img)
            print(f"Render image shape: {render_img.size}{mask.shape}")
            render_img.save(f"imgs/render_in_top_bottom_{idx}.png")
            cond_scale = 0.7
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
            image = image.resize((1024, 1024), Image.LANCZOS)
            image.save(f"imgs/render_top_bottom_out_{idx}.png")
            yaw_deg = view["yaw"]

            # if cidx == 0:
            #    yaw_deg = -view["yaw"]
            # else:
            #    yaw_deg = view["yaw"]

            inital_pano_np = project_perspective_to_equirect(
                cv2.cvtColor(pil_to_cv2(image), cv2.COLOR_BGR2RGB),
                initial_pano_np,
                yaw_deg=yaw_deg,
                pitch_deg=view["pitch"],
                h_fov_deg=view["fov"],
                v_fov_deg=view["fov"],
                mask=new_mask,
            )
            inital_pano = Image.fromarray(initial_pano_np).convert("RGB")
            inital_pano.save(f"imgs/top_bottom_pano_{idx}.png")
