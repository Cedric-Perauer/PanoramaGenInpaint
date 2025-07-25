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
from image_utils import create_mask_from_black, project_perspective_to_equirect, render_perspective, image_to_equirectangular, focal_to_fov
import copy
from tqdm import tqdm

GEN =  True
USE_SDXL = False
REFINER = False
COMPOSITE = True
TOP_BOTTOM_VIEWS = True
GEN_TOP_BOTTOM = True
IMAGE_SIZE = 1024
SIDE_VIEWS = True
cond_scale = 0.9

LAPLACIAN_BLENDING = False
BLUR_BLENDING = False

scene_prompt = "a modern japanese garden with a pond and a waterfall"
scene_prompt_sides = "a modern japanese garden "

if GEN:
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    image = pipe(
        scene_prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    image.save("imgs/middle.jpg")
    clear_gpu_memory()
    focals = get_focals("imgs/middle.jpg", 512)
    h_fov = focal_to_fov(focals[0], 512)
    source_image_cv2 = pil_to_cv2(image)
    equirectangular_image,non_wrapped = image_to_equirectangular(source_image_cv2, h_fov, 2028*4,1024*4,wrap_around=True)
    cv2_to_pil(equirectangular_image).save("imgs/initial_pano_with_back.png")
    cv2_to_pil(non_wrapped).save("imgs/initial_pano_center.png")                              

all_views_data = vis_inpaint_strategy()

top_and_bottom_views = all_views_data[:4]
side_views = all_views_data[4:]


right_side_nums = []
left_side_nums = []
clear_gpu_memory()

pipeline = load_contolnet_pipeline()
if GEN:
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
        prompt=scene_prompt,
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


# top_and_bottom_views =

print("Processing top and bottom views")
cur_pano = Image.open("imgs/cur_pano_initial.png")
if TOP_BOTTOM_VIEWS and GEN_TOP_BOTTOM:
        for idx, view in tqdm(enumerate(top_and_bottom_views), desc="Processing top and bottom views"):
            if "Bottom" not in view["label"]:
                prompt = f"floor of {scene_prompt}"
            else:
                prompt = "A Blue sky"

            render_img = render_perspective(
                initial_pano_np, view["yaw"], -view["pitch"], view["fov"], view["vfov"], IMAGE_SIZE
            )

            mask = create_mask_from_black(render_img, threshold=10)
            new_mask = mask
            if idx == 2:
                new_mask = fix_mask_region(mask, extension=100)
            else:
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

            if REFINER:
                run_with_conda_env(
                    "diffusers33",
                    f"refiner.py imgs/render_top_bottom_out_{idx}.png --output_path imgs/refined_top_bottom_{idx}.png",
                )
                image = Image.open(f"imgs/refined_top_bottom_{idx}.png")

            if idx == 2:
                new_mask = None

            inital_pano_np = project_perspective_to_equirect(
                cv2.cvtColor(pil_to_cv2(image), cv2.COLOR_BGR2RGB),
                initial_pano_np,
                yaw_deg=-view["yaw"],
                pitch_deg=view["pitch"],
                h_fov_deg=view["fov"],
                v_fov_deg=view["fov"],
                mask=new_mask,
                blur_blending=BLUR_BLENDING,
            )
            inital_pano = Image.fromarray(initial_pano_np).convert("RGB")
            inital_pano.save(f"imgs/top_bottom_pano_{idx}.png")

idx = len(top_and_bottom_views) - 1

if TOP_BOTTOM_VIEWS:
    print("Processing side views from top bottom load")
    side_view_pano = Image.open(f"imgs/top_bottom_pano_{idx}.png")

else:
    side_view_pano = Image.open("imgs/cur_pano_initial.png")

side_view_pano_np = np.array(side_view_pano)
side_view_middle_only = Image.open("imgs/initial_pano_center.png")
# side_view_middle_only = Image.open("imgs/cur_pano_5.png")
side_view_middle_only_np = np.array(side_view_middle_only)

if SIDE_VIEWS:
    print("Processing side views")
    for idx, view in enumerate(tqdm(side_views, desc="Processing side views")):
        if TOP_BOTTOM_VIEWS:
            mask_img = render_perspective(
                side_view_middle_only_np, view["yaw"], -view["pitch"], view["fov"], view["vfov"], IMAGE_SIZE
            )
            
            mask = create_mask_from_black(mask_img, threshold=10)
            new_mask = mask
            if idx == 0:
                new_mask[:180, :] = 0
                new_mask[-100:, :] = 0
                
                
            else:
                new_mask[:135, :] = 0
                new_mask[-130:, :] = 0
    
        render_img = render_perspective(
            side_view_pano_np, view["yaw"], -view["pitch"], view["fov"], view["vfov"], IMAGE_SIZE
        )
             
        if TOP_BOTTOM_VIEWS == False:
            mask = create_mask_from_black(render_img, threshold=10)
            new_mask = fix_inpaint_mask(mask, extend_amount=20)

        extension = 20
        if idx == 7:
            extension = 100

        if idx != 0:
            new_mask = fix_mask_region(new_mask, extension=extension)
        save_mask = Image.fromarray(new_mask).convert("L")
        save_mask.save(f"imgs/new_mask_{idx}.png")
        render_img = Image.fromarray(render_img).convert("RGB")

        
        print(f"Render image shape: {render_img.size}{mask.shape}")
        render_img.save(f"imgs/render_{idx}.png")

        cond_scale = 0.9

        image = outpaint_controlnet(
            pipeline,
            render_img,
            new_mask,
            vis=True,
            prompt=scene_prompt_sides,
            num_steps=50,
            guidance_scale=3.5,
            cond_scale=cond_scale,
        )

        image = image.resize((1024, 1024), Image.LANCZOS)
        image.save(f"imgs/render_in_{idx}.png")
        if COMPOSITE and idx != 0:
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

            # Apply Gaussian blur to the mask array
            mask_array = cv2.blur(mask_array, (20, 20))

            # Ensure mask is in the right format for broadcasting
            maskf = cv2.merge([mask_array, mask_array, mask_array])

            composite = maskf * outpainted_np + (1 - maskf) * rendered_np
            composite = composite.clip(0, 255).astype(np.uint8)

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
        side_view_pano_np = project_perspective_to_equirect(
            cv2.cvtColor(pil_to_cv2(image), cv2.COLOR_BGR2RGB),
            side_view_pano_np,
            yaw_deg=view["yaw"],
            pitch_deg=view["pitch"],
            h_fov_deg=view["fov"],
            v_fov_deg=view["fov"],
            mask=cur_mask,  # Original mask - only project where we outpainted,
            blur_blending=BLUR_BLENDING,
            laplacian_blending=LAPLACIAN_BLENDING,
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
            blur_blending=BLUR_BLENDING,
            laplacian_blending=LAPLACIAN_BLENDING,
            mirror=False,
        )

        cur_pano = cv2.cvtColor(side_view_pano_np, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"imgs/cur_pano_{idx}.png", cur_pano)

