import os
import os.path as osp

from copy import deepcopy
from typing import Union, Dict, List
from types import SimpleNamespace
from pathlib import Path
from tqdm import tqdm

from PIL import Image, ImageFilter
import cv2
import numpy as np

import torch
import torch.utils.data
import torch.utils
import torch.nn as nn
from torchvision import transforms as TF
from diffusers import DDIMScheduler, AutoencoderKL

def mask_dilate(mask, kernel_size):
        if type(mask) != np.ndarray:
            mask = np.array(mask)

        if kernel_size!=0:
            mask = cv2.dilate(mask,
                            np.ones((kernel_size, kernel_size), np.uint8),
                            iterations=1)
        return mask

def mask_morphologyEx(mask, kernel_size):
        if type(mask) != np.ndarray:
            mask = np.array(mask)

        if kernel_size!=0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((kernel_size, kernel_size), np.uint8),
                            iterations=1)
        return mask


def get_timesteps(scheduler, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(
            int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = scheduler.timesteps[t_start * scheduler.order:]

        return timesteps, num_inference_steps - t_start

def predict_noise(diff_model, noisy_latents, resized_masks, masked_latents, timesteps, input_ids, guidance_scale=1.0):
    CFG_GUIDANCE = guidance_scale != 1 

    if CFG_GUIDANCE:
        noisy_latents = torch.cat([noisy_latents] * 2)
        resized_masks = torch.cat([resized_masks] * 2)
        masked_latents = torch.cat([masked_latents] * 2)

        assert input_ids.shape[0] % 2 == 0

    latent_model_input = torch.cat([
        noisy_latents, resized_masks, masked_latents], dim=1)

    # Predict the noise residual
    noise_pred = diff_model(
        latent_model_input,
        timesteps=timesteps,
        input_ids=input_ids
    ).sample

    if CFG_GUIDANCE:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)

    return noise_pred  

def get_masked_stat(img:np.ndarray, mask=None, per_channel = True):
    ax = (0,1) if per_channel else None
    ret = dict(mean=img.mean(ax), std=img.std(ax))

    if mask is not None:
        mask = np.stack([mask]*3, axis=-1)
        unmask = 1 - mask
        neighbor_unmask = np.stack(
            [get_bbox_unmask(mask[...,0])]*3, axis=-1)

        masked_pixels = mask.sum(ax)
        masked_mean = (img*mask).sum(ax)/masked_pixels
        # breakpoint()
        masked_squared_diffs = ((img - masked_mean) ** 2) * mask
        masked_var = masked_squared_diffs.sum(ax) / masked_pixels
        masked_std = np.sqrt(masked_var)

        unmasked_pixels = unmask.sum(ax)
        unmasked_mean = (img*unmask).sum(ax)/unmasked_pixels
        unmasked_squared_diffs = ((img - unmasked_mean) ** 2) * unmask
        unmasked_var = unmasked_squared_diffs.sum(ax) / unmasked_pixels
        unmasked_std = np.sqrt(unmasked_var)

        nbr_unmasked_pixels = neighbor_unmask.sum(ax)
        nbr_unmasked_mean = (img*neighbor_unmask).sum(ax)/nbr_unmasked_pixels
        nbr_unmasked_squared_diffs = ((img - nbr_unmasked_mean) ** 2) * neighbor_unmask
        nbr_unmasked_var = nbr_unmasked_squared_diffs.sum(ax) / nbr_unmasked_pixels
        nbr_unmasked_std = np.sqrt(nbr_unmasked_var)


        # ret['count']=masked_pixels
        ret['masked_mean']=masked_mean
        ret['masked_std']=masked_std
        ret['masked_var']=masked_var
        # ret['masked_sq_err']=masked_squared_diffs
        
        ret['unmasked_mean']=unmasked_mean
        ret['unmasked_std']=unmasked_std
        ret['unmasked_var']=unmasked_var

        ret['nbr_unmasked_mean']=nbr_unmasked_mean
        ret['nbr_unmasked_std']=nbr_unmasked_std
        ret['nbr_unmasked_var']=nbr_unmasked_var

    return ret

def get_unmasked_stat(img:np.ndarray, mask=None, per_channel = True):
    ''' a minimal ver. of get_masked_stat '''
    ax = (0,1) if per_channel else None
    ret = dict(mean=img.mean(ax), std=img.std(ax))

    if mask is not None:
        mask = np.stack([mask]*3, axis=-1)
        neighbor_unmask = np.stack(
            [get_bbox_unmask(mask[...,0])]*3, axis=-1)

        nbr_unmasked_pixels = neighbor_unmask.sum(ax)
        nbr_unmasked_mean = (img*neighbor_unmask).sum(ax)/nbr_unmasked_pixels

        ret['nbr_unmasked_mean']=nbr_unmasked_mean

    return ret



def add_text(image:Image, text, position=(0,0), fontsize=16):
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(fontsize)
    draw.multiline_text(position, text, fill='violet', font=font)

def add_stat(img:Image, st, position=(0,240), fontsize=32):
    format_ = lambda arr: ', '.join(map(lambda v: f"{v:.2f}", arr))
    total_means = format_(st['mean'])
    total_stds = format_(st['std'])
    masked_means = format_(st['masked_mean'])
    masked_stds = format_(st['masked_std'])
    unmasked_means = format_(st['unmasked_mean'])
    unmasked_stds = format_(st['unmasked_std'])

    msg = [f"total_mean:{total_means}",
           f"total_std:{total_stds}",
           "",
           f"masked_mean:{masked_means}",
           f"masked_std:{masked_stds}",
           "",
           f"unmasked_mean:{unmasked_means}",
           f"unmasked_std:{unmasked_stds}"]
    
    if 'nbr_unmasked_mean' in st.keys():
        nbr_unmasked_means = format_(st['nbr_unmasked_mean'])
        nbr_unmasked_stds = format_(st['nbr_unmasked_std'])
        msg.extend(["",
            f"nbr_unmasked_mean:{nbr_unmasked_means}",
            f"nbr_unmasked_std:{nbr_unmasked_stds}"])

    add_text(img, '\n'.join(msg), position, fontsize)



def get_normalize(ori:np.ndarray,ipt:np.ndarray,mask:np.ndarray,stat:dict) -> Image:
    mask = np.stack([mask]*3, axis=-1)

    from PIL import ImageFilter
    m_img = Image.fromarray((mask*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=3))
    blur_mask = np.asarray(m_img) / 255.0
  
    ours_np_ = (ipt.astype(np.float32) - stat['ipt']['masked_mean'])/stat['ipt']['masked_std']
    ours_np_ = ours_np_ * stat['ori']['unmasked_std'] + stat['ori']['unmasked_mean']
    ours_np = ours_np_ / 255.  # ours_np = ipt / 255.0
    
    img_np = ori / 255.0

    ours_np = ours_np * blur_mask + (1 - blur_mask) * img_np
    image_inpaint_compensate = np.uint8(ours_np * 255)
    return Image.fromarray(image_inpaint_compensate.astype(np.uint8))



def get_compensation(ori:np.ndarray,ipt:np.ndarray,mask:np.ndarray,delta_mean:float, fac=1.0) -> Image:
    mask = np.stack([mask]*3, axis=-1)

    from PIL import ImageFilter
    m_img = Image.fromarray((mask*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=3))
    blur_mask = np.asarray(m_img) / 255.0  # blur_mask = mask #

    ori_ = ori/255.0
    ipt_ = ipt/255.0
    cps = (ipt_ + fac * delta_mean/255.0) * blur_mask + (1 - blur_mask) * ori_
    
    image_inpaint_compensate = np.uint8(cps.clip(0, 1.) * 255)
    return Image.fromarray(image_inpaint_compensate)


def get_bbox_unmask(mask:np.ndarray, off=10)-> np.ndarray:
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    
    min_r, max_r = np.argmax(rows), len(rows) - np.argmax(rows[::-1]) - 1
    min_c, max_c = np.argmax(cols), len(cols) - np.argmax(cols[::-1]) - 1

    # Adjust the box size
    h, w = max_r - min_r + 1, max_c - min_c + 1
    new_h, new_w = int(h + off), int(w + off) # new_h, new_w = int(h * fac), int(w * fac)
    
    # Cal new center
    c_r, c_c = (min_r + max_r) // 2, (min_c + max_c) // 2
    
    # Cal new corners
    new_min_r = max(0, c_r - new_h // 2)
    new_max_r = min(mask.shape[0] - 1, c_r + new_h // 2)
    new_min_c = max(0, c_c - new_w // 2)
    new_max_c = min(mask.shape[1] - 1, c_c + new_w // 2)

    new_mask = np.zeros_like(mask, dtype=np.bool_)
    new_mask[new_min_r:new_max_r + 1, new_min_c:new_max_c + 1] = True
    new_mask[np.where(mask == True)] = False

    return new_mask


def paste_compensate(mask:Image, image:Image, result:Image, fac=1.0):
    np_img = np.asarray(image)
    np_msk = np.asarray(mask) > 128
    np_ipt = np.asarray(result)
    ori_stat = get_unmasked_stat(np_img, np_msk)
    ipt_stat = get_unmasked_stat(np_ipt, np_msk)

    delta_mean = ori_stat['nbr_unmasked_mean'] - ipt_stat['nbr_unmasked_mean']

    return get_compensation(np_img, np_ipt, np_msk, delta_mean, fac=fac)
