import os
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import gc
import torch
import subprocess


def run_with_conda_env(env_name, script_path):
    cmd = f"""
    source ~/miniconda/etc/profile.d/conda.sh
    conda activate {env_name}
    python {script_path}
    """
    subprocess.run(['bash', '-c', cmd], check=True)


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


def preprocess_image(image):
    image = image.convert("RGB")
    image = transforms.CenterCrop(
        (image.size[1] // 64 * 64, image.size[0] // 64 * 64))(image)
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    return image


def preprocess_mask(mask):
    mask = mask.convert("L")
    mask = transforms.CenterCrop(
        (mask.size[1] // 64 * 64, mask.size[0] // 64 * 64))(mask)
    mask = transforms.ToTensor()(mask)
    mask = mask
    return mask


def show_image_cv2(img):
    plt.figure(figsize=(10, 8))
    # Convert BGR to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_img)
    plt.axis('off')
    plt.show()


def pil_to_cv2(pil_image):
    """Convert a PIL Image to an OpenCV image (numpy array in BGR format)"""
    # Convert PIL image to numpy array (RGB format)
    rgb_image = np.array(pil_image)
    # Convert RGB to BGR (OpenCV uses BGR)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return bgr_image

# OpenCV to PIL Image


def cv2_to_pil(cv2_image):
    """Convert an OpenCV image to a PIL Image"""
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(rgb_image)
    return pil_image
