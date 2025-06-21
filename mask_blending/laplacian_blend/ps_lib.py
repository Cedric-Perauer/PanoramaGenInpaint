import numpy as np
from PIL import Image
import cv2


def read_image(path):
    """
    Read an image from path and return as numpy array
    """
    img = Image.open(path)
    return np.array(img)


def write_image(path, image):
    """
    Write a numpy array image to path
    """
    if image.dtype != np.uint8:
        # Normalize to 0-255 range if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(image)
    img.save(path)


def pad(image, u_pad=0, v_pad=0, mode='constant', constant_values=0):
    """
    Pad an image with zeros or other values
    """
    if len(image.shape) == 2:
        return np.pad(image, ((u_pad, u_pad), (v_pad, v_pad)),
                     mode=mode, constant_values=constant_values)
    elif len(image.shape) == 3:
        return np.pad(image, ((u_pad, u_pad), (v_pad, v_pad), (0, 0)),
                     mode=mode, constant_values=constant_values)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


def resize(image, size):
    """
    Resize an image to the specified size
    size: [height, width] or (height, width)
    """
    if len(image.shape) == 2:
        # Grayscale image
        resized = cv2.resize(image, (size[1], size[0]),
                           interpolation=cv2.INTER_LINEAR)
    elif len(image.shape) == 3:
        # Color image
        resized = cv2.resize(image, (size[1], size[0]),
                           interpolation=cv2.INTER_LINEAR)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    
    return resized 