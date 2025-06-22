import cv2
import numpy as np


def fix_mask_region(mask, extension=0):
    """
    Extract the largest white region from a mask and optionally extend it.

    Args:
        mask: Input mask image (grayscale or BGR)
        extension: Number of pixels to extend the mask region (default: 0)

    Returns:
        numpy.ndarray: Binary mask with only the largest white region, optionally extended
    """
    if mask.shape[-1] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour by area
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Create an empty mask
        largest_mask = np.zeros_like(binary)
        # Draw the largest contour filled in white
        cv2.drawContours(largest_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # Extend the mask if extension > 0
        if extension > 0:
            kernel = np.ones((extension, extension), np.uint8)
            largest_mask = cv2.dilate(largest_mask, kernel, iterations=1)

        return largest_mask
    else:
        # Return empty mask if no contours found
        return np.zeros_like(binary)


def extract_largest_white_region(mask_path, extension=0):
    """
    Load a mask image and keep only the largest white region by finding contours.

    Args:
        mask_path (str): Path to the mask image file
        extension (int): Number of pixels to extend the mask region (default: 0)

    Returns:
        numpy.ndarray: Binary mask with only the largest white region
    """
    # Load the mask as grayscale
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Use the fix_mask_region function
    return fix_mask_region(img, extension)
