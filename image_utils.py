import os
import cv2
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
import torch


def focal_to_fov(focal_length, sensor_dimension):
    """Convert focal length to field of view in degrees"""
    import math

    fov_radians = 2 * math.atan(sensor_dimension / (2 * focal_length))
    return fov_radians * (180 / math.pi)


def image_to_equirectangular(image, h_fov, output_width=2048, output_height=1024, wrap_around=False):
    """
    Maps a single image to equirectangular coordinates with correct wrap-around handling.
    """
    # Convert PIL to numpy if needed
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Calculate vertical FOV based on aspect ratio
    img_height, img_width = image.shape[:2]
    aspect_ratio = img_height / img_width
    v_fov = h_fov * aspect_ratio

    # Create empty equirectangular image
    equirect = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    # Create pixel coordinates for the equirectangular image
    x_eq = np.linspace(0, output_width - 1, output_width).astype(np.float32)
    y_eq = np.linspace(0, output_height - 1, output_height).astype(np.float32)
    x_eq, y_eq = np.meshgrid(x_eq, y_eq)

    # Convert coordinates to spherical angles
    theta = (x_eq / output_width) * 2 * np.pi - np.pi
    phi = (y_eq / output_height) * np.pi - np.pi / 2

    # Convert FOVs to radians
    h_fov_rad = np.radians(h_fov)
    v_fov_rad = np.radians(v_fov)

    valid_points = (
        (theta >= -h_fov_rad / 2) & (theta <= h_fov_rad / 2) & (phi >= -v_fov_rad / 2) & (phi <= v_fov_rad / 2)
    )

    u = theta / (h_fov_rad / 2)
    v = phi / (v_fov_rad / 2)

    x_img = ((u + 1) / 2) * (img_width - 1)
    y_img = ((v + 1) / 2) * (img_height - 1)

    valid_y = y_eq[valid_points].astype(int)
    valid_x = x_eq[valid_points].astype(int)
    valid_img_y = y_img[valid_points].astype(int)
    valid_img_x = x_img[valid_points].astype(int)

    equirect[valid_y, valid_x] = image[valid_img_y, valid_img_x]

    non_wrapped = equirect.copy()

    if wrap_around:
        # Wrap-around handling (split between left/right edges)
        left_edge = (theta <= -np.pi + h_fov_rad / 2) & (phi >= -v_fov_rad / 2) & (phi <= v_fov_rad / 2)
        right_edge = (theta >= np.pi - h_fov_rad / 2) & (phi >= -v_fov_rad / 2) & (phi <= v_fov_rad / 2)

        # Left edge maps to RIGHT half of image
        u_left = (theta[left_edge] + np.pi) / (h_fov_rad / 2)  # Corrected mapping
        v_left = phi[left_edge] / (v_fov_rad / 2)

        # Right edge maps to LEFT half of image
        u_right = (theta[right_edge] - np.pi) / (h_fov_rad / 2)  # Corrected mapping
        v_right = phi[right_edge] / (v_fov_rad / 2)

        # Convert to pixel coordinates
        x_img_left = ((u_left) * 0.5 + 0.5) * (img_width - 1)
        y_img_left = ((v_left + 1) / 2) * (img_height - 1)

        x_img_right = ((u_right + 1) / 2) * (img_width - 1)
        y_img_right = ((v_right + 1) / 2) * (img_height - 1)

        # Handle array indices
        valid_y_left = y_eq[left_edge].astype(int)
        valid_x_left = x_eq[left_edge].astype(int)
        valid_img_y_left = np.clip(y_img_left.astype(int), 0, img_height - 1)
        valid_img_x_left = np.clip(x_img_left.astype(int), 0, img_width - 1)

        valid_y_right = y_eq[right_edge].astype(int)
        valid_x_right = x_eq[right_edge].astype(int)
        valid_img_y_right = np.clip(y_img_right.astype(int), 0, img_height - 1)
        valid_img_x_right = np.clip(x_img_right.astype(int), 0, img_width - 1)

        # Populate equirectangular image
        equirect[valid_y_left, valid_x_left] = image[valid_img_y_left, valid_img_x_left]
        equirect[valid_y_right, valid_x_right] = image[valid_img_y_right, valid_img_x_right]

        # Combine extracted parts
        valid_img_y = np.concatenate([valid_img_y_left, valid_img_y_right])
        valid_img_x = np.concatenate([valid_img_x_left, valid_img_x_right])

    return equirect, non_wrapped


def render_perspective(
    equirect_image, yaw_deg, pitch_deg, h_fov_deg, v_fov_deg, output_size=1024, wrap_x=True
):  # ← new flag
    """
    Renders a perspective view from an equirectangular image.

    Parameters
    ----------
    wrap_x : bool, default=True
        • True  –  old behaviour; horizontal indices are taken modulo width,
                   so rays that cross ±180° re-enter on the opposite side.
        • False –  indices are clamped to [0 , width-1]; pixels whose ray
                   would have crossed the seam stay black in the output.
    """
    eq_h, eq_w = equirect_image.shape[:2]
    out_w = out_h = output_size

    # ---------- angles ----------
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    h_fov = math.radians(h_fov_deg)
    v_fov = math.radians(v_fov_deg)

    # ---------- pixel grid in NDC ----------
    out_x = np.linspace(-1, 1, out_w)
    out_y = np.linspace(1, -1, out_h)
    out_xx, out_yy = np.meshgrid(out_x, out_y)

    # ---------- direction vectors ----------
    dir_x = np.tan(h_fov / 2) * out_xx
    dir_y = np.tan(v_fov / 2) * out_yy
    dir_z = np.ones_like(dir_x)
    norm = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
    dir_x /= norm
    dir_y /= norm
    dir_z /= norm

    # ---------- rotate (pitch, yaw) ----------
    # pitch (look-down positive, look-up negative)
    tmp_y = dir_y * math.cos(pitch) - dir_z * math.sin(pitch)
    tmp_z = dir_y * math.sin(pitch) + dir_z * math.cos(pitch)
    dir_y, dir_z = tmp_y, tmp_z

    # yaw (around world-Y axis, right positive)
    tmp_x = dir_x * math.cos(yaw) + dir_z * math.sin(yaw)
    tmp_z = -dir_x * math.sin(yaw) + dir_z * math.cos(yaw)
    dir_x, dir_z = tmp_x, tmp_z

    # ---------- to spherical ----------
    theta = np.arctan2(dir_x, dir_z)  # [-π , +π]
    phi = np.arcsin(dir_y)  # [-π/2 , +π/2]

    # ---------- map to equirect ----------
    eq_x = (theta + np.pi) * (eq_w / (2 * np.pi))
    eq_y = (np.pi / 2 - phi) * (eq_h / np.pi)

    # ---------- horizontal wrap OR clamp ----------
    if wrap_x:
        eq_x = np.mod(eq_x, eq_w)
        valid = np.ones_like(eq_x, dtype=bool)
    else:  # ← clamp + validity mask
        valid = (eq_x >= 0) & (eq_x < eq_w)
        eq_x = np.clip(eq_x, 0, eq_w - 1)

    # ---------- vertical clamp ----------
    eq_y = np.clip(eq_y, 0, eq_h - 1)

    map_x = eq_x.astype(np.int32)
    map_y = eq_y.astype(np.int32)

    # ---------- gather ----------
    perspective = np.zeros((out_h, out_w, 3), dtype=equirect_image.dtype)
    perspective[valid] = equirect_image[map_y[valid], map_x[valid]]

    return perspective


# Placeholder for create_mask_from_black


def create_mask_from_black(image, threshold=10):
    # ... (Implementation from the previous response) ...
    # --- Start of actual create_mask_from_black logic (copy from previous answer) ---
    if len(image.shape) == 3:
        # Handle cases where image might be read as read-only
        image_writable = np.copy(image)
        mask = np.all(image_writable <= threshold, axis=2)
    else:
        mask = image <= threshold
    binary_mask = (mask * 255).astype(np.uint8)
    # --- End of actual create_mask_from_black logic ---
    return binary_mask  # Return mask as 0 or 255


def project_mask_onto_equirect(
    mask_to_project,
    equirect_to_highlight,
    original_equirect,
    yaw_deg,
    pitch_deg,
    h_fov_deg,
    v_fov_deg,
    highlight_color=np.array([255, 0, 0]),
    alpha=0.5,
):
    """
    Projects a perspective mask onto an equirectangular image as a highlight.
    FIXED: Uses correct inverse rotation for pitch to match render_perspective.
    """
    persp_h, persp_w = mask_to_project.shape[:2]
    eq_h, eq_w = equirect_to_highlight.shape[:2]

    # Ensure highlight color is broadcastable
    highlight_color = np.array(highlight_color).reshape(1, 1, 3)

    # Convert angles to radians
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    h_fov = math.radians(h_fov_deg)
    v_fov = math.radians(v_fov_deg)

    # Equirectangular pixel coordinates
    eq_x_coords = np.arange(eq_w)
    eq_y_coords = np.arange(eq_h)
    eq_xx, eq_yy = np.meshgrid(eq_x_coords, eq_y_coords)

    # Convert equirectangular pixel to spherical coords
    theta = (eq_xx / eq_w) * 2 * np.pi - np.pi  # Azimuth [-pi, pi]
    phi = (eq_yy / eq_h) * np.pi - np.pi / 2  # Elevation [-pi/2, pi/2]

    # Convert spherical to Cartesian world coords
    world_x = np.cos(phi) * np.sin(theta)
    world_y = np.sin(phi)
    world_z = np.cos(phi) * np.cos(theta)

    # Apply inverse Yaw rotation
    cam_x = world_x * math.cos(-yaw) + world_z * math.sin(-yaw)
    cam_z = -world_x * math.sin(-yaw) + world_z * math.cos(-yaw)
    world_x = cam_x
    world_z = cam_z

    # FIXED: Apply inverse Pitch rotation that correctly matches render_perspective
    cam_y = world_y * math.cos(-pitch) + world_z * math.sin(-pitch)
    cam_z = -world_y * math.sin(-pitch) + world_z * math.cos(-pitch)
    world_y = cam_y
    world_z = cam_z

    # Project onto camera plane
    valid_mask_geom = world_z > 1e-6
    persp_ndc_x = world_x / world_z
    persp_ndc_y = world_y / world_z

    # Map to pixel coordinates
    persp_u = (persp_ndc_x / np.tan(h_fov / 2) + 1) * 0.5 * persp_w
    persp_v = (-persp_ndc_y / np.tan(v_fov / 2) + 1) * 0.5 * persp_h

    # Identify pixels within the perspective view
    in_view_mask = (persp_u >= 0) & (persp_u < persp_w) & (persp_v >= 0) & (persp_v < persp_h) & valid_mask_geom

    # Get valid coordinates
    eq_y_valid = eq_yy[in_view_mask].astype(int)
    eq_x_valid = eq_xx[in_view_mask].astype(int)
    persp_v_valid = np.clip(persp_v[in_view_mask].astype(int), 0, persp_h - 1)
    persp_u_valid = np.clip(persp_u[in_view_mask].astype(int), 0, persp_w - 1)

    # Identify masked pixels
    is_masked_in_persp = mask_to_project[persp_v_valid, persp_u_valid] > 128

    # Get equirectangular coords to highlight
    eq_y_to_highlight = eq_y_valid[is_masked_in_persp]
    eq_x_to_highlight = eq_x_valid[is_masked_in_persp]

    # Apply alpha blending
    original_colors = original_equirect[eq_y_to_highlight, eq_x_to_highlight].astype(np.float32)
    highlight_color_float = highlight_color.astype(np.float32)
    blended_colors = original_colors * (1.0 - alpha) + highlight_color_float * alpha

    equirect_to_highlight[eq_y_to_highlight, eq_x_to_highlight] = np.clip(blended_colors, 0, 255).astype(np.uint8)

    # Return coordinates for further use if needed
    return eq_y_to_highlight, eq_x_to_highlight


# --- Assume other functions (create_mask_from_black, project_mask_onto_equirect, etc.) exist ---


# Placeholder for create_mask_from_black
def create_mask_from_black(image, threshold=10):
    # ... (Implementation from the previous response) ...
    # --- Start of actual create_mask_from_black logic (copy from previous answer) ---
    if len(image.shape) == 3:
        # Handle cases where image might be read as read-only
        image_writable = np.copy(image)
        mask = np.all(image_writable <= threshold, axis=2)
    else:
        mask = image <= threshold
    binary_mask = (mask * 255).astype(np.uint8)
    # --- End of actual create_mask_from_black logic ---
    return binary_mask  # Return mask as 0 or 255


# Placeholder for project_mask_onto_equirect
# Modify this function


def project_mask_onto_equirect(
    mask_to_project,
    equirect_to_highlight,
    original_equirect,
    yaw_deg,
    pitch_deg,
    h_fov_deg,
    v_fov_deg,
    highlight_color=np.array([255, 0, 0]),
    alpha=0.5,
):
    """
    Projects a perspective mask onto an equirectangular image as a highlight.
    FIXED: Uses rotation matrices that are exactly inverse to render_perspective.
    """
    persp_h, persp_w = mask_to_project.shape[:2]
    eq_h, eq_w = equirect_to_highlight.shape[:2]

    # Ensure highlight color is broadcastable
    highlight_color = np.array(highlight_color).reshape(1, 1, 3)

    # Convert angles to radians
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    h_fov = math.radians(h_fov_deg)
    v_fov = math.radians(v_fov_deg)

    # Equirectangular pixel coordinates
    eq_x_coords = np.arange(eq_w)
    eq_y_coords = np.arange(eq_h)
    eq_xx, eq_yy = np.meshgrid(eq_x_coords, eq_y_coords)

    # Convert equirectangular pixel to spherical coords
    theta = (eq_xx / eq_w) * 2 * np.pi - np.pi  # Azimuth [-pi, pi]
    phi = (eq_yy / eq_h) * np.pi - np.pi / 2  # Elevation [-pi/2, pi/2]

    # Convert spherical to Cartesian world coords
    world_x = np.cos(phi) * np.sin(theta)
    world_y = np.sin(phi)
    world_z = np.cos(phi) * np.cos(theta)

    # Apply inverse Yaw rotation
    cam_x = world_x * math.cos(-yaw) + world_z * math.sin(-yaw)
    cam_z = -world_x * math.sin(-yaw) + world_z * math.cos(-yaw)
    world_x = cam_x
    world_z = cam_z

    # FIXED: Use the TRUE inverse of the rotation matrix used in render_perspective
    # In render_perspective we use:
    # rotated_y = dir_y * math.cos(pitch) - dir_z * math.sin(pitch)
    # rotated_z = dir_y * math.sin(pitch) + dir_z * math.cos(pitch)
    # The exact inverse of this rotation is:
    cam_y = world_y * math.cos(pitch) + world_z * math.sin(pitch)
    cam_z = -world_y * math.sin(pitch) + world_z * math.cos(pitch)
    world_y = cam_y
    world_z = cam_z

    # Project onto camera plane
    valid_mask_geom = world_z > 1e-6
    persp_ndc_x = world_x / world_z
    persp_ndc_y = world_y / world_z

    # Map to pixel coordinates
    persp_u = (persp_ndc_x / np.tan(h_fov / 2) + 1) * 0.5 * persp_w
    persp_v = (-persp_ndc_y / np.tan(v_fov / 2) + 1) * 0.5 * persp_h

    # Identify pixels within the perspective view
    in_view_mask = (persp_u >= 0) & (persp_u < persp_w) & (persp_v >= 0) & (persp_v < persp_h) & valid_mask_geom

    # Get valid coordinates
    eq_y_valid = eq_yy[in_view_mask].astype(int)
    eq_x_valid = eq_xx[in_view_mask].astype(int)
    persp_v_valid = np.clip(persp_v[in_view_mask].astype(int), 0, persp_h - 1)
    persp_u_valid = np.clip(persp_u[in_view_mask].astype(int), 0, persp_w - 1)

    # Identify masked pixels
    is_masked_in_persp = mask_to_project[persp_v_valid, persp_u_valid] > 128

    # Get equirectangular coords to highlight
    eq_y_to_highlight = eq_y_valid[is_masked_in_persp]
    eq_x_to_highlight = eq_x_valid[is_masked_in_persp]

    # Apply alpha blending
    original_colors = original_equirect[eq_y_to_highlight, eq_x_to_highlight].astype(np.float32)
    highlight_color_float = highlight_color.astype(np.float32)
    blended_colors = original_colors * (1.0 - alpha) + highlight_color_float * alpha

    equirect_to_highlight[eq_y_to_highlight, eq_x_to_highlight] = np.clip(blended_colors, 0, 255).astype(np.uint8)

    # Return coordinates for further use if needed
    return eq_y_to_highlight, eq_x_to_highlight


def crosses_seam(yaw_deg: float, h_fov_deg: float) -> bool:
    """Return True if the FoV centred at yaw cuts the –180/+180 seam."""
    half = h_fov_deg / 2.0
    yaw_local = ((yaw_deg + 180) % 360) - 180  # -> (-180, +180]
    return (180 - abs(yaw_local)) < half  # True → crosses


def visualize_all_inpainting_masks(initial_panorama, side_pano, output_size=1024):
    """
    Generates visualizations for top, bottom, and side view masks.
    FIXED: Swapped top/bottom labels to match the correct renders.
    """
    visualization_data = []
    eq_h, eq_w = initial_panorama.shape[:2]
    side_params = [(i * 45, 0) for i in range(8)]
    # Reverse last three for correct order of Meta paper
    side_params[-3:] = side_params[-3:][::-1]

    # Define view parameters with CORRECTED labels
    view_sets = {
        # What was labeled "Bottom" is actually showing top views
        "Top": {
            # looking up
            "params": [
                (0, -60),
                (180, -60),
                # (130, -60),
                # (230, -60),
            ],
            "wraps": [False, True],
            "fov": 120.0,
            "color": [0, 100, 255],
        },
        # What was labeled "Top" is actually showing bottom views
        "Bottom": {
            # Positive pitch actually looks down in render_perspective
            "params": [(0, 60), (180, 60)],
            # (130, 60), (230, 60)],
            "fov": 120.0,
            "color": [0, 180, 50],  # Greenish
        },
        "Side": {
            "params": side_params,  # Yaw, Pitch (0, 45, 90, ..., 315)
            "fov": 85.0,
            "color": [255, 100, 0],  # Orangish
        },
    }

    for view_type, config in view_sets.items():
        fov = config["fov"]
        v_fov = fov  # Assuming square output_size
        color = config["color"]

        for i, (yaw, pitch) in enumerate(config["params"]):
            label = f"{view_type} {i+1}"

            if view_type == "Side":
                initial_panorama = side_pano.copy()
            else:
                initial_panorama = initial_panorama.copy()

            wrap = crosses_seam(yaw, fov)
            perspective_render = render_perspective(initial_panorama, yaw, -pitch, fov, v_fov, output_size, wrap_x=wrap)

            mask_persp = render_perspective(initial_panorama, yaw, pitch, fov, v_fov, output_size, wrap_x=wrap)

            # Create the mask for black regions in this view
            mask = create_mask_from_black(perspective_render, threshold=10)
            mask_fill = create_mask_from_black(mask_persp, threshold=10)

            # Create a fresh copy of the initial panorama for this specific highlight
            pano_for_highlight = initial_panorama.copy()

            # Project this mask back onto the highlight panorama
            if np.sum(mask) > 0:
                project_mask_onto_equirect(
                    mask_fill,
                    pano_for_highlight,
                    initial_panorama,  # Base for blending
                    yaw,
                    pitch,
                    fov,
                    v_fov,
                    highlight_color=np.array(color),
                    alpha=0.6,
                )
            else:
                raise ValueError(f"Mask for {label} is empty. Check the input image or mask generation.")
            # Store results for this view
            visualization_data.append(
                {
                    "label": label,
                    "yaw": yaw,
                    "pitch": pitch,
                    "fov": fov,
                    "vfov": v_fov,
                    "render": perspective_render,
                    "mask": mask,
                    "mask_fill": mask_fill,
                    "highlighted_pano": pano_for_highlight,
                }
            )

    return visualization_data


def project_perspective_to_equirect(
    perspective_image, equirect_target, yaw_deg, pitch_deg, h_fov_deg, v_fov_deg, mask=None, mirror=False, wrap_x=True
):
    """
    Projects a perspective image back onto an equirectangular panorama.
    Carefully matches the exact inverse transformations of render_perspective.
    """
    persp_h, persp_w = perspective_image.shape[:2]
    eq_h, eq_w = equirect_target.shape[:2]

    # Convert angles to radians
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    h_fov = math.radians(h_fov_deg)
    v_fov = math.radians(v_fov_deg)

    # Create equirectangular coordinate maps
    eq_x_coords = np.arange(eq_w)
    eq_y_coords = np.arange(eq_h)
    eq_xx, eq_yy = np.meshgrid(eq_x_coords, eq_y_coords)

    # Convert equirectangular pixel to spherical coords
    theta = (eq_xx / eq_w) * 2 * np.pi - np.pi  # Azimuth [-pi, pi]
    phi = (eq_yy / eq_h) * np.pi - np.pi / 2  # Elevation [-pi/2, pi/2]

    # Convert spherical to Cartesian world coords
    world_x = np.cos(phi) * np.sin(theta)
    world_y = np.sin(phi)
    world_z = np.cos(phi) * np.cos(theta)

    # Apply inverse Yaw rotation (around Y-axis)
    # This matches the inverse of: tmp_x = dir_x * math.cos(yaw) + dir_z * math.sin(yaw)
    cam_x = world_x * math.cos(-yaw) + world_z * math.sin(-yaw)
    cam_z = -world_x * math.sin(-yaw) + world_z * math.cos(-yaw)
    world_x = cam_x
    world_z = cam_z

    # Apply inverse Pitch rotation (around X-axis)
    # This matches the inverse of:
    # tmp_y = dir_y * math.cos(pitch) - dir_z * math.sin(pitch)
    # tmp_z = dir_y * math.sin(pitch) + dir_z * math.cos(pitch)
    cam_y = world_y * math.cos(-pitch) + world_z * math.sin(-pitch)
    cam_z = -world_y * math.sin(-pitch) + world_z * math.cos(-pitch)
    world_y = cam_y
    world_z = cam_z

    # Project onto camera plane
    valid_mask_geom = world_z > 1e-6  # Points in front of the camera
    persp_ndc_x = world_x / world_z
    persp_ndc_y = world_y / world_z

    # Map to pixel coordinates in the perspective image
    persp_u = (persp_ndc_x / np.tan(h_fov / 2) + 1) * 0.5 * persp_w
    persp_v = (-persp_ndc_y / np.tan(v_fov / 2) + 1) * 0.5 * persp_h

    # Identify pixels within the perspective view
    in_view_mask = (persp_u >= 0) & (persp_u < persp_w) & (persp_v >= 0) & (persp_v < persp_h) & valid_mask_geom

    # Get valid equirectangular coordinates to update
    eq_y_valid = eq_yy[in_view_mask]
    eq_x_valid = eq_xx[in_view_mask]

    # Get corresponding perspective view coordinates
    persp_v_valid = np.clip(persp_v[in_view_mask], 0, persp_h - 1)
    persp_u_valid = np.clip(persp_u[in_view_mask], 0, persp_w - 1)

    # Apply mask if provided
    if mask is not None:
        # Sample mask at perspective coordinates
        persp_v_int = persp_v_valid.astype(int)
        persp_u_int = persp_u_valid.astype(int)
        mask_values = mask[persp_v_int, persp_u_int]

        # Only update where mask is non-zero
        update_mask = mask_values > 0
        eq_y_valid = eq_y_valid[update_mask].astype(int)
        eq_x_valid = eq_x_valid[update_mask].astype(int)
        persp_v_valid = persp_v_valid[update_mask]
        persp_u_valid = persp_u_valid[update_mask]
    else:
        eq_y_valid = eq_y_valid.astype(int)
        eq_x_valid = eq_x_valid.astype(int)

    # Sample the perspective image at floating-point coordinates
    persp_v_int = persp_v_valid.astype(int)
    persp_u_int = persp_u_valid.astype(int)

    # Update the equirectangular image
    # IMPORTANT: No need to flip y-coordinates - we're using consistent coordinate systems
    fixed_eq_valid = eq_h - eq_y_valid
    fixed_eq_valid[fixed_eq_valid < 0] = 0
    fixed_eq_valid[fixed_eq_valid >= eq_h] = eq_h - 1
    if mirror:
        eq_x_valid = eq_w - eq_x_valid - 1
        eq_x_valid[eq_x_valid < 0] = 0
        eq_x_valid[eq_x_valid >= eq_w] = eq_w - 1

    equirect_target[fixed_eq_valid, eq_x_valid] = perspective_image[persp_v_int, persp_u_int]

    print(f"Projected {len(eq_y_valid)} pixels from perspective to equirect with yaw={yaw_deg}, pitch={pitch_deg}")
    return equirect_target
