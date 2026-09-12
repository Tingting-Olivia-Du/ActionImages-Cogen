import os
import math
import torch
import imageio
import numpy as np
from PIL import Image
from einops import rearrange
from scipy.spatial.transform import Rotation as R
from typing import List, Optional, Tuple, Union
from torchvision.transforms import v2, functional as F


class CenterCropToAspect:
    """Center-crop to target aspect ratio (height/width) with max area."""

    def __init__(self, h, w):
        self.r = h / w

    def __call__(self, img):
        ih, iw = F.get_image_size(img)
        if ih / iw > self.r:
            crop_w = iw
            crop_h = int(round(iw * self.r))
        else:
            crop_h = ih
            crop_w = int(round(ih / self.r))
        return F.center_crop(img, [crop_h, crop_w])


def convert_intrinsics_after_center_crop_resize(
    intrinsics_list: List[np.ndarray],
    orig_res_list: List[Tuple[int, int]],
    new_res_list: List[Tuple[int, int]],
) -> List[np.ndarray]:
    """
    Convert camera intrinsics for a crop-then-resize image transform.

    Input:
      - intrinsics_list: length V, each element is an array of shape (N, 3, 3)
      - orig_res_list:   length V, each element is (H, W) for the original image
      - new_res_list:    length V, each element is (H_new, W_new) for the target image

    Output:
      - new_intrinsics_list: length V, each element is an array of shape (N, 3, 3)

    Image transform (per view):
      1) Center-crop the original image to the *maximal* rectangle with aspect ratio W_new / H_new
      2) Resize the crop to (H_new, W_new)

    Intrinsics update (per K):
      - Let crop offset be (dx, dy) in pixels from the original top-left
      - Let scale be (sx, sy) from crop to new resolution
      - Then:
          fx' = fx * sx
          fy' = fy * sy
          cx' = (cx - dx) * sx
          cy' = (cy - dy) * sy
          (and keep the last row as [0, 0, 1])

    Notes:
      - Center crop is symmetric; if (W - Wc) or (H - Hc) is odd, dx/dy will be .5 — that’s fine.
      - No assumptions about square pixels; fx, fy are scaled independently.
      - Works for any N per view (including N=1).
    """
    if not (len(intrinsics_list) == len(orig_res_list) == len(new_res_list)):
        raise ValueError("All input lists must have the same length V.")

    new_intrinsics_list: List[np.ndarray] = []

    for Ks, (H, W), (Hn, Wn) in zip(intrinsics_list, orig_res_list, new_res_list):
        if Ks.ndim != 3 or Ks.shape[1:] != (3, 3):
            raise ValueError("Each intrinsics element must have shape (N, 3, 3).")

        # Aspect ratios
        r_target = Wn / float(Hn)  # target aspect (width / height)
        r_orig = W / float(H)

        # Compute maximal center crop that matches target aspect
        if r_orig >= r_target:
            # Original is wider; keep full height, crop width
            Hc = float(H)
            Wc = Hc * r_target
            dx = (W - Wc) / 2.0
            dy = 0.0
        else:
            # Original is taller; keep full width, crop height
            Wc = float(W)
            Hc = Wc / r_target
            dx = 0.0
            dy = (H - Hc) / 2.0

        # Resize scales (crop -> new)
        sx = Wn / Wc
        sy = Hn / Hc

        # Apply transform to intrinsics
        K_out = Ks.copy().astype(np.float64)
        # Scale focal lengths
        K_out[:, 0, 0] = K_out[:, 0, 0] * sx  # fx
        K_out[:, 1, 1] = K_out[:, 1, 1] * sy  # fy

        # Shift by crop, then scale principal point
        K_out[:, 0, 2] = (K_out[:, 0, 2] - dx) * sx  # cx
        K_out[:, 1, 2] = (K_out[:, 1, 2] - dy) * sy  # cy

        # Enforce last row [0, 0, 1]
        K_out[:, 2, 0] = 0.0
        K_out[:, 2, 1] = 0.0
        K_out[:, 2, 2] = 1.0

        new_intrinsics_list.append(K_out)

    return new_intrinsics_list


def get_rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def get_action_2d(action_3d, extrinsics, intrinsics):
    # Project 3D actions to 2D for each view
    actions_2d_list = []
    for ext, intr in zip(extrinsics, intrinsics):
        projected_2d = project_point_3d_to_2d_batch(action_3d, ext, intr)
        actions_2d_list.append(projected_2d)
    action_2d = np.stack(actions_2d_list)  # view, N, 2
    return action_2d


def project_point_3d_to_2d(point_3d: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """
    Project a 3D point (in world coordinates) onto the 2D image plane.

    Args:
        point_3d: 3D point in world coordinates [x, y, z]
        extrinsics: 4x4 camera extrinsics matrix
        intrinsics: 3x3 camera intrinsics matrix

    Returns:
        Array of shape [2] containing 2D pixel coordinates
    """
    # Convert to homogeneous coordinates
    point_3d_hom = np.append(point_3d, 1.0)

    # Transform to camera coordinate system
    world_to_cam = np.linalg.inv(extrinsics)
    cam_coords = world_to_cam @ point_3d_hom

    # Project to image plane
    pixel = intrinsics @ cam_coords[:3]
    pixel = pixel[:2] / pixel[2]
    return pixel


def project_point_3d_to_2d_batch(points_3d: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """
    Project multiple 3D action points to 2D image coordinates.

    Args:
        points_3d: Array of shape [T, 3]
        extrinsics: camera extrinsics matrix of shape [T, 4, 4]
        intrinsics: camera intrinsics matrix of shape [T, 3, 3]

    Returns:
        Array of shape [T, 2] containing 2D pixel coordinates
    """
    # Extract positions (first 3 columns)
    positions_3d = points_3d[:, :3]  # T, 3
    T = positions_3d.shape[0]
    # Convert to homogeneous coordinates # T, 4
    ones = np.ones((T, 1))
    points_3d_hom = np.concatenate([positions_3d, ones], axis=1)  # T, 4

    # Transform to camera coordinate system
    # Batch matrix inverse: # T, 4, 4
    world_to_cam = np.linalg.inv(extrinsics)  # T, 4, 4
    # Batch matrix multiplication: # T, 4, 4 * T, 4, 1 -> T, 4
    cam_coords = np.einsum("tij,tj->ti", world_to_cam, points_3d_hom)  # T, 4
    # Project to image plane: # T, 3, 3 * T, 3, 1 -> T, 3
    pixel_hom = np.einsum("tij,tj->ti", intrinsics, cam_coords[:, :3])  # T, 3
    # Perspective division: normalize by z-coordinate
    pixel_coords = pixel_hom[:, :2] / pixel_hom[:, 2:3]  # T, 2

    return pixel_coords


def project_point_3d_to_2d_torch_batch(
    points_3d: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor
) -> torch.Tensor:
    """
    Project 3D points to 2D image coordinates using PyTorch tensors.

    Args:
        points_3d: Tensor of shape [..., 3]
        extrinsics: Camera extrinsics matrices of shape [..., 4, 4] (camera-to-world)
        intrinsics: Camera intrinsics matrices of shape [..., 3, 3]

    Returns:
        Tensor of shape [..., 2] containing 2D pixel coordinates
    """
    device = points_3d.device
    dtype = points_3d.dtype
    # Convert to homogeneous coordinates
    ones = torch.ones(*points_3d.shape[:-1], 1, device=device, dtype=dtype)
    points_3d_hom = torch.cat([points_3d, ones], dim=-1)  # N, 4
    # Transform to camera coordinate system (invert camera-to-world extrinsics)
    R = extrinsics[..., :3, :3]  # N, 3, 3
    t = extrinsics[..., :3, 3]  # N, 3
    R_inv = R.transpose(-2, -1)  # N, 3, 3
    t_inv = -(R_inv @ t[..., None])[..., 0]  # N, 3
    world_to_cam = torch.zeros_like(extrinsics)
    world_to_cam[..., :3, :3] = R_inv
    world_to_cam[..., :3, 3] = t_inv
    world_to_cam[..., 3, 3] = 1.0
    # Apply transformation and project to image plane
    cam_coords = torch.einsum("...ij,...j->...i", world_to_cam, points_3d_hom)  # N, 4
    pixel_hom = torch.einsum("...ij,...j->...i", intrinsics, cam_coords[..., :3])  # N, 3
    # Perspective division
    pixel_coords = pixel_hom[..., :2] / pixel_hom[..., 2:3]  # N, 2
    return pixel_coords


def project_actions_7d_to_5d_batch(
    actions_7d: np.ndarray, extrinsics: list, intrinsics: list, length: float = 0.1
) -> np.ndarray:
    """
    Project multiple 7 DoF action points to 5 DoF action points.

    Args:
        actions_3d: Array of shape [T, 7]. [x, y, z, euler_x, euler_y, euler_z, openness]
        extrinsics: list of camera extrinsics matrix of shape [T, 4, 4]
        intrinsics: list of camera intrinsics matrix of shape [T, 3, 3]

    Returns:
        Array of shape [T, 7] - [pos_x, pos_y, normal_x, normal_y, up_x, up_y, openness]
    """
    pos_3d = actions_7d[:, :3]
    # Use rotation to map a fixed tool axis to world, avoiding Euler wrap-around
    rot = R.from_euler("xyz", actions_7d[:, 3:6], degrees=False)
    # Use tool-frame x-axis as forward direction ([1, 0, 0])
    normal_axis = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=actions_7d.dtype), (actions_7d.shape[0], 1))
    dir_world = rot.apply(normal_axis)
    normal_3d = pos_3d + dir_world * length
    up_axis = np.tile(np.array([[0.0, 0.0, -1.0]], dtype=actions_7d.dtype), (actions_7d.shape[0], 1))
    dir_world = rot.apply(up_axis)
    up_3d = pos_3d + dir_world * length
    pos_2d = project_point_3d_to_2d_batch(pos_3d, extrinsics, intrinsics)
    normal_2d = project_point_3d_to_2d_batch(normal_3d, extrinsics, intrinsics)
    up_2d = project_point_3d_to_2d_batch(up_3d, extrinsics, intrinsics)
    return np.concatenate([pos_2d, normal_2d, up_2d, actions_7d[:, 6:]], axis=1)


def project_actions_7d_to_5d_torch_batch(
    actions_7d: torch.Tensor, extrinsics: torch.Tensor, intrinsics: torch.Tensor, length: float = 0.1
) -> torch.Tensor:
    """
    Project multiple 7 DoF action points to 5 DoF action points using PyTorch tensors.

    Args:
        actions_7d: Tensor of shape [B, T, 7]. [x, y, z, euler_x, euler_y, euler_z, openness]
        extrinsics: Camera extrinsics matrices of shape [B, T, 4, 4]
        intrinsics: Camera intrinsics matrices of shape [B, T, 3, 3]
        length: Length of the direction vector for end points

    Returns:
        Tensor of shape [B, T, 7] - [pos_x, pos_y, normal_x, normal_y, up_x, up_y, openness]
    """
    # Get two 3D points from actions_7d
    B, T = actions_7d.shape[:2]
    pos_3d = actions_7d[:, :, :3]  # B, T, 3
    # Compute world-space direction by rotating fixed tool axis with Euler angles
    angles = actions_7d[:, :, 3:6]  # B, T, 3 (rx, ry, rz)
    cx, cy, cz = torch.cos(angles[..., 0]), torch.cos(angles[..., 1]), torch.cos(angles[..., 2])
    sx, sy, sz = torch.sin(angles[..., 0]), torch.sin(angles[..., 1]), torch.sin(angles[..., 2])
    device = actions_7d.device
    dtype = actions_7d.dtype
    # Rotation matrices for intrinsic xyz (apply Rx, then Ry, then Rz)
    Rx = torch.zeros(B, T, 3, 3, device=device, dtype=dtype)
    Rx[..., 0, 0] = 1.0
    Rx[..., 1, 1] = cx
    Rx[..., 1, 2] = -sx
    Rx[..., 2, 1] = sx
    Rx[..., 2, 2] = cx
    Ry = torch.zeros(B, T, 3, 3, device=device, dtype=dtype)
    Ry[..., 0, 0] = cy
    Ry[..., 0, 2] = sy
    Ry[..., 1, 1] = 1.0
    Ry[..., 2, 0] = -sy
    Ry[..., 2, 2] = cy
    Rz = torch.zeros(B, T, 3, 3, device=device, dtype=dtype)
    Rz[..., 0, 0] = cz
    Rz[..., 0, 1] = -sz
    Rz[..., 1, 0] = sz
    Rz[..., 1, 1] = cz
    Rz[..., 2, 2] = 1.0

    # up axis
    v = torch.zeros(B, T, 3, device=device, dtype=dtype)
    v[..., 0] = 1.0
    v = v[..., None]  # B, T, 3, 1
    v = torch.matmul(Rx, v)
    v = torch.matmul(Ry, v)
    v = torch.matmul(Rz, v)
    dir_world = v[..., 0]  # B, T, 3
    normal_3d = pos_3d + dir_world * length  # B, T, 3

    # normal axis
    v = torch.zeros(B, T, 3, device=device, dtype=dtype)
    v[..., 2] = -1.0
    v = v[..., None]  # B, T, 3, 1
    v = torch.matmul(Rx, v)
    v = torch.matmul(Ry, v)
    v = torch.matmul(Rz, v)
    dir_world = v[..., 0]  # B, T, 3
    up_3d = pos_3d + dir_world * length  # B, T, 3

    # Project both points to 2D
    pos_2d = project_point_3d_to_2d_torch_batch(pos_3d, extrinsics, intrinsics)  # B, T, 2
    normal_2d = project_point_3d_to_2d_torch_batch(normal_3d, extrinsics, intrinsics)  # B, T, 2
    up_2d = project_point_3d_to_2d_torch_batch(up_3d, extrinsics, intrinsics)  # B, T, 2

    # Combine projected points with openness value
    return torch.cat([pos_2d, normal_2d, up_2d, actions_7d[:, :, 6:]], dim=-1)  # B, T, 7


def project_action_2d_to_hw(
    actions_2d: np.ndarray, height: int, width: int, value: float = 1.0, sigma: float = 0.05
) -> np.ndarray:
    """
    Project multiple 2D action points to height and width.

    Args:
        actions_2d: Array of shape [T, 2]. [x, y]

    Returns:
        Array of shape [T, height, width]
    """
    T = actions_2d.shape[0]
    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:height, 0:width]  # height, width
    # Parameters for Gaussian distribution
    sigma = min(height, width) * sigma  # Adjust sigma based on image size
    # Vectorized computation using broadcasting
    # Expand dimensions for broadcasting: actions_2d # T, 2 -> T, 1, 1, 2
    # x_coords/y_coords: # height, width -> 1, height, width
    x_centers = actions_2d[:, 0][:, np.newaxis, np.newaxis]  # T, 1, 1
    y_centers = actions_2d[:, 1][:, np.newaxis, np.newaxis]  # T, 1, 1
    # Broadcast coordinate grids to match batch dimension
    x_coords_broadcast = x_coords[np.newaxis, :, :]  # 1, height, width
    y_coords_broadcast = y_coords[np.newaxis, :, :]  # 1, height, width
    # Calculate Gaussian distribution for all time steps at once
    # Broadcasting: # T, 1, 1 - 1, height, width -> T, height, width
    gaussian = np.exp(-((x_coords_broadcast - x_centers) ** 2 + (y_coords_broadcast - y_centers) ** 2) / (2 * sigma**2))

    return gaussian.astype(np.float32) * value


def project_action_2d_to_hw_torch(
    actions_2d: torch.Tensor, height: int, width: int, value: float = 1.0, sigma: float = 0.05
) -> torch.Tensor:
    """
    Project multiple 2D action points to height and width using PyTorch tensors.

    Args:
        actions_2d: Tensor of shape [B, T, 2]. [x, y]
        height: Height of output grid
        width: Width of output grid
        value: Maximum value for Gaussian peaks
        sigma: Standard deviation factor relative to image size

    Returns:
        Tensor of shape [B, T, height, width]
    """
    B, T = actions_2d.shape[:2]
    device = actions_2d.device
    dtype = actions_2d.dtype

    # Create coordinate grids
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype), torch.arange(width, device=device, dtype=dtype), indexing="ij"
    )  # height, width

    # Adjust sigma based on image size
    sigma_adj = min(height, width) * sigma

    # Expand dimensions for broadcasting: actions_2d [B, T, 2] -> [B, T, 1, 1, 2]
    x_centers = actions_2d[:, :, 0][:, :, None, None]  # B, T, 1, 1
    y_centers = actions_2d[:, :, 1][:, :, None, None]  # B, T, 1, 1

    # Broadcast coordinate grids to match batch dimensions
    x_coords_broadcast = x_coords[None, None, :, :]  # 1, 1, height, width
    y_coords_broadcast = y_coords[None, None, :, :]  # 1, 1, height, width

    # Calculate Gaussian distribution for all batch and time steps at once
    # Broadcasting: [B, T, 1, 1] - [1, 1, height, width] -> [B, T, height, width]
    gaussian = torch.exp(
        -((x_coords_broadcast - x_centers) ** 2 + (y_coords_broadcast - y_centers) ** 2) / (2 * sigma_adj**2)
    )

    return gaussian * value


def project_action_5d_to_rgb(actions_5d: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Project multiple 5D action points to RGB image.

    Args:
        actions_5d: Array of shape [T, 5]. [start_x, start_y, end_x, end_y, openness]
        height: int
        width: int

    Returns:
        Array of shape [T, height, width, 3]
    """
    # R channel: first 2 points (pos_x, pos_y) to hw
    R = project_action_2d_to_hw(actions_5d[:, :2], height, width)
    # G channel: middle 2 points (normal_x, normal_y) to hw
    G = project_action_2d_to_hw(actions_5d[:, 2:4], height, width)
    # B channel: last 2 points (up_x, up_y) to hw + openness > 0.5
    B = project_action_2d_to_hw(actions_5d[:, 4:6], height, width)
    B_open = np.ones_like(R) * (actions_5d[:, 6] > 0.5)[:, None, None]
    mask = B <= 0.25
    B[mask] = B_open[mask] * 0.25
    return np.stack([R, G, B], axis=-1)


def project_action_5d_to_rgb_torch(actions_5d: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Project multiple 5D action points to RGB image using PyTorch tensors.

    Args:
        actions_5d: Tensor of shape [B, T, 7]. [pos_x, pos_y, normal_x, normal_y, up_x, up_y, openness]
        height: Height of output image
        width: Width of output image

    Returns:
        Tensor of shape [B, T, height, width, 3]
    """
    # R channel: first 2 points (pos_x, pos_y) to hw
    R = project_action_2d_to_hw_torch(actions_5d[:, :, :2], height, width)  # B, T, H, W

    # G channel: middle 2 points (normal_x, normal_y) to hw
    G = project_action_2d_to_hw_torch(actions_5d[:, :, 2:4], height, width)  # B, T, H, W

    # B channel: last 2 points (up_x, up_y) to hw + openness > 0.5
    B = project_action_2d_to_hw_torch(actions_5d[:, :, 4:6], height, width)
    B_open = torch.ones_like(R) * (actions_5d[:, :, 6] > 0.5)[:, :, None, None]
    mask = B <= 0.25
    B[mask] = B_open[mask] * 0.25

    # Stack channels: B, T, H, W, 3
    return torch.stack([R, G, B], dim=-1)


def get_relative_pose(source_extrinsics: np.ndarray, target_extrinsics: np.ndarray) -> np.ndarray:
    """
    Calculate relative transformation between two camera poses.

    Args:
        source_extrinsics: 4x4 source camera extrinsics matrix - camera to world
        target_extrinsics: 4x4 target camera extrinsics matrix - camera to world

    Returns:
        3x4 relative transformation matrix
    """
    # Calculate relative transformation
    relative_transform = np.linalg.inv(source_extrinsics) @ target_extrinsics  # 4, 4
    return relative_transform[:3, :]  # 3, 4


def get_relative_pose_batch(source_extrinsics: np.ndarray, target_extrinsics: np.ndarray) -> np.ndarray:
    """
    Calculate relative transformation between two camera poses (batched version).

    Args:
        source_extrinsics: 4x4 source camera extrinsics matrix OR Nx4x4 batch of matrices
        target_extrinsics: Nx4x4 batch of target camera extrinsics matrices

    Returns:
        Nx3x4 batch of relative transformation matrices
    """
    # Handle both single source and batch source
    if source_extrinsics.ndim == 2:
        # Single source matrix, broadcast to batch size
        batch_size = target_extrinsics.shape[0]
        source_extrinsics = np.tile(source_extrinsics[None, :, :], (batch_size, 1, 1))

    # Calculate relative transformation (batched)
    relative_transform = np.linalg.inv(source_extrinsics) @ target_extrinsics  # N, 4, 4

    return relative_transform[:, :3, :]  # N, 3, 4


def get_plucker_embeddings(extrinsics: np.ndarray, intrinsics: np.ndarray, resolution: int = (512, 512)) -> np.ndarray:
    """
    Generate Plücker coordinates for camera rays.

    Args:
        extrinsics: Camera extrinsics matrix of shape (T, 3, 4) - [R|T] where R is 3x3 rotation, T is 3x1 translation
        intrinsics: Camera intrinsics matrix of shape (T, 3, 3)
        resolution: Image resolution

    Returns:
        np.ndarray: Plücker coordinates of shape (T, H, W, 6) where each pixel has 6D Plücker coords
                    (normalized direction: 3D + moment: 3D)
    """
    T = extrinsics.shape[0]

    # Extract rotation matrix R and translation vector t from extrinsics
    R = extrinsics[:, :3, :3]  # T, 3, 3
    t = extrinsics[:, :3, 3]  # T, 3

    # Create pixel coordinates
    coord_w, coord_h = np.meshgrid(
        np.linspace(0, resolution[1] - 1, resolution[1]),
        np.linspace(0, resolution[0] - 1, resolution[0]),
        indexing="xy",
    )
    coord_w = coord_w[None, :, :] + 0.5  # 1, H, W
    coord_h = coord_h[None, :, :] + 0.5  # 1, H, W

    # Extract intrinsics parameters
    fx = intrinsics[:, 0, 0:1]  # T, 1
    fy = intrinsics[:, 1, 1:2]  # T, 1
    cx = intrinsics[:, 0, 2:3]  # T, 1
    cy = intrinsics[:, 1, 2:3]  # T, 1

    # Reshape for broadcasting
    fx = fx[:, :, None]  # T, 1, 1
    fy = fy[:, :, None]  # T, 1, 1
    cx = cx[:, :, None]  # T, 1, 1
    cy = cy[:, :, None]  # T, 1, 1

    # Compute ray directions in camera coordinates
    x = (coord_w - cx) / fx  # T, H, W
    y = (coord_h - cy) / fy  # T, H, W
    z = np.ones_like(x)  # T, H, W

    # Stack to get ray directions in camera coordinates
    direction_cam = np.stack([x, y, z], axis=-1)  # T, H, W, 3

    # Transform ray directions to world coordinates
    # R is camera-to-world
    direction_world = np.einsum("tij,thwj->thwi", R, direction_cam)  # T, H, W, 3

    # Normalize direction vectors
    direction_norm = np.linalg.norm(direction_world, axis=-1, keepdims=True)
    direction_norm = direction_world / (direction_norm + 1e-8)  # T, H, W, 3

    # Compute ray origins in world coordinates (camera center)
    # Camera center in world coordinates: t
    origin = t[:, None, None, :].expand(T, resolution[0], resolution[1], 3)  # T, H, W, 3

    # Compute Plücker coordinates: moment = origin × direction
    moment = np.cross(origin, direction_norm, axis=-1)  # T, H, W, 3

    # Concatenate normalized direction and moment to get Plücker coordinates
    plucker_coords = np.concatenate([direction_norm, moment], axis=-1)  # T, H, W, 6

    return plucker_coords


def get_plucker_embeddings_torch(
    extrinsics: torch.Tensor, intrinsics: torch.Tensor, resolution: Tuple[int, int] = (512, 512)
) -> torch.Tensor:
    """
    Generate Plücker coordinates for camera rays using PyTorch tensors.

    Args:
        extrinsics: Camera extrinsics matrix of shape (B, T, 3, 4) - [R|T] where R is 3x3 rotation, T is 3x1 translation
        intrinsics: Camera intrinsics matrix of shape (B, T, 3, 3)
        resolution: Image resolution

    Returns:
        torch.Tensor: Plücker coordinates of shape (B, T, H, W, 6) where each pixel has 6D Plücker coords
                     (normalized direction: 3D + moment: 3D)
    """
    B, T = extrinsics.shape[:2]
    device = extrinsics.device
    dtype = extrinsics.dtype
    intrinsics = intrinsics.to(dtype)

    # Extract rotation matrix R and translation vector t from extrinsics
    R = extrinsics[:, :, :3, :3]  # B, T, 3, 3
    t = extrinsics[:, :, :3, 3]  # B, T, 3

    # Create pixel coordinates
    coord_w, coord_h = torch.meshgrid(
        torch.linspace(0, resolution[1] - 1, resolution[1], device=device, dtype=dtype),
        torch.linspace(0, resolution[0] - 1, resolution[0], device=device, dtype=dtype),
        indexing="xy",
    )
    coord_w = coord_w[None, None, :, :] + 0.5  # 1, 1, H, W
    coord_h = coord_h[None, None, :, :] + 0.5  # 1, 1, H, W

    # Extract intrinsics parameters
    fx = intrinsics[:, :, 0:1, 0:1]  # B, T, 1, 1
    fy = intrinsics[:, :, 1:2, 1:2]  # B, T, 1, 1
    cx = intrinsics[:, :, 0:1, 2:3]  # B, T, 1, 1
    cy = intrinsics[:, :, 1:2, 2:3]  # B, T, 1, 1

    # Compute ray directions in camera coordinates
    x = (coord_w - cx) / fx  # B, T, H, W
    y = (coord_h - cy) / fy  # B, T, H, W
    z = torch.ones_like(x)  # B, T, H, W

    # Stack to get ray directions in camera coordinates
    direction_cam = torch.stack([x, y, z], dim=-1)  # B, T, H, W, 3

    # Transform ray directions to world coordinates
    direction_world = torch.einsum("btij,bthwj->bthwi", R, direction_cam)  # B, T, H, W, 3

    # Normalize direction vectors
    direction_norm_magnitude = torch.linalg.norm(direction_world, dim=-1, keepdim=True)
    direction_norm = direction_world / (direction_norm_magnitude + 1e-8)  # B, T, H, W, 3

    # Compute ray origins in world coordinates (camera center)
    # Camera center in world coordinates: t
    origin = t[:, :, None, None, :].expand(B, T, resolution[0], resolution[1], 3)  # B, T, H, W, 3

    # Compute Plücker coordinates: moment = origin × direction
    moment = torch.cross(origin, direction_norm, dim=-1)  # B, T, H, W, 3

    # Concatenate normalized direction and moment to get Plücker coordinates
    plucker_coords = torch.cat([direction_norm, moment], dim=-1)  # B, T, H, W, 6

    return plucker_coords


def visualize_action_video(action_video, video, save_path="tmp/action_video.mp4", fps=10, other_videos=None):
    """
    Visualize action image and video.
    Args:
        action_video: Array of shape [3, T, H, W] or [T, H, W, 3]
        video: Array of shape [3, T, H, W] or [T, H, W, 3]
        save_path: str
        fps: int
        other_videos: List[Array] or None
    """
    if isinstance(action_video, torch.Tensor):
        action_video = action_video.cpu().numpy()
    if isinstance(video, torch.Tensor):
        video = video.cpu().numpy()
    if action_video.shape[0] == 3:
        action_video = action_video.transpose(1, 2, 3, 0)
    if video.shape[0] == 3:
        video = video.transpose(1, 2, 3, 0)
    if other_videos is not None:
        comb_video = np.concatenate([action_video, video, video * 0.5 + action_video * 0.5, *other_videos], axis=2)
    else:
        comb_video = np.concatenate([action_video, video, video * 0.5 + action_video * 0.5], axis=2)
    comb_video = (comb_video - comb_video.min()) / (comb_video.max() - comb_video.min())
    comb_video = (comb_video * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Saving video to {save_path}")
    imageio.mimsave(save_path, comb_video, fps=fps)


def check_heatmap_max_on_edge_torch(
    heatmaps: torch.Tensor,  # [T, V, H, W]
    edge_threshold: float = 0.05,
) -> torch.Tensor:
    """
    Check if the maximum value in each heatmap is on the edge of the image.

    Args:
        heatmaps: Heatmaps tensor of shape [T, V, H, W]
        edge_threshold: Percentage of H/W to consider as edge (default: 0.05 = 5%)

    Returns:
        torch.Tensor: Boolean mask of shape [T] where True indicates that at least
                     one view has its heatmap maximum on the edge for that timestep
    """
    T, V, H, W = heatmaps.shape
    device = heatmaps.device

    # Calculate edge margins
    edge_margin_h = int(H * edge_threshold)
    edge_margin_w = int(W * edge_threshold)

    # For each heatmap, find the location of maximum value
    # Flatten spatial dimensions and find argmax
    heatmaps_flat = heatmaps.reshape(T, V, H * W)  # [T, V, H*W]
    max_indices = torch.argmax(heatmaps_flat, dim=2)  # [T, V]

    # Convert flat indices to 2D coordinates
    max_y = max_indices // W  # [T, V]
    max_x = max_indices % W  # [T, V]

    # Check if max position is on the edge for each view
    on_edge_top = max_y < edge_margin_h
    on_edge_bottom = max_y >= (H - edge_margin_h)
    on_edge_left = max_x < edge_margin_w
    on_edge_right = max_x >= (W - edge_margin_w)

    # Combine all edge conditions
    on_edge = on_edge_top | on_edge_bottom | on_edge_left | on_edge_right  # [T, V]

    # Check if ANY view has max on edge for each timestep
    any_view_on_edge = on_edge.any(dim=1)  # [T]

    return any_view_on_edge


def smooth_3d_points_torch(
    points_3d: torch.Tensor,  # [T, 3]
    invalid_mask: torch.Tensor,  # [T], boolean mask where True = invalid
    smoothing_window: int = 3,
) -> torch.Tensor:
    """
    Smooth 3D points for invalid timesteps using interpolation from valid neighbors.

    For invalid timesteps (where invalid_mask is True), interpolate the 3D position
    from nearby valid timesteps using a weighted average.

    Args:
        points_3d: 3D point positions of shape [T, 3]
        invalid_mask: Boolean mask of shape [T] indicating which timesteps are invalid
        smoothing_window: Window size for smoothing (default: 3)

    Returns:
        torch.Tensor: Smoothed 3D points of shape [T, 3]
    """
    T = points_3d.shape[0]
    device = points_3d.device
    dtype = points_3d.dtype

    if T == 0:
        return points_3d

    # Create output tensor (clone to avoid modifying input)
    smoothed_points = points_3d.clone()

    # If all points are valid or all are invalid, return as is
    valid_mask = ~invalid_mask
    num_valid = valid_mask.sum()

    if num_valid == 0:
        # All invalid - can't smooth, return original
        return smoothed_points

    if num_valid == T:
        # All valid - no smoothing needed
        return smoothed_points

    # For each invalid timestep, interpolate from nearest valid neighbors
    for t in range(T):
        if not invalid_mask[t]:
            continue

        # Find nearest valid neighbors
        # Search backward for valid point
        prev_valid_idx = None
        for i in range(t - 1, -1, -1):
            if valid_mask[i]:
                prev_valid_idx = i
                break

        # Search forward for valid point
        next_valid_idx = None
        for i in range(t + 1, T):
            if valid_mask[i]:
                next_valid_idx = i
                break

        # Interpolate based on available neighbors
        if prev_valid_idx is not None and next_valid_idx is not None:
            # Both neighbors available - linear interpolation
            dist_prev = t - prev_valid_idx
            dist_next = next_valid_idx - t
            total_dist = dist_prev + dist_next

            weight_prev = dist_next / total_dist  # Closer = higher weight
            weight_next = dist_prev / total_dist

            smoothed_points[t] = (
                weight_prev * smoothed_points[prev_valid_idx] + weight_next * smoothed_points[next_valid_idx]
            )
        elif prev_valid_idx is not None:
            # Only previous neighbor available
            smoothed_points[t] = smoothed_points[prev_valid_idx]
        elif next_valid_idx is not None:
            # Only next neighbor available
            smoothed_points[t] = smoothed_points[next_valid_idx]
        # If neither neighbor exists, keep original (shouldn't happen if num_valid > 0)

    return smoothed_points


def fuse_multiview_heatmaps_to_3d_point_torch(
    heatmaps: torch.Tensor,  # [..., V, H, W], values in [0,1]
    extrinsics: torch.Tensor,  # [..., V, 3, 4]  (camera-to-world: [R|t])
    intrinsics: torch.Tensor,  # [..., V, 3, 3]
    near: float = 0.1,
    far: float = 5.0,
    num_depth_samples: int = 64,
    edge_threshold: float = 0.01,
    apply_edge_smoothing: bool = True,
) -> torch.Tensor:
    """
    Fuse multi-view heatmaps to estimate 3D point locations via soft volumetric aggregation.

    This function performs multiview triangulation by:
    1. Finding the weighted 2D centroid in each view's heatmap
    2. Casting a ray through each centroid
    3. Sampling 3D points along the principal ray (from first view)
    4. Finding the 3D point that minimizes reprojection error to all views' heatmaps
    5. (Optional) Detecting timesteps where heatmap max is on edge and applying smoothing

    Args:
        heatmaps: Heatmaps tensor of shape [..., V, H, W] with values in [0, 1]
                  Each heatmap represents a probability distribution over image locations
        extrinsics: Camera extrinsics matrices of shape [..., V, 3, 4]
                   Format is [R|t] where transformation is camera-to-world
        intrinsics: Camera intrinsics matrices of shape [..., V, 3, 3]
        near: Near clipping plane distance (in world units)
        far: Far clipping plane distance (in world units)
        num_depth_samples: Number of depth samples along each ray
        edge_threshold: Percentage of image dimensions to consider as edge (default: 0.05 = 5%)
        apply_edge_smoothing: Whether to apply smoothing for edge cases (default: True)

    Returns:
        torch.Tensor: 3D point positions of shape [..., 3]

    Note:
        - The extrinsics format is camera-to-world [R|t], so camera center is at t
        - For world-to-camera format, you'd need to invert the transformation first
        - If apply_edge_smoothing is True, timesteps where any view's heatmap maximum
          is on the image edge (within edge_threshold%) will be smoothed using
          interpolation from neighboring valid timesteps
    """
    # Get dimensions
    *batch_dims, V, H, W = heatmaps.shape
    device = heatmaps.device
    dtype = heatmaps.dtype

    extrinsics = extrinsics.to(device=device, dtype=dtype)
    intrinsics = intrinsics.to(device=device, dtype=dtype)

    # Flatten batch dimensions for easier processing
    batch_size = int(np.prod(batch_dims)) if batch_dims else 1
    heatmaps_flat = heatmaps.reshape(batch_size, V, H, W)
    extrinsics_flat = extrinsics.reshape(batch_size, V, 3, 4)
    intrinsics_flat = intrinsics.reshape(batch_size, V, 3, 3)

    # Step 1: Find weighted 2D centroid for each view's heatmap
    # Create pixel coordinate grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype), torch.arange(W, device=device, dtype=dtype), indexing="ij"
    )  # (H, W)

    # Add 0.5 to sample from pixel centers
    x_coords = x_coords + 0.5  # (H, W)
    y_coords = y_coords + 0.5  # (H, W)

    # Compute weighted centroids for each heatmap
    # Shape: (B, V, H, W)
    heatmap_weights = heatmaps_flat  # (B, V, H, W)

    # Normalize heatmaps to get probability distributions
    heatmap_sum = heatmap_weights.sum(dim=(2, 3), keepdim=True) + 1e-8  # (B, V, 1, 1)
    heatmap_probs = heatmap_weights / heatmap_sum  # (B, V, H, W)

    # Compute 2D centroids (weighted average of pixel coordinates)
    centroid_x = (heatmap_probs * x_coords[None, None, :, :]).sum(dim=(2, 3))  # (B, V)
    centroid_y = (heatmap_probs * y_coords[None, None, :, :]).sum(dim=(2, 3))  # (B, V)

    # Step 2: Unproject centroids to get ray directions for each view
    # Extract rotation and translation from extrinsics
    R = extrinsics_flat[..., :3, :3]  # (B, V, 3, 3) - camera-to-world rotation
    t = extrinsics_flat[..., :3, 3]  # (B, V, 3) - camera center in world coords

    # Extract intrinsic parameters
    fx = intrinsics_flat[..., 0, 0]  # (B, V)
    fy = intrinsics_flat[..., 1, 1]  # (B, V)
    cx = intrinsics_flat[..., 0, 2]  # (B, V)
    cy = intrinsics_flat[..., 1, 2]  # (B, V)

    # Unproject centroids to camera-space ray directions
    x_cam = (centroid_x - cx) / fx  # (B, V)
    y_cam = (centroid_y - cy) / fy  # (B, V)
    z_cam = torch.ones_like(x_cam)  # (B, V)

    # Stack to get ray directions in camera space
    ray_dirs_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (B, V, 3)

    # Transform ray directions to world space
    ray_dirs_world = torch.einsum("bvij,bvj->bvi", R, ray_dirs_cam)  # (B, V, 3)

    # Normalize ray directions
    ray_dirs_world = ray_dirs_world / (torch.linalg.norm(ray_dirs_world, dim=-1, keepdim=True) + 1e-8)

    # Step 3: Sample 3D points along the rays and find best triangulation
    # Use the first view as the principal ray for sampling
    principal_ray_origin = t[:, 0, :]  # (B, 3)
    principal_ray_dir = ray_dirs_world[:, 0, :]  # (B, 3)

    # Sample depths along the principal ray.
    # FORK: `near`/`far` may be per-sample tensors, not just scalars. A scalar keeps the
    # original behaviour bit-for-bit; a tensor lets the caller search a narrow band that
    # depends on the frame, which is what `fuse_multiview_heatmaps_to_pose_torch` needs to
    # keep an axis point from running to the edge of a global sweep (see `constrain_axis_depth`).
    if torch.is_tensor(near) or torch.is_tensor(far):
        near_t = torch.as_tensor(near, device=device, dtype=dtype).reshape(-1)
        far_t = torch.as_tensor(far, device=device, dtype=dtype).reshape(-1)
        near_t = near_t.expand(batch_size) if near_t.numel() == 1 else near_t
        far_t = far_t.expand(batch_size) if far_t.numel() == 1 else far_t
        alpha = torch.linspace(0.0, 1.0, num_depth_samples, device=device, dtype=dtype)  # (D,)
        depth_samples = near_t[:, None] + (far_t - near_t)[:, None] * alpha[None, :]  # (B, D)
        candidate_points = (
            principal_ray_origin[:, None, :] + principal_ray_dir[:, None, :] * depth_samples[:, :, None]
        )  # (B, D, 3)
    else:
        depth_samples = torch.linspace(near, far, num_depth_samples, device=device, dtype=dtype)  # (D,)

        # Generate 3D candidate points along the principal ray
        # Shape: (B, 1, 3) + (B, 1, 3) * (D, 1) -> (B, D, 3)
        candidate_points = (
            principal_ray_origin[:, None, :] + principal_ray_dir[:, None, :] * depth_samples[None, :, None]
        )  # (B, D, 3)

    # Step 4: Score each candidate point by projecting to all views
    # For each candidate point, compute its reprojection score in all views

    # Convert world-to-camera transformation
    R_inv = R.transpose(-2, -1)  # (B, V, 3, 3)
    t_inv = -torch.einsum("bvij,bvj->bvi", R_inv, t)  # (B, V, 3)

    # Project candidate points to all views
    # candidate_points: (B, D, 3) -> (B, 1, D, 3)
    # Need to transform to camera coordinates for each view
    points_expanded = candidate_points[:, None, :, :]  # (B, 1, D, 3)

    # Transform to camera coordinates: R_inv @ (point - t)
    # Broadcast: (B, V, 1, 3, 3) @ ((B, 1, D, 3) - (B, V, 1, 3))
    points_cam = torch.einsum(
        "bvij,bvdj->bvdi", R_inv, points_expanded.expand(batch_size, V, num_depth_samples, 3) - t[:, :, None, :]
    )  # (B, V, D, 3)

    # Project to image plane using intrinsics
    fx_exp = fx[:, :, None]  # (B, V, 1)
    fy_exp = fy[:, :, None]  # (B, V, 1)
    cx_exp = cx[:, :, None]  # (B, V, 1)
    cy_exp = cy[:, :, None]  # (B, V, 1)

    # Perspective projection
    z_cam = points_cam[..., 2]  # (B, V, D)
    x_proj = fx_exp * (points_cam[..., 0] / (z_cam + 1e-8)) + cx_exp  # (B, V, D)
    y_proj = fy_exp * (points_cam[..., 1] / (z_cam + 1e-8)) + cy_exp  # (B, V, D)

    # Clamp projections to valid image coordinates
    x_proj = torch.clamp(x_proj, 0, W - 1)
    y_proj = torch.clamp(y_proj, 0, H - 1)

    # Sample heatmap values at projected locations using bilinear interpolation
    # Normalize coordinates to [-1, 1] for grid_sample
    x_norm = 2.0 * x_proj / (W - 1) - 1.0  # (B, V, D)
    y_norm = 2.0 * y_proj / (H - 1) - 1.0  # (B, V, D)

    # grid_sample expects (B, C, H, W) input and (B, H_out, W_out, 2) grid
    # Reshape for grid_sample: (B*V, 1, H, W)
    heatmaps_for_sample = heatmaps_flat.reshape(batch_size * V, 1, H, W)

    # Create sampling grid: (B*V, D, 1, 2)
    grid = torch.stack(
        [x_norm.reshape(batch_size * V, num_depth_samples), y_norm.reshape(batch_size * V, num_depth_samples)], dim=-1
    )
    grid = grid[:, :, None, :]  # (B*V, D, 1, 2)

    # Sample heatmap values
    sampled_scores = torch.nn.functional.grid_sample(
        heatmaps_for_sample, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )  # (B*V, 1, D, 1)

    # Reshape back: (B, V, D)
    sampled_scores = sampled_scores.reshape(batch_size, V, num_depth_samples)

    # Aggregate scores across views (multiply or sum in log space)
    # Using product (geometric mean) for multi-view consistency
    aggregated_scores = sampled_scores.prod(dim=1)  # (B, D)

    # Find the depth with maximum score
    best_depth_idx = aggregated_scores.argmax(dim=1)  # (B,)

    # Extract the 3D point at the best depth
    batch_indices = torch.arange(batch_size, device=device)
    points_3d = candidate_points[batch_indices, best_depth_idx]  # (B, 3)

    # Step 5: Apply edge detection and smoothing if enabled
    if apply_edge_smoothing and batch_dims:
        # Only apply if we have a temporal dimension (batch_dims should be [T] or similar)
        # Check if heatmaps are in temporal format [T, V, H, W]
        if len(batch_dims) == 1:
            T = batch_dims[0]
            # Reshape heatmaps back to [T, V, H, W]
            heatmaps_temporal = heatmaps_flat.reshape(T, V, H, W)

            # Check which timesteps have heatmap max on edge
            invalid_mask = check_heatmap_max_on_edge_torch(heatmaps_temporal, edge_threshold=edge_threshold)  # [T]

            # If any timesteps are invalid, apply smoothing
            if invalid_mask.any():
                points_3d = smooth_3d_points_torch(points_3d, invalid_mask, smoothing_window=3)  # [T, 3]  # [T]

    # Reshape back to original batch dimensions
    if batch_dims:
        points_3d = points_3d.reshape(*batch_dims, 3)

    return points_3d


def fuse_multiview_heatmaps_to_6d_point_torch(
    start_heatmaps: torch.Tensor,  # [..., V, H, W]
    end_heatmaps: torch.Tensor,  # [..., V, H, W]
    extrinsics: torch.Tensor,  # [..., V, 3, 4]  (camera-to-world: [R|t])
    intrinsics: torch.Tensor,  # [..., V, 3, 3]
    near: float = 0.1,
    far: float = 5.0,
    num_depth_samples: int = 64,
    edge_threshold: float = 0.01,
    apply_edge_smoothing: bool = True,
) -> torch.Tensor:
    """
    Fuse multi-view heatmaps for a 7D action represented as a 6D spatial state.

    Representation:
        - We model the 7D action [x, y, z, euler_x, euler_y, euler_z, openness] via
          two 3D points in world coordinates:
              * start: position [x, y, z]
              * end:   position + tool-forward direction * length
        - This function recovers a 6D state [x, y, z, dir_x, dir_y, dir_z]
          from two sets of heatmaps (start/end), where dir is a unit vector:
              dir = normalize(end - start)

    Args:
        start_heatmaps: Heatmaps tensor for the start point of shape [..., V, H, W]
        end_heatmaps:   Heatmaps tensor for the end   point of shape [..., V, H, W]
        extrinsics:     Camera extrinsics matrices of shape [..., V, 3, 4]
                        (camera-to-world: [R|t])
        intrinsics:     Camera intrinsics matrices of shape [..., V, 3, 3]
        near, far, num_depth_samples, edge_threshold, apply_edge_smoothing:
            Same semantics as in `fuse_multiview_heatmaps_to_3d_point_torch`.

    Returns:
        torch.Tensor: 6D state tensor of shape [..., 6]
                      [x, y, z, dir_x, dir_y, dir_z]
    """
    # Recover start/end 3D points using the existing 3D fusion routine
    start_points_3d = fuse_multiview_heatmaps_to_3d_point_torch(
        start_heatmaps,
        extrinsics,
        intrinsics,
        near=near,
        far=far,
        num_depth_samples=num_depth_samples,
        edge_threshold=edge_threshold,
        apply_edge_smoothing=apply_edge_smoothing,
    )  # [..., 3]

    end_points_3d = fuse_multiview_heatmaps_to_3d_point_torch(
        end_heatmaps,
        extrinsics,
        intrinsics,
        near=near,
        far=far,
        num_depth_samples=num_depth_samples,
        edge_threshold=edge_threshold,
        apply_edge_smoothing=apply_edge_smoothing,
    )  # [..., 3]

    # Direction is the normalized vector from start to end
    dir_vec = end_points_3d - start_points_3d  # [..., 3]
    dir_norm = torch.linalg.norm(dir_vec, dim=-1, keepdim=True)  # [..., 1]
    dir_unit = dir_vec / (dir_norm + 1e-8)  # [..., 3]

    # Concatenate position and direction to form 6D state
    state_6d = torch.cat([start_points_3d, dir_unit], dim=-1)  # [..., 6]

    return state_6d


def fuse_multiview_heatmaps_to_7d_point_torch(
    heatmaps_rgb: torch.Tensor,  # [..., V, H, W, 3], values typically in [0, 255]
    extrinsics: torch.Tensor,  # [..., V, 3, 4]  (camera-to-world: [R|t])
    intrinsics: torch.Tensor,  # [..., V, 3, 3]
    near: float = 0.1,
    far: float = 5.0,
    num_depth_samples: int = 64,
    edge_threshold: float = 0.01,
    apply_edge_smoothing: bool = True,
    grip_close_threshold: float = 128.0,
) -> torch.Tensor:
    """
    Fuse multi-view RGB heatmaps into a full 7D action:
        [x, y, z, dir_x, dir_y, dir_z, gripper]

    UPSTREAM FUNCTION -- DELIBERATELY LEFT AS-IS. It is the official baseline path and is
    still what `inference.py:export_action_point_cloud_from_pred_video` calls (that consumer
    only uses `[:3]`, so neither defect below affects it). Two known defects mean it must NOT
    be used to produce executable poses or a gripper metric:

      1. Only the R and G channels are triangulated, so `[3:6]` is a single unit direction
         vector and the roll about it is undetermined -- not enough for IK.
      2. `any(B > grip_close_threshold)` is degenerate: the blue channel also carries the
         up-point Gaussian blob, whose peak is 255, so ~697 pixels exceed 128 on EVERY frame
         regardless of openness. The bit is constant "closed" (measured accuracy 0.500).
         Openness actually lives in the low-response BACKGROUND of that channel.

    Use `fuse_multiview_heatmaps_to_pose_torch` instead, which implements the paper's full
    decoder (Sec. 3.2) and returns [x, y, z, qx, qy, qz, qw, openness].
    `tests/test_decode_6dof.py` pins both defects as regression tests.

    Encoding convention for the input heatmap video:
        - R channel: start-point heatmap (position)
        - G channel: end-point heatmap (position + direction * length)
        - B channel: gripper state; if any pixel > `grip_close_threshold`
                     for any view at a timestep, we set action[6] = 1 (closed),
                     otherwise 0 (open).

    Args:
        heatmaps_rgb: Tensor of shape [..., V, H, W, 3] with values in [0, 255]
                      (uint8 or float).
        extrinsics:   Camera extrinsics matrices of shape [..., V, 3, 4]
                      (camera-to-world: [R|t]).
        intrinsics:   Camera intrinsics matrices of shape [..., V, 3, 3].
        near, far, num_depth_samples, edge_threshold, apply_edge_smoothing:
            Same semantics as in `fuse_multiview_heatmaps_to_3d_point_torch`.
        grip_close_threshold:
            Threshold on the blue channel (0-255 scale) to treat the gripper as closed.

    Returns:
        torch.Tensor of shape [..., 7]:
            [x, y, z, dir_x, dir_y, dir_z, gripper]
    """
    # Ensure floating point for processing
    heatmaps_rgb = heatmaps_rgb.to(torch.float32)

    # Split channels: R=start, G=end, B=gripper
    start_heatmaps = heatmaps_rgb[..., 0] / 255.0  # [..., V, H, W]
    end_heatmaps = heatmaps_rgb[..., 1] / 255.0  # [..., V, H, W]
    gripper_channel = heatmaps_rgb[..., 2]  # [..., V, H, W], still in [0, 255] scale

    # 1) Recover 6D state [x, y, z, dir_x, dir_y, dir_z] from R/G heatmaps
    state_6d = fuse_multiview_heatmaps_to_6d_point_torch(
        start_heatmaps,
        end_heatmaps,
        extrinsics,
        intrinsics,
        near=near,
        far=far,
        num_depth_samples=num_depth_samples,
        edge_threshold=edge_threshold,
        apply_edge_smoothing=apply_edge_smoothing,
    )  # [..., 6]

    # 2) Infer gripper open/close from B channel
    #    If any pixel > grip_close_threshold in any view, mark as closed (1), else open (0)
    *batch_dims, V, H, W = start_heatmaps.shape
    gripper_flat = gripper_channel.reshape(*batch_dims, V, H * W)  # [..., V, HW]
    closed_any_pixel = gripper_flat > grip_close_threshold  # [..., V, HW]
    closed_any_view = closed_any_pixel.any(dim=-1).any(dim=-1)  # [...], bool

    gripper_state = closed_any_view.to(state_6d.dtype)[..., None]  # [..., 1]

    # 3) Concatenate to form 7D action
    action_7d = torch.cat([state_6d, gripper_state], dim=-1)  # [..., 7]

    return action_7d


def decode_gripper_openness_torch(
    blue_channel: torch.Tensor,  # [..., V, H, W], values in [0, 255]
    mode: str = "paper",
    low_response_threshold: float = 0.25,
) -> torch.Tensor:
    """
    Decode gripper openness from the blue channel background level.

    FORK: new. The encoder (`project_action_5d_to_rgb_torch`) writes openness into the
    LOW-RESPONSE background of the blue channel:

        B = gaussian(up_point)               # peak 1.0
        B[B <= 0.25] = openness * 0.25       # pedestal at 0.25 (open) or 0.0 (closed)

    so recovering openness means estimating that pedestal level and dividing by 0.25.
    This is paper Sec. 3.2 Eq. (7). The shipped `fuse_multiview_heatmaps_to_7d_point_torch`
    does something else entirely (`any(B > 128)`), which is degenerate -- see the note in
    that function.

    Args:
        blue_channel: Blue channel of the action image, shape [..., V, H, W], scale [0, 255].
        mode:
            "paper"  -- Eq. (7): mean of the low-response pixels, divided by 0.25.
                        NOTE the boundary convention: the paper writes the selection set as
                        {A < 0.25} (strict), but the encoder fills the pedestal using a
                        `B <= 0.25` mask and writes exactly 0.25 on an OPEN frame. With a
                        strict `<` the selected set is therefore EMPTY on every open frame
                        (measured: 0 pixels vs 128217 for `<=`), and the estimate collapses
                        to 0 for all frames. We select with `<=`, which is the encoder's own
                        mask and makes the formula exact on clean renders (1.0000 / 0.0000).
            "median" -- same estimand, robust estimator: median over all pixels / 0.25.
                        The background occupies ~98% of the frame so the median lands on the
                        pedestal. Prefer this on GENERATED images: the threshold-selected
                        mean is biased low once the pedestal is blurred/noised (measured
                        ghat_open 1.00 -> 0.94 -> 0.84 -> 0.68 -> 0.44 as noise grows, and it
                        flips the decision at the last step), while the median stays at
                        1.00-1.01 throughout.
        low_response_threshold: The 0.25 constant from the encoder. Do not change unless
            `project_action_5d_to_rgb_torch` changes with it.

    Returns:
        torch.Tensor of shape [...]: continuous openness estimate, nominally in [0, 1].
        Threshold at 0.5 for a binary open/closed decision.
    """
    b = blue_channel.to(torch.float32) / 255.0
    *batch_dims, V, H, W = b.shape
    flat = b.reshape(*batch_dims, V * H * W) if batch_dims else b.reshape(1, V * H * W)

    if mode == "median":
        est = flat.median(dim=-1).values
    elif mode == "paper":
        low = flat <= low_response_threshold
        count = low.sum(dim=-1)
        est = (flat * low).sum(dim=-1) / count.clamp(min=1)
        # If no pixel qualifies the formula is undefined; the encoder guarantees a pedestal
        # exists, so this only fires on badly corrupted input. Fall back to the median rather
        # than silently reporting "closed".
        est = torch.where(count > 0, est, flat.median(dim=-1).values)
    else:
        raise ValueError(f"unknown gripper decode mode {mode!r}, expected 'paper' or 'median'")

    openness = est / low_response_threshold
    return openness.reshape(*batch_dims) if batch_dims else openness.reshape(())


def rotation_matrix_to_quaternion_xyzw(rot: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to quaternions in (x, y, z, w) order.

    FORK: new. Kept in torch (no scipy) so the decoder stays differentiable and device-
    agnostic. The (x, y, z, w) order matches PyRep/RLBench `gripper_pose[3:7]` and
    `scipy.spatial.transform.Rotation.from_quat`, i.e. the same convention
    `RLBenchMVDataset.get_7d_action` already assumes.

    Args:
        rot: Rotation matrices of shape [..., 3, 3].

    Returns:
        torch.Tensor of shape [..., 4] -- (qx, qy, qz, qw), with qw >= 0.
    """
    m = rot.reshape(-1, 3, 3)
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = torch.zeros(m.shape[0], 4, dtype=m.dtype, device=m.device)

    # Branch on the largest diagonal term for numerical stability (Shepperd's method).
    t0 = trace > 0
    t1 = (~t0) & (m[:, 0, 0] >= m[:, 1, 1]) & (m[:, 0, 0] >= m[:, 2, 2])
    t2 = (~t0) & (~t1) & (m[:, 1, 1] >= m[:, 2, 2])
    t3 = ~(t0 | t1 | t2)

    def _fill(mask, s, qx, qy, qz, qw):
        if mask.any():
            q[mask, 0] = qx / s
            q[mask, 1] = qy / s
            q[mask, 2] = qz / s
            q[mask, 3] = qw / s

    if t0.any():
        s = torch.sqrt(trace[t0] + 1.0) * 2.0
        mm = m[t0]
        _fill(t0, s, mm[:, 2, 1] - mm[:, 1, 2], mm[:, 0, 2] - mm[:, 2, 0], mm[:, 1, 0] - mm[:, 0, 1], 0.25 * s * s)
    if t1.any():
        mm = m[t1]
        s = torch.sqrt(1.0 + mm[:, 0, 0] - mm[:, 1, 1] - mm[:, 2, 2]) * 2.0
        _fill(t1, s, 0.25 * s * s, mm[:, 0, 1] + mm[:, 1, 0], mm[:, 0, 2] + mm[:, 2, 0], mm[:, 2, 1] - mm[:, 1, 2])
    if t2.any():
        mm = m[t2]
        s = torch.sqrt(1.0 + mm[:, 1, 1] - mm[:, 0, 0] - mm[:, 2, 2]) * 2.0
        _fill(t2, s, mm[:, 0, 1] + mm[:, 1, 0], 0.25 * s * s, mm[:, 1, 2] + mm[:, 2, 1], mm[:, 0, 2] - mm[:, 2, 0])
    if t3.any():
        mm = m[t3]
        s = torch.sqrt(1.0 + mm[:, 2, 2] - mm[:, 0, 0] - mm[:, 1, 1]) * 2.0
        _fill(t3, s, mm[:, 0, 2] + mm[:, 2, 0], mm[:, 1, 2] + mm[:, 2, 1], 0.25 * s * s, mm[:, 1, 0] - mm[:, 0, 1])

    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    q = torch.where(q[:, 3:4] < 0, -q, q)  # canonical hemisphere
    return q.reshape(*rot.shape[:-2], 4)


def _axis_ray_from_main_view(
    heatmaps: torch.Tensor,  # [..., V, H, W], values in [0, 1]
    extrinsics: torch.Tensor,  # [..., V, 3, 4]
    intrinsics: torch.Tensor,  # [..., V, 3, 3]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The main view's ray through a heatmap's weighted centroid: `(origin, unit_direction)`.

    FORK: new. Mirrors steps 1-2 of `fuse_multiview_heatmaps_to_3d_point_torch` exactly, so the
    ray is the same one that routine casts before it searches for a depth.
    """
    *batch, V, H, W = heatmaps.shape
    device, dtype = heatmaps.device, heatmaps.dtype
    n = int(np.prod(batch)) if batch else 1
    hm = heatmaps.reshape(n, V, H, W)
    ext = extrinsics.to(device=device, dtype=dtype).reshape(n, V, 3, 4)
    itr = intrinsics.to(device=device, dtype=dtype).reshape(n, V, 3, 3)

    ys, xs = torch.meshgrid(torch.arange(H, device=device, dtype=dtype),
                            torch.arange(W, device=device, dtype=dtype), indexing="ij")
    probs = hm / (hm.sum(dim=(2, 3), keepdim=True) + 1e-8)
    cx_pix = (probs * (xs + 0.5)[None, None]).sum(dim=(2, 3))[:, 0]
    cy_pix = (probs * (ys + 0.5)[None, None]).sum(dim=(2, 3))[:, 0]

    fx, fy = itr[:, 0, 0, 0], itr[:, 0, 1, 1]
    cx, cy = itr[:, 0, 0, 2], itr[:, 0, 1, 2]
    d_cam = torch.stack([(cx_pix - cx) / fx, (cy_pix - cy) / fy, torch.ones_like(cx_pix)], dim=-1)
    d_world = torch.einsum("bij,bj->bi", ext[:, 0, :3, :3], d_cam)
    d_world = d_world / (d_world.norm(dim=-1, keepdim=True) + 1e-8)
    origin = ext[:, 0, :3, 3]
    shape = (*batch, 3) if batch else (3,)
    return origin.reshape(shape), d_world.reshape(shape)


def _snap_axis_point_to_length(
    q_axis: torch.Tensor,   # [..., 3] unconstrained triangulation
    q_pos: torch.Tensor,    # [..., 3] decoded position point
    origin: torch.Tensor,   # [..., 3] main-view camera centre
    direction: torch.Tensor,  # [..., 3] unit ray through the axis blob's centroid
    length: float,
    tolerance: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Re-solve an axis point's depth using the length the ENCODER guaranteed.

    `project_actions_7d_to_5d_torch_batch` places both axis points at exactly `length` metres
    from the position point, but the decoder searches depth freely and can settle at the wrong
    point along the ray. When it does, the direction is badly wrong while the heatmaps still
    look perfectly healthy: measured on ground-truth renders, frames whose decoded
    `|q_pos - q_axis|` was more than 3 cm off 0.1 m had a median axis error of 93.4 deg, versus
    0.3 deg on the rest -- and those are exactly the frames that made closed-loop rollouts
    unexecutable, since a ~180 deg rotation target cannot be reached by Jacobian IK.

    The fix intersects the ray with the sphere of radius `length` centred at `q_pos`. Of the (at
    most two) intersections we keep the one nearer the unconstrained estimate, which preserves
    the triangulation's choice of branch while enforcing the known scale. Rays that miss the
    sphere entirely are left alone and reported, so the caller can flag those frames instead of
    trusting a fabricated point.

    Returns `(q_corrected, ok)` where `ok` marks entries whose length is now trustworthy.
    """
    oc = origin - q_pos
    b = 2.0 * (direction * oc).sum(-1)
    c = (oc * oc).sum(-1) - length * length
    disc = b * b - 4.0 * c
    hit = disc >= 0
    sq = torch.sqrt(disc.clamp(min=0))
    t1, t2 = (-b - sq) / 2.0, (-b + sq) / 2.0
    p1 = origin + t1[..., None] * direction
    p2 = origin + t2[..., None] * direction
    # Keep the branch the free search already preferred.
    pick2 = (p2 - q_axis).norm(dim=-1) < (p1 - q_axis).norm(dim=-1)
    snapped = torch.where(pick2[..., None], p2, p1)
    # A point in front of the camera only; a negative depth is not a real solution.
    hit = hit & (torch.where(pick2, t2, t1) > 0)
    corrected = torch.where(hit[..., None], snapped, q_axis)
    already_ok = ((q_axis - q_pos).norm(dim=-1) - length).abs() <= tolerance
    return corrected, hit | already_ok


def _fibonacci_sphere(n: int, device, dtype) -> torch.Tensor:
    """`n` roughly-uniform unit vectors. Deterministic, no RNG."""
    i = torch.arange(n, device=device, dtype=dtype) + 0.5
    phi = torch.arccos(1.0 - 2.0 * i / n)
    theta = math.pi * (1.0 + 5.0 ** 0.5) * i
    return torch.stack([torch.sin(phi) * torch.cos(theta),
                        torch.sin(phi) * torch.sin(theta),
                        torch.cos(phi)], dim=-1)


def solve_axis_direction_on_sphere(
    heatmaps: torch.Tensor,      # [..., V, H, W] in [0, 1]
    q_pos: torch.Tensor,         # [..., 3] the already-decoded position point
    extrinsics: torch.Tensor,    # [..., V, 3, 4] camera-to-world
    intrinsics: torch.Tensor,    # [..., V, 3, 3]
    length: float = 0.1,
    num_directions: int = 1024,
    refine: bool = False,
    refine_halfwidth: float = 0.14,
    refine_steps: int = 17,
) -> torch.Tensor:
    """Recover an axis point by searching DIRECTIONS on a sphere, not depths along a ray.

    ⚠️ THIS DEVIATES FROM THE PAPER. Action-Images Sec. 3.2 specifies ray marching, and says so
    for every semantic point, not just the position:

        "we cast a ray from the main-view camera center through u_hat^(1), and sample a set of
         candidate 3D points along the ray between a near plane and a far plane. Each candidate
         is then projected into the side view, where it is scored against the corresponding
         side-view heatmap... In practice, this procedure is repeated for each semantic point
         heatmap in the action image."

    The paper's Discussion attributes the residual decoding error to "the sampling interval
    along the ray" and "the spatial resolution of the heatmaps" -- i.e. to discretisation. It
    does not mention the failure this function exists to avoid, in which the ray
    parameterisation is not merely coarse but DEGENERATE (see below). Both solvers are kept:
    pass `axis_solver="ray"` for the paper's method. Any number compared against the paper's
    Table 4 (3DErr = 12.2 mm) should be produced with `axis_solver="ray"`, since that is the
    algorithm those numbers came from.

    FORK: new. The ray-marching decoder casts a ray through the axis blob's centroid in the
    MAIN view and searches depth. That is ill-posed exactly when the axis is foreshortened in
    that view -- the axis then projects on top of the position blob, every candidate along the
    ray lands on the same pixels, the score is flat, and the argmax slides to whichever end of
    the sweep it started from. Measured on a real open_drawer window this put frames 0-1 at
    depth 0.6000 (the near plane), an axis length of 0.532 m instead of 0.100 m, and ~95 deg of
    axis error -- the "gripper points at the sky" rollouts. Constraining the depth band helps
    (95 -> 87 deg) but cannot fix it, because the degeneracy is in the parameterisation.

    This searches the actual unknown instead. The encoder guarantees the axis point lies on the
    sphere of radius `length` about the position point, so the only free parameter is a
    direction: 2 DOF, bounded, and both views score it symmetrically. Foreshortening in one
    view no longer collapses the search, because the other view still separates the candidates.

    Returns the axis POINT `[..., 3]` (`q_pos + length * best_direction`).
    """
    *batch, V, H, W = heatmaps.shape
    device, dtype = heatmaps.device, heatmaps.dtype
    n = int(np.prod(batch)) if batch else 1
    hm = heatmaps.reshape(n, V, H, W)
    ext = extrinsics.to(device=device, dtype=dtype).reshape(n, V, 3, 4)
    itr = intrinsics.to(device=device, dtype=dtype).reshape(n, V, 3, 3)
    pos = q_pos.reshape(n, 3)

    dirs = _fibonacci_sphere(num_directions, device, dtype)                 # (D, 3)
    cand = pos[:, None, :] + length * dirs[None, :, :]                      # (n, D, 3)

    R = ext[..., :3, :3]                                                    # (n, V, 3, 3)
    t = ext[..., :3, 3]                                                     # (n, V, 3)
    R_inv = R.transpose(-2, -1)
    pts_cam = torch.einsum("nvij,nvdj->nvdi", R_inv,
                           cand[:, None, :, :].expand(n, V, num_directions, 3) - t[:, :, None, :])
    z = pts_cam[..., 2]
    eps = 1e-6
    u = itr[:, :, 0, 0][..., None] * pts_cam[..., 0] / (z + eps) + itr[:, :, 0, 2][..., None]
    v = itr[:, :, 1, 1][..., None] * pts_cam[..., 1] / (z + eps) + itr[:, :, 1, 2][..., None]

    # grid_sample expects normalised coords; align_corners=True matches the ray-marching path.
    gx = (u / max(W - 1, 1)) * 2 - 1
    gy = (v / max(H - 1, 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).reshape(n * V, num_directions, 1, 2)
    scores = torch.nn.functional.grid_sample(
        hm.reshape(n * V, 1, H, W), grid, mode="bilinear",
        padding_mode="zeros", align_corners=True,
    ).reshape(n, V, num_directions)

    # Product across views, as the ray-marching decoder does: a candidate must be supported by
    # EVERY view, so a point that only one camera likes cannot win.
    agg = scores.prod(dim=1)                                                # (n, D)
    agg = torch.where(z.min(dim=1).values > 0, agg, torch.full_like(agg, -1.0))  # in front of both
    best = agg.argmax(dim=1)                                                # (n,)
    d0 = dirs[best]                                                         # (n, 3)

    # A uniform sphere of D points has ~sqrt(4*pi/D) rad between neighbours: 1024 directions
    # quantise to ~6 deg, which showed up as the median axis error rising from 0.55 to 3.39 deg
    # even while the degenerate frames were fixed. One local refinement pass removes that floor
    # at ~1.3x the cost, instead of the 16x a globally finer sphere would need.
    if refine:
        # Orthonormal basis around the coarse winner.
        tmp = torch.zeros_like(d0)
        tmp[..., 0] = 1.0
        alt = torch.zeros_like(d0)
        alt[..., 1] = 1.0
        seed = torch.where((d0[..., 0].abs() > 0.9)[..., None], alt, tmp)
        e1 = torch.cross(d0, seed, dim=-1)
        e1 = e1 / e1.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        e2 = torch.cross(d0, e1, dim=-1)
        g = torch.linspace(-refine_halfwidth, refine_halfwidth, refine_steps, device=device, dtype=dtype)
        aa, bb = torch.meshgrid(g, g, indexing="ij")
        offs = torch.stack([aa.reshape(-1), bb.reshape(-1)], dim=-1)        # (M, 2)
        M = offs.shape[0]
        fine = (d0[:, None, :] + offs[None, :, 0:1] * e1[:, None, :]
                + offs[None, :, 1:2] * e2[:, None, :])                      # (n, M, 3)
        fine = fine / fine.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cand_f = pos[:, None, :] + length * fine
        pts_cam_f = torch.einsum("nvij,nvdj->nvdi", R_inv,
                                 cand_f[:, None, :, :].expand(n, V, M, 3) - t[:, :, None, :])
        zf = pts_cam_f[..., 2]
        uf = itr[:, :, 0, 0][..., None] * pts_cam_f[..., 0] / (zf + eps) + itr[:, :, 0, 2][..., None]
        vf = itr[:, :, 1, 1][..., None] * pts_cam_f[..., 1] / (zf + eps) + itr[:, :, 1, 2][..., None]
        grid_f = torch.stack([(uf / max(W - 1, 1)) * 2 - 1,
                              (vf / max(H - 1, 1)) * 2 - 1], dim=-1).reshape(n * V, M, 1, 2)
        sf = torch.nn.functional.grid_sample(
            hm.reshape(n * V, 1, H, W), grid_f, mode="bilinear",
            padding_mode="zeros", align_corners=True).reshape(n, V, M)
        agg_f = sf.prod(dim=1)
        agg_f = torch.where(zf.min(dim=1).values > 0, agg_f, torch.full_like(agg_f, -1.0))
        cand = cand_f
        best = agg_f.argmax(dim=1)

    out = cand[torch.arange(n, device=device), best]                        # (n, 3)
    return out.reshape(*batch, 3) if batch else out.reshape(3)


def fuse_multiview_heatmaps_to_pose_torch(
    heatmaps_rgb: torch.Tensor,  # [..., V, H, W, 3], values in [0, 255]
    extrinsics: torch.Tensor,  # [..., V, 3, 4]  (camera-to-world: [R|t])
    intrinsics: torch.Tensor,  # [..., V, 3, 3]
    near: float = 0.1,
    far: float = 5.0,
    num_depth_samples: int = 64,
    edge_threshold: float = 0.01,
    apply_edge_smoothing: bool = True,
    gripper_mode: str = "paper",
    strip_openness_pedestal: bool = True,
    return_matrix: bool = False,
    # OFF by default. The length constraint DETECTS bad frames reliably (see
    # `return_confidence`) but does not reliably repair them: on a ground-truth open_drawer
    # window it improved the median rotation error 0.55 -> 0.18 deg yet left the two genuinely
    # degenerate frames at ~95 deg (their ray misses the sphere entirely, so they are left
    # alone) and pushed two good frames from ~1.5 to ~9 deg, for a slightly worse mean
    # (5.13 -> 5.32). Enable only if you have re-measured it on your data.
    constrain_axis_length: Optional[float] = None,
    axis_length_tolerance: float = 0.02,
    return_confidence: bool = False,
    # ON by default: this is the fix for the boundary artefact described below, it is cheap
    # (one extra triangulation over a narrower band), and it cannot make a well-conditioned
    # frame worse -- the true axis point is inside the band by construction.
    constrain_axis_depth: bool = True,
    axis_length: float = 0.1,
    axis_depth_band: float = 1.3,
    # "ray"    = the PAPER's method (Sec. 3.2): march depth along the main view's ray.
    # "sphere" = a FORK DEVIATION: search the axis direction on the sphere of known radius.
    #
    # Sphere is the default because the ray parameterisation is not merely coarse but
    # DEGENERATE when the axis is foreshortened in the main view -- it decodes a ~95 deg-wrong
    # axis, i.e. an IK target the arm cannot reach. The paper does not describe that failure.
    # ⚠️ Every reported number must state which solver produced it, and anything compared with
    # the paper's Table 4 must use "ray". See `solve_axis_direction_on_sphere`.
    axis_solver: str = "sphere",
    # 16384 directions, no local refinement. Chosen on TWO criteria, because optimising only
    # the first one produces a decoder that wiggles:
    #
    #   (a) worst-case axis error -- a 95 deg target is unreachable by IK, a 11 deg one is not;
    #   (b) FRAME-TO-FRAME jitter -- an error that is independent per frame is executed by the
    #       robot as shaking, even when its median is small. Ground truth at frame_interval 3
    #       turns 1.70 deg between consecutive frames, so anything the decoder adds on top of
    #       that is visible wiggle.
    #
    # Measured on a real open_drawer window at 256^2 (max deg / median deg / per-frame turn):
    #   ray (upstream)   95.4 / 0.55 / 1.81   <- clean, but the "sky" failure on 2 of 41 frames
    #   sphere  4096     10.2 / 2.82 / 4.08   <- fixes the tail, ADDS 2.4 deg/frame of wiggle
    #   sphere  8192     12.3 / 2.72 / 2.12
    #   sphere 16384     11.2 / 1.74 / 1.83   <- tail fixed AND jitter back to the ray level
    #   sphere 32768     11.5 / 1.26 / 1.80   <- marginally better, 2x the cost
    num_axis_directions: int = 16384,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    Decode a full executable 6-DoF pose + gripper from multi-view action images.

    FORK: new. This is the paper's decoder (Sec. 3.2, p. 8) implemented in full. The shipped
    `fuse_multiview_heatmaps_to_7d_point_torch` implements only part of it -- it triangulates
    the red and green channels and then collapses the blue channel to a boolean threshold, so
    its output is [pos, single_axis, constant] and the roll about the forward axis is left
    undetermined. That is not enough to command `EndEffectorPoseViaIK`/`ViaPlanning`.
    Nothing about the REPRESENTATION is lossy: `project_actions_7d_to_5d_torch_batch` writes
    three 3D points, i.e. two independent axes, which determine SO(3) exactly.

    Encoding convention (mirror of `project_actions_7d_to_5d_torch_batch`):
        R channel: position point           q_pos    = p
        G channel: paper's "up" point       q_up     = p + L * R(theta) @ (+x)
        B channel: paper's "normal" point   q_normal = p + L * R(theta) @ (-z)
                   plus the openness pedestal in the low-response background
    (The code's local variable names for the G/B points are swapped relative to the paper;
    only the naming differs, the geometry is identical.)

    Decoding, following the paper verbatim:
        p_hat  = q_pos
        e_x    = norm(q_up  - q_pos)
        e_z    = norm(q_pos - q_normal)
        e_y    = norm(e_z x e_x)
        e_z'   = e_x x e_y                  # re-orthogonalize; the paper's e_x and e_z are
                                            # only approximately perpendicular once decoded
                                            # (measured 89.94 deg), and a valid rotation
                                            # matrix needs them exactly so.
        R_hat  = [e_x | e_y | e_z']         # columns
    `e_x` is taken as the anchor because the green channel carries no openness pedestal.

    Args:
        heatmaps_rgb: [..., V, H, W, 3] in [0, 255].
        extrinsics:   [..., V, 3, 4] camera-to-world [R|t].
        intrinsics:   [..., V, 3, 3].
        near, far, num_depth_samples, edge_threshold, apply_edge_smoothing:
            Same semantics as `fuse_multiview_heatmaps_to_3d_point_torch`.
        gripper_mode: passed to `decode_gripper_openness_torch`.
        strip_openness_pedestal: subtract the estimated `0.25 * openness` background from the
            blue channel before triangulating it. Without this, an OPEN frame has its whole
            background sitting at 0.25, which drags the weighted centroid toward the image
            center and biases the decoded axis.
        return_matrix: also return the [..., 3, 3] rotation matrices.

    Returns:
        torch.Tensor of shape [..., 8]: [x, y, z, qx, qy, qz, qw, openness]
        -- directly consumable as an RLBench end-effector pose action.
        If `return_matrix`, returns `(pose_8d, rotation_matrices)`.
    """
    heatmaps_rgb = heatmaps_rgb.to(torch.float32)
    pos_heatmaps = heatmaps_rgb[..., 0] / 255.0  # [..., V, H, W]
    up_heatmaps = heatmaps_rgb[..., 1] / 255.0
    blue = heatmaps_rgb[..., 2]  # keep in [0, 255] for the openness decoder

    # 1) Gripper openness from the blue-channel pedestal (paper Eq. 7).
    openness = decode_gripper_openness_torch(blue, mode=gripper_mode)  # [...]

    # 2) Strip that pedestal so the blue channel is a clean blob before triangulation.
    normal_heatmaps = blue / 255.0
    if strip_openness_pedestal:
        pedestal = (0.25 * openness).clamp(0.0, 0.25)[..., None, None, None]  # [..., 1, 1, 1]
        normal_heatmaps = (normal_heatmaps - pedestal).clamp(min=0.0)

    # 3) Triangulate all three semantic points with the same multi-view routine.
    fuse_kwargs = dict(
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        near=near,
        far=far,
        num_depth_samples=num_depth_samples,
        edge_threshold=edge_threshold,
        apply_edge_smoothing=apply_edge_smoothing,
    )
    q_pos = fuse_multiview_heatmaps_to_3d_point_torch(pos_heatmaps, **fuse_kwargs)  # [..., 3]
    q_up = fuse_multiview_heatmaps_to_3d_point_torch(up_heatmaps, **fuse_kwargs)  # [..., 3]
    q_normal = fuse_multiview_heatmaps_to_3d_point_torch(normal_heatmaps, **fuse_kwargs)  # [..., 3]

    # 3b) Enforce the axis length the encoder guaranteed. See `_snap_axis_point_to_length`:
    # a free depth search can land at the wrong point along an axis ray, which leaves the
    # heatmaps looking healthy while the recovered rotation is ~180 deg wrong.
    conf = torch.ones(q_pos.shape[:-1], dtype=torch.bool, device=q_pos.device)

    if axis_solver == "sphere" and axis_length:
        # Search the axis DIRECTION on the sphere of known radius -- see
        # `solve_axis_direction_on_sphere` for why the depth search cannot fix this.
        q_up = solve_axis_direction_on_sphere(
            up_heatmaps, q_pos, extrinsics, intrinsics, axis_length, num_axis_directions)
        q_normal = solve_axis_direction_on_sphere(
            normal_heatmaps, q_pos, extrinsics, intrinsics, axis_length, num_axis_directions)
    elif constrain_axis_depth and axis_length:
        # The encoder places both axis points at EXACTLY `axis_length` metres from the position
        # point, so an axis point's distance from the main camera is bounded by
        # |q_pos - camera| +/- axis_length. Searching the same global [near, far] the position
        # used is what produces the "gripper points at the sky" failure: when the axis is
        # foreshortened it projects on top of the position blob, a whole segment of the ray
        # scores alike, and the argmax slides to the sweep boundary. Measured on a real
        # open_drawer window, frames 0-1 decoded to depth 0.6000 -- exactly the near plane --
        # giving |q_pos - q_axis| = 0.532 m instead of 0.100 m and ~95 deg of axis error.
        # Widening the global range makes it worse (95 -> 117 -> 134 -> 159 deg as near goes
        # 0.6 -> 0.3 -> 0.2 -> 0.1), which confirms the search is degenerate rather than the
        # true point being out of range. Restricting each axis ray to the band around the
        # already-decoded position removes the boundary AND samples it ~5x more finely.
        cam0 = extrinsics[..., 0, :3, 3].to(q_pos.dtype)
        d_pos = (q_pos - cam0).norm(dim=-1)
        band = float(axis_length) * float(axis_depth_band)
        lo = (d_pos - band).clamp(min=1e-3)
        hi = d_pos + band
        band_kwargs = dict(fuse_kwargs)
        band_kwargs.update(near=lo, far=hi)
        q_up = fuse_multiview_heatmaps_to_3d_point_torch(up_heatmaps, **band_kwargs)
        q_normal = fuse_multiview_heatmaps_to_3d_point_torch(normal_heatmaps, **band_kwargs)

    if return_confidence and not constrain_axis_length:
        # Diagnostic only: flag frames whose decoded axis length is far from the encoded
        # 0.1 m. This separates trustworthy rotations from useless ones almost perfectly --
        # measured on ground-truth renders, flagged frames had a median axis error of 93.4 deg
        # against 0.3 deg for the rest -- so it is worth reporting even when no correction is
        # applied. A caller driving a robot should hold position on a flagged frame rather than
        # command what is probably a 180 deg-wrong target.
        for point in (q_up, q_normal):
            gap = ((point - q_pos).norm(dim=-1) - 0.1).abs()
            conf = conf & (gap <= axis_length_tolerance)
    if constrain_axis_length:
        for heat, point, name in ((up_heatmaps, q_up, "up"), (normal_heatmaps, q_normal, "normal")):
            origin, direction = _axis_ray_from_main_view(heat, extrinsics, intrinsics)
            fixed, ok = _snap_axis_point_to_length(
                point, q_pos, origin, direction, constrain_axis_length, axis_length_tolerance)
            if name == "up":
                q_up = fixed
            else:
                q_normal = fixed
            conf = conf & ok

    # 4) Points -> rotation, exactly as in the paper, plus a minimal re-orthogonalization.
    def _norm(v):
        return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    e_x = _norm(q_up - q_pos)
    e_z = _norm(q_pos - q_normal)
    e_y = _norm(torch.cross(e_z, e_x, dim=-1))
    e_z = torch.cross(e_x, e_y, dim=-1)
    rot = torch.stack([e_x, e_y, e_z], dim=-1)  # columns are the axes

    quat = rotation_matrix_to_quaternion_xyzw(rot)  # [..., 4] (x, y, z, w)
    pose_8d = torch.cat([q_pos, quat, openness[..., None].to(q_pos.dtype)], dim=-1)
    out = (pose_8d,)
    if return_matrix:
        out = out + (rot,)
    if return_confidence:
        out = out + (conf,)
    return out[0] if len(out) == 1 else out


def intrinsics_transform(intrinsics, source_size: Tuple[int, int], target_size: Tuple[int, int]):
    """
    Transform camera intrinsics from source image size to target image size.

    Args:
        intrinsics: Camera intrinsics matrix of shape ..., 3, 3
        source_size: (H_source, W_source) - source image dimensions
        target_size: (H_target, W_target) - target image dimensions

    Returns:
        Transformed intrinsics matrix of the same shape as input
    """
    H_source, W_source = source_size
    H_target, W_target = target_size

    # Compute scale factors
    scale_x = W_target / float(W_source)
    scale_y = H_target / float(H_source)

    # Create a copy to avoid in-place modification
    if isinstance(intrinsics, torch.Tensor):
        transformed = intrinsics.clone()
    else:
        transformed = intrinsics.copy()

    # Scale focal lengths and principal point
    transformed[..., 0, 0] *= scale_x  # fx
    transformed[..., 1, 1] *= scale_y  # fy
    transformed[..., 0, 2] *= scale_x  # cx
    transformed[..., 1, 2] *= scale_y  # cy

    return transformed
