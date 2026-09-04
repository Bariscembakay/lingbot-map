

import torch
import logging
from typing import Optional, Tuple
from zipmap.utils.geometry import closed_form_inverse_se3
from train_utils.general import check_and_fix_inf_nan
from einops import rearrange
import math



def check_valid_tensor(input_tensor: Optional[torch.Tensor], name: str = "tensor") -> None:
    """
    Check if a tensor contains NaN or Inf values and log a warning if found.
    
    Args:
        input_tensor: The tensor to check
        name: Name of the tensor for logging purposes
    """
    if input_tensor is not None:
        if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
            logging.warning(f"NaN or Inf found in tensor: {name}")


def normalize_camera_extrinsics_and_points_batch(
    extrinsics: torch.Tensor,
    has_extrinsics_c2w: bool = True,
    cam_points: Optional[torch.Tensor] = None,
    world_points: Optional[torch.Tensor] = None,
    depths: Optional[torch.Tensor] = None,
    scale_by_points: bool = True,
    point_masks: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Normalize camera extrinsics and corresponding 3D points.
    
    This function transforms the coordinate system to be centered at the first camera
    and optionally scales the scene to have unit average distance.
    
    Args:
        extrinsics: Camera extrinsic matrices of shape (B, S, 3, 4)
        cam_points: 3D points in camera coordinates of shape (B, S, H, W, 3) or (*,3)
        world_points: 3D points in world coordinates of shape (B, S, H, W, 3) or (*,3)
        depths: Depth maps of shape (B, S, H, W)
        scale_by_points: Whether to normalize the scale based on point distances
        point_masks: Boolean masks for valid points of shape (B, S, H, W)
    
    Returns:
        Tuple containing:
        - Normalized camera extrinsics of shape (B, S, 3, 4)
        - Normalized camera points (same shape as input cam_points)
        - Normalized world points (same shape as input world_points)
        - Normalized depths (same shape as input depths)
    """
    # Validate inputs
    check_valid_tensor(extrinsics, "extrinsics")
    check_valid_tensor(cam_points, "cam_points")
    check_valid_tensor(world_points, "world_points")
    check_valid_tensor(depths, "depths")

    B, S, _, _ = extrinsics.shape
    device = extrinsics.device
    assert device == torch.device("cpu")


    # Convert extrinsics to homogeneous form: (B, N,4,4)
    extrinsics_homog = torch.cat(
        [
            extrinsics,
            torch.zeros((B, S, 1, 4), device=device),
        ],
        dim=-2,
    )
    extrinsics_homog[:, :, -1, -1] = 1.0

    # first_cam_extrinsic_inv, the inverse of the first camera's extrinsic matrix
    # which can be also viewed as the cam_to_world extrinsic matrix
    first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])
    # new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv)
    new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))  # (B,N,4,4)


    if world_points is not None:
        # since we are transforming the world points to the first camera's coordinate system
        # we directly use the cam_from_world extrinsic matrix of the first camera
        # instead of using the inverse of the first camera's extrinsic matrix
        R = extrinsics[:, 0, :3, :3]
        t = extrinsics[:, 0, :3, 3]
        new_world_points = (world_points @ R.transpose(-1, -2).unsqueeze(1).unsqueeze(2)) + t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    else:
        new_world_points = None


    if scale_by_points:
        new_cam_points = cam_points.clone()
        new_depths = depths.clone()

        dist = new_world_points.norm(dim=-1)
        dist_sum = (dist * point_masks).sum(dim=[1,2,3])
        valid_count = point_masks.sum(dim=[1,2,3])
        avg_scale = (dist_sum / (valid_count + 1e-3)).clamp(min=1e-6, max=1e6)


        new_world_points = new_world_points / avg_scale.view(-1, 1, 1, 1, 1)
        new_extrinsics[:, :, :3, 3] = new_extrinsics[:, :, :3, 3] / avg_scale.view(-1, 1, 1)
        if depths is not None:
            new_depths = new_depths / avg_scale.view(-1, 1, 1, 1)
        if cam_points is not None:
            new_cam_points = new_cam_points / avg_scale.view(-1, 1, 1, 1, 1)
    else:
        return new_extrinsics[:, :, :3], cam_points, new_world_points, depths

    new_extrinsics = new_extrinsics[:, :, :3] # 4x4 -> 3x4
    new_extrinsics = check_and_fix_inf_nan(new_extrinsics, "new_extrinsics", hard_max=None)
    new_cam_points = check_and_fix_inf_nan(new_cam_points, "new_cam_points", hard_max=None)
    new_world_points = check_and_fix_inf_nan(new_world_points, "new_world_points", hard_max=None)
    new_depths = check_and_fix_inf_nan(new_depths, "new_depths", hard_max=None)

    if has_extrinsics_c2w:
        # convert back to cam_from_world extrinsics
        new_extrinsics_c2w = closed_form_inverse_se3(new_extrinsics.reshape(-1, 3, 4)).reshape(B, S, 4, 4)
        new_extrinsics_c2w = new_extrinsics_c2w[:, :, :3]  # 4x4 -> 3x4
        new_extrinsics_c2w = check_and_fix_inf_nan(new_extrinsics_c2w, "new_extrinsics_c2w", hard_max=None)

    else:
        new_extrinsics_c2w = None
    return new_extrinsics, new_extrinsics_c2w, new_cam_points, new_world_points, new_depths




def select_camera_views(normalized_extrinsics_c2w: torch.Tensor, max_total_images: int = -1):
    """
    Select input and target camera views for novel view synthesis (NVS) tasks.
    Args:
        normalized_extrinsics_c2w: Normalized camera extrinsic matrices in camera-to-world (C2W) format with shape (B, S, 3, 4)
        max_total_images: Maximum total number of images per GPU. If -1, doesn't consider this
    Returns:
        Tuple containing:
        - input_view_indices: Indices of selected input views with shape (B, input_view_num)
        - target_view_indices: Indices of selected target views with shape (B, target_view_num)
    """
    B, S, _, _ = normalized_extrinsics_c2w.shape
    # select the input views and target views for NVS
    input_view_num = math.floor(S / 2)
    input_view_num = max(2, int(input_view_num)) # at least 2 input views
    if max_total_images > 0:
        max_image_per_scene = max_total_images // B
        target_view_num = min(max_image_per_scene - input_view_num, S + 1 - input_view_num) # we at most include 1 input views in the target view pool
    else:
        target_view_num = S - input_view_num
    target_view_num = max(1, int(target_view_num)) # at least 1 target view
    target_view_num = min(target_view_num, S)  # cannot exceed available views
    
    device = normalized_extrinsics_c2w.device

    # Generate different random indices for each scene in the batch (vectorized)
    # input view indices: first view and last view + uniformally select (input_view_num - 2) views from the remaining views
    # Use random values + argsort to get random permutation for each batch element
    input_view_interval = S // (input_view_num - 1)
    input_view_indices = [0, S - 1]
    valid_indices = torch.tensor([i for i in range(1, S - 1) if i % input_view_interval == 0], device=device)
    left_input_view_indices = valid_indices[torch.randperm(valid_indices.size(0), device=device)]
    input_view_indices += left_input_view_indices[:input_view_num - 2].tolist()
    input_view_indices = torch.tensor(input_view_indices, device=device).unsqueeze(0).expand(B, -1)  # (B, input_view_num)

    # Step 1: Find views that are NOT selected as input views (target view candidates)
    input_view_set = set(input_view_indices[0].tolist())
    non_input_indices = [i for i in range(S) if i not in input_view_set]
    target_view_candidates = torch.tensor(non_input_indices, device=device)

    # Step 2: Randomly select 2 input views and add them to target view candidates
    num_input_to_add = min(1, input_view_indices.shape[1])
    random_input_idx = torch.randperm(input_view_indices.shape[1], device=device)[:num_input_to_add]
    selected_input_views = input_view_indices[0][random_input_idx]
    target_view_candidates = torch.cat([target_view_candidates, selected_input_views])

    # Step 3: Randomly select target_view_num from all target view candidates
    num_candidates = target_view_candidates.shape[0]
    target_view_num_actual = min(target_view_num, num_candidates)
    rand_perm = torch.randperm(num_candidates, device=device)[:target_view_num_actual]
    target_view_indices = target_view_candidates[rand_perm]

    # Expand to batch dimension
    target_view_indices = target_view_indices.unsqueeze(0).expand(B, -1)  # (B, target_view_num)

    return input_view_indices, target_view_indices


def normalize_camera_c2w(
            extrinsics_c2w: torch.Tensor, 
            input_view_indices: torch.Tensor = None, 
            target_view_indices: torch.Tensor = None,
            ):
    """
    Normalize camera extrinsics from camera-to-world (C2W) format to a standard form.

    Args:
        extrinsics_c2w: Camera-to-world (C2W) extrinsic matrices of shape (B, S, 3, 4)
        input_view_indices: Optional input view indices of shape (B, input_view_num). If provided,
                           normalization scale is computed from these views only.
        target_view_indices: Optional target view indices of shape (B, target_view_num). If provided,
                            returns normalized target views separately.

    Returns:
        If input_view_indices and target_view_indices are provided:
            Tuple of (normalized_input_extrinsics_c2w, normalized_target_extrinsics_c2w)
            - normalized_input_extrinsics_c2w: shape (B, input_view_num, 3, 4)
            - normalized_target_extrinsics_c2w: shape (B, target_view_num, 3, 4)
        Otherwise:
            Normalized camera extrinsics of shape (B, S, 3, 4)
    """
    # we have defined the orientation of the camera coordinate system in the reference frame (first camera)
    # we only need to adjust the translation part of the extrinsics
    B, _, _, _ = extrinsics_c2w.shape
    device = extrinsics_c2w.device

    if input_view_indices is not None and target_view_indices is not None:
        # Select input views to compute normalization scale
        input_view_num = input_view_indices.shape[1]
        target_view_num = target_view_indices.shape[1]
        batch_indices_input = torch.arange(B, device=device)[:, None].expand(B, input_view_num)
        input_extrinsics_c2w = extrinsics_c2w[batch_indices_input, input_view_indices]  # (B, input_view_num, 3, 4)

        # Compute scale from input views only (average distance)
        # scale = torch.amax(input_extrinsics_c2w[:, :, :3, 3].abs(), dim=(1, 2))  # (B,)
        scale = torch.amax(input_extrinsics_c2w[:, :, :3, 3].abs(), dim=(1, 2))  # (B,)

        scale = scale.clamp(min=1e-6, max=1e6)

        # Normalize input views
        normalized_input = input_extrinsics_c2w.clone()
        normalized_input[:, :, :3, 3] = normalized_input[:, :, :3, 3] / scale.view(-1, 1, 1)

        # Select and normalize target views with the same scale
        batch_indices_target = torch.arange(B, device=device)[:, None].expand(B, target_view_num)
        target_extrinsics_c2w = extrinsics_c2w[batch_indices_target, target_view_indices]  # (B, target_view_num, 3, 4)
        normalized_target = target_extrinsics_c2w.clone()
        normalized_target[:, :, :3, 3] = normalized_target[:, :, :3, 3] / scale.view(-1, 1, 1)

        return normalized_input, normalized_target, scale
    else:
        # Original behavior: normalize all views
        # scale = torch.norm(extrinsics_c2w[:, :, :3, 3], dim=-1).mean(dim=1)  # (B,)
        scale = torch.amax(extrinsics_c2w[:, :, :3, 3].abs(), dim=(1, 2))  # (B,)
        scale = scale.clamp(min=1e-6, max=1e6)
        new_extrinsics_c2w = extrinsics_c2w.clone()
        new_extrinsics_c2w[:, :, :3, 3] = new_extrinsics_c2w[:, :, :3, 3] / scale.view(-1, 1, 1)

        return new_extrinsics_c2w






def compute_rays(c2w, intrinsics, h=None, w=None):
        """
        Args:
            c2w (torch.tensor): [b, v, 3, 4] or [b, v, 4, 4]
            intrinsics (torch.tensor): [b, v, 3, 3] - Camera intrinsics matrix in format:
                [[fx, 0, cx],
                 [0, fy, cy],
                 [0,  0,  1]]
            h (int): height of the image
            w (int): width of the image
        Returns:
            ray_o (torch.tensor): [b, v, 3, h, w]
            ray_d (torch.tensor): [b, v, 3, h, w]
        """

        b, v = c2w.size()[:2]
        # Convert to homogeneous form if input is 3x4
        if c2w.shape[-2] == 3:
            # Add [0, 0, 0, 1] row to make it 4x4
            device = c2w.device
            bottom_row = torch.zeros((b, v, 1, 4), device=device)
            bottom_row[:, :, 0, 3] = 1.0
            c2w = torch.cat([c2w, bottom_row], dim=-2)

        c2w = c2w.reshape(b * v, 4, 4)

        # Extract fx, fy, cx, cy from intrinsics matrix
        fx, fy, cx, cy = intrinsics[:, :, 0, 0], intrinsics[:, :, 1, 1], intrinsics[:, :, 0, 2], intrinsics[:, :, 1, 2]
        h_orig = int(2 * cy.max().item())  # Original height (estimated from the intrinsic matrix)
        w_orig = int(2 * cx.max().item())  # Original width (estimated from the intrinsic matrix)
        if h is None or w is None:
            h, w = h_orig, w_orig

        # in case the ray/image map has different resolution than the original image
        if h_orig != h or w_orig != w:
            fx = fx * w / w_orig
            fy = fy * h / h_orig
            cx = cx * w / w_orig
            cy = cy * h / h_orig

        # Reshape for batched operations
        fx = fx.reshape(b * v, 1)
        fy = fy.reshape(b * v, 1)
        cx = cx.reshape(b * v, 1)
        cy = cy.reshape(b * v, 1)

        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")

        x = x[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        y = y[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        x = (x + 0.5 - cx) / fx
        y = (y + 0.5 - cy) / fy
        z = torch.ones_like(x)
        ray_d = torch.stack([x, y, z], dim=2)  # [b*v, h*w, 3]
        ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [b*v, h*w, 3]
        ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # [b*v, h*w, 3]
        ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [b*v, h*w, 3]

        ray_o = rearrange(ray_o, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)
        ray_d = rearrange(ray_d, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)

        return ray_o, ray_d


def get_ray_conditions(
    extrinsics_c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size_hw: Tuple[int, int],
    type: str = "default_plucker"
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Get ray conditionings from camera extrinsics and intrinsics.
    Args:
        extrinsics_c2w (torch.Tensor): Camera extrinsic parameters in camera-to-world format with shape BxSx3x4.
        intrinsics (torch.Tensor): Camera intrinsic parameters with shape BxSx3x3.
        image_size_hw (Tuple[int, int]): Image height and width as (H, W).
        type (str): Type of ray conditioning to compute. Options are "custom_plucker" or "relative_ray".
    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: Ray conditionings and optional additional data with shape BxSxCxhxw,
    """
    ray_o, ray_d = compute_rays(extrinsics_c2w, intrinsics, h=image_size_hw[0], w=image_size_hw[1])  # BxSx3xHxW
    if type == "custom_plucker":
        o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
        nearest_pts = ray_o + o_dot_d * ray_d
        ray_cond = torch.cat([ray_d, nearest_pts], dim=2)
    elif type == "aug_plucker":
        o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
        nearest_pts = ray_o + o_dot_d * ray_d
        o_cross_d = torch.cross(ray_o, ray_d, dim=2)
        ray_cond = torch.cat([o_cross_d, ray_d, nearest_pts], dim=2)
    elif type == "default_plucker":
        o_cross_d = torch.cross(ray_o, ray_d, dim=2)
        ray_cond = torch.cat([o_cross_d, ray_d], dim=2)
    elif type == "relative_ray":
        # Compute relative ray coordinates
        raise NotImplementedError("relative_ray not implemented yet")
    elif type == "plucker_and_common_origin":
        o_cross_d = torch.cross(ray_o, ray_d, dim=2)
        ray_cond = torch.cat([ray_o, ray_d, o_cross_d], dim=2)
    else:
        raise ValueError(f"Unknown ray condition type: {type}")
    return ray_cond