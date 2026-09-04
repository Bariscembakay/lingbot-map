# Code for ZipMap (CVPR 2026); created by Haian Jin


import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from dataclasses import dataclass
from zipmap.utils.pose_enc import extri_intri_to_pose_encoding
from train_utils.general import check_and_fix_inf_nan, safe_mean
from math import ceil, floor
import numpy as np
from train_utils.alignment import align_points_scale
from typing import *
import math
from easydict import EasyDict as edict
import warnings
import lpips
from einops import rearrange


HIGH_QUALITY_DATASETS = ['aria_syn_envir','gta_sfm', 'mvs_synth', 'hypersim', 'matrixcity', 'midair', 'tartanair', 'tartanairv2', 'unreal4k', 'virtual_kitti', 'omniworld_game', 'pointodyssey', 'scenenet', 'dynamic_replica', 'bedlam', 'spring', 'kubric']
MIDDLE_QUALITY_DATASETS = ['blendedmvs', 'DTU', 'ETH3D', 'scannet', 'scannetpp', 'taskonomy', 'arkitscenes', 'mp3d']

DEPTH_DETAIL_LIST = HIGH_QUALITY_DATASETS + MIDDLE_QUALITY_DATASETS




class NVS_Loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lpips_loss_weight = self.config.get("lpips_loss_weight", 0.0)
        self.l2_loss_weight = self.config.get("l2_loss_weight", 1.0)
        if self.lpips_loss_weight > 0.0:
            # Suppress torchvision deprecation warnings from lpips library
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message=".*pretrained.*deprecated.*")
                warnings.filterwarnings("ignore", category=UserWarning, message=".*Arguments other than a weight enum.*")

                # avoid multiple GPUs from downloading the same LPIPS model multiple times
                if torch.distributed.get_rank() == 0:
                    self.lpips_loss_module = self._init_frozen_module(lpips.LPIPS(net="vgg"))
                torch.distributed.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", 0))] if torch.cuda.is_available() else None)

                if torch.distributed.get_rank() != 0:
                    self.lpips_loss_module = self._init_frozen_module(lpips.LPIPS(net="vgg"))
    

    def _init_frozen_module(self, module):
        """Helper method to initialize and freeze a module's parameters."""
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
        # Move module to current CUDA device for distributed training
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            module = module.to(device)
        return module

    def forward(
        self,
        rendering,
        target,
    ):
        """
        Calculate various losses between rendering and target images.
        
        Args:
            rendering: [B, S, H, W, 3], value range [0, 1]
            target: [B, S, 3, H, W], value range [0, 1]

        Returns:
            Dictionary of loss metrics
        """

        rendering = rearrange(rendering, 'b s h w c -> (b s) c h w')
        target = rearrange(target, 'b s c h w -> (b s) c h w')

        # Handle alpha channel if present
        if target.size(1) > 3:
            target, _ = target.split([3, target.size(1) - 3], dim=1)
        if rendering.size(1) > 3:
            rendering, _ = rendering.split([3, rendering.size(1) - 3], dim=1)

        l2_loss = torch.tensor(1e-8).to(rendering.device)
        if self.l2_loss_weight > 0.0:
            l2_loss = F.mse_loss(rendering, target)

        psnr = -10.0 * torch.log10(l2_loss)

        lpips_loss = torch.tensor(0.0).to(l2_loss.device)
        if self.lpips_loss_weight > 0.0:
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                # Scale from [0,1] to [-1,1] as required by LPIPS
                lpips_loss = self.lpips_loss_module(
                    rendering * 2.0 - 1.0, target.detach() * 2.0 - 1.0
                ).mean()


        loss = (
            self.l2_loss_weight * l2_loss
            + self.lpips_loss_weight * lpips_loss
            # + self.config.training.perceptual_loss_weight * perceptual_loss
        )


        loss_metrics = edict(
            loss=loss,
            l2_loss=l2_loss,
            psnr=psnr,
            lpips_loss=lpips_loss,

        )
        return loss_metrics


def weighted_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(dim=dim, keepdim=keepdim).add(eps)

def _smooth(err: torch.FloatTensor, beta: float = 0.0) -> torch.FloatTensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)

def angle_diff_vec3(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12):
    return torch.atan2(torch.cross(v1, v2, dim=-1).norm(dim=-1) + eps, (v1 * v2).sum(dim=-1))



def homogenize_points(
    points,
):
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def depth_edge(depth: torch.Tensor, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: torch.Tensor = None) -> torch.BoolTensor:
    """
    Compute the edge mask of a depth map. The edge is defined as the pixels whose neighbors have a large difference in depth.
    
    Args:
        depth (torch.Tensor): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        edge (torch.Tensor): shape (..., height, width) of dtype torch.bool
    """
    shape = depth.shape
    depth = depth.reshape(-1, 1, *shape[-2:])
    if mask is not None:
        mask = mask.reshape(-1, 1, *shape[-2:])

    if mask is None:
        diff = (F.max_pool2d(depth, kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(-depth, kernel_size, stride=1, padding=kernel_size // 2))
    else:
        diff = (F.max_pool2d(torch.where(mask, depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(torch.where(mask, -depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2))

    edge = torch.zeros_like(depth, dtype=torch.bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= (diff / depth).nan_to_num_() > rtol
    edge = edge.reshape(*shape)
    return edge

# ---------------------------------------------------------------------------
# PointLoss: Scale-invariant Local Pointmap
# ---------------------------------------------------------------------------


def prepare_ROE(pts, mask, target_size=4096):
    B, N, H, W, C = pts.shape
    output = []
    
    for i in range(B):
        valid_pts = pts[i][mask[i]]

        if valid_pts.shape[0] > 0:
            valid_pts = valid_pts.permute(1, 0).unsqueeze(0)  # (1, 3, N1)
            # NOTE: Is is important to use nearest interpolate. Linear interpolate will lead to unstable result!
            valid_pts = F.interpolate(valid_pts, size=target_size, mode='nearest')  # (1, 3, target_size)
            valid_pts = valid_pts.squeeze(0).permute(1, 0)  # (target_size, 3)
        else:
            valid_pts = torch.ones((target_size, C), device=valid_pts.device)

        output.append(valid_pts)

    return torch.stack(output, dim=0)



@dataclass(eq=False)
class Full_MultitaskLoss(torch.nn.Module):
    """
    Multi-task loss module that combines different loss types for ZipMap.
    
    Supports:
    - Camera loss
    - Depth loss 
    - Local Point loss
    - NVS RGB loss
    - NVS Depth loss
    """
    def __init__(self, 
                 camera=None, 
                 camera_svd=None,
                 camera_mlp=None, 
                 depth=None, 
                 local_point=None, 
                 nvs_rgb=None, 
                 nvs_depth=None, 
                 track=None,
                 scale_invariant=True, 
                 **kwargs):
        super().__init__()
        # Loss configuration dictionaries for each task
        self.camera = camera
        self.camera_svd = camera_svd
        self.camera_mlp = camera_mlp
        self.depth = depth
        self.local_point = local_point
        self.nvs_rgb = nvs_rgb
        self.nvs_depth = nvs_depth
        self.track = track

        # Initialize NVS loss module if nvs_rgb config is provided
        if self.nvs_rgb is not None:
            self.nvs_loss_module = NVS_Loss(self.nvs_rgb)
        self.scale_invariant = scale_invariant

    def normalize_pred(self, pred, gt):
        local_points = pred['local_points']
        B, N, H, W, _ = local_points.shape
        masks = gt['point_masks']
        # normalize predict points
        mask_float = masks.to(local_points.dtype)
        point_norms = local_points.norm(dim=-1)
        norm_factor = (point_norms * mask_float).sum(dim=[1, 2, 3]) / (
            mask_float.sum(dim=[1, 2, 3]) + 1e-6
        )
        # Add a clamp to prevent norm_factor from being zero or too small, which can cause NaNs.
        norm_factor = torch.clamp(norm_factor, min=1e-6, max=1e6)
        # local points normalization
        local_points  = local_points / norm_factor[..., None, None, None, None]
        pred['local_points'] = local_points

        # camera translation normalization (w2c)
        if 'pose_enc_list' in pred:
            for i, cur_camera_pred in enumerate(pred["pose_enc_list"]):
                new_translation = cur_camera_pred[..., :3] / norm_factor.view(B, 1, 1)
                pred["pose_enc_list"][i] = torch.cat([new_translation, cur_camera_pred[..., 3:]], dim=-1)
            pred["pose_enc"] = pred["pose_enc_list"][-1]

        if 'pose_enc_svd_list' in pred:
            for i, cur_camera_pred in enumerate(pred["pose_enc_svd_list"]):
                new_translation = cur_camera_pred[..., :3] / norm_factor.view(B, 1, 1)
                pred["pose_enc_svd_list"][i] = torch.cat([new_translation, cur_camera_pred[..., 3:]], dim=-1)
            pred["pose_enc_svd"] = pred["pose_enc_svd_list"][-1]

        if 'pose_enc_mlp_list' in pred:
            for i, cur_camera_pred in enumerate(pred["pose_enc_mlp_list"]):
                new_translation = cur_camera_pred[..., :3] / norm_factor.view(B, 1, 1)
                pred["pose_enc_mlp_list"][i] = torch.cat([new_translation, cur_camera_pred[..., 3:]], dim=-1)
            pred["pose_enc_mlp"] = pred["pose_enc_mlp_list"][-1]

        # depth normalization
        if "depth" in pred:
            pred["depth"] = pred["depth"] / norm_factor.view(B, 1, 1, 1, 1)

        if "nvs_pred" in pred and pred["nvs_pred"].shape[-1] > 3:
            nvs_depth_pred = pred['nvs_pred'][..., 3:] / norm_factor.view(B, 1, 1, 1, 1)
            pred['nvs_pred'] = torch.cat([pred['nvs_pred'][..., :3], nvs_depth_pred], dim=-1)

        return pred



    def forward(self, predictions, batch) -> torch.Tensor:
        """
        Compute the total multi-task loss.
        
        Args:
            predictions: Dict containing model predictions for different tasks
            batch: Dict containing ground truth data and masks
            
        Returns:
            Dict containing individual losses and total objective
        """
        total_loss = 0
        loss_dict = {}
        scale = torch.full((predictions['local_points'].shape[0],), 1.0, device=predictions['local_points'].device)
        with torch.amp.autocast(device_type='cuda', enabled=False):
            if 'local_points' in predictions and self.local_point is not None:
                if self.scale_invariant:
                    predictions = self.normalize_pred(predictions, batch)
                local_pts_loss_dict, scale = compute_local_point_loss(predictions, batch, local_align_res=4096, scale_invariant=self.scale_invariant)
                total_local_pts_loss = local_pts_loss_dict['loss_local_pts'] + local_pts_loss_dict['loss_local_pts_normal']
                total_loss = total_loss + total_local_pts_loss * self.local_point["weight"]
                loss_dict.update(local_pts_loss_dict)

            # Camera pose loss - if pose encodings are predicted
            if "pose_enc_list" in predictions and self.camera is not None:
                # camera translation normalization
                B = predictions["pose_enc_list"][0].shape[0]
                for i, cur_camera_pred in enumerate(predictions["pose_enc_list"]):
                    new_translation = cur_camera_pred[..., :3] * scale.view(B, 1, 1)
                    predictions["pose_enc_list"][i] = torch.cat([new_translation, cur_camera_pred[..., 3:]], dim=-1)
                predictions["pose_enc"] = predictions["pose_enc_list"][-1]
                if self.camera.get("affine_invariant", False):
                    camera_loss_dict = compute_camera_loss_affine_invariant(predictions, batch, **self.camera)
                else:
                    camera_loss_dict = compute_camera_loss(predictions, batch, **self.camera)
                camera_loss = camera_loss_dict["loss_camera"] * self.camera["weight"]
                total_loss = total_loss + camera_loss
                loss_dict.update(camera_loss_dict)

            if "pose_enc_mlp_list" in predictions and self.camera_mlp is not None:
                # camera translation normalization
                assert len(predictions["pose_enc_mlp_list"]) == 1, "Only support single stage camera_mlp loss"
                cur_camera_pred = predictions["pose_enc_mlp_list"][-1]
                B = cur_camera_pred.shape[0]
                new_translation = cur_camera_pred[..., :3] * scale.view(B, 1, 1)
                predictions["pose_enc_mlp_list"][-1] = torch.cat([new_translation, cur_camera_pred[..., 3:]], dim=-1)
                predictions["pose_enc_list"] = predictions["pose_enc_mlp_list"]
                predictions["pose_enc"] = predictions["pose_enc_mlp_list"][-1]
                if self.camera_mlp.get("affine_invariant", False):
                    camera_loss_dict = compute_camera_loss_affine_invariant(predictions, batch, **self.camera_mlp)
                else:
                    camera_loss_dict = compute_camera_loss(predictions, batch, **self.camera_mlp)
                camera_loss = camera_loss_dict["loss_camera"] * self.camera_mlp["weight"]
                total_loss = total_loss + camera_loss
                loss_dict.update(camera_loss_dict)

            # Camera pose loss - if pose encodings are predicted
            if "pose_enc_svd_list" in predictions and self.camera_svd is not None:
                # camera translation normalization
                B = predictions["pose_enc_svd_list"][0].shape[0]
                for i, cur_camera_pred in enumerate(predictions["pose_enc_svd_list"]):
                    new_translation = cur_camera_pred[..., :3] * scale.view(B, 1, 1)
                    predictions["pose_enc_svd_list"][i] = torch.cat([new_translation, cur_camera_pred[..., 3:]], dim=-1)
                predictions["pose_enc_svd"] = predictions["pose_enc_svd_list"][-1]

                camera_svd_loss_dict = compute_camera_svd_loss(predictions, batch, **self.camera_svd)   
                camera_svd_loss = camera_svd_loss_dict["loss_camera_svd"] * self.camera_svd["weight"]   
                total_loss = total_loss + camera_svd_loss
                loss_dict.update(camera_svd_loss_dict)

            # Depth estimation loss - if depth maps are predicted
            if "depth" in predictions and self.depth is not None:
                predictions['depth'] = predictions['depth'] * scale.view(-1, 1, 1, 1, 1)
                depth_loss_dict = compute_depth_loss(predictions, batch, **self.depth)
                depth_loss = depth_loss_dict["loss_conf_depth"] + depth_loss_dict["loss_reg_depth"] + depth_loss_dict["loss_grad_depth"]
                depth_loss = depth_loss * self.depth["weight"]
                total_loss = total_loss + depth_loss
                loss_dict.update(depth_loss_dict)

            if "nvs_pred" in predictions and self.nvs_rgb is not None:
                # Compute novel view synthesis RGB loss
                nvs_loss_dict = self.nvs_loss_module(predictions['nvs_pred'], batch['nvs_target_images'])
                nvs_rgb_loss = nvs_loss_dict['loss'] * self.nvs_rgb.get('weight', 1.0)
                total_loss = total_loss + nvs_rgb_loss
                # Add NVS loss components to loss_dict with renamed keys
                loss_dict.update({
                    'loss_nvs_rgb': nvs_loss_dict['loss'],
                    'loss_nvs_l2': nvs_loss_dict['l2_loss'],
                    'loss_nvs_psnr': nvs_loss_dict['psnr'],
                    'loss_nvs_lpips': nvs_loss_dict['lpips_loss']
                })
            if "nvs_pred" in predictions and self.nvs_depth is not None:
                if predictions['nvs_pred'].shape[-1] <= 3:
                    pass  # No depth channel predicted
                else:
                    # Compute novel view synthesis Depth loss
                    nvs_depth_pred = predictions['nvs_pred'][..., 3:] * scale.view(-1, 1, 1, 1, 1)
                    # Avoid inplace modification of view - use torch.cat instead
                    predictions['nvs_pred'] = torch.cat([predictions['nvs_pred'][..., :3], nvs_depth_pred], dim=-1)
                    nvs_depth_loss_dict = compute_nvs_depth_loss(predictions, batch, **self.nvs_depth)
                    nvs_depth_loss = (nvs_depth_loss_dict["loss_nvs_conf_depth"] + nvs_depth_loss_dict["loss_nvs_reg_depth"] + nvs_depth_loss_dict["loss_nvs_grad_depth"])
                    nvs_depth_loss = nvs_depth_loss * self.nvs_depth["weight"]
                    total_loss = total_loss + nvs_depth_loss
                    loss_dict.update(nvs_depth_loss_dict)

            # Tracking loss - not cleaned yet, dirty code is at the bottom of this file
            if "track" in predictions:
                raise NotImplementedError("Track loss is not cleaned up yet")
            
            loss_dict["loss_objective"] = total_loss

        return loss_dict


def compute_camera_loss(
    pred_dict,              # predictions dict, contains pose encodings
    batch_data,             # ground truth and mask batch dict
    loss_type="l1",         # "l1" or "l2" loss
    gamma=0.6,              # temporal decay weight for multi-stage training
    pose_encoding_type="absT_quaR_FoV",
    weight_trans=1.0,       # weight for translation loss
    weight_rot=1.0,         # weight for rotation loss
    weight_focal=0.5,       # weight for focal length loss
    **kwargs
):
    # List of predicted pose encodings per stage
    pred_pose_encodings = pred_dict['pose_enc_list']
    # Binary mask for valid points per frame (B, N, H, W)
    point_masks = batch_data['point_masks']

    # Check valid points per sample (sum over spatial dimensions only)
    valid_points_per_sample = point_masks.sum(dim=[-2, -1]).max(dim=-1)[0]  # [B] - max across views
    has_valid_points = valid_points_per_sample >= 50
    # print the "seq_name" name of the invalid samples
    if has_valid_points.sum() < has_valid_points.shape[0]:
        invalid_seq_names = np.array(batch_data['seq_name'])[~has_valid_points.cpu().numpy()]
        print(f"Warning: {len(invalid_seq_names)} samples have less than 50 valid points. They are: {invalid_seq_names}")

        # print per view valid point number for the invalid samples
        valid_points_per_sample = point_masks.sum(dim=[-2, -1]) # (B, N)
        invalid_sample_indices = (~has_valid_points).nonzero(as_tuple=True)[0]
        for i in invalid_sample_indices:
            print(f"Sample {batch_data['seq_name'][i]} has valid points per view: {valid_points_per_sample[i].cpu().numpy()}")
    # Only consider frames with enough valid points (>100)
    valid_frame_mask = point_masks[:, 0].sum(dim=[-1, -2]) > 100 # first view mask have enough valid points
    # Combine with sample-level valid points check
    valid_frame_mask = valid_frame_mask & has_valid_points

    # Number of prediction stages
    n_stages = len(pred_pose_encodings)

    # Get ground truth camera extrinsics and intrinsics
    gt_extrinsics = batch_data['extrinsics']
    gt_intrinsics = batch_data['intrinsics']
    image_hw = batch_data['images'].shape[-2:]

    # Encode ground truth pose to match predicted encoding format
    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsics, gt_intrinsics, image_hw, pose_encoding_type=pose_encoding_type
    )

    # Initialize loss accumulators for translation, rotation, focal length
    total_loss_T = total_loss_R = total_loss_FL = 0

    # Compute loss for each prediction stage with temporal weighting
    for stage_idx in range(n_stages):
        # Later stages get higher weight (gamma^0 = 1.0 for final stage)
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pred_pose_stage = pred_pose_encodings[stage_idx]

        if valid_frame_mask.sum() == 0:
            # If no valid frames, set losses to zero to avoid gradient issues
            loss_T_stage = (pred_pose_stage * 0).mean()
            loss_R_stage = (pred_pose_stage * 0).mean()
            loss_FL_stage = (pred_pose_stage * 0).mean()
        else:
            # Only consider valid frames for loss computation
            loss_T_stage, loss_R_stage, loss_FL_stage = camera_loss_single(
                pred_pose_stage[valid_frame_mask].clone(),
                gt_pose_encoding[valid_frame_mask].clone(),
                loss_type=loss_type
            )
        # Accumulate weighted losses across stages
        total_loss_T += loss_T_stage * stage_weight
        total_loss_R += loss_R_stage * stage_weight
        total_loss_FL += loss_FL_stage * stage_weight

    # Average over all stages
    avg_loss_T = total_loss_T / n_stages
    avg_loss_R = total_loss_R / n_stages
    avg_loss_FL = total_loss_FL / n_stages

    # Compute total weighted camera loss
    total_camera_loss = (
        avg_loss_T * weight_trans +
        avg_loss_R * weight_rot +
        avg_loss_FL * weight_focal
    )

    # Return loss dictionary with individual components
    return {
        "loss_camera": total_camera_loss,
        "loss_T": avg_loss_T,
        "loss_R": avg_loss_R,
        "loss_FL": avg_loss_FL
    }


def compute_camera_loss_affine_invariant(
    pred_dict,              # predictions dict, contains pose encodings
    batch_data,             # ground truth and mask batch dict
    gamma=0.6,              # temporal decay weight for multi-stage training
    pose_encoding_type="absT_quaR_FoV",
    weight_trans=10.0,       # weight for translation loss
    weight_rot=0.2,         # weight for rotation loss
    weight_focal=0.5,       # weight for focal length loss
    **kwargs
):
    """
    Compute affine-invariant camera loss using pairwise relative poses.

    This loss is invariant to global scale and translation by comparing
    all pairwise relative poses between views instead of absolute poses.

    Args:
        pred_dict: predictions dict, contains pose encodings
        batch_data: ground truth and mask batch dict
        gamma: temporal decay weight for multi-stage training
        pose_encoding_type: type of pose encoding ("absT_quaR_FoV")
        weight_trans: weight for translation loss
        weight_rot: weight for rotation loss
        weight_focal: weight for focal length loss
        alpha: scaling factor for translation loss (alpha * trans_loss + rot_loss)
    """
    from zipmap.utils.rotation import quat_to_mat

    # List of predicted pose encodings per stage
    pred_pose_encodings = pred_dict['pose_enc_list']
    # Binary mask for valid points per frame (B, N, H, W)
    point_masks = batch_data['point_masks']

    # Check valid points per sample
    valid_points_per_sample = point_masks.sum(dim=[-2, -1]).max(dim=-1)[0]  # [B] - max across views
    has_valid_points = valid_points_per_sample >= 50

    # Print warning for invalid samples
    if has_valid_points.sum() < has_valid_points.shape[0]:
        invalid_seq_names = np.array(batch_data['seq_name'])[~has_valid_points.cpu().numpy()]
        print(f"Warning: {len(invalid_seq_names)} samples have less than 50 valid points. They are: {invalid_seq_names}")

    # Only consider frames with enough valid points
    valid_frame_mask = point_masks[:, 0].sum(dim=[-1, -2]) > 100
    valid_frame_mask = valid_frame_mask & has_valid_points

    # Number of prediction stages
    n_stages = len(pred_pose_encodings)

    # Get ground truth camera extrinsics and intrinsics
    gt_extrinsics = batch_data['extrinsics']
    gt_intrinsics = batch_data['intrinsics']
    image_hw = batch_data['images'].shape[-2:]

    # Encode ground truth pose to match predicted encoding format
    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsics, gt_intrinsics, image_hw, pose_encoding_type=pose_encoding_type
    )

    # Initialize loss accumulators
    total_loss_T = total_loss_R = total_loss_FL = 0

    # Compute loss for each prediction stage with temporal weighting
    for stage_idx in range(n_stages):
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pred_pose_stage = pred_pose_encodings[stage_idx]

        if valid_frame_mask.sum() == 0:
            # If no valid frames, set losses to zero
            loss_T_stage = (pred_pose_stage * 0).mean()
            loss_R_stage = (pred_pose_stage * 0).mean()
            loss_FL_stage = (pred_pose_stage * 0).mean()
        else:
            # Compute affine-invariant loss
            valid_pred = pred_pose_stage[valid_frame_mask].clone()  # (B', N, D)
            valid_gt = gt_pose_encoding[valid_frame_mask].clone()

            # Extract components
            pred_T = valid_pred[..., :3]  # (B', N, 3)
            pred_quat = valid_pred[..., 3:7]  # (B', N, 4)
            gt_T = valid_gt[..., :3]
            gt_quat = valid_gt[..., 3:7]

            # Convert quaternions to rotation matrices
            pred_R = quat_to_mat(pred_quat)  # (B', N, 3, 3)
            gt_R = quat_to_mat(gt_quat)

            # Construct SE(3) matrices from pose encodings (w2c format)
            B_valid, N = pred_T.shape[:2]

            # Build 4x4 transformation matrices
            pred_w2c = torch.zeros(B_valid, N, 4, 4, device=pred_T.device, dtype=pred_T.dtype)
            pred_w2c[..., :3, :3] = pred_R
            pred_w2c[..., :3, 3] = pred_T
            pred_w2c[..., 3, 3] = 1.0

            gt_w2c = torch.zeros(B_valid, N, 4, 4, device=gt_T.device, dtype=gt_T.dtype)
            gt_w2c[..., :3, :3] = gt_R
            gt_w2c[..., :3, 3] = gt_T
            gt_w2c[..., 3, 3] = 1.0

            # Convert w2c to c2w using se3_inverse
            pred_c2w = se3_inverse(pred_w2c.reshape(-1, 4, 4)).reshape(B_valid, N, 4, 4)
            gt_c2w = se3_inverse(gt_w2c.reshape(-1, 4, 4)).reshape(B_valid, N, 4, 4)

            # Compute all pairwise relative poses: T_ij = w2c[i] @ c2w[j]
            pred_w2c_exp = pred_w2c.unsqueeze(2)  # (B', N, 1, 4, 4)
            pred_c2w_exp = pred_c2w.unsqueeze(1)  # (B', 1, N, 4, 4)

            gt_w2c_exp = gt_w2c.unsqueeze(2)
            gt_c2w_exp = gt_c2w.unsqueeze(1)

            pred_rel_all = torch.matmul(pred_w2c_exp, pred_c2w_exp)  # (B', N, N, 4, 4)
            gt_rel_all = torch.matmul(gt_w2c_exp, gt_c2w_exp)

            # Mask to exclude diagonal (i == j)
            mask = ~torch.eye(N, dtype=torch.bool, device=pred_pose_stage.device)

            # Extract translation and rotation from relative poses
            pred_rel_T = pred_rel_all[..., :3, 3][:, mask, :]  # (B', N*(N-1), 3)
            pred_rel_R = pred_rel_all[..., :3, :3][:, mask, :, :]  # (B', N*(N-1), 3, 3)

            gt_rel_T = gt_rel_all[..., :3, 3][:, mask, :]
            gt_rel_R = gt_rel_all[..., :3, :3][:, mask, :, :]

            # Translation loss (Huber loss)
            loss_T_stage = F.huber_loss(pred_rel_T, gt_rel_T, reduction='mean', delta=0.1)

            # Rotation loss (angular error)
            pred_rel_R_masked = pred_rel_R.reshape(-1, 3, 3)
            gt_rel_R_masked = gt_rel_R.reshape(-1, 3, 3)

            residual = torch.matmul(pred_rel_R_masked.transpose(-2, -1), gt_rel_R_masked)
            trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
            cosine = (trace - 1) / 2
            eps = 1e-6
            R_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))
            loss_R_stage = R_err.mean()

            # Focal length loss (L1)
            loss_FL_stage = (valid_pred[..., 7:] - valid_gt[..., 7:]).abs().mean()

        # Check/fix numerical issues
        loss_T_stage = check_and_fix_inf_nan(loss_T_stage, "loss_T")
        loss_R_stage = check_and_fix_inf_nan(loss_R_stage, "loss_R")
        loss_FL_stage = check_and_fix_inf_nan(loss_FL_stage, "loss_FL")

        # Accumulate weighted losses
        total_loss_T += loss_T_stage * stage_weight
        total_loss_R += loss_R_stage * stage_weight
        total_loss_FL += loss_FL_stage * stage_weight

    # Average over stages
    avg_loss_T = total_loss_T / n_stages
    avg_loss_R = total_loss_R / n_stages
    avg_loss_FL = total_loss_FL / n_stages

    # Compute total weighted camera loss
    # Use alpha scaling: trans_loss + rot_loss (like in reference)
    total_camera_loss = (
        avg_loss_T * weight_trans +
        avg_loss_R * weight_rot +
        avg_loss_FL * weight_focal
    )

    # Return loss dictionary
    return {
        "loss_camera": total_camera_loss,
        "loss_T": avg_loss_T * weight_trans,
        "loss_R": avg_loss_R * weight_rot,
        "loss_FL": avg_loss_FL
    }


def compute_camera_svd_loss(
    pred_dict,              # predictions dict, contains pose encodings
    batch_data,             # ground truth and mask batch dict
    loss_type="l1",         # "l1" or "l2" loss
    gamma=0.6,              # temporal decay weight for multi-stage training
    pose_encoding_type="absT_svdR_FoV",
    weight_trans=1.0,       # weight for translation loss
    weight_rot=1.0,         # weight for rotation loss
    weight_focal=0.5,       # weight for focal length loss
    **kwargs
):
    # List of predicted pose encodings per stage
    pred_pose_encodings = pred_dict['pose_enc_svd_list']
    # Binary mask for valid points per frame (B, N, H, W)
    point_masks = batch_data['point_masks']

    # Check valid points per sample (sum over spatial dimensions only)
    valid_points_per_sample = point_masks.sum(dim=[-2, -1]).max(dim=-1)[0]  # [B] - min across views
    has_valid_points = valid_points_per_sample >= 50

    # Only consider frames with enough valid points (>100)
    valid_frame_mask = point_masks[:, 0].sum(dim=[-1, -2]) > 100 # first view mask have enough valid points
    # Combine with sample-level valid points check
    valid_frame_mask = valid_frame_mask & has_valid_points

    # Number of prediction stages
    n_stages = len(pred_pose_encodings)

    # Get ground truth camera extrinsics and intrinsics
    gt_extrinsics_c2w = batch_data['extrinsics_c2w']
    gt_intrinsics = batch_data['intrinsics']
    image_hw = batch_data['images'].shape[-2:]

    # Encode ground truth pose to match predicted encoding format
    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsics_c2w, gt_intrinsics, image_hw, pose_encoding_type=pose_encoding_type
    )

    # Initialize loss accumulators for translation, rotation, focal length
    total_loss_T_svd = total_loss_R_svd = total_loss_FL_svd = 0

    # Compute loss for each prediction stage with temporal weighting
    for stage_idx in range(n_stages):
        # Later stages get higher weight (gamma^0 = 1.0 for final stage)
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pred_pose_stage = pred_pose_encodings[stage_idx]

        if valid_frame_mask.sum() == 0:
            # If no valid frames, set losses to zero to avoid gradient issues
            loss_T_svd_stage = (pred_pose_stage * 0).mean()
            loss_R_svd_stage = (pred_pose_stage * 0).mean()
            loss_FL_svd_stage = (pred_pose_stage * 0).mean()
        else:
            # Only consider valid frames for loss computation
            loss_T_svd_stage, loss_R_svd_stage, loss_FL_svd_stage = camera_svd_loss_single(
                pred_pose_stage[valid_frame_mask].clone(),
                gt_pose_encoding[valid_frame_mask].clone(),
                loss_type=loss_type
            )
        # Accumulate weighted losses across stages
        total_loss_T_svd += loss_T_svd_stage * stage_weight
        total_loss_R_svd += loss_R_svd_stage * stage_weight
        total_loss_FL_svd += loss_FL_svd_stage * stage_weight

    # Average over all stages
    avg_loss_T_svd = total_loss_T_svd / n_stages
    avg_loss_R_svd = total_loss_R_svd / n_stages
    avg_loss_FL_svd = total_loss_FL_svd / n_stages

    # Compute total weighted camera loss
    total_camera_loss = (
        avg_loss_T_svd * weight_trans +
        avg_loss_R_svd * weight_rot +
        avg_loss_FL_svd * weight_focal
    )

    # Return loss dictionary with individual components
    return {
        "loss_camera_svd": total_camera_loss,
        "loss_T_svd": avg_loss_T_svd,
        "loss_R_svd": avg_loss_R_svd,
        "loss_FL_svd": avg_loss_FL_svd
    }


def camera_loss_single(pred_pose_enc, gt_pose_enc, loss_type="l1"):
    """
    Computes translation, rotation, and focal loss for a batch of pose encodings.
    
    Args:
        pred_pose_enc: (N, D) predicted pose encoding
        gt_pose_enc: (N, D) ground truth pose encoding
        loss_type: "l1" (abs error) or "l2" (euclidean error)
    Returns:
        loss_T: translation loss (mean)
        loss_R: rotation loss (mean)
        loss_FL: focal length/intrinsics loss (mean)
    
    NOTE: The paper uses smooth l1 loss, but we found l1 loss is more stable than smooth l1 and l2 loss.
        So here we use l1 loss.
    """
    if loss_type == "l1":
        # Translation: first 3 dims; Rotation: next 4 (quaternion); Focal/Intrinsics: last dims
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs()
    elif loss_type == "l2":
        # L2 norm for each component
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1)
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).norm(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Check/fix numerical issues (nan/inf) for each loss component
    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")

    # Clamp outlier translation loss to prevent instability, then average
    loss_T = safe_mean(loss_T.clamp(max=100))
    loss_R = safe_mean(loss_R)
    loss_FL = safe_mean(loss_FL)

    return loss_T, loss_R, loss_FL

def camera_svd_loss_single(pred_pose_enc, gt_pose_enc, loss_type="l1"):
    """
    Computes translation, rotation, and focal loss for a batch of pose encodings.
    
    Args:
        pred_pose_enc: (N, D) predicted pose encoding
        gt_pose_enc: (N, D) ground truth pose encoding
        loss_type: "l1" (abs error) or "l2" (euclidean error)
    Returns:
        loss_T_svd: translation loss (mean)
        loss_R_svd: rotation loss (mean)
        loss_FL_svd: focal length/intrinsics loss (mean)
    
    NOTE: The paper uses smooth l1 loss, but we found l1 loss is more stable than smooth l1 and l2 loss.
        So here we use l1 loss.
    """
    if loss_type == "l1":
        # Translation: first 3 dims; Rotation: next 4 (quaternion); Focal/Intrinsics: last dims
        loss_T_svd = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
        loss_R_svd = (pred_pose_enc[..., 3:12] - gt_pose_enc[..., 3:12]).abs()
        loss_FL_svd = (pred_pose_enc[..., 12:] - gt_pose_enc[..., 12:]).abs()
    elif loss_type == "l2":
        # L2 norm for each component
        loss_T_svd = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, keepdim=True)
        loss_R_svd = (pred_pose_enc[..., 3:12] - gt_pose_enc[..., 3:12]).norm(dim=-1)
        loss_FL_svd = (pred_pose_enc[..., 12:] - gt_pose_enc[..., 12:]).norm(dim=-1)
    
    elif loss_type == "huber":
        delta = 1.0
        loss_T_svd = F.huber_loss(pred_pose_enc[..., :3], gt_pose_enc[..., :3], reduction='mean', delta=delta)
        loss_R_svd = F.huber_loss(pred_pose_enc[..., 3:12], gt_pose_enc[..., 3:12], reduction='mean', delta=delta)
        loss_FL_svd = F.huber_loss(pred_pose_enc[..., 12:], gt_pose_enc[..., 12:], reduction='mean', delta=delta)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Check/fix numerical issues (nan/inf) for each loss component
    loss_T_svd = check_and_fix_inf_nan(loss_T_svd, "loss_T")
    loss_R_svd = check_and_fix_inf_nan(loss_R_svd, "loss_R")
    loss_FL_svd = check_and_fix_inf_nan(loss_FL_svd, "loss_FL")

    # Clamp outlier translation loss to prevent instability, then average
    loss_T_svd = safe_mean(loss_T_svd.clamp(max=100))
    loss_R_svd = safe_mean(loss_R_svd)
    loss_FL_svd = safe_mean(loss_FL_svd)

    return loss_T_svd, loss_R_svd, loss_FL_svd


def compute_point_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1, **kwargs):
    """
    Compute point loss.
    
    Args:
        predictions: Dict containing 'world_points' and 'world_points_conf'
        batch: Dict containing ground truth 'world_points' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_points = predictions['world_points']
    pred_points_conf = predictions['world_points_conf']
    gt_points = batch['world_points']
    gt_points_mask = batch['point_masks']
    
    gt_points = check_and_fix_inf_nan(gt_points, "gt_points")
    
    if gt_points_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_points).mean()
        loss_dict = {f"loss_conf_point": dummy_loss,
                    f"loss_reg_point": dummy_loss,
                    f"loss_grad_point": dummy_loss,}
        return loss_dict
    
    # Compute confidence-weighted regression loss with optional gradient loss
    loss_conf, loss_grad, loss_reg = regression_loss(pred_points, gt_points, gt_points_mask, conf=pred_points_conf,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range)
    
    loss_dict = {
        f"loss_conf_point": loss_conf,
        f"loss_reg_point": loss_reg,
        f"loss_grad_point": loss_grad,
    }
    
    return loss_dict



def compute_local_point_loss(pred, gt, local_align_res=4096, scale_invariant=True):
    pred_local_pts = pred['local_points']
    gt_local_pts = gt['cam_points']
    valid_masks = gt['point_masks']

    gt_local_pts = check_and_fix_inf_nan(gt_local_pts, "gt_local_pts")

    B, N, H, W, _ = pred_local_pts.shape

    # Check valid points per sample (sum over spatial dimensions only)
    valid_points_per_sample = valid_masks.sum(dim=[-2, -1]).max(dim=-1)[0]  # [B] - max across views
    has_valid_points = valid_points_per_sample >= 50

    if not has_valid_points.any():
        # If no samples have valid points, return zero loss
        dummy_loss = (0.0 * pred_local_pts).mean()
        loss_dict = {
            f"loss_local_pts": dummy_loss,
            f"loss_local_pts_normal": dummy_loss,
        }
        return loss_dict, torch.ones(B, device=pred_local_pts.device)

    weights_ = gt_local_pts[..., 2]
    weights_ = weights_.clamp_min(0.1 * weighted_mean(weights_, valid_masks, dim=(-2, -1), keepdim=True))
    weights_ = 1 / (weights_ + 1e-6)

    if scale_invariant:
        # alignment
        with torch.no_grad():
            # xyz_pred_local shape (B, local_align_res, 3)
            xyz_pred_local = prepare_ROE(pred_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=local_align_res).contiguous()
            xyz_gt_local = prepare_ROE(gt_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=local_align_res).contiguous()
            xyz_weights_local = prepare_ROE((weights_[..., None]).reshape(B, N, H, W, 1), valid_masks.reshape(B, N, H, W), target_size=local_align_res).contiguous()[:, :, 0]
            S_opt_local = align_points_scale(xyz_pred_local, xyz_gt_local, xyz_weights_local)
            S_opt_local[S_opt_local <= 0] *= -1

        aligned_local_pts = S_opt_local.view(B, 1, 1, 1, 1) * pred_local_pts
    else:
        S_opt_local = torch.full((B,), 1.0, device=pred_local_pts.device)
        aligned_local_pts = pred_local_pts

    # Only compute loss for samples with valid points
    loss_mask = valid_masks & has_valid_points.view(B, 1, 1, 1)
    if loss_mask.sum() > 0:
        loss_local_pts = nn.functional.l1_loss(aligned_local_pts[loss_mask].float(), gt_local_pts[loss_mask].float(), reduction='none') * weights_[loss_mask].float()[..., None]
        loss_local_pts = check_and_fix_inf_nan(loss_local_pts, "loss_local_pts")
        loss_local_pts = safe_mean(loss_local_pts)
    else:
        loss_local_pts = (0.0 * pred_local_pts).mean() # dummy_loss

    # normal loss for local points
    normal_batch_idx = [i for i in range(len(gt['dataset_name'])) if gt['dataset_name'][i] in DEPTH_DETAIL_LIST]

    if len(normal_batch_idx) == 0 or loss_mask.sum() == 0:
        loss_local_pts_normal =  0.0 * loss_local_pts
    else:
        loss_local_pts_normal = local_point_normal_loss(aligned_local_pts[normal_batch_idx], gt_local_pts[normal_batch_idx], valid_masks[normal_batch_idx])
        loss_local_pts_normal = check_and_fix_inf_nan(loss_local_pts_normal, "loss_local_point_normal")
        loss_local_pts_normal = safe_mean(loss_local_pts_normal)


    loss_dict = {
        f"loss_local_pts": loss_local_pts,
        f"loss_local_pts_normal": loss_local_pts_normal,
    }


    return loss_dict, S_opt_local

    
    

def compute_depth_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1, **kwargs):
    """
    Compute depth loss.

    Args:
        predictions: Dict containing 'depth' and 'depth_conf'
        batch: Dict containing ground truth 'depths' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_depth = predictions['depth']
    pred_depth_conf = predictions['depth_conf']

    gt_depth = batch['depths']
    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")
    gt_depth = gt_depth[..., None]              # (B, N, H, W, 1)
    gt_depth_mask = batch['point_masks'].clone()   # 3D points derived from depth map, so we use the same mask

    # Check valid points per sample (sum over spatial dimensions only)
    valid_points_per_sample = gt_depth_mask.sum(dim=[-2, -1]).max(dim=-1)[0]  # [B] - min across views
    has_valid_points = valid_points_per_sample >= 50

    if not has_valid_points.any():
        # If no samples have valid points, return zero loss
        dummy_loss = (0.0 * pred_depth).mean()
        loss_dict = {f"loss_conf_depth": dummy_loss,
                    f"loss_reg_depth": dummy_loss,
                    f"loss_grad_depth": dummy_loss,}
        return loss_dict

    # Only compute loss for samples with valid points
    loss_mask = gt_depth_mask & has_valid_points.view(-1, 1, 1, 1)

    # NOTE: we put conf inside regression_loss so that we can also apply conf loss to the gradient loss in a multi-scale manner
    # this is hacky, but very easier to implement
    loss_conf, loss_grad, loss_reg = regression_loss(pred_depth, gt_depth, loss_mask, conf=pred_depth_conf,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range)

    loss_dict = {
        f"loss_conf_depth": loss_conf,
        f"loss_reg_depth": loss_reg,
        f"loss_grad_depth": loss_grad,
    }

    return loss_dict


def compute_nvs_depth_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1, **kwargs):
    """
    Compute depth loss.

    Args:
        predictions: Dict containing 'depth' and 'depth_conf'
        batch: Dict containing ground truth 'depths' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_depth = predictions['nvs_pred'][..., 3:4]
    pred_depth_conf = predictions['nvs_depth_conf']

    gt_depth = batch['nvs_target_depths']
    gt_depth = check_and_fix_inf_nan(gt_depth, "nvs_gt_depth")
    gt_depth = gt_depth[..., None]              # (B, N, H, W, 1)
    gt_depth_mask = batch['nvs_target_point_masks'].clone()   # 3D points derived from depth map, so we use the same mask

    # Check valid points per sample (sum over spatial dimensions only)
    valid_points_per_sample = gt_depth_mask.sum(dim=[-2, -1]).max(dim=-1)[0]  # [B] - min across views
    has_valid_points = valid_points_per_sample >= 50

    if not has_valid_points.any():
        # If no samples have valid points, return zero loss
        dummy_loss = (0.0 * pred_depth).mean()
        loss_dict = {f"loss_nvs_conf_depth": dummy_loss,
                    f"loss_nvs_reg_depth": dummy_loss,
                    f"loss_nvs_grad_depth": dummy_loss,}
        return loss_dict

    # Only compute loss for samples with valid points
    loss_mask = gt_depth_mask & has_valid_points.view(-1, 1, 1, 1)

    # NOTE: we put conf inside regression_loss so that we can also apply conf loss to the gradient loss in a multi-scale manner
    # this is hacky, but very easier to implement
    loss_conf, loss_grad, loss_reg = regression_loss(pred_depth, gt_depth, loss_mask, conf=pred_depth_conf,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range)

    loss_dict = {
        f"loss_nvs_conf_depth": loss_conf,
        f"loss_nvs_reg_depth": loss_reg,
        f"loss_nvs_grad_depth": loss_grad,
    }

    return loss_dict


def regression_loss(pred, gt, mask, conf=None, gradient_loss_fn=None, gamma=1.0, alpha=0.2, valid_range=-1):
    """
    Core regression loss function with confidence weighting and optional gradient loss.
    
    Computes:
    1. gamma * ||pred - gt||^2 * conf - alpha * log(conf)
    2. Optional gradient loss
    
    Args:
        pred: (B, S, H, W, C) predicted values
        gt: (B, S, H, W, C) ground truth values
        mask: (B, S, H, W) valid pixel mask
        conf: (B, S, H, W) confidence weights (optional)
        gradient_loss_fn: Type of gradient loss ("normal", "grad", etc.)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        valid_range: Quantile range for outlier filtering
    
    Returns:
        loss_conf: Confidence-weighted loss
        loss_grad: Gradient loss (0 if not specified)
        loss_reg: Regular L2 loss
    """
    bb, ss, hh, ww, nc = pred.shape

    # Compute L2 distance between predicted and ground truth points
    loss_reg = torch.norm(gt[mask] - pred[mask], dim=-1)
    loss_reg = check_and_fix_inf_nan(loss_reg, "loss_reg")

    # Confidence-weighted loss: gamma * loss * conf - alpha * log(conf)
    # This encourages the model to be confident on easy examples and less confident on hard ones
    loss_conf = gamma * loss_reg * conf[mask] - alpha * torch.log(conf[mask])
    loss_conf = check_and_fix_inf_nan(loss_conf, "loss_conf")
        
    # Initialize gradient loss
    loss_grad = 0

    # Prepare confidence for gradient loss if needed
    if "conf" in gradient_loss_fn:
        to_feed_conf = conf.reshape(bb*ss, hh, ww)
    else:
        to_feed_conf = None

    # Compute gradient loss if specified for spatial smoothness
    if "normal" in gradient_loss_fn:
        # Surface normal-based gradient loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=normal_loss,
            scales=3,
            conf=to_feed_conf,
        )
    elif "grad" in gradient_loss_fn:
        # Standard gradient-based loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=gradient_loss,
            conf=to_feed_conf,
        )

    loss_grad = check_and_fix_inf_nan(loss_grad, "loss_grad")
    # Process confidence-weighted loss
    if loss_conf.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_conf = filter_by_quantile(loss_conf, valid_range)

        loss_conf = check_and_fix_inf_nan(loss_conf, f"loss_conf_depth")
        loss_conf = safe_mean(loss_conf)
    else:
        loss_conf = (0.0 * pred).mean()

    # Process regular regression loss
    if loss_reg.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_reg = filter_by_quantile(loss_reg, valid_range)

        loss_reg = check_and_fix_inf_nan(loss_reg, f"loss_reg_depth")
        loss_reg = safe_mean(loss_reg)
    else:
        loss_reg = (0.0 * pred).mean()

    return loss_conf, loss_grad, loss_reg


def gradient_loss_multi_scale_wrapper(prediction, target, mask, scales=4, gradient_loss_fn = None, conf=None):
    """
    Multi-scale gradient loss wrapper. Applies gradient loss at multiple scales by subsampling the input.
    This helps capture both fine and coarse spatial structures.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values  
        mask: (B, H, W) valid pixel mask
        scales: Number of scales to use
        gradient_loss_fn: Gradient loss function to apply
        conf: (B, H, W) confidence weights (optional)
    """
    total = 0
    for scale in range(scales):
        step = pow(2, scale)  # Subsample by 2^scale

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None
        )

    total = total / scales
    return total


def local_point_normal_loss(points, gt_points, mask):
    not_edge = ~depth_edge(gt_points[..., 2], rtol=0.03)
    mask = torch.logical_and(mask, not_edge)

    leftup, rightup, leftdown, rightdown = points[..., :-1, :-1, :], points[..., :-1, 1:, :], points[..., 1:, :-1, :], points[..., 1:, 1:, :]
    upxleft = torch.cross(rightup - rightdown, leftdown - rightdown, dim=-1)
    leftxdown = torch.cross(leftup - rightup, rightdown - rightup, dim=-1)
    downxright = torch.cross(leftdown - leftup, rightup - leftup, dim=-1)
    rightxup = torch.cross(rightdown - leftdown, leftup - leftdown, dim=-1)

    gt_leftup, gt_rightup, gt_leftdown, gt_rightdown = gt_points[..., :-1, :-1, :], gt_points[..., :-1, 1:, :], gt_points[..., 1:, :-1, :], gt_points[..., 1:, 1:, :]
    gt_upxleft = torch.cross(gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1)
    gt_leftxdown = torch.cross(gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1)
    gt_downxright = torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1)
    gt_rightxup = torch.cross(gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1)

    mask_leftup, mask_rightup, mask_leftdown, mask_rightdown = mask[..., :-1, :-1], mask[..., :-1, 1:], mask[..., 1:, :-1], mask[..., 1:, 1:]
    mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
    mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
    mask_downxright = mask_leftdown & mask_rightup & mask_leftup
    mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown

    MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(1), math.radians(90), math.radians(3)

    loss = mask_upxleft * _smooth(angle_diff_vec3(upxleft, gt_upxleft).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
            + mask_leftxdown * _smooth(angle_diff_vec3(leftxdown, gt_leftxdown).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
            + mask_downxright * _smooth(angle_diff_vec3(downxright, gt_downxright).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
            + mask_rightxup * _smooth(angle_diff_vec3(rightxup, gt_rightxup).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)

    loss = loss.mean() / (4 * max(points.shape[-3:-1]))

    return loss



def normal_loss(prediction, target, mask, cos_eps=1e-8, conf=None, gamma=1.0, alpha=0.2):
    """
    Surface normal-based loss for geometric consistency.
    
    Computes surface normals from 3D point maps using cross products of neighboring points,
    then measures the angle between predicted and ground truth normals.
    
    Args:
        prediction: (B, H, W, 3) predicted 3D coordinates/points
        target: (B, H, W, 3) ground-truth 3D coordinates/points
        mask: (B, H, W) valid pixel mask
        cos_eps: Epsilon for numerical stability in cosine computation
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Convert point maps to surface normals using cross products
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals,   gt_valids   = point_map_to_normal(target,     mask, eps=cos_eps)

    # Only consider regions where both predicted and GT normals are valid
    all_valid = pred_valids & gt_valids  # shape: (4, B, H, W)

    # Early return if not enough valid points
    divisor = torch.sum(all_valid)
    if divisor < 10:
        return 0

    # Extract valid normals
    pred_normals = pred_normals[all_valid].clone()
    gt_normals = gt_normals[all_valid].clone()

    # Compute cosine similarity between corresponding normals
    dot = torch.sum(pred_normals * gt_normals, dim=-1)

    # Clamp dot product to [-1, 1] for numerical stability
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)

    # Compute loss as 1 - cos(theta), instead of arccos(dot) for numerical stability
    loss = 1 - dot

    # Return mean loss if we have enough valid points
    if loss.numel() < 10:
        return 0
    else:
        loss = check_and_fix_inf_nan(loss, "normal_loss")

        if conf is not None:
            # Apply confidence weighting
            conf = conf[None, ...].expand(4, -1, -1, -1)
            conf = conf[all_valid].clone()

            loss = gamma * loss * conf - alpha * torch.log(conf)
            return safe_mean(loss)
        else:
            return safe_mean(loss)


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """
    Gradient-based loss. Computes the L1 difference between adjacent pixels in x and y directions.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values
        mask: (B, H, W) valid pixel mask
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Expand mask to match prediction channels
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    # Compute difference between prediction and target
    diff = prediction - target
    diff = torch.mul(mask, diff)

    # Compute gradients in x direction (horizontal)
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Compute gradients in y direction (vertical)
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Clamp gradients to prevent outliers
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    # Apply confidence weighting if provided
    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    # Sum gradients and normalize by number of valid pixels
    grad_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        grad_loss = torch.sum(grad_loss) / divisor

    return grad_loss


def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    Convert 3D point map to surface normal vectors using cross products.
    
    Computes normals by taking cross products of neighboring point differences.
    Uses 4 different cross-product directions for robustness.
    
    Args:
        point_map: (B, H, W, 3) 3D points laid out in a 2D grid
        mask: (B, H, W) valid pixels (bool)
        eps: Epsilon for numerical stability in normalization
    
    Returns:
        normals: (4, B, H, W, 3) normal vectors for each of the 4 cross-product directions
        valids: (4, B, H, W) corresponding valid masks
    """
    with torch.amp.autocast('cuda', enabled=False):
        # Pad inputs to avoid boundary issues
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1,1,1,1), mode='constant', value=0).permute(0, 2, 3, 1)

        # Get neighboring points for each pixel
        center = pts[:, 1:-1, 1:-1, :]   # B,H,W,3
        up     = pts[:, :-2,  1:-1, :]
        left   = pts[:, 1:-1, :-2 , :]
        down   = pts[:, 2:,   1:-1, :]
        right  = pts[:, 1:-1, 2:,   :]

        # Compute direction vectors from center to neighbors
        up_dir    = up    - center
        left_dir  = left  - center
        down_dir  = down  - center
        right_dir = right - center

        # Compute four cross products for different normal directions
        n1 = torch.cross(up_dir,   left_dir,  dim=-1)  # up x left
        n2 = torch.cross(left_dir, down_dir,  dim=-1)  # left x down
        n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
        n4 = torch.cross(right_dir,up_dir,    dim=-1)  # right x up

        # Validity masks - require both direction pixels to be valid
        v1 = padded_mask[:, :-2,  1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2 ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:,   1:-1]
        v3 = padded_mask[:, 2:,   1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:  ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2,  1:-1]

        # Stack normals and validity masks
        normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
        valids  = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

        # Normalize normal vectors
        normals = F.normalize(normals, p=2, dim=-1, eps=eps)

    return normals, valids


def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """
    Filter loss tensor by keeping only values below a certain quantile threshold.
    
    This helps remove outliers that could destabilize training.
    
    Args:
        loss_tensor: Tensor containing loss values
        valid_range: Float between 0 and 1 indicating the quantile threshold
        min_elements: Minimum number of elements required to apply filtering
        hard_max: Maximum allowed value for any individual loss
    
    Returns:
        Filtered and clamped loss tensor
    """
    if loss_tensor.numel() <= min_elements:
        # Too few elements, just return as-is
        return loss_tensor

    # Randomly sample if tensor is too large to avoid memory issues
    if loss_tensor.numel() > 100000000:
        # Flatten and randomly select 1M elements
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]

    # First clamp individual values to prevent extreme outliers
    loss_tensor = loss_tensor.clamp(max=hard_max)

    # Compute quantile threshold
    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)

    # Apply quantile filtering if enough elements remain
    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    return loss_tensor


def torch_quantile(
    input,
    q,
    dim = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Handle dim=None case
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Set interpolation method
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Validate out parameter
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Compute k-th value
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Handle keepdim and dim=None cases
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out





def se3_inverse(T):
    """
    Computes the inverse of a batch of SE(3) matrices.
    T: Tensor of shape (B, 4, 4)
    """
    if len(T.shape) == 2:
        T = T[None]
        unseq_flag = True
    else:
        unseq_flag = False

    if torch.is_tensor(T):
        R = T[:, :3, :3]
        t = T[:, :3, 3].unsqueeze(-1)
        R_inv = R.transpose(-2, -1)
        t_inv = -torch.matmul(R_inv, t)
        T_inv = torch.cat([
            torch.cat([R_inv, t_inv], dim=-1),
            torch.tensor([0, 0, 0, 1], device=T.device, dtype=T.dtype).repeat(T.shape[0], 1, 1)
        ], dim=1)
    else:
        R = T[:, :3, :3]
        t = T[:, :3, 3, np.newaxis]

        R_inv = np.swapaxes(R, -2, -1)
        t_inv = -R_inv @ t

        bottom_row = np.zeros((T.shape[0], 1, 4), dtype=T.dtype)
        bottom_row[:, :, 3] = 1

        top_part = np.concatenate([R_inv, t_inv], axis=-1)
        T_inv = np.concatenate([top_part, bottom_row], axis=1)

    if unseq_flag:
        T_inv = T_inv[0]
    return T_inv


