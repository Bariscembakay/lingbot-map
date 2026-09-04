"""
Dataset debugging utilities that mirror the training preprocessing pipeline.
This allows dataloaders to apply the same preprocessing used during training for debugging/visualization.
"""
import numpy as np
import torch
import os
import sys

# Import preprocessing functions from training utils
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch


def process_batch_for_training_consistency(batch):
    """
    Process a batch with the same preprocessing used during training.
    This function mirrors the preprocessing in trainer.py _process_batch method.
    
    Args:
        batch: Dictionary containing batch data with keys like:
               - images: list/array of images
               - depths: list/array of depth maps
               - extrinsics: list/array of camera extrinsics
               - intrinsics: list/array of camera intrinsics
               - cam_points: list/array of camera coordinate points
               - world_points: list/array of world coordinate points
               - point_masks: list/array of point masks
               
    Returns:
        Processed batch dictionary with normalized camera extrinsics and points
    """
    # Convert numpy arrays to tensors if needed (exclude images as they're used for visualization)
    for key in ["depths", "extrinsics", "intrinsics", "cam_points", "world_points", "point_masks"]:
        if key in batch and isinstance(batch[key], (list, np.ndarray)):
            if isinstance(batch[key], list):
                # Convert list of arrays to single tensor
                if isinstance(batch[key][0], np.ndarray):
                    batch[key] = torch.from_numpy(np.stack(batch[key]))
                else:
                    batch[key] = torch.stack(batch[key])
            else:
                # Convert numpy array to tensor
                batch[key] = torch.from_numpy(batch[key])
    
    # Ensure tensors are on CPU for normalization function (required by normalize_camera_extrinsics_and_points_batch)
    for key in ["extrinsics", "cam_points", "world_points", "depths", "point_masks"]:
        if key in batch and isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].cpu()
    
    # Add batch dimension if needed (the normalization function expects batch dimension)
    needs_batch_dim = False
    if len(batch["extrinsics"].shape) == 3:  # (S, 3, 4) instead of (B, S, 3, 4)
        needs_batch_dim = True
        for key in ["extrinsics", "cam_points", "world_points", "depths", "point_masks"]:
            if key in batch and isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].unsqueeze(0)  # Add batch dimension
    
    # Normalize camera extrinsics and points using the same function as training
    # This centers the coordinate system at the first camera and optionally scales the scene
    normalized_extrinsics, normalized_extrinsics_c2w,  normalized_cam_points, normalized_world_points, normalized_depths = \
        normalize_camera_extrinsics_and_points_batch(
            extrinsics=batch["extrinsics"],
            has_extrinsics_c2w=True,
            cam_points=batch["cam_points"],
            world_points=batch["world_points"],
            depths=batch["depths"],
            point_masks=batch["point_masks"],
        )

    # Replace the original values in the batch with the normalized ones
    batch["extrinsics"] = normalized_extrinsics
    batch["cam_points"] = normalized_cam_points
    batch["world_points"] = normalized_world_points
    batch["depths"] = normalized_depths
    batch["extrinsics_c2w"] = normalized_extrinsics_c2w
    # Remove batch dimension if we added it
    if needs_batch_dim:
        for key in ["extrinsics", "extrinsics_c2w", "cam_points", "world_points", "depths", "point_masks"]:
            if key in batch and isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].squeeze(0)  # Remove batch dimension
    
    # Convert tensors back to numpy arrays for compatibility with existing visualization code
    # (images were never converted to tensors, so no need to convert them back)
    for key in ["extrinsics", "cam_points", "world_points", "depths", "point_masks"]:
        if key in batch and isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].cpu().numpy()
    
    return batch