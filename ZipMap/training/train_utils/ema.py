# Code for ZipMap (CVPR 2026); created by Haian Jin

"""
Exponential Moving Average (EMA) utilities for PyTorch models.

This module provides a simplified EMA implementation that works with both
DDP and FSDP training setups.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from .logging import _format_param_count


class EMAParams:
    """Maintains EMA weights for trainable model parameters.

    This implementation stores EMA values for all trainable parameters and provides
    utilities for updating, swapping, and checkpointing EMA weights. It works with
    both DDP and FSDP by operating directly on parameter references.
    """

    def __init__(
        self,
        name_to_trainable_params: Dict[str, nn.Parameter],
        ema_weight: float,
        device: Optional[str] = None,
    ):
        """Initialize EMA tracking for trainable parameters.

        Args:
            name_to_trainable_params: Dictionary mapping parameter names to parameters
            ema_weight: EMA decay weight (typically 0.999)
            device: Device to store EMA parameters (e.g., 'cpu' to save GPU memory). If None, uses same device as model.
        """
        self.ema_weight = ema_weight
        self.name_to_trainable_params = name_to_trainable_params
        self.name_to_ema_params: Dict[str, torch.Tensor] = {}
        self.device = torch.device(device) if device is not None else None

        # Initialize EMA parameters as clones of current model parameters
        for name, param in name_to_trainable_params.items():
            ema_param = param.data.detach().clone()
            # Move to specified device if provided (e.g., CPU to save GPU memory)
            if self.device is not None:
                ema_param = ema_param.to(self.device)
            self.name_to_ema_params[name] = ema_param

        # Calculate and log parameter statistics
        total_elems = sum(t.numel() for t in self.name_to_ema_params.values())
        total_readable = _format_param_count(total_elems)
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        logging.info(
            f"EMA initialized on rank {rank}: {len(self.name_to_ema_params)} tensors "
            f"(~{total_readable} params, decay={ema_weight}), dtype={ema_param.dtype}, device={ema_param.device}"
        )

    def update(self) -> None:
        """Update EMA values from current model parameters."""
        for name, ema_param in self.name_to_ema_params.items():
            # Filter out FSDP wrapped params by checking size
            # FSDP may have empty shards on some ranks
            if ema_param.numel() > 0:
                model_param = self.name_to_trainable_params[name]
                # Additional check: ensure model param also has data
                if model_param.numel() > 0:
                    # Move model param to EMA device if necessary
                    model_param_detached = model_param.detach()
                    if ema_param.device != model_param_detached.device:
                        model_param_detached = model_param_detached.to(ema_param.device)
                    ema_param.data.mul_(self.ema_weight).add_(
                        model_param_detached,
                        alpha=1 - self.ema_weight
                    )

    def copy_to_model(self) -> None:
        """Copy EMA parameters to the model (for evaluation/checkpointing)."""
        for name, ema_param in self.name_to_ema_params.items():
            if ema_param.numel() > 0:
                model_param = self.name_to_trainable_params[name]
                if model_param.numel() > 0:
                    # Move EMA param to model device if necessary
                    if ema_param.device != model_param.device:
                        model_param.data.copy_(ema_param.data.to(model_param.device))
                    else:
                        model_param.data.copy_(ema_param.data)

    def copy_from_model(self) -> None:
        """Copy current model parameters to EMA storage (for initialization)."""
        for name, ema_param in self.name_to_ema_params.items():
            if ema_param.numel() > 0:
                model_param = self.name_to_trainable_params[name]
                if model_param.numel() > 0:
                    # Move model param to EMA device if necessary
                    model_param_detached = model_param.data.detach()
                    if ema_param.device != model_param_detached.device:
                        ema_param.data.copy_(model_param_detached.to(ema_param.device))
                    else:
                        ema_param.data.copy_(model_param_detached)

    def cache_model(self, cpu: bool = False) -> None:
        """Cache current model parameters before swapping in EMA weights.

        Usage pattern for checkpointing:
            ema.cache_model()
            ema.copy_to_model()
            # Save model weights here
            ema.restore_model_from_cache()

        Args:
            cpu: If True, cache parameters on CPU to save GPU memory
        """
        cache_dict = {}
        for name, param in self.name_to_trainable_params.items():
            if cpu:
                cache_dict[name] = param.data.detach().cpu().clone()
            else:
                cache_dict[name] = param.data.detach().clone()

        self.cache_dict = cache_dict

    def restore_model_from_cache(self) -> None:
        """Restore model parameters from cache and clear the cache."""
        for name, param in self.name_to_trainable_params.items():
            param.data.copy_(self.cache_dict[name].to(param.device))

        # Clear cache to free memory
        self.cache_dict.clear()
        self.cache_dict = None

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return EMA state dict for checkpointing."""
        return {name: param.detach().cpu() for name, param in self.name_to_ema_params.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load EMA state from checkpoint.

        Args:
            state_dict: Dictionary mapping parameter names to EMA values
        """
        missing_keys = []
        unexpected_keys = []

        for name in self.name_to_ema_params.keys():
            if name in state_dict:
                self.name_to_ema_params[name].data.copy_(
                    state_dict[name].to(self.name_to_ema_params[name].device)
                )
            else:
                missing_keys.append(name)

        for name in state_dict.keys():
            if name not in self.name_to_ema_params:
                unexpected_keys.append(name)

        if missing_keys:
            logging.warning(f"EMA missing keys: {missing_keys}")
        if unexpected_keys:
            logging.warning(f"EMA unexpected keys: {unexpected_keys}")
        logging.info("EMA state loaded successfully")