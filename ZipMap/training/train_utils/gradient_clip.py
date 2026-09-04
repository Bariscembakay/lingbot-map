import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Union, Optional
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed._tensor import DTensor


class GradientClipper:
    """
    Gradient clipping utils that works for both FSDP and DDP with support for different
    clipping configurations for different parts of the model.
    """
    def __init__(self, configs, *args, **kwargs):
        """
        Args:
            configs: List of dictionaries, each containing:
                - module_name: str or list of str, module names to apply clipping to
                - max_norm: float, maximum norm for gradient clipping
                - norm_type: int, type of norm (default: 2)
        """
        self.configs = []
        self.params_to_clip_by_config = None
        self.is_initialized = False
        
        for config in configs:
            module_names = config['module_name']
            if isinstance(module_names, str):
                module_names = [module_names]
            
            self.configs.append({
                'module_names': module_names,
                'max_norm': float(config['max_norm']) if config['max_norm'] is not None else None,
                'norm_type': config.get('norm_type', 2)
            })

    def setup_clipping(self, model: nn.Module) -> None:
        """
        Set up gradient clipping by finding all parameters that should be clipped
        based on module names and validating that all parameters are covered.
        
        This should be called once at the beginning of training.
        
        Args:
            model: The model to set up gradient clipping for
        """
        # First, collect all parameters that should be clipped based on module names
        params_to_clip_by_config = []
        all_clipped_params = set()
        
        # Check if model is FSDP wrapped
        has_fsdp_modules = any(isinstance(submodule, FSDP) 
                        for name, submodule in model.named_children())
        is_fsdp = isinstance(model, FSDP) or has_fsdp_modules
        
        for config in self.configs:
            current_config_params = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # For FSDP, parameter names have _fsdp_wrapped_module prefix
                    # We need to check the original module name
                    clean_name = name
                    if is_fsdp and '_fsdp_wrapped_module.' in name:
                        clean_name = name.replace('_fsdp_wrapped_module.', '')
                    
                    for module_name in config['module_names']:
                        if module_name in clean_name:
                            current_config_params.append(param)
                            all_clipped_params.add(param)
                            break
            params_to_clip_by_config.append((config, current_config_params))

        # Check for remaining parameters
        remaining_params = []
        remaining_param_names = []
        for name, param in model.named_parameters():
            if param.requires_grad and param not in all_clipped_params:
                remaining_params.append(param)
                # Clean the name for FSDP for better debugging
                clean_name = name
                if is_fsdp and '_fsdp_wrapped_module.' in name:
                    clean_name = name.replace('_fsdp_wrapped_module.', '')
                remaining_param_names.append(clean_name)

        if len(remaining_params) > 0:
            print(f"Found {len(remaining_params)} parameters that won't be clipped")
            print("Parameter names:", remaining_param_names)
            raise ValueError("Some parameters are not configured for gradient clipping")
        
        # Store the computed parameters
        self.params_to_clip_by_config = params_to_clip_by_config
        self.is_initialized = True

    def __call__(self, model: nn.Module) -> Optional[torch.Tensor]:
        """
        Perform gradient clipping using the pre-computed parameter groups.
        
        Args:
            model: The model (not used, kept for backward compatibility)
            
        Returns:
            Dictionary of gradient norms for each configuration
        """
        if not self.is_initialized:
            raise RuntimeError("GradientClipper must be initialized with setup_clipping() before use")
        
        grad_norms = {}
        has_fsdp_modules = any(isinstance(submodule, FSDP) 
                        for name, submodule in model.named_children())
        is_fsdp = isinstance(model, FSDP) or has_fsdp_modules

        if is_fsdp:
            # Use efficient FSDP gradient clipping for each module
            for config_item in self.configs:
                module_names = config_item['module_names']
                max_norm = config_item['max_norm']
                norm_type = config_item['norm_type']

                for module_name in module_names:
                    if hasattr(model, module_name):
                        module = getattr(model, module_name)
                        grad_norm = module.clip_grad_norm_(
                            max_norm=max_norm,
                            norm_type=norm_type
                        )
                        grad_norms[module_name] = grad_norm.detach()
                    else:
                        print(f"Warning: Module {module_name} not found in FSDP model")
                        grad_norms[module_name] = 0.0

        else:
            for config, params_to_clip in self.params_to_clip_by_config:
                if not params_to_clip or config['max_norm'] is None:
                    continue

                grad_norm = nn.utils.clip_grad_norm_(
                    params_to_clip,
                    max_norm=config['max_norm'],
                    norm_type=config['norm_type']
                )

                if grad_norm is None:
                    continue

                grad_norms[",".join(config['module_names'])] = grad_norm.detach()

        return grad_norms

    @torch.no_grad()
    def _efficient_fsdp_clip_grad_norm(self, model: nn.Module, max_norm: float) -> torch.Tensor:
        """
        # Deprecated: Use FSDP's built-in clip_grad_norm_ instead.
        borrowed from: https://github.com/Kai-46/minFM/blob/main/utils/clip_grad.py#L9

        """
        shard_size, replicate_size = 1, dist.get_world_size()

  
        if hasattr(model, '_get_fsdp_state'): # FSDP2
            try:
                fsdp_state = model._get_fsdp_state()
                if hasattr(fsdp_state, '_fsdp_param_group') and hasattr(fsdp_state._fsdp_param_group, 'mesh_info'):
                    shard_size = fsdp_state._fsdp_param_group.mesh_info.shard_mesh_size
                    replicate_size = dist.get_world_size() // shard_size
            except:
                # Fallback to default values
                pass
        if hasattr(model, 'fsdp_handles'): # FSDP1
            try:
                fsdp_state = model.fsdp_handles[0]._get_fsdp_state()
                if hasattr(fsdp_state, '_fsdp_param_group') and hasattr(fsdp_state._fsdp_param_group, 'mesh_info'):
                    shard_size = fsdp_state._fsdp_param_group.mesh_info.shard_mesh_size
                    replicate_size = dist.get_world_size() // shard_size
            except:
                # Fallback to default values
                pass

        # Separate DTensor and non-DTensor parameters
        all_param_grads = []
        dtensor_param_grads = []
        regular_param_grads = []

        for p in model.parameters():
            if (not p.requires_grad) or (p.grad is None):
                continue

            if isinstance(p.grad, DTensor):
                local_p_grad = p.grad.to_local()
                dtensor_param_grads.append(local_p_grad.ravel())
            else:
                local_p_grad = p.grad
                regular_param_grads.append(local_p_grad.ravel())

            all_param_grads.append(local_p_grad)

        if not all_param_grads:
            return torch.tensor(0.0, device=model.device if hasattr(model, 'device') else torch.cuda.current_device())

        # Compute local squared sum properly (matches original math, avoids concat)
        local_sq_sum = torch.tensor(0.0, device=all_param_grads[0].device)

        if dtensor_param_grads:
            # Compute sum of squared norms for DTensor grads (sharded)
            dtensor_norms = torch._foreach_norm(dtensor_param_grads, 2.0)
            dtensor_sq_sum = sum(n ** 2 for n in dtensor_norms)
            local_sq_sum = local_sq_sum + dtensor_sq_sum

        if regular_param_grads:
            # Compute sum of squared norms for regular grads (possibly replicated)
            regular_norms = torch._foreach_norm(regular_param_grads, 2.0)
            regular_sq_sum = sum(n ** 2 for n in regular_norms)
            # Divide by shard_size to avoid double-counting replicated grads
            local_sq_sum = local_sq_sum + regular_sq_sum / shard_size

        # Single all-reduce operation
        global_sq_sum = local_sq_sum.clone()
        dist.all_reduce(global_sq_sum, op=dist.ReduceOp.SUM)
        global_sq_sum = global_sq_sum / replicate_size

        total_norm = global_sq_sum.sqrt()

        # Only apply clipping when exceeding threshold
        # Use torch.clamp to avoid CPU-GPU sync from boolean comparison
        clip_factor = torch.clamp(max_norm / (total_norm + 1e-8), max=1.0)
        # Only clip if factor < 1.0 (avoid no-op when not needed)
        if max_norm > 0:
            torch._foreach_mul_(all_param_grads, clip_factor)

        return total_norm