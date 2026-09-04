# Code for ZipMap (CVPR 2026); created by Haian Jin


import os

# --- Environment Variable Setup for Performance and Debugging ---
# Helps with memory fragmentation in PyTorch's memory allocator.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Specifies the threading layer for MKL, can prevent hangs in some environments.
os.environ["MKL_THREADING_LAYER"] = "GNU"
# Provides full Hydra stack traces on error for easier debugging.
os.environ["HYDRA_FULL_ERROR"] = "1"
# Enables asynchronous error handling for NCCL, which can prevent hangs.
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ['NCCL_TIMEOUT'] = '1800'

import contextlib
import gc
import json
import logging
import time
from datetime import timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    StateDictOptions,
)
from torch.distributed.device_mesh import DeviceMesh

from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from train_utils.checkpoint import DDPCheckpointSaver
from train_utils.distributed import get_machine_local_and_dist_rank
from train_utils.freeze import freeze_modules
from train_utils.general import *
from train_utils.logging import setup_logging, _format_param_count
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch, select_camera_views, normalize_camera_c2w, get_ray_conditions
from train_utils.optimizer import construct_optimizers
from train_utils.ema import EMAParams

from zipmap.utils.pose_enc import pose_encoding_to_extri_intri
from zipmap.utils.geometry import unproject_depth_map_to_point_map, save_3d_points
from zipmap.utils.image import depth_to_np_arr, stack_images



class Trainer:
    """
    A generic trainer for DDP and FSDP training. This should naturally support multi-node training.

    This class orchestrates the entire training and validation process, including:
    - Setting up the distributed environment (DDP or FSDP).
    - Initializing the model, optimizers, loss functions, and data loaders.
    - Handling checkpointing for resuming training.
    - Executing the main training and validation loops.
    - Logging metrics and visualizations to TensorBoard.
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        device: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = {},
        cuda: Dict[str, bool] = {},
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        accum_steps: int = 1,
        **kwargs,
    ):
        """
        Initializes the Trainer.

        Args:
            data: Hydra config for datasets and dataloaders.
            model: Hydra config for the model.
            logging: Hydra config for logging (TensorBoard, log frequencies).
            checkpoint: Hydra config for checkpointing.
            max_epochs: Total number of epochs to train.
            mode: "train" for training and validation, "val" for validation only.
            device: "cuda" or "cpu".
            seed_value: A random seed for reproducibility.
            val_epoch_freq: Frequency (in epochs) to run validation.
            distributed: Hydra config for DDP settings.
            cuda: Hydra config for CUDA-specific settings (e.g., cuDNN).
            limit_train_batches: Limit the number of training batches per epoch (for debugging).
            limit_val_batches: Limit the number of validation batches per epoch (for debugging).
            optim: Hydra config for optimizers and schedulers.
            loss: Hydra config for the loss function.
            env_variables: Dictionary of environment variables to set.
            accum_steps: Number of steps to accumulate gradients before an optimizer step.
        """
        self._setup_env_variables(env_variables)
        self._setup_timers()

        # Store Hydra configurations
        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint
        self.optim_conf = optim
        self.distributed_conf = distributed
        # Store hyperparameters
        self.accum_steps = accum_steps
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.seed_value = seed_value

        # 'where' tracks training progress from 0.0 to 1.0 for schedulers
        self.where = 0.0

        self._configure_ema()

        self._setup_device(device)
        self._setup_torch_dist_and_backend(cuda, distributed)

        # Setup logging directory and configure logger
        safe_makedirs(self.logging_conf['log_dir'])
        setup_logging(
            __name__,
            output_dir=os.path.join(self.logging_conf['log_dir'], self.logging_conf['exp_name']),
            rank=self.rank,
            log_level_primary=self.logging_conf['log_level_primary'],
            log_level_secondary=self.logging_conf['log_level_secondary'],
            all_ranks=self.logging_conf['all_ranks'],
        )
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)

        assert is_dist_avail_and_initialized(), "Torch distributed needs to be initialized before calling the trainer."

        # Instantiate components (model, loss, etc.)
        self._setup_components()
        self._setup_dataloaders()

        # Move model to the correct device
        self.model.to(self.device)
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.4f")

        # Construct optimizers (after moving model to device)
        if self.mode != "val":
            self.optims = construct_optimizers(self.model, self.optim_conf)

        self._resume_from_checkpoint()

        # Ensure all parameters are properly contiguous after checkpoint loading
        for param in self.model.parameters():
            if param.data.is_cuda:
                param.data = param.data.contiguous()

        # Wrap the model with DDP or FSDP
        if distributed.get('use_fsdp', False):
            self._setup_fsdp_distributed_training(distributed, device)
            # Disable scaler for FSDP as it handles mixed precision internally
            self.scaler = None
            # Load stored optimizer state after FSDP setup
            self._load_fsdp_optimizer_state()
        else:
            self._setup_ddp_distributed_training(distributed, device)

        self._setup_ema_manager()

        dist.barrier(device_ids=[self.local_rank] if torch.cuda.is_available() else None)

    def _setup_timers(self):
        """Initializes timers for tracking total elapsed time."""
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0

    def _setup_env_variables(self, env_variables_conf: Optional[Dict[str, Any]]) -> None:
        """Sets environment variables from the configuration."""
        if env_variables_conf:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value
        logging.info(f"Environment:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}")

    def _setup_torch_dist_and_backend(self, cuda_conf: Dict, distributed_conf: Dict) -> None:
        """Initializes the distributed process group and configures PyTorch backends."""
        if torch.cuda.is_available():
            # Configure CUDA backend settings for performance
            torch.backends.cudnn.deterministic = cuda_conf['cudnn_deterministic']
            torch.backends.cudnn.benchmark = cuda_conf['cudnn_benchmark']
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf['allow_tf32']
            torch.backends.cudnn.allow_tf32 = cuda_conf['allow_tf32']

        # Initialize the DDP process group
        dist.init_process_group(
            backend=distributed_conf['backend'],
            timeout=timedelta(minutes=distributed_conf['timeout_mins'])
        )
        self.rank = dist.get_rank()


    def _load_resuming_checkpoint(self, ckpt_path: str, re_init_train: bool = False):  
        """Loads a checkpoint from the given path to resume training."""
        logging.info(f"Resuming training from {ckpt_path} (rank {self.rank})")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu", weights_only=True)

        # Load model state and immediately remove from checkpoint dict to free memory
        model_state_dict = checkpoint.pop("model") if "model" in checkpoint else checkpoint.copy()

        if self.ema_enabled and not re_init_train:
            self._stored_ema_state = checkpoint.pop("ema", None)
        else:
            self._stored_ema_state = None

        # if has optimizer state, load other training progress info, and the optimizer will be load later
        has_optimizer_in_ckpt = "optimizer" in checkpoint
        if has_optimizer_in_ckpt and not re_init_train:
            logging.info(f"Optimizer state found in checkpoint (rank {self.rank})")
            # Load training progress
            if "epoch" in checkpoint:
                self.epoch = checkpoint["epoch"] + 1  # Start from the next epoch
            
            self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)

            # Load AMP scaler state if available (only for DDP mode)
            if (self.optim_conf['amp'].get('enabled', False) and "scaler" in checkpoint and 
                self.scaler is not None):
                self.scaler.load_state_dict(checkpoint["scaler"])

            if "update_steps" in checkpoint:
                self.update_steps = checkpoint["update_steps"]


        missing, unexpected = self.model.load_state_dict(
            model_state_dict, strict=self.checkpoint_conf.get('strict', False)
        )
        # Init QKV matrix for TTT model from previous self-attention blocks if specified
        if self.model.ttt_config['init_ttt_from_pretrained'] and len(unexpected) > 0 and len(missing) > 0:
            # if "ttt_block" in missing and 'att' in unexpected:
            if 'aggregator.global_blocks.0.ttt_block.to_qkv.weight' in missing and 'aggregator.global_blocks.0.attn.qkv.weight' in unexpected:
                # Special handling for TTT model loading from pretrained non-TTT model
                logging.info("Init TTT model qkv matrix from pretrained non-TTT model")
                for i in range(len(self.model.aggregator.global_blocks)):
                    self.model.aggregator.global_blocks[i].ttt_block.to_qkv.weight.data = model_state_dict[f'aggregator.global_blocks.{i}.attn.qkv.weight'].to(self.device)
                    missing.remove(f'aggregator.global_blocks.{i}.ttt_block.to_qkv.weight')
                    unexpected.remove(f'aggregator.global_blocks.{i}.attn.qkv.weight')

                    self.model.aggregator.global_blocks[i].ttt_block.c_proj.weight.data = model_state_dict[f'aggregator.global_blocks.{i}.attn.proj.weight'].to(self.device)
                    missing.remove(f'aggregator.global_blocks.{i}.ttt_block.c_proj.weight')
                    unexpected.remove(f'aggregator.global_blocks.{i}.attn.proj.weight')

                    if self.model.ttt_config.params.bias:
                        self.model.aggregator.global_blocks[i].ttt_block.to_qkv.bias.data = model_state_dict[f'aggregator.global_blocks.{i}.attn.qkv.bias'].to(self.device)
                        missing.remove(f'aggregator.global_blocks.{i}.ttt_block.to_qkv.bias')
                        unexpected.remove(f'aggregator.global_blocks.{i}.attn.qkv.bias')

                        self.model.aggregator.global_blocks[i].ttt_block.c_proj.bias.data = model_state_dict[f'aggregator.global_blocks.{i}.attn.proj.bias'].to(self.device)
                        missing.remove(f'aggregator.global_blocks.{i}.ttt_block.c_proj.bias')
                        unexpected.remove(f'aggregator.global_blocks.{i}.attn.proj.bias')

        del model_state_dict
        gc.collect()

        if self.rank == 0:
            logging.info(f"Model state loaded")
            logging.info(f"------------------------------------------------")
            logging.info(f"Missing keys: {missing or 'None'}.")
            logging.info(f"------------------------------------------------")
            logging.info(f"Unexpected keys: {unexpected or 'None'}.")
            logging.info(f"------------------------------------------------")

        # Store optimizer state for later loading (after FSDP setup)
        if has_optimizer_in_ckpt and self.optims is not None and re_init_train is False:
            logging.info(f"Storing optimizer state dict for later loading (rank {self.rank})")

            # Check if we will use FSDP
            will_use_fsdp = self.distributed_conf.get('use_fsdp', False)

            if will_use_fsdp:
                # Store the optimizer state to load after FSDP setup
                self._stored_optimizer_state = checkpoint["optimizer"]
                logging.info(f"Optimizer state stored for FSDP loading after model wrapping")
            else:
                # For DDP, load immediately
                try:
                    if len(self.optims) == 1:
                        if self.optims[0] is not None:
                            self.optims[0].optimizer.load_state_dict(checkpoint["optimizer"])
                    else:
                        for i, optim in enumerate(self.optims):
                            if optim is not None:
                                optim.optimizer.load_state_dict(checkpoint["optimizer"][i])
                    logging.info(f"Successfully loaded optimizer state dict (rank {self.rank})")

                except (RuntimeError, KeyError, ValueError) as e:
                    logging.warning(f"Failed to load optimizer state dict (rank {self.rank}): {e}")
                    logging.warning("Skipping optimizer state loading and starting with fresh optimizer state.")
                    # Reset optimizer state to start fresh
                    for optim in self.optims:
                        if optim is not None:
                            optim.optimizer.state.clear()

                # Clear optimizer state from memory after loading
                gc.collect()

        elif re_init_train:
            logging.info("Re-Init Training, not loading optimizer state or training progress.")

        checkpoint.clear()
        del checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _load_fsdp_optimizer_state(self):
        """Load optimizer state after FSDP setup using new distributed checkpoint API."""
        if self._stored_optimizer_state is None or self.optims is None:
            return

        logging.info(f"Loading FSDP optimizer state dict (rank {self.rank})")

        try:
            # Prepare optimizers and state for the new API
            optimizers_to_load = [optim.optimizer for optim in self.optims]

            # set_state_dict will correctly load and shard the full optimizer state
            # across all ranks. All ranks must call this function.
            set_state_dict(
                model=self.model,
                optimizers=optimizers_to_load,
                model_state_dict={},
                optim_state_dict=self._stored_optimizer_state
            )

            logging.info(f"Successfully loaded FSDP optimizer state dict (rank {self.rank})")

        except (RuntimeError, KeyError, ValueError) as e:
            logging.warning(f"Failed to load FSDP optimizer state dict (rank {self.rank}): {e}")
            logging.warning("Skipping optimizer state loading and starting with fresh optimizer state.")
            for optim in self.optims:
                if optim is not None:
                    optim.optimizer.state.clear()
        finally:
            if self._stored_optimizer_state is not None:
                # Clear the dict contents first if it's a dict
                if isinstance(self._stored_optimizer_state, dict):
                    self._stored_optimizer_state.clear()
                del self._stored_optimizer_state
                self._stored_optimizer_state = None

            # Synchronize all ranks and clean up memory
            dist.barrier()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    def _setup_device(self, device: str):
        """Sets up the device for training (CPU or CUDA)."""
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if device == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device: {device}")

    def _setup_components(self):
        """Initializes all core training components using Hydra configs."""
        logging.info("Setting up components: Model, Loss, Logger, etc.")
        self.epoch = 0
        self.update_steps = 0

        # Instantiate logger (WandB or TensorBoard)
        self.logger = instantiate(self.logging_conf['logging_writer'], _recursive_=False)
        
        self.model = instantiate(self.model_conf, _recursive_=False)
        self.loss = instantiate(self.loss_conf, _recursive_=False)
        self.gradient_clipper = instantiate(self.optim_conf['gradient_clip']) if self.optim_conf and 'gradient_clip' in self.optim_conf else None
        self.scaler = None
        if self.optim_conf['amp']['enabled'] and self.distributed_conf.get('use_fsdp', False) is False:
            self.scaler = torch.amp.GradScaler("cuda", enabled=True)
        # Freeze specified model parameters if any
        if self.optim_conf.get('frozen_module_names', None):
            logging.info(
                f"[Start] Freezing modules: {self.optim_conf['frozen_module_names']} on rank {self.distributed_rank}"
            )
            self.model = freeze_modules(
                self.model,
                patterns=self.optim_conf['frozen_module_names'],
            )
            logging.info(
                f"[Done] Freezing modules: {self.optim_conf['frozen_module_names']} on rank {self.distributed_rank}"
            )

        # Log model summary on rank 0
        if self.rank == 0:
            model_summary_path = os.path.join(self.logging_conf['log_dir'], self.logging_conf['exp_name'], "model.txt")
            model_summary(self.model, log_file=model_summary_path)
            logging.info(f"Model summary saved to {model_summary_path}")

        logging.info("Successfully initialized training components.")

    def _setup_dataloaders(self):
        """Initializes train and validation datasets and dataloaders."""
        self.train_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(
                self.data_conf.get('val', None), _recursive_=False
            )
            if self.val_dataset is not None:
                self.val_dataset.seed = self.seed_value

        if self.mode in ["train"]:
            self.train_dataset = instantiate(self.data_conf['train'], _recursive_=False)
            self.train_dataset.seed = self.seed_value

    def _configure_ema(self) -> None:
        """Initialise EMA-related attributes from the optimizer config."""
        ema_conf = {}
        if self.optim_conf is not None:
            ema_conf = self.optim_conf.get('ema', {}) or {}

        self.ema_enabled = bool(ema_conf.get('enabled', False)) and self.mode == "train"
        if self.ema_enabled:
            self.ema_decay = float(ema_conf.get('decay', 0.999))
            self.ema_start_step = int(ema_conf.get('start_step', 0))
            self.ema_update_interval = max(1, int(ema_conf.get('update_interval', 1) or 1))
            self.ema_device = ema_conf.get('device', None)
        else:
            self.ema_decay = None
            self.ema_start_step = 0
            self.ema_update_interval = 1
            self.ema_device = None

        self.ema = None
        self._stored_ema_state = None

    def _resume_from_checkpoint(self) -> None:
        """Handle any checkpoint resumption logic prior to wrapping the model."""
        dist.barrier(device_ids=[self.local_rank] if torch.cuda.is_available() else None)
        self._stored_optimizer_state = None

        # If the logging directory already has a checkpoint, it means the previous run might have been interrupted, 
        # so we try to resume from that checkpoint to avoid losing training progress. 
        ckpt_path = get_resume_checkpoint(self.checkpoint_conf['save_dir'])
        if ckpt_path is not None:
            self._load_resuming_checkpoint(ckpt_path, re_init_train=False)  # "re_init_train=False" means we try to resume the optimizer
            return

        resume_path = self.checkpoint_conf.get('resume_checkpoint_path', None)
        if resume_path is not None:
            assert g_pathmgr.exists(resume_path), f"Specified resume checkpoint path does not exist: {resume_path}. Check the path and try again."
            re_init_train = self.checkpoint_conf.get('re_init_train', False)
            # All ranks load the checkpoint from the specified path
            self._load_resuming_checkpoint(resume_path, re_init_train=re_init_train)

    def _setup_ema_manager(self) -> None:
        """Initialise EMA tracking once the model is wrapped for training."""
        if not self.ema_enabled:
            self.ema = None
            self._stored_ema_state = None
            return

        # For DDP, unwrap to get the underlying model
        # For FSDP, use the wrapped model directly (FSDP doesn't have .module)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model_for_ema = self.model.module
        else:
            # This handles both FSDP and non-distributed cases
            model_for_ema = self.model

        try:
            # Collect trainable parameters
            name_to_trainable_params = {
                name: param for name, param in model_for_ema.named_parameters()
                if param.requires_grad
            }

            self.ema = EMAParams(
                name_to_trainable_params=name_to_trainable_params,
                ema_weight=self.ema_decay if self.ema_decay is not None else 0.999,
                device=self.ema_device,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logging.error(f"Failed to set up EMA: {exc}")
            self.ema = None
            self.ema_enabled = False
            return

        # Load EMA state from checkpoint if available
        if self._stored_ema_state is not None:
            # Check if we're using FSDP
            has_fsdp_modules = any(isinstance(submodule, FSDP)
                                  for submodule in self.model.modules())

            if has_fsdp_modules:
                # FSDP-aware loading: use set_state_dict to properly shard the full EMA tensors
                logging.info("Loading EMA state for FSDP model...")

                # Backup current model parameters
                model_backup = {
                    name: param.data.clone()
                    for name, param in model_for_ema.named_parameters()
                    if param.requires_grad
                }

                # Load full EMA state into model (set_state_dict handles sharding)
                load_options = StateDictOptions(full_state_dict=True, cpu_offload=False)
                set_state_dict(
                    model=self.model,
                    optimizers=[],
                    model_state_dict=self._stored_ema_state,
                    optim_state_dict={},
                    options=load_options
                )

                # Copy sharded EMA weights from model to EMA storage
                self.ema.copy_from_model()

                # Restore original model parameters
                for name, param in model_for_ema.named_parameters():
                    if name in model_backup:
                        param.data.copy_(model_backup[name])

                model_backup.clear()
                del model_backup
                gc.collect()
                if self.rank == 0:
                    logging.info("FSDP EMA state loaded successfully")
            else:
                # DDP/single-GPU: direct loading works fine
                self.ema.load_state_dict(self._stored_ema_state)
                if self.rank == 0:
                    logging.info("Loaded EMA state from checkpoint")

        # Note: EMA will be (re-)initialized from model at ema_start_step during training
        # This ensures EMA starts with current trained weights, not stale initialization
        if self._stored_ema_state is not None:
            if isinstance(self._stored_ema_state, dict):
                self._stored_ema_state.clear()
            del self._stored_ema_state
            self._stored_ema_state = None

        gc.collect()

    def _setup_ddp_distributed_training(self, distributed_conf: Dict, device: str):
        """Wraps the model with DistributedDataParallel (DDP)."""
        assert isinstance(self.model, torch.nn.Module)

        ddp_options = dict(
            find_unused_parameters=distributed_conf['find_unused_parameters'],
            gradient_as_bucket_view=distributed_conf['gradient_as_bucket_view'],
            bucket_cap_mb=distributed_conf['bucket_cap_mb'],
            broadcast_buffers=distributed_conf['broadcast_buffers'],
        )

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if device == "cuda" else [],
            **ddp_options,
        )

    def _setup_fsdp_distributed_training(self, distributed_conf: Dict, device: str):
        """Wraps the model with FullyShardedDataParallel (FSDP)."""
        assert isinstance(self.model, torch.nn.Module)

        # Configure mixed precision for FSDP
        mixed_precision = None
        safe_precision = MixedPrecision(
                    param_dtype=torch.float32,  # Keep parameters in FP32
                    reduce_dtype=torch.float32,  # Use FP32 for gradient reduction
                    buffer_dtype=torch.float32,  # Keep buffers in FP32
                )
        if self.optim_conf['amp']['enabled']:
            amp_dtype = self.optim_conf['amp'].get('amp_dtype', 'bfloat16')
            if amp_dtype == 'bfloat16':
                # Only convert reduce operations, keep params in FP32 to avoid dtype mismatches
                mixed_precision = MixedPrecision(
                    param_dtype=torch.bfloat16,  # bf16 parameters
                    reduce_dtype=torch.float32,  # Use FP32 for gradient reduction
                    buffer_dtype=torch.float32,  # Keep buffers in FP32
                )
            else:
                mixed_precision = safe_precision

        # Configure backward prefetch
        backward_prefetch = BackwardPrefetch.BACKWARD_PRE
        if distributed_conf.get('fsdp_backward_prefetch', 'backward_pre') == 'backward_post':
            backward_prefetch = BackwardPrefetch.BACKWARD_POST

        # Configure sharding strategy
        fsdp_strategy_dict = {
            "full_shard": ShardingStrategy.FULL_SHARD,
            "hybrid_full": ShardingStrategy.HYBRID_SHARD, # do FULL_SHARD within node, and NO_SHARD across nodes
            "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2, # do SHARD_GRAD_OP within node, and NO_SHARD across nodes
            "no_shard": ShardingStrategy.NO_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        }

        fsdp_strategy = fsdp_strategy_dict[distributed_conf.get('fsdp_sharding_strategy', 'full_shard')]

        
        # Wrap individual modules with FSDP for per-module gradient clipping
        fsdp_kwargs = {
            "backward_prefetch": backward_prefetch,
            "device_id": torch.cuda.current_device() if device == "cuda" else None,
            "use_orig_params": True,
            "sharding_strategy": fsdp_strategy,
            "sync_module_states": True,
            "param_init_fn": None,
        }
        
        # For hybrid strategies, we can use DeviceMesh for cleaner process group management
        if fsdp_strategy in [ShardingStrategy._HYBRID_SHARD_ZERO2, ShardingStrategy.HYBRID_SHARD]:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            
            # For single node, create a 2D mesh with 1 "node" and all GPUs
            gpus_per_node = int(os.environ.get('LOCAL_WORLD_SIZE', torch.cuda.device_count()))
            if world_size <= gpus_per_node:  # Single node if world_size <= GPUs per node
                # Create 2D mesh: [1 node, all GPUs] to satisfy hybrid strategy requirements
                mesh_shape = (1, world_size)
                device_mesh = DeviceMesh("cuda", torch.arange(world_size).reshape(mesh_shape))
                fsdp_kwargs["device_mesh"] = device_mesh
                logging.info(f"Single node detected, using 2D DeviceMesh: {mesh_shape}, {device_mesh} (rank {rank})")
            else:
                # Multi-node setup: create 2D mesh [num_nodes, gpus_per_node]
                num_nodes = world_size // gpus_per_node
                
                # Create 2D mesh: first dimension is replicated across nodes, 
                # second dimension is sharded within nodes
                mesh_shape = (num_nodes, gpus_per_node)
                device_mesh = DeviceMesh("cuda", torch.arange(world_size).reshape(mesh_shape))
                fsdp_kwargs["device_mesh"] = device_mesh
                
                logging.info(f"Multi-node detected, using 2D DeviceMesh: {mesh_shape}, {device_mesh} (rank {rank})")
        
        # Wrap specific modules that need separate gradient clipping
        modules_to_wrap = self.distributed_conf.get('fsdp_modules_to_wrap', [])
        safe_fsdp_precision_modules = self.distributed_conf.get('safe_fsdp_precision_modules', ['point_head'])
        for module_name in modules_to_wrap:
            if hasattr(self.model, module_name):
                module = getattr(self.model, module_name)
                if module is None:
                    logging.warning(f"Module {module_name} is None, skipping FSDP wrapping.")
                    continue
                # 1. Find all parameters within the module that are frozen.
                ignored_params = [p for p in module.parameters() if not p.requires_grad]
                if ignored_params:
                    logging.info(f"Module '{module_name}' has frozen parameters that will be passed to ignored_states.")
                    ignore_param_size = sum(p.numel() for p in ignored_params)
                    ignore_param_size = _format_param_count(ignore_param_size)
                    logging.info(f"Number of frozen parameters in '{module_name}': {ignore_param_size} elements.")
                    # 2. Add the list of ignored parameters to the FSDP config.
                    fsdp_kwargs["ignored_states"] = ignored_params

                if module_name in safe_fsdp_precision_modules:
                    current_precision = safe_precision
                    logging.info(f"Using safe FSDP precision for module '{module_name}'")
                else:
                    current_precision = mixed_precision
                    logging.info(f"Using AMP FSDP precision for module '{module_name}'")
                wrapped_module = FSDP(module, mixed_precision=current_precision, **fsdp_kwargs)
                
                # Important: Reset ignored_states for the next module in the loop
                if "ignored_states" in fsdp_kwargs:
                    del fsdp_kwargs["ignored_states"]

                setattr(self.model, module_name, wrapped_module)


    def save_checkpoint(self, epoch: int, checkpoint_names: Optional[List[str]] = None):
        """
        Saves a training checkpoint.

        Args:
            epoch: The current epoch number.
            checkpoint_names: A list of names for the checkpoint file (e.g., "checkpoint_latest").
                              If None, saves "checkpoint" and "checkpoint_{epoch}" on frequency.
        """
        checkpoint_folder = self.checkpoint_conf['save_dir']
        safe_makedirs(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf['save_freq'] > 0
                and int(epoch) % self.checkpoint_conf['save_freq'] == 0
                and (int(epoch) > 0 or self.checkpoint_conf['save_freq'] == 1)
            ):
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        # Check if any module is FSDP wrapped before creating checkpoint content
        # Use model.modules() to check all nested modules, not just direct children
        has_fsdp_modules = any(isinstance(m, FSDP) for m in self.model.modules())
        # Only save EMA if it's enabled and has started tracking (reached ema_start_step)
        save_ema = self.ema_enabled and self.ema is not None and self.update_steps >= self.ema_start_step
        
        checkpoint_content = {
            "epoch": epoch,
            "update_steps": self.update_steps,
            "time_elapsed": self.time_elapsed_meter.val,
        }
        
        # Handle optimizer state dict differently for FSDP vs DDP
        if self.optims is not None:
            if has_fsdp_modules:
                # For FSDP, we need to collect optimizer state using FSDP's full state dict API
                # This will be handled in the FSDP-specific section below
                optimizer_state_dict = None
            else:
                # For DDP, use regular state dict
                optimizer_state_dict = [optim.optimizer.state_dict() for optim in self.optims]
                if len(self.optims) == 1:
                    optimizer_state_dict = optimizer_state_dict[0]
                checkpoint_content["optimizer"] = optimizer_state_dict
        # Only save scaler state for DDP mode
        if (self.optim_conf['amp'].get('enabled', False) and self.scaler is not None):
            checkpoint_content["scaler"] = self.scaler.state_dict()

        if has_fsdp_modules:
            dist.barrier()

            # Define options for the new state_dict API
            save_options = StateDictOptions(full_state_dict=True, cpu_offload=True) # default to gather to rank0 only

            # Get model and optimizer state dicts using new API - all ranks call this
            optimizers = [o.optimizer for o in self.optims] if self.optims is not None else []
            model_state_dict_result, optim_state_dict_result = get_state_dict(
                self.model,
                optimizers=optimizers,
                options=save_options
            )

            model_state_dict = model_state_dict_result if self.distributed_rank == 0 else {}

            if self.optims is not None and self.distributed_rank == 0:
                checkpoint_content["optimizer"] = optim_state_dict_result

            # Handle EMA state - swap entire model at once for simplicity
            ema_state_dict = None
            if save_ema:
                # Cache current model params, swap in EMA, get state dict, then restore
                self.ema.cache_model(cpu=True)
                self.ema.copy_to_model()

                ema_state_dict, _ = get_state_dict(
                    self.model,
                    optimizers=[],
                    options=save_options
                )

                self.ema.restore_model_from_cache()

            dist.barrier()
            # Only save on rank 0 after all ranks have synchronized
            if self.distributed_rank == 0:
                checkpoint_content["model"] = model_state_dict
                if save_ema and ema_state_dict is not None:
                    checkpoint_content["ema"] = {key: value.cpu() for key, value in ema_state_dict.items()}
                
                # Save checkpoint only on rank 0
                for ckpt_name in checkpoint_names:
                    checkpoint_path = os.path.join(checkpoint_folder, f"{ckpt_name}.pt")
                    logging.info(f"Saving FSDP checkpoint at epoch {epoch} to {checkpoint_path}")
                    with g_pathmgr.open(checkpoint_path, "wb") as f:
                        torch.save(checkpoint_content, f)
                # delete local checkpoint_{epoch - save_freq} if exists to save disk space
                if f"checkpoint_{int(epoch)}" in checkpoint_names:
                    if self.checkpoint_conf['save_freq'] > 0 and epoch - self.checkpoint_conf['save_freq'] >= 0:
                        ckpt_to_delete = os.path.join(checkpoint_folder, f"checkpoint_{int(epoch - self.checkpoint_conf['save_freq'])}.pt")
                        if g_pathmgr.exists(ckpt_to_delete):
                            logging.info(f"Deleting old checkpoint to save disk space: {ckpt_to_delete}")
                            g_pathmgr.rm(ckpt_to_delete)

                # Clean up checkpoint content on rank 0 after saving
                checkpoint_content.clear()
                model_state_dict.clear()
                del optim_state_dict_result

                if save_ema and ema_state_dict is not None:
                    ema_state_dict.clear()
                    del ema_state_dict

            # Synchronize again after saving
            dist.barrier()

            # Clean up on all ranks
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        else:
            # DDP checkpoint saving
            saver = DDPCheckpointSaver(
                checkpoint_folder,
                checkpoint_names=checkpoint_names,
                rank=self.distributed_rank,
                epoch=epoch,
            )

            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                model = self.model.module
            else:
                model = self.model

            if save_ema:
                checkpoint_content["ema"] = self.ema.state_dict()

            saver.save_checkpoint(
                model=model,
                skip_saving_parameters=[],
                **checkpoint_content,
            )
            


    def _get_scalar_log_keys(self, phase: str) -> List[str]:
        """Retrieves keys for scalar values to be logged for a given phase."""
        if self.logging_conf is not None and 'scalar_keys_to_log' in self.logging_conf and self.logging_conf['scalar_keys_to_log']:
            return self.logging_conf['scalar_keys_to_log'][phase]['keys_to_log']
        return []

    def run(self):
        """Main entry point to start the training or validation process."""
        assert self.mode in ["train", "val"], f"Invalid mode: {self.mode}"
        if self.mode == "train":
            self.run_train()
            # Optionally run a final validation after all training is done
            self.run_val()
        elif self.mode == "val":
            self.run_val()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def run_train(self):
        """Runs the main training loop over all epochs."""
        while self.epoch < self.max_epochs:
            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, self.distributed_rank)
            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank)) if self.train_dataset is not None else None
            if dataloader is not None:
                # clean up before each epoch to avoid memory accumulation
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                self.train_epoch(dataloader)
                del dataloader
                gc.collect()
                # clean up after each epoch to avoid memory accumulation
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                # Save checkpoint after each training epoch
                self.save_checkpoint(self.epoch)

            # Run validation at the specified frequency
            # Skips validation after the last training epoch, as it can be run separately.
            if self.epoch % self.val_epoch_freq == 0 and self.epoch < self.max_epochs - 1:
                self.run_val()
            self.epoch += 1
        self.epoch -= 1

    @torch.no_grad()
    def run_val(self):
        """Runs a full validation epoch if a validation dataset is available."""
        if not self.val_dataset:
            logging.info("No validation dataset configured. Skipping validation.")
            return
        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch)) if self.val_dataset is not None else None
        if dataloader is not None:
            self.val_epoch(dataloader)
            del dataloader
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


    @torch.no_grad()
    def val_epoch(self, val_loader):
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'val'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        progress = ProgressMeter(
            num_batches=len(val_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        self.model.eval()
        end = time.time()

        iters_per_epoch = len(val_loader)
        limit_val_batches = (
            iters_per_epoch
            if self.limit_val_batches is None
            else self.limit_val_batches
        )

        for data_iter, batch in enumerate(val_loader):
            if data_iter >= limit_val_batches:
                break
            
            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)
            
            with torch.amp.autocast("cuda", enabled=False):
                batch = self._process_batch(batch)
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            amp_type = self.optim_conf['amp'].get('amp_dtype', 'bfloat16')
            assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
            if amp_type == "bfloat16":
                amp_type = torch.bfloat16
            else:
                amp_type = torch.float16

            # compute output
            with torch.no_grad():
                with torch.amp.autocast(
                    "cuda",
                    enabled=self.optim_conf['amp']['enabled'],
                    dtype=amp_type,
                ):
                    query_info = batch['query_cond']
                    y_hat = self.model(images=batch["input_images"], query_info=query_info)
                    loss_dict = self.loss(y_hat, batch)
                    val_loss_dict= {**loss_dict, **y_hat, **batch}

            self._update_and_log_scalars(val_loss_dict, phase, loss_meters, batch_size=batch["input_images"].shape[0], log_to_wandb=False)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)


            if data_iter % self.logging_conf['log_visual_frequency'][phase] == 0:
                self._log_visual_locally(val_loss_dict, phase, self.update_steps + data_iter)

        # --- Log validation metrics to wandb at the end of the epoch ---
        if self.rank == 0 and hasattr(self, 'logger') and self.logger is not None:
            val_log_dict = {}
            for name, meter in loss_meters.items():
                val_name = name.replace("Loss/val_", "val/")
                val_log_dict[val_name] = meter.avg
            # Optionally add other meters (batch_time, data_time, mem, etc.)
            val_log_dict["val/BatchTime"] = batch_time.avg
            val_log_dict["val/DataTime"] = data_time.avg
            val_log_dict["val/MemGB"] = mem.avg
            self.logger.log_dict(val_log_dict, step=self.update_steps)

        return True

    def train_epoch(self, train_loader):
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'train'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        if self.gradient_clipper is not None:
            for config in self.gradient_clipper.configs: 
                param_names = ",".join(config['module_names'])
                loss_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", self.device, ":.4f")
                

        progress = ProgressMeter(
            num_batches=len(train_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()
        end = time.time()

        iters_per_epoch = len(train_loader)
        limit_train_batches = (
            iters_per_epoch
            if self.limit_train_batches is None
            else self.limit_train_batches
        )
        
        if self.gradient_clipper is not None:
            # setup gradient clipping at the beginning of training
            self.gradient_clipper.setup_clipping(self.model)

        for data_iter, batch in enumerate(train_loader):
            if data_iter >= limit_train_batches:
                break
            
            # Ensure gradients are zeroed for the first iteration
            if data_iter == 0:
                for optim in self.optims:   
                    optim.zero_grad(set_to_none=True)
            
            # measure data loading time
            current_data_time = time.time() - end
            data_time.update(current_data_time)
            data_times.append(current_data_time)

            if self.rank == 0:
                self.logger.log(f"Detail/Data Time", current_data_time, self.update_steps)

            with torch.amp.autocast("cuda", enabled=False):
                batch = self._process_batch(batch)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)


            accum_steps = self.accum_steps

            if accum_steps==1:
                chunked_batches = [batch]
            else:
                chunked_batches = chunk_batch_for_accum_steps(batch, accum_steps)

            self._run_steps_on_batch_chunks(
                chunked_batches, phase, loss_meters
            )
            # compute gradient and do SGD step
            assert data_iter < limit_train_batches  # allow for off by one errors
            exact_epoch = self.epoch + float(data_iter) / limit_train_batches
            self.where = float(exact_epoch) / self.max_epochs
            
            assert self.where <= 1 + self.EPSILON
            if self.where < 1.0:
                for optim in self.optims:
                    optim.step_schedulers(self.where)
            else:
                logging.warning(
                    f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                )
                    
            # Log schedulers
            if self.update_steps % self.logging_conf['log_freq'] == 0:
                for i, optim in enumerate(self.optims):
                    for j, param_group in enumerate(optim.optimizer.param_groups):
                        for option in optim.schedulers[j]:
                            optim_prefix = (
                                f"{i}_"
                                if len(self.optims) > 1
                                else (
                                    "" + f"{j}_"
                                    if len(optim.optimizer.param_groups) > 1
                                    else ""
                                )
                            )
                            self.logger.log(
                                os.path.join("Detail", f"{optim_prefix}", option),
                                param_group[option],
                                self.update_steps,
                            )
                self.logger.log(
                    os.path.join("Detail", "where"),
                    self.where,
                    self.update_steps,
                )
            skip_iter = False
            # Clipping gradients and detecting diverging gradients
            if self.gradient_clipper is not None:
                # Only unscale if using scaler (DDP mode)
                if self.scaler is not None:
                    for optim in self.optims:
                        self.scaler.unscale_(optim.optimizer)

                grad_norm_dict = self.gradient_clipper(model=self.model)

                for key, grad_norm in grad_norm_dict.items():

                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        grad_norm_val = grad_norm.item()  # Only sync if we detect a problem
                        error_msg = f"Gradient norm for {key} is {grad_norm_val}, skipping current iteration"
                        logging.warning(error_msg)
                        skip_iter = True
                        loss_meters[f"Grad/{key}"].update(0.0)
                        continue

                    loss_meters[f"Grad/{key}"].update(grad_norm)
                    if phase == "train" and self.rank == 0:
                        self.logger.log(f"Grad/{key}", grad_norm, self.update_steps)

            # Optimizer step
            if not skip_iter:
                for optim in self.optims:   
                    if self.scaler is not None:
                        self.scaler.step(optim.optimizer)
                    else:
                        # FSDP mode - direct optimizer step
                        optim.optimizer.step()
                
                if self.scaler is not None:
                    self.scaler.update()

            if self.ema_enabled and self.ema is not None:
                current_step = self.update_steps
                if current_step >= self.ema_start_step:
                    # Re-initialize EMA with current model weights at the start step
                    if current_step == self.ema_start_step:
                        self.ema.copy_from_model()
                        if self.rank == 0:
                            logging.info(f"Re-initialized EMA weights from model at step {current_step}")

                    if (current_step - self.ema_start_step) % self.ema_update_interval == 0:
                        self.ema.update()

            # Clear gradients after optimizer step for better memory management
            # and mixed precision compatibility
            for optim in self.optims:   
                optim.zero_grad(set_to_none=True)

            # Measure elapsed time
            current_batch_time = time.time() - end
            batch_time.update(current_batch_time)
            if phase == "train" and self.rank == 0:
                self.logger.log(f"Detail/Batch Time", current_batch_time, self.update_steps)
            end = time.time()
            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )
            mem_usage = torch.cuda.max_memory_allocated() // 1e9    
            mem.update(mem_usage)

            if phase == "train" and self.rank == 0:
                self.logger.log(f"Detail/Mem (GB)", mem_usage, self.update_steps)
            if data_iter % self.logging_conf['log_freq'] == 0:
                progress.display(data_iter)

            self.update_steps += 1

        return True

    def _run_steps_on_batch_chunks(
        self,
        chunked_batches: List[Any],
        phase: str,
        loss_meters: Dict[str, AverageMeter],
    ):
        """
        Run the forward / backward as many times as there are chunks in the batch,
        accumulating the gradients on each backward
        """        
        # Note: zero_grad is now called at the END of each training iteration
        # for better memory management and mixed precision compatibility
        # for optim in self.optims:   
        #     optim.zero_grad(set_to_none=True)

        accum_steps = len(chunked_batches)

        should_collect_logs = self.rank == 0
        temp_return_dict_list: Optional[List[Mapping[str, Any]]] = [] if should_collect_logs else None
        amp_type = self.optim_conf['amp'].get('amp_dtype', 'bfloat16')
        assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
        if amp_type == "bfloat16":
            amp_type = torch.bfloat16
        else:
            amp_type = torch.float16

        for i, chunked_batch in enumerate(chunked_batches):
            # Handle gradient synchronization for DDP only - FSDP handles this automatically
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                sync_context = (
                    self.model.no_sync()
                    if i < accum_steps - 1
                    else contextlib.nullcontext()
                )
            else:
                # FSDP and single-GPU models don't need manual sync control
                sync_context = contextlib.nullcontext()

            with sync_context:
                with torch.amp.autocast(
                    "cuda",
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):  

                    query_info = chunked_batch["query_cond"]
                    y_hat = self.model(images=chunked_batch["input_images"], query_info=query_info)
                    loss_dict = self.loss(y_hat, chunked_batch)
                    

                loss = loss_dict["loss_objective"]
                if torch.isnan(loss) or torch.isinf(loss):
                    # Only sync to CPU when we actually detect a problem (rare case)
                    loss_val = loss.item()
                    for loss_name, loss_value in loss_dict.items():
                        if isinstance(loss_value, torch.Tensor) and loss_name.startswith("loss"):
                            logging.error(f"{loss_name}: {loss_value.item()}")
                    error_msg = f"Loss is {loss_val}, attempting to stop training"
                    logging.error(error_msg)
                    return
                # torch.distributed.barrier()
                
                loss /= accum_steps
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    # FSDP mode - direct backward pass
                    loss.backward()

                if should_collect_logs:
                    temp_return_dict_list.append(
                        self._prepare_log_data(y_hat, loss_dict, chunked_batch, phase)
                    )


        if should_collect_logs:
            combined_return_dict = self._combine_return_dict_list(temp_return_dict_list)
            batch_size = combined_return_dict["input_images"].shape[0]
            self._update_and_log_scalars(combined_return_dict, phase, loss_meters, batch_size)
            if self.update_steps % self.logging_conf['log_visual_frequency'][phase] == 0:
                # Create minimal visual data to prevent memory leaks
                visual_data = {
                    k: v for k, v in combined_return_dict.items()
                    if k in ['input_images', 'depth', 'depths', 'seq_name', 'pose_enc', "nvs_pred", "nvs_target_images", "nvs_target_depths", "nvs_target_point_masks"]
                }
                self._log_visual_locally(visual_data, phase, self.update_steps)
                # Explicit cleanup of visual data
                del visual_data
            
            # Explicit cleanup of combined dictionary
            del combined_return_dict

        if should_collect_logs:
            del temp_return_dict_list

        
    def _apply_batch_repetition(self, batch: Mapping) -> Mapping:
        """
        Applies a data augmentation by concatenating the original batch with a
        flipped version of itself.
        """
        tensor_keys = [
            "images", "depths", "extrinsics", "intrinsics", 
            "cam_points", "world_points", "point_masks", 
        ]        
        string_keys = ["seq_name", "dataset_name"]
        

        for key in tensor_keys:
            if key in batch:
                original_tensor = batch[key]
                batch[key] = torch.concatenate([original_tensor, 
                                                torch.flip(original_tensor, dims=[1])], 
                                                dim=0)
        
        for key in string_keys:
            if key in batch:
                batch[key] = batch[key] * 2
        
        return batch
    
    def _combine_return_dict_list(self, return_dict_list: List[Mapping]) -> Mapping:
        """
        Combine a list of return dictionaries into a single dictionary.
        """
        if len(return_dict_list) == 1:
            return return_dict_list[0]
        else:
            combined_return_dict = {}
            for key in return_dict_list[0].keys():
                if isinstance(return_dict_list[0][key], torch.Tensor):
                    # compute the average of the tensor is it is a scalar (e.g. loss)
                    if return_dict_list[0][key].dim() == 0:
                        combined_return_dict[key] = sum([d[key] for d in return_dict_list]) / len(return_dict_list)
                    else:
                        combined_return_dict[key] = torch.cat([d[key] for d in return_dict_list], dim=0)
                elif isinstance(return_dict_list[0][key], list):
                    temp_list = []
                    if key == "seq_name":
                        for d in return_dict_list:
                            temp_list.extend(d[key])
                        combined_return_dict[key] = temp_list
                    elif key == "pose_enc_list":
                        # return_dict_list has T elements, each element is a list of M tensors, with shape (B, S, P)
                        # make it a list of M tensors, with shape (T*B, S, P)
                        for list_idx in range(len(return_dict_list[0][key])):
                            temp_list.append(torch.cat([d[key][list_idx] for d in return_dict_list], dim=0))
                        combined_return_dict[key] = temp_list
                    else:
                        assert False, f"Unsupported key: {key}"
                else:
                    assert False, f"Unsupported type: {type(return_dict_list[0][key])}"
            return combined_return_dict

    def _prepare_log_data(
        self,
        predictions: Mapping[str, Any],
        loss_dict: Mapping[str, Any],
        batch: Mapping[str, Any],
        phase: str,
    ) -> Dict[str, Any]:
        """Detach and move required values to CPU to reduce GPU memory pressure."""

        def to_cpu_detached(value: Any, key: str = "") -> Any:
            if isinstance(value, torch.Tensor):
                # For scalar losses, keep on GPU - will sync only when needed
                if value.numel() == 1 and key.startswith("loss"):
                    return value.detach()
                # For large tensors, move to CPU asynchronously
                return value.detach().to('cpu', non_blocking=True)
            if isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                return [v.detach().to('cpu', non_blocking=True) for v in value]
            return value

        log_dict: Dict[str, Any] = {}

        for container in (predictions, loss_dict):
            for key, value in container.items():
                log_dict[key] = to_cpu_detached(value, key)

        for key in self._batch_keys_to_keep_for_logging(phase):
            if key in batch:
                log_dict[key] = to_cpu_detached(batch[key], key)

        return log_dict

    def _batch_keys_to_keep_for_logging(self, phase: str) -> Set[str]:
        """Return batch keys that are necessary for logging on rank 0."""
        keys: Set[str] = {"images", "seq_name", "input_images", "nvs_target_images", "nvs_target_depths", "nvs_target_point_masks"}

        if 'visuals_keys_to_log' in self.logging_conf:
            phase_visual_conf = self.logging_conf['visuals_keys_to_log']
            if phase_visual_conf is not None and phase in phase_visual_conf:
                keys_to_log = phase_visual_conf[phase].get('keys_to_log', [])
                if 'depth' in keys_to_log:
                    keys.add('depths')

        return keys

    def _process_batch(self, batch: Mapping):      
        if self.data_conf.train.common_config.repeat_batch:
            batch = self._apply_batch_repetition(batch)
        
        # Normalize camera extrinsics and points. The function returns new tensors.
        normalized_extrinsics, normalized_extrinsics_c2w, normalized_cam_points, normalized_world_points, normalized_depths = \
            normalize_camera_extrinsics_and_points_batch(
                extrinsics=batch["extrinsics"],
                has_extrinsics_c2w=True,
                cam_points=batch["cam_points"],
                world_points=batch["world_points"],
                depths=batch["depths"],
                point_masks=batch["point_masks"],
            )
        

        # Replace the original values in the batch with the normalized ones.
        batch["extrinsics"] = normalized_extrinsics
        batch["extrinsics_c2w"] = normalized_extrinsics_c2w
        batch["cam_points"] = normalized_cam_points
        batch["world_points"] = normalized_world_points
        batch["depths"] = normalized_depths

        if hasattr(self.model, 'nvs_head') and self.model.nvs_head is not None:
            B, S, _, _, _ = batch["images"].shape
            device = batch["images"].device
            # Select input and target views for NVS
            input_view_indices, target_view_indices = select_camera_views(
                normalized_extrinsics_c2w,
                max_total_images=self.data_conf.train.max_img_per_gpu,
            )
            input_view_num = input_view_indices.shape[1]
            target_view_num = target_view_indices.shape[1]
            batch_indices_target = torch.arange(B, device=device)[:, None].expand(B, target_view_num)
            batch_indices_input = torch.arange(B, device=device)[:, None].expand(B, input_view_num)

            nvs_input_type = self.model.nvs_input_type
            if nvs_input_type == "unposed_ray":
                # Normalize with scale computed from input views only, applied to both input and target views
                nvs_normalized_input_extrinsics_c2w, nvs_normalized_target_extrinsics_c2w, scale = normalize_camera_c2w(
                    extrinsics_c2w=normalized_extrinsics_c2w,
                    input_view_indices=input_view_indices,
                    target_view_indices=target_view_indices
                )
            else:
                raise ValueError(f"Unsupported NVS input type: {nvs_input_type}. Only 'unposed_ray' is currently supported for this implementation.")
            
            # Get ray conditions for target views
            target_intrinsics = batch["intrinsics"][batch_indices_target, target_view_indices]  # (B, target_view_num, 3, 3)
            input_intrinsics = batch["intrinsics"][batch_indices_input, input_view_indices] # (B, input_view_num, 3, 3)


            input_images = batch["images"][batch_indices_input, input_view_indices]  # (B, input_view_num, C, H, W)
            target_images = batch["images"][batch_indices_target, target_view_indices]  # (B, target_view_num, C, H, W)
            target_depths = batch["depths"][batch_indices_target, target_view_indices]  # (B, target_view_num, H, W)
            target_point_masks = batch["point_masks"][batch_indices_target, target_view_indices]  # (B, target_view_num, H, W)

            target_ray_cond = get_ray_conditions(
                extrinsics_c2w=nvs_normalized_target_extrinsics_c2w,
                intrinsics=target_intrinsics,
                image_size_hw=batch["images"].shape[-2:],
                type=self.model.nvs_ray_cond_type,
            )

            input_ray_cond = get_ray_conditions(
                extrinsics_c2w=nvs_normalized_input_extrinsics_c2w,
                intrinsics=input_intrinsics,
                image_size_hw=batch["images"].shape[-2:],
                type=self.model.nvs_ray_cond_type,
            )

            # every key in batch will be selected based on input indices
            for key in batch:
                if key == "images":
                    continue
                if isinstance(batch[key], torch.Tensor) and batch[key].dim() >=3 and batch[key].shape[1] == S:
                    batch[key] = batch[key][batch_indices_input, input_view_indices]
                
            query_cond = dict()
            query_cond["nvs_input_extrinsics_c2w"] = nvs_normalized_input_extrinsics_c2w
            query_cond["nvs_target_extrinsics_c2w"] = nvs_normalized_target_extrinsics_c2w
            query_cond["nvs_target_ray_cond"] = target_ray_cond
            query_cond["nvs_input_ray_cond"] = input_ray_cond
            query_cond["nvs_input_view_indices"] = input_view_indices # (B, input_view_num)
            query_cond["nvs_target_view_indices"] = target_view_indices # (B, target_view_num)
            query_cond["target_view_num"] = target_view_num
            query_cond["nvs_input_intrinsics"] = input_intrinsics
            query_cond["nvs_target_intrinsics"] = target_intrinsics

            batch["query_cond"] = query_cond
            batch["input_images"] = input_images
            batch["nvs_target_images"] = target_images
            batch["nvs_target_depths"] = target_depths
            batch["nvs_target_point_masks"] = target_point_masks

        else:
            query_cond = dict()
            
            query_cond["nvs_input_extrinsics_c2w"] = None
            query_cond["nvs_target_extrinsics_c2w"] = None
            query_cond["nvs_target_ray_cond"] = None
            query_cond["nvs_input_ray_cond"] = None
            query_cond["target_view_num"] = 0    
            query_cond["nvs_input_intrinsics"] = None
            query_cond["nvs_target_intrinsics"] = None
            batch["query_cond"] = query_cond
            batch["input_images"] = batch["images"]

        return batch


    def _update_and_log_scalars(self, data: Mapping, phase: str, loss_meters: dict, batch_size: int, log_to_wandb: bool = True):
        """Updates average meters and logs scalar values to the logger (WandB)."""
        keys_to_log = self._get_scalar_log_keys(phase)

        for key in keys_to_log:
            if key in data:
                # Convert tensor to Python scalar
                if torch.is_tensor(data[key]):
                    # Ensure tensor is on CPU before calling .item() to avoid sync
                    tensor = data[key].cpu() if data[key].is_cuda else data[key]
                    value = tensor.item() if tensor.numel() == 1 else tensor.mean().item()
                else:
                    value = data[key]
                loss_meters[f"Loss/{phase}_{key}"].update(value, batch_size)
                if log_to_wandb:
                    self.logger.log(f"{phase}/{key}", value, self.update_steps)



    @torch.no_grad()
    def _log_visual_locally(self, batch: Mapping, phase: str, step: int) -> None:
        """Logs image or video visualizations to the logger (WandB)."""

        if phase in self.logging_conf.visuals_keys_to_log:
            keys_to_log = self.logging_conf.visuals_keys_to_log[phase][
                "keys_to_log"
            ]
            if keys_to_log is None or len(keys_to_log) == 0:
                return
            input_images = batch["input_images"] # (B, S, 3, H, W)
            input_images = input_images.detach().cpu().numpy()
            input_images = input_images.transpose(0, 1, 3, 4, 2) # (B, S, H, W, 3)
            input_images = (input_images * 255).astype(np.uint8) # (B, S, H, W, 3)

            base_out_dir = os.path.join(self.logging_conf.log_dir, self.logging_conf.exp_name)
            relative_folder = os.path.join(phase, f"step_{step:08d}")
            os.makedirs(os.path.join(base_out_dir, relative_folder), exist_ok=True)
            rank = dist.get_rank()
            saving_depth = False
            for key in keys_to_log:
                # depth visualization
                if key == "depth" and "depth" in batch and "depths" in batch:
                    saving_depth = True
                    pred_depth = batch["depth"].squeeze(-1) # (B, S, H, W)
                    gt_depth = batch["depths"] # (B, S, H, W)
                    
                    batch_size = input_images.shape[0]
                    scene_name_list = []
                    image_grid_list = []
                    for i in range(batch_size):
                        pred_depth_np = depth_to_np_arr(pred_depth[i])
                        gt_depth_np = depth_to_np_arr(gt_depth[i])
                        input_image_np = input_images[i]
                        image_rows = [input_image_np, pred_depth_np, gt_depth_np]
                        image_grid = stack_images(image_rows)
                        image_grid_list.append([image_grid])
                        scene_name = batch["seq_name"][i]
                        scene_name_list.append(scene_name)

                    # save the scene name list
                    current_rel_path = os.path.join(relative_folder, f"scene_name_list_{rank}.txt")
                    with open(os.path.join(base_out_dir, current_rel_path), "w") as f:
                        for scene_name in scene_name_list:
                            f.write(f"{scene_name}\n")
                    final_image_grid = stack_images(image_grid_list)
                    current_rel_path = os.path.join(relative_folder, f"depth_vis_{rank}.png")
                    Image.fromarray(final_image_grid).save(os.path.join(base_out_dir, current_rel_path))
                if key == "depth_plus_cam":
                    pred_pose_enc = batch["pose_enc"]
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred_pose_enc, input_images.shape[-3:-1])
                    pred_depth = batch["depth"].squeeze(-1) # (B, S, H, W)
                    pred_3d_points = unproject_depth_map_to_point_map(pred_depth.reshape(-1, *pred_depth.shape[2:]), extrinsic.reshape(-1, *extrinsic.shape[2:]), intrinsic.reshape(-1, *intrinsic.shape[2:])) # (B*S, H, W, 3)
                    
                    batch_size = input_images.shape[0]
                    seq_len = input_images.shape[1]
                    
                    # Save point clouds for each sample individually
                    for sample_idx in range(batch_size):
                        sample_scene_name = batch["seq_name"][sample_idx]
                        
                        # Extract points and colors for this sample across all sequence frames
                        start_idx = sample_idx * seq_len
                        end_idx = (sample_idx + 1) * seq_len
                        sample_points = pred_3d_points[start_idx:end_idx]  # (S, H, W, 3)
                        sample_colors = input_images[sample_idx]  # (S, H, W, 3)
                        
                        # Flatten points and colors for this sample
                        sample_points_flat = sample_points.reshape(-1, 3)  # (S*H*W, 3)
                        sample_colors_flat = sample_colors.reshape(-1, 3)  # (S*H*W, 3)
                        
                        # Save individual sample point cloud
                        current_rel_path = os.path.join(relative_folder, "depth_plus_cam",f"sample_{sample_idx:03d}_{sample_scene_name}_3d_points.glb")
                        glb_file_path = os.path.join(base_out_dir, current_rel_path)
                        save_3d_points(sample_points_flat, sample_colors_flat, glb_file_path)
                if key == "nvs_rgb":
                    if not saving_depth:
                        # save the scene name list and the input image
                        batch_size = input_images.shape[0]
                        scene_name_list = []
                        image_grid_list = []
                        for i in range(batch_size):
                            scene_name = batch["seq_name"][i]
                            scene_name_list.append(scene_name)
                            input_image_np = input_images[i]
                            image_rows = [input_image_np]
                            image_grid = stack_images(image_rows)
                            image_grid_list.append([image_grid])
                        final_image_grid = stack_images(image_grid_list)
                        current_rel_path = os.path.join(relative_folder, f"input_images_{rank}.png")
                        Image.fromarray(final_image_grid).save(os.path.join(base_out_dir, current_rel_path))

                        current_rel_path = os.path.join(relative_folder, f"scene_name_list_{rank}.txt")
                        with open(os.path.join(base_out_dir, current_rel_path), "w") as f:
                            for scene_name in scene_name_list:
                                f.write(f"{scene_name}\n")

                    nvs_pred = batch["nvs_pred"][..., :3] # (B, target_view_num, H, W， 3)
                    nvs_target_images = batch["nvs_target_images"] # (B, target_view_num, 3, H, W)
                    nvs_target_images = nvs_target_images.detach().cpu().numpy().transpose(0,1,3,4,2) # (B, target_view_num, H, W, 3)
                    nvs_pred = (nvs_pred.detach().cpu().numpy() * 255).astype(np.uint8)
                    nvs_target_images = (nvs_target_images * 255).astype(np.uint8)

                    nvs_depth_gt = batch.get("nvs_target_depths", None)

                    if_pred_nvs_depth = False
                    if batch["nvs_pred"].shape[-1] > 3 and nvs_depth_gt is not None:
                        nvs_pred_depth = batch["nvs_pred"][..., 3] # (B, target_view_num, H, W)
                        nvs_pred_depth = nvs_pred_depth
                        nvs_depth_gt = nvs_depth_gt
                        if_pred_nvs_depth = True

                    batch_size = input_images.shape[0]
                    image_grid_list = []
                    for i in range(batch_size):
                        nvs_pred_np = nvs_pred[i]  # (target_view_num, H, W, 3)
                        nvs_target_np = nvs_target_images[i]  # (target_view_num, H, W, 3)
                        if if_pred_nvs_depth:
                            nvs_pred_depth_np = depth_to_np_arr(nvs_pred_depth[i])  # (target_view_num, H, W)
                            nvs_gt_depth_np = depth_to_np_arr(nvs_depth_gt[i])  # (target_view_num, H, W)
                            
                            image_rows = [nvs_target_np, nvs_pred_np, nvs_gt_depth_np, nvs_pred_depth_np]
                        else:
                            image_rows = [nvs_target_np, nvs_pred_np]
                        image_grid = stack_images(image_rows)
                        image_grid_list.append([image_grid])

                    final_image_grid = stack_images(image_grid_list)
                    current_rel_path = os.path.join(relative_folder, f"nvs_rgb_vis_{rank}.png")
                    Image.fromarray(final_image_grid).save(os.path.join(base_out_dir, current_rel_path))


def chunk_batch_for_accum_steps(batch: Mapping, accum_steps: int) -> List[Mapping]:
    """Splits a batch into smaller chunks for gradient accumulation."""
    if accum_steps == 1:
        return [batch]
    return [get_chunk_from_data(batch, i, accum_steps) for i in range(accum_steps)]

def is_sequence_of_primitives(data: Any) -> bool:
    """Checks if data is a sequence of primitive types (str, int, float, bool)."""
    return (
        isinstance(data, Sequence)
        and not isinstance(data, str)
        and len(data) > 0
        and isinstance(data[0], (str, int, float, bool))
    )

def get_chunk_from_data(data: Any, chunk_id: int, num_chunks: int) -> Any:
    """
    Recursively splits tensors and sequences within a data structure into chunks.

    Args:
        data: The data structure to split (e.g., a dictionary of tensors).
        chunk_id: The index of the chunk to retrieve.
        num_chunks: The total number of chunks to split the data into.

    Returns:
        A chunk of the original data structure.
    """
    if isinstance(data, torch.Tensor) or is_sequence_of_primitives(data):
        # either a tensor or a list of primitive objects
        # assert len(data) % num_chunks == 0
        start = (len(data) // num_chunks) * chunk_id
        end = (len(data) // num_chunks) * (chunk_id + 1)
        chunk = data[start:end]
        if isinstance(chunk, torch.Tensor):
            chunk = chunk.contiguous()
        return chunk
    elif isinstance(data, Mapping):
        return {
            key: get_chunk_from_data(value, chunk_id, num_chunks)
            for key, value in data.items()
        }
    elif isinstance(data, str):
        # NOTE: this is a hack to support string keys in the batch
        return data
    elif isinstance(data, Sequence):
        return [get_chunk_from_data(value, chunk_id, num_chunks) for value in data]
    else:
        return data
