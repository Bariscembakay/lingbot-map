# Code for ZipMap (CVPR 2026); created by Haian Jin



import atexit
import logging
from typing import Any, Dict, Optional, Union
import os
import torch
import wandb
from easydict import EasyDict as edict
from .distributed import get_machine_local_and_dist_rank
import yaml


class WandBLogger:
    """A wrapper around wandb logging with distributed training support.

    Only writes from rank 0 in distributed settings to avoid conflicts.
    Automatically handles cleanup on exit.
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        api_key_path: Optional[str] = 'training/config/wandb_key.yaml',
        **kwargs: Any,
    ) -> None:
        _, self._rank = get_machine_local_and_dist_rank()
        self._run = None
        self._dir = dir
        if self._rank == 0:
            try:
                assert os.path.exists(api_key_path), f"API key file does not exist: {api_key_path}"
                with open(api_key_path, "r") as f:
                    api_keys = edict(yaml.safe_load(f))
                assert api_keys.wandb is not None, "Wandb API key not found in api key file"
                os.environ["WANDB_API_KEY"] = api_keys.wandb
                logging.info(f"wandb run initialized. Project: {project}, Name: {name}, Dir: {dir}")
                self._run = wandb.init(project=project, name=name, dir=dir, config=config, **kwargs)
            except Exception as e:
                logging.error(f"Failed to initialize wandb: {e}")
                self._run = None

        atexit.register(self.close)

    @property
    def run(self):
        return self._run

    @property
    def path(self) -> Optional[str]:
        return self._dir

    def flush(self) -> None:
        if self._run:
            wandb.log({}, commit=True)

    def close(self) -> None:
        if self._run:
            self._run.finish()
            self._run = None

    def log_dict(self, payload: Dict[str, Any], step: int) -> None:
        if not self._run:
            return
        wandb.log(payload, step=step)

    def log(self, name: str, data: Any, step: int) -> None:
        if not self._run:
            return
        wandb.log({name: data}, step=step)



    def log_visuals(
        self,
        name: str,
        data: Union[torch.Tensor, Any],
        step: int,
        fps: int = 4
    ) -> None:
        if not self._run:
            return
        # Images: 3D tensor (C, H, W), Videos: 5D tensor (N, T, C, H, W)
        if isinstance(data, torch.Tensor):
            if data.ndim == 3:
                # Single image
                img = data.detach().cpu().numpy()
                wandb.log({name: [wandb.Image(img)]}, step=step)
            elif data.ndim == 5:
                # Video: (N, T, C, H, W)
                video = data.detach().cpu().numpy()
                wandb.log({name: wandb.Video(video, fps=fps, format="mp4")}, step=step)
            else:
                raise ValueError(f"Unsupported data dimensions: {data.ndim}. Expected 3D for images or 5D for videos.")
        else:
            raise ValueError(f"Unsupported data type for visuals: {type(data)}") 