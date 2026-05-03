"""End-to-end SmolVLA wrapper.

Thin shim over LeRobot's SmolVLAPolicy that conforms to our BaseVLA
interface. Lets us use SmolVLA interchangeably with a future decoupled
(VLM + small policy) approach.

Implementation note: LeRobot's API may shift slightly across versions.
Pin the lerobot version in pyproject.toml. Touch points likely to change:
  - module path of SmolVLAPolicy
  - exact tensor key names ("observation.images.wrist" etc.)
Keep all of those LeRobot-specific details inside this file so the rest
of the codebase stays clean.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.models.base_vla import ActionChunk, BaseVLA, Observation


class SmolVLAWrapper(BaseVLA):
    def __init__(
        self,
        policy: torch.nn.Module,
        preprocessor=None,
        postprocessor=None,
        chunk_size: int = 50,
        camera_keys: tuple[str, ...] = ("main",),
        device: str = "cuda",
    ):
        self._policy = policy
        self._pre = preprocessor
        self._post = postprocessor
        self._chunk_size = chunk_size
        self._camera_keys = camera_keys
        self._device = device
        self._action_buffer: list[np.ndarray] = []

    # ----- BaseVLA interface -----

    def predict(self, obs: Observation) -> ActionChunk:
        # Build the raw batch in the same shape train.py feeds the policy,
        # then run the saved preprocessor (state/image normalization with
        # the dataset stats baked into the checkpoint). Skipping this is
        # the bug repair_checkpoint_processors.py warns about.
        batch = self._obs_to_batch(obs)
        if self._pre is not None:
            batch = self._pre(batch)

        with torch.inference_mode():
            action_tensor = self._policy.predict_action_chunk(batch)

        # Unnormalize the action chunk back into robot units before sending.
        if self._post is not None:
            action_tensor = self._post(action_tensor)

        actions = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        return ActionChunk(actions=actions, chunk_size=actions.shape[0])

    def reset(self) -> None:
        self._action_buffer.clear()
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    @property
    def active_param_count(self) -> int:
        return sum(p.numel() for p in self._policy.parameters())

    def to(self, device):
        self._device = str(device)
        self._policy = self._policy.to(device)
        return self

    def eval(self):
        self._policy.eval()
        return self

    @classmethod
    def from_checkpoint(cls, path: str, **kwargs) -> "SmolVLAWrapper":
        # Defer the import so just importing this module doesn't pull in
        # all of lerobot. Path is for lerobot v0.5.x (no more `common/`).
        from lerobot.policies.smolvla.modeling_smolvla import (  # type: ignore
            SmolVLAPolicy,
        )
        from lerobot.processor.pipeline import (  # type: ignore
            DataProcessorPipeline,
        )

        policy = SmolVLAPolicy.from_pretrained(path)

        # Load the pre/post-processors saved alongside the model. Without
        # these, state + image go in un-normalized and actions come out
        # un-unnormalized — robot jitters / overshoots / does nonsense.
        try:
            preprocessor = DataProcessorPipeline.from_pretrained(
                path, config_filename="policy_preprocessor.json"
            )
            postprocessor = DataProcessorPipeline.from_pretrained(
                path, config_filename="policy_postprocessor.json"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load preprocessor/postprocessor from {path}. "
                "The checkpoint must contain policy_preprocessor.json + "
                "policy_postprocessor.json + their *_processor.safetensors "
                "stat tensors. For old overfit checkpoints rebuild them "
                "with scripts/repair_checkpoint_processors.py."
            ) from e

        return cls(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            **kwargs,
        )

    # ----- internals -----

    def _obs_to_batch(self, obs: Observation) -> dict[str, torch.Tensor]:
        """Convert our Observation dataclass into the dict SmolVLA expects.

        Adjust key names to match the LeRobot dataset features your
        teleop pipeline actually produces. Run `python -c "import json,
        torch; d = torch.load(...); print(d.keys())"` on a training batch
        to confirm.
        """
        batch: dict[str, torch.Tensor] = {}
        for cam in self._camera_keys:
            img = obs.images[cam]  # (H, W, 3) uint8
            # SmolVLA expects float in [0, 1], (B, C, H, W). The saved
            # preprocessor handles the dataset-stat normalization on top.
            t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            batch[f"observation.images.{cam}"] = t.unsqueeze(0).to(self._device)

        batch["observation.state"] = (
            torch.from_numpy(obs.state).float().unsqueeze(0).to(self._device)
        )
        batch["task"] = [obs.prompt]
        return batch
