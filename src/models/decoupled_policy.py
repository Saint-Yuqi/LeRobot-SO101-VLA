"""Decoupled policy: VLM target selector + small action policy.

PLACEHOLDER — fill in once the team decides whether to go this route for
Eval 2 / Eval 3.

The idea: a frozen VLM looks at (image, prompt) once at the start of an
episode and outputs a target description (e.g. a 2D pixel coordinate, a
cropped image, or a one-hot over candidate bowls). The action policy
then conditions on this *resolved* target instead of the raw prompt,
which makes it much smaller and easier to train.

Why this might be needed:
  - Eval 2 compositional reasoning ("mix red and blue") is a language
    task; SmolVLA's tiny LM may not handle it from a few hundred episodes.
  - Eval 3 celebrity recognition needs world knowledge a small VLA
    fine-tune cannot inject.

Open questions for the team:
  1. What is the target representation? Options:
     a. (x, y) pixel coordinate of bowl center
     b. Crop of the target object
     c. Index over a fixed set of candidates
  2. How do we get a frozen VLM that runs locally? (Qwen2-VL 2B? Gemma-3?)
  3. Bonus point implication: does "active params" include the VLM?
     Ask in #project-1-vla.
"""
from __future__ import annotations

from src.models.base_vla import ActionChunk, BaseVLA, Observation


class DecoupledPolicy(BaseVLA):
    """Stub — to be implemented if/when the team chooses this approach."""

    def __init__(self, target_selector, action_policy):
        self._selector = target_selector  # frozen VLM
        self._action = action_policy      # small fine-tuned policy
        self._cached_target = None

    def predict(self, obs: Observation) -> ActionChunk:
        raise NotImplementedError(
            "DecoupledPolicy.predict not implemented yet. "
            "See module docstring for design questions."
        )

    def reset(self) -> None:
        self._cached_target = None

    @property
    def active_param_count(self) -> int:
        # IMPORTANT: confirm with TAs whether this counts the VLM. If the
        # VLM is invoked once per episode, the question is whether they
        # measure peak active params or sum across forward passes.
        sel = sum(p.numel() for p in self._selector.parameters())
        act = sum(p.numel() for p in self._action.parameters())
        return sel + act

    def to(self, device):
        self._selector = self._selector.to(device)
        self._action = self._action.to(device)
        return self

    def eval(self):
        self._selector.eval()
        self._action.eval()
        return self

    @classmethod
    def from_checkpoint(cls, path: str, **kwargs) -> "DecoupledPolicy":
        raise NotImplementedError
