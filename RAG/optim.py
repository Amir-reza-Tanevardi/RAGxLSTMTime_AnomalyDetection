"""
Learning‑rate utilities (no fairseq dependency).

Supported schedules
-------------------
constant
linear_warmup
cosine_cyclic          – implemented with CosineAnnealingWarmRestarts
polynomial_decay_warmup
flat_and_anneal        – flat LR portion followed by cosine anneal
"""

from typing import Dict, Any, Callable

import numpy as np
import torch
from dotmap import DotMap
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import (
    LambdaLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)
from transformers import (
    get_constant_schedule,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def clip_gradient(model: nn.Module, clip: float) -> None:
    """Gradient‑clipping utility (in‑place)."""
    nn.utils.clip_grad_norm_(model.parameters(), clip)


class ConcatLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Concatenate two schedulers (scheduler1 → scheduler2).

    The first scheduler runs for ``pct_start`` • total_steps,
    the second for the remainder.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        scheduler1: torch.optim.lr_scheduler._LRScheduler,
        scheduler2: torch.optim.lr_scheduler._LRScheduler,
        total_steps: int,
        pct_start: float = 0.5,
        last_epoch: int = -1,
    ):
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.switch_step = int(pct_start * total_steps)
        self._step_count = 0
        super().__init__(optimizer, last_epoch)

    # ---- public API ----------------------------------------------------- #
    def step(self, epoch: int | None = None) -> None:  # type: ignore[override]
        if self._step_count < self.switch_step:
            self.scheduler1.step(epoch)
        else:
            self.scheduler2.step(epoch)
        self._step_count += 1
        super().step(epoch)

    # ---- required for state‑dict serialization -------------------------- #
    def state_dict(self) -> Dict[str, Any]:  # type: ignore[override]
        return {
            "scheduler1": self.scheduler1.state_dict(),
            "scheduler2": self.scheduler2.state_dict(),
            "_step_count": self._step_count,
            "switch_step": self.switch_step,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:  # noqa: D401
        self.scheduler1.load_state_dict(state_dict["scheduler1"])
        self.scheduler2.load_state_dict(state_dict["scheduler2"])
        self._step_count = state_dict["_step_count"]
        self.switch_step = state_dict["switch_step"]


# --------------------------------------------------------------------------- #
# Main wrapper
# --------------------------------------------------------------------------- #
class LRScheduler:
    """
    Thin wrapper that builds the requested LR scheduler and exposes
    `.step()`, `.state_dict()`, `.load_state_dict()`, and `__repr__`.
    """

    def __init__(self, c: DotMap, name: str, optimizer: Optimizer):
        """
        Parameters
        ----------
        c : DotMap
            Experiment‑wide configuration object.  Needs the following keys:

            * exp_train_total_epochs
            * exp_optimizer_warmup_proportion  (float ∈ [0,1])   **or**
            * exp_optimizer_warmup_fixed_n_steps (int)
            * exp_lr  (base learning rate)

        name : str
            One of the supported schedules listed in the module docstring.

        optimizer : torch.optim.Optimizer
            Optimizer whose LR we are scheduling.
        """
        self.c = c
        self.name = name
        self.optimizer = optimizer
        self.num_steps = 0

        self._build()
        print(f'[LRScheduler] Using "{self.name}" schedule.')

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        total_steps = self.c.exp_train_total_epochs

        # Warm‑up steps: proportion *or* fixed integer
        if self.c.exp_optimizer_warmup_proportion >= 0:
            num_warmup_steps = int(
                total_steps * self.c.exp_optimizer_warmup_proportion
            )
        else:
            num_warmup_steps = int(self.c.exp_optimizer_warmup_fixed_n_steps)

        print(f'[LRScheduler] warm‑up = {num_warmup_steps}/{total_steps} steps.')

        # ---- choose implementation ------------------------------------ #
        if self.name == "constant":
            self.scheduler = get_constant_schedule(self.optimizer)

        elif self.name == "linear_warmup":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
            )

        elif self.name == "cosine_cyclic":
            # Cosine‑annealing with warm restarts (SGDR‑style).  We mimic
            # fairseq’s defaults: first cycle ─ length = 2× warm‑up, then
            # T_mult = 2 so the next cycle doubles, etc.
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=num_warmup_steps * 2,  # first cycle length
                T_mult=2,  # cycle length multiplier
                eta_min=1e-7,  # minimal LR
            )

        elif self.name == "polynomial_decay_warmup":
            # HuggingFace helper (BERT style poly decay)
            self.scheduler = get_polynomial_decay_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                lr_end=1e-7,
                power=1.0,
            )

        elif self.name == "flat_and_anneal":
            # flat LR (dummy) → cosine anneal
            flat = LambdaLR(self.optimizer, lambda _: 1.0)
            cosine = CosineAnnealingLR(
                self.optimizer,
                int(total_steps * (1 - self.c.exp_optimizer_warmup_proportion)),
            )
            self.scheduler = ConcatLR(
                self.optimizer,
                flat,
                cosine,
                total_steps,
                self.c.exp_optimizer_warmup_proportion,
            )

        else:
            raise NotImplementedError(f"Unknown lr‑schedule: {self.name}")

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #
    def step(self) -> None:
        """Advance one training step (call *after* optimizer.step())."""
        self.num_steps += 1

        if self.name == "cosine_cyclic":
            # CosineAnnealingWarmRestarts needs `step(current_step)`
            self.scheduler.step(self.num_steps)
        else:
            self.scheduler.step()

    # ---- serialization helpers --------------------------------------- #
    def state_dict(self) -> Dict[str, Any]:
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.scheduler.load_state_dict(state_dict)

    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        return f"LRScheduler(name={self.name})"
