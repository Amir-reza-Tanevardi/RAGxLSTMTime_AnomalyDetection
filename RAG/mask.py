import math
import random
import itertools
from typing import List, Tuple

import torch


def generate_mask_train(
    data_shape: Tuple[int, int, int],
    p_mask: float = 0.15,
    force_mask: bool = False
) -> torch.BoolTensor:
    """
    BERT‑style random masking on a batch of time‑series windows.

    Args:
      data_shape: (B, L, D)
      p_mask:     probability of masking each element
      force_mask: ensure at least one mask per sample

    Returns:
      mask: BoolTensor of shape (B, L, D), where True = masked
    """
    B, L, D = data_shape
    # sample according to p_mask
    mask = torch.rand(B, L, D) < p_mask

    if force_mask:
        # ensure each sample has at least one True
        for b in range(B):
            if not mask[b].any():
                # randomly pick one time‑feature to mask
                t = random.randrange(L)
                f = random.randrange(D)
                mask[b, t, f] = True

    return mask


def generate_mask_val(
    data_shape: Tuple[int, int, int],
    num_masks: int = 10,
    p_mask: float = 0.15,
    deterministic: bool = False
) -> List[torch.BoolTensor]:
    """
    Pre-generate a small set of masks for validation.

    Args:
      data_shape:   (B, L, D)
      num_masks:    how many different masks to produce
      p_mask:       probability of masking each element (if not deterministic)
      deterministic: if True, generate masks that each mask exactly k elements,
                     for k in 1..min(L*D-1, num_masks), else random.

    Returns:
      List of BoolTensors, each of shape (B, L, D)
    """
    B, L, D = data_shape
    masks: List[torch.BoolTensor] = []

    if deterministic:
        total_positions = L * D
        # choose up to `num_masks` distinct numbers of masked positions
        ks = list(range(1, min(total_positions, num_masks) + 1))
        for k in ks:
            # generate one mask per k by choosing k positions uniformly
            idxs = random.sample(range(total_positions), k)
            flat = torch.zeros(total_positions, dtype=torch.bool)
            flat[idxs] = True
            m = flat.view(L, D).unsqueeze(0).expand(B, L, D)
            masks.append(m)
    else:
        # just random masks
        while len(masks) < num_masks:
            m = generate_mask_train((B, L, D), p_mask=p_mask, force_mask=False)
            masks.append(m)

    return masks


def apply_mask(
    data: torch.Tensor,
    p_mask: float = 0.15,
    force_mask: bool = False,
    device: torch.device = None,
    eval_mode: bool = False,
    mask_matrix: torch.BoolTensor = None,
) -> dict:
    """
    Apply masking to a batch of time‑series windows.

    Args:
      data:        Tensor [B, L, D]
      p_mask:      probability for training mask
      force_mask:  ensure ≥1 mask per sample
      eval_mode:   if True, uses provided mask_matrix instead of random
      mask_matrix: BoolTensor [B, L, D] to apply when eval_mode=True

    Returns:
      {
        'masked_tensor': Tensor[B, L, D],
        'ground_truth':  Tensor[B, L, D],
        'mask_matrix':   BoolTensor[B, L, D]  (True = masked)
      }
    """
    B, L, D = data.shape
    if not eval_mode:
        mask = generate_mask_train((B, L, D), p_mask=p_mask, force_mask=force_mask)
    else:
        assert mask_matrix is not None, "Must supply mask_matrix in eval_mode"
        mask = mask_matrix

    mask = mask.to(device)
    original = data.to(device)
    # zero out masked positions
    masked = original.clone()
    masked[mask] = 0.0

    return {
        'masked_tensor': masked,       # feed this to your model
        'ground_truth':  original,     # for loss calculation
        'mask_matrix':   mask          # True where model should predict
    }
