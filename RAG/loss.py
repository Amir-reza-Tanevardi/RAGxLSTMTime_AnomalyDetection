import sys
import torch
import torch.nn as nn
from collections import defaultdict

class Loss:
    """
    Loss for time-series masked reconstruction.
    Computes MSE over masked entries in the input window.
    """

    def __init__(self, args, is_batchlearning: bool, device=None):
        self.args = args
        self.is_batchlearning = is_batchlearning
        self.device = device

        # we’ll accumulate raw (sum) losses and counts
        self._mode = 'train'
        self.reset_batch()
        self.reset_epoch()

    def set_mode(self, mode: str):
        assert mode in ('train', 'val')
        self._mode = mode
        # clear val-specific storage at start of val
        if mode == 'val':
            self.reset_epoch()

    def reset_batch(self):
        self.batch_loss = 0.0
        self.batch_count = 0

    def reset_epoch(self):
        self.epoch_loss = 0.0
        self.epoch_count = 0

    def compute(self, output: torch.Tensor,
                ground_truth: torch.Tensor,
                mask_matrix: torch.BoolTensor):
        """
        Args:
          output        : [B, L, D]  model predictions
          ground_truth  : [B, L, D]  original data
          mask_matrix   : [B, L, D]  True where masked (we compute loss there)
        """
        # sanity check
        if torch.isnan(output).any():
            print("NaN in model output—exiting.") ; sys.exit(1)

        # squared error on masked positions
        se = (output - ground_truth).pow(2)
        masked_se = se[mask_matrix]

        # sum over all masked entries
        loss_sum = masked_se.sum()
        count   = masked_se.numel()

        # store for batch
        self.batch_loss  = loss_sum
        self.batch_count = count

        # accumulate into epoch
        self.epoch_loss += loss_sum.detach()  # detach so we don't track grads
        self.epoch_count += count

    def finalize_batch_loss(self):
        """
        Return a dict compatible with the old API:
        {'train': {'total_loss': <scalar Tensor>}}
        """
        if self.batch_count == 0:
            val = torch.tensor(0.0, device=self.device)
        else:
            val = self.batch_loss / self.batch_count
        return {'train': {'total_loss': val}}

    def finalize_epoch_losses(self, eval_model: bool = False):
        """
        Return dict compatible with old API:
        { 'train': {'total_loss': <avg Tensor>}, 'val': {...} }
        """
        avg = (self.epoch_loss / self.epoch_count) if self.epoch_count>0 else torch.tensor(0.0, device=self.device)
        return {self._mode: {'total_loss': avg}}

    def compute_per_sample(self, output, ground_truth, mask_matrix):
        """
        Return MSE per‑sample as a 1‑D tensor of length B
        instead of a single averaged scalar.
        """
        se = (output - ground_truth).pow(2)          # [B,L,D]
        masked_se = se * mask_matrix                 # zeros where not masked
        # sum over (L,D)  →  [B]
        loss_per_sample = masked_se.view(output.size(0), -1).sum(1)
        # normalise by number of masked positions per sample
        denom = mask_matrix.view(output.size(0), -1).sum(1).clamp(min=1)
        return loss_per_sample / denom
    

    def get_individual_val_loss(self):
        """
        For validation, you may want the per-sample total loss:
        here we cannot reconstruct that easily since we aggregated
        per-epoch; if you need it, you'd have to store per-batch
        masked losses per sample.
        """
        raise NotImplementedError("Per-sample val loss not supported in this version.")
