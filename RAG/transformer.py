# model.py
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from retrieval import Retrieval


class ReconstructionTransformer(nn.Module):
    """
    Transformer encoder‑decoder that reconstructs masked values in a
    time‑series window for anomaly detection.
    """

    def __init__(
        self,
        D: int,
        seq_len: int,
        hidden_dim: int,
        num_layers_e: int,
        num_layers_d: int,
        num_heads_e: int,
        num_heads_d: int,
        p_dropout: float,
        layer_norm_eps: float,
        gradient_clipping: Optional[float],
        retrieval: Optional[Retrieval],
        device: torch.device,
        args,
    ):
        super().__init__()
        self.D = D
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.device = device
        self.retrieval_enabled = retrieval is not None


        # ------------------------------------------------------------------ #
        # 1)  Embedding projection (D  ->  hidden_dim)
        # ------------------------------------------------------------------ #
        self.input_proj = nn.Linear(D, hidden_dim)

        # time / position embeddings
        self.pos_embed = nn.Embedding(seq_len, hidden_dim)

        # ------------------------------------------------------------------ #
        # 2)  Encoder
        # ------------------------------------------------------------------ #
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads_e,
            batch_first=True,
            dropout=p_dropout,
            layer_norm_eps=layer_norm_eps,
            activation=args.model_act_func,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers_e)

        # ------------------------------------------------------------------ #
        # 3)  Bottleneck token (CLS‑style)  – optional but handy
        # ------------------------------------------------------------------ #
        self.bottleneck_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.bottleneck_token, std=0.02)

        # ------------------------------------------------------------------ #
        # 4)  Decoder
        # ------------------------------------------------------------------ #
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads_d,
            batch_first=True,
            dropout=p_dropout,
            layer_norm_eps=layer_norm_eps,
            activation=args.model_act_func,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers_d)

        # ------------------------------------------------------------------ #
        # 5)  Output projection (hidden_dim  ->  D)
        # ------------------------------------------------------------------ #
        self.output_proj = nn.Linear(hidden_dim, D)

        # ------------------------------------------------------------------ #
        # 6)  Retrieval settings
        # ------------------------------------------------------------------ #
        self.retrieval_module = retrieval
        if self.retrieval_enabled and hasattr(args, "exp_retrieval_location"):
            self.retrieval_loc     = args.exp_retrieval_location.lower()
            self.retrieval_agg_loc = args.exp_retrieval_agg_location.lower()
            self.agg_lambda        = args.exp_retrieval_agg_lambda
        else:                       # no retrieval ⇒ harmless defaults
            self.retrieval_loc     = "post-encoder"
            self.retrieval_agg_loc = "post-encoder"
            self.agg_lambda        = 0.0


        # sanity‑check ordering
        ORDER = {"pre-embedding": 0, "post-embedding": 1, "post-encoder": 2}
        assert (
            ORDER[self.retrieval_loc] <= ORDER[self.retrieval_agg_loc]
        ), "retrieval location must come before—or be equal to—aggregation point"

        # ------------------------------------------------------------------ #
        # 7)  Optional gradient clipping
        # ------------------------------------------------------------------ #
        if gradient_clipping:
            clip_val = float(gradient_clipping)

            def _clip_hook(grad):
                return grad.clamp_(min=-clip_val, max=clip_val)

            for p in self.parameters():
                p.register_hook(_clip_hook)

    # ====================================================================== #
    #                                Forward
    # ====================================================================== #

    def forward(
        self,
        x: Tensor,                      # [B, L, D]  (masked input)
        retrieval_cand: Optional[Tensor] = None,  # [B, K, L, D] or flattened
    ) -> Tensor:                       # returns [B, L, D] (reconstruction)
        B, L, _ = x.shape
        assert L == self.seq_len, "input length mismatch"

        # 0) retrieval **pre‑embedding**
        if self.retrieval_enabled and self.retrieval_loc == "pre-embedding":
            x = self._aggregate_pre_embedding(x, retrieval_cand)

        # ------------------------------------------------------------------ #
        # Embed + add positional information
        # ------------------------------------------------------------------ #
        x = self.input_proj(x)                           # [B, L, H]
        pos_idx = torch.arange(L, device=self.device)
        x = x + self.pos_embed(pos_idx).unsqueeze(0)

        # retrieval **post‑embedding**
        if self.retrieval_enabled and self.retrieval_loc == "post-embedding":
            x = self._aggregate_post_embedding(x, retrieval_cand)

        # ------------------------------------------------------------------ #
        # Encoder (optionally prepend bottleneck token)
        # ------------------------------------------------------------------ #
        bn_token = self.bottleneck_token.expand(B, -1, -1)   # [B,1,H]
        enc_in = torch.cat([bn_token, x], dim=1)             # [B,1+L,H]
        memory = self.encoder(enc_in)                        # [B,1+L,H]
        memory_main = memory[:, 1:, :]                       # discard cls

        # retrieval **post‑encoder**
        if self.retrieval_enabled and self.retrieval_loc == "post-encoder":
            memory_main = self._aggregate_post_embedding(memory_main, retrieval_cand)

        # ------------------------------------------------------------------ #
        # Decoder – teacher forcing with full sequence as tgt
        # ------------------------------------------------------------------ #
        tgt = torch.zeros_like(memory_main)  # start with zeros
        dec_out = self.decoder(tgt, memory_main)             # [B,L,H]

        # ------------------------------------------------------------------ #
        # Project back to feature space
        # ------------------------------------------------------------------ #
        reconstruction = self.output_proj(dec_out)           # [B,L,D]
        return reconstruction

    # ====================================================================== #
    #                           Retrieval helpers
    # ====================================================================== #

    def _aggregate_pre_embedding(self, x: Tensor, cand: Tensor) -> Tensor:
        """
        x:     [B,L,D]     (raw)
        cand:  [B,K,L,D]   (raw)
        """
        mean_cand = cand.mean(dim=1)
        return (1 - self.agg_lambda) * x + self.agg_lambda * mean_cand

    def _aggregate_post_embedding(self, x_emb: Tensor, cand: Tensor) -> Tensor:
        """
        Aggregates encoded representations (either post‑embedding or
        post‑encoder) using the Retrieval module’s similarity scores.
        Shapes:
            x_emb  : [B, L, H]
            cand   : [B, K, L, H]  (must already be projected to H)
        """
        B, L, H = x_emb.shape
        flat_x = x_emb.reshape(B, L * H)                   # [B, L*H]
        flat_cand = cand.reshape(B, cand.size(1), L * H)   # [B, K, L*H]

        idxs, weights = self.retrieval_module(flat_x, flat_cand)  # [B,K]
        gathered = torch.stack(
            [flat_cand[b, idxs[b]] for b in range(B)], dim=0
        )                                                 # [B,K,L*H]
        weights = weights.unsqueeze(-1)                   # [B,K,1]
        mean_cand = (gathered * weights).sum(1) / (weights.sum(1) + 1e-8)

        blended = (1 - self.agg_lambda) * flat_x + self.agg_lambda * mean_cand
        return blended.view(B, L, H)

    # ------------------------------------------------------------------ #
    # Retrieval‑module mode helper  ✨ NEW ✨
    # ------------------------------------------------------------------ #
    def set_retrieval_module_mode(self, mode: str) -> None:
        """
        Forward `mode` to the retrieval component (if any).

        Args
        ----
        mode : str
            One of {'train', 'eval', 'off'}.
        """
        if self.retrieval_module is None:
            return                       # nothing to do

        mode = mode.lower()
        if mode == 'train':
            self.retrieval_module.train()   # enable gradients / dropout
            self.retrieval_enabled = True
        elif mode == 'eval':
            self.retrieval_module.eval()    # eval mode but still queried
            self.retrieval_enabled = True
        elif mode == 'off':
            # completely skip retrieval during forward
            self.retrieval_enabled = False
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")    


# convenient wrapper identical to the old class name
Model = ReconstructionTransformer
