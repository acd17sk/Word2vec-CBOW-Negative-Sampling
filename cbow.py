"""
CBOW with Negative Sampling — pure NumPy implementation.

Architecture
============
Two embedding matrices:
    W_in  : (vocab_size, embed_dim)  — embeddings for context words
    W_out : (vocab_size, embed_dim)  — embeddings for target / negative words

Forward pass (for one sample)
=============================
    h = (1 / 2C) * Σ  W_in[c]          for c in context_ids   (C = window size)
    score_pos = W_out[target] · h
    score_neg = W_out[neg_j]  · h       for each negative j

Loss  (negative sampling objective)
====================================
    L = -log σ(score_pos) - Σ_j log σ(-score_neg_j)

    where σ is the sigmoid function.

Gradients (derived analytically)
================================
    Let  σ_pos = σ(score_pos),  σ_neg_j = σ(score_neg_j)

    ∂L/∂W_out[target]  = (σ_pos - 1) · h
    ∂L/∂W_out[neg_j]   = σ_neg_j      · h
    ∂L/∂h              = (σ_pos - 1) · W_out[target]
                        + Σ_j σ_neg_j · W_out[neg_j]
    ∂L/∂W_in[c]        = (1 / 2C) · ∂L/∂h        for each context word c

Parameter update (SGD with optional learning-rate schedule):
    θ ← θ - lr · ∂L/∂θ
"""

import numpy as np


# ------------------------------------------------------------------
# numerically stable sigmoid
# ------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: avoids overflow in exp."""
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    # for x >= 0:  1 / (1 + exp(-x))
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    # for x < 0:   exp(x) / (1 + exp(x))
    ez = np.exp(x[neg])
    out[neg] = ez / (1.0 + ez)
    return out


class CBOWModel:
    """
    Pure NumPy CBOW with negative sampling.
    Supports mini-batch training with SGD (+ optional linear LR decay).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        seed: int = 42,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        rng = np.random.default_rng(seed)

        # Xavier-style init
        scale = 1.0 / np.sqrt(embed_dim)
        self.W_in = rng.uniform(-scale, scale, (vocab_size, embed_dim))
        self.W_out = np.zeros((vocab_size, embed_dim))  # zero-init for output embeddings

    # ------------------------------------------------------------------
    # forward + loss  (batched)
    # ------------------------------------------------------------------

    def forward(
        self,
        context_ids: np.ndarray,   # (B, 2*window)
        target_ids: np.ndarray,    # (B,)
        neg_ids: np.ndarray,       # (B, K)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Returns
        -------
        h          : (B, D)   — hidden (context) vectors
        sig_pos    : (B,)     — σ(score) for positive targets
        sig_neg    : (B, K)   — σ(score) for negatives
        loss       : scalar   — mean loss over the batch
        """
        B = context_ids.shape[0]
        num_ctx = context_ids.shape[1]  # = 2 * window

        # h = mean of context embeddings  →  (B, D)
        # W_in[context_ids] → (B, 2*window, D)
        h = self.W_in[context_ids].mean(axis=1)          # (B, D)

        # positive scores
        v_target = self.W_out[target_ids]                 # (B, D)
        score_pos = np.sum(h * v_target, axis=1)          # (B,)
        sig_pos = sigmoid(score_pos)                       # (B,)

        # negative scores
        v_neg = self.W_out[neg_ids]                        # (B, K, D)
        score_neg = np.einsum("bd,bkd->bk", h, v_neg)     # (B, K)
        sig_neg = sigmoid(score_neg)                       # (B, K)

        # loss = - log σ(pos) - Σ log σ(-neg)
        eps = 1e-7  # numerical stability for log
        loss_pos = -np.log(sig_pos + eps)                  # (B,)
        loss_neg = -np.log(1.0 - sig_neg + eps)            # (B, K)
        loss = (loss_pos + loss_neg.sum(axis=1)).mean()

        return h, sig_pos, sig_neg, loss

    # ------------------------------------------------------------------
    # backward  (batched)
    # ------------------------------------------------------------------

    def backward(
        self,
        context_ids: np.ndarray,
        target_ids: np.ndarray,
        neg_ids: np.ndarray,
        h: np.ndarray,
        sig_pos: np.ndarray,
        sig_neg: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradients for W_in (context words), W_out (target), W_out (negatives).

        Returns
        -------
        grad_in   : dict  {word_id: grad_vector}   accumulated for context words
        grad_out_target : (B, D)
        grad_out_neg    : (B, K, D)
        """
        B = context_ids.shape[0]
        num_ctx = context_ids.shape[1]

        # ------- gradients w.r.t. W_out -------
        # ∂L/∂W_out[target] = (σ_pos - 1) * h     → (B, D)
        err_pos = (sig_pos - 1.0)[:, None]           # (B, 1)
        grad_out_target = err_pos * h                 # (B, D)

        # ∂L/∂W_out[neg_j] = σ_neg_j * h            → (B, K, D)
        grad_out_neg = sig_neg[:, :, None] * h[:, None, :]  # (B, K, D)

        # ------- gradient w.r.t. h -------
        # ∂L/∂h = (σ_pos - 1)*W_out[target] + Σ σ_neg_j * W_out[neg_j]
        v_target = self.W_out[target_ids]              # (B, D)
        v_neg = self.W_out[neg_ids]                    # (B, K, D)

        grad_h = err_pos * v_target                    # (B, D)
        grad_h += np.einsum("bk,bkd->bd", sig_neg, v_neg)  # (B, D)

        # ------- gradient w.r.t. W_in[c] -------
        # ∂L/∂W_in[c] = (1/num_ctx) * ∂L/∂h    for each context word
        grad_ctx = grad_h / num_ctx                    # (B, D)

        return grad_ctx, grad_out_target, grad_out_neg

    # ------------------------------------------------------------------
    # parameter update  (SGD)
    # ------------------------------------------------------------------

    def update(
        self,
        context_ids: np.ndarray,
        target_ids: np.ndarray,
        neg_ids: np.ndarray,
        grad_ctx: np.ndarray,
        grad_out_target: np.ndarray,
        grad_out_neg: np.ndarray,
        lr: float,
    ):
        """
        Apply SGD updates using np.add.at for sparse index-based updates.
        We average gradients over the batch before updating.
        """
        B = context_ids.shape[0]
        inv_B = lr / B

        # update W_in for each context word
        # context_ids: (B, 2*window) — each row shares the same grad_ctx
        for j in range(context_ids.shape[1]):
            np.add.at(self.W_in, context_ids[:, j], -inv_B * grad_ctx)

        # update W_out for positive targets
        np.add.at(self.W_out, target_ids, -inv_B * grad_out_target)

        # update W_out for negatives
        K = neg_ids.shape[1]
        for k in range(K):
            np.add.at(self.W_out, neg_ids[:, k], -inv_B * grad_out_neg[:, k, :])

    # ------------------------------------------------------------------
    # single training step (convenience)
    # ------------------------------------------------------------------

    def train_step(
        self,
        context_ids: np.ndarray,
        target_ids: np.ndarray,
        neg_ids: np.ndarray,
        lr: float,
    ) -> float:
        """Forward → backward → update.  Returns the batch loss."""
        h, sig_pos, sig_neg, loss = self.forward(context_ids, target_ids, neg_ids)
        grad_ctx, grad_out_target, grad_out_neg = self.backward(
            context_ids, target_ids, neg_ids, h, sig_pos, sig_neg
        )
        self.update(
            context_ids, target_ids, neg_ids,
            grad_ctx, grad_out_target, grad_out_neg,
            lr,
        )
        return loss

    # ------------------------------------------------------------------
    # get embeddings (final)
    # ------------------------------------------------------------------

    def get_embedding(self, word_id: int) -> np.ndarray:
        """Return the input embedding for a word."""
        return self.W_in[word_id]

    def get_embeddings(self) -> np.ndarray:
        """Return the full W_in matrix (vocab_size, embed_dim)."""
        return self.W_in.copy()
