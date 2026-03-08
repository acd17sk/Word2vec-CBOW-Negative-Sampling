"""
Training loop and evaluation helpers for CBOW Word2Vec.
"""

import time
import numpy as np
from tqdm.auto import tqdm
from cbow import CBOWModel
from dataset import CBOWDataset
from vocab import Vocabulary


def train(
    model: CBOWModel,
    dataset: CBOWDataset,
    epochs: int = 10,
    batch_size: int = 256,
    lr_start: float = 0.025,
    lr_end: float = 0.0001,
    verbose: bool = True,
) -> list[float]:
    """
    Train the CBOW model with linear learning-rate decay.

    Returns a list of average losses per epoch.
    """
    total_batches_est = len(dataset) // batch_size * epochs
    global_step = 0
    epoch_losses = []

    epoch_bar = tqdm(range(1, epochs + 1), desc="Training", disable=not verbose)

    for epoch in epoch_bar:
        running_loss = 0.0
        n_batches = 0

        batch_iter = dataset.batches(batch_size, shuffle=True)
        batch_bar = tqdm(
            batch_iter,
            desc=f"Epoch {epoch}/{epochs}",
            total=len(dataset) // batch_size + 1,
            leave=False,
            disable=not verbose,
        )

        for ctx, tgt, neg in batch_bar:
            # linear LR decay
            progress = global_step / max(total_batches_est, 1)
            lr = lr_start - (lr_start - lr_end) * progress
            lr = max(lr, lr_end)

            loss = model.train_step(ctx, tgt, neg, lr)
            running_loss += loss
            n_batches += 1
            global_step += 1

            # update inner bar with running average loss
            batch_bar.set_postfix(loss=f"{running_loss / n_batches:.4f}", lr=f"{lr:.5f}")

        avg_loss = running_loss / max(n_batches, 1)
        epoch_losses.append(avg_loss)

        # update outer bar
        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}")

    return epoch_losses


# ------------------------------------------------------------------
# evaluation helpers
# ------------------------------------------------------------------

def most_similar(
    word: str,
    vocab: Vocabulary,
    model: CBOWModel,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """
    Find the top-k most similar words by cosine similarity
    (using the W_in embeddings).
    """
    if word not in vocab.word2index:
        print(f"'{word}' not in vocabulary")
        return []

    idx = vocab.word2index[word]
    vec = model.W_in[idx]                            # (D,)
    norms = np.linalg.norm(model.W_in, axis=1)      # (V,)
    norm_vec = np.linalg.norm(vec)

    # cosine similarity against all words
    sims = model.W_in @ vec / (norms * norm_vec + 1e-9)

    # exclude the word itself and UNK
    sims[idx] = -np.inf
    sims[0] = -np.inf  # UNK

    top_ids = np.argsort(sims)[-top_k:][::-1]
    return [(vocab.index2word[i], float(sims[i])) for i in top_ids]


def analogy(
    a: str, b: str, c: str,
    vocab: Vocabulary,
    model: CBOWModel,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """
    Solve: a is to b as c is to ?
    vector arithmetic: result ≈ W_in[b] - W_in[a] + W_in[c]
    """
    w2i = vocab.word2index
    for w in (a, b, c):
        if w not in w2i:
            print(f"'{w}' not in vocabulary")
            return []

    vec = model.W_in[w2i[b]] - model.W_in[w2i[a]] + model.W_in[w2i[c]]
    norms = np.linalg.norm(model.W_in, axis=1)
    sims = model.W_in @ vec / (norms * np.linalg.norm(vec) + 1e-9)

    # exclude the query words
    for w in (a, b, c):
        sims[w2i[w]] = -np.inf
    sims[0] = -np.inf

    top_ids = np.argsort(sims)[-top_k:][::-1]
    return [(vocab.index2word[i], float(sims[i])) for i in top_ids]