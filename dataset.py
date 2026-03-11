import numpy as np
from vocab import Vocabulary


class CBOWDataset:
    """
    Given encoded sentences, yields mini-batches of
    (context_ids, target_id, negative_ids) tuples.

    context_ids : (batch, 2*window)   int array
    target_ids  : (batch,)            int array
    neg_ids     : (batch, num_neg)    int array
    """

    def __init__(
        self,
        corpus: list[list[str]],
        vocab: Vocabulary,
        window: int = 5,
        num_neg: int = 5,
        seed: int = 42,
    ):
        self.vocab = vocab
        self.window = window
        self.num_neg = num_neg
        self.rng = np.random.default_rng(seed)

        # encode & subsample each sentence
        self.data: list[list[int]] = []
        for tokens in corpus:
            ids = vocab.encode(tokens)
            mask = vocab.subsample_mask(ids, self.rng)
            filtered = [i for i, keep in zip(ids, mask) if keep]
            if len(filtered) >= 2 * window + 1:
                self.data.append(filtered)

    def __len__(self) -> int:
        """Rough count of training examples."""
        return sum(max(0, len(s) - 2 * self.window) for s in self.data)

    def batches(self, batch_size: int = 256, shuffle: bool = True):
        """
        Generator that yields (context, target, negatives) mini-batches.
        Each epoch re-shuffles the sentences and re-samples negatives.
        """
        order = np.arange(len(self.data))
        if shuffle:
            self.rng.shuffle(order)

        ctx_buf, tgt_buf, neg_buf = [], [], []

        for si in order:
            sent = self.data[si]
            for i in range(self.window, len(sent) - self.window):
                # context: window words on each side (fixed window, no dynamic shrink for simplicity)
                context = sent[i - self.window : i] + sent[i + 1 : i + self.window + 1]
                target = sent[i]

                ctx_buf.append(context)
                tgt_buf.append(target)
                neg_buf.append(
                    self.vocab.sample_negatives(self.num_neg, self.rng)
                )

                if len(ctx_buf) == batch_size:
                    yield (
                        np.array(ctx_buf, dtype=np.int64),
                        np.array(tgt_buf, dtype=np.int64),
                        np.array(neg_buf, dtype=np.int64),
                    )
                    ctx_buf, tgt_buf, neg_buf = [], [], []

        # leftover
        if ctx_buf:
            yield (
                np.array(ctx_buf, dtype=np.int64),
                np.array(tgt_buf, dtype=np.int64),
                np.array(neg_buf, dtype=np.int64),
            )
