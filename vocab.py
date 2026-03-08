"""
Vocabulary builder and text preprocessing for Word2Vec.
"""

import re
import numpy as np
from collections import Counter

import nltk

_NLTK_STOPWORDS: set[str] | None = None

def _get_stopwords() -> set[str]:
    """Lazy-load nltk English stopwords on first use."""
    global _NLTK_STOPWORDS
    if _NLTK_STOPWORDS is None:
        nltk.download("stopwords", quiet=True)
        _NLTK_STOPWORDS = set(nltk.corpus.stopwords.words("english"))
    return _NLTK_STOPWORDS


class Vocabulary:
    """
    Builds a word-to-index mapping, computes unigram frequencies,
    and provides a noise distribution for negative sampling.
    """

    UNK_TOKEN = "<UNK>"

    def __init__(self, min_freq: int = 5, subsample_t: float = 1e-4):
        """
        Args:
            min_freq: minimum count for a word to be kept in the vocab.
            subsample_t: threshold for frequent-word subsampling
                         (Mikolov et al., 2013). Set to 0 to disable.
        """
        self.min_freq = min_freq
        self.subsample_t = subsample_t

        self.word2index: dict[str, int] = {}
        self.index2word: list[str] = []
        self.word_counts: np.ndarray | None = None  # raw counts, indexed by id
        self.noise_dist: np.ndarray | None = None    # for negative sampling

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def build(self, corpus: list[list[str]]) -> "Vocabulary":
        """
        Build vocab from a tokenised corpus (list of sentences,
        each sentence is a list of tokens).
        """
        counter = Counter(w for sent in corpus for w in sent)

        # keep words with count >= min_freq
        self.index2word = [self.UNK_TOKEN]
        self.word2index = {self.UNK_TOKEN: 0}

        for word, count in counter.most_common():
            if count < self.min_freq:
                break
            idx = len(self.index2word)
            self.index2word.append(word)
            self.word2index[word] = idx

        # store raw counts (index-aligned)
        counts = np.zeros(len(self.index2word), dtype=np.float64)
        for word, idx in self.word2index.items():
            counts[idx] = counter.get(word, 0)
        self.word_counts = counts

        # noise distribution: p(w) ∝ count(w)^0.75  (Mikolov et al.)
        powered = counts ** 0.75
        self.noise_dist = powered / powered.sum()

        return self

    @property
    def size(self) -> int:
        return len(self.index2word)

    def encode(self, tokens: list[str]) -> list[int]:
        """Map a list of tokens to their indices (UNK for unknown)."""
        unk = self.word2index[self.UNK_TOKEN]
        return [self.word2index.get(t, unk) for t in tokens]

    def subsample_mask(self, ids: list[int], rng: np.random.Generator) -> list[bool]:
        """
        Return a boolean mask indicating which tokens to *keep*.
        Frequent words are randomly dropped with probability
            p_discard = 1 - sqrt(t / f(w))
        where f(w) = count(w) / total_count.
        """
        if self.subsample_t <= 0:
            return [True] * len(ids)
        total = self.word_counts.sum()
        keep = []
        for idx in ids:
            freq = self.word_counts[idx] / total
            # UNK or zero-count words: drop them (also avoids division by zero)
            if freq <= 0:
                keep.append(False)
                continue
            prob_keep = min(1.0, (np.sqrt(freq / self.subsample_t) + 1)
                           * (self.subsample_t / freq))
            keep.append(rng.random() < prob_keep)
        return keep

    def sample_negatives(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample n word indices from the noise distribution."""
        return rng.choice(self.size, size=n, p=self.noise_dist)

    # ------------------------------------------------------------------
    # preprocessing (static)
    # ------------------------------------------------------------------

    @staticmethod
    def preprocess(
        text: str,
        stopwords: set[str] | None = None,
        min_word_len: int = 3,
    ) -> list[str]:
        """
        Clean and tokenise a single document.

        Steps:
            1. Remove words containing digits  (e.g. '20th', 'h2o')
            2. Lowercase, keep only letters and spaces
            3. Remove short words  (length < min_word_len)
            4. Collapse multiple spaces
            5. Tokenise on whitespace
            6. Remove stopwords

        Args:
            text:          raw input string
            stopwords:     set of words to drop (defaults to English stopwords)
            min_word_len:  minimum word length to keep (default 3)

        Returns:
            list of cleaned tokens
        """
        if stopwords is None:
            stopwords = _get_stopwords()

        x = re.sub(r'\w*\d\w*', ' ', text)                       # drop words with digits
        x = re.sub(r'[^a-zA-Z\s]', ' ', x.lower())               # lowercase, letters only
        pattern = rf'\b\w{{1,{min_word_len - 1}}}\b'
        x = re.sub(pattern, ' ', x)                               # drop short words
        x = re.sub(r' +', ' ', x).strip()                         # collapse spaces

        tokens = x.split()
        return [t for t in tokens if t not in stopwords]

    @staticmethod
    def preprocess_corpus(
        documents: list[str],
        stopwords: set[str] | None = None,
        min_word_len: int = 3,
    ) -> list[list[str]]:
        """
        Apply preprocess() to every document.

        Args:
            documents:     list of raw text strings
            stopwords:     set of words to drop
            min_word_len:  minimum word length to keep

        Returns:
            list of token lists (one per document)
        """
        return [
            Vocabulary.preprocess(doc, stopwords, min_word_len)
            for doc in documents
        ]


# ------------------------------------------------------------------
# simple tokeniser (kept for convenience / quick experiments)
# ------------------------------------------------------------------

def tokenise(text: str) -> list[str]:
    """Lowercase + keep only alphabetic tokens of length >= 2."""
    text = text.lower()
    tokens = re.findall(r"[a-z]+", text)
    return [t for t in tokens if len(t) >= 2]