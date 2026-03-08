# Word2vec CBOW with Negative Sampling

A pure NumPy implementation of the Continuous Bag-of-Words (CBOW) architecture using Negative Sampling (NEG) for efficient word embedding training.

## 🚀 Overview
This repository contains a from-scratch implementation of the Word2vec CBOW model. Unlike the skip-gram model which predicts context from a target word, **CBOW predicts the target word from a bag of context words**. To handle large vocabularies efficiently, this implementation uses **Negative Sampling**, avoiding the computationally expensive full softmax layer.

## 🧠 Architecture & Math

### 1. Forward Pass
For a target word and its context (window size $C$):
1. **Projection ($h$):** The hidden state is the average of the input vectors ($W_{in}$) of context words:
   $$h = \frac{1}{2C} \sum_{c \in \text{context}} W_{in}[c]$$
2. **Scores:** - **Positive:** $\text{score}_{pos} = W_{out}[\text{target}] \cdot h$
   - **Negative:** $\text{score}_{neg,j} = W_{out}[\text{neg}_j] \cdot h$

### 2. Loss Function
The model maximizes the probability of the actual target word while minimizing the probability of $K$ noise samples using the binary logistic regression objective:
$$L = -\log \sigma(\text{score}_{pos}) - \sum_{j=1}^{K} \log \sigma(-\text{score}_{neg,j})$$

## 📂 Project Structure
- `cbow.py`: The core model class including forward, backward, and SGD update logic.
- `vocab.py`: Text preprocessing (digit removal, stopword filtering) and vocabulary building.
- `dataset.py`: Batch generator that handles context window slicing and negative sampling.
- `train.py`: Training loop with linear learning rate decay and evaluation helpers (similarity/analogy).
- `w2v_training_example.ipynb`: A complete walkthrough using a TripAdvisor hotel review dataset.

## 🛠️ Usage

### Installation
Ensure you have the following dependencies:
```bash
pip install numpy pandas tqdm nltk matplotlib
```
## Basic Training
```python
from vocab import Vocabulary
from dataset import CBOWDataset
from cbow import CBOWModel
from train import train

# 1. Prepare data
corpus = Vocabulary.preprocess_corpus(your_list_of_sentences)
vocab = Vocabulary(min_freq=5).build(corpus)

# 2. Create Dataset
dataset = CBOWDataset(corpus, vocab, window=5, num_neg=5)

# 3. Train
model = CBOWModel(vocab.size, embed_dim=100)
losses = train(model, dataset, epochs=10, lr_start=0.025)
```

## Evaluation
Find similar words using cosine similarity:
```python
from train import most_similar
results = most_similar("hotel", vocab, model, top_k=5)
```

Solve analogies ($a$ is to $b$ as $c$ is to ?):
```python
from train import analogy
# Example: dirty -> clean as rude -> ?
results = analogy("dirty", "clean", "rude", vocab, model)
```

## 📉 Implementation Details

* **Numerical Stability**: Uses a custom sigmoid function to prevent overflow by handling positive and negative inputs separately.
* **Sparse Updates**: Utilizes `np.add.at` for efficient index-based weight updates, ensuring that gradients for the same word index in a batch are accumulated correctly.
* **Preprocessing**: Includes automatic NLTK stopword downloading and regex-based cleaning to remove digits, handle casing, and filter short words.
