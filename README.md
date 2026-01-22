# GibbsLDA

A drop-in replacement for `sklearn.decomposition.LatentDirichletAllocation` using collapsed Gibbs sampling instead of variational inference.

## Installation

```bash
pip install numpy scipy scikit-learn numba
```

Numba is optional but highly recommended for performance (150x speedup).

## Quick Start

```python
from sklearn.feature_extraction.text import CountVectorizer
from gibbs_lda import GibbsLDA

# Prepare documents
docs = [
    "machine learning is useful for data science",
    "neural networks are a type of machine learning",
    "python is great for data analysis",
    "deep learning uses neural networks",
]

# Create document-term matrix
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)

# Fit LDA model
lda = GibbsLDA(n_components=2, random_state=42)
lda.fit(X)

# Get document-topic distributions
doc_topics = lda.transform(X)

# Print top words per topic
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
    print(f"Topic {topic_idx}: {' '.join(top_words)}")
```

## API Reference

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_components` | int | 10 | Number of topics |
| `doc_topic_prior` | float | None | Alpha prior (default: 1/n_components) |
| `topic_word_prior` | float | None | Beta prior (default: 1/n_components) |
| `max_iter` | int | 100 | Number of Gibbs sampling iterations |
| `random_state` | int | None | Random seed for reproducibility |
| `verbose` | int | 0 | Verbosity level |
| `burn_in` | int | 50 | Iterations before collecting samples |
| `sample_every` | int | 5 | Sample collection interval |
| `n_samples` | int | 10 | Number of samples to average |

### Attributes (after fitting)

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `components_` | (n_components, n_features) | Topic-word distribution |
| `n_features_in_` | int | Vocabulary size |
| `n_iter_` | int | Iterations completed |
| `bound_` | float | Final perplexity |
| `doc_topic_prior_` | float | Effective alpha |
| `topic_word_prior_` | float | Effective beta |

### Methods

#### `fit(X, y=None)`
Train the model using Gibbs sampling.

```python
lda = GibbsLDA(n_components=10)
lda.fit(X)
```

#### `transform(X)`
Get document-topic distributions for documents.

```python
doc_topics = lda.transform(X)  # shape: (n_docs, n_components)
```

#### `fit_transform(X, y=None)`
Fit and return document-topic distributions.

```python
doc_topics = lda.fit_transform(X)
```

#### `partial_fit(X, y=None)`
Incremental learning with new documents.

```python
lda = GibbsLDA(n_components=10)
lda.partial_fit(X_batch1)
lda.partial_fit(X_batch2)
```

#### `score(X, y=None)`
Return log-likelihood of data (higher is better).

```python
ll = lda.score(X)
```

#### `perplexity(X)`
Return perplexity (lower is better).

```python
perp = lda.perplexity(X)
```

## Comparison with sklearn

```python
from sklearn.decomposition import LatentDirichletAllocation
from gibbs_lda import GibbsLDA

# sklearn (variational inference)
sklearn_lda = LatentDirichletAllocation(n_components=20, max_iter=10)
sklearn_lda.fit(X)

# GibbsLDA (collapsed Gibbs sampling)
gibbs_lda = GibbsLDA(n_components=20, max_iter=100)
gibbs_lda.fit(X)

# Same API
assert sklearn_lda.components_.shape == gibbs_lda.components_.shape
assert sklearn_lda.transform(X).shape == gibbs_lda.transform(X).shape
```

### Performance (2000 docs, 5000 vocab, 229K tokens)

| Method | Time | Iterations | Perplexity |
|--------|------|------------|------------|
| sklearn Variational | 6.2s | 10 | 1280 |
| GibbsLDA (Numba) | 2.4s | 100 | 765 |

GibbsLDA is 2.6x faster while achieving 40% better perplexity.

## When to Use Gibbs vs Variational

**Use GibbsLDA when:**
- Topic quality matters more than speed
- You need accurate posterior estimates
- Working with smaller/medium corpora (< 100K docs)
- You want to collect multiple samples for uncertainty

**Use sklearn's variational LDA when:**
- You need online/streaming updates
- Working with very large corpora
- Approximate inference is acceptable

## Advanced Usage

### Adjusting Sampling Parameters

```python
# More burn-in for better convergence
lda = GibbsLDA(
    n_components=20,
    max_iter=200,
    burn_in=100,      # Wait longer before collecting
    sample_every=10,  # Less correlated samples
    n_samples=10,     # Average over 10 samples
)
```

### Symmetric vs Asymmetric Priors

```python
# Sparse topics (low alpha)
lda = GibbsLDA(n_components=20, doc_topic_prior=0.01)

# Dense topics (high alpha)
lda = GibbsLDA(n_components=20, doc_topic_prior=1.0)
```

### Working with Sparse Matrices

```python
from scipy.sparse import csr_matrix

X_sparse = csr_matrix(X)
lda.fit(X_sparse)  # Works directly with sparse matrices
```

## Algorithm Details

GibbsLDA uses collapsed Gibbs sampling where topic-word and document-topic distributions are integrated out. Each iteration samples a new topic for every word token:

```
p(z = k) ∝ (n_dk + α) × (n_kw + β) / (n_k + Vβ)
```

Where:
- `n_dk` = count of topic k in document d
- `n_kw` = count of word w in topic k
- `n_k` = total words assigned to topic k
- `α` = document-topic prior
- `β` = topic-word prior
- `V` = vocabulary size

After burn-in, multiple samples are collected and averaged for final estimates.

## License

MIT
