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
| `doc_topic_prior` | float | None | Initial alpha prior (default: 1/n_components) |
| `topic_word_prior` | float | None | Beta prior (default: 1/n_components) |
| `max_iter` | int | 100 | Number of Gibbs sampling iterations |
| `random_state` | int | None | Random seed for reproducibility |
| `verbose` | int | 0 | Verbosity level |
| `burn_in` | int | 50 | Iterations before collecting samples |
| `sample_every` | int | 5 | Sample collection interval |
| `n_samples` | int | 10 | Number of samples to average |
| `optimize_doc_topic_prior` | bool | True | Learn asymmetric alpha from data |
| `doc_topic_prior_optimize_interval` | int | 10 | Optimize alpha every N iterations |
| `use_sparse_lda` | bool | True | Use SparseLDA for faster sampling with many topics |

### Attributes (after fitting)

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `components_` | (n_components, n_features) | Topic-word distribution |
| `n_features_in_` | int | Vocabulary size |
| `n_iter_` | int | Iterations completed |
| `bound_` | float | Final perplexity |
| `doc_topic_prior_` | (n_components,) | Learned alpha vector (asymmetric) |
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

### Alpha Hyperparameter Optimization

By default, GibbsLDA learns an asymmetric alpha (document-topic prior) from the data using fixed-point iteration with digamma functions. This allows some topics to be more prevalent than others.

```python
# Default: learn asymmetric alpha from data
lda = GibbsLDA(n_components=20)
lda.fit(X)
print(lda.doc_topic_prior_)  # array of learned values per topic

# Disable optimization (use fixed symmetric alpha)
lda = GibbsLDA(n_components=20, optimize_doc_topic_prior=False)

# Custom initial alpha (will be optimized unless disabled)
lda = GibbsLDA(n_components=20, doc_topic_prior=0.1)
```

The learned alpha values reveal topic prevalence:
- High alpha topics appear frequently across documents
- Low alpha topics are more specialized/rare

### Working with Sparse Matrices

```python
from scipy.sparse import csr_matrix

X_sparse = csr_matrix(X)
lda.fit(X_sparse)  # Works directly with sparse matrices
```

## Algorithm Details

GibbsLDA uses collapsed Gibbs sampling where topic-word and document-topic distributions are integrated out. Each iteration samples a new topic for every word token:

```
p(z = k) ∝ (n_dk + α_k) × (n_kw + β) / (n_k + Vβ)
```

Where:
- `n_dk` = count of topic k in document d
- `n_kw` = count of word w in topic k
- `n_k` = total words assigned to topic k
- `α_k` = document-topic prior for topic k (learned)
- `β` = topic-word prior
- `V` = vocabulary size

After burn-in, multiple samples are collected and averaged for final estimates.

### Alpha Optimization

The document-topic prior α is optimized using fixed-point iteration (Wallach, Mimno, McCallum, NIPS 2009):

```
α_k^{new} = α_k × Σ_d[ψ(n_dk + α_k) - ψ(α_k)] / Σ_d[ψ(n_d + Σα) - ψ(Σα)]
```

This learns an asymmetric prior where each topic can have different prevalence across the corpus.

### SparseLDA Optimization

When `use_sparse_lda=True` (default), sampling uses the three-bucket decomposition from Yao, Mimno, McCallum (KDD 2009):

```
p(z=k) ∝ s_k + r_k + q_k

where:
  s_k = α_k × β / (n_k + Vβ)              [smoothing bucket]
  r_k = n_dk × β / (n_k + Vβ)             [doc-topic bucket]
  q_k = (n_dk + α_k) × n_kw / (n_k + Vβ)  [topic-word bucket]
```

This exploits sparsity using incrementally-maintained sparse indices for both doc-topic and topic-word counts:
- Smoothing bucket: precomputed sum, updated incrementally
- Doc-topic bucket: iterates only over K_d topics present in document (~10 on average)
- Topic-word bucket: iterates only over K_w topics using the word (~4 on average)

In practice, ~98% of samples come from the topic-word bucket with O(K_w) cost instead of O(K).

| Topics (K) | Standard | SparseLDA | Speedup |
|------------|----------|-----------|---------|
| 50         | 2.1s     | 1.5s      | 1.4x    |
| 100        | 3.7s     | 2.3s      | 1.6x    |
| 200        | 7.5s     | 4.0s      | 1.9x    |
| 500        | 22.2s    | 10.4s     | 2.1x    |

Speedup increases with K since K_d and K_w grow much slower than K.

## License

MIT
