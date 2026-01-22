"""
Collapsed Gibbs Sampling LDA - sklearn-compatible implementation.

A drop-in replacement for sklearn.decomposition.LatentDirichletAllocation
using collapsed Gibbs sampling instead of variational inference.
"""

import numpy as np
from scipy.sparse import issparse, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, check_array

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        """Dummy decorator when numba is not available."""
        def decorator(func):
            return func
        return decorator


@jit(nopython=True, cache=True)
def _gibbs_iteration_numba(doc_ids, word_ids, topic_assignments, n_dk, n_kw, n_k,
                           alpha, beta, beta_V, n_components, random_state):
    """Numba-optimized Gibbs sampling iteration.

    Parameters
    ----------
    doc_ids : ndarray of shape (n_tokens,)
        Document index for each token.
    word_ids : ndarray of shape (n_tokens,)
        Word index for each token.
    topic_assignments : ndarray of shape (n_tokens,)
        Current topic assignment for each token.
    n_dk : ndarray of shape (n_docs, n_components)
        Document-topic counts.
    n_kw : ndarray of shape (n_components, n_vocab)
        Topic-word counts.
    n_k : ndarray of shape (n_components,)
        Topic totals.
    alpha : float
        Document-topic prior.
    beta : float
        Topic-word prior.
    beta_V : float
        beta * vocabulary_size.
    n_components : int
        Number of topics.
    random_state : int
        Random seed for this iteration.
    """
    np.random.seed(random_state)
    n_tokens = len(doc_ids)
    p = np.zeros(n_components)

    for i in range(n_tokens):
        d = doc_ids[i]
        w = word_ids[i]
        k_old = topic_assignments[i]

        # Decrement counts
        n_dk[d, k_old] -= 1
        n_kw[k_old, w] -= 1
        n_k[k_old] -= 1

        # Compute unnormalized probabilities
        for k in range(n_components):
            p[k] = (n_dk[d, k] + alpha) * (n_kw[k, w] + beta) / (n_k[k] + beta_V)

        # Normalize
        p_sum = 0.0
        for k in range(n_components):
            p_sum += p[k]
        for k in range(n_components):
            p[k] /= p_sum

        # Sample from cumulative distribution
        u = np.random.random()
        cumsum = 0.0
        k_new = n_components - 1
        for k in range(n_components):
            cumsum += p[k]
            if u < cumsum:
                k_new = k
                break

        # Increment counts
        topic_assignments[i] = k_new
        n_dk[d, k_new] += 1
        n_kw[k_new, w] += 1
        n_k[k_new] += 1


@jit(nopython=True, cache=True)
def _init_counts_numba(doc_ids, word_ids, topic_assignments, n_dk, n_kw, n_k):
    """Numba-optimized count initialization."""
    n_tokens = len(doc_ids)
    for i in range(n_tokens):
        d = doc_ids[i]
        w = word_ids[i]
        k = topic_assignments[i]
        n_dk[d, k] += 1
        n_kw[k, w] += 1
        n_k[k] += 1


@jit(nopython=True, cache=True)
def _inference_iteration_numba(doc_ids, word_ids, topic_assignments, n_dk,
                                phi, alpha, n_components, random_state):
    """Numba-optimized inference iteration (fixed phi)."""
    np.random.seed(random_state)
    n_tokens = len(doc_ids)
    p = np.zeros(n_components)

    for i in range(n_tokens):
        d = doc_ids[i]
        w = word_ids[i]
        k_old = topic_assignments[i]

        # Decrement
        n_dk[d, k_old] -= 1

        # Compute probabilities
        for k in range(n_components):
            p[k] = (n_dk[d, k] + alpha) * phi[k, w]

        # Normalize
        p_sum = 0.0
        for k in range(n_components):
            p_sum += p[k]
        for k in range(n_components):
            p[k] /= p_sum

        # Sample
        u = np.random.random()
        cumsum = 0.0
        k_new = n_components - 1
        for k in range(n_components):
            cumsum += p[k]
            if u < cumsum:
                k_new = k
                break

        # Increment
        topic_assignments[i] = k_new
        n_dk[d, k_new] += 1


class GibbsLDA(TransformerMixin, BaseEstimator):
    """Latent Dirichlet Allocation with collapsed Gibbs sampling.

    Parameters
    ----------
    n_components : int, default=10
        Number of topics.

    doc_topic_prior : float, default=None
        Prior of document topic distribution `theta`. If None, defaults to
        `1 / n_components`.

    topic_word_prior : float, default=None
        Prior of topic word distribution `beta`. If None, defaults to
        `1 / n_components`.

    max_iter : int, default=100
        Maximum number of Gibbs sampling iterations.

    n_jobs : int, default=None
        Reserved for future parallelization. Currently unused.

    verbose : int, default=0
        Verbosity level.

    random_state : int, RandomState instance, default=None
        Controls the random seed for reproducibility.

    burn_in : int, default=50
        Number of iterations before collecting samples.

    sample_every : int, default=5
        Collect a sample every `sample_every` iterations after burn-in.

    n_samples : int, default=10
        Number of samples to average for final estimates.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Topic-word distribution. `components_[k, w]` represents the
        pseudo-count of word `w` in topic `k` (plus prior).

    n_features_in_ : int
        Number of features (vocabulary size) seen during fit.

    n_iter_ : int
        Number of iterations completed.

    bound_ : float
        Final perplexity value.

    doc_topic_prior_ : float
        Effective document-topic prior (alpha).

    topic_word_prior_ : float
        Effective topic-word prior (beta).
    """

    def __init__(
        self,
        n_components=10,
        doc_topic_prior=None,
        topic_word_prior=None,
        max_iter=100,
        n_jobs=None,
        verbose=0,
        random_state=None,
        burn_in=50,
        sample_every=5,
        n_samples=10,
    ):
        self.n_components = n_components
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.burn_in = burn_in
        self.sample_every = sample_every
        self.n_samples = n_samples

    def _expand_sparse_to_word_sequence(self, X):
        """Convert document-word matrix to word-level arrays.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_docs, n_features)
            Document-word count matrix.

        Returns
        -------
        doc_ids : ndarray of shape (n_total_words,)
            Document index for each word token.
        word_ids : ndarray of shape (n_total_words,)
            Word (feature) index for each word token.
        """
        if issparse(X):
            X = X.tocsr()
        else:
            X = csr_matrix(X)

        n_docs = X.shape[0]

        doc_ids = []
        word_ids = []

        for d in range(n_docs):
            start, end = X.indptr[d], X.indptr[d + 1]
            for idx in range(start, end):
                w = X.indices[idx]
                count = int(X.data[idx])
                doc_ids.extend([d] * count)
                word_ids.extend([w] * count)

        return np.array(doc_ids, dtype=np.int32), np.array(word_ids, dtype=np.int32)

    def _init_latent_vars(self, n_docs, n_features):
        """Initialize count matrices and set prior defaults.

        Parameters
        ----------
        n_docs : int
            Number of documents.
        n_features : int
            Vocabulary size.
        """
        self.n_features_in_ = n_features

        # Set default priors if not specified
        if self.doc_topic_prior is None:
            self.doc_topic_prior_ = 1.0 / self.n_components
        else:
            self.doc_topic_prior_ = self.doc_topic_prior

        if self.topic_word_prior is None:
            self.topic_word_prior_ = 1.0 / self.n_components
        else:
            self.topic_word_prior_ = self.topic_word_prior

        # Initialize count matrices
        self._n_dk = np.zeros((n_docs, self.n_components), dtype=np.float64)
        self._n_kw = np.zeros((self.n_components, n_features), dtype=np.float64)
        self._n_k = np.zeros(self.n_components, dtype=np.float64)

        # Sample accumulation
        self._n_kw_samples = []

    def _initialize_topic_assignments(self, rng):
        """Randomly assign initial topics to all word tokens.

        Parameters
        ----------
        rng : RandomState
            Random number generator.
        """
        n_tokens = len(self._doc_ids)
        self._topic_assignments = rng.randint(0, self.n_components, size=n_tokens).astype(np.int32)

        # Update count matrices based on initial assignments
        _init_counts_numba(
            self._doc_ids, self._word_ids, self._topic_assignments,
            self._n_dk, self._n_kw, self._n_k
        )

    def _gibbs_iteration(self, rng):
        """Perform one Gibbs sampling sweep through all word tokens.

        Parameters
        ----------
        rng : RandomState
            Random number generator.
        """
        alpha = self.doc_topic_prior_
        beta = self.topic_word_prior_
        V = self.n_features_in_
        beta_V = beta * V

        # Generate a random seed for this iteration
        iteration_seed = rng.randint(0, 2**31 - 1)

        _gibbs_iteration_numba(
            self._doc_ids, self._word_ids, self._topic_assignments,
            self._n_dk, self._n_kw, self._n_k,
            alpha, beta, beta_V, self.n_components, iteration_seed
        )

    def _compute_components(self):
        """Average collected samples into components_."""
        if len(self._n_kw_samples) == 0:
            # No samples collected, use current counts
            self.components_ = self._n_kw + self.topic_word_prior_
        else:
            # Average over collected samples
            avg_n_kw = np.mean(self._n_kw_samples, axis=0)
            self.components_ = avg_n_kw + self.topic_word_prior_

    def fit(self, X, y=None):
        """Fit the model with X using Gibbs sampling.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Document-word matrix.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = check_array(X, accept_sparse='csr', dtype=[np.float64, np.float32, np.int64, np.int32])
        n_docs, n_features = X.shape

        rng = check_random_state(self.random_state)

        # Expand to word sequences
        self._doc_ids, self._word_ids = self._expand_sparse_to_word_sequence(X)

        # Initialize
        self._init_latent_vars(n_docs, n_features)
        self._initialize_topic_assignments(rng)

        # Calculate total iterations needed
        total_iter = self.burn_in + self.n_samples * self.sample_every
        if self.max_iter < total_iter and self.verbose:
            print(f"Warning: max_iter ({self.max_iter}) < burn_in + n_samples * sample_every ({total_iter})")

        # Run Gibbs sampling
        samples_collected = 0
        for iteration in range(self.max_iter):
            self._gibbs_iteration(rng)

            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}")

            # Collect samples after burn-in
            if iteration >= self.burn_in and samples_collected < self.n_samples:
                if (iteration - self.burn_in) % self.sample_every == 0:
                    self._n_kw_samples.append(self._n_kw.copy())
                    samples_collected += 1
                    if self.verbose:
                        print(f"  Collected sample {samples_collected}/{self.n_samples}")

        self.n_iter_ = self.max_iter

        # Compute final components
        self._compute_components()

        # Compute final perplexity
        self.bound_ = self.perplexity(X)

        return self

    def _run_inference(self, X, max_iter=None):
        """Run Gibbs sampling on new documents with fixed topic-word distribution.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Document-word matrix.

        max_iter : int, default=None
            Number of inference iterations. Defaults to burn_in + sample_every.

        Returns
        -------
        doc_topic_distr : ndarray of shape (n_samples, n_components)
            Document-topic distribution.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse='csr', dtype=[np.float64, np.float32, np.int64, np.int32])

        n_docs, n_features = X.shape

        if n_features != self.n_features_in_:
            raise ValueError(
                f"Number of features {n_features} does not match "
                f"n_features_in_ {self.n_features_in_}"
            )

        if max_iter is None:
            max_iter = self.burn_in + self.sample_every

        rng = check_random_state(self.random_state)

        # Expand to word sequences
        doc_ids, word_ids = self._expand_sparse_to_word_sequence(X)
        n_tokens = len(doc_ids)

        if n_tokens == 0:
            # Empty documents
            return np.full((n_docs, self.n_components), 1.0 / self.n_components)

        # Initialize doc-topic counts
        n_dk = np.zeros((n_docs, self.n_components), dtype=np.float64)

        # Random initial topic assignments
        topic_assignments = rng.randint(0, self.n_components, size=n_tokens).astype(np.int32)

        # Initialize counts
        for i in range(n_tokens):
            d = doc_ids[i]
            k = topic_assignments[i]
            n_dk[d, k] += 1

        # Compute normalized topic-word distribution (phi) from components_
        # phi[k, w] = components_[k, w] / sum_w'(components_[k, w'])
        phi = np.ascontiguousarray(self.components_ / self.components_.sum(axis=1, keepdims=True))

        alpha = self.doc_topic_prior_

        # Run inference Gibbs sampling
        for iteration in range(max_iter):
            iteration_seed = rng.randint(0, 2**31 - 1)
            _inference_iteration_numba(
                doc_ids, word_ids, topic_assignments, n_dk,
                phi, alpha, self.n_components, iteration_seed
            )

        # Compute document-topic distribution
        # theta[d, k] = (n_dk[d, k] + alpha) / sum_k'(n_dk[d, k'] + alpha)
        doc_topic_distr = n_dk + alpha
        doc_topic_distr /= doc_topic_distr.sum(axis=1, keepdims=True)

        return doc_topic_distr

    def transform(self, X):
        """Transform documents to topic distribution.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Document-word matrix.

        Returns
        -------
        doc_topic_distr : ndarray of shape (n_samples, n_components)
            Document-topic distribution for X.
        """
        return self._run_inference(X)

    def fit_transform(self, X, y=None):
        """Fit the model and return document-topic distribution.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Document-word matrix.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        doc_topic_distr : ndarray of shape (n_samples, n_components)
            Document-topic distribution for X.
        """
        self.fit(X, y)

        # Compute document-topic distribution from training counts
        doc_topic_distr = self._n_dk + self.doc_topic_prior_
        doc_topic_distr /= doc_topic_distr.sum(axis=1, keepdims=True)

        return doc_topic_distr

    def partial_fit(self, X, y=None):
        """Incrementally fit the model with new documents.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Document-word matrix.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = check_array(X, accept_sparse='csr', dtype=[np.float64, np.float32, np.int64, np.int32])
        n_docs, n_features = X.shape

        rng = check_random_state(self.random_state)

        # Expand documents to word sequences
        new_doc_ids, new_word_ids = self._expand_sparse_to_word_sequence(X)
        n_new_tokens = len(new_doc_ids)

        if n_new_tokens == 0:
            return self

        # Check if this is the first call
        is_first_call = not hasattr(self, 'components_') or self.components_ is None

        if is_first_call:
            # First call - initialize from scratch
            self._init_latent_vars(n_docs, n_features)
            self.n_iter_ = 0
        else:
            # Subsequent calls - verify feature count matches
            if n_features != self.n_features_in_:
                raise ValueError(
                    f"Number of features {n_features} does not match "
                    f"n_features_in_ {self.n_features_in_}"
                )

        # Initialize counts for new documents
        new_n_dk = np.zeros((n_docs, self.n_components), dtype=np.float64)
        new_topic_assignments = np.zeros(n_new_tokens, dtype=np.int32)

        alpha = self.doc_topic_prior_
        beta = self.topic_word_prior_
        V = self.n_features_in_
        beta_V = beta * V

        # Initialize topic assignments for new tokens
        if is_first_call:
            # Random initialization on first call
            for i in range(n_new_tokens):
                w = new_word_ids[i]
                d = new_doc_ids[i]

                k = rng.randint(0, self.n_components)

                new_topic_assignments[i] = k
                new_n_dk[d, k] += 1
                self._n_kw[k, w] += 1
                self._n_k[k] += 1
        else:
            # Use existing model to initialize
            phi = self.components_ / self.components_.sum(axis=1, keepdims=True)

            for i in range(n_new_tokens):
                w = new_word_ids[i]
                d = new_doc_ids[i]

                # Sample initial topic based on phi
                p = phi[:, w].copy()
                p /= p.sum()
                k = rng.choice(self.n_components, p=p)

                new_topic_assignments[i] = k
                new_n_dk[d, k] += 1
                self._n_kw[k, w] += 1
                self._n_k[k] += 1

        # Run a few iterations on new documents
        n_partial_iter = min(20, self.max_iter)

        for iteration in range(n_partial_iter):
            for i in range(n_new_tokens):
                d = new_doc_ids[i]
                w = new_word_ids[i]
                k_old = new_topic_assignments[i]

                # Decrement
                new_n_dk[d, k_old] -= 1
                self._n_kw[k_old, w] -= 1
                self._n_k[k_old] -= 1

                # Sample new topic
                p = (new_n_dk[d, :] + alpha) * (self._n_kw[:, w] + beta) / (self._n_k + beta_V)
                p /= p.sum()
                k_new = rng.choice(self.n_components, p=p)

                # Increment
                new_topic_assignments[i] = k_new
                new_n_dk[d, k_new] += 1
                self._n_kw[k_new, w] += 1
                self._n_k[k_new] += 1

        # Update components
        self.components_ = self._n_kw + self.topic_word_prior_
        self.n_iter_ += n_partial_iter

        return self

    def _log_likelihood(self, X):
        """Compute log-likelihood of data under the model.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Document-word matrix.

        Returns
        -------
        ll : float
            Log-likelihood of the data.
        total_words : int
            Total number of word tokens.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse='csr', dtype=[np.float64, np.float32, np.int64, np.int32])

        # Get document-topic distribution
        doc_topic_distr = self.transform(X)

        # Get topic-word distribution
        phi = self.components_ / self.components_.sum(axis=1, keepdims=True)

        # Compute log-likelihood: sum_d sum_w n_dw * log(sum_k theta_dk * phi_kw)
        if issparse(X):
            X = X.tocsr()
        else:
            X = csr_matrix(X)

        ll = 0.0
        total_words = 0

        n_docs = X.shape[0]
        for d in range(n_docs):
            start, end = X.indptr[d], X.indptr[d + 1]
            for idx in range(start, end):
                w = X.indices[idx]
                count = X.data[idx]

                # p(w|d) = sum_k theta_dk * phi_kw
                p_w_d = np.dot(doc_topic_distr[d, :], phi[:, w])

                if p_w_d > 0:
                    ll += count * np.log(p_w_d)

                total_words += count

        return ll, int(total_words)

    def score(self, X, y=None):
        """Calculate log-likelihood of data (higher is better).

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Document-word matrix.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        score : float
            Log-likelihood of the data.
        """
        ll, _ = self._log_likelihood(X)
        return ll

    def perplexity(self, X, sub_sampling=False):
        """Calculate perplexity (lower is better).

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Document-word matrix.

        sub_sampling : bool, default=False
            Ignored, present for sklearn API compatibility.

        Returns
        -------
        perplexity : float
            Perplexity value.
        """
        ll, total_words = self._log_likelihood(X)

        if total_words == 0:
            return float('inf')

        # perplexity = exp(-log_likelihood / total_words)
        return np.exp(-ll / total_words)
