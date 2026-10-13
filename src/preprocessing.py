"""Text Preprocessing, TF-IDF Vectorizer & Domain Feature Extraction.

Implements:
  1. `normalize_ticket_text`: Regex entity masking for URLs, emails, invoice/order IDs,
     error codes, and currency amounts.
  2. `TfidfVectorizer`: Word & Character N-Gram TF-IDF Vectorizer built
     with NumPy (supporting `ngram_range`, `max_features`, `sublinear_tf`,
     smooth IDF weighting, and L2 normalization).
  3. `TicketMetadataExtractor`: Extracts 9 structural and urgency signals (exclamation
     density, uppercase ratio, error code indicators, urgency keyword hits) with
     z-score standardization.
  4. `HybridFeaturePipeline`: Concatenates Word TF-IDF + Char N-Gram TF-IDF + Standardized
     Metadata into a unified feature matrix.
"""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b")
ORDER_ID_PATTERN = re.compile(r"(?:#?ORD-\d+|INV-\d+|TKT-\d+)", re.IGNORECASE)
ERR_CODE_PATTERN = re.compile(
    r"\b(?:ERR_\d+|E_TIMEOUT_\d+|AUTH_FAIL_\d+|SQL_DEADLOCK|HTTP_\d{3})\b",
    re.IGNORECASE,
)
AMOUNT_PATTERN = re.compile(r"\$\d+(?:,\d{3})*(?:\.\d{2})?")
MULTI_SPACE_PATTERN = re.compile(r"\s+")

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "he", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "was",
    "were", "will", "with", "this", "my", "our", "we", "you", "your", "i", "me",
    "hi", "hello", "hey", "there", "team", "thanks", "thank", "please", "best",
    "regards",
}

URGENCY_KEYWORDS = {
    "urgent", "critical", "emergency", "immediately", "asap", "blocked",
    "locked", "crashes", "unresponsive", "unauthorized", "shattered",
    "suspended", "broken", "frozen", "wipes", "lost", "double", "twice",
}

LOW_PRIORITY_KEYWORDS = {
    "wondering", "suggestion", "idea", "minor", "typo", "question",
    "future", "nice", "cool", "appreciate", "when", "where", "how",
}


def normalize_ticket_text(text: str) -> str:
    """Normalize raw support ticket text while preserving domain tokens."""
    if not isinstance(text, str):
        text = str(text)
    cleaned = URL_PATTERN.sub(" __url__ ", text)
    cleaned = EMAIL_PATTERN.sub(" __email__ ", cleaned)
    cleaned = ORDER_ID_PATTERN.sub(" __ref_id__ ", cleaned)
    cleaned = ERR_CODE_PATTERN.sub(" __error_code__ ", cleaned)
    cleaned = AMOUNT_PATTERN.sub(" __currency_amount__ ", cleaned)
    cleaned = cleaned.lower()
    cleaned = re.sub(r"[^a-z0-9_!?\s]", " ", cleaned)
    cleaned = MULTI_SPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned


class TfidfVectorizer:
    """TF-IDF Vectorizer implemented with NumPy.

    Supports word n-grams, character n-grams, stop-word filtering, sublinear TF
    scaling (`1 + log(tf)`), smooth IDF (`log((1 + N) / (1 + df)) + 1`), and L2
    row normalization.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 3000,
        sublinear_tf: bool = True,
        analyzer: str = "word",
    ) -> None:
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.analyzer = analyzer
        self.vocab_: Dict[str, int] = {}
        self.feature_names_: List[str] = []
        self.idf_: np.ndarray = np.array([])

    def _extract_ngrams(self, text: str) -> List[str]:
        cleaned = normalize_ticket_text(text)
        min_n, max_n = self.ngram_range
        if self.analyzer == "char":
            padded = f" {cleaned} "
            tokens = []
            for n in range(min_n, max_n + 1):
                for i in range(len(padded) - n + 1):
                    tokens.append(padded[i : i + n])
            return tokens

        words = [
            w for w in re.findall(r"[a-z0-9_]+", cleaned)
            if w not in STOP_WORDS and len(w) > 1
        ]
        ngrams = []
        for n in range(min_n, max_n + 1):
            if n == 1:
                ngrams.extend(words)
            else:
                for i in range(len(words) - n + 1):
                    ngrams.append(" ".join(words[i : i + n]))
        return ngrams

    def fit(self, texts: Iterable[str]) -> "TfidfVectorizer":
        docs = list(texts)
        n_docs = len(docs)
        doc_freq: Counter = Counter()
        term_freq_total: Counter = Counter()

        for doc in docs:
            ngrams = self._extract_ngrams(doc)
            term_freq_total.update(ngrams)
            doc_freq.update(set(ngrams))

        most_common = [term for term, _ in term_freq_total.most_common(self.max_features)]
        most_common.sort()

        self.feature_names_ = most_common
        self.vocab_ = {term: idx for idx, term in enumerate(most_common)}

        df_vec = np.array([doc_freq[term] for term in most_common], dtype=np.float64)
        self.idf_ = np.log((1.0 + n_docs) / (1.0 + df_vec)) + 1.0
        return self

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        docs = list(texts)
        n_docs = len(docs)
        n_feat = len(self.vocab_)
        mat = np.zeros((n_docs, n_feat), dtype=np.float64)

        for row_idx, doc in enumerate(docs):
            counts = Counter(self._extract_ngrams(doc))
            for term, count in counts.items():
                col_idx = self.vocab_.get(term)
                if col_idx is not None:
                    tf = 1.0 + math.log(count) if (self.sublinear_tf and count > 0) else float(count)
                    mat[row_idx, col_idx] = tf

        mat *= self.idf_
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        mat /= norms
        return mat

    def fit_transform(self, texts: Iterable[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)


# Compatibility alias for serialized models
ScratchTfidfVectorizer = TfidfVectorizer


class TicketMetadataExtractor:
    """Extracts 9 numeric structural & urgency signals and applies z-score standardization."""

    def __init__(self) -> None:
        self.mean_: np.ndarray = np.zeros(9, dtype=np.float64)
        self.std_: np.ndarray = np.ones(9, dtype=np.float64)

    def _raw_features(self, texts: Iterable[str]) -> np.ndarray:
        rows = []
        for raw in texts:
            text = str(raw) if raw is not None else ""
            exclamations = float(text.count("!"))
            questions = float(text.count("?"))
            alpha_chars = [c for c in text if c.isalpha()]
            upper_ratio = (
                sum(1 for c in alpha_chars if c.isupper()) / max(len(alpha_chars), 1)
            )
            words = re.findall(r"\w+", text.lower())
            word_count = float(len(words))
            has_err = 1.0 if ERR_CODE_PATTERN.search(text) else 0.0
            has_amt = 1.0 if AMOUNT_PATTERN.search(text) else 0.0
            has_ref = 1.0 if ORDER_ID_PATTERN.search(text) else 0.0
            urgency_hits = float(sum(1 for w in words if w in URGENCY_KEYWORDS))
            low_pri_hits = float(sum(1 for w in words if w in LOW_PRIORITY_KEYWORDS))

            rows.append([
                exclamations,
                questions,
                upper_ratio,
                word_count,
                has_err,
                has_amt,
                has_ref,
                urgency_hits,
                low_pri_hits,
            ])
        return np.asarray(rows, dtype=np.float64)

    def fit(self, texts: Iterable[str]) -> "TicketMetadataExtractor":
        raw = self._raw_features(texts)
        self.mean_ = np.mean(raw, axis=0)
        self.std_ = np.std(raw, axis=0)
        self.std_[self.std_ == 0.0] = 1.0
        return self

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        raw = self._raw_features(texts)
        return (raw - self.mean_) / self.std_

    def fit_transform(self, texts: Iterable[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)


class HybridFeaturePipeline:
    """Combines Word N-Gram TF-IDF, Char N-Gram TF-IDF, and Structural Metadata."""

    def __init__(
        self,
        word_ngram_range: Tuple[int, int] = (1, 2),
        word_max_features: int = 2500,
        char_max_features: int = 1500,
        sublinear_tf: bool = True,
        metadata_weight: float = 0.25,
    ) -> None:
        self.word_tfidf = TfidfVectorizer(
            ngram_range=word_ngram_range,
            max_features=word_max_features,
            sublinear_tf=sublinear_tf,
            analyzer="word",
        )
        self.char_tfidf = TfidfVectorizer(
            ngram_range=(3, 4),
            max_features=char_max_features,
            sublinear_tf=sublinear_tf,
            analyzer="char",
        )
        self.meta_extractor = TicketMetadataExtractor()
        self.metadata_weight = metadata_weight

    def fit(self, texts: Iterable[str]) -> "HybridFeaturePipeline":
        docs = list(texts)
        self.word_tfidf.fit(docs)
        self.char_tfidf.fit(docs)
        self.meta_extractor.fit(docs)
        return self

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        docs = list(texts)
        X_word = self.word_tfidf.transform(docs)
        X_char = self.char_tfidf.transform(docs)
        X_meta = self.meta_extractor.transform(docs) * self.metadata_weight
        return np.hstack([X_word, X_char, X_meta])

    def fit_transform(self, texts: Iterable[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)


class SoftmaxLogisticClassifier:
    """Multi-class Softmax Logistic Regression with Momentum and L2 Regularization."""

    def __init__(
        self,
        lr: float = 0.8,
        l2_reg: float = 1e-4,
        epochs: int = 120,
        batch_size: int = 64,
        momentum: float = 0.9,
        class_weight: Optional[str] = "balanced",
        seed: int = 42,
    ) -> None:
        self.lr = lr
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.class_weight = class_weight
        self.seed = seed
        self.classes_: np.ndarray = np.array([])
        self.W_: np.ndarray = np.array([])
        self.b_: np.ndarray = np.array([])
        self.loss_history_: List[float] = []

    def fit(self, X: np.ndarray, y: Iterable) -> "SoftmaxLogisticClassifier":
        rng = np.random.default_rng(self.seed)
        y_arr = np.asarray(list(y))
        self.classes_ = np.unique(y_arr)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[c] for c in y_arr], dtype=np.int64)

        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        Y_onehot = np.zeros((n_samples, n_classes), dtype=np.float64)
        Y_onehot[np.arange(n_samples), y_idx] = 1.0

        if self.class_weight == "balanced":
            counts = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
            class_weights = n_samples / (n_classes * np.maximum(counts, 1.0))
            sample_weights = class_weights[y_idx][:, None]
        else:
            sample_weights = np.ones((n_samples, 1), dtype=np.float64)

        self.W_ = rng.normal(0.0, 0.01, size=(n_features, n_classes))
        self.b_ = np.zeros((1, n_classes), dtype=np.float64)

        vW = np.zeros_like(self.W_)
        vb = np.zeros_like(self.b_)
        self.loss_history_ = []

        for epoch in range(self.epochs):
            perm = rng.permutation(n_samples)
            lr_t = self.lr * (0.5 * (1.0 + np.cos(np.pi * epoch / self.epochs)))
            epoch_loss = 0.0

            for start in range(0, n_samples, self.batch_size):
                batch_idx = perm[start : start + self.batch_size]
                Xb = X[batch_idx]
                Yb = Y_onehot[batch_idx]
                wb = sample_weights[batch_idx]
                bs = len(batch_idx)

                logits = Xb @ self.W_ + self.b_
                probs = _softmax(logits)

                diff = (probs - Yb) * wb
                grad_W = (Xb.T @ diff) / bs + self.l2_reg * self.W_
                grad_b = np.sum(diff, axis=0, keepdims=True) / bs

                vW = self.momentum * vW - lr_t * grad_W
                vb = self.momentum * vb - lr_t * grad_b
                self.W_ += vW
                self.b_ += vb

                eps = 1e-12
                ce_loss = -np.sum(wb * Yb * np.log(np.clip(probs, eps, 1.0))) / bs
                epoch_loss += ce_loss * bs

            self.loss_history_.append(float(epoch_loss / n_samples))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.W_ + self.b_
        return _softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]


class LinearSVMClassifier:
    """Multi-class Linear Support Vector Classifier (Squared Hinge Loss + L2 Regularization)."""

    def __init__(
        self,
        lr: float = 0.4,
        C: float = 1.5,
        epochs: int = 100,
        batch_size: int = 64,
        seed: int = 42,
    ) -> None:
        self.lr = lr
        self.C = C
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.classes_: np.ndarray = np.array([])
        self.W_: np.ndarray = np.array([])
        self.b_: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: Iterable) -> "LinearSVMClassifier":
        rng = np.random.default_rng(self.seed)
        y_arr = np.asarray(list(y))
        self.classes_ = np.unique(y_arr)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[c] for c in y_arr], dtype=np.int64)

        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        Y_ovr = -np.ones((n_samples, n_classes), dtype=np.float64)
        Y_ovr[np.arange(n_samples), y_idx] = 1.0

        self.W_ = rng.normal(0.0, 0.01, size=(n_features, n_classes))
        self.b_ = np.zeros((1, n_classes), dtype=np.float64)

        l2_lambda = 1.0 / (max(self.C, 1e-5) * n_samples)

        for epoch in range(self.epochs):
            perm = rng.permutation(n_samples)
            lr_t = self.lr / (1.0 + 0.02 * epoch)
            for start in range(0, n_samples, self.batch_size):
                batch_idx = perm[start : start + self.batch_size]
                Xb = X[batch_idx]
                Yb = Y_ovr[batch_idx]
                bs = len(batch_idx)

                scores = Xb @ self.W_ + self.b_
                margins = np.maximum(0.0, 1.0 - Yb * scores)
                dscores = -2.0 * Yb * margins

                grad_W = (Xb.T @ dscores) / bs + l2_lambda * self.W_
                grad_b = np.mean(dscores, axis=0, keepdims=True)

                self.W_ -= lr_t * grad_W
                self.b_ -= lr_t * grad_b

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = X @ self.W_ + self.b_
        return _softmax(scores * 2.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = X @ self.W_ + self.b_
        return self.classes_[np.argmax(scores, axis=1)]


class MultinomialNBClassifier:
    """Multinomial Naive Bayes with Lidstone/Laplace smoothing (`alpha`)."""

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self.classes_: np.ndarray = np.array([])
        self.class_log_prior_: np.ndarray = np.array([])
        self.feature_log_prob_: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: Iterable) -> "MultinomialNBClassifier":
        X_pos = np.maximum(X, 0.0)
        y_arr = np.asarray(list(y))
        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)
        n_features = X_pos.shape[1]

        class_counts = np.zeros(n_classes, dtype=np.float64)
        feature_counts = np.zeros((n_classes, n_features), dtype=np.float64)

        for i, c in enumerate(self.classes_):
            mask = y_arr == c
            class_counts[i] = np.sum(mask)
            feature_counts[i] = np.sum(X_pos[mask], axis=0)

        self.class_log_prior_ = np.log(class_counts / np.sum(class_counts))
        smoothed_fc = feature_counts + self.alpha
        smoothed_denom = np.sum(smoothed_fc, axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(smoothed_fc / smoothed_denom)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_pos = np.maximum(X, 0.0)
        jll = X_pos @ self.feature_log_prob_.T + self.class_log_prior_
        return _softmax(jll)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


def stratified_train_test_split(
    df, strat_col: str, test_size: float = 0.20, seed: int = 42
):
    """Split a DataFrame into stratified train and test sets."""
    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []

    for _, group in df.groupby(strat_col):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_size)))
        test_indices.extend(idx[:n_test])
        train_indices.extend(idx[n_test:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return df.loc[train_indices].reset_index(drop=True), df.loc[test_indices].reset_index(drop=True)


def stratified_kfold_indices(
    y: np.ndarray, n_splits: int = 5, seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate stratified K-Fold `(train_idx, val_idx)` split indices."""
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y)
    folds: List[List[int]] = [[] for _ in range(n_splits)]

    for c in np.unique(y_arr):
        c_idx = np.where(y_arr == c)[0]
        rng.shuffle(c_idx)
        for i, idx_val in enumerate(c_idx):
            folds[i % n_splits].append(int(idx_val))

    splits = []
    all_indices = np.arange(len(y_arr))
    for k in range(n_splits):
        val_idx = np.array(sorted(folds[k]), dtype=np.int64)
        train_mask = np.ones(len(y_arr), dtype=bool)
        train_mask[val_idx] = False
        train_idx = all_indices[train_mask]
        splits.append((train_idx, val_idx))
    return splits


def compute_classification_metrics(
    y_true: Iterable, y_pred: Iterable, labels: List[str]
) -> Dict[str, Any]:
    """Compute Accuracy, Macro-F1, Weighted-F1, Per-Class P/R/F1, and Confusion Matrix."""
    yt = np.asarray(list(y_true))
    yp = np.asarray(list(y_pred))
    n = len(yt)
    acc = float(np.sum(yt == yp) / max(n, 1))

    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t_val, p_val in zip(yt, yp):
        if t_val in label_to_idx and p_val in label_to_idx:
            cm[label_to_idx[t_val], label_to_idx[p_val]] += 1

    per_class = {}
    f1_list = []
    supports = []

    for i, lbl in enumerate(labels):
        tp = float(cm[i, i])
        fp = float(np.sum(cm[:, i]) - tp)
        fn = float(np.sum(cm[i, :]) - tp)
        support = int(np.sum(cm[i, :]))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[lbl] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": support,
        }
        f1_list.append(f1)
        supports.append(support)

    macro_f1 = float(np.mean(f1_list))
    weighted_f1 = float(np.sum(np.array(f1_list) * np.array(supports)) / max(np.sum(supports), 1))

    return {
        "accuracy": round(acc, 4),
        "f1_macro": round(macro_f1, 4),
        "f1_weighted": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


class EndToEndTriageModel:
    """Bundles `HybridFeaturePipeline` and a classifier into a single serializable object."""

    def __init__(self, feature_pipeline: HybridFeaturePipeline, classifier: Any) -> None:
        self.feature_pipeline = feature_pipeline
        self.classifier = classifier

    @property
    def classes_(self) -> np.ndarray:
        return self.classifier.classes_

    def fit(self, texts: List[str], y: List[str]) -> "EndToEndTriageModel":
        X = self.feature_pipeline.fit_transform(texts)
        self.classifier.fit(X, y)
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        X = self.feature_pipeline.transform(texts)
        return self.classifier.predict(X)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X = self.feature_pipeline.transform(texts)
        return self.classifier.predict_proba(X)


