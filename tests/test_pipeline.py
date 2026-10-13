"""Unit & Integration Test Suite for TicketPulse NLP Triage."""

import pathlib
import unittest
import numpy as np
import pandas as pd

from src.data_generator import generate_dataset, CATEGORIES, PRIORITIES
from src.preprocessing import (
    normalize_ticket_text,
    TfidfVectorizer,
    TicketMetadataExtractor,
    HybridFeaturePipeline,
    SoftmaxLogisticClassifier,
    LinearSVMClassifier,
    MultinomialNBClassifier,
    stratified_train_test_split,
    stratified_kfold_indices,
    compute_classification_metrics,
)
from src.predict import TicketTriageEngine, SLA_POLICY, ROUTING_QUEUES


class TestTicketPulsePipeline(unittest.TestCase):
    def test_normalize_ticket_text_masks_entities(self):
        raw = "URGENT: Charged $49.00 on invoice INV-202501 for order #ORD-12345 with ERR_502! Visit https://example.com"
        cleaned = normalize_ticket_text(raw)
        self.assertIn("__currency_amount__", cleaned)
        self.assertIn("__ref_id__", cleaned)
        self.assertIn("__error_code__", cleaned)
        self.assertIn("__url__", cleaned)
        self.assertNotIn("$49.00", cleaned)
        self.assertNotIn("https://example.com", cleaned)

    def test_metadata_extractor_dimensions_and_urgency(self):
        extractor = TicketMetadataExtractor()
        samples = [
            "URGENT!! System crashed with ERR_502 and billed $99.00 on #ORD-12345!",
            "Just wondering how to change my profile avatar?",
        ]
        raw_mat = extractor._raw_features(samples)
        self.assertEqual(raw_mat.shape, (2, 9))
        self.assertGreaterEqual(raw_mat[0, 0], 2.0)  # exclamation marks
        self.assertEqual(raw_mat[0, 4], 1.0)  # has error
        self.assertEqual(raw_mat[0, 5], 1.0)  # has amount
        self.assertEqual(raw_mat[0, 6], 1.0)  # has reference
        self.assertGreater(raw_mat[0, 7], raw_mat[1, 7])  # urgency hits

        transformed = extractor.fit_transform(samples)
        self.assertEqual(transformed.shape, (2, 9))

    def test_tfidf_vectorizer_word_and_char(self):
        docs = [
            "Payment failed with billing error",
            "Refund needed for double charge",
            "Hardware shipment delayed in transit",
        ]
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50, sublinear_tf=True)
        X = vec.fit_transform(docs)
        self.assertEqual(X.shape[0], 3)
        self.assertGreater(X.shape[1], 5)
        # Check L2 normalization
        row_norms = np.linalg.norm(X, axis=1)
        np.testing.assert_allclose(row_norms, np.ones(3), atol=1e-5)

        char_vec = TfidfVectorizer(ngram_range=(3, 3), max_features=50, analyzer="char")
        X_char = char_vec.fit_transform(docs)
        self.assertEqual(X_char.shape[0], 3)

    def test_synthetic_dataset_balance(self):
        df = generate_dataset(n_samples=250, seed=123)
        self.assertEqual(len(df), 250)
        self.assertEqual(set(df["category"].unique()), set(CATEGORIES))
        self.assertEqual(set(df["priority"].unique()), set(PRIORITIES))
        self.assertIn("ticket_id", df.columns)
        self.assertIn("text", df.columns)

    def test_linear_svm_and_naive_bayes_classifiers(self):
        df = generate_dataset(n_samples=100, seed=42)
        fp = HybridFeaturePipeline(word_max_features=150, char_max_features=80)
        X = fp.fit_transform(df["text"].tolist())
        y = df["priority"].tolist()

        svm = LinearSVMClassifier(epochs=30, seed=42)
        svm.fit(X, y)
        svm_preds = svm.predict(X[:10])
        self.assertEqual(len(svm_preds), 10)
        svm_probs = svm.predict_proba(X[:10])
        self.assertEqual(svm_probs.shape, (10, len(PRIORITIES)))
        np.testing.assert_allclose(np.sum(svm_probs, axis=1), np.ones(10), atol=1e-5)

        nb = MultinomialNBClassifier(alpha=0.5)
        nb.fit(X, y)
        nb_preds = nb.predict(X[:10])
        self.assertEqual(len(nb_preds), 10)
        nb_probs = nb.predict_proba(X[:10])
        self.assertEqual(nb_probs.shape, (10, len(PRIORITIES)))
        np.testing.assert_allclose(np.sum(nb_probs, axis=1), np.ones(10), atol=1e-5)

    def test_softmax_logistic_classifier(self):
        df = generate_dataset(n_samples=120, seed=7)
        fp = HybridFeaturePipeline(word_max_features=200, char_max_features=100)
        X = fp.fit_transform(df["text"].tolist())
        clf = SoftmaxLogisticClassifier(lr=0.8, epochs=40, seed=7)
        clf.fit(X, df["category"].tolist())
        probs = clf.predict_proba(X[:5])
        self.assertEqual(probs.shape, (5, len(CATEGORIES)))
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(5), atol=1e-5)

    def test_stratified_split_and_kfold(self):
        df = pd.DataFrame({
            "text": [f"ticket {i}" for i in range(100)],
            "label": ["A"] * 50 + ["B"] * 30 + ["C"] * 20,
        })
        train_df, test_df = stratified_train_test_split(df, strat_col="label", test_size=0.20, seed=42)
        self.assertEqual(len(train_df) + len(test_df), 100)
        self.assertAlmostEqual(len(test_df), 20, delta=2)

        splits = stratified_kfold_indices(df["label"].to_numpy(), n_splits=5, seed=42)
        self.assertEqual(len(splits), 5)
        for tr_idx, va_idx in splits:
            self.assertEqual(len(set(tr_idx).intersection(set(va_idx))), 0)
            self.assertEqual(len(tr_idx) + len(va_idx), 100)

    def test_compute_classification_metrics(self):
        y_true = ["A", "A", "B", "B", "C"]
        y_pred = ["A", "B", "B", "B", "C"]
        labels = ["A", "B", "C"]
        metrics = compute_classification_metrics(y_true, y_pred, labels)
        self.assertEqual(metrics["accuracy"], 0.8)
        self.assertIn("f1_macro", metrics)
        self.assertIn("per_class", metrics)
        self.assertEqual(metrics["per_class"]["C"]["f1_score"], 1.0)

    def test_trained_engine_predictions(self):
        models_dir = pathlib.Path(__file__).resolve().parent.parent / "models"
        if not (models_dir / "category_pipeline.pkl").exists():
            self.skipTest("Trained models not yet generated.")

        engine = TicketTriageEngine(models_dir=models_dir)
        res = engine.predict(
            "URGENT: My corporate card was charged twice ($499.00) for invoice INV-202601! Need immediate refund."
        )
        self.assertEqual(res.category, "Billing & Payments")
        self.assertEqual(res.priority, "High")
        self.assertTrue(0.0 <= res.category_confidence <= 1.0)
        self.assertTrue(0.0 <= res.priority_confidence <= 1.0)
        self.assertTrue(res.auto_page_oncall)
        self.assertEqual(res.routing_queue, ROUTING_QUEUES["Billing & Payments"])
        self.assertEqual(res.sla_target, SLA_POLICY["High"]["target_response"])
        self.assertIsInstance(res.top_signals, list)

        # Batch prediction test
        batch = engine.predict_batch([
            "Refund required for duplicate payment INV-9999",
            "Cannot log in after SSO password change",
        ])
        self.assertEqual(len(batch), 2)
        self.assertEqual(batch[0].category, "Billing & Payments")

        with self.assertRaises(ValueError):
            engine.predict("   ")
        with self.assertRaises(ValueError):
            engine.predict_batch(["valid", ""])


if __name__ == "__main__":
    unittest.main()

