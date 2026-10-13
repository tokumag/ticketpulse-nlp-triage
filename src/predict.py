"""Local Inference & Explainability Engine for TicketPulse.

Loads trained ticket triage pipelines from disk and predicts:
  - Ticket Category + Calibrated Probability
  - Priority / Urgency Level + Calibrated Probability
  - Recommended SLA Target & Routing Queue
  - Top Explainable Signal Tokens driving the prediction (via weight attribution)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import pathlib
import pickle
from typing import Dict, List

import numpy as np


SLA_POLICY = {
    "High": {"target_response": "< 1 hour (P1 Escalation)", "auto_page": True},
    "Medium": {"target_response": "< 4 hours (Standard Queue)", "auto_page": False},
    "Low": {"target_response": "< 24 hours (Backlog Queue)", "auto_page": False},
}

ROUTING_QUEUES = {
    "Billing & Payments": "finance-billing-ops",
    "Technical Bug": "tier2-engineering-bug-triage",
    "Account & Access": "identity-security-support",
    "Shipping & Delivery": "logistics-fulfillment-desk",
    "Feature Request": "product-feedback-backlog",
}


@dataclass
class TriageResult:
    text: str
    category: str
    category_confidence: float
    priority: str
    priority_confidence: float
    routing_queue: str
    sla_target: str
    auto_page_oncall: bool
    top_signals: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)


class TicketTriageEngine:
    """Offline NLP inference engine for customer support ticket triage."""

    def __init__(self, models_dir: str | pathlib.Path = "models") -> None:
        models_path = pathlib.Path(models_dir)
        cat_path = models_path / "category_pipeline.pkl"
        pri_path = models_path / "priority_pipeline.pkl"

        if not cat_path.exists() or not pri_path.exists():
            raise FileNotFoundError(
                f"Trained model artifacts not found in '{models_path}'. "
                "Run `python -m src.train` first to train and tune the models."
            )

        with open(cat_path, "rb") as f:
            self.cat_pipeline = pickle.load(f)
        with open(pri_path, "rb") as f:
            self.pri_pipeline = pickle.load(f)

    def _extract_top_word_signals(self, text: str, predicted_cat: str, top_k: int = 4) -> List[str]:
        """Compute token-level linear attribution (`tfidf_val * weight_c`) for explainability."""
        try:
            word_vec = self.cat_pipeline.feature_pipeline.word_tfidf
            x_word = word_vec.transform([text])[0]
            vocab = word_vec.feature_names_
            word_dim = len(vocab)

            clf = self.cat_pipeline.classifier
            class_idx = list(clf.classes_).index(predicted_cat)
            weights = clf.W_[:word_dim, class_idx]

            contributions = x_word * weights
            nonzero = np.where(x_word > 0)[0]
            if len(nonzero) == 0:
                return []
            ranked = sorted(nonzero, key=lambda idx: contributions[idx], reverse=True)
            return [str(vocab[i]) for i in ranked[:top_k]]
        except (AttributeError, KeyError, IndexError, ValueError):
            return []

    def predict(self, text: str) -> TriageResult:
        """Predict category, priority, routing queue, SLA, and top features for a single ticket."""
        if not text or not str(text).strip():
            raise ValueError("Ticket text cannot be empty.")

        cleaned_text = str(text).strip()
        cat_probs = self.cat_pipeline.predict_proba([cleaned_text])[0]
        cat_idx = int(np.argmax(cat_probs))
        category = str(self.cat_pipeline.classes_[cat_idx])
        cat_conf = round(float(cat_probs[cat_idx]), 4)

        pri_probs = self.pri_pipeline.predict_proba([cleaned_text])[0]
        pri_idx = int(np.argmax(pri_probs))
        priority = str(self.pri_pipeline.classes_[pri_idx])
        pri_conf = round(float(pri_probs[pri_idx]), 4)

        sla_info = SLA_POLICY.get(priority, SLA_POLICY["Medium"])
        queue = ROUTING_QUEUES.get(category, "general-support")
        signals = self._extract_top_word_signals(cleaned_text, category)

        return TriageResult(
            text=cleaned_text,
            category=category,
            category_confidence=cat_conf,
            priority=priority,
            priority_confidence=pri_conf,
            routing_queue=queue,
            sla_target=sla_info["target_response"],
            auto_page_oncall=bool(sla_info["auto_page"]),
            top_signals=signals,
        )

    def predict_batch(self, texts: List[str]) -> List[TriageResult]:
        """Run batch inference across multiple tickets with vectorized transforms."""
        if not texts:
            return []

        cleaned_texts = [str(t).strip() for t in texts]
        for t in cleaned_texts:
            if not t:
                raise ValueError("Ticket text cannot be empty.")

        cat_probs = self.cat_pipeline.predict_proba(cleaned_texts)
        pri_probs = self.pri_pipeline.predict_proba(cleaned_texts)

        results: List[TriageResult] = []
        for i, text in enumerate(cleaned_texts):
            cat_idx = int(np.argmax(cat_probs[i]))
            pri_idx = int(np.argmax(pri_probs[i]))
            category = str(self.cat_pipeline.classes_[cat_idx])
            cat_conf = round(float(cat_probs[i, cat_idx]), 4)
            priority = str(self.pri_pipeline.classes_[pri_idx])
            pri_conf = round(float(pri_probs[i, pri_idx]), 4)

            sla_info = SLA_POLICY.get(priority, SLA_POLICY["Medium"])
            queue = ROUTING_QUEUES.get(category, "general-support")
            signals = self._extract_top_word_signals(text, category)

            results.append(
                TriageResult(
                    text=text,
                    category=category,
                    category_confidence=cat_conf,
                    priority=priority,
                    priority_confidence=pri_conf,
                    routing_queue=queue,
                    sla_target=sla_info["target_response"],
                    auto_page_oncall=bool(sla_info["auto_page"]),
                    top_signals=signals,
                )
            )
        return results
