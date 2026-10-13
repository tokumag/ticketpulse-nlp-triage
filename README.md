# TicketPulse: NLP Customer Support Ticket Triage

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

TicketPulse is a lightweight, local NLP classification and routing engine designed to automate the triage of incoming customer support tickets. It simultaneously predicts **issue category**, assesses **urgency priority**, assigns **SLA response targets**, routes tickets to appropriate engineering/ops queues, and extracts **explainability signals** driving each classification.

---

## Architecture Overview

```
                          Incoming Support Ticket
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │    Domain Regex Normalization     │
                   │ (URLs, Emails, Order IDs, Amounts)│
                   └─────────────────┬─────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 │        Hybrid Feature Pipeline        │
                 ├───────────────────┬───────────────────┤
                 │  Word N-Gram      │  Char N-Gram      │  Structural & Urgency
                 │  TF-IDF (1-2)     │  TF-IDF (3-4)     │  Metadata Signals (9)
                 └─────────┬─────────┴─────────┬─────────┴───────────┬─────────┘
                           └───────────────────┼─────────────────────┘
                                               │ (Concatenated & L2-normalized)
                                               ▼
                         ┌───────────────────────────┐
                         │ Dual Softmax Classifiers  │
                         └─────────────┬─────────────┘
                                       │
                      ┌────────────────┴────────────────┐
                      ▼                                 ▼
           Category Classification            Priority & SLA Triage
           ───────────────────────            ─────────────────────
           - Billing & Payments               - High   (< 1h SLA, Auto-Page)
           - Technical Bug                    - Medium (< 4h SLA)
           - Account & Access                 - Low    (< 24h SLA)
           - Shipping & Delivery
           - Feature Request
                      │                                 │
                      └────────────────┬────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │  Queue Routing & Signal Attribution│
                     └───────────────────────────────────┘
```

---

## Key Features

- **Dual-Task Triage**: Simultaneously predicts ticket category (5 classes) and urgency priority (3 levels).
- **Interpretable Signal Extraction**: Linear weight attribution maps TF-IDF tokens to model confidence, explaining *why* a ticket was classified into a specific category.
- **SLA & Queue Integration**: Maps predictions directly to team routing queues (`identity-security-support`, `tier2-engineering-bug-triage`, `finance-billing-ops`) and auto-pages on-call engineers for critical issues.
- **Fast Local Inference**: Pure NumPy vectorization provides sub-millisecond inference per ticket without external API calls or GPU requirements.
- **Batch Processing**: Vectorized multi-ticket batch evaluation for processing large ticket logs via CSV.

---

## Benchmark Results

Evaluated on a 5-fold stratified cross-validation setup and hold-out test set:

| Model Architecture | Category Macro-F1 (CV) | Priority Macro-F1 (CV) | Fit Time (5 folds) |
|---|---|---|---|
| **Softmax Logistic Regression (Tuned)** | **98.7%** | **98.2%** | ~22.1s |
| Linear Support Vector Machine | 98.7% | 98.1% | ~17.3s |
| Multinomial Naive Bayes | 98.3% | 96.9% | ~4.8s |

The evaluation dashboard and confusion matrices can be inspected in `reports/model_evaluation.png`.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/tokumag/ticketpulse-nlp-triage.git
cd ticketpulse-nlp-triage

# Install dependencies
pip install -r requirements.txt

# Or install as an editable package
pip install -e .
```

---

## Command-Line Usage

### 1. Interactive Demo
Run inference on sample tickets across different categories:

```bash
python cli.py --demo
```

### 2. Classify a Single Ticket
Provide any ticket text directly in your shell:

```bash
python cli.py --text "URGENT: Locked out of account after SSO certificate renewal! Entire ops team blocked."
```

Output:
```
+--------------------------------------------------------------------------+
| Ticket:   URGENT: Locked out of account after SSO certificate renewal!   |
+--------------------------------------------------------------------------+
| Category: Account & Access         (Confidence:  82.1%)                  |
| Priority: High                     (Confidence: 100.0%)                  |
| Queue:    identity-security-support   Auto-Page: YES (Page On-Call)      |
| SLA:      < 1 hour (P1 Escalation)                                       |
| Signals:  locked, access, sso, renewal                                   |
+--------------------------------------------------------------------------+
```

### 3. Structured JSON Output
Use the `--json` flag for integration into webhooks or internal services:

```bash
python cli.py --text "Credit card billed twice on invoice INV-202611" --json
```

```json
{
  "text": "Credit card billed twice on invoice INV-202611",
  "category": "Billing & Payments",
  "category_confidence": 0.985,
  "priority": "Medium",
  "priority_confidence": 0.862,
  "routing_queue": "finance-billing-ops",
  "sla_target": "< 4 hours (Standard Queue)",
  "auto_page_oncall": false,
  "top_signals": ["card", "credit", "invoice", "billed"]
}
```

### 4. Batch CSV Classification
Process incoming tickets in bulk:

```bash
python cli.py --batch data/support_tickets.csv --output reports/triage_results.csv
```

---

## Python API Usage

```python
from src.predict import TicketTriageEngine

# Initialize inference engine
engine = TicketTriageEngine(models_dir="models")

# Classify single ticket
result = engine.predict("Payment gateway timed out with ERR_502 on checkout")
print(result.category)          # Technical Bug
print(result.priority)          # High
print(result.routing_queue)     # tier2-engineering-bug-triage
print(result.top_signals)       # ['__error_code__', 'timed', 'checkout']

# Vectorized batch prediction
batch_results = engine.predict_batch([
    "Need refund for order #ORD-12345",
    "How to change notification settings in profile",
])
```

---

## Training & Evaluation

To regenerate synthetic support data and run 5-fold cross-validation with grid search tuning:

```bash
python -m src.train --data data/support_tickets.csv --models-dir models --reports-dir reports
```

This will:
1. Benchmark candidate classifiers across 5 stratified folds.
2. Perform grid search tuning over TF-IDF n-gram ranges, vocabulary caps, and regularization strengths.
3. Evaluate the optimal model on the hold-out test set.
4. Export updated pipelines to `models/` and charts to `reports/model_evaluation.png`.

---

## Running Tests

Run the unit and integration test suite:

```bash
python -m unittest discover tests
```

Or using pytest:

```bash
pytest tests/ -v
```

---

## Project Structure

```
ticketpulse-nlp-triage/
├── data/
│   └── support_tickets.csv       # Synthetic support ticket dataset
├── models/
│   ├── category_pipeline.pkl     # Serialized category triage pipeline
│   └── priority_pipeline.pkl     # Serialized priority triage pipeline
├── reports/
│   ├── evaluation_metrics.json   # Model performance and grid search metrics
│   └── model_evaluation.png      # Confusion matrices and benchmark charts
├── src/
│   ├── __init__.py               # Package metadata
│   ├── data_generator.py         # Realistic noisy dataset generation
│   ├── predict.py                # Inference engine & token attribution
│   ├── preprocessing.py          # TF-IDF, metadata extractor & classifiers
│   └── train.py                  # Training pipeline & grid search tuner
├── tests/
│   └── test_pipeline.py          # Unit & integration test suite
├── cli.py                        # Command-line interface
├── pyproject.toml                # Build system & packaging configuration
├── requirements.txt              # Project dependencies
├── LICENSE                       # MIT License
└── README.md                     # Documentation
```

---

## License

This project is licensed under the [MIT License](LICENSE).
