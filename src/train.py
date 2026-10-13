"""Model Training, Baseline Benchmarking & Hyperparameter Tuning.

Evaluates 3 classical ML architectures in vectorized NumPy:
  1. `SoftmaxLogisticClassifier`: Multi-class Logistic Regression trained with
     Mini-Batch Gradient Descent + Momentum, L2 regularization (`l2_reg`), and
     balanced class weighting.
  2. `LinearSVMClassifier`: Multi-class OvR Margin Classifier with
     temperature-scaled probability calibration.
  3. `MultinomialNBClassifier`: Multinomial Naive Bayes with Lidstone/Laplace
     smoothing (`alpha`) on non-negative TF-IDF features.

Performs:
  - Stratified train/test splitting (80% train / 20% hold-out test)
  - 5-Fold Stratified Cross-Validation across candidate model architectures
  - Exhaustive Grid Search Hyperparameter Tuning over feature and model hyperparameters
  - Hold-out test evaluation (Accuracy, Macro-F1, Weighted-F1, Per-Class Report)
  - Visual evaluation dashboard generation (`reports/model_evaluation.png`)
  - Model serialization (`models/category_pipeline.pkl`, `models/priority_pipeline.pkl`)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data_generator import generate_dataset
from src.preprocessing import (
    EndToEndTriageModel,
    HybridFeaturePipeline,
    LinearSVMClassifier,
    MultinomialNBClassifier,
    SoftmaxLogisticClassifier,
    compute_classification_metrics,
    stratified_kfold_indices,
    stratified_train_test_split,
)


def benchmark_baselines(
    texts: List[str], labels: List[str], class_names: List[str], seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """Run 5-Fold Stratified CV across candidate architectures."""
    y_arr = np.asarray(labels)
    splits = stratified_kfold_indices(y_arr, n_splits=5, seed=seed)

    candidate_specs = {
        "Softmax Logistic Regression": lambda: SoftmaxLogisticClassifier(
            lr=0.8, l2_reg=1e-4, epochs=100, seed=seed
        ),
        "Linear Support Vector Machine": lambda: LinearSVMClassifier(
            lr=0.35, C=1.2, epochs=90, seed=seed
        ),
        "Multinomial Naive Bayes": lambda: MultinomialNBClassifier(alpha=0.5),
    }

    results = {}
    for model_name, clf_factory in candidate_specs.items():
        t0 = time.time()
        fold_f1s = []
        for train_idx, val_idx in splits:
            train_texts = [texts[i] for i in train_idx]
            val_texts = [texts[i] for i in val_idx]
            y_tr = y_arr[train_idx]
            y_va = y_arr[val_idx]

            feat = HybridFeaturePipeline(word_ngram_range=(1, 2), word_max_features=2000, char_max_features=1000)
            X_tr = feat.fit_transform(train_texts)
            X_va = feat.transform(val_texts)

            clf = clf_factory()
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_va)
            metrics = compute_classification_metrics(y_va, preds, labels=class_names)
            fold_f1s.append(metrics["f1_macro"])

        elapsed = time.time() - t0
        results[model_name] = {
            "cv_f1_macro_mean": round(float(np.mean(fold_f1s)), 4),
            "cv_f1_macro_std": round(float(np.std(fold_f1s)), 4),
            "fit_time_sec": round(elapsed, 2),
        }
    return results


def grid_search_tune(
    texts: List[str],
    labels: List[str],
    class_names: List[str],
    task_name: str,
    seed: int = 42,
) -> Tuple[EndToEndTriageModel, Dict[str, Any]]:
    """Run exhaustive Grid Search over feature extraction and Softmax Logistic hyperparameters."""
    y_arr = np.asarray(labels)
    splits = stratified_kfold_indices(y_arr, n_splits=5, seed=seed)

    feature_configs = [
        {"word_ngram_range": (1, 1), "word_max_features": 2000, "metadata_weight": 0.20},
        {"word_ngram_range": (1, 2), "word_max_features": 3000, "metadata_weight": 0.30},
    ]
    clf_configs = [
        {"lr": 0.6, "l2_reg": 5e-4, "class_weight": None},
        {"lr": 0.9, "l2_reg": 1e-4, "class_weight": "balanced"},
        {"lr": 1.1, "l2_reg": 5e-5, "class_weight": "balanced"},
    ]

    best_score = -1.0
    best_params: Dict[str, Any] = {}
    all_trials = []

    print(f"[{task_name}] Running 5-Fold Grid Search across {len(feature_configs)*len(clf_configs)} hyperparameter combinations...")

    for f_cfg in feature_configs:
        fold_matrices = []
        for train_idx, val_idx in splits:
            train_texts = [texts[i] for i in train_idx]
            val_texts = [texts[i] for i in val_idx]
            fp = HybridFeaturePipeline(
                word_ngram_range=f_cfg["word_ngram_range"],
                word_max_features=f_cfg["word_max_features"],
                char_max_features=1200,
                metadata_weight=f_cfg["metadata_weight"],
            )
            X_tr = fp.fit_transform(train_texts)
            X_va = fp.transform(val_texts)
            fold_matrices.append((X_tr, y_arr[train_idx], X_va, y_arr[val_idx]))

        for c_cfg in clf_configs:
            fold_scores = []
            for X_tr, y_tr, X_va, y_va in fold_matrices:
                clf = SoftmaxLogisticClassifier(
                    lr=c_cfg["lr"],
                    l2_reg=c_cfg["l2_reg"],
                    epochs=120,
                    class_weight=c_cfg["class_weight"],
                    seed=seed,
                )
                clf.fit(X_tr, y_tr)
                preds = clf.predict(X_va)
                m = compute_classification_metrics(y_va, preds, labels=class_names)
                fold_scores.append(m["f1_macro"])

            mean_score = float(np.mean(fold_scores))
            combined_cfg = {**f_cfg, **c_cfg}
            all_trials.append({"params": {k: str(v) for k, v in combined_cfg.items()}, "cv_f1_macro": round(mean_score, 4)})

            if mean_score > best_score:
                best_score = mean_score
                best_params = combined_cfg

    best_fp = HybridFeaturePipeline(
        word_ngram_range=best_params["word_ngram_range"],
        word_max_features=best_params["word_max_features"],
        char_max_features=1200,
        metadata_weight=best_params["metadata_weight"],
    )
    best_clf = SoftmaxLogisticClassifier(
        lr=best_params["lr"],
        l2_reg=best_params["l2_reg"],
        epochs=140,
        class_weight=best_params["class_weight"],
        seed=seed,
    )
    final_model = EndToEndTriageModel(best_fp, best_clf).fit(texts, labels)

    tuning_summary = {
        "best_cv_f1_macro": round(best_score, 4),
        "best_params": {k: str(v) for k, v in best_params.items()},
        "trials": all_trials,
    }
    return final_model, tuning_summary


def plot_evaluation_dashboard(
    cat_eval: Dict[str, Any],
    pri_eval: Dict[str, Any],
    cat_bench: Dict[str, Dict[str, float]],
    pri_bench: Dict[str, Dict[str, float]],
    output_path: pathlib.Path,
) -> None:
    """Render a clean 2x2 evaluation visual summarizing confusion matrices and model benchmarks."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Note: matplotlib or seaborn not installed; skipping plot generation.")
        return

    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=180)
    fig.patch.set_facecolor("#F8FAFC")

    ax1 = axes[0, 0]
    cm_cat = np.array(cat_eval["confusion_matrix"])
    short_cat_labels = ["Billing", "Tech Bug", "Account", "Shipping", "Feature Req"]
    sns.heatmap(
        cm_cat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=short_cat_labels,
        yticklabels=short_cat_labels,
        cbar=False,
        ax=ax1,
        annot_kws={"size": 11, "weight": "bold"},
    )
    ax1.set_title(
        f"Category Confusion Matrix (Test Macro-F1: {cat_eval['f1_macro']*100:.1f}%)",
        fontsize=13,
        fontweight="bold",
        pad=12,
        color="#0F172A",
    )
    ax1.set_xlabel("Predicted Category", fontsize=10, fontweight="bold", color="#334155")
    ax1.set_ylabel("True Category", fontsize=10, fontweight="bold", color="#334155")

    ax2 = axes[0, 1]
    cm_pri = np.array(pri_eval["confusion_matrix"])
    pri_labels = pri_eval["labels"]
    sns.heatmap(
        cm_pri,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=pri_labels,
        yticklabels=pri_labels,
        cbar=False,
        ax=ax2,
        annot_kws={"size": 12, "weight": "bold"},
    )
    ax2.set_title(
        f"Priority Triage Confusion Matrix (Test Macro-F1: {pri_eval['f1_macro']*100:.1f}%)",
        fontsize=13,
        fontweight="bold",
        pad=12,
        color="#0F172A",
    )
    ax2.set_xlabel("Predicted Priority", fontsize=10, fontweight="bold", color="#334155")
    ax2.set_ylabel("True Priority", fontsize=10, fontweight="bold", color="#334155")

    ax3 = axes[1, 0]
    short_names = ["Softmax LogReg (CV)", "Linear SVM (CV)", "Multinomial NB (CV)", "Tuned Pipeline (Test)"]
    cat_scores = [cat_bench[m]["cv_f1_macro_mean"] * 100 for m in cat_bench] + [
        cat_eval["f1_macro"] * 100
    ]
    colors_cat = ["#94A3B8", "#94A3B8", "#94A3B8", "#2563EB"]
    bars1 = ax3.barh(short_names, cat_scores, color=colors_cat, height=0.52)
    ax3.set_xlim(75, 103)
    ax3.set_xlabel("Macro-F1 Score (%)", fontsize=10, fontweight="bold", color="#334155")
    ax3.set_title("Category Classifier: Baseline Models (CV) vs. Tuned Pipeline", fontsize=12.5, fontweight="bold", pad=12, color="#0F172A")
    ax3.grid(axis="x", linestyle="--", alpha=0.5)
    for bar in bars1:
        w = bar.get_width()
        ax3.text(w + 0.4, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=10, fontweight="bold", color="#0F172A")

    ax4 = axes[1, 1]
    pri_scores = [pri_bench[m]["cv_f1_macro_mean"] * 100 for m in pri_bench] + [
        pri_eval["f1_macro"] * 100
    ]
    colors_pri = ["#94A3B8", "#94A3B8", "#94A3B8", "#EA580C"]
    bars2 = ax4.barh(short_names, pri_scores, color=colors_pri, height=0.52)
    ax4.set_xlim(75, 103)
    ax4.set_xlabel("Macro-F1 Score (%)", fontsize=10, fontweight="bold", color="#334155")
    ax4.set_title("Priority Classifier: Baseline Models (CV) vs. Tuned Pipeline", fontsize=12.5, fontweight="bold", pad=12, color="#0F172A")
    ax4.grid(axis="x", linestyle="--", alpha=0.5)
    for bar in bars2:
        w = bar.get_width()
        ax4.text(w + 0.4, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=10, fontweight="bold", color="#0F172A")

    fig.suptitle(
        "TicketPulse NLP Triage — Model Benchmarks & Evaluation Dashboard",
        fontsize=15.5,
        fontweight="bold",
        color="#0F172A",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def run_training_pipeline(
    data_path: pathlib.Path,
    models_dir: pathlib.Path,
    reports_dir: pathlib.Path,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute full data loading, baseline benchmarking, hyperparameter tuning, and saving."""
    if not data_path.exists():
        print(f"Dataset not found at {data_path}; generating 1,500 synthetic tickets...")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df = generate_dataset(n_samples=1500, seed=seed)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    df["strat_key"] = df["category"] + "__" + df["priority"]
    train_df, test_df = stratified_train_test_split(df, strat_col="strat_key", test_size=0.20, seed=seed)

    X_train = train_df["text"].tolist()
    X_test = test_df["text"].tolist()
    y_cat_train = train_df["category"].tolist()
    y_cat_test = test_df["category"].tolist()
    y_pri_train = train_df["priority"].tolist()
    y_pri_test = test_df["priority"].tolist()

    cat_labels = [
        "Billing & Payments",
        "Technical Bug",
        "Account & Access",
        "Shipping & Delivery",
        "Feature Request",
    ]
    pri_labels = ["High", "Medium", "Low"]

    print("--- 1. Benchmarking Candidate Baseline Architectures (Category) ---")
    cat_benchmarks = benchmark_baselines(X_train, y_cat_train, cat_labels, seed=seed)
    for k, v in cat_benchmarks.items():
        print(f"  {k}: Macro-F1 = {v['cv_f1_macro_mean']:.4f} (+/- {v['cv_f1_macro_std']:.4f})")

    print("\n--- 2. Benchmarking Candidate Baseline Architectures (Priority) ---")
    pri_benchmarks = benchmark_baselines(X_train, y_pri_train, pri_labels, seed=seed)
    for k, v in pri_benchmarks.items():
        print(f"  {k}: Macro-F1 = {v['cv_f1_macro_mean']:.4f} (+/- {v['cv_f1_macro_std']:.4f})")

    print("\n--- 3. Hyperparameter Tuning via 5-Fold Grid Search ---")
    best_cat_model, cat_tuning = grid_search_tune(X_train, y_cat_train, cat_labels, "Category", seed=seed)
    best_pri_model, pri_tuning = grid_search_tune(X_train, y_pri_train, pri_labels, "Priority", seed=seed)

    cat_preds = best_cat_model.predict(X_test)
    pri_preds = best_pri_model.predict(X_test)

    cat_eval = compute_classification_metrics(y_cat_test, cat_preds, labels=cat_labels)
    pri_eval = compute_classification_metrics(y_pri_test, pri_preds, labels=pri_labels)

    print(f"\n[Test Set] Category Accuracy: {cat_eval['accuracy']*100:.2f}% | Macro-F1: {cat_eval['f1_macro']*100:.2f}%")
    print(f"[Test Set] Priority Accuracy: {pri_eval['accuracy']*100:.2f}% | Macro-F1: {pri_eval['f1_macro']*100:.2f}%")

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    cat_model_path = models_dir / "category_pipeline.pkl"
    pri_model_path = models_dir / "priority_pipeline.pkl"
    with open(cat_model_path, "wb") as f:
        pickle.dump(best_cat_model, f)
    with open(pri_model_path, "wb") as f:
        pickle.dump(best_pri_model, f)
    print(f"Saved tuned pipelines to {cat_model_path} and {pri_model_path}")

    full_metrics = {
        "dataset": {
            "total_samples": len(df),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
        },
        "category_task": {
            "baseline_cv": cat_benchmarks,
            "grid_search": cat_tuning,
            "test_evaluation": cat_eval,
        },
        "priority_task": {
            "baseline_cv": pri_benchmarks,
            "grid_search": pri_tuning,
            "test_evaluation": pri_eval,
        },
    }

    metrics_path = reports_dir / "evaluation_metrics.json"
    metrics_path.write_text(json.dumps(full_metrics, indent=2))

    plot_path = reports_dir / "model_evaluation.png"
    plot_evaluation_dashboard(cat_eval, pri_eval, cat_benchmarks, pri_benchmarks, plot_path)
    print(f"Saved evaluation dashboard -> {plot_path}")

    return full_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and tune TicketPulse NLP classification models.")
    parser.add_argument("--data", type=str, default="data/support_tickets.csv")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--reports-dir", type=str, default="reports")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_training_pipeline(
        data_path=pathlib.Path(args.data),
        models_dir=pathlib.Path(args.models_dir),
        reports_dir=pathlib.Path(args.reports_dir),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
