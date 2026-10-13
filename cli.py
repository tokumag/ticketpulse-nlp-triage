"""Command-Line Interface for TicketPulse NLP Triage.

Examples:
  # Run built-in demo on sample tickets
  python cli.py --demo

  # Predict a single support ticket from the terminal
  python cli.py --text "URGENT: Charged $499 twice on invoice INV-202510! Need immediate refund!"

  # Run batch predictions on a CSV file
  python cli.py --batch data/support_tickets.csv --output reports/sample_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pandas as pd

import time
from src.predict import TicketTriageEngine


DEMO_TICKETS = [
    "URGENT: Our entire team is locked out after SAML SSO rotation (AUTH_FAIL_04)! Production access blocked!!",
    "Hi support, my credit card was billed $199.00 twice on invoice INV-202611. Can you refund the duplicate charge?",
    "Dashboard export crashes with ERR_502 whenever we try to download weekly CSV analytics reports.",
    "Tracking for shipment #ORD-48291 hasn't updated in 5 days since leaving the fulfillment center.",
    "Suggestion: it would be cool to have custom color themes or compact row spacing in the sidebar.",
]


def format_result_card(res) -> str:
    page_badge = "YES (Page On-Call)" if res.auto_page_oncall else "No"
    signals_str = ", ".join(res.top_signals) if res.top_signals else "structural/char-ngram"
    return (
        f"+--------------------------------------------------------------------------+\n"
        f"| Ticket:   {res.text[:62]:<62} |\n"
        f"+--------------------------------------------------------------------------+\n"
        f"| Category: {res.category:<24} (Confidence: {res.category_confidence*100:5.1f}%)          |\n"
        f"| Priority: {res.priority:<24} (Confidence: {res.priority_confidence*100:5.1f}%)          |\n"
        f"| Queue:    {res.routing_queue:<28}Auto-Page: {page_badge:<15} |\n"
        f"| SLA:      {res.sla_target:<62} |\n"
        f"| Signals:  {signals_str[:62]:<62} |\n"
        f"+--------------------------------------------------------------------------+"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="TicketPulse — Support Ticket Triage CLI")
    parser.add_argument("--text", type=str, help="Single support ticket string to classify")
    parser.add_argument("--demo", action="store_true", help="Run triage demo on 5 sample tickets")
    parser.add_argument("--batch", type=str, help="Input CSV file with a 'text' column")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of rows to process in batch mode")
    parser.add_argument("--output", type=str, default="reports/batch_predictions.csv", help="Output CSV path for batch mode")
    parser.add_argument("--json", action="store_true", help="Output prediction as formatted JSON")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory containing trained .pkl pipelines")
    args = parser.parse_args()

    engine = TicketTriageEngine(models_dir=args.models_dir)

    if args.text:
        res = engine.predict(args.text)
        if args.json:
            print(json.dumps(res.to_dict(), indent=2))
        else:
            print(format_result_card(res))
        return

    if args.batch:
        df = pd.read_csv(args.batch)
        if "text" not in df.columns:
            raise ValueError("Batch input CSV must contain a 'text' column.")
        texts = df["text"].dropna().astype(str).tolist()
        if args.limit and args.limit > 0:
            texts = texts[: args.limit]
        t0 = time.time()
        results = engine.predict_batch(texts)
        preds = [r.to_dict() for r in results]
        elapsed = time.time() - t0
        out_df = pd.DataFrame(preds)
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"Processed {len(out_df)} tickets in {elapsed:.2f}s -> {out_path}")
        return

    print("=== TicketPulse Support Ticket Triage Demo ===\n")
    for sample in DEMO_TICKETS:
        res = engine.predict(sample)
        print(format_result_card(res))
        print()


if __name__ == "__main__":
    main()
