"""Synthetic Customer Support Ticket Dataset Generator.

Generates a realistic, noisy multi-class dataset of customer support tickets
with two target labels:
  1. category: Billing & Payments | Technical Bug | Account & Access | Shipping & Delivery | Feature Request
  2. priority: High | Medium | Low

Includes realistic variations, order IDs, error codes, currency amounts, typos,
and cross-category lexical overlap so models must learn genuine patterns.
"""

from __future__ import annotations

import argparse
import pathlib
import random
import numpy as np
import pandas as pd


CATEGORIES = [
    "Billing & Payments",
    "Technical Bug",
    "Account & Access",
    "Shipping & Delivery",
    "Feature Request",
]

PRIORITIES = ["Low", "Medium", "High"]


def _build_templates() -> dict:
    return {
        "Billing & Payments": {
            "High": [
                "URGENT: My corporate card was charged twice ({amount}) for invoice {inv_id}! Need immediate refund.",
                "Unauthorized deduction of {amount} on my account after canceling subscription {order_id}. Fix this now!",
                "Payment gateway threw {err_code} but still deducted {amount} from our bank account. Production account suspended!",
                "Double billed {amount} on invoice {inv_id} and now our team workspace is locked for non-payment!!",
                "Critical billing error: charged {amount} for enterprise tier instead of starter plan {inv_id}. Refund immediately.",
            ],
            "Medium": [
                "Hi, I noticed an extra charge of {amount} on my monthly statement {inv_id}. Can you clarify what this is for?",
                "My promo code didn't apply during checkout for order {order_id}, so I was billed {amount}. Could you adjust it?",
                "Need a VAT-compliant PDF invoice for payment {inv_id} ({amount}) for our accounting department.",
                "Tried updating our company credit card on file but got error {err_code} when saving billing details.",
                "I downgraded my plan last week, yet invoice {inv_id} still shows the old {amount} renewal amount.",
            ],
            "Low": [
                "Where can I download past billing receipts and tax invoices like {inv_id} from my dashboard?",
                "Quick question: if we switch from monthly to annual billing, what is the prorated discount?",
                "Could you update the billing contact email address for future receipts on account {order_id}?",
                "Do you accept wire transfers or ACH payments for annual plans over {amount}?",
                "Just checking when my next billing cycle renews for subscription {order_id}. Thanks!",
            ],
        },
        "Technical Bug": {
            "High": [
                "CRITICAL: Dashboard crashes with {err_code} whenever exporting CSV reports! Entire analytics team blocked.",
                "Production webhook integration is failing with HTTP 500 ({err_code}) since this morning's release!",
                "Data sync pipeline froze at 0% and throws {err_code}—customer records are not updating in real time!!",
                "App completely unresponsive after v2.4 update; throws {err_code} on launch across all client browsers.",
                "Urgent bug: saving project changes wipes out existing configuration and logs error {err_code}!",
            ],
            "Medium": [
                "Search filter returns empty results when filtering by date range, showing warning {err_code} in console.",
                "Dark mode layout overlaps the save button on mobile screens when editing ticket {order_id}.",
                "PDF export takes over 2 minutes to generate and occasionally times out with {err_code}.",
                "Notification emails are being sent twice whenever a task status changes in workspace {order_id}.",
                "Chart labels are cut off when viewing weekly performance metrics on Firefox.",
            ],
            "Low": [
                "Minor visual glitch: the tooltip icon on the settings page stays visible after hovering away.",
                "Typo on the onboarding confirmation modal ('Succesfully saved' instead of 'Successfully').",
                "Sorting table columns by alphabetical order resets when navigating to page 2.",
                "The loading spinner animation stutters slightly when switching between tabs.",
                "Keyboard shortcut Ctrl+K doesn't focus the search bar on the archive page.",
            ],
        },
        "Account & Access": {
            "High": [
                "LOCKED OUT: Admin account disabled after SSO rotation ({err_code}). Nobody in our org can log in!!",
                "Urgent security alert: seeing unrecognized login attempts on admin profile {order_id}. Need immediate session reset!",
                "2FA SMS codes are not arriving and backup codes fail with {err_code}—completely locked out of production portal!",
                "All team members lost role permissions suddenly ({err_code}) and cannot access live client projects!",
                "Emergency: primary owner account {order_id} was accidentally deactivated during SAML migration!",
            ],
            "Medium": [
                "Password reset link expires immediately with {err_code} when I click it from my corporate email.",
                "Unable to invite a new developer to our workspace {order_id}; the invite stays stuck in 'Pending'.",
                "Need help transferring workspace ownership from a former employee's email to my account.",
                "Google OAuth login redirects back to the sign-in screen with code {err_code} on Safari.",
                "How do I revoke API token access for a contractor who finished their project last Friday?",
            ],
            "Low": [
                "How can I change my display name and profile avatar in the account settings?",
                "Can I link both my personal and work email addresses to the same user profile?",
                "Where do I find the audit log of past sign-in sessions for my account {order_id}?",
                "What are the password complexity requirements for standard viewer accounts?",
                "I'd like to update my timezone preference from UTC to EAT in my profile.",
            ],
        },
        "Shipping & Delivery": {
            "High": [
                "URGENT: Hardware shipment {order_id} marked delivered 3 days ago but never arrived! Paid {amount} for express.",
                "Damaged equipment received in package {order_id}—screen is shattered and we have a client event tomorrow!!",
                "Customs hold on urgent replacement order {order_id} due to missing commercial invoice {inv_id}! Please resolve ASAP.",
                "Sent the wrong hardware model in order {order_id}! Our field installation crew is onsite waiting right now.",
                "Lost package {order_id} worth {amount}: carrier says sender must file an urgent trace immediately!",
            ],
            "Medium": [
                "Tracking number for order {order_id} hasn't updated in 5 business days since leaving the warehouse.",
                "Need to update the delivery suite number for shipment {order_id} before it goes out for delivery tomorrow.",
                "Only 2 of the 3 items in order {order_id} arrived in today's box; is the remaining item shipping separately?",
                "Carrier left package {order_id} at the wrong building entrance; had to track it down with reception.",
                "Can you provide the courier tracking link for replacement order {order_id}?",
            ],
            "Low": [
                "What are your standard international shipping rates and estimated delivery windows?",
                "Do you ship orders like {order_id} to PO boxes or only physical street addresses?",
                "Can I opt for eco-friendly minimal packaging on future deliveries?",
                "How do I print a prepaid return shipping label if I decide to exchange an item?",
                "Just wondering which courier service you use for standard ground deliveries.",
            ],
        },
        "Feature Request": {
            "High": [
                "Blocking enterprise renewal ({amount}): we urgently need role-based audit exports to pass SOC2 compliance next month!",
                "Critical feature gap: without bulk webhook retry support, our ops team spends hours manually re-running failed jobs.",
                "High-priority request from our engineering leads: need SSO SCIM auto-provisioning before onboarding 200 seats.",
                "We need custom SLA escalation rules in workspace {order_id} to prevent missed enterprise client deadlines!",
                "Strict requirement for our Q4 rollout: API rate-limit headers must be exposed in responses.",
            ],
            "Medium": [
                "Would love to see a native Slack integration so high-priority tickets post directly to our #ops channel.",
                "Can you add support for exporting monthly analytics reports as Excel (.xlsx) workbooks in addition to CSV?",
                "It would be really helpful to have saved filter presets on the main ticket queue dashboard.",
                "Please consider adding multi-currency display support on billing summaries like {inv_id}.",
                "Could you add webhook triggers when a ticket's priority changes from Medium to High?",
            ],
            "Low": [
                "Suggestion: adding custom color themes or accent colors to the workspace sidebar would be nice.",
                "It would be cool to have markdown keyboard shortcuts when writing internal notes.",
                "Feature idea: allow users to pin their top 3 favorite dashboards to the navigation bar.",
                "Would appreciate an option to toggle compact row spacing on the ticket list view.",
                "Could you add emoji reactions to internal team comments on tickets?",
            ],
        },
    }


NOISE_PREFIXES = [
    "",
    "",
    "",
    "Hello support team, ",
    "Hi there - ",
    "Good morning, ",
    "Ticket ref {order_id}: ",
    "[Customer Portal] ",
    "Hey team, quick note: ",
]

NOISE_SUFFIXES = [
    "",
    "",
    "",
    " Thanks in advance.",
    " Please advise.",
    " Let me know as soon as possible.",
    " Appreciate your help!",
    " Sent from my mobile device.",
    " Best regards.",
    " Thanks!",
    " Thank you for looking into this.",
]

CROSS_DOMAIN_MODIFIERS = [
    "",
    "",
    "",
    " Also checked my account settings and everything looked normal.",
    " Note: this started right after our latest billing cycle renewed.",
    " I attached a screenshot of the dashboard screen for reference.",
    " Our team relies on this workflow every single day.",
]


def _inject_typo(text: str, rng: random.Random) -> str:
    """Inject realistic minor character typos in 15% of samples."""
    if rng.random() > 0.15 or len(text) < 25:
        return text
    words = text.split(" ")
    candidates = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    if not candidates:
        return text
    idx = rng.choice(candidates)
    w = words[idx]
    pos = rng.randint(1, len(w) - 2)
    w_typo = w[:pos] + w[pos + 1] + w[pos] + w[pos + 2 :]
    words[idx] = w_typo
    return " ".join(words)


def generate_dataset(n_samples: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Generate a balanced, realistic customer support ticket dataset."""
    rng = random.Random(seed)
    templates = _build_templates()

    records = []
    priority_weights = {"High": 0.25, "Medium": 0.45, "Low": 0.30}

    for i in range(n_samples):
        cat = CATEGORIES[i % len(CATEGORIES)]
        pri = rng.choices(
            population=["High", "Medium", "Low"],
            weights=[priority_weights["High"], priority_weights["Medium"], priority_weights["Low"]],
            k=1,
        )[0]

        base_template = rng.choice(templates[cat][pri])
        prefix = rng.choice(NOISE_PREFIXES)
        suffix = rng.choice(NOISE_SUFFIXES)
        modifier = rng.choice(CROSS_DOMAIN_MODIFIERS)

        order_id = f"#ORD-{rng.randint(10000, 99999)}"
        inv_id = f"INV-{rng.randint(202400, 202699)}"
        err_code = rng.choice(["ERR_502", "ERR_403", "E_TIMEOUT_99", "AUTH_FAIL_04", "SQL_DEADLOCK", "HTTP_500"])
        amount = f"${rng.choice([19, 29, 49, 99, 199, 499, 1200])}.00"

        full_text = f"{prefix}{base_template}{modifier}{suffix}".format(
            order_id=order_id,
            inv_id=inv_id,
            err_code=err_code,
            amount=amount,
        )
        full_text = _inject_typo(full_text, rng)

        # Add subtle label noise (2.5% realistic ambiguity)
        if rng.random() < 0.025:
            pri = rng.choice(["High", "Medium", "Low"])
        if rng.random() < 0.015:
            cat = rng.choice(CATEGORIES)

        records.append(
            {
                "ticket_id": f"TKT-{10001 + i}",
                "text": full_text.strip(),
                "category": cat,
                "priority": pri,
            }
        )

    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic support ticket dataset.")
    parser.add_argument("--output", type=str, default="data/support_tickets.csv", help="Output CSV file path")
    parser.add_argument("--samples", type=int, default=1500, help="Number of tickets to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(n_samples=args.samples, seed=args.seed)
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} support tickets -> {out_path}")


if __name__ == "__main__":
    main()
