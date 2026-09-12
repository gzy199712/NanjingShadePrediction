"""Validate and register the completed Step32 human explanation review.

This post-evaluation step never edits the frozen benchmark, model configuration,
Step31 integration, or human-entered CSV. It only creates an auditable summary
and updates the formal status to show that manual review is no longer pending.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
REVIEW_CSV = (
    ROOT
    / "routing/data/llm_evaluation/manual_review"
    / "step32_explanation_manual_review.csv"
)
RESULTS = ROOT / "routing/data/llm_evaluation/results"
REPORTS = ROOT / "routing/reports"
SUMMARY_JSON = RESULTS / "step32_manual_review_summary.json"
FORMAL_SUMMARY = RESULTS / "step32_formal_summary.json"
REPORT = REPORTS / "STEP32_MANUAL_REVIEW_REPORT.md"
ALLOWED_STATUSES = {"PASS", "FAIL", "UNCERTAIN"}


def main() -> int:
    with REVIEW_CSV.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    if len(rows) != 20:
        raise RuntimeError(f"Expected 20 review rows, found {len(rows)}")

    failures: list[str] = []
    for row in tqdm(
        rows,
        total=len(rows),
        desc="Validating Step32 manual review",
        unit="case",
        dynamic_ncols=True,
    ):
        case_id = row.get("case_id", "")
        status = row.get("reviewer_status", "").strip().upper()
        comment = row.get("reviewer_comment", "").strip()
        if status not in ALLOWED_STATUSES:
            failures.append(f"{case_id}: invalid reviewer_status={status!r}")
        if not comment:
            failures.append(f"{case_id}: reviewer_comment is blank")

    if failures:
        raise RuntimeError("; ".join(failures))

    counts = Counter(
        row["reviewer_status"].strip().upper()
        for row in rows
    )
    completed_at = datetime.now().astimezone().isoformat()
    summary = {
        "phase": "G11",
        "step": 32,
        "review_type": "human_explanation_review",
        "completed_at": completed_at,
        "reviewed_count": len(rows),
        "pass_count": counts["PASS"],
        "fail_count": counts["FAIL"],
        "uncertain_count": counts["UNCERTAIN"],
        "blank_status_count": 0,
        "blank_comment_count": 0,
        "manual_review_pending": False,
        "manual_review_pass": counts["FAIL"] == 0 and counts["UNCERTAIN"] == 0,
        "source_csv": str(REVIEW_CSV),
        "human_fields_preserved": True,
    }
    SUMMARY_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    formal = json.loads(FORMAL_SUMMARY.read_text(encoding="utf-8"))
    formal["MANUAL_REVIEW_PENDING"] = False
    formal["manual_review"] = summary
    # Existing automated failures remain authoritative. Manual completion does
    # not turn a failed reliability or faithfulness result into a pass.
    formal["READY_FOR_PHASE_H1_PREFLIGHT"] = bool(
        formal.get("STEP32_RELIABILITY_PASS")
        and formal.get("STEP32_FAITHFULNESS_PASS")
        and formal.get("STEP32_PRIVACY_PASS")
        and formal.get("STEP32_REGRESSION_PASS")
        and formal.get("frozen_input_hashes_unchanged")
    )
    FORMAL_SUMMARY.write_text(
        json.dumps(formal, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    REPORT.write_text(
        "\n".join(
            [
                "# Step32 Manual Explanation Review Report",
                "",
                f"- Completed at: {completed_at}",
                f"- Reviewed cases: {len(rows)}",
                f"- PASS: {counts['PASS']}",
                f"- FAIL: {counts['FAIL']}",
                f"- UNCERTAIN: {counts['UNCERTAIN']}",
                "- Blank reviewer statuses: 0",
                "- Blank reviewer comments: 0",
                "- Manual review pending: **False**",
                f"- Manual review pass: **{summary['manual_review_pass']}**",
                "",
                "The human-entered reviewer fields were read without modification.",
                "Completing manual review does not override the frozen automatic",
                "reliability or faithfulness thresholds.",
                "",
                f"- STEP32_RELIABILITY_PASS: **{formal['STEP32_RELIABILITY_PASS']}**",
                f"- STEP32_FAITHFULNESS_PASS: **{formal['STEP32_FAITHFULNESS_PASS']}**",
                f"- READY_FOR_PHASE_H1_PREFLIGHT: **{formal['READY_FOR_PHASE_H1_PREFLIGHT']}**",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"MANUAL_REVIEW_PENDING = {formal['MANUAL_REVIEW_PENDING']}")
    print(f"MANUAL_REVIEW_PASS = {summary['manual_review_pass']}")
    print(
        "READY_FOR_PHASE_H1_PREFLIGHT = "
        f"{formal['READY_FOR_PHASE_H1_PREFLIGHT']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
