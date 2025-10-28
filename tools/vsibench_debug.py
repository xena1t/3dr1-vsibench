"""Helpers to inspect VSI-Bench evaluation logs.

The evaluation pipeline already stores rich per-sample metadata whenever
``--log_samples`` is enabled.  This script surfaces the key fields so that
mis-scored runs (for example, a wall of ``0.0`` results) can be investigated
without re-running the full benchmark.

Example usage::

    python tools/vsibench_debug.py \
        --log logs/20250101/3dr1_vsibench_test/.../vsibench.json

    # Show only the first five rows and aggregate prediction statistics
    python tools/vsibench_debug.py --log ... --limit 5 --summary

The output highlights the prompt identifier, the model prediction after
post-processing, the ground-truth label supplied by VSI-Bench, and whether the
answer matched under the task's scoring rule.  For numeric-answer tasks the
absolute error is also reported to make it obvious when every prediction is the
same constant (a common reason for ``0.0`` overall scores).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


NA_TYPES = {
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
}


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_samples(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise TypeError("Expected the log to contain a JSON list of samples")
    return data


def _iter_vsibench_docs(samples: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for sample in samples:
        doc = sample.get("vsibench_score")
        if isinstance(doc, dict):
            yield doc


def _describe_row(doc: Dict[str, Any]) -> str:
    question_type = doc.get("question_type", "<unknown>")
    prompt_id = doc.get("question_id") or doc.get("id") or doc.get("scene_name")
    prediction = doc.get("prediction")
    ground_truth = doc.get("ground_truth")
    metric_keys = [
        key for key in ("accuracy", "MRA:.5:.95:.05") if key in doc
    ]
    metric_str = ", ".join(f"{key}={doc[key]}" for key in metric_keys)

    fragments = [
        f"type={question_type}",
        f"prompt={prompt_id}",
        f"pred={prediction}",
        f"gt={ground_truth}",
    ]

    if question_type in NA_TYPES:
        pred_val, gt_val = _to_float(prediction), _to_float(ground_truth)
        if pred_val is not None and gt_val is not None:
            fragments.append(f"abs_err={abs(pred_val - gt_val):.3f}")

    if metric_str:
        fragments.append(metric_str)

    return " | ".join(fragments)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect VSI-Bench log outputs")
    parser.add_argument("--log", required=True, type=Path, help="Path to vsibench.json produced by --log_samples")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of rows to print (default: 10)")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a prediction frequency summary after the sample dump",
    )
    args = parser.parse_args()

    samples = _load_samples(args.log)
    docs = list(_iter_vsibench_docs(samples))

    if not docs:
        print("No VSI-Bench samples found in the provided log.")
        return

    print(f"Loaded {len(docs)} VSI-Bench samples from {args.log}")
    limit = args.limit if args.limit is None or args.limit > 0 else len(docs)

    for idx, doc in enumerate(docs[:limit]):
        print(f"[{idx}] {_describe_row(doc)}")

    if args.summary:
        counter = Counter(doc.get("prediction") for doc in docs)
        print("\nPrediction distribution (top 10):")
        for value, count in counter.most_common(10):
            print(f"  {value!r}: {count}")


if __name__ == "__main__":
    main()
