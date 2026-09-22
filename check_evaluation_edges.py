#!/usr/bin/env python3
"""Synthetic regression checks for lidc-rebuild's evaluation functions.

Run inside the project environment (NumPy required):
    python check_evaluation_edges.py --evaluate evaluate.py

This script reads selected functions by AST to avoid importing Ultralytics,
loading model weights, or reading patient data. It does NOT validate real-data
performance. Exit status: 0 = tested conditions satisfied; 1 = regression
condition failed; 2 = unable to run. Only run it on source files you trust.
"""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
from typing import Any
import numpy as np


def load_functions(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    names = {"to_mm", "match_scan", "froc", "sens_at"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    found = {n.name for n in nodes}
    if found != names:
        raise ValueError(f"Required functions not found: {sorted(names - found)}")
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    scope: dict[str, Any] = {"np": np}
    exec(compile(module, str(path), "exec"), scope)
    return scope


def scan(nodules: list[dict[str, Any]]) -> dict[str, Any]:
    return {"pixel_spacing_mm": 1.0, "slice_spacing_mm": 1.0, "nodules": nodules}


def nodule(i: int, x: float, status: str) -> dict[str, Any]:
    return {"id": i, "center_px": [x, 10.0, 10.0],
            "diameter_mm": 10.0, "status": status}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", type=Path, default=Path("evaluate.py"))
    args = parser.parse_args()
    try:
        f = load_functions(args.evaluate)
        prefix = [(0.95, False), (0.90, False)]
        tp_first = prefix + [(0.80, True), (0.80, False)]
        fp_first = prefix + [(0.80, False), (0.80, True)]
        a = float(f["sens_at"](f["froc"](tp_first, 1, 1), 2.0))
        b = float(f["sens_at"](f["froc"](fp_first, 1, 1), 2.0))
        # With a single deterministic score threshold, the .80 detections
        # must enter together. The only feasible sensitivity at <=2 FP is 0.
        tie_ok = a == b == 0.0

        gt = scan([nodule(1, 10.0, "included"), nodule(2, 14.0, "ignore")])
        preds = [{"center_px": [10.0, 10.0, 10.0], "score": 0.9}]
        records, n_gt, n_ign, n_dup = f["match_scan"](preds, gt)
        # Under positive-reference-first scoring, a prediction at the included
        # nodule center must not be discarded by an overlapping ignore sphere.
        actual_tp = sum(int(t) for _, t in records)
        positive_ok = actual_tp == 1 and n_ign == 0

        ignore_gt = scan([nodule(2, 10.0, "ignore")])
        two_candidates = [
            {"center_px": [10.0, 10.0, 10.0], "score": 0.9},
            {"center_px": [11.0, 10.0, 10.0], "score": 0.8},
        ]
        _, _, ignored_candidates, _ = f["match_scan"](two_candidates, ignore_gt)
        report = {
            "scope": "synthetic function tests only; not real CT/model evaluation",
            "source_file": str(args.evaluate),
            "froc_equal_scores": {
                "tp_first_recall_at_2fp": a,
                "fp_first_recall_at_2fp": b,
                "expected_stepwise_recall_at_2fp": 0.0,
                "pass": tie_ok,
            },
            "included_ignore_overlap": {
                "actual_tp": actual_tp, "actual_ignored_candidates": n_ign,
                "expected_tp_positive_reference_first": 1,
                "pass": positive_ok,
            },
            "ignore_counter_semantics": {
                "input_unique_ignore_nodules": 1,
                "input_candidates_in_same_ignore_nodule": 2,
                "reported_ignored_candidates": ignored_candidates,
                "note": "Candidate count is NOT unique ignored-nodule recall."
            },
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if tie_ok and positive_ok else 1
    except (OSError, ValueError, SyntaxError, KeyError, TypeError, NameError) as exc:
        print(f"Unable to run checks: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
