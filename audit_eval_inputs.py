#!/usr/bin/env python3
"""Read-only input-consistency audit for Howc61148/lidc-rebuild.

Run in the existing lidc-prep environment:
  python audit_eval_inputs.py --data D:/LIDC-IDRI/processed --split test

Only a new JSON report is written. No input, evaluator, or model is changed.
Uses pylidc DATABASE z positions, not a new reading of the source DICOM files.
This is NOT a full geometric, annotation, pixel-content, or model validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any


LIMITATIONS = [
    "Checks file/metadata consistency; does not run a model or recompute metrics.",
    "z positions come from pylidc's database, not a fresh DICOM-header audit.",
    "Does not verify PNG pixel identity, slice ordering by pixel content, HU conversion, or annotation correctness.",
    "Does not validate DICOM orientation, gantry tilt, or the full 3-D physical transform.",
    "Default 0.01 mm tolerance is a diagnostic choice, not a LUNA16 standard or an accuracy guarantee.",
]


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path):
    with path.open(encoding="utf-8-sig") as f:
        return json.load(f, object_pairs_hook=unique_object)


def positive_float(value, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} is not a number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return value


def positive_int(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def check_z(z_values, stored_spacing: float, tolerance: float) -> dict[str, Any]:
    """Compare z-index * stored_spacing with the DB coordinates, up to translation."""
    z = [float(v) for v in z_values]
    out: dict[str, Any] = {"n_positions": len(z), "issues": []}
    if len(z) < 2 or not all(math.isfinite(v) for v in z):
        out["issues"].append("Need at least two finite z positions")
        return out
    gaps = [b - a for a, b in zip(z, z[1:])]
    if any(g <= 0 for g in gaps):
        out["issues"].append("z positions must be strictly increasing")
        return out
    med = statistics.median(gaps)
    max_gap_delta = max(abs(g - med) for g in gaps)
    max_mapping_error = max(abs((v - z[0]) - i * stored_spacing)
                            for i, v in enumerate(z))
    out.update({
        "z_min_mm": z[0], "z_max_mm": z[-1],
        "min_gap_mm": min(gaps), "median_gap_mm": med,
        "max_gap_mm": max(gaps), "stored_spacing_mm": stored_spacing,
        "max_gap_deviation_from_median_mm": max_gap_delta,
        "max_z_mapping_error_mm": max_mapping_error,
        "tolerance_mm": tolerance,
    })
    if max_gap_delta > tolerance:
        out["issues"].append("Nonuniform z gaps exceed the diagnostic tolerance")
    if abs(med - stored_spacing) > tolerance:
        out["issues"].append("gt.json spacing differs from median DB spacing")
    if max_mapping_error > tolerance:
        out["issues"].append("z_index * stored_spacing differs from actual relative DB z positions")
    return out


def add_issue(report, code, message, scan=None):
    row = {"code": code, "message": message}
    if scan is not None:
        row["scan"] = scan
    report["issues"].append(row)


def audit(data: Path, split: str, tolerance: float, db_scans, decode_image):
    """db_scans and decode_image are injected so this can be tested on fixtures."""
    report: dict[str, Any] = {
        "scope": "PNG, split-manifest and pylidc-database metadata consistency only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data": str(data.resolve()), "split": split,
        "tolerance_mm": tolerance, "limitations": LIMITATIONS,
        "issues": [], "incomplete_checks": [], "scans": [],
        "summary": {},
    }
    eval_dir = data / f"eval_{split}"
    gt_path, split_path = eval_dir / "gt.json", data / "split.json"
    try:
        gt = read_json(gt_path)
        splits = read_json(split_path)
        if not isinstance(gt, dict) or not gt:
            raise ValueError("gt.json must be a nonempty scan-key dictionary")
        if not isinstance(splits, dict):
            raise ValueError("split.json must be a dictionary")
        split_sets = {}
        for name in ("train", "val", "test"):
            pids = splits.get(name)
            if not isinstance(pids, list) or not all(isinstance(p, str) and p for p in pids):
                raise ValueError(f"split.json[{name!r}] must be a list of patient IDs")
            if len(set(pids)) != len(pids):
                add_issue(report, "duplicate_split_patient", f"Duplicate ID in {name}")
            split_sets[name] = set(pids)
        if not split_sets[split]:
            raise ValueError(f"Selected split {split} is empty")
        for a, b in combinations(split_sets, 2):
            overlap = sorted(split_sets[a] & split_sets[b])
            if overlap:
                add_issue(report, "split_overlap", f"{a}/{b}: {overlap}")
    except Exception as exc:
        add_issue(report, "manifest_error", f"{type(exc).__name__}: {exc}")
        report["status"] = "FAIL"
        return report

    report["gt_sha256"] = hashlib.sha256(gt_path.read_bytes()).hexdigest()
    report["split_sha256"] = hashlib.sha256(split_path.read_bytes()).hexdigest()
    by_id = None
    if db_scans is None:
        report["incomplete_checks"].append("pylidc database checks unavailable")
    else:
        by_id = {int(s.id): s for s in db_scans}
        expected_keys = {f"{s.patient_id}_s{s.id}" for s in db_scans
                         if s.patient_id in split_sets[split]}
        if expected_keys != set(gt):
            add_issue(report, "db_scan_set_mismatch", json.dumps({
                "missing_from_gt": sorted(expected_keys - set(gt)),
                "unexpected_in_gt": sorted(set(gt) - expected_keys)}))
    if decode_image is None:
        report["incomplete_checks"].append("PNG decoding unavailable (requires OpenCV and NumPy)")

    img_dir = eval_dir / "images"
    if not img_dir.is_dir():
        add_issue(report, "images_missing", str(img_dir))
        report["status"] = "FAIL"
        return report
    try:
        actual_pngs = {p.name: p for p in img_dir.iterdir() if p.suffix.lower() == ".png"}
    except OSError as exc:
        add_issue(report, "images_unreadable", str(exc))
        report["status"] = "FAIL"
        return report
    expected_pngs = set()
    pids_found = set()
    n_declared = n_decoded = n_included = n_ignored = 0
    geometry_checked = 0

    for index, (key, s) in enumerate(sorted(gt.items()), 1):
        row: dict[str, Any] = {"scan_key": key}
        report["scans"].append(row)
        try:
            if not isinstance(s, dict):
                raise ValueError("scan record must be a dictionary")
            pid = s["patient_id"]
            if not isinstance(pid, str) or not pid:
                raise ValueError("patient_id must be a nonempty string")
            sid = positive_int(s["scan_id"], "scan_id")
            n = positive_int(s["n_slices"], "n_slices")
            h = positive_int(s["img_h"], "img_h")
            w = positive_int(s["img_w"], "img_w")
            ps = positive_float(s["pixel_spacing_mm"], "pixel_spacing_mm")
            ss = positive_float(s["slice_spacing_mm"], "slice_spacing_mm")
            if key != f"{pid}_s{sid}":
                raise ValueError("scan key does not match patient_id and scan_id")
            pids_found.add(pid)
            n_declared += n
            row.update({"patient_id": pid, "scan_id": sid, "expected_slices": n,
                        "height": h, "width": w})
            nds = s["nodules"]
            if not isinstance(nds, list):
                raise ValueError("nodules must be a list")
            ids = set()
            for nd in nds:
                nid = nd["id"]
                if isinstance(nid, bool) or not isinstance(nid, (str, int)):
                    raise ValueError("nodule id must be a string or integer")
                if nid in ids:
                    raise ValueError(f"Duplicate nodule id: {nid}")
                ids.add(nid)
                status = nd["status"]
                if status not in ("included", "ignore"):
                    raise ValueError(f"Unknown nodule status: {status}")
                center = [float(v) for v in nd["center_px"]]
                if len(center) != 3 or not all(math.isfinite(v) for v in center):
                    raise ValueError("center_px must contain 3 finite coordinates")
                x, y, z = center
                if not (0 <= x <= w-1 and 0 <= y <= h-1 and 0 <= z <= n-1):
                    raise ValueError(f"Nodule {nid} center outside volume bounds")
                positive_float(nd["diameter_mm"], "diameter_mm")
                n_included += status == "included"
                n_ignored += status == "ignore"
        except Exception as exc:
            add_issue(report, "gt_schema", f"{type(exc).__name__}: {exc}", key)
            continue

        expected = {f"{key}_{k:04d}.png" for k in range(n)}
        expected_pngs.update(expected)
        missing = sorted(expected - set(actual_pngs))
        row["missing_files"] = missing
        if missing:
            add_issue(report, "missing_pngs", f"{len(missing)} missing; examples: {missing[:5]}", key)
        bad = []
        if decode_image is not None:
            for name in sorted(expected & set(actual_pngs)):
                try:
                    im = decode_image(actual_pngs[name])
                    if im is None:
                        raise ValueError("decoder returned no image")
                    if im.ndim != 2 or im.shape != (h, w) or str(im.dtype) != "uint8":
                        raise ValueError(f"Expected grayscale uint8 {(h,w)}, got {im.shape} {im.dtype}")
                    n_decoded += 1
                except Exception as exc:
                    bad.append({"file": name, "error": f"{type(exc).__name__}: {exc}"})
            if bad:
                add_issue(report, "png_decode_or_shape", f"{len(bad)} bad images; see per-scan list", key)
        row["bad_images"] = bad

        if by_id is not None:
            scan = by_id.get(sid)
            if scan is None:
                add_issue(report, "db_scan_missing", f"scan_id {sid} absent from pylidc DB", key)
            else:
                try:
                    if scan.patient_id != pid:
                        raise ValueError("pylidc patient_id differs from gt.json")
                    positions = [float(v) for v in scan.slice_zvals]
                    geom = check_z(positions, ss, tolerance)
                    row["z_geometry"] = geom
                    geometry_checked += 1
                    if len(positions) != n:
                        add_issue(report, "slice_count_db_mismatch", f"gt: {n}, pylidc DB: {len(positions)}", key)
                    db_ps = positive_float(scan.pixel_spacing, "pylidc pixel_spacing")
                    if not math.isclose(ps, db_ps, rel_tol=1e-6, abs_tol=1e-6):
                        add_issue(report, "xy_spacing_mismatch", f"gt: {ps}, pylidc DB: {db_ps}", key)
                    for message in geom["issues"]:
                        add_issue(report, "z_geometry", message, key)
                except Exception as exc:
                    add_issue(report, "db_scan_read", f"{type(exc).__name__}: {exc}", key)
        if index % 10 == 0 or index == len(gt):
            print(f"Checked {index}/{len(gt)} scans", flush=True)

    extras = sorted(set(actual_pngs) - expected_pngs)
    report["unexpected_png_files"] = extras
    if extras:
        add_issue(report, "unexpected_pngs", f"{len(extras)} unexpected filenames; see report")
    if pids_found != split_sets[split]:
        add_issue(report, "split_patient_set_mismatch", json.dumps({
            "missing_from_gt": sorted(split_sets[split] - pids_found),
            "unexpected_in_gt": sorted(pids_found - split_sets[split])}))
    report["summary"] = {
        "n_patients": len(pids_found), "n_scans": len(gt),
        "n_expected_slices": n_declared, "n_png_files_on_disk": len(actual_pngs),
        "n_pngs_decoded_ok": n_decoded, "n_included_nodules": n_included,
        "n_ignore_nodules": n_ignored, "n_scans_z_checked": geometry_checked,
        "issue_count": len(report["issues"]),
        "incomplete_check_count": len(report["incomplete_checks"]),
    }
    report["status"] = ("FAIL" if report["issues"] else
                        "INCOMPLETE" if report["incomplete_checks"] else "PASS")
    return report


def write_new_report(report, requested: Path) -> Path:
    """Never overwrite a previous report or an input, even if --out points to one."""
    requested.parent.mkdir(parents=True, exist_ok=True)
    for idx in range(10000):
        p = requested if idx == 0 else requested.with_name(f"{requested.stem}_{idx}{requested.suffix}")
        try:
            with p.open("x", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
            return p
        except FileExistsError:
            continue
    raise RuntimeError("Unable to select a new report filename")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="D:/LIDC-IDRI/processed")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--z-tol-mm", type=float, default=0.01)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if not math.isfinite(args.z_tol_mm) or args.z_tol_mm <= 0:
        parser.error("--z-tol-mm must be finite and positive")
    dependency_errors = []
    try:
        import pylidc as pl
        db_scans = pl.query(pl.Scan).all()
    except Exception as exc:
        db_scans = None
        dependency_errors.append(f"pylidc: {type(exc).__name__}: {exc}")
    try:
        import cv2
        import numpy as np
        def decode_image(path):
            # fromfile + imdecode also supports Unicode paths on Windows.
            return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception as exc:
        decode_image = None
        dependency_errors.append(f"PNG decoder: {type(exc).__name__}: {exc}")
    print("Scope: input-consistency checks only; no CT/model evaluation.")
    report = audit(Path(args.data), args.split, args.z_tol_mm, db_scans, decode_image)
    report["dependency_errors"] = dependency_errors
    local_eval = Path.cwd() / "evaluate.py"
    if local_eval.is_file():
        report["local_evaluate_sha256"] = hashlib.sha256(local_eval.read_bytes()).hexdigest()
    out = Path(args.out) if args.out else Path(f"audit_eval_{args.split}.json")
    try:
        saved = write_new_report(report, out)
    except Exception as exc:
        print(f"Could not write report: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    print("\nStatus:", report["status"])
    print(json.dumps(report["summary"], indent=2))
    for problem in report["issues"][:10]:
        # ensure_ascii avoids console encoding errors with local path messages.
        print("ISSUE:", json.dumps(problem, ensure_ascii=True))
    for problem in report["incomplete_checks"] + dependency_errors:
        print("INCOMPLETE:", problem.encode("ascii", "backslashreplace").decode("ascii"))
    print("Report:", str(saved.resolve()).encode("ascii", "backslashreplace").decode("ascii"))
    if dependency_errors:
        print("Use the existing lidc-prep environment; do not reinstall old packages into lidc-yolo.")
    print("PASS means only the listed consistency checks passed; see limitations in the report.")
    return {"PASS": 0, "FAIL": 2, "INCOMPLETE": 3}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
