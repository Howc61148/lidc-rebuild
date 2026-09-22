#!/usr/bin/env python3
"""Additional synthetic tests, not CT/model validation.

python test_evaluation_scoring.py --evaluate evaluate.py
Requires NumPy and check_evaluation_edges.py in the same directory.
"""
from __future__ import annotations
import argparse
import itertools
from pathlib import Path
import random
import sys
import unittest
from check_evaluation_edges import load_functions, scan, nodule

FUNCTIONS = {}


class ScoringTests(unittest.TestCase):
    def test_tied_scores_enter_together(self):
        prefix = [(0.95, False), (0.90, False)]
        for suffix in [[(0.8, True), (0.8, False)], [(0.8, False), (0.8, True)]]:
            curve = FUNCTIONS['froc'](prefix + suffix, 1, 1)
            self.assertEqual(curve, [(1.0, 0.0), (2.0, 0.0), (3.0, 1.0)])
            self.assertEqual(FUNCTIONS['sens_at'](curve, 2.0), 0.0)

    def test_all_tied_permutations(self):
        records = [(0.8, True), (0.8, True), (0.8, False), (0.8, False)]
        for order in itertools.permutations(records):
            self.assertEqual(FUNCTIONS['froc'](list(order), 2, 2), [(1.0, 1.0)])

    def test_matches_brute_force_thresholds(self):
        rng = random.Random(17)
        for _ in range(100):
            records = [(rng.choice([0.1, 0.2, 0.3, 0.6, 0.9]), bool(rng.randrange(2)))
                       for _ in range(20)]
            expected = []
            for threshold in sorted({s for s, _ in records}, reverse=True):
                selected = [is_tp for score, is_tp in records if score >= threshold]
                tp = sum(selected)
                expected.append(((len(selected) - tp) / 3, tp / 25))
            self.assertEqual(FUNCTIONS['froc'](records, 3, 25), expected)
            rng.shuffle(records)
            self.assertEqual(FUNCTIONS['froc'](records, 3, 25), expected)

    def test_distinct_scores_keep_previous_values(self):
        records = [(0.9, False), (0.8, True), (0.7, False), (0.6, True)]
        self.assertEqual(FUNCTIONS['froc'](records, 2, 2),
                         [(0.5, 0.0), (0.5, 0.5), (1.0, 0.5), (1.0, 1.0)])

    def test_empty_records(self):
        self.assertEqual(FUNCTIONS['froc']([], 1, 1), [])
        self.assertEqual(FUNCTIONS['sens_at']([], 2.0), 0.0)

    def test_zero_fp_true_positives(self):
        curve = FUNCTIONS['froc']([(0.9, True), (0.8, True)], 1, 2)
        self.assertEqual(FUNCTIONS['sens_at'](curve, 0.0), 1.0)

    def test_positive_overrides_ignore(self):
        gt = scan([nodule(1, 10.0, 'included'), nodule(2, 14.0, 'ignore')])
        preds = [{'center_px': [10.0, 10.0, 10.0], 'score': 0.9}]
        self.assertEqual(FUNCTIONS['match_scan'](preds, gt), ([(0.9, True)], 1, 0, 0))

    def test_duplicate_positive_overrides_ignore(self):
        gt = scan([nodule(1, 10.0, 'included'), nodule(2, 14.0, 'ignore')])
        preds = [{'center_px': [10.0, 10.0, 10.0], 'score': s} for s in [0.8, 0.9]]
        self.assertEqual(FUNCTIONS['match_scan'](preds, gt), ([(0.9, True)], 1, 0, 1))

    def test_ignore_counts_candidates_not_nodules(self):
        gt = scan([nodule(1, 10.0, 'ignore')])
        preds = [{'center_px': [10.0, 10.0, 10.0], 'score': s} for s in [0.8, 0.9]]
        self.assertEqual(FUNCTIONS['match_scan'](preds, gt), ([], 0, 2, 0))

    def test_unmatched_is_fp(self):
        gt = scan([nodule(1, 10.0, 'included')])
        preds = [{'center_px': [40.0, 10.0, 10.0], 'score': 0.9}]
        self.assertEqual(FUNCTIONS['match_scan'](preds, gt), ([(0.9, False)], 1, 0, 0))

    def test_strict_radius_boundary_preserved(self):
        gt = scan([nodule(1, 10.0, 'included')])
        preds = [{'center_px': [15.0, 10.0, 10.0], 'score': 0.9}]
        self.assertEqual(FUNCTIONS['match_scan'](preds, gt), ([(0.9, False)], 1, 0, 0))

    def test_two_nonoverlapping_included_nodules(self):
        gt = scan([nodule(1, 10.0, 'included'), nodule(2, 30.0, 'included')])
        preds = [{'center_px': [x, 10.0, 10.0], 'score': 0.9} for x in [10.0, 30.0]]
        self.assertEqual(FUNCTIONS['match_scan'](preds, gt),
                         ([(0.9, True), (0.9, True)], 2, 0, 0))

    def test_empty_predictions_preserve_gt_denominator(self):
        gt = scan([nodule(1, 10.0, 'included')])
        self.assertEqual(FUNCTIONS['match_scan']([], gt), ([], 1, 0, 0))

    def test_increasing_threshold_budget_cannot_reduce_sensitivity(self):
        curve = FUNCTIONS['froc']([(0.9, False), (0.8, True), (0.8, False),
                                  (0.7, True), (0.6, False)], 1, 2)
        vals = [FUNCTIONS['sens_at'](curve, x) for x in [0, 0.5, 1, 1.5, 2, 3]]
        self.assertEqual(vals, sorted(vals))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate', type=Path, default=Path('evaluate.py'))
    args = parser.parse_args()
    try:
        FUNCTIONS.update(load_functions(args.evaluate))
    except Exception as exc:
        print(f'Unable to load reviewed functions: {exc}', file=sys.stderr)
        return 2
    print('Scope: synthetic scoring-function tests only; not CT/model validation.', flush=True)
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ScoringTests))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
