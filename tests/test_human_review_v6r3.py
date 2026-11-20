import json
from pathlib import Path
import tempfile
import unittest

from petlm.human_review_assignment import create_assignment
from petlm.human_review_pipeline import finalize
from petlm.human_review_record_approval import record_approvals
from petlm.human_review_v6 import FAMILIES
from petlm.human_review_v6r3 import build


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


class HumanReviewV6R3Tests(unittest.TestCase):
    def test_builds_disjoint_pending_development_and_internal_test(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'v6r3'
            manifest = build(output, count_per_family=4, seed=20261108)
            self.assertEqual(manifest['splits'], ['development', 'internal_test'])
            self.assertEqual(manifest['counts'], {'development': 16, 'internal_test': 16})
            self.assertFalse(manifest['eligible_for_training'])
            self.assertFalse(manifest['eligible_for_scoring'])
            self.assertTrue(manifest['leakage_report']['passed'])
            development = read_jsonl(output / 'development-candidates.jsonl')
            internal = read_jsonl(output / 'internal_test-candidates.jsonl')
            self.assertFalse({row['id'] for row in development} & {row['id'] for row in internal})
            self.assertTrue(all(row['review_status'] == 'pending' for row in development + internal))
            self.assertEqual(set(manifest['counts_by_family']),
                             {family for families in FAMILIES.values() for family in families})
            self.assertFalse((output / 'frozen-candidates.jsonl').exists())

    def test_assignment_accepts_manifest_declared_two_way_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / 'v6r3'
            build(candidates, count_per_family=2, seed=20261108)
            report = create_assignment(candidates, root / 'assigned', 'reviewer-001', 10)
            self.assertEqual(report['splits'], ['development', 'internal_test'])
            self.assertEqual(report['count'], 16)
            self.assertEqual(report['batch_count'], 2)

    def test_attested_two_way_review_finalizes_without_frozen_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / 'v6r3'
            assigned = root / 'assigned'
            recorded = root / 'recorded'
            final = root / 'final'
            build(candidates, count_per_family=2, seed=20261108)
            create_assignment(candidates, assigned, 'reviewer-001', 10)
            report = record_approvals(
                assigned, recorded, 'reviewer-001', 'ZR',
                '2026-11-08T15:34:28+08:00', True)
            finalize(candidates, Path(report['combined_decisions']), final, 'reviewer-001')
            manifest = json.loads((final / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['splits'], ['development', 'internal_test'])
            self.assertTrue((final / 'development.jsonl').exists())
            self.assertTrue((final / 'internal_test.jsonl').exists())
            self.assertFalse((final / 'frozen').exists())


if __name__ == '__main__':
    unittest.main()
