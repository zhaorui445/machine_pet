import json
from pathlib import Path
import tempfile
import unittest

from petlm.human_review_assignment import create_assignment
from petlm.human_review_v6 import FAMILIES
from petlm.human_review_v6r4 import build


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


class HumanReviewV6R4Tests(unittest.TestCase):
    def test_new_cycle_is_two_way_pending_and_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'v6r4'
            manifest = build(output, count_per_family=4, seed=20261109)
            self.assertEqual(manifest['splits'], ['development', 'internal_test'])
            self.assertEqual(manifest['counts'], {'development': 16, 'internal_test': 16})
            self.assertFalse(manifest['eligible_for_training'])
            self.assertFalse(manifest['eligible_for_scoring'])
            self.assertTrue(manifest['leakage_report']['passed'])
            development = read_jsonl(output / 'development-candidates.jsonl')
            internal = read_jsonl(output / 'internal_test-candidates.jsonl')
            self.assertFalse({row['id'] for row in development} & {row['id'] for row in internal})
            self.assertTrue(all(row['review_status'] == 'pending' for row in development + internal))
            self.assertTrue(all('开发练习' in row['user'] for row in development))
            self.assertTrue(all('独立测试' in row['user'] for row in internal))
            self.assertEqual(set(manifest['counts_by_family']),
                             {family for families in FAMILIES.values() for family in families})

    def test_assignment_preserves_two_way_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / 'v6r4'
            build(candidates, count_per_family=2, seed=20261109)
            report = create_assignment(candidates, root / 'assigned', 'reviewer-001', 8)
            self.assertEqual(report['splits'], ['development', 'internal_test'])
            self.assertEqual(report['count'], 16)
            self.assertEqual(report['batch_count'], 2)


if __name__ == '__main__':
    unittest.main()
