import json
from pathlib import Path
import tempfile
import unittest

from petlm.human_review_assignment import create_assignment
from petlm.human_review_v6 import FAMILIES
from petlm.human_review_v6r5 import build


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


class HumanReviewV6R5Tests(unittest.TestCase):
    def test_cycle_is_pending_balanced_and_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'v6r5'
            manifest = build(output, count_per_family=4, seed=20261110)
            self.assertEqual(manifest['splits'], ['development', 'internal_test'])
            self.assertEqual(manifest['counts'], {'development': 16, 'internal_test': 16})
            self.assertFalse(manifest['eligible_for_training'])
            self.assertFalse(manifest['eligible_for_scoring'])
            self.assertTrue(manifest['leakage_report']['passed'])
            dev = read_jsonl(output / 'development-candidates.jsonl')
            test = read_jsonl(output / 'internal_test-candidates.jsonl')
            self.assertFalse({row['id'] for row in dev} & {row['id'] for row in test})
            self.assertTrue(all('开发记录' in row['user'] for row in dev))
            self.assertTrue(all('隔离复核' in row['user'] for row in test))
            self.assertTrue(all(row['review_status'] == 'pending' for row in dev + test))
            self.assertEqual(set(manifest['counts_by_family']),
                             {family for families in FAMILIES.values() for family in families})

    def test_assignment_supports_two_way_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / 'v6r5'
            build(candidates, count_per_family=2, seed=20261110)
            report = create_assignment(candidates, root / 'assigned', 'reviewer-001', 8)
            self.assertEqual(report['splits'], ['development', 'internal_test'])
            self.assertEqual(report['count'], 16)
            self.assertEqual(report['batch_count'], 2)


if __name__ == '__main__':
    unittest.main()
