import json
from pathlib import Path
import tempfile
import unittest

from petlm.human_review_v6 import FAMILIES, build


class HumanReviewV6Tests(unittest.TestCase):
    def test_new_cycle_is_pending_isolated_and_three_way_split(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'v6'
            manifest = build(output, count_per_family=12, seed=20261220)
            self.assertEqual(manifest['dataset_id'], 'v6')
            self.assertFalse(manifest['eligible_for_training'])
            self.assertFalse(manifest['eligible_for_scoring'])
            self.assertEqual(manifest['counts'], {'development': 32, 'internal_test': 32, 'frozen': 32})
            self.assertEqual(set(manifest['counts_by_family']),
                             {family for families in FAMILIES.values() for family in families})
            public = [json.loads(line) for line in
                      (output / 'frozen-prompts.pending.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertTrue(all('proposed_answer' not in row and 'review_checks' not in row
                                and 'prompt' not in row for row in public))
            templates = [json.loads(line) for line in
                         (output / 'review-decisions.template.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertTrue(all(row['decision'] == 'pending' and not row['reviewed_at'] for row in templates))

    def test_physical_actuation_lock_remains_disabled(self):
        policy = json.loads((Path(__file__).parents[1] / 'configs' / 'physical-actuation-lock.json')
                            .read_text(encoding='utf-8'))
        self.assertEqual(policy['status'], 'disabled')
        self.assertTrue(all(policy[name] is False for name in (
            'real_motors_enabled', 'wheels_enabled', 'gpio_enabled',
            'serial_actuation_enabled', 'autonomous_navigation_enabled')))


if __name__ == '__main__':
    unittest.main()
