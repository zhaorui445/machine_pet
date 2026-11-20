import json
from pathlib import Path
import tempfile
import unittest

from petlm.human_review_v6r6 import FAMILIES, _case, build


class HumanReviewV6R6Tests(unittest.TestCase):
    def test_context_switch_and_hazard_cycle_is_pending_and_split(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'candidates'
            manifest = build(output, rows_per_family=6)
            self.assertEqual(manifest['dataset_id'], 'v6r6')
            self.assertEqual(manifest['counts'], {'development': 36, 'internal_test': 36})
            self.assertFalse(manifest['eligible_for_training'])
            self.assertFalse(manifest['eligible_for_scoring'])
            self.assertTrue(manifest['leakage_report']['passed'])
            self.assertEqual(set(manifest['counts_by_family']), set(FAMILIES))
            dev = [json.loads(line) for line in
                   (output / 'development-candidates.jsonl').read_text(encoding='utf-8').splitlines()]
            test = [json.loads(line) for line in
                    (output / 'internal_test-candidates.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertTrue(all(row['review_status'] == 'pending' for row in dev + test))
            self.assertFalse({row['id'] for row in dev} & {row['id'] for row in test})
            switch_rows = [row for row in dev + test
                           if row['behavior_family'].startswith('switch_after_')]
            hazard_rows = [row for row in dev + test
                           if row['behavior_family'] in ('unknown_cleaner_observation',
                                                        'unknown_battery_contact',
                                                        'switch_back_to_new_hazard')]
            self.assertTrue(all(row['forbidden'] for row in switch_rows + hazard_rows))
            self.assertTrue(all('论文' in row['forbidden'] for row in switch_rows))
            self.assertTrue(all(row['behavior_family'] in FAMILIES for row in hazard_rows))
            decisions = [json.loads(line) for line in
                         (output / 'review-decisions.template.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(decisions), 72)
            self.assertTrue(all(row['decision'] == 'pending' and not row['reviewed_at']
                                for row in decisions))

    def test_external_history_near_duplicate_fails_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            external = root / 'history.jsonl'
            external.write_text(json.dumps({
                'id': 'old',
                'prompt': _case('unknown_cleaner_observation', 0, 'development')['prompt'],
            }, ensure_ascii=False) + '\n', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'leakage'):
                build(root / 'candidates', rows_per_family=6, external=[external])


if __name__ == '__main__':
    unittest.main()
