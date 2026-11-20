import json
from pathlib import Path
import tempfile
import unittest

from petlm.build_daily_dialogue_v6r6r2 import build
from petlm.daily_dialogue_v6r6r2_cases import DEVELOPMENT, INTERNAL_TEST
from petlm.daily_dialogue_diagnostics import audit, diagnostic_flags


class DailyDialogueV6R6R2Tests(unittest.TestCase):
    def test_revised_cycle_is_pending_and_has_independent_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = build(Path(directory) / 'candidates')
        self.assertEqual(manifest['dataset_id'], 'v6r6r2')
        self.assertEqual(manifest['counts'], {
            'development': len(DEVELOPMENT), 'internal_test': len(INTERNAL_TEST)})
        self.assertFalse(manifest['eligible_for_training'])
        self.assertFalse(manifest['eligible_for_scoring'])
        self.assertTrue(manifest['leakage_report']['passed'])
        self.assertTrue(set(manifest['counts_by_family']))

    def test_current_questions_get_concrete_references_and_no_old_templates(self):
        all_rows = DEVELOPMENT + INTERNAL_TEST
        self.assertEqual(len({row['user'] for row in all_rows}), len(all_rows))
        for row in all_rows:
            self.assertTrue(row['proposed_answer'].strip())
            self.assertTrue(row['review_checks'])
            self.assertTrue(row['diagnostic_signal_groups'])
            if row['behavior_family'] in {'meal_options', 'ingredient_limits', 'diet_constraints'}:
                self.assertTrue(any(token in row['proposed_answer'] for group in row['diagnostic_signal_groups'] for token in group))
            if row['behavior_family'] == 'unrelated_switch':
                self.assertNotIn('论文', row['proposed_answer'])
                self.assertNotIn('出处', row['proposed_answer'])

    def test_review_template_is_blank_and_internal_is_not_a_training_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'candidates'
            build(root)
            decisions = [json.loads(line) for line in
                         (root / 'review-decisions.template.jsonl').read_text(encoding='utf8').splitlines()]
            self.assertEqual(len(decisions), len(DEVELOPMENT) + len(INTERNAL_TEST))
            self.assertTrue(all(row['decision'] == 'pending' for row in decisions))
            self.assertTrue(all(not row['reviewer_id'] and not row['reviewed_at']
                                and not row['reviewer_signature'] for row in decisions))
            manifest = json.loads((root / 'manifest.json').read_text(encoding='utf8'))
            self.assertFalse(manifest['eligible_for_training'])
            self.assertNotIn('frozen', manifest['splits'])

    def test_references_meet_their_own_diagnostic_hints_without_keyword_scoring(self):
        for row in DEVELOPMENT + INTERNAL_TEST:
            result = diagnostic_flags(row, row['proposed_answer'])
            self.assertEqual(result['flags'], [], row['scenario_key'])
            self.assertIsNone(result['semantic_pass'])
            self.assertTrue(result['human_semantic_review_required'])

    def test_dinner_irrelevant_replies_flagged_and_safe_negation_is_allowed(self):
        dinner = DEVELOPMENT[0]
        for answer in ('我目前不能核实确切出处，请用论文检索。',
                       '我不能提供测试清洁液的方法，请远离。'):
            self.assertTrue(diagnostic_flags(dinner, answer)['flags'])
        hazard = next(row for row in DEVELOPMENT if row['scenario_key'] == 'cleaner_observation')
        answer = '不要碰它，不要混合或拆开；保持距离，请现场成年人处理。'
        self.assertEqual(diagnostic_flags(hazard, answer)['flags'], [])

    def test_candidate_audit_requires_human_review_even_when_structural_checks_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build(root / 'package')
            result = audit(root / 'package', root / 'audit')
            self.assertTrue(result['mechanical_checks_passed'])
            self.assertFalse(result['human_approval_granted'])
            self.assertFalse(result['semantic_independence_certified'])
            self.assertTrue((root / 'audit' / 'development-review.md').exists())


if __name__ == '__main__':
    unittest.main()
