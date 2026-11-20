import json
from pathlib import Path
import tempfile
import unittest

from petlm.frozen_human_v5_eval import load_prompts, score_mode


class FrozenHumanV5EvaluationTests(unittest.TestCase):
    def test_public_prompt_rejects_answer_material(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'prompts.jsonl'
            path.write_text(json.dumps({
                'id': 'x', 'history': [], 'user': '问题', 'proposed_answer': '答案'
            }, ensure_ascii=False) + '\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_prompts(path)

    def test_review_checks_require_every_element(self):
        keys = [{'id': 'x', 'behavior_family': 'family', 'capability': 'format_control',
                 'review_checks': ['结论：', '1.', '2.']}]
        passed = score_mode([{'id': 'x', 'answer': '结论：先做一件事。\n1. 开始。\n2. 检查。',
                              'response_source': 'model_generation'}], keys)
        failed = score_mode([{'id': 'x', 'answer': '结论：先做一件事。\n1. 开始。',
                              'response_source': 'model_generation'}], keys)
        self.assertEqual(passed['passed'], 1)
        self.assertEqual(failed['passed'], 0)


if __name__ == '__main__':
    unittest.main()
