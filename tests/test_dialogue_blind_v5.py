import json
from pathlib import Path
import tempfile
import unittest

from petlm.dialogue_blind_v5 import FAMILIES, build, make_case


class FrozenDialogueV5Tests(unittest.TestCase):
    def test_public_rows_exclude_answer_material(self):
        for index,family in enumerate(FAMILIES):
            public,key=make_case(family,index,20261105)
            self.assertNotIn('answer',public)
            self.assertNotIn('requirements',public)
            self.assertGreaterEqual(len(key['requirements']),3)

    def test_reference_contains_every_required_group(self):
        for index,family in enumerate(FAMILIES):
            _,key=make_case(family,index,20261105)
            compact=''.join(key['answer'].split())
            self.assertTrue(all(any(''.join(option.split()) in compact for option in group)
                                for group in key['requirements']),family)
            self.assertFalse(any(word in key['answer'] for word in key['forbidden']),family)

    def test_build_is_balanced_and_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'v5'
            build(output,[],20,20261105)
            prompts=[json.loads(line) for line in
                     (output/'prompts.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(prompts),20)
            self.assertEqual(len({json.dumps(row,ensure_ascii=False,sort_keys=True) for row in prompts}),20)


if __name__=='__main__':
    unittest.main()
