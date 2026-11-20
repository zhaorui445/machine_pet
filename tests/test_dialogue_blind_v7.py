import json
from collections import Counter
from pathlib import Path
import tempfile
import unittest

from petlm.dialogue_blind_v7 import FAMILIES, FAMILY_CAPABILITY, build, make_case


class FrozenDialogueV7Tests(unittest.TestCase):
    def test_public_rows_are_keyless_and_references_are_consistent(self):
        for index,family in enumerate(FAMILIES):
            public,key=make_case(family,index,20261108)
            self.assertNotIn('answer',public)
            self.assertNotIn('requirements',public)
            self.assertEqual(public['capability'],FAMILY_CAPABILITY[family])
            compact=''.join(key['answer'].split())
            self.assertTrue(all(any(''.join(option.split()) in compact for option in group)
                                for group in key['requirements']),family)
            self.assertFalse(any(word in key['answer'] for word in key['forbidden']),family)

    def test_build_is_balanced_and_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'v7'
            build(output,[],[],16,20261108)
            rows=[json.loads(line) for line in
                  (output/'prompts.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(rows),16)
            self.assertEqual(len({row['id'] for row in rows}),16)
            counts=Counter(row['capability'] for row in rows)
            self.assertEqual(set(counts.values()),{4})


if __name__=='__main__':
    unittest.main()
