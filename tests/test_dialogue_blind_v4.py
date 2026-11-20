import json
from pathlib import Path
import tempfile
import unittest

from petlm.dialogue_blind_v4 import FAMILIES, build, make_case, score


class UnseenDialogueV4Tests(unittest.TestCase):
    def test_public_prompt_has_no_answer_material(self):
        public,key=make_case('session_memory',2,20261104)
        self.assertNotIn('answer',public)
        self.assertNotIn('requirements',public)
        self.assertIn('answer',key)
        self.assertEqual(public['id'],key['id'])

    def test_every_family_has_scorable_requirements(self):
        for index,family in enumerate(FAMILIES):
            public,key=make_case(family,index,20261104)
            self.assertTrue(public['history'])
            self.assertTrue(public['user'])
            self.assertGreaterEqual(len(key['requirements']),3)

    def test_build_skips_exact_overlap(self):
        public,key=make_case(FAMILIES[0],0,20261104)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            compared=root/'compared'
            compared.mkdir()
            (compared/'train.jsonl').write_text(
                json.dumps({'prompt':key['prompt'],'answer':'x'},ensure_ascii=False)+'\n',encoding='utf-8')
            build(root/'blind',[compared],10,20261104)
            rows=[json.loads(line) for line in
                  (root/'blind'/'answer-key.generated.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(rows),10)
            self.assertNotIn(key['prompt'],[row['prompt'] for row in rows])

    def test_reference_answers_pass_generic_scorer(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            blind=root/'blind'
            build(blind,[],10,20261104)
            keys=[json.loads(line) for line in
                  (blind/'answer-key.generated.jsonl').read_text(encoding='utf-8').splitlines()]
            outputs=root/'outputs.jsonl'
            outputs.write_text(''.join(json.dumps(
                {'id':row['id'],'answer':row['answer'],'response_source':'reference'},
                ensure_ascii=False)+'\n' for row in keys),encoding='utf-8')
            report=root/'score.json'
            score(outputs,blind/'answer-key.generated.jsonl',report)
            result=json.loads(report.read_text(encoding='utf-8'))
            self.assertEqual(result['passed'],10)


if __name__=='__main__':
    unittest.main()
