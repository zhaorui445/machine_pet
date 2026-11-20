import json
from pathlib import Path
import tempfile
import unittest

from petlm.dialogue_blind_v6 import (FAMILIES, FAMILY_CAPABILITY, build,
                                     make_case, score)


class FrozenDialogueV6Tests(unittest.TestCase):
    def test_public_rows_exclude_answer_material(self):
        for index,family in enumerate(FAMILIES):
            public,key=make_case(family,index,20261107)
            self.assertNotIn('answer',public)
            self.assertNotIn('requirements',public)
            self.assertEqual(public['capability'],FAMILY_CAPABILITY[family])
            self.assertGreaterEqual(len(key['requirements']),2)

    def test_reference_satisfies_requirements(self):
        for index,family in enumerate(FAMILIES):
            _,key=make_case(family,index,20261107)
            compact=''.join(key['answer'].split())
            self.assertTrue(all(any(''.join(option.split()) in compact for option in group)
                                for group in key['requirements']),family)
            self.assertFalse(any(word in key['answer'] for word in key['forbidden']),family)

    def test_build_is_balanced_unique_and_excludes_prior_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            prior=root/'prior.jsonl'
            public,_=make_case(FAMILIES[0],0,20261107)
            prior.write_text(json.dumps(public,ensure_ascii=False)+'\n',encoding='utf-8')
            output=root/'v6'
            build(output,[],[prior],16,20261107)
            prompts=[json.loads(line) for line in
                     (output/'prompts.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(prompts),16)
            self.assertEqual(len({row['id'] for row in prompts}),16)
            counts={name:sum(row['capability']==name for row in prompts)
                    for name in set(FAMILY_CAPABILITY.values())}
            self.assertEqual(set(counts.values()),{4})

    def test_score_combines_capability_output_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            blind=root/'v6'
            build(blind,[],[],8,20261107)
            keys=[json.loads(line) for line in
                  (blind/'answer-key.generated.jsonl').read_text(encoding='utf-8').splitlines()]
            paths=[]
            for capability in sorted(set(FAMILY_CAPABILITY.values())):
                path=root/f'{capability}.jsonl'
                rows=[{'id':row['id'],'answer':row['answer'],'response_source':'model_generation'}
                      for row in keys if row['capability']==capability]
                path.write_text('\n'.join(json.dumps(row,ensure_ascii=False) for row in rows)+'\n',
                                encoding='utf-8')
                paths.append(path)
            report=root/'score.json'
            score(paths,blind/'answer-key.generated.jsonl',report)
            result=json.loads(report.read_text(encoding='utf-8'))
            self.assertEqual(result['passed'],8)


if __name__=='__main__':
    unittest.main()
