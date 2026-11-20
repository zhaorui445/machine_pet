import json
from pathlib import Path
import tempfile
import unittest

from petlm.dialogue_expand import approve_batch, generate, materialize, read_jsonl
from petlm.blind_eval import load_blind_prompts


class DialogueExpansionTests(unittest.TestCase):
    def test_generation_keeps_blind_set_separate_and_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'expansion'
            generate(output,48,20,123)
            candidates=read_jsonl(output/'candidates.pending.jsonl')
            prompts=read_jsonl(output/'blind'/'prompts.jsonl')
            keys=read_jsonl(output/'blind'/'answer-key.pending.jsonl')
            self.assertEqual((len(candidates),len(prompts),len(keys)),(48,20,20))
            self.assertTrue(all(row['review_status']=='pending' for row in candidates))
            self.assertTrue(all(row['prompt'].count('用户：')>=2 for row in candidates))
            self.assertTrue(all('answer' not in row and 'rubric' not in row for row in prompts))
            self.assertFalse({row['template_id'] for row in candidates}&
                             {row['template_id'] for row in keys})

    def test_only_explicit_decisions_are_materialized(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'source.jsonl'
            source.write_text(
                json.dumps({'id':'a','prompt':'用户：甲\n助手：乙\n用户：丙\n助手：',
                    'answer':'丁','review_status':'pending','reviewer':'','reviewed_at':''},
                    ensure_ascii=False)+'\n'+
                json.dumps({'id':'b','prompt':'用户：一\n助手：二\n用户：三\n助手：',
                    'answer':'四','review_status':'pending','reviewer':'','reviewed_at':''},
                    ensure_ascii=False)+'\n',encoding='utf-8')
            decisions=root/'decisions.jsonl'
            decisions.write_text(json.dumps({'id':'a','decision':'approved','reviewer':'审核员',
                'reviewed_at':'2026-11-03'},ensure_ascii=False)+'\n',encoding='utf-8')
            output=root/'reviewed.jsonl'
            materialize(source,decisions,output)
            rows=read_jsonl(output)
            self.assertEqual(rows[0]['review_status'],'approved')
            self.assertEqual(rows[1]['review_status'],'pending')

    def test_blind_runner_rejects_answer_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'prompts.jsonl'
            path.write_text(json.dumps({'id':'blind-1','history':[],'user':'问题',
                                        'answer':'泄漏答案'},ensure_ascii=False)+'\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'answer material'):
                load_blind_prompts(path)

    def test_batch_approval_is_not_labeled_item_level(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'source.jsonl'
            source.write_text(json.dumps({'id':'a'},ensure_ascii=False)+'\n',encoding='utf-8')
            decisions=root/'decisions.jsonl'
            approve_batch(source,decisions,'用户确认','整批数据合格')
            item=read_jsonl(decisions)[0]
            self.assertEqual(item['decision'],'approved')
            self.assertEqual(item['review_method'],'user_batch_confirmation')
            self.assertEqual(item['review_scope'],'corpus_level_not_item_by_item')


if __name__=='__main__':
    unittest.main()
