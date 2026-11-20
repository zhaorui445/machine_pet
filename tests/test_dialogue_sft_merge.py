import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from petlm import dialogue_sft_merge


class FakeTokenizer:
    def encode(self,text):
        return list(range(len(text)))


class DialogueSftMergeTests(unittest.TestCase):
    def snapshot(self,root,name,rows):
        folder=root/name
        folder.mkdir()
        (folder/'manifest.json').write_text(json.dumps({'counts':{}}),encoding='utf-8')
        for split in ('train','validation','test'):
            with (folder/f'{split}.jsonl').open('w',encoding='utf-8') as handle:
                for row in rows:
                    if row['split']==split:
                        handle.write(json.dumps(row,ensure_ascii=False)+'\n')
        return folder

    def test_test_tree_and_prompt_win_over_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            tokenizer=root/'tokenizer.model'
            tokenizer.write_bytes(b'test')
            shared_tree='89136054-ecb8-4102-b45f-d41b8cbc7777'
            first=self.snapshot(root,'pet',[{'id':'a','tree_id':shared_tree,'split':'train',
                'prompt':'用户：同树问题\n助手：','answer':'训练回答'},
                {'id':'c','tree_id':'other','split':'train','prompt':'用户：重复问题\n助手：',
                 'answer':'另一训练回答'}])
            second=self.snapshot(root,'human',[{'id':'b','tree_id':shared_tree,'split':'test',
                'prompt':'用户：测试问题\n助手：','answer':'测试回答'},
                {'id':'d','tree_id':'test-other','split':'test','prompt':' 用户：重复问题 \n 助手：',
                 'answer':'重复测试回答'}])
            output=root/'merged'
            with patch.object(dialogue_sft_merge.spm,'SentencePieceProcessor',return_value=FakeTokenizer()):
                dialogue_sft_merge.build([first,second],output,tokenizer,context=100)
            train=(output/'train.jsonl').read_text(encoding='utf-8')
            test=(output/'test.jsonl').read_text(encoding='utf-8')
            manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(train,'')
        self.assertIn('测试回答',test)
        self.assertIn('重复测试回答',test)
        self.assertEqual(manifest['rejected']['tree_leakage_removed_from_train'],1)
        self.assertEqual(manifest['rejected']['prompt_duplicate_removed_from_train'],1)

    def test_reencodes_sources_with_target_tokenizer_and_records_source_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            tokenizer=root/'tokenizer.model'
            tokenizer.write_bytes(b'target-tokenizer')
            source=self.snapshot(root,'legacy',[{'id':'a','tree_id':'a','split':'train',
                'prompt':'用户：你好\n助手：','answer':'你好。'}])
            (source/'manifest.json').write_text(json.dumps({'counts':{'train':1},
                'tokenizer_sha256':'legacy-tokenizer-hash'}),encoding='utf-8')
            output=root/'merged'
            target_hash=dialogue_sft_merge.checksum(tokenizer)
            with patch.object(dialogue_sft_merge.spm,'SentencePieceProcessor',return_value=FakeTokenizer()):
                dialogue_sft_merge.build([source],output,tokenizer,context=100)
            manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(json.loads((output/'train.jsonl').read_text(encoding='utf-8'))['answer'],'你好。')
        self.assertEqual(manifest['tokenizer_sha256'],target_hash)
        self.assertEqual(manifest['sources'][0]['source_tokenizer_sha256'],'legacy-tokenizer-hash')

    def test_preserves_explicit_pet_human_approval_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            tokenizer=root/'tokenizer.model'
            tokenizer.write_bytes(b'target-tokenizer')
            source=self.snapshot(root,'pet',[{'id':'a','tree_id':'a','split':'train',
                'prompt':'用户：你好\n助手：','answer':'你好。','review_status':'approved'}])
            output=root/'merged'
            with patch.object(dialogue_sft_merge.spm,'SentencePieceProcessor',return_value=FakeTokenizer()):
                dialogue_sft_merge.build([source],output,tokenizer,context=100)
            manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['training_groups'],{'human_approved_pet':1})

    def test_automated_approval_is_not_labeled_human(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            tokenizer=root/'tokenizer.model'
            tokenizer.write_bytes(b'target-tokenizer')
            source=self.snapshot(root,'pet',[{'id':'a','tree_id':'a','split':'train',
                'prompt':'用户：你好\n助手：','answer':'你好。','review_status':'approved',
                'review_scope':'item_level_automated_not_human',
                'training_group':'natural_human_rule_assisted_item_audit'}])
            output=root/'merged'
            with patch.object(dialogue_sft_merge.spm,'SentencePieceProcessor',return_value=FakeTokenizer()):
                dialogue_sft_merge.build([source],output,tokenizer,context=100)
            manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['training_groups'],
                         {'natural_human_rule_assisted_item_audit':1})


if __name__=='__main__':
    unittest.main()
