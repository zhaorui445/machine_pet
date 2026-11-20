import json
from pathlib import Path
import tempfile
import unittest

from petlm.dialogue_merge import merge


class DialogueMergeTests(unittest.TestCase):
    def source(self,root,name,rows):
        folder=root/name
        folder.mkdir()
        (folder/'manifest.json').write_text(json.dumps({'source':f'OpenAssistant/{name}',
            'revision':f'{name}-revision','license':'Apache-2.0'}),encoding='utf-8')
        for split in ('train','validation','test'):
            with (folder/f'{split}.jsonl').open('w',encoding='utf-8') as handle:
                for row in rows:
                    if row['split']==split:
                        handle.write(json.dumps(row,ensure_ascii=False)+'\n')
        return folder

    def test_holdout_prompt_wins_over_training_duplicate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            first=self.source(root,'oasst1',[{'id':'a','tree_id':'one','split':'train',
                'prompt':'用户：你好\n助手：','answer':'训练回答'}])
            second=self.source(root,'oasst2',[{'id':'b','tree_id':'two','split':'test',
                'prompt':' 用户：你好 \n 助手：','answer':'测试回答'}])
            output=root/'merged'
            merge([first,second],output)
            train=(output/'train.jsonl').read_text(encoding='utf-8')
            test=(output/'test.jsonl').read_text(encoding='utf-8')
        self.assertEqual(train,'')
        self.assertIn('测试回答',test)


if __name__=='__main__':
    unittest.main()
