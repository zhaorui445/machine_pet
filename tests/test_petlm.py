import bz2
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import torch

from petlm import data
from petlm.model import PetLM
from petlm.run import continuation_state


class PipelineTests(unittest.TestCase):
    def test_future_tokens_cannot_change_past_logits(self):
        torch.manual_seed(7)
        model = PetLM({'vocab_size':64,'width':32,'heads':4,'layers':2,'context':16}).eval()
        first = torch.randint(0,64,(1,12))
        second = first.clone()
        second[:,6:] = (second[:,6:]+1)%64
        with torch.no_grad():
            torch.testing.assert_close(model(first)[:,:6],model(second)[:,:6],rtol=1e-5,atol=1e-6)

    def test_privacy_filter_and_duplicate_key(self):
        self.assertNotIn('13812345678',data.clean('联系13812345678或test@example.com'))
        self.assertNotIn('test@example.com',data.clean('test@example.com'))
        self.assertEqual(data.digest('学习 科学\n知识'),data.digest('学习科学知识'))

    def test_cross_source_duplicates_share_one_split(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root/'input.jsonl'
            path.write_text(json.dumps({'text':'我们一起学习科学，理解生活中的自然现象。'*20},ensure_ascii=False)+'\n',encoding='utf-8')
            with patch.object(data,'BASE',root):
                for source in ('wikipedia','fineweb2'):
                    data.import_data(source,path,'test fixture','test only',10)
                with closing(sqlite3.connect(root/'corpus.sqlite')) as conn:
                    self.assertEqual(conn.execute('SELECT count(*) FROM documents').fetchone()[0],1)
                    self.assertEqual(conn.execute('SELECT count(*) FROM provenance').fetchone()[0],2)
                with self.assertRaises(RuntimeError):
                    data.export_data(require_all=True)

    def test_wiki_namespace_redirect_and_markup(self):
        xml = '''<mediawiki xmlns="urn:test"><page><title>测试</title><ns>0</ns><id>1</id><revision><text>知识[[科学|学习]]{{模板}}</text></revision></page><page><title>讨论</title><ns>1</ns><id>2</id><revision><text>排除</text></revision></page><page><title>重定向</title><ns>0</ns><id>3</id><redirect title="测试"/><revision><text>排除</text></revision></page></mediawiki>'''
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'sample.xml.bz2'
            path.write_bytes(bz2.compress(xml.encode('utf-8')))
            rows = list(data.records(path))
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['text'],'知识学习')

    def test_model_context_limit(self):
        model = PetLM({'vocab_size':64,'width':32,'heads':4,'layers':1,'context':8})
        with self.assertRaises(ValueError):
            model(torch.zeros((1,9),dtype=torch.long))

    def test_continuation_checks_architecture_data_and_step(self):
        original = {'vocab_size':64,'width':32,'layers':2,'heads':4,'context':16,
                    'batch_size':2,'accumulation':4,'seed':42,'steps':1000,'learning_rate':0.0003}
        saved = {'config':original,'step':1000,'dataset_hash':'data-hash','tokenizer_sha256':'tokenizer-hash'}
        extended = dict(original,steps=1500,learning_rate=0.00003)
        self.assertEqual(continuation_state(saved,extended,'data-hash','tokenizer-hash'),1000)
        for changed in (dict(extended,steps=1000),dict(extended,width=64)):
            with self.assertRaises(RuntimeError):
                continuation_state(saved,changed,'data-hash','tokenizer-hash')
        with self.assertRaises(RuntimeError):
            continuation_state(saved,extended,'other-data','tokenizer-hash')


if __name__ == '__main__':
    unittest.main()
