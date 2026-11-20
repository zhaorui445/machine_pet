import io
import bz2
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from petlm import fetch


class FineWebTests(unittest.TestCase):
    def test_wikipedia_offset_skips_eligible_pages(self):
        pages=''.join(
            f'<page><title>标题{i}</title><ns>0</ns><id>{i}</id><revision><text>{"中文正文。"*30}</text></revision></page>'
            for i in range(4))
        archive=bz2.compress(f'<mediawiki>{pages}</mediawiki>'.encode())

        def handler(request):
            if request.url.path=='/zhwiki/':
                return httpx.Response(200,text='<a href="20260101/">20260101</a>')
            if request.url.path.endswith('/dumpstatus.json'):
                return httpx.Response(200,json={'jobs':{'articlesdump':{'status':'done','files':{
                    'zhwiki.xml.bz2':{'url':'/zhwiki/20260101/zhwiki.xml.bz2'}}}}})
            return httpx.Response(200,content=archive)

        output=io.StringIO()
        metadata={}
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            self.assertEqual(fetch.wiki(client,'wikipedia',2,output,metadata,start_offset=1),2)
        rows=[json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([row['id'] for row in rows],['1','2'])
        self.assertEqual([row['eligible_index'] for row in rows],[1,2])
        self.assertEqual(metadata['start_offset'],1)
        self.assertEqual(metadata['next_offset'],3)
        self.assertEqual(metadata['dump_date'],'20260101')

    def test_wikipedia_pins_exact_archive_and_retains_revision(self):
        archive_name='zhwiki-20261101-pages-articles2.xml-p187713p630160.bz2'
        page=('<page><title>新页面</title><ns>0</ns><id>123</id><revision>'
              '<id>456</id><timestamp>2026-10-31T00:00:00Z</timestamp>'
              f'<text>{"中文自然段落。"*30}</text></revision></page>')
        archive=bz2.compress(f'<mediawiki>{page}</mediawiki>'.encode())

        def handler(request):
            if request.url.path.endswith('/dumpstatus.json'):
                self.assertEqual(request.url.path,'/zhwiki/20261101/dumpstatus.json')
                return httpx.Response(200,json={'jobs':{'articlesdump':{'status':'done','files':{
                    archive_name:{'url':f'/zhwiki/20261101/{archive_name}'},
                    'zhwiki-20261101-pages-articles1.xml-p1p187712.bz2':{'url':'/wrong'}}}}})
            self.assertEqual(request.url.path,f'/zhwiki/20261101/{archive_name}')
            return httpx.Response(200,content=archive)

        output=io.StringIO()
        metadata={}
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            self.assertEqual(fetch.wiki(client,'wikipedia',1,output,metadata,
                dump_date='20261101',archive_name=archive_name,preserve_wikitext=True),1)
        row=json.loads(output.getvalue())
        self.assertEqual(row['revision_id'],'456')
        self.assertEqual(row['raw_wikitext'], '中文自然段落。'*30)
        self.assertEqual(row['timestamp'],'2026-10-31T00:00:00Z')
        self.assertEqual(metadata['archive_file'],archive_name)
        self.assertEqual(metadata['dump_date'],'20261101')

    def test_correct_config_pagination_and_truncation(self):
        offsets = []

        def handler(request):
            if '/api/datasets/' in request.url.path:
                return httpx.Response(200,json={'id':'HuggingFaceFW/fineweb-2','sha':'test-revision','cardData':{'license':'odc-by'}})
            if request.url.path == '/splits':
                return httpx.Response(200,json={'splits':[{'config':'cmn_Hani','split':'train'}]})
            self.assertEqual(request.url.params['config'],'cmn_Hani')
            offset = int(request.url.params['offset'])
            offsets.append(offset)
            indexes = [0,1] if offset == 0 else [2]
            rows = [{'row_idx':i,'truncated_cells':['text'] if i==0 else [],
                     'row':{'id':f'id-{i}','text':'中文测试文本','language':'cmn','language_script':'Hani','url':'https://example.org'}} for i in indexes]
            return httpx.Response(200,json={'rows':rows})

        output = io.StringIO()
        metadata = {}
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            self.assertEqual(fetch.fineweb(client,2,output,metadata),2)
        self.assertEqual(offsets,[0,2])
        self.assertEqual([json.loads(line)['id'] for line in output.getvalue().splitlines()],['id-1','id-2'])
        self.assertEqual(metadata['skipped'],1)
        self.assertEqual(metadata['license'],'odc-by')

    def test_offset_fetch_starts_after_existing_sample(self):
        def handler(request):
            if '/api/datasets/' in request.url.path:
                return httpx.Response(200,json={'id':'HuggingFaceFW/fineweb-2','sha':'rev','cardData':{'license':'odc-by'}})
            if request.url.path == '/splits':
                return httpx.Response(200,json={'splits':[{'config':'cmn_Hani','split':'train'}]})
            self.assertEqual(request.url.params['offset'],'1000')
            return httpx.Response(200,json={'rows':[{'row_idx':1000,'truncated_cells':[],
                'row':{'id':'new-id','text':'新的中文内容','language':'cmn','language_script':'Hani'}}]})
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            output=io.StringIO()
            meta={}
            self.assertEqual(fetch.fineweb(client,1,output,meta,start_offset=1000),1)
        self.assertEqual(json.loads(output.getvalue())['row_index'],1000)
        self.assertEqual(meta['next_offset'],1001)

    def test_network_retry_then_success(self):
        attempts = []

        def handler(request):
            attempts.append(1)
            if len(attempts)==1:
                raise httpx.ConnectError('test reset',request=request)
            return httpx.Response(200,json={'ok':True})

        with httpx.Client(transport=httpx.MockTransport(handler)) as client, patch.object(fetch.time,'sleep'):
            self.assertEqual(fetch.get_json(client,'https://example.org'),{'ok':True})
        self.assertEqual(len(attempts),2)

    def test_permanent_http_error_is_not_retried(self):
        with httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(403))) as client:
            with patch.object(fetch.time,'sleep') as pause:
                with self.assertRaises(httpx.HTTPStatusError):
                    fetch.get_json(client,'https://example.org')
                pause.assert_not_called()

    def test_single_source_keeps_other_statuses(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = {'wikipedia':{'status':'downloaded','documents':300},
                        'clue2020':{'status':'requires_application'}}
            (root/'download-status.json').write_text(json.dumps(original),encoding='utf-8')

            def sample(client,limit,output,metadata,start_offset=0):
                output.write(json.dumps({'text':'中文','id':'test'})+'\n')
                return 1

            with patch.object(fetch,'BASE',root), patch.object(fetch,'fineweb',side_effect=sample), patch('sys.argv',['fetch','--source','fineweb2','--limit','1']):
                fetch.main()
            status = json.loads((root/'download-status.json').read_text(encoding='utf-8'))
            self.assertEqual(status['wikipedia'],original['wikipedia'])
            self.assertEqual(status['clue2020'],original['clue2020'])
            self.assertEqual(status['fineweb2']['status'],'downloaded')


if __name__=='__main__':
    unittest.main()
