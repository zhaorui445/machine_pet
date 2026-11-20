"""Acquire no code: extract complete prose paragraphs from pinned WangDoc text."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import zipfile

import sentencepiece as spm

from .data import split_for
from .foundation_diagnostics import compact
from .pretrain_increment import canonical_hashes, PHONE, EMAIL, CN_ID, CARD, PRIVATE_DETAIL
from .prose_supplement import save
from .run import checksum
from .wikimedia_prose import fingerprints, deduplicate, normalize, END

REV = '84d50bb63511427d8e3c15e642ba456917e9649e'


def extract(markdown):
    section, fenced = '', False
    # Fences become paragraph boundaries; never join across examples or images.
    blocks, current = [], []
    for line in markdown.splitlines():
        if re.match(r'^\s*(```|~~~)', line):
            if current:
                blocks.append((section,'\n'.join(current)))
                current=[]
            fenced=not fenced
            continue
        if fenced:
            continue
        if line.startswith('#') or not line.strip():
            if current:
                blocks.append((section,'\n'.join(current)))
                current=[]
            if line.startswith('#'):
                section=line.lstrip('# ').strip()
        else:
            current.append(line)
    if current:
        blocks.append((section,'\n'.join(current)))
    for i,(section,block) in enumerate(blocks):
        if re.search(r'`|!\[|^\s*(?:>|[-+*] |\d+[.)] |---)|<|>|\|',block,re.M):
            continue
        text=re.sub(r'\[([^\]]+)\]\([^\)]+\)',r'\1',block)
        text=normalize(re.sub(r'\s+',' ',text.replace('**','')))
        if not 100<=len(text)<=3000 or not END.search(text) or len(re.findall('[。！？]',text))<2:
            continue
        norm=compact(text)
        if len(re.findall(r'[\u3400-\u9fff]',norm))/len(norm)<.72:
            continue
        if re.search(r'上面|下面|上述|上例|下例|如下|这段代码|这行代码|前面|后面|参见|本站|本书|本章|本节|这时|此时|^因此|^所以|^但是|^另外|^此外|^这|^该|^它|^其|^而|^由于|^有了|^为了解决|https?://|\{\{|\[|\]',text):
            continue
        if any(p.search(text) for p in (PHONE,EMAIL,CN_ID,CARD,PRIVATE_DETAIL)):
            continue
        yield i,section,text


def build(raw,base,tokens,output):
    if output.exists():
        raise FileExistsError(output)
    evidence={}
    for name in ['wangdoc.zip','wangdoc-tree.json','wangdoc-home.html','cc-by-sa-3.0.html']:
        meta=json.loads((raw/(name+'.receipt.json')).read_text(encoding='utf-8'))
        if checksum(raw/name)!=meta['sha256']:
            raise ValueError('Changed source evidence')
        evidence[name]=meta
    if 'creativecommons.org/licenses/by-sa/3.0' not in (raw/'wangdoc-home.html').read_text(encoding='utf-8'):
        raise ValueError('Explicit official text license missing')
    tree=json.loads((raw/'wangdoc-tree.json').read_text(encoding='utf-8'))
    if tree.get('truncated'):
        raise ValueError('Incomplete Git tree')
    blobs={r['path']:r for r in tree['tree'] if r['type']=='blob'}
    parent=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    index=json.loads((base/'clean-index.meta.json').read_text(encoding='utf-8'))
    if checksum(base/'manifest.json')!=index['snapshot_manifest_sha256'] or checksum(base/'clean-index.sqlite')!=index['index_sha256']:
        raise ValueError('Changed parent index')
    token_meta=json.loads((tokens/'manifest.json').read_text(encoding='utf-8'))
    if checksum(tokens/'tokenizer.model')!=token_meta['tokenizer_sha256']:
        raise ValueError('Changed tokenizer')
    sp=spm.SentencePieceProcessor(model_proto=(tokens/'tokenizer.model').read_bytes())
    db=sqlite3.connect((base/'clean-index.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    pending,rejected,documents=[],Counter(),0
    try:
        with zipfile.ZipFile(raw/'wangdoc.zip') as archive:
            for name in sorted(archive.namelist()):
                relative=name.split('/',1)[-1]
                if not relative.startswith('docs/') or not relative.endswith('.md'):
                    continue
                info=archive.getinfo(name)
                if info.file_size>2_000_000:
                    raise ValueError('Unexpectedly large document')
                content=archive.read(name)
                git_sha=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
                if git_sha!=blobs[relative]['sha']:
                    raise ValueError('Archive text does not match pinned Git tree')
                documents+=1
                # Whole main chapters stay together (e.g. all async subchapters).
                group=hashlib.sha256(('wangdoc/javascript:'+relative.split('/')[1]).encode()).hexdigest()
                split=split_for(group)
                for i,section,text in extract(content.decode('utf-8')):
                    full,prefix=canonical_hashes(text)
                    if db.execute('SELECT 1 FROM hashes WHERE full=? OR prefix=? LIMIT 1',(full,prefix)).fetchone():
                        rejected['parent_exact_prefix']+=1
                        continue
                    ids=sp.encode(text)
                    if sp.unk_id() in ids or compact(sp.decode(ids))!=compact(text):
                        rejected['tokenizer_roundtrip']+=1
                        continue
                    pending.append({'id':full,'source':'wangdoc_javascript','text':text,'split':split,
                        'split_group':group,'title':relative,'section':section,'paragraph_index':i,
                        'origin':f'https://github.com/wangdoc/javascript-tutorial/blob/{REV}/{relative}',
                        'history_url':f'https://github.com/wangdoc/javascript-tutorial/commits/{REV}/{relative}',
                        'attribution':'网道 WangDoc.com，阮一峰及教程贡献者',
                        'license':'CC-BY-SA-3.0','license_url':'https://creativecommons.org/licenses/by-sa/3.0/',
                        'git_blob_sha1':git_sha,'raw_document_sha256':hashlib.sha256(content).hexdigest(),
                        'tokens_with_eos':len(ids)+1,'human_review_status':'pending','training_eligible':False})
    finally:
        db.close()
    print(f'Checked {documents} source documents; {len(pending)} prose candidates',flush=True)
    wanted=set().union(*(fingerprints(r['text']) for r in pending)) if pending else set()
    overlap=set()
    for split in ('train','validation','test'):
        path=base/f'{split}.jsonl'
        if checksum(path)!=parent['files'][path.name]:
            raise ValueError('Changed parent split')
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                overlap.update(fingerprints(json.loads(line)['text'])&wanted)
        print(f'Parent {split} overlap checked',flush=True)
    kept,dups=deduplicate(pending,overlap)
    rejected.update(dups)
    output.mkdir(parents=True)
    shutil.copy2(Path(__file__),output/'screening-source.py')
    counts,token_counts=Counter(),Counter()
    for split in ('train','validation','test'):
        with (output/f'{split}-candidates.jsonl').open('x',encoding='utf-8',newline='\n') as out:
            for r in kept:
                if r['split']==split:
                    counts[split]+=1
                    token_counts[split]+=r['tokens_with_eos']
                    out.write(json.dumps(r,ensure_ascii=False)+'\n')
    review=[dict(r,human_decision=None,reviewer=None,review_notes='') for r in kept if r['split']!='test']
    save(output/'review.json',{'status':'pending_human_review','scope':'All non-test prose candidates','rows':review})
    lines=['# 作者中文教程：完整段落审核','',
        '来源：网道 JavaScript 教程；作者官网明确 CC BY-SA 3.0。仅作中文解释性文字补充，不是日常多轮对话。',
        '去除格式、转简体，未补写、未跨代码段拼接。所有行人工结论待填写。','']
    for i,r in enumerate(review,1):
        lines += [f'## A{i:03d} · {r["split"]} · {r["section"]}',f'ID：{r["id"]}',f'来源：{r["origin"]}',
            '',r['text'],'','人工结论：待审核','审核人：','理由：','']
    (output/'review.md').write_text('\n'.join(lines),encoding='utf-8')
    (output/'ATTRIBUTION.md').write_text('来源：网道 WangDoc.com，阮一峰及教程贡献者。\n许可：CC BY-SA 3.0，https://creativecommons.org/licenses/by-sa/3.0/\n每条记录保留固定版本正文与贡献者历史链接。变更：原段落格式清理、Unicode 及空白规范化、繁转简。未改写、补写或拼接正文。分享改编文本应保留署名、许可链接、变更说明和相同方式共享要求。\n',encoding='utf-8')
    report={'kind':'author_licensed_chinese_prose_v1','revision':REV,'source_documents':documents,
        'evidence':evidence,'counts':dict(counts),'tokens_with_eos':dict(token_counts),'rejected':dict(rejected),
        'tokenizer_sha256':token_meta['tokenizer_sha256'],'parent_manifest_sha256':checksum(base/'manifest.json'),
        'review_rows':len(review),'training_started':False,'training_ready':False,'human_review_completed':False,
        'physical_actuation':'disabled','split_policy':'whole top-level textbook chapters; no official upstream split; not a frozen blind test',
        'limitations':['Explanatory programming prose, not balanced everyday conversation.',
            'Group partitioning may yield few or no validation/test rows for a small source.',
            'Hash overlap exclusion is not semantic deduplication or proof of novelty against every historical corpus.',
            'Source quality must still be reviewed; official license does not approve factual or linguistic quality.'],
        'files':{p.name:checksum(p) for p in output.iterdir() if p.is_file()}}
    save(output/'manifest.json',report)
    print(json.dumps({k:report[k] for k in ('counts','tokens_with_eos','review_rows')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--raw',type=Path,default=Path('data/petlm/raw/author-chinese-source-r1'))
    p.add_argument('--base',type=Path,default=Path('data/petlm/prepared-chinese-v26r1-strict'))
    p.add_argument('--tokens',type=Path,default=Path('data/petlm/tokens-chinese-v26r1-strict-24k'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    build(a.raw,a.base,a.tokens,a.output)
