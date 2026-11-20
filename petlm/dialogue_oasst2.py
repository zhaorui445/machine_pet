"""Fetch and prepare pinned, human-authored Chinese OpenAssistant 2 dialogue."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import time

import httpx
import sentencepiece as spm

from .data import BASE
from .dialogue_data import MODEL_IDENTITY, acceptable, normalize
from .run import checksum


DATASET='OpenAssistant/oasst2'
REVISION='179dd21fc55192153d94adb0e0ce8f69e222bf75'
LICENSE='Apache-2.0'
FILE='2023-11-05_oasst2_ready.messages.jsonl.gz'
RAW=BASE/'dialogue'/'raw'/FILE
URLS=(
    f'https://hf-mirror.com/datasets/{DATASET}/resolve/{REVISION}/{FILE}',
    f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{FILE}',
)
HEADERS={'User-Agent':'PetLMResearch/0.1 (local educational corpus experiment)'}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row,ensure_ascii=False)+'\n')


def fetch(output=RAW):
    if output.exists():
        print(json.dumps({'status':'existing','path':str(output),'sha256':checksum(output)},ensure_ascii=False))
        return
    output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_suffix(output.suffix+'.partial')
    errors=[]
    for url in URLS:
        try:
            with httpx.Client(timeout=httpx.Timeout(60,read=60),follow_redirects=True,
                              headers=HEADERS) as client, client.stream('GET',url) as response:
                response.raise_for_status()
                with partial.open('wb') as handle:
                    for block in response.iter_bytes(1024*1024):
                        handle.write(block)
            with gzip.open(partial,'rt',encoding='utf-8') as handle:
                first=json.loads(next(handle))
                required=('message_id','parent_id','role','text','lang','message_tree_id')
                if not all(key in first for key in required):
                    raise RuntimeError('Unexpected OpenAssistant 2 schema.')
            partial.replace(output)
            metadata={'dataset':DATASET,'revision':REVISION,'license':LICENSE,'file':FILE,
                'url':url,'retrieved_utc':datetime.now(timezone.utc).isoformat(),
                'bytes':output.stat().st_size,'sha256':checksum(output)}
            output.with_suffix('.metadata.json').write_text(
                json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(metadata,ensure_ascii=False,indent=2))
            return
        except Exception as exc:
            errors.append(f'{url}: {type(exc).__name__}: {exc}')
            if partial.exists():
                partial.unlink()
            time.sleep(2)
    raise RuntimeError('OpenAssistant 2 download failed: '+' | '.join(errors))


def label_value(row, name):
    labels=row.get('labels') or []
    if isinstance(labels,dict):
        value=(labels.get(name) or {}).get('value')
    else:
        value=next((item.get('value') for item in labels
                    if isinstance(item,dict) and item.get('name')==name),None)
    try:
        return None if value is None else float(value)
    except (TypeError,ValueError):
        return None


def eligible(row):
    return (row.get('lang')=='zh' and row.get('role') in ('prompter','assistant')
            and row.get('deleted') is False and row.get('review_result') is True
            and row.get('synthetic') is False and not row.get('model_name')
            and isinstance(row.get('text'),str))


def safe(row):
    detox=row.get('detoxify') or {}
    limits={'toxicity':0.35,'severe_toxicity':0.1,'obscene':0.2,
            'identity_attack':0.2,'insult':0.25,'threat':0.15,'sexual_explicit':0.15}
    return all(detox.get(name) is None or float(detox[name])<=limit
               for name,limit in limits.items())


def chain_to(messages, parent):
    chain=[]
    current=parent
    while current is not None:
        chain.append(current)
        previous=current.get('parent_id')
        if previous and previous not in messages:
            return None
        current=messages.get(previous)
        if len(chain)>20:
            return None
    chain.reverse()
    if any(row['role'] != ('prompter' if index%2==0 else 'assistant')
           for index,row in enumerate(chain)):
        return None
    return chain


def render_prompt(chain):
    return ''.join(f"{'用户' if row['role']=='prompter' else '助手'}：{row['text']}\n"
                   for row in chain)+'助手：'


def split_for_tree(tree):
    bucket=int(hashlib.sha256(('oasst2-split\0'+tree).encode()).hexdigest()[:8],16)%100
    return 'test' if bucket<10 else 'validation' if bucket<20 else 'train'


def prepare(raw, output, tokenizer_path, context=512, top_rank_only=True):
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable output path.')
    tokenizer=spm.SentencePieceProcessor(model_proto=tokenizer_path.read_bytes())
    messages={}
    with gzip.open(raw,'rt',encoding='utf-8') as handle:
        for line in handle:
            row=json.loads(line)
            if eligible(row):
                row['text']=normalize(row['text'])
                messages[row['message_id']]=row
    rows=[]
    rejected=Counter()
    seen=set()
    for row in messages.values():
        if row['role']!='assistant':
            continue
        if top_rank_only and row.get('rank') not in (0,None):
            rejected['not_top_ranked']+=1
            continue
        if not safe(row):
            rejected['detoxify_risk']+=1
            continue
        quality=label_value(row,'quality')
        helpfulness=label_value(row,'helpfulness')
        if quality is not None and quality<0.5:
            rejected['low_quality']+=1
            continue
        if helpfulness is not None and helpfulness<0.5:
            rejected['low_helpfulness']+=1
            continue
        parent=messages.get(row.get('parent_id'))
        if not parent or parent.get('role')!='prompter':
            rejected['missing_parent']+=1
            continue
        chain=chain_to(messages,parent)
        if not chain:
            rejected['broken_history']+=1
            continue
        if any(item['role']=='assistant' and MODEL_IDENTITY.search(item['text']) for item in chain):
            rejected['model_identity_history']+=1
            continue
        if MODEL_IDENTITY.search(row['text']):
            rejected['model_identity']+=1
            continue
        if any(not safe(item) for item in chain):
            rejected['history_risk']+=1
            continue
        if not acceptable(chain[-1]['text'],4,300) or not acceptable(row['text'],8,600):
            rejected['content_filter']+=1
            continue
        chain=chain[-5:]
        while chain:
            prompt=render_prompt(chain)
            if len(tokenizer.encode(prompt))+len(tokenizer.encode(row['text']))+1<=context:
                break
            chain=chain[2:]
        if not chain:
            rejected['context_limit']+=1
            continue
        pair=(prompt,row['text'])
        if pair in seen:
            rejected['duplicate_pair']+=1
            continue
        seen.add(pair)
        tree=str(row.get('message_tree_id') or parent.get('message_tree_id') or parent['message_id'])
        rows.append({'id':hashlib.sha256(('oasst2\0'+row['message_id']).encode()).hexdigest(),
            'tree_id':tree,'source_message_id':row['message_id'],'split':split_for_tree(tree),
            'prompt':prompt,'answer':row['text'],'training_group':'natural_human_oasst2',
            'source_kind':'human_authored_openassistant','source_dataset':DATASET,
            'source_revision':REVISION,'source_license':LICENSE,
            'source_review_result':True,'source_rank':row.get('rank'),
            'source_quality':quality,'source_helpfulness':helpfulness})
    counts=Counter(row['split'] for row in rows)
    if min(counts.values(),default=0)<1 or not all(counts[split] for split in ('train','validation','test')):
        raise RuntimeError('OpenAssistant 2 filtering did not produce all data splits.')
    output.mkdir(parents=True)
    for split in ('train','validation','test'):
        write_jsonl(output/f'{split}.jsonl',(row for row in rows if row['split']==split))
    report={'source':DATASET,'revision':REVISION,'license':LICENSE,'raw_sha256':checksum(raw),
        'tokenizer_sha256':checksum(tokenizer_path),'counts':dict(counts),
        'trees':len({row['tree_id'] for row in rows}),'rejected':dict(rejected),
        'policy':{'human_authored':True,'source_review_result_required':True,
            'synthetic_excluded':True,'model_name_excluded':True,'top_rank_only':top_rank_only,
            'minimum_quality_if_present':0.5,'minimum_helpfulness_if_present':0.5,
            'maximum_context_tokens':context,'split_by_whole_tree':True},
        'limitations':['Source human feedback is not a new local human audit.',
            'Automatic safety and quality filters can retain errors or reject useful dialogue.']}
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    get=sub.add_parser('fetch')
    get.add_argument('--output',type=Path,default=RAW)
    prep=sub.add_parser('prepare')
    prep.add_argument('--raw',type=Path,default=RAW)
    prep.add_argument('--output',type=Path,required=True)
    prep.add_argument('--tokenizer',type=Path,default=BASE/'tokens-chinese-v7'/'tokenizer.model')
    prep.add_argument('--context',type=int,default=512)
    prep.add_argument('--all-reviewed-ranks',action='store_true',
        help='Keep every quality-filtered human answer branch, not only rank zero.')
    args=parser.parse_args()
    if args.command=='fetch':
        fetch(args.output)
    else:
        prepare(args.raw,args.output,args.tokenizer,args.context,not args.all_reviewed_ranks)


if __name__=='__main__':
    main()
