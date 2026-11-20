"""Prepare human-authored Chinese OpenAssistant dialogue pairs for local SFT."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re

import httpx
from opencc import OpenCC
import sentencepiece as spm

from .data import BASE, clean
from .run import checksum

DATASET = 'OpenAssistant/oasst1'
REVISION = 'fdf72ae0827c1cda404aff25b6603abec9e3399b'
FILE = '2023-04-12_oasst_ready.messages.jsonl.gz'
URL = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{FILE}'
RAW = BASE/'dialogue'/'raw'/FILE
CONVERTER = OpenCC('t2s')
CHINESE = re.compile(r'[\u3400-\u9fff]')
MODEL_IDENTITY = re.compile(
    r'Open\s*Assistant|OpenAI|ChatGPT|\bGPT\b|语言模型|人工智能模型'
    r'|(?:我是|我只是|作为)[^。！？\n]{0,40}(?:AI|人工智能|语言大模型|聊天机器人)',
    re.I,
)


def fetch(output=RAW):
    if output.exists():
        print(f'Existing file preserved: {output}; sha256={checksum(output)}')
        return
    output.parent.mkdir(parents=True,exist_ok=True)
    partial = output.with_suffix(output.suffix+'.partial')
    with httpx.Client(timeout=httpx.Timeout(60,read=60),follow_redirects=True,
                      headers={'User-Agent':'PetLMResearch/0.1 (educational local dialogue training)'}) as client:
        with client.stream('GET',URL) as response:
            response.raise_for_status()
            with partial.open('wb') as handle:
                for chunk in response.iter_bytes(1024*1024):
                    handle.write(chunk)
    with gzip.open(partial,'rt',encoding='utf-8') as handle:
        first=json.loads(next(handle))
        if not all(k in first for k in ('message_id','role','text','lang')):
            raise RuntimeError('Unexpected OpenAssistant message schema')
    partial.replace(output)
    print(json.dumps({'dataset':DATASET,'revision':REVISION,'license':'Apache-2.0',
        'file':str(output),'sha256':checksum(output),'bytes':output.stat().st_size},ensure_ascii=False))


def eligible(row):
    return (row.get('lang') == 'zh' and row.get('role') in ('prompter','assistant')
            and row.get('deleted') is False and row.get('review_result') is True
            and row.get('synthetic') is False and not row.get('model_name')
            and isinstance(row.get('text'),str))


def normalize(text):
    text = CONVERTER.convert(clean(text))
    return text.strip()


def acceptable(text, minimum, maximum):
    if not minimum <= len(text) <= maximum:
        return False
    if len(CHINESE.findall(text))/len(text) < 0.35:
        return False
    if text.count('http') > 1 or text.count('```') > 0:
        return False
    return True


def prompt_history(messages, parent, tokenizer, answer_tokens, context):
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
    if any(row['role']=='assistant' and MODEL_IDENTITY.search(row['text']) for row in chain):
        return None
    # Keep complete recent turns; the last item is always the user message.
    chain=chain[-5:]
    while chain:
        prompt=''.join(f"{'用户' if row['role']=='prompter' else '助手'}：{row['text']}\n" for row in chain)+'助手：'
        if len(tokenizer.encode(prompt))+answer_tokens <= context:
            return prompt
        chain=chain[2:]
    return None


def prepare(raw, output, tokenizer_file, context=256):
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new snapshot path.')
    tokenizer = spm.SentencePieceProcessor(model_proto=tokenizer_file.read_bytes())
    messages = {}
    rejected = Counter()
    with gzip.open(raw,'rt',encoding='utf-8') as handle:
        for line in handle:
            row=json.loads(line)
            if not eligible(row):
                rejected['ineligible_message'] += 1
                continue
            row['text']=normalize(row['text'])
            messages[row['message_id']]=row
    pairs=[]
    seen=set()
    trees=set()
    for row in messages.values():
        if row['role']!='assistant':
            continue
        parent=messages.get(row.get('parent_id'))
        if not parent or parent['role']!='prompter':
            rejected['missing_chinese_parent'] += 1
            continue
        user,answer=parent['text'],row['text']
        if MODEL_IDENTITY.search(answer):
            rejected['other_model_identity'] += 1
            continue
        if not acceptable(user,4,250) or not acceptable(answer,8,450):
            rejected['content_filter'] += 1
            continue
        answer_ids=tokenizer.encode(answer)+[tokenizer.eos_id()]
        prompt=prompt_history(messages,parent,tokenizer,len(answer_ids),context)
        if prompt is None or len(answer_ids)<4:
            rejected['context_limit'] += 1
            continue
        key=hashlib.sha256((user+'\0'+answer).encode('utf-8')).hexdigest()
        if key in seen:
            rejected['duplicate_pair'] += 1
            continue
        seen.add(key)
        tree=str(row.get('message_tree_id') or parent.get('message_tree_id') or parent['message_id'])
        bucket=int(hashlib.sha256(tree.encode()).hexdigest()[:8],16)%100
        split='test' if bucket<5 else 'validation' if bucket<10 else 'train'
        trees.add(tree)
        pairs.append({'id':key,'tree_id':tree,'source_message_id':row['message_id'],
            'split':split,'prompt':prompt,'answer':answer})
    if not pairs:
        raise RuntimeError('No eligible Chinese dialogue pairs found.')
    output.mkdir(parents=True)
    counts=Counter(p['split'] for p in pairs)
    for split in ('train','validation','test'):
        with (output/f'{split}.jsonl').open('w',encoding='utf-8') as handle:
            for pair in pairs:
                if pair['split']==split:
                    handle.write(json.dumps(pair,ensure_ascii=False)+'\n')
    report={'source':DATASET,'revision':REVISION,'license':'Apache-2.0',
        'raw_sha256':checksum(raw),'tokenizer_sha256':checksum(tokenizer_file),
        'counts':dict(counts),'trees':len(trees),'rejected':dict(rejected),
        'format':'Chinese user/assistant pairs; assistant tokens only are training targets',
        'limitations':['Human-authored assistant responses are general-purpose, not reviewed for pet persona.',
            'Automatic filtering does not guarantee factual correctness or age suitability.',
            'Pairs from the same conversation tree stay in one split.']}
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if min(counts['train'],counts['validation'],counts['test'])<10:
        raise RuntimeError('Too few pairs in one split for a useful dialogue experiment.')


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('fetch')
    prep=sub.add_parser('prepare')
    prep.add_argument('--raw',type=Path,default=RAW)
    prep.add_argument('--output',type=Path,default=BASE/'dialogue'/'prepared-oasst1-zh')
    prep.add_argument('--tokenizer',type=Path,default=BASE/'tokens-curated-expanded'/'tokenizer.model')
    prep.add_argument('--context',type=int,default=256)
    args=parser.parse_args()
    if args.command=='fetch':
        fetch()
    else:
        prepare(args.raw,args.output,args.tokenizer,args.context)


if __name__=='__main__':
    main()
