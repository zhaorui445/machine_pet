"""Prepare high-quality human-authored Chinese multi-turn dialogue for item review."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import sentencepiece as spm

from .data import BASE
from .dialogue_data import (DATASET, FILE, MODEL_IDENTITY, REVISION, acceptable,
                            eligible, normalize)
from .run import checksum


RISK_LABELS = ('spam', 'lang_mismatch', 'pii', 'not_appropriate',
               'hate_speech', 'sexual_content')


def label_value(row, name, default):
    value=((row.get('labels') or {}).get(name) or {}).get('value',default)
    try:
        return float(value)
    except (TypeError,ValueError):
        return float(default)


def safe_labels(row):
    return (all(label_value(row,name,1.0)<=0.2 for name in RISK_LABELS)
            and label_value(row,'toxicity',1.0)<=0.5)


def safe_history_labels(row):
    """Reject explicit risk flags while tolerating absent legacy label fields."""
    return (all(label_value(row,name,0.0)<=0.2 for name in RISK_LABELS)
            and label_value(row,'toxicity',0.0)<=0.5)


def high_quality_assistant(row):
    return (eligible(row) and row.get('role')=='assistant' and safe_labels(row)
            and label_value(row,'quality',0.0)>=0.5
            and label_value(row,'helpfulness',0.0)>=0.5)


def history_eligible(row):
    return (row.get('lang')=='zh' and row.get('role') in ('prompter','assistant')
            and row.get('deleted') is False and row.get('synthetic') is False
            and not row.get('model_name') and isinstance(row.get('text'),str)
            and safe_history_labels(row))


def partition_for_tree(tree_id):
    bucket=int(hashlib.sha256(tree_id.encode()).hexdigest()[:8],16)%100
    if bucket<15:
        return 'blind'
    if bucket<30:
        return 'validation'
    return 'train'


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
    if any(row.get('role') != ('prompter' if index%2==0 else 'assistant')
           for index,row in enumerate(chain)):
        return None
    return chain


def render_prompt(chain):
    return ''.join(f"{'用户' if row['role']=='prompter' else '助手'}：{row['text']}\n"
                   for row in chain)+'助手：'


def topic_for(text):
    categories=(
        ('学习与解释',('学习','作业','考试','解释','数学','物理','历史','英语','编程')),
        ('情绪与关系',('难过','焦虑','生气','朋友','家人','同学','关系','压力')),
        ('生活规划',('计划','时间','工作','生活','习惯','睡眠','运动','旅行')),
        ('安全与健康',('安全','药','生病','疼','危险','医院','医生','健康')),
        ('创作与兴趣',('写','故事','音乐','绘画','电影','游戏','摄影','创作')),
    )
    for topic,keywords in categories:
        if any(keyword in text for keyword in keywords):
            return topic
    return '自然多轮问答'


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row,ensure_ascii=False)+'\n')


def build(raw, output, tokenizer_path, context=256, minimum_user_turns=1):
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable output path.')
    tokenizer=spm.SentencePieceProcessor(model_proto=tokenizer_path.read_bytes())
    messages={}
    rejected=Counter()
    with gzip.open(raw,'rt',encoding='utf-8') as handle:
        for line in handle:
            row=json.loads(line)
            if not history_eligible(row):
                continue
            row['text']=normalize(row['text'])
            messages[row['message_id']]=row

    candidates=[]
    blind_prompts=[]
    blind_keys=[]
    seen_pairs=set()
    tree_partitions={}
    minimum_chain_length=minimum_user_turns*2-1
    for row in messages.values():
        if not high_quality_assistant(row):
            rejected['assistant_quality']+=1
            continue
        parent=messages.get(row.get('parent_id'))
        if not parent or parent.get('role')!='prompter':
            rejected['missing_parent']+=1
            continue
        chain=chain_to(messages,parent)
        if chain is None or len(chain)<minimum_chain_length:
            rejected['insufficient_turns']+=1
            continue
        if any(item['role']=='assistant' and MODEL_IDENTITY.search(item['text'])
               for item in chain):
            rejected['model_identity_history']+=1
            continue
        if MODEL_IDENTITY.search(row['text']):
            rejected['model_identity']+=1
            continue
        if any(not acceptable(item['text'],4 if item['role']=='prompter' else 8,
                              250 if item['role']=='prompter' else 450)
               for item in chain) or not acceptable(row['text'],8,450):
            rejected['content_filter']+=1
            continue
        # Keep complete recent turns without splitting user/assistant pairs.
        chain=chain[-5:]
        while len(chain)>=minimum_chain_length:
            prompt=render_prompt(chain)
            if len(tokenizer.encode(prompt))+len(tokenizer.encode(row['text']))+1<=context:
                break
            chain=chain[2:]
        if len(chain)<minimum_chain_length:
            rejected['context_limit']+=1
            continue
        pair_key=(prompt,row['text'])
        if pair_key in seen_pairs:
            rejected['duplicate_pair']+=1
            continue
        seen_pairs.add(pair_key)
        tree=str(row.get('message_tree_id') or parent.get('message_tree_id') or parent['message_id'])
        partition=partition_for_tree(tree)
        if tree in tree_partitions and tree_partitions[tree]!=partition:
            raise RuntimeError('Conversation tree crossed data partitions.')
        tree_partitions[tree]=partition
        identifier=hashlib.sha256(('oasst-natural-v1\0'+row['message_id']).encode()).hexdigest()
        topic=topic_for(prompt+row['text'])
        common={'id':identifier,'tree_id':tree,'source_message_id':row['message_id'],
                'topic':topic,'prompt':prompt,'source_kind':'human_authored_openassistant',
                'source_dataset':DATASET,'source_revision':REVISION,'source_license':'Apache-2.0'}
        if partition=='blind':
            history=[{'user':chain[index]['text'],'assistant':chain[index+1]['text']}
                     for index in range(0,len(chain)-1,2)]
            blind_prompts.append({'id':identifier,'tree_id':tree,'topic':topic,
                                  'history':history,'user':chain[-1]['text'],
                                  'source_kind':'human_authored_openassistant'})
            blind_keys.append(dict(common,answer=row['text'],review_status='pending',
                                   reviewer='',reviewed_at='',
                                   rubric=['切题','语义连贯','事实与安全边界合理']))
        else:
            candidates.append(dict(common,answer=row['text'],split_hint=partition,
                                   training_group='natural_human_review_pending',
                                   review_status='pending',reviewer='',reviewed_at=''))

    candidate_trees={row['tree_id'] for row in candidates}
    blind_trees={row['tree_id'] for row in blind_prompts}
    if candidate_trees & blind_trees:
        raise RuntimeError('Blind conversation trees leaked into training candidates.')
    if not candidates or not blind_prompts:
        raise RuntimeError('Quality filters produced an empty candidate or blind partition.')
    output.mkdir(parents=True)
    write_jsonl(output/'candidates.pending.jsonl',candidates)
    write_jsonl(output/'blind'/'prompts.jsonl',blind_prompts)
    write_jsonl(output/'blind'/'answer-key.pending.jsonl',blind_keys)
    candidate_path=output/'candidates.pending.jsonl'
    blind_prompt_path=output/'blind'/'prompts.jsonl'
    blind_key_path=output/'blind'/'answer-key.pending.jsonl'
    split_counts=Counter(row['split_hint'] for row in candidates)
    topic_counts=Counter(row['topic'] for row in candidates)
    candidate_depths=Counter(row['prompt'].count('用户：') for row in candidates)
    blind_depths=Counter(len(row['history'])+1 for row in blind_prompts)
    report={'source':DATASET,'revision':REVISION,'license':'Apache-2.0',
        'raw_file':FILE,'raw_sha256':checksum(raw),'tokenizer_sha256':checksum(tokenizer_path),
        'candidates':len(candidates),'candidate_splits':dict(split_counts),
        'blind_count':len(blind_prompts),'candidate_trees':len(candidate_trees),
        'blind_trees':len(blind_trees),'topics':dict(sorted(topic_counts.items())),
        'candidate_user_turns':dict(sorted(candidate_depths.items())),
        'blind_user_turns':dict(sorted(blind_depths.items())),
        'rejected':dict(rejected),
        'quality_policy':{'human_authored':True,'original_review_passed':True,
            'minimum_quality':0.5,'minimum_helpfulness':0.5,
            'maximum_risk_labels':0.2,'maximum_toxicity':0.5,
            'minimum_user_turns':minimum_user_turns,'maximum_context_tokens':context,
            'final_answer_original_review_required':True,
            'history_original_review_required':False},
        'review_status':'pending_item_level_review',
        'isolation':'Training/validation candidates and blind prompts are disjoint by conversation tree.',
        'files':{
            'candidates':{'path':str(candidate_path),'sha256':checksum(candidate_path)},
            'frozen_blind_prompts':{'path':str(blind_prompt_path),'sha256':checksum(blind_prompt_path)},
            'pending_blind_answer_key':{'path':str(blind_key_path),'sha256':checksum(blind_key_path)}},
        'warning':('Automatic filtering is only pre-screening. Candidates and blind references remain '
                   'pending until explicit item-level human review.')}
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw',type=Path,default=BASE/'dialogue'/'raw'/FILE)
    parser.add_argument('--output',type=Path,default=BASE/'dialogue'/'natural-review-v1')
    parser.add_argument('--tokenizer',type=Path,default=BASE/'tokens-chinese-v7'/'tokenizer.model')
    parser.add_argument('--context',type=int,default=256)
    parser.add_argument('--minimum-user-turns',type=int,default=1)
    args=parser.parse_args()
    if args.minimum_user_turns<1:
        parser.error('minimum-user-turns must be positive.')
    build(args.raw,args.output,args.tokenizer,args.context,args.minimum_user_turns)


if __name__=='__main__':
    main()
