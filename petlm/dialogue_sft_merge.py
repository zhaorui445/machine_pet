"""Build a cross-source SFT snapshot with global tree and prompt isolation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import sentencepiece as spm

from .data import BASE
from .run import checksum


PRIORITY={'train':0,'validation':1,'test':2}


def normalized_prompt(prompt):
    return re.sub(r'\s+','',prompt).casefold()


def canonical_tree(row, snapshot_name):
    tree=str(row.get('tree_id') or row.get('id'))
    if tree.startswith('OpenAssistant/'):
        tree=tree.split(':',1)[-1]
    if re.fullmatch(r'[0-9a-fA-F-]{32,36}',tree):
        return 'openassistant:'+tree.lower()
    return snapshot_name+':'+tree


def build(inputs, output, tokenizer_path, context=512):
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable output path.')
    tokenizer_hash=checksum(tokenizer_path)
    tokenizer=spm.SentencePieceProcessor(model_proto=tokenizer_path.read_bytes())
    rows=[]
    sources=[]
    rejected=Counter()
    for folder in inputs:
        manifest_path=folder/'manifest.json'
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        source_tokenizer=manifest.get('tokenizer_sha256')
        sources.append({'path':str(folder),'manifest_sha256':checksum(manifest_path),
                        'source_tokenizer_sha256':source_tokenizer,
                        'counts':manifest.get('counts',{})})
        snapshot_name=folder.name
        for split in ('train','validation','test'):
            with (folder/f'{split}.jsonl').open(encoding='utf-8') as handle:
                for line in handle:
                    row=json.loads(line)
                    if row.get('split',split)!=split:
                        raise ValueError(f'Split mismatch in {folder/f"{split}.jsonl"}')
                    row={**row,'split':split}
                    if (row.get('review_status')=='approved'
                            and row.get('review_scope')!='item_level_automated_not_human'):
                        row['training_group']='human_approved_pet'
                    else:
                        row.setdefault('training_group','natural_human_openassistant')
                    row['_tree']=canonical_tree(row,snapshot_name)
                    if len(tokenizer.encode(row['prompt']))+len(tokenizer.encode(row['answer']))+1>context:
                        rejected[f'context_limit_{split}']+=1
                        continue
                    rows.append(row)

    tree_priority={}
    for row in rows:
        tree_priority[row['_tree']]=max(tree_priority.get(row['_tree'],0),PRIORITY[row['split']])
    isolated=[]
    for row in rows:
        if PRIORITY[row['split']]<tree_priority[row['_tree']]:
            rejected[f'tree_leakage_removed_from_{row["split"]}']+=1
        else:
            isolated.append(row)

    selected={}
    for row in isolated:
        key=hashlib.sha256(normalized_prompt(row['prompt']).encode()).hexdigest()
        previous=selected.get(key)
        if previous is None:
            selected[key]=row
            continue
        current_rank=(PRIORITY[row['split']],row.get('source_quality',-1) or -1,
                      row.get('source_helpfulness',-1) or -1)
        previous_rank=(PRIORITY[previous['split']],previous.get('source_quality',-1) or -1,
                       previous.get('source_helpfulness',-1) or -1)
        if current_rank>previous_rank:
            rejected[f'prompt_duplicate_removed_from_{previous["split"]}']+=1
            selected[key]=row
        else:
            rejected[f'prompt_duplicate_removed_from_{row["split"]}']+=1

    final_rows=sorted(selected.values(),key=lambda row:(row['split'],row['_tree'],row['id']))
    counts=Counter(row['split'] for row in final_rows)
    groups=Counter(row['training_group'] for row in final_rows if row['split']=='train')
    output.mkdir(parents=True)
    for split in ('train','validation','test'):
        with (output/f'{split}.jsonl').open('w',encoding='utf-8') as handle:
            for row in final_rows:
                if row['split']==split:
                    handle.write(json.dumps({k:v for k,v in row.items() if k!='_tree'},
                                            ensure_ascii=False)+'\n')
    report={'kind':'cross_source_pet_sft','tokenizer_sha256':tokenizer_hash,
        'context':context,'counts':dict(counts),'training_groups':dict(groups),
        'sources':sources,'rejected':dict(rejected),
        'isolation':('Whole-tree split priority is test > validation > train, followed by global '
                     'normalized-prompt deduplication with the same priority.'),
        'warning':('Synthetic and automatically reviewed rows retain their original provenance; '
                   'they are not represented as human item-level review.')}
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path,
        default=BASE/'tokens-chinese-v8-50m-frozen'/'tokenizer.model')
    parser.add_argument('--context',type=int,default=512)
    args=parser.parse_args()
    build(args.input,args.output,args.tokenizer,args.context)


if __name__=='__main__':
    main()
