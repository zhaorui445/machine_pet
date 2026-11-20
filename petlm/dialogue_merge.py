"""Merge pinned human-authored dialogue snapshots without holdout leakage."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from .data import BASE
from .run import checksum


SPLIT_PRIORITY={'train':0,'validation':1,'test':2}


def canonical_prompt(text):
    return re.sub(r'\s+','',text).casefold()


def quality(row):
    values=[row.get('source_quality'),row.get('source_helpfulness')]
    numeric=[float(value) for value in values if isinstance(value,(int,float))]
    return sum(numeric)/len(numeric) if numeric else -1.0


def merge(inputs, output):
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable output path.')
    candidates=[]
    sources=[]
    for folder in inputs:
        manifest_path=folder/'manifest.json'
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        dataset=manifest.get('source')
        revision=manifest.get('revision')
        license_note=manifest.get('license')
        if not all(isinstance(value,str) and value for value in (dataset,revision,license_note)):
            raise ValueError(f'Incomplete source provenance in {manifest_path}')
        sources.append({'dataset':dataset,'revision':revision,'license':license_note,
                        'manifest_sha256':checksum(manifest_path),'path':str(folder)})
        for split in ('train','validation','test'):
            with (folder/f'{split}.jsonl').open(encoding='utf-8') as handle:
                for line in handle:
                    row=json.loads(line)
                    if row.get('split',split)!=split:
                        raise ValueError(f'Split mismatch in {folder/f"{split}.jsonl"}')
                    row={**row,'split':split,'source_dataset':row.get('source_dataset',dataset),
                         'source_revision':row.get('source_revision',revision),
                         'source_license':row.get('source_license',license_note)}
                    row['tree_id']=f'{dataset}:{row.get("tree_id",row["id"])}'
                    candidates.append(row)
    selected={}
    removed=Counter()
    for row in candidates:
        key=hashlib.sha256(canonical_prompt(row['prompt']).encode()).hexdigest()
        previous=selected.get(key)
        if previous is None:
            selected[key]=row
            continue
        previous_rank=(SPLIT_PRIORITY[previous['split']],quality(previous))
        current_rank=(SPLIT_PRIORITY[row['split']],quality(row))
        if current_rank>previous_rank:
            removed[f'prompt_duplicate_removed_from_{previous["split"]}']+=1
            selected[key]=row
        else:
            removed[f'prompt_duplicate_removed_from_{row["split"]}']+=1
    rows=sorted(selected.values(),key=lambda row:(row['split'],row['source_dataset'],row['id']))
    counts=Counter(row['split'] for row in rows)
    dataset_counts=Counter(row['source_dataset'] for row in rows)
    output.mkdir(parents=True)
    for split in ('train','validation','test'):
        with (output/f'{split}.jsonl').open('w',encoding='utf-8') as handle:
            for row in rows:
                if row['split']==split:
                    handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    report={'kind':'human_authored_chinese_dialogue_merge','counts':dict(counts),
        'counts_by_dataset':dict(dataset_counts),'sources':sources,'removed':dict(removed),
        'dedup_policy':'Normalized prompt text is globally unique; test wins over validation and train.',
        'limitations':['Source review is not a new local item-by-item human audit.',
            'Automatic source filters do not guarantee factual correctness or pet-persona suitability.']}
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    merge(args.input,args.output)


if __name__=='__main__':
    main()
