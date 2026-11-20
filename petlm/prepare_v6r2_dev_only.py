"""Prepare capability LoRA data from approved v6r2 development rows only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .capability_router import CAPABILITIES
from .run import checksum


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def write(path, rows):
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def build(reviewed, tokenizer, output):
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable path.')
    manifest_path=reviewed/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('dataset_id')!='v6r2' or manifest.get('review_status')!='independent_review_complete':
        raise RuntimeError('Expected independently reviewed v6r2 data.')
    development_path=reviewed/'development.jsonl'
    rows=read(development_path)
    if len(rows)!=manifest['counts']['development'] or any(row.get('split')!='development' for row in rows):
        raise RuntimeError('Development split is incomplete or contaminated.')
    output.mkdir(parents=True)
    root={'kind':'v6r2_approved_development_only_lora_v1','dataset_id':'v6r2-dev-only',
          'single_group_training':True,'source_review_manifest_sha256':checksum(manifest_path),
          'source_development_sha256':checksum(development_path),
          'tokenizer_sha256':checksum(tokenizer),'capabilities':{},
          'isolation':('Only the approved development.jsonl file was read. Internal-test and frozen files '
                       'were not read or copied into this training snapshot.')}
    for capability in CAPABILITIES:
        selected=sorted((row for row in rows if row['capability']==capability),key=lambda row:row['id'])
        if len(selected)!=40:
            raise RuntimeError(f'Expected 40 development rows for {capability}.')
        # Hold out every fifth row within each behavior family for development validation.
        family_positions=Counter()
        train=[]; validation=[]
        for row in selected:
            position=family_positions[row['behavior_family']]
            family_positions[row['behavior_family']]+=1
            item={'id':row['id'],'prompt':row['prompt'],'answer':row['proposed_answer'],
                  'training_group':'human_reviewed_dialogue_v6r2_dev_only',
                  'capability':capability,'behavior_family':row['behavior_family'],
                  'reviewer_id':row['reviewer_id'],'reviewed_at':row['reviewed_at'],
                  'reviewer_signature':row['reviewer_signature']}
            (validation if position%5==0 else train).append(item)
        target=output/capability
        target.mkdir()
        write(target/'train.jsonl',train); write(target/'validation.jsonl',validation)
        report={'kind':'v6r2_approved_development_only_lora_v1','capability':capability,
                'single_group_training':True,'tokenizer_sha256':checksum(tokenizer),
                'counts':{'train':len(train),'validation':len(validation)},
                'source_split':'development_only','internal_test_used':False,'frozen_used':False,
                'files':{'train':checksum(target/'train.jsonl'),
                         'validation':checksum(target/'validation.jsonl')}}
        (target/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        root['capabilities'][capability]=report
    (output/'manifest.json').write_text(json.dumps(root,ensure_ascii=False,indent=2),encoding='utf-8')
    return root


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reviewed',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.reviewed,args.tokenizer,args.output),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
