"""Merge explicitly human-approved pet dialogue drafts into a training snapshot."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import sentencepiece as spm

from .data import BASE
from .dialogue_train import encode_pair
from .run import checksum


def read_approved(path):
    approved=[]
    statuses=Counter()
    seen=set()
    with path.open(encoding='utf-8') as handle:
        for line_number,line in enumerate(handle,1):
            row=json.loads(line)
            status=row.get('review_status')
            statuses[status or 'missing']+=1
            if status!='approved':
                if status not in ('pending','rejected'):
                    raise ValueError(f'Line {line_number}: review_status must be pending, rejected or approved.')
                continue
            if not row.get('reviewer','').strip() or not row.get('reviewed_at','').strip():
                raise ValueError(f'Line {line_number}: approved items require reviewer and reviewed_at.')
            if not all(isinstance(row.get(key),str) and row[key].strip()
                       for key in ('id','prompt','answer')):
                raise ValueError(f'Line {line_number}: approved items require non-empty id, prompt and answer.')
            if row['id'] in seen:
                raise ValueError(f'Line {line_number}: duplicate id {row["id"]!r}.')
            seen.add(row['id'])
            approved.append(row)
    return approved,statuses


def prepare(drafts, base, output, tokenizer_path, manual_validation_percent=0.0):
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new snapshot directory.')
    base_manifest_path=base/'manifest.json'
    base_manifest=json.loads(base_manifest_path.read_text(encoding='utf-8'))
    if checksum(tokenizer_path)!=base_manifest['tokenizer_sha256']:
        raise RuntimeError('Tokenizer differs from the base dialogue snapshot.')
    tokenizer=spm.SentencePieceProcessor(model_proto=tokenizer_path.read_bytes())
    approved,statuses=read_approved(drafts)
    if not approved:
        raise RuntimeError('No human-approved dialogue drafts found; nothing was added to training.')

    rows=[]
    seen_pairs=set()
    for split in ('train','validation','test'):
        with (base/f'{split}.jsonl').open(encoding='utf-8') as handle:
            for line in handle:
                row=json.loads(line)
                seen_pairs.add((row['prompt'],row['answer']))
                rows.append(row)

    context=256
    if rows:
        # The base model context is fixed by the prepared project snapshots.
        for row in approved:
            encode_pair(tokenizer,row['prompt'],row['answer'],context)
            key=(row['prompt'],row['answer'])
            if key in seen_pairs:
                raise ValueError(f'Duplicate dialogue content in approved item {row["id"]!r}.')
            seen_pairs.add(key)
            bucket=int(hashlib.sha256(row['id'].encode()).hexdigest()[:8],16)%10000
            split='validation' if bucket<round(manual_validation_percent*100) else 'train'
            rows.append({'id':row['id'],'tree_id':f'manual:{row["id"]}',
                'source_message_id':row['id'],'split':split,'prompt':row['prompt'],
                'answer':row['answer'],'topic':row.get('topic',''),'review_status':'approved',
                'reviewer':row['reviewer'].strip(),'reviewed_at':row['reviewed_at'].strip(),
                'review_method':row.get('review_method','item_level'),
                'review_scope':row.get('review_scope','item_by_item')})

    output.mkdir(parents=True)
    counts=Counter(row['split'] for row in rows)
    for split in ('train','validation','test'):
        with (output/f'{split}.jsonl').open('w',encoding='utf-8') as handle:
            for row in rows:
                if row['split']==split:
                    handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    report={
        'base_snapshot':str(base.resolve()),
        'base_manifest_sha256':checksum(base_manifest_path),
        'manual_drafts':str(drafts.resolve()),
        'manual_drafts_sha256':checksum(drafts),
        'tokenizer_sha256':base_manifest['tokenizer_sha256'],
        'counts':dict(counts),
        'manual_approved_added':len(approved),
        'manual_review_statuses':dict(statuses),
        'manual_review_methods':dict(Counter(row.get('review_method','item_level') for row in approved)),
        'manual_validation_percent':manual_validation_percent,
        'review_policy':('Only rows marked approved with non-empty reviewer and reviewed_at are added; '
                         'manual validation assignment is deterministic by item id.'),
        'limitations':['Approval metadata is an auditable declaration, not independent verification of reviewer identity.',
            'Human reviewers must check factuality, age suitability, privacy and safety before approving.',
            'The retained validation and test splits come unchanged from the base snapshot.']
    }
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare',))
    parser.add_argument('--drafts',type=Path,default=BASE/'dialogue'/'pet-scenarios-review.jsonl')
    parser.add_argument('--base',type=Path,default=BASE/'dialogue'/'prepared-oasst1-zh-persona-clean-v2')
    parser.add_argument('--output',type=Path,default=BASE/'dialogue'/'prepared-pet-reviewed-v1')
    parser.add_argument('--tokenizer',type=Path,default=BASE/'tokens-curated-wiki-expanded'/'tokenizer.model')
    parser.add_argument('--manual-validation-percent',type=float,default=0.0)
    args=parser.parse_args()
    if not 0<=args.manual_validation_percent<100:
        parser.error('manual-validation-percent must be at least 0 and below 100.')
    prepare(args.drafts,args.base,args.output,args.tokenizer,args.manual_validation_percent)


if __name__=='__main__':
    main()
