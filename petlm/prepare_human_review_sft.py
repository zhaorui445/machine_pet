"""Create capability SFT snapshots using the finalized human-review v2 data.

Only the finalized development split is added to training. The finalized
internal_test split is copied to validation for process monitoring, while the
frozen blind prompts and answer key are never read by this command.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from .run import checksum

CAPABILITIES=('safety_behavior','memory_management','motion_clarification','format_control')


def read(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def reviewed_rows(path, capability, split, dataset_id):
    rows=[]
    for row in read(path / f'{split}.jsonl'):
        if row.get('capability') != capability:
            continue
        rows.append({
            'id':row['id'],
            'prompt':row['prompt'],
            'answer':row['proposed_answer'],
            'training_group':f'human_reviewed_dialogue_{dataset_id}',
            'source_kind':'independent_human_reviewed_dialogue',
            'source_review_status':'independent_review_complete',
            'reviewer_id':row.get('reviewer_id'),
            'reviewed_at':row.get('reviewed_at'),
            'review_signature':row.get('reviewer_signature'),
            'behavior_family':row['behavior_family'],
            'adapter_capability':capability,
        })
    return rows


def build(source, reviewed, output, frozen_prompts):
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable path.')
    source_manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    reviewed_manifest=json.loads((reviewed/'manifest.json').read_text(encoding='utf-8'))
    if reviewed_manifest.get('review_status') != 'independent_review_complete':
        raise RuntimeError('Reviewed source is not finalized independently.')
    frozen_ids={row['id'] for row in read(frozen_prompts)}
    output.mkdir(parents=True)
    dataset_id=reviewed_manifest.get('dataset_id','reviewed')
    root={'kind':'human_reviewed_capability_sft_v3','dataset_id':dataset_id,
          'source':str(source),'source_manifest_sha256':checksum(source/'manifest.json'),
          'reviewed_source':str(reviewed),'reviewed_manifest_sha256':checksum(reviewed/'manifest.json'),
          'frozen_prompts':str(frozen_prompts),'frozen_prompts_sha256':checksum(frozen_prompts),
          'frozen_prompt_overlap':0,'capabilities':{}}
    for capability in CAPABILITIES:
        target=output/capability
        target.mkdir()
        capability_source=source/capability
        capability_manifest=json.loads((capability_source/'manifest.json').read_text(encoding='utf-8'))
        source_rows={split:read(capability_source/f'{split}.jsonl') for split in ('train','validation','test')}
        development=reviewed_rows(reviewed,capability,'development',dataset_id)
        internal=reviewed_rows(reviewed,capability,'internal_test',dataset_id)
        for row in development+internal:
            if row['id'] in frozen_ids:
                raise RuntimeError(f'Human reviewed row overlaps frozen prompt id: {row["id"]}')
        # Preserve the established natural/behavior snapshot and append only
        # reviewed development rows to train. Internal test rows are validation.
        rows={'train':source_rows['train']+development,
              'validation':source_rows['validation']+internal,
              'test':source_rows['test']}
        for split,items in rows.items():
            with (target/f'{split}.jsonl').open('w',encoding='utf-8') as handle:
                for row in items:
                    handle.write(json.dumps(row,ensure_ascii=False)+'\n')
        report={'kind':'human_reviewed_capability_sft_v3','dataset_id':dataset_id,
                'capability':capability,
                'tokenizer_sha256':capability_manifest['tokenizer_sha256'],
                'source_manifest_sha256':checksum(capability_source/'manifest.json'),
                'reviewed_manifest_sha256':checksum(reviewed/'manifest.json'),
                'review_status':'independent_review_complete',
                'training_added_from':'development_only','internal_test_role':'validation_only',
                'frozen_prompt_overlap':0,
                'counts':{split:len(items) for split,items in rows.items()},
                'reviewed_counts':{'development':len(development),'internal_test':len(internal)},
                'files':{split:checksum(target/f'{split}.jsonl') for split in rows}}
        (target/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        root['capabilities'][capability]=report
    (output/'manifest.json').write_text(json.dumps(root,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(root,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--reviewed',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frozen-prompts',type=Path,required=True)
    args=parser.parse_args()
    build(args.source,args.reviewed,args.output,args.frozen_prompts)


if __name__=='__main__':
    main()
