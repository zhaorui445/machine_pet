"""Merge reviewed behavior SFT rows with the natural dialogue snapshot."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from .run import checksum


def read(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized(prompt):
    return re.sub(r'\s+','',prompt).casefold()


def merge(natural, behavior, output):
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable path.')
    output.mkdir(parents=True)
    rows_by_split={}
    counts=Counter()
    for split in ('train','validation','test'):
        rows=[]
        seen=set()
        for source in (natural,behavior):
            path=source/f'{split}.jsonl'
            if not path.exists():
                continue
            for row in read(path):
                key=normalized(row['prompt'])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
                counts[(split,row.get('training_group','unknown'))]+=1
        if not rows:
            raise RuntimeError(f'No rows for split {split}.')
        rows_by_split[split]=rows
        with (output/f'{split}.jsonl').open('w',encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    natural_manifest=natural/'manifest.json'
    behavior_manifest=behavior/'manifest.json'
    manifest={'kind':'combined_natural_and_reviewed_behavior_sft_v1',
              'sources':{'natural':str(natural),'natural_manifest_sha256':checksum(natural_manifest),
                         'behavior':str(behavior),'behavior_manifest_sha256':checksum(behavior_manifest)},
              'tokenizer_sha256':json.loads(behavior_manifest.read_text(encoding='utf-8'))['tokenizer_sha256'],
              'counts':{split:len(rows) for split,rows in rows_by_split.items()},
              'training_groups':dict(Counter(row.get('training_group','unknown')
                                             for row in rows_by_split['train'])),
              'counts_by_split_and_group':{f'{split}:{group}':value for (split,group),value in sorted(counts.items())},
              'deduplication':'normalized prompt, natural rows retained before behavior rows',
              'warning':'Behavior rows are item-reviewed in this workspace; independent human audit remains recommended.'}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--natural',type=Path,required=True)
    parser.add_argument('--behavior',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    merge(args.natural,args.behavior,args.output)


if __name__=='__main__':
    main()
