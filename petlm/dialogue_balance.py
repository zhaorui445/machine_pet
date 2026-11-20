"""Create an immutable SFT snapshot selected on natural-dialogue validation rows."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from .data import BASE
from .run import checksum


SYNTHETIC_GROUPS={'diverse_synthetic','confirmed_template'}


def line_count(path):
    with path.open(encoding='utf-8') as handle:
        return sum(1 for line in handle if line.strip())


def build(source,output):
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable output path.')
    source_manifest_path=source/'manifest.json'
    source_manifest=json.loads(source_manifest_path.read_text(encoding='utf-8'))
    output.mkdir(parents=True)
    shutil.copyfile(source/'train.jsonl',output/'train.jsonl')
    shutil.copyfile(source/'test.jsonl',output/'test.jsonl')
    counts=Counter()
    groups=Counter()
    with (source/'validation.jsonl').open(encoding='utf-8') as handle, \
            (output/'validation.jsonl').open('w',encoding='utf-8') as target:
        for line in handle:
            row=json.loads(line)
            group=row.get('training_group','legacy')
            if group in SYNTHETIC_GROUPS:
                counts['validation_synthetic_removed']+=1
                continue
            target.write(line)
            counts['validation_natural_kept']+=1
            groups[group]+=1
    report={'kind':'natural_validation_balanced_sft','source_snapshot':str(source),
        'source_manifest_sha256':checksum(source_manifest_path),
        'tokenizer_sha256':source_manifest['tokenizer_sha256'],
        'counts':{'train':line_count(output/'train.jsonl'),
                  'validation':counts['validation_natural_kept'],
                  'test':line_count(output/'test.jsonl')},
        'validation_groups':dict(groups),'filter_counts':dict(counts),
        'train_sha256':checksum(output/'train.jsonl'),'validation_sha256':checksum(output/'validation.jsonl'),
        'test_sha256':checksum(output/'test.jsonl'),
        'selection_policy':('Training and test rows are byte-preserved. Model selection uses only '
                            'pre-existing non-synthetic validation rows.'),
        'warning':('Source labels and automated filters do not constitute a new human item-level audit.')}
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=BASE/'dialogue'/'prepared-pet-human-v11')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    build(args.source,args.output)


if __name__=='__main__':
    main()
