"""Create a higher-precision SFT snapshot while preserving the frozen test split."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from .data import BASE
from .run import checksum


KEEP_GROUPS={'legacy','diverse_synthetic','natural_human_rule_assisted_item_audit'}


def keep(row):
    group=row.get('training_group')
    if group in KEEP_GROUPS:
        return True,'retained_group'
    if group=='natural_human_oasst2':
        quality=row.get('source_quality')
        helpfulness=row.get('source_helpfulness')
        if (isinstance(quality,(int,float)) and isinstance(helpfulness,(int,float))
                and quality>=0.75 and helpfulness>=0.75 and row.get('source_rank') in (0,None)):
            return True,'high_score_oasst2'
        return False,'oasst2_below_quality_threshold'
    return False,'unreviewed_or_unsupported_natural_source'


def build(source,output):
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable output path.')
    manifest_path=source/'manifest.json'
    source_manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    output.mkdir(parents=True)
    counts=Counter()
    groups=Counter()
    decisions=Counter()
    for split in ('train','validation','test'):
        with (source/f'{split}.jsonl').open(encoding='utf-8') as handle, \
                (output/f'{split}.jsonl').open('w',encoding='utf-8') as target:
            for line in handle:
                row=json.loads(line)
                accepted,reason=(True,'frozen_test_preserved') if split=='test' else keep(row)
                decisions[f'{split}/{reason}/{"kept" if accepted else "removed"}']+=1
                if not accepted:
                    continue
                target.write(json.dumps(row,ensure_ascii=False)+'\n')
                counts[split]+=1
                if split=='train':
                    groups[row.get('training_group','unspecified')]+=1
    report={'kind':'quality_filtered_cross_source_pet_sft',
        'source_snapshot':str(source),'source_manifest_sha256':checksum(manifest_path),
        'tokenizer_sha256':source_manifest['tokenizer_sha256'],
        'counts':dict(counts),'training_groups':dict(groups),'decisions':dict(decisions),
        'quality_policy':{'oasst2_minimum_quality':0.75,'oasst2_minimum_helpfulness':0.75,
            'oasst2_top_rank_only':True,
            'retained_groups':sorted(KEEP_GROUPS),
            'generic_unreviewed_oasst1_excluded':True,'test_split_preserved_byte_rows':True},
        'limitations':['OASST2 source scores and automated filters do not guarantee correctness.',
            'Rule-assisted audits are not human item-by-item review.',
            'Synthetic dialogue remains the majority of unique training rows.']}
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


def build_natural(source, approved, output, tokenizer_path, minimum_score=0.8):
    """Build a small natural-dialogue snapshot without synthetic replay rows."""
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable output path.')
    source_manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    rows=[]
    decisions=Counter()
    for split in ('train','validation','test'):
        with (source/f'{split}.jsonl').open(encoding='utf-8') as handle:
            for line in handle:
                row=json.loads(line)
                if row.get('training_group')!='natural_human_oasst2':
                    decisions[f'{split}/non_oasst2_removed']+=1
                    continue
                quality=row.get('source_quality')
                helpfulness=row.get('source_helpfulness')
                if (not row.get('source_review_result') or row.get('source_rank') not in (0,None)
                        or not isinstance(quality,(int,float)) or not isinstance(helpfulness,(int,float))
                        or quality<minimum_score or helpfulness<minimum_score):
                    decisions[f'{split}/below_natural_threshold']+=1
                    continue
                rows.append(dict(row,source_selection='human_authored_oasst2_high_score'))
                decisions[f'{split}/natural_kept']+=1
    if approved:
        with approved.open(encoding='utf-8') as handle:
            for line in handle:
                row=json.loads(line)
                if row.get('review_status')!='approved':
                    continue
                rows.append(dict(row,split='train',training_group='human_approved_pet',
                                 source_selection='user_confirmed_pet_scenarios'))
                decisions['approved_pet_added']+=1
    # Holdout priority prevents prompt leakage between the selected splits.
    priority={'train':0,'validation':1,'test':2}
    selected={}
    for row in rows:
        key=re.sub(r'\s+','',row['prompt']).casefold()
        previous=selected.get(key)
        if previous is None or priority[row['split']]>priority[previous['split']]:
            selected[key]=row
    rows=sorted(selected.values(),key=lambda row:(priority[row['split']],row['id']))
    output.mkdir(parents=True)
    counts=Counter(row['split'] for row in rows)
    groups=Counter(row.get('training_group','unknown') for row in rows if row['split']=='train')
    for split in ('train','validation','test'):
        with (output/f'{split}.jsonl').open('w',encoding='utf-8') as handle:
            for row in rows:
                if row['split']==split:
                    handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    report={'kind':'natural_high_quality_multiturn_sft','source_snapshot':str(source.resolve()),
        'source_manifest_sha256':checksum(source/'manifest.json'),'approved_pet':str(approved) if approved else None,
        'tokenizer_sha256':checksum(tokenizer_path),'counts':dict(counts),'training_groups':dict(groups),
        'decisions':dict(decisions),'selection_policy':{'oasst2_minimum_quality':minimum_score,
            'oasst2_minimum_helpfulness':minimum_score,'oasst2_top_rank_only':True,
            'synthetic_rows_excluded':True,'prompt_dedup':'test > validation > train'},
        'warning':'Human-authored source scores and user confirmation are retained; this snapshot is not a new independent item-by-item human audit.'}
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=BASE/'dialogue'/'prepared-pet-human-v11')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--natural-only',action='store_true')
    parser.add_argument('--approved',type=Path)
    parser.add_argument('--tokenizer',type=Path)
    parser.add_argument('--minimum-score',type=float,default=0.8)
    args=parser.parse_args()
    if args.natural_only:
        if args.tokenizer is None:
            parser.error('--tokenizer is required with --natural-only')
        build_natural(args.source,args.approved,args.output,args.tokenizer,args.minimum_score)
    else:
        build(args.source,args.output)


if __name__=='__main__':
    main()
