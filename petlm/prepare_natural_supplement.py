"""Select an unseen natural multi-turn supplement from human-authored OASST2."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from .data import BASE
from .run import checksum


def read(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def norm(text):
    return re.sub(r'\s+','',text).casefold()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--compare',type=Path,action='append',default=[])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--count',type=int,default=180)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'{args.output} exists; choose a new immutable path.')
    excluded=set()
    for folder in args.compare:
        for split in ('train','validation','test'):
            path=folder/f'{split}.jsonl'
            if path.exists():
                excluded.update(norm(row['prompt']) for row in read(path))
    candidates=[]
    candidate_seen=set()
    for row in read(args.source/'train.jsonl'):
        quality=float(row.get('source_quality') or 0)
        helpfulness=float(row.get('source_helpfulness') or 0)
        if quality<0.75 or helpfulness<0.75:
            continue
        prompt_key=norm(row['prompt'])
        if row['prompt'].count('用户：')<2 or prompt_key in excluded or prompt_key in candidate_seen:
            continue
        if re.search(r'我是(?:一个)?(?:语言模型|人工智能|OpenAssistant)|作为(?:一个)?(?:语言模型|AI)',row['prompt']+row['answer']):
            continue
        candidate_seen.add(prompt_key)
        candidates.append(row)
    candidates.sort(key=lambda row:(-float(row.get('source_quality') or 0),
                                    -float(row.get('source_helpfulness') or 0),
                                    int(row.get('source_rank') or 0),row['id']))
    selected=candidates[:args.count]
    if len(selected)<args.count:
        raise RuntimeError(f'Only {len(selected)} eligible rows, requested {args.count}.')
    args.output.mkdir(parents=True)
    split_counts={'train':max(0,args.count-30),'validation':15,'test':15}
    index=0
    rows=[]
    for split,count in split_counts.items():
        with (args.output/f'{split}.jsonl').open('w',encoding='utf-8') as handle:
            for _ in range(count):
                source=selected[index]
                row=dict(source,training_group='natural_human_oasst2_supplement_v1',
                         source_selection='human_authored_multiturn_quality_supplement',
                         supplement_split=split,review_status='approved_by_source_review',
                         reviewer='OpenAssistant human feedback + local eligibility checks',
                         review_scope='source_review_not_new_independent_local_human_audit',
                         synthetic=False)
                rows.append(row)
                handle.write(json.dumps(row,ensure_ascii=False)+'\n')
                index+=1
    manifest={'kind':'natural_human_multiturn_supplement_v1','source':str(args.source),
              'source_manifest_sha256':checksum(args.source/'manifest.json'),
              'tokenizer_sha256':checksum(BASE/'tokens-chinese-v18-strict2-24k'/'tokenizer.model'),
              'eligibility':{'minimum_quality':0.75,'minimum_helpfulness':0.75,
                             'minimum_user_turns':2,'synthetic_excluded':True,
                             'identity_mismatch_excluded':True},
              'counts':split_counts,'selected_count':len(rows),
              'source_quality_mean':sum(float(r.get('source_quality') or 0) for r in rows)/len(rows),
              'source_helpfulness_mean':sum(float(r.get('source_helpfulness') or 0) for r in rows)/len(rows),
              'excluded_prompt_count':len(excluded),
              'warning':'Human-authored source review and local eligibility checks are retained; this is not a new independent local human audit.'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
