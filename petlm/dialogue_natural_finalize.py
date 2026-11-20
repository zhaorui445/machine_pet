"""Build a replacement dialogue snapshot after natural-dialogue review."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .data import BASE
from .run import checksum


def read_jsonl(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row,ensure_ascii=False)+'\n')


def keep_base_row(row, split):
    if split=='test':
        return True
    return (row.get('training_group')=='diverse_synthetic'
            or (row.get('review_status')=='approved'
                and str(row.get('tree_id','')).startswith('manual:')))


def require_reviewed(rows, label):
    if not rows:
        raise RuntimeError(f'{label} is empty.')
    invalid=[row.get('id') for row in rows
             if row.get('review_status') not in ('approved','rejected')
             or not row.get('reviewer') or not row.get('reviewed_at')]
    if invalid:
        raise RuntimeError(f'{label} contains pending or incomplete reviews: {invalid[0]!r}')


def finalize(base, reviewed_candidates, blind_prompts, reviewed_keys,
             output, blind_output, tokenizer_path):
    if output.exists() or blind_output.exists():
        raise FileExistsError('Output snapshot exists; choose new immutable paths.')
    base_manifest=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    if checksum(tokenizer_path)!=base_manifest['tokenizer_sha256']:
        raise RuntimeError('Tokenizer differs from the base dialogue snapshot.')
    candidates=read_jsonl(reviewed_candidates)
    keys=read_jsonl(reviewed_keys)
    prompts=read_jsonl(blind_prompts)
    require_reviewed(candidates,'candidate reviews')
    require_reviewed(keys,'blind answer reviews')

    rows=[]
    removed=Counter()
    for split in ('train','validation','test'):
        for row in read_jsonl(base/f'{split}.jsonl'):
            if keep_base_row(row,split):
                rows.append(dict(row,split=split))
            else:
                removed['unreviewed_legacy']+=1
    seen={(row['prompt'],row['answer']) for row in rows}
    added=Counter()
    review_methods=Counter()
    review_scopes=Counter()
    for row in candidates:
        if row['review_status']!='approved':
            continue
        pair=(row['prompt'],row['answer'])
        if pair in seen:
            raise RuntimeError(f'Approved natural pair duplicates retained data: {row["id"]!r}')
        seen.add(pair)
        split=row['split_hint']
        method=row.get('review_method','item_level')
        rendered=dict(row,split=split,
                      training_group='natural_human_'+method)
        rows.append(rendered)
        added[split]+=1
        review_methods[method]+=1
        review_scopes[row.get('review_scope','unspecified')]+=1
    output.mkdir(parents=True)
    split_counts=Counter(row['split'] for row in rows)
    groups=Counter(row.get('training_group','legacy') for row in rows if row['split']=='train')
    for split in ('train','validation','test'):
        write_jsonl(output/f'{split}.jsonl',(row for row in rows if row['split']==split))

    prompt_by_id={row['id']:row for row in prompts}
    key_by_id={row['id']:row for row in keys if row['review_status']=='approved'}
    if set(key_by_id)-set(prompt_by_id):
        raise RuntimeError('Reviewed blind key contains an unknown prompt id.')
    final_prompts=[prompt_by_id[identifier] for identifier in prompt_by_id if identifier in key_by_id]
    final_keys=[key_by_id[row['id']] for row in final_prompts]
    if not final_prompts:
        raise RuntimeError('No approved blind prompts remain.')
    blind_output.mkdir(parents=True)
    prompt_output=blind_output/'prompts.jsonl'
    key_output=blind_output/'answer-key.reviewed.jsonl'
    write_jsonl(prompt_output,final_prompts)
    write_jsonl(key_output,final_keys)

    report={'base_snapshot':str(base.resolve()),
        'base_manifest_sha256':checksum(base/'manifest.json'),
        'tokenizer_sha256':checksum(tokenizer_path),'counts':dict(split_counts),
        'training_groups':dict(groups),'removed':dict(removed),
        'approved_natural_added':dict(added),'review_methods':dict(review_methods),
        'review_scopes':dict(review_scopes),
        'reviewed_candidates':str(reviewed_candidates),
        'reviewed_candidates_sha256':checksum(reviewed_candidates),
        'test_policy':'The existing 56-pair test split is preserved unchanged and never trained.',
        'warning':'Review provenance is retained exactly; automated review is not human review.'}
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    blind_report={'count':len(final_prompts),'prompts':str(prompt_output),
        'prompts_sha256':checksum(prompt_output),'answer_key':str(key_output),
        'answer_key_sha256':checksum(key_output),
        'source_prompts_sha256':checksum(blind_prompts),
        'source_reviewed_keys_sha256':checksum(reviewed_keys),
        'review_methods':dict(Counter(row.get('review_method','item_level') for row in final_keys)),
        'review_scopes':dict(Counter(row.get('review_scope','unspecified') for row in final_keys)),
        'isolation':'Blind prompts are disjoint from training candidates by OpenAssistant tree id.',
        'warning':'Reference review provenance is retained exactly; automated review is not human review.'}
    (blind_output/'manifest.json').write_text(json.dumps(blind_report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'training':report,'blind':blind_report},ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,default=BASE/'dialogue'/'prepared-pet-diverse-v5')
    parser.add_argument('--reviewed-candidates',type=Path,required=True)
    parser.add_argument('--blind-prompts',type=Path,required=True)
    parser.add_argument('--reviewed-keys',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--blind-output',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path,default=BASE/'tokens-chinese-v7'/'tokenizer.model')
    args=parser.parse_args()
    finalize(args.base,args.reviewed_candidates,args.blind_prompts,args.reviewed_keys,
             args.output,args.blind_output,args.tokenizer)


if __name__=='__main__':
    main()
