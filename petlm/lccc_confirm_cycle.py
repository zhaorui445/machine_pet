"""Materialize user-confirmed assistant-triage rows without claiming item review."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import sentencepiece as spm

from .run import checksum


TOKENIZER = Path('data/petlm/tokens-chinese-v8-50m-frozen/tokenizer.model')
REVISION = '5bd582fa28cd7143f2f9c852e08e23089d677c44'
LICENSE = 'MIT (dataset mirror; upstream privacy provenance incomplete)'


def rows(path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha_pair(row):
    value = row['prompt'] + '\0' + row['answer']
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def materialize(source_root, output_root, tokenizer_path=TOKENIZER, context=512):
    if output_root.exists():
        raise FileExistsError(f'{output_root} exists; choose a new immutable cycle.')
    tokenizer = spm.SentencePieceProcessor(model_proto=tokenizer_path.read_bytes())
    selected = {}
    rejected = Counter()
    for split in ('train', 'development', 'internal_test'):
        source = source_root / f'{split}-assistant-audit.jsonl'
        kept = []
        for row in rows(source):
            if row.get('assistant_preliminary_decision') != 'preliminary_clear':
                rejected[f'{split}/not_preliminary_clear'] += 1
                continue
            if len(tokenizer.encode(row['prompt'])) + len(tokenizer.encode(row['answer'])) + 1 > context:
                rejected[f'{split}/context_overflow'] += 1
                continue
            kept.append({
                'id': row['id'], 'conversation_id': row['conversation_id'],
                'prompt': row['prompt'], 'answer': row['answer'],
                'training_group': 'user_confirmed_assistant_triage_lccc_v1',
                'source': 'silver/lccc', 'source_revision': REVISION,
                'source_license': LICENSE,
                'selection_basis': 'user_confirmed_assistant_preliminary_clear',
                'human_item_reviewed': False,
            })
        selected[split] = kept

    # Internal test remains evaluation-only. Ensure no prompt/pair leakage among
    # materialized train and development rows, with holdout priority.
    def key(value):
        return hashlib.sha256(''.join(value.split()).casefold().encode('utf-8')).hexdigest()
    test_prompts = {key(row['prompt']) for row in selected['internal_test']}
    test_pairs = {sha_pair(row) for row in selected['internal_test']}
    dev_prompts = set(test_prompts)
    dev_pairs = set(test_pairs)
    development = []
    for row in selected['development']:
        if key(row['prompt']) in dev_prompts or sha_pair(row) in dev_pairs:
            rejected['development/holdout_overlap'] += 1
            continue
        dev_prompts.add(key(row['prompt']))
        dev_pairs.add(sha_pair(row))
        development.append(row)
    train = []
    train_prompts = set(dev_prompts)
    train_pairs = set(dev_pairs)
    for row in selected['train']:
        if key(row['prompt']) in train_prompts or sha_pair(row) in train_pairs:
            rejected['train/holdout_overlap'] += 1
            continue
        train_prompts.add(key(row['prompt']))
        train_pairs.add(sha_pair(row))
        train.append(row)
    selected['train'], selected['development'] = train, development

    output_root.mkdir(parents=True)
    for split in ('train', 'development', 'internal_test'):
        path = output_root / f'{split}.jsonl'
        with path.open('w', encoding='utf-8', newline='\n') as handle:
            for row in selected[split]:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest = {
        'kind': 'lccc_user_confirmed_triage_sft_cycle_v1',
        'source': 'silver/lccc', 'source_revision': REVISION,
        'source_license': LICENSE,
        'source_audit_sha256': checksum(source_root / 'report.json'),
        'user_confirmation': 'User confirmed assistant preliminary-clear rows; this is not independent item-level human review.',
        'assistant_audit_only': True, 'human_item_reviewed': False,
        'tokenizer_sha256': checksum(tokenizer_path), 'context_tokens': context,
        'official_test_opened': False, 'official_test_used': False,
        'counts': {split: len(selected[split]) for split in selected},
        'rejected_after_confirmation': dict(rejected),
        'files': {split: checksum(output_root / f'{split}.jsonl') for split in selected},
        'training_policy': 'Only train rows may enter exploratory SFT; development and internal_test are evaluation-only.',
        'limitations': [
            'User confirmation of assistant triage is not an independent item-level audit.',
            'Automated privacy/safety rules may miss personal or harmful content.',
            'LCCC social-media dialogue may contain dialect, slang, context gaps, or factual errors.',
        ],
    }
    (output_root / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    status = {
        'status': 'ready_for_exploratory_sft_user_confirmed_triage',
        'counts': manifest['counts'], 'sft_started': False,
        'human_item_reviewed': False, 'official_test_used': False,
        'next_action': 'Run low-learning-rate exploratory SFT and compare against the frozen parent; do not call this independent human approval.',
    }
    (output_root / 'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'counts': manifest['counts'], 'rejected': dict(rejected), 'files': manifest['files']}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, default=TOKENIZER)
    args = parser.parse_args()
    materialize(args.source_root, args.output_root, args.tokenizer)


if __name__ == '__main__':
    main()
