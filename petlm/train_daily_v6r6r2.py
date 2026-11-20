"""Prepare and train a new daily-dialogue adapter from user-confirmed dev rows.

Only development candidates are opened. New internal questions and all old
blind material are excluded from both training and checkpoint selection.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .human_review_pipeline import read_jsonl, write_jsonl
from .run import checksum


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STYLE_TURN = (
    '对话设定：你是虚拟宠物团团，用中文回应，表达风格为温柔自然，先回应感受，不说教。'
    '你没有真实身体，不声称已执行现实动作、写入记忆或设置提醒。',
    '我会按这个风格交流。',
)


def styled_prompt(row):
    return f'用户：{STYLE_TURN[0]}\n助手：{STYLE_TURN[1]}\n' + row['prompt']


def prepare(source, confirmation_root, output, tokenizer, seed=20261108):
    if output.exists():
        raise FileExistsError(f'{output} exists; use a new snapshot path.')
    manifest_path = source / 'manifest.json'
    source_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    confirmation_path = confirmation_root / 'confirmation.json'
    confirmation = json.loads(confirmation_path.read_text(encoding='utf-8'))
    records_path = confirmation_root / 'candidate-confirmations.jsonl'
    if (source_manifest.get('dataset_id') != 'v6r6r2' or
            confirmation.get('dataset_id') != 'v6r6r2' or
            confirmation.get('status') != 'user_confirmed_all_candidates_qualified' or
            confirmation['source_manifest_sha256'] != checksum(manifest_path) or
            confirmation['confirmations_sha256'] != checksum(records_path)):
        raise RuntimeError('Confirmation does not match these candidate files.')
    approvals = read_jsonl(records_path)
    by_id = {row['id']: row for row in approvals}
    if len(by_id) != len(approvals) or len(approvals) != confirmation['count']:
        raise RuntimeError('Duplicate or missing confirmation IDs.')
    # Do not open internal_test-candidates.jsonl, even for splitting.
    development_path = source / 'development-candidates.jsonl'
    if checksum(development_path) != source_manifest['candidate_files']['development']:
        raise RuntimeError('Development candidates changed after user confirmation.')
    development = read_jsonl(development_path)
    if len(development) != source_manifest['counts']['development']:
        raise RuntimeError('Development count mismatch.')
    if len({row['id'] for row in development}) != len(development):
        raise RuntimeError('Duplicate development ID.')
    grouped = defaultdict(list)
    for row in development:
        approval = by_id.get(row['id'], {})
        if (row.get('split') != 'development' or approval.get('split') != 'development' or
                approval.get('decision') != 'user_confirmed_qualified'):
            raise RuntimeError(f'Development row is not covered by user confirmation: {row["id"]}')
        grouped[row['behavior_family']].append(row)
    training, validation = [], []
    for family in sorted(grouped):
        rows = sorted(grouped[family], key=lambda row: hashlib.sha256(
            f'{seed}:{row["id"]}'.encode()).hexdigest())
        if len(rows) < 2:
            raise RuntimeError('Each family needs separate training and dev-validation rows.')
        validation.append(rows[0])
        training.extend(rows[1:])

    def training_row(row):
        return {
            'id': row['id'], 'source_split': 'development',
            'behavior_family': row['behavior_family'],
            'prompt': styled_prompt(row), 'answer': row['proposed_answer'],
            'review_status': 'user_batch_confirmed',
            'approval_method': 'user_package_level_statement',
            'independent_signed_item_review_complete': False,
            'training_group': 'user_confirmed_daily_dialogue_v6r6r2',
        }

    output.mkdir(parents=True)
    write_jsonl(output / 'train.jsonl', [training_row(row) for row in training])
    write_jsonl(output / 'validation.jsonl', [training_row(row) for row in validation])
    manifest = {
        'kind': 'user_confirmed_daily_dialogue_development_only_v1',
        'dataset_id': 'v6r6r2-development-only', 'capability': 'natural_dialogue',
        'single_group_training': True,
        'tokenizer_sha256': checksum(tokenizer),
        'counts': {'train': len(training), 'validation': len(validation)},
        'families': dict(Counter(row['behavior_family'] for row in development)),
        'train_ids': [row['id'] for row in training],
        'validation_ids': [row['id'] for row in validation],
        'split_seed': seed,
        'split_policy': 'One development row per family held out; no internal-test checkpoint selection.',
        'source_split': 'development_only', 'internal_test_used': False,
        'old_blind_data_used': False,
        'source_development_sha256': checksum(development_path),
        'source_confirmation_sha256': checksum(confirmation_path),
        'approval_method': 'user_package_level_statement',
        'independent_signed_item_review_complete': False,
        'training_authorization': 'User explicitly requested 开始训练 after confirming 全部合格.',
        'style_turn': list(STYLE_TURN),
        'files': {name: checksum(output / f'{name}.jsonl') for name in ('train', 'validation')},
        'hardware_policy': 'Real motors, wheels, GPIO/serial actuation, and autonomous navigation remain disabled.',
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


def run(args):
    import sentencepiece as spm
    import torch
    from .dialogue_train import load_pairs
    from .dialogue_lora_train import train

    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this training run.')
    freeze_path = PROJECT_ROOT / 'data/petlm/evaluation/v6r5-candidate-freeze.json'
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    names = ('base_checkpoint', 'tokenizer', 'safety_adapter', 'memory_adapter',
             'motion_adapter', 'format_adapter', 'behavior_state_code', 'router_code', 'chat_code')
    protected = {name: PROJECT_ROOT / freeze['files'][name]['path'] for name in names}
    frozen_before = {name: checksum(path) for name, path in protected.items()}
    if any(frozen_before[name] != freeze['files'][name]['sha256'] for name in names):
        raise RuntimeError('Frozen runtime assets have changed.')
    snapshot = prepare(args.source, args.confirmation, args.data, args.tokenizer, args.seed)
    base = torch.load(args.base, map_location='cpu', weights_only=True)
    if base['tokenizer_sha256'] != snapshot['tokenizer_sha256']:
        raise RuntimeError('Base tokenizer does not match the training snapshot.')
    tokenizer = spm.SentencePieceProcessor(model_proto=args.tokenizer.read_bytes())
    # Validate every pair before allocating the trainable model.
    for split in ('train', 'validation'):
        load_pairs(args.data / f'{split}.jsonl', tokenizer, base['config']['context'])
    del base
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    settings = SimpleNamespace(
        data=args.data, tokenizer=args.tokenizer, base=args.base, output=args.output,
        rank=16, alpha=32.0, dropout=0.05, steps=120, learning_rate=0.0002,
        batch_size=1, accumulation=4, eval_every=10, patience=3,
        natural_fraction=0.0, behavior_validation_weight=1.0, seed=args.seed,
    )
    print(json.dumps({'status': 'starting', 'data_counts': snapshot['counts'],
                      'rank': settings.rank, 'target_steps': settings.steps,
                      'batch_size': settings.batch_size, 'accumulation': settings.accumulation,
                      'base_unchanged': True, 'internal_test_used': False}, ensure_ascii=False), flush=True)
    train(settings)
    frozen_after = {name: checksum(path) for name, path in protected.items()}
    best = torch.load(args.output / 'best.pt', map_location='cpu', weights_only=True)
    training_report = json.loads((args.output / 'report.json').read_text(encoding='utf-8'))
    receipt = {
        'kind': 'daily_dialogue_candidate_training_receipt_v1',
        'status': 'training_complete_candidate_not_accepted',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'completed_steps': training_report['completed_steps'],
        'best_step': best['step'],
        'best_checkpoint_sha256': checksum(args.output / 'best.pt'),
        'latest_checkpoint_sha256': checksum(args.output / 'latest.pt'),
        'snapshot_manifest_sha256': checksum(args.data / 'manifest.json'),
        'approval_method': 'user_package_level_statement',
        'independent_signed_item_review_complete': False,
        'base_checkpoint': str(args.base), 'tokenizer': str(args.tokenizer),
        'candidate_selection': 'Development validation loss only; not internal-test or blind results.',
        'internal_test_used_for_training_or_selection': False,
        'v6r5_blind_data_used': False,
        'frozen_assets_unchanged': frozen_before == frozen_after,
        'frozen_asset_hashes': frozen_after,
        'training_seed': args.seed,
        'application_replaced': False,
        'next_check': 'Read-only regression on new internal questions, with the candidate fixed before generation.',
    }
    (args.output / 'training-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    if not receipt['frozen_assets_unchanged']:
        raise RuntimeError('Frozen asset hash changed during training.')
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('data/petlm/dialogue/human-review-v6r6r2-pending'))
    parser.add_argument('--confirmation', type=Path, default=Path('data/petlm/dialogue/human-review-v6r6r2-user-confirmed'))
    parser.add_argument('--data', type=Path, default=Path('data/petlm/dialogue/prepared-v6r6r2-development-only-v1'))
    parser.add_argument('--base', type=Path, default=Path('data/petlm/models/dialogue-v26-balanced-sft/best.pt'))
    parser.add_argument('--tokenizer', type=Path, default=Path('data/petlm/tokens-chinese-v17-stage7-24k/tokenizer.model'))
    parser.add_argument('--output', type=Path, default=Path('data/petlm/models/dialogue-v6r6r2-daily-r16-v1'))
    parser.add_argument('--seed', type=int, default=20261108)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
