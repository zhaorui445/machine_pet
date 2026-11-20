"""Read-only internal regression of a fixed daily-dialogue adapter candidate.

This is an isolated adapter harness, not a replacement of the web application
or a frozen blind evaluation. The internal set has already been exposed.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from .daily_dialogue_diagnostics import diagnostic_flags, read_package
from .human_review_pipeline import normalized, write_jsonl
from .run import checksum
from .train_daily_v6r6r2 import PROJECT_ROOT, STYLE_TURN


def evaluate(package, candidate, output):
    import sentencepiece as spm
    import torch
    from .behavior_state import BehaviorController
    from .capability_router import route_capability
    from .dialogue_chat import protected_reply, reply
    from .lora import apply_adapter_checkpoint
    from .model import PetLM

    if output.exists():
        raise FileExistsError(output)
    receipt_path = candidate / 'training-receipt.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    checkpoint_path = candidate / 'best.pt'
    candidate_hash = checksum(checkpoint_path)
    if (receipt.get('status') != 'training_complete_candidate_not_accepted' or
            receipt.get('internal_test_used_for_training_or_selection') is not False or
            candidate_hash != receipt['best_checkpoint_sha256']):
        raise RuntimeError('The candidate must be fixed from development-only training first.')
    # Only now open the new internal questions. Nothing is fed back into training.
    _, rows = read_package(package)
    rows = [row for row in rows if row['split'] == 'internal_test']
    freeze_path = PROJECT_ROOT / 'data/petlm/evaluation/v6r5-candidate-freeze.json'
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    frozen_hashes = {}
    for name, expected in receipt['frozen_asset_hashes'].items():
        path = PROJECT_ROOT / freeze['files'][name]['path']
        frozen_hashes[name] = checksum(path)
        if frozen_hashes[name] != expected:
            raise RuntimeError(f'Frozen runtime changed: {name}')
    base_path, tokenizer_path = Path(receipt['base_checkpoint']), Path(receipt['tokenizer'])
    base_hash = checksum(base_path)
    base = torch.load(base_path, map_location='cpu', weights_only=True)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    tokenizer = spm.SentencePieceProcessor(model_proto=tokenizer_path.read_bytes())
    if state['tokenizer_sha256'] != checksum(tokenizer_path):
        raise RuntimeError('Candidate tokenizer mismatch.')
    model = PetLM(base['config'])
    model.load_state_dict(base['model'])
    apply_adapter_checkpoint(model, state, base_hash)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    del base, state
    output.mkdir(parents=True)
    records = []
    decoding = dict(freeze['decoding'], seed_policy='frozen seed plus case index')
    for index, row in enumerate(rows):
        seed = decoding['seed'] + index
        torch.manual_seed(seed)
        history = [STYLE_TURN] + [(turn['user'], turn['assistant']) for turn in row['history']]
        generated = reply(model, tokenizer, history, row['user'],
                          max_new_tokens=decoding['max_new_tokens'], temperature=decoding['temperature'],
                          use_protection=False)
        controller = BehaviorController()
        protected = controller.handle(history, row['user'])
        source = controller.last_source
        if protected is None:
            protected = protected_reply(history, row['user'])
            source = 'legacy_protected_reply' if protected is not None else 'daily_adapter_generation'
        if protected is None:
            protected = generated
        record = {
            'id': row['id'], 'scenario_key': row['scenario_key'], 'history': row['history'],
            'user': row['user'], 'reference': row['proposed_answer'],
            'semantic_review_checks': row['review_checks'],
            'seed': seed, 'model_answer': generated, 'protected_answer': protected,
            'protected_source': source,
            'unchanged_web_router_capability': route_capability(history, row['user']),
            'model_diagnostic': diagnostic_flags(row, generated),
            'protected_diagnostic': diagnostic_flags(row, protected),
            'human_score': None,
        }
        records.append(record)
        print(json.dumps({'progress': f'{index + 1}/{len(rows)}', 'id': row['id'],
                          'flags': record['model_diagnostic']['flags']}, ensure_ascii=False), flush=True)
    write_jsonl(output / 'outputs.jsonl', records)
    source_counts = Counter(row['protected_source'] for row in records)
    repeated = Counter(normalized(row['model_answer']) for row in records)
    after = {name: checksum(PROJECT_ROOT / freeze['files'][name]['path']) for name in frozen_hashes}
    report = {
        'kind': 'daily_adapter_internal_regression_v1',
        'status': 'internal_regression_complete_not_acceptance',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'candidate': str(checkpoint_path), 'candidate_sha256': candidate_hash,
        'selected_step': receipt['best_step'], 'completed_training_steps': receipt['completed_steps'],
        'count': len(records),
        'model_flagged_count': sum(bool(row['model_diagnostic']['flags']) for row in records),
        'protected_flagged_count': sum(bool(row['protected_diagnostic']['flags']) for row in records),
        'protected_response_sources': dict(source_counts),
        'model_repeated_response_group_sizes': [count for count in repeated.values() if count > 1],
        'semantic_pass_rate': None, 'human_scoring_complete': False,
        'candidate_unchanged': candidate_hash == checksum(checkpoint_path),
        'frozen_assets_unchanged': after == frozen_hashes,
        'internal_test_used_for_training_or_checkpoint_selection': False,
        'v6r5_blind_data_used': False, 'decoding': decoding,
        'application_replaced': False,
        'harness': 'Standalone daily adapter, with pure generation and unchanged protection reported separately; existing web router is not replaced.',
        'outputs_sha256': checksum(output / 'outputs.jsonl'),
        'limits': [
            'The internal set is a known diagnostic set, not unseen final acceptance.',
            'Topic-signal flags are not semantic scores; all answers still need human quality review.',
            'Loss improvement or fewer flags does not establish safe, fluent conversation.',
        ],
    }
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    review = ['# v6r6r2 新适配器：实际输出逐条核对\n',
              '这是内部回归结果，不是人工评分或最终验收；纯模型回复和保护后回复分别列出。\n']
    for row in records:
        review.append(f'## {row["id"]}\n')
        for turn in row['history']:
            review.extend([f'历史用户：{turn["user"]}\n', f'历史助手：{turn["assistant"]}\n'])
        review.extend([f'当前问题：{row["user"]}\n', f'纯模型回复：{row["model_answer"]}\n',
                       f'保护后回复：{row["protected_answer"]}\n',
                       f'参考表达：{row["reference"]}\n',
                       '核对重点：' + '；'.join(row['semantic_review_checks']) + '。\n'])
    (output / 'OUTPUT_REVIEW.md').write_text('\n'.join(review), encoding='utf-8')
    if not report['candidate_unchanged'] or not report['frozen_assets_unchanged']:
        raise RuntimeError('Candidate or frozen runtime asset changed during regression.')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, default=Path('data/petlm/dialogue/human-review-v6r6r2-pending'))
    parser.add_argument('--candidate', type=Path, default=Path('data/petlm/models/dialogue-v6r6r2-daily-r16-v1'))
    parser.add_argument('--output', type=Path, default=Path('data/petlm/evaluation/v6r6r2-trained-internal-regression-v1'))
    args = parser.parse_args()
    print(json.dumps(evaluate(args.package, args.candidate, args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
