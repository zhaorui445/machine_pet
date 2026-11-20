"""Two-phase formal evaluation for the independently reviewed human v5 set."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import sentencepiece as spm
import torch

from .behavior_state import BehaviorController
from .capability_router import CAPABILITIES, CapabilityRouter
from .dialogue_chat import FALLBACK_REPLY, protected_reply, reply
from .run import checksum


def read_jsonl(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def load_prompts(path):
    rows = read_jsonl(path)
    seen = set()
    for row in rows:
        if any(key in row for key in ('answer', 'proposed_answer', 'review_checks', 'rubric')):
            raise ValueError('Frozen public prompts contain answer or scoring material.')
        if row.get('id') in seen or not row.get('user') or not isinstance(row.get('history'), list):
            raise ValueError(f'Invalid or duplicate public prompt: {row.get("id")}')
        seen.add(row['id'])
    if not rows:
        raise RuntimeError('Frozen prompt set is empty.')
    return rows


def adapter_paths(root, explicit):
    if explicit:
        paths = {name: Path(value) for name, value in zip(CAPABILITIES, explicit)}
    else:
        paths = {name: root / f'dialogue-v5-lora-human-{suffix}-r8' / 'best.pt'
                 for name, suffix in zip(CAPABILITIES, ('safety', 'memory', 'motion', 'format'))}
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError('One or more v5 capability adapter checkpoints are missing.')
    return paths


def generate(prompts_path, review_root, checkpoint, tokenizer_path, adapters_root,
             output, seed, temperature, max_new_tokens, explicit_adapters=None):
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable output directory.')
    manifest_path = review_root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    prompts = load_prompts(prompts_path)
    if (manifest.get('review_status') != 'independent_review_complete'
            or manifest.get('counts', {}).get('frozen') != len(prompts)):
        raise RuntimeError('The reviewed frozen prompt set is incomplete or has the wrong size.')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if checksum(tokenizer_path) != state['tokenizer_sha256']:
        raise RuntimeError('Tokenizer does not match the base checkpoint.')
    paths = adapter_paths(adapters_root, explicit_adapters)
    processor = spm.SentencePieceProcessor(model_proto=tokenizer_path.read_bytes())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    router = CapabilityRouter(checkpoint, adapters_root, device=device,
                              adapter_paths={name: paths[name] for name in CAPABILITIES})
    results = {'model': [], 'application': []}
    route_errors = []
    for index, item in enumerate(prompts):
        history = [(turn['user'], turn['assistant']) for turn in item['history']]
        capability, model = router.selected_model(history, item['user'])
        if capability != item.get('capability'):
            route_errors.append({'id': item['id'], 'expected': item.get('capability'), 'routed': capability})
        if model is None:
            model = router.model_for(CAPABILITIES[0])
            model = model.base_model if hasattr(model, 'base_model') else model
        for offset, mode in enumerate(('model', 'application')):
            item_seed = seed + index * 2 + offset
            torch.manual_seed(item_seed)
            if device == 'cuda':
                torch.cuda.manual_seed_all(item_seed)
            protected = mode == 'application'
            controller = BehaviorController() if protected else None
            answer = reply(model, processor, history, item['user'], max_new_tokens, temperature,
                           use_protection=protected, controller=controller)
            if controller is not None and controller.last_source:
                source = controller.last_source
            elif protected and protected_reply(history, item['user']) is not None:
                source = 'deterministic_rule'
            elif answer == FALLBACK_REPLY:
                source = 'generation_fallback'
            else:
                source = 'model_generation'
            results[mode].append({'id': item['id'], 'capability': item.get('capability'),
                                 'routed_capability': capability, 'route_correct': capability == item.get('capability'),
                                 'answer': answer, 'response_source': source, 'seed': item_seed})
    output.mkdir(parents=True)
    outputs = {}
    for mode, rows in results.items():
        path = output / f'{mode}-outputs.jsonl'
        write_jsonl(path, rows)
        outputs[mode] = {'path': str(path.resolve()), 'count': len(rows), 'sha256': checksum(path)}
    report = {
        'kind': 'petlm_human_reviewed_v5_keyless_generation_v1',
        'phase': 'outputs_frozen_before_answer_key_access',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'dataset_manifest': str(manifest_path.resolve()),
        'dataset_manifest_sha256': checksum(manifest_path),
        'prompts': str(prompts_path.resolve()), 'prompts_sha256': checksum(prompts_path),
        'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': checksum(checkpoint),
        'tokenizer_sha256': checksum(tokenizer_path),
        'adapters': {name: {'path': str(path.resolve()), 'sha256': checksum(path)} for name, path in paths.items()},
        'decoding': {'seed': seed, 'temperature': temperature, 'max_new_tokens': max_new_tokens},
        'count': len(prompts), 'route_correct': len(prompts) - len(route_errors),
        'route_errors': route_errors, 'outputs': outputs,
        'isolation': 'Generation read only keyless frozen prompts; no answer key or review decisions were loaded.'
    }
    (output / 'generation-manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def score_mode(predictions, keys):
    by_id = {row['id']: row for row in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {row['id'] for row in keys}:
        raise RuntimeError('Predictions and frozen answer key do not cover identical unique ids.')
    families = defaultdict(Counter)
    capabilities = defaultdict(Counter)
    sources = Counter()
    failures = []
    passed = 0
    for key in keys:
        prediction = by_id[key['id']]
        answer = ''.join(prediction['answer'].split())
        checks = [str(check) in answer for check in key.get('review_checks', [])]
        ok = all(checks)
        for groups, label in ((families, key['behavior_family']), (capabilities, key['capability'])):
            groups[label]['total'] += 1
            groups[label]['passed'] += int(ok)
        sources[prediction.get('response_source', 'missing')] += 1
        passed += int(ok)
        if not ok:
            failures.append({'id': key['id'], 'behavior_family': key['behavior_family'],
                             'capability': key['capability'], 'answer': prediction['answer'],
                             'review_checks': key.get('review_checks', []), 'checks_passed': checks})

    def summarize(groups):
        return {name: {'passed': row['passed'], 'total': row['total'],
                       'pass_rate': row['passed'] / row['total']} for name, row in sorted(groups.items())}

    return {'passed': passed, 'total': len(keys), 'pass_rate': passed / len(keys),
            'response_sources': dict(sources), 'by_capability': summarize(capabilities),
            'by_behavior_family': summarize(families), 'failures': failures}


def score(generation_dir, review_root, output):
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable score directory.')
    gen_path = generation_dir / 'generation-manifest.json'
    gen = json.loads(gen_path.read_text(encoding='utf-8'))
    if gen.get('phase') != 'outputs_frozen_before_answer_key_access':
        raise RuntimeError('Keyless output freezing is not established.')
    manifest_path = review_root / 'manifest.json'
    review = json.loads(manifest_path.read_text(encoding='utf-8'))
    decision_path = review_root / 'review-decisions.jsonl'
    key_path = review_root / 'frozen' / 'answer-key.reviewed.jsonl'
    if (review.get('review_status') != 'independent_review_complete'
            or review.get('approved_count') != review.get('counts', {}).get('development', 0)
                + review.get('counts', {}).get('internal_test', 0)
                + review.get('counts', {}).get('frozen', 0)
            or checksum(decision_path) != review.get('review_decisions_sha256')):
        raise RuntimeError('Reviewer approval evidence is incomplete or does not match its manifest.')
    decisions = read_jsonl(decision_path)
    frozen_ids = {row['id'] for row in read_jsonl(Path(gen['prompts']))}
    frozen_decisions = [row for row in decisions if row.get('id') in frozen_ids]
    if (len(frozen_decisions) != len(frozen_ids)
            or any(row.get('decision') != 'approved' or row.get('approve_for_split') != 'frozen'
                   for row in frozen_decisions)):
        raise RuntimeError('Frozen prompts do not have one approved decision each.')
    keys = read_jsonl(key_path)
    if {row['id'] for row in keys} != frozen_ids:
        raise RuntimeError('Reviewed answer key does not exactly match the frozen prompts.')
    for item in gen['outputs'].values():
        path = Path(item['path'])
        if checksum(path) != item['sha256'] or len(read_jsonl(path)) != item['count']:
            raise RuntimeError('Frozen model outputs failed hash/count verification.')
    reports = {mode: score_mode(read_jsonl(Path(item['path'])), keys)
               for mode, item in gen['outputs'].items()}
    output.mkdir(parents=True)
    for mode, report in reports.items():
        (output / f'{mode}-score.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {'kind': 'petlm_human_reviewed_v5_formal_evaluation_v1',
               'created_at': datetime.now(timezone.utc).isoformat(),
               'generation_manifest': str(gen_path.resolve()),
               'generation_manifest_sha256': checksum(gen_path),
               'review_manifest': str(manifest_path.resolve()),
               'review_manifest_sha256': checksum(manifest_path),
               'answer_key': str(key_path.resolve()), 'answer_key_sha256': checksum(key_path),
               'modes': {mode: {k: v for k, v in report.items() if k != 'failures'}
                         for mode, report in reports.items()},
               'isolation_verified': True,
               'note': 'Deterministic review-check scoring is a minimum-element check, not a complete semantic or human quality rating.'}
    (output / 'report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    gen = sub.add_parser('generate')
    gen.add_argument('--prompts', type=Path, required=True)
    gen.add_argument('--review-root', type=Path, required=True)
    gen.add_argument('--checkpoint', type=Path, required=True)
    gen.add_argument('--tokenizer', type=Path, required=True)
    gen.add_argument('--adapters-root', type=Path, required=True)
    gen.add_argument('--output', type=Path, required=True)
    gen.add_argument('--seed', type=int, default=20261215)
    gen.add_argument('--temperature', type=float, default=0.3)
    gen.add_argument('--max-new-tokens', type=int, default=80)
    for option, _ in zip(('safety-adapter', 'memory-adapter', 'motion-adapter', 'format-adapter'), CAPABILITIES):
        gen.add_argument(f'--{option}', type=Path)
    evaluate = sub.add_parser('score')
    evaluate.add_argument('--generation', type=Path, required=True)
    evaluate.add_argument('--review-root', type=Path, required=True)
    evaluate.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'generate':
        values = [getattr(args, f'{name.replace("-", "_")}_adapter')
                  for name in ('safety', 'memory', 'motion', 'format')]
        if any(values) and not all(values):
            parser.error('Supply all four capability adapter paths together.')
        result = generate(args.prompts, args.review_root, args.checkpoint, args.tokenizer,
                          args.adapters_root, args.output, args.seed, args.temperature,
                          args.max_new_tokens, values if all(values) else None)
    else:
        result = score(args.generation, args.review_root, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
