"""Compare a completed foundation run with its parent; never starts training."""
import argparse
import contextlib
from datetime import datetime, timezone
import gc
import io
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .evaluate import evaluate
from .run import checksum, generate


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def completion_check(plan, output):
    report_path = output / 'report.json'
    if not report_path.is_file():
        raise RuntimeError('Training completion report is absent; evaluation must wait.')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if (report.get('status') != 'schedule_complete'
            or report.get('completed_steps') != plan['planned_steps']
            or report.get('target_steps') != plan['planned_steps']):
        raise RuntimeError('Training did not finish the planned schedule; evaluation must wait.')
    return report


def run(plan_path, cases_path, destination, preflight_only=False):
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    output = Path(plan['output'])
    report = completion_check(plan, output)
    tokens = Path(plan['tokens'])
    parent = Path(plan['parent_checkpoint'])
    candidate = output / 'latest.pt'
    config = Path(plan['config'])
    if checksum(config) != plan['config_sha256']:
        raise RuntimeError('Training configuration differs from the recorded plan.')
    if checksum(tokens / 'manifest.json') != plan['token_manifest_sha256']:
        raise RuntimeError('Token manifest differs from the recorded plan.')
    if checksum(parent) != plan['parent_checkpoint_sha256']:
        raise RuntimeError('Parent checkpoint differs from the recorded plan.')
    manifest = json.loads((tokens / 'manifest.json').read_text(encoding='utf-8'))
    if checksum(tokens / 'tokenizer.model') != manifest['tokenizer_sha256']:
        raise RuntimeError('Tokenizer content changed.')
    candidate_hash = checksum(candidate)
    state = torch.load(candidate, map_location='cpu', weights_only=True)
    if (state['step'] != plan['planned_steps']
            or state['dataset_hash'] != plan['token_manifest_sha256']
            or state['tokenizer_sha256'] != manifest['tokenizer_sha256']
            or state['config'] != json.loads(config.read_text(encoding='utf-8'))):
        raise RuntimeError('Candidate checkpoint does not match the completed run.')
    if not all(torch.isfinite(value).all().item() for value in state['model'].values()):
        raise RuntimeError('Candidate contains non-finite weights.')
    del state
    gc.collect()
    if preflight_only:
        return {'status': 'completed_checkpoint_verified'}
    if destination.exists():
        raise FileExistsError('Evaluation output exists; preserve the previous result.')
    destination.mkdir(parents=True)
    cases = json.loads(cases_path.read_text(encoding='utf-8'))
    results = {'kind': 'foundation_development_comparison_v1',
               'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'status': 'running', 'training_steps': report['completed_steps'],
               'checkpoint_sha256': {'parent': checksum(parent), 'candidate': candidate_hash},
               'case_sha256': checksum(cases_path), 'decoding': cases['decoding'],
               'evaluation_scope': 'Development inspection only; not an independent frozen blind test.',
               'human_review_status': 'pending', 'accepted_for_deployment': False,
               'validation': {}, 'generation': {}}
    result_path = destination / 'results.json'
    write_json(result_path, results)
    for label, checkpoint in (('parent', parent), ('candidate', candidate)):
        print(f'Validating {label} on the same 100 validation windows.', flush=True)
        results['validation'][label] = evaluate(checkpoint, tokens, 'validation', 100)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        results['generation'][label] = []
        for case in cases['cases']:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                generate(SimpleNamespace(checkpoint=checkpoint, tokens=tokens,
                                         prompt=case['prompt'], new_tokens=80, seed=42))
            # Preserve the complete raw text, including prompts and empty continuations.
            results['generation'][label].append(dict(case, raw_generation=buffer.getvalue().rstrip('\r\n')))
            write_json(result_path, results)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f'{label}: generated {case["id"]}', flush=True)
    if checksum(candidate) != candidate_hash:
        raise RuntimeError('Candidate changed during evaluation; discard this comparison.')
    results['validation_loss_delta'] = (results['validation']['candidate']['loss']
                                        - results['validation']['parent']['loss'])
    results['status'] = 'complete_awaiting_human_generation_review'
    results['next_step'] = ('Inspect naturalness, relevance, repetition and topic switching in raw outputs; '
                            'choose further pretraining or separately reviewed SFT data. No automatic SFT.')
    write_json(result_path, results)
    lines = ['# v26r1 基础训练后开发检查', '',
             '这是开发检查，不是冻结盲测；自动计算的损失不代表对话验收通过。', '',
             f'父模型验证损失：{results["validation"]["parent"]["loss"]:.6f}',
             f'本轮模型验证损失：{results["validation"]["candidate"]["loss"]:.6f}', '',
             '请检查语句自然度、上下文相关性、重复和话题切换。以下保留原始生成结果。', '']
    for index, case in enumerate(cases['cases']):
        lines += [f'## {case["id"]} · {case["category"]}', '']
        for label, title in (('parent', '父模型'), ('candidate', '本轮模型')):
            lines += [f'### {title}', '', '    ' + results['generation'][label][index]['raw_generation'].replace('\n', '\n    '), '']
    (destination / 'review.md').write_text('\n'.join(lines), encoding='utf-8')
    plan.update(status='training_complete_development_review_pending',
                observed_at_utc=datetime.now(timezone.utc).isoformat(),
                completed_steps=report['completed_steps'], posttraining_report=str(result_path))
    write_json(plan_path, plan)
    status_path = plan_path.parent / 'status.json'
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding='utf-8'))
        active = status.get('active_training_run')
        if active and Path(active).resolve() == plan_path.resolve():
            status.update(status='base_training_complete_development_review_pending',
                          posttraining_report=str(result_path), sft_started=False)
            write_json(status_path, status)
    print(json.dumps({'status': results['status'], 'output': str(destination)}, ensure_ascii=False), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    try:
        run(args.plan, args.cases, args.output, args.preflight_only)
    except Exception as exc:
        error = {'status': 'posttraining_check_failed', 'error': f'{type(exc).__name__}: {exc}',
                 'observed_at_utc': datetime.now(timezone.utc).isoformat()}
        if not args.preflight_only:
            write_json(args.plan.with_name('v26r1-posttraining-error.json'), error)
        raise


if __name__ == '__main__':
    main()
