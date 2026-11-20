"""Audit pending daily-dialogue data and run a labelled, non-acceptance baseline.

No training is performed. No old blind prompts, keys or scores are loaded.
The new internal questions are visible to the author and therefore constitute
an internal diagnostic set, not an independent human-authored frozen blind set.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from .human_review_pipeline import normalized, read_jsonl, similarity, write_jsonl
from .run import checksum


def read_package(package):
    package = Path(package)
    manifest = json.loads((package / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('dataset_id') != 'v6r6r2':
        raise ValueError('This diagnostic accepts only the new v6r6r2 cycle.')
    rows = []
    for split in ('development', 'internal_test'):
        path = package / f'{split}-candidates.jsonl'
        if checksum(path) != manifest['candidate_files'][split]:
            raise RuntimeError(f'Candidate hash changed: {split}')
        selected = read_jsonl(path)
        if any(row.get('split') != split for row in selected):
            raise RuntimeError(f'Split mismatch: {split}')
        rows.extend(selected)
    return manifest, rows


def diagnostic_flags(row, answer):
    """Weak lexical hints ONLY; absence of flags never means semantic success."""
    flags = []
    if not answer.strip():
        flags.append('empty_response')
    missing = [group for group in row['diagnostic_signal_groups']
               if not any(word in answer for word in group)]
    if missing:
        flags.append('missing_expected_topic_signals')
    if row['behavior_family'] not in {'hazard_observation', 'resolved_hazard_switch'}:
        if any(term in answer for term in ('论文', '检索', '不能核实确切出处')):
            flags.append('unrelated_source_search')
        if any(term in answer for term in ('清洁液', '清洁剂', '电池', '未知危险', '不能提供测试')):
            flags.append('unrelated_hazard_response')
    if '我还没理解好这个问题' in answer:
        flags.append('fallback_response')
    if any(normalized(turn['assistant']) == normalized(answer) for turn in row['history']):
        flags.append('verbatim_history_repetition')
    return {'flags': flags, 'missing_signal_groups': missing,
            'human_semantic_review_required': True, 'semantic_pass': None}


def audit(package, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest, rows = read_package(package)
    development = [row for row in rows if row['split'] == 'development']
    internal = [row for row in rows if row['split'] == 'internal_test']
    findings = []
    for row in rows:
        if not row['review_checks'] or not row['proposed_answer'].strip():
            findings.append({'id': row['id'], 'kind': 'empty_reference_or_rubric'})
        result = diagnostic_flags(row, row['proposed_answer'])
        if result['flags']:
            findings.append({'id': row['id'], 'kind': 'reference_diagnostic_conflict', **result})
        if 'question_mark_count' in row or 'requirements' in row:
            findings.append({'id': row['id'], 'kind': 'unexpected_legacy_literal_scoring'})
    comparisons = []
    for left in development:
        for right in internal:
            same_user = normalized(left['user']) == normalized(right['user'])
            same_reference = normalized(left['proposed_answer']) == normalized(right['proposed_answer'])
            score = similarity(left['user'], right['user'])
            if same_user or same_reference:
                findings.append({'kind': 'cross_split_exact_question_or_answer',
                                 'development': left['id'], 'internal_test': right['id']})
            if score >= 0.72:
                comparisons.append({'development': left['id'], 'internal_test': right['id'],
                                    'current_question_similarity': round(score, 4)})
    # Directory leakage scanning in the shared legacy utility only discovers
    # finalized names such as development.jsonl. Include the rejected pending
    # package's actual candidate filenames explicitly here.
    external = [Path(path) for path in manifest['external_paths']]
    for path in tuple(external):
        if path.is_dir():
            external.extend(p for p in path.glob('*-candidates.jsonl'))
    from .human_review_pipeline import leakage_report
    leakage = leakage_report(rows, external, threshold=0.92)
    if not leakage['passed']:
        findings.extend(leakage['findings'])
    report = {
        'kind': 'daily_dialogue_candidate_audit_v1',
        'dataset_id': manifest['dataset_id'],
        'status': 'automatic_checks_passed_pending_human_review' if not findings else 'revision_required',
        'counts': manifest['counts'],
        'findings': findings,
        'historical_nonblind_leakage_check': leakage,
        'cross_split_pairs_needing_similarity_review': comparisons,
        'mechanical_checks_passed': not findings,
        'semantic_independence_certified': False,
        'human_approval_granted': False,
        'training_started': False,
        'limits': [
            'Different questions and answers do not prove semantic independence.',
            'The same AI authored both splits; an independent human must review them.',
            'Topic signals are diagnostic hints, never final semantic scores.',
            'The candidate internal set becomes a diagnostic set if its results guide changes; create a new holdout for final acceptance.',
        ],
    }
    output.mkdir(parents=True)
    (output / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for split in ('development', 'internal_test'):
        view = [f'# v6r6r2 {split} 候选逐条审核\n',
                '这些是 AI 编写、待人工复核的候选。不要批量默认通过；参考答案仅为一种可接受表达。\n',
                '审核：是否直接切题、遵守用户约束、自然完整、不会编造状态或给出危险建议。问号数量不作为统一门槛。\n']
        for row in (r for r in rows if r['split'] == split):
            view.append(f'## {row["id"]}\n')
            for turn in row['history']:
                view.extend([f'历史用户：{turn["user"]}\n', f'历史助手：{turn["assistant"]}\n'])
            view.extend([f'当前用户：{row["user"]}\n', f'候选回答：{row["proposed_answer"]}\n',
                         '审核重点：' + '；'.join(row['review_checks']) + '。\n'])
        (output / f'{split}-review.md').write_text('\n'.join(view), encoding='utf-8')
    return report


def baseline(package, output, project_root):
    """Use the existing app's weights/routing/protection with fixed decoding."""
    import torch
    import sentencepiece as spm
    from .behavior_state import BehaviorController
    from .capability_router import CapabilityRouter
    from .dialogue_chat import protected_reply, reply
    from .model import PetLM
    from virtual_pet.local_model import STYLE_GUIDANCE

    project_root, output = Path(project_root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest, all_rows = read_package(package)
    rows = [row for row in all_rows if row['split'] == 'internal_test']
    freeze_path = project_root / 'data/petlm/evaluation/v6r5-candidate-freeze.json'
    freeze_hash = checksum(freeze_path)
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    # Deliberately do NOT traverse internal_test, seals, regression reports or
    # review evidence referenced by this manifest.
    names = ('base_checkpoint', 'tokenizer', 'safety_adapter', 'memory_adapter',
             'motion_adapter', 'format_adapter', 'behavior_state_code', 'router_code', 'chat_code')
    before = {}
    for name in names:
        item = freeze['files'][name]
        path = project_root / item['path']
        before[name] = checksum(path)
        if before[name] != item['sha256']:
            raise RuntimeError(f'Frozen asset mismatch: {name}')
    base_path = project_root / freeze['files']['base_checkpoint']['path']
    adapter_paths = {cap: project_root / freeze['files'][f'{key}_adapter']['path']
                     for cap, key in (('safety_behavior', 'safety'), ('memory_management', 'memory'),
                                      ('motion_clarification', 'motion'), ('format_control', 'format'))}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state = torch.load(base_path, map_location='cpu', weights_only=True)
    tokenizer = spm.SentencePieceProcessor(model_proto=(project_root / freeze['files']['tokenizer']['path']).read_bytes())
    model = PetLM(state['config'])
    model.load_state_dict(state['model'])
    model.to(device).eval()
    router = CapabilityRouter(base_path, base_path.parent.parent, device=device, adapter_paths=adapter_paths)
    output.mkdir(parents=True)
    records = []
    decoding = freeze['decoding']
    for index, row in enumerate(rows):
        torch.manual_seed(decoding['seed'] + index)
        history = [(turn['user'], turn['assistant']) for turn in row['history']]
        history.insert(0, (
            f'对话设定：你是虚拟宠物团团，用中文回应，表达风格为{STYLE_GUIDANCE["gentle"]}。'
            '你没有真实身体，不声称已执行现实动作、写入记忆或设置提醒。',
            '我会按这个风格交流。'))
        controller = BehaviorController()
        capability, selected = router.selected_model(history, row['user'])
        answer = controller.handle(history, row['user'])
        source = controller.last_source
        if answer is None:
            answer = protected_reply(history, row['user'])
            source = 'legacy_protected_reply' if answer is not None else None
        if answer is None:
            answer = reply(selected if selected is not None else model, tokenizer, history, row['user'],
                           max_new_tokens=decoding['max_new_tokens'], temperature=decoding['temperature'],
                           use_protection=False)
            source = 'model_generation'
        record = {
            'id': row['id'], 'scenario_key': row['scenario_key'], 'history': row['history'],
            'user': row['user'], 'answer': answer, 'capability': capability,
            'response_source': source, 'seed': decoding['seed'] + index,
            'diagnostic': diagnostic_flags(row, answer),
        }
        records.append(record)
        print(json.dumps({'progress': f'{index + 1}/{len(rows)}', 'id': row['id'],
                          'source': source, 'flags': record['diagnostic']['flags']}, ensure_ascii=False), flush=True)
    write_jsonl(output / 'outputs.jsonl', records)
    frequencies = Counter(normalized(record['answer']) for record in records)
    repeated = [{'ids': [record['id'] for record in records if normalized(record['answer']) == answer],
                 'count': count} for answer, count in frequencies.items() if count > 1]
    after = {name: checksum(project_root / freeze['files'][name]['path']) for name in names}
    report = {
        'kind': 'daily_dialogue_internal_baseline_diagnostic_v1',
        'status': 'diagnostic_only_not_acceptance',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'dataset_id': manifest['dataset_id'], 'count': len(records),
        'flagged_count': sum(bool(record['diagnostic']['flags']) for record in records),
        'response_sources': dict(Counter(record['response_source'] for record in records)),
        'exact_repeated_response_groups': repeated,
        'frozen_assets_unchanged': before == after and freeze_hash == checksum(freeze_path),
        'frozen_asset_hashes': after,
        'old_blind_prompts_or_answers_loaded': False,
        'training_performed': False,
        'human_review_complete': False,
        'semantic_pass_rate': None,
        'decoding': dict(decoding, seed_policy='frozen seed plus case index'),
        'outputs_sha256': checksum(output / 'outputs.jsonl'),
        'limitations': [
            'Pending references were not used for training or final acceptance.',
            'Flags are weak lexical checks; a flagged response needs review and an unflagged response is not a pass.',
            'Independent human review is still required. This internal baseline is not a frozen blind evaluation.',
        ],
    }
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('audit', 'baseline'))
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, default=Path('.'))
    args = parser.parse_args()
    result = (audit(args.package, args.output) if args.command == 'audit'
              else baseline(args.package, args.output, args.project_root))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
