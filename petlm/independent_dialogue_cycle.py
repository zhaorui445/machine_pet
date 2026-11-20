"""Build a disjoint natural Chinese dialogue cycle for low-rate SFT.

The source responses are human-authored OpenAssistant records.  A whole-tree
exclusion against the current parent corpus is required; source feedback is
preserved but is not relabeled as a new local human audit.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import sentencepiece as spm

from .run import checksum


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/petlm'
PARENT_DIALOGUE = BASE / 'dialogue/prepared-human-openassistant-v1/train.jsonl'
SOURCES = tuple(
    BASE / 'dialogue' / snapshot / f'{split}.jsonl'
    for snapshot in ('prepared-oasst1-zh-context', 'prepared-oasst2-chinese-v17-all-reviewed')
    for split in ('train', 'validation', 'test')
)
TOKENIZER = BASE / 'tokens-chinese-v8-50m-frozen/tokenizer.model'
DEFAULT_ROOT = BASE / 'experiments/natural-dialogue-cycle-v1'
CJK = re.compile(r'[\u3400-\u9fff]')
MODEL_IDENTITY = re.compile(
    r'Open\s*Assistant|OpenAI|ChatGPT|\bGPT\b|语言模型|人工智能|\bAI\b|聊天机器人', re.I)
OUT_OF_SCOPE = re.compile(
    r'习近平|中国共产党|总统|选举|身高|体重|减肥|减重|布洛芬|对乙酰氨基酚|诊断|处方')


def normalized(text):
    return re.sub(r'\s+', '', text).casefold()


def tree_key(value):
    return str(value or '').rsplit(':', 1)[-1]


def read_rows(path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_rows(path, rows):
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def pair_key(row):
    return normalized(row.get('prompt', '') + '\0' + row.get('answer', ''))


def build(root, context=512):
    output = root / 'dialogue'
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable cycle root.')
    if not PARENT_DIALOGUE.exists():
        raise FileNotFoundError(PARENT_DIALOGUE)
    tokenizer = spm.SentencePieceProcessor(model_proto=TOKENIZER.read_bytes())

    parent_pairs = set()
    parent_trees = set()
    for row in read_rows(PARENT_DIALOGUE):
        parent_pairs.add(pair_key(row))
        parent_trees.add(tree_key(row.get('tree_id')))

    candidates = []
    rejected = Counter()
    seen_pairs = set(parent_pairs)
    seen_trees = set()
    source_counts = Counter()
    for source in SOURCES:
        for row in read_rows(source):
            source_name = row.get('source_dataset', source.parent.name)
            source_counts[source_name] += 1
            tree = tree_key(row.get('tree_id'))
            if not tree:
                rejected['missing_tree_id'] += 1
                continue
            if tree in parent_trees:
                rejected['parent_seen_conversation_tree'] += 1
                continue
            if pair_key(row) in seen_pairs:
                rejected['parent_seen_or_duplicate_pair'] += 1
                continue
            prompt, answer = row.get('prompt', ''), row.get('answer', '')
            if prompt.count('用户：') < 2 or prompt.count('助手：') < 1:
                rejected['not_multiturn'] += 1
                continue
            if not prompt.startswith('用户：') or not prompt.rstrip().endswith('助手：'):
                rejected['unexpected_dialogue_format'] += 1
                continue
            if len(answer.strip()) < 12:
                rejected['answer_too_short'] += 1
                continue
            if len(CJK.findall(answer)) / max(1, len(answer)) < .45:
                rejected['answer_not_chinese_prose'] += 1
                continue
            if MODEL_IDENTITY.search(prompt + answer):
                rejected['model_identity_or_promotion'] += 1
                continue
            if OUT_OF_SCOPE.search(prompt + answer):
                rejected['political_or_medical_advice_out_of_scope'] += 1
                continue
            if row.get('source_review_result') is False:
                rejected['source_review_rejected'] += 1
                continue
            if row.get('source_quality') is not None and row['source_quality'] < .75:
                rejected['source_quality_below_threshold'] += 1
                continue
            if row.get('source_helpfulness') is not None and row['source_helpfulness'] < .75:
                rejected['source_helpfulness_below_threshold'] += 1
                continue
            if row.get('source_rank') not in (None, 0):
                rejected['non_top_source_rank'] += 1
                continue
            token_count = len(tokenizer.encode(prompt)) + len(tokenizer.encode(answer)) + 1
            if token_count > context:
                rejected['context_overflow'] += 1
                continue
            seen_pairs.add(pair_key(row))
            seen_trees.add(tree)
            candidates.append(dict(row,
                tree_id=tree,
                source_tree_id=row.get('tree_id'),
                source_path=str(source),
                source_human_authored=True,
                independent_of_parent_training=True,
                local_human_review_status='pending_local_item_review',
                training_group='independent_natural_multiturn_v1'))

    # Split by whole conversation tree. A sorted hash order avoids tiny buckets
    # when this independent source is small, while remaining reproducible.
    ordered_trees = sorted(
        seen_trees,
        key=lambda tree: hashlib.sha256(('natural-dialogue-cycle-v1:' + tree).encode()).hexdigest())
    total_trees = len(ordered_trees)
    test_end = max(1, round(total_trees * .10))
    validation_end = test_end + max(1, round(total_trees * .15))
    development_end = validation_end + max(1, round(total_trees * .15))
    tree_buckets = {}
    for index, tree in enumerate(ordered_trees):
        tree_buckets[tree] = ('test' if index < test_end else
                              'validation' if index < validation_end else
                              'development' if index < development_end else 'train')
    for row in candidates:
        row['split'] = tree_buckets[row['tree_id']]

    counts = Counter(row['split'] for row in candidates)
    trees_by_split = {split: {row['tree_id'] for row in candidates if row['split'] == split}
                      for split in ('train', 'development', 'validation', 'test')}
    if min(counts.get(split, 0) for split in ('train', 'development', 'validation', 'test')) < 1:
        raise RuntimeError(f'Independent cycle split too small: {dict(counts)}')
    if any(trees_by_split[a] & trees_by_split[b]
           for a in trees_by_split for b in trees_by_split if a != b):
        raise RuntimeError('Conversation tree leakage across independent splits.')

    output.mkdir(parents=True)
    for split in ('train', 'development', 'validation', 'test'):
        write_rows(output / f'{split}.jsonl', (row for row in candidates if row['split'] == split))

    files = {split: checksum(output / f'{split}.jsonl')
             for split in ('train', 'development', 'validation', 'test')}
    review_dir = root / 'review'
    review_dir.mkdir()
    review_counts = {}
    for split in ('train', 'development', 'validation'):
        split_rows = [row for row in candidates if row['split'] == split]
        review_counts[split] = len(split_rows)
        lines = [f'# {split} 对话逐条复核\n',
                 '来源为人类撰写的 OpenAssistant 对话，但这些记录尚未由本项目逐条复核。请逐项检查自然度、事实、年龄适宜性、安全性和多轮上下文；不要将整批筛选结果视为批准。\n',
                 '复核结论应记录在配套 decisions.pending.jsonl；不要更改原始候选文件。\n']
        decisions = []
        for row in split_rows:
            lines.append(f'## {row["id"]}\n')
            for turn in re.findall(r'用户：(.*?)\n助手：(.*?)(?=\n用户：|$)', row['prompt'], re.S):
                lines.extend([f'用户：{turn[0].strip()}\n', f'助手：{turn[1].strip()}\n'])
            lines.extend([f'当前用户：{row["prompt"].rsplit("用户：", 1)[-1].splitlines()[0]}\n',
                          f'参考回答：{row["answer"]}\n'])
            decisions.append({
                'id': row['id'], 'split': split, 'decision': 'pending',
                'naturalness': '', 'factuality': '', 'age_suitability': '', 'safety': '',
                'conversation_context': '', 'reviewer_id': '', 'reviewed_at': '',
                'independent_of_author': False, 'reviewer_signature': '', 'notes': '',
            })
        (review_dir / f'{split}.md').write_text('\n'.join(lines), encoding='utf-8')
        write_rows(review_dir / f'{split}-decisions.pending.jsonl', decisions)
    manifest = {
        'kind': 'independent_natural_multiturn_dialogue_cycle_v1',
        'created_from_sources': [str(path) for path in SOURCES],
        'source_rows_by_file_before_filter': dict(source_counts),
        'parent_dialogue_training_file': str(PARENT_DIALOGUE),
        'parent_dialogue_training_sha256': checksum(PARENT_DIALOGUE),
        'parent_tree_count_excluded': len(parent_trees),
        'parent_pair_count_excluded': len(parent_pairs),
        'tokenizer_sha256': checksum(TOKENIZER),
        'context_tokens': context,
        'counts': dict(counts),
        'trees': {split: len(value) for split, value in trees_by_split.items()},
        'files': files,
        'rejected': dict(rejected),
        'source_human_authored': True,
        'source_feedback_preserved': True,
        'local_human_review_complete': False,
        'local_human_review_status': 'pending_item_level_review',
        'review_material_counts': review_counts,
        'independent_of_parent_training': True,
        'independent_of_old_blind_data': True,
        'split_policy': 'whole conversation trees; deterministic sorted-hash buckets target train 60%, development 15%, validation 15%, test 10%',
        'test_policy': 'test is held out from SFT and development decisions',
        'limitations': [
            'Independent means conversation trees and retained pairs are excluded from the current parent training snapshot; it is not a guarantee against every historical checkpoint.',
            'Source feedback is not a substitute for a new local item-level human review.',
            'No synthetic or templated dialogue was added.',
        ],
    }
    write_json(output / 'manifest.json', manifest)
    sufficient = counts.get('train', 0) >= 200 and counts.get('development', 0) >= 20 and counts.get('validation', 0) >= 20
    write_json(root / 'status.json', {
        'kind': 'independent_natural_dialogue_cycle_status_v1',
        'status': 'prepared_pending_local_review' if sufficient else 'insufficient_data_for_sft_pending_review',
        'manifest_sha256': checksum(output / 'manifest.json'),
        'counts': dict(counts),
        'training_started': False,
        'sft_started': False,
        'sft_data_volume_sufficient': sufficient,
        'sft_blockers': ([] if sufficient else [
            'Need at least 200 training dialogues, 20 development dialogues, and 20 validation dialogues for this exploratory SFT gate.',
            'Every training/development/validation reference is still pending local item-level review.',
        ]),
        'local_human_review_complete': False,
        'old_blind_data_used': False,
    })
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def verify(root):
    """Verify hashes and non-test split isolation without reading the test rows."""
    output = root / 'dialogue'
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    splits = ('train', 'development', 'validation')
    rows_by_split = {split: list(read_rows(output / f'{split}.jsonl')) for split in splits}
    for split, rows in rows_by_split.items():
        if checksum(output / f'{split}.jsonl') != manifest['files'][split]:
            raise RuntimeError(f'{split} hash mismatch.')
        if len(rows) != manifest['counts'].get(split, 0):
            raise RuntimeError(f'{split} count mismatch.')
    trees = {split: {row['tree_id'] for row in rows} for split, rows in rows_by_split.items()}
    pairs = {split: {pair_key(row) for row in rows} for split, rows in rows_by_split.items()}
    parent_trees = {tree_key(row.get('tree_id')) for row in read_rows(PARENT_DIALOGUE)}
    overlap = {f'{a}_{b}_{kind}': len(
        (trees[a] & trees[b]) if kind == 'tree' else (pairs[a] & pairs[b]))
        for a, b in (('train', 'development'), ('train', 'validation'), ('development', 'validation'))
        for kind in ('tree', 'pair')}
    parent_seen = sum(len(values & parent_trees) for values in trees.values())
    if any(overlap.values()) or parent_seen:
        raise RuntimeError(f'Leakage check failed: overlaps={overlap}, parent_seen={parent_seen}')
    audit_path = root / 'review/assistant-audit.jsonl'
    audit_rows = list(read_rows(audit_path)) if audit_path.exists() else []
    expected_ids = {row['id'] for split_rows in rows_by_split.values() for row in split_rows}
    audit_ids = [row.get('id') for row in audit_rows]
    if audit_rows and (len(audit_ids) != len(set(audit_ids)) or set(audit_ids) != expected_ids):
        raise RuntimeError('Assistant audit does not cover each train/development/validation row exactly once.')
    audit_counts = Counter(row.get('assistant_screen') for row in audit_rows)
    result = {
        'status': 'independent_splits_verified', 'counts': {s: len(rows) for s, rows in rows_by_split.items()},
        'cross_split_overlap': overlap, 'parent_seen_tree_count': parent_seen,
        'test_content_read': False,
        'test_sha256': checksum(output / 'test.jsonl'),
        'test_count_from_manifest': manifest['counts'].get('test', 0),
        'assistant_audit_count': len(audit_rows),
        'assistant_audit_counts': dict(audit_counts),
        'sft_data_volume_sufficient': (len(rows_by_split['train']) >= 200 and
                                       len(rows_by_split['development']) >= 20 and
                                       len(rows_by_split['validation']) >= 20),
    }
    write_json(root / 'verification.json', result)
    status_path = root / 'status.json'
    status = json.loads(status_path.read_text(encoding='utf-8'))
    status['verification_sha256'] = checksum(root / 'verification.json')
    status['split_isolation_verified'] = True
    status['assistant_audit_sha256'] = checksum(audit_path) if audit_path.exists() else None
    status['assistant_audit_counts'] = dict(audit_counts)
    status['sft_blockers'] = [
        f"Only {len(rows_by_split['train'])} training, {len(rows_by_split['development'])} development, and {len(rows_by_split['validation'])} validation examples remain; the exploratory minimum is 200/20/20.",
        f"Assistant screening recommends rejecting {audit_counts.get('reject', 0)} candidates; {audit_counts.get('candidate_pending_human_review', 0)} still require local human review.",
        'No independently reviewed replacement dialogue corpus is available in this snapshot.',
    ]
    status['sft_started'] = False
    write_json(status_path, status)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('build', 'verify'))
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--context', type=int, default=512)
    args = parser.parse_args()
    build(args.root, args.context) if args.command == 'build' else verify(args.root)


if __name__ == '__main__':
    main()
