"""Build the v6r4 development cycle and its independent internal test."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .human_review_pipeline import REVIEW_SCHEMA, leakage_report, write_jsonl
from .human_review_v6 import FAMILIES, make_case
from .run import checksum


SPLITS = ('development', 'internal_test')
AUTHOR_ID = 'codex-v6r4-candidate-generator-v1'
RECORD_CARDS = (
    ('记录卡注明：目前没有执行任何操作。', '我会把询问和已执行动作严格分开。'),
    ('记录卡注明：关键信息来自当前现场描述。', '我只采用当前明确说明的信息。'),
    ('记录卡注明：需要保留对象名称和限制条件。', '我会在答复中保留具体对象和边界。'),
    ('记录卡注明：如果条件不足，应先停下确认。', '条件不足时我不会假定已经满足。'),
    ('记录卡注明：这次只处理一个明确目标。', '我会避免同时引入无关任务。'),
    ('记录卡注明：答复要能让监护人复核。', '我会说明依据、限制和需要谁来确认。'),
    ('记录卡注明：临时信息与长期记录必须分开。', '我会明确区分本次使用和长期保存。'),
    ('记录卡注明：所有移动描述都只限软件仿真。', '我不会把仿真提议表述成真实动作。'),
)
REVIEW_NOTES = (
    '复核时请先确认对象，再检查限制。',
    '请按当前轮次的明确要求作答。',
    '请不要沿用未经确认的旧结论。',
    '请保留问题里的具体主题和感受。',
    '请把是否执行和如何确认分开说明。',
    '请严格遵守本轮指定的输出形式。',
    '请不要把临时上下文写成长期事实。',
    '请在信息冲突时采用更保守的处理。',
)


def choose(seed, family, index, split, values, label):
    digest = hashlib.sha256(
        f'v6r4|{seed}|{family}|{index}|{split}|{label}'.encode()).digest()
    return values[int.from_bytes(digest[:8], 'big') % len(values)]


def rebuild_prompt(row):
    return ''.join(
        f'用户：{turn["user"]}\n助手：{turn["assistant"]}\n'
        for turn in row['history']) + f'用户：{row["user"]}\n助手：'


def build(output, count_per_family=80, seed=20261109, external=(), dataset_id='v6r4'):
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable path.')
    if count_per_family < 2 or count_per_family % 2:
        raise ValueError('count_per_family must be an even number of at least two.')

    rows = []
    half = count_per_family // 2
    for capability, families in FAMILIES.items():
        for family in families:
            for index in range(count_per_family):
                split = 'development' if index < half else 'internal_test'
                row = make_case(capability, family, index + 100, seed, split)
                row['id'] = f'hr-{dataset_id}-{capability}-{family}-{index:04d}'
                row['author_id'] = AUTHOR_ID
                row['dataset_id'] = dataset_id
                card_user, card_assistant = choose(
                    seed, family, index, split, RECORD_CARDS, 'card')
                review_note = choose(seed, family, index, split, REVIEW_NOTES, 'review')
                if split == 'development':
                    row['history'] = [
                        {'user': '我新建了一张事件记录卡，先核对背景。',
                         'assistant': '我会按记录卡逐项核对，不补充未说明的事实。'},
                        *row['history'][:1],
                        {'user': card_user, 'assistant': card_assistant},
                        *row['history'][1:],
                    ]
                    row['user'] = f'这是开发练习中的当前问题。{review_note}{row["user"]}'
                else:
                    row['history'] = [
                        {'user': '这是一次独立复核，请重新判断当前情况。',
                         'assistant': '我会从当前记录重新判断，不引用其他测试结论。'},
                        {'user': card_user, 'assistant': card_assistant},
                        *row['history'],
                        {'user': '提交答复前，再检查是否遗漏对象、限制或必要追问。',
                         'assistant': '我会以当前明确约束为准完成答复。'},
                    ]
                    row['user'] = f'这是独立测试中的新问题。{review_note}{row["user"]}'
                row['prompt'] = rebuild_prompt(row)
                rows.append(row)

    leak = leakage_report(rows, external, threshold=0.92)
    if not leak['passed']:
        raise RuntimeError(f'v6r4 leakage detected: {leak["findings"][:3]}')

    output.mkdir(parents=True)
    for split in SPLITS:
        write_jsonl(output / f'{split}-candidates.jsonl',
                    [row for row in rows if row['split'] == split])
    decisions = [{
        'schema': REVIEW_SCHEMA,
        'id': row['id'],
        'decision': 'pending',
        'reviewer_id': '',
        'reviewed_at': '',
        'independent_of_author': False,
        'factuality': '',
        'age_suitability': '',
        'safety': '',
        'naturalness': '',
        'approve_for_split': row['split'],
        'notes': '',
        'reviewer_signature': '',
    } for row in rows]
    write_jsonl(output / 'review-decisions.template.jsonl', decisions)

    manifest = {
        'kind': 'human_review_v6r4_candidates_v1',
        'dataset_id': dataset_id,
        'author_id': AUTHOR_ID,
        'seed': seed,
        'splits': list(SPLITS),
        'review_status': 'pending_independent_item_review',
        'eligible_for_training': False,
        'eligible_for_scoring': False,
        'counts': dict(Counter(row['split'] for row in rows)),
        'counts_by_capability': dict(Counter(row['capability'] for row in rows)),
        'counts_by_family': dict(Counter(row['behavior_family'] for row in rows)),
        'candidate_files': {
            split: checksum(output / f'{split}-candidates.jsonl') for split in SPLITS
        },
        'review_template_sha256': checksum(output / 'review-decisions.template.jsonl'),
        'leakage_report': leak,
        'external_paths': [str(path) for path in external],
        'training_policy': (
            'Only independently approved development rows may enter a later SFT snapshot. '
            'Internal-test rows are evaluation-only and must never enter training or tuning.'
        ),
        'independence': (
            'Development and internal-test use disjoint index ranges, different context framing, '
            'and immutable file hashes. Internal-test is sealed after review until one-time acceptance.'
        ),
        'physical_actuation_policy': (
            'Real motors, wheels, GPIO/serial actuation, and autonomous navigation remain disabled. '
            'Motion dialogue is simulation-only.'
        ),
    }
    (output / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count-per-family', type=int, default=80)
    parser.add_argument('--seed', type=int, default=20261109)
    parser.add_argument('--external', type=Path, action='append', default=[])
    parser.add_argument('--dataset-id', default='v6r4')
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.count_per_family, args.seed,
                           args.external, args.dataset_id),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
