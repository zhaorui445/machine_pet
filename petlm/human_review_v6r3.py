"""Build a fresh two-way development/internal-test review cycle.

The package intentionally has no frozen or training-ready split. Development
rows may be prepared for training only after independent item review; internal
test rows remain evaluation-only and are never copied into a training snapshot.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .human_review_pipeline import REVIEW_SCHEMA, leakage_report, write_jsonl
from .human_review_v6 import FAMILIES, make_case
from .run import checksum


SPLITS = ('development', 'internal_test')
AUTHOR_ID = 'codex-v6r3-candidate-generator-v1'
SCENE_NOTES = (
    ('我把今天的安排写在纸质清单上。', '我会区分清单里已经确定和还没确定的内容。'),
    ('我刚检查过现场提示牌，仍有一项需要确认。', '我不会把未确认的信息当作事实。'),
    ('我准备把结果转述给照看我的大人。', '我会用清楚、不过度推测的说法。'),
    ('我现在只想处理眼前这一件事。', '我会聚焦当前请求，不擅自扩展。'),
    ('我刚发现前后两条信息有一点不一样。', '我会先指出不确定之处，再决定是否回答。'),
    ('我希望之后能知道这件事有没有被记录。', '我会把当前对话和长期保存分开说明。'),
    ('我正在安静的地方和你确认安排。', '我会保持简洁，也会保留必要条件。'),
    ('我还没有执行任何动作，只是在询问。', '我不会把讨论说成已经发生的操作。'),
)


def scene_note(seed, family, index, split):
    digest = hashlib.sha256(f'{seed}|{family}|{index}|{split}|v6r3'.encode()).digest()
    return SCENE_NOTES[int.from_bytes(digest[:8], 'big') % len(SCENE_NOTES)]


def build(output, count_per_family=80, seed=20261108, external=(), dataset_id='v6r3'):
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
                row = make_case(capability, family, index, seed, split)
                row['id'] = row['id'].replace('hr-v6-', f'hr-{dataset_id}-', 1)
                row['author_id'] = AUTHOR_ID
                row['dataset_id'] = dataset_id
                note_user, note_assistant = scene_note(seed, family, index, split)
                if split == 'development':
                    row['history'].insert(1, {'user': note_user,
                                              'assistant': note_assistant})
                    row['user'] = f'补充说明：请把这条确认信息也纳入考虑。{row["user"]}'
                else:
                    row['history'].insert(1, {
                        'user': f'在继续之前，我又核对了一点：{note_user}',
                        'assistant': f'{note_assistant}如果和前文冲突，就以当前明确的信息为准。',
                    })
                    row['user'] = f'请检查上下文是否完整，再回答：{row["user"]}'
                row['prompt'] = ''.join(
                    f'用户：{turn["user"]}\n助手：{turn["assistant"]}\n'
                    for turn in row['history']) + f'用户：{row["user"]}\n助手：'
                rows.append(row)

    leak = leakage_report(rows, external, threshold=0.92)
    if not leak['passed']:
        raise RuntimeError(f'v6r3 leakage detected: {leak["findings"][:3]}')

    output.mkdir(parents=True)
    for split in SPLITS:
        write_jsonl(output / f'{split}-candidates.jsonl',
                    [row for row in rows if row['split'] == split])
    decisions = []
    for row in rows:
        decisions.append({
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
        })
    write_jsonl(output / 'review-decisions.template.jsonl', decisions)

    manifest = {
        'kind': 'human_review_v6r3_candidates_v1',
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
            'Internal-test rows are evaluation-only and must never enter training.'
        ),
        'physical_actuation_policy': (
            'Real motors, wheels, GPIO/serial actuation, and autonomous navigation remain disabled. '
            'Motion dialogue is simulation-only.'
        ),
        'independence': (
            'Development and internal-test prompts use disjoint generated indices and distinct '
            'context construction; the internal-test package is sealed after review and is not used '
            'for adapter selection.'
        ),
    }
    (output / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count-per-family', type=int, default=80)
    parser.add_argument('--seed', type=int, default=20261108)
    parser.add_argument('--external', type=Path, action='append', default=[])
    parser.add_argument('--dataset-id', default='v6r3')
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.count_per_family, args.seed,
                           args.external, args.dataset_id),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
