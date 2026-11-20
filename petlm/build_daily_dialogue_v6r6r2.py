"""Package the revised daily-dialogue cycle for independent human review.

This script creates pending development and internal-test material. It does
not approve rows, train a model, or create a blind-set answer key.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from .daily_dialogue_v6r6r2_cases import DEVELOPMENT, INTERNAL_TEST
from .human_review_pipeline import leakage_report, write_jsonl, REVIEW_SCHEMA
from .run import checksum


DATASET_ID = 'v6r6r2'
AUTHOR_ID = 'codex-daily-dialogue-v6r6r2-author-v1'


def _prompt(row):
    return ''.join(f'用户：{turn["user"]}\n助手：{turn["assistant"]}\n'
                   for turn in row['history']) + f'用户：{row["user"]}\n助手：'


def _materialize(source, split):
    rows = []
    for number, item in enumerate(source, 1):
        row = dict(item)
        row.update({
            'id': f'hr-{DATASET_ID}-{split}-{number:03d}',
            'dataset_id': DATASET_ID,
            'split': split,
            'capability': 'safety_behavior' if item['behavior_family'] in {
                'hazard_observation', 'resolved_hazard_switch'} else 'natural_dialogue',
            'prompt': _prompt(item),
            'author_id': AUTHOR_ID,
            'synthetic': True,
            'review_status': 'pending',
        })
        rows.append(row)
    return rows


def build(output, external=()):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable path.')
    development = _materialize(DEVELOPMENT, 'development')
    internal = _materialize(INTERNAL_TEST, 'internal_test')
    rows = development + internal
    leakage = leakage_report(rows, [Path(path) for path in external], threshold=0.92)
    if not leakage['passed']:
        raise RuntimeError(f'v6r6r2 leakage detected: {leakage["findings"][:3]}')
    output.mkdir(parents=True)
    write_jsonl(output / 'development-candidates.jsonl', development)
    write_jsonl(output / 'internal_test-candidates.jsonl', internal)
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
        'kind': 'petlm_daily_dialogue_review_candidates_v1',
        'dataset_id': DATASET_ID,
        'author_id': AUTHOR_ID,
        'review_status': 'pending_independent_item_review',
        'eligible_for_training': False,
        'eligible_for_scoring': False,
        'splits': ['development', 'internal_test'],
        'counts': {'development': len(development), 'internal_test': len(internal)},
        'counts_by_family': dict(Counter(row['behavior_family'] for row in rows)),
        'candidate_files': {
            'development': checksum(output / 'development-candidates.jsonl'),
            'internal_test': checksum(output / 'internal_test-candidates.jsonl'),
        },
        'review_template_sha256': checksum(output / 'review-decisions.template.jsonl'),
        'leakage_report': leakage,
        'external_paths': [str(Path(path)) for path in external],
        'validation_focus': [
            'answer the current daily question before asking anything',
            'honour stated ingredients, allergies, time, equipment, and preferences',
            'follow up on the latest turn without blindly copying older replies',
            'switch away from source-search, emotion, and motion topics when appropriate',
            'keep safety priority when a hazard is still active and resume normal chat only after it is resolved',
        ],
        'blind_set_policy': 'No frozen blind set is created in this cycle until independent review and development checks are complete.',
        'hardware_policy': 'Real motors, wheels, GPIO/serial actuation, and autonomous navigation remain disabled.',
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--external', type=Path, action='append', default=[])
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.external), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
