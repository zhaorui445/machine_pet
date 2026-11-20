"""Record a user's package-level approval without inventing item-review evidence."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .human_review_pipeline import read_jsonl, write_jsonl
from .run import checksum


def record(assignment_root, output, statement):
    assignment_root, output = Path(assignment_root), Path(output)
    if output.exists():
        raise FileExistsError(f'{output} already exists.')
    if not statement.strip():
        raise ValueError('The actual user statement is required.')
    assignment_path = assignment_root / 'assignment.json'
    assignment = json.loads(assignment_path.read_text(encoding='utf-8'))
    source = Path(assignment['source_candidates'])
    if checksum(source / 'manifest.json') != assignment['source_manifest_sha256']:
        raise RuntimeError('Source manifest changed after assignment.')
    source_manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    source_rows = []
    for split in source_manifest['splits']:
        path = source / f'{split}-candidates.jsonl'
        if checksum(path) != source_manifest['candidate_files'][split]:
            raise RuntimeError(f'Source candidate hash mismatch: {split}')
        source_rows.extend(read_jsonl(path))
    source_by_id = {row['id']: row for row in source_rows}
    if len(source_by_id) != len(source_rows):
        raise RuntimeError('Source candidate IDs are not unique.')
    confirmations = []
    batch_records = []
    recorded_at = datetime.now(timezone.utc).isoformat()
    for batch in assignment['batches']:
        candidate_path, decision_path = Path(batch['candidates']), Path(batch['decisions'])
        if (checksum(candidate_path) != batch['candidates_sha256'] or
                checksum(decision_path) != batch['decisions_sha256']):
            raise RuntimeError(f'Assigned batch changed: {batch["batch"]}')
        candidates, decisions = read_jsonl(candidate_path), read_jsonl(decision_path)
        if ([r['id'] for r in candidates] != [r['id'] for r in decisions] or
                len(candidates) != batch['count']):
            raise RuntimeError('Batch coverage mismatch.')
        for candidate in candidates:
            if source_by_id.get(candidate['id']) != candidate:
                raise RuntimeError('Assigned candidate differs from source.')
            confirmations.append({
                'id': candidate['id'],
                'split': candidate['split'],
                'decision': 'user_confirmed_qualified',
                'confirmation_method': 'user_package_level_statement',
                'source_statement': statement,
                'assigned_reviewer_id': assignment['reviewer_id'],
                'recorded_at': recorded_at,
                'reviewed_at': None,
                'reviewer_signature': None,
                'factuality': None,
                'age_suitability': None,
                'safety': None,
                'naturalness': None,
                'independent_item_review_attested': False,
            })
        batch_records.append(dict(batch=batch['batch'], count=len(candidates),
                                  candidates_sha256=batch['candidates_sha256']))
    if (len(confirmations) != assignment['count'] or
            len({r['id'] for r in confirmations}) != assignment['count'] or
            {r['id'] for r in confirmations} != set(source_by_id)):
        raise RuntimeError('Confirmation must cover every candidate exactly once.')
    report = {
        'kind': 'petlm_user_batch_data_confirmation_v1',
        'dataset_id': assignment['dataset_id'],
        'status': 'user_confirmed_all_candidates_qualified',
        'source_statement': statement,
        'recorded_at': recorded_at,
        'recorded_at_meaning': 'Time the assistant saved this confirmation, not an asserted inspection time.',
        'count': len(confirmations),
        'counts': source_manifest['counts'],
        'source_assignment': str(assignment_root),
        'source_assignment_sha256': checksum(assignment_path),
        'source_manifest_sha256': checksum(source / 'manifest.json'),
        'assigned_reviewer_id': assignment['reviewer_id'],
        'batches': batch_records,
        'approval_scope': 'Candidate data quality only; not generated model replies or model acceptance.',
        'independent_signed_item_review_complete': False,
        'formal_item_decisions_modified': False,
        'training_started': False,
        'policy': 'Only development rows may be considered for later training. Internal-test rows remain evaluation-only.',
        'existing_model_acceptance_changed': False,
        'v6r5_blind_data_used': False,
        'hardware_policy': 'Real motors, wheels, GPIO/serial actuation, and autonomous navigation remain disabled.',
    }
    output.mkdir(parents=True)
    write_jsonl(output / 'candidate-confirmations.jsonl', confirmations)
    report['confirmations_sha256'] = checksum(output / 'candidate-confirmations.jsonl')
    (output / 'confirmation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assignment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--statement', required=True)
    args = parser.parse_args()
    print(json.dumps(record(args.assignment, args.output, args.statement), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
