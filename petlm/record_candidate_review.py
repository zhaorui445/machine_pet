"""Bind a user-reported review decision to an immutable candidate snapshot."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .run import checksum


def record(candidate, decision, reviewer, statement):
    manifest_path = candidate/'manifest.json'
    review_path = candidate/'review.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    review = json.loads(review_path.read_text(encoding='utf-8'))
    for name, expected in manifest.get('files', {}).items():
        if checksum(candidate/name) != expected:
            raise ValueError(f'Candidate snapshot changed: {name}')
    if manifest.get('files', {}).get('review.json') != checksum(review_path):
        raise ValueError('Review JSON is not bound by the candidate manifest')
    rows = review.get('rows')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Review snapshot has no rows')
    target = candidate/'user-review-record.json'
    if target.exists():
        raise FileExistsError(target)
    record = {'kind':'user_reported_human_review_v1', 'decision':decision,
        'recorded_utc':datetime.now(timezone.utc).isoformat(),
        'reviewer':reviewer, 'reviewer_identity_independently_verified':False,
        'user_statement':statement, 'scope':review.get('scope'),
        'candidate_path':str(candidate), 'manifest_sha256':checksum(manifest_path),
        'review_json_sha256':checksum(review_path),
        'review_md_sha256':checksum(candidate/'review.md'),
        'candidate_counts':manifest.get('counts'),
        'rows':[{'id':row['id'], 'text_sha256':hashlib.sha256(row['text'].encode()).hexdigest(),
                 'decision':decision} for row in rows]}
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'record':str(target), 'decision':decision, 'rows':len(rows),
        'manifest_sha256':record['manifest_sha256'],
        'review_json_sha256':record['review_json_sha256']}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--decision', choices=('pass','reject'), required=True)
    parser.add_argument('--reviewer', default='current_user')
    parser.add_argument('--statement', required=True)
    args = parser.parse_args()
    record(args.candidate, args.decision, args.reviewer, args.statement)


if __name__ == '__main__':
    main()
