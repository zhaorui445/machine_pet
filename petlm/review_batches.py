"""Split immutable JSONL review candidates into small, hashed review batches."""
import argparse
import json
from pathlib import Path

from .run import checksum


def split_batches(root, batch_size=100):
    target = root / 'review-batches'
    if target.exists():
        raise FileExistsError(f'{target} exists; review batches are immutable.')
    target.mkdir(parents=True)
    manifest = {'kind': 'petlm_review_batches_v1', 'batch_size': batch_size,
                'source_root': str(root), 'splits': {}, 'review_status': 'pending'}
    for source in sorted(root.glob('*-review.jsonl')):
        split = source.name.removesuffix('-review.jsonl')
        rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()
                if line.strip()]
        split_dir = target / split
        split_dir.mkdir()
        batches = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            path = split_dir / f'batch-{start // batch_size + 1:03d}.jsonl'
            with path.open('w', encoding='utf-8', newline='\n') as handle:
                for row in batch:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            batches.append({'batch': start // batch_size + 1, 'count': len(batch),
                            'path': str(path), 'sha256': checksum(path)})
        manifest['splits'][split] = {'source_sha256': checksum(source),
                                     'count': len(rows), 'batches': batches}
    (target / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({split: {'count': value['count'], 'batches': len(value['batches'])}
                      for split, value in manifest['splits'].items()},
                     ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=100)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    split_batches(args.root, args.batch_size)


if __name__ == '__main__':
    main()
