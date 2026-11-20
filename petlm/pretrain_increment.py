"""Incrementally extend a strict pretraining snapshot with a pinned raw batch."""
import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
import re
import shutil
import sqlite3

from .pretrain_corpus import FINEWEB_SPAM_ORIGIN, clean_document
from .data import split_for
from .run import checksum


PHONE = re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)')
EMAIL = re.compile(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', re.I)
CN_ID = re.compile(r'(?<!\d)\d{17}[\dXx](?!\d)')
CARD = re.compile(r'(?<!\d)(?:\d[ -]?){15,19}(?!\d)')
PRIVATE_DETAIL = re.compile(r'身份证号|银行卡号|护照号|账号密码|支付密码|验证码|家庭住址|详细地址')


def canonical_hashes(text):
    canonical = re.sub(r'\s+', '', text)
    return (hashlib.sha256(canonical.encode()).hexdigest(),
            hashlib.sha256(canonical[:300].encode()).hexdigest())


def append_file(base, addition, target):
    with target.open('wb') as output:
        with base.open('rb') as source:
            shutil.copyfileobj(source, output, 1024 * 1024)
        with addition.open('rb') as source:
            shutil.copyfileobj(source, output, 1024 * 1024)


def connect_index(base, building):
    target = building / 'clean-index.sqlite'
    parent_index = base / 'clean-index.sqlite'
    parent_meta = base / 'clean-index.meta.json'
    if parent_index.is_file() and parent_meta.is_file():
        meta = json.loads(parent_meta.read_text(encoding='utf-8'))
        if (meta.get('snapshot_manifest_sha256') != checksum(base / 'manifest.json')
                or meta.get('index_sha256') != checksum(parent_index)):
            raise RuntimeError('Parent incremental-clean index metadata does not match the snapshot.')
        shutil.copy2(parent_index, target)
        connection = sqlite3.connect(target)
        count = connection.execute('SELECT count(*) FROM hashes').fetchone()[0]
        if count != meta['records']:
            raise RuntimeError('Parent incremental-clean index row count mismatch.')
        return connection, False, count

    connection = sqlite3.connect(target)
    connection.execute('CREATE TABLE hashes (full TEXT PRIMARY KEY, prefix TEXT)')
    connection.execute('CREATE INDEX hashes_prefix_idx ON hashes(prefix)')
    count = 0
    for split in ('train', 'validation', 'test'):
        with (base / f'{split}.jsonl').open(encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                full, prefix = canonical_hashes(row['text'])
                connection.execute('INSERT INTO hashes VALUES (?,?)', (full, prefix))
                count += 1
                if count % 10000 == 0:
                    connection.commit()
    connection.commit()
    return connection, True, count


def add_sample(heap, row, limit=100):
    priority = int(row['id'][:16], 16)
    entry = (-priority, row['id'], row)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)


def extend(base, raw, output):
    metadata_path = raw.with_suffix('.metadata.json')
    if not metadata_path.is_file():
        raise FileNotFoundError(f'Pinned raw metadata is required: {metadata_path}')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    raw_hash = checksum(raw)
    if metadata.get('sha256') != raw_hash:
        raise RuntimeError('Raw FineWeb2 batch hash differs from its metadata.')
    if metadata.get('dataset') != 'HuggingFaceFW/fineweb-2' or metadata.get('config') != 'cmn_Hani':
        raise RuntimeError('Unexpected incremental dataset identity or configuration.')
    if not (base / 'manifest.json').is_file():
        raise FileNotFoundError('Parent snapshot manifest is required.')
    if output.exists():
        raise FileExistsError(f'{output} exists; incremental snapshots are immutable.')
    building = output.with_name(output.name + '.building')
    if building.exists():
        raise FileExistsError(f'Incomplete build exists: {building}')
    building.mkdir(parents=True)

    base_manifest = json.loads((base / 'manifest.json').read_text(encoding='utf-8'))
    connection, bootstrapped, indexed = connect_index(base, building)
    expected = sum(base_manifest['splits'].values())
    if indexed != expected:
        connection.close()
        raise RuntimeError(f'Parent index has {indexed} rows; snapshot manifest declares {expected}.')

    spools = {split: building / f'.new-{split}.jsonl'
              for split in ('train', 'validation', 'test')}
    provenance_spool = building / '.new-provenance.jsonl'
    handles = {split: path.open('w', encoding='utf-8', newline='\n')
               for split, path in spools.items()}
    provenance_handle = provenance_spool.open('w', encoding='utf-8', newline='\n')
    accepted = Counter()
    rejected = Counter()
    quality = Counter()
    sample = []
    inspected = 0
    try:
        with raw.open(encoding='utf-8') as source:
            for line in source:
                if not line.strip():
                    continue
                inspected += 1
                row = json.loads(line)
                raw_text = row.get('text')
                if not isinstance(raw_text, str):
                    rejected['missing_text'] += 1
                    continue
                origin = row.get('url') or metadata.get('endpoint')
                if origin and FINEWEB_SPAM_ORIGIN.search(origin):
                    rejected['fineweb_spam_origin'] += 1
                    continue
                if any(pattern.search(raw_text) for pattern in (PHONE, EMAIL, CN_ID, CARD, PRIVATE_DETAIL)):
                    rejected['raw_personal_or_contact_information'] += 1
                    continue
                text, reason, signals = clean_document(raw_text, 'fineweb2', 'strict')
                if reason:
                    rejected[reason] += 1
                    continue
                rechecked, recheck_reason, _ = clean_document(text, 'fineweb2', 'standard')
                if recheck_reason or rechecked != text:
                    rejected['quality_recheck_not_idempotent'] += 1
                    continue
                full, prefix = canonical_hashes(text)
                exists = connection.execute(
                    'SELECT 1 FROM hashes WHERE full=? OR prefix=? LIMIT 1', (full, prefix)).fetchone()
                if exists:
                    rejected['duplicate_against_snapshot_or_batch'] += 1
                    continue
                connection.execute('INSERT INTO hashes VALUES (?,?)', (full, prefix))
                split = split_for(full)
                prepared = {'id': full, 'source': 'fineweb2', 'text': text}
                handles[split].write(json.dumps(prepared, ensure_ascii=False) + '\n')
                provenance = {'id': full, 'source': 'fineweb2', 'origin': origin,
                              'license': row.get('license') or metadata.get('license'),
                              'record_id': str(row.get('id') or row.get('row_index'))}
                provenance_handle.write(json.dumps(provenance, ensure_ascii=False) + '\n')
                accepted[split] += 1
                quality['duplicate_lines_removed'] += signals['duplicate_lines_removed']
                quality['wiki_tails_removed'] += int(signals['wiki_tail_removed'])
                add_sample(sample, dict(prepared, split=split, origin=origin,
                                        review_status='pending_sample_review'))
                if inspected % 5000 == 0:
                    connection.commit()
                    print(f'incremental_clean inspected={inspected}; accepted={sum(accepted.values())}', flush=True)
        connection.commit()
    finally:
        for handle in handles.values():
            handle.close()
        provenance_handle.close()
        connection.close()

    for split in ('train', 'validation', 'test'):
        append_file(base / f'{split}.jsonl', spools[split], building / f'{split}.jsonl')
        spools[split].unlink()
    append_file(base / 'provenance.jsonl', provenance_spool, building / 'provenance.jsonl')
    provenance_spool.unlink()

    sample_rows = [entry[2] for entry in sorted(sample, key=lambda entry: entry[1])]
    with (building / 'increment-review-sample.jsonl').open('w', encoding='utf-8', newline='\n') as handle:
        for row in sample_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')

    sources = dict(base_manifest['sources'])
    sources['fineweb2'] = sources.get('fineweb2', 0) + sum(accepted.values())
    splits = {split: base_manifest['splits'][split] + accepted[split]
              for split in ('train', 'validation', 'test')}
    cumulative_rejected = Counter()
    for key, value in base_manifest.get('rejected', {}).items():
        cumulative_rejected[key] += value
    for reason, value in rejected.items():
        cumulative_rejected[f'fineweb2/{reason}'] += value
    cumulative_quality = Counter(base_manifest.get('quality_actions', {}))
    cumulative_quality.update(quality)
    batches = list(base_manifest.get('incremental_batches', []))
    batches.append({'dataset': metadata['dataset'], 'config': metadata['config'],
                    'start_offset': metadata['start_offset'], 'next_offset': metadata['next_offset'],
                    'documents': metadata['documents'], 'raw_sha256': raw_hash,
                    'metadata_sha256': checksum(metadata_path), 'inspected': inspected,
                    'accepted': dict(accepted), 'rejected': dict(rejected)})
    manifest = {
        'kind': 'strict_incremental_chinese_pretraining_snapshot_v1',
        'parent_snapshot': str(base), 'parent_manifest_sha256': checksum(base / 'manifest.json'),
        'sources': sources, 'splits': splits,
        'rejected': dict(sorted(cumulative_rejected.items())),
        'quality_actions': dict(cumulative_quality),
        'dialogue_training_rows_added': base_manifest.get('dialogue_training_rows_added', 0),
        'dialogue_training_rows_by_dataset': base_manifest.get('dialogue_training_rows_by_dataset', {}),
        'dialogue_provenance': base_manifest.get('dialogue_provenance', {}),
        'policy': dict(base_manifest.get('policy', {}),
                       incremental_cleaning=True,
                       raw_personal_contact_documents_rejected=True,
                       dedup_index='exact normalized text and identical first 300 non-space characters'),
        'incremental_batches': batches,
        'latest_increment': {'inspected': inspected, 'accepted': dict(accepted),
                             'rejected': dict(rejected), 'index_bootstrapped': bootstrapped},
        'files': {name: checksum(building / name) for name in
                  ('train.jsonl', 'validation.jsonl', 'test.jsonl', 'provenance.jsonl',
                   'increment-review-sample.jsonl')},
        'limitations': [
            'Heuristic filters can retain errors and reject useful text.',
            'FineWeb2 ODC-BY metadata does not replace rights review for original web pages.',
            'A fixed accepted-document sample requires human inspection before model training.',
        ],
    }
    (building / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    index_meta = {'kind': 'strict_clean_dedup_index_v1',
                  'snapshot_manifest_sha256': checksum(building / 'manifest.json'),
                  'index_sha256': checksum(building / 'clean-index.sqlite'),
                  'records': sum(splits.values())}
    (building / 'clean-index.meta.json').write_text(
        json.dumps(index_meta, ensure_ascii=False, indent=2), encoding='utf-8')
    building.replace(output)
    print(json.dumps({'output': str(output), 'inspected': inspected,
                      'accepted': dict(accepted), 'rejected': dict(rejected),
                      'splits': splits, 'index_bootstrapped': bootstrapped,
                      'sample_count': len(sample_rows)}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    extend(args.base, args.raw, args.output)


if __name__ == '__main__':
    main()
