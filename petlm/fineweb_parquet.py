"""Inspect or extract a bounded FineWeb2 range through Parquet HTTP ranges."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import fsspec
import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from .run import checksum


DATASET = 'HuggingFaceFW/fineweb-2'
CONFIG = 'cmn_Hani'
SPLIT = 'train'
DEFAULT_URL = ('https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/'
               'refs%2Fconvert%2Fparquet/cmn_Hani/train/0000.parquet')
HEADERS = {'User-Agent': 'PetLMResearch/1.0 (non-commercial local corpus preparation)'}


def remote_info(url):
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=120) as client:
        response = client.head(url)
        response.raise_for_status()
        return {'requested_url': url, 'resolved_url': str(response.url),
                'content_length': int(response.headers.get('content-length', 0)),
                'etag': response.headers.get('etag'),
                'last_modified': response.headers.get('last-modified')}


def parquet_file(url):
    filesystem = fsspec.filesystem('http', headers=HEADERS)
    handle = filesystem.open(url, 'rb', block_size=16 * 1024 * 1024,
                             cache_type='blockcache')
    return handle, pq.ParquetFile(handle)


def layout(parquet):
    groups = []
    start = 0
    for index in range(parquet.metadata.num_row_groups):
        group = parquet.metadata.row_group(index)
        groups.append({'group': index, 'start': start, 'rows': group.num_rows,
                       'uncompressed_bytes': group.total_byte_size})
        start += group.num_rows
    return groups


def read_row_groups_resilient(url, group_ids, columns, chunk_size=10, attempts=5,
                              opener=None, sleeper=time.sleep):
    """Read bounded row-group chunks, reopening the HTTP file after failures."""
    if chunk_size < 1 or attempts < 1:
        raise ValueError('chunk_size and attempts must be positive')
    opener = opener or parquet_file
    tables = []
    for position in range(0, len(group_ids), chunk_size):
        chunk = group_ids[position:position + chunk_size]
        for attempt in range(attempts):
            handle = None
            try:
                handle, parquet = opener(url)
                table = parquet.read_row_groups(chunk, columns=columns)
                tables.append(table)
                print(f'Parquet row groups: {position + len(chunk)}/{len(group_ids)}',
                      flush=True)
                break
            except Exception:
                if attempt == attempts - 1:
                    raise
                sleeper(min(30, 2 ** (attempt + 1)))
            finally:
                if handle is not None:
                    handle.close()
    if not tables:
        raise RuntimeError('No Parquet row groups were read.')
    return tables[0] if len(tables) == 1 else pa.concat_tables(tables)


def inspect(url):
    info = remote_info(url)
    handle, parquet = parquet_file(info['resolved_url'])
    try:
        groups = layout(parquet)
        report = dict(info, rows=parquet.metadata.num_rows,
                      row_groups=parquet.metadata.num_row_groups,
                      schema=parquet.schema_arrow.names, groups=groups)
    finally:
        handle.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))


def extract(url, start, limit, output):
    if output.exists() or output.with_suffix('.partial').exists():
        raise FileExistsError('Output or partial output already exists.')
    info = remote_info(url)
    handle, parquet = parquet_file(info['resolved_url'])
    try:
        groups = layout(parquet)
        end = start + limit
        selected = [item for item in groups
                    if item['start'] < end and item['start'] + item['rows'] > start]
        if not selected or end > parquet.metadata.num_rows:
            raise ValueError(f'Requested rows {start}:{end} exceed this shard.')
        available = set(parquet.schema_arrow.names)
        wanted = ['id', 'text', 'url', 'language', 'language_script',
                  'dump', 'date', 'file_path']
        columns = [name for name in wanted if name in available]
        group_ids = [item['group'] for item in selected]
        handle.close()
        handle = None
        # Reopen through the stable Hugging Face URL for every chunk. The CDN
        # redirect is signed and may expire during a long multi-chunk read.
        table = read_row_groups_resilient(info['requested_url'], group_ids, columns)
        group_start = selected[0]['start']
        table = table.slice(start - group_start, limit)
        rows = table.to_pylist()
    finally:
        if handle is not None:
            handle.close()
    if len(rows) != limit:
        raise RuntimeError(f'Expected {limit} extracted rows, got {len(rows)}.')
    partial = output.with_suffix('.partial')
    with partial.open('w', encoding='utf-8', newline='\n') as stream:
        for offset, row in enumerate(rows):
            stream.write(json.dumps({
                'id': str(row.get('id') or start + offset),
                'text': row['text'], 'url': row.get('url') or info['requested_url'],
                'dataset': DATASET, 'config': CONFIG, 'split': SPLIT,
                'row_index': start + offset, 'license': 'odc-by',
                'language': row.get('language'),
                'language_script': row.get('language_script'),
                'dump': row.get('dump'), 'date': str(row.get('date') or ''),
                'file_path': row.get('file_path'),
            }, ensure_ascii=False) + '\n')
    partial.replace(output)
    metadata = dict(info, dataset=DATASET, config=CONFIG, split=SPLIT,
                    retrieved_utc=datetime.now(timezone.utc).isoformat(),
                    start_offset=start, next_offset=end, documents=limit,
                    parquet_row_groups=selected,
                    extraction='Parquet HTTP range requests; local JSONL hash pins extracted rows.',
                    license='odc-by', sha256=checksum(output))
    output.with_suffix('.metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'path': str(output), 'documents': limit,
                      'sha256': metadata['sha256'], 'row_groups': selected},
                     ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('inspect', 'extract'))
    parser.add_argument('--url', default=DEFAULT_URL)
    parser.add_argument('--start', type=int)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'inspect':
        inspect(args.url)
    else:
        if args.start is None or args.limit is None or args.output is None:
            parser.error('extract requires --start, --limit and --output')
        if args.start < 0 or args.limit < 1:
            parser.error('start must be non-negative and limit must be positive')
        extract(args.url, args.start, args.limit, args.output)


if __name__ == '__main__':
    main()
