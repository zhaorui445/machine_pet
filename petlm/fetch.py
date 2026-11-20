"""Bounded downloads from official sources; never silently substitute datasets."""
import argparse
import bz2
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

import httpx
import mwparserfromhell

from .data import BASE, SOURCES

HEADERS = {'User-Agent': 'PetLMResearch/0.1 (local educational corpus experiment; Python httpx)'}


def wiki(client, source, limit, output, metadata=None, start_offset=0,
         max_compressed_bytes=512 * 1024 * 1024, dump_date=None, archive_name=None,
         preserve_wikitext=False):
    project = 'zhwiki' if source == 'wikipedia' else 'zhwikisource'
    base = f'https://dumps.wikimedia.org/{project}/'
    job = {}
    if dump_date is not None:
        dates = [dump_date]
    else:
        response = client.get(base)
        response.raise_for_status()
        dates = sorted(set(re.findall(r'href="(\d{8})/"',response.text)),reverse=True)[:3]
    for date in dates:
        response = client.get(base+date+'/dumpstatus.json')
        if response.status_code == 404:
            continue
        response.raise_for_status()
        job = response.json()['jobs'].get('articlesdump',{})
        if job.get('status') == 'done':
            break
    if job.get('status') != 'done':
        raise RuntimeError('No completed article dump among latest three dates; import a dated dump manually.')
    files = sorted((name, info) for name, info in job['files'].items()
                   if '.xml' in name and name.endswith('.bz2')
                   and (archive_name is None or name == archive_name))
    if not files:
        raise RuntimeError('Requested article XML archive is absent from dump metadata')
    if archive_name is not None and len(files) != 1:
        raise RuntimeError('Pinned archive selection is ambiguous')
    selected_name, selected_info = files[0]
    url = 'https://dumps.wikimedia.org' + selected_info['url']
    parser = ET.XMLPullParser(events=('start','end'))
    decoder = bz2.BZ2Decompressor()
    count = 0
    eligible_seen = 0
    received = 0
    root = None
    with client.stream('GET', url) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes(65536):
            received += len(chunk)
            if received > max_compressed_bytes:
                raise RuntimeError(
                    f'Compressed Wikimedia read limit reached after {received} bytes; '
                    f'only {count} of {limit} requested documents were collected.')
            parser.feed(decoder.decompress(chunk))
            for event, elem in parser.read_events():
                if root is None and event == 'start':
                    root = elem
                if event != 'end' or elem.tag.rsplit('}',1)[-1] != 'page':
                    continue
                fields = {c.tag.rsplit('}',1)[-1]: c for c in elem}
                if fields['ns'].text == '0' and 'redirect' not in fields:
                    revision = next((x for x in elem if x.tag.rsplit('}',1)[-1] == 'revision'), None)
                    revision_fields = ({x.tag.rsplit('}',1)[-1]: x for x in revision}
                                       if revision is not None else {})
                    text_node = revision_fields.get('text')
                    raw = text_node.text or '' if text_node is not None else ''
                    text = mwparserfromhell.parse(raw).strip_code()
                    if len(text) >= 120:
                        if eligible_seen >= start_offset:
                            output.write(json.dumps({'id': fields['id'].text, 'text': text,
                                'title': fields['title'].text, 'url': url,
                                'revision_id': (revision_fields.get('id').text
                                               if revision_fields.get('id') is not None else None),
                                'timestamp': (revision_fields.get('timestamp').text
                                              if revision_fields.get('timestamp') is not None else None),
                                **({'raw_wikitext': raw} if preserve_wikitext else {}),
                                'eligible_index':eligible_seen}, ensure_ascii=False)+'\n')
                            count += 1
                        eligible_seen += 1
                        if eligible_seen % 1000 == 0:
                            print(f'{source}: scanned={eligible_seen}; collected={count}/{limit}',
                                  flush=True)
                root.clear()
                if count >= limit:
                    if metadata is not None:
                        metadata.update({'project':project,'dump_date':date,'archive_file':selected_name,'url':url,
                            'preserve_wikitext':preserve_wikitext,
                            'retrieved_utc':datetime.now(timezone.utc).isoformat(),
                            'documents':count,'start_offset':start_offset,
                            'next_offset':eligible_seen,'compressed_bytes_read':received,
                            'sampling':'Sequential eligible main-namespace pages from a pinned Wikimedia dump.'})
                    return count
    return count


def get_json(client, url, **kwargs):
    """Retry transient network errors and rate limiting; reject permanent HTTP errors."""
    attempts=20
    for attempt in range(attempts):
        try:
            response = client.get(url, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in (429, 500, 502, 503, 504) or attempt == attempts-1:
                raise
            retry_after = exc.response.headers.get('retry-after', '')
            delay = min(30, float(retry_after)) if retry_after.isdigit() else min(30, 2 ** (attempt + 1))
        except httpx.TransportError:
            if attempt == attempts-1:
                raise
            delay = min(30, 2 ** (attempt + 1))
        time.sleep(delay)


def fineweb(client, limit, output, metadata=None, start_offset=0):
    base = 'https://datasets-server.huggingface.co'
    dataset = 'HuggingFaceFW/fineweb-2'
    card = get_json(client, f'https://huggingface.co/api/datasets/{dataset}')
    if card.get('id') != dataset:
        raise RuntimeError('Unexpected dataset identity returned by official API')
    splits = get_json(client, base+'/splits', params={'dataset': dataset})
    configs = {x['config'] for x in splits['splits'] if x['split'] == 'train'}
    config = 'cmn_Hani'
    if config not in configs:
        raise RuntimeError(f'Expected Chinese configuration {config} is not present; inspect dataset metadata.')
    count = 0
    offset = start_offset
    skipped = 0
    seen = set()
    reported = 0
    while count < limit and offset < start_offset + limit * 4:
        payload = get_json(client, base+'/rows', params={'dataset':dataset,'config':config,
            'split':'train','offset':offset,'length':min(100,limit-count)})
        rows = payload['rows']
        if not rows:
            break
        for item in rows:
            row = item['row']
            if item['row_idx'] < offset:
                raise RuntimeError('Dataset viewer pagination went backwards')
            record_id = str(row.get('id') or item['row_idx'])
            if ('text' in item.get('truncated_cells', []) or not isinstance(row.get('text'), str)
                    or not row['text'].strip() or row.get('language') != 'cmn'
                    or row.get('language_script') != 'Hani' or record_id in seen):
                skipped += 1
                continue
            seen.add(record_id)
            output.write(json.dumps({'id':record_id, 'text':row['text'],
                'url':row.get('url') or f'https://huggingface.co/datasets/{dataset}',
                'dataset':dataset,'config':config,'split':'train','row_index':item['row_idx'],
                'license':card.get('cardData',{}).get('license','See official dataset card'),
                'language':row.get('language'),'language_script':row.get('language_script'),
                'dump':row.get('dump'),'date':row.get('date'),'file_path':row.get('file_path')},
                ensure_ascii=False)+'\n')
            count += 1
        offset = max(item['row_idx'] for item in rows) + 1
        if count//1000>reported or count==limit:
            reported=count//1000
            print(f'FineWeb2: {count}/{limit} complete Chinese documents; skipped={skipped}',flush=True)
    if metadata is not None:
        metadata.update({'dataset':dataset,'config':config,'split':'train',
            'repository_revision_observed':card.get('sha'),
            'license':card.get('cardData',{}).get('license'),
            'retrieved_utc':datetime.now(timezone.utc).isoformat(),
            'documents':count,'start_offset':start_offset,'next_offset':offset,
            'rows_scanned':offset-start_offset,'skipped':skipped,
            'endpoint':base,'sampling':'First complete eligible rows, not representative.',
            'revision_note':'Viewer API is not revision-pinned; local JSONL SHA256 identifies exact downloaded sample.'})
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=(*SOURCES,'all'), default='all')
    parser.add_argument('--limit', type=int, default=300)
    parser.add_argument('--offset', type=int, default=0,
                        help='Eligible-row/page offset for a separate, non-overlapping sample')
    parser.add_argument('--output', type=Path,
                        help='Write an additional sample to a new file without replacing existing data')
    parser.add_argument('--dump-date', help='Pin a Wikimedia dump date, for example 20261101')
    parser.add_argument('--dump-archive', help='Pin an exact article XML archive filename')
    parser.add_argument('--preserve-wikitext', action='store_true', help='Retain original wiki markup for rights and paragraph screening')
    args = parser.parse_args()
    if not 1 <= args.limit <= 200000:
        parser.error('Sample limit is 1..200000.')
    if args.offset < 0 or (args.offset and args.source not in ('fineweb2','wikipedia','wikisource')):
        parser.error('--offset is non-negative and requires one downloadable --source')
    if (args.dump_date or args.dump_archive) and args.source != 'wikipedia':
        parser.error('--dump-date and --dump-archive currently require --source wikipedia')
    if args.dump_archive and not args.dump_date:
        parser.error('--dump-archive requires an explicit --dump-date')
    raw = BASE/'raw'
    raw.mkdir(parents=True,exist_ok=True)
    status_path = BASE/'download-status.json'
    status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
    requested = SOURCES if args.source == 'all' else (args.source,)
    with httpx.Client(headers=HEADERS,timeout=25,follow_redirects=True) as client:
        for source in requested:
            if source == 'clue2020':
                status[source] = {'status':'requires_application','url':'https://github.com/CLUEbenchmark/CLUECorpus2020',
                    'reason':'Official README requires email application and no third-party redistribution. Import authorized local files.'}
                continue
            path = args.output if args.output else raw / (
                f'{source}-{args.offset:09d}-{args.offset + args.limit - 1:09d}.jsonl'
                if args.offset else f'{source}.jsonl')
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                previous = status.get(source,{})
                status[source] = {**previous,'status':'existing','path':str(path)}
                continue
            temp = path.with_suffix('.partial')
            try:
                metadata = {}
                completed = 0
                next_offset = args.offset
                if source == 'fineweb2' and temp.exists():
                    with temp.open(encoding='utf-8') as previous:
                        for line in previous:
                            row = json.loads(line)
                            completed += 1
                            next_offset = max(next_offset, row['row_index'] + 1)
                    if completed > args.limit:
                        raise RuntimeError('Partial file contains more rows than the requested limit.')
                with temp.open('a' if completed else 'w',encoding='utf-8') as handle:
                    additional = fineweb(client,args.limit-completed,handle,metadata,start_offset=next_offset) if source == 'fineweb2' and completed < args.limit else 0
                    if source != 'fineweb2':
                        additional = wiki(client,source,args.limit,handle,metadata,
                                          start_offset=args.offset,dump_date=args.dump_date,
                                          archive_name=args.dump_archive,preserve_wikitext=args.preserve_wikitext)
                count = completed + additional
                if count == 0:
                    raise RuntimeError('No documents downloaded')
                if count < args.limit:
                    raise RuntimeError(f'Only {count} of {args.limit} requested {source} documents available.')
                temp.replace(path)
                if metadata:
                    metadata['start_offset'] = args.offset
                    metadata['documents'] = count
                    if source == 'fineweb2':
                        metadata['rows_scanned'] = metadata['next_offset'] - args.offset
                    metadata['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                    path.with_suffix('.metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
                status[source] = {'status':'downloaded','documents':count,'path':str(path),
                                  'sampling':'First eligible records, not a representative sample.'}
                if args.offset:
                    status[source]['start_offset'] = args.offset
            except Exception as exc:
                # Keep partial data for diagnosis, but never promote it to a successful source.
                status[source] = {'status':'failed','error':f'{type(exc).__name__}: {exc}'}
            print(source, json.dumps(status[source],ensure_ascii=False),flush=True)
    status_path.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    if any(status[source]['status'] not in ('downloaded','existing') for source in requested):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
