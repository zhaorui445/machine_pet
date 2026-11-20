"""Streaming import, provenance, exact deduplication and held-out splits."""
import argparse
import bz2
import gzip
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import unicodedata
import xml.etree.ElementTree as ET

import mwparserfromhell

SOURCES = ('wikipedia', 'fineweb2', 'wikisource', 'clue2020')
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data' / 'petlm'


def clean(text):
    text = unicodedata.normalize('NFKC', text).replace('\x00', '')
    text = re.sub(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', '[邮箱]', text)
    text = re.sub(r'(?<!\d)1[3-9]\d{9}(?!\d)', '[手机号]', text)
    return '\n'.join(line.strip() for line in text.splitlines() if line.strip())


def digest(text):
    return hashlib.sha256(re.sub(r'\s+', '', text).encode('utf-8')).hexdigest()


def split_for(key):
    bucket = int(key[:8], 16) % 100
    return 'test' if bucket < 5 else 'validation' if bucket < 10 else 'train'


def records(path):
    """JSONL fields: text, optional id/url/license; TXT uses blank-line documents."""
    name = path.name.lower()
    opener = bz2.open if name.endswith('.bz2') else gzip.open if name.endswith('.gz') else open
    if '.xml' in name:
        with opener(path, 'rb') as handle:
            context = ET.iterparse(handle, events=('start', 'end'))
            _, root = next(context)
            for event, elem in context:
                if event != 'end' or elem.tag.rsplit('}', 1)[-1] != 'page':
                    continue
                fields = {child.tag.rsplit('}', 1)[-1]: child for child in elem}
                if fields.get('ns') is not None and fields['ns'].text == '0' and 'redirect' not in fields:
                    text_node = next((n for n in elem.iter() if n.tag.rsplit('}', 1)[-1] == 'text'), None)
                    raw = text_node.text or '' if text_node is not None else ''
                    # Exclude template content and XML tags while retaining visible wiki text.
                    parsed = mwparserfromhell.parse(raw)
                    yield {'text': parsed.strip_code(normalize=True, collapse=True),
                           'id': fields['id'].text, 'title': fields['title'].text}
                root.clear()
    else:
        with opener(path, 'rt', encoding='utf-8-sig') as handle:
            if '.jsonl' in name or '.ndjson' in name or name.endswith('.partial'):
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        if not isinstance(row.get('text'), str):
                            raise ValueError(f'{path}: JSONL record requires a text string')
                        yield row
            elif '.txt' in name:
                paragraph = []
                for line in handle:
                    if line.strip():
                        paragraph.append(line.strip())
                    elif paragraph:
                        yield {'text': '\n'.join(paragraph)}
                        paragraph = []
                if paragraph:
                    yield {'text': '\n'.join(paragraph)}
            else:
                raise ValueError(f'Unsupported input format: {path}')


def import_data(source, path, origin, license_note, limit):
    if source not in SOURCES:
        raise ValueError(source)
    BASE.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(BASE / 'corpus.sqlite')
    connection.execute('CREATE TABLE IF NOT EXISTS documents (hash TEXT PRIMARY KEY, source TEXT, split TEXT, text TEXT)')
    connection.execute('CREATE TABLE IF NOT EXISTS provenance (hash TEXT, source TEXT, origin TEXT, license TEXT, record_id TEXT, UNIQUE(hash,source,origin,record_id))')
    accepted = 0
    inspected = 0
    files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
    try:
        for file in files:
            if not any(ext in file.name.lower() for ext in ('.txt', '.jsonl', '.ndjson', '.xml', '.partial')):
                continue
            for index, row in enumerate(records(file)):
                inspected += 1
                text = clean(row['text'])
                chinese = len(re.findall(r'[\u3400-\u9fff]', text))
                if not 120 <= len(text) <= 200_000 or chinese / max(len(text), 1) < 0.25:
                    continue
                key = digest(text)
                cursor = connection.execute('INSERT OR IGNORE INTO documents VALUES (?,?,?,?)', (key, source, split_for(key), text))
                connection.execute('INSERT OR IGNORE INTO provenance VALUES (?,?,?,?,?)',
                                   (key, source, row.get('url') or origin, row.get('license') or license_note,
                                    str(row.get('id', f'{file.name}:{index}'))))
                accepted += cursor.rowcount
                if inspected % 1000 == 0:
                    connection.commit()
                if accepted >= limit:
                    break
            if accepted >= limit:
                break
        connection.commit()
    finally:
        connection.close()
    print(json.dumps({'source': source, 'new_documents': accepted, 'inspected': inspected}, ensure_ascii=False))


def export_data(require_all, output=None):
    connection = sqlite3.connect(BASE / 'corpus.sqlite')
    counts = dict(connection.execute('SELECT source,count(*) FROM documents GROUP BY source'))
    missing = sorted(set(SOURCES) - counts.keys())
    if require_all and missing:
        connection.close()
        raise RuntimeError(f'Required sources missing: {missing}. No four-source dataset was built.')
    target = output or BASE / 'prepared'
    if target.exists():
        connection.close()
        raise FileExistsError('Prepared dataset exists; use a new project/data snapshot to avoid invalidating checkpoints.')
    target.mkdir(parents=True)
    stats = {}
    for split in ('train', 'validation', 'test'):
        count = 0
        with (target / f'{split}.jsonl').open('w', encoding='utf-8') as handle:
            for key, source, text in connection.execute('SELECT hash,source,text FROM documents WHERE split=? ORDER BY hash', (split,)):
                handle.write(json.dumps({'id': key, 'source': source, 'text': text}, ensure_ascii=False) + '\n')
                count += 1
        stats[split] = count
    with (target / 'provenance.jsonl').open('w', encoding='utf-8') as handle:
        for row in connection.execute('SELECT hash,source,origin,license,record_id FROM provenance'):
            handle.write(json.dumps(dict(zip(('id','source','origin','license','record_id'), row)), ensure_ascii=False) + '\n')
    connection.close()
    report = {'sources': counts, 'missing_sources': missing, 'splits': stats,
              'limitations': ['Exact whitespace-normalized deduplication only; near-duplicates may remain.',
                              'Basic Chinese-length and phone/email filters; not a complete content or privacy audit.',
                              'Traditional and simplified characters are preserved; no conversion.']}
    (target / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    imp = sub.add_parser('import')
    imp.add_argument('--source', choices=SOURCES, required=True)
    imp.add_argument('--path', type=Path, required=True)
    imp.add_argument('--origin', required=True)
    imp.add_argument('--license-note', required=True)
    imp.add_argument('--limit', type=int, default=10000)
    export = sub.add_parser('export')
    export.add_argument('--allow-partial', action='store_true')
    export.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.command == 'import':
        import_data(args.source, args.path, args.origin, args.license_note, args.limit)
    else:
        export_data(not args.allow_partial,args.output)


if __name__ == '__main__':
    main()
