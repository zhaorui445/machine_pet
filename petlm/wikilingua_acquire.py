"""Pin public source evidence and safely acquire Chinese Parquet (no remote code)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import httpx

REVISION = 'ea3db3510cbd34d0f8dc612419ae40e4732f3b40'
GEM_REVISION = 'af5d0f00b59a6933165c97b384f50d8b563c314d'
DATA_FILE = 'chinese/train-00000-of-00001.parquet'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fetch(url, target):
    receipt = target.with_name(target.name + '.receipt.json')
    if target.exists():
        meta = json.loads(receipt.read_text(encoding='utf-8'))
        if meta['url'] != url or meta['sha256'] != digest(target):
            raise ValueError(f'Existing evidence changed: {target}')
        return meta
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.partial')
    for attempt in range(3):
        try:
            with httpx.Client(follow_redirects=True, timeout=30,
                              headers={'User-Agent': 'PetLMResearch/0.1 (local noncommercial corpus audit; python-httpx)'}) as client:
                with client.stream('GET', url) as response:
                    response.raise_for_status()
                    with temporary.open('wb') as output:
                        for chunk in response.iter_bytes():
                            output.write(chunk)
                    # Do not record signed redirect query strings.
                    final_origin = str(response.url.copy_with(query=None))
            meta = {'url': url, 'final_url_without_query': final_origin,
                    'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
                    'bytes': temporary.stat().st_size, 'sha256': digest(temporary)}
            temporary.rename(target)
            receipt.write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
            print(f'Saved {target.name}: {meta["bytes"]} bytes', flush=True)
            return meta
        except (httpx.HTTPError, OSError):
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def evidence(output):
    fetch('https://api.github.com/repos/esdurmus/Wikilingua/commits/master', output/'author-commit.json')
    commit = json.loads((output/'author-commit.json').read_text(encoding='utf-8'))['sha']
    sources = {
        'author-README.md': f'https://raw.githubusercontent.com/esdurmus/Wikilingua/{commit}/README.md',
        'author-LICENSE.txt': f'https://raw.githubusercontent.com/esdurmus/Wikilingua/{commit}/LICENSE',
        'hf-metadata.json': f'https://huggingface.co/api/datasets/esdurmus/wiki_lingua/revision/{REVISION}?blobs=true',
        'hf-README.md': f'https://huggingface.co/datasets/esdurmus/wiki_lingua/raw/{REVISION}/README.md',
        'gem-README.md': f'https://huggingface.co/datasets/GEM/wiki_lingua/raw/{GEM_REVISION}/README.md',
        'cc-by-nc-sa-3.0.html': 'https://creativecommons.org/licenses/by-nc-sa/3.0/legalcode.en',
        'wikihow-attribution.html': 'https://www.wikihow.com/wikiHow:Attribution',
    }
    errors = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {name: pool.submit(fetch, url, output/name) for name, url in sources.items()}
        for name, job in jobs.items():
            try:
                job.result()
            except Exception as error:
                errors[name] = f'{type(error).__name__}: {error}'
    if errors:
        raise RuntimeError(json.dumps(errors))


def download(output):
    raise RuntimeError('Acquisition held: historical license metadata conflicts with current wikiHow AI training terms. See source audit.')


def _download_after_rights_resolution(output):
    metadata = json.loads((output/'hf-metadata.json').read_text(encoding='utf-8'))
    if metadata['sha'] != REVISION:
        raise ValueError('Wrong upstream revision')
    file = next(row for row in metadata['siblings'] if row['rfilename'] == DATA_FILE)
    target = output/'chinese.parquet'
    receipt = fetch(f'https://huggingface.co/datasets/esdurmus/wiki_lingua/resolve/{REVISION}/{DATA_FILE}', target)
    if receipt['sha256'] != file['lfs']['sha256'] or receipt['bytes'] != file['size']:
        raise ValueError('Downloaded Parquet does not match pinned upstream LFS metadata')
    print('Pinned LFS size and SHA256 match.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['evidence', 'download'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'evidence': evidence, 'download': download}[args.command](args.output)


if __name__ == '__main__':
    main()
