"""Pin source licenses and fetch a bounded selection of plain Markdown files."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from .wikilingua_acquire import fetch

REPOS = {
    'd2l_v1': ('d2l-ai/d2l-zh', 'v1.0.0'),
    'd2l': ('d2l-ai/d2l-zh', 'master'),
    'hello_algo': ('krahets/hello-algo', 'main'),
    'cook': ('Anduin2017/HowToCook', 'master'),
}


def batch(jobs):
    failures = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {str(path): pool.submit(fetch, url, path) for url, path in jobs}
        for path, future in futures.items():
            try:
                future.result()
            except Exception as error:
                failures[path] = f'{type(error).__name__}: {error}'
    if failures:
        raise RuntimeError(json.dumps(failures, ensure_ascii=False))


def evidence(root):
    batch([(f'https://api.github.com/repos/{repo}/commits/{branch}', root/key/'commit.json')
           for key, (repo, branch) in REPOS.items() if key != 'd2l_v1'])
    jobs = []
    for key, (repo, _) in REPOS.items():
        if key == 'd2l_v1':
            continue
        commit = json.loads((root/key/'commit.json').read_text(encoding='utf-8'))['sha']
        jobs.extend([(f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1', root/key/'tree.json'),
                     (f'https://raw.githubusercontent.com/{repo}/{commit}/README.md', root/key/'README.md'),
                     (f'https://raw.githubusercontent.com/{repo}/{commit}/LICENSE', root/key/'LICENSE')])
    batch(jobs)


def download(root, selection, verification_name='download-verification.json'):
    spec = json.loads(selection.read_text(encoding='utf-8'))
    jobs = []
    for key, paths in spec['files'].items():
        repo, _ = REPOS[key]
        commit = json.loads((root/key/'commit.json').read_text(encoding='utf-8'))['sha']
        tree = json.loads((root/key/'tree.json').read_text(encoding='utf-8'))
        if tree.get('truncated'):
            raise ValueError('Cannot use truncated tree')
        files = {r['path']: r for r in tree['tree'] if r['type'] == 'blob'}
        for path in paths:
            if path not in files or not path.endswith('.md') or files[path]['size'] > 200_000:
                raise ValueError('Unexpected source path or size')
            if '..' in Path(path).parts or Path(path).is_absolute():
                raise ValueError('Unsafe source path')
            jobs.append((f'https://raw.githubusercontent.com/{repo}/{commit}/{path}',root/key/'text'/path))
    batch(jobs)
    verified = []
    for key, paths in spec['files'].items():
        tree = json.loads((root/key/'tree.json').read_text(encoding='utf-8'))
        files = {r['path']: r for r in tree['tree'] if r['type'] == 'blob'}
        for path in paths:
            content = (root/key/'text'/path).read_bytes()
            sha = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
            if sha != files[path]['sha']:
                raise ValueError('Source does not match pinned Git blob')
            verified.append({'source':key,'path':path,'bytes':len(content),'git_blob_sha1':sha,
                             'sha256':hashlib.sha256(content).hexdigest()})
    if Path(verification_name).name != verification_name:
        raise ValueError('Verification must be a filename in the source directory')
    target = root/verification_name
    record = {'files':verified,'complete':True,'selection_sha256':hashlib.sha256(selection.read_bytes()).hexdigest()}
    with target.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    print(f'Verified {len(verified)} pinned Markdown files; {sum(r["bytes"] for r in verified)} bytes.',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['evidence','download'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--verification-name', default='download-verification.json')
    args = parser.parse_args()
    if args.command == 'evidence':
        evidence(args.root)
    else:
        download(args.root, args.selection, args.verification_name)
