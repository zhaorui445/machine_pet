"""Create a new, quality-screened pretraining snapshot without changing earlier data."""
import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import unicodedata

from opencc import OpenCC

from .data import BASE, SOURCES, clean, split_for

CONVERTER = OpenCC('t2s')
SPAM = re.compile(r'博彩|六合彩|时时彩|送彩金|皇冠现金网|皇冠代理|成人视频|裸聊|代开发票|加微信|联系QQ', re.I)
BOILERPLATE = re.compile(r'\[编辑\]|\[編輯\]|\{\{[^{}]*\}\}|\[\[[^\]]+\]\]|(?:thumb\s*\|[^\n]{0,180})', re.I)
HTML = re.compile(r'<[^>]{1,300}>')
CJK = re.compile(r'[\u3400-\u9fff]')
CHINESE_PUNCTUATION = set('，。！？；：、“”‘’（）《》〈〉【】〔〕…—·￥')
REPEATED_FRAGMENT = re.compile(r'(.{1,20})\1{5,}', re.S)


def unexpected_character_ratio(text):
    unexpected=0
    for character in text:
        code=ord(character)
        category=unicodedata.category(character)
        if character.isspace():
            continue
        prohibited=(
            0x1100<=code<=0x11ff or 0x3130<=code<=0x318f or 0xac00<=code<=0xd7af
            or 0x3040<=code<=0x30ff or 0x31f0<=code<=0x31ff
            or 0x1f000<=code<=0x1faff or 0xe000<=code<=0xf8ff
            or character=='\ufffd' or category in ('Cc','Cs','Co')
        )
        unexpected+=int(prohibited)
    return unexpected/max(1,len(text))


def screen(text, source, include_wikisource=False):
    if source == 'wikisource' and not include_wikisource:
        return None, 'classical_source'
    if SPAM.search(text):
        return None, 'spam_pattern'
    text = BOILERPLATE.sub(' ', text)
    text = HTML.sub(' ', text)
    text = CONVERTER.convert(clean(text))
    if not 400 <= len(text) <= 30_000:
        return None, 'length'
    if '\ufffd' in text or unexpected_character_ratio(text)>0:
        return None, 'unexpected_script_or_symbols'
    if len(CJK.findall(text)) / len(text) < 0.55:
        return None, 'low_chinese_ratio'
    if len(re.findall(r'https?://|www\.|@', text)) > 3:
        return None, 'many_links'
    blocks = [text[i:i + 60] for i in range(0, len(text) - 59, 60)]
    if blocks and len(set(blocks)) / len(blocks) < 0.7:
        return None, 'repeated_blocks'
    if REPEATED_FRAGMENT.search(text):
        return None, 'repeated_fragment'
    if sum(text.count(mark) for mark in ('上一篇', '下一篇', '点击下载', '友情链接', '相关推荐')) > 5:
        return None, 'navigation_text'
    return text, None


def curate(output, include_wikisource=False):
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable snapshot path.')
    output.mkdir(parents=True)
    rejected = Counter()
    accepted = Counter()
    splits = Counter()
    whole_hashes = set()
    prefix_hashes = set()
    rows = []
    with closing(sqlite3.connect(BASE/'corpus.sqlite')) as connection:
        for key, source, raw in connection.execute('SELECT hash,source,text FROM documents ORDER BY hash'):
            text, reason = screen(raw, source, include_wikisource)
            if reason:
                rejected[(source, reason)] += 1
                continue
            canonical = re.sub(r'\s+', '', text)
            full_hash = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
            prefix_hash = hashlib.sha256(canonical[:200].encode('utf-8')).hexdigest()
            if full_hash in whole_hashes or prefix_hash in prefix_hashes:
                rejected[(source, 'duplicate_after_normalization_or_prefix')] += 1
                continue
            whole_hashes.add(full_hash)
            prefix_hashes.add(prefix_hash)
            split = split_for(full_hash)
            rows.append((split, full_hash, source, text, key))
            accepted[source] += 1
            splits[split] += 1
        with (output/'provenance.jsonl').open('w', encoding='utf-8') as handle:
            for split, full_hash, source, _, original_key in rows:
                for origin, license_note, record_id in connection.execute(
                    'SELECT origin,license,record_id FROM provenance WHERE hash=?', (original_key,)):
                    handle.write(json.dumps({'id':full_hash,'source':source,'split':split,
                        'origin':origin,'license':license_note,'record_id':record_id},ensure_ascii=False)+'\n')
    for split in ('train','validation','test'):
        with (output/f'{split}.jsonl').open('w',encoding='utf-8') as handle:
            for row_split, full_hash, source, text, _ in rows:
                if row_split == split:
                    handle.write(json.dumps({'id':full_hash,'source':source,'text':text},ensure_ascii=False)+'\n')
    manifest = {'sources':dict(accepted),'missing_sources':sorted(set(SOURCES)-set(accepted)),
        'splits':dict(splits),'rejected':{f'{s}/{reason}':n for (s,reason),n in sorted(rejected.items())},
        'policy':{'min_chars':400,'max_chars':30000,'min_chinese_ratio':0.55,
            'max_unexpected_character_ratio':0.0,
            'traditional_to_simplified':'OpenCC t2s','wikisource_included':include_wikisource,
            'dedup':'exact normalized text and identical first 200 non-space characters'},
        'limitations':['Heuristic filters can reject useful text and retain poor text.',
            'Prefix dedup is not a complete near-duplicate detector.',
            'Web content rights and suitability require human review.',
            'Dataset viewer samples are not representative or revision-pinned.']}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
    if not accepted or splits['validation'] == 0 or splits['test'] == 0:
        raise RuntimeError('Too few accepted documents for a usable split; inspect screening policy.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--include-wikisource',action='store_true',
        help='Include classical literary source; default targets modern assistant language')
    args=parser.parse_args()
    curate(args.output,args.include_wikisource)


if __name__=='__main__':
    main()
