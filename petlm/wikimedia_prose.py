"""Build attributable, group-isolated Chinese prose candidates; never train/approve."""
import argparse
import bz2
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import unicodedata
from urllib.parse import quote
import xml.etree.ElementTree as ET

import mwparserfromhell
import sentencepiece as spm

from .data import split_for
from .foundation_diagnostics import compact
from .pretrain_corpus import to_simplified, prohibited_character_ratio, repeated_shingle_ratio
from .pretrain_increment import canonical_hashes, PHONE, EMAIL, CN_ID, CARD, PRIVATE_DETAIL
from .prose_supplement import save
from .run import checksum

SOURCES = {'books': ('zhwikibooks', 'zh.wikibooks.org'),
           'voyage': ('zhwikivoyage', 'zh.wikivoyage.org')}
DATE = '20261101'
LICENSE = 'CC-BY-SA-4.0'
LICENSE_URL = 'https://creativecommons.org/licenses/by-sa/4.0/'
END = re.compile(r'[。！？][”’」』）\)\"]*$')
UNSUITABLE_TITLE = re.compile(
    '医|药|病|护理|解剖|临床|健康|性学|性教育|法律|命令注解|法条|投资|股票|博彩|赌博|军|枪|武器|爆炸|'
    '圣经|佛|易经|周易|论语|孟子|史记|古文|文言|唐诗|宋词|三字经|诗经|正气歌|风水|占卜|'
    '模拟联合国|俄罗斯|乌克兰|巴勒斯坦|以色列|阿富汗|叙利亚|伊朗|伊拉克|朝鲜|缅甸|'
    '版权|首页|沙盒|模板|用户|讨论|手册/测试')
RIGHTS_CONCERN = re.compile(r'copyvio|copyright violation|fair[ _-]?use|non[ _-]?free|侵权|侵權|版权所有|版權所有|未经.{0,8}许可|未經.{0,8}許可|禁止转载|禁止轉載|保留所有权利|保留所有權利', re.I)
OUTSIDE_SCOPE = re.compile(
    '治疗|药物|处方|服药|用药|疗效|诊断|癌症|糖尿病|疫苗|感染|疾病|艾滋|疟疾|中毒|'
    '自杀|自伤|轻生|毒品|大麻|海洛因|色情|性交|性爱|强奸|强暴|裸露|勃起|阴茎|阴道|'
    '赌博|博彩|赌场|彩票|股票|证券|期货|贷款|借贷|炸药|爆炸物|枪支|弹药|射击|战斗|恐怖袭击|'
    '抗议|示威|战争|歧视|偷渡|行贿|贿赂|偷窃|诈骗|盗窃|骗术|逃税|避税|走私|'
    '微信号|加微信|热线|客服电话|预订|预约|特价|促销|包邮|限时优惠')
DEPENDENT = re.compile(r'^(?:因此|所以|此外|然而|不过|但是|而且|接着|然后|最后|综上|由此|上述|这时|此时|这就|同样|相反|例如|比如)|如[上下]图|下图|上图|下表|上表|见图|如图|参见|前文|前面提到|后文|见上文|见下文|以下.*[:：]$')


def fingerprints(text):
    """Exact text/prefix, intact paragraphs and complete long sentences."""
    norm = compact(text)
    pieces = [norm, norm[:300]]
    pieces.extend(compact(p) for p in text.splitlines() if len(compact(p)) >= 80)
    pieces.extend(compact(p) for p in re.split(r'[。！？!?\n]+', text) if len(compact(p)) >= 40)
    return {hashlib.sha256(p.encode()).digest() for p in pieces}


def normalize(text):
    return to_simplified(unicodedata.normalize('NFKC', text)).strip()


def group_key(source, title):
    # All chapters/variants of one book or city remain in the same partition.
    return hashlib.sha256((source + ':' + normalize(title.split('/')[0])).encode()).hexdigest()


def paragraphs(raw):
    """Do not repair gaps made by templates, maths, lists, or image removal.

    A candidate is one original paragraph, with only link/formatting cleanup.
    Reject paragraphs containing non-prose constructs rather than splice prose.
    """
    section = ''
    for index, block in enumerate(re.split(r'\n\s*\n|(?=^={1,6}[^=])', raw, flags=re.M)):
        block = block.strip()
        if not block:
            continue
        heading = re.match(r'^=+\s*(.*?)\s*=+\s*(?:\n|$)', block)
        if heading:
            section = mwparserfromhell.parse(heading[1]).strip_code().strip()
            block = block[heading.end():].strip()
        if not block:
            continue
        if re.search(r'\{\{|\}\}|\{\||\|\}|<|>|-\{|\}-|__|https?://|www\.|^\s*[*#;:=|]|\[\[(?:File|Image|Category|文件|檔案|图像|圖片|分类|分類):', block, re.M | re.I):
            yield index, section, None, 'non_prose_structure_or_external_link'
            continue
        text = mwparserfromhell.parse(block).strip_code(normalize=True, collapse=False)
        text = normalize(re.sub(r'\s+', ' ', text))
        yield index, section, text, None


def reason(text):
    if not 200 <= len(text) <= 4000:
        return 'paragraph_length'
    if OUTSIDE_SCOPE.search(text):
        return 'outside_low_risk_prose_scope'
    if any(p.search(text) for p in (PHONE, EMAIL, CN_ID, CARD, PRIVATE_DETAIL)):
        return 'personal_or_contact_pattern'
    if prohibited_character_ratio(text) or re.search(r'[\[\]{}<>|=\\]|\ufffd|的的|了了|。。', text):
        return 'residual_markup_or_defect'
    norm = compact(text)
    if len(re.findall(r'[\u3400-\u9fff]', norm))/len(norm) < .78:
        return 'low_chinese_ratio'
    if len(re.findall('[A-Za-z]', norm))/len(norm) > .025:
        return 'latin_heavy'
    if len(re.findall(r'\d', norm))/len(norm) > .035:
        return 'digit_heavy'
    if not END.search(text) or len(re.findall('[。！？]', text)) < 3:
        return 'insufficient_complete_sentences'
    if DEPENDENT.search(text):
        return 'context_dependent'
    if len(re.findall('的|了|是|在|可以|如果|因为|这个|一个', text))/len(text) < .025:
        return 'insufficient_modern_prose_signals'
    if repeated_shingle_ratio(text) > .08:
        return 'repetition'
    for opening, closing in [('“','”'), ('「','」'), ('『','』'), ('(',')')]:
        if text.count(opening) != text.count(closing):
            return 'unbalanced_pair'
    return None


def page_rows(path):
    with bz2.open(path) as stream:
        context = ET.iterparse(stream, events=('start', 'end'))
        _, root = next(context)
        for event, elem in context:
            if event == 'end' and elem.tag.endswith('}page'):
                ns = elem.tag[:-4]
                if elem.findtext(ns+'ns') == '0' and elem.find(ns+'redirect') is None:
                    yield {'title': elem.findtext(ns+'title'), 'page_id': elem.findtext(ns+'id'),
                           'revision_id': elem.findtext(ns+'revision/'+ns+'id'),
                           'timestamp': elem.findtext(ns+'revision/'+ns+'timestamp'),
                           'raw': elem.findtext(ns+'revision/'+ns+'text') or ''}
                root.clear()


def verify_source(root):
    receipts = {}
    for name in ['dumps-legal.html', 'cc-by-sa-4.0.html', 'books-sha1.txt', 'voyage-sha1.txt'] + [
            f'{source}-{DATE}-pages-articles.xml.bz2' for source, _ in SOURCES.values()]:
        path = root/name
        meta = json.loads((root/(name+'.receipt.json')).read_text(encoding='utf-8'))
        if checksum(path) != meta['sha256'] or path.stat().st_size != meta['bytes']:
            raise ValueError(f'Changed source: {name}')
        receipts[name] = meta
    for key, (source, _) in SOURCES.items():
        name = f'{source}-{DATE}-pages-articles.xml.bz2'
        expected = next(line.split()[0] for line in (root/f'{key}-sha1.txt').read_text().splitlines()
                        if line.split()[-1] == name)
        with (root/name).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha1').hexdigest() != expected:
                raise ValueError('Official SHA1 mismatch')
    if 'https://creativecommons.org/licenses/by-sa/4.0/' not in (root/'dumps-legal.html').read_text(encoding='utf-8'):
        raise ValueError('Expected official text license evidence missing')
    return receipts


def deduplicate(rows, parent_fingerprints):
    kept, seen, rejected = [], set(), Counter()
    for row in sorted(rows, key=lambda r: ({'test':0, 'validation':1, 'train':2}[r['split']], r['id'])):
        hashes = fingerprints(row['text'])
        if hashes & parent_fingerprints:
            rejected['parent_text_prefix_paragraph_or_sentence_overlap'] += 1
        elif hashes & seen:
            rejected['candidate_text_prefix_paragraph_or_sentence_duplicate'] += 1
        else:
            seen.update(hashes)
            kept.append(row)
    return kept, rejected


def build(raw, base, tokenizer_dir, output):
    if output.exists():
        raise FileExistsError(output)
    receipts = verify_source(raw)
    parent = json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    index_meta = json.loads((base/'clean-index.meta.json').read_text(encoding='utf-8'))
    if index_meta['snapshot_manifest_sha256'] != checksum(base/'manifest.json') or index_meta['index_sha256'] != checksum(base/'clean-index.sqlite'):
        raise ValueError('Parent index changed')
    token_meta = json.loads((tokenizer_dir/'manifest.json').read_text(encoding='utf-8'))
    if checksum(tokenizer_dir/'tokenizer.model') != token_meta['tokenizer_sha256']:
        raise ValueError('Tokenizer changed')
    sp = spm.SentencePieceProcessor(model_proto=(tokenizer_dir/'tokenizer.model').read_bytes())
    db = sqlite3.connect((base/'clean-index.sqlite').resolve().as_uri()+'?mode=ro', uri=True)
    rows, rejected, pages = [], Counter(), Counter()
    try:
        for key, (source, host) in SOURCES.items():
            for page in page_rows(raw/f'{source}-{DATE}-pages-articles.xml.bz2'):
                pages[source] += 1
                title = normalize(page['title'])
                if UNSUITABLE_TITLE.search(title) or RIGHTS_CONCERN.search(page['raw']):
                    rejected[f'{source}/page_scope_or_rights_concern'] += 1
                    continue
                group = group_key(source, page['title'])
                split = split_for(group)
                for index, section, text, why in paragraphs(page['raw']):
                    why = why or reason(text)
                    if why:
                        rejected[f'{source}/{why}'] += 1
                        continue
                    full, prefix = canonical_hashes(text)
                    if db.execute('SELECT 1 FROM hashes WHERE full=? OR prefix=? LIMIT 1', (full, prefix)).fetchone():
                        rejected[f'{source}/parent_exact_prefix_duplicate'] += 1
                        continue
                    ids = sp.encode(text)
                    if sp.unk_id() in ids or compact(sp.decode(ids)) != compact(text):
                        rejected[f'{source}/tokenizer_roundtrip'] += 1
                        continue
                    origin = f'https://{host}/w/index.php?title={quote(page["title"])}&oldid={page["revision_id"]}'
                    rows.append({'id': full, 'source': source, 'text': text, 'title': title,
                        'original_title': page['title'], 'section': normalize(section),
                        'split': split, 'split_group': group, 'upstream_split': None,
                        'origin': origin, 'history_url': f'https://{host}/w/index.php?title={quote(page["title"])}&action=history',
                        'license': LICENSE, 'license_url': LICENSE_URL,
                        'attribution': f'{page["title"]} — {host} contributors; see revision and history URLs.',
                        'revision_id': page['revision_id'], 'revision_timestamp': page['timestamp'],
                        'page_id': page['page_id'], 'paragraph_index': index,
                        'raw_page_text_sha256': hashlib.sha256(page['raw'].encode()).hexdigest(),
                        'tokens_with_eos': len(ids)+1, 'human_review_status': 'pending',
                        'training_eligible': False})
            print(f'{source}: {pages[source]} main-namespace pages; total pending paragraphs={len(rows)}', flush=True)
    finally:
        db.close()
    wanted = set().union(*(fingerprints(row['text']) for row in rows)) if rows else set()
    overlap = set()
    for split in ('train', 'validation', 'test'):
        path = base/f'{split}.jsonl'
        if checksum(path) != parent['files'][path.name]:
            raise ValueError('Changed parent split')
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                overlap.update(fingerprints(json.loads(line)['text']) & wanted)
        print(f'Parent {split}: overlap scan finished', flush=True)
    kept, duplicates = deduplicate(rows, overlap)
    rejected.update(duplicates)
    output.mkdir(parents=True)
    shutil.copy2(Path(__file__), output/'screening-source.py')
    counts, tokens, groups, chinese_chars, visible_chars = Counter(), Counter(), {}, 0, 0
    for split in ('train', 'validation', 'test'):
        with (output/f'{split}-candidates.jsonl').open('x', encoding='utf-8', newline='\n') as out:
            for row in kept:
                if row['split'] != split:
                    continue
                assert groups.setdefault(row['split_group'], split) == split
                counts[f'{row["source"]}/{split}'] += 1
                tokens[f'{row["source"]}/{split}'] += row['tokens_with_eos']
                chinese_chars += len(re.findall(r'[\u3400-\u9fff]', compact(row['text'])))
                visible_chars += len(compact(row['text']))
                out.write(json.dumps(row, ensure_ascii=False)+'\n')
    # Spread a reproducible sample across books/cities, not just the largest book.
    sample, selected_groups = [], set()
    for source, _ in SOURCES.values():
        for split in ('train', 'validation'):
            pool = sorted((r for r in kept if r['source'] == source and r['split'] == split), key=lambda r:r['id'])
            local = []
            for row in pool:
                if row['split_group'] not in selected_groups:
                    local.append(row)
                    selected_groups.add(row['split_group'])
                if len(local) == (24 if split == 'train' else 8):
                    break
            sample.extend(local)
    review = [dict(r, human_decision=None, reviewer=None, review_notes='') for r in sample]
    save(output/'review.json', {'status':'pending_human_review', 'scope':'Stratified sample only, not full corpus approval', 'rows':review})
    lines = ['# 新来源自然中文：人工抽样复核', '',
        '中文维基教科书/维基导游；CC BY-SA 4.0。正文由原段落去格式、转简体，没有补写或拼接。',
        '这是新的候选批次，旧审核不适用。自动筛选不能证明语法、事实、隐私及语境均合格。',
        '逐条检查自然度、完整性、上下文依赖、事实/建议、隐私、版权例外；填写通过/拒绝/待定和理由。',
        '验证材料不进入训练；test 不在本审核表展示，也不称为正式冻结盲测。', '']
    for i, row in enumerate(review, 1):
        lines += [f'## W{i:03d} · {row["source"]} · {row["split"]}',
            f'标题：{row["title"]} / {row["section"]}', f'ID：{row["id"]}',
            f'来源：{row["origin"]}', f'作者历史：{row["history_url"]}', '', row['text'], '',
            '人工结论：待审核', '审核人：', '理由：', '']
    (output/'review.md').write_text('\n'.join(lines), encoding='utf-8')
    attribution = ['# Attribution and transformations', '',
        'Source: Wikimedia contributors, Chinese Wikibooks and Chinese Wikivoyage, official 2026-11-01 dumps.',
        'Text license: CC BY-SA 4.0 — https://creativecommons.org/licenses/by-sa/4.0/',
        'Each JSONL record retains its permanent revision link, author-history link and title.',
        'Changes: isolated original paragraph; stripped inline wiki formatting/links; normalized whitespace and Unicode NFKC; converted Traditional Chinese to Simplified Chinese.',
        'No images, lists, formula-bearing paragraphs, templates, or generated paraphrases are included.',
        'When sharing adapted text retain attribution, license link, modification notice and ShareAlike terms.',
        'Page-specific exceptions override general site licensing. Flagged pages were excluded; reviewer checks remain necessary.',
        'This does not make a blanket determination about copyright status of future model weights.', '']
    (output/'ATTRIBUTION.md').write_text('\n'.join(attribution), encoding='utf-8')
    report = {'kind':'wikimedia_natural_chinese_candidate_v1', 'status':'pending_fresh_human_sample_review',
        'source_receipts':receipts, 'license':LICENSE, 'parent_snapshot':str(base),
        'parent_manifest_sha256':checksum(base/'manifest.json'), 'tokenizer_sha256':token_meta['tokenizer_sha256'],
        'main_namespace_pages':dict(pages), 'counts':dict(counts), 'tokens_with_eos':dict(tokens),
        'candidate_chinese_character_ratio': chinese_chars/max(1,visible_chars),
        'split_groups':dict(Counter(groups.values())), 'rejected':dict(rejected),
        'review_samples':len(review), 'human_review_completed':False, 'training_started':False,
        'training_ready':False, 'physical_actuation':'disabled',
        'split_policy':'Local deterministic 90/5/5 hash by normalized book/city root, not upstream official or a frozen blind set',
        'overlap_policy':'Parent v26r1 all pretraining splits: normalized full/prefix/paragraph and long-sentence hashes; held-out wins candidate collisions',
        'limitations':['Not semantic deduplication or proof of novelty against every historical corpus.',
            'Chinese-character ratio is not a naturalness score. Automated filters require human review.',
            'General text license does not override page-specific non-free exceptions.',
            'Reference prose and travel descriptions are not natural multi-turn conversation data.',
            'No claim that these alone meet the language-repair data target.'],
        'files':{p.name:checksum(p) for p in output.iterdir() if p.is_file()}}
    save(output/'manifest.json', report)
    print(json.dumps({k:report[k] for k in ('counts','tokens_with_eos','review_samples','candidate_chinese_character_ratio')}, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, default=Path('data/petlm/raw/wikimedia-zh-source-r1'))
    parser.add_argument('--base', type=Path, default=Path('data/petlm/prepared-chinese-v26r1-strict'))
    parser.add_argument('--tokenizer-dir', type=Path, default=Path('data/petlm/tokens-chinese-v26r1-strict-24k'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.raw, args.base, args.tokenizer_dir, args.output)


if __name__ == '__main__':
    main()
