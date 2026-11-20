"""Prepare a conservative prose-only candidate subset; never approve or train it.

Preserves source text, IDs and original splits. Filtering is intentionally lossy
and cannot establish grammar, facts, rights, privacy or semantic coherence.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
from urllib.parse import urlparse

import sentencepiece as spm

from .foundation_diagnostics import category, compact, write
from .pretrain_increment import PHONE, EMAIL, CN_ID, CARD, PRIVATE_DETAIL
from .run import checksum


NOISE = re.compile(r'神途|登录器|开服|私服|房屋二胎|娱乐城|博彩|代开发票|未完待续|本章完|点击阅读|全文阅读|相关阅读|转载地址|上一篇|下一篇|目前分类|返回首页|网页快照|\[手机号\]|\[邮箱\]')
PROMO = re.compile(r'在线.{0,8}(?:专家|咨询)|专家在线|关注.{0,20}(?:资讯|公众号)|加微信|加群|客服|优惠|联系电话|购买|包邮|点击|扫码|招商|瘦身|减肥')
SENSITIVE = re.compile(r'癫痫|癌症|肿瘤|药物|处方|治疗|用药|疗效|精神病|自杀|自伤|性爱|色情|阴茎|阴道|勃起|裸露|性侵|股票|证券|期货|融资|理财|贷款|借贷|股票代码')
MARKUP = re.compile(r'-\{|\}-|\d+px|\bthumb\b|[|｜<>]|https?://|www\.|\(\s*\)|（\s*）|\{\{|\}\}|\[\[|\]\]|\b(?:class|style)=', re.I)
BROKEN = re.compile(r'[:：]\s*[。！？]|[,，、]{2,}|(?:认识损失|俄然认识|治装潢党|切烧烤实)')
ENDING = re.compile(r'[。！？!?][”’」』）\)\"]*$')
SENTENCES = re.compile(r'[。！？!?]+')
RESIDUAL_NOISE = re.compile(r'新葡京|葡京赌场|威尼斯人|博天堂|太阳城|幼女|催情|灭口|妈的|他妈|我靠|灵力|恶灵|修炼|飞剑|妖精|提督|舰娘|神尊|飞升|三世情劫|医师|血糖|中药|药茶|开业兴隆|销售代理|合作伙伴|高额授信')


def reason(row):
    if row['source'] not in ('fineweb2', 'wikipedia'):
        return 'not_plain_prose_source'
    text = row['text']
    if NOISE.search(text):
        return 'web_spam_or_serial_navigation'
    if RESIDUAL_NOISE.search(text):
        return 'residual_spam_or_outside_prose_pilot_scope'
    if MARKUP.search(text):
        return 'markup_or_missing_slot'
    if BROKEN.search(text):
        return 'broken_or_keyword_injected_text'
    if PROMO.search(text):
        return 'promotional_or_advice_marketing'
    if SENSITIVE.search(text):
        return 'outside_low_risk_prose_pilot_scope'
    if any(p.search(text) for p in (PHONE, EMAIL, CN_ID, CARD, PRIVATE_DETAIL)):
        return 'personal_or_contact_pattern'
    if not 400 <= len(text) <= 6000:
        return 'pilot_length'
    normalized = compact(text)
    if re.search(r'\(\s*[,;]|（\s*[，；]|。。', text):
        return 'missing_slot_or_double_stop'
    grams = [normalized[i:i+24] for i in range(max(0,len(normalized)-23))]
    if len(grams) != len(set(grams)):
        return 'repeated_long_phrase'
    if len(re.findall(r'[\u3400-\u9fff]', normalized))/len(normalized) < .70:
        return 'low_chinese_ratio'
    if len(re.findall('[A-Za-z]', normalized))/len(normalized) > .02:
        return 'latin_heavy'
    if len(re.findall(r'\d', normalized))/len(normalized) > .03:
        return 'digit_heavy'
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Reject rather than remove headings/tails: no synthetic repair of sentences.
    if any(len(line) < 24 or not ENDING.search(line) for line in lines):
        return 'non_sentence_line_or_incomplete_tail'
    sentences = [compact(s) for s in SENTENCES.split(text) if len(compact(s)) >= 8]
    if len(sentences) < 4:
        return 'too_few_complete_sentences'
    if len(set(sentences)) != len(sentences):
        return 'duplicate_sentence'
    for opening, closing in [('“','”'),('「','」'),('『','』'),('（','）'),('(',')')]:
        if text.count(opening) != text.count(closing):
            return 'unbalanced_pair'
    return None


def hashes(text):
    norm = compact(text)
    pieces = [norm, norm[:300]]
    pieces.extend(compact(p) for p in text.splitlines() if len(compact(p)) >= 80)
    return {hashlib.sha256(piece.encode()).hexdigest() for piece in pieces}


def prepare(cycle, source, output_name):
    if Path(output_name).name != output_name:
        raise ValueError('Output name must be a single directory name')
    destination = cycle/output_name
    if destination.exists():
        raise FileExistsError(destination)
    audit = json.loads((cycle/'data-audit.json').read_text(encoding='utf-8'))
    tokens = Path('data/petlm/tokens-chinese-v26r1-strict-24k')
    token_manifest = json.loads((tokens/'manifest.json').read_text(encoding='utf-8'))
    sp = spm.SentencePieceProcessor(model_proto=(tokens/'tokenizer.model').read_bytes())
    if checksum(tokens/'tokenizer.model') != token_manifest['tokenizer_sha256']:
        raise RuntimeError('Tokenizer hash mismatch')
    for split in ('train','validation'):
        if checksum(source/f'{split}.jsonl') != token_manifest['input_sha256'][split]:
            raise RuntimeError('Parent text changed')
    # The validation split stays held out. Protect full/prefix/long paragraph
    # matches against ALL original validation documents, even rejected ones.
    heldout = set()
    with (source/'validation.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            heldout.update(hashes(json.loads(line)['text']))
    destination.mkdir()
    shutil.copyfile(Path(__file__), destination/'screening-source.py')
    report = {'kind':'prose_repair_candidate_v2', 'status':'pending_human_sample_review',
        'source_snapshot':str(source), 'source_manifest_sha256':checksum(source/'manifest.json'),
        'screening_code_sha256':checksum(Path(__file__)), 'text_modified':False,
        'selection_uses':'Structural rules and defects observed in new training-text audit; no blind results.',
        'validation_scope':'Inherited monitoring validation, not new independent acceptance.',
        'training_started':False, 'human_approval_granted':False, 'splits':{},
        'limits':['Filtering cannot certify semantic quality, grammar, privacy, rights or factual correctness.',
                  'No near-duplicate semantic independence claim. New reserved internal prompts remain unevaluated.',
                  'Rejected documents may contain useful prose; precision is preferred for this small pilot.']}
    review_rows = []
    for split in ('train','validation'):
        rejected, kept_sources, seen, reservoirs = Counter(), Counter(), set(), {}
        inspected, accepted, token_count = 0, 0, 0
        with (source/f'{split}.jsonl').open(encoding='utf-8') as handle, (destination/f'{split}-candidates.jsonl').open('x', encoding='utf-8') as output:
            for index, line in enumerate(handle):
                row = json.loads(line)
                inspected += 1
                reject = reason(row)
                fingerprint = hashes(row['text']) if reject is None else set()
                if reject is None and split == 'train' and fingerprint & heldout:
                    reject = 'overlap_with_original_validation'
                if reject is None and fingerprint & seen:
                    reject = 'within_split_exact_prefix_or_paragraph_duplicate'
                if reject:
                    rejected[reject] += 1
                    continue
                ids = sp.encode(row['text'])
                if sp.unk_id() in ids or compact(sp.decode(ids)) != compact(row['text']):
                    rejected['tokenizer_roundtrip'] += 1
                    continue
                seen.update(fingerprint)
                accepted += 1
                token_count += len(ids)+1
                kept_sources[row['source']] += 1
                candidate = dict(row, original_split=split, parent_row_index=index,
                    parent_text_sha256=hashlib.sha256(row['text'].encode()).hexdigest(),
                    approval_status='pending_sample_review')
                output.write(json.dumps(candidate, ensure_ascii=False)+'\n')
                length = 'short' if len(row['text'])<1000 else 'long'
                group = (row['source'],length,category(row['text']))
                score = hashlib.sha256(('prose-candidate-sample-v1:'+row['id']).encode()).hexdigest()
                reservoirs.setdefault(group, []).append((score,candidate))
                reservoirs[group] = sorted(reservoirs[group], key=lambda x:x[0])[:2]
        assert accepted+sum(rejected.values()) == inspected
        report['splits'][split] = {'inspected':inspected, 'accepted':accepted,
            'tokens_with_eos':token_count, 'sources':dict(kept_sources), 'rejected':dict(rejected),
            'file_sha256':checksum(destination/f'{split}-candidates.jsonl')}
        for group, entries in sorted(reservoirs.items()):
            for _, row in entries:
                review_rows.append(dict(row, stratum=list(group)))
        print(f'{split}: retained {accepted}/{inspected}; tokens={token_count}', flush=True)
    # Explicitly check known defects from the observed audit, not blind data.
    report['audited_training_samples'] = [{'id':s['id'], 'reason':reason(s)}
        for s in audit['samples'] if s['split']=='train']
    report['review_samples'] = len(review_rows)
    write(destination/'manifest.json', report)
    write(destination/'sample-review.json', {'manifest_sha256':checksum(destination/'manifest.json'),
        'origin':'Real source text; automatic subset selection, human decisions pending',
        'rows':[dict(row, decision=None, reviewer=None, notes='') for row in review_rows]})
    lines = ['# 连贯中文预训练候选：抽样审核', '',
        '本文件为经过更严格结构筛选后的原文样本，尚未获人工批准。没有改写原文。',
        '请逐条判断：语句自然完整、指代连贯、没有插词或话题拼接、内容合适、没有不应保留的个人信息。',
        '可填写“通过 / 拒绝 / 待定”和原因。validation 样本只作验证，不能进入训练。', '']
    for i, row in enumerate(review_rows,1):
        lines += [f'## R{i:03d} · {row["original_split"]} · {row["source"]}', '',
            f'ID：{row["id"]}', '', row['text'], '', '人工结论：待审核', '问题说明：', '']
    (destination/'sample-review.md').write_text('\n'.join(lines), encoding='utf-8')
    return report


def restrict_sources(cycle, parent, source, output_name):
    """Additional provenance-based selection after regex filters still leak spam."""
    if Path(output_name).name != output_name:
        raise ValueError('Output name must be a single directory name')
    dest = cycle/output_name
    if dest.exists():
        raise FileExistsError(dest)
    parent_manifest = json.loads((parent/'manifest.json').read_text(encoding='utf-8'))
    selected = {}
    for split in ('train','validation'):
        path = parent/f'{split}-candidates.jsonl'
        if checksum(path) != parent_manifest['splits'][split]['file_sha256']:
            raise RuntimeError('Parent candidates changed')
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                selected[row['id']] = (split,row)
    origins = {}
    with (source/'provenance.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if row['id'] in selected:
                if row['id'] in origins and origins[row['id']] != row:
                    raise RuntimeError('Conflicting provenance')
                origins[row['id']] = row
    allowed = ['people.com.cn','xinhuanet.com','news.cn','gov.cn','edu.cn','cas.cn',
               'kepu.net.cn','sciencenet.cn','thepaper.cn','chinadaily.com.cn',
               'wikipedia.org','wikimedia.org']
    dest.mkdir()
    shutil.copyfile(Path(__file__),dest/'screening-source.py')
    report = {'kind':'source_restricted_prose_candidate_v1','status':'pending_human_sample_review',
        'parent':str(parent),'parent_manifest_sha256':checksum(parent/'manifest.json'),
        'provenance_sha256':checksum(source/'provenance.jsonl'),
        'screening_code_sha256':checksum(Path(__file__)), 'allowed_domain_suffixes':allowed,
        'origin_checked':'local recorded provenance; no fresh network or page authenticity verification',
        'text_modified':False,'training_started':False,'human_approval_granted':False,
        'limits':['Source restriction is a heuristic, not proof of quality, rights, correctness or privacy.',
                  'Retained text was already available to the parent model; this is not new unique pretraining data.',
                  'Validation preserves original split and is monitoring-only, not new independent acceptance.'],
        'splits':{}}
    sp = spm.SentencePieceProcessor(model_proto=Path('data/petlm/tokens-chinese-v26r1-strict-24k/tokenizer.model').read_bytes())
    samples = []
    with (dest/'provenance.jsonl').open('x',encoding='utf-8') as provenance:
        for split in ('train','validation'):
            rejected, groups, accepted, tokens, source_counts = Counter(), {}, 0, 0, Counter()
            with (dest/f'{split}-candidates.jsonl').open('x',encoding='utf-8') as handle:
                for row_split,row in selected.values():
                    if row_split != split:
                        continue
                    origin = origins.get(row['id'])
                    host = (urlparse(origin.get('origin','')).hostname or '').lower() if origin else ''
                    if not any(host == d or host.endswith('.'+d) for d in allowed):
                        rejected['outside_source_allowlist'] += 1
                        continue
                    if any(part in host.split('.') for part in ('bbs','forum','blog')):
                        rejected['user_generated_subdomain'] += 1
                        continue
                    if re.search('加盟|营收|招商|总部.{0,20}支持|开店|赞助|品牌推广',row['text']):
                        rejected['residual_promotion'] += 1
                        continue
                    row = dict(row,origin=origin['origin'],license=origin.get('license'),
                               source_record_id=origin.get('record_id'))
                    handle.write(json.dumps(row,ensure_ascii=False)+'\n')
                    provenance.write(json.dumps(origin,ensure_ascii=False)+'\n')
                    accepted += 1
                    tokens += len(sp.encode(row['text']))+1
                    source_counts[row['source']] += 1
                    group = (row['source'],'short' if len(row['text'])<1000 else 'long',category(row['text']))
                    priority = hashlib.sha256(('source-restricted-review-v1:'+row['id']).encode()).hexdigest()
                    groups.setdefault(group,[]).append((priority,row))
                    groups[group] = sorted(groups[group],key=lambda x:x[0])[:2]
            report['splits'][split] = {'accepted':accepted,'tokens_with_eos':tokens,
                'sources':dict(source_counts),'rejected':dict(rejected),
                'file_sha256':checksum(dest/f'{split}-candidates.jsonl')}
            for group, entries in sorted(groups.items()):
                samples.extend(dict(row,stratum=list(group)) for _,row in entries)
    report['review_samples'] = len(samples)
    write(dest/'manifest.json',report)
    write(dest/'sample-review.json',{'manifest_sha256':checksum(dest/'manifest.json'),
        'origin':'Pending human sample review, not full corpus approval',
        'rows':[dict(row,decision=None,reviewer=None,notes='') for row in samples]})
    lines = ['# 来源约束后的候选语料：人工抽样审核','',
        '这是原有语料的候选子集，不是新增独立语料。来源限制与自动过滤不能代替人工判断。',
        '逐条检查语法、指代和上下文、残缺内容、广告拼接、隐私与内容适宜性。填写通过/拒绝/待定及原因。',
        'validation 样本保持验证用途，禁止加入训练；不要把通过抽样说成全量逐条审核。','']
    for i,row in enumerate(samples,1):
        lines += [f'## R{i:03d} · {row["original_split"]} · {row["source"]}','',
            f'ID：{row["id"]}', f'来源记录：{row["origin"]}','',row['text'],'',
            '人工结论：待审核','问题说明：','']
    (dest/'sample-review.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'splits':report['splits'],'review_samples':len(samples)},ensure_ascii=False),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cycle',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=Path('data/petlm/prepared-chinese-v26r1-strict'))
    parser.add_argument('--output-name',default='prose-candidates-r2')
    parser.add_argument('--restrict-from',type=Path)
    args = parser.parse_args()
    if args.restrict_from:
        restrict_sources(args.cycle,args.restrict_from,args.source,args.output_name)
    else:
        prepare(args.cycle,args.source,args.output_name)


if __name__ == '__main__':
    main()
