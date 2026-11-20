"""Conservative assistant preliminary audit for LCCC review candidates.

This produces triage only. It never marks a record as human reviewed or
approved for training.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from .run import checksum


CJK = re.compile(r'[\u3400-\u9fff]')
PHONE = re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)')
EMAIL = re.compile(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', re.I)
IDENTIFIER = re.compile(r'(?<!\d)\d{17}[\dXx](?!\d)|(?:身份证|银行卡|护照)(?:号|号码)?')
CONTACT = re.compile(
    r'电话号码|手机号|微信号|QQ号|联系方式|加我(?:微信|QQ)?|私聊我|'
    r'(?:地址|定位)(?:发|给|是)|发(?:个|一下)?(?:地址|定位)|你住哪|住在哪里')
PERSONAL_LOCATION = re.compile(
    r'(?:我|我们|我家|学校|公司).{0,14}(?:省|市|县|区|镇|村|小区|路|街|号楼|宿舍)')
HIGH_RISK = re.compile(
    r'自杀|自残|轻生|制毒|毒品|炸弹|爆炸物|枪支|强奸|性交易|裸聊|色情|'
    r'杀人|人肉搜索|跟踪定位|开盒|赌博|博彩|赌球|诈骗|自制武器')
ABUSE = re.compile(
    r'操你|他妈的|傻逼|傻B|狗杂种|滚你妈|去死|贱人|废物|脑残|有病吧|fuck|shit', re.I)
RISKY_CONTEXT = re.compile(r'劫|打架|打死|喝酒|醉|抽烟|约炮|报复|气死你|撞墙|跳楼')
DISMISSIVE = re.compile(r'不信算了|关我屁事|懒得理|滚开|闭嘴|活该|气死你')
ARTIFACT = re.compile(r'\|\s*\||=\s*=|@{2,}|#{2,}|\.{4,}|(?:…\s*){3,}')
DIALECT = re.compile(r'啷个|啥子|阔爱|好黑人|冷个远|晓得|莫得|勒个|爪子嘛|整啥')
PROMOTION = re.compile(r'扫码|关注公众号|点击下载|优惠券|代购|加群|兼职刷单|贷款广告')
GENERIC_SHORT = re.compile(r'^(?:好的?|是啊|对啊|嗯+|哈哈+|呵呵+|加油|不知道|随便|行吧|可以)$')
STOP = set('我你他她它们的是了吧呢啊呀嘛哦在有和也就都还又这那个很太会要想说去来让给把被与及而或吗么怎为什么哪谁多少')


def last_user_turn(prompt):
    turns = re.findall(r'用户：(.*?)(?=\n助手：|$)', prompt, re.S)
    return turns[-1].strip() if turns else ''


def content_chars(text):
    return {ch for ch in CJK.findall(text) if ch not in STOP}


def audit(row):
    prompt = row['prompt']
    answer = row['answer'].strip()
    text = prompt + '\n' + answer
    privacy = 'clear_by_rules'
    safety = 'clear_by_rules'
    naturalness = 'plausible_by_rules'
    relevance = 'plausible_by_lexical_check'
    tags = []

    if any(pattern.search(text) for pattern in (PHONE, EMAIL, IDENTIFIER, CONTACT)):
        privacy = 'reject'
        tags.append('personal_or_contact_information')
    elif PERSONAL_LOCATION.search(text):
        privacy = 'needs_human_review'
        tags.append('possible_personal_location_context')

    if HIGH_RISK.search(text) or ABUSE.search(text):
        safety = 'reject'
        tags.append('unsafe_or_abusive_content')
    elif RISKY_CONTEXT.search(text):
        safety = 'needs_human_review'
        tags.append('risky_or_age_sensitive_context')
    if PROMOTION.search(text):
        safety = 'reject'
        tags.append('promotion_or_solicitation')

    compact = re.sub(r'\s+', '', answer)
    chinese = CJK.findall(compact)
    if not chinese or ARTIFACT.search(answer) or DISMISSIVE.search(answer):
        naturalness = 'reject'
        tags.append('unnatural_artifact_or_hostile_style')
    elif len(chinese) < 4 or GENERIC_SHORT.fullmatch(compact):
        naturalness = 'needs_human_review'
        tags.append('short_or_generic_answer')
    elif DIALECT.search(text):
        naturalness = 'needs_human_review'
        tags.append('dialect_or_nonstandard_expression')
    elif len(chinese) / max(1, len(compact)) < 0.55:
        naturalness = 'needs_human_review'
        tags.append('low_chinese_answer_ratio')

    user = last_user_turn(prompt)
    user_content = content_chars(user)
    answer_content = content_chars(answer)
    overlap = len(user_content & answer_content)
    if not user or not answer:
        relevance = 'reject'
        tags.append('missing_turn_or_answer')
    elif overlap == 0 and len(chinese) < 12:
        relevance = 'needs_human_review'
        tags.append('weak_lexical_connection')
    elif overlap == 0:
        relevance = 'needs_human_review'
        tags.append('semantic_connection_not_verifiable_by_rules')

    dimensions = (privacy, safety, naturalness, relevance)
    if 'reject' in dimensions:
        decision = 'preliminary_reject'
    elif 'needs_human_review' in dimensions:
        decision = 'needs_human_review'
    else:
        decision = 'preliminary_clear'
    return {
        'id': row['id'], 'conversation_id': row['conversation_id'], 'split': row['split'],
        'prompt': prompt, 'answer': answer,
        'assistant_preliminary_decision': decision,
        'privacy': privacy, 'safety_age_suitability': safety,
        'naturalness': naturalness, 'relevance': relevance,
        'error_tags': sorted(set(tags)),
        'human_reviewed': False, 'approved_for_training': False,
        'review_method': 'assistant_preliminary_rules_v1',
        'review_note': 'Assistant triage only; a person must inspect the full prompt and answer.',
    }


def write_rows(path, rows):
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def run(source_root, output_root):
    if output_root.exists():
        raise FileExistsError(f'{output_root} exists; choose a new immutable output.')
    output_root.mkdir(parents=True)
    report = {'kind': 'lccc_assistant_preliminary_audit_v1',
              'status': 'assistant_preliminary_only_not_human_review',
              'source_root': str(source_root), 'splits': {},
              'human_reviewed': False, 'approved_for_training': False}
    for source in sorted(source_root.glob('*-review.jsonl')):
        split = source.name.removesuffix('-review.jsonl')
        rows = [audit(json.loads(line)) for line in source.read_text(encoding='utf-8').splitlines()
                if line.strip()]
        target = output_root / f'{split}-assistant-audit.jsonl'
        write_rows(target, rows)
        decisions = Counter(row['assistant_preliminary_decision'] for row in rows)
        tags = Counter(tag for row in rows for tag in row['error_tags'])
        report['splits'][split] = {
            'count': len(rows), 'decisions': dict(decisions), 'error_tags': dict(tags),
            'source_sha256': checksum(source), 'output_sha256': checksum(target),
        }
    batch_manifest_path = source_root / 'review-batches/manifest.json'
    if batch_manifest_path.is_file():
        source_batches = json.loads(batch_manifest_path.read_text(encoding='utf-8'))
        batch_report = {'kind': 'lccc_assistant_preliminary_batch_audit_v1',
                        'source_batch_manifest_sha256': checksum(batch_manifest_path),
                        'human_reviewed': False, 'approved_for_training': False,
                        'splits': {}}
        for split, split_info in source_batches['splits'].items():
            target_dir = output_root / 'batch-audits' / split
            target_dir.mkdir(parents=True)
            summaries = []
            for item in split_info['batches']:
                source_batch = Path(item['path'])
                if checksum(source_batch) != item['sha256']:
                    raise RuntimeError(f'Source review batch changed: {source_batch}')
                rows = [audit(json.loads(line))
                        for line in source_batch.read_text(encoding='utf-8').splitlines()
                        if line.strip()]
                target = target_dir / f"batch-{item['batch']:03d}.assistant-audit.jsonl"
                write_rows(target, rows)
                decisions = Counter(row['assistant_preliminary_decision'] for row in rows)
                tags = Counter(tag for row in rows for tag in row['error_tags'])
                summaries.append({'batch': item['batch'], 'count': len(rows),
                                  'decisions': dict(decisions), 'error_tags': dict(tags),
                                  'source_sha256': item['sha256'],
                                  'output_sha256': checksum(target), 'path': str(target)})
            batch_report['splits'][split] = summaries
        batch_report_path = output_root / 'batch-report.json'
        batch_report_path.write_text(
            json.dumps(batch_report, ensure_ascii=False, indent=2), encoding='utf-8')
        report['batch_report_sha256'] = checksum(batch_report_path)
        report['batch_count'] = sum(len(items) for items in batch_report['splits'].values())
    report['warning'] = ('Automated and assistant-authored triage can miss privacy, safety, '
                         'naturalness and relevance problems. It is not approval for SFT.')
    (output_root / 'report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    run(args.source_root, args.output_root)


if __name__ == '__main__':
    main()
