"""Privacy-filter and prepare DuRecDial 2.0 for non-commercial SFT research.

Only conversation text is emitted. Source profile metadata is never copied to
the working dataset. The official source test split is deliberately unopened.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import zipfile

import sentencepiece as spm

from .run import checksum


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/petlm'
ARCHIVE = BASE / 'dialogue/raw/durecdial2-data.zip'
ARCHIVE_META = BASE / 'dialogue/raw/durecdial2-data.metadata.json'
TOKENIZER = BASE / 'tokens-chinese-v8-50m-frozen/tokenizer.model'
DEFAULT_ROOT = BASE / 'experiments/natural-dialogue-cycle-v5'
CJK = re.compile(r'[\u3400-\u9fff]')
PHONE = re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)')
EMAIL = re.compile(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', re.I)
CN_ID = re.compile(r'(?<!\d)\d{17}[\dXx](?!\d)')
BANK_CARD = re.compile(r'(?<!\d)(?:\d[ -]?){15,19}(?!\d)')
PERSONAL_IDENTITY = re.compile(
    r'(?:我叫[\u3400-\u9fff]{2,4}(?:[,，。！!？?]|$)|本人叫[\u3400-\u9fff]{2,4}|'
    r'我的(?:姓名|名字|手机号|电话号码|微信号|QQ号|身份证号|银行卡号)|'
    r'家庭住址|住址是|详细地址|家庭地址|我住在.{0,10}(?:街|路|小区|号)|'
    r'出生日期|生日|身份证|护照号|学校名称|公司名称|工作单位|个人主页|社交账号)')
PRIVACY_SECRET = re.compile(r'(?:密码|验证码|支付密码|账号密码|门牌号)', re.I)
HIGH_RISK = re.compile(
    r'自杀|自残|轻生|毒品|制毒|爆炸物|炸弹制作|枪支制作|色情|裸聊|性交易|'
    r'强奸|杀人方法|人肉搜索|跟踪定位|开盒', re.I)
HEALTH_LEGAL_POLITICAL = re.compile(
    r'诊断|处方|剂量|药物治疗|法律规定|行政许可|习近平|中国共产党|选举|总统|'
    r'公安机关|身份证办理|签证政策', re.I)
TIME_SENSITIVE_FACT = re.compile(r'今天天气|当前气温|现在.*(?:价格|汇率)|今年.*(?:票房|排名|榜单)|最新|截至\d{4}年')
FACTUAL_QA = re.compile(
    r'(?:是谁|是什么|什么时候|哪一年|哪年|几岁|多少岁|身高|体重|生日|出生于|'
    r'哪部(?:电影|电视剧)|主演|导演|票房|获得过什么奖|哪里人|来自哪里|在哪(?:里|个)|'
    r'谁演的|有什么作品|哪首歌|哪个学校|哪个国家|具体地址|电话号码|联系方式|'
    r'历史上|成立于|发行于|上映于|排名第几|价格是多少|汇率是多少|气温多少)')
ENTITY_MARKER = re.compile(r'\[[0-9]+\]')
FACTUAL_CLAIM = re.compile(
    r'(?:\b(?:19|20)\d{2}\b|\d{1,3}(?:\s?(?:岁|米|厘米|公斤|斤|元|美元|名|位))|'
    r'出生于|身高\s?\d|票房|主演是|导演是|获得.{0,8}(?:奖|称号)|'
    r'上映于|发行于|位于.{0,8}(?:省|市|县|区))')
SPAM_OR_TEMPLATE = re.compile(r'扫码|加微信|点击下载|关注公众号|广告|充值|优惠券', re.I)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def write_rows(path, rows):
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def read_rows(path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def digest_text(text):
    return hashlib.sha256(re.sub(r'\s+', '', text).casefold().encode('utf-8')).hexdigest()


def normalize(text):
    text = re.sub(r'\s+', ' ', str(text)).strip()
    text = re.sub(r'(?<=[\u3400-\u9fff])\s+|\s+(?=[\u3400-\u9fff，。！？；：、])', '', text)
    return text.strip()


def screen_conversation(turns):
    reasons = []
    whole = '\n'.join(turns)
    if any(pattern.search(whole) for pattern in (PHONE, EMAIL, CN_ID, BANK_CARD, PERSONAL_IDENTITY)):
        reasons.append('possible_personal_or_contact_information')
    if PRIVACY_SECRET.search(whole):
        reasons.append('credential_or_private_detail')
    if HIGH_RISK.search(whole):
        reasons.append('high_risk_or_age_inappropriate')
    if SPAM_OR_TEMPLATE.search(whole):
        reasons.append('spam_or_promotion')
    if HEALTH_LEGAL_POLITICAL.search(whole):
        reasons.append('sensitive_factual_domain_requires_review')
    if TIME_SENSITIVE_FACT.search(whole):
        reasons.append('time_sensitive_claim_requires_review')
    if FACTUAL_QA.search(whole) or ENTITY_MARKER.search(whole):
        reasons.append('entity_or_factual_qa_excluded')
    if FACTUAL_CLAIM.search(whole):
        reasons.append('specific_factual_claim_excluded')
    odd = sum(bool(re.search(r'(.)\1{5,}', turn)) for turn in turns)
    if odd:
        reasons.append('repeated_character_noise')
    if not all(turn.strip() for turn in turns):
        reasons.append('empty_turn')
    if any(len(turn) > 1400 for turn in turns):
        reasons.append('overlong_turn')
    if len(CJK.findall(whole)) / max(1, len(whole)) < .45:
        reasons.append('low_chinese_ratio')
    return reasons


def make_examples(conversation_id, turns, tokenizer, context):
    examples = []
    for answer_index in range(1, len(turns), 2):
        history = turns[:answer_index]
        current_user = history[-1]
        previous = history[:-1]
        while True:
            prompt = ''.join(
                f'{"用户" if i % 2 == 0 else "助手"}：{turn}\n'
                for i, turn in enumerate(previous)
            ) + f'用户：{current_user}\n助手：'
            answer = turns[answer_index]
            if len(tokenizer.encode(prompt)) + len(tokenizer.encode(answer)) + 1 <= context:
                break
            if len(previous) >= 2:
                previous = previous[2:]
            else:
                return examples, 'context_overflow'
        examples.append({
            'id': hashlib.sha256(f'{conversation_id}:{answer_index}'.encode()).hexdigest(),
            'conversation_id': conversation_id,
            'answer_turn': answer_index,
            'prompt': prompt,
            'answer': answer,
            'source': 'DuRecDial 2.0',
            'source_license': 'CC BY-NC-SA 4.0; non-commercial use only',
            'source_quality_review': 'crowd_quality_control_source_not_local_item_review',
            'training_group': 'human_authored_recommendation_dialogue_v1',
        })
    return examples, None


def parse_split(archive, member, tokenizer, context, source_split):
    accepted = []
    rejected = Counter()
    accepted_dialogues = 0
    rejected_rows = []
    records = 0
    seen_records = set()
    with zipfile.ZipFile(archive) as zf, zf.open(member) as binary, __import__('io').TextIOWrapper(binary, encoding='utf-8') as stream:
        for line_number, line in enumerate(stream):
            if not line.strip():
                continue
            records += 1
            try:
                raw = json.loads(line)
                # Read no profile fields into normalized rows, review files or checkpoints.
                turns = [normalize(item) for item in raw.get('conversation', [])]
            except Exception:
                rejected['malformed_json_or_schema'] += 1
                continue
            if len(turns) < 4 or len(turns) % 2:
                rejected['malformed_turn_count'] += 1
                rejected_rows.append({'source_record': line_number, 'reasons': ['malformed_turn_count']})
                continue
            key = digest_text('\n'.join(turns))
            if key in seen_records:
                rejected['duplicate_dialogue'] += 1
                rejected_rows.append({'source_record': line_number, 'reasons': ['duplicate_dialogue']})
                continue
            seen_records.add(key)
            reasons = screen_conversation(turns)
            if reasons:
                for reason in reasons:
                    rejected[reason] += 1
                rejected_rows.append({'source_record': line_number, 'reasons': reasons})
                continue
            examples, error = make_examples(f'durecdial2-{source_split}-{line_number}', turns, tokenizer, context)
            if error:
                rejected[error] += 1
                rejected_rows.append({'source_record': line_number, 'reasons': [error]})
                continue
            accepted.extend(examples)
            accepted_dialogues += 1
    return accepted, {'source_records': records, 'accepted_dialogues': accepted_dialogues,
        'accepted_turn_pairs': len(accepted), 'rejected': dict(rejected),
                 'rejected_records': rejected_rows,
                 'source_split': source_split, 'archive_member': member}


def prepare(root, context=512):
    output = root / 'dialogue'
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new immutable cycle root.')
    metadata = json.loads((ARCHIVE_META).read_text(encoding='utf-8'))
    archive_hash = checksum(ARCHIVE)
    if archive_hash != metadata['archive_sha256']:
        raise RuntimeError('DuRecDial archive hash mismatch.')
    tokenizer = spm.SentencePieceProcessor(model_proto=TOKENIZER.read_bytes())

    train, train_report = parse_split(ARCHIVE, 'data/zh_train.txt', tokenizer, context, 'train')
    heldout, heldout_report = parse_split(ARCHIVE, 'data/zh_dev.txt', tokenizer, context, 'dev')
    # The publisher's official test split is not opened, parsed or copied.
    train_records = {}
    for row in train:
        train_records.setdefault(row['conversation_id'], []).append(row)
    dev_records = {}
    for row in heldout:
        dev_records.setdefault(row['conversation_id'], []).append(row)
    ordered = sorted(dev_records, key=lambda value: hashlib.sha256(('durecdial2-dev:' + value).encode()).hexdigest())
    cut = max(1, len(ordered) // 2)
    validation_ids, development_ids = set(ordered[:cut]), set(ordered[cut:])

    train_pairs = {hashlib.sha256((r['prompt'] + '\0' + r['answer']).encode()).hexdigest() for r in train}
    development = [r for r in heldout if r['conversation_id'] in development_ids]
    validation = [r for r in heldout if r['conversation_id'] in validation_ids]
    development = [r for r in development if hashlib.sha256((r['prompt'] + '\0' + r['answer']).encode()).hexdigest() not in train_pairs]
    validation = [r for r in validation if hashlib.sha256((r['prompt'] + '\0' + r['answer']).encode()).hexdigest() not in train_pairs]

    output.mkdir(parents=True)
    write_rows(output / 'train.jsonl', train)
    write_rows(output / 'development.jsonl', development)
    write_rows(output / 'validation.jsonl', validation)
    review_log = root / 'screening-rejections.json'
    write_json(review_log, {
        'kind': 'automated_screening_rejections_v1',
        'contains_source_text': False,
        'records': {
            'train': train_report.pop('rejected_records'),
            'source_dev': heldout_report.pop('rejected_records'),
        },
    })
    files = {s: checksum(output / f'{s}.jsonl') for s in ('train', 'development', 'validation')}
    manifest = {
        'kind': 'durecdial2_privacy_screened_sft_v1',
        'source_metadata_sha256': checksum(ARCHIVE_META), 'source_archive_sha256': archive_hash,
        'source_repository_commit': metadata['repository_commit'],
        'dataset_license': metadata['license'], 'user_noncommercial_confirmation': True,
        'profile_fields_discarded': True, 'profile_fields_serialized': False,
        'official_test_split_opened': False, 'official_test_split_used': False,
        'tokenizer_sha256': checksum(TOKENIZER), 'context_tokens': context,
        'counts': {'train': len(train), 'development': len(development), 'validation': len(validation)},
        'source_reports': {'train': train_report, 'dev': heldout_report},
        'development_source': 'half of the publisher development split grouped by whole dialogue record',
        'validation_source': 'half of the publisher development split grouped by whole dialogue record',
        'files': files,
        'screening_rejections_sha256': checksum(review_log),
        'screening_method': 'automated pattern and structure filters; not item-level human review',
        'limitations': [
            'This is a recommendation dialogue corpus; it is not a general-purpose pet assistant corpus.',
            'Pattern-based privacy and content screening cannot guarantee that all personal or unsafe content was removed.',
            'Assistant screening is not a new independent human item-level review.',
            'The non-commercial share-alike license constrains downstream use and redistribution.',
        ],
    }
    write_json(output / 'manifest.json', manifest)

    review_dir = root / 'review'
    review_dir.mkdir()
    for split, rows in (('train', train), ('development', development), ('validation', validation)):
        # Review indexes contain hashes, text and source split only; never profile fields.
        write_rows(review_dir / f'{split}-review.jsonl', ({
            'id': row['id'], 'conversation_id': row['conversation_id'], 'split': split,
            'prompt': row['prompt'], 'answer': row['answer'], 'decision': 'pending',
            'privacy_review': '', 'content_review': '', 'naturalness': '',
            'factuality': '', 'reviewer_id': '', 'reviewer_signature': '',
        } for row in rows))

    count_gate = len(train) >= 200 and len(development) >= 20 and len(validation) >= 20
    write_json(root / 'status.json', {
        'kind': 'durecdial2_sft_status_v1',
        'status': 'prepared_pending_item_review' if count_gate else 'insufficient_data_after_screening',
        'counts': manifest['counts'], 'training_started': False, 'sft_started': False,
        'profile_discarded': True, 'profile_fields_written': False,
        'official_test_used': False, 'count_gate_passed': count_gate,
        'local_item_review_complete': False,
        'next_action': 'Automated screening only. Sample and review all retained records or obtain independent item review before SFT.'
    })
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--context', type=int, default=512)
    args = parser.parse_args()
    prepare(args.root, args.context)


if __name__ == '__main__':
    main()
