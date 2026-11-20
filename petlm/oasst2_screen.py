"""Prepare a privacy/content-screened OASST2 SFT cycle without reading test data."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import sentencepiece as spm

from .data import BASE
from .run import checksum


SOURCE = BASE / 'dialogue/prepared-oasst2-chinese-v17-all-reviewed'
PRETRAINED_DIALOGUE = BASE / 'dialogue/prepared-human-openassistant-v1/train.jsonl'
DEFAULT_ROOT = BASE / 'experiments/oasst2-screened-cycle-v3'
PHONE = re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)')
EMAIL = re.compile(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', re.I)
CN_ID = re.compile(r'(?<!\d)\d{17}[\dXx](?!\d)')
BANK_CARD = re.compile(r'(?<!\d)(?:\d[ -]?){15,19}(?!\d)')
PERSONAL = re.compile(
    r'(?:我叫[\u3400-\u9fff]{2,4}(?:[,，。！!？?]|$)|本人叫[\u3400-\u9fff]{2,4}|'
    r'我的(?:姓名|名字|手机号|电话号码|微信号|QQ号|身份证号|银行卡号)|'
    r'家庭住址|住址是|详细地址|家庭地址|我住在.{0,10}(?:街|路|小区|号)|'
    r'出生日期|身份证号|护照号|账号密码|验证码|门牌号)')
PRIVATE = re.compile(r'(?:密码|验证码|银行卡号|身份证号|护照号|私密照片|隐私地址)', re.I)
RISK = re.compile(
    r'自杀|自残|轻生|制毒|炸弹制作|枪支制作|色情|裸聊|性交易|强奸|'
    r'杀人方法|人肉搜索|跟踪定位|开盒|仇恨言论|种族灭绝', re.I)
SENSITIVE = re.compile(
    r'诊断|处方|用药剂量|药物治疗|法律意见|诉讼策略|选举操纵|竞选捐款', re.I)
PROMO = re.compile(r'扫码|加微信|点击下载|关注公众号|广告|充值|优惠券', re.I)


def read_jsonl(path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def norm_hash(text):
    normalized = re.sub(r'\s+', '', text).casefold()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def screen(row):
    text = row.get('prompt', '') + '\n' + row.get('answer', '')
    reasons = []
    if any(p.search(text) for p in (PHONE, EMAIL, CN_ID, BANK_CARD, PERSONAL)):
        reasons.append('possible_personal_or_contact_information')
    if PRIVATE.search(text):
        reasons.append('credential_or_private_detail')
    if RISK.search(text):
        reasons.append('high_risk_or_age_inappropriate')
    if SENSITIVE.search(text):
        reasons.append('sensitive_domain_excluded')
    if PROMO.search(text):
        reasons.append('spam_or_promotion')
    if len(text) > 9000:
        reasons.append('overlong_context')
    if len(re.findall(r'用户：', row.get('prompt', ''))) < 2:
        reasons.append('not_multiturn')
    if not row.get('answer', '').strip():
        reasons.append('empty_answer')
    return reasons


def fit_context(row, tokenizer, context=512):
    turns = [part.strip() for part in re.split(
        r'(?=^(?:用户|助手)：)', row['prompt'], flags=re.MULTILINE) if part.strip()]
    answer_ids = tokenizer.encode(row['answer']) + [tokenizer.eos_id()]
    while turns:
        while turns and turns[0].startswith('助手：'):
            turns.pop(0)
        prompt = '\n'.join(turns)
        if prompt and len(tokenizer.encode(prompt)) + len(answer_ids) <= context:
            fitted = dict(row)
            fitted['prompt'] = prompt
            return fitted
        if turns:
            turns.pop(0)
    return None


def prepare(root):
    if root.exists():
        raise FileExistsError(f'{root} exists; choose a new immutable cycle root.')
    source_manifest = json.loads((SOURCE / 'manifest.json').read_text(encoding='utf-8'))
    tokenizer_path = BASE / 'tokens-chinese-v8-50m-frozen/tokenizer.model'
    tokenizer = spm.SentencePieceProcessor(model_proto=tokenizer_path.read_bytes())
    pretrain_trees = set()
    for row in read_jsonl(PRETRAINED_DIALOGUE):
        if row.get('source_dataset') == 'OpenAssistant/oasst2':
            if row.get('tree_id'):
                pretrain_trees.add(row['tree_id'])

    rows_by_split = {'train': [], 'validation': []}
    all_rows = defaultdict(list)
    rejected = {s: Counter() for s in rows_by_split}
    source_trees = {s: set() for s in rows_by_split}
    for split in rows_by_split:
        for raw in read_jsonl(SOURCE / f'{split}.jsonl'):
            tree = raw.get('tree_id')
            if not tree:
                rejected[split]['missing_tree_id'] += 1
                continue
            source_trees[split].add(tree)
            reasons = screen(raw)
            if raw.get('source_review_result') is not True:
                reasons.append('source_review_not_passed')
            if (raw.get('source_quality') or 0) < 0.5 or (raw.get('source_helpfulness') or 0) < 0.5:
                reasons.append('source_quality_below_threshold')
            if reasons:
                for reason in set(reasons):
                    rejected[split][reason] += 1
            else:
                all_rows[split].append(raw)

    if source_trees['train'] & source_trees['validation']:
        raise RuntimeError('Source train/validation tree leakage detected.')

    for split, rows in all_rows.items():
        for row in rows:
            if row['tree_id'] in pretrain_trees:
                rejected[split]['parent_pretraining_tree_overlap'] += 1
            else:
                fitted = fit_context(row, tokenizer)
                if fitted is None:
                    rejected[split]['context_overflow'] += 1
                else:
                    rows_by_split[split].append(fitted)

    # Validation takes priority; identical prompts cannot cross into training.
    val_prompts = {norm_hash(r['prompt']) for r in rows_by_split['validation']}
    val_pairs = {norm_hash(r['prompt'] + '\0' + r['answer']) for r in rows_by_split['validation']}
    dedup_validation = []
    seen = set()
    for row in rows_by_split['validation']:
        key = norm_hash(row['prompt'] + '\0' + row['answer'])
        if key in seen:
            rejected['validation']['duplicate_pair'] += 1
        else:
            seen.add(key)
            dedup_validation.append(row)
    rows_by_split['validation'] = dedup_validation
    dedup_train = []
    seen_train = set()
    for row in rows_by_split['train']:
        prompt_key = norm_hash(row['prompt'])
        pair_key = norm_hash(row['prompt'] + '\0' + row['answer'])
        if prompt_key in val_prompts or pair_key in val_pairs:
            rejected['train']['validation_overlap_removed'] += 1
        elif pair_key in seen_train:
            rejected['train']['duplicate_pair'] += 1
        else:
            seen_train.add(pair_key)
            dedup_train.append(row)
    rows_by_split['train'] = dedup_train

    root.mkdir(parents=True)
    for split, rows in rows_by_split.items():
        # Whitelist serialized fields: no profile, raw source metadata, or source text beyond the pair.
        write_jsonl(root / f'{split}.jsonl', ({
            'id': row['id'], 'conversation_id': row['tree_id'],
            'prompt': row['prompt'], 'answer': row['answer'],
            'training_group': 'natural_human_oasst2',
            'source': 'OpenAssistant/oasst2', 'source_revision': source_manifest['revision'],
            'source_license': source_manifest['license'],
        } for row in rows))

    counts = {s: len(rows) for s, rows in rows_by_split.items()}
    output_hashes = {s: checksum(root / f'{s}.jsonl') for s in rows_by_split}
    manifest = {
        'kind': 'oasst2_privacy_screened_multiturn_sft_v1',
        'source': source_manifest['source'], 'source_revision': source_manifest['revision'],
        'source_license': source_manifest['license'],
        'source_manifest_sha256': checksum(SOURCE / 'manifest.json'),
        'source_train_sha256': checksum(SOURCE / 'train.jsonl'),
        'source_validation_sha256': checksum(SOURCE / 'validation.jsonl'),
        'tokenizer_sha256': source_manifest['tokenizer_sha256'],
        'context_tokens': 512,
        'source_test_opened': False, 'source_test_used': False,
        'parent_pretraining_dialogue_sha256': checksum(PRETRAINED_DIALOGUE),
        'parent_overlap_policy': 'Remove every source tree seen in parent pretraining dialogue rows.',
        'profile_fields_serialized': False,
        'screening': 'Automated PII/contact, sensitive-content, source-rating, length, and multi-turn checks over each full prompt/answer context; not local item-level human review.',
        'split_policy': 'Source splits retained by whole tree; validation-priority prompt and pair deduplication.',
        'counts': counts, 'rejected_row_counts': {s: dict(c) for s, c in rejected.items()},
        'files': output_hashes,
        'limitations': [
            'Automated PII rules cannot guarantee removal of all identifying information.',
            'OASST2 source ratings and review labels are not a new local human audit.',
            'General assistant dialogue is not a pet-specific behavior dataset.',
            'Apache-2.0 attribution and notices must be preserved when redistributing derived data.',
        ],
    }
    write_json(root / 'manifest.json', manifest)
    eligible = counts['train'] >= 200 and counts['validation'] >= 30
    write_json(root / 'status.json', {
        'status': 'ready_for_exploratory_sft' if eligible else 'insufficient_data_after_screening',
        'counts': counts, 'profile_fields_serialized': False,
        'source_test_opened': False, 'parent_overlap_removed': True,
        'automated_screening_only': True, 'local_item_review_complete': False,
        'sft_started': False, 'count_gate_passed': eligible,
    })
    print(json.dumps({'counts': counts, 'rejected': manifest['rejected_row_counts'],
                      'files': output_hashes, 'ready': eligible}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    prepare(args.root)


if __name__ == '__main__':
    main()
