"""Build a review-first LCCC dialogue candidate package from pinned local files."""
import argparse
import gzip
import hashlib
import heapq
import json
import re
from pathlib import Path

from .data import BASE
from .run import checksum


RAW = BASE / 'dialogue/raw/lccc-5bd582f'
DEFAULT_ROOT = BASE / 'dialogue/lccc-review-v1'
REVISION = '5bd582fa28cd7143f2f9c852e08e23089d677c44'
LICENSE = 'MIT (Hugging Face dataset card); upstream source/privacy provenance is incomplete.'
CJK = re.compile(r'[\u3400-\u9fff]')
PHONE = re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)')
EMAIL = re.compile(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', re.I)
CN_ID = re.compile(r'(?<!\d)\d{17}[\dXx](?!\d)')
CARD = re.compile(r'(?<!\d)(?:\d[ -]?){15,19}(?!\d)')
PII = re.compile(
    r'我叫[\u3400-\u9fff]{2,4}|本人叫|(?:我|你|他|她|我的|你的|他的|她的)?(?:手机号|电话号码|微信号|QQ号)|'
    r'(?:家庭住址|详细地址|住址是|门牌号|身份证号|银行卡号|护照号|账号密码|验证码)')
RISK = re.compile(
    r'自杀|自残|轻生|毒品|制毒|炸弹|爆炸物|枪支|色情|裸聊|性交易|强奸|'
    r'杀人|人肉搜索|跟踪定位|开盒|赌博|博彩|赌球|诈骗|自制武器')
ABUSE = re.compile(r'操你|他妈的|傻逼|傻B|狗杂种|滚你妈|去死|贱人|废物|脑残|fuck|shit', re.I)
PROMO = re.compile(r'加微信|加\s*qq|扫码|关注公众号|点击下载|优惠券|博彩|兼职刷单|贷款广告', re.I)
URL = re.compile(r'https?://|www\.|\b\S+@\S+')


def normalize_turn(turn):
    text = re.sub(r'\s+', ' ', str(turn)).strip()
    text = re.sub(r'(?<=[\u3400-\u9fff])\s+|\s+(?=[\u3400-\u9fff，。！？；：、,.!?])', '', text)
    return text.strip(' \t\r\n，,。.!！?？')


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def reject_reason(turns):
    text = '\n'.join(turns)
    if any(p.search(text) for p in (PHONE, EMAIL, CN_ID, CARD, PII)):
        return 'possible_personal_or_contact_information'
    if RISK.search(text):
        return 'high_risk_or_age_inappropriate'
    if ABUSE.search(text):
        return 'abusive_or_profane'
    if PROMO.search(text) or URL.search(text):
        return 'spam_or_external_contact'
    if len(CJK.findall(text)) / max(1, len(text)) < 0.55:
        return 'low_chinese_ratio'
    if any(len(turn) < 2 or len(turn) > 260 for turn in turns):
        return 'turn_length'
    if any(re.search(r'(.)\1{5,}', turn) for turn in turns):
        return 'repeated_character_noise'
    return None


def pairs_from_dialogue(raw):
    if not isinstance(raw, list) or len(raw) < 4 or len(raw) % 2:
        return None, 'malformed_turn_count'
    turns = [normalize_turn(t) for t in raw]
    if not all(turns):
        return None, 'empty_turn'
    reason = reject_reason(turns)
    if reason:
        return None, reason
    # One final user/assistant response per source dialogue prevents prolific
    # conversations from dominating the candidate set.
    prompt_turns = turns[-6:-1]
    if len(prompt_turns) < 3 or len(prompt_turns) % 2 == 0:
        return None, 'insufficient_context'
    if len(prompt_turns) >= 2 and len(prompt_turns[-1]) < 4:
        return None, 'weak_final_user_turn'
    prompt = ''.join(f'{"用户" if i % 2 == 0 else "助手"}：{turn}\n'
                     for i, turn in enumerate(prompt_turns)) + '助手：'
    answer = turns[-1]
    if len(answer) > 120:
        return None, 'answer_too_long'
    return {'prompt': prompt, 'answer': answer, 'turn_count': len(turns)}, None


def priority(split, line_number, size):
    key = digest(f'lccc:{REVISION}:{split}:{line_number}')
    value = int(key[:16], 16)
    return (-value, key)


def add_candidate(heap, limit, candidate):
    priority_value = int(candidate['_priority'][:16], 16)
    entry = (-priority_value, candidate['_priority'], candidate)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)


def sample_split(path, split, limit, reject_counts):
    heap = []
    read = 0
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle):
            if not line.strip():
                continue
            read += 1
            try:
                raw = json.loads(line)
            except Exception:
                reject_counts[split]['malformed_json'] = reject_counts[split].get('malformed_json', 0) + 1
                continue
            pair, reason = pairs_from_dialogue(raw)
            if reason:
                reject_counts[split][reason] = reject_counts[split].get(reason, 0) + 1
                continue
            conversation_id = digest(split + ':' + str(line_number) + ':' + '\n'.join(map(normalize_turn, raw)))
            p = digest(f'lccc:{REVISION}:{split}:{line_number}')
            pair.update({'conversation_id': conversation_id, 'source_line': line_number,
                         'source_split': split, '_priority': p})
            add_candidate(heap, limit, pair)
    chosen = [item[2] for item in heap]
    chosen.sort(key=lambda item: item['_priority'])
    return chosen, read


def review_row(candidate, split):
    identifier = digest(candidate['conversation_id'] + ':' + split)
    return {
        'id': identifier, 'conversation_id': candidate['conversation_id'],
        'split': split, 'prompt': candidate['prompt'], 'answer': candidate['answer'],
        'source': 'silver/lccc', 'source_revision': REVISION, 'source_license': LICENSE,
        'decision': 'pending', 'privacy': '', 'safety_age_suitability': '',
        'naturalness': '', 'relevance': '', 'reviewer_id': '', 'reviewer_signature': '',
    }


def row_key(row, field):
    return digest(re.sub(r'\s+', '', row[field]).casefold())


def build(root, train_count=2400, eval_count=300):
    if root.exists():
        raise FileExistsError(f'{root} exists; choose a new immutable review cycle.')
    train_path = RAW / 'lccc_base_train.jsonl.gz'
    valid_path = RAW / 'lccc_base_valid.jsonl.gz'
    if not train_path.is_file() or not valid_path.is_file():
        raise FileNotFoundError('Pinned LCCC train and validation archives are required.')
    rejects = {'train': {}, 'validation': {}}
    train, train_read = sample_split(train_path, 'train', train_count, rejects)
    validation, valid_read = sample_split(valid_path, 'validation', eval_count * 2, rejects)
    if len(train) < train_count or len(validation) < eval_count * 2:
        raise RuntimeError('Not enough screened LCCC samples for the requested review cycle.')
    development = validation[:eval_count]
    internal_test = validation[eval_count:]
    # Holdouts have priority: remove any normalized prompt or pair duplicates
    # from development and training while keeping the internal test untouched.
    test_prompt_keys = {row_key(row, 'prompt') for row in internal_test}
    test_pair_keys = {digest(re.sub(r'\s+', '', row['prompt'] + '\0' + row['answer']).casefold())
                      for row in internal_test}
    dev_prompt_seen = set(test_prompt_keys)
    dev_pair_seen = set(test_pair_keys)
    kept_development = []
    for row in development:
        prompt_key = row_key(row, 'prompt')
        pair_key = digest(re.sub(r'\s+', '', row['prompt'] + '\0' + row['answer']).casefold())
        if prompt_key in dev_prompt_seen or pair_key in dev_pair_seen:
            rejects['validation']['cross_holdout_duplicate_removed'] = rejects['validation'].get('cross_holdout_duplicate_removed', 0) + 1
            continue
        dev_prompt_seen.add(prompt_key)
        dev_pair_seen.add(pair_key)
        kept_development.append(row)
    development = kept_development
    holdout_prompt_keys = dev_prompt_seen
    holdout_pair_keys = dev_pair_seen
    kept_train = []
    seen_train_prompts = set(holdout_prompt_keys)
    seen_train_pairs = set(holdout_pair_keys)
    for row in train:
        prompt_key = row_key(row, 'prompt')
        pair_key = digest(re.sub(r'\s+', '', row['prompt'] + '\0' + row['answer']).casefold())
        if prompt_key in seen_train_prompts or pair_key in seen_train_pairs:
            rejects['train']['holdout_overlap_removed'] = rejects['train'].get('holdout_overlap_removed', 0) + 1
            continue
        seen_train_prompts.add(prompt_key)
        seen_train_pairs.add(pair_key)
        kept_train.append(row)
    train = kept_train
    root.mkdir(parents=True)
    for split, candidates in (('train', train), ('development', development),
                              ('internal_test', internal_test)):
        with (root / f'{split}-review.jsonl').open('w', encoding='utf-8', newline='\n') as handle:
            for candidate in candidates:
                handle.write(json.dumps(review_row(candidate, split), ensure_ascii=False) + '\n')
    files = {p.name: checksum(p) for p in root.glob('*-review.jsonl')}
    manifest = {
        'kind': 'lccc_human_review_candidate_cycle_v1',
        'source': 'silver/lccc', 'revision': REVISION, 'license': LICENSE,
        'dataset_card_url': f'https://huggingface.co/datasets/silver/lccc/tree/{REVISION}',
        'source_privacy_note': 'The dataset card does not document personal/sensitive-information handling; local automated filters are not a guarantee.',
        'raw_train_sha256': checksum(train_path), 'raw_validation_sha256': checksum(valid_path),
        'official_test_opened': False, 'official_test_downloaded': False,
        'read_records': {'train': train_read, 'validation': valid_read},
        'candidate_counts': {'train': len(train), 'development': len(development),
                             'internal_test': len(internal_test)},
        'automated_rejections': rejects,
        'selection': {'one_response_per_source_dialogue': True,
                      'train_from_source_train_only': True,
                      'development_and_internal_test_from_source_validation_only': True,
                      'validation_subsplit_by_hash_priority': True,
                      'max_answer_characters': 120, 'max_turn_characters': 260},
        'files': files,
        'review_status': 'pending_item_level_human_review',
        'sft_started': False,
        'limitations': [
            'MIT metadata in the dataset mirror is not a complete provenance/privacy audit of upstream social-media text.',
            'Automated PII and safety filters can miss identifying or harmful content.',
            'No candidate is approved until a human reviews every retained prompt and answer.',
            'Internal-test candidates must not be used for fitting or model selection.',
        ],
    }
    (root / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    status = {'status': 'pending_item_level_human_review',
              'counts': manifest['candidate_counts'], 'sft_started': False,
              'official_test_opened': False, 'review_complete': False,
              'next_action': 'Review every train/development/internal_test row; only approved train rows may enter SFT.'}
    (root / 'status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'counts': manifest['candidate_counts'], 'read': manifest['read_records'],
                      'rejections': rejects, 'files': files}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--train-count', type=int, default=2400)
    parser.add_argument('--eval-count', type=int, default=300)
    args = parser.parse_args()
    build(args.root, args.train_count, args.eval_count)


if __name__ == '__main__':
    main()
