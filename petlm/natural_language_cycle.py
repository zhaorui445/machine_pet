"""Natural-prose pretraining and natural-dialogue SFT in a separate experiment.

No frozen conversational evaluation material is opened. Source-level feedback is
not relabeled as a new independent human audit. Original checkpoints stay intact.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from types import SimpleNamespace
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/petlm'


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


PARENT = BASE / 'models/chinese-v8-97m-extended-24k/latest.pt'
TOKENIZER = BASE / 'tokens-chinese-v8-50m-frozen/tokenizer.model'
OLD_DATA = BASE / 'prepared-chinese-v8-50m-frozen'
SOURCE = BASE / 'prepared-chinese-v18-strict2'
DEFAULT = BASE / 'experiments/natural-language-cycle-v5'
CJK = re.compile(r'[\u3400-\u9fff]')
ARTIFACT = re.compile(r'https?://|www\.|\[\[|\{\{|\}\}|File:|Category:|\d+px\||[<>|｜]|用户：|助手：|\(\s*\)|（\s*）', re.I)
PROMOTION = re.compile(r'扫码|二维码|微信公众号|联系电话|联系地址|客服电话|点击下载|下载附件|上一篇|下一篇|网友评论|责任编辑|版权所有|转载声明|开通会员|免费领取|广告位|加微信|博彩|百家乐|老虎机|投注|色情|裸聊')
END = re.compile(r'[^。！？!?]+[。！？!?][”’」』）)]*')
TOPICS = {
    'daily_life': ('饮食', '烹饪', '蔬菜', '水果', '豆腐', '米饭', '厨房', '食材', '卫生', '洗手', '作息', '睡眠', '运动', '锻炼', '收纳', '房间'),
    'learning': ('学习', '阅读', '写作', '作业', '解题', '课堂', '课程', '学生', '教育', '教学', '科学实验', '观察', '测量'),
    'basic_science': ('植物', '光合', '叶片', '根系', '树木', '种子', '昆虫', '鸟类', '动物', '生态', '水分', '土壤', '温度', '热量', '蒸发', '凝结', '融化', '熔化', '结冰', '天气', '降雨', '雨水', '气候', '海洋', '河流', '毛细', '摩擦', '重力', '浮力', '声音', '振动', '光线', '反射', '折射', '电流', '电路'),
    'arts_and_play': ('绘画', '画画', '色彩', '音乐', '旋律', '节奏', '故事', '积木', '纸飞机', '手工', '剪纸', '书籍'),
}


def prose_topic(text):
    # This is a scoped language-training mixture, not a general encyclopedia.
    if re.search(r'政治局|总理|军队|殖民|王朝|朝廷|内阁|教皇|先知|菩萨|比丘|世尊|吸血鬼|死刑|枪支|炮弹|狙击|阿弥陀|菩提|活活烧死', text):
        return None
    counts = {name: sum(word in text for word in words) for name, words in TOPICS.items()}
    winner = max(counts, key=counts.get)
    return winner if counts[winner] >= 2 else None


def read_rows(path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_rows(path, rows):
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def key(text):
    normalized = re.sub(r'[^\w\u3400-\u9fff]', '', text).casefold()
    return hashlib.sha256(normalized.encode()).hexdigest()


def anchors(text):
    compact = re.sub(r'[^\w\u3400-\u9fff]', '', text).casefold()
    result = {key(text)}
    if len(compact) >= 160:
        result.add(key(compact[:160]))
    for sentence in END.findall(text):
        if len(CJK.findall(sentence)) >= 56:
            result.add(key(sentence))
    return result


def prose_segments(text):
    """Keep complete sentence groups; never cut a sentence at a fixed character."""
    current = []
    size = 0
    for line in text.splitlines():
        line = re.sub(r'\[\d{1,3}\]', '', line).strip()
        if len(line) < 24 or ARTIFACT.search(line) or PROMOTION.search(line):
            if current:
                yield ''.join(current)
                current, size = [], 0
            continue
        for sentence in END.findall(line):
            if size + len(sentence) > 1800 and current:
                yield ''.join(current)
                current, size = [], 0
            current.append(sentence.strip())
            size += len(sentence)
        if size >= 300:
            yield ''.join(current)
            current, size = [], 0
    if current:
        yield ''.join(current)


def prose_rejection(text):
    if not 150 <= len(text) <= 2200:
        return 'length'
    if ARTIFACT.search(text) or PROMOTION.search(text):
        return 'artifact_or_promotion'
    if re.search(r'copyright|未经.{0,8}(?:许可|允许).{0,12}(?:转载|复制)|ISBN|\-\{|\}\-', text, re.I):
        return 'copyright_bibliography_or_markup'
    if re.search(r'\[(?:DB|OL|J|M|EB)[/A-Z]*\]|\[\d{4}\]|(?:拟音|学名)\s*[:：]\s*[)）]|\([^\w\u3400-\u9fff)]{1,8}\)', text):
        return 'reference_or_missing_pronunciation'
    if re.search(r'阴茎|阴道|性交|射精|丰乳|翘臀|他妈的|穷B|肉毒杆菌|整形美容|注射美容|网络斗地主|热刺vs', text, re.I):
        return 'adult_spam_or_cosmetic_ad'
    if re.search(r'[,，]\s*[,，]|(?:表示为|符号表示|如|给定两点)\s*[,，。]|[：:]\s*[,，。]', text):
        return 'missing_formula_or_text'
    if re.search(r'介于\s*(?:之间|[,，。;；])|(?:可达到|降水量约为|称为)\s*[,，。;；]|[。！？]\s*[,，]', text):
        return 'missing_value_or_term'
    if len(re.findall(r'[,，、]', text)) / len(text) > .115:
        return 'enumeration_heavy'
    if re.search(r'出版社|(?:于|於)\d{4}年\d{1,2}月\d{1,2}日查阅', text):
        return 'inline_bibliography'
    if len(re.findall(r'[兮矣焉朕汝吾曰]', text)) / len(text) > .008:
        return 'classical_quotation_heavy'
    sentences = END.findall(text)
    if len(sentences) < 3 or not re.search(r'[。！？!?][”’」』）)]*$', text):
        return 'incomplete_prose'
    if len(CJK.findall(text)) / len(text) < .67:
        return 'low_chinese'
    if len(re.findall(r'\d', text)) / len(text) > .055:
        return 'number_heavy'
    if len(re.findall(r'[A-Za-z]', text)) / len(text) > .08:
        return 'latin_heavy'
    if sum(len(s) < 10 or len(s) > 240 for s in sentences) / len(sentences) > .3:
        return 'sentence_layout'
    compact = re.sub(r'\s+', '', text)
    shingles = Counter(compact[i:i+12] for i in range(0, len(compact)-11, 3))
    repetition = sum(n-1 for n in shingles.values()) / max(1, sum(shingles.values()))
    if repetition > .08:
        return 'repetition'
    return None


class Groups:
    def __init__(self):
        self.parents = {}

    def find(self, node):
        self.parents.setdefault(node, node)
        if self.parents[node] != node:
            self.parents[node] = self.find(self.parents[node])
        return self.parents[node]

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        self.parents[max(a, b)] = min(a, b)


def prepare(root):
    output = root / 'prepared'
    if output.exists():
        raise FileExistsError(output)
    root.mkdir(parents=True, exist_ok=True)
    old_anchors = defaultdict(set)
    for split in ('train', 'validation', 'test'):
        for row in read_rows(OLD_DATA / f'{split}.jsonl'):
            for anchor in anchors(row['text']):
                old_anchors[anchor].add(split)
    groups = Groups()
    anchor_owner = {}
    old_flags = defaultdict(set)
    candidates = []
    rejected = Counter()
    considered = 0
    # Only the newer source's TRAIN partition is a candidate pool.
    for row in read_rows(SOURCE / 'train.jsonl'):
        if row['source'] not in ('wikipedia', 'fineweb2'):
            rejected['dialogue_excluded_from_pretraining'] += 1
            continue
        considered += 1
        node = row['id']
        groups.find(node)
        for anchor in anchors(row['text']):
            old_flags[node].update(old_anchors.get(anchor, ()))
            if anchor in anchor_owner:
                groups.union(node, anchor_owner[anchor])
            else:
                anchor_owner[anchor] = node
        kept = 0
        for segment in prose_segments(row['text']):
            reason = prose_rejection(segment)
            if reason:
                rejected[reason] += 1
                continue
            candidates.append({'document_id': node, 'source': row['source'], 'text': segment})
            kept += 1
            if kept == 8:
                break  # no single long article should dominate the corpus
    component_flags = defaultdict(set)
    for node in groups.parents:
        component_flags[groups.find(node)].update(old_flags[node])
    rows = []
    seen_text, seen_prefix = set(), set()
    split_groups = defaultdict(set)
    for row in candidates:
        group = groups.find(row['document_id'])
        flags = component_flags[group]
        if flags & {'validation', 'test'}:
            rejected['parent_holdout_or_shared_long_sentence'] += 1
            continue
        bucket = int(key('natural-cycle-v1:' + group)[:8], 16) % 100
        split = 'train' if flags or bucket >= 15 else ('test' if bucket < 5 else 'validation')
        full = key(row['text'])
        prefix = key(re.sub(r'\W', '', row['text'])[:160])
        if full in seen_text or prefix in seen_prefix:
            rejected['duplicate_segment_or_prefix'] += 1
            continue
        seen_text.add(full)
        seen_prefix.add(prefix)
        split_groups[split].add(group)
        rows.append(dict(row, id=full, split=split, group_id=group,
                         parent_corpus_overlap=bool(flags)))
    for left, right in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        if split_groups[left] & split_groups[right]:
            raise RuntimeError('Document groups crossed partitions.')
    counts = Counter(row['split'] for row in rows)
    if min(counts.values(), default=0) < 100 or len(counts) != 3:
        raise RuntimeError(f'Insufficient independent natural prose: {dict(counts)}')
    output.mkdir()
    for split in ('train', 'validation', 'test'):
        write_rows(output / f'{split}.jsonl', (r for r in rows if r['split'] == split))
    ids = {r['document_id'] for r in rows}
    write_rows(output / 'provenance.jsonl',
               (r for r in read_rows(SOURCE / 'provenance.jsonl') if r['id'] in ids))
    report = {
        'kind': 'natural_chinese_prose_cycle_v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'source': str(SOURCE), 'source_train_sha256': checksum(SOURCE / 'train.jsonl'),
        'source_manifest_sha256': checksum(SOURCE / 'manifest.json'),
        'parent_corpus': str(OLD_DATA),
        'parent_corpus_sha256': {s: checksum(OLD_DATA / f'{s}.jsonl') for s in ('train', 'validation', 'test')},
        'counts': dict(counts), 'sources': dict(Counter(r['source'] for r in rows)),
        'characters': {s: sum(len(r['text']) for r in rows if r['split'] == s) for s in counts},
        'documents_considered': considered, 'rejected': dict(rejected),
        'files': {s: checksum(output / f'{s}.jsonl') for s in counts},
        'cross_split_document_group_overlap': 0,
        'holdout_parent_overlap_by_record_prefix_and_long_sentence': 0,
        'policy': 'Complete Chinese sentence groups; max 8 segments per document; normalized exact/prefix dedup; document components share full-text, prefix or long-sentence anchors. Parent holdout components excluded; parent-seen components restricted to train.',
        'independent_human_review_complete': False, 'old_conversation_blind_used': False,
        'limitations': ['Automatic natural-prose screening, not semantic or factual certification.',
                       'Anchor-based duplicate grouping is not exhaustive semantic near-deduplication.',
                       'Holdout novelty checked against this parent corpus, not every historical model.',
                       'Existing corpus reuse improves composition; no new external documents downloaded.'],
    }
    write_json(output / 'manifest.json', report)
    # Deterministic inspection sample, with no fabricated human approval.
    sample = sorted((r for r in rows if r['split'] == 'train'), key=lambda r: key('sample:' + r['id']))[:24]
    write_rows(root / 'prose-inspection-sample.jsonl', sample)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def tokenize(root):
    from .run import tokenize as run_tokenize
    run_tokenize(SimpleNamespace(data=root / 'prepared', tokens=root / 'tokens',
                                tokenizer_from=TOKENIZER, vocab_size=16000,
                                byte_fallback=True, character_coverage=.9995))
    from .corpus_verify import verify_tokens
    result = verify_tokens(root / 'tokens', root / 'prepared')
    write_json(root / 'token-verification.json', result)
    return result


def refine(root):
    source = BASE / 'experiments/natural-language-cycle-v1/prepared'
    output = root / 'prepared'
    if output.exists():
        raise FileExistsError(output)
    root.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    trusted_web = set()
    trusted_domains = ('gov.cn', 'edu.cn', 'ac.cn', 'who.int', 'news.cn',
                       'xinhuanet.com', 'people.com.cn', 'kepuchina.cn')
    for entry in read_rows(source / 'provenance.jsonl'):
        host = (urlparse(entry.get('origin', '')).hostname or '').casefold()
        if entry['source'] == 'fineweb2' and any(host == d or host.endswith('.' + d) for d in trusted_domains):
            trusted_web.add(entry['id'])
    rows = []
    rejected = Counter()
    for split in ('train', 'validation', 'test'):
        for row in read_rows(source / f'{split}.jsonl'):
            reason = ('web_origin_not_in_current_allowlist'
                      if row['source'] == 'fineweb2' and row['document_id'] not in trusted_web
                      else prose_rejection(row['text']))
            topic = prose_topic(row['text'])
            if not reason and topic is None:
                reason = 'outside_daily_learning_science_prose_mixture'
            if reason:
                rejected[reason] += 1
            else:
                rows.append(dict(row, topic=topic))
        write_rows(output / f'{split}.jsonl', (r for r in rows if r['split'] == split))
    ids = {r['document_id'] for r in rows}
    write_rows(output / 'provenance.jsonl', (r for r in read_rows(source / 'provenance.jsonl') if r['id'] in ids))
    report = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    report.update(kind='natural_chinese_prose_cycle_v5', refinement_source=str(source),
                  refined_at=datetime.now(timezone.utc).isoformat(),
                  fineweb_origin_domain_allowlist=list(trusted_domains),
                  refinement_source_manifest_sha256=checksum(source / 'manifest.json'),
                  refinement_rejections=dict(rejected),
                  counts=dict(Counter(r['split'] for r in rows)),
                  sources=dict(Counter(r['source'] for r in rows)),
                  topics=dict(Counter(r['topic'] for r in rows)),
                  characters={s: sum(len(r['text']) for r in rows if r['split'] == s) for s in ('train', 'validation', 'test')},
                  files={s: checksum(output / f'{s}.jsonl') for s in ('train', 'validation', 'test')})
    write_json(output / 'manifest.json', report)
    write_rows(root / 'prose-inspection-sample.jsonl', sorted(
        (r for r in rows if r['split'] == 'train'), key=lambda r: key('refined-inspect:' + r['id']))[:24])
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def verify_prepared(root):
    """Pure text verification can run without loading training DLLs."""
    source = root / 'prepared'
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    sets = {s: {'id': set(), 'group_id': set(), 'document_id': set(), 'prefix': set()}
            for s in ('train', 'validation', 'test')}
    counts, failures = Counter(), []
    for split in sets:
        if checksum(source / f'{split}.jsonl') != manifest['files'][split]:
            raise RuntimeError(f'Prepared snapshot hash mismatch: {split}')
        for row in read_rows(source / f'{split}.jsonl'):
            reason = prose_rejection(row['text'])
            if reason or prose_topic(row['text']) != row.get('topic') or row['split'] != split or key(row['text']) != row['id']:
                failures.append(row['id'])
            if split != 'train' and row['parent_corpus_overlap']:
                raise RuntimeError('Parent-seen record in holdout.')
            if row['id'] in sets[split]['id']:
                raise RuntimeError('Duplicate row ID.')
            for field in ('id', 'group_id', 'document_id'):
                sets[split][field].add(row[field])
            sets[split]['prefix'].add(key(re.sub(r'\W', '', row['text'])[:160]))
            counts[split] += 1
    overlaps = {f'{a}_{b}_{field}': len(sets[a][field] & sets[b][field])
                for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))
                for field in ('id', 'group_id', 'document_id', 'prefix')}
    if any(overlaps.values()) or failures or dict(counts) != manifest['counts']:
        raise RuntimeError(f'Prepared verification failed: {overlaps}, failures={failures[:3]}')
    selected_ids = set().union(*(sets[s]['document_id'] for s in sets))
    provenance_ids = {r['id'] for r in read_rows(source / 'provenance.jsonl')}
    missing = selected_ids - provenance_ids
    if missing:
        raise RuntimeError(f'Missing source provenance for {len(missing)} documents.')
    result = {'status': 'prepared_verified', 'counts': dict(counts), 'cross_split_overlap': overlaps,
              'policy_failures': 0, 'missing_document_provenance': 0,
              'manifest_sha256': checksum(source / 'manifest.json'),
              'provenance_sha256': checksum(source / 'provenance.jsonl'),
              'independent_human_review_complete': False}
    write_json(root / 'prepared-verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def runtime_check(root):
    root.mkdir(parents=True, exist_ok=True)
    checks = {}
    for name, code in (
        ('numpy', 'import numpy; print(numpy.__version__)'),
        ('torch', 'import torch; print(torch.__version__); print(torch.cuda.is_available()); assert torch.cuda.is_available(), "CUDA unavailable"'),
        ('sentencepiece', 'import sentencepiece; print(sentencepiece.__version__)'),
    ):
        result = subprocess.run([sys.executable, '-X', 'utf8', '-c', code],
                                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=45)
        checks[name] = {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'python': sys.executable,
              'checks': checks, 'status': 'ready' if all(v['exit_code'] == 0 for v in checks.values()) else 'runtime_blocked',
              'training_started': False,
              'system_security_policy_changed': False}
    write_json(root / 'runtime-check.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def cycle_progress(root, runtime_status):
    """Report saved training evidence without resetting a completed run to pending."""
    report_path = root / 'pretrain/report.json'
    report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else None
    verification_path = root / 'token-verification.json'
    tokenized = (verification_path.exists() and
                 json.loads(verification_path.read_text(encoding='utf-8')).get('status') == 'token_snapshot_verified')
    latest_exists = (root / 'pretrain/latest.pt').is_file()
    started = latest_exists or bool(report and report.get('completed_steps', 0) > 0)
    completed = bool(report and report.get('status') in ('schedule_complete', 'early_stopped'))
    if report:
        status = 'pretraining_' + report['status']
    elif (root / 'pretrain').exists():
        status = 'pretraining_initializing_or_interrupted'
    elif runtime_status != 'ready':
        status = 'prepared_complete_training_blocked'
    else:
        status = 'tokenized_ready_for_pretraining' if tokenized else 'prepared_ready_for_tokenization'
    return {
        'status': status, 'training_started': started, 'training_completed': completed,
        'new_checkpoint_created': latest_exists or (root / 'pretrain/best.pt').is_file(),
        'final_corpus_tokenization_complete': tokenized,
        'pretrain_report': report,
        'code_validation': ('GPU pretraining completed; see recorded validation losses and development outputs.'
                            if completed else 'Text-processing tests passed; GPU training completion not yet recorded.'),
    }


def seal_status(root):
    comparison = json.loads((BASE / 'evaluation/checkpoint-comparison-dev-v1/recommendation.json').read_text(encoding='utf-8'))
    expected = comparison['provisional_further_pretraining_candidate']
    if checksum(PARENT) != expected['checkpoint_sha256'] or checksum(TOKENIZER) != expected['tokenizer_sha256']:
        raise RuntimeError('Selected parent or tokenizer changed.')
    freeze = json.loads((BASE / 'evaluation/v6r5-candidate-freeze.json').read_text(encoding='utf-8'))
    protected = ('base_checkpoint', 'tokenizer', 'safety_adapter', 'memory_adapter', 'motion_adapter',
                 'format_adapter', 'behavior_state_code', 'router_code', 'chat_code', 'evaluation_code')
    actual = {name: checksum(ROOT / freeze['files'][name]['path']) for name in protected}
    if any(actual[name] != freeze['files'][name]['sha256'] for name in protected):
        raise RuntimeError('Frozen runtime asset changed.')
    prior_dialogue_audit = BASE / 'experiments/natural-language-cycle-v2/dialogue-preparation-blocked.json'
    if prior_dialogue_audit.exists():
        audit = json.loads(prior_dialogue_audit.read_text(encoding='utf-8'))
        audit['audit_copied_from'] = str(prior_dialogue_audit)
        audit['original_audit_sha256'] = checksum(prior_dialogue_audit)
        write_json(root / 'dialogue-preparation-blocked.json', audit)
    prepared = json.loads((root / 'prepared/manifest.json').read_text(encoding='utf-8'))
    runtime = json.loads((root / 'runtime-check.json').read_text(encoding='utf-8'))
    result = {
        'kind': 'natural_language_cycle_status_v1', 'updated_at': datetime.now(timezone.utc).isoformat(),
        'status': 'prepared_complete_training_blocked' if runtime['status'] != 'ready' else 'prepared_ready_for_tokenization',
        'parent_checkpoint': str(PARENT), 'parent_checkpoint_sha256': expected['checkpoint_sha256'],
        'tokenizer': str(TOKENIZER), 'tokenizer_sha256': expected['tokenizer_sha256'],
        'prepared_counts': prepared['counts'], 'prepared_characters': prepared['characters'],
        'prepared_manifest_sha256': checksum(root / 'prepared/manifest.json'),
        'prepared_verification_sha256': checksum(root / 'prepared-verification.json'),
        'runtime_check_sha256': checksum(root / 'runtime-check.json'),
        'training_started': False, 'new_checkpoint_created': False, 'application_replaced': False,
        'final_corpus_tokenization_complete': False,
        'superseded_v1_v2_token_snapshots_used_for_training': False,
        'planned_pretrain_step_ceiling': 2000, 'planned_context': 512, 'planned_batch_size': 1,
        'planned_accumulation': 8, 'planned_peak_learning_rate': 3e-5,
        'planned_training_token_ceiling': 2000 * 512 * 8,
        'planned_repetition_policy': 'Actual steps further capped to at most one pool token-equivalent after tokenization; no padding by repeated epochs.',
        'sft_status': 'blocked_missing_independent_natural_dialogue_validation',
        'independent_human_review_complete': False, 'old_blind_data_used': False,
        'frozen_runtime_assets_match_manifest': True, 'frozen_runtime_asset_sha256': actual,
        'hardware_policy': 'Real motors, wheels, GPIO/serial actions and autonomous navigation remain disabled.',
        'code_validation': 'Text-processing unit tests passed; GPU trainer has not been integration-tested because native library loading is blocked.',
    }
    result.update(cycle_progress(root, runtime['status']))
    write_json(root / 'status.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def prepare_dialogue(root):
    import sentencepiece as spm
    source = BASE / 'dialogue/prepared-oasst2-chinese-v17-all-reviewed'
    old = BASE / 'dialogue/prepared-human-openassistant-v1/train.jsonl'
    output = root / 'dialogue'
    if output.exists():
        raise FileExistsError(output)
    parent_trees = {r['tree_id'].split(':')[-1] for r in read_rows(old)}
    tokenizer = spm.SentencePieceProcessor(model_proto=TOKENIZER.read_bytes())
    rejected = Counter()
    candidates = []
    seen = set()
    for row in read_rows(source / 'train.jsonl'):
        if (row.get('source_kind') != 'human_authored_openassistant' or
                row.get('source_review_result') is not True or
                (row.get('source_quality') or 0) < .75 or
                (row.get('source_helpfulness') or 0) < .75 or
                row.get('source_rank') not in (0, None)):
            rejected['source_feedback_threshold'] += 1
            continue
        prompt, answer = row['prompt'], row['answer']
        if PROMOTION.search(prompt + answer) or re.search(
                r'OpenAssistant|OpenAI|ChatGPT|语言模型|作为.{0,6}AI|不能提供|无法帮助|论文.*检索',
                prompt + answer, re.I):
            rejected['identity_promotion_or_refusal'] += 1
            continue
        # Retain whole dialogue turns; no reference rewriting or artificial expansion.
        while len(tokenizer.encode(prompt)) + len(tokenizer.encode(answer)) + 1 > 512:
            turns = prompt.split('用户：')
            if len(turns) <= 3:
                prompt = ''
                break
            prompt = '用户：' + '用户：'.join(turns[2:])
        if not prompt or prompt.count('用户：') < 2 or len(answer) < 12:
            rejected['not_complete_multiturn_within_512_tokens'] += 1
            continue
        if len(CJK.findall(answer)) / max(1, len(answer)) < .5:
            rejected['answer_not_chinese_prose'] += 1
            continue
        identifier = key(prompt + answer)
        if identifier in seen:
            rejected['duplicate_pair'] += 1
            continue
        seen.add(identifier)
        tree = row['tree_id'].split(':')[-1]
        bucket = int(key('natural-sft-cycle-v1:' + tree)[:8], 16) % 100
        split = 'train' if tree in parent_trees or bucket >= 35 else ('test' if bucket < 12 else 'validation')
        candidates.append(dict(row, prompt=prompt, split=split, parent_corpus_tree_seen=tree in parent_trees,
                               local_independent_human_review_complete=False,
                               review_status='original_source_feedback_only'))
    counts = Counter(r['split'] for r in candidates)
    if counts['train'] < 30 or counts['validation'] < 5:
        write_json(root / 'dialogue-preparation-blocked.json', {
            'status': 'insufficient_unseen_source_dialogue_validation', 'counts': dict(counts),
            'source_train_sha256': checksum(source / 'train.jsonl'),
            'source_manifest_sha256': checksum(source / 'manifest.json'),
            'parent_dialogue_train_sha256': checksum(old), 'rejected': dict(rejected),
            'training_started': False, 'quality_threshold': .75, 'helpfulness_threshold': .75,
            'minimum_user_turns': 2, 'maximum_context_tokens': 512,
            'note': 'No independent holdout created; do not relabel parent-seen trees as unseen. Need genuinely additional natural dialogue or a separately collected development-validation set.',
        })
        raise RuntimeError(f'Not enough natural multi-turn source data: {dict(counts)}')
    trees = {s: {r['tree_id'] for r in candidates if r['split'] == s} for s in counts}
    if any(trees[a] & trees[b] for a in trees for b in trees if a != b):
        raise RuntimeError('Conversation tree leakage.')
    output.mkdir()
    for split in ('train', 'validation', 'test'):
        write_rows(output / f'{split}.jsonl', (r for r in candidates if r['split'] == split))
    report = {
        'kind': 'source_feedback_natural_multiturn_cycle_v1',
        'source_train_sha256': checksum(source / 'train.jsonl'),
        'source_manifest_sha256': checksum(source / 'manifest.json'),
        'parent_dialogue_train_sha256': checksum(old),
        'source': str(source), 'tokenizer_sha256': checksum(TOKENIZER),
        'counts': dict(counts), 'trees': {s: len(v) for s, v in trees.items()},
        'rejected': dict(rejected), 'files': {s: checksum(output / f'{s}.jsonl') for s in ('train', 'validation', 'test')},
        'minimum_user_turns': 2, 'max_context_tokens': 512,
        'quality_threshold': .75, 'helpfulness_threshold': .75, 'top_rank_only': True,
        'synthetic_examples_added': 0, 'cross_split_tree_overlap': 0,
        'holdout_parent_seen_tree_count': sum(r['parent_corpus_tree_seen'] for r in candidates if r['split'] != 'train'),
        'independent_local_human_review_complete': False,
        'warning': 'Original source feedback only; no new independent human review or acceptance is claimed. Small exploratory SFT dataset, not sufficient for production.',
        'old_conversation_blind_used': False,
    }
    write_json(output / 'manifest.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def train(root, stage, parent, steps):
    """Full-parameter training with an unchanged architecture and modest memory."""
    import numpy as np
    import sentencepiece as spm
    import torch
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint_sequential
    from torch.utils.tensorboard import SummaryWriter
    from .model import PetLM
    from .run import batch
    from .dialogue_train import load_pairs, make_batch

    class TrainModel(PetLM):
        def forward(self, ids):
            x = self.tokens(ids) + self.positions(torch.arange(ids.shape[1], device=ids.device))
            if self.training and torch.is_grad_enabled():
                x = checkpoint_sequential(self.blocks, 4, x, use_reentrant=False)
            else:
                x = self.blocks(x)
            return F.linear(self.norm(x), self.tokens.weight)

    output = root / stage
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required.')
    data = root / ('tokens' if stage == 'pretrain' else 'dialogue')
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    if checksum(TOKENIZER) != manifest['tokenizer_sha256']:
        raise RuntimeError('Tokenizer mismatch.')
    state = torch.load(parent, map_location='cpu', weights_only=True, mmap=True)
    if state['tokenizer_sha256'] != manifest['tokenizer_sha256']:
        raise RuntimeError('Parent tokenizer mismatch.')
    seed = 20261108
    random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = dict(state['config'])
    parent_step = state['step']
    is_pretrain = stage == 'pretrain'
    accumulation = 8 if is_pretrain else 4
    peak_lr = 3e-5 if is_pretrain else 8e-6
    interval = 200 if is_pretrain else 20
    warmup = 100 if is_pretrain else 10
    config.update(batch_size=1, accumulation=accumulation, learning_rate=peak_lr,
                  steps=steps, warmup_steps=warmup, eval_every=interval, seed=seed)
    if is_pretrain:
        for split in ('train', 'validation', 'test'):
            if checksum(data / f'{split}.bin') != manifest['binary_sha256'][split]:
                raise RuntimeError(f'{split} token hash mismatch.')
        training = np.memmap(data / 'train.bin', mode='r', dtype='<u4')
        validation = np.memmap(data / 'validation.bin', mode='r', dtype='<u4')
        # A small curated pool must not silently become many repeated epochs.
        steps = min(steps, max(1, len(training) // (accumulation * config['context'])))
        config['steps'] = steps
        interval = min(interval, max(10, steps // 5))
        warmup = min(warmup, max(1, steps // 10))
        config.update(eval_every=interval, warmup_steps=warmup)
        def get_batch(which, generator):
            return batch(which, 1, config['context'], generator, 'cuda')
    else:
        for split in ('train', 'validation'):
            if checksum(data / f'{split}.jsonl') != manifest['files'][split]:
                raise RuntimeError('SFT data changed.')
        training = load_pairs(data / 'train.jsonl', spm.SentencePieceProcessor(model_proto=TOKENIZER.read_bytes()), config['context'])
        validation = load_pairs(data / 'validation.jsonl', spm.SentencePieceProcessor(model_proto=TOKENIZER.read_bytes()), config['context'])
        def get_batch(which, generator):
            return make_batch(which, [int(generator.integers(0, len(which)))], 'cuda')

    model = TrainModel(config)
    model.load_state_dict(state['model'])
    del state
    gc.collect()
    model.to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=.1 if is_pretrain else .01, foreach=False)
    dtype = torch.bfloat16
    output.mkdir()
    parent_hash = checksum(parent)
    data_hash = checksum(data / 'manifest.json')
    history = []
    token_count = 0
    torch.cuda.reset_peak_memory_stats()

    def evaluate():
        model.eval()
        value, count = 0., 0
        generator = np.random.default_rng(20261108)
        with torch.inference_mode(), torch.autocast('cuda', dtype=dtype):
            batches = 48 if is_pretrain else len(validation)
            for index in range(batches):
                x, y = get_batch(validation, generator) if is_pretrain else make_batch(validation, [index], 'cuda')
                logits = model(x)
                value += F.cross_entropy(logits.float().reshape(-1, config['vocab_size']), y.reshape(-1), ignore_index=-100, reduction='sum').item()
                count += (y != -100).sum().item()
        return value / count

    def save(name, step, score, include_optimizer=False):
        saved = {'model': model.state_dict(), 'config': config, 'step': step,
                 'tokenizer_sha256': manifest['tokenizer_sha256'], 'dataset_hash': data_hash,
                 'parent_checkpoint': str(parent.resolve()), 'parent_checkpoint_sha256': parent_hash,
                 'parent_step': parent_step, 'training_type': 'natural prose pretraining' if is_pretrain else 'natural assistant-only SFT',
                 'validation_loss': score, 'best_validation_loss': best,
                 'processed_target_tokens': token_count}
        if include_optimizer:
            saved.update(optimizer=optimizer.state_dict(), numpy_rng=rng.bit_generator.state,
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        temp = output / f'{name}.tmp'
        torch.save(saved, temp)
        temp.replace(output / f'{name}.pt')

    baseline = best = evaluate()
    save('best', 0, baseline)
    best_step, stale = 0, 0
    print(json.dumps({'stage': stage, 'baseline_validation_loss': baseline,
                      'target_steps': steps, 'accumulation': accumulation,
                      'train_items_or_tokens': len(training), 'validation_items_or_tokens': len(validation)}, ensure_ascii=False), flush=True)
    started = time.perf_counter()
    with SummaryWriter(str(output / 'runs')) as writer:
        writer.add_scalar('loss/validation', baseline, 0)
        for step in range(1, steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            progress = max(0., (step - warmup) / max(1, steps - warmup))
            lr = peak_lr * min(1., step / warmup) * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
            for group in optimizer.param_groups:
                group['lr'] = lr
            train_loss = 0.
            for _ in range(accumulation):
                x, y = get_batch(training, rng)
                with torch.autocast('cuda', dtype=dtype):
                    logits = model(x)
                    loss = F.cross_entropy(logits.float().reshape(-1, config['vocab_size']), y.reshape(-1), ignore_index=-100)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite loss.')
                (loss / accumulation).backward()
                train_loss += loss.item() / accumulation
                token_count += (y != -100).sum().item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            writer.add_scalar('loss/train', train_loss, step)
            if step == 1 or step % 25 == 0:
                print(f'{stage} step={step}/{steps} loss={train_loss:.4f} elapsed={time.perf_counter()-started:.1f}s', flush=True)
            if step % interval == 0 or step == steps:
                score = evaluate()
                history.append({'step': step, 'train_loss': train_loss, 'validation_loss': score})
                if score < best - .001:
                    best, best_step, stale = score, step, 0
                    save('best', step, score)
                else:
                    stale += 1
                save('latest', step, score, include_optimizer=True)
                writer.add_scalar('loss/validation', score, step)
                report = {'stage': stage, 'status': 'running', 'completed_steps': step, 'target_steps': steps,
                          'best_step': best_step, 'baseline_validation_loss': baseline, 'best_validation_loss': best,
                          'processed_target_tokens': token_count, 'seconds': time.perf_counter()-started,
                          'parent_checkpoint': str(parent), 'parent_sha256': parent_hash,
                          'tokenizer_sha256': manifest['tokenizer_sha256'], 'data_manifest_sha256': data_hash,
                          'history': history, 'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                          'old_blind_data_used': False, 'application_replaced': False}
                write_json(output / 'report.json', report)
                print(json.dumps({'stage': stage, 'step': step, 'validation_loss': score, 'best_step': best_step, 'best': best}), flush=True)
                if stale >= 4:
                    break
    report['status'] = 'schedule_complete' if step == steps else 'early_stopped'
    report['best_checkpoint_sha256'] = checksum(output / 'best.pt')
    report['latest_checkpoint_sha256'] = checksum(output / 'latest.pt')
    report['warning'] = 'Loss improvement is not evidence of normal conversation; inspect raw outputs.'
    write_json(output / 'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'history'}, ensure_ascii=False, indent=2), flush=True)
    return report


def inspect(root, checkpoint, label):
    import sentencepiece as spm
    import torch
    from .model import PetLM
    from .checkpoint_comparison import PROBES, probe_prompt, raw_generate, mechanical_observations, reference_loss
    output = root / 'development-comparison'
    output.mkdir(exist_ok=True)
    target = output / f'{label}.json'
    if target.exists():
        raise FileExistsError(target)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    model = PetLM(state['config'])
    model.load_state_dict(state['model'])
    if checksum(TOKENIZER) != state['tokenizer_sha256']:
        raise RuntimeError('Inspection tokenizer mismatch.')
    del state
    model.to(device='cuda', dtype=torch.bfloat16).eval()
    tokenizer = spm.SentencePieceProcessor(model_proto=TOKENIZER.read_bytes())
    rows = []
    for index, probe in enumerate(PROBES):
        prompt = probe_prompt(probe)
        text, count = raw_generate(model, tokenizer, prompt, 20261109 + index)
        rows.append({'probe_id': probe['id'], 'kind': probe['kind'], 'prompt': prompt,
                     'raw_output': text, 'tokens': count, 'observations': mechanical_observations(text),
                     **reference_loss(model, tokenizer, prompt, probe['reference'])})
    result = {'checkpoint': str(checkpoint), 'checkpoint_sha256': checksum(checkpoint),
              'scope': 'Previously used development probes only; not independent acceptance.',
              'outputs': rows, 'semantic_human_pass_rate': None}
    write_json(target, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'refine', 'verify-prepared', 'runtime-check', 'seal-status', 'tokenize', 'prepare-dialogue', 'pretrain', 'sft', 'inspect'))
    parser.add_argument('--root', type=Path, default=DEFAULT)
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--label', default='parent')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('Steps must be positive.')
    if args.command == 'prepare':
        prepare(args.root)
    elif args.command == 'refine':
        refine(args.root)
    elif args.command == 'verify-prepared':
        verify_prepared(args.root)
    elif args.command == 'runtime-check':
        runtime_check(args.root)
    elif args.command == 'seal-status':
        seal_status(args.root)
    elif args.command == 'tokenize':
        tokenize(args.root)
    elif args.command == 'prepare-dialogue':
        prepare_dialogue(args.root)
    elif args.command in ('pretrain', 'sft'):
        train(args.root, args.command, args.parent, args.steps)
    else:
        inspect(args.root, args.parent, args.label)


if __name__ == '__main__':
    main()
