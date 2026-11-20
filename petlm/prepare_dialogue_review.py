"""Screen a pinned OASST2 batch and a small BiToD development split for SFT review."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import unicodedata

import sentencepiece as spm


ROOT = Path(__file__).resolve().parents[1] / "data/petlm"
OASST = ROOT / "experiments/oasst2-screened-cycle-v4"
BITOD = ROOT / "dialogue/raw/bitod-a9bd74d-zh-dev"
PARENT = ROOT / "prepared-chinese-v26r1-strict"
TOKENIZER = ROOT / "tokens-chinese-v26r1-strict-24k/tokenizer.model"
PARENT_DIALOGUES = ("dialogue/prepared-human-openassistant-v1",
                    "dialogue/prepared-human-openassistant-v17-expanded")
SFT_HISTORY = ("dialogue/prepared-natural-plus-approved-behavior-v25",
               "dialogue/lccc-confirmed-cycle-v1",
               "experiments/generation-repair-v2-cycle-r1/final-sft-snapshot-r1")
ROLE = re.compile(r"(?m)^(用户|助手)[：:]")
CJK = re.compile(r"[\u3400-\u9fff]")
PRIVATE = re.compile(
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?<!\d)(?:\d[ -]?){7,19}(?!\d)|"
    r"(?<!\d)\d{17}[\dXx](?!\d)|手机号|电话号码|身份证|银行卡号|账号密码|验证码|"
    r"家庭住址|门牌号|微信号|QQ号|我叫[\u3400-\u9fff]{2,4}[，。！？,\n]|"
    r"我家住在|详细地址|[\u3400-\u9fff]{2,12}(?:路|街|道|巷)\s*\d+号", re.I
)
RISK = re.compile(
    r"自杀|自残|轻生|跳楼|制毒|毒品|爆炸|枪支|枪械|武器|恐怖袭击|强奸|色情|"
    r"裸聊|性交易|人肉搜索|开盒|盗号|诈骗|赌博|博彩|杀人|攻击.{0,8}(?:系统|服务器)|"
    r"漏洞.{0,8}(?:利用|复现)|傻[逼比]|诊断|处方|剂量|服药|法律意见|投资建议|贷款", re.I
)
ARTIFACT = re.compile(
    r"https?://|<[^>]{2,}>|\[redact|\[邮箱\]|\[手机号\]|"
    r"作为.{0,12}(?:AI|人工智能|语言模型)|ChatGPT|OpenAssistant|"
    r"扫码|优惠券|加微信|关注公众号|(?m:^\s*[#*]|\s*\d+[.)、]\s*)", re.I
)
BOOKISH = re.compile(r"综上所述|总而言之|总的来说|希望以上|希望这些|以下是|首先.{0,180}其次")
LIVE = re.compile(
    r"预[订约定]|订好|付款|房价|营业时间|开门时间|气温|港币|港元|人民币|美元|"
    r"\d+\s*(?:元|点钟|号房|人入住)|地址是|评分|星级|订单|已找到|我查到|为您找到"
)
TOPICS = {key: re.compile(pat) for key, pat in {
    "饮食": "厨房|炒菜|晚餐|晚饭|早餐|午饭|午餐|下厨|咖啡|吃什么|面包|餐厅|用餐|做菜|豆腐|菜谱",
    "出行": "旅行|旅游|公园|散步|出游|地铁|火车|徒步|出门|景点|出行",
    "生活": "周末|收纳|整理房间|家务|阳台|下雨|日常|休息|宠物|养猫|养狗|骑车|摄影|衣服|做饭|锻炼",
    "学习": "学习|读书|阅读|笔记|复习|作业|同学|课堂|学英语|单词|写作",
    "兴趣": "电影|音乐|唱歌|吉他|画画|绘画|小说|看书|玩游戏|书籍",
}.items()}


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path, values):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def norm(text):
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).casefold())


def parse(row):
    prompt = row["prompt"]
    markers = list(ROLE.finditer(prompt))
    if not markers or prompt[:markers[0].start()].strip():
        return []
    turns = [(m[1], prompt[m.end():markers[i+1].start() if i+1 < len(markers)
                           else len(prompt)].strip()) for i, m in enumerate(markers)]
    if turns[-1] == ("助手", ""):
        turns.pop()
    return turns + [("助手", row["answer"].strip())]


def pair_key(turns):
    return "\0".join(norm(text) for _, text in turns[-2:])


def grams(text, size):
    return {text[i:i+size] for i in range(len(text) - size + 1)}


class NearIndex:
    def __init__(self):
        self.exact = set()
        self.sizes = []
        self.postings = defaultdict(list)

    def add(self, key):
        if not key or key in self.exact:
            return
        self.exact.add(key)
        signature = grams(key, 4)
        if len(signature) >= 16:
            i = len(self.sizes)
            self.sizes.append(len(signature))
            for char_seq in signature:
                self.postings[char_seq].append(i)

    def reason(self, key):
        if key in self.exact:
            return "exact"
        signature = grams(key, 4)
        if len(signature) < 16:
            return None
        totals = Counter(i for seq in signature for i in self.postings.get(seq, ()))
        if any(n / (self.sizes[i] + len(signature) - n) >= .80 for i, n in totals.items()):
            return "near"
        return None


def bitod_rows(raw):
    for cid, record in raw.items():
        turns = []
        for i, event in enumerate(record["Events"]):
            agent = event["Agent"]
            text = event.get("Text")
            if agent == "KnowledgeBase" or not isinstance(text, str) or not text.strip():
                turns.clear()
                continue
            role = {"User": "用户", "Wizard": "助手"}[agent]
            turns.append((role, text.strip()))
            if role == "助手" and len(turns) >= 4:
                yield f"{cid}:{i}", cid, turns[-6:].copy()


def check(turns, tokenizer, database, topic_required):
    if len(turns) < 4 or len(turns) % 2 or any(
        r != ("用户" if i % 2 == 0 else "助手") or not t or ROLE.search(t)
        for i, (r, t) in enumerate(turns)
    ):
        return ["not_complete_multiturn"], []
    texts = [text for _, text in turns]
    whole = "\n".join(texts)
    topics = [tag for tag, pattern in TOPICS.items() if pattern.search(whole)]
    flags = [tag for pattern, tag in (
        (PRIVATE, "possible_private_data"), (RISK, "safety_or_sensitive_advice"),
        (ARTIFACT, "format_identity_or_promotion"), (LIVE, "booking_or_live_fact_dependency"),
    ) if pattern.search(whole)]
    if topic_required and not topics:
        flags.append("outside_daily_topics")
    normalized = [norm(t) for t in texts]
    if any(len(CJK.findall(t)) / max(1, len(n)) < .70 for t, n in zip(texts, normalized)):
        flags.append("not_predominantly_chinese")
    if not 10 <= len(normalized[-1]) <= 140:
        flags.append("target_length")
    if len(set(normalized)) < len(normalized) or any(
        len(n) > 28 and len(grams(n, 8)) / (len(n) - 7) < .78 for n in normalized
    ):
        flags.append("turn_or_phrase_repetition")
    if BOOKISH.search(texts[-1]):
        flags.append("formulaic_target")
    rendered = "\n".join(f"{r}：{t}" for r, t in turns)
    encoded = tokenizer.encode(rendered)
    if len(encoded) + 1 > 512 or tokenizer.unk_id() in encoded:
        flags.append("tokenizer_context_or_unknown")
    for text in (whole, *(t for t, n in zip(texts, normalized) if len(n) >= 35)):
        canonical = re.sub(r"\s+", "", text)
        keys = [hashlib.sha256(s.encode()).hexdigest() for s in (canonical, canonical[:300])]
        if database.execute("SELECT 1 FROM hashes WHERE full=? OR prefix=? LIMIT 1", keys).fetchone():
            flags.append("parent_corpus_full_or_prefix_match")
            break
    return flags, topics


def prepare(dest):
    if dest.exists():
        raise FileExistsError(dest)
    oasst_manifest = load(OASST / "manifest.json")
    receipt = load(BITOD / "receipt.json")
    index = PARENT / "clean-index.sqlite"
    meta = load(PARENT / "clean-index.meta.json")
    if checksum(index) != meta["index_sha256"] or checksum(PARENT / "manifest.json") != meta["snapshot_manifest_sha256"]:
        raise ValueError("Parent index integrity mismatch.")
    for file in ("README.md", "LICENSE", "zh_valid.json"):
        key = "data/zh_valid.json" if file == "zh_valid.json" else file
        if checksum(BITOD / file) != receipt["files"][key]["sha256"]:
            raise ValueError(f"BiToD input changed: {file}.")
    tok_hash = checksum(TOKENIZER)
    if tok_hash != "ebab3a8c222a7d05b0162d2b0487141ba0182a48041774492e3979f2eb859df4":
        raise ValueError("Tokenizer changed.")
    tokenizer = spm.SentencePieceProcessor(model_proto=TOKENIZER.read_bytes())
    history, parent_dialogue = NearIndex(), NearIndex()
    parent_ids, history_ids, refs = set(), set(), {}
    for folder in (*PARENT_DIALOGUES, *SFT_HISTORY):
        parent = folder in PARENT_DIALOGUES
        for split in (("train",) if parent else ("train", "validation", "development")):
            path = ROOT / folder / f"{split}.jsonl"
            if not path.is_file():
                if split == "train":
                    raise FileNotFoundError(path)
                continue
            refs[(Path(folder) / path.name).as_posix()] = checksum(path)
            for row in rows(path):
                cid = row.get("tree_id") or row.get("conversation_id")
                if cid:
                    (parent_ids if parent else history_ids).add(cid)
                if "prompt" in row and "answer" in row:
                    turns = parse(row)
                    if len(turns) >= 2:
                        (parent_dialogue if parent else history).add(pair_key(turns))
    oasst = {}
    for split in ("validation", "train"):
        path = OASST / f"{split}.jsonl"
        oasst[split] = list(rows(path))
        if checksum(path) != oasst_manifest["files"][split] or len(oasst[split]) != oasst_manifest["counts"][split]:
            raise ValueError(f"OASST2 input changed: {path}.")
    if {r["conversation_id"] for r in oasst["validation"]} & {r["conversation_id"] for r in oasst["train"]}:
        raise ValueError("OASST2 split overlap.")
    sources = {
        "OASST2": {"source": "OpenAssistant/oasst2", "revision": oasst_manifest["source_revision"],
                   "license": oasst_manifest["source_license"]},
        "BiToD": {key: receipt[key] for key in ("source", "revision", "license", "citation")},
    }
    audit, queue, seen, trees = [], [], NearIndex(), set()
    prior_oasst = NearIndex()
    for source_rows in oasst.values():
        for row in source_rows:
            if len(turns := parse(row)) >= 2:
                prior_oasst.add(pair_key(turns))
    database = sqlite3.connect(index.as_uri() + "?mode=ro", uri=True)

    def assess(group, split, row_id, cid, turns):
        flags, topics = check(turns, tokenizer, database, topic_required=True)
        key = pair_key(turns)
        for name, lookup, ids in (("parent_dialogue", parent_dialogue, parent_ids),
                                  ("prior_sft", history, history_ids)):
            reason = "tree" if cid in ids else lookup.reason(key)
            if reason:
                flags.append(f"{name}_{reason}_overlap")
        if group == "OASST2" and split == "validation":
            flags.append("previously_used_exploratory_validation")
        if group == "BiToD" and (reason := prior_oasst.reason(key)):
            flags.append(f"prior_oasst2_{reason}_overlap")
        if not flags and ((group, cid) in trees or seen.reason(key)):
            flags.append("intake_duplicate_or_same_tree")
        ids = {"dataset": group, **sources[group], "split": split, "record_id": row_id,
               "conversation_id": cid, "topics": topics}
        audit.append({**ids, "flags": flags, "result": "excluded" if flags else "assistant_reading_queue",
                      "human_reviewed": False, "included_in_sft": False})
        if not flags:
            seen.add(key)
            trees.add((group, cid))
            queue.append({**ids, "turns": turns, "human_reviewed": False, "included_in_sft": False})

    try:
        for split, source_rows in oasst.items():
            for row in source_rows:
                assess("OASST2", split, row["id"], row["conversation_id"], parse(row))
        raw = load(BITOD / "zh_valid.json")
        for row_id, cid, turns in bitod_rows(raw):
            assess("BiToD", "validation", row_id, cid, turns)
    finally:
        database.close()
    dest.mkdir(parents=True)
    write_rows(dest / "item-screening.jsonl", audit)
    write_rows(dest / "assistant-reading-queue.jsonl", queue)
    manifest = {
        "status": "assistant_triage_not_admitted", "sources": sources,
        "counts": {s: dict(Counter(r["result"] for r in audit if r["dataset"] == s)) for s in sources},
        "flags": {s: dict(Counter(f for r in audit if r["dataset"] == s for f in r["flags"])) for s in sources},
        "bitod_development_conversations": len(raw), "oasst2_input_counts": oasst_manifest["counts"],
        "oasst2_manifest_sha256": checksum(OASST / "manifest.json"),
        "bitod_receipt_sha256": checksum(BITOD / "receipt.json"),
        "tokenizer_sha256": tok_hash, "parent_index": meta, "comparison_dialogue_files": refs,
        "comparison_method": "Parent corpus full/prefix hashes; parent dialogue and selected historical SFT trees, normalized exact final exchange, and 4-character shingle Jaccard >= 0.80. This detects text similarity, not general paraphrases.",
        "oasst2_prior_use": "Existing exploratory cycle. Its validation set is not independent of that checkpoint; train samples are not claimed as unseen.",
        "tests_opened": False, "sft_rows_admitted": 0, "pretraining_changed": False, "sft_started": False,
        "files": {f: checksum(dest / f) for f in ("item-screening.jsonl", "assistant-reading-queue.jsonl")},
    }
    save(dest / "screening-manifest.json", manifest)
    print(json.dumps({"counts": manifest["counts"], "flags": manifest["flags"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    prepare(parser.parse_args().destination)
