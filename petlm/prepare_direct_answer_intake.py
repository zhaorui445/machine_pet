"""Find unused, directly answering Chinese multi-turn dialogue review candidates."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re
import sqlite3

from prepare_dialogue_review import (
    ROOT, PARENT, TOKENIZER, PARENT_DIALOGUES, SFT_HISTORY, NearIndex,
    check, checksum, load, norm, pair_key, parse, rows, save, spm, write_rows,
)


CYCLE = ROOT / "experiments/generation-repair-v2-cycle-r1"
POOL = CYCLE / "chinese-task-dialogue-candidate-r2"
REFERENCE = (
    *PARENT_DIALOGUES, *SFT_HISTORY,
    "experiments/generation-repair-v2-cycle-r1/balanced-chinese-dialogue-source-r1",
    "experiments/oasst2-screened-cycle-v4",
    "dialogue/independent-sft-reviewed-r1",
)
TOPIC = {
    "饮食": r"午饭|晚饭|晚餐|吃什么|下厨|餐厅|饭店|素食|忌口|煮饭|做菜|菜系|面馆|咖啡",
    "学习": r"记单词|学英语|学习方法|复习|做笔记|查字典|阅读方法|作业|写作文",
    "出行": r"坐地铁|怎么去|去哪儿|哪里玩|景点|步行|旅游|公园|旅行|散步|出行",
    "兴趣": r"推荐.{0,14}(?:电影|音乐|歌曲|小说)|电影|乐器|吉他|钢琴|摇滚|民谣|画画|绘画",
    "生活": r"周末|收纳|家务|整理房间|下雨|休息|宠物|拍照|打发时间|做饭",
}
TOPIC = {k: re.compile(v) for k, v in TOPIC.items()}
HELPFUL = re.compile(r"可以|推荐|建议|适合|是|比如|有|分为|先|主要|就在|通常")
LIVE_ANSWER = re.compile(
    r"您要|你要|您还|你还|请问|您需要|我帮您|我给您查|为您查|帮您安排|"
    r"我查了|查到了|搜到了|数据库|空房|已为|打车费|多少钱|门票|电话|营业|"
    r"星期[一二三四五六日]|今日|最近|目前|现有|(?:19|20)\d\d年|"
    r"\d.*(?:点|元|公里|星|分钟|米)|(?:路|街|巷)\d+号"
)
SOURCE = {
    "GEM/RiSAWOZ": ("RiSAWOZ", "CC-BY-4.0", "https://aclanthology.org/2020.emnlp-main.67/"),
    "GEM/CrossWOZ": ("CrossWOZ", "Apache-2.0", "https://aclanthology.org/2020.tacl-1.19/"),
    "thu-coai/kdconv": ("KdConv", "Apache-2.0", "https://aclanthology.org/2020.acl-main.635/"),
}


def conversation_key(value):
    return value.removeprefix("natural_task_dialogue:")


def scout(dest):
    if dest.exists():
        raise FileExistsError(dest)
    manifest_path = POOL / "manifest.json"
    manifest = load(manifest_path)
    refs = {}
    ref_ids, ref_cids = set(), set()
    history = NearIndex()
    for name in REFERENCE:
        for split in ("train", "validation", "development"):
            path = ROOT / name / f"{split}.jsonl"
            if not path.is_file():
                continue
            refs[path.relative_to(ROOT).as_posix()] = checksum(path)
            for row in rows(path):
                for key in ("id", "source_record_id"):
                    if row.get(key):
                        ref_ids.add(row[key])
                for key in ("conversation_id", "tree_id"):
                    if row.get(key):
                        ref_cids.add(conversation_key(row[key]))
                if row.get("prompt") and row.get("answer"):
                    turns = parse(row)
                    for end in range(2, len(turns) + 1, 2):
                        if [role for role, _ in turns[end-2:end]] == ["用户", "助手"]:
                            history.add(pair_key(turns[:end]))
    review = POOL / "review"
    for split in ("train", "validation"):
        path = review / f"{split}-sample-decisions.completed.jsonl"
        refs[path.relative_to(ROOT).as_posix()] = checksum(path)
        for row in rows(path):
            for key in ("id", "source_record_id", "candidate_id"):
                if row.get(key):
                    ref_ids.add(row[key])
            if row.get("conversation_id"):
                ref_cids.add(conversation_key(row["conversation_id"]))

    index = PARENT / "clean-index.sqlite"
    meta = load(PARENT / "clean-index.meta.json")
    if checksum(index) != meta["index_sha256"] or checksum(PARENT / "manifest.json") != meta["snapshot_manifest_sha256"]:
        raise ValueError("Parent index integrity mismatch.")
    tokenizer = spm.SentencePieceProcessor(model_proto=TOKENIZER.read_bytes())
    if checksum(TOKENIZER) != manifest["tokenizer_sha256"]:
        raise ValueError("Unexpected tokenizer.")

    buckets = defaultdict(list)
    counts = Counter()
    selected_cids = set()
    selected_history = NearIndex()
    limits = {"train": 5, "validation": 3}
    with sqlite3.connect(index.as_uri() + "?mode=ro", uri=True) as db:
        for split in ("validation", "train"):
            path = POOL / f"{split}-candidate.jsonl"
            if checksum(path) != manifest["files_sha256"][split]:
                raise ValueError(f"Source snapshot changed: {split}.")
            for row in rows(path):
                counts[f"{split}/considered"] += 1
                cid = conversation_key(row["conversation_id"])
                if row["id"] in ref_ids or cid in ref_cids:
                    counts[f"{split}/existing_row_or_conversation"] += 1
                    continue
                if cid in selected_cids:
                    continue
                turns = parse(row)
                if len(turns) < 4 or len(turns) % 2:
                    continue
                # Keep the original visible context; do not invent a lead-in to make a row fit.
                user, answer = turns[-2][1], turns[-1][1]
                if not 14 <= len(norm(answer)) <= 105 or not HELPFUL.search(answer) or LIVE_ANSWER.search(answer):
                    continue
                if re.search(r"[?？]|么[。！!]*$|吗[。！!]*$", answer):
                    continue
                topic = next((k for k, p in TOPIC.items() if p.search(user + answer)), None)
                if not topic:
                    continue
                dataset = row["source_dataset"]
                label, license_, citation = SOURCE[dataset]
                bucket = (split, topic, label)
                if len(buckets[bucket]) >= limits[split]:
                    continue
                flags, _ = check(turns, tokenizer, db, topic_required=False)
                if flags:
                    counts.update(f"{split}/filter/{f}" for f in flags)
                    continue
                keys = [pair_key(turns[:n]) for n in range(2, len(turns)+1, 2)]
                if history.reason(keys[-1]) or any(selected_history.reason(key) for key in keys):
                    counts[f"{split}/exact_or_near_duplicate"] += 1
                    continue
                source = manifest["sources"][label]
                if (row["split"], row["source_license"], row["source_revision"]) != (
                    split, license_, source["revision"]
                ):
                    raise ValueError("Candidate and source metadata disagree.")
                item = {
                    "id": row["id"], "conversation_id": cid, "turn_index": row["turn_index"],
                    "prompt": row["prompt"], "answer": row["answer"], "turns": turns,
                    "source": dataset, "source_license": license_, "source_revision": source["revision"],
                    "source_file": row["source_file"], "citation": citation, "split": split, "topic": topic,
                    "human_decision": None, "training_eligible": False,
                }
                buckets[bucket].append(item)
                selected_cids.add(cid)
                for key in keys:
                    selected_history.add(key)
    output = [item for key in sorted(buckets) for item in buckets[key]]
    dest.mkdir(parents=True)
    write_rows(dest / "assistant-reading-queue.jsonl", output)
    save(dest / "screening.json", {
        "status": "not_admitted", "counts": counts,
        "queued_counts": {"/".join(k): len(v) for k, v in sorted(buckets.items()) if v},
        "source_manifest_sha256": checksum(manifest_path), "source_candidate_sha256": manifest["files_sha256"],
        "reference_sha256": refs, "parent_index": meta,
        "reading_queue_sha256": checksum(dest / "assistant-reading-queue.jsonl"),
        "comparison_scope": "Named parent/SFT history and previous review records; 0.80 4-character Jaccard on normalized exchanges. Source test and frozen blind sets not accessed.",
        "sft_started": False, "pretraining_changed": False,
    })
    print(dict(Counter((r["split"], r["topic"]) for r in output)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dest", type=Path)
    scout(parser.parse_args().dest)
