"""Package the authored direct-answer examples for development-only human review."""
from collections import Counter
from pathlib import Path
import re
import sqlite3

from prepare_dialogue_review import (
    PARENT, ROOT, TOKENIZER, PRIVATE, RISK, NearIndex, checksum, load, parse,
    norm, pair_key, rows, save, spm, write_rows,
)
from prepare_direct_answer_intake import REFERENCE, POOL


EXTERNAL_EXCLUSIONS = (
    "复述景点宣传文案，夸张且无法从对话核实",
    "概括商场品牌覆盖，宣传性强",
    "交通陈述可能失效，且复用了相同商场宣传",
    "包含未经实时核验的开放时段",
    "包含未核实的未来车次和票务承诺",
    "无依据判断交通条件，语气生硬",
    "前文断言符合地铁直达及水乡古镇条件，未核实",
    "宣称仅有一个满足条件的景点，后续复述宣传语",
    "直接回答但缺少说明，分类也有歧义",
    "只给景点名，没解释如何满足价位约束",
    "酒店服务与消费档位判断依赖旧数据",
    "无实时天气依据",
    "无实时天气依据",
    "无实时天气依据",
    "无实时天气依据",
    "有具体菜名但商家的当前菜单未知",
    "有具体推荐但区域未确认，价位和评价依赖旧数据",
    "对食客行为的概括没有依据",
    "没有回应地铁要求，还包含生活区域描述",
    "无依据断言推荐商家当前可由地铁直达",
)


def pairs(turns):
    for i in range(2, len(turns) + 1, 2):
        if [role for role, _ in turns[i-2:i]] == ["用户", "助手"]:
            yield pair_key(turns[:i])


def publish(dest):
    names = ("review.md", "development-pending.jsonl", "external-source-review.jsonl", "review-manifest.json")
    if any((dest / name).exists() for name in names):
        raise FileExistsError("Review artifacts already exist.")
    report = load(dest / "screening.json")
    queue = dest / "assistant-reading-queue.jsonl"
    if checksum(queue) != report["reading_queue_sha256"]:
        raise ValueError("External queue changed.")
    external = list(rows(queue))
    if len(external) != len(EXTERNAL_EXCLUSIONS):
        raise ValueError("Every external candidate must have an explicit decision.")
    exclusions = [{
        "id": row["id"], "conversation_id": row["conversation_id"], "source": row["source"],
        "source_license": row["source_license"], "source_revision": row["source_revision"],
        "disposition": "excluded_after_assistant_reading", "reason": reason, "human_reviewed": False,
    } for row, reason in zip(external, EXTERNAL_EXCLUSIONS)]

    tokenizer = spm.SentencePieceProcessor(model_proto=TOKENIZER.read_bytes())
    history, compared = NearIndex(), {}
    pools = [ROOT / ref / f"{split}.jsonl" for ref in REFERENCE for split in ("train", "validation", "development")]
    for path in pools:
        if not path.is_file():
            continue
        compared[path.relative_to(ROOT).as_posix()] = checksum(path)
        for row in rows(path):
            if row.get("prompt") and row.get("answer"):
                for key in pairs(parse(row)):
                    history.add(key)
    initial = dest / "development-drafts.json"
    drafts = load(initial)
    development = []
    db_path = PARENT / "clean-index.sqlite"
    meta = load(PARENT / "clean-index.meta.json")
    if checksum(db_path) != meta["index_sha256"] or checksum(PARENT / "manifest.json") != meta["snapshot_manifest_sha256"]:
        raise ValueError("Parent index integrity mismatch.")
    if len(drafts) != len({row["id"] for row in drafts}):
        raise ValueError("Repeated development ID.")
    import hashlib
    with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as db:
        for row in drafts:
            turns = row["turns"]
            if ([role for role, _ in turns] != ["用户", "助手"] * 2
                    or row["split"] != "development" or not row["synthetic"]
                    or row["training_eligible"] or row["human_decision"] is not None
                    or row["source"] != "project_original_assistant_authored"
                    or any(not norm(text) for _, text in turns)):
                raise ValueError(f"Invalid draft: {row['id']}.")
            whole = "\n".join(text for _, text in turns)
            prompt = "\n".join(f"{role}：{text}" for role, text in turns[:-1]) + "\n助手："
            answer = turns[-1][1]
            encoded = tokenizer.encode(prompt + answer)
            if PRIVATE.search(whole) or RISK.search(whole) or len(encoded) + 1 > 512 or tokenizer.unk_id() in encoded:
                raise ValueError(f"Privacy, safety, or context filter failed: {row['id']}.")
            keys = list(pairs(turns))
            if any(history.reason(key) for key in keys):
                raise ValueError(f"Historical or intra-batch text overlap: {row['id']}.")
            for text in (whole, *(t for _, t in turns if len(norm(t)) >= 35)):
                normalized = re.sub(r"\s+", "", text)
                sha = tuple(hashlib.sha256(t.encode()).hexdigest() for t in (normalized, normalized[:300]))
                if db.execute("SELECT 1 FROM hashes WHERE full=? OR prefix=? LIMIT 1", sha).fetchone():
                    raise ValueError(f"Parent corpus duplicate: {row['id']}.")
            for key in keys:
                history.add(key)
            development.append({**row, "prompt": prompt, "answer": answer,
                                "human_review_status": "pending", "admitted": False})
    lines = [
        "# 日常多轮直接回答：开发样例审核 r1", "",
        f"共 {len(development)} 条，由项目助手编写，模拟日常多轮交流。"
        "仅供开发检查，非真人语料、SFT 训练数据或冻结盲测。", "",
        "请逐条检查：是否直接解决最后一问、是否接住前文约束或切换到新话题、"
        "说法是否自然且有用。按编号标记通过或不通过；有问题请简写原因。", "",
    ]
    for row in development:
        lines.extend([f"## {row['id']} · {row['topic']}", "",
                      f"检查点：{row['focus']}", "",
                      *[f"> **{role}：** {text}\n>" for role, text in row["turns"]],
                      "", "人工结论：待审核  ", "修改建议：", ""])
    (dest / names[0]).write_text("\n".join(lines), encoding="utf-8")
    write_rows(dest / names[1], development)
    write_rows(dest / names[2], exclusions)
    save(dest / names[3], {
        "status": "pending_human_review_development_only",
        "counts_by_topic": Counter(row["topic"] for row in development),
        "pending_development_rows": len(development),
        "external_full_text_reviewed": len(external), "external_retained_rows": 0,
        "train_rows_admitted": 0, "validation_rows_admitted": 0,
        "drafts_sha256": checksum(initial), "external_screening_sha256": checksum(dest / "screening.json"),
        "comparison_input_sha256": compared, "parent_index": meta,
        "dedup_scope": "Named local train/development/validation files and intra-batch pairs, normalized exact and character-4gram Jaccard at 0.80; parent prose full/prefix index.",
        "frozen_blind_sets_opened": False, "sft_started": False, "pretraining_changed": False,
        "files": {name: checksum(dest / name) for name in names[:-1]},
    })
    print(f"Published {len(development)} development cases; excluded all {len(external)} external cases.")


if __name__ == "__main__":
    publish(ROOT / "dialogue/direct-answer-intake-r1")
