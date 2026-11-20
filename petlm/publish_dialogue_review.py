"""Attach assistant reading decisions and create an immutable human review batch."""
import argparse
from pathlib import Path

from prepare_dialogue_review import checksum, load, rows, save, write_rows


def publish(directory):
    screened = directory / "screening-manifest.json"
    notes_path = directory / "assistant-decisions.json"
    manifest, notes = load(screened), load(notes_path)
    for name, expected in manifest["files"].items():
        if checksum(directory / name) != expected:
            raise ValueError(f"Screened input changed: {name}")
    queue = list(rows(directory / "assistant-reading-queue.jsonl"))
    ids = {row["record_id"] for row in queue}
    if len(ids) != len(queue) or ids != notes.keys():
        raise ValueError("Every reading-queue row must have exactly one decision.")
    candidates = []
    for row in queue:
        note = notes[row["record_id"]]
        if not note.startswith(("候选：", "排除：")):
            raise ValueError("Missing assistant decision.")
        if note.startswith("候选："):
            if row["dataset"] != "BiToD" or row["split"] != "validation":
                raise ValueError("Only validation candidates from BiToD are permitted.")
            candidates.append({
                **row, "review_id": f"BTD-DEV-{len(candidates)+1:03d}", "assistant_note": note,
                "human_decision": None, "intended_use": "SFT validation only: restaurant clarification",
            })
    if len({r["conversation_id"] for r in candidates}) != len(candidates):
        raise ValueError("Repeated conversation.")
    pending_ids = {row["record_id"] for row in candidates}
    outputs = ("review.md", "human-review-pending.jsonl", "item-review-ledger.jsonl", "review-manifest.json")
    if any((directory / f).exists() for f in outputs):
        raise FileExistsError("Review batch already exists.")
    lines = [
        "# 独立 SFT 对话候选审核 r1", "",
        f"共 {len(candidates)} 条。仅供未来 SFT 的餐饮需求澄清验证分区使用，"
        "不能代表通用闲聊能力，也不是冻结盲测。待逐条人工审核。", "",
        "请核对整段上下文：中文是否自然；最后一句是否接住用户需求、只问一个必要问题；"
        "是否含个人信息或冒称可以查到商户、下单。每条分别标记通过或不通过。", "",
        "署名：Lin 等，BiToD（NeurIPS 2021），港科大 HLTCHKUST/BiToD。"
        "仅转写连续对话，未引入场景、动作、状态、数据库或个人资料字段。", "",
    ]
    for row in candidates:
        origin = f"https://github.com/{row['source']}/blob/{row['revision']}/data/zh_valid.json"
        lines.extend([f"## {row['review_id']}", "",
                      f"来源：[固定版本中文开发分区]({origin}) · Apache-2.0  ",
                      f"原记录：{row['record_id']}", "",
                      *[f"> **{role}：** {text}\n>" for role, text in row["turns"]],
                      "", row["assistant_note"], "", "人工结论：待审核  ", "理由：", ""])
    (directory / "review.md").write_text("\n".join(lines), encoding="utf-8")
    write_rows(directory / "human-review-pending.jsonl", candidates)
    write_rows(directory / "item-review-ledger.jsonl", (
        {**row, "assistant_full_text_read": row["record_id"] in notes,
         "assistant_note": notes.get(row["record_id"]),
         "disposition": ("pending_human_review" if row["record_id"] in pending_ids
                         else "excluded_after_assistant_reading" if row["record_id"] in notes
                         else "excluded_by_automatic_rules")}
        for row in rows(directory / "item-screening.jsonl")
    ))
    save(directory / "review-manifest.json", {
        "status": "pending_human_review", "screening_manifest_sha256": checksum(screened),
        "assistant_decisions_sha256": checksum(notes_path),
        "assistant_full_text_reads": dict.fromkeys(manifest["sources"], 0) | {
            name: sum(row["dataset"] == name for row in queue) for name in manifest["sources"]
        },
        "pending_validation_rows": len(candidates), "pending_training_rows": 0,
        "sft_rows_admitted": 0, "original_tests_opened": False,
        "pretraining_changed": False, "sft_started": False,
        "files": {f: checksum(directory / f) for f in outputs[:-1]},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    publish(parser.parse_args().directory)
