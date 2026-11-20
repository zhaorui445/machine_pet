"""Publish the three user-confirmed BiToD validation rows as a separate snapshot."""
import argparse
from datetime import date
import json
from pathlib import Path
from shutil import copyfile

from prepare_dialogue_review import (
    BITOD, ROOT, NearIndex, bitod_rows, checksum, load, pair_key, rows, save, write_rows,
)


REVIEW_IDS = ("BTD-DEV-001", "BTD-DEV-002", "BTD-DEV-003")
REVISION = "a9bd74de9eecdc3d875cb4ebf6a6beaf9c30c2ff"


def publish(confirmed_on):
    dialogue = ROOT / "dialogue"
    source = dialogue / "independent-sft-intake-r1"
    dest = dialogue / "independent-sft-reviewed-r1"
    if dest.exists():
        raise FileExistsError(dest)
    manifest_path = source / "review-manifest.json"
    manifest = load(manifest_path)
    if (manifest["status"], manifest["pending_validation_rows"], manifest["pending_training_rows"]) != (
        "pending_human_review", 3, 0
    ):
        raise ValueError("Unexpected review scope.")
    for name, expected in manifest["files"].items():
        if checksum(source / name) != expected:
            raise ValueError(f"Review artifact changed: {name}")
    for name, field in (("screening-manifest.json", "screening_manifest_sha256"),
                        ("assistant-decisions.json", "assistant_decisions_sha256")):
        if checksum(source / name) != manifest[field]:
            raise ValueError(f"Review provenance changed: {name}")
    screening = load(source / "screening-manifest.json")
    receipt_path = BITOD / "receipt.json"
    if checksum(receipt_path) != screening["bitod_receipt_sha256"]:
        raise ValueError("BiToD receipt changed.")
    receipt = load(receipt_path)
    if (receipt["source"], receipt["revision"], receipt["license"], receipt["source_split"]) != (
        "HLTCHKUST/BiToD", REVISION, "Apache-2.0", "zh_valid"
    ):
        raise ValueError("Unexpected publisher source.")
    for key, path in (("LICENSE", BITOD / "LICENSE"), ("data/zh_valid.json", BITOD / "zh_valid.json")):
        if checksum(path) != receipt["files"][key]["sha256"]:
            raise ValueError(f"Source file changed: {key}")
    pending = list(rows(source / "human-review-pending.jsonl"))
    if tuple(row["review_id"] for row in pending) != REVIEW_IDS:
        raise ValueError("Unexpected review IDs.")
    raw = {identifier: (conversation, turns) for identifier, conversation, turns in
           bitod_rows(load(BITOD / "zh_valid.json"))}
    seen, trees = NearIndex(), set()
    approved, decisions = [], []
    for item in pending:
        if (item["dataset"], item["source"], item["revision"], item["license"], item["split"]) != (
            "BiToD", receipt["source"], REVISION, receipt["license"], "validation"
        ) or item["human_decision"] is not None or item["included_in_sft"]:
            raise ValueError("Invalid pending row state.")
        conversation, original = raw[item["record_id"]]
        turns = [tuple(turn) for turn in item["turns"]]
        if turns != original or conversation != item["conversation_id"]:
            raise ValueError(f"Dialogue text changed: {item['review_id']}")
        key = pair_key(turns)
        if conversation in trees or seen.reason(key):
            raise ValueError("Repeated dialogue in the approved batch.")
        seen.add(key)
        trees.add(conversation)
        decision = {
            "review_id": item["review_id"], "source_record_id": item["record_id"],
            "decision": "pass", "reviewer": "user-confirmed-in-chat",
            "confirmed_on": confirmed_on.isoformat(), "scope": "validation_only",
        }
        approved.append({
            "id": item["review_id"], "conversation_id": conversation, "split": "validation",
            "prompt": "\n".join(f"{role}：{text}" for role, text in turns[:-1]) + "\n助手：",
            "answer": turns[-1][1], "source": receipt["source"], "source_revision": REVISION,
            "source_license": receipt["license"], "source_record_id": item["record_id"],
            "source_url": f"https://github.com/{receipt['source']}/blob/{REVISION}/data/zh_valid.json",
            "citation": item["citation"], "capability": "restaurant_requirement_clarification",
            "training_eligible": False, "human_review": decision,
        })
        decisions.append(decision)
    dest.mkdir()
    write_rows(dest / "validation.jsonl", approved)
    write_rows(dest / "review-decisions.jsonl", decisions)
    copyfile(BITOD / "LICENSE", dest / "LICENSE-BiToD")
    save(dest / "manifest.json", {
        "status": "human_approved_validation_only", "counts": {"train": 0, "validation": len(approved)},
        "review_batch": source.name, "review_manifest_sha256": checksum(manifest_path),
        "source_receipt_sha256": checksum(receipt_path),
        "scope": "Restaurant requirement clarification pilot; neither general dialogue nor a frozen blind set.",
        "training_ready": False, "sft_started": False, "pretraining_changed": False,
        "files": {name: checksum(dest / name)
                  for name in ("validation.jsonl", "review-decisions.jsonl", "LICENSE-BiToD")},
    })
    print(json.dumps({"validation_rows": len(approved), "train_rows": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmed-on", required=True, type=date.fromisoformat)
    publish(parser.parse_args().confirmed_on)
