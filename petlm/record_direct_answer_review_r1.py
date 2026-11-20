"""Record the user's confirmation of the 12 direct-answer development examples."""
import argparse
from datetime import date

from prepare_dialogue_review import ROOT, checksum, load, rows, save, write_rows


def publish(confirmed_on):
    root = ROOT / "dialogue"
    source = root / "direct-answer-intake-r1"
    target = root / "direct-answer-development-reviewed-r1"
    if target.exists():
        raise FileExistsError(target)
    pending_manifest = source / "review-manifest.json"
    manifest = load(pending_manifest)
    source_rows = source / "development-pending.jsonl"
    if (manifest["status"], manifest["pending_development_rows"]) != (
        "pending_human_review_development_only", 12
    ):
        raise ValueError("Unexpected review scope.")
    for name, expected in manifest["files"].items():
        if checksum(source / name) != expected:
            raise ValueError(f"Review artifact changed: {name}")
    if checksum(source_rows) != "352d49543cdee13720e1d28bfe92f56b05361818e66a4db42b4879c64f8f37af":
        raise ValueError("Development batch does not match the reviewed snapshot.")
    pending = list(rows(source_rows))
    if [row["id"] for row in pending] != [f"DA-DEV-{i:03d}" for i in range(1, 13)]:
        raise ValueError("Unexpected reviewed IDs.")
    approved, decisions = [], []
    for row in pending:
        if (row["split"] != "development" or not row["synthetic"] or row["training_eligible"]
                or row["admitted"] or row["human_decision"] is not None):
            raise ValueError(f"Unexpected pending state: {row['id']}")
        decision = {
            "id": row["id"], "decision": "pass", "reviewer": "user-confirmed-in-chat",
            "confirmed_on": confirmed_on.isoformat(), "scope": "development_only",
            "criteria": {
                "naturalness": "pass", "context_continuity": "pass", "last_question_resolved": "pass",
            },
            "user_confirmation": "审核通过，回复自然，承接前文，解决了最后一问",
        }
        approved.append({
            **row, "human_decision": "pass", "human_review_status": "approved",
            "admitted": True, "admission_scope": "development_only", "training_eligible": False,
        })
        decisions.append(decision)
    target.mkdir()
    outputs = ("development.jsonl", "review-decisions.jsonl")
    write_rows(target / outputs[0], approved)
    write_rows(target / outputs[1], decisions)
    save(target / "manifest.json", {
        "status": "human_approved_development_only", "counts": {
            "development": len(approved), "train": 0, "validation": 0,
        },
        "scope": "Authored examples for visible development checks; not an independent or blind test.",
        "source_review_manifest_sha256": checksum(pending_manifest),
        "source_development_sha256": checksum(source_rows), "confirmed_on": confirmed_on.isoformat(),
        "sft_started": False, "pretraining_changed": False,
        "files": {name: checksum(target / name) for name in outputs},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmed-on", required=True, type=date.fromisoformat)
    publish(parser.parse_args().confirmed_on)
