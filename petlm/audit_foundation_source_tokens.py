"""Count tokens by source in the frozen v26r1 training binary, without retokenizing it."""
import argparse
from collections import Counter
from datetime import date
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import sentencepiece as spm


ROOT = Path(__file__).resolve().parents[1] / "data/petlm"
CORPUS = ROOT / "prepared-chinese-v26r1-strict"
TOKENS = ROOT / "tokens-chinese-v26r1-strict-24k"
CYCLE = ROOT / "experiments/foundation-chinese-quality-cycle-r1"
BLOCK_TOKENS = 4_194_304
VERIFY_EVERY = 97


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def boundaries(tokens, eos):
    for offset in range(0, len(tokens), BLOCK_TOKENS):
        for end in np.flatnonzero(tokens[offset:offset + BLOCK_TOKENS] == eos):
            yield offset + int(end) + 1


def calculate():
    token_manifest = TOKENS / "manifest.json"
    prepared_manifest = CORPUS / "manifest.json"
    ledger = CYCLE / "corpus-readiness-r1.json"
    tm, pm, readiness = (load(path) for path in (token_manifest, prepared_manifest, ledger))
    tokenizer_path, train_binary = TOKENS / "tokenizer.model", TOKENS / "train.bin"
    model_bytes = tokenizer_path.read_bytes()
    tokenizer_sha = hashlib.sha256(model_bytes).hexdigest()
    require(tokenizer_sha == tm["tokenizer_sha256"], "Frozen tokenizer hash mismatch.")
    sp = spm.SentencePieceProcessor(model_proto=model_bytes)
    eos = sp.eos_id()
    require(eos >= 0 and sp.is_control(eos), "The end marker is not a reserved token.")
    n = tm["tokens"]["train"]
    require(n == readiness["current_snapshot"]["training_tokens"], "Training totals differ in manifests.")
    require(train_binary.stat().st_size == n * 4, "Unexpected training binary byte count.")
    binary_hash = sha(train_binary)
    require(binary_hash == tm["binary_sha256"]["train"], "Training binary hash mismatch.")
    train = np.memmap(train_binary, mode="r", dtype="<u4")
    end_positions = boundaries(train, eos)
    text_hash = hashlib.sha256()
    counts, docs, verified = Counter(), Counter(), Counter()
    start = 0
    with (CORPUS / "train.jsonl").open("rb") as stream:
        for line in stream:
            text_hash.update(line)
            row = json.loads(line)
            source = row["source"]
            require(source in pm["sources"], f"Unregistered source: {source}")
            end = next(end_positions, None)
            require(end is not None, "Training binary ends before the last text row.")
            counts[source] += end - start
            if docs[source] % VERIFY_EVERY == 0:
                encoded = sp.encode(row["text"]) + [eos]
                require(np.array_equal(train[start:end], encoded),
                        f"Tokenizer check failed for {source}, source row {docs[source] + 1}.")
                verified[source] += 1
            docs[source] += 1
            start = end
    text_sha = text_hash.hexdigest()
    require(text_sha == tm["input_sha256"]["train"] == pm["files"]["train.jsonl"],
            "Training text differs from the pinned binary input.")
    require(start == n and next(end_positions, None) is None, "Unmatched tokens at end of binary.")
    require(sum(counts.values()) == n and sum(docs.values()) == pm["splits"]["train"],
            "Source counts do not reconcile with the frozen training split.")
    require(all(docs[source] <= all_docs for source, all_docs in pm["sources"].items()),
            "Training source document count exceeds the full snapshot count.")
    targets = readiness["admission_targets"]
    max_share = Fraction(str(targets["single_dataset_token_fraction_maximum"]))
    cap = max_share.numerator * n // max_share.denominator
    fineweb = counts["fineweb2"]
    min_total = (fineweb * max_share.denominator + max_share.numerator - 1) // max_share.numerator
    pilots = readiness["approved_candidate_pilots"]
    combined_plan = n + pilots["approved_unique_train_tokens"]
    require(combined_plan == pilots["combined_train_tokens_including_current_snapshot"],
            "Combined training count differs from the readiness ledger.")
    gaps = {
        "training_tokens_from_frozen_snapshot": max(0, targets["unique_train_tokens_minimum"] - n),
        "training_tokens_after_approved_pilots_not_yet_in_snapshot":
            max(0, targets["unique_train_tokens_minimum"] - combined_plan),
        "independent_validation_tokens_after_approved_pilots": max(
            0, targets["independent_validation_tokens_minimum"] - pilots["approved_unique_validation_tokens"]),
    }
    return {
        "status": "source_token_audit_complete_corpus_not_training_authorized",
        "unit": "Training tokens from frozen train.bin, including exactly one EOS per document.",
        "scope": "Frozen train split only. Approved pilots are separate; no validation/test text was read.",
        "method": "Read-only scan of EOS boundaries in little-endian uint32 train.bin aligned to train.jsonl row order. "
                  "Each source's first row and then every 97th row re-encoded and compared token-for-token.",
        "total_train_tokens": n,
        "total_train_documents": sum(docs.values()),
        "tokenizer_sample_rows_verified": sum(verified.values()),
        "sources": [
            {"source": source, "train_documents": docs[source], "train_tokens_with_eos": count,
             "token_share": count / n, "tokenizer_sample_rows_verified": verified[source],
             "exceeds_single_source_cap": count > cap}
            for source, count in counts.most_common()
        ],
        "targets": targets,
        "fineweb2_balance": {
            "max_fraction": float(max_share),
            "additional_non_fineweb2_tokens_for_cap_if_current_fineweb2_is_unchanged":
                max(0, min_total - n),
            "minimum_total_train_tokens_for_cap_if_current_fineweb2_is_unchanged": min_total,
        },
        "approved_pilots_not_in_frozen_token_counts": {
            "train_tokens": pilots["approved_unique_train_tokens"],
            "independent_validation_tokens": pilots["approved_unique_validation_tokens"],
        },
        "gaps": gaps,
        "e_drive_free_bytes_at_audit": shutil.disk_usage(ROOT).free,
        "inputs_sha256": {
            "token_manifest": sha(token_manifest), "corpus_manifest": sha(prepared_manifest),
            "readiness_ledger": sha(ledger), "train_jsonl": text_sha,
            "train_bin": binary_hash, "tokenizer": tokenizer_sha,
        },
        "training_started": False,
        "sft_started": False,
    }


def report(output, audited_on):
    if output.exists():
        raise FileExistsError(output)
    result = calculate()
    result["audited_on"] = audited_on.isoformat()
    fw = next(row for row in result["sources"] if row["source"] == "fineweb2")
    balance = result["fineweb2_balance"]
    gap = result["gaps"]
    pilots = result["approved_pilots_not_in_frozen_token_counts"]
    cap = balance["max_fraction"]
    reserve = balance["additional_non_fineweb2_tokens_for_cap_if_current_fineweb2_is_unchanged"]
    decision = "Pause adding FineWeb2 under the current source cap." if fw["exceeds_single_source_cap"] else (
        "Keep checking the source cap before considering additional FineWeb2."
    )
    lines = [
        "# Frozen Chinese corpus: training tokens by source", "",
        f"Audit date: {audited_on.isoformat()}. Frozen train split only; no training or SFT started.", "",
        "| Source | Train documents | Tokens (with EOS) | Token share |",
        "| --- | ---: | ---: | ---: |",
        *[f"| {s['source']} | {s['train_documents']:,} | {s['train_tokens_with_eos']:,} | {s['token_share']:.4%} |"
          for s in result["sources"]],
        f"| Total | {result['total_train_documents']:,} | {result['total_train_tokens']:,} | 100% |", "",
        f"FineWeb2 is {fw['token_share']:.4%} of train tokens; the single-source cap is {cap:.0%}. {decision}",
        f"With the frozen FineWeb2 portion unchanged, the balance alone requires at least {reserve:,} "
        "new non-FineWeb2 tokens. This requirement is part of the overall training gap, not additional to it.", "",
        f"After counting {pilots['train_tokens']:,} approved training-pilot tokens separately, "
        f"{gap['training_tokens_after_approved_pilots_not_yet_in_snapshot']:,} unique training tokens remain "
        f"to the volume gate. With {pilots['independent_validation_tokens']:,} approved independent-validation tokens, "
        f"{gap['independent_validation_tokens_after_approved_pilots']:,} more are required. The frozen legacy "
        "validation set is not counted toward the independent-source gate.", "",
        "Next intake: prioritize batch-accessible, explicitly licensed Chinese education, daily-life, and narrative "
        "sources. Record source and topic token totals separately. Before downloading prose, pin versions and assign "
        "distinct training and validation sources. Then clean, deduplicate, and review a stratified batch before admission.", "",
        f"Verification: hashes of the frozen inputs matched, totals matched both manifests, and "
        f"{result['tokenizer_sample_rows_verified']:,} stratified rows matched the frozen tokenizer token-for-token. "
        f"E-drive free space at audit: {result['e_drive_free_bytes_at_audit']/10**9:.2f} GB.", "",
    ]
    output.mkdir(parents=True)
    (output / "source-token-audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"sources": result["sources"], "gaps": gap, "fineweb2_balance": balance}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=date.fromisoformat)
    args = parser.parse_args()
    report(CYCLE / "source-token-audit-r1", args.date)
