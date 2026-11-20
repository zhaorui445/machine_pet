"""Acquire only the pinned Chinese BiToD development split and provenance."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


REPO = "HLTCHKUST/BiToD"
REVISION = "a9bd74de9eecdc3d875cb4ebf6a6beaf9c30c2ff"
FILES = {
    "README.md": (8979, "dad8c1f335e2b1e57f73fe7545a47ad213744460"),
    "LICENSE": (11357, "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64"),
    "data/zh_valid.json": (6234896, "69e10066797c62e9608a47011dec4de584839248"),
}


def fetch(path):
    expected_size, expected_blob = FILES[path]
    url = f"https://cdn.jsdelivr.net/gh/{REPO}@{REVISION}/{path}"
    request = Request(url, headers={"User-Agent": "PetLM-dataset-audit/1.0"})
    with urlopen(request, timeout=20) as response:
        body = response.read(expected_size + 1)
        final_url = response.geturl()
    if len(body) != expected_size:
        raise ValueError(f"Unexpected byte count for {path}: {len(body)}")
    blob = hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
    if blob != expected_blob:
        raise ValueError(f"Git blob for {path} does not match the publisher tree: {blob}")
    return body, {
        "url": url, "final_url": final_url, "bytes": len(body),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(body).hexdigest(),
        "git_blob_sha1": blob,
    }


def acquire(dest):
    if dest.exists():
        raise FileExistsError(dest)
    data_path = "data/zh_valid.json"
    readme, readme_receipt = fetch("README.md")
    license_body, license_receipt = fetch("LICENSE")
    if b"includes the dataset and baselines" not in readme or b"validation and test data" not in readme:
        raise RuntimeError("Could not verify publisher attribution and dataset scope.")
    if b"Apache License" not in license_body or b"Version 2.0" not in license_body:
        raise RuntimeError("Cannot verify Apache-2.0.")
    body, data_receipt = fetch(data_path)
    if not isinstance(json.loads(body), dict):
        raise ValueError("Unexpected dialogue schema.")
    dest.mkdir(parents=True)
    (dest / "README.md").write_bytes(readme)
    (dest / "LICENSE").write_bytes(license_body)
    (dest / "zh_valid.json").write_bytes(body)
    (dest / "receipt.json").write_text(json.dumps({
        "source": REPO, "revision": REVISION, "license": "Apache-2.0",
        "citation": "Lin et al., BiToD, NeurIPS 2021, https://arxiv.org/abs/2106.02787",
        "source_split": "zh_valid", "project_split": "validation",
        "license_basis": "Repository LICENSE and README explicitly identifying the data directory of this publisher release.",
        "source_test_opened": False,
        "files": {"README.md": readme_receipt, "LICENSE": license_receipt, data_path: data_receipt},
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    acquire(parser.parse_args().destination)
