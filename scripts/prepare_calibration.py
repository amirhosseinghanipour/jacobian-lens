#!/usr/bin/env python3
"""Sample disjoint 128-token web documents and freeze evaluation splits.

Requires pyarrow, huggingface-hub==0.30.2 and tokenizers==0.22.2. Output data retains the
upstream FineWeb dataset's ODC-BY attribution and source document URLs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import requests

SEED = 20261105
DATASET = "HuggingFaceFW/fineweb"

def digest(data):
    return hashlib.sha256(data).hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

def words(text):
    return re.findall(r"\w+", text.casefold())

def main():
    from huggingface_hub import HfApi, HfFileSystem
    import pyarrow.parquet as pq
    import random
    from tokenizers import Tokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--revision")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("refusing to replace a frozen corpus")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    revision = args.revision
    if revision is None:
        response = requests.get(f"https://huggingface.co/api/datasets/{DATASET}", timeout=120)
        response.raise_for_status()
        revision = response.json()["sha"]
    evaluation_hashes = {}
    excluded = set()
    splits = {}
    previous_smoke = {"carnival-ocean", "amazon-language", "mars-color", "firstletter-paris-country"}
    for path in sorted(Path("data/evaluations").glob("*.json")):
        evaluation_hashes[str(path)] = digest(path.read_bytes())
        items = json.loads(path.read_text())["items"]
        # Rank by prompt hash, keeping exact prompt duplicates together.
        hashes = sorted(set(digest(item["prompt"].encode()) for item in items))
        development = set(sorted(hashes, key=lambda h: digest(f"{SEED}:{h}".encode()))[:max(1, len(hashes) // 5)])
        for item in items:
            tokens = words(item["prompt"])
            excluded.update(tuple(tokens[i:i+8]) for i in range(len(tokens) - 7))
            if item["name"] in previous_smoke:
                development.add(digest(item["prompt"].encode()))
        splits[path.stem] = {item["name"]: "development" if digest(item["prompt"].encode()) in development else "test" for item in items}
    # Bounded shuffle of the pinned streaming corpus; not an IID sample of all
    # FineWeb or a claim to reconstruct Qwen's undisclosed pretraining mixture.
    files = sorted(f for f in HfApi().list_repo_files(DATASET, repo_type="dataset", revision=revision)
                   if f.startswith("sample/10BT/") and f.endswith(".parquet"))
    rng = random.Random(SEED)
    rng.shuffle(files)
    fs = HfFileSystem()
    def rows():
        for filename in files:
            print(f"Reading {filename}", flush=True)
            with fs.open(f"datasets/{DATASET}@{revision}/{filename}", "rb") as handle:
                parquet = pq.ParquetFile(handle)
                for batch in parquet.iter_batches(batch_size=256, columns=["text", "id", "url"]):
                    yield from batch.to_pylist()
    def shuffled():
        buffer = []
        for row in rows():
            if len(buffer) < 10000:
                buffer.append(row)
                continue
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = row
        rng.shuffle(buffer)
        yield from buffer
    stream = shuffled()
    records = []
    seen = set()
    rejected = {"duplicate": 0, "short": 0, "evaluation_overlap": 0}
    for row in stream:
        text = row["text"]
        identity = digest(text.encode())
        if identity in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(identity)
        tokens = words(text)
        if any(tuple(tokens[i:i+8]) in excluded for i in range(len(tokens) - 7)):
            rejected["evaluation_overlap"] += 1
            continue
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if len(ids) < 128:
            rejected["short"] += 1
            continue
        start = int(digest(f"window:{SEED}:{identity}".encode()), 16) % (len(ids) - 127)
        window = ids[start:start+128]
        records.append({"index": len(records), "document_id": row.get("id"), "url": row.get("url"), "document_sha256": identity, "token_start": start, "ids": window})
        if len(records) == 1100:
            break
    if len(records) != 1100:
        raise ValueError("not enough eligible documents")
    for shard in range(110):
        save(args.output / f"shard-{shard:03}.json", records[shard*10:(shard+1)*10])
    save(args.output / "splits.json", splits)
    manifest = {"schema": 1, "dataset": DATASET, "dataset_revision": revision, "config": "sample-10BT", "dataset_license": "ODC-BY-1.0", "dataset_url": f"https://huggingface.co/datasets/{DATASET}", "sampling": "streaming bounded shuffle, buffer 10000, one deterministic random 128-token window per distinct document", "seed": SEED, "ordered_source_files": files, "tokenizer_sha256": digest(args.tokenizer.read_bytes()), "main_100": [0, 100], "main_1000": [0, 1000], "replicate_100": [1000, 1100], "rejected": rejected, "evaluation_sha256": evaluation_hashes, "files": {p.name: digest(p.read_bytes()) for p in sorted(args.output.iterdir())}}
    save(args.output / "manifest.json", manifest)
    print(json.dumps({"dataset_revision": revision, "samples": len(records), "manifest_sha256": digest((args.output / "manifest.json").read_bytes())}))

if __name__ == "__main__":
    main()
