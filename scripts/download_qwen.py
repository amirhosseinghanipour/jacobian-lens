#!/usr/bin/env python3
"""Download Qwen2.5-0.5B-Instruct using only Python's standard library.

Usage: python3 scripts/download_qwen.py [destination] [--revision COMMIT_OR_TAG]
Resolves revision to an immutable commit and verifies upstream LFS SHA-256.
Interrupted downloads resume from .part files. No Hugging Face login required.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FILES = ("config.json", "generation_config.json", "tokenizer.json",
         "tokenizer_config.json", "merges.txt", "vocab.json", "LICENSE",
         "model.safetensors")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, destination, size, expected_sha):
    if destination.exists() and destination.stat().st_size == size:
        if expected_sha and sha256(destination) == expected_sha:
            return
    partial = destination.with_name(destination.name + ".part")
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            partial.unlink()
            offset = 0
        try:
            if offset < size:
                headers = {"User-Agent": "jacobian-lens-research/0.1", "Accept-Encoding": "identity"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status == 206:
                        content_range = response.headers.get("Content-Range", "")
                        if not content_range.startswith(f"bytes {offset}-"):
                            raise ValueError(f"unexpected Content-Range: {content_range}")
                        mode = "ab"
                    elif response.status == 200:
                        mode = "wb"  # Server ignored Range; restart, never append twice.
                    else:
                        raise ValueError(f"unexpected status {response.status}")
                    with partial.open(mode) as target:
                        for block in iter(lambda: response.read(1024 * 1024), b""):
                            target.write(block)
                        target.flush()
                        os.fsync(target.fileno())
            if partial.stat().st_size != size:
                raise ValueError(f"size mismatch for {destination.name}: expected {size}")
            if expected_sha and sha256(partial) != expected_sha:
                partial.unlink()
                raise ValueError(f"SHA-256 mismatch for {destination.name}")
            os.replace(partial, destination)
            return
        except (OSError, ValueError) as error:
            if attempt == 4:
                raise
            print(f"Retry {attempt + 1}/4: {error}", flush=True)
            time.sleep(min(2 ** attempt, 8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", nargs="?", default="models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    revision = urllib.parse.quote(args.revision, safe="")
    api = f"https://huggingface.co/api/models/{MODEL}/revision/{revision}?blobs=true"
    with urllib.request.urlopen(api, timeout=60) as response:
        metadata = json.load(response)
    commit = metadata["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"invalid resolved revision: {commit!r}")
    siblings = {entry["rfilename"]: entry for entry in metadata["siblings"]}
    manifest = {"model": MODEL, "revision": commit, "files": {}}
    print(f"Downloading {MODEL} at {commit} to {destination}", flush=True)
    for filename in FILES:
        entry = siblings[filename]
        size = entry["size"]
        digest = entry.get("lfs", {}).get("sha256")
        if filename == "model.safetensors" and not digest:
            raise ValueError("upstream weight SHA-256 missing; refusing unverified weights")
        url = f"https://huggingface.co/{MODEL}/resolve/{commit}/{filename}?download=true"
        path = destination / filename
        print(f"  {filename}: {size / 1_000_000:.2f} MB", flush=True)
        download(url, path, size, digest)
        manifest["files"][filename] = {"size": size, "sha256": sha256(path),
                                       "upstream_sha256_verified": digest is not None}
    pending = destination / "download-manifest.json.part"
    pending.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(pending, destination / "download-manifest.json")
    print("Download complete. Immutable revision and file hashes recorded.")


if __name__ == "__main__":
    main()
