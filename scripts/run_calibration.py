#!/usr/bin/env python3
"""Run resumable Rust calibration shards, merge lenses and measure held-out tasks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',type=Path);p.add_argument('corpus',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--binary',type=Path,default=Path('target/release/jlens-experiment'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.corpus/'manifest.json').read_text())
    for filename,expected in manifest['files'].items():
        if hashlib.sha256((a.corpus/filename).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'corpus file changed: {filename}')
    for filename,expected in manifest['evaluation_sha256'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'evaluation file changed: {filename}')
    env=os.environ.copy();env['JLENS_DEVICE']='cuda'
    # Fixed prefix and independent replication are completed before expansion.
    for shard in list(range(10))+list(range(100,110))+list(range(10,100)):
        subprocess.run([str(a.binary),'fit',str(a.model),str(a.corpus/f'shard-{shard:03}.json'),str(a.output/f'shard-{shard:03}.safetensors')],env=env,check=True)
    subprocess.run([str(a.binary),'merge',str(a.model),str(a.output),str(a.output)],env=env,check=True)
    subprocess.run([str(a.binary),'eval',str(a.model),str(a.output),str(a.output/'readouts.json')],env=env,check=True)
    subprocess.run([str(a.binary),'causal',str(a.model),str(a.output/'main-1000.safetensors'),str(a.output/'causal.json')],env=env,check=True)
    subprocess.run(['python3','scripts/summarize_calibration.py',str(a.output/'readouts.json'),str(a.corpus/'splits.json'),str(a.output/'summary.json')],check=True)

if __name__ == '__main__':
    main()
