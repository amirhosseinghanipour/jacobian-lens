#!/usr/bin/env python3
"""Compare disjoint calibration Jacobians, including their non-identity component."""
import argparse
import json
from pathlib import Path
import numpy as np


def tensors(path):
    with path.open('rb') as f:
        size = int.from_bytes(f.read(8), 'little')
        header = json.loads(f.read(size))
    return {key: np.memmap(path, dtype='<f4', mode='r', offset=8+size+value['data_offsets'][0], shape=tuple(value['shape']))
            for key, value in header.items() if key != '__metadata__' and value['dtype'] == 'F32'}


def compare(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return {'cosine': float(np.vdot(a, b)/(na*nb)) if na*nb else None,
            'relative_frobenius_difference': float(np.linalg.norm(a-b)/((na+nb)/2)) if na+nb else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('shards',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    main_sum={};rep_sum={}
    for sums, indices in [(main_sum,range(10)),(rep_sum,range(100,110))]:
        for i in indices:
            for key,value in tensors(a.shards/f'shard-{i:03}.safetensors').items():
                if key not in sums:sums[key]=np.zeros(value.shape,dtype=np.float64)
                sums[key]+=value/10
    if main_sum.keys()!=rep_sum.keys() or len(main_sum)!=5:
        raise ValueError('incomplete or mismatched layers')
    result={'prompts_per_sample':100,'comparison':'disjoint main and replication calibration samples','layers':{}}
    for key,first in main_sum.items():
        second=rep_sum[key];identity=np.eye(first.shape[0])
        result['layers'][key]={'full_jacobian':compare(first,second),'jacobian_minus_identity':compare(first-identity,second-identity)}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
