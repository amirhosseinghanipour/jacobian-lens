#!/usr/bin/env python3
"""Summarize frozen held-out measurements with paired, task-stratified bootstrap."""
import argparse
import json
import random
from pathlib import Path
from statistics import mean


def metric(item, method, k=10, layer=None):
    rows = item['ranks'][method]
    ranks = [rows[str(layer)]] if layer is not None else list(rows.values())
    best = [min((row[i] for row in ranks if row[i] is not None), default=None)
            for i in range(len(item['concepts']))]
    return mean(float(r is not None and r < k) if k else
                (1 / (r + 1) if r is not None else 0) for r in best)


def paired_interval(groups, method, baseline, k=10, repeats=2000):
    # Preserve prompt pairing between methods and equal weight between tasks.
    differences = [[metric(i, method, k)-metric(i, baseline, k) for i in items]
                   for items in groups.values()]
    if not differences or any(not values for values in differences):
        raise ValueError('all tasks must contain examples')
    rng = random.Random(20261105)
    boot = sorted(mean(mean(rng.choices(values, k=len(values))) for values in differences)
                  for _ in range(repeats))
    return {'difference': mean(mean(values) for values in differences),
            'paired_bootstrap_95_percent': [boot[int(repeats*.025)], boot[int(repeats*.975)-1]],
            'replicates': repeats}


def summarize(raw, splits):
    output = {}
    methods = list(raw['items'][0]['ranks'])
    for split in ['development', 'test']:
        groups = {}
        for item in raw['items']:
            if splits['lens-eval-'+item['task']][item['name']] == split:
                groups.setdefault(item['task'], []).append(item)
        tasks = {}
        for task, items in groups.items():
            capabilities = [i['capability'] for i in items if i['capability'] is not None]
            tasks[task] = {'items': len(items), 'single_token_concept_coverage':
                sum(bool(t) for i in items for t in i['concept_token_ids']) /
                sum(len(i['concepts']) for i in items),
                'exact_greedy_target_prefix_accuracy': mean(c['exact_greedy_prefix'] for c in capabilities) if capabilities else None,
                'methods': {m: {'recall': {str(k): mean(metric(i,m,k) for i in items) for k in [1,5,10]},
                    'mean_reciprocal_rank': mean(metric(i,m,0) for i in items),
                    'per_layer': {str(l): {str(k): mean(metric(i,m,k,l) for i in items) for k in [1,5,10]} for l in raw['layers']}}
                    for m in methods}}
        output[split] = {'tasks': tasks, 'primary_comparisons':
            {m: paired_interval(groups,m,'logit-lens') for m in methods if m != 'logit-lens'}}
        if 'replicate-100' in methods and 'main-100' in methods:
            output[split]['replication_difference'] = paired_interval(groups,'replicate-100','main-100')
        if 'main-1000' in methods and 'main-100' in methods:
            output[split]['calibration_size_difference'] = paired_interval(groups,'main-1000','main-100')
    return output


def summarize_causal(raw):
    measurements = raw['measurements']
    output = {'skipped': raw['skipped'], 'endpoint': raw['endpoint'], 'layers': {}}
    for layer in sorted({r['layer'] for r in measurements}):
        rows = [r for r in measurements if r['layer'] == layer]
        conditions = {}
        for condition in sorted({r['condition'] for r in rows}):
            values = [r for r in rows if r['condition'] == condition]
            eligible = [r for r in values if not r['answer_tokens_identical']]
            capable = [r for r in eligible if r['source_task_answer_rank'] == 0 and r['counterfactual_task_answer_rank'] == 0]
            conditions[condition] = {
                'items': len(values), 'distinct_answer_items': len(eligible),
                'mean_counterfactual_log_odds_change': mean(r['counterfactual_log_odds_change'] for r in eligible) if eligible else None,
                'mean_kl': mean(r['counterfactual_measurement']['kl_before_after'] for r in values),
                'both_unmodified_tasks_correct_items': len(capable),
                'capability_conditioned_log_odds_change': mean(r['counterfactual_log_odds_change'] for r in capable) if capable else None,
            }
        # Pair each concept swap with its three norm-matched controls. Resample
        # entire function templates to preserve argument dependence within them.
        clusters = {}
        for row in rows:
            if row['condition'] != 'swap-1' or row['answer_tokens_identical']:
                continue
            controls = [r for r in rows if r['item'] == row['item'] and r['condition'].startswith('random-swap-norm-')]
            if len(controls) != 3:
                raise ValueError('missing matched causal controls')
            difference = row['counterfactual_log_odds_change'] - mean(r['counterfactual_log_odds_change'] for r in controls)
            clusters.setdefault((row['category'], row['function']), []).append(difference)
        values = list(clusters.values())
        rng = random.Random(20261105)
        boot = sorted(mean(mean(cluster) for cluster in rng.choices(values, k=len(values))) for _ in range(2000)) if values else []
        output['layers'][str(layer)] = {'conditions': conditions, 'swap_minus_random': {
            'template_clusters': len(values), 'mean_log_odds_difference': mean(map(mean, values)) if values else None,
            'template_bootstrap_95_percent': [boot[50], boot[1949]] if boot else None}}
    return output

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('raw', type=Path);p.add_argument('splits', type=Path);p.add_argument('output', type=Path);p.add_argument('--causal', type=Path)
    a=p.parse_args()
    splits=json.loads(a.splits.read_text())
    amendment_path=Path('experiments/qwen-calibration/protocol-amendments.json')
    amendments=json.loads(amendment_path.read_text())
    for task,names in amendments['development_exclusions'].items():
        for name in names:
            splits[task][name]='development'
    result=summarize(json.loads(a.raw.read_text()),splits)
    result['protocol_amendments']=amendments
    causal_path = a.causal or a.raw.with_name('causal.json')
    if causal_path.exists():
        result['causal'] = summarize_causal(json.loads(causal_path.read_text()))
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['test']['primary_comparisons'],indent=2))

if __name__ == '__main__':
    main()
