#!/usr/bin/env python3
"""Audit saved calibration artifacts against independent eager PyTorch execution.

Run from the repository root. Requires torch, transformers and safetensors.
Checks all shard merges, selected gradient rows in four complete shards, two
readout items per task, and all swaps/ablations for one pair per function.
Sampling is fixed by input order, never by observed effect size.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


LAYERS = [6, 10, 14, 18, 22]
DIMS = [0, 1, 7, 31, 127, 255, 511, 895]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual, expected, atol, rtol):
    actual, expected = torch.as_tensor(actual).double(), torch.as_tensor(expected).double()
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    error = (actual - expected).abs()
    assert (error <= atol + rtol * expected.abs()).all(), (
        f"max absolute error {error.max().item()}, tolerance {atol} + {rtol}*abs(expected)"
    )
    return {"max_absolute_error": error.max().item(),
            "relative_l2_error": (error.norm() / expected.norm().clamp_min(1e-30)).item()}


def rank(logits, token):
    ids = torch.arange(logits.numel(), device=logits.device)
    return int(((logits > logits[token]) | ((logits == logits[token]) & (ids < token))).sum())


def audit_artifacts(model, corpus, results):
    manifest = read(corpus / "manifest.json")
    identity = hashlib.sha256()
    for name in ["model.safetensors", "config.json", "tokenizer.json"]:
        identity.update(name.encode())
        identity.update(sha(model / name).encode())
    assert sha(model / "tokenizer.json") == manifest["tokenizer_sha256"]
    for file, digest in manifest["files"].items():
        assert sha(corpus / file) == digest, file
    for file, digest in manifest["evaluation_sha256"].items():
        assert sha(file) == digest, file
    groups = {"main-100": range(10), "main-1000": range(100), "replicate-100": range(100, 110)}
    sums = {name: {l: torch.zeros(896, 896, dtype=torch.float64) for l in LAYERS}
            for name in groups}
    records = []
    identities = set()
    for i in range(110):
        path = results / f"shard-{i:03}.safetensors"
        with safe_open(path, framework="pt") as f:
            meta = f.metadata()
            assert int(meta["n_prompts"]) == 10
            assert meta["corpus_id"] == sha(corpus / f"shard-{i:03}.json")
            identities.add(meta["model_id"])
            config = json.loads(meta["fit_config"])
            assert config == dict(source_layers=LAYERS, target_layer=23, skip_first=16,
                                  max_seq_len=128, dim_batch=4)
            assert set(f.keys()) == {f"jacobians.{l}" for l in LAYERS}
            for l in LAYERS:
                j = f.get_tensor(f"jacobians.{l}")
                assert j.shape == (896, 896) and torch.isfinite(j).all()
                for name, indices in groups.items():
                    if i in indices:
                        sums[name][l] += j.double()
        records.extend(read(corpus / f"shard-{i:03}.json"))
    assert identities == {identity.hexdigest()}
    assert len(records) == len({r["document_id"] for r in records}) == 1100
    assert len({r["document_sha256"] for r in records}) == 1100
    comparisons = {}
    for name, indices in groups.items():
        with safe_open(results / f"{name}.safetensors", framework="pt") as f:
            assert f.metadata()["model_id"] in identities
            assert int(f.metadata()["n_prompts"]) == len(indices) * 10
            comparisons[name] = {str(l): close(f.get_tensor(f"jacobians.{l}"),
                                               sums[name][l] / len(indices), 2e-6, 2e-6)
                                 for l in LAYERS}
    causal = read(results / "causal.json")
    assert causal["skipped"] == []
    rows = causal["measurements"]
    assert len(rows) == 2560
    keys = {(r["item"], r["layer"], r["condition"]) for r in rows}
    assert len(keys) == len(rows)
    pairs = {r["item"] for r in rows}
    conditions = {"swap-0", "swap-0.5", "swap-1", "concept-ablation",
                  "random-span-ablation-matched-norm", *[f"random-swap-norm-{i}" for i in (1, 2, 3)]}
    assert keys == {(p, l, c) for p in pairs for l in LAYERS for c in conditions}
    for r in rows:
        if r["condition"] == "swap-0":
            assert r["perturbation_norm"] == r["counterfactual_log_odds_change"] == 0
            assert r["counterfactual_measurement"]["kl_before_after"] == 0
        if r["condition"].startswith("random-"):
            reference = "concept-ablation" if "ablation" in r["condition"] else "swap-1"
            target = next(x for x in rows if x["item"] == r["item"] and
                          x["layer"] == r["layer"] and x["condition"] == reference)
            close(r["perturbation_norm"], target["perturbation_norm"], 2e-5, 2e-5)
    return {"unique_calibration_documents": len(records), "shards": 110,
            "model_id": identity.hexdigest(),
            "merged_matrices": comparisons, "causal_records": len(rows),
            "identity_controls": 320, "matched_norm_controls": 1280}


def audit_statistics(corpus, results):
    raw, summary = read(results / "readouts.json"), read(results / "summary.json")
    splits = read(corpus / "splits.json")
    amendments = read("experiments/qwen-calibration/protocol-amendments.json")
    assert summary["protocol_amendments"] == amendments
    for task, names in amendments["development_exclusions"].items():
        for name in names:
            splits[task][name] = "development"
    keys = {(i["task"], i["name"]) for i in raw["items"]}
    expected = {(task, i["name"]) for task in ["association", "multihop", "multilingual", "order-ops", "poetry", "typo"]
                for i in read(f"data/evaluations/lens-eval-{task}.json")["items"]}
    assert keys == expected and len(keys) == len(raw["items"]) == 551
    counts = {}
    for split in ["development", "test"]:
        counts[split] = 0
        macros = {}
        for task, table in summary[split]["tasks"].items():
            items = [i for i in raw["items"] if i["task"] == task and
                     splits["lens-eval-" + task][i["name"]] == split]
            assert len(items) == table["items"]
            counts[split] += len(items)
            for method, metrics in table["methods"].items():
                scores = []
                for i in items:
                    matrix = i["ranks"][method]
                    assert set(matrix) == set(map(str, LAYERS))
                    assert all(len(row) == len(i["concepts"]) for row in matrix.values())
                    scores.append([min((matrix[str(l)][c] for l in LAYERS
                                        if matrix[str(l)][c] is not None), default=float("inf"))
                                   for c in range(len(i["concepts"]))])
                for k in [1, 5, 10]:
                    recall = sum(sum(r < k for r in item) / len(item) for item in scores) / len(scores)
                    assert abs(recall - metrics["recall"][str(k)]) < 1e-12
                    if k == 10:
                        macros.setdefault(method, []).append(recall)
                mrr = sum(sum(1 / (r + 1) for r in item) / len(item) for item in scores) / len(scores)
                assert abs(mrr - metrics["mean_reciprocal_rank"]) < 1e-12
        for method, comparison in summary[split]["primary_comparisons"].items():
            difference = sum(macros[method]) / 6 - sum(macros["logit-lens"]) / 6
            assert abs(difference - comparison["difference"]) < 1e-12
    assert counts == {"development": 112, "test": 439}
    return {"unique_evaluation_items": len(keys), "split_counts": counts,
            "recall_and_mrr_recomputed": True, "task_macro_differences_recomputed": True}


def capture(model, ids, *, gradients=False, edit=None):
    states, handles = {}, []
    def hook(layer):
        def record(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if gradients and layer == LAYERS[0]:
                h = h.detach().requires_grad_(True)
            if edit is not None and layer == edit[0]:
                h = h.clone()
                h[:, -1] = edit[1]
            states[layer] = h
            return (h, *output[1:]) if isinstance(output, tuple) else h
        return record
    try:
        for layer in [*LAYERS, 23]:
            handles.append(model.model.layers[layer].register_forward_hook(hook(layer)))
        with torch.set_grad_enabled(gradients):
            # Decoder output avoids materializing logits at every prompt position.
            model.model(input_ids=ids, use_cache=False)
        return states
    finally:
        for handle in handles:
            handle.remove()


def audit_gradients(model, corpus, results, device):
    output = {}
    for shard in [0, 55, 99, 100]:
        totals = {l: torch.zeros(len(DIMS), 896, device=device) for l in LAYERS}
        prompts = read(corpus / f"shard-{shard:03}.json")
        for prompt in prompts:
            ids = torch.tensor([prompt["ids"]], device=device)
            states = capture(model, ids, gradients=True)
            for row, dimension in enumerate(DIMS):
                objective = states[23][:, 16:-1, dimension].sum()
                gradients = torch.autograd.grad(objective, [states[l] for l in LAYERS],
                                                retain_graph=row < len(DIMS) - 1)
                for l, g in zip(LAYERS, gradients):
                    totals[l][row] += g[0, 16:-1].mean(0)
            del states, gradients, objective
        stored = load_file(results / f"shard-{shard:03}.safetensors", device=device)
        output[str(shard)] = {str(l): close(stored[f"jacobians.{l}"][DIMS],
                                           totals[l] / len(prompts), 1e-3, 2e-3)
                              for l in LAYERS}
        # Also require small aggregate relative error; elementwise absolute
        # tolerance alone could conceal corruption in small gradient entries.
        assert all(x["relative_l2_error"] < 1e-3 for x in output[str(shard)].values())
        print(f"Gradient parity: shard {shard}, all five layers passed", flush=True)
    return {"shards": output, "output_dimensions": DIMS, "prompts": 40,
            "compared_entries": 4 * 5 * len(DIMS) * 896}


def readout_ids(tokenizer, item, task):
    text = item["prompt"]
    if task in ["multihop", "multilingual", "order-ops"]:
        encoding = tokenizer(text + item["target"], add_special_tokens=False,
                             return_offsets_mapping=True)
        end = next(i for i, (_, b) in enumerate(encoding["offset_mapping"]) if b > len(text))
    else:
        encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        end = len(encoding["input_ids"])
        if task == "poetry":
            newline = text.rindex("\n")
            end = max(i for i, (a, b) in enumerate(encoding["offset_mapping"])
                      if a <= newline < b) + 1
    return encoding["input_ids"][:end]


@torch.no_grad()
def audit_readouts(model, tokenizer, results, device):
    raw = read(results / "readouts.json")
    lenses = {name: load_file(results / f"{name}.safetensors", device=device)
              for name in ["main-100", "main-1000", "replicate-100"]}
    comparisons, disagreements, max_rank_error = 0, 0, 0
    for task in sorted({i["task"] for i in raw["items"]}):
        dataset = read(f"data/evaluations/lens-eval-{task}.json")["items"]
        for original in dataset[:2]:
            saved = next(i for i in raw["items"] if i["task"] == task and i["name"] == original["name"])
            ids = readout_ids(tokenizer, original, task)
            assert len(ids) - 1 == saved["readout_position"]
            tokens = [sorted({tokenizer.encode(v, add_special_tokens=False)[0]
                             for v in [c, " " + c] if len(tokenizer.encode(v, add_special_tokens=False)) == 1})
                      for c in original["intermediates"]]
            assert tokens == saved["concept_token_ids"]
            states = capture(model, torch.tensor([ids], device=device))
            for method in ["logit-lens", *lenses]:
                for l in LAYERS:
                    h = states[l][:, -1]
                    if method != "logit-lens":
                        h = h @ lenses[method][f"jacobians.{l}"].T
                    logits = model.lm_head(model.model.norm(h)).flatten()
                    for ts, expected in zip(tokens, saved["ranks"][method][str(l)]):
                        actual = min((rank(logits, t) for t in ts), default=None)
                        comparisons += 1
                        if actual is None or expected is None:
                            assert actual == expected
                            continue
                        error = abs(actual - expected)
                        max_rank_error = max(max_rank_error, error)
                        disagreements += actual != expected
                        # FP32 backend rounding may reorder near-ties deep in the
                        # vocabulary. It must not change reported recall here.
                        assert error <= max(3, expected * .001), (task, original["name"], method, l, actual, expected)
                        assert all((actual < k) == (expected < k) for k in (1, 5, 10))
        print(f"Readout parity: {task} passed", flush=True)
    return {"items": 12, "concept_rank_comparisons": comparisons,
            "nonidentical_ranks": disagreements, "max_rank_difference": max_rank_error,
            "recall_at_1_5_10_disagreements": 0}


@torch.no_grad()
def audit_interventions(model, tokenizer, results, device):
    stored = read(results / "causal.json")["measurements"]
    lens = load_file(results / "main-1000.safetensors", device=device)
    comparisons = []
    for category in read("data/experiments/flexible-generalization.json")["categories"]:
        source, target = category["args"][:2]
        for function in category["funcs"]:
            key = f"{category['name']}:{function['name']}:{source}->{target}"
            text = function["template"].replace("{arg}", source)
            ids = torch.tensor([tokenizer.encode(text, add_special_tokens=False)], device=device)
            states = capture(model, ids)
            before = model.lm_head(model.model.norm(states[23][:, -1])).flatten().double()
            concept_ids = []
            for word in [source, target]:
                concept_ids.append(next(ts[0] for ts in
                                        [tokenizer.encode(v, add_special_tokens=False) for v in [" " + word, word]]
                                        if len(ts) == 1))
            for l in LAYERS:
                basis = model.lm_head.weight[concept_ids] @ lens[f"jacobians.{l}"]
                dual = torch.linalg.pinv(basis.double(), rtol=1e-6).float()
                h = states[l][:, -1]
                coords = h @ dual
                swap = (coords[:, [1, 0]] - coords) @ basis
                for condition, delta in [("swap-0", swap * 0), ("swap-0.5", swap * .5),
                                         ("swap-1", swap), ("concept-ablation", -(coords @ basis))]:
                    row = next(r for r in stored if r["item"] == key and r["layer"] == l and r["condition"] == condition)
                    edited = capture(model, ids, edit=(l, h + delta))
                    after = model.lm_head(model.model.norm(edited[23][:, -1])).flatten().double()
                    o, c = row["original_answer_token"], row["counterfactual_answer_token"]
                    for answer, token in [(row["original_answer"], o), (row["counterfactual_answer"], c)]:
                        full = tokenizer.encode(text + " " + answer, add_special_tokens=False)
                        assert full[:ids.shape[1]] == ids[0].tolist() and full[ids.shape[1]] == token
                    logodds = (after[c] - after[o]) - (before[c] - before[o])
                    error = close(float(logodds), row["counterfactual_log_odds_change"], 5e-3, 2e-3)
                    close(float(delta.norm()), row["perturbation_norm"], 2e-4, 2e-4)
                    lb, la = before.log_softmax(0), after.log_softmax(0)
                    for token, field in [(c, "counterfactual_measurement"), (o, "original_measurement")]:
                        m = row[field]
                        close(float(lb[token].exp()), m["probability_before"], 2e-4, 2e-3)
                        close(float(la[token].exp()), m["probability_after"], 2e-4, 2e-3)
                        close(float(after[token] - before[token]), m["target_logit_delta"], 5e-3, 2e-3)
                        close(float((lb.exp() * (lb - la)).sum()), m["kl_before_after"], 2e-4, 3e-3)
                    comparisons.append(error["max_absolute_error"])
            print(f"Causal parity: {key} passed", flush=True)
    return {"pairs": 16, "conditions_per_layer": 4, "layers": 5,
            "measurements": len(comparisons), "max_log_odds_absolute_error": max(comparisons)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("corpus", type=Path)
    p.add_argument("results", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = {"schema": 1, "torch": torch.__version__, "device": a.device,
              "inputs": {n: sha(a.results / n) for n in
                         ["main-1000.safetensors", "readouts.json", "causal.json", "summary.json"]}}
    report["artifacts"] = audit_artifacts(a.model, a.corpus, a.results)
    report["statistics"] = audit_statistics(a.corpus, a.results)
    print("Artifact integrity and all merge/identity/control checks passed", flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True,
                                               dtype=torch.float32, attn_implementation="eager").to(a.device).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    report["gradients"] = audit_gradients(model, a.corpus, a.results, a.device)
    report["readouts"] = audit_readouts(model, tokenizer, a.results, a.device)
    report["interventions"] = audit_interventions(model, tokenizer, a.results, a.device)
    report["passed"] = True
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"All independent audit checks passed: {a.output}", flush=True)


if __name__ == "__main__":
    main()
