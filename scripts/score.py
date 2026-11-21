"""Score units under one or more conditions; results go to results/scores_<condition>.csv.

python scripts/score.py --model Qwen/Qwen2.5-7B-Instruct --condition full --sources all
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from rhw import config, data, windows  # noqa: E402
from rhw.detector import Cache, Detector, aggregate, key  # noqa: E402

def scores_path(model, condition):
    suffix = "" if model == config.MAIN_MODEL else "_" + model.split("/")[-1]
    return config.RESULTS_DIR / f"scores_{condition}{suffix}.csv"

def save_cost(row):
    """Upsert one (model, condition) row, written per condition so a disconnect loses nothing."""
    path = config.RESULTS_DIR / "cost.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=list(row))
    old = old[~((old.model == row["model"]) & (old.condition == row["condition"]))]
    pd.concat([old, pd.DataFrame([row])]).to_csv(path, index=False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=config.MAIN_MODEL)
    p.add_argument("--condition", default="full", help="condition name, comma list, or 'all'")
    p.add_argument("--sources", default="all", help="comma list of sources or 'all'")
    p.add_argument("--seeds", default=",".join(map(str, config.SEEDS)))
    p.add_argument("--limit", type=int, default=None, help="score only the first N units (debug)")
    a = p.parse_args()

    conds = windows.conditions()
    names = list(conds) if a.condition == "all" else a.condition.split(",")
    units = data.load_units()
    ids = set().union(*(data.sample_split(units, int(s)) for s in a.seeds.split(",")))
    units = [u for u in units if u["id"] in ids]
    if a.sources != "all":
        units = [u for u in units if u["source"] in a.sources.split(",")]
    units = units[: a.limit]

    det = Detector(a.model)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    for name in names:
        cache = Cache(a.model, name)
        rows, seconds, calls = [], 0.0, 0
        for u in tqdm(units, desc=name):
            texts = conds[name](u, det.tok)
            s = [cache.score(det, t) for t in texts]
            seconds += sum(cache.data[key(t)]["seconds"] for t in texts)
            calls += len(texts)
            rows.append({"id": u["id"], "source": u["source"], "label": u["label"],
                         "category": u["category"], "n_tokens": u["n_tokens"],
                         "n_turns": u["n_turns"], "truncated": u["n_tokens"] > config.MAX_TOKENS,
                         "n_windows": len(texts), "score_max": aggregate(s, "max"),
                         "score_mean": aggregate(s, "mean")})
        pd.DataFrame(rows).to_csv(scores_path(a.model, name), index=False)
        save_cost({"model": a.model, "condition": name, "n_units": len(units),
                   "detector_calls": calls, "gpu_seconds": round(seconds, 1)})
        print(f"{name}: {len(rows)} units, {calls} calls, {seconds:.0f}s")

if __name__ == "__main__":
    main()
