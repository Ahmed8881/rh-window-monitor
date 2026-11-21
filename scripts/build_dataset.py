"""Build data/units.jsonl and results/dataset_stats.csv from the raw sources."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from rhw import config, data, windows  # noqa: E402

if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(config.MAIN_MODEL)
    units = data.load_all()
    for u in tqdm(units, desc="tokenizing"):
        u["n_tokens"] = len(tok.encode("\n\n".join(windows.render_turn(t) for t in u["turns"])))

    config.UNITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.UNITS_PATH, "w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(u) + "\n")

    scored = set().union(*(data.sample_split(units, s) for s in config.SEEDS))
    df = pd.DataFrame([{k: u[k] for k in ("id", "source", "label", "n_turns", "n_tokens")} for u in units])
    df["truncated"] = df.n_tokens > config.MAX_TOKENS
    df["scored"] = df.id.isin(scored)
    stats = df.groupby("source").agg(
        n_units=("id", "size"), n_pos=("label", "sum"),
        n_neg=("label", lambda s: int((s == 0).sum())),
        median_turns=("n_turns", "median"), median_tokens=("n_tokens", "median"),
        pct_truncated=("truncated", lambda s: round(100 * s.mean(), 1)),
        n_scored_any_seed=("scored", "sum"),
    ).reset_index()
    config.RESULTS_DIR.mkdir(exist_ok=True)
    stats.to_csv(config.RESULTS_DIR / "dataset_stats.csv", index=False)
    print(stats.to_string(index=False))
    print(f"wrote {len(units)} units to {config.UNITS_PATH}")
