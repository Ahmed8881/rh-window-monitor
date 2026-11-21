"""Rough GPU-time estimate per condition, scaled from the measured cost of the full condition.

Approximates tokens by characters (chars/token measured on the agent units) and assumes cost is
linear in tokens, with 150 prompt tokens per call. Writes results/cost_estimate.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from rhw import config, data, windows  # noqa: E402

PROMPT_TOKENS = 150

if __name__ == "__main__":
    units = data.load_units()
    ids = set().union(*(data.sample_split(units, s) for s in config.SEEDS))
    units = [u for u in units if u["id"] in ids]
    agent = [u for u in units if u["source"] != "control"]
    chars = sum(len("\n\n".join(windows.render_turn(t) for t in u["turns"])) for u in agent)
    cpt = chars / sum(u["n_tokens"] for u in agent)

    class CharTok:  # one token per cpt characters; fast stand-in for the real tokenizer
        n = max(1, round(cpt))

        def encode(self, s):
            return [s[i:i + self.n] for i in range(0, len(s), self.n)]

        def decode(self, ids):
            return "".join(ids)

    tok = CharTok()

    def cost(fn, subset):
        texts = {t: len(t) / cpt + PROMPT_TOKENS for u in subset for t in fn(u, tok)}
        return len(texts), sum(texts.values())

    measured = pd.read_csv(config.RESULTS_DIR / "cost.csv").query(
        "model == @config.MAIN_MODEL and condition == 'full'")["gpu_seconds"].iloc[0]
    _, full_tok = cost(windows.full, units)
    rate = measured / full_tok
    rows = []
    for name, fn in windows.conditions().items():
        # control texts are the same under every condition and already cached from full
        n, t = cost(fn, units if name == "full" else agent)
        rows.append({"condition": name, "new_calls": n, "approx_tokens_M": round(t / 1e6, 2),
                     "est_gpu_hours": round(t * rate / 3600, 2), "max_windows": config.MAX_WINDOWS})
    out = pd.DataFrame(rows)
    out.to_csv(config.RESULTS_DIR / "cost_estimate.csv", index=False)
    print(f"chars/token {cpt:.2f}, {rate * 1000:.3f} GPU-s per 1k tokens (from full)")
    print(out.to_string(index=False))
