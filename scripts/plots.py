"""Figures from results/: AUROC vs length, gap by condition, ROC curves."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from analyze import AGENT_SOURCES, load_scores, split  # noqa: E402
from rhw import config, data  # noqa: E402
from rhw.metrics import auroc, bootstrap_ci  # noqa: E402

R, F = config.RESULTS_DIR, config.FIGURES_DIR


def auroc_vs_length(scores, ids, best):
    agent = split(scores["full"], ids)
    agent = agent[agent.source.isin(AGENT_SOURCES)]
    edges = np.quantile(agent.n_tokens, [0, 0.25, 0.5, 0.75, 1.0])
    rows = []
    fig, ax = plt.subplots(figsize=(6, 4))
    for cond in [c for c in ("full", best) if c]:
        d = split(scores[cond], ids)
        d = d[d.source.isin(AGENT_SOURCES)]
        bins = pd.cut(d.n_tokens, edges, include_lowest=True, labels=False)
        xs, ys, lo, hi = [], [], [], []
        for b in range(4):
            g = d[bins == b]
            if g.label.nunique() < 2:
                rows.append({"condition": cond, "quartile": b + 1, "auroc": np.nan, "n": len(g)})
                continue
            a = auroc(g.label, g.score)
            l, h = bootstrap_ci(auroc, g.label, g.score, n=1000, seed=0)
            xs.append(b + 1); ys.append(a); lo.append(a - l); hi.append(h - a)
            rows.append({"condition": cond, "quartile": b + 1, "auroc": a, "ci_lo": l, "ci_hi": h,
                         "n": len(g), "n_pos": int(g.label.sum()),
                         "tokens_lo": edges[b], "tokens_hi": edges[b + 1]})
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, label=cond)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels([f"Q{i}\n{int(edges[i-1])}-{int(edges[i])}" for i in range(1, 5)])
    ax.set_xlabel("transcript length quartile (tokens), MALT + TRACE, seed 0")
    ax.set_ylabel("AUROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(F / "auroc_vs_length.png", dpi=150)
    pd.DataFrame(rows).to_csv(R / "auroc_by_length.csv", index=False)


def gap_by_condition():
    t = pd.read_csv(R / "conditions.csv")
    t = t[t.source.isin(AGENT_SOURCES)].pivot(index="condition", columns="source", values="gap_vs_control")
    ax = t.plot.bar(figsize=(8, 4))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("accuracy gap (control - source)")
    ax.set_title("Accuracy gap at the common threshold, mean over seeds")
    plt.tight_layout()
    plt.savefig(F / "gap_by_condition.png", dpi=150)


def roc_malt_trace(scores, ids, best):
    fig, axes = plt.subplots(1, len(AGENT_SOURCES), figsize=(10, 4))
    for ax, src in zip(axes, AGENT_SOURCES):
        for cond in [c for c in ("full", best) if c]:
            d = split(scores[cond], ids)
            d = d[d.source == src]
            if d.label.nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(d.label, d.score)
            ax.plot(fpr, tpr, label=f"{cond} (AUROC {auroc(d.label, d.score):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(f"{src}, seed 0")
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(F / "roc_malt_trace.png", dpi=150)


if __name__ == "__main__":
    F.mkdir(exist_ok=True)
    units = data.load_units()
    ids = data.sample_split(units, 0)
    scores = load_scores()
    # before any windowed condition is scored, plot the full condition alone
    bw = R / "best_window.csv"
    best = pd.read_csv(bw)["best_window"].iloc[0] if bw.exists() else None
    auroc_vs_length(scores, ids, best)
    gap_by_condition()
    roc_malt_trace(scores, ids, best)
    print("wrote figures/auroc_vs_length.png, figures/gap_by_condition.png, figures/roc_malt_trace.png")
