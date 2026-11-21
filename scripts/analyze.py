"""Turn results/scores_*.csv into baseline, condition, regression and paired-test tables."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rhw import config, data  # noqa: E402
from rhw.metrics import acc_at, auroc, bootstrap_ci, common_threshold, paired_bootstrap_diff  # noqa: E402

CONTROL = "control"
AGENT_SOURCES = ("malt", "trace")
R = config.RESULTS_DIR


def load_scores(suffix=""):
    """{condition: DataFrame(id, source, label, n_tokens, truncated, score)}; sliding gets _max/_mean."""
    out = {}
    for f in sorted(R.glob("scores_*.csv")):
        name = f.stem[len("scores_"):]
        if suffix:
            if not name.endswith(suffix):
                continue
            name = name[: -len(suffix)]
        elif "_Qwen" in name:
            continue
        df = pd.read_csv(f)
        if name.startswith("sliding"):
            for how in ("max", "mean"):
                out[f"{name}_{how}"] = df.assign(score=df[f"score_{how}"])
        else:
            out[name] = df.assign(score=df["score_max"])
    return out


def split(df, ids):
    return df[df["id"].isin(ids)]


def per_seed_metrics(scores, splits):
    rows = []
    for cond, df in scores.items():
        for seed, ids in splits.items():
            d = split(df, ids)
            ctrl = d[d.source == CONTROL]
            thr = common_threshold(ctrl.label, ctrl.score) if ctrl.label.nunique() == 2 else 0.5
            for src, g in d.groupby("source"):
                if g.label.nunique() < 2:
                    a, lo, hi = np.nan, np.nan, np.nan
                else:
                    a = auroc(g.label, g.score)
                    lo, hi = bootstrap_ci(auroc, g.label, g.score, n=1000, seed=seed)
                rows.append({"condition": cond, "source": src, "seed": seed, "auroc": a,
                             "auroc_lo": lo, "auroc_hi": hi, "threshold": thr,
                             "acc_common_thr": acc_at(g.label, g.score, thr),
                             "tpr_common_thr": float((g.score[g.label == 1] >= thr).mean()),
                             "fpr_common_thr": float((g.score[g.label == 0] >= thr).mean()), "n": len(g),
                             "n_pos": int(g.label.sum())})
    return pd.DataFrame(rows)


def condition_table(m):
    t = m.groupby(["condition", "source"]).agg(
        auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
        acc_mean=("acc_common_thr", "mean"), acc_std=("acc_common_thr", "std"),
        tpr_mean=("tpr_common_thr", "mean"), fpr_mean=("fpr_common_thr", "mean"), n=("n", "mean")
    ).reset_index()
    ctrl = t[t.source == CONTROL].set_index("condition")["acc_mean"]
    t["gap_vs_control"] = t["condition"].map(ctrl) - t["acc_mean"]
    return t


def best_window(t):
    w = t[(t.condition != "full") & t.source.isin(AGENT_SOURCES)]
    return w.groupby("condition")["auroc_mean"].mean().idxmax()


def length_regression(scores, splits, m, conds):
    import statsmodels.formula.api as smf

    rows = []
    for cond in conds:
        d = split(scores[cond], splits[0]).copy()
        thr = m[(m.condition == cond) & (m.seed == 0)]["threshold"].iloc[0]
        d["correct"] = ((d.score >= thr).astype(int) == d.label).astype(int)
        d["log_tokens"] = np.log(d.n_tokens)
        # second spec adds the label: an always-NO detector is "correct" on every negative
        for formula in ("correct ~ log_tokens + C(source)", "correct ~ log_tokens + C(source) + label"):
            fit = smf.logit(formula, data=d).fit(disp=0)
            ci = fit.conf_int()
            for term in fit.params.index:
                rows.append({"condition": cond, "formula": formula, "term": term,
                             "coef": fit.params[term], "ci_lo": ci.loc[term, 0], "ci_hi": ci.loc[term, 1],
                             "p_value": fit.pvalues[term], "n": int(fit.nobs)})
    return pd.DataFrame(rows)


def paired_tests(scores, splits, best):
    rows = []
    for seed, ids in splits.items():
        a = split(scores[best], ids).set_index("id")
        b = split(scores["full"], ids).set_index("id")
        for src in AGENT_SOURCES:
            idx = a.index[a.source == src].intersection(b.index)
            if len(idx) == 0 or a.loc[idx, "label"].nunique() < 2:
                continue
            y = a.loc[idx, "label"].values
            point, lo, hi = paired_bootstrap_diff(auroc, y, a.loc[idx, "score"].values,
                                                  b.loc[idx, "score"].values, n=1000, seed=seed)
            rows.append({"source": src, "seed": seed, "comparison": f"{best} - full",
                         "auroc_diff": point, "ci_lo": lo, "ci_hi": hi, "n": len(idx)})
    return pd.DataFrame(rows)


def truncated_subset(scores, splits):
    rows = []
    df = scores["full"]
    for seed, ids in splits.items():
        d = split(df, ids)
        for (src, trunc), g in d.groupby(["source", "truncated"]):
            a = auroc(g.label, g.score) if g.label.nunique() == 2 else np.nan
            rows.append({"source": src, "seed": seed, "truncated": trunc, "auroc": a,
                         "n": len(g), "n_pos": int(g.label.sum())})
    return pd.DataFrame(rows)


def trace_categories(df):
    """AUROC of each TRACE hack subcategory (codes like 1.1.2) against all clean TRACE runs."""
    d = df[df.source == "trace"]
    neg = d[d.label == 0]
    pos = d[d.label == 1].assign(code=lambda x: x.category.str.split(r",\s*")).explode("code")
    rows = []
    for code, g in pos.groupby("code"):
        y = np.r_[np.ones(len(g)), np.zeros(len(neg))]
        s = np.r_[g.score.values, neg.score.values]
        rows.append({"code": code, "n_pos": len(g), "n_neg": len(neg), "auroc": auroc(y, s),
                     "mean_score_pos": g.score.mean(), "median_tokens_pos": g.n_tokens.median()})
    return pd.DataFrame(rows).sort_values("code")


def main():
    units = data.load_units()
    splits = {s: data.sample_split(units, s) for s in config.SEEDS}
    scores = load_scores()
    if "full" not in scores:
        sys.exit("no results/scores_full.csv yet; run scripts/score.py first")

    m = per_seed_metrics(scores, splits)
    m.to_csv(R / "per_seed_metrics.csv", index=False)
    m[m.condition == "full"][["source", "seed", "auroc", "auroc_lo", "auroc_hi",
                              "acc_common_thr", "n"]].to_csv(R / "baseline.csv", index=False)
    truncated_subset(scores, splits).to_csv(R / "truncated_subset.csv", index=False)

    t = condition_table(m)
    t.to_csv(R / "conditions.csv", index=False)
    print(t.to_string(index=False))
    trace_categories(scores["full"]).to_csv(R / "trace_categories.csv", index=False)
    if len(scores) == 1:
        print("only the full condition is scored; skipping window comparisons")
        length_regression(scores, splits, m, ["full"]).to_csv(R / "length_regression.csv", index=False)
        return
    best = best_window(t)
    print(f"best windowed condition (mean MALT+TRACE AUROC): {best}")
    pd.DataFrame([{"best_window": best}]).to_csv(R / "best_window.csv", index=False)
    length_regression(scores, splits, m, ["full", best]).to_csv(R / "length_regression.csv", index=False)
    paired_tests(scores, splits, best).to_csv(R / "paired_test.csv", index=False)

    small = load_scores(suffix="_" + config.SMALL_MODEL.split("/")[-1])
    if small:
        ms = per_seed_metrics(small, splits)
        condition_table(ms).to_csv(R / "conditions_1.5B.csv", index=False)


if __name__ == "__main__":
    main()
