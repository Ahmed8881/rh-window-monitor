"""Loaders that turn each source into units of a common schema.

unit = {id, source, label (1 = reward hacking, 0 = not), category,
        turns: [{role, content, is_tool}], n_task, n_turns, n_tokens}
n_task = number of leading turns that state the task (always shown to the detector).
See DATA.md for how each source's labels map to 0/1.
"""
import json

import numpy as np
import pandas as pd

from rhw import config

RHBENCH = config.RAW_DIR / "rhbench" / "rh_bench_unified.parquet"
TRACE = config.RAW_DIR / "trace" / "train.parquet"


def turn(role, content, is_tool=False):
    return {"role": role, "content": content or "", "is_tool": is_tool}


def make_unit(uid, source, label, category, turns, n_task=1):
    return {"id": uid, "source": source, "label": int(label), "category": category or "",
            "turns": turns, "n_task": n_task, "n_turns": len(turns), "n_tokens": None}


def load_trace():
    """TRACE from the original Patronus parquet: label '0' = benign, anything else = hack codes."""
    df = pd.read_parquet(TRACE)
    units = []
    for r in df.itertuples():
        turns = []
        for m in json.loads(r.conversation):
            calls = m.get("tool_calls") or []
            content = m.get("content") or ""
            if calls:
                content += "".join(f"\nCALL {c['name']} {json.dumps(c.get('parameters', {}))}" for c in calls)
            turns.append(turn(m["role"], content))
            for res in m.get("tool_results") or []:
                turns.append(turn("tool", str(res.get("output", res)), is_tool=True))
        label = 0 if str(r.label).strip() == "0" else 1
        units.append(make_unit(f"trace:{r.trajectory_id}", "trace", label, str(r.label), turns))
    return units


def _rhbench():
    return pd.read_parquet(RHBENCH)


def _rhbench_units(r, source, response_col, label, agentic):
    prompt = json.loads(r.prompt)
    turns = [turn(m["role"], m["content"]) for m in prompt]
    for m in json.loads(getattr(r, response_col)):
        # MALT agents have no human in the loop: every later user message is a tool output
        if agentic and m["role"] == "user":
            turns.append(turn("tool", m["content"], is_tool=True))
        else:
            turns.append(turn(m["role"], m["content"]))
    cat = r.hacking_subcategory if label else ""
    uid = f"{source}:{r.source_id}:{'hack' if label else 'clean'}"
    return make_unit(uid, source, label, cat, turns, n_task=len(prompt))


def load_malt(df=None):
    """MALT units from rh-bench. malt = unprompted hacks, malt_prompted = prompted hacks;
    both use the same pool of clean MALT runs as negatives."""
    df = _rhbench() if df is None else df
    m = df[df.source_dataset == "metr_malt"]
    units = []
    for r in m.itertuples():
        if r.label == "hacking":
            src = "malt" if r.construction == "unprompted" else "malt_prompted"
            units.append(_rhbench_units(r, src, "response_hacking", 1, True))
        elif r.label == "clean":
            for src in ("malt", "malt_prompted"):
                units.append(_rhbench_units(r, src, "response_clean", 0, True))
    return units


def load_control(df=None):
    """School of Reward Hacks (single-turn) from rh-bench: one unit per non-null response."""
    df = _rhbench() if df is None else df
    s = df[df.source_dataset == "school_of_reward_hacks"]
    units = []
    for r in s.itertuples():
        if r.response_hacking is not None:
            units.append(_rhbench_units(r, "control", "response_hacking", 1, False))
        if r.response_clean is not None:
            units.append(_rhbench_units(r, "control", "response_clean", 0, False))
    return units


def load_all():
    df = _rhbench()
    return load_trace() + load_malt(df) + load_control(df)


def load_units(path=config.UNITS_PATH):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def sample_split(units, seed):
    """Ids for one seed: all positives, plus up to NEG_RATIO x positives negatives (capped) per source."""
    rng = np.random.default_rng(seed)
    ids = set()
    for src in sorted({u["source"] for u in units}):
        pos = sorted(u["id"] for u in units if u["source"] == src and u["label"] == 1)
        neg = sorted(u["id"] for u in units if u["source"] == src and u["label"] == 0)
        k = min(len(neg), config.NEG_RATIO * len(pos), config.NEG_CAP)
        ids.update(pos)
        ids.update(rng.choice(neg, size=k, replace=False).tolist() if k else [])
    return ids
