"""Download TRACE, MALT and the single-turn control source into data/raw/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset  # noqa: E402

from rhw import config  # noqa: E402

SOURCES = {
    "TRACE": "PatronusAI/trace-dataset",
    "MALT": "metr-evals/malt-public",
    "CONTROL": "longtermrisk/school-of-reward-hacks",
}

if __name__ == "__main__":
    for name, repo in SOURCES.items():
        out = config.RAW_DIR / name.lower()
        try:
            ds = load_dataset(repo)
            ds.save_to_disk(str(out))
            for split, d in ds.items():
                print(f"{name}: {d.num_rows} rows (split={split}), columns={d.column_names}")
        except Exception as e:  # gated / missing datasets must not stop the others
            print(f"{name}: FAILED to download {repo}: {type(e).__name__}: {str(e)[:300]}")
