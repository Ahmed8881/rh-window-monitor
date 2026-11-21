"""Download the raw sources into data/raw/.

- rh-bench (RH-Detect's unit table, ungated): MALT units and the School of Reward Hacks control.
- TRACE original parquet (gated, auto-approved): needs `huggingface-cli login` or HF_TOKEN.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from rhw import config  # noqa: E402

FILES = {
    "RHBENCH": ("ktolnos/rh-bench", "data/rh_bench_unified.parquet", "rhbench"),
    "TRACE": ("PatronusAI/trace-dataset", "data/train-00000-of-00001.parquet", "trace"),
}

if __name__ == "__main__":
    for name, (repo, filename, sub) in FILES.items():
        out = config.RAW_DIR / sub
        target = out / ("train.parquet" if name == "TRACE" else Path(filename).name)
        try:
            if not target.exists():
                p = hf_hub_download(repo, filename, repo_type="dataset", local_dir=out)
                Path(p).replace(target)
            df = pd.read_parquet(target)
            print(f"{name}: {len(df)} rows, columns={df.columns.tolist()}")
            if name == "RHBENCH":
                for src, n in df.source_dataset.value_counts().items():
                    print(f"  {src}: {n} rows")
        except Exception as e:  # a gated / missing dataset must not stop the others
            print(f"{name}: FAILED to download {repo}: {type(e).__name__}: {str(e)[:300]}")
