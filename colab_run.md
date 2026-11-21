# Running the scoring on Google Colab (free T4)
nEasiest: open the ready notebook https://colab.research.google.com/github/Ahmed8881/rh-window-monitor/blob/main/colab_run.ipynb and run its cells top to bottom. The cells below are the same steps.

Runtime -> Change runtime type -> T4 GPU. Paste each block into its own cell.

```python
!git clone https://github.com/Ahmed8881/rh-window-monitor.git
%cd rh-window-monitor
!pip install -q -r requirements.txt "bitsandbytes>=0.44"
```

```python
# required: TRACE is gated (accept its terms on the HF page first)
from huggingface_hub import notebook_login
notebook_login()
```

```python
# rebuilds data/units.jsonl (not committed, it is large)
!python scripts/download.py
!python scripts/build_dataset.py
```

```python
# optional: restore a cache from an earlier session so nothing is rescored
# upload cache.zip with the file browser first
!unzip -oq cache.zip
```

```python
# baseline: full transcripts with the 7B detector (resumable, cached in cache/)
!python scripts/score.py --model Qwen/Qwen2.5-7B-Instruct --condition full --sources all
```

```python
# windowed conditions
!python scripts/score.py --model Qwen/Qwen2.5-7B-Instruct --condition last_2,last_4,last_8,tool_only
!python scripts/score.py --model Qwen/Qwen2.5-7B-Instruct --condition sliding_2,sliding_4,sliding_8
```

```python
# size ablation: full + best windowed condition (replace BEST after running analyze.py)
!python scripts/analyze.py
!python scripts/score.py --model Qwen/Qwen2.5-1.5B-Instruct --condition full,BEST
```

```python
# save progress: download the cache and the result CSVs
!zip -qr cache.zip cache results
from google.colab import files
files.download("cache.zip")
```

If the session disconnects, rerun the setup cells, restore `cache.zip`, and rerun the
score cell. Texts already in `cache/` are skipped.
