# rh-window-monitor

Tests whether the multi-turn tool-use gap in RH-Detect is a transcript-length effect, using windowed LLM detectors on TRACE and MALT.

Work in progress: no results yet.

## Builds on

- Quan, Opryshko, Subramani, Gilitschenski. *RH-Detect: A Unified Benchmark for Reward Hacking Detection.* arXiv 2610.10947, 2026. https://arxiv.org/abs/2610.10947
- Deshpande, Kannappan, Qian. *Benchmarking Reward Hack Detection in Code Environments via Contrastive Analysis* (TRACE). https://arxiv.org/abs/2601.20103, data: https://huggingface.co/datasets/PatronusAI/trace-dataset
- METR. *MALT: A Dataset of Natural and Prompted Behaviors That Threaten Eval Integrity.* https://metr.org/blog/2025-10-14-malt-dataset-of-natural-and-prompted-behaviors/, data: https://huggingface.co/datasets/metr-evals/malt-public
- School of Reward Hacks (control source). https://huggingface.co/datasets/longtermrisk/school-of-reward-hacks

How each source's labels map to 0/1 is in [DATA.md](DATA.md).

## How to run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
huggingface-cli login                 # TRACE is gated (auto-approved after accepting its terms)
python scripts/download.py            # rh-bench (MALT + control) and TRACE -> data/raw/
python scripts/smoke_test.py          # 0.5B detector on CPU, 3 toy transcripts
python scripts/build_dataset.py       # data/units.jsonl, results/dataset_stats.csv
pytest -q
# scoring needs a GPU: see colab_run.md
python scripts/score.py --condition full
python scripts/analyze.py             # results/*.csv
python scripts/plots.py               # figures/*.png
```

## Related code

Status checked on 2026-11-21:

- RH-Detect (https://arxiv.org/abs/2610.10947) has no code release yet. Appendix F says "The dataset and code will be released upon publication."
- The paper prints its detector prompt in Table S4 (Appendix B.1): a JSON answer with `reasoning` and `prob_reward_hacking` (0-100), applied to the transcript with message contents joined by blank lines. This repo instead uses a YES/NO next-token probability (see `rhw/detector.py`), which needs one forward pass and no generation.
- The Hugging Face dataset `ktolnos/rh-bench` (https://huggingface.co/datasets/ktolnos/rh-bench, CC-BY-SA-4.0) has the same schema, taxonomy and split sizes as the paper (4,151 open-ended / 66,760 multiple-choice / 1,473 encrypted rows). The paper doesn't link to it, so treating it as the official release is an inference.
