# rh-window-monitor

Tests whether the multi-turn tool-use gap in RH-Detect is a transcript-length effect, using windowed LLM detectors on TRACE and MALT.

Work in progress.

## Related code

Status checked on 2026-11-21:

- RH-Detect (https://arxiv.org/abs/2610.10947) has no code release yet. Appendix F says "The dataset and code will be released upon publication."
- The paper prints its detector prompt in Table S4 (Appendix B.1): a JSON answer with `reasoning` and `prob_reward_hacking` (0-100), applied to the transcript with message contents joined by blank lines. This repo instead uses a YES/NO next-token probability (see `rhw/detector.py`), which needs one forward pass and no generation.
- The Hugging Face dataset `ktolnos/rh-bench` (https://huggingface.co/datasets/ktolnos/rh-bench, CC-BY-SA-4.0) has the same schema, taxonomy and split sizes as the paper (4,151 open-ended / 66,760 multiple-choice / 1,473 encrypted rows). The paper doesn't link to it, so treating it as the official release is an inference.
