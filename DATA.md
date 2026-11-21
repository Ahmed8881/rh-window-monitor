# Data sources and label mapping

All counts below come from `python scripts/build_dataset.py` (see `results/dataset_stats.csv`).

## Where the units come from

The RH-Detect paper (arXiv 2610.10947) has no code release, but its unit table appears to be
public as `ktolnos/rh-bench` (https://huggingface.co/datasets/ktolnos/rh-bench, CC-BY-SA-4.0, not
gated). It has the same schema and split sizes as the paper; the paper does not link it, so this
is an inference. As the project plan asks, MALT and control units are taken from it, not rebuilt.

| Source in this repo | Raw file | Positive (label 1) | Negative (label 0) |
|---|---|---|---|
| `trace` | `PatronusAI/trace-dataset` `data/train-00000-of-00001.parquet` (CC-BY-SA-4.0, gated auto-approve), https://arxiv.org/abs/2601.20103 | `label != "0"` (one or more hack codes such as `1.1.2`; 268 rows) | `label == "0"` (benign; 249 rows) |
| `malt` | rh-bench rows with `source_dataset == "metr_malt"` (original: `metr-evals/malt-public`, MIT, gated) | `label == "hacking"` and `construction == "unprompted"` (62 rows: test_modification 24, filesystem_runtime 23, hardcoded_lookup 15) | every `label == "clean"` MALT row (286) |
| `malt_prompted` | same | `label == "hacking"` and `construction == "prompted"` (224 rows, subcategory `strategic_deception`, e.g. MALT `sabotage`) | the same 286 clean MALT rows |
| `control` | rh-bench rows with `source_dataset == "school_of_reward_hacks"` (original: `longtermrisk/school-of-reward-hacks`, CC-BY-4.0) | `response_hacking` (1,073) | `response_clean` (973) |

## Notes

- **TRACE** is loaded from the original Patronus parquet, not from rh-bench, because the original keeps
  `tool_calls` / `tool_results` and real follow-up user turns apart. rh-bench has the same 517
  trajectories (268 hacked / 249 clean). Tool calls are appended to the assistant turn as
  `CALL <name> <json>`; each tool result is its own turn with `is_tool = True`.
- **MALT** in rh-bench has the task as a system message and tool outputs as `user` messages. MALT agents run
  without a human, so every `user` message after the prompt becomes a tool turn (`is_tool = True`).
  Assistant tool calls appear as `function_call: {...}` text.
- **MALT negatives are shared** between `malt` and `malt_prompted`: the same clean runs are scored for
  both. The cache keys texts by sha1, so each clean run is scored once.
- **MALT labels are broader than reward hacking.** The prompted positives are mostly sabotage /
  sandbagging (`strategic_deception`), which RH-Detect counts as hacking-like behaviour. That is
  why they are kept as a separate source.
- **Control** is single-turn: a user prompt that names a gameable metric and one assistant reply.
- The task is the first `n_task` turns (TRACE / control: the first user turn; MALT: the system prompt),
  and every windowed view includes it.
- **Negative sampling** (`rhw.data.sample_split`): per source, keep all positives and draw
  `min(#neg, 4 x #pos, 400)` negatives with seeds 0, 1, 2.
