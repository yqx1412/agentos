# A4: planner vs. plain loop

## Repeated run: 38 tasks x 3 repeats (headline)

Run `20260930-215230` on commit `0c8cbc8`, both agents in one pass, same settings as below.

| Model | Plain | Planner | Per repeat (plain / planner) | Avg tokens | Avg time |
|---|---|---|---|---|---|
| qwen3:8b | 95/114 (83%) | 101/114 (89%) | 32, 31, 32 / 33, 34, 34 | 2,807 -> 4,919 | 2.0 s -> 4.0 s |
| qwen3:14b | 99/114 (87%) | 108/114 (95%) | 33, 33, 33 / 36, 36, 36 | 2,220 -> 4,488 | 3.0 s -> 6.9 s |
| llama3.1:8b | 24/114 (21%) | 29/114 (25%) | 8, 8, 8 / 11, 9, 9 | 1,308 -> 3,758 | 2.5 s -> 6.2 s |

At temperature 0 the repeats barely differ: almost every task passes 0/3 or 3/3 times.
So repeats confirm the single-run numbers were not luck of one run, but the real sample is
the 38 tasks. Tasks where the two agents differ (pass count out of 3):

| Model | Planner better | Planner worse |
|---|---|---|
| qwen3:8b | ch-csv-to-json 0->3, fo-overwrite 0->3, ma-sort 0->3, mcp-lecture-summary 2->3 | ma-max-csv 3->2, mcp-top-word 3->0 |
| qwen3:14b | ch-combine 0->3, fo-overwrite 0->3, ma-sqrt-cube 0->3, mcp-find-todo 0->3 | mcp-find-similar 3->0 |
| llama3.1:8b | fo-append 0->3, fo-uppercase 0->3, ma-time 0->3, ch-combine 0->1, fo-wordcount 0->1 | fo-json-array 3->0, ma-sqrt-cube 3->0 |

The planner wins more tasks than it loses on every model (13 vs 5 in total), and
`fo-overwrite`, where both qwen models repeated a failing calculator call in the plain
loop, flips 0->3 for both. But per model a sign test over the differing tasks gives
p = 0.69 / 0.38 / 0.45, and pooled p ~ 0.1: consistent in direction, not yet
statistically solid with 38 tasks. The cost is solid: ~1.8-2.9x tokens, ~2-2.5x time.

## Single run (first measurement)

38 tasks, 1 run each, temperature 0, thinking off, `num_ctx` 8192, `num_predict` 2048,
RTX 5060 Ti 16 GB. Planner settings: `step_max_steps=8`, `max_replans=2`, total budget
`2 * max_steps` model turns.

- **plain**: run `20260930-210234` (`--agents plain,planner`, first half). The plain loop
  code did not change afterwards.
- **planner**: run `20260930-212301` (`--agents planner`), after the two fixes below.

## Summary

| Model | Plain | Planner | Tool errors (plain -> planner) | Avg tokens | Avg time |
|---|---|---|---|---|---|
| qwen3:8b | 32/38 (84%) | 34/38 (89%) | 33 -> 30 | 2,667 -> 4,741 | 2.2 s -> 3.9 s |
| qwen3:14b | 33/38 (87%) | 35/38 (92%) | 18 -> 3 | 2,211 -> 4,599 | 3.0 s -> 7.1 s |
| llama3.1:8b | 8/38 (21%) | 9/38 (24%) | 27 -> 37 | 1,227 -> 3,768 | 2.5 s -> 6.2 s |

| Category | Tasks | qwen3:8b plain / planner | qwen3:14b plain / planner | llama3.1:8b plain / planner |
|---|---|---|---|---|
| chaining | 9 | 8 / 9 | 8 / 9 | 0 / 0 |
| file_ops | 11 | 10 / 11 | 10 / 10 | 5 / 6 |
| math | 9 | 8 / 8 | 8 / 9 | 2 / 2 |
| mcp | 6 | 3 / 3 | 4 / 4 | 0 / 0 |
| recovery | 2 | 2 / 2 | 2 / 2 | 0 / 0 |
| safety | 1 | 1 / 1 | 1 / 1 | 1 / 1 |

A 1-2 task gain on a single run is within run-to-run noise; `--repeats 3` is needed before
calling it real. The cost is not noise: the planner roughly doubles tokens and time.

## First attempt (before the fixes): the planner made things worse

| Model | Plain | Planner v1 |
|---|---|---|
| qwen3:8b | 32/38 | 26/38 |
| qwen3:14b | 33/38 | 32/38 |
| llama3.1:8b | 8/38 | 6/38 |

Two causes, both visible in the traces:

1. **Over-planning.** qwen3:8b made a multi-step plan for 30 of 38 tasks, splitting even
   "copy a file" into "read" and "write". Fix: the planner prompt says one output = one
   step and anything under ~4 tool calls = one step (then the plain loop runs directly).
   Afterwards qwen3:8b ran 21 tasks direct.
2. **Summaries lose data.** Steps only passed a model-written summary forward. On `fo-copy`
   step 1 reported "The contents of source.txt have been read successfully", and step 2
   wrote that sentence into the backup. Fix: later steps also get the finished steps' raw
   tool outputs (capped at 800 chars per call, 2,000 per step).

## What the planner does not fix

- **llama3.1:8b** still writes tool calls as text instead of making them; inside a step it
  now "explains" the call (`{"name": "write_file", ...}` as prose), so most planned runs
  leave no output file. Planning is not the bottleneck for this model.
- **Unread inputs.** In the 6-step demo qwen3:8b's step 2 summed made-up numbers without
  opening the CSVs. Nothing checks that a step used the data it was supposed to. A5
  (verification) targets this.

## Full report of the planner run

| Model | Direct (pass) | Planned (pass) | Fallback (pass) | Avg plan steps | Replans |
|---|---|---|---|---|---|
| qwen3:8b | 21 (19) | 17 (15) | 0 (0) | 3.6 | 2 |
| qwen3:14b | 7 (7) | 31 (28) | 0 (0) | 3.3 | 3 |
| llama3.1:8b | 6 (5) | 32 (4) | 0 (0) | 3.7 | 3 |

Remaining qwen failures with the planner:

- `qwen3:8b` **ma-time**: answered 12:55, expected 12:35
- `qwen3:8b` **mcp-top-word**: wrote `essay` (passed the file name to textkit instead of its text)
- `qwen3:8b` **mcp-find-todo**: todos.txt does not exist
- `qwen3:8b` **mcp-word-total**: plan_failed after 2 replans
- `qwen3:14b` **fo-json-array**: list.json does not exist
- `qwen3:14b` **mcp-word-total**: never called textkit__text_stats
- `qwen3:14b` **mcp-lecture-summary**: top words `the` / `a` (textkit got file names, not text)
