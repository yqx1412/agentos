# A4: planner vs. plain loop

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
