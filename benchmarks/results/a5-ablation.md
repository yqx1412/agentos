# A5: verification + retries, and the full ablation

Run `20261001-004748`: 38 tasks x 2 repeats x 4 agent kinds x 3 models (912 runs),
temperature 0, thinking off, `num_ctx` 8192, `num_predict` 2048, RTX 5060 Ti 16 GB.
Each kind adds one mechanism to the previous one:

| Kind | Adds |
|---|---|
| `plain` | the A1 loop |
| `planner` | A4 plan -> task graph -> per-step loop, replan on failure |
| `planner-verify` | every finished step is checked; a rejected step is treated as failed (-> replan) |
| `planner-verify-retry` | a rejected step is first re-run up to 2 times with the rejection reason |

## Headline: the ablation table

Passed runs out of 76 (38 tasks x 2 repeats):

| Model | plain | + planner | + verify | + retry | Avg tokens (plain -> +retry) | Avg time |
|---|---|---|---|---|---|---|
| qwen3:8b | 62 (82%) | 63 (83%) | **69 (91%)** | 67 (88%) | 2,620 -> 5,563 | 1.8 s -> 4.9 s |
| qwen3:14b | 64 (84%) | **66 (87%)** | 64 (84%) | 65 (86%) | 2,228 -> 6,335 | 2.8 s -> 11.1 s |
| llama3.1:8b | 15 (20%) | 21 (28%) | 10 (13%) | **22 (29%)** | 1,504 -> 11,002 | 2.7 s -> 14.7 s |

The two repeats agree to within 2 runs for every row, so the sample that matters is the
38 tasks. Tasks whose pass count changed between consecutive kinds (out of 2):

| Model | planner -> + verify | + verify -> + retry |
|---|---|---|
| qwen3:8b | +3: fo-append 0->2, mcp-find-todo 0->2, mcp-top-word 0->2; -0 | +0; -2: fo-append 2->1, mcp-find-todo 2->1 |
| qwen3:14b | +1: fo-json-array; -3: fo-append, fo-subdir, mcp-fs-count | +1: mcp-fs-count; -0 |
| llama3.1:8b | +0; -7 (p = 0.02) | +9; -1 (p = 0.02) |

## What each part does, per model

- **qwen3:8b: verification is the part that helps.** plain -> +verify gains 4 tasks and
  loses none. The `filename_as_content` check catches textkit being called with a file
  name (`word_frequency(text="essay.txt")`, the failure that A3 and A4 could not fix), and
  the retry or replan then passes the text. Only 1 of qwen3:8b's 22 rejections came in a
  run that failed anyway, so the checks almost never flagged work that was fine. Retries add
  nothing on top: with this few rejections, a replan already recovers.
- **qwen3:14b: verification costs without helping.** It already made few mistakes of the
  kinds the deterministic checks catch; only the reflection call fired, and 22 of its 35
  rejections came in runs that failed anyway. Net -2 runs for verify and -1 for retry,
  within noise, at +30-40% tokens and +55-65% time over the planner. The planner alone is
  the best configuration.
- **llama3.1:8b: verification alone hurts, verification + retries helps.** 167 of its 182
  rejections without retries are `text_tool_call`: it describes a call as JSON instead of
  making it. Without retries, a rejected step goes to the replanner, whose new steps get
  rejected the same way, so the runs end in `plan_failed`. This includes tasks that plain
  "passed" by doing nothing (`fo-escape`, `fo-missing-file`). With retries, the rejection
  text ("call the tool through the tool interface") is fed back and llama often makes the
  real call on the next try: +9 tasks. Net: 29%, the best llama result so far, at 7x
  plain's tokens.

## The verifier (`src/agentos/verifier.py`)

Deterministic checks run first; they cost nothing and cannot be talked out of a verdict:

| Check | Catches | Fired (all models, both verify kinds) |
|---|---|---|
| `text_tool_call` | a tool call written as JSON text (`"name": "write_file"`) in a step that made no real call | 515, all llama |
| `filename_as_content` | a bare file name passed as `text`/`data`/`body` | 7, all qwen3:8b |
| `ungrounded` | a tool argument number >= 100 that is in no tool output, the task or an earlier step | 0 |
| `reflection` | model check: goal achieved, values follow from the outputs | 147 (15 qwen3:8b, 35 qwen3:14b, 97 llama) |

`ungrounded` did not fire in this run: no step used a number it had not read. In the A4
demo qwen3:8b made up CSV totals; in this run's demo it read them. The unit tests
reproduce the A4 case.

## Fixes found while building it

The first full run (`20260930-230648`) gave verify 36 -> 31 on qwen3:14b and 9 -> 3 on
llama. The traces showed three verifier problems and two planner problems:

1. **Reflection saw only the current step.** "Pick the largest region" with no tool call
   was rejected as "no tool calls were made". Fix: reflection now gets the earlier steps'
   results, and the prompt says reasoning-only steps and display rounding are fine.
2. **The text-call check matched prose.** `calculator(9*60+40)` in an explanation counted
   as a text call. Fix: only the JSON shape counts.
3. **Retry evidence vouched for itself.** A rejected attempt's own arguments were added to
   the next attempt's evidence, so a made-up number passed the second time. Fix: evidence
   is only what tools returned.
4. **Impossible steps.** The planner planned "create the backup directory", which no tool
   does, and verification kept rejecting it. Fix: `write_file` says it creates parent
   directories, and the planner prompt says to plan only steps a tool can do.
5. **Replans rejected for stale dependencies.** A revised plan that still said
   `depends_on: [1]` after step 1 failed was invalid, and that used up the replan budget.
   Fix: in a revision, dependencies on steps that never finished are dropped.

A fix-only second run (`20261001-000508`, verify kinds only, 1 repeat) gave qwen3:8b 36/38
and qwen3:14b 37/38 with verify. The final 2-repeat run above also includes fixes 3-5,
which change the planner, so all four kinds were re-measured together.

## Known gaps

- **Malformed file contents pass.** In the 6-step demo, qwen3:8b + verify wrote
  `totals.json` with a real `write_file` call whose content was escaped
  (`{\"north\": 3530, ...}`), which is not valid JSON, and wrote `report.txt` as JSON when
  the task asked for two `key: value` lines. Reflection passed both steps. A cheap
  deterministic check, "a written `.json` file must parse", would catch the first case (and
  A3's `ch-csv-to-json` failure). It is not in this run.
- **Reflection is a weak judge.** It is the same model grading its own work, and on
  qwen3:14b most of its rejections came in runs that failed anyway.
- **The planner rows moved.** qwen3:14b + planner is 33/38 per repeat here, compared with
  36/38 in the A4 run. The A5 planner changes (fixes 4 and 5) are the only code
  difference. I have not isolated which one caused it.

## Recommendation

| Model | Use | Why |
|---|---|---|
| qwen3:8b | `planner-verify` | +7 runs over plain, no task lost, ~2.2x tokens |
| qwen3:14b | `planner` | verification adds cost and no accuracy |
| llama3.1:8b | `planner-verify-retry` if at all | 29% is still unusable; its problem is tool calling itself |

The plain loop stays the CLI default; the kinds are opt-in with `--agent`.
