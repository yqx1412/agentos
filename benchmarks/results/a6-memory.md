# A6: memory

**Question:** can a local model use what it learned in an earlier session, and does it need
the runtime to hand it memories, or will it look them up itself?

## Setup

- 11 multi-session tasks in `benchmarks/memory/memory.yaml`. Each runs 1-2 *setup* sessions
  (fresh agent, fresh conversation, fresh MCP servers) and then the scored session. All
  sessions of a task share one SQLite memory store; files listed under `setup_files` are
  deleted before the scored session, so the value only survives in memory.
- Memory modes:
  - `none`: no memory. Control: only `mem-unrelated` should pass.
  - `tools`: `remember` / `recall` / `forget` tools only; the model decides when to use them.
  - `auto`: the same tools, plus the runtime searches memory with the task text (SQLite
    FTS5, BM25) and puts the matching facts and past task outcomes in front of the task.
- Plain A1 agent, temperature 0, 2 repeats (22 runs per cell), git `888914f`.
- Tokens include every session of a task, so the cost of writing memories is counted.

```powershell
uv run agentos bench --tasks benchmarks/memory --models qwen3:8b,qwen3:14b,llama3.1:8b `
  --memory none,tools,auto --repeats 2
```

## Results

| Model | none | tools | auto | Tokens (none -> auto) |
|---|---|---|---|---|
| qwen3:8b | 2/22 (9%) | 6/22 (27%) | **20/22 (91%)** | 2.0k -> 3.6k |
| qwen3:14b | 2/22 (9%) | 4/22 (18%) | **20/22 (91%)** | 1.8k -> 3.9k |
| llama3.1:8b | 0/22 | 0/22 | 0/22 | 2.5k -> 2.5k |

The two repeats agreed on every task for every model.

Per task (qwen3:8b and qwen3:14b had identical results in `auto`):

| Task | Tests | none | tools (8b / 14b) | auto |
|---|---|---|---|---|
| mem-told | fact stated by the user | fail | fail / fail | pass |
| mem-port | value from a file that is gone in session 2 | fail | fail / fail | pass |
| mem-deploy | two values combined (host:port) | fail | fail / fail | pass |
| mem-update | conflicting facts, the newer one wins | fail | fail / fail | pass |
| mem-distractor | one fact among three similar ones | fail | pass / fail | pass |
| mem-episode | nothing stored; only the earlier task's outcome has it | fail | pass / pass | pass |
| mem-synonym | "manager" stored, "boss" asked | fail | fail / fail | **fail** |
| mem-two-facts | two facts from two sessions | fail | fail / fail | pass |
| mem-compute | arithmetic on remembered numbers | fail | fail / fail | pass |
| mem-path | remembered file path, file still there | fail | fail / fail | pass |
| mem-unrelated | memory irrelevant (interference control) | pass | pass / pass | pass |

## What the traces show

- **Writing is easy, reading is not.** In `tools` mode both qwen models called `remember`
  in 24 of 26 setup sessions, but in the scored session qwen3:8b called `recall` in only
  6 of 22 runs and qwen3:14b in 2 of 22. The rest wrote a placeholder or a guess instead:
  `host:port`, `<codename>`, `API_PORT`. The
  `none` runs fail the same way: the models invent a value rather than saying they don't
  know it. Automatic injection is what makes memory work; the tools alone barely help.
  I didn't test a system-prompt hint like "check memory before answering", which might
  close some of that gap more cheaply.
- **The one `auto` failure is keyword search.** `mem-synonym` stores "my manager is Priya
  Raman" and asks "who is my boss?". FTS finds nothing, so nothing is injected; the model
  then called `recall("boss")`, got nothing, and gave up. That is the case DomainGraph's
  embedding search (D3/D4) has to fix.
- **Conflicting facts are handled by recency.** In `mem-update` both "Teahouse" and the
  later "renamed to Coffeehouse" are injected, newest first, with a note that the newer
  entry wins. Both qwen models picked the new name.
- **llama3.1:8b doesn't get there.** It stores facts and receives them in `auto`, then
  fails the way it failed A3-A5: tool calls written as text, or arguments of one tool sent
  to another (`recall(path=..., content=...)`). It also fails `mem-unrelated` without any
  memory, so this is not a memory problem.

## Does memory hurt tasks that don't need it?

The 38 regular tasks, with an empty store per task (so `auto` only adds the three memory
tools to the tool list), one run each:

| Model | none | auto | Tokens |
|---|---|---|---|
| qwen3:8b | 31/38 | 32/38 | 2.5k -> 3.0k |
| qwen3:14b | 32/38 | 33/38 | 2.3k -> 3.0k |
| llama3.1:8b | 7/38 | 7/38 | 1.4k -> 2.1k |

Six tasks changed result in either direction, one or two per model, which is within run
noise. The cost is about 20-30% more prompt tokens for the extra tool schemas. llama hit
`max_steps` 4 times with memory on versus 0 without.

## Short-term memory: compaction

When a request's prompt exceeds `--context-budget` (default 75% of `--num-ctx`), the plain
loop replaces the middle of its conversation with one message: the old tool calls plus each
old tool output clipped to its first and last 300 characters. The result object still
keeps every message, so benchmark checks are unaffected.

The first version asked the model to write that summary. In a live check (5 files of
filler, each ending with `KEYn = value`, sum them, `--context-budget 1500`), qwen3:8b was
given the reads of `part1.txt` and `part2.txt` and returned a "summary" of reads of
`part3-5.txt`, with values it had guessed from the file-name pattern. The deterministic
version ran the same task with 5 compactions and wrote the correct sum, 1665.
`compact_messages(..., llm=...)` still supports a model summary, but the loop doesn't use it.

The benchmark never reaches the default budget (6144 tokens), so this is checked by unit
tests and that one live run, not by the benchmark.
