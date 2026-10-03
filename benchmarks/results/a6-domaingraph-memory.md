# A6 memory on DomainGraph: SQLite vs graph-backed long-term memory

DomainGraph's D5 milestone is done when AgentOS's long-term memory can run on DomainGraph
instead of SQLite. It can: `--memory-backend domaingraph` swaps the store, and the A6 memory
benchmark passes **22/22 on both qwen models with automatic injection**. With SQLite it
passes 20/22. The two failures SQLite can't fix are the synonym task, where the question asks
for a "boss" and the stored fact names a "manager". Embedding search finds it.

```powershell
# needs Neo4j (docker compose up -d in ../domaingraph) and Ollama with bge-m3
uv run agentos bench --tasks benchmarks/memory --models qwen3:8b,qwen3:14b `
    --memory tools,auto --memory-backend domaingraph --repeats 2
uv run agentos run --memory auto --memory-backend domaingraph "..."
uv run agentos memory --memory-backend domaingraph list
```

## How it works

- **The same interface on both stores.** `agentos.memory_domaingraph.DomainGraphMemory`
  implements the `Memory` protocol that `MemoryStore` (SQLite) also implements. The `remember`,
  `recall` and `forget` tools, `MemoryAgent`'s injection and episode recording, and the
  `agentos memory` command all work unchanged on either one.
- **Every call goes to DomainGraph's MCP server.** The backend calls the `add_fact`,
  `recall_facts`, `forget_fact` and `list_facts` tools. Memories are `(:Fact:Memory)` nodes in
  Neo4j with a bge-m3 embedding, linked by `ABOUT` edges to the lecture concepts they name.
- **The server comes from the `[mcp_servers.domaingraph]` config entry.** The entry is in
  `agentos.toml` and disabled by default, so a plain `agentos run` doesn't need Neo4j. The
  backend uses it anyway: `enabled` only decides whether the agent also sees DomainGraph's
  lecture tools.
- **Each task gets its own memory.** The benchmark gives every task a fresh scope
  (`bench-<task>-<random>`) and deletes it afterwards. That's the analogue of the temporary
  SQLite file each task got in A6.
- **Recall is by meaning.** A memory is returned when its cosine similarity to the query is
  at least 0.5, and the newest come first, as with SQLite.

## Results

The memory benchmark: 11 multi-session tasks, 2 repeats each, so 22 runs per cell. Both
backends ran on the same commit, one after the other, against the same Ollama models.

| Model | Memory mode | SQLite (FTS5 keywords) | DomainGraph (bge-m3) |
|---|---|---|---|
| qwen3:8b | tools only | 8/22 | 10/22 |
| qwen3:8b | **tools + auto-injection** | 20/22 | **22/22** |
| qwen3:14b | tools only | 4/22 | 4/22 |
| qwen3:14b | **tools + auto-injection** | 20/22 | **22/22** |

- **Auto-injection, SQLite:** the only failure is `mem-synonym`, on both models and both repeats.
  The setup session stores "My manager is Priya Raman", and the question asks "Who is my
  boss?". The keywords don't overlap, so nothing is injected. qwen3:8b then gives up
  without writing the file, and qwen3:14b writes "John Doe".
- **Auto-injection, DomainGraph:** `mem-synonym` passes. "boss" and "manager" are close in
  embedding space, so the memory is injected. That is the failure the A6 write-up said
  DomainGraph's embedding search should fix.
- **Tools only is still weak on both backends.** As in A6, the models store facts readily
  but rarely call `recall` in the next session, and guess instead (`host:port`, `<day>`,
  `Project Horizon`). The backend can't help when the model doesn't ask. qwen3:8b's +2 is
  `mem-synonym`, in both repeats: when it does call `recall` with "boss", DomainGraph finds
  the "manager" fact, and SQLite doesn't.
- **Cost:** qwen3:8b auto-injection averaged 3,760 tokens per run vs 3,593 with SQLite.
  The extra comes from memories injected just above the 0.5 cutoff. Time per run is about the
  same (2.1 s vs 1.8 s, which includes the embedding call).

## Limits

- **Each scope holds only that task's few memories**, so the benchmark doesn't test
  retrieval among many unrelated memories. In a long-lived store, a 0.5 cutoff will also
  inject loosely related memories. Measured on this benchmark's texts, the best *unrelated*
  memory scored up to 0.69 against a question, while the related ones scored 0.55-0.85. A
  bigger store would need a reranker or a higher cutoff.
- **The 0.5 cutoff was set before either run.** Afterwards I checked the cosine
  distribution (above) but didn't tune on it.
- **DomainGraph needs Neo4j and Ollama running**, where SQLite needs nothing. That's why
  SQLite stays the default.
