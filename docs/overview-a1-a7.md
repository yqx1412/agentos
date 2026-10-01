# AgentOS, milestones A1-A7

A plain-language walkthrough of how AgentOS was built, one milestone at a time, with the
measured results. The detailed write-ups behind each number are in
[`benchmarks/results/`](../benchmarks/results/).

## The short version

AgentOS is a program that lets a small AI model, running free on your own graphics card, do
real tasks on your computer. Examples are "read these files, add up the numbers and write a
report," or "remember my server address for next time."

On its own, a language model only produces text. AgentOS gives it **tools** (read a file,
write a file, do arithmetic, run code), runs whatever the model asks for, and hands the
results back. Each milestone added one capability, and most of them were measured with a
test set to see whether it actually helped.

Think of it as a new employee, built up in seven steps:

| Step | Employee analogy | What AgentOS gained |
|---|---|---|
| A1 | Gets hands | It can use tools |
| A2 | Gets a toolbox it can extend | It can plug in external tools without code changes |
| A3 | Gets an exam | 38 graded tasks to measure progress |
| A4 | Learns to make a to-do list | It plans before it acts |
| A5 | Learns to check its own work | Each step is verified, and a rejected step is retried |
| A6 | Gets a notebook | It remembers across sessions |
| A7 | Gets rules and a supervisor | It can run code safely, with permission levels |

The models tested are all local and free: **qwen3:8b**, **qwen3:14b** and **llama3.1:8b**.

## A1: the basic loop

**Simple:** The model reads the task and says, for example, "I want to read notes.txt."
AgentOS reads the file and shows the model the contents. The model then says "now write 21
to result.txt," and AgentOS writes it. This repeats until the model says it's done.

**In detail:**

- **The loop:** the model's reply either asks for tool calls or gives a final answer. If it
  asks for tools, AgentOS runs them, adds the results to the conversation and asks the model
  again. The loop is capped at a maximum number of steps, so it can't run forever.
- **Tools are described precisely.** Each tool's inputs are defined in code, and AgentOS
  turns that definition into a description the model reads. That's how the model knows
  `write_file` needs a `path` and a `content`.
- **Mistakes don't crash it.** If the model calls a tool that doesn't exist, sends broken
  input or leaves out an argument, it gets back `ERROR: ...` and can fix its call. This was
  the A1 "done when" test.
- **Safety from the start:**
  - The file tools can't reach outside the working folder.
  - The calculator works out the expression itself instead of executing it as code, because
    executing it would let the model run any code it liked.
- **Every run is recorded** step by step in a log file, so failures can be investigated
  afterwards.

**Result:** the qwen models solved the demo task. llama3.1:8b failed by putting one tool
call inside another, and that became a recurring pattern.

## A2: plugging in external tools (MCP)

**Simple:** MCP (Model Context Protocol) is like USB for AI tools: thousands of programs
offer tools in one standard format. After A2, adding a new ability means adding three lines
to a config file, with no code changes.

**In detail:**

- At startup, AgentOS launches each configured tool server as a separate program, asks what
  tools it has, and adds them to the tool list. Their names get a prefix, such as
  `filesystem__read_text_file`, so they can't clash with the built-in tools.
- **A technical difficulty:** the MCP library is asynchronous (it does many things at
  once), while the agent loop is sequential. AgentOS runs the MCP side on a background
  thread and passes tool calls across to it.
- **Failures are handled differently by phase:**
  - A server that fails at startup stops the whole run, so a benchmark never quietly runs
    with tools missing.
  - A tool that fails during a run returns an error to the model, which can then correct
    its call.
- We wrote our own small tool server, **textkit** (word counts and word frequencies), to
  show both directions: using MCP tools and writing them.

## A3: the exam

**Simple:** Without measurement, "it works" is just a feeling. A3 is a set of 38 tasks, each
with an automatic grader, so any change can be scored.

**In detail:**

- The tasks fall into four groups: file operations, math, chaining several tools, and MCP
  tools. Each task starts in a fresh, empty folder.
- Graders only check results: the files left behind, the final answer and which tools were
  called. A task passes only if the model finished properly and every check passes.
- **Baseline scores:**

  | Model | Score |
  |---|---|
  | qwen3:8b | 31/38 (82%) |
  | qwen3:14b | 33/38 (87%) |
  | llama3.1:8b | 8/38 (21%) |

- **Common failures:**
  - llama writes tool calls out as text instead of actually making them.
  - The qwen models sometimes repeat the same broken call up to 11 times.
  - Models sometimes ignore what a tool returned.
  - Models pass a file name where a tool expected the file's text.
- **The exam also found bugs in the setup.** Ollama, the program that runs the models,
  cuts off long prompts silently by default, and a stuck model could run until the
  timeout. Both settings were fixed. One grader was too strict, and it was fixed before the
  scores were recorded.

## A4: planning

**Simple:** Before doing anything, the model writes a to-do list. Then each item is done on
its own, with a fresh, short conversation. A small model does better with one small job at
a time than with everything at once.

**In detail:**

- The model writes a plan: up to 8 steps, each listing which earlier steps it depends on.
  AgentOS checks the plan (no missing steps, no circular dependencies) and runs the steps
  in order.
- Each step sees the task, what earlier steps found, and the raw output of their tools.
- If a step fails, the model writes a new plan for the rest, at most twice.
- If the plan has only one step, the simple loop runs instead, so easy tasks barely cost
  more.
- **The first version made things worse** (qwen3:8b dropped to 26/38). The logs showed two
  causes:
  - It split trivial jobs into pointless steps.
  - It passed only summaries between steps. On a copy task, the model wrote "the file was
    read successfully" into the backup instead of the file's contents.

  Both were fixed.
- **Result (3 repeats):**

  | Model | Without planning | With planning |
  |---|---|---|
  | qwen3:8b | 83% | 89% |
  | qwen3:14b | 87% | 95% |
  | llama3.1:8b | 21% | 25% |

  Planning roughly doubles the cost in tokens (the units of text the model reads and
  writes) and in time. It was better on all three models, but with 38 tasks that isn't
  statistically conclusive yet.

## A5: checking the work

**Simple:** After each step, a checker asks "did this step really do its job?" If not, the
step is redone, and this time the model is told what was wrong.

**In detail:** the checks run cheapest first, in order:

1. Did the model write a tool call out as text instead of making it?
2. Did it pass a file name where the file's contents belonged?
3. Did it use a large number that appears nowhere: not in any file, any tool output or the
   task? That means it made the number up.
4. Finally, the model itself judges whether the step achieved its goal.

When a step is rejected, it either triggers a new plan or is retried with the reason
attached.

**This is Project 1's headline table** (passed runs out of 76: 38 tasks, 2 repeats):

| Model | Plain | + Planning | + Checking | + Retry |
|---|---|---|---|---|
| qwen3:8b | 62 | 63 | **69** | 67 |
| qwen3:14b | 64 | **66** | 64 | 65 |
| llama3.1:8b | 15 | 21 | 10 | **22** |

What it shows:

- **There's no single best setup.**
- **qwen3:8b** gains most from checking, because it makes the kind of mistakes the checks
  catch.
- **qwen3:14b** rarely makes those mistakes, so planning alone is enough.
- **llama** does worse with checking alone: it keeps getting rejected for the same thing.
  With retries, it often fixes the mistake once it sees the reason.

**Honest part:** the first version of the checker made results worse. It rejected correct
work for five different reasons, all found by reading the logs and all fixed. One gap
remains: a step that writes broken JSON can still pass the checks.

## A6: memory

**Simple:** A notebook that survives restarts. Tell it today "the deploy server is
build-07, port 8443." Tomorrow, in a brand-new session, ask it to write the server address
to a file, and it does.

**In detail:**

- **Long-term memory** is one database file holding two things:
  - facts the model chose to save with a `remember` tool
  - a short record of every past task and how it went
- Looking things up is keyword search, like a search engine for your notes.
- **Short-term memory:** when a conversation gets too long for the model, older tool
  outputs are cut down to their beginning and end. The first version asked the model to
  write a summary instead. In a test, qwen3:8b "summarized" file reads it had never been
  shown, so that approach was dropped.
- **Two ways to use memory:**
  - **tools:** the model decides itself when to look things up.
  - **auto:** AgentOS searches the notebook for the task and hands the matching notes to
    the model up front.
- **Result (11 tasks spanning two sessions, 2 repeats each):**

  | Model | No memory | Tools only | Auto |
  |---|---|---|---|
  | qwen3:8b | 2/22 | 6/22 | **20/22** |
  | qwen3:14b | 2/22 | 4/22 | **20/22** |

**The key finding: models write notes readily but rarely read them.** With tools only, they
saved facts in 24 of 26 sessions, but in the next session they almost never looked
anything up. They guessed instead, writing things like `host:port` or `<codename>`. Memory
only works when the system hands it over.

The one remaining failure: the note said "manager" and the question said "boss." Keyword
search can't connect the two, and that's what Project 2's meaning-based search is meant to
fix.

## A7: safety rules and a supervisor

**Simple:** The agent can now run Python code and commands, which is powerful and risky. So:

- every tool has a risk level
- risky actions need a human "yes"
- the code runs in a locked room it can't easily get out of

**In detail:**

- **Three permission levels:**
  - **read:** looking only. Runs freely.
  - **write:** changes files or memory. Runs freely by default.
  - **dangerous:** runs code or commands. Shows you the exact call and asks y/N, defaulting
    to no. If no human is there to answer, the answer is no.
- **The locked room, layer by layer:**
  1. **Operating-system limits:** a time limit, a memory limit and a cap on how many
     programs it can start. When the time is up, everything it started is stopped,
     including programs those started.
  2. **For Python code:** a guard installed before the agent's code runs. It blocks
     starting programs, network access, low-level system access, and any file outside the
     working folder.
  3. **For commands:**
     - Only allowed programs run, by default just `git`, and never through a shell that
       could reinterpret the input.
     - git gets only safe local subcommands, and its configuration tricks are blocked.
     - Any argument that looks like a path must stay inside the working folder.
  4. **A clean environment:** the agent's code can't see your passwords or API keys stored
     in environment variables.
  5. **The `.git` folder is off limits to every tool.** Editing it is a known way to turn
     "may run git" into "may run anything."
- **The A7 "done when":** a 60-test escape suite tries dozens of tricks: reading files above
  the working folder, deleting outside it, starting hidden programs, memory bombs, endless
  loops, flooding the output, leaking secrets, and git configuration tricks. All of them
  are blocked on both Windows and Linux.
- **Honest part:** this guards against a careless or confused model. It is not a real
  container. The human approval step is the true safety gate.

## The big picture

```text
Task ──► [memory: matching notes]           (A6)
          ──► [planner: steps]               (A4)
                ──► for each step: loop      (A1)
                      ──► tools: built-in, MCP, sandbox   (A1, A2, A7)
                      ──► permission check   (A7)
                ──► checker: accept / retry / replan      (A5)
          ──► answer + saved to memory       (A6)
Everything measured by the exam              (A3)
```

**What it all shows:** a free model on a home graphics card can do real multi-step work,
scoring about 85-95% on these tasks. But the system around the model matters as much as the
model itself:

- **Which support helps depends on the model:** checking for the 8B qwen, planning for the
  14B.
- **Some things the system must do itself:** the models wouldn't open their own notebook,
  so memory only worked once AgentOS handed the notes over.
- **Measuring caught my own mistakes:** three of the first versions made results worse (the
  planner, the checker and the summarizer), and the logs showed why.
