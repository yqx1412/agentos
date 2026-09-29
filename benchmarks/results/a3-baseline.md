# AgentOS benchmark results

- started: 2026-09-29T22:59:00, git: `9b15eeb` + uncommitted A3 branch, tasks: 38, repeats: 1
- settings: `{'num_ctx': 8192, 'num_predict': 2048, 'temperature': 0.0, 'think': False, 'timeout': 180.0, 'agent': 'plain loop (A1)'}`

## Summary

| Model | Passed | Success | Avg steps | Avg tool calls | Tool errors | Avg tokens | Avg time | Backend errors | Hit max steps |
|---|---|---|---|---|---|---|---|---|---|
| qwen3:8b | 31/38 | 82% | 3.8 | 3.2 | 35 | 2,656 | 1.9 s | 0 | 1 |
| qwen3:14b | 33/38 | 87% | 3.5 | 2.8 | 18 | 2,212 | 2.9 s | 0 | 1 |
| llama3.1:8b | 8/38 | 21% | 2.6 | 5.2 | 27 | 1,333 | 2.9 s | 0 | 0 |

## By category

| Category | Tasks | qwen3:8b | qwen3:14b | llama3.1:8b |
|---|---|---|---|---|
| chaining | 9 | 8/9 | 8/9 | 0/9 |
| file_ops | 11 | 10/11 | 10/11 | 5/11 |
| safety | 1 | 1/1 | 1/1 | 1/1 |
| math | 9 | 7/9 | 8/9 | 2/9 |
| recovery | 2 | 2/2 | 2/2 | 0/2 |
| mcp | 6 | 3/6 | 4/6 | 0/6 |

## Failures

- `qwen3:8b` **ch-csv-to-json**: inventory.json is not valid JSON: Expecting property name enclosed in double quotes
- `qwen3:8b` **fo-overwrite**: version.txt: expected '1.4.3', got '1.4.2'
- `qwen3:8b` **ma-sort**: sorted.txt: expected ['3', '7', '19', '23', '42', '88'], got ['3', '3', '7', '19', '23', '88']
- `qwen3:8b` **ma-time**: answer lacks all of ['12:35']; got 'The train arrives at 12:55. \\n\\nFinal answer: 12:55'
- `qwen3:8b` **mcp-find-todo**: todos.txt does not exist
- `qwen3:8b` **mcp-word-total**: stopped: max_steps
- `qwen3:8b` **mcp-lecture-summary**: summary.txt !~ /^\s*(lectures[/\\])?raft(\.txt)?\s*:\s*leader\s*$/; got 'paxos.txt: text\\nraft.txt: text'; summary.txt !~ /^\s*(lectures[/\\])?paxos(\.txt)?\s*:\s*acceptors\s*$/; got 'paxos.txt: text\\nraft.txt: text'
- `qwen3:14b` **ch-combine**: greeting.txt: expected 'Hello World', got 'first_word second_word'
- `qwen3:14b` **fo-overwrite**: stopped: max_steps
- `qwen3:14b` **ma-sqrt-cube**: answer lacks 69; got '72.0'
- `qwen3:14b` **mcp-find-todo**: todos.txt !~ /^\s*3\s*,\s*7\s*$/; got ''
- `qwen3:14b` **mcp-word-total**: never called textkit__text_stats; called ['read_file', 'write_file']
- `llama3.1:8b` **ch-pointer**: found.txt: expected 'otter', got 'calculator(expression = "read_file(path = "start.txt")")'
- `llama3.1:8b` **ch-combine**: greeting.txt: expected 'Hello World', got 'read_file(path = "first.txt") + " " + read_file(path = "second.txt")'
- `llama3.1:8b` **ch-json-config**: samples.txt has no number; got 'result'
- `llama3.1:8b` **ch-quarters**: year_total.txt: expected 5000, got 1
- `llama3.1:8b` **ch-csv-to-json**: inventory.json is not valid JSON: Expecting value
- `llama3.1:8b` **ch-log-report**: report.txt lacks ['errors: 3', 'warnings: 2']; got 'errors: N\\nwarnings: M'
- `llama3.1:8b` **ch-contact-json**: contact.json does not exist
- `llama3.1:8b` **ch-template**: letter.txt: expected 'Dear Sam, your order A-1043 ships on Oct 3.', got 'read_file(path = "template.txt")'
- `llama3.1:8b` **ch-price-update**: new_prices.txt !~ /^\s*coffee:\s*\$?3\.30\s*$/; got 'item: 11.00\\nitem: 11.00\\nitem: 11.00\\nitem: 11.00'; new_prices.txt !~ /^\s*bagel:\s*\$?2\.75\s*$/; got 'item: 11.00\\nitem: 11.00\\nitem: 11.00\\nitem: 11.00'; new_prices.txt !~ /^\s*juice:\s*\$?4\.62\s*$/; got 'item: 11.00\\nitem: 11.00\\nitem: 11.00\\nitem: 11.00'
- `llama3.1:8b` **fo-wordcount**: result.txt has no number; got 'calculator(expression = "len(read_file(path = "notes.txt"))")'
- `llama3.1:8b` **fo-uppercase**: shout_upper.txt: expected 'MAKE ME LOUD', got 'TO BE ADDED'
- `llama3.1:8b` **fo-append**: todo.txt: expected ['buy milk', 'walk dog', 'call mom'], got ['call mom']
- `llama3.1:8b` **fo-linecount**: lines.txt does not exist
- `llama3.1:8b` **fo-overwrite**: version.txt: expected '1.4.3', got 'join(split(read_file(\'version.txt\'), "."), [split(read_file(\'version.txt\'), "...'
- `llama3.1:8b` **fo-reverse-lines**: reversed.txt: expected ['four', 'three', 'two', 'one'], got ['cat list.txt | tac']
- `llama3.1:8b` **ma-compound**: amount.txt has no number; got 'The final amount is: '
- `llama3.1:8b` **ma-sum-file**: total.txt has no number; got 'calculator(expression = "read_file(path = "numbers.txt")")'
- `llama3.1:8b` **ma-average-csv**: average.txt has no number; got 'calculator(expression='
- `llama3.1:8b` **ma-energy**: kwh.txt has no number; got 'calculator(expression='
- `llama3.1:8b` **ma-max-csv**: top.txt does not exist
- `llama3.1:8b` **ma-sort**: sorted.txt: expected ['3', '7', '19', '23', '42', '88'], got ["sort(l='split(d='\\n',l=read_file('unsorted.txt'))')"]
- `llama3.1:8b` **ma-time**: answer lacks all of ['12:35']; got '11:35'
- `llama3.1:8b` **ma-percent-of**: tip.txt: expected 351, got 348
- `llama3.1:8b` **mcp-top-word**: top_word.txt does not exist
- `llama3.1:8b` **mcp-find-todo**: todos.txt !~ /^\s*3\s*,\s*7\s*$/; got 'join'
- `llama3.1:8b` **mcp-chars**: chars.txt does not exist
- `llama3.1:8b` **mcp-word-total**: total_words.txt has no number; got 'calculator(expression="(calculator(expression="textkit__text_stats(text="a.tx...'
- `llama3.1:8b` **mcp-fs-count**: count.txt: expected 3, got 45
- `llama3.1:8b` **mcp-lecture-summary**: summary.txt !~ /^\s*(lectures[/\\])?raft(\.txt)?\s*:\s*leader\s*$/; got '[C:\\\\Users\\\\PC\\\\.kiro\\\\crew\\\\scratch\\\\runtime-42c0a4c7\\\\agentos-bench-91jqbmx...'; summary.txt !~ /^\s*(lectures[/\\])?paxos(\.txt)?\s*:\s*acceptors\s*$/; got '[C:\\\\Users\\\\PC\\\\.kiro\\\\crew\\\\scratch\\\\runtime-42c0a4c7\\\\agentos-bench-91jqbmx...'
- `llama3.1:8b` **mcp-find-similar**: first_line.txt does not exist
