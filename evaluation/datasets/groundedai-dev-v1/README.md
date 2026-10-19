# groundedai-dev-v1 (development set)

A small synthetic set used **only for developing** prompts and answer handling. Results on it
are not reported as evaluation results.

Why it exists: the 56 questions in `groundedai-synthetic-v1` are the evaluation set. Changing
the prompt until those questions score well would tune the system to its own test. Prompt
changes are therefore tried here first, and the evaluation set is run once with the frozen
version (see `docs/EVALUATION.md`).

* 7 documents about fictional organisations (Kestrel Valley Rail, Fenwick Public Library,
  Halden Coworking, Quillon Software): 6 plain-text files and 1 Markdown file. None of them
  appears in the evaluation set.
* 25 questions in the same format and categories as the evaluation set: factual (12, two of
  them paraphrased), multi_chunk (2), synthesis (1), unanswerable (4), out_of_scope (2),
  conflicting (2) and injection (2).
* Deliberate traps: the 2026 rail policy and the 2024 FAQ give different claim windows
  (28 vs 14 days); the library's borrowing rules and quick guide give different renewal
  limits (three times vs twice); the coworking community board contains a planted
  instruction.

Written 2026-10-13. Licence: MIT, like the rest of the repository.
