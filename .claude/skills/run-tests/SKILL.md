---
name: run-tests
description: Run PortfolioOptimisers.jl test files, coverage runs and gates the way CI and the push guard read them - the machine slots, the targeted tests, the coverage rules and the gate stamps. Use before running any test file, a coverage run or a code_health gate.
---

# Running tests, coverage and gates

`CLAUDE.md` § *Running Julia* points here. Test files, doctests and the `code_health/` gates run
in a fresh bare process, not in a kaimon REPL. The `run-doctests` skill owns the doctests, and the
`sweep-conform` skill gives the order of the gates before a push.

- **A run that loads the package queues for a machine slot.** The test, doctest, JET and
  Expansion Bound drivers wait for a free slot and free memory (`code_health/julia_slot.jl`), and a
  `SLOT wait` line names the holders. Shut down your own kaimon REPL before a full doctest run or a
  JET run. A probe that needs the test dependencies starts its REPL in the test folder of the
  worktree.
- Run **targeted** `test_*.jl` files for the area you changed with
  `julia -t 1 --project=test test/run_files.jl --done=<path> test_X.jl …`, as a background Bash
  command. It runs each file after the `init_code` preamble of `test/runtests.jl`,
  prints one `RESULT` line per file, and writes the verdict to `<path>` when it ends. The header of
  `test/run_files.jl` explains every option. Rerun a failure that only CI shows with `--ci`.
- **A targeted test is one whose result the change can alter.** Trace from the changed code to the
  test files that assert a value passing through it, add the file of the new test, and add the
  censuses that read what the commit touched. A file that only names a changed type is not
  targeted. After a rebase, run `bash code_health/gate_stamp.sh check`. Rerun the files it names
  that your change reaches, and paste the `drop` line it prints for the rest.
- **Every line that your change adds to the code is covered before the push.** Run the targeted
  files with `--coverage=<absolute dir>`. Each `PATCH` line names an added line that CI reads as a
  miss, also in a method that no test calls. Test it, or exempt it (ADR 0082). The push guard asks
  for the `coverage` stamp that a clean run records (#1912).
- **A coverage run stands for every line that a later edit does not reach.** After an edit that
  follows a coverage run, such as the fix of a red file or a test for a `PATCH` line, rerun with
  `--coverage` only the test files that the edit reaches, scoped to the directory of the edited
  source file. Keep the verdicts of the earlier run for the other files. The driver records no
  stamp for a run that measures part of the change, so when the two runs together leave no
  `PATCH` miss, record it with `bash code_health/gate_stamp.sh waive coverage <reason>`, and
  name both runs in the reason.
- If a session hangs, shut it down and start a new one. Do not try to recover it.
