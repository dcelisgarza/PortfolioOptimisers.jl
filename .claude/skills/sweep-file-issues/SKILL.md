---
name: sweep-file-issues
description: Write, wire and reopen the sweep parent and sub-issues of #404 on the GitHub tracker for a file that joined the library without conforming — reopen the closed child map and the umbrella, open one sub-issue per file, and parent it. Use when sweep-conform reports step 3 or step 4 unmet, or when an addition already landed unswept.
---

# Filing the sweep issues

Use this when a file under `src/` or `ext/` reads `swept = false` and owes steps 3 and 4 of
`CLAUDE.md` § *Functionality you add*: reopen the child map of #404 that owns it, reopen #404, and
open one sub-issue of that map for the file. ADR 0084 is the decision.

**This skill writes to the tracker.** Read the dry run before the apply, every time.

## The three commands

```bash
# 1. Plan. It reads the tracker and writes code_health/_sweep/. It opens nothing.
julia --project=code_health code_health/sweep_triage.jl --fetch --file src/26_New.jl

# 2. Read the plan.
code_health/sweep_issues.sh apply --dry-run

# 3. Apply it.
code_health/sweep_issues.sh apply
```

Drop `--file` to plan every file the weekly job would file, which is every row that reads
`swept = false` under a **closed** child map. Repeat `--file` for more than one path.

`sweep_issues.sh` refuses any repository but `dcelisgarza/PortfolioOptimisers.jl`, and
`.github/workflows/Sweep.yml` calls the same script, so a hand run and the weekly job cannot drift.

## When `--file` is needed, and when it is not

The weekly job's trigger is one condition: a row that reads `swept = false` **and** a child map
that is **closed**. That trigger is also its deduplication — run 1 reopens the map, and run 2 then
sees an open map and files nothing.

So the job cannot reach a file that joined an **open** child map. That is the residual gap ADR 0084
names, and `--file` is what closes it. A path named with `--file` is planned whatever the state of
its map, and its sub-issue body says the map was already open rather than claiming a reopen that
never happened.

Nothing else is relaxed. The path must have a row, the row must read `swept = false`, and a path an
open `sweep` issue already names is skipped rather than filed twice. A path with no row throws, and
the message routes you to the census that prints the line to paste — **fix the row first**, in the
commit that owes it.

## What the plan holds

`code_health/_sweep/` after step 1:

| File | What it is |
| --- | --- |
| `maps.tsv` | every `wayfinder:map` issue, as `number<TAB>state<TAB>title` |
| `existing.tsv` | every `sweep` issue, the same shape |
| `reopen.tsv` | the issues to reopen, one number per line, **each of them closed** |
| `plan.tsv` | one row per sub-issue, as `stem<TAB>path<TAB>parent issue` |
| `NN.title`, `NN.body` | the text of each sub-issue |
| `summary.md` | the same plan as a table |

Read `NN.body` before the apply. Every field of it is generated from `sweep/manifest.toml` and
`code_health/coverage_baseline.toml`, so a wrong number there is a wrong number in a committed
file, not a wrong number in the plan.

`reopen.tsv` lists only issues the dump reported closed, because `gh issue reopen` fails on an
issue that is already open.

## When it throws

The manifest and the tracker must agree before one issue is planned, and three disagreements each
throw with no plan written:

1. A child map the manifest names has no issue titled `Child map <n>: <name>`.
2. An issue's name differs from the manifest's, so one of the two was renamed alone.
3. A row names a map that `[map]` does not list.

A loud refusal is right. A wrong parent is work somebody must undo by hand.

## After the apply

The sub-issue closes when the file's row reads `swept = true`. Close it yourself in the commit that
finishes the sweep of that file, and name the issue in the commit message:

```bash
gh issue close <number> --comment "Swept in <commit>."
```

A closing keyword in the commit message does not close it. GitHub acts on the keyword only when the
commit reaches `main`, and `main` is current to the last release.

## Before it

Run the `sweep-conform` skill first. Steps 1 and 2 — the manifest row and the coverage entry — are
committed files, and this skill neither reads a missing row into existence nor writes one.
