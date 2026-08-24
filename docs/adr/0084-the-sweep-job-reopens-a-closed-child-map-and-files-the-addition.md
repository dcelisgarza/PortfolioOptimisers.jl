---
status: accepted
---

# The sweep job reopens a closed child map, and files the addition as its sub-issue

## Context

[#404](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/404) sweeps every file under
`src/` and `ext/` for three things at once: that its documentation states the mathematics, that its
code agrees with that statement when checked with real numbers, and that its lines are covered or
exempted with a reason. Thirteen child maps carry the work, one per subsystem.

Its standing rule is that any addition joins its child map, and that a closed child map — and the
umbrella — is reopened. `CLAUDE.md`, section *Functionality you add*, states the four steps a person
takes. `test/test_45_sweep_census.jl` reds the build the day the addition lands, so step 1, the
manifest row, is enforced at the gate.

Steps 3 and 4 are not. **A Julia test cannot read the state of an issue, and it cannot reopen one.**
A file whose row says `swept = false` under a child map that has already closed gets none of the
sweep's attention, and nothing says so. #351 is the evidence that this matters: it swept 719 type
docstrings over 32 tickets and found more than fifteen code defects on the way, in a sweep whose
destination was documentation alone. Not one pass came back clean.

Three measurements made while charting this decision each changed the answer.

1. **`map` is not derivable from a path.** Each of the nine subdirectories of `src/` and `ext/` maps
   to exactly one child map, but the top level of `src/` holds sixteen files across **five**. The
   numeric prefix does not rescue the lookup: the blocks are not contiguous, `10_` sits between map
   2 and map 8 and `25_` returns to map 1.
2. **Writing the row needs a capability the job must not have.** A manifest writer needs
   `contents: write` plus a commit or a pull request.
3. **A job that filed every unswept file would file 196 on its first run.** All thirteen child maps
   are open today and every one holds zero sub-issues.

## Decision

### The trigger is one condition, and the trigger is also the deduplication

A file is filed when its row reads `swept = false` **and** its child map is **closed**. Nothing
else. For each such file the job reopens the child map, reopens #404 with it, and opens one
sub-issue of that child map under the `sweep` label.

Two properties follow.

- **Self-suppressing.** Run 1 sees a closed map, reopens it and files. Run 2 sees an open map, so
  the trigger is false and nothing is filed. ADR 0078's deduplication rule is therefore not needed
  here. One cheap safeguard stands on top of it: skip a path that an **open** sweep issue already
  names, by the same whole-word title match ADR 0078 uses.
- **Self-healing.** A sub-issue closed while the flag stays `false` lets the map close again, and
  the next run then refiles. The repair is delayed to the map's close, never lost.

The first run opens nothing, because all thirteen child maps are open.

### There is no cap, and a cap would be wrong

ADR 0078 tops its queue up to five open issues, because its candidates are a standing backlog that
the ratchet already holds still. These are not. Each candidate is a file that entered the library
after its child map finished, and **the first run is the only run that sees the trigger true**: run
2 sees the map that run 1 reopened. A capped run would drop the remainder for ever.

### The census stays red, so the job never writes a row

`test/test_45_sweep_census.jl` keeps failing a file that has no row. A pull request stays red until
the author pastes the row, so the row always arrives by hand and the job never finds one to write.
The job therefore keeps `contents: read`, and **ADR 0073 is not amended**: `sweep/manifest.toml`
keeps its hand-set `map` and `swept` beside its measured `units`.

### The deciding half is `code_health/sweep_triage.jl`

A sibling of `triage.jl`, which is the pattern it copies. It is **not** a fifth instrument: the four
instruments each measure the tree and hold a baseline, and this one measures nothing. It reads two
committed files and a dump of the tracker, and it writes a plan. It opens nothing and it reopens
nothing.

It reuses `CodeHealth.jl`'s TOML handling, its command parsing and its artifact plumbing.
`CodeHealth.jl` loads only `TOML`, which is stdlib, so **the job needs no package instantiate** and
runs in seconds where `CodeHealth.yml` runs for sixteen minutes.

`names_path` moves from `triage.jl` into `CodeHealth.jl`, because both jobs search titles they
already hold and a drifted second copy would silently stop suppressing a duplicate.

### A child map's issue number is read from the tracker, not written down

The job must know whether a child map is **open**, and only the tracker holds that, so the dump is
needed whatever else is committed. The thirteen are picked out of it by the title `Child map <n>:
<name>`, and each name is checked against `sweep/manifest.toml`. A committed table of thirteen issue
numbers would buy nothing and could go stale in silence.

The manifest and the tracker must agree before one issue is planned. A child map with no issue, a
name that differs, or a row naming a map that `[map]` does not list each **throws**, and the job
then writes no plan. A wrong parent is work a person must undo by hand.

### The workflow is `.github/workflows/Sweep.yml`, weekly

Monday 06:00 UTC, the same cadence as `CodeHealth.yml`, plus a `workflow_dispatch` whose `dry_run`
defaults to true. It copies that workflow's fork guard, its `issues: write` permission and its
plan-then-open split. It needs **no** `contents: write` and **no** `fetch-depth: 0`, because its
trigger is a flag in the checked-out tree rather than a rise read from history.

Its shell half differs in what it does: it runs `gh issue reopen` on the child map and on #404, and
it attaches each new issue through the sub-issues endpoint. `CodeHealth.yml` runs a flat
`gh issue create` loop. `reopen.tsv` lists only issues the dump reported closed, because
`gh issue reopen` fails on an issue that is already open.

### The sub-issue mirrors the child map, one file wide

Its title names the path. Every field of its body is generated, so the job needs no judgement: the
path, `map` and `units` from `sweep/manifest.toml`, and `lines` and `misses` from
`code_health/coverage_baseline.toml`. **The body is never machine-read.** The safeguard reads the
title, and every other fact a later run needs comes from a committed file or from the tracker's own
metadata.

### The census prints the candidate maps

Check 1 of `test/test_45_sweep_census.jl` replaces its bare `map = ?`. The candidates are the maps
the file's own directory already uses, so nothing repeats the cut. A file in one of the nine
subdirectories gets its map named outright; a file at the top level of `src/` gets the five
candidates and the person chooses by subject. A file in a brand-new directory gets all thirteen.

### `CLAUDE.md` is unchanged

Its four steps are what a person does. Naming a weekly backstop inside the rule invites the reader
to skip a step, and the job does not cover the open-map case anyway.

## Consequences

- A file added after its child map closed reaches a reopened map and one sub-issue, within a week.
- `CodeHealth.yml` and `Sweep.yml` are two scheduled jobs with the same permission set and the same
  fork guard. Neither writes to the repository.
- The residual gap is a forgotten step 4 under an **open** child map. The census forces step 1 at
  the gate and the job covers steps 3 and 4 only when the map is closed, so that case is caught by
  nothing. It stands as fog on #404.
- ADR 0073 and ADR 0078 are neither amended nor superseded. ADR 0078's title names a rise and its
  refile clause reads git history; this job does neither, so folding it in would make one ADR
  describe two mechanisms.
- `STANDARDS.md` routes the subject of a late addition to `Sweep.yml` beside the census.

## Verification

`code_health/sweep_triage.jl` was run against an isolated scratch repository holding a new file,
`src/11_Phylogeny/07_NewThing.jl`, whose row read `swept = false` under a child map 4 the dump
reported closed, with every other file of that map marked `swept = true`.

| Run | Tracker state | Planned | Reopened |
| --- | --- | ---: | --- |
| 1 | map 4 closed, #404 open, no sweep issue | 1 | #418 |
| 2 | map 4 open | 0 | none |
| 3 | map 4 closed, an open sweep issue names the path | 0 | none |
| 4 | map 4 closed, that sweep issue closed | 1 | #418 |
| 5 | map 4 closed, #404 closed | 1 | #418, #404 |

A run against the live tracker planned nothing and reopened nothing, which is the first run's own
predicted outcome. A missing child map, a renamed child map and an omitted `--maps` each threw and
wrote no plan. A run under an empty `JULIA_DEPOT_PATH` succeeded, which is the no-instantiate claim.

`test/test_45_sweep_census.jl` passes with 10 assertions. On a scratch tree holding two files with
no row it printed the one map for `src/11_Phylogeny/07_NewThing.jl` and the five candidates 1, 2, 8,
10 and 13 for `src/26_Scratch.jl`.

## Amendment (2026-08-24)

The body carries a fixed `## Routing` block above its Destination. It names `STANDARDS.md`, the two
Authorities a sweeper reads, and `CONTEXT.md`, each one directly.

The original body routed through #404 alone: *"Every rule for this effort lives on #404. Read it
first."* #404 does name those files, inside five thousand words. Measured over the six open sweep
tickets on 2026-08-24, three named an Authority and none named `STANDARDS.md`. A standard that the
sweeper is never routed to does not hold, however well it is written.

The block is constant text, so the sentence *"Every field of its body is generated, so the job
needs no judgement"* still holds: a constant needs no judgement either. Nothing here is
machine-read, which is unchanged. #404 keeps the rules of this effort alone, and the Notes keep
their pointer to it.

The same block was back-filled by hand into the 30 open tickets of child maps 1 to 4. #478 is the
map that asked for this, ADR 0085 records the decision, and #484 is the ticket that carried it out.
