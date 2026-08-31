---
name: sweep-conform
description: Make an addition under src/ or ext/ conform to the sweep of #404 before the commit — the manifest row, the unit count, the child map, the coverage entry, the include line, and the two tracker steps no Julia test can reach. Use after adding or changing any file under src/ or ext/, and before committing.
---

# Conforming to the sweep

Run this before you commit an addition under `src/` or `ext/`. It reports every duty the sweep
places on the files your branch touches, and it prints the exact line to paste for each one.

```bash
julia --project=code_health code_health/sweep_check.jl --fetch
```

Without `--fetch` it reads no tracker, which is faster and works offline, but it then cannot check
steps 3 and 4. It never reads a dump an earlier run left behind: a stale answer about an open issue
is worse than no answer. Other forms:

| Command | What it checks |
| --- | --- |
| `sweep_check.jl` | the files that differ from `origin/dev`, plus the untracked ones |
| `sweep_check.jl --base <ref>` | the same, against a ref you name |
| `sweep_check.jl --file <path>` | one path, whatever the diff says |
| `sweep_check.jl --all` | every file in `sweep/manifest.toml`, as a survey |
| `sweep_check.jl --fetch` | any of the above, plus the child map and the sub-issue |

It measures and writes nothing. A `[fail]` line exits the run non-zero. A `[note]` line is a duty
the check cannot settle by itself, and it never fails the run.

`--all` is a survey, and there one level changes: a missing sub-issue is a note rather than a
failure. Most unswept files in the manifest are the standing backlog of an open child map, which
the sweeper files as the map progresses. A file your branch touched is a different thing, and there
the missing sub-issue is a failure.

## The four steps it checks

They are `CLAUDE.md` § *Functionality you add*, and ADR 0084 is the decision behind steps 3 and 4.

1. **The manifest row.** Every tracked `.jl` file under `src/` and `ext/` carries one row in
   `sweep/manifest.toml`. A new file needs a new row with `swept = false`. A file that gained a
   documented unit needs its `units` corrected. `test/test_45_sweep_census.jl` reds the build on
   both.
2. **Coverage.** A new file enters with every line covered, or with a named Coverage Exemption in
   `code_health/rulings.toml`. ADR 0082 owns that rule.
3. **Reopen the child map** that owns the file, and reopen the umbrella, issue #404.
4. **Open one sub-issue** of that child map for the addition.

Steps 3 and 4 write to the tracker. The `sweep-file-issues` skill does them.

## What each failure means

**`no row in sweep/manifest.toml`.** Paste the line the check printed, and choose `map` yourself.
**`map` is not derivable from a path.** Each of the nine subdirectories of `src/` and `ext/` uses
exactly one child map, and there the check names it outright. The top level of `src/` holds files
across five maps, and the numeric prefix does not rescue the lookup: the blocks are not contiguous.
The check prints the candidates and you choose by subject.

**`the unit count moved`.** A documented unit joined the file. This is the case the rule is really
aimed at — a type or a function added to an *existing* file, which already has a row — so record
the new count *and* take steps 3 and 4 for that file. A unit is a docstring that attaches to a
binding. A field docstring inside a struct body is not one.

**`src/PortfolioOptimisers.jl` holds 0 `include` line(s) for it.** Add the `include` in the load
order the file needs. `test/test_47_alias_and_module_census.jl` demands exactly one.

**`child map #N is CLOSED`.** The file joined a map that had finished. Take steps 3 and 4 with the
`sweep-file-issues` skill.

**`no open sweep issue names this path`.** Step 4 is outstanding. The same skill does it.

## Two things the check reports as notes, and you must still do

**A file whose row reads `swept = true` is held to the swept standard now.** Its addition owes a
`# Algorithm` section where the docstring standard demands one, no `# Details` section, a `Where:`
bullet that interpolates `math_dict` rather than copying it, and `# Related` on a dispatch alias.
`test/test_26_docs.jl` holds all four, and it holds the row's `algorithm` count as a floor. Raise
that count in the same commit when the new unit carries the section.

**A new file has no coverage row yet.** The gate ratchets `misses` per file, so a file with no row
enters at zero misses. Write the tests, or write the Coverage Exemption with its rationale.

## Where the rules live

The check is a convenience, never an Authority. Each rule it reports is owned elsewhere:

| Duty | Authority | Gate |
| --- | --- | --- |
| the row, the unit count, the child map | `CLAUDE.md` § *Functionality you add* | `test/test_45_sweep_census.jl` |
| the coverage entry | ADR 0082 | `.github/workflows/ReusableTest.yml` |
| the `include` line | `test/test_47_alias_and_module_census.jl` | the same file |
| the swept standard | `.github/instructions/julia-docstrings.instructions.md` | `test/test_26_docs.jl` |
| the reopened map and the sub-issue | ADR 0084 | `.github/workflows/Sweep.yml` |

`STANDARDS.md` routes any subject the table does not carry.
