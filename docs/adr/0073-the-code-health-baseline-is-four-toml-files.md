---
status: accepted
---

# The code-health baseline is four TOML files, split on who writes them

## Context

The JET and complexity gate is a ratchet against a committed baseline. Six earlier decisions each
named a committed artifact and left it unsited, so this decision inherits seven things to place, not
one.

| Artifact | Written by | Fixed by |
| --- | --- | --- |
| The CodeComplexity number per file | a generator | the complexity measurement |
| The JET raw and reviewed count per file | a generator | the JET measurement |
| The Expansion Bound, about 21 rows | a generator | [ADR 0072](0072-the-complexity-gate-measures-src-and-ext.md) |
| Dismissals and Rationales | a human | [ADR 0071](0071-a-dismissed-jet-report-is-keyed-by-file-kind-and-message.md) |
| Exemptions, keyed `(path, definition, metric)` | a human | the scheduled job's rules |
| Unmeasured Paths | a human | [ADR 0072](0072-the-complexity-gate-measures-src-and-ext.md) |
| The thresholds, 10 / 15 / 5 / 1 | a human | the scheduled job's rules |

Four measurements constrain the answer.

1. **The three generated units cost two orders of magnitude apart.** The complexity measurement is
   4.8 s warm. The JET measurement is 5 minutes 46 seconds and 2.4 GiB peak. The Expansion Bound is
   about 80 seconds.
2. **JET's 315 reports land in 68 of the 196 files in scope**, so 128 files measure zero.
3. **The attribution rule changes the file's size.** Keying on the last frame keeps 77 reports over
   33 files. Keying on the deepest repository frame keeps all 315 over 68 files.
4. **A `src/` file can be named by two different JET runs.** `report_package` needs a separate call
   per extension module, and 4 of the Plots extension's 70 reports attribute back into `src/`.

## Decision

### Four files under a new top-level `code_health/`, split on authorship

```text
code_health/
  complexity_baseline.toml   generated
  jet_baseline.toml          generated
  expansion_bound.toml       generated
  rulings.toml               hand-written
```

The split is on **who writes the file**, not on which tool the data belongs to and not on which job
reads it. A refresh rewrites a whole generated file, so no human paragraph may share one.

Splitting per tool was rejected for that reason: each file would then mix generated rows with
hand-written Rationales, and every refresh would need a partial rewrite. One file for everything was
rejected because every branch touching any part of the gate would conflict with every other.

The Expansion Bound takes its own file rather than riding inside the complexity baseline, because
the three generated units **refresh independently**. Coupling them would make a 4.8 s refresh cost
5 minutes 46 seconds.

`rulings.toml` is named for what it holds. All five of its sections are human judgements, and
nothing in it is measured.

The directory is neither source nor test. `.github/` was rejected because the refresh must run
locally, so GitHub does not own the data. A hidden `.code_health/` was rejected because
`rulings.toml` is hand-edited often. A `.toml` file adds nothing to ADR 0072's coverage assertion,
which counts tracked `.jl` files alone.

### TOML, one inline table per line, printed by the generator

TOML is a Julia stdlib, so the gate takes no dependency to read its own data. The stdlib parser
accepts an **inline table**, so a file's whole measurement fits on one line:

```toml
[file]
"src/01_Base.jl" = { cyc = 3, cog = 1, arg = 2, sum = 40, macros = [] }
```

`TOML.print` cannot write that shape. Given the document above it emits six lines for the one entry,
and the keys come out in hash order rather than declaration order:

```toml
[file."src/01_Base.jl"]
argcount = 2
sum = 40
cyclomatic = 3
macros = []
cognitive = 1
```

An unstable key order makes every regeneration a whole-file diff. So the generator prints the lines
itself, sorted by path. That is about ten lines of code, and it buys a fixed key order and a
one-line diff per file.

JSON was rejected: it takes a package dependency, it carries no comments, and appending an entry
edits the line above it. A `.jl` data file was rejected: it makes the gate's data executable code
that Revise tracks and JuliaFormatter reformats, and a malformed refresh becomes a load error
rather than a parse error.

### Every file in scope gets a row, so absent means out of scope

All 196 files appear in both baselines, the zeros included. The JET baseline repeats the set in
every run, so it carries about 588 rows.

The alternative reading — a baseline that lists only files with a finding — makes "absent" mean
measured-zero, or new, or out of scope, and no rule can tell those apart. It also leaves ADR 0072's
macro marker with no row to live in, so the five `Windowed*` structural zeros would be
indistinguishable from clean files.

### A JET report attributes to its deepest repository frame

Selection uses JET's own `AnyFrameModule(PortfolioOptimisers)` keyword. Attribution walks
`report.vst` and takes the deepest frame under `src/` or `ext/`. This is the rule
[ADR 0071](0071-a-dismissed-jet-report-is-keyed-by-file-kind-and-message.md) refers to as "the
attribution rule the baseline uses".

| Rule | Reports kept | Files |
| --- | --- | --- |
| **Deepest repository frame** | **315 of 315** | **68** |
| Last frame | 77 of 315 | 33 |

Last-frame keying discards 238 reports, 164 of them ending in `Base_compiler.jl`. A defect in
package code is thrown away whenever execution ends inside a dependency. Shallowest-frame keying
was rejected separately: it piles hundreds of reports onto the few entry-point files, so a
regression deep in a helper turns an entry file red and the per-file key stops localising anything.

The objection that a stack-walking rule cannot be stable is answered by measurement. Across two
fresh processes the multiset of kind, reported site and **attribution site** was identical for all
315 reports, even though one report held 7 frames in one run and 8 in the other. The depth flickers.
The attribution does not.

### The JET baseline is keyed by run and file

Each `report_package` call is a run with its own load set, and **each run holds a row for every
file in scope**, zeros included. That is about 588 rows over the three runs.

```toml
[run.main]
load_set = ["StatsPlots", "GraphRecipes", "Impute"]

[run.main.file]
"src/01_Base.jl" = { raw = 0, reviewed = 0 }
"src/19_RiskMeasures/Plotting.jl" = { raw = 0, reviewed = 0 }
"ext/PortfolioOptimisersPlotsExt.jl" = { raw = 0, reviewed = 0 }

[run.plots_ext]
load_set = ["StatsPlots", "GraphRecipes"]

[run.plots_ext.file]
"src/01_Base.jl" = { raw = 0, reviewed = 0 }
"src/19_RiskMeasures/Plotting.jl" = { raw = 1, reviewed = 0 }
"ext/PortfolioOptimisersPlotsExt.jl" = { raw = 66, reviewed = 0 }
```

A sparse run table was rejected by
[ADR 0074](0074-the-baseline-row-set-is-total-and-a-rename-pairs-by-measurement.md). Set equality
needs one expected file set per run, and under a sparse table a `src/` file that gains its first
extension-run report reads as an added row rather than a rise.

**No number is ever summed**, so the double count cannot arise. The rule is chosen because it is
correct whether or not the 4 re-attributed reports duplicate ones the main run already found, which
nobody has established. A flat summing table would be correct only under one answer and would
inflate a file's number forever under the other.

The complexity baseline stays flat. It is one parse pass over every file, so it has no runs.

Two things follow. `load_set` and `filter` live inside a run's section, while `julia`, `jet`,
`code_complexity` and `commit` stay top-level. And a Report Fingerprint keeps its three components
with **no run**, so one Dismissal matches in every run that produces the class, and
`reviewed = raw − matched dismissals` is computed per run and file.

### Provenance is compared, and any mismatch turns the gate red

```toml
[provenance]
julia = "1.12.0"
jet = "0.12.1"
code_complexity = "0.2.0"
commit = "6d7dd10cef"
```

The Julia version, the JET version, the CodeComplexity version, the attribution filter and the load
set are compared against the run. Any mismatch fails. The commit is context for a reader and is
never compared, because the baseline is always older than the tree.

The direction matches every other rule the gate makes: the `JET_AVAILABLE` guard fails loudly, a
broken fingerprint fails red, a dead baseline row fails red. A tool bump moves the numbers, so it
must force a refresh rather than print a warning nobody reads.

### A refresh lowers by default, and a rise needs an explicit act

The ordinary refresh moves a number **down** only. Its diff is therefore always an improvement, and
it is safe to run without thought. A row that rose is an error naming the file, the old number and
the new one, and recording it needs a second, named act.

```text
$ refresh
ERROR: src/20_Optimisation/11_MeanRisk.jl rose 12 -> 15.
Re-run with --accept-rise to record it.
```

A single measured-truth command was rejected: it lets a contributor whose build is red go green with
one unremarkable command, and the only control left is that a reviewer notices the number in the
diff. A refresh that can never raise a number was also rejected: a genuinely more complex algorithm
could then never land, which makes the gate unmergeable rather than strict.

### A merge conflict on a row is accepted

One line per file means two branches that lower **different** files merge cleanly. A conflict arises
only when both change the **same** source file's measurement, and there it is a true statement:
neither number is right for the merge.

**The resolution is to re-run the refresh, never to hand-pick a number.**

A `merge=union` driver in `.gitattributes` was rejected. It produces a duplicate TOML key, so the
merge succeeds and the gate fails later with a parse error, instead of stopping the person who is
merging. One committed file per source file was rejected because two branches changing the same
source file still collide, so it buys nothing and costs 196 paths.

## Consequences

**The tools cannot float behind a bare `Pkg.add`.** `Aqua.yml`'s precedent installs its tool in the
workflow with no version bound. Under the provenance rule that would turn CI red on every routine
JET or CodeComplexity release. The gate's workflow must pin both to an exact version and bump them
deliberately, in the commit that refreshes the baseline.

**The rule for a file the baseline does not name became a single question**, and
[ADR 0074](0074-the-baseline-row-set-is-total-and-a-rename-pairs-by-measurement.md) settled it.
Absent means out of scope, so the gate compares two file sets rather than inventing a number.

**The JET measuring script makes three calls, not one** — the package with its extension triggers
loaded, and one call per extension module.

**The JET baseline carries mostly zeros.** 128 of the 196 files measure zero in the main run, and
the two extension runs are nearly all zeros. That is the price of a total baseline and it is
accepted.

**The vocabulary stays out of `CONTEXT.md`**, for the reason
[ADR 0071](0071-a-dismissed-jet-report-is-keyed-by-file-kind-and-message.md) and
[ADR 0072](0072-the-complexity-gate-measures-src-and-ext.md) both gave. That file's preamble scopes
it to the library's domain, and a baseline file is repository process.

**"policy", "judgements" and "code_health" were considered for the hand-written file and rejected.**
"policy" carries none of this map's vocabulary. "judgements" has a spelling that splits between
British and American English. "code_health" repeats the directory name and does not distinguish
itself from the three generated files beside it.
