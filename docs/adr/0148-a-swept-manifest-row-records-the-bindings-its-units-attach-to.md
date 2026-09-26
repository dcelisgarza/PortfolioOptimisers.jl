---
status: accepted
---

# A swept manifest row records the bindings its units attach to

## Context

`code_health/sweep_manifest.toml` carries one row per file under `src/` and `ext/`, and
`test/test_45_sweep_census.jl` compares each row's `units` — the count of documented units — against
the file. The count was chosen because it rises on every addition the docstring standard demands a
docstring for, so a type or a function added to an existing file reds the build on the day it
lands, and the addition joins its child map of #404 (ADR 0084).

A count sees an addition and a deletion. It cannot see a **replacement**, and it cannot say which
units the row's number stands for. Issue #1065 is the case: `src/09_Distance/05_FeatureDistance.jl`
was swept under #450 at 23 units, and map #802 then rewrote its estimator and its entry points:
ticket #811 removed `assert_feature_selector` and the integer entry of `sel`, and ticket #810
removed `assert_feature_matrix_supplied` and added `feature_matrix` and `feature_labels`. The count moved
at each commit and the row was corrected each time, to 23 and then to 24, with `swept = true` kept
on every correction. Against the swept text the file has lost one unit and gained two, and the
manifest vouched for a text the sweep had never read. Neither the census nor
`code_health/sweep_check.jl` had anything to say, because each asked only that the number match.

The review that found it offered two shapes: reopen the one row and re-sweep the file, or teach
the census to read the unit **names** beside the count. The first repairs one file. The second
makes the next replacement visible on the day it lands, which is what the count was built to do
for an addition.

## Decision

**A row whose `swept` is true carries a `bindings` key: the name of the binding each documented
unit attaches to, sorted, one entry per unit.** A documented method names its function, so a
function with three documented methods is listed three times, and the list is a multiset whose
length is `units`. `CodeHealth.documented_bindings` is the one definition, and it walks the same
`Core.@doc` macrocalls `CodeHealth.documented_units` counts.

**The census compares it.** `test/test_45_sweep_census.jl` gains a fourth check: a swept row's
`bindings` equals the measured list, and the failure prints the names added, the names removed,
and the line to paste. `code_health/sweep_check.jl` reports the same drift before the commit. A
swept row with no list, and an unswept row with one, are each a failure of their own: the key is
written by the session that flips `swept`, as `algorithm` is, and a list on an unswept row would be
a stale claim the moment the file changed.

**What a drift asks for is what a count drift asks for.** The rewritten units join the file's
child map as an addition would — a sub-issue under the map, and the map and the umbrella reopened
if they had closed — and the session records the new list. The row keeps `swept = true`. The flag
arms the swept standard in `test/test_26_docs.jl`, and a rewrite must still meet it; flipping the
flag would relax the `# Algorithm` floor and the three other swept-only demands on a file that
meets them.

**The list is recorded in the row, not hashed.** A hash would keep every row on one short line,
and it would tell a reader nothing. The list lets the census print which units changed, and lets
a reviewer of the manifest's diff read the same names in the pull request.

**The instance is reopened the same way.** The row of `05_FeatureDistance.jl` records its current
bindings and keeps `swept = true`, and a sub-issue under child map 2 (#416) names the six rewritten
units for a re-sweep.

## Consequences

A unit replaced one for one under a swept row reds the build. A unit rewritten **under its own
name** — a method whose signature changed but whose function did not — does not, and that is
accepted: the names gate the set of units, and the sweep gates the text of each. A signature would
gate more and move under more, and the case that was found is the case the names see.

A swept row is long. `src/01_Base/08_TypeAliases.jl` lists forty-five names on one line. The
manifest is machine-read and pasted, never typed, and a long line in a data file costs a reader
nothing that a diff does not give back.

Every printer of a manifest row takes the list, so a line pasted from any of them keeps it. The
one hand-built printer, in `test/test_26_docs.jl`, now calls `CodeHealth.row_line` with the rest.

`test/test_51_code_health_root_seam.jl` drives `documented_bindings` over a fixture tree: the
sorted names, a second documented method, a `module` naming itself, and a docstring whose target
names nothing, which is recorded as `<anonymous>` so that the list never loses an entry.

## Alternatives considered

- **Reopen the row alone.** Set `swept = false`, re-sweep, and change nothing else. The next
  rewrite passes as silently as this one did, and the flag's relaxation costs the file its
  swept-only gates while the re-sweep waits.
- **A hash of the names.** One short token per row, and an opaque failure. Rejected above.
- **The signature of each unit rather than its name.** Tighter, and it moves on a keyword added,
  a type bound widened or a default changed, none of which the sweep's three conditions ask a
  session to re-read. The names catch the found case at the granularity the count already uses.
- **A separate `code_health/sweep_bindings.toml`.** ADR 0073 splits the code-health files on
  authorship, and this list is measured and pasted like `units`, so it belongs on the row beside
  the number it extends.

## Amendment (2026-09-24)

**A drift under a swept row no longer asks for a sub-issue.** The commit that adds a unit to a
swept file, or replaces one in it, sweeps the new units itself. It records the new row, keeps
`swept = true`, and does not reopen the child map. Issue #1310 is the cause.

The Decision above says that a drift asks for what a count drift asks for: a sub-issue under the
child map. The tooling did not agree. `code_health/sweep_check.jl` asked for the sub-issue, and
`code_health/sweep_triage.jl --file`, the one tool that plans it, refuses a row that reads
`swept = true`. #1308 met the conflict: it added one private unit to a file that #1173 had swept
one commit before, and #1309 was filed by hand. That sub-issue recorded nothing that the commit
did not, because the commit wrote the unit to the swept standard and #1309 closed on the same
day.

Two fixes were offered. The first let `--file` plan a swept row and name the added units. The
second removed the demand. The maintainer chose the second, so the tracker records only the work
that the commit leaves.

What the commit owes is the swept standard, applied to the new units:

- the three conditions of the manifest header: the documentation states the mathematics, the code
  agrees with that statement, and the lines are covered or exempted;
- the prose passes `/unslop`;
- the four swept-only demands that `test/test_26_docs.jl` holds, with the `algorithm` floor raised
  when a new unit carries the section.

**A commit that cannot meet it sets `swept = false`.** It drops `algorithm` and `bindings` from
the row, and the change then takes the four steps of an addition to an unswept file, which
`sweep_triage.jl --file` plans. This relaxes the swept-only gates on that file until the re-sweep,
which the Decision above rejected as the default. It is kept here as the exception, because it is
the only path that leaves the tracker holding the work that no commit did.

`code_health/sweep_check.jl` prints the new demand on both failures of a swept row and no longer
asks for steps 3 and 4 there. `test/test_45_sweep_census.jl`, the manifest header, `CLAUDE.md`
§ *Functionality you add* and the `sweep-conform` skill state the same rule. The census, the
`bindings` list and the rest of the Decision are unchanged.
