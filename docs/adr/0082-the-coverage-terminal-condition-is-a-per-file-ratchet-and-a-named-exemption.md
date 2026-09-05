---
status: accepted
---

# The coverage terminal condition is a per-file ratchet and an exemption named by definition

## Context

[Issue #404](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/404) sweeps every file in
`src/` and `ext/` for three things, and the third is coverage: a file is done when every line in it
is covered or carries a reasoned exemption. It reports that **nothing gates coverage today**. The
repository has no `codecov.yml` at all. `lcov.info` is written by `julia-processcoverage` in one
matrix cell of `ReusableTest.yml`, uploaded to Codecov, and shown on the badge, and no check reads
it.

Measured on `main` at `c238c764`: **90.0 %** over 190 files, 18516 lines and **1851 misses**. 96
files stand at 100 %. The misses concentrate hard. `ext/PortfolioOptimisersPlotsExt.jl` alone holds
**376 of them, 20 % of every miss in the library**, and two files stand at exactly **0.0 %**:
`src/08_Moments/36_RegimeAdjustedExpWeightedVariance.jl` and
`src/08_Moments/37_RegimeAdjustedExpWeightedCovariance.jl`.

Those two zeros are the case this ADR is built around. Nothing stopped them. A file entered the
library with no test at all, and the repository-wide number moved by less than one point.

Four ADRs already fix the shape of a gate in this repository, and this one takes all four rather
than inventing a fifth shape. [ADR 0073](0073-the-code-health-baseline-is-four-toml-files.md) splits
the baseline files on **who writes them**, so no human paragraph shares a file that a refresh
overwrites. [ADR 0074](0074-the-baseline-row-set-is-total-and-a-rename-pairs-by-measurement.md)
makes the row set total and pairs a rename by equal measurement.
[ADR 0075](0075-a-run-that-trips-publishes-the-refresh-artifact.md) hands a contributor the Refresh
Artifact that clears a trip. [ADR 0076](0076-the-code-health-pass-rule-is-a-ratchet.md) makes the
pass rule a ratchet against a committed baseline.

## Decision

### The ratchet is per file, and the miss count is what binds

One row per file in scope, in `code_health/coverage_baseline.toml`:

```toml
"src/04_PosdefMatrix.jl" = { lines = 137, misses = 1 }
```

`misses` binds. `lines` is context and never binds, on the same split `complexity.jl` draws between
a file's maximum and its sums.

**Per file, not repository-wide.** A repository-wide number is cheaper to read and it is blind to
exactly the failure that produced the two zeros: 126 misses arriving in one new file move an 18516
line total by less than a point, and a fall anywhere else pays for them. Every other instrument in
`code_health/` is already per file, so a repository-wide number could also not be annotated on a
pull request, could not be attributed to an owner, and could not be paired across a rename.

**The miss count, not the percentage.** A percentage rises when an uncovered line is **deleted** and
rises when a covered line is **added**, so a file can improve its percentage while gaining misses.
The miss count is the number #404 drives to zero and it moves in the right direction on its own.
Adding a covered function raises `lines` and leaves `misses` alone, so ordinary work is quiet.

### A file added to the tree enters terminal

ADR 0074 asks an added file to pass an entry test before it may take a row. The complexity gate's
entry test is the scheduled job's candidacy rule. Coverage has no scheduled job and no threshold, so
its entry test is the destination itself: **a new file enters with every line covered, or with a
Coverage Exemption for every line that is not.**

A new file has no legacy to plead. Without this rule a per-file ratchet would have recorded the two
zeros rather than refusing them, which is a gate that measures a defect instead of stopping it.

### A Coverage Exemption is keyed by definition, and states its count

The hand-written half lives in `code_health/rulings.toml`, beside the Exemptions and Dismissals it
resembles, so one file holds every human judgement about this library's code and the `[rationale.…]`
mechanism, its maintainer-approval rule and its CI flag are all reused unchanged.

```toml
[[coverage_exemption]]
path       = "src/…/….jl"
definition = "show"
misses     = 3
rationale  = "a-named-rationale-block"
```

**The key holds no line number.** A line number churns on every insertion above it, and
JuliaFormatter reformats this whole repository in place, so a line-keyed row would be wrong more
often than right. A definition name survives both. ADR 0071 keys a Dismissal on (file, kind,
message) for the same reason, and the Exemptions above it are already keyed (path, definition,
metric).

An uncovered line is attributed to the innermost **top-level** definition whose source range holds
it, found with `Meta.parseall` alone. A closure inside a function is attributed to that function,
because a human writes and reads these rows and a human names the method. A line inside no named
definition is attributed to `<toplevel>`, which is a legitimate target: a `const`, an `include` and
a `precompile` block all land there. A Declaration Macro call is named by the definition it wraps,
so several `@concrete` calls in one file do not collapse onto one ambiguous key.

**The count is part of the claim, and it is held to equality in both directions.** A claim above the
truth is a stale row whose lines were covered, and the fix is to lower it or delete it. A claim
below the truth is the one leak a file total cannot see: a line covered elsewhere in the file pays
for a new uncovered line inside the exempted definition, and the ratchet reads a flat number.
Equality closes both, so an exempted definition never goes blind.

### An Exemption binds the terminal condition, never the baseline

A Coverage Exemption **never moves a baseline number**. The baseline records what was measured, and
the ratchet still sees every miss. The row binds one thing: whether a file is **terminal**, which
is the condition #404 sets for a child map to close on coverage.

```text
residue(file) = misses(file) - the misses its Coverage Exemptions account for
terminal      = residue == 0
```

`julia --project=code_health code_health/coverage.jl terminal src/08_Moments/` answers for a child
map's own files and names the definitions that still hold an uncovered line. This mirrors ADR 0072,
where an Exemption binds issue candidacy alone and never touches a baseline number.

### The exemption file and the sweep manifest are separate, and neither implies the other

`code_health/coverage_baseline.toml` is generated, `code_health/rulings.toml` is hand-written, and
the sweep manifest of #404 is a third file with a third writer. ADR 0073 splits on exactly this
line, and merging any two of them would make one file that two jobs rewrite.

They are orthogonal by design. `swept = true` says the sweep visited the file and its documentation
states its mathematics. A Coverage Exemption says why one line in it stays uncovered. **A file may
be `swept` and still hold exemption rows**, and a file at 100 % coverage may be unswept.

### Nothing in the coverage provenance binds

The other three baselines pin their analyser and fail the run on a mismatch, because a parser that
moves invalidates the numbers it took. Coverage is produced by Julia itself in the test job, and
[ADR 0056](0056-the-julia-setup-action-floats-on-latest.md) floats that job on the newest release. A pin here
would turn the gate red on a Julia patch release for no defect. The provenance block records the
Julia version and the commit for a reader and neither is compared. A move in Julia's line
attribution appears as an ordinary rise, and the Refresh Artifact clears it on the same route as any
other rise.

### `codecov.yml` is the visible half and blocks nothing

Every status in `codecov.yml` is `informational: true`. Codecov keeps the badge, the per-line view a
reviewer reads and the pull request comment. The binding gate is `coverage.jl`, which reads the
`lcov.info` the run has just written.

Two blocking gates on one number disagree the first time Codecov's report and the run's own
`lcov.info` differ, and they differ for reasons that are nobody's defect: a dropped upload, a retried
job, a base commit whose report never finished. `codecov.yml`'s `ignore` list is the Unmeasured Paths
of ADR 0072 stated the other way round, and the two lists move together.

### A line reached only by its own `jldoctest` is a miss

Coverage is the test suite's number. The `doctest` job of `Docs.yml` runs
`doctest(PortfolioOptimisers)` under `--project=docs`. It passes no `--code-coverage` flag, it runs
`julia-actions/julia-processcoverage` over nothing, and it uploads nothing. `coverage.jl` reads the
`lcov.info` that the test job wrote, and reads nothing else. A line whose only exercise is its own
example is therefore a miss. This is the intended answer, not an accident of the wiring.

Three reasons.

- **A `jldoctest` block asserts a rendering, not a result.** The job sets `set_compact_show!(false)`
  to hold the printed form stable. A block that prints a struct exercises `show`, and it proves
  nothing about the mathematics that built the struct. Check 2 of
  [#404](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/404) asks for the opposite:
  compute the number, then compare it with the number the code returns.
- **Counting the examples would shrink the sweep's work list without doing the work.** The lines it
  would mark covered are the lines a child map must read hardest, because a function that no test
  calls is a function whose result nobody has checked.
- **The sweep writes the test anyway.** A child map that reaches such a line already holds the value
  it computed for check 2, and turning that value into a `@test` is one line.

**A Coverage Exemption never cites an example.** "The doctest reaches it" states that the line is
easy to reach, not that it cannot be reached, so the row is refused.

Instrumenting the `doctest` job was weighed and set aside. It buys a lower miss count rather than
more checked mathematics, and it costs a second `lcov.info` to merge and a second Codecov upload. A
separate Codecov flag that measures the examples without counting them was set aside for the same
reason, plus a second number on the badge page that no gate reads.

### The gate runs in its own job, fed by an artifact

`ReusableTest.yml` gains a `coverage` job. The test job uploads `lcov.info` as an artifact, and the
`coverage` job downloads it, sets up the Julia that `code_health/Manifest.toml` pins, and runs
`coverage.jl check`. The hand-over keeps both pins intact: the test job stays on the floating
release of ADR 0056, and the gate stays on the pinned one of ADR 0077. Putting the job inside the
reusable workflow gives `Test.yml` and `TestOnPRs.yml` the same gate from one place.

`needs: test` means a failing suite skips the ratchet. Coverage taken from a run that did not finish
is partial, and a ratchet fed partial data reports a fall that is not one.

## Consequences

- **A file cannot silently arrive with no test.** The entry test refuses it, and the two files at
  0.0 % are the measured case it answers.
- **A regression is attributed to its file** and annotated on the pull request, and the Refresh
  Artifact records it in one drop-in when the rise is deliberate.
- **A ratchet is blind to a fall and a rise inside one file**, which ADR 0076 already accepts. The
  exemption's equality rule closes the part of that blindness that sits inside an exempted
  definition, and no more.
- **Coverage is execution-derived, so it is less deterministic than a parser's number.** A test
  skipped by a failed solve moves a miss count with no code change. The route is the same as any
  other rise, and if such a trip proves common the answer is a Coverage Exemption with a Rationale
  that says so, not a tolerance.
- **The first baseline is a provisional seed.** It was taken from the Codecov report for `main` at
  `c238c764`, because #404 forbids starting the full suite locally and no other measurement of the
  whole library exists. 147 of the 196 files in scope changed after that commit, so a row may stand
  below the truth. The first CI run on the landing commit either passes, which confirms the seed, or
  trips and publishes the exact file that replaces it. This is ADR 0075's route used for its first
  purpose rather than a new mechanism.
- **The `<toplevel>` key is coarse.** Every uncovered line of a file that lies outside a named
  definition shares one row. This is accepted: such lines are few, and splitting them would need a
  key that a reformat can break.
- **A line reached only by its own example stands as a miss**, and no Coverage Exemption may cite
  an example. The sweep writes the test instead. CI does exercise such a line on every push, so the
  number understates what runs, and that cost is accepted.

## Amendment (2026-08-27)

### A tripped ratchet does not red the test check, and the gate still runs beside the suite

*The gate runs in its own job, fed by an artifact* above put the `coverage` job inside
`ReusableTest.yml`. A job of a reusable workflow is a job of whichever caller ran it, so a tripped
ratchet turned `Test` and `Test on PRs` red. That conflates two claims. The test check answers
whether the library works. The ratchet answers whether one file's miss count rose, which is a
question about the sweep's progress and never about whether the code runs.

Splitting the two claims does not need a second workflow, and a second workflow costs the gate its
reach. A workflow triggered by `workflow_run` fires only for a workflow file that is already on the
default branch, which is GitHub's rule. `main` is current to the last release, so such a gate stays
silent on `dev` and on every pull request until a release carries the file over, and every later
edit to it is inert until the next release. A gate that cannot run on the branch where the work
happens gates nothing.

The `coverage` job therefore stays where the ADR put it, and the split is made at the step instead:

```yaml
- name: Coverage ratchet
  id: ratchet
  continue-on-error: true
  env:
    COVERAGE_LCOV: coverage_download/lcov.info
  run: julia --project=code_health code_health/coverage.jl check
```

`continue-on-error` on the step, rather than on the job, is deliberate. It settles the conclusion
inside the job, where the rule is plain: the step is marked failed and ignored, the job passes, and
the calling workflow passes with it. The job-level form asks the caller to derive the same answer
across a `workflow_call` boundary, and the whole point of the change is that the caller must not go
red.

The ratchet keeps its voice. `coverage.jl` writes one `::error` annotation per offending file and a
rise table to the run summary, so a trip is legible on the run's own page. A further step states in
one line that the ratchet tripped and that the test check is green on purpose, and the Refresh
Artifact is published on `steps.ratchet.outcome == 'failure'` where it was published on `failure()`.

### Consequences of the amendment

- **A tripped ratchet no longer reds the test check**, and no longer reds anything. It is a warning
  annotation, a set of `::error` annotations, a rise table, and a Refresh Artifact. The reader of
  the test check sees the suite's verdict alone.
- **The verdict is advisory, not blocking.** The baseline is enforced by whoever reads the run, and
  a rise can reach `dev` unnoticed by a contributor who does not open the run. That cost is
  accepted, because the alternative on offer was a gate that does not run at all outside `main`.
  Turning the gate blocking again is the removal of one line.
- **The gate runs wherever the suite runs**, on `main`, on `dev*`, on `agents/*`, on a tag, and on
  every pull request that `TestOnPRs.yml` covers, with no dependency on which files are on the
  default branch.
- **The gate reads the same commit as the test job**, which on a pull request is the merge commit.
  This holds ADR 0075's rule for `Complexity.yml` rather than diverging from it.
- **The artifact belongs to the gate's own run**, so `actions/download-artifact` needs neither a
  `run-id` nor a token, and the workflow needs no scope above `contents: read`.

## Amendment (2026-08-27): a `COV_EXCL` pair is not an exemption

*A Coverage Exemption is keyed by definition, and states its count* above describes the exemption
mechanism as though it were the only one. It was not.
[Issue #552](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/552) found a second one
already in the tree. `CoverageTools` honours a `# COV_EXCL_START` / `# COV_EXCL_STOP` comment pair
when `julia-processcoverage` converts the test job's `.cov` files to `lcov.info`, and it drops every
line between the two from `lcov.info` altogether. Such a line is neither a hit nor a miss. The gate
reads `lcov.info` and reads nothing else, so it never sees the line at all.

Two files carried a pair. `src/13_Prior/04_HighOrderPrior.jl` hid 304 source lines holding
`duplication_matrix`, `elimination_matrix` and `summation_matrix`, and reported `misses = 0` on a
row whose `lines = 66` counted less than half of the file. `src/08_Moments/10_Histogram.jl` hid 113
source lines holding `mutual_variation_info`.

### The ruling

**`COV_EXCL` is not admitted. A Coverage Exemption is the one mechanism, and it is a row in
`code_health/rulings.toml`.**

The row carries four things this ADR asks of a claim that a line may stay uncovered: a key that
names the definition, an exact count held to equality in both directions, a Rationale block that
states why, and the maintainer's approval on that block. A `COV_EXCL` pair carries none of them. It
is unkeyed, uncounted, unexplained and unapproved, and it is invisible to `terminal`, so a file it
touches cannot be judged against the condition #404 sets. It is also strictly stronger than the
mechanism it evades: an exemption leaves the miss in the baseline and binds only whether the file is
terminal, whereas the pair removes the line from the measurement.

The two pairs were deleted, and the code they hid was covered rather than exempted.
`test_07_linear_algebra.jl::"Duplication, elimination and summation matrices"` reaches the three
structure matrices under both settings of `diag`, and `test_08_moments.jl` reaches
`mutual_variation_info` under both settings of `normalise`. Neither file needed a
`[[coverage_exemption]]` row, which is the outcome the ADR prefers: the sweep writes the test.

### The census

Check 4 of `test/test_49_coverage_attribution_census.jl` fails when any tracked `.jl` file under
`src/` or `ext/` holds the string `COV_EXCL`. The marker works only as a comment, so a plain text
search over the measured roots is the whole check, and it costs one pass over the corpus.

It is a test rather than a step of `coverage.jl` for one reason: the coverage gate is advisory by
the amendment above, and this rule is not. A pair that arrives in a pull request must turn something
red on the day it lands, because every later measurement of that file is wrong while it stands.

### Consequences of forbidding `COV_EXCL`

- **The two rows the pairs stood behind rise in `lines`.** The region was instrumented all along, so
  the counters were allocated and the numbers were only stripped on the way out.
  `src/13_Prior/04_HighOrderPrior.jl` goes from `lines = 66` to `lines = 170`, and
  `src/08_Moments/10_Histogram.jl` from `lines = 112` to `lines = 137`. `misses` does not rise for
  either file, because the code the pairs hid is covered.
- **`lines` cannot be measured outside CI, and it does not bind.** `code_health/coverage.jl` reads
  the `lcov.info` that only the test job writes. The rows were amended from a local `CoverageTools`
  measurement of the same source, and the first CI run either confirms them or publishes the
  Refresh Artifact that replaces them. This is the route the first baseline took, stated in
  *The first baseline is a provisional seed* above.
- **`src/08_Moments/10_Histogram.jl` falls from `misses = 1` to `misses = 0`.** Its row had never
  moved since the provisional seed, because a ratchet trips on a rise alone and a stale-high row
  therefore stands until a refresh runs. The local measurement finds no uncovered line in the file,
  and it is a strict subset of the suite that CI runs.
- **A file's coverage number now means what it says.** Before this ruling a row could read
  `misses = 0` over a third of a file, and nothing in the baseline, the rulings or the manifest
  recorded the gap.
