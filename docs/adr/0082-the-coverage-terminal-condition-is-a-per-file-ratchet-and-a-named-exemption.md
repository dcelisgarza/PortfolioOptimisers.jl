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

## Amendment (2026-08-26)

### The gate is its own workflow, not a job of `ReusableTest.yml`

*The gate runs in its own job, fed by an artifact* above put the `coverage` job inside
`ReusableTest.yml`. A job of a reusable workflow is a job of whichever caller ran it, so a tripped
ratchet turned `Test` and `Test on PRs` red. That conflates two claims. The test check answers
whether the library works. The ratchet answers whether one file's miss count rose, which is a
question about the sweep's progress and never about whether the code runs.

The ratchet now lives in **`.github/workflows/Coverage.yml`**, on `workflow_run`:

```yaml
on:
  workflow_run:
    workflows: ["Test", "Test on PRs"]
    types: [completed]
```

`ReusableTest.yml` keeps the `lcov` artifact upload and nothing else. What made the hand-over worth
doing survives unchanged: the test job stays on the floating release of
[ADR 0056](0056-the-julia-setup-action-floats-on-latest.md), the gate stays on the pinned one of
[ADR 0077](0077-the-code-health-gate-is-two-workflows-pinned-by-a-manifest.md), and both callers
still get one gate from one file.

Four things follow from the trigger, and each replaces a mechanism the job form had for free.

- **`if: github.event.workflow_run.conclusion == 'success'`** replaces `needs: test`. A suite that
  failed or was cancelled publishes partial coverage or none, and a ratchet fed partial data
  reports a fall that is not one.
- **The download names the other run.** `actions/download-artifact` takes a `run-id` and a
  `github-token`, and the workflow takes `actions: read` so that they work.
- **The checkout names the commit and the repository.** A `workflow_run` job checks out the default
  branch unless it is told otherwise. `coverage.jl` reads its file set with `git ls-files`, so it
  needs the tested tree as a git checkout and an unpacked artifact would not serve.
- **The gate reads the head commit where the job read the merge commit.** On a pull request
  `workflow_run.head_sha` is the branch's own tip, while the `lcov.info` it is compared against was
  measured on the merge commit. The baseline that must bind an author is the one the author
  committed, so the head commit is the right file to read here, but this does diverge from
  [ADR 0075](0075-a-run-that-trips-publishes-the-refresh-artifact.md)'s rule for `Complexity.yml`.

### Consequences of the amendment

- **A tripped ratchet no longer reds the test check.** It reds `Coverage`, which is its own check
  with its own history, and a reader of `Test` sees the suite's verdict alone.
- **GitHub fires `workflow_run` only for a workflow file on the default branch.** The gate is
  silent on every branch until `Coverage.yml` reaches `main`, and a later edit to it is inert on
  `dev` until that edit is merged. `CodeHealth.yml` carries the same constraint for its schedule.
- **The verdict arrives after the suite rather than beside it**, and GitHub attaches a
  `workflow_run` check to the head commit rather than to a pull request's merge box. A reviewer
  reads it in the Actions tab or in the commit's checks, and whoever configures branch protection
  later must reach it by that name.
- **The gate runs the checked-out tree's own `coverage.jl`.** On a fork pull request that is code
  the fork wrote, run under the base repository's token. The workflow therefore takes
  `contents: read` and `actions: read`, reads no secret, and sets `persist-credentials: false`. Any
  scope or secret added to it would be a write capability handed to a fork.
