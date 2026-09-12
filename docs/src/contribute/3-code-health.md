<!-- markdownlint-disable MD046 -->
<!-- A Documenter admonition body is indented, and markdownlint reads it as an indented
     code block, so a page that mixes admonitions with fenced code can never satisfy
     MD046's consistency check. 2-developer.md has the same shape. -->

# [Code health: the maintenance loop](@id code_health)

!!! note "Developer documentation"

    If you haven't, please read the [Developer documentation](2-developer.md) first. This page
    assumes you can already build, test and format the package.

Three tools measure this repository continuously. **JET** looks for inference and correctness
problems, **CodeComplexity** measures cyclomatic complexity, cognitive complexity and argument
count, and **JuliaSyntax** counts the code lines in a file. Their numbers are recorded per file in
`code_health/`, and a CI gate turns red when a number rises.

This page is the procedure for making a number **fall**. It is what you follow when you pick up an
issue labelled `code-health` and sit down to spend a session on it.

## What the gate does, and what this loop does

The two halves are deliberately separate.

| | The gate | This loop |
| --- | --- | --- |
| Runs | on every push and pull request | when a person sits down |
| Asks | has a number risen? | can a number fall? |
| Outcome | red or green | a commit |
| Cannot | judge anything | be automated |

The gate is a **ratchet**. It compares each file's measured number against the number the
repository recorded, and it fails only on a rise. It is green the day it lands, and it stays green
however high a number already stands. See
`docs/adr/0076-the-code-health-pass-rule-is-a-ratchet.md`.

!!! note "The size ratchet reads its threshold, and the other three do not"

    `size.jl` counts the **code** lines in a file, where a docstring line, a comment line and a
    blank line are not code. A file's ceiling is the greater of the threshold in
    `code_health/rulings.toml` and the number the baseline records for it. So a file under 500 code
    lines is free to grow, and a file over 500 may fall and may not rise. A file that crosses 500
    trips on the crossing.

    A plain ratchet was rejected here. A line count moves on every added helper, so a plain ratchet
    would redden on ordinary work, and that is the noise issue #336 already measured. See
    `docs/adr/0101-the-size-gate-counts-code-lines-and-binds-over-a-threshold.md`.

Because the ratchet enforces no improvement, a **scheduled job** files the work. Every Monday it
looks for files above a threshold, ranks them, and tops the `code-health` label up to **five open
issues**. The cap is on the open count rather than on one run, so the queue paces itself to what
gets closed. It files each file once, so the queue is also finite. See
`docs/adr/0078-the-scheduled-job-files-a-file-once-and-refiles-it-on-a-rise.md`.

!!! tip "Your pull request went red and you have no `code-health` issue"

    That is the other direction, and it is covered by
    the section "When a code-health gate turns red" on the
    [Developer documentation](2-developer.md) page. Come back here when you want to lower a
    number rather than clear one.

## The priority order

When two changes compete for one session, this order decides.

 1. **Correctness.** A JET finding may be a real defect. Nothing outranks it.
 2. **Type stability.** *No command on this page measures it.* Measuring type stability needs
    `report_opt`, which needs a curated corpus of concrete calls rather than a package sweep, and
    that corpus does not exist yet. Use rung 2 as a tie-breaker by judgement, and see
    [issue #347](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/347).
 3. **Maintainability and performance**, together. Neither outranks the other.

Where full type stability is impossible, aim for the **narrowest inferable bound** rather than for
`Any`.

## The procedure

### 1. Take one issue

Assign yourself the issue. One issue is one **file**, and one session is one issue. The issue body
carries the path, every breached metric with its value, its baseline and its threshold, the worst
definitions by name and line, and the JET reports by kind and site.

### 2. Confirm the file is still an offender

Run the check that the issue's metric belongs to, from the repository root.

!!! note "A `check` answers green or red. It prints no per-file number."

    Every check is a **ratchet**, so it compares each file against its recorded number and reports
    only a rise. A green run is the normal state while an issue is open, and it tells you that
    nobody has made the file worse since the issue was filed. The numbers themselves come from the
    issue body. To re-measure one file, use `measure_file` below for a complexity number, and
    [Dismissing a JET report](@ref code_health_dismissal) for the JET reports.

```bash
julia --project=code_health code_health/complexity.jl check
julia --project=code_health code_health/expansion.jl check
julia --project=code_health code_health/size.jl check
julia --project=code_health code_health/jet.jl check
COVERAGE_LCOV=path/to/lcov.info julia --project=code_health code_health/coverage.jl check
```

The complexity check takes about 15 seconds, the Expansion Bound about 30 seconds and the size
ratchet about 5 seconds, so you can run any of the three as often as you like. **The JET check takes about 6 minutes 30 seconds and peaks at 2.6 GiB.**

The coverage check takes about 10 seconds, and it needs an `lcov.info` that it cannot make itself.
In CI the `coverage` job of `.github/workflows/ReusableTest.yml` downloads the one the test job of
the same run wrote, so the gate runs wherever the suite runs. The step that runs the ratchet carries
`continue-on-error: true`, so a ratchet that trips annotates the run and leaves the test check
green. Locally, run the suite with coverage on and process it, or download the `lcov`
artifact from any green run of `Test.yml` and point `COVERAGE_LCOV` at it. ADR 0082.

`coverage.jl` takes one more verb than the others. `terminal` answers #404's closing question
for a child map's own files, and names the definitions that still hold an uncovered line.

```bash
COVERAGE_LCOV=path/to/lcov.info julia --project=code_health code_health/coverage.jl terminal src/08_Moments/
```

!!! warning "There is no fast inner loop for JET"

    `report_package` analyses a whole package. It cannot be pointed at one file, so a JET session
    gets roughly one measurement every six minutes. Plan the session around that: read the reports
    once, make every change you believe in, and measure again. Do not measure after each edit.

For a fast complexity reading on one file while you refactor, measure the file directly.

`measure_file` returns a `FileMeasure` and prints nothing on its own, so display it. It then names
the file's total and every definition with its value and its line, in line order.

```bash
julia --project=code_health -e 'using CodeComplexity;
    display(measure_file(CyclomaticComplexity(), "src/path/to/File.jl"))'
```

### 3. Work JET first

Correctness outranks everything, and a correctness fix often simplifies the code, which lowers the
complexity number for free. So a file with both kinds of finding is worked in that order.

For each JET report on the file, reach one of two verdicts.

- **It is real.** Fix it. Reproduce the failure first, so you know what you fixed.
- **It is a false positive.** Dismiss it. See [Dismissing a JET report](@ref code_health_dismissal) below.

About 180 of the 315 reports measured at baseline belong to a single systematic noise class, so
expect most reports on a file to end in a Dismissal rather than a fix.

### 4. Work complexity second

Lower the number by ordinary refactoring. The file's number is the **maximum** over its definitions,
so only the worst definition moves it. Fix the worst one, remeasure, and repeat until the file's
number falls below the threshold or you reach a definition you judge irreducible.

If a definition cannot fall, see [Exempting a complexity number](@ref code_health_exemption) below.

!!! warning "When the worst definition is a macro, run the Expansion Bound too"

    A **Declaration Macro** is measured where it is declared, and a second number holds what it
    emits. The two move independently, so a refactor that lowers the declaration can raise the
    emission and trip `expansion.jl` while `complexity.jl` goes green. That is not a hypothetical
    shape: the first file the job filed, `src/02_Tools.jl`, takes both its cyclomatic and its
    cognitive maximum from `@forward_properties`, which the Expansion Bound also holds.

    Run both checks after every such refactor, and record any accepted rise in the Expansion Bound
    the same way. ADR 0072 owns the rule.

### 5. Close the issue with a commit

**No `code-health` issue closes without a commit.** An issue ends in exactly one of three states,
and all three are commits:

 1. the number fell,
 2. a Dismissal landed for a JET class, or
 3. an Exemption landed for a complexity definition.

If none of the three happened, leave the issue open. It keeps its slot in the queue, and a queue
that is visibly stuck is a better signal than a judgement that no file records.

Record any accepted rise in the same commit, and refresh the baseline.

```bash
julia --project=code_health code_health/complexity.jl refresh
```

A refresh **lowers by default**. Recording a rise needs the explicit flag, and it must be a
deliberate act.

```bash
julia --project=code_health code_health/complexity.jl refresh --accept-rise
```

## Run each check in a fresh process

Run every check in a **fresh `julia` process**, never in a working REPL, and never two checks in one
process.

The reason is measurable rather than theoretical. **JET's report count moves with the load set.**
The baseline measurement found 315 reports with the package alone and 312 once the three extension
trigger packages were loaded. A REPL that has done `using Plots` therefore measures a different
number from CI, and the difference looks like a change you made. `jet.jl` asserts that the recorded
load set is loaded before it measures, which catches the case where a package is missing, but it
cannot see extra packages you loaded an hour ago.

The same rule governs the doctests, for the same class of reason. See the `run-doctests` skill.

## [Dismissing a JET report](@id code_health_dismissal)

A Dismissal records that a report is not a real defect. It is keyed by the **Report Fingerprint**:
the attribution file, the report kind, and the report message. On a `BuiltinErrorReport` the
message carries the **builtin** as well: JET prints a bare constant for that kind, so without it
one Dismissal would cover every builtin error in the file. The fingerprint carries **no line number
and no stack trace**, so it survives an edit that moves the code. See
`docs/adr/0071-a-dismissed-jet-report-is-keyed-by-file-kind-and-message.md` and its amendment.

**Take the message from a live measurement, not from an issue body.** The scheduled job writes an
issue body once and never updates it, so an older issue can quote a message that an earlier
fingerprint rule produced. A Dismissal copied from such a body matches nothing. That is safe — the
reviewed count simply does not fall — but it wastes a review.

No verb of `jet.jl` prints a live report. `check` answers green or red, and its red names a file
and two numbers. Call the script's own `measure` instead, which is what the scheduled job does,
and print the reports of the file you are working on. It costs one full JET measurement.

```bash
julia --project=code_health -e '
    include("code_health/CodeHealth.jl")
    module Jet
    include(joinpath(pwd(), "code_health", "jet.jl"))
    end
    m = Jet.measure()
    for r in m.reviewed["src/02_Tools.jl"]
        println(r.run, "  ", r.kind, ": ", r.message)
    end'
```

Copy the `kind` and the `message` into the Dismissal exactly. The `run` and the line are context,
and neither belongs to the fingerprint. Bind the measurement to `m`, as above: `measure` runs the
whole analysis on every call, so a second call costs a second six minutes.

Every Dismissal cites a named **Rationale**: one paragraph explaining why that class of report is
not a defect. One Rationale serves the dozens of Dismissals a systematic class needs.

- **Anyone may add a Dismissal** that cites an existing, approved Rationale.
- **A new Rationale needs the maintainer.** CI flags a diff that adds one.

A Dismissal cannot quietly relax the gate. The reviewed count is `raw − matched dismissals`, so a
report that matches no Dismissal always turns CI red. The arithmetic forces the safe direction, and
no rule has to be enforced by hand.

## [Exempting a complexity number](@id code_health_exemption)

An Exemption records that a definition's number will never fall. It is keyed
`(path, definition, metric)` and it **cites an approved Rationale**, in the same shared namespace
the Dismissals use. It is dropped before the file's maximum is taken, so it binds **candidacy
only** — it keeps the file out of the scheduled job's pool, and it never changes the baseline or the
gate.

The rule is therefore the same one, for both escapes:

- **Anyone may add an Exemption** that cites an existing, approved Rationale.
- **A new Rationale needs the maintainer.** CI flags a diff that adds one.

An Exemption is also **the written form of a stalemate**. When you have worked a file and concluded
that the number cannot fall without making the code worse, the Exemption is how you say so. If an
approved Rationale already covers your case, you can close the issue yourself. If none does, write
the Rationale you need and hand it over — that is the one step a contributor without write access
cannot finish alone.

Write a Rationale for a reader who does not have your session in their head. Say what was tried, and
say what the reduction would cost. One Rationale then serves every Exemption of that shape: the
17 constructor Exemptions that ship before the job first runs all cite a single one.

## When a correctness fix raises a complexity number

This is the one place the two tools genuinely fight, and the priority order resolves it: **take the
correctness fix**.

The complexity ratchet will then trip. Record the rise with `refresh --accept-rise` in the same
commit, so the reviewer sees the correctness fix and the number it cost, together, in one diff.

The reverse case needs no rule. A complexity refactor that introduces a JET report turns the JET
gate red, so it simply cannot land.

The scheduled job notices the rise. It normally skips a file it has already filed, but it refiles a
file whose number now stands above the number it carried when its last issue closed, so the higher
number returns to the queue rather than becoming permanent.

## When to stop

**The loop terminates on its own.** The scheduled job files each file once, so the queue is finite
and then the job goes quiet. It stands at 81 files today: 31 above a complexity threshold and 72
carrying a JET report no Dismissal covers, with 22 in both groups. It shrinks as Dismissals land,
because a dismissed report stops making its file a candidate. You do not need to decide when the whole effort
is over. You only decide when *one* issue is over.

One issue is over when it reaches one of the three committed states in step 5. If the number could
not fall and you could not justify an Exemption, the issue stays open. That is not a failure state.
It is the queue telling you, and everyone else, that a file is genuinely hard.

## A worked example

The two halves come from two files. The JET half is `src/08_Moments/01_Base_Moments.jl`, and it has
already been through the loop. The complexity half is `src/20_Optimisation/10_JuMPOptimiser.jl`,
which is still a candidate.

### The JET half

The baseline measurement reported 315 raw findings across 68 files. Most belong to one systematic
noise class, so the honest estimate of real findings was 1 to 10. One was confirmed by hand:
`moment_window_and_weights` declared a keyword default that refers to itself.

```julia
function moment_window_and_weights(X::MatNum, w::Option{<:ObsWeights}, args...; dims = dims,
                                   kwargs...)
```

The default `dims` resolves to a module-level `dims` that does not exist, so any call that omitted
the keyword failed.

```julia-repl
julia> PortfolioOptimisers.moment_window_and_weights(rand(10, 3), nothing)
ERROR: UndefVarError: `dims` not defined in `PortfolioOptimisers`
```

Every sibling method of that function declares `dims::Int = 1`, so the fix was to match them, in
the signature and in the docstring's two signature blocks. `test/test_08_moments.jl` then gave 743
passes and 0 failures.

Two details of this case are worth carrying to the next one.

- **The verdict came from reading the code, not from JET's message.** JET said a variable was
  undefined. Whether that is a defect or a false positive is exactly the judgement the gate cannot
  make, and it is the reason this loop exists.
- **The line moved and the defect did not.** The measurement recorded the finding at line 872. By
  the time it was fixed it sat at line 878, with no change to that function. A verdict keyed on a
  line would already have been stale. That is why a Report Fingerprint carries no line.

### The complexity half

The clearest complexity case is in another file. `JuMPOptimiser`, in
`src/20_Optimisation/10_JuMPOptimiser.jl`, breaches all three metrics at once.

| metric | value | threshold | ratio |
| --- | --- | --- | --- |
| argument count | 44 | 10 | 4.4 |
| cyclomatic | 100 | 10 | **10.0** |
| cognitive | 96 | 15 | 6.4 |

At a ratio of 10.0 it ranks first among the complexity-only candidates, because the scheduled job
ranks by `max(value / threshold)`.

**The three metrics get two different verdicts, and that is the lesson.**

The **argument count** cannot fall. It is a struct's inner constructor, so Julia gives it one
argument per field and no other form. It is already covered by one of the 17 Exemptions that ship
before the job first runs, all citing a single shared Rationale.

The **cyclomatic and cognitive numbers can** fall. Those come from validation branches written
inside the constructor body, and a branch is extractable where an argument per field is not. So the
file stays a candidate, and the work is an ordinary refactor: move the validation out, remeasure,
repeat.

Read that pair carefully before you reach for an Exemption. **An Exemption is per metric, not per
definition.** The same definition can be irreducible on one metric and plainly reducible on
another, and exempting the whole definition would silence work that is genuinely available.

The general shape in this repository: at the frozen thresholds cyclomatic reports **22** of the 196
files, cognitive **22** and argument count **21**, **39** files breach at least one, and **31**
remain once the Exemptions drop. Of the 21 argument-count offenders, 11 are struct inner
constructors and are exempted up front, so argument count contributes 10 candidate files rather
than the 101 it contributed at the textbook threshold of 5.

Those counts move with the code. They are a shape to expect, not a number to check against.
