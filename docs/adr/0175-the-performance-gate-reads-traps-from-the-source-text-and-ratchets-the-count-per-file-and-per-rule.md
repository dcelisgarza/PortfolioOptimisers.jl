---
status: accepted
---

# The performance gate reads traps from the source text, and ratchets the count per file and per rule

## Context

The code-health gate measures complexity, size, JET error reports and coverage. Nothing measures
cost. The maintainer asked for a performance review in the shape of the JET check: find the traps,
keep the ones that are not traps out of the count with a written reason, and stop the count from
rising. The traps named were an index where a view reads, a broadcast that is not needed, an
intermediate array in a chain of linear algebra, a computation made twice, and an allocation inside
a loop.

A trap is only worth a gate when it has a replacement with the **same meaning**. A view of
`X[:, idx]` with a vector `idx` is not faster than the copy, because BLAS cannot stride it. A view
of `x[1:3]` passed to `sort!` changes the parent. `copy(w)` written twice is often two arrays that
are mutated apart. A rule that flags these shapes asks for a change that is slower or wrong, and a
gate that asks for wrong changes gets switched off.

Five measurements, taken on `dev` at `1ade09e2e9` over the 324 files of `src/` and `ext/`, decide
the shape.

 1. **The replacements pay.** On one thread, `sum(abs.(x))` over 10 000 entries allocates 80 104
    bytes in 6.5 µs, and `sum(abs, x)` allocates 16 in 1.8 µs. `tr(A * B)` at 200 × 200 allocates
    320 104 bytes in 751 µs, and `dot(A', B)` allocates 32 in 39 µs with the same answer. Summing
    50 column copies of a 2000 × 50 matrix costs 805 984 bytes and 126 µs, and the views cost 4 784
    and 36 µs. `x .* x + x` costs 160 176 bytes and 11.3 µs, and `x .* x .+ x` costs 80 136 and
    3.9 µs.
 2. **A first rule set was noisy, and the noise had four causes.** The first pass flagged 191
    sites. A hand review found the declaration `lw[1:N] >= 0` inside JuMP's `@variables`, the
    scalar `inv(pi)` and `inv(alpha)`, a call repeated in the two arms of one ternary, a call inside
    the message of a `DomainError`, and `copy(w)` repeated on purpose. Each cause is now a rule
    exclusion with a test.
 3. **A broadcast rule that reads inside a broadcast cannot be precise.** `f.(w - v)` is a trap and
    `(1 - alpha) .* u` is not, and the parser cannot tell `w - v` from `1 - alpha`. That rule
    flagged 99 sites, most of them the scalar form. The rule now reads a plain `+` or `-` whose
    operand is a broadcast, which is always an array, and flags 13.
 4. **A loop rule must report a site once, and must know growth.** Its first pass reported
    `vcat(wb, [fill(a, s) fill(b, s)])` three times, once per allocating call, and read
    `wb = if … vcat(wb, …) … end` as one more buffer. The second is worse than a buffer: it copies
    the whole array on every iteration, which is O(n²) over the loop. The rule now reports the
    outer allocation alone and names growth with its own replacement: 18 of its 78 Findings are
    growth, 52 are a buffer that changes with the iteration, and 8 are the same array every time.
 5. **The tree holds 222 Findings under the seven rules**: 78 `loop_allocation`, 69 `slice_copy`,
    45 `reduce_temporary`, 14 `linalg_temporary`, 13 `unfused_broadcast`, 2 `repeated_call` and 1
    `search_temporary`, over 71 files. A scan costs about 7 seconds, most of it compilation.

## Decision

### Seven rules, each with a replacement of the same meaning

`code_health/perf.jl` parses each file with `Meta.parseall` and applies seven rules. Each rule's
docstring names the shapes it flags, the near shapes it leaves alone, and why.

| rule | flags | replacement |
| --- | --- | --- |
| `slice_copy` | a strided slice, `A[:, j]` or `x[a:b]`, passed to a function that does not mutate and is not a constructor | `@view` or `@views` |
| `reduce_temporary` | a reduction over a broadcast or an array comprehension | `sum(f, x)`, a generator, `dot` |
| `unfused_broadcast` | a plain `+` or `-` with a broadcast operand | `.+` or `.-` |
| `search_temporary` | `length(findall(m))`, `sort(x)[1]`, `sortperm(x)[1:k]`, `for i in collect(r)` | `count`, `minimum`, `partialsortperm`, the range |
| `linalg_temporary` | `inv(A) * b`, `diagm(v) * A`, `tr(A * B)`, `diag(A * B)`, `x' * A * y` with or without parentheses, `(A * B) * x` | `A \ b`, `Diagonal`, `dot`, `dot(x, A, y)`, `A * B * x` |
| `repeated_call` | one scalar reduction or factorisation made twice in one definition, on inputs it never mutates, where both calls can run | a local name |
| `loop_allocation` | an allocating call or an array comprehension inside a `for` or `while` body, that the loop does not keep | one buffer before the loop, the array hoisted, or `push!`/`append!` for growth |

**A slice is strided only when every index is a colon, a range, a literal, or a loop variable over
a range.** `X[:, idx]` is silent, because `idx` could be a vector.

**A cold path is not read.** String interpolation, `throw`, `error`, a call to a constructor whose
name ends in `Error`, and a logging macro run once, on the call that fails or logs.

**An allocation the loop keeps is its output.** `push!(out, zeros(n))` and `out[i] = copy(w)` are
not flagged, and a comprehension outside a `for` or `while` is the output of its own loop. A
`copy(x)` bound with a plain `=` inside a loop may still be kept, as `best = copy(x)` is, and the
parser cannot tell: that one takes a Dismissal.

**A replacement can move a result by an ulp.** `dot` and `sum` add in different orders. The
replacement is correct, and a test that pins the old last digit is the one to widen.

### `scan` is the review, and `check` is the gate

`perf.jl scan [files…] [--rule NAME]` prints every Finding with its file, line, definition, code
and replacement, and writes nothing. It is what a contributor runs on the files of a change, and
what a sweep ticket runs on its file.

`check` and `refresh` follow the other gates without change:
[ADR 0073](0073-the-code-health-baseline-is-four-toml-files.md) splits the generated baseline
`code_health/perf_baseline.toml` from the hand-written `code_health/rulings.toml`,
[ADR 0074](0074-the-baseline-row-set-is-total-and-a-rename-pairs-by-measurement.md) makes the row
set total and pairs a rename by equal counts,
[ADR 0075](0075-a-run-that-trips-publishes-the-refresh-artifact.md) publishes the Refresh Artifact,
and [ADR 0076](0076-the-code-health-pass-rule-is-a-ratchet.md) makes the pass rule a ratchet.

### The count per file and per rule binds, and a Dismissal subtracts from it

One row per file, one column per rule, and a `raw` column for context:

```toml
"src/13_Fees.jl" = { slice_copy = 0, reduce_temporary = 1, unfused_broadcast = 0, search_temporary = 0, linalg_temporary = 0, repeated_call = 0, loop_allocation = 0, raw = 1 }
```

[ADR 0101](0101-the-size-gate-counts-code-lines-and-binds-over-a-threshold.md) rejected a
sum-shaped ratchet for size, because every added helper moves a line count. A Finding is not a
size. It is a flagged shape, like a JET report, and ordinary work adds one only when it writes a
trap. So the count binds as JET's reviewed count binds, per file and per rule, and an added file
enters at zero under `perf_reviewed = 1` in `[thresholds]`.

A Finding that is not a trap takes a `[[perf_dismissal]]` in `code_health/rulings.toml`, in one of
two widths:

```toml
# One Finding: the four keys are copied from the line `scan` prints.
[[perf_dismissal]]
file = "src/09_ConstraintGeneration/02_LinearConstraintGeneration.jl"
rule = "slice_copy"
definition = "parse_equation"
code = "ops2[2:end]"
rationale = "a-slice-of-a-string-feeds-a-string-api"

# Every Finding of one rule in one definition: no `code`.
[[perf_dismissal]]
file = "src/08_Phylogeny/06_DBHT/04_CliqueHierarchy.jl"
rule = "loop_allocation"
definition = "clique3"
rationale = "a-graph-walk-grows-its-frontier"
```

The first is the Performance Fingerprint `(file, rule, definition, code)`, which carries no line,
so it survives an edit above it, and whose `code` is the printed `Expr` on one line, so it survives
a reformat. The second is the key of a complexity Exemption, `(path, definition, metric)`: a
definition whose loop must allocate on every iteration takes one entry rather than one per call.
Both cite a Rationale as the other rulings do, and a new Rationale needs the maintainer. The two
Rationales above are examples that no ruling has approved.

No Dismissal ships with this ADR. The 222 Findings the tree holds are recorded in the baseline, as
the first JET baseline recorded its reports, and
[issue #1312](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1312) carries
them to be worked or dismissed.

**A change to a rule raises `RULES_VERSION`.** The version is in the provenance, so a baseline
written under other rules is refused rather than compared, and the commit that changes a rule
refreshes the baseline.

### It rides in the Complexity workflow

The script loads no package under measurement, so it joins the size ratchet's step list in
`.github/workflows/Complexity.yml` rather than JET's workflow.
`test/test_73_performance_trap_census.jl` drives each rule over a fixture tree.

### The scheduled job files a file with a Finding

[ADR 0078](0078-the-scheduled-job-files-a-file-once-and-refiles-it-on-a-rise.md)'s job includes
`perf.jl` beside `complexity.jl` and `jet.jl`. A file with `perf_reviewed` or more Findings that no
Dismissal covers is a candidate, its issue lists the Findings with their replacements, and the
refile clause reads `perf:<rule>` from the committed baseline as it reads `jet:<run>`.

A Finding carries no ratio, because it is a count of sites and not a distance past a threshold. So
the ranking keeps its two keys: a reviewed JET report first, then the complexity excess ratio. A
file with Findings alone has a ratio of at most 1, and ranks after every file that breaches a
complexity threshold. The queue holds five open issues, so performance work reaches it as the
complexity work drains, and the backlog of the day this landed is carried by issue #1312 rather
than by 71 filings.

### A swept file carries no performance trap

The sweep of [#404](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/404) gains a fifth
condition: `perf.jl scan <file>` lists no Finding that a Dismissal does not cover. #404, its
thirteen child maps and the sub-issue that `code_health/sweep_triage.jl` writes state five
conditions. The pass runs after condition 2, because a replacement changes code and condition 2 is
what checks code against its statement, and before the `/unslop` pass.

The rows that read `swept = true` keep the flag, as [ADR 0169](0169-the-docstring-standard-gains-the-unslop-pass-and-a-swept-file-owes-it.md)
chose for the fourth condition. Unlike that condition, no closed ticket is reopened: every Finding
of the day this landed, in swept and unswept files alike, is listed in issue #1312, and a
Finding is a site with a named replacement rather than a pass over prose, so one ticket can work
them without the context of the ticket that swept the file.

## Considered options

**A dynamic layer, measured rather than guessed.** A parser sees neither a type nor a cost. The
largest performance traps in Julia are type instability, runtime dispatch and a boxed captured
variable, and none of them is visible in the text. JET's optimisation analysis, `report_opt`, reads
them from inferred code, and `@allocated` over a fixed workload reads the bytes a call allocates.
Both are deterministic for a pinned Julia and a pinned Manifest, where a timing is not. A gate on
them would ratchet two numbers per workload entry: the count of `report_opt` reports, attributed to
a file as `jet.jl` attributes, and the allocated bytes. The cost is a workload to curate and a
load of the package, as `jet.jl` pays. It is the better gate for cost that the text does not show,
and this ADR does not decide it, because the workload is the maintainer's choice.
[Issue #347](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/347) already holds that
problem as a corpus for `report_opt`; the allocated bytes are the second number the same corpus
can hold. The two layers do not compete: the text rules name a site and a replacement, and the
workload measures what the text cannot see.

A probe on the pinned environment, Julia 1.13.0 and JET 0.12.1, over a 500 × 20 returns matrix,
shows where that layer stands today:

| call | allocated bytes | `report_opt` |
| --- | --- | --- |
| `cov(PortfolioOptimisersCovariance(), X)` | 87 496 | 0 reports |
| `prior(EmpiricalPrior(), rd)` | 89 720 | 0 reports |
| `optimise(InverseVolatility(), rd)` | 94 024 | crashes inside JET |
| `optimise(HierarchicalRiskParity(), rd)` | 321 864 | crashes inside JET |

The crash is JET's own: `f Base.BottomRF{typeof(+)}(+) with type Base.BottomRF{typeof(+)} not
supported`, thrown while the optimisation analyser builds a runtime-dispatch report. So the count
of `report_opt` reports cannot be held for an optimiser until JET reads it, and the allocated bytes
can be held now.

**Timing in CI** was rejected. A shared runner's timing varies by more than most of these gains.

**A ratio for a performance Finding** was rejected. Any scale that puts a count of sites beside a
complexity ratio is a number chosen without a measurement behind it, and it would decide which of
two kinds of work the queue takes first.

**Reopening the swept tickets for the fifth condition**, as ADR 0169 did for the fourth, was
rejected. One issue lists every Finding with its replacement, and a Finding needs no knowledge of
the ticket that swept its file.
