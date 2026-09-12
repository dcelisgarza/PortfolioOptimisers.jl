---
status: accepted
---

# The size gate counts code lines, and binds only over a threshold

## Context

A code-quality review of PR 625 reported four files that crossed 1000 total lines, raised none of
them, and left a suggestion:

> If a size gate is ever wanted, ratchet **code lines per file**, not total lines. The complexity
> ratchet already counts what matters per function. A total-line gate on this tree fires on
> documentation and stays silent on sprawl.

No rule about file size is written anywhere in this repository. The 1000-line rule the review
applied is the reviewer's own, and the review's finding is that the rule reads false here. This ADR
records the gate that reads true.

Four measurements over the 227 files of `src/` and `ext/` decide the shape. They were taken on
`dev` at `5a90c100`.

 1. **The tree is mostly prose.** Of 143042 lines, 73417 are docstring, 37950 are code, 29753 are
    blank and 1922 are comment. Docstring is **64.8 %** of every non-blank line.
 2. **A total-line gate at 1000 flags 40 files, and 25 of them carry under 500 code lines.**
    `src/19_RiskMeasures/01_Base_RiskMeasures.jl` is 2257 lines and 403 of them are code.
    `src/22_Preselection.jl`, one of the four the review named, is 1438 lines and 313 of them are
    code. The gate would fire on the documentation that [issue #404](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/404)
    asks for.
 3. **A code-line gate at 500 flags 15 files, and every one of them the total-line gate flags
    too.** The code-line set is the strict subset with the documentation removed. Nothing is lost
    by counting code, and 25 false readings go away.
 4. **The distribution has a cliff.** The median file carries 97 code lines and the 90th percentile
    carries 376. The largest gap in the region sits between 530 and 440, and 500 sits in it.

Four ADRs already fix the shape of a gate here, and this one takes all four rather than inventing a
fifth. [ADR 0073](0073-the-code-health-baseline-is-four-toml-files.md) splits the files on **who
writes them**. [ADR 0074](0074-the-baseline-row-set-is-total-and-a-rename-pairs-by-measurement.md)
makes the row set total and pairs a rename by equal measurement.
[ADR 0075](0075-a-run-that-trips-publishes-the-refresh-artifact.md) hands a contributor the Refresh
Artifact. [ADR 0076](0076-the-code-health-pass-rule-is-a-ratchet.md) makes the pass rule a ratchet
against a committed baseline.

## Decision

### The gate counts code lines, and a docstring line is not one

One row per file in scope, in `code_health/size_baseline.toml`:

```toml
"src/22_Preselection.jl" = { code = 313, doc = 761, comment = 0, blank = 364, total = 1438 }
```

`code` binds. The other four are context and never bind, on the same split `complexity.jl` draws
between a file's maximum and its sums.

A line is **code** when it carries one byte that is neither whitespace, nor inside a comment, nor
inside a docstring. The four kinds partition the file, so they always sum to `total`.

**The classification is a tokeniser and a parse, not a line scan.** A scan for `"""` and `#`
reads a long string literal inside a function as a docstring, and this library writes many. So
`code_health/size.jl` marks comment bytes from `JuliaSyntax.tokenize` and docstring bytes from the
`SyntaxNode` tree, and classifies a line from what is left. Like the complexity measurement it
loads no package under measurement, and it costs about five seconds.

**The parser is `JuliaSyntax.parseall(SyntaxNode, …)` rather than `Meta.parseall`, and a field
docstring is why.** Most of this library's prose is written as a field docstring, and a struct body
binds nothing for `Core.@doc` to attach to, so the `Expr` front end that `CodeHealth.isdocstring`
reads drops the wrapper and leaves a bare string literal. The `SyntaxNode` tree keeps the `doc`
node in both places, so one rule reads both shapes. A second rule for a bare string literal in a
block is not merely unnecessary. It is wrong: a string literal in any other block is a **value**,
and reading one as prose undercounted `src/01_Base/09_ObservationWeights.jl` by two lines.
`test/test_52_size_classification_census.jl` holds every case.

### A file's ceiling is the greater of the threshold and its recorded number

```text
ceiling = max(threshold, recorded code)
measured <= ceiling  ->  green
measured >  ceiling  ->  red
```

One sentence says two things.

- **A file under the threshold is free.** Its `code` is context, and ordinary work moves it without
  turning the gate red.
- **A file over the threshold is held where it stands.** It may fall and it may not rise, so the
  largest files in the tree cannot grow further while no one is looking.

A file that **crosses** the threshold trips on the crossing, which is the moment to ask whether the
file wants splitting. A file added to the tree has no recorded number, so it takes the threshold,
and an added file over 500 code lines is refused. That is ADR 0074's entry test, falling out of the
one rule rather than needing a second mechanism.

### The threshold is 500, and it is in the pass rule on purpose

`code_health/rulings.toml` gains a `[size]` section holding one number.

```toml
[size]
code_lines = 500
```

It sits in a section of its own rather than in `[thresholds]`, because
[ADR 0076](0076-the-code-health-pass-rule-is-a-ratchet.md) rules that a `[thresholds]` number
selects the scheduled job's work and never reaches a pass rule. This number does reach the pass
rule, and the split of sections says so.

**A plain ratchet on code lines was rejected**, although it is what the other three gates do.
Issue #336 already measured the failure: a sum-shaped number moves on every added helper, the gate
goes red on ordinary work, and a noisy gate gets switched off. Cyclomatic complexity escapes this
because a file's **maximum** is quiet under an added helper of complexity 2. A line count has no
such maximum to hide behind. The threshold is what makes the gate quiet, so the threshold has to be
in the rule.

**500 was chosen once, from the cliff**, in the same way and for the same reason as the argument
count threshold of issue #356. It is not a rule to re-derive, because a re-derived threshold never
terminates.

### The gate is green the day it lands, and 15 files enter held

The baseline is generated from the tree as it stands, so the first check is green by construction.
The 15 files already over 500 code lines are seeded with `refresh --accept-rise`, because seeding
them is a rise against the threshold and a rise is a named act. The largest are
`ext/PortfolioOptimisersPlotsExt.jl` at 1963 and `src/01_Base/01_DocstringDictionaries.jl` at 1062.

### It rides in the Complexity workflow

`code_health/size.jl check` runs as a third step of `.github/workflows/Complexity.yml`, after
`complexity.jl` and `expansion.jl`. It is a pure JuliaSyntax parser like the complexity
measurement, it reads the same tree through the same scope rule, and it adds about five seconds.
A workflow of its own would pay the checkout and the instantiate a second time for that.
[ADR 0077](0077-the-code-health-gate-is-two-workflows-pinned-by-a-manifest.md) splits the gate on
**cost**, and this step is on the cheap side of that split.

## Consequences

- The library gains its first written rule about file size, and it is measured rather than
  asserted. A file may hold 3000 lines of docstring and stay green.
- 15 files are now frozen at their code-line count. Work on one of them either leaves the count
  alone, lowers it, or records the rise in a reviewed diff.
- The gate says nothing about the 25 files that a total-line rule would flag. That is the decision,
  not an omission: documentation is what issue #404 asks for, and a gate that fights it is a gate
  that gets switched off.
- No scheduled job files size work. The ratchet stops decay, and nothing here drives improvement.
  A file over the threshold is a candidate for a split, and a split is a person's judgement.
- The classification depends on JuliaSyntax, so `julia_syntax` is a provenance field of the
  baseline. A ratchet against a moving tokeniser measures nothing.
