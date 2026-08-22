---
status: accepted
---

# The code-health pass rule is a ratchet against a committed baseline

## Context

[Issue #250](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/250) asks for two numbers
to fall over time: the JET report count and the CodeComplexity measurement. A gate that holds a
number needs a pass rule, and two rules are available. An **absolute threshold** fails a file whose
number stands above a fixed line. A **ratchet** fails a file whose number stands above the number
the repository last recorded for it.

Five ADRs already assume the ratchet. [ADR 0071](0071-a-dismissed-jet-report-is-keyed-by-file-kind-and-message.md)
keys a Dismissal so a verdict survives an edit. [ADR 0072](0072-the-complexity-gate-measures-src-and-ext.md)
fixes the scope the baseline covers. [ADR 0073](0073-the-code-health-baseline-is-four-toml-files.md)
sites the baseline files and opens with the sentence "the gate is a ratchet against a committed
baseline". [ADR 0074](0074-the-baseline-row-set-is-total-and-a-rename-pairs-by-measurement.md) makes
the row set total. [ADR 0075](0075-a-run-that-trips-publishes-the-refresh-artifact.md) hands a
contributor the file that clears a trip. None of the five **decides** the ratchet. This ADR records
the decision those five stand on.

Two measurements against the working tree decide it.

1. **A large minority of files in scope stand above a threshold today.** Measured per file as the
   maximum over definitions, of the 196 files under `src/` and `ext/`, each of the three metrics
   reports **21** files at its frozen threshold — cyclomatic 10, cognitive 15 and argument count
   10. **38** stand above at least one, and **30** remain once the 17 constructor Exemptions drop.
   Argument count's threshold was 5 while this map was charted, where it reported 101 files. The
   scheduled job's rules moved it to 10 on the distribution's own cliff.
2. **JET reports 315 raw findings over 68 `src/` files**, of which the honest estimate of real
   findings is 1 to 10. The rest are a systematic noise class.

An absolute threshold therefore cannot go green. It would need 30 files refactored and 315 reports
adjudicated **before** the gate could run at all. That inverts the order this effort chose: the
destination is the instrument, and turning the crank is the loop's work.

## Decision

### A file passes when its number has not risen

For every file in scope and every metric, the gate compares the measured number against the number
recorded for that file in the committed baseline.

```text
measured <= recorded  ->  green
measured >  recorded  ->  red
```

The comparison is per file, never per total, so a fall in one file cannot pay for a rise in
another. It is blind to a fall and a rise inside the **same** file, and that is accepted.

### The gate is green the day it lands

The baseline is generated from the tree as it stands. Every recorded number therefore equals the
measured number at the moment of generation, so the first run is green by construction. No file has
to be improved before the instrument works.

### The ratchet turns one way

A refresh lowers a number by default. Recording a **rise** needs the explicit `--accept-rise`
spelling, and the rise then lands as a reviewed line in a committed file. So a number can rise, but
only through a diff a person approved. [ADR 0075](0075-a-run-that-trips-publishes-the-refresh-artifact.md)
gives a contributor without write access the artifact that produces that diff.

### The thresholds drive candidacy, never the gate

The thresholds — cyclomatic 10, cognitive 15, argument count 10, and one reviewed-real JET report —
never appear in the pass rule. They select which files the scheduled job files as
work. A file far above every threshold is green while its number holds steady, and a file far below
every threshold is red the moment its number rises. The two mechanisms are separate on purpose:
**the ratchet stops decay, and the scheduled job drives improvement.**

## Consequences

- The gate never blocks a contributor on a number they did not cause. That is what makes it
  deployable in one commit against a codebase where 30 files breach a threshold.
- The gate enforces no improvement whatever. Improvement is a procedure a person follows, and it
  lives in [`docs/src/contribute/3-code-health.md`](../src/contribute/3-code-health.md).
- Every number in the baseline is a claim about a specific analyser at a specific version, so the
  provenance comparison of [ADR 0073](0073-the-code-health-baseline-is-four-toml-files.md) is not
  optional. A ratchet against a moving analyser measures nothing.
- The baseline is a permanent, growing file that every contributor's pull request can touch. Merge
  conflicts on a row are expected, and ADR 0073 resolves them by re-running the refresh.
