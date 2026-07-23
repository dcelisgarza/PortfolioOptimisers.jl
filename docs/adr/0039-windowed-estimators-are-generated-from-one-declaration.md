---
status: accepted
---

# Windowed moment estimators are generated from one declaration, not merged into one type

## Context

The five windowed moment estimators — `WindowedExpectedReturns`, `WindowedCovariance`,
`WindowedVariance`, `WindowedCoskewness`, `WindowedCokurtosis` — are the same idea five
times: wrap an inner moment estimator, restrict it to a sub-window of observations, rebind
observation weights to that window, delegate. The 2026-07-19 architecture review made them
its top candidate and proposed collapsing them into one parametric `Windowed{E}` whose trait
maps the inner estimator to the generic it answers, with "the five types becoming aliases or
vanishing".

**That merge is not available in Julia.** Each wrapper subtypes a different abstract
estimator, and the hierarchy is single-inheritance:

| Type | Supertype | constraint sites in `src/` |
| --- | --- | --- |
| `WindowedExpectedReturns` | `AbstractExpectedReturnsEstimator <: AbstractEstimator` | 50 |
| `WindowedCovariance` | `AbstractCovarianceEstimator <: StatsBase.CovarianceEstimator` | 115 |
| `WindowedVariance` | `AbstractVarianceEstimator <: AbstractCovarianceEstimator` | 55 |
| `WindowedCoskewness` | `CoskewnessEstimator <: AbstractEstimator` | 15 |
| `WindowedCokurtosis` | `CokurtosisEstimator <: AbstractEstimator` | 15 |

A struct's supertype cannot depend on a type parameter, and those supertypes are load-bearing
for dispatch at roughly 250 sites. Five nominal types must survive. This is recorded here so
the merge is not re-proposed.

The review's second claim also needed correcting. Its "~600 lines of interface" measured
681 lines across the five files, but those are **539 lines of docstring and 137 lines of
code**. The depth the review credited already existed — `windowed_preamble` has been the
shared implementation since the family was written. Collapsing the types would not delete
539 lines of documentation; it would merge five docstrings into one larger one.

What the duplication had actually cost was **synchronisation**. Before this change the five
had drifted in seven independent ways:

- `mean` was a named keyword on `cov`/`var`/`std` but rode in `kwargs...` for
  `cor`/`coskewness`/`cokurtosis` — where it leaked into `windowed_preamble` →
  `moment_window_and_weights` → `factory`. Harmless only because every one of those splats.
- Only `WindowedExpectedReturns` documented its `## Propagated parameters` /
  `## View parameters` sections, though all five carry identical `@fprop @vprop` / `@wprop` tags.
- `WindowedCokurtosis` hardcoded `"Cokurtosis estimator."` rather than a `field_dict` lookup.
- `WindowedVariance` documented its inner estimator with `field_dict[:ce]` — *"Covariance
  estimator"* — for a variance field.
- Return values were hand-written per file, ignoring `ret_dict` entirely.
- `WindowedVariance` named its field `ce` and `WindowedCokurtosis` named its `ke`, against
  the library convention (`ve`, `kte`) used by `KendallCovariance`,
  `ImpliedVolatilityRegression`, `DimensionReductionRegression` and `25_Aliases.jl`.
- `windowed_preamble`'s own docstring claimed `iv` is subset "when `window` is a vector of
  indices" and "otherwise returned unchanged". False: `get_window(::Integer, …)` resolves an
  `Int` window to a `UnitRange`, which *is* a `VecInt`, so an `Int` window subsets `iv` too.
  Only `window = nothing` — which resolves to a `Colon` — leaves it whole. Found by writing
  the test that asserted the documented behaviour and watching it fail.

Seven drifts in five files is the signal. None was caught by tests, because none changes a
number.

## Decision

### 1. One declaration per windowed estimator

`@windowed_estimator`, defined in
[01_Base_Moments.jl](../../src/08_Moments/01_Base_Moments.jl) beside its only collaborator
`windowed_preamble`, emits the whole family member from one block: the `@propagatable`
`@concrete` struct, both constructors with their validation, one forwarding method per
`forward` entry, the `export`, and every docstring.

```julia
@windowed_estimator WindowedVariance <: AbstractVarianceEstimator begin
    ve::AbstractVarianceEstimator = SimpleVariance()
    noun = "Variance"
    forward = [Statistics.var(::MatNum; mean) => :vararr,
               Statistics.var(::VecNum; mean) => :varnum,
               Statistics.std(::MatNum; mean) => :stdarr,
               Statistics.std(::VecNum; mean) => :stdnum]
    doctest = """…"""
end
```

The five source files survive, one declaration each, so `docs/src/api/08_Moments/{28..32}.md`
keep resolving and `grep WindowedCovariance src/` still lands in a file named after it.

### 2. Forwarded generics are declared as mini-signatures

Each `forward` entry is a signature paired with the `ret_dict` key(s) documenting its
returns. Naming `mean` in the signature is what emits it as a named keyword rather than
letting it ride in `kwargs...`. The alternative — a lookup table mapping generic to shape —
was rejected: it puts the knowledge somewhere other than the call site, which is the
non-locality this change removes.

### 3. Docstrings are generated as interpolation ASTs

The macro emits `Core.@doc Expr(:string, …) …`, not a pre-built `String`. This is
load-bearing: it keeps `$(DocStringExtensions.TYPEDEF)`, `$(FIELDS)` and the
`field_dict`/`arg_dict`/`ret_dict` lookups as live parts of the `DocStr`, exactly as a
hand-written docstring would. `@propagatable` already ends in `esc(quote … end)` and marks
its target with `Base.@__doc__`, so the docstring attaches through both macro layers.

A method's summary line names the **generic**, not the type's noun — `std` on a
`WindowedVariance` computes a standard deviation, not a variance.

### 4. Expansion-time validation, fail closed

Unknown block keys, malformed `forward` entries, and unknown `field_dict`/`ret_dict` keys
are rejected at macro-expansion time with a `did_you_mean` suggestion, in the style
established by ADRs [0025](0025-enumerated-parser-allowlist.md)/[0026](0026-lenient-constraint-names-with-suggestions.md)/[0027](0027-cap-equation-parser-recursion.md).
Codegen earns its keep only if a typo cannot silently produce a missing method or a blank
docstring.

The suggestions run under a **deliberately looser** scoped configuration than the global
default — Damerau-Levenshtein at `min_score = 0.5` rather than Levenshtein at `0.7`. The
strict default exists to stop near-miss probes echoing real *asset names* back to the caller
([ADR 0026](0026-lenient-constraint-names-with-suggestions.md)); that boundary does not apply
to a macro whose candidates are compile-time constants with nothing to leak. At the global
setting the suggestion is dead code for short keys: `nuon` scores 0.5 against `noun`, `muu`
scores 0.67 against `mu`, and neither would ever be offered. The override is scoped via
`with_string_distance`, so the global default is untouched.

### 5. All seven drifts closed; two fields renamed

The normalisations of the Context section are applied to all five. Two are breaking keyword
changes, taken as a clean break at 0.26:

- `WindowedVariance(; ce = …)` → `WindowedVariance(; ve = …)`
- `WindowedCokurtosis(; ke = …)` → `WindowedCokurtosis(; kte = …)`

`ret_dict` gains `:cskew` and `:cskewV` for `coskewness`, which returns two values.

### 6. The macro is internal

Not exported, not `public`, no api page. Its surface has had exactly five uses; committing
to it under semver would freeze a design on its first outing. Downstream packages wanting a
windowed variant of their own estimator write it by hand, as they must today.

## Consequences

**Adding a windowed variant of a sixth moment kind is one declaration**, and it cannot drift
from the other five — the shape, the docstring sections, the propagation tags and the
`mean`-keyword discipline all come from one place.

**This is the codebase's first type-generating macro.** `@propagatable` and
`@forward_properties` transform a struct the author wrote; this one writes the struct. That
is a genuine cost — `improve-maintainability` flags "opaque load-time codegen" as a
weakness, and the flag is fair. It is accepted here because the alternative was not "five
explicit files" but "five explicit files that had already drifted seven ways", and because
`@macroexpand1` on a declaration prints the exact code the file used to contain.

**Verification was a macroexpand-diff**, not a golden-value comparison: expanding each
declaration and comparing against the pre-change source proved the emitted struct,
constructors and method bodies identical modulo the agreed normalisations. That diff is what
caught the summary-line bug fixed in §3.

**The review's candidate 1 is now closed**, but not as it was written. Anyone re-reading it
should read this ADR first: the headline refactor was infeasible, the line count was
documentation, and the win was synchronisation.
