---
status: accepted — single-argument functor `r(x::VecNum)` as the precomputed-returns contract, gated by `supports_precomputed_returns` (amended, see Corrections)
---

# Feed already-reduced return series through the functor `r(x::VecNum)`, not a `SingletonVector` sentinel

## Context

Three internal call sites need the expected risk (or a moment) of a return series they
*already hold*, with no weights to apply:

- **`get_pr_value`** (entropy pooling) — the skewness / kurtosis of one asset's column,
  `view(pr.X, :, i)`;
- **the seven portfolio-level plot functions** (`plot_portfolio_cumulative_returns`,
  `plot_drawdowns`, `plot_histogram`, `plot_rolling_measure`, `plot_benchmark`,
  `plot_performance_summary`, `plot_rolling_drawdowns`) applied to a `PredictionResult`,
  whose `rd.X` is the portfolio series.

But the consuming interface is shaped for *weights + an asset matrix*: risk evaluation goes
through `calc_net_returns(w, X, fees) = X * w`, and the moment functors are
`r(w, X, fees)`, reducing via the same product. To satisfy that interface with an
already-reduced series, the code wrapped the series as a one-column matrix
`reshape(x, :, 1)` and passed a bespoke weights stand-in:

```julia
struct SingletonVector{T} <: AbstractVector{T} end          # length 1, [1] == 1
Base.:*(M::Matrix, ::SingletonVector) = dropdims(M; dims = 2)
```

so that `X * SingletonVector()` short-circuits to the single column. Problems:

1. **It is a fake weight vector that means "no weights."** A reader meets a sentinel type
   whose entire purpose is to make a matrix-vector product disappear, and must reverse-engineer
   that `(SingletonVector, reshape(x, :, 1))` spells "x is already the portfolio series."
2. **The optimisation it exists for barely fires.** `Base.:*(M::Matrix, …)` only matches a
   concrete `Matrix`. At the entropy-pooling sites the argument is
   `reshape(view(pr.X, :, i), :, 1)` — a `ReshapedArray`, not a `Matrix` — so the product
   falls through to the generic `getindex`-based matvec anyway. The sentinel saves nothing there.
3. **It silently does the wrong thing for measures that genuinely need weights.** Routed
   through `expected_risk`, a `WeightsInput` measure (`Turnover`, `EqualRisk`) receives
   `r(SingletonVector())` — turnover of `[1]`, a meaningless number with no error. A moment
   measure carrying a per-asset target `mu::VecNum` hits `dot(SingletonVector, r.mu)` and throws
   an opaque length-mismatch. The "precomputed series" path was only ever well-defined for
   measures whose result is a function of the series alone.

## Decision

Treat the **single-argument functor `r(x::VecNum)` as the canonical "expected risk of the
net-return series `x`."** This is not a new convention: `expected_risk` already *defines*
the `NetReturnsInput` case as `r(calc_net_returns(w, X, fees))` (ADR 0006), i.e. the functor
applied to a return vector. We make that the universal contract and complete it where it was
missing.

- **`NetReturnsInput` measures** (quantile / drawdown families) already have `r(x::VecNum)` —
  nothing to do.
- **The moment family** (`LowOrderMoment`, `HighOrderMoment`, `Skewness`, `Kurtosis`,
  `MedianAbsoluteDeviation`, `ThirdCentralMoment` — `WeightsReturnsFeesInput`) gains a
  single-argument arity. Each functor's `(w, X, fees)` dependence lives *entirely* in its
  first line, `dev = calc_deviations_vec(r, w, X, fees)`; everything after operates on `dev`.
  We extract that post-deviation math into a shared `moment_risk(r, dev)` kernel, reuse the
  existing weight-independent `calc_moment_target(r, ::Any, x)` methods via a new
  `calc_deviations_vec(r, x::VecNum)`, and add one generic `r(x::VecNum)` arity per measure
  that forwards `moment_risk(r, calc_deviations_vec(r, x))`. `VarianceSkewKurtosis` is
  *excluded*: although it is `WeightsReturnsFeesInput`, its variance component `r.vr(w)` is a
  weights-only `w'Σw` with no return-series form, so it cannot be evaluated on a bare series
  and falls to the fallback below.
- **The boundary is principled, not incidental.** `r(x::VecNum)` exists exactly when the
  measure's result is a function of the series alone — i.e. its target is weight-independent.
  A measure carrying a per-asset `mu::VecNum`/`VecScalar` (target `dot(w, mu)`) has *no*
  single-vector method, because reducing a per-asset target needs the weights the series no
  longer carries. This matches what was reachable before: `dot(SingletonVector, mu_vec)`
  already errored.
- **`WeightsInput` measures get no single-vector form**, by design — "risk of a bare return
  series" is undefined for a weights-only measure.
- **Eligibility is gated, not left to the bare functor.** *(amended — see Correction; the
  original wording of this bullet is preserved there.)* The contract cannot be enforced on the
  functor alone: a `WeightsInput` measure's own `r(w)` *is* `r(::VecNum)`, so dispatch cannot
  tell a weight vector from a return series, and a catch-all `(::AbstractBaseRiskMeasure)(::VecNum)`
  is shadowed by it (and, for the moment family, by the generic `r(x)`). Instead a predicate
  `supports_precomputed_returns(r)` — `true` for `NetReturnsInput` measures and the
  weight-independent-target moment family, `false` for `WeightsInput`/tracking/variance-carrying
  composites — is consulted by a single internal contract entry `expected_risk_from_returns(r, x)`,
  which throws an explanatory `ArgumentError` for an ineligible measure *before* ever calling the
  functor. The call sites that hold a precomputed series (cross-validation prediction scoring)
  route through this entry. The bare `(::AbstractBaseRiskMeasure)(::VecNum)` survives only as a
  backstop for measures with no `VecNum` functor at all (a variance-carrying composite such as
  `VarianceSkewKurtosis`).

The call sites then state their intent directly:

```julia
# get_pr_value
Skewness()(view(pr.X, :, i))
HighOrderMoment(; alg = StandardisedHighOrderMoment(; alg = FourthMoment()))(view(pr.X, :, i))
```

**Plotting is unified onto the same idea rather than left on a literal `[1]`.** Each
portfolio-plot `(w, X, fees)` method computes `ret = calc_net_returns(w, X, fees)` and then
operates only on `ret`; we factor everything past that line into a `_plot_*_from_returns(ret; …)`
helper. The `(w, X, fees)` method computes `ret` and delegates; the `PredictionResult`
wrappers pass `rd.X` straight to the helper. **`_pred_rd_to_matrix` is deleted** — its only
job was the `(SingletonVector, reshape(x, :, 1))` disguise, and the multi-period `VecVecNum`
case becomes natural (`ret::VecVecNum` is "several series," which the existing multi-portfolio
branches already handle).

`SingletonVector`, its four `Base` methods (`length`, `getindex`, `size`, `*`), and its unit
test are removed.

## Considered options

- **A. Keep `SingletonVector`.** Rejected — it is the sentinel-readability and
  silent-wrong-answer problem itself, for an optimisation that often does not even fire.
- **B. Swap the sentinel for a literal one-element weight vector** (`r([one(T)], reshape(x, :, 1))`).
  Rejected for the risk path — it deletes the *type* but keeps the "disguise a series as a
  one-asset matrix and multiply by 1" dance, trading one stand-in for a smaller one.
- **C. A dedicated named entry** (`expected_risk_from_returns(r, x)`). Rejected — it adds API
  surface for something `expected_risk` already expresses: the functor on a return vector *is*
  the entry for `NetReturnsInput` measures. A second name would diverge from that.
- **D. Overload `expected_risk(r, x::VecNum)` to mean "x is returns."** Rejected — collides
  with the existing `expected_risk(r, w::VecNum, args...)`, whose first `VecNum` is *weights*.
  A lone vector cannot mean both.
- **E. Single-argument functor `r(x::VecNum)` as the contract.** Adopted. It is the smallest,
  most honest design: it is what `expected_risk` already means for net-returns measures,
  removes both `SingletonVector` and the `reshape` workaround, and turns the previously silent
  weights-needed case into an explicit error.

## Consequences

- **Intent is stated directly.** `Skewness()(col)` reads as "skewness of this series"; the
  `(SingletonVector, reshape(col, :, 1))` puzzle is gone from every call site.
- **Plotting loses a helper and gains symmetry.** `_pred_rd_to_matrix` is deleted; each plot
  function has a series-first core reused by both the `(w, X, fees)` method and the prediction
  wrapper. Net plotting code shrinks.
- **A silent wrong answer becomes a loud error — at the contract entry.** Scoring a prediction
  through `expected_risk_from_returns` with a `WeightsInput` measure, a moment measure with a
  per-asset `mu`, or a variance-carrying composite now throws an actionable `ArgumentError` via
  the `supports_precomputed_returns` gate, instead of returning nonsense (turnover of the series
  read as weights) or an opaque `MethodError`. The guarantee lives at the gate, not the bare
  functor: `EqualRisk()(v)` remains a legitimate `r(weights)` call and cannot be told
  apart from a misuse by dispatch.
- **The boundary is documented by construction.** Which measures support a bare return series
  is now a property visible at each measure's functor definition, not an emergent consequence
  of a sentinel's arithmetic.
- **Extends ADR 0006.** Same dispatch surface; this closes the "I already hold the reduced
  series" case that the input-kind trait left implicit, using the trait's own
  `NetReturnsInput` shape (`r(::VecNum)`) as the model for the whole moment family.

## Correction — gated eligibility (amended in place)

The original Decision delivered the explanatory error through the bare-functor catch-all:

> **An explanatory fallback replaces the bare `MethodError`.** A catch-all
> `(::AbstractBaseRiskMeasure)(::VecNum)` throws an `ArgumentError` naming the measure and
> stating it cannot be evaluated on a precomputed return series (it needs weights / per-asset
> data), turning the contract boundary into a discoverable rule.

That holds only for measures with **no** `VecNum` functor at all (e.g. `VarianceSkewKurtosis`).
It is **not** true for the two cases the ADR explicitly named as now-loud:

- a `WeightsInput` measure's functor `r(w)` shares the `r(::VecNum)` signature, so it shadows
  the catch-all and **silently scores the series as weights** (`EqualRisk()(x)` → `1/N`,
  `TurnoverRiskMeasure()(x)` → turnover-of-series), or throws an opaque `DimensionMismatch`
  (`Variance`/`StandardDeviation`);
- a moment measure with a per-asset `mu` is shadowed by the moment family's generic `r(x)` and
  dies on a raw `MethodError` inside `calc_moment_target`, not the explanatory error.

So the version first shipped reproduced — in a new shape — the very silent-wrong-answer this ADR
set out to remove (cf. option D's rejection, *"a lone vector cannot mean both"*, which recurs at
the functor level). This is the same failure the `SingletonVector` Context described: *"it
silently does the wrong thing for measures that genuinely need weights."*

**Fix.** Keep the single-argument functor as the contract for *eligible* measures, but move the
safety guarantee off the functor onto a gated contract entry: a predicate
`supports_precomputed_returns(r)` (kind-derived, with the moment family's eligibility turning on
whether its `mu` target is weight-independent) consulted by `expected_risk_from_returns(r, x)`,
through which the precomputed-series call sites route. Ratio composites recurse into their
constituents. The amended Decision bullet above describes the result. Guarded by a behavioural +
completeness test in `test/test_09c_risk_input_kind.jl`.

## Correction 2 — eligibility moves to the definition site (ADR 0006 consistency)

The gated-eligibility fix above first shipped `supports_precomputed_returns` with the moment
family's eligibility recorded **centrally**, as a `Union` alias:

```julia
const PrecomputedMomentRM = Union{<:LowOrderMoment, <:HighOrderMoment, <:Skewness,
                                  <:Kurtosis, <:MedianAbsoluteDeviation, <:ThirdCentralMoment}
supports_precomputed_returns(::WeightsReturnsFeesInput, ::Any)                 = false
supports_precomputed_returns(::WeightsReturnsFeesInput, r::PrecomputedMomentRM) =
    weight_independent_target(r.mu)
```

This reintroduced exactly the pattern **ADR 0006** retired. "Whether `Kurtosis` supports a
precomputed series" is a fact *about Kurtosis*, but it lived in a distant union — the same
**locality** complaint 0006 made against the routing unions, in the very file whose own ADR text
argues that is the wrong place. The failure mode was milder than 0006's silent wrong number (a
forgotten measure falls to `false` → a *loud refusal* at `expected_risk_from_returns`, not a
wrong result), but it was **silent at classification time and untested**: the completeness test
only asserted the predicate *resolves* to a `Bool`, which a forgotten measure still does
(`false`), so a future moment measure dropped from the union would be silently mis-refused with a
green CI.

**Fix.** Delete `PrecomputedMomentRM` and push the eligibility decision to each measure's
definition site, mirroring how `risk_input_kind` is already declared there (ADR 0006 / 0001
lineage):

```julia
# at the Kurtosis definition site, beside `risk_input_kind(::Kurtosis)`:
supports_precomputed_returns(r::Kurtosis) = weight_independent_target(r.mu)
# at the TrackingRiskMeasure site:
supports_precomputed_returns(::TrackingRiskMeasure) = false
```

The two **uniform** kinds stay kind-derived — re-stating them per measure would be noise
(`supports_precomputed_returns(::NetReturnsInput, ::Any) = true`,
`(::WeightsInput, ::Any) = false`). Only the **`WeightsReturnsFeesInput`** branch — the one kind
where eligibility is *not* uniform (the 6 moment measures are instance-dependent on `r.mu`; the
3 weights-dependent ones — `TrackingRiskMeasure`, `RiskTrackingRiskMeasure`,
`VarianceSkewKurtosis` — are `false`) — drops its `false` default for an **erroring leaf**, the
same "erroring default, not a fallback value" discipline ADR 0006 chose for `risk_input_kind`:

```julia
supports_precomputed_returns(::WeightsReturnsFeesInput, r) =
    throw(ArgumentError("`$(typeof(r))` … does not declare `supports_precomputed_returns` …"))
```

So all 9 `WeightsReturnsFeesInput` measures now declare at their site; the ratio composites keep
their existing public-method overrides. A future WRF measure that forgets **throws** at the leaf,
and the completeness test — `Bool <: return_types(supports_precomputed_returns, (T,))` — now
**fails** on it (the throwing leaf infers `Union{}`), closing the untested-misclassification gap
the first correction left. Net effect: the ADR-0006 locality property is restored for the
precomputed-returns axis, at the cost of 3 explicit `= false` lines for the weights-dependent WRF
measures.
