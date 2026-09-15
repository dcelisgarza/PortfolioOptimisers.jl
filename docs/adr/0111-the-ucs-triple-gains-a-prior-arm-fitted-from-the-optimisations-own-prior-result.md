---
status: accepted
---

# The ucs triple gains a prior arm, fitted from the optimisation's own prior result

## Context

Every `AbstractUncertaintySetEstimator` in the library carries its own prior estimator, `pe`, and
fits it on the returns it is handed. The triple `ucs`, `mu_ucs` and `sigma_ucs` states that
contract: each verb takes returns data, and each estimator's fit is defined on `X` and `F`. The set
that comes back therefore knows nothing about the prior the optimiser is solving on, and that has
been enough, because every set the library builds is a neighbourhood of a statistic it measured for
itself.

[Issue #643](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/643) needs a set that is
not. Both orthogonal sets it charts are built from the fitted factor model of the prior the
optimiser is solving on: the effective loading matrix, the idiosyncratic covariance, and a
cross-sectional weighting read off the block. They keep the point estimate where it is and add a
penalty on the portfolio's exposure to the subspace the factors do not span. Those inputs exist only
on a fitted prior result, and no returns matrix carries them. The reference implementation fits its
set estimator inside the problem build and hands it the fitted return distribution as an extra
argument.

[Issue #654](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/654) measured every site at
which a set estimator is fitted today, and there are five:

1. The robust-return builder, `mu_ucs(pret.ucs, rd)` in
   `20_Optimisation/09_JuMPConstraints/02_Returns_and_ObjectiveFunctions.jl`.
2. The uncertainty set variance builder, `sigma_ucs(ucs, rd)` in
   `20_Optimisation/20_RiskMeasureConstraints/02_VarianceConstraints.jl`.
3. The near-optimal-centering pre-fit, `ucs_risk_measure(noc.r, rd)` in
   `20_Optimisation/13_NearOptimalCentering.jl`, which runs **before** any prior exists.
4. The Pipeline step, `run_uncertainty_step` in `25_Pipeline/02_StepExecution.jl`, which reads the
   `:returns` slot.
5. The hierarchical route, where `port_opt_view` carries an estimator across a cluster boundary
   unchanged and the cluster's own sub-optimiser fits it.

The first two hold the reduced prior result beside the returns already. The third does not, and the
fourth reads a slot that the Pipeline may or may not have filled.

[ADR 0050](0050-an-uncertainty-set-carries-the-quantity-it-bounds.md) fixes what such a set does
about its centre, and it is cited here rather than amended. A member of the new root copies its
fit's `pr.mu` into `val`, exactly as every estimator-built set already copies its own fit's mean, so
the rule that ADR states needs no widening.

## Decision

### A prior arm on the triple, not a preprocessing step

An unexported root, **`AbstractPriorUncertaintySetEstimator <: AbstractUncertaintySetEstimator`**,
declares an interface of three verbs that take a prior result in place of returns data:

```julia
ucs(ue::AbstractPriorUncertaintySetEstimator, pr::AbstractPriorResult; kwargs...)
mu_ucs(ue::AbstractPriorUncertaintySetEstimator, pr::AbstractPriorResult; kwargs...)
sigma_ucs(ue::AbstractPriorUncertaintySetEstimator, pr::AbstractPriorResult; kwargs...)
```

A member carries no `pe`, because there is nothing for it to fit: its inputs are on the result it is
handed.

The triple gains a **three-argument form**, `ucs(uc, rd, pr)` and its two siblings, and one
per-type predicate, `reads_prior_result(uc)`, decides which argument the form hands on. An
estimator that answers `false` — every estimator that carries its own `pe` — has the prior dropped
and is forwarded to the two-argument returns method. An estimator that answers `true` is forwarded
to the two-argument prior method, and the returns travel beside it as an `rd` keyword the prior
method defaults to `nothing`, so a rule that needs an optimiser can run one
([ADR 0127](0127-the-compact-covariance-radius-is-sized-in-family-not-through-the-calibration-channel.md)).
The new root answers `true` by its type. A built set, or an empty slot, passes through — the
existing passthrough already takes `args...`, so it answers the new form with no new method. Both
JuMP builders pass the prior they hold as one more positional argument, so one call site serves
every estimator and the predicate decides which argument is read.

The prior is *dropped* where the predicate answers `false` rather than checked. An estimator that
carries its own `pe` fits it on the returns it is handed, so the optimisation's own prior is not an
input of that fit and passing it changes no number.

The three-argument form was first written as three methods per verb, one on each of the two roots
and the passthrough, and the two consumers below dispatched on the new root the same way.
[ADR 0138](0138-an-uncertainty-set-with-no-prior-of-its-own-is-calibrated-on-the-prior-result-it-is-handed-and-a-scenario-cap-states-its-count.md)
replaced that dispatch with the predicate, because the four returns-data families gained a
`pe = nothing` mode that reads the prior result too, and a mode of a type is not a type a method
can be written on. The three consumers ask the predicate and agree; the root's members declare
nothing, because the root's own method of the predicate answers for them.

### A standalone fit reduces to the Investable Mask and expands

The prior arm offers a mode the reference lacks: `mu_ucs(ue, pr)` on a stored prior result builds
the set with no returns and no optimiser. Inside an optimiser the result arrives reduced, because
every family reduces once at its entry
([ADR 0115](0115-every-optimisation-estimator-reduces-once-at-its-entry-and-its-result-carries-the-investable-mask.md)).
Standalone it arrives whole, and a prior fitted on a point-in-time Asset Panel writes `NaN` on
every asset outside its Investable Mask, in the moments and in every block the factor fit wrote.

The prior arm therefore takes the shape
[ADR 0117](0117-a-prior-reduces-to-the-coverage-universe-and-a-plain-moment-estimator-refuses-a-non-finite-sample.md)
gives every prior result: **reduce, fit, expand.** `investable_ucs_reduction` derives the mask
and takes the optimiser's own view of the prior result and of the returns beside it, the fit runs
on that view, and `expand_investable_ucs` writes the built set back onto the full universe. On a
result that arrived reduced the mask is `nothing`, and both verbs are passthroughs.

What the expansion writes outside the mask is the set's own business, and each built set states it
beside its `port_opt_view`, because the two are inverses: a view of the expanded set at the mask
recovers the reduced fit. The orthogonal geometry writes **zero rows** — a zero row of the norm
ball's map moves nothing on that asset, a zero row of the compact basis is dropped by the view
without moving the span and leaves the sliced basis orthonormal, and a zero metric entry states
nothing — and carries the full `pr.mu` or `pr.sigma` as `val`, `NaN` frame and all, because
[ADR 0050](0050-an-uncertainty-set-carries-the-quantity-it-bounds.md) says a set carries the
quantity it was calibrated on and that quantity lives on the full universe. A box carries the
moment's own `NaN` frame instead, because a bound on `mu` lives where `mu` lives; an ellipsoid's
shape matrix is zero outside the mask, because the same set converts into a norm ball whose map
must be finite. The three returns-data families under `pe = nothing` take the same shape, and
their returns route reaches it through the tail the two routes share
([#1063](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1063)).

The gain is the pre-built route. A set fitted standalone on the full universe passes through
`port_opt_view(opt, idx, pr.X)` at the optimiser's entry, so the same weights are reached whether
the caller hands the optimiser the estimator or the set it fitted itself. A set fitted on the
reduced prior could not be handed back: its rows do not line up with the full universe.

The refusal a non-finite loading raises inside `orthogonal_factor_span` stays, and now names a
defect rather than a mode: every asset it counts is inside the mask, with a finite moment and a
loadings row that is not.

### The three other fit sites

**The near-optimal-centering pre-fit passes the new root through unchanged.** `ucs_risk_measure`
on an `UncertaintySetVariance` asks `reads_prior_result(r.ucs)` and returns the risk measure as it
stands when the answer is `true`. The pre-fit runs before any prior exists, so there is
nothing to fit there; the estimator travels to the builder, and each corner solve fits it against
the prior that solve was handed. That is one fit per corner rather than one shared fit, and it is
the correct answer rather than a compromise: each corner solves on its own prior.

**The Pipeline's `run_uncertainty_step` reads the `:prior` slot for the new root.** The step asks
the same predicate through `uncertainty_step_source`, serves the same three targets — `:mu`,
`:sigma` and `:both` — and requires the `:prior` slot instead of `:returns`. A
prior step must therefore come earlier. The returns are not read at all, so a pipeline that writes
`:prior` from a precomputed result needs no returns for this step.

**The hierarchical route needs nothing.** The estimator passthrough of `port_opt_view` already
carries an estimator across a cluster boundary unchanged, and `port_opt_view(pr, i)` slices the
loading rows and the idiosyncratic variances, so the cluster's sub-optimiser refits the set from its
own prior view. That is the rule
[#659](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/659) stated for a subspace basis:
refit or re-orthonormalise, never slice.

### The idiosyncratic variances travel on the loadings block

The default cross-sectional weighting of the orthogonal sets is the inverse idiosyncratic variance,
so a member of the new root must read those variances off `pr.rr`. `CrossSectionalFactorModel`
carries them under `esigma`; `Regression` carried nothing.

**`Regression` gains an optional fourth field, `esigma`**, with the same name, the same two shapes
and the same guards as the cross-sectional block's field. `factor_lift` returns the residual
variances it already measured beside `mu`, `sigma` and `chol`, and the two priors that call it —
`FactorPrior` and `FactorBlackLittermanPrior` — write them onto the block under `rsd = true` and
leave the field unset under `rsd = false`. `HighOrderFactorPriorEstimator` forwards the wrapped
prior's block and builds none of its own, so it writes nothing;
`AugmentedBlackLittermanPrior` builds a block but never calls `factor_lift`, so it writes nothing
either.

One reader, `idiosyncratic_variances`, answers the variance vector off either block: a vector comes
back unchanged, a matrix comes back as its diagonal, and an unset field raises. The two refusals
name the filler of the block they came from — `rsd` for a `Regression`, the field itself for a
cross-sectional model — because the caller's next action differs. There is no fallback: a vector of
ones is a different weighting, not a missing one.

The write needs a verb of its own, `set_idiosyncratic_covariance`, rather than `Accessors.@set`.
`Regression` declares a `swap(L, M)` property rule, so `re.L` answers `re.M` whenever `L` is unset,
and a property-based rewrite materialises `L` as a copy of `M` and loses the unset-ness the rule
exists to express. The verb reads `L` and `b` with `getfield`, as `port_opt_view` already does for
the same reason.

## Alternatives rejected

| Option | How the set would reach the prior | Why it was not taken |
| --- | --- | --- |
| **A prior arm on the triple** | Both builders pass `pr` beside `rd`; a per-type predicate picks the argument. | Taken. |
| **`factory(rt, pr)`** | Resolve the slot to a fitted set inside `factory`, which already visits the return term with the prior in hand. | `factory` runs on the return term but on **no risk measure** on the JuMP route, so the covariance side would need a new call anyway. The reference fits its estimator inside the problem build and hands it the fitted result, which is the builder site rather than a preprocessing step. |
| **A Calibration Rule** | Put the set in the `ucs` slot as a rule resolved against the prior. | A Calibration Rule returns one number. It cannot read a factor model or build a set. |
| **A Deferred Quantity** | Refit the set from `pr.original_X` at resolution. | A Deferred Quantity refits from the returns, which is the input the orthogonal set does not read. The `ucs` slot holds an Estimator, not a Deferred Quantity, and the comment at that slot says so. |

## Consequences

- A member of the new root inherits the three-argument form through the root's `true` answer to
  `reads_prior_result`, and needs no method of the returns-data interface. No consumer reaches that
  interface through this root.
- Both builders now pass one more positional argument. Every existing estimator drops it, so no
  number in the library moves.
- A `Regression` widens from three fields to four. All source and test construction sites are
  keyword-based, so none breaks, and the doctests that print the result gain a field only when it is
  set.
- `FactorPrior` and `FactorBlackLittermanPrior` under `rsd = true` now return a block that carries
  the residual variances. A consumer that recomputed them from the reconstruction error can read
  them instead; `HighOrderFactorPriorEstimator` still recomputes them, because it subtracts a block
  the *wrapped* estimator added and reads its configuration through `factor_residual_config`.
- A near-optimal-centering solve with a prior-reading set fits that set once per corner rather than
  once for the problem. That is more work, and it is the only arrangement in which each corner's set
  matches each corner's prior.
- A Pipeline that runs a prior-reading uncertainty step must run a prior step first. The refusal
  names the `:prior` slot.
