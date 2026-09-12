---
status: accepted
---

# The compact covariance radius is sized in family, not through the calibration channel

## Context

`OrthogonalUncertaintySet` carries two radii, and only one of them was calibrated.

The **mean radius** inverts a sampling distribution. `k_norm_ball(ue.method, ue.q, nothing, lambda, r)`
sizes it at `r`, the dimension of the Orthogonal Subspace, through the `method::Num_UcSK` slot and
the `AbstractUncertaintyKAlgorithm` family. The **covariance radius** `kappa` was a plain `Number`
defaulting to `1.0`, and its own docstring called it "a size the caller states rather than a
quantile". It multiplies the quadratic penalty of the `CompactCovarianceUncertaintySet` the same
estimator builds, and `0` leaves the nominal variance untouched.

[#928](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/928) asked for a rule, or for
evidence that no rule of the ADR 0095 kind can exist. It was the *Selecting a radius* patch that
kept map [#643](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/643) open. The
reference implementation states the radius as `radius : float, default=1.0` with no calibration and
no selection procedure, so anything found here is beyond parity, which was the intent.

Two rules exist. The exploration also found that neither can be a **Calibration Rule**.

## Decision

### The radius is sized in family, because its units move with a sibling field

ADR 0095's channel hands a rule `(key, pr, w, slv, ctx)` and nothing else. That is enough for every
quantity it serves, and it is not enough for this one.

The penalty is `κ‖(I − QQᵀ)Cw‖²` with `C = W^{-1/2}`, and `W` is the cross-sectional metric the
`metric` field of the *same estimator* names. Under `IdentityMetric` `C = I`, so `κ` carries
**variance units**. Under the default `InverseIdiosyncraticVarianceMetric` `C = D^{1/2}`, the
penalty matrix is the idiosyncratic covariance projected onto the complement, and `κ` is
**dimensionless**. A rule that saw only the prior result would have to commit to one of the two
readings and be wrong under the other, and no argument the channel carries names which.

So the radius is sized where the metric, the loadings block, the diagonal metric `C` and the span
`Q` are all in hand: inside `orthogonal_sigma_set`, through a slot on the estimator. This is the
shape the mean radius of the same estimator already has, and ADR 0070 is the Authority for it — a
radius slot admits the rule that computes it.

| | Mean radius | Covariance radius |
| :--- | :--- | :--- |
| Slot | `method::Num_UcSK` | `kappa::Num_CptRad` |
| Family | `AbstractUncertaintyKAlgorithm` | `AbstractCompactRadiusAlgorithm` |
| Verb | `k_norm_ball` | `k_compact` |
| Sized in | `orthogonal_mu_set` | `orthogonal_sigma_set` |

`Num_CptRad = Union{<:AbstractCompactRadiusAlgorithm, <:Number}`. A stated number is the radius, and
`k_compact` returns it unchanged, so the widening costs the existing caller nothing. The bound admits
**no plain `Function`**, and that is the one place this family parts from ADR 0095's five: a rule
here reads eight arguments the site settles, so a closure would have to restate every one of them to
be callable at all.

The root and the bound live in `src/14_UncertaintySets/01_Base_UncertaintySets.jl`, beside
`AbstractUncertaintyKAlgorithm` and `Num_UcSK`, because the field they bound is declared in file 09
and a bound must load before the field. The rules live in
`src/14_UncertaintySets/10_CompactRadiusRules.jl`, appended so that nothing renumbers.

### Two rules ship, and each answers one of the ticket's questions

**`ResidualInflation` is a quantile, so the answer to "is there a sampling distribution?" is yes.**
The penalty lives exactly where the idiosyncratic variance lives: the spanned directions pay nothing,
and the complement carries `D`. So the quantity to bound is the estimation error of `D`, and a
variance has a chi-squared bound.

```text
ρ = ν / χ²⁻¹_ν(q) − 1
κ = ρ ‖D^{1/2} W^{1/2} (I − QQᵀ)‖₂²
```

`ρ` is the *excess* of the level-`1 − q` upper bound on `d̂ᵢ` over the estimate itself, and it is
dimensionless. One formula serves every metric, and no method dispatches on the metric: under the
default metric `W = D⁻¹` leaves a bare projector inside the norm, whose operator norm is `1`, so `κ`
collapses to `ρ`; under `IdentityMetric` the same norm is `λmax(PDP)` and carries the variance units
`κ` needs there. The norm is the *tightest* radius satisfying the set's own bound, because
conjugating `W^{-1/2}` out of `κCᵀ(I − QQᵀ)C ⪰ ρΠDΠᵀ` leaves `κP ⪰ ρPW^{1/2}DW^{1/2}P`, whose
solution on the range of the projector is that norm. A span that covers the cross-section leaves
`P = 0` and a radius of zero, which is the answer the mean axis already gives for the same span.

**`VarianceFraction` is a scale match, so the answer to "is there a rule that is not a quantile?" is
also yes.**

```text
κ = f (w₀ᵀΣ̂w₀) / ‖(I − QQᵀ)Cw₀‖²
```

The denominator is the penalty `w₀` pays at a unit radius, so the quotient is exactly the radius at
which that penalty reaches `f` of the nominal variance. It converts a magnitude into a unit a caller
can reason about — *robustify by ten percent of nominal variance* — and it assumes no distribution,
which serves a block whose idiosyncratic variances were measured under an estimator whose degrees of
freedom nobody can state.

### The degrees of freedom are stated, because no block records them

`T − K − 1` prices a textbook OLS residual, and the library never produces one. `FactorPrior` writes
`esigma` as the column variances of the reconstruction error under whatever variance estimator it was
handed, and a Cross-Sectional Factor Prior writes the idiosyncratic covariance its own fit measured,
which `16_Base_CrossSectionalFactorPrior.jl` states explicitly is *not* `var(ve, X - posterior_X)`.

So `compact_radius_dof` states the count the **fit** spent, per block type: `T − K − 1` on a
`Regression`, whose per-asset time-series fit spends `K` factors and an intercept out of `T`
observations, and `T(N − K)/N` on a `CrossSectionalFactorModel`, whose per-period fit spends `K` of
the `N` assets each period rather than `K` of the `T` observations once. `dof = nothing` derives it
and a stated number overrides it, so a caller who knows what their estimator cost says so.

`T` is Kish's effective sample size when the prior carries observation weights, and the raw row count
otherwise. This is the reading `ConcentrationRadius` already takes, for the same reason: a rule that
prices estimation error reads the sample size the information is in.

### The two `q`s are the same kind of number over two different errors

`ue.q` and `ResidualInflation.q` are both tail probabilities inverting a chi-squared, and they differ
in what is uncertain, at what degrees of freedom, and how they scale.

| | `ue.q` | `ResidualInflation.q` |
| :--- | :--- | :--- |
| What is uncertain | the mean vector `μ` | the idiosyncratic variances `d` |
| Degrees of freedom | `r`, the subspace dimension | the residual dof of the fit |
| Which tail | the upper `1 − q` quantile | the lower `q` quantile |
| Reads the sample length | no | yes, and shrinks like `√(2/T)` |

Both tighten as `q` falls, so "smaller `q` is more conservative" reads the same on both axes. `q =
nothing` therefore reads `ue.q`, and one stated level governs both. A caller who means them apart
states the rule's own, because `T` cancels out of one and dominates the other: a long sample leaves
the mean radius untouched while collapsing `κ` toward zero.

### The optimisation roots move to `02_TypeRoots.jl`

`VarianceFraction.w0` admits `nothing`, a weight vector, or any
`NonFiniteAllocationOptimisationEstimator` — naive, clustering, JuMP or stacking. Each carries its own
solver, so **nothing is threaded into the uncertainty-set fit**, and `ucs`, `mu_ucs` and `sigma_ucs`
keep their signatures.

That type was declared in `src/20_Optimisation/01_Base_Optimisation.jl`, a hundred includes after
`src/14_UncertaintySets/`. A bound is the enforcement this library prefers over a runtime check, so
the chain the bound names — `AbstractOptimisationEstimator`, `OptimisationEstimator` and
`NonFiniteAllocationOptimisationEstimator` — moves to `src/01_Base/02_TypeRoots.jl`, where
`CrossValidationEstimator` already stands for the same reason. `BaseOptimisationEstimator`, `VecOptE`
and every method stay where they were: only the three declarations moved, and no export changed.

The returns data the optimiser runs on already reached the estimator and was thrown away.
`ucs(uc::AbstractPriorUncertaintySetEstimator, ::ReturnsResult, pr)` and its two siblings discarded
their second argument; they now forward it as an `rd` keyword, which the two-argument entry points
default to `nothing`. A rule that needs an optimiser and meets no returns data refuses by name rather
than failing inside `optimise`.

### A vanishing penalty at the reference portfolio is documented, not guarded

`VarianceFraction`'s denominator vanishes two ways, and they are not one case.

`r = N` means the factors span the whole cross-section, the penalty is identically zero on **every**
portfolio, and the set is inert. That is a correct configuration and not an error — the mean axis
already returns a zero radius for it — so an exact rank test, `size(Q, 2) == size(Q, 1)`, returns
zero. It is a statement about the rank of an orthonormal basis and not a tolerance.

`Cw₀ ∈ col(Q)` with `r < N` means the *reference portfolio* pays no penalty while others do, and no
finite radius states a fraction of nothing. The quotient diverges. **No guard is written**, because
the two ways it arrives do not share a refusal: an exactly vanishing penalty gives a value that is
not finite, which `CompactCovarianceUncertaintySet`'s own constructor already refuses, while a
projector that leaves a rounding residue gives a finite and enormous radius that no threshold
separates from a legitimately large one. The docstring states the failure and names the two ways out.

### A view carries the number, and the reason is that a Result holds no rule

`port_opt_view` on a `CompactCovarianceUncertaintySet` carries `kappa` through unchanged. Its own
prose said that was because the radius is caller-stated rather than a quantile of a dimension, which
this decision makes false. The reason is that a **set is a Result**: it holds the radius as a number
and no rule, and the estimator, the prior result and the metric that would size one are all out of
reach at a view.

`orthonormalise_basis` can lower the rank of the sliced basis, so a radius sized against the full
span is no longer the tightest one for the sliced problem. It stays a valid multiplier, and the mean
axis already carries the same staleness, its radius being a quantile at a subspace dimension a slice
also moves. A caller who needs the radius sized against the smaller universe fits the set on a prior
that was reduced first, which is the route the two JuMP builders already take.

### The concrete rules are exported, and nothing else is

`ResidualInflation` and `VarianceFraction` are exported. `AbstractCompactRadiusAlgorithm`,
`Num_CptRad` and `k_compact` are not, exactly as `AbstractUncertaintyKAlgorithm`, `Num_UcSK` and
`k_norm_ball` are not. So no row is added to the exported-abstract-type allow-list and no Capability
Catalogue entry is owed for a function. Both unexported names are documented on the API pages and
reachable by `@ref`.

### The search path is written down

Walk-forward *selection* of the radius needed nothing built: `kappa` is a plain field, so a
`"ucs.kappa" => values` pair in a `GridSearchCrossValidation` or `RandomisedSearchCrossValidation`
grid reaches it through `parse_lens`. Nothing pinned that, and neither the `kappa` docstring nor the
deep-dive example said so. The field docstring now states all three routes — a stated size, a rule,
or a searched value — a test pins the lens path over numbers *and* rules, and the deep-dive example
shows the calibrated route beside the hand sweep it already had.

## Rejected alternatives

**A sixth family in ADR 0095's calibration channel.** Uniform with the eleven rules that ship, and
with the lens and schedule stories already written for them. Rejected because the channel would have
to be widened twice to carry it: `CalibrationContext` would gain the metric and the span, which
belong to one estimator and to no risk measure, and a resolution route for
`AbstractPriorUncertaintySetEstimator` would have to be invented, since no uncertainty-set estimator
declares `calibration_slots` today. The in-family slot needs neither, and it is the shape the sibling
radius on the same estimator already has.

**Dispatching `k_compact` on the metric.** A method per `AbstractOrthogonalityMetric`, each stating
that metric's reading of the units. Rejected once the general form was worked out: one operator norm
is the exact tightest radius under **every** metric, so the four methods would be four spellings of
one formula, three of them approximations of it, and a fifth metric added later would silently get
none.

**A `dof` field with no derivation, mandatory like `ScenarioCount.n`.** Honest about the library
never producing an OLS residual. Rejected because the derivation is right for the common case and the
sentinel already lets a caller who knows better say so; a mandatory keyword would charge every caller
for a number that the block type determines.

**Guarding the vanishing denominator with a tolerance.** A message naming `w0` and the span reads
better than `kappa must be finite and >= 0`. Rejected because the comparison needs a tolerance on a
norm, and this library documents such a failure rather than guarding it. The rank case, which needs
no tolerance, *is* branched.

**A new `AbstractReferencePortfolio` family for `w0`.** Two members, `EqualWeight` and
`MinimumVariance`, with no load-order problem to solve. Rejected because the library already names
these objects: `EqualWeighted` and `InverseVolatility` are shipped optimisers, and a parallel family
would restate them under second names. Moving three abstract declarations buys the whole
`NonFiniteAllocationOptimisationEstimator` family instead of two hand-written members.

## Consequences

- `OrthogonalUncertaintySet.kappa` widens from `Number` to `Num_CptRad`. A stated number behaves
  exactly as before, and a rule of another family is refused at construction by the bound.
- A calibrated radius is far smaller than the shipped default: `ResidualInflation` returns about
  `0.17` for a 250-observation, 5-factor fit at `q = 0.05`, against a default of `1.0` and a
  deep-dive example that used `100.0`. A caller who moves from a stated radius to a rule gets a much
  less conservative set, and the docstrings say so.
- `ResidualInflation` reads `rr.esigma`, so it refuses a block fitted with `rsd = false`.
  `VarianceFraction` reads none, so it serves that block.
- Three abstract types moved files, so `src/01_Base/02_TypeRoots.jl` and
  `src/20_Optimisation/01_Base_Optimisation.jl` both move in the sweep manifest and in the size and
  complexity baselines.
