---
status: proposed
---

# The second online selection set binds each rule to the geometry of its own projection, and a weighting that reads cumulative wealth carries the prior

## Context

[ADR 0156](0156-every-online-selection-algorithm-ships-as-a-closed-form-rule-an-expert-mixture-or-a-follow-the-leader-over-a-selected-sample.md)
put the closed-form rules beyond the prototype in a second set and
[ADR 0159](0159-a-constrained-online-update-projects-onto-an-allocation-set-in-the-rules-own-geometry-and-the-default-set-needs-no-solver.md)
stated the Projection Geometry bound of every first-set rule and said each later set states its
own. The parity decision ([ADR 0165](0165-a-first-order-online-rule-is-one-mirror-descent-step-over-any-geometry-and-the-evaluation-surface-reaches-dynamic-regret.md))
moved the switching portfolio and the gradient projection out of this set into constructors of
the mixture and of the mirror-descent rule, kept the expectation-maximisation update as a struct
because it is Soft-Bayes formula for formula and not a mirror step, and the paper task
([issue #1185](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1185)) added the weak
aggregating algorithm from its two applied papers. What is left to decide is per rule: the
geometry its `proj` slot is bound to, what its carrier holds, and how a weighting that reads the
experts' cumulative wealth rather than their last return fits the Rule State of
[ADR 0157](0157-an-online-selection-state-is-a-rule-state-beside-the-rows-held-once-and-a-block-of-rows-is-that-many-single-row-updates.md),
which hands a weighting the previous weight vector and one return vector.
[Issue #1177](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1177) builds the set.

Three things were measured before deciding.

- **The confidence-weighted rule's own projection is Euclidean.** Its geometry is the Gaussian
  relative entropy over `(μ, Σ)`, but the paper's algorithm projects the mean alone, in squared
  Euclidean distance onto the simplex, and rescales the covariance to a fixed trace, `Σ / (N² tr Σ)`.
  The 2011 text writes the constraint on `log⟨μ, x⟩` with `ε = −0.5`; the 2013 text on `⟨μ, x⟩`
  with `ε = 0.5`, the same step with the relative divided by `M_t` and the threshold compared
  with `log M_t`. The library's rule follows the 2013 text and matches its recursion to `1e-13`
  on the fixture under both formulations; the covariance stays diagonal throughout, so the
  carrier is one vector.
- **The anti-correlation transfer never leaves the simplex.** It moves at most `w_i` out of asset
  `i` and conserves the budget, so the paper's step needs no projection; the correlation is
  recomputed from the last `2w` rows every period and nothing else is carried.
- **The weak aggregating algorithm's weight is not a function of the previous weight.** It is
  `p_{t+1} ∝ p_1 exp(G_t / √(t + 1))` with `G_t` the experts' cumulative log wealth, so a weighting
  that receives `(p_t, r_t)` cannot compute it without carrying `G_t` and `p_1`. The same carrier
  serves the top-`k` selection, which ranks by `G_t`. Both applied papers fix Kalnishkan and
  Vyugin's constant at one, so the rate is `1 / √(t + 1)` with no field.

## Decision

### The bound table of the second set

| Rule | `proj` bound | Why |
| --- | --- | --- |
| `ConfidenceWeightedMeanReversion` | `EuclideanProjection` | the set constrains the mean alone, and the paper's own projection of the mean is Euclidean; the covariance is rescaled to its trace, never projected |
| `AntiCorrelation` | `EuclideanProjection` | the raw step lies in the simplex by construction, so on the default set the projection is the identity; on a stated set it is the Euclidean repair |
| `ExpectationMaximisation` | `EuclideanProjection` | the raw step is positive and sums to one wherever `w_t` does, so the default projection is the identity; the 1997 derivation's chi-squared divergence is a weighted Euclidean distance, and the library carries no chi-squared geometry |
| `AggregatingAlgorithm` | `EntropicProjection` | a multiplicative update of a non-negative vector, normalised |
| `TopK` | `EuclideanProjection` | a selection, `1 / k` on the top `k` and zero elsewhere, in the simplex by construction; the Euclidean repair on a stated set |
| `WeakAggregatingAlgorithm` | `EntropicProjection` | a Bayesian mixture weight, normalised |

Each bound is the geometry the rule's own step is already the projection in on the bare simplex,
which is ADR 0159's rule; none of the six admits a second geometry, so none is a `Union`.

### The carriers

- `ConfidenceWeightedMeanReversionState(n, sigma)`: the diagonal of the belief covariance, seeded at
  `1 / N²` per asset and rescaled to trace `1 / N²` after every step, as the paper's algorithm
  states. The mean of the belief is the allocation held, so the rule reads `w` and carries no
  mean; the covariance is the rule's own and never a Prior's, because it is a belief over weights
  and not a moment of returns ([ADR 0158](0158-a-forecast-reading-rule-holds-an-expected-returns-estimator-and-a-covariance-enters-on-the-constraint-that-reads-it.md)).
- `AntiCorrelation` carries `nothing` and reads the head's rows with `rows_needed = 2 · window`.
  The ticket named the lagged correlation as the carrier; it is recomputed from the rows every
  period, so a carrier would duplicate what the head already holds once. Until `2w` rows are held
  the rule continues from the allocation it is handed.
- `ExpectationMaximisationState(n, w1)`: the Start Allocation, the prior the online form of
  Orseau, Lattimore and Legg's Eq. 14 pulls towards whenever the rate falls. The update is
  written in that form, `w′ = (w ⊙ (1 − η_t + η_t g))·η_{t+1}/η_t + (1 − η_{t+1}/η_t)·w_1`, through
  `learning_rates(eta, t) -> (η_t, η_{t+1})`, which a number answers as `(eta, eta)` so the pull
  vanishes and the update is the plain step exactly. A Learning-Rate Schedule of ADR 0165 lands on
  the `eta` slot by adding a method of `learning_rates`, and the test file proves it with a probe
  rate type. The bound `eta::Real` is this ticket's; the schedule build widens it. `eta ∈ (0, 1)`
  is enforced at construction, both papers are cited, and the `O(√(T N log N))` bound with no lower
  bound on the price relatives is stated as the rule's own.
- `CumulativeWealthState(n, G, p0)`: the cumulative log wealth of every asset — every expert on a
  mixture's slot — and the Start Allocation, shared by `TopK` and `WeakAggregatingAlgorithm`.
  The prior of the weak aggregating step is the Start Allocation the seed receives, which on an
  `ExpertMixture` is its `p`, so the weighting carries no `p0` field of its own. `TopK` checks
  `k ≤ length(w)` at the seed, where the count is first known, and breaks equal wealth by index.
- `AggregatingAlgorithm` carries nothing: `w′ ∝ w ⊙ x^η`, buy-and-hold at `η = 1` on the head and
  on the mixture.

### One constructor for the aggregation of exponentiated-gradient experts

`AggregatingExponentialGradient(; etas = 0.01:0.01:0.2, eset, proj)` is the `ExpertMixture` under
`WeakAggregatingAlgorithm()` over one `ExponentiatedGradient` expert per rate. The continuous form
of the second paper discretises to the same grid in its own experiments and computes the same
numbers, so it is a sentence in the docstring and not a second name.

## Considered options

1. **A `theta` field on the confidence-weighted rule resolved to `phi = Φ⁻¹(θ)`.** Rejected: the
   paper's algorithm takes `φ` as its input and reports it as not decisive; a quantile of a
   normal is one call away for a caller who thinks in `θ`.
2. **The confidence-weighted rule under the trace normalisation `Σ / (N tr Σ)`, which keeps the
   seed's trace.** Rejected: the paper's algorithm box states `N² tr Σ`, and the map's parity is
   with the paper; the two differ by `4e-5` on the fixture.
3. **The anti-correlation rule on price levels.** Rejected: the paper's windows are log price
   relatives.
4. **A `p0` field on `WeakAggregatingAlgorithm`.** Rejected: the mixture's `p` is the prior
   already, and a second field is a second place for one number.
5. **`ContinuousAggregatingExponentialGradient` as a second constructor.** Rejected: the same
   numbers under a second name.
6. **A chi-squared Projection Geometry for the expectation-maximisation rule.** Deferred: the
   default set needs none, a stated set has the Euclidean repair, and a geometry with a theorem
   of its own is a fourth-set concern.

## Consequences

- New `src/17_Optimisation/10_OnlinePortfolioSelection/06_SecondSetRules.jl`; new
  `test/test_69_second_set.jl`; every rule joins the batch–online identity loop of `test_67`.
- `CONTEXT.md`'s roster gains `WeakAggregatingAlgorithm` · WAA and the constructor
  `AggregatingExponentialGradient` · WAEG, CAEG.
- The Learning-Rate Schedule build of ADR 0165 widens `ExpectationMaximisation`'s `eta` bound and
  adds a `learning_rates` method; nothing else in the rule moves.
- `SwitchingPortfolio` and `GradientProjection` are built on their own tickets as ADR 0165 rules.
