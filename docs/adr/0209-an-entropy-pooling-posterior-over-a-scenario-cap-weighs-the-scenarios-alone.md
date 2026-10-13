---
status: accepted
---

# An entropy pooling posterior over a Scenario Cap weighs the scenarios alone

## Context

[#1667](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1667) found that
`EntropyPoolingPrior` and `MeucciEntropyPoolingPrior` with a view over a prior with a Scenario Cap
raised a bare `DimensionMismatch: Inconsistent array dimension.` from `StatsBase`. On the data of
the issue, `EmpiricalPrior(; max_scenarios = 40)` fits `mu` and `sigma` over 60 observations and
carries the last 40 rows as `pr.X` (ADR 0136). The prior probabilities `w0` and the posterior
weights `w1` live on these 40 rows (ADR 0116). The refit through `scenario_prior` gave the 40
weights to `me` and `ce` by `factory`, and they read all 60 rows. The refit under a caller's
`pe.w` had the same shape. A `FactorPrior` over a capped factor prior failed in the same way, on
the rows of `F`.

`OpinionPoolingPrior` with the cap in `pe1` did not fail, because it fits its opinions on the 40
rows of `prior(pe1, X).X`. `HighOrderPriorEstimator` reads `pr.w` on the last `length(w)` rows
since ADR 0206.

Two readings existed. The posterior moments are those of the 40 scenarios under `w1`. Or the
library refuses a Scenario Cap under an entropy pooling prior with a view.

## Decision

**The posterior moments of an entropy pooling prior over a Scenario Cap are the moments of the
scenarios under the posterior weights.** Entropy pooling holds the scenarios and changes only their
probabilities (ADR 0205). The posterior states no probability for an observation that the cap
removed, so that observation has the probability zero.

1. `scenario_weights(w, t)` puts the probabilities of the `s` scenarios on the `t` observations
   that the moments read: `t - s` zeros, then `w`. A Scenario Cap keeps the last rows, so the
   zeros go first. It refuses `s > t`, and it returns `w` when `s == t`.
2. `scenario_factory` of an `EmpiricalPrior` gives `scenario_weights(w, size(X, 1))` to `me` and
   `ce`. Every other method of `scenario_factory` already walks to the `EmpiricalPrior` that fits
   the scenarios, on `X` or on `F`, so the rule reaches a `FactorPrior`, a Black-Litterman prior
   and a nested entropy pooling prior with no more code. The regression and the residual block of
   a factor prior keep the weights of the fit that the views read (ADR 0205).
3. A caller's `pe.w` reaches the embedded prior through `prior_probability_factory`. It is
   `factory` with each weighted field on its own rows. A capped `EmpiricalPrior` takes
   `scenario_weights` on the rows of `X`. The regression and the residual block of a factor prior
   take `scenario_weights` on the rows of `F`, because a caller's prior probabilities weigh that
   fit (ADR 0116). A nested entropy pooling prior keeps `w` as its own prior probabilities, on its
   scenarios.
4. A weighted moment with a zero weight equals the same moment over the rows of positive weight.
   On the data of the issue, the refit equals `prior(factory(EmpiricalPrior(), w1), X[21:end, :])`
   to `1.7e-21` in `sigma`, and to zero in `mu`. The bias correction of `sigma` counts the 40
   scenarios, as the fit on the last rows does.

## Alternatives rejected

- **Refuse a Scenario Cap under an entropy pooling prior with a view.** Entropy pooling describes
  the distribution of its scenarios, and the high order prior reads it so since ADR 0206. A
  refusal removes a configuration that has one correct answer.
- **Cut `X`, `F` and the Asset Panel to the last `length(w)` rows before the refit**, the rule of
  `scenario_rows`. A `FactorPrior` then fits its regression again on the last rows alone. Its
  loadings and its scenarios move, and a view stated on the first fit no longer holds on the
  read-out. The zero weights reach only the fields that read the weights.
- **Pad the weights before `factory` at the top of the tree.** A `CrossSectionalFactorPrior`
  drops the rows of its Descriptor warm-up, and every one of its fields reads the rows that it
  keeps (ADR 0116). A nested entropy pooling prior states its `w` on its scenarios. Only the
  estimator that holds the cap knows how many rows its moments read.

## Consequences

- No released number moves. Every configuration that this decision changes raised an error
  before it.
- `mu == pr.X' * pr.w` holds under a mean view over `EmpiricalPrior(; max_scenarios = 40)`, and
  over `FactorPrior(; pe = EmpiricalPrior(; max_scenarios = 40))`, for `H0_EntropyPooling`, the
  staged algorithms and `MeucciEntropyPoolingPrior`, with and without a caller's `w`.
- Under a posterior, a Scenario Cap no longer means "moments over the long window". The moments
  of a capped prior without a posterior are still fitted over every observation.
- A `CrossSectionalFactorPrior` whose `pe` holds a Scenario Cap takes the same rule since
  [#1685](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1685). Its fields read
  the factor returns of the rows it keeps. The refit gives the posterior weights to `pe` alone,
  through `scenario_weights` on those rows, and holds every other field (ADR 0205). A caller's
  `pe.w` reaches every field through `scenario_weights` on the same rows.
