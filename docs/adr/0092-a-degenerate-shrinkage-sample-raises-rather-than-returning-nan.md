---
status: accepted
---

# A degenerate shrinkage sample raises, rather than returning `NaN` or `Inf`

## Context

[#497](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/497), a child of
[#417](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/417), opened on a question the
sweep of [#460](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/460) raised. Three
legal inputs made
[`ShrunkExpectedReturns`](../../src/08_Moments/16_ShrunkExpectedReturns.jl) answer `NaN` or `Inf`
rather than a number or a raise:

```julia
X = randn(StableRNG(987654321), 40, 5) ./ 100

# 1. One asset, JamesStein, GrandMean or VolatilityWeighted.
mean(ShrunkExpectedReturns(; alg = JamesStein()), X[:, 1:1])                              # [NaN]
mean(ShrunkExpectedReturns(; alg = JamesStein(; tgt = VolatilityWeighted())), X[:, 1:1])  # [NaN]

# 2. One asset, BodnarOkhrinParolya, every target.
mean(ShrunkExpectedReturns(; alg = BodnarOkhrinParolya()), X[:, 1:1])                     # [NaN]

# 3. A square returns matrix, BodnarOkhrinParolya.
mean(ShrunkExpectedReturns(; alg = BodnarOkhrinParolya()), X[1:5, :])   # [NaN, NaN, NaN, Inf, NaN]
```

### The cause

Each case is a property of the published closed form, not a translation error. The sweep checked
all nine algorithm-target combinations against hand-computed closed forms and found no
disagreement.

| # | Site | Cause |
| ---: | --- | --- |
| 1 | the `alpha` assignment of the [`JamesStein`](../../src/08_Moments/16_ShrunkExpectedReturns.jl) method | `alpha` divides by `T * dot(mb, mb)`. `GrandMean` and `VolatilityWeighted` both reduce to the sample mean at `N == 1`, so `mb` is the zero vector and the denominator is exactly zero. `alpha` is `-Inf`, and `(1 - alpha) * mu + alpha * b` with `mu == b` is `Inf - Inf`. `MeanSquaredError` never reads the sample mean, so it stays finite. |
| 2 | the `alpha /= u * v - w^2` line of the [`BodnarOkhrinParolya`](../../src/08_Moments/16_ShrunkExpectedReturns.jl) method | `u * v - w^2` is a Cauchy-Schwarz gap in the inner product that `inv(sigma)` induces, so it vanishes exactly when the target is a multiple of the sample mean. At `N == 1` every vector is such a multiple, so the gap is exactly zero under all three targets and `alpha` is `0 / 0`. |
| 3 | the `N / (T - N)` term of the same line | Undefined at `T == N`. The term is positive for `T > N` and negative for `T < N`, so a wide returns matrix silently flips its sign. |

### The two readings

The ticket named both.

 1. **Keep the current behaviour.** A caller who sweeps a grid of universes and drops the `NaN`
    rows today keeps working. The three estimators are exported, so a raise where a number is
    returned today is a behaviour change on public API.
 2. **Guard each degeneracy.** A `NaN` expected-returns vector flows into an optimiser and fails
    far from its cause. Case 3 does not even fail: at `T < N` it produces one finite-looking
    coefficient of the wrong sign, which no caller can detect downstream.

## Decision

**Each degeneracy raises a `DomainError` at its own site.** Reading 2 wins, because the value a
caller receives today is not an answer. Two of the three cases produce a vector that a `NaN` filter
catches, and the third produces one that no filter catches.

Three guards ship, on the `@argcheck` pattern the rest of `src/` uses.

```julia
# JamesStein
mb2 = LinearAlgebra.dot(mb, mb)
@argcheck(!iszero(mb2), DomainError(mb2, "…the $(nameof(typeof(me.alg.tgt))) target equals the sample mean…"))

# BodnarOkhrinParolya, before the coefficients are formed
@argcheck(T > N, DomainError((T, N), "…they need more observations than assets…"))

# BodnarOkhrinParolya, after the three quadratic forms
gap = u * v - w^2
@argcheck(!iszero(gap), DomainError(gap, "…the $(nameof(typeof(me.alg.tgt))) target is a multiple of the sample mean…"))
```

**A `DomainError` is the right error, not an `ArgumentError`.** The input is legal — a returns
matrix of any shape is a well-formed argument. The estimator is not defined on it. That is what
`DomainError` states, and `src/22_Preselection.jl`, `src/06_Detone.jl` and
`src/08_Moments/35_GerberIQCovariance.jl` already use it for the same distinction.

**Each test is exact, not a tolerance.** The three denominators are exactly zero on the degenerate
input, so `iszero` is the whole condition. A near-zero denominator gives a large but finite
coefficient, which is a property of the closed form rather than a degeneracy, and it is documented
under `# Mathematical definition` where it was.

**`T > N` is checked before the coefficients are formed, and the gap after.** The shape guard needs
only `size(X)`, so it fires on a square or wide matrix without solving for `isigma`. The gap guard
needs `u`, `v` and `w`, so it sits after them. A one-asset sample therefore reaches the gap guard,
which is the guard whose message names the cause.

**`BayesStein` gains no guard.** Its denominator is `(N + 2) + T * dot(mb, isigma, mb)`, which is
strictly positive whenever `isigma` is positive semidefinite. It answers a finite vector on all
three degenerate samples.

## Consequences

- **Three exported estimators raise where they returned a number.** That is the whole size of the
  behaviour change, and it is the point of the ADR. A caller who sweeps universes and drops the
  `NaN` rows must catch a `DomainError` instead.
- **No non-degenerate result moves.** Each guard fires on an exact zero or on `T <= N`, so every
  sample that answered a finite vector before answers the same vector, bit for bit.
- **`JamesStein` with `MeanSquaredError` still works at one asset.** That target does not read the
  sample mean, so its denominator is not zero and the first guard does not fire.
- **`BodnarOkhrinParolya` states its own published condition.** `T > N` is the condition the source
  states, and the method now enforces it rather than documenting it.
- **The three docstrings state the raise where a reader looks for it.** Each method gains a
  `# Validation` section, and the sentences that promised a `NaN` or an `Inf` under
  `# Mathematical definition` now name the raise.
- **`test_08_moments.jl` holds the contract.** The sub-testset
  `the degenerate samples the docstrings name`, under `The shrunk expected returns sweep of #460`,
  turns the three `NaN` and `Inf` assertions into `@test_throws DomainError`, pins that the
  one-asset `BodnarOkhrinParolya` raise is the gap guard rather than the shape guard, and adds the
  `T < N` case that returned a finite coefficient of the wrong sign.
