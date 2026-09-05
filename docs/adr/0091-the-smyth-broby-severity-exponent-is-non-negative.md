---
status: accepted
---

# The Smyth-Broby severity exponent is non-negative, and an infinite one is legal

## Context

[#496](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/496), a child of
[#417](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/417), opened on a raise from
the sweep of [#455](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/455).
[`SmythBrobyCovariance`](../../src/08_Moments/06_SmythBrobyCovariance.jl) has four numeric fields
and guarded three of them. `n` was read by no guard at all, and a negative `n` answered a matrix
whose diagonal is zero:

```julia
# The contribution is INVERTED: two magnitudes that agree contribute nothing.
PortfolioOptimisers.sb_delta(1.5, 1.5, -2)   # 0.0
PortfolioOptimisers.sb_delta(1.5, 0.5, -2)   # 0.9682458365518543

X = randn(StableRNG(11), 60, 4)
rho = cor(SmythBrobyCovariance(; n = -2, alg = SmythBroby1(), pdm = nothing), X)
diag(rho)          # [0.0, 0.0, 0.0, 0.0]  -- not a correlation matrix
any(!iszero, rho)  # true -- the off-diagonal is not zero
```

### The cause

The kernel is `sb_delta(ri, rj, n) = kappa / (1 + gamma^n)`, with `gamma` the absolute difference
of the two standardised magnitudes and `n` the severity of the divergence penalty. An asset has
`gamma = 0` against itself. At `n < 0` the term `gamma^n` is `Inf` there, so every diagonal
contribution is exactly zero. Off the diagonal `gamma^n` is small, the divisor approaches one, and
a pair whose magnitudes *disagree* contributes fully. That is the opposite of the source's
equation (1), whose purpose is to penalise divergence.

### The three readings

The ticket named three guards, and they differ at the two boundary values.

| Guard | Rejects | Keeps |
| --- | --- | --- |
| `assert_nonempty_nonneg_finite_val(n, :n)` | `n < 0`, `Inf`, `NaN` | `n = 0` |
| `assert_gt0` and finite | `n <= 0`, `Inf`, `NaN` | — |
| `assert_nonneg(n, :n)` | `n < 0`, `NaN` | `n = 0`, `Inf` |

Both boundary values name a statistic that already runs. `n = 0` gives `gamma^0 = 1` for every
`gamma`, so `delta = kappa / 2`: the amplitude with no divergence penalty. `n = Inf` gives
`gamma^Inf = 0` below one and `Inf` above, so `delta` is the amplitude when the two magnitudes are
within one of each other and zero when they are not: a hard divergence gate.

## Decision

**`n` is guarded by `assert_nonneg(n, :n)`.** A negative `n` and `NaN` are rejected, because
`0 <= NaN` is false. `n = 0` and `n = Inf` stay legal.

**The guard rejects what breaks the statistic, and nothing else.** This is the rule
[ADR 0090](0090-a-zero-gerber-band-edge-makes-a-zero-return-neutral.md) applied to the sibling
family: there `t = 0` kept a threshold with a clean meaning, and here `n = 0` and `n = Inf` keep an
exponent with a clean meaning. Each of the two narrower guards rejects a value a caller may be
passing today, and neither value is wrong.

**`n` is guarded differently from `c1`, `c2` and `c3`, and the difference is principled.** The
three thresholds are read on the scale of the data. An infinite threshold admits no observation at
all, so `assert_nonempty_nonneg_finite_val` is right for them. `n` is an exponent, and its infinite
limit is a statistic rather than an empty one. The `# Validation` section of
[`SmythBrobyCovariance`](../../src/08_Moments/06_SmythBrobyCovariance.jl) states that difference,
so a reader does not read the two guards as an inconsistency.

**`sb_delta` itself is not guarded.** It is an inner kernel called once per admitted observation of
each pair, and the estimator that reaches it has already checked `n`. A guard there would be a
per-observation cost for a condition that is a property of the estimator.

**The range of `c1` and `c2` is not narrowed.** The ticket raised a second question in the same
family: the source writes `0 < c_1 <= 1`, and the constructor enforces neither bound. The answer is
the same rule. A `c1` above one is a wider confusion zone and a `c1` of zero is an empty one; both
are well-defined admission rules, and neither answers a matrix that is not a correlation matrix.
`val_dict[:c1]` and `val_dict[:c2]` keep `0 <= c1` and `0 <= c2`, which is what the code checks.

## Consequences

- **A negative `n` and `NaN` now raise a `DomainError` at construction.** That is the whole size of
  the behaviour change. No matrix moves, because a caller who passed a negative `n` was not
  answered a correlation matrix.
- **`n = 0` and `n = Inf` are unchanged.** Both already gave a unit diagonal and a finite matrix,
  and `test_08h_smythbroby.jl` pinned that before this ADR.
- **No shipped number moves.** The default is `n = 2`, and the guard does not touch the kernel.
- **`val_dict[:sbn]` states the range once.** It reads ``0 <= n``, and the
  [`SmythBrobyCovariance`](../../src/08_Moments/06_SmythBrobyCovariance.jl) docstring interpolates
  it beside the three thresholds.
- **`test_08h_smythbroby.jl` holds the contract.** The testset
  `the constructor guards, and the severity exponent (#496, ADR 0091)` pins that `-2`, `-1e-9` and
  `NaN` raise, that `0`, `0.5`, `2` and `Inf` construct, that each of those four answers a unit
  diagonal, and that the kernel's inversion at a negative `n` is still visible one level down.
