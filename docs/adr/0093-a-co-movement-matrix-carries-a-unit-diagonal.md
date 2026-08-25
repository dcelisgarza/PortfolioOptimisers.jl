---
status: accepted
---

# A co-movement matrix carries a unit diagonal

## Context

[#495](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/495), a child of
[#417](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/417), opened on a raise from
the documentation ticket
[#456](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/456).

An asset whose every return sits inside its own noise zone crosses no threshold. `comovement_ratio`
then finds a zero denominator for **every** pair that asset belongs to and returns the guarded
`zero(T)`. That is the documented behaviour of the statistic, and it is the right answer for an
**off-diagonal** entry: two assets, one of which never moved, have no measured co-movement.

The guard also reached the **diagonal**, where a correlation is one by definition. The matrix then
carried a zero on its diagonal, so it was not a correlation matrix. `posdef!` divides by the square
root of the diagonal, makes a `NaN`, and raises:

```text
ArgumentError: matrix contains Infs or NaNs
```

The message names neither the asset nor the cause, so a caller saw an opaque failure with nothing
to act on.

### The reach

The reduction is shared, so the defect was shared. Every branch of both measured families failed on
the same sample:

| Estimator | `Gerber0` | `Gerber1` | `Gerber2` |
| --- | --- | --- | --- |
| `GerberCovariance` | raised | raised | raised |
| `GerberIQCovariance` | raised | raised | raised |

`SmythBrobyCovariance` shares `comovement_ratio` and `posdef!` and reaches the zero the same way,
through an asset whose every return stays inside the confusion zone.

### The cause

`comovement_ratio` reduces one pair's accumulators and does not know whether the pair is `(i, i)`.
It therefore cannot separate the two cases. The `2` variants reach the same zero by the other
route: `standardise_comovement!` clamps the root of the diagonal from below by `sqrt(eps)`, so a
zero diagonal entry divides to zero rather than to `NaN`, and the zero survives.

### The two readings

[#495](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/495) named both.

 1. **Write one onto the diagonal** after the reduction and before `posdef!`. The matrix is then a
    formal correlation matrix in every sample, and an asset that crosses no threshold reads as
    uncorrelated with every other asset.
 2. **Raise a clear error** naming the asset and the threshold that excluded it.

## Decision

**A co-movement matrix carries a unit diagonal, in every sample.** A new function,
[`comovement_unit_diagonal!`](../../src/08_Moments/06_SmythBrobyCovariance.jl), writes it:

```julia
function comovement_unit_diagonal!(rho::AbstractMatrix)
    o = one(eltype(rho))
    for i in axes(rho, 1)
        if iszero(rho[i, i])
            rho[i, i] = o
        end
    end
    return nothing
end
```

Five sites call it, each after the reduction and before `posdef!`: the three `gerber` methods of
[`GerberCovariance`](../../src/08_Moments/05_GerberCovariance.jl), `smythbroby`, and `gerber_IQ`.

**Reading 1, not reading 2.** A zero row is the honest reading of a sample in which one asset never
moved, and the caller who asked for a correlation matrix gets one. Reading 2 rejects a sample that
the statistic answers correctly everywhere except on one entry it defines by convention.

**The write is guarded by `iszero`, and the guard is load-bearing.** A zero is the only diagonal
entry the reduction can produce that is not already one, so the guard reaches every case the
defect names:

- `Gerber0` and its siblings: on `(i, i)` the discordant count is zero, so `(p - 0) / (p + 0)` is
    exactly one whenever `p` is not zero.
- `Gerber1` and its siblings: on `(i, i)` the neutral term is zero too. A pair of one asset with
    itself is neutral on the same observations on both sides, so the count of observations on which
    exactly one side is neutral is zero. The reduction is again `p / p`, and again exactly one.
- `Gerber2` and its siblings: `standardise_comovement!` divides the diagonal by its own square
    root twice, which is one to within a **unit in the last place** rather than exactly one.

The first draft of this ADR wrote the diagonal unconditionally, on the argument that a guard states
the same rule twice. Measurement rejected that draft. `posdef!` reads

```julia
s = LinearAlgebra.diag(X)
iscov = any(!isone, s)
```

with an **exact** `isone` test, and the covariance branch it selects runs `StatsBase.cov2cor!`,
which clamps every off-diagonal entry into `[-1, 1]`. Writing an exact one over a `Gerber2` diagonal
that already reads as one flipped that test, dropped the clamp, and moved the pinned covariance of
`test_08_moments.jl` for `GerberIQCovariance(; kind = PartialGerberIQ(), alg = Gerber2())` by
`0.0355` on the `SP500` sample. That sample carries no degenerate asset, so the move had nothing to
do with this defect. The guard keeps the fix to the defect.

**The unit diagonal is written before `posdef!`, not after.** `posdef!` reads the diagonal to decide
whether it holds a correlation matrix or a covariance matrix, and it divides by its square root.
The repair must see a diagonal it can use.

## Consequences

- **`ArgumentError: matrix contains Infs or NaNs` is gone** from this route, for all three variants
  of all four families of the Gerber lineage.
- **A caller who passes `pdm = nothing` sees one changed entry per degenerate asset.** The diagonal
  entry moves from `0` to `1`. No other entry moves. That is the whole size of the behaviour
  change, and it applies only to a sample that carries such an asset.
- **A zero row is still a zero row.** The statistic keeps saying that the asset has no measured
  co-movement with any other. The fix does not invent a correlation; it writes the one entry that
  is a definition rather than a measurement.
- **The covariance path is unchanged in its answer and clearer in its reason.** `cor2cov!` already
  wrote the variance onto the diagonal whatever the correlation carried, so the covariance diagonal
  was right even while the correlation diagonal was wrong. It now follows from a unit correlation
  diagonal rather than in spite of a zero one.
- **No pinned number of the library moves.** The guard writes nothing on a sample in which every
  asset votes at least once, and every pinned regression sample is such a sample.
- **Two findings that the measurement exposed are left open, and neither is this defect.** The
  `Gerber2` normalisation does not bound its own statistic: eleven off-diagonal entries of the
  `PartialGerberIQ` matrix above have a magnitude greater than one, the largest `1.224`. Whether
  those entries are clamped is then decided by `posdef!`'s exact `isone` test on a diagonal that
  differs from one by a unit in the last place. Both belong to the same ticket, and neither is
  fixed here.
- **The guard also keeps this ADR from masking a neighbouring defect.**
  [#498](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/498) reported a `Gerber1`
  diagonal that is not one, and its cause was an inflated neutral count at `c = 0` that also moves
  the off-diagonal entries. ADR 0090 fixed it on its own terms, and its diagonal is now exactly one
  by the reduction. An unconditional write would have hidden that symptom while leaving the cause;
  the `iszero` guard writes nothing there.
- **`test_08_moments.jl` holds the contract.** The testset `Co-movement unit diagonal (#495)` pins
  that a degenerate asset gives a zero row and a unit diagonal, that the default `pdm` no longer
  raises, and that a sample with no degenerate asset answers the same matrix as before.
