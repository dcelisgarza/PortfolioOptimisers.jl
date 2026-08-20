---
status: accepted
---

# The weighted moment seam densifies its observations

## Context

`robust_cov` and `robust_cor` are the package's only seam into `StatsBase`'s weighted moment
API. Each has two methods, unweighted and weighted, and each wrapped its call in a
`MethodError` retry that densified `X`:

```julia
return try
    compat_cov(ce, X, w; dims = dims, mean = mean, kwargs...)
catch err
    if !(err isa MethodError)
        rethrow()
    end
    compat_cov(ce, Matrix(X), w; dims = dims, mean = mean, kwargs...)
end
```

The retry assumes that an array type the upstream API cannot take will announce itself with a
`MethodError`. For the weighted API with a `mean`, it does not.

`StatsBase`'s weighted heads are typed on `DenseMatrix`. `cov(::SimpleCovariance, X, w; dims,
mean)` forwards four positional arguments:

```julia
covm(X, mean, w, dims; corrected = sc.corrected)
```

`StatsBase.covm(x::DenseMatrix, mean, w::AbstractWeights, dims::Int = 1; corrected)` matches
that only when `X` is dense. When it is not — a `Transpose`, an `Adjoint`, a `SubArray`, a
sparse matrix — the call falls through to

```julia
Statistics.covm(x::AbstractVecOrMat, xmean, y::AbstractVecOrMat, ymean, vardim::Int = 1; corrected)
```

which is the **cross-covariance of two matrices**. `w`, an `AbstractWeights` and therefore an
`AbstractVector`, binds to `y`; `dims` binds to `ymean`. The result is the `N × 1`
cross-covariance of `X` against the weight vector, and nothing raises. The retry never fires,
because there is no `MethodError` to catch.

Two live paths reached it:

- **`dims = 2`.** Every prior orients its input through `dims_oriented`, which returns a
    `Transpose` for `dims = 2`. So `prior(EntropyPoolingPrior(…), X; dims = 2)` failed for every
    algorithm: `ep_prior` calls `factory(pe, w0)` first, so its very first inner prior is
    already weighted.
- **A window.** `moment_window_and_weights(X, w, window)` returns `view(X, window, :)`, and a
    `SubArray` is not a `DenseArray` either.

Both surfaced far from the cause, as a `DimensionMismatch` inside `posdef!`:
`size(X, 1) == size(X, 2) must hold. Got size(X, 1) => 5, size(X, 2) => 1.` Neither is covered
by a test, which is why the suite was green.

Found on 2026-08-18 while closing ADR 0064.

## Decision

The weighted methods materialise their observation matrix before the call, rather than
recovering from a failure that does not happen:

```julia
function densify(X::DenseMatrix{<:Union{<:Number, <:JuMP.AbstractJuMPScalar}})
    return X
end
function densify(X::MatNum)
    return Matrix(X)
end
```

`robust_cov(ce, X, w; …)` and `robust_cor(ce, X, w; …)` call `compat_cov`/`compat_cor` on
`densify(X)` and drop their retry. The retry is unreachable once `X` is dense: an estimator
that raises a `MethodError` on a `Matrix` would raise the same one on the retry.

The unweighted methods keep their retry unchanged. They route into
`cov(X::AbstractMatrix; dims, mean)`, which is generic and correct on a lazy `X`, so
densifying them would buy a copy and nothing else.

`densify` returns `X` itself when it is already dense, so the common path allocates nothing.

## Consequences

A weighted covariance or correlation over a `Transpose`, a view, or a sparse matrix now
returns the same numbers as the same call over a dense copy. It previously returned an `N × 1`
matrix, or raised a `MethodError` that the retry absorbed. No stored result changes: the wrong
shape reached `posdef!` and raised there, so no assembled model or test asset could ever have
carried it. The one shape that could have passed unnoticed is a single-asset universe, where
`N × 1` and `N × N` coincide.

`dims = 2` works for the entropy pooling prior for the first time. A weighted windowed
covariance works for the first time.

The weighted path costs one copy for a lazy `X`. It already paid that copy whenever the
`MethodError` retry fired, which was every weighted call without a `mean`.

The rule is narrow on purpose: it binds the two weighted methods, not every moment verb. If a
future seam forwards positional arguments into an upstream API typed on `DenseMatrix`, it owes
the same call. Nothing in ADR 0043 (the `ObsWeights` `nothing` contract) or ADR 0064 is
contradicted.
