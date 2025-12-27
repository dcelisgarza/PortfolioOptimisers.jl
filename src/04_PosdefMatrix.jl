"""
    abstract type AbstractPosdefEstimator <: AbstractEstimator end

Abstract supertype for all positive definite matrix estimator types in `PortfolioOptimisers.jl`.

All concrete types that implement positive definite matrix projection or estimation (e.g., for covariance or correlation matrices) should subtype `AbstractPosdefEstimator`. This enables a consistent interface for positive definite matrix estimation routines throughout the package.

# Related

  - [`AbstractEstimator`](@ref)
  - [`Posdef`](@ref)
  - [`posdef!`](@ref)
  - [`posdef`](@ref)
"""
abstract type AbstractPosdefEstimator <: AbstractEstimator end
"""
    struct Posdef{T1, T2} <: AbstractPosdefEstimator
        alg::T1
        kwargs::T2
    end

A concrete estimator type for projecting a matrix to the nearest positive definite matrix, typically used for co-moment matrices.

`Posdef` encapsulates all parameters required for positive definite matrix projection in [`posdef!`](@ref) and [`posdef`](@ref) to perform the nearest positive definite projection according to the estimator.

# Fields

  - `alg`: The algorithm used for the nearest correlation matrix projection.
  - `kwargs`: A named tuple of keyword arguments to be passed to the algorithm.

# Constructor

    Posdef(; alg::Any = NearestCorrelationMatrix.Newton, kwargs::NamedTuple = (;))

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> Posdef()
Posdef
     alg ┼ UnionAll: NearestCorrelationMatrix.Newton
  kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractPosdefEstimator`](@ref)
  - [`posdef!`](@ref)
  - [`posdef`](@ref)
  - [`NearestCorrelationMatrix.jl`](https://github.com/adknudson/NearestCorrelationMatrix.jl)
"""
struct Posdef{T1, T2} <: AbstractPosdefEstimator
    alg::T1
    kwargs::T2
    function Posdef(alg::Any, kwargs::NamedTuple)
        return new{typeof(alg), typeof(kwargs)}(alg, kwargs)
    end
end
function Posdef(; alg::Any = NearestCorrelationMatrix.Newton, kwargs::NamedTuple = (;))
    return Posdef(alg, kwargs)
end
"""
    posdef!(pdm::Posdef, X::MatNum)
    posdef!(::Nothing, args...)

In-place projection of a matrix to the nearest positive definite matrix using the specified estimator.

For matrices without unit diagonal, the function converts them into correlation matrices i.e. matrices with unit diagonal, applies the algorithm, and rescales them back.

# Arguments

  - `pdm`: The estimator specifying the positive definite projection algorithm.

      + `pdm::Posdef`: The algorithm specified in `pdm.alg` is used to project `X` to the nearest PD matrix. If `X` is already positive definite, it is left unchanged.
      + `pdm::Nothing`: No-op.

  - `X`: The matrix to be projected in-place.

# Returns

  - `nothing`. The input matrix `X` is modified in-place.

# Validation

  - If the matrix cannot be made positive definite, a warning is emitted.

# Examples

```jldoctest
julia> using LinearAlgebra

julia> est = Posdef();

julia> X = [1.0 0.9; 0.9 1.0];

julia> X[1, 2] = 2.0;  # Not PD

julia> posdef!(est, X)

julia> X
2×2 Matrix{Float64}:
 1.0  1.0
 1.0  1.0

julia> LinearAlgebra.isposdef(X)
true
```

# Related

  - [`posdef`](@ref)
  - [`Posdef`](@ref)
  - [`MatNum`](@ref)
"""
function posdef!(::Nothing, args...)
    return nothing
end
function posdef!(pdm::Posdef, X::MatNum)
    if LinearAlgebra.isposdef(X)
        return nothing
    end
    assert_matrix_issquare(X, :X)
    s = LinearAlgebra.diag(X)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        StatsBase.cov2cor!(X, s)
    end
    NearestCorrelationMatrix.nearest_cor!(X, pdm.alg; pdm.kwargs...)
    if !LinearAlgebra.isposdef(X)
        @warn("Matrix could not be made positive definite.")
    end
    if iscov
        StatsBase.cor2cov!(X, s)
    end
    return nothing
end
"""
    posdef(pdm::Posdef, X::MatNum)
    posdef(::Nothing, args...)

Out-of-place version of [`posdef!`](@ref).

# Related

  - [`posdef!`](@ref)
  - [`Posdef`](@ref)
  - [`MatNum`](@ref)
"""
function posdef(::Nothing, args...)
    return nothing
end
function posdef(pdm::Posdef, X::MatNum)
    X = copy(X)
    posdef!(pdm, X)
    return X
end

export Posdef, posdef, posdef!
