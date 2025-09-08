"""
    AbstractPosdefEstimator <: AbstractEstimator

Abstract supertype for all positive definite matrix estimator types in PortfolioOptimisers.jl.

All concrete types that implement positive definite matrix projection or estimation (e.g., for covariance or correlation matrices) should subtype `AbstractPosdefEstimator`. This enables a consistent interface for positive definite matrix estimation routines throughout the package.

# Related

  - [`AbstractEstimator`](@ref)
  - [`Posdef`](@ref)
  - [`posdef!`](@ref)
  - [`posdef`](@ref)
"""
abstract type AbstractPosdefEstimator <: AbstractEstimator end

"""
    struct Posdef{T1} <: AbstractPosdefEstimator
        alg::T1
    end

A concrete estimator type for projecting a matrix to the nearest positive definite (PD) matrix, typically used for covariance or correlation matrices.

# Fields

  - `alg`: The algorithm used for the nearest correlation matrix projection.

# Constructor

    Posdef(; alg::Any = NearestCorrelationMatrix.Newton)

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> using LinearAlgebra

julia> est = Posdef()
Posdef
  alg | UnionAll: NearestCorrelationMatrix.Newton
```

# Related

  - [`AbstractPosdefEstimator`](@ref)
  - [`posdef!`](@ref)
  - [`posdef`](@ref)
"""
struct Posdef{T1} <: AbstractPosdefEstimator
    alg::T1
end
function Posdef(; alg::Any = NearestCorrelationMatrix.Newton)
    return Posdef(alg)
end

"""
    posdef!(method::Posdef, X::AbstractMatrix)
    posdef!(::Nothing, args...)

In-place projection of a matrix to the nearest positive definite (PD) matrix using the specified estimator.

  - If `method` is `nothing`, this is a no-op and returns `nothing`.
  - If `method` is a [`Posdef`](@ref), the algorithm specified in `method.alg` is used to project `X` to the nearest PD matrix. If `X` is already positive definite, it is left unchanged.

For covariance matrices, the function internally converts to a correlation matrix, applies the projection, and then rescales back to covariance.

# Arguments

  - `method`: The estimator specifying the projection algorithm.
  - `X`: The matrix to be projected in-place.

# Returns

  - `nothing`. The input matrix `X` is modified in-place.

# Validation

  - If the matrix cannot be made positive definite, a warning is emitted.

# Examples

```jldoctest
julia> using LinearAlgebra

julia> est = Posdef()
Posdef
  alg | UnionAll: NearestCorrelationMatrix.Newton

julia> X = [1.0 0.9; 0.9 1.0];

julia> X[1, 2] = 2.0;  # Not PD

julia> posdef!(est, X)

julia> X
2×2 Matrix{Float64}:
 1.0  1.0
 1.0  1.0

julia> isposdef(X)
true
```

# Related

  - [`posdef`](@ref)
  - [`Posdef`](@ref)
"""
function posdef!(::Nothing, args...)
    return nothing
end
function posdef!(method::Posdef, X::AbstractMatrix)
    if isposdef(X)
        return nothing
    end
    assert_matrix_issquare(X)
    s = diag(X)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        StatsBase.cov2cor!(X, s)
    end
    nearest_cor!(X, method.alg)
    if !isposdef(X)
        @warn("Matrix could not be made positive definite.")
    end
    if iscov
        StatsBase.cor2cov!(X, s)
    end
    return nothing
end

"""
    posdef(method::Posdef, X::AbstractMatrix)
    posdef(::Nothing, args...)

Same as [`posdef!`](@ref), but returns a new matrix instead of modifying `X` in-place.

  - If `method` is `nothing`, this is a no-op and returns `nothing`.

# Examples

```jldoctest
julia> using LinearAlgebra

julia> est = Posdef();

julia> X = [1.0 2.0; 0.9 1.0];  # Not PD

julia> X_pd = posdef(est, X);

julia> isposdef(X_pd)
true

julia> X_pd
2×2 Matrix{Float64}:
 1.0  1.0
 1.0  1.0
```

# Related

  - [`posdef!`](@ref)
  - [`Posdef`](@ref)
"""
function posdef(::Nothing, args...)
    return nothing
end
function posdef(method::Posdef, X::AbstractMatrix)
    X = copy(X)
    posdef!(method, X)
    return X
end

export Posdef, posdef, posdef!
