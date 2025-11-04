"""
    abstract type AbstractDetoneEstimator <: AbstractEstimator end

Abstract supertype for all detoning estimators in PortfolioOptimisers.jl.

All concrete types representing detoning estimators (such as [`Detone`](@ref)) should subtype `AbstractDetoneEstimator`. This enables a consistent interface for detoning routines and downstream analysis.

# Related

  - [`Detone`](@ref)
  - [`detone!`](@ref)
  - [`detone`](@ref)
"""
abstract type AbstractDetoneEstimator <: AbstractEstimator end
"""
    struct Detone{T1} <: AbstractDetoneEstimator
        n::T1
    end

A concrete detoning estimator for removing the top `n` principal components (market modes) from a covariance or correlation matrix.

# Fields

  - `n`: Number of leading principal components to remove.

# Constructor

    Detone(; n::Integer = 1)

Keyword arguments correspond to the fields above.

## Validation

  - `n > 0`.

# Examples

```jldoctest
julia> Detone(; n = 2)
Detone
  n ┴ Int64: 2
```

# Related

  - [`detone!`](@ref)
  - [`detone`](@ref)
"""
struct Detone{T1} <: AbstractDetoneEstimator
    n::T1
    function Detone(n::Integer)
        @argcheck(zero(n) < n, DomainError)
        return new{typeof(n)}(n)
    end
end
function Detone(; n::Integer = 1)
    return Detone(n)
end
"""
    detone!(dt::Detone, X::AbstractMatrix; pdm::Union{Nothing, <:Posdef} = Posdef())
    detone!(::Nothing, args...)

In-place removal of the top `n` principal components (market modes) from a covariance or correlation matrix.

For covariance matrices, the function internally converts to a correlation matrix, applies the algorithm, and then rescales back to covariance.

# Arguments

  - `dt`: The estimator specifying the detoning algorithm.

      + `dt::Detone`: The top `n` principal components are removed from `X` in-place.
      + `dt::Nothing`: No-op and returns `nothing`.

  - `X`: The covariance or correlation matrix to be detoned (modified in-place).
  - `pdm`: Optional Positive definite matrix estimator. If provided, ensures the output is positive definite.

# Returns

  - `nothing`. The input matrix `X` is modified in-place.

# Validation

  - `1 <= dt.n <= size(X, 2)`.

# Examples

```jldoctest
julia> using StableRNGs

julia> rng = StableRNG(123456789);

julia> X = rand(rng, 10, 5);

julia> X = X' * X
5×5 Matrix{Float64}:
 3.29494  2.0765   1.73334  2.01524  1.77493
 2.0765   2.46967  1.39953  1.97242  2.07886
 1.73334  1.39953  1.90712  1.17071  1.30459
 2.01524  1.97242  1.17071  2.24818  1.87091
 1.77493  2.07886  1.30459  1.87091  2.44414

julia> detone!(Detone(), X)

julia> X
5×5 Matrix{Float64}:
  3.29494    -1.14673     0.0868439  -0.502106   -1.71581
 -1.14673     2.46967    -0.876289   -0.0864304   0.274663
  0.0868439  -0.876289    1.90712    -1.18851    -0.750345
 -0.502106   -0.0864304  -1.18851     2.24818    -0.0774753
 -1.71581     0.274663   -0.750345   -0.0774753   2.44414
```

# Related

  - [`detone`](@ref)
  - [`Detone`](@ref)
"""
function detone!(::Nothing, args...)
    return nothing
end
function detone!(ce::Detone, X::AbstractMatrix, pdm::Union{Nothing, <:Posdef} = Posdef())
    n = ce.n
    @argcheck(one(n) <= n <= size(X, 2),
              DomainError("1 <= n <= size(X, 2) must hold. Got\nn => $n\nsize(X, 2) => $(size(X, 2))."))
    n -= 1
    s = diag(X)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        StatsBase.cov2cor!(X, s)
    end
    vals, vecs = eigen(X)
    vals = Diagonal(vals)[(end - n):end, (end - n):end]
    vecs = vecs[:, (end - n):end]
    X .-= vecs * vals * transpose(vecs)
    X .= cov2cor(X)
    posdef!(pdm, X)
    if iscov
        StatsBase.cor2cov!(X, s)
    end
    return nothing
end
"""
    detone(dt::Detone, X::AbstractMatrix; pdm::Union{Nothing, <:Posdef} = Posdef())
    detone(::Nothing, args...)

Out-of-place version of [`detone!`](@ref).

# Related

  - [`detone!`](@ref)
  - [`Detone`](@ref)
"""
function detone(::Nothing, args...)
    return nothing
end
function detone(ce::Detone, X::AbstractMatrix, pdm::Union{Nothing, <:Posdef} = Posdef())
    X = copy(X)
    detone!(ce, X, pdm)
    return X
end

export Detone, detone, detone!
