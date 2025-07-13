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
struct Detone{T1 <: Integer} <: AbstractDetoneEstimator
n::T1
end

A concrete detoning estimator for removing the top `n` principal components (market modes) from a covariance or correlation matrix.

# Fields

  - `n::Integer`: Number of leading principal components to remove.

# Related

  - [`Detone`](@ref)
  - [`detone!`](@ref)
  - [`detone`](@ref)
"""
struct Detone{T1 <: Integer} <: AbstractDetoneEstimator
    n::T1
end

"""
    Detone(; n::Integer = 1)

Construct a [`Detone`](@ref) estimator for removing the top `n` principal components (market modes) from a covariance or correlation matrix.

# Arguments

  - `n::Integer`: Number of leading principal components to remove. Must satisfy `n ≥ 0`.

# Returns

  - `Detone`: A detoning estimator.

# Examples

```jldoctest
julia> dt = Detone(; n = 2)
Detone
  n | Int64: 2
```

# Related

  - [`detone!`](@ref)
  - [`detone`](@ref)
"""
function Detone(; n::Integer = 1)
    @smart_assert(n >= zero(n))
    return Detone{typeof(n)}(n)
end
function Base.show(io::IO, dt::Detone)
    println(io, "Detone")
    for field in fieldnames(typeof(dt))
        val = getfield(dt, field)
        print(io, lpad(string(field), 3), " ")
        println(io, "| $(typeof(val)): ", repr(val))
    end
end

"""
    detone!(dt::Detone, X::AbstractMatrix, pdm::Union{Nothing, <:PosdefEstimator} = PosdefEstimator())
    detone!(::Nothing, args...)

In-place removal of the top `n` principal components (market modes) from a covariance or correlation matrix.

  - If `dt` is `nothing`, this is a no-op and returns `nothing`.
  - If `dt` is a [`Detone`](@ref) object, the top `n` principal components are removed from `X` in-place. Optionally, a [`PosdefEstimator`](@ref) can be provided to ensure the output is positive definite.

# Arguments

  - `dt::Detone`: The detoning estimator specifying the number of components to remove.
  - `X::AbstractMatrix`: The covariance or correlation matrix to be detoned (modified in-place).
  - `pdm::Union{Nothing, <:PosdefEstimator}`: Optional positive definite matrix estimator.

# Returns

  - `nothing`. The input matrix `X` is modified in-place.

# Validation

  - If `X` is a covariance matrix, it is internally converted to a correlation matrix for detoning and then rescaled.
  - The number of components removed is validated to be within the matrix size.
  - If `pdm` is provided, the result is projected to the nearest positive definite matrix.

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
function detone!(ce::Detone, X::AbstractMatrix,
                 pdm::Union{Nothing, <:PosdefEstimator} = PosdefEstimator())
    n = ce.n
    @smart_assert(one(size(X, 1)) <= n <= size(X, 1))
    n -= 1
    s = diag(X)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        StatsBase.cov2cor!(X, s)
    end
    vals, vecs = eigen(X)
    _vals = Diagonal(vals)[(end - n):end, (end - n):end]
    _vecs = vecs[:, (end - n):end]
    X .-= _vecs * _vals * transpose(_vecs)
    X .= cov2cor(X)
    posdef!(pdm, X)
    if iscov
        StatsBase.cor2cov!(X, s)
    end
    return nothing
end

"""
    detone(dt::Detone, X::AbstractMatrix, pdm::Union{Nothing, <:PosdefEstimator} = PosdefEstimator())
    detone(::Nothing, args...)

Same as [`detone!`](@ref), but returns a new matrix instead of modifying `X` in-place.

  - If `dt` is `nothing`, this is a no-op and returns `nothing`.

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

julia> Xd = detone(Detone(), X)
5×5 Matrix{Float64}:
  3.29494    -1.14673     0.0868439  -0.502106   -1.71581
 -1.14673     2.46967    -0.876289   -0.0864304   0.274663
  0.0868439  -0.876289    1.90712    -1.18851    -0.750345
 -0.502106   -0.0864304  -1.18851     2.24818    -0.0774753
 -1.71581     0.274663   -0.750345   -0.0774753   2.44414
```

# Related

  - [`detone!`](@ref)
  - [`Detone`](@ref)
"""
function detone(::Nothing, args...)
    return nothing
end
function detone(ce::Detone, X::AbstractMatrix,
                pdm::Union{Nothing, <:PosdefEstimator} = PosdefEstimator())
    X = copy(X)
    detone!(ce, X, pdm)
    return X
end

export Detone, detone, detone!
