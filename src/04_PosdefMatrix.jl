"""
    abstract type AbstractPosdefEstimator <: AbstractEstimator end

Abstract supertype for all positive definite matrix estimator types in `PortfolioOptimisers.jl`.

All concrete types that implement positive definite matrix projection or estimation should subtype `AbstractPosdefEstimator`.

# Interfaces

In order to implement a new positive definite matrix estimator which will work seamlessly with the library, subtype `AbstractPosdefEstimator` including all necessary parameters as part of the struct, and implement the following methods:

  - `posdef!(pdm::AbstractPosdefEstimator, X::MatNum)`: In-place projection of a matrix to the nearest positive definite matrix.
  - `posdef(pdm::AbstractPosdefEstimator, X::MatNum)`: Optional out-of-place projection of a matrix to the nearest positive definite matrix.

For example, we can create a dummy positive definite estimator as follows:

```jldoctest
julia> struct MyPosdefEstimator <: PortfolioOptimisers.AbstractPosdefEstimator end

julia> function PortfolioOptimisers.posdef!(pdm::MyPosdefEstimator, X::PortfolioOptimisers.MatNum)
           # Implement your in-place PD projection logic here.
           println("Projecting to positive definite matrix in-place...")
           return nothing
       end

julia> function PortfolioOptimisers.posdef(pdm::MyPosdefEstimator, X::PortfolioOptimisers.MatNum)
           X = copy(X)
           posdef!(pdm, X)
           return X
       end

julia> posdef!(MyPosdefEstimator(), [1.0 2.0; 2.0 1.0])
Projecting to positive definite matrix in-place...

julia> posdef(MyPosdefEstimator(), [1.0 2.0; 2.0 1.0])
Projecting to positive definite matrix in-place...
2×2 Matrix{Float64}:
 1.0  2.0
 2.0  1.0
```

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

  - $(glossary[:opdm])

      + `::Posdef`: The algorithm specified in `pdm.alg` is used to project `X` to the nearest PD matrix. If `X` is already positive definite, it is left unchanged.
      + `::Nothing`: No-op.

  - $(glossary[:sigrhoX])

# Returns

  - `nothing`. The input matrix `X` is modified in-place.

# Validation

  - `X` is validated with [`assert_matrix_issquare`](@ref).
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
        StatsBase.StatsBase.cov2cor!(X, s)
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
