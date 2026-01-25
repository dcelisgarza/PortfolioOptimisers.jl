"""
    abstract type AbstractDetoneEstimator <: AbstractEstimator end

Abstract supertype for all detoning estimators in `PortfolioOptimisers.jl`.

All concrete types representing detoning estimators should subtype `AbstractDetoneEstimator`.

# Interfaces

In order to implement a new detoning estimator which will work seamlessly with the library, subtype `AbstractDetoneEstimator` with all necessary parameters as part of the struct, and implement the following methods:

  - `detone!(dt::AbstractDetoneEstimator, X::MatNum)`: In-place detoning.
  - `detone(dt::AbstractDetoneEstimator, X::MatNum)`: Optional out-of-place detoning.

## Arguments

  - $(glossary[:odt])
  - $(glossary[:sigrhoX])

## Returns

  - `X::MatNum`: The detoned input matrix `X`.

# Examples

We can create a dummy detoning estimator as follows:

```jldoctest
julia> struct MyDetoneEstimator <: PortfolioOptimisers.AbstractDetoneEstimator end

julia> function PortfolioOptimisers.detone!(dt::MyDetoneEstimator, X::PortfolioOptimisers.MatNum)
           # Implement your in-place detoning estimator here.
           println("Detoning matrix in-place...")
           return X
       end

julia> function PortfolioOptimisers.detone(dt::MyDetoneEstimator, X::PortfolioOptimisers.MatNum)
           X = copy(X)
           println("Copy X...")
           detone!(dt, X)
           return X
       end

julia> detone!(MyDetoneEstimator(), [1.0 2.0; 2.0 1.0])
Detoning matrix in-place...
2×2 Matrix{Float64}:
 1.0  2.0
 2.0  1.0

julia> detone(MyDetoneEstimator(), [1.0 2.0; 2.0 1.0])
Copy X...
Detoning matrix in-place...
2×2 Matrix{Float64}:
 1.0  2.0
 2.0  1.0
```

# Related

  - [`Detone`](@ref)
  - [`detone!`](@ref)
  - [`detone`](@ref)
"""
abstract type AbstractDetoneEstimator <: AbstractEstimator end
"""
    struct Detone{T1, T2} <: AbstractDetoneEstimator
        n::T1
        pdm::T2
    end

A concrete detoning estimator for removing the largest `n` principal components (market modes) from a covariance or correlation matrix in [`detone!`](@ref) and [`detone`](@ref).

For financial data, the leading principal components often represent market-wide movements that can obscure asset-specific signals. The `Detone` estimator allows users to specify the number of these leading components to remove, thereby enhancing the focus on idiosyncratic relationships between market members [mlp1](@cite).

Detoned matrices may not be suitable for non-clustering optimisations because it can make the matrix non-positive definite. However, they can be quite effective for clustering optimsations.

# Fields

  - `n`: Number of leading principal components to remove.
  - $(glossary[:opdm])

# Constructor

    Detone(; n::Integer = 1, pdm::Option{<:Posdef} = Posdef())

Keyword arguments correspond to the fields above.

## Validation

  - `n > 0`.

# Examples

```jldoctest
julia> Detone(; n = 2)
Detone
    n ┼ Int64: 2
  pdm ┼ Posdef
      │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │   kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`detone!`](@ref)
  - [`detone`](@ref)

# References

  - [mlp1](@cite) M. M. De Prado. *Machine learning for asset managers* (Cambridge University Press, 2020). Chapter 2.
"""
struct Detone{T1, T2} <: AbstractDetoneEstimator
    n::T1
    pdm::T2
    function Detone(n::Integer, pdm::Option{<:Posdef} = Posdef())
        @argcheck(zero(n) < n, DomainError)
        return new{typeof(n), typeof(pdm)}(n, pdm)
    end
end
function Detone(; n::Integer = 1, pdm::Option{<:Posdef} = Posdef())
    return Detone(n, pdm)
end
"""
    detone!(dt::Detone, X::MatNum)
    detone!(::Nothing, X::MatNum)

In-place removal of the top `n` principal components (market modes) from a covariance or correlation matrix.

For matrices without unit diagonal, the function converts them into correlation matrices i.e. matrices with unit diagonal, applies the algorithm, and rescales them back.

# Arguments

  - $(glossary[:odt])

      + `::Detone`: The top `n` principal components are removed from `X` in-place.
      + `::Nothing`: No-op and returns `nothing`.

  - $(glossary[:sigrhoX])

# Returns

  - `X::MatNum`: The input matrix `X` is modified in-place.

# Validation

  - `0 < dt.n <= size(X, 2)`.

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
  - [`MatNum`](@ref)
  - [`Option`](@ref)
  - [`Posdef`](@ref)

# References

  - [mlp1](@cite) M. M. De Prado. *Machine learning for asset managers* (Cambridge University Press, 2020). Chapter 2.
"""
function detone!(::Nothing, X::MatNum)
    return X
end
function detone!(de::Detone, X::MatNum)
    n = de.n
    @argcheck(zero(n) < n <= size(X, 2),
              DomainError("0 < n <= size(X, 2) must hold. Got\nn => $n\nsize(X, 2) => $(size(X, 2))."))
    n -= 1
    s = LinearAlgebra.diag(X)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        StatsBase.StatsBase.cov2cor!(X, s)
    end
    vals, vecs = LinearAlgebra.eigen(X)
    vals = LinearAlgebra.Diagonal(vals)[(end - n):end, (end - n):end]
    vecs = vecs[:, (end - n):end]
    X .-= vecs * vals * transpose(vecs)
    X .= StatsBase.cov2cor(X)
    posdef!(de.pdm, X)
    if iscov
        StatsBase.cor2cov!(X, s)
    end
    return X
end
"""
    detone(dt::Detone, X::MatNum)
    detone(::Nothing, X::MatNum)

Out-of-place version of [`detone!`](@ref).

# Related

  - [`detone!`](@ref)
  - [`Detone`](@ref)
  - [`MatNum`](@ref)
  - [`Option`](@ref)
  - [`Posdef`](@ref)

# References

  - [mlp1](@cite) M. M. De Prado. *Machine learning for asset managers* (Cambridge University Press, 2020). Chapter 2.
"""
function detone(::Nothing, X::MatNum)
    return X
end
function detone(de::Detone, X::MatNum)
    X = copy(X)
    detone!(de, X)
    return X
end

export Detone, detone, detone!
