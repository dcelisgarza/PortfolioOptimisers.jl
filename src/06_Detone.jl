"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all detoning estimators in `PortfolioOptimisers.jl`.

All concrete and/or abstract types representing detoning estimators should be subtypes of `AbstractDetoneEstimator`.

# Interfaces

In order to implement a new detoning estimator which will work seamlessly with the library, subtype `AbstractDetoneEstimator` with all necessary parameters as part of the struct, and implement the following methods:

  - `detone!(dt::AbstractDetoneEstimator, X::MatNum) -> MatNum`: In-place detoning.
  - `detone(dt::AbstractDetoneEstimator, X::MatNum) -> MatNum`: Optional out-of-place detoning.

## Arguments

  - $(arg_dict[:odt])
  - $(arg_dict[:sigrhoX])

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
$(DocStringExtensions.TYPEDEF)

A concrete detoning estimator for removing the largest `n` principal components (market modes) from a covariance or correlation matrix in [`detone!`](@ref) and [`detone`](@ref).

For financial data, the leading principal components often represent market-wide movements that can obscure asset-specific signals. The `Detone` estimator allows users to specify the number of these leading components to remove, thereby enhancing the focus on idiosyncratic relationships between market members [mlp1](@cite).

Detoned matrices may not be suitable for non-clustering optimisations because it can make the matrix non-positive definite. However, they can be quite effective for clustering optimsations.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Detone(;
        pdm::Option{<:Posdef} = Posdef(),
        n::Integer = 1,
    ) -> Detone

Keywords correspond to the struct's fields.

## Validation

  - `n > 0`.

# Examples

```jldoctest
julia> Detone(; n = 2)
Detone
  pdm ┼ Posdef
      │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │   kwargs ┴ @NamedTuple{}: NamedTuple()
    n ┴ Int64: 2
```

# Related

  - [`detone!`](@ref)
  - [`detone`](@ref)

# References

  - [mlp1](@cite) M. M. De Prado. *Machine learning for asset managers* (Cambridge University Press, 2020). Chapter 2.
"""
@concrete struct Detone <: AbstractDetoneEstimator
    "$(field_dict[:opdm])"
    pdm
    "Number of leading principal components to remove."
    n
    function Detone(pdm::Option{<:Posdef}, n::Integer)
        @argcheck(zero(n) < n, DomainError)
        return new{typeof(pdm), typeof(n)}(pdm, n)
    end
end
function Detone(; pdm::Option{<:Posdef} = Posdef(), n::Integer = 1)
    return Detone(pdm, n)
end
"""
    detone!(dt::Option{<:Detone}, X::MatNum) -> MatNum

In-place removal of the top `n` principal components (market modes) from a covariance or correlation matrix.

For matrices without unit diagonal, the function converts them into correlation matrices i.e. matrices with unit diagonal, applies the algorithm, and rescales them back.

# Mathematical definition

The detoned matrix removes the ``n`` largest eigenmodes:

```math
\\begin{align}
\\tilde{\\mathbf{X}} &= \\mathbf{X} - \\sum_{k=N-n+1}^{N} \\lambda_k \\boldsymbol{v}_k \\boldsymbol{v}_k^\\intercal\\,.
\\end{align}
```

Where:

  - ``\\tilde{\\mathbf{X}}``: Detoned matrix.
  - ``\\mathbf{X}``: Original correlation or covariance matrix.
  - ``\\lambda_k``: ``k``-th largest eigenvalue of ``\\mathbf{X}``.
  - ``\\boldsymbol{v}_k``: ``k``-th largest eigenvector of ``\\mathbf{X}``.
  - ``n``: Number of eigenmodes (market modes) to remove.
  - $(math_dict[:N])

# Arguments

  - $(arg_dict[:odt])

      + `::Detone`: The top `n` principal components are removed from `X` in-place.
      + `::Nothing`: No-op.

  - $(arg_dict[:sigrhoX])

# Validation

  - `0 < dt.n <= size(X, 2)`.

# Returns

  - `X::MatNum`: The input matrix `X` is modified in-place.

# Details

  - Asserts the number of elements to remove is within the valid range.
  - If `X` is not a correlation matrix, it is converted to one before applying the algorithm.
  - Performs an eigenvector decomposition of `X`.
  - Removes the top `n` principal components (market modes) from the eigenvalues and eigenvectors of `X`.
  - Reconstructs the correlation matrix `X` in-place from the modified eigenvalues `vals` and eigenvectors `vecs`.
  - If `X` was not originally a correlation matrix, it is converted back.
  - Returns `X`.

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
function detone!(::Nothing, X::MatNum)::MatNum
    return X
end
function detone!(dt::Detone, X::MatNum)
    n = dt.n
    @argcheck(zero(n) < n <= size(X, 2),
              DomainError("0 < n <= size(X, 2) must hold. Got\nn => $n\nsize(X, 2) => $(size(X, 2))."))
    n -= 1
    s = LinearAlgebra.diag(X)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        StatsBase.cov2cor!(X, s)
    end
    vals, vecs = LinearAlgebra.eigen(X)
    vals = LinearAlgebra.Diagonal(vals)[(end - n):end, (end - n):end]
    vecs = vecs[:, (end - n):end]
    X .-= vecs * vals * transpose(vecs)
    X .= StatsBase.cov2cor(X)
    posdef!(dt.pdm, X)
    if iscov
        StatsBase.cor2cov!(X, s)
    end
    return X
end
"""
    detone(dt::Option{<:Detone}, X::MatNum)

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
function detone(::Nothing, X::MatNum)::MatNum
    return X
end
function detone(dt::Detone, X::MatNum)
    X = copy(X)
    detone!(dt, X)
    return X
end

export Detone, detone, detone!
