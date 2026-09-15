"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all positive definite matrix estimator types.

All concrete and/or abstract types that implement positive definite matrix projection or estimation should be subtypes of `AbstractPosdefEstimator`.

# Interfaces

In order to implement a new positive definite matrix estimator which will work seamlessly with the library, subtype `AbstractPosdefEstimator` with all necessary parameters as part of the struct, and implement the following methods:

  - `posdef!(pdm::AbstractPosdefEstimator, X::MatNum) -> MatNum`: In-place projection of a matrix to the nearest positive definite matrix.
  - `posdef(pdm::AbstractPosdefEstimator, X::MatNum) -> MatNum`: Optional out-of-place projection of a matrix to the nearest positive definite matrix. A fallback method copies `X` and calls `posdef!`, so it is only needed if the copy can be avoided.

## Arguments

  - $(arg_dict[:pdm])
  - $(arg_dict[:sigrhoX])

## Returns

  - `X::MatNum`: The projected input matrix `X`.

# Examples

We can create a dummy positive definite estimator as follows:

```jldoctest
julia> struct MyPosdefEstimator <: PortfolioOptimisers.AbstractPosdefEstimator end

julia> function PortfolioOptimisers.posdef!(pdm::MyPosdefEstimator, X::PortfolioOptimisers.MatNum)
           # Implement your in-place PD projection logic here.
           println(\"Projecting to positive definite matrix in-place...\")
           return X
       end

julia> function PortfolioOptimisers.posdef(pdm::MyPosdefEstimator, X::PortfolioOptimisers.MatNum)
           X = copy(X)
           println(\"Copy X...\")
           posdef!(pdm, X)
           return X
       end

julia> posdef!(MyPosdefEstimator(), [1.0 2.0; 2.0 1.0])
Projecting to positive definite matrix in-place...
2×2 Matrix{Float64}:
 1.0  2.0
 2.0  1.0

julia> posdef(MyPosdefEstimator(), [1.0 2.0; 2.0 1.0])
Copy X...
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
$(DocStringExtensions.TYPEDEF)

Projects a matrix to the nearest positive definite matrix, typically used for co-moment matrices.

`Posdef` encapsulates all parameters required for positive definite matrix projection in [`posdef!`](@ref) and [`posdef`](@ref) to perform the nearest positive definite projection according to the estimator.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Posdef(;
        alg::Any = NearestCorrelationMatrix.Newton,
        kwargs::NamedTuple = (;),
    ) -> Posdef

Keywords correspond to the struct's fields.

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

# References

  - $(ref_dict[:higham2002])
  - $(ref_dict[:qisun2006])
"""
@concrete struct Posdef <: AbstractPosdefEstimator
    """
    The algorithm used for the nearest correlation matrix projection.
    """
    alg
    """
    A named tuple of keyword arguments to be passed to the algorithm.
    """
    kwargs
    function Posdef(alg::Any, kwargs::NamedTuple)::Posdef
        return new{typeof(alg), typeof(kwargs)}(alg, kwargs)
    end
end
function Posdef(; alg::Any = NearestCorrelationMatrix.Newton,
                kwargs::NamedTuple = (;))::Posdef
    return Posdef(alg, kwargs)
end
"""
    posdef!(pdm::Option{<:AbstractPosdefEstimator}, X::MatNum) -> MatNum

In-place projection of a matrix to the nearest positive definite matrix using the specified estimator.

For matrices without unit diagonal, the function converts them into correlation matrices i.e. matrices with unit diagonal, applies the algorithm, and rescales them back.

# Mathematical definition

Solves the nearest correlation matrix problem:

```math
\\begin{align}
\\hat{\\mathbf{C}} &= \\underset{\\mathbf{Y} \\succeq 0,\\; Y_{ii} = 1}{\\arg\\min} \\lVert \\mathbf{C} - \\mathbf{Y} \\rVert_F\\,.
\\end{align}
```

Where:

  - ``\\hat{\\mathbf{C}}``: Nearest positive semidefinite correlation matrix.
  - ``\\mathbf{C}``: Input correlation matrix.
  - ``\\mathbf{Y}``: Feasible correlation matrix (positive semidefinite, unit diagonal).
  - ``\\lVert \\cdot \\rVert_F``: Frobenius norm.

For covariance matrices, first standardise ``\\mathbf{C} = \\mathrm{diag}(\\mathbf{\\Sigma})^{-1/2} \\mathbf{\\Sigma}\\, \\mathrm{diag}(\\mathbf{\\Sigma})^{-1/2}``, project, then rescale back.

# Algorithm

 1. Return `X` unchanged when it is already positive definite. The projection has nothing to do, and the check runs before every other step.
 2. Check that `X` is square.
 3. Read the diagonal of `X` into `s`. When any entry of `s` is not one, `X` is a covariance matrix: replace `s` with its square roots and convert `X` to a correlation matrix with `StatsBase.cov2cor!`. The test is `any(!isone, s)`, so it is the value of the diagonal that decides, never the type of `X`.
 4. Project `X` onto the nearest correlation matrix with `NearestCorrelationMatrix.nearest_cor!`, under the algorithm `pdm.alg` and the keyword arguments `pdm.kwargs`.
 5. Warn when the projected `X` is still not positive definite. `X` is returned either way, so the caller must check the result when it cannot tolerate an unrepaired matrix.
 6. When step 3 converted a covariance matrix, convert `X` back with `StatsBase.cor2cov!`. The standard deviations are the ones read in step 3, so the original diagonal returns exactly.

# Arguments

  - $(arg_dict[:opdm])

      + `::Posdef`: The algorithm specified in `pdm.alg` is used to project `X` to the nearest PD matrix. If `X` is already positive definite, it is left unchanged.
      + `::Nothing`: No-op.

  - $(arg_dict[:sigrhoX])

# Validation

  - `X` is validated with [`assert_matrix_issquare`](@ref). The check runs after the early return, so it is reached only when `X` is not already positive definite.

# Returns

  - `X::MatNum`: The input matrix `X` modified in-place.

# Examples

```jldoctest
julia> using LinearAlgebra

julia> est = Posdef();

julia> X = [1.0 0.9; 0.9 1.0];

julia> X[1, 2] = 2.0;  # Not PD

julia> posdef!(est, X)
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
function posdef!(::Nothing, X::MatNum)::MatNum
    return X
end
function posdef!(pdm::Posdef, X::MatNum)
    if LinearAlgebra.isposdef(X)
        return X
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
    return X
end
"""
    posdef(pdm::Option{<:AbstractPosdefEstimator}, X::MatNum) -> MatNum

Out-of-place version of [`posdef!`](@ref).

# Algorithm

 1. Copy `X`.
 2. Apply [`posdef!`](@ref) to the copy, and return it. The input is never modified.

# Arguments

  - $(arg_dict[:opdm])
  - $(arg_dict[:sigrhoX])

# Returns

  - `X::MatNum`: A new matrix equal to the nearest positive definite projection of the input.

# Examples

```jldoctest
julia> using LinearAlgebra

julia> X = [1.0 2.0; 2.0 1.0];

julia> Xpd = posdef(Posdef(), X);

julia> LinearAlgebra.isposdef(Xpd)
true
```

# Related

  - [`posdef!`](@ref)
  - [`Posdef`](@ref)
  - [`MatNum`](@ref)
"""
function posdef(::Nothing, X::MatNum)::MatNum
    return X
end
function posdef(pdm::AbstractPosdefEstimator, X::MatNum)
    X = copy(X)
    posdef!(pdm, X)
    return X
end

export Posdef, posdef, posdef!
