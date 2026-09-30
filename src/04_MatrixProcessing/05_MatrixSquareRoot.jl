"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the algorithms that take a square root of a covariance matrix.

A square root ``\\mathbf{L}`` of a covariance matrix ``\\mathbf{\\Sigma}`` satisfies ``\\mathbf{L} \\mathbf{L}^{\\intercal} = \\mathbf{\\Sigma}``. Each algorithm states what it does when ``\\mathbf{\\Sigma}`` has no Cholesky factor. `nothing` in the place of an algorithm takes the plain Cholesky factor, which raises a `LinearAlgebra.PosDefException` on such a matrix.

# Interfaces

In order to implement a new square-root algorithm that works with the library, subtype `AbstractMatrixSquareRootAlgorithm` with its parameters as fields, and implement the following method:

## `matrix_square_root`

  - `matrix_square_root(alg::MyMatrixSquareRootAlgorithm, sigma::MatNum) -> MatNum`: The square root of `sigma`.

### Arguments

  - `alg`: The square-root algorithm.
  - `sigma`: The covariance matrix, `assets × assets`.

### Returns

  - `L::MatNum`: A square root with `L * L' == sigma`, `assets × assets`.

# Examples

```jldoctest
julia> struct MyMatrixSquareRootAlgorithm <: PortfolioOptimisers.AbstractMatrixSquareRootAlgorithm end

julia> function PortfolioOptimisers.matrix_square_root(::MyMatrixSquareRootAlgorithm,
                                                       sigma::PortfolioOptimisers.MatNum)
           return sqrt(sigma)
       end

julia> matrix_square_root(MyMatrixSquareRootAlgorithm(), [4.0 0.0; 0.0 9.0])
2×2 Matrix{Float64}:
 2.0  0.0
 0.0  3.0
```

# Related

  - [`RidgeCholeskySquareRoot`](@ref)
  - [`EigenFallbackSquareRoot`](@ref)
  - [`matrix_square_root`](@ref)
  - [`FactorPrior`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
abstract type AbstractMatrixSquareRootAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Adds a growing ridge to the diagonal of a covariance matrix until its Cholesky factor exists.

The first try is the plain Cholesky factor of the lower triangle. Where it fails, each later try adds a ridge to the diagonal of the symmetrised matrix, so the square root reproduces the matrix to within the last ridge. A singular matrix, such as the covariance of two factors with one return series, gets a positive definite square root.

# Mathematical definition

```math
\\begin{align}
s &= \\frac{1}{N} \\sum_{i=1}^{N} \\lvert \\Sigma_{ii} \\rvert\\,, \\\\
\\lambda_{k} &= 10^{k-1} \\max(r s, \\epsilon s)\\,, \\\\
\\mathbf{L}_{k} \\mathbf{L}_{k}^{\\intercal} &= \\frac{\\mathbf{\\Sigma} + \\mathbf{\\Sigma}^{\\intercal}}{2} + \\lambda_{k} \\mathbf{I}\\,.
\\end{align}
```

Where:

  - ``s``: Scale of the ridge, the mean absolute variance. Where it is not a finite positive number, ``s`` is the largest absolute entry of the symmetrised matrix, and at least one.
  - ``N``: Number of rows of ``\\mathbf{\\Sigma}``.
  - ``\\Sigma_{ii}``: Variance ``i`` of the covariance matrix ``\\mathbf{\\Sigma}``.
  - ``r``: The relative ridge `ridge`.
  - ``\\epsilon``: The machine epsilon of the element type, which keeps the ridge above zero when ``r`` is zero.
  - ``\\lambda_{k}``: Ridge of try ``k``, for ``k = 1, \\ldots,`` `tries`.
  - ``\\mathbf{L}_{k}``: Lower Cholesky factor of try ``k``.
  - $(math_dict[:I_identity])

The ridge is relative to ``s``, so the square root of ``c^2 \\mathbf{\\Sigma}`` is ``c`` times the square root of ``\\mathbf{\\Sigma}``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RidgeCholeskySquareRoot(;
        ridge::Number = 1e-12,
        tries::Integer = 3,
    ) -> RidgeCholeskySquareRoot

Keywords correspond to the struct's fields.

## Validation

  - `ridge` is finite and `>= 0`.
  - `tries > 0`.

# Examples

```jldoctest
julia> RidgeCholeskySquareRoot()
RidgeCholeskySquareRoot
  ridge ┼ Float64: 1.0e-12
  tries ┴ Int64: 3
```

A matrix of rank one has no Cholesky factor. The first ridge, `1e-12` of the mean variance, makes it positive definite.

```jldoctest
julia> sigma = [1.0 1.0; 1.0 1.0];

julia> L = matrix_square_root(RidgeCholeskySquareRoot(), sigma);

julia> maximum(abs, L * L' - sigma) <= 2e-12
true
```

# Related

  - [`AbstractMatrixSquareRootAlgorithm`](@ref)
  - [`EigenFallbackSquareRoot`](@ref)
  - [`matrix_square_root`](@ref)
  - [`ridge_cholesky`](@ref)
"""
@concrete struct RidgeCholeskySquareRoot <: AbstractMatrixSquareRootAlgorithm
    """
    Ridge of the first try relative to the mean absolute variance, ``r``. Each later try multiplies the ridge by ten.
    """
    ridge
    """
    Number of tries with a ridge, after the plain Cholesky factor fails.
    """
    tries
    function RidgeCholeskySquareRoot(ridge::Number, tries::Integer)
        assert_finite(ridge, :ridge)
        assert_nonneg(ridge, :ridge)
        assert_gt0(tries, :tries)
        return new{typeof(ridge), typeof(tries)}(ridge, tries)
    end
end
function RidgeCholeskySquareRoot(; ridge::Number = 1e-12, tries::Integer = 3)
    return RidgeCholeskySquareRoot(ridge, tries)
end
"""
$(DocStringExtensions.TYPEDEF)

Takes the square root of a positive semidefinite covariance matrix from its eigendecomposition where the matrix has no Cholesky factor.

A positive definite matrix gives its lower Cholesky factor. A singular positive semidefinite matrix, such as a covariance of rank one, gives the square root of its eigendecomposition, which reproduces the matrix to rounding and adds no ridge. An indefinite matrix is refused.

# Mathematical definition

```math
\\begin{align}
\\mathbf{L} &= \\begin{cases}
    \\operatorname{chol}(\\mathbf{\\Sigma}) & \\mathbf{\\Sigma} \\succ 0\\,, \\\\
    \\mathbf{V} \\max(\\mathbf{\\Lambda}, 0)^{1/2} & \\text{otherwise, with } \\mathbf{\\Sigma} = \\mathbf{V} \\mathbf{\\Lambda} \\mathbf{V}^{\\intercal}\\,.
\\end{cases}
\\end{align}
```

Where:

  - ``\\mathbf{L}``: Square root of ``\\mathbf{\\Sigma}``, with ``\\mathbf{L} \\mathbf{L}^{\\intercal} = \\mathbf{\\Sigma}``.
  - ``\\operatorname{chol}``: Lower Cholesky factor.
  - ``\\mathbf{V}``, ``\\mathbf{\\Lambda}``: Eigenvectors, and the diagonal matrix of the eigenvalues. The maximum with zero removes a negative eigenvalue of rounding.

# Constructors

    EigenFallbackSquareRoot() -> EigenFallbackSquareRoot

# Examples

```jldoctest
julia> sigma = [1.0 1.0; 1.0 1.0];

julia> L = matrix_square_root(EigenFallbackSquareRoot(), sigma);

julia> maximum(abs, L * L' - sigma) < 1e-15
true
```

# Related

  - [`AbstractMatrixSquareRootAlgorithm`](@ref)
  - [`RidgeCholeskySquareRoot`](@ref)
  - [`matrix_square_root`](@ref)
"""
struct EigenFallbackSquareRoot <: AbstractMatrixSquareRootAlgorithm end
"""
    ridge_cholesky(alg::RidgeCholeskySquareRoot, sigma::MatNum)
        -> Option{<:LinearAlgebra.Cholesky}

Factorises a covariance matrix under the ridge schedule of a [`RidgeCholeskySquareRoot`](@ref), and returns `nothing` where no try succeeds.

# Algorithm

 1. Try the plain Cholesky factorisation of the lower triangle of `sigma`. Return it where it succeeds.
 2. Symmetrise `sigma`, and take the mean absolute diagonal as the scale. Where that is not a finite positive number, take the largest absolute entry, and at least one.
 3. Add a ridge of `max(alg.ridge * scale, eps * scale)` to the diagonal, and try again. Multiply the ridge by ten after each failure, for `alg.tries` tries in all.
 4. Return `nothing` where every try fails.

# Arguments

  - `alg`: The ridge algorithm.
  - `sigma`: The covariance matrix, `assets × assets`.

# Returns

  - `chol::Option{<:LinearAlgebra.Cholesky}`: The factorisation, or `nothing` where no ridge makes the matrix factorise.

# Related

  - [`RidgeCholeskySquareRoot`](@ref)
  - [`matrix_square_root`](@ref)
  - [`safe_regime_cholesky`](@ref)
"""
function ridge_cholesky(alg::RidgeCholeskySquareRoot, sigma::MatNum)
    chol = LinearAlgebra.cholesky(LinearAlgebra.Hermitian(sigma, :L); check = false)
    if LinearAlgebra.issuccess(chol)
        return chol
    end
    S = (sigma + transpose(sigma)) / 2
    base = Statistics.mean(abs, LinearAlgebra.diag(S))
    scale = if base > zero(base) && isfinite(base)
        base
    else
        max(maximum(abs, S), one(base))
    end
    ridge = max(alg.ridge * scale, eps(typeof(scale)) * scale)
    for _ in 1:(alg.tries)
        chol = LinearAlgebra.cholesky(LinearAlgebra.Hermitian(S + ridge * LinearAlgebra.I,
                                                              :L); check = false)
        if LinearAlgebra.issuccess(chol)
            return chol
        end
        ridge *= 10
    end
    return nothing
end
"""
    matrix_square_root(::Nothing, sigma::MatNum) -> MatNum
    matrix_square_root(alg::RidgeCholeskySquareRoot, sigma::MatNum) -> MatNum
    matrix_square_root(alg::EigenFallbackSquareRoot, sigma::MatNum) -> MatNum

Takes a square root ``\\mathbf{L}`` of a covariance matrix, with ``\\mathbf{L} \\mathbf{L}^{\\intercal} = \\mathbf{\\Sigma}``.

`nothing` takes the plain lower Cholesky factor. [`RidgeCholeskySquareRoot`](@ref) takes the Cholesky factor of the matrix with the first ridge that makes it positive definite. [`EigenFallbackSquareRoot`](@ref) takes the Cholesky factor where it exists, and the square root of the eigendecomposition of a positive semidefinite matrix otherwise. A [`FactorPrior`](@ref) and a [`CrossSectionalFactorPrior`](@ref) take the square root of their factor covariance through this function.

# Arguments

  - `alg`: The square-root algorithm, or `nothing`.
  - `sigma`: The covariance matrix, `assets × assets`.

# Validation

  - `nothing`: `sigma` has a Cholesky factor. Raises a `LinearAlgebra.PosDefException`.
  - [`RidgeCholeskySquareRoot`](@ref): one try of [`ridge_cholesky`](@ref) succeeds. Raises a `LinearAlgebra.PosDefException` with the `info` of the plain factorisation.
  - [`EigenFallbackSquareRoot`](@ref): a matrix that has no Cholesky factor is Hermitian, and its smallest eigenvalue is at least ``-N \\epsilon \\max_i \\lvert \\lambda_i \\rvert``. Here ``N`` is the number of rows and ``\\epsilon`` is the machine epsilon of the element type. Raises `LinearAlgebra.PosDefException(-1)` and `LinearAlgebra.PosDefException(1)`.

# Returns

  - `L::MatNum`: The square root, `assets × assets`. It is lower triangular except for the eigen square root.

# Examples

The three members differ only on a matrix that has no Cholesky factor.

```jldoctest
julia> using LinearAlgebra

julia> sigma = [1.0 1.0; 1.0 1.0];

julia> try
           matrix_square_root(nothing, sigma)
       catch err
           err isa PosDefException
       end
true

julia> L = matrix_square_root(RidgeCholeskySquareRoot(), sigma);

julia> maximum(abs, L * L' - sigma - 1e-12I) < 1e-15
true

julia> L = matrix_square_root(EigenFallbackSquareRoot(), sigma);

julia> maximum(abs, L * L' - sigma) < 1e-15
true
```

# Related

  - [`AbstractMatrixSquareRootAlgorithm`](@ref)
  - [`RidgeCholeskySquareRoot`](@ref)
  - [`EigenFallbackSquareRoot`](@ref)
  - [`ridge_cholesky`](@ref)
"""
function matrix_square_root(::Nothing, sigma::MatNum)
    return LinearAlgebra.cholesky(sigma).L
end
function matrix_square_root(alg::RidgeCholeskySquareRoot, sigma::MatNum)
    chol = ridge_cholesky(alg, sigma)
    if isnothing(chol)
        # The ridges do not say which minor failed, and the plain factorisation does.
        F = LinearAlgebra.cholesky(LinearAlgebra.Hermitian(sigma, :L); check = false)
        throw(LinearAlgebra.PosDefException(F.info))
    end
    return chol.L
end
function matrix_square_root(::EigenFallbackSquareRoot, sigma::MatNum)
    F = LinearAlgebra.cholesky(sigma; check = false)
    if LinearAlgebra.issuccess(F)
        return F.L
    end
    @argcheck(LinearAlgebra.ishermitian(sigma), LinearAlgebra.PosDefException(-1))
    E = LinearAlgebra.eigen(LinearAlgebra.Symmetric(sigma))
    tol = -length(E.values) * eps(eltype(E.values)) * maximum(abs, E.values)
    @argcheck(minimum(E.values) >= tol, LinearAlgebra.PosDefException(1))
    return E.vectors * LinearAlgebra.Diagonal(sqrt.(max.(E.values, zero(eltype(E.values)))))
end

export RidgeCholeskySquareRoot, EigenFallbackSquareRoot, matrix_square_root
public AbstractMatrixSquareRootAlgorithm
