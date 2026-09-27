"""
$(DocStringExtensions.TYPEDEF)

Estimates the covariance of a short window as a rank-one matrix along the principal direction of its price relatives.

Lai, Tan, Wu and Fang (2020) built the estimate for short-term portfolio optimisation, where the window holds a few observations of many assets. The matrix is positive semidefinite and singular, so it has no Cholesky factor. A consumer reads it as a quadratic form, as [`Variance`](@ref) does under [`QuadRiskExpr`](@ref).

# Mathematical definition

```math
\\begin{align}
\\mathbf{X}^\\intercal \\mathbf{X} \\boldsymbol{u}_1 &= \\theta_1 \\boldsymbol{u}_1\\,, \\\\
\\hat{\\mathbf{\\Sigma}}_{\\mathrm{MP}} &= \\mathbf{X}^\\intercal \\left( \\mathbf{I} - \\frac{1}{w} \\boldsymbol{1} \\boldsymbol{1}^\\intercal \\right) \\mathbf{X}\\,, \\\\
\\zeta_1^\\star &= \\theta_1 \\left( \\frac{\\operatorname{tr}(\\hat{\\mathbf{\\Sigma}}_{\\mathrm{MP}})}{N (w - 1)} \\right)^{-1/2}\\,, \\\\
\\hat{\\mathbf{\\Sigma}}_{\\mathrm{RO}} &= \\zeta_1^\\star \\boldsymbol{u}_1 \\boldsymbol{u}_1^\\intercal\\,.
\\end{align}
```

Where:

  - ``\\mathbf{X}``: ``w \\times N`` window of price relatives, one row per observation, not centred.
  - ``w``: Number of observations in the window.
  - $(math_dict[:N])
  - ``\\theta_1``: Largest eigenvalue of ``\\mathbf{X}^\\intercal \\mathbf{X}``, the square of the largest singular value of ``\\mathbf{X}``.
  - ``\\boldsymbol{u}_1``: Unit eigenvector of ``\\mathbf{X}^\\intercal \\mathbf{X}`` at ``\\theta_1``, the principal right singular vector of ``\\mathbf{X}``.
  - ``\\hat{\\mathbf{\\Sigma}}_{\\mathrm{MP}}``: Scatter matrix of the window, the sum of the outer products of its centred rows.
  - $(math_dict[:I_identity])
  - ``\\zeta_1^\\star``: Energy of the estimate along ``\\boldsymbol{u}_1``.
  - $(math_dict[:Sigma_hat_RO])

This is Algorithm 1 of the paper. Its equations 42 and 49 write the trace of ``\\hat{\\mathbf{\\Sigma}}_{\\mathrm{MP}}`` in the basis of the singular vectors of ``\\mathbf{X}``, as ``\\operatorname{tr}(D)``, which is the same number. The quotient ``\\operatorname{tr}(\\hat{\\mathbf{\\Sigma}}_{\\mathrm{MP}}) / (N (w - 1))`` is the mean of the ``N`` sample variances of the window. So ``\\zeta_1^\\star`` is ``\\theta_1`` over the root of the mean variance. That value minimises the trade-off of equation 48 between the magnitude of ``\\theta_1 \\boldsymbol{u}_1 \\boldsymbol{u}_1^\\intercal`` and the magnitude of ``\\hat{\\mathbf{\\Sigma}}_{\\mathrm{MP}}``.

Equation 47 divides the trace of a matrix by its rank, and equation 49 puts the rank of ``\\hat{\\mathbf{\\Sigma}}_{\\mathrm{MP}}`` at ``w - 1``. That is the rank of a short window, ``w \\leq N + 1``, which is the case the paper studies. On a longer window the rank is at most ``N``, but the definition keeps ``w - 1``, as Algorithm 1 and the authors' code do. When every row of the window is the same, ``\\operatorname{tr}(\\hat{\\mathbf{\\Sigma}}_{\\mathrm{MP}}) = 0`` and ``\\zeta_1^\\star`` is not defined.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RankOneCovariance(; shift::Real = 1) -> RankOneCovariance

Keywords correspond to the struct's fields.

## Validation

  - `isfinite(shift)`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> RankOneCovariance()
RankOneCovariance
  shift ┴ Int64: 1
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`ShortTermLossControlPortfolio`](@ref): the programme that reads the estimate.
  - [`Variance`](@ref)
  - [`QuadRiskExpr`](@ref)

# References

  - $(ref_dict[:lai2020spolc]) Algorithm 1 and equations 42 to 49.
"""
struct RankOneCovariance{T1 <: Real} <: AbstractCovarianceEstimator
    """
    Number added to every entry of the returns to form the price relatives ``\\mathbf{X}``. The default `1` turns returns into the price relatives that the paper reads, and `0` decomposes the rows as given.
    """
    shift::T1
    function RankOneCovariance(shift::Real)
        @argcheck(isfinite(shift), DomainError(shift, "shift must be finite"))
        return new{typeof(shift)}(shift)
    end
end
function RankOneCovariance(; shift::Real = 1)::RankOneCovariance
    return RankOneCovariance(shift)
end
"""
    Statistics.cov(ce::RankOneCovariance, X::MatNum; dims::Int = 1, kwargs...)

Computes the rank-one covariance of a window of returns, `assets × assets`.

[`RankOneCovariance`](@ref) states the mathematics.

# Algorithm

 1. Orient `X` to `observations × assets` with [`dims_oriented`](@ref).
 2. Add `ce.shift` to every entry of `X` to form the price relatives `Xs`.
 3. Subtract the column means from `Xs`, and sum the squares of the result. The sum `trD` is the trace of the scatter matrix.
 4. Compute the singular value decomposition of `Xs`. Take `theta`, the square of the largest singular value, and `u`, its right singular vector.
 5. Compute `zeta = theta * sqrt(N * (w - 1) / trD)`. When every row of `Xs` is the same, or when `trD` is zero, `zeta` is zero. The method compares the rows, because the column means of equal rows carry round-off, and `trD` of a constant window is often a tiny positive number.
 6. Return `zeta .* (u .* transpose(u))`. The method forms the outer product entry by entry, so the matrix is symmetric to the bit.

# Arguments

  - `ce`: The rank-one covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments. The method ignores them, because the estimate reads no mean and no observation weights.

# Validation

  - $(val_dict[:dims])
  - `size(X, 1) >= 2` after orientation. An `ArgumentError` is thrown otherwise, because one row has no centred energy.

# Returns

  - `sigma::Matrix`: The rank-one covariance matrix, positive semidefinite and symmetric. It is the zero matrix when every row of the window is the same.

# Related

  - [`RankOneCovariance`](@ref)
"""
function Statistics.cov(ce::RankOneCovariance, X::MatNum; dims::Int = 1, kwargs...)
    X = dims_oriented(dims, X)
    w, N = size(X)
    @argcheck(w >= 2,
              ArgumentError("the rank-one covariance needs at least two observations, got $w: the centred energy of one row is zero"))
    Xs = X .+ ce.shift
    # The centred energy `tr(D)` is the Frobenius norm of the column-centred window, which
    # is what `Ξ² − (1/w) Ξ Vᵀ 1 1ᵀ V Ξ` traces to.
    Xc = Xs .- Statistics.mean(Xs; dims = 1)
    trD = sum(abs2, Xc)
    F = LinearAlgebra.svd(Xs)
    theta = abs2(F.S[1])
    u = F.V[:, 1]
    # The column means of equal rows carry round-off, so `trD` of a constant window is often
    # a tiny positive number and not zero. The row test is exact.
    zeta = if iszero(trD) || allequal(eachrow(Xs))
        zero(theta)
    else
        theta * sqrt(N * (w - 1) / trD)
    end
    # Entry by entry, so the product is symmetric to the bit and a consumer's Hermitian check
    # holds; a matrix product need not be.
    return zeta .* (u .* transpose(u))
end
"""
    Statistics.cor(ce::RankOneCovariance, X::MatNum; dims::Int = 1, kwargs...)

Computes the correlation matrix of the rank-one covariance of a window of returns, `assets × assets`.

# Mathematical definition

```math
\\begin{align}
\\rho_{ij} &= \\frac{\\hat{\\Sigma}_{ij}}{\\sqrt{\\hat{\\Sigma}_{ii} \\hat{\\Sigma}_{jj}}} = \\operatorname{sign}(u_{1i} u_{1j})\\,.
\\end{align}
```

Where:

  - ``\\rho_{ij}``: Correlation between assets ``i`` and ``j``.
  - ``\\hat{\\Sigma}_{ij}``: Entry of ``\\hat{\\mathbf{\\Sigma}}_{\\mathrm{RO}}`` in row ``i`` and column ``j``.
  - ``u_{1i}``: Entry ``i`` of the principal vector ``\\boldsymbol{u}_1`` of [`RankOneCovariance`](@ref).

The second equality holds when ``\\zeta_1^\\star > 0`` and both assets have a non-zero entry in ``\\boldsymbol{u}_1``. Otherwise the variance of one of the two assets is zero, and ``\\rho_{ij}`` is not defined.

# Algorithm

 1. Compute `sigma` with the `cov` method of [`RankOneCovariance`](@ref).
 2. Take the sign of every entry of `sigma`.
 3. Set the diagonal to one.

# Arguments

  - `ce`: The rank-one covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments. The method passes them to `cov`, which ignores them.

# Validation

  - Every check of the `cov` method applies.

# Returns

  - `rho::Matrix`: The correlation matrix, with ones on the diagonal and ``\\pm 1`` off it. An entry whose correlation is not defined is zero.

# Related

  - [`RankOneCovariance`](@ref)
"""
function Statistics.cor(ce::RankOneCovariance, X::MatNum; dims::Int = 1, kwargs...)
    sigma = Statistics.cov(ce, X; dims = dims, kwargs...)
    C = sign.(sigma)
    for i in axes(C, 1)
        C[i, i] = one(eltype(C))
    end
    return C
end
export RankOneCovariance
