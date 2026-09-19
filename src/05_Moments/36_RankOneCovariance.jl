"""
$(DocStringExtensions.TYPEDEF)

The rank-one covariance estimate of Lai, Tan, Wu and Fang (2020): the principal spectral component of a short window of price relatives, scaled so that its energy trades off against the centred window's.

# Mathematical definition

With ``X \\in \\mathbb{R}^{w \\times N}`` the window of price relatives, uncentred, and its singular value decomposition ``X = V \\Xi U^\\intercal``,

```math
\\begin{align}
\\theta_1 &= \\Xi_{11}^2\\,,\\quad
D = \\Xi^2 - \\frac{1}{w} \\Xi V^\\intercal \\boldsymbol{1} \\boldsymbol{1}^\\intercal V \\Xi\\,,\\\\
\\zeta_1^\\star &= \\theta_1 \\left( \\frac{\\operatorname{tr}(D)}{N (w - 1)} \\right)^{-1/2}\\,,\\quad
\\hat{\\Sigma}_{\\mathrm{RO}} = \\zeta_1^\\star \\boldsymbol{u}_1 \\boldsymbol{u}_1^\\intercal\\,,
\\end{align}
```

where ``\\boldsymbol{u}_1`` is the principal right singular vector of ``X``, the eigenvector of ``X^\\intercal X`` at its largest eigenvalue ``\\theta_1``. ``\\operatorname{tr}(D)`` is the total energy of the column-centred window, ``\\lVert X - \\tfrac{1}{w} \\boldsymbol{1} \\boldsymbol{1}^\\intercal X \\rVert_F^2``, which is what the sample covariance's nuclear norm carries, so ``\\zeta_1^\\star`` is the paper's trade-off between the principal tangent direction of the uncentred Gram and the sample covariance's magnitude (its equations 42, 46 and 49). The matrix has rank one: it is positive semidefinite and singular, so a consumer that factorises it takes a quadratic form rather than a Cholesky factor — [`Variance`](@ref) under [`QuadRiskExpr`](@ref).

The paper decomposes price relatives, and the library hands a covariance estimator returns, so `shift` is added to every entry before the decomposition: one, the default, turns returns into the price relatives the paper reads, and zero decomposes the rows as given. The estimator reads `w` observations of `N` assets and needs at least two rows, since the centred energy of one row is zero.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RankOneCovariance(; shift::Real = 1) -> RankOneCovariance

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> RankOneCovariance()
RankOneCovariance
  shift ┴ Int64: 1
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`ShortTermLossControlPortfolio`](@ref)
  - [`Variance`](@ref)
  - [`QuadRiskExpr`](@ref)

# References

  - $(ref_dict[:lai2020spolc])
"""
struct RankOneCovariance{T1 <: Real} <: AbstractCovarianceEstimator
    """
    The number added to every entry of the rows before the decomposition: `1` reads returns as the price relatives the paper decomposes, `0` decomposes the rows as given.
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

The rank-one covariance of a window: ``\\zeta_1^\\star \\boldsymbol{u}_1 \\boldsymbol{u}_1^\\intercal`` over the shifted rows, `assets × assets`.

# Validation

  - `size(X, 1) >= 2` after orientation. An `ArgumentError` is thrown otherwise.

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
    zeta = if iszero(trD)
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

The correlation of the rank-one covariance: ``\\pm 1`` between two assets the principal vector loads, the sign of the product of their loadings, one on the diagonal, and zero at an asset the vector does not load, whose variance under the estimate is zero.

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
