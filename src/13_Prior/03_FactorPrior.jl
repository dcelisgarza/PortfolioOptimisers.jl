"""
$(DocStringExtensions.TYPEDEF)

Factor-based prior estimator for asset returns.

`FactorPrior` is a low order prior estimator that computes the mean and covariance of asset returns using a factor model. It combines a factor prior estimator, matrix post-processing, regression, and variance estimation to produce posterior moments. Optionally, it can add residual variance to the posterior covariance for robust estimation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FactorPrior(;
        pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
        mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
        re::AbstractRegressionEstimator = StepwiseRegression(),
        ve::AbstractVarianceEstimator = SimpleVariance(),
        rsd::Bool = true
    ) -> FactorPrior

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> FactorPrior()
FactorPrior
   pe ┼ EmpiricalPrior
      │        ce ┼ PortfolioOptimisersCovariance
      │           │   ce ┼ Covariance
      │           │      │    me ┼ SimpleExpectedReturns
      │           │      │       │   w ┴ nothing
      │           │      │    ce ┼ GeneralCovariance
      │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │           │      │       │    w ┴ nothing
      │           │      │   alg ┴ Full()
      │           │   mp ┼ MatrixProcessing
      │           │      │     pdm ┼ Posdef
      │           │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │           │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │           │      │      dn ┼ nothing
      │           │      │      dt ┼ nothing
      │           │      │     alg ┼ nothing
      │           │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
      │        me ┼ SimpleExpectedReturns
      │           │   w ┴ nothing
      │   horizon ┴ nothing
   mp ┼ MatrixProcessing
      │     pdm ┼ Posdef
      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │      dn ┼ nothing
      │      dt ┼ nothing
      │     alg ┼ nothing
      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
   re ┼ StepwiseRegression
      │   crit ┼ PValue
      │        │   t ┴ Float64: 0.05
      │    alg ┼ Forward()
      │    tgt ┼ LinearModel
      │        │   kwargs ┴ @NamedTuple{}: NamedTuple()
   ve ┼ SimpleVariance
      │          me ┼ SimpleExpectedReturns
      │             │   w ┴ nothing
      │           w ┼ nothing
      │   corrected ┴ Bool: true
  rsd ┴ Bool: true
```

# Related

  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`AbstractLowOrderPriorEstimator_A_AF`](@ref)
  - [`AbstractMatrixProcessingEstimator`](@ref)
  - [`AbstractRegressionEstimator`](@ref)
  - [`AbstractVarianceEstimator`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`StepwiseRegression`](@ref)
  - [`SimpleVariance`](@ref)
  - [`prior`](@ref)
"""
@concrete struct FactorPrior <: AbstractLowOrderPriorEstimator_F
    """
    $(field_dict[:pe])
    """
    pe
    """
    $(field_dict[:mp])
    """
    mp
    """
    $(field_dict[:re])
    """
    re
    """
    $(field_dict[:ve])
    """
    ve
    """
    $(field_dict[:rsd])
    """
    rsd
    function FactorPrior(pe::AbstractLowOrderPriorEstimator_A_AF,
                         mp::AbstractMatrixProcessingEstimator,
                         re::AbstractRegressionEstimator, ve::AbstractVarianceEstimator,
                         rsd::Bool)
        return new{typeof(pe), typeof(mp), typeof(re), typeof(ve), typeof(rsd)}(pe, mp, re,
                                                                                ve, rsd)
    end
end
function FactorPrior(; pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
                     mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                     re::AbstractRegressionEstimator = StepwiseRegression(),
                     ve::AbstractVarianceEstimator = SimpleVariance(),
                     rsd::Bool = true)::FactorPrior
    return FactorPrior(pe, mp, re, ve, rsd)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a new [`FactorPrior`](@ref) estimator with observation weights `w` applied to the underlying prior, regression, and variance estimators.

# Related

  - [`FactorPrior`](@ref)
  - [`factory`](@ref)
"""
function factory(pe::FactorPrior, w::ObsWeights)::FactorPrior
    return FactorPrior(; pe = factory(pe.pe, w), mp = pe.mp, re = factory(pe.re, w),
                       ve = factory(pe.ve, w), rsd = pe.rsd)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a new [`FactorPrior`](@ref) estimator restricted to the assets at index `i`.

# Related

  - [`FactorPrior`](@ref)
"""
function prior_view(pe::FactorPrior, i)::FactorPrior
    return FactorPrior(; pe = pe.pe, mp = pe.mp, re = regression_view(pe.re, i),
                       ve = moment_view(pe.ve, i), rsd = pe.rsd)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Access properties of [`FactorPrior`](@ref). Exposes `:me` and `:ce` from the embedded asset prior estimator `obj.pe` for transparent access.
"""
function Base.getproperty(obj::FactorPrior, sym::Symbol)
    return if sym == :me
        obj.pe.me
    elseif sym == :ce
        obj.pe.ce
    else
        getfield(obj, sym)
    end
end
"""
    prior(pe::FactorPrior, X::MatNum, F::MatNum; dims::Int = 1, kwargs...)

Compute factor-based prior moments for asset returns using a factor model.

`prior` estimates the mean and covariance of asset returns using the specified factor prior estimator, regression, and matrix post-processing. The factor returns matrix `F` is used to compute factor moments, which are then mapped to asset space via regression. Optionally, residual variance is added to the posterior covariance for robust estimation. The result is returned as a [`LowOrderPrior`](@ref) object.

# Mathematical definition

The factor model maps factor moments to asset space via the loadings matrix ``\\mathbf{B}`` (with intercepts ``\\boldsymbol{\\alpha}``):

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}} &= \\mathbf{B} \\hat{\\boldsymbol{f}} + \\boldsymbol{\\alpha}\\,.
\\end{align}
```

```math
\\begin{align}
\\hat{\\mathbf{\\Sigma}} &= \\mathbf{B} \\mathbf{\\Sigma}_f \\mathbf{B}^\\intercal + \\mathbf{\\Sigma}_\\varepsilon\\,.
\\end{align}
```

Where:

  - ``\\mathbf{B}``: ``N \\times K`` factor loadings matrix.
  - ``\\hat{\\boldsymbol{f}}``: ``K \\times 1`` vector of factor expected returns.
  - ``\\boldsymbol{\\alpha}``: ``N \\times 1`` vector of regression intercepts.
  - ``\\mathbf{\\Sigma}_f``: ``K \\times K`` factor covariance matrix.
  - ``\\mathbf{\\Sigma}_\\varepsilon``: ``N \\times N`` diagonal matrix of residual variances (when `rsd = true`).

# Arguments

  - `pe`: Factor prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor returns matrix (observations × factors).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to matrix processing and estimators.

# Returns

  - `pr::LowOrderPrior`: Result object containing posterior asset returns, mean vector, covariance matrix, Cholesky factor, regression result, and factor moments.

# Validation

  - `dims in (1, 2)`.

# Related

  - [`FactorPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`prior`](@ref)
"""
function prior(pe::FactorPrior, X::MatNum, F::MatNum; dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
        F = transpose(F)
    end
    f_prior = prior(pe.pe, F)
    f_mu, f_sigma = f_prior.mu, f_prior.sigma
    rr = regression(pe.re, X, F)
    (; b, M) = rr
    posterior_X = F * transpose(M) .+ transpose(b)
    posterior_mu = M * f_mu + b
    posterior_sigma = M * f_sigma * transpose(M)
    matrix_processing!(pe.mp, posterior_sigma, posterior_X; kwargs...)
    posterior_csigma = M * LinearAlgebra.cholesky(f_sigma).L
    if pe.rsd
        err = X - posterior_X
        err_sigma = LinearAlgebra.diagm(vec(Statistics.var(pe.ve, err; dims = 1)))
        posterior_sigma .+= err_sigma
        posdef!(pe.mp.pdm, posterior_sigma)
        posterior_csigma = hcat(posterior_csigma, sqrt.(err_sigma))
    end
    return LowOrderPrior(; X = posterior_X, mu = posterior_mu, sigma = posterior_sigma,
                         chol = transpose(reshape(posterior_csigma, length(posterior_mu),
                                                  :)), w = f_prior.w, rr = rr, f_mu = f_mu,
                         f_sigma = f_sigma, f_w = f_prior.w)
end

export FactorPrior
