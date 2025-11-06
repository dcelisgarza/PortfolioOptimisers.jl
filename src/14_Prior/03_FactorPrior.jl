"""
    struct FactorPrior{T1, T2, T3, T4, T5} <: AbstractLowOrderPriorEstimator_F
        pe::T1
        mp::T2
        re::T3
        ve::T4
        rsd::T5
    end

Factor-based prior estimator for asset returns.

`FactorPrior` is a low order prior estimator that computes the mean and covariance of asset returns using a factor model. It combines a factor prior estimator, matrix post-processing, regression, and variance estimation to produce posterior moments. Optionally, it can add residual variance to the posterior covariance for robust estimation.

# Fields

  - `pe`: Factor prior estimator.
  - `mp`: Matrix post-processing estimator.
  - `re`: Regression estimator.
  - `ve`: Variance estimator for residuals.
  - `rsd`: Boolean flag to add residual variance to posterior covariance.

# Constructor

    FactorPrior(; pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
                mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing(),
                re::AbstractRegressionEstimator = StepwiseRegression(),
                ve::AbstractVarianceEstimator = SimpleVariance(), rsd::Bool = true)

Keyword arguments correspond to the fields above.

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
      │           │   mp ┼ DefaultMatrixProcessing
      │           │      │       pdm ┼ Posdef
      │           │      │           │   alg ┴ UnionAll: NearestCorrelationMatrix.Newton
      │           │      │   denoise ┼ nothing
      │           │      │    detone ┼ nothing
      │           │      │       alg ┴ nothing
      │        me ┼ SimpleExpectedReturns
      │           │   w ┴ nothing
      │   horizon ┴ nothing
   mp ┼ DefaultMatrixProcessing
      │       pdm ┼ Posdef
      │           │   alg ┴ UnionAll: NearestCorrelationMatrix.Newton
      │   denoise ┼ nothing
      │    detone ┼ nothing
      │       alg ┴ nothing
   re ┼ StepwiseRegression
      │     crit ┼ PValue
      │          │   threshold ┴ Float64: 0.05
      │      alg ┼ Forward()
      │   target ┼ LinearModel
      │          │   kwargs ┴ @NamedTuple{}: NamedTuple()
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
struct FactorPrior{T1, T2, T3, T4, T5} <: AbstractLowOrderPriorEstimator_F
    pe::T1
    mp::T2
    re::T3
    ve::T4
    rsd::T5
    function FactorPrior(pe::AbstractLowOrderPriorEstimator_A_AF,
                         mp::AbstractMatrixProcessingEstimator,
                         re::AbstractRegressionEstimator, ve::AbstractVarianceEstimator,
                         rsd::Bool)
        return new{typeof(pe), typeof(mp), typeof(re), typeof(ve), typeof(rsd)}(pe, mp, re,
                                                                                ve, rsd)
    end
end
function FactorPrior(; pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
                     mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing(),
                     re::AbstractRegressionEstimator = StepwiseRegression(),
                     ve::AbstractVarianceEstimator = SimpleVariance(), rsd::Bool = true)
    return FactorPrior(pe, mp, re, ve, rsd)
end
function factory(pe::FactorPrior, w::WeightsType = nothing)
    return FactorPrior(; pe = factory(pe.pe, w), mp = pe.mp, re = factory(pe.re, w),
                       ve = factory(pe.ve, w), rsd = pe.rsd)
end
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
    prior(pe::FactorPrior, X::AbstractMatrix, F::AbstractMatrix; dims::Int = 1, kwargs...)

Compute factor-based prior moments for asset returns using a factor model.

`prior` estimates the mean and covariance of asset returns using the specified factor prior estimator, regression, and matrix post-processing. The factor returns matrix `F` is used to compute factor moments, which are then mapped to asset space via regression. Optionally, residual variance is added to the posterior covariance for robust estimation. The result is returned as a [`LowOrderPrior`](@ref) object.

# Arguments

  - `pe`: Factor prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor returns matrix (observations × factors).
  - `dims`: Dimension along which to compute moments.
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
function prior(pe::FactorPrior, X::AbstractMatrix, F::AbstractMatrix; dims::Int = 1,
               kwargs...)
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
    posterior_csigma = M * cholesky(f_sigma).L
    if pe.rsd
        err = X - posterior_X
        err_sigma = diagm(vec(var(pe.ve, err; dims = 1)))
        posterior_sigma .+= err_sigma
        posterior_csigma = hcat(posterior_csigma, sqrt.(err_sigma))
    end
    return LowOrderPrior(; X = posterior_X, mu = posterior_mu, sigma = posterior_sigma,
                         chol = transpose(reshape(posterior_csigma, length(posterior_mu),
                                                  :)), w = f_prior.w, rr = rr, f_mu = f_mu,
                         f_sigma = f_sigma, f_w = f_prior.w)
end

export FactorPrior
