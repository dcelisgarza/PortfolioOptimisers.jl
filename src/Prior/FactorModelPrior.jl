struct FactorPriorModel{T1 <: AbstractMatrix, T2 <: AbstractVector, T3 <: AbstractMatrix,
                        T4 <: AbstractMatrix} <: AbstractPriorModel
    X::T1
    mu::T2
    sigma::T3
    csigma::T4
end
function FactorPriorModel(; X::AbstractMatrix, mu::AbstractVector, sigma::AbstractMatrix,
                          csigma::AbstractMatrix)
    @smart_assert(size(X, 2) ==
                  length(mu) ==
                  size(sigma, 1) ==
                  size(sigma, 2) ==
                  size(csigma, 2))
    return FactorPriorModel{typeof(X), typeof(mu), typeof(sigma), typeof(csigma)}(X, mu,
                                                                                  sigma,
                                                                                  csigma)
end
struct FactorModelPriorEstimator{T1 <: PriorEstimator, T2 <: MatrixProcessing,
                                 T3 <: RegressionMethod,
                                 T4 <: PortfolioOptimisersVarianceEstimator, T5 <: Bool}
    pe::T1
    mp::T2
    re::T3
    ve::T4
    residuals::T5
end
function FactorModelPriorEstimator(; pe::PriorEstimator = EmpiricalPriorEstimator(),
                                   mp::MatrixProcessing = DefaultMatrixProcessing(),
                                   re::RegressionMethod = ForwardRegression(),
                                   ve::PortfolioOptimisersVarianceEstimator = SimpleVariance(),
                                   residuals::Bool = true)
    return FactorModelPriorEstimator{typeof(pe), typeof(mp), typeof(re), typeof(ve),
                                     typeof(residuals)}(pe, mp, re, ve, residuals)
end
function prior(pe::FactorModelPriorEstimator, X::AbstractMatrix, F::AbstractMatrix;
               dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
        F = transpose(F)
    end
    factor_prior = prior(pe.pe, F)
    f_mu = factor_prior.mu
    f_sigma = factor_prior.sigma
    loadings = regression(pe.re, X, F)
    (; c, M) = loadings
    posterior_X = F * transpose(M) .+ transpose(c)
    posterior_mu = M * f_mu .+ c
    posterior_sigma = M * f_sigma * transpose(M)
    posterior_csigma = M * cholesky(f_sigma).L
    if pe.residuals
        err = X - posterior_X
        err_sigma = diagm(vec(var(pe.ve, err; dims = 1)))
        posterior_sigma .+= err_sigma
        posterior_csigma = hcat(posterior_csigma, sqrt.(err_sigma))
    end
    mtx_process!(pe.mp, posterior_sigma, posterior_X)
    return FactorPriorModel(; X = posterior_X, mu = posterior_mu, sigma = posterior_sigma,
                            csigma = transpose(reshape(posterior_csigma,
                                                       length(posterior_mu), :)))
end

export FactorPriorModel, FactorModelPriorEstimator
