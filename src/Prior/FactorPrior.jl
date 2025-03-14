struct FactorPriorModel{T1 <: AbstractMatrix, T2 <: AbstractVector, T3 <: AbstractMatrix,
                        T4 <: AbstractMatrix, T5 <: AbstractVector, T6 <: AbstractMatrix,
                        T7 <: LoadingsMatrix} <: AbstractPriorModel_AFC
    X::T1
    mu::T2
    sigma::T3
    chol::T4
    f_mu::T5
    f_sigma::T6
    loadings::T7
end
function FactorPriorModel(; X::AbstractMatrix, mu::AbstractVector, sigma::AbstractMatrix,
                          chol::AbstractMatrix, f_mu::AbstractVector,
                          f_sigma::AbstractMatrix, loadings::LoadingsMatrix)
    @smart_assert(!isempty(X) &&
                  !isempty(mu) &&
                  !isempty(sigma) &&
                  !isempty(chol) &&
                  !isempty(f_mu) &&
                  !isempty(f_sigma))
    @smart_assert(size(X, 2) ==
                  length(mu) ==
                  size(sigma, 1) ==
                  size(sigma, 2) ==
                  size(loadings.M, 1) ==
                  length(loadings.b))
    @smart_assert(length(f_mu) ==
                  size(f_sigma, 1) ==
                  size(f_sigma, 2) ==
                  size(loadings.M, 2))
    @smart_assert(size(chol, 2) == length(mu))
    return FactorPriorModel{typeof(X), typeof(mu), typeof(sigma), typeof(chol),
                            typeof(f_mu), typeof(f_sigma), typeof(loadings)}(X, mu, sigma,
                                                                             chol, f_mu,
                                                                             f_sigma,
                                                                             loadings)
end
struct FactorPriorEstimator{T1 <: AbstractPriorEstimatorMap_2_1, T2 <: MatrixProcessing,
                            T3 <: RegressionMethod,
                            T4 <: PortfolioOptimisersVarianceEstimator, T5 <: Bool} <:
       AbstractPriorEstimator_2_1
    pe::T1
    mp::T2
    re::T3
    ve::T4
    residuals::T5
end
function FactorPriorEstimator(;
                              pe::AbstractPriorEstimatorMap_2_1 = EmpiricalPriorEstimator(),
                              mp::MatrixProcessing = DefaultMatrixProcessing(),
                              re::RegressionMethod = ForwardRegression(),
                              ve::PortfolioOptimisersVarianceEstimator = SimpleVariance(),
                              residuals::Bool = true)
    return FactorPriorEstimator{typeof(pe), typeof(mp), typeof(re), typeof(ve),
                                typeof(residuals)}(pe, mp, re, ve, residuals)
end
function Base.getproperty(obj::FactorPriorEstimator, sym::Symbol)
    return if sym == :me
        obj.pe.me
    elseif sym == :ce
        obj.pe.ce
    else
        getfield(obj, sym)
    end
end
function prior(pe::FactorPriorEstimator, X::AbstractMatrix, F::AbstractMatrix;
               dims::Int = 1, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
        F = transpose(F)
    end
    f_prior = prior(pe.pe, F)
    f_mu, f_sigma = f_prior.mu, f_prior.sigma
    loadings = regression(pe.re, X, F)
    (; b, M) = loadings
    posterior_X = F * transpose(M) .+ transpose(b)
    posterior_mu = M * f_mu .+ b
    posterior_sigma = M * f_sigma * transpose(M)
    mtx_process!(pe.mp, posterior_sigma, posterior_X)
    posterior_csigma = M * cholesky(f_sigma).L
    if pe.residuals
        err = X - posterior_X
        err_sigma = diagm(vec(var(pe.ve, err; dims = 1)))
        posterior_sigma .+= err_sigma
        posterior_csigma = hcat(posterior_csigma, sqrt.(err_sigma))
    end
    return FactorPriorModel(; X = posterior_X, mu = posterior_mu, sigma = posterior_sigma,
                            chol = transpose(reshape(posterior_csigma, length(posterior_mu),
                                                     :)), f_mu = f_mu, f_sigma = f_sigma,
                            loadings = loadings)
end

export FactorPriorModel, FactorPriorEstimator
