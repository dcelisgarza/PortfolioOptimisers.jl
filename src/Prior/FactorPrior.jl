struct PartialFactorModel{T1 <: AbstractVector, T2 <: AbstractMatrix, T3 <: LoadingsMatrix}
    mu::T1
    sigma::T2
    loadings::T3
end
function PartialFactorModel(; mu::AbstractVector, sigma::AbstractMatrix,
                            loadings::LoadingsMatrix)
    @smart_assert(!isempty(mu) && !isempty(sigma))
    @smart_assert(length(mu) == size(sigma, 1) == size(sigma, 2) == size(loadings.M, 2))
    return PartialFactorModel{typeof(mu), typeof(sigma), typeof(loadings)}(mu, sigma,
                                                                           loadings)
end
struct FactorPriorModel{T1 <: EmpiricalPriorModel, T2 <: PartialFactorModel,
                        T3 <: AbstractMatrix} <: AbstractPriorModel_AFC
    pm::T1
    fm::T2
    chol::T3
end
function FactorPriorModel(; pm::EmpiricalPriorModel, fm::PartialFactorModel,
                          chol::AbstractMatrix)
    @smart_assert(!isempty(chol))
    @smart_assert(size(pm.X, 2) == size(chol, 2) == size(fm.loadings.M, 1))
    return FactorPriorModel{typeof(pm), typeof(fm), typeof(chol)}(pm, fm, chol)
end
function Base.getproperty(obj::FactorPriorModel, sym::Symbol)
    return if sym == :X
        obj.pm.X
    elseif sym == :mu
        obj.pm.mu
    elseif sym == :sigma
        obj.pm.sigma
    elseif sym == :f_mu
        obj.fm.mu
    elseif sym == :f_sigma
        obj.fm.sigma
    elseif sym == :loadings
        obj.fm.loadings
    else
        getfield(obj, sym)
    end
end
struct FactorPriorEstimator{T1 <: AbstractPriorEstimatorMap_2_1,
                            T2 <: AbstractMatrixProcessingEstimator, T3 <: RegressionMethod,
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
                              mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing(),
                              re::RegressionMethod = ForwardRegression(),
                              ve::PortfolioOptimisersVarianceEstimator = SimpleVariance(),
                              residuals::Bool = true)
    return FactorPriorEstimator{typeof(pe), typeof(mp), typeof(re), typeof(ve),
                                typeof(residuals)}(pe, mp, re, ve, residuals)
end
function w_moment_factory(pe::FactorPriorEstimator,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return FactorPriorEstimator(; pe = w_moment_factory(pe.pe, w), mp = pe.mp, re = pe.re,
                                ve = w_moment_factory(pe.ve, w), residuals = pe.residuals)
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
    f_prior = prior(pe.pe, F; kwargs...)
    f_mu, f_sigma = f_prior.mu, f_prior.sigma
    loadings = regression(pe.re, X, F)
    (; b, M) = loadings
    posterior_X = F * transpose(M) .+ transpose(b)
    posterior_mu = M * f_mu .+ b
    posterior_sigma = M * f_sigma * transpose(M)
    fit_estimator!(pe.mp, posterior_sigma, posterior_X)
    posterior_csigma = M * cholesky(f_sigma).L
    if pe.residuals
        err = X - posterior_X
        err_sigma = diagm(vec(var(pe.ve, err; dims = 1)))
        posterior_sigma .+= err_sigma
        posterior_csigma = hcat(posterior_csigma, sqrt.(err_sigma))
    end
    return FactorPriorModel(;
                            pm = EmpiricalPriorModel(; X = posterior_X, mu = posterior_mu,
                                                     sigma = posterior_sigma),
                            fm = PartialFactorModel(; mu = f_mu, sigma = f_sigma,
                                                    loadings = loadings),
                            chol = transpose(reshape(posterior_csigma, length(posterior_mu),
                                                     :)))
end

export FactorPriorModel, FactorPriorEstimator
