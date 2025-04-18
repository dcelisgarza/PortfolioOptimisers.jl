struct PartialFactorPriorResult{T1 <: AbstractVector, T2 <: AbstractMatrix,
                                T3 <: RegressionResult}
    mu::T1
    sigma::T2
    loadings::T3
end
function PartialFactorPriorResult(; mu::AbstractVector, sigma::AbstractMatrix,
                                  loadings::RegressionResult)
    @smart_assert(!isempty(mu) && !isempty(sigma))
    @smart_assert(length(mu) == size(sigma, 1) == size(sigma, 2) == size(loadings.M, 2))
    return PartialFactorPriorResult{typeof(mu), typeof(sigma), typeof(loadings)}(mu, sigma,
                                                                                 loadings)
end
function prior_view(pr::PartialFactorPriorResult, i::AbstractVector)
    return PartialFactorPriorResult(; mu = view(pr.mu, i), sigma = view(pr.sigma, i, i),
                                    loadings = regression_result_view(pr.loadings, i))
end
struct FactorPriorResult{T1 <: EmpiricalPriorResult, T2 <: PartialFactorPriorResult,
                         T3 <: AbstractMatrix} <: AbstractPriorResult_AFC
    pm::T1
    fm::T2
    chol::T3
end
function FactorPriorResult(; pm::EmpiricalPriorResult, fm::PartialFactorPriorResult,
                           chol::AbstractMatrix)
    @smart_assert(!isempty(chol))
    @smart_assert(size(pm.X, 2) == size(chol, 2) == size(fm.loadings.M, 1))
    return FactorPriorResult{typeof(pm), typeof(fm), typeof(chol)}(pm, fm, chol)
end
function prior_view(pm::FactorPriorResult, i::AbstractVector)
    return FactorPriorResult(; pm = prior_view(pm.pm, i), fm = prior_view(pm.fm, i),
                             chol = view(pm.chol, :, i))
end
function Base.getproperty(obj::FactorPriorResult, sym::Symbol)
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
                            T2 <: AbstractMatrixProcessingEstimator,
                            T3 <: AbstractRegressionEstimator,
                            T4 <: AbstractVarianceEstimator, T5 <: Bool} <:
       AbstractPriorEstimator_2_1
    pe::T1
    mp::T2
    re::T3
    ve::T4
    rsd::T5
end
function FactorPriorEstimator(;
                              pe::AbstractPriorEstimatorMap_2_1 = EmpiricalPriorEstimator(),
                              mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing(),
                              re::AbstractRegressionEstimator = StepwiseRegression(),
                              ve::AbstractVarianceEstimator = SimpleVariance(),
                              rsd::Bool = true)
    return FactorPriorEstimator{typeof(pe), typeof(mp), typeof(re), typeof(ve),
                                typeof(rsd)}(pe, mp, re, ve, rsd)
end
function factory(pe::FactorPriorEstimator, w::Union{Nothing, <:AbstractWeights} = nothing)
    return FactorPriorEstimator(; pe = factory(pe.pe, w), mp = pe.mp, re = pe.re,
                                ve = factory(pe.ve, w), rsd = pe.rsd)
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
    matrix_processing!(pe.mp, posterior_sigma, posterior_X)
    posterior_csigma = M * cholesky(f_sigma).L
    if pe.rsd
        err = X - posterior_X
        err_sigma = diagm(vec(var(pe.ve, err; dims = 1)))
        posterior_sigma .+= err_sigma
        posterior_csigma = hcat(posterior_csigma, sqrt.(err_sigma))
    end
    return FactorPriorResult(;
                             pm = EmpiricalPriorResult(; X = posterior_X, mu = posterior_mu,
                                                       sigma = posterior_sigma),
                             fm = PartialFactorPriorResult(; mu = f_mu, sigma = f_sigma,
                                                           loadings = loadings),
                             chol = transpose(reshape(posterior_csigma,
                                                      length(posterior_mu), :)))
end

export FactorPriorResult, FactorPriorEstimator
