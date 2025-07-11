struct FactorPriorEstimator{T1 <: AbstractLowOrderPriorEstimatorMap_2_1,
                            T2 <: AbstractMatrixProcessingEstimator,
                            T3 <: AbstractRegressionEstimator,
                            T4 <: AbstractVarianceEstimator, T5 <: Bool} <:
       AbstractLowOrderPriorEstimator_2_1
    pe::T1
    mp::T2
    re::T3
    ve::T4
    rsd::T5
end
function FactorPriorEstimator(;
                              pe::AbstractLowOrderPriorEstimatorMap_2_1 = EmpiricalPriorEstimator(),
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
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
        F = transpose(F)
    end
    f_prior = prior(pe.pe, F)
    f_mu, f_sigma = f_prior.mu, f_prior.sigma
    loadings = regression(pe.re, X, F)
    (; b, M) = loadings
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
    return LowOrderPriorResult(; X = posterior_X, mu = posterior_mu,
                               sigma = posterior_sigma,
                               chol = transpose(reshape(posterior_csigma,
                                                        length(posterior_mu), :)),
                               w = f_prior.w, loadings = loadings, f_mu = f_mu,
                               f_sigma = f_sigma, f_w = f_prior.w)
end

export FactorPriorEstimator
