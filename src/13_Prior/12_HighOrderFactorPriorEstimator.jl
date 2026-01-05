function coskewness_residuals(X::MatNum, me::AbstractExpectedReturnsEstimator)
    N = size(X, 2)
    N2 = N^2
    sk_err = SparseArrays.spzeros(N, N2)
    idx = 1:(N2 + N + 1):(N2 * N)
    sk_err[idx] .= vec(Statistics.mean(me, X .^ 3; dims = 1))
    return sk_err
end
function cokurtosis_residuals end
struct HighOrderFactorPriorEstimator{T1, T2, T3, T4, T5, T6, T7} <:
       AbstractLowOrderPriorEstimator_F
    pe::T1
    mp::T2
    kte::T3
    ske::T4
    re::T5
    ve::T6
    rsd::T7
    function HighOrderFactorPriorEstimator(pe::AbstractLowOrderPriorEstimator_A_AF,
                                           mp::AbstractMatrixProcessingEstimator,
                                           kte::Option{<:CokurtosisEstimator},
                                           ske::Option{<:CoskewnessEstimator},
                                           re::AbstractRegressionEstimator,
                                           ve::AbstractVarianceEstimator, rsd::Bool)
        return new{typeof(pe), typeof(mp), typeof(kte), typeof(ske), typeof(re), typeof(ve),
                   typeof(rsd)}(pe, mp, kte, ske, re, ve, rsd)
    end
end
function HighOrderFactorPriorEstimator(;
                                       pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
                                       mp::AbstractMatrixProcessingEstimator = DenoiseDetoneAlgMatrixProcessing(),
                                       kte::Option{<:CokurtosisEstimator} = Cokurtosis(;
                                                                                       alg = Full()),
                                       ske::Option{<:CoskewnessEstimator} = Coskewness(;
                                                                                       alg = Full()),
                                       re::AbstractRegressionEstimator = StepwiseRegression(),
                                       ve::AbstractVarianceEstimator = SimpleVariance(),
                                       rsd::Bool = true)
    return HighOrderFactorPriorEstimator(pe, mp, kte, ske, re, ve, rsd)
end
function factory(pe::HighOrderFactorPriorEstimator,
                 w::Option{<:StatsBase.AbstractWeights} = nothing)
    return HighOrderFactorPriorEstimator(; pe = factory(pe.pe, w), mp = pe.mp,
                                         kte = factory(pe.kte, w), ske = factory(pe.ske, w),
                                         re = factory(pe.re, w), ve = factory(pe.ve, w),
                                         rsd = pe.rsd)
end
function Base.getproperty(obj::HighOrderFactorPriorEstimator, sym::Symbol)
    return if sym == :me
        obj.pe.me
    elseif sym == :ce
        obj.pe.ce
    else
        getfield(obj, sym)
    end
end
function prior(pe::HighOrderFactorPriorEstimator, X::MatNum, F::MatNum; dims::Int = 1,
               kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
        F = transpose(F)
    end
    N = size(X, 2)
    f_prior = prior(pe.pe, F)
    f_mu, f_sigma = f_prior.mu, f_prior.sigma
    rr = regression(pe.re, X, F)
    (; b, M) = rr
    posterior_X = F * transpose(M) .+ transpose(b)
    posterior_mu = M * f_mu + b
    posterior_sigma = M * f_sigma * transpose(M)
    matrix_processing!(pe.mp, posterior_sigma, posterior_X; kwargs...)
    posterior_csigma = M * LinearAlgebra.cholesky(f_sigma).L
    f_kt = cokurtosis(pe.kte, F; kwargs...)
    kM = nothing
    if !isnothing(f_kt)
        kM = kron(M, M)
        L2, S2 = dup_elim_sum_matrices(size(posterior_X, 2))[2:3]
        posterior_kt = kM * f_kt * transpose(kM)
        matrix_processing!(pe.ke.mp, posterior_kt, posterior_X; kwargs...)
        posterior_ckt = kM * LinearAlgebra.cholesky(f_kt).L
    else
        L2, S2, posterior_kt, posterior_ckt = nothing, nothing, nothing, nothing
    end
    f_sk, f_V = coskewness(pe.ske, F; kwargs...)
    if !isnothing(f_sk)
        if isnothing(kM)
            kM = kron(M, M)
        end
        posterior_sk = M * f_sk * transpose(kM)
    else
        posterior_sk, posterior_V = nothing, nothing
    end
    if pe.rsd
        err = X - posterior_X
        err_sigma = LinearAlgebra.diagm(vec(Statistics.var(pe.ve, err; dims = 1)))
        posterior_sigma .+= err_sigma
        posterior_csigma = hcat(posterior_csigma, sqrt.(err_sigma))
        if !isnothing(f_sk)
            posterior_sk .+= coskewness_residuals(err, pe.ske.me)
        end
        if !isnothing(f_kt)
            err_kt = cokurtosis_residuals()
            posterior_kt .+= err_kt
            posterior_ckt = hcat(posterior_ckt, err_kt .^ 0.25)
        end
    end
    if !isnothing(f_sk)
        posterior_V = __coskewness(posterior_sk, posterior_X, pe.ske.mp)
    end
    pr = LowOrderPrior(; X = posterior_X, mu = posterior_mu, sigma = posterior_sigma,
                       chol = transpose(reshape(posterior_csigma, length(posterior_mu), :)),
                       w = f_prior.w, rr = rr, f_mu = f_mu, f_sigma = f_sigma,
                       f_w = f_prior.w)
    return HighOrderPrior(; pr = pr, kt = posterior_kt, L2 = L2, S2 = S2, sk = posterior_sk,
                          V = posterior_V,
                          skmp = isnothing(posterior_sk) ? nothing : pe.ske.mp,
                          #=
                          f_kt = f_kt,
                          kt_chol = if isnothing(posterior_ckt)
                              nothing
                          else
                              transpose(reshape(posterior_ckt, size(posterior_kt, 1), :))
                          end,
                          f_sk = f_sk,
                          f_V = f_V
                          =#
                          )
end

export HighOrderFactorPriorEstimator
