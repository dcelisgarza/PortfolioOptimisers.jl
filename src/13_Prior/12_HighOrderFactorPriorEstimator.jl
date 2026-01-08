function coskewness_residuals(X::MatNum, me::AbstractExpectedReturnsEstimator)
    N = size(X, 2)
    N2 = N^2
    X3 = X .^ 3
    sk_err = SparseArrays.spzeros(eltype(X3), N, N2)
    idx = 1:(N2 + N + 1):(N2 * N)
    sk_err[idx] .= vec(Statistics.mean(me, X3; dims = 1))
    return sk_err
end
function cokurtosis_residuals(sigma::MatNum, X::MatNum,
                              me::AbstractExpectedReturnsEstimator,
                              ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
    N = size(X, 2)
    N2 = N^2
    X2 = X .^ 2
    X4 = X2 .^ 2
    e2 = vec(mean(me, X2; dims = 1))
    e4 = vec(mean(me, X4; dims = 1))
    kt_res = Matrix{promote_type(eltype(e4), eltype(sigma))}(undef, N2, N2)

    @inbounds FLoops.@floop ex for j in 1:N, l in 1:N
        col = (j - 1) * N + l
        for i in 1:N, k in 1:N
            row = (i - 1) * N + k
            if row > col
                continue
            end
            # Conditional logic optimized for most common cases first
            val = if i == j == k == l
                6 * e2[i] * sigma[i, i] + e4[i]
            elseif i == j == k
                3 * e2[i] * sigma[i, l]
            elseif i == j == l
                3 * e2[i] * sigma[i, k]
            elseif i == k == l
                3 * e2[i] * sigma[i, j]
            elseif j == k == l
                3 * e2[j] * sigma[j, i]
            elseif i == j && k == l
                e2[k] * sigma[i, i] + e2[i] * sigma[k, k] + e2[i] * e2[k]
            elseif i == k && j == l
                e2[j] * sigma[i, i] + e2[i] * sigma[j, j] + e2[i] * e2[j]
            elseif i == l && j == k
                e2[j] * sigma[i, i] + e2[i] * sigma[j, j] + e2[i] * e2[j]
            elseif i == j
                e2[i] * sigma[k, l]
            elseif i == k
                e2[i] * sigma[j, l]
            elseif i == l
                e2[i] * sigma[j, k]
            elseif j == k
                e2[j] * sigma[i, l]
            elseif j == l
                e2[j] * sigma[i, k]
            elseif k == l
                e2[k] * sigma[i, j]
            else
                zero(promote_type(eltype(e4), eltype(sigma)))
            end
            kt_res[row, col] = kt_res[col, row] = val
        end
    end
    return kt_res
end
"""
"""
struct HighOrderFactorPriorEstimator{T1, T2, T3, T4, T5} <:
       AbstractHighOrderPriorEstimator_F
    pe::T1
    kte::T2
    ske::T3
    ex::T4
    rsd::T5
    function HighOrderFactorPriorEstimator(pe::AbstractLowOrderPriorEstimator_F_AF,
                                           kte::Option{<:CokurtosisEstimator},
                                           ske::Option{<:CoskewnessEstimator},
                                           ex::FLoops.Transducers.Executor, rsd::Bool)
        return new{typeof(pe), typeof(kte), typeof(ske), typeof(ex), typeof(rsd)}(pe, kte,
                                                                                  ske, ex,
                                                                                  rsd)
    end
end
function HighOrderFactorPriorEstimator(;
                                       pe::AbstractLowOrderPriorEstimator_F_AF = FactorPrior(),
                                       kte::Option{<:CokurtosisEstimator} = Cokurtosis(;
                                                                                       alg = Full()),
                                       ske::Option{<:CoskewnessEstimator} = Coskewness(;
                                                                                       alg = Full()),
                                       ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                                       rsd::Bool = true)
    return HighOrderFactorPriorEstimator(pe, kte, ske, ex, rsd)
end
function factory(pe::HighOrderFactorPriorEstimator,
                 w::Option{<:StatsBase.AbstractWeights} = nothing)
    return HighOrderFactorPriorEstimator(; pe = factory(pe.pe, w), kte = factory(pe.kte, w),
                                         ske = factory(pe.ske, w), ex = pe.ex, rsd = pe.rsd)
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
    kM = nothing
    L2 = nothing
    S2 = nothing
    posterior_kt = nothing
    posterior_sk = nothing
    posterior_V = nothing
    pr = prior(pe.pe, X, F; dims = 1, kwargs...)
    posterior_X = pr.X
    M = pr.rr.M
    f_kt = cokurtosis(pe.kte, F; kwargs...)
    if !isnothing(f_kt)
        kM = kron(M, M)
        L2, S2 = dup_elim_sum_matrices(size(posterior_X, 2))[2:3]
        posterior_kt = kM * f_kt * transpose(kM)
        matrix_processing!(pe.kte.mp, posterior_kt, posterior_X; kwargs...)
    end
    f_sk, f_V = coskewness(pe.ske, F; kwargs...)
    if !isnothing(f_sk)
        if isnothing(kM)
            kM = kron(M, M)
        end
        posterior_sk = M * f_sk * transpose(kM)
    end
    if pe.rsd
        err = X - posterior_X
        if !isnothing(f_sk)
            posterior_sk .+= coskewness_residuals(err, pe.ske.me)
        end
        if !isnothing(f_kt)
            if isnothing(pr.chol)
                sigma = pr.sigma
            else
                err_sigma = vec(Statistics.var(pe.pe.ve, err; dims = 1))
                sigma = if any(map((x, y) -> x > y, err_sigma,
                                   LinearAlgebra.diag(pr.sigma)))
                    @warn("Some residual variances are larger than prior variances; using the prior variances to error correct the posterior kurtosis.")
                    pr.sigma
                else
                    pr.sigma - LinearAlgebra.diagm(err_sigma)
                end
                posdef!(pe.pe.mp.pdm, sigma)
            end
            err_kt = cokurtosis_residuals(sigma, err, pe.kte.me, pe.ex)
            posterior_kt .+= err_kt
            posdef!(pe.kte.mp.pdm, posterior_kt)
        end
    end
    if !isnothing(f_sk)
        posterior_V = negative_spectral_coskewness(posterior_sk, posterior_X, pe.ske.mp)
    end
    return HighOrderPrior(; pr = pr, kt = posterior_kt, L2 = L2, S2 = S2, sk = posterior_sk,
                          V = posterior_V, skmp = isnothing(f_sk) ? nothing : pe.ske.mp,
                          f_kt = f_kt, f_sk = f_sk, f_V = f_V)
end

export HighOrderFactorPriorEstimator
