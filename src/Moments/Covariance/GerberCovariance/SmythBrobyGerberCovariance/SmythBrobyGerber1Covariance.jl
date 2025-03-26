struct SmythBrobyGerber1Covariance{T1 <: ExpectedReturnsEstimator,
                                   T2 <: StatsBase.CovarianceEstimator,
                                   T3 <: FixNonPositiveDefiniteMatrix, T4 <: Real,
                                   T5 <: Real, T6 <: Real, T7 <: Real, T8 <: Real} <:
       SmythBrobyGerberCovariance
    me::T1
    ve::T2
    fnpdm::T3
    threshold::T4
    c1::T5
    c2::T6
    c3::T7
    n::T8
end
function SmythBrobyGerber1Covariance(;
                                     me::ExpectedReturnsEstimator = SimpleExpectedReturns(),
                                     ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                                     fnpdm::Union{Nothing, <:FixNonPositiveDefiniteMatrix} = FNPDM_NearestCorrelationMatrix(),
                                     threshold::Real = 0.5, c1::Real = 0.5, c2::Real = 0.5,
                                     c3::Real = 4.0, n::Real = 2.0)
    @smart_assert(zero(threshold) < threshold < one(threshold))
    @smart_assert(zero(c1) < c1 <= one(c1))
    @smart_assert(zero(c2) < c2 <= one(c2) && c3 > c2)
    return SmythBrobyGerber1Covariance{typeof(me), typeof(ve), typeof(fnpdm),
                                       typeof(threshold), typeof(c1), typeof(c2),
                                       typeof(c3), typeof(n)}(me, ve, fnpdm, threshold, c1,
                                                              c2, c3, n)
end
function moment_factory_w(ce::SmythBrobyGerber1Covariance,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return SmythBrobyGerber1Covariance(; me = moment_factory_w(ce.me, w),
                                       ve = moment_factory_w(ce.ve, w), fnpdm = ce.fnpdm,
                                       threshold = ce.threshold, c1 = ce.c1, c2 = ce.c2,
                                       c3 = ce.c3, n = ce.n)
end
function _smythbrobygerber1(ce::SmythBrobyGerber1Covariance, X::AbstractMatrix, mean_vec,
                            std_vec)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    for j ∈ axes(X, 2)
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i ∈ 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            nn = zero(eltype(X))
            cneg = 0
            cpos = 0
            cnn = 0
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k ∈ 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold * sigmai
                tj = threshold * sigmaj
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                    cpos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                    cneg += 1
                elseif abs(xi) < ti && abs(xj) < tj
                    nn += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                    cnn += 1
                end
            end
            tpos = pos * cpos
            tneg = neg * cneg
            tnn = nn * cnn
            den = (tpos + tneg + tnn)
            rho[j, i] = rho[i, j] = if !iszero(den)
                (tpos - tneg) / den
            else
                zero(eltype(X))
            end
        end
    end
    fix_non_positive_definite_matrix!(ce.fnpdm, rho)
    return rho
end
function StatsBase.cor(ce::SmythBrobyGerber1Covariance, X::AbstractMatrix; dims::Int = 1,
                       kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mean_vec = mean(ce.me, X; dims = 1)
    std_vec = std(ce.ve, X; dims = 1, mean = mean_vec)
    idx = iszero.(std_vec)
    std_vec[idx] .= eps(eltype(X))
    return _smythbrobygerber1(ce, X, mean_vec, std_vec)
end
function StatsBase.cov(ce::SmythBrobyGerber1Covariance, X::AbstractMatrix; dims::Int = 1,
                       kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mean_vec = mean(ce.me, X; dims = 1)
    std_vec = std(ce.ve, X; dims = 1, mean = mean_vec)
    idx = iszero.(std_vec)
    std_vec[idx] .= eps(eltype(X))
    return _smythbrobygerber1(ce, X, mean_vec, std_vec) .* (std_vec ⊗ std_vec)
end

export SmythBrobyGerber1Covariance
