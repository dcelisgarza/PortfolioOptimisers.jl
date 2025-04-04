struct SmythBrobyGerber0NormalisedCovariance{T1 <: ExpectedReturnsEstimator,
                                             T2 <: StatsBase.CovarianceEstimator,
                                             T3 <: FixNonPositiveDefiniteMatrix, T4 <: Real,
                                             T5 <: Real, T6 <: Real, T7 <: Real,
                                             T8 <: Real} <: SmythBrobyGerberCovariance
    me::T1
    ve::T2
    fnpdm::T3
    threshold::T4
    c1::T5
    c2::T6
    c3::T7
    n::T8
end
function SmythBrobyGerber0NormalisedCovariance(;
                                               me::ExpectedReturnsEstimator = SimpleExpectedReturns(),
                                               ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                                               fnpdm::Union{Nothing,
                                                            FixNonPositiveDefiniteMatrix} = FNPDM_NearestCorrelationMatrix(),
                                               threshold::Real = 0.5, c1::Real = 0.5,
                                               c2::Real = 0.5, c3::Real = 4.0,
                                               n::Real = 2.0)
    @smart_assert(zero(threshold) < threshold < one(threshold))
    @smart_assert(zero(c1) < c1 <= one(c1))
    @smart_assert(zero(c2) < c2 <= one(c2) && c3 > c2)
    return SmythBrobyGerber0NormalisedCovariance{typeof(me), typeof(ve), typeof(fnpdm),
                                                 typeof(threshold), typeof(c1), typeof(c2),
                                                 typeof(c3), typeof(n)}(me, ve, fnpdm,
                                                                        threshold, c1, c2,
                                                                        c3, n)
end
function moment_factory_w(ce::SmythBrobyGerber0NormalisedCovariance,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return SmythBrobyGerber0NormalisedCovariance(; me = moment_factory_w(ce.me, w),
                                                 ve = moment_factory_w(ce.ve, w),
                                                 fnpdm = ce.fnpdm, threshold = ce.threshold,
                                                 c1 = ce.c1, c2 = ce.c2, c3 = ce.c3,
                                                 n = ce.n)
end
function _smythbrobygerber0normalised(ce::SmythBrobyGerber0NormalisedCovariance,
                                      X::AbstractMatrix)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    for j ∈ axes(X, 2)
        for i ∈ 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            cneg = 0
            cpos = 0
            for k ∈ 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold
                tj = threshold
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                    one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                    cpos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                    one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                    cneg += 1
                end
            end
            tpos = pos * cpos
            tneg = neg * cneg
            den = (tpos + tneg)
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
function StatsBase.cor(ce::SmythBrobyGerber0NormalisedCovariance, X::AbstractMatrix;
                       dims::Int = 1, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mean_vec = mean(ce.me, X; dims = 1)
    std_vec = std(ce.ve, X; dims = 1, mean = mean_vec)
    idx = iszero.(std_vec)
    std_vec[idx] .= eps(eltype(X))
    X = (X .- mean_vec) ./ std_vec
    return _smythbrobygerber0normalised(ce, X)
end
function StatsBase.cov(ce::SmythBrobyGerber0NormalisedCovariance, X::AbstractMatrix;
                       dims::Int = 1, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mean_vec = mean(ce.me, X; dims = 1)
    std_vec = std(ce.ve, X; dims = 1, mean = mean_vec)
    idx = iszero.(std_vec)
    std_vec[idx] .= eps(eltype(X))
    X = (X .- mean_vec) ./ std_vec
    return _smythbrobygerber0normalised(ce, X) .* (std_vec ⊗ std_vec)
end

export SmythBrobyGerber0NormalisedCovariance
