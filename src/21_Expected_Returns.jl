function expected_returns(::ArithmeticReturn, w::AbstractVector{<:Real},
                          pr::AbstractPriorResult, fees::Union{Nothing, Fees} = nothing)
    mu = pr.mu
    return dot(w, mu) - calc_fees(w, fees)
end
function expected_returns(ret::KellyReturn, w::AbstractVector{<:Real},
                          pr::AbstractPriorResult, fees::Union{Nothing, Fees} = nothing)
    rw = ret.w
    X = pr.X
    kret = isnothing(rw) ? mean(log1p.(X * w)) : mean(log1p.(X * w), rw)
    return kret - calc_fees(w, fees)
end
function expected_returns(ret::JuMPReturnsEstimator, w::AbstractVector{<:AbstractVector},
                          pr::AbstractPriorResult, fees::Union{Nothing, Fees} = nothing)
    return expected_returns.(Ref(ret), w, Ref(pr), Ref(fees))
end
function expected_sharpe_ratio(r::AbstractBaseRiskMeasure, ret::JuMPReturnsEstimator,
                               w::AbstractVector, pr::AbstractPriorResult,
                               fees::Union{Nothing, Fees} = nothing; rf::Real = 0,
                               kwargs...)
    rk = expected_risk(r, w, pr.X, fees; kwargs...)
    rt = expected_returns(ret, w, pr, fees; kwargs...)
    return (rt - rf) / rk
end
function expected_risk_ret_sharpe_ratio(r::AbstractBaseRiskMeasure,
                                        ret::JuMPReturnsEstimator, w::AbstractVector,
                                        pr::AbstractPriorResult,
                                        fees::Union{Nothing, Fees} = nothing; rf::Real = 0,
                                        kwargs...)
    rk = expected_risk(r, w, pr.X, fees; kwargs...)
    rt = expected_returns(ret, w, pr, fees; kwargs...)
    return rk, rt, (rt - rf) / rk
end
function expected_sric(r::AbstractBaseRiskMeasure, ret::JuMPReturnsEstimator,
                       w::AbstractVector, pr::AbstractPriorResult,
                       fees::Union{Nothing, Fees} = nothing; rf::Real = 0, kwargs...)
    T, N = size(pr.X)
    sr = expected_sharpe_ratio(r, ret, w, pr; fees = fees, rf = rf, kwargs...)
    return sr - N / (T * sr)
end
function expected_risk_ret_sric(r::AbstractBaseRiskMeasure, ret::JuMPReturnsEstimator,
                                w::AbstractVector, pr::AbstractPriorResult,
                                fees::Union{Nothing, Fees} = nothing; rf::Real = 0,
                                kwargs...)
    T, N = size(pr.X)
    rk, rt, sr = expected_risk_ret_sharpe_ratio(r, ret, w, pr; fees = fees, rf = rf,
                                                kwargs...)
    return rk, rt, sr - N / (T * sr)
end

export expected_returns, expected_sharpe_ratio, expected_risk_ret_sharpe_ratio,
       expected_sric, expected_risk_ret_sric
