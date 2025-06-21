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
struct RatioRiskMeasure{T1 <: JuMPReturnsEstimator, T2 <: AbstractBaseRiskMeasure,
                        T3 <: Real} <: NoOptimisationRiskMeasure
    rt::T1
    rk::T2
    rf::T3
end
function RatioRiskMeasure(; rt::JuMPReturnsEstimator = ArithmeticReturn(),
                          rk::AbstractBaseRiskMeasure = Variance(), rf::Real = 0.0)
    return RatioRiskMeasure{typeof(rt), typeof(rk), typeof(rf)}(rt, rk, rf)
end
function factory(r::RatioRiskMeasure, prior::AbstractPriorResult, args...; kwargs...)
    rt = jump_returns_factory(r.rt, prior, args...; kwargs...)
    rk = factory(r.rk, prior, args...; kwargs...)
    return RatioRiskMeasure(; rt = rt, rk = rk, r.rf)
end
function factory(r::RatioRiskMeasure, w::AbstractVector)
    return RatioRiskMeasure(; rt = r.rt, rk = factory(r.rk, w), rf = r.rf)
end
#! Need to change other expected risks
function expected_risk(r::RatioRiskMeasure, w::AbstractVector{<:Real},
                       pr::AbstractPriorResult, fees::Union{Nothing, <:Fees} = nothing;
                       kwargs...)
    return expected_sharpe_ratio(r.rk, r.rt, w, pr, fees; rf = r.rf, kwargs...)
end

export expected_returns, expected_sharpe_ratio, expected_risk_ret_sharpe_ratio,
       expected_sric, expected_risk_ret_sric
