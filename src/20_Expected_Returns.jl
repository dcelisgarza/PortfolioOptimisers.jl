function expected_return(::ArithmeticReturn, w::AbstractVector{<:Real},
                         pr::AbstractPriorResult, fees::Union{Nothing, Fees} = nothing;
                         kwargs...)
    mu = pr.mu
    return dot(w, mu) - calc_fees(w, fees)
end
function expected_return(ret::KellyReturn, w::AbstractVector{<:Real},
                         pr::AbstractPriorResult, fees::Union{Nothing, Fees} = nothing;
                         kwargs...)
    rw = ret.w
    X = pr.X
    kret = isnothing(rw) ? mean(log1p.(X * w)) : mean(log1p.(X * w), rw)
    return kret - calc_fees(w, fees)
end
function expected_return(ret::JuMPReturnsEstimator, w::AbstractVector{<:AbstractVector},
                         pr::AbstractPriorResult, fees::Union{Nothing, Fees} = nothing;
                         kwargs...)
    return [expected_return(ret, wi, pr, fees; kwargs...) for wi in w]
end
function expected_ratio(r::AbstractBaseRiskMeasure, ret::JuMPReturnsEstimator,
                        w::AbstractVector, pr::AbstractPriorResult,
                        fees::Union{Nothing, Fees} = nothing; rf::Real = 0, kwargs...)
    rk = expected_risk(r, w, pr.X, fees; kwargs...)
    rt = expected_return(ret, w, pr, fees; kwargs...)
    return (rt - rf) / rk
end
function expected_risk_ret_ratio(r::AbstractBaseRiskMeasure, ret::JuMPReturnsEstimator,
                                 w::AbstractVector, pr::AbstractPriorResult,
                                 fees::Union{Nothing, Fees} = nothing; rf::Real = 0,
                                 kwargs...)
    rk = expected_risk(r, w, pr.X, fees; kwargs...)
    rt = expected_return(ret, w, pr, fees; kwargs...)
    return rk, rt, (rt - rf) / rk
end
function expected_sric(r::AbstractBaseRiskMeasure, ret::JuMPReturnsEstimator,
                       w::AbstractVector, pr::AbstractPriorResult,
                       fees::Union{Nothing, Fees} = nothing; rf::Real = 0, kwargs...)
    T, N = size(pr.X)
    sr = expected_ratio(r, ret, w, pr; fees = fees, rf = rf, kwargs...)
    return sr - N / (T * sr)
end
function expected_risk_ret_sric(r::AbstractBaseRiskMeasure, ret::JuMPReturnsEstimator,
                                w::AbstractVector, pr::AbstractPriorResult,
                                fees::Union{Nothing, Fees} = nothing; rf::Real = 0,
                                kwargs...)
    T, N = size(pr.X)
    rk, rt, sr = expected_risk_ret_ratio(r, ret, w, pr; fees = fees, rf = rf, kwargs...)
    return rk, rt, sr - N / (T * sr)
end
struct ReturnRiskMeasure{T1} <: NoOptimisationRiskMeasure
    rt::T1
end
function ReturnRiskMeasure(; rt::JuMPReturnsEstimator = ArithmeticReturn())
    return ReturnRiskMeasure(rt)
end
function factory(r::ReturnRiskMeasure, prior::AbstractPriorResult, args...; kwargs...)
    rt = jump_returns_factory(r.rt, prior, args...; kwargs...)
    return ReturnRiskMeasure(; rt = rt)
end
function factory(r::ReturnRiskMeasure, args...)
    return r
end
function expected_risk(r::ReturnRiskMeasure, w::AbstractVector{<:Real},
                       pr::AbstractPriorResult, fees::Union{Nothing, <:Fees} = nothing;
                       kwargs...)
    return expected_return(r.rt, w, pr, fees)
end
struct RatioRiskMeasure{T1, T2, T3} <: NoOptimisationRiskMeasure
    rt::T1
    rk::T2
    rf::T3
end
function RatioRiskMeasure(; rt::JuMPReturnsEstimator = ArithmeticReturn(),
                          rk::AbstractBaseRiskMeasure = Variance(), rf::Real = 0.0)
    return RatioRiskMeasure(rt, rk, rf)
end
function factory(r::RatioRiskMeasure, prior::AbstractPriorResult, args...; kwargs...)
    rt = jump_returns_factory(r.rt, prior, args...; kwargs...)
    rk = factory(r.rk, prior, args...; kwargs...)
    return RatioRiskMeasure(; rt = rt, rk = rk, r.rf)
end
function factory(r::RatioRiskMeasure, w::AbstractVector)
    return RatioRiskMeasure(; rt = r.rt, rk = factory(r.rk, w), rf = r.rf)
end
function expected_risk(r::RatioRiskMeasure, w::AbstractVector{<:Real},
                       pr::AbstractPriorResult, fees::Union{Nothing, <:Fees} = nothing;
                       kwargs...)
    return expected_ratio(r.rk, r.rt, w, pr, fees; rf = r.rf, kwargs...)
end

export expected_return, expected_ratio, expected_risk_ret_ratio, expected_sric,
       expected_risk_ret_sric, RatioRiskMeasure, ReturnRiskMeasure
