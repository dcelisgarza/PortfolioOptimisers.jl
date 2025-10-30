"""
    expected_return(ret::ArithmeticReturn, w::AbstractVector{<:Real}, pr::AbstractPriorResult;
                    fees::Union{Nothing, Fees} = nothing, kwargs...)
    expected_return(ret::KellyReturn, w::AbstractVector{<:Real}, pr::AbstractPriorResult;
                    fees::Union{Nothing, Fees} = nothing, kwargs...)
    expected_return(ret::JuMPReturnsEstimator, w::AbstractVector{<:AbstractVector},
                    pr::AbstractPriorResult; fees::Union{Nothing, Fees} = nothing, kwargs...)

Compute the expected portfolio return using the specified return estimator.

`expected_return` calculates the expected return for a portfolio given its weights, a prior result, and optional transaction fees. Supports arithmetic, Kelly, and JuMP-based return estimators. For Kelly returns, computes the mean log-growth rate. For JuMP-based estimators, returns a vector of expected returns for each portfolio.

# Arguments

  - `ret`: Return estimator.
  - `w`: Portfolio weights.
  - `pr`: Prior result.
  - `fees`: Optional fees.
  - `kwargs...`: Additional keyword arguments passed to underlying routines.

# Returns

  - `rt::Union{<:Real, Vector{<:Real}}`: Expected portfolio return(s), net of fees if provided.

# Details

  - For the third method, `expected_return` is broadcast over the vector of vectors of portfolio weights.

# Related

  - [`ArithmeticReturn`](@ref)
  - [`KellyReturn`](@ref)
  - [`JuMPReturnsEstimator`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`expected_ratio`](@ref)
  - [`expected_risk_ret_ratio`](@ref)
  - [`expected_sric`](@ref)
  - [`expected_risk_ret_sric`](@ref)
  - [`calc_fees`](@ref)
"""
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
"""
    expected_ratio(r::AbstractBaseRiskMeasure, ret::JuMPReturnsEstimator, w::AbstractVector,
                   pr::AbstractPriorResult; fees::Union{Nothing, Fees} = nothing, rf::Real = 0,
                   kwargs...)

Compute the expected risk-adjusted return ratio for a portfolio.

`expected_ratio` calculates the ratio of expected portfolio return (net of fees and risk-free rate) to expected portfolio risk, using the specified risk measure and return estimator.

# Arguments

  - `r`: Risk measure.
  - `ret`: Return estimator.
  - `w`: Portfolio weights.
  - `pr`: Prior result.
  - `fees`: Optional transaction fees.
  - `rf`: Risk-free rate.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `ratio::Real`: Risk-adjusted return ratio.

# Related

  - [`JuMPReturnsEstimator`](@ref)
  - [`expected_return`](@ref)
  - [`expected_risk_ret_ratio`](@ref)
  - [`expected_sric`](@ref)
"""
function expected_ratio(r::AbstractBaseRiskMeasure, ret::JuMPReturnsEstimator,
                        w::AbstractVector, pr::AbstractPriorResult,
                        fees::Union{Nothing, Fees} = nothing; rf::Real = 0, kwargs...)
    rk = expected_risk(r, w, pr.X, fees; kwargs...)
    rt = expected_return(ret, w, pr, fees; kwargs...)
    return (rt - rf) / rk
end
"""
    expected_risk_ret_ratio(r::AbstractBaseRiskMeasure, ret::JuMPReturnsEstimator,
                            w::AbstractVector, pr::AbstractPriorResult;
                            fees::Union{Nothing, Fees} = nothing, rf::Real = 0, kwargs...)

Compute expected risk, expected return, and risk-adjusted return ratio for a portfolio.

`expected_risk_ret_ratio` returns a tuple containing the expected portfolio risk, expected portfolio return, and the risk-adjusted return ratio, using the specified risk measure and return estimator.

# Arguments

  - `r`: Risk measure.
  - `ret`: Return estimator.
  - `w`: Portfolio weights.
  - `pr`: Prior result.
  - `fees`: Optional transaction fees.
  - `rf`: Risk-free rate.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `(risk::Real, return::Real, ratio::Real)`: Tuple of expected risk, expected return, and risk-adjusted return ratio.

# Related

  - [`JuMPReturnsEstimator`](@ref)
  - [`expected_ratio`](@ref)
  - [`expected_return`](@ref)
  - [`expected_sric`](@ref)
"""
function expected_risk_ret_ratio(r::AbstractBaseRiskMeasure, ret::JuMPReturnsEstimator,
                                 w::AbstractVector, pr::AbstractPriorResult,
                                 fees::Union{Nothing, Fees} = nothing; rf::Real = 0,
                                 kwargs...)
    rk = expected_risk(r, w, pr.X, fees; kwargs...)
    rt = expected_return(ret, w, pr, fees; kwargs...)
    return rk, rt, (rt - rf) / rk
end
"""
    expected_sric(r::AbstractBaseRiskMeasure, ret::JuMPReturnsEstimator, w::AbstractVector,
                  pr::AbstractPriorResult; fees::Union{Nothing, Fees} = nothing, rf::Real = 0,
                  kwargs...)

Compute the risk-adjusted ratio information criterion (SRIC) for a portfolio.

`expected_sric` calculates the SRIC, which adjusts the risk-adjusted return ratio for estimation error, penalizing overfitting in portfolio optimization. The SRIC is computed as the risk-adjusted return ratio minus a penalty term based on the number of assets and sample size.

# Arguments

  - `r`: Risk measure.
  - `ret`: Return estimator.
  - `w`: Portfolio weights.
  - `pr`: Prior result.
  - `fees`: Optional transaction fees.
  - `rf`: Risk-free rate.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `sric::Real`: Sharpe Ratio Information Criterion.

# Related

  - [`JuMPReturnsEstimator`](@ref)
  - [`expected_ratio`](@ref)
  - [`expected_risk_ret_ratio`](@ref)
  - [`expected_risk_ret_sric`](@ref)
"""
function expected_sric(r::AbstractBaseRiskMeasure, ret::JuMPReturnsEstimator,
                       w::AbstractVector, pr::AbstractPriorResult,
                       fees::Union{Nothing, Fees} = nothing; rf::Real = 0, kwargs...)
    T, N = size(pr.X)
    sr = expected_ratio(r, ret, w, pr; fees = fees, rf = rf, kwargs...)
    return sr - N / (T * sr)
end
"""
    expected_risk_ret_sric(r::AbstractBaseRiskMeasure, ret::JuMPReturnsEstimator,
                           w::AbstractVector, pr::AbstractPriorResult;
                           fees::Union{Nothing, Fees} = nothing, rf::Real = 0, kwargs...)

Compute expected risk, expected return, and SRIC for a portfolio.

`expected_risk_ret_sric` returns a tuple containing the expected portfolio risk, expected portfolio return, and the Sharpe Ratio Information Criterion (SRIC), which adjusts the risk-adjusted return ratio for estimation error.

# Arguments

  - `r`: Risk measure (`AbstractBaseRiskMeasure`).
  - `ret`: Return estimator (`JuMPReturnsEstimator`).
  - `w`: Portfolio weights.
  - `pr`: Prior result (must contain asset return matrix `X`).
  - `fees`: Optional transaction fees.
  - `rf`: Risk-free rate (default: 0).
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `(risk::Real, return::Real, sric::Real)`: Tuple of expected risk, expected return, and SRIC.

# Related

  - [`JuMPReturnsEstimator`](@ref)
  - [`expected_sric`](@ref)
  - [`expected_ratio`](@ref)
  - [`expected_risk_ret_ratio`](@ref)
"""
function expected_risk_ret_sric(r::AbstractBaseRiskMeasure, ret::JuMPReturnsEstimator,
                                w::AbstractVector, pr::AbstractPriorResult,
                                fees::Union{Nothing, Fees} = nothing; rf::Real = 0,
                                kwargs...)
    T, N = size(pr.X)
    rk, rt, sr = expected_risk_ret_ratio(r, ret, w, pr; fees = fees, rf = rf, kwargs...)
    return rk, rt, sr - N / (T * sr)
end
"""
    struct ReturnRiskMeasure{T1} <: NoOptimisationRiskMeasure
        rt::T1
    end

Return-based risk measure.

`ReturnRiskMeasure` is a risk measure that uses the expected portfolio return as its risk metric. This is useful for algorithms or analyses where the risk is defined as the expected return, used in portfolio performance analysis.

# Fields

  - `rt`: Return estimator.

# Constructor

    ReturnRiskMeasure(; rt::JuMPReturnsEstimator = ArithmeticReturn())

# Examples

```jldoctest
julia> ReturnRiskMeasure()
ReturnRiskMeasure
  rt ┼ ArithmeticReturn
     │   ucs ┼ nothing
     │    lb ┴ nothing
```

# Related

  - [`JuMPReturnsEstimator`](@ref)
  - [`RatioRiskMeasure`](@ref)
  - [`expected_return`](@ref)
  - [`expected_risk`](@ref)
"""
struct ReturnRiskMeasure{T1} <: NoOptimisationRiskMeasure
    rt::T1
    function ReturnRiskMeasure(rt::JuMPReturnsEstimator)
        return new{typeof(rt)}(rt)
    end
end
function ReturnRiskMeasure(; rt::JuMPReturnsEstimator = ArithmeticReturn())
    return ReturnRiskMeasure(rt)
end
function factory(r::ReturnRiskMeasure, prior::AbstractPriorResult, args...; kwargs...)
    rt = jump_returns_factory(r.rt, prior, args...; kwargs...)
    return ReturnRiskMeasure(; rt = rt)
end
function factory(r::ReturnRiskMeasure, args...; kwargs...)
    return r
end
"""
    expected_risk(r::ReturnRiskMeasure, w::AbstractVector{<:Real}, pr::AbstractPriorResult;
                  fees::Union{Nothing, <:Fees} = nothing, kwargs...)

Compute the expected risk for a portfolio using a return-based risk measure.

`expected_risk` returns the expected portfolio return as the risk metric, using the specified return estimator in the [`ReturnRiskMeasure`](@ref). This is useful for algorithms where risk is defined as expected return.

# Arguments

  - `r`: [`ReturnRiskMeasure`](@ref) containing a return estimator.
  - `w`: Portfolio weights.
  - `pr`: Prior result.
  - `fees`: Optional transaction fees.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `risk::Real`: Expected portfolio return (net of fees if provided).

# Related

  - [`ReturnRiskMeasure`](@ref)
  - [`expected_return`](@ref)
  - [`expected_ratio`](@ref)
"""
function expected_risk(r::ReturnRiskMeasure, w::AbstractVector{<:Real},
                       pr::AbstractPriorResult, fees::Union{Nothing, <:Fees} = nothing;
                       kwargs...)
    return expected_return(r.rt, w, pr, fees)
end
"""
    struct RatioRiskMeasure{T1, T2, T3} <: NoOptimisationRiskMeasure
        rt::T1
        rk::T2
        rf::T3
    end

Ratio-based risk measure.

`RatioRiskMeasure` is a risk measure that computes the risk-adjusted return ratio, such as the Sharpe ratio, for a portfolio. It combines a return estimator, a risk measure, and a risk-free rate to produce a ratio metric, used in portfolio performance analysis.

# Fields

  - `rt`: Return estimator.
  - `rk`: Risk measure.
  - `rf`: Risk-free rate.

# Constructor

    RatioRiskMeasure(; rt::JuMPReturnsEstimator = ArithmeticReturn(),
                     rk::AbstractBaseRiskMeasure = Variance(), rf::Real = 0.0)

# Examples

```jldoctest
julia> RatioRiskMeasure()
RatioRiskMeasure
  rt ┼ ArithmeticReturn
     │   ucs ┼ nothing
     │    lb ┴ nothing
  rk ┼ Variance
     │   settings ┼ RiskMeasureSettings
     │            │   scale ┼ Float64: 1.0
     │            │      ub ┼ nothing
     │            │     rke ┴ Bool: true
     │      sigma ┼ nothing
     │         rc ┼ nothing
     │        alg ┴ SquaredSOCRiskExpr()
  rf ┴ Float64: 0.0
```

# Related

  - [`JuMPReturnsEstimator`](@ref)
  - [`ReturnRiskMeasure`](@ref)
  - [`expected_ratio`](@ref)
  - [`expected_risk`](@ref)
"""
struct RatioRiskMeasure{T1, T2, T3} <: NoOptimisationRiskMeasure
    rt::T1
    rk::T2
    rf::T3
    function RatioRiskMeasure(rt::JuMPReturnsEstimator, rk::AbstractBaseRiskMeasure,
                              rf::Real)
        return new{typeof(rt), typeof(rk), typeof(rf)}(rt, rk, rf)
    end
end
function RatioRiskMeasure(; rt::JuMPReturnsEstimator = ArithmeticReturn(),
                          rk::AbstractBaseRiskMeasure = Variance(), rf::Real = 0.0)
    return RatioRiskMeasure(rt, rk, rf)
end
function factory(r::RatioRiskMeasure, prior::AbstractPriorResult, args...; kwargs...)
    rt = jump_returns_factory(r.rt, prior, args...; kwargs...)
    rk = factory(r.rk, prior, args...; kwargs...)
    return RatioRiskMeasure(; rt = rt, rk = rk, rf = r.rf)
end
function factory(r::RatioRiskMeasure, w::AbstractVector)
    return RatioRiskMeasure(; rt = r.rt, rk = factory(r.rk, w), rf = r.rf)
end
"""
    expected_risk(r::RatioRiskMeasure, w::AbstractVector{<:Real}, pr::AbstractPriorResult;
                  fees::Union{Nothing, <:Fees} = nothing, kwargs...)

Compute the expected risk for a portfolio using a ratio-based risk measure.

`expected_risk` returns the risk-adjusted return ratio (e.g., Sharpe ratio) for the portfolio, using the specified return estimator, risk measure, and risk-free rate in the [`RatioRiskMeasure`](@ref).

# Arguments

  - `r`: [`RatioRiskMeasure`](@ref) containing a return estimator, risk measure, and risk-free rate.
  - `w`: Portfolio weights.
  - `pr`: Prior result.
  - `fees`: Optional transaction fees.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `risk::Real`: Risk-adjusted return ratio.

# Related

  - [`RatioRiskMeasure`](@ref)
  - [`expected_ratio`](@ref)
  - [`expected_return`](@ref)
"""
function expected_risk(r::RatioRiskMeasure, w::AbstractVector{<:Real},
                       pr::AbstractPriorResult, fees::Union{Nothing, <:Fees} = nothing;
                       kwargs...)
    return expected_ratio(r.rk, r.rt, w, pr, fees; rf = r.rf, kwargs...)
end

export expected_return, expected_ratio, expected_risk_ret_ratio, expected_sric,
       expected_risk_ret_sric, RatioRiskMeasure, ReturnRiskMeasure
