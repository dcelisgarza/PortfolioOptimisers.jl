"""
    expected_return(ret::ArithmeticReturn, w::VecNum, pr::AbstractPriorResult;
                    fees::Option{<:Fees} = nothing, kwargs...)
    expected_return(ret::LogarithmicReturn, w::VecNum, pr::AbstractPriorResult;
                    fees::Option{<:Fees} = nothing, kwargs...)
    expected_return(ret::JuMPReturnsEstimator, w::VecVecNum, pr::AbstractPriorResult;
                    fees::Option{<:Fees} = nothing, kwargs...)

Compute the expected portfolio return using the specified return estimator.

`expected_return` computes the expected return for a portfolio given its weights, a prior result, and optional transaction fees. Supports arithmetic, logarithmic, and JuMP-based return estimators. For logarithmic returns, computes the mean log-growth rate. For JuMP-based estimators, returns a vector of expected returns for each portfolio.

# Arguments

  - `ret`: Return estimator.
  - `w`: Portfolio weights.
  - `pr`: Prior result.
  - `fees`: Optional fees.
  - `kwargs...`: Additional keyword arguments passed to underlying routines.

# Returns

  - `rt::Num_VecNum`: Expected portfolio return(s), net of fees if provided.

# Details

  - For the third method, `expected_return` is broadcast over the vector of vectors of portfolio weights.

# Related

  - [`ArithmeticReturn`](@ref)
  - [`LogarithmicReturn`](@ref)
  - [`JuMPReturnsEstimator`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`VecNum`](@ref)
  - [`VecVecNum`](@ref)
  - [`Num_VecNum`](@ref)
  - [`Option`](@ref)
  - [`expected_ratio`](@ref)
  - [`expected_risk_ret_ratio`](@ref)
  - [`expected_sric`](@ref)
  - [`expected_risk_ret_sric`](@ref)
  - [`calc_fees`](@ref)
"""
function expected_return(r::ArithmeticReturn, w::VecNum, pr::AbstractPriorResult,
                         fees::Option{<:Fees} = nothing; kwargs...)
    mu = nothing_scalar_array_selector(r.mu, pr.mu)
    return LinearAlgebra.dot(w, mu) - calc_fees(w, fees)
end
function expected_return(ret::LogarithmicReturn, w::VecNum, pr::AbstractPriorResult,
                         fees::Option{<:Fees} = nothing; kwargs...)
    rw = ret.w
    X = pr.X
    kret = if isnothing(rw)
        Statistics.mean(log1p.(X * w))
    else
        Statistics.mean(log1p.(X * w), rw)
    end
    return kret - calc_fees(w, fees)
end
function expected_return(ret::JuMPReturnsEstimator, w::VecVecNum, pr::AbstractPriorResult,
                         fees::Option{<:Fees} = nothing; kwargs...)
    return [expected_return(ret, wi, pr, fees; kwargs...) for wi in w]
end
"""
    expected_ratio(r::AbstractBaseRiskMeasure, ret::JuMPReturnsEstimator, w::VecNum,
                   pr::AbstractPriorResult; fees::Option{<:Fees} = nothing, rf::Number = 0,
                   kwargs...)

Compute the expected risk-adjusted return ratio for a portfolio.

`expected_ratio` computes the ratio of expected portfolio return (net of fees and risk-free rate) to expected portfolio risk, using the specified risk measure and return estimator.

# Arguments

  - `r`: Risk measure.
  - `ret`: Return estimator.
  - `w`: Portfolio weights.
  - `pr`: Prior result.
  - `fees`: Optional transaction fees.
  - `rf`: Risk-free rate.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `ratio::Number`: Risk-adjusted return ratio.

# Related

  - [`AbstractBaseRiskMeasure`](@ref)
  - [`JuMPReturnsEstimator`](@ref)
  - [`VecNum`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`Option`](@ref)
  - [`Fees`](@ref)
  - [`expected_return`](@ref)
  - [`expected_risk_ret_ratio`](@ref)
  - [`expected_sric`](@ref)
"""
function expected_ratio(r::AbstractBaseRiskMeasure, ret::JuMPReturnsEstimator, w::VecNum,
                        pr::AbstractPriorResult, fees::Option{<:Fees} = nothing;
                        rf::Number = 0, kwargs...)
    rk = expected_risk(r, w, pr.X, fees; kwargs...)
    rt = expected_return(ret, w, pr, fees; kwargs...)
    return (rt - rf) / rk
end
"""
    expected_risk_ret_ratio(r::AbstractBaseRiskMeasure, ret::JuMPReturnsEstimator, w::VecNum,
                            pr::AbstractPriorResult; fees::Option{<:Fees} = nothing,
                            rf::Number = 0, kwargs...)

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

  - `risk::Number`: Expected portfolio risk.
  - `return::Number`: Expected portfolio return.
  - `ratio::Number`: Risk-adjusted return ratio.

# Related

  - [`AbstractBaseRiskMeasure`](@ref)
  - [`JuMPReturnsEstimator`](@ref)
  - [`VecNum`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`Option`](@ref)
  - [`Fees`](@ref)
  - [`expected_ratio`](@ref)
  - [`expected_return`](@ref)
  - [`expected_sric`](@ref)
"""
function expected_risk_ret_ratio(r::AbstractBaseRiskMeasure, ret::JuMPReturnsEstimator,
                                 w::VecNum, pr::AbstractPriorResult,
                                 fees::Option{<:Fees} = nothing; rf::Number = 0, kwargs...)
    rk = expected_risk(r, w, pr.X, fees; kwargs...)
    rt = expected_return(ret, w, pr, fees; kwargs...)
    return rk, rt, (rt - rf) / rk
end
"""
    expected_sric(r::AbstractBaseRiskMeasure, ret::JuMPReturnsEstimator, w::VecNum,
                  pr::AbstractPriorResult; fees::Option{<:Fees} = nothing, rf::Number = 0,
                  kwargs...)

Compute the risk-adjusted ratio information criterion (SRIC) for a portfolio.

`expected_sric` computes the SRIC, which adjusts the risk-adjusted return ratio for estimation error, penalizing overfitting in portfolio optimization. The SRIC is computed as the risk-adjusted return ratio minus a penalty term based on the number of assets and sample size.

# Arguments

  - `r`: Risk measure.
  - `ret`: Return estimator.
  - `w`: Portfolio weights.
  - `pr`: Prior result.
  - `fees`: Optional transaction fees.
  - `rf`: Risk-free rate.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `sric::Number`: Sharpe Ratio Information Criterion.

# Related

  - [`AbstractBaseRiskMeasure`](@ref)
  - [`JuMPReturnsEstimator`](@ref)
  - [`VecNum`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`Option`](@ref)
  - [`Fees`](@ref)
  - [`expected_ratio`](@ref)
  - [`expected_risk_ret_ratio`](@ref)
  - [`expected_risk_ret_sric`](@ref)
"""
function expected_sric(r::AbstractBaseRiskMeasure, ret::JuMPReturnsEstimator, w::VecNum,
                       pr::AbstractPriorResult, fees::Option{<:Fees} = nothing;
                       rf::Number = 0, kwargs...)
    T, N = size(pr.X)
    sr = expected_ratio(r, ret, w, pr; fees = fees, rf = rf, kwargs...)
    return sr - N / (T * sr)
end
"""
    expected_risk_ret_sric(r::AbstractBaseRiskMeasure, ret::JuMPReturnsEstimator, w::VecNum,
                           pr::AbstractPriorResult; fees::Option{<:Fees} = nothing,
                           rf::Number = 0, kwargs...)

Compute expected risk, expected return, and SRIC for a portfolio.

`expected_risk_ret_sric` returns a tuple containing the expected portfolio risk, expected portfolio return, and the Sharpe Ratio Information Criterion (SRIC), which adjusts the risk-adjusted return ratio for estimation error.

# Arguments

  - `r`: Risk measure.
  - `ret`: Return estimator.
  - `w`: Portfolio weights.
  - `pr`: Prior result (must contain asset return matrix `X`).
  - `fees`: Optional transaction fees.
  - `rf`: Risk-free rate (default: 0).
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `risk::Number`: Expected portfolio risk.
  - `return::Number`: Expected portfolio return.
  - `sric::Number`: Sharpe Ratio Information Criterion.

# Related

  - [`AbstractBaseRiskMeasure`](@ref)
  - [`JuMPReturnsEstimator`](@ref)
  - [`VecNum`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`Option`](@ref)
  - [`Fees`](@ref)
  - [`expected_sric`](@ref)
  - [`expected_ratio`](@ref)
  - [`expected_risk_ret_ratio`](@ref)
"""
function expected_risk_ret_sric(r::AbstractBaseRiskMeasure, ret::JuMPReturnsEstimator,
                                w::VecNum, pr::AbstractPriorResult,
                                fees::Option{<:Fees} = nothing; rf::Number = 0, kwargs...)
    T, N = size(pr.X)
    rk, rt, sr = expected_risk_ret_ratio(r, ret, w, pr; fees = fees, rf = rf, kwargs...)
    return rk, rt, sr - N / (T * sr)
end
"""
$(DocStringExtensions.TYPEDEF)

Return-based risk measure.

`ExpectedReturn` is a risk measure that uses the expected portfolio return as its risk metric. This is useful for algorithms or analyses where the risk is defined as the expected return, used in portfolio performance analysis.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExpectedReturn(;
        rt::JuMPReturnsEstimator = ArithmeticReturn()
    ) -> ExpectedReturn

Keywords correspond to the struct's fields.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `rt`: Recursively updated via [`factory`](@ref).

# Examples

```jldoctest
julia> ExpectedReturn()
ExpectedReturn
  rt ┼ ArithmeticReturn
     │   ucs ┼ nothing
     │    lb ┼ nothing
     │    mu ┴ nothing
```

# Related

  - [`JuMPReturnsEstimator`](@ref)
  - [`ExpectedReturnRiskRatio`](@ref)
  - [`expected_return`](@ref)
  - [`expected_risk`](@ref)
  - [`factory`](@ref)
"""
@propagatable @concrete struct ExpectedReturn <: NonOptimisationRiskMeasure
    """
    $(field_dict[:rt])
    """
    @fprop rt
    function ExpectedReturn(rt::JuMPReturnsEstimator)
        return new{typeof(rt)}(rt)
    end
end
function ExpectedReturn(; rt::JuMPReturnsEstimator = ArithmeticReturn())::ExpectedReturn
    return ExpectedReturn(rt)
end
"""
    expected_risk(r::ExpectedReturn, w::VecNum, pr::AbstractPriorResult;
                  fees::Option{<:Fees} = nothing, kwargs...)

Compute the expected risk for a portfolio using a return-based risk measure.

`expected_risk` returns the expected portfolio return as the risk metric, using the specified return estimator in the [`ExpectedReturn`](@ref). This is useful for algorithms where risk is defined as expected return.

# Arguments

  - `r`: [`ExpectedReturn`](@ref) containing a return estimator.
  - `w`: Portfolio weights.
  - `pr`: Prior result.
  - `fees`: Optional transaction fees.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `risk::Number`: Expected portfolio return (net of fees if provided).

# Related

  - [`ExpectedReturn`](@ref)
  - [`expected_return`](@ref)
  - [`expected_ratio`](@ref)
"""
function expected_risk(r::ExpectedReturn, w::VecNum, pr::AbstractPriorResult,
                       fees::Option{<:Fees} = nothing; kwargs...)
    return expected_return(r.rt, w, pr, fees)
end
"""
$(DocStringExtensions.TYPEDEF)

Ratio-based risk measure.

`ExpectedReturnRiskRatio` is a risk measure that computes the risk-adjusted return ratio, such as the Sharpe ratio, for a portfolio. It combines a return estimator, a risk measure, and a risk-free rate to produce a ratio metric, used in portfolio performance analysis.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExpectedReturnRiskRatio(;
        rt::JuMPReturnsEstimator = ArithmeticReturn(),
        rk::AbstractBaseRiskMeasure = Variance(),
        rf::Number = 0.0,
    ) -> ExpectedReturnRiskRatio

Keywords correspond to the struct's fields.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `rt`: Recursively updated via [`factory`](@ref).
  - `rk`: Recursively updated via [`factory`](@ref).

# Examples

```jldoctest
julia> ExpectedReturnRiskRatio()
ExpectedReturnRiskRatio
  rt ┼ ArithmeticReturn
     │   ucs ┼ nothing
     │    lb ┼ nothing
     │    mu ┴ nothing
  rk ┼ Variance
     │   settings ┼ RiskMeasureSettings
     │            │   scale ┼ Float64: 1.0
     │            │      ub ┼ nothing
     │            │     rke ┴ Bool: true
     │      sigma ┼ nothing
     │       chol ┼ nothing
     │         rc ┼ nothing
     │        alg ┴ SquaredSOCRiskExpr()
  rf ┴ Float64: 0.0
```

# Related

  - [`JuMPReturnsEstimator`](@ref)
  - [`AbstractBaseRiskMeasure`](@ref)
  - [`expected_ratio`](@ref)
  - [`expected_risk`](@ref)
  - [`factory`](@ref)
"""
@propagatable @concrete struct ExpectedReturnRiskRatio <: NonOptimisationRiskMeasure
    """
    $(field_dict[:rt])
    """
    @fprop rt
    """
    $(field_dict[:rk])
    """
    @fprop rk
    """
    $(field_dict[:rf])
    """
    rf
    function ExpectedReturnRiskRatio(rt::JuMPReturnsEstimator, rk::AbstractBaseRiskMeasure,
                                     rf::Number)
        return new{typeof(rt), typeof(rk), typeof(rf)}(rt, rk, rf)
    end
end
function ExpectedReturnRiskRatio(; rt::JuMPReturnsEstimator = ArithmeticReturn(),
                                 rk::AbstractBaseRiskMeasure = Variance(),
                                 rf::Number = 0.0)::ExpectedReturnRiskRatio
    return ExpectedReturnRiskRatio(rt, rk, rf)
end
"""
    expected_risk(r::ExpectedReturnRiskRatio, w::VecNum, pr::AbstractPriorResult;
                  fees::Option{<:Fees} = nothing, kwargs...)

Compute the expected risk for a portfolio using a ratio-based risk measure.

`expected_risk` returns the risk-adjusted return ratio (e.g., Sharpe ratio) for the portfolio, using the specified return estimator, risk measure, and risk-free rate in the [`ExpectedReturnRiskRatio`](@ref).

# Arguments

  - `r`: [`ExpectedReturnRiskRatio`](@ref) containing a return estimator, risk measure, and risk-free rate.
  - `w`: Portfolio weights.
  - `pr`: Prior result.
  - `fees`: Optional transaction fees.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `risk::Number`: Risk-adjusted return ratio.

# Related

  - [`ExpectedReturnRiskRatio`](@ref)
  - [`VecNum`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`Option`](@ref)
  - [`Fees`](@ref)
  - [`expected_ratio`](@ref)
  - [`expected_return`](@ref)
"""
function expected_risk(r::ExpectedReturnRiskRatio, w::VecNum, pr::AbstractPriorResult,
                       fees::Option{<:Fees} = nothing; kwargs...)
    return expected_ratio(r.rk, r.rt, w, pr, fees; rf = r.rf, kwargs...)
end
function needs_previous_weights(r::ExpectedReturnRiskRatio)
    return needs_previous_weights(r.rk)
end
"""
    const PerfRM = Union{...}

Union of performance risk measures used to compute portfolio performance metrics (returns and return/risk ratios).

# Related

  - [`MeanReturn`](@ref)
  - [`MeanReturnRiskRatio`](@ref)
  - [`ExpectedReturn`](@ref)
  - [`ExpectedReturnRiskRatio`](@ref)
"""
const PerfRM = Union{<:MeanReturn, <:MeanReturnRiskRatio, <:ExpectedReturn,
                     <:ExpectedReturnRiskRatio}
"""
    const PrRM = Union{<:ExpectedReturn, <:ExpectedReturnRiskRatio}

Union of prior-based return risk measures that are incompatible with [`PredictionResult`](@ref) inputs and require the use of [`MeanReturn`](@ref) or [`MeanReturnRiskRatio`](@ref) instead.

# Related

  - [`ExpectedReturn`](@ref)
  - [`ExpectedReturnRiskRatio`](@ref)
  - [`PerfRM`](@ref)
"""
const PrRM = Union{<:ExpectedReturn, <:ExpectedReturnRiskRatio}
function bigger_is_better(::PerfRM)
    return true
end
function expected_risk(r::PrRM,
                       pred::PredictionResult{<:Any,
                                              <:PredictionReturnsResult{<:Any, <:VecNum}};
                       kwargs...)
    return throw(ArgumentError("risk measure $r is incompatible for computing the `expected_risk` of a $(Base.typename(typeof(pred)).wrapper), please use $(isa(r, ExpectedReturn) ? MeanReturn : MeanReturnRiskRatio) instead"))
end
function expected_risk(r::PrRM,
                       pred::PredictionResult{<:Any,
                                              <:PredictionReturnsResult{<:Any, <:VecVecNum}};
                       kwargs...)
    return throw(ArgumentError("risk measure $r is incompatible for computing the `expected_risk` of a $(Base.typename(typeof(pred)).wrapper), please use $(isa(r, ExpectedReturn) ? MeanReturn : MeanReturnRiskRatio) instead"))
end
function expected_risk(r::PrRM, pred::MultiPeriodPredictionResult; kwargs...)
    return throw(ArgumentError("risk measure $r is incompatible for computing the `expected_risk` of a $(Base.typename(typeof(pred)).wrapper), please use $(isa(r, ExpectedReturn) ? MeanReturn : MeanReturnRiskRatio) instead"))
end
function expected_risk(r::PrRM, pred::PopulationPredictionResult; kwargs...)
    return throw(ArgumentError("risk measure $r is incompatible for computing the `expected_risk` of a $(Base.typename(typeof(pred)).wrapper), please use $(isa(r, ExpectedReturn) ? MeanReturn : MeanReturnRiskRatio) instead"))
end
"""
    brinson_attribution(X::TimeArray, w::VecNum, wb::VecNum,
                        asset_classes::DataFrame, col; date0 = nothing, date1 = nothing)

Compute Brinson performance attribution aggregated per asset class [brinson_attribution](@cite).

`brinson_attribution` generates a DataFrame summarizing the Brinson performance attribution, decomposing total excess return into asset allocation, security selection, interaction, and total effect for each asset class. The calculation is performed over the specified date range, using the provided asset weights, benchmark weights, and asset class assignments.

# Arguments

  - `X`: TimeArray of asset prices or returns.
  - `w`: Vector of portfolio weights.
  - `wb`: Vector of benchmark weights.
  - `asset_classes`: DataFrame containing asset class assignments for each asset.
  - `col`: Column in `asset_classes` specifying the class for each asset.
  - `date0`: (Optional) Start date for the attribution period.
  - `date1`: (Optional) End date for the attribution period.

# Returns

  - `df::DataFrame`: DataFrame with rows for asset allocation, security selection, interaction, and total effect, and columns for each asset class and the total.

# Details

  - Computes returns for the specified period.
  - Aggregates attribution effects by asset class.
  - Supports custom date ranges via `date0` and `date1`.
  - Returns a DataFrame with attribution breakdown for each class and the total.

# Related

  - [`VecNum`](@ref)

# References

  - [brinson_attribution](@cite) G. P. Brinson and N. Fachler. *Measuring non-US. equity portfolio performance*. The Journal of Portfolio Management 11, 73–76 (1985).
"""
function brinson_attribution(X::TimeSeries.TimeArray, w::VecNum, wb::VecNum,
                             asset_classes::DataFrames.DataFrame, col, date0 = nothing,
                             date1 = nothing)
    # Efficient filtering of date range
    idx1, idx2 = if !isnothing(date0) && !isnothing(date1)
        timestamps = TimeSeries.timestamp(X)
        idx = (Dates.DateTime(date0) .<= timestamps) .&
              (timestamps .<= Dates.DateTime(date1))
        findfirst(idx), findlast(idx)
    else
        1, length(X)
    end

    ret = vec(values(X[idx2]) ./ values(X[idx1]) .- 1)
    ret_b = LinearAlgebra.dot(ret, wb)

    classes = asset_classes[!, col]
    unique_classes = unique(classes)

    df = DataFrames.DataFrame(;
                              index = ["Asset Allocation", "Security Selection",
                                       "Interaction", "Total Excess Return"])

    # Precompute class membership matrix for efficiency
    sets_mat = [class_j == class_i for class_j in classes, class_i in unique_classes]

    for (i, class_i) in enumerate(unique_classes)
        sets_i = view(sets_mat, :, i)

        w_i = LinearAlgebra.dot(sets_i, w)
        wb_i = LinearAlgebra.dot(sets_i, wb)

        ret_i = LinearAlgebra.dot(ret .* sets_i, w) / w_i
        ret_b_i = LinearAlgebra.dot(ret .* sets_i, wb) / wb_i

        w_diff_i = w_i - wb_i
        ret_diff_i = ret_i - ret_b_i

        AA_i = w_diff_i * (ret_b_i - ret_b)
        SS_i = wb_i * ret_diff_i
        I_i = w_diff_i * ret_diff_i
        TER_i = AA_i + SS_i + I_i

        df[!, class_i] = [AA_i, SS_i, I_i, TER_i]
    end

    df[!, "Total"] = sum(eachcol(df[!, 2:end]))

    return df
end

export expected_return, expected_ratio, expected_risk_ret_ratio, expected_sric,
       expected_risk_ret_sric, ExpectedReturn, ExpectedReturnRiskRatio, brinson_attribution
