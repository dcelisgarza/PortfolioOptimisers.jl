# https://portfoliooptimizationbook.com/slides/slides-index-tracking.pdf
struct RiskTrackingError{T1, T2, T3, T4} <: AbstractTracking
    tr::T1
    r::T2
    err::T3
    alg::T4
    function RiskTrackingError(tr::WeightsTracking, r::AbstractBaseRiskMeasure, err::Number,
                               alg::VariableTracking)
        @argcheck(isfinite(err))
        @argcheck(err >= zero(err))
        r = no_bounds_no_risk_expr_risk_measure(r)
        return new{typeof(tr), typeof(r), typeof(err), typeof(alg)}(tr, r, err, alg)
    end
end
function RiskTrackingError(; tr::WeightsTracking,
                           r::AbstractBaseRiskMeasure = StandardDeviation(),
                           err::Number = 0.0,
                           alg::VariableTracking = IndependentVariableTracking())
    return RiskTrackingError(tr, r, err, alg)
end
function tracking_view(::Nothing, args...)
    return nothing
end
function tracking_view(tr::RiskTrackingError, i, X::MatNum)
    return RiskTrackingError(; tr = tracking_view(tr.tr, i),
                             r = risk_measure_view(tr.r, i, X), err = tr.err, alg = tr.alg)
end
function factory(tr::RiskTrackingError, prior::AbstractPriorResult, args...; kwargs...)
    return RiskTrackingError(; tr = tr.tr, r = factory(tr.r, prior, args...; kwargs...),
                             err = tr.err, alg = tr.alg)
end
function factory(tr::RiskTrackingError, w::VecNum)
    return RiskTrackingError(; tr = factory(tr.tr, w), r = tr.r, err = tr.err, alg = tr.alg)
end
struct TrackingRiskMeasure{T1, T2, T3} <: RiskMeasure
    settings::T1
    tr::T2
    alg::T3
    function TrackingRiskMeasure(settings::RiskMeasureSettings,
                                 tr::AbstractTrackingAlgorithm, alg::NormTracking)
        return new{typeof(settings), typeof(tr), typeof(alg)}(settings, tr, alg)
    end
end
function TrackingRiskMeasure(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                             tr::AbstractTrackingAlgorithm,
                             alg::NormTracking = SOCTracking())
    return TrackingRiskMeasure(settings, tr, alg)
end
function (r::TrackingRiskMeasure)(w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)
    benchmark = tracking_benchmark(r.tr, X)
    return norm_tracking(r.alg, calc_net_returns(w, X, fees), benchmark, size(X, 1))
end
function risk_measure_view(r::TrackingRiskMeasure, i, args...)
    tr = tracking_view(r.tr, i)
    return TrackingRiskMeasure(; settings = r.settings, tr = tr, alg = r.alg)
end
function factory(r::TrackingRiskMeasure, prior::AbstractPriorResult, args...; kwargs...)
    return TrackingRiskMeasure(; settings = r.settings, tr = r.tr, alg = r.alg)
end
function factory(r::TrackingRiskMeasure, w::VecNum)
    return TrackingRiskMeasure(; settings = r.settings, tr = factory(r.tr, w), alg = r.alg)
end
struct RiskTrackingRiskMeasure{T1, T2, T3, T4} <: RiskMeasure
    settings::T1
    tr::T2
    r::T3
    alg::T4
    function RiskTrackingRiskMeasure(settings::RiskMeasureSettings, tr::WeightsTracking,
                                     r::AbstractBaseRiskMeasure, alg::VariableTracking)
        if isa(alg, DependentVariableTracking) && isa(r, QuadExpressionRiskMeasures)
            @warn("Risk measures that produce QuadExpr risk expressions are not guaranteed to work. The variance with SDP constraints works because the risk measure is the trace of a matrix, an affine expression.")
        end
        r = no_bounds_no_risk_expr_risk_measure(r)
        return new{typeof(settings), typeof(tr), typeof(r), typeof(alg)}(settings, tr, r,
                                                                         alg)
    end
end
function RiskTrackingRiskMeasure(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                 tr::WeightsTracking,
                                 r::AbstractBaseRiskMeasure = Variance(),
                                 alg::VariableTracking = IndependentVariableTracking())
    return RiskTrackingRiskMeasure(settings, tr, r, alg)
end
function (r::RiskTrackingRiskMeasure{<:Any, <:Any, <:AbstractBaseRiskMeasure,
                                     <:IndependentVariableTracking})(w::VecNum, X::MatNum,
                                                                     fees::Option{<:Fees} = nothing)
    wb = r.tr.w
    wd = w - wb
    return expected_risk(r.r, wd, X, fees)
end
function (r::RiskTrackingRiskMeasure{<:Any, <:Any, <:AbstractBaseRiskMeasure,
                                     <:DependentVariableTracking})(w::VecNum, X::MatNum,
                                                                   fees::Option{<:Fees} = nothing)
    wb = r.tr.w
    r1 = expected_risk(r.r, w, X, fees)
    r2 = expected_risk(r.r, wb, X, fees)
    return abs(r1 - r2)
end
function risk_measure_view(r::RiskTrackingRiskMeasure, i, X::MatNum)
    tr = tracking_view(r.tr, i)
    return RiskTrackingRiskMeasure(; settings = r.settings, tr = tr,
                                   r = risk_measure_view(r.r, i, X), alg = r.alg)
end
function factory(r::RiskTrackingRiskMeasure, prior::AbstractPriorResult, args...; kwargs...)
    return RiskTrackingRiskMeasure(; settings = r.settings, tr = r.tr,
                                   r = factory(r.r, prior, args...; kwargs...), alg = r.alg)
end
function factory(r::RiskTrackingRiskMeasure, w::VecNum)
    return RiskTrackingRiskMeasure(; settings = r.settings, tr = factory(r.tr, w), r = r.r,
                                   alg = r.alg)
end

export TrackingRiskMeasure, RiskTrackingRiskMeasure, RiskTrackingError
