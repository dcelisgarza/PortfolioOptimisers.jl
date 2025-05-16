abstract type AbstractMomentMeasureAlgorithm <: AbstractAlgorithm end
abstract type AbstractUnionLowOrderMomentMeasureAlgorithm <: AbstractMomentMeasureAlgorithm end
abstract type AbstractLowOrderMomentMeasureAlgorithm <:
              AbstractUnionLowOrderMomentMeasureAlgorithm end
abstract type AbstractLowOrderDeviationMeasureAlgorithm <:
              AbstractUnionLowOrderMomentMeasureAlgorithm end
function risk_moment_algorithm_factory(alg::AbstractMomentMeasureAlgorithm, args...;
                                       kwargs...)
    return alg
end
struct FirstLowerMoment <: AbstractLowOrderMomentMeasureAlgorithm end
struct SecondLowerMoment{T1 <: SecondMomentFormulation} <:
       AbstractLowOrderMomentMeasureAlgorithm
    formulation::T1
end
function SecondLowerMoment(; formulation::SecondMomentFormulation = SOCRiskExpr())
    return SecondLowerMoment{typeof(formulation)}(formulation)
end
struct SecondCentralMoment{T1 <: SecondMomentFormulation} <:
       AbstractLowOrderMomentMeasureAlgorithm
    formulation::T1
end
struct MeanAbsoluteDeviation <: AbstractLowOrderMomentMeasureAlgorithm end
struct LowOrderDeviation{T1 <: AbstractVarianceEstimator,
                         T2 <: AbstractLowOrderMomentMeasureAlgorithm} <:
       AbstractLowOrderDeviationMeasureAlgorithm
    ve::T1
    alg::T2
end
function LowOrderDeviation(; ve::AbstractVarianceEstimator = SimpleVariance(; me = nothing),
                           alg::AbstractLowOrderMomentMeasureAlgorithm = FirstLowerMoment())
    @smart_assert(!isa(alg, MeanAbsoluteDeviation))
    return LowOrderDeviation{typeof(ve), typeof(alg)}(ve, alg)
end
abstract type AbstractUnionHighOrderMomentMeasureAlgorithm <: AbstractMomentMeasureAlgorithm end
abstract type AbstractHighOrderMomentMeasureAlgorithm <:
              AbstractUnionHighOrderMomentMeasureAlgorithm end
abstract type AbstractHighOrderDeviationMeasureAlgorithm <:
              AbstractUnionHighOrderMomentMeasureAlgorithm end
struct ThirdLowerMoment <: AbstractHighOrderMomentMeasureAlgorithm end
struct FourthLowerMoment <: AbstractHighOrderMomentMeasureAlgorithm end
struct FourthCentralMoment <: AbstractHighOrderMomentMeasureAlgorithm end
struct HighOrderDeviation{T1 <: AbstractVarianceEstimator,
                          T2 <: AbstractHighOrderMomentMeasureAlgorithm} <:
       AbstractHighOrderDeviationMeasureAlgorithm
    ve::T1
    alg::T2
end
function HighOrderDeviation(;
                            ve::AbstractVarianceEstimator = SimpleVariance(; me = nothing),
                            alg::AbstractHighOrderMomentMeasureAlgorithm = ThirdLowerMoment())
    return HighOrderDeviation{typeof(ve), typeof(alg)}(ve, alg)
end
for alg ∈ (LowOrderDeviation, HighOrderDeviation)
    eval(quote
             function factory(alg::$(alg), w::Union{Nothing, <:AbstractWeights} = nothing)
                 return $(alg)(; ve = factory(alg.ve, w), alg = alg.alg)
             end
         end)
end
struct LowOrderMoment{T1 <: RiskMeasureSettings, T2 <: Union{Nothing, <:AbstractWeights},
                      T3 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                      T4 <: AbstractUnionLowOrderMomentMeasureAlgorithm} <:
       AbstractMomentRiskMeasure
    settings::T1
    w::T2
    mu::T3
    alg::T4
end
function LowOrderMoment(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                        w::Union{Nothing, <:AbstractWeights} = nothing,
                        mu::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                        alg::AbstractUnionLowOrderMomentMeasureAlgorithm = FirstLowerMoment())
    if isa(mu, AbstractVector)
        @smart_assert(!isempty(mu) && all(isfinite, mu))
    elseif isa(mu, Real)
        @smart_assert(isfinite(mu))
    end
    return LowOrderMoment{typeof(settings), typeof(w), typeof(mu), typeof(alg)}(settings, w,
                                                                                mu, alg)
end
struct HighOrderMoment{T1 <: RiskMeasureSettings, T2 <: Union{Nothing, <:AbstractWeights},
                       T3 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                       T4 <: AbstractUnionHighOrderMomentMeasureAlgorithm} <:
       AbstractMomentHierarchicalRiskMeasure
    settings::T1
    w::T2
    mu::T3
    alg::T4
end
function HighOrderMoment(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                         w::Union{Nothing, <:AbstractWeights} = nothing,
                         mu::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                         alg::AbstractUnionHighOrderMomentMeasureAlgorithm = ThirdLowerMoment())
    if isa(mu, AbstractVector)
        @smart_assert(!isempty(mu) && all(isfinite, mu))
    elseif isa(mu, Real)
        @smart_assert(isfinite(mu))
    end
    return HighOrderMoment{typeof(settings), typeof(w), typeof(mu), typeof(alg)}(settings,
                                                                                 w, mu, alg)
end
function calc_moment_target(::Union{<:LowOrderMoment{<:Any, <:Any, Nothing, Nothing},
                                    <:HighOrderMoment{<:Any, <:Any, Nothing, Nothing}},
                            ::Any, x::AbstractVector)
    return mean(x)
end
function calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:Any, <:AbstractWeights,
                                                      Nothing},
                                     <:HighOrderMoment{<:Any, <:Any, <:AbstractWeights,
                                                       Nothing}}, ::Any, x::AbstractVector)
    return mean(x, r.w)
end
function calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:Any, <:Any,
                                                      <:AbstractVector},
                                     <:HighOrderMoment{<:Any, <:Any, <:Any,
                                                       <:AbstractVector}},
                            w::AbstractVector, ::Any)
    return dot(w, r.mu)
end
function calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:Any, <:Any, <:Real},
                                     <:HighOrderMoment{<:Any, <:Any, <:Any, <:Real}}, ::Any,
                            ::Any)
    return r.mu
end
function calc_moment_val(r::Union{<:AbstractMomentRiskMeasure,
                                  <:AbstractMomentHierarchicalRiskMeasure,
                                  <:AbstractMomentNoOptimisationRiskMeasure},
                         w::AbstractVector, X::AbstractMatrix,
                         fees::Union{Nothing, <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    target = calc_moment_target(r, w, x)
    return x .- target
end
function (r::LowOrderMoment{<:Any, <:FirstLowerMoment, <:Any, <:Any})(w::AbstractVector,
                                                                      X::AbstractMatrix,
                                                                      fees::Union{Nothing,
                                                                                  <:Fees} = nothing)
    val = min.(calc_moment_val(r, w, X, fees), zero(eltype(X)))
    return isnothing(r.w) ? -mean(val) : -mean(val, r.w)
end
function (r::LowOrderMoment{<:Any, <:LowOrderDeviation{<:Any, <:FirstLowerMoment}, <:Any,
                            <:Any})(w::AbstractVector, X::AbstractMatrix,
                                    fees::Union{Nothing, <:Fees} = nothing)
    val = min.(calc_moment_val(r, w, X, fees), zero(eltype(X)))
    return StatsBase.std(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::LowOrderMoment{<:Any, <:LowOrderDeviation{<:Any, <:SecondLowerMoment}, <:Any,
                            <:Any})(w::AbstractVector, X::AbstractMatrix,
                                    fees::Union{Nothing, <:Fees} = nothing)
    val = min.(calc_moment_val(r, w, X, fees), zero(eltype(X)))
    return StatsBase.var(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::LowOrderMoment{<:Any, <:LowOrderDeviation{<:Any, <:SecondCentralMoment}, <:Any,
                            <:Any})(w::AbstractVector, X::AbstractMatrix,
                                    fees::Union{Nothing, <:Fees} = nothing)
    val = calc_moment_val(r, w, X, fees)
    return StatsBase.var(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::LowOrderMoment{<:Any, <:MeanAbsoluteDeviation, <:Any, <:Any})(w::AbstractVector,
                                                                           X::AbstractMatrix,
                                                                           fees::Union{Nothing,
                                                                                       <:Fees} = nothing)
    val = abs.(calc_moment_val(r, w, X, fees))
    return isnothing(r.w) ? mean(val) : mean(val, r.w)
end
function (r::HighOrderMoment{<:Any, <:ThirdLowerMoment, <:Any, <:Any})(w::AbstractVector,
                                                                       X::AbstractMatrix,
                                                                       fees::Union{Nothing,
                                                                                   <:Fees} = nothing)
    val = min.(calc_moment_val(r, w, X, fees), zero(eltype(X)))
    val .= val .^ 3
    return isnothing(r.w) ? -mean(val) : -mean(val, r.w)
end
function (r::HighOrderMoment{<:Any, <:FourthLowerMoment, <:Any, <:Any})(w::AbstractVector,
                                                                        X::AbstractMatrix,
                                                                        fees::Union{Nothing,
                                                                                    <:Fees} = nothing)
    val = min.(calc_moment_val(r, w, X, fees), zero(eltype(X)))
    val .= val .^ 4
    return isnothing(r.w) ? mean(val) : mean(val, r.w)
end
function (r::HighOrderMoment{<:Any, <:FourthCentralMoment, <:Any, <:Any})(w::AbstractVector,
                                                                          X::AbstractMatrix,
                                                                          fees::Union{Nothing,
                                                                                      <:Fees} = nothing)
    val = calc_moment_val(r, w, X, fees)
    val .= val .^ 4
    return isnothing(r.w) ? mean(val) : mean(val, r.w)
end
function (r::HighOrderMoment{<:Any, <:HighOrderDeviation{<:Any, <:ThirdLowerMoment}, <:Any,
                             <:Any})(w::AbstractVector, X::AbstractMatrix,
                                     fees::Union{Nothing, <:Fees} = nothing)
    val = min.(calc_moment_val(r, w, X, fees), zero(eltype(X)))
    sigma = StatsBase.std(r.alg.ve, val; mean = zero(eltype(val)))
    val .= val .^ 3
    res = isnothing(r.w) ? -mean(val) : -mean(val, r.w)
    return res / sigma^3
end
function (r::HighOrderMoment{<:Any, <:HighOrderDeviation{<:Any, <:FourthLowerMoment}, <:Any,
                             <:Any})(w::AbstractVector, X::AbstractMatrix,
                                     fees::Union{Nothing, <:Fees} = nothing)
    val = min.(calc_moment_val(r, w, X, fees), zero(eltype(X)))
    sigma = StatsBase.var(r.alg.ve, val; mean = zero(eltype(val)))
    val .= val .^ 4
    res = isnothing(r.w) ? mean(val) : mean(val, r.w)
    return res / sigma^2
end
function (r::HighOrderMoment{<:Any, <:HighOrderDeviation{<:Any, <:FourthCentralMoment},
                             <:Any, <:Any})(w::AbstractVector, X::AbstractMatrix,
                                            fees::Union{Nothing, <:Fees} = nothing)
    val = calc_moment_val(r, w, X, fees)
    sigma = StatsBase.var(r.alg.ve, val; mean = zero(eltype(val)))
    val .= val .^ 4
    res = isnothing(r.w) ? mean(val) : mean(val, r.w)
    return res / sigma^2
end
for rt ∈ (LowOrderMoment, HighOrderMoment)
    eval(quote
             function risk_measure_factory(r::$(rt), prior::AbstractPriorResult, args...;
                                           kwargs...)
                 w = risk_measure_nothing_scalar_array_factory(r.w, prior.w)
                 mu = risk_measure_nothing_scalar_array_factory(r.mu, prior.mu)
                 alg = risk_moment_algorithm_factory(r.alg, prior.w)
                 return $(rt)(; settings = r.settings, alg = alg, w = w, mu = mu)
             end
             function risk_measure_view(r::$(rt), i::AbstractVector, args...)
                 mu = nothing_scalar_array_view(r.mu, i)
                 return $(rt)(; settings = r.settings, alg = r.alg, w = r.w, mu = mu)
             end
         end)
end

export FirstLowerMoment, SecondLowerMoment, MeanAbsoluteDeviation, ThirdLowerMoment,
       FourthLowerMoment, FourthCentralMoment, LowOrderDeviation, HighOrderDeviation,
       LowOrderMoment, HighOrderMoment
