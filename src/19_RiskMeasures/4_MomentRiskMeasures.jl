abstract type AbstractMomentMeasureAlgorithm <: AbstractAlgorithm end
abstract type AbstractLowOrderMomentMeasureAlgorithm <: AbstractMomentMeasureAlgorithm end
abstract type AbstractHighOrderMomentMeasureAlgorithm <: AbstractMomentMeasureAlgorithm end
function risk_moment_algorithm_factory(alg::AbstractMomentMeasureAlgorithm, args...;
                                       kwargs...)
    return alg
end
struct FirstLowerMoment <: AbstractLowOrderMomentMeasureAlgorithm end
struct SemiDeviation{T1 <: Integer} <: AbstractLowOrderMomentMeasureAlgorithm
    ddof::T1
end
function SemiDeviation(; ddof::Integer = 1)
    @smart_assert(ddof >= 0)
    return SemiDeviation{typeof(ddof)}(ddof)
end
struct SemiVariance{T1 <: Integer, T2 <: VarianceFormulation} <:
       AbstractLowOrderMomentMeasureAlgorithm
    ddof::T1
    formulation::T2
end
function SemiVariance(; ddof::Integer = 1, formulation::VarianceFormulation = SOC())
    @smart_assert(ddof >= 0)
    return SemiVariance{typeof(ddof), typeof(formulation)}(ddof, formulation)
end
struct MeanAbsoluteDeviation{T1 <: Union{Nothing, <:AbstractWeights}} <:
       AbstractLowOrderMomentMeasureAlgorithm
    w::T1
end
function MeanAbsoluteDeviation(; w::Union{Nothing, <:AbstractWeights} = nothing)
    if isa(w, AbstractVector)
        @smart_assert(!isempty(w))
    end
    return MeanAbsoluteDeviation{typeof(w)}(w)
end
function risk_moment_algorithm_factory(alg::MeanAbsoluteDeviation,
                                       w::Union{<:Nothing, <:AbstractWeights}; kwargs...)
    return MeanAbsoluteDeviation(; w = w)
end
struct ThirdLowerMoment <: AbstractHighOrderMomentMeasureAlgorithm end
struct FourthLowerMoment <: AbstractHighOrderMomentMeasureAlgorithm end
struct FourthCentralMoment <: AbstractHighOrderMomentMeasureAlgorithm end
abstract type AbstractHighOrderDeviationAlgorithm <: AbstractHighOrderMomentMeasureAlgorithm end
struct HighOrderDeviation{T1 <: AbstractVarianceEstimator,
                          T2 <: AbstractHighOrderMomentMeasureAlgorithm} <:
       AbstractHighOrderDeviationAlgorithm
    ve::T1
    alg::T2
end
function HighOrderDeviation(;
                            ve::AbstractVarianceEstimator = SimpleVariance(; me = nothing),
                            alg::AbstractHighOrderMomentMeasureAlgorithm = ThirdLowerMoment())
    if hasproperty(ve, :me)
        @smart_assert(isnothing(ve.me))
    end
    return HighOrderDeviation{typeof(ve), typeof(alg)}(ve, alg)
end
function risk_moment_algorithm_factory(alg::HighOrderDeviation,
                                       w::Union{<:Nothing, <:AbstractWeights}; kwargs...)
    return HighOrderDeviation(; ve = factory(alg.ve, w), alg = alg.alg)
end
struct LowOrderMoment{T1 <: RiskMeasureSettings,
                      T2 <: AbstractLowOrderMomentMeasureAlgorithm,
                      T3 <: Union{Nothing, <:AbstractWeights},
                      T4 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}}} <:
       AbstractMomentRiskMeasure
    settings::T1
    alg::T2
    w::T3
    mu::T4
end
function LowOrderMoment(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                        alg::AbstractLowOrderMomentMeasureAlgorithm = FirstLowerMoment(),
                        w::Union{Nothing, <:AbstractWeights} = nothing,
                        mu::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing)
    if isa(mu, AbstractVector)
        @smart_assert(!isempty(mu) && all(isfinite, mu))
    elseif isa(mu, Real)
        @smart_assert(isfinite(mu))
    end
    return LowOrderMoment{typeof(settings), typeof(alg), typeof(w), typeof(mu)}(settings,
                                                                                alg, w, mu)
end
struct HighOrderMoment{T1 <: RiskMeasureSettings,
                       T2 <: AbstractHighOrderMomentMeasureAlgorithm,
                       T3 <: Union{Nothing, <:AbstractWeights},
                       T4 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}}} <:
       AbstractMomentHierarchicalRiskMeasure
    settings::T1
    alg::T2
    w::T3
    mu::T4
end
function HighOrderMoment(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                         alg::AbstractHighOrderMomentMeasureAlgorithm = ThirdLowerMoment(),
                         w::Union{Nothing, <:AbstractWeights} = nothing,
                         mu::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing)
    if isa(mu, AbstractVector)
        @smart_assert(!isempty(mu) && all(isfinite, mu))
    elseif isa(mu, Real)
        @smart_assert(isfinite(mu))
    end
    return HighOrderMoment{typeof(settings), typeof(alg), typeof(w), typeof(mu)}(settings,
                                                                                 alg, w, mu)
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
    return x ⊖ target
end
function (r::LowOrderMoment{<:Any, <:FirstLowerMoment, <:Any, <:Any})(w::AbstractVector,
                                                                      X::AbstractMatrix,
                                                                      fees::Union{Nothing,
                                                                                  <:Fees} = nothing)
    val = calc_moment_val(r, w, X, fees)
    val = val[val .<= zero(eltype(val))]
    return -sum(val) / size(X, 1)
end
function (r::LowOrderMoment{<:Any, <:SemiDeviation, <:Any, <:Any})(w::AbstractVector,
                                                                   X::AbstractMatrix,
                                                                   fees::Union{Nothing,
                                                                               <:Fees} = nothing)
    val = calc_moment_val(r, w, X, fees)
    val = val[val .<= zero(eltype(val))]
    return sqrt(dot(val, val) / (size(X, 1) - r.alg.ddof))
end
function (r::LowOrderMoment{<:Any, <:SemiVariance, <:Any, <:Any})(w::AbstractVector,
                                                                  X::AbstractMatrix,
                                                                  fees::Union{Nothing,
                                                                              <:Fees} = nothing)
    val = calc_moment_val(r, w, X, fees)
    val = val[val .<= zero(eltype(val))]
    return dot(val, val) / (size(X, 1) - r.alg.ddof)
end
function mean_abs_dev(::Nothing, x::AbstractVector)
    return mean(abs.(x))
end
function mean_abs_dev(w::AbstractWeights, x::AbstractVector)
    return mean(abs.(x), w)
end
function (r::LowOrderMoment{<:Any, <:MeanAbsoluteDeviation, <:Any, <:Any})(w::AbstractVector,
                                                                           X::AbstractMatrix,
                                                                           fees::Union{Nothing,
                                                                                       <:Fees} = nothing)
    val = calc_moment_val(r, w, X, fees)
    return mean_abs_dev(r.alg.w, val)
end
function (r::HighOrderMoment{<:Any, <:ThirdLowerMoment, <:Any, <:Any})(w::AbstractVector,
                                                                       X::AbstractMatrix,
                                                                       fees::Union{Nothing,
                                                                                   <:Fees} = nothing)
    val = calc_moment_val(r, w, X, fees)
    val = val[val .<= zero(eltype(val))]
    return -sum(val .^ 3) / size(X, 1)
end
function (r::HighOrderMoment{<:Any, <:FourthLowerMoment, <:Any, <:Any})(w::AbstractVector,
                                                                        X::AbstractMatrix,
                                                                        fees::Union{Nothing,
                                                                                    <:Fees} = nothing)
    val = calc_moment_val(r, w, X, fees)
    val = val[val .<= zero(eltype(val))]
    return sum(val .^ 4) / size(X, 1)
end
function (r::HighOrderMoment{<:Any, <:FourthCentralMoment, <:Any, <:Any})(w::AbstractVector,
                                                                          X::AbstractMatrix,
                                                                          fees::Union{Nothing,
                                                                                      <:Fees} = nothing)
    val = calc_moment_val(r, w, X, fees)
    return sum(val .^ 4) / size(X, 1)
end
function (r::HighOrderMoment{<:Any, <:HighOrderDeviation{<:Any, <:ThirdLowerMoment}, <:Any,
                             <:Any})(w::AbstractVector, X::AbstractMatrix,
                                     fees::Union{Nothing, <:Fees} = nothing)
    val = calc_moment_val(r, w, X, fees)
    val = val[val .<= zero(eltype(val))]
    sigma = StatsBase.std(r.alg.ve, val; mean = zero(eltype(val)))
    return -sum(val .^ 3) / size(X, 1) / sigma^3
end
function (r::HighOrderMoment{<:Any, <:HighOrderDeviation{<:Any, <:FourthLowerMoment}, <:Any,
                             <:Any})(w::AbstractVector, X::AbstractMatrix,
                                     fees::Union{Nothing, <:Fees} = nothing)
    val = calc_moment_val(r, w, X, fees)
    val = val[val .<= zero(eltype(val))]
    sigma = StatsBase.std(r.alg.ve, val; mean = zero(eltype(val)))
    return sum(val .^ 4) / size(X, 1) / sigma^4
end
function (r::HighOrderMoment{<:Any, <:HighOrderDeviation{<:Any, <:FourthCentralMoment},
                             <:Any, <:Any})(w::AbstractVector, X::AbstractMatrix,
                                            fees::Union{Nothing, <:Fees} = nothing)
    val = calc_moment_val(r, w, X, fees)
    sigma = StatsBase.std(r.alg.ve, val; mean = zero(eltype(val)))
    return sum(val .^ 4) / size(X, 1) / sigma^4
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

export FirstLowerMoment, SemiDeviation, SemiVariance, MeanAbsoluteDeviation,
       ThirdLowerMoment, FourthLowerMoment, FourthCentralMoment, HighOrderDeviation,
       LowOrderMoment, HighOrderMoment
