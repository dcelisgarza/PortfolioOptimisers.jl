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
    if isa(w, AbstractWeights)
        @smart_assert(!isempty(w))
    end
    return MeanAbsoluteDeviation{typeof(w)}(w)
end
function risk_moment_algorithm_factory(alg::MeanAbsoluteDeviation,
                                       w::Union{<:Nothing, AbstractWeights}; kwargs...)
    return MeanAbsoluteDeviation(; w = w)
end
struct ThirdLowerMoment <: AbstractHighOrderMomentMeasureAlgorithm end
struct FourthLowerMoment <: AbstractHighOrderMomentMeasureAlgorithm end
struct FourthCentralMoment <: AbstractHighOrderMomentMeasureAlgorithm end
abstract type AbstractHighOrderDeviationAlgorithm <: AbstractHighOrderMomentMeasureAlgorithm end
struct HighOrderDeviation{T1 <: AbstractHighOrderMomentMeasureAlgorithm,
                          T2 <: AbstractVarianceEstimator} <:
       AbstractHighOrderDeviationAlgorithm
    alg::T1
    ve::T2
end
function HighOrderDeviation(;
                            alg::AbstractHighOrderMomentMeasureAlgorithm = ThirdLowerMoment(),
                            ve::AbstractVarianceEstimator = SimpleVariance())
    return HighOrderDeviation{typeof(alg), typeof(ve)}(alg, ve)
end
function risk_moment_algorithm_factory(alg::HighOrderDeviation,
                                       w::Union{<:Nothing, <:AbstractWeights}; kwargs...)
    return HighOrderDeviation(; alg = alg.alg, ve = factory(alg.ve, w))
end
struct LowOrderMoment{T1 <: RiskMeasureSettings,
                      T2 <: AbstractLowOrderMomentMeasureAlgorithm,
                      T3 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                      T4 <: Union{Nothing, <:AbstractWeights},
                      T5 <: Union{Nothing, <:AbstractVector{<:Real}}} <:
       AbstractMomentRiskMeasure
    settings::T1
    alg::T2
    target::T3
    w::T4
    mu::T5
end
function LowOrderMoment(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                        alg::AbstractLowOrderMomentMeasureAlgorithm = FirstLowerMoment(),
                        target::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = 0.0,
                        w::Union{Nothing, <:AbstractWeights} = nothing,
                        mu::Union{Nothing, <:AbstractVector{<:Real}} = nothing)
    if isa(target, AbstractVector)
        @smart_assert(!isempty(target))
    end
    if isa(mu, AbstractVector)
        @smart_assert(!isempty(mu))
    end
    return LowOrderMoment{typeof(settings), typeof(alg), typeof(target), typeof(w),
                          typeof(mu)}(settings, alg, target, w, mu)
end
function (r::LowOrderMoment{<:Any, <:FirstLowerMoment, <:Any, <:Any, <:Any})(w::AbstractVector,
                                                                             X::AbstractMatrix,
                                                                             fees::Union{Nothing,
                                                                                         <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    target = calc_target_ret_mu(x, w, r)
    val = x .- target
    val = val[val .<= zero(eltype(val))]
    return -sum(val) / length(x)
end
function (r::LowOrderMoment{<:Any, <:SemiDeviation, <:Any, <:Any, <:Any})(w::AbstractVector,
                                                                          X::AbstractMatrix,
                                                                          fees::Union{Nothing,
                                                                                      <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    mu = calc_target_ret_mu(x, w, r)
    val = x .- mu
    val = val[val .<= zero(eltype(val))]
    return sqrt(dot(val, val) / (length(x) - r.alg.ddof))
end
function (r::LowOrderMoment{<:Any, <:SemiVariance, <:Any, <:Any, <:Any})(w::AbstractVector,
                                                                         X::AbstractMatrix,
                                                                         fees::Union{Nothing,
                                                                                     <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    mu = calc_target_ret_mu(x, w, r)
    val = x .- mu
    val = val[val .<= zero(eltype(val))]
    return dot(val, val) / (length(x) - r.alg.ddof)
end
function (r::LowOrderMoment{<:Any, <:MeanAbsoluteDeviation, <:Any, <:Any, <:Any})(w::AbstractVector,
                                                                                  X::AbstractMatrix,
                                                                                  fees::Union{Nothing,
                                                                                              <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    mu = calc_target_ret_mu(x, w, r)
    w = r.alg.w
    return isnothing(w) ? mean(abs.(x .- mu)) : mean(abs.(x .- mu), w)
end
struct HighOrderMoment{T1 <: RiskMeasureSettings,
                       T2 <: AbstractHighOrderMomentMeasureAlgorithm,
                       T3 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                       T4 <: Union{Nothing, <:AbstractWeights},
                       T5 <: Union{Nothing, <:AbstractVector{<:Real}}} <:
       AbstractMomentHierarchicalRiskMeasure
    settings::T1
    alg::T2
    target::T3
    w::T4
    mu::T5
end
function HighOrderMoment(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                         alg::AbstractHighOrderMomentMeasureAlgorithm = ThirdLowerMoment(),
                         target::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                         w::Union{Nothing, <:AbstractWeights} = nothing,
                         mu::Union{Nothing, <:AbstractVector{<:Real}} = nothing)
    if isa(target, AbstractVector)
        @smart_assert(!isempty(target))
    end
    if isa(mu, AbstractVector)
        @smart_assert(!isempty(mu))
    end
    return HighOrderMoment{typeof(settings), typeof(alg), typeof(target), typeof(w),
                           typeof(mu)}(settings, alg, target, w, mu)
end
function (r::HighOrderMoment{<:Any, <:ThirdLowerMoment, <:Any, <:Any, <:Any})(w::AbstractVector,
                                                                              X::AbstractMatrix,
                                                                              fees::Union{Nothing,
                                                                                          <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    target = calc_target_ret_mu(x, w, r)
    val = x .- target
    return -sum(val[val .<= zero(eltype(val))] .^ 3) / length(x)
end
function (r::HighOrderMoment{<:Any, <:FourthLowerMoment, <:Any, <:Any, <:Any})(w::AbstractVector,
                                                                               X::AbstractMatrix,
                                                                               fees::Union{Nothing,
                                                                                           <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    target = calc_target_ret_mu(x, w, r)
    val = x .- target
    return sum(val[val .<= zero(eltype(val))] .^ 4) / length(x)
end
function (r::HighOrderMoment{<:Any, <:FourthCentralMoment, <:Any, <:Any, <:Any})(w::AbstractVector,
                                                                                 X::AbstractMatrix,
                                                                                 fees::Union{Nothing,
                                                                                             <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    target = calc_target_ret_mu(x, w, r)
    val = x .- target
    return sum(val .^ 4) / length(x)
end
function (r::HighOrderMoment{<:Any, <:HighOrderDeviation{<:ThirdLowerMoment, <:Any}, <:Any,
                             <:Any, <:Any})(w::AbstractVector, X::AbstractMatrix,
                                            fees::Union{Nothing, <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    target = calc_target_ret_mu(x, w, r)
    val = x .- target
    val = val[val <= zero(target)]
    sigma = std(r.alg.ve, val; mean = zero(target))
    return -sum(val[val <= zero(target)] .^ 3) / length(x) / sigma^3
end
function (r::HighOrderMoment{<:Any, <:HighOrderDeviation{<:FourthLowerMoment, <:Any}, <:Any,
                             <:Any, <:Any})(w::AbstractVector, X::AbstractMatrix,
                                            fees::Union{Nothing, <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    target = calc_target_ret_mu(x, w, r)
    val = x .- target
    val = val[val <= zero(target)]
    sigma = std(r.alg.ve, val; mean = zero(target))
    return sum(val[val <= zero(target)] .^ 4) / length(x) / sigma^4
end
function (r::HighOrderMoment{<:Any, <:HighOrderDeviation{<:FourthCentralMoment, <:Any},
                             <:Any, <:Any, <:Any})(w::AbstractVector, X::AbstractMatrix,
                                                   fees::Union{Nothing, <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    target = calc_target_ret_mu(x, w, r)
    val = x .- target
    sigma = std(r.alg.ve, x)
    return sum(val .^ 4) / length(x) / sigma^4
end
for rt ∈ (LowOrderMoment, HighOrderMoment)
    eval(quote
             function risk_measure_factory(r::$(rt), prior::AbstractPriorResult, args...;
                                           kwargs...)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
                 return $(rt)(; settings = r.settings, alg = r.alg, target = r.target,
                              w = r.w, mu = mu)
             end
             function risk_measure_factory(r::$(rt), prior::EntropyPoolingResult, args...;
                                           kwargs...)
                 w = risk_measure_nothing_vec_factory(r.w, prior.w)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
                 alg = risk_moment_algorithm_factory(r.alg, prior.w)
                 return $(rt)(; settings = r.settings, alg = alg, target = r.target, w = w,
                              mu = mu)
             end
             function risk_measure_factory(r::$(rt),
                                           prior::HighOrderPriorResult{<:EntropyPoolingResult,
                                                                       <:Any, <:Any, <:Any,
                                                                       <:Any}, args...;
                                           kwargs...)
                 w = risk_measure_nothing_vec_factory(r.w, prior.pm.w)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
                 alg = risk_moment_algorithm_factory(r.alg, prior.pm.w)
                 return $(rt)(; settings = r.settings, alg = alg, target = r.target, w = w,
                              mu = mu)
             end
             function risk_measure_view(r::$(rt), prior::AbstractPriorResult,
                                        i::AbstractVector, args...; kwargs...)
                 target = risk_measure_nothing_real_vec_factory(r.target, i)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, i)
                 return $(rt)(; settings = r.settings, alg = r.alg, target = target,
                              w = r.w, mu = mu)
             end
             function risk_measure_view(r::$(rt), prior::EntropyPoolingResult,
                                        i::AbstractVector, args...; kwargs...)
                 target = risk_measure_nothing_real_vec_factory(r.target, i)
                 w = risk_measure_nothing_vec_factory(r.w, prior.w)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, i)
                 alg = risk_moment_algorithm_factory(r.alg, prior.w)
                 return $(rt)(; settings = r.settings, alg = alg, target = target, w = w,
                              mu = mu)
             end
             function risk_measure_view(r::$(rt),
                                        prior::HighOrderPriorResult{<:EntropyPoolingResult,
                                                                    <:Any, <:Any, <:Any,
                                                                    <:Any},
                                        i::AbstractVector, args...; kwargs...)
                 target = risk_measure_nothing_real_vec_factory(r.target, i)
                 w = risk_measure_nothing_vec_factory(r.w, prior.pm.w)
                 mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, i)
                 alg = risk_moment_algorithm_factory(r.alg, prior.pm.w)
                 return $(rt)(; settings = r.settings, alg = alg, target = target, w = w,
                              mu = mu)
             end
         end)
end

export FirstLowerMoment, SemiDeviation, SemiVariance, MeanAbsoluteDeviation,
       ThirdLowerMoment, FourthLowerMoment, FourthCentralMoment, HighOrderDeviation,
       LowOrderMoment, HighOrderMoment
