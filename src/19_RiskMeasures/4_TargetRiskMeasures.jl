abstract type AbstractRiskMomentAlgorithm <: AbstractAlgorithm end
abstract type AbstractLowOrderRiskMomentAlgorithm <: AbstractRiskMomentAlgorithm end
struct Moment <: AbstractLowOrderRiskMomentAlgorithm end
struct Deviation{T1 <: Integer} <: AbstractLowOrderRiskMomentAlgorithm
    ddof::T1
end
function Deviation(; ddof::Integer = 1)
    @smart_assert(ddof >= 0)
    return Deviation{typeof(ddof)}(ddof)
end
abstract type AbstractHighOrderMomentAlgorithm <: AbstractRiskMomentAlgorithm end
struct ThirdLower <: AbstractHighOrderMomentAlgorithm end
struct FourthLower <: AbstractHighOrderMomentAlgorithm end
struct FourthCentral <: AbstractHighOrderMomentAlgorithm end
struct FirstLowerPartialMoment{T1 <: RiskMeasureSettings,
                               T2 <: AbstractLowOrderRiskMomentAlgorithm,
                               T3 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                               T4 <: Union{Nothing, <:AbstractWeights},
                               T5 <: Union{Nothing, <:AbstractVector{<:Real}}} <:
       TargetRiskMeasure
    settings::T1
    alg::T2
    target::T3
    w::T4
    mu::T5
end
function FirstLowerPartialMoment(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                 alg::AbstractLowOrderRiskMomentAlgorithm = Moment(),
                                 target::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = 0.0,
                                 w::Union{Nothing, <:AbstractWeights} = nothing,
                                 mu::Union{Nothing, <:AbstractVector{<:Real}} = nothing)
    if isa(target, AbstractVector)
        @smart_assert(!isempty(target))
    end
    if isa(mu, AbstractVector)
        @smart_assert(!isempty(mu))
    end
    return FirstLowerPartialMoment{typeof(settings), typeof(alg), typeof(target), typeof(w),
                                   typeof(mu)}(settings, alg, target, w, mu)
end
function (r::FirstLowerPartialMoment{<:Any, <:Moment, <:Any, <:Any, <:Any})(w::AbstractVector,
                                                                            X::AbstractMatrix,
                                                                            fees::Union{Nothing,
                                                                                        <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    target = calc_target_ret_mu(x, w, r)
    val = x .- target
    val = val[val .<= zero(eltype(val))]
    return -sum(val) / length(x)
end
function (r::FirstLowerPartialMoment{<:Any, <:Deviation, <:Any, <:Any, <:Any})(w::AbstractVector,
                                                                               X::AbstractMatrix,
                                                                               fees::Union{Nothing,
                                                                                           <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    mu = calc_target_ret_mu(x, w, r)
    val = x .- mu
    val = val[val .<= zero(eltype(val))]
    return sqrt(dot(val, val) / (length(x) - r.alg.ddof))
end
struct MeanAbsoluteDeviation{T1 <: RiskMeasureSettings,
                             T2 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                             T3 <: Union{Nothing, <:AbstractWeights},
                             T4 <: Union{Nothing, <:AbstractVector{<:Real}},
                             T5 <: Union{Nothing, <:AbstractWeights}} <: TargetRiskMeasure
    settings::T1
    target::T2
    w::T3
    mu::T4
    we::T5
end
function MeanAbsoluteDeviation(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                               target::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                               w::Union{Nothing, <:AbstractWeights} = nothing,
                               mu::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                               we::Union{Nothing, <:AbstractWeights} = nothing)
    if isa(target, AbstractVector)
        @smart_assert(!isempty(target))
    end
    if isa(mu, AbstractVector)
        @smart_assert(!isempty(mu))
    end
    return MeanAbsoluteDeviation{typeof(settings), typeof(target), typeof(w), typeof(mu),
                                 typeof(we)}(settings, target, w, mu, we)
end
function (r::MeanAbsoluteDeviation)(w::AbstractVector, X::AbstractMatrix,
                                    fees::Union{Nothing, <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    mu = calc_target_ret_mu(x, w, r)
    we = r.we
    return isnothing(we) ? mean(abs.(x .- mu)) : mean(abs.(x .- mu), we)
end
function risk_measure_factory(r::MeanAbsoluteDeviation, prior::AbstractPriorResult, args...;
                              kwargs...)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
    return MeanAbsoluteDeviation(; settings = r.settings, target = r.target, w = r.w,
                                 mu = mu, we = r.we)
end
function risk_measure_factory(r::MeanAbsoluteDeviation, prior::EntropyPoolingResult,
                              args...; kwargs...)
    w = risk_measure_nothing_vec_factory(r.w, prior.w)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
    we = risk_measure_nothing_vec_factory(r.we, prior.w)
    return MeanAbsoluteDeviation(; settings = r.settings, target = r.target, w = w, mu = mu,
                                 we = we)
end
function risk_measure_factory(r::MeanAbsoluteDeviation,
                              (prior::HighOrderPriorResult{<:EntropyPoolingResult, <:Any,
                                                           <:Any, <:Any, <:Any}), args...;
                              kwargs...)
    w = risk_measure_nothing_vec_factory(r.w, prior.pm.w)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
    we = risk_measure_nothing_vec_factory(r.we, prior.pm.w)
    return MeanAbsoluteDeviation(; settings = r.settings, target = r.target, w = w, mu = mu,
                                 we = we)
end
function risk_measure_view(r::MeanAbsoluteDeviation, prior::AbstractPriorResult,
                           cluster::AbstractVector, args...; kwargs...)
    target = risk_measure_nothing_real_vec_factory(r.target, cluster)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
    return MeanAbsoluteDeviation(; settings = r.settings, target = target, w = r.w, mu = mu,
                                 we = r.we)
end
function risk_measure_view(r::MeanAbsoluteDeviation, prior::EntropyPoolingResult,
                           cluster::AbstractVector, args...; kwargs...)
    target = risk_measure_nothing_real_vec_factory(r.target, cluster)
    w = risk_measure_nothing_vec_factory(r.w, prior.w)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
    we = risk_measure_nothing_vec_factory(r.we, prior.w)
    return MeanAbsoluteDeviation(; settings = r.settings, target = target, w = w, mu = mu,
                                 we = we)
end
function risk_measure_view(r::MeanAbsoluteDeviation,
                           prior::HighOrderPriorResult{<:EntropyPoolingResult, <:Any, <:Any,
                                                       <:Any, <:Any},
                           cluster::AbstractVector, args...; kwargs...)
    target = risk_measure_nothing_real_vec_factory(r.target, cluster)
    w = risk_measure_nothing_vec_factory(r.w, prior.pm.w)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
    we = risk_measure_nothing_vec_factory(r.we, prior.pm.w)
    return MeanAbsoluteDeviation(; settings = r.settings, target = target, w = w, mu = mu,
                                 we = we)
end
struct SemiVariance{T1 <: RiskMeasureSettings, T2 <: VarianceFormulation,
                    T3 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                    T4 <: Union{Nothing, <:AbstractWeights},
                    T5 <: Union{Nothing, <:AbstractVector{<:Real}}} <: TargetRiskMeasure
    settings::T1
    formulation::T2
    target::T3
    w::T4
    mu::T5
end
function SemiVariance(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                      formulation::VarianceFormulation = SOC(),
                      target::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                      w::Union{Nothing, <:AbstractWeights} = nothing,
                      mu::Union{Nothing, <:AbstractVector{<:Real}} = nothing)
    return SemiVariance(settings, formulation, target, w, mu)
end
function (r::SemiVariance)(w::AbstractVector, X::AbstractMatrix,
                           fees::Union{Nothing, <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    mu = calc_target_ret_mu(x, w, r)
    val = x .- mu
    val = val[val .<= zero(eltype(val))]
    return dot(val, val) / (length(x) - 1)
end
function risk_measure_factory(r::SemiVariance, prior::AbstractPriorResult, args...;
                              kwargs...)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
    return SemiVariance(; settings = r.settings, formulation = r.formulation,
                        target = r.target, w = r.w, mu = mu)
end
function risk_measure_factory(r::SemiVariance, prior::EntropyPoolingResult, args...;
                              kwargs...)
    w = risk_measure_nothing_vec_factory(r.w, prior.w)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
    return SemiVariance(; settings = r.settings, formulation = r.formulation,
                        target = r.target, w = w, mu = mu)
end
function risk_measure_factory(r::SemiVariance,
                              prior::HighOrderPriorResult{<:EntropyPoolingResult, <:Any,
                                                          <:Any, <:Any, <:Any}, args...;
                              kwargs...)
    w = risk_measure_nothing_vec_factory(r.w, prior.pm.w)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu)
    return SemiVariance(; settings = r.settings, formulation = r.formulation,
                        target = r.target, w = w, mu = mu)
end
function risk_measure_view(r::SemiVariance, prior::AbstractPriorResult,
                           cluster::AbstractVector, args...; kwargs...)
    target = risk_measure_nothing_real_vec_factory(r.target, cluster)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
    return SemiVariance(; settings = r.settings, formulation = r.formulation,
                        target = target, w = r.w, mu = mu)
end
function risk_measure_view(r::SemiVariance, prior::EntropyPoolingResult,
                           cluster::AbstractVector, args...; kwargs...)
    target = risk_measure_nothing_real_vec_factory(r.target, cluster)
    w = risk_measure_nothing_vec_factory(r.w, prior.w)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
    return SemiVariance(; settings = r.settings, formulation = r.formulation,
                        target = target, w = w, mu = mu)
end
function risk_measure_view(r::SemiVariance,
                           prior::HighOrderPriorResult{<:EntropyPoolingResult, <:Any, <:Any,
                                                       <:Any, <:Any},
                           cluster::AbstractVector, args...; kwargs...)
    target = risk_measure_nothing_real_vec_factory(r.target, cluster)
    w = risk_measure_nothing_vec_factory(r.w, prior.pm.w)
    mu = risk_measure_nothing_vec_factory(r.mu, prior.mu, cluster)
    return SemiVariance(; settings = r.settings, formulation = r.formulation,
                        target = target, w = w, mu = mu)
end
struct HighOrderMoment{T1 <: RiskMeasureSettings, T2 <: AbstractHighOrderMomentAlgorithm,
                       T3 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                       T4 <: Union{Nothing, <:AbstractWeights},
                       T5 <: Union{Nothing, <:AbstractVector{<:Real}}} <:
       TargetHierarchicalRiskMeasure
    settings::T1
    alg::T2
    target::T3
    w::T4
    mu::T5
end
function HighOrderMoment(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                         alg::AbstractHighOrderMomentAlgorithm = FourthLower(),
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
function (r::HighOrderMoment{<:Any, <:ThirdLower, <:Any, <:Any, <:Any})(w::AbstractVector,
                                                                        X::AbstractMatrix,
                                                                        fees::Union{Nothing,
                                                                                    <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    target = calc_target_ret_mu(x, w, r)
    val = x .- target
    return -sum(val[val .<= zero(eltype(val))] .^ 3) / length(x)
end
function (r::HighOrderMoment{<:Any, <:FourthLower, <:Any, <:Any, <:Any})(w::AbstractVector,
                                                                         X::AbstractMatrix,
                                                                         fees::Union{Nothing,
                                                                                     <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    target = calc_target_ret_mu(x, w, r)
    val = x .- target
    return sum(val[val .<= zero(eltype(val))] .^ 4) / length(x)
end
function (r::HighOrderMoment{<:Any, <:FourthCentral, <:Any, <:Any, <:Any})(w::AbstractVector,
                                                                           X::AbstractMatrix,
                                                                           fees::Union{Nothing,
                                                                                       <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    target = calc_target_ret_mu(x, w, r)
    val = x .- target
    return sum(val .^ 4) / length(x)
end

export Moment, Deviation, ThirdLower, FourthLower, FourthCentral, FirstLowerPartialMoment,
       MeanAbsoluteDeviation, SemiVariance, HighOrderMoment
