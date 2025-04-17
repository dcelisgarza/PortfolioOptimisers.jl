abstract type VarianceFormulation end
struct Quad <: VarianceFormulation end
struct SOC <: VarianceFormulation end
struct RSOC <: VarianceFormulation end
struct Variance{T1 <: RiskMeasureSettings, T2 <: VarianceFormulation,
                T3 <: Union{Nothing, <:AbstractMatrix},
                T4 <:
                Union{Nothing, <:LinearConstraint, <:AbstractVector{<:LinearConstraint},
                      <:LinearConstraintResult}} <: JuMPRiskContributionSigmaRiskMeasure
    settings::T1
    formulation::T2
    sigma::T3
    rc::T4
end
function Variance(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                  formulation::VarianceFormulation = SOC(),
                  sigma::Union{Nothing, <:AbstractMatrix} = nothing,
                  rc::Union{Nothing, <:LinearConstraint,
                            <:AbstractVector{<:LinearConstraint}, <:LinearConstraintResult} = nothing)
    if isa(sigma, AbstractMatrix)
        @smart_assert(!isempty(sigma))
        issquare(sigma)
    end
    return Variance{typeof(settings), typeof(formulation), typeof(sigma), typeof(rc)}(settings,
                                                                                      formulation,
                                                                                      sigma,
                                                                                      rc)
end
function (r::Variance)(w::AbstractVector)
    return dot(w, r.sigma, w)
end
function risk_measure_factory(r::Variance, prior::AbstractPriorResult, args...; kwargs...)
    sigma = risk_measure_nothing_real_array_factory(r.sigma, prior.sigma)
    return Variance(; settings = r.settings, formulation = r.formulation, sigma = sigma,
                    rc = r.rc)
end
function risk_measure_view(r::Variance, prior::AbstractPriorResult, i::AbstractVector,
                           args...; kwargs...)
    sigma = risk_measure_nothing_real_array_factory(r.sigma, prior.sigma, i)
    if isa(r.rc, LinearConstraintResult)
        throw(ArgumentError("`rc` cannot be a `LinearConstraintResult` because there is no way to only consider items from a specific cluster."))
    end
    return Variance(; settings = r.settings, formulation = r.formulation, sigma = sigma,
                    rc = r.rc)
end
struct StandardDeviation{T1 <: RiskMeasureSettings,
                         T2 <: Union{Nothing, <:AbstractMatrix}} <: SigmaRiskMeasure
    settings::T1
    sigma::T2
end
function StandardDeviation(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                           sigma::Union{Nothing, <:AbstractMatrix} = nothing)
    if isa(sigma, AbstractMatrix)
        @smart_assert(!isempty(sigma))
        issquare(sigma)
    end
    return StandardDeviation{typeof(settings), typeof(sigma)}(settings, sigma)
end
function (r::StandardDeviation)(w::AbstractVector)
    return sqrt(dot(w, r.sigma, w))
end
function risk_measure_factory(r::StandardDeviation, prior::AbstractPriorResult, args...;
                              kwargs...)
    sigma = risk_measure_nothing_real_array_factory(r.sigma, prior.sigma)
    return StandardDeviation(; settings = r.settings, sigma = sigma)
end
function risk_measure_view(r::StandardDeviation, prior::AbstractPriorResult,
                           i::AbstractVector, args...; kwargs...)
    sigma = risk_measure_nothing_real_array_factory(r.sigma, prior.sigma, i)
    return StandardDeviation(; settings = r.settings, sigma = sigma)
end
struct UncertaintySetVariance{T1 <: RiskMeasureSettings,
                              T2 <: Union{Nothing, <:AbstractUncertaintySetResult,
                                          <:AbstractUncertaintySetEstimator},
                              T3 <: Union{Nothing, <:AbstractMatrix{<:Real}}} <:
       SigmaRiskMeasure
    settings::T1
    ucs::T2
    sigma::T3
end
function UncertaintySetVariance(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                ucs::Union{Nothing, <:AbstractUncertaintySetResult,
                                           <:AbstractUncertaintySetEstimator} = nothing,
                                sigma::Union{Nothing, <:AbstractMatrix{<:Real}} = nothing)
    if isa(sigma, AbstractMatrix)
        @smart_assert(!isempty(sigma))
    end
    return UncertaintySetVariance{typeof(settings), typeof(ucs), typeof(sigma)}(settings,
                                                                                ucs, sigma)
end
function (r::UncertaintySetVariance)(w::AbstractVector)
    return dot(w, r.sigma, w)
end
function risk_measure_factory(r::UncertaintySetVariance, prior::AbstractPriorResult, ::Any,
                              ucs::Union{Nothing, <:AbstractUncertaintySetResult,
                                         <:AbstractUncertaintySetEstimator} = nothing,
                              args...; kwargs...)
    uset = ucs_factory(r.ucs, ucs)
    sigma = risk_measure_nothing_real_array_factory(r.sigma, prior.sigma)
    return UncertaintySetVariance(; settings = r.settings, ucs = uset, sigma = sigma)
end
function risk_measure_view(r::UncertaintySetVariance, prior::AbstractPriorResult,
                           i::AbstractVector, ::Any,
                           ucs::Union{Nothing, <:AbstractUncertaintySetResult,
                                      <:AbstractUncertaintySetEstimator} = nothing, args...;
                           kwargs...)
    uset = ucs_view(r.ucs, ucs, i)
    sigma = risk_measure_nothing_real_array_factory(r.sigma, prior.sigma, i)
    return UncertaintySetVariance(; settings = r.settings, ucs = uset, sigma = sigma)
end

export Quad, SOC, RSOC, Variance, StandardDeviation, UncertaintySetVariance
