abstract type SecondMomentAlgorithm <: AbstractAlgorithm end
abstract type VarianceAlgorithm <: SecondMomentAlgorithm end
struct QuadRiskExpr <: VarianceAlgorithm end
struct SOCRiskExpr <: VarianceAlgorithm end
struct RSOCRiskExpr <: SecondMomentAlgorithm end
struct SqrtRiskExpr <: SecondMomentAlgorithm end
const QuadSqrtRiskExpr = Union{<:SqrtRiskExpr, <:QuadRiskExpr}
struct Variance{T1 <: RiskMeasureSettings, T2 <: Union{Nothing, <:AbstractMatrix},
                T3 <:
                Union{Nothing, <:AbstractString, Expr, <:AbstractVector{<:AbstractString},
                      <:AbstractVector{Expr},
                      <:AbstractVector{<:Union{<:AbstractString, Expr}},
                      #! Start: to delete
                      <:LinearConstraint, <:AbstractVector{<:LinearConstraint},
                      #! End: to delete
                      <:LinearConstraintResult}, T4 <: VarianceAlgorithm} <:
       JuMPRiskContributionSigmaRiskMeasure
    settings::T1
    sigma::T2
    rc::T3
    alg::T4
end
function Variance(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                  sigma::Union{Nothing, <:AbstractMatrix} = nothing,
                  rc::Union{Nothing, <:AbstractString, Expr,
                            <:AbstractVector{<:AbstractString}, <:AbstractVector{Expr},
                            <:AbstractVector{<:Union{<:AbstractString, Expr}},
                            #! Start: to delete
                            <:LinearConstraint, <:AbstractVector{<:LinearConstraint},
                            #! End: to delete
                            <:LinearConstraintResult} = nothing,
                  alg::VarianceAlgorithm = SOCRiskExpr())
    if isa(sigma, AbstractMatrix)
        @smart_assert(!isempty(sigma))
        assert_matrix_issquare(sigma)
    end
    return Variance{typeof(settings), typeof(sigma), typeof(rc), typeof(alg)}(settings,
                                                                              sigma, rc,
                                                                              alg)
end
function (r::Variance)(w::AbstractVector)
    return dot(w, r.sigma, w)
end
function factory(r::Variance, prior::AbstractPriorResult, args...; kwargs...)
    sigma = nothing_scalar_array_factory(r.sigma, prior.sigma)
    return Variance(; settings = r.settings, sigma = sigma, rc = r.rc, alg = r.alg)
end
function risk_measure_view(r::Variance, i::AbstractVector, args...)
    sigma = nothing_scalar_array_view(r.sigma, i)
    @smart_assert(!isa(r.rc, LinearConstraintResult),
                  "`rc` cannot be a `LinearConstraintResult` because there is no way to only consider items from a specific cluster.")
    # rc = linear_constraint_view(r.rc, i)
    return Variance(; settings = r.settings, sigma = sigma, rc = r.rc, alg = r.alg)
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
        assert_matrix_issquare(sigma)
    end
    return StandardDeviation{typeof(settings), typeof(sigma)}(settings, sigma)
end
function (r::StandardDeviation)(w::AbstractVector)
    return sqrt(dot(w, r.sigma, w))
end
function factory(r::StandardDeviation, prior::AbstractPriorResult, args...; kwargs...)
    sigma = nothing_scalar_array_factory(r.sigma, prior.sigma)
    return StandardDeviation(; settings = r.settings, sigma = sigma)
end
function risk_measure_view(r::StandardDeviation, i::AbstractVector, args...)
    sigma = nothing_scalar_array_view(r.sigma, i)
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
                                           <:AbstractUncertaintySetEstimator} = NormalUncertaintySetEstimator(;),
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
function no_bounds_risk_measure(r::UncertaintySetVariance, flag::Bool = true)
    return if flag
        UncertaintySetVariance(RiskMeasureSettings(; rke = r.settings.rke,
                                                   scale = r.settings.scale), r.ucs,
                               r.sigma)
    else
        Variance(;
                 settings = RiskMeasureSettings(; rke = r.settings.rke,
                                                scale = r.settings.scale), nothing,
                 sigma = r.sigma)
    end
end
function factory(r::UncertaintySetVariance, prior::AbstractPriorResult, ::Any,
                 ucs::Union{Nothing, <:AbstractUncertaintySetResult,
                            <:AbstractUncertaintySetEstimator} = nothing, args...;
                 kwargs...)
    ucs = ucs_factory(r.ucs, ucs)
    sigma = nothing_scalar_array_factory(r.sigma, prior.sigma)
    return UncertaintySetVariance(; settings = r.settings, ucs = ucs, sigma = sigma)
end
function risk_measure_view(r::UncertaintySetVariance, i::AbstractVector, args...)
    ucs = ucs_view(r.ucs, i)
    sigma = nothing_scalar_array_view(r.sigma, i)
    return UncertaintySetVariance(; settings = r.settings, ucs = ucs, sigma = sigma)
end

export QuadRiskExpr, SOCRiskExpr, RSOCRiskExpr, Variance, StandardDeviation,
       UncertaintySetVariance
