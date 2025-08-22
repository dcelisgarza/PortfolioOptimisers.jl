abstract type SecondMomentAlgorithm <: AbstractAlgorithm end
abstract type VarianceAlgorithm <: SecondMomentAlgorithm end
struct QuadRiskExpr <: VarianceAlgorithm end
struct SOCRiskExpr <: VarianceAlgorithm end
struct RSOCRiskExpr <: SecondMomentAlgorithm end
struct SqrtRiskExpr <: SecondMomentAlgorithm end
const QuadSqrtRiskExpr = Union{<:SqrtRiskExpr, <:QuadRiskExpr}
struct Variance{T1, T2, T3, T4} <: JuMPRiskContributionSigmaRiskMeasure
    settings::T1
    sigma::T2
    rc::T3
    alg::T4
end
function Variance(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                  sigma::Union{Nothing, <:AbstractMatrix} = nothing,
                  rc::Union{Nothing, <:LinearConstraintEstimator, <:LinearConstraint} = nothing,
                  alg::VarianceAlgorithm = SOCRiskExpr())
    if isa(sigma, AbstractMatrix)
        @argcheck(!isempty(sigma))
        assert_matrix_issquare(sigma)
    end
    return Variance(settings, sigma, rc, alg)
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
    @argcheck(!isa(r.rc, LinearConstraint),
              "`rc` cannot be a `LinearConstraint` because there is no way to only consider items from a specific group and because this would break factor risk contribution")
    return Variance(; settings = r.settings, sigma = sigma, rc = r.rc, alg = r.alg)
end
struct StandardDeviation{T1, T2} <: SigmaRiskMeasure
    settings::T1
    sigma::T2
end
function StandardDeviation(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                           sigma::Union{Nothing, <:AbstractMatrix} = nothing)
    if isa(sigma, AbstractMatrix)
        @argcheck(!isempty(sigma))
        assert_matrix_issquare(sigma)
    end
    return StandardDeviation(settings, sigma)
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
struct UncertaintySetVariance{T1, T2, T3} <: SigmaRiskMeasure
    settings::T1
    ucs::T2
    sigma::T3
end
function UncertaintySetVariance(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                ucs::Union{Nothing, <:AbstractUncertaintySetResult,
                                           <:AbstractUncertaintySetEstimator} = NormalUncertaintySet(;),
                                sigma::Union{Nothing, <:AbstractMatrix{<:Real}} = nothing)
    if isa(sigma, AbstractMatrix)
        @argcheck(!isempty(sigma))
    end
    return UncertaintySetVariance(settings, ucs, sigma)
end
function (r::UncertaintySetVariance)(w::AbstractVector)
    return dot(w, r.sigma, w)
end
function no_bounds_risk_measure(r::UncertaintySetVariance, flag::Bool = true)
    return if flag
        UncertaintySetVariance(;
                               settings = RiskMeasureSettings(; rke = r.settings.rke,
                                                              scale = r.settings.scale),
                               r.ucs, sigma = r.sigma)
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
