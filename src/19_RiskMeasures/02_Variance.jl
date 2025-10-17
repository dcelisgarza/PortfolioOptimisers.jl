"""
    abstract type SecondMomentAlgorithm <: AbstractAlgorithm end

Abstract supertype for optimisation formulations of second moment risk measures in PortfolioOptimisers.jl.

# Related Types

  - [`VarianceAlgorithm`](@ref)
  - [`QuadRiskExpr`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
  - [`RSOCRiskExpr`](@ref)
  - [`SOCRiskExpr`](@ref)
"""
abstract type SecondMomentAlgorithm <: AbstractAlgorithm end
"""
    abstract type VarianceAlgorithm <: SecondMomentAlgorithm end

Abstract supertype for optimisation formulations of variance-based risk measures in PortfolioOptimisers.jl.

# Related Types

  - [`QuadRiskExpr`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
"""
abstract type VarianceAlgorithm <: SecondMomentAlgorithm end
"""
Direct quadratic risk expression optimisation formulation for variance-based risk measures. Implements the variance directly.
"""
struct QuadRiskExpr <: VarianceAlgorithm end
"""
Squared second-order cone risk expression optimisation formulation for variance-based risk measures. Implements the variance as the square of the standard deviation.
"""
struct SquaredSOCRiskExpr <: VarianceAlgorithm end
"""
Rotated second-order cone risk expression optimisation formulation for variance-based risk measures. Used as a sum of squares formulation using historical data.
"""
struct RSOCRiskExpr <: SecondMomentAlgorithm end
"""
Second-order cone risk expression optimisation formulation for variance-based risk measures. Implements the standard deviation (squar root of the variance)
"""
struct SOCRiskExpr <: SecondMomentAlgorithm end
const UnionSOCRiskExpr = Union{<:SOCRiskExpr, <:SquaredSOCRiskExpr}
struct Variance{T1, T2, T3, T4} <: RiskMeasure
    settings::T1
    sigma::T2
    rc::T3
    alg::T4
    function Variance(settings::RiskMeasureSettings,
                      sigma::Union{Nothing, <:AbstractMatrix},
                      rc::Union{Nothing, <:LinearConstraintEstimator, <:LinearConstraint},
                      alg::VarianceAlgorithm)
        if isa(sigma, AbstractMatrix)
            @argcheck(!isempty(sigma))
            assert_matrix_issquare(sigma)
        end
        return new{typeof(settings), typeof(sigma), typeof(rc), typeof(alg)}(settings,
                                                                             sigma, rc, alg)
    end
end
function Variance(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                  sigma::Union{Nothing, <:AbstractMatrix} = nothing,
                  rc::Union{Nothing, <:LinearConstraintEstimator, <:LinearConstraint} = nothing,
                  alg::VarianceAlgorithm = SquaredSOCRiskExpr())
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
struct StandardDeviation{T1, T2} <: RiskMeasure
    settings::T1
    sigma::T2
    function StandardDeviation(settings::RiskMeasureSettings,
                               sigma::Union{Nothing, <:AbstractMatrix})
        if isa(sigma, AbstractMatrix)
            @argcheck(!isempty(sigma))
            assert_matrix_issquare(sigma)
        end
        return new{typeof(settings), typeof(sigma)}(settings, sigma)
    end
end
function StandardDeviation(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                           sigma::Union{Nothing, <:AbstractMatrix} = nothing)
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
struct UncertaintySetVariance{T1, T2, T3} <: RiskMeasure
    settings::T1
    ucs::T2
    sigma::T3
    function UncertaintySetVariance(settings::RiskMeasureSettings,
                                    ucs::Union{Nothing, <:AbstractUncertaintySetResult,
                                               <:AbstractUncertaintySetEstimator},
                                    sigma::Union{Nothing, <:AbstractMatrix{<:Real}})
        if isa(sigma, AbstractMatrix)
            @argcheck(!isempty(sigma))
        end
        return new{typeof(settings), typeof(ucs), typeof(sigma)}(settings, ucs, sigma)
    end
end
function UncertaintySetVariance(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                ucs::Union{Nothing, <:AbstractUncertaintySetResult,
                                           <:AbstractUncertaintySetEstimator} = NormalUncertaintySet(),
                                sigma::Union{Nothing, <:AbstractMatrix{<:Real}} = nothing)
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

export SOCRiskExpr, QuadRiskExpr, SquaredSOCRiskExpr, RSOCRiskExpr, Variance,
       StandardDeviation, UncertaintySetVariance
