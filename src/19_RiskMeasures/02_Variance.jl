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
    struct QuadRiskExpr <: VarianceAlgorithm end

Direct quadratic risk expression optimisation formulation for variance-like risk measures. The risk measure is implemented using an explicitly quadratic form `w' * Σ * w`.

# Related Types

  - [`VarianceAlgorithm`](@ref)
  - [`Variance`](@ref)
  - [`SOCRiskExpr`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
"""
struct QuadRiskExpr <: VarianceAlgorithm end
"""
    struct SquaredSOCRiskExpr <: VarianceAlgorithm end

Squared second-order cone risk expression optimisation formulation for applicable risk measures. The risk measure is implemented using the square of a variable constrained by a second order cone.

# Related

  - [`VarianceAlgorithm`](@ref)
  - [`QuadRiskExpr`](@ref)
  - [`SOCRiskExpr`](@ref)
  - [`Variance`](@ref)
"""
struct SquaredSOCRiskExpr <: VarianceAlgorithm end
"""
    struct RSOCRiskExpr <: SecondMomentAlgorithm end

Rotated second-order cone risk expression optimisation formulation for applicable risk measures. The risk measure using a variable constrained to be in a rotated second order cone representing the sum of squares.

# Related Types

  - [`SecondMomentAlgorithm`](@ref)
  - [`VarianceAlgorithm`](@ref)
  - [`SOCRiskExpr`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
"""
struct RSOCRiskExpr <: SecondMomentAlgorithm end
"""
    struct SOCRiskExpr <: SecondMomentAlgorithm end

Second-order cone risk expression optimisation formulation for applicable risk measures. The risk measure is implemented using a variable constrained by a second order cone.

# Related

  - [`SecondMomentAlgorithm`](@ref)
  - [`VarianceAlgorithm`](@ref)
  - [`QuadRiskExpr`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
  - [`RSOCRiskExpr`](@ref)
"""
struct SOCRiskExpr <: SecondMomentAlgorithm end
"""
    struct Variance{T1, T2, T3, T4} <: RiskMeasure
        settings::T1
        sigma::T2
        rc::T3
        alg::T4
    end

Risk measure estimator for variance-based risk calculations.

Variance quantifies portfolio second-moment risk using a (possibly user-provided) covariance matrix and an optimisation formulation specified by `alg`. The `sigma` field allows overriding the covariance produced by a prior; `rc` carries optional risk contribution constraints; `settings` configures scaling and bounds for the risk measure.

# Fields

  - `settings`: Risk measure configuration.
  - `sigma`: Optional covariance matrix that overrides the prior covariance when provided. Also used to compute the risk represented by a vector of weights.
  - `rc`: Optional specification of risk contribution constraints.
  - `alg`: The optimisation formulation used to represent the variance risk expression.

# Constructors

    Variance(; settings::RiskMeasureSettings = RiskMeasureSettings(),
             sigma::Union{Nothing, <:AbstractMatrix} = nothing,
             rc::Union{Nothing, <:LinearConstraintEstimator, <:LinearConstraint} = nothing,
             alg::VarianceAlgorithm = SquaredSOCRiskExpr())

Keyword arguments correspond to the fields above.

## Validation

  - If `sigma` is provided, `!isempty(sigma)` and `size(sigma, 1) == size(sigma, 2)`.

# Examples

```jldoctest
julia> Variance()
Variance
  settings | RiskMeasureSettings
           |   scale | Float64: 1.0
           |      ub | nothing
           |     rke | Bool: true
     sigma | nothing
        rc | nothing
       alg | SquaredSOCRiskExpr()
```

# Related

  - [`RiskMeasureSettings`](@ref)
  - [`VarianceAlgorithm`](@ref)
  - [`QuadRiskExpr`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
  - [`SOCRiskExpr`](@ref)
  - [`RSOCRiskExpr`](@ref)
  - [`expected_risk`](@ref)
"""
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
