"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all regime-adjustment target structures used in
[`RegimeAdjustedExpWeightedCovariance`](@ref).

A target defines how the regime-adjusted covariance update is structured (e.g., which
baseline covariance form is shrunk toward).

# Interfaces

In order to implement a new regime-adjustment target, subtype `RegimeAdjustedTarget`
and optionally implement [`min_active_assets`](@ref).

## `min_active_assets` interface

  - `min_active_assets(target::RegimeAdjustedTarget) -> Int`: Returns the minimum number
    of active assets required to use this target. Defaults to `1`.

### Arguments

  - `target`: The concrete target instance.

### Returns

  - `n::Int`: Minimum required active assets.

### Examples

```jldoctest
julia> struct MyTarget <: PortfolioOptimisers.RegimeAdjustedTarget end

julia> PortfolioOptimisers.min_active_assets(MyTarget())
1
```

## Related

  - [`MahalanobisTarget`](@ref)
  - [`DiagonalTarget`](@ref)
  - [`PortfolioTarget`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
"""
abstract type RegimeAdjustedTarget <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the minimum number of active assets required for this regime-adjustment target.

# Arguments

  - `::RegimeAdjustedTarget`: Regime-adjustment target (unused by this default method).

# Returns

  - `1::Int`: The default minimum is one active asset.

## Related

  - [`RegimeAdjustedTarget`](@ref)
  - [`MahalanobisTarget`](@ref)
"""
function min_active_assets(::RegimeAdjustedTarget)
    return 1
end
"""
$(DocStringExtensions.TYPEDEF)

Regime-adjustment target that uses a Mahalanobis-distance-based baseline covariance
structure. Requires at least two active assets.

## Related

  - [`RegimeAdjustedTarget`](@ref)
  - [`DiagonalTarget`](@ref)
  - [`PortfolioTarget`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
"""
struct MahalanobisTarget <: RegimeAdjustedTarget end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the minimum number of active assets required for the Mahalanobis target.

# Arguments

  - `::MahalanobisTarget`: Mahalanobis regime-adjustment target (unused).

# Returns

  - `2::Int`: At least two active assets are required.

## Related

  - [`MahalanobisTarget`](@ref)
  - [`RegimeAdjustedTarget`](@ref)
"""
function min_active_assets(::MahalanobisTarget)
    return 2
end
"""
$(DocStringExtensions.TYPEDEF)

Regime-adjustment target that uses a diagonal baseline covariance structure.

## Related

  - [`RegimeAdjustedTarget`](@ref)
  - [`MahalanobisTarget`](@ref)
  - [`PortfolioTarget`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
"""
struct DiagonalTarget <: RegimeAdjustedTarget end
"""
$(DocStringExtensions.TYPEDEF)

Regime-adjustment target that uses a portfolio-weighted baseline covariance structure.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PortfolioTarget(;
        w::Option{<:EstValType} = nothing
    ) -> PortfolioTarget

Keywords correspond to the struct's fields.

## Validation

  - If `w` is an `AbstractVector`, `!isempty(w)`.

# Examples

```jldoctest
julia> PortfolioTarget()
PortfolioTarget
  w ┴ nothing
```

## Related

  - [`RegimeAdjustedTarget`](@ref)
  - [`MahalanobisTarget`](@ref)
  - [`DiagonalTarget`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
"""
@concrete struct PortfolioTarget <: RegimeAdjustedTarget
    "$(field_dict[:ra_w])"
    w
    function PortfolioTarget(w::Option{<:EstValType})
        if isa(w, AbstractVector)
            @argcheck(!isempty(w))
        end
        return new{typeof(w)}(w)
    end
end
function PortfolioTarget(; w::Option{<:EstValType} = nothing)::PortfolioTarget
    return PortfolioTarget(w)
end
"""
$(DocStringExtensions.TYPEDEF)

Online exponentially weighted covariance estimator with regime-state adjustment.

Maintains separate exponentially weighted running means for the covariance and the
correlation, combining them at each step. After processing all observations, the result is
scaled by the squared regime multiplier derived from the smoothed regime state.

# Mathematical definition

EWM covariance update (decay ``\\lambda``) and correlation (decay ``\\lambda_c``):

```math
C_{ij,t} = \\lambda C_{ij,t-1} + (1-\\lambda)(r_{i,t}-\\bar{r}_i)(r_{j,t}-\\bar{r}_j)
```

```math
\\rho_{ij,t} = \\lambda_c \\rho_{ij,t-1} + (1-\\lambda_c)\\frac{(r_{i,t}-\\bar{r}_i)(r_{j,t}-\\bar{r}_j)}{\\sqrt{v_{i,t} v_{j,t}}}
```

The regime state ``s_t`` is smoothed from ``z_t^2`` using `regime_decay`. Final covariance:

```math
\\hat{\\Sigma}_{ij} = \\mathrm{mult}(s_T)^2 \\cdot \\rho_{ij,T}\\sqrt{v_{i,T} v_{j,T}}
```

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RegimeAdjustedExpWeightedCovariance(;
        decay::Number                        = exp2(-inv(40.0)),
        cor_decay::Number                    = exp2(-inv(20.0)),
        hac_lags::Option{<:Integer}          = nothing,
        regime_method::RegimeAdjustedMethod  = FirstMomentRegimeAdjusted(),
        regime_decay::Number                 = exp2(-2 / inv(log2(inv(decay)))),
        regime_target::RegimeAdjustedTarget  = DiagonalTarget()
    ) -> RegimeAdjustedExpWeightedCovariance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:decay])
  - $(val_dict[:hac_lags])

# Examples

```jldoctest
julia> ce = RegimeAdjustedExpWeightedCovariance();

julia> ce.decay ≈ exp2(-inv(40.0))
true
```

## Related

  - [`RegimeAdjustedTarget`](@ref)
  - [`RegimeAdjustedMethod`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
"""
@concrete struct RegimeAdjustedExpWeightedCovariance <: AbstractCovarianceEstimator
    "$(field_dict[:decay])"
    decay
    "$(field_dict[:cor_decay])"
    cor_decay
    "$(field_dict[:hac_lags])"
    hac_lags
    "$(field_dict[:regime_method])"
    regime_method
    "$(field_dict[:regime_decay])"
    regime_decay
    "$(field_dict[:regime_target])"
    regime_target
    function RegimeAdjustedExpWeightedCovariance(decay::Number, cor_decay::Number,
                                                 hac_lags::Option{<:Integer},
                                                 regime_method::RegimeAdjustedMethod,
                                                 regime_decay::Number,
                                                 regime_target::RegimeAdjustedTarget)
        assert_nonempty_gt0_finite_val(decay, :decay)
        assert_nonempty_gt0_finite_val(cor_decay, :cor_decay)
        if !isnothing(hac_lags)
            assert_nonempty_gt0_finite_val(hac_lags, :hac_lags)
        end
        return new{typeof(decay), typeof(cor_decay), typeof(hac_lags),
                   typeof(regime_method), typeof(regime_decay), typeof(regime_target)}(decay,
                                                                                       cor_decay,
                                                                                       hac_lags,
                                                                                       regime_method,
                                                                                       regime_decay,
                                                                                       regime_target)
    end
end
function RegimeAdjustedExpWeightedCovariance(; decay::Number = exp2(-inv(40.0)),
                                             cor_decay::Number = exp2(-inv(20.0)),
                                             hac_lags::Option{<:Integer} = nothing,
                                             regime_method::RegimeAdjustedMethod = FirstMomentRegimeAdjusted(),
                                             regime_decay::Number = exp2(-2 /
                                                                         inv(log2(inv(decay)))),
                                             regime_target::RegimeAdjustedTarget = DiagonalTarget())::RegimeAdjustedExpWeightedCovariance
    return RegimeAdjustedExpWeightedCovariance(decay, cor_decay, hac_lags, regime_method,
                                               regime_decay, regime_target)
end

# export RegimeAdjustedTarget, MahalanobisTarget, DiagonalTarget, PortfolioTarget,
#       RegimeAdjustedExpWeightedCovariance
