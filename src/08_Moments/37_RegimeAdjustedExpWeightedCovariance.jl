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

# Related

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

# Related

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

# Related

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

# Related

  - [`MahalanobisTarget`](@ref)
  - [`RegimeAdjustedTarget`](@ref)
"""
function min_active_assets(::MahalanobisTarget)
    return 2
end
"""
$(DocStringExtensions.TYPEDEF)

Regime-adjustment target that uses a diagonal baseline covariance structure.

# Related

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
        w::Option{<:Union{<:VecNum, <:MatNum}} = nothing
    ) -> PortfolioTarget

Keywords correspond to the struct's fields. The bound is the pair of shapes the fit can honour,
because [`Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref) reads a bare matrix and carries no asset names. The count of assets, the sign of each weight and the sum of each row are checked at the fit, where the universe is known.

## Validation

  - If `w` is not `nothing`, `!isempty(w)`.

# Examples

```jldoctest
julia> PortfolioTarget()
PortfolioTarget
  w ┴ nothing
```

# Related

  - [`RegimeAdjustedTarget`](@ref)
  - [`MahalanobisTarget`](@ref)
  - [`DiagonalTarget`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
"""
@concrete struct PortfolioTarget <: RegimeAdjustedTarget
    """
    $(field_dict[:ra_w])
    """
    w
    function PortfolioTarget(w::Option{<:Union{<:VecNum, <:MatNum}})
        if !isnothing(w)
            @argcheck(!isempty(w), IsEmptyError("w cannot be empty"))
        end
        return new{typeof(w)}(w)
    end
end
function PortfolioTarget(;
                         w::Option{<:Union{<:VecNum, <:MatNum}} = nothing)::PortfolioTarget
    return PortfolioTarget(w)
end
"""
$(DocStringExtensions.TYPEDEF)

Online exponentially weighted covariance estimator with regime-state adjustment.

At each observation it updates a running exponentially weighted covariance, and it forms a
one-step-ahead statistic that compares the realised risk of that observation against the risk
the state predicted before it. The statistic is smoothed by `regime_decay` into a scalar regime
state, and the covariance is scaled by the square of the multiplier that state names.

The estimator is the covariance twin of [`RegimeAdjustedExpWeightedVariance`](@ref). It shares
that type's [`RegimeAdjustedMethod`](@ref) family, and it adds a [`RegimeAdjustedTarget`](@ref),
which states what the statistic measures: one portfolio direction, the marginal volatilities
alone, or the whole covariance structure.

A `regime_method` of `nothing` turns the adjustment off: no regime state advances, so the
multiplier stays at one and the estimator is the plain exponentially weighted recursion.

# Mathematical definition

Write ``\\lambda`` for `decay` and ``\\lambda_c`` for `cor_decay`. Where `cor_decay` is
`nothing`, one recursion carries the whole matrix:

```math
\\begin{align}
S_{t} &= \\lambda S_{t-1} + (1-\\lambda) \\boldsymbol{u}_{t} \\boldsymbol{u}_{t}^{\\intercal}\\,.
\\end{align}
```

Where:

  - ``S_{t}``: Raw exponentially weighted covariance state at time ``t``, seeded at zero.
  - ``\\boldsymbol{u}_{t}``: Observation ``t``, centred where `centred` is `false`, and
    HAC-adjusted where `hac_lags` is not `nothing`.

Where `cor_decay` is not `nothing`, the variance and the correlation run at their own decays and
are recombined:

```math
\\begin{align}
v_{i,t} &= \\lambda v_{i,t-1} + (1-\\lambda) u_{i,t}^{2}\\,, \\\\
Q_{ij,t} &= \\lambda_c Q_{ij,t-1} + (1-\\lambda_c) \\frac{u_{i,t} u_{j,t}}{\\sqrt{v_{i,t} v_{j,t}}}\\,, \\\\
\\rho_{ij,t} &= \\frac{Q_{ij,t}}{\\sqrt{Q_{ii,t} Q_{jj,t}}}\\,.
\\end{align}
```

Where:

  - ``v_{i,t}``: Raw exponentially weighted variance of asset ``i``.
  - ``Q_{ij,t}``: Raw exponentially weighted correlation state.
  - ``\\rho_{ij,t}``: Correlation, normalised from ``Q``.

A zero seed damps the state, so the read-out removes the damping before it reports:

```math
\\begin{align}
\\hat{\\Sigma}_{ij} &= \\mathrm{mult}(s_T)^{2}\\, \\frac{S_{ij,T}}{\\sqrt{(1-\\lambda^{n_i})(1-\\lambda^{n_j})}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:Sigma_hat])
  - ``n_i``: Count of valid observations of asset ``i``. A pairwise count corrects ``Q``.
  - ``\\mathrm{mult}(s_T)``: Regime multiplier of the smoothed regime state ``s_T``, clamped to
    `regime_lohi_mult` where that field is not `nothing`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RegimeAdjustedExpWeightedCovariance(;
        decay::Number                                         = exp2(-inv(40.0)),
        cor_decay::Option{<:Number}                           = nothing,
        min_obs::Integer                                      = round(Int, max(1, inv(log2(inv(decay))), isnothing(cor_decay) ? 1 : inv(log2(inv(cor_decay))))),
        hac_lags::Option{<:Integer}                           = nothing,
        regime_method::Option{<:RegimeAdjustedMethod}         = FirstMomentRegimeAdjusted(),
        regime_decay::Number                                  = exp2(-2 / inv(log2(inv(decay)))),
        regime_min_obs::Integer                               = round(Int, max(1, inv(log2(inv(decay))) / 2)),
        regime_target::RegimeAdjustedTarget                   = PortfolioTarget(),
        regime_lohi_mult::Option{<:Tuple{<:Number, <:Number}} = nothing,
        min_val::Number                                       = sqrt(eps()),
        centred::Bool                                         = false,
        cache::Option{<:AbstractPartialFitState}              = nothing
    ) -> RegimeAdjustedExpWeightedCovariance

Keywords correspond to the struct's fields. Where `cor_decay` is not `nothing`, the default
`min_obs` reads the slower of the two decays.

## Validation

  - $(val_dict[:decay])
  - If `cor_decay` is not `nothing`, `cor_decay > 0`, finite, and non-empty.
  - `min_obs > 0` and `regime_min_obs > 0`.
  - $(val_dict[:hac_lags])
  - If `regime_lohi_mult` is not `nothing`, `0 < regime_lohi_mult[1] < regime_lohi_mult[2]`.

# Examples

```jldoctest
julia> ce = RegimeAdjustedExpWeightedCovariance();

julia> ce.decay ≈ exp2(-inv(40.0))
true

julia> isnothing(ce.cor_decay)
true
```

# Related

  - [`RegimeAdjustedTarget`](@ref)
  - [`RegimeAdjustedMethod`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
  - [`RegimeAdjustedCovarianceState`](@ref)
  - [`partial_fit!`](@ref)
"""
@concrete struct RegimeAdjustedExpWeightedCovariance <: AbstractCovarianceEstimator
    """
    $(field_dict[:decay])
    """
    decay
    """
    $(field_dict[:cor_decay])
    """
    cor_decay
    """
    $(field_dict[:min_obs])
    """
    min_obs
    """
    $(field_dict[:hac_lags])
    """
    hac_lags
    """
    $(field_dict[:regime_method])
    """
    regime_method
    """
    $(field_dict[:regime_decay])
    """
    regime_decay
    """
    $(field_dict[:regime_min_obs])
    """
    regime_min_obs
    """
    $(field_dict[:regime_target])
    """
    regime_target
    """
    $(field_dict[:regime_lohi_mult])
    """
    regime_lohi_mult
    """
    $(field_dict[:min_val])
    """
    min_val
    """
    $(field_dict[:centred])
    """
    centred
    """
    Running state of an incremental fit, or `nothing` before the first call to [`partial_fit!`](@ref). It is the one Result this estimator holds, and its type bound is the enforcement of the rule that ADR 0106 excepts. [`Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance)`](@ref) reads it, and a fit over a matrix ignores it.
    """
    cache
    function RegimeAdjustedExpWeightedCovariance(decay::Number, cor_decay::Option{<:Number},
                                                 min_obs::Integer,
                                                 hac_lags::Option{<:Integer},
                                                 regime_method::Option{<:RegimeAdjustedMethod},
                                                 regime_decay::Number,
                                                 regime_min_obs::Integer,
                                                 regime_target::RegimeAdjustedTarget,
                                                 regime_lohi_mult::Option{<:Tuple{<:Number,
                                                                                  <:Number}},
                                                 min_val::Number, centred::Bool,
                                                 cache::Option{<:AbstractPartialFitState})
        assert_nonempty_gt0_finite_val(decay, :decay)
        if !isnothing(cor_decay)
            assert_nonempty_gt0_finite_val(cor_decay, :cor_decay)
        end
        assert_nonempty_gt0_finite_val(min_obs, :min_obs)
        assert_nonempty_gt0_finite_val(regime_min_obs, :regime_min_obs)
        if !isnothing(regime_lohi_mult)
            @argcheck(zero(regime_lohi_mult[1]) < regime_lohi_mult[1] < regime_lohi_mult[2],
                      DomainError)
        end
        if !isnothing(hac_lags)
            assert_nonempty_gt0_finite_val(hac_lags, :hac_lags)
        end
        return new{typeof(decay), typeof(cor_decay), typeof(min_obs), typeof(hac_lags),
                   typeof(regime_method), typeof(regime_decay), typeof(regime_min_obs),
                   typeof(regime_target), typeof(regime_lohi_mult), typeof(min_val),
                   typeof(centred), typeof(cache)}(decay, cor_decay, min_obs, hac_lags,
                                                   regime_method, regime_decay,
                                                   regime_min_obs, regime_target,
                                                   regime_lohi_mult, min_val, centred,
                                                   cache)
    end
end
function RegimeAdjustedExpWeightedCovariance(; decay::Number = exp2(-inv(40.0)),
                                             cor_decay::Option{<:Number} = nothing,
                                             min_obs::Integer = round(Int,
                                                                      max(1,
                                                                          inv(log2(inv(decay))),
                                                                          if isnothing(cor_decay)
                                                                              1
                                                                          else
                                                                              inv(log2(inv(cor_decay)))
                                                                          end)),
                                             hac_lags::Option{<:Integer} = nothing,
                                             regime_method::Option{<:RegimeAdjustedMethod} = FirstMomentRegimeAdjusted(),
                                             regime_decay::Number = exp2(-2 *
                                                                         log2(inv(decay))),
                                             regime_min_obs::Integer = round(Int,
                                                                             max(1,
                                                                                 inv(log2(inv(decay))) /
                                                                                 2)),
                                             regime_target::RegimeAdjustedTarget = PortfolioTarget(),
                                             regime_lohi_mult::Option{<:Tuple{<:Number,
                                                                              <:Number}} = nothing,
                                             min_val::Number = sqrt(eps()),
                                             centred::Bool = false,
                                             cache::Option{<:AbstractPartialFitState} = nothing)::RegimeAdjustedExpWeightedCovariance
    return RegimeAdjustedExpWeightedCovariance(decay, cor_decay, min_obs, hac_lags,
                                               regime_method, regime_decay, regime_min_obs,
                                               regime_target, regime_lohi_mult, min_val,
                                               centred, cache)
end
"""
$(DocStringExtensions.TYPEDEF)

Internal mutable cache for the online covariance update in [`RegimeAdjustedExpWeightedCovariance`](@ref).

This type is an implementation detail and is not intended for direct use.

The three fields that carry the separate correlation recursion are `nothing` where `cor_decay`
is `nothing`, because one decay then carries the whole matrix and no correlation state exists.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`RegimeAdjustedVarianceState`](@ref)
"""
@concrete struct RegimeAdjustedCovarianceState <: AbstractPartialFitState
    """
    $(field_dict[:ret_buffer])
    """
    ret_buffer
    """
    $(field_dict[:ra_covariance])
    """
    covariance
    """
    $(field_dict[:ra_variance])
    """
    variance
    """
    $(field_dict[:ra_cor_state])
    """
    cor_state
    """
    $(field_dict[:ra_pair_obs_count])
    """
    pair_obs_count
    """
    $(field_dict[:ra_XXt])
    """
    XXt
    """
    $(field_dict[:ra_Xi])
    """
    Xi
    """
    $(field_dict[:ra_X_old_i])
    """
    X_old_i
    """
    $(field_dict[:ra_location])
    """
    location
    """
    $(field_dict[:obs_count])
    """
    obs_count
    """
    $(field_dict[:ra_active])
    """
    active
    """
    $(field_dict[:regime_state])
    """
    regime_state
    """
    $(field_dict[:n_regime_obs])
    """
    n_regime_obs
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

States whether the estimator runs the variance and the correlation at separate decays.

A `cor_decay` of `nothing` states one decay for the whole matrix, and a `cor_decay` equal to
`decay` states the same recursion in two places. Both take the single covariance recursion, so
the cache allocates no correlation state.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator.

# Returns

  - `flag::Bool`: `true` where `cor_decay` is not `nothing` and differs from `decay`.

# Related

  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`RegimeAdjustedCovarianceState`](@ref)
"""
function has_separate_cor_decay(ce::RegimeAdjustedExpWeightedCovariance)
    return !isnothing(ce.cor_decay) && !isapprox(ce.cor_decay, ce.decay)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the stationary expectation of the log regime statistic for a target that reads the
whole active block.

[`MahalanobisTarget`](@ref) and [`DiagonalTarget`](@ref) both sum `n` standardised squares, so
the statistic is a ``\\chi^2(n)`` variate under correct calibration and its log has expectation
``\\psi(x n) + \\ln y``. The scalar case of [`RegimeAdjustedExpWeightedVariance`](@ref) is this
expression at `n = 1`.

# Arguments

  - `method::LogRegimeAdjusted`: Log regime adjustment method.
  - `::Union{MahalanobisTarget, DiagonalTarget}`: Regime-adjustment target.
  - `n::Integer`: Count of assets that contribute to the statistic.

# Returns

  - `kappa::Number`: `digamma(method.x * n) + log(method.y)`.

# Related

  - [`LogRegimeAdjusted`](@ref)
  - [`get_regime_state`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
"""
function regime_kappa(method::LogRegimeAdjusted,
                      ::Union{<:MahalanobisTarget, <:DiagonalTarget}, n::Integer)
    return SpecialFunctions.digamma(method.x * n) + log(method.y)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the stationary expectation of the log regime statistic for the portfolio target.

[`PortfolioTarget`](@ref) reads one direction and multiplies by `n`, so its statistic is `n`
times a ``\\chi^2(1)`` variate and the log has expectation ``\\ln n + \\psi(x) + \\ln y``.

# Arguments

  - `method::LogRegimeAdjusted`: Log regime adjustment method.
  - `::PortfolioTarget`: Portfolio regime-adjustment target.
  - `n::Integer`: Count of assets that contribute to the statistic.

# Returns

  - `kappa::Number`: `log(n) + digamma(method.x) + log(method.y)`.

# Related

  - [`LogRegimeAdjusted`](@ref)
  - [`get_regime_state`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
"""
function regime_kappa(method::LogRegimeAdjusted, ::PortfolioTarget, n::Integer)
    return log(n) + SpecialFunctions.digamma(method.x) + log(method.y)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the first-moment normalisation of the regime statistic for the portfolio target.

The statistic is `n` times a squared standard normal, so the root has expectation
``\\sqrt{n}\\,x`` with ``x = \\sqrt{2/\\pi}``.

# Arguments

  - `method::FirstMomentRegimeAdjusted`: First-moment regime adjustment method.
  - `::PortfolioTarget`: Portfolio regime-adjustment target.
  - `n::Integer`: Count of assets that contribute to the statistic.

# Returns

  - `denom::Number`: `sqrt(n) * method.x`.

# Related

  - [`FirstMomentRegimeAdjusted`](@ref)
  - [`get_regime_state`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
"""
function regime_denom(method::FirstMomentRegimeAdjusted, ::PortfolioTarget, n::Integer)
    return sqrt(n) * method.x
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the first-moment normalisation of the regime statistic for the Mahalanobis target.

The statistic is a ``\\chi^2(n)`` variate, so the root has expectation
``\\sqrt{2}\\,\\Gamma((n+1)/2)/\\Gamma(n/2)``. The expression is evaluated through the log
gamma function, which stays finite for a wide universe. At `n = 1` it is ``\\sqrt{2/\\pi}``,
which is the constant the scalar case of [`RegimeAdjustedExpWeightedVariance`](@ref) carries.

# Arguments

  - `::FirstMomentRegimeAdjusted`: First-moment regime adjustment method (unused).
  - `::MahalanobisTarget`: Mahalanobis regime-adjustment target.
  - `n::Integer`: Count of assets that contribute to the statistic.

# Returns

  - `denom::Number`: `sqrt(2) * exp(loggamma((n + 1) / 2) - loggamma(n / 2))`.

# Related

  - [`FirstMomentRegimeAdjusted`](@ref)
  - [`get_regime_state`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
"""
function regime_denom(::FirstMomentRegimeAdjusted, ::MahalanobisTarget, n::Integer)
    return sqrt(2.0) * exp(SpecialFunctions.loggamma(0.5 * (n + 1)) -
                           SpecialFunctions.loggamma(0.5 * n))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the first-moment normalisation of the regime statistic for the diagonal target.

The diagonal statistic ignores the correlations, so only the second-moment calibration is exact
in general. The root of `n` is kept as the diagonal-risk proxy.

# Arguments

  - `::FirstMomentRegimeAdjusted`: First-moment regime adjustment method (unused).
  - `::DiagonalTarget`: Diagonal regime-adjustment target.
  - `n::Integer`: Count of assets that contribute to the statistic.

# Returns

  - `denom::Number`: `sqrt(n)`.

# Related

  - [`FirstMomentRegimeAdjusted`](@ref)
  - [`get_regime_state`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
"""
function regime_denom(::FirstMomentRegimeAdjusted, ::DiagonalTarget, n::Integer)
    return sqrt(float(n))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Transforms the root-mean-squared regime statistics into the values the regime state smooths.

# Arguments

  - `::RootMeanSquaredAdjusted`: Root-mean-squared regime adjustment method (unused).
  - `::RegimeAdjustedTarget`: Regime-adjustment target (unused).
  - `stats::VecNum`: One statistic per calibration direction.
  - `n::Integer`: Count of assets that contribute to the statistic.
  - `::Any`: Ignored minimum value argument.

# Returns

  - `val::VecNum`: `stats ./ n`.

# Related

  - [`RootMeanSquaredAdjusted`](@ref)
  - [`regime_statistic`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
"""
function get_regime_state(::RootMeanSquaredAdjusted, ::RegimeAdjustedTarget, stats::VecNum,
                          n::Integer, ::Any)
    return stats ./ n
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Transforms the first-moment regime statistics into the values the regime state smooths.

# Arguments

  - `method::FirstMomentRegimeAdjusted`: First-moment regime adjustment method.
  - `target::RegimeAdjustedTarget`: Regime-adjustment target, which names the normalisation.
  - `stats::VecNum`: One statistic per calibration direction.
  - `n::Integer`: Count of assets that contribute to the statistic.
  - `::Any`: Ignored minimum value argument.

# Returns

  - `val::VecNum`: `sqrt.(max.(stats, 0)) ./ regime_denom(method, target, n)`.

# Related

  - [`FirstMomentRegimeAdjusted`](@ref)
  - [`regime_denom`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
"""
function get_regime_state(method::FirstMomentRegimeAdjusted, target::RegimeAdjustedTarget,
                          stats::VecNum, n::Integer, ::Any)
    return sqrt.(max.(stats, zero(eltype(stats)))) ./ regime_denom(method, target, n)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Transforms the log regime statistics into the values the regime state smooths.

# Arguments

  - `method::LogRegimeAdjusted`: Log regime adjustment method.
  - `target::RegimeAdjustedTarget`: Regime-adjustment target, which names the expectation.
  - `stats::VecNum`: One statistic per calibration direction.
  - `n::Integer`: Count of assets that contribute to the statistic.
  - `min_val::Number`: Floor applied before the logarithm.

# Returns

  - `val::VecNum`: `log.(max.(stats, min_val)) .- regime_kappa(method, target, n)`.

# Related

  - [`LogRegimeAdjusted`](@ref)
  - [`regime_kappa`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
"""
function get_regime_state(method::LogRegimeAdjusted, target::RegimeAdjustedTarget,
                          stats::VecNum, n::Integer, min_val::Number)
    return log.(max.(stats, min_val)) .- regime_kappa(method, target, n)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Factorises a covariance block for the Mahalanobis regime statistic, and refuses rather than
throws when no ridge makes it factorise.

A block that carries a late-listed asset is not yet positive definite, and the regime statistic
is one observation of a smoother rather than a result a caller reads. A refusal therefore skips
that observation's regime update, and the fit continues.

# Algorithm

 1. Try the plain factorisation of the lower triangle of `C`. Return it where it succeeds.
 2. Symmetrise `C`, and take the mean absolute diagonal as the scale. Where that is not a finite
    positive number, take the largest absolute entry, and at least one.
 3. Add a ridge of `max(min_val * scale, eps * scale)` to the diagonal, and try again. Multiply
    the ridge by ten after each failure, for three tries in all.
 4. Return `nothing` where every try fails.

# Arguments

  - `C::MatNum`: Covariance block of the assets that contribute to the statistic.
  - `min_val::Number`: Scale of the first ridge.

# Returns

  - `chol::Option{<:LinearAlgebra.Cholesky}`: The factorisation, or `nothing` where no ridge
    makes the block factorise.

# Related

  - [`regime_statistic`](@ref)
  - [`MahalanobisTarget`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
"""
function safe_regime_cholesky(C::MatNum, min_val::Number)
    chol = LinearAlgebra.cholesky(LinearAlgebra.Hermitian(C, :L); check = false)
    if LinearAlgebra.issuccess(chol)
        return chol
    end
    S = (C + transpose(C)) / 2
    base = Statistics.mean(abs, LinearAlgebra.diag(S))
    scale = if base > zero(base) && isfinite(base)
        base
    else
        max(maximum(abs, S), one(base))
    end
    ridge = max(min_val * scale, eps(float(scale)) * scale)
    for _ in 1:3
        chol = LinearAlgebra.cholesky(LinearAlgebra.Hermitian(S + ridge * LinearAlgebra.I,
                                                              :L); check = false)
        if LinearAlgebra.issuccess(chol)
            return chol
        end
        ridge *= 10
    end

    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the squared Mahalanobis distance of one observation against the covariance block.

The target reads every eigen-direction of the block, so it calibrates the whole covariance
structure. It needs two active assets, which [`min_active_assets`](@ref) states.

# Arguments

  - `::MahalanobisTarget`: Mahalanobis regime-adjustment target.
  - `X::VecNum`: Centred returns of the assets that contribute to the statistic.
  - `C::MatNum`: Bias-corrected covariance block of those assets.
  - `::AbstractVector{<:Integer}`: Ignored index of those assets.
  - `min_val::Number`: Scale of the ridge that [`safe_regime_cholesky`](@ref) applies.

# Returns

  - `stats::Option{<:VecNum}`: One statistic, or `nothing` where the block does not factorise.

# Related

  - [`MahalanobisTarget`](@ref)
  - [`safe_regime_cholesky`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
"""
function regime_statistic(::MahalanobisTarget, X::VecNum, C::MatNum,
                          ::AbstractVector{<:Integer}, min_val::Number)
    chol = safe_regime_cholesky(C, min_val)
    if isnothing(chol)
        return nothing
    end
    y = chol.L \ X

    return [max(sum(abs2, y), zero(eltype(y)))]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the squared standardised Euclidean distance of one observation against the marginal
volatilities of the covariance block.

The target divides each return by its own volatility and sums the squares, so it calibrates the
diagonal risk scale and reads no correlation.

# Arguments

  - `::DiagonalTarget`: Diagonal regime-adjustment target.
  - `X::VecNum`: Centred returns of the assets that contribute to the statistic.
  - `C::MatNum`: Bias-corrected covariance block of those assets.
  - `::AbstractVector{<:Integer}`: Ignored index of those assets.
  - `min_val::Number`: Floor applied to each variance before its root is taken.

# Returns

  - `stats::VecNum`: One statistic.

# Related

  - [`DiagonalTarget`](@ref)
  - [`get_regime_state`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
"""
function regime_statistic(::DiagonalTarget, X::VecNum, C::MatNum,
                          ::AbstractVector{<:Integer}, min_val::Number)
    sigma = sqrt.(max.(LinearAlgebra.diag(C), min_val))

    return [sum(abs2, X ./ sigma)]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the squared standardised return of one observation along each portfolio direction.

Where `target.w` is `nothing`, one inverse-volatility direction is rebuilt from the block at
every observation, which neutralises the dispersion of the volatilities so that a loud asset
does not carry the statistic on its own. Where `target.w` holds weights, each row of that matrix
is one direction, its entries are restricted to the assets that contribute, and a row that keeps
no weight is dropped. Each direction gives its own statistic, and the caller averages the
transformed values.

# Algorithm

 1. Build the weight matrix over the contributing assets. Return `nothing` where no row keeps a
    positive weight.
 2. Normalise each row to sum to one.
 3. Return the count of contributing assets times the squared portfolio return, divided by the
    portfolio variance, for each row.

# Arguments

  - `target::PortfolioTarget`: Portfolio regime-adjustment target.
  - `X::VecNum`: Centred returns of the assets that contribute to the statistic.
  - `C::MatNum`: Bias-corrected covariance block of those assets.
  - `idx::AbstractVector{<:Integer}`: Index of those assets in the universe.
  - `min_val::Number`: Floor applied to each variance and to each portfolio variance.

# Returns

  - `stats::Option{<:VecNum}`: One statistic per direction, or `nothing` where no row keeps a
    positive weight.

# Related

  - [`PortfolioTarget`](@ref)
  - [`get_regime_state`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
"""
function regime_statistic(target::PortfolioTarget, X::VecNum, C::MatNum,
                          idx::AbstractVector{<:Integer}, min_val::Number)
    w = target.w
    W = if isnothing(w)
        inv_sigma = inv.(sqrt.(max.(LinearAlgebra.diag(C), min_val)))
        permutedims(inv_sigma / sum(inv_sigma))
    else
        Wi = isa(w, AbstractMatrix) ? w[:, idx] : permutedims(w[idx])
        keep = vec(sum(Wi; dims = 2)) .> zero(eltype(Wi))
        if !any(keep)
            return nothing
        end
        Wi = Wi[keep, :]
        Wi ./ sum(Wi; dims = 2)
    end
    r = W * X
    v = max.(vec(sum((W * C) .* W; dims = 2)), min_val)

    return length(X) * r .^ 2 ./ v
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the outer product of one observation, with the Newey-West HAC correction where
`hac_lags` is not `nothing`.

The correction adds each lagged cross-product and its transpose, weighted by the Bartlett kernel
``w_j = 1 - j/(L+1)``, so the update reads the serial correlation of the returns. An entry of a
lagged observation that is not finite is read as zero, which freezes that pair's contribution.

# Arguments

  - `cache::RegimeAdjustedCovarianceState`: Online covariance computation cache (mutated).
  - `ce::RegimeAdjustedExpWeightedCovariance`: Covariance estimator configuration.
  - `X::VecNum`: Current centred returns vector.

# Returns

  - `XXt::MatNum`: The HAC-adjusted outer product stored in `cache.XXt`.

# Related

  - [`RegimeAdjustedCovarianceState`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`hac_squared_returns!`](@ref)
"""
function hac_outer_product!(cache::RegimeAdjustedCovarianceState,
                            ce::RegimeAdjustedExpWeightedCovariance, X::VecNum)
    cache.XXt .= X .* transpose(X)
    if isnothing(cache.ret_buffer) || isempty(cache.ret_buffer)
        return cache.XXt
    end

    for (i, X_old) in enumerate(Iterators.reverse(cache.ret_buffer))
        wi = one(eltype(X)) - i / (ce.hac_lags + 1)
        cache.X_old_i .= replace(X_old, NaN => zero(eltype(X_old)))
        cross = X .* transpose(cache.X_old_i)
        cache.XXt .+= wi * (cross + transpose(cross))
    end

    return cache.XXt
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Advances the separate variance and correlation recursions, and rebuilds the covariance from
them.

This is the path `cor_decay` opens. The variance runs at `decay` and the correlation state at
`cor_decay`, which lets a volatility that mean-reverts quickly sit beside a correlation that
needs more data. The two are recombined into `cache.covariance`, which the regime statistic
reads.

# Algorithm

 1. Advance the variance with the diagonal of the outer product, floored at zero.
 2. Standardise the outer product by the running volatilities. An asset whose variance is not
    above `min_val` contributes zero.
 3. Advance the correlation state where both assets of the pair are valid, and count the pair.
 4. Normalise the correlation state of the active block to a unit diagonal, symmetrise it, and
    rescale it by the running volatilities into `cache.covariance`.

# Arguments

  - `cache::RegimeAdjustedCovarianceState`: Online covariance computation cache (mutated).
  - `ce::RegimeAdjustedExpWeightedCovariance`: Covariance estimator configuration.
  - `valid::AbstractVector{<:Bool}`: Assets with a finite return that are active.
  - `pair_valid::AbstractMatrix{<:Bool}`: Pairs whose two assets are both valid.

# Returns

  - `nothing`: The cache is mutated in place.

# Related

  - [`RegimeAdjustedCovarianceState`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`has_separate_cor_decay`](@ref)
"""
function update_var_cor!(cache::RegimeAdjustedCovarianceState,
                         ce::RegimeAdjustedExpWeightedCovariance,
                         valid::AbstractVector{<:Bool}, pair_valid::AbstractMatrix{<:Bool})
    T = eltype(cache.variance)
    hac_var = max.(LinearAlgebra.diag(cache.XXt), zero(T))
    cache.variance[valid] .= ce.decay * view(cache.variance, valid) +
                             (one(ce.decay) - ce.decay) * view(hac_var, valid)
    positive = valid .& (cache.variance .> ce.min_val)
    inv_sigma = ifelse.(positive, inv.(sqrt.(ifelse.(positive, cache.variance, one(T)))),
                        zero(T))
    outer_std = cache.XXt .* (inv_sigma .* transpose(inv_sigma))
    cache.cor_state .= ifelse.(pair_valid,
                               ce.cor_decay * cache.cor_state +
                               (one(ce.cor_decay) - ce.cor_decay) * outer_std,
                               cache.cor_state)
    cache.pair_obs_count[pair_valid] .+= 1

    active = cache.active .& (cache.variance .> zero(T))
    if !any(active)
        return nothing
    end
    idx = findall(active)
    cor_raw = cache.cor_state[idx, idx]
    d = LinearAlgebra.diag(cor_raw)
    if !all(d .> ce.min_val)
        return nothing
    end
    inv_d = inv.(sqrt.(d))
    rho = clamp.(cor_raw .* (inv_d .* transpose(inv_d)), -one(T), one(T))
    rho .= (rho + transpose(rho)) / 2
    rho[LinearAlgebra.diagind(rho)] .= one(T)
    sigma = sqrt.(view(cache.variance, idx))
    cache.covariance[idx, idx] = rho .* sigma .* transpose(sigma)

    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Removes the damping a zero seed leaves in the running state, and returns the covariance that
read-out reports.

The recursion is seeded at zero, which keeps the state positive semi-definite at every step and
damps it by ``1 - \\lambda^{n}`` after `n` observations. The correction is a congruence
transform, so it restores the scale without breaking that property. Where `cor_decay` opens the
separate path, the variance and the correlation are corrected at their own decays, and the
correlation reads a pairwise count so that an asynchronous listing is corrected pair by pair.

# Arguments

  - `cache::RegimeAdjustedCovarianceState`: Online covariance computation cache.
  - `ce::RegimeAdjustedExpWeightedCovariance`: Covariance estimator configuration.

# Returns

  - `sigma::MatNum`: The bias-corrected covariance matrix. An inactive asset is `NaN` in its own
    row and column.

# Related

  - [`RegimeAdjustedCovarianceState`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`regime_adjusted_covariance`](@ref)
"""
function bias_corrected_covariance(cache::RegimeAdjustedCovarianceState,
                                   ce::RegimeAdjustedExpWeightedCovariance)
    T = eltype(cache.covariance)
    N = length(cache.obs_count)
    counted = cache.obs_count .> zero(eltype(cache.obs_count))
    if !has_separate_cor_decay(ce)
        sigma = copy(cache.covariance)
        correction = ifelse.(counted,
                             inv.(sqrt.(max.(one(ce.decay) .- ce.decay .^ cache.obs_count,
                                             eps(ce.decay)))), one(T))
        sigma .*= correction .* transpose(correction)
        inactive = .!cache.active
        if any(inactive)
            sigma[inactive, :] .= T(NaN)
            sigma[:, inactive] .= T(NaN)
        end
        return sigma
    end

    sigma = fill(T(NaN), N, N)
    active = cache.active .& counted
    if !any(active)
        return sigma
    end
    idx = findall(active)
    var_bc = view(cache.variance, idx) .*
             inv.(max.(one(ce.decay) .- ce.decay .^ view(cache.obs_count, idx),
                       eps(ce.decay)))
    vol = sqrt.(max.(var_bc, ce.min_val))
    cor_raw = cache.cor_state[idx, idx] .*
              inv.(max.(one(ce.cor_decay) .- ce.cor_decay .^ cache.pair_obs_count[idx, idx],
                        eps(ce.cor_decay)))
    inv_d = inv.(sqrt.(clamp.(LinearAlgebra.diag(cor_raw), ce.min_val, T(Inf))))
    rho = clamp.(cor_raw .* (inv_d .* transpose(inv_d)), -one(T), one(T))
    rho .= (rho + transpose(rho)) / 2
    rho[LinearAlgebra.diagind(rho)] .= one(T)
    sigma[idx, idx] = rho .* vol .* transpose(vol)

    return sigma
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the bias-corrected covariance block of the assets that contribute to the regime
statistic.

This block calibrates the smoother and is never reported, so it applies the per-asset variance
correction alone. Where the separate path runs, the correlation already sits inside
`cache.covariance` normalised, and its pairwise correction cancels in that normalisation for a
synchronous sample. [`bias_corrected_covariance`](@ref) is the exact read-out.

# Arguments

  - `cache::RegimeAdjustedCovarianceState`: Online covariance computation cache.
  - `ce::RegimeAdjustedExpWeightedCovariance`: Covariance estimator configuration.
  - `idx::AbstractVector{<:Integer}`: Index of the assets that contribute to the statistic.

# Returns

  - `sigma::MatNum`: The bias-corrected covariance block of those assets.

# Related

  - [`RegimeAdjustedCovarianceState`](@ref)
  - [`regime_statistic`](@ref)
  - [`bias_corrected_covariance`](@ref)
"""
function regime_covariance_block(cache::RegimeAdjustedCovarianceState,
                                 ce::RegimeAdjustedExpWeightedCovariance,
                                 idx::AbstractVector{<:Integer})
    sigma = cache.covariance[idx, idx]
    correction = inv.(sqrt.(max.(one(ce.decay) .- ce.decay .^ view(cache.obs_count, idx),
                                 eps(ce.decay))))
    sigma .*= correction .* transpose(correction)

    return sigma
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Advances the smoothed regime state with one observation, one step ahead of the covariance
update.

The observation is scored against the state that stands before it, so the statistic compares
realised risk against predicted risk rather than against itself. Only an asset that was already
above `min_obs` contributes, because the statistic is sensitive to a poorly estimated asset. A
`regime_method` of `nothing` advances nothing.

# Arguments

  - `cache::RegimeAdjustedCovarianceState`: Online covariance computation cache.
  - `ce::RegimeAdjustedExpWeightedCovariance`: Covariance estimator configuration.
  - `X::VecNum`: Current centred returns vector, zeroed where the asset is not valid.
  - `valid::AbstractVector{<:Bool}`: Assets with a finite return that are active.
  - `estimation_mask::Option{<:AbstractVector{<:Bool}}`: Optional mask restricting which assets
    contribute to the statistic.

# Returns

  - `cache::RegimeAdjustedCovarianceState`: The cache to read on and to pass to the next
    observation. `regime_state` and `n_regime_obs` are immutable fields that `Accessors.@reset`
    replaces, so the caller must rebind the cache to this return value.

# Related

  - [`RegimeAdjustedCovarianceState`](@ref)
  - [`regime_statistic`](@ref)
  - [`get_regime_state`](@ref)
"""
function update_regime!(cache::RegimeAdjustedCovarianceState,
                        ce::RegimeAdjustedExpWeightedCovariance, X::VecNum,
                        valid::AbstractVector{<:Bool},
                        estimation_mask::Option{<:AbstractVector{<:Bool}})
    if isnothing(ce.regime_method)
        return cache
    end
    regime_mask = valid .& cache.active .& (cache.obs_count .>= ce.min_obs)
    if !isnothing(estimation_mask)
        regime_mask .&= estimation_mask
    end
    n = count(regime_mask)
    if n < min_active_assets(ce.regime_target)
        return cache
    end
    idx = findall(regime_mask)
    stats = regime_statistic(ce.regime_target, X[idx],
                             regime_covariance_block(cache, ce, idx), idx, ce.min_val)
    if isnothing(stats)
        return cache
    end
    transformed = Statistics.mean(get_regime_state(ce.regime_method, ce.regime_target,
                                                   stats, n, ce.min_val))

    Accessors.@reset cache.regime_state = if isnothing(cache.regime_state)
        transformed
    else
        ce.regime_decay * cache.regime_state +
        (one(ce.regime_decay) - ce.regime_decay) * transformed
    end

    Accessors.@reset cache.n_regime_obs += 1

    return cache
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Processes a single observation row (or column) to update the online covariance cache.

Updates the running location, advances the regime state one step ahead of the covariance, and
then advances the covariance itself. An asset that turns inactive at this observation has its
row and column zeroed and its counts reset, so a later listing starts from a cold state.

# Arguments

  - `cache::RegimeAdjustedCovarianceState`: Online covariance computation cache (mutated).
  - `ce::RegimeAdjustedExpWeightedCovariance`: Covariance estimator configuration.
  - `X::VecNum`: Returns vector for the current observation.
  - `estimation_mask::Option{<:AbstractVector{<:Bool}}`: Optional mask restricting which assets
    contribute to the regime state update.
  - `active_mask::Option{<:AbstractVector{<:Bool}}`: Optional mask of currently active assets.

# Returns

  - `cache::RegimeAdjustedCovarianceState`: The cache to read on and to pass to the next
    observation. The arrays are mutated in place, and the caller must rebind the cache to this
    return value because `regime_state` and `n_regime_obs` are immutable fields.

# Related

  - [`RegimeAdjustedCovarianceState`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`update_regime!`](@ref)
"""
function process_observation!(cache::RegimeAdjustedCovarianceState,
                              ce::RegimeAdjustedExpWeightedCovariance, X::VecNum,
                              estimation_mask::Option{<:AbstractVector{<:Bool}},
                              active_mask::Option{<:AbstractVector{<:Bool}})
    T = eltype(cache.covariance)
    finite_mask = isfinite.(X)
    valid = isnothing(active_mask) ? finite_mask : (finite_mask .& active_mask)
    filled = ifelse.(valid, X, zero(T))
    if ce.centred
        cache.Xi .= filled
    else
        loc = replace(cache.location, NaN => zero(T))
        cache.Xi .= filled .- loc
        cache.location[valid] .= ce.decay * view(loc, valid) +
                                 (one(ce.decay) - ce.decay) * view(filled, valid)
    end
    cache.Xi .= ifelse.(valid, cache.Xi, zero(T))

    cache = update_regime!(cache, ce, cache.Xi, valid, estimation_mask)

    hac_outer_product!(cache, ce, cache.Xi)
    pair_valid = valid .& transpose(valid)
    if has_separate_cor_decay(ce)
        update_var_cor!(cache, ce, valid, pair_valid)
    else
        cache.covariance .= ifelse.(pair_valid,
                                    ce.decay * cache.covariance +
                                    (one(ce.decay) - ce.decay) * cache.XXt,
                                    cache.covariance)
    end
    cache.obs_count[valid] .+= 1

    if !isnothing(cache.ret_buffer)
        X_new = copy(cache.Xi)
        X_new[.!valid] .= T(NaN)
        push!(cache.ret_buffer, X_new)
    end

    if isnothing(active_mask)
        cache.active .= true
        return cache
    end

    newly_inactive = cache.active .& .!active_mask
    if any(newly_inactive)
        cache.covariance[newly_inactive, :] .= zero(T)
        cache.covariance[:, newly_inactive] .= zero(T)
        cache.obs_count[newly_inactive] .= 0
        if has_separate_cor_decay(ce)
            cache.variance[newly_inactive] .= zero(T)
            cache.cor_state[newly_inactive, :] .= zero(T)
            cache.cor_state[:, newly_inactive] .= zero(T)
            cache.pair_obs_count[newly_inactive, :] .= 0
            cache.pair_obs_count[:, newly_inactive] .= 0
        end
        if !ce.centred
            cache.location[newly_inactive] .= T(NaN)
        end
    end
    cache.active .= active_mask

    return cache
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Accepts every regime-adjustment target that names no weights.

# Arguments

  - `::RegimeAdjustedTarget`: Regime-adjustment target (unused).
  - `::Integer`: Ignored count of assets.

# Returns

  - `nothing`.

# Related

  - [`RegimeAdjustedTarget`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
"""
function assert_regime_target(::RegimeAdjustedTarget, ::Integer)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Checks the portfolio weights of the regime-adjustment target against the universe of the fit.

The target holds the weights, and the fit holds the universe, so the two meet here and nowhere
earlier. Weights of `nothing` name the inverse-volatility direction, which needs no check.

# Arguments

  - `target::PortfolioTarget`: Portfolio regime-adjustment target.
  - `N::Integer`: Count of assets in the fit.

# Validation

  - `target.w` names `N` assets.
  - Every weight is non-negative, and every row sums to a positive number.

# Returns

  - `nothing`.

# Related

  - [`PortfolioTarget`](@ref)
  - [`regime_statistic`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
"""
function assert_regime_target(target::PortfolioTarget, N::Integer)
    w = target.w
    if isnothing(w)
        return nothing
    end
    W = isa(w, AbstractMatrix) ? w : permutedims(w)
    @argcheck(size(W, 2) == N,
              DimensionMismatch("`regime_target.w` names $(size(W, 2)) assets, and `X` holds $N"))
    @argcheck(all(x -> x >= zero(x), W), DomainError)
    @argcheck(all(x -> x > zero(x), vec(sum(W; dims = 2))), DomainError)

    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Run one forward pass of the online covariance update over the observations of `X`, and call `f`
after each observation.

The pass owns the argument validation, the orientation and the cache, so every verb that reads
it states the recursion once.

# Arguments

  - `f`: Called as `f(i, cache)` after observation `i`. Pass `(args...) -> nothing` to run the
    pass for its final cache alone.
  - `ce::RegimeAdjustedExpWeightedCovariance`: Covariance estimator configuration.
  - `X::MatNum`: Observation matrix.
  - `dims::Int`: Dimension along which the observations lie.
  - `estimation_mask::Option{<:AbstractMatrix{<:Bool}}`: Optional mask restricting which assets
    contribute to the regime state update.
  - `active_mask::Option{<:AbstractMatrix{<:Bool}}`: Optional mask of active assets.
  - `state::Option{<:RegimeAdjustedCovarianceState}`: Optional state to continue. A `nothing`
    starts a cold cache, which is what a fit over a whole sample needs. [`partial_fit!`](@ref)
    passes the estimator's own state.

# Validation

  - $(val_dict[:dims])
  - If `estimation_mask` is not `nothing`, `size(X) == size(estimation_mask)`.
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`.
  - The portfolio weights of `ce.regime_target`, where it names any.
  - If `state` is not `nothing`, it holds as many assets as `X`.

# Returns

  - `cache::RegimeAdjustedCovarianceState`: The cache after the last observation.

# Related

  - [`RegimeAdjustedCovarianceState`](@ref)
  - [`process_observation!`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
"""
function regime_adjusted_covariance_pass!(f, ce::RegimeAdjustedExpWeightedCovariance,
                                          X::MatNum, dims::Int,
                                          estimation_mask::Option{<:AbstractMatrix{<:Bool}},
                                          active_mask::Option{<:AbstractMatrix{<:Bool}},
                                          state::Option{<:RegimeAdjustedCovarianceState} = nothing)
    assert_dims(dims)
    est_flag = !isnothing(estimation_mask)
    act_flag = !isnothing(active_mask)
    itr, v = ifelse(isone(dims), (eachrow, (x, y) -> view(x, y, :)),
                    (eachcol, (x, y) -> view(x, :, y)))
    if est_flag
        @argcheck(size(X) == size(estimation_mask),
                  DimensionMismatch("size(X) ($(size(X))) must match size(estimation_mask) ($(size(estimation_mask)))"))
    end
    if act_flag
        @argcheck(size(X) == size(active_mask),
                  DimensionMismatch("size(X) ($(size(X))) must match size(active_mask) ($(size(active_mask)))"))
    end
    N = size(X, setdiff((1, 2), (dims,))[1])
    assert_regime_target(ce.regime_target, N)

    cache = if isnothing(state)
        separate = has_separate_cor_decay(ce)
        RegimeAdjustedCovarianceState(if isnothing(ce.hac_lags)
                                          nothing
                                      else
                                          DataStructures.CircularBuffer{Vector{eltype(X)}}(ce.hac_lags)
                                      end, zeros(eltype(X), N, N),
                                      separate ? zeros(eltype(X), N) : nothing,
                                      separate ? zeros(eltype(X), N, N) : nothing,
                                      separate ? zeros(Int, N, N) : nothing,
                                      zeros(eltype(X), N, N), zeros(eltype(X), N),
                                      zeros(eltype(X), N),
                                      ce.centred ? zeros(eltype(X), N) : fill(NaN, N),
                                      zeros(Int, N), trues(N), nothing, 0)
    else
        @argcheck(size(state.covariance, 1) == N,
                  DimensionMismatch("the state holds $(size(state.covariance, 1)) assets, and `X` holds $N"))
        state
    end
    for (i, Xi) in enumerate(itr(X))
        emi = est_flag ? v(estimation_mask, i) : nothing
        ami = act_flag ? v(active_mask, i) : nothing
        cache = process_observation!(cache, ce, Xi, emi, ami)
        f(i, cache)
    end

    return cache
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Read the regime-adjusted covariance out of a cache, as it stands.

Removes the damping of the zero seed, blanks every asset that is not ready, symmetrises the
block that is, and scales by the square of the regime multiplier. The cache is read, never
written, so the same cache answers this call after every observation of a forward pass.

Where `ce.regime_method` is `nothing`, [`process_observation!`](@ref) advances no regime state,
so `cache.n_regime_obs` stays at zero, which is below every admissible `regime_min_obs`. The
multiplier is then one and the covariance is the plain recursion.

Where `ce.regime_lohi_mult` is not `nothing`, the multiplier is clamped to that `(lo, hi)` range
before it is squared. Where it is `nothing`, no clamp runs.

# Arguments

  - `cache::RegimeAdjustedCovarianceState`: Online covariance computation cache.
  - `ce::RegimeAdjustedExpWeightedCovariance`: Covariance estimator configuration.

# Returns

  - `sigma::MatNum`: Regime-adjusted covariance matrix. An asset with fewer than `ce.min_obs`
    observations, or one that is not active, is `NaN` in its own row and column.

# Related

  - [`RegimeAdjustedCovarianceState`](@ref)
  - [`bias_corrected_covariance`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
"""
function regime_adjusted_covariance(cache::RegimeAdjustedCovarianceState,
                                    ce::RegimeAdjustedExpWeightedCovariance)
    T = eltype(cache.covariance)
    sigma = bias_corrected_covariance(cache, ce)
    not_ready = cache.obs_count .< ce.min_obs
    if any(not_ready)
        sigma[not_ready, :] .= T(NaN)
        sigma[:, not_ready] .= T(NaN)
    end
    ready = isfinite.(LinearAlgebra.diag(sigma))
    if any(ready)
        idx = findall(ready)
        block = sigma[idx, idx]
        sigma[idx, idx] = (block + transpose(block)) / 2
    end

    factor = if cache.n_regime_obs < ce.regime_min_obs
        one(T)
    else
        regime_multiplier(ce.regime_method, cache.regime_state)
    end
    if !isnothing(ce.regime_lohi_mult)
        factor = clamp(factor, ce.regime_lohi_mult[1], ce.regime_lohi_mult[2])
    end

    return sigma * factor^2
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Rescales a regime-adjusted covariance matrix to a unit diagonal.

An asset that the covariance blanks carries a `NaN` variance, so its row and column stay `NaN`
here, and its diagonal entry stays `NaN` rather than becoming one. A finite entry outside
`[-1, 1]` is round-off, and it is clamped.

# Arguments

  - `sigma::MatNum`: Regime-adjusted covariance matrix.

# Returns

  - `rho::MatNum`: The correlation matrix of `sigma`.

# Related

  - [`regime_adjusted_covariance`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
"""
function regime_adjusted_correlation(sigma::MatNum)
    T = eltype(sigma)
    d = LinearAlgebra.diag(sigma)
    inv_vol = inv.(sqrt.(d))
    rho = clamp.(sigma .* (inv_vol .* transpose(inv_vol)), -one(T), one(T))
    rho[LinearAlgebra.diagind(rho)] .= ifelse.(isfinite.(d), one(T), T(NaN))

    return rho
end
"""
    Statistics.cov(
        ce::RegimeAdjustedExpWeightedCovariance,
        X::MatNum;
        dims::Int = 1,
        estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        kwargs...
    ) -> MatNum

Compute the regime-adjusted exponentially weighted covariance matrix.

Iterates over the observation dimension of `X`, updating an online covariance cache at each
step. After the last observation, removes the damping of the zero seed and scales the result by
the square of the regime multiplier.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `estimation_mask`: Optional boolean matrix with the same size as `X`. When provided, only
    assets where `estimation_mask[i, :]` (or `[:, i]`) is `true` contribute to the regime state
    update for observation `i`.
  - `active_mask`: Optional boolean matrix with the same size as `X`. When provided, assets that
    become inactive have their covariance entries and observation count reset.
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `estimation_mask` is not `nothing`, `size(X) == size(estimation_mask)`.
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`.

# Returns

  - $(ret_dict[:sigma])

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012];

julia> ce = RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 2, regime_min_obs = 2);

julia> size(cov(ce, X))
(2, 2)
```

# Related

  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`RegimeAdjustedCovarianceState`](@ref)
  - [`regime_adjusted_covariance`](@ref)
  - [`Statistics.cor(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
"""
function Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1,
                        estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    cache = regime_adjusted_covariance_pass!((args...) -> nothing, ce, X, dims,
                                             estimation_mask, active_mask)
    if !ce.centred
        unseen = cache.obs_count .< one(eltype(cache.obs_count))
        if any(unseen)
            cache.location[unseen] .= NaN
        end
    end

    return regime_adjusted_covariance(cache, ce)
end
"""
    Statistics.cor(
        ce::RegimeAdjustedExpWeightedCovariance,
        X::MatNum;
        dims::Int = 1,
        estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        kwargs...
    ) -> MatNum

Compute the regime-adjusted exponentially weighted correlation matrix.

This is the covariance of the same call, rescaled to a unit diagonal. The regime multiplier
scales the whole matrix, so it cancels in the rescale and the correlation does not read it.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `estimation_mask`: Optional boolean matrix with the same size as `X`. When provided, only
    assets where `estimation_mask[i, :]` (or `[:, i]`) is `true` contribute to the regime state
    update for observation `i`.
  - `active_mask`: Optional boolean matrix with the same size as `X`. When provided, assets that
    become inactive have their covariance entries and observation count reset.
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `estimation_mask` is not `nothing`, `size(X) == size(estimation_mask)`.
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`.

# Returns

  - $(ret_dict[:rho])

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012];

julia> ce = RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 2, regime_min_obs = 2);

julia> LinearAlgebra.diag(cor(ce, X))
2-element Vector{Float64}:
 1.0
 1.0
```

# Related

  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`regime_adjusted_correlation`](@ref)
  - [`Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
"""
function Statistics.cor(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1,
                        estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    return regime_adjusted_correlation(Statistics.cov(ce, X; dims = dims,
                                                      estimation_mask = estimation_mask,
                                                      active_mask = active_mask, kwargs...))
end
"""
    partial_fit!(
        ce::RegimeAdjustedExpWeightedCovariance,
        X::MatNum;
        dims::Int = 1,
        estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        kwargs...
    ) -> RegimeAdjustedExpWeightedCovariance

Fold a block of observations into the estimator's own online covariance state.

The recursion reads one observation at a time, so a block folded on top of an existing state is
what the same observations give when they are read in one pass. That is what this verb gives and
what [`merge_states`](@ref) refuses: two blocks each fitted from a cold start do not add.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `estimation_mask`: Optional boolean matrix with the same size as `X`, restricting which
    assets contribute to the regime state update.
  - `active_mask`: Optional boolean matrix with the same size as `X`. An asset that becomes
    inactive has its covariance entries and observation count reset.
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `estimation_mask` is not `nothing`, `size(X) == size(estimation_mask)`.
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`.
  - If `ce.cache` is not `nothing`, it holds as many assets as `X`.

# Returns

  - `ce::RegimeAdjustedExpWeightedCovariance`: The estimator, with `cache` holding the state
    after the block.

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012; 0.008 -0.02; -0.02 0.03];

julia> ce = RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 2, regime_min_obs = 2);

julia> isequal(cov(partial_fit!(ce, X)), cov(ce, X))
true

julia> halves = partial_fit!(partial_fit!(ce, X[1:3, :]), X[4:6, :]);

julia> isequal(cov(halves), cov(ce, X))
true
```

# Related

  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`RegimeAdjustedCovarianceState`](@ref)
  - [`regime_adjusted_covariance_pass!`](@ref)
  - [`Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance)`](@ref)
"""
function partial_fit!(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1,
                      estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                      active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    cache = regime_adjusted_covariance_pass!((args...) -> nothing, ce, X, dims,
                                             estimation_mask, active_mask, ce.cache)
    Accessors.@reset ce.cache = cache

    return ce
end
"""
    partial_fit!(
        ce::RegimeAdjustedExpWeightedCovariance,
        x::VecNum;
        estimation_mask::Option{<:AbstractVector{<:Bool}} = nothing,
        active_mask::Option{<:AbstractVector{<:Bool}} = nothing,
        kwargs...
    ) -> RegimeAdjustedExpWeightedCovariance

Fold one observation into the estimator's own online covariance state.

The entries of `x` are the assets of a single observation, which is the row the matrix method
folds one at a time. This is the shape a caller has when the observations arrive one by one.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator.
  - `x::VecNum`: One observation, with one entry per asset.
  - `estimation_mask`: Optional boolean vector with the same length as `x`, restricting which
    assets contribute to the regime state update.
  - `active_mask`: Optional boolean vector with the same length as `x`. An asset that becomes
    inactive has its covariance entries and observation count reset.
  - $(arg_dict[:ignkwargs])

# Validation

  - If `estimation_mask` is not `nothing`, `length(x) == length(estimation_mask)`.
  - If `active_mask` is not `nothing`, `length(x) == length(active_mask)`.
  - If `ce.cache` is not `nothing`, it holds as many assets as `x`.

# Returns

  - `ce::RegimeAdjustedExpWeightedCovariance`: The estimator, with `cache` holding the state
    after the observation.

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012; 0.008 -0.02; -0.02 0.03];

julia> ce = RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 2, regime_min_obs = 2);

julia> one_at_a_time = foldl((c, i) -> partial_fit!(c, view(X, i, :)), axes(X, 1); init = ce);

julia> isequal(cov(one_at_a_time), cov(ce, X))
true
```

# Related

  - [`partial_fit!(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance)`](@ref)
"""
function partial_fit!(ce::RegimeAdjustedExpWeightedCovariance, x::VecNum;
                      estimation_mask::Option{<:AbstractVector{<:Bool}} = nothing,
                      active_mask::Option{<:AbstractVector{<:Bool}} = nothing, kwargs...)
    return partial_fit!(ce, permutedims(x); dims = 1,
                        estimation_mask = if isnothing(estimation_mask)
                            nothing
                        else
                            permutedims(estimation_mask)
                        end, active_mask = if isnothing(active_mask)
                            nothing
                        else
                            permutedims(active_mask)
                        end)
end
"""
    Statistics.cov(
        ce::RegimeAdjustedExpWeightedCovariance,
        state::RegimeAdjustedCovarianceState;
        kwargs...
    ) -> MatNum

Read the regime-adjusted covariance out of a state held by hand.

This is [`regime_adjusted_covariance`](@ref) under the family's public verb, so a state a caller
keeps outside an estimator answers the same call as one the estimator holds. The state is read
and never written.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator.
  - `state`: Running state of an incremental fit.
  - $(arg_dict[:ignkwargs])

# Returns

  - $(ret_dict[:sigma])

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012];

julia> ce = partial_fit!(RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 2,
                                                             regime_min_obs = 2), X);

julia> isequal(cov(ce, ce.cache), cov(ce))
true
```

# Related

  - [`RegimeAdjustedCovarianceState`](@ref)
  - [`regime_adjusted_covariance`](@ref)
  - [`Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance)`](@ref)
"""
function Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance,
                        state::RegimeAdjustedCovarianceState; kwargs...)
    return regime_adjusted_covariance(state, ce)
end
"""
    Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance; kwargs...) -> MatNum

Read the regime-adjusted covariance out of the estimator's own state.

The one-argument form is what an incremental fit answers: [`partial_fit!`](@ref) leaves the
state in the `cache` field, and this verb turns it into the ordinary answer. An estimator that
has been given no observation carries no state, so the call is refused rather than answered with
a zero.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator carrying a state.
  - $(arg_dict[:ignkwargs])

# Validation

  - `ce.cache` is not `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - $(ret_dict[:sigma])

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012];

julia> ce = partial_fit!(RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 2,
                                                             regime_min_obs = 2), X);

julia> size(cov(ce))
(2, 2)

julia> cov(RegimeAdjustedExpWeightedCovariance())
ERROR: ArgumentError: `ce` holds no partial-fit state, so there is nothing to read. Call `partial_fit!(ce, X)` first, or `cov(ce, X)` for a fit over a whole sample.
[...]
```

# Related

  - [`partial_fit!(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
  - [`Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance, state::RegimeAdjustedCovarianceState; kwargs...)`](@ref)
  - [`RegimeAdjustedCovarianceState`](@ref)
"""
function Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance; kwargs...)
    state = ce.cache
    @argcheck(!isnothing(state),
              ArgumentError("`ce` holds no partial-fit state, so there is nothing to read. Call `partial_fit!(ce, X)` first, or `cov(ce, X)` for a fit over a whole sample."))
    return cov(ce, state)
end
"""
    Statistics.cor(
        ce::RegimeAdjustedExpWeightedCovariance,
        state::RegimeAdjustedCovarianceState;
        kwargs...
    ) -> MatNum

Read the regime-adjusted correlation out of a state held by hand.

The rescale of [`Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance, state::RegimeAdjustedCovarianceState; kwargs...)`](@ref) to a unit diagonal. The state is read and never written.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator.
  - `state`: Running state of an incremental fit.
  - $(arg_dict[:ignkwargs])

# Returns

  - $(ret_dict[:rho])

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012];

julia> ce = partial_fit!(RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 2,
                                                             regime_min_obs = 2), X);

julia> isequal(cor(ce, ce.cache), cor(ce))
true
```

# Related

  - [`RegimeAdjustedCovarianceState`](@ref)
  - [`regime_adjusted_correlation`](@ref)
  - [`Statistics.cor(ce::RegimeAdjustedExpWeightedCovariance)`](@ref)
"""
function Statistics.cor(ce::RegimeAdjustedExpWeightedCovariance,
                        state::RegimeAdjustedCovarianceState; kwargs...)
    return regime_adjusted_correlation(regime_adjusted_covariance(state, ce))
end
"""
    Statistics.cor(ce::RegimeAdjustedExpWeightedCovariance; kwargs...) -> MatNum

Read the regime-adjusted correlation out of the estimator's own state.

The one-argument form is what an incremental fit answers: [`partial_fit!`](@ref) leaves the
state in the `cache` field, and this verb turns it into the ordinary answer. An estimator that
has been given no observation carries no state, so the call is refused rather than answered with
a zero.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator carrying a state.
  - $(arg_dict[:ignkwargs])

# Validation

  - `ce.cache` is not `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - $(ret_dict[:rho])

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012];

julia> ce = partial_fit!(RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 2,
                                                             regime_min_obs = 2), X);

julia> size(cor(ce))
(2, 2)

julia> cor(RegimeAdjustedExpWeightedCovariance())
ERROR: ArgumentError: `ce` holds no partial-fit state, so there is nothing to read. Call `partial_fit!(ce, X)` first, or `cor(ce, X)` for a fit over a whole sample.
[...]
```

# Related

  - [`partial_fit!(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
  - [`Statistics.cor(ce::RegimeAdjustedExpWeightedCovariance, state::RegimeAdjustedCovarianceState; kwargs...)`](@ref)
  - [`RegimeAdjustedCovarianceState`](@ref)
"""
function Statistics.cor(ce::RegimeAdjustedExpWeightedCovariance; kwargs...)
    state = ce.cache
    @argcheck(!isnothing(state),
              ArgumentError("`ce` holds no partial-fit state, so there is nothing to read. Call `partial_fit!(ce, X)` first, or `cor(ce, X)` for a fit over a whole sample."))
    return regime_adjusted_correlation(regime_adjusted_covariance(state, ce))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a pair of regime-adjusted covariance states, because this family does not merge.

A block fitted from a cold start is not what the same block contributes after another one. The
regime statistic scores each observation against the state that stands before it, so a cold
block loses every comparison its first `min_obs` observations would have made, and a correlation
state that is normalised by a running variance carries that variance with it.

Fold the second block into the first with [`partial_fit!`](@ref) instead. A sequential fit is
exact, and it is the route this family gives.

# Arguments

  - `a`: The first state.
  - `b`: The second state.

# Returns

  - Never returns. An `ArgumentError` is thrown.

# Related

  - [`RegimeAdjustedCovarianceState`](@ref)
  - [`merge_states`](@ref)
  - [`partial_fit!(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
"""
function merge_states(a::RegimeAdjustedCovarianceState, b::RegimeAdjustedCovarianceState)
    return throw(ArgumentError("a `RegimeAdjustedCovarianceState` pair does not merge, because a block fitted from a cold start is not what the same block contributes after another one. The regime statistic scores each observation against the state that stands before it, and it is gated by the running observation count. Fold the second block into the first with `partial_fit!` instead."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies a [`RegimeAdjustedCovarianceState`](@ref), so the copy shares no array with the original.

The `copy` method of the [`AbstractPartialFitState`](@ref) interface, which [`partial_fit`](@ref)
calls before it folds. Every array field is copied, and the two scalar fields pass through. The
circular buffer of recent centred returns is rebuilt at the same capacity, and each observation
it holds is copied into it, so a fold on the copy pushes into a buffer of its own. The three
fields of the separate correlation recursion pass through as `nothing` where they are `nothing`.

# Arguments

  - `x`: The cache to copy.

# Returns

  - `state::RegimeAdjustedCovarianceState`: A fresh cache, equal to `x`, whose arrays are fresh.

# Related

  - [`RegimeAdjustedCovarianceState`](@ref)
  - [`partial_fit`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
function Base.copy(x::RegimeAdjustedCovarianceState)
    ret_buffer = if isnothing(x.ret_buffer)
        nothing
    else
        buffer = DataStructures.CircularBuffer{eltype(x.ret_buffer)}(DataStructures.capacity(x.ret_buffer))
        for X_old in x.ret_buffer
            push!(buffer, copy(X_old))
        end
        buffer
    end

    variance = isnothing(x.variance) ? nothing : copy(x.variance)
    cor_state = isnothing(x.cor_state) ? nothing : copy(x.cor_state)
    pair_obs_count = isnothing(x.pair_obs_count) ? nothing : copy(x.pair_obs_count)

    return RegimeAdjustedCovarianceState(ret_buffer, copy(x.covariance), variance,
                                         cor_state, pair_obs_count, copy(x.XXt), copy(x.Xi),
                                         copy(x.X_old_i), copy(x.location),
                                         copy(x.obs_count), copy(x.active), x.regime_state,
                                         x.n_regime_obs)
end
export RegimeAdjustedTarget, MahalanobisTarget, DiagonalTarget, PortfolioTarget,
       RegimeAdjustedExpWeightedCovariance
