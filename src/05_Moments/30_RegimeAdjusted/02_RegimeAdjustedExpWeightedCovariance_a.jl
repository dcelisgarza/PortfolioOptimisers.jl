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

The statistic reads the inverse of an estimated block, and the inverse of an estimate is too
large on average (Jensen's inequality): at 12 assets and a half-life of 10 the squared distance
of a correctly calibrated return is 1.6 times its dimension. Before the block has more than
`n + 1` observations the mean is not finite, and before it has more than `n + 3` the variance is
not finite, so one observation can hold the smoothed regime state. Where the `debias` field of
[`RegimeAdjustedExpWeightedCovariance`](@ref) is `true`, the target skips those observations and
divides the rest by the known size of the bias, so the statistic has the mean ``n`` that its
calibration functions assume.

# Mathematical definition

```math
\\begin{align}
d^{2} &= \\frac{u^{\\top} \\hat{C}^{-1} u}{b}\\,, &
\\frac{1}{b} &= \\sum_{j=0}^{K-1} \\frac{w_{j}}{1 + (n + 1)\\, w_{j}\\, b}\\,, &
w_{j} &= \\frac{(1 - \\lambda)\\, \\lambda^{j}}{1 - \\lambda^{K}}\\,.
\\end{align}
```

Where:

  - ``u``: Returns of the ``n`` contributing assets at the observation.
  - ``\\hat{C}``: Bias-corrected covariance block of those assets, from the observations before it.
  - ``K``: Smallest count of observations among the contributing assets. The update is skipped
    while ``K \\le n + 3``. The fixed point has a solution from ``K > n + 1``, where the mean is
    finite.
  - ``\\lambda``: `cor_decay` where the separate correlation path runs, else `decay`.
  - ``b``: Bias factor, ``\\mathbb{E}[\\operatorname{tr}(W^{-1})] / n`` of
    ``W = \\sum_{j} w_{j} z_{j} z_{j}^{\\top}``, ``z_{j} \\sim N(0, I_{n})``, by its deterministic
    equivalent. With equal weights it is ``K / (K - n - 1)``, the exact inverse-Wishart mean. On
    exponential weights it agrees with a Monte Carlo of ``b`` within 1 %, from 2 to 100 assets.

With `debias = false`, ``b = 1`` and every observation above `min_obs` is scored: a block that
is not positive definite takes the ridge of [`safe_regime_cholesky`](@ref). Its statistic can
then be about ``10^{12}``, and that one value holds the regime state for many half-lives.

The factor assumes one shared history of returns, and the weights of the recursion. On iid
Normal returns the squared multiplier of `RootMeanSquaredAdjusted` is then 0.95 to 1.03, with or
without the separate correlation path and the centring. ``b`` is the mean of the inverse, which is
the moment that `RootMeanSquaredAdjusted` reads. `FirstMomentRegimeAdjusted` reads the mean of
the root and `LogRegimeAdjusted` the mean of the log, which need smaller factors, so the target
over-corrects them: the squared multiplier is 0.975 and 0.956 at 12 assets and a half-life of 10.
A HAC estimate reads the fixed point on the spectrum of its banded weight matrix, and skips a
block with too few effective observations: at two lags, 12 assets and a half-life of 10 the
squared multiplier of `RootMeanSquaredAdjusted` is 0.981 over 8 seeds, from 2.48 raw, which is the
error of the deterministic equivalent on an estimate with half the degrees of freedom. On the
separate correlation path the factor reads `cor_decay` alone, and the noise of the variance at
`decay` leaves 1.031 without HAC and 1.090 at two lags.

# Related

  - [`RegimeAdjustedTarget`](@ref)
  - [`DiagonalTarget`](@ref)
  - [`PortfolioTarget`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`mahalanobis_bias`](@ref)
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

Targets the diagonal of a covariance matrix, in a regime adjustment and in a geodesic shrinkage.

In a regime adjustment, the baseline covariance structure is diagonal, so the regime statistic reads the variances alone. In a [`GeodesicShrinkageCovariance`](@ref), the target matrix is the diagonal of the matrix being shrunk, which keeps the variances and removes every correlation.

Each term of the regime statistic reads one estimated variance, and the inverse of an estimate is too large on average. Where the estimator has `debias = true`, [`regime_target_statistic`](@ref) divides each term by the mean of that inverse at the count of its asset, so the statistic has the mean ``n`` at every correlation. The root and the log of the statistic read its law, which the correlation of the assets sets, so for the first-moment and the log methods it then divides the sum by [`diagonal_law_factor`](@ref).

# Related

  - [`RegimeAdjustedTarget`](@ref)
  - [`MahalanobisTarget`](@ref)
  - [`PortfolioTarget`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`GeodesicShrinkageCovariance`](@ref)
  - [`AbstractCovarianceShrinkageTarget`](@ref): the other target rules of a geodesic shrinkage.
"""
struct DiagonalTarget <: RegimeAdjustedTarget end
"""
$(DocStringExtensions.TYPEDEF)

Regime-adjustment target that uses a portfolio-weighted baseline covariance structure.

Each direction reads one estimated variance, and the inverse of an estimate is too large on
average. Where the estimator has `debias = true`, [`regime_target_statistic`](@ref) divides each
direction by the bias that the regime method reads. The factor is exact for fixed weights. The
inverse-volatility direction of `w = nothing` is built from the same estimate, so a smaller bias
remains, which that function states.

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

This estimator is mask-aware, so a prior fitted with it keeps a young asset investable and
zero-fills the rows the asset was missing through [`scenario_fill`](@ref): every consumer of a
Prior Result reads its returns matrix, and a scenario-based measure then reads a zero return
where the asset had none and understates that asset's risk over those rows, while the covariance
stays the estimate this recursion made from the rows it saw. The fill is silent at or below
the fitting prior's own `fill_limit` field, a share of that asset's own observations, warns
above it, and refuses any fill under `strict`; `fill_limit` defaults to `nothing`, and this
family carries no `CoveragePolicy` to derive a limit from, so every fill is named.

A `regime_method` of `nothing` turns the adjustment off: no regime state advances, so the
multiplier stays at one and the estimator is the plain exponentially weighted recursion.

The statistic divides a realised square by an estimated variance, and the inverse of an estimate
is too large on average (Jensen's inequality). With `debias = true`, the default, the statistic
skips an estimate too young for a finite variance and divides the rest by the known size of the
bias, as each target states (ADR 0190). `debias = false` scores the raw statistic.

# Mathematical definition

Write ``\\lambda`` for `decay` and ``\\lambda_c`` for `cor_decay`. Where `cor_decay` is
`nothing`, one recursion carries the whole matrix:

```math
\\begin{align}
S_{ij,t} &= \\lambda S_{ij,t-1} + (1-\\lambda) u_{i,t} u_{j,t}\\,, \\\\
W_{ij,t} &= \\lambda W_{ij,t-1} + (1-\\lambda)\\,,
\\end{align}
```

on the pairs of which both assets are valid at ``t``. Every other entry holds.

Where:

  - ``S_{ij,t}``: Raw exponentially weighted covariance state at time ``t``, seeded at zero.
  - ``W_{ij,t}``: Weight that the pair holds in ``S``, seeded at zero. It is
    ``1 - \\lambda^{n_{ij}}``, with ``n_{ij}`` the count of the common valid observations of the
    pair.
  - ``u_{i,t}``: Observation ``t`` of asset ``i``, centred where `centred` is `false`, and
    HAC-adjusted where `hac_lags` is not `nothing`.

Where `cor_decay` is not `nothing`, the variance and the correlation run at their own decays and
are recombined:

```math
\\begin{align}
v_{i,t} &= \\lambda v_{i,t-1} + (1-\\lambda) u_{i,t}^{2}\\,, \\\\
Q_{ij,t} &= \\lambda_c Q_{ij,t-1} + (1-\\lambda_c) \\frac{u_{i,t} u_{j,t}}{\\sqrt{v_{i,t} v_{j,t}}}\\,, \\\\
W_{ij,t} &= \\lambda_c W_{ij,t-1} + (1-\\lambda_c)\\,, \\\\
\\rho_{ij,t} &= \\frac{Q_{ij,t} / W_{ij,t}}{\\sqrt{(Q_{ii,t} / W_{ii,t})(Q_{jj,t} / W_{jj,t})}}\\,.
\\end{align}
```

Where:

  - ``v_{i,t}``: Raw exponentially weighted variance of asset ``i``.
  - ``Q_{ij,t}``: Raw exponentially weighted correlation state. Like ``S``, it steps only on
    the pairs of which both assets are valid, so a holiday of asset ``i`` holds every entry of
    ``i``.
  - ``W_{ij,t}``: Weight that the pair holds in ``Q``, with the step of ``Q`` on a unit
    product: ``1 - \\lambda_c^{n_{ij}}``.
  - ``\\rho_{ij,t}``: Correlation, the weighted mean of the standardised product over the
    common observations of the pair, over the root of the two weighted means of the squares.

A zero seed damps the state, so [`bias_corrected_covariance`](@ref) divides each pair by its
weight before it reports:

```math
\\begin{align}
\\hat{\\Sigma}_{ij} &= \\mathrm{mult}(s_T)^{2}\\, \\frac{S_{ij,T}}{W_{ij,T}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:Sigma_hat])
  - ``\\mathrm{mult}(s_T)^{2}\\, S_{ij,T} / W_{ij,T}``: The entry where `cor_decay` is
    `nothing`. On the separate path it is ``\\mathrm{mult}(s_T)^{2}\\, \\rho_{ij,T} \\sqrt{\\hat{v}_{i} \\hat{v}_{j}}``, with ``\\hat{v}_{i} = v_{i,T} / (1 - \\lambda^{n_i})``.
  - ``\\mathrm{mult}(s_T)``: Regime multiplier of the smoothed regime state ``s_T``, clamped to
    `regime_lohi_mult` where that field is not `nothing`.

Each entry reads its own common observations, so where the assets do not share one history, a
holiday or a late listing, the matrix need not be positive semidefinite. Where the smallest
eigenvalue of its correlation is below ``-n\\,\\varepsilon`` times its largest, the report clips
the negative eigenvalues to zero, restores the unit diagonal and keeps the variances. Where every
pair shares one history, the division is a congruence of a sum of outer products, so the matrix
is positive semidefinite and the repair changes nothing. A congruence on different histories
would shrink each correlation of the pair towards zero, which ADR 0181 measures.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RegimeAdjustedExpWeightedCovariance(;
        decay::Number                                         = exp2(-inv(40.0)),
        cor_decay::Option{<:Number}                           = nothing,
        min_obs::Integer                                      = round(Int, max(1, decay_half_life(decay), isnothing(cor_decay) ? 1 : decay_half_life(cor_decay, :cor_decay))),
        hac_lags::Option{<:Integer}                           = nothing,
        hac_floor::Bool                                       = false,
        regime_method::Option{<:RegimeAdjustedMethod}         = FirstMomentRegimeAdjusted(),
        regime_decay::Number                                  = exp2(-2 / decay_half_life(decay)),
        regime_min_obs::Integer                               = round(Int, max(1, decay_half_life(decay) / 2)),
        regime_target::RegimeAdjustedTarget                   = PortfolioTarget(),
        regime_lohi_mult::Option{<:Tuple{<:Number, <:Number}} = (0.7, 1.6),
        min_val::Number                                       = 1e-12,
        centred::Bool                                         = false,
        debias::Bool                                          = true,
        cache::Option{<:AbstractPartialFitState}              = nothing
    ) -> RegimeAdjustedExpWeightedCovariance

Keywords correspond to the struct's fields. Where `cor_decay` is not `nothing`, the default
`min_obs` reads the slower of the two decays.

## Validation

  - $(val_dict[:decay])
  - If `cor_decay` is not `nothing`, `0 < cor_decay < 1`.
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
  - [`scenario_fill`](@ref)
  - [`EmpiricalPrior`](@ref)
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
    $(field_dict[:hac_floor])
    """
    hac_floor
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
    $(field_dict[:ra_debias])
    """
    debias
    """
    Running state of an incremental fit, or `nothing` before the first call to [`partial_fit!`](@ref). It is the one Result this estimator holds, and its type bound is the enforcement of that exception. [`Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance)`](@ref) reads it, and a fit over a matrix ignores it.
    """
    cache
    function RegimeAdjustedExpWeightedCovariance(decay::Number, cor_decay::Option{<:Number},
                                                 min_obs::Integer,
                                                 hac_lags::Option{<:Integer},
                                                 hac_floor::Bool,
                                                 regime_method::Option{<:RegimeAdjustedMethod},
                                                 regime_decay::Number,
                                                 regime_min_obs::Integer,
                                                 regime_target::RegimeAdjustedTarget,
                                                 regime_lohi_mult::Option{<:Tuple{<:Number,
                                                                                  <:Number}},
                                                 min_val::Number, centred::Bool,
                                                 debias::Bool,
                                                 cache::Option{<:AbstractPartialFitState})
        assert_unit_interval(decay, :decay)
        if !isnothing(cor_decay)
            assert_unit_interval(cor_decay, :cor_decay)
        end
        assert_nonempty_gt0_finite_val(min_obs, :min_obs)
        assert_nonempty_gt0_finite_val(regime_min_obs, :regime_min_obs)
        if !isnothing(regime_lohi_mult)
            @argcheck(zero(regime_lohi_mult[1]) < regime_lohi_mult[1] < regime_lohi_mult[2],
                      DomainError(regime_lohi_mult,
                                  "`RegimeAdjustedExpWeightedCovariance.regime_lohi_mult` is $regime_lohi_mult, and it clamps the regime multiplier to `(lo, hi)`, so the pair must satisfy `0 < lo < hi`. State such a pair, or `nothing` for no clamp."))
        end
        if !isnothing(hac_lags)
            assert_nonempty_gt0_finite_val(hac_lags, :hac_lags)
        end
        return new{typeof(decay), typeof(cor_decay), typeof(min_obs), typeof(hac_lags),
                   typeof(hac_floor), typeof(regime_method), typeof(regime_decay),
                   typeof(regime_min_obs), typeof(regime_target), typeof(regime_lohi_mult),
                   typeof(min_val), typeof(centred), typeof(debias), typeof(cache)}(decay,
                                                                                    cor_decay,
                                                                                    min_obs,
                                                                                    hac_lags,
                                                                                    hac_floor,
                                                                                    regime_method,
                                                                                    regime_decay,
                                                                                    regime_min_obs,
                                                                                    regime_target,
                                                                                    regime_lohi_mult,
                                                                                    min_val,
                                                                                    centred,
                                                                                    debias,
                                                                                    cache)
    end
end
function RegimeAdjustedExpWeightedCovariance(; decay::Number = exp2(-inv(40.0)),
                                             cor_decay::Option{<:Number} = nothing,
                                             min_obs::Integer = round(Int,
                                                                      max(1,
                                                                          decay_half_life(decay),
                                                                          if isnothing(cor_decay)
                                                                              1
                                                                          else
                                                                              decay_half_life(cor_decay,
                                                                                              :cor_decay)
                                                                          end)),
                                             hac_lags::Option{<:Integer} = nothing,
                                             hac_floor::Bool = false,
                                             regime_method::Option{<:RegimeAdjustedMethod} = FirstMomentRegimeAdjusted(),
                                             regime_decay::Number = exp2(-2 /
                                                                         decay_half_life(decay)),
                                             regime_min_obs::Integer = round(Int,
                                                                             max(1,
                                                                                 decay_half_life(decay) /
                                                                                 2)),
                                             regime_target::RegimeAdjustedTarget = PortfolioTarget(),
                                             regime_lohi_mult::Option{<:Tuple{<:Number,
                                                                              <:Number}} = (0.7,
                                                                                            1.6),
                                             min_val::Number = 1e-12, centred::Bool = false,
                                             debias::Bool = true,
                                             cache::Option{<:AbstractPartialFitState} = nothing)::RegimeAdjustedExpWeightedCovariance
    return RegimeAdjustedExpWeightedCovariance(decay, cor_decay, min_obs, hac_lags,
                                               hac_floor, regime_method, regime_decay,
                                               regime_min_obs, regime_target,
                                               regime_lohi_mult, min_val, centred, debias,
                                               cache)
end
"""
$(DocStringExtensions.TYPEDEF)

Internal mutable cache for the online covariance update in [`RegimeAdjustedExpWeightedCovariance`](@ref).

This type is an implementation detail and is not intended for direct use.

The three fields that carry the separate correlation recursion are `nothing` where `cor_decay`
is `nothing`, because one decay then carries the whole matrix and no correlation state exists.
The field `weight` is `nothing` where `cor_decay` opens the separate path, because the covariance
is then rebuilt from the variance and the correlation at each observation.

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
    $(field_dict[:ra_weight])
    """
    weight
    """
    $(field_dict[:ra_variance])
    """
    variance
    """
    $(field_dict[:ra_cor_state])
    """
    cor_state
    """
    $(field_dict[:ra_cor_weight])
    """
    cor_weight
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
    """
    $(field_dict[:ra_bias])
    """
    bias
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
expression at `n = 1`. The squares of the diagonal target are correlated, so its sum is not a
``\\chi^2(n)`` variate: the debiased statistic is first divided by
[`diagonal_law_factor`](@ref), which makes this constant exact at the correlation of the assets.

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

The root of `n` is the root of the mean of the statistic, not the mean of its root, so it is not
the expectation of the root at any correlation. The raw statistic of `debias = false` divides by
it. The debiased statistic is first divided by [`diagonal_law_factor`](@ref), which makes this
constant exact at the correlation of the assets.

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
    return sqrt(n)
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

Returns the correlation of the separate path on a block of assets, each pair normalised by the
weight it holds.

The correlation state ``Q`` and the weight state ``W`` step only on the pairs of which both
assets are valid, ``W`` on a unit product, so ``Q_{ij} / W_{ij}`` is the weighted mean of the
standardised product over the common observations of the pair. The correlation divides it by the
root of the two weighted means of the squares, each over the observations of its own asset. An
asset that lists late, or that has a holiday, holds fewer observations than an asset that did
not, so its pairs hold less weight than the product of the two diagonal weights. A normalisation of
``Q`` alone divides by that product, and it shrinks the correlation towards zero. Where every
pair shares one history, ``W`` is one scalar on every entry, and the two normalisations agree.

A matrix of pairs, each normalised by its own weight, need not be positive semidefinite. The
function makes no repair: [`regime_adjusted_covariance`](@ref) restores the report with
[`restore_psd!`](@ref), and the regime statistic reads the block as it is.

# Arguments

  - `cache::RegimeAdjustedCovarianceState`: Online covariance computation cache.
  - `idx::AbstractVector{<:Integer}`: Index of the assets of the block.
  - `min_val::Number`: Floor applied to each diagonal mean before its root is taken.

# Returns

  - `rho::MatNum`: The correlation block, with a unit diagonal.

# Related

  - [`RegimeAdjustedCovarianceState`](@ref)
  - [`pair_weighted_block`](@ref)
  - [`restore_psd!`](@ref)
  - [`update_var_cor!`](@ref)
  - [`bias_corrected_covariance`](@ref)
"""
function pair_weighted_correlation(cache::RegimeAdjustedCovarianceState,
                                   idx::AbstractVector{<:Integer}, min_val::Number)
    C = pair_weighted_block(cache.cor_state, cache.cor_weight, idx)
    T = eltype(C)
    inv_d = inv.(sqrt.(clamp.(LinearAlgebra.diag(C), min_val, T(Inf))))
    rho = clamp.(C .* (inv_d .* transpose(inv_d)), -one(T), one(T))
    rho .= (rho + transpose(rho)) / 2
    rho[LinearAlgebra.diagind(rho)] .= one(T)
    return rho
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
 3. On the pairs of which both assets are valid, advance the correlation state by the step
    ``Q_{ij} \\leftarrow \\lambda_c Q_{ij} + (1 - \\lambda_c) \\Delta_{ij}``, where ``\\Delta`` is
    the standardised outer product, and the weight state by the same step with ``\\Delta_{ij}``
    one. Every other entry holds, so a holiday holds every entry of its asset.
 4. Normalise each pair of the active block by its weight with [`pair_weighted_correlation`](@ref),
    and rescale the correlation by the running volatilities into `cache.covariance`.

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
    hac_var = LinearAlgebra.diag(cache.XXt)
    if ce.hac_floor
        hac_var .= max.(hac_var, zero(T))
    end
    cache.variance[valid] .= ce.decay * view(cache.variance, valid) +
                             (one(ce.decay) - ce.decay) * view(hac_var, valid)
    positive = valid .& (cache.variance .> ce.min_val)
    inv_sigma = ifelse.(positive, inv.(sqrt.(ifelse.(positive, cache.variance, one(T)))),
                        zero(T))
    outer_std = cache.XXt .* (inv_sigma .* transpose(inv_sigma))
    # Each pair ages on its common observations, and a holiday holds it (ADR 0181).
    step = one(ce.cor_decay) - ce.cor_decay
    cache.cor_state .= ifelse.(pair_valid,
                               ce.cor_decay .* cache.cor_state .+ step .* outer_std,
                               cache.cor_state)
    cache.cor_weight .= ifelse.(pair_valid, ce.cor_decay .* cache.cor_weight .+ step,
                                cache.cor_weight)

    active = cache.active .& (cache.variance .> zero(T))
    if !any(active)
        return nothing
    end
    idx = findall(active)
    if !all(>(ce.min_val), view(LinearAlgebra.diag(cache.cor_state), idx))
        return nothing
    end
    # The regime statistic reads this block at every observation, and its Cholesky factor
    # refuses a block that is not positive definite, so the block takes no repair.
    rho = pair_weighted_correlation(cache, idx, ce.min_val)
    sigma = sqrt.(view(cache.variance, idx))
    cache.covariance[idx, idx] = rho .* sigma .* transpose(sigma)

    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Removes the damping a zero seed leaves in the running state, and returns the covariance that
[`regime_adjusted_covariance`](@ref) reports.

The recursion is seeded at zero, and it damps each pair by the weight ``W_{ij}`` that it holds.
Where `cor_decay` is `nothing`, each entry of the covariance state is divided by its weight with
[`pair_weighted_block`](@ref). Where `cor_decay` opens the separate path, the variance is
corrected at `decay`, and each pair of the correlation state is divided by its weight with
[`pair_weighted_correlation`](@ref). A holiday or a late listing leaves a pair less weight than a
congruence assumes, so the division is exact where a congruence would shrink the correlation.
The division can break positive semidefiniteness. [`regime_adjusted_covariance`](@ref) restores
it on the block of the ready assets, so an asset in its warm-up moves nothing.

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
        sigma = fill(T(NaN), N, N)
        act = findall(cache.active)
        if !isempty(act)
            sigma[act, act] = pair_weighted_block(cache.covariance, cache.weight, act)
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
    # The report divides by no volatility, so a variance of zero stays zero; only round-off
    # below zero is removed.
    vol = sqrt.(max.(var_bc, zero(T)))
    rho = pair_weighted_correlation(cache, idx, ce.min_val)
    sigma[idx, idx] = rho .* vol .* transpose(vol)

    return sigma
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the bias-corrected covariance block of the assets that contribute to the regime
statistic.

This block calibrates the smoother and is never reported, so it takes no repair: the Cholesky
factor of the Mahalanobis target refuses a block that is not positive definite. Where one decay
runs, each entry is divided by the weight its pair holds with [`pair_weighted_block`](@ref). Where
the separate path runs, the correlation already sits inside `cache.covariance`, each pair
normalised by its weight, and the block applies the per-asset variance correction alone.
[`bias_corrected_covariance`](@ref) makes the reported estimate from the state.

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
    if !has_separate_cor_decay(ce)
        return pair_weighted_block(cache.covariance, cache.weight, idx)
    end
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
  - [`regime_target_statistic`](@ref)
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
    regime_mask = valid .& cache.active .& (cache.obs_count .>= ce.min_obs) .&
                  regime_bias_open.(ce.debias, 1, ce.decay, cache.obs_count, ce.hac_lags)
    if !isnothing(estimation_mask)
        regime_mask .&= estimation_mask
    end
    n = count(regime_mask)
    if n < min_active_assets(ce.regime_target)
        return cache
    end
    idx = findall(regime_mask)
    stats = regime_target_statistic(ce.regime_target, cache, ce, X, idx)
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
then advances the covariance itself. On the path with one decay, the covariance takes the step
``S_{ij} \\leftarrow \\lambda S_{ij} + (1 - \\lambda) \\Delta_{ij}`` on the pairs of which both
assets are valid, where ``\\Delta`` is the outer product, and the weight state takes the same step
with ``\\Delta_{ij}`` one. Every other entry holds, so a holiday holds every entry of its asset, as
in [`ExpWeightedCovariance`](@ref). The path with a separate `cor_decay` takes the same step on
its correlation state, at ``\\lambda_c``, in [`update_var_cor!`](@ref). An asset that turns
inactive at this observation has its rows and columns zeroed and its count reset, so a later
listing starts from a cold state.

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
        # Each pair ages on its common observations, and a holiday holds it (ADR 0181).
        step = one(ce.decay) - ce.decay
        cache.covariance .= ifelse.(pair_valid,
                                    ce.decay .* cache.covariance .+ step .* cache.XXt,
                                    cache.covariance)
        cache.weight .= ifelse.(pair_valid, ce.decay .* cache.weight .+ step, cache.weight)
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
        if !isnothing(cache.weight)
            cache.weight[newly_inactive, :] .= zero(T)
            cache.weight[:, newly_inactive] .= zero(T)
        end
        if has_separate_cor_decay(ce)
            cache.variance[newly_inactive] .= zero(T)
            cache.cor_state[newly_inactive, :] .= zero(T)
            cache.cor_state[:, newly_inactive] .= zero(T)
            cache.cor_weight[newly_inactive, :] .= zero(T)
            cache.cor_weight[:, newly_inactive] .= zero(T)
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
    assert_nonneg(W, "regime_target.w")
    assert_gt0(vec(sum(W; dims = 2)), "sum(regime_target.w; dims = 2)")

    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Run one forward pass of the online covariance update over the observations of `X`, and call `f`
after each observation.

The pass owns the argument validation, the orientation and the cache, so every verb that reads
it states the recursion once.

# Arguments

  - `f`: Called as `f(i, cache)` after observation `i`. A caller that wants the final cache
    alone reads the method that takes no `f`.
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

    # The state holds a covariance and a `NaN`, so an integer panel computes in its floating
    # point type, and every other type is kept: a `Float32` panel keeps a `Float32` state.
    Tf = float_if_integer(eltype(X))
    # An uncentred estimator seeds its location from the first observation it sees, so the
    # location starts as `NaN`.
    location = ce.centred ? zeros(Tf, N) : fill(convert(Tf, NaN), N)
    cache = if isnothing(state)
        separate = has_separate_cor_decay(ce)
        RegimeAdjustedCovarianceState(if isnothing(ce.hac_lags)
                                          nothing
                                      else
                                          DataStructures.CircularBuffer{Vector{Tf}}(ce.hac_lags)
                                      end, zeros(Tf, N, N),
                                      separate ? nothing : zeros(Tf, N, N),
                                      separate ? zeros(Tf, N) : nothing,
                                      separate ? zeros(Tf, N, N) : nothing,
                                      separate ? zeros(Tf, N, N) : nothing, zeros(Tf, N, N),
                                      zeros(Tf, N), zeros(Tf, N), location, zeros(Int, N),
                                      trues(N), nothing, 0, regime_bias_state(ce, Tf))
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

Run one forward pass of the online covariance update over the observations of `X`, and read no
intermediate cache.

This is the callback method with a callback that does nothing, so a verb that wants the last
cache alone states no callback of its own. `cov` and `partial_fit!` read the pass this way, and
`variance_series` reads the callback method.

# Arguments

  - `ce::RegimeAdjustedExpWeightedCovariance`: Covariance estimator configuration.
  - `X::MatNum`: Observation matrix.
  - `dims::Int`: Dimension along which the observations lie.
  - `estimation_mask::Option{<:AbstractMatrix{<:Bool}}`: Optional mask restricting which assets
    contribute to the regime state update.
  - `active_mask::Option{<:AbstractMatrix{<:Bool}}`: Optional mask of active assets.
  - `state::Option{<:RegimeAdjustedCovarianceState}`: Optional state to continue. A `nothing`
    starts a cold cache, which is what a fit over a whole sample needs. [`partial_fit!`](@ref)
    passes the estimator's own state.

# Returns

  - `cache::RegimeAdjustedCovarianceState`: The cache after the last observation.

# Related

  - [`RegimeAdjustedCovarianceState`](@ref)
  - [`regime_adjusted_covariance_pass!`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
"""
function regime_adjusted_covariance_pass!(ce::RegimeAdjustedExpWeightedCovariance,
                                          X::MatNum, dims::Int,
                                          estimation_mask::Option{<:AbstractMatrix{<:Bool}},
                                          active_mask::Option{<:AbstractMatrix{<:Bool}},
                                          state::Option{<:RegimeAdjustedCovarianceState} = nothing)
    return regime_adjusted_covariance_pass!((args...) -> nothing, ce, X, dims,
                                            estimation_mask, active_mask, state)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Read the regime-adjusted covariance out of a cache, as it stands.

Removes the damping of the zero seed, blanks every asset that is not ready, symmetrises the
block that is, and scales by the square of the regime multiplier. Where `repair` is `true`, it
restores a positive semidefinite block of the ready assets with [`restore_psd!`](@ref), which
keeps the diagonal. The cache is read, never written, so the same cache answers this call after
every observation of a forward pass.

Where `ce.regime_method` is `nothing`, [`process_observation!`](@ref) advances no regime state,
so `cache.n_regime_obs` stays at zero, which is below every admissible `regime_min_obs`. The
multiplier is then one and the covariance is the plain recursion.

Where `ce.regime_lohi_mult` is not `nothing`, the multiplier is clamped to that `(lo, hi)` range
before it is squared. Where it is `nothing`, no clamp runs. The clamp bounds an estimate, so the
multiplier of one before `regime_min_obs` holds even where `lo > 1` or `hi < 1`.

# Arguments

  - `cache::RegimeAdjustedCovarianceState`: Online covariance computation cache.
  - `ce::RegimeAdjustedExpWeightedCovariance`: Covariance estimator configuration.
  - `repair::Bool`: Whether to restore a positive semidefinite matrix. A caller that reads the
    diagonal alone passes `false`.

# Returns

  - `sigma::MatNum`: Regime-adjusted covariance matrix. An asset with fewer than `ce.min_obs`
    observations, or one that is not active, is `NaN` in its own row and column.

# Related

  - [`RegimeAdjustedCovarianceState`](@ref)
  - [`bias_corrected_covariance`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
"""
function regime_adjusted_covariance(cache::RegimeAdjustedCovarianceState,
                                    ce::RegimeAdjustedExpWeightedCovariance;
                                    repair::Bool = true)
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
        block .= (block .+ transpose(block)) ./ 2
        if repair
            # The repair reads the ready block alone, so an asset in its warm-up moves nothing.
            restore_psd!(block)
        end
        sigma[idx, idx] = block
    end

    # The clamp bounds an estimate; the warm-up factor of one is no estimate, so it holds.
    factor = if cache.n_regime_obs < ce.regime_min_obs
        one(T)
    elseif isnothing(ce.regime_lohi_mult)
        regime_multiplier(ce.regime_method, cache.regime_state)
    else
        clamp(regime_multiplier(ce.regime_method, cache.regime_state),
              ce.regime_lohi_mult[1], ce.regime_lohi_mult[2])
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

export RegimeAdjustedTarget, MahalanobisTarget, DiagonalTarget, PortfolioTarget,
       RegimeAdjustedExpWeightedCovariance
public min_active_assets
