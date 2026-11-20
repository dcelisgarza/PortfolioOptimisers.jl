"""
    ew_variance_estimator(half_life::Real,
                          warm_up::Real = half_life) -> RegimeAdjustedExpWeightedVariance

Build the exponentially weighted variance estimator an [`EWVolatility`](@ref) reads by default.

The estimator is the plain recursion. It is uncentred, it divides by `1 - λ^n` to correct the bias, it restarts an asset that turns inactive, and it applies **no regime adjustment**. `centring = PreCentred()` gives the uncentred form, because it declares the returns already centred, so the estimator tracks no location. `regime_method = nothing` turns off the regime multiplier of [`RegimeAdjustedExpWeightedVariance`](@ref), because a volatility Descriptor is not scaled by a regime.

# Mathematical definition

```math
\\begin{align}
\\lambda &= 2^{-1/h}\\,, \\\\
S_{t,i} &= \\lambda S_{t-1,i} + (1 - \\lambda) y_{t,i}^2\\,, \\quad S_{0,i} = 0\\,, \\\\
\\hat{\\sigma}_{t,i}^2 &= \\begin{cases} S_{t,i} / \\left(1 - \\lambda^{n_i}\\right) & \\text{if } n_i \\geq m \\\\ \\mathrm{NaN} & \\text{otherwise} \\end{cases}\\,, \\\\
m &= \\max\\left(1, \\lceil h_w \\rceil\\right)\\,.
\\end{align}
```

The state of asset ``i`` moves only at an observation where the asset is active and ``y_{t,i}`` is finite. At every other observation it holds, ``S_{t,i} = S_{t-1,i}``. Where the asset turns inactive, ``S_{t,i}`` and ``n_i`` restart from zero, and the variance of an inactive asset is `NaN`. The divisor ``1 - \\lambda^{n_i}`` is the total weight of the ``n_i`` observations, so a constant input ``y`` gives ``\\hat{\\sigma}_{t,i}^2 = y^2`` from the first observation.

Where:

  - $(math_dict[:lambda_ew])
  - ``h``: `half_life`, the half-life of the recursion in observations.
  - ``h_w``: `warm_up`, the half-life the warm-up is taken from.
  - ``S_{t,i}``: State of the recursion of asset ``i`` after observation ``t``.
  - ``y_{t,i}``: Input of the recursion for asset ``i`` at observation ``t``.
  - $(math_dict[:n_i_ew])
  - ``m``: `min_obs`, the warm-up in valid observations.
  - ``\\hat{\\sigma}_{t,i}^2``: Variance of asset ``i`` after observation ``t``.

# Arguments

  - `half_life`: The half-life of the recursion, in observations.
  - `warm_up`: The half-life the warm-up is taken from. It is the half-life of the recursion itself, except where a Descriptor's estimate is held back by a second recursion of its own, as [`EWResidualVolatility`](@ref) is held back by its beta.

# Validation

  - `0 < half_life < Inf`. Raises a `DomainError` that names `half_life`.
  - `0 < warm_up < Inf`. Raises a `DomainError` that names `warm_up`.

# Returns

  - `ce::RegimeAdjustedExpWeightedVariance`: The estimator, with `half_life` converted through [`half_life_decay`](@ref) and `warm_up` through [`half_life_min_obs`](@ref).

# Examples

```jldoctest
julia> ce = PortfolioOptimisers.ew_variance_estimator(5.0);

julia> (ce.decay ≈ exp2(-inv(5.0)), ce.min_obs, ce.centring, ce.regime_method)
(true, 5, PreCentred()
, nothing)

julia> PortfolioOptimisers.ew_variance_estimator(5.0, 8.0).min_obs
8
```

# Related

  - [`EWVolatility`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
  - [`variance_series`](@ref)
"""
function ew_variance_estimator(half_life::Real,
                               warm_up::Real = half_life)::RegimeAdjustedExpWeightedVariance
    return RegimeAdjustedExpWeightedVariance(; decay = half_life_decay(half_life),
                                             min_obs = half_life_min_obs(warm_up, :warm_up),
                                             centring = PreCentred(),
                                             regime_method = nothing)
end
"""
$(DocStringExtensions.TYPEDEF)

Exponentially weighted volatility of the returns, at every observation.

This is the archetype of every volatility Descriptor. It takes the square root of the [`variance_series`](@ref) of the variance estimator in its `ce` field. That estimator carries the recursion, the bias correction and the reset of an asset that turns inactive, so this type repeats none of them. The `alg` field selects the downside form, as it does in [`Covariance`](@ref), so the downside form needs no struct of its own.

# Mathematical definition

```math
\\begin{align}
y_{t,i} &= \\begin{cases} \\min(x_{t,\\,i} - \\mathrm{mar},\\, 0) & \\text{if } \\texttt{alg} \\text{ is a } \\mathrm{SemiMoment} \\\\ x_{t,\\,i} & \\text{otherwise} \\end{cases}\\,, \\\\
d_{t,i} &= \\sqrt{\\hat{\\sigma}_{t,i}^2}\\,.
\\end{align}
```

Where:

  - ``d_{t,i}``: Descriptor of asset ``i`` at observation ``t``.
  - $(math_dict[:x_ti_ret])
  - ``y_{t,i}``: Input the variance estimator reads for asset ``i`` at observation ``t``.
  - ``\\hat{\\sigma}_{t,i}^2``: Variance of asset ``i`` after observation ``t``, as `ce` estimates it from the inputs ``y`` up to ``t``. The default `ce` is the recursion that [`ew_variance_estimator`](@ref) states.
  - ``\\mathrm{mar}``: `mar`, the minimum acceptable return.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EWVolatility(; half_life::Real = 40.0,
                 ce::AbstractCovarianceEstimator = ew_variance_estimator(half_life),
                 alg::AbstractMomentAlgorithm = FullMoment(),
                 mar::Real = 0.0,
                 cache::Option{<:AbstractPartialFitState} = nothing) -> EWVolatility

`ce`, `alg`, `mar` and `cache` correspond to the struct's fields. `half_life` is not a field. It fixes the default of `ce` through [`ew_variance_estimator`](@ref), and the constructor keeps a `ce` that the caller passes as it is. The default half-life of `40` weights about as far back as a window of two months of daily observations.

## Validation

  - `isfinite(mar)`.
  - `0 < half_life < Inf`, where it fixes the default of `ce`. Raises a `DomainError`.

# Examples

```jldoctest
julia> de = EWVolatility(; half_life = 5);

julia> (de.ce.decay ≈ exp2(-inv(5.0)), isa(de.alg, FullMoment), de.mar)
(true, true, 0.0)
```

# Related

  - [`AbstractDescriptorEstimator`](@ref)
  - [`descriptor`](@ref)
  - [`EWDownsideVolatility`](@ref)
  - [`ew_variance_estimator`](@ref)
  - [`variance_series`](@ref)
  - [`EWMean`](@ref)
"""
@concrete struct EWVolatility <: AbstractDescriptorEstimator
    """
    Variance estimator whose [`variance_series`](@ref) the Descriptor takes the square root of. It carries the recursion, the warm-up and the reset of an asset that turns inactive. A [`PortfolioOptimisersCovariance`](@ref) with a step in its `mp`, as its default `mp` holds, raises on this series, as its docstring states.
    """
    ce
    """
    $(field_dict[:malg])
    """
    alg
    """
    Minimum acceptable return that the downside form measures the returns against. The Descriptor reads it only where `alg` is a [`SemiMoment`](@ref), which clips every excess above it to zero.
    """
    mar
    """
    $(field_dict[:ew_vol_desc_cache])
    """
    cache
    function EWVolatility(ce::AbstractCovarianceEstimator, alg::AbstractMomentAlgorithm,
                          mar::Real, cache::Option{<:AbstractPartialFitState})
        assert_finite(mar, :mar)
        return new{typeof(ce), typeof(alg), typeof(mar), typeof(cache)}(ce, alg, mar, cache)
    end
end
function EWVolatility(; half_life::Real = 40.0,
                      ce::AbstractCovarianceEstimator = ew_variance_estimator(half_life),
                      alg::AbstractMomentAlgorithm = FullMoment(), mar::Real = 0.0,
                      cache::Option{<:AbstractPartialFitState} = nothing)::EWVolatility
    return EWVolatility(ce, alg, mar, cache)
end
"""
    ew_volatility_input(alg::FullMoment, X::AbstractMatrix{<:Number},
                        mar::Number) -> AbstractMatrix
    ew_volatility_input(alg::SemiMoment, X::AbstractMatrix{<:Real},
                        mar::Real) -> AbstractMatrix{<:Real}

Transform the returns an [`EWVolatility`](@ref) measures the variance of.

The moment Algorithm chooses the transformation, so the variance estimator sees one matrix and needs no downside form of its own.

# Arguments

  - `alg`: Moment Algorithm.
  - `X`: The returns, `observations × assets`.
  - `mar`: The minimum acceptable return.

# Returns

  - `Y::AbstractMatrix{<:Real}`: The returns for a [`FullMoment`](@ref), and `min.(X .- mar, 0)` for a [`SemiMoment`](@ref).

# Examples

```jldoctest
julia> PortfolioOptimisers.ew_volatility_input(SemiMoment(), [0.1 -0.2], 0.0)
1×2 Matrix{Float64}:
 0.0  -0.2
```

# Related

  - [`EWVolatility`](@ref)
  - [`EWDownsideVolatility`](@ref)
  - [`descriptor`](@ref)
"""
function ew_volatility_input(::FullMoment, X::AbstractMatrix{<:Number},
                             ::Number)::AbstractMatrix
    return X
end
function ew_volatility_input(::SemiMoment, X::AbstractMatrix{<:Real},
                             mar::Real)::AbstractMatrix{<:Real}
    return min.(X .- mar, zero(mar))
end
"""
    descriptor(de::EWVolatility, rd::ReturnsResult) -> Matrix{<:Real}

Compute an exponentially weighted volatility Descriptor from a [`ReturnsResult`](@ref).

# Algorithm

 1. Transform the returns through [`ew_volatility_input`](@ref), which reads the `alg` field, giving `Y`.
 2. Take the point-in-time variance series `V` of `Y` through [`variance_series`](@ref). It reads the active mask of the Asset Panel, so an asset that turns inactive restarts its recursion.
 3. Take the square root `D`, and write `NaN` into the inactive cells through [`descriptor_active_fill!`](@ref).

The variance estimator owns the warm-up and the bias correction, so a cell where it returns `NaN` is `NaN` in the Descriptor. The call is [`ew_volatility_fold`](@ref) from no state, so a step of the carry fold runs the same arithmetic.

# Arguments

  - `de`: Descriptor Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.

# Validation

  - `rd.pnl` is an [`AssetPanel`](@ref). Raises an [`IsNothingError`](@ref).
  - No variance overflows the arithmetic. Raises an `IsNonFiniteError`, as [`assert_ew_volatility_finite`](@ref) states.

# Returns

  - `D::Matrix{<:Real}`: The Descriptor, `observations × assets`.

# Examples

```jldoctest
julia> pnl = asset_panel([NumericPanelInput(; name = \"market_cap\",
                                            vals = [1.0 2.0; 3.0 4.0; 5.0 6.0])];
                         amsk = trues(3, 2), emsk = trues(3, 2));

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = [0.1 0.2; -0.1 0.0; 0.05 0.05], pnl = pnl);

julia> descriptor(EWVolatility(; half_life = 1), rd)
3×2 Matrix{Float64}:
 0.1        0.2
 0.1        0.11547
 0.0755929  0.0845154
```

# Related

  - [`EWVolatility`](@ref)
  - [`EWDownsideVolatility`](@ref)
  - [`ew_volatility_input`](@ref)
  - [`variance_series`](@ref)
  - [`descriptor_active_fill!`](@ref)
"""
function descriptor(de::EWVolatility, rd::ReturnsResult)::Matrix{<:Real}
    return ew_volatility_fold(de, rd, nothing).D
end
"""
    EWDownsideVolatility(; half_life::Real = 40.0, mar::Real = 0.0,
                         ce::AbstractCovarianceEstimator = ew_variance_estimator(half_life)) -> EWVolatility

Exponentially weighted volatility of the returns that fall short of a minimum acceptable return.

The value is the volatility of [`EWVolatility`](@ref) measured on `min(r - mar, 0)`, so a return above the target contributes zero and only the shortfall carries risk. An investor who fears a loss and not a gain reads this in place of the two-sided volatility. The default half-life of `40` and the default target of zero are the two-sided Descriptor's own.

# Arguments

  - `half_life`: Half-life of the recursion, in observations. It fixes the default of `ce`.
  - `mar`: Minimum acceptable return. A return above it contributes zero.
  - `ce`: Variance estimator the Descriptor reads.

# Validation

  - The rules of [`EWVolatility`](@ref).

# Returns

  - `de::EWVolatility`: The estimator, with the [`SemiMoment`](@ref) Algorithm and the target fixed.

# Examples

```jldoctest
julia> de = EWDownsideVolatility(; half_life = 5);

julia> (de.ce.min_obs, isa(de.alg, SemiMoment), de.mar)
(5, true, 0.0)
```

# Related

  - [`EWVolatility`](@ref)
  - [`descriptor`](@ref)
  - [`ew_volatility_input`](@ref)
  - [`SemiMoment`](@ref)
"""
function EWDownsideVolatility(; half_life::Real = 40.0, mar::Real = 0.0,
                              ce::AbstractCovarianceEstimator = ew_variance_estimator(half_life))::EWVolatility
    return EWVolatility(; ce = ce, alg = SemiMoment(), mar = mar)
end

"""
$(DocStringExtensions.TYPEDEF)

Exponentially weighted volatility of the market-model residual, at every observation.

This is the archetype of every residual volatility Descriptor. It removes the part of a return the market explains, and measures the volatility of what is left, so the Descriptor carries the risk that belongs to the asset alone. A caller who already models market beta reads this in place of the total volatility, because the two together would count the market twice.

[`ew_regression_series!`](@ref) estimates the beta at its own decay, as [`EWBeta`](@ref) states, and the residual of an observation takes the beta of that same observation. The volatility of the residual is the [`variance_series`](@ref) of the estimator in the `ce` field. That estimator carries the recursion, the bias correction and the reset of an asset that turns inactive, so this type repeats none of them.

# Mathematical definition

```math
\\begin{align}
\\varepsilon_{t,i} &= x_{t,\\,i} - \\beta_{t,i} r_{m,t}\\,, \\\\
y_{t,i} &= \\begin{cases} \\min(\\varepsilon_{t,i} - \\mathrm{mar},\\, 0) & \\text{if } \\texttt{alg} \\text{ is a } \\mathrm{SemiMoment} \\\\ \\varepsilon_{t,i} & \\text{otherwise} \\end{cases}\\,, \\\\
d_{t,i} &= \\sqrt{\\hat{\\sigma}_{t,i}^2}\\,.
\\end{align}
```

The beta includes observation ``t``, and it is fitted with an intercept that the residual keeps. A beta from a few observations can be extreme and inflate the residual, so the beta exists only from the `beta_min_obs`-th valid observation of the asset, and the residual exists only where the beta does. A residual that does not exist does not enter the variance, so the warm-up of `ce` counts the residuals alone. The Descriptor is `NaN` where the beta of the observation is `NaN`, for example where the market does not vary, and the variance still keeps its history there.

Where:

  - ``d_{t,i}``: Descriptor of asset ``i`` at observation ``t``.
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:r_mt_ewb])
  - $(math_dict[:beta_ti_ewb])
  - ``\\varepsilon_{t,i}``: Market-model residual of asset ``i`` at observation ``t``, `NaN` where the asset is inactive, its return is not finite, or its beta is `NaN`.
  - ``y_{t,i}``: Input the variance estimator reads for asset ``i`` at observation ``t``.
  - ``\\hat{\\sigma}_{t,i}^2``: Variance of asset ``i`` after observation ``t``, as `ce` estimates it from the inputs ``y`` up to ``t``. The default `ce` is the recursion that [`ew_variance_estimator`](@ref) states.
  - ``\\mathrm{mar}``: `mar`, the minimum acceptable return.

[`market_return_series`](@ref) builds ``r_{m,t}``, and [`ew_regression_series!`](@ref) runs the recursion of ``\\beta_{t,i}`` at the decay `beta_decay`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EWResidualVolatility(; mcap::AbstractString = "market_cap", half_life::Real = 40.0,
                         beta_half_life::Real = 60.0,
                         beta_decay::Real = half_life_decay(beta_half_life, :beta_half_life),
                         beta_min_obs::Integer = 10,
                         ce::AbstractCovarianceEstimator = ew_variance_estimator(half_life,
                                                                                 max(half_life,
                                                                                     beta_half_life)),
                         alg::AbstractMomentAlgorithm = FullMoment(), mar::Real = 0.0,
                         cache::Option{<:AbstractPartialFitState} = nothing) -> EWResidualVolatility

Keywords correspond to the struct's fields, except `half_life` and `beta_half_life`, which are not fields. `half_life` fixes the decay of `ce`, `beta_half_life` fixes `beta_decay`, and the **longer** of the two fixes the warm-up of `ce`, because a residual is no better estimated than the beta that formed it. The constructor keeps a `ce` or a `beta_decay` that the caller passes as it is. The default half-lives of `40` and `60` weight about as far back as two months and one quarter of daily observations. With the defaults and no missing return, the first value is at observation `69`: the beta needs ten observations, and `ce` then needs sixty residuals.

## Validation

  - `mcap` is not empty.
  - `ce` is mask-aware. Raises an `ArgumentError`, as [`assert_mask_aware_variance`](@ref) states.
  - `0 < beta_decay < 1`.
  - `beta_min_obs >= 1`.
  - `isfinite(mar)`.
  - `0 < half_life < Inf` and `0 < beta_half_life < Inf`, where each fixes a default. Raises a `DomainError` that names the half-life.

# Examples

```jldoctest
julia> de = EWResidualVolatility(; half_life = 5, beta_half_life = 8);

julia> (de.ce.decay ≈ exp2(-inv(5.0)), de.ce.min_obs, de.beta_decay ≈ exp2(-inv(8.0)),
        de.beta_min_obs)
(true, 8, true, 10)
```

# Related

  - [`AbstractDescriptorEstimator`](@ref)
  - [`descriptor`](@ref)
  - [`EWResidualDownsideVolatility`](@ref)
  - [`EWVolatility`](@ref)
  - [`ew_regression_series!`](@ref)
  - [`ew_variance_estimator`](@ref)
  - [`market_return_series`](@ref)
  - [`variance_series`](@ref)
"""
@concrete struct EWResidualVolatility <: AbstractDescriptorEstimator
    """
    $(field_dict[:mcap])
    """
    mcap
    """
    Variance estimator whose [`variance_series`](@ref) the Descriptor takes the square root of, over the residual rather than over the return. It carries the recursion, the warm-up and the reset of an asset that turns inactive. It must be mask-aware, because the residual of every asset is `NaN` until its beta exists, as [`assert_mask_aware_variance`](@ref) states.
    """
    ce
    """
    Decay factor of the beta recursion. It is separate from the decay of `ce`, so the beta can weight observations further back than the variance of the residual does.
    """
    beta_decay
    """
    Number of valid observations of an asset, since its last reset, that its beta needs before a residual is formed from it. A beta from a few observations can be extreme, and its residual then inflates the volatility.
    """
    beta_min_obs
    """
    $(field_dict[:malg])
    """
    alg
    """
    Minimum acceptable return that the downside form measures the residuals against. The Descriptor reads it only where `alg` is a [`SemiMoment`](@ref), which clips every excess above it to zero.
    """
    mar
    """
    $(field_dict[:ew_vol_desc_cache])
    """
    cache
    function EWResidualVolatility(mcap::AbstractString, ce::AbstractCovarianceEstimator,
                                  beta_decay::Real, beta_min_obs::Integer,
                                  alg::AbstractMomentAlgorithm, mar::Real,
                                  cache::Option{<:AbstractPartialFitState})
        assert_panel_terms(mcap, :mcap)
        assert_mask_aware_variance(ce)
        assert_ew_decay(beta_decay)
        assert_nonempty_gt0_finite_val(beta_min_obs, :beta_min_obs)
        assert_finite(mar, :mar)
        return new{typeof(mcap), typeof(ce), typeof(beta_decay), typeof(beta_min_obs),
                   typeof(alg), typeof(mar), typeof(cache)}(mcap, ce, beta_decay,
                                                            beta_min_obs, alg, mar, cache)
    end
end
function EWResidualVolatility(; mcap::AbstractString = "market_cap", half_life::Real = 40.0,
                              beta_half_life::Real = 60.0,
                              beta_decay::Real = half_life_decay(beta_half_life,
                                                                 :beta_half_life),
                              beta_min_obs::Integer = 10,
                              ce::AbstractCovarianceEstimator = ew_variance_estimator(half_life,
                                                                                      max(half_life,
                                                                                          beta_half_life)),
                              alg::AbstractMomentAlgorithm = FullMoment(), mar::Real = 0.0,
                              cache::Option{<:AbstractPartialFitState} = nothing)::EWResidualVolatility
    return EWResidualVolatility(mcap, ce, beta_decay, beta_min_obs, alg, mar, cache)
end
"""
    assert_mask_aware_variance(ce::AbstractCovarianceEstimator)
    assert_mask_aware_variance(ce::Union{RegimeAdjustedExpWeightedVariance,
                                         RegimeAdjustedExpWeightedCovariance,
                                         ExpWeightedVariance, ExpWeightedCovariance,
                                         Covariance{<:Any, <:Any, <:Any, <:Any,
                                                    <:CoveragePolicy},
                                         SimpleVariance{<:Any, <:Any, <:Any,
                                                        <:CoveragePolicy}})
    assert_mask_aware_variance(ce::WindowedVariance)
    assert_mask_aware_variance(ce::WindowedCovariance)

Refuses a variance estimator whose [`variance_series`](@ref) over an Asset Panel is not mask-aware, as the `ce` of an [`EWResidualVolatility`](@ref).

[`ew_residual_returns`](@ref) writes `NaN` into the residual of every asset from its first observation until its beta exists. A plain estimator reduces each window of the series to its Coverage Universe, and every window holds the first observation. So no asset is in the Coverage Universe of a window, and every cell of the Descriptor is `NaN`, on every input. A windowed estimator over a plain one raises an `IsEmptyError` on its first window. A mask-aware estimator reads each asset from its first finite residual, so it answers. The configuration alone decides this, so the constructor refuses it.

The admitted estimators are the ones whose series has a mask-aware method of its own: the exponentially weighted and the regime-adjusted families, a [`Covariance`](@ref) or a [`SimpleVariance`](@ref) with a [`CoveragePolicy`](@ref), and a windowed estimator over an admitted one. A [`PortfolioOptimisersCovariance`](@ref) has no such method, so it is refused whatever its inner estimator.

# Arguments

  - `ce`: The variance estimator of the Descriptor.

# Validation

  - `ce` is mask-aware. An `ArgumentError` that names the plain estimator is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`EWResidualVolatility`](@ref)
  - [`ew_residual_returns`](@ref)
  - [`variance_series(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
  - [`CoveragePolicy`](@ref)
"""
function assert_mask_aware_variance(ce::AbstractCovarianceEstimator)::Nothing
    return throw(ArgumentError("the `ce` of an EWResidualVolatility holds a `$(typeof(ce).name.name)`, which is not mask-aware: its variance series reduces each window to its Coverage Universe, and the residual of every asset is `NaN` until its beta exists, so every cell of the Descriptor would be `NaN`. Pass the default `ce`, an `ExpWeightedVariance`, or a `SimpleVariance(; cvg = CoveragePolicy())`."))
end
function assert_mask_aware_variance(::Union{RegimeAdjustedExpWeightedVariance,
                                            RegimeAdjustedExpWeightedCovariance,
                                            ExpWeightedVariance, ExpWeightedCovariance,
                                            Covariance{<:Any, <:Any, <:Any, <:Any,
                                                       <:CoveragePolicy},
                                            SimpleVariance{<:Any, <:Any, <:Any,
                                                           <:CoveragePolicy}})::Nothing
    return nothing
end
function assert_mask_aware_variance(ce::WindowedVariance)::Nothing
    return assert_mask_aware_variance(ce.ve)
end
function assert_mask_aware_variance(ce::WindowedCovariance)::Nothing
    return assert_mask_aware_variance(ce.ce)
end
"""
    ew_residual_returns(X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                        B::AbstractMatrix{<:Real},
                        amsk::AbstractMatrix{Bool}) -> Matrix{<:Real}

Remove the part of every return that the market explains.

A residual exists only where the asset is active and its return is finite, so every other cell is `NaN` and the variance recursion holds its state there rather than reading a zero.

# Mathematical definition

```math
\\begin{align}
\\varepsilon_{t,i} &= \\begin{cases} x_{t,\\,i} - \\beta_{t,i} r_{m,t} & \\text{if asset } i \\text{ is active at } t \\text{ and } x_{t,\\,i} \\text{ is finite} \\\\ \\mathrm{NaN} & \\text{otherwise} \\end{cases}\\,.
\\end{align}
```

Where:

  - ``\\varepsilon_{t,i}``: Market-model residual of asset ``i`` at observation ``t``.
  - $(math_dict[:x_ti_ret])
  - ``\\beta_{t,i}``: Market beta of asset ``i`` at observation ``t``, the entry of `B`.
  - $(math_dict[:r_mt_ewb])

# Arguments

  - `X`: The returns, `observations × assets`.
  - `rm`: The market return, one entry per observation.
  - `B`: The market betas, `observations × assets`.
  - `amsk`: The active mask of the Asset Panel, `observations × assets`.

# Returns

  - `E::Matrix{<:Real}`: The residuals, `observations × assets`.

# Examples

```jldoctest
julia> PortfolioOptimisers.ew_residual_returns([0.1 0.2; NaN 0.0], [0.1, -0.05],
                                               [1.0 2.0; 1.0 2.0], trues(2, 2))
2×2 Matrix{Float64}:
   0.0  0.0
 NaN    0.1
```

# Related

  - [`EWResidualVolatility`](@ref)
  - [`ew_regression_series!`](@ref)
  - [`descriptor`](@ref)
"""
function ew_residual_returns(X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                             B::AbstractMatrix{<:Real},
                             amsk::AbstractMatrix{Bool})::Matrix{<:Real}
    Tf = promote_type(eltype(X), eltype(rm), eltype(B))
    E = fill(Tf(NaN), size(X))
    for t in axes(X, 1), i in axes(X, 2)
        x = X[t, i]
        if amsk[t, i] && isfinite(x)
            E[t, i] = x - B[t, i] * rm[t]
        end
    end
    return E
end
"""
    descriptor(de::EWResidualVolatility, rd::ReturnsResult) -> Matrix{<:Real}

Compute an exponentially weighted residual volatility Descriptor from a [`ReturnsResult`](@ref).

# Algorithm

 1. Build the market return `rm` through [`market_return_series`](@ref).
 2. Run the beta regression through [`ew_regression_series!`](@ref), giving the betas `B`. The regression resets an asset that turns inactive, and it writes a beta from the `beta_min_obs`-th valid observation of the asset.
 3. Remove the market from every return through [`ew_residual_returns`](@ref), giving the residuals `E`. A residual is `NaN` where the beta is `NaN`.
 4. Transform the residuals through [`ew_volatility_input`](@ref), which reads the `alg` field, giving `Y`.
 5. Take the point-in-time variance series `V` of `Y` through [`variance_series`](@ref), and take its square root `D`.
 6. Write `NaN` where the beta is `NaN` through [`ew_residual_unresolved!`](@ref), and into the inactive cells through [`descriptor_active_fill!`](@ref).

The variance estimator owns the warm-up and the bias correction, so a cell where it returns `NaN` is `NaN` in the Descriptor. The call is [`ew_volatility_fold`](@ref) from no state, so a step of the carry fold runs the same arithmetic.

# Arguments

  - `de`: Descriptor Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.

# Validation

  - `rd.pnl` is an [`AssetPanel`](@ref). Raises an [`IsNothingError`](@ref).
  - The rules of [`market_return_series`](@ref).
  - No variance of a residual overflows the arithmetic. Raises an `IsNonFiniteError`, as [`assert_ew_volatility_finite`](@ref) states.

# Returns

  - `D::Matrix{<:Real}`: The Descriptor, `observations × assets`.

# Examples

```jldoctest
julia> pnl = asset_panel([NumericPanelInput(; name = \"market_cap\",
                                            vals = [1.0 2.0; 3.0 4.0; 5.0 6.0])];
                         amsk = trues(3, 2), emsk = trues(3, 2));

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = [0.1 0.2; -0.1 0.0; 0.05 0.05], pnl = pnl);

julia> descriptor(EWResidualVolatility(; half_life = 1, beta_half_life = 1, beta_min_obs = 2), rd)
3×2 Matrix{Float64}:
 NaN          NaN
   0.0590909    0.0409091
   0.0341457    0.0239417
```

# Related

  - [`EWResidualVolatility`](@ref)
  - [`EWResidualDownsideVolatility`](@ref)
  - [`ew_regression_series!`](@ref)
  - [`ew_residual_returns`](@ref)
  - [`ew_volatility_input`](@ref)
  - [`variance_series`](@ref)
"""
function descriptor(de::EWResidualVolatility, rd::ReturnsResult)::Matrix{<:Real}
    return ew_volatility_fold(de, rd, nothing).D
end
"""
    EWResidualDownsideVolatility(; mcap::AbstractString = "market_cap",
                                 half_life::Real = 40.0, beta_half_life::Real = 60.0,
                                 mar::Real = 0.0,
                                 beta_decay::Real = half_life_decay(beta_half_life,
                                                                    :beta_half_life),
                                 ce::AbstractCovarianceEstimator = ew_variance_estimator(half_life,
                                                                                         max(half_life,
                                                                                             beta_half_life)),
                                 beta_min_obs::Integer = 10) -> EWResidualVolatility

Exponentially weighted volatility of the market-model residuals that fall short of a minimum acceptable return.

The value is the volatility of [`EWResidualVolatility`](@ref) measured on `min(ε - mar, 0)`, so a residual above the target contributes zero and only the shortfall carries risk. It is the risk an asset carries alone, on the side an investor fears. The default half-lives and the default target are the two-sided Descriptor's own.

# Arguments

  - $(arg_dict[:mcap])
  - `half_life`: Half-life of the volatility recursion, in observations.
  - `beta_half_life`: Half-life of the beta recursion, in observations.
  - `mar`: Minimum acceptable return. A residual above it contributes zero.
  - `beta_decay`: Decay factor of the beta recursion.
  - `ce`: Variance estimator the Descriptor reads.
  - `beta_min_obs`: Number of valid observations that the beta of an asset needs before a residual is formed from it.

# Validation

  - The rules of [`EWResidualVolatility`](@ref).

# Returns

  - `de::EWResidualVolatility`: The estimator, with the [`SemiMoment`](@ref) Algorithm and the target fixed.

# Examples

```jldoctest
julia> de = EWResidualDownsideVolatility(; half_life = 5, beta_half_life = 8);

julia> (de.ce.min_obs, isa(de.alg, SemiMoment), de.mar)
(8, true, 0.0)
```

# Related

  - [`EWResidualVolatility`](@ref)
  - [`descriptor`](@ref)
  - [`EWDownsideVolatility`](@ref)
  - [`ew_volatility_input`](@ref)
  - [`SemiMoment`](@ref)
"""
function EWResidualDownsideVolatility(; mcap::AbstractString = "market_cap",
                                      half_life::Real = 40.0, beta_half_life::Real = 60.0,
                                      mar::Real = 0.0,
                                      beta_decay::Real = half_life_decay(beta_half_life,
                                                                         :beta_half_life),
                                      ce::AbstractCovarianceEstimator = ew_variance_estimator(half_life,
                                                                                              max(half_life,
                                                                                                  beta_half_life)),
                                      beta_min_obs::Integer = 10)::EWResidualVolatility
    return EWResidualVolatility(; mcap = mcap, ce = ce, beta_decay = beta_decay,
                                beta_min_obs = beta_min_obs, alg = SemiMoment(), mar = mar)
end

"""
$(DocStringExtensions.TYPEDEF)

The carried state of an [`EWVolatility`](@ref) or of an [`EWResidualVolatility`](@ref), after the observations that it folded.

The state holds the partial-fit state of the variance estimator `ce`, and for an [`EWResidualVolatility`](@ref) the state of its beta recursion too. A step of the carry fold runs the two recursions from it with the arithmetic of the batch call, as [`ew_volatility_fold`](@ref) states.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EWVolatilityState(; ve::AbstractPartialFitState,
                      beta::Option{<:EWBetaState} = nothing) -> EWVolatilityState

Keywords correspond to the struct's fields.

# Related

  - [`EWVolatility`](@ref)
  - [`EWResidualVolatility`](@ref)
  - [`ew_volatility_fold`](@ref)
  - [`descriptor_step`](@ref)
"""
@concrete struct EWVolatilityState <: AbstractPartialFitState
    """
    The partial-fit state of the variance estimator `ce`, a [`RegimeAdjustedVarianceState`](@ref).
    """
    ve
    """
    The state of the beta recursion of an [`EWResidualVolatility`](@ref), an [`EWBetaState`](@ref), or `nothing` for an [`EWVolatility`](@ref).
    """
    beta
    function EWVolatilityState(ve::AbstractPartialFitState, beta::Option{<:EWBetaState})
        return new{typeof(ve), typeof(beta)}(ve, beta)
    end
end
function EWVolatilityState(; ve::AbstractPartialFitState,
                           beta::Option{<:EWBetaState} = nothing)::EWVolatilityState
    return EWVolatilityState(ve, beta)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses to merge two [`EWVolatilityState`](@ref): the recursion of the second block starts from the state after the first one, so two states fitted on disjoint blocks do not give the state of their union to the last bit.

# Validation

  - Always throws an `ArgumentError`.

# Related

  - [`EWVolatilityState`](@ref)
  - [`partial_fit!`](@ref)
"""
function merge_states(::EWVolatilityState, ::EWVolatilityState)
    return throw(ArgumentError("an EWVolatilityState cannot merge two states fitted on disjoint blocks: the recursion of the second block starts from the state after the first one. Fold the second block into the state of the first with partial_fit!."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies an [`EWVolatilityState`](@ref), so that the copy shares no vector with the original.

# Arguments

  - `x`: The state to copy.

# Returns

  - `state::EWVolatilityState`: A new state, equal to `x`.

# Related

  - [`EWVolatilityState`](@ref)
"""
function Base.copy(x::EWVolatilityState)
    return EWVolatilityState(copy(x.ve), isnothing(x.beta) ? nothing : copy(x.beta))
end
"""
    ew_volatility_parts(::Nothing)
    ew_volatility_parts(st::EWVolatilityState)

Returns the two parts of the carried state of a volatility Descriptor, `ve` and `beta`: both `nothing` for no state, or the two fields of the state.

# Arguments

  - `st`: The carried state, or `nothing`.

# Returns

  - `(ve, beta)::Tuple`: The plain tuple of the state of the variance estimator and the beta state, each `nothing` when the state does not hold it.

# Related

  - [`EWVolatilityState`](@ref)
  - [`ew_volatility_fold`](@ref)
"""
function ew_volatility_parts(::Nothing)::Tuple{Nothing, Nothing}
    return nothing, nothing
end
function ew_volatility_parts(st::EWVolatilityState)::Tuple
    return st.ve, st.beta
end
"""
    ew_volatility_state(::Nothing, ::Option{<:EWBetaState})
    ew_volatility_state(ve::AbstractPartialFitState, beta::Option{<:EWBetaState})

Returns the state of a volatility Descriptor after a fold: `nothing` where the variance estimator `ce` has no state, because it does not fold, or the [`EWVolatilityState`](@ref) of the two parts.

# Arguments

  - `ve`: The state of `ce` after the fold, or `nothing`.
  - `beta`: The state of the beta recursion after the fold, or `nothing`.

# Returns

  - `st::Option{<:EWVolatilityState}`: The state, or `nothing`.

# Related

  - [`EWVolatilityState`](@ref)
  - [`ew_volatility_fold`](@ref)
"""
function ew_volatility_state(::Nothing, ::Option{<:EWBetaState})::Nothing
    return nothing
end
function ew_volatility_state(ve::AbstractPartialFitState,
                             beta::Option{<:EWBetaState})::EWVolatilityState
    return EWVolatilityState(ve, beta)
end
"""
    ew_volatility_variance(ce::RegimeAdjustedExpWeightedVariance, Y::AbstractMatrix{<:Real},
                           pnl::AssetPanel, st::Option{<:RegimeAdjustedVarianceState})
    ew_volatility_variance(ce::AbstractCovarianceEstimator, Y::AbstractMatrix{<:Real},
                           pnl::AssetPanel, ::Any)

Returns the point-in-time variance series of the input of a volatility Descriptor, and the state of the variance estimator after it.

A [`RegimeAdjustedExpWeightedVariance`](@ref) runs the forward pass of its [`variance_series`](@ref) with [`regime_adjusted_variance_pass!`](@ref), from a copy of the carried state or from the cold state, and reads the variance of each observation with [`regime_adjusted_variance`](@ref). The batch call and the step thus run one pass, and a folded row equals the batch call to the last bit. The copy leaves the carried state as it was. Every other estimator takes its [`variance_series`](@ref) over the whole input, and has no state. It never makes a state, so it only gets `nothing`.

# Arguments

  - `ce`: The variance estimator.
  - `Y`: The input of the variance, `observations × assets`.
  - `pnl`: The Asset Panel of the observations. Its two masks reset and gate the recursion.
  - `st`: The carried state of `ce`, or `nothing`.

# Validation

  - A carried state holds the assets of `Y`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `(V, st)::Tuple`: The plain tuple of `V`, the variance of each observation, `observations × assets`, and `st`, the state of `ce` after the observations, or `nothing` for an estimator that does not fold.

# Related

  - [`ew_volatility_fold`](@ref)
  - [`variance_series`](@ref)
  - [`RegimeAdjustedVarianceState`](@ref)
"""
function ew_volatility_variance(ce::RegimeAdjustedExpWeightedVariance,
                                Y::AbstractMatrix{<:Real}, pnl::AssetPanel,
                                st::Option{<:RegimeAdjustedVarianceState})
    amsk, emsk = panel_moment_masks(pnl)
    V = Matrix{eltype(Y)}(undef, size(Y))
    vs = regime_adjusted_variance_pass!(ce, Y, 1, emsk, amsk,
                                        isnothing(st) ? nothing : copy(st)) do i, cache
        V[i, :] = regime_adjusted_variance(cache, ce)
        return nothing
    end
    return V, vs
end
function ew_volatility_variance(ce::AbstractCovarianceEstimator, Y::AbstractMatrix{<:Real},
                                pnl::AssetPanel, ::Any)
    return variance_series(ce, Y, pnl; dims = 1), nothing
end
"""
    ew_volatility_fold(de::EWVolatility, rd::ReturnsResult,
                       cache::Option{<:EWVolatilityState})
    ew_volatility_fold(de::EWResidualVolatility, rd::ReturnsResult,
                       cache::Option{<:EWVolatilityState})

Folds the observations of a [`ReturnsResult`](@ref) into the recursions of a volatility Descriptor from a state, and returns the state after them and the Descriptor of each observation.

The batch call [`descriptor`](@ref) is this function from no state, so a step equals the batch call by construction. Every input of the recursions reads one row: the transformed return, the market return of [`market_return_series`](@ref) and the residual of [`ew_residual_returns`](@ref). So the step needs no row before its own.

# Algorithm

 1. For an [`EWResidualVolatility`](@ref), build the market return, and run the beta regression with [`ew_regression_series!`](@ref) from the beta state, or from zero. Remove the market from every return.
 2. Transform the returns, or the residuals, through [`ew_volatility_input`](@ref).
 3. Take the variance of each observation and the state of `ce` after them with [`ew_volatility_variance`](@ref), refuse an infinite variance with [`assert_ew_volatility_finite`](@ref), and take the square root.
 4. For an [`EWResidualVolatility`](@ref), write `NaN` where the beta is `NaN` through [`ew_residual_unresolved!`](@ref). Write `NaN` into the inactive cells through [`descriptor_active_fill!`](@ref).

# Arguments

  - `de`: Descriptor Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `cache`: The carried state, or `nothing` for the first observation.

# Validation

  - The rules of [`descriptor`](@ref), of [`ew_beta_state`](@ref), of [`ew_volatility_variance`](@ref) and of [`assert_ew_volatility_finite`](@ref).
  - The state of an [`EWResidualVolatility`](@ref) holds a beta state, and the state of an [`EWVolatility`](@ref) holds none. An `ArgumentError` is thrown otherwise.

# Returns

  - `fold::DescriptorFold`: `st`, the state after the observations, or `nothing` where `ce` does not fold, and `D`, the Descriptor of each observation, `observations × assets`.

# Related

  - [`descriptor_step`](@ref)
  - [`EWVolatilityState`](@ref)
"""
function ew_volatility_fold(de::EWVolatility, rd::ReturnsResult,
                            cache::Option{<:EWVolatilityState})
    ve, beta = ew_volatility_parts(cache)
    @argcheck(isnothing(beta),
              ArgumentError("the state of this EWVolatility holds a beta state, so it is the state of an EWResidualVolatility. An EWVolatility folds from its own state, or from no state."))
    pnl = descriptor_asset_panel(rd)
    Y = ew_volatility_input(de.alg, rd.X, de.mar)
    V, vst = ew_volatility_variance(de.ce, Y, pnl, ve)
    assert_ew_volatility_finite(V)
    D = sqrt.(V)
    descriptor_active_fill!(D, pnl)
    return DescriptorFold(ew_volatility_state(vst, nothing), D)
end
function ew_volatility_fold(de::EWResidualVolatility, rd::ReturnsResult,
                            cache::Option{<:EWVolatilityState})
    ve, beta = ew_volatility_parts(cache)
    @argcheck(isnothing(cache) == isnothing(beta),
              ArgumentError("the state of this EWResidualVolatility holds no beta state, so it is the state of an EWVolatility. An EWResidualVolatility folds from its own state, or from no state."))
    pnl = descriptor_asset_panel(rd)
    rm = market_return_series(rd, de.mcap)
    X = rd.X
    amsk = pnl.amsk
    bs = ew_beta_state(beta, X, rm, nothing, de.beta_decay, nothing, 1)
    B, bst = ew_regression_series!(bs, nothing, X, rm, nothing, amsk, de.beta_min_obs, 1)
    E = ew_residual_returns(X, rm, B, amsk)
    Y = ew_volatility_input(de.alg, E, de.mar)
    V, vst = ew_volatility_variance(de.ce, Y, pnl, ve)
    assert_ew_volatility_finite(V)
    D = sqrt.(V)
    ew_residual_unresolved!(D, B)
    descriptor_active_fill!(D, pnl)
    return DescriptorFold(ew_volatility_state(vst, bst), D)
end
"""
    ew_residual_unresolved!(D::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real}) -> D

Writes `NaN` into every cell of the residual volatility `D` whose beta in `B` is `NaN`, in place. A beta that the regression cannot resolve leaves no residual, so the volatility of that observation is not stated, and the variance keeps its history for the next observation.

# Arguments

  - `D`: The residual volatility, `observations × assets`, changed in place.
  - `B`: The betas of the same cells.

# Returns

  - `D::AbstractMatrix{<:Real}`: The residual volatility.

# Related

  - [`ew_volatility_fold`](@ref)
  - [`EWResidualVolatility`](@ref)
"""
function ew_residual_unresolved!(D::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real})
    for k in eachindex(D, B)
        if isnan(B[k])
            D[k] = eltype(D)(NaN)
        end
    end
    return D
end
"""
    assert_ew_volatility_finite(V::AbstractMatrix{<:Real}) -> nothing

Checks that no variance of a volatility Descriptor is infinite. The returns are finite, so an infinite variance is the square of a return or of a residual that overflows the arithmetic. Without the check, the Descriptor would be an infinite volatility, where every other cell is finite or `NaN`. A `NaN` variance, as in the warm-up, passes.

# Arguments

  - `V`: The variance of each observation, `observations × assets`.

# Validation

  - No entry of `V` is infinite. An `IsNonFiniteError` that names the observation and the asset is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`ew_volatility_fold`](@ref)
  - [`ew_volatility_variance`](@ref)
"""
function assert_ew_volatility_finite(V::AbstractMatrix{<:Real})::Nothing
    k = findfirst(isinf, V)
    @argcheck(isnothing(k),
              IsNonFiniteError("the exponentially weighted variance of asset $(k[2]) at observation $(k[1]) is not finite, because the square of its return or of its residual overflows the arithmetic. Rescale the returns."))
    return nothing
end
"""
    descriptor_step(de::Union{EWVolatility{<:RegimeAdjustedExpWeightedVariance},
                              EWResidualVolatility{<:Any,
                                                   <:RegimeAdjustedExpWeightedVariance}},
                    rd::ReturnsResult)

Folds the observations of a [`ReturnsResult`](@ref) into the carried state of a volatility Descriptor whose `ce` is a [`RegimeAdjustedExpWeightedVariance`](@ref), and returns the Descriptor of each one.

The Descriptor of an observation equals the one of the batch call [`descriptor`](@ref) over every observation that the state folded and the observations before it, to the last bit, because the state runs the recursions with the same arithmetic from the same first observation, as [`ew_volatility_fold`](@ref) states. The step copies the state, so the estimator it gets keeps its state.

A volatility Descriptor with any other `ce` has no state to carry, so it takes the fallback of [`descriptor_step`](@ref), which answers `nothing`. The carry fold of a [`CrossSectionalFactorPrior`](@ref) then keeps every row for it, or refuses it under the strict carry rule.

# Arguments

  - `de`: The estimator, with or without a state.
  - $(arg_dict[:rd]) It holds the new observations alone.

# Validation

  - The rules of [`ew_volatility_fold`](@ref).

# Returns

  - `step::DescriptorStepResult`: `de`, the estimator with the state after the observations in `cache`, and `D`, the Descriptor of each observation, `observations × assets`.

# Related

  - [`EWVolatilityState`](@ref)
  - [`partial_fit!`](@ref)
  - [`descriptor_carry`](@ref)
"""
function descriptor_step(de::Union{EWVolatility{<:RegimeAdjustedExpWeightedVariance},
                                   EWResidualVolatility{<:Any,
                                                        <:RegimeAdjustedExpWeightedVariance}},
                         rd::ReturnsResult)
    (; st, D) = ew_volatility_fold(de, rd, de.cache)
    return DescriptorStepResult(rebuild_struct(de, (; cache = st)), D)
end
"""
    partial_fit!(de::Union{EWVolatility{<:RegimeAdjustedExpWeightedVariance},
                           EWResidualVolatility{<:Any,
                                                <:RegimeAdjustedExpWeightedVariance}},
                 rd::ReturnsResult)

Folds the observations of a [`ReturnsResult`](@ref) into the carried state of a volatility Descriptor, and returns the estimator with the state after them in `cache`. [`descriptor_step`](@ref) states the fold, and also returns the Descriptor of each observation.

# Arguments

  - `de`: The estimator, with no state or with its state.
  - $(arg_dict[:rd]) It holds the new observations alone.

# Validation

  - The rules of [`descriptor_step`](@ref).

# Returns

  - `de::Union{EWVolatility, EWResidualVolatility}`: The estimator, with its `cache` field set to the state after the observations.

# Related

  - [`descriptor_step`](@ref)
  - [`EWVolatilityState`](@ref)
"""
function partial_fit!(de::Union{EWVolatility{<:RegimeAdjustedExpWeightedVariance},
                                EWResidualVolatility{<:Any,
                                                     <:RegimeAdjustedExpWeightedVariance}},
                      rd::ReturnsResult)
    return descriptor_step(de, rd).de
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of an [`EWVolatility`](@ref) or of an [`EWResidualVolatility`](@ref) except `cache`.

The state a `cache` holds is the running detail of an incremental fit, not the configuration a reader looks the type up for, and it prints under the estimator at every site that renders one, such as a [`CompositeExposure`](@ref). Set `set_show_nothing_fields!(:EWVolatility, true)` to render it.

# Arguments

  - `de`: The estimator.

# Returns

  - `fields::Tuple`: The field names to render, every field name but `:cache`.

# Related

  - [`EWVolatility`](@ref)
  - [`EWResidualVolatility`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
function show_fields(de::Union{EWVolatility, EWResidualVolatility})
    return filter(!=(:cache), fieldnames(typeof(de)))
end

export EWVolatility, EWDownsideVolatility, EWResidualVolatility,
       EWResidualDownsideVolatility
