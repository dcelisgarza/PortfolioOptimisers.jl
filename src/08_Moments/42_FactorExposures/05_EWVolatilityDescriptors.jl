"""
    ew_variance_estimator(half_life::Real) -> RegimeAdjustedExpWeightedVariance

Build the exponentially weighted variance estimator an [`EWVolatility`](@ref) reads by default.

The estimator is the plain recursion: uncentred, bias corrected by `1 / (1 - λ^n)`, reset where an asset turns inactive, and with **no regime adjustment**. `centred = true` is what states the uncentred form, because the flag says that the returns are already centred and no location is estimated online. `regime_method = nothing` is the off switch [`RegimeAdjustedExpWeightedVariance`](@ref) carries for exactly this: a volatility Descriptor scales nothing by a regime multiplier.

# Arguments

  - `half_life`: The half-life of the recursion, in observations.

# Returns

  - `ce::RegimeAdjustedExpWeightedVariance`: The estimator, with the half-life converted through [`half_life_decay`](@ref) and [`half_life_min_obs`](@ref).

# Examples

```jldoctest
julia> ce = PortfolioOptimisers.ew_variance_estimator(5.0);

julia> (ce.decay ≈ exp2(-inv(5.0)), ce.min_obs, ce.centred, ce.regime_method)
(true, 5, true, nothing)
```

# Related

  - [`EWVolatility`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
  - [`variance_series`](@ref)
"""
function ew_variance_estimator(half_life::Real)::RegimeAdjustedExpWeightedVariance
    return RegimeAdjustedExpWeightedVariance(; decay = half_life_decay(half_life),
                                             min_obs = half_life_min_obs(half_life),
                                             centred = true, regime_method = nothing)
end
"""
$(DocStringExtensions.TYPEDEF)

Exponentially weighted volatility of the returns, at every observation.

This is the archetype of every volatility Descriptor. It holds a variance estimator in a slot and takes the square root of its [`variance_series`](@ref), so the recursion, the bias correction and the reset of an asset that turns inactive are stated once, by the estimator, and not again here. The downside form is the `alg` slot rather than a struct of its own, as [`Covariance`](@ref) spells the same choice.

# Mathematical definition

```math
\\begin{align}
d_{t,i} &= \\sqrt{\\hat{\\sigma}_{t,i}^2\\left(f(\\mathbf{X})\\right)}\\,, \\\\
f(x) &= \\begin{cases} \\min(x - \\mathrm{mar},\\, 0) & \\text{if } \\texttt{alg} \\text{ is a } \\mathrm{SemiMoment} \\\\ x & \\text{otherwise} \\end{cases}\\,.
\\end{align}
```

Where:

  - ``d_{t,i}``: Descriptor of asset ``i`` at observation ``t``.
  - ``\\hat{\\sigma}_{t,i}^2``: Variance series of asset ``i`` after observation ``t``, as `ce` estimates it.
  - ``\\mathrm{mar}``: The minimum acceptable return.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EWVolatility(; half_life::Real = 40.0,
                 ce::AbstractCovarianceEstimator = ew_variance_estimator(half_life),
                 alg::AbstractMomentAlgorithm = FullMoment(),
                 mar::Real = 0.0) -> EWVolatility

`ce`, `alg` and `mar` correspond to the struct's fields. `half_life` is not a field: it fixes the default of `ce` through [`ew_variance_estimator`](@ref), and a `ce` passed by the caller is used as it stands. The default half-life of `40` weights about as far back as a window of two months of daily observations.

## Validation

  - `isfinite(mar)`.

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
    Variance estimator whose [`variance_series`](@ref) the Descriptor takes the square root of. It carries the recursion, the warm-up and the reset of an asset that turns inactive.
    """
    ce
    """
    $(field_dict[:malg])
    """
    alg
    """
    Minimum acceptable return the returns are measured against. It is read only where `alg` is a [`SemiMoment`](@ref), which clips every excess above it to zero.
    """
    mar
    function EWVolatility(ce::AbstractCovarianceEstimator, alg::AbstractMomentAlgorithm,
                          mar::Real)
        assert_finite(mar, :mar)
        return new{typeof(ce), typeof(alg), typeof(mar)}(ce, alg, mar)
    end
end
function EWVolatility(; half_life::Real = 40.0,
                      ce::AbstractCovarianceEstimator = ew_variance_estimator(half_life),
                      alg::AbstractMomentAlgorithm = FullMoment(),
                      mar::Real = 0.0)::EWVolatility
    return EWVolatility(ce, alg, mar)
end
"""
    ew_volatility_input(alg::FullMoment, X::AbstractMatrix{<:Real},
                        mar::Real) -> AbstractMatrix{<:Real}
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
function ew_volatility_input(::FullMoment, X::AbstractMatrix{<:Real},
                             ::Real)::AbstractMatrix{<:Real}
    return X
end
function ew_volatility_input(::SemiMoment, X::AbstractMatrix{<:Real},
                             mar::Real)::AbstractMatrix{<:Real}
    return min.(X .- mar, zero(mar))
end
"""
    descriptor(de::EWVolatility, rd::ReturnsResult) -> Matrix{<:Real}

Compute an exponentially weighted volatility Descriptor from a carrier.

# Algorithm

 1. Transform the returns through [`ew_volatility_input`](@ref), which reads the `alg` slot.
 2. Take the point-in-time variance series through [`variance_series`](@ref), passing the Asset Panel's active mask so an asset that turns inactive restarts its recursion.
 3. Take the square root, and write `NaN` into the inactive cells through [`descriptor_active_fill!`](@ref).

The warm-up and the bias correction belong to the variance estimator, so a cell it answers `NaN` for is `NaN` here.

# Arguments

  - `de`: Descriptor Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.

# Validation

  - `rd.pnl` is an [`AssetPanel`](@ref). Raises an [`IsNothingError`](@ref).

# Returns

  - `D::Matrix{<:Real}`: The Descriptor, `observations × assets`.

# Examples

```jldoctest
julia> res = asset_panel([NumericPanelInput(; name = \"market_cap\",
                                            vals = [1.0 2.0; 3.0 4.0; 5.0 6.0])];
                         amsk = trues(3, 2), emsk = trues(3, 2));

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = [0.1 0.2; -0.1 0.0; 0.05 0.05], res...);

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
    pnl = descriptor_asset_panel(rd)
    Y = ew_volatility_input(de.alg, rd.X, de.mar)
    V = variance_series(de.ce, Y; dims = 1, active_mask = pnl.amsk)
    D = sqrt.(V)
    descriptor_active_fill!(D, pnl)
    return D
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

export EWVolatility, EWDownsideVolatility
