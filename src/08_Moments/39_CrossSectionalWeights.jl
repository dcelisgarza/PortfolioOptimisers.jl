"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all cross-sectional regression weight policy types.

A member says what weight an asset carries in the cross-sectional fit of an observation, and whether that weight is decided once or twice. A one-pass member reads the cross-section alone. A two-pass member reads the residuals of a first fit, so the caller runs the regression again with the weights the member returns.

All concrete and/or abstract types representing cross-sectional weight policies should be subtypes of `AbstractCrossSectionalWeightsAlgorithm`.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractCrossSectionalWeightsAlgorithm` and implement the following methods:

## `cs_weights_initial`

  - [`cs_weights_initial(alg::AbstractCrossSectionalWeightsAlgorithm, mcap::Option{<:MatNum}, mask::AbstractMatrix{Bool})`](@ref): Returns the weights of the first pass.

### Arguments

  - `alg`: The concrete subtype instance.
  - `mcap`: Market capitalisation matrix `observations × assets`, or `nothing`.
  - `mask`: Eligibility mask `observations × assets`.

### Returns

  - `W0::Matrix{<:Number}`: First-pass weights `observations × assets`, zero outside `mask`.

## `cs_weights_refine`

  - [`cs_weights_refine(alg::AbstractCrossSectionalWeightsAlgorithm, W0::MatNum, eps::MatNum, ve::AbstractCovarianceEstimator, mask::AbstractMatrix{Bool}; kwargs...)`](@ref): Returns the weights of the second pass. A member whose [`needs_second_pass`](@ref) is `false` needs no method, because the caller never calls it.

### Arguments

  - `alg`: The concrete subtype instance.
  - `W0`: First-pass weights `observations × assets`.
  - `eps`: First-pass residual matrix `observations × assets`.
  - `ve`: Variance estimator.
  - `mask`: Eligibility mask `observations × assets`.

### Returns

  - `W1::Matrix{<:Number}`: Second-pass weights `observations × assets`, zero outside `mask`.

## `needs_second_pass`

  - [`needs_second_pass(alg::AbstractCrossSectionalWeightsAlgorithm)`](@ref): Returns `true` when the caller must fit twice. The root answers `false`, so a one-pass member needs no method.

### Arguments

  - `alg`: The concrete subtype instance.

### Returns

  - `val::Bool`: `true` when [`cs_weights_refine`](@ref) is called, `false` otherwise.

### Examples

```jldoctest
julia> struct MyWeights <: PortfolioOptimisers.AbstractCrossSectionalWeightsAlgorithm end

julia> PortfolioOptimisers.needs_second_pass(MyWeights())
false
```

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`MarketCapWeights`](@ref)
  - [`BlendedInverseVarianceWeights`](@ref)
  - [`cross_sectional_regression`](@ref)
"""
abstract type AbstractCrossSectionalWeightsAlgorithm <: AbstractAlgorithm end
"""
    needs_second_pass(alg::AbstractCrossSectionalWeightsAlgorithm) -> Bool

Return whether a weight policy asks the caller to fit the cross-sectional regression twice.

The trait keeps the fit loop in the caller, which reads top to bottom, so the weight family and the regression family never name each other's types. It follows [`needs_previous_weights`](@ref), which the tracking family uses the same way.

# Arguments

  - `alg`: Cross-sectional weight policy.

# Returns

  - `val::Bool`: `true` when the caller must run [`cs_weights_refine`](@ref) and fit again, `false` when the first-pass weights are the answer.

# Examples

```jldoctest
julia> PortfolioOptimisers.needs_second_pass(MarketCapWeights())
false

julia> PortfolioOptimisers.needs_second_pass(BlendedInverseVarianceWeights(; lambda = 0.5))
true
```

# Related

  - [`AbstractCrossSectionalWeightsAlgorithm`](@ref)
  - [`MarketCapWeights`](@ref)
  - [`BlendedInverseVarianceWeights`](@ref)
  - [`cs_weights_refine`](@ref)
  - [`needs_previous_weights`](@ref)
"""
function needs_second_pass(::AbstractCrossSectionalWeightsAlgorithm)::Bool
    return false
end
"""
$(DocStringExtensions.TYPEDEF)

Weights an asset by a power of its market capitalisation, in one pass.

The power sets how concentrated the cross-section is: `0` gives every eligible asset the same weight, `1` gives raw capitalisation weights, and a value between the two shrinks the concentration of the largest assets.

# Mathematical definition

```math
\\begin{align}
w_{t,i} &= \\begin{cases} m_{t,i}^{p} & \\text{if } (t, i) \\in \\mathcal{M} \\\\ 0 & \\text{otherwise} \\end{cases}\\,.
\\end{align}
```

Where:

  - ``w_{t,i}``: Weight of asset ``i`` at observation ``t``.
  - ``m_{t,i}``: Market capitalisation of asset ``i`` at observation ``t``.
  - ``p``: Market capitalisation power.
  - ``\\mathcal{M}``: Eligibility mask.

The weights are relative, so they need not sum to one. A weighted least squares is invariant to their scale.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MarketCapWeights(; p::Real = 0.5) -> MarketCapWeights

Keywords correspond to the struct's fields.

## Validation

  - `isfinite(p)`.
  - `p >= 0`.

# Examples

```jldoctest
julia> MarketCapWeights()
MarketCapWeights
  p ┴ Float64: 0.5
```

# Related

  - [`AbstractCrossSectionalWeightsAlgorithm`](@ref)
  - [`BlendedInverseVarianceWeights`](@ref)
  - [`cs_weights_initial`](@ref)
  - [`cross_sectional_cap_weights`](@ref)
"""
@concrete struct MarketCapWeights <: AbstractCrossSectionalWeightsAlgorithm
    """
    Exponent applied to the market capitalisation. A value of `0` needs no capitalisation matrix, because every eligible asset takes the same weight.
    """
    p
    function MarketCapWeights(p::Real)
        assert_finite(p, :p)
        assert_nonneg(p, :p)
        return new{typeof(p)}(p)
    end
end
function MarketCapWeights(; p::Real = 0.5)::MarketCapWeights
    return MarketCapWeights(p)
end
"""
$(DocStringExtensions.TYPEDEF)

Blends market capitalisation weights with inverse idiosyncratic variance weights, in two passes.

The blend is a two-step feasible generalised least squares: a first fit under capitalisation weights gives the residuals, a variance estimated from those residuals gives the second component, and the caller fits again. The variance paired with observation `t` reads residuals up to `t - 1` only, so a weight never reads the return it multiplies.

# Mathematical definition

```math
\\begin{align}
u_{t,i} &= \\min\\left(\\mathrm{clip}\\left(\\frac{1}{v_{t-1,i}},\\, q_{t}^{\\mathrm{lo}},\\, q_{t}^{\\mathrm{hi}}\\right),\\, \\rho \\, \\mathrm{med}_{j}\\left(u_{t,j}\\right)\\right) \\\\
w_{t,i} &= \\lambda \\frac{u_{t,i}}{\\sum_{j} u_{t,j}} + \\left(1 - \\lambda\\right) \\frac{m_{t,i}^{p}}{\\sum_{j} m_{t,j}^{p}}\\,.
\\end{align}
```

Where:

  - ``w_{t,i}``: Weight of asset ``i`` at observation ``t``.
  - ``v_{t-1,i}``: Idiosyncratic variance of asset ``i`` after observation ``t - 1``.
  - ``q_{t}^{\\mathrm{lo}}``, ``q_{t}^{\\mathrm{hi}}``: Cross-sectional quantiles of the inverse variances at the levels `wins`.
  - ``\\rho``: Median ratio cap.
  - ``\\lambda``: Blend coefficient.
  - ``m_{t,i}``: Market capitalisation of asset ``i`` at observation ``t``.
  - ``p``: Market capitalisation power.

Both components are normalised over the same eligible set before the blend, so the realised blend equals the nominal ``\\lambda``. An observation with no variance estimate yet takes the capitalisation weights alone.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BlendedInverseVarianceWeights(;
        p::Real = 0.5,
        lambda::Real,
        ratio::Real = 20.0,
        wins::Tuple{<:Real, <:Real} = (0.025, 0.975)
    ) -> BlendedInverseVarianceWeights

Keywords correspond to the struct's fields. `lambda` takes no default, because naming this member is the request for the second pass. A caller that wants one pass writes [`MarketCapWeights`](@ref) instead.

## Validation

  - `isfinite(p)` and `p >= 0`.
  - `0 <= lambda <= 1`.
  - `isfinite(ratio)` and `ratio > 0`.
  - `0 <= wins[1] < wins[2] <= 1`.

# Examples

```jldoctest
julia> BlendedInverseVarianceWeights(; lambda = 0.5)
BlendedInverseVarianceWeights
       p ┼ Float64: 0.5
  lambda ┼ Float64: 0.5
   ratio ┼ Float64: 20.0
    wins ┴ Tuple{Float64, Float64}: (0.025, 0.975)
```

# Related

  - [`AbstractCrossSectionalWeightsAlgorithm`](@ref)
  - [`MarketCapWeights`](@ref)
  - [`cs_weights_initial`](@ref)
  - [`cs_weights_refine`](@ref)
  - [`variance_series`](@ref)
"""
@concrete struct BlendedInverseVarianceWeights <: AbstractCrossSectionalWeightsAlgorithm
    """
    Exponent applied to the market capitalisation of the first blend component. A value of `0` needs no capitalisation matrix, because every eligible asset takes the same weight.
    """
    p
    """
    Blend coefficient. It is the share of the inverse variance component in the second-pass weights, so `0` recovers the capitalisation weights and `1` drops them.
    """
    lambda
    """
    Largest multiple of the cross-sectional median an inverse variance weight may take. It caps the influence of an asset whose estimated variance is very small.
    """
    ratio
    """
    Lower and upper quantile levels of the cross-sectional winsorisation of the inverse variances, applied before the median cap.
    """
    wins
    function BlendedInverseVarianceWeights(p::Real, lambda::Real, ratio::Real,
                                           wins::Tuple{<:Real, <:Real})
        assert_finite(p, :p)
        assert_nonneg(p, :p)
        assert_closed_unit_interval(lambda, :lambda)
        assert_finite(ratio, :ratio)
        assert_gt0(ratio, :ratio)
        @argcheck(zero(wins[1]) <= wins[1] < wins[2] <= one(wins[2]),
                  DomainError(wins,
                              "wins must satisfy 0 <= wins[1] < wins[2] <= 1, got $(wins)"))
        return new{typeof(p), typeof(lambda), typeof(ratio), typeof(wins)}(p, lambda, ratio,
                                                                           wins)
    end
end
function BlendedInverseVarianceWeights(; p::Real = 0.5, lambda::Real, ratio::Real = 20.0,
                                       wins::Tuple{<:Real, <:Real} = (0.025, 0.975))::BlendedInverseVarianceWeights
    return BlendedInverseVarianceWeights(p, lambda, ratio, wins)
end
function needs_second_pass(::BlendedInverseVarianceWeights)::Bool
    return true
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the market capitalisation weights of an eligibility mask, raised to a power.

The two members of the family share this body, so the formula is written once. A power of zero short-circuits, because the capitalisation matrix is then unread and a caller that owns no capitalisation data may pass `nothing`.

# Algorithm

 1. When `p` is zero, return the mask as a floating point matrix, giving every eligible pair the weight one.
 2. Otherwise check `mcap` against the mask, and write `mcap[t, i]^p` into every eligible pair, leaving zero elsewhere.

# Arguments

  - `p::Real`: Exponent applied to the market capitalisation.
  - `mcap::Option{<:MatNum}`: Market capitalisation matrix `observations × assets`, or `nothing` when `p` is zero.
  - `mask::AbstractMatrix{Bool}`: Eligibility mask `observations × assets`.

# Validation

  - `!isempty(mask)`.
  - `!isnothing(mcap)` when `p` is not zero.
  - `size(mcap) == size(mask)`.
  - Every eligible pair carries a finite, non-negative capitalisation. The `DomainError` names the observation and the asset.

# Returns

  - `W0::Matrix{<:Number}`: Weights `observations × assets`, zero outside `mask`.

# Examples

```jldoctest
julia> mask = [true true false];

julia> PortfolioOptimisers.cross_sectional_cap_weights(0.5, [4.0 9.0 1.0], mask)
1×3 Matrix{Float64}:
 2.0  3.0  0.0

julia> PortfolioOptimisers.cross_sectional_cap_weights(0.0, nothing, mask)
1×3 Matrix{Float64}:
 1.0  1.0  0.0
```

# Related

  - [`MarketCapWeights`](@ref)
  - [`BlendedInverseVarianceWeights`](@ref)
  - [`cs_weights_initial`](@ref)
"""
function cross_sectional_cap_weights(p::Real, mcap::Option{<:MatNum},
                                     mask::AbstractMatrix{Bool})::MatNum
    @argcheck(!isempty(mask), IsEmptyError("mask cannot be empty"))
    if iszero(p)
        return Matrix{float(typeof(p))}(mask)
    end
    @argcheck(!isnothing(mcap),
              IsNothingError("mcap cannot be nothing when p is not zero, got p = $(p)"))
    @argcheck(size(mcap) == size(mask),
              DimensionMismatch("mcap ($(size(mcap, 1))×$(size(mcap, 2))) must match mask ($(size(mask, 1))×$(size(mask, 2)))"))
    Tf = promote_type(float(real(eltype(mcap))), float(typeof(p)))
    W0 = zeros(Tf, size(mask))
    for t in axes(mask, 1), i in axes(mask, 2)
        if mask[t, i]
            @argcheck(isfinite(mcap[t, i]) && mcap[t, i] >= zero(eltype(mcap)),
                      DomainError(mcap[t, i],
                                  "observation $t and asset $i are eligible, so their market capitalisation must be finite and >= 0"))
            W0[t, i] = mcap[t, i]^p
        end
    end
    return W0
end
"""
    cs_weights_initial(alg::MarketCapWeights, mcap::Option{<:MatNum},
                       mask::AbstractMatrix{Bool}) -> Matrix{<:Number}
    cs_weights_initial(alg::BlendedInverseVarianceWeights, mcap::Option{<:MatNum},
                       mask::AbstractMatrix{Bool}) -> Matrix{<:Number}

Return the first-pass cross-sectional regression weights of a weight policy.

Both members read the capitalisation the same way, so both call [`cross_sectional_cap_weights`](@ref). They part on what the caller does next: [`needs_second_pass`](@ref) is `false` for one and `true` for the other.

# Arguments

  - `alg`: Cross-sectional weight policy.
  - `mcap::Option{<:MatNum}`: Market capitalisation matrix `observations × assets`, or `nothing` when the policy's `p` is zero.
  - `mask::AbstractMatrix{Bool}`: Eligibility mask `observations × assets`.

# Validation

  - The rules of [`cross_sectional_cap_weights`](@ref).

# Returns

  - `W0::Matrix{<:Number}`: First-pass weights `observations × assets`, zero outside `mask`.

# Examples

```jldoctest
julia> PortfolioOptimisers.cs_weights_initial(MarketCapWeights(; p = 1.0), [2.0 3.0 5.0],
                                              [true true false])
1×3 Matrix{Float64}:
 2.0  3.0  0.0
```

# Related

  - [`AbstractCrossSectionalWeightsAlgorithm`](@ref)
  - [`MarketCapWeights`](@ref)
  - [`BlendedInverseVarianceWeights`](@ref)
  - [`cross_sectional_cap_weights`](@ref)
  - [`cs_weights_refine`](@ref)
"""
function cs_weights_initial(alg::MarketCapWeights, mcap::Option{<:MatNum},
                            mask::AbstractMatrix{Bool})::MatNum
    return cross_sectional_cap_weights(alg.p, mcap, mask)
end
function cs_weights_initial(alg::BlendedInverseVarianceWeights, mcap::Option{<:MatNum},
                            mask::AbstractMatrix{Bool})::MatNum
    return cross_sectional_cap_weights(alg.p, mcap, mask)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the lagged inverse idiosyncratic variances of a residual matrix.

Row `t` reads the variance series at row `t - 1`, so an entry never carries the squared residual of its own observation. Row `1` is `NaN`, because no residual precedes the first observation.

# Algorithm

 1. Take the variance series of `eps` through [`variance_series`](@ref).
 2. Fill the result with `NaN`, which is what row `1` and every pair outside `mask` keep. Both blend components then normalise over the same universe.
 3. Write the reciprocal of row `t - 1` of the series into every eligible pair of row `t`, for every later observation.

# Arguments

  - `ve`: Variance estimator.
  - `eps::MatNum`: First-pass residual matrix `observations × assets`.
  - `mask::AbstractMatrix{Bool}`: Eligibility mask `observations × assets`.
  - `kwargs...`: Additional keyword arguments passed to [`variance_series`](@ref).

# Validation

  - `!isempty(eps)`.
  - `size(eps) == size(mask)`.

# Returns

  - `IV::Matrix{<:Number}`: Inverse variances `observations × assets`, `NaN` where no estimate exists and outside `mask`.

# Examples

```jldoctest
julia> eps = [1.0 2.0; 3.0 6.0; 2.0 4.0];

julia> PortfolioOptimisers.cross_sectional_lagged_inverse_variance(SimpleVariance(), eps,
                                                                   trues(3, 2))
3×2 Matrix{Float64}:
 NaN    NaN
 NaN    NaN
   0.5    0.125
```

# Related

  - [`BlendedInverseVarianceWeights`](@ref)
  - [`cs_weights_refine`](@ref)
  - [`variance_series`](@ref)
"""
function cross_sectional_lagged_inverse_variance(ve::AbstractCovarianceEstimator,
                                                 eps::MatNum, mask::AbstractMatrix{Bool};
                                                 kwargs...)::MatNum
    @argcheck(!isempty(eps), IsEmptyError("eps cannot be empty"))
    @argcheck(size(eps) == size(mask),
              DimensionMismatch("eps ($(size(eps, 1))×$(size(eps, 2))) must match mask ($(size(mask, 1))×$(size(mask, 2)))"))
    V = variance_series(ve, eps; dims = 1, kwargs...)
    Tf = float(real(eltype(V)))
    IV = fill(Tf(NaN), size(eps))
    for t in 2:size(eps, 1), i in axes(eps, 2)
        if mask[t, i]
            IV[t, i] = inv(V[t - 1, i])
        end
    end
    return IV
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Clamp the inverse variance weights of each observation to two cross-sectional quantiles, in place.

The clamp protects the blend from an asset whose estimated variance is very small, which would otherwise carry almost the whole cross-section. The quantiles come from the estimation universe, which is the set of pairs with a positive first-pass weight, and the clamp reaches the whole row, so a pair outside that universe still leaves inside the same bounds.

# Algorithm

 1. For each observation, gather the entries in the estimation universe that are not `NaN`.
 2. Take the quantiles `wins` of that set, and clamp the whole row between them.
 3. Write an all-`NaN` row where the set carries no finite entry, because no bound exists there.

# Arguments

  - `IV::Matrix{<:Number}`: Inverse variances `observations × assets`, changed in place.
  - `W0::MatNum`: First-pass weights `observations × assets`. Only the sign of an entry is read, which names the estimation universe.
  - `wins::Tuple{<:Real, <:Real}`: Lower and upper quantile levels of the clamp.

# Validation

  - `size(IV) == size(W0)`.

# Returns

  - `nothing`. `IV` carries the clamped weights.

# Examples

```jldoctest
julia> IV = [1.0 3.0 100.0];

julia> PortfolioOptimisers.cross_sectional_winsorise!(IV, [1.0 1.0 0.0], (0.0, 1.0))

julia> IV
1×3 Matrix{Float64}:
 1.0  3.0  3.0
```

# Related

  - [`BlendedInverseVarianceWeights`](@ref)
  - [`cs_weights_refine`](@ref)
  - [`cross_sectional_median_cap!`](@ref)
"""
function cross_sectional_winsorise!(IV::Matrix{<:Number}, W0::MatNum,
                                    wins::Tuple{<:Real, <:Real})::Nothing
    @argcheck(size(IV) == size(W0),
              DimensionMismatch("IV ($(size(IV, 1))×$(size(IV, 2))) must match W0 ($(size(W0, 1))×$(size(W0, 2)))"))
    Tf = eltype(IV)
    for t in axes(IV, 1)
        row = view(IV, t, :)
        u = [x for (x, w) in zip(row, view(W0, t, :)) if w > zero(eltype(W0)) && !isnan(x)]
        if any(isfinite, u)
            row .= clamp.(row, Statistics.quantile(u, wins[1]),
                          Statistics.quantile(u, wins[2]))
        else
            row .= Tf(NaN)
        end
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Cap the inverse variance weights of each observation at a multiple of their median, then normalise them, in place.

The cap is the second of the two bounds, and it is what answers an asset whose estimated variance is exactly zero: its entry is infinite, and it leaves at `ratio` times the median, so the observation still returns a weight. The normalisation puts the row on the same scale as the capitalisation component, so the blend of the two realises the shrinkage the caller wrote.

# Algorithm

 1. For each observation, record whether the row carries a finite entry.
 2. Take the median of the entries that are not `NaN`, and cap the row at `ratio` times it.
 3. Divide the row by the sum of its entries that are not `NaN`.

# Arguments

  - `IV::Matrix{<:Number}`: Inverse variances `observations × assets`, changed in place.
  - `ratio::Real`: Largest multiple of the cross-sectional median an entry may take.

# Returns

  - `ready::BitVector`: One entry per observation, `true` where the row carries a usable inverse variance estimate. A row that answers `false` is untouched.

# Examples

```jldoctest
julia> IV = [1.0 3.0 100.0];

julia> PortfolioOptimisers.cross_sectional_median_cap!(IV, 2.0)
1-element BitVector:
 1

julia> IV
1×3 Matrix{Float64}:
 0.1  0.3  0.6
```

# Related

  - [`BlendedInverseVarianceWeights`](@ref)
  - [`cs_weights_refine`](@ref)
  - [`cross_sectional_winsorise!`](@ref)
"""
function cross_sectional_median_cap!(IV::Matrix{<:Number}, ratio::Real)::BitVector
    ready = falses(size(IV, 1))
    for t in axes(IV, 1)
        row = view(IV, t, :)
        ready[t] = any(isfinite, row)
        if !ready[t]
            continue
        end
        row .= min.(row, Statistics.median([x for x in row if !isnan(x)]) * ratio)
        row ./= sum([x for x in row if !isnan(x)])
    end
    return ready
end
"""
    cs_weights_refine(alg::BlendedInverseVarianceWeights, W0::MatNum, eps::MatNum,
                      ve::AbstractCovarianceEstimator, mask::AbstractMatrix{Bool};
                      kwargs...) -> Matrix{<:Number}

Return the second-pass cross-sectional regression weights of a weight policy.

The verb takes the first-pass weights and residuals as arguments rather than reading them off the policy, because a weight policy is configuration and holds no fitted data. A caller runs it only when [`needs_second_pass`](@ref) answers `true`. An observation whose eligible first-pass weights sum to zero returns `NaN` rather than a weight, because both components are normalised. An eligible pair carries a positive weight whenever `alg.p` is zero or its market capitalisation is positive, so a caller reaches that state only by handing in a weight matrix that [`cs_weights_initial`](@ref) did not build.

# Algorithm

 1. Take the lagged inverse variances through [`cross_sectional_lagged_inverse_variance`](@ref).
 2. Clamp them to the cross-sectional quantiles through [`cross_sectional_winsorise!`](@ref).
 3. Cap and normalise them through [`cross_sectional_median_cap!`](@ref), which also names the observations that carry an estimate.
 4. Normalise the first-pass weights over the same universe.
 5. Write zero over every entry that is still `NaN`, so a missing entry does not contribute.
 6. Write the normalised first-pass weights into every observation that carries no estimate.
 7. Blend the two components by `alg.lambda`, and write zero outside `mask`.

# Arguments

  - `alg`: Cross-sectional weight policy.
  - `W0::MatNum`: First-pass weights `observations × assets`.
  - `eps::MatNum`: First-pass residual matrix `observations × assets`.
  - `ve`: Variance estimator.
  - `mask::AbstractMatrix{Bool}`: Eligibility mask `observations × assets`.
  - `kwargs...`: Additional keyword arguments passed to [`variance_series`](@ref).

# Validation

  - `size(W0) == size(eps)`.
  - `all(isfinite, W0)` and `all(x -> x >= 0, W0)`.
  - The rules of [`cross_sectional_lagged_inverse_variance`](@ref), which hold `eps` against `mask`.

# Returns

  - `W1::Matrix{<:Number}`: Second-pass weights `observations × assets`, zero outside `mask`.

# Examples

```jldoctest
julia> eps = [1.0 2.0; 3.0 6.0; 2.0 4.0];

julia> alg = BlendedInverseVarianceWeights(; p = 0.0, lambda = 1.0);

julia> W0 = PortfolioOptimisers.cs_weights_initial(alg, nothing, trues(3, 2));

julia> PortfolioOptimisers.cs_weights_refine(alg, W0, eps, SimpleVariance(), trues(3, 2))
3×2 Matrix{Float64}:
 0.5    0.5
 0.5    0.5
 0.785  0.215
```

# Related

  - [`AbstractCrossSectionalWeightsAlgorithm`](@ref)
  - [`BlendedInverseVarianceWeights`](@ref)
  - [`cs_weights_initial`](@ref)
  - [`needs_second_pass`](@ref)
  - [`cross_sectional_lagged_inverse_variance`](@ref)
  - [`cross_sectional_winsorise!`](@ref)
  - [`cross_sectional_median_cap!`](@ref)
"""
function cs_weights_refine(alg::BlendedInverseVarianceWeights, W0::MatNum, eps::MatNum,
                           ve::AbstractCovarianceEstimator, mask::AbstractMatrix{Bool};
                           kwargs...)::MatNum
    @argcheck(size(W0) == size(eps),
              DimensionMismatch("W0 ($(size(W0, 1))×$(size(W0, 2))) must match eps ($(size(eps, 1))×$(size(eps, 2)))"))
    @argcheck(all(isfinite, W0), IsNonFiniteError("all entries of W0 must be finite"))
    @argcheck(all(x -> x >= zero(x), W0), DomainError(W0, "all entries of W0 must be >= 0"))
    IV = cross_sectional_lagged_inverse_variance(ve, eps, mask; kwargs...)
    cross_sectional_winsorise!(IV, W0, alg.wins)
    ready = cross_sectional_median_cap!(IV, alg.ratio)
    Tf = promote_type(eltype(IV), float(real(eltype(W0))), float(typeof(alg.lambda)))
    Wm = Tf.(W0)
    Wm ./= sum(Wm; dims = 2)
    U = Tf.(IV)
    U[isnan.(U)] .= zero(Tf)
    U[.!ready, :] = Wm[.!ready, :]
    W1 = alg.lambda .* U .+ (one(Tf) - alg.lambda) .* Wm
    W1[.!mask] .= zero(Tf)
    return W1
end

export MarketCapWeights, BlendedInverseVarianceWeights
