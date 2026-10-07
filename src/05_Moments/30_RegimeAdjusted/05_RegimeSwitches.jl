"""
$(DocStringExtensions.TYPEDEF)

Divides the regime statistic by the exact bias that its method reads in the inverse of an estimated variance. This is the default debias rule.

The inverse of an estimated variance is too large on average (Jensen's inequality), so a raw statistic has a mean above the one the method assumes. Each method reads its own moment of the ratio ``Q = \\hat{v} / \\sigma^2``: the root mean square reads ``\\mathbb{E}[1 / Q]``, the first moment reads ``\\mathbb{E}[Q^{-1/2}]^2`` and the log reads ``\\exp(-\\mathbb{E}[\\ln Q])``. [`regime_bias_table`](@ref) computes each one exactly from the law of the estimate. The statistic skips an estimate too young for its variance to be finite.

Under [`EstimatedCentring`](@ref) the deviation that the statistic reads shares its location with the terms of the estimate, so the two are not independent, and each method reads the moment of the deviation and the estimate together: ``\\mathbb{E}[u^{2} / Q]`` for the root mean square, and so on. This rule reads that dependence for every method and every target. [`LawDebias`](@ref) reads the law of the estimate alone.

# Constructors

    ExactDebias() -> ExactDebias

# Examples

```jldoctest
julia> RegimeAdjustedExpWeightedVariance().debias
ExactDebias()
```

# Related

  - [`AbstractRegimeDebias`](@ref)
  - [`LawDebias`](@ref)
  - [`RawStatistic`](@ref)
  - [`regime_bias_table`](@ref)
"""
struct ExactDebias <: AbstractRegimeDebias end
"""
$(DocStringExtensions.TYPEDEF)

Divides the regime statistic by the bias of the law of the estimate alone, without the dependence of the deviation on the estimate.

The factor of each method is the one of [`ExactDebias`](@ref) with the deviation taken as independent of the estimate. Under [`PreCentred`](@ref) the two rules agree. Under [`EstimatedCentring`](@ref) the deviation shares its location with the terms of the estimate, and this rule omits that part: on the scalar estimator the root mean square reads 0.2 % low at a half-life of 10 and two lags, and on the Mahalanobis target at 5 to 12 assets the three methods read 0.2 % to 0.7 % high without HAC and 0.4 % to 1.4 % high with it. The Mahalanobis nodes cost about half of those of [`ExactDebias`](@ref) for the first moment and the log, and a third for the root mean square.

# Constructors

    LawDebias() -> LawDebias

# Examples

```jldoctest
julia> RegimeAdjustedExpWeightedCovariance(; debias = LawDebias()).debias
LawDebias()
```

# Related

  - [`AbstractRegimeDebias`](@ref)
  - [`ExactDebias`](@ref)
  - [`reads_dependence`](@ref)
"""
struct LawDebias <: AbstractRegimeDebias end
"""
$(DocStringExtensions.TYPEDEF)

Scores the raw regime statistic, with no correction of the bias of the estimated variance it reads.

Every observation above `min_obs` is scored. The mean of the statistic is above the one the method assumes, by 7 % at a half-life of 10 and 1.7 % at a half-life of 40 for the root mean square, so the regime multiplier is too large on average.

# Constructors

    RawStatistic() -> RawStatistic

# Examples

```jldoctest
julia> RegimeAdjustedExpWeightedVariance(; debias = RawStatistic()).debias
RawStatistic()
```

# Related

  - [`AbstractRegimeDebias`](@ref)
  - [`ExactDebias`](@ref)
  - [`LawDebias`](@ref)
"""
struct RawStatistic <: AbstractRegimeDebias end
"""
    debiases(::Union{ExactDebias, LawDebias}) -> Bool
    debiases(::RawStatistic) -> Bool

Whether the regime statistic corrects the bias of the estimate it reads.

# Returns

  - `flag::Bool`: `true` for [`ExactDebias`](@ref) and [`LawDebias`](@ref), `false` for
    [`RawStatistic`](@ref).

# Related

  - [`AbstractRegimeDebias`](@ref)
  - [`reads_dependence`](@ref)
  - [`regime_bias_open`](@ref)
"""
function debiases(::Union{ExactDebias, LawDebias})
    return true
end
function debiases(::RawStatistic)
    return false
end
"""
    reads_dependence(::ExactDebias) -> Bool
    reads_dependence(::Union{LawDebias, RawStatistic}) -> Bool

Whether the bias of the regime statistic reads the dependence of the deviation on the estimate,
which an estimated location puts in.

# Returns

  - `flag::Bool`: `true` for [`ExactDebias`](@ref), `false` for [`LawDebias`](@ref) and
    [`RawStatistic`](@ref).

# Related

  - [`AbstractRegimeDebias`](@ref)
  - [`debiases`](@ref)
  - [`regime_bias_table`](@ref)
"""
function reads_dependence(::ExactDebias)
    return true
end
function reads_dependence(::Union{LawDebias, RawStatistic})
    return false
end
"""
$(DocStringExtensions.TYPEDEF)

Lets a HAC-adjusted product enter the variance recursion as it is, negative or not. This is the default floor rule.

A HAC term adds the lagged products of the deviations to the square, so it can be negative. Under returns with no autocorrelation each term has the mean of the variance, and the recursion is an unbiased quadratic form. Only the variance that the estimator returns is floored at zero.

# Constructors

    NoHacFloor() -> NoHacFloor

# Examples

```jldoctest
julia> RegimeAdjustedExpWeightedVariance().hac_floor
NoHacFloor()
```

# Related

  - [`AbstractHacFloor`](@ref)
  - [`PerTermHacFloor`](@ref)
"""
struct NoHacFloor <: AbstractHacFloor end
"""
$(DocStringExtensions.TYPEDEF)

Floors each HAC-adjusted square at zero before it enters the variance recursion.

The floor removes the negative part of each term, so the mean of a term is above the variance. Under returns with no autocorrelation the variance is 6.7 %, 16 % and 41 % too large at one, two and five lags.

# Constructors

    PerTermHacFloor() -> PerTermHacFloor

# Examples

```jldoctest
julia> RegimeAdjustedExpWeightedVariance(; hac_floor = PerTermHacFloor()).hac_floor
PerTermHacFloor()
```

# Related

  - [`AbstractHacFloor`](@ref)
  - [`NoHacFloor`](@ref)
"""
struct PerTermHacFloor <: AbstractHacFloor end
"""
    hac_floor!(::NoHacFloor, X2::VecNum, valid::AbstractVector{<:Bool})
    hac_floor!(::PerTermHacFloor, X2::VecNum, valid::AbstractVector{<:Bool})

Applies the floor rule to the HAC-adjusted squares of the valid assets.

# Arguments

  - `X2`: The HAC-adjusted squares. The method writes it in place.
  - `valid`: Mask of the valid assets.

# Returns

  - `X2`: The squares, unchanged under [`NoHacFloor`](@ref), and floored at zero on the valid assets under [`PerTermHacFloor`](@ref).

# Related

  - [`AbstractHacFloor`](@ref)
"""
function hac_floor!(::NoHacFloor, X2::VecNum, ::AbstractVector{<:Bool})
    return X2
end
function hac_floor!(::PerTermHacFloor, X2::VecNum, valid::AbstractVector{<:Bool})
    X2[valid] .= max.(view(X2, valid), zero(eltype(X2)))
    return X2
end
"""
$(DocStringExtensions.TYPEDEF)

Divides each row of the separate correlation path under HAC by the volatility before the variance takes the row. This is the default rule.

Under HAC the diagonal of a row can be negative. A volatility that already holds the row damps a positive row and amplifies a negative one, so the diagonal of the correlation state is skewed down and can fall to zero. The volatility before the step is independent of the row. At two lags the rule after the step has 9 % to 19 % more mean square error.

# Constructors

    VolatilityBeforeUpdate() -> VolatilityBeforeUpdate

# Examples

```jldoctest
julia> RegimeAdjustedExpWeightedCovariance().hac_vol_before
VolatilityBeforeUpdate()
```

# Related

  - [`AbstractHacVolatilityTiming`](@ref)
  - [`VolatilityAfterUpdate`](@ref)
"""
struct VolatilityBeforeUpdate <: AbstractHacVolatilityTiming end
"""
$(DocStringExtensions.TYPEDEF)

Divides each row of the separate correlation path under HAC by the volatility after the variance takes the row.

This is the rule of the path without HAC, where it is the better one. Under HAC it skews the diagonal of the correlation state down.

# Constructors

    VolatilityAfterUpdate() -> VolatilityAfterUpdate

# Examples

```jldoctest
julia> RegimeAdjustedExpWeightedCovariance(; hac_vol_before = VolatilityAfterUpdate()).hac_vol_before
VolatilityAfterUpdate()
```

# Related

  - [`AbstractHacVolatilityTiming`](@ref)
  - [`VolatilityBeforeUpdate`](@ref)
"""
struct VolatilityAfterUpdate <: AbstractHacVolatilityTiming end
"""
    volatility_before_update(::VolatilityBeforeUpdate) -> Bool
    volatility_before_update(::VolatilityAfterUpdate) -> Bool

Whether a row of the separate correlation path under HAC is divided by the volatility before the variance takes the row.

# Returns

  - `flag::Bool`: `true` for [`VolatilityBeforeUpdate`](@ref), `false` for [`VolatilityAfterUpdate`](@ref).

# Related

  - [`AbstractHacVolatilityTiming`](@ref)
"""
function volatility_before_update(::VolatilityBeforeUpdate)
    return true
end
function volatility_before_update(::VolatilityAfterUpdate)
    return false
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the outer product of one observation, with the Newey-West HAC correction where `hac_lags` is not `nothing`, and divides each product by its exact factor.

The correction adds each lagged cross-product and its transpose, weighted by the Bartlett kernel ``w_j = 1 - j/(L+1)``, so the update reads the serial correlation of the returns. An entry of a lagged observation that is not finite is read as zero, which freezes that pair's contribution. Each product is then divided by [`hac_pair_factor`](@ref), its mean in units of the covariance of the pair, which is one under [`PreCentred`](@ref).

# Arguments

  - `cache::RegimeAdjustedCovarianceState`: Online covariance computation cache (mutated).
  - `ce::RegimeAdjustedExpWeightedCovariance`: Covariance estimator configuration.
  - `X::VecNum`: The deviations of the current observation, zero where an asset gives none.
  - `dvalid::AbstractVector{<:Bool}`: Mask of the assets whose deviation exists.

# Returns

  - `XXt::MatNum`: The HAC-adjusted outer product stored in `cache.XXt`.

# Related

  - [`RegimeAdjustedCovarianceState`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`hac_squared_returns!`](@ref)
  - [`hac_pair_factor`](@ref)
"""
function hac_outer_product!(cache::RegimeAdjustedCovarianceState,
                            ce::RegimeAdjustedExpWeightedCovariance, X::VecNum,
                            dvalid::AbstractVector{<:Bool})
    cache.XXt .= X .* transpose(X)
    if !(isnothing(cache.ret_buffer) || isempty(cache.ret_buffer))
        for (i, X_old) in enumerate(Iterators.reverse(cache.ret_buffer))
            wi = one(eltype(X)) - i / (ce.hac_lags + 1)
            cache.X_old_i .= replace(X_old, NaN => zero(eltype(X_old)))
            cross = X .* transpose(cache.X_old_i)
            cache.XXt .+= wi * (cross + transpose(cross))
        end
    end
    cache.XXt ./= hac_pair_factor(cache, ce, dvalid)

    return cache.XXt
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Exact factor of each product of the regime covariance at the current observation, in units of the covariance of the pair.

The product is the outer product of the deviations, with the HAC products where `hac_lags` is set. Its mean, for returns that are independent in time, is the covariance of the pair times [`centring_pair_factor`](@ref), plus [`centring_lag_factor`](@ref) once the buffer holds a lagged observation. Under [`PreCentred`](@ref) the factor is one.

# Arguments

  - `cache`: Online covariance computation cache, before the observation folds into its counts and overlap.
  - `ce`: Covariance estimator configuration.
  - `dvalid`: Mask of the assets whose deviation exists at the observation.

# Returns

  - `F::Union{<:Number, <:MatNum}`: The `N × N` matrix of the factors, or one.

# Related

  - [`hac_outer_product!`](@ref)
  - [`centring_pair_factor`](@ref)
  - [`centring_lag_factor`](@ref)
"""
function hac_pair_factor(cache::RegimeAdjustedCovarianceState,
                         ce::RegimeAdjustedExpWeightedCovariance,
                         dvalid::AbstractVector{<:Bool})
    F = centring_pair_factor(cache.overlap, cache.obs_count, ce.decay,
                             axes(cache.covariance, 1))
    if isnothing(cache.ret_buffer) || isempty(cache.ret_buffer)
        return F
    end
    w = l -> one(ce.decay) - l / (ce.hac_lags + 1)
    return F .+ centring_lag_factor(ce.centring, ce.decay, cache.obs_count, dvalid,
                                    cache.lag_records, w, false)
end
export ExactDebias, LawDebias, RawStatistic, NoHacFloor, PerTermHacFloor,
       VolatilityBeforeUpdate, VolatilityAfterUpdate
public debiases, reads_dependence, hac_floor!, volatility_before_update
