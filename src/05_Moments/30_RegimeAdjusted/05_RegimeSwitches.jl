"""
$(DocStringExtensions.TYPEDEF)

Divides the regime statistic by the exact bias that its method reads in the inverse of an estimated variance. This is the default debias rule.

The inverse of an estimated variance is too large on average (Jensen's inequality), so a raw statistic has a mean above the one the method assumes. Each method reads its own moment of the ratio ``Q = \\hat{v} / \\sigma^2``: the root mean square reads ``\\mathbb{E}[1 / Q]``, the first moment reads ``\\mathbb{E}[Q^{-1/2}]^2`` and the log reads ``\\exp(-\\mathbb{E}[\\ln Q])``. [`regime_bias_table`](@ref) computes each one exactly from the law of the estimate. The statistic skips an estimate too young for its variance to be finite.

Under [`EstimatedCentring`](@ref) the deviation that the statistic reads shares its location with the terms of the estimate, so the two are not independent, and each method reads the moment of the deviation and the estimate together: ``\\mathbb{E}[u^{2} / Q]`` for the root mean square, and so on. This rule reads that dependence for every method, for the scalar estimator and the diagonal target on every path, and for the portfolio and the Mahalanobis targets on one decay. On the separate correlation path the portfolio and the Mahalanobis targets read the pre-centred law at `cor_decay`. [`LawDebias`](@ref) reads the law of the estimate alone.

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
    debiases(::AbstractRegimeDebias) -> Bool
    debiases(::Nothing) -> Bool

Whether the regime statistic corrects the bias of the estimate it reads.

With `debias = nothing`, the statistic is raw. Every observation above `min_obs` is scored. The mean of the statistic is above the one the method assumes, by 7 % at a half-life of 10 and 1.7 % at a half-life of 40 for the root mean square, so the regime multiplier is too large on average.

# Returns

  - `flag::Bool`: `true` for every [`AbstractRegimeDebias`](@ref), such as [`ExactDebias`](@ref) and [`LawDebias`](@ref), and `false` for `nothing`.

# Related

  - [`AbstractRegimeDebias`](@ref)
  - [`reads_dependence`](@ref)
  - [`regime_bias_open`](@ref)
"""
function debiases(::AbstractRegimeDebias)
    return true
end
function debiases(::Nothing)
    return false
end
"""
    reads_dependence(::ExactDebias) -> Bool
    reads_dependence(::Option{<:LawDebias}) -> Bool

Whether the bias of the regime statistic reads the dependence of the deviation on the estimate,
which an estimated location puts in.

# Returns

  - `flag::Bool`: `true` for [`ExactDebias`](@ref), `false` for [`LawDebias`](@ref) and
    `nothing`.

# Related

  - [`AbstractRegimeDebias`](@ref)
  - [`debiases`](@ref)
  - [`regime_bias_table`](@ref)
"""
function reads_dependence(::ExactDebias)
    return true
end
function reads_dependence(::Option{<:LawDebias})
    return false
end
"""
$(DocStringExtensions.TYPEDEF)

Floors each HAC-adjusted square at zero before it enters the variance recursion.

The floor removes the negative part of each term, so the mean of a term is above the variance. Under returns with no autocorrelation the variance is 6.7 %, 16 % and 41 % too large at one, two and five lags. The default `hac_floor = nothing` puts no floor on a term.

# Constructors

    PerTermHacFloor() -> PerTermHacFloor

# Examples

```jldoctest
julia> RegimeAdjustedExpWeightedVariance(; hac_floor = PerTermHacFloor()).hac_floor
PerTermHacFloor()
```

# Related

  - [`AbstractHacFloor`](@ref)
  - [`hac_floor!`](@ref)
"""
struct PerTermHacFloor <: AbstractHacFloor end
"""
    hac_floor!(::Nothing, X2::VecNum, valid::AbstractVector{<:Bool})
    hac_floor!(::PerTermHacFloor, X2::VecNum, valid::AbstractVector{<:Bool})

Applies the floor rule to the HAC-adjusted squares of the valid assets.

With no rule, each HAC-adjusted product enters the variance recursion as it is, negative or not. Under returns with no autocorrelation each term then has the mean of the variance, and the recursion is an unbiased quadratic form. Only the variance that the estimator returns is floored at zero.

# Arguments

  - `X2`: The HAC-adjusted squares. The method writes it in place.
  - `valid`: Mask of the valid assets.

# Returns

  - `X2`: The squares, unchanged under `nothing`, and floored at zero on the valid assets under [`PerTermHacFloor`](@ref).

# Related

  - [`AbstractHacFloor`](@ref)
"""
function hac_floor!(::Nothing, X2::VecNum, ::AbstractVector{<:Bool})
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
$(DocStringExtensions.TYPEDEF)

Standardises each product of the separate correlation path by the bias-corrected variance. This is the default rule.

The variance state is seeded at zero, so after ``K`` steps it holds ``1 - \\lambda^{K}`` times the variance. A product divided by the raw state thus carries the weight ``1 / (1 - \\lambda^{K})``, which falls with ``K``. The correlation cancels a scale that is the same for every product, but not this one, so in the warm-up the first products weigh more than the weights that the bias tables of [`PortfolioTarget`](@ref) assume. This rule divides the state by ``1 - \\lambda^{K}`` first. At 4 uncorrelated assets and 20 observations it moved the factor of `PortfolioTarget` from 3.5 % high to 0.05 %, and at 12 assets with a correlation of 0.5 from 8.0 % to 1.7 %. The steady state does not change. `cor_vol = nothing` reads the raw state, the rule of the oracle.

# Constructors

    BiasCorrectedVolatility() -> BiasCorrectedVolatility

# Examples

```jldoctest
julia> RegimeAdjustedExpWeightedCovariance().cor_vol
BiasCorrectedVolatility()
```

# Related

  - [`AbstractCorrelationVolatility`](@ref)
  - [`correlation_variance`](@ref)
"""
struct BiasCorrectedVolatility <: AbstractCorrelationVolatility end
"""
    correlation_variance(::BiasCorrectedVolatility, v::VecNum, decay::Number, K::AbstractVector{<:Integer})
    correlation_variance(::Nothing, v::VecNum, decay::Number, K::AbstractVector{<:Integer})

The variance that standardises each product of the separate correlation path.

With `cor_vol = nothing`, each product divides by the raw variance state, seeded at zero. This is the rule of the oracle. In the warm-up it weighs the first products of the correlation more than the bias tables of [`PortfolioTarget`](@ref) assume. The steady state is the same as under [`BiasCorrectedVolatility`](@ref).

# Arguments

  - `v`: The variance state that the product divides by.
  - `decay`: The decay of the variance.
  - `K`: The count of the steps that each entry of `v` holds.

# Returns

  - `v::VecNum`: `v` divided by its weight ``1 - \\lambda^{K}`` under [`BiasCorrectedVolatility`](@ref), and `v` unchanged under `nothing`.

# Related

  - [`AbstractCorrelationVolatility`](@ref)
"""
function correlation_variance(::BiasCorrectedVolatility, v::VecNum, decay::Number,
                              K::AbstractVector{<:Integer})
    return v ./ max.(one(decay) .- decay .^ K, eps(decay))
end
function correlation_variance(::Nothing, v::VecNum, ::Number, ::AbstractVector{<:Integer})
    return v
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

Return the effective number of observations of the correlation of a regime-adjusted covariance over `n` rows.

The denoising step processes the correlation, so the count is the count of the correlation. Under a separate correlation decay, the correlation reads the pre-centred law at `cor_decay`, as the regime bias does. Otherwise it reads `decay` and the centring of the estimator. The count ignores the regime multiplier and the lags of a HAC adjustment, as [`variance_count`](@ref) does.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator.
  - `n`: Number of rows.

# Returns

  - `T_e::Number`: The effective number of observations, from [`exp_weighted_covariance_count`](@ref).

# Related

  - [`effective_observation_count`](@ref)
  - [`has_separate_cor_decay`](@ref)
  - [`exp_weighted_covariance_count`](@ref)
"""
function regime_correlation_count(ce::RegimeAdjustedExpWeightedCovariance, n::Integer)
    return if has_separate_cor_decay(ce)
        exp_weighted_covariance_count(ce.cor_decay, n, PreCentred())
    else
        exp_weighted_covariance_count(ce.decay, n, ce.centring)
    end
end
function effective_observation_count(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum)
    return regime_correlation_count(ce, size(X, 1))
end
function effective_observation_count(ce::RegimeAdjustedExpWeightedCovariance,
                                     state::RegimeAdjustedCovarianceState)
    return regime_correlation_count(ce, observation_count(state))
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

  - `F::Union{Number, MatNum}`: The `N × N` matrix of the factors, or one.

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
"""
$(DocStringExtensions.TYPEDEF)

Refuses a covariance block with a negative eigenvalue while the block is young, and scores it once the block is old. This is the default rule of [`PortfolioTarget`](@ref).

The block is young while the gate that [`MahalanobisTarget`](@ref) reads under [`ExactDebias`](@ref) is closed for the same assets: [`regime_bias_open`](@ref) under `ExactDebias()` at the count of the youngest asset, under every `debias` of the estimator, on the decay of the correlation structure, with the HAC lags of the estimator. Under HAC that gate asks that the effective count of observations of the banded weight matrix pass ``n + 1``, and in the warm-up before it a HAC block need not be positive definite. A block of assets with different histories need not be positive definite at any count, and once the gate is open this rule scores it, so the steady state of a panel with a holiday or a late listing does not move. An estimate whose steady state never opens the gate, such as 12 assets at a half-life of 5 under HAC, is young at every count.

On a case of iid Normal returns at 8 assets, an equicorrelation of 0.5, a half-life of 10, two lags and 30 seeds of 8 rows, the largest regime state of the inverse-volatility direction was 51.4 on the separate path and 14.8 on one decay where the block was scored as it is. On one decay the young blocks of those rows are singular but have no negative eigenvalue, so this rule scores them, and the state stays at 14.8. The separate path refuses every one of those first rows, and it scores its first row at row 14 in the median and at row 21 at most, where its gate opens at 20.

# Constructors

    RefuseIndefiniteWarmUp() -> RefuseIndefiniteWarmUp

# Examples

```jldoctest
julia> PortfolioTarget().indefinite
RefuseIndefiniteWarmUp()
```

# Related

  - [`AbstractIndefiniteBlockRule`](@ref)
  - [`ClipIndefiniteBlock`](@ref)
  - [`ScoreIndefiniteBlock`](@ref)
  - [`indefinite_block`](@ref)
"""
struct RefuseIndefiniteWarmUp <: AbstractIndefiniteBlockRule end
"""
$(DocStringExtensions.TYPEDEF)

Clips the eigenvalues of the correlation of a covariance block at zero, keeps the variances, and scores the direction on the clipped block, at every count.

The clip is the one that [`regime_adjusted_covariance`](@ref) applies to the report, [`restore_psd!`](@ref). A block that has a Cholesky factor is kept as it is, so the rule costs one Cholesky factor per row, and one eigendecomposition only where the block is not positive definite. It moves the statistic of a panel with a holiday or a late listing in the steady state too, where the block need not be positive semidefinite. On the case of iid Normal returns above, the largest regime state of the inverse-volatility direction after 8 rows fell from 51.4 to 24.3 on the separate path, and on one decay it stayed at 14.8.

# Constructors

    ClipIndefiniteBlock() -> ClipIndefiniteBlock

# Examples

```jldoctest
julia> PortfolioTarget(; indefinite = ClipIndefiniteBlock()).indefinite
ClipIndefiniteBlock()
```

# Related

  - [`AbstractIndefiniteBlockRule`](@ref)
  - [`RefuseIndefiniteWarmUp`](@ref)
  - [`ScoreIndefiniteBlock`](@ref)
  - [`indefinite_block`](@ref)
"""
struct ClipIndefiniteBlock <: AbstractIndefiniteBlockRule end
"""
$(DocStringExtensions.TYPEDEF)

Scores a covariance block as it is, positive definite or not.

A direction whose variance is at or below `min_val` is still refused, as [`regime_statistic`](@ref) states. A direction with a small positive variance on an indefinite block scores a large statistic, which the clip `regime_lohi_mult` of the estimator bounds where it is set.

# Constructors

    ScoreIndefiniteBlock() -> ScoreIndefiniteBlock

# Examples

```jldoctest
julia> PortfolioTarget(; indefinite = ScoreIndefiniteBlock()).indefinite
ScoreIndefiniteBlock()
```

# Related

  - [`AbstractIndefiniteBlockRule`](@ref)
  - [`RefuseIndefiniteWarmUp`](@ref)
  - [`ClipIndefiniteBlock`](@ref)
  - [`indefinite_block`](@ref)
"""
struct ScoreIndefiniteBlock <: AbstractIndefiniteBlockRule end
"""
    indefinite_block(::RefuseIndefiniteWarmUp, C::MatNum, young::Bool)
    indefinite_block(::ClipIndefiniteBlock, C::MatNum, young::Bool)
    indefinite_block(::ScoreIndefiniteBlock, C::MatNum, young::Bool)

The covariance block that the portfolio regime statistic reads.

# Arguments

  - `C`: The bias-corrected covariance block of the assets that contribute to the statistic.
  - `young`: Whether the block is young, that is whether the gate that [`MahalanobisTarget`](@ref) reads under `ExactDebias()` is closed for the same assets.

# Returns

  - `block::Option{<:MatNum}`: `nothing` under [`RefuseIndefiniteWarmUp`](@ref) where the block is young and has an eigenvalue below ``-n \\epsilon`` times the largest in size, and `C` else. A singular block, such as a block of fewer observations than assets, is kept: each direction reads a true variance on it. A copy of `C` with the clip of [`restore_psd!`](@ref) under [`ClipIndefiniteBlock`](@ref). `C` under [`ScoreIndefiniteBlock`](@ref).

# Related

  - [`AbstractIndefiniteBlockRule`](@ref)
  - [`PortfolioTarget`](@ref)
"""
function indefinite_block(::RefuseIndefiniteWarmUp, C::MatNum, young::Bool)
    if !young || LinearAlgebra.issuccess(LinearAlgebra.cholesky(LinearAlgebra.Symmetric(C);
                                                                check = false))
        return C
    end
    # A singular block of fewer rows than assets is positive semidefinite, and each direction
    # reads a true variance on it: only a negative eigenvalue refuses the block.
    vals = LinearAlgebra.eigvals(LinearAlgebra.Symmetric(C))
    return vals[1] < -length(vals) * eps(eltype(vals)) * maximum(abs, vals) ? nothing : C
end
function indefinite_block(::ClipIndefiniteBlock, C::MatNum, ::Bool)
    return restore_psd!(copy(C))
end
function indefinite_block(::ScoreIndefiniteBlock, C::MatNum, ::Bool)
    return C
end
export ExactDebias, LawDebias, PerTermHacFloor, VolatilityBeforeUpdate,
       VolatilityAfterUpdate, BiasCorrectedVolatility, RefuseIndefiniteWarmUp,
       ClipIndefiniteBlock, ScoreIndefiniteBlock
public debiases, reads_dependence, hac_floor!, volatility_before_update,
       correlation_variance, indefinite_block
