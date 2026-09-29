"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the regime statistic of one observation for a target, from the covariance block that
stands before it.

A target that has no method of its own reads its [`regime_statistic`](@ref) of the block that
[`regime_covariance_block`](@ref) returns, and takes no bias correction.

# Arguments

  - `target::RegimeAdjustedTarget`: Regime-adjustment target.
  - `cache::RegimeAdjustedCovarianceState`: Online covariance computation cache.
  - `ce::RegimeAdjustedExpWeightedCovariance`: Covariance estimator configuration.
  - `X::VecNum`: Current centred returns vector of every asset.
  - `idx::AbstractVector{<:Integer}`: Index of the assets that contribute to the statistic.

# Returns

  - `stats::Option{<:VecNum}`: One statistic per calibration direction, or `nothing` where the
    observation takes no regime update.

# Related

  - [`update_regime!`](@ref)
  - [`regime_statistic`](@ref)
  - [`RegimeAdjustedTarget`](@ref)
"""
function regime_target_statistic(target::RegimeAdjustedTarget,
                                 cache::RegimeAdjustedCovarianceState,
                                 ce::RegimeAdjustedExpWeightedCovariance, X::VecNum,
                                 idx::AbstractVector{<:Integer})
    return regime_statistic(target, X[idx], regime_covariance_block(cache, ce, idx), idx,
                            ce.min_val)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the regime statistic of the diagonal target, with each term divided by the bias of its
own estimated variance.

Each term ``u_{i}^{2} / \\hat{C}_{ii}`` reads one estimated variance of ``K_{i}`` observations,
so its mean is ``\\mathbb{E}[Q^{-1}]`` at ``K_{i}``, from [`regime_bias_table`](@ref), and not
one. Each term is divided by that factor whatever the method, so the sum has the mean ``n`` at
every correlation of the assets. A sum of ``n`` terms averages the errors of the ``n``
estimates, so the root and the log that the other methods read see almost the same factor: at 12
assets and a half-life of 10 the bias is 1.073 for the mean, 1.068 for the root and 1.063 for the
log at a correlation of 0.3. The factor of each method alone is exact only at one asset.

# Arguments

  - `::DiagonalTarget`: Diagonal regime-adjustment target.
  - `cache::RegimeAdjustedCovarianceState`: Online covariance computation cache.
  - `ce::RegimeAdjustedExpWeightedCovariance`: Covariance estimator configuration.
  - `X::VecNum`: Current centred returns vector of every asset.
  - `idx::AbstractVector{<:Integer}`: Index of the assets that contribute to the statistic.

# Returns

  - `stats::VecNum`: One statistic.

# Related

  - [`DiagonalTarget`](@ref)
  - [`regime_bias!`](@ref)
  - [`update_regime!`](@ref)
"""
function regime_target_statistic(::DiagonalTarget, cache::RegimeAdjustedCovarianceState,
                                 ce::RegimeAdjustedExpWeightedCovariance, X::VecNum,
                                 idx::AbstractVector{<:Integer})
    f = regime_bias!.(Ref(cache.bias), Ref(RootMeanSquaredAdjusted()), ce.decay,
                      view(cache.obs_count, idx))
    return regime_statistic(DiagonalTarget(), X[idx] ./ sqrt.(f),
                            regime_covariance_block(cache, ce, idx), idx, ce.min_val)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the regime statistic of the portfolio target, divided by the bias of the estimated
variance of each direction.

A direction reads one estimated variance ``p^{\\top} \\hat{C} p``, so its statistic is biased by
the factor of [`regime_bias_table`](@ref) that the method reads, at ``K``, the smallest count of
observations among the contributing assets. That factor is exact for a fixed direction on one
shared history. On the separate correlation path the factor reads `cor_decay`, because the
correlations carry most of the variance of a direction: at 12 assets, a half-life of 10 and a
correlation half-life of 20, the squared multiplier of an equal-weight direction is 1.004 with it
and 0.971 with `decay`. A HAC adjustment makes the estimate noisier than its weights state, so
the factor under-corrects it: 1.055 at two lags.

The default inverse-volatility direction is built from the same estimate, so the direction and
the error of the estimate are correlated, and a bias remains: at 12 assets the squared multiplier
of `RootMeanSquaredAdjusted` on iid Normal returns is 1.134 before and 1.058 after the factor at
a half-life of 10, and 1.032 before at a half-life of 40.

# Arguments

  - `target::PortfolioTarget`: Portfolio regime-adjustment target.
  - `cache::RegimeAdjustedCovarianceState`: Online covariance computation cache.
  - `ce::RegimeAdjustedExpWeightedCovariance`: Covariance estimator configuration.
  - `X::VecNum`: Current centred returns vector of every asset.
  - `idx::AbstractVector{<:Integer}`: Index of the assets that contribute to the statistic.

# Returns

  - `stats::Option{<:VecNum}`: One statistic per direction, or `nothing` where no direction keeps
    a positive weight.

# Related

  - [`PortfolioTarget`](@ref)
  - [`regime_bias!`](@ref)
  - [`update_regime!`](@ref)
"""
function regime_target_statistic(target::PortfolioTarget,
                                 cache::RegimeAdjustedCovarianceState,
                                 ce::RegimeAdjustedExpWeightedCovariance, X::VecNum,
                                 idx::AbstractVector{<:Integer})
    stats = regime_statistic(target, X[idx], regime_covariance_block(cache, ce, idx), idx,
                             ce.min_val)
    if isnothing(stats)
        return nothing
    end

    decay = has_separate_cor_decay(ce) ? ce.cor_decay : ce.decay
    return stats ./ regime_bias!(cache.bias, ce.regime_method, decay,
                                 minimum(view(cache.obs_count, idx)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Makes the empty table of bias factors for a new state of a regime-adjusted estimator.

# Arguments

  - `ce::Union{<:RegimeAdjustedExpWeightedVariance, <:RegimeAdjustedExpWeightedCovariance}`:
    Estimator configuration.
  - `::Type{T}`: Element type of the state.

# Returns

  - `bias::Option{<:VecNum}`: An empty vector where `ce.debias` is `true` and a regime method is
    set, else `nothing`.

# Related

  - [`regime_bias!`](@ref)
  - [`RegimeAdjustedVarianceState`](@ref)
  - [`RegimeAdjustedCovarianceState`](@ref)
"""
function regime_bias_state(ce::Union{<:RegimeAdjustedExpWeightedVariance,
                                     <:RegimeAdjustedExpWeightedCovariance},
                           ::Type{T}) where {T}
    return ce.debias && !isnothing(ce.regime_method) ? T[] : nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Factorises a covariance block for the Mahalanobis regime statistic, and refuses rather than
throws when no ridge makes it factorise.

A block with fewer observations than assets is singular. Where the estimator has
`debias = true`, [`regime_target_statistic`](@ref) skips such a block before it reaches this
function, so the ridge serves a block that is singular in the data, and the raw statistic of
`debias = false`.
The regime statistic is one observation of a smoother rather than a result a caller reads, so a
refusal skips that observation's regime update, and the fit continues.

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
    ridge = max(min_val * scale, eps(scale) * scale)
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

Computes the squared Mahalanobis distance of one observation, divided by the bias of the
estimated block where `ce.debias` is `true`.

# Algorithm

 1. Where `ce.debias` is `false`, the factor is one. Else take ``K``, the smallest count of
    observations among the contributing assets, and return `nothing` where `K <= n + 3`. Else
    find the factor with [`mahalanobis_bias`](@ref) at the decay of the correlation structure:
    `cor_decay` on the separate path, else `decay`.
 2. Compute the squared distance with [`regime_statistic`](@ref), and return `nothing` where the
    block does not factorise.
 3. Divide the statistic by the factor.

# Arguments

  - `target::MahalanobisTarget`: Mahalanobis regime-adjustment target.
  - `cache::RegimeAdjustedCovarianceState`: Online covariance computation cache.
  - `ce::RegimeAdjustedExpWeightedCovariance`: Covariance estimator configuration.
  - `X::VecNum`: Current centred returns vector of every asset.
  - `idx::AbstractVector{<:Integer}`: Index of the assets that contribute to the statistic.

# Returns

  - `stats::Option{<:VecNum}`: One statistic, or `nothing` where the observation takes no
    regime update.

# Related

  - [`MahalanobisTarget`](@ref)
  - [`mahalanobis_bias`](@ref)
  - [`update_regime!`](@ref)
"""
function regime_target_statistic(target::MahalanobisTarget,
                                 cache::RegimeAdjustedCovarianceState,
                                 ce::RegimeAdjustedExpWeightedCovariance, X::VecNum,
                                 idx::AbstractVector{<:Integer})
    K = minimum(view(cache.obs_count, idx))
    b = if !ce.debias
        one(ce.decay)
    elseif K > length(idx) + 3
        mahalanobis_bias(has_separate_cor_decay(ce) ? ce.cor_decay : ce.decay, K,
                         length(idx))
    else
        nothing
    end
    if isnothing(b)
        return nothing
    end
    stats = regime_statistic(target, X[idx], regime_covariance_block(cache, ce, idx), idx,
                             ce.min_val)

    return isnothing(stats) ? nothing : stats ./ b
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the bias factor of the squared Mahalanobis distance against an exponentially weighted
covariance estimate.

The factor is ``b = \\mathbb{E}[\\operatorname{tr}(W^{-1})] / n`` of
``W = \\sum_{j} w_{j} z_{j} z_{j}^{\\top}``, ``z_{j} \\sim N(0, I_{n})``, which is the mean of
``u^{\\top} \\hat{C}^{-1} u / n`` for a correctly calibrated return ``u`` that is independent of the
estimate ``\\hat{C}``. It depends on the weights and on `n` alone, and it is the fixed point that
[`MahalanobisTarget`](@ref) states. The fixed point exists where `K > n + 1`.

# Algorithm

 1. Return `nothing` where `K <= n + 1`.
 2. Start at ``b = 1``. The map ``h(b) = b\\, g(b) - 1``, with ``g`` the right-hand side of the
    fixed point, is concave and increasing, and ``h(1) < 0``, so Newton's method from ``b = 1``
    rises to the root without overshoot.
 3. Evaluate ``g`` and ``h'`` with [`mahalanobis_bias_sums`](@ref), and stop when the step is
    below four units in the last place of ``b``.

# Arguments

  - `decay::Number`: Decay of the weights.
  - `K::Integer`: Count of observations in the estimate.
  - `n::Integer`: Count of assets that contribute to the statistic.

# Returns

  - `b::Option{<:Number}`: The bias factor, or `nothing` where `K <= n + 1`.

# Related

  - [`MahalanobisTarget`](@ref)
  - [`regime_target_statistic`](@ref)
  - [`mahalanobis_bias_sums`](@ref)
"""
function mahalanobis_bias(decay::Number, K::Integer, n::Integer)
    if K <= n + 1
        return nothing
    end
    w0 = (one(decay) - decay) / (one(decay) - decay^K)
    b = one(w0)
    for _ in 1:100
        g, dh = mahalanobis_bias_sums(decay, K, n + 1, w0, b)
        step = (b * g - one(b)) / dh
        b -= step
        if abs(step) <= 4 * eps(b)
            break
        end
    end

    return b
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the right-hand side of the fixed point of [`mahalanobis_bias`](@ref) and the
derivative of ``b\\, g(b)``.

The weights fall geometrically, and a weight ``w_{j}`` with ``c\\, w_{j}\\, b`` below the
machine epsilon of `decay` enters both sums as itself, so the loop stops there and adds the
remaining mass ``1 - \\sum_{i < j} w_{i}`` to each sum. That keeps the cost bounded on a long
history.

# Arguments

  - `decay::Number`: Decay of the weights.
  - `K::Integer`: Count of observations in the estimate.
  - `c::Integer`: `n + 1`.
  - `w0::Number`: Normalised weight of the newest observation, `(1 - decay) / (1 - decay^K)`.
  - `b::Number`: Current value of the bias factor.

# Returns

  - `(g, dh)::Tuple{<:Number, <:Number}`: ``\\sum_{j} w_{j} / (1 + c\\, w_{j}\\, b)`` and
    ``\\sum_{j} w_{j} / (1 + c\\, w_{j}\\, b)^{2}``.

# Related

  - [`mahalanobis_bias`](@ref)
"""
function mahalanobis_bias_sums(decay::Number, K::Integer, c::Integer, w0::Number, b::Number)
    g = zero(w0)
    dh = zero(w0)
    used = zero(w0)
    w = w0
    for _ in 1:K
        t = c * w * b
        if t <= eps(decay)
            break
        end
        g += w / (one(t) + t)
        dh += w / (one(t) + t)^2
        used += w
        w *= decay
    end
    tail = one(w0) - used

    return g + tail, dh + tail
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
    cache = regime_adjusted_covariance_pass!(ce, X, dims, estimation_mask, active_mask)
    if !ce.centred
        unseen = cache.obs_count .< one(eltype(cache.obs_count))
        if any(unseen)
            cache.location[unseen] .= NaN
        end
    end

    return regime_adjusted_covariance(cache, ce)
end
"""
    gap_fill_value(ce::RegimeAdjustedExpWeightedCovariance) -> Float64

Answer `NaN`, so a gapped sample reaches the recursion with its gaps intact.

The recursion updates only the sub-block of the assets that are valid at each observation, and the regime weight is taken from that sub-block alone, so a fill would both decay a frozen block and move the regime it is weighted by. The consumer therefore hands the sample as it stands, together with the active mask that explains the gap.

# Arguments

  - $(arg_dict[:ce])

# Returns

  - `fv::Float64`: `NaN`.

# Related

  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`gap_fill_value`](@ref)
  - [`Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
"""
function gap_fill_value(::RegimeAdjustedExpWeightedCovariance)
    return NaN
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
function partial_fit!(ce::RegimeAdjustedExpWeightedCovariance{<:Any, <:Any, <:Any, <:Any,
                                                              <:Any, <:Any, <:Any, <:Any,
                                                              <:Any, <:Any, <:Any, <:Any,
                                                              <:Option{<:RegimeAdjustedCovarianceState}},
                      X::MatNum; dims::Int = 1,
                      estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                      active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    cache = regime_adjusted_covariance_pass!(ce, X, dims, estimation_mask, active_mask,
                                             ce.cache)
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
function partial_fit!(ce::RegimeAdjustedExpWeightedCovariance{<:Any, <:Any, <:Any, <:Any,
                                                              <:Any, <:Any, <:Any, <:Any,
                                                              <:Any, <:Any, <:Any, <:Any,
                                                              <:Option{<:RegimeAdjustedCovarianceState}},
                      x::VecNum;
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

# Algorithm

 1. Refuse the pair with [`assert_mergeable_states`](@ref), which names a type mismatch and an asset-count mismatch first, as the [`AbstractPartialFitState`](@ref) interface asks of every family.
 2. Throw an `ArgumentError` naming the reason this family does not merge.

# Arguments

  - `a`: The first state.
  - `b`: The second state.

# Validation

  - `a` and `b` pass [`assert_mergeable_states`](@ref).

# Returns

  - Never returns. An `ArgumentError` is thrown.

# Related

  - [`RegimeAdjustedCovarianceState`](@ref)
  - [`merge_states`](@ref)
  - [`assert_mergeable_states`](@ref)
  - [`partial_fit!(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
"""
function merge_states(a::RegimeAdjustedCovarianceState, b::RegimeAdjustedCovarianceState)
    assert_mergeable_states(a, b)
    return throw(ArgumentError("a `RegimeAdjustedCovarianceState` pair does not merge, because a block fitted from a cold start is not what the same block contributes after another one. The regime statistic scores each observation against the state that stands before it, and it is gated by the running observation count. Fold the second block into the first with `partial_fit!` instead."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies a [`RegimeAdjustedCovarianceState`](@ref), so the copy shares no array with the original.

The `copy` method of the [`AbstractPartialFitState`](@ref) interface, which [`partial_fit`](@ref)
calls before it folds. Every array field is copied, and the two scalar fields pass through. The
circular buffer of recent centred returns is rebuilt at the same capacity, and each observation
it holds is copied into it, so a fold on the copy pushes into a buffer of its own. The two
fields of the separate correlation recursion pass through as `nothing` where they are `nothing`.
The table of bias factors is `nothing` or an array, and `deepcopy` copies an array and returns
`nothing` unchanged.

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

    weight = isnothing(x.weight) ? nothing : copy(x.weight)
    variance = isnothing(x.variance) ? nothing : copy(x.variance)
    cor_state = isnothing(x.cor_state) ? nothing : copy(x.cor_state)
    cor_weight = isnothing(x.cor_weight) ? nothing : copy(x.cor_weight)

    return RegimeAdjustedCovarianceState(ret_buffer, copy(x.covariance), weight, variance,
                                         cor_state, cor_weight, copy(x.XXt), copy(x.Xi),
                                         copy(x.X_old_i), copy(x.location),
                                         copy(x.obs_count), copy(x.active), x.regime_state,
                                         x.n_regime_obs, deepcopy(x.bias))
end
"""
    Statistics.cov(
        ce::RegimeAdjustedExpWeightedCovariance,
        X::MatNum,
        pnl::Option{<:AssetPanel};
        dims::Int = 1,
        kwargs...
    ) -> MatNum

Compute the regime-adjusted exponentially weighted covariance from a window of an Asset Panel.

This estimator is mask-aware, so it overrides the reduce-and-expand root of the verb and reads the panel's two masks itself: the active mask drives the freeze and the reset, and the estimation mask restricts which assets feed the regime statistic. The answer therefore lives on the whole universe rather than on the Coverage Universe, and a young asset that lists inside the window is answered from the observations it has.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Returns

  - `sigma::MatNum`: Covariance matrix of size `assets × assets`.

# Related

  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`panel_moment_masks`](@ref)
  - [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    amsk, emsk = dims_oriented(dims, panel_moment_masks(pnl)...)
    return Statistics.cov(ce, X; dims = dims, estimation_mask = emsk, active_mask = amsk,
                          kwargs...)
end
"""
    Statistics.cor(
        ce::RegimeAdjustedExpWeightedCovariance,
        X::MatNum,
        pnl::Option{<:AssetPanel};
        dims::Int = 1,
        kwargs...
    ) -> MatNum

Compute the regime-adjusted exponentially weighted correlation from a window of an Asset Panel.

This is the covariance of the same call, rescaled to a unit diagonal, and it reads the panel's two masks through the same override.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Returns

  - `rho::MatNum`: Correlation matrix of size `assets × assets`.

# Related

  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cor(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    amsk, emsk = dims_oriented(dims, panel_moment_masks(pnl)...)
    return Statistics.cor(ce, X; dims = dims, estimation_mask = emsk, active_mask = amsk,
                          kwargs...)
end
"""
    Statistics.var(
        ce::RegimeAdjustedExpWeightedCovariance,
        X::MatNum,
        pnl::Option{<:AssetPanel};
        dims::Int = 1,
        kwargs...
    ) -> MatNum

Compute the marginal variance of the regime-adjusted exponentially weighted covariance from a window of an Asset Panel.

This is the diagonal of the covariance of the same call, and it reads the panel's two masks through the same override.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Returns

  - `var::MatNum`: Marginal variance, as a row where `dims` is `1` and as a column otherwise.

# Related

  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.var(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    amsk, emsk = dims_oriented(dims, panel_moment_masks(pnl)...)
    return Statistics.var(ce, X; dims = dims, estimation_mask = emsk, active_mask = amsk,
                          kwargs...)
end
"""
    Statistics.std(
        ce::RegimeAdjustedExpWeightedCovariance,
        X::MatNum,
        pnl::Option{<:AssetPanel};
        dims::Int = 1,
        kwargs...
    ) -> MatNum

Compute the marginal volatility of the regime-adjusted exponentially weighted covariance from a window of an Asset Panel.

This is the square root of the diagonal of the covariance of the same call, and it reads the panel's two masks through the same override.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Returns

  - `std::MatNum`: Marginal volatility, as a row where `dims` is `1` and as a column otherwise.

# Related

  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`Statistics.var(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.std(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    amsk, emsk = dims_oriented(dims, panel_moment_masks(pnl)...)
    return Statistics.std(ce, X; dims = dims, estimation_mask = emsk, active_mask = amsk,
                          kwargs...)
end

"""
    variance_series(
        ce::RegimeAdjustedExpWeightedCovariance,
        X::MatNum;
        dims::Int = 1,
        estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        kwargs...
    ) -> Matrix{<:Number}

Compute the point-in-time regime-adjusted exponentially weighted variance series.

Row `t` holds the diagonal of what `cov` returns for the first `t` observations of `X`, so no row reads an observation after its own. The update is a recursion over one observation, so this method overrides the expanding-window fallback with a **single forward pass**: it reads the cache after each observation instead of refitting.

The fallback cannot answer this estimator. It slices `X` once per row and passes every keyword unsliced, so a mask of the whole window meets a window of `t` observations and the size check refuses the call.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `estimation_mask`: Optional boolean matrix with the same size as `X`. When provided,
    only assets where `estimation_mask[i, :]` (or `[:, i]`) is `true` contribute to the
    regime state update for observation `i`.
  - `active_mask`: Optional boolean matrix with the same size as `X`. When provided,
    assets that become inactive have their covariance and observation count reset.
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `estimation_mask` is not `nothing`, `size(X) == size(estimation_mask)`.
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`.

# Returns

  - `val::Matrix{<:Number}`: Variance series, shaped as `(T, N)` if `dims == 1` or `(N, T)` if
    `dims == 2`. An asset with fewer than `ce.min_obs` observations at row `t` is `NaN` there.

# Related

  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`regime_adjusted_covariance`](@ref)
  - [`variance_series(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function variance_series(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1,
                         estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                         active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    assert_dims(dims)
    val = Matrix{float_if_integer(eltype(X))}(undef, size(X, dims),
                                              size(X, setdiff((1, 2), (dims,))[1]))
    regime_adjusted_covariance_pass!(ce, X, dims, estimation_mask, active_mask) do i, cache
        val[i, :] = LinearAlgebra.diag(regime_adjusted_covariance(cache, ce;
                                                                  repair = false))
        return nothing
    end

    return isone(dims) ? val : permutedims(val)
end
"""
    variance_series(
        ce::RegimeAdjustedExpWeightedCovariance,
        X::MatNum,
        pnl::Option{<:AssetPanel};
        dims::Int = 1,
        kwargs...
    ) -> Matrix{<:Number}

Compute the point-in-time regime-adjusted exponentially weighted variance series from a window of an Asset Panel.

This is the diagonal of the covariance of the same call, read after each observation, and it reads the panel's two masks through the same override.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Returns

  - `val::Matrix{<:Number}`: Variance series on the full asset universe, shaped as `(T, N)` if
    `dims == 1` or `(N, T)` if `dims == 2`.

# Related

  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`variance_series(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
  - [`variance_series(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function variance_series(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum,
                         pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    amsk, emsk = dims_oriented(dims, panel_moment_masks(pnl)...)
    return variance_series(ce, X; dims = dims, estimation_mask = emsk, active_mask = amsk,
                           kwargs...)
end

# Folds in every configuration; only its merge refuses (see [`supports_partial_fit`](@ref)).
function supports_partial_fit(::RegimeAdjustedExpWeightedCovariance)
    return true
end
