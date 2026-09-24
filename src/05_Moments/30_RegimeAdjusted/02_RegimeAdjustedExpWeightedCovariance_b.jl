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
                                                              <:Any, <:Any, <:Any,
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
                                                              <:Any, <:Any, <:Any,
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
    val = Matrix{eltype(X)}(undef, size(X, dims), size(X, setdiff((1, 2), (dims,))[1]))
    regime_adjusted_covariance_pass!(ce, X, dims, estimation_mask, active_mask) do i, cache
        val[i, :] = LinearAlgebra.diag(regime_adjusted_covariance(cache, ce))
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
