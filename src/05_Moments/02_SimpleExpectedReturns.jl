"""
$(DocStringExtensions.TYPEDEF)

Computes the expected returns as the sample mean of the asset returns.

`w` holds the optional observation weights. If `w` is `nothing`, the mean is unweighted. The library uses this estimator as its default expected returns estimator.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SimpleExpectedReturns(;
        w::Option{<:ObsWeights} = nothing,
        cvg::Option{<:CoveragePolicy} = nothing,
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> SimpleExpectedReturns

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `w`: Replaced with the incoming [`ObsWeights`](@ref).
  - `cache`: Carried unchanged through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `cache`: Sliced to the selected assets through [`port_opt_view`](@ref).

## Observation weight parameters

[`obs_weights_view`](@ref) indexes these fields to the selected observations:

  - `w`: Indexed to the selected observations through [`obs_weights_view`](@ref).
  - `cache`: Removed by [`obs_weights_view`](@ref), because a state has no slice on the observation axis.

# Examples

```jldoctest
julia> SimpleExpectedReturns()
SimpleExpectedReturns
  w ┴ nothing

julia> SimpleExpectedReturns(; w = StatsBase.Weights([0.5, 0.5]))
SimpleExpectedReturns
  w ┴ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.5, 0.5]
```

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`Option`](@ref)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
  - [`mean(me::SimpleExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
  - [`SimpleExpectedReturnsState`](@ref)
  - [`partial_fit!`](@ref)
  - [`factory`](@ref)
  - [`obs_weights_view`](@ref)
"""
@propagatable @concrete struct SimpleExpectedReturns <: AbstractExpectedReturnsEstimator
    """
    $(field_dict[:oow])
    """
    @wprop w
    """
    $(field_dict[:cvg])
    """
    cvg
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    function SimpleExpectedReturns(w::Option{<:ObsWeights}, cvg::Option{<:CoveragePolicy},
                                   cache::Option{<:AbstractPartialFitState})::SimpleExpectedReturns
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(w), typeof(cvg), typeof(cache)}(w, cvg, cache)
    end
end
function SimpleExpectedReturns(; w::Option{<:ObsWeights} = nothing,
                               cvg::Option{<:CoveragePolicy} = nothing,
                               cache::Option{<:AbstractPartialFitState} = nothing)::SimpleExpectedReturns
    return SimpleExpectedReturns(w, cvg, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a [`SimpleExpectedReturns`](@ref) except `cache`, and `cvg` only where a policy is set.

A `cache` holds the state of an incremental fit. That state is not part of the configuration, and it prints under the estimator at each site that renders the estimator. Set `set_show_nothing_fields!(:SimpleExpectedReturns, true)` to render it. The method reads `cvg` from the instance and not from the type, because an option that the caller did not set is not part of the configuration that the caller chose. An estimator whose `cvg` is `nothing` renders as it did before the field existed. An estimator that holds a [`CoveragePolicy`](@ref) renders it.

# Arguments

  - `me`: The expected returns estimator. The method reads only its `cvg` field.

# Returns

  - `fields::Tuple`: The field names to render, which is `(:w,)` with no policy and `(:w, :cvg)` with one.

# Related

  - [`SimpleExpectedReturns`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
show_fields(me::SimpleExpectedReturns) = isnothing(me.cvg) ? (:w,) : (:w, :cvg)
"""
    Statistics.mean(
        me::SimpleExpectedReturns,
        X::MatNum;
        dims::Int = 1,
        kwargs...
    ) -> ArrNum

Compute the mean of the asset returns with a [`SimpleExpectedReturns`](@ref) estimator.

The method computes the expected returns as the sample mean of `X`, as `me` configures it.

# Mathematical definition

Unweighted:

```math
\\begin{align}
\\hat{\\mu}_j &= \\frac{1}{T} \\sum_{t=1}^{T} r_{tj}\\,.
\\end{align}
```

Weighted:

```math
\\begin{align}
\\hat{\\mu}_j &= \\frac{\\sum_{t=1}^{T} w_t \\, r_{tj}}{\\sum_{t=1}^{T} w_t}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\boldsymbol{\\mu}}``: The ``N \\times 1`` vector of estimated expected returns. Its ``j``-th entry is ``\\hat{\\mu}_j``.
  - $(math_dict[:mu_hat_j])
  - $(math_dict[:r_tj])
  - $(math_dict[:T])
  - $(math_dict[:w_t_obs])

# Algorithm

 1. Check that `dims` is `1` or `2`.
 2. Get the observation weights `w` from `me.w` and `X`.
 3. When `w` is `nothing`, take the unweighted mean of `X` along `dims`.
 4. Otherwise take the mean of `X` weighted by `w` along `dims`.

# Arguments

  - $(arg_dict[:me])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: More keyword arguments for [`get_observation_weights`](@ref), where a [`DynamicAbstractWeights`](@ref) can read them. The method does not pass them to `Statistics.mean`.

# Validation

  - $(val_dict[:dims])

# Returns

  - $(ret_dict[:mu])

# Examples

```jldoctest
julia> X = [0.01 0.02; 0.03 0.04];

julia> ser = SimpleExpectedReturns()
SimpleExpectedReturns
  w ┴ nothing

julia> mean(ser, X)
1×2 Matrix{Float64}:
 0.02  0.03

julia> serw = SimpleExpectedReturns(; w = StatsBase.Weights([0.2, 0.8]))
SimpleExpectedReturns
  w ┴ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.2, 0.8]

julia> mean(serw, X)
1×2 Matrix{Float64}:
 0.026  0.036
```

# Related

  - [`SimpleExpectedReturns`](@ref)
  - [`MatNum`](@ref)
  - [`VecNum`](@ref)
  - [`Statistics.mean`](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.mean)
"""
function Statistics.mean(me::SimpleExpectedReturns, X::MatNum; dims::Int = 1,
                         active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    return coverage_mean(me, me.cvg, X; dims = dims, active_mask = active_mask, kwargs...)
end
function effective_observation_count(me::SimpleExpectedReturns, X::MatNum)
    return observation_weight_count(get_observation_weights(me.w, X; dims = 1), size(X, 1))
end
"""
    coverage_mean(me, cvg, X; dims::Int = 1, active_mask = nothing, kwargs...) -> ArrNum

Routes a sample-mean fit to the Coverage Universe arm or to the available-case arm.

The `cvg` field of the estimator is the second argument, so dispatch on the policy selects the arm, not a branch on its value. The field costs nothing for an estimator that holds no policy. The `Nothing` method is the body that [`mean(me::SimpleExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref) had before the field existed. The compiler resolves it statically, because the type of the field is concrete.

# Arguments

  - $(arg_dict[:me])
  - `cvg`: The policy that the estimator holds. It selects the arm.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `active_mask`: The active mask of the Asset Panel, `observations × assets`, or `nothing`. The Coverage Universe arm refuses an inactive cell, as it refuses a return that is not finite.
  - `kwargs...`: More keyword arguments. The `Nothing` method passes them to [`get_observation_weights`](@ref). The [`CoveragePolicy`](@ref) method ignores them.

# Returns

  - $(ret_dict[:mu])

# Related

  - [`CoveragePolicy`](@ref)
  - [`SimpleExpectedReturns`](@ref)
  - [`mean(me::SimpleExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function coverage_mean end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The `Nothing` method of [`coverage_mean`](@ref) is the Coverage Universe arm. It returns the sample mean of a window that every asset covers. This is the result that [`mean(me::SimpleExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref) gave before the `cvg` field existed. It refuses a gapped sample with [`assert_finite_sample`](@ref). A cell is a gap when its return is not finite or when `active_mask` is `false` at that cell, so the arm refuses an inactive cell too. An estimator with no policy reads the universe through [`coverage_reduction`](@ref), which gives it a block with no gap.

# Related

  - [`coverage_mean`](@ref)
  - [`SimpleExpectedReturns`](@ref)
  - [`assert_finite_sample`](@ref)
"""
function coverage_mean(me::SimpleExpectedReturns, ::Nothing, X::MatNum; dims::Int = 1,
                       active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    assert_dims(dims)
    assert_finite_sample(X, active_mask)
    w = get_observation_weights(me.w, X; dims = dims, kwargs...)
    return if isnothing(w)
        Statistics.mean(X; dims = dims)
    else
        Statistics.mean(X, w; dims = dims)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The [`CoveragePolicy`](@ref) method of [`coverage_mean`](@ref) is the available-case arm. It fits the mean of each asset on the finite and active observations of that asset. The batch result is the incremental result, because the arm folds the block through [`partial_fit!`](@ref) and then reads the state.

Because the batch arm is the fold, the batch result and the incremental result agree to the last bit, not only to a tolerance. There is one recursion, so a caller who gives the same rows one at a time gets the same floating-point number.

# Algorithm

 1. Fold each row of `X` into a new state with [`partial_fit!`](@ref), together with its row of the active mask.
 2. Read the state with [`mean(me::SimpleExpectedReturns, state::SimpleExpectedReturnsState)`](@ref).
 3. Orient the result as the `dims` of the caller sets.

# Related

  - [`coverage_mean`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`partial_fit!`](@ref)
"""
function coverage_mean(me::SimpleExpectedReturns, cvg::CoveragePolicy, X::MatNum;
                       dims::Int = 1,
                       active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    assert_dims(dims)
    me = partial_fit!(SimpleExpectedReturns(; w = me.w, cvg = cvg), X; dims = dims,
                      active_mask = active_mask)
    mu = Statistics.mean(me)
    return isone(dims) ? permutedims(mu) : reshape(mu, :, 1)
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the observation count and the running mean of an incremental sample-mean fit.

It is the state of [`SimpleExpectedReturns`](@ref) under [`partial_fit!`](@ref). It holds no second-moment accumulator, because the mean is the whole estimate. [`merge_states`](@ref) folds the two counts and the two means, and it discards the accumulator that [`chan_merge`](@ref) returns.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SimpleExpectedReturnsState(;
        n::Integer = 0,
        mu::VecNum,
        cvg::Option{<:CoverageCounts} = nothing
    ) -> SimpleExpectedReturnsState

Keywords correspond to the struct's fields. A state seeded for `N` assets is `SimpleExpectedReturnsState(; mu = zeros(N))`, which [`partial_fit!`](@ref) builds when the `cache` field of the estimator holds `nothing`.

## Validation

  - `n >= 0`. A `DomainError` is thrown otherwise.
  - `!isempty(mu)`. An `IsEmptyError` is thrown otherwise.
  - Every entry of `mu` is finite. An `IsNonFiniteError` is thrown otherwise.

## View parameters

[`port_opt_view`](@ref) takes the view of its fields at the selected assets:

  - `mu`: Sliced to the selected indices through [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> PortfolioOptimisers.SimpleExpectedReturnsState(; mu = [0.0, 0.0])
PortfolioOptimisers.SimpleExpectedReturnsState
    n ┼ Int64: 0
   mu ┼ Vector{Float64}: [0.0, 0.0]
  cvg ┴ nothing
```

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`SimpleExpectedReturns`](@ref)
  - [`partial_fit!`](@ref)
  - [`merge_states`](@ref)
"""
@concrete struct SimpleExpectedReturnsState <: AbstractPartialFitState
    """
    $(field_dict[:pf_n])
    """
    n
    """
    $(field_dict[:pf_mu])
    """
    mu
    """
    $(field_dict[:pf_cvg])
    """
    cvg
end
function SimpleExpectedReturnsState(; n::Integer = 0, mu::VecNum,
                                    cvg::Option{<:CoverageCounts} = nothing)::SimpleExpectedReturnsState
    assert_partial_fit_state(n, mu)
    return SimpleExpectedReturnsState(n, mu, cvg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds two [`SimpleExpectedReturnsState`](@ref) fitted on disjoint blocks into the state of the concatenated block.

# Algorithm

 1. Refuse the pair with [`assert_mergeable_states`](@ref).
 2. If one of the two states holds no coverage counts, fold the counts and the means with [`chan_merge`](@ref). Its accumulator argument is `false`, which is the zero of a state that has no accumulator. Discard the accumulator that it returns.
 3. Otherwise, fold the mean of each asset with the per-asset counts `nu` of the two states. An asset with no observation in the two blocks keeps a mean of zero. Merge the coverage counts too.

# Arguments

  - `a`: The state of the first block of observations.
  - `b`: The state of the second block of observations.

# Validation

  - `a` and `b` pass [`assert_mergeable_states`](@ref).

# Returns

  - `state::SimpleExpectedReturnsState`: The state of the two blocks, as if one fit read them as one block.

# Related

  - [`SimpleExpectedReturnsState`](@ref)
  - [`merge_states`](@ref)
  - [`chan_merge`](@ref)
"""
function merge_states(a::SimpleExpectedReturnsState, b::SimpleExpectedReturnsState)
    assert_mergeable_states(a, b)
    ca, cb = a.cvg, b.cvg
    if isnothing(ca) || isnothing(cb)
        n, mu, _ = chan_merge(a.n, a.mu, false, b.n, b.mu, false)
        return SimpleExpectedReturnsState(n, mu, nothing)
    end
    nu = ca.nu .+ cb.nu
    mu = similar(a.mu)
    for i in eachindex(mu, nu)
        mu[i] = if iszero(nu[i])
            zero(eltype(mu))
        else
            a.mu[i] + (b.mu[i] - a.mu[i]) * (cb.nu[i] / nu[i])
        end
    end
    return SimpleExpectedReturnsState(a.n + b.n, mu,
                                      CoverageCounts(nu, nothing, copy(cb.active),
                                                     coverage_merge_stale(ca, cb, b.n)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies a [`SimpleExpectedReturnsState`](@ref), so the copy shares no array with the original.

This method is the `copy` method of the [`AbstractPartialFitState`](@ref) interface, which [`partial_fit`](@ref) calls before it folds. The count is a scalar, and the method uses it without change. The method copies the running mean and the coverage counts.

# Arguments

  - `x`: The state to copy.

# Returns

  - `state::SimpleExpectedReturnsState`: A new state, equal to `x`, whose `mu` is a new vector.

# Related

  - [`SimpleExpectedReturnsState`](@ref)
  - [`partial_fit`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
function Base.copy(x::SimpleExpectedReturnsState)
    return SimpleExpectedReturnsState(x.n, copy(x.mu),
                                      isnothing(x.cvg) ? nothing : copy(x.cvg))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Slices a [`SimpleExpectedReturnsState`](@ref) to the selected assets.

The Welford mean of one asset reads only the observations of that asset, so the slice of the state is the state of the sliced universe, entry for entry. All the assets share the count, so the method uses it without change. The slice copies by index and does not use a `view`. With a `view`, a later [`partial_fit!`](@ref) on the viewed estimator writes into the arrays of the original estimator.

# Arguments

  - `x`: The state to slice.
  - `i`: The index or the indices of the assets to keep.
  - `args...`: More positional arguments, which the method ignores.

# Returns

  - `state::SimpleExpectedReturnsState`: The state of the same sample over the selected assets.

# Related

  - [`SimpleExpectedReturnsState`](@ref)
  - [`port_opt_view`](@ref)
  - [`partial_fit!`](@ref)
"""
function port_opt_view(x::SimpleExpectedReturnsState, i, args...)
    return SimpleExpectedReturnsState(x.n, x.mu[i], coverage_counts_view(x.cvg, i))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The [`SimpleExpectedReturnsState`](@ref) method of [`partial_fit!`](@ref) folds one observation into the count and the running mean.

# Mathematical definition

```math
\\begin{align}
n &\\leftarrow n + 1\\\\
\\boldsymbol{d} &= \\boldsymbol{x} - \\boldsymbol{\\mu}\\\\
\\boldsymbol{\\mu} &\\leftarrow \\boldsymbol{\\mu} + \\frac{\\boldsymbol{d}}{n}\\, .
\\end{align}
```

Where:

  - ``n``: The observation count.
  - ``\\boldsymbol{x}``: The observation.
  - ``\\boldsymbol{\\mu}``: The running mean.
  - ``\\boldsymbol{d}``: The deviation of the observation from the mean before the fold.

# Algorithm

 1. Refuse an observation whose length is not the number of assets that the state describes.
 2. Add one to the count.
 3. Add the deviation, divided by the new count, to `mu` in place.
 4. Build the state with the new count through its positional constructor, and return it.
"""
function partial_fit!(state::SimpleExpectedReturnsState, x::VecNum)
    @argcheck(length(x) == length(state.mu),
              DimensionMismatch("the observation must have one entry per asset, but the state describes $(length(state.mu)) assets and `x` has $(length(x)) entries."))
    n = state.n + 1
    state.mu .+= (x .- state.mu) ./ n
    return SimpleExpectedReturnsState(n, state.mu, state.cvg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

This is the `Nothing` method of the coverage arm of [`partial_fit!`](@ref). An estimator that holds no [`CoveragePolicy`](@ref) folds through [`partial_fit!(state::SimpleExpectedReturnsState, x::VecNum)`](@ref). A plain state holds one count for all the assets, so it cannot skip an inactive cell. [`inactive_as_nonfinite`](@ref) writes `NaN` in that cell, and the mean of the asset is then `NaN`. The Asset Panel method gives the same `NaN` for an asset outside the Coverage Universe of the same rows.

# Related

  - [`partial_fit!`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`SimpleExpectedReturnsState`](@ref)
"""
function partial_fit!(state::SimpleExpectedReturnsState, x::VecNum, ::Nothing,
                      active_mask::Option{<:AbstractVector{<:Bool}})
    return partial_fit!(state, inactive_as_nonfinite(x, active_mask))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

This is the [`CoveragePolicy`](@ref) method of the coverage arm of [`partial_fit!`](@ref). It folds one observation into the count and the running mean of each asset. Each asset reads only its own observations.

# Mathematical definition

```math
\\begin{align}
\\nu_j &\\leftarrow \\nu_j + 1\\\\
\\mu_j &\\leftarrow \\mu_j + \\frac{r_{tj} - \\mu_j}{\\nu_j}\\, ,
\\end{align}
```

for each asset ``j`` that is finite and active at observation ``t``. The method applies neither line to an asset that is not finite and active. Where:

  - ``\\nu_j``: The number of observations at which asset ``j`` is finite and active.
  - $(math_dict[:r_tj])
  - ``\\mu_j``: The running mean of asset ``j``.

An asset with no observation keeps ``\\mu_j = 0`` and ``\\nu_j = 0``. `mean(me)` with no data returns `NaN` for it, so no caller reads the zero as an estimate. This is the Welford recursion for each asset. A mean that folds one observation at a time is the mean of the same rows in one block fit.

# Algorithm

 1. Refuse an observation whose length is not the number of assets that the state describes.
 2. Read the valid assets, and the assets that became inactive at this observation, with [`coverage_valid`](@ref).
 3. Apply the fold-time rule of the coverage algorithm with [`fold_inactive!`](@ref).
 4. Fold the return of each valid asset into the count and the mean of that asset.
 5. Update the per-asset records with [`coverage_step!`](@ref). Add one to the observation count, and return the state.

# Arguments

  - `state`: The state to fold into. The method changes it in place.
  - `x`: One observation, with one entry for each asset.
  - `cvg`: The policy that the estimator holds.
  - `active_mask`: The active mask of the Asset Panel at this observation, or `nothing`.

# Validation

  - `length(x)` is the number of assets that the state describes. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `state::SimpleExpectedReturnsState`: The state after the observation.

# Related

  - [`partial_fit!`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`coverage_valid`](@ref)
  - [`fold_inactive!`](@ref)
  - [`coverage_step!`](@ref)
"""
function partial_fit!(state::SimpleExpectedReturnsState, x::VecNum, cvg::CoveragePolicy,
                      active_mask::Option{<:AbstractVector{<:Bool}})
    @argcheck(length(x) == length(state.mu),
              DimensionMismatch("the observation must have one entry per asset, but the state describes $(length(state.mu)) assets and `x` has $(length(x)) entries."))
    counts = state.cvg
    valid, ni = coverage_valid(x, active_mask, counts)
    state = fold_inactive!(cvg.alg, state, ni)
    counts = state.cvg
    for i in eachindex(x, valid)
        if valid[i]
            nu = counts.nu[i] + 1
            counts.nu[i] = nu
            state.mu[i] += (x[i] - state.mu[i]) / nu
        end
    end
    coverage_step!(counts, valid, active_mask)
    return SimpleExpectedReturnsState(state.n + 1, state.mu, counts)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

This is the [`SimpleExpectedReturnsState`](@ref) method of [`fold_inactive!`](@ref) under [`ResetCoverage`](@ref). It sets to zero the count, the centre and the running mean of each asset that became inactive at this observation. A relisted asset then starts cold. The centre of a per-asset state is `nothing`, because its `mu` is already the centre of the cell. [`coverage_reset!`](@ref) does nothing to a `nothing`.

# Related

  - [`fold_inactive!`](@ref)
  - [`ResetCoverage`](@ref)
  - [`SimpleExpectedReturnsState`](@ref)
"""
function fold_inactive!(::ResetCoverage, state::SimpleExpectedReturnsState,
                        ni::AbstractVector{<:Bool})
    coverage_reset!(state.cvg.nu, ni)
    coverage_reset!(state.cvg.centre, ni)
    coverage_reset!(state.mu, ni)
    return state
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds every observation of a block into the partial-fit state of a [`SimpleExpectedReturns`](@ref) estimator.

This method is the block arm of the [`partial_fit!`](@ref) interface. The Welford update reads one observation at a time, so the method folds the block one row at a time. The result equals the result of the same rows given one at a time.

# Algorithm

 1. Orient `X` and `active_mask` to `observations × assets`. Transpose them when `dims == 2`.
 2. Fold each row in order with the single-observation arm of [`partial_fit!`](@ref), together with its row of the active mask. Rebind the estimator after each row.

# Arguments

  - `me`: The expected returns estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `active_mask`: The active mask of the Asset Panel, with the same size as `X`, or `nothing`.

# Validation

  - $(val_dict[:dims])
  - If `active_mask` is not `nothing`, its size is the size of `X` after the orientation. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `me::SimpleExpectedReturns`: The estimator that holds the state after the last row.

# Related

  - [`SimpleExpectedReturns`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit!(me::SimpleExpectedReturns{<:Any, <:Any,
                                                <:Option{<:SimpleExpectedReturnsState}},
                      X::MatNum; dims::Int = 1,
                      active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing)
    X = dims_oriented(dims, X)
    amsk = isnothing(active_mask) ? nothing : dims_oriented(dims, active_mask)
    if !isnothing(amsk)
        @argcheck(size(amsk) == size(X),
                  DimensionMismatch("size(X) ($(size(X))) must match size(active_mask) ($(size(amsk)))"))
    end
    for i in axes(X, 1)
        me = partial_fit!(me, view(X, i, :);
                          active_mask = isnothing(amsk) ? nothing : view(amsk, i, :))
    end
    return me
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The [`SimpleExpectedReturns`](@ref) method of [`partial_fit!`](@ref) folds one observation into the state of the `cache` field. On the first call, it seeds that state.

# Algorithm

 1. Refuse a configuration that no incremental fit can reproduce, with [`assert_partial_fittable`](@ref).
 2. Seed a [`SimpleExpectedReturnsState`](@ref) of zeros over `length(x)` assets when `me.cache` holds `nothing`, with [`expected_returns_state_seed`](@ref).
 3. Fold `x` into the state, with the policy `me.cvg` and the active mask.
 4. Rebuild the estimator with the new state in `cache` through [`rebuild_estimator`](@ref), and return it.
"""
function partial_fit!(me::SimpleExpectedReturns{<:Any, <:Any,
                                                <:Option{<:SimpleExpectedReturnsState}},
                      x::VecNum; active_mask::Option{<:AbstractVector{<:Bool}} = nothing)
    assert_partial_fittable(me, me.w, "SimpleExpectedReturns")
    state = expected_returns_state_seed(me.cache, x, me.cvg)
    return rebuild_estimator(me, (; cache = partial_fit!(state, x, me.cvg, active_mask)))
end
"""
    expected_returns_state_seed(cache::Nothing, x::VecNum,
                                cvg::Option{<:CoveragePolicy} = nothing) -> SimpleExpectedReturnsState
    expected_returns_state_seed(cache::SimpleExpectedReturnsState, x::VecNum,
                                cvg::Option{<:CoveragePolicy} = nothing) -> SimpleExpectedReturnsState

Returns the [`SimpleExpectedReturnsState`](@ref) that an incremental mean fit folds into. If the estimator holds no state, the function seeds a state of zeros.

The seed is in this function and not in [`partial_fit!`](@ref), so the fold is one line, and the dispatch on the type of the `cache` field is in one place.

# Arguments

  - `cache`: The state that the estimator holds, or `nothing`.
  - `x`: One observation, `assets × 1`. The function reads only its length and its element type.
  - `cvg`: The policy that the estimator holds. It decides if the seed holds per-asset counts.

# Returns

  - `state::SimpleExpectedReturnsState`: The state that `cache` holds, or a state of zeros over `length(x)` assets.

# Related

  - [`SimpleExpectedReturnsState`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`coverage_counts_seed`](@ref)
  - [`partial_fit!`](@ref)
"""
function expected_returns_state_seed(::Nothing, x::VecNum,
                                     cvg::Option{<:CoveragePolicy} = nothing)
    N = length(x)
    Tf = float_if_integer(eltype(x))
    return SimpleExpectedReturnsState(0, zeros(Tf, N),
                                      coverage_counts_seed(cvg, nothing, N, Tf, false))
end
function expected_returns_state_seed(cache::SimpleExpectedReturnsState, x::VecNum,
                                     cvg::Option{<:CoveragePolicy} = nothing)
    counts = coverage_counts_seed(cvg, cache.cvg, length(x), float_if_integer(eltype(x)),
                                  false)
    return if counts === cache.cvg
        cache
    else
        SimpleExpectedReturnsState(cache.n, cache.mu, counts)
    end
end
"""
    Statistics.mean(
        me::SimpleExpectedReturns,
        state::SimpleExpectedReturnsState
    ) -> VecNum
    Statistics.mean(
        me::SimpleExpectedReturns
    ) -> VecNum

Read the mean of an incremental fit from a [`SimpleExpectedReturnsState`](@ref).

The two-argument method reads a state that the caller holds. The one-argument method reads the state in the `cache` field of `me`. Both methods return the running mean as a vector, `assets × 1`. The batch method over a matrix returns a row when `dims = 1`.

# Algorithm

 1. Refuse a configuration that no incremental fit can reproduce, with [`assert_partial_fittable`](@ref).
 2. Return a vector of `NaN` when the state holds no observation. `min_obs` reads an asset with too few observations in the same way.
 3. Otherwise return the running mean.

# Arguments

  - $(arg_dict[:me])
  - `state`: The state to read.

# Validation

  - `me` holds no observation weights. An `ArgumentError` is thrown otherwise.
  - `me.cache` is not `nothing`, for the one-argument method. An `ArgumentError` is thrown otherwise.

# Returns

  - `mu::VecNum`: The running mean of the fit, `assets × 1`, with `NaN` where the state holds no observation.

# Examples

```jldoctest
julia> me = foldl(partial_fit!, eachrow([1.0 2.0; 3.0 4.0]); init = SimpleExpectedReturns());

julia> mean(me)
2-element Vector{Float64}:
 2.0
 3.0
```

# Related

  - [`SimpleExpectedReturns`](@ref)
  - [`SimpleExpectedReturnsState`](@ref)
  - [`partial_fit!`](@ref)
  - [`mean(me::SimpleExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.mean(me::SimpleExpectedReturns, state::SimpleExpectedReturnsState)
    assert_partial_fittable(me, me.w, "SimpleExpectedReturns")
    return coverage_mean(me, me.cvg, state)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

This is the `Nothing` method of [`coverage_mean`](@ref) for a state. `mean(me)` with no data calls it. All the assets share one count, so the result is all `NaN` before the first observation, and the running mean after it.

The method returns a copy of the running mean, not the vector in the state. `partial_fit!` writes the Welford recursion into `state.mu` in place. A returned reference to that vector changes at the next fold, with no message to the caller. A prior that puts that `mu` into a Result then has a Result that changes after the next observation. The [`CoveragePolicy`](@ref) method of this file copies for the same reason, through [`coverage_frame`](@ref).

# Related

  - [`coverage_mean`](@ref)
  - [`coverage_frame`](@ref)
  - [`SimpleExpectedReturnsState`](@ref)
"""
function coverage_mean(::SimpleExpectedReturns, ::Nothing,
                       state::SimpleExpectedReturnsState)
    return if state.n >= one(state.n)
        copy(state.mu)
    else
        fill(convert(eltype(state.mu), NaN), length(state.mu))
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

This is the [`CoveragePolicy`](@ref) method of [`coverage_mean`](@ref) for a state. `mean(me)` with no data calls it. The method reads the mean of each asset with the count of that asset. An asset that the policy refuses is `NaN`.

# Algorithm

 1. Read the admitted assets with [`coverage_admission`](@ref).
 2. Put the running mean in a frame with [`coverage_frame`](@ref), which copies the mean of each asset that has an observation. A running per-asset mean is already the ratio, so the method does not divide it again.

# Related

  - [`coverage_mean`](@ref)
  - [`coverage_admission`](@ref)
  - [`coverage_frame`](@ref)
"""
function coverage_mean(::SimpleExpectedReturns, cvg::CoveragePolicy,
                       state::SimpleExpectedReturnsState)
    counts = state.cvg
    return coverage_frame(state.mu, counts.nu, coverage_admission(cvg, counts, state.n))
end
function Statistics.mean(me::SimpleExpectedReturns)
    return Statistics.mean(me, partial_fit_cache(me))
end
# Every configuration of this family folds: the recursion is Welford's and reads the
# running mean alone (see [`supports_partial_fit`](@ref)).
function supports_partial_fit(::SimpleExpectedReturns)
    return true
end
export SimpleExpectedReturns, mean
