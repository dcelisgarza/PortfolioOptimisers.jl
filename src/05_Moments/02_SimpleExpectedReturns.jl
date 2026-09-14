"""
$(DocStringExtensions.TYPEDEF)

Computes the expected returns as the sample mean of the asset returns.

`w` carries optional observation weights. If `w` is `nothing`, the mean is unweighted. This is the default expected returns estimator throughout the library.

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

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `w`: Replaced with the incoming [`ObsWeights`](@ref).
  - `cache`: Carried unchanged via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `cache`: Sliced to the selected assets via [`port_opt_view`](@ref).

## Observation weight parameters

When [`obs_weights_view`](@ref) is called on this type, the following fields are automatically indexed to the selected observations:

  - `w`: Indexed to the selected observations via [`obs_weights_view`](@ref).
  - `cache`: Dropped via [`obs_weights_view`](@ref), because no slice of a state exists on the observation axis.

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

The state a `cache` holds is the running detail of an incremental fit, not the configuration a reader looks the type up for, and it prints under the estimator at every site that renders one. Set `set_show_nothing_fields!(:SimpleExpectedReturns, true)` to render it. ADR 0105 records the decision. `cvg` is read from the instance rather than from the type, because an opt-in that a caller has not taken is not part of the configuration they chose: an estimator whose `cvg` is `nothing` renders exactly as it did before the field existed, and one that carries a [`CoveragePolicy`](@ref) renders it.

# Arguments

  - `me`: Expected returns estimator, read for its `cvg` field.

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

Compute the mean of asset returns using a [`SimpleExpectedReturns`](@ref) estimator.

This method computes the expected returns as the sample mean of the input data `X` according to `me`.

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

  - ``\\hat{\\boldsymbol{\\mu}}``: ``N \\times 1`` vector of estimated expected returns, whose ``j``-th entry is ``\\hat{\\mu}_j``.
  - $(math_dict[:mu_hat_j])
  - $(math_dict[:r_tj])
  - $(math_dict[:T])
  - $(math_dict[:w_t_obs])

# Algorithm

 1. Check that `dims` is `1` or `2`.
 2. Resolve the observation weights from `me.w` against `X`, giving `w`.
 3. When `w` is `nothing`, take the unweighted mean of `X` along `dims`.
 4. Otherwise take the mean of `X` weighted by `w` along `dims`.

# Arguments

  - $(arg_dict[:me])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to [`Statistics.mean`](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.mean).

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
"""
    coverage_mean(me, cvg, X; dims::Int = 1, active_mask = nothing, kwargs...) -> ArrNum

Routes a sample-mean fit to the Coverage Universe arm or to the available-case arm.

The `cvg` field of the estimator is passed as the second argument, so the arm is chosen by **dispatch on the policy** rather than by a branch on its value. An estimator that carries no policy therefore pays nothing for the field: the `Nothing` method is the body [`mean(me::SimpleExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref) has always had, and it is statically resolved because the field's type is concrete.

# Arguments

  - $(arg_dict[:me])
  - `cvg`: The policy the estimator carries, which selects the arm.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `active_mask`: The active mask of the Asset Panel, `observations × assets`, or `nothing`. The Coverage Universe arm ignores it.
  - `kwargs...`: Additional keyword arguments passed to [`Statistics.mean`](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.mean).

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

`Nothing` method of [`coverage_mean`](@ref). The Coverage Universe arm, which is the sample mean of a window every asset covers, and the answer [`mean(me::SimpleExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref) gave before the `cvg` field existed. It refuses a gapped sample with [`assert_finite_sample`](@ref) and ignores the active mask, because an estimator with no policy reads the universe through [`coverage_reduction`](@ref) and never sees a gap.

# Related

  - [`coverage_mean`](@ref)
  - [`SimpleExpectedReturns`](@ref)
  - [`assert_finite_sample`](@ref)
"""
function coverage_mean(me::SimpleExpectedReturns, ::Nothing, X::MatNum; dims::Int = 1,
                       active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    assert_dims(dims)
    assert_finite_sample(X)
    w = get_observation_weights(me.w, X; dims = dims, kwargs...)
    return if isnothing(w)
        Statistics.mean(X; dims = dims)
    else
        Statistics.mean(X, w; dims = dims)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`CoveragePolicy`](@ref) method of [`coverage_mean`](@ref). The available-case arm: each asset's mean is fitted on that asset's own finite and active observations, and the batch answer **is** the incremental one, because the arm folds the block through [`partial_fit!`](@ref) and reads the state out.

Writing the batch arm as the fold is what makes the map's oracle hold to the last bit rather than to a tolerance: there is one recursion, and a caller who hands the same rows over one at a time reaches the same floating-point number.

# Algorithm

 1. Fold every row of `X` into a fresh state with [`partial_fit!`](@ref), carrying the active mask.
 2. Read the state out with [`mean(me::SimpleExpectedReturns, state::SimpleExpectedReturnsState)`](@ref).
 3. Orient the answer as the caller's `dims` asks.

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

Carries the running observation count and mean of an incremental sample-mean fit.

The state of [`SimpleExpectedReturns`](@ref) under [`partial_fit!`](@ref). It holds no second-moment accumulator, because a mean is the whole estimate, so [`merge_states`](@ref) folds the two counts and the two means and discards the accumulator [`chan_merge`](@ref) returns.

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

When [`port_opt_view`](@ref) is called on this type, its fields are subset to the selected assets:

  - `mu`: Sliced to the selected indices via [`port_opt_view`](@ref).

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
 2. Fold the counts and the means with [`chan_merge`](@ref), whose accumulator argument is `false`, the zero of a state that carries no accumulator. The accumulator it returns is discarded.

# Arguments

  - `a`: The state of the first block of observations.
  - `b`: The state of the second block of observations.

# Validation

  - `a` and `b` pass [`assert_mergeable_states`](@ref).

# Returns

  - `state::SimpleExpectedReturnsState`: The state the two blocks give when they are fitted as one block.

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

The `copy` method of the [`AbstractPartialFitState`](@ref) interface, which [`partial_fit`](@ref) calls before it folds. The count is a scalar and passes through, and the running mean is copied.

# Arguments

  - `x`: The state to copy.

# Returns

  - `state::SimpleExpectedReturnsState`: A fresh state, equal to `x`, whose `mu` is a fresh vector.

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

The Welford mean of one asset reads that asset's observations alone, so the slice of the state is the state of the sliced universe, entry for entry, and the count is shared by every asset and passes through. The slice copies by index and does not `view`: a later [`partial_fit!`](@ref) on the viewed estimator would otherwise write through into the arrays of the estimator the view was taken from.

# Arguments

  - `x`: The state to slice.
  - `i`: Index or indices of the assets to keep.
  - `args...`: Additional positional arguments (ignored).

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

[`SimpleExpectedReturnsState`](@ref) method of [`partial_fit!`](@ref). Folds one observation into the running count and mean.

# Mathematical definition

```math
\\begin{align}
n &\\leftarrow n + 1\\\\
\\boldsymbol{d} &= \\boldsymbol{x} - \\boldsymbol{\\mu}\\\\
\\boldsymbol{\\mu} &\\leftarrow \\boldsymbol{\\mu} + \\frac{\\boldsymbol{d}}{n}\\, .
\\end{align}
```

Where:

  - ``n``: observation count.
  - ``\\boldsymbol{x}``: the observation.
  - ``\\boldsymbol{\\mu}``: the running mean.
  - ``\\boldsymbol{d}``: deviation of the observation from the mean before the fold.

# Algorithm

 1. Refuse an observation whose length is not the number of assets the state describes.
 2. Add one to the count.
 3. Move `mu` in place along the deviation, by the reciprocal of the new count.
 4. Rebind the count with `Accessors.@reset`, and return the state.
"""
function partial_fit!(state::SimpleExpectedReturnsState, x::VecNum)
    @argcheck(length(x) == length(state.mu),
              DimensionMismatch("the observation must have one entry per asset, but the state describes $(length(state.mu)) assets and `x` has $(length(x)) entries."))
    n = state.n + 1
    state.mu .+= (x .- state.mu) ./ n
    return Accessors.@reset state.n = n
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

`Nothing` method of the coverage arm of [`partial_fit!`](@ref). An estimator that carries no [`CoveragePolicy`](@ref) folds through [`partial_fit!(state::SimpleExpectedReturnsState, x::VecNum)`](@ref), and the active mask is ignored: a plain state carries no per-cell count for the mask to gate, and its universe is the Coverage Universe the read-out already reduces to.

# Related

  - [`partial_fit!`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`SimpleExpectedReturnsState`](@ref)
"""
function partial_fit!(state::SimpleExpectedReturnsState, x::VecNum, ::Nothing,
                      ::Option{<:AbstractVector{<:Bool}})
    return partial_fit!(state, x)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`CoveragePolicy`](@ref) method of the coverage arm of [`partial_fit!`](@ref). Folds one observation into the running per-asset count and mean, reading each asset's own observations alone.

# Mathematical definition

```math
\\begin{align}
\\nu_j &\\leftarrow \\nu_j + 1\\\\
\\mu_j &\\leftarrow \\mu_j + \\frac{r_{tj} - \\mu_j}{\\nu_j}\\, ,
\\end{align}
```

for every asset ``j`` that is finite and active at observation ``t``, and neither line for an asset that is not. Where:

  - ``\\nu_j``: the number of observations at which asset ``j`` was finite and active.
  - $(math_dict[:r_tj])
  - ``\\mu_j``: the running mean of asset ``j``.

An asset with no observation keeps ``\\mu_j = 0`` and ``\\nu_j = 0``, and the read-out answers `NaN` for it, so the zero is never read as an estimate. This is Welford's recursion per asset, so a mean folded observation by observation is the mean of the same rows fitted as a block.

# Algorithm

 1. Refuse an observation whose length is not the number of assets the state describes.
 2. Read the valid assets and the newly inactive ones with [`coverage_valid`](@ref).
 3. Apply the algorithm's fold-time rule with [`fold_inactive!`](@ref).
 4. Fold each valid asset's return into its own count and mean.
 5. Move the per-asset bookkeeping on with [`coverage_step!`](@ref), add one to the observation count, and return the state.

# Arguments

  - `state`: The state to fold into, mutated in place.
  - `x`: One observation, one entry per asset.
  - `cvg`: The policy the estimator carries.
  - `active_mask`: The active mask of the Asset Panel at this observation, or `nothing`.

# Validation

  - `length(x)` is the number of assets the state describes. A `DimensionMismatch` is thrown otherwise.

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
    return Accessors.@reset state.n = state.n + 1
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`SimpleExpectedReturnsState`](@ref) method of [`fold_inactive!`](@ref) under [`ResetCoverage`](@ref). Zeroes the count, the centre and the running mean of every asset that has just gone inactive, so that a relisting starts the asset cold. The centre of a per-asset state is `nothing`, because its `mu` is already the cell's centre, and [`coverage_reset!`](@ref) passes that through.

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

The block arm of the [`partial_fit!`](@ref) interface. Welford's update reads one observation at a time, so the block is folded row by row and the answer is the answer of the same rows handed over one at a time.

# Algorithm

 1. Orient `X` to `observations × assets`, transposing it when `dims == 2`.
 2. Fold each row in turn with the single-observation arm of [`partial_fit!`](@ref), rebinding the estimator each time.

# Arguments

  - `me`: Expected returns estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])

# Validation

  - $(val_dict[:dims])

# Returns

  - `me::SimpleExpectedReturns`: The estimator carrying the state after the last row.

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

[`SimpleExpectedReturns`](@ref) method of [`partial_fit!`](@ref). Folds one observation into the state the `cache` field carries, seeding it on the first call.

# Algorithm

 1. Refuse a configuration no incremental fit reproduces, with [`assert_partial_fittable`](@ref).
 2. Seed a [`SimpleExpectedReturnsState`](@ref) of zeros over `length(x)` assets when `me.cache` holds `nothing`, with [`expected_returns_state_seed`](@ref).
 3. Fold `x` into the state.
 4. Rebind `me.cache` with `Accessors.@reset`, and return the estimator.
"""
function partial_fit!(me::SimpleExpectedReturns{<:Any, <:Any,
                                                <:Option{<:SimpleExpectedReturnsState}},
                      x::VecNum; active_mask::Option{<:AbstractVector{<:Bool}} = nothing)
    assert_partial_fittable(me, me.w, "SimpleExpectedReturns")
    state = expected_returns_state_seed(me.cache, x, me.cvg)
    return Accessors.@reset me.cache = partial_fit!(state, x, me.cvg, active_mask)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the [`SimpleExpectedReturnsState`](@ref) an incremental mean fit folds into, seeding one of zeros when the estimator carries none.

The seed is written here rather than inside [`partial_fit!`](@ref), so the fold reads as one line and the branch that reads the `cache` field has one home.

# Arguments

  - `cache`: The state the estimator carries, or `nothing`.
  - `x`: One observation, `assets × 1`, read for its length and its element type.
  - `cvg`: The policy the estimator carries, which decides whether the seed carries per-asset counts.

# Returns

  - `state::SimpleExpectedReturnsState`: The state `cache` holds, or a state of zeros over `length(x)` assets.

# Related

  - [`SimpleExpectedReturnsState`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`coverage_counts_seed`](@ref)
  - [`partial_fit!`](@ref)
"""
function expected_returns_state_seed(cache::Option{<:SimpleExpectedReturnsState}, x::VecNum,
                                     cvg::Option{<:CoveragePolicy} = nothing)
    N = length(x)
    Tf = typeof(zero(eltype(x)) / one(Int))
    return if isnothing(cache)
        SimpleExpectedReturnsState(0, zeros(Tf, N),
                                   coverage_counts_seed(cvg, nothing, N, Tf, false))
    else
        Accessors.@reset cache.cvg = coverage_counts_seed(cvg, cache.cvg, N, Tf, false)
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

Read the mean of an incremental fit out of a [`SimpleExpectedReturnsState`](@ref).

The two-argument method reads a state the caller holds, and the one-argument method reads the state the `cache` field of `me` carries. Both return the running mean as a vector, `assets × 1`, where the batch method over a matrix returns a row when `dims = 1`.

# Algorithm

 1. Refuse a configuration no incremental fit reproduces, with [`assert_partial_fittable`](@ref).
 2. Return a vector of `NaN` when the state holds no observation, in the way `min_obs` reads an asset with too few observations.
 3. Otherwise return the running mean.

# Arguments

  - $(arg_dict[:me])
  - `state`: The state to read.

# Validation

  - `me` carries no observation weights. An `ArgumentError` is thrown otherwise.
  - `me.cache` is not `nothing`, for the one-argument method. An `ArgumentError` is thrown otherwise.

# Returns

  - `mu::VecNum`: Running mean of the fit, `assets × 1`, or `NaN` where the state holds no observation.

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

`Nothing` method of the read-out arm of [`coverage_mean`](@ref). Every asset shares one count, so the whole answer is `NaN` before the first observation and the running mean afterwards.

The running mean is **copied** rather than handed out. `partial_fit!` writes the Welford recursion into `state.mu` in place, so a read-out that returned the accumulator itself would hand the caller a vector that the next fold silently rewrites — and a prior that read its `mu` out and carried it into a Result would find the Result changed under it at the next observation. The [`CoveragePolicy`](@ref) method beside this one copies for the same reason, through [`coverage_frame`](@ref).

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

[`CoveragePolicy`](@ref) method of the read-out arm of [`coverage_mean`](@ref). Each asset's mean is read out against that asset's own count, and an asset the policy refuses is `NaN`.

# Algorithm

 1. Read the admitted assets with [`coverage_admission`](@ref).
 2. Frame the running mean with [`coverage_frame`](@ref), which copies it where its asset has an observation. A running per-asset mean is already the ratio, so it is never divided again.

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
