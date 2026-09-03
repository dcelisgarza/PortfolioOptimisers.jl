"""
$(DocStringExtensions.TYPEDEF)

Computes the expected returns as the sample mean of the asset returns.

`w` carries optional observation weights. If `w` is `nothing`, the mean is unweighted. This is the default expected returns estimator throughout the library.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SimpleExpectedReturns(;
        w::Option{<:ObsWeights} = nothing,
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> SimpleExpectedReturns

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `cache`: Recursively viewed via [`port_opt_view`](@ref).

## Observation weight parameters

When [`obs_weights_view`](@ref) is called on this type, the following fields are automatically indexed to the selected observations:

  - `w`: Indexed to the selected observations via [`obs_weights_view`](@ref).

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
    $(field_dict[:pfcache])
    """
    cache
    function SimpleExpectedReturns(w::Option{<:ObsWeights},
                                   cache::Option{<:AbstractPartialFitState})::SimpleExpectedReturns
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(w), typeof(cache)}(w, cache)
    end
end
function SimpleExpectedReturns(; w::Option{<:ObsWeights} = nothing,
                               cache::Option{<:AbstractPartialFitState} = nothing)::SimpleExpectedReturns
    return SimpleExpectedReturns(w, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a [`SimpleExpectedReturns`](@ref) except `cache`.

The state a `cache` holds is the running detail of an incremental fit, not the configuration a reader looks the type up for, and it prints under the estimator at every site that renders one. Set `set_show_nothing_fields!(:SimpleExpectedReturns, true)` to render it. ADR 0105 records the decision.

# Arguments

  - `::SimpleExpectedReturns`: Expected returns estimator, read for its type alone.

# Returns

  - `fields::Tuple`: The field names to render, which is `(:w,)`.

# Related

  - [`SimpleExpectedReturns`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
show_fields(::SimpleExpectedReturns) = (:w,)
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
function Statistics.mean(me::SimpleExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)
    assert_dims(dims)
    w = get_observation_weights(me.w, X; dims = dims, kwargs...)
    return if isnothing(w)
        Statistics.mean(X; dims = dims)
    else
        Statistics.mean(X, w; dims = dims)
    end
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
        mu::VecNum
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
  mu ┴ Vector{Float64}: [0.0, 0.0]
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
end
function SimpleExpectedReturnsState(; n::Integer = 0,
                                    mu::VecNum)::SimpleExpectedReturnsState
    assert_partial_fit_state(n, mu)
    return SimpleExpectedReturnsState(n, mu)
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
    n, mu, _ = chan_merge(a.n, a.mu, false, b.n, b.mu, false)
    return SimpleExpectedReturnsState(n, mu)
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
function partial_fit!(me::SimpleExpectedReturns, X::MatNum; dims::Int = 1)
    X = dims_oriented(dims, X)
    for i in axes(X, 1)
        me = partial_fit!(me, view(X, i, :))
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
function partial_fit!(me::SimpleExpectedReturns, x::VecNum)
    assert_partial_fittable(me, me.w, "SimpleExpectedReturns")
    return Accessors.@reset me.cache = partial_fit!(expected_returns_state_seed(me.cache,
                                                                                x), x)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the [`SimpleExpectedReturnsState`](@ref) an incremental mean fit folds into, seeding one of zeros when the estimator carries none.

The seed is written here rather than inside [`partial_fit!`](@ref), so the fold reads as one line and the branch that reads the `cache` field has one home.

# Arguments

  - `cache`: The state the estimator carries, or `nothing`.
  - `x`: One observation, `assets × 1`, read for its length and its element type.

# Returns

  - `state::SimpleExpectedReturnsState`: The state `cache` holds, or a state of zeros over `length(x)` assets.

# Related

  - [`SimpleExpectedReturnsState`](@ref)
  - [`partial_fit!`](@ref)
"""
function expected_returns_state_seed(cache::Option{<:SimpleExpectedReturnsState}, x::VecNum)
    return if isnothing(cache)
        SimpleExpectedReturnsState(0, zeros(eltype(x), length(x)))
    else
        cache
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
    return if state.n >= one(state.n)
        state.mu
    else
        fill(convert(eltype(state.mu), NaN), length(state.mu))
    end
end
function Statistics.mean(me::SimpleExpectedReturns)
    return Statistics.mean(me, partial_fit_cache(me))
end
export SimpleExpectedReturns, mean
