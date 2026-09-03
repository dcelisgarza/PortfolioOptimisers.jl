"""
$(DocStringExtensions.TYPEDEF)

Computes the marginal variance and standard deviation, optionally weighted and optionally bias-corrected.

`me` centres the data when no `mean` is supplied, `w` weights the observations, and `corrected` selects the bias correction. `me` reaches the matrix methods only: the vector methods leave the centring to `Statistics`.

`w` weights the whole estimate, so it reaches the centre as well as the deviations. The matrix methods send `me` through [`factory`](@ref), which replaces the weights of `me` with `w`, and `Statistics` centres a weighted vector on its weighted mean. Both paths therefore answer the same number over the same data, and `w` wins over the weights that `me` carries. Pass `mean` for a centre that `w` does not describe. ADR 0088 records the decision.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SimpleVariance(;
        me::Option{<:AbstractExpectedReturnsEstimator} = SimpleExpectedReturns(),
        w::Option{<:ObsWeights} = nothing,
        corrected::Bool = true,
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> SimpleVariance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])
  - `corrected = true` needs a weight type that carries a bias correction. See the bias-correction bullet of [`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref).

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `me`: Recursively updated via [`factory`](@ref).
  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `me`: Recursively viewed via [`port_opt_view`](@ref).

## Observation weight parameters

When [`obs_weights_view`](@ref) is called on this type, the following fields are automatically indexed to the selected observations:

  - `me`: Recursively indexed via [`obs_weights_view`](@ref).
  - `w`: Indexed to the selected observations via [`obs_weights_view`](@ref).

# Examples

```jldoctest
julia> SimpleVariance()
SimpleVariance
         me ┼ SimpleExpectedReturns
            │   w ┴ nothing
          w ┼ nothing
  corrected ┴ Bool: true

julia> SimpleVariance(; w = StatsBase.Weights([0.2, 0.3, 0.5]), corrected = false)
SimpleVariance
         me ┼ SimpleExpectedReturns
            │   w ┴ nothing
          w ┼ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.2, 0.3, 0.5]
  corrected ┴ Bool: false
```

# Related

  - [`AbstractVarianceEstimator`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`SimpleExpectedReturns`](@ref)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
  - [`std(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`std(ve::SimpleVariance, X::VecNum; mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::VecNum; mean = nothing)`](@ref)
  - [`SimpleVarianceState`](@ref)
  - [`partial_fit!`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
  - [`obs_weights_view`](@ref)
"""
@propagatable @concrete struct SimpleVariance <: AbstractVarianceEstimator
    """
    $(field_dict[:ome])
    """
    @fprop @vprop me
    """
    $(field_dict[:ow])
    """
    @wprop w
    """
    $(field_dict[:corrected])
    """
    corrected
    """
    $(field_dict[:pfcache])
    """
    cache
    function SimpleVariance(me::Option{<:AbstractExpectedReturnsEstimator},
                            w::Option{<:ObsWeights}, corrected::Bool,
                            cache::Option{<:AbstractPartialFitState})
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(me), typeof(w), typeof(corrected), typeof(cache)}(me, w,
                                                                            corrected,
                                                                            cache)
    end
end
function SimpleVariance(;
                        me::Option{<:AbstractExpectedReturnsEstimator} = SimpleExpectedReturns(),
                        w::Option{<:ObsWeights} = nothing, corrected::Bool = true,
                        cache::Option{<:AbstractPartialFitState} = nothing)::SimpleVariance
    return SimpleVariance(me, w, corrected, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a [`SimpleVariance`](@ref) except `cache`.

The state a `cache` holds is the running detail of an incremental fit, not the configuration a reader looks the type up for, and it prints under the estimator at every site that renders one. Set `set_show_nothing_fields!(:SimpleVariance, true)` to render it. ADR 0105 records the decision.

# Arguments

  - `::SimpleVariance`: Variance estimator, read for its type alone.

# Returns

  - `fields::Tuple`: The field names to render, which is `(:me, :w, :corrected)`.

# Related

  - [`SimpleVariance`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
show_fields(::SimpleVariance) = (:me, :w, :corrected)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Dispersion kernel shared by the [`SimpleVariance`](@ref) methods of `Statistics.std` and `Statistics.var`.

# Algorithm

The matrix method:

 1. Check that `dims` is `1` or `2`.
 2. Resolve the centring vector `mu` from `me` and `ve.w` with [`weighted_centre`](@ref), which reads `mean` when the caller gave one.
 3. Resolve the observation weights from `ve.w` against `X` with [`get_observation_weights`](@ref), giving `w`.
 4. When `w` is `nothing`, call `f(X; dims = dims, corrected = ve.corrected, mean = mu)`.
 5. Otherwise call `f(X, w, dims; corrected = ve.corrected, mean = mu)`.

The vector method:

 1. Resolve the observation weights from `ve.w` against `X`, giving `w`.
 2. When `w` is `nothing`, call `f(X; corrected = ve.corrected, mean = mean)`.
 3. Otherwise call `f(X, w; corrected = ve.corrected, mean = mean)`.

The two methods reach one centre by two routes. The matrix method resolves a centre before it calls `f`, and [`weighted_centre`](@ref) takes that centre from `me` after [`factory`](@ref) writes `ve.w` into it. The vector method passes `mean` through, so a `mean` of `nothing` leaves `f` to centre on the **weighted** mean of `X`. One `SimpleVariance` therefore answers a one-column matrix and the matching vector with one number. `ve.w` wins over the weights that `me` carries, which is what [`factory`](@ref) does on every other path. ADR 0088 records the decision, and `mean` takes any other centre.

[`weighted_centre`](@ref) calls [`factory`](@ref) only when `ve.w` is not `nothing`. That test is a performance guard and not a second contract: `ve.w` is a field, so its type decides the branch, and the guard keeps a windowed loop from rebuilding the estimator tree of `me` once per window.

# Arguments

  - `f`: Dispersion function to apply, either `Statistics.std` or `Statistics.var`.
  - `ve::SimpleVariance`: Variance estimator. Supplies the observation weights and the `corrected` flag.
  - `me::AbstractExpectedReturnsEstimator`: Expected returns estimator used when no `mean` is provided. Matrix methods only.
  - `X::VecNum_MatNum`: Data matrix or vector.
  - `dims::Int = 1`: Dimension along which to operate. Matrix methods only.
  - `mean = nothing`: Precomputed mean.
  - `kwargs...`: Forwarded to the mean and weight resolution. Matrix methods only.

# Validation

  - $(val_dict[:dims]) Matrix methods only.

# Returns

  - `sigma::Union{<:Number, <:ArrNum}`: Dispersion of `X` computed by `f`.

# Related

  - [`SimpleVariance`](@ref)
  - [`weighted_centre`](@ref)
  - [`get_observation_weights`](@ref)
"""
function simple_variance_kernel(f::F, ve::SimpleVariance,
                                me::AbstractExpectedReturnsEstimator, X::MatNum;
                                dims::Int = 1, mean = nothing, kwargs...) where {F}
    assert_dims(dims)
    mu = weighted_centre(X, me, ve.w; dims = dims, mean = mean, kwargs...)
    w = get_observation_weights(ve.w, X; dims = dims, kwargs...)
    return if isnothing(w)
        f(X; dims = dims, corrected = ve.corrected, mean = mu)
    else
        f(X, w, dims; corrected = ve.corrected, mean = mu)
    end
end
function simple_variance_kernel(f::F, ve::SimpleVariance, X::VecNum;
                                mean = nothing) where {F}
    w = get_observation_weights(ve.w, X)
    return if isnothing(w)
        f(X; corrected = ve.corrected, mean = mean)
    else
        f(X, w; corrected = ve.corrected, mean = mean)
    end
end
"""
    Statistics.std(
        ve::SimpleVariance,
        X::MatNum;
        dims::Int = 1,
        mean = nothing,
        kwargs...,
    ) -> ArrNum

Compute the standard deviation using a [`SimpleVariance`](@ref) estimator for a matrix.

This method computes the standard deviation of the input matrix `X` using the configuration specified in `ve`.

# Mathematical definition

```math
\\begin{align}
\\hat{\\sigma}_j &= \\sqrt{\\hat{\\sigma}^2_j}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\sigma}_j``: Estimated standard deviation of asset ``j``.
  - $(math_dict[:sigma2_hat_j])

[`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref) defines ``\\hat{\\sigma}^2_j`` in each of the four cases that `ve.w` and `ve.corrected` select.

# Algorithm

 1. Check that `dims` is `1` or `2`.
 2. When `mean` is `nothing`, compute the centring vector `mu` with `ve.me`, after [`factory`](@ref) writes `ve.w` into it; otherwise take `mu` from `mean`.
 3. Resolve the observation weights from `ve.w` against `X`, giving `w`.
 4. When `w` is `nothing`, take the unweighted standard deviation of `X` along `dims`, centred on `mu`.
 5. Otherwise take the standard deviation of `X` weighted by `w` along `dims`, centred on `mu`.

``\\hat{\\mu}_j`` comes from `ve.me`, and `ve.w` reaches `ve.me` through [`factory`](@ref). A `SimpleVariance` whose `w` is set therefore weights the centre and the squared deviations alike, so a vector and its one-column matrix answer the same number. Pass `mean` for any other centre. ADR 0088 records the decision.

# Arguments

  - $(arg_dict[:ve])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: Additional keyword arguments passed to the mean estimator.

# Validation

  - $(val_dict[:dims])
  - `corrected = true` needs a weight type that carries a bias correction. A plain `StatsBase.Weights` carries none, and `StatsBase` raises an `ArgumentError`.

# Returns

  - $(ret_dict[:stdarr])

# Examples

```jldoctest
julia> sv = SimpleVariance()
SimpleVariance
         me ┼ SimpleExpectedReturns
            │   w ┴ nothing
          w ┼ nothing
  corrected ┴ Bool: true

julia> Xmat = [1.0 2.0; 3.0 4.0];

julia> std(sv, Xmat; dims = 1)
1×2 Matrix{Float64}:
 1.41421  1.41421
```

# Related

  - [`SimpleVariance`](@ref)
  - [`Statistics.std`](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.std)
  - [`std(ve::SimpleVariance, X::VecNum; mean = nothing)`](@ref)
  - [`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::VecNum; mean = nothing)`](@ref)
"""
function Statistics.std(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing,
                        kwargs...)
    return simple_variance_kernel(Statistics.std, ve, ve.me, X; dims = dims, mean = mean,
                                  kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

`SimpleVariance{Nothing}` overload of [`std(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref). Uses [`SimpleExpectedReturns`](@ref) to compute the mean when none is provided, ignoring the `me` field.
"""
function Statistics.std(ve::SimpleVariance{Nothing}, X::MatNum; dims::Int = 1,
                        mean = nothing, kwargs...)
    return simple_variance_kernel(Statistics.std, ve, SimpleExpectedReturns(), X;
                                  dims = dims, mean = mean, kwargs...)
end
"""
    Statistics.std(
        ve::SimpleVariance,
        X::VecNum;
        mean = nothing
    ) -> Number

Compute the standard deviation using a [`SimpleVariance`](@ref) estimator for a vector.

This method computes the standard deviation of the input vector `X` using the configuration specified in `ve`.

# Mathematical definition

[`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref) defines the variance in each of the four cases that `ve.w` and `ve.corrected` select, and this method returns its square root.

# Algorithm

 1. Resolve the observation weights from `ve.w` against `X`, giving `w`.
 2. When `w` is `nothing`, take the unweighted standard deviation of `X`, centred on `mean`.
 3. Otherwise take the standard deviation of `X` weighted by `w`, centred on `mean`.

The vector methods ignore `ve.me`: a `mean` of `nothing` reaches `Statistics.std`, which centres on the mean of `X` — the **weighted** mean when `w` is not `nothing`. The matrix methods resolve the centre from `ve.me` under the same `ve.w`, so the two paths answer the same number for the same data. ADR 0088 records the decision.

# Arguments

  - $(arg_dict[:ve])
  - $(arg_dict[:Xv])
  - $(arg_dict[:omean])

# Validation

  - `corrected = true` needs a weight type that carries a bias correction. A plain `StatsBase.Weights` carries none, and `StatsBase` raises an `ArgumentError`.

# Returns

  - $(ret_dict[:stdnum])

# Examples

```jldoctest
julia> sv = SimpleVariance()
SimpleVariance
         me ┼ SimpleExpectedReturns
            │   w ┴ nothing
          w ┼ nothing
  corrected ┴ Bool: true

julia> X = [1.0, 2.0, 3.0];

julia> std(sv, X)
1.0

julia> svw = SimpleVariance(; w = StatsBase.Weights([0.2, 0.3, 0.5]), corrected = false)
SimpleVariance
         me ┼ SimpleExpectedReturns
            │   w ┴ nothing
          w ┼ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.2, 0.3, 0.5]
  corrected ┴ Bool: false

julia> std(svw, X)
0.7810249675906654
```

# Related

  - [`SimpleVariance`](@ref)
  - [`Statistics.std`](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.std)
  - [`std(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::VecNum; mean = nothing)`](@ref)
"""
function Statistics.std(ve::SimpleVariance, X::VecNum; mean = nothing)
    return simple_variance_kernel(Statistics.std, ve, X; mean = mean)
end
"""
    Statistics.var(
        ve::SimpleVariance,
        X::MatNum;
        dims::Int = 1,
        mean = nothing,
        kwargs...
    ) -> ArrNum

Compute the variance using a [`SimpleVariance`](@ref) estimator for a matrix.

This method computes the variance of the input matrix `X` using the configuration specified in `ve`.

# Mathematical definition

Unweighted, `corrected = true`:

```math
\\begin{align}
\\hat{\\sigma}^2_j &= \\frac{1}{T-1} \\sum_{t=1}^{T} (r_{tj} - \\hat{\\mu}_j)^2\\,.
\\end{align}
```

Unweighted, `corrected = false`:

```math
\\begin{align}
\\hat{\\sigma}^2_j &= \\frac{1}{T} \\sum_{t=1}^{T} (r_{tj} - \\hat{\\mu}_j)^2\\,.
\\end{align}
```

Weighted:

```math
\\begin{align}
\\hat{\\sigma}^2_j &= \\frac{\\sum_{t=1}^{T} w_t (r_{tj} - \\hat{\\mu}_j)^2}{\\sum_{t=1}^{T} w_t - c}\\,.
\\end{align}
```

Where:

  - $(math_dict[:sigma2_hat_j])
  - $(math_dict[:r_tj])
  - $(math_dict[:mu_hat_j])
  - $(math_dict[:T])
  - $(math_dict[:w_t_obs])
  - $(math_dict[:c_weight_bias])

# Algorithm

 1. Check that `dims` is `1` or `2`.
 2. When `mean` is `nothing`, compute the centring vector `mu` with `ve.me`, after [`factory`](@ref) writes `ve.w` into it; otherwise take `mu` from `mean`.
 3. Resolve the observation weights from `ve.w` against `X`, giving `w`.
 4. When `w` is `nothing`, take the unweighted variance of `X` along `dims`, centred on `mu`.
 5. Otherwise take the variance of `X` weighted by `w` along `dims`, centred on `mu`.

``\\hat{\\mu}_j`` comes from `ve.me`, and `ve.w` reaches `ve.me` through [`factory`](@ref). A `SimpleVariance` whose `w` is set therefore weights the centre and the squared deviations alike, so a vector and its one-column matrix answer the same number. Pass `mean` for any other centre. ADR 0088 records the decision.

# Arguments

  - $(arg_dict[:ve])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: Additional keyword arguments passed to the mean estimator.

# Validation

  - $(val_dict[:dims])
  - `corrected = true` needs a weight type that carries a bias correction. A plain `StatsBase.Weights` carries none, and `StatsBase` raises an `ArgumentError`.

# Returns

  - $(ret_dict[:vararr])

# Examples

```jldoctest
julia> sv = SimpleVariance()
SimpleVariance
         me ┼ SimpleExpectedReturns
            │   w ┴ nothing
          w ┼ nothing
  corrected ┴ Bool: true

julia> Xmat = [1.0 2.0; 3.0 4.0];

julia> var(sv, Xmat; dims = 1)
1×2 Matrix{Float64}:
 2.0  2.0
```

# Related

  - [`SimpleVariance`](@ref)
  - [`Statistics.var`](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.var)
  - [`std(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`std(ve::SimpleVariance, X::VecNum; mean = nothing)`](@ref)
  - [`var(ve::SimpleVariance, X::VecNum; mean = nothing)`](@ref)
"""
function Statistics.var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing,
                        kwargs...)
    return simple_variance_kernel(Statistics.var, ve, ve.me, X; dims = dims, mean = mean,
                                  kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

`SimpleVariance{Nothing}` overload of [`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref). Uses [`SimpleExpectedReturns`](@ref) to compute the mean when none is provided, ignoring the `me` field.
"""
function Statistics.var(ve::SimpleVariance{Nothing}, X::MatNum; dims::Int = 1,
                        mean = nothing, kwargs...)
    return simple_variance_kernel(Statistics.var, ve, SimpleExpectedReturns(), X;
                                  dims = dims, mean = mean, kwargs...)
end
"""
    Statistics.var(
        ve::SimpleVariance,
        X::VecNum;
        mean = nothing
    ) -> Number

Compute the variance using a [`SimpleVariance`](@ref) estimator for a vector.

This method computes the variance of the input vector `X` using the configuration specified in `ve`.

# Mathematical definition

[`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref) defines the variance in each of the four cases that `ve.w` and `ve.corrected` select, and this method returns it for a single series.

# Algorithm

 1. Resolve the observation weights from `ve.w` against `X`, giving `w`.
 2. When `w` is `nothing`, take the unweighted variance of `X`, centred on `mean`.
 3. Otherwise take the variance of `X` weighted by `w`, centred on `mean`.

The vector methods ignore `ve.me`: a `mean` of `nothing` reaches `Statistics.var`, which centres on the mean of `X` — the **weighted** mean when `w` is not `nothing`. The matrix methods resolve the centre from `ve.me` under the same `ve.w`, so the two paths answer the same number for the same data. ADR 0088 records the decision.

# Arguments

  - $(arg_dict[:ve])
  - $(arg_dict[:Xv])
  - $(arg_dict[:omean])

# Validation

  - `corrected = true` needs a weight type that carries a bias correction. A plain `StatsBase.Weights` carries none, and `StatsBase` raises an `ArgumentError`.

# Returns

  - $(ret_dict[:varnum])

# Examples

```jldoctest
julia> sv = SimpleVariance()
SimpleVariance
         me ┼ SimpleExpectedReturns
            │   w ┴ nothing
          w ┼ nothing
  corrected ┴ Bool: true

julia> X = [1.0, 2.0, 3.0];

julia> var(sv, X)
1.0

julia> svw = SimpleVariance(; w = StatsBase.Weights([0.2, 0.3, 0.5]), corrected = false)
SimpleVariance
         me ┼ SimpleExpectedReturns
            │   w ┴ nothing
          w ┼ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.2, 0.3, 0.5]
  corrected ┴ Bool: false

julia> var(svw, X)
0.61
```

# Related

  - [`SimpleVariance`](@ref)
  - [`Statistics.var`](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.var)
  - [`std(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`std(ve::SimpleVariance, X::VecNum; mean = nothing)`](@ref)
  - [`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.var(ve::SimpleVariance, X::VecNum; mean = nothing)
    return simple_variance_kernel(Statistics.var, ve, X; mean = mean)
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the running observation count, mean and per-asset second-moment accumulator of an incremental variance fit.

The state of [`SimpleVariance`](@ref) under [`partial_fit!`](@ref). `M` is the accumulator ``\\sum_t (r_{tj} - \\hat{\\mu}_j)^2`` and not the variance, so [`var(ve::SimpleVariance, state::SimpleVarianceState)`](@ref) divides it by the count, or by the count less one when `ve.corrected` holds.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SimpleVarianceState(;
        n::Integer = 0,
        mu::VecNum,
        M::VecNum = zeros(eltype(mu), length(mu))
    ) -> SimpleVarianceState

Keywords correspond to the struct's fields. A state seeded for `N` assets is `SimpleVarianceState(; mu = zeros(N))`, which [`partial_fit!`](@ref) builds when the `cache` field of the estimator holds `nothing`.

## Validation

  - `n >= 0`. A `DomainError` is thrown otherwise.
  - `!isempty(mu)`. An `IsEmptyError` is thrown otherwise.
  - Every entry of `mu` and of `M` is finite. An `IsNonFiniteError` is thrown otherwise.
  - `length(M) == length(mu)`. A `DimensionMismatch` is thrown otherwise.

## View parameters

When [`port_opt_view`](@ref) is called on this type, its fields are subset to the selected assets:

  - `mu`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `M`: Sliced to the selected indices via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> PortfolioOptimisers.SimpleVarianceState(; mu = [0.0, 0.0])
PortfolioOptimisers.SimpleVarianceState
   n ┼ Int64: 0
  mu ┼ Vector{Float64}: [0.0, 0.0]
   M ┴ Vector{Float64}: [0.0, 0.0]
```

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`SimpleVariance`](@ref)
  - [`partial_fit!`](@ref)
  - [`merge_states`](@ref)
"""
@concrete struct SimpleVarianceState <: AbstractPartialFitState
    """
    $(field_dict[:pf_n])
    """
    n
    """
    $(field_dict[:pf_mu])
    """
    mu
    """
    $(field_dict[:pf_M])
    """
    M
end
function SimpleVarianceState(; n::Integer = 0, mu::VecNum,
                             M::VecNum = zeros(eltype(mu), length(mu)))::SimpleVarianceState
    assert_partial_fit_state(n, mu, M)
    return SimpleVarianceState(n, mu, M)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds two [`SimpleVarianceState`](@ref) fitted on disjoint blocks into the state of the concatenated block.

# Algorithm

 1. Refuse the pair with [`assert_mergeable_states`](@ref).
 2. Fold the counts, the means and the accumulators with [`chan_merge`](@ref), whose elementwise method reads a per-asset accumulator.

# Arguments

  - `a`: The state of the first block of observations.
  - `b`: The state of the second block of observations.

# Validation

  - `a` and `b` pass [`assert_mergeable_states`](@ref).

# Returns

  - `state::SimpleVarianceState`: The state the two blocks give when they are fitted as one block.

# Related

  - [`SimpleVarianceState`](@ref)
  - [`merge_states`](@ref)
  - [`chan_merge`](@ref)
"""
function merge_states(a::SimpleVarianceState, b::SimpleVarianceState)
    assert_mergeable_states(a, b)
    n, mu, M = chan_merge(a.n, a.mu, a.M, b.n, b.mu, b.M)
    return SimpleVarianceState(n, mu, M)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`SimpleVarianceState`](@ref) method of [`partial_fit!`](@ref). Folds one observation into the running count, mean and per-asset accumulator.

# Mathematical definition

```math
\\begin{align}
n &\\leftarrow n + 1\\\\
\\boldsymbol{d} &= \\boldsymbol{x} - \\boldsymbol{\\mu}\\\\
\\boldsymbol{\\mu} &\\leftarrow \\boldsymbol{\\mu} + \\frac{\\boldsymbol{d}}{n}\\\\
\\boldsymbol{M} &\\leftarrow \\boldsymbol{M} + \\boldsymbol{d} \\odot (\\boldsymbol{x} - \\boldsymbol{\\mu})\\, .
\\end{align}
```

Where:

  - ``n``: observation count.
  - ``\\boldsymbol{x}``: the observation.
  - ``\\boldsymbol{\\mu}``: the running mean.
  - ``\\boldsymbol{d}``: deviation of the observation from the mean **before** the fold.
  - ``\\boldsymbol{M}``: the running per-asset accumulator.

The last line reads ``\\boldsymbol{\\mu}`` **after** the third line moved it, where ``\\boldsymbol{d}`` read it before. That asymmetry is Welford's, and it is what keeps the accumulator non-negative.

# Algorithm

 1. Refuse an observation whose length is not the number of assets the state describes.
 2. Add one to the count.
 3. Take the deviation of the observation from the mean before the fold, giving `d`.
 4. Move `mu` in place along `d`, by the reciprocal of the new count.
 5. Add `d` times the deviation from the mean **after** the fold to `M`, in place.
 6. Rebind the count with `Accessors.@reset`, and return the state.
"""
function partial_fit!(state::SimpleVarianceState, x::VecNum)
    @argcheck(length(x) == length(state.mu),
              DimensionMismatch("the observation must have one entry per asset, but the state describes $(length(state.mu)) assets and `x` has $(length(x)) entries."))
    n = state.n + 1
    d = x .- state.mu
    state.mu .+= d ./ n
    state.M .+= d .* (x .- state.mu)
    return Accessors.@reset state.n = n
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds every observation of a block into the partial-fit state of a [`SimpleVariance`](@ref) estimator.

The block arm of the [`partial_fit!`](@ref) interface. Welford's update reads one observation at a time, so the block is folded row by row and the answer is the answer of the same rows handed over one at a time.

# Algorithm

 1. Orient `X` to `observations × assets`, transposing it when `dims == 2`.
 2. Fold each row in turn with the single-observation arm of [`partial_fit!`](@ref), rebinding the estimator each time.

# Arguments

  - `ve`: Variance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])

# Validation

  - $(val_dict[:dims])

# Returns

  - `ve::SimpleVariance`: The estimator carrying the state after the last row.

# Related

  - [`SimpleVariance`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit!(ve::SimpleVariance, X::MatNum; dims::Int = 1)
    X = dims_oriented(dims, X)
    for i in axes(X, 1)
        ve = partial_fit!(ve, view(X, i, :))
    end
    return ve
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`SimpleVariance`](@ref) method of [`partial_fit!`](@ref). Folds one observation into the state the `cache` field carries, seeding it on the first call.

# Algorithm

 1. Refuse a configuration no incremental fit reproduces, with [`assert_partial_fittable`](@ref).
 2. Seed a [`SimpleVarianceState`](@ref) of zeros over `length(x)` assets when `ve.cache` holds `nothing`, with [`variance_state_seed`](@ref).
 3. Fold `x` into the state.
 4. Rebind `ve.cache` with `Accessors.@reset`, and return the estimator.
"""
function partial_fit!(ve::SimpleVariance, x::VecNum)
    assert_partial_fittable(ve.me, ve.w, "SimpleVariance")
    return Accessors.@reset ve.cache = partial_fit!(variance_state_seed(ve.cache, x), x)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the [`SimpleVarianceState`](@ref) an incremental variance fit folds into, seeding one of zeros when the estimator carries none.

The seed is written here rather than inside [`partial_fit!`](@ref), so the fold reads as one line and the branch that reads the `cache` field has one home.

# Arguments

  - `cache`: The state the estimator carries, or `nothing`.
  - `x`: One observation, `assets × 1`, read for its length and its element type.

# Returns

  - `state::SimpleVarianceState`: The state `cache` holds, or a state of zeros over `length(x)` assets.

# Related

  - [`SimpleVarianceState`](@ref)
  - [`partial_fit!`](@ref)
"""
function variance_state_seed(cache::Option{<:SimpleVarianceState}, x::VecNum)
    return if isnothing(cache)
        SimpleVarianceState(0, zeros(eltype(x), length(x)), zeros(eltype(x), length(x)))
    else
        cache
    end
end
"""
    Statistics.var(
        ve::SimpleVariance,
        state::SimpleVarianceState
    ) -> VecNum
    Statistics.var(
        ve::SimpleVariance
    ) -> VecNum

Read the variance of an incremental fit out of a [`SimpleVarianceState`](@ref).

The two-argument method reads a state the caller holds, and the one-argument method reads the state the `cache` field of `ve` carries. Both return the per-asset variance as a vector, `assets × 1`, where the batch method over a matrix returns a row when `dims = 1`.

# Mathematical definition

```math
\\begin{align}
\\hat{\\sigma}^2_j &= \\frac{M_j}{n - c}\\,.
\\end{align}
```

Where:

  - $(math_dict[:sigma2_hat_j])
  - ``M_j``: running accumulator of asset ``j``.
  - ``n``: observation count.
  - ``c``: one when `ve.corrected` holds, and zero otherwise.

# Algorithm

 1. Refuse a configuration no incremental fit reproduces, with [`assert_partial_fittable`](@ref).
 2. Take the divisor `n - c`, and return a vector of `NaN` when it is below one, in the way `min_obs` reads an asset with too few observations.
 3. Otherwise divide the accumulator by the divisor.

# Arguments

  - $(arg_dict[:ve])
  - `state`: The state to read.

# Validation

  - `ve` carries no observation weights. An `ArgumentError` is thrown otherwise.
  - `ve.me` is a [`SimpleExpectedReturns`](@ref) carrying no observation weights, or `nothing`. An `ArgumentError` is thrown otherwise.
  - `ve.cache` is not `nothing`, for the one-argument method. An `ArgumentError` is thrown otherwise.

# Returns

  - `vr::VecNum`: Per-asset variance of the fit, `assets × 1`, or `NaN` where the state holds too few observations.

# Examples

```jldoctest
julia> ve = foldl(partial_fit!, eachrow([1.0 2.0; 3.0 4.0]); init = SimpleVariance());

julia> var(ve)
2-element Vector{Float64}:
 2.0
 2.0
```

# Related

  - [`SimpleVariance`](@ref)
  - [`SimpleVarianceState`](@ref)
  - [`partial_fit!`](@ref)
  - [`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.var(ve::SimpleVariance, state::SimpleVarianceState)
    assert_partial_fittable(ve.me, ve.w, "SimpleVariance")
    k = state.n - ve.corrected
    return k >= one(k) ? state.M ./ k : fill(convert(eltype(state.M), NaN), length(state.M))
end
function Statistics.var(ve::SimpleVariance)
    return Statistics.var(ve, partial_fit_cache(ve))
end
export SimpleVariance, var, std
