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
        corrected::Bool = true
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
    function SimpleVariance(me::Option{<:AbstractExpectedReturnsEstimator},
                            w::Option{<:ObsWeights}, corrected::Bool)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(me), typeof(w), typeof(corrected)}(me, w, corrected)
    end
end
function SimpleVariance(;
                        me::Option{<:AbstractExpectedReturnsEstimator} = SimpleExpectedReturns(),
                        w::Option{<:ObsWeights} = nothing,
                        corrected::Bool = true)::SimpleVariance
    return SimpleVariance(me, w, corrected)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Dispersion kernel shared by the [`SimpleVariance`](@ref) methods of `Statistics.std` and `Statistics.var`.

# Algorithm

The matrix method:

 1. Check that `dims` is `1` or `2`.
 2. When `mean` is `nothing`, compute the centring vector `mu` with `me`; otherwise take `mu` from `mean`. When `ve.w` is not `nothing`, send `me` through [`factory`](@ref) first, so that `mu` carries the same weights as the deviations.
 3. Resolve the observation weights from `ve.w` against `X` with [`get_observation_weights`](@ref), giving `w`.
 4. When `w` is `nothing`, call `f(X; dims = dims, corrected = ve.corrected, mean = mu)`.
 5. Otherwise call `f(X, w, dims; corrected = ve.corrected, mean = mu)`.

The vector method:

 1. Resolve the observation weights from `ve.w` against `X`, giving `w`.
 2. When `w` is `nothing`, call `f(X; corrected = ve.corrected, mean = mean)`.
 3. Otherwise call `f(X, w; corrected = ve.corrected, mean = mean)`.

The two methods reach one centre by two routes. The matrix method resolves a centre before it calls `f`, and it takes that centre from `me` after [`factory`](@ref) writes `ve.w` into it. The vector method passes `mean` through, so a `mean` of `nothing` leaves `f` to centre on the **weighted** mean of `X`. One `SimpleVariance` therefore answers a one-column matrix and the matching vector with one number. `ve.w` wins over the weights that `me` carries, which is what [`factory`](@ref) does on every other path. ADR 0088 records the decision, and `mean` takes any other centre.

Step 2 calls [`factory`](@ref) only when `ve.w` is not `nothing`. That test is a performance guard and not a second contract: `ve.w` is a field, so its type decides the branch, and the guard keeps a windowed loop from rebuilding the estimator tree of `me` once per window.

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
  - [`get_observation_weights`](@ref)
"""
function simple_variance_kernel(f::F, ve::SimpleVariance,
                                me::AbstractExpectedReturnsEstimator, X::MatNum;
                                dims::Int = 1, mean = nothing, kwargs...) where {F}
    assert_dims(dims)
    mu = if !isnothing(mean)
        mean
    elseif isnothing(ve.w)
        Statistics.mean(me, X; dims = dims, kwargs...)
    else
        Statistics.mean(factory(me, ve.w), X; dims = dims, kwargs...)
    end
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
  - $(math_dict[:w_t_moment])
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
export SimpleVariance, var, std
