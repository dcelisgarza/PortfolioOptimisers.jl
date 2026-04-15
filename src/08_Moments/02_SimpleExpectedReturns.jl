"""
$(DocStringExtensions.TYPEDEF)

A simple expected returns estimator for `PortfolioOptimisers.jl`, representing the sample mean with optional observation weights.

`SimpleExpectedReturns` is the standard estimator for computing expected returns as the possibly weighted mean of asset returns.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SimpleExpectedReturns(;
        w::Option{<:ObsWeights} = nothing
    ) -> SimpleExpectedReturns

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])

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
"""
@concrete struct SimpleExpectedReturns <: AbstractExpectedReturnsEstimator
    "$(field_dict[:oow])"
    w
    function SimpleExpectedReturns(w::Option{<:ObsWeights})
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(w)}(w)
    end
end
function SimpleExpectedReturns(; w::Option{<:ObsWeights} = nothing)
    return SimpleExpectedReturns(w)
end
"""
    Statistics.mean(
        me::SimpleExpectedReturns,
        X::MatNum;
        dims::Int = 1,
        kwargs...
    ) -> ArrNum

Compute the mean of asset returns using a [`SimpleExpectedReturns`](@ref) estimator.

This method computes the expected returns as the sample mean of the input data `X` according to `ce`.

# Arguments

  - $(arg_dict[:me])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to [`Statistics.mean`](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.mean).

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
    w = get_observation_weights(me.w, X; dims = dims, kwargs...)
    return if isnothing(w)
        Statistics.mean(X; dims = dims)
    else
        Statistics.mean(X, w; dims = dims)
    end
end
"""
    factory(
        me::SimpleExpectedReturns,
        w::ObsWeights
    ) -> SimpleExpectedReturns

Create a new `SimpleExpectedReturns` estimator with observation weights `w`.

This function constructs a new [`SimpleExpectedReturns`](@ref) object, replacing the weights stored in the input estimator with the provided weights.

# Arguments

  - $(arg_dict[:me])
  - $(arg_dict[:ow])

# Returns

  - $(ret_dict[:me])

# Examples

```jldoctest
julia> me = SimpleExpectedReturns()
SimpleExpectedReturns
  w ┴ nothing

julia> factory(me, StatsBase.Weights([0.1, 0.2, 0.7]))
SimpleExpectedReturns
  w ┴ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.1, 0.2, 0.7]
```

# Related

  - [`SimpleExpectedReturns`](@ref)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
"""
function factory(::SimpleExpectedReturns, w::ObsWeights)
    return SimpleExpectedReturns(; w = w)
end

export SimpleExpectedReturns, mean
