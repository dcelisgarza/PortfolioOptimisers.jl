"""
$(DocStringExtensions.TYPEDEF)

A simple expected returns estimator for `PortfolioOptimisers.jl`, representing the sample mean with optional observation weights.

`SimpleExpectedReturns` is the standard estimator for computing expected returns as the possibly weighted mean of asset returns.

# Fields

$(DocStringExtensions.FIELDS)

# Constructor

    SimpleExpectedReturns(; w::Option{<:StatsBase.AbstractWeights} = nothing,
                           idx::Option{<:VecInt} = nothing)

Keywords correspond to the struct's fields.

## Validation

    - $(val_dict[:oow])
    - $(val_dict[:oidx])

# Examples

```jldoctest
julia> SimpleExpectedReturns()
SimpleExpectedReturns
    w ┼ nothing
  idx ┴ nothing

julia> SimpleExpectedReturns(; w = StatsBase.Weights([0.5, 0.5]))
SimpleExpectedReturns
    w ┼ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.5, 0.5]
  idx ┴ nothing
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
    "$(field_dict[:oidx])"
    idx
    function SimpleExpectedReturns(w::Option{<:StatsBase.AbstractWeights},
                                   idx::Option{<:VecInt})
        assert_nonempty_finite_val(w, :w)
        assert_nonempty_gt0_finite_val(idx, :idx)
        return new{typeof(w), typeof(idx)}(w, idx)
    end
end
function SimpleExpectedReturns(; w::Option{<:StatsBase.AbstractWeights} = nothing,
                               idx::Option{<:VecInt} = nothing)
    return SimpleExpectedReturns(w, idx)
end
"""
    Statistics.mean(
        me::SimpleExpectedReturns,
        X::MatNum;
        dims::Int = 1,
        kwargs...
    ) -> ArrNum

Compute the mean of asset returns using a [`SimpleExpectedReturns`](@ref) estimator.

This method computes the expected returns as the sample mean of the input data `X`, optionally using observation weights stored in the estimator. If no weights are provided, the unweighted mean is computed.

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
    w ┼ nothing
  idx ┴ nothing

julia> mean(ser, X)
1×2 Matrix{Float64}:
 0.02  0.03

julia> serw = SimpleExpectedReturns(; w = StatsBase.Weights([0.2, 0.8]))
SimpleExpectedReturns
    w ┼ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.2, 0.8]
  idx ┴ nothing

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
function Statistics.mean(me::SimpleExpectedReturns{<:Any, Nothing}, X::MatNum;
                         dims::Int = 1, kwargs...)
    return if isnothing(me.w)
        Statistics.mean(X; dims = dims)
    else
        Statistics.mean(X, me.w; dims = dims)
    end
end
function Statistics.mean(me::SimpleExpectedReturns{<:Any, <:VecInt}, X::MatNum;
                         dims::Int = 1, kwargs...)
    X = view(X, me.idx, :)
    return if isnothing(me.w)
        Statistics.mean(X; dims = dims)
    else
        Statistics.mean(X, me.w; dims = dims)
    end
end
"""
    factory(
        me::SimpleExpectedReturns,
        w::StatsBase.AbstractWeights
    ) -> SimpleExpectedReturns

Create a new `SimpleExpectedReturns` estimator with observation weights `w`.

This function constructs a new [`SimpleExpectedReturns`](@ref) object, replacing the weights stored in the input estimator with the provided weights.

# Arguments

  - $(arg_dict[:me])
  - $(arg_dict[:ow])

# Returns

  - $(ret_dict[:me])

# Related

  - [`SimpleExpectedReturns`](@ref)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
  - [`mean(me::SimpleExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function factory(me::SimpleExpectedReturns, w::StatsBase.AbstractWeights)
    return SimpleExpectedReturns(; w = w, idx = me.idx)
end

export SimpleExpectedReturns, mean
