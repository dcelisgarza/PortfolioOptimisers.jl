"""
$(DocStringExtensions.TYPEDEF)

Expected returns estimator that returns the asset Medians.

`CustomValueExpectedReturns` computes "expected returns" as the Median of each asset, as estimated by the underlying covariance estimator. This can be useful in certain risk-based portfolio construction approaches where the expected return proxy is the asset's volatility.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CustomValueExpectedReturns(;
        w::Option{<:ObsWeights} = nothing
    ) -> CustomValueExpectedReturns

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> CustomValueExpectedReturns()
CustomValueExpectedReturns
  ce ┼ PortfolioOptimisersCovariance
     │   ce ┼ Covariance
     │      │    me ┼ SimpleExpectedReturns
     │      │       │   w ┴ nothing
     │      │    ce ┼ GeneralCovariance
     │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
     │      │       │    w ┴ nothing
     │      │   alg ┴ Full()
     │   mp ┼ DenoiseDetoneAlgMatrixProcessing
     │      │     pdm ┼ Posdef
     │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
     │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
     │      │      dn ┼ nothing
     │      │      dt ┼ nothing
     │      │     alg ┼ nothing
     │      │   order ┴ DenoiseDetoneAlg()
```

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`PortfolioOptimisersCovariance`](@ref)
"""
@concrete struct CustomValueExpectedReturns <: AbstractExpectedReturnsEstimator
    "$(field_dict[:oow])"
    w
    function CustomValueExpectedReturns(w::Option{<:ObsWeights})
        return new{typeof(w)}(w)
    end
end
function CustomValueExpectedReturns(; w::Option{<:ObsWeights} = nothing)
    return CustomValueExpectedReturns(w)
end
"""
    factory(ce::CustomValueExpectedReturns, w::ObsWeights) -> CustomValueExpectedReturns

    Return a new [`CustomValueExpectedReturns`](@ref) estimator with observation weights `w` applied to the underlying covariance estimator.

# Arguments

  - `ce`: Median expected returns estimator.
  - $(arg_dict[:ow])

# Returns

  - `me::CustomValueExpectedReturns`: Updated estimator with weights applied.

# Related

  - [`CustomValueExpectedReturns`](@ref)
  - [`factory`](@ref)
"""
function factory(::CustomValueExpectedReturns, w::ObsWeights)
    return CustomValueExpectedReturns(; w = w)
end
"""
    Statistics.mean(me::CustomValueExpectedReturns, X::AbstractMatrix{<:Real};
                    dims::Int = 1, kwargs...)

Compute expected returns as the Median of each asset.

This method returns the Median vector of `X` as estimated by the covariance estimator `me.ce`.

# Arguments

  - `me`: Median expected returns estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the covariance estimator.

# Returns

  - `mu::Matrix{<:Number}`: Median vector, shaped as `(1, N)` if `dims == 1` or `(N, 1)` if `dims == 2`.

# Related

  - [`CustomValueExpectedReturns`](@ref)
"""
function Statistics.mean(me::CustomValueExpectedReturns{Nothing}, X::AbstractMatrix{<:Real};
                         dims::Int = 1, kwargs...)
    return Statistics.median(X; dims = dims)
end
function Statistics.mean(me::CustomValueExpectedReturns{<:ObsWeights},
                         X::AbstractMatrix{<:Real}; dims::Int = 1, kwargs...)
    @argcheck(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    w = get_observation_weights(me.w, X)
    Y = Vector{eltype(X)}(undef, size(X, 2))
    for i in axes(X, 2)
        Y[i] = Statistics.median(view(X, :, i), w)
    end
    return insertdims(Y; dims = dims)
end

export CustomValueExpectedReturns
