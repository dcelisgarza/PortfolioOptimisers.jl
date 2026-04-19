"""
$(DocStringExtensions.TYPEDEF)

Expected returns estimator that returns the asset Medians.

`CustomValueExpectedReturns` computes "expected returns" as the Median of each asset, as estimated by the underlying covariance estimator. This can be useful in certain risk-based portfolio construction approaches where the expected return proxy is the asset's volatility.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CustomValueExpectedReturns(;
        val::Number = 0.0
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
    "Custom value."
    val
    function CustomValueExpectedReturns(val::Number)
        return new{typeof(val)}(val)
    end
end
function CustomValueExpectedReturns(; val::Number = 0.0)
    return CustomValueExpectedReturns(val)
end
"""
    Statistics.mean(me::CustomValueExpectedReturns, X::MatNum;
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
function Statistics.mean(me::CustomValueExpectedReturns, X::MatNum; dims::Int = 1,
                         kwargs...)
    @argcheck(dims in (1, 2))
    return insertdims(fill(me.val, size(X, setdiff((1, 2), (dims,)))); dims = dims)
end

export CustomValueExpectedReturns
