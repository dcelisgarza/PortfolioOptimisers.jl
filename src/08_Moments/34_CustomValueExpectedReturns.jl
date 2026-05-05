"""
$(DocStringExtensions.TYPEDEF)

Expected returns estimator that returns custom values for each asset.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CustomValueExpectedReturns(;
        val::Func_Num_VecNum = 0.0
    ) -> CustomValueExpectedReturns

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> CustomValueExpectedReturns()
CustomValueExpectedReturns
  val ┴ Float64: 0.0
```

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`PortfolioOptimisersCovariance`](@ref)
"""
@concrete struct CustomValueExpectedReturns <: AbstractExpectedReturnsEstimator
    "Custom value.\n  - If a scalar, all assets are assigned this value.\n    - If a vector, each element corresponds to an asset.\n    - If a function, it is evaluated for each asset from `X` using passed through `kwargs`."
    val
    function CustomValueExpectedReturns(val::Func_Num_VecNum)
        if isa(val, VecNum)
            @argcheck(!isempty(val))
        end
        return new{typeof(val)}(val)
    end
end
function CustomValueExpectedReturns(; val::Func_Num_VecNum = 0.0)
    return CustomValueExpectedReturns(val)
end
"""
    Statistics.mean(me::CustomValueExpectedReturns, X::MatNum;
                    dims::Int = 1, kwargs...)

Compute expected returns as custom values.

# Arguments

  - `me`: Custom value expected returns estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `mu::Matrix{<:Num_VecNum}`: Expected returns vector, shaped as `(1, N)` if `dims == 1` or `(N, 1)` if `dims == 2`.

# Related

  - [`CustomValueExpectedReturns`](@ref)
"""
function Statistics.mean(me::CustomValueExpectedReturns{<:Number}, X::MatNum; dims::Int = 1,
                         kwargs...)
    @argcheck(dims in (1, 2))
    return insertdims(fill(me.val, size(X, setdiff((1, 2), (dims,))[1])); dims = dims)
end
function Statistics.mean(me::CustomValueExpectedReturns{<:VecNum}, X::MatNum; dims::Int = 1,
                         kwargs...)
    @argcheck(dims in (1, 2))
    @argcheck(length(me.val) == size(X, setdiff((1, 2), (dims,))[1]))
    return insertdims(me.val; dims = dims)
end
function Statistics.mean(me::CustomValueExpectedReturns{<:Function}, X::MatNum;
                         dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2))
    return me.val(X; dims = dims, kwargs...)
end

export CustomValueExpectedReturns
