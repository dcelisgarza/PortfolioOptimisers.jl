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
    """
    Custom value.\n  - If a scalar, all assets are assigned this value.\n    - If a vector, each element corresponds to an asset.\n    - If a function, it is called with the full `X` matrix and `dims`, with additional keyword arguments passed through.
    """
    val
    function CustomValueExpectedReturns(val::Func_Num_VecNum)
        if isa(val, VecNum)
            @argcheck(!isempty(val), IsEmptyError)
        end
        return new{typeof(val)}(val)
    end
end
function CustomValueExpectedReturns(;
                                    val::Func_Num_VecNum = 0.0)::CustomValueExpectedReturns
    return CustomValueExpectedReturns(val)
end
"""
    Statistics.mean(me::CustomValueExpectedReturns, X::MatNum;
                    dims::Int = 1, kwargs...)

Compute expected returns as custom values.

# Mathematical definition

Returns a user-supplied constant, vector, or function result as the expected returns:

```math
\\begin{align}
\\hat{\\mu}_j &= v_j, \\quad j = 1, \\ldots, N\\,.
\\end{align}
```

Where:

  - ``\\hat{\\mu}_j``: Expected return of asset ``j``.
  - ``v_j``: ``j``-th element of the custom value `me.val` (broadcast from a scalar, taken directly from a vector, or evaluated from a callable).
  - $(math_dict[:N])

# Arguments

  - `me`: Custom value expected returns estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `mu::Matrix{<:Number}`: Expected returns matrix, shaped as `(1, N)` if `dims == 1` or `(N, 1)` if `dims == 2`.

# Related

  - [`CustomValueExpectedReturns`](@ref)
"""
function Statistics.mean(me::CustomValueExpectedReturns{<:Number}, X::MatNum; dims::Int = 1,
                         kwargs...)
    @argcheck(dims in (1, 2), DomainError(dims, "dims must be 1 or 2"))
    return insertdims(fill(me.val, size(X, setdiff((1, 2), (dims,))[1])); dims = dims)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Vector overload of [`mean(me::CustomValueExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref). Returns the stored vector `me.val` reshaped to match `dims`.
"""
function Statistics.mean(me::CustomValueExpectedReturns{<:VecNum}, X::MatNum; dims::Int = 1,
                         kwargs...)
    @argcheck(dims in (1, 2), DomainError(dims, "dims must be 1 or 2"))
    _ncols = size(X, setdiff((1, 2), (dims,))[1])
    @argcheck(length(me.val) == _ncols,
              DimensionMismatch("length(val) ($(length(me.val))) must match the number of assets ($_ncols)"))
    return insertdims(me.val; dims = dims)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Function overload of [`mean(me::CustomValueExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref). Delegates to the callable `me.val` with the same arguments.
"""
function Statistics.mean(me::CustomValueExpectedReturns{<:Function}, X::MatNum;
                         dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2), DomainError(dims, "dims must be 1 or 2"))
    return me.val(X; dims = dims, kwargs...)
end

export CustomValueExpectedReturns
