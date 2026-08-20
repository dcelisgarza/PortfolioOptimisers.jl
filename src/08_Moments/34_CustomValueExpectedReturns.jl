"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for custom expected returns value algorithms. These are used to define the type of the `val` field in [`CustomValueExpectedReturns`](@ref).

# Interfaces

In order to implement a new concrete custom expected returns value algorithm type that works seamlessly with the library, subtype `CustomExpectedReturnsValueAlgorithm` and make it callable such that it returns a vector of length consistent with the returns matrix:

  - `(alg::MyCustomValAlg)(X::MatNum; dims::Int=1, kwargs...) -> Vector{<:Number}`

Where:

  - `alg`: Custom expected returns value algorithm.
  - `X`: Data matrix of asset returns.
  - `dims`: Dimension along which to compute the expected returns (1 for columns/assets, 2 for rows/observations).
  - `kwargs...`: Additional keyword arguments.

# Examples

```jldoctest
julia> struct MyCustomValAlg <: PortfolioOptimisers.CustomExpectedReturnsValueAlgorithm end

julia> function (alg::MyCustomValAlg)(X::PortfolioOptimisers.MatNum; dims::Int = 1, kwargs...)
           return fill(0.0, size(X, setdiff((1, 2), (dims,))[1]))
       end

julia> mean(CustomValueExpectedReturns(; val = MyCustomValAlg()), [1 2 3; 4 5 6])
3-element Vector{Float64}:
 0.0
 0.0
 0.0
```

# Related

  - [`CustomValueExpectedReturns`](@ref)
  - [`CER_Func_Num_VecNum`](@ref)
"""
abstract type CustomExpectedReturnsValueAlgorithm <: AbstractCustomValue end
"""
    const CER_Func_Num_VecNum = Union{<:CustomExpectedReturnsValueAlgorithm,<:Func_Num_VecNum}

Alias for supported types for the `val` field in [`CustomValueExpectedReturns`](@ref).
"""
const CER_Func_Num_VecNum = Union{<:CustomExpectedReturnsValueAlgorithm, <:Func_Num_VecNum}
"""
$(DocStringExtensions.TYPEDEF)

Expected returns estimator that returns custom values for each asset.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CustomValueExpectedReturns(;
        val::CER_Func_Num_VecNum = 0.0
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
    function CustomValueExpectedReturns(val::CER_Func_Num_VecNum)
        if isa(val, VecNum)
            @argcheck(!isempty(val), IsEmptyError)
        end
        return new{typeof(val)}(val)
    end
end
function CustomValueExpectedReturns(;
                                    val::CER_Func_Num_VecNum = 0.0)::CustomValueExpectedReturns
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
    assert_dims(dims)
    return insertdims(fill(me.val, size(X, setdiff((1, 2), (dims,))[1])); dims = dims)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that a custom expected returns value is a per-asset vector of the expected length.

Both the vector field of [`CustomValueExpectedReturns`](@ref) and the value returned by a callable `val` must be a vector of numbers with one element per asset. The callable is checked at the point of call, which is the only seam that can see what the callable returned.

# Arguments

  - `val`: Custom value to validate. Either the stored `me.val` vector or the value returned by a callable `me.val`.
  - `N`: Number of assets implied by the data matrix and `dims`.
  - `val_sym`: Symbolic name used in the error messages.

# Returns

  - `nothing`.

# Validation

  - `isa(val, VecNum)`.
  - `length(val) == N`.

# Details

  - Throws `ArgumentError` if `val` is not a vector of numbers.
  - Throws `DimensionMismatch` if the length of `val` does not match the number of assets.

# Related

  - [`CustomValueExpectedReturns`](@ref)
  - [`CustomExpectedReturnsValueAlgorithm`](@ref)
"""
function assert_custom_expected_returns_val(val, N::Integer,
                                            val_sym::Sym_Str = :val)::Nothing
    @argcheck(isa(val, VecNum),
              ArgumentError("$val_sym must be a vector of numbers, one element per asset. Got\n$val_sym => $(typeof(val))"))
    @argcheck(length(val) == N,
              DimensionMismatch("length($val_sym) ($(length(val))) must match the number of assets ($N)"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Vector overload of [`mean(me::CustomValueExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref). Returns the stored vector `me.val` reshaped to match `dims`.
"""
function Statistics.mean(me::CustomValueExpectedReturns{<:VecNum}, X::MatNum; dims::Int = 1,
                         kwargs...)
    assert_dims(dims)
    _ncols = size(X, setdiff((1, 2), (dims,))[1])
    assert_custom_expected_returns_val(me.val, _ncols)
    return insertdims(me.val; dims = dims)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Function overload of [`mean(me::CustomValueExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref). Delegates to the callable `me.val` with the same arguments, and validates the value it returns against the number of assets.
"""
function Statistics.mean(me::CustomValueExpectedReturns{<:Union{<:Function,
                                                                <:CustomExpectedReturnsValueAlgorithm}},
                         X::MatNum; dims::Int = 1, kwargs...)
    assert_dims(dims)
    _ncols = size(X, setdiff((1, 2), (dims,))[1])
    val = me.val(X; dims = dims, kwargs...)
    assert_custom_expected_returns_val(val, _ncols, "val(X; dims = $dims, kwargs...)")
    return val
end

export CustomValueExpectedReturns
