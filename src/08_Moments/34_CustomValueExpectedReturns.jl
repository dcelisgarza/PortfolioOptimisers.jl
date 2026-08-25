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

The group exists because the three shapes are a single field's contract, not three fields. The bound is written once here, so the field, the keyword constructor and the three `mean` methods cannot drift apart on what `val` may be.

# Related

  - [`CustomExpectedReturnsValueAlgorithm`](@ref): A callable the caller subtypes, which the estimator calls as `val(X; dims = dims, kwargs...)`.
  - [`Func_Num_VecNum`](@ref): The scalar, the per-asset vector and the plain `Function`, which this alias widens with the algorithm supertype.
  - [`CustomValueExpectedReturns`](@ref): The estimator whose `val` field carries this bound.
"""
const CER_Func_Num_VecNum = Union{<:CustomExpectedReturnsValueAlgorithm, <:Func_Num_VecNum}
"""
$(DocStringExtensions.TYPEDEF)

Returns a caller-supplied value for each asset instead of estimating one from the data.

`val` holds a scalar, a per-asset vector, or a callable that the estimator calls with the data matrix. The type of `val` selects the branch that [`mean(me::CustomValueExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref) takes, and any other type is rejected by the `CER_Func_Num_VecNum` bound of the field.

The constructor checks only that a vector is not empty, because the number of assets is not known until the data arrive. The length of a stored vector, and the shape and the length of what a callable returns, are checked at the point of use by [`assert_custom_expected_returns_val`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CustomValueExpectedReturns(;
        val::CER_Func_Num_VecNum = 0.0
    ) -> CustomValueExpectedReturns

Keywords correspond to the struct's fields.

## Validation

  - If `val` is a vector, `!isempty(val)`.

# Examples

```jldoctest
julia> CustomValueExpectedReturns()
CustomValueExpectedReturns
  val ┴ Float64: 0.0
```

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`CER_Func_Num_VecNum`](@ref)
  - [`CustomExpectedReturnsValueAlgorithm`](@ref)
  - [`assert_custom_expected_returns_val`](@ref)
"""
@concrete struct CustomValueExpectedReturns <: AbstractExpectedReturnsEstimator
    """
    $(field_dict[:me_cval])
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

# Algorithm

 1. Check `dims`.

 2. Read the asset count `_ncols` from `X` and `dims`, as the size of the dimension that `dims` does not name.

 3. Take the branch that the type of `me.val` selects.

      + `me.val::Number`: Fill a vector of length `_ncols` with `me.val`.
      + `me.val::VecNum`: Check the stored `me.val` against `_ncols` with [`assert_custom_expected_returns_val`](@ref), and take it unchanged.
      + `me.val::Function` or `me.val::CustomExpectedReturnsValueAlgorithm`: Call `me.val(X; dims = dims, kwargs...)`, giving `val`, and check `val` against `_ncols` with [`assert_custom_expected_returns_val`](@ref).

 4. On the first two branches, insert the reduced dimension with `insertdims`. The callable branch returns `val` unchanged, and inserts no dimension.

# Arguments

  - `me`: Custom value expected returns estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments. The callable branch passes them to `me.val`; the other two branches ignore them.

# Validation

  - $(val_dict[:dims])
  - The vector branch and the callable branch both check the value against the number of assets with [`assert_custom_expected_returns_val`](@ref).

# Returns

  - `mu`: Expected returns, one value per asset. The shape depends on the branch.

      + `me.val::Number` and `me.val::VecNum`: A `Matrix{<:Number}`, shaped as `(1, N)` if `dims == 1` or `(N, 1)` if `dims == 2`, as the other expected returns estimators return.
      + `me.val::Function` and `me.val::CustomExpectedReturnsValueAlgorithm`: The vector the callable returned, of length `N`, passed through unchanged. This branch inserts no dimension.

# Related

  - [`CustomValueExpectedReturns`](@ref)
  - [`assert_custom_expected_returns_val`](@ref)
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

# Validation

  - `isa(val, VecNum)`, or an `ArgumentError` is thrown. A scalar, a function and a matrix all fail here, and the message names `val_sym` and the type it was given.
  - `length(val) == N`, or a `DimensionMismatch` is thrown. The message names `val_sym`, the length it was given, and `N`.

# Returns

  - `nothing`.

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
