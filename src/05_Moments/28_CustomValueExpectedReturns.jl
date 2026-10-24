"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the custom expected returns value algorithms. A subtype is one of the types that the `val` field of [`CustomValueExpectedReturns`](@ref) accepts.

# Interfaces

To add a custom expected returns value algorithm, subtype `CustomExpectedReturnsValueAlgorithm` and make the subtype callable. The call must return a vector with one element per asset of the returns matrix:

  - `(alg::MyCustomValAlg)(X::MatNum; dims::Int=1, kwargs...) -> Vector{<:Number}`

Where:

  - `alg`: The custom expected returns value algorithm.
  - `X`: The data matrix of asset returns.
  - `dims`: `1` when the observations run down the rows of `X`, and `2` when they run along the columns.
  - `kwargs...`: The keyword arguments of the `mean` call, which the estimator passes on.

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

Alias for the types that the `val` field of [`CustomValueExpectedReturns`](@ref) accepts.

The three shapes of `val` are a scalar, a per-asset vector and a callable. The keyword constructor and the checked positional constructor take this bound. The three `mean` methods dispatch on the shape, each on its own bound.

# Related

  - [`CustomExpectedReturnsValueAlgorithm`](@ref): A callable the caller subtypes, which the estimator calls as `val(X; dims = dims, kwargs...)`.
  - [`Func_Num_VecNum`](@ref): The scalar, the per-asset vector and the plain `Function`, which this alias widens with the algorithm supertype.
  - [`CustomValueExpectedReturns`](@ref): The estimator whose `val` field carries this bound.
"""
const CER_Func_Num_VecNum = Union{<:CustomExpectedReturnsValueAlgorithm, <:Func_Num_VecNum}
"""
$(DocStringExtensions.TYPEDEF)

Returns a caller-supplied value for each asset instead of estimating one from the data.

`val` holds a scalar, a per-asset vector, or a callable that the estimator calls with the data matrix. The type of `val` selects the branch that [`mean(me::CustomValueExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref) takes. The keyword constructor takes a `val` of the [`CER_Func_Num_VecNum`](@ref) bound alone, and raises a `TypeError` on any other type. The field itself has no bound, so a positional call with another type builds the estimator without a check.

The constructor checks only that a vector is not empty, because it does not know the number of assets before the data arrive. [`assert_custom_expected_returns_val`](@ref) checks the length of a stored vector, and the shape and the length of what a callable returns, when `mean` uses them.

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

The method returns the scalar, the vector, or the result of the callable that the caller gave as `val`.

```math
\\begin{align}
\\hat{\\mu}_j &= v_j, \\quad j = 1, \\ldots, N\\,.
\\end{align}
```

Where:

  - ``\\hat{\\mu}_j``: The expected return of asset ``j``.
  - ``v_j``: The ``j``-th element of the custom value `me.val`. A scalar gives the same ``v_j`` to every asset. A vector gives its ``j``-th element. A callable gives the ``j``-th element of its result.
  - $(math_dict[:N])

# Algorithm

 1. Check `dims`.

 2. Read the asset count `_ncols` from `X` and `dims`, as the size of the dimension that `dims` does not name.

 3. Take the branch that the type of `me.val` selects.

      + `me.val::Number`: Fill a vector of length `_ncols` with `me.val`.
      + `me.val::VecNum`: Check the stored `me.val` against `_ncols` with [`assert_custom_expected_returns_val`](@ref), and take it unchanged.
      + `me.val::Function` or `me.val::CustomExpectedReturnsValueAlgorithm`: Call `me.val(X; dims = dims, kwargs...)`. The result is `val`. Check `val` against `_ncols` with [`assert_custom_expected_returns_val`](@ref).

 4. On the first two branches, insert the reduced dimension with `insertdims`. The callable branch returns `val` unchanged, and inserts no dimension.

# Arguments

  - `me`: The custom value expected returns estimator.
  - `X`: The data matrix of asset returns, `observations × assets`.
  - $(arg_dict[:dims])
  - `kwargs...`: More keyword arguments. The callable branch passes them to `me.val`. The other two branches ignore them.

# Validation

  - $(val_dict[:dims])
  - The vector branch and the callable branch both check the value against the number of assets with [`assert_custom_expected_returns_val`](@ref).

# Returns

  - `mu`: The expected returns, one value per asset. The shape depends on the branch.

      + `me.val::Number` and `me.val::VecNum`: A `Matrix{<:Number}`. It is a `(1, N)` matrix if `dims == 1` or an `(N, 1)` matrix if `dims == 2`, as most expected returns estimators return.
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

The vector field of [`CustomValueExpectedReturns`](@ref) and the result of a callable `val` must both be a vector of numbers with one element per asset. `mean` checks the result of a callable after it calls it, because no earlier step can see that result.

# Arguments

  - `val`: The custom value to check. It is the stored vector `me.val`, or the result of a callable `me.val`.
  - `N`: The number of assets, which the data matrix and `dims` give.
  - `val_sym`: The name of `val` in the error messages.

# Validation

  - `isa(val, VecNum)`, or the function raises an `ArgumentError`. A scalar, a function and a matrix all fail here. The message names `val_sym` and the type of `val`.
  - `length(val) == N`, or the function raises a `DimensionMismatch`. The message names `val_sym`, the length of `val`, and `N`.

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

The vector method of [`mean(me::CustomValueExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref). It returns the stored vector `me.val` as a `(1, N)` matrix if `dims == 1` or an `(N, 1)` matrix if `dims == 2`.
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

The callable method of [`mean(me::CustomValueExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref). It calls `me.val` with the same arguments, and checks the result against the number of assets.
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
"""
    port_opt_view(
        me::CustomValueExpectedReturns{<:VecNum},
        i,
        args...
    ) -> CustomValueExpectedReturns

The [`port_opt_view`](@ref) method for the per-asset vector shape of
[`CustomValueExpectedReturns`](@ref). It slices `val` to the selected assets.

`val` is the one field of the estimator, and only the vector shape has one entry per asset. A
scalar does not depend on the universe. A callable gets the reduced matrix and returns one value
per column of it. Both therefore keep the identity method of
[`AbstractExpectedReturnsEstimator`](@ref). A stored vector has one entry per asset of the universe
that the caller wrote it for. If a reduction left it whole, the vector would keep the full universe
while the sample has fewer assets, and
[`mean(me::CustomValueExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref) would raise a
`DimensionMismatch`.

The bound of `val` is a union of the three shapes. An [`@vprop`](@ref) tag on the field would
slice all three, so this file declares a method on the vector shape alone.

# Algorithm

 1. Return a new estimator whose `val` is `view(me.val, i)`.

# Arguments

  - `me::CustomValueExpectedReturns{<:VecNum}`: The estimator whose `val` is one entry per asset.
  - `i`: The index or the mask of the assets to select.
  - `args...`: More positional arguments of the `port_opt_view` call, which the method ignores.

# Returns

  - `me::CustomValueExpectedReturns`: A new estimator whose `val` is restricted to the selected
    assets.

# Related

  - [`CustomValueExpectedReturns`](@ref)
  - [`port_opt_view`](@ref)
  - [`investable_reduction`](@ref)
"""
function port_opt_view(me::CustomValueExpectedReturns{<:VecNum}, i,
                       args...)::CustomValueExpectedReturns
    return CustomValueExpectedReturns(; val = view(me.val, i))
end

export CustomValueExpectedReturns
public CustomExpectedReturnsValueAlgorithm
