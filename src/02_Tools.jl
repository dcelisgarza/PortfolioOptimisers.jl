"""
    abstract type AbstractReturnsResult <: AbstractResult end

Abstract supertype for all returns result types in PortfolioOptimisers.jl.

All concrete types representing the result of returns calculations (e.g., asset returns, factor returns)
should subtype `AbstractReturnsResult`. This enables a consistent interface for downstream analysis
and optimization routines.

# Related

  - [`AbstractResult`](@ref)
  - [`ReturnsResult`](@ref)
"""
abstract type AbstractReturnsResult <: AbstractResult end
"""
    assert_nonempty_nonneg_finite_val(val::Union{<:Number, <:Pair, <:AbstractDict,
                                        <:VecNum, <:VecPair};
                             val_sym::Symbol = :val)
    assert_nonempty_nonneg_finite_val(args...)

Validate that the input value is non-negative and finite.

Checks that the provided value (scalar, vector, dictionary, or pair) contains only finite and non-negative entries. Used for defensive programming and input validation throughout PortfolioOptimisers.jl.

# Arguments

  - `val`: Input value to validate.
  - `val_sym`: Symbolic name for the value (used in error messages).

# Returns

  - `nothing`: Returns nothing if validation passes.

# Validation

  - `args...`: always passes.
  - `Number`: `isfinite(val)` and `val >= 0`.
  - `Pair`: `isfinite(val[2])` and `val[2] >= 0`.
  - `AbstractDict`: `!isempty(val)`, `any(isfinite, values(val))`, `all(x -> x >= 0, values(val))`.
  - `VecNum`: `!isempty(val)`, `any(isfinite, val)`, `all(x -> x >= 0, val)`.
  - `VecPair`: `!isempty(val)`, `any(isfinite, getindex.(val, 2))`, `all(x -> x[2] >= 0, val)`.
"""
function assert_nonempty_nonneg_finite_val(val::AbstractDict, val_sym::Symbol = :val)
    @argcheck(!isempty(val),
              IsEmptyError("!isempty($val_sym) must hold. Got\n!isempty($val_sym) => $(isempty(val))"))
    @argcheck(any(isfinite, values(val)),
              DomainError("any(isfinite, values($val_sym)) must hold. Got\nany(isfinite, values($val_sym)) => $(any(isfinite, values(val)))"))
    @argcheck(all(x -> zero(x) <= x, values(val)),
              DomainError("all(x -> 0 <= x, values($val_sym)) must hold. Got\nall(x -> 0 <= x, values($val_sym)) => $(all(x -> zero(x) <= x, values(val)))"))
    return nothing
end
function assert_nonempty_nonneg_finite_val(val::VecPair, val_sym::Symbol = :val)
    @argcheck(!isempty(val),
              IsEmptyError("!isempty($val_sym) must hold. Got\n!isempty($val_sym) => $(isempty(val))"))
    @argcheck(any(isfinite, getindex.(val, 2)),
              DomainError("any(isfinite, getindex.($val_sym, 2)) must hold. Got\nany(isfinite, getindex.($val_sym, 2)) => $(any(isfinite, getindex.(val, 2)))"))
    @argcheck(all(x -> zero(x[2]) <= x[2], val),
              DomainError("all(x -> 0 <= x[2], $val_sym) must hold. Got\nall(x -> 0 <= x[2], $val_sym) => $(all(x -> zero(x[2]) <= x[2], val))"))
    return nothing
end
function assert_nonempty_nonneg_finite_val(val::ArrNum, val_sym::Symbol = :val)
    @argcheck(!isempty(val),
              IsEmptyError("!isempty($val_sym) must hold. Got\n!isempty($val_sym) => $(isempty(val))"))
    @argcheck(any(isfinite, val),
              DomainError("any(isfinite, $val_sym) must hold. Got\nany(isfinite, $val_sym) => $(any(isfinite, val))"))
    @argcheck(all(x -> zero(x) <= x, val),
              DomainError("all(x -> 0 <= x, $val_sym) must hold. Got\nall(x -> 0 <= x, $val_sym) => $(all(x -> zero(x) <= x, val))"))
    return nothing
end
function assert_nonempty_nonneg_finite_val(val::Pair, val_sym::Symbol = :val)
    @argcheck(isfinite(val[2]),
              DomainError("isfinite($val_sym[2]) must hold. Got\nisfinite($val_sym[2]) => $(isfinite(val[2]))"))
    @argcheck(zero(val[2]) <= val[2],
              DomainError("0 <= $(val[2]) must hold. Got\n$(val[2]) => $(val[2])"))
    return nothing
end
function assert_nonempty_nonneg_finite_val(val::Number, val_sym::Symbol = :val)
    @argcheck(isfinite(val),
              DomainError("isfinite($val_sym) must hold. Got\nisfinite($val_sym) => $(isfinite(val))"))
    @argcheck(zero(val) <= val, DomainError("0 <= $(val) must hold. Got\n$(val) => $(val)"))
    return nothing
end
function assert_nonempty_nonneg_finite_val(args...)
    return nothing
end
function assert_nonempty_finite_val(val::AbstractDict, val_sym::Symbol = :val)
    @argcheck(!isempty(val),
              IsEmptyError("!isempty($val_sym) must hold. Got\n!isempty($val_sym) => $(isempty(val))"))
    @argcheck(any(isfinite, values(val)),
              DomainError("any(isfinite, values($val_sym)) must hold. Got\nany(isfinite, values($val_sym)) => $(any(isfinite, values(val)))"))
    return nothing
end
function assert_nonempty_finite_val(val::VecPair, val_sym::Symbol = :val)
    @argcheck(!isempty(val),
              IsEmptyError("!isempty($val_sym) must hold. Got\n!isempty($val_sym) => $(isempty(val))"))
    @argcheck(any(isfinite, getindex.(val, 2)),
              DomainError("any(isfinite, getindex.($val_sym, 2)) must hold. Got\nany(isfinite, getindex.($val_sym, 2)) => $(any(isfinite, getindex.(val, 2)))"))
    return nothing
end
function assert_nonempty_finite_val(val::ArrNum, val_sym::Symbol = :val)
    @argcheck(!isempty(val),
              IsEmptyError("!isempty($val_sym) must hold. Got\n!isempty($val_sym) => $(isempty(val))"))
    @argcheck(any(isfinite, val),
              DomainError("any(isfinite, $val_sym) must hold. Got\nany(isfinite, $val_sym) => $(any(isfinite, val))"))
    return nothing
end
function assert_nonempty_finite_val(val::Pair, val_sym::Symbol = :val)
    @argcheck(isfinite(val[2]),
              DomainError("isfinite($val_sym[2]) must hold. Got\nisfinite($val_sym[2]) => $(isfinite(val[2]))"))
    return nothing
end
function assert_nonempty_finite_val(val::Number, val_sym::Symbol = :val)
    @argcheck(isfinite(val),
              DomainError("isfinite($val_sym) must hold. Got\nisfinite($val_sym) => $(isfinite(val))"))
    return nothing
end
function assert_nonempty_finite_val(args...)
    return nothing
end
function assert_nonempty_geq0_finite_val(val::AbstractDict, val_sym::Symbol = :val)
    @argcheck(!isempty(val),
              IsEmptyError("!isempty($val_sym) must hold. Got\n!isempty($val_sym) => $(isempty(val))"))
    @argcheck(any(isfinite, values(val)),
              DomainError("any(isfinite, values($val_sym)) must hold. Got\nany(isfinite, values($val_sym)) => $(any(isfinite, values(val)))"))
    @argcheck(all(x -> zero(x) < x, values(val)),
              DomainError("all(x -> 0 < x, values($val_sym)) must hold. Got\nall(x -> 0 < x, values($val_sym)) => $(all(x -> zero(x) < x, values(val)))"))
    return nothing
end
function assert_nonempty_geq0_finite_val(val::VecPair, val_sym::Symbol = :val)
    @argcheck(!isempty(val),
              IsEmptyError("!isempty($val_sym) must hold. Got\n!isempty($val_sym) => $(isempty(val))"))
    @argcheck(any(isfinite, getindex.(val, 2)),
              DomainError("any(isfinite, getindex.($val_sym, 2)) must hold. Got\nany(isfinite, getindex.($val_sym, 2)) => $(any(isfinite, getindex.(val, 2)))"))
    @argcheck(all(x -> zero(x[2]) < x[2], val),
              DomainError("all(x -> 0 < x[2], $val_sym) must hold. Got\nall(x -> 0 < x[2], $val_sym) => $(all(x -> zero(x[2]) < x[2], val))"))
    return nothing
end
function assert_nonempty_geq0_finite_val(val::ArrNum, val_sym::Symbol = :val)
    @argcheck(!isempty(val),
              IsEmptyError("!isempty($val_sym) must hold. Got\n!isempty($val_sym) => $(isempty(val))"))
    @argcheck(any(isfinite, val),
              DomainError("any(isfinite, $val_sym) must hold. Got\nany(isfinite, $val_sym) => $(any(isfinite, val))"))
    @argcheck(all(x -> zero(x) < x, val),
              DomainError("all(x -> 0 < x, $val_sym) must hold. Got\nall(x -> 0 < x, $val_sym) => $(all(x -> zero(x) < x, val))"))
    return nothing
end
function assert_nonempty_geq0_finite_val(val::Pair, val_sym::Symbol = :val)
    @argcheck(isfinite(val[2]),
              DomainError("isfinite($val_sym[2]) must hold. Got\nisfinite($val_sym[2]) => $(isfinite(val[2]))"))
    @argcheck(zero(val[2]) < val[2],
              DomainError("0 < $(val[2]) must hold. Got\n$(val[2]) => $(val[2])"))
    return nothing
end
function assert_nonempty_geq0_finite_val(val::Number, val_sym::Symbol = :val)
    @argcheck(isfinite(val),
              DomainError("isfinite($val_sym) must hold. Got\nisfinite($val_sym) => $(isfinite(val))"))
    @argcheck(zero(val) < val, DomainError("0 < $(val) must hold. Got\n$(val) => $(val)"))
    return nothing
end
function assert_nonempty_geq0_finite_val(args...)
    return nothing
end
"""
    assert_matrix_issquare(A::MatNum, A_sym::Symbol = :A)

Assert that `size(A, 1) == size(A, 2)`.
"""
function assert_matrix_issquare(A::MatNum, A_sym::Symbol = :A)
    @argcheck(size(A, 1) == size(A, 2),
              DimensionMismatch("size($A_sym, 1) == size($A_sym, 2) must hold. Got\nsize($A_sym, 1) => $(size(A, 1))\nsize($A_sym, 2) => $(size(A, 2))."))
end
"""
    brinson_attribution(X::TimeArray, w::VecNum, wb::VecNum,
                        asset_classes::DataFrame, col; date0 = nothing, date1 = nothing)
"""
function brinson_attribution(X::TimeArray, w::VecNum, wb::VecNum, asset_classes::DataFrame,
                             col, date0 = nothing, date1 = nothing)
    # Efficient filtering of date range
    idx1, idx2 = if !isnothing(date0) && !isnothing(date1)
        timestamps = timestamp(X)
        idx = (DateTime(date0) .<= timestamps) .& (timestamps .<= DateTime(date1))
        findfirst(idx), findlast(idx)
    else
        1, length(X)
    end

    ret = vec(values(X[idx2]) ./ values(X[idx1]) .- 1)
    ret_b = dot(ret, wb)

    classes = asset_classes[!, col]
    unique_classes = unique(classes)

    df = DataFrame(;
                   index = ["Asset Allocation", "Security Selection", "Interaction",
                            "Total Excess Return"])

    # Precompute class membership matrix for efficiency
    sets_mat = [class_j == class_i for class_j in classes, class_i in unique_classes]

    for (i, class_i) in enumerate(unique_classes)
        sets_i = view(sets_mat, :, i)

        w_i = dot(sets_i, w)
        wb_i = dot(sets_i, wb)

        ret_i = dot(ret .* sets_i, w) / w_i
        ret_b_i = dot(ret .* sets_i, wb) / wb_i

        w_diff_i = w_i - wb_i
        ret_diff_i = ret_i - ret_b_i

        AA_i = w_diff_i * (ret_b_i - ret_b)
        SS_i = wb_i * ret_diff_i
        I_i = w_diff_i * ret_diff_i
        TER_i = AA_i + SS_i + I_i

        df[!, class_i] = [AA_i, SS_i, I_i, TER_i]
    end

    df[!, "Total"] = sum(eachcol(df[!, 2:end]))

    return df
end
"""
    ⊗(A::ArrNum, B::ArrNum)

Tensor product of two arrays. Returns a matrix of size `(length(A), length(B))` where each element is the product of elements from `A` and `B`.

# Examples

```jldoctest
julia> PortfolioOptimisers.:⊗([1, 2], [3, 4])
2×2 Matrix{Int64}:
 3  4
 6  8
```
"""
⊗(A::ArrNum, B::ArrNum) = reshape(kron(B, A), (length(A), length(B)))
"""
    ⊙(A, B)

Elementwise multiplication.

# Examples

```jldoctest
julia> PortfolioOptimisers.:⊙([1, 2], [3, 4])
2-element Vector{Int64}:
 3
 8

julia> PortfolioOptimisers.:⊙([1, 2], 2)
2-element Vector{Int64}:
 2
 4

julia> PortfolioOptimisers.:⊙(2, [3, 4])
2-element Vector{Int64}:
 6
 8

julia> PortfolioOptimisers.:⊙(2, 3)
6
```
"""
⊙(A::ArrNum, B::ArrNum) = A .* B
⊙(A::ArrNum, B) = A * B
⊙(A, B::ArrNum) = A * B
⊙(A, B) = A * B
"""
    ⊘(A, B)

Elementwise division.

# Examples

```jldoctest
julia> PortfolioOptimisers.:⊘([4, 9], [2, 3])
2-element Vector{Float64}:
 2.0
 3.0

julia> PortfolioOptimisers.:⊘([4, 6], 2)
2-element Vector{Float64}:
 2.0
 3.0

julia> PortfolioOptimisers.:⊘(8, [2, 4])
2-element Vector{Float64}:
 4.0
 2.0

julia> PortfolioOptimisers.:⊘(8, 2)
4.0
```
"""
⊘(A::ArrNum, B::ArrNum) = A ./ B
⊘(A::ArrNum, B) = A / B
⊘(A, B::ArrNum) = A ./ B
⊘(A, B) = A / B
"""
    ⊕(A, B)

Elementwise addition.

# Examples

```jldoctest
julia> PortfolioOptimisers.:⊕([1, 2], [3, 4])
2-element Vector{Int64}:
 4
 6

julia> PortfolioOptimisers.:⊕([1, 2], 2)
2-element Vector{Int64}:
 3
 4

julia> PortfolioOptimisers.:⊕(2, [3, 4])
2-element Vector{Int64}:
 5
 6

julia> PortfolioOptimisers.:⊕(2, 3)
5
```
"""
⊕(A::ArrNum, B::ArrNum) = A .+ B
⊕(A::ArrNum, B) = A .+ B
⊕(A, B::ArrNum) = A .+ B
⊕(A, B) = A + B
"""
    ⊖(A, B)

Elementwise subtraction.

# Examples

```jldoctest
julia> PortfolioOptimisers.:⊖([4, 6], [1, 2])
2-element Vector{Int64}:
 3
 4

julia> PortfolioOptimisers.:⊖([4, 6], 2)
2-element Vector{Int64}:
 2
 4

julia> PortfolioOptimisers.:⊖(8, [2, 4])
2-element Vector{Int64}:
 6
 4

julia> PortfolioOptimisers.:⊖(8, 2)
6
```
"""
⊖(A::ArrNum, B::ArrNum) = A - B
⊖(A::ArrNum, B) = A .- B
⊖(A, B::ArrNum) = A .- B
⊖(A, B) = A - B
"""
    dot_scalar(a::Number, b::VecNum)
    dot_scalar(a::VecNum, b::Number)
    dot_scalar(a::VecNum, b::VecNum)

Efficient scalar and vector dot product utility.

  - If one argument is a `Number` and the other an `VecNum`, returns the scalar times the sum of the vector.
  - If both arguments are `VecNum`s, returns their `dot` product.

# Returns

  - `Number`: The resulting scalar.

# Examples

```jldoctest
julia> PortfolioOptimisers.dot_scalar(2.0, [1.0, 2.0, 3.0])
12.0

julia> PortfolioOptimisers.dot_scalar([1.0, 2.0, 3.0], 2.0)
12.0

julia> PortfolioOptimisers.dot_scalar([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
32.0
```
"""
function dot_scalar(a::Union{<:Number, <:AbstractJuMPScalar}, b::VecNum)
    return a * sum(b)
end
function dot_scalar(a::VecNum, b::Union{<:Number, <:AbstractJuMPScalar})
    return sum(a) * b
end
function dot_scalar(a::VecNum, b::VecNum)
    return dot(a, b)
end
"""
    struct VecScalar{T1, T2} <: AbstractResult
        v::T1
        s::T2
    end

Represents a composite result containing a vector and a scalar in PortfolioOptimisers.jl.

Encapsulates a vector and a scalar value, commonly used for storing results that combine both types of data (e.g., weighted statistics, risk measures).

# Fields

  - `v`: Vector value.
  - `s`: Scalar value.

# Constructors

    VecScalar(; v::VecNum, s::Number)

Arguments correspond to the fields above.

## Validation

  - `v`: `!isempty(v)` and `all(isfinite, v)`.
  - `s`: `isfinite(s)`.

# Examples

```jldoctest
julia> VecScalar([1.0, 2.0, 3.0], 4.2)
VecScalar
  v ┼ Vector{Float64}: [1.0, 2.0, 3.0]
  s ┴ Float64: 4.2
```

# Related

  - [`AbstractResult`](@ref)
"""
struct VecScalar{T1, T2} <: AbstractResult
    v::T1
    s::T2
    function VecScalar(v::VecNum, s::Number)
        assert_nonempty_finite_val(v, :v)
        assert_nonempty_finite_val(s, :s)
        return new{typeof(v), typeof(s)}(v, s)
    end
end
function VecScalar(; v::VecNum, s::Number)
    return VecScalar(v, s)
end
const Num_VecNum_VecScalar = Union{<:Num_VecNum, <:VecScalar}
const Num_ArrNum_VecScalar = Union{<:Num_ArrNum, <:VecScalar}
"""
    nothing_scalar_array_view(x, i)

Utility for safely viewing or indexing into possibly `nothing`, scalar, or array values.

  - `x`: Input value.

      + `nothing`: returns `nothing`.
      + `Number`: returns `x`.
      + `VecNum`: returns `view(x, i)`.
      + `VecVecNum`: returns `[view(xi, i) for xi in x]`.
      + `ArrNum`: returns `view(x, i, i)`.

# Arguments

  - `x`: Input value, which may be `nothing`, a scalar, vector, or array.
  - `i`: Index or indices to view.

# Returns

  - The corresponding view or value, or `nothing` if `x` is `nothing`.

# Examples

```jldoctest
julia> PortfolioOptimisers.nothing_scalar_array_view(nothing, 1:2)

julia> PortfolioOptimisers.nothing_scalar_array_view(3.0, 1:2)
3.0

julia> PortfolioOptimisers.nothing_scalar_array_view([1.0, 2.0, 3.0], 2:3)
2-element view(::Vector{Float64}, 2:3) with eltype Float64:
 2.0
 3.0

julia> PortfolioOptimisers.nothing_scalar_array_view([[1, 2], [3, 4]], 1)
2-element Vector{SubArray{Int64, 0, Vector{Int64}, Tuple{Int64}, true}}:
 fill(1)
 fill(3)
```
"""
function nothing_scalar_array_view(x::Union{Nothing, <:Number, <:Pair, <:VecPair, <:Dict},
                                   ::Any)
    return x
end
function nothing_scalar_array_view(x::AbstractVector, i)
    return view(x, i)
end
function nothing_scalar_array_view(x::VecScalar, i)
    return VecScalar(; v = view(x.v, i), s = x.s)
end
function nothing_scalar_array_view(x::AbstractVector{<:Union{<:AbstractVector, <:VecScalar}},
                                   i)
    return [view(xi, i) for xi in x]
end
function nothing_scalar_array_view(x::ArrNum, i)
    return view(x, i, i)
end
"""
    nothing_scalar_array_view_odd_order(x, i, j)

Utility for safely viewing or indexing into possibly `nothing` or array values with two indices.

  - If `x` is `nothing`, returns `nothing`.
  - Otherwise, returns `view(x, i, j)`.

# Arguments

  - `x`: Input value, which may be `nothing` or an array.
  - `i`, `j`: Indices to view.

# Returns

  - The corresponding view or `nothing`.

# Examples

```jldoctest
julia> PortfolioOptimisers.nothing_scalar_array_view_odd_order(nothing, 1, 2)

julia> PortfolioOptimisers.nothing_scalar_array_view_odd_order([1 2; 3 4], 1, 2)
0-dimensional view(::Matrix{Int64}, 1, 2) with eltype Int64:
2
```
"""
function nothing_scalar_array_view_odd_order(::Nothing, i, j)
    return nothing
end
function nothing_scalar_array_view_odd_order(x::ArrNum, i, j)
    return view(x, i, j)
end
"""
    nothing_scalar_array_getindex(x, i)
    nothing_scalar_array_getindex(x, i, j)

Utility for safely indexing into possibly `nothing`, scalar, vector, or array values.

  - `x`: Input value.

      + `nothing`: returns `nothing`.
      + `Number`: returns `x`.
      + `VecNum`: returns `x[i]`.
      + `MatNum`: returns `x[i, i]` or `x[i, j]`.

# Arguments

  - `x`: Input value, which may be `nothing`, a scalar, vector, or matrix.
  - `i`, `j`: Indices.

# Returns

  - The corresponding value or `nothing`.

# Examples

```jldoctest
julia> PortfolioOptimisers.nothing_scalar_array_getindex(nothing, 1)

julia> PortfolioOptimisers.nothing_scalar_array_getindex(3.0, 1)
3.0

julia> PortfolioOptimisers.nothing_scalar_array_getindex([1.0, 2.0, 3.0], 2)
2.0

julia> PortfolioOptimisers.nothing_scalar_array_getindex([1 2; 3 4], 2)
4

julia> PortfolioOptimisers.nothing_scalar_array_getindex([1 2; 3 4], 1, 2)
2
```
"""
function nothing_scalar_array_getindex(x::Number, ::Any)
    return x
end
function nothing_scalar_array_getindex(::Nothing, ::Any)
    return nothing
end
function nothing_scalar_array_getindex(x::VecNum, i)
    return x[i]
end
function nothing_scalar_array_getindex(x::MatNum, i)
    return x[i, i]
end
function nothing_scalar_array_getindex(::Nothing, i, j)
    return nothing
end
function nothing_scalar_array_getindex(x::MatNum, i, j)
    return x[i, j]
end
function nothing_scalar_array_getindex(x::VecScalar, i)
    return VecScalar(x.v[i], x.s)
end
"""
    fourth_moment_index_factory(N::Integer, i)

Constructs an index vector for extracting the fourth moment submatrix corresponding to indices `i` from a covariance matrix of size `N × N`.

# Arguments

  - `N`: Size of the full covariance matrix.
  - `i`: Indices of the variables of interest.

# Returns

  - `idx::Vector{Int}`: Indices for extracting the fourth moment submatrix.

# Examples

```jldoctest
julia> PortfolioOptimisers.fourth_moment_index_factory(3, [1, 2])
4-element Vector{Int64}:
 1
 2
 4
 5
```
"""
function fourth_moment_index_factory(N::Integer, i)
    idx = sizehint!(Int[], length(i)^2)
    for c in i
        append!(idx, (((c - 1) * N + 1):(c * N))[i])
    end
    return idx
end
"""
    traverse_concrete_subtypes(t; ctarr::Option{<:VecNum} = nothing)

Recursively traverse all subtypes of the given abstract type `t` and collect all concrete struct types into `ctarr`.

# Arguments

  - `t`: An abstract type whose subtypes will be traversed.
  - `ctarr`: Optional An array to collect the concrete types. If not provided, a new empty array is created.

# Returns

An array containing all concrete struct types that are subtypes (direct or indirect) of `types`.

# Examples

```julia
julia> abstract type MyAbstract end

julia> struct MyConcrete1 <: MyAbstract end

julia> struct MyConcrete2 <: MyAbstract end

julia> traverse_concrete_subtypes(MyAbstract)
2-element Vector{Any}:
 MyConcrete1
 MyConcrete2
```
"""
function traverse_concrete_subtypes(t, ctarr::Option{<:VecNum} = nothing)
    if isnothing(ctarr)
        ctarr = []
    end
    sts = subtypes(t)
    for st in sts
        if !isstructtype(st)
            traverse_concrete_subtypes(st, ctarr)
        else
            push!(ctarr, st)
        end
    end
    return ctarr
end
"""
    concrete_typed_array(A::AbstractArray)

Convert an `AbstractArray` `A` to a concrete typed array, where each element is of the same type as the elements of `A`.

This is useful for converting arrays with abstract element types to arrays with concrete element types, which can improve performance in some cases.

# Arguments

  - `A`: The input array.

# Returns

A new array with the same shape as `A`, but with a concrete element type inferred from the elements of `A`.

# Examples

```jldoctest
julia> A = Any[1, 2.0, 3];

julia> PortfolioOptimisers.concrete_typed_array(A)
3-element Vector{Union{Float64, Int64}}:
 1
 2.0
 3
```
"""
function concrete_typed_array(A::AbstractArray)
    return reshape(Union{typeof.(A)...}[A...], size(A))
end
function factory(::Nothing, args...; kwargs...)
    return nothing
end

export VecScalar, brinson_attribution, factory, traverse_concrete_subtypes

export Num_VecNum_VecScalar, Num_ArrNum_VecScalar
