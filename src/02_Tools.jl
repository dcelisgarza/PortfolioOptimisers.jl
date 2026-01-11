"""
    assert_nonempty_nonneg_finite_val(val::AbstractDict, val_sym::Symbol = :val)
    assert_nonempty_nonneg_finite_val(val::VecPair, val_sym::Symbol = :val)
    assert_nonempty_nonneg_finite_val(val::ArrNum, val_sym::Symbol = :val)
    assert_nonempty_nonneg_finite_val(val::Pair, val_sym::Symbol = :val)
    assert_nonempty_nonneg_finite_val(val::Number, val_sym::Symbol = :val)
    assert_nonempty_nonneg_finite_val(args...)

Validate that the input value is non-empty, non-negative and finite.

# Arguments

  - `val`: Input value to validate.
  - `val_sym`: Symbolic name used in the error messages.

# Returns

  - `nothing`.

# Details

  - `val`: Input value to validate.

      + `::AbstractDict`: `!isempty(val)`, `any(isfinite, values(val))`, `all(x -> x >= 0, values(val))`.
      + `::VecPair`: `!isempty(val)`, `any(isfinite, getindex.(val, 2))`, `all(x -> x[2] >= 0, val)`.
      + `::ArrNum`: `!isempty(val)`, `any(isfinite, val)`, `all(x -> x >= 0, val)`.
      + `::Pair`: `isfinite(val[2])` and `val[2] >= 0`.
      + `::Number`: `isfinite(val)` and `val >= 0`.
      + `args...`: Always passes.

# Related

  - [`assert_nonempty_finite_val`](@ref)
  - [`assert_nonempty_gt0_finite_val`](@ref)
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
"""
    assert_nonempty_gt0_finite_val(val::AbstractDict, val_sym::Symbol = :val)
    assert_nonempty_gt0_finite_val(val::VecPair, val_sym::Symbol = :val)
    assert_nonempty_gt0_finite_val(val::ArrNum, val_sym::Symbol = :val)
    assert_nonempty_gt0_finite_val(val::Pair, val_sym::Symbol = :val)
    assert_nonempty_gt0_finite_val(val::Number, val_sym::Symbol = :val)
    assert_nonempty_gt0_finite_val(args...)

Validate that the input value is non-empty, greater than zero, and finite.

# Arguments

  - `val`: Input value to validate.
  - `val_sym`: Symbolic name used in the error messages.

# Returns

  - `nothing`.

# Details

  - `val`: Input value to validate.

      + `::AbstractDict`: `!isempty(val)`, `any(isfinite, values(val))`, `all(x -> x > 0, values(val))`.
      + `::VecPair`: `!isempty(val)`, `any(isfinite, getindex.(val, 2))`, `all(x -> x[2] > 0, val)`.
      + `::ArrNum`: `!isempty(val)`, `any(isfinite, val)`, `all(x -> x > 0, val)`.
      + `::Pair`: `isfinite(val[2])` and `val[2] > 0`.
      + `::Number`: `isfinite(val)` and `val > 0`.
      + `args...`: Always passes.

# Related

  - [`assert_nonempty_nonneg_finite_val`](@ref)
  - [`assert_nonempty_finite_val`](@ref)
"""
function assert_nonempty_gt0_finite_val(val::AbstractDict, val_sym::Symbol = :val)
    @argcheck(!isempty(val),
              IsEmptyError("!isempty($val_sym) must hold. Got\n!isempty($val_sym) => $(isempty(val))"))
    @argcheck(any(isfinite, values(val)),
              DomainError("any(isfinite, values($val_sym)) must hold. Got\nany(isfinite, values($val_sym)) => $(any(isfinite, values(val)))"))
    @argcheck(all(x -> zero(x) < x, values(val)),
              DomainError("all(x -> 0 < x, values($val_sym)) must hold. Got\nall(x -> 0 < x, values($val_sym)) => $(all(x -> zero(x) < x, values(val)))"))
    return nothing
end
function assert_nonempty_gt0_finite_val(val::VecPair, val_sym::Symbol = :val)
    @argcheck(!isempty(val),
              IsEmptyError("!isempty($val_sym) must hold. Got\n!isempty($val_sym) => $(isempty(val))"))
    @argcheck(any(isfinite, getindex.(val, 2)),
              DomainError("any(isfinite, getindex.($val_sym, 2)) must hold. Got\nany(isfinite, getindex.($val_sym, 2)) => $(any(isfinite, getindex.(val, 2)))"))
    @argcheck(all(x -> zero(x[2]) < x[2], val),
              DomainError("all(x -> 0 < x[2], $val_sym) must hold. Got\nall(x -> 0 < x[2], $val_sym) => $(all(x -> zero(x[2]) < x[2], val))"))
    return nothing
end
function assert_nonempty_gt0_finite_val(val::ArrNum, val_sym::Symbol = :val)
    @argcheck(!isempty(val),
              IsEmptyError("!isempty($val_sym) must hold. Got\n!isempty($val_sym) => $(isempty(val))"))
    @argcheck(any(isfinite, val),
              DomainError("any(isfinite, $val_sym) must hold. Got\nany(isfinite, $val_sym) => $(any(isfinite, val))"))
    @argcheck(all(x -> zero(x) < x, val),
              DomainError("all(x -> 0 < x, $val_sym) must hold. Got\nall(x -> 0 < x, $val_sym) => $(all(x -> zero(x) < x, val))"))
    return nothing
end
function assert_nonempty_gt0_finite_val(val::Pair, val_sym::Symbol = :val)
    @argcheck(isfinite(val[2]),
              DomainError("isfinite($val_sym[2]) must hold. Got\nisfinite($val_sym[2]) => $(isfinite(val[2]))"))
    @argcheck(zero(val[2]) < val[2],
              DomainError("0 < $(val[2]) must hold. Got\n$(val[2]) => $(val[2])"))
    return nothing
end
function assert_nonempty_gt0_finite_val(val::Number, val_sym::Symbol = :val)
    @argcheck(isfinite(val),
              DomainError("isfinite($val_sym) must hold. Got\nisfinite($val_sym) => $(isfinite(val))"))
    @argcheck(zero(val) < val, DomainError("0 < $(val) must hold. Got\n$(val) => $(val)"))
    return nothing
end
function assert_nonempty_gt0_finite_val(args...)
    return nothing
end
"""
    assert_nonempty_finite_val(val::AbstractDict, val_sym::Symbol = :val)
    assert_nonempty_finite_val(val::VecPair, val_sym::Symbol = :val)
    assert_nonempty_finite_val(val::ArrNum, val_sym::Symbol = :val)
    assert_nonempty_finite_val(val::Pair, val_sym::Symbol = :val)
    assert_nonempty_finite_val(val::Number, val_sym::Symbol = :val)
    assert_nonempty_finite_val(args...)

Validate that the input value is non-empty and finite.

# Arguments

  - `val`: Input value to validate.
  - `val_sym`: Symbolic name used in the error messages.

# Returns

  - `nothing`.

# Details

  - `val`: Input value to validate.

      + `::AbstractDict`: `!isempty(val)`, `any(isfinite, values(val))`.
      + `::VecPair`: `!isempty(val)`, `any(isfinite, getindex.(val, 2))`.
      + `::ArrNum`: `!isempty(val)`, `any(isfinite, val)`.
      + `::Pair`: `isfinite(val[2])`.
      + `::Number`: `isfinite(val).
      + `args...`: Always passes.

# Related

  - [`assert_nonempty_nonneg_finite_val`](@ref)
  - [`assert_nonempty_gt0_finite_val`](@ref)
"""
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
"""
    assert_matrix_issquare(A::MatNum, A_sym::Symbol = :A)

Assert that the input matrix is square.

# Arguments

  - `A`: Input matrix to validate.
  - `A_sym`: Symbolic name used in error messages.

# Returns

  - `nothing`.

# Validation

  - `size(A, 1) == size(A, 2)`.

# Details

  - Throws `DimensionMismatch` if the check fails.
"""
function assert_matrix_issquare(A::MatNum, A_sym::Symbol = :A)
    @argcheck(size(A, 1) == size(A, 2),
              DimensionMismatch("size($A_sym, 1) == size($A_sym, 2) must hold. Got\nsize($A_sym, 1) => $(size(A, 1))\nsize($A_sym, 2) => $(size(A, 2))."))
    return nothing
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

# Related

  - [`ArrNum`](@ref)
  - [`kron`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#Base.kron)
"""
⊗(A::ArrNum, B::ArrNum) = reshape(kron(B, A), (length(A), length(B)))
"""
    ⊙(A, B)

Elementwise (Hadamard) multiplication.

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

Elementwise (Hadamard) division.

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

Elementwise (Hadamard) addition.

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

Elementwise (Hadamard) subtraction.

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
⊖(A::ArrNum, B::ArrNum) = A .- B
⊖(A::ArrNum, B) = A .- B
⊖(A, B::ArrNum) = A .- B
⊖(A, B) = A - B
"""
    dot_scalar(a::Union{<:Number, <:JuMP.AbstractJuMPScalar}, b::VecNum)
    dot_scalar(a::VecNum, b::Union{<:Number, <:JuMP.AbstractJuMPScalar})
    dot_scalar(a::VecNum, b::VecNum)

Efficient scalar and vector dot product utility.

  - If one argument is a `Union{<:Number, <:JuMP.AbstractJuMPScalar}` and the other an `VecNum`, returns the scalar times the sum of the vector.
  - If both arguments are `VecNum`s, returns their `dot` product.

# Returns

  - `res::Number`: The resulting scalar.

# Examples

```jldoctest
julia> PortfolioOptimisers.dot_scalar(2.0, [1.0, 2.0, 3.0])
12.0

julia> PortfolioOptimisers.dot_scalar([1.0, 2.0, 3.0], 2.0)
12.0

julia> PortfolioOptimisers.dot_scalar([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
32.0
```

# Related

  - [`VecNum`](@ref)
  - [`JuMP.AbstractJuMPScalar`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.JuMP.AbstractJuMPScalar)
"""
function dot_scalar(a::Union{<:Number, <:JuMP.AbstractJuMPScalar}, b::VecNum)
    return a * sum(b)
end
function dot_scalar(a::VecNum, b::Union{<:Number, <:JuMP.AbstractJuMPScalar})
    return sum(a) * b
end
function dot_scalar(a::VecNum, b::VecNum)
    return LinearAlgebra.dot(a, b)
end
"""
    nothing_scalar_array_view(x::Union{Nothing, <:Number, <:Pair, <:VecPair, <:Dict,
                                       AbstractEstimatorValueAlgorithm}, ::Any)
    nothing_scalar_array_view(x::AbstractVector, i)
    nothing_scalar_array_view(x::VecScalar, i)
    nothing_scalar_array_view(x::AbstractVector{<:Union{<:AbstractVector, <:VecScalar}}, i)
    nothing_scalar_array_view(x::AbstractMatrix, i)

Utility for safely viewing into possibly `nothing`, scalar, or array values.

# Arguments

  - `x`: Input value.
  - `i`: Index or indices to view.

# Returns

  - `x`: Input value.

      + `::Union{Nothing, <:Number, <:Pair, <:VecPair, <:Dict}`: Returns `x` unchanged.
      + `::AbstractVector`: Returns `view(x, i)`.
      + `::VecScalar`: Returns `VecScalar(; v = view(x.v, i), s = x.s)`.
      + `::AbstractVector{<:Union{<:AbstractVector, <:VecScalar}}`: Returns a vector of views for each element in `x`.
      + `::AbstractMatrix`: Returns `view(x, i, i)`.

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
function nothing_scalar_array_view(x::Union{Nothing, <:Number, <:Pair, <:VecPair, <:Dict,
                                            <:AbstractEstimatorValueAlgorithm}, ::Any)
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
function nothing_scalar_array_view(x::AbstractMatrix, i)
    return view(x, i, i)
end
"""
    nothing_scalar_array_view_odd_order(x::AbstractMatrix, i, j)

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
function nothing_scalar_array_view_odd_order(x::AbstractMatrix, i, j)
    return view(x, i, j)
end
"""
    nothing_scalar_array_getindex(x::Union{Nothing, <:Number, <:Pair, <:VecPair, <:Dict}, ::Any)
    nothing_scalar_array_getindex(x::AbstractVector, i)
    nothing_scalar_array_getindex(x::VecScalar, i)
    nothing_scalar_array_getindex(x::AbstractVector{<:Union{<:AbstractVector, <:VecScalar}}, i)
    nothing_scalar_array_getindex(x::AbstractMatrix, i)

Utility for safely viewing into possibly `nothing`, scalar, or array values.

# Arguments

  - `x`: Input value.
  - `i`: Index or indices to view.

# Returns

  - `x`: Input value.

      + `::Union{Nothing, <:Number, <:Pair, <:VecPair, <:Dict}`: Returns `x` unchanged.
      + `::AbstractVector`: Returns `view(x, i)`.
      + `::VecScalar`: Returns `VecScalar(; v = view(x.v, i), s = x.s)`.
      + `::AbstractVector{<:Union{<:AbstractVector, <:VecScalar}}`: Returns a vector of views for each element in `x`.
      + `::AbstractMatrix`: Returns `view(x, i, i)`.

# Examples

```jldoctest
julia> PortfolioOptimisers.nothing_scalar_array_getindex(nothing, 1:2)

julia> PortfolioOptimisers.nothing_scalar_array_getindex(3.0, 1:2)
3.0

julia> PortfolioOptimisers.nothing_scalar_array_getindex([1.0, 2.0, 3.0], 2:3)
2-element Vector{Float64}:
 2.0
 3.0

julia> PortfolioOptimisers.nothing_scalar_array_getindex([[1, 2], [3, 4]], 1)
2-element Vector{Int64}:
 1
 3
```
"""
function nothing_scalar_array_getindex(x::Union{Nothing, <:Number, <:Pair, <:VecPair,
                                                <:Dict}, ::Any)
    return x
end
function nothing_scalar_array_getindex(x::AbstractVector, i)
    return x[i]
end
function nothing_scalar_array_getindex(x::VecScalar, i)
    return VecScalar(; v = x.v[i], s = x.s)
end
function nothing_scalar_array_getindex(x::AbstractVector{<:Union{<:AbstractVector,
                                                                 <:VecScalar}}, i)
    return [xi[i] for xi in x]
end
function nothing_scalar_array_getindex(x::AbstractMatrix, i)
    return x[i, i]
end
"""
    nothing_scalar_array_getindex_odd_order(x::AbstractMatrix, i, j)

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
julia> PortfolioOptimisers.nothing_scalar_array_getindex_odd_order(nothing, 1, 2)

julia> PortfolioOptimisers.nothing_scalar_array_getindex_odd_order([1 2; 3 4], 1, 2)
2
```
"""
function nothing_scalar_array_getindex_odd_order(::Nothing, i, j)
    return nothing
end
function nothing_scalar_array_getindex_odd_order(x::AbstractMatrix, i, j)
    return x[i, j]
end
"""
    fourth_moment_index_generator(N::Integer, i)

Constructs an index vector for extracting the fourth moment submatrix corresponding to indices `i` from a covariance matrix of size `N × N`.

# Arguments

  - `N`: Size of the full covariance matrix.
  - `i`: Indices of the variables of interest.

# Returns

  - `idx::Vector{Int}`: Indices for extracting the fourth moment submatrix.

# Examples

```jldoctest
julia> PortfolioOptimisers.fourth_moment_index_generator(3, [1, 2])
4-element Vector{Int64}:
 1
 2
 4
 5
```
"""
function fourth_moment_index_generator(N::Integer, i)
    idx = sizehint!(Int[], length(i)^2)
    for c in i
        append!(idx, (((c - 1) * N + 1):(c * N))[i])
    end
    return idx
end
"""
    traverse_concrete_subtypes(t, ctarr::Option{<:AbstractVector} = nothing)

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
function traverse_concrete_subtypes(t, ctarr::Option{<:AbstractVector} = nothing)
    if isnothing(ctarr)
        ctarr = []
    end
    sts = InteractiveUtils.subtypes(t)
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
"""
    factory(::Nothing, args...; kwargs...)

No-op factory function for constructing objects with a uniform interface.

Defining methods which dispatch on the first argument allows for a consistent factory interface across different types.

# Arguments

  - `::Nothing`: Indicates no object should be constructed.
  - `args...`: Arbitrary positional arguments (ignored).
  - `kwargs...`: Arbitrary keyword arguments (ignored).

# Returns

  - `nothing`.

# Related

  - [`factory`](@ref)
  - [`AbstractEstimator`](@ref)
  - [`AbstractAlgorithm`](@ref)
  - [`AbstractResult`](@ref)
"""
function factory(::Nothing, args...; kwargs...)
    return nothing
end
"""
    abstract type VectorToScalarMeasure <: AbstractAlgorithm end

Abstract supertype for algorithms mapping a vector of real values to a single real value.

`VectorToScalarMeasure` provides a unified interface for algorithms that reduce a vector of real numbers to a scalar, such as minimum, mean, median, or maximum. These are used in constraint generation and centrality-based portfolio constraints to aggregate asset-level metrics.

# Related

  - [`MinValue`](@ref)
  - [`MeanValue`](@ref)
  - [`MedianValue`](@ref)
  - [`MaxValue`](@ref)
  - [`CentralityConstraint`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
abstract type VectorToScalarMeasure <: AbstractAlgorithm end
"""
    const Num_VecToScaM = Union{<:Number, <:VectorToScalarMeasure}

Union type representing either a numeric value or a `VectorToScalarMeasure`.

This type is used to allow functions and fields to accept both plain numbers and objects that implement the `VectorToScalarMeasure` interface, providing flexibility in handling scalar and vector-to-scalar computations.

# Related

  - [`VectorToScalarMeasure`](@ref)
"""
const Num_VecToScaM = Union{<:Number, <:VectorToScalarMeasure}
"""
    struct MinValue <: VectorToScalarMeasure end

Algorithm for reducing a vector of real values to its minimum.

`MinValue` is a concrete subtype of [`VectorToScalarMeasure`](@ref) that returns the minimum value of a vector. It is used in constraint generation and centrality-based portfolio constraints to aggregate asset-level metrics by their minimum.

# Examples

```jldoctest
julia> PortfolioOptimisers.vec_to_real_measure(MinValue(), [1.2, 3.4, 0.7])
0.7
```

# Related

  - [`VectorToScalarMeasure`](@ref)
  - [`MeanValue`](@ref)
  - [`MedianValue`](@ref)
  - [`MaxValue`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
struct MinValue <: VectorToScalarMeasure end
"""
    struct MeanValue{T1} <: VectorToScalarMeasure
        w::T1
    end

Algorithm for reducing a vector of real values to its mean.

`MeanValue` is a concrete subtype of [`VectorToScalarMeasure`](@ref) that returns the mean (average) value of a vector. It is used in constraint generation and centrality-based portfolio constraints to aggregate asset-level metrics by their mean.

# Fields

  - `w`: Optional weights to use for the mean calculation.

# Constructors

```julia
MeanValue(; w::Option{<:StatsBase.AbstractWeights} = nothing)
```

Keyword arguments correspond to the fields above.

## Validation

  - If `w` is not `nothing`, `!isempty(w)`.

# Examples

```jldoctest
julia> PortfolioOptimisers.vec_to_real_measure(MeanValue(), [1.2, 3.4, 0.7])
1.7666666666666666
```

# Related

  - [`VectorToScalarMeasure`](@ref)
  - [`MinValue`](@ref)
  - [`MedianValue`](@ref)
  - [`MaxValue`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
struct MeanValue{T1} <: VectorToScalarMeasure
    w::T1
    function MeanValue(w::Option{<:StatsBase.AbstractWeights})
        if !isnothing(w)
            @argcheck(!isempty(w), IsEmptyError)
        end
        return new{typeof(w)}(w)
    end
end
function MeanValue(; w::Option{<:StatsBase.AbstractWeights} = nothing)
    return MeanValue(w)
end
"""
    factory(mv::MeanValue, w::Option{<:StatsBase.AbstractWeights} = nothing)

Construct a `MeanValue` instance with optional weights, or return the input if weights are unchanged.

# Arguments

  - `mv`: A `MeanValue` instance to update or return.
  - `w`: Optional weights to use for the mean calculation. If `nothing`, returns `mv` unchanged.

# Returns

  - `mv::MeanValue`: A new `MeanValue` with the specified weights, or the original if `w` is `nothing`.

# Details

  - Returns a new `MeanValue` if `w` is provided.
  - Returns the original `mv` if `w` is `nothing`.
  - Ensures consistent construction and updating of `MeanValue` instances.

# Related

  - [`MeanValue`](@ref)
  - [`factory`](@ref)
"""
function factory(mv::MeanValue, w::Option{<:StatsBase.AbstractWeights} = nothing)
    return isnothing(w) ? mv : MeanValue(; w = w)
end
"""
    struct MedianValue{T1} <: VectorToScalarMeasure
        w::T1
    end

Algorithm for reducing a vector of real values to its median.

`MedianValue` is a concrete subtype of [`VectorToScalarMeasure`](@ref) that returns the median value of a vector. It is used in constraint generation and centrality-based portfolio constraints to aggregate asset-level metrics by their median.

# Fields

  - `w`: Optional weights to use for the median calculation.

# Constructors

```julia
MedianValue(; w::Option{<:StatsBase.AbstractWeights} = nothing)
```

Keyword arguments correspond to the fields above.

## Validation

  - If `w` is not `nothing`, `!isempty(w)`.

# Examples

```jldoctest
julia> PortfolioOptimisers.vec_to_real_measure(MedianValue(), [1.2, 3.4, 0.7])
1.2
```

# Related

  - [`VectorToScalarMeasure`](@ref)
  - [`MinValue`](@ref)
  - [`MeanValue`](@ref)
  - [`MaxValue`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
struct MedianValue{T1} <: VectorToScalarMeasure
    w::T1
    function MedianValue(w::Option{<:StatsBase.AbstractWeights})
        if !isnothing(w)
            @argcheck(!isempty(w), IsEmptyError)
        end
        return new{typeof(w)}(w)
    end
end
function MedianValue(; w::Option{<:StatsBase.AbstractWeights} = nothing)
    return MedianValue(w)
end
"""
    factory(mdv::MedianValue, w::Option{<:StatsBase.AbstractWeights} = nothing)

Constructs a `MedianValue` instance with optional weights, or returns the input if weights are unchanged.

# Arguments

  - `mdv`: A `MedianValue` instance to update or return.
  - `w`: Optional weights to use for the median calculation. If `nothing`, returns `mdv` unchanged.

# Returns

  - `mdv::MedianValue`: A new `MedianValue` with the specified weights, or the original if `w` is `nothing`.

# Details

  - Returns a new `MedianValue` if `w` is provided.
  - Returns the original `mdv` if `w` is `nothing`.
  - Ensures consistent construction and updating of `MedianValue` instances.

# Related

  - [`MedianValue`](@ref)
  - [`factory`](@ref)
"""
function factory(mdv::MedianValue, w::Option{<:StatsBase.AbstractWeights} = nothing)
    return isnothing(w) ? mdv : MedianValue(; w = w)
end
"""
    struct MaxValue <: VectorToScalarMeasure end

Algorithm for reducing a vector of real values to its maximum.

`MaxValue` is a concrete subtype of [`VectorToScalarMeasure`](@ref) that returns the maximum value of a vector. It is used in constraint generation and centrality-based portfolio constraints to aggregate asset-level metrics by their maximum.

# Examples

```jldoctest
julia> PortfolioOptimisers.vec_to_real_measure(MaxValue(), [1.2, 3.4, 0.7])
3.4
```

# Related

  - [`VectorToScalarMeasure`](@ref)
  - [`MinValue`](@ref)
  - [`MeanValue`](@ref)
  - [`MedianValue`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
struct MaxValue <: VectorToScalarMeasure end
"""
    struct StdValue{T1, T2} <: VectorToScalarMeasure
        w::T1
        corrected::T2
    end

Algorithm for reducing a vector of real values to its standard deviation.

`StdValue` is a concrete subtype of [`VectorToScalarMeasure`](@ref) that returns the standard deviation of a vector. It is used in constraint generation and centrality-based portfolio constraints to aggregate asset-level metrics by their standard deviation.

# Fields

  - `w`: Optional weights to use for the standard deviation calculation.
  - `corrected`: Indicates whether to use Bessel's correction (`true` for sample standard deviation, `false` for population).

# Constructors

```julia
StdValue(; w::Option{<:StatsBase.AbstractWeights} = nothing, corrected::Bool = true)
```

Keyword arguments correspond to the fields above.

## Validation

  - If `w` is not `nothing`, `!isempty(w)`.

# Examples

```jldoctest
julia> PortfolioOptimisers.vec_to_real_measure(StdValue(), [1.2, 3.4, 0.7])
1.4364307617610164
```

# Related

  - [`VectorToScalarMeasure`](@ref)
  - [`MeanValue`](@ref)
  - [`VarValue`](@ref)
  - [`StandardisedValue`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
struct StdValue{T1, T2} <: VectorToScalarMeasure
    w::T1
    corrected::T2
    function StdValue(w::Option{<:StatsBase.AbstractWeights}, corrected::Bool)
        if !isnothing(w)
            @argcheck(!isempty(w), IsEmptyError)
        end
        return new{typeof(w), typeof(corrected)}(w, corrected)
    end
end
function StdValue(; w::Option{<:StatsBase.AbstractWeights} = nothing,
                  corrected::Bool = true)
    return StdValue(w, corrected)
end
"""
    factory(sv::StdValue, w::Option{<:StatsBase.AbstractWeights} = nothing)

Constructs a `StdValue` instance with optional weights, or returns the input if weights are unchanged.

# Arguments

  - `sv`: A `StdValue` instance to update or return.
  - `w`: Optional weights to use for the standard deviation calculation. If `nothing`, returns `sv` unchanged.

# Returns

  - `sv::StdValue`: A new `StdValue` with the specified weights, or the original if `w` is `nothing`.

# Details

  - Returns a new `StdValue` if `w` is provided.
  - Returns the original `sv` if `w` is `nothing`.
  - Ensures consistent construction and updating of `StdValue` instances.

# Related

  - [`StdValue`](@ref)
  - [`factory`](@ref)
"""
function factory(sv::StdValue, w::Option{<:StatsBase.AbstractWeights} = nothing)
    return isnothing(w) ? sv : StdValue(; w = w, corrected = sv.corrected)
end
"""
    struct VarValue{T1, T2} <: VectorToScalarMeasure
        w::T1
        corrected::T2
    end

Algorithm for reducing a vector of real values to its variance.

`VarValue` is a concrete subtype of [`VectorToScalarMeasure`](@ref) that returns the variance of a vector. It is used in constraint generation and centrality-based portfolio constraints to aggregate asset-level metrics by their variance.

# Fields

  - `w`: Optional weights to use for the variance calculation.
  - `corrected`: Indicates whether to use Bessel's correction (`true` for sample variance, `false` for population).

# Constructors

```julia
VarValue(; w::Option{<:StatsBase.AbstractWeights} = nothing, corrected::Bool = true)
```

Keyword arguments correspond to the fields above.

## Validation

  - If `w` is not `nothing`, `!isempty(w)`.

# Examples

```jldoctest
julia> PortfolioOptimisers.vec_to_real_measure(VarValue(), [1.2, 3.4, 0.7])
2.0633333333333335
```

# Related

  - [`VectorToScalarMeasure`](@ref)
  - [`MeanValue`](@ref)
  - [`StdValue`](@ref)
  - [`StandardisedValue`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
struct VarValue{T1, T2} <: VectorToScalarMeasure
    w::T1
    corrected::T2
    function VarValue(w::Option{<:StatsBase.AbstractWeights}, corrected::Bool)
        if !isnothing(w)
            @argcheck(!isempty(w), IsEmptyError)
        end
        return new{typeof(w), typeof(corrected)}(w, corrected)
    end
end
function VarValue(; w::Option{<:StatsBase.AbstractWeights} = nothing,
                  corrected::Bool = true)
    return VarValue(w, corrected)
end
"""
    factory(vv::VarValue, w::Option{<:StatsBase.AbstractWeights} = nothing)

Constructs a `VarValue` instance with optional weights, or returns the input if weights are unchanged.

# Arguments

  - `vv`: A `VarValue` instance to update or return.
  - `w`: Optional weights to use for the variance calculation. If `nothing`, returns `vv` unchanged.

# Returns

  - `VarValue`: A new `VarValue` with the specified weights, or the original if `w` is `nothing`.

# Details

  - Returns a new `VarValue` if `w` is provided.
  - Returns the original `vv` if `w` is `nothing`.
  - Ensures consistent construction and updating of `VarValue` instances.

# Related

  - [`VarValue`](@ref)
  - [`factory`](@ref)
"""
function factory(vv::VarValue, w::Option{<:StatsBase.AbstractWeights} = nothing)
    return isnothing(w) ? vv : VarValue(; w = w, corrected = vv.corrected)
end
"""
    SumValue <: VectorToScalarMeasure

Algorithm for reducing a vector of real values to its sum.

`SumValue` is a concrete subtype of [`VectorToScalarMeasure`](@ref) that returns the sum of all elements in a vector. It is used in constraint generation and centrality-based portfolio constraints to aggregate asset-level metrics by their sum.

# Examples

```jldoctest
julia> PortfolioOptimisers.vec_to_real_measure(SumValue(), [1.2, 3.4, 0.7])
5.3
```

# Related

  - [`VectorToScalarMeasure`](@ref)
  - [`ProdValue`](@ref)
  - [`ModeValue`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
struct SumValue <: VectorToScalarMeasure end
"""
    ProdValue <: VectorToScalarMeasure

Algorithm for reducing a vector of real values to its product.

`ProdValue` is a concrete subtype of [`VectorToScalarMeasure`](@ref) that returns the product of all elements in a vector. It is used in constraint generation and centrality-based portfolio constraints to aggregate asset-level metrics by their product.

# Examples

```jldoctest
julia> PortfolioOptimisers.vec_to_real_measure(ProdValue(), [1.2, 3.4, 0.7])
2.856
```

# Related

  - [`VectorToScalarMeasure`](@ref)
  - [`SumValue`](@ref)
  - [`ModeValue`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
struct ProdValue <: VectorToScalarMeasure end
"""
    ModeValue <: VectorToScalarMeasure

Algorithm for reducing a vector of real values to its mode.

`ModeValue` is a concrete subtype of [`VectorToScalarMeasure`](@ref) that returns the mode (most frequent value) of a vector. It is used in constraint generation and centrality-based portfolio constraints to aggregate asset-level metrics by their mode.

# Examples

```jldoctest
julia> PortfolioOptimisers.vec_to_real_measure(ModeValue(), [1.2, 3.4, 0.7, 1.2])
1.2
```

# Related

  - [`VectorToScalarMeasure`](@ref)
  - [`SumValue`](@ref)
  - [`ProdValue`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
struct ModeValue <: VectorToScalarMeasure end
"""
    struct StandardisedValue{T1, T2} <: VectorToScalarMeasure
        mv::T1
        sv::T2
    end

Algorithm for reducing a vector of real values to its mean divided by its standard deviation.

`StandardisedValue` is a concrete subtype of [`VectorToScalarMeasure`](@ref) that returns the mean of a vector divided by its standard deviation (i.e., a standardised score). This is used in constraint generation and centrality-based portfolio constraints to aggregate asset-level metrics by their standardised value.

# Fields

  - `mv`: The mean value measure used for the numerator.
  - `sv`: The standard deviation measure used for the denominator.

# Constructors

```julia
StandardisedValue(; mv::MeanValue = MeanValue(), sv::StdValue = StdValue())
```

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> PortfolioOptimisers.vec_to_real_measure(StandardisedValue(), [1.2, 3.4, 0.7])
1.229837387624884
```

# Related

  - [`VectorToScalarMeasure`](@ref)
  - [`MeanValue`](@ref)
  - [`StdValue`](@ref)
  - [`VarValue`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
struct StandardisedValue{T1, T2} <: VectorToScalarMeasure
    mv::T1
    sv::T2
    function StandardisedValue(mv::MeanValue, sv::StdValue)
        return new{typeof(mv), typeof(sv)}(mv, sv)
    end
end
function StandardisedValue(; mv::MeanValue = MeanValue(), sv::StdValue = StdValue())
    return StandardisedValue(mv, sv)
end
"""
    factory(msv::StandardisedValue, w::Option{<:StatsBase.AbstractWeights} = nothing)

Construct a `StandardisedValue` instance with optional weights, or return the input if weights are unchanged.

# Arguments

  - `msv`: A `StandardisedValue` instance to update or return.
  - `w`: Optional weights to use for both the mean and standard deviation measures. If `nothing`, returns `msv` unchanged.

# Returns

  - `msv::StandardisedValue`: A new `StandardisedValue` with the specified weights applied to both `mv` and `sv`, or the original if `w` is `nothing`.

# Details

  - Returns a new `StandardisedValue` if `w` is provided.
  - Returns the original `msv` if `w` is `nothing`.
  - Applies the weights to both the mean (`mv`) and standard deviation (`sv`) fields using their respective `factory` methods.
  - Ensures consistent construction and updating of `StandardisedValue` instances.

# Related

  - [`StandardisedValue`](@ref)
  - [`MeanValue`](@ref)
  - [`StdValue`](@ref)
  - [`factory`](@ref)
"""
function factory(msv::StandardisedValue, w::Option{<:StatsBase.AbstractWeights} = nothing)
    return if isnothing(w)
        msv
    else
        StandardisedValue(; mv = factory(msv.mv, w), sv = factory(msv.sv, w))
    end
end
"""
    vec_to_real_measure(measure::Num_VecToScaM, val::VecNum)

Reduce a vector of real values to a single real value using a specified measure.

`vec_to_real_measure` applies a reduction algorithm (such as minimum, mean, median, or maximum) to a vector of real numbers, as specified by the concrete subtype of [`VectorToScalarMeasure`](@ref). This is used in constraint generation and centrality-based portfolio constraints to aggregate asset-level metrics.

# Arguments

  - `measure`: An instance of a concrete subtype of [`VectorToScalarMeasure`](@ref), or the predefined value to return.
  - `val`: A vector of real values to be reduced (ignored if `measure` is a `Number`).

# Returns

  - `score::Number`: Computed value according to `measure`.

# Examples

```jldoctest
julia> PortfolioOptimisers.vec_to_real_measure(MaxValue(), [1.2, 3.4, 0.7])
3.4

julia> PortfolioOptimisers.vec_to_real_measure(0.9, [1.2, 3.4, 0.7])
0.9
```

# Related

  - [`VectorToScalarMeasure`](@ref)
  - [`MinValue`](@ref)
  - [`MeanValue`](@ref)
  - [`MedianValue`](@ref)
  - [`MaxValue`](@ref)
"""
function vec_to_real_measure(::MinValue, val::VecNum; kwargs...)
    return minimum(val)
end
function vec_to_real_measure(mv::MeanValue, val::VecNum; kwargs...)
    return isnothing(mv.w) ? Statistics.mean(val) : Statistics.mean(val, mv.w)
end
function vec_to_real_measure(mdv::MedianValue, val::VecNum; kwargs...)
    return isnothing(mdv.w) ? Statistics.median(val) : Statistics.median(val, mdv.w)
end
function vec_to_real_measure(::MaxValue, val::VecNum; kwargs...)
    return maximum(val)
end
function vec_to_real_measure(val::Number, ::VecNum; kwargs...)
    return val
end
function vec_to_real_measure(sv::StdValue, val::VecNum; kwargs...)
    return if isnothing(sv.w)
        Statistics.std(val; corrected = sv.corrected, kwargs...)
    else
        Statistics.std(val, sv.w; corrected = sv.corrected, kwargs...)
    end
end
function vec_to_real_measure(vv::VarValue, val::VecNum; kwargs...)
    return if isnothing(vv.w)
        Statistics.var(val; corrected = vv.corrected, kwargs...)
    else
        Statistics.var(val, vv.w; corrected = vv.corrected, kwargs...)
    end
end
function vec_to_real_measure(msv::StandardisedValue, val::VecNum; kwargs...)
    m = vec_to_real_measure(msv.mv, val)
    s = vec_to_real_measure(msv.sv, val; mean = m)
    s = ifelse(iszero(s), sqrt(eps(eltype(s))), s)
    return m / s
end
function vec_to_real_measure(::SumValue, val::VecNum; kwargs...)
    return sum(val)
end
function vec_to_real_measure(::ProdValue, val::VecNum; kwargs...)
    return prod(val)
end
function vec_to_real_measure(::ModeValue, val::VecNum; kwargs...)
    return StatsBase.mode(val)
end

export factory, traverse_concrete_subtypes, concrete_typed_array, MinValue, MeanValue,
       MedianValue, MaxValue, StandardisedValue, StdValue, VarValue, SumValue, ProdValue,
       ModeValue
