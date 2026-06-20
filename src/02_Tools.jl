"""
    ⊗(A::ArrNum, B::ArrNum) -> Matrix{promote_type(eltype(A), eltype(B))}

Tensor product of two arrays. Returns a matrix of size `(length(A), length(B))` where each element is the product of elements from `A` and `B`.

# Arguments

  - `A::ArrNum`: First array.
  - `B::ArrNum`: Second array.

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
    ⊙(A::ArrNum, B::ArrNum) -> Matrix{promote_type(eltype(A), eltype(B))}
    ⊙(A::ArrNum, B) -> Vector{promote_type(eltype(A), eltype(B))}
    ⊙(A, B::ArrNum) -> Vector{promote_type(eltype(A), eltype(B))}
    ⊙(A, B) -> promote_type(eltype(A), eltype(B))

Elementwise (Hadamard) multiplication.

# Arguments

  - `A`: First operand (array or scalar).
  - `B`: Second operand (array or scalar).

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

# Related

  - [`⊗`](@ref)
  - [`⊘`](@ref)
  - [`⊕`](@ref)
  - [`⊖`](@ref)
  - [`ArrNum`](@ref)
"""
⊙(A::ArrNum, B::ArrNum) = A .* B
⊙(A::ArrNum, B) = A * B
⊙(A, B::ArrNum) = A * B
⊙(A, B) = A * B
"""
    ⊘(A::ArrNum, B::ArrNum) -> Matrix{promote_type(eltype(A), eltype(B))}
    ⊘(A::ArrNum, B) -> Vector{promote_type(eltype(A), eltype(B))}
    ⊘(A, B::ArrNum) -> Vector{promote_type(eltype(A), eltype(B))}
    ⊘(A, B) -> promote_type(eltype(A), eltype(B))

Elementwise (Hadamard) division.

# Arguments

  - `A`: Dividend (array or scalar).
  - `B`: Divisor (array or scalar).

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

# Related

  - [`⊗`](@ref)
  - [`⊙`](@ref)
  - [`⊕`](@ref)
  - [`⊖`](@ref)
  - [`ArrNum`](@ref)
"""
⊘(A::ArrNum, B::ArrNum) = A ./ B
⊘(A::ArrNum, B) = A / B
⊘(A, B::ArrNum) = A ./ B
⊘(A, B) = A / B
"""
    ⊕(A::ArrNum, B::ArrNum) -> Matrix{promote_type(eltype(A), eltype(B))}
    ⊕(A::ArrNum, B) -> Vector{promote_type(eltype(A), eltype(B))}
    ⊕(A, B::ArrNum) -> Vector{promote_type(eltype(A), eltype(B))}
    ⊕(A, B) -> promote_type(eltype(A), eltype(B))

Elementwise (Hadamard) addition.

# Arguments

  - `A`: First summand (array or scalar).
  - `B`: Second summand (array or scalar).

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

# Related

  - [`⊗`](@ref)
  - [`⊙`](@ref)
  - [`⊘`](@ref)
  - [`⊖`](@ref)
  - [`ArrNum`](@ref)
"""
⊕(A::ArrNum, B::ArrNum) = A + B
⊕(A::ArrNum, B) = A .+ B
⊕(A, B::ArrNum) = A .+ B
⊕(A, B) = A + B
"""
    ⊖(A::ArrNum, B::ArrNum) -> Matrix{promote_type(eltype(A), eltype(B))}
    ⊖(A::ArrNum, B) -> Vector{promote_type(eltype(A), eltype(B))}
    ⊖(A, B::ArrNum) -> Vector{promote_type(eltype(A), eltype(B))}
    ⊖(A, B) -> promote_type(eltype(A), eltype(B))

Elementwise (Hadamard) subtraction.

# Arguments

  - `A`: Minuend (array or scalar).
  - `B`: Subtrahend (array or scalar).

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

# Related

  - [`⊗`](@ref)
  - [`⊙`](@ref)
  - [`⊘`](@ref)
  - [`⊕`](@ref)
  - [`ArrNum`](@ref)
"""
⊖(A::ArrNum, B::ArrNum) = A - B
⊖(A::ArrNum, B) = A .- B
⊖(A, B::ArrNum) = A .- B
⊖(A, B) = A - B
"""
    dot_scalar(a::Union{<:Number, <:JuMP.AbstractJuMPScalar}, b::VecNum) -> Number
    dot_scalar(a::VecNum, b::Union{<:Number, <:JuMP.AbstractJuMPScalar}) -> Number
    dot_scalar(a::VecNum, b::VecNum) -> Number

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
    nothing_scalar_array_view(
        x::Union{Nothing, <:Number, <:Pair, <:VecPair, <:Dict,
                 <:AbstractEstimatorValueAlgorithm,
                 <:DynamicAbstractWeights, <:AbstractEstimator, <:AbstractAlgorithm},
        ::Any
    ) -> x
    nothing_scalar_array_view(x::AbstractVector, i) -> view(x, i)
    nothing_scalar_array_view(x::VecScalar, i) -> VecScalar(; v = view(x.v, i), s = x.s)
    nothing_scalar_array_view(x::AbstractMatrix, i) -> view(x, i, i)
    nothing_scalar_array_view(
        x::AbstractVector{<:Union{<:AbstractVector, <:AbstractMatrix, <:VecScalar}},
        i
    ) -> [nothing_scalar_array_view(xi, i) for xi in x]

Utility for safely viewing into possibly `nothing`, scalar, or array values.

# Arguments

  - `x`: Input value.
  - `i`: Index or indices to view.

# Returns

  - `x`: Input value.

      + `::Union{Nothing, <:Number, <:Pair, <:VecPair, <:Dict, <:AbstractEstimatorValueAlgorithm, <:DynamicAbstractWeights, <:AbstractEstimator, <:AbstractAlgorithm}`: Returns `x` unchanged.
      + `::AbstractVector`: Returns `view(x, i)`.
      + `::VecScalar`: Returns `VecScalar(; v = view(x.v, i), s = x.s)`.
      + `::AbstractMatrix`: Returns `view(x, i, i)`.
      + `::AbstractVector{<:Union{<:AbstractVector, <:AbstractMatrix, <:VecScalar}}`: Returns a vector of views for each element in `x`.

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

# Related

  - [`nothing_scalar_array_getindex`](@ref)
  - [`VecScalar`](@ref)
"""
function nothing_scalar_array_view(x::Union{Nothing, <:Number, <:Pair, <:VecPair, <:Dict,
                                            <:AbstractEstimatorValueAlgorithm,
                                            <:DynamicAbstractWeights, <:AbstractEstimator,
                                            <:AbstractAlgorithm}, ::Any)
    return x
end
function nothing_scalar_array_view(x::AbstractVector, i)
    return view(x, i)
end
function nothing_scalar_array_view(x::VecScalar, i)
    return VecScalar(; v = view(x.v, i), s = x.s)
end
function nothing_scalar_array_view(x::AbstractMatrix, i)
    return view(x, i, i)
end
function nothing_scalar_array_view(x::AbstractVector{<:Union{<:AbstractVector,
                                                             <:AbstractMatrix, <:VecScalar}},
                                   i)
    return [nothing_scalar_array_view(xi, i) for xi in x]
end
"""
    port_opt_view(x, i, args...; kwargs...) -> nothing_scalar_array_view(x, i)

Universal fallback for [`port_opt_view`](@ref). Any value that has no more specific
`port_opt_view` method is treated as leaf data: it is delegated to
[`nothing_scalar_array_view`](@ref), which slices arrays/`VecScalar`s and passes
scalars, `nothing`, estimators, and algorithms through unchanged. Composed structs that
need to recurse into children define their own (more specific) method — emitted by the
[`@vprop`](@ref) tag or hand-written.

The threaded tail `args...` (typically the returns matrix `X` for the JuMP families) and
any `kwargs` are accepted and dropped here, so a macro-threaded
`port_opt_view(child, i, X)` never `MethodError`s on a leaf field.

# Related

  - [`nothing_scalar_array_view`](@ref)
  - [`@vprop`](@ref)
"""
port_opt_view(x, i, args...; kwargs...) = nothing_scalar_array_view(x, i)
"""
    port_opt_view(x::VecScalar, i, args...) -> nothing_scalar_array_view(x, i)

First-class [`port_opt_view`](@ref) method for [`VecScalar`](@ref): slices the vector
component and preserves the scalar component, delegating to
[`nothing_scalar_array_view`](@ref).
"""
port_opt_view(x::VecScalar, i, args...) = nothing_scalar_array_view(x, i)
"""
    port_opt_view(::Nothing, ::Any; kwargs...) -> nothing
    port_opt_view(::Nothing, ::Any, args...; kwargs...) -> nothing

Canonical absent-value fallback for [`port_opt_view`](@ref): an index view of a
missing (`nothing`) estimator, algorithm, result, or constraint is itself `nothing`.

These methods serve every propagation family. Because many optional fields are typed
`Option{T} = Union{Nothing, T}`, the `::Nothing`-specific methods are also what
disambiguate a `nothing` argument from the family-specific `Option{T}` passthroughs and
from the universal leaf fallback. Both carry a fixed second positional so they dominate
the universal `port_opt_view(x, i, args...)` method.

# Examples

```jldoctest
julia> PortfolioOptimisers.port_opt_view(nothing, 1)

```
"""
port_opt_view(::Nothing, ::Any; kwargs...) = nothing
port_opt_view(::Nothing, ::Any, args...; kwargs...) = nothing
"""
    get_window(window::Option{<:Colon}, args...) -> Option{<:Colon}
    get_window(window::Integer, X::MatNum, dims::Int = 1) -> VecInt
    get_window(window::Integer, X::VecNum, args...) -> VecInt
    get_window(window::VecInt, args...) -> VecInt

Get the observation window index range for a data array.

# Arguments

  - $(arg_dict[:window])
      + `::Option{<:Colon}`: Returns `Colon()`.
      + `::Integer`: Returns the last `window` observations. This operation is safe, so it doesn't error if `window` is larger than the number of observations.
      + `::VecInt`: Returns the `window` argument.
  - $(arg_dict[:X_Xv])
  - $(arg_dict[:dims])

# Returns

  - `window::Option{Union{Colon, <:VecInt}}`: The window index range.

# Related

  - [`moment_window_and_weights`](@ref)
"""
function get_window(::Option{<:Colon}, args...)
    return Colon()
end
function get_window(window::Integer, X::MatNum, dims::Int = 1)
    start = firstindex(X, dims)
    stop = lastindex(X, dims)
    return max(start, stop - window + 1):stop
end
function get_window(window::Integer, X::VecNum, args...)
    start = firstindex(X)
    stop = lastindex(X)
    return max(start, stop - window + 1):stop
end
function get_window(window::VecInt, args...)
    return window
end
"""
    nothing_scalar_array_view_odd_order(::Nothing, i, j)
    nothing_scalar_array_view_odd_order(x::AbstractMatrix, i, j) -> view(x, i, j)

Utility for safely viewing into possibly `nothing` or array values with two indices.

  - If `x` is `nothing`, returns `nothing`.
  - Otherwise, returns `view(x, i, j)`.

# Arguments

  - `x`: Input value.
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

# Related

  - [`nothing_scalar_array_view`](@ref)
  - [`nothing_scalar_array_getindex_odd_order`](@ref)
"""
function nothing_scalar_array_view_odd_order(::Nothing, i, j)
    return nothing
end
function nothing_scalar_array_view_odd_order(x::AbstractMatrix, i, j)
    return view(x, i, j)
end
"""
    nothing_scalar_array_getindex(
        x::Union{Nothing, <:Number, <:Pair, <:VecPair, <:Dict,
                 <:AbstractEstimatorValueAlgorithm,
                 <:DynamicAbstractWeights},
        ::Any
    ) -> x
    nothing_scalar_array_getindex(x::AbstractVector, i) -> x[i]
    nothing_scalar_array_getindex(x::VecScalar, i) -> VecScalar(; v = x.v[i], s = x.s)
    nothing_scalar_array_getindex(x::AbstractMatrix, i) -> x[i, i]
    nothing_scalar_array_getindex(
        x::AbstractVector{<:Union{<:AbstractVector, <:AbstractMatrix, <:VecScalar}},
        i
    ) -> [nothing_scalar_array_getindex(xi, i) for xi in x]

Utility for safely viewing into possibly `nothing`, scalar, or array values.

# Arguments

  - `x`: Input value.
  - `i`: Index or indices to view.

# Returns

  - `x`: Input value.

      + `::Union{Nothing, <:Number, <:Pair, <:VecPair, <:Dict, <:AbstractEstimatorValueAlgorithm, <:DynamicAbstractWeights}`: Returns `x` unchanged.
      + `::AbstractVector`: Returns `x[i]`.
      + `::VecScalar`: Returns `VecScalar(; v = x.v[i], s = x.s)`.
      + `::AbstractVector{<:Union{<:AbstractVector, <:AbstractMatrix, <:VecScalar}}`: Returns a vector of elements indexed by `i`.
      + `::AbstractMatrix`: Returns `x[i, i]`.

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

# Related

  - [`nothing_scalar_array_view`](@ref)
  - [`VecScalar`](@ref)
"""
function nothing_scalar_array_getindex(x::Union{Nothing, <:Number, <:Pair, <:VecPair,
                                                <:Dict, <:AbstractEstimatorValueAlgorithm,
                                                <:DynamicAbstractWeights}, ::Any)
    return x
end
function nothing_scalar_array_getindex(x::AbstractVector, i)
    return x[i]
end
function nothing_scalar_array_getindex(x::VecScalar, i)
    return VecScalar(; v = x.v[i], s = x.s)
end
function nothing_scalar_array_getindex(x::AbstractMatrix, i)
    return x[i, i]
end
function nothing_scalar_array_getindex(x::AbstractVector{<:Union{<:AbstractVector,
                                                                 <:AbstractMatrix,
                                                                 <:VecScalar}}, i)
    return [nothing_scalar_array_getindex(xi, i) for xi in x]
end
"""
    nothing_scalar_array_getindex_odd_order(::Nothing, i, j)
    nothing_scalar_array_getindex_odd_order(x::AbstractMatrix, i, j) -> x[i, j]

Utility for safely indexing into possibly `nothing` or array values with two indices.

  - If `x` is `nothing`, returns `nothing`.
  - Otherwise, returns `x[i, j]`.

# Arguments

  - `x`: Input value.
  - `i`, `j`: Indices to access.

# Returns

  - The corresponding matrix element or `nothing`.

# Examples

```jldoctest
julia> PortfolioOptimisers.nothing_scalar_array_getindex_odd_order(nothing, 1, 2)

julia> PortfolioOptimisers.nothing_scalar_array_getindex_odd_order([1 2; 3 4], 1, 2)
2
```

# Related

  - [`nothing_scalar_array_getindex`](@ref)
  - [`nothing_scalar_array_view_odd_order`](@ref)
"""
function nothing_scalar_array_getindex_odd_order(::Nothing, i, j)
    return nothing
end
function nothing_scalar_array_getindex_odd_order(x::AbstractMatrix, i, j)
    return x[i, j]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Constructs an index vector for extracting the fourth moment submatrix corresponding to indices `i` from a covariance matrix of size `N × N`.

# Arguments

  - `N`: Size of the full covariance matrix.
  - `i`: Indices of the variables of interest.

# Returns

  - `idx::VecInt`: Indices for extracting the fourth moment submatrix.

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
    traverse_concrete_subtypes(t, ctarr::Option{<:AbstractVector} = nothing) -> AbstractVector

Recursively traverse all subtypes of the given abstract type `t` and collect all concrete struct types into `ctarr`.

# Arguments

  - `t`: An abstract type whose subtypes will be traversed.
  - `ctarr`: Optional An array to collect the concrete types. If not provided, a new empty array is created.

# Returns

  - `types::Vector{Any}`: An array containing all concrete struct types that are subtypes (direct or indirect) of `types`.

# Examples

```jldoctest
julia> abstract type MyAbstract end

julia> struct MyConcrete1 <: MyAbstract end

julia> struct MyConcrete2 <: MyAbstract end

julia> traverse_concrete_subtypes(MyAbstract)
2-element Vector{Any}:
 MyConcrete1
 MyConcrete2
```

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractAlgorithm`](@ref)
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
    concrete_typed_array(A::AbstractArray) -> Array{Union{...}}

Convert an `AbstractArray` `A` to a concrete typed array, where each element is of the same type as the elements of `A`.

This is useful for converting arrays with abstract element types to arrays with concrete element types, which can improve performance in some cases.

# Arguments

  - `A`: The input array.

# Returns

  - `A_new::Vector{Union{...}}`: A new array with the same shape as `A`, but with a concrete element type inferred from the elements of `A`.

# Examples

```jldoctest
julia> A = Any[1, 2.0, 3];

julia> PortfolioOptimisers.concrete_typed_array(A)
3-element Vector{Union{Float64, Int64}}:
 1
 2.0
 3
```

# Related

  - [`ArrNum`](@ref)
"""
function concrete_typed_array(A::AbstractArray)
    return reshape(Union{typeof.(A)...}[A...], size(A))
end
"""
    factory(a::Union{Nothing, <:AbstractEstimator, <:AbstractAlgorithm,
                     <:AbstractResult}, args...; kwargs...) -> a

No-op factory function for constructing objects with a uniform interface.

Defining methods which dispatch on the first argument allows for a consistent factory interface across different types.

# Arguments

  - `a`: Indicates no object should be constructed.
  - `args...`: Arbitrary positional arguments (ignored).
  - `kwargs...`: Arbitrary keyword arguments (ignored).

# Returns

  - `a`: The input unchanged.

# Examples

```jldoctest
julia> factory(nothing, 1, 2; x = 3)

julia> factory(MeanValue())
MeanValue
  w ┴ nothing
```

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractAlgorithm`](@ref)
  - [`AbstractResult`](@ref)
"""
function factory(a::Union{Nothing, <:AbstractEstimator, <:AbstractAlgorithm,
                          <:AbstractResult}, args...; kwargs...)
    return a
end
function factory(a::AbstractVector{<:Union{<:AbstractEstimator, <:AbstractAlgorithm,
                                           <:AbstractResult}}, args...; kwargs...)
    return [factory(ai, args...; kwargs...) for ai in a]
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Per-field recursion helper called by [`@propagatable`](@ref)-generated [`factory`](@ref) methods.

Dispatches on the field value type: estimators, algorithms, and results recurse via [`factory`](@ref); observation-weight fields (`::Nothing` or `::StatsBase.AbstractWeights`) are replaced by the incoming [`ObsWeights`](@ref) argument; everything else passes through unchanged.

# Related

  - [`@propagatable`](@ref)
  - [`factory`](@ref)
"""
factory_child(v, args...; kwargs...) = v
function factory_child(v::Union{<:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult},
                       args...; kwargs...)
    return factory(v, args...; kwargs...)
end
function factory_child(v::AbstractArray{<:Union{<:AbstractEstimator, <:AbstractAlgorithm,
                                                <:AbstractResult}}, args...; kwargs...)
    return [factory_child(vi, args...; kwargs...) for vi in v]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the new value of a [`@wprop`](@ref)-tagged observation-weights field during
[`factory`](@ref) propagation.

When an [`ObsWeights`](@ref) argument is threaded through `factory`, the field is
**replaced** by those weights; otherwise the existing field value is kept. This is
distinct from [`factory_child`](@ref) (used by [`@fprop`](@ref)), which recurses into
sub-estimators and leaves `nothing`/non-estimator values unchanged — a weights slot
must not be confused with an optional sub-estimator that happens to be `nothing`.

# Related

  - [`@wprop`](@ref)
  - [`@propagatable`](@ref)
  - [`factory_child`](@ref)
"""
_wprop(field, args...; kwargs...) = field
_wprop(::Any, w::ObsWeights, args...; kwargs...) = w
# ---------------------------------------------------------------------------
# @propagatable — struct-definition macro for factory propagation
# ---------------------------------------------------------------------------

# --- private AST helpers ----------------------------------------------------

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if `x` is a reference to the [`@fprop`](@ref) macro (bare `Symbol` or `GlobalRef`).

Used by [`propagatable_parse_body`](@ref) to detect `@fprop`-tagged fields in a struct body.

# Related

  - [`@fprop`](@ref)
  - [`propagatable_parse_body`](@ref)
  - [`@propagatable`](@ref)
"""
function is_fprop_macro(x)
    return x == Symbol("@fprop") || (x isa GlobalRef && x.name == Symbol("@fprop"))
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if `x` is a reference to the [`@vprop`](@ref) macro (bare `Symbol` or `GlobalRef`).

Used by [`propagatable_parse_body`](@ref) to detect `@vprop`-tagged fields in a struct body.

# Related

  - [`@vprop`](@ref)
  - [`propagatable_parse_body`](@ref)
  - [`@propagatable`](@ref)
"""
function is_vprop_macro(x)
    return x == Symbol("@vprop") || (x isa GlobalRef && x.name == Symbol("@vprop"))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if `x` is a reference to the [`@pprop`](@ref) macro (bare `Symbol` or `GlobalRef`).

Used by [`propagatable_parse_body`](@ref) to detect `@pprop`-tagged fields in a struct body.

# Related

  - [`@pprop`](@ref)
  - [`propagatable_parse_body`](@ref)
  - [`@propagatable`](@ref)
"""
function is_pprop_macro(x)
    return x == Symbol("@pprop") || (x isa GlobalRef && x.name == Symbol("@pprop"))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if `x` is a reference to the [`@cprop`](@ref) macro (bare `Symbol` or `GlobalRef`).

Used by [`propagatable_parse_body`](@ref) to detect `@cprop`-tagged fields in a struct body.

# Related

  - [`@cprop`](@ref)
  - [`propagatable_parse_body`](@ref)
  - [`@propagatable`](@ref)
"""
function is_cprop_macro(x)
    return x == Symbol("@cprop") || (x isa GlobalRef && x.name == Symbol("@cprop"))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if `x` is a reference to the [`@wprop`](@ref) macro (bare `Symbol` or `GlobalRef`).

Used by [`propagatable_parse_body`](@ref) to detect `@wprop`-tagged fields in a struct body.

# Related

  - [`@wprop`](@ref)
  - [`propagatable_parse_body`](@ref)
  - [`@propagatable`](@ref)
"""
function is_wprop_macro(x)
    return x == Symbol("@wprop") || (x isa GlobalRef && x.name == Symbol("@wprop"))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if `x` is a macro call to any of [`@fprop`](@ref), [`@vprop`](@ref),
[`@pprop`](@ref) or [`@cprop`](@ref).

Used by [`propagatable_parse_body`](@ref) to detect tagged fields in a struct body.

# Related

  - [`@fprop`](@ref)
  - [`@vprop`](@ref)
  - [`@pprop`](@ref)
  - [`@cprop`](@ref)
  - [`propagatable_parse_body`](@ref)
  - [`@propagatable`](@ref)
"""
function is_prop_tag_call(x)
    return x isa Expr &&
           x.head == :macrocall &&
           (is_fprop_macro(x.args[1]) ||
            is_vprop_macro(x.args[1]) ||
            is_pprop_macro(x.args[1]) ||
            is_cprop_macro(x.args[1]) ||
            is_wprop_macro(x.args[1]))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Peel any stack of [`@fprop`](@ref)/[`@vprop`](@ref)/[`@pprop`](@ref)/[`@cprop`](@ref) tag
macrocalls off a field expression, recording which tags were present.

Tags may be stacked in either order (`@pprop @fprop field`), which parses as nested
`:macrocall` nodes; this unwraps them all and returns the bare field expression.

# Returns

  - `is_f::Bool`: whether an [`@fprop`](@ref) tag was present.
  - `is_v::Bool`: whether a [`@vprop`](@ref) tag was present.
  - `is_p::Bool`: whether a [`@pprop`](@ref) tag was present.
  - `is_c::Bool`: whether a [`@cprop`](@ref) tag was present.
  - `is_w::Bool`: whether a [`@wprop`](@ref) tag was present.
  - `stripped`: the field expression with all tags removed.

# Related

  - [`propagatable_parse_body`](@ref)
"""
function peel_prop_tags(expr)
    is_f = false
    is_v = false
    is_p = false
    is_c = false
    is_w = false
    while is_prop_tag_call(expr)
        if is_fprop_macro(expr.args[1])
            is_f = true
        elseif is_vprop_macro(expr.args[1])
            is_v = true
        elseif is_pprop_macro(expr.args[1])
            is_p = true
        elseif is_cprop_macro(expr.args[1])
            is_c = true
        else
            is_w = true
        end
        expr = expr.args[end]
    end
    return is_f, is_v, is_p, is_c, is_w, expr
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if `x` is a reference to Julia's `@doc` macro (bare `Symbol` or `GlobalRef`).

Used by [`propagatable_parse_body`](@ref) to recognise docstring-prefixed fields in a struct body.

# Related

  - [`propagatable_parse_body`](@ref)
  - [`@propagatable`](@ref)
"""
is_doc_macro(x) = (x isa GlobalRef && x.name == Symbol("@doc")) || x == Symbol("@doc")

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Extract the field name `Symbol` from a bare field or `field::Type` expression.

Errors with a descriptive message when `expr` is neither a bare `Symbol` nor a
`field::Type` annotation, since only those forms are valid after [`@fprop`](@ref).

# Arguments

  - `expr`: A `Symbol`, an `Expr` with head `:(::)`, or any other expression (triggers an error).

# Returns

  - `name::Symbol`: The field name.

# Related

  - [`@fprop`](@ref)
  - [`propagatable_parse_body`](@ref)
  - [`@propagatable`](@ref)
"""
function extract_field_name(expr)
    if expr isa Symbol
        return expr
    end
    if expr isa Expr && expr.head == :(::)
        return expr.args[1]
    end
    return error("@propagatable: @fprop must precede a bare field name or field::Type, got: $(repr(expr))")
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Recursively unwrap macro call chains to locate the innermost `:struct` node.

Returns `(struct_node, rebuild_fn)` where `rebuild_fn(new_struct)` reconstructs the
original macro chain with `new_struct` in place of the original struct. This allows
[`@propagatable`](@ref) to inject modified struct definitions back into arbitrary macro
wrappers such as `@concrete`.

# Arguments

  - `expr`: A `:struct` expression or a `:macrocall` expression wrapping one.

# Returns

  - `struct_node::Expr`: The innermost `:struct` expression.
  - `rebuild_fn::Function`: A function that, given a replacement `:struct`, returns the
    full macro chain with the replacement in place of the original.

# Related

  - [`propagatable_parse_body`](@ref)
  - [`propagatable_bare_name`](@ref)
  - [`@propagatable`](@ref)
"""
function propagatable_find_struct(expr)
    if !(expr isa Expr)
        error("@propagatable: expected a struct or macro-wrapped struct, got $(typeof(expr))")
    end
    if expr.head == :struct
        return expr, identity
    elseif expr.head == :macrocall
        inner = expr.args[end]
        struct_node, rebuild = propagatable_find_struct(inner)
        prefix = expr.args[1:(end - 1)]
        return struct_node, s -> Expr(:macrocall, prefix..., rebuild(s))
    else
        error("@propagatable: expected a struct definition (possibly wrapped in macros), " *
              "got Expr with head :$(expr.head)")
    end
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Extract the plain struct name `Symbol` from a potentially parameterised or
supertype-constrained name expression.

Handles the forms `Name`, `Name{T, ...}`, and `Name{T, ...} <: SuperType` by
recursively peeling `:curly` and `:<:` wrappers until a bare `Symbol` is reached.

# Arguments

  - `n`: A `Symbol`, or an `Expr` with head `:curly` or `:<:`.

# Returns

  - `name::Symbol`: The plain struct name.

# Related

  - [`propagatable_find_struct`](@ref)
  - [`@propagatable`](@ref)
"""
function propagatable_bare_name(n)
    if n isa Symbol
        return n
    end
    if n isa Expr && n.head == :curly
        return propagatable_bare_name(n.args[1])
    end
    if n isa Expr && n.head == :<:
        return propagatable_bare_name(n.args[1])
    end
    return error("@propagatable: cannot extract struct name from: $(repr(n))")
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the field name `Symbol` for a plain field declaration, or `nothing` for
non-field nodes.

Recognises bare `Symbol` fields and `field::Type` annotations. Returns `nothing` for
`LineNumberNode`s, inner constructors, and any other expression that does not declare
a single named field.

# Arguments

  - `expr`: Any expression appearing in a struct body.

# Returns

  - `name::Symbol`: The field name, if `expr` is a plain field declaration.
  - `nothing`: If `expr` is not a plain field declaration.

# Related

  - [`propagatable_parse_body`](@ref)
  - [`@propagatable`](@ref)
"""
function try_field_name(expr)
    if expr isa Symbol
        return expr
    end
    if expr isa Expr && expr.head == :(::) && expr.args[1] isa Symbol
        return expr.args[1]
    end
    return nothing
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Walk a struct body, collecting [`@fprop`](@ref)-, [`@vprop`](@ref)-, [`@pprop`](@ref)- and
[`@cprop`](@ref)-tagged field names (and all field names) and stripping the tags from the
body.

Handles bare tagged fields (`@fprop field`, …), stacked tags
(`@pprop @fprop field`, in any order), and docstring-prefixed forms
(`"doc" \\n @fprop field`). Non-field nodes (line numbers, inner constructors) are carried
through unchanged.

# Arguments

  - `body::Expr`: The `:block` expression forming the struct body.

# Returns

  - `fprop_fields::Vector{Symbol}`: Names of [`@fprop`](@ref)-tagged fields.
  - `vprop_fields::Vector{Symbol}`: Names of [`@vprop`](@ref)-tagged fields.
  - `pprop_fields::Vector{Symbol}`: Names of [`@pprop`](@ref)-tagged fields.
  - `cprop_fields::Vector{Symbol}`: Names of [`@cprop`](@ref)-tagged fields.
  - `wprop_fields::Vector{Symbol}`: Names of [`@wprop`](@ref)-tagged fields.
  - `all_fields::Vector{Symbol}`: Names of every declared field (tagged or not).
  - `new_body::Expr`: The struct body with all tags stripped.

# Related

  - [`is_fprop_macro`](@ref)
  - [`is_doc_macro`](@ref)
  - [`extract_field_name`](@ref)
  - [`try_field_name`](@ref)
  - [`@fprop`](@ref)
  - [`@propagatable`](@ref)
"""
function propagatable_parse_body(body)
    fprop_fields = Symbol[]
    vprop_fields = Symbol[]
    pprop_fields = Symbol[]
    cprop_fields = Symbol[]
    wprop_fields = Symbol[]
    all_fields   = Symbol[]
    new_args     = Any[]
    function _record!(fname, is_f, is_v, is_p, is_c, is_w)
        if is_f
            push!(fprop_fields, fname)
        end
        if is_v
            push!(vprop_fields, fname)
        end
        if is_p
            push!(pprop_fields, fname)
        end
        if is_c
            push!(cprop_fields, fname)
        end
        if is_w
            push!(wprop_fields, fname)
        end
        push!(all_fields, fname)
        return nothing
    end
    for arg in body.args
        if arg isa Expr && arg.head == :macrocall && is_doc_macro(arg.args[1])
            # Core.@doc "doc" (field or tagged field)
            inner = arg.args[end]
            is_f, is_v, is_p, is_c, is_w, stripped = peel_prop_tags(inner)
            if is_f || is_v || is_p || is_c || is_w
                fname = extract_field_name(stripped)
                _record!(fname, is_f, is_v, is_p, is_c, is_w)
                # Rebuild @doc node with tags stripped: replace last arg with bare field
                push!(new_args, Expr(:macrocall, arg.args[1:(end - 1)]..., stripped))
            else
                # plain docstring'd field — carry through unchanged
                fname = try_field_name(inner)
                if fname !== nothing
                    push!(all_fields, fname)
                end
                push!(new_args, arg)
            end
        elseif is_prop_tag_call(arg)
            # Bare @fprop/@vprop/@pprop/@cprop/@wprop field (tags may be stacked) — no docstring
            is_f, is_v, is_p, is_c, is_w, stripped = peel_prop_tags(arg)
            fname = extract_field_name(stripped)
            _record!(fname, is_f, is_v, is_p, is_c, is_w)
            push!(new_args, stripped)               # strip tags, keep field expr
        else
            # LineNumberNode, bare Symbol field, field::Type, inner constructor, …
            fname = try_field_name(arg)
            if fname !== nothing
                push!(all_fields, fname)
            end
            push!(new_args, arg)
        end
    end
    return fprop_fields, vprop_fields, pprop_fields, cprop_fields, wprop_fields, all_fields,
           Expr(:block, new_args...)
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    @fprop field

Field tag for use inside a [`@propagatable`](@ref) struct body.
Marks the field as participating in [`factory`](@ref) propagation —
`factory_child` will be called on it when `factory` is invoked on the
enclosing struct.

Raises an error if used outside a `@propagatable` struct body.
"""
macro fprop(expr)
    return error("@fprop may only appear inside a @propagatable struct body")
end

"""
    @vprop field

Field tag for use inside a [`@propagatable`](@ref) struct body.
Marks the field as participating in [`port_opt_view`](@ref) propagation —
`port_opt_view` will be called on it when a view (index selection) is propagated
through the enclosing struct.

Orthogonal to [`@fprop`](@ref); the two may be stacked on one field
(`@fprop @vprop field`) when it participates in both factory and view propagation.

Raises an error if used outside a `@propagatable` struct body.
"""
macro vprop(expr)
    return error("@vprop may only appear inside a @propagatable struct body")
end

"""
    @pprop field

Field tag for use inside a [`@propagatable`](@ref) struct body.
Marks the field as **prior-selected**: when `factory(x, pr::AbstractPriorResult, …)` is
invoked, the field is set to `sel(getfield(x, :field), getproperty(pr, :field))` — the
risk-measure value if present, else the same-named moment from the prior result.

Orthogonal to, and stackable with, [`@wprop`](@ref) (`@pprop @wprop w` gives a weights
field both a prior factory and an `ObsWeights` factory) or [`@fprop`](@ref); `@pprop` wins
in the prior method. Mutually exclusive with [`@cprop`](@ref) on a single field. See ADR 0012.

Raises an error if used outside a `@propagatable` struct body.
"""
macro pprop(expr)
    return error("@pprop may only appear inside a @propagatable struct body")
end

"""
    @cprop field

Field tag for use inside a [`@propagatable`](@ref) struct body.
Marks the field as **context-selected**: when `factory(x, pr::AbstractPriorResult, …)` is
invoked, the field is set to `sel(getfield(x, :field), _ctx(args...))` — the risk-measure
value if present, else the threaded optimiser value (a solver) located by type in the
variadic tail. Used for `slv` fields, whose source is a threaded argument rather than the
prior. Mutually exclusive with [`@pprop`](@ref) on a single field. See ADR 0012.

Raises an error if used outside a `@propagatable` struct body.
"""
macro cprop(expr)
    return error("@cprop may only appear inside a @propagatable struct body")
end

"""
    @wprop field

Field tag for use inside a [`@propagatable`](@ref) struct body.
Marks the field as an **observation-weights slot**: when `factory(x, w::ObsWeights, …)`
is invoked, the field is **replaced** by the incoming weights via `_wprop`; when no
[`ObsWeights`](@ref) is threaded, it is left unchanged.

Distinct from [`@fprop`](@ref), which recurses into a sub-estimator value and leaves a
`nothing` value untouched. A weights field defaults to `nothing` (meaning "uniform")
and must become the incoming weights — so it cannot share `@fprop`'s `nothing`-handling
without the two semantics colliding. Use `@wprop` for the `w`/weights field and `@fprop`
for sub-estimators.

Raises an error if used outside a `@propagatable` struct body.
"""
macro wprop(expr)
    return error("@wprop may only appear inside a @propagatable struct body")
end

"""
    @propagatable expr

Define a struct and automatically generate its propagation methods from five
orthogonal, stackable field tags:

  - [`@fprop`](@ref) (factory propagation): tagged fields receive `factory_child`
    calls when [`factory`](@ref) is invoked, recursing runtime *values*
    (observation weights, prior results, solvers, …) down the composition tree.
    A `factory(x, args...)` method is always generated (it is the identity when no
    field is tagged `@fprop`/`@wprop`).
  - [`@wprop`](@ref) (weights replacement): tagged observation-weights fields are
    *replaced* by an incoming [`ObsWeights`](@ref) argument via `_wprop` (and left
    unchanged when none is threaded). Use `@wprop` for the weights slot and `@fprop`
    for sub-estimators — a weights field that defaults to `nothing` must become the
    incoming weights, which conflicts with `@fprop`'s `nothing`-passthrough.
  - [`@vprop`](@ref) (view propagation): tagged fields receive
    [`port_opt_view`](@ref) calls when a view (an index selection) is propagated,
    recursing into composed children and slicing data arrays. A `port_opt_view`
    method is generated **only when at least one field is tagged `@vprop`**.
  - [`@pprop`](@ref) (prior selection): tagged fields are selected from the
    same-named field on a prior result via `sel(getfield(x, :f), getproperty(pr, :f))`.
  - [`@cprop`](@ref) (context selection): tagged fields are selected against a
    threaded optimiser value (a solver) found by type via `sel(getfield(x, :f), _ctx(args...))`.

When at least one field is tagged `@pprop` or `@cprop`, a second method
`factory(x, pr::AbstractPriorResult, args...)` is generated. It selects `@pprop`/`@cprop`
fields as above and threads `@fprop`-only fields with `pr` (`factory_child(getfield(x, :f), pr, args...)`);
a field tagged both `@pprop` and `@fprop` is prior-selected in this method (`@pprop` wins).
Because this method is more specific than the general `factory(x, args...)`, it is chosen
whenever a prior is passed.

Untagged fields pass through unchanged in every method, regardless of type —
tagging is explicit and opt-in. The tags are independent and the relevant field sets
genuinely diverge. `@pprop` and `@cprop` are mutually exclusive on one field (a value comes
from exactly one source); legal stacks are `@pprop @fprop` (sub-estimator) and
`@pprop @wprop` (weights slot). See ADR 0010 and 0012.

Composes with `@concrete` (put `@propagatable` outermost):

```julia
@propagatable @concrete struct MyMeasure <: RiskMeasure
    @pprop @wprop w       # prior factory selects pr.w; ObsWeights factory fills w
    @pprop sigma          # prior-selected from pr.sigma
    @fprop alg            # threaded (recursed) with pr / args
    config                # passed through unchanged
    function MyMeasure(w, sigma, alg, config)
        return new{typeof(w), typeof(sigma), typeof(alg), typeof(config)}(w, sigma, alg,
                                                                          config)
    end
end
```

The generated `factory`/`port_opt_view` methods are added to the
`PortfolioOptimisers` functions, so `@propagatable` works correctly for types
defined in external packages.

Docstrings on the enclosing definition are forwarded correctly via
`Base.@__doc__`.
"""
macro propagatable(expr)
    struct_node, rebuild = propagatable_find_struct(expr)

    type_head   = struct_node.args[2]
    body        = struct_node.args[3]
    struct_name = propagatable_bare_name(type_head)

    fprop_fields, vprop_fields, pprop_fields, cprop_fields, wprop_fields, all_fields, new_body = propagatable_parse_body(body)

    new_struct = Expr(:struct, struct_node.args[1], type_head, new_body)
    chain      = rebuild(new_struct)

    # --- factory propagation (@fprop recurses sub-estimators, @wprop replaces weights) ---
    if isempty(fprop_fields) && isempty(wprop_fields)
        factory_body = :x
    else
        factory_pairs = Any[]
        for f in all_fields
            xf = Expr(:., :x, QuoteNode(f))
            if f in fprop_fields
                push!(factory_pairs,
                      Expr(:kw, f, :(factory_child($xf, args...; kwargs...))))
            elseif f in wprop_fields
                push!(factory_pairs, Expr(:kw, f, :(_wprop($xf, args...; kwargs...))))
            else
                push!(factory_pairs, Expr(:kw, f, xf))
            end
        end
        factory_body = Expr(:call, struct_name, Expr(:parameters, factory_pairs...))
    end

    factory_def = quote
        function factory(x::$struct_name, args...; kwargs...)
            return $factory_body
        end
    end

    defs = Any[factory_def]

    # --- view propagation (@vprop) — emit only when a field opts in ---
    if !isempty(vprop_fields)
        vpass = [f for f in all_fields if !(f in vprop_fields)]
        view_prop_pairs = [Expr(:kw, f,
                                :(port_opt_view($(Expr(:., :x, QuoteNode(f))), i, args...)))
                           for f in vprop_fields]
        view_pass_pairs = [Expr(:kw, f, Expr(:., :x, QuoteNode(f))) for f in vpass]
        view_body = Expr(:call, struct_name,
                         Expr(:parameters, view_prop_pairs..., view_pass_pairs...))
        view_def = quote
            function port_opt_view(x::$struct_name, i, args...)
                return $view_body
            end
        end
        push!(defs, view_def)
    end

    # --- prior/context selection (@pprop / @cprop) — emit only when a field opts in ---
    if !isempty(pprop_fields) || !isempty(cprop_fields)
        sel_pairs = Any[]
        for f in all_fields
            xf = Expr(:., :x, QuoteNode(f))
            if f in pprop_fields            # @pprop wins over @fprop in the prior method
                push!(sel_pairs,
                      Expr(:kw, f, :(sel($xf, getproperty(pr, $(QuoteNode(f)))))))
            elseif f in cprop_fields
                push!(sel_pairs, Expr(:kw, f, :(sel($xf, _ctx(args...)))))
            elseif f in wprop_fields
                push!(sel_pairs, Expr(:kw, f, :(_wprop($xf, args...; kwargs...))))
            elseif f in fprop_fields
                push!(sel_pairs,
                      Expr(:kw, f, :(factory_child($xf, pr, args...; kwargs...))))
            else
                push!(sel_pairs, Expr(:kw, f, xf))
            end
        end
        prior_body = Expr(:call, struct_name, Expr(:parameters, sel_pairs...))
        prior_def = quote
            function factory(x::$struct_name, pr::AbstractPriorResult, args...; kwargs...)
                return $prior_body
            end
        end
        push!(defs, prior_def)
    end

    return esc(quote
                   Base.@__doc__ $chain
                   $(defs...)
               end)
end

# ---------------------------------------------------------------------------
# @forward_properties — standalone property-forwarding macro (ADR 0013)
# ---------------------------------------------------------------------------

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Guard one intermediate node of a [`@forward_properties`](@ref) nested path.

Return `v` unchanged when it is not `nothing`; otherwise throw a
[`PropertyPathError`](@ref) naming the receiver type `T`, the full declared path
`pathstr`, and the `nodestr` node that resolved to `nothing`. Called once per
intermediate hop in the descent generated for a depth-≥2 locator.

# Related

  - [`@forward_properties`](@ref)
  - [`PropertyPathError`](@ref)
"""
function forward_nonnothing(v, ::Type{T}, pathstr, nodestr) where {T}
    if isnothing(v)
        throw(PropertyPathError("cannot descend path `$(pathstr)` on `$(T)`: intermediate `$(nodestr)` is `nothing`"))
    end
    return v
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Flatten a [`@forward_properties`](@ref) locator into its path of field symbols.

A bare identifier `a` becomes `[:a]`; a dotted expression `a.b.c` becomes
`[:a, :b, :c]`. Any other expression raises an error.

# Related

  - [`@forward_properties`](@ref)
  - [`forward_walk_expr`](@ref)
"""
function forward_flatten_path(expr)
    if expr isa Symbol
        return Symbol[expr]
    elseif expr isa Expr && expr.head == :. && length(expr.args) == 2
        leaf = expr.args[2]
        leaf = leaf isa QuoteNode ? leaf.value : leaf
        if !(leaf isa Symbol)
            return error("@forward_properties: invalid locator leaf $(repr(expr.args[2]))")
        end
        return Symbol[forward_flatten_path(expr.args[1])..., leaf]
    else
        return error("@forward_properties: locator must be a bare name or a dotted path (`a.b.c`), got: $(repr(expr))")
    end
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the expression that descends a [`@forward_properties`](@ref) `path` (a
vector of field symbols) on the receiver `x`, returning the value at the path.

A depth-1 path is a single `getfield`. A depth-≥2 path descends hop by hop,
guarding every intermediate with [`forward_nonnothing`](@ref) (keyed on the
receiver type `struct_name`) so a `nothing` node throws a path-naming
[`PropertyPathError`](@ref). When `broadcast` is `true`, the final hop maps over
the penultimate value if it is an `AbstractVector` (the scalar-or-vector
solution case), otherwise it is a plain access.

# Related

  - [`@forward_properties`](@ref)
  - [`forward_nonnothing`](@ref)
"""
function forward_walk_expr(path, struct_name, broadcast::Bool)
    if length(path) == 1
        return :(getfield(x, $(QuoteNode(path[1]))))
    end
    pathstr = join(string.(path), ".")
    stmts = Any[:(__v = getfield(x, $(QuoteNode(path[1]))))]
    for k in 2:length(path)
        nodestr = join(string.(path[1:(k - 1)]), ".")
        push!(stmts, :(__v = $(forward_nonnothing)(__v, $struct_name, $pathstr, $nodestr)))
        leaf = QuoteNode(path[k])
        if k == length(path) && broadcast
            push!(stmts, :(__v = if isa(__v, AbstractVector)
                               getproperty.(__v, $leaf)
                           else
                               getproperty(__v, $leaf)
                           end))
        else
            push!(stmts, :(__v = getproperty(__v, $leaf)))
        end
    end
    push!(stmts, :__v)
    return Expr(:let, Expr(:block), Expr(:block, stmts...))
end

"""
    @forward_properties T begin
        forward(loc)
        forward(loc, names...)
        alias(exposed, loc)
        compute(exposed, loc; broadcast)
        compute(exposed, fn)
        swap(field, loc)
        swap(field, fn)
    end

Generate the `Base.getproperty` / `Base.propertynames` pair for type `T` from a
block of declarative forwarding rules, so the property-forwarding decision lives
in one declared surface instead of a hand-written `getproperty` body (ADR 0013).
`T` may be a bare type name or a parametric/`UnionAll` signature
(`Foo{<:Any, Nothing, <:Any}`), so a `swap` can be specialised per type parameter.

All names are written as **bare identifiers**. Every rule names its source via a
**locator** — a bare name `a` (the field `a` of the receiver) or a dotted path
`a.b.c` (the receiver-rooted path `obj.a.b.c`, any depth). Nesting is simply more
dots; a depth-≥2 path guards each intermediate and throws a
[`PropertyPathError`](@ref) naming the path when a node is `nothing`.

`forward`/`alias`/`compute` only *add new virtual names* and so resolve **after**
the receiver's own fields; `swap` *replaces the value of an existing field* and so
resolves **before** the field check.

# Rules

  - `forward(loc)`: forward *all* properties of the value at `loc`
    (`sym in propertynames(value)` ? `getproperty(value, sym)`).
  - `forward(loc, names...)`: forward only the named subset from the value at `loc`.
  - `alias(exposed, loc)`: expose `exposed` as the value at `loc` (renaming).
  - `compute(exposed, loc; broadcast)`: expose `exposed` via a dotted locator
    (depth ≥ 2); `broadcast` maps the final hop over a vector penultimate value.
  - `compute(exposed, fn)`: expose `exposed` as `fn(obj)`; `fn` must be an
    anonymous function (a lambda), which would otherwise be ambiguous with a
    dotted path.
  - `swap(field, loc)` / `swap(field, fn)`: override an *existing* field's value
    with the value at `loc` (bare name, e.g. `swap(L, M)`, or dotted path) or with
    `fn(obj)`. Unlike the others it takes precedence over the own-field check, and
    is the only rule that may name a real field. Typically specialised on a
    parametric `T` (`swap(L, M)` on `Regression{<:Any, Nothing, <:Any}`).
    The locator form reads through `getfield` and is recursion-safe; in the
    **function form the body must read the swapped field via `getfield(obj, :field)`,
    never `obj.field`**, since dot-access on the swapped field re-enters
    `getproperty` and recurses (`StackOverflowError`). Other fields may use
    dot-access freely.

# Details

The generated `getproperty` applies any `swap` rules first (in declaration order),
then checks the receiver's own `fieldnames` (via `getfield`, so it never recurses),
then each remaining rule in declaration order with first-match-wins, then falls
through to `getfield(x, sym)` (the standard "no field" error on `T`). The generated
`propertynames` unions the own field names with every forwarded, subset, aliased,
computed and swapped name, deduplicated.

# Related

  - [`PropertyPathError`](@ref)
  - [`@propagatable`](@ref)
"""
macro forward_properties(T, block)
    if !(block isa Expr && block.head == :block)
        return error("@forward_properties: expected a `begin ... end` block of rules")
    end
    getprop_branches = Any[]
    swap_branches = Any[]
    propname_contribs = Any[]
    for stmt in block.args
        if stmt isa LineNumberNode
            continue
        end
        if !(stmt isa Expr && stmt.head == :call)
            return error("@forward_properties: each rule must be a `forward`/`alias`/`compute`/`swap` call, got: $(repr(stmt))")
        end
        marker = stmt.args[1]
        args = stmt.args[2:end]
        broadcast = false
        if !isempty(args) && args[1] isa Expr && args[1].head == :parameters
            for p in args[1].args
                if p === :broadcast
                    broadcast = true
                else
                    return error("@forward_properties: unknown option $(repr(p)) (only `broadcast` is supported)")
                end
            end
            args = args[2:end]
        end
        if marker == :forward
            if isempty(args)
                return error("@forward_properties: `forward` needs a locator")
            end
            path = forward_flatten_path(args[1])
            walk = forward_walk_expr(path, T, false)
            if length(args) == 1
                # forward all properties of the located value
                push!(getprop_branches, quote
                          let __c = $walk
                              if sym in propertynames(__c)
                                  return getproperty(__c, sym)
                              else
                                  false
                              end
                          end
                      end)
                push!(propname_contribs, Expr(:..., :(propertynames($walk))))
            else
                names = args[2:end]
                for n in names
                    if !(n isa Symbol)
                        return error("@forward_properties: `forward` names must be bare identifiers, got: $(repr(n))")
                    else
                        true
                    end
                end
                nameset = Expr(:tuple, (QuoteNode(n) for n in names)...)
                push!(getprop_branches,
                      :(sym in $nameset && return getproperty($walk, sym)))
                append!(propname_contribs, (QuoteNode(n) for n in names))
            end
        elseif marker == :alias
            if !(length(args) == 2)
                return error("@forward_properties: `alias` takes `(exposed, locator)`, got: $(repr(stmt))")
            end
            exposed = args[1]
            if !(exposed isa Symbol)
                return error("@forward_properties: `alias` exposed name must be a bare identifier, got: $(repr(exposed))")
            end
            path = forward_flatten_path(args[2])
            walk = forward_walk_expr(path, T, false)
            push!(getprop_branches, :(sym === $(QuoteNode(exposed)) && return $walk))
            push!(propname_contribs, QuoteNode(exposed))
        elseif marker == :compute
            if !(length(args) == 2)
                return error("@forward_properties: `compute` takes `(exposed, locator|fn)`, got: $(repr(stmt))")
            end
            exposed = args[1]
            if !(exposed isa Symbol)
                return error("@forward_properties: `compute` exposed name must be a bare identifier, got: $(repr(exposed))")
            end
            src = args[2]
            if src isa Expr && src.head == :->
                if broadcast
                    return error("@forward_properties: `broadcast` does not apply to the function form of `compute`")
                end
                push!(getprop_branches,
                      :(sym === $(QuoteNode(exposed)) && return ($src)(x)))
            elseif src isa Expr && src.head == :.
                path = forward_flatten_path(src)
                walk = forward_walk_expr(path, T, broadcast)
                push!(getprop_branches, :(sym === $(QuoteNode(exposed)) && return $walk))
            else
                return error("@forward_properties: `compute` source must be a dotted path (depth ≥ 2) or an anonymous function, got: $(repr(src))")
            end
            push!(propname_contribs, QuoteNode(exposed))
        elseif marker == :swap
            if !(length(args) == 2)
                return error("@forward_properties: `swap` takes `(field, locator|fn)`, got: $(repr(stmt))")
            end
            exposed = args[1]
            if !(exposed isa Symbol)
                return error("@forward_properties: `swap` field name must be a bare identifier, got: $(repr(exposed))")
            end
            src = args[2]
            if src isa Expr && src.head == :->
                if broadcast
                    return error("@forward_properties: `broadcast` does not apply to the function form of `swap`")
                end
                push!(swap_branches, :(sym === $(QuoteNode(exposed)) && return ($src)(x)))
            elseif (src isa Expr && src.head == :.) || src isa Symbol
                path = forward_flatten_path(src)
                walk = forward_walk_expr(path, T, broadcast)
                push!(swap_branches, :(sym === $(QuoteNode(exposed)) && return $walk))
            else
                return error("@forward_properties: `swap` source must be a bare name, a dotted path, or an anonymous function, got: $(repr(src))")
            end
            push!(propname_contribs, QuoteNode(exposed))
        else
            return error("@forward_properties: unknown rule `$(marker)` (expected `forward`, `alias`, `compute`, or `swap`)")
        end
    end
    getproperty_def = quote
        function Base.getproperty(x::$T, sym::Symbol)
            $(swap_branches...)
            if sym in fieldnames($T)
                return getfield(x, sym)
            end
            $(getprop_branches...)
            return getfield(x, sym)
        end
    end
    propertynames_tuple = Expr(:tuple, Expr(:..., :(fieldnames($T))), propname_contribs...)
    propertynames_def = quote
        function Base.propertynames(x::$T)
            return Tuple(unique($propertynames_tuple))
        end
    end
    return esc(quote
                   $getproperty_def
                   $propertynames_def
               end)
end

"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for algorithms mapping a vector of real values to a single real value.

`VectorToScalarMeasure` provides a unified interface for algorithms that reduce a vector of real numbers to a scalar, such as minimum, mean, median, or maximum. These are used in constraint generation and centrality-based portfolio constraints to aggregate asset-level metrics.

# Interfaces

In order to implement a new vector-to-scalar measure that works seamlessly with the library, subtype `VectorToScalarMeasure` and implement the following method:

## Reduction method

  - `vec_to_real_measure(measure::VectorToScalarMeasure, val::VecNum) -> Number`: Reduces `val` to a single scalar.

### Arguments

  - `measure`: Concrete subtype instance.
  - `val`: Vector of real values to reduce.

### Returns

  - `score::Number`: Computed scalar.

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
    const Num_VecToScaM = Union{<:Number, <:VectorToScalarMeasure, <:Function}

Union type representing either a numeric value or a `VectorToScalarMeasure`.

This type is used to allow functions and fields to accept both plain numbers and objects that implement the `VectorToScalarMeasure` interface, providing flexibility in handling scalar and vector-to-scalar computations.

# Related

  - [`VectorToScalarMeasure`](@ref)
"""
const Num_VecToScaM = Union{<:Number, <:VectorToScalarMeasure, <:Function}
"""
$(DocStringExtensions.TYPEDEF)

Algorithm for reducing a vector of real values to its minimum.

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
$(DocStringExtensions.TYPEDEF)

Algorithm for reducing a vector of real values to its optionally weighted mean.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MeanValue(;
        w::Option{<:ObsWeights} = nothing,
    ) -> MeanValue

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])

## Curried parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

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
  - [`factory`](@ref)
"""
@propagatable @concrete struct MeanValue <: VectorToScalarMeasure
    """
    $(field_dict[:oow])
    """
    @wprop w
    function MeanValue(w::Option{<:ObsWeights})
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(w)}(w)
    end
end
function MeanValue(; w::Option{<:ObsWeights} = nothing)
    return MeanValue(w)
end
"""
$(DocStringExtensions.TYPEDEF)

Algorithm for reducing a vector of real values to its optionally weighted median.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MedianValue(;
        w::Option{<:ObsWeights} = nothing,
    ) -> MedianValue

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])

## Curried parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

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
  - [`factory`](@ref)
"""
@propagatable @concrete struct MedianValue <: VectorToScalarMeasure
    """
    $(field_dict[:oow])
    """
    @wprop w
    function MedianValue(w::Option{<:ObsWeights})
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(w)}(w)
    end
end
function MedianValue(; w::Option{<:ObsWeights} = nothing)
    return MedianValue(w)
end
"""
$(DocStringExtensions.TYPEDEF)

Algorithm for reducing a vector of real values to its maximum.

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
$(DocStringExtensions.TYPEDEF)

Algorithm for reducing a vector of real values to its optionally weighted standard deviation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    StdValue(;
        w::Option{<:ObsWeights} = nothing,
        corrected::Bool = true,
    ) -> StdValue

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])

## Curried parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

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
  - [`factory`](@ref)
"""
@propagatable @concrete struct StdValue <: VectorToScalarMeasure
    """
    $(field_dict[:oow])
    """
    @wprop w
    """
    $(field_dict[:corrected])
    """
    corrected
    function StdValue(w::Option{<:ObsWeights}, corrected::Bool)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(w), typeof(corrected)}(w, corrected)
    end
end
function StdValue(; w::Option{<:ObsWeights} = nothing, corrected::Bool = true)
    return StdValue(w, corrected)
end
"""
$(DocStringExtensions.TYPEDEF)

Algorithm for reducing a vector of real values to its optionally weighted variance.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    VarValue(;
        w::Option{<:ObsWeights} = nothing,
        corrected::Bool = true,
    ) -> VarValue

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])

## Curried parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

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
  - [`factory`](@ref)
"""
@propagatable @concrete struct VarValue <: VectorToScalarMeasure
    """
    $(field_dict[:oow])
    """
    @wprop w
    """
    $(field_dict[:corrected])
    """
    corrected
    function VarValue(w::Option{<:ObsWeights}, corrected::Bool)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(w), typeof(corrected)}(w, corrected)
    end
end
function VarValue(; w::Option{<:ObsWeights} = nothing, corrected::Bool = true)
    return VarValue(w, corrected)
end
"""
$(DocStringExtensions.TYPEDEF)

Algorithm for reducing a vector of real values to its sum.

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
$(DocStringExtensions.TYPEDEF)

Algorithm for reducing a vector of real values to its product.

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
$(DocStringExtensions.TYPEDEF)

Algorithm for reducing a vector of real values to its mode.

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
$(DocStringExtensions.TYPEDEF)

Algorithm for reducing a vector of real values to its optionally weighted mean divided by its optionally weighted standard deviation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    StandardisedValue(;
        mv::MeanValue = MeanValue(),
        sv::StdValue = StdValue(),
    ) -> StandardisedValue

Keywords correspond to the struct's fields.

## Curried parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `mv`: Recursively updated via [`factory`](@ref).
  - `sv`: Recursively updated via [`factory`](@ref).

# Examples

```jldoctest
julia> PortfolioOptimisers.vec_to_real_measure(StandardisedValue(), [1.2, 3.4, 0.7])
1.2299003291330186
```

# Related

  - [`VectorToScalarMeasure`](@ref)
  - [`MeanValue`](@ref)
  - [`StdValue`](@ref)
  - [`VarValue`](@ref)
  - [`vec_to_real_measure`](@ref)
  - [`factory`](@ref)
"""
@propagatable @concrete struct StandardisedValue <: VectorToScalarMeasure
    """
    The mean value measure used for the numerator.
    """
    @fprop mv
    """
    The standard deviation measure used for the denominator.
    """
    @fprop sv
    function StandardisedValue(mv::MeanValue, sv::StdValue)
        return new{typeof(mv), typeof(sv)}(mv, sv)
    end
end
function StandardisedValue(; mv::MeanValue = MeanValue(), sv::StdValue = StdValue())
    return StandardisedValue(mv, sv)
end
"""
    vec_to_real_measure(measure::Num_VecToScaM, val::VecNum) -> Number

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
  - [`Num_VecToScaM`](@ref)
"""
function vec_to_real_measure(::MinValue,
                             val::Union{<:VecNum, NTuple{N, <:Number} where {N}}; kwargs...)
    return minimum(val)
end
function vec_to_real_measure(mv::MeanValue{Nothing},
                             val::Union{<:VecNum, NTuple{N, <:Number} where {N}}; kwargs...)
    return Statistics.mean(val)
end
function vec_to_real_measure(mv::MeanValue{<:ObsWeights}, val::VecNum; kwargs...)
    return Statistics.mean(val, mv.w)
end
function vec_to_real_measure(mv::MeanValue{<:ObsWeights},
                             val::NTuple{N, <:Number} where {N}; kwargs...)
    return Statistics.mean(collect(val), mv.w)
end
function vec_to_real_measure(mdv::MedianValue{Nothing},
                             val::Union{<:VecNum, NTuple{N, <:Number} where {N}}; kwargs...)
    return Statistics.median(val)
end
function vec_to_real_measure(mdv::MedianValue{<:ObsWeights}, val::VecNum; kwargs...)
    return Statistics.median(val, mdv.w)
end
function vec_to_real_measure(mdv::MedianValue{<:ObsWeights},
                             val::NTuple{N, <:Number} where {N}; kwargs...)
    return Statistics.median(collect(val), mdv.w)
end
function vec_to_real_measure(::MaxValue,
                             val::Union{<:VecNum, NTuple{N, <:Number} where {N}}; kwargs...)
    return maximum(val)
end
function vec_to_real_measure(val::Number, ::Union{<:VecNum, NTuple{N, <:Number} where {N}};
                             kwargs...)
    return val
end
function vec_to_real_measure(sv::StdValue{Nothing},
                             val::Union{<:VecNum, NTuple{N, <:Number} where {N}}; kwargs...)
    return Statistics.std(val; corrected = sv.corrected, kwargs...)
end
function vec_to_real_measure(sv::StdValue{<:ObsWeights}, val::VecNum; kwargs...)
    return Statistics.std(val, sv.w; corrected = sv.corrected, kwargs...)
end
function vec_to_real_measure(sv::StdValue{<:ObsWeights}, val::NTuple{N, <:Number} where {N};
                             kwargs...)
    return Statistics.std(collect(val), sv.w; corrected = sv.corrected, kwargs...)
end
function vec_to_real_measure(vv::VarValue{Nothing},
                             val::Union{<:VecNum, NTuple{N, <:Number} where {N}}; kwargs...)
    return Statistics.var(val; corrected = vv.corrected, kwargs...)
end
function vec_to_real_measure(vv::VarValue{<:ObsWeights}, val::VecNum; kwargs...)
    return Statistics.var(val, vv.w; corrected = vv.corrected, kwargs...)
end
function vec_to_real_measure(vv::VarValue{<:ObsWeights}, val::NTuple{N, <:Number} where {N};
                             kwargs...)
    return Statistics.var(collect(val), vv.w; corrected = vv.corrected, kwargs...)
end
function vec_to_real_measure(msv::StandardisedValue,
                             val::Union{<:VecNum, NTuple{N, <:Number} where {N}}; kwargs...)
    m = vec_to_real_measure(msv.mv, val)
    s = vec_to_real_measure(msv.sv, val; mean = m)
    s = ifelse(iszero(s), sqrt(eps(eltype(s))), s)
    return m / s
end
function vec_to_real_measure(::SumValue,
                             val::Union{<:VecNum, NTuple{N, <:Number} where {N}}; kwargs...)
    return sum(val)
end
function vec_to_real_measure(::ProdValue,
                             val::Union{<:VecNum, NTuple{N, <:Number} where {N}}; kwargs...)
    return prod(val)
end
function vec_to_real_measure(::ModeValue,
                             val::Union{<:VecNum, NTuple{N, <:Number} where {N}}; kwargs...)
    return StatsBase.mode(val)
end
function vec_to_real_measure(f::Function,
                             val::Union{<:VecNum, NTuple{N, <:Number} where {N}}; kwargs...)
    return f(val)
end

export @propagatable, @fprop, @vprop, @pprop, @cprop, @wprop, @forward_properties, factory,
       traverse_concrete_subtypes, concrete_typed_array, MinValue, MeanValue, MedianValue,
       MaxValue, StandardisedValue, StdValue, VarValue, SumValue, ProdValue, ModeValue
