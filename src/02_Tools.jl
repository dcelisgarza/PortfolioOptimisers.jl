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
                                            <:AbstractAlgorithm,
                                            <:StatsBase.CovarianceEstimator}, ::Any)
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

Sub-select an estimator, result, or algorithm to the asset/observation index `i`.

`port_opt_view` is the **index-selection counterpart of [`factory`](@ref)**: where `factory`
threads *runtime values* down a composed struct tree, `port_opt_view` threads an *index
selection* — restricting every data-bearing field and composed child to the subset `i`.
It is the mechanism that makes meta-optimisers ([`NestedClustered`](@ref),
[`SubsetResampling`](@ref)) and cross-validation variants operate on subproblems with
identical struct shapes.

Callers do not normally call `port_opt_view` directly; it is driven by meta-optimisers and
cross-validation internals. It is `public` (not exported) because extension authors who
implement a new composed estimator may need to define a method. Use [`@vprop`](@ref) on
data-bearing fields to have the method generated automatically.

This universal fallback handles *leaf* values: arrays are sliced via
[`nothing_scalar_array_view`](@ref); scalars, `nothing`, estimators without data fields,
and algorithms pass through unchanged. Composed structs that recurse into children define
their own (more specific) method — emitted by [`@vprop`](@ref) or hand-written.

The threaded tail `args...` (typically the returns matrix `X` for the JuMP families) and
any `kwargs` are accepted and dropped here, so a macro-threaded
`port_opt_view(child, i, X)` never `MethodError`s on a leaf field.

# Related

  - [`factory`](@ref)
  - [`@vprop`](@ref)
  - [`nothing_scalar_array_view`](@ref)
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
    port_opt_view(x::AbstractVector{<:Union{Nothing, <:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult}}, i, args...; kwargs...) -> Vector

Generic vector method for [`port_opt_view`](@ref): view each element of `x` at the index selection `i`.

This is the index-selection twin of the vector [`factory`](@ref) method, and it is the **one forwarding contract** for every vector-valued propagation field. The tail `args...` (typically the returns matrix `X`) and every keyword reach each element unchanged, so a family that admits a vector of estimators, algorithms, or results needs no method of its own.

Without it such a vector falls through to the universal leaf method `port_opt_view(x, i, args...)`, which slices the vector itself through [`nothing_scalar_array_view`](@ref) — the asset index would select *elements* instead of assets. A family that needs more than the forward, such as a concrete element type ([`concrete_typed_array_if_abstract`](@ref)) or a passthrough, defines its own more specific method.

# Arguments

  - `x`: Vector of estimators, algorithms, results, or `nothing`.
  - `i`: Index selection.
  - `args...`: Threaded tail, forwarded to each element.
  - `kwargs...`: Keyword arguments, forwarded to each element.

# Returns

  - `v::Vector`: The element-wise views.

# Related

  - [`factory`](@ref)
  - [`concrete_typed_array_if_abstract`](@ref)
  - [`@vprop`](@ref)
"""
function port_opt_view(x::AbstractVector{<:Union{Nothing, <:AbstractEstimator,
                                                 <:AbstractAlgorithm, <:AbstractResult}}, i,
                       args...; kwargs...)
    return [port_opt_view(xi, i, args...; kwargs...) for xi in x]
end
"""
    obs_weights_view(x, i) -> typeof(x)

Sub-select an estimator's **observation weights** to the observations `i`.

`obs_weights_view` is the observation-axis counterpart of [`port_opt_view`](@ref), and it is generated by [`@propagatable`](@ref) from the tags a struct already carries: [`@wprop`](@ref) marks the field that *holds* the weights, so that field is indexed, and [`@fprop`](@ref) marks a composed child, so the verb recurses into it. Every other field is carried through unchanged, and the struct's type does not change.

# Why the observation axis needs its own verb

[`port_opt_view`](@ref) threads one index into every `@vprop`-tagged field, and at its call sites — the meta-optimisers and the cross-validation splitters — that index selects **assets**. An observation weight is one value per row of the sample, so slicing it there would be wrong. The two axes are told apart by which verb is called, not by the index.

[`factory`](@ref) reads the same `@wprop` tag on the same field, and does a different thing with it: it **replaces** the field with an incoming [`ObsWeights`](@ref) value, at every level of the tree at once. That is why a slice cannot go through `factory`. A [`SimpleVariance`](@ref) holding a weighted mean and an unweighted dispersion comes back from `factory` with both weighted, which is a different estimator; here each field is indexed on its own, so a field that held `nothing` still holds `nothing`.

# Arguments

  - `x`: Estimator, algorithm, result, weights vector, or `nothing`. The argument is untyped, because the variance estimators subtype `StatsBase.CovarianceEstimator` while the expected returns estimators subtype [`AbstractEstimator`](@ref), and both reach this verb.
  - `i`: Index or indices of the observations to keep.

# Returns

  - `x`: The value, with every observation-weights field indexed to `i`.

# Details

  - This universal fallback returns `x` unchanged. An estimator that carries no weights, and one whose struct is not [`@propagatable`](@ref), therefore behave as they did before the verb existed.
  - A struct gains a generated method when it has at least one `@wprop`-tagged field. A hand-written type that holds weights outside that tag must define its own method, or its weights keep their full-sample length and the windowed call raises.
  - [`realised_vol`](@ref) is the site that drives it.

# Related

  - [`port_opt_view`](@ref)
  - [`factory`](@ref)
  - [`@wprop`](@ref)
  - [`PROP_TAG_CHANNELS`](@ref)
  - [`nothing_scalar_array_getindex`](@ref)
  - [`ObsWeights`](@ref)
"""
obs_weights_view(x, ::Any) = x
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Vector overload of [`obs_weights_view`](@ref). Applies the verb to every element, so an `@fprop`-tagged field holding a vector of composed children is not silently skipped.
"""
function obs_weights_view(x::AbstractVector{<:Union{Nothing, <:AbstractEstimator,
                                                    <:AbstractAlgorithm, <:AbstractResult}},
                          i)
    return [obs_weights_view(xi, i) for xi in x]
end
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

julia> PortfolioOptimisers.traverse_concrete_subtypes(MyAbstract)
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
    concrete_typed_array_if_abstract(A::AbstractArray) -> AbstractArray

Narrow the element type of `A` with [`concrete_typed_array`](@ref), but only when that element type is abstract.

The generic vector methods of [`factory`](@ref) and [`port_opt_view`](@ref) rebuild a vector field element by element. A comprehension over a heterogeneous vector infers an abstract element type, which costs a dynamic dispatch at every later use. This is the opt-in narrowing step for the families that want the concrete element type back.

# Arguments

  - `A`: The rebuilt array.

# Returns

  - `A`: Unchanged if `eltype(A)` is concrete, else [`concrete_typed_array`](@ref) of `A`.

# Examples

```jldoctest
julia> PortfolioOptimisers.concrete_typed_array_if_abstract([1, 2, 3])
3-element Vector{Int64}:
 1
 2
 3

julia> PortfolioOptimisers.concrete_typed_array_if_abstract(Any[1, 2.0])
2-element Vector{Union{Float64, Int64}}:
 1
 2.0
```

# Related

  - [`concrete_typed_array`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
"""
function concrete_typed_array_if_abstract(A::AbstractArray)
    return isabstracttype(eltype(A)) ? concrete_typed_array(A) : A
end
"""
    factory(a::Union{Nothing, <:AbstractEstimator, <:AbstractAlgorithm,
                     <:AbstractResult}, args...; kwargs...) -> a
    factory(a::AbstractVector{<:Union{Nothing, <:AbstractEstimator, <:AbstractAlgorithm,
                                      <:AbstractResult}}, args...; kwargs...) -> Vector

No-op factory function for constructing objects with a uniform interface.

Defining methods which dispatch on the first argument allows for a consistent factory interface across different types.

`factory` and [`port_opt_view`](@ref) are the two propagation mechanisms in this library.
They are duals: `factory` threads **runtime values** (prior moments, observation weights,
previous portfolio weights) down through a composed struct tree; `port_opt_view` threads an
**index selection** (a subset of assets or observations) down through the same tree.

The vector method is the **one forwarding contract** for every vector-valued propagation field: it applies `factory` to each element and forwards `args...` and `kwargs...` unchanged, so a family that admits a vector of estimators, algorithms, or results needs no method of its own. A family that needs more than the forward, such as a concrete element type ([`concrete_typed_array_if_abstract`](@ref)), defines its own more specific method.

# Arguments

  - `a`: Indicates no object should be constructed, or a vector whose elements are rebuilt one by one.
  - `args...`: Arbitrary positional arguments (ignored by the scalar method, forwarded by the vector method).
  - `kwargs...`: Arbitrary keyword arguments (ignored by the scalar method, forwarded by the vector method).

# Returns

  - `a`: The input unchanged.
  - `v::Vector`: The element-wise rebuilds, for the vector method.

# Examples

```jldoctest
julia> factory(nothing, 1, 2; x = 3)

julia> factory(MeanValue())
MeanValue
  w ┴ nothing
```

# Related

  - [`port_opt_view`](@ref)
  - [`AbstractEstimator`](@ref)
  - [`AbstractAlgorithm`](@ref)
  - [`AbstractResult`](@ref)
"""
function factory(a::Union{Nothing, <:AbstractEstimator, <:AbstractAlgorithm,
                          <:AbstractResult}, args...; kwargs...)
    return a
end
function factory(a::AbstractVector{<:Union{Nothing, <:AbstractEstimator,
                                           <:AbstractAlgorithm, <:AbstractResult}}, args...;
                 kwargs...)
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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve every **Deferred Quantity** held by `x` against prior result `pr`, returning a
struct of the same type whose deferred slots hold plain values.

This resolves the deferred state and **nothing else**. A slot left unstated stays `nothing`,
so whichever fallback the consumer already applies — `sel` on the factory path,
[`chol_sigma_selector`](@ref) and its siblings on the `JuMP` path — keeps working unchanged.
The two paths are separate: a `JuMP` model builder reads the risk measure's slots directly
and never calls [`factory`](@ref), so both entry points resolve.

This method is the identity, and it is the arm for a second argument that is not a prior
result: with no prior in hand nothing can be fitted, so the deferred state travels on.

Given a prior result the rule has two halves. **Container recursion is derived** from
[`deferred_slots`](@ref), so a type that only holds children needs no method at all. **A
type that resolves a quantity of its own defines a method**, which overrides the derived one.
Writing that half per type — rather than per field — is what lets slots that travel together
be resolved together: a deferred `sigma` supplies `chol` from the same fit, so the pair is
never mixed across two sources.

# Related

  - [`@propagatable`](@ref)
  - [`resolve_slot`](@ref)
  - [`deferred_slots`](@ref)
  - [`factory`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
resolve_deferred_quantities(x, ::Any) = x
# ---------------------------------------------------------------------------
# @propagatable — struct-definition macro for factory propagation
# ---------------------------------------------------------------------------

# --- private AST helpers ----------------------------------------------------

"""
    PROP_TAG_NAMES

The propagation tag set of [`@propagatable`](@ref), as data.

One entry per field tag. The recognition layer is derived from this tuple rather than
spelled out per tag: the macro names ([`PROP_TAG_MACRO_NAMES`](@ref)), the lookup
([`prop_tag`](@ref)), the gate ([`is_prop_tag_call`](@ref)), the peeler
([`peel_prop_tags`](@ref)) and the parser ([`propagatable_parse_body`](@ref)) all read it.
A new propagation channel is one row here, one branch in [`prop_tag_expr`](@ref), one entry
in [`PROP_TAG_CHANNELS`](@ref) and one stub macro; [`check_prop_tag_macros`](@ref) refuses
to load the module when a row lacks any of the three.

# Related

  - [`PROP_TAG_MACRO_NAMES`](@ref)
  - [`PROP_TAG_CHANNELS`](@ref)
  - [`prop_tag`](@ref)
  - [`prop_tag_expr`](@ref)
  - [`check_prop_tag_macros`](@ref)
  - [`@propagatable`](@ref)
"""
const PROP_TAG_NAMES = (:fprop, :vprop, :pprop, :cprop, :wprop)

"""
    PROP_TAG_MACRO_NAMES

The macro name of every tag of [`PROP_TAG_NAMES`](@ref), in the same order.

Derived from the tag names, so a row of the table carries no second spelling.
[`prop_tag`](@ref) matches a `:macrocall` head against this tuple.

# Related

  - [`PROP_TAG_NAMES`](@ref)
  - [`prop_tag`](@ref)
"""
const PROP_TAG_MACRO_NAMES = map(tag -> Symbol("@", tag), PROP_TAG_NAMES)

"""
    PROP_TAG_CHANNELS

The propagation channels of [`@propagatable`](@ref), with their tag precedence as data.

One entry per generated method. Each entry has two tuples of tags:

  - `gate`: the tags that make [`@propagatable`](@ref) emit the method at all.
  - `precedence`: the order in which a field's tags are consulted. The first tag of this
    tuple that the field carries decides the field's transform; the rest are ignored for
    that channel.

The `factory` channel prefers `@fprop` over `@wprop`. The `prior` channel prefers `@pprop`,
then `@cprop`, then `@wprop`, then `@fprop`, so `@pprop` wins over `@fprop` on one field
(ADR 0012). The precedence used to live in two hand-written `if`/`elseif` chains that no
comment linked; it is now read by [`prop_channel_pairs`](@ref) for every channel.

**A tag means what its channel says it means.** The `obs` channel reads the same `@wprop`
and `@fprop` tags as `factory` and gives them different transforms: `factory` **replaces** a
`@wprop` field with an incoming [`ObsWeights`](@ref) value, while `obs` **indexes** the value
already there. This is why [`prop_tag_expr`](@ref) takes the channel as well as the tag. It
also means a weights field opts into [`obs_weights_view`](@ref) by carrying `@wprop`, with no
second tag to write and no second tag to forget.

# Related

  - [`PROP_TAG_NAMES`](@ref)
  - [`prop_channel_pairs`](@ref)
  - [`prop_channel_active`](@ref)
  - [`@propagatable`](@ref)
"""
const PROP_TAG_CHANNELS = (factory = (gate = (:fprop, :wprop),
                                      precedence = (:fprop, :wprop)),
                           view = (gate = (:vprop,), precedence = (:vprop,)),
                           prior = (gate = (:pprop, :cprop),
                                    precedence = (:pprop, :cprop, :wprop, :fprop)),
                           obs = (gate = (:wprop,), precedence = (:wprop, :fprop)))

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the tag of [`PROP_TAG_NAMES`](@ref) that `x` names, or `nothing`.

`x` is the head of a `:macrocall` node. It is a bare `Symbol` in a struct body written by
hand, and a `GlobalRef` once another macro has expanded around it, so both spellings
resolve. A name outside [`PROP_TAG_MACRO_NAMES`](@ref) gives `nothing`, which is how the
callers tell a tag from any other macro; no tag falls through to another tag.

# Arguments

  - `x`: The first argument of a `:macrocall` expression.

# Returns

  - `tag::Symbol`: The tag name, without the `@`.
  - `nothing`: If `x` names no tag.

# Related

  - [`PROP_TAG_NAMES`](@ref)
  - [`is_prop_tag_call`](@ref)
  - [`peel_prop_tags`](@ref)
  - [`@propagatable`](@ref)
"""
function prop_tag(x)
    name = if x isa GlobalRef
        x.name
    elseif x isa Symbol
        x
    else
        return nothing
    end
    for (tag, macro_name) in zip(PROP_TAG_NAMES, PROP_TAG_MACRO_NAMES)
        if name === macro_name
            return tag
        end
    end
    return nothing
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if `x` is a macro call to any tag of [`PROP_TAG_NAMES`](@ref).

Used by [`peel_prop_tags`](@ref) and [`propagatable_parse_body`](@ref) to detect tagged
fields in a struct body.

# Arguments

  - `x`: Any expression appearing in a struct body.

# Related

  - [`prop_tag`](@ref)
  - [`peel_prop_tags`](@ref)
  - [`propagatable_parse_body`](@ref)
  - [`@propagatable`](@ref)
"""
function is_prop_tag_call(x)
    return x isa Expr && x.head == :macrocall && prop_tag(x.args[1]) !== nothing
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Peel any stack of tag macrocalls off a field expression, recording which tags were present.

Tags may be stacked in either order (`@pprop @fprop field`), which parses as nested
`:macrocall` nodes; this unwraps them all and returns the bare field expression. Each tag
is looked up with [`prop_tag`](@ref), so an untagged macro stops the peel and no tag is
reached by falling through the others.

# Arguments

  - `expr`: A field expression, with or without tag macrocalls around it.

# Returns

  - `tags::Set{Symbol}`: The tags of [`PROP_TAG_NAMES`](@ref) that `expr` carries.
  - `stripped`: The field expression with all tags removed.

# Related

  - [`prop_tag`](@ref)
  - [`is_prop_tag_call`](@ref)
  - [`propagatable_parse_body`](@ref)
"""
function peel_prop_tags(expr)
    tags = Set{Symbol}()
    while is_prop_tag_call(expr)
        push!(tags, prop_tag(expr.args[1]))
        expr = expr.args[end]
    end
    return tags, expr
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the expression that a tag substitutes for one field, inside a generated method.

This is the `name → field transform` half of the tag table: one branch per tag of
[`PROP_TAG_NAMES`](@ref), read by [`prop_channel_pairs`](@ref) for every channel. A tag of
the table with no branch here errors, so it cannot silently take another tag's transform.

# Arguments

  - `tag::Symbol`: A tag of [`PROP_TAG_NAMES`](@ref).
  - `fname::Symbol`: The field name, needed by `@pprop` to name the prior property.
  - `xf`: The expression that reads the field off the incoming struct.
  - `mod::Module`: The module that defines [`@propagatable`](@ref). Every emitted name is
    qualified against it, because the expansion is escaped into the caller.
  - `thread`: Extra positional arguments the channel threads before `args...`.

# Returns

  - `expr::Expr`: The value of the field in the generated constructor call.

# Related

  - [`PROP_TAG_CHANNELS`](@ref)
  - [`prop_channel_pairs`](@ref)
  - [`check_prop_tag_macros`](@ref)
  - [`@propagatable`](@ref)
"""
function prop_tag_expr(channel::Symbol, tag::Symbol, fname::Symbol, xf, mod::Module, thread)
    if channel === :obs
        # The observation channel reads the same two tags as `factory` and gives them
        # different transforms. `@wprop` holds the weights themselves, so the field is
        # INDEXED rather than replaced; `@fprop` holds a composed child, so the verb
        # recurses into it. Indexing preserves the `AbstractWeights` subtype, which a
        # `view` would not.
        if tag === :wprop
            return :($mod.nothing_scalar_array_getindex($xf, $(thread...)))
        end
        tag === :fprop && return :($mod.obs_weights_view($xf, $(thread...)))
    else
        if tag === :fprop
            return :($mod.factory_child($xf, $(thread...), args...; kwargs...))
        end
        if tag === :vprop
            return :($mod.port_opt_view($xf, $(thread...), args...))
        end
        if tag === :pprop
            return :($mod.sel($xf, getproperty(pr, $(QuoteNode(fname)))))
        end
        if tag === :cprop
            return :($mod.sel($xf, $mod._ctx(args...)))
        end
        tag === :wprop && return :($mod._wprop($xf, args...; kwargs...))
    end
    return error("@propagatable: tag `@$(tag)` has no field transform in channel " *
                 "`:$(channel)`. Add a branch to `prop_tag_expr`.")
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if a channel of [`PROP_TAG_CHANNELS`](@ref) must emit a method.

A channel is active when at least one field carries a tag of the channel's `gate`.

# Arguments

  - `channel::Symbol`: A channel name of [`PROP_TAG_CHANNELS`](@ref).
  - `tagged::AbstractDict`: Tag name to the field names that carry it, from
    [`propagatable_parse_body`](@ref).

# Related

  - [`PROP_TAG_CHANNELS`](@ref)
  - [`prop_channel_pairs`](@ref)
  - [`@propagatable`](@ref)
"""
function prop_channel_active(channel::Symbol, tagged::AbstractDict)
    return any(tag -> !isempty(tagged[tag]), getproperty(PROP_TAG_CHANNELS, channel).gate)
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the keyword pairs of the constructor call that one channel generates.

Every declared field gets one pair, in declaration order. The field's tags are consulted in
the channel's `precedence` order; the first match gives the value through
[`prop_tag_expr`](@ref), and a field carrying no tag of the channel is passed through
unchanged.

# Arguments

  - `channel::Symbol`: A channel name of [`PROP_TAG_CHANNELS`](@ref).
  - `tagged::AbstractDict`: Tag name to the field names that carry it.
  - `all_fields::AbstractVector{Symbol}`: Every declared field, in declaration order.
  - `obj::Symbol`: The struct the generated method reads the fields off.
  - `mod::Module`: The module that defines [`@propagatable`](@ref).
  - `thread`: Extra positional arguments the channel threads before `args...`.

# Returns

  - `pairs::Vector{Any}`: One `Expr(:kw, field, value)` per declared field.

# Related

  - [`PROP_TAG_CHANNELS`](@ref)
  - [`prop_tag_expr`](@ref)
  - [`@propagatable`](@ref)
"""
function prop_channel_pairs(channel::Symbol, tagged::AbstractDict,
                            all_fields::AbstractVector{Symbol}, obj::Symbol, mod::Module,
                            thread)
    precedence = getproperty(PROP_TAG_CHANNELS, channel).precedence
    pairs      = Any[]
    for fname in all_fields
        xf  = Expr(:., obj, QuoteNode(fname))
        idx = findfirst(tag -> fname in tagged[tag], precedence)
        val = isnothing(idx) ? xf : prop_tag_expr(channel, precedence[idx], fname, xf, mod, thread)
        push!(pairs, Expr(:kw, fname, val))
    end
    return pairs
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

Walk a struct body, collecting the tagged field names (and all field names) and stripping
the tags from the body.

Handles bare tagged fields (`@fprop field`, …), stacked tags
(`@pprop @fprop field`, in any order), and docstring-prefixed forms
(`"doc" \\n @fprop field`). Non-field nodes (line numbers, inner constructors) are carried
through unchanged. The tags are the rows of [`PROP_TAG_NAMES`](@ref), so a new tag needs no
change here.

# Arguments

  - `body::Expr`: The `:block` expression forming the struct body.

# Returns

  - `tagged::Dict{Symbol, Vector{Symbol}}`: One entry per tag of [`PROP_TAG_NAMES`](@ref),
    holding the names of the fields that carry it, in declaration order.
  - `all_fields::Vector{Symbol}`: Names of every declared field (tagged or not).
  - `new_body::Expr`: The struct body with all tags stripped.

# Related

  - [`PROP_TAG_NAMES`](@ref)
  - [`peel_prop_tags`](@ref)
  - [`is_doc_macro`](@ref)
  - [`extract_field_name`](@ref)
  - [`try_field_name`](@ref)
  - [`@propagatable`](@ref)
"""
function propagatable_parse_body(body)
    tagged     = Dict{Symbol, Vector{Symbol}}(tag => Symbol[] for tag in PROP_TAG_NAMES)
    all_fields = Symbol[]
    new_args   = Any[]
    function _record!(fname, tags)
        for tag in tags
            push!(tagged[tag], fname)
        end
        push!(all_fields, fname)
        return nothing
    end
    for arg in body.args
        if arg isa Expr && arg.head == :macrocall && is_doc_macro(arg.args[1])
            # Core.@doc "doc" (field or tagged field)
            inner = arg.args[end]
            tags, stripped = peel_prop_tags(inner)
            if !isempty(tags)
                _record!(extract_field_name(stripped), tags)
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
            # Bare tagged field (tags may be stacked) — no docstring
            tags, stripped = peel_prop_tags(arg)
            _record!(extract_field_name(stripped), tags)
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
    return tagged, all_fields, Expr(:block, new_args...)
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
$(DocStringExtensions.TYPEDSIGNATURES)

Check that every tag of [`PROP_TAG_NAMES`](@ref) is complete.

A tag row is complete when it has a stub macro, a channel of [`PROP_TAG_CHANNELS`](@ref)
that names it, and a field transform in [`prop_tag_expr`](@ref) **for every channel that
names it**. A row that lacks one of the three is a tag that parses but never propagates,
which is the failure the table exists to stop. The per-channel probe is what catches a tag
added to a second channel without a transform there, now that a tag means what its channel
says it means. All the violations are collected and reported together.

Runs once at the end of the module. Throws an
[`ArgumentError`](https://docs.julialang.org/en/v1/base/base/#Core.ArgumentError) listing
every violation, so the package refuses to precompile rather than shipping a dead tag.

# Related

  - [`PROP_TAG_NAMES`](@ref)
  - [`PROP_TAG_CHANNELS`](@ref)
  - [`prop_tag_expr`](@ref)
  - [`check_propagatable_contracts`](@ref)
  - [`@propagatable`](@ref)
"""
function check_prop_tag_macros()
    violations = String[]
    for (tag, macro_name) in zip(PROP_TAG_NAMES, PROP_TAG_MACRO_NAMES)
        if !isdefined(@__MODULE__, macro_name)
            push!(violations, "`:$(tag)` declares no `$(macro_name)` stub macro.")
        end
        if !any(tag in channel.precedence for channel in PROP_TAG_CHANNELS)
            push!(violations, "`:$(tag)` appears in no channel of `PROP_TAG_CHANNELS`.")
        end
        for channel in keys(PROP_TAG_CHANNELS)
            if !(tag in getproperty(PROP_TAG_CHANNELS, channel).precedence)
                continue
            end
            try
                prop_tag_expr(channel, tag, :probe, :probe, @__MODULE__, ())
            catch
                push!(violations,
                      "`:$(tag)` has no field transform in channel `:$(channel)`.")
            end
        end
    end
    if !isempty(violations)
        throw(ArgumentError("Incomplete `PROP_TAG_NAMES` rows:\n" *
                            join("  - " .* violations, "\n")))
    end
    return nothing
end

# ---------------------------------------------------------------------------
# @propagatable — the declaration-time contract check
# ---------------------------------------------------------------------------

"""
    PROPAGATABLE_CONTRACTS

Every type declared with [`@propagatable`](@ref), paired with its `@pprop`-tagged field names.

One entry is appended by the macro itself, immediately after the struct it declares, so the
list is complete by the time the module finishes loading — including types declared in
external packages. [`check_propagatable_contracts`](@ref) is what reads it.

# Related

  - [`@propagatable`](@ref)
  - [`propagatable_register!`](@ref)
  - [`check_propagatable_contracts`](@ref)
"""
const PROPAGATABLE_CONTRACTS = Vector{Tuple{Type, Tuple{Vararg{Symbol}}}}()
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Record type `T` and its `@pprop`-tagged field names in [`PROPAGATABLE_CONTRACTS`](@ref).

Called by [`@propagatable`](@ref) at the declaration itself. It only records: the outer
keyword constructor is written *below* the struct, so it does not exist yet and cannot be
checked here. [`check_propagatable_contracts`](@ref) does the checking once the module is
complete.

# Related

  - [`@propagatable`](@ref)
  - [`PROPAGATABLE_CONTRACTS`](@ref)
  - [`check_propagatable_contracts`](@ref)
"""
function propagatable_register!(@nospecialize(T::Type), pprops::Tuple{Vararg{Symbol}})
    push!(PROPAGATABLE_CONTRACTS, (T, pprops))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the keyword names accepted by the outer constructors of `T`, unioned over its methods.

A `kwargs...` slurp is dropped rather than counted. A slurp accepts `field = value` and then
discards it, which is the silent failure this check exists to catch, so it must not satisfy
the contract.

# Related

  - [`check_propagatable_contracts`](@ref)
"""
function propagatable_keywords(@nospecialize(T::Type))
    kws = Symbol[]
    for m in methods(T)
        append!(kws, Base.kwarg_decl(m))
    end
    return filter!(k -> !endswith(string(k), "..."), unique!(kws))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the broken clauses of one type's [`@propagatable`](@ref) contract, as messages.

Two clauses are checked, and both are properties of the code the macro emits:

  - **Every field name is a keyword of the outer constructor.** Each generated method rebuilds
    the struct with `StructName(; field = …)` over *all* fields, tagged or not, so one field
    that the keyword constructor does not name is a `MethodError` at the first
    [`factory`](@ref) or [`port_opt_view`](@ref) call.
  - **Every `@pprop` field is a property of a prior result.** The generated
    `factory(x, pr::AbstractPriorResult, args...)` reads `getproperty(pr, :field)`, so a name
    absent from [`prior_result_property_pool`](@ref) throws when a prior is threaded.

The messages carry a [`suggest_declared_key`](@ref) suggestion, so a transposed or mistyped
field name names its intended neighbour.

# Related

  - [`check_propagatable_contracts`](@ref)
  - [`propagatable_keywords`](@ref)
  - [`prior_result_property_pool`](@ref)
"""
function propagatable_contract_violations(@nospecialize(T::Type), pprops, pool)
    msgs = String[]
    kws = propagatable_keywords(T)
    for f in fieldnames(T)
        if !(f in kws)
            push!(msgs,
                  "`$(nameof(T))`: field `$(f)` is not a keyword of its outer constructor" *
                  suggest_declared_key(f, kws) *
                  ".")
        end
    end
    for f in pprops
        if !(f in pool)
            push!(msgs,
                  "`$(nameof(T))`: `@pprop` field `$(f)` is not a property of a " *
                  "prior result" *
                  suggest_declared_key(f, pool) *
                  ".")
        end
    end
    return msgs
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
`factory(x, pr::AbstractPriorResult, args...)` is generated. It first calls
[`resolve_deferred_quantities`](@ref) on `x` — the identity unless the type declares a
method — so every slot holding a **Deferred Quantity** becomes a plain value before
selection runs. It then selects `@pprop`/`@cprop` fields as above and threads
`@fprop`-only fields with `pr` (`factory_child(getfield(x, :f), pr, args...)`);
a field tagged both `@pprop` and `@fprop` is prior-selected in this method (`@pprop` wins).
Because this method is more specific than the general `factory(x, args...)`, it is chosen
whenever a prior is passed.

Untagged fields pass through unchanged in every method, regardless of type —
tagging is explicit and opt-in. The tags are independent and the relevant field sets
genuinely diverge. `@pprop` and `@cprop` are mutually exclusive on one field (a value comes
from exactly one source); legal stacks are `@pprop @fprop` (sub-estimator) and
`@pprop @wprop` (weights slot). See ADR 0010 and 0012.

Two consequences of the emitted code are **contracts on the declaration**, and both are checked
where the struct is written rather than at the first call:

  - Every generated method rebuilds the struct as `StructName(; field = …)` over *all* fields,
    tagged or not, so **every field name must also be a keyword of the outer constructor**. A
    `kwargs...` slurp does not satisfy this: it accepts the keyword and then discards it.
  - The prior method reads `getproperty(pr, :field)`, so **every `@pprop` field name must be a
    property of a prior result** (see [`prior_result_property_pool`](@ref)).

The macro registers each declaration in [`PROPAGATABLE_CONTRACTS`](@ref), and
[`check_propagatable_contracts`](@ref) checks the whole registry once the module is complete.
A mistyped field name therefore fails at precompilation with a
[`suggest_declared_key`](@ref) suggestion, rather than surfacing as a `MethodError` at the
first [`factory`](@ref) call. [`forward_prior`](@ref) leans on the same contract (ADR 0046).

The tag set itself is data. [`PROP_TAG_NAMES`](@ref) holds the rows,
[`prop_tag_expr`](@ref) holds each tag's field transform, and [`PROP_TAG_CHANNELS`](@ref)
holds each channel's gate and tag precedence, so a new propagation channel is a table row
rather than an edit at seven sites (ADR 0061).

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

`@wprop` drives two channels at once, and they do different things to the same field:
`factory` **replaces** it with an incoming [`ObsWeights`](@ref) value, while
[`obs_weights_view`](@ref) **indexes** the value already there, to a set of observations.
A weights field therefore needs no second tag to join the observation-axis view.

The generated `factory`/`port_opt_view`/`obs_weights_view` methods are added to the
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

    tagged, all_fields, new_body = propagatable_parse_body(body)

    new_struct = Expr(:struct, struct_node.args[1], type_head, new_body)
    chain      = rebuild(new_struct)

    # Every name the expansion emits is qualified against the module that *defines* the
    # macro, because the emitted block is escaped and a bare name resolves where the struct
    # is declared. `factory` is exported, which does not help: `using PortfolioOptimisers`
    # binds it implicitly, so `function factory(...)` in the caller declares a **new**
    # function of the caller's own, and `PortfolioOptimisers.factory` never gains the
    # method. That failure is silent -- the declaration compiles, the contract registers,
    # and the type simply never joins the propagation chain. `@__MODULE__` in a macro body
    # is the defining module, and interpolating the module object needs no binding in the
    # caller at all (ADR 0002, decision 4). `prop_tag_expr` qualifies the names it emits the
    # same way.
    POMOD = @__MODULE__
    _factory = :($POMOD.factory)
    _port_opt_view = :($POMOD.port_opt_view)
    _obs_weights_view = :($POMOD.obs_weights_view)
    _resolve_fn = :($POMOD.resolve_deferred_quantities)
    _prior_result = :($POMOD.AbstractPriorResult)
    _register_fn = :($POMOD.propagatable_register!)

    # Every channel below reads its tag precedence off `PROP_TAG_CHANNELS`. The emission
    # differs only in the method head and in the arguments the channel threads.

    # --- factory propagation (@fprop recurses sub-estimators, @wprop replaces weights) ---
    factory_body = if prop_channel_active(:factory, tagged)
        Expr(:call, struct_name,
             Expr(:parameters,
                  prop_channel_pairs(:factory, tagged, all_fields, :x, POMOD, ())...))
    else
        :x
    end
    factory_def = quote
        function $_factory(x::$struct_name, args...; kwargs...)
            return $factory_body
        end
    end

    defs = Any[factory_def]

    # --- view propagation (@vprop) — emit only when a field opts in ---
    if prop_channel_active(:view, tagged)
        view_body = Expr(:call, struct_name,
                         Expr(:parameters,
                              prop_channel_pairs(:view, tagged, all_fields, :x, POMOD,
                                                 (:i,))...))
        view_def = quote
            function $_port_opt_view(x::$struct_name, i, args...)
                return $view_body
            end
        end
        push!(defs, view_def)
    end

    # --- observation-weights view (@wprop) — emit only when a weights field exists ---
    # The same `@wprop` tag the factory channel reads, given the other transform: the
    # weights field is indexed to `i` rather than replaced, and `@fprop` children recurse.
    if prop_channel_active(:obs, tagged)
        obs_body = Expr(:call, struct_name,
                        Expr(:parameters,
                             prop_channel_pairs(:obs, tagged, all_fields, :x, POMOD,
                                                (:i,))...))
        obs_def = quote
            function $_obs_weights_view(x::$struct_name, i)
                return $obs_body
            end
        end
        push!(defs, obs_def)
    end

    # --- prior/context selection (@pprop / @cprop) — emit only when a field opts in ---
    if prop_channel_active(:prior, tagged)
        # Every field is read off the Deferred-Quantity-resolved struct, not the argument.
        prior_body = Expr(:call, struct_name,
                          Expr(:parameters,
                               prop_channel_pairs(:prior, tagged, all_fields, :xr, POMOD,
                                                  (:pr,))...))
        prior_def = quote
            function $_factory(x::$struct_name, pr::$_prior_result, args...; kwargs...)
                xr = $_resolve_fn(x, pr)
                return $prior_body
            end
        end
        push!(defs, prior_def)
    end

    pprop_tuple = Expr(:tuple, QuoteNode.(tagged[:pprop])...)

    return esc(quote
                   Base.@__doc__ $chain
                   $(defs...)
                   $_register_fn($struct_name, $pprop_tuple)
               end)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Check the [`@propagatable`](@ref) contract of every type in [`PROPAGATABLE_CONTRACTS`](@ref).

Runs once at the end of the module, so the contract behind 231 generated methods is enforced
where the structs are *declared* rather than at the first [`factory`](@ref) call. The
violations of every type are collected and reported together, because a run that stops at the
first one hides the rest. See [`propagatable_contract_violations`](@ref) for the two clauses.

Throws an [`ArgumentError`](https://docs.julialang.org/en/v1/base/base/#Core.ArgumentError)
listing every violation; the package refuses to precompile rather than shipping a type whose
generated methods throw on first use. A package that declares its own `@propagatable` types
calls this at the end of its own module to get the same guarantee.

# Related

  - [`@propagatable`](@ref)
  - [`PROPAGATABLE_CONTRACTS`](@ref)
  - [`propagatable_contract_violations`](@ref)
  - [`@windowed_estimator`](@ref)
"""
function check_propagatable_contracts()
    pool = prior_result_property_pool()
    msgs = String[]
    for (T, pprops) in PROPAGATABLE_CONTRACTS
        append!(msgs, propagatable_contract_violations(T, pprops, pool))
    end
    if !isempty(msgs)
        throw(ArgumentError("@propagatable: $(length(msgs)) broken contract(s). The generated `factory`/`port_opt_view` methods rebuild the struct by keyword, so each of these throws at its first call:\n  - " *
                            join(msgs, "\n  - ")))
    end
    return nothing
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
  - [`StdValue`](@ref)
  - [`VarValue`](@ref)
  - [`SumValue`](@ref)
  - [`ProdValue`](@ref)
  - [`ModeValue`](@ref)
  - [`StandardisedValue`](@ref)
  - [`Num_VecToScaM`](@ref)
  - [`CentralityConstraint`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
abstract type VectorToScalarMeasure <: AbstractAlgorithm end
"""
    const Num_VecToScaM = Union{<:Number, <:VectorToScalarMeasure, <:Function}

Union type representing a numeric value, a `VectorToScalarMeasure`, or a `Function`.

This type lets functions and fields accept all three, so a caller can give a fixed number, an object that implements the `VectorToScalarMeasure` interface, or a plain reduction function. [`vec_to_real_measure`](@ref) returns a `Number` unchanged, dispatches a `VectorToScalarMeasure` to its reduction, and applies a `Function` to the vector.

# Related

  - [`VectorToScalarMeasure`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
const Num_VecToScaM = Union{<:Number, <:VectorToScalarMeasure, <:Function}
"""
$(DocStringExtensions.TYPEDEF)

Algorithm for reducing a vector of real values to its minimum.

# Constructors

    MinValue() -> MinValue

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

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@wprop`-tagged field is automatically propagated:

  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

## Observation weight parameters

When [`obs_weights_view`](@ref) is called on this type, the following fields are automatically indexed to the selected observations:

  - `w`: Indexed to the selected observations via [`obs_weights_view`](@ref).

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
  - [`obs_weights_view`](@ref)
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

# Details

  - The weighted case is not an order statistic. `Statistics.median(val, w)` is the weighted 0.5-quantile, so it **interpolates** between the two values that bracket half the weight mass. On `[1.0, 2.0, 3.0, 4.0]` with weights `[0.1, 0.2, 0.3, 0.4]` the result is `2.8333`, which is not an element of the input. The unweighted case, `w = nothing`, is the ordinary median and gives `2.5`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MedianValue(;
        w::Option{<:ObsWeights} = nothing,
    ) -> MedianValue

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@wprop`-tagged field is automatically propagated:

  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

## Observation weight parameters

When [`obs_weights_view`](@ref) is called on this type, the following fields are automatically indexed to the selected observations:

  - `w`: Indexed to the selected observations via [`obs_weights_view`](@ref).

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
  - [`obs_weights_view`](@ref)
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

# Constructors

    MaxValue() -> MaxValue

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

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@wprop`-tagged field is automatically propagated:

  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

## Observation weight parameters

When [`obs_weights_view`](@ref) is called on this type, the following fields are automatically indexed to the selected observations:

  - `w`: Indexed to the selected observations via [`obs_weights_view`](@ref).

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
  - [`obs_weights_view`](@ref)
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

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@wprop`-tagged field is automatically propagated:

  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

## Observation weight parameters

When [`obs_weights_view`](@ref) is called on this type, the following fields are automatically indexed to the selected observations:

  - `w`: Indexed to the selected observations via [`obs_weights_view`](@ref).

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
  - [`obs_weights_view`](@ref)
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

# Constructors

    SumValue() -> SumValue

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

# Constructors

    ProdValue() -> ProdValue

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

# Constructors

    ModeValue() -> ModeValue

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

# Mathematical definition

```math
\\begin{align}
z &= \\frac{\\hat{\\mu}}{\\tilde{\\sigma}}\\,, \\\\
\\tilde{\\sigma} &= \\begin{cases} \\sqrt{\\varepsilon} & \\hat{\\sigma} = 0 \\\\ \\hat{\\sigma} & \\hat{\\sigma} \\neq 0 \\end{cases}\\,.
\\end{align}
```

Where:

  - ``z``: Standardised value.
  - ``\\hat{\\mu}``: The value computed by `mv`.
  - ``\\hat{\\sigma}``: The value computed by `sv`, taken about ``\\hat{\\mu}``.
  - ``\\tilde{\\sigma}``: The guarded denominator.
  - ``\\varepsilon``: Machine epsilon of the element type of ``\\hat{\\sigma}``.

# Details

  - `sv` receives ``\\hat{\\mu}`` as its `mean` keyword, so the standard deviation is always taken about the mean that `mv` produced. Weighting `mv` without weighting `sv` therefore changes the denominator too.
  - The guard fires on an **exact** zero only, not on a small denominator. On the constant vector `[2.0, 2.0, 2.0]` the result is `1.342e8`, which is `2 / sqrt(eps(Float64))`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    StandardisedValue(;
        mv::MeanValue = MeanValue(),
        sv::StdValue = StdValue(),
    ) -> StandardisedValue

Keywords correspond to the struct's fields.

## Propagated parameters

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
    vec_to_real_measure(
        measure::Num_VecToScaM,
        val::Union{<:VecNum, NTuple{N, <:Number} where {N}};
        kwargs...
    ) -> Number

Reduce a vector of real values to a single real value using a specified measure.

`vec_to_real_measure` applies a reduction algorithm (such as minimum, mean, median, or maximum) to a vector of real numbers, as specified by the concrete subtype of [`VectorToScalarMeasure`](@ref). This is used in constraint generation and centrality-based portfolio constraints to aggregate asset-level metrics.

# Arguments

  - `measure`: One of three things.

      + `::VectorToScalarMeasure`: The reduction to apply to `val`.
      + `::Number`: The value to return, whatever `val` holds.
      + `::Function`: Applied to `val` directly, as `measure(val)`.

  - `val`: A vector or tuple of real values to be reduced. It is ignored when `measure` is a `Number`.

  - `kwargs...`: Forwarded to the underlying reduction. Only the [`StdValue`](@ref) and [`VarValue`](@ref) reductions read them.

# Returns

  - `score::Number`: Computed value according to `measure`.

# Details

  - A tuple is accepted wherever a vector is. The weighted reductions `collect` it first, because `Statistics` needs an `AbstractVector` beside its weights.

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

export factory, concrete_typed_array, MinValue, MeanValue, MedianValue, MaxValue,
       StandardisedValue, StdValue, VarValue, SumValue, ProdValue, ModeValue
public @propagatable, @fprop, @vprop, @pprop, @cprop, @wprop, @forward_properties,
       traverse_concrete_subtypes, factory_child, port_opt_view, obs_weights_view
