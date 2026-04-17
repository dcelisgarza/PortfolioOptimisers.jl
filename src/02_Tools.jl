"""
    ⊗(A::ArrNum, B::ArrNum) -> Matrix{promote_type(eltype(A), eltype(B))}

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
    ⊙(A::ArrNum, B::ArrNum) -> Matrix{promote_type(eltype(A), eltype(B))}
    ⊙(A::ArrNum, B) -> Vector{promote_type(eltype(A), eltype(B))}
    ⊙(A, B::ArrNum) -> Vector{promote_type(eltype(A), eltype(B))}
    ⊙(A, B) -> promote_type(eltype(A), eltype(B))

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
    ⊘(A::ArrNum, B::ArrNum) -> Matrix{promote_type(eltype(A), eltype(B))}
    ⊘(A::ArrNum, B) -> Vector{promote_type(eltype(A), eltype(B))}
    ⊘(A, B::ArrNum) -> Vector{promote_type(eltype(A), eltype(B))}
    ⊘(A, B) -> promote_type(eltype(A), eltype(B))

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
    ⊕(A::ArrNum, B::ArrNum) -> Matrix{promote_type(eltype(A), eltype(B))}
    ⊕(A::ArrNum, B) -> Vector{promote_type(eltype(A), eltype(B))}
    ⊕(A, B::ArrNum) -> Vector{promote_type(eltype(A), eltype(B))}
    ⊕(A, B) -> promote_type(eltype(A), eltype(B))

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
    ⊖(A::ArrNum, B::ArrNum) -> Matrix{promote_type(eltype(A), eltype(B))}
    ⊖(A::ArrNum, B) -> Vector{promote_type(eltype(A), eltype(B))}
    ⊖(A, B::ArrNum) -> Vector{promote_type(eltype(A), eltype(B))}
    ⊖(A, B) -> promote_type(eltype(A), eltype(B))

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
                 <:DynamicAbstractWeights},
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

      + `::Union{Nothing, <:Number, <:Pair, <:VecPair, <:Dict, <:AbstractEstimatorValueAlgorithm, <:DynamicAbstractWeights}`: Returns `x` unchanged.
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
"""
function nothing_scalar_array_view(x::Union{Nothing, <:Number, <:Pair, <:VecPair, <:Dict,
                                            <:AbstractEstimatorValueAlgorithm,
                                            <:DynamicAbstractWeights}, ::Any)
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
    get_window(window::Option{<:Colon}, args...) -> Option{<:Colon}
    get_window(window::Integer, X::MatNum, dims::Int = 1) -> VecInt
    get_window(window::Integer, X::VecNum, args...) -> VecInt
    get_window(window::VecInt, args...) -> VecInt

Get the observation window index range for a data array.

# Arguments

  - $(arg_dict[:window])
      + `::Option{<:Colon}`: Returns the argument.
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

Utility for safely viewing or indexing into possibly `nothing` or array values with two indices.

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

Utility for safely viewing or indexing into possibly `nothing` or array values with two indices.

  - If `x` is `nothing`, returns `nothing`.
  - Otherwise, returns `view(x, i, j)`.

# Arguments

  - `x`: Input value.
  - `i`, `j`: Indices to view.

# Returns

  - The corresponding matrix index or `nothing`.

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

# Related

  - [`factory`](@ref)
  - [`AbstractEstimator`](@ref)
  - [`AbstractAlgorithm`](@ref)
  - [`AbstractResult`](@ref)
"""
function factory(a::Union{Nothing, <:AbstractEstimator, <:AbstractAlgorithm,
                          <:AbstractResult}, args...; kwargs...)
    return a
end
"""
$(DocStringExtensions.TYPEDEF)

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
@concrete struct MeanValue <: VectorToScalarMeasure
    "$(field_dict[:oow])"
    w
    function MeanValue(w::Option{<:ObsWeights})
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(w)}(w)
    end
end
function MeanValue(; w::Option{<:ObsWeights} = nothing)
    return MeanValue(w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Construct a `MeanValue` instance with observation weights `w`.

# Arguments

  - `mv`: Instance to update.
  - $(arg_dict[:ow])

# Returns

  - `mv::MeanValue`: A new `MeanValue` with observation weights `w`.

# Examples

```jldoctest
julia> factory(MeanValue(), StatsBase.Weights([1.2, 3.4, 0.7]))
MeanValue
  w ┴ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [1.2, 3.4, 0.7]
```

# Related

  - [`MeanValue`](@ref)
  - [`factory`](@ref)
"""
function factory(::MeanValue, w::ObsWeights)
    return MeanValue(; w = w)
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
@concrete struct MedianValue <: VectorToScalarMeasure
    "$(field_dict[:oow])"
    w
    function MedianValue(w::Option{<:ObsWeights})
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(w)}(w)
    end
end
function MedianValue(; w::Option{<:ObsWeights} = nothing)
    return MedianValue(w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Constructs a `MedianValue` instance with observation weights `w`.

# Arguments

  - `mv`: Instance to update.
  - $(arg_dict[:ow])

# Returns

  - `mdv::MedianValue`: A new `MedianValue` with observation weights `w`.

# Examples

```jldoctest
julia> factory(MedianValue(), StatsBase.Weights([1.2, 3.4, 0.7]))
MedianValue
  w ┴ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [1.2, 3.4, 0.7]
```

# Related

  - [`MedianValue`](@ref)
  - [`factory`](@ref)
"""
function factory(::MedianValue, w::ObsWeights)
    return MedianValue(; w = w)
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
@concrete struct StdValue <: VectorToScalarMeasure
    "$(field_dict[:oow])"
    w
    "$(field_dict[:corrected])"
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
$(DocStringExtensions.TYPEDSIGNATURES)

Constructs a `StdValue` instance with observation weights `w`.

# Arguments

  - `sv`: Instance to update.
  - $(arg_dict[:ow])

# Returns

  - `sv::StdValue`: A new `StdValue` with observation weights `w`.

# Examples

```jldoctest
julia> factory(StdValue(), StatsBase.Weights([1.2, 3.4, 0.7]))
StdValue
          w ┼ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [1.2, 3.4, 0.7]
  corrected ┴ Bool: true
```

# Related

  - [`StdValue`](@ref)
  - [`factory`](@ref)
"""
function factory(sv::StdValue, w::ObsWeights)
    return StdValue(; w = w, corrected = sv.corrected)
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
@concrete struct VarValue <: VectorToScalarMeasure
    "$(field_dict[:oow])"
    w
    "Indicates whether to use Bessel's correction (`true` for sample standard deviation, `false` for population)."
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
$(DocStringExtensions.TYPEDSIGNATURES)

Constructs a `VarValue` instance with observation weights `w`.

# Arguments

  - `vv`: Instance to update.
  - $(arg_dict[:ow])

# Returns

  - `vv::VarValue`: A new `VarValue` with observation weights `w`.

# Examples

```jldoctest
julia> factory(VarValue(), StatsBase.Weights([1.2, 3.4, 0.7]))
VarValue
          w ┼ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [1.2, 3.4, 0.7]
  corrected ┴ Bool: true
```

# Related

  - [`VarValue`](@ref)
  - [`factory`](@ref)
"""
function factory(vv::VarValue, w::ObsWeights)
    return VarValue(; w = w, corrected = vv.corrected)
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
"""
@concrete struct StandardisedValue <: VectorToScalarMeasure
    "The mean value measure used for the numerator."
    mv
    "The standard deviation measure used for the denominator."
    sv
    function StandardisedValue(mv::MeanValue, sv::StdValue)
        return new{typeof(mv), typeof(sv)}(mv, sv)
    end
end
function StandardisedValue(; mv::MeanValue = MeanValue(), sv::StdValue = StdValue())
    return StandardisedValue(mv, sv)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Construct a `StandardisedValue` instance with observation weights `w` for both `mv` and `sv`.

# Arguments

  - `msv`: Instance to update.
  - $(arg_dict[:ow])

# Returns

  - `msv::StandardisedValue`: A new `StandardisedValue` with observation weights `w` applied to both `mv` and `sv`.

# Examples

```jldoctest
julia> factory(StandardisedValue(), StatsBase.Weights([1.2, 3.4, 0.7]))
StandardisedValue
  mv ┼ MeanValue
     │   w ┴ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [1.2, 3.4, 0.7]
  sv ┼ StdValue
     │           w ┼ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [1.2, 3.4, 0.7]
     │   corrected ┴ Bool: true
```

# Related

  - [`StandardisedValue`](@ref)
  - [`MeanValue`](@ref)
  - [`StdValue`](@ref)
  - [`factory`](@ref)
"""
function factory(msv::StandardisedValue, w::ObsWeights)
    return StandardisedValue(; mv = factory(msv.mv, w), sv = factory(msv.sv, w))
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
function vec_to_real_measure(f::Function, val::VecNum; kwargs...)
    return f(val)
end

export factory, traverse_concrete_subtypes, concrete_typed_array, MinValue, MeanValue,
       MedianValue, MaxValue, StandardisedValue, StdValue, VarValue, SumValue, ProdValue,
       ModeValue
