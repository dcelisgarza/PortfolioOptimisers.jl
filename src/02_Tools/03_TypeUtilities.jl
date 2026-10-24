"""
    traverse_concrete_subtypes(t, ctarr::Option{<:AbstractVector} = nothing) -> AbstractVector

Collect every struct type below the abstract type `t` into `ctarr`, through a recursion over the subtypes of `t`.

A **struct type** is not the same as a concrete type. `InteractiveUtils.subtypes` gives a parametric struct as its `UnionAll`, and a `UnionAll` is not concrete. The function collects a parametric struct under its bare name, and `isconcretetype` is `false` for it. Every entry satisfies `isstructtype`. A caller that needs concrete types must set the parameters.

# Algorithm

 1. When `ctarr` is `nothing`, make it an empty `Vector{Any}`. The recursion passes the same array down, so one array collects every branch.
 2. Read `sts`, the direct subtypes of `t`, with `InteractiveUtils.subtypes`.
 3. For each subtype `st` of `sts`, take one of two branches:
     1. `st` is not a struct type, so it is a further abstract type: call this function again on `st` with the same `ctarr`.
     2. `st` is a struct type: push `st` onto `ctarr`. The test is `isstructtype` and not `isconcretetype`, so the function collects a parametric struct.
 4. Return `ctarr`.

The recursion goes through the whole tree below `t`. It opens an abstract type at any depth and never collects it. The order of `ctarr` is the depth-first order of the tree, which follows the order that `InteractiveUtils.subtypes` gives.

# Arguments

  - `t`: The abstract type whose subtypes the function reads.
  - `ctarr`: Array that collects the struct types. If `nothing`, the function makes a new empty array.

# Returns

  - `types::Vector{Any}`: An array holding every struct type that is a subtype, direct or indirect, of `t`. A parametric struct appears as its `UnionAll`.

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

Copy of `A` whose element type is the union of the types of its elements.

An array with an abstract element type, such as `Any`, costs a dynamic dispatch at each use of an element. The compiler can split a small union, so the copy can make later code faster.

# Algorithm

 1. Read the concrete type of every element of `A` with `typeof.(A)`.
 2. Build the element type `Union{typeof.(A)...}`, the union of only those types. A type that no element has is not in the union.
 3. Splat `A` into a vector of that element type, which flattens `A` to one dimension.
 4. Reshape the vector to `size(A)`, and return it.

The function copies the elements into a new array, and each element keeps its type. The union comes from the values, so it is only as narrow as the contents of the array. For example, an `Any` array whose elements are all `Int64` becomes an array of `Int64`.

# Arguments

  - `A`: The input array.

# Returns

  - `A_new::Array{Union{...}}`: A new array with the shape of `A`. Its element type is the union of the types of the elements of `A`.

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

The generic vector methods of [`factory`](@ref) and [`port_opt_view`](@ref) rebuild a vector field one element at a time. A comprehension over a vector whose elements have different types can infer an abstract element type. Then each later use of an element costs a dynamic dispatch. A family that needs a narrow element type calls this function on the rebuilt vector.

# Algorithm

 1. Test `eltype(A)` with `isabstracttype`.
 2. The element type is abstract: return [`concrete_typed_array`](@ref) of `A`, which copies.
 3. The element type is not abstract: return `A`, which copies nothing. A concrete type, a `Union` and a `UnionAll` such as `Vector` all take this branch, because `isabstracttype` is `false` for each of them.

# Arguments

  - `A`: The rebuilt array.

# Returns

  - `A`: With no change if `eltype(A)` is not abstract, else [`concrete_typed_array`](@ref) of `A`.

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
    float_if_integer(::Type{T}) -> Type

Numeric type that holds the values of a computation over data of element type `T`. It is the floating-point type that Base `float` gives for `T` when `T` is an `Integer`, and `T` for every other type.

This function is the one place where the library changes a derived numeric type to a floating-point type. An integer type cannot hold a mean, a variance, a weight or a `NaN`. So an integer sample must use a floating-point working type, or the first write of a fraction raises an `InexactError`. The function keeps every other type. A `Float32` sample stays in `Float32`, and a `Rational` sample stays exact. The function does not convert an automatic-differentiation dual, or a number type from another package, to the type that `float` gives. A call of `float` on `T` does not make that distinction. The type of a division such as `one(T) / one(T)` does not make it either, because a number type can define the division in a different type. A site that must repair an integer input calls this function on the type of its data, and never calls `float`.

Some operations change the type, for example the square root of a `Rational`. A site with such an operation takes its type from that operation, applied to the result of this function.

# Algorithm

 1. `T` is a subtype of `Integer`, which includes `Bool`: return the floating-point type that Base `float` gives for `T`, for example `Float64` for `Int` and `BigFloat` for `BigInt`.
 2. `T` is any other type: return `T`.

Dispatch on `Type{T}` makes the choice, which costs nothing at run time. Inference reads the returned type as a constant.

# Arguments

  - `T`: The element type of the data, usually `eltype` of an argument, or a `promote_type` of the element types of several arguments.

# Returns

  - `Tf::Type`: The floating-point type of `T` if `T <: Integer`, else `T`.

# Examples

```jldoctest
julia> PortfolioOptimisers.float_if_integer(Int)
Float64

julia> PortfolioOptimisers.float_if_integer(Float32)
Float32

julia> PortfolioOptimisers.float_if_integer(Rational{Int})
Rational{Int64}
```

# Related

  - [`concrete_typed_array`](@ref)
"""
float_if_integer(::Type{T}) where {T <: Integer} = float(T)
float_if_integer(::Type{T}) where {T} = T

export concrete_typed_array
public traverse_concrete_subtypes
