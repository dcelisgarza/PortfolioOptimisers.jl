"""
    traverse_concrete_subtypes(t, ctarr::Option{<:AbstractVector} = nothing) -> AbstractVector

Recursively traverse all subtypes of the given abstract type `t` and collect all struct types into `ctarr`.

A **struct type** is not the same as a concrete type. `InteractiveUtils.subtypes` reports a parametric struct as its `UnionAll`, and a `UnionAll` is not concrete, so a parametric struct is collected under its bare name and `isconcretetype` is `false` for it. What every entry does satisfy is `isstructtype`. A caller that needs concrete types must instantiate the parameters itself.

# Algorithm

 1. When `ctarr` is `nothing`, make it an empty `Vector{Any}`. The accumulator is threaded through the recursion, so one array collects every branch.
 2. Read `sts`, the direct subtypes of `t`, with `InteractiveUtils.subtypes`.
 3. For each subtype `st` of `sts`, take one of two branches:
     1. `st` is not a struct type, so it is a further abstract type: call this function again on `st` with the same `ctarr`.
     2. `st` is a struct type: push `st` onto `ctarr`. The test is `isstructtype` and not `isconcretetype`, which is why a parametric struct is collected.
 4. Return `ctarr`.

The recursion descends the whole tree below `t`, so an abstract type at any depth is opened and never collected. The order of `ctarr` is the depth-first order of the tree, which follows the order that `InteractiveUtils.subtypes` reports.

# Arguments

  - `t`: An abstract type whose subtypes will be traversed.
  - `ctarr`: Optional. An array to collect the struct types into. If not provided, a new empty array is created.

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

Convert an `AbstractArray` `A` to a concrete typed array, where each element is of the same type as the elements of `A`.

This is useful for converting arrays with abstract element types to arrays with concrete element types, which can improve performance in some cases.

# Algorithm

 1. Read the concrete type of every element of `A` with `typeof.(A)`.
 2. Build the element type `Union{typeof.(A)...}`, the union of exactly those types. An element type that no element carries is absent from the union.
 3. Splat `A` into a vector of that element type, which flattens `A` to one dimension.
 4. Reshape the vector back to `size(A)`, and return it.

The elements are copied into a new array, and each keeps its own type. The union is built from the values, so the result is only as narrow as the array's contents allow: an `Any` array holding one `Int64` comes back as a `Vector{Int64}`.

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

# Algorithm

 1. Test `eltype(A)` with `isabstracttype`.
 2. The element type is abstract: return [`concrete_typed_array`](@ref) of `A`, which copies.
 3. The element type is concrete: return `A` itself, which copies nothing.

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

export concrete_typed_array
public traverse_concrete_subtypes
