"""
    nothing_scalar_array_view(
        x::Union{Nothing, <:Number, <:Pair, <:VecPair, <:Dict,
                 <:AbstractEstimatorValueAlgorithm,
                 <:DynamicAbstractWeights, <:AbstractEstimator, <:AbstractAlgorithm,
                 <:StatsBase.CovarianceEstimator},
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

# Algorithm

The method that Julia selects is the algorithm. Each step is one method, and no method allocates a copy of the data.

 1. `x` carries no asset axis, because it is `nothing`, a scalar, a pair, a dictionary, a value algorithm, a set of dynamic weights, an estimator, an algorithm or a `StatsBase.CovarianceEstimator`: return `x` itself.
 2. `x` is a vector: return `view(x, i)`, one entry per selected asset.
 3. `x` is a [`VecScalar`](@ref): return a new [`VecScalar`](@ref) whose vector part is `view(x.v, i)` and whose scalar part `x.s` is carried through. The scalar part carries no asset axis.
 4. `x` is a matrix: return `view(x, i, i)`, which selects the **same** index on **both** axes. This is the rule for a square per-asset matrix, such as a covariance matrix or a similarity matrix. A matrix whose two axes are different needs [`nothing_scalar_array_view_odd_order`](@ref) instead.
 5. `x` is a vector of vectors, matrices or [`VecScalar`](@ref)s: apply step 2, 3 or 4 to each element, and collect the views into a new vector. The outer vector is rebuilt, so its own length is unchanged. The vector's **element type** selects this step, and it must be a subtype of the `Union` the signature names. A vector holding both a vector and a matrix has the element type `Array{T}`, which is a subtype of neither `AbstractVector` nor `AbstractMatrix`, so it resolves on step 2 and the index selects the elements of the outer vector.

# Arguments

  - `x`: Input value.
  - `i`: Index or indices to view.

# Returns

  - `x`: Input value.

      + `::Union{Nothing, <:Number, <:Pair, <:VecPair, <:Dict, <:AbstractEstimatorValueAlgorithm, <:DynamicAbstractWeights, <:AbstractEstimator, <:AbstractAlgorithm, <:StatsBase.CovarianceEstimator}`: Returns `x` unchanged.
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

# Algorithm

 1. Drop `args...` and `kwargs...`. This method is the leaf of the recursion, so it threads nothing further.
 2. Return [`nothing_scalar_array_view`](@ref) of `x` at `i`, whose own algorithm names the rule for each leaf type.

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

# Algorithm

 1. Drop `args...`.
 2. Return [`nothing_scalar_array_view`](@ref) of `x` at `i`, whose [`VecScalar`](@ref) method views `x.v` at `i` and carries `x.s` through.

This method exists so that a [`VecScalar`](@ref) reaching the verb with a threaded tail resolves here rather than through the universal leaf method. Both routes give the same value.

# Related

  - [`VecScalar`](@ref)
  - [`nothing_scalar_array_view`](@ref)
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

# Algorithm

 1. Return `nothing`. Neither method reads its index, its tail or its keywords.

The two methods differ only in whether they accept a tail, and both are needed: a call site that threads no tail resolves on the first, and one that threads a returns matrix resolves on the second.

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

# Algorithm

 1. For each element `xi` of `x`, call [`port_opt_view`](@ref) on `xi` at `i`, and forward `args...` and `kwargs...` unchanged.
 2. Collect the results into a new vector, in the order of `x`, and return it.

The length of `x` is unchanged, because `i` reaches the elements and never the outer vector. The result is a comprehension, so its element type is whatever Julia infers; a family that needs a concrete element type wraps the call in [`concrete_typed_array_if_abstract`](@ref).

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
    view_child(v, i, args...)

Per-field view helper called by [`@propagatable`](@ref)-generated [`port_opt_view`](@ref) methods, and by the hand-written view of every optimiser that holds a fallback `fb`.

It is the view twin of [`factory_child`](@ref). A child is viewed at the asset index, with one exception: a precomputed optimisation result is kept as it is. A result sits in a field as a fallback `fb`, and the fallback loop of [`optimise`](@ref) answers it without a solve, on the universe it was solved on. A door such as [`investable_reduction`](@ref) views an optimiser whose own fallback it never reads, so a refusal there would reject a fallback that nothing reaches.

A [`TimeDependent`](@ref) schedule is not a result, so it reaches [`port_opt_view`](@ref), and a schedule that holds a result still refuses a subset view.

# Algorithm

The method that Julia selects is the algorithm.

 1. `v` is a precomputed optimisation result: return `v` unchanged.
 2. `v` is anything else: return [`port_opt_view`](@ref) of `v` at `i`, and forward `args...`.

# Arguments

  - `v`: The field value.
  - `i`: Index selection.
  - `args...`: Threaded tail, forwarded to [`port_opt_view`](@ref).

# Returns

  - The viewed field value, or `v` itself when it is a precomputed optimisation result.

# Related

  - [`port_opt_view`](@ref)
  - [`factory_child`](@ref)
  - [`@vprop`](@ref)
"""
function view_child(v, i, args...)
    return port_opt_view(v, i, args...)
end
"""
    obs_weights_view(x, i) -> typeof(x)

Sub-select an estimator's **observation weights** to the observations `i`.

`obs_weights_view` is the observation-axis counterpart of [`port_opt_view`](@ref), and it is generated by [`@propagatable`](@ref) from the tags a struct already carries: [`@wprop`](@ref) marks the field that *holds* the weights, so that field is indexed, and [`@fprop`](@ref) marks a composed child, so the verb recurses into it. Every other field is carried through unchanged, and the struct's type does not change.

# Why the observation axis needs its own verb

[`port_opt_view`](@ref) threads one index into every `@vprop`-tagged field, and at its call sites — the meta-optimisers and the cross-validation splitters — that index selects **assets**. An observation weight is one value per row of the sample, so slicing it there would be wrong. The two axes are told apart by which verb is called, not by the index.

[`factory`](@ref) reads the same `@wprop` tag on the same field, and does a different thing with it: it **replaces** the field with an incoming [`ObsWeights`](@ref) value, at every level of the tree at once. That is why a slice cannot go through `factory`. A [`SimpleVariance`](@ref) holding a weighted mean and an unweighted dispersion comes back from `factory` with both weighted, which is a different estimator; here each field is indexed on its own, so a field that held `nothing` still holds `nothing`.

# Algorithm

 1. Return `x` unchanged. This universal fallback reads neither its index nor the fields of `x`. An estimator that carries no weights, and one whose struct is not [`@propagatable`](@ref), therefore behave as they did before the verb existed.

A [`@propagatable`](@ref) struct with at least one `@wprop`-tagged field carries a generated method that dominates this one. That method rebuilds the struct with the same constructor, indexing each `@wprop` field to `i` through [`nothing_scalar_array_getindex`](@ref) and recursing into each `@fprop` field through this verb. A hand-written type that holds weights outside that tag must define its own method, or its weights keep their full-sample length and the windowed call raises.

# Arguments

  - `x`: Estimator, algorithm, result, weights vector, or `nothing`. The argument is untyped, because the variance estimators subtype `StatsBase.CovarianceEstimator` while the expected returns estimators subtype [`AbstractEstimator`](@ref), and both reach this verb.
  - `i`: Index or indices of the observations to keep.

# Returns

  - `x`: The value, with every observation-weights field indexed to `i`.

# Related

  - [`port_opt_view`](@ref)
  - [`factory`](@ref)
  - [`@wprop`](@ref)
  - [`PROP_TAG_CHANNELS`](@ref)
  - [`nothing_scalar_array_getindex`](@ref)
  - [`ObsWeights`](@ref)
  - [`realised_vol`](@ref): the site that drives this verb.
"""
obs_weights_view(x, ::Any) = x
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Vector overload of [`obs_weights_view`](@ref). Applies the verb to every element, so an `@fprop`-tagged field holding a vector of composed children is not silently skipped.

# Algorithm

 1. For each element `xi` of `x`, call [`obs_weights_view`](@ref) on `xi` at `i`.
 2. Collect the results into a new vector, in the order of `x`, and return it.

The length of `x` is unchanged, because `i` selects observations inside each element and never elements of `x`. Without this method such a vector reaches the universal fallback and comes back with full-sample weights.

# Related

  - [`obs_weights_view`](@ref)
  - [`@fprop`](@ref)
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

# Algorithm

The type of `window` names the rule, and each rule is one step.

 1. `window` is `nothing` or a `Colon`: return `Colon()`, which selects every observation.
 2. `window` is an integer: read `start`, the first index of `X` along the observation axis, and `stop`, its last index. Return the range `max(start, stop - window + 1):stop`, which is the last `window` observations.
 3. `window` is a vector of integers: return `window` itself, so the caller states the observations directly.

Step 2 clamps the lower end at `start`, so a `window` larger than the number of observations gives every observation rather than an error. The observation axis of a matrix is `dims`, and a vector carries one axis, so its method drops `dims`.

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

# Algorithm

 1. `x` is `nothing`: return `nothing`.
 2. `x` is a matrix: return `view(x, i, j)`, which selects `i` on the row axis and `j` on the column axis.

The two axes take **different** indices, which is what separates this verb from [`nothing_scalar_array_view`](@ref). An odd-order co-moment matrix is ``N \\times N^{k}`` for an odd order ``k``, so the row index selects assets and the column index selects the tuples of assets that the columns hold. The caller supplies `j`, and this verb does not derive it; [`fourth_moment_index_generator`](@ref) is the counterpart that builds such a column index.

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

# Algorithm

The method that Julia selects is the algorithm. It is the copying twin of [`nothing_scalar_array_view`](@ref): every step returns a new array rather than a view.

 1. `x` carries no asset axis, because it is `nothing`, a scalar, a pair, a dictionary, a value algorithm or a set of dynamic weights: return `x` itself.
 2. `x` is a vector: return `x[i]`, a new vector with one entry per selected asset.
 3. `x` is a [`VecScalar`](@ref): return a new [`VecScalar`](@ref) whose vector part is `x.v[i]` and whose scalar part `x.s` is carried through.
 4. `x` is a matrix: return `x[i, i]`, which selects the **same** index on **both** axes. This is the rule for a square per-asset matrix. A matrix whose two axes are different needs [`nothing_scalar_array_getindex_odd_order`](@ref) instead.
 5. `x` is a vector of vectors, matrices or [`VecScalar`](@ref)s: apply step 2, 3 or 4 to each element, and collect the results into a new vector. The vector's **element type** selects this step, and it must be a subtype of the `Union` the signature names. A vector holding both a vector and a matrix has the element type `Array{T}`, which is a subtype of neither `AbstractVector` nor `AbstractMatrix`, so it resolves on step 2 and the index selects the elements of the outer vector.

The type list of step 1 is **shorter** than the one [`nothing_scalar_array_view`](@ref) carries: an estimator, an algorithm and a `StatsBase.CovarianceEstimator` reach the view verb and not this one, because only the view verb is the leaf of [`port_opt_view`](@ref).

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

# Algorithm

 1. `x` is `nothing`: return `nothing`.
 2. `x` is a matrix: return `x[i, j]`, which selects `i` on the row axis and `j` on the column axis, and copies.

This is the copying twin of [`nothing_scalar_array_view_odd_order`](@ref), and it takes **different** indices on the two axes for the same reason: an odd-order co-moment matrix is ``N \\times N^{k}`` for an odd order ``k``.

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

# Mathematical definition

```math
\\begin{align}
\\mathrm{idx} &= \\left( (c - 1) N + r \\right)_{c \\in \\boldsymbol{i},\\ r \\in \\boldsymbol{i}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:N])
  - ``\\boldsymbol{i}``: The selected asset indices, of length ``n``.
  - ``r``, ``c``: The row and the column of an asset pair in the ``N \\times N`` grid of pairs.

``(c - 1) N + r`` is the column-major linear index of the pair ``(r, c)`` in that grid, which is the axis of the square cokurtosis matrix ``\\mathbf{K}``, of size ``N^{2} \\times N^{2}``. So `idx` selects the ``n^{2}`` pairs that the ``n`` selected assets make, on either axis of ``\\mathbf{K}``. `c` runs on the outside, so the order of `idx` is the column-major order of the sub-grid too.

# Algorithm

 1. Make `idx`, an empty vector of integers, with room for `length(i)^2` entries.
 2. For each `c` in `i`, take the linear index range of column `c`, which is `((c - 1) * N + 1):(c * N)`, and select the entries `i` of that range.
 3. Append the selected entries to `idx`.
 4. Return `idx`.

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

public port_opt_view, obs_weights_view
