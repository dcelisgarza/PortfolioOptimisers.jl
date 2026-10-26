"""
    nothing_scalar_array_view(
        x::Union{Nothing, Number, Pair, VecPair, Dict,
                 AbstractEstimatorValueAlgorithm,
                 DynamicAbstractWeights, AbstractEstimator, AbstractAlgorithm,
                 StatsBase.CovarianceEstimator},
        ::Any
    ) -> x
    nothing_scalar_array_view(x::AbstractVector, i) -> view(x, i)
    nothing_scalar_array_view(x::VecScalar, i) -> VecScalar(; v = view(x.v, i), s = x.s)
    nothing_scalar_array_view(x::AbstractMatrix, i) -> view(x, i, i)
    nothing_scalar_array_view(
        x::AbstractVector{<:Union{AbstractVector, AbstractMatrix, VecScalar}},
        i
    ) -> [nothing_scalar_array_view(xi, i) for xi in x]

View of a value at the asset index `i`, where the value can be `nothing`, a scalar or an array.

# Algorithm

The method that Julia selects is the algorithm. Each step is one method, and no method allocates a copy of the data.

 1. `x` carries no asset axis, because it is `nothing`, a scalar, a pair, a dictionary, a value algorithm, a set of dynamic weights, an estimator, an algorithm or a `StatsBase.CovarianceEstimator`: return `x` itself.
 2. `x` is a vector: return `view(x, i)`, one entry per selected asset.
 3. `x` is a [`VecScalar`](@ref): return a new [`VecScalar`](@ref) whose vector part is `view(x.v, i)` and whose scalar part is `x.s`. The scalar part has no asset axis.
 4. `x` is a matrix: return `view(x, i, i)`, which selects the **same** index on **both** axes. This is the rule for a square matrix over the assets, such as a covariance matrix or a similarity matrix. A matrix whose two axes are different needs [`nothing_scalar_array_view_odd_order`](@ref).
 5. `x` is a vector of vectors, matrices or [`VecScalar`](@ref)s: apply step 2, 3 or 4 to each element, and collect the views into a new outer vector of the same length. The **element type** of the vector selects this step, and it must be a subtype of the `Union` that the signature names. A vector that holds both a vector and a matrix has the element type `Array{T}`. That type is a subtype of neither `AbstractVector` nor `AbstractMatrix`, so the vector resolves on step 2, and the index selects elements of the outer vector.

# Arguments

  - `x`: Input value.
  - `i`: Index or indices to view.

# Returns

  - `x`: Input value.

      + `::Union{Nothing, Number, Pair, VecPair, Dict, AbstractEstimatorValueAlgorithm, DynamicAbstractWeights, AbstractEstimator, AbstractAlgorithm, StatsBase.CovarianceEstimator}`: Returns `x` unchanged.
      + `::AbstractVector`: Returns `view(x, i)`.
      + `::VecScalar`: Returns `VecScalar(; v = view(x.v, i), s = x.s)`.
      + `::AbstractMatrix`: Returns `view(x, i, i)`.
      + `::AbstractVector{<:Union{AbstractVector, AbstractMatrix, VecScalar}}`: Returns a vector of views for each element in `x`.

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
function nothing_scalar_array_view(x::Union{Nothing, Number, Pair, VecPair, Dict,
                                            AbstractEstimatorValueAlgorithm,
                                            DynamicAbstractWeights, AbstractEstimator,
                                            AbstractAlgorithm,
                                            StatsBase.CovarianceEstimator}, ::Any)
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
function nothing_scalar_array_view(x::AbstractVector{<:Union{AbstractVector, AbstractMatrix,
                                                             VecScalar}}, i)
    return [nothing_scalar_array_view(xi, i) for xi in x]
end
"""
    port_opt_view(x, i, args...; kwargs...) -> nothing_scalar_array_view(x, i)

Select the assets `i` of an estimator, a result or an algorithm.

`port_opt_view` is the **index-selection counterpart of [`factory`](@ref)**. `factory`
passes *runtime values* down a tree of composed structs, and `port_opt_view` passes an
*index selection* down the same tree. It restricts every field that holds data, and every
composed child, to the subset `i`. The meta-optimisers, such as [`NestedClustered`](@ref)
and [`SubsetResampling`](@ref), and the cross-validation variants use it to solve
sub-problems with structs of the same shape.

A caller does not usually call `port_opt_view`. The meta-optimisers and the
cross-validation internals call it. It is `public` and not exported, because the author of
a new composed estimator can need to define a method. Tag each field that holds data with
[`@vprop`](@ref), and [`@propagatable`](@ref) makes the method.

This universal fallback handles *leaf* values. It views an array through
[`nothing_scalar_array_view`](@ref). It returns a scalar, `nothing`, an estimator with no
data fields and an algorithm with no change. A composed struct that recurses into its
children has its own, more specific method, which [`@propagatable`](@ref) makes from the
[`@vprop`](@ref) tags or which the author writes by hand.

This method accepts and drops the threaded tail `args...` and all `kwargs`. The tail is
usually the returns matrix `X` of the JuMP families. With this method, a call
`port_opt_view(child, i, X)` that a macro makes never raises a `MethodError` on a leaf field.

# Algorithm

 1. Drop `args...` and `kwargs...`. This method is the leaf of the recursion, so it passes nothing further.
 2. Return [`nothing_scalar_array_view`](@ref) of `x` at `i`. The algorithm of that function gives the rule for each leaf type.

# Related

  - [`factory`](@ref)
  - [`@vprop`](@ref)
  - [`nothing_scalar_array_view`](@ref)
"""
port_opt_view(x, i, args...; kwargs...) = nothing_scalar_array_view(x, i)
"""
    port_opt_view(x::VecScalar, i, args...) -> nothing_scalar_array_view(x, i)

[`port_opt_view`](@ref) method for [`VecScalar`](@ref). It views the vector part, keeps
the scalar part, and calls [`nothing_scalar_array_view`](@ref).

# Algorithm

 1. Drop `args...`.
 2. Return [`nothing_scalar_array_view`](@ref) of `x` at `i`. Its [`VecScalar`](@ref) method views `x.v` at `i` and keeps `x.s`.

A [`VecScalar`](@ref) that reaches the verb with a threaded tail resolves on this method, not on the universal leaf method. Both methods give the same value.

# Related

  - [`VecScalar`](@ref)
  - [`nothing_scalar_array_view`](@ref)
"""
port_opt_view(x::VecScalar, i, args...) = nothing_scalar_array_view(x, i)
"""
    port_opt_view(::Nothing, ::Any; kwargs...) -> nothing
    port_opt_view(::Nothing, ::Any, args...; kwargs...) -> nothing

Fallback of [`port_opt_view`](@ref) for an absent value. The index view of a `nothing`
estimator, algorithm, result or constraint is `nothing`.

Every propagation family uses these methods. Many optional fields have the type
`Option{T} = Union{Nothing, T}`. A `nothing` argument matches both the `Option{T}`
passthrough of a family and the universal leaf fallback, and the `::Nothing` methods
resolve that ambiguity. Both methods have a fixed second positional argument, so they are
more specific than the universal `port_opt_view(x, i, args...)` method.

# Algorithm

 1. Return `nothing`. Neither method reads its index, its tail or its keywords.

The two methods differ only in the tail, and both are necessary. A call site that threads no tail resolves on the first method. A call site that threads a returns matrix resolves on the second method.

# Examples

```jldoctest
julia> PortfolioOptimisers.port_opt_view(nothing, 1)

```
"""
port_opt_view(::Nothing, ::Any; kwargs...) = nothing
port_opt_view(::Nothing, ::Any, args...; kwargs...) = nothing
"""
    port_opt_view(x::AbstractVector{<:Union{Nothing, AbstractEstimator, AbstractAlgorithm, AbstractResult}}, i, args...; kwargs...) -> Vector

Generic vector method for [`port_opt_view`](@ref): view each element of `x` at the index selection `i`.

This method is the index-selection twin of the vector [`factory`](@ref) method. It is the **one forwarding contract** for every propagation field that holds a vector. Each element gets the tail `args...` and every keyword with no change. The tail is usually the returns matrix `X`. A family that accepts a vector of estimators, algorithms or results needs no method of its own.

Without this method, such a vector goes to the universal leaf method `port_opt_view(x, i, args...)`. That method views the vector itself through [`nothing_scalar_array_view`](@ref), so the asset index selects *elements* and not assets. A family that needs more than the forward defines its own, more specific method. A concrete element type, through [`concrete_typed_array_if_abstract`](@ref), and a passthrough are two examples.

# Algorithm

 1. For each element `xi` of `x`, call [`port_opt_view`](@ref) on `xi` at `i`, and forward `args...` and `kwargs...` with no change.
 2. Collect the results into a new vector, in the order of `x`, and return it.

The result has the length of `x`, because `i` goes to the elements and never to the outer vector. The result is a comprehension, so Julia infers its element type. A family that needs a concrete element type wraps the call in [`concrete_typed_array_if_abstract`](@ref).

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
function port_opt_view(x::AbstractVector{<:Union{Nothing, AbstractEstimator,
                                                 AbstractAlgorithm, AbstractResult}}, i,
                       args...; kwargs...)
    return [port_opt_view(xi, i, args...; kwargs...) for xi in x]
end
"""
    view_child(v, i, args...)

View of one field. The [`port_opt_view`](@ref) methods that [`@propagatable`](@ref) makes call it, and so does the hand-written view of every optimiser that holds a fallback `fb`.

It is the view twin of [`factory_child`](@ref). It views a child at the asset index, but it keeps a precomputed optimisation result with no change. A result is in a field as a fallback `fb`. The fallback loop of [`optimise`](@ref) returns that result without a solve, on the universe of its own solve. A function such as [`investable_reduction`](@ref) views an optimiser and never reads its fallback. A refusal of the result there refuses a fallback that nothing reads.

A [`TimeDependent`](@ref) schedule is not a result, so `view_child` sends it to [`port_opt_view`](@ref). A schedule that holds a result still refuses a view of an asset subset.

# Algorithm

The method that Julia selects is the algorithm.

 1. `v` is a precomputed optimisation result: return `v` unchanged.
 2. `v` is anything else: return [`port_opt_view`](@ref) of `v` at `i`, and forward `args...`.

# Arguments

  - `v`: The field value.
  - `i`: Index selection.
  - `args...`: Threaded tail, forwarded to [`port_opt_view`](@ref).

# Returns

  - The view of the field value, or `v` with no change when it is a precomputed optimisation result.

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

`obs_weights_view` is the observation-axis counterpart of [`port_opt_view`](@ref). [`@propagatable`](@ref) makes its methods from the tags that a struct already has. [`@wprop`](@ref) marks the field that *holds* the weights, and the method indexes that field. [`@fprop`](@ref) marks a composed child, and the method recurses into it. All other fields keep their values, and the type of the struct does not change.

# Why the observation axis needs its own verb

[`port_opt_view`](@ref) passes one index into every field with a `@vprop` tag. Its callers are the meta-optimisers and the cross-validation splitters, and there the index selects **assets**. An observation weight is one value per row of the sample, so a cut of it by that index is wrong. The verb that the caller calls tells the two axes apart, not the index.

[`factory`](@ref) reads the same `@wprop` tag on the same field, but it does a different operation. It **replaces** the field with an incoming [`ObsWeights`](@ref) value, at every level of the tree at the same time. For this reason, a cut cannot go through `factory`. Take a [`SimpleVariance`](@ref) with a weighted mean and an unweighted dispersion. `factory` returns it with both weighted, which is a different estimator. This verb indexes each field separately, so a field that held `nothing` still holds `nothing`.

# Algorithm

 1. Return `x` with no change. This universal fallback reads neither its index nor the fields of `x`. An estimator that holds no weights, and an estimator whose struct is not [`@propagatable`](@ref), come back with no change.

A [`@propagatable`](@ref) struct with at least one `@wprop` field has a generated method that is more specific than this one. That method builds the struct again with the same constructor. It indexes each `@wprop` field at `i` through [`nothing_scalar_array_getindex`](@ref), and it calls this verb on each `@fprop` field. A hand-written type that holds weights outside that tag must define its own method. If it does not, its weights keep the length of the full sample, and the call on a window raises an error.

# Arguments

  - `x`: Estimator, algorithm, result, weights vector, or `nothing`. The argument has no type, because the variance estimators subtype `StatsBase.CovarianceEstimator`, the expected returns estimators subtype [`AbstractEstimator`](@ref), and both reach this verb.
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
  - [`realised_vol`](@ref): the caller of this verb in the library.
"""
obs_weights_view(x, ::Any) = x
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Vector method of [`obs_weights_view`](@ref). It applies the verb to every element, so the verb reaches each composed child of an `@fprop` field that holds a vector.

# Algorithm

 1. For each element `xi` of `x`, call [`obs_weights_view`](@ref) on `xi` at `i`.
 2. Collect the results into a new vector, in the order of `x`, and return it.

The result has the length of `x`, because `i` selects observations inside each element and never elements of `x`. Without this method, such a vector goes to the universal fallback, and its weights keep the length of the full sample.

# Related

  - [`obs_weights_view`](@ref)
  - [`@fprop`](@ref)
"""
function obs_weights_view(x::AbstractVector{<:Union{Nothing, AbstractEstimator,
                                                    AbstractAlgorithm, AbstractResult}}, i)
    return [obs_weights_view(xi, i) for xi in x]
end
"""
    get_window(window::Option{<:Colon}, args...) -> Colon
    get_window(window::Integer, X::MatNum, dims::Int = 1) -> VecInt
    get_window(window::Integer, X::VecNum, args...) -> VecInt
    get_window(window::VecInt, args...) -> VecInt

Indices of the observation window of a data array.

# Algorithm

The type of `window` names the rule, and each rule is one step.

 1. `window` is `nothing` or a `Colon`: return `Colon()`, which selects every observation.
 2. `window` is an integer: read `start`, the first index of `X` along the observation axis, and `stop`, its last index. Return the range `max(start, stop - window + 1):stop`, which is the last `window` observations.
 3. `window` is a vector of integers: return `window` itself, so the caller states the observations directly.

Step 2 clamps the lower end at `start`, so a `window` larger than the number of observations gives every observation and not an error. The observation axis of a matrix is `dims`. A vector has one axis, so its method ignores `dims`.

# Arguments

  - $(arg_dict[:window])
      + `::Option{<:Colon}`: Returns `Colon()`.
      + `::Integer`: Returns the last `window` observations. If `window` is larger than the number of observations, the result is every observation and not an error.
      + `::VecInt`: Returns the `window` argument.
  - $(arg_dict[:X_Xv])
  - $(arg_dict[:dims])

# Returns

  - `window::Union{Colon, VecInt}`: The indices of the window.

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

View of a value that can be `nothing` or a matrix, with a different index on each axis.

# Algorithm

 1. `x` is `nothing`: return `nothing`.
 2. `x` is a matrix: return `view(x, i, j)`, which selects `i` on the row axis and `j` on the column axis.

The two axes take **different** indices. This is the difference from [`nothing_scalar_array_view`](@ref). The main case is the coskewness matrix, of size ``N \\times N^{2}``. Its row index selects assets, and its column index selects the pairs of assets that the columns hold. The caller gives `j`, and this verb does not make it. [`fourth_moment_index_generator`](@ref) makes that column index. Other callers pass `:` as one of the two indices.

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
        x::Union{Nothing, Number, Pair, VecPair, Dict,
                 AbstractEstimatorValueAlgorithm,
                 DynamicAbstractWeights},
        ::Any
    ) -> x
    nothing_scalar_array_getindex(x::AbstractVector, i) -> x[i]
    nothing_scalar_array_getindex(x::StatsBase.AbstractWeights, i) -> weights of the same kind
    nothing_scalar_array_getindex(x::VecScalar, i) -> VecScalar(; v = x.v[i], s = x.s)
    nothing_scalar_array_getindex(x::AbstractMatrix, i) -> x[i, i]
    nothing_scalar_array_getindex(
        x::AbstractVector{<:Union{AbstractVector, AbstractMatrix, VecScalar}},
        i
    ) -> [nothing_scalar_array_getindex(xi, i) for xi in x]

Copy of a value at the asset index `i`, where the value can be `nothing`, a scalar or an array.

# Algorithm

The method that Julia selects is the algorithm. It is the copying twin of [`nothing_scalar_array_view`](@ref). Every step returns a new array and not a view.

 1. `x` carries no asset axis, because it is `nothing`, a scalar, a pair, a dictionary, a value algorithm or a set of dynamic weights: return `x` itself.
 2. `x` is a vector: return `x[i]`, a new vector with one entry per selected asset.
      + `x` is a set of `StatsBase.Weights`, `AnalyticWeights`, `FrequencyWeights` or `ProbabilityWeights`: build weights of the same kind from `x.values[i]`. The `getindex` of `StatsBase` builds the result with the type of the stored values, so it fails on weights that store a range.
 3. `x` is a [`VecScalar`](@ref): return a new [`VecScalar`](@ref) whose vector part is `x.v[i]` and whose scalar part is `x.s`.
 4. `x` is a matrix: return `x[i, i]`, which selects the **same** index on **both** axes. This is the rule for a square matrix over the assets. A matrix whose two axes are different needs [`nothing_scalar_array_getindex_odd_order`](@ref).
 5. `x` is a vector of vectors, matrices or [`VecScalar`](@ref)s: apply step 2, 3 or 4 to each element, and collect the results into a new vector. The **element type** of the vector selects this step, and it must be a subtype of the `Union` that the signature names. A vector that holds both a vector and a matrix has the element type `Array{T}`. That type is a subtype of neither `AbstractVector` nor `AbstractMatrix`, so the vector resolves on step 2, and the index selects elements of the outer vector.

The type list of step 1 is **shorter** than the list of [`nothing_scalar_array_view`](@ref). An estimator, an algorithm and a `StatsBase.CovarianceEstimator` reach the view verb and not this one, because only the view verb is the leaf of [`port_opt_view`](@ref).

# Arguments

  - `x`: Input value.
  - `i`: Index or indices to copy.

# Returns

  - `x`: Input value.

      + `::Union{Nothing, Number, Pair, VecPair, Dict, AbstractEstimatorValueAlgorithm, DynamicAbstractWeights}`: Returns `x` unchanged.
      + `::AbstractVector`: Returns `x[i]`.
      + `::StatsBase.AbstractWeights`, except `StatsBase.UnitWeights`: Returns weights of the same kind that hold `x.values[i]`.
      + `::VecScalar`: Returns `VecScalar(; v = x.v[i], s = x.s)`.
      + `::AbstractVector{<:Union{AbstractVector, AbstractMatrix, VecScalar}}`: Returns a vector of elements indexed by `i`.
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
function nothing_scalar_array_getindex(x::Union{Nothing, Number, Pair, VecPair, Dict,
                                                AbstractEstimatorValueAlgorithm,
                                                DynamicAbstractWeights}, ::Any)
    return x
end
function nothing_scalar_array_getindex(x::AbstractVector, i)
    return x[i]
end
function nothing_scalar_array_getindex(x::Union{StatsBase.Weights,
                                                StatsBase.AnalyticWeights,
                                                StatsBase.FrequencyWeights,
                                                StatsBase.ProbabilityWeights}, i)
    return Base.typename(typeof(x)).wrapper(x.values[i])
end
function nothing_scalar_array_getindex(x::VecScalar, i)
    return VecScalar(; v = x.v[i], s = x.s)
end
function nothing_scalar_array_getindex(x::AbstractMatrix, i)
    return x[i, i]
end
function nothing_scalar_array_getindex(x::AbstractVector{<:Union{AbstractVector,
                                                                 AbstractMatrix, VecScalar}},
                                       i)
    return [nothing_scalar_array_getindex(xi, i) for xi in x]
end
"""
    nothing_scalar_array_getindex_odd_order(::Nothing, i, j)
    nothing_scalar_array_getindex_odd_order(x::AbstractMatrix, i, j) -> x[i, j]

Copy of a value that can be `nothing` or a matrix, with a different index on each axis.

# Algorithm

 1. `x` is `nothing`: return `nothing`.
 2. `x` is a matrix: return the copy `x[i, j]`, which selects `i` on the row axis and `j` on the column axis.

This is the copying twin of [`nothing_scalar_array_view_odd_order`](@ref). It takes **different** indices on the two axes, because the two axes of the matrix are different.

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

Index vector of the asset pairs that the assets `i` make, among the ``N^{2}`` pairs of ``N`` assets. It selects the sub-matrix of the assets `i` from a cokurtosis matrix, and the columns of the assets `i` from a coskewness matrix.

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

``(c - 1) N + r`` is the column-major linear index of the pair ``(r, c)`` in that grid. The grid is the axis of the square cokurtosis matrix ``\\mathbf{K}``, of size ``N^{2} \\times N^{2}``. `idx` selects the ``n^{2}`` pairs that the ``n`` selected assets make, on either axis of ``\\mathbf{K}``. `c` is the outer loop, so `idx` is also in the column-major order of the sub-grid.

# Algorithm

 1. Make `idx`, an empty vector of integers, with room for `length(i)^2` entries.
 2. For each `c` in `i`, take the linear index range of column `c`, which is `((c - 1) * N + 1):(c * N)`, and select the entries `i` of that range.
 3. Append the selected entries to `idx`.
 4. Return `idx`.

# Arguments

  - `N`: Number of assets, the size of the full covariance matrix.
  - `i`: Indices of the selected assets.

# Returns

  - `idx::VecInt`: Linear indices of the selected asset pairs.

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
