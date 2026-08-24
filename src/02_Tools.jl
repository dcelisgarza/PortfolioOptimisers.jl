"""
    ⊗(A::ArrNum, B::ArrNum) -> Matrix{promote_type(eltype(A), eltype(B))}

Tensor product of two arrays. Returns a matrix of size `(length(A), length(B))` where each element is the product of elements from `A` and `B`.

# Mathematical definition

```math
\\begin{align}
(\\boldsymbol{a} \\otimes \\boldsymbol{b})_{ij} &= a_{i} b_{j}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{a}``: Vectorised first array `A`, of length ``n``.
  - ``\\boldsymbol{b}``: Vectorised second array `B`, of length ``m``.
  - ``a_{i}``, ``b_{j}``: Entries of ``\\boldsymbol{a}`` and ``\\boldsymbol{b}`` in linear index order.

The result is the outer product ``\\boldsymbol{a} \\boldsymbol{b}^\\intercal``, an ``n \\times m`` matrix. `A` and `B` may carry any shape, because both are read in linear index order.

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

# Mathematical definition

```math
\\begin{align}
(\\boldsymbol{a} \\odot \\boldsymbol{b})_{i} &= a_{i} b_{i}\\,, \\\\
(\\boldsymbol{a} \\odot \\beta)_{i} &= a_{i} \\beta\\,, \\\\
(\\alpha \\odot \\boldsymbol{b})_{i} &= \\alpha b_{i}\\,, \\\\
\\alpha \\odot \\beta &= \\alpha \\beta\\,.
\\end{align}
```

Where:

  - $(math_dict[:ab_operands])
  - $(math_dict[:alpha_beta_scalars])
  - $(math_dict[:i_linear])

Both array operands must carry the same length. A scalar operand multiplies every entry of the array operand.

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

# Mathematical definition

```math
\\begin{align}
(\\boldsymbol{a} \\oslash \\boldsymbol{b})_{i} &= \\frac{a_{i}}{b_{i}}\\,, \\\\
(\\boldsymbol{a} \\oslash \\beta)_{i} &= \\frac{a_{i}}{\\beta}\\,, \\\\
(\\alpha \\oslash \\boldsymbol{b})_{i} &= \\frac{\\alpha}{b_{i}}\\,, \\\\
\\alpha \\oslash \\beta &= \\frac{\\alpha}{\\beta}\\,.
\\end{align}
```

Where:

  - $(math_dict[:ab_operands])
  - $(math_dict[:alpha_beta_scalars])
  - $(math_dict[:i_linear])

Both array operands must carry the same length. The division is not guarded, so a zero divisor gives an infinity or a `NaN`.

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

# Mathematical definition

```math
\\begin{align}
(\\boldsymbol{a} \\oplus \\boldsymbol{b})_{i} &= a_{i} + b_{i}\\,, \\\\
(\\boldsymbol{a} \\oplus \\beta)_{i} &= a_{i} + \\beta\\,, \\\\
(\\alpha \\oplus \\boldsymbol{b})_{i} &= \\alpha + b_{i}\\,, \\\\
\\alpha \\oplus \\beta &= \\alpha + \\beta\\,.
\\end{align}
```

Where:

  - $(math_dict[:ab_operands])
  - $(math_dict[:alpha_beta_scalars])
  - $(math_dict[:i_linear])

Both array operands must carry the same length. A scalar operand is added to every entry of the array operand, which the built-in `+` refuses.

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

# Mathematical definition

```math
\\begin{align}
(\\boldsymbol{a} \\ominus \\boldsymbol{b})_{i} &= a_{i} - b_{i}\\,, \\\\
(\\boldsymbol{a} \\ominus \\beta)_{i} &= a_{i} - \\beta\\,, \\\\
(\\alpha \\ominus \\boldsymbol{b})_{i} &= \\alpha - b_{i}\\,, \\\\
\\alpha \\ominus \\beta &= \\alpha - \\beta\\,.
\\end{align}
```

Where:

  - $(math_dict[:ab_operands])
  - $(math_dict[:alpha_beta_scalars])
  - $(math_dict[:i_linear])

Both array operands must carry the same length. A scalar operand is subtracted from every entry of the array operand, which the built-in `-` refuses.

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

# Mathematical definition

```math
\\begin{align}
\\mathrm{dot\\_scalar}(\\alpha, \\boldsymbol{b}) &= \\alpha \\sum_{i=1}^{n} b_{i}\\,, \\\\
\\mathrm{dot\\_scalar}(\\boldsymbol{a}, \\beta) &= \\beta \\sum_{i=1}^{n} a_{i}\\,, \\\\
\\mathrm{dot\\_scalar}(\\boldsymbol{a}, \\boldsymbol{b}) &= \\boldsymbol{a}^\\intercal \\boldsymbol{b}\\,.
\\end{align}
```

Where:

  - ``\\alpha``, ``\\beta``: Scalar operand, a number or a `JuMP` scalar.
  - ``\\boldsymbol{a}``, ``\\boldsymbol{b}``: Vector operand of length ``n``.

The first two forms are the dot product of the vector with a constant vector of value ``\\alpha``, so the scalar stands for a uniform vector. The sum replaces that constant vector, so no ``n``-length array is built.

# Arguments

  - `a`: First operand, a scalar or a vector.
  - `b`: Second operand, a scalar or a vector.

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

# Algorithm

The scalar method:

 1. Return `a` unchanged, and drop `args...` and `kwargs...`. This method is the leaf of the recursion, and it is what makes an untagged type safe to call the verb on.

The vector method:

 1. For each element `ai` of `a`, call `factory` on `ai`, and forward `args...` and `kwargs...` unchanged.
 2. Collect the results into a new vector, in the order of `a`, and return it.

A [`@propagatable`](@ref) struct with at least one `@fprop`- or `@wprop`-tagged field carries a generated method that dominates the scalar method. That method rebuilds the struct with its keyword constructor, sending each `@fprop` field through [`factory_child`](@ref) and each `@wprop` field through [`_wprop`](@ref).

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

# Algorithm

The method that Julia selects is the algorithm.

 1. `v` is an estimator, an algorithm or a result: return [`factory`](@ref) of `v`, forwarding `args...` and `kwargs...`. The recursion descends one level of the struct tree.
 2. `v` is an array of them: apply step 1 to each element, and collect the results into a new vector.
 3. `v` is anything else: return `v` unchanged. A data field, a scalar and a `nothing` all take this branch.

Step 3 is why a `nothing` field is **not** filled in by this verb. A weights field that must be replaced when it holds `nothing` carries `@wprop` and reaches [`_wprop`](@ref) instead.

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

# Algorithm

The method that Julia selects is the algorithm, and the selection reads `args...`, never the field.

 1. The first threaded positional argument is an [`ObsWeights`](@ref): return that value, whatever the field held.
 2. No such argument is threaded: return `field` unchanged.

The field's own value never selects the branch, so a field holding `nothing` and a field holding weights are both replaced by an incoming [`ObsWeights`](@ref), and both are kept when none is threaded.

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

Given a prior result the rule has two halves. **Container recursion is derived** from
[`deferred_slots`](@ref), so a type that only holds children needs no method at all. **A
type that resolves a quantity of its own defines a method**, which overrides the derived one.
Writing that half per type — rather than per field — is what lets slots that travel together
be resolved together: a deferred `sigma` supplies `chol` from the same fit, so the pair is
never mixed across two sources.

# Algorithm

 1. Return `x` unchanged. This method is the arm for a second argument that is **not** a prior result: with no prior in hand nothing can be fitted, so the deferred state travels on.

A more specific method dominates this one on a prior result: the one that [`deferred_slots`](@ref) derives for a container, and the hand-written one of a type that resolves a quantity of its own.

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

# Algorithm

 1. Map each tag of [`PROP_TAG_NAMES`](@ref) to `Symbol("@", tag)`, which is the name Julia gives that tag's macro.

The map preserves the order, so the ``k``-th entry here is the macro name of the ``k``-th tag there. [`prop_tag`](@ref) and [`check_prop_tag_macros`](@ref) both walk the two tuples together, and that pairing is what the shared order guarantees.

# Related

  - [`PROP_TAG_NAMES`](@ref)
  - [`prop_tag`](@ref)
"""
const PROP_TAG_MACRO_NAMES = Symbol.("@", PROP_TAG_NAMES)

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

# Algorithm

 1. Read `name` from `x`. A `GlobalRef` gives its `name` field, a `Symbol` gives itself, and any other value returns `nothing` at once.
 2. Walk [`PROP_TAG_NAMES`](@ref) and [`PROP_TAG_MACRO_NAMES`](@ref) together. Return the tag whose macro name is identical to `name`.
 3. No macro name matches: return `nothing`.

The comparison is `===` on a `Symbol`, so a macro whose name merely resembles a tag never matches.

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

# Algorithm

 1. Return `true` when all three hold: `x` is an `Expr`, its head is `:macrocall`, and [`prop_tag`](@ref) of its first argument is not `nothing`.
 2. Return `false` otherwise.

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

# Algorithm

 1. Make `tags`, an empty `Set{Symbol}`.
 2. While [`is_prop_tag_call`](@ref) of `expr` holds, push [`prop_tag`](@ref) of its first argument onto `tags`, and replace `expr` with its **last** argument, which is the expression the tag wraps.
 3. Return `tags` and the peeled `expr`.

`tags` is a set, so a tag written twice on one field is recorded once. The loop stops at the first node that is not a tag call, so a non-tag macro between two tags hides the tags below it.

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

# Algorithm

The channel is read first, and then the tag, because one tag has two transforms.

 1. `channel` is `:obs`:
     1. `tag` is `:wprop`: return `nothing_scalar_array_getindex(xf, thread...)`. The field **is** the weights, so it is indexed to the selected observations. Indexing keeps the `AbstractWeights` subtype, which a view would not.
     2. `tag` is `:fprop`: return `obs_weights_view(xf, thread...)`. The field is a composed child, so the verb recurses into it.
 2. `channel` is any other channel:
     1. `tag` is `:fprop`: return `factory_child(xf, thread..., args...; kwargs...)`.
     2. `tag` is `:vprop`: return `port_opt_view(xf, thread..., args...)`. This channel forwards no keywords.
     3. `tag` is `:pprop`: return `sel(xf, getproperty(pr, fname))`, which is why the field name is an argument. The prior result supplies the property of the **same name**.
     4. `tag` is `:cprop`: return `sel(xf, _ctx(args...))`, which reads the context out of the threaded arguments rather than the prior.
     5. `tag` is `:wprop`: return `_wprop(xf, args...; kwargs...)`, which **replaces** the field with an incoming [`ObsWeights`](@ref).
 3. No branch matched: raise an error naming the tag and the channel, and ask for a branch here.

Steps 1.1 and 2.5 are the same tag with two transforms, so the channel decides what `@wprop` means. Every emitted name is qualified against `mod`, because the expansion is escaped into the caller's module.

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

# Algorithm

 1. Read the `gate` tuple of the channel from [`PROP_TAG_CHANNELS`](@ref).
 2. Return `true` when at least one tag of `gate` has a non-empty entry in `tagged`, and `false` otherwise.

The `precedence` tuple is **not** read here, so a tag that a channel consults but does not gate on never makes that channel emit a method on its own. The `obs` channel gates on `@wprop` alone and consults `@fprop`, so a type carrying `@fprop` and no `@wprop` gains no `obs` method.

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

# Algorithm

 1. Read the `precedence` tuple of the channel from [`PROP_TAG_CHANNELS`](@ref).
 2. For each field name `fname` of `all_fields`, in declaration order:
     1. Build `xf`, the expression `obj.fname` that reads the field off the incoming struct.
     2. Find `idx`, the position of the **first** tag of `precedence` that `fname` carries.
     3. When `idx` is `nothing`, the field carries no tag of this channel: the value is `xf` itself.
     4. Otherwise the value is [`prop_tag_expr`](@ref) of that tag, in this channel.
     5. Push `Expr(:kw, fname, value)` onto `pairs`.
 3. Return `pairs`.

Step 2.2 is where the precedence decides one field's transform. A field carrying `@pprop` and `@fprop` takes the `@pprop` transform on the `prior` channel and the `@fprop` transform on the `factory` channel, because the two channels order the tags differently.

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

# Algorithm

 1. Return `true` when `x` is a `GlobalRef` whose `name` is `Symbol("@doc")`.
 2. Return `true` when `x` is equal to `Symbol("@doc")`.
 3. Return `false` otherwise.

Both spellings are needed for the same reason [`prop_tag`](@ref) needs both: a struct body written by hand carries the bare `Symbol`, and one that another macro has already expanded carries the `GlobalRef`.

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

# Algorithm

 1. `expr` is a `Symbol`: return it.
 2. `expr` is an `Expr` whose head is `:(::)`: return its first argument, which is the field name.
 3. `expr` is anything else: raise an error naming the expression.

Step 3 is what separates this function from [`try_field_name`](@ref), which returns `nothing` in the same case. A tag states that the node **is** a field, so a node that is not one is a defect in the struct body and not a node to skip.

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

# Algorithm

 1. `expr` is not an `Expr`: raise an error naming its type.
 2. `expr` has head `:struct`: return `expr` itself and `identity`, which rebuilds nothing.
 3. `expr` has head `:macrocall`: take `inner`, its **last** argument, and call this function again on it. That call gives `struct_node` and `rebuild`. Read `prefix`, every argument of `expr` except the last. Return `struct_node` and the function `s -> Expr(:macrocall, prefix..., rebuild(s))`.
 4. `expr` has any other head: raise an error naming the head.

Step 3 rebuilds the chain from the inside out, so a struct wrapped in several macros comes back wrapped in the same macros, in the same order, with the same arguments. The prefix carries the macro's own arguments and its `LineNumberNode`, so nothing of the call is lost.

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

# Algorithm

 1. `n` is a `Symbol`: return it.
 2. `n` has head `:curly`: call this function again on its first argument, which drops the type parameters.
 3. `n` has head `:<:`: call this function again on its first argument, which drops the supertype.
 4. `n` is anything else: raise an error naming the expression.

Steps 2 and 3 compose, so `Name{T} <: Super` peels the supertype first and then the parameters.

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

# Algorithm

 1. `expr` is a `Symbol`: return it.
 2. `expr` has head `:(::)` **and** its first argument is a `Symbol`: return that argument.
 3. `expr` is anything else: return `nothing`.

Step 2 tests the first argument as well as the head, which [`extract_field_name`](@ref) does not. A node such as `::Type`, which annotates no name, therefore gives `nothing` here and reaches step 3.

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

# Algorithm

 1. Make `tagged`, one empty vector per tag of [`PROP_TAG_NAMES`](@ref); `all_fields`, an empty vector; and `new_args`, an empty vector for the stripped body.
 2. For each node `arg` of the struct body, in declaration order, take one of three branches:
     1. `arg` is a `@doc` macrocall, which is how a documented field parses. Peel the tags off `inner`, its last argument.
         1. The field carries at least one tag: record the field name under each of its tags and in `all_fields`, then push a rebuilt `@doc` node whose last argument is the **stripped** field.
         2. The field carries no tag: record its name in `all_fields` when [`try_field_name`](@ref) finds one, and push `arg` unchanged.
     2. `arg` is a tag macrocall with no docstring: peel the tags, record the field name under each of them and in `all_fields`, and push the stripped field expression.
     3. `arg` is anything else — a `LineNumberNode`, an untagged field, an inner constructor: record its name in `all_fields` when [`try_field_name`](@ref) finds one, and push `arg` unchanged.
 3. Return `tagged`, `all_fields`, and the new body as one `:block` expression.

`all_fields` holds **every** declared field, tagged or not, and it is what makes the generated constructor call name every keyword. The returned body carries no tag, so the wrapped macros and Julia itself never see one.

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

# Algorithm

The tag never expands. [`@propagatable`](@ref) runs first and consumes it, so the steps below are what happens to the tagged field, not what this macro does.

 1. [`propagatable_parse_body`](@ref) peels the tag off the field, records the field name under `:fprop`, and puts the **stripped** field into the struct body. Neither Julia nor a wrapped macro such as `@concrete` ever sees the tag.

 2. [`prop_channel_active`](@ref) reads the recorded name. The channels this tag gates are the `factory` channel and the `obs` channel, and each active channel makes [`@propagatable`](@ref) emit one method.

 3. [`prop_channel_pairs`](@ref) builds the keyword pair of the field for each emitted method, and [`prop_tag_expr`](@ref) gives the value:

      + `factory` channel: `factory_child(x.field, args...; kwargs...)`, which recurses into the child.
      + `obs` channel: `obs_weights_view(x.field, i)`, which recurses into the child on the observation axis.

 4. The generated method rebuilds the struct with its keyword constructor, so every validation the constructor carries runs again on the propagated value.

Step 3 is the whole meaning of the tag. The same tag has two transforms, because a channel decides what a tag means. The `obs` channel does not gate on `@fprop`, so a struct whose only tag is `@fprop` gains no [`obs_weights_view`](@ref) method; the tag is consulted there only when a sibling field carries [`@wprop`](@ref).

This macro body itself raises an error. It is reached only when the tag is written outside a [`@propagatable`](@ref) struct body, where nothing consumed it.

# Related

  - [`@propagatable`](@ref)
  - [`@vprop`](@ref)
  - [`@wprop`](@ref)
  - [`factory`](@ref)
  - [`factory_child`](@ref)
  - [`obs_weights_view`](@ref)
  - [`PROP_TAG_CHANNELS`](@ref)
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

# Algorithm

The tag never expands. [`@propagatable`](@ref) runs first and consumes it, so the steps below are what happens to the tagged field, not what this macro does.

 1. [`propagatable_parse_body`](@ref) peels the tag off the field, records the field name under `:vprop`, and puts the **stripped** field into the struct body. Neither Julia nor a wrapped macro such as `@concrete` ever sees the tag.
 2. [`prop_channel_active`](@ref) reads the recorded name. The channels this tag gates are the `view` channel alone, and each active channel makes [`@propagatable`](@ref) emit one method.
 3. [`prop_channel_pairs`](@ref) builds the keyword pair of the field for each emitted method, and [`prop_tag_expr`](@ref) gives the value: `port_opt_view(x.field, i, args...)`. The channel forwards the threaded tail and **no** keywords.
 4. The generated method rebuilds the struct with its keyword constructor, so every validation the constructor carries runs again on the propagated value.

Step 3 is the whole meaning of the tag. `@vprop` appears in one channel, so it carries one transform and no channel can give it a second meaning. The index that the method threads selects **assets**; the observation axis has its own verb, [`obs_weights_view`](@ref).

This macro body itself raises an error. It is reached only when the tag is written outside a [`@propagatable`](@ref) struct body, where nothing consumed it.

# Related

  - [`@propagatable`](@ref)
  - [`@fprop`](@ref)
  - [`port_opt_view`](@ref)
  - [`PROP_TAG_CHANNELS`](@ref)
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

# Algorithm

The tag never expands. [`@propagatable`](@ref) runs first and consumes it, so the steps below are what happens to the tagged field, not what this macro does.

 1. [`propagatable_parse_body`](@ref) peels the tag off the field, records the field name under `:pprop`, and puts the **stripped** field into the struct body. Neither Julia nor a wrapped macro such as `@concrete` ever sees the tag.
 2. [`prop_channel_active`](@ref) reads the recorded name. The channels this tag gates are the `prior` channel alone, and each active channel makes [`@propagatable`](@ref) emit one method.
 3. [`prop_channel_pairs`](@ref) builds the keyword pair of the field for each emitted method, and [`prop_tag_expr`](@ref) gives the value: `sel(x.field, getproperty(pr, :field))`. The prior result supplies the property of the **same name** as the field, so the tag names no source of its own.
 4. The generated method rebuilds the struct with its keyword constructor, so every validation the constructor carries runs again on the propagated value.

Step 3 is the whole meaning of the tag. `@pprop` is first in the `prior` channel's precedence, so a field carrying both `@pprop` and [`@fprop`](@ref) takes the prior transform on that channel and the factory transform on the `factory` channel. ADR 0012 owns that rule.

This macro body itself raises an error. It is reached only when the tag is written outside a [`@propagatable`](@ref) struct body, where nothing consumed it.

# Related

  - [`@propagatable`](@ref)
  - [`@cprop`](@ref)
  - [`@wprop`](@ref)
  - [`factory`](@ref)
  - [`PROP_TAG_CHANNELS`](@ref)
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

# Algorithm

The tag never expands. [`@propagatable`](@ref) runs first and consumes it, so the steps below are what happens to the tagged field, not what this macro does.

 1. [`propagatable_parse_body`](@ref) peels the tag off the field, records the field name under `:cprop`, and puts the **stripped** field into the struct body. Neither Julia nor a wrapped macro such as `@concrete` ever sees the tag.
 2. [`prop_channel_active`](@ref) reads the recorded name. The channels this tag gates are the `prior` channel alone, and each active channel makes [`@propagatable`](@ref) emit one method.
 3. [`prop_channel_pairs`](@ref) builds the keyword pair of the field for each emitted method, and [`prop_tag_expr`](@ref) gives the value: `sel(x.field, _ctx(args...))`. `_ctx` finds the value **by type** in the threaded tail, so the source is an argument and not the prior result.
 4. The generated method rebuilds the struct with its keyword constructor, so every validation the constructor carries runs again on the propagated value.

Step 3 is the whole meaning of the tag. `@cprop` follows [`@pprop`](@ref) in the `prior` channel's precedence, and the two are mutually exclusive on one field. ADR 0012 owns that rule.

This macro body itself raises an error. It is reached only when the tag is written outside a [`@propagatable`](@ref) struct body, where nothing consumed it.

# Related

  - [`@propagatable`](@ref)
  - [`@pprop`](@ref)
  - [`factory`](@ref)
  - [`PROP_TAG_CHANNELS`](@ref)
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

# Algorithm

The tag never expands. [`@propagatable`](@ref) runs first and consumes it, so the steps below are what happens to the tagged field, not what this macro does.

 1. [`propagatable_parse_body`](@ref) peels the tag off the field, records the field name under `:wprop`, and puts the **stripped** field into the struct body. Neither Julia nor a wrapped macro such as `@concrete` ever sees the tag.

 2. [`prop_channel_active`](@ref) reads the recorded name. The channels this tag gates are the `factory` channel and the `obs` channel, and each active channel makes [`@propagatable`](@ref) emit one method.

 3. [`prop_channel_pairs`](@ref) builds the keyword pair of the field for each emitted method, and [`prop_tag_expr`](@ref) gives the value:

      + `factory` channel: `_wprop(x.field, args...; kwargs...)`, which **replaces** the field with an incoming [`ObsWeights`](@ref) and keeps it when none is threaded.
      + `obs` channel: `nothing_scalar_array_getindex(x.field, i)`, which **indexes** the value already there to the selected observations. Indexing rather than viewing is what keeps the `AbstractWeights` subtype.

 4. The generated method rebuilds the struct with its keyword constructor, so every validation the constructor carries runs again on the propagated value.

Step 3 is the whole meaning of the tag. **The two channels do different things to the same field**, which is the one place a reader learns that [`factory`](@ref) and [`obs_weights_view`](@ref) are not two names for one operation. `@wprop` is also the only tag that gates the `obs` channel, so a field opts a struct into [`obs_weights_view`](@ref) by carrying this tag and no second one.

This macro body itself raises an error. It is reached only when the tag is written outside a [`@propagatable`](@ref) struct body, where nothing consumed it.

# Related

  - [`@propagatable`](@ref)
  - [`@fprop`](@ref)
  - [`@pprop`](@ref)
  - [`factory`](@ref)
  - [`_wprop`](@ref)
  - [`obs_weights_view`](@ref)
  - [`ObsWeights`](@ref)
  - [`PROP_TAG_CHANNELS`](@ref)
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

The three tables are arguments, and each one defaults to the table the module ships. The
shipped tables are complete, so the call the module makes never reports a violation; a caller
that passes a table of its own drives each of the three clauses and reads the message it gives.

# Algorithm

 1. Make `violations`, an empty vector of strings.
 2. For each tag of [`PROP_TAG_NAMES`](@ref), with its macro name from [`PROP_TAG_MACRO_NAMES`](@ref):
     1. The macro name is not defined in this module: push a message that the tag declares no stub macro.
     2. The tag appears in the `precedence` of no channel of [`PROP_TAG_CHANNELS`](@ref): push a message that the tag appears in no channel.
     3. For each channel whose `precedence` names the tag, call [`prop_tag_expr`](@ref) with the probe name `:probe`. When that call raises, push a message naming the tag and the channel.
 3. `violations` is not empty: throw an `ArgumentError` listing every one of them.
 4. Return `nothing`.

Step 2.3 probes **each** channel that names the tag, not the tag alone. This is what catches a tag added to a second channel with no transform there, which the whole-tag probe of an earlier design let through.

# Arguments

  - `tags`: The tag names to check.
  - `macro_names`: The stub macro name of each tag, in the order of `tags`.
  - `channels`: The channel table whose `precedence` tuples are read.
  - `mod::Module`: The module the stub macros are looked up in, and the module that qualifies the names [`prop_tag_expr`](@ref) emits.

# Returns

  - `nothing`: Every row of `tags` is complete.

# Related

  - [`PROP_TAG_NAMES`](@ref)
  - [`PROP_TAG_CHANNELS`](@ref)
  - [`prop_tag_expr`](@ref)
  - [`check_propagatable_contracts`](@ref)
  - [`@propagatable`](@ref)
"""
function check_prop_tag_macros(tags = PROP_TAG_NAMES, macro_names = PROP_TAG_MACRO_NAMES,
                               channels = PROP_TAG_CHANNELS, mod::Module = @__MODULE__)
    violations = String[]
    for (tag, macro_name) in zip(tags, macro_names)
        if !isdefined(mod, macro_name)
            push!(violations, "`:$(tag)` declares no `$(macro_name)` stub macro.")
        end
        if !any(tag in channel.precedence for channel in channels)
            push!(violations, "`:$(tag)` appears in no channel of `PROP_TAG_CHANNELS`.")
        end
        for channel in keys(channels)
            if !(tag in getproperty(channels, channel).precedence)
                continue
            end
            try
                prop_tag_expr(channel, tag, :probe, :probe, mod, ())
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

# Algorithm

 1. Push the pair `(T, pprops)` onto [`PROPAGATABLE_CONTRACTS`](@ref).
 2. Return `nothing`.

`T` is `@nospecialize`d, so one method serves every registered type and the registration costs no compilation.

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

# Algorithm

 1. Make `kws`, an empty vector of symbols.
 2. For each method `m` of the constructor `T`, append `Base.kwarg_decl(m)` to `kws`. The union runs over every outer constructor, so a keyword that any one of them names counts.
 3. Remove the repeats from `kws`.
 4. Remove every name whose string ends in `...`, which is how `Base.kwarg_decl` reports a slurp.
 5. Return `kws`.

Step 4 is the whole point of the function. A constructor that carries `kwargs...` reports the slurp as a keyword name, and counting it would let every field satisfy the contract.

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

# Algorithm

 1. Make `msgs`, an empty vector of strings.
 2. Read `kws`, the keywords of the outer constructors of `T`, with [`propagatable_keywords`](@ref).
 3. For each field name of `T` that is absent from `kws`, push a message naming the type, the field and the [`suggest_declared_key`](@ref) suggestion drawn from `kws`.
 4. For each name of `pprops` that is absent from `pool`, push a message naming the type, the field and the suggestion drawn from `pool`.
 5. Return `msgs`.

Every clause is collected, and none stops the walk, so one call reports every violation of one type at once. An empty result means that the type's contract holds.

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

# Algorithm

 1. Find the struct with [`propagatable_find_struct`](@ref), which gives `struct_node` and `rebuild`, the function that puts a replacement struct back inside the same chain of wrapping macros.
 2. Read `type_head` and `body` off `struct_node`, and read `struct_name` off `type_head` with [`propagatable_bare_name`](@ref), which drops the type parameters and the supertype.
 3. Parse the body with [`propagatable_parse_body`](@ref), which gives `tagged`, the field names per tag; `all_fields`, every declared field in declaration order; and `new_body`, the body with every tag stripped.
 4. Build `new_struct` from `new_body`, and `chain` from `rebuild(new_struct)`. `chain` is the original declaration with the tags gone, so `@concrete` and Julia both see an ordinary struct.
 5. Bind `POMOD` to the module that **defines** the macro, and qualify every emitted name against it. A bare name would resolve in the caller's module, where `function factory(…)` declares a new function of the caller's own and the method never reaches `PortfolioOptimisers.factory`. That failure is silent, because the declaration compiles and the type never joins the propagation chain. ADR 0002, decision 4, owns the rule.
 6. Emit the `factory` method. When [`prop_channel_active`](@ref) holds for the `factory` channel, the body is a call to the keyword constructor whose pairs come from [`prop_channel_pairs`](@ref); otherwise the body is `x` itself. **This method is always emitted**, so an untagged [`@propagatable`](@ref) struct still answers [`factory`](@ref) with the identity.
 7. When the `view` channel is active, emit `port_opt_view(x::StructName, i, args...)`, whose channel threads `i` before `args...`.
 8. When the `obs` channel is active, emit `obs_weights_view(x::StructName, i)`, whose channel threads `i` and takes no tail.
 9. When the `prior` channel is active, emit `factory(x::StructName, pr::AbstractPriorResult, args...; kwargs...)`. Its body first binds `xr` to [`resolve_deferred_quantities`](@ref) of `x` against `pr`, and **every field is then read off `xr` rather than off `x`**, so a Deferred Quantity is a plain value before selection runs. This method is more specific than the one of step 6, so a call that threads a prior chooses it.
10. Build `pprop_tuple`, the `@pprop`-tagged field names as a tuple of quoted symbols.
11. Return one escaped block holding, in order: `Base.@__doc__ chain`, so a docstring on the declaration reaches the struct; the emitted methods; and the call to [`propagatable_register!`](@ref) that records the type and `pprop_tuple`.

Steps 6 to 9 differ only in the method head and in the arguments that the channel threads. Each reads its gate and its tag precedence off [`PROP_TAG_CHANNELS`](@ref), so a new channel is a row of that table, a branch in [`prop_tag_expr`](@ref) and a stub macro, rather than an edit at seven sites. ADR 0061 owns that rule.

# Related

  - [`@fprop`](@ref)
  - [`@vprop`](@ref)
  - [`@pprop`](@ref)
  - [`@cprop`](@ref)
  - [`@wprop`](@ref)
  - [`PROP_TAG_NAMES`](@ref)
  - [`PROP_TAG_CHANNELS`](@ref)
  - [`prop_channel_active`](@ref)
  - [`prop_channel_pairs`](@ref)
  - [`propagatable_parse_body`](@ref)
  - [`propagatable_register!`](@ref)
  - [`check_propagatable_contracts`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
  - [`obs_weights_view`](@ref)
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

Runs once at the end of the module, so the contract behind every generated method is enforced
where the structs are *declared* rather than at the first [`factory`](@ref) call. The
violations of every type are collected and reported together, because a run that stops at the
first one hides the rest. See [`propagatable_contract_violations`](@ref) for the two clauses.

Throws an [`ArgumentError`](https://docs.julialang.org/en/v1/base/base/#Core.ArgumentError)
listing every violation; the package refuses to precompile rather than shipping a type whose
generated methods throw on first use. A package that declares its own `@propagatable` types
calls this at the end of its own module to get the same guarantee.

# Algorithm

 1. Read `pool`, the property names that a prior result can carry, with [`prior_result_property_pool`](@ref).
 2. Make `msgs`, an empty vector of strings.
 3. For each pair `(T, pprops)` of [`PROPAGATABLE_CONTRACTS`](@ref), append the messages that [`propagatable_contract_violations`](@ref) reports for that type.
 4. `msgs` is not empty: throw an `ArgumentError` naming the count and listing every message.
 5. Return `nothing`.

Step 3 collects and never stops, so one run reports the violations of every registered type. The registry is filled by [`propagatable_register!`](@ref) at each declaration, so this function must run **after** the last one, which is why the module calls it at its end.

Both the registry and the pool are arguments, and each defaults to the value the module ships. A caller that passes a registry of its own reads the message a broken contract gives, without registering a broken type.

# Arguments

  - `contracts`: The pairs of a type and its `@pprop`-tagged field names to check.
  - `pool`: The property names a prior result can carry.

# Returns

  - `nothing`: Every pair of `contracts` satisfies the contract.

# Related

  - [`@propagatable`](@ref)
  - [`PROPAGATABLE_CONTRACTS`](@ref)
  - [`propagatable_contract_violations`](@ref)
  - [`@windowed_estimator`](@ref)
"""
function check_propagatable_contracts(contracts = PROPAGATABLE_CONTRACTS,
                                      pool = prior_result_property_pool())
    msgs = String[]
    for (T, pprops) in contracts
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

# Algorithm

 1. `v` is `nothing`: throw a [`PropertyPathError`](@ref) whose message names `pathstr`, the type `T` and the node `nodestr`.
 2. `v` is anything else: return `v`.

The guard runs on the **intermediate** hops only, so a path whose last hop gives `nothing` returns that `nothing` rather than raising. That is deliberate: an absent leaf is a value, and an absent intermediate is a path that cannot be walked.

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

# Algorithm

 1. `expr` is a `Symbol`: return the one-element vector holding it.
 2. `expr` is an `Expr` with head `:.` and two arguments:
     1. Read `leaf`, its second argument, and unwrap a `QuoteNode` to its value.
     2. `leaf` is not a `Symbol`: raise an error naming the leaf.
     3. Call this function again on the first argument, and append `leaf` to the result.
 3. `expr` is anything else: raise an error naming the expression.

The recursion of step 2.3 is what makes the path any depth: `a.b.c` parses as `(a.b).c`, so the walk descends to the bare name and rebuilds the path from the left.

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

# Algorithm

 1. The path holds one name: return `getfield(x, name)` and stop. `getfield` is used rather than `getproperty`, so the generated `Base.getproperty` never re-enters itself.
 2. Build `pathstr`, the whole path joined by dots, for the error message.
 3. Start `stmts` with `__v = getfield(x, first_name)`.
 4. For each further hop `k` of the path:
     1. Push `__v = forward_nonnothing(__v, struct_name, pathstr, nodestr)`, where `nodestr` names the part of the path walked so far.
     2. `k` is the last hop and `broadcast` is `true`: push an assignment that reads the leaf with `getproperty.` when `__v` is an `AbstractVector`, and with `getproperty` otherwise.
     3. Otherwise: push `__v = getproperty(__v, leaf)`.
 5. Push `__v` as the value of the block.
 6. Return the statements wrapped in a `let` block, so `__v` never escapes into the caller.

Step 4.1 runs before **every** hop after the first, so the guard covers each intermediate exactly once and never the leaf. Only the leaf of step 4.2 broadcasts, so an intermediate vector is still a path error rather than a silent map.

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

# Algorithm

 1. `block` is not a `begin … end` block: raise an error.
 2. Make three empty vectors: `swap_branches`, `getprop_branches` and `propname_contribs`.
 3. For each rule of the block, skipping a `LineNumberNode`:
     1. The rule is not a call: raise an error naming it.
     2. Read `marker`, the rule name, and `args`, its arguments. When the first argument is a `:parameters` node, read the `broadcast` option out of it and drop it from `args`. Any other option raises an error.
     3. `marker` is `forward`: flatten the locator with [`forward_flatten_path`](@ref) and build `walk` with [`forward_walk_expr`](@ref). With no further argument, push a branch that returns `getproperty(walk, sym)` when `sym` is in `propertynames(walk)`, and contribute every one of those names. With further arguments, check that each is a bare identifier, push a branch that matches `sym` against that name set, and contribute the named subset.
     4. `marker` is `alias`: check the exposed name, build `walk` from the locator, push a branch that matches the exposed name and returns `walk`, and contribute the name.
     5. `marker` is `compute`: check the exposed name. An anonymous-function source pushes a branch returning `fn(x)`, and `broadcast` with that form raises an error. A dotted source builds `walk` with the `broadcast` flag and pushes the matching branch. Any other source raises an error. Contribute the exposed name.
     6. `marker` is `swap`: as for `compute`, but a bare name is also a legal source, and the branch is pushed onto `swap_branches` rather than `getprop_branches`.
     7. `marker` is anything else: raise an error naming it.
 4. Build `Base.getproperty(x::T, sym::Symbol)` in this order: the `swap` branches; the own-field check, which returns `getfield(x, sym)`; the remaining branches in declaration order; and `getfield(x, sym)` as the fallthrough, which raises the standard error for an absent field.
 5. Build `Base.propertynames(x::T)` from `fieldnames(T)` followed by every contributed name, and return the unique names as a tuple.
 6. Return both definitions in one escaped block.

Step 4 is where the two orderings in the first paragraph come from: a `swap` runs **before** the own-field check, so it replaces a real field, and every other rule runs **after** it, so it can only add a name. Within each group the first branch that matches wins, and the order of the branches is the declaration order of the rules.

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

# Mathematical definition

```math
\\begin{align}
\\mathrm{MinValue}(\\boldsymbol{v}) &= \\underset{i}{\\min}\\ v_{i}\\,.
\\end{align}
```

Where:

  - $(math_dict[:v_reduce])
  - $(math_dict[:v_i_entry])

The reduction carries no weights, so a weighted call gives the same value as an unweighted one.

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

# Mathematical definition

```math
\\begin{align}
\\mathrm{MeanValue}(\\boldsymbol{v}) &= \\frac{1}{n} \\sum_{i=1}^{n} v_{i}\\,,
&&w = \\mathrm{nothing}\\,, \\\\
\\mathrm{MeanValue}(\\boldsymbol{v}) &= \\frac{\\sum_{i=1}^{n} w_{i} v_{i}}{\\sum_{i=1}^{n} w_{i}}\\,,
&&\\mathrm{otherwise}\\,.
\\end{align}
```

Where:

  - $(math_dict[:v_reduce])
  - $(math_dict[:v_i_entry])
  - ``w_{i}``: The ``i``-th observation weight, from the field `w`.

The weighted form normalises by the total weight, so a weight vector scaled by a positive constant gives the same value. `w` must carry one entry per entry of ``\\boldsymbol{v}``.

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

# Mathematical definition

```math
\\begin{align}
\\mathrm{MedianValue}(\\boldsymbol{v}) &= Q_{\\boldsymbol{v}}(0.5)\\,, \\\\
\\mathrm{MedianValue}(\\boldsymbol{v}) &= Q_{\\boldsymbol{v}, \\boldsymbol{w}}(0.5)\\,.
\\end{align}
```

Where:

  - $(math_dict[:v_reduce])
  - ``\\boldsymbol{w}``: The observation weights, from the field `w`. The first line is the case `w = nothing`.
  - ``Q_{\\boldsymbol{v}}(p)``: The ``p``-quantile of ``\\boldsymbol{v}``.
  - ``Q_{\\boldsymbol{v}, \\boldsymbol{w}}(p)``: The weighted ``p``-quantile of ``\\boldsymbol{v}``, as `StatsBase` defines it.

**Both forms are quantiles, and both interpolate.** Neither is an order statistic, so the result need not be an entry of ``\\boldsymbol{v}``. On a vector of even length the unweighted form averages the two middle entries, and the weighted form interpolates between the two entries that bracket half the weight mass. On ``\\boldsymbol{v} = [1, 2, 3, 4]`` the unweighted form gives ``2.5``, and the weighted form under ``\\boldsymbol{w} = [0.1, 0.2, 0.3, 0.4]`` gives ``2.8333``, which is not an entry of ``\\boldsymbol{v}``.

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

# Mathematical definition

```math
\\begin{align}
\\mathrm{MaxValue}(\\boldsymbol{v}) &= \\underset{i}{\\max}\\ v_{i}\\,.
\\end{align}
```

Where:

  - $(math_dict[:v_reduce])
  - $(math_dict[:v_i_entry])

The reduction carries no weights, so a weighted call gives the same value as an unweighted one.

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

Algorithm for reducing a vector of real values to its optionally weighted standard deviation. The unweighted default is safe and the weighted default is not: `corrected = true` under a plain `StatsBase.Weights` raises an `ArgumentError`, because that type declares no bias correction. Pass an `AnalyticWeights`, a `FrequencyWeights` or a `ProbabilityWeights`, or set `corrected = false`.

# Mathematical definition

```math
\\begin{align}
\\mathrm{StdValue}(\\boldsymbol{v}) &= \\sqrt{\\mathrm{VarValue}(\\boldsymbol{v})}\\,.
\\end{align}
```

Where:

  - $(math_dict[:v_reduce])
  - ``\\mathrm{VarValue}(\\boldsymbol{v})``: The variance under the same `w` and the same `corrected`, whose four denominators [`VarValue`](@ref) states.

`corrected` selects the denominator of the variance, and the square root carries that choice through. The unweighted default `corrected = true` divides by ``n - 1``.

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
  - [`VarValue`](@ref): the four denominators that `corrected` and the type of `w` select.
  - [`StandardisedValue`](@ref): reaches `Statistics.std` with a `mean` keyword, which is how it makes the deviation be taken about the mean that its `mv` produced.
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

Algorithm for reducing a vector of real values to its optionally weighted variance. The weighted default raises: a plain `StatsBase.Weights` declares no bias correction, so `corrected = true` under it raises an `ArgumentError` rather than returning a value. Pass one of the three corrected weight types below, or set `corrected = false`.

# Mathematical definition

```math
\\begin{align}
\\mathrm{VarValue}(\\boldsymbol{v}) &= \\frac{1}{d} \\sum_{i=1}^{n} w_{i} \\left(v_{i} - \\bar{v}\\right)^{2}\\,, \\\\
\\bar{v} &= \\frac{\\sum_{i=1}^{n} w_{i} v_{i}}{\\sum_{i=1}^{n} w_{i}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:v_reduce])
  - ``w_{i}``: The ``i``-th observation weight. The unweighted case is ``w_{i} = 1``.
  - ``\\bar{v}``: The mean of ``\\boldsymbol{v}`` under those weights.
  - ``d``: The denominator, which `corrected` and the **type** of `w` together select.

``d`` takes one of four values:

  - `w = nothing`: ``d = n - 1`` when `corrected` is `true`, and ``d = n`` when it is `false`.
  - `w::AnalyticWeights`: ``d = \\sum w_{i} - \\sum w_{i}^{2} / \\sum w_{i}`` when `corrected` is `true`.
  - `w::FrequencyWeights`: ``d = \\sum w_{i} - 1`` when `corrected` is `true`.
  - `w::ProbabilityWeights`: ``d = \\left(\\sum w_{i}\\right)(m - 1) / m`` when `corrected` is `true`, where ``m`` is the count of non-zero weights.

With `corrected = false` every weighted case takes ``d = \\sum w_{i}``.

``d`` is selected by the **type** of ``\\boldsymbol{w}`` and not by its values, so two numerically identical weight vectors of different types give different variances.

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

# Mathematical definition

```math
\\begin{align}
\\mathrm{SumValue}(\\boldsymbol{v}) &= \\sum_{i=1}^{n} v_{i}\\,.
\\end{align}
```

Where:

  - $(math_dict[:v_reduce])
  - $(math_dict[:v_i_entry])

The reduction carries no weights. [`MeanValue`](@ref) is the weighted sum normalised by the total weight, so a weighted sum is that value multiplied by the total weight.

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

# Mathematical definition

```math
\\begin{align}
\\mathrm{ProdValue}(\\boldsymbol{v}) &= \\prod_{i=1}^{n} v_{i}\\,.
\\end{align}
```

Where:

  - $(math_dict[:v_reduce])
  - $(math_dict[:v_i_entry])

The reduction carries no weights. One zero entry gives zero, and the product of many entries below one underflows, so this reduction is for a short vector of values near one.

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

# Mathematical definition

```math
\\begin{align}
\\mathrm{ModeValue}(\\boldsymbol{v}) &= \\underset{u \\in \\boldsymbol{v}}{\\arg\\max}\\ \\left| \\left\\{ i : v_{i} = u \\right\\} \\right|\\,.
\\end{align}
```

Where:

  - $(math_dict[:v_reduce])
  - ``u``: A value that ``\\boldsymbol{v}`` carries.
  - ``\\left| \\cdot \\right|``: The count of a set.

`StatsBase.mode` breaks a tie by the **first** value that reaches the highest count, so the result is a value of the input and never an average of two. The comparison is exact equality, so this reduction is for a vector of repeated exact values and not for a continuous one.

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

Algorithm for reducing a vector of real values to its optionally weighted mean divided by its optionally weighted standard deviation. A weighted [`factory`](@ref) call can make the reduction raise: `factory` replaces the `w` field of both `mv` and `sv` with the incoming [`ObsWeights`](@ref), and `sv` keeps its default `corrected = true`, which raises an `ArgumentError` under a plain `StatsBase.Weights`. Thread an `AnalyticWeights`, a `FrequencyWeights` or a `ProbabilityWeights`, or declare `sv = StdValue(; corrected = false)`.

# Mathematical definition

```math
\\begin{align}
z &= \\frac{\\hat{\\mu}}{\\tilde{\\sigma}}\\,, \\\\
\\tilde{\\sigma} &= \\begin{cases} 1 & \\hat{\\sigma} \\ \\mathrm{undefined} \\\\ \\sqrt{\\varepsilon} & \\hat{\\sigma} = 0 \\\\ \\hat{\\sigma} & \\mathrm{otherwise} \\end{cases}\\,.
\\end{align}
```

Where:

  - ``z``: Standardised value.
  - ``\\hat{\\mu}``: The value computed by `mv`.
  - ``\\hat{\\sigma}``: The value computed by `sv`, taken about ``\\hat{\\mu}``.
  - ``\\tilde{\\sigma}``: The guarded denominator.
  - ``\\varepsilon``: Machine epsilon of the element type of ``\\hat{\\sigma}``.

``\\hat{\\sigma}`` is undefined on a vector of one entry, because a corrected standard deviation needs two. The first case then gives ``\\tilde{\\sigma} = 1`` and ``z = \\hat{\\mu}``, so the reduction is defined on every non-empty vector.

# Algorithm

 1. Reduce `val` with `mv`, giving `m`.
 2. Reduce `val` with `sv`, and pass `m` as the `mean` keyword, giving `s`. The deviation is therefore always taken about the mean that step 1 produced, so weighting `mv` without weighting `sv` changes the denominator too.
 3. Guard `s`:
     1. `s` is `NaN`: replace it with `one(s)`.
     2. `s` is an exact zero: replace it with `sqrt(eps(eltype(s)))`. The test is an equality, so a small `s` is not guarded: on the constant vector `[2.0, 2.0, 2.0]` the result is `1.342e8`, which is `2 / sqrt(eps(Float64))`.
     3. Otherwise: keep `s`.
 4. Return `m / s`.

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

julia> PortfolioOptimisers.vec_to_real_measure(StandardisedValue(), [0.37])
0.37
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

# Algorithm

The method that Julia selects is the algorithm. `measure` names the reduction, and the type parameter of a weighted measure names the branch, `MeanValue{Nothing}` against `MeanValue{<:ObsWeights}`, so the branch is chosen at compile time and the field is never tested at run time.

 1. `measure` is a `Number`: return it, and read nothing of `val`.
 2. `measure` is a `Function`: return `measure(val)`.
 3. `measure` is a [`MinValue`](@ref), a [`MaxValue`](@ref), a [`SumValue`](@ref) or a [`ProdValue`](@ref): return `minimum`, `maximum`, `sum` or `prod` of `val`.
 4. `measure` is a [`ModeValue`](@ref): return `StatsBase.mode` of `val`.
 5. `measure` is a [`MeanValue`](@ref) or a [`MedianValue`](@ref): return `Statistics.mean` or `Statistics.median` of `val`, with the weights `measure.w` when the measure carries them. A tuple is `collect`ed first on the weighted branch.
 6. `measure` is a [`StdValue`](@ref) or a [`VarValue`](@ref): return `Statistics.std` or `Statistics.var` of `val`, with `corrected = measure.corrected`, with the weights `measure.w` when the measure carries them, and with `kwargs...` forwarded. A tuple is `collect`ed first on the weighted branch.
 7. `measure` is a [`StandardisedValue`](@ref): follow that type's own algorithm, which reduces twice and guards the denominator.

Step 1 is the case that makes a plain number a legal `measure`: a caller that already holds the value writes it where a reduction goes, and the seam needs no second signature.

# Arguments

  - `measure`: One of three things.

      + `::VectorToScalarMeasure`: The reduction to apply to `val`.
      + `::Number`: The value to return, whatever `val` holds.
      + `::Function`: Applied to `val` directly, as `measure(val)`.

  - `val`: A vector or tuple of real values to be reduced. A tuple is accepted wherever a vector is, and the weighted reductions `collect` it first, because `Statistics` needs an `AbstractVector` beside its weights. It is ignored when `measure` is a `Number`.

  - `kwargs...`: Forwarded to the underlying reduction. Only the [`StdValue`](@ref) and [`VarValue`](@ref) reductions read them.

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
    s = if isnan(s)
        one(s)
    elseif iszero(s)
        sqrt(eps(eltype(s)))
    else
        s
    end
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
