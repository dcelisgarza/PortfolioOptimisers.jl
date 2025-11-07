function validate_bounds(lb::Number, ub::Number)
    @argcheck(lb <= ub)
    return nothing
end
function validate_bounds(lb::NumVec, ub::Number)
    @argcheck(!isempty(lb))
    @argcheck(all(x -> x <= ub, lb))
    return nothing
end
function validate_bounds(lb::Number, ub::NumVec)
    @argcheck(!isempty(ub))
    @argcheck(all(x -> lb <= x, ub))
    return nothing
end
function validate_bounds(lb::NumVec, ub::NumVec)
    @argcheck(!isempty(lb))
    @argcheck(!isempty(ub))
    @argcheck(length(lb) == length(ub))
    @argcheck(all(map((x, y) -> x <= y, lb, ub)))
    return nothing
end
function validate_bounds(lb::NumVec, ::Any)
    @argcheck(!isempty(lb))
    return nothing
end
function validate_bounds(::Any, ub::NumVec)
    @argcheck(!isempty(ub))
    return nothing
end
function validate_bounds(args...)
    return nothing
end
function weight_bounds_view(::Nothing, ::Any)
    return nothing
end
"""
    struct WeightBounds{T1, T2} <: AbstractConstraintResult
        lb::T1
        ub::T2
    end

Container for lower and upper portfolio weight bounds.

`WeightBounds` stores the lower (`lb`) and upper (`ub`) bounds for portfolio weights, which can be scalars, vectors, or `nothing`. This type is used to represent weight constraints in portfolio optimisation problems, supporting both global and asset-specific bounds.

# Fields

  - `lb`: Lower bound(s) for portfolio weights.
  - `ub`: Upper bound(s) for portfolio weights.

# Constructor

    WeightBounds(lb::Option{<:NumUNumVec},
                 ub::Option{<:NumUNumVec})

## Validation

  - `all(lb .<= ub)`.

# Details

  - If `lb` or `ub` is `nothing`, it indicates no bound in that direction.
  - Supports scalar bounds (same for all assets) or vector bounds (asset-specific).

# Examples

```jldoctest
julia> WeightBounds(0.0, 1.0)
WeightBounds
  lb ┼ Float64: 0.0
  ub ┴ Float64: 1.0

julia> WeightBounds([0.0, 0.1], [0.8, 1.0])
WeightBounds
  lb ┼ Vector{Float64}: [0.0, 0.1]
  ub ┴ Vector{Float64}: [0.8, 1.0]
```

# Related

  - [`WeightBoundsEstimator`](@ref)
  - [`weight_bounds_constraints`](@ref)
"""
struct WeightBounds{T1, T2} <: AbstractConstraintResult
    lb::T1
    ub::T2
    function WeightBounds(lb::Option{<:NumUNumVec}, ub::Option{<:NumUNumVec})
        validate_bounds(lb, ub)
        return new{typeof(lb), typeof(ub)}(lb, ub)
    end
end
function WeightBounds(; lb::Option{<:NumUNumVec} = 0.0, ub::Option{<:NumUNumVec} = 1.0)
    return WeightBounds(lb, ub)
end
function weight_bounds_view(wb::WeightBounds, i)
    lb = nothing_scalar_array_view(wb.lb, i)
    ub = nothing_scalar_array_view(wb.ub, i)
    return WeightBounds(; lb = lb, ub = ub)
end
"""
    abstract type CustomWeightBoundsConstraint <: AbstractConstraintEstimator end

Abstract supertype for custom portfolio weight bounds constraints.

`CustomWeightBoundsConstraint` provides an interface for implementing user-defined or algorithmic portfolio weight bounds. Subtypes can encode advanced or non-standard weight constraints, such as scaled, grouped, or dynamically computed bounds, for use in portfolio optimisation workflows.

# Related

  - [`WeightBoundsEstimator`](@ref)
  - [`WeightBounds`](@ref)
  - [`UniformlyDistributedBounds`](@ref)
"""
abstract type CustomWeightBoundsConstraint <: AbstractConstraintEstimator end
function nothing_scalar_array_view(x::CustomWeightBoundsConstraint, ::Any)
    return x
end
"""
    struct UniformlyDistributedBounds <: CustomWeightBoundsConstraint end

Custom weight bounds constraint for uniformly distributing asset weights, `1/N` for lower bounds and `1` for upper bounds, where `N` is the number of assets.

# Examples

```jldoctest
julia> sets = AssetSets(; dict = Dict("nx" => ["A", "B", "C"]));

julia> PortfolioOptimisers.estimator_to_val(UniformlyDistributedBounds(), sets)
0.3333333333333333
```

# Related

  - [`CustomWeightBoundsConstraint`](@ref)
  - [`WeightBoundsEstimator`](@ref)
  - [`WeightBounds`](@ref)
"""
struct UniformlyDistributedBounds <: CustomWeightBoundsConstraint end
function estimator_to_val(::UniformlyDistributedBounds, sets::AssetSets, args...;
                          datatype::DataType = Float64, kwargs...)
    return datatype(inv(length(sets.dict[sets.key])))
end
"""
    struct WeightBoundsEstimator{T1, T2} <: AbstractConstraintEstimator
        lb::T1
        ub::T2
    end

Estimator for portfolio weight bounds constraints.

`WeightBoundsEstimator` constructs lower (`lb`) and upper (`ub`) bounds for portfolio weights, supporting scalars, vectors, dictionaries, pairs, or custom constraint types. This estimator enables flexible specification of global, asset-specific, or algorithmic bounds for use in portfolio optimisation workflows.

# Fields

  - `lb`: Lower bound(s) for portfolio weights.
  - `ub`: Upper bound(s) for portfolio weights.

# Constructor

    WeightBoundsEstimator(;
                          lb::Union{Nothing, <:EstValType,
                                    <:CustomWeightBoundsConstraint} = nothing,
                          ub::Union{Nothing, <:EstValType,
                                    <:CustomWeightBoundsConstraint} = nothing)

## Validation

  - If `lb` or `ub` is a `AbstractDict` or `AbstractVector`, it must be non-empty.

# Details

  - If `lb` or `ub` is `nothing`, it indicates no bound in that direction.

# Examples

```jldoctest
julia> WeightBoundsEstimator(; lb = Dict("A" => 0.1, "B" => 0.2),
                             ub = Dict("A" => 0.8, "B" => 0.9))
WeightBoundsEstimator
   lb ┼ Dict{String, Float64}: Dict("B" => 0.2, "A" => 0.1)
   ub ┼ Dict{String, Float64}: Dict("B" => 0.9, "A" => 0.8)
  dlb ┼ nothing
  dub ┴ nothing

julia> WeightBoundsEstimator(; lb = UniformlyDistributedBounds(), ub = nothing)
WeightBoundsEstimator
   lb ┼ UniformlyDistributedBounds()
   ub ┼ nothing
  dlb ┼ nothing
  dub ┴ nothing
```

# Related

  - [`WeightBounds`](@ref)
  - [`CustomWeightBoundsConstraint`](@ref)
  - [`UniformlyDistributedBounds`](@ref)
  - [`weight_bounds_constraints`](@ref)
"""
struct WeightBoundsEstimator{T1, T2, T3, T4} <: AbstractConstraintEstimator
    lb::T1
    ub::T2
    dlb::T3
    dub::T4
    function WeightBoundsEstimator(lb::Union{Nothing, <:EstValType,
                                             <:CustomWeightBoundsConstraint},
                                   ub::Union{Nothing, <:EstValType,
                                             <:CustomWeightBoundsConstraint},
                                   dlb::Option{<:Number} = nothing,
                                   dub::Option{<:Number} = nothing)
        if isa(lb, Union{<:AbstractDict, <:AbstractVector})
            @argcheck(!isempty(lb), IsEmptyError)
        end
        if isa(ub, Union{<:AbstractDict, <:AbstractVector})
            @argcheck(!isempty(ub), IsEmptyError)
        end
        if !isnothing(dlb) && !isnothing(dub)
            @argcheck(dlb <= dub)
        end
        return new{typeof(lb), typeof(ub), typeof(dlb), typeof(dub)}(lb, ub, dlb, dub)
    end
end
function WeightBoundsEstimator(;
                               lb::Union{Nothing, <:EstValType,
                                         <:CustomWeightBoundsConstraint} = nothing,
                               ub::Union{Nothing, <:EstValType,
                                         <:CustomWeightBoundsConstraint} = nothing,
                               dlb::Option{<:Number} = nothing,
                               dub::Option{<:Number} = nothing)
    return WeightBoundsEstimator(lb, ub, dlb, dub)
end
function weight_bounds_view(wb::WeightBoundsEstimator, i)
    lb = nothing_scalar_array_view(wb.lb, i)
    ub = nothing_scalar_array_view(wb.ub, i)
    return wb = WeightBoundsEstimator(; lb = lb, ub = ub, dlb = wb.dlb, dub = wb.dub)
end
"""
    weight_bounds_constraints(wb::WeightBoundsEstimator, sets::AssetSets; strict::Bool = false,
                              datatype::DataType = Float64, kwargs...)

Generate portfolio weight bounds constraints from a `WeightBoundsEstimator` and asset set.

`weight_bounds_constraints` constructs a [`WeightBounds`](@ref) object representing lower and upper portfolio weight bounds for the assets in `sets`, using the specifications in `wb`. Supports scalar, vector, dictionary, pair, or custom constraint types for flexible bound assignment.

# Arguments

  - `wb`: [`WeightBoundsEstimator`](@ref) specifying lower and upper bounds.
  - `sets`: [`AssetSets`](@ref) containing asset names or indices.
  - `strict`: If `true`, enforces strict matching between assets and bounds (throws error on mismatch); if `false`, issues a warning.
  - `datatype`: Output data type for bounds.
  - `kwargs...`: Additional keyword arguments passed to bound extraction routines.

# Returns

  - `wb::WeightBounds`: Object containing lower and upper bounds aligned with `sets`.

# Details

  - Lower and upper bounds are extracted using [`estimator_to_val`](@ref), mapped to assets in `sets`.
  - Supports composable and asset-specific constraints.
  - If a bound is `nothing`, indicates no constraint in that direction.

# Examples

```jldoctest
julia> sets = AssetSets(; dict = Dict("nx" => ["A", "B", "C"]));

julia> wb = WeightBoundsEstimator(; lb = Dict("A" => 0.1, "B" => 0.2), ub = 1.0);

julia> weight_bounds_constraints(wb, sets)
WeightBounds
  lb ┼ Vector{Float64}: [0.1, 0.2, 0.0]
  ub ┴ Float64: 1.0
```

# Related

  - [`WeightBoundsEstimator`](@ref)
  - [`WeightBounds`](@ref)
  - [`estimator_to_val`](@ref)
  - [`AssetSets`](@ref)
"""
function weight_bounds_constraints(wb::WeightBoundsEstimator, sets::AssetSets;
                                   strict::Bool = false, datatype::DataType = Float64,
                                   kwargs...)
    return WeightBounds(;
                        lb = estimator_to_val(wb.lb, sets,
                                              ifelse(isnothing(wb.dlb), zero(datatype),
                                                     wb.dlb); datatype = datatype,
                                              strict = strict),
                        ub = estimator_to_val(wb.ub, sets,
                                              ifelse(isnothing(wb.dub), one(datatype),
                                                     wb.dub); datatype = datatype,
                                              strict = strict))
end
"""
    weight_bounds_constraints_side(::Nothing, N::Integer, val::Number)

Generate a vector of portfolio weight bounds when no constraint is specified.

`weight_bounds_constraints_side` returns a vector of length `N` filled with the value `val` when the input bound is `nothing`. This is used to represent unconstrained portfolio weights (e.g., `-Inf` for lower bounds, `Inf` for upper bounds) in constraint generation routines.

# Arguments

  - `::Nothing`: Indicates no constraint for this bound direction.
  - `N`: Number of assets (length of the output vector).
  - `val`: Value to fill (typically `-Inf` or `Inf`).

# Returns

  - `wb::Vector{<:Number}`: Vector of length `N` filled with `val`.

# Examples

```jldoctest
julia> PortfolioOptimisers.weight_bounds_constraints_side(nothing, 3, -Inf)
3-element Vector{Float64}:
 -Inf
 -Inf
 -Inf
```

# Related

  - [`weight_bounds_constraints`](@ref)
  - [`WeightBounds`](@ref)
"""
function weight_bounds_constraints_side(::Nothing, N::Integer, val::Number)
    return fill(val, N)
end
"""
    weight_bounds_constraints_side(wb::Number, N::Integer, val::Number)

Generate a vector of portfolio weight bounds from a scalar bound.

`weight_bounds_constraints_side` returns a vector of length `N` filled with `val` if `wb` is infinite, or a vector of length `N` with all elements equal to `wb` otherwise. This is used to propagate scalar portfolio weight bounds to all assets in constraint generation routines.

# Arguments

  - `wb::Number`: Scalar bound for portfolio weights (can be finite or infinite).
  - `N::Integer`: Number of assets (length of the output vector).
  - `val::Number`: Value to fill if `wb` is infinite (typically `-Inf` or `Inf`).

# Returns

  - `wb::Union{<:Vector{<:Number}, <:StepRangeLen}`: Vector of length `N` filled with `wb` or `val`.

# Examples

```jldoctest
julia> PortfolioOptimisers.weight_bounds_constraints_side(0.1, 3, -Inf)
StepRangeLen(0.1, 0.0, 3)

julia> PortfolioOptimisers.weight_bounds_constraints_side(Inf, 3, -Inf)
3-element Vector{Float64}:
 -Inf
 -Inf
 -Inf
```

# Related

  - [`weight_bounds_constraints`](@ref)
  - [`WeightBounds`](@ref)
  - [`weight_bounds_constraints_side`](@ref)
"""
function weight_bounds_constraints_side(wb::Number, N::Integer, val::Number)
    return if isinf(wb)
        fill(val, N)
    else
        range(wb, wb; length = N)
    end
end
"""
    weight_bounds_constraints_side(wb::NumVec, args...)

Propagate asset-specific portfolio weight bounds from a vector.

`weight_bounds_constraints_side` returns the input vector `wb` unchanged when asset-specific bounds are provided as a vector. This method is used to propagate explicit per-asset bounds in constraint generation routines.

# Arguments

  - `wb`: Vector of bounds for portfolio weights (one per asset).
  - `args...`: Additional positional arguments (ignored).

# Returns

  - `wb::AbstractVector`: The input vector, unchanged.

# Examples

```jldoctest
julia> PortfolioOptimisers.weight_bounds_constraints_side([0.1, 0.2, 0.3])
3-element Vector{Float64}:
 0.1
 0.2
 0.3
```

# Related

  - [`weight_bounds_constraints`](@ref)
  - [`WeightBounds`](@ref)
  - [`weight_bounds_constraints_side`](@ref)
"""
function weight_bounds_constraints_side(wb::NumVec, args...)
    return wb
end
"""
    weight_bounds_constraints(wb::WeightBounds{<:Any, <:Any}, args...; N::Integer = 0, kwargs...)

Propagate or expand portfolio weight bounds constraints from a `WeightBounds` object.

`weight_bounds_constraints` returns the input [`WeightBounds`](@ref) object unchanged if `scalar` is `true` or `N` is zero. Otherwise, it expands scalar or `nothing` bounds to vectors or ranges of length `N` using [`weight_bounds_constraints_side`](@ref), ensuring per-asset constraints are properly propagated.

# Arguments

  - `wb`: [`WeightBounds`](@ref) object containing lower and upper bounds.
  - `args...`: Additional positional arguments (ignored).
  - `scalar`: If `true`, treat bounds as scalar and return unchanged.
  - `N`: Number of assets (length for expansion; if zero, treat as scalar).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `wb::WeightBounds`: Expanded or unchanged bounds object.

# Details

  - If `scalar` is `true` or `N == 0`, returns `wb` unchanged.
  - Otherwise, expands `lb` and `ub` using [`weight_bounds_constraints_side`](@ref) to vectors of length `N`.

# Examples

```jldoctest
julia> weight_bounds_constraints(WeightBounds(0.0, 1.0); N = 3)
WeightBounds
  lb ┼ StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}: StepRangeLen(0.0, 0.0, 3)
  ub ┴ StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}: StepRangeLen(1.0, 0.0, 3)

julia> weight_bounds_constraints(WeightBounds([0.1, 0.2, 0.3], 1.0); N = 3)
WeightBounds
  lb ┼ Vector{Float64}: [0.1, 0.2, 0.3]
  ub ┴ StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}: StepRangeLen(1.0, 0.0, 3)
```

# Related

  - [`WeightBounds`](@ref)
  - [`weight_bounds_constraints_side`](@ref)
  - [`WeightBoundsEstimator`](@ref)
"""
function weight_bounds_constraints(wb::WeightBounds{<:Any, <:Any}, args...; N::Integer = 0,
                                   kwargs...)
    return WeightBounds(; lb = weight_bounds_constraints_side(wb.lb, N, -Inf),
                        ub = weight_bounds_constraints_side(wb.ub, N, Inf))
end
"""
    weight_bounds_constraints(wb::WeightBounds{<:NumVec, <:NumVec}, args...;
                              kwargs...)

Propagate asset-specific portfolio weight bounds constraints from a `WeightBounds` object with vector bounds.

`weight_bounds_constraints` returns the input [`WeightBounds`](@ref) object unchanged when both lower and upper bounds are provided as vectors. This method is used to propagate explicit per-asset bounds in constraint generation workflows, ensuring that asset-specific constraints are preserved.

# Arguments

  - `wb`: [`WeightBounds`](@ref) object with vector lower and upper bounds.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `wb::WeightBounds`: The input bounds object, unchanged.

# Examples

```jldoctest
julia> weight_bounds_constraints(WeightBounds([0.1, 0.2, 0.3], [0.8, 0.9, 1.0]))
WeightBounds
  lb ┼ Vector{Float64}: [0.1, 0.2, 0.3]
  ub ┴ Vector{Float64}: [0.8, 0.9, 1.0]
```

# Related

  - [`WeightBounds`](@ref)
  - [`WeightBoundsEstimator`](@ref)
  - [`weight_bounds_constraints_side`](@ref)
  - [`weight_bounds_constraints`](@ref)
"""
function weight_bounds_constraints(wb::WeightBounds{<:NumVec, <:NumVec}, args...; kwargs...)
    return wb
end
"""
    weight_bounds_constraints(wb::Nothing, args...; N::Integer = 0, kwargs...)

Generate unconstrained portfolio weight bounds when no bounds are specified.

`weight_bounds_constraints` returns a [`WeightBounds`](@ref) object with lower bounds set to `-Inf` and upper bounds set to `Inf` for all assets when `wb` is `nothing`.

# Arguments

  - `wb::Nothing`: Indicates no constraint for portfolio weights.
  - `args...`: Additional positional arguments (ignored).
  - `N::Integer`: Number of assets (length for expansion; if zero, treat as scalar).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `wb::WeightBounds`: Object with unconstrained lower and upper bounds.

# Details

  - If `scalar` is `true` or `N == 0`, returns `WeightBounds(-Inf, Inf)`.
  - Otherwise, returns `WeightBounds(fill(-Inf, N), fill(Inf, N))`.

# Examples

```jldoctest
julia> weight_bounds_constraints(nothing; N = 3)
WeightBounds
  lb ┼ Vector{Float64}: [-Inf, -Inf, -Inf]
  ub ┴ Vector{Float64}: [Inf, Inf, Inf]
```

# Related

  - [`WeightBounds`](@ref)
  - [`WeightBoundsEstimator`](@ref)
  - [`weight_bounds_constraints_side`](@ref)
"""
function weight_bounds_constraints(wb::Nothing, args...; N::Integer = 0, kwargs...)
    return WeightBounds(; lb = fill(-Inf, N), ub = fill(Inf, N))
end

export WeightBoundsEstimator, WeightBounds, weight_bounds_constraints,
       UniformlyDistributedBounds
