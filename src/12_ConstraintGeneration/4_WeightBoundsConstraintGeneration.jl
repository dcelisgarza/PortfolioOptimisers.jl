function validate_bounds(lb::Real, ub::Real)
    @argcheck(lb <= ub)
    return nothing
end
function validate_bounds(lb::AbstractVector, ub::Real)
    @argcheck(!isempty(lb) && all(x -> x <= ub, lb))
    return nothing
end
function validate_bounds(lb::Real, ub::AbstractVector)
    @argcheck(!isempty(ub) && all(x -> lb <= x, ub))
    return nothing
end
function validate_bounds(lb::AbstractVector, ub::AbstractVector)
    @argcheck(!isempty(lb) &&
              !isempty(ub) &&
              length(lb) == length(ub) &&
              all(map((x, y) -> x <= y, lb, ub)))
    return nothing
end
function validate_bounds(lb::AbstractVector, ::Any)
    @argcheck(!isempty(lb))
    return nothing
end
function validate_bounds(::Any, ub::AbstractVector)
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
```julia
struct WeightBounds{T1, T2} <: AbstractConstraintResult
    lb::T1
    ub::T2
end
```

Container for lower and upper portfolio weight bounds.

`WeightBounds` stores the lower (`lb`) and upper (`ub`) bounds for portfolio weights, which can be scalars, vectors, or `nothing`. This type is used to represent weight constraints in portfolio optimisation problems, supporting both global and asset-specific bounds.

# Fields

  - `lb`: Lower bound(s) for portfolio weights.
  - `ub`: Upper bound(s) for portfolio weights.

# Constructor

```julia
WeightBounds(lb::Union{Nothing, <:Real, <:AbstractVector{<:Real}},
             ub::Union{Nothing, <:Real, <:AbstractVector{<:Real}})
```

## Validation

  - `all(lb .<= ub)`.

# Details

  - If `lb` or `ub` is `nothing`, it indicates no bound in that direction.
  - Supports scalar bounds (same for all assets) or vector bounds (asset-specific).

# Examples

```jldoctest
julia> WeightBounds(0.0, 1.0)
WeightBounds
  lb | Float64: 0.0
  ub | Float64: 1.0

julia> WeightBounds([0.0, 0.1], [0.8, 1.0])
WeightBounds
  lb | Vector{Float64}: [0.0, 0.1]
  ub | Vector{Float64}: [0.8, 1.0]
```

# Related

  - [`WeightBoundsEstimator`](@ref)
  - [`weight_bounds_constraints`](@ref)
"""
struct WeightBounds{T1, T2} <: AbstractConstraintResult
    lb::T1
    ub::T2
    function WeightBounds(lb::Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                          ub::Union{Nothing, <:Real, <:AbstractVector{<:Real}})
        validate_bounds(lb, ub)
        return new{typeof(lb), typeof(ub)}(lb, ub)
    end
end
function WeightBounds(; lb::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = 0.0,
                      ub::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = 1.0)
    return WeightBounds(lb, ub)
end
function weight_bounds_view(wb::WeightBounds, i::AbstractVector)
    lb = nothing_scalar_array_view(wb.lb, i)
    ub = nothing_scalar_array_view(wb.ub, i)
    return WeightBounds(; lb = lb, ub = ub)
end
"""
```julia
abstract type CustomWeightBoundsConstraint <: AbstractConstraintEstimator end
```

Abstract supertype for custom portfolio weight bounds constraints.

`CustomWeightBoundsConstraint` provides an interface for implementing user-defined or algorithmic portfolio weight bounds. Subtypes can encode advanced or non-standard weight constraints, such as scaled, grouped, or dynamically computed bounds, for use in portfolio optimisation workflows.

# Related

  - [`WeightBoundsEstimator`](@ref)
  - [`WeightBounds`](@ref)
  - [`UniformlyDistributedBounds`](@ref)
"""
abstract type CustomWeightBoundsConstraint <: AbstractConstraintEstimator end
"""
```julia
struct UniformlyDistributedBounds <: CustomWeightBoundsConstraint end
```

Custom weight bounds constraint for uniformly distributing asset weights, `1/N` for lower bounds and `1` for upper bounds, where `N` is the number of assets.

# Examples

```jldoctest
julia> sets = AssetSets(; dict = Dict("nx" => ["A", "B", "C"]));

julia> PortfolioOptimisers.get_weight_bounds(UniformlyDistributedBounds(), true, sets)
0.3333333333333333
```

# Related

  - [`CustomWeightBoundsConstraint`](@ref)
  - [`WeightBoundsEstimator`](@ref)
  - [`WeightBounds`](@ref)
"""
struct UniformlyDistributedBounds <: CustomWeightBoundsConstraint end
"""
```julia
struct WeightBoundsEstimator{T1, T2} <: AbstractConstraintEstimator
    lb::T1
    ub::T2
end
```

Estimator for portfolio weight bounds constraints.

`WeightBoundsEstimator` constructs lower (`lb`) and upper (`ub`) bounds for portfolio weights, supporting scalars, vectors, dictionaries, pairs, or custom constraint types. This estimator enables flexible specification of global, asset-specific, or algorithmic bounds for use in portfolio optimisation workflows.

# Fields

  - `lb`: Lower bound(s) for portfolio weights.
  - `ub`: Upper bound(s) for portfolio weights.

# Constructor

```julia
WeightBoundsEstimator(;
                      lb::Union{Nothing, <:Real, <:AbstractDict,
                                <:Pair{<:AbstractString, <:Real},
                                <:AbstractVector{<:Union{<:Real,
                                                         <:Pair{<:AbstractString, <:Real}}},
                                <:CustomWeightBoundsConstraint} = nothing,
                      ub::Union{Nothing, <:Real, <:AbstractDict,
                                <:Pair{<:AbstractString, <:Real},
                                <:AbstractVector{<:Union{<:Real,
                                                         <:Pair{<:AbstractString, <:Real}}},
                                <:CustomWeightBoundsConstraint} = nothing)
```

## Validation

  - If `lb` or `ub` is a dictionary or vector, it must be non-empty.

# Details

  - If `lb` or `ub` is `nothing`, it indicates no bound in that direction.

# Examples

```jldoctest
julia> WeightBoundsEstimator(; lb = 0.0, ub = 1.0)
WeightBoundsEstimator
  lb | Float64: 0.0
  ub | Float64: 1.0

julia> WeightBoundsEstimator(; lb = Dict("A" => 0.1, "B" => 0.2),
                             ub = Dict("A" => 0.8, "B" => 0.9))
WeightBoundsEstimator
  lb | Dict{String, Float64}: Dict("B" => 0.2, "A" => 0.1)
  ub | Dict{String, Float64}: Dict("B" => 0.9, "A" => 0.8)

julia> WeightBoundsEstimator(; lb = UniformlyDistributedBounds(), ub = nothing)
WeightBoundsEstimator
  lb | UniformlyDistributedBounds()
  ub | nothing
```

# Related

  - [`WeightBounds`](@ref)
  - [`CustomWeightBoundsConstraint`](@ref)
  - [`UniformlyDistributedBounds`](@ref)
  - [`weight_bounds_constraints`](@ref)
"""
struct WeightBoundsEstimator{T1, T2} <: AbstractConstraintEstimator
    lb::T1
    ub::T2
    function WeightBoundsEstimator(lb::Union{Nothing, <:Real, <:AbstractDict,
                                             <:Pair{<:AbstractString, <:Real},
                                             <:AbstractVector{<:Union{<:Real,
                                                                      <:Pair{<:AbstractString,
                                                                             <:Real}}},
                                             <:CustomWeightBoundsConstraint},
                                   ub::Union{Nothing, <:Real, <:AbstractDict,
                                             <:Pair{<:AbstractString, <:Real},
                                             <:AbstractVector{<:Union{<:Real,
                                                                      <:Pair{<:AbstractString,
                                                                             <:Real}}},
                                             <:CustomWeightBoundsConstraint})
        if isa(lb, Union{<:AbstractDict, <:AbstractVector})
            @argcheck(!isempty(lb), IsEmptyError(non_empty_msg("`lb`") * "."))
        end
        if isa(ub, Union{<:AbstractDict, <:AbstractVector})
            @argcheck(!isempty(ub), IsEmptyError(non_empty_msg("`ub`") * "."))
        end
        return new{typeof(lb), typeof(ub)}(lb, ub)
    end
end
function WeightBoundsEstimator(;
                               lb::Union{Nothing, <:Real, <:AbstractDict,
                                         <:Pair{<:AbstractString, <:Real},
                                         <:AbstractVector{<:Union{<:Real,
                                                                  <:Pair{<:AbstractString,
                                                                         <:Real}}},
                                         <:CustomWeightBoundsConstraint} = nothing,
                               ub::Union{Nothing, <:Real, <:AbstractDict,
                                         <:Pair{<:AbstractString, <:Real},
                                         <:AbstractVector{<:Union{<:Real,
                                                                  <:Pair{<:AbstractString,
                                                                         <:Real}}},
                                         <:CustomWeightBoundsConstraint} = nothing)
    return WeightBoundsEstimator(lb, ub)
end
function weight_bounds_view(wb::Union{<:AbstractString, Expr,
                                      <:AbstractVector{<:AbstractString},
                                      <:AbstractVector{Expr},
                                      <:AbstractVector{<:Union{<:AbstractString, Expr}},
                                      <:WeightBoundsEstimator}, ::Any)
    return wb
end
"""
```julia
get_weight_bounds(wb::Union{Nothing, <:Real, <:AbstractVector{<:Real}}, args...; kwargs...)
```

Extracts portfolio weight bounds from a scalar, vector, or `nothing`.

`get_weight_bounds` returns the input value unchanged when the weight bounds are specified as a scalar, vector, or `nothing`. This method is used internally to propagate simple bound specifications in portfolio optimisation workflows.

# Arguments

  - `wb`: Lower or upper bound(s) for portfolio weights.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `wb`: The input value, unchanged.

# Related

  - [`WeightBoundsEstimator`](@ref)
  - [`WeightBounds`](@ref)
  - [`weight_bounds_constraints`](@ref)
"""
function get_weight_bounds(wb::Union{Nothing, <:Real, <:AbstractVector{<:Real}}, args...;
                           kwargs...)
    return wb
end
"""
```julia
get_weight_bounds(wb::Union{<:AbstractDict, <:Pair{<:AbstractString, <:Real},
                            <:AbstractVector{<:Pair{<:AbstractString, <:Real}}}, lub::Bool,
                  sets::AssetSets; strict::Bool = false, datatype::DataType = Float64)
```

Extracts portfolio weight bounds from a dictionary, pair, or vector of pairs.

`get_weight_bounds` converts asset-specific bound specifications into a vector of bounds aligned with the assets in `sets`. This method is used to map named or indexed bounds to the corresponding assets for portfolio optimisation.

# Arguments

  - `wb`: Asset-specific bounds as a dictionary.

  - `lub`: Boolean flag indicating whether the bounds are lower or upper.

      + `true`: lower bounds
      + `false`: upper bounds.
  - `sets`: [`AssetSets`](@ref) object containing asset names or indices.
  - `strict`: If `true`, throws error enforcing strict matching between assets and bounds, else throws a warning.
  - `datatype`: Output data type for bounds.

# Returns

  - `wb::Vector{<:Real}`: Vector of bounds aligned with the assets in `sets`.

# Related

  - [`WeightBoundsEstimator`](@ref)
  - [`WeightBounds`](@ref)
  - [`AssetSets`](@ref)
  - [`weight_bounds_constraints`](@ref)
"""
function get_weight_bounds(wb::Union{<:AbstractDict, <:Pair{<:AbstractString, <:Real},
                                     <:AbstractVector{<:Pair{<:AbstractString, <:Real}}},
                           lub::Bool, sets::AssetSets; strict::Bool = false,
                           datatype::DataType = Float64)
    return estimator_to_val(wb, sets, ifelse(lub, zero(datatype), one(datatype));
                            strict = strict)
end
"""
```julia
get_weight_bounds(wb::UniformlyDistributedBounds, lub::Bool, sets::AssetSets;
                  datatype::DataType = Float64, kwargs...)
```

Extracts uniformly distributed portfolio weight bounds.

`get_weight_bounds` returns a uniform bound for all assets in `sets` when using [`UniformlyDistributedBounds`](@ref). For lower bounds (`lub = true`), it returns `1/N` where `N` is the number of assets; for upper bounds (`lub = false`), it returns `1` (or `one(datatype)`).

# Arguments

  - `wb`: An instance of [`UniformlyDistributedBounds`](@ref).

  - `lub`: Boolean flag indicating whether the bounds are lower or upper.

      + `true`: lower bounds
      + `false`: upper bounds.
  - `sets`: [`AssetSets`](@ref) object containing asset names or indices.
  - `datatype`: Output data type for bounds.
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `wb::Real`: `1/N` for lower, `1` for upper as `datatype`.

# Related

  - [`UniformlyDistributedBounds`](@ref)
  - [`WeightBoundsEstimator`](@ref)
  - [`WeightBounds`](@ref)
  - [`AssetSets`](@ref)
  - [`weight_bounds_constraints`](@ref)
"""
function get_weight_bounds(wb::UniformlyDistributedBounds, lub::Bool, sets::AssetSets;
                           datatype::DataType = Float64, kwargs...)
    return lub ? inv(length(sets.dict[sets.key])) : one(datatype)
end
"""
    weight_bounds_constraints(wb::WeightBoundsEstimator, sets::AssetSets;
                             strict::Bool = false, datatype::DataType = Float64,
                             kwargs...)

Generate portfolio weight bounds constraints from a `WeightBoundsEstimator` and asset set.

`weight_bounds_constraints` constructs a [`WeightBounds`](@ref) object representing lower and upper portfolio weight bounds for the assets in `sets`, using the specifications in `wb`. Supports scalar, vector, dictionary, pair, or custom constraint types for flexible bound assignment.

# Arguments

  - `wb`: [`WeightBoundsEstimator`](@ref) specifying lower and upper bounds.
  - `sets`: [`AssetSets`](@ref) containing asset names or indices.
  - `strict`: If `true`, enforces strict matching between assets and bounds (throws error on mismatch); if `false`, issues a warning.
  - `datatype`: Output data type for bounds (default: `Float64`).
  - `kwargs...`: Additional keyword arguments passed to bound extraction routines.

# Returns

  - `wb::WeightBounds`: Object containing lower and upper bounds aligned with `sets`.

# Details

  - Lower and upper bounds are extracted using [`get_weight_bounds`](@ref), mapped to assets in `sets`.
  - Supports composable and asset-specific constraints.
  - If a bound is `nothing`, indicates no constraint in that direction.

# Examples

```jldoctest
julia> sets = AssetSets(; dict = Dict("nx" => ["A", "B", "C"]));

julia> wb = WeightBoundsEstimator(; lb = Dict("A" => 0.1, "B" => 0.2), ub = 1.0);

julia> weight_bounds_constraints(wb, sets)
WeightBounds
  lb | Vector{Float64}: [0.1, 0.2, 0.0]
  ub | Float64: 1.0
```

# Related

  - [`WeightBoundsEstimator`](@ref)
  - [`WeightBounds`](@ref)
  - [`get_weight_bounds`](@ref)
  - [`AssetSets`](@ref)
"""
function weight_bounds_constraints(wb::WeightBoundsEstimator, sets::AssetSets;
                                   strict::Bool = false, datatype::DataType = Float64,
                                   kwargs...)
    return WeightBounds(;
                        lb = get_weight_bounds(wb.lb, true, sets; strict = strict,
                                               datatype = datatype),
                        ub = get_weight_bounds(wb.ub, false, sets; strict = strict,
                                               datatype = datatype))
end
"""
    weight_bounds_constraints_side(::Nothing, N::Integer, val::Real)

Generate a vector of portfolio weight bounds when no constraint is specified.

`weight_bounds_constraints_side` returns a vector of length `N` filled with the value `val` when the input bound is `nothing`. This is used to represent unconstrained portfolio weights (e.g., `-Inf` for lower bounds, `Inf` for upper bounds) in constraint generation routines.

# Arguments

  - `::Nothing`: Indicates no constraint for this bound direction.
  - `N`: Number of assets (length of the output vector).
  - `val`: Value to fill (typically `-Inf` or `Inf`).

# Returns

  - `wb::Vector{<:Real}`: Vector of length `N` filled with `val`.

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
function weight_bounds_constraints_side(::Nothing, N::Integer, val::Real)
    return fill(val, N)
end
"""
    weight_bounds_constraints_side(wb::Real, N::Integer, val::Real)

Generate a vector of portfolio weight bounds from a scalar bound.

`weight_bounds_constraints_side` returns a vector of length `N` filled with `val` if `wb` is infinite, or a vector of length `N` with all elements equal to `wb` otherwise. This is used to propagate scalar portfolio weight bounds to all assets in constraint generation routines.

# Arguments

  - `wb::Real`: Scalar bound for portfolio weights (can be finite or infinite).
  - `N::Integer`: Number of assets (length of the output vector).
  - `val::Real`: Value to fill if `wb` is infinite (typically `-Inf` or `Inf`).

# Returns

  - `wb::Union{<:Vector{<:Real}, <:StepRangeLen}`: Vector of length `N` filled with `wb` or `val`.

# Examples

```jldoctest
julia> PortfolioOptimisers.weight_bounds_constraints_side(0.1, 3, -Inf)
3-element StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}:
 0.1
 0.1
 0.1

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
function weight_bounds_constraints_side(wb::Real, N::Integer, val::Real)
    return if isinf(wb)
        fill(val, N)
    elseif isa(wb, Real)
        range(; start = wb, stop = wb, length = N)
    end
end
"""
    weight_bounds_constraints_side(wb::AbstractVector, args...)

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
function weight_bounds_constraints_side(wb::AbstractVector, args...)
    return wb
end
"""
    weight_bounds_constraints(wb::WeightBounds{<:Any, <:Any}, args...;
                             scalar::Bool = false, N::Integer = 0, kwargs...)

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
  lb | StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}: StepRangeLen(0.0, 0.0, 3)
  ub | StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}: StepRangeLen(1.0, 0.0, 3)

julia> weight_bounds_constraints(WeightBounds([0.1, 0.2, 0.3], 1.0); N = 3)
WeightBounds
  lb | Vector{Float64}: [0.1, 0.2, 0.3]
  ub | StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}: StepRangeLen(1.0, 0.0, 3)
```

# Related

  - [`WeightBounds`](@ref)
  - [`weight_bounds_constraints_side`](@ref)
  - [`WeightBoundsEstimator`](@ref)
"""
function weight_bounds_constraints(wb::WeightBounds{<:Any, <:Any}, args...;
                                   scalar::Bool = false, N::Integer = 0, kwargs...)
    if scalar || iszero(N)
        return wb
    end
    return WeightBounds(; lb = weight_bounds_constraints_side(wb.lb, N, -Inf),
                        ub = weight_bounds_constraints_side(wb.ub, N, Inf))
end
"""
    weight_bounds_constraints(wb::WeightBounds{<:AbstractVector, <:AbstractVector},
                             args...; kwargs...)

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
  lb | Vector{Float64}: [0.1, 0.2, 0.3]
  ub | Vector{Float64}: [0.8, 0.9, 1.0]
```

# Related

  - [`WeightBounds`](@ref)
  - [`WeightBoundsEstimator`](@ref)
  - [`weight_bounds_constraints_side`](@ref)
  - [`weight_bounds_constraints`](@ref)
"""
function weight_bounds_constraints(wb::WeightBounds{<:AbstractVector, <:AbstractVector},
                                   args...; kwargs...)
    return wb
end
"""
    weight_bounds_constraints(wb::Nothing, args...; scalar::Bool = false,
                             N::Integer = 0, kwargs...)

Generate unconstrained portfolio weight bounds when no bounds are specified.

`weight_bounds_constraints` returns a [`WeightBounds`](@ref) object with lower bounds set to `-Inf` and upper bounds set to `Inf` for all assets when `wb` is `nothing`. If `scalar` is `true` or `N` is zero, returns scalar bounds; otherwise, returns vectors of length `N` filled with `-Inf` and `Inf`.

# Arguments

  - `wb::Nothing`: Indicates no constraint for portfolio weights.
  - `args...`: Additional positional arguments (ignored).
  - `scalar::Bool`: If `true`, return scalar bounds (`-Inf`, `Inf`).
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
  lb | Vector{Float64}: [-Inf, -Inf, -Inf]
  ub | Vector{Float64}: [Inf, Inf, Inf]

julia> weight_bounds_constraints(nothing; scalar = true)
WeightBounds
  lb | Float64: -Inf
  ub | Float64: Inf
```

# Related

  - [`WeightBounds`](@ref)
  - [`WeightBoundsEstimator`](@ref)
  - [`weight_bounds_constraints_side`](@ref)
"""
function weight_bounds_constraints(wb::Nothing, args...; scalar::Bool = false,
                                   N::Integer = 0, kwargs...)
    if scalar || iszero(N)
        return WeightBounds(; lb = -Inf, ub = Inf)
    end
    return WeightBounds(; lb = fill(-Inf, N), ub = fill(Inf, N))
end

export WeightBoundsEstimator, WeightBounds, weight_bounds_constraints,
       UniformlyDistributedBounds
