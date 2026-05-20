"""
    validate_bounds(lb, ub)

Validate that lower bounds do not exceed upper bounds.

Checks that all lower bound values are less than or equal to corresponding upper bound values. Throws an `ArgCheck` error if validation fails. Various overloads handle scalar, vector, and mixed combinations.

# Arguments

  - `lb`: Lower bound (scalar, vector, or `nothing`).
  - `ub`: Upper bound (scalar, vector, or `nothing`).

# Returns

  - `nothing`.
"""
function validate_bounds(lb::Number, ub::Number)::Nothing
    @argcheck(lb <= ub)
    return nothing
end
function validate_bounds(lb::VecNum, ub::Number)::Nothing
    @argcheck(!isempty(lb))
    @argcheck(all(x -> x <= ub, lb))
    return nothing
end
function validate_bounds(lb::Number, ub::VecNum)::Nothing
    @argcheck(!isempty(ub))
    @argcheck(all(x -> lb <= x, ub))
    return nothing
end
function validate_bounds(lb::VecNum, ub::VecNum)::Nothing
    @argcheck(!isempty(lb))
    @argcheck(!isempty(ub))
    @argcheck(length(lb) == length(ub))
    @argcheck(all(map((x, y) -> x <= y, lb, ub)))
    return nothing
end
function validate_bounds(lb::VecNum, ::Any)::Nothing
    @argcheck(!isempty(lb))
    return nothing
end
function validate_bounds(::Any, ub::VecNum)::Nothing
    @argcheck(!isempty(ub))
    return nothing
end
function validate_bounds(args...)::Nothing
    return nothing
end
"""
    weight_bounds_view(wb, i)

Get a view or subset of portfolio weight bounds for asset index `i`.

Returns a view of the weight bounds for the specified asset index `i`. If `wb` is `nothing`, returns `nothing`. For [`WeightBounds`](@ref) and [`WeightBoundsEstimator`](@ref), slices the bounds appropriately.

# Arguments

  - `wb`: Weight bounds object, estimator, or `nothing`.
  - `i`: Asset index or range to slice.

# Returns

  - Sliced weight bounds or `nothing`.

# Related

  - [`WeightBounds`](@ref)
  - [`WeightBoundsEstimator`](@ref)
"""
function weight_bounds_view(::Nothing, ::Any)::Nothing
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Container for lower and upper portfolio weight bounds.

`WeightBounds` stores the lower (`lb`) and upper (`ub`) bounds for portfolio weights, which can be scalars, vectors, or `nothing`. This type is used to represent weight constraints in portfolio optimisation problems, supporting both global and asset-specific bounds.

# Fields

  - `lb`: Lower bound(s) for portfolio weights.
  - `ub`: Upper bound(s) for portfolio weights.

# Constructors

    WeightBounds(
        lb::Option{<:Num_VecNum} = 0.0,
        ub::Option{<:Num_VecNum} = 1.0
    ) -> WeightBounds

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

  - [`w_neg_flag`](@ref)
  - [`w_finite_flag`](@ref)
  - [`set_weight_constraints!`](@ref)
  - [`set_linear_weight_constraints!`](@ref)
  - [`WeightBoundsEstimator`](@ref)
  - [`weight_bounds_constraints`](@ref)
"""
@concrete struct WeightBounds <: AbstractConstraintResult
    lb
    ub
    function WeightBounds(lb::Option{<:Num_VecNum}, ub::Option{<:Num_VecNum})::WeightBounds
        validate_bounds(lb, ub)
        return new{typeof(lb), typeof(ub)}(lb, ub)
    end
end
function WeightBounds(; lb::Option{<:Num_VecNum} = 0.0,
                      ub::Option{<:Num_VecNum} = 1.0)::WeightBounds
    return WeightBounds(lb, ub)
end
function weight_bounds_view(wb::WeightBounds, i)::WeightBounds
    lb = nothing_scalar_array_view(wb.lb, i)
    ub = nothing_scalar_array_view(wb.ub, i)
    return WeightBounds(; lb = lb, ub = ub)
end
"""
$(DocStringExtensions.TYPEDEF)

Estimator for portfolio weight bounds constraints.

`WeightBoundsEstimator` constructs lower (`lb`) and upper (`ub`) bounds for portfolio weights, supporting scalars, vectors, dictionaries, pairs, or custom constraint types. This estimator enables flexible specification of global, asset-specific, or algorithmic bounds for use in portfolio optimisation workflows.

# Fields

  - `lb`: Lower bound(s) for portfolio weights.
  - `ub`: Upper bound(s) for portfolio weights.
  - `dlb`: Default lower bound applied to assets when no specific lower bound is specified.
  - `dub`: Default upper bound applied to assets when no specific upper bound is specified.

# Constructors

    WeightBoundsEstimator(;
        lb::Option{<:EstValType} = nothing,
        ub::Option{<:EstValType} = nothing,
        dlb::Option{<:Number} = nothing,
        dub::Option{<:Number} = nothing
    ) -> WeightBoundsEstimator

## Validation

  - If `lb` or `ub` is a `AbstractDict` or `AbstractVector`, it must be non-empty.

# Details

  - If `lb` or `ub` is `nothing`, it indicates no constraint in that direction.
  - If `lb` or `ub` is not `nothing`, unspecified assets will use `dlb` or `dub` respectively. If these are also `nothing`, defaults to `0.0` for `dlb` and `1.0` for `dub`.

# Examples

```jldoctest
julia> WeightBoundsEstimator(; lb = Dict("A" => 0.1, "B" => 0.2),
                             ub = Dict("A" => 0.8, "B" => 0.9))
WeightBoundsEstimator
   lb ┼ Dict{String, Float64}: Dict("B" => 0.2, "A" => 0.1)
   ub ┼ Dict{String, Float64}: Dict("B" => 0.9, "A" => 0.8)
  dlb ┼ nothing
  dub ┴ nothing

julia> WeightBoundsEstimator(; lb = UniformValues(), ub = nothing)
WeightBoundsEstimator
   lb ┼ UniformValues()
   ub ┼ nothing
  dlb ┼ nothing
  dub ┴ nothing
```

# Related

  - [`WeightBounds`](@ref)
  - [`weight_bounds_constraints`](@ref)
"""
@concrete struct WeightBoundsEstimator <: AbstractConstraintEstimator
    lb
    ub
    dlb
    dub
    function WeightBoundsEstimator(lb::Option{<:EstValType}, ub::Option{<:EstValType},
                                   dlb::Option{<:Number} = nothing,
                                   dub::Option{<:Number} = nothing)::WeightBoundsEstimator
        if isa(lb, Dict_Vec)
            @argcheck(!isempty(lb), IsEmptyError)
        end
        if isa(ub, Dict_Vec)
            @argcheck(!isempty(ub), IsEmptyError)
        end
        if isa(lb, VecNum) && isa(ub, VecNum)
            @argcheck(length(lb) == length(ub))
            validate_bounds(lb, ub)
        elseif isa(lb, Num_VecNum) && isa(ub, Num_VecNum)
            validate_bounds(lb, ub)
        end
        if !isnothing(dlb) && !isnothing(dub)
            @argcheck(dlb <= dub)
        end
        return new{typeof(lb), typeof(ub), typeof(dlb), typeof(dub)}(lb, ub, dlb, dub)
    end
end
function WeightBoundsEstimator(; lb::Option{<:EstValType} = nothing,
                               ub::Option{<:EstValType} = nothing,
                               dlb::Option{<:Number} = nothing,
                               dub::Option{<:Number} = nothing)::WeightBoundsEstimator
    return WeightBoundsEstimator(lb, ub, dlb, dub)
end
function weight_bounds_view(wb::WeightBoundsEstimator, i)::WeightBoundsEstimator
    lb = nothing_scalar_array_view(wb.lb, i)
    ub = nothing_scalar_array_view(wb.ub, i)
    return wb = WeightBoundsEstimator(; lb = lb, ub = ub, dlb = wb.dlb, dub = wb.dub)
end
"""
    const WbE_Wb = Union{<:WeightBoundsEstimator, <:WeightBounds}

Alias for a weight bounds estimator or result.

Matches either a [`WeightBoundsEstimator`](@ref) (specifying how to generate weight bounds constraints) or a [`WeightBounds`](@ref) result. Used internally for dispatch in weight bounds constraint generation.

# Related

  - [`WeightBoundsEstimator`](@ref)
  - [`WeightBounds`](@ref)
  - [`weight_bounds_constraints`](@ref)
"""
const WbE_Wb = Union{<:WeightBoundsEstimator, <:WeightBounds}
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
                                   kwargs...)::WeightBounds
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

  - `wb::VecNum`: Vector of length `N` filled with `val`.

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

  - `wb::VecNum`: Vector of length `N` filled with `wb` or `val`.

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
    weight_bounds_constraints_side(wb::VecNum, args...)

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
function weight_bounds_constraints_side(wb::VecNum, args...)
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
                                   kwargs...)::WeightBounds
    return WeightBounds(; lb = weight_bounds_constraints_side(wb.lb, N, -Inf),
                        ub = weight_bounds_constraints_side(wb.ub, N, Inf))
end
"""
    weight_bounds_constraints(wb::WeightBounds{<:VecNum, <:VecNum}, args...;
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
function weight_bounds_constraints(wb::WeightBounds{<:VecNum, <:VecNum}, args...;
                                   kwargs...)::WeightBounds
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
function weight_bounds_constraints(wb::Nothing, args...; N::Integer = 0,
                                   kwargs...)::WeightBounds
    return WeightBounds(; lb = fill(-Inf, N), ub = fill(Inf, N))
end

export WeightBoundsEstimator, WeightBounds, weight_bounds_constraints
