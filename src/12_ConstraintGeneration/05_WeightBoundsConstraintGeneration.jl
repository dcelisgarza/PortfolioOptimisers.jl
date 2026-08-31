"""
    validate_bounds(lb::Number, ub::Number) -> Nothing
    validate_bounds(lb::VecNum, ub::Number) -> Nothing
    validate_bounds(lb::Number, ub::VecNum) -> Nothing
    validate_bounds(lb::VecNum, ub::VecNum) -> Nothing
    validate_bounds(lb::VecNum, ::Any) -> Nothing
    validate_bounds(::Any, ub::VecNum) -> Nothing
    validate_bounds(args...) -> Nothing

Check that a lower bound does not exceed the corresponding upper bound.

Seven methods split the work by the pair of argument types, and the pair alone decides which preconditions run. Two of the seven exist to catch a `nothing` on one side: they check that the vector side is non-empty and compare nothing. The catch-all closes the family and checks nothing at all, so `(nothing, nothing)`, `(::Number, nothing)` and `(nothing, ::Number)` are accepted without a comparison.

# Arguments

  - $(arg_dict[:lb])
  - $(arg_dict[:ub])

# Validation

Each bullet names the method that runs the check, the condition it demands and the error it raises.

  - `(Number, Number)`: `lb <= ub`, `DomainError` otherwise.
  - `(VecNum, Number)`: `!isempty(lb)`, `IsEmptyError` otherwise. Every entry of `lb` is at most `ub`, `DomainError` otherwise.
  - `(Number, VecNum)`: `!isempty(ub)`, `IsEmptyError` otherwise. Every entry of `ub` is at least `lb`, `DomainError` otherwise.
  - `(VecNum, VecNum)`: `!isempty(lb)` and `!isempty(ub)`, `IsEmptyError` otherwise. `length(lb) == length(ub)`, `DimensionMismatch` otherwise. Entry by entry, `lb` is at most `ub`, `DomainError` otherwise. The length check runs first, so the entry-by-entry comparison never reads a truncated pair.
  - `(VecNum, Any)`: `!isempty(lb)`, `IsEmptyError` otherwise. This method takes the `(vector, nothing)` pair, so it compares the two sides with nothing.
  - `(Any, VecNum)`: `!isempty(ub)`, `IsEmptyError` otherwise. This method takes the `(nothing, vector)` pair, so it compares the two sides with nothing.
  - `(args...)`: no precondition and no raise. This method takes every remaining pair, of which `(nothing, nothing)`, `(::Number, nothing)` and `(nothing, ::Number)` are the ones the two bound types reach.

# Returns

  - `nothing`.

# Examples

```jldoctest
julia> isnothing(PortfolioOptimisers.validate_bounds(0.0, 1.0))
true

julia> isnothing(PortfolioOptimisers.validate_bounds(nothing, nothing))
true
```

# Related

  - [`WeightBounds`](@ref)
  - [`WeightBoundsEstimator`](@ref)
"""
function validate_bounds(lb::Number, ub::Number)::Nothing
    @argcheck(lb <= ub, DomainError("lb ($lb) must be <= ub ($ub)"))
    return nothing
end
function validate_bounds(lb::VecNum, ub::Number)::Nothing
    @argcheck(!isempty(lb), IsEmptyError("lb cannot be empty"))
    @argcheck(all(x -> x <= ub, lb),
              DomainError("all entries of lb must be <= ub ($ub), got lb = $lb"))
    return nothing
end
function validate_bounds(lb::Number, ub::VecNum)::Nothing
    @argcheck(!isempty(ub), IsEmptyError("ub cannot be empty"))
    @argcheck(all(x -> lb <= x, ub),
              DomainError("all entries of ub must be >= lb ($lb), got ub = $ub"))
    return nothing
end
function validate_bounds(lb::VecNum, ub::VecNum)::Nothing
    @argcheck(!isempty(lb), IsEmptyError("lb cannot be empty"))
    @argcheck(!isempty(ub), IsEmptyError("ub cannot be empty"))
    @argcheck(length(lb) == length(ub),
              DimensionMismatch("lb ($(length(lb))) and ub ($(length(ub))) must have the same length"))
    @argcheck(all(map((x, y) -> x <= y, lb, ub)),
              DomainError("all entries of lb must be <= corresponding entries of ub"))
    return nothing
end
function validate_bounds(lb::VecNum, ::Any)::Nothing
    @argcheck(!isempty(lb), IsEmptyError("lb cannot be empty"))
    return nothing
end
function validate_bounds(::Any, ub::VecNum)::Nothing
    @argcheck(!isempty(ub), IsEmptyError("ub cannot be empty"))
    return nothing
end
function validate_bounds(args...)::Nothing
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Bounds every portfolio weight between a lower and an upper limit.

A bound is a scalar shared by every asset, a vector of one limit per asset, or `nothing` for no limit in that direction. The bounds also serve the mixed-integer builders, which read them as the big-M that links a weight to its held indicator.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{l} \\leq \\boldsymbol{w} \\leq \\boldsymbol{u}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - ``\\boldsymbol{l}``: Lower bound vector ``N \\times 1``. An entry of ``-\\infty`` leaves that weight unbounded below, and an entry of ``+\\infty`` admits no weight at all.
  - ``\\boldsymbol{u}``: Upper bound vector ``N \\times 1``. An entry of ``+\\infty`` leaves that weight unbounded above, and an entry of ``-\\infty`` admits no weight at all.
  - $(math_dict[:N])

A scalar bound is the case in which every entry of the vector holds the same value, and a `nothing` bound is the case in which every entry of that side is the infinity that leaves the weight free.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    WeightBounds(
        lb::Option{<:Num_VecNum},
        ub::Option{<:Num_VecNum}
    ) -> WeightBounds
    WeightBounds(;
        lb::Option{<:Num_VecNum} = 0.0,
        ub::Option{<:Num_VecNum} = 1.0
    ) -> WeightBounds

Keywords correspond to the struct's fields. Only the keyword form carries the defaults, and it forwards to the positional form, so both run the same checks.

## Validation

[`validate_bounds`](@ref) runs on the pair, and the pair of types decides which of the checks below runs.

  - A vector bound is non-empty, `IsEmptyError` otherwise.
  - Two vector bounds have the same length, `DimensionMismatch` otherwise.
  - `lb` is not above `ub`, entry by entry where both are vectors, `DomainError` otherwise.
  - A `nothing` on one side leaves the other side's non-emptiness as the only check, because no comparison is possible.
  - Two `nothing` bounds are checked by nothing.
  - Two infinite scalar bounds of the same sign pass, because the comparison holds. `WeightBounds(; lb = Inf, ub = Inf)` and `WeightBounds(; lb = -Inf, ub = -Inf)` are both admitted, and both admit no weight.

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `lb`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `ub`: Sliced to the selected indices via [`port_opt_view`](@ref).

A field that [`weight_bounds_constraints`](@ref) expanded to a constant `range` is sliced as any other vector is, and the slice is a view of that range.

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
  - [`validate_bounds`](@ref): the seven methods that carry this type's checks.
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct WeightBounds <: AbstractConstraintResult
    """
    $(field_dict[:lb])
    """
    @vprop lb
    """
    $(field_dict[:ub])
    """
    @vprop ub
    function WeightBounds(lb::Option{<:Num_VecNum}, ub::Option{<:Num_VecNum})::WeightBounds
        validate_bounds(lb, ub)
        return new{typeof(lb), typeof(ub)}(lb, ub)
    end
end
function WeightBounds(; lb::Option{<:Num_VecNum} = 0.0,
                      ub::Option{<:Num_VecNum} = 1.0)::WeightBounds
    return WeightBounds(lb, ub)
end
"""
$(DocStringExtensions.TYPEDEF)

Resolves weight bounds written in asset or group names against a universe.

[`weight_bounds_constraints`](@ref) turns it into a [`WeightBounds`](@ref): every name is mapped to its indices in the universe, and an unnamed asset takes `dlb` or `dub`. A bound may also be a scalar, a vector, or an algorithmic rule such as [`UniformValues`](@ref). A `dlb` or a `dub` left at `nothing` is filled at resolution time, with `zero(datatype)` on the lower side and `one(datatype)` on the upper side.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    WeightBoundsEstimator(;
        lb::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}} = 0.0,
        ub::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}} = 1.0,
        dlb::Option{<:Number} = nothing,
        dub::Option{<:Number} = nothing
    ) -> WeightBoundsEstimator

Keywords correspond to the struct's fields.

## Validation

  - If `lb` or `ub` is a `AbstractDict` or `AbstractVector`, it must be non-empty, `IsEmptyError` otherwise.
  - Two vector bounds must have the same length, `DimensionMismatch` otherwise.
  - Where both bounds are numbers or vectors of numbers, `lb` is not above `ub`, through [`validate_bounds`](@ref), `DomainError` otherwise.
  - Where one side is a `AbstractDict`, a `Pair` or an algorithmic rule, the two sides are not compared here, because neither side is resolved yet. [`weight_bounds_constraints`](@ref) builds a [`WeightBounds`](@ref) from the resolved pair, and that constructor compares them.
  - If neither `dlb` nor `dub` is `nothing`, `dlb <= dub`, `DomainError` otherwise.

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `lb`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `ub`: Sliced to the selected indices via [`port_opt_view`](@ref).

Only a vector bound is sliced. A bound that is a scalar, a `Dict`, a `Pair` or an algorithmic rule is not indexed by asset, so a view passes it through untouched and it resolves against the viewed universe when the estimator runs.

# Examples

```jldoctest
julia> WeightBoundsEstimator(; lb = Dict(\"A\" => 0.1, \"B\" => 0.2),
                             ub = Dict(\"A\" => 0.8, \"B\" => 0.9))
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
  - [`validate_bounds`](@ref): the checks this constructor runs on a pair of numbers or a pair of vectors.
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct WeightBoundsEstimator <: AbstractConstraintEstimator
    """
    $(field_dict[:lb])
    """
    @vprop lb
    """
    $(field_dict[:ub])
    """
    @vprop ub
    """
    $(field_dict[:dlb])
    """
    dlb
    """
    $(field_dict[:dub])
    """
    dub
    function WeightBoundsEstimator(lb::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}},
                                   ub::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}},
                                   dlb::Option{<:Number} = nothing,
                                   dub::Option{<:Number} = nothing)::WeightBoundsEstimator
        if isa(lb, Dict_Vec)
            @argcheck(!isempty(lb), IsEmptyError)
        end
        if isa(ub, Dict_Vec)
            @argcheck(!isempty(ub), IsEmptyError)
        end
        if isa(lb, VecNum) && isa(ub, VecNum)
            @argcheck(length(lb) == length(ub),
                      DimensionMismatch("lb ($(length(lb))) and ub ($(length(ub))) must have the same length"))
            validate_bounds(lb, ub)
        elseif isa(lb, Num_VecNum) && isa(ub, Num_VecNum)
            validate_bounds(lb, ub)
        end
        if !isnothing(dlb) && !isnothing(dub)
            @argcheck(dlb <= dub, DomainError("dlb ($dlb) must be <= dub ($dub)"))
        end
        return new{typeof(lb), typeof(ub), typeof(dlb), typeof(dub)}(lb, ub, dlb, dub)
    end
end
function WeightBoundsEstimator(;
                               lb::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}} = 0.0,
                               ub::Option{<:EstValType{<:VectorAbstractEstimatorValueAlgorithm}} = 1.0,
                               dlb::Option{<:Number} = nothing,
                               dub::Option{<:Number} = nothing)::WeightBoundsEstimator
    return WeightBoundsEstimator(lb, ub, dlb, dub)
end
"""
    const WbE_Wb = Union{<:WeightBoundsEstimator, <:WeightBounds}

Alias for a weight bounds estimator or result.

Matches either a [`WeightBoundsEstimator`](@ref) (specifying how to generate weight bounds constraints) or a [`WeightBounds`](@ref) result. Used internally for dispatch in weight bounds constraint generation.

There is no vector counterpart, and [`weight_bounds_constraints`](@ref) has no vector method. Weight bounds are one box over the whole universe, so an optimiser holds exactly one. See [`RkbE_Rkb`](@ref) for why some constraint families are singular and others are not.

# Related

  - [`WeightBoundsEstimator`](@ref)
  - [`WeightBounds`](@ref)
  - [`weight_bounds_constraints`](@ref)
"""
const WbE_Wb = Union{<:WeightBoundsEstimator, <:WeightBounds}
"""
    weight_bounds_constraints(wb::WeightBoundsEstimator, sets::UniverseSets; strict::Bool = false,
                              datatype::DataType = Float64, kwargs...)

Generate portfolio weight bounds constraints from a `WeightBoundsEstimator` and asset set.

`weight_bounds_constraints` constructs a [`WeightBounds`](@ref) object representing lower and upper portfolio weight bounds for the assets in `sets`, using the specifications in `wb`. A bound may be a scalar, a vector, a dictionary, a pair, or an algorithmic rule, so one estimator carries both a universe-wide limit and an asset-specific one. A bound that is `nothing` states no limit in that direction, and survives the resolution as `nothing`.

# Algorithm

 1. Resolve the lower side with [`estimator_to_val`](@ref), giving `lb`. The fill value for an asset the estimator does not name is `wb.dlb`, or `zero(datatype)` when `wb.dlb` is `nothing`.
 2. Resolve the upper side the same way, giving `ub`. The fill value is `wb.dub`, or `one(datatype)` when `wb.dub` is `nothing`.
 3. Build `WeightBounds(; lb = lb, ub = ub)`. That constructor runs [`validate_bounds`](@ref) on the resolved pair, which is where a `AbstractDict` bound on one side and a scalar on the other are first compared.

# Arguments

  - `wb`: [`WeightBoundsEstimator`](@ref) specifying lower and upper bounds.
  - `sets`: [`UniverseSets`](@ref) containing asset names or indices.
  - `strict`: If `true`, enforces strict matching between assets and bounds (throws error on mismatch); if `false`, issues a warning.
  - `datatype`: Output data type for bounds.
  - `kwargs...`: Additional keyword arguments passed to bound extraction routines.

# Validation

  - The resolved pair passes [`validate_bounds`](@ref), through the [`WeightBounds`](@ref) constructor of step 3. A resolved `lb` above a resolved `ub` raises `DomainError` there, and a resolved vector of the wrong length raises `DimensionMismatch` inside [`estimator_to_val`](@ref).

# Returns

  - `wb::WeightBounds`: Object containing lower and upper bounds aligned with `sets`.

# Examples

```jldoctest
julia> sets = UniverseSets(; dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"]));

julia> wb = WeightBoundsEstimator(; lb = Dict(\"A\" => 0.1, \"B\" => 0.2), ub = 1.0);

julia> weight_bounds_constraints(wb, sets)
WeightBounds
  lb ┼ Vector{Float64}: [0.1, 0.2, 0.0]
  ub ┴ Float64: 1.0

julia> wb = WeightBoundsEstimator(; lb = Dict(\"A\" => 0.1), ub = Dict(\"A\" => 0.5), dlb = 0.02,
                                  dub = 0.7);

julia> weight_bounds_constraints(wb, sets)
WeightBounds
  lb ┼ Vector{Float64}: [0.1, 0.02, 0.02]
  ub ┴ Vector{Float64}: [0.5, 0.7, 0.7]
```

# Related

  - [`WeightBoundsEstimator`](@ref)
  - [`WeightBounds`](@ref)
  - [`estimator_to_val`](@ref)
  - [`UniverseSets`](@ref)
"""
function weight_bounds_constraints(wb::WeightBoundsEstimator, sets::UniverseSets;
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

# Algorithm

 1. Return `fill(val, N)`, the vector that gives every asset the same free bound.

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

`weight_bounds_constraints_side` gives every asset the scalar bound `wb`. A finite `wb` becomes a constant `range` and an infinite one becomes a filled `Vector`, because `range(Inf, Inf; length = N)` collects to `NaN`. The value `val` is not read by this method: an infinite bound keeps its own sign, so a lower bound of `Inf` stays `Inf` and admits no weight rather than becoming the free bound `-Inf`.

# Algorithm

 1. When `isinf(wb)`, return `fill(wb, N)`, a `Vector` that repeats the infinite bound.
 2. Otherwise return `range(wb, wb; length = N)`, a constant `StepRangeLen` of length `N`.

# Arguments

  - `wb::Number`: Scalar bound for portfolio weights (can be finite or infinite).
  - `N::Integer`: Number of assets (length of the output vector).
  - `val::Number`: Free bound of this side, `-Inf` for a lower bound and `Inf` for an upper one. This method does not read it, and it is present so that every method of the function takes the same three arguments.

# Returns

  - `wb::VecNum`: Vector or range of length `N`, every entry of which holds `wb`.

# Examples

```jldoctest
julia> PortfolioOptimisers.weight_bounds_constraints_side(0.1, 3, -Inf)
StepRangeLen(0.1, 0.0, 3)

julia> PortfolioOptimisers.weight_bounds_constraints_side(Inf, 3, -Inf)
3-element Vector{Float64}:
 Inf
 Inf
 Inf
```

# Related

  - [`weight_bounds_constraints`](@ref)
  - [`WeightBounds`](@ref)
  - [`weight_bounds_constraints_side`](@ref)
"""
function weight_bounds_constraints_side(wb::Number, N::Integer, val::Number)
    return if isinf(wb)
        fill(wb, N)
    else
        range(wb, wb; length = N)
    end
end
"""
    weight_bounds_constraints_side(wb::VecNum, N::Integer = 0, args...)

Propagate asset-specific portfolio weight bounds from a vector.

`weight_bounds_constraints_side` returns the input vector `wb` unchanged when asset-specific bounds are provided as a vector. This method is used to propagate explicit per-asset bounds in constraint generation routines.

# Algorithm

 1. When `N` is not zero, check `length(wb) == N`.
 2. Return `wb`.

# Arguments

  - `wb`: Vector of bounds for portfolio weights (one per asset).
  - `N`: Number of assets. A zero states that the caller knows no asset count, and then the length is not checked.
  - `args...`: Additional positional arguments (ignored).

# Validation

  - `iszero(N) || length(wb) == N`, `DimensionMismatch` otherwise. A reader of a bound compares it with the weights through `map`, which truncates to the shorter argument, so a bound shorter than the universe leaves every asset past `length(wb)` unchecked.

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
function weight_bounds_constraints_side(wb::VecNum, N::Integer = 0, args...)
    @argcheck(iszero(N) || length(wb) == N,
              DimensionMismatch("bound of length $(length(wb)) does not match the $N assets it must bound"))
    return wb
end
"""
    weight_bounds_constraints(wb::WeightBounds{<:Any, <:Any}, args...; N::Integer = 0, kwargs...)

Expand portfolio weight bounds constraints from a `WeightBounds` object to length `N`.

`weight_bounds_constraints` expands a scalar, `nothing` or infinite bound to a vector or range of length `N` using [`weight_bounds_constraints_side`](@ref), so every bound reaches the model per asset. A vector bound is passed through unchanged, and its length is checked against `N`.

`N` is required in practice. The default `N = 0` builds two empty bound vectors, and [`WeightBounds`](@ref)'s own validation then throws `IsEmptyError: lb cannot be empty`.

# Algorithm

 1. Expand the lower side with `weight_bounds_constraints_side(wb.lb, N, -Inf)`, giving `lb`.
 2. Expand the upper side with `weight_bounds_constraints_side(wb.ub, N, Inf)`, giving `ub`.
 3. Build `WeightBounds(; lb = lb, ub = ub)`, which validates the expanded pair.

# Arguments

  - `wb`: [`WeightBounds`](@ref) object containing lower and upper bounds.
  - `args...`: Additional positional arguments (ignored).
  - `N`: Number of assets, the length of the expansion.
  - `kwargs...`: Additional keyword arguments (ignored).

# Validation

  - A vector bound has length `N`, `DimensionMismatch` otherwise, through [`weight_bounds_constraints_side`](@ref).
  - The expanded pair passes [`validate_bounds`](@ref), through the [`WeightBounds`](@ref) constructor of step 3. With the default `N = 0` both sides expand to an empty vector and that constructor raises `IsEmptyError`.

# Returns

  - `wb::WeightBounds`: Expanded bounds object.

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
    weight_bounds_constraints(wb::WeightBounds{<:VecNum, <:VecNum}, args...; N::Integer = 0,
                              kwargs...)

Propagate asset-specific portfolio weight bounds constraints from a `WeightBounds` object with vector bounds.

`weight_bounds_constraints` returns the input [`WeightBounds`](@ref) object unchanged when both lower and upper bounds are provided as vectors. This method is used to propagate explicit per-asset bounds in constraint generation workflows, ensuring that asset-specific constraints are preserved.

# Algorithm

 1. When `N` is not zero, check `length(wb.lb) == N`. The constructor of `wb` already held the two sides to one length.
 2. Return `wb`.

# Arguments

  - `wb`: [`WeightBounds`](@ref) object with vector lower and upper bounds.
  - `args...`: Additional positional arguments (ignored).
  - `N`: Number of assets. A zero states that the caller knows no asset count, and then the length is not checked.
  - `kwargs...`: Additional keyword arguments (ignored).

# Validation

  - `iszero(N) || length(wb.lb) == N`, `DimensionMismatch` otherwise.

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
                                   N::Integer = 0, kwargs...)::WeightBounds
    @argcheck(iszero(N) || length(wb.lb) == N,
              DimensionMismatch("bound of length $(length(wb.lb)) does not match the $N assets it must bound"))
    return wb
end
"""
    weight_bounds_constraints(wb::Nothing, args...; N::Integer = 0, kwargs...)

Generate unconstrained portfolio weight bounds when no bounds are specified.

`weight_bounds_constraints` returns a [`WeightBounds`](@ref) object with lower bounds set to `-Inf` and upper bounds set to `Inf` for all assets when `wb` is `nothing`. `N` is required in practice. The default `N = 0` builds two empty vectors, and [`WeightBounds`](@ref)'s own validation then throws `IsEmptyError: lb cannot be empty`.

# Algorithm

 1. Fill both sides to length `N`, `-Inf` on the lower side and `Inf` on the upper side.
 2. Build `WeightBounds(; lb = lb, ub = ub)`, which validates the pair.

# Arguments

  - `wb::Nothing`: Indicates no constraint for portfolio weights.
  - `args...`: Additional positional arguments (ignored).
  - `N::Integer`: Number of assets, the length of the two bound vectors.
  - `kwargs...`: Additional keyword arguments (ignored).

# Validation

  - The pair passes [`validate_bounds`](@ref), through the [`WeightBounds`](@ref) constructor of step 2. With the default `N = 0` both sides are empty and that constructor raises `IsEmptyError`.

# Returns

  - `wb::WeightBounds`: Object with unconstrained lower and upper bounds.

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
