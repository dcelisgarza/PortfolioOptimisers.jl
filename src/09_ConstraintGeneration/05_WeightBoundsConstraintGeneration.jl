"""
    validate_bounds(lb::Number, ub::Number) -> Nothing
    validate_bounds(lb::VecNum, ub::Number) -> Nothing
    validate_bounds(lb::Number, ub::VecNum) -> Nothing
    validate_bounds(lb::VecNum, ub::VecNum) -> Nothing
    validate_bounds(lb::VecNum, ::Any) -> Nothing
    validate_bounds(::Any, ub::VecNum) -> Nothing
    validate_bounds(args...) -> Nothing

Check that a lower bound does not exceed the corresponding upper bound.

The pair of argument types selects one of seven methods, and that method alone decides which preconditions run. Two of the seven take a `nothing` on one side. They check that the vector side is not empty, and they do not compare the two sides. The last method takes every other pair and checks nothing. It accepts `(nothing, nothing)`, `(::Number, nothing)` and `(nothing, ::Number)` with no comparison.

# Arguments

  - $(arg_dict[:lb])
  - $(arg_dict[:ub])

# Validation

Each bullet names the method that does the check, the condition that the method needs and the error that it raises.

  - `(Number, Number)`: `lb <= ub`, `DomainError` otherwise.
  - `(VecNum, Number)`: `!isempty(lb)`, `IsEmptyError` otherwise. Every entry of `lb` is at most `ub`, `DomainError` otherwise.
  - `(Number, VecNum)`: `!isempty(ub)`, `IsEmptyError` otherwise. Every entry of `ub` is at least `lb`, `DomainError` otherwise.
  - `(VecNum, VecNum)`: `!isempty(lb)` and `!isempty(ub)`, `IsEmptyError` otherwise. `length(lb) == length(ub)`, `DimensionMismatch` otherwise. Entry by entry, `lb` is at most `ub`, `DomainError` otherwise. The length check runs first, so the entry-by-entry comparison never reads a truncated pair.
  - `(VecNum, Any)`: `!isempty(lb)`, `IsEmptyError` otherwise. This method takes the `(vector, nothing)` pair, so it does not compare the two sides.
  - `(Any, VecNum)`: `!isempty(ub)`, `IsEmptyError` otherwise. This method takes the `(nothing, vector)` pair, so it does not compare the two sides.
  - `(args...)`: no precondition and no raise. This method takes every other pair. The two bound types reach it with `(nothing, nothing)`, `(::Number, nothing)` and `(nothing, ::Number)`.

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
    @argcheck(lb <= ub,
              DomainError((lb, ub),
                          "`lb` is $lb and `ub` is $ub. A lower bound is at most its upper bound, so `lb <= ub` must hold."))
    return nothing
end
function validate_bounds(lb::VecNum, ub::Number)::Nothing
    @argcheck(!isempty(lb), IsEmptyError("lb cannot be empty"))
    @argcheck(all(x -> x <= ub, lb),
              DomainError(lb,
                          "`ub` is $ub, and `lb` holds $(count(x -> !(x <= ub), lb)) entries above it. A lower bound is at most its upper bound, so every entry of `lb` must satisfy `lb <= ub`."))
    return nothing
end
function validate_bounds(lb::Number, ub::VecNum)::Nothing
    @argcheck(!isempty(ub), IsEmptyError("ub cannot be empty"))
    @argcheck(all(x -> lb <= x, ub),
              DomainError(ub,
                          "`lb` is $lb, and `ub` holds $(count(x -> !(lb <= x), ub)) entries below it. A lower bound is at most its upper bound, so every entry of `ub` must satisfy `lb <= ub`."))
    return nothing
end
function validate_bounds(lb::VecNum, ub::VecNum)::Nothing
    @argcheck(!isempty(lb), IsEmptyError("lb cannot be empty"))
    @argcheck(!isempty(ub), IsEmptyError("ub cannot be empty"))
    @argcheck(length(lb) == length(ub),
              DimensionMismatch("lb ($(length(lb))) and ub ($(length(ub))) must have the same length"))
    @argcheck(all(map((x, y) -> x <= y, lb, ub)),
              DomainError((lb, ub),
                          "`lb` and `ub` disagree at $(count(map((x, y) -> !(x <= y), lb, ub))) entries. A lower bound is at most its upper bound, so `lb[i] <= ub[i]` must hold at every `i`."))
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

Holds the lower bound and the upper bound of every portfolio weight.

A bound is a scalar that applies to every asset, a vector with one bound per asset, or `nothing` for no bound on that side. The mixed-integer builders also read the bounds as the big-M that links a weight to its held indicator.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{l} \\leq \\boldsymbol{w} \\leq \\boldsymbol{u}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - ``\\boldsymbol{l}``: Lower bound vector ``N \\times 1``. An entry of ``-\\infty`` leaves that weight unbounded below. An entry of ``+\\infty`` admits no value of that weight.
  - ``\\boldsymbol{u}``: Upper bound vector ``N \\times 1``. An entry of ``+\\infty`` leaves that weight unbounded above. An entry of ``-\\infty`` admits no value of that weight.
  - $(math_dict[:N])

A scalar bound is a vector with the same value in every entry. A `nothing` bound is a vector with the free bound in every entry, ``-\\infty`` on the lower side and ``+\\infty`` on the upper side.

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

Keywords correspond to the struct's fields. Only the keyword form has the defaults. It calls the positional form, so the two forms run the same checks.

## Validation

The constructor calls [`validate_bounds`](@ref) on the pair. The types of the pair decide which of the checks below run.

  - A vector bound must not be empty, `IsEmptyError` otherwise.
  - Two vector bounds must have the same length, `DimensionMismatch` otherwise.
  - `lb` must not be above `ub`, entry by entry if both are vectors, `DomainError` otherwise.
  - If one side is `nothing`, the constructor does not compare the two sides. It checks only that a vector on the other side is not empty.
  - If both sides are `nothing`, the constructor checks nothing.
  - Two infinite scalar bounds of the same sign pass, because `lb <= ub` holds. The constructor accepts `WeightBounds(; lb = Inf, ub = Inf)` and `WeightBounds(; lb = -Inf, ub = -Inf)`, and no weight satisfies either pair.

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `lb`: Sliced to the selected indices through [`port_opt_view`](@ref).
  - `ub`: Sliced to the selected indices through [`port_opt_view`](@ref).

[`weight_bounds_constraints`](@ref) can expand a field to a constant `range`. [`port_opt_view`](@ref) slices that field like any other vector, and the slice is a view of the range.

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
  - [`validate_bounds`](@ref): the seven methods that do the checks of this type.
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

[`weight_bounds_constraints`](@ref) turns it into a [`WeightBounds`](@ref). It maps every name to its indices in the universe, and it gives an asset with no name in the bound the value `dlb` or `dub`. A bound can also be a scalar, a vector, or an algorithmic rule such as [`UniformValues`](@ref). If `dlb` is `nothing`, the resolution uses `zero(datatype)`. If `dub` is `nothing`, it uses `one(datatype)`.

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

  - If `lb` or `ub` is an `AbstractDict` or an `AbstractVector`, it must not be empty, `IsEmptyError` otherwise.
  - Two vector bounds must have the same length, `DimensionMismatch` otherwise. [`validate_bounds`](@ref) does this check.
  - If both bounds are numbers or vectors of numbers, `lb` must not be above `ub`, `DomainError` otherwise. [`validate_bounds`](@ref) does this check.
  - If one side is an `AbstractDict`, a `Pair` or an algorithmic rule, the constructor does not compare the two sides, because that side has no values yet. [`weight_bounds_constraints`](@ref) builds a [`WeightBounds`](@ref) from the resolved pair, and that constructor compares them.
  - If neither `dlb` nor `dub` is `nothing`, `dlb <= dub` must hold, `DomainError` otherwise.

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `lb`: Sliced to the selected indices through [`port_opt_view`](@ref).
  - `ub`: Sliced to the selected indices through [`port_opt_view`](@ref).

[`port_opt_view`](@ref) slices only a vector bound. A scalar, a `Dict`, a `Pair` or an algorithmic rule has no index by asset. The view keeps such a bound unchanged, and the bound resolves against the universe of the view when the estimator runs.

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
  - [`validate_bounds`](@ref): the checks that this constructor runs on a pair of numbers or a pair of vectors.
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
            @argcheck(!isempty(lb), IsEmptyError("lb cannot be empty"))
        end
        if isa(ub, Dict_Vec)
            @argcheck(!isempty(ub), IsEmptyError("ub cannot be empty"))
        end
        if isa(lb, Num_VecNum) && isa(ub, Num_VecNum)
            validate_bounds(lb, ub)
        end
        if !isnothing(dlb) && !isnothing(dub)
            @argcheck(dlb <= dub,
                      DomainError((dlb, dub),
                                  "`dlb` is $dlb and `dub` is $dub. A default lower bound is at most its default upper bound, so `dlb <= dub` must hold."))
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

The alias matches a [`WeightBoundsEstimator`](@ref) or a [`WeightBounds`](@ref). The library uses it for dispatch when it resolves weight bounds.

The alias has no vector counterpart, and [`weight_bounds_constraints`](@ref) has no vector method. Weight bounds are one box over the whole universe, so an optimiser holds one of them. [`RkbE_Rkb`](@ref) states why some constraint families are singular and others are not.

# Related

  - [`WeightBoundsEstimator`](@ref)
  - [`WeightBounds`](@ref)
  - [`weight_bounds_constraints`](@ref)
"""
const WbE_Wb = Union{<:WeightBoundsEstimator, <:WeightBounds}
"""
    weight_bounds_constraints(wb::WeightBoundsEstimator, sets::UniverseSets; strict::Bool = false,
                              datatype::DataType = Float64, kwargs...)

Resolve a `WeightBoundsEstimator` against a universe.

`weight_bounds_constraints` builds the [`WeightBounds`](@ref) of the assets in `sets` from the bounds that `wb` states. A bound can be a scalar, a vector, a dictionary, a pair, or an algorithmic rule. So one estimator can state one bound for the whole universe and a different bound for each named asset. A `nothing` bound states no bound on that side, and the resolution keeps it as `nothing`.

# Algorithm

 1. Resolve the lower side with [`estimator_to_val`](@ref). The result is `lb`. An asset that the estimator does not name gets `wb.dlb`, or `zero(datatype)` if `wb.dlb` is `nothing`.
 2. Resolve the upper side the same way. The result is `ub`. An asset that the estimator does not name gets `wb.dub`, or `one(datatype)` if `wb.dub` is `nothing`.
 3. Build `WeightBounds(; lb = lb, ub = ub)`. That constructor runs [`validate_bounds`](@ref) on the resolved pair. This is the first comparison of an `AbstractDict` bound on one side with a scalar on the other.

# Arguments

  - `wb`: [`WeightBoundsEstimator`](@ref) that states the lower and upper bounds.
  - `sets`: [`UniverseSets`](@ref) that holds the asset names and the groups.
  - $(arg_dict[:strict_opt])
  - `datatype`: Number type of the resolved bounds, and of the fill values `zero(datatype)` and `one(datatype)`.
  - $(arg_dict[:ignkwargs])

# Validation

  - The resolved pair must pass [`validate_bounds`](@ref), which the [`WeightBounds`](@ref) constructor of step 3 runs. A resolved `lb` above a resolved `ub` raises `DomainError` there. [`estimator_to_val`](@ref) raises `DimensionMismatch` for a vector bound of the wrong length.

# Returns

  - `wb::WeightBounds`: The resolved bounds, in the asset order of `sets`.

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

Expand a `nothing` bound to a vector of free bounds.

`weight_bounds_constraints_side` returns a vector of length `N` with `val` in every entry. The caller gives the free bound of the side as `val`, `-Inf` for a lower bound and `Inf` for an upper bound.

# Algorithm

 1. Return `fill(val, N)`, the vector that gives every asset the same free bound.

# Arguments

  - `::Nothing`: No bound on this side.
  - `N`: Number of assets, which is the length of the returned vector.
  - `val`: Free bound of this side, `-Inf` for a lower bound and `Inf` for an upper bound.

# Returns

  - `wb::VecNum`: Vector of length `N` with `val` in every entry.

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

Expand a scalar bound to one bound per asset.

`weight_bounds_constraints_side` gives every asset the scalar bound `wb`. A finite `wb` becomes a constant `range`. An infinite `wb` becomes a `Vector`, because `range(Inf, Inf; length = N)` collects to `NaN`. This method does not read `val`. An infinite bound keeps its sign, so a lower bound of `Inf` stays `Inf` and admits no weight. It does not become the free bound `-Inf`.

# Algorithm

 1. When `isinf(wb)`, return `fill(wb, N)`, a `Vector` that repeats the infinite bound.
 2. Otherwise return `range(wb, wb; length = N)`, a constant `StepRangeLen` of length `N`.

# Arguments

  - `wb::Number`: Scalar bound of every portfolio weight, finite or infinite.
  - `N::Integer`: Number of assets, which is the length of the returned vector.
  - `val::Number`: Free bound of this side, `-Inf` for a lower bound and `Inf` for an upper bound. This method does not read it. It is in the signature so that every method of the function takes the same three arguments.

# Returns

  - `wb::VecNum`: Vector or range of length `N` with `wb` in every entry.

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

Return a vector bound unchanged, after a check of its length.

A vector bound already holds one bound per asset, so `weight_bounds_constraints_side` returns `wb` unchanged. If the caller gives an asset count `N`, the function first checks the length of `wb` against it.

# Algorithm

 1. If `N` is not zero, check `length(wb) == N`.
 2. Return `wb`.

# Arguments

  - `wb`: Vector with one bound per asset.
  - `N`: Number of assets. A zero states that the caller knows no asset count, and the method then does not check the length.
  - $(arg_dict[:ignargs])

# Validation

  - `iszero(N) || length(wb) == N`, `DimensionMismatch` otherwise. A reader of a bound compares it with the weights through `map`, which stops at the end of the shorter argument. So a bound shorter than the universe leaves every asset past `length(wb)` with no check.

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
    weight_bounds_constraints(wb::WeightBounds{<:Any, <:Any}, args...; N::Integer = 0,
                              datatype::DataType = Float64, kwargs...)

Expand each side of a `WeightBounds` to `N` bounds.

[`weight_bounds_constraints_side`](@ref) expands a scalar or a `nothing` bound to a vector or range of length `N`, so the model gets one bound per asset. A vector bound stays unchanged, and its length must equal `N`.

The caller must give `N`. With the default `N = 0`, every scalar or `nothing` side expands to an empty vector, and the [`WeightBounds`](@ref) constructor raises `IsEmptyError`.

# Algorithm

 1. Expand the lower side with `weight_bounds_constraints_side(wb.lb, N, -Inf)`. The result is `lb`. The free bound `-Inf` has the type `datatype`, so a side with no bound takes the number type of the data, not `Float64`.
 2. Expand the upper side with `weight_bounds_constraints_side(wb.ub, N, Inf)`. The result is `ub`, and its free bound `Inf` has the type `datatype`.
 3. Build `WeightBounds(; lb = lb, ub = ub)`. Its constructor checks the expanded pair.

# Arguments

  - `wb`: [`WeightBounds`](@ref) to expand.
  - $(arg_dict[:ignargs])
  - `N`: Number of assets, which is the length of each expanded side.
  - `datatype`: Number type of the free bound that a `nothing` side expands to.
  - $(arg_dict[:ignkwargs])

# Validation

  - A vector bound must have length `N`, `DimensionMismatch` otherwise. [`weight_bounds_constraints_side`](@ref) does this check.
  - The expanded pair must pass [`validate_bounds`](@ref), which the [`WeightBounds`](@ref) constructor of step 3 runs. With the default `N = 0`, every scalar or `nothing` side expands to an empty vector, and that constructor raises `IsEmptyError`.

# Returns

  - `wb::WeightBounds`: The expanded bounds.

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
                                   datatype::DataType = Float64, kwargs...)::WeightBounds
    return WeightBounds(;
                        lb = weight_bounds_constraints_side(wb.lb, N,
                                                            convert(datatype, -Inf)),
                        ub = weight_bounds_constraints_side(wb.ub, N,
                                                            convert(datatype, Inf)))
end
"""
    weight_bounds_constraints(wb::WeightBounds{<:VecNum, <:VecNum}, args...; N::Integer = 0,
                              kwargs...)

Return a `WeightBounds` with two vector sides unchanged, after a check of its length.

Both sides of `wb` already hold one bound per asset, so `weight_bounds_constraints` returns the [`WeightBounds`](@ref) unchanged. If the caller gives an asset count `N`, the method first checks the length of `wb.lb` against it.

# Algorithm

 1. If `N` is not zero, check `length(wb.lb) == N`. The constructor of `wb` already checked that the two sides have the same length.
 2. Return `wb`.

# Arguments

  - `wb`: [`WeightBounds`](@ref) with a vector on each side.
  - $(arg_dict[:ignargs])
  - `N`: Number of assets. A zero states that the caller knows no asset count, and the method then does not check the length.
  - $(arg_dict[:ignkwargs])

# Validation

  - `iszero(N) || length(wb.lb) == N`, `DimensionMismatch` otherwise.

# Returns

  - `wb::WeightBounds`: The input bounds, unchanged.

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
    weight_bounds_constraints(wb::Nothing, args...; N::Integer = 0,
                              datatype::DataType = Float64, kwargs...)

Make free weight bounds for `N` assets.

`weight_bounds_constraints` returns a [`WeightBounds`](@ref) with `-Inf` in every entry of the lower side and `Inf` in every entry of the upper side. The caller must give `N`. With the default `N = 0`, both sides are empty, and the [`WeightBounds`](@ref) constructor raises `IsEmptyError: lb cannot be empty`.

# Algorithm

 1. Fill the lower side with `-Inf` and the upper side with `Inf`, each to length `N`. Both free bounds have the type `datatype`, so they take the number type of the data, not `Float64`.
 2. Build `WeightBounds(; lb = lb, ub = ub)`. Its constructor checks the pair.

# Arguments

  - `wb::Nothing`: No weight bounds.
  - $(arg_dict[:ignargs])
  - `N::Integer`: Number of assets, which is the length of each side.
  - `datatype`: Number type of the two free bounds.
  - $(arg_dict[:ignkwargs])

# Validation

  - The pair must pass [`validate_bounds`](@ref), which the [`WeightBounds`](@ref) constructor of step 2 runs. With the default `N = 0`, both sides are empty, and that constructor raises `IsEmptyError`.

# Returns

  - `wb::WeightBounds`: Free lower and upper bounds.

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
                                   datatype::DataType = Float64, kwargs...)::WeightBounds
    return WeightBounds(; lb = fill(convert(datatype, -Inf), N),
                        ub = fill(convert(datatype, Inf), N))
end

export WeightBoundsEstimator, WeightBounds, weight_bounds_constraints
