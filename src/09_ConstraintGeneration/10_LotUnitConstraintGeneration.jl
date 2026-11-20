"""
$(DocStringExtensions.TYPEDEF)

Resolves a lot unit written in asset or group names against a universe.

The lot unit of an asset is the number of shares in one lot. A finite allocation buys each asset in whole lots of its own lot unit. [`lot_unit_constraints`](@ref) resolves this estimator to a [`LotUnit`](@ref). The function finds the indices of each name in the universe that `key` selects, and gives `dval` to each unnamed asset. A lot unit can also be a scalar, a vector, or an algorithmic rule such as [`UniformValues`](@ref).

One book can mix the two cases. A broker that sells fractional shares trades an asset in lots of, for example, `0.001` shares, and an exchange that trades in board lots sells a stock only in lots of, for example, `100` shares.

A `dval` of `nothing` resolves to `one(datatype)`, so an unnamed asset trades in whole shares. This is different from [`ThresholdEstimator`](@ref), whose `nothing` gives no threshold.

The family takes the empty fallback of [`pipe_constraint_targets`](@ref), so a lot unit is not a [`Pipeline`](@ref) step. No optimiser of a pipeline reads a lot unit. The finite allocators [`DiscreteAllocation`](@ref) and [`GreedyAllocation`](@ref) read it, and they resolve it against their own `sets`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LotUnitEstimator(;
        val::VectorEstValType,
        key::Option{<:AbstractString} = nothing,
        dval::Option{<:Number} = nothing
    ) -> LotUnitEstimator

Keywords correspond to the struct's fields.

## Validation

  - [`assert_nonempty_gt0_finite_val`](@ref) checks `val` and `dval`, so a lot unit must be non-empty, positive and finite.
  - If `key` is not `nothing`, it must be a non-empty string.

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `val`: Sliced to the selected indices through [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> LotUnitEstimator(; val = Dict(\"A\" => 0.001, \"B\" => 100))
LotUnitEstimator
   val ┼ Dict{String, Real}: Dict{String, Real}("B" => 100, "A" => 0.001)
   key ┼ nothing
  dval ┴ nothing

julia> LotUnitEstimator(; val = \"A\" => 0.001, dval = 1.0)
LotUnitEstimator
   val ┼ Pair{String, Float64}: "A" => 0.001
   key ┼ nothing
  dval ┴ Float64: 1.0
```

# Related

  - [`LotUnit`](@ref)
  - [`VectorEstValType`](@ref)
  - [`lot_unit_constraints`](@ref)
  - [`AbstractConstraintEstimator`](@ref)
  - [`UniverseSets`](@ref)
  - [`DiscreteAllocation`](@ref)
  - [`GreedyAllocation`](@ref)
  - [`ThresholdEstimator`](@ref): has the same shape, and its `nothing` default gives no threshold.
"""
@propagatable @concrete struct LotUnitEstimator <: AbstractConstraintEstimator
    """
    $(field_dict[:lu_val])
    """
    @vprop val
    """
    $(field_dict[:ekey])
    """
    key
    """
    $(field_dict[:lu_dval])
    """
    dval
    function LotUnitEstimator(val::VectorEstValType,
                              key::Option{<:AbstractString} = nothing,
                              dval::Option{<:Number} = nothing)::LotUnitEstimator
        assert_nonempty_gt0_finite_val(val, :val)
        assert_nonempty_gt0_finite_val(dval, :dval)
        if !isnothing(key)
            @argcheck(!isempty(key), IsEmptyError("key cannot be empty"))
        end
        return new{typeof(val), typeof(key), typeof(dval)}(val, key, dval)
    end
end
function LotUnitEstimator(; val::VectorEstValType, key::Option{<:AbstractString} = nothing,
                          dval::Option{<:Number} = nothing)::LotUnitEstimator
    return LotUnitEstimator(val, key, dval)
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the lot unit of each asset of a finite allocation.

The lot unit is one scalar for every asset, or a vector with one value for each asset. A finite allocation buys each asset in whole lots, so the share count of an asset is a non-negative integer multiple of its lot unit. A lot unit of one buys whole shares. A lot unit below one buys fractional shares, and a lot unit above one buys board lots.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{x} &= \\boldsymbol{l} \\odot \\boldsymbol{z}\\,.
\\end{align}
```

Where:

  - $(math_dict[:x_shares])
  - $(math_dict[:l_lot_unit]) It is the `val` field.
  - $(math_dict[:z_lots])
  - ``\\odot``: Element-wise (Hadamard) product.
  - $(math_dict[:N])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LotUnit(
        val::Num_VecNum
    ) -> LotUnit
    LotUnit(;
        val::Num_VecNum = 1
    ) -> LotUnit

Keywords correspond to the struct's fields.

## Validation

  - [`assert_nonempty_gt0_finite_val`](@ref) checks `val`, so a lot unit must be non-empty, positive and finite.

Both constructors run the same check, because the positional form is the inner constructor and the keyword form calls it.

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `val`: Sliced to the selected indices through [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> LotUnit()
LotUnit
  val ┴ Int64: 1

julia> LotUnit([0.001, 1.0, 100.0])
LotUnit
  val ┴ Vector{Float64}: [0.001, 1.0, 100.0]
```

# Related

  - [`LotUnitEstimator`](@ref)
  - [`lot_unit_constraints`](@ref)
  - [`AbstractConstraintResult`](@ref)
  - [`DiscreteAllocation`](@ref)
  - [`GreedyAllocation`](@ref)
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct LotUnit <: AbstractConstraintResult
    """
    $(field_dict[:lu_res_val])
    """
    @vprop val
    function LotUnit(val::Num_VecNum)::LotUnit
        assert_nonempty_gt0_finite_val(val, :val)
        return new{typeof(val)}(val)
    end
end
function LotUnit(; val::Num_VecNum = 1)::LotUnit
    return LotUnit(val)
end
"""
    const LuE_Lu = Union{LotUnit, LotUnitEstimator}

Alias for a lot unit result or estimator.

Matches a [`LotUnit`](@ref) or a [`LotUnitEstimator`](@ref).

The `unit` field of a finite allocator accepts both forms. A [`LotUnit`](@ref) needs no universe, and a [`LotUnitEstimator`](@ref) resolves against a universe. [`lot_unit_constraints`](@ref) resolves the estimator to a [`LotUnit`](@ref), and returns a [`LotUnit`](@ref) unchanged. One call gives a [`LotUnit`](@ref) for either form.

# Related

  - [`LotUnit`](@ref)
  - [`LotUnitEstimator`](@ref)
  - [`lot_unit_constraints`](@ref)
  - [`BtE_Bt`](@ref): the matching alias of the threshold family.
"""
const LuE_Lu = Union{LotUnit, LotUnitEstimator}
"""
    lot_unit_constraints(u::LotUnit, args...; kwargs...)

Returns a resolved lot unit unchanged.

This method lets a caller give a [`LotUnit`](@ref) wherever [`lot_unit_constraints`](@ref) also accepts a [`LotUnitEstimator`](@ref). The method makes no new object, so `lot_unit_constraints(u) === u` holds.

# Algorithm

 1. Return `u`. The method reads none of its other arguments and none of its keywords.

# Arguments

  - `u`: A [`LotUnit`](@ref).
  - `args...`: More positional arguments, which the method ignores.
  - `kwargs...`: More keyword arguments, which the method ignores.

# Returns

  - `lu::LotUnit`: The input `u`, unchanged.

# Examples

```jldoctest
julia> lot_unit_constraints(LotUnit(0.001))
LotUnit
  val ┴ Float64: 0.001
```

# Related

  - [`LotUnitEstimator`](@ref)
  - [`LotUnit`](@ref)
  - [`lot_unit_constraints`](@ref)
"""
function lot_unit_constraints(u::LotUnit, args...; kwargs...)::LotUnit
    return u
end
"""
    lot_unit_constraints(u::LotUnitEstimator, sets::UniverseSets;
                         datatype::DataType = Float64, strict::Bool = false)

Resolves a [`LotUnitEstimator`](@ref) against a universe.

The method resolves the names of `u.val` against `sets`, and returns a [`LotUnit`](@ref) with one value for each asset. `u.val` can be a dictionary, a pair, a vector of pairs, a scalar, a vector or an algorithmic rule.

# Algorithm

 1. Take the fill value `u.dval`, or `one(datatype)` when `u.dval` is `nothing`.
 2. Resolve `u.val` with [`estimator_to_val`](@ref) against the universe that `u.key` selects in `sets`. If `u.key` is `nothing`, use the universe of `sets.xkey`. An asset that `u.val` does not name takes the fill value of step 1.
 3. Wrap the resolved value in a [`LotUnit`](@ref), which checks that it is non-empty, positive and finite.

# Arguments

  - `u`: [`LotUnitEstimator`](@ref) that states the lot unit of each asset.
  - `sets`: [`UniverseSets`](@ref) that holds the asset names and the groups.
  - `datatype`: Number type of the fill value when `u.dval` is `nothing`, and of the range that [`UniformValues`](@ref) resolves to.
  - `strict`: If `true`, a name in `u.val` that `sets` does not resolve raises `ArgumentError`. If `false`, the method logs a warning and skips the name.

# Validation

  - A name in `u.val` that names neither an asset nor a group raises `ArgumentError` when `strict` is `true`. Otherwise the method logs a warning. An unnamed asset always takes the fill value of step 1.
  - The [`LotUnit`](@ref) constructor of step 3 checks the resolved value with [`assert_nonempty_gt0_finite_val`](@ref).

# Returns

  - `lu::LotUnit`: A [`LotUnit`](@ref) with one value for each asset of the selected universe, or the scalar of a scalar `u.val`.

# Examples

```jldoctest
julia> sets = UniverseSets(; dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"]));

julia> u = LotUnitEstimator(; val = Dict(\"A\" => 0.001, \"B\" => 100.0));

julia> lot_unit_constraints(u, sets)
LotUnit
  val ┴ Vector{Float64}: [0.001, 100.0, 1.0]
```

# Related

  - [`LotUnitEstimator`](@ref)
  - [`LotUnit`](@ref)
  - [`lot_unit_constraints`](@ref)
  - [`UniverseSets`](@ref)
"""
function lot_unit_constraints(u::LotUnitEstimator, sets::UniverseSets;
                              datatype::DataType = Float64, strict::Bool = false)::LotUnit
    dval = isnothing(u.dval) ? one(datatype) : u.dval
    return LotUnit(;
                   val = estimator_to_val(u.val, sets, dval, u.key; datatype = datatype,
                                          strict = strict))
end
"""
    lot_unit_constraints(u::LotUnitEstimator, ::Nothing; kwargs...)

Refuses to resolve a [`LotUnitEstimator`](@ref) without a universe.

The estimator names assets and groups, and only a [`UniverseSets`](@ref) resolves a name to an index. The constructors of [`DiscreteAllocation`](@ref) and [`GreedyAllocation`](@ref) refuse this pair before an allocation runs, so this method is reached only by a direct call.

# Algorithm

 1. Throw an [`IsNothingError`](@ref) that names the missing `sets`.

# Arguments

  - `u`: [`LotUnitEstimator`](@ref) that states the lot unit of each asset.
  - `::Nothing`: The missing universe.
  - `kwargs...`: More keyword arguments, which the method ignores.

# Validation

  - Always throws an [`IsNothingError`](@ref).

# Related

  - [`LotUnitEstimator`](@ref)
  - [`lot_unit_constraints`](@ref)
  - [`UniverseSets`](@ref)
"""
function lot_unit_constraints(::LotUnitEstimator, ::Nothing; kwargs...)
    return throw(IsNothingError("sets cannot be nothing when the lot unit is a LotUnitEstimator: the estimator names assets, and only a UniverseSets resolves a name"))
end

export LotUnit, LotUnitEstimator, lot_unit_constraints
