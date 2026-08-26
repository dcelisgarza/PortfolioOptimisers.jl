"""
$(DocStringExtensions.TYPEDEF)

Resolves a minimum-holding threshold written in asset or group names against a universe.

[`threshold_constraints`](@ref) turns it into a [`Threshold`](@ref): every name is mapped to its indices in the universe `key` selects, and an unnamed asset takes `dval`, which defaults to no threshold. A threshold may also be a scalar, a vector, or an algorithmic rule such as [`UniformValues`](@ref).

The universe key is a **field** here, and [`estimator_to_val`](@ref) resolves a `nothing` key to `sets.xkey`. [`RiskBudgetEstimator`](@ref) is the same shape with the key taken positionally by its verb instead. A `dval` of `nothing` is passed straight through and becomes `zero(datatype)`, so an unnamed asset carries no threshold — where a risk budget's `nothing` default becomes the uniform share of its axis.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ThresholdEstimator(;
        val::EstValType,
        key::Option{<:AbstractString} = nothing,
        dval::Option{<:Number} = nothing
    ) -> ThresholdEstimator

Keywords correspond to the struct's fields.

## Validation

  - `val` and `dval` are both validated with [`assert_nonempty_nonneg_finite_val`](@ref), so a threshold is non-empty, non-negative and finite.
  - If `key` is not `nothing`, it is a non-empty string.

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `val`: Sliced to the selected indices via [`port_opt_view`](@ref).

Only a vector `val` is sliced. A `val` that is a scalar, a `Dict`, a `Pair` or an algorithmic rule is not indexed by asset, so a view passes it through untouched and it resolves against the viewed universe when the estimator runs.

# Examples

```jldoctest
julia> ThresholdEstimator(; val = Dict(\"A\" => 0.05, \"B\" => 0.1))
ThresholdEstimator
   val ┼ Dict{String, Float64}: Dict("B" => 0.1, "A" => 0.05)
   key ┼ nothing
  dval ┴ nothing

julia> ThresholdEstimator(; val = \"A\" => 0.05)
ThresholdEstimator
   val ┼ Pair{String, Float64}: "A" => 0.05
   key ┼ nothing
  dval ┴ nothing

julia> ThresholdEstimator(; val = 0.05)
ThresholdEstimator
   val ┼ Float64: 0.05
   key ┼ nothing
  dval ┴ nothing

julia> ThresholdEstimator(; val = [0.05])
ThresholdEstimator
   val ┼ Vector{Float64}: [0.05]
   key ┼ nothing
  dval ┴ nothing

julia> ThresholdEstimator(; val = UniformValues())
ThresholdEstimator
   val ┼ UniformValues()
   key ┼ nothing
  dval ┴ nothing
```

# Related

  - [`Threshold`](@ref)
  - [`EstValType`](@ref)
  - [`threshold_constraints`](@ref)
  - [`AbstractConstraintEstimator`](@ref)
  - [`UniverseSets`](@ref)
  - [`port_opt_view`](@ref)
  - [`RiskBudgetEstimator`](@ref): the same shape with the universe key taken positionally by its verb.

# References

  - $(ref_dict[:cajas2025]) Section 9.4.
"""
@propagatable @concrete struct ThresholdEstimator <: AbstractConstraintEstimator
    """
    $(field_dict[:thr_val])
    """
    @vprop val
    """
    $(field_dict[:ekey])
    """
    key
    """
    $(field_dict[:dval])
    """
    dval
    function ThresholdEstimator(val::EstValType, key::Option{<:AbstractString} = nothing,
                                dval::Option{<:Number} = nothing)::ThresholdEstimator
        assert_nonempty_nonneg_finite_val(val, :val)
        assert_nonempty_nonneg_finite_val(dval, :dval)
        if !isnothing(key)
            @argcheck(!isempty(key), IsEmptyError("key cannot be empty"))
        end
        return new{typeof(val), typeof(key), typeof(dval)}(val, key, dval)
    end
end
function ThresholdEstimator(; val::EstValType, key::Option{<:AbstractString} = nothing,
                            dval::Option{<:Number} = nothing)::ThresholdEstimator
    return ThresholdEstimator(val, key, dval)
end
"""
$(DocStringExtensions.TYPEDEF)

Forces every held position to reach a minimum size, and drives anything smaller to zero.

The threshold is a scalar shared by every asset or a vector of one value per asset. It exists to keep a mixed-integer model from answering with a long tail of positions too small to trade. The source writes the same constraint over positive and negative trades against a reference portfolio; this library writes it over the position, so no reference portfolio enters it.

**There is no upper bound on a threshold**, and a threshold above the largest weight the budget admits makes the asset unholdable. On a three-asset long-only variance model with a cardinality cap of three, `Threshold(0.15)` held all three and `Threshold(0.5)` held two at exactly `0.5` each, while `Threshold(1.5)` returned an [`OptimisationFailure`](@ref) whose solver status is `INFEASIBLE` — the budget row forces the weights to sum to one, and no held weight can reach `1.5`. Use a threshold above one to state that an asset must not be held only when the model has some other way to satisfy its budget.

# Mathematical definition

The threshold is the lower half of a buy-in constraint, stated against the held binary:

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\mathrm{opt}}\\quad & \\phi(\\boldsymbol{w})\\\\
\\textrm{s.t.}\\quad & \\ell_i z_i \\leq w_i \\leq u_i z_i\\,,\\quad \\forall i = 1,\\ldots,N\\,,\\\\
& \\boldsymbol{z} \\in \\{0, 1\\}^{N}\\,,\\quad \\boldsymbol{w} \\in \\mathcal{W}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - ``\\boldsymbol{z}``: Held binary, one entry per asset.
  - ``\\ell_i``: Minimum-holding threshold for asset ``i``, the `val` field.
  - ``u_i``: Upper weight bound for asset ``i``, from [`WeightBounds`](@ref).
  - $(math_dict[:N])
  - ``\\phi``: Objective function of the optimiser.
  - ``\\mathcal{W}``: Rest of the feasible set.

The binary carries both halves. Where ``z_i = 0`` the two bounds collapse to ``w_i = 0``; where ``z_i = 1`` the position must reach ``\\ell_i``. A long and a short threshold are separate objects, each bound to its own side's binary.

A threshold above the corresponding ``u_i`` admits only ``z_i = 0``, because ``\\ell_i z_i \\leq w_i \\leq u_i z_i`` is infeasible for ``z_i = 1``. So the pair of bounds carries the exclusion as well as the minimum, and no separate row states it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Threshold(
        val::Num_VecNum
    ) -> Threshold
    Threshold(;
        val::Num_VecNum
    ) -> Threshold

Keywords correspond to the struct's fields.

## Validation

  - `val` is validated with [`assert_nonempty_nonneg_finite_val`](@ref), so a threshold is non-empty, non-negative and finite. **No upper bound is checked**, and a value above one is admitted.

Both constructors run the same check, because the positional form is the inner one and the keyword form forwards to it.

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `val`: Sliced to the selected indices via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> Threshold(0.05)
Threshold
  val ┴ Float64: 0.05

julia> Threshold([0.05, 0.1, 0.0])
Threshold
  val ┴ Vector{Float64}: [0.05, 0.1, 0.0]
```

# Related

  - [`short_mip_threshold_constraints`](@ref)
  - [`mip_constraints`](@ref)
  - [`set_mip_constraints!`](@ref)
  - [`ThresholdEstimator`](@ref)
  - [`threshold_constraints`](@ref)
  - [`AbstractConstraintResult`](@ref)
  - [`WeightBounds`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 9.4.
"""
@propagatable @concrete struct Threshold <: AbstractConstraintResult
    """
    $(field_dict[:thr_res_val])
    """
    @vprop val
    function Threshold(val::Num_VecNum)::Threshold
        assert_nonempty_nonneg_finite_val(val, :val)
        return new{typeof(val)}(val)
    end
end
function Threshold(; val::Num_VecNum)::Threshold
    return Threshold(val)
end
"""
    const BtE_Bt = Union{<:Threshold, <:ThresholdEstimator}

Alias for a threshold constraint result or estimator.

Matches either a [`Threshold`](@ref) result or a [`ThresholdEstimator`](@ref). Used internally for dispatch in threshold constraint generation.

The group exists because a caller writes a threshold in either form and an optimiser field must take both: a resolved [`Threshold`](@ref) needs no universe, and a [`ThresholdEstimator`](@ref) resolves against one. [`threshold_constraints`](@ref) is the seam that maps the second onto the first, so a field typed on this alias accepts either and reaches a [`Threshold`](@ref) in one call. It admits neither `nothing` nor a vector; [`VecOptBtE_Bt`](@ref) and [`BtE_Bt_VecOptBtE_Bt`](@ref) widen it in those two directions.

# Related

  - [`Threshold`](@ref)
  - [`ThresholdEstimator`](@ref)
  - [`threshold_constraints`](@ref)
  - [`RkbE_Rkb`](@ref): the risk budget family's counterpart, which has no vector arm.
"""
const BtE_Bt = Union{<:Threshold, <:ThresholdEstimator}
"""
    const VecOptBtE_Bt = AbstractVector{<:Option{<:BtE_Bt}}

Alias for a vector of optional threshold estimators or results.

Represents a collection of optional [`BtE_Bt`](@ref) elements (threshold estimators or results, or `nothing`).

The group exists because the threshold's routing targets are **positional lists**. ADR 0038 puts `:slt`, `:sst`, `:sglt` and `:sgst` in the set of targets that accumulate, and entry `i` of such a list belongs to scenario or group block `i`, paired with `scard` or `sgcarde`. A block with no threshold is written as `nothing` rather than dropped, because dropping it would move every later block by one. That is why the element type is optional and why [`threshold_constraints`](@ref) carries a method for this alias, while [`RkbE_Rkb`](@ref) needs neither.

# Related

  - [`BtE_Bt`](@ref)
  - [`BtE_Bt_VecOptBtE_Bt`](@ref)
  - [`threshold_constraints`](@ref): the method that broadcasts over this alias and returns a [`VecOptBt`](@ref).
"""
const VecOptBtE_Bt = AbstractVector{<:Option{<:BtE_Bt}}
"""
    const BtE_Bt_VecOptBtE_Bt = Union{<:BtE_Bt, <:VecOptBtE_Bt}

Alias for a single or vector of optional threshold estimators or results.

Matches either a single [`BtE_Bt`](@ref) or a vector of optional ones.

The group exists because one optimiser field takes both shapes. The plain targets `:lt` and `:st` hold one threshold over the whole universe, and the scenario targets `:slt`, `:sst`, `:sglt` and `:sgst` hold one per block, so a field typed on this alias accepts either mandate without a second field. It is the type an optimiser keyword is written against, and [`threshold_constraints`](@ref) has a method for each of its two arms.

# Related

  - [`BtE_Bt`](@ref)
  - [`VecOptBtE_Bt`](@ref)
  - [`threshold_constraints`](@ref)
"""
const BtE_Bt_VecOptBtE_Bt = Union{<:BtE_Bt, <:VecOptBtE_Bt}
"""
    const VecOptBt = AbstractVector{<:Option{<:Threshold}}

Alias for a vector of optional threshold results.

Represents a collection of optional [`Threshold`](@ref) elements.

The group exists to name the **resolved** half of [`VecOptBtE_Bt`](@ref). It is what [`threshold_constraints`](@ref)'s broadcast method returns, and what the mixed-integer builders read: they index the list by block and expect a resolved value or `nothing` at each entry, never an estimator. Separating the two lets a builder's signature refuse an unresolved list rather than resolve one late.

# Related

  - [`Threshold`](@ref)
  - [`Bt_VecOptBt`](@ref)
  - [`VecOptBtE_Bt`](@ref): the unresolved counterpart that [`threshold_constraints`](@ref) maps onto this one.
"""
const VecOptBt = AbstractVector{<:Option{<:Threshold}}
"""
    const Bt_VecOptBt = Union{<:Threshold, <:VecOptBt}

Alias for a single threshold result or a vector of optional threshold results.

Matches either a single [`Threshold`](@ref) or a vector of optional [`Threshold`](@ref) objects.

The group exists as the resolved counterpart of [`BtE_Bt_VecOptBtE_Bt`](@ref). A mixed-integer builder takes one threshold over the whole universe or one per scenario block, and this alias is the pair of shapes it accepts after resolution. It admits no estimator, which is what makes a builder's signature state that resolution has already happened.

# Related

  - [`Threshold`](@ref)
  - [`VecOptBt`](@ref)
  - [`BtE_Bt_VecOptBtE_Bt`](@ref): the unresolved counterpart.
"""
const Bt_VecOptBt = Union{<:Threshold, <:VecOptBt}
"""
    threshold_constraints(t::Option{<:Threshold}, args...; kwargs...)

Propagate or pass through buy-in threshold portfolio constraints.

`threshold_constraints` returns the input [`Threshold`](@ref) object or `nothing` unchanged. This method is used to propagate already constructed buy-in threshold constraints, enabling composability and uniform interface handling in constraint generation workflows.

It builds nothing and allocates nothing: the returned object is the same object, so `threshold_constraints(t) === t` holds. Accepting `nothing` is what lets [`VecOptBtE_Bt`](@ref)'s broadcast carry an empty block through untouched.

# Algorithm

 1. Return `t`. The method reads none of its other arguments and none of its keywords.

# Arguments

  - `t`: An existing [`Threshold`](@ref) object or `nothing`.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `bt::Option{<:Threshold}`: The input constraint object, unchanged.

# Examples

```jldoctest
julia> threshold_constraints(Threshold(0.05))
Threshold
  val ┴ Float64: 0.05

julia> threshold_constraints(nothing)

```

# Related

  - [`ThresholdEstimator`](@ref)
  - [`Threshold`](@ref)
  - [`threshold_constraints`](@ref)
"""
function threshold_constraints(t::Option{<:Threshold}, args...;
                               kwargs...)::Option{<:Threshold}
    return t
end
"""
    threshold_constraints(t::ThresholdEstimator, sets::UniverseSets;
                          datatype::DataType = Float64, strict::Bool = false)

Generate buy-in threshold portfolio constraints from a `ThresholdEstimator` and asset set.

`threshold_constraints` constructs a [`Threshold`](@ref) object representing minimum allocation thresholds for the assets in `sets`, using the specifications in `t`. Supports scalar, vector, dictionary, pair, or custom threshold types for flexible assignment and validation.

It does **not** normalise the resolved vector, unlike [`risk_budget_constraints`](@ref)'s estimator branch. A threshold is a bound on one weight and not a share of a total, so scaling the vector changes the model.

# Algorithm

 1. Resolve `t.val` against `sets` with [`estimator_to_val`](@ref), against the universe `t.key` selects, or `sets.xkey`'s universe when `t.key` is `nothing`. The fill value for an asset that `t.val` does not name is `t.dval`, or `zero(datatype)` when `t.dval` is `nothing`, which is no threshold.
 2. Wrap the resolved vector in a [`Threshold`](@ref), which checks that it is non-empty, non-negative and finite.

# Arguments

  - `t`: [`ThresholdEstimator`](@ref) specifying asset-specific threshold values.
  - `sets`: [`UniverseSets`](@ref) containing asset names or indices.
  - `datatype`: Output data type for thresholds.
  - `strict`: If `true`, a name in `t.val` that `sets` does not resolve throws; if `false`, it issues a warning and is skipped.

# Validation

  - `strict` governs an unresolvable **name**, not an unnamed asset. A name in `t.val` that names neither an asset nor a group raises `ArgumentError` when `strict` is `true`, and issues a warning otherwise. An unnamed asset always takes the default of step 1 and never throws.
  - The resolved vector passes [`assert_nonempty_nonneg_finite_val`](@ref) through the [`Threshold`](@ref) constructor of step 2.

# Returns

  - `bt::Threshold`: Object containing threshold values aligned with `sets`.

# Examples

```jldoctest
julia> sets = UniverseSets(; dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"]));

julia> t = ThresholdEstimator(Dict(\"A\" => 0.05, \"B\" => 0.1));

julia> threshold_constraints(t, sets)
Threshold
  val ┴ Vector{Float64}: [0.05, 0.1, 0.0]
```

# Related

  - [`ThresholdEstimator`](@ref)
  - [`Threshold`](@ref)
  - [`threshold_constraints`](@ref)
  - [`UniverseSets`](@ref)
"""
function threshold_constraints(t::ThresholdEstimator, sets::UniverseSets;
                               datatype::DataType = Float64,
                               strict::Bool = false)::Threshold
    return Threshold(;
                     val = estimator_to_val(t.val, sets, t.dval, t.key; datatype = datatype,
                                            strict = strict))
end
"""
    threshold_constraints(t::VecOptBtE_Bt, sets::UniverseSets;
                          kwargs...)

Broadcasts [`threshold_constraints`](@ref) over the vector.

Provides a uniform interface for processing multiple constraint estimators simultaneously. Each entry is resolved by the method its own type selects, so an estimator resolves, a [`Threshold`](@ref) passes through and a `nothing` stays `nothing`. The result is a [`VecOptBt`](@ref) of the same length and the same order, which is what a scenario-block target needs: entry `i` belongs to block `i`, so an empty block must survive as `nothing` rather than be dropped.

**This is the difference from [`risk_budget_constraints`](@ref), which has no vector method.** ADR 0038 decides it: the threshold targets `:slt`, `:sst`, `:sglt` and `:sgst` are in the set of routing targets that accumulate and hold a positional list, and a risk budget's target `:rkb` is not. [`RkbE_Rkb`](@ref) states the same decision from the other side.

# Algorithm

 1. For each entry `ti` of `t`, in order, call `threshold_constraints(ti, sets; kwargs...)`.
 2. Collect the results into a vector of the same length.

# Arguments

  - `t`: A [`VecOptBtE_Bt`](@ref), one entry per scenario or group block. An entry is a [`ThresholdEstimator`](@ref), a [`Threshold`](@ref) or `nothing`.
  - `sets`: [`UniverseSets`](@ref) containing asset names or indices.
  - `kwargs...`: Additional keyword arguments forwarded to each entry's method.

# Returns

  - `bt::VecOptBt`: One resolved [`Threshold`](@ref) or `nothing` per entry of `t`, in the order of `t`.

# Examples

```jldoctest
julia> sets = UniverseSets(; dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"]));

julia> t = Union{Nothing, PortfolioOptimisers.BtE_Bt}[nothing, Threshold(0.05),
                                                      ThresholdEstimator(; val = Dict(\"A\" => 0.1))];

julia> bt = threshold_constraints(t, sets);

julia> length(bt), isnothing(bt[1]), bt[2].val, bt[3].val
(3, true, 0.05, [0.1, 0.0, 0.0])
```

# Related

  - [`VecOptBtE_Bt`](@ref)
  - [`VecOptBt`](@ref)
  - [`Threshold`](@ref)
  - [`ThresholdEstimator`](@ref)
  - [`threshold_constraints`](@ref)
  - [`RkbE_Rkb`](@ref): the risk budget family, which is singular by the same decision.
"""
function threshold_constraints(t::VecOptBtE_Bt, sets::UniverseSets; kwargs...)
    return [threshold_constraints(ti, sets; kwargs...) for ti in t]
end

export Threshold, ThresholdEstimator, threshold_constraints
