"""
$(DocStringExtensions.TYPEDEF)

Resolves a minimum-holding threshold written in asset or group names against a universe.

[`threshold_constraints`](@ref) resolves it to a [`Threshold`](@ref). The function finds the indices of each name in the universe that `key` selects, and gives `dval` to each unnamed asset. The default `dval` gives no threshold. A threshold can also be a scalar, a vector, or an algorithmic rule such as [`UniformValues`](@ref).

The universe key is a field of this estimator, and [`estimator_to_val`](@ref) uses `sets.xkey` when the key is `nothing`. [`RiskBudgetEstimator`](@ref) has the same shape, but its function takes the key as a positional argument.

A `dval` of `nothing` goes unchanged to [`estimator_to_val`](@ref), which uses `zero(datatype)`. An unnamed asset has no threshold. The `nothing` default of a risk budget is different, because it becomes the uniform share of its axis.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ThresholdEstimator(;
        val::VectorEstValType,
        key::Option{<:AbstractString} = nothing,
        dval::Option{<:Number} = nothing
    ) -> ThresholdEstimator

Keywords correspond to the struct's fields.

## Validation

  - [`assert_nonempty_nonneg_finite_val`](@ref) checks `val` and `dval`, so a threshold must be non-empty, non-negative and finite.
  - If `key` is not `nothing`, it must be a non-empty string.

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `val`: Sliced to the selected indices through [`port_opt_view`](@ref).

[`port_opt_view`](@ref) slices only a vector `val`. A scalar, a `Dict`, a `Pair` or an algorithmic rule has no asset index. The view keeps such a `val` unchanged, and the estimator resolves it against the universe of the view.

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
  - [`VectorEstValType`](@ref)
  - [`threshold_constraints`](@ref)
  - [`AbstractConstraintEstimator`](@ref)
  - [`UniverseSets`](@ref)
  - [`port_opt_view`](@ref)
  - [`RiskBudgetEstimator`](@ref): has the same shape, and its function takes the universe key as a positional argument.

# References

  - $(ref_dict[:chang2000]) Section 2.4, Equation 12.
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
    function ThresholdEstimator(val::VectorEstValType,
                                key::Option{<:AbstractString} = nothing,
                                dval::Option{<:Number} = nothing)::ThresholdEstimator
        assert_nonempty_nonneg_finite_val(val, :val)
        assert_nonempty_nonneg_finite_val(dval, :dval)
        if !isnothing(key)
            @argcheck(!isempty(key), IsEmptyError("key cannot be empty"))
        end
        return new{typeof(val), typeof(key), typeof(dval)}(val, key, dval)
    end
end
function ThresholdEstimator(; val::VectorEstValType,
                            key::Option{<:AbstractString} = nothing,
                            dval::Option{<:Number} = nothing)::ThresholdEstimator
    return ThresholdEstimator(val, key, dval)
end
"""
$(DocStringExtensions.TYPEDEF)

Makes each held position reach a minimum size, and sets each smaller position to zero.

The threshold is one scalar for every asset, or a vector with one value for each asset. It stops a mixed-integer model from returning many positions that are too small to trade. The original source writes the bound over the position, as this library does, so no reference portfolio enters it. [cajas2025](@cite) also writes a buy-in threshold over the positive and negative trades against a reference portfolio. This library does not make that threshold.

No upper bound applies to a threshold. A threshold above the largest weight that the budget lets an asset hold excludes that asset. On a three-asset long-only variance model with a cardinality cap of three, `Threshold(0.15)` held all three assets. `Threshold(0.5)` held two assets at exactly `0.5` each. `Threshold(1.5)` returned an [`OptimisationFailure`](@ref) with the solver status `INFEASIBLE`, because the budget row needs a sum of one and no asset can be held.

A vector threshold above one on a single asset excludes only that asset. On a 20-asset variance model, an asset held at `0.37` without a threshold went to zero under a threshold of `1.5`, and the weights still summed to one.

# Mathematical definition

The threshold is the lower half of a buy-in constraint on the held binary:

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

The binary controls both bounds. If ``z_i = 0``, the two bounds give ``w_i = 0``. If ``z_i = 1``, the position must reach ``\\ell_i``. A long threshold and a short threshold are separate objects, and each one uses the binary of its own side.

A threshold above the matching ``u_i`` permits only ``z_i = 0``, because ``\\ell_i z_i \\leq w_i \\leq u_i z_i`` is infeasible for ``z_i = 1``. The pair of bounds states the exclusion and the minimum, so no separate row is necessary.

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

  - [`assert_nonempty_nonneg_finite_val`](@ref) checks `val`, so a threshold must be non-empty, non-negative and finite. No check sets an upper bound, and the constructor accepts a value above one.

Both constructors run the same check, because the positional form is the inner constructor and the keyword form calls it.

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `val`: Sliced to the selected indices through [`port_opt_view`](@ref).

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

  - $(ref_dict[:chang2000]) Section 2.4, Equation 12.
  - $(ref_dict[:cajas2025]) Section 9.4, Equation 9.31. It adds the threshold over trades.
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
    const BtE_Bt = Union{Threshold, ThresholdEstimator}

Alias for a threshold constraint result or estimator.

Matches a [`Threshold`](@ref) or a [`ThresholdEstimator`](@ref).

A caller can write a threshold in either form, so an optimiser field must accept both. A [`Threshold`](@ref) needs no universe, and a [`ThresholdEstimator`](@ref) resolves against a universe. [`threshold_constraints`](@ref) resolves the estimator to a [`Threshold`](@ref), and returns a [`Threshold`](@ref) unchanged. One call gives a [`Threshold`](@ref) for either form. This alias does not match `nothing` or a vector. [`VecOptBtE_Bt`](@ref) and [`BtE_Bt_VecOptBtE_Bt`](@ref) add those two cases.

# Related

  - [`Threshold`](@ref)
  - [`ThresholdEstimator`](@ref)
  - [`threshold_constraints`](@ref)
  - [`RkbE_Rkb`](@ref): the matching alias of the risk budget family, which has no vector type.
"""
const BtE_Bt = Union{Threshold, ThresholdEstimator}
"""
    const VecOptBtE_Bt = AbstractVector{<:Option{<:BtE_Bt}}

Alias for a vector of optional threshold estimators or results.

Each entry is a [`Threshold`](@ref), a [`ThresholdEstimator`](@ref) or `nothing`.

The routing targets of a threshold are positional lists. `:slt`, `:sst`, `:sglt` and `:sgst` are in the set of targets that accumulate. Entry `i` of such a list belongs to scenario or group block `i`, and pairs with `scard` or `sgcarde`. A block with no threshold is `nothing` in the list. If the list dropped it, every later block moves by one. Thus the element type includes `nothing`, and [`threshold_constraints`](@ref) has a method for this alias, but [`RkbE_Rkb`](@ref) needs neither.

# Related

  - [`BtE_Bt`](@ref)
  - [`BtE_Bt_VecOptBtE_Bt`](@ref)
  - [`threshold_constraints`](@ref): the method that broadcasts over this alias and returns a [`VecOptBt`](@ref).
"""
const VecOptBtE_Bt = AbstractVector{<:Option{<:BtE_Bt}}
"""
    const BtE_Bt_VecOptBtE_Bt = Union{BtE_Bt, VecOptBtE_Bt}

Alias for a single or vector of optional threshold estimators or results.

Matches one [`BtE_Bt`](@ref) or a vector of optional [`BtE_Bt`](@ref) entries.

One optimiser field accepts both shapes. The plain targets `:lt` and `:st` hold one threshold over the whole universe. The scenario targets `:slt`, `:sst`, `:sglt` and `:sgst` hold one threshold for each block. A field of this type accepts either shape, so the optimiser needs no second field. [`threshold_constraints`](@ref) has a method for each of the two types of the union.

# Related

  - [`BtE_Bt`](@ref)
  - [`VecOptBtE_Bt`](@ref)
  - [`threshold_constraints`](@ref)
"""
const BtE_Bt_VecOptBtE_Bt = Union{BtE_Bt, VecOptBtE_Bt}
"""
    const VecOptBt = AbstractVector{<:Option{<:Threshold}}

Alias for a vector of optional threshold results.

Each entry is a [`Threshold`](@ref) or `nothing`.

This alias is the resolved form of [`VecOptBtE_Bt`](@ref). The broadcast method of [`threshold_constraints`](@ref) returns it, and the mixed-integer builders read it. A builder indexes the list by block, and needs a [`Threshold`](@ref) or `nothing` at each entry. A builder types its argument on [`Bt_VecOptBt`](@ref), which contains this alias, so it refuses a list that holds an estimator.

# Related

  - [`Threshold`](@ref)
  - [`Bt_VecOptBt`](@ref)
  - [`VecOptBtE_Bt`](@ref): the unresolved counterpart that [`threshold_constraints`](@ref) maps onto this one.
"""
const VecOptBt = AbstractVector{<:Option{<:Threshold}}
"""
    const Bt_VecOptBt = Union{Threshold, VecOptBt}

Alias for a single threshold result or a vector of optional threshold results.

Matches one [`Threshold`](@ref) or a vector of optional [`Threshold`](@ref) entries.

This alias is the resolved form of [`BtE_Bt_VecOptBtE_Bt`](@ref). A mixed-integer builder takes one threshold over the whole universe, or one threshold for each scenario block. This alias matches those two shapes after the resolution. It does not match an estimator, so a builder that types an argument on it accepts only resolved thresholds.

# Related

  - [`Threshold`](@ref)
  - [`VecOptBt`](@ref)
  - [`BtE_Bt_VecOptBtE_Bt`](@ref): the unresolved counterpart.
"""
const Bt_VecOptBt = Union{Threshold, VecOptBt}
"""
    threshold_constraints(t::Option{<:Threshold}, args...; kwargs...)

Returns a resolved threshold or `nothing` unchanged.

This method lets a caller give a [`Threshold`](@ref) wherever [`threshold_constraints`](@ref) also accepts a [`ThresholdEstimator`](@ref).

The method makes no new object and allocates no memory, so `threshold_constraints(t) === t` holds. Because it accepts `nothing`, the broadcast over a [`VecOptBtE_Bt`](@ref) keeps an empty block as `nothing`.

# Algorithm

 1. Return `t`. The method reads none of its other arguments and none of its keywords.

# Arguments

  - `t`: A [`Threshold`](@ref) or `nothing`.
  - `args...`: More positional arguments, which the method ignores.
  - `kwargs...`: More keyword arguments, which the method ignores.

# Returns

  - `bt::Option{<:Threshold}`: The input `t`, unchanged.

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

Resolves a [`ThresholdEstimator`](@ref) against a universe.

The method resolves the names of `t.val` against `sets`, and returns a [`Threshold`](@ref) with one value for each asset. `t.val` can be a dictionary, a pair, a vector of pairs, a scalar, a vector or an algorithmic rule.

This method does not normalise the resolved vector, unlike the estimator method of [`risk_budget_constraints`](@ref). A threshold bounds one weight and is not a share of a total, so a scaled vector gives a different model.

# Algorithm

 1. Resolve `t.val` with [`estimator_to_val`](@ref) against the universe that `t.key` selects in `sets`. If `t.key` is `nothing`, use the universe of `sets.xkey`. The fill value of an asset that `t.val` does not name is `t.dval`. If `t.dval` is `nothing`, the fill value is `zero(datatype)`, which is no threshold.
 2. Wrap the resolved vector in a [`Threshold`](@ref), which checks that it is non-empty, non-negative and finite.

# Arguments

  - `t`: [`ThresholdEstimator`](@ref) that states the threshold of each asset.
  - `sets`: [`UniverseSets`](@ref) that holds the asset names and the groups.
  - `datatype`: Number type of the fill value when `t.dval` is `nothing`, and of the range that [`UniformValues`](@ref) resolves to.
  - `strict`: If `true`, a name in `t.val` that `sets` does not resolve raises `ArgumentError`. If `false`, the method logs a warning and skips the name.

# Validation

  - `strict` applies to a name that does not resolve, not to an unnamed asset. A name in `t.val` that names neither an asset nor a group raises `ArgumentError` when `strict` is `true`. Otherwise the method logs a warning. An unnamed asset always takes the fill value of step 1, and never raises an error.
  - The [`Threshold`](@ref) constructor of step 2 checks the resolved vector with [`assert_nonempty_nonneg_finite_val`](@ref).

# Returns

  - `bt::Threshold`: A [`Threshold`](@ref) with one value for each asset of the selected universe.

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

The method of each entry follows from the type of that entry. An estimator resolves, a [`Threshold`](@ref) stays unchanged, and `nothing` stays `nothing`. The result is a [`VecOptBt`](@ref) with the same length and the same order. A scenario-block target needs this, because entry `i` belongs to block `i`. An empty block stays in the list as `nothing`.

[`risk_budget_constraints`](@ref) has no vector method. The threshold targets `:slt`, `:sst`, `:sglt` and `:sgst` are routing targets that accumulate, and each holds a positional list. The risk budget target `:rkb` is not. [`RkbE_Rkb`](@ref) states the same decision for the risk budget.

# Algorithm

 1. For each entry `ti` of `t`, in order, call `threshold_constraints(ti, sets; kwargs...)`.
 2. Collect the results into a vector of the same length.

# Arguments

  - `t`: A [`VecOptBtE_Bt`](@ref), with one entry for each scenario or group block. An entry is a [`ThresholdEstimator`](@ref), a [`Threshold`](@ref) or `nothing`.
  - `sets`: [`UniverseSets`](@ref) that holds the asset names and the groups.
  - `kwargs...`: More keyword arguments, which the method gives to the method of each entry.

# Returns

  - `bt::VecOptBt`: One resolved [`Threshold`](@ref) or `nothing` for each entry of `t`, in the order of `t`.

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
  - [`RkbE_Rkb`](@ref): the alias of the risk budget family, which has no vector type for the same reason.
"""
function threshold_constraints(t::VecOptBtE_Bt, sets::UniverseSets; kwargs...)
    return [threshold_constraints(ti, sets; kwargs...) for ti in t]
end

export Threshold, ThresholdEstimator, threshold_constraints
