"""
$(DocStringExtensions.TYPEDEF)

Carries the vector of non-negative risk budgets that a risk-budgeting optimiser targets.

The budget names how much of the total risk each entry of an axis is asked to carry. [`AssetRiskBudgeting`](@ref) writes it over the assets and [`FactorRiskBudgeting`](@ref) over the factors. A budget built by hand keeps the sum it was written with, and still reaches the same weights as its normalised twin, because the barrier row that reads it is homogeneous in the budget.

# Mathematical definition

A risk-budgeting model minimises the risk of an unnormalised weight vector under a logarithmic barrier, and recovers the portfolio by dividing out the budget variable:

```math
\\begin{align}
\\underset{\\boldsymbol{y},\\, k}{\\min}\\quad & \\phi(\\boldsymbol{y})\\\\
\\textrm{s.t.}\\quad & \\boldsymbol{b}^\\intercal \\ln(\\boldsymbol{y}) \\geq 0\\,,\\\\
& \\boldsymbol{1}^\\intercal \\boldsymbol{y} = k\\,,\\\\
& \\boldsymbol{y},\\, k \\geq 0\\,,\\\\
& \\boldsymbol{w} = \\boldsymbol{y} / k\\,.
\\end{align}
```

The barrier is what states the budget. At the optimum the Karush-Kuhn-Tucker conditions give the risk-contribution identity:

```math
\\begin{align}
y_i \\frac{\\partial \\phi(\\boldsymbol{y})}{\\partial y_i} &= \\lambda\\, b_i\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{b}``: Risk budget vector, the `val` field.
  - ``\\boldsymbol{y}``: Unnormalised weight vector.
  - $(math_dict[:k_budget])
  - $(math_dict[:w_port])
  - ``\\phi``: Risk measure of the optimiser.
  - ``\\lambda``: Multiplier of the barrier constraint.

Only the **relative** entries of ``\\boldsymbol{b}`` reach the solution. The barrier is positively homogeneous of degree one in ``\\boldsymbol{b}``, so ``c\\, \\boldsymbol{b}^\\intercal \\ln(\\boldsymbol{y}) \\geq 0`` and ``\\boldsymbol{b}^\\intercal \\ln(\\boldsymbol{y}) \\geq 0`` cut the same set for every ``c > 0``, and the multiplier ``\\lambda`` of the risk-contribution identity absorbs the scale.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RiskBudget(
        val::Num_VecNum
    ) -> RiskBudget
    RiskBudget(;
        val::Num_VecNum
    ) -> RiskBudget

Keywords correspond to the struct's fields. The keyword form forwards to the positional one, so both run the same checks.

## Validation

  - `!isempty(val)`, `IsEmptyError` otherwise.
  - `all(x -> zero(x) <= x, val)`, `DomainError` otherwise. A zero entry is admitted; only a negative one raises.

The inner constructor is typed on `Num_VecNum` rather than on `VecNum`, and that width is load-bearing. `@concrete` emits a generic `RiskBudget(val::__T_val) where __T_val`, which every argument matches. A `VecNum`-typed inner method does not apply to a scalar at all, so a scalar would reach the generic method and pass neither check. `Num_VecNum` applies, and it is more specific than the unbounded generic, so both routes run both checks.

A scalar `val` is a real input, not a mistake to refuse: a scalar [`RiskBudgetEstimator`](@ref) resolves to one, because [`risk_budget_constraints`](@ref) normalises the budget to sum to one and a scalar divided by itself is `1.0`. Such a budget states one allocation over a one-entry axis, and an optimiser accepts it only for a one-asset universe, because the risk-budgeting builder checks the budget's length against the weight count.

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `val`: Sliced to the selected indices via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> RiskBudget(; val = [0.2, 0.3, 0.5])
RiskBudget
  val ┴ Vector{Float64}: [0.2, 0.3, 0.5]
```

# Related

  - [`RiskBudgetEstimator`](@ref)
  - [`risk_budget_constraints`](@ref): normalises the budget to sum to one on its estimator branch only. The identity branch returns this object untouched, so a hand-built budget keeps its own sum.
  - [`AbstractConstraintResult`](@ref)
  - [`AssetRiskBudgeting`](@ref)
  - [`FactorRiskBudgeting`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 10.1.3.
"""
@propagatable @concrete struct RiskBudget <: AbstractConstraintResult
    """
    $(field_dict[:rkb_val])
    """
    @vprop val
    # `Num_VecNum` and not `VecNum`. `@concrete` emits a generic
    # `RiskBudget(val::__T_val) where __T_val` that every argument matches. A `VecNum`
    # method does not apply to a scalar at all, so `RiskBudget(-1.0)` reached the generic
    # one and passed neither check, while `RiskBudget(; val = -1.0)` raised because the
    # keyword form carried a copy of them. `Num_VecNum` applies to a scalar and is more
    # specific than the unbounded generic, so both routes now run these two checks and the
    # copy is gone. A scalar is a real input -- `risk_budget_constraints` on a scalar
    # estimator ends at `RiskBudget(; val = val / sum(val))`, which is `1.0` -- so it is
    # validated, not refused. `Threshold` in `07_ThresholdConstraintGeneration.jl` has
    # always had the wider inner constructor and has never had the hole. Issue #518.
    function RiskBudget(val::Num_VecNum)::RiskBudget
        @argcheck(!isempty(val), IsEmptyError("val cannot be empty"))
        @argcheck(all(x -> zero(x) <= x, val),
                  DomainError(val, "all entries of val must be >= 0"))
        return new{typeof(val)}(val)
    end
end
function RiskBudget(; val::Num_VecNum)::RiskBudget
    return RiskBudget(val)
end
"""
$(DocStringExtensions.TYPEDEF)

Resolves a risk budget written in asset or group names against a universe.

[`risk_budget_constraints`](@ref) turns it into a [`RiskBudget`](@ref): every name in `val` is mapped to its indices in the selected universe, an unnamed asset takes `dval`, and the result is normalised to sum to one. A group name assigns its value to every asset in the group.

This estimator carries **no `key` field**. [`risk_budget_constraints`](@ref) takes the key as a positional argument instead, and resolves a `nothing` key to `sets.xkey` inside the verb. [`ThresholdEstimator`](@ref) is the same shape with the key held as a field. A `dval` of `nothing` becomes `1/length` of the axis the key selects, so an unnamed entry starts at the uniform share of that axis rather than at zero — on a universe of four assets and two factors, a factor budget naming only `F1` fills `F2` with `0.5`.

A **scalar** `val` is accepted, and resolves to a scalar rather than to a uniform vector. The normalisation divides the scalar by itself, so every scalar resolves to the same `RiskBudget(1.0)` and the number written here reaches nothing. Only a one-entry axis can consume such a budget: [`RiskBudgeting`](@ref) reads the budget against an `N`-vector of weights. Write the uniform budget as `nothing`, which [`risk_budget_constraints`](@ref) turns into `1/N` over `N` entries, or as [`UniformValues`](@ref), which resolves to the same vector through this estimator.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RiskBudgetEstimator(;
        val::EstValType,
        dval::Option{<:Number} = nothing
    ) -> RiskBudgetEstimator

Keywords correspond to the struct's fields.

## Validation

  - `val` is validated with [`assert_nonempty_nonneg_finite_val`](@ref).

# Examples

```jldoctest
julia> RiskBudgetEstimator(; val = Dict(\"A\" => 0.2, \"B\" => 0.3, \"C\" => 0.5))
RiskBudgetEstimator
   val ┼ Dict{String, Float64}: Dict("B" => 0.3, "A" => 0.2, "C" => 0.5)
  dval ┴ nothing

julia> RiskBudgetEstimator(; val = [\"A\" => 0.2, \"B\" => 0.3, \"C\" => 0.5])
RiskBudgetEstimator
   val ┼ Vector{Pair{String, Float64}}: ["A" => 0.2, "B" => 0.3, "C" => 0.5]
  dval ┴ nothing
```

Two different scalars resolve to the same budget, and the uniform budget over the universe is written as `nothing`:

```jldoctest
julia> sets = UniverseSets(; dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"]));

julia> risk_budget_constraints(RiskBudgetEstimator(; val = 0.2), sets)
RiskBudget
  val ┴ Float64: 1.0

julia> risk_budget_constraints(RiskBudgetEstimator(; val = 0.9), sets)
RiskBudget
  val ┴ Float64: 1.0

julia> risk_budget_constraints(nothing; N = 3)
RiskBudget
  val ┴ StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}: StepRangeLen(0.3333333333333333, 0.0, 3)
```

# Related

  - [`RiskBudget`](@ref)
  - [`risk_budget_constraints`](@ref)
  - [`UniverseSets`](@ref)
  - [`ThresholdEstimator`](@ref): the same shape with the universe key held as a field.

# References

  - $(ref_dict[:cajas2025]) Section 10.1.3.
"""
@concrete struct RiskBudgetEstimator <: AbstractConstraintEstimator
    """
    $(field_dict[:rkbe_val])
    """
    val
    """
    $(field_dict[:dval])
    """
    dval
    function RiskBudgetEstimator(val::EstValType,
                                 dval::Option{<:Number})::RiskBudgetEstimator
        assert_nonempty_nonneg_finite_val(val, :val)
        assert_nonempty_nonneg_finite_val(dval, :dval)
        return new{typeof(val), typeof(dval)}(val, dval)
    end
end
function RiskBudgetEstimator(; val::EstValType,
                             dval::Option{<:Number} = nothing)::RiskBudgetEstimator
    return RiskBudgetEstimator(val, dval)
end
"""
    const RkbE_Rkb = Union{<:RiskBudgetEstimator, <:RiskBudget}

Alias for a risk budget estimator or result.

Matches either a [`RiskBudgetEstimator`](@ref) (specifying how to generate risk budget constraints) or a [`RiskBudget`](@ref) result (a pre-computed risk budget allocation). Used internally to accept either form in constraint generation dispatch.

There is no vector counterpart, and [`risk_budget_constraints`](@ref) has no vector method. A risk budget is one allocation over the whole universe, so an optimiser holds exactly one. This is the same reason [`WbE_Wb`](@ref) and [`FeesE_Fees`](@ref) are singular, and the reason [`TnE_Tn`](@ref), [`LcE_Lc`](@ref) and [`PlCE_PlC`](@ref) are not: several turnover, linear or phylogeny constraints can hold at once, and each of those does carry a vector alias and a broadcast method.

The gap is a decision, not an omission. ADR 0038 names the routing targets that accumulate — `(:lcse, :cte, :ple, :slt, :sst, :sglt, :sgst, :smtx, :sgmtx)` — and a risk budget's target `:rkb` is not among them, while four of the threshold's six targets — `:slt`, `:sst`, `:sglt` and `:sgst` — are, each holding a positional list with one entry per scenario or group block. That is why [`threshold_constraints`](@ref) carries a [`VecOptBtE_Bt`](@ref) method and this family carries none. Passing a vector here raises `MethodError`; no broader method takes it silently.

# Related

  - [`RiskBudgetEstimator`](@ref)
  - [`RiskBudget`](@ref)
  - [`risk_budget_constraints`](@ref)
  - [`BtE_Bt_VecOptBtE_Bt`](@ref): the threshold family's alias, which does carry a vector arm.
"""
const RkbE_Rkb = Union{<:RiskBudgetEstimator, <:RiskBudget}
"""
    risk_budget_constraints(::Nothing, args...; N::Integer, kwargs...)

No-op fallback for risk budget constraint generation.

This method returns a uniform risk budget allocation when no explicit risk budget is specified (`nothing`). It creates a [`RiskBudget`](@ref) with equal budgets summing to one over `N` entries. This is useful as a default in workflows where a risk budget is optional or omitted.

The vector is a constant `range`, not an `Array`, and its element type is always `Float64` because it comes from `inv(N)`. This method takes no `datatype` keyword; one passed here is swallowed by `kwargs...` and changes nothing.

This is the one branch whose budget sums to **exactly** one. `sum` of a range reads the arithmetic series rather than the entries, so `sum` of the result is `1.0` for `N = 3`, `N = 7` and `N = 10` alike, while `sum(collect(...))` returns `0.9999999999999998` for `N = 7`. The estimator branch divides by a computed sum instead, and lands within round-off of one rather than on it.

# Algorithm

 1. Compute `iN`, the reciprocal of `N`.
 2. Build the constant range of length `N` whose start and stop are both `iN`.
 3. Wrap it in a [`RiskBudget`](@ref), which checks that it is non-empty and non-negative.

# Arguments

  - `::Nothing`: Indicates that no explicit risk budget is specified.
  - `args...`: Additional positional arguments (ignored).
  - `N::Integer`: Number of entries of the budgeted axis (required).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `rb::RiskBudget`: A result object containing a uniform risk budget vector of length `N`, with each entry equal to `1/N`.

# Examples

```jldoctest
julia> risk_budget_constraints(nothing; N = 3)
RiskBudget
  val ┴ StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}: StepRangeLen(0.3333333333333333, 0.0, 3)
```

# Related

  - [`RiskBudget`](@ref)
  - [`risk_budget_constraints`](@ref)
"""
function risk_budget_constraints(::Nothing, args...; N::Integer, kwargs...)::RiskBudget
    iN = inv(N)
    return RiskBudget(; val = range(iN, iN; length = N))
end
"""
    risk_budget_constraints(rb::RiskBudget, args...; kwargs...)

No-op fallback for risk budget constraint propagation.

This method returns the input [`RiskBudget`](@ref) object unchanged. It is used to pass through an already constructed risk budget allocation result, enabling composability and uniform interface handling in risk budgeting workflows.

It does **not** normalise, unlike the estimator branch, so a hand-built `RiskBudget(; val = [1, 2, 3])` reaches an optimiser summing to `6`. That is harmless: the barrier row the optimiser writes is homogeneous in the budget, and the same three-asset variance model returns the same weights for `[1, 2, 3]` as for `[1/6, 2/6, 3/6]` to within `3.5e-5`, which is solver tolerance.

# Algorithm

 1. Return `rb`. The method reads none of its other arguments and none of its keywords.

# Arguments

  - `rb`: An existing [`RiskBudget`](@ref) object.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `rb::RiskBudget`: The input `RiskBudget` object, unchanged.

# Examples

```jldoctest
julia> risk_budget_constraints(RiskBudget(; val = [0.2, 0.3, 0.5]))
RiskBudget
  val ┴ Vector{Float64}: [0.2, 0.3, 0.5]

julia> risk_budget_constraints(RiskBudget(; val = [1, 2, 3]))
RiskBudget
  val ┴ Vector{Int64}: [1, 2, 3]
```

# Related

  - [`RiskBudget`](@ref)
  - [`risk_budget_constraints`](@ref)
"""
function risk_budget_constraints(rb::RiskBudget, args...; kwargs...)::RiskBudget
    return rb
end
"""
    risk_budget_constraints(rb::EstValType, sets::UniverseSets,
                            dval::Option{<:Number} = nothing,
                            key::Option{<:AbstractString} = nothing; strict::Bool = false,
                            kwargs...)

Generate a risk budget allocation from asset/group mappings and asset sets.

This method constructs a [`RiskBudget`](@ref) from a mapping of asset or group names to risk budget values, using the provided [`UniverseSets`](@ref). The mapping can be a dictionary, a single pair, or a vector of pairs. Names are resolved against the universe `key` selects, and the resulting risk budget vector is normalised to sum to one.

A **scalar** `rb` resolves to a scalar, not to a uniform vector, so the normalisation returns `1.0` whatever the scalar was. The result is a `RiskBudget` holding one number, and an optimiser accepts it only for a one-asset universe. Write the uniform budget as `nothing`, which the no-op method turns into `1/N` over `N` entries.

The normalisation divides by a computed sum, so the result lands within round-off of one rather than on it: a seven-asset budget naming two assets sums to `1.0000000000000002`. Only the `nothing` branch sums to exactly one.

# Algorithm

 1. Take `dval` as the fill value. When it is `nothing`, use `1/length` of the universe that `key` selects, or of `sets.xkey`'s universe when `key` is `nothing` too.
 2. Resolve `rb` against `sets` with [`estimator_to_val`](@ref), giving `val`. A name that names an asset writes one entry, a name that names a group writes one entry per member of the group, and every unnamed entry keeps the fill value of step 1.
 3. Divide `val` by `sum(val)`, giving the normalised budget.
 4. Wrap it in a [`RiskBudget`](@ref), which checks that it is non-empty and non-negative.

# Arguments

  - `rb`: A dictionary, pair, or vector of pairs mapping asset or group names to risk budget values.
  - `sets`: A [`UniverseSets`](@ref) object specifying the universe and groupings.
  - `dval`: Default value to use for names not found in `rb`. If `nothing`, a default value of `1/length(sets.dict[key])` is used.
  - $(arg_dict[:ekey]) [`FactorRiskBudgeting`](@ref) passes `sets.fkey`, because its budget is written in factor names.
  - `strict`: If `true`, throws an error if a key in `rb` is not found in `sets`; if `false`, issues a warning.

# Validation

  - A name in `rb` that names neither an asset nor a group of `sets` raises `ArgumentError` when `strict` is `true`. A warning is issued otherwise, and the name writes nothing.
  - A budget whose entries are all zero divides by zero in step 3, and the resulting `NaN` vector fails [`RiskBudget`](@ref)'s own non-negativity check, so the call raises `DomainError` rather than returning a `NaN` budget. `RiskBudget` admits a zero **entry**, and only an all-zero budget reaches this raise.

# Returns

  - `rb::RiskBudget`: A result object containing the normalised risk budget vector.

# Examples

```jldoctest
julia> sets = UniverseSets(; xkey = \"nx\",
                           dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"], \"group1\" => [\"A\", \"B\"]));

julia> risk_budget_constraints(Dict(\"A\" => 0.2, \"group1\" => 0.8), sets)
RiskBudget
  val ┴ Vector{Float64}: [0.41379310344827586, 0.41379310344827586, 0.17241379310344826]
```

A budget written in factor names resolves against the declared factor axis, which is what [`FactorRiskBudgeting`](@ref) passes — the unspecified factors take the `1/length(sets.dict[key])` default before normalisation:

```jldoctest
julia> sets = UniverseSets(; dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"], \"nf\" => [\"F1\", \"F2\"]));

julia> risk_budget_constraints(Dict(\"F1\" => 0.25), sets, nothing, sets.fkey)
RiskBudget
  val ┴ Vector{Float64}: [0.3333333333333333, 0.6666666666666666]
```

# Related

  - [`RiskBudget`](@ref)
  - [`UniverseSets`](@ref)
  - [`estimator_to_val`](@ref)
  - [`risk_budget_constraints`](@ref)
  - [`FactorRiskBudgeting`](@ref)
"""
function risk_budget_constraints(rb::EstValType, sets::UniverseSets,
                                 dval::Option{<:Number} = nothing,
                                 key::Option{<:AbstractString} = nothing;
                                 strict::Bool = false, kwargs...)::RiskBudget
    if isnothing(dval)
        dval = inv(length(sets.dict[ifelse(isnothing(key), sets.xkey, key)]))
    end
    val = estimator_to_val(rb, sets, dval, key; strict = strict)
    return RiskBudget(; val = val / sum(val))
end
"""
    risk_budget_constraints(rb::RiskBudgetEstimator, sets::UniverseSets,
                            key::Option{<:AbstractString} = nothing; strict::Bool = false,
                            kwargs...)

This method unpacks a [`RiskBudgetEstimator`](@ref) and calls the mapping method with its two fields. It is used for type stability and to give a uniform interface for processing constraint estimators.

The key is a **positional** argument here, not a field of the estimator. [`ThresholdEstimator`](@ref) holds its key instead, so [`threshold_constraints`](@ref) needs no such argument. Both routes end at the same [`estimator_to_val`](@ref), and a key naming the factor axis resolves on either: on a universe of four assets and two factors, a budget naming only `F1` returns a two-entry vector.

There is **no vector method**. A vector of estimators raises `MethodError`, and no broader method takes it silently. [`RkbE_Rkb`](@ref) states the decision that governs the gap.

# Algorithm

 1. Call `risk_budget_constraints(rb.val, sets, rb.dval, key; strict = strict, kwargs...)`, which resolves the mapping, normalises it and builds the [`RiskBudget`](@ref).

# Arguments

  - `rb`: A [`RiskBudgetEstimator`](@ref) carrying the mapping and the default value.
  - `sets`: A [`UniverseSets`](@ref) object specifying the universe and groupings.
  - $(arg_dict[:ekey]) [`FactorRiskBudgeting`](@ref) passes `sets.fkey`, because its budget is written in factor names.
  - `strict`: If `true`, throws an error if a name in `rb.val` is not found in `sets`; if `false`, issues a warning.
  - `kwargs...`: Additional keyword arguments forwarded to the mapping method.

# Returns

  - `rb::RiskBudget`: A result object containing the normalised risk budget vector.

# Examples

```jldoctest
julia> sets = UniverseSets(; dict = Dict(\"nx\" => [\"A\", \"B\", \"C\", \"D\"], \"nf\" => [\"F1\", \"F2\"]));

julia> risk_budget_constraints(RiskBudgetEstimator(; val = Dict(\"F1\" => 0.4)), sets, sets.fkey)
RiskBudget
  val ┴ Vector{Float64}: [0.4444444444444445, 0.5555555555555556]
```

# Related

  - [`RiskBudgetEstimator`](@ref)
  - [`RiskBudget`](@ref)
  - [`RkbE_Rkb`](@ref)
  - [`estimator_to_val`](@ref)
  - [`risk_budget_constraints`](@ref)
"""
function risk_budget_constraints(rb::RiskBudgetEstimator, sets::UniverseSets,
                                 key::Option{<:AbstractString} = nothing;
                                 strict::Bool = false, kwargs...)::RiskBudget
    return risk_budget_constraints(rb.val, sets, rb.dval, key; strict = strict, kwargs...)
end
export RiskBudget, RiskBudgetEstimator, risk_budget_constraints
