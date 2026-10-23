"""
$(DocStringExtensions.TYPEDEF)

Holds the non-negative risk budgets that a risk-budgeting optimiser targets.

The budget sets the target share of the total risk for each entry of an axis. [`AssetRiskBudgeting`](@ref) reads it over the assets, and [`FactorRiskBudgeting`](@ref) reads it over the factors. A budget built by hand keeps its own sum. It gives the same weights as the normalised budget, because the barrier row is homogeneous in the budget.

# Mathematical definition

A risk-budgeting model minimises the risk of an unnormalised weight vector under a logarithmic barrier. It divides that vector by the budget variable to get the portfolio weights.

```math
\\begin{align}
\\underset{\\boldsymbol{y},\\, k}{\\min}\\quad & \\phi(\\boldsymbol{y})\\\\
\\textrm{s.t.}\\quad & \\boldsymbol{b}^\\intercal \\ln(\\boldsymbol{y}) \\geq 0\\,,\\\\
& \\boldsymbol{1}^\\intercal \\boldsymbol{y} = k\\,,\\\\
& \\boldsymbol{y},\\, k \\geq 0\\,,\\\\
& \\boldsymbol{w} = \\boldsymbol{y} / k\\,.
\\end{align}
```

The barrier row states the budget. At the optimum, the Karush-Kuhn-Tucker conditions give the risk-contribution identity:

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

The original source writes the barrier as ``\\boldsymbol{b}^\\intercal \\ln(\\boldsymbol{y}) \\geq c`` with an arbitrary constant ``c``. It scales ``\\boldsymbol{y}`` to sum to one after the solve. This library writes ``c = 0``.

Only the relative entries of ``\\boldsymbol{b}`` change the solution. With ``c = 0``, the barrier row is positively homogeneous of degree one in ``\\boldsymbol{b}``. Thus ``s\\, \\boldsymbol{b}^\\intercal \\ln(\\boldsymbol{y}) \\geq 0`` and ``\\boldsymbol{b}^\\intercal \\ln(\\boldsymbol{y}) \\geq 0`` define the same set for every ``s > 0``. The multiplier ``\\lambda`` of the risk-contribution identity takes the scale.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RiskBudget(
        val::Num_VecNum
    ) -> RiskBudget
    RiskBudget(;
        val::Num_VecNum
    ) -> RiskBudget

Keywords correspond to the struct's fields. The keyword form calls the positional form, so both forms run the same checks.

## Validation

  - `!isempty(val)`, `IsEmptyError` otherwise.
  - `all(x -> zero(x) <= x, val)`, `DomainError` otherwise. The check accepts a zero entry and refuses a negative entry.

The inner constructor takes `Num_VecNum` and not `VecNum`, because `@concrete` makes a generic method `RiskBudget(val::__T_val) where __T_val`. Every argument matches that generic method. An inner method on `VecNum` does not apply to a scalar, so a scalar goes to the generic method and skips both checks. `Num_VecNum` applies to a scalar, and it is more specific than the generic method, so both forms run both checks.

A scalar `val` is a valid input. A scalar [`RiskBudgetEstimator`](@ref) resolves to the scalar `1.0`, because [`risk_budget_constraints`](@ref) divides the scalar by itself. Such a budget states one allocation over a one-entry axis. An optimiser accepts it only for a one-asset universe, because the risk-budgeting builder compares the length of the budget with the number of weights.

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `val`: Sliced to the selected indices through [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> RiskBudget(; val = [0.2, 0.3, 0.5])
RiskBudget
  val ┴ Vector{Float64}: [0.2, 0.3, 0.5]
```

# Related

  - [`RiskBudgetEstimator`](@ref)
  - [`risk_budget_constraints`](@ref): only its estimator method normalises the budget to sum to one. Its method for a `RiskBudget` returns the object unchanged, so a budget built by hand keeps its own sum.
  - [`AbstractConstraintResult`](@ref)
  - [`AssetRiskBudgeting`](@ref)
  - [`FactorRiskBudgeting`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:bruderroncalli2012]) Section 2.3.3, Equation 9, Section 2.4, and Appendix A.2.
  - $(ref_dict[:maillard2008]) Section 3.3, Equation 7, and Appendix A.2. It states the barrier first, for equal budgets.
  - $(ref_dict[:cajas2025]) Section 10.1.3, Equation 10.8. It adds the budget variable ``k``.
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

[`risk_budget_constraints`](@ref) resolves it to a [`RiskBudget`](@ref). The function finds the indices of each name of `val` in the selected universe, and gives `dval` to each unnamed entry. Then it normalises the budget to sum to one. A group name gives its value to every asset in the group.

This estimator has no `key` field. [`risk_budget_constraints`](@ref) takes the key as a positional argument, and uses `sets.xkey` when the key is `nothing`. [`ThresholdEstimator`](@ref) has the same shape, but it holds the key in a field.

A `dval` of `nothing` becomes the reciprocal of the length of the axis that the key selects. An unnamed entry starts at the uniform share of that axis, not at zero. On a universe of four assets and two factors, a factor budget that names only `F1` gives `F2` the value `0.5`.

A scalar `val` resolves to a scalar, not to a uniform vector. The normalisation divides the scalar by itself, so every scalar resolves to `RiskBudget(1.0)`, and the value of the scalar has no effect. Only a one-entry axis can use such a budget, because [`RiskBudgeting`](@ref) compares the budget with a vector of `N` weights. Write the uniform budget as `nothing`, which [`risk_budget_constraints`](@ref) turns into `1/N` over `N` entries. [`UniformValues`](@ref) gives the same range through this estimator.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RiskBudgetEstimator(;
        val::EstValType{<:VectorAbstractEstimatorValueAlgorithm},
        dval::Option{<:Number} = nothing
    ) -> RiskBudgetEstimator

Keywords correspond to the struct's fields.

## Validation

  - [`assert_nonempty_nonneg_finite_val`](@ref) checks `val`, and it checks `dval` when `dval` is not `nothing`.

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

Two different scalars resolve to the same budget. The uniform budget over the universe is `nothing`:

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
  - [`ThresholdEstimator`](@ref): has the same shape, and holds the universe key in a field.

# References

  - $(ref_dict[:bruderroncalli2012]) Section 2.3.3, Equation 9, and Section 2.4.
  - $(ref_dict[:cajas2025]) Section 10.1.3, Equation 10.8. It adds the budget variable ``k``.
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
    function RiskBudgetEstimator(val::EstValType{<:VectorAbstractEstimatorValueAlgorithm},
                                 dval::Option{<:Number})::RiskBudgetEstimator
        assert_nonempty_nonneg_finite_val(val, :val)
        assert_nonempty_nonneg_finite_val(dval, :dval)
        return new{typeof(val), typeof(dval)}(val, dval)
    end
end
function RiskBudgetEstimator(; val::EstValType{<:VectorAbstractEstimatorValueAlgorithm},
                             dval::Option{<:Number} = nothing)::RiskBudgetEstimator
    return RiskBudgetEstimator(val, dval)
end
"""
    const RkbE_Rkb = Union{<:RiskBudgetEstimator, <:RiskBudget}

Alias for a risk budget estimator or result.

Matches a [`RiskBudgetEstimator`](@ref), which states how to make a risk budget, or a [`RiskBudget`](@ref), which holds a resolved budget. The `rkb` field of [`AssetRiskBudgeting`](@ref) and [`FactorRiskBudgeting`](@ref) takes this type, so a caller can give either form.

This alias has no vector counterpart, and [`risk_budget_constraints`](@ref) has no vector method. A risk budget is one allocation over the whole universe, so an optimiser holds one budget. [`WbE_Wb`](@ref) and [`FeesE_Fees`](@ref) have no vector counterpart for the same reason. [`TnE_Tn`](@ref), [`LcE_Lc`](@ref) and [`PlCE_PlC`](@ref) have one, because an optimiser can hold several turnover, linear or phylogeny constraints. Each of those three families has a vector alias and a broadcast method.

The gap is deliberate. The routing targets that accumulate are `(:lcse, :cte, :ple, :slt, :sst, :sglt, :sgst, :smtx, :sgmtx)`. The target of a risk budget, `:rkb`, is not in this set. Four of the six targets of a threshold are in it: `:slt`, `:sst`, `:sglt` and `:sgst`. Each of the four holds a positional list, with one entry for each scenario or group block. Thus [`threshold_constraints`](@ref) has a [`VecOptBtE_Bt`](@ref) method, and this family has none.

A vector given to [`risk_budget_constraints`](@ref) raises `MethodError`, because no broader method accepts it.

# Related

  - [`RiskBudgetEstimator`](@ref)
  - [`RiskBudget`](@ref)
  - [`risk_budget_constraints`](@ref)
  - [`BtE_Bt_VecOptBtE_Bt`](@ref): the alias of the threshold family, which includes a vector type.
"""
const RkbE_Rkb = Union{<:RiskBudgetEstimator, <:RiskBudget}
"""
    risk_budget_constraints(::Nothing, args...; N::Integer, kwargs...)

Makes the uniform risk budget when the caller gives no risk budget.

This method returns a [`RiskBudget`](@ref) with `N` equal entries of `1/N`.

The vector is a constant `range`, not an `Array`. `inv(N)` sets its element type, which is `Float64` for an `Int` `N` and `BigFloat` for a `BigInt` `N`. This method takes no `datatype` keyword, and it ignores a `datatype` in `kwargs...`.

`sum` of a range uses the formula of the arithmetic series, and does not add the entries. For each `N` from 1 to 20000, the sum of the budget is within `1.1e-16` of one, and the sum of the collected entries is within `2.0e-15` of one. The sum of the budget is exactly `1.0` for `N = 3`, `7` and `10`, but it is `0.9999999999999999` for `N = 49`. The estimator method divides by a computed sum, and its sum is also within round-off of one.

# Algorithm

 1. Compute `iN`, the reciprocal of `N`.
 2. Make the constant range of length `N` whose start and stop are both `iN`.
 3. Wrap it in a [`RiskBudget`](@ref), which checks that it is non-empty and non-negative.

# Arguments

  - `::Nothing`: Shows that the caller gives no risk budget.
  - `args...`: More positional arguments, which the method ignores.
  - `N::Integer`: Number of entries of the budgeted axis. This keyword has no default.
  - `kwargs...`: More keyword arguments, which the method ignores.

# Returns

  - `rb::RiskBudget`: A [`RiskBudget`](@ref) whose `val` is a range of `N` entries, each equal to `1/N`.

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

Returns a resolved risk budget unchanged.

This method lets a caller give a [`RiskBudget`](@ref) wherever [`risk_budget_constraints`](@ref) also accepts a [`RiskBudgetEstimator`](@ref).

This method does not normalise the budget, unlike the estimator method. A budget built by hand, such as `RiskBudget(; val = [1, 2, 3])`, reaches the optimiser with a sum of `6`. The solution does not change, because the barrier row is homogeneous in the budget. On a three-asset variance model, the weights for `[1, 2, 3]` and for `[1/6, 2/6, 3/6]` differ by at most `3.5e-5`, which is the solver tolerance.

# Algorithm

 1. Return `rb`. The method reads none of its other arguments and none of its keywords.

# Arguments

  - `rb`: A [`RiskBudget`](@ref).
  - `args...`: More positional arguments, which the method ignores.
  - `kwargs...`: More keyword arguments, which the method ignores.

# Returns

  - `rb::RiskBudget`: The input `RiskBudget`, unchanged.

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
    risk_budget_constraints(rb::EstValType{<:VectorAbstractEstimatorValueAlgorithm},
                            sets::UniverseSets, dval::Option{<:Number} = nothing,
                            key::Option{<:AbstractString} = nothing; strict::Bool = false,
                            kwargs...)

Resolves a map from asset or group names to risk budget values against a universe.

The map can be a dictionary, a pair or a vector of pairs. `rb` can also be a scalar, a vector or an algorithmic rule such as [`UniformValues`](@ref). The method resolves each name against the universe that `key` selects in `sets`, and normalises the result to sum to one.

A scalar `rb` resolves to a scalar, not to a uniform vector, so the normalisation returns `1.0` for every value of the scalar. An optimiser accepts this budget only for a one-asset universe. Write the uniform budget as `nothing`, which the method for `Nothing` turns into `1/N` over `N` entries.

The normalisation divides by a computed sum, so the sum of the result is within round-off of one, and not always exactly one.

# Algorithm

 1. Take `dval` as the fill value. If `dval` is `nothing`, use the reciprocal of the length of the universe that `key` selects. If `key` is also `nothing`, use the universe of `sets.xkey`.
 2. Resolve `rb` against `sets` with [`estimator_to_val`](@ref), and call the result `val`. An asset name writes one entry. A group name writes one entry for each member of the group. Each unnamed entry keeps the fill value of step 1.
 3. Divide `val` by `sum(val)`.
 4. Wrap the result in a [`RiskBudget`](@ref), which checks that it is non-empty and non-negative.

# Arguments

  - `rb`: Map from asset or group names to risk budget values, or a scalar, a vector or an algorithmic rule.
  - `sets`: [`UniverseSets`](@ref) that holds the universe and the groups.
  - `dval`: Fill value of each entry that `rb` does not name. If `nothing`, the method uses `1/length(sets.dict[key])`, with `sets.xkey` in place of a `nothing` key.
  - $(arg_dict[:ekey]) [`FactorRiskBudgeting`](@ref) passes the factor axis key of its own `re`, because its budget is written in factor names.
  - `strict`: If `true`, a name in `rb` that `sets` does not resolve raises `ArgumentError`. If `false`, the name logs a warning.

# Validation

  - A name in `rb` that names neither an asset nor a group of `sets` raises `ArgumentError` when `strict` is `true`. Otherwise the method logs a warning, and the name writes nothing.
  - A budget whose entries are all zero divides by zero in step 3. The result is a `NaN` vector, which fails the non-negativity check of [`RiskBudget`](@ref). The call raises `DomainError`, and does not return a `NaN` budget. A zero entry is valid, and only an all-zero budget raises this error.

# Returns

  - `rb::RiskBudget`: A [`RiskBudget`](@ref) that holds the normalised budget.

# Examples

```jldoctest
julia> sets = UniverseSets(; xkey = \"nx\",
                           dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"], \"group1\" => [\"A\", \"B\"]));

julia> risk_budget_constraints([\"A\" => 0.2, \"group1\" => 0.8], sets)
RiskBudget
  val ┴ Vector{Float64}: [0.41379310344827586, 0.41379310344827586, 0.17241379310344826]
```

A budget written in factor names resolves against the factor axis, which is the key that [`FactorRiskBudgeting`](@ref) passes. Each factor that the budget does not name takes the default `1/length(sets.dict[key])` before the normalisation:

```jldoctest
julia> sets = UniverseSets(; dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"], \"nf\" => [\"F1\", \"F2\"]));

julia> risk_budget_constraints(Dict(\"F1\" => 0.25), sets, nothing, sets.tfkey)
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
function risk_budget_constraints(rb::EstValType{<:VectorAbstractEstimatorValueAlgorithm},
                                 sets::UniverseSets, dval::Option{<:Number} = nothing,
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

Resolves a [`RiskBudgetEstimator`](@ref) against a universe.

This method calls the mapping method with the two fields of the estimator.

The key is a positional argument of this method, not a field of the estimator. [`ThresholdEstimator`](@ref) holds its key in a field, so [`threshold_constraints`](@ref) takes no key argument. Both functions call [`estimator_to_val`](@ref), and both resolve a key of the factor axis. On a universe of four assets and two factors, a budget that names only `F1` returns a vector of two entries.

This function has no vector method. A vector of estimators raises `MethodError`, because no broader method accepts it. [`RkbE_Rkb`](@ref) gives the reason.

# Algorithm

 1. Call `risk_budget_constraints(rb.val, sets, rb.dval, key; strict = strict, kwargs...)`. This call resolves the map, normalises it and makes the [`RiskBudget`](@ref).

# Arguments

  - `rb`: A [`RiskBudgetEstimator`](@ref) that holds the map and the fill value.
  - `sets`: [`UniverseSets`](@ref) that holds the universe and the groups.
  - $(arg_dict[:ekey]) [`FactorRiskBudgeting`](@ref) passes the factor axis key of its own `re`, because its budget is written in factor names.
  - `strict`: If `true`, a name in `rb.val` that `sets` does not resolve raises `ArgumentError`. If `false`, the name logs a warning.
  - `kwargs...`: More keyword arguments, which the method gives to the mapping method.

# Returns

  - `rb::RiskBudget`: A [`RiskBudget`](@ref) that holds the normalised budget.

# Examples

```jldoctest
julia> sets = UniverseSets(; dict = Dict(\"nx\" => [\"A\", \"B\", \"C\", \"D\"], \"nf\" => [\"F1\", \"F2\"]));

julia> risk_budget_constraints(RiskBudgetEstimator(; val = Dict(\"F1\" => 0.4)), sets, sets.tfkey)
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
