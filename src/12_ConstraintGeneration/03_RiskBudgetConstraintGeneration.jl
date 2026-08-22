"""
$(DocStringExtensions.TYPEDEF)

Carries the vector of non-negative risk budgets that a risk-budgeting optimiser targets.

The budget names how much of the total risk each entry of an axis is asked to carry. [`AssetRiskBudgeting`](@ref) writes it over the assets and [`FactorRiskBudgeting`](@ref) over the factors.

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

Only the **relative** entries of ``\\boldsymbol{b}`` reach the solution: scaling the whole vector shifts the barrier by a constant. [`risk_budget_constraints`](@ref) normalises it to sum to one anyway, so a budget that arrives through an estimator always does.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RiskBudget(;
        val::Num_VecNum
    ) -> RiskBudget

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(val)`.
  - `all(x -> zero(x) <= x, val)`.

Both checks run in the keyword constructor as well as in the inner one. `@concrete` emits a generic single-argument constructor that a scalar `val` matches ahead of the inner `VecNum` method, so a scalar reaches the struct without passing either check unless the keyword constructor applies them itself.

A scalar `val` is a real input, not a mistake to refuse: a scalar [`RiskBudgetEstimator`](@ref) resolves to one, because [`risk_budget_constraints`](@ref) normalises the budget to sum to one and a scalar divided by itself is `1.0`. Such a budget states one allocation over a one-entry axis, and an optimiser accepts it only for a one-asset universe.

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
  - [`risk_budget_constraints`](@ref)
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
    function RiskBudget(val::VecNum)::RiskBudget
        @argcheck(!isempty(val), IsEmptyError("val cannot be empty"))
        @argcheck(all(x -> zero(x) <= x, val),
                  DomainError(val, "all entries of val must be >= 0"))
        return new{typeof(val)}(val)
    end
end
function RiskBudget(; val::Num_VecNum)::RiskBudget
    # The guards live here as well as in the inner constructor. `@concrete` emits a generic
    # `RiskBudget(val::__T_val)` that is more specific than the inner `VecNum` method for a
    # scalar, so a scalar reaches the struct without passing either check. A scalar is a
    # real input -- `risk_budget_constraints` on a scalar estimator ends at
    # `RiskBudget(; val = val / sum(val))`, which is `1.0` -- so it is validated, not refused.
    @argcheck(!isempty(val), IsEmptyError("val cannot be empty"))
    @argcheck(all(x -> zero(x) <= x, val),
              DomainError(val, "all entries of val must be >= 0"))
    return RiskBudget(val)
end
"""
$(DocStringExtensions.TYPEDEF)

Resolves a risk budget written in asset or group names against a universe.

[`risk_budget_constraints`](@ref) turns it into a [`RiskBudget`](@ref): every name in `val` is mapped to its indices in the selected universe, an unnamed asset takes `dval`, and the result is normalised to sum to one. A group name assigns its value to every asset in the group.

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

# Related

  - [`RiskBudgetEstimator`](@ref)
  - [`RiskBudget`](@ref)
  - [`risk_budget_constraints`](@ref)
"""
const RkbE_Rkb = Union{<:RiskBudgetEstimator, <:RiskBudget}
"""
    risk_budget_constraints(::Nothing, args...; N::Integer, kwargs...)

No-op fallback for risk budget constraint generation.

This method returns a uniform risk budget allocation when no explicit risk budget is specified (`nothing`). It creates a [`RiskBudget`](@ref) with equal budgets summing to one over `N` entries. This is useful as a default in workflows where a risk budget is optional or omitted.

The vector is a constant `range`, not an `Array`, and its element type is always `Float64` because it comes from `inv(N)`. This method takes no `datatype` keyword; one passed here is swallowed by `kwargs...` and changes nothing.

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
```

The vector is passed through as given. This method does **not** normalise it, unlike the estimator method, so a hand-built budget keeps whatever sum it was written with.

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

# Arguments

  - `rb`: A dictionary, pair, or vector of pairs mapping asset or group names to risk budget values.
  - `sets`: A [`UniverseSets`](@ref) object specifying the universe and groupings.
  - `dval`: Default value to use for names not found in `rb`. If `nothing`, a default value of `1/length(sets.dict[key])` is used.
  - $(arg_dict[:ekey]) [`FactorRiskBudgeting`](@ref) passes `sets.fkey`, because its budget is written in factor names.
  - `strict`: If `true`, throws an error if a key in `rb` is not found in `sets`; if `false`, issues a warning.

# Details

  - Names and groups in `rb` are mapped to indices in the selected universe using `sets`.
  - If a key is a group, all assets in the group are assigned the specified value.
  - The resulting vector is normalised to sum to one.
  - If `strict` is `true`, missing keys cause an error; otherwise, a warning is issued.

A **scalar** `rb` resolves to a scalar, not to a uniform vector, so the normalisation returns `1.0` whatever the scalar was. The result is a `RiskBudget` holding one number, and an optimiser accepts it only for a one-asset universe. Write the uniform budget as `nothing`, which the no-op method turns into `1/N` over `N` entries.

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

This method is a wrapper calling:

    risk_budget_constraints(rb.val, sets, rb.dval, key; strict = strict)

It is used for type stability and to provide a uniform interface for processing constraint estimators, as well as simplifying the use of multiple estimators simulatneously.

# Related

  - [`risk_budget_constraints`](@ref)
"""
function risk_budget_constraints(rb::RiskBudgetEstimator, sets::UniverseSets,
                                 key::Option{<:AbstractString} = nothing;
                                 strict::Bool = false, kwargs...)::RiskBudget
    return risk_budget_constraints(rb.val, sets, rb.dval, key; strict = strict, kwargs...)
end
export RiskBudget, RiskBudgetEstimator, risk_budget_constraints
