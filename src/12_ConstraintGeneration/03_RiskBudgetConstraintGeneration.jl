"""
$(DocStringExtensions.TYPEDEF)

Container for the result of a risk budget constraint.

`RiskBudget` stores the vector of risk budget allocations resulting from risk budget constraint generation or normalisation. This type is used to encapsulate the output of risk budgeting routines in a consistent, composable format for downstream processing and reporting.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RiskBudget(;
        val::VecNum
    ) -> RiskBudget

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(val)`.
  - `all(x -> zero(x) <= x, val)`.

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
  - [`port_opt_view`](@ref)
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
    return RiskBudget(val)
end
"""
$(DocStringExtensions.TYPEDEF)

Container for a risk budget allocation mapping or vector.

`RiskBudgetEstimator` stores a mapping from asset or group names to risk budget values, or a vector of such pairs, for use in risk budgeting constraint generation. This type enables composable and validated workflows for specifying risk budgets in portfolio optimisation routines.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RiskBudgetEstimator(;
        val::EstValType
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

# Related

  - [`RiskBudget`](@ref)
  - [`risk_budget_constraints`](@ref)
  - [`UniverseSets`](@ref)
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
    risk_budget_constraints(::Nothing, args...; N::Number, datatype::DataType = Float64,
                            kwargs...)

No-op fallback for risk budget constraint generation.

This method returns a uniform risk budget allocation when no explicit risk budget is specified (`nothing`). It creates a [`RiskBudget`](@ref) with equal weights summing to one, using the specified number of assets `N` and numeric type `datatype`. This is useful as a default in workflows where a risk budget is optional or omitted.

# Arguments

  - `::Nothing`: Indicates that no explicit risk budget is specified.
  - `args...`: Additional positional arguments (ignored).
  - `N::Number`: Number of assets (required).
  - `datatype::DataType`: Numeric type for the risk budget vector.
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
julia> RiskBudget(; val = [0.2, 0.3, 0.5])
RiskBudget
  val ┴ Vector{Float64}: [0.2, 0.3, 0.5]
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
