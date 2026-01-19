"""
    struct RiskBudget{T1} <: AbstractConstraintResult
        val::T1
    end

Container for the result of a risk budget constraint.

`RiskBudget` stores the vector of risk budget allocations resulting from risk budget constraint generation or normalisation. This type is used to encapsulate the output of risk budgeting routines in a consistent, composable format for downstream processing and reporting.

# Fields

  - `val`: Vector of risk budget allocations (typically `VecNum`).

# Constructor

    RiskBudget(; val::VecNum)

Keyword arguments correspond to the fields above.

## Validation

  - `!isempty(val)`.
  - `all(x -> zero(x) <= x, val)`.

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
"""
struct RiskBudget{T1} <: AbstractConstraintResult
    val::T1
    function RiskBudget(val::VecNum)
        @argcheck(!isempty(val))
        @argcheck(all(x -> zero(x) <= x, val))
        return new{typeof(val)}(val)
    end
end
function RiskBudget(; val::Num_VecNum)
    return RiskBudget(val)
end
function risk_budget_view(::Nothing, args...)
    return nothing
end
function risk_budget_view(rb::RiskBudget, i)
    val = nothing_scalar_array_view(rb.val, i)
    return RiskBudget(; val = val)
end
"""
    struct RiskBudgetEstimator{T1} <: AbstractConstraintEstimator
        val::T1
    end

Container for a risk budget allocation mapping or vector.

`RiskBudgetEstimator` stores a mapping from asset or group names to risk budget values, or a vector of such pairs, for use in risk budgeting constraint generation. This type enables composable and validated workflows for specifying risk budgets in portfolio optimisation routines.

# Fields

  - `val`: A dictionary, pair, or vector of pairs mapping asset or group names to risk budget values.

# Constructor

    RiskBudgetEstimator(; val::EstValType)

Keyword arguments correspond to the fields above.

## Validation

  - `val` is validated with [`assert_nonempty_nonneg_finite_val`](@ref).

# Examples

```jldoctest
julia> RiskBudgetEstimator(; val = Dict("A" => 0.2, "B" => 0.3, "C" => 0.5))
RiskBudgetEstimator
  val ┴ Dict{String, Float64}: Dict("B" => 0.3, "A" => 0.2, "C" => 0.5)

julia> RiskBudgetEstimator(; val = ["A" => 0.2, "B" => 0.3, "C" => 0.5])
RiskBudgetEstimator
  val ┴ Vector{Pair{String, Float64}}: ["A" => 0.2, "B" => 0.3, "C" => 0.5]
```

# Related

  - [`RiskBudget`](@ref)
  - [`risk_budget_constraints`](@ref)
  - [`AssetSets`](@ref)
"""
struct RiskBudgetEstimator{T1} <: AbstractConstraintEstimator
    val::T1
    function RiskBudgetEstimator(val::EstValType)
        assert_nonempty_nonneg_finite_val(val)
        return new{typeof(val)}(val)
    end
end
function RiskBudgetEstimator(; val::EstValType)
    return RiskBudgetEstimator(val)
end
const VecRkbE = AbstractVector{<:RiskBudgetEstimator}
const RkbE_Rkb = Union{<:RiskBudgetEstimator, <:RiskBudget}
function risk_budget_view(rb::RiskBudgetEstimator, ::Any)
    return rb
end
"""
    risk_budget_constraints(::Nothing, args...; N::Number, datatype::DataType = Float64,
                            kwargs...)

No-op fallback for risk budget constraint generation.

This method returns a uniform risk budget allocation when no explicit risk budget is not `nothing`. It creates a [`RiskBudget`](@ref) with equal weights summing to one, using the specified number of assets `N` and numeric type `datatype`. This is useful as a default in workflows where a risk budget is optional or omitted.

# Arguments

  - `::Nothing`: Indicates that no risk budget is not `nothing`.
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
function risk_budget_constraints(::Nothing, args...; N::Number, kwargs...)
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
function risk_budget_constraints(rb::RiskBudget, args...; kwargs...)
    return rb
end
"""
    risk_budget_constraints(rb::EstValType, sets::AssetSets;
                            N::Number = length(sets.dict[sets.key]), strict::Bool = false)

Generate a risk budget allocation from asset/group mappings and asset sets.

This method constructs a [`RiskBudget`](@ref) from a mapping of asset or group names to risk budget values, using the provided [`AssetSets`](@ref). The mapping can be a dictionary, a single pair, or a vector of pairs. Asset and group names are resolved using `sets`, and the resulting risk budget vector is normalised to sum to one.

# Arguments

  - `rb`: A dictionary, pair, or vector of pairs mapping asset or group names to risk budget values.
  - `sets`: An [`AssetSets`](@ref) object specifying the asset universe and groupings.
  - `N`: Number of assets in the universe.
  - `strict`: If `true`, throws an error if a key in `rb` is not found in `sets`; if `false`, issues a warning.

# Details

  - Asset and group names in `rb` are mapped to indices in the asset universe using `sets`.
  - If a key is a group, all assets in the group are assigned the specified value.
  - The resulting vector is normalised to sum to one.
  - If `strict` is `true`, missing keys cause an error; otherwise, a warning is issued.

# Returns

  - `rb::RiskBudget`: A result object containing the normalised risk budget vector.

# Examples

```jldoctest
julia> sets = AssetSets(; key = "nx", dict = Dict("nx" => ["A", "B", "C"], "group1" => ["A", "B"]));

julia> risk_budget_constraints(Dict("A" => 0.2, "group1" => 0.8), sets)
RiskBudget
  val ┴ Vector{Float64}: [0.41379310344827586, 0.41379310344827586, 0.17241379310344826]
```

# Related

  - [`RiskBudget`](@ref)
  - [`AssetSets`](@ref)
  - [`estimator_to_val`](@ref)
  - [`risk_budget_constraints`](@ref)
"""
function risk_budget_constraints(rb::EstValType, sets::AssetSets;
                                 N::Number = length(sets.dict[sets.key]),
                                 strict::Bool = false)
    val = estimator_to_val(rb, sets, inv(N); strict = strict)
    return RiskBudget(; val = val / sum(val))
end
"""
    risk_budget_constraints(rb::Union{<:RiskBudgetEstimator,
                                      <:VecRkbE}, sets::AssetSets;
                            strict::Bool = false, kwargs...)

If `rb` is a vector of [`RiskBudgetEstimator`](@ref) objects, this function is broadcast over the vector.

This method is a wrapper calling:

    risk_budget_constraints(rb.val, sets; strict = strict)

It is used for type stability and to provide a uniform interface for processing constraint estimators, as well as simplifying the use of multiple estimators simulatneously.

# Related

  - [`risk_budget_constraints`](@ref)
"""
function risk_budget_constraints(rb::RiskBudgetEstimator, sets::AssetSets;
                                 strict::Bool = false, kwargs...)
    return risk_budget_constraints(rb.val, sets; strict = strict)
end
function risk_budget_constraints(rb::VecRkbE, sets::AssetSets; strict::Bool = false,
                                 kwargs...)
    return [risk_budget_constraints(rbi, sets; strict = strict) for rbi in rb]
end

export RiskBudget, RiskBudgetEstimator, risk_budget_constraints
