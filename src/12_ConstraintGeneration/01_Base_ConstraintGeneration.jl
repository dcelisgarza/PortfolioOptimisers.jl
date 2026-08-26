"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all constraint result types.

All concrete and/or abstract types representing the result of constraint generation or evaluation should be subtypes of `AbstractConstraintResult`.

# Related

  - [`AbstractConstraintEstimator`](@ref)
  - [`AbstractResult`](@ref)
"""
abstract type AbstractConstraintResult <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all constraint estimator types.

All concrete and/or abstract types implementing constraint generation or estimation algorithms should be subtypes of `AbstractConstraintEstimator`.

# Related

  - [`AbstractConstraintResult`](@ref)
  - [`AbstractEstimator`](@ref)
"""
abstract type AbstractConstraintEstimator <: AbstractEstimator end
"""
    const ComparisonOperator = Union{typeof(==), typeof(<=), typeof(>=)}

Union type representing supported comparison operators for constraint generation.

This type is used to specify which comparison operators are valid for defining constraints. It includes equality and both directions of inequality.

# Related

  - [`comparison_sign_ineq_flag`](@ref)
"""
const ComparisonOperator = Union{typeof(==), typeof(<=), typeof(>=)}
"""
    comparison_sign_ineq_flag(op::ComparisonOperator)
    comparison_sign_ineq_flag(op::AbstractString)

Return the multiplicative sign and inequality flag for a given comparison operator.

This is the one table mapping a comparison operator to the pair every constraint generator needs: the sign that files a `>=` row as a `<=` row, and the flag that sorts the row into the inequality block or the equality block. A parsed constraint carries its operator as a `String`, and a constraint estimator carries it as a function, so the table dispatches on both spellings and neither caller writes its own copy.

# Arguments

  - `op::ComparisonOperator`: The comparison operator, as a function.
  - `op::AbstractString`: The comparison operator, as the string a [`ParsingResult`](@ref) carries.

# Validation

  - A string `op` is one of `"=="`, `"<="` or `">="`.

# Returns

  - `sign::Int`: The multiplicative sign for the constraint.
  - `is_inequality::Bool`: `true` if the operator is an inequality, `false` for equality.

# Examples

```jldoctest
julia> PortfolioOptimisers.comparison_sign_ineq_flag(==)
(1, false)

julia> PortfolioOptimisers.comparison_sign_ineq_flag(<=)
(1, true)

julia> PortfolioOptimisers.comparison_sign_ineq_flag(>=)
(-1, true)

julia> PortfolioOptimisers.comparison_sign_ineq_flag(\">=\")
(-1, true)
```

# Related

  - [`ComparisonOperator`](@ref)
  - [`ParsingResult`](@ref)
"""
function comparison_sign_ineq_flag(::typeof(==))::Tuple{Int, Bool}
    return 1, false
end
function comparison_sign_ineq_flag(::typeof(<=))::Tuple{Int, Bool}
    return 1, true
end
function comparison_sign_ineq_flag(::typeof(>=))::Tuple{Int, Bool}
    return -1, true
end
function comparison_sign_ineq_flag(op::AbstractString)::Tuple{Int, Bool}
    return if op == "=="
        1, false
    elseif op == "<="
        1, true
    elseif op == ">="
        -1, true
    else
        throw(ArgumentError("`op` must be one of \"==\", \"<=\", \">=\". Got\nop => $op"))
    end
end
"""
    resolve_axis_name(name, nx::AbstractVector, sdict::AbstractDict) -> Option{Vector}

Resolve one name against an axis: an axis entry names itself, a `sdict` key expands to its members, and an unknown name gives `nothing`.

This is the precedence every name-taking constraint generator uses — asset first, then group — written once. The caller diagnoses the `nothing`, because the suggestion pool differs by caller. The member vector is a **copy**: `sdict` is the caller's [`UniverseSets`](@ref) dictionary, which is configuration reused across folds and optimisers, so de-duplication must never edit it.

# Arguments

  - `name`: The name to resolve.
  - `nx`: The axis, usually the asset universe.
  - `sdict`: Dictionary mapping group names to vectors of member names.

# Returns

  - `[name]` if `name` is on the axis.
  - `unique(members)` if `name` is a key of `sdict`.
  - `nothing` if `name` is neither.

# Related

  - [`axis_name_indices`](@ref)
  - [`name_to_val!`](@ref)
  - [`estimator_to_val`](@ref)
  - [`UniverseSets`](@ref)
"""
function resolve_axis_name(name, nx::AbstractVector, sdict::AbstractDict)
    if name in nx
        return [name]
    end
    members = get(sdict, name, nothing)
    return isnothing(members) ? nothing : unique(members)
end
"""
    axis_name_indices(members, nx::AbstractVector, on_missing) -> Vector

Map resolved member names to axis indices, drop the members that miss the axis, and report them once through `on_missing`.

`on_missing` takes the vector of missing members and decides the policy, so a caller can throw, warn, or stay silent without a second copy of the mapping. `on_missing` is not called when every member is on the axis.

# Arguments

  - `members`: Member names, as [`resolve_axis_name`](@ref) returns them.
  - `nx`: The axis, usually the asset universe.
  - `on_missing`: Callable applied to the vector of members that miss the axis.

# Returns

  - `idx`: Axis indices of the members that are on the axis, in the order of `members`.

# Related

  - [`resolve_axis_name`](@ref)
  - [`missing_group_assets_msg`](@ref)
"""
function axis_name_indices(members, nx::AbstractVector, on_missing)
    idx = [findfirst(x -> isequal(x, m), nx) for m in members]
    missing_members = members[isnothing.(idx)]
    filter!(!isnothing, idx)
    if !isempty(missing_members)
        on_missing(missing_members)
    end
    return idx
end
