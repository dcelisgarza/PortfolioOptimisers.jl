"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all constraint result types.

A subtype holds the output of a constraint generator. Most subtypes are assembled constraints, and a field of a [`JuMPOptimiser`](@ref) accepts them as precomputed values.

# Interfaces

To make a new constraint result, subtype `AbstractConstraintResult` and put the assembled constraint in its fields. Each method below has a fallback for the whole library. A result that the fallback describes correctly implements neither method. A result that the fallback describes wrongly implements one or both.

## Routing target

  - `PortfolioOptimisers.implicit_constraint_target(res::AbstractConstraintResult) -> Option{Symbol}`: The one [routing target](@ref PIPELINE_ROUTING_TARGETS) that a result of this type can go to, or `nothing` when the type names none.

The fallback returns `nothing`. `add_constraint_result` then puts the value in a [`TargetedConstraint`](@ref), with the target that its step declared. Implement the method only when the type names exactly one field of a [`JuMPOptimiser`](@ref). Only then can the pipeline put the value in that field without the wrapper.

### Arguments

  - `res`: The constraint result.

### Returns

  - `target::Option{Symbol}`: One of [`PIPELINE_ROUTING_TARGETS`](@ref), or `nothing`.

## Asset view

  - `PortfolioOptimisers.port_opt_view(res::AbstractConstraintResult, i, args...) -> AbstractConstraintResult`: A copy of the result over the selected assets.

The generic fallback slices an array, and returns a scalar, `nothing`, an estimator or an algorithm unchanged. It has no method for a result struct, and raises a `MethodError` on one. A result that a view can reach needs its own method. Tag each field that the view slices with `@vprop`, and `@propagatable` makes the method. Write the method by hand when the tags cannot state the view. [`LinearConstraint`](@ref) is that case. Its method reads neither the index nor the arguments after it, and the method states why the identity is correct.

### Arguments

  - `res`: The constraint result.
  - `i`: The asset index the view keeps.

### Returns

  - `res::AbstractConstraintResult`: The constraint over the selected assets.

# Related

  - [`AbstractConstraintEstimator`](@ref)
  - [`AbstractResult`](@ref)
  - [`constraint_results`](@ref)
  - [`implicit_constraint_target`](@ref)
  - [`port_opt_view`](@ref)
"""
abstract type AbstractConstraintResult <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all constraint estimator types.

A subtype holds the configuration of one constraint family. The generation verb of the family turns the configuration into a constraint over a universe.

# Interfaces

To make a new constraint family, subtype `AbstractConstraintEstimator` and put its parameters in the fields of the struct. Then implement the methods below.

## Constraint generation

  - `<family>_constraints(ce::AbstractConstraintEstimator, sets::UniverseSets)`: Assemble the constraint of the family over the declared universe.

Each family has its own verb, and the library has no shared verb name. The verbs are [`linear_constraints`](@ref), [`weight_bounds_constraints`](@ref), [`threshold_constraints`](@ref), [`risk_budget_constraints`](@ref), [`asset_sets_matrix`](@ref), [`phylogeny_constraints`](@ref) and [`centrality_constraints`](@ref). A family that fits its structure to data takes a [`ReturnsResult`](@ref) in place of the sets. A caller calls this method directly, and the pipeline step below calls it too.

### Arguments

  - `ce`: The constraint estimator.
  - `sets`: The declared universe the names resolve against.

### Returns

  - `res`: The assembled constraint. Every family except the asset sets matrix returns an [`AbstractConstraintResult`](@ref). [`asset_sets_matrix`](@ref) returns a plain membership matrix, so the [routing target](@ref PIPELINE_ROUTING_TARGETS) tells the pipeline where the matrix goes, and the type does not.

## Pipeline step

Implement both methods, or neither. The fallback of [`pipe_constraint_targets`](@ref) is an empty tuple. That tuple declares that the family computes no value for the `constraints` slot, so the family is not a [`Pipeline`](@ref) step. `resolve_constraint_target` raises on the empty tuple before it reads the value. [`JuMPConstraintEstimator`](@ref) is the family that takes the fallback, because it configures the model and computes nothing from data.

  - `PortfolioOptimisers.pipe_constraint_targets(ce::AbstractConstraintEstimator) -> Tuple`: The [routing targets](@ref PIPELINE_ROUTING_TARGETS) that the result of the family can go to. With one target, the step places the value with no annotation. With more than one, the [`PipelineStep`](@ref) wrapper must name one.
  - `PortfolioOptimisers.constraint_step_value(ce::AbstractConstraintEstimator, ctx::PipelineContext)`: The value that the step adds. The method calls the generation verb of the family to compute it.

`pipe_reads` and `pipe_writes` need no method. They return `(:returns,)` and `:constraints` for every subtype. A family that also reads a computed slot declares it with its own `pipe_reads` method, as [`ExposureConstraintEstimator`](@ref) does for `:prior`.

### Arguments

  - `ce`: The constraint estimator.
  - `ctx`: The pipeline context.

### Returns

  - `res`: The computed value, `nothing`, or a vector of either.

# Related

  - [`AbstractConstraintResult`](@ref)
  - [`AbstractEstimator`](@ref)
  - [`constraint_step_value`](@ref)
  - [`pipe_constraint_targets`](@ref)
  - [`run_constraint_step`](@ref)
"""
abstract type AbstractConstraintEstimator <: AbstractEstimator end
"""
    const ComparisonOperator = Union{typeof(==), typeof(<=), typeof(>=)}

Union of the comparison operators that a constraint accepts.

It matches the equality `==` and the two inequalities `<=` and `>=`.

The union is a type bound, not a check. A field that it bounds refuses a fourth operator when the constructor stores the value. Thus `CentralityConstraint(; comp = <)` raises a `TypeError` from the keyword constructor. No constraint generator needs a branch for an operator that it cannot receive.

# Related

  - [`comparison_sign_ineq_flag`](@ref)
  - [`CentralityConstraint`](@ref)
"""
const ComparisonOperator = Union{typeof(==), typeof(<=), typeof(>=)}
"""
    comparison_sign_ineq_flag(op::ComparisonOperator)
    comparison_sign_ineq_flag(op::AbstractString)

Return the sign and the inequality flag of a comparison operator.

This function is the one table from a comparison operator to the pair that every constraint generator needs. The sign turns a `>=` row into a `<=` row. The flag puts the row in the inequality block or in the equality block. A parsed constraint holds its operator as a `String`, and a constraint estimator holds it as a function. The table has a method for each form, so no caller keeps its own copy.

# Arguments

  - `op::ComparisonOperator`: The comparison operator, as a function.
  - `op::AbstractString`: The comparison operator, as the string a [`ParsingResult`](@ref) carries.

# Validation

  - A string `op` is one of `"=="`, `"<="` or `">="`. Another string raises an `ArgumentError` that names the three.
  - A function `op` outside [`ComparisonOperator`](@ref) matches no method and raises a `MethodError`. A field that the alias bounds refuses such an operator before this, at construction.

# Returns

  - `sign::Int`: The sign that multiplies the row and its right-hand side.
  - `is_inequality::Bool`: `true` for an inequality, `false` for an equality.

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

Resolve one name against an axis. An entry of the axis resolves to itself, a key of `sdict` resolves to its members, and an unknown name gives `nothing`.

Every constraint generator that takes names uses this order, asset first and then group, and this function holds it once. The caller diagnoses `nothing`, because each caller suggests names from a different pool. The member vector is a copy. `sdict` is the dictionary of the [`UniverseSets`](@ref) of the caller. The removal of duplicates must not change it, because folds and optimisers reuse it as configuration.

# Algorithm

 1. When `name` is on `nx`, return `[name]`. The function reads the axis before the dictionary. Thus an asset and a group with the same name resolve to the asset, and the group is unreachable.
 2. Otherwise read `members`, the entry of `sdict` for `name`, or `nothing` when `sdict` has no such entry.
 3. Return `unique(members)`, so a group that names one asset twice gives one member. Return `nothing` when `members` is `nothing`.

# Arguments

  - `name`: The name to resolve.
  - `nx`: The axis, usually the asset universe.
  - `sdict`: Dictionary from group names to vectors of member names.

# Returns

  - `[name]` if `name` is on the axis.
  - `unique(members)` if `name` is a key of `sdict`.
  - `nothing` if `name` is neither.

# Examples

An asset name wins over a group with the same name, and a repeated member counts once.

```jldoctest
julia> nx = [\"A\", \"B\", \"C\"];

julia> sdict = Dict(\"A\" => [\"B\", \"C\"], \"G\" => [\"A\", \"A\", \"B\"]);

julia> PortfolioOptimisers.resolve_axis_name(\"A\", nx, sdict)
1-element Vector{String}:
 \"A\"

julia> PortfolioOptimisers.resolve_axis_name(\"G\", nx, sdict)
2-element Vector{String}:
 \"A\"
 \"B\"

julia> isnothing(PortfolioOptimisers.resolve_axis_name(\"Q\", nx, sdict))
true
```

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

Map resolved member names to indices on an axis, drop the members that are not on the axis, and report them once through `on_missing`.

`on_missing` receives the vector of missing members and applies the policy of the caller. Each caller can then throw, warn or do nothing, and the mapping has one copy. The function does not call `on_missing` when every member is on the axis.

# Algorithm

 1. Map each member to its first index on `nx`, or to `nothing` when it is not on `nx`, and call the result `idx`. A repeated axis name is reachable only at its first index.
 2. Collect `missing_members`, the members whose index is `nothing`.
 3. Drop the `nothing`s from `idx`.
 4. When `missing_members` is not empty, call `on_missing` on it. This call comes after step 3. Thus a callback that throws discards the indices that remain, and a callback that warns gives them to the caller.

# Arguments

  - `members`: Member names, as [`resolve_axis_name`](@ref) returns them.
  - `nx`: The axis, usually the asset universe.
  - `on_missing`: Function that receives the vector of members that are not on the axis.

# Returns

  - `idx`: The indices on the axis of the members that are on it, in the order of `members`. The element type is `Int` when no member is missing, and `Union{Nothing, Int}` when one is. Step 3 removes the `nothing`s from the vector, but not from its element type.

# Examples

```jldoctest
julia> nx = [\"A\", \"B\", \"C\"];

julia> PortfolioOptimisers.axis_name_indices([\"C\", \"A\"], nx, m -> nothing)
2-element Vector{Int64}:
 3
 1

julia> missed = String[];

julia> PortfolioOptimisers.axis_name_indices([\"A\", \"Z\", \"C\"], nx, m -> append!(missed, m))
2-element Vector{Union{Nothing, Int64}}:
 1
 3

julia> missed
1-element Vector{String}:
 \"Z\"
```

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

public AbstractConstraintResult, AbstractConstraintEstimator, implicit_constraint_target,
       pipe_constraint_targets, constraint_step_value
