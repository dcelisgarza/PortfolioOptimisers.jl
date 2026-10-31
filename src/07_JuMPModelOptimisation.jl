"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of the records that a solve of a JuMP model returns.

A subtype records what happened when [`optimise_JuMP_model!`](@ref) tried the solvers on a model. It holds no solution values. After a solve that succeeds, the caller reads them from the model.

# Related

  - [`JuMPResult`](@ref)
  - [`optimise_JuMP_model!`](@ref)
"""
abstract type AbstractJuMPResult <: AbstractResult end
"""
    const Dict_VecPair = Union{AbstractDict, AbstractVector{<:Pair}}

Groups a dictionary and a vector of pairs, the two collections of solver attributes that [`set_solver_attributes`](@ref) applies.

Both collections iterate as pairs of a key and a value. The key names a solver attribute, and the value is the value to set.

# Related

  - [`SlvSettings`](@ref)
  - [`Solver`](@ref)
  - [`set_solver_attributes`](@ref)
"""
const Dict_VecPair = Union{AbstractDict, AbstractVector{<:Pair}}
"""
    const SlvSettings = Union{Pair, Dict_VecPair}

Groups the two forms that the `settings` of a [`Solver`](@ref) accepts.

The forms are one pair of a solver attribute and its value, and a [`Dict_VecPair`](@ref) of such pairs.

# Related

  - [`Dict_VecPair`](@ref)
  - [`Solver`](@ref)
  - [`set_solver_attributes`](@ref)
"""
const SlvSettings = Union{Pair, Dict_VecPair}
"""
    const SlvKeys = Union{AbstractString, JuMP.MOI.AbstractModelAttribute,
                          JuMP.MOI.AbstractOptimizerAttribute}

Groups the three kinds of key that `JuMP.set_attribute` accepts for an attribute of a model.

The kinds are the name of a raw solver attribute as a string, a `MOI.AbstractModelAttribute`, and a `MOI.AbstractOptimizerAttribute` such as `MOI.Silent()` or `MOI.TimeLimitSec()`. [`Solver`](@ref) refuses a key of `settings` that is not in this group.

# Related

  - [`SlvSettings`](@ref)
  - [`Solver`](@ref)
  - [`set_solver_attributes`](@ref)
"""
const SlvKeys = Union{AbstractString, JuMP.MOI.AbstractModelAttribute,
                      JuMP.MOI.AbstractOptimizerAttribute}
"""
$(DocStringExtensions.TYPEDEF)

Names one JuMP solver, the attributes to set on it, and the solution statuses to accept.

An estimator that solves a JuMP model takes one `Solver` or a vector of them. [`optimise_JuMP_model!`](@ref) tries the solvers of a vector in order, and stops at the first solution that the `check_sol` of its solver accepts.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Solver(;
        name::Sym_Str = "",
        solver::Any,
        settings::Option{<:SlvSettings} = nothing,
        check_sol::NamedTuple = (;),
        add_bridges::Bool = true
    ) -> Solver

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:settings])

# Examples

The default `check_sol` accepts a solution only when the solver reports `OPTIMAL` or `LOCALLY_SOLVED`, and a `FEASIBLE_POINT`.

```jldoctest
julia> Solver(; solver = nothing)
Solver
         name ┼ String: ""
       solver ┼ nothing
     settings ┼ nothing
    check_sol ┼ @NamedTuple{}: NamedTuple()
  add_bridges ┴ Bool: true
```

This `check_sol` also accepts an approximate solution. The examples, the user guide and the tests use it.

```jldoctest
julia> Solver(; solver = nothing, check_sol = (; allow_local = true, allow_almost = true))
Solver
         name ┼ String: ""
       solver ┼ nothing
     settings ┼ nothing
    check_sol ┼ @NamedTuple{allow_local::Bool, allow_almost::Bool}: (allow_local = true, allow_almost = true)
  add_bridges ┴ Bool: true
```

# Related

  - [`optimise_JuMP_model!`](@ref)
  - [`set_optimizer`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.set_optimizer)
  - [`set_attribute`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.set_attribute)
  - [`Sym_Str`](@ref)
  - [`Option`](@ref)
  - [`SlvSettings`](@ref)
"""
@concrete struct Solver <: AbstractEstimator
    """
    $(field_dict[:name])
    """
    name
    """
    $(field_dict[:solver])
    """
    solver
    """
    $(field_dict[:settings])
    """
    settings
    """
    $(field_dict[:check_sol])
    """
    check_sol
    """
    $(field_dict[:add_bridges])
    """
    add_bridges
    function Solver(name::Sym_Str, solver::Any, settings::Option{<:SlvSettings},
                    check_sol::NamedTuple, add_bridges::Bool)::Solver
        if isa(settings, Dict_VecPair)
            @argcheck(!isempty(settings), IsEmptyError)
            if isa(settings, AbstractVector)
                @argcheck(all(x -> isa(x[1], SlvKeys), settings),
                          ArgumentError("every key of settings must be a SlvKeys (AbstractString, MOI.AbstractModelAttribute or MOI.AbstractOptimizerAttribute)"))
            else
                @argcheck(all(x -> isa(x, SlvKeys), keys(settings)),
                          ArgumentError("every key of settings must be a SlvKeys (AbstractString, MOI.AbstractModelAttribute or MOI.AbstractOptimizerAttribute)"))
            end
        elseif isa(settings, Pair)
            @argcheck(isa(settings[1], SlvKeys),
                      ArgumentError("settings[1] must be a SlvKeys (AbstractString, MOI.AbstractModelAttribute or MOI.AbstractOptimizerAttribute), got $(settings[1])"))
        end
        return new{typeof(name), typeof(solver), typeof(settings), typeof(check_sol),
                   typeof(add_bridges)}(name, solver, settings, check_sol, add_bridges)
    end
end
function Solver(; name::Sym_Str = "", solver::Any,
                settings::Option{<:SlvSettings} = nothing, check_sol::NamedTuple = (;),
                add_bridges::Bool = true)::Solver
    return Solver(name, solver, settings, check_sol, add_bridges)
end
"""
    const VecSlv = AbstractVector{<:Solver}

Groups the vectors of [`Solver`](@ref)s.

[`optimise_JuMP_model!`](@ref) tries the solvers of such a vector in order, until one of them gives an accepted solution.

# Related

  - [`Solver`](@ref)
"""
const VecSlv = AbstractVector{<:Solver}
"""
    const Slv_VecSlv = Union{Solver, VecSlv}

Groups one [`Solver`](@ref) and a vector of them, the two forms in which an estimator takes its solvers.

A single `Solver` iterates as a vector of one solver. So [`optimise_JuMP_model!`](@ref) takes both forms through one loop.

# Related

  - [`Solver`](@ref)
  - [`VecSlv`](@ref)
"""
const Slv_VecSlv = Union{Solver, VecSlv}
"""
$(DocStringExtensions.TYPEDEF)

Records which solvers failed, at which stage, and whether one of them succeeded.

The constructor warns when `success` is `false`. [`failed_solve_msg`](@ref) writes the warning, with one line for each failed stage of each solver. A line holds the name of the solver, the stage and the first line of the error. The warning holds no settings and no full exception, and `trials` keeps them.

`trials` records failures only. A solver that succeeds adds no entry, so an empty `trials` with `success = true` means that the first solver succeeded. The key of an entry is the `name` of the solver. The value is a dictionary from the failed stage, `:set_optimizer`, `:optimize!` or `:assert_is_solved_and_feasible`, to its exception. An entry of the last stage also holds the termination status of the solver under `:termination_status`, so the warning can state that the problem is infeasible or unbounded, and the `settings` of the solver under `:settings`.

Two solvers that share a name share one entry. Every solver has the default name `\"\"`, so a vector of unnamed solvers keeps only its last failure.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    JuMPResult(;
        trials::AbstractDict,
        success::Bool
    ) -> JuMPResult

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> JuMPResult(; trials = Dict(:HiGHS => Dict(:optimize! => \"error\")), success = true)
JuMPResult
   trials ┼ Dict{Symbol, Dict{Symbol, String}}: Dict(:HiGHS => Dict(:optimize! => "error"))
  success ┴ Bool: true
```

# Related

  - [`optimise_JuMP_model!`](@ref)
"""
@concrete struct JuMPResult <: AbstractJuMPResult
    """
    The failures of the solvers. The key of an entry is the name of a solver, and the value maps each failed stage to its exception.
    """
    trials
    """
    `true` when one solver gives a solution that its `check_sol` accepts.
    """
    success
    function JuMPResult(trials::AbstractDict, success::Bool)::JuMPResult
        if !success
            # Summarised via the shared builder: one bounded line per failed stage, never
            # the whole trials dict, the solver settings, or full exception payloads.
            @warn(failed_solve_msg(trials), maxlog = log_limit(:warn), _id = @log_id())
        end
        return new{typeof(trials), typeof(success)}(trials, success)
    end
end
function JuMPResult(; trials::AbstractDict, success::Bool)::JuMPResult
    return JuMPResult(trials, success)
end
"""
    set_solver_attributes(args...)

Set no solver attribute.

[`optimise_JuMP_model!`](@ref) calls `set_solver_attributes(model, nothing)` for a [`Solver`](@ref) whose `settings` is `nothing`, and this method takes that call.

# Arguments

  - `args...`: Ignored.

# Returns

  - `nothing`.

# Examples

```jldoctest
julia> PortfolioOptimisers.set_solver_attributes()

```

# Related

  - [`set_solver_attributes`](@ref)
  - [`Solver`](@ref)
"""
function set_solver_attributes(args...)::Nothing
    return nothing
end
"""
    set_solver_attributes(model::JuMP.Model, settings::Dict_VecPair)

Set each solver attribute of `settings` on `model`, in the iteration order of `settings`.

An attribute that the solver refuses raises the error of `JuMP.set_attribute`, and the attributes before it stay set.

# Arguments

  - `model`: The JuMP model, with a solver attached.
  - `settings`: The pairs of a solver attribute and its value.

# Returns

  - `nothing`.

# Related

  - [`JuMP.Model`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.Model)
  - [`set_attribute`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.set_attribute)
  - [`Dict_VecPair`](@ref)
"""
function set_solver_attributes(model::JuMP.Model, settings::Dict_VecPair)::Nothing
    for (k, v) in settings
        JuMP.set_attribute(model, k, v)
    end
    return nothing
end
"""
    set_solver_attributes(model::JuMP.Model, settings::Pair)

Set one solver attribute on `model`.

An attribute that the solver refuses raises the error of `JuMP.set_attribute`.

# Arguments

  - `model`: The JuMP model, with a solver attached.
  - `settings`: The pair of a solver attribute and its value.

# Returns

  - `nothing`.

# Related

  - [`JuMP.Model`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.Model)
  - [`set_attribute`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.set_attribute)
"""
function set_solver_attributes(model::JuMP.Model, settings::Pair)::Nothing
    JuMP.set_attribute(model, settings...)
    return nothing
end
"""
    optimise_JuMP_model!(model::JuMP.Model, slv::Slv_VecSlv) -> JuMPResult

Solve `model` with each solver of `slv` in turn, until one gives a solution that its `check_sol` accepts.

The loop guards three stages: `JuMP.set_optimizer`, `JuMP.optimize!` and `JuMP.assert_is_solved_and_feasible`. It does not guard [`set_solver_attributes`](@ref). So an attribute that the solver refuses raises out of this function, the loop records no trial, and it tries no later solver. A misspelled attribute is an error of the configuration, not a failure of the solver, and the caller must see it.

Give each solver of a vector its own `name`. The name is the key of `trials`, and every solver has the default name `\"\"`. Under one key, a later failure overwrites an earlier one. Two solvers that both fail at `JuMP.set_optimizer` leave one entry under the default names, and two entries under distinct names.

# Algorithm

 1. For each solver of `slv`, in order:
     1. Attach the solver to `model` with `JuMP.set_optimizer`. When that raises, record the error under `:set_optimizer` in `trials[solver.name]`, and go to the next solver.
     2. Set the attributes of the solver from its `settings` with [`set_solver_attributes`](@ref).
     3. Solve with `JuMP.optimize!`. When that raises, record the error under `:optimize!`, and go to the next solver.
     4. Check the solution with `JuMP.assert_is_solved_and_feasible` under the `check_sol` keywords of the solver. When the check passes, set `success` to `true` and stop the loop. Otherwise record the error under `:assert_is_solved_and_feasible`, the termination status of the solver under `:termination_status`, and the `settings` of the solver under `:settings`.
 2. Build the [`JuMPResult`](@ref) from `trials` and `success`. Its constructor warns when `success` is `false`.

# Arguments

  - `model`: The JuMP model to solve. After a success, it holds the solution of the solver that succeeded.
  - `slv`: One [`Solver`](@ref), or a vector of them in the order to try.

# Returns

  - `res::JuMPResult`: The failures of the solvers, and whether one solver succeeded.

# Related

  - [`JuMP.Model`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.Model)
  - [`set_optimizer`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.set_optimizer)
  - [`assert_is_solved_and_feasible`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.assert_is_solved_and_feasible)
  - [`set_solver_attributes`](@ref)
  - [`JuMPResult`](@ref)
  - [`Slv_VecSlv`](@ref)
"""
function optimise_JuMP_model!(model::JuMP.Model, slv::Slv_VecSlv)
    trials = Dict()
    success = false
    for solver in slv
        try
            JuMP.set_optimizer(model, solver.solver; add_bridges = solver.add_bridges)
        catch err
            trials[solver.name] = Dict(:set_optimizer => err)
            continue
        end
        set_solver_attributes(model, solver.settings)
        try
            JuMP.optimize!(model)
        catch err
            trials[solver.name] = Dict(:optimize! => err)
            continue
        end
        try
            JuMP.assert_is_solved_and_feasible(model; solver.check_sol...)
            success = true
            break
        catch err
            trials[solver.name] = Dict(:assert_is_solved_and_feasible => err,
                                       :termination_status =>
                                           JuMP.termination_status(model),
                                       :settings => solver.settings)
        end
    end
    return JuMPResult(; trials = trials, success = success)
end

export Solver, JuMPResult
