"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all JuMP-based optimisation result types.

All concrete and/or abstract types representing the result of a JuMP model optimisation should be subtypes of `AbstractJuMPResult`.

A subtype records what happened during a solve attempt. It is not the portfolio: the weights live on the optimiser's own result type, which carries one of these alongside them.

# Related

  - [`JuMPResult`](@ref)
  - [`optimise_JuMP_model!`](@ref)
"""
abstract type AbstractJuMPResult <: AbstractResult end
"""
    const Dict_VecPair = Union{<:AbstractDict, <:AbstractVector{<:Pair}}

Alias for a dictionary or vector of pairs.

Represents solver settings as either a dictionary mapping constraint attributes to values, or a vector of pairs where the first element is a constraint attribute and the second is its value. Used for passing attribute settings to JuMP solvers.

# Related

  - [`SlvSettings`](@ref)
  - [`Solver`](@ref)
  - [`set_solver_attributes`](@ref)
"""
const Dict_VecPair = Union{<:AbstractDict, <:AbstractVector{<:Pair}}
"""
    const SlvSettings = Union{<:Pair, <:Dict_VecPair}

Alias for solver settings used in JuMP-based optimisation.

Represents solver settings as either a single solver attribute, or a collection of solver attributes.

# Related

  - [`Dict_VecPair`](@ref)
  - [`Solver`](@ref)
  - [`set_solver_attributes`](@ref)
"""
const SlvSettings = Union{<:Pair, <:Dict_VecPair}
"""
    const SlvKeys = Union{<:AbstractString, <:JuMP.MOI.AbstractModelAttribute}

Alias for JuMP solver attribute keys.

Matches either a string key or a `JuMP.MOI.AbstractModelAttribute` instance. Used internally for validating and applying solver settings to JuMP models.

# Related

  - [`SlvSettings`](@ref)
  - [`Solver`](@ref)
  - [`set_solver_attributes`](@ref)
"""
const SlvKeys = Union{<:AbstractString, <:JuMP.MOI.AbstractModelAttribute}
"""
$(DocStringExtensions.TYPEDEF)

Configures one solver backend, its attributes, and the statuses its solutions must reach.

Every optimiser takes one `Solver` or a vector of them. [`optimise_JuMP_model!`](@ref) tries them in order and stops at the first that returns a solution `check_sol` accepts, so a vector is a fallback chain.

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

The default `check_sol` is strict — it accepts only a solution the solver reports as `OPTIMAL` or `LOCALLY_SOLVED` at a `FEASIBLE_POINT`.

```jldoctest
julia> Solver(; solver = nothing)
Solver
         name ┼ String: ""
       solver ┼ nothing
     settings ┼ nothing
    check_sol ┼ @NamedTuple{}: NamedTuple()
  add_bridges ┴ Bool: true
```

To also accept an approximate solution, which is the common case and what the examples, user guide and tests use:

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
                          ArgumentError("all keys in settings must be a SlvKeys (AbstractString or MOI.AbstractModelAttribute)"))
            else
                @argcheck(all(x -> isa(x, SlvKeys), keys(settings)),
                          ArgumentError("all keys in settings must be a SlvKeys (AbstractString or MOI.AbstractModelAttribute)"))
            end
        elseif isa(settings, Pair)
            @argcheck(isa(settings[1], SlvKeys),
                      ArgumentError("settings[1] must be a SlvKeys (AbstractString or MOI.AbstractModelAttribute), got $(settings[1])"))
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

Alias for a vector of `Solver` objects.

Represents a collection of solver configurations to be used in JuMP-based optimisation routines. Enables sequential or fallback solver strategies by passing multiple solver setups.

# Related

  - [`Solver`](@ref)
"""
const VecSlv = AbstractVector{<:Solver}
"""
    const Slv_VecSlv = Union{<:Solver, <:VecSlv}

Alias for a single `Solver` or a vector of `Solver` objects.

Represents either a single solver configuration or a collection of solver configurations for JuMP-based optimisation routines. Enables flexible dispatch for optimisation functions that accept one or multiple solvers.

# Related

  - [`Solver`](@ref)
  - [`VecSlv`](@ref)
"""
const Slv_VecSlv = Union{<:Solver, <:VecSlv}
"""
$(DocStringExtensions.TYPEDEF)

Records which solvers failed, at which stage, and whether any of them succeeded.

When `success` is `false` the constructor emits a warning built by [`failed_solve_msg`](@ref): one bounded line per failed solver stage (name, stage, first line of the error). The full per-solver exceptions and settings stay available on `trials` and are never dumped into the log.

`trials` records **failures only**. A solver that succeeds leaves no entry, so an empty `trials` with `success = true` means the first solver answered, and it is not a record of the solve. Each entry is keyed by the solver's `name` and holds a dictionary from the failed stage — `:set_optimizer`, `:optimize!` or `:assert_is_solved_and_feasible` — to the exception. Two solvers that share a name share one entry, and the default name is `\"\"` for all of them, so a vector of unnamed solvers keeps only its last failure.

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
    Dictionary of solver trials and errors.
    """
    trials
    """
    Boolean indicating whether optimisation succeeded.
    """
    success
    function JuMPResult(trials::AbstractDict, success::Bool)::JuMPResult
        if !success
            # Summarised via the shared builder: one bounded line per failed stage, never
            # the whole trials dict, the solver settings, or full exception payloads.
            @warn(failed_solve_msg(trials))
        end
        return new{typeof(trials), typeof(success)}(trials, success)
    end
end
function JuMPResult(; trials::AbstractDict, success::Bool)::JuMPResult
    return JuMPResult(trials, success)
end
"""
    set_solver_attributes(args...)

Set solver attributes for a JuMP model.

This is a generic fallback that does nothing if no model or settings are provided.

# Arguments

  - `args...`: Arguments (ignored).

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

Set multiple solver attributes on a JuMP model.

Iterates over the provided settings and applies each as a solver attribute.

# Arguments

  - `model`: JuMP model.
  - `settings`: Dictionary or vector of pairs of solver settings.

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

Set a single solver attribute on a JuMP model.

# Arguments

  - `model`: JuMP model.
  - `settings`: Pair of attribute name and value.

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
    optimise_JuMP_model!(model::JuMP.Model, slv::Slv_VecSlv)

Attempt to optimise a JuMP model using one or more configured solvers.

Tries each solver in order, applying settings and checking for solution feasibility. Returns a `JuMPResult` with trial errors and success status.

# Arguments

  - `model`: JuMP model to optimise.
  - `slv`: Single `Solver` or vector of `Solver` objects.

# Returns

  - `res::JuMPResult`: Result object containing trial errors and success flag.

# Details

  - For each solver, sets the optimizer and attributes, runs `JuMP.optimize!`, and checks solution feasibility.
  - If a solver fails at one of the three guarded stages, records the error under the solver's `name` and tries the next.
  - Stops at the first successful solution, and leaves no `trials` entry for it.

Three stages are guarded: `JuMP.set_optimizer`, `JuMP.optimize!` and `JuMP.assert_is_solved_and_feasible`. [`set_solver_attributes`](@ref) is **not**. A solver attribute the backend refuses throws straight out of this function, so no trial is recorded and no later solver is tried. This is deliberate: a misspelled attribute is a configuration error, not a solver failure, and swallowing it would silently drop a setting the caller asked for.

Give each solver of a vector its own `name`. The name is the `trials` key, the default is `\"\"` for every solver, and a later failure overwrites an earlier one under the same key. Measured on two solvers that both fail at `set_optimizer`: `trials` holds **one** entry with the default names and **two** with distinct names.

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
                                       :settings => solver.settings)
        end
    end
    return JuMPResult(; trials = trials, success = success)
end

export Solver, JuMPResult
