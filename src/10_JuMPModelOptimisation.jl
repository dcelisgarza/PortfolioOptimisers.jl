"""
    abstract type AbstractJuMPResult <: AbstractResult end

Abstract supertype for all JuMP-based optimisation result types in PortfolioOptimisers.jl.

All concrete types representing the result of a JuMP model optimisation should subtype `AbstractJuMPResult`. This enables a consistent interface for handling solver results throughout the package.

# Related

  - [`JuMPResult`](@ref)
"""
abstract type AbstractJuMPResult <: AbstractResult end
"""
    const DictStrA_VecPairStrA = Union{<:AbstractDict{<:AbstractString, <:Any},
                                       <:AbstractVector{<:Pair{<:AbstractString, <:Any}}}

Alias for a dictionary or vector of pairs with string keys.

Represents solver settings as either a dictionary mapping strings to values, or a vector of pairs where the first element is a string and the second is any value. Used for passing attribute settings to JuMP solvers.

# Related

  - [`SlvSettings`](@ref)
  - [`Solver`](@ref)
  - [`set_solver_attributes`](@ref)
"""
const DictStrA_VecPairStrA = Union{<:AbstractDict{<:AbstractString, <:Any},
                                   <:AbstractVector{<:Pair{<:AbstractString, <:Any}}}
"""
    const SlvSettings = Union{<:Pair{<:AbstractString, <:Any}, <:DictStrA_VecPairStrA}

Alias for solver settings used in JuMP-based optimisation.

Represents solver settings as either a single pair of string key and value, or as a dictionary/vector of pairs with string keys. Used for passing attribute settings to JuMP solvers.

# Related

  - [`DictStrA_VecPairStrA`](@ref)
  - [`Solver`](@ref)
  - [`set_solver_attributes`](@ref)
"""
const SlvSettings = Union{<:Pair{<:AbstractString, <:Any}, <:DictStrA_VecPairStrA}
"""
    struct Solver{T1, T2, T3, T4, T5} <: AbstractEstimator
        name::T1
        solver::T2
        settings::T3
        check_sol::T4
        add_bridges::T5
    end

Container for configuring a JuMP solver and its settings.

The `Solver` struct encapsulates all information needed to set up and run a JuMP optimisation, including the solver backend, solver-specific settings, solution checks, and bridge options.

# Fields

  - `name`: Symbol or string identifier for the solver.
  - `solver`: The `optimizer_factory` in [`set_optimizer`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.set_optimizer).
  - `settings`: Solver-specific settings used in [`set_attribute`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.set_attribute).
  - `check_sol`: Named tuple of solution for keyword arguments in [`assert_is_solved_and_feasible`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.assert_is_solved_and_feasible).
  - `add_bridges`: The `add_bridges` keyword argument in [`set_optimizer`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.set_optimizer).

# Constructor

    Solver(; name::Sym_Str = "", solver::Any = nothing,
           settings::Option{<:SlvSettings} = nothing, check_sol::NamedTuple = (;),
           add_bridges::Bool = true)

Keyword arguments correspond to the fields above.

## Validation

  - `settings`:

      + `Dict_Vec`: `!isempty(settings)`.

# Examples

```jldoctest
julia> Solver()
Solver
         name ┼ String: ""
       solver ┼ nothing
     settings ┼ nothing
    check_sol ┼ @NamedTuple{}: NamedTuple()
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
struct Solver{T1, T2, T3, T4, T5} <: AbstractEstimator
    name::T1
    solver::T2
    settings::T3
    check_sol::T4
    add_bridges::T5
    function Solver(name::Sym_Str, solver::Any, settings::Option{<:SlvSettings},
                    check_sol::NamedTuple, add_bridges::Bool)
        if isa(settings, Dict_Vec)
            @argcheck(!isempty(settings), IsEmptyError)
        end
        return new{typeof(name), typeof(solver), typeof(settings), typeof(check_sol),
                   typeof(add_bridges)}(name, solver, settings, check_sol, add_bridges)
    end
end
function Solver(; name::Sym_Str = "", solver::Any = nothing,
                settings::Option{<:SlvSettings} = nothing, check_sol::NamedTuple = (;),
                add_bridges::Bool = true)
    return Solver(name, solver, settings, check_sol, add_bridges)
end
"""
    const VecSlv = AbstractVector{<:Solver}

Alias for a vector of `Solver` objects.

Represents a collection of solver configurations to be used in JuMP-based optimisation routines. Enables sequential or fallback solver strategies by passing multiple solver setups.

# Related Types

  - [`Solver`](@ref)
"""
const VecSlv = AbstractVector{<:Solver}
"""
    const Slv_VecSlv = Union{<:Solver, <:VecSlv}

Alias for a single `Solver` or a vector of `Solver` objects.

Represents either a single solver configuration or a collection of solver configurations for JuMP-based optimisation routines. Enables flexible dispatch for optimisation functions that accept one or multiple solvers.

# Related Types

  - [`Solver`](@ref)
  - [`VecSlv`](@ref)
"""
const Slv_VecSlv = Union{<:Solver, <:VecSlv}
"""
    struct JuMPResult{T1, T2} <: AbstractJuMPResult
        trials::T1
        success::T2
    end

Result type for JuMP model optimisation.

The `JuMPResult` struct records the outcome of a JuMP optimisation, including trial errors and success status.

# Fields

  - `trials`: Dictionary of solver trials and errors.
  - `success`: Boolean indicating whether optimisation succeeded.

# Constructor

    JuMPResult(; trials::AbstractDict, success::Bool)

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> JuMPResult(; trials = Dict(:HiGHS => Dict(:optimize! => "error")), success = true)
JuMPResult
   trials ┼ Dict{Symbol, Dict{Symbol, String}}: Dict(:HiGHS => Dict(:optimize! => "error"))
  success ┴ Bool: true
```

# Related

  - [`optimise_JuMP_model!`](@ref)
"""
struct JuMPResult{T1, T2} <: AbstractJuMPResult
    trials::T1
    success::T2
    function JuMPResult(trials::AbstractDict, success::Bool)
        if !success
            @warn("Model could not be solved satisfactorily.\n$trials")
        end
        return new{typeof(trials), typeof(success)}(trials, success)
    end
end
function JuMPResult(; trials::AbstractDict, success::Bool)
    return JuMPResult(trials, success)
end
"""
    set_solver_attributes(args...)

Set solver attributes for a JuMP model.

This is a generic fallback that does nothing if no model or settings are provided.

# Arguments

  - `args...`: Arguments (ignored).

# Returns

  - `nothing`
"""
function set_solver_attributes(args...)
    return nothing
end
"""
    set_solver_attributes(model::JuMP.Model, settings::DictStrA_VecPairStrA)

Set multiple solver attributes on a JuMP model.

Iterates over the provided settings and applies each as a solver attribute.

# Arguments

  - `model`: JuMP model.
  - `settings`: Dictionary or vector of pairs of solver settings.

# Returns

  - `nothing`

# Related

  - [`JuMP.Model`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.Model)
  - [`set_attribute`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.set_attribute)
  - [`DictStrA_VecPairStrA`](@ref)
"""
function set_solver_attributes(model::JuMP.Model, settings::DictStrA_VecPairStrA)
    for (k, v) in settings
        set_attribute(model, k, v)
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

  - `nothing`

# Related

  - [`JuMP.Model`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.Model)
  - [`set_attribute`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.set_attribute)
"""
function set_solver_attributes(model::JuMP.Model, settings::Pair)
    set_attribute(model, settings...)
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

  - `JuMPResult`: Result object containing trial errors and success flag.

# Details

  - For each solver, sets the optimizer and attributes, runs `JuMP.optimize!`, and checks solution feasibility.
  - If a solver fails, records the error and tries the next.
  - Stops at the first successful solution.

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
            set_optimizer(model, solver.solver; add_bridges = solver.add_bridges)
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
            assert_is_solved_and_feasible(model; solver.check_sol...)
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
