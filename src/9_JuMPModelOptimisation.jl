"""
    abstract type AbstractJuMPResult <: AbstractResult end

Abstract supertype for all JuMP-based optimisation result types in PortfolioOptimisers.jl.

All concrete types representing the result of a JuMP model optimisation should subtype `AbstractJuMPResult`. This enables a consistent interface for handling solver results throughout the package.

# Related

  - [`JuMPResult`](@ref)
"""
abstract type AbstractJuMPResult <: AbstractResult end

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

    Solver(; name::Union{Symbol, <:AbstractString} = "", solver = nothing,
             settings::Union{Nothing, <:AbstractDict, <:Pair, <:AbstractVector{<:Pair}} = nothing,
             check_sol::NamedTuple = (;), add_bridges::Bool = true)

# Related

  - [`optimise_JuMP_model!`](@ref)
"""
struct Solver{T1, T2, T3, T4, T5} <: AbstractEstimator
    name::T1
    solver::T2
    settings::T3
    check_sol::T4
    add_bridges::T5
end
"""
    Solver(; name::Union{Symbol, <:AbstractString} = "", solver::Any = nothing,
            settings::Union{Nothing, <:AbstractDict, <:Pair, <:AbstractVector{<:Pair}} = nothing,
            check_sol::NamedTuple = (;), add_bridges::Bool = true)

Construct a `Solver` object for configuring a JuMP solver.

This constructor validates and packages the solver backend, settings, and options for use in model optimisation.

# Arguments

  - `name`: Symbol or string identifier for the solver.
  - `solver`: The `optimizer_factory` in [`set_optimizer`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.set_optimizer).
  - `settings`: Solver-specific settings used in [`set_attribute`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.set_attribute).
  - `check_sol`: Named tuple of solution for keyword arguments in [`assert_is_solved_and_feasible`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.assert_is_solved_and_feasible).
  - `add_bridges`: The `add_bridges` keyword argument in [`set_optimizer`](https://jump.dev/JuMP.jl/stable/api/JuMP/#JuMP.set_optimizer).

# Returns

  - `Solver`: Configured solver object.

# Validation

  - Asserts that `settings` is non-empty if provided as a dictionary or vector.

# Examples

```jldoctest
julia> Solver()
Solver
         name | String: ""
       solver | nothing
     settings | nothing
    check_sol | @NamedTuple{}: NamedTuple()
  add_bridges | Bool: true
```
"""
function Solver(; name::Union{Symbol, <:AbstractString} = "", solver::Any = nothing,
                settings::Union{Nothing, <:AbstractDict, <:Pair, <:AbstractVector{<:Pair}} = nothing,
                check_sol::NamedTuple = (;), add_bridges::Bool = true)
    if isa(settings, Union{<:AbstractDict, <:AbstractVector})
        @argcheck(!isempty(settings), IsEmptyError(non_empty_msg("`settings`") * "."))
    end
    return Solver(name, solver, settings, check_sol, add_bridges)
end

"""
    struct JuMPResult{T1, T2} <: AbstractJuMPResult
        trials::T1
        success::T2
    end

Result type for JuMP model optimisation.

The `JuMPResult` struct records the outcome of a JuMP optimisation, including trial errors and success status.

# Fields

  - `trials`: Dictionary of solver attempts and errors.
  - `success`: Boolean indicating whether optimisation succeeded.

# Constructor

    JuMPResult(; trials::AbstractDict, success::Bool)

# Related

  - [`optimise_JuMP_model!`](@ref)
"""
struct JuMPResult{T1, T2} <: AbstractJuMPResult
    trials::T1
    success::T2
end
"""
    JuMPResult(; trials::AbstractDict, success::Bool)

Construct a `JuMPResult` object from solver trials and success status.

If `success` is `false`, a warning is emitted with the trial errors.

# Arguments

  - `trials`: Dictionary of solver attempts and errors.
  - `success`: Boolean indicating whether optimisation succeeded.

# Returns

  - `JuMPResult`: Result object.

# Examples

```jldoctest
julia> JuMPResult(; trials = Dict(:HiGHS => Dict(:optimize! => "error")), success = true)
JuMPResult
   trials | Dict{Symbol, Dict{Symbol, String}}: Dict(:HiGHS => Dict(:optimize! => "error"))
  success | Bool: true
```
"""
function JuMPResult(; trials::AbstractDict, success::Bool)
    if !success
        @warn("Model could not be solved satisfactorily.\n$trials")
    end
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
    set_solver_attributes(model::JuMP.Model,
                          settings::Union{<:AbstractDict, <:AbstractVector{<:Pair}})

Set multiple solver attributes on a JuMP model.

Iterates over the provided settings and applies each as a solver attribute.

# Arguments

  - `model`: JuMP model.
  - `settings`: Dictionary or vector of pairs of solver settings.

# Returns

  - `nothing`
"""
function set_solver_attributes(model::JuMP.Model,
                               settings::Union{<:AbstractDict, <:AbstractVector{<:Pair}})
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
"""
function set_solver_attributes(model::JuMP.Model, settings::Pair)
    set_attribute(model, settings...)
    return nothing
end

"""
    optimise_JuMP_model!(model::JuMP.Model,
                         slv::Union{<:Solver, <:AbstractVector{<:Solver}})

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
"""
function optimise_JuMP_model!(model::JuMP.Model,
                              slv::Union{<:Solver, <:AbstractVector{<:Solver}})
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
