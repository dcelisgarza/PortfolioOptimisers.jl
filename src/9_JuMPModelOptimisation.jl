abstract type AbstractJuMPResult <: AbstractResult end
struct Solver{T1, T2, T3, T4, T5} <: AbstractEstimator
    name::T1
    solver::T2
    settings::T3
    check_sol::T4
    add_bridges::T5
end
function Solver(; name::Union{Symbol, <:AbstractString} = "", solver::Any = nothing,
                settings::Union{Nothing, <:AbstractDict, <:Pair, <:AbstractVector{<:Pair}} = nothing,
                check_sol::NamedTuple = (;), add_bridges::Bool = true)
    if isa(settings, Union{<:AbstractDict, <:AbstractVector})
        @assert(!isempty(settings), AssertionError("`settings` must be non-empty."))
    end
    return Solver(name, solver, settings, check_sol, add_bridges)
end
Base.iterate(S::Solver, state = 1) = state > 1 ? nothing : (S, state + 1)
struct JuMPResult{T1, T2} <: AbstractJuMPResult
    trials::T1
    success::T2
end
function JuMPResult(; trials::AbstractDict, success::Bool)
    if !success
        @warn("Model could not be solved satisfactorily.\n$trials")
    end
    return JuMPResult(trials, success)
end
function set_solver_attributes(args...)
    return nothing
end
function set_solver_attributes(model::JuMP.Model,
                               settings::Union{<:AbstractDict, <:AbstractVector{<:Pair}})
    for (k, v) in settings
        set_attribute(model, k, v)
    end
    return nothing
end
function set_solver_attributes(model::JuMP.Model, settings::Pair)
    set_attribute(model, settings...)
    return nothing
end
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
