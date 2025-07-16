abstract type AbstractJuMPResult <: AbstractResult end
struct Solver{T1 <: Union{Symbol, <:AbstractString}, T2,
              T3 <: Union{Nothing, <:AbstractDict}, T4 <: NamedTuple, T5 <: Bool}
    name::T1
    solver::T2
    settings::T3
    check_sol::T4
    add_bridges::T5
end
function Solver(; name::Union{Symbol, <:AbstractString} = "", solver::Any = nothing,
                settings::Union{Nothing, <:AbstractDict} = nothing,
                check_sol::NamedTuple = (;), add_bridges::Bool = true)
    if !isnothing(settings)
        @smart_assert(!isempty(settings))
    end
    return Solver{typeof(name), typeof(solver), typeof(settings), typeof(check_sol),
                  typeof(add_bridges)}(name, solver, settings, check_sol, add_bridges)
end
#=
function Base.show(io::IO, slv::Solver)
    println(io, "Solver")
    for field in fieldnames(typeof(slv))
        val = getfield(slv, field)
        print(io, "  ", lpad(string(field), 11), " ")
        if isnothing(val)
            println(io, "| nothing")
        elseif isa(val, AbstractDict)
            if isempty(val)
                println(io, "| $(typeof(val)): Dict{Any, Any}()")
            else
                println(io, "| $(typeof(val)): ", repr(val))
            end
        elseif isa(val, NamedTuple)
            println(io, "| $(typeof(val)): ", repr(val))
        else
            println(io, "| $(typeof(val)): ", repr(val))
        end
    end
end
=#
Base.iterate(S::Solver, state = 1) = state > 1 ? nothing : (S, state + 1)
struct JuMPResult{T1 <: AbstractDict, T2 <: Bool} <: AbstractJuMPResult
    trials::T1
    success::T2
end
function JuMPResult(; trials::AbstractDict, success::Bool)
    if !success
        @warn("Model could not be solved satisfactorily.\n$trials")
    end
    return JuMPResult{typeof(trials), typeof(success)}(trials, success)
end
#=
function Base.show(io::IO, res::JuMPResult)
    println(io, "JuMPResult")
    for field in fieldnames(typeof(res))
        val = getfield(res, field)
        print(io, "  ", lpad(string(field), 7), " ")
        if isnothing(val)
            println(io, "| nothing")
        elseif isa(val, AbstractDict)
            if isempty(val)
                println(io, "| $(typeof(val)): Dict()")
            else
                println(io, "| $(typeof(val)): ", repr(val))
            end
        else
            println(io, "| $(typeof(val)): ", repr(val))
        end
    end
end
=#
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
        if !isnothing(solver.settings)
            for (k, v) in solver.settings
                set_attribute(model, k, v)
            end
        end
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
