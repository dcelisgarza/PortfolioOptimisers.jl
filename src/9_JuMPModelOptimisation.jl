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
# function Base.isequal(A::Solver, B::Solver)
#     for property ∈ propertynames(A)
#         prop_A = getproperty(A, property)
#         prop_B = getproperty(B, property)
#         if !isequal(prop_A, prop_B)
#             return false
#         end
#     end
#     return true
# end
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
function optimise_JuMP_model!(model::JuMP.Model,
                              slv::Union{<:Solver, <:AbstractVector{<:Solver}})
    trials = Dict()
    success = false
    for solver ∈ slv
        name = solver.name
        solver_i = solver.solver
        settings = solver.settings
        add_bridges = solver.add_bridges
        check_sol = solver.check_sol
        set_optimizer(model, solver_i; add_bridges = add_bridges)
        if !isnothing(settings)
            for (k, v) ∈ settings
                set_attribute(model, k, v)
            end
        end
        try
            JuMP.optimize!(model)
        catch jump_error
            push!(trials, name => Dict(:jump_error => jump_error))
            continue
        end
        try
            assert_is_solved_and_feasible(model; check_sol...)
            success = true
            break
        catch err
            push!(trials,
                  name => Dict(:objective_val => objective_value(model), :err => err,
                               :settings => settings))
        end
    end
    return JuMPResult(; trials = trials, success = success)
end

export Solver
