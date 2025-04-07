struct Solver{T1 <: Union{Symbol, <:AbstractString}, T2, T3 <: NamedTuple,
              T4 <: Union{Nothing, <:AbstractDict}, T5 <: Bool}
    name::T1
    solver::T2
    check_sol::T3
    settings::T4
    add_bridges::T5
end
function Solver(; name::Union{Symbol, <:AbstractString} = "", solver::Any = nothing,
                check_sol::NamedTuple = (;),
                settings::Union{Nothing, <:AbstractDict} = nothing,
                add_bridges::Bool = true)
    return Solver{typeof(name), typeof(solver), typeof(check_sol), typeof(settings),
                  typeof(add_bridges)}(name, solver, check_sol, settings, add_bridges)
end
function Base.isequal(A::Solver, B::Solver)
    for property ∈ propertynames(A)
        prop_A = getproperty(A, property)
        prop_B = getproperty(B, property)
        if !isequal(prop_A, prop_B)
            return false
        end
    end
    return true
end
Base.iterate(S::Solver, state = 1) = state > 1 ? nothing : (S, state + 1)
abstract type AbstractJuMPResult <: AbstractResult end
struct JuMPResult{T1 <: AbstractDict, T2 <: Bool} <: AbstractJuMPResult
    trials::T1
    success::T2
end
function optimise_JuMP_model!(model::JuMP.Model,
                              slv::Union{<:Solver, <:AbstractVector{<:Solver}})
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    trials = Dict()
    success = false
    for solver ∈ slv
        name = solver.name
        solver_i = solver.solver
        settings = solver.settings
        add_bridges = solver.add_bridges
        check_sol = solver.check_sol
        set_optimizer(model, solver_i; add_bridges = add_bridges)
        if !isnothing(settings) && !isempty(settings)
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
    return JuMPResult(trials, success)
end

export Solver
