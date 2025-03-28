function optimise_JuMP_model(model::JuMP.Model,
                             slv::Union{<:Solver, <:AbstractVector{<:Solver}})
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    solvers_tried = Dict()
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
            push!(solvers_tried, name => Dict(:jump_error => jump_error))
            continue
        end
        try
            assert_is_solved_and_feasible(model; check_sol...)
            success = true
            break
        catch err
            push!(solvers_tried,
                  name => Dict(:objective_val => objective_value(model), :err => err,
                               :settings => settings))
        end
    end
    return success, solvers_tried
end
