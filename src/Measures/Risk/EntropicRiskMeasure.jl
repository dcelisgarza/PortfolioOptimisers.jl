function ERM(x::AbstractVector{<:Real},
             solvers::Union{<:Solver, <:AbstractVector{<:Solver}}, alpha::Real = 0.05)
    if isa(solvers, AbstractVector)
        @smart_assert(!isempty(solvers))
    end
    model = JuMP.Model()
    set_string_names_on_creation(model, false)
    T = length(x)
    @variables(model, begin
                   t
                   z >= 0
                   u[1:T]
               end)
    @constraints(model, begin
                     sum(u) <= z
                     [i = 1:T], [-x[i] - t, z, u[i]] ∈ MOI.ExponentialCone()
                 end)
    @expression(model, risk, t - z * log(alpha * T))
    @objective(model, Min, risk)
    success, solvers_tried = optimise_JuMP_model(model, solvers)
    return if success
        objective_value(model)
    else
        @warn("model could not be optimised satisfactorily.\nSolvers: $solvers_tried.")
        NaN
    end
end
