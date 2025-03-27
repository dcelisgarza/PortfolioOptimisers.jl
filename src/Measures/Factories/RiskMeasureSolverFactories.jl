function risk_measure_solver_factory(risk_solvers::Union{<:Solver,
                                                         <:AbstractVector{<:Solver}}, ::Any)
    return risk_solvers
end
function risk_measure_solver_factory(::Nothing,
                                     solvers::Union{<:Solver, <:AbstractVector{<:Solver}})
    return solvers
end
function risk_measure_solver_factory(::Nothing, ::Nothing)
    throw(ArgumentError("Both risk_solver and prior_solver are nothing, cannot solve JuMP model."))
end
