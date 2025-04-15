function risk_measure_nothing_vec_factory(::Nothing, ::Nothing)
    throw(ArgumentError("Both risk_variable and prior_variable are nothing."))
end
function risk_measure_nothing_vec_factory(risk_variable::AbstractVector{<:Real}, ::Any)
    return risk_variable
end
function risk_measure_nothing_vec_factory(::Nothing, prior_variable::AbstractVector{<:Real})
    return prior_variable
end
function risk_measure_nothing_vec_factory(::Nothing, ::Nothing, i)
    throw(ArgumentError("Both risk_variable and prior_variable are nothing."))
end
function risk_measure_nothing_vec_factory(risk_variable::AbstractVector{<:Real}, ::Any, i)
    return view(risk_variable, i)
end
function risk_measure_nothing_vec_factory(::Nothing, prior_variable::AbstractVector{<:Real},
                                          i)
    return view(prior_variable, i)
end
function risk_measure_nothing_matrix_factory(::Nothing, ::Nothing)
    throw(ArgumentError("Both risk_variable and prior_variable are nothing."))
end
function risk_measure_nothing_matrix_factory(risk_variable::AbstractMatrix{<:Real}, ::Any)
    return risk_variable
end
function risk_measure_nothing_matrix_factory(::Nothing,
                                             prior_variable::AbstractMatrix{<:Real})
    return prior_variable
end
function risk_measure_nothing_matrix_factory(::Nothing, ::Nothing, i)
    throw(ArgumentError("Both risk_variable and prior_variable are nothing."))
end
function risk_measure_nothing_matrix_factory(risk_variable::AbstractMatrix{<:Real}, ::Any,
                                             i)
    return view(risk_variable, i, i)
end
function risk_measure_nothing_matrix_factory(::Nothing, prior_variable::AbstractMatrix, i)
    return view(prior_variable, i, i)
end
function risk_measure_nothing_real_vec_factory(risk_variable::AbstractVector{<:Real}, i)
    return view(risk_variable, i)
end
function risk_measure_nothing_real_vec_factory(risk_variable::Union{Nothing, Real}, ::Any)
    return risk_variable
end
function risk_measure_solver_factory(risk_solvers::Union{<:Solver,
                                                         <:AbstractVector{<:Solver}}, ::Any)
    return risk_solvers
end
function risk_measure_solver_factory(::Nothing,
                                     slv::Union{<:Solver, <:AbstractVector{<:Solver}})
    return slv
end
function risk_measure_solver_factory(::Nothing, ::Nothing)
    throw(ArgumentError("Both risk_solver and prior_solver are nothing, cannot solve JuMP model."))
end
