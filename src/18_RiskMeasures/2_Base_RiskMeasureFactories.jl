function nothing_scalar_array_factory(::Nothing, ::Nothing)
    return nothing
end
function nothing_scalar_array_factory(risk_variable::Union{<:Real, <:AbstractArray}, ::Any)
    return risk_variable
end
function nothing_scalar_array_factory(::Nothing, prior_variable::AbstractArray)
    return prior_variable
end
function risk_measure_nothing_scalar_array_view(::Nothing, ::Nothing, i::AbstractVector)
    throw(ArgumentError("Both risk_variable and prior_variable are nothing."))
end
function risk_measure_nothing_scalar_array_view(risk_variable::Union{<:Real,
                                                                     <:AbstractArray},
                                                ::Any, i::AbstractVector)
    return nothing_scalar_array_view(risk_variable, i)
end
function risk_measure_nothing_scalar_array_view(::Nothing, prior_variable::AbstractArray,
                                                i::AbstractVector)
    return nothing_scalar_array_view(prior_variable, i)
end
function risk_measure_nothing_scalar_array_view(risk_variable::Union{Nothing, <:Real},
                                                ::AbstractVector)
    return risk_variable
end
function solver_factory(risk_solvers::Union{<:Solver, <:AbstractVector{<:Solver}}, ::Any)
    return risk_solvers
end
function solver_factory(::Nothing, slv::Union{<:Solver, <:AbstractVector{<:Solver}})
    return slv
end
function solver_factory(::Nothing, ::Nothing)
    throw(ArgumentError("Both risk_solver and prior_solver are nothing, cannot solve JuMP model."))
end
