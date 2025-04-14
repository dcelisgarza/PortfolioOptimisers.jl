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
function ucs_factory(::Nothing, ::Nothing)
    return nothing
end
function ucs_factory(::Nothing, ::Nothing, ::Any)
    return nothing
end
function ucs_factory(risk_ucs::Union{<:AbstractUncertaintySetResult,
                                     <:AbstractUncertaintySetEstimator}, ::Any)
    return risk_ucs
end
function ucs_factory(::Nothing,
                     prior_ucs::Union{<:AbstractUncertaintySetResult,
                                      <:AbstractUncertaintySetEstimator})
    return prior_ucs
end
function ucs_factory(risk_ucs::AbstractUncertaintySetEstimator, ::Any, ::Any)
    return risk_ucs
end
function ucs_factory(::Nothing, prior_ucs::AbstractUncertaintySetEstimator, ::Any)
    return prior_ucs
end
function ucs_factory(risk_ucs::BoxUncertaintySetResult{<:AbstractVector, <:AbstractVector},
                     ::Any, i)
    return BoxUncertaintySetResult(; lb = view(risk_ucs.lb, i), ub = view(risk_ucs.ub, i))
end
function ucs_factory(risk_ucs::BoxUncertaintySetResult{<:AbstractMatrix, <:AbstractMatrix},
                     ::Any, i)
    return BoxUncertaintySetResult(; lb = view(risk_ucs.lb, i, i),
                                   ub = view(risk_ucs.ub, i, i))
end
function ucs_factory(::Nothing,
                     prior_ucs::BoxUncertaintySetResult{<:AbstractVector, <:AbstractVector},
                     i)
    return BoxUncertaintySetResult(; lb = view(prior_ucs.lb, i), ub = view(prior_ucs.ub, i))
end
function ucs_factory(::Nothing,
                     prior_ucs::BoxUncertaintySetResult{<:AbstractMatrix, <:AbstractMatrix},
                     i)
    return BoxUncertaintySetResult(; lb = view(prior_ucs.lb, i, i),
                                   ub = view(prior_ucs.ub, i, i))
end
function ucs_factory(risk_ucs::EllipseUncertaintySetResult{<:AbstractMatrix, <:Any,
                                                           <:SigmaEllipseUncertaintySetResult},
                     ::Any, i)
    i = fourth_moment_index_factory(floor(Int, sqrt(size(risk_ucs.sigma, 1))), i)
    return EllipseUncertaintySetResult(; sigma = view(risk_ucs.sigma, i, i), k = risk_ucs.k)
end
function ucs_factory(::Nothing,
                     prior_ucs::EllipseUncertaintySetResult{<:AbstractMatrix, <:Any,
                                                            <:SigmaEllipseUncertaintySetResult},
                     i)
    i = fourth_moment_index_factory(floor(Int, sqrt(size(prior_ucs.sigma, 1))), i)
    return EllipseUncertaintySetResult(; sigma = view(prior_ucs.sigma, i, i),
                                       k = prior_ucs.k)
end
function ucs_factory(risk_ucs::EllipseUncertaintySetResult{<:AbstractMatrix, <:Any,
                                                           <:MuEllipseUncertaintySetResult},
                     ::Any, i)
    return EllipseUncertaintySetResult(; sigma = view(risk_ucs.sigma, i, i), k = risk_ucs.k)
end
function ucs_factory(::Nothing,
                     prior_ucs::EllipseUncertaintySetResult{<:AbstractMatrix, <:Any,
                                                            <:MuEllipseUncertaintySetResult},
                     i)
    return EllipseUncertaintySetResult(; sigma = view(prior_ucs.sigma, i, i),
                                       k = prior_ucs.k)
end
