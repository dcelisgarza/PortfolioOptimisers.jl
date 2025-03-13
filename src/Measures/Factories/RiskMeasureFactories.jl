function risk_measure_nothing_vec_factory(::Nothing, ::Nothing)
    throw(ArgumentError("Both risk_variable and prior_variable are nothing."))
end
function risk_measure_nothing_vec_factory(risk_variable::AbstractVector{<:Real}, ::Any)
    return risk_variable
end
function risk_measure_nothing_vec_factory(::Nothing, prior_variable::AbstractVector{<:Real})
    return prior_variable
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
function risk_measure_nothing_real_vec_factory(risk_variable::AbstractVector{<:Real},
                                               cluster::AbstractVector)
    return view(risk_variable, cluster)
end
function risk_measure_nothing_real_vec_factory(risk_variable::Union{Nothing, Real},
                                               ::AbstractVector)
    return risk_variable
end
function uncertainty_set_factory(::NoUncertaintySet, ::NoUncertaintySet)
    return NoUncertaintySet()
end
function uncertainty_set_factory(::NoUncertaintySet,
                                 prior_uncertainty_set::Union{<:BoxUncertaintySet,
                                                              <:EllipseUncertaintySet})
    return prior_uncertainty_set
end
function uncertainty_set_factory(risk_uncertainty_set::Union{<:BoxUncertaintySet,
                                                             <:EllipseUncertaintySet},
                                 ::Any)
    return risk_uncertainty_set
end
