function risk_measure_nothing_vec_factory(risk_variable::AbstractVector{<:Real},
                                          prior_variable::AbstractVector{<:Real})
    return if !isempty(risk_variable)
        risk_variable
    elseif !isempty(prior_variable)
        prior_variable
    else
        throw(ArgumentError("Both risk_variable and prior_variable are empty."))
    end
end
function risk_measure_nothing_vec_factory(::Nothing, ::Nothing)
    throw(ArgumentError("Both risk_variable and prior_variable are nothing."))
end
function risk_measure_nothing_vec_factory(::Nothing, prior_variable::AbstractVector{<:Real})
    return if !isempty(prior_variable)
        prior_variable
    else
        throw(ArgumentError("prior_variable is empty."))
    end
end
function risk_measure_nothing_vec_factory(risk_variable::AbstractVector{<:Real}, ::Nothing)
    return if !isempty(risk_variable)
        risk_variable
    else
        throw(ArgumentError("risk_variable is empty."))
    end
end
function risk_measure_nothing_matrix_factory(risk_variable::AbstractMatrix{<:Real},
                                             prior_variable::AbstractMatrix{<:Real})
    return if !isempty(risk_variable)
        risk_variable
    elseif !isempty(prior_variable)
        prior_variable
    else
        throw(ArgumentError("Both risk_variable and prior_variable are empty."))
    end
end
function risk_measure_nothing_matrix_factory(::Nothing, ::Nothing)
    throw(ArgumentError("Both risk_variable and prior_variable are nothing."))
end
function risk_measure_nothing_matrix_factory(::Nothing,
                                             prior_variable::AbstractMatrix{<:Real})
    return if !isempty(prior_variable)
        prior_variable
    else
        throw(ArgumentError("prior_variable is empty."))
    end
end
function risk_measure_nothing_matrix_factory(risk_variable::AbstractMatrix{<:Real},
                                             ::Nothing)
    return if !isempty(risk_variable)
        risk_variable
    else
        throw(ArgumentError("risk_variable is empty."))
    end
end
function risk_measure_nothing_real_vec_factory(risk_variable::AbstractVector{<:Real},
                                               cluster::AbstractVector)
    return if !isempty(risk_variable)
        view(risk_variable, cluster)
    else
        risk_variable
    end
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
