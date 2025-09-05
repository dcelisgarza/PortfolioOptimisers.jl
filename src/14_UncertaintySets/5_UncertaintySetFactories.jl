function ucs_factory(::Nothing, ::Nothing)
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
function ucs_view(risk_ucs::Union{Nothing, <:AbstractUncertaintySetEstimator}, ::Any)
    return risk_ucs
end
function ucs_view(risk_ucs::BoxUncertaintySet{<:AbstractVector, <:AbstractVector},
                  i::AbstractVector)
    return BoxUncertaintySet(; lb = view(risk_ucs.lb, i), ub = view(risk_ucs.ub, i))
end
function ucs_view(risk_ucs::BoxUncertaintySet{<:AbstractMatrix, <:AbstractMatrix},
                  i::AbstractVector)
    return BoxUncertaintySet(; lb = view(risk_ucs.lb, i, i), ub = view(risk_ucs.ub, i, i))
end
function ucs_view(risk_ucs::EllipseUncertaintySet{<:AbstractMatrix, <:Any,
                                                  <:SigmaEllipseUncertaintySet},
                  i::AbstractVector)
    i = fourth_moment_index_factory(floor(Int, sqrt(size(risk_ucs.sigma, 1))), i)
    return EllipseUncertaintySet(; sigma = view(risk_ucs.sigma, i, i), k = risk_ucs.k,
                                 class = risk_ucs.class)
end
function ucs_view(risk_ucs::EllipseUncertaintySet{<:AbstractMatrix, <:Any,
                                                  <:MuEllipseUncertaintySet},
                  i::AbstractVector)
    return EllipseUncertaintySet(; sigma = view(risk_ucs.sigma, i, i), k = risk_ucs.k,
                                 class = risk_ucs.class)
end
