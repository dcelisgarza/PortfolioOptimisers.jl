function ucs_factory(::Nothing, ::Nothing, args...)
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
function ucs_factory(risk_ucs::BoxUncertaintySetResult{<:Union{<:AbstractVector,
                                                               <:SubArray{<:Any, 0, <:Any,
                                                                          <:Any, <:Any},
                                                               <:SubArray{<:Any, 1, <:Any,
                                                                          <:Any, <:Any}},
                                                       <:Union{<:AbstractVector,
                                                               <:SubArray{<:Any, 0, <:Any,
                                                                          <:Any, <:Any},
                                                               <:SubArray{<:Any, 1, <:Any,
                                                                          <:Any, <:Any}}},
                     ::Any, i)
    return BoxUncertaintySetResult(; lb = view(risk_ucs.lb, i), ub = view(risk_ucs.ub, i))
end
function ucs_factory(risk_ucs::BoxUncertaintySetResult{<:Union{<:AbstractMatrix,
                                                               <:SubArray{<:Any, 2, <:Any,
                                                                          <:Any, <:Any}},
                                                       <:Union{<:AbstractMatrix,
                                                               <:SubArray{<:Any, 2, <:Any,
                                                                          <:Any, <:Any}}},
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
