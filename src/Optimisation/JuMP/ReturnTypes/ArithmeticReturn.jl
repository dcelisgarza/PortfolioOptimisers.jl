struct ArithmeticReturn{T1 <: Union{Nothing, <:Real},
                        T2 <: Union{Nothing, <:AbstractUncertaintySetResult,
                                    <:AbstractUncertaintySetEstimator}} <:
       PortfolioReturnType
    lb::T1
    ucs::T2
end
function ArithmeticReturn(; lb::Union{Nothing, <:Real} = nothing,
                          ucs::Union{Nothing, <:AbstractUncertaintySetResult,
                                     <:AbstractUncertaintySetEstimator} = nothing)
    if isa(lb, Real)
        @smart_assert(isfinite(lb))
    end
    if isa(ucs, EllipseUncertaintySetResult)
        @smart_assert(isa(ucs,
                          <:EllipseUncertaintySetResult{<:Any, <:Any,
                                                        <:MuEllipseUncertaintySetResult}))
    end
    return ArithmeticReturn{typeof(lb), typeof(ucs)}(lb, ucs)
end
function cluster_return_factory(r::ArithmeticReturn, cluster::AbstractVector,
                                ucs::Union{Nothing, <:AbstractUncertaintySetResult,
                                           <:AbstractUncertaintySetEstimator}, args...)
    uset = ucs_factory(r.ucs, ucs, cluster)
    return ArithmeticReturn(; lb = r.lb, ucs = uset)
end

export ArithmeticReturn
