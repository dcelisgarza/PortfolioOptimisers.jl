struct ArithmeticReturn{T1 <: Union{Nothing, <:UncertaintySet}} <: PortfolioReturnType
    uncertainty_set::T1
end
function ArithmeticReturn(; uncertainty_set::Union{Nothing, <:UncertaintySet} = nothing)
    return ArithmeticReturn{typeof(uncertainty_set)}(uncertainty_set)
end
function cluster_return_factory(r::ArithmeticReturn, cluster::AbstractVector,
                                uncertainty_set::Union{Nothing, <:UncertaintySet}, args...)
    uset = uncertainty_set_factory(r.uncertainty_set, uncertainty_set, cluster)
    return ArithmeticReturn(; uncertainty_set = uset)
end

export ArithmeticReturn
