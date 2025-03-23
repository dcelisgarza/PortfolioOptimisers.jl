struct ArithmeticReturn{T1 <: UncertaintySet} <: PortfolioReturnType
    uncertainty_set::T1
end
function ArithmeticReturn(; uncertainty_set::UncertaintySet = NoUncertaintySet())
    return ArithmeticReturn{typeof(uncertainty_set)}(uncertainty_set)
end
function cluster_return_factory(r::ArithmeticReturn; uncertainty_set::UncertaintySet,
                                cluster::AbstractVector, kwargs...)
    uset = uncertainty_set_factory(r.uncertainty_set, uncertainty_set, cluster)
    return ArithmeticReturn(; uncertainty_set = uset)
end

export ArithmeticReturn
