struct PCATarget{T1 <: NamedTuple} <: AbstractPCATarget
    kwargs::T1
end
function PCATarget(; kwargs::NamedTuple = (;))
    return PCATarget{typeof(kwargs)}(kwargs)
end
function MultivariateStats.fit(type::PCATarget, X::AbstractMatrix)
    return MultivariateStats.fit(MultivariateStats.PCA, X; type.kwargs...)
end

export PCATarget
