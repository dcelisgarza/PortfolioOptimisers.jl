struct PPCATarget{T1 <: NamedTuple} <: AbstractPCATarget
    kwargs::T1
end
function PPCATarget(; kwargs::NamedTuple = (;))
    return PPCATarget{typeof(kwargs)}(kwargs)
end
function MultivariateStats.fit(type::PPCATarget, X::AbstractMatrix)
    return MultivariateStats.fit(MultivariateStats.PPCA, X; type.kwargs...)
end

export PPCATarget
