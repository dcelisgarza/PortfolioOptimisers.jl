abstract type RankCovarianceEstimator <: AbstractCovarianceEstimator end
struct KendallCovariance{T1 <: AbstractVarianceEstimator} <: RankCovarianceEstimator
    ve::T1
end
function KendallCovariance(; ve::AbstractVarianceEstimator = SimpleVariance())
    return KendallCovariance{typeof(ve)}(ve)
end
function Statistics.cor(::KendallCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return corkendall(X)
end
function Statistics.cov(ce::KendallCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1, kwargs...)
    return corkendall(X) ⊙ (std_vec ⊗ std_vec)
end
function factory(ce::KendallCovariance, w::Union{Nothing, <:AbstractWeights} = nothing)
    return KendallCovariance(; ve = factory(ce.ve, w))
end
struct SpearmanCovariance{T1 <: AbstractVarianceEstimator} <: RankCovarianceEstimator
    ve::T1
end
function SpearmanCovariance(; ve::AbstractVarianceEstimator = SimpleVariance())
    return SpearmanCovariance{typeof(ve)}(ve)
end
function Statistics.cor(::SpearmanCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return corspearman(X)
end
function Statistics.cov(ce::SpearmanCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1, kwargs...)
    return corspearman(X) ⊙ (std_vec ⊗ std_vec)
end
function factory(ce::SpearmanCovariance, w::Union{Nothing, <:AbstractWeights} = nothing)
    return SpearmanCovariance(; ve = factory(ce.ve, w))
end
#=
function Base.show(io::IO, ce::RankCovarianceEstimator)
    name = string(typeof(ce))
    name = name[1:(findfirst(x -> x == '{', name) - 1)]
    println(io, name)
    for field in fieldnames(typeof(ce))
        val = getfield(ce, field)
        print(io, "  ", string(field), " ")
        if isnothing(val)
            println(io, "| nothing")
        elseif isa(val, AbstractVarianceEstimator)
            ioalg = IOBuffer()
            show(ioalg, val)
            algstr = String(take!(ioalg))
            alglines = split(algstr, '\n')
            println(io, "| ", alglines[1])
            for l in alglines[2:end]
                println(io, "     | ", l)
            end
        else
            println(io, "| $(typeof(val)): ", repr(val))
        end
    end
end
=#

export KendallCovariance, SpearmanCovariance
