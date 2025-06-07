struct MutualInfoCovariance{T1 <: AbstractVarianceEstimator,
                            T2 <: Union{<:AbstractBins, <:Integer}, T3 <: Bool,
                            T4 <: FLoops.Transducers.Executor} <:
       AbstractCovarianceEstimator
    ve::T1
    bins::T2
    normalise::T3
    threads::T4
end
function MutualInfoCovariance(; ve::AbstractVarianceEstimator = SimpleVariance(),
                              bins::Union{<:AbstractBins, <:Integer} = HacineGharbiRavier(),
                              normalise::Bool = true,
                              threads::FLoops.Transducers.Executor = ThreadedEx())
    if isa(bins, Integer)
        @smart_assert(bins > zero(bins))
    end
    return MutualInfoCovariance{typeof(ve), typeof(bins), typeof(normalise),
                                typeof(threads)}(ve, bins, normalise, threads)
end
function factory(ce::MutualInfoCovariance, w::Union{Nothing, <:AbstractWeights} = nothing)
    return MutualInfoCovariance(; ve = factory(ce.ve, w), bins = ce.bins,
                                normalise = ce.normalise, threads = ce.threads)
end
function StatsBase.cor(ce::MutualInfoCovariance, X::AbstractMatrix; dims::Int = 1,
                       kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return mutual_info(X, ce.bins, ce.normalise, ce.threads)
end
function StatsBase.cov(ce::MutualInfoCovariance, X::AbstractMatrix; dims::Int = 1,
                       kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1)
    return mutual_info(X, ce.bins, ce.normalise, ce.threads) ⊙ (std_vec ⊗ std_vec)
end

export MutualInfoCovariance
