struct SemiCokurtosis{T1 <: AbstractExpectedReturnsEstimator,
                      T2 <: AbstractMatrixProcessingEstimator} <: CokurtosisEstimator
    me::T1
    mp::T2
end
function SemiCokurtosis(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                        mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing())
    return SemiCokurtosis{typeof(me), typeof(mp)}(me, mp)
end
function cokurtosis(ke::SemiCokurtosis, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mu = mean(ke.me, X)
    X = min.(X .- mu, zero(eltype(X)))
    return _cokurosis(X, ke.mp)
end
function w_moment_factory(ce::SemiCokurtosis,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return SemiCokurtosis(; me = w_moment_factory(ce.me, w), mp = ce.mp)
end

export SemiCokurtosis
