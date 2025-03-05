struct FullCokurtosis{T1 <: ExpectedReturnsEstimator, T2 <: MatrixProcessing} <:
       CokurtosisEstimator
    me::T1
    mp::T2
end
function FullCokurtosis(; me::ExpectedReturnsEstimator = SimpleExpectedReturns(),
                        mp::MatrixProcessing = DefaultMatrixProcessing())
    return FullCokurtosis{typeof(me), typeof(mp)}(me, mp)
end
function cokurtosis(ke::FullCokurtosis, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mu = mean(ke.me, X; kwargs...)
    X = X .- mu
    return _cokurosis(X, ke.mp)
end

export FullCokurtosis
