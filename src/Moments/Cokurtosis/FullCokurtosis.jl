struct FullCokurtosis{T1 <: ExpectedReturnsEstimator, T2 <: MatrixProcessing} <:
       CokurtosisEstimator
    me::T1
    mp::T2
end
function FullCokurtosis(; me::ExpectedReturnsEstimator = SimpleExpectedReturns(),
                        mp::MatrixProcessing = DefaultMatrixProcessing())
    return FullCokurtosis{typeof(me), typeof(mp)}(me, mp)
end
function cokurtosis(ke::FullCokurtosis, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mu = mean(ke.me, X)
    X = X .- mu
    return _cokurosis(X, ke.mp)
end
function moment_factory_w(ce::FullCokurtosis,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return FullCokurtosis(; me = moment_factory_w(ce.me, w), mp = ce.mp)
end

export FullCokurtosis
