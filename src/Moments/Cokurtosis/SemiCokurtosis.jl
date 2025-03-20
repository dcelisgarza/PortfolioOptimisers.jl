struct SemiCokurtosis{T1 <: ExpectedReturnsEstimator, T2 <: MatrixProcessing} <:
       CokurtosisEstimator
    me::T1
    mp::T2
end
function SemiCokurtosis(; me::ExpectedReturnsEstimator = SimpleExpectedReturns(),
                        mp::MatrixProcessing = DefaultMatrixProcessing())
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
function moment_factory_w(ce::SemiCokurtosis,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return SemiCokurtosis(; me = moment_factory_w(ce.me, w), mp = ce.mp)
end

export SemiCokurtosis
