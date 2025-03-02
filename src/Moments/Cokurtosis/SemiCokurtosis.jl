struct SemiCokurtosis{T1 <: ExpectedReturnsEstimator, T2 <: MatrixProcessing} <:
       CokurtosisEstimator
    me::T1
    mp::T2
end
function SemiCokurtosis(; me::ExpectedReturnsEstimator = SimpleExpectedReturns(),
                        mp::MatrixProcessing = DefaultMatrixProcessing())
    return SemiCokurtosis{typeof(me), typeof(mp)}(me, mp)
end
function cokurt(ke::SemiCokurtosis, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    T, N = size(X)
    mu = mean(ke.me, X)
    X = min.(X .- mu, zero(eltype(X)))
    o = transpose(range(; start = one(eltype(X)), stop = one(eltype(X)), length = N))
    z = kron(o, X) .* kron(X, o)
    sckurt = transpose(z) * z / T
    mtx_process!(ke.mp, sckurt, X)
    return sckurt
end

export SemiCokurtosis
