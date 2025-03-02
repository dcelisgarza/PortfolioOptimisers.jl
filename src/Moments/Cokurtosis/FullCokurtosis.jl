struct FullCokurtosis{T1 <: ExpectedReturnsEstimator, T2 <: MatrixProcessing} <:
       CokurtosisEstimator
    me::T1
    mp::T2
end
function FullCokurtosis(; me::ExpectedReturnsEstimator = SimpleExpectedReturns(),
                        mp::MatrixProcessing = DefaultMatrixProcessing())
    return FullCokurtosis{typeof(me), typeof(mp)}(me, mp)
end
function cokurt(ke::FullCokurtosis, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    T, N = size(X)
    mu = mean(ke.me, X)
    X = X .- mu
    o = transpose(range(; start = one(eltype(X)), stop = one(eltype(X)), length = N))
    z = kron(o, X) .* kron(X, o)
    ckurt = transpose(z) * z / T
    mtx_process!(ke.mp, ckurt, X)
    return ckurt
end

export FullCokurtosis
