struct Detone{T1 <: Integer} <: AbstractEstimator
    n::T1
end
function Detone(; n::Integer = 1)
    @smart_assert(n >= zero(n))
    return Detone{typeof(n)}(n)
end
function fit_estimator!(ce::Detone, pdm::Union{Nothing, <:PosDefEstimator},
                        X::AbstractMatrix)
    n = ce.n
    @smart_assert(one(size(X, 1)) <= n <= size(X, 1))
    n -= 1
    s = diag(X)
    iscov = any(.!isone.(s))
    if iscov
        s .= sqrt.(s)
        StatsBase.cov2cor!(X, s)
    end
    vals, vecs = eigen(X)
    _vals = Diagonal(vals)[(end - n):end, (end - n):end]
    _vecs = vecs[:, (end - n):end]
    X .-= _vecs * _vals * transpose(_vecs)
    X .= cov2cor(X)
    fit_estimator!(pdm, X)
    if iscov
        StatsBase.cor2cov!(X, s)
    end
    return nothing
end
function fit_estimator(ce::Detone, pdm::Union{Nothing, <:PosDefEstimator},
                       X::AbstractMatrix)
    X = copy(X)
    fit_estimator!(ce, pdm, X)
    return X
end

export Detone
