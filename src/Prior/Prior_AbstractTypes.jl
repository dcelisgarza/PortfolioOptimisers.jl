abstract type PriorEstimator end
abstract type AbstractPriorModel end
struct Prior{T1 <: AbstractMatrix, T2 <: AbstractVector, T3 <: AbstractMatrix} <:
       AbstractPriorModel
    X::T1
    mu::T2
    sigma::T3
end
function Prior(; X::AbstractMatrix, mu::AbstractVector, sigma::AbstractMatrix)
    @smart_assert(size(X, 2) == length(mu) == size(sigma, 1) == size(sigma, 2))
    return Prior{typeof(X), typeof(mu), typeof(sigma)}(X, mu, sigma)
end

function prior end

export Prior, prior
