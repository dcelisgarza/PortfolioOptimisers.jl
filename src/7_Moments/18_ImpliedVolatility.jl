abstract type ImpliedVolatilityAlgorithm <: AbstractAlgorithm end
abstract type ImpliedVolatilityRegressionEstimator <: ImpliedVolatilityAlgorithm end

struct ImpliedVolatilityRegression{T1 <: ImpliedVolatilityRegressionEstimator} <:
       ImpliedVolatilityAlgorithm
    re::T1
end
struct ImpliedVolatility{T1 <: AbstractCovarianceEstimator,
                         T2 <: AbstractMatrixProcessingEstimator, T3, T4 <: Real,
                         T5 <: Integer, T6 <: Union{Nothing, <:Real, <:AbstractVector}} <:
       AbstractCovarianceEstimator
    ce::T1
    mp::T2
    re::T3
    af::T4
    ws::T5
    vrpa::T6
end
function ImpliedVolatility(; ce::AbstractCovarianceEstimator = Covariance(),
                           mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing(),
                           re, af::Real = 252, ws::Integer = 20,
                           vrpa::Union{Nothing, <:Real, <:AbstractVector} = nothing)
    @smart_assert(ws > 2)
    #! either re or vrpa must be provided
    if isa(vrpa, AbstractVector)
        @smart_assert(!isempty(vrpa) &&
                      all(isfinite, vrpa) &&
                      all(vrpa .>= zero(eltype(vrpa))))
    end
    return ImpliedVolatility{typeof(ce), typeof(mp), typeof(re), typeof(af), typeof(ws),
                             typeof(vrpa)}(ce, mp, re, af, ws, vrpa)
end
function factory(ce::ImpliedVolatility, w::Union{Nothing, <:AbstractWeights} = nothing)
    return ImpliedVolatility(; ce = factory(ce.ce, w), mp = ce.mp)
end
function realised_vol(ce::AbstractVarianceEstimator, X::AbstractMatrix, ws::Integer)
    T, N = size(X)
    chunk = div(T, ws)
    return reshape(StatsBase.std(ce,
                                 reshape(view(X, (1 + T - chunk * ws):T, :), chunk, ws, N);
                                 dims = 1), :, N)
end
function implied_vol(X::AbstractMatrix, ws::Integer)
    T = size(X, 1)
    chunk = div(T, ws)
    view(X, (T - (chunk - 1) * ws):ws:T, :)
    return nothing
end
export ImpliedVolatility, realised_vol
