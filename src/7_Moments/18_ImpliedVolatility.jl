abstract type ImpliedVolatilityAlgorithm <: AbstractAlgorithm end
abstract type ImpliedVolatilityRegressionEstimator <: ImpliedVolatilityAlgorithm end
struct ImpliedVolatilityRegression{T1 <: Real} <: ImpliedVolatilityAlgorithm
    ws::T1
end
function ImpliedVolatilityRegression(; ws::Real = 20)
    @smart_assert(ws > 2)
    return ImpliedVolatilityRegression{typeof(ws)}(ws)
end
struct ImpliedVolatilityPremium{T1 <: Union{<:Real, <:AbstractVector{<:Real}}} <:
       ImpliedVolatilityAlgorithm
    val::T1
end
function ImpliedVolatilityPremium(; val::Union{<:Real, <:AbstractVector{<:Real}} = 20)
    if isa(val, Real)
        @smart_assert(isfinite(val) && val >= zero(val))
    elseif isa(val, AbstractVector)
        @smart_assert(!isempty(val) &&
                      all(isfinite, val) &&
                      all(x -> x >= zero(eltype(val)), val))
    end
    return ImpliedVolatilityPremium{typeof(val)}(val)
end
struct ImpliedVolatility{T1 <: AbstractCovarianceEstimator,
                         T2 <: AbstractMatrixProcessingEstimator,
                         T3 <: ImpliedVolatilityRegressionEstimator, T4 <: Real} <:
       AbstractCovarianceEstimator
    ce::T1
    mp::T2
    alg::T3
    af::T4
end
#! Rework
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
function realised_vol(ce::AbstractVarianceEstimator, X::AbstractMatrix, ws::Integer,
                      chunk::Union{Nothing, <:Integer} = nothing,
                      T::Union{Nothing, <:Integer} = nothing,
                      N::Union{Nothing, <:Integer} = nothing)
    if isnothing(chunk) || isnothing(T) || isnothing(N)
        T, N = size(X)
        chunk = div(T, ws)
    end
    return dropdims(StatsBase.std(ce,
                                  reshape(view(X, (1 + T - chunk * ws):T, :), ws, chunk, N);
                                  dims = 1); dims = 1)
end
function implied_vol(X::AbstractMatrix, ws::Integer,
                     chunk::Union{Nothing, <:Integer} = nothing,
                     T::Union{Nothing, <:Integer} = nothing,
                     N::Union{Nothing, <:Integer} = nothing)
    if isnothing(chunk) || isnothing(T) || isnothing(N)
        T, N = size(X)
        chunk = div(T, ws)
    end
    return view(X, (T - (chunk - 1) * ws):ws:T, :)
end
#! dispatch based on ImpliedVolatilityAlgorithm
function predict_realised_vols(ce::AbstractVarianceEstimator, X::AbstractMatrix,
                               iv::AbstractMatrix, ws::Integer)
    T, N = size(X)
    chunk = div(T, ws)
    @smart_assert(chunk > 2)
    rv = realised_vol(ce, X, ws, chunk, T, N)
    iv = implied_vol(iv, ws, chunk, T, N)
    @smart_assert(size(rv) == size(iv))
    T2 = size(iv, 1)
    rv = log.(rv)
    iv = log.(iv)
    ovec = range(; start = one(promote_type(eltype(rv), eltype(iv))),
                 stop = one(promote_type(eltype(rv), eltype(iv))), length = T2 - 1)
    reg = Matrix{promote_type(eltype(rv), eltype(iv))}(undef, N, 3)
    r2s = Vector{promote_type(eltype(rv), eltype(iv))}(undef, N)
    rv_p = Vector{promote_type(eltype(rv), eltype(iv))}(undef, N)
    fr = []
    for i ∈ 1:N
        X = [view(iv, :, i) view(rv, :, i)]
        X_t = [ovec view(X, 1:(T2 - 1), :)]
        X_p = [one(eltype(X)) transpose(X[T2, :])]
        y_t = view(rv, 2:T2, i)
        #! Call fit(LinearModel, X, y, args...; kwargs...) where you dispatch on the first argument.
        fri = GLM.lm(X_t, y_t)
        params = coef(fri)
        reg[i, 1] = params[1]
        reg[i, 2:3] .= params[2:end]
        #! Use an AbstractMinValStepwiseRegressionCriterion
        r2s[i] = r2(fri)
        rvpi = predict(fri, X_p)[1]
        rv_p[i] = exp(rvpi)
        push!(fr, fri)
    end
    return RegressionResult(; b = view(reg, :, 1), M = view(reg, :, 2:3)), r2s, rv_p, fr
end
function StatsBase.cov(ce::ImpliedVolatility, X::AbstractMatrix; dims::Int = 1,
                       mean = nothing, iv::AbstractMatrix, kwargs...)
    @smart_assert(size(X) == size(iv))
    @smart_assert(all(x -> x >= zero(eltype(iv)), iv))
    rho = cor(ce.ce, X; dims = dims, mean = mean, iv = iv, kwargs...)
    iv = iv / sqrt(ce.af)
    #iv= predict_realised_vols(ce::AbstractVarianceEstimator, X::AbstractMatrix,
    #                       iv::AbstractMatrix, ws::Integer)
    return nothing
end
function StatsBase.cor(ce::ImpliedVolatility, X::AbstractMatrix; dims::Int = 1,
                       mean = nothing, kwargs...) end
export ImpliedVolatility, predict_realised_vols
