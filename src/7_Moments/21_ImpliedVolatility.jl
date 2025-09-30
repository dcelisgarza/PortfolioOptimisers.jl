abstract type ImpliedVolatilityAlgorithm <: AbstractAlgorithm end
abstract type ImpliedVolatilityRegressionEstimator <: ImpliedVolatilityAlgorithm end
struct ImpliedVolatilityRegression{T1, T2,
                                   #    T3 <: AbstractStepwiseRegressionCriterion,
                                   T3} <: ImpliedVolatilityAlgorithm
    ve::T1
    ws::T2
    # crit::T3
    re::T3
    function ImpliedVolatilityRegression(ve::AbstractVarianceEstimator, ws::Real,
                                         re::AbstractRegressionTarget)
        @argcheck(ws > 2)
        return new{typeof(ve), typeof(ws), typeof(re)}(ve, ws, re)
    end
end
function ImpliedVolatilityRegression(; ve::AbstractVarianceEstimator = SimpleVariance(),
                                     ws::Real = 20,
                                     #  crit::AbstractStepwiseRegressionCriterion = RSquared(),
                                     re::AbstractRegressionTarget = LinearModel())
    return ImpliedVolatilityRegression(ve, ws, re)
end
struct ImpliedVolatilityPremium <: ImpliedVolatilityAlgorithm end
struct ImpliedVolatility{T1, T2, T3, T4} <: AbstractCovarianceEstimator
    ce::T1
    mp::T2
    alg::T3
    af::T4
    function ImpliedVolatility(ce::AbstractCovarianceEstimator,
                               mp::AbstractMatrixProcessingEstimator,
                               alg::ImpliedVolatilityAlgorithm, af::Real)
        @argcheck(isfinite(af) && af > zero(af))
        return new{typeof(ce), typeof(mp), typeof(alg), typeof(af)}(ce, mp, alg, af)
    end
end
function ImpliedVolatility(; ce::AbstractCovarianceEstimator = Covariance(),
                           mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing(),
                           alg::ImpliedVolatilityAlgorithm = ImpliedVolatilityRegression(),
                           af::Real = 252)
    return ImpliedVolatility(ce, mp, alg, af)
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
    return dropdims(Statistics.std(ce,
                                   reshape(view(X, (1 + T - chunk * ws):T, :), ws, chunk,
                                           N); dims = 1); dims = 1)
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
function predict_realised_vols(::ImpliedVolatilityPremium, iv::AbstractMatrix, ::Any,
                               ivpa::Nothing)
    throw(ArgumentError("ImpliedVolatilityPremium requires `ivpa`to be a `<:Real` or `<:AbstractVector{<:Real}`"))
end
function predict_realised_vols(::ImpliedVolatilityPremium, iv::AbstractMatrix, ::Any,
                               ivpa::Union{<:Real, <:AbstractVector{<:Real}})
    return view(iv, size(iv, 1), :) ⊘ ivpa
end
function predict_realised_vols(alg::ImpliedVolatilityRegression, iv::AbstractMatrix,
                               X::AbstractMatrix, ::Any)
    T, N = size(X)
    chunk = div(T, alg.ws)
    @argcheck(chunk > 2)
    rv = realised_vol(alg.ve, X, alg.ws, chunk, T, N)
    iv = implied_vol(iv, alg.ws, chunk, T, N)
    @argcheck(size(rv) == size(iv))
    T2 = size(iv, 1)
    rv = log.(rv)
    iv = log.(iv)
    # criterion_func = regression_criterion_func(alg.crit)
    ovec = range(; start = one(promote_type(eltype(rv), eltype(iv))),
                 stop = one(promote_type(eltype(rv), eltype(iv))), length = T2 - 1)
    # reg = Matrix{promote_type(eltype(rv), eltype(iv))}(undef, N, 3)
    # r2s = Vector{promote_type(eltype(rv), eltype(iv))}(undef, N)
    rv_p = Vector{promote_type(eltype(rv), eltype(iv))}(undef, N)
    # fr = []
    for i in 1:N
        X = [view(iv, :, i) view(rv, :, i)]
        X_t = [ovec view(X, 1:(T2 - 1), :)]
        X_p = [one(eltype(X)) transpose(view(X, T2, :))]
        y_t = view(rv, 2:T2, i)
        fri = fit(alg.re, X_t, y_t)
        # params = coef(fri)
        # reg[i, 1] = params[1]
        # reg[i, 2:3] .= params[2:end]
        # r2s[i] = criterion_func(fri)
        rv_pi = predict(fri, X_p)[1]
        rv_p[i] = exp(rv_pi)
        # push!(fr, fri)
    end
    #, Regression(; b = view(reg, :, 1), M = view(reg, :, 2:3)), r2s, fr
    return rv_p
end
function Statistics.cov(ce::ImpliedVolatility, X::AbstractMatrix; dims::Int = 1,
                        mean = nothing, iv::AbstractMatrix,
                        ivpa::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                        kwargs...)
    sigma = cor(ce.ce, X; dims = dims, mean = mean, iv = iv, kwargs...)
    iv = iv / sqrt(ce.af)
    iv = predict_realised_vols(ce.alg, X, iv, ivpa)
    StatsBase.cov2cor!(sigma, iv)
    matrix_processing!(ce.mp, sigma, X; kwargs...)
    return sigma
end
function Statistics.cor(ce::ImpliedVolatility, X::AbstractMatrix; dims::Int = 1,
                        mean = nothing, iv::AbstractMatrix,
                        ivpa::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                        kwargs...)
    rho = cor(ce.ce, X; dims = dims, mean = mean, iv = iv, kwargs...)
    iv = iv / sqrt(ce.af)
    iv = predict_realised_vols(ce.alg, X, iv, ivpa)
    StatsBase.cor2cov!(rho, iv)
    StatsBase.cov2cor!(rho)
    matrix_processing!(ce.mp, rho, X; kwargs...)
    return rho
end
export ImpliedVolatility
