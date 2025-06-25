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
# Compute realised volatility over non-overlapping windows
function compute_realised_vol(returns::AbstractMatrix, window_size::Int;
                              corrected::Bool = true)
    n_observations, n_assets = size(returns)
    chunks = div(n_observations, window_size)
    # Only use the last (chunks * window_size) rows so we can reshape
    X = returns[(n_observations - chunks * window_size + 1):end, :]
    Xr = reshape(X, window_size, chunks, n_assets)
    # Compute std along the window dimension (dim=1), result is (1, chunks, n_assets)
    # Drop the first dimension to get (chunks, n_assets)
    stds = dropdims(std(Xr; dims = 1, corrected = corrected); dims = 1)
    return stds
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
export ImpliedVolatility, predict_realised_vols
