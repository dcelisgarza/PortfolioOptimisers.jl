struct PCARegression{T1 <: ExpectedReturnsEstimator,
                     T2 <: PortfolioOptimisersVarianceEstimator,
                     T3 <: DimensionReductionTarget} <: DimensionReductionRegression
    me::T1
    ve::T2
    target::T3
end
function PCARegression(; me::ExpectedReturnsEstimator = SimpleExpectedReturns(),
                       ve::PortfolioOptimisersVarianceEstimator = SimpleVariance(),
                       target::DimensionReductionTarget = PCATarget())
    return PCARegression{typeof(me), typeof(ve), typeof(target)}(me, ve, target)
end
function prep_dim_red_reg(type::PCARegression, X::AbstractMatrix)
    N = size(X, 1)
    X_std = StatsBase.standardize(StatsBase.ZScoreTransform, transpose(X); dims = 2)
    model = fit(type.target, X_std)
    Xp = transpose(predict(model, X_std))
    Vp = projection(model)
    x1 = [ones(eltype(X), N) Xp]
    return x1, Vp
end
function _regression(mu::AbstractVector, sigma::AbstractVector, x1::AbstractMatrix,
                     Vp::AbstractMatrix, y::AbstractVector)
    fit_result = GLM.lm(x1, y)
    beta_pc = coef(fit_result)[2:end]
    beta = Vp * beta_pc ./ sigma
    beta0 = mean(y) - dot(beta, mu)
    pushfirst!(beta, beta0)
    return beta
end
function regression(method::PCARegression, F::AbstractMatrix, X::AbstractMatrix)
    cols = size(F, 2) + 1
    rows = size(X, 2)
    loadings = zeros(promote_type(eltype(F), eltype(X)), rows, cols)
    f1, Vp = prep_dim_red_reg(method, F)
    mu = mean(method.me, F; dims = 1)
    sigma = vec(std(method.ve, F; mean = mu, dims = 1))
    mu = vec(mu)
    for i ∈ axes(loadings, 1)
        loadings[i, :] .= _regression(mu, sigma, f1, Vp, view(X, :, i))
    end
    return LoadingsMatrix(; c = view(loadings, :, 1), M = view(loadings, :, 2:cols))
end

export PCARegression
