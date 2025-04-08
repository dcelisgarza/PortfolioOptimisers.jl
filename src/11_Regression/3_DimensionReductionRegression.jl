abstract type DimensionReductionTarget <: AbstractRegressionAlgorithm end
struct PCA{T1 <: NamedTuple} <: DimensionReductionTarget
    kwargs::T1
end
function PCA(; kwargs::NamedTuple = (;))
    return PCA{typeof(kwargs)}(kwargs)
end
function MultivariateStats.fit(type::PCA, X::AbstractMatrix)
    return MultivariateStats.fit(MultivariateStats.PCA, X; type.kwargs...)
end
struct PPCA{T1 <: NamedTuple} <: DimensionReductionTarget
    kwargs::T1
end
function PPCA(; kwargs::NamedTuple = (;))
    return PPCA{typeof(kwargs)}(kwargs)
end
function MultivariateStats.fit(type::PPCA, X::AbstractMatrix)
    return MultivariateStats.fit(MultivariateStats.PPCA, X; type.kwargs...)
end
struct DimensionReductionRegression{T1 <: AbstractExpectedReturnsEstimator,
                                    T2 <: AbstractVarianceEstimator,
                                    T3 <: DimensionReductionTarget} <:
       AbstractRegressionEstimator
    me::T1
    ve::T2
    alg::T3
end
function DimensionReductionRegression(;
                                      me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                                      ve::AbstractVarianceEstimator = SimpleVariance(),
                                      alg::DimensionReductionTarget = PCA())
    return DimensionReductionRegression{typeof(me), typeof(ve), typeof(alg)}(me, ve, alg)
end
function prep_dim_red_reg(type::DimensionReductionRegression, X::AbstractMatrix)
    N = size(X, 1)
    X_std = StatsBase.standardize(StatsBase.ZScoreTransform, transpose(X); dims = 2)
    model = fit(type.alg, X_std)
    Xp = transpose(predict(model, X_std))
    Vp = projection(model)
    x1 = [ones(eltype(X), N) Xp]
    return x1, Vp
end
function regression(y::AbstractVector, mu::AbstractVector, sigma::AbstractVector,
                    x1::AbstractMatrix, Vp::AbstractMatrix)
    fit_result = GLM.lm(x1, y)
    beta_pc = coef(fit_result)[2:end]
    beta = Vp * beta_pc ./ sigma
    beta0 = mean(y) - dot(beta, mu)
    pushfirst!(beta, beta0)
    return beta
end
function regression(method::DimensionReductionRegression, X::AbstractMatrix,
                    F::AbstractMatrix)
    cols = size(F, 2) + 1
    rows = size(X, 2)
    loadings = zeros(promote_type(eltype(F), eltype(X)), rows, cols)
    f1, Vp = prep_dim_red_reg(method, F)
    mu = mean(method.me, F; dims = 1)
    sigma = vec(std(method.ve, F; mean = mu, dims = 1))
    mu = vec(mu)
    for i ∈ axes(loadings, 1)
        loadings[i, :] .= regression(view(X, :, i), mu, sigma, f1, Vp)
    end
    return RegressionResult(; b = view(loadings, :, 1), M = view(loadings, :, 2:cols))
end

export PCA, PPCA, DimensionReductionRegression
