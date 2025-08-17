abstract type DimensionReductionTarget <: AbstractRegressionAlgorithm end
struct PCA{T1} <: DimensionReductionTarget
    kwargs::T1
end
function PCA(; kwargs::NamedTuple = (;))
    return PCA(kwargs)
end
function MultivariateStats.fit(drtgt::PCA, X::AbstractMatrix)
    return MultivariateStats.fit(MultivariateStats.PCA, X; drtgt.kwargs...)
end
struct PPCA{T1} <: DimensionReductionTarget
    kwargs::T1
end
function PPCA(; kwargs::NamedTuple = (;))
    return PPCA(kwargs)
end
function MultivariateStats.fit(drtgt::PPCA, X::AbstractMatrix)
    return MultivariateStats.fit(MultivariateStats.PPCA, X; drtgt.kwargs...)
end
struct DimensionReductionRegression{T1, T2, T3, T4} <: AbstractRegressionEstimator
    me::T1
    ve::T2
    drtgt::T3
    retgt::T4
end
function DimensionReductionRegression(;
                                      me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                                      ve::AbstractVarianceEstimator = SimpleVariance(),
                                      drtgt::DimensionReductionTarget = PCA(),
                                      retgt::AbstractRegressionTarget = LinearModel())
    return DimensionReductionRegression(me, ve, drtgt, retgt)
end
function prep_dim_red_reg(drtgt::DimensionReductionTarget, X::AbstractMatrix)
    N = size(X, 1)
    X_std = StatsBase.standardize(StatsBase.ZScoreTransform, transpose(X); dims = 2)
    model = fit(drtgt, X_std)
    Xp = transpose(predict(model, X_std))
    Vp = projection(model)
    x1 = [ones(eltype(X), N) Xp]
    return x1, Vp
end
function regression(retgt::AbstractRegressionTarget, y::AbstractVector, mu::AbstractVector,
                    sigma::AbstractVector, x1::AbstractMatrix, Vp::AbstractMatrix)
    fit_result = fit(retgt, x1, y)
    beta_pc = coef(fit_result)[2:end]
    beta = Vp * beta_pc ./ sigma
    beta0 = mean(y) - dot(beta, mu)
    pushfirst!(beta, beta0)
    return beta
end
function regression(re::DimensionReductionRegression, X::AbstractMatrix, F::AbstractMatrix)
    cols = size(F, 2) + 1
    rows = size(X, 2)
    rr = zeros(promote_type(eltype(F), eltype(X)), rows, cols)
    f1, Vp = prep_dim_red_reg(re.drtgt, F)
    mu = mean(re.me, F; dims = 1)
    sigma = vec(std(re.ve, F; mean = mu, dims = 1))
    mu = vec(mu)
    for i in axes(rr, 1)
        rr[i, :] .= regression(re.retgt, view(X, :, i), mu, sigma, f1, Vp)
    end
    b = view(rr, :, 1)
    M = view(rr, :, 2:cols)
    L = transpose(pinv(Vp) * transpose(M .* transpose(sigma)))
    return Regression(; b = b, M = M, L = L)
end

export PCA, PPCA, DimensionReductionRegression
