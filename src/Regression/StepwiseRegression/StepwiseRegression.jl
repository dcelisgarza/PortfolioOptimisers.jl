abstract type StepwiseRegression <: RegressionMethod end
abstract type StepwiseRegressionCriteria end
abstract type MinValStepwiseRegressionCriteria <: StepwiseRegressionCriteria end
abstract type MaxValStepwiseRegressionCriteria <: StepwiseRegressionCriteria end
struct AIC <: MinValStepwiseRegressionCriteria end
struct AICC <: MinValStepwiseRegressionCriteria end
struct BIC <: MinValStepwiseRegressionCriteria end
struct RSquared <: MaxValStepwiseRegressionCriteria end
struct AdjustedRSquared <: MaxValStepwiseRegressionCriteria end
struct PValue{T1 <: Real} <: StepwiseRegressionCriteria
    threshold::T1
end
function PValue(; threshold::Real = 0.05)
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return PValue{typeof(threshold)}(threshold)
end
function regression_criterion_func(::AIC)
    return GLM.aic
end
function regression_criterion_func(::AICC)
    return GLM.aicc
end
function regression_criterion_func(::BIC)
    return GLM.bic
end
function regression_criterion_func(::RSquared)
    return GLM.r2
end
function regression_criterion_func(::AdjustedRSquared)
    return GLM.adjr2
end
function regression_threshold(::MinValStepwiseRegressionCriteria)
    return Inf
end
function regression_threshold(::MaxValStepwiseRegressionCriteria)
    return -Inf
end
function add_best_asset_after_failure_pval!(included::AbstractVector, F::AbstractMatrix,
                                            x::AbstractVector)
    if !isempty(included)
        return nothing
    end
    T, N = size(F)
    ovec = range(; start = 1, stop = 1, length = T)
    indices = 1:N
    excluded = setdiff(indices, included)
    best_pval = Inf
    new_feature = 0
    for i ∈ excluded
        factors = [included; i]
        f1 = [ovec view(F, :, factors)]
        fit_result = GLM.lm(f1, x)
        new_pvals = coeftable(fit_result).cols[4][2:end]
        idx = findfirst(x -> x == i, factors)
        test_pval = new_pvals[idx]
        if best_pval > test_pval
            best_pval = test_pval
            new_feature = i
        end
    end
    @warn("No asset with p-value lower than threshold. Best we can do is feature $new_feature, with p-value $best_pval.")
    push!(included, new_feature)
    return nothing
end
function regression(method::StepwiseRegression, X::AbstractMatrix, F::AbstractMatrix)
    features = 1:size(F, 2)
    cols = size(F, 2) + 1
    N, rows = size(X)
    ovec = range(; start = 1, stop = 1, length = N)
    loadings = zeros(promote_type(eltype(F), eltype(X)), rows, cols)
    for i ∈ axes(loadings, 1)
        included = _regression(method, view(X, :, i), F)
        x1 = !isempty(included) ? [ovec view(F, :, included)] : reshape(ovec, :, 1)
        fit_result = GLM.lm(x1, view(X, :, i))
        params = coef(fit_result)
        loadings[i, 1] = params[1]
        if isempty(included)
            continue
        end
        idx = [findfirst(x -> x == i, features) + 1 for i ∈ included]
        loadings[i, idx] .= params[2:end]
    end
    return LoadingsMatrix(; b = view(loadings, :, 1), M = view(loadings, :, 2:cols))
end

export AIC, AICC, BIC, RSquared, AdjustedRSquared, PValue
