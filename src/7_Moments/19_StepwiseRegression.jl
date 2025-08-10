struct PValue{T1} <: AbstractStepwiseRegressionCriterion
    threshold::T1
end
function PValue(; threshold::Real = 0.05)
    @argcheck(zero(threshold) < threshold < one(threshold))
    return PValue(threshold)
end
struct Forward <: AbstractStepwiseRegressionAlgorithm end
struct Backward <: AbstractStepwiseRegressionAlgorithm end
struct StepwiseRegression{T1, T2, T3} <: AbstractRegressionEstimator
    crit::T1
    alg::T2
    target::T3
end
function StepwiseRegression(; crit::AbstractStepwiseRegressionCriterion = PValue(),
                            alg::AbstractStepwiseRegressionAlgorithm = Forward(),
                            target::AbstractRegressionTarget = LinearModel())
    return StepwiseRegression(crit, alg, target)
end
function add_best_asset_after_failure_pval!(target::AbstractRegressionTarget,
                                            included::AbstractVector, F::AbstractMatrix,
                                            x::AbstractVector)
    if !isempty(included)
        return nothing
    end
    T, N = size(F)
    ovec = range(; start = 1, stop = 1, length = T)
    indices = 1:N
    excluded = setdiff(indices, included)
    best_pval = typemax(eltype(x))
    new_feature = 0
    for i in excluded
        factors = [included; i]
        f1 = [ovec view(F, :, factors)]
        fri = fit(target, f1, x)
        new_pvals = coeftable(fri).cols[4][2:end]
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
function regression(re::StepwiseRegression{<:PValue, <:Forward}, x::AbstractVector,
                    F::AbstractMatrix)
    ovec = range(; start = 1, stop = 1, length = length(x))
    indices = 1:size(F, 2)
    included = Vector{eltype(indices)}(undef, 0)
    pvals = Vector{promote_type(eltype(F), eltype(x))}(undef, 0)
    val = zero(promote_type(eltype(F), eltype(x)))
    while val <= re.crit.threshold
        excluded = setdiff(indices, included)
        best_pval = typemax(eltype(x))
        new_feature = 0
        for i in excluded
            factors = [included; i]
            f1 = [ovec view(F, :, factors)]
            fri = fit(re.target, f1, x)
            new_pvals = coeftable(fri).cols[4][2:end]
            idx = findfirst(x -> x == i, factors)
            test_pval = new_pvals[idx]
            if best_pval > test_pval && maximum(new_pvals) <= re.crit.threshold
                best_pval = test_pval
                new_feature = i
                pvals = copy(new_pvals)
            end
        end
        iszero(new_feature) ? break : push!(included, new_feature)
        if !isempty(pvals)
            val = maximum(pvals)
        end
    end
    add_best_asset_after_failure_pval!(re.target, included, F, x)
    return included
end
function get_forward_reg_incl_excl!(::AbstractMinValStepwiseRegressionCriterion, value,
                                    excluded, included, threshold)
    val, key = findmin(value)
    idx = findfirst(x -> x == key, excluded)
    if val < threshold
        push!(included, popat!(excluded, idx))
        threshold = val
    end
    return threshold
end
function get_forward_reg_incl_excl!(::AbstractMaxValStepwiseRegressionCriteria, value,
                                    excluded, included, threshold)
    val, key = findmax(value)
    idx = findfirst(x -> x == key, excluded)
    if val > threshold
        push!(included, popat!(excluded, idx))
        threshold = val
    end
    return threshold
end
function regression(re::StepwiseRegression{<:Union{<:AbstractMinValStepwiseRegressionCriterion,
                                                   <:AbstractMaxValStepwiseRegressionCriteria},
                                           <:Forward}, x::AbstractVector, F::AbstractMatrix)
    T, N = size(F)
    ovec = range(; start = 1, stop = 1, length = T)
    indices = 1:N
    criterion_func = regression_criterion_func(re.crit)
    threshold = regression_threshold(re.crit)
    included = Vector{eltype(indices)}(undef, 0)
    excluded = collect(indices)
    value = Vector{promote_type(eltype(F), eltype(x))}(undef, N)
    for _ in eachindex(x)
        ni = length(excluded)
        for i in excluded
            factors = copy(included)
            push!(factors, i)
            f1 = [ovec view(F, :, factors)]
            fri = fit(re.target, f1, x)
            value[i] = criterion_func(fri)
        end
        if isempty(value)
            break
        end
        threshold = get_forward_reg_incl_excl!(re.crit, value, excluded, included,
                                               threshold)
        if ni == length(excluded)
            break
        end
    end
    return included
end
function regression(re::StepwiseRegression{<:PValue, <:Backward}, x::AbstractVector,
                    F::AbstractMatrix)
    ovec = range(; start = 1, stop = 1, length = length(x))
    fri = fit(re.target, [ovec F], x)
    included = 1:size(F, 2)
    indices = 1:size(F, 2)
    excluded = Vector{eltype(indices)}(undef, 0)
    pvals = coeftable(fri).cols[4][2:end]
    val = maximum(pvals)
    while val > re.crit.threshold
        factors = setdiff(indices, excluded)
        included = factors
        if isempty(factors)
            break
        end
        f1 = [ovec view(F, :, factors)]
        fri = fit(re.target, f1, x)
        pvals = coeftable(fri).cols[4][2:end]
        val, idx = findmax(pvals)
        push!(excluded, factors[idx])
    end
    add_best_asset_after_failure_pval!(re.target, included, F, x)
    return included
end
function get_backward_reg_incl!(::AbstractMinValStepwiseRegressionCriterion, value,
                                included, threshold)
    val, idx = findmin(value)
    if val < threshold
        i = findfirst(x -> x == idx, included)
        popat!(included, i)
        threshold = val
    end
    return threshold
end
function get_backward_reg_incl!(::AbstractMaxValStepwiseRegressionCriteria, value, included,
                                threshold)
    val, idx = findmax(value)
    if val > threshold
        i = findfirst(x -> x == idx, included)
        popat!(included, i)
        threshold = val
    end
    return threshold
end
function regression(re::StepwiseRegression{<:Union{<:AbstractMinValStepwiseRegressionCriterion,
                                                   <:AbstractMaxValStepwiseRegressionCriteria},
                                           <:Backward}, x::AbstractVector,
                    F::AbstractMatrix)
    T, N = size(F)
    ovec = range(; start = 1, stop = 1, length = T)
    included = collect(1:N)
    fri = fit(re.target, [ovec F], x)
    criterion_func = regression_criterion_func(re.crit)
    threshold = criterion_func(fri)
    value = Vector{promote_type(eltype(F), eltype(x))}(undef, N)
    for _ in eachindex(x)
        ni = length(included)
        for (i, factor) in pairs(included)
            factors = copy(included)
            popat!(factors, i)
            if !isempty(factors)
                f1 = [ovec view(F, :, factors)]
            else
                f1 = reshape(ovec, :, 1)
            end
            fri = fit(re.target, f1, x)
            value[factor] = criterion_func(fri)
        end
        if isempty(value)
            break
        end
        threshold = get_backward_reg_incl!(re.crit, value, included, threshold)
        if ni == length(included)
            break
        end
    end
    return included
end
function regression(re::StepwiseRegression, X::AbstractMatrix, F::AbstractMatrix)
    features = 1:size(F, 2)
    cols = size(F, 2) + 1
    N, rows = size(X)
    ovec = range(; start = 1, stop = 1, length = N)
    loadings = zeros(promote_type(eltype(F), eltype(X)), rows, cols)
    for i in axes(loadings, 1)
        included = regression(re, view(X, :, i), F)
        x1 = !isempty(included) ? [ovec view(F, :, included)] : reshape(ovec, :, 1)
        fri = fit(re.target, x1, view(X, :, i))
        params = coef(fri)
        loadings[i, 1] = params[1]
        if isempty(included)
            continue
        end
        idx = [findfirst(x -> x == i, features) + 1 for i in included]
        loadings[i, idx] .= params[2:end]
    end
    return Regression(; b = view(loadings, :, 1), M = view(loadings, :, 2:cols))
end

export AIC, AICC, BIC, RSquared, AdjustedRSquared, PValue, Forward, Backward,
       StepwiseRegression
