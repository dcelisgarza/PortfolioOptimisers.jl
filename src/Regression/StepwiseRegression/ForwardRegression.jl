struct ForwardRegression{T1 <: StepwiseRegressionCriteria} <: StepwiseRegression
    criterion::T1
end
function ForwardRegression(; criterion::StepwiseRegressionCriteria = PValue())
    return ForwardRegression{typeof(criterion)}(criterion)
end
function _regression(re::ForwardRegression{<:PValue}, x::AbstractVector, F::AbstractMatrix)
    ovec = range(; start = 1, stop = 1, length = length(x))
    indices = 1:size(F, 2)
    included = Vector{eltype(indices)}(undef, 0)
    pvals = Vector{promote_type(eltype(F), eltype(x))}(undef, 0)
    val = zero(promote_type(eltype(F), eltype(x)))
    while val <= re.criterion.threshold
        excluded = setdiff(indices, included)
        best_pval = typemax(eltype(x))
        new_feature = 0
        for i ∈ excluded
            factors = [included; i]
            f1 = [ovec view(F, :, factors)]
            fit_result = GLM.lm(f1, x)
            new_pvals = coeftable(fit_result).cols[4][2:end]
            idx = findfirst(x -> x == i, factors)
            test_pval = new_pvals[idx]
            if best_pval > test_pval && maximum(new_pvals) <= re.criterion.threshold
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
    add_best_asset_after_failure_pval!(included, F, x)
    return included
end
function get_forward_reg_incl_excl!(::MinValStepwiseRegressionCriteria, value, excluded,
                                    included, threshold)
    val, key = findmin(value)
    idx = findfirst(x -> x == key, excluded)
    if val < threshold
        push!(included, popat!(excluded, idx))
        threshold = val
    end
    return threshold
end
function get_forward_reg_incl_excl!(::MaxValStepwiseRegressionCriteria, value, excluded,
                                    included, threshold)
    val, key = findmax(value)
    idx = findfirst(x -> x == key, excluded)
    if val > threshold
        push!(included, popat!(excluded, idx))
        threshold = val
    end
    return threshold
end
function _regression(re::ForwardRegression{<:Union{<:MinValStepwiseRegressionCriteria,
                                                   <:MaxValStepwiseRegressionCriteria}},
                     x::AbstractVector, F::AbstractMatrix)
    T, N = size(F)
    ovec = range(; start = 1, stop = 1, length = T)
    indices = 1:N
    criterion_func = regression_criterion_func(re.criterion)
    threshold = regression_threshold(re.criterion)
    included = Vector{eltype(indices)}(undef, 0)
    excluded = collect(indices)
    value = Vector{promote_type(eltype(F), eltype(x))}(undef, N)
    for _ ∈ eachindex(x)
        ni = length(excluded)
        for i ∈ excluded
            factors = copy(included)
            push!(factors, i)
            f1 = [ovec view(F, :, factors)]
            fit_result = GLM.lm(f1, x)
            value[i] = criterion_func(fit_result)
        end
        if isempty(value)
            break
        end
        threshold = get_forward_reg_incl_excl!(re.criterion, value, excluded, included,
                                               threshold)
        if ni == length(excluded)
            break
        end
    end
    return included
end

export ForwardRegression
