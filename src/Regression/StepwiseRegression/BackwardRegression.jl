struct BackwardRegression{T1 <: StepwiseRegressionCriteria} <: StepwiseRegression
    criterion::T1
end
function BackwardRegression(; criterion::StepwiseRegressionCriteria = PVal())
    return BackwardRegression{typeof(criterion)}(criterion)
end
function _regression(re::BackwardRegression{<:PVal}, F::AbstractMatrix, x::AbstractVector)
    ovec = range(; start = 1, stop = 1, length = length(x))
    fit_result = GLM.lm([ovec F], x)
    included = 1:size(F, 2)
    indices = 1:size(F, 2)
    excluded = Vector{eltype(indices)}(undef, 0)
    pvals = coeftable(fit_result).cols[4][2:end]
    val = maximum(pvals)
    while val > re.criterion.threshold
        factors = setdiff(indices, excluded)
        included = factors
        if isempty(factors)
            break
        end
        f1 = [ovec view(F, :, factors)]
        fit_result = GLM.lm(f1, x)
        pvals = coeftable(fit_result).cols[4][2:end]
        val, idx = findmax(pvals)
        push!(excluded, factors[idx])
    end
    add_best_asset_after_failure_pval!(included, F, x)
    return included
end
function get_backward_reg_incl!(::MinValStepwiseRegressionCriteria, value, included,
                                threshold)
    val, idx = findmin(value)
    if val < threshold
        i = findfirst(x -> x == idx, included)
        popat!(included, i)
        threshold = val
    end
    return threshold
end
function get_backward_reg_incl!(::MaxValStepwiseRegressionCriteria, value, included,
                                threshold)
    val, idx = findmax(value)
    if val > threshold
        i = findfirst(x -> x == idx, included)
        popat!(included, i)
        threshold = val
    end
    return threshold
end
function _regression(re::BackwardRegression{<:Union{<:MinValStepwiseRegressionCriteria,
                                                    <:MaxValStepwiseRegressionCriteria}},
                     F::AbstractMatrix, x::AbstractVector)
    T, N = size(F)
    ovec = range(; start = 1, stop = 1, length = T)
    included = collect(1:N)
    fit_result = GLM.lm([ovec F], x)
    criterion_func = regression_criterion_func(re.criterion)
    threshold = criterion_func(fit_result)
    value = Vector{promote_type(eltype(F), eltype(x))}(undef, N)
    for _ ∈ eachindex(x)
        ni = length(included)
        for (i, factor) ∈ pairs(included)
            factors = copy(included)
            popat!(factors, i)
            if !isempty(factors)
                f1 = [ovec view(F, :, factors)]
            else
                f1 = reshape(ovec, :, 1)
            end
            fit_result = GLM.lm(f1, x)
            value[factor] = criterion_func(fit_result)
        end
        if isempty(value)
            break
        end
        threshold = get_backward_reg_incl!(re.criterion, value, included, threshold)
        if ni == length(included)
            break
        end
    end
    return included
end

export BackwardRegression
