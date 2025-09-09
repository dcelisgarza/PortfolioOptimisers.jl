"""
    struct PValue{T1} <: AbstractStepwiseRegressionCriterion
        threshold::T1
    end

Stepwise regression criterion based on p-value thresholding.

`PValue` is used as a criterion for stepwise regression algorithms, where variables are included or excluded from the model based on their statistical significance (p-value). The `threshold` field specifies the maximum p-value for a variable to be considered significant and included in the model.

# Fields

  - `threshold`: The p-value threshold for variable inclusion.

# Constructor

    PValue(; threshold::Real = 0.05)

Keyword arguments correspond to the fields above.

## Validation

  - `0 < threshold < 1`.

# Examples

```jldoctest
julia> PValue()
PValue
  threshold | Float64: 0.05
```

# Related

  - [`AbstractStepwiseRegressionCriterion`](@ref)
  - [`StepwiseRegression`](@ref)
"""
struct PValue{T1} <: AbstractStepwiseRegressionCriterion
    threshold::T1
end
function PValue(; threshold::Real = 0.05)
    @argcheck(zero(threshold) < threshold < one(threshold))
    return PValue(threshold)
end

"""
    struct Forward <: AbstractStepwiseRegressionAlgorithm end

Stepwise regression algorithm: forward selection.

`Forward` specifies the forward selection strategy for stepwise regression. In forward selection, variables are added to the model one at a time based on a criterion (such as p-value or information criterion), starting from an empty model and including the variable that most improves the model at each step. The process continues until no further improvement is possible or a stopping criterion is met.

# Related

  - [`AbstractStepwiseRegressionAlgorithm`](@ref)
  - [`Backward`](@ref)
  - [`StepwiseRegression`](@ref)
"""
struct Forward <: AbstractStepwiseRegressionAlgorithm end

"""
    struct Backward <: AbstractStepwiseRegressionAlgorithm end

Stepwise regression algorithm: backward elimination.

`Backward` specifies the backward elimination strategy for stepwise regression. In backward elimination, all candidate variables are initially included in the model, and variables are removed one at a time based on a criterion (such as p-value or information criterion). At each step, the variable whose removal most improves the model (or least degrades it) is excluded, until no further improvement is possible or a stopping criterion is met.

# Related

  - [`AbstractStepwiseRegressionAlgorithm`](@ref)
  - [`Forward`](@ref)
  - [`StepwiseRegression`](@ref)
"""
struct Backward <: AbstractStepwiseRegressionAlgorithm end

"""
    struct StepwiseRegression{T1, T2, T3} <: AbstractRegressionEstimator
        crit::T1
        alg::T2
        target::T3
    end

Estimator for stepwise regression-based moment estimation.

`StepwiseRegression` is a flexible estimator type for performing stepwise regression, supporting both forward selection and backward elimination strategies. It allows users to specify the criterion for variable selection (such as p-value, AIC, BIC, or R²), the stepwise algorithm, and the regression target (e.g., linear or generalised linear models).

# Fields

  - `crit`: Criterion for variable selection.
  - `alg`: Stepwise algorithm.
  - `target`: Regression target type.

# Constructor

    StepwiseRegression(; crit::AbstractStepwiseRegressionCriterion = PValue(),
                         alg::AbstractStepwiseRegressionAlgorithm = Forward(),
                         target::AbstractRegressionTarget = LinearModel())

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> StepwiseRegression()
StepwiseRegression
    crit | PValue
         |   threshold | Float64: 0.05
     alg | Forward()
  target | LinearModel
         |   kwargs | @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractStepwiseRegressionCriterion`](@ref)
  - [`AbstractStepwiseRegressionAlgorithm`](@ref)
  - [`AbstractRegressionTarget`](@ref)
"""
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

"""
    add_best_feature_after_pval_failure!(target::AbstractRegressionTarget,
                                      included::AbstractVector,
                                      F::AbstractMatrix,
                                      x::AbstractVector)

Helper for stepwise regression: add the "best" asset by p-value if no variables are included.

This function is used in stepwise regression routines when no variables meet the p-value threshold for inclusion. It scans all excluded variables, fits a regression for each, and selects the variable with the lowest p-value (even if above the threshold). The index of this variable is pushed to `included`, ensuring the model always includes at least one variable.

# Arguments

  - `target`: Regression target type (e.g., `LinearModel()`).
  - `included`: Indices of currently included variables (modified in-place).
  - `F`: Factor matrix (features × observations).
  - `x`: Response vector.

# Returns

  - `nothing`: Modifies `included` in-place.

# Details

If `included` is not empty, the function does nothing. Otherwise, it evaluates each excluded variable by fitting a regression and extracting its p-value, then adds the variable with the lowest p-value to `included`. A warning is issued if no variable meets the threshold.

# Related

  - [`StepwiseRegression`](@ref)
  - [`regression`](@ref)
"""
function add_best_feature_after_pval_failure!(target::AbstractRegressionTarget,
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

"""
    regression(re::StepwiseRegression{<:PValue, <:Forward}, x::AbstractVector, F::AbstractMatrix)

Perform forward stepwise regression using a p-value criterion.

This method implements forward selection for stepwise regression, where variables (columns of `F`) are added to the model one at a time based on their statistical significance (p-value), starting from an empty model. At each step, the variable with the lowest p-value (and all p-values below the specified threshold) is added. The process continues until no remaining variable meets the p-value threshold. If no variable meets the threshold at any step, the variable with the lowest p-value is included to ensure at least one variable is selected.

# Arguments

  - `re`: Stepwise regression estimator with a `PValue` criterion and `Forward` algorithm.
  - `x`: Response vector.
  - `F`: Feature matrix (observations × variables).

# Returns

  - `included::Vector{Int}`: Indices of variables selected by the forward stepwise regression.

# Details

  - Starts with no variables included in the regression.
  - Tries to add variables one at a time based on p-value, stopping when no further variables can be added under the threshold.
  - If no variables are included at the end, the variable with the lowest p-value is added (see [`add_best_feature_after_pval_failure!`](@ref)).

# Related

  - [`StepwiseRegression`](@ref)

  - [`PValue`](@ref)
  - [`Forward`](@ref)
  - [`add_best_feature_after_pval_failure!`](@ref)
"""
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
    add_best_feature_after_pval_failure!(re.target, included, F, x)
    return included
end

"""
    get_forward_reg_incl_excl!(::AbstractMinValStepwiseRegressionCriterion,
                               value::AbstractVector,
                               excluded::AbstractVector,
                               included::AbstractVector,
                               threshold::Real)

Helper for forward stepwise regression with minimum-value criteria (e.g., p-value, AIC).

This function updates the `included` and `excluded` variable sets in forward stepwise regression when the selection criterion is minimized (such as p-value or AIC). It finds the variable among `excluded` with the lowest value, and if this value is less than the current `threshold`, moves it from `excluded` to `included` and updates the threshold.

# Arguments

  - `::AbstractMinValStepwiseRegressionCriterion`: Stepwise regression criterion type where lower values are better.
  - `value`: Vector of criterion values for each variable.
  - `excluded`: Indices of currently excluded variables (modified in-place).
  - `included`: Indices of currently included variables (modified in-place).
  - `threshold`: Current threshold value for inclusion.

# Returns

  - `threshold::Real`: Updated threshold value after inclusion (if any).

# Details

  - Finds the variable in `excluded` with the minimum value in `value`.
  - If this value is less than `threshold`, moves the variable from `excluded` to `included` and updates `threshold`.
  - If no variable meets the criterion, the sets remain unchanged and the threshold is not updated.

# Related

  - [`StepwiseRegression`](@ref)
  - [`AbstractMinValStepwiseRegressionCriterion`](@ref)
  - [`regression`](@ref)
"""
function get_forward_reg_incl_excl!(::AbstractMinValStepwiseRegressionCriterion,
                                    value::AbstractVector, excluded::AbstractVector,
                                    included::AbstractVector, threshold::Real)
    val, key = findmin(value)
    idx = findfirst(x -> x == key, excluded)
    if val < threshold
        push!(included, popat!(excluded, idx))
        threshold = val
    end
    return threshold
end

"""
    get_forward_reg_incl_excl!(::AbstractMaxValStepwiseRegressionCriteria,
                               value::AbstractVector,
                               excluded::AbstractVector,
                               included::AbstractVector,
                               threshold::Real)

Helper for forward stepwise regression with maximum-value criteria (e.g., R²).

This function updates the `included` and `excluded` variable sets in forward stepwise regression when the selection criterion is maximized (such as R²). It finds the variable among `excluded` with the highest value, and if this value is greater than the current `threshold`, moves it from `excluded` to `included` and updates the threshold.

# Arguments

  - `::AbstractMaxValStepwiseRegressionCriteria`: Stepwise regression criterion type where higher values are better.
  - `value`: Vector of criterion values for each variable.
  - `excluded`: Indices of currently excluded variables (modified in-place).
  - `included`: Indices of currently included variables (modified in-place).
  - `threshold`: Current threshold value for inclusion.

# Returns

  - `threshold::Real`: Updated threshold value after inclusion (if any).

# Details

  - Finds the variable in `excluded` with the maximum value in `value`.
  - If this value is greater than `threshold`, moves the variable from `excluded` to `included` and updates `threshold`.
  - If no variable meets the criterion, the sets remain unchanged and the threshold is not updated.

# Related

  - [`StepwiseRegression`](@ref)
  - [`AbstractMaxValStepwiseRegressionCriteria`](@ref)
  - [`regression`](@ref)
"""
function get_forward_reg_incl_excl!(::AbstractMaxValStepwiseRegressionCriteria,
                                    value::AbstractVector, excluded::AbstractVector,
                                    included::AbstractVector, threshold::Real)
    val, key = findmax(value)
    idx = findfirst(x -> x == key, excluded)
    if val > threshold
        push!(included, popat!(excluded, idx))
        threshold = val
    end
    return threshold
end

"""
    regression(re::StepwiseRegression{<:Union{<:AbstractMinValStepwiseRegressionCriterion,
                                              <:AbstractMaxValStepwiseRegressionCriteria},
                                      <:Forward},
               x::AbstractVector, F::AbstractMatrix)

Perform forward stepwise regression using a general criterion (minimization or maximization).

This method implements forward selection for stepwise regression, where variables (columns of `F`) are added to the model one at a time based on a user-specified criterion. The criterion can be either minimized (e.g., p-value, AIC) or maximized (e.g., R²). At each step, the variable with the best criterion value (lowest for minimization, highest for maximization) is considered for inclusion if it improves upon the current threshold. The process continues until no remaining variable meets the criterion for inclusion.

# Arguments

  - `re`: Stepwise regression estimator with a minimization or maximization criterion and `Forward` algorithm.
  - `x`: Response vector.
  - `F`: Feature matrix (observations × variables).

# Returns

  - `included::Vector{Int}`: Indices of variables selected by the forward stepwise regression.

# Details

  - At each iteration, the method fits a regression model for each excluded variable, computes the criterion value, and adds the variable with the best value if it improves upon the current threshold.
  - The process stops when no further variables can be added under the criterion.
  - The criterion function and threshold are determined by the estimator's `crit` field.
  - Supports both minimization and maximization criteria via dispatch.

# Related

  - [`StepwiseRegression`](@ref)
  - [`AbstractMinValStepwiseRegressionCriterion`](@ref)
  - [`AbstractMaxValStepwiseRegressionCriteria`](@ref)
  - [`Forward`](@ref)
  - [`get_forward_reg_incl_excl!`](@ref)
"""
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

"""
    regression(re::StepwiseRegression{<:PValue, <:Backward}, x::AbstractVector, F::AbstractMatrix)

Perform backward stepwise regression using a p-value criterion.

This method implements backward elimination for stepwise regression, where all variables (columns of `F`) are initially included in the model. At each step, the variable with the highest p-value is considered for removal if its p-value exceeds the specified threshold. The process continues until all remaining variables have p-values below the threshold. If all variables are excluded, the variable with the lowest p-value is included to ensure at least one variable is selected.

# Arguments

  - `re`: Stepwise regression estimator with a `PValue` criterion and `Backward` algorithm.
  - `x`: Response vector.
  - `F`: Feature matrix (observations × variables).

# Returns

  - `included::Vector{Int}`: Indices of variables selected by the backward stepwise regression.

# Details

  - Starts with all variables included in the regression.
  - Removes variables one at a time based on whichever has the largest p-value, stopping when the p-value falls under the threshold.
  - If no variables are included at the end, the variable with the lowest p-value is added (see [`add_best_feature_after_pval_failure!`](@ref)).

# Related

  - [`StepwiseRegression`](@ref)
  - [`PValue`](@ref)
  - [`Backward`](@ref)
  - [`add_best_feature_after_pval_failure!`](@ref)
"""
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
    add_best_feature_after_pval_failure!(re.target, included, F, x)
    return included
end

"""
    get_backward_reg_incl!(::AbstractMinValStepwiseRegressionCriterion,
                           value::AbstractVector,
                           included::AbstractVector,
                           threshold::Real)

Helper for backward stepwise regression with minimum-value criteria (e.g., p-value, AIC).

This function updates the `included` variable set in backward stepwise regression when the selection criterion is minimized (such as p-value or AIC). It finds the variable among `included` with the lowest value, and if this value is less than the current `threshold`, removes it from `included` and updates the threshold.

# Arguments

  - `::AbstractMinValStepwiseRegressionCriterion`: Stepwise regression criterion type where lower values are better.
  - `value`: Vector of criterion values for each variable.
  - `included`: Indices of currently included variables (modified in-place).
  - `threshold`: Current threshold value for exclusion.

# Returns

  - `threshold::Real`: Updated threshold value after exclusion (if any).

# Details

  - Finds the variable in `included` with the minimum value in `value`.
  - If this value is less than `threshold`, removes the variable from `included` and updates `threshold`.
  - If no variable meets the criterion, the set remains unchanged and the threshold is not updated.

# Related

  - [`StepwiseRegression`](@ref)
  - [`regression`](@ref)
"""
function get_backward_reg_incl!(::AbstractMinValStepwiseRegressionCriterion,
                                value::AbstractVector, included::AbstractVector,
                                threshold::Real)
    val, idx = findmin(value)
    if val < threshold
        i = findfirst(x -> x == idx, included)
        popat!(included, i)
        threshold = val
    end
    return threshold
end

"""
    get_backward_reg_incl!(::AbstractMaxValStepwiseRegressionCriteria,
                           value::AbstractVector,
                           included::AbstractVector,
                           threshold::Real)

Helper for backward stepwise regression with maximum-value criteria (e.g., R²).

This function updates the `included` variable set in backward stepwise regression when the selection criterion is maximized (such as R²). It finds the variable among `included` with the highest value, and if this value is greater than the current `threshold`, removes it from `included` and updates the threshold.

# Arguments

  - `::AbstractMaxValStepwiseRegressionCriteria`: Stepwise regression criterion type where higher values are better.
  - `value`: Vector of criterion values for each variable.
  - `included`: Indices of currently included variables (modified in-place).
  - `threshold`: Current threshold value for exclusion.

# Returns

  - `threshold::Real`: Updated threshold value after exclusion (if any).

# Details

  - Finds the variable in `included` with the maximum value in `value`.
  - If this value is greater than `threshold`, removes the variable from `included` and updates `threshold`.
  - If no variable meets the criterion, the set remains unchanged and the threshold is not updated.

# Related

  - [`StepwiseRegression`](@ref)
  - [`regression`](@ref)
"""
function get_backward_reg_incl!(::AbstractMaxValStepwiseRegressionCriteria,
                                value::AbstractVector, included::AbstractVector,
                                threshold::Real)
    val, idx = findmax(value)
    if val > threshold
        i = findfirst(x -> x == idx, included)
        popat!(included, i)
        threshold = val
    end
    return threshold
end

"""
    regression(re::StepwiseRegression{<:Union{<:AbstractMinValStepwiseRegressionCriterion,
                                              <:AbstractMaxValStepwiseRegressionCriteria},
                                      <:Backward},
               x::AbstractVector, F::AbstractMatrix)

Perform backward stepwise regression using a general criterion (minimization or maximization).

This method implements backward elimination for stepwise regression, where all variables (columns of `F`) are initially included in the model. At each step, the variable with the worst criterion value (highest for minimization, lowest for maximization) is considered for removal if it does not meet the specified threshold. The process continues until all remaining variables satisfy the criterion for inclusion.

# Arguments

  - `re`: Stepwise regression estimator with a minimization or maximization criterion and `Backward` algorithm.
  - `x`: Response vector.
  - `F`: Feature matrix (observations × variables).

# Returns

  - `included::Vector{Int}`: Indices of variables selected by the backward stepwise regression.

# Details

  - Starts with all variables included.
  - At each iteration, fits a regression model for all included variables, computes the criterion value for each, and removes the variable with the worst value if it does not meet the threshold.
  - The criterion function and threshold are determined by the estimator's `crit` field.
  - The process stops when all included variables satisfy the criterion or no variables remain.
  - Supports both minimization and maximization criteria via dispatch.

# Related

  - [`StepwiseRegression`](@ref)
  - [`AbstractMinValStepwiseRegressionCriterion`](@ref)
  - [`AbstractMaxValStepwiseRegressionCriteria`](@ref)
  - [`Backward`](@ref)
  - [`get_backward_reg_incl!`](@ref)
"""
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

"""
    regression(re::StepwiseRegression, X::AbstractMatrix, F::AbstractMatrix)

Apply stepwise regression to each column of a response matrix.

This method fits a stepwise regression model (as specified by `re`) to each column of the response matrix `X`, using the feature matrix `F` as predictors. For each response vector (column of `X`), the function selects variables via stepwise regression, fits the final model, and stores the estimated intercept and coefficients in the result.

# Arguments

  - `re`: Stepwise regression estimator specifying the criterion, algorithm, and regression target.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor returns matrix (observations × factors or features).

# Returns

  - `Regression`: A regression result object containing:

      + `b`: Vector of intercepts for each asset.
      + `M`: Matrix of coefficients for each asset and feature (zeros for excluded features).

# Details

  - For each column in `X`, stepwise regression is performed using the specified criterion and algorithm.
  - Only the features selected by the stepwise procedure are included in the final model for each response.
  - The output `Regression` object contains the intercepts and a coefficient matrix with zeros for features not selected for each response.

# Related

  - [`StepwiseRegression`](@ref)
  - [`regression`](@ref)
  - [`Regression`](@ref)
"""
function regression(re::StepwiseRegression, X::AbstractMatrix, F::AbstractMatrix)
    features = 1:size(F, 2)
    cols = size(F, 2) + 1
    N, rows = size(X)
    ovec = range(; start = 1, stop = 1, length = N)
    rr = zeros(promote_type(eltype(F), eltype(X)), rows, cols)
    for i in axes(rr, 1)
        included = regression(re, view(X, :, i), F)
        x1 = !isempty(included) ? [ovec view(F, :, included)] : reshape(ovec, :, 1)
        fri = fit(re.target, x1, view(X, :, i))
        params = coef(fri)
        rr[i, 1] = params[1]
        if isempty(included)
            continue
        end
        idx = [findfirst(x -> x == i, features) + 1 for i in included]
        rr[i, idx] .= params[2:end]
    end
    return Regression(; b = view(rr, :, 1), M = view(rr, :, 2:cols))
end

export PValue, Forward, Backward, StepwiseRegression
