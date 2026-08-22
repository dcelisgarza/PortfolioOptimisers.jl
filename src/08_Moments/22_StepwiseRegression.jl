"""
$(DocStringExtensions.TYPEDEF)

Selects factors by the statistical significance of their coefficients.

A candidate model is admissible when **every** one of its coefficient p-values is at or below `t`, the intercept excluded. This is the only criterion that reads the fitted coefficients rather than one model-wide score, so it is not a [`MinMaxValStepwiseRegressionCriterion`](@ref) and it takes its own stepwise methods.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PValue(;
        t::Number = 0.05
    ) -> PValue

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:t])

# Examples

```jldoctest
julia> PValue()
PValue
  t ┴ Float64: 0.05
```

# Details

  - Under either algorithm the selection never returns an empty factor set. When no factor clears `t`, [`add_best_factor_after_pval_failure!`](@ref) adds the single best one and warns.

# Related

  - [`AbstractStepwiseRegressionCriterion`](@ref)
  - [`StepwiseRegression`](@ref)
  - [`add_best_factor_after_pval_failure!`](@ref)

# References

  - $(ref_dict[:efroymson1960])
"""
@concrete struct PValue <: AbstractStepwiseRegressionCriterion
    """
    $(field_dict[:t])
    """
    t
    function PValue(t::Number)
        assert_unit_interval(t, :t)
        return new{typeof(t)}(t)
    end
end
function PValue(; t::Number = 0.05)::PValue
    return PValue(t)
end
"""
$(DocStringExtensions.TYPEDEF)

Grows the factor set from empty, adding the factor that most improves the criterion.

At each step the algorithm fits one model per excluded factor, keeps the best of them, and stops when no addition improves on the score of the set it already holds. Under a [`MinMaxValStepwiseRegressionCriterion`](@ref) the starting score is [`regression_threshold`](@ref), the worst value that criterion can take, so the first addition always happens; under [`PValue`](@ref) the step instead admits a candidate whose p-values all clear `t`.

# Related

  - [`AbstractStepwiseRegressionAlgorithm`](@ref)
  - [`BackwardElimination`](@ref)
  - [`StepwiseRegression`](@ref)
  - [`regression_threshold`](@ref)

# References

  - $(ref_dict[:efroymson1960])
"""
struct ForwardSelection <: AbstractStepwiseRegressionAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Shrinks the factor set from full, removing the factor whose removal most improves the criterion.

At each step the algorithm fits one model per included factor, each with that factor dropped, and removes the factor whose reduced model scores best. It stops when no removal improves on the score of the set it already holds. The starting score is the score of the **full** model, not [`regression_threshold`](@ref); under [`PValue`](@ref) the step instead drops the factor with the largest p-value while any exceeds `t`.

# Related

  - [`AbstractStepwiseRegressionAlgorithm`](@ref)
  - [`ForwardSelection`](@ref)
  - [`StepwiseRegression`](@ref)

# References

  - $(ref_dict[:efroymson1960])
"""
struct BackwardElimination <: AbstractStepwiseRegressionAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Estimates a loadings matrix by selecting a factor subset per asset, one factor at a time.

`crit` scores a candidate model, `alg` sets the direction the factor set moves in, and `tgt` fits it. Each asset gets its own subset, so a factor a given asset never selected carries an exact zero in that row of the loadings matrix.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    StepwiseRegression(;
        crit::Union{Symbol, MinMaxValStepwiseRegressionCriterion,
                    AbstractStepwiseRegressionCriterion} = PValue(),
        alg::AbstractStepwiseRegressionAlgorithm = ForwardSelection(),
        tgt::AbstractRegressionTarget = LinearModel()
    ) -> StepwiseRegression

Keywords correspond to the struct's fields.

## Validation

  - If `crit` is a `Symbol`, `crit in STEPWISE_REGRESSION_CRITERIA`. The constructor stores `Val(crit)`.
  - If `crit` is `Val(:adjr2)`, `tgt` is a [`GeneralisedLinearModel`](@ref) and `tgt.variant` is set, `tgt.variant in ADJUSTED_PSEUDO_R2_VARIANTS`.
  - If `tgt.kwargs` carries a `weights` entry, it must be an `ObsWeights` and, when it is a vector, `!isempty(tgt.kwargs.weights)`.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `tgt`: Recursively updated via [`factory`](@ref).

# Examples

```jldoctest
julia> StepwiseRegression()
StepwiseRegression
  crit ┼ PValue
       │   t ┴ Float64: 0.05
   alg ┼ ForwardSelection()
   tgt ┼ LinearModel
       │   kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractStepwiseRegressionCriterion`](@ref)
  - [`AbstractStepwiseRegressionAlgorithm`](@ref)
  - [`AbstractRegressionTarget`](@ref)
  - [`Regression`](@ref)
  - [`DimensionReductionRegression`](@ref)
  - [`factory`](@ref)

# References

  - $(ref_dict[:efroymson1960])
  - $(ref_dict[:hocking1976])
"""
@propagatable @concrete struct StepwiseRegression <: AbstractRegressionEstimator
    """
    $(field_dict[:crit])
    """
    crit
    """
    $(field_dict[:realg])
    """
    alg
    """
    $(field_dict[:retgt])
    """
    @fprop tgt
    function StepwiseRegression(crit::Union{MinMaxValStepwiseRegressionCriterion,
                                            AbstractStepwiseRegressionCriterion},
                                alg::AbstractStepwiseRegressionAlgorithm,
                                tgt::AbstractRegressionTarget)
        if isa(crit, Val{:adjr2}) &&
           isa(tgt, GeneralisedLinearModel) &&
           !isnothing(tgt.variant)
            @argcheck(tgt.variant in ADJUSTED_PSEUDO_R2_VARIANTS,
                      "The :adjr2 criterion reads StatsAPI.adjr2, which accepts a variant in $ADJUSTED_PSEUDO_R2_VARIANTS. Got\ntgt.variant => $(tgt.variant)")
        end
        if haskey(tgt.kwargs, :weights)
            @argcheck(isa(tgt.kwargs.weights, ObsWeights),
                      ArgumentError("tgt.kwargs.weights must be a vector of observation weights, one element per observation, of type ObsWeights = Union{<:DynamicAbstractWeights, <:StatsBase.AbstractWeights}. Got\ntgt.kwargs.weights => $(typeof(tgt.kwargs.weights))"))
            if isa(tgt.kwargs.weights, AbstractVector)
                @argcheck(!isempty(tgt.kwargs.weights), IsEmptyError)
            end
        end
        return new{typeof(crit), typeof(alg), typeof(tgt)}(crit, alg, tgt)
    end
end
function StepwiseRegression(;
                            crit::Union{Symbol, MinMaxValStepwiseRegressionCriterion,
                                        AbstractStepwiseRegressionCriterion} = PValue(),
                            alg::AbstractStepwiseRegressionAlgorithm = ForwardSelection(),
                            tgt::AbstractRegressionTarget = LinearModel())::StepwiseRegression
    if isa(crit, Symbol)
        @argcheck(crit in STEPWISE_REGRESSION_CRITERIA,
                  "crit must be one of $STEPWISE_REGRESSION_CRITERIA. Got\ncrit => $crit")
        crit = Val(crit)
    end
    return StepwiseRegression(crit, alg, tgt)
end
"""
    add_best_factor_after_pval_failure!(tgt::AbstractRegressionTarget,
                                        included::VecInt, F::MatNum,
                                        x::VecNum)

Helper for stepwise regression: add the "best" asset by p-value if no variables are included.

This function is used in stepwise regression routines when no variables meet the p-value threshold for inclusion. It scans all excluded variables, fits a regression for each, and selects the variable with the lowest p-value (even if above the threshold). The index of this variable is pushed to `included`, ensuring the model always includes at least one variable.

# Arguments

  - `tgt`: Regression target type (e.g., `LinearModel()`).
  - `included`: Indices of currently included variables (modified in-place).
  - `F`: Factor matrix (observations × factors).
  - `x`: Response vector.

# Returns

  - `nothing`: Modifies `included` in-place.

# Details

If `included` is not empty, the function does nothing. Otherwise, it evaluates each excluded variable by fitting a regression and extracting its p-value, then adds the variable with the lowest p-value to `included`. A warning is issued if no variable meets the threshold.

# Related

  - [`StepwiseRegression`](@ref)
  - [`regression`](@ref)
"""
function add_best_factor_after_pval_failure!(tgt::AbstractRegressionTarget,
                                             included::VecInt, F::MatNum, x::VecNum)
    if !isempty(included)
        return nothing
    end
    T, N = size(F)
    ovec = range(one(eltype(F)), one(eltype(F)); length = T)
    best_pval = typemax(eltype(x))
    new_factor = 0
    for i in 1:N
        factors = [included; i]
        f1 = [ovec view(F, :, factors)]
        fri = StatsAPI.fit(tgt, f1, x)
        new_pvals = StatsAPI.coeftable(fri).cols[4][2:end]
        idx = searchsortedfirst(factors, i)
        test_pval = new_pvals[idx]
        if best_pval > test_pval
            best_pval = test_pval
            new_factor = i
        end
    end
    @warn("No factor with p-value lower than threshold. Best we can do is factor $new_factor, with p-value $best_pval.")
    push!(included, new_factor)
    return nothing
end
"""
    _regression(re::StepwiseRegression{<:PValue, <:ForwardSelection}, x::VecNum,
               F::MatNum)

Perform forward stepwise regression using a p-value criterion.

This method implements forward selection for stepwise regression, where variables (columns of `F`) are added to the model one at a time based on their statistical significance (p-value), starting from an empty model. At each step, the variable with the lowest p-value (and all p-values below the specified threshold) is added. The process continues until no remaining variable meets the p-value threshold. If no variable meets the threshold at any step, the variable with the lowest p-value is included to ensure at least one variable is selected.

# Arguments

  - `re`: Stepwise regression estimator with a `PValue` criterion and `ForwardSelection` algorithm.
  - `x`: Response vector.
  - `F`: Feature matrix (observations × variables).

# Returns

  - `included::Vector{Int}`: Indices of variables selected by the forward stepwise regression.

# Details

  - Starts with no variables included in the regression.
  - Tries to add variables one at a time based on p-value, stopping when no further variables can be added under the threshold.
  - If no variables are included at the end, the variable with the lowest p-value is added (see [`add_best_factor_after_pval_failure!`](@ref)).

# Related

  - [`StepwiseRegression`](@ref)
  - [`PValue`](@ref)
  - [`ForwardSelection`](@ref)
  - [`add_best_factor_after_pval_failure!`](@ref)
"""
function _regression(re::StepwiseRegression{<:PValue, <:ForwardSelection}, x::VecNum,
                     F::MatNum)
    ovec = range(one(eltype(F)), one(eltype(F)); length = length(x))
    indices = 1:size(F, 2)
    included = Vector{eltype(indices)}(undef, 0)
    pvals = nothing
    val = zero(promote_type(eltype(F), eltype(x)))
    while val <= re.crit.t
        excluded = setdiff(indices, included)
        best_pval = typemax(eltype(x))
        new_factor = 0
        for i in excluded
            factors = [included; i]
            f1 = [ovec view(F, :, factors)]
            fri = StatsAPI.fit(re.tgt, f1, x)
            new_pvals = StatsAPI.coeftable(fri).cols[4][2:end]
            idx = findfirst(x -> x == i, factors)
            test_pval = new_pvals[idx]
            if best_pval > test_pval && maximum(new_pvals) <= re.crit.t
                best_pval = test_pval
                new_factor = i
                pvals = copy(new_pvals)
            end
        end
        iszero(new_factor) ? break : push!(included, new_factor)
        if !isempty(pvals)
            val = maximum(pvals)
        end
    end
    add_best_factor_after_pval_failure!(re.tgt, included, F, x)
    return included
end
"""
    get_forward_reg_incl_excl!(::MinValStepwiseRegressionCriterion,
                               value::VecNum, excluded::VecInt,
                               included::VecInt, t::Number)

Helper for forward stepwise regression with minimum-value criteria (e.g., a p-value, `:aic`).

This function updates the `included` and `excluded` variable sets in forward stepwise regression when the selection criterion is minimized (such as `:aic` or `:bic`). It finds the variable with the lowest value, and if this value is less than the current `t`, moves it from `excluded` to `included` and updates the threshold.

# Arguments

  - `::MinValStepwiseRegressionCriterion`: Stepwise regression criterion type where lower values are better.
  - `value`: Vector of criterion values for each variable.
  - `excluded`: Indices of currently excluded variables (modified in-place).
  - `included`: Indices of currently included variables (modified in-place).
  - `t`: Current threshold value for inclusion.

# Returns

  - `t::Number`: Updated threshold value after inclusion (if any).

# Details

  - Finds the minimum of `value`.
  - Searches the **whole** `value` vector, not only the entries of `excluded`. An entry of an already-included variable holds the score that selected it, and every one of those equals or exceeds the current `t`, so a value strictly beyond `t` can only belong to an excluded variable. `searchsortedfirst` is therefore safe, and the answer is the best excluded variable.
  - If this value is less than `t`, moves the variable from `excluded` to `included` and updates `t`.
  - If no variable meets the criterion, the sets remain unchanged and the threshold is not updated.

# Related

  - [`StepwiseRegression`](@ref)
  - [`MinValStepwiseRegressionCriterion`](@ref)
  - [`regression`](@ref)
"""
function get_forward_reg_incl_excl!(::MinValStepwiseRegressionCriterion, value::VecNum,
                                    excluded::VecInt, included::VecInt, t::Number)
    val, idx = findmin(value)
    if val < t
        i = searchsortedfirst(excluded, idx)
        push!(included, popat!(excluded, i))
        t = val
    end
    return t
end
"""
    get_forward_reg_incl_excl!(::MaxValStepwiseRegressionCriterion,
                               value::VecNum, excluded::VecInt,
                               included::VecInt, t::Number)

Helper for forward stepwise regression with maximum-value criteria (e.g., R²).

This function updates the `included` and `excluded` variable sets in forward stepwise regression when the selection criterion is maximized (such as R²). It finds the variable with the highest value, and if this value is greater than the current `t`, moves it from `excluded` to `included` and updates the threshold.

# Arguments

  - `::MaxValStepwiseRegressionCriterion`: Stepwise regression criterion type where higher values are better.
  - `value`: Vector of criterion values for each variable.
  - `excluded`: Indices of currently excluded variables (modified in-place).
  - `included`: Indices of currently included variables (modified in-place).
  - `t`: Current threshold value for inclusion.

# Returns

  - `t::Number`: Updated threshold value after inclusion (if any).

# Details

  - Finds the maximum of `value`.
  - Searches the **whole** `value` vector, not only the entries of `excluded`. An entry of an already-included variable holds the score that selected it, and every one of those equals or exceeds the current `t`, so a value strictly beyond `t` can only belong to an excluded variable. `searchsortedfirst` is therefore safe, and the answer is the best excluded variable.
  - If this value is greater than `t`, moves the variable from `excluded` to `included` and updates `t`.
  - If no variable meets the criterion, the sets remain unchanged and the threshold is not updated.

# Related

  - [`StepwiseRegression`](@ref)
  - [`MaxValStepwiseRegressionCriterion`](@ref)
  - [`regression`](@ref)
"""
function get_forward_reg_incl_excl!(::MaxValStepwiseRegressionCriterion, value::VecNum,
                                    excluded::VecInt, included::VecInt, t::Number)
    val, idx = findmax(value)
    if val > t
        i = searchsortedfirst(excluded, idx)
        push!(included, popat!(excluded, i))
        t = val
    end
    return t
end
"""
    _regression(re::StepwiseRegression{<:MinMaxValStepwiseRegressionCriterion,
                                      <:ForwardSelection}, x::VecNum, F::MatNum)

Perform forward stepwise regression using a general criterion (minimization or maximization).

This method implements forward selection for stepwise regression, where variables (columns of `F`) are added to the model one at a time based on a user-specified criterion. The criterion can be either minimized (e.g., a p-value, `:aic`) or maximized (e.g., `:r2`). At each step, the variable with the best criterion value (lowest for minimization, highest for maximization) is considered for inclusion if it improves upon the current threshold. The process continues until no remaining variable meets the criterion for inclusion.

# Arguments

  - `re`: Stepwise regression estimator with a minimization or maximization criterion and `ForwardSelection` algorithm.
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
  - [`MinValStepwiseRegressionCriterion`](@ref)
  - [`MaxValStepwiseRegressionCriterion`](@ref)
  - [`ForwardSelection`](@ref)
  - [`get_forward_reg_incl_excl!`](@ref)
"""
function _regression(re::StepwiseRegression{<:MinMaxValStepwiseRegressionCriterion,
                                            <:ForwardSelection}, x::VecNum, F::MatNum)
    T, N = size(F)
    ovec = range(one(eltype(F)), one(eltype(F)); length = T)
    indices = 1:N
    criterion_func = regression_criterion_func(re.crit, re.tgt)
    t = regression_threshold(re.crit)
    included = Vector{eltype(indices)}(undef, 0)
    excluded = collect(indices)
    value = fill(ifelse(isa(re.crit, MinValStepwiseRegressionCriterion),
                        typemax(promote_type(eltype(F), eltype(x))),
                        typemin(promote_type(eltype(F), eltype(x)))), N)
    for _ in eachindex(x)
        ni = length(excluded)
        for i in excluded
            factors = copy(included)
            push!(factors, i)
            f1 = [ovec view(F, :, factors)]
            fri = StatsAPI.fit(re.tgt, f1, x)
            value[i] = criterion_func(fri)
        end
        t = get_forward_reg_incl_excl!(re.crit, value, excluded, included, t)
        if ni == length(excluded)
            break
        end
    end
    return included
end
"""
    _regression(re::StepwiseRegression{<:PValue, <:BackwardElimination}, x::VecNum,
               F::MatNum)

Perform backward stepwise regression using a p-value criterion.

This method implements backward elimination for stepwise regression, where all variables (columns of `F`) are initially included in the model. At each step, the variable with the highest p-value is considered for removal if its p-value exceeds the specified threshold. The process continues until all remaining variables have p-values below the threshold. If all variables are excluded, the variable with the lowest p-value is included to ensure at least one variable is selected.

# Arguments

  - `re`: Stepwise regression estimator with a `PValue` criterion and `BackwardElimination` algorithm.
  - `x`: Response vector.
  - `F`: Feature matrix (observations × variables).

# Returns

  - `included::Vector{Int}`: Indices of variables selected by the backward stepwise regression.

# Details

  - Starts with all variables included in the regression.
  - Removes variables one at a time based on whichever has the largest p-value, stopping when the p-value falls under the threshold.
  - If no variables are included at the end, the variable with the lowest p-value is added (see [`add_best_factor_after_pval_failure!`](@ref)).

# Related

  - [`StepwiseRegression`](@ref)
  - [`PValue`](@ref)
  - [`BackwardElimination`](@ref)
  - [`add_best_factor_after_pval_failure!`](@ref)
"""
function _regression(re::StepwiseRegression{<:PValue, <:BackwardElimination}, x::VecNum,
                     F::MatNum)
    ovec = range(one(eltype(F)), one(eltype(F)); length = length(x))
    fri = StatsAPI.fit(re.tgt, [ovec F], x)
    included = 1:size(F, 2)
    indices = 1:size(F, 2)
    excluded = Vector{eltype(indices)}(undef, 0)
    pvals = StatsAPI.coeftable(fri).cols[4][2:end]
    val = maximum(pvals)
    while val > re.crit.t
        included = setdiff(indices, excluded)
        if isempty(included)
            break
        end
        f1 = [ovec view(F, :, included)]
        fri = StatsAPI.fit(re.tgt, f1, x)
        pvals = StatsAPI.coeftable(fri).cols[4][2:end]
        val, idx = findmax(pvals)
        push!(excluded, included[idx])
    end
    add_best_factor_after_pval_failure!(re.tgt, included, F, x)
    return included
end
"""
    get_backward_reg_incl!(::MinValStepwiseRegressionCriterion, value::VecNum,
                           included::VecInt, t::Number)

Helper for backward stepwise regression with minimum-value criteria (e.g., a p-value, `:aic`).

This function updates the `included` variable set in backward stepwise regression when the selection criterion is minimized (such as `:aic` or `:bic`). Each entry of `value` is the score of the model that **omits** that variable, so the lowest entry names the removal that helps most. It finds that variable, and if its value is less than the current `t`, removes it from `included` and updates the threshold.

# Arguments

  - `::MinValStepwiseRegressionCriterion`: Stepwise regression criterion type where lower values are better.
  - `value`: Vector of criterion values for each variable.
  - `included`: Indices of currently included variables (modified in-place).
  - `t`: Current threshold value for exclusion.

# Returns

  - `t::Number`: Updated threshold value after exclusion (if any).

# Details

  - Finds the minimum of `value`, which is the best model reachable by removing one variable.
  - Searches the **whole** `value` vector, not only the entries of `included`. An entry of an already-removed variable holds the score that removed it, and every one of those equals or exceeds the current `t`, so a value strictly beyond `t` can only belong to an included variable. `searchsortedfirst` is therefore safe, and the answer is the best included variable.
  - If this value is less than `t`, removes the variable from `included` and updates `t`.
  - If no variable meets the criterion, the set remains unchanged and the threshold is not updated.

# Related

  - [`StepwiseRegression`](@ref)
  - [`regression`](@ref)
"""
function get_backward_reg_incl!(::MinValStepwiseRegressionCriterion, value::VecNum,
                                included::VecInt, t::Number)
    val, idx = findmin(value)
    if val < t
        i = searchsortedfirst(included, idx)
        popat!(included, i)
        t = val
    end
    return t
end
"""
    get_backward_reg_incl!(::MaxValStepwiseRegressionCriterion, value::VecNum,
                           included::VecInt, t::Number)

Helper for backward stepwise regression with maximum-value criteria (e.g., R²).

This function updates the `included` variable set in backward stepwise regression when the selection criterion is maximized (such as R²). Each entry of `value` is the score of the model that **omits** that variable, so the highest entry names the removal that helps most. It finds that variable, and if its value is greater than the current `t`, removes it from `included` and updates the threshold.

# Arguments

  - `::MaxValStepwiseRegressionCriterion`: Stepwise regression criterion type where higher values are better.
  - `value`: Vector of criterion values for each variable.
  - `included`: Indices of currently included variables (modified in-place).
  - `t`: Current threshold value for exclusion.

# Returns

  - `t::Number`: Updated threshold value after exclusion (if any).

# Details

  - Finds the maximum of `value`, which is the best model reachable by removing one variable.
  - Searches the **whole** `value` vector, not only the entries of `included`. An entry of an already-removed variable holds the score that removed it, and every one of those equals or exceeds the current `t`, so a value strictly beyond `t` can only belong to an included variable. `searchsortedfirst` is therefore safe, and the answer is the best included variable.
  - If this value is greater than `t`, removes the variable from `included` and updates `t`.
  - If no variable meets the criterion, the set remains unchanged and the threshold is not updated.

# Related

  - [`StepwiseRegression`](@ref)
  - [`regression`](@ref)
"""
function get_backward_reg_incl!(::MaxValStepwiseRegressionCriterion, value::VecNum,
                                included::VecInt, t::Number)
    val, idx = findmax(value)
    if val > t
        i = searchsortedfirst(included, idx)
        popat!(included, i)
        t = val
    end
    return t
end
"""
    _regression(re::StepwiseRegression{<:MinMaxValStepwiseRegressionCriterion,
                                      <:BackwardElimination}, x::VecNum, F::MatNum)

Perform backward stepwise regression using a general criterion (minimization or maximization).

This method implements backward elimination for stepwise regression, where all variables (columns of `F`) are initially included in the model. At each step it fits one model per included variable, each with that variable dropped, and removes the variable whose reduced model scores best, provided that score improves on the current threshold. The process continues until no removal improves the score.

# Arguments

  - `re`: Stepwise regression estimator with a minimization or maximization criterion and `BackwardElimination` algorithm.
  - `x`: Response vector.
  - `F`: Feature matrix (observations × variables).

# Returns

  - `included::Vector{Int}`: Indices of variables selected by the backward stepwise regression.

# Details

  - Starts with all variables included, and with the threshold set to the criterion value of the full model.
  - `value[j]` is the score of the model that **omits** `j`, so the reading of "best" is inverted with respect to the forward direction: for a minimisation criterion the code removes the factor with the **lowest** value, and for a maximisation criterion the one with the **highest**. On a 200×5 sample the five reduced-model `:aic` values were `[493.57, 271.05, 327.82, 274.62, 271.34]` against a full-model `:aic` score of `272.25`; the code removed factor 2, the lowest, and factor 1 — the highest, and the response's true signal factor — was the one kept.
  - The criterion function is determined by the estimator's `crit` field.
  - The process stops on the first iteration that removes nothing, or when no variables remain.
  - Supports both minimization and maximization criteria via dispatch.

# Related

  - [`StepwiseRegression`](@ref)
  - [`MinValStepwiseRegressionCriterion`](@ref)
  - [`MaxValStepwiseRegressionCriterion`](@ref)
  - [`BackwardElimination`](@ref)
  - [`get_backward_reg_incl!`](@ref)
"""
function _regression(re::StepwiseRegression{<:MinMaxValStepwiseRegressionCriterion,
                                            <:BackwardElimination}, x::VecNum, F::MatNum)
    T, N = size(F)
    ovec = range(one(eltype(F)), one(eltype(F)); length = T)
    included = collect(1:N)
    fri = StatsAPI.fit(re.tgt, [ovec F], x)
    criterion_func = regression_criterion_func(re.crit, re.tgt)
    t = criterion_func(fri)
    value = fill(ifelse(isa(re.crit, MinValStepwiseRegressionCriterion),
                        typemax(promote_type(eltype(F), eltype(x))),
                        typemin(promote_type(eltype(F), eltype(x)))), N)
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
            fri = StatsAPI.fit(re.tgt, f1, x)
            value[factor] = criterion_func(fri)
        end
        t = get_backward_reg_incl!(re.crit, value, included, t)
        if ni == length(included)
            break
        end
    end
    return included
end
"""
    regression(re::StepwiseRegression, X::MatNum, F::MatNum)

Apply stepwise regression to each column of a response matrix.

This method fits a stepwise regression model (as specified by `re`) to each column of the response matrix `X`, using the factor matrix `F` as predictors. For each response vector (column of `X`), the function selects variables via stepwise regression, fits the final model, and stores the estimated intercept and coefficients in the result.

# Arguments

  - `re`: Stepwise regression estimator specifying the criterion, algorithm, and regression target.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor returns matrix (observations × factors).

# Returns

  - `reg::Regression`: A regression result object containing:

      + `b`: Vector of intercepts for each asset.
      + `M`: Matrix of coefficients for each asset and factor (zeros for excluded factors).

# Details

  - For each column in `X`, stepwise regression is performed using the specified criterion and algorithm.
  - Only the factors selected by the stepwise procedure are included in the final model for each response.
  - The output `Regression` object contains the intercepts and a coefficient matrix with zeros for factors not selected for each response.
  - `L` is left **unset**. The regression runs in the original factor basis, so `reg.L` reads back as `reg.M` through the result's `swap(L, M)` property rule, and `size(reg.L, 2)` is the number of columns of `F`.
  - `M` and `b` are views into one dense buffer, not freshly allocated matrices.

# Related

  - [`StepwiseRegression`](@ref)
  - [`regression`](@ref)
  - [`Regression`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 4.1, Equations 4.2-4.3.
"""
function regression(re::StepwiseRegression, X::MatNum, F::MatNum)
    factors = 1:size(F, 2)
    cols = size(F, 2) + 1
    N, rows = size(X)
    ovec = range(one(eltype(F)), one(eltype(F)); length = N)
    rr = zeros(promote_type(eltype(F), eltype(X)), rows, cols)
    for i in axes(rr, 1)
        included = _regression(re, view(X, :, i), F)
        x1 = !isempty(included) ? [ovec view(F, :, included)] : reshape(ovec, :, 1)
        fri = StatsAPI.fit(re.tgt, x1, view(X, :, i))
        params = StatsAPI.coef(fri)
        rr[i, 1] = params[1]
        idx = [searchsortedfirst(factors, i) + 1 for i in included]
        rr[i, idx] = params[2:end]
    end
    return Regression(; b = view(rr, :, 1), M = view(rr, :, 2:cols))
end

export PValue, ForwardSelection, BackwardElimination, StepwiseRegression
