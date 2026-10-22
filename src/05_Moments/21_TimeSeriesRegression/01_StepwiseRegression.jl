"""
$(DocStringExtensions.TYPEDEF)

Selects factors by the statistical significance of their coefficients.

A candidate model is admissible when each p-value of its factor coefficients is at or below `t`. The test does not read the p-value of the intercept. This criterion reads the fitted coefficients and not one score of the whole model, so it is not a [`MinMaxValStepwiseRegressionCriterion`](@ref), and it has its own stepwise methods. The selection is never empty under either algorithm. When no factor clears `t`, [`add_best_factor_after_pval_failure!`](@ref) adds the factor whose one-factor model has the smallest p-value, and warns.

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

# Related

  - [`AbstractStepwiseRegressionCriterion`](@ref)
  - [`StepwiseRegression`](@ref)
  - [`add_best_factor_after_pval_failure!`](@ref)

# References

  - $(ref_dict[:efroymson1960]) Pages 191-203. $(ref_dict[:no_original_source])
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

At each step the algorithm fits one model per excluded factor and keeps the best of them. It stops when no addition improves on the score of the set that it holds. Under a [`MinMaxValStepwiseRegressionCriterion`](@ref) the starting score is [`regression_threshold`](@ref), the worst value that the criterion can take, so the first addition always happens. Under [`PValue`](@ref) the step admits a candidate whose p-values all clear `t`.

The selection is never empty under either criterion. Under a [`MinMaxValStepwiseRegressionCriterion`](@ref), [`BackwardElimination`](@ref) can return an empty selection, and this tag cannot. The two methods that run the steps state them: `_regression(::StepwiseRegression{<:PValue, <:ForwardSelection}, ::VecNum, ::MatNum)` and `_regression(::StepwiseRegression{<:MinMaxValStepwiseRegressionCriterion, <:ForwardSelection}, ::VecNum, ::MatNum)`.

# Related

  - [`AbstractStepwiseRegressionAlgorithm`](@ref)
  - [`BackwardElimination`](@ref)
  - [`StepwiseRegression`](@ref)
  - [`regression_threshold`](@ref)

# References

  - $(ref_dict[:efroymson1960]) Pages 191-203. $(ref_dict[:no_original_source])
"""
struct ForwardSelection <: AbstractStepwiseRegressionAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Shrinks the factor set from full, removing the factor whose removal most improves the criterion.

At each step the algorithm fits one model per included factor, each model without that factor, and removes the factor whose reduced model scores best. It stops when no removal improves on the score of the set that it holds. The starting score is the score of the full model, not [`regression_threshold`](@ref). Under [`PValue`](@ref) the step removes the factor with the largest p-value while a p-value exceeds `t`.

Under a [`MinMaxValStepwiseRegressionCriterion`](@ref) the selection can be empty. When each removal improves the score, the search removes every factor. The asset then gets an intercept-only model, and its row of the loadings matrix is all zeros. [`stepwise_factor_set`](@ref) warns and names the asset when the search selects no factor. The two methods that run the steps state them: `_regression(::StepwiseRegression{<:PValue, <:BackwardElimination}, ::VecNum, ::MatNum)` and `_regression(::StepwiseRegression{<:MinMaxValStepwiseRegressionCriterion, <:BackwardElimination}, ::VecNum, ::MatNum)`.

# Related

  - [`AbstractStepwiseRegressionAlgorithm`](@ref)
  - [`ForwardSelection`](@ref)
  - [`StepwiseRegression`](@ref)

# References

  - $(ref_dict[:efroymson1960]) Pages 191-203. $(ref_dict[:no_original_source])
"""
struct BackwardElimination <: AbstractStepwiseRegressionAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Estimates a loadings matrix by selecting a factor subset per asset, one factor at a time.

`crit` scores a candidate model, `alg` sets the direction in which the factor set moves, and `tgt` fits each model. Each asset gets its own subset, so a factor that an asset does not select has an exact zero in the row of that asset in the loadings matrix.

The search reads every observation, so a fit over more observations can choose another factor set. `included` fixes the factor set of each asset, and the fit then runs no search for that asset. `choice` states whether the online step of a prior that fits this regression writes the factor set of the first fit into `included`. A batch fit gives the same answer under both rules.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    StepwiseRegression(;
        crit::Union{Symbol, MinMaxValStepwiseRegressionCriterion,
                    AbstractStepwiseRegressionCriterion} = PValue(),
        alg::AbstractStepwiseRegressionAlgorithm = ForwardSelection(),
        tgt::AbstractRegressionTarget = LinearModel(),
        choice::AbstractChoiceRule = BatchChoice(),
        included::Option{<:AbstractVector{<:Option{<:AbstractVector{<:Integer}}}} = nothing,
        x_src::Option{<:AbstractReturnsSource} = nothing
    ) -> StepwiseRegression

Keywords correspond to the struct's fields.

## Validation

  - If `crit` is a `Symbol`, `crit in STEPWISE_REGRESSION_CRITERIA`. The constructor stores `Val(crit)`.
  - If `crit` is `Val(:adjr2)`, `tgt` is a [`GeneralisedLinearModel`](@ref) and `tgt.variant` is set, `tgt.variant in ADJUSTED_PSEUDO_R2_VARIANTS`.
  - If `tgt` is a [`LinearModel`](@ref) or a [`GeneralisedLinearModel`](@ref) and `tgt.kwargs` holds a `weights` entry, the entry must be an `ObsWeights`. When it is a vector, `!isempty(tgt.kwargs.weights)`. The constructor does not check a target that the caller defines. The fit reads the weights of that target through its own `factory` and `StatsAPI.fit` methods, and the `# Interfaces` section of [`AbstractRegressionTarget`](@ref) states those methods.
  - The rules of [`assert_stepwise_included`](@ref) for `included` alone. An entry can be empty, and the asset then gets an intercept-only model.
  - `x_src in ALL_RETURNS_SOURCES`, as [`assert_returns_source`](@ref) checks. A regression reader admits every source.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `tgt`: Recursively updated via [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `included`: The factor sets of the selected assets.

# Examples

```jldoctest
julia> StepwiseRegression()
StepwiseRegression
      crit ┼ PValue
           │   t ┴ Float64: 0.05
       alg ┼ ForwardSelection()
       tgt ┼ LinearModel
           │   kwargs ┴ @NamedTuple{}: NamedTuple()
    choice ┼ BatchChoice()
  included ┼ nothing
     x_src ┴ nothing
```

# Related

  - [`AbstractStepwiseRegressionCriterion`](@ref)
  - [`AbstractStepwiseRegressionAlgorithm`](@ref)
  - [`AbstractRegressionTarget`](@ref)
  - [`Regression`](@ref)
  - [`DimensionReductionRegression`](@ref)
  - [`PinnedChoice`](@ref)
  - [`AbstractReturnsSource`](@ref)
  - [`resolve_factor_regression`](@ref)
  - [`factory`](@ref)

# References

  - $(ref_dict[:efroymson1960]) Pages 191-203. $(ref_dict[:no_original_source])
"""
@propagatable @concrete struct StepwiseRegression <: AbstractTimeSeriesRegressionEstimator
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
    """
    Choice Rule of the factor set of each asset. Under [`BatchChoice`](@ref), the default, each fit searches again over every observation. Under [`PinnedChoice`](@ref), the online step of a prior that fits this regression writes the factor set of the first fit of each asset into `included` of the estimator that it returns. The two agree in a batch fit. An entropy pooling prior refits the prior under the posterior weights. It holds the regression of the fit that its views read, so its refits agree too.
    """
    choice
    """
    Factor set of each asset, one entry per column of the returns, or `nothing`. An entry holds the indices of the factors that the fit of that asset uses, in the order in which the search added them. The fit then runs no search for that asset. An entry of `nothing` makes the fit search for that asset.
    """
    @vprop included
    """
    $(field_dict[:x_src_re])
    """
    x_src
    function StepwiseRegression(crit::Union{MinMaxValStepwiseRegressionCriterion,
                                            AbstractStepwiseRegressionCriterion},
                                alg::AbstractStepwiseRegressionAlgorithm,
                                tgt::AbstractRegressionTarget, choice::AbstractChoiceRule,
                                included::Option{<:AbstractVector{<:Option{<:AbstractVector{<:Integer}}}},
                                x_src::Option{<:AbstractReturnsSource})
        if isa(crit, Val{:adjr2}) &&
           isa(tgt, GeneralisedLinearModel) &&
           !isnothing(tgt.variant)
            @argcheck(tgt.variant in ADJUSTED_PSEUDO_R2_VARIANTS,
                      "The :adjr2 criterion reads StatsAPI.adjr2, which accepts a variant in $ADJUSTED_PSEUDO_R2_VARIANTS. Got\ntgt.variant => $(tgt.variant)")
        end
        if isa(tgt, Union{LinearModel, GeneralisedLinearModel}) &&
           haskey(tgt.kwargs, :weights)
            @argcheck(isa(tgt.kwargs.weights, ObsWeights),
                      ArgumentError("tgt.kwargs.weights must be a vector of observation weights, one element per observation, of type ObsWeights = Union{<:DynamicAbstractWeights, <:StatsBase.AbstractWeights}. Got\ntgt.kwargs.weights => $(typeof(tgt.kwargs.weights))"))
            if isa(tgt.kwargs.weights, AbstractVector)
                @argcheck(!isempty(tgt.kwargs.weights), IsEmptyError)
            end
        end
        assert_stepwise_included(included)
        return new{typeof(crit), typeof(alg), typeof(tgt), typeof(choice), typeof(included),
                   typeof(x_src)}(crit, alg, tgt, choice, included, x_src)
    end
end
function StepwiseRegression(;
                            crit::Union{Symbol, MinMaxValStepwiseRegressionCriterion,
                                        AbstractStepwiseRegressionCriterion} = PValue(),
                            alg::AbstractStepwiseRegressionAlgorithm = ForwardSelection(),
                            tgt::AbstractRegressionTarget = LinearModel(),
                            choice::AbstractChoiceRule = BatchChoice(),
                            included::Option{<:AbstractVector{<:Option{<:AbstractVector{<:Integer}}}} = nothing,
                            x_src::Option{<:AbstractReturnsSource} = nothing)::StepwiseRegression
    assert_returns_source(x_src, ALL_RETURNS_SOURCES, :StepwiseRegression)
    if isa(crit, Symbol)
        @argcheck(crit in STEPWISE_REGRESSION_CRITERIA,
                  "crit must be one of $STEPWISE_REGRESSION_CRITERIA. Got\ncrit => $crit")
        crit = Val(crit)
    end
    return StepwiseRegression(crit, alg, tgt, choice, included, x_src)
end
"""
    add_best_factor_after_pval_failure!(tgt::AbstractRegressionTarget,
                                        included::VecInt, F::MatNum,
                                        x::VecNum)

Adds the factor of smallest p-value when a p-value search selected none.

Both [`PValue`](@ref) methods of `_regression` call it last, so a p-value search never returns an empty factor set. It does nothing when `included` already holds a factor. It warns each time that it adds a factor, because that factor failed the threshold in the search.

# Algorithm

 1. Return at once when `included` is not empty.
 2. For each factor `i` of `F`, fit `tgt` to an intercept column and column `i`, and read the p-value of column `i`. Call it `test_pval`.
 3. Keep the smallest `test_pval` of step 2 as `best_pval`, and its factor as `new_factor`.
 4. Warn, and name `new_factor` and `best_pval`.
 5. Push `new_factor` onto `included`.

# Arguments

  - `tgt`: Regression target that fits each candidate model.
  - `included`: Indices of the factors that the search selected. The function writes into it.
  - $(arg_dict[:F])
  - `x`: Response vector, `observations × 1`.

# Returns

  - `nothing`. `included` gains one index when it is empty, and it does not change otherwise.

# Related

  - [`StepwiseRegression`](@ref)
  - [`PValue`](@ref)
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
    @warn("No factor with p-value lower than threshold. Best we can do is factor $new_factor, with p-value $best_pval.",
          maxlog = log_limit(:warn), _id = @log_id())
    push!(included, new_factor)
    return nothing
end
"""
    _regression(re::StepwiseRegression{<:PValue, <:ForwardSelection}, x::VecNum,
               F::MatNum)

Grows a factor set from empty, and admits a model only when its p-values all clear the threshold.

A candidate is admissible only when each factor coefficient of the candidate model is significant. So the method rejects a factor that is significant alone when it makes a factor of the set insignificant.

# Algorithm

 1. Set `included` empty and `val` to zero, so the loop runs at least once.
 2. While `val` does not exceed `re.crit.t`, do steps 3 to 6.
 3. Take the factors that `included` does not hold, and call them `excluded`.
 4. For each `i` of `excluded`, fit `re.tgt` to an intercept column and the columns `[included; i]`, and read the p-value of column `i`. Call it `test_pval`. Keep `i` as `new_factor` when `test_pval` is the smallest so far and no p-value of that candidate model exceeds `re.crit.t`. Keep the p-values of that model as `pvals`.
 5. Stop the loop when step 4 kept no factor. Otherwise push `new_factor` onto `included`.
 6. Set `val` to the largest entry of `pvals`.
 7. Call [`add_best_factor_after_pval_failure!`](@ref), which acts only when `included` is still empty.

# Arguments

  - `re`: Stepwise regression estimator with a [`PValue`](@ref) criterion and the [`ForwardSelection`](@ref) algorithm.
  - `x`: Response vector, `observations × 1`.
  - $(arg_dict[:F])

# Returns

  - `included::Vector{Int}`: Indices of the selected factors, in the order in which the loop added them, so the vector is not sorted. It holds at least one index.

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
                pvals = new_pvals
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
    get_forward_reg_incl_excl!(crit::MinMaxValStepwiseRegressionCriterion,
                               value::VecNum, excluded::VecInt,
                               included::VecInt, t::Number)

Moves the best excluded factor into `included` when it improves the criterion.

[`regression_polarity`](@ref) states the direction. A minimised criterion uses `findmin` and `<`, and a maximised criterion uses `findmax` and `>`. The search reads the whole `value` vector, and it still finds the best excluded factor. The entry of an included factor holds the score of the step that added it. `t` holds the score of the last step, and each step improves on the step before it, so no entry of an included factor improves on `t`. Thus a score that improves on `t` belongs to an excluded factor, and `searchsortedfirst` finds its index in `excluded`.

# Algorithm

 1. Read `best` and `improves` from [`regression_polarity`](@ref).
 2. Find the best entry of `value` with `best`. Call it `val`, and its index `idx`.
 3. Return `t` unchanged when `val` does not improve on `t`.
 4. Otherwise find the position of `idx` in `excluded`, move that entry to the end of `included`, and set `t` to `val`.

# Arguments

  - `crit`: Stepwise regression criterion. Its polarity states which direction is better.
  - `value`: Criterion score of each factor. Entry `i` is the score of the model that adds factor `i` to `included`.
  - `excluded`: Indices of the factors outside the model, in ascending order. The function writes into it.
  - `included`: Indices of the factors inside the model, in the order in which the search added them. The function writes into it.
  - $(arg_dict[:t])

# Returns

  - `t::Number`: The score of the factor that moved, or the input `t` when no factor moved.

# Related

  - [`StepwiseRegression`](@ref)
  - [`regression_polarity`](@ref)
  - [`MinValStepwiseRegressionCriterion`](@ref)
  - [`MaxValStepwiseRegressionCriterion`](@ref)
  - [`get_backward_reg_incl!`](@ref)
  - [`regression`](@ref)
"""
function get_forward_reg_incl_excl!(crit::MinMaxValStepwiseRegressionCriterion,
                                    value::VecNum, excluded::VecInt, included::VecInt,
                                    t::Number)
    (; best, improves) = regression_polarity(crit)
    val, idx = best(value)
    if improves(val, t)
        i = searchsortedfirst(excluded, idx)
        push!(included, popat!(excluded, i))
        t = val
    end
    return t
end
"""
    _regression(re::StepwiseRegression{<:MinMaxValStepwiseRegressionCriterion,
                                      <:ForwardSelection}, x::VecNum, F::MatNum)

Grows a factor set from empty, adding the factor whose model scores best.

The starting score is the worst value that the criterion can take, so the first addition always happens and the selection is never empty. [`get_forward_reg_incl_excl!`](@ref) reads the direction from [`regression_polarity`](@ref), so one loop works for a minimised and for a maximised criterion.

# Algorithm

 1. Set `included` empty and `excluded` to every factor index, in ascending order.
 2. Read the criterion from `re.crit` and `re.tgt` with [`regression_criterion_func`](@ref), and call it `criterion_func`. Read the starting score from [`regression_threshold`](@ref), and call it `t`.
 3. Fill `value` with that same worst score, one entry per factor.
 4. Do steps 5 to 7 at most once per observation.
 5. For each `i` of `excluded`, fit `re.tgt` to an intercept column and the columns `[included; i]`, and write `criterion_func` of that fit into `value[i]`.
 6. Call [`get_forward_reg_incl_excl!`](@ref), which moves the best factor and returns the new `t`.
 7. Stop when step 6 moved no factor.

# Arguments

  - `re`: Stepwise regression estimator with a [`MinMaxValStepwiseRegressionCriterion`](@ref) criterion and the [`ForwardSelection`](@ref) algorithm.
  - `x`: Response vector, `observations × 1`.
  - $(arg_dict[:F])

# Returns

  - `included::Vector{Int}`: Indices of the selected factors, in the order in which the loop added them, so the vector is not sorted. It holds at least one index.

# Related

  - [`StepwiseRegression`](@ref)
  - [`MinValStepwiseRegressionCriterion`](@ref)
  - [`MaxValStepwiseRegressionCriterion`](@ref)
  - [`ForwardSelection`](@ref)
  - [`get_forward_reg_incl_excl!`](@ref)
  - [`regression_criterion_func`](@ref)
  - [`regression_threshold`](@ref)
"""
function _regression(re::StepwiseRegression{<:MinMaxValStepwiseRegressionCriterion,
                                            <:ForwardSelection}, x::VecNum, F::MatNum)
    T, N = size(F)
    ovec = range(one(eltype(F)), one(eltype(F)); length = T)
    indices = 1:N
    criterion_func = regression_criterion_func(re.crit, re.tgt)
    t = regression_threshold(re.crit)
    included = Vector{eltype(indices)}(undef, 0)
    factors = similar(included)
    excluded = collect(indices)
    value = fill(regression_polarity(re.crit).worst(promote_type(eltype(F), eltype(x))), N)
    for _ in eachindex(x)
        ni = length(excluded)
        for i in excluded
            append!(empty!(factors), included)
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

Shrinks a factor set from full, and removes the factor of largest p-value while a p-value exceeds the threshold.

# Algorithm

 1. Fit `re.tgt` to an intercept column and every column of `F`, read the p-values of the factor coefficients, and set `val` to the largest of them.
 2. Set `included` to every factor index and `excluded` empty.
 3. While `val` exceeds `re.crit.t`, do steps 4 to 6.
 4. Set `included` to the factors that `excluded` does not hold. Stop the loop when `included` is empty.
 5. Fit `re.tgt` to an intercept column and the columns `included`, and read the p-values of the factor coefficients. Call them `pvals`.
 6. Set `val` to the largest entry of `pvals`, and push the factor of that entry onto `excluded`. Only step 4 reads `excluded`, so the push has an effect only when step 3 runs the loop again. The push of the last iteration does not change the result.
 7. Call [`add_best_factor_after_pval_failure!`](@ref), which acts only when `included` is empty.

# Arguments

  - `re`: Stepwise regression estimator with a [`PValue`](@ref) criterion and the [`BackwardElimination`](@ref) algorithm.
  - `x`: Response vector, `observations × 1`.
  - $(arg_dict[:F])

# Returns

  - `included::VecInt`: Indices of the selected factors, in ascending order. It holds at least one index. It is the `1:size(F, 2)` range itself when the full model already passes the threshold, and a `Vector{Int}` otherwise.

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
    get_backward_reg_incl!(crit::MinMaxValStepwiseRegressionCriterion, value::VecNum,
                           included::VecInt, t::Number)

Removes the best included factor from `included` when its removal improves the criterion.

[`regression_polarity`](@ref) states the direction. A minimised criterion uses `findmin` and `<`, and a maximised criterion uses `findmax` and `>`. The search reads the whole `value` vector, and it still finds the best included factor. The entry of a removed factor holds the score of the step that removed it. `t` holds the score of the last step, and each step improves on the step before it, so no entry of a removed factor improves on `t`. Thus a score that improves on `t` belongs to an included factor, and `searchsortedfirst` finds its index in `included`.

# Algorithm

 1. Read `best` and `improves` from [`regression_polarity`](@ref).
 2. Find the best entry of `value` with `best`. Call it `val`, and its index `idx`. This entry is the best model that one removal can make.
 3. Return `t` unchanged when `val` does not improve on `t`.
 4. Otherwise find the position of `idx` in `included`, remove that entry, and set `t` to `val`.

# Arguments

  - `crit`: Stepwise regression criterion. Its polarity states which direction is better.
  - `value`: Criterion score of each factor. Entry `j` is the score of the model that omits factor `j`, so the best entry names the removal that improves the score most.
  - `included`: Indices of the factors inside the model, in ascending order. The function writes into it.
  - $(arg_dict[:t])

# Returns

  - `t::Number`: The score of the model that the removal leaves, or the input `t` when the function removed no factor.

# Related

  - [`StepwiseRegression`](@ref)
  - [`regression_polarity`](@ref)
  - [`MinValStepwiseRegressionCriterion`](@ref)
  - [`MaxValStepwiseRegressionCriterion`](@ref)
  - [`get_forward_reg_incl_excl!`](@ref)
  - [`regression`](@ref)
"""
function get_backward_reg_incl!(crit::MinMaxValStepwiseRegressionCriterion, value::VecNum,
                                included::VecInt, t::Number)
    (; best, improves) = regression_polarity(crit)
    val, idx = best(value)
    if improves(val, t)
        i = searchsortedfirst(included, idx)
        popat!(included, i)
        t = val
    end
    return t
end
"""
    _regression(re::StepwiseRegression{<:MinMaxValStepwiseRegressionCriterion,
                                      <:BackwardElimination}, x::VecNum, F::MatNum)

Shrinks a factor set from full, removing the factor whose reduced model scores best.

`value[j]` is the score of the model that omits `j`. So "best" has the same meaning as in the forward direction, and not the inverse meaning. Under a minimised criterion the method removes the factor of lowest value, and under a maximised criterion it removes the factor of highest value.

# Algorithm

 1. Set `included` to every factor index, in ascending order.
 2. Read the criterion from `re.crit` and `re.tgt` with [`regression_criterion_func`](@ref), and call it `criterion_func`.
 3. Fit `re.tgt` to an intercept column and every column of `F`, and set `t` to `criterion_func` of that full model.
 4. Fill `value` with the worst score the criterion can take, one entry per factor.
 5. Do steps 6 to 8 at most once per observation.
 6. For each factor of `included`, fit `re.tgt` to an intercept column and the other columns of `included`, and write `criterion_func` of that fit into that factor's entry of `value`. Fit the intercept column alone when `included` holds one factor.
 7. Call [`get_backward_reg_incl!`](@ref), which removes the best factor and returns the new `t`.
 8. Stop when step 7 removed no factor.

# Arguments

  - `re`: Stepwise regression estimator with a [`MinMaxValStepwiseRegressionCriterion`](@ref) criterion and the [`BackwardElimination`](@ref) algorithm.
  - `x`: Response vector, `observations × 1`.
  - $(arg_dict[:F])

# Returns

  - `included::Vector{Int}`: Indices of the selected factors, in ascending order. It can be empty. When each removal improves the score, the search removes every factor. This happens often under an information criterion on a response that the factors do not explain. [`stepwise_factor_set`](@ref) warns and names the asset on an empty return, and [`regression`](@ref) then fits the intercept column alone.

# Related

  - [`StepwiseRegression`](@ref)
  - [`MinValStepwiseRegressionCriterion`](@ref)
  - [`MaxValStepwiseRegressionCriterion`](@ref)
  - [`BackwardElimination`](@ref)
  - [`get_backward_reg_incl!`](@ref)
  - [`regression_criterion_func`](@ref)
"""
function _regression(re::StepwiseRegression{<:MinMaxValStepwiseRegressionCriterion,
                                            <:BackwardElimination}, x::VecNum, F::MatNum)
    T, N = size(F)
    ovec = range(one(eltype(F)), one(eltype(F)); length = T)
    included = collect(1:N)
    factors = similar(included)
    fri = StatsAPI.fit(re.tgt, [ovec F], x)
    criterion_func = regression_criterion_func(re.crit, re.tgt)
    t = criterion_func(fri)
    value = fill(regression_polarity(re.crit).worst(promote_type(eltype(F), eltype(x))), N)
    for _ in eachindex(x)
        ni = length(included)
        for (i, factor) in pairs(included)
            append!(empty!(factors), included)
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
    assert_stepwise_included(included::Nothing)
    assert_stepwise_included(included::AbstractVector)
    assert_stepwise_included(included::Nothing, X::MatNum, F::MatNum)
    assert_stepwise_included(included::AbstractVector, X::MatNum, F::MatNum)

Checks the factor sets that the `included` field of a [`StepwiseRegression`](@ref) holds.

The one-argument form is the check of the constructor, which sees no data. The three-argument form is the check of the fit. It compares the field with the returns and the factor returns.

# Algorithm

The method that Julia selects is the algorithm. `included` of `nothing` passes both forms.

# Arguments

  - `included`: The factor sets, one entry per asset, or `nothing`.
  - $(arg_dict[:X])
  - $(arg_dict[:F])

# Validation

  - One argument: `!isempty(included)`, or the check raises an `IsEmptyError`. Each entry is `nothing` or a vector of positive factor indices with no repeated index, or the check raises a `DomainError`.
  - Three arguments: `length(included) == size(X, 2)`, and every index of each entry is at most `size(F, 2)`. The check raises a `DimensionMismatch` otherwise.

# Returns

  - `nothing`.

# Related

  - [`StepwiseRegression`](@ref)
  - [`regression`](@ref)
  - [`pin_regression_choice`](@ref)
"""
function assert_stepwise_included(::Nothing)::Nothing
    return nothing
end
function assert_stepwise_included(included::AbstractVector)::Nothing
    @argcheck(!isempty(included), IsEmptyError)
    @argcheck(all(s -> isnothing(s) || (allunique(s) && all(>(0), s)), included),
              DomainError(included,
                          "each entry of included must be nothing, or a vector of positive factor indices with no repeated index. Got\nincluded => $included"))
    return nothing
end
function assert_stepwise_included(::Nothing, ::MatNum, ::MatNum)::Nothing
    return nothing
end
function assert_stepwise_included(included::AbstractVector, X::MatNum, F::MatNum)::Nothing
    @argcheck(length(included) == size(X, 2) &&
              all(s -> isnothing(s) || all(<=(size(F, 2)), s), included),
              DimensionMismatch("included holds one factor set per asset, and each index names a column of F. Got\nlength(included) => $(length(included))\nsize(X, 2) => $(size(X, 2))\nsize(F, 2) => $(size(F, 2))"))
    return nothing
end
"""
    stepwise_factor_set(re::StepwiseRegression, x::VecNum, F::MatNum, i::Integer)

Returns the factor set of one asset, which `re.included` fixes or the stepwise search selects.

# Algorithm

 1. Take entry `i` of `re.included`. Return it when it is not `nothing`.
 2. Otherwise run the stepwise search of `re` on `x`, and call the result `included`. The method also takes this step when `re.included` is `nothing`.
 3. Warn, and name the asset, when `included` is empty. The asset then gets an intercept-only model, and its row of the loadings matrix is all zeros.
 4. Return `included`.

# Arguments

  - `re`: Stepwise regression estimator.
  - `x`: Returns of asset `i`, `observations × 1`.
  - $(arg_dict[:F])
  - `i`: Index of the asset, which names it in the warning.

# Returns

  - `included::AbstractVector{<:Integer}`: Indices of the factors of the asset, in the order of the search or of `re.included`.

# Related

  - [`StepwiseRegression`](@ref)
  - [`regression`](@ref)
"""
function stepwise_factor_set(re::StepwiseRegression, x::VecNum, F::MatNum, i::Integer)
    pinned = isnothing(re.included) ? nothing : re.included[i]
    if !isnothing(pinned)
        return pinned
    end
    included = _regression(re, x, F)
    if isempty(included)
        @warn("Asset $i: the stepwise search selected no factor. The asset gets an intercept-only model, and its row of the loadings matrix is all zeros.",
              maxlog = log_limit(:warn), _id = @log_id())
    end
    return included
end
"""
    regression(re::StepwiseRegression, X::MatNum, F::MatNum)

Runs one stepwise search per asset and assembles the loadings matrix from the fits.

Each asset has its own search. The search of one asset reads only the column of that asset and `F`, so one asset cannot change the factor set of another. An asset whose entry of `re.included` holds a factor set takes that set, and no search runs for it.

# Algorithm

 1. Allocate `rr`, a dense `assets × (factors + 1)` buffer of zeros. A factor that an asset does not select keeps its zero.
 2. For each asset `i`, do steps 3 to 6.
 3. Take the factor set of asset `i` with [`stepwise_factor_set`](@ref), and call it `included`.
 4. Fit `re.tgt` to an intercept column and the columns `included` of `F`, and read its coefficients. Call them `params`. Fit the intercept column alone when `included` is empty.
 5. Write `params[1]` into `rr[i, 1]`, and the remaining coefficients into the columns of `rr` that `included` names, in the order of `included`.
 6. Write the number of observations less the number of columns of the fit of step 4 into `edof[i]`. This is the number of degrees of freedom that the fit leaves in the residuals of the asset.
 7. Build a [`Regression`](@ref) from the first column of `rr`, its remaining columns, and `edof`.

# Arguments

  - `re`: Stepwise regression estimator that gives the criterion, the algorithm and the regression target.
  - $(arg_dict[:X])
  - $(arg_dict[:F])

# Validation

  - The rules of [`assert_stepwise_included`](@ref) for `re.included`, `X` and `F`.

# Returns

  - `reg::Regression`: Regression result that holds:

      + `b`: Intercept of each asset, a view of the first column of `rr`.
      + `M`: Coefficient of each asset and factor, a view of the remaining columns of `rr`. A factor that the asset does not use has an exact zero. A whole row is zero when the asset uses no factor. An empty entry of `re.included` gives such a row with no warning. A search that selects no factor gives it too, and [`stepwise_factor_set`](@ref) warns. Only [`BackwardElimination`](@ref) under a [`MinMaxValStepwiseRegressionCriterion`](@ref) can select no factor.
      + `L`: Not set. The regression runs in the original factor basis, so `reg.L` returns `reg.M` through the `swap(L, M)` property rule of the result, and `size(reg.L, 2)` is the number of columns of `F`.
      + `edof`: Degrees of freedom of the residuals of each asset, the number of observations less the intercept and the factors that the search selected for that asset. The count reads the selected factors alone, and not the search that selected them.

# Related

  - [`StepwiseRegression`](@ref)
  - [`regression`](@ref)
  - [`Regression`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 4.1, Equations 4.2-4.3. $(ref_dict[:no_original_source])
"""
function regression(re::StepwiseRegression, X::MatNum, F::MatNum)
    assert_finite_factor_rows(F)
    assert_stepwise_included(re.included, X, F)
    cols = size(F, 2) + 1
    N, rows = size(X)
    ovec = range(one(eltype(F)), one(eltype(F)); length = N)
    rr = zeros(promote_type(eltype(F), eltype(X)), rows, cols)
    edof = Vector{Int}(undef, rows)
    for i in axes(rr, 1)
        included = stepwise_factor_set(re, view(X, :, i), F, i)
        x1 = !isempty(included) ? [ovec view(F, :, included)] : reshape(ovec, :, 1)
        fri = StatsAPI.fit(re.tgt, x1, view(X, :, i))
        params = StatsAPI.coef(fri)
        rr[i, 1] = params[1]
        view(rr, i, 2:cols)[included] = view(params, 2:lastindex(params))
        edof[i] = N - size(x1, 2)
    end
    return Regression(; b = view(rr, :, 1), M = view(rr, :, 2:cols), edof = edof)
end
"""
    pin_regression_choice(re::StepwiseRegression{<:Any, <:Any, <:Any, <:PinnedChoice},
                          X::MatNum, F::MatNum, w::Option{<:ObsWeights} = nothing)

Writes the factor set of the first fit of each asset into `included` of a stepwise regression under a [`PinnedChoice`](@ref).

The online step of a prior calls it after the fold, over the rows of the buffer of the prior. The search of one asset reads only the column of that asset and `F`. So the factor set that it writes is the set that the fit over the same rows takes, whichever other assets the prior gives to that fit. A prior that refits the regression under observation weights, as [`OpinionPoolingPrior`](@ref) refits `pe2`, gives those weights as `w`, and the search reads them as that fit does. The fit does not cover an asset whose column holds a value that is not finite. That asset keeps `nothing`, and a later step writes its set at the first fit that covers it. Under [`BatchChoice`](@ref) the generic method returns `re` unchanged.

# Algorithm

 1. Return `re` when `F` holds fewer than two rows. The regression refuses one row, so no fit happens yet.
 2. Check `re.included` with [`assert_stepwise_included`](@ref). Copy it into a vector with one entry per column of `X`, and call it `included`. A field of `nothing` gives a vector of `nothing`.
 3. For each asset `j` whose entry is `nothing` and whose column of `X` is finite, run the stepwise search of `re` on that column, and write the result into entry `j`. When `w` is not `nothing`, the search is the search of `factory(re, w)`.
 4. Return `re` rebuilt with `included`.

# Arguments

  - `re`: Stepwise regression estimator under a [`PinnedChoice`](@ref).
  - $(arg_dict[:X])
  - $(arg_dict[:F])
  - $(arg_dict[:w_pin])

# Validation

  - The rules of [`assert_stepwise_included`](@ref) for `re.included`, `X` and `F`.

# Returns

  - `re::StepwiseRegression`: The regression with the factor set of each covered asset in `included`.

# Related

  - [`StepwiseRegression`](@ref)
  - [`PinnedChoice`](@ref)
  - [`pin_prior_choice`](@ref)
"""
function pin_regression_choice(re::StepwiseRegression{<:Any, <:Any, <:Any, <:PinnedChoice},
                               X::MatNum, F::MatNum, w::Option{<:ObsWeights} = nothing)
    if size(F, 1) < 2
        return re
    end
    assert_stepwise_included(re.included, X, F)
    rw = isnothing(w) ? re : factory(re, w)
    included = Vector{Union{Nothing, Vector{Int}}}(nothing, size(X, 2))
    if !isnothing(re.included)
        included .= re.included
    end
    for j in axes(X, 2)
        x = view(X, :, j)
        if isnothing(included[j]) && all(isfinite, x)
            included[j] = _regression(rw, x, F)
        end
    end
    return rebuild_estimator(re, (; included = included))
end
# A method of `pending_choice`, which `src/10_Prior/12_PriorPartialFit.jl` documents. It lives
# beside the pin it answers for.
function pending_choice(re::StepwiseRegression{<:Any, <:Any, <:Any, <:PinnedChoice})
    return isnothing(re.included) || any(isnothing, re.included)
end

export PValue, ForwardSelection, BackwardElimination, StepwiseRegression
