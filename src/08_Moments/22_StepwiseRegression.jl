"""
$(DocStringExtensions.TYPEDEF)

Selects factors by the statistical significance of their coefficients.

A candidate model is admissible when **every** one of its coefficient p-values is at or below `t`, the intercept excluded. This is the only criterion that reads the fitted coefficients rather than one model-wide score, so it is not a [`MinMaxValStepwiseRegressionCriterion`](@ref) and it takes its own stepwise methods. Under either algorithm the selection never returns an empty factor set: when no factor clears `t`, [`add_best_factor_after_pval_failure!`](@ref) adds the single best one and warns.

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

At each step the algorithm fits one model per excluded factor, keeps the best of them, and stops when no addition improves on the score of the set it already holds. Under a [`MinMaxValStepwiseRegressionCriterion`](@ref) the starting score is [`regression_threshold`](@ref), the worst value that criterion can take, so the first addition always happens; under [`PValue`](@ref) the step instead admits a candidate whose p-values all clear `t`. **The selection is therefore never empty under either criterion**, which is the one behaviour that separates this tag from [`BackwardElimination`](@ref). The steps are stated on the two methods that run them, `_regression(::StepwiseRegression{<:PValue, <:ForwardSelection}, ::VecNum, ::MatNum)` and `_regression(::StepwiseRegression{<:MinMaxValStepwiseRegressionCriterion, <:ForwardSelection}, ::VecNum, ::MatNum)`.

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

At each step the algorithm fits one model per included factor, each with that factor dropped, and removes the factor whose reduced model scores best. It stops when no removal improves on the score of the set it already holds. The starting score is the score of the **full** model, not [`regression_threshold`](@ref); under [`PValue`](@ref) the step instead drops the factor with the largest p-value while any exceeds `t`. **Under a [`MinMaxValStepwiseRegressionCriterion`](@ref) the selection can therefore empty**, because a criterion that rewards every removal removes every factor; the asset then gets an intercept-only model, and its row of the loadings matrix is all zeros. [`regression`](@ref) warns when that happens, naming the asset, so an unexplained asset is never silent. The steps are stated on the two methods that run them, `_regression(::StepwiseRegression{<:PValue, <:BackwardElimination}, ::VecNum, ::MatNum)` and `_regression(::StepwiseRegression{<:MinMaxValStepwiseRegressionCriterion, <:BackwardElimination}, ::VecNum, ::MatNum)`.

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

Adds the factor of smallest p-value when a p-value search selected none.

Both [`PValue`](@ref) methods of `_regression` call it last, so a p-value search never returns an empty factor set. It does nothing when `included` already holds a factor, and it warns whenever it adds one, because the factor it adds failed the threshold by construction.

# Algorithm

 1. Return at once when `included` is not empty.
 2. For each factor `i` of `F`, fit `tgt` to an intercept column and column `i`, and read that column's p-value, giving `test_pval`.
 3. Keep the smallest `test_pval` of step 2 and the factor that carries it, giving `best_pval` and `new_factor`.
 4. Warn, naming `new_factor` and `best_pval`.
 5. Push `new_factor` onto `included`.

# Arguments

  - `tgt`: Regression target that fits each candidate model.
  - `included`: Indices of the factors the search selected. Written in place.
  - $(arg_dict[:F])
  - `x`: Response vector, `observations × 1`.

# Returns

  - `nothing`: `included` gains exactly one index when it was empty, and is unchanged otherwise.

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
    @warn("No factor with p-value lower than threshold. Best we can do is factor $new_factor, with p-value $best_pval.")
    push!(included, new_factor)
    return nothing
end
"""
    _regression(re::StepwiseRegression{<:PValue, <:ForwardSelection}, x::VecNum,
               F::MatNum)

Grows a factor set from empty, admitting the model whose p-values all clear the threshold.

A candidate is admissible only when **every** coefficient of that candidate model is significant, so a factor that is significant on its own is rejected when it makes an incumbent insignificant.

# Algorithm

 1. Set `included` empty and `val` to zero, so the loop runs at least once.
 2. While `val` does not exceed `re.crit.t`, do steps 3 to 6.
 3. Take the factors that `included` does not hold, giving `excluded`.
 4. For each `i` of `excluded`, fit `re.tgt` to an intercept column and the columns `[included; i]`, and read the p-value of column `i`, giving `test_pval`. Keep `i` as `new_factor` when `test_pval` is the smallest so far **and** no p-value of that candidate model exceeds `re.crit.t`, and keep that model's p-values as `pvals`.
 5. Stop the loop when step 4 kept no factor. Otherwise push `new_factor` onto `included`.
 6. Set `val` to the largest entry of `pvals`.
 7. Call [`add_best_factor_after_pval_failure!`](@ref), which acts only when `included` is still empty.

# Arguments

  - `re`: Stepwise regression estimator with a [`PValue`](@ref) criterion and the [`ForwardSelection`](@ref) algorithm.
  - `x`: Response vector, `observations × 1`.
  - $(arg_dict[:F])

# Returns

  - `included::Vector{Int}`: Indices of the selected factors, in the order the loop added them, so it is not sorted. It holds at least one index.

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

Moves the best excluded factor into `included` when it lowers a minimised criterion.

`findmin` searches the **whole** `value` vector, and the answer is still the best **excluded** factor. An entry of an included factor holds the score that selected it; `t` is that same score, and it only falls, so every included entry is at or above the current `t`. A score strictly below `t` therefore belongs to an excluded factor, and `searchsortedfirst` always finds its index in `excluded`.

# Algorithm

 1. Find the smallest entry of `value`, giving `val` and its index `idx`.
 2. Return `t` unchanged when `val` is not below `t`.
 3. Otherwise find the position of `idx` in `excluded`, move that entry to the end of `included`, and set `t` to `val`.

# Arguments

  - `::MinValStepwiseRegressionCriterion`: Stepwise regression criterion whose lower values are better.
  - `value`: Criterion score of each factor. Entry `i` is the score of the model that **adds** factor `i` to `included`.
  - `excluded`: Indices of the factors outside the model, in ascending order. Written in place.
  - `included`: Indices of the factors inside the model, in the order the search added them. Written in place.
  - $(arg_dict[:t])

# Returns

  - `t::Number`: The score of the factor that moved, or the input `t` when no factor moved.

# Related

  - [`StepwiseRegression`](@ref)
  - [`MinValStepwiseRegressionCriterion`](@ref)
  - [`get_backward_reg_incl!`](@ref)
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

Moves the best excluded factor into `included` when it raises a maximised criterion.

`findmax` searches the **whole** `value` vector, and the answer is still the best **excluded** factor. An entry of an included factor holds the score that selected it; `t` is that same score, and it only rises, so every included entry is at or below the current `t`. A score strictly above `t` therefore belongs to an excluded factor, and `searchsortedfirst` always finds its index in `excluded`.

# Algorithm

 1. Find the largest entry of `value`, giving `val` and its index `idx`.
 2. Return `t` unchanged when `val` is not above `t`.
 3. Otherwise find the position of `idx` in `excluded`, move that entry to the end of `included`, and set `t` to `val`.

# Arguments

  - `::MaxValStepwiseRegressionCriterion`: Stepwise regression criterion whose higher values are better.
  - `value`: Criterion score of each factor. Entry `i` is the score of the model that **adds** factor `i` to `included`.
  - `excluded`: Indices of the factors outside the model, in ascending order. Written in place.
  - `included`: Indices of the factors inside the model, in the order the search added them. Written in place.
  - $(arg_dict[:t])

# Returns

  - `t::Number`: The score of the factor that moved, or the input `t` when no factor moved.

# Related

  - [`StepwiseRegression`](@ref)
  - [`MaxValStepwiseRegressionCriterion`](@ref)
  - [`get_backward_reg_incl!`](@ref)
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

Grows a factor set from empty, adding the factor whose model scores best.

The starting score is the worst value the criterion can take, so the first addition always happens and the selection is never empty. [`get_forward_reg_incl_excl!`](@ref) dispatches on the direction, so one loop serves a minimised and a maximised criterion alike.

# Algorithm

 1. Set `included` empty and `excluded` to every factor index, in ascending order.
 2. Read the criterion from `re.crit` and `re.tgt` with [`regression_criterion_func`](@ref), giving `criterion_func`, and the starting score from [`regression_threshold`](@ref), giving `t`.
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

  - `included::Vector{Int}`: Indices of the selected factors, in the order the loop added them, so it is not sorted. It holds at least one index.

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

Shrinks a factor set from full, dropping the factor of largest p-value while any exceeds the threshold.

# Algorithm

 1. Fit `re.tgt` to an intercept column and every column of `F`, read the coefficient p-values, and set `val` to the largest of them.
 2. Set `included` to every factor index and `excluded` empty.
 3. While `val` exceeds `re.crit.t`, do steps 4 to 6.
 4. Set `included` to the factors that `excluded` does not hold. Stop the loop when `included` is empty.
 5. Fit `re.tgt` to an intercept column and the columns `included`, giving `pvals`.
 6. Set `val` to the largest entry of `pvals`, and push that entry's factor onto `excluded`. The push acts only when step 3 runs the loop again, because step 4 is what reads `excluded`, so the last iteration's push is discarded.
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
    get_backward_reg_incl!(::MinValStepwiseRegressionCriterion, value::VecNum,
                           included::VecInt, t::Number)

Removes the best included factor from `included` when its removal lowers a minimised criterion.

`findmin` searches the **whole** `value` vector, and the answer is still the best **included** factor. An entry of a removed factor holds the score that removed it; `t` is that same score, and it only falls, so every removed entry is at or above the current `t`. A score strictly below `t` therefore belongs to an included factor, and `searchsortedfirst` always finds its index in `included`.

# Algorithm

 1. Find the smallest entry of `value`, giving `val` and its index `idx`. That is the best model reachable by one removal.
 2. Return `t` unchanged when `val` is not below `t`.
 3. Otherwise find the position of `idx` in `included`, remove that entry, and set `t` to `val`.

# Arguments

  - `::MinValStepwiseRegressionCriterion`: Stepwise regression criterion whose lower values are better.
  - `value`: Criterion score of each factor. Entry `j` is the score of the model that **omits** factor `j`, so the best entry names the removal that helps most.
  - `included`: Indices of the factors inside the model, in ascending order. Written in place.
  - $(arg_dict[:t])

# Returns

  - `t::Number`: The score of the model left by the removal, or the input `t` when no factor was removed.

# Related

  - [`StepwiseRegression`](@ref)
  - [`MinValStepwiseRegressionCriterion`](@ref)
  - [`get_forward_reg_incl_excl!`](@ref)
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

Removes the best included factor from `included` when its removal raises a maximised criterion.

`findmax` searches the **whole** `value` vector, and the answer is still the best **included** factor. An entry of a removed factor holds the score that removed it; `t` is that same score, and it only rises, so every removed entry is at or below the current `t`. A score strictly above `t` therefore belongs to an included factor, and `searchsortedfirst` always finds its index in `included`.

# Algorithm

 1. Find the largest entry of `value`, giving `val` and its index `idx`. That is the best model reachable by one removal.
 2. Return `t` unchanged when `val` is not above `t`.
 3. Otherwise find the position of `idx` in `included`, remove that entry, and set `t` to `val`.

# Arguments

  - `::MaxValStepwiseRegressionCriterion`: Stepwise regression criterion whose higher values are better.
  - `value`: Criterion score of each factor. Entry `j` is the score of the model that **omits** factor `j`, so the best entry names the removal that helps most.
  - `included`: Indices of the factors inside the model, in ascending order. Written in place.
  - $(arg_dict[:t])

# Returns

  - `t::Number`: The score of the model left by the removal, or the input `t` when no factor was removed.

# Related

  - [`StepwiseRegression`](@ref)
  - [`MaxValStepwiseRegressionCriterion`](@ref)
  - [`get_forward_reg_incl_excl!`](@ref)
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

Shrinks a factor set from full, removing the factor whose reduced model scores best.

`value[j]` is the score of the model that **omits** `j`, so the reading of "best" is the same as the forward direction's and not its inverse: under a minimised criterion the code removes the factor of **lowest** value, and under a maximised one the factor of **highest**. On a 200×5 sample whose response is built from factors 1 and 3, the five reduced-model `:aic` scores were `[487.92, -372.72, 82.79, -369.15, -372.44]` against a full-model score of `-371.52`; the code removed factor 2, the lowest, and kept factor 1, the highest and the strongest signal in the response.

# Algorithm

 1. Set `included` to every factor index, in ascending order.
 2. Read the criterion from `re.crit` and `re.tgt` with [`regression_criterion_func`](@ref), giving `criterion_func`.
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

  - `included::Vector{Int}`: Indices of the selected factors, in ascending order. **It can be empty**: a criterion that rewards every removal removes every factor, which is the common outcome on a response the factors do not explain. The caller [`regression`](@ref) warns on an empty return, naming the asset, and fits the intercept column alone.

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

Runs one stepwise search per asset and assembles the loadings matrix from the fits.

Each asset takes its own search, so the searches see one another only through the buffer they write into.

# Algorithm

 1. Allocate `rr`, a dense `assets × (factors + 1)` buffer of zeros. A factor an asset never selected keeps its zero.
 2. For each asset `i`, do steps 3 to 5.
 3. Run the stepwise search of `re` on column `i` of `X`, giving `included`.
 4. Fit `re.tgt` to an intercept column and the columns `included` of `F`, and read its coefficients, giving `params`. Warn, naming the asset, and fit the intercept column alone when `included` is empty.
 5. Write `params[1]` into `rr[i, 1]`, and the remaining coefficients into the columns of `rr` that `included` names, in the order `included` holds them.
 6. Build a [`Regression`](@ref) from the first column of `rr` and its remaining columns.

# Arguments

  - `re`: Stepwise regression estimator that supplies the criterion, the algorithm and the regression target.
  - $(arg_dict[:X])
  - $(arg_dict[:F])

# Returns

  - `reg::Regression`: Regression result carrying:

      + `b`: Intercept of each asset, a view of the first column of `rr`.
      + `M`: Coefficient of each asset and factor, a view of the remaining columns of `rr`. An unselected factor is an exact zero. A whole row is zero when the search selected no factor for that asset, which only [`BackwardElimination`](@ref) under a [`MinMaxValStepwiseRegressionCriterion`](@ref) can do, and which the loop warns about.
      + `L`: Left **unset**. The regression runs in the original factor basis, so `reg.L` reads back as `reg.M` through the result's `swap(L, M)` property rule, and `size(reg.L, 2)` is the number of columns of `F`.

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
        if isempty(included)
            @warn("Asset $i: the stepwise search selected no factor. The asset gets an intercept-only model, and its row of the loadings matrix is all zeros.")
        end
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
