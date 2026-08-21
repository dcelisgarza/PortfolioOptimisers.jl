"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all regression estimator types.

All concrete and/or abstract types implementing regression estimation algorithms should be subtypes of `AbstractRegressionEstimator`.

# Interfaces

In order to implement a new regression estimator which will work seamlessly with the library, subtype `AbstractRegressionEstimator` with all necessary parameters as part of the struct, and implement the following methods:

## Regression

  - `PortfolioOptimisers.regression(re::AbstractRegressionEstimator, X::MatNum, F::MatNum) -> Regression`: Computes the regression result from asset returns `X` and factor returns `F`.

### Arguments

  - `re`: Regression estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:F])

### Returns

  - `reg::Regression`: Regression result containing the coefficient matrix and optional intercept.

# Examples

We can create a dummy regression estimator as follows:

```jldoctest
julia> struct MyRegressionEstimator <: PortfolioOptimisers.AbstractRegressionEstimator end

julia> function PortfolioOptimisers.regression(::MyRegressionEstimator,
                                               X::PortfolioOptimisers.MatNum,
                                               F::PortfolioOptimisers.MatNum)
           return PortfolioOptimisers.Regression(; M = F \\ X)
       end

julia> regression(MyRegressionEstimator(), [1.0 2.0; 3.0 4.0; 5.0 6.0],
                  [1.0 0.0; 0.0 1.0; 0.5 0.5])
Regression
  M ┼ 2×2 Matrix{Float64}
  L ┼ 2×2 Matrix{Float64}
  b ┴ nothing
```

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractRegressionAlgorithm`](@ref)
  - [`AbstractRegressionResult`](@ref)
"""
abstract type AbstractRegressionEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all regression result types.

All concrete and/or abstract types representing the output of regression-based moment estimation should be subtypes of `AbstractRegressionResult`.

# Related

  - [`AbstractResult`](@ref)
  - [`Regression`](@ref)
  - [`AbstractRegressionEstimator`](@ref)
"""
abstract type AbstractRegressionResult <: AbstractResult end
"""
    const RegE_Reg = Union{<:AbstractRegressionResult, <:AbstractRegressionEstimator}

Alias for a regression result or estimator.

Matches either an [`AbstractRegressionResult`](@ref) (pre-computed regression result) or an [`AbstractRegressionEstimator`](@ref) (regression specification). Used for dispatch in factor model and regression-based risk routines.

# Related

  - [`AbstractRegressionResult`](@ref)
  - [`AbstractRegressionEstimator`](@ref)
"""
const RegE_Reg = Union{<:AbstractRegressionResult, <:AbstractRegressionEstimator}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all regression algorithm types.

All concrete and/or abstract types implementing specific regression algorithms should be subtypes of `AbstractRegressionAlgorithm`.

These types are used to specify the algorithm when constructing a regression estimator.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractRegressionAlgorithm`](@ref)
  - [`AbstractStepwiseRegressionAlgorithm`](@ref)
  - [`AbstractStepwiseRegressionCriterion`](@ref)
  - [`AbstractRegressionTarget`](@ref)
"""
abstract type AbstractRegressionAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all stepwise regression algorithm types.

All concrete and/or abstract types implementing stepwise regression algorithms should be subtypes of `AbstractStepwiseRegressionAlgorithm`. A stepwise algorithm decides the *direction* the factor set moves in, and an [`AbstractStepwiseRegressionCriterion`](@ref) decides which move is an improvement.

# Related

  - [`AbstractRegressionAlgorithm`](@ref)
  - [`AbstractStepwiseRegressionCriterion`](@ref)
  - [`AbstractRegressionTarget`](@ref)
  - [`ForwardSelection`](@ref)
  - [`BackwardElimination`](@ref)

# References

  - $(ref_dict[:efroymson1960])
"""
abstract type AbstractStepwiseRegressionAlgorithm <: AbstractRegressionAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all stepwise regression criterion types.

All concrete and/or abstract types representing criteria for stepwise regression algorithms should be subtypes of `AbstractStepwiseRegressionCriterion`. A criterion scores a fitted model, and the stepwise algorithm keeps the move that improves the score.

# Related

  - [`AbstractStepwiseRegressionAlgorithm`](@ref)
  - [`AbstractRegressionTarget`](@ref)
  - [`AbstractMinMaxValStepwiseRegressionCriterion`](@ref)
  - [`PValue`](@ref)

# References

  - $(ref_dict[:hocking1976])
"""
abstract type AbstractStepwiseRegressionCriterion <: AbstractRegressionAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all regression target types.

All concrete and/or abstract types representing regression targets (such as linear or generalised linear models) should be subtypes of `AbstractRegressionTarget`.

# Related

  - [`AbstractRegressionAlgorithm`](@ref)
"""
abstract type AbstractRegressionTarget <: AbstractRegressionAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Fits each response by ordinary least squares through `GLM.LinearModel`.

The `kwargs` field is forwarded verbatim to `GLM`, so any option that routine accepts — observation weights among them — reaches the fit. This is the default target of every regression estimator in the library.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LinearModel(;
        kwargs::NamedTuple = (;)
    ) -> LinearModel

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> LinearModel()
LinearModel
  kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractRegressionTarget`](@ref)
  - [`GeneralisedLinearModel`](@ref)
  - [`StatsAPI.fit(::LinearModel, ::MatNum, ::VecNum)`](@ref)
"""
@concrete struct LinearModel <: AbstractRegressionTarget
    """
    Keyword arguments passed to `fit(GLM.LinearModel, X, y; kwargs...)`.
    """
    kwargs
    function LinearModel(kwargs::NamedTuple)
        return new{typeof(kwargs)}(kwargs)
    end
end
function LinearModel(; kwargs::NamedTuple = (;))::LinearModel
    return LinearModel(kwargs)
end
"""
    factory(re::LinearModel, w::ObsWeights) -> LinearModel

Return a new [`LinearModel`](@ref) regression target with observation weights `w` added to the keyword arguments.

# Arguments

  - `re`: Linear model regression target.
  - $(arg_dict[:ow])

# Returns

  - `re::LinearModel`: Updated regression target with weights included in `kwargs`.

# Related

  - [`LinearModel`](@ref)
  - [`factory`](@ref)
"""
function factory(re::LinearModel, w::ObsWeights)::LinearModel
    return LinearModel(; kwargs = (; re.kwargs..., weights = w))
end
"""
    StatsAPI.fit(tgt::LinearModel, X::MatNum, y::VecNum)

Fit a standard linear regression model using a [`LinearModel`](@ref) regression target.

This method dispatches to `StatsAPI.fit` with the `GLM.LinearModel` type, passing the design matrix `X`, response vector `y`, and any keyword arguments stored in `tgt.kwargs`. It enables flexible configuration of the underlying linear model fitting routine within the regression estimation framework.

# Arguments

  - `tgt`: Regression target specifying model options.
  - `X`: The design matrix (observations × factors).
  - `y`: The response vector.

# Returns

  - `model::GLM.LinearModel`: A fitted linear model object from the GLM.jl package.

# Related

  - [`LinearModel`](@ref)
  - [`GLM.LinearModel`](https://juliastats.org/GLM.jl/stable/api/#GLM.LinearModel)
"""
function StatsAPI.fit(tgt::LinearModel, X::MatNum, y::VecNum)
    kwargs = if haskey(tgt.kwargs, :weights) &&
                isa(tgt.kwargs.weights, DynamicAbstractWeights)
        w = get_observation_weights(tgt.kwargs.weights, X)
        (; tgt.kwargs..., weights = w)
    else
        tgt.kwargs
    end
    return StatsAPI.fit(GLM.LinearModel, X, y; kwargs...)
end
"""
$(DocStringExtensions.TYPEDEF)

Fits each response by a generalised linear model through `GLM.GeneralizedLinearModel`.

The `args` field carries the response distribution and, optionally, the link function; `kwargs` carries the remaining `GLM` options. The default `args = (Normal(),)` with the canonical identity link reproduces ordinary least squares.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GeneralisedLinearModel(;
        args::Tuple = (Normal(),),
        kwargs::NamedTuple = (;)
    ) -> GeneralisedLinearModel

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> GeneralisedLinearModel()
GeneralisedLinearModel
    args ┼ Tuple{Distributions.Normal{Float64}}: (Distributions.Normal{Float64}(μ=0.0, σ=1.0),)
  kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractRegressionTarget`](@ref)
  - [`LinearModel`](@ref)
  - [`StatsAPI.fit(::GeneralisedLinearModel, ::MatNum, ::VecNum)`](@ref)

# References

  - $(ref_dict[:nelder1972])
"""
@concrete struct GeneralisedLinearModel <: AbstractRegressionTarget
    """
    Positional arguments passed to `fit(GLM.GeneralizedLinearModel, X, y, args...; kwargs...)`.
    """
    args
    """
    Keyword arguments passed to `fit(GLM.GeneralizedLinearModel, X, y, args...; kwargs...)`.
    """
    kwargs
    function GeneralisedLinearModel(args::Tuple, kwargs::NamedTuple)
        return new{typeof(args), typeof(kwargs)}(args, kwargs)
    end
end
function GeneralisedLinearModel(; args::Tuple = (Distributions.Normal(),),
                                kwargs::NamedTuple = (;))::GeneralisedLinearModel
    return GeneralisedLinearModel(args, kwargs)
end
"""
    factory(re::GeneralisedLinearModel, w::ObsWeights) -> GeneralisedLinearModel

Return a new [`GeneralisedLinearModel`](@ref) regression target with observation weights `w` added to the keyword arguments.

# Arguments

  - `re`: Generalised linear model regression target.
  - $(arg_dict[:ow])

# Returns

  - `re::GeneralisedLinearModel`: Updated regression target with weights included in `kwargs`.

# Related

  - [`GeneralisedLinearModel`](@ref)
  - [`factory`](@ref)
"""
function factory(re::GeneralisedLinearModel, w::ObsWeights)::GeneralisedLinearModel
    return GeneralisedLinearModel(; args = re.args, kwargs = (; re.kwargs..., weights = w))
end
"""
    StatsAPI.fit(tgt::GeneralisedLinearModel, X::MatNum, y::VecNum)

Fit a generalised linear regression model using a [`GeneralisedLinearModel`](@ref) regression target.

This method dispatches to `StatsAPI.fit` with the `GLM.GeneralizedLinearModel` type, passing the design matrix `X`, response vector `y`, any positional arguments in `tgt.args`, and any keyword arguments in `tgt.kwargs`.

# Arguments

  - `tgt`: A [`GeneralisedLinearModel`](@ref) regression target specifying model options.
  - `X`: The design matrix (observations × factors).
  - `y`: The response vector.

# Returns

  - `model::GLM.GeneralizedLinearModel`: A fitted generalised linear model object from the GLM.jl package.

# Related

  - [`GeneralisedLinearModel`](@ref)
  - [`GLM.GeneralizedLinearModel`](https://juliastats.org/GLM.jl/stable/examples/#Probit-regression)
"""
function StatsAPI.fit(tgt::GeneralisedLinearModel, X::MatNum, y::VecNum)
    kwargs = if haskey(tgt.kwargs, :weights) &&
                isa(tgt.kwargs.weights, DynamicAbstractWeights)
        w = get_observation_weights(tgt.kwargs.weights, X)
        (; tgt.kwargs..., weights = w)
    else
        tgt.kwargs
    end
    return StatsAPI.fit(GLM.GeneralizedLinearModel, X, y, tgt.args...; kwargs...)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all stepwise regression criteria that score a fitted model with one number.

All concrete and/or abstract types representing stepwise regression criteria (such as AIC, BIC, R², or Adjusted R²) should be subtypes of `AbstractMinMaxValStepwiseRegressionCriterion`. Its two subfamilies carry the polarity of that number, which is what [`regression_threshold`](@ref) and the `get_*_reg_incl*!` helpers dispatch on. [`PValue`](@ref) is not a member: it reads the coefficient p-values of the fitted model instead of one score.

# Related

  - [`AbstractStepwiseRegressionCriterion`](@ref)
  - [`AbstractMinValStepwiseRegressionCriterion`](@ref)
  - [`AbstractMaxValStepwiseRegressionCriteria`](@ref)
  - [`regression_criterion_func`](@ref)
  - [`regression_threshold`](@ref)
"""
abstract type AbstractMinMaxValStepwiseRegressionCriterion <:
              AbstractStepwiseRegressionCriterion end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all stepwise regression criteria where lower values indicate better model fit.

All concrete and/or abstract types implementing minimisation-based stepwise regression criteria (such as AIC, AICC, or BIC) should be subtypes of `AbstractMinValStepwiseRegressionCriterion`. These criteria are used to guide variable selection in stepwise regression algorithms by minimising the criterion value.

# Related

  - [`AbstractMinMaxValStepwiseRegressionCriterion`](@ref)
  - [`AIC`](@ref)
  - [`AICC`](@ref)
  - [`BIC`](@ref)
"""
abstract type AbstractMinValStepwiseRegressionCriterion <:
              AbstractMinMaxValStepwiseRegressionCriterion end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all stepwise regression criteria where higher values indicate better model fit.

All concrete and/or abstract types implementing maximisation-based stepwise regression criteria (such as R² or Adjusted R²) should be subtypes of `AbstractMaxValStepwiseRegressionCriteria`. These criteria are used to guide variable selection in stepwise regression algorithms by maximising the criterion value.

# Related

  - [`AbstractMinMaxValStepwiseRegressionCriterion`](@ref)
  - [`RSquared`](@ref)
  - [`AdjustedRSquared`](@ref)
"""
abstract type AbstractMaxValStepwiseRegressionCriteria <:
              AbstractMinMaxValStepwiseRegressionCriterion end
"""
$(DocStringExtensions.TYPEDEF)

Selects factors by minimising the Akaike Information Criterion.

The criterion trades the fitted likelihood against the number of estimated parameters, so a lower value is a better model. [`regression_criterion_func`](@ref) maps it to `StatsAPI.aic`.

# Mathematical definition

```math
\\begin{align}
\\mathrm{AIC} &= 2k - 2\\ln\\hat{L}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{AIC}``: Akaike Information Criterion.
  - ``k``: Number of estimated parameters, which is `StatsAPI.dof` of the fitted model: the regression coefficients, the intercept, and the residual variance.
  - ``\\hat{L}``: Maximum likelihood of the model.

# Related

  - [`AbstractMinValStepwiseRegressionCriterion`](@ref)
  - [`AICC`](@ref)
  - [`BIC`](@ref)
  - [`regression_criterion_func(::AIC)`](@ref)

# References

  - $(ref_dict[:akaike1974])
"""
struct AIC <: AbstractMinValStepwiseRegressionCriterion end
"""
$(DocStringExtensions.TYPEDEF)

Selects factors by minimising the Akaike Information Criterion corrected for a small sample.

The correction term grows as ``T`` approaches ``k``, so `AICC` penalises a large model more heavily than [`AIC`](@ref) does on a short sample. [`regression_criterion_func`](@ref) maps it to `StatsAPI.aicc`.

# Mathematical definition

```math
\\begin{align}
\\mathrm{AICC} &= \\mathrm{AIC} + \\frac{2k(k+1)}{T - k - 1}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{AICC}``: Corrected Akaike Information Criterion.
  - ``\\mathrm{AIC}``: Standard Akaike Information Criterion.
  - ``k``: Number of estimated parameters, which is `StatsAPI.dof` of the fitted model: the regression coefficients, the intercept, and the residual variance.
  - $(math_dict[:T])

# Details

  - The correction term divides by ``T - k - 1``. The criterion is undefined when ``T = k + 1`` and changes sign below it, so `AICC` needs a sample longer than the largest model the search can reach.

# Related

  - [`AbstractMinValStepwiseRegressionCriterion`](@ref)
  - [`AIC`](@ref)
  - [`BIC`](@ref)
  - [`regression_criterion_func(::AICC)`](@ref)

# References

  - $(ref_dict[:hurvich1989])
"""
struct AICC <: AbstractMinValStepwiseRegressionCriterion end
"""
$(DocStringExtensions.TYPEDEF)

Selects factors by minimising the Bayesian Information Criterion.

The penalty is ``k \\ln T`` rather than ``2k``, which is heavier than [`AIC`](@ref)'s for ``T \\geq 8``, so the search usually stops with fewer factors. [`regression_criterion_func`](@ref) maps it to `StatsAPI.bic`.

# Mathematical definition

```math
\\begin{align}
\\mathrm{BIC} &= k\\ln T - 2\\ln\\hat{L}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{BIC}``: Bayesian Information Criterion.
  - ``k``: Number of estimated parameters, which is `StatsAPI.dof` of the fitted model: the regression coefficients, the intercept, and the residual variance.
  - $(math_dict[:T])
  - ``\\hat{L}``: Maximum likelihood of the model.

# Related

  - [`AbstractMinValStepwiseRegressionCriterion`](@ref)
  - [`AIC`](@ref)
  - [`AICC`](@ref)
  - [`regression_criterion_func(::BIC)`](@ref)

# References

  - $(ref_dict[:schwarz1978])
"""
struct BIC <: AbstractMinValStepwiseRegressionCriterion end
"""
$(DocStringExtensions.TYPEDEF)

Selects factors by maximising the coefficient of determination.

The score is the share of the response variance the model explains, so a higher value is a better fit. [`regression_criterion_func`](@ref) maps it to `GLM.r2`.

# Mathematical definition

```math
\\begin{align}
R^2 &= 1 - \\frac{\\mathrm{SS}_{\\mathrm{res}}}{\\mathrm{SS}_{\\mathrm{tot}}} = 1 - \\frac{\\sum_t (y_t - \\hat{y}_t)^2}{\\sum_t (y_t - \\bar{y})^2}\\,.
\\end{align}
```

Where:

  - ``R^2``: Coefficient of determination.
  - ``\\mathrm{SS}_{\\mathrm{res}}``: Residual sum of squares.
  - ``\\mathrm{SS}_{\\mathrm{tot}}``: Total sum of squares.
  - ``y_t``: Observed response at time ``t``.
  - ``\\hat{y}_t``: Fitted response at time ``t``.
  - ``\\bar{y}``: Mean of observed responses.

# Details

  - ``R^2`` never falls when a factor is added, so it penalises no complexity at all. Under [`ForwardSelection`](@ref) it admits **every** factor and under [`BackwardElimination`](@ref) it removes **none**. On a 200×5 sample where [`AIC`](@ref) kept factors 1, 3 and 4 and [`BIC`](@ref) kept 1 and 3, `RSquared` kept all five in both directions. Use [`AdjustedRSquared`](@ref), [`AIC`](@ref), [`AICC`](@ref) or [`BIC`](@ref) when the criterion must pay for size.
  - `GLM` defines this quantity for a fitted [`LinearModel`](@ref) only. Paired with a [`GeneralisedLinearModel`](@ref) target, the criterion throws a `MethodError` from `GLM` when the first candidate model is scored.

# Related

  - [`AbstractMaxValStepwiseRegressionCriteria`](@ref)
  - [`AdjustedRSquared`](@ref)
  - [`regression_criterion_func(::RSquared)`](@ref)

# References

  - $(ref_dict[:hocking1976])
"""
struct RSquared <: AbstractMaxValStepwiseRegressionCriteria end
"""
$(DocStringExtensions.TYPEDEF)

Selects factors by maximising the coefficient of determination adjusted for the model size.

The adjustment discounts ``R^2`` by the degrees of freedom the predictors consume, so unlike [`RSquared`](@ref) the score can fall when a factor is added. This makes it usable as a stopping rule.

# Mathematical definition

```math
\\begin{align}
\\bar{R}^2 &= 1 - (1 - R^2) \\frac{T - 1}{T - k - 1}\\,.
\\end{align}
```

Where:

  - ``\\bar{R}^2``: Adjusted coefficient of determination.
  - ``R^2``: Standard coefficient of determination.
  - ``k``: Number of predictors, excluding the intercept. [`AIC`](@ref), [`AICC`](@ref) and [`BIC`](@ref) write ``k`` for a different count: the predictors, plus the intercept, plus the residual variance.
  - $(math_dict[:T])

# Details

  - `GLM` defines this quantity for a fitted [`LinearModel`](@ref) only. Paired with a [`GeneralisedLinearModel`](@ref) target, the criterion throws a `MethodError` from `GLM` when the first candidate model is scored.

# Related

  - [`AbstractMaxValStepwiseRegressionCriteria`](@ref)
  - [`RSquared`](@ref)
  - [`regression_criterion_func(::AdjustedRSquared)`](@ref)

# References

  - $(ref_dict[:theil1961])
"""
struct AdjustedRSquared <: AbstractMaxValStepwiseRegressionCriteria end
"""
    regression_criterion_func(::AbstractStepwiseRegressionCriterion)

Return the function used to compute the value of a stepwise regression criterion.

This utility dispatches on the concrete criterion subtype of [`AbstractStepwiseRegressionCriterion`](@ref), returning the corresponding function from [`GLM.jl`](https://juliastats.org/GLM.jl/stable/#Methods-applied-to-fitted-models). Used internally by stepwise regression algorithms to evaluate model quality.

# Arguments

  - `criterion`: A stepwise regression criterion type (e.g., `AIC()`, `BIC()`, `RSquared()`).

# Returns

  - `f::Function`: The function that computes the criterion value for a fitted model.

# Details

The map is:

| Criterion                  | Function        |
|:-------------------------- |:--------------- |
| [`AIC`](@ref)              | `StatsAPI.aic`  |
| [`AICC`](@ref)             | `StatsAPI.aicc` |
| [`BIC`](@ref)              | `StatsAPI.bic`  |
| [`RSquared`](@ref)         | `GLM.r2`        |
| [`AdjustedRSquared`](@ref) | `GLM.adjr2`     |

`StatsAPI.aic`, `StatsAPI.aicc` and `StatsAPI.bic` accept a fitted [`LinearModel`](@ref) and a fitted [`GeneralisedLinearModel`](@ref) alike. `GLM.r2` and `GLM.adjr2` accept the linear model only, so [`RSquared`](@ref) and [`AdjustedRSquared`](@ref) throw a `MethodError` under a [`GeneralisedLinearModel`](@ref) target.

[`PValue`](@ref) has no method here. It reads the coefficient p-values of the fitted model, not one score, so its stepwise methods are separate.

# Related

  - [`AIC`](@ref)
  - [`AICC`](@ref)
  - [`BIC`](@ref)
  - [`RSquared`](@ref)
  - [`AdjustedRSquared`](@ref)
"""
function regression_criterion_func(::AIC)
    return StatsAPI.aic
end
function regression_criterion_func(::AICC)
    return StatsAPI.aicc
end
function regression_criterion_func(::BIC)
    return StatsAPI.bic
end
function regression_criterion_func(::RSquared)
    return GLM.r2
end
function regression_criterion_func(::AdjustedRSquared)
    return GLM.adjr2
end
"""
    regression_threshold(alg)

Return the starting threshold for a forward stepwise regression search.

The value is the worst score the criterion can take, so the first candidate model always improves on it. Dispatches on the polarity of the criterion.

# Arguments

  - `alg`: Stepwise regression criterion.

# Returns

  - `t::Number`: `Inf` for a minimisation criterion, `-Inf` for a maximisation criterion.

# Details

Only [`ForwardSelection`](@ref) reads this. [`BackwardElimination`](@ref) starts from the score of the full model instead, because its first move must beat a model that already exists.

# Related

  - [`AbstractMinValStepwiseRegressionCriterion`](@ref)
  - [`AbstractMaxValStepwiseRegressionCriteria`](@ref)
"""
function regression_threshold(::AbstractMinValStepwiseRegressionCriterion)
    return Inf
end
function regression_threshold(::AbstractMaxValStepwiseRegressionCriteria)
    return -Inf
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the loadings matrix, the intercept vector and the reduced-basis loadings of a fitted factor model.

`M` and `b` are the loadings matrix ``B`` and the intercept vector ``\\alpha`` of the factor model, one row per asset. `L` carries the same loadings written in the reduced basis a dimension reduction produced; it is unset when the estimator regresses on the original factors.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Regression(;
        M::MatNum,
        L::Option{<:MatNum} = nothing,
        b::Option{<:VecNum} = nothing
    ) -> Regression

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(M)`.
  - If provided, `!isempty(b)`, and `length(b) == size(M, 1)`.
  - If provided, `size(L, 1) == size(M, 1)`. The constructor does **not** reject an empty `L`, so an `L` with the right number of rows and no columns is accepted.

# Details

  - **An unset `L` reads back as `M`.** A `@forward_properties` `swap(L, M)` rule makes `re.L` return `re.M` whenever `L` was not given, and a consumer that decomposes risk in the factor basis needs no `Nothing` branch. [`StepwiseRegression`](@ref) leaves `L` unset; [`DimensionReductionRegression`](@ref) sets it. Use `getfield(re, :L)` when the unset case must be told apart, as [`port_opt_view`](@ref) does.
  - `size(L, 2)` is therefore the width of the basis risk is decomposed in: the original factors under a stepwise regression, and the retained principal components under a dimension reduction regression.

# Examples

```jldoctest
julia> Regression(; M = [1 2 3; 4 5 6], L = [1 2 3 4; 5 6 7 8], b = [1, 2])
Regression
  M ┼ 2×3 Matrix{Int64}
  L ┼ 2×4 Matrix{Int64}
  b ┴ Vector{Int64}: [1, 2]
```

# Related

  - [`AbstractRegressionResult`](@ref)
  - [`StepwiseRegression`](@ref)
  - [`DimensionReductionRegression`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 4.1, Equations 4.2-4.3.
"""
@concrete struct Regression <: AbstractRegressionResult
    """
    $(arg_dict[:M])
    """
    M
    """
    $(arg_dict[:L])
    """
    L
    """
    $(arg_dict[:b])
    """
    b
    function Regression(M::MatNum, L::Option{<:MatNum}, b::Option{<:VecNum})
        @argcheck(!isempty(M), IsEmptyError)
        if isa(b, VecNum)
            @argcheck(!isempty(b), IsEmptyError)
            @argcheck(length(b) == size(M, 1), DimensionMismatch)
        end
        if !isnothing(L)
            @argcheck(size(L, 1) == size(M, 1), DimensionMismatch)
        end
        return new{typeof(M), typeof(L), typeof(b)}(M, L, b)
    end
end
function Regression(; M::MatNum, L::Option{<:MatNum} = nothing,
                    b::Option{<:VecNum} = nothing)::Regression
    return Regression(M, L, b)
end
# When `L` is unset (`Nothing` type parameter), `:L` falls back to the loadings matrix `M`;
# when `L` is a stored matrix the default field access already returns it, so only the
# `Nothing` specialisation needs a rule (see [`@forward_properties`](@ref)'s `swap`).
@forward_properties Regression{<:Any, Nothing, <:Any} begin
    swap(L, M)
end
"""
    port_opt_view(re::Regression, i)

Return a view of a [`Regression`](@ref) result object, selecting only the rows indexed by `i`.

This function constructs a new `Regression` result, where the coefficient matrix `M`, optional auxiliary matrix `L`, and intercept vector `b` are restricted to the rows specified by the index vector `i`. This is useful for extracting or operating on a subset of regression results, such as for a subset of assets.

# Arguments

  - `re`: A regression result object.
  - `i`: Indices of the rows to select.

# Returns

  - `reg::Regression`: A new regression result object with fields restricted to the selected rows.

# Examples

```jldoctest
julia> re = Regression(; M = [1 2; 3 4; 5 6], L = [10 20; 30 40; 50 60], b = [7, 8, 9])
Regression
  M ┼ 3×2 Matrix{Int64}
  L ┼ 3×2 Matrix{Int64}
  b ┴ Vector{Int64}: [7, 8, 9]

julia> PortfolioOptimisers.port_opt_view(re, [1, 3])
Regression
  M ┼ 2×2 SubArray{Int64, 2, Matrix{Int64}, Tuple{Vector{Int64}, Base.Slice{Base.OneTo{Int64}}}, false}
  L ┼ 2×2 SubArray{Int64, 2, Matrix{Int64}, Tuple{Vector{Int64}, Base.Slice{Base.OneTo{Int64}}}, false}
  b ┴ SubArray{Int64, 1, Vector{Int64}, Tuple{Vector{Int64}}, false}: [7, 9]
```

# Related

  - [`Regression`](@ref)
"""
function port_opt_view(re::Regression, i, args...)::Regression
    # `L` and `b` are both optional, and `L` must be read with `getfield`: the
    # `swap(L, M)` property rule above makes `re.L` return `re.M` when `L` is unset, so
    # `isnothing(re.L)` is never true and a viewed result would materialise `L` as a copy
    # of `M`, silently losing the unset-ness the rule exists to express.
    L = getfield(re, :L)
    b = getfield(re, :b)
    return Regression(; M = view(re.M, i, :), L = isnothing(L) ? nothing : view(L, i, :),
                      b = isnothing(b) ? nothing : view(b, i))
end
"""
    regression(re::Regression, args...)

Return the regression result object unchanged.

This method is a pass-through for [`Regression`](@ref) result objects, allowing generic code to call `regression` on a result and receive the same object. It enables a unified interface for both estimator and result types.

# Arguments

  - `re`: A regression result object.
  - `args...`: Additional arguments (ignored).

# Returns

  - The input `re`, unchanged.

# Related

  - [`Regression`](@ref)
"""
function regression(re::Regression, args...)
    return re
end
"""
    regression(re::AbstractRegressionEstimator, rd::ReturnsResult)

Compute or extract a regression result from an estimator or result and a [`ReturnsResult`](@ref).

This method dispatches to `regression(re, rd.X, rd.F)`, allowing both regression estimators and regression result objects to be used interchangeably in generic workflows. If `re` is an estimator, it computes the regression result using the data in `rd`. If `re` is already a result, it is returned unchanged.

# Arguments

  - `re`: A regression estimator or result object.
  - `rd`: A returns result object containing data matrices `X` and `F`.

# Returns

  - `reg::Regression`: The computed or extracted regression result.

# Related

  - [`Regression`](@ref)
  - [`ReturnsResult`](@ref)
"""
function regression(re::AbstractRegressionEstimator, rd::ReturnsResult)
    @argcheck(!isnothing(rd.X), IsNothingError)
    @argcheck(!isnothing(rd.F), IsNothingError)
    return regression(re, rd.X, rd.F)
end

export regression, Regression, LinearModel, GeneralisedLinearModel, AIC, AICC, BIC,
       RSquared, AdjustedRSquared
