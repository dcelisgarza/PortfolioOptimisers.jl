"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of every regression estimator, over both the time-series family and the cross-sectional family.

The type is an umbrella and declares no interface of its own, because the two families fit different models and answer different verbs. A time-series estimator fits one model per asset over the observations and answers [`regression`](@ref). A cross-sectional estimator fits one model per observation across the assets and answers [`cross_sectional_regression`](@ref). Subtype the child that names the family, never this root, so a consumer of one family never receives a value of the other.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractTimeSeriesRegressionEstimator`](@ref)
  - [`AbstractCrossSectionalRegressionEstimator`](@ref)
  - [`AbstractRegressionAlgorithm`](@ref)
  - [`AbstractRegressionResult`](@ref)
"""
abstract type AbstractRegressionEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all time-series regression estimator types.

All concrete and/or abstract types implementing regression estimation algorithms that fit one model per asset over the observations should be subtypes of `AbstractTimeSeriesRegressionEstimator`.

# Interfaces

In order to implement a new time-series regression estimator which will work seamlessly with the library, subtype `AbstractTimeSeriesRegressionEstimator` with all necessary parameters as part of the struct, and implement the following methods:

## Regression

  - `PortfolioOptimisers.regression(re::AbstractTimeSeriesRegressionEstimator, X::MatNum, F::MatNum) -> Regression`: Computes the regression result from asset returns `X` and factor returns `F`.

### Arguments

  - `re`: Regression estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:F])

### Returns

  - `reg::Regression`: Regression result containing the coefficient matrix and optional intercept.

# Examples

We can create a dummy regression estimator as follows:

```jldoctest
julia> struct MyRegressionEstimator <: PortfolioOptimisers.AbstractTimeSeriesRegressionEstimator end

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

  - [`AbstractRegressionEstimator`](@ref)
  - [`AbstractCrossSectionalRegressionEstimator`](@ref)
  - [`AbstractTimeSeriesRegressionResult`](@ref)
  - [`StepwiseRegression`](@ref)
  - [`DimensionReductionRegression`](@ref)
"""
abstract type AbstractTimeSeriesRegressionEstimator <: AbstractRegressionEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all cross-sectional regression estimator types.

All concrete and/or abstract types implementing regression estimation algorithms that fit one model per observation across the assets should be subtypes of `AbstractCrossSectionalRegressionEstimator`.

# Interfaces

In order to implement a new cross-sectional regression estimator which will work seamlessly with the library, subtype `AbstractCrossSectionalRegressionEstimator` with all necessary parameters as part of the struct, and implement the following methods:

## Cross-sectional regression

  - `PortfolioOptimisers.cross_sectional_regression(cre::AbstractCrossSectionalRegressionEstimator, Z::Arr3Num, X::MatNum, W::MatNum) -> CrossSectionalRegression`: Computes the cross-sectional regression result from the exposure tensor `Z`, the asset returns `X` and the cross-sectional weights `W`.

### Arguments

  - `cre`: Cross-sectional regression estimator.
  - `Z`: Exposure tensor `observations × assets × factors`.
  - `X`: Asset returns matrix `observations × assets`.
  - `W`: Cross-sectional weights matrix `observations × assets`.

### Returns

  - `csr::CrossSectionalRegression`: Cross-sectional regression result carrying the factor returns, the residuals, the counts and the optional intercept.

# Examples

We can create a dummy cross-sectional regression estimator as follows:

```jldoctest
julia> struct MyCrossSectionalRegressionEstimator <:
              PortfolioOptimisers.AbstractCrossSectionalRegressionEstimator end

julia> function PortfolioOptimisers.cross_sectional_regression(::MyCrossSectionalRegressionEstimator,
                                                               Z::PortfolioOptimisers.Arr3Num,
                                                               X::PortfolioOptimisers.MatNum,
                                                               W::PortfolioOptimisers.MatNum)
           f = permutedims(reduce(hcat, Z[t, :, :] \\ X[t, :] for t in axes(X, 1)))
           eps = X - permutedims(reduce(hcat, Z[t, :, :] * f[t, :] for t in axes(X, 1)))
           return PortfolioOptimisers.CrossSectionalRegression(; f = f, eps = eps,
                                                               n = fill(size(X, 2), size(X, 1)))
       end

julia> cross_sectional_regression(MyCrossSectionalRegressionEstimator(),
                                  reshape([1.0, 0.0, 0.5, 0.0, 1.0, 0.5], 1, 3, 2), [1.0 2.0 1.5],
                                  ones(1, 3))
CrossSectionalRegression
    f ┼ 1×2 Matrix{Float64}
  eps ┼ 1×3 Matrix{Float64}
    n ┼ Vector{Int64}: [3]
    b ┴ nothing
```

# Related

  - [`AbstractRegressionEstimator`](@ref)
  - [`AbstractTimeSeriesRegressionEstimator`](@ref)
  - [`AbstractCrossSectionalRegressionResult`](@ref)
  - [`CrossSectionalLinearRegression`](@ref)
  - [`CrossSectionalTargetRegression`](@ref)
"""
abstract type AbstractCrossSectionalRegressionEstimator <: AbstractRegressionEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of every regression result, over both the time-series family and the cross-sectional family.

The type is an umbrella, and the two families disagree on what an asset index means. A time-series result holds one row per asset, so [`port_opt_view`](@ref) slices its rows. A cross-sectional result holds one row per observation and one column per asset, so the same index slices its columns. Subtype the child that names the family, never this root.

# Related

  - [`AbstractResult`](@ref)
  - [`AbstractTimeSeriesRegressionResult`](@ref)
  - [`AbstractCrossSectionalRegressionResult`](@ref)
  - [`AbstractRegressionEstimator`](@ref)
"""
abstract type AbstractRegressionResult <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all time-series regression result types.

All concrete and/or abstract types representing the output of a regression fitted per asset over the observations should be subtypes of `AbstractTimeSeriesRegressionResult`. A member carries the loadings matrix `M`, so every consumer that re-bases a constraint or decomposes risk in the factor basis binds this type rather than the umbrella.

# Related

  - [`AbstractRegressionResult`](@ref)
  - [`AbstractCrossSectionalRegressionResult`](@ref)
  - [`AbstractTimeSeriesRegressionEstimator`](@ref)
  - [`Regression`](@ref)
"""
abstract type AbstractTimeSeriesRegressionResult <: AbstractRegressionResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all cross-sectional regression result types.

All concrete and/or abstract types representing the output of a regression fitted per observation across the assets should be subtypes of `AbstractCrossSectionalRegressionResult`. A member carries no loadings matrix, because the exposures are the regression's input and an Exposure Estimator produces them.

# Related

  - [`AbstractRegressionResult`](@ref)
  - [`AbstractTimeSeriesRegressionResult`](@ref)
  - [`AbstractCrossSectionalRegressionEstimator`](@ref)
  - [`CrossSectionalRegression`](@ref)
"""
abstract type AbstractCrossSectionalRegressionResult <: AbstractRegressionResult end
"""
    const RegE_Reg = Union{<:AbstractTimeSeriesRegressionResult,
                           <:AbstractTimeSeriesRegressionEstimator}

Alias for a time-series regression result or estimator.

Matches either an [`AbstractTimeSeriesRegressionResult`](@ref) (pre-computed regression result) or an [`AbstractTimeSeriesRegressionEstimator`](@ref) (regression specification). Used for dispatch in factor model and regression-based risk routines. It names the time-series pair rather than the umbrella, because every consumer of the alias reads the loadings matrix `M`, which only a time-series result carries.

# Related

  - [`AbstractTimeSeriesRegressionResult`](@ref)
  - [`AbstractTimeSeriesRegressionEstimator`](@ref)
  - [`AbstractRegressionResult`](@ref)
  - [`AbstractRegressionEstimator`](@ref)
"""
const RegE_Reg = Union{<:AbstractTimeSeriesRegressionResult,
                       <:AbstractTimeSeriesRegressionEstimator}
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
  - [`MinMaxValStepwiseRegressionCriterion`](@ref)
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

# Algorithm

 1. Merge `w` into `re.kwargs` under the key `weights`, replacing any entry already stored there, giving the keyword arguments of the new target.
 2. Build a new [`LinearModel`](@ref) from them.

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

# Algorithm

 1. Read `tgt.kwargs`. When it carries a `weights` entry holding a [`DynamicAbstractWeights`](@ref), resolve that entry against `X` with [`get_observation_weights`](@ref) and write the resolved weights back under the same key, giving `kwargs`. Otherwise take `tgt.kwargs` unchanged.
 2. Call `StatsAPI.fit(GLM.LinearModel, X, y; kwargs...)`, giving the fitted model.

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
    PSEUDO_R2_VARIANTS

Tuple of the pseudo-``R^2`` variants `StatsAPI.r2` accepts for a fitted [`GeneralisedLinearModel`](@ref).

The members are `:McFadden`, `:CoxSnell`, `:Nagelkerke` and `:devianceratio`. The `variant` field of [`GeneralisedLinearModel`](@ref) is checked against this tuple at construction. A generalised linear model has no classical ``R^2``, so each member scores the fitted model against the intercept-only model of the same family instead.

# Mathematical definition

```math
\\begin{align}
R^2_{\\mathrm{McF}} &= 1 - \\frac{\\ln\\hat{L}}{\\ln\\hat{L}_{0}}\\,,\\\\
R^2_{\\mathrm{CS}} &= 1 - \\left(\\frac{\\hat{L}_{0}}{\\hat{L}}\\right)^{2/T}\\,,\\\\
R^2_{\\mathrm{N}} &= \\frac{R^2_{\\mathrm{CS}}}{1 - \\hat{L}_{0}^{\\,2/T}}\\,,\\\\
R^2_{\\mathrm{dev}} &= 1 - \\frac{D}{D_{0}}\\,.
\\end{align}
```

Where:

  - ``R^2_{\\mathrm{McF}}``: `:McFadden`.
  - ``R^2_{\\mathrm{CS}}``: `:CoxSnell`.
  - ``R^2_{\\mathrm{N}}``: `:Nagelkerke`.
  - ``R^2_{\\mathrm{dev}}``: `:devianceratio`.
  - ``\\hat{L}``: Maximum likelihood of the fitted model.
  - ``\\hat{L}_{0}``: Maximum likelihood of the intercept-only model of the same family.
  - ``D``: Deviance of the fitted model.
  - ``D_{0}``: Deviance of the intercept-only model of the same family.
  - $(math_dict[:T])

Three consequences follow for the Normal family, which is the default of [`GeneralisedLinearModel`](@ref). Its deviance is the residual sum of squares, so ``R^2_{\\mathrm{dev}}`` is the classical ``R^2`` of the same fit. Its maximum likelihood carries the fitted dispersion, so ``\\left(\\hat{L}_{0}/\\hat{L}\\right)^{2/T} = D/D_{0}`` and ``R^2_{\\mathrm{CS}}`` equals ``R^2_{\\mathrm{dev}}`` exactly. Its likelihood is a density rather than a probability, so ``\\hat{L}_{0}`` sits on either side of one and the two forms built on the log-likelihood, ``R^2_{\\mathrm{McF}}`` and ``R^2_{\\mathrm{N}}``, leave ``[0, 1]`` in either direction: rescaling the response alone moves both from above one to below zero, while ``R^2_{\\mathrm{dev}}`` does not move at all. Only ``R^2_{\\mathrm{dev}}`` is continuous with the [`LinearModel`](@ref) path, which is why [`default_regression_criterion_variant`](@ref) returns `:devianceratio`.

# Related

  - [`ADJUSTED_PSEUDO_R2_VARIANTS`](@ref)
  - [`GeneralisedLinearModel`](@ref)
  - [`default_regression_criterion_variant`](@ref)
  - [`regression_criterion_func`](@ref)

# References

  - $(ref_dict[:mcfadden1974])
  - $(ref_dict[:coxsnell1989])
  - $(ref_dict[:nagelkerke1991])
  - $(ref_dict[:nelder1972])
"""
const PSEUDO_R2_VARIANTS = (:McFadden, :CoxSnell, :Nagelkerke, :devianceratio)
"""
    ADJUSTED_PSEUDO_R2_VARIANTS

Tuple of the pseudo-``R^2`` variants `StatsAPI.adjr2` accepts for a fitted [`GeneralisedLinearModel`](@ref).

The members are `:McFadden` and `:devianceratio`, a strict subset of [`PSEUDO_R2_VARIANTS`](@ref). [`GeneralisedLinearModel`](@ref) cannot check against this tuple, because it does not know which criterion will read its `variant`. [`StepwiseRegression`](@ref) checks it instead: it is the first type that holds the criterion and the target together. `StatsAPI.adjr2` raises an `ArgumentError` on either variant this tuple omits.

# Mathematical definition

Each member discounts its unadjusted form of [`PSEUDO_R2_VARIANTS`](@ref) by the parameters the model consumes.

```math
\\begin{align}
\\bar{R}^2_{\\mathrm{McF}} &= 1 - \\frac{\\ln\\hat{L} - k}{\\ln\\hat{L}_{0}}\\,,\\\\
\\bar{R}^2_{\\mathrm{dev}} &= 1 - \\frac{D\\,(T - 1)}{D_{0}\\,(T - k)}\\,.
\\end{align}
```

Where:

  - ``\\bar{R}^2_{\\mathrm{McF}}``: `:McFadden`.
  - ``\\bar{R}^2_{\\mathrm{dev}}``: `:devianceratio`.
  - ``\\hat{L}``: Maximum likelihood of the fitted model.
  - ``\\hat{L}_{0}``: Maximum likelihood of the intercept-only model of the same family.
  - ``D``: Deviance of the fitted model.
  - ``D_{0}``: Deviance of the intercept-only model of the same family.
  - ``k``: Number of estimated parameters, which is `StatsAPI.dof` of the fitted model: the regression coefficients, the intercept, and the dispersion.
  - $(math_dict[:T])

The ``k`` here is the one `:aic`, `:aicc` and `:bic` read, not the predictor count `:adjr2` reads on a fitted [`LinearModel`](@ref). See [`STEPWISE_REGRESSION_CRITERIA`](@ref), which states both.

# Related

  - [`PSEUDO_R2_VARIANTS`](@ref)
  - [`GeneralisedLinearModel`](@ref)
  - [`STEPWISE_REGRESSION_CRITERIA`](@ref)
  - [`StepwiseRegression`](@ref)

# References

  - $(ref_dict[:mcfadden1974])
  - $(ref_dict[:nelder1972])
"""
const ADJUSTED_PSEUDO_R2_VARIANTS = (:McFadden, :devianceratio)
"""
$(DocStringExtensions.TYPEDEF)

Fits each response by a generalised linear model through `GLM.GeneralizedLinearModel`.

The `args` field carries the response distribution and, optionally, the link function; `kwargs` carries the remaining `GLM` options. The default `args = (Normal(),)` with the canonical identity link reproduces ordinary least squares. `GLM` defines ``R^2`` for a fitted [`LinearModel`](@ref) only, so `variant` names the pseudo-``R^2`` a maximisation criterion reads instead, and it supplies it to the `:r2` and `:adjr2` members of [`STEPWISE_REGRESSION_CRITERIA`](@ref). A `nothing` `variant` takes the default of the criterion, which [`default_regression_criterion_variant`](@ref) states. The field is dead under a minimisation criterion, which reads no variant at all.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GeneralisedLinearModel(;
        args::Tuple = (Normal(),),
        kwargs::NamedTuple = (;),
        variant::Option{Symbol} = nothing
    ) -> GeneralisedLinearModel

Keywords correspond to the struct's fields.

## Validation

  - If provided, `variant in PSEUDO_R2_VARIANTS`, the wider of the two variant tuples. `StatsAPI.adjr2` accepts [`ADJUSTED_PSEUDO_R2_VARIANTS`](@ref) alone, and [`StepwiseRegression`](@ref) rejects the difference when its criterion is `:adjr2`.

# Examples

```jldoctest
julia> GeneralisedLinearModel()
GeneralisedLinearModel
     args ┼ Tuple{Distributions.Normal{Float64}}: (Distributions.Normal{Float64}(μ=0.0, σ=1.0),)
   kwargs ┼ @NamedTuple{}: NamedTuple()
  variant ┴ nothing
```

# Related

  - [`AbstractRegressionTarget`](@ref)
  - [`LinearModel`](@ref)
  - [`PSEUDO_R2_VARIANTS`](@ref)
  - [`ADJUSTED_PSEUDO_R2_VARIANTS`](@ref)
  - [`STEPWISE_REGRESSION_CRITERIA`](@ref)
  - [`StepwiseRegression`](@ref)
  - [`default_regression_criterion_variant`](@ref)
  - [`regression_criterion_func`](@ref)
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
    """
    $(field_dict[:r2variant])
    """
    variant
    function GeneralisedLinearModel(args::Tuple, kwargs::NamedTuple,
                                    variant::Option{Symbol})
        if !isnothing(variant)
            @argcheck(variant in PSEUDO_R2_VARIANTS,
                      "variant must be one of $PSEUDO_R2_VARIANTS. Got\nvariant => $variant")
        end
        return new{typeof(args), typeof(kwargs), typeof(variant)}(args, kwargs, variant)
    end
end
function GeneralisedLinearModel(; args::Tuple = (Distributions.Normal(),),
                                kwargs::NamedTuple = (;),
                                variant::Option{Symbol} = nothing)::GeneralisedLinearModel
    return GeneralisedLinearModel(args, kwargs, variant)
end
"""
    factory(re::GeneralisedLinearModel, w::ObsWeights) -> GeneralisedLinearModel

Return a new [`GeneralisedLinearModel`](@ref) regression target with observation weights `w` added to the keyword arguments.

# Algorithm

 1. Merge `w` into `re.kwargs` under the key `weights`, replacing any entry already stored there, giving the keyword arguments of the new target.
 2. Build a new [`GeneralisedLinearModel`](@ref) from them, carrying `re.args` and `re.variant` across unchanged.

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
    return GeneralisedLinearModel(; args = re.args, kwargs = (; re.kwargs..., weights = w),
                                  variant = re.variant)
end
"""
    StatsAPI.fit(tgt::GeneralisedLinearModel, X::MatNum, y::VecNum)

Fit a generalised linear regression model using a [`GeneralisedLinearModel`](@ref) regression target.

This method dispatches to `StatsAPI.fit` with the `GLM.GeneralizedLinearModel` type, passing the design matrix `X`, response vector `y`, any positional arguments in `tgt.args`, and any keyword arguments in `tgt.kwargs`.

# Algorithm

 1. Read `tgt.kwargs`. When it carries a `weights` entry holding a [`DynamicAbstractWeights`](@ref), resolve that entry against `X` with [`get_observation_weights`](@ref) and write the resolved weights back under the same key, giving `kwargs`. Otherwise take `tgt.kwargs` unchanged.
 2. Call `StatsAPI.fit(GLM.GeneralizedLinearModel, X, y, tgt.args...; kwargs...)`, giving the fitted model.

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
    MIN_VAL_STEPWISE_REGRESSION_CRITERIA

Tuple of the symbols naming a stepwise regression criterion that a lower value scores better.

The members are `:aic`, `:aicc` and `:bic`. [`MinValStepwiseRegressionCriterion`](@ref) is built from this tuple, and [`STEPWISE_REGRESSION_CRITERIA`](@ref) documents what each symbol computes.

# Related

  - [`MAX_VAL_STEPWISE_REGRESSION_CRITERIA`](@ref)
  - [`MinValStepwiseRegressionCriterion`](@ref)
  - [`STEPWISE_REGRESSION_CRITERIA`](@ref)
"""
const MIN_VAL_STEPWISE_REGRESSION_CRITERIA = (:aic, :aicc, :bic)
"""
    MAX_VAL_STEPWISE_REGRESSION_CRITERIA

Tuple of the symbols naming a stepwise regression criterion that a higher value scores better.

The members are `:r2` and `:adjr2`. [`MaxValStepwiseRegressionCriterion`](@ref) is built from this tuple, and [`STEPWISE_REGRESSION_CRITERIA`](@ref) documents what each symbol computes.

# Related

  - [`MIN_VAL_STEPWISE_REGRESSION_CRITERIA`](@ref)
  - [`MaxValStepwiseRegressionCriterion`](@ref)
  - [`STEPWISE_REGRESSION_CRITERIA`](@ref)
"""
const MAX_VAL_STEPWISE_REGRESSION_CRITERIA = (:r2, :adjr2)
"""
    STEPWISE_REGRESSION_CRITERIA

Tuple of the symbols that name a stepwise regression criterion scoring a fitted model with one number.

[`StepwiseRegression`](@ref) accepts any symbol of this tuple in its `crit` field and stores it as a `Val`, which is what [`regression_criterion_func`](@ref), [`regression_threshold`](@ref) and the `get_*_reg_incl*!` helpers dispatch on. A symbol outside the tuple is rejected at construction. [`PValue`](@ref) is not a member: it reads the coefficient p-values of the fitted model instead of one score, so it stays a type and takes its own stepwise methods. `:aic`, `:aicc` and `:bic` score a fitted [`LinearModel`](@ref) and a fitted [`GeneralisedLinearModel`](@ref) alike, while `:r2` and `:adjr2` are defined for a fitted [`LinearModel`](@ref) only and read a named pseudo-``R^2`` variant under the other target.

# Mathematical definition

## `:aic` — Akaike Information Criterion

Trades the fitted likelihood against the number of estimated parameters, so a lower value is a better model.

```math
\\begin{align}
\\mathrm{AIC} &= 2k - 2\\ln\\hat{L}\\,.
\\end{align}
```

Where:

  - ``k``: Number of estimated parameters, which is `StatsAPI.dof` of the fitted model: the regression coefficients, the intercept, and the residual variance.
  - ``\\hat{L}``: Maximum likelihood of the model.

## `:aicc` — Akaike Information Criterion corrected for a small sample

The correction term grows as ``T`` approaches ``k``, so `:aicc` penalises a large model more heavily than `:aic` does on a short sample.

```math
\\begin{align}
\\mathrm{AICC} &= \\mathrm{AIC} + \\frac{2k(k+1)}{T - k - 1}\\,.
\\end{align}
```

Where:

  - ``k``: Number of estimated parameters, as for `:aic`.
  - $(math_dict[:T])

The correction term divides by ``T - k - 1``. The criterion is undefined when ``T = k + 1`` and changes sign below it, so `:aicc` needs a sample longer than the largest model the search can reach.

## `:bic` — Bayesian Information Criterion

The penalty is ``k \\ln T`` rather than ``2k``, which is heavier than the penalty of `:aic` for ``T \\geq 8``, so the search usually stops with fewer factors.

```math
\\begin{align}
\\mathrm{BIC} &= k\\ln T - 2\\ln\\hat{L}\\,.
\\end{align}
```

Where:

  - ``k``: Number of estimated parameters, as for `:aic`.
  - $(math_dict[:T])
  - ``\\hat{L}``: Maximum likelihood of the model.

## `:r2` — coefficient of determination

The share of the response variance the model explains, so a higher value is a better fit.

```math
\\begin{align}
R^2 &= 1 - \\frac{\\mathrm{SS}_{\\mathrm{res}}}{\\mathrm{SS}_{\\mathrm{tot}}} = 1 - \\frac{\\sum_t (y_t - \\hat{y}_t)^2}{\\sum_t (y_t - \\bar{y})^2}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{SS}_{\\mathrm{res}}``: Residual sum of squares.
  - ``\\mathrm{SS}_{\\mathrm{tot}}``: Total sum of squares.
  - ``y_t``: Observed response at time ``t``.
  - ``\\hat{y}_t``: Fitted response at time ``t``.
  - ``\\bar{y}``: Mean of observed responses.

``R^2`` never falls when a factor is added, so it penalises no complexity at all. Under [`ForwardSelection`](@ref) it admits **every** factor and under [`BackwardElimination`](@ref) it removes **none**. On a 200×5 sample where `:aic` kept factors 1, 3 and 4 and `:bic` kept 1 and 3, `:r2` kept all five in both directions. Use `:adjr2`, `:aic`, `:aicc` or `:bic` when the criterion must pay for size.

## `:adjr2` — coefficient of determination adjusted for the model size

The adjustment discounts ``R^2`` by the degrees of freedom the predictors consume, so unlike `:r2` the score can fall when a factor is added. This makes it usable as a stopping rule.

```math
\\begin{align}
\\bar{R}^2 &= 1 - (1 - R^2) \\frac{T - 1}{T - k - 1}\\,.
\\end{align}
```

Where:

  - ``k``: Number of predictors, excluding the intercept. `:aic`, `:aicc` and `:bic` write ``k`` for a different count: the predictors, plus the intercept, plus the residual variance.
  - $(math_dict[:T])

# Related

  - [`MinValStepwiseRegressionCriterion`](@ref)
  - [`MaxValStepwiseRegressionCriterion`](@ref)
  - [`StepwiseRegression`](@ref)
  - [`PSEUDO_R2_VARIANTS`](@ref)
  - [`regression_criterion_func`](@ref)
  - [`default_regression_criterion_variant`](@ref)
  - [`regression_threshold`](@ref)
  - [`PValue`](@ref)

# References

  - $(ref_dict[:akaike1974])
  - $(ref_dict[:hurvich1989])
  - $(ref_dict[:schwarz1978])
  - $(ref_dict[:hocking1976])
  - $(ref_dict[:theil1961])
"""
const STEPWISE_REGRESSION_CRITERIA = (MIN_VAL_STEPWISE_REGRESSION_CRITERIA...,
                                      MAX_VAL_STEPWISE_REGRESSION_CRITERIA...)
"""
    MinValStepwiseRegressionCriterion

Union of the `Val` types naming a stepwise regression criterion that a lower value scores better.

The members are built from [`MIN_VAL_STEPWISE_REGRESSION_CRITERIA`](@ref), so the union is `Union{Val{:aic}, Val{:aicc}, Val{:bic}}`. [`StepwiseRegression`](@ref) stores `Val(crit)` in its `crit` field, so a member of this union is the first type parameter of the estimator whenever the criterion minimises.

# Related

  - [`MIN_VAL_STEPWISE_REGRESSION_CRITERIA`](@ref)
  - [`MaxValStepwiseRegressionCriterion`](@ref)
  - [`MinMaxValStepwiseRegressionCriterion`](@ref)
  - [`STEPWISE_REGRESSION_CRITERIA`](@ref)
  - [`regression_threshold`](@ref)
"""
const MinValStepwiseRegressionCriterion = Union{map(c -> Val{c},
                                                    MIN_VAL_STEPWISE_REGRESSION_CRITERIA)...}
"""
    MaxValStepwiseRegressionCriterion

Union of the `Val` types naming a stepwise regression criterion that a higher value scores better.

The members are built from [`MAX_VAL_STEPWISE_REGRESSION_CRITERIA`](@ref), so the union is `Union{Val{:r2}, Val{:adjr2}}`. Both members read a pseudo-``R^2`` variant under a [`GeneralisedLinearModel`](@ref) target.

# Related

  - [`MAX_VAL_STEPWISE_REGRESSION_CRITERIA`](@ref)
  - [`MinValStepwiseRegressionCriterion`](@ref)
  - [`MinMaxValStepwiseRegressionCriterion`](@ref)
  - [`STEPWISE_REGRESSION_CRITERIA`](@ref)
  - [`default_regression_criterion_variant`](@ref)
  - [`regression_threshold`](@ref)
"""
const MaxValStepwiseRegressionCriterion = Union{map(c -> Val{c},
                                                    MAX_VAL_STEPWISE_REGRESSION_CRITERIA)...}
"""
    MinMaxValStepwiseRegressionCriterion

Union of every `Val` type naming a stepwise regression criterion that scores a fitted model with one number.

This is the union of [`MinValStepwiseRegressionCriterion`](@ref) and [`MaxValStepwiseRegressionCriterion`](@ref). It is the bound the `crit` field of [`StepwiseRegression`](@ref) takes when the criterion is one score rather than a [`PValue`](@ref).

# Related

  - [`MinValStepwiseRegressionCriterion`](@ref)
  - [`MaxValStepwiseRegressionCriterion`](@ref)
  - [`STEPWISE_REGRESSION_CRITERIA`](@ref)
  - [`StepwiseRegression`](@ref)
"""
const MinMaxValStepwiseRegressionCriterion = Union{MinValStepwiseRegressionCriterion,
                                                   MaxValStepwiseRegressionCriterion}
"""
    default_regression_criterion_variant(crit::MaxValStepwiseRegressionCriterion)

Return the pseudo-``R^2`` variant a maximisation criterion reads when the target names none.

`:devianceratio` is the only member of [`PSEUDO_R2_VARIANTS`](@ref) that both `StatsAPI.r2` and `StatsAPI.adjr2` accept and that also reproduces the classical ``R^2`` of a fitted [`LinearModel`](@ref) on a Normal family. The default therefore keeps the score continuous with the [`LinearModel`](@ref) path under either criterion.

# Arguments

  - `crit`: Maximisation criterion, as the `Val` the `crit` field of [`StepwiseRegression`](@ref) holds.

# Returns

  - `variant::Symbol`: `:devianceratio`.

# Related

  - [`MaxValStepwiseRegressionCriterion`](@ref)
  - [`GeneralisedLinearModel`](@ref) — a target whose `variant` is not `nothing` overrides this default.
  - [`regression_criterion_func`](@ref) — the only caller, and it reads the default only when the target names no variant.
  - [`PSEUDO_R2_VARIANTS`](@ref)
  - [`ADJUSTED_PSEUDO_R2_VARIANTS`](@ref)
"""
function default_regression_criterion_variant(::MaxValStepwiseRegressionCriterion)
    return :devianceratio
end
"""
    regression_criterion_func(crit::MinMaxValStepwiseRegressionCriterion,
                              tgt::AbstractRegressionTarget)

Return the function that scores a fitted model under a stepwise regression criterion.

The method dispatches on the `Val` naming the criterion and on the regression target, because the two maximisation criteria read a different quantity under each target. The map is:

| Criterion | [`LinearModel`](@ref) | [`GeneralisedLinearModel`](@ref)          |
|:--------- |:--------------------- |:----------------------------------------- |
| `:aic`    | `StatsAPI.aic`        | `StatsAPI.aic`                            |
| `:aicc`   | `StatsAPI.aicc`       | `StatsAPI.aicc`                           |
| `:bic`    | `StatsAPI.bic`        | `StatsAPI.bic`                            |
| `:r2`     | `StatsAPI.r2`         | `model -> StatsAPI.r2(model, variant)`    |
| `:adjr2`  | `StatsAPI.adjr2`      | `model -> StatsAPI.adjr2(model, variant)` |

`StatsAPI.aic`, `StatsAPI.aicc` and `StatsAPI.bic` accept a fitted model of either target, so the three minimisation criteria take one method each. `StatsAPI.r2` and `StatsAPI.adjr2` accept a fitted [`LinearModel`](@ref) without a variant, and a fitted [`GeneralisedLinearModel`](@ref) needs a named pseudo-``R^2`` variant, so the two maximisation criteria take two methods each. [`PValue`](@ref) has no method here: it reads the coefficient p-values of the fitted model rather than one score, so its stepwise methods are separate.

# Algorithm

The five methods that return a `StatsAPI` function run no steps. The two that close over a variant run these:

 1. Read `tgt.variant`. When it is `nothing`, take [`default_regression_criterion_variant`](@ref) of `crit` instead, giving `variant`.
 2. Build a closure over `variant` that calls `StatsAPI.r2(model, variant)`, or `StatsAPI.adjr2(model, variant)` under `:adjr2`.

# Arguments

  - `crit`: Criterion, as the `Val` the `crit` field of [`StepwiseRegression`](@ref) holds.
  - `tgt`: Regression target the candidate models are fitted with.

# Returns

  - `f::Function`: The function that computes the criterion value for a fitted model.

# Related

  - [`STEPWISE_REGRESSION_CRITERIA`](@ref)
  - [`MinMaxValStepwiseRegressionCriterion`](@ref)
  - [`default_regression_criterion_variant`](@ref)
  - [`GeneralisedLinearModel`](@ref)
  - [`StepwiseRegression`](@ref)
"""
function regression_criterion_func(::Val{:aic}, ::AbstractRegressionTarget)
    return StatsAPI.aic
end
function regression_criterion_func(::Val{:aicc}, ::AbstractRegressionTarget)
    return StatsAPI.aicc
end
function regression_criterion_func(::Val{:bic}, ::AbstractRegressionTarget)
    return StatsAPI.bic
end
function regression_criterion_func(::Val{:r2}, ::LinearModel)
    return StatsAPI.r2
end
function regression_criterion_func(::Val{:adjr2}, ::LinearModel)
    return StatsAPI.adjr2
end
function regression_criterion_func(crit::Val{:r2}, tgt::GeneralisedLinearModel)
    variant = ifelse(isnothing(tgt.variant), default_regression_criterion_variant(crit),
                     tgt.variant)
    return model -> StatsAPI.r2(model, variant)
end
function regression_criterion_func(crit::Val{:adjr2}, tgt::GeneralisedLinearModel)
    variant = ifelse(isnothing(tgt.variant), default_regression_criterion_variant(crit),
                     tgt.variant)
    return model -> StatsAPI.adjr2(model, variant)
end
"""
    regression_polarity(crit::MinMaxValStepwiseRegressionCriterion)

Return the three functions that state which direction of a stepwise criterion is better.

A stepwise search asks the same three questions of every criterion: which entry of a score vector is the best one, whether a candidate score improves on the score in hand, and what the worst score of a type is. Each answer is one function under a minimised criterion and its opposite under a maximised one. This is the only method pair in the library that states the pairing, so [`regression_threshold`](@ref), [`get_forward_reg_incl_excl!`](@ref), [`get_backward_reg_incl!`](@ref) and the two `_regression` methods that seed a score vector all read it rather than restate it.

# Arguments

  - `crit`: Criterion, as the `Val` the `crit` field of [`StepwiseRegression`](@ref) holds.

# Returns

  - `polarity::NamedTuple`: Three functions.

      + `best`: `findmin` under a minimised criterion, `findmax` under a maximised one. Returns the best entry of a score vector and its index.
      + `improves`: `<` under a minimised criterion, `>` under a maximised one. Answers whether the first score is better than the second.
      + `worst`: `typemax` under a minimised criterion, `typemin` under a maximised one. Returns the worst score of the type it is given.

# Related

  - [`MinValStepwiseRegressionCriterion`](@ref)
  - [`MaxValStepwiseRegressionCriterion`](@ref)
  - [`STEPWISE_REGRESSION_CRITERIA`](@ref)
  - [`regression_threshold`](@ref)
  - [`get_forward_reg_incl_excl!`](@ref)
  - [`get_backward_reg_incl!`](@ref)
"""
function regression_polarity(::MinValStepwiseRegressionCriterion)
    return (; best = findmin, improves = <, worst = typemax)
end
function regression_polarity(::MaxValStepwiseRegressionCriterion)
    return (; best = findmax, improves = >, worst = typemin)
end
"""
    regression_threshold(crit::MinMaxValStepwiseRegressionCriterion)

Return the starting threshold for a forward stepwise regression search.

The value is the worst score the criterion can take, so the first candidate model always improves on it. [`regression_polarity`](@ref) states which of `typemax` and `typemin` that is, and `typemax(Float64)` is `Inf`. Only [`ForwardSelection`](@ref) reads it: [`BackwardElimination`](@ref) starts from the score of the full model instead, because its first move must beat a model that already exists.

# Arguments

  - `crit`: Criterion, as the `Val` the `crit` field of [`StepwiseRegression`](@ref) holds.

# Returns

  - `t::Number`: `Inf` for a minimisation criterion, `-Inf` for a maximisation criterion.

# Related

  - [`regression_polarity`](@ref) — the pairing this method reads.
  - [`MinValStepwiseRegressionCriterion`](@ref)
  - [`MaxValStepwiseRegressionCriterion`](@ref)
  - [`STEPWISE_REGRESSION_CRITERIA`](@ref)
  - [`ForwardSelection`](@ref) — the only caller.
  - [`BackwardElimination`](@ref) — starts from the score of the full model, never from this value.
"""
function regression_threshold(crit::MinMaxValStepwiseRegressionCriterion)
    return regression_polarity(crit).worst(Float64)
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the loadings matrix, the intercept vector and the reduced-basis loadings of a fitted factor model.

`M` and `b` are the loadings matrix and the intercept vector of the factor model, one row per asset. `L` carries the same loadings written in the reduced basis a dimension reduction produced; it is unset when the estimator regresses on the original factors. **An unset `L` reads back as `M`.** A [`@forward_properties`](@ref) `swap(L, M)` rule makes `re.L` return `re.M` whenever `L` was not given, so a consumer that decomposes risk in the factor basis needs no `Nothing` branch, and `isnothing(re.L)` is never true. Read `getfield(re, :L)` when the unset case must be told apart, as [`port_opt_view`](@ref) does. [`StepwiseRegression`](@ref) leaves `L` unset and [`DimensionReductionRegression`](@ref) sets it, so `size(L, 2)` is the width of the basis risk is decomposed in: the original factors under the first, the retained principal components under the second.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{x}_{t} &= \\boldsymbol{b} + \\mathbf{M} \\boldsymbol{f}_{t} + \\boldsymbol{\\varepsilon}_{t}\\,.
\\end{align}
```

Where:

  - $(math_dict[:x_t_obs])
  - ``\\boldsymbol{b}``: Intercept vector ``N \\times 1``, `b`. The term is absent when `b` is unset.
  - ``\\mathbf{M}``: Loadings matrix ``N \\times K`` of the factor model, `M`.
  - ``\\boldsymbol{f}_{t}``: Factor returns for observation ``t``, the ``t``-th row of the factor matrix.
  - ``\\boldsymbol{\\varepsilon}_{t}``: Residual returns for observation ``t``, the part of ``\\boldsymbol{x}_{t}`` the factors do not explain.
  - $(math_dict[:N])
  - ``K``: Number of factors.

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
  - If provided, `!isempty(L)`, and `size(L, 1) == size(M, 1)`.

# Examples

```jldoctest
julia> Regression(; M = [1 2 3; 4 5 6], L = [1 2 3 4; 5 6 7 8], b = [1, 2])
Regression
  M ┼ 2×3 Matrix{Int64}
  L ┼ 2×4 Matrix{Int64}
  b ┴ Vector{Int64}: [1, 2]
```

# Related

  - [`AbstractTimeSeriesRegressionResult`](@ref)
  - [`StepwiseRegression`](@ref)
  - [`DimensionReductionRegression`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 4.1, Equations 4.2-4.3.
"""
@concrete struct Regression <: AbstractTimeSeriesRegressionResult
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
            @argcheck(!isempty(L), IsEmptyError)
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

# Algorithm

 1. Read `L` and `b` with `getfield`, never through property access. The `swap(L, M)` rule of [`Regression`](@ref) makes `re.L` return `re.M` when `L` is unset, so a property read would materialise `L` as a copy of `M` and lose the unset-ness.
 2. Take a row view of `M` over `i`, giving the loadings of the selected assets.
 3. Take a row view of `L` over `i` when step 1 found a matrix, and `nothing` otherwise.
 4. Take an element view of `b` over `i` when step 1 found a vector, and `nothing` otherwise.
 5. Build a new [`Regression`](@ref) from the three, which re-runs every guard of the constructor.

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
    regression(re::AbstractTimeSeriesRegressionEstimator, rd::ReturnsResult)

Compute or extract a regression result from an estimator or result and a [`ReturnsResult`](@ref).

This method dispatches to `regression(re, rd.X, rd.F)`, allowing both regression estimators and regression result objects to be used interchangeably in generic workflows. If `re` is an estimator, it computes the regression result using the data in `rd`. If `re` is already a result, it is returned unchanged.

# Algorithm

 1. Check that `rd` carries both matrices, per `# Validation` below.
 2. Call `regression(re, rd.X, rd.F)`, giving the regression result.

# Arguments

  - `re`: A regression estimator or result object.
  - `rd`: A returns result object containing data matrices `X` and `F`.

# Validation

  - `!isnothing(rd.X)`. A regression needs the asset returns it explains.
  - `!isnothing(rd.F)`. A regression needs the factor returns it explains them with.

# Returns

  - `reg::Regression`: The computed or extracted regression result.

# Related

  - [`Regression`](@ref)
  - [`ReturnsResult`](@ref)
"""
function regression(re::AbstractTimeSeriesRegressionEstimator, rd::ReturnsResult)
    @argcheck(!isnothing(rd.X), IsNothingError)
    @argcheck(!isnothing(rd.F), IsNothingError)
    return regression(re, rd.X, rd.F)
end

export regression, Regression, LinearModel, GeneralisedLinearModel
