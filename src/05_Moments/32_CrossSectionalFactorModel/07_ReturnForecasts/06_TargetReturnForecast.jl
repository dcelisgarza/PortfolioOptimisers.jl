"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the calibration warm-up of a [`TargetReturnForecast`](@ref), the rule that states the uncalibrated prediction the calibration reads while no prediction is out of fold.

A cross-validation estimator in `cv` needs two valid samples per fold, which is twice the split count that [`n_splits`](@ref) reports. Below that count no out-of-fold prediction exists, so no out-of-fold slope exists. [`NaNWarmup`](@ref) states that with a `NaN` slope. [`InSampleWarmup`](@ref) calibrates on the in-sample predictions of the fitted model, which gives a slope that is biased upward. The rule acts only below that count and only under a cross-validation estimator. [`PrequentialCalibration`](@ref), the default, states its own warm-up, so no warm-up rule acts under it.

# Interfaces

A warm-up is a marker for dispatch, and it holds no data. A new warm-up needs a method of [`target_forecast_warmup`](@ref).

# Related

  - [`NaNWarmup`](@ref)
  - [`InSampleWarmup`](@ref)
  - [`TargetReturnForecast`](@ref)
  - [`target_forecast_uncalibrated`](@ref)
"""
abstract type AbstractCalibrationWarmup <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Gives a `NaN` uncalibrated prediction to every sample while no prediction is out of fold, so the calibration coefficient is `NaN`. This is the default warm-up.

Below two valid samples per fold, no fold has a model that did not train on its test samples. So no out-of-fold slope exists, and `NaN` states that.

# Constructors

    NaNWarmup() -> NaNWarmup

# Examples

```jldoctest
julia> TargetReturnForecast(;
                            scores = DescriptorScores(;
                                                      descriptors = [Passthrough(; field = \"a\")])).warmup
NaNWarmup()
```

# Related

  - [`AbstractCalibrationWarmup`](@ref)
  - [`InSampleWarmup`](@ref)
"""
struct NaNWarmup <: AbstractCalibrationWarmup end
"""
$(DocStringExtensions.TYPEDEF)

Calibrates on the in-sample predictions of the fitted model while no prediction is out of fold.

The in-sample prediction of a sample comes from a model that trained on that sample, so it agrees with its own target by construction. Its slope is biased upward: on scores that carry no information the slope is zero, and the in-sample slope is positive. The rule gives a biased estimate, not a wrong formula, so it stays available as an option. The calibration under `cv = nothing` reads the same predictions at every sample count.

# Constructors

    InSampleWarmup() -> InSampleWarmup

# Examples

```jldoctest
julia> TargetReturnForecast(;
                            scores = DescriptorScores(;
                                                      descriptors = [Passthrough(; field = \"a\")]),
                            warmup = InSampleWarmup()).warmup
InSampleWarmup()
```

# Related

  - [`AbstractCalibrationWarmup`](@ref)
  - [`NaNWarmup`](@ref)
"""
struct InSampleWarmup <: AbstractCalibrationWarmup end
"""
    target_forecast_warmup(warmup::NaNWarmup, rfe::TargetReturnForecast, model,
                           Sf::MatNum, yf::VecNum, ok::AbstractVector{Bool}) -> VecNum
    target_forecast_warmup(warmup::InSampleWarmup, rfe::TargetReturnForecast, model,
                           Sf::MatNum, yf::VecNum, ok::AbstractVector{Bool}) -> VecNum

Return the uncalibrated prediction that the calibration of a [`TargetReturnForecast`](@ref) reads while no prediction is out of fold.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`NaNWarmup`](@ref): every sample reads `NaN`.
 2. [`InSampleWarmup`](@ref): the fitted model predicts its own valid samples, as [`target_forecast_uncalibrated`](@ref) does under `cv = nothing`, and every other sample reads `NaN`.

# Arguments

  - `warmup`: The calibration warm-up. It is `rfe.warmup`.
  - `rfe`: Target Return Forecast Estimator.
  - `model`: The model fitted on every valid sample.
  - `Sf`: The flattened design.
  - `yf`: The flattened target.
  - `ok`: The mask of the valid samples.

# Returns

  - `p::VecNum`: The uncalibrated prediction of every sample.

# Related

  - [`AbstractCalibrationWarmup`](@ref)
  - [`target_forecast_uncalibrated`](@ref)
"""
function target_forecast_warmup(::NaNWarmup, ::Any, ::Any, Sf::MatNum, ::VecNum,
                                ok::AbstractVector{Bool})::VecNum
    return fill(real(eltype(Sf))(NaN), length(ok))
end
function target_forecast_warmup(::InSampleWarmup, rfe, model, Sf::MatNum, yf::VecNum,
                                ok::AbstractVector{Bool})::VecNum
    return target_forecast_insample(model, Sf, ok)
end
"""
$(DocStringExtensions.TYPEDEF)

The prequential rule of the calibration of a [`TargetReturnForecast`](@ref). Each matured observation reads the prediction of the model fitted on the valid samples whose target had matured at that observation. This is the default rule.

The member forecasts observation ``t`` at ``t``. At ``t`` the target of observation ``s`` has matured when ``s \\leq t - \\ell - h + 1``, with ``\\ell`` the lag and ``h`` the horizon. So the prediction of observation ``t`` is the forecast that the member publishes at ``t``, before its calibration. No prediction reads a later observation, and the forward window of no training target overlaps the forward window of the observation that it predicts. A new observation leaves every earlier prediction as it stands, so the member can fold one observation at a time.

An observation predicts `NaN` while no valid sample matured before it, which is the warm-up of the member itself. The calibration then counts its own warm-up in `min_obs`.

A cross-validation estimator in `cv` gives a different reading. Its folds cut the samples in order of observation, so the model that predicts an early fold trains on later observations. Under `horizon > 1` the forward windows at the edge of a fold also overlap those of the next fold. A new sample moves every edge, so that rule cannot fold.

[`LinearModel`](@ref) with no keyword argument folds its fit. The member adds the valid samples of each observation to `X'X` and `X'y` and solves them, and the fitted model is a [`NormalEquationsFit`](@ref). Every other regression target fits again on the matured samples at each observation that adds one. This is exact, and it costs one fit for each observation.

# Constructors

    PrequentialCalibration() -> PrequentialCalibration

# Examples

```jldoctest
julia> TargetReturnForecast(;
                            scores = DescriptorScores(;
                                                      descriptors = [Passthrough(; field = \"a\")])).cv
PrequentialCalibration()
```

# Related

  - [`TargetReturnForecast`](@ref)
  - [`NormalEquationsFit`](@ref)
  - [`target_forecast_prequential`](@ref)
  - [`CrossValidationEstimator`](@ref)
"""
struct PrequentialCalibration <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

The least squares fit of a [`LinearModel`](@ref) with no keyword argument, solved from its normal equations. [`TargetReturnForecast`](@ref) fits it under [`PrequentialCalibration`](@ref).

The fit carries its sufficient statistics. The next observation adds its valid samples to `XtX` and `Xty`, and it reads nothing else of the past fit. The coefficients solve the normal equations through a Cholesky factorisation with pivots. A column that the factorisation finds collinear with the columns before it takes a zero coefficient, as `GLM.LinearModel` does under its default `dropcollinear = true`.

# Fields

$(DocStringExtensions.TYPEDFIELDS)

# Constructors

    NormalEquationsFit(XtX::MatNum, Xty::VecNum, n::Integer) -> NormalEquationsFit

The constructor solves the coefficients through [`normal_equations_coef`](@ref).

## Validation

  - `XtX` is square, and its side is the length of `Xty`. Raises a `DimensionMismatch`.
  - `n >= 0`.

# Examples

```jldoctest
julia> m = NormalEquationsFit([4.0 0.0; 0.0 16.0], [4.0, 8.0], 3);

julia> PortfolioOptimisers.StatsAPI.coef(m)
2-element Vector{Float64}:
 1.0
 0.5

julia> PortfolioOptimisers.StatsAPI.predict(m, [1.0 2.0])
1-element Vector{Float64}:
 2.0
```

# Related

  - [`PrequentialCalibration`](@ref)
  - [`LinearModel`](@ref)
  - [`normal_equations_add!`](@ref)
  - [`normal_equations_coef`](@ref)
"""
@concrete struct NormalEquationsFit <: AbstractResult
    """
    The sum of the outer products of the valid design rows, `X'X`.
    """
    XtX
    """
    The sum of each valid design row times its target, `X'y`.
    """
    Xty
    """
    The number of valid samples in the two sums.
    """
    n
    """
    The coefficients that solve the normal equations.
    """
    coef
    function NormalEquationsFit(XtX::MatNum, Xty::VecNum, n::Integer)
        @argcheck(size(XtX, 1) == size(XtX, 2) == length(Xty),
                  DimensionMismatch("XtX must be square with the side of Xty, got size(XtX) = $(size(XtX)) and length(Xty) = $(length(Xty))"))
        @argcheck(n >= zero(n), DomainError(n, "n must be non-negative"))
        coef = normal_equations_coef(XtX, Xty)
        return new{typeof(XtX), typeof(Xty), typeof(n), typeof(coef)}(XtX, Xty, n, coef)
    end
end
function StatsAPI.coef(m::NormalEquationsFit)::VecNum
    return m.coef
end
function StatsAPI.predict(m::NormalEquationsFit, X::MatNum)::VecNum
    return X * m.coef
end
"""
    normal_equations_coef(XtX::MatNum, Xty::VecNum) -> VecNum

Solve the normal equations `XtX * b = Xty` of a least squares fit, with a zero coefficient for each collinear column.

# Algorithm

 1. Factorise `XtX` through a Cholesky factorisation with pivots, with the default tolerance of `LinearAlgebra`, as `GLM.LinearModel` does. This gives the rank `r` and the order `p` of the columns.
 2. Solve the leading `r × r` block for the first `r` columns of `p`, and give each other column a zero coefficient.

# Arguments

  - `XtX`: The sum of the outer products of the design rows.
  - `Xty`: The sum of each design row times its target.

# Returns

  - `b::VecNum`: The coefficients.

# Related

  - [`NormalEquationsFit`](@ref)
  - [`normal_equations_add!`](@ref)
"""
function normal_equations_coef(XtX::MatNum, Xty::VecNum)::VecNum
    Tf = promote_type(real(eltype(XtX)), real(eltype(Xty)))
    b = zeros(Tf, length(Xty))
    # A dense copy, so the factorisation in place leaves the carried sums as they are.
    F = LinearAlgebra.cholesky!(LinearAlgebra.Symmetric(Matrix{Tf}(XtX)),
                                LinearAlgebra.RowMaximum(); tol = -one(Tf), check = false)
    r = F.rank
    if r > 0
        p = F.p[1:r]
        U = LinearAlgebra.UpperTriangular(F.U[1:r, 1:r])
        b[p] = U \ (transpose(U) \ Xty[p])
    end
    return b
end
"""
    normal_equations_add!(XtX::MatNum, Xty::VecNum, Sf::MatNum, yf::VecNum,
                          ok::AbstractVector{Bool}, js::AbstractUnitRange) -> Integer

Add the valid samples `js` of a flat design to the normal equations of a least squares fit, in place.

The function adds one sample at a time, in the order of `js`. So a sum over many ranges, taken one range after the other, is the sum over their union to the last bit. This is what lets [`PrequentialCalibration`](@ref) fold one observation at a time and still equal the batch fit.

# Arguments

  - `XtX`: The sum of the outer products of the design rows. The function changes it.
  - `Xty`: The sum of each design row times its target. The function changes it.
  - `Sf`: The flattened design.
  - `yf`: The flattened target.
  - `ok`: The mask of the valid samples.
  - `js`: The samples to add.

# Returns

  - `m::Integer`: The number of valid samples that the function added.

# Related

  - [`NormalEquationsFit`](@ref)
  - [`normal_equations_coef`](@ref)
  - [`target_forecast_prequential`](@ref)
"""
function normal_equations_add!(XtX::MatNum, Xty::VecNum, Sf::MatNum, yf::VecNum,
                               ok::AbstractVector{Bool}, js::AbstractUnitRange)::Integer
    m = 0
    for j in js
        if !ok[j]
            continue
        end
        m += 1
        for l in axes(Sf, 2)
            x = Sf[j, l]
            Xty[l] += x * yf[j]
            for k in axes(Sf, 2)
                XtX[k, l] += Sf[j, k] * x
            end
        end
    end
    return m
end
"""
$(DocStringExtensions.TYPEDEF)

A Return Forecast fitted by a regression target over every observation and asset at once.

The member turns its Descriptors into scores with the recipe in `scores`, and hands every `(observation, asset)` pair whose forward target has matured to a regression target as one sample. The combination of the Descriptors is whatever the target fits, so this member admits a nonlinear one. [`LinearModel`](@ref) and [`GeneralisedLinearModel`](@ref) are two targets the library defines. A caller's own target runs here when it states `StatsAPI.fit` and `StatsAPI.predict`, as the `# Interfaces` section of [`AbstractRegressionTarget`](@ref) states. The member fits the target without observation weights, so the target needs no weight method here.

The member transforms the target of the fit cross-sectionally before the fit reads it, so one extreme observation does not set the shape of the whole model. The transformed target has no unit, and neither has the prediction. One exponentially weighted scalar regression of the forward return on the prediction puts the prediction into return units. `cv` chooses the predictions that regression reads. A prediction of a sample the model trained on agrees with its own target by construction, so an in-sample calibration gives a positive slope to scores that carry no information. A calibration slope is a measure of skill, and skill is an out-of-sample quantity. The default, [`PrequentialCalibration`](@ref), predicts each observation with the model fitted on the observations whose target had matured at it. That prediction is the forecast the member publishes at that observation, so it reads no later observation, and the member can fold one observation at a time. A cross-validation estimator gives out-of-fold predictions, but a fold that comes before another trains on later observations.

`intercept` states whether the fit adds a constant to the scores. The intercept adds the same value to the prediction of every asset of an observation. The target is the idiosyncratic return, and under a market factor its weighted cross-sectional mean is zero at every observation, so an intercept estimates a quantity near zero. The default fits none.

The member computes no history. Its in-sample predictions are not a forecast, so `hist` on its Result is `nothing`.

`whole_history` states the rows the fit trains on. Under `true` the member places the block's idiosyncratic returns into the rows of the returns data they were fitted on. Every pair with a finite score, a finite target and a positive weight is then one sample, so a signal row before the block whose forward window reaches into the block trains the model too. Under `false` the fit trains on the block's rows alone. The calibration reads the idiosyncratic variance at the signal row, and a row before the block has none, so such a row never enters the calibration. In [`IdiosyncraticSharpeUnit`](@ref) the target reads that variance too, so a row before the block trains nothing, and the two values of `whole_history` fit the same model.

# Mathematical definition

```math
\\begin{align}
z_{ti} &= \\mathcal{T}\\left(\\frac{\\bar{\\varepsilon}_{ti}}{g_{ti}}\\right)\\,, \\\\
\\mathcal{M} &= \\operatorname{fit}\\left(\\texttt{tgt}, \\left\\{\\left(\\boldsymbol{s}_{ti}, z_{ti}\\right) : (t, i) \\in \\mathcal{S}\\right\\}\\right)\\,, \\\\
p_{ti} &= g_{ti} \\operatorname{predict}\\left(\\mathcal{M}, \\boldsymbol{s}_{ti}\\right)\\,, \\\\
\\omega_{ti} &= \\frac{u_{ti} / v_{ti}}{\\frac{1}{\\lvert \\mathcal{A}_{t} \\rvert} \\sum_{j \\in \\mathcal{A}_{t}} u_{tj} / v_{tj}}\\,, \\\\
a_{t} &= \\sum_{i \\in \\mathcal{A}_{t}} \\omega_{ti} \\, p_{ti}^{2}\\,, \\\\
c_{t} &= \\sum_{i \\in \\mathcal{A}_{t}} \\omega_{ti} \\, p_{ti} \\, \\bar{\\varepsilon}_{ti}\\,, \\\\
A_{k} &= \\lambda A_{k-1} + (1 - \\lambda) \\, a_{t_{k}}\\,, \\\\
C_{k} &= \\lambda C_{k-1} + (1 - \\lambda) \\, c_{t_{k}}\\,, \\\\
\\kappa &= \\frac{C_{n}}{(1 + \\varrho) A_{n}}\\,, \\\\
\\alpha_{Ti} &= \\gamma \\, \\kappa \\, p_{Ti}\\,.
\\end{align}
```

Where:

  - $(math_dict[:eps_ti_idio])
  - ``\\bar{\\varepsilon}_{ti}``: Forward mean idiosyncratic return of asset ``i`` at observation ``t``, the mean of the finite ``\\varepsilon_{si}`` over ``s`` from ``t + \\ell`` to ``t + \\ell + h - 1``, with ``\\ell`` the lag and ``h`` the horizon.
  - $(math_dict[:v_ti_idio])
  - $(math_dict[:g_ti_unit])
  - ``\\mathcal{T}``: The cross-sectional transform in `target_outlier` followed by the one in `target_scoring`. The member applies it to each observation, and an absent transform leaves the target as it is.
  - ``z_{ti}``: Target of the fit for asset ``i`` at observation ``t``.
  - ``\\boldsymbol{s}_{ti}``: Descriptor scores of asset ``i`` at observation ``t``. Under `intercept`, the member appends a one to them.
  - $(math_dict[:u_ti_cs])
  - ``\\mathcal{S}``: The valid samples, the pairs whose target has matured, ``t \\leq T - \\ell - h + 1``, with ``u_{ti} > 0`` and a finite ``z_{ti}`` and ``\\boldsymbol{s}_{ti}``.
  - ``\\mathcal{M}``: The model the regression target fits on the valid samples.
  - ``p_{ti}``: Uncalibrated prediction of asset ``i`` at observation ``t``, in return units. Under [`PrequentialCalibration`](@ref), the calibration reads the prediction of each matured observation ``t`` from the model fitted on the valid samples of the observations up to ``t - \\ell - h + 1``, and `NaN` while there is no such sample. Under a cross-validation estimator in `cv`, the calibration reads the prediction of each valid sample from the model fitted on the folds that do not hold that sample. Below two valid samples per fold, no prediction is out of fold, and `warmup` states the prediction the calibration reads.
  - ``\\mathcal{A}_{t}``: The assets of a matured observation ``t`` that enter the calibration, those with ``u_{ti} > 0`` and a finite ``v_{ti}``, ``p_{ti}`` and ``\\bar{\\varepsilon}_{ti}``.
  - ``\\omega_{ti}``: Calibration weight of asset ``i`` at observation ``t``. The weights of one observation have a mean of one.
  - ``a_{t}``, ``c_{t}``: The weighted normal product and cross product of observation ``t``.
  - ``t_{1} < \\dots < t_{n}``: The observations that advance the calibration, those with at least two assets in ``\\mathcal{A}_{t}``, a finite ``c_{t}`` and a finite ``a_{t}`` above the machine epsilon. An observation that does not advance it leaves ``A`` and ``C`` as they are and does not decay them.
  - ``A_{k}``, ``C_{k}``: The two exponentially weighted accumulators, with ``A_{0} = C_{0} = 0``.
  - $(math_dict[:lambda_ew])
  - ``\\varrho``: The relative ridge, ``10^{-6}``. On a scalar regression it shrinks the coefficient by the factor ``1 / (1 + \\varrho)``.
  - ``\\kappa``: The calibration coefficient. It is ``1`` when `calibrate` is `false`, and `NaN` when ``n`` is below `min_obs`.
  - $(math_dict[:gamma_rf_scale])
  - $(math_dict[:alpha_ti_fc]) The member publishes the row of the latest observation ``T``.
  - $(math_dict[:T])

# Fields

$(DocStringExtensions.TYPEDFIELDS)

# Constructors

    TargetReturnForecast(; scores::DescriptorScores,
                         tgt::AbstractRegressionTarget = LinearModel(),
                         horizon::Integer = 1, lag::Integer = 1,
                         whole_history::Bool = true,
                         target_outlier::Option{<:AbstractCrossSectionalTransform} = CrossSectionalWinsoriser(),
                         target_scoring::Option{<:AbstractCrossSectionalTransform} = nothing,
                         calibrate::Bool = true, scale::Real = 1.0,
                         half_life::Real = 20.0, decay::Real = half_life_decay(half_life),
                         min_obs::Integer = 1,
                         cv::Union{Nothing, CrossValidationEstimator, PrequentialCalibration} = PrequentialCalibration(),
                         warmup::AbstractCalibrationWarmup = NaNWarmup(),
                         unit::AbstractForecastUnit = IdiosyncraticReturnUnit(),
                         intercept::Bool = false) -> TargetReturnForecast

Every keyword but `half_life` corresponds to a field. `half_life` is not a field. It fixes the default of `decay` of the calibration, and the constructor keeps a value passed for `decay` as it stands. The default `min_obs = 1` calibrates from the first observation that states a slope. The weight ``1 - \\lambda^{n}`` that the first ``n`` observations carry multiplies both accumulators and the ridge alike, so the coefficient after ``n`` observations is the weighted least squares slope of those observations, with no start-up bias for a warm-up to wait out.

## Validation

  - $(val_dict[:decay])
  - `min_obs >= 1`, `horizon >= 1` and `lag >= 1`.
  - `scale > 0` and is finite.

# Examples

```jldoctest
julia> ds = DescriptorScores(; descriptors = [Passthrough(; field = \"a\")]);

julia> TargetReturnForecast(; scores = ds, target_outlier = nothing, half_life = 2,
                            calibrate = false)
TargetReturnForecast
          scores ┼ DescriptorScores
                 │   descriptors ┼ 1-element Vector{Passthrough}
                 │               │ Passthrough ⋯
                 │    neutralise ┼ nothing
                 │            nw ┼ EstimationMaskWeights()
                 │           cre ┼ CrossSectionalLinearRegression
                 │               │         alg ┼ PseudoInverseFallback()
                 │               │   intercept ┼ Bool: true
                 │               │          ex ┴ Transducers.ThreadedEx{@NamedTuple{}}: Transducers.ThreadedEx()
                 │       outlier ┼ CrossSectionalWinsoriser
                 │               │    low ┼ Float64: 0.01
                 │               │   high ┴ Float64: 0.99
                 │       scoring ┼ CrossSectionalStandardiser
                 │               │   min_group_size ┼ Int64: 8
                 │               │             atol ┴ Float64: 1.0e-12
                 │         group ┼ nothing
                 │            ex ┴ Transducers.ThreadedEx{@NamedTuple{}}: Transducers.ThreadedEx()
             tgt ┼ LinearModel
                 │   kwargs ┴ @NamedTuple{}: NamedTuple()
         horizon ┼ Int64: 1
             lag ┼ Int64: 1
   whole_history ┼ Bool: true
  target_outlier ┼ nothing
  target_scoring ┼ nothing
       calibrate ┼ Bool: false
           scale ┼ Float64: 1.0
           decay ┼ Float64: 0.7071067811865476
         min_obs ┼ Int64: 1
              cv ┼ PrequentialCalibration()
          warmup ┼ NaNWarmup()
            unit ┼ IdiosyncraticReturnUnit()
       intercept ┴ Bool: false
```

# Related

  - [`AbstractReturnForecastEstimator`](@ref)
  - [`TargetReturnForecastResult`](@ref)
  - [`return_forecast`](@ref)
  - [`DescriptorScores`](@ref)
  - [`AbstractRegressionTarget`](@ref)
  - [`CrossValidationEstimator`](@ref)
  - [`ExpWeightedReturnForecast`](@ref)
"""
@concrete struct TargetReturnForecast <: AbstractReturnForecastEstimator
    """
    $(field_dict[:rf_scores])
    """
    scores
    """
    $(field_dict[:retgt])
    """
    tgt
    """
    $(field_dict[:rf_horizon])
    """
    horizon
    """
    $(field_dict[:rf_lag])
    """
    lag
    """
    $(field_dict[:rf_whole_history])
    """
    whole_history
    """
    Cross-sectional transform the member applies first to the target of the fit, or `nothing` to skip the step.
    """
    target_outlier
    """
    Cross-sectional transform the member applies to the target of the fit after `target_outlier`, or `nothing` to skip the step. It is `nothing` by default, so under the default `target_outlier` the fit predicts the winsorised return itself.
    """
    target_scoring
    """
    Whether the member puts its prediction back into return units by an exponentially weighted scalar regression of the forward return on it.
    """
    calibrate
    """
    $(field_dict[:rf_scale])
    """
    scale
    """
    $(field_dict[:decay]) It weighs the calibration alone.
    """
    decay
    """
    $(field_dict[:min_obs]) It counts the observations that advanced the calibration.
    """
    min_obs
    """
    The rule that gives the uncalibrated predictions the calibration reads. [`PrequentialCalibration`](@ref), the default, reads the prediction of each observation from the model fitted on the targets that had matured at it. A cross-validation estimator reads the out-of-fold predictions of its folds, and `nothing` reads the in-sample predictions. The field takes an estimator and never an integer: `KFold(; n = k)` gives `k` consecutive folds with no shuffle, which is the split that an integer `k` means in a K-fold short form.
    """
    cv
    """
    Calibration warm-up, the rule that states the prediction the calibration reads below two valid samples per fold of `cv`. [`NaNWarmup`](@ref) gives a `NaN` coefficient there, and [`InSampleWarmup`](@ref) calibrates on the in-sample predictions. The field acts only under a cross-validation estimator.
    """
    warmup
    """
    $(field_dict[:rf_unit])
    """
    unit
    """
    Whether the fit appends a column of ones to the scores, so that the model carries an intercept.
    """
    intercept
    function TargetReturnForecast(scores::DescriptorScores, tgt::AbstractRegressionTarget,
                                  horizon::Integer, lag::Integer, whole_history::Bool,
                                  target_outlier::Option{<:AbstractCrossSectionalTransform},
                                  target_scoring::Option{<:AbstractCrossSectionalTransform},
                                  calibrate::Bool, scale::Real, decay::Real,
                                  min_obs::Integer,
                                  cv::Union{Nothing, CrossValidationEstimator,
                                            PrequentialCalibration},
                                  warmup::AbstractCalibrationWarmup,
                                  unit::AbstractForecastUnit, intercept::Bool)
        assert_nonempty_gt0_finite_val(horizon, :horizon)
        assert_nonempty_gt0_finite_val(lag, :lag)
        assert_finite(scale, :scale)
        assert_gt0(scale, :scale)
        assert_ew_decay(decay)
        assert_nonempty_gt0_finite_val(min_obs, :min_obs)
        return new{typeof(scores), typeof(tgt), typeof(horizon), typeof(lag),
                   typeof(whole_history), typeof(target_outlier), typeof(target_scoring),
                   typeof(calibrate), typeof(scale), typeof(decay), typeof(min_obs),
                   typeof(cv), typeof(warmup), typeof(unit), typeof(intercept)}(scores, tgt,
                                                                                horizon,
                                                                                lag,
                                                                                whole_history,
                                                                                target_outlier,
                                                                                target_scoring,
                                                                                calibrate,
                                                                                scale,
                                                                                decay,
                                                                                min_obs, cv,
                                                                                warmup,
                                                                                unit,
                                                                                intercept)
    end
end
function TargetReturnForecast(; scores::DescriptorScores,
                              tgt::AbstractRegressionTarget = LinearModel(),
                              horizon::Integer = 1, lag::Integer = 1,
                              whole_history::Bool = true,
                              target_outlier::Option{<:AbstractCrossSectionalTransform} = CrossSectionalWinsoriser(),
                              target_scoring::Option{<:AbstractCrossSectionalTransform} = nothing,
                              calibrate::Bool = true, scale::Real = 1.0,
                              half_life::Real = 20.0,
                              decay::Real = half_life_decay(half_life),
                              min_obs::Integer = 1,
                              cv::Union{Nothing, CrossValidationEstimator,
                                        PrequentialCalibration} = PrequentialCalibration(),
                              warmup::AbstractCalibrationWarmup = NaNWarmup(),
                              unit::AbstractForecastUnit = IdiosyncraticReturnUnit(),
                              intercept::Bool = false)::TargetReturnForecast
    return TargetReturnForecast(scores, tgt, horizon, lag, whole_history, target_outlier,
                                target_scoring, calibrate, scale, decay, min_obs, cv,
                                warmup, unit, intercept)
end
function fits_idiosyncratic_target(::TargetReturnForecast)::Bool
    return true
end
function calibrates_orthogonal_part(rfe::TargetReturnForecast)::Bool
    return rfe.calibrate
end
"""
$(DocStringExtensions.TYPEDEF)

Result type produced by [`TargetReturnForecast`](@ref).

Beside the two reads [`AbstractReturnForecastResult`](@ref) states, it carries the fitted model and the calibration coefficient. A reader can inspect the combination the target fitted and the coefficient that puts the prediction into return units. `hist` is `nothing`, because the member computes no history. When the caller asks for it, the Result also carries the calibration coefficient of the orthogonal part of the prediction.

# Fields

$(DocStringExtensions.TYPEDFIELDS)

# Related

  - [`AbstractReturnForecastResult`](@ref)
  - [`TargetReturnForecast`](@ref)
  - [`return_forecast`](@ref)
"""
@concrete struct TargetReturnForecastResult <: AbstractReturnForecastResult
    """
    $(field_dict[:rf_mu])
    """
    mu
    """
    $(field_dict[:rf_hist])
    """
    hist
    """
    The model the regression target fits on the valid samples, or `nothing` when no sample is valid.
    """
    model
    """
    The calibration coefficient. It is `NaN` when the member does not calibrate, when it fits no model, and while the calibration is in its warm-up.
    """
    calib
    """
    The calibration coefficient of the orthogonal part of the prediction, `κ⊥`, when the caller asks for it through the four-argument method of [`return_forecast`](@ref), and `nothing` otherwise. It is `NaN` where `calib` is, and while its own regression is in its warm-up.
    """
    ocalib
    function TargetReturnForecastResult(mu::VecNum, model, calib::Number,
                                        ocalib::Option{<:Number})
        @argcheck(!isempty(mu), IsEmptyError("mu cannot be empty"))
        return new{typeof(mu), Nothing, typeof(model), typeof(calib), typeof(ocalib)}(mu,
                                                                                      nothing,
                                                                                      model,
                                                                                      calib,
                                                                                      ocalib)
    end
end
function TargetReturnForecastResult(; mu::VecNum, model = nothing, calib::Number = NaN,
                                    ocalib::Option{<:Number} = nothing)::TargetReturnForecastResult
    return TargetReturnForecastResult(mu, model, calib, ocalib)
end
"""
    target_forecast_variances(unit::IdiosyncraticReturnUnit,
                              csfm::CrossSectionalFactorModel,
                              calibrate::Bool) -> Option{<:MatNum}
    target_forecast_variances(unit::IdiosyncraticSharpeUnit,
                              csfm::CrossSectionalFactorModel, calibrate::Bool) -> MatNum

Return the idiosyncratic variance history a [`TargetReturnForecast`](@ref) reads, or `nothing`.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`IdiosyncraticSharpeUnit`](@ref): the variances scale the target and the forecast, so the method reads them whatever `calibrate` says.
 2. [`IdiosyncraticReturnUnit`](@ref): the variances weigh the calibration alone, so the method reads them when `calibrate` is set and returns `nothing` when it is not.

# Arguments

  - `unit`: The Forecast Unit the member scores in.
  - `csfm`: The fitted factor-model block.
  - `calibrate`: Whether the member calibrates its prediction to return units.

# Validation

  - The rules of [`forecast_idiosyncratic_variances`](@ref), which reads the history.

# Returns

  - `vs::Option{<:MatNum}`: Idiosyncratic variances, or `nothing` when the member reads none.

# Related

  - [`TargetReturnForecast`](@ref)
  - [`forecast_idiosyncratic_variances`](@ref)
  - [`AbstractForecastUnit`](@ref)
"""
function target_forecast_variances(::IdiosyncraticReturnUnit,
                                   csfm::CrossSectionalFactorModel,
                                   calibrate::Bool)::Option{<:MatNum}
    return calibrate ? forecast_idiosyncratic_variances(csfm) : nothing
end
function target_forecast_variances(::IdiosyncraticSharpeUnit,
                                   csfm::CrossSectionalFactorModel, ::Bool)::MatNum
    return forecast_idiosyncratic_variances(csfm)
end
"""
    target_forecast_samples(S::Arr3Num, y::MatNum, w::MatNum, nt::Integer) -> Tuple

Flatten the Descriptor scores and the target of a [`TargetReturnForecast`](@ref) into one sample per `(observation, asset)` pair.

The regression target of this member does not fit one cross-section at a time. Every pair of an observation whose forward target has matured is one sample, and the fit reads all of them at once. The function writes the pairs observation by observation, so the flat index of the pair `(t, i)` is `(t - 1) * assets + i`.

# Arguments

  - `S`: The Descriptor scores, `observations × assets × descriptors`.
  - `y`: The target in the Forecast Unit, `observations × assets`.
  - `w`: Cross-sectional weights, `observations × assets`.
  - `nt`: Number of observations whose target has matured.

# Returns

  - `(Sf, yf, ok)::Tuple`: The design `nt · assets × descriptors`, the target of the same length, and the mask of the pairs that carry a positive weight, a finite target and a finite score for every Descriptor.

# Related

  - [`TargetReturnForecast`](@ref)
  - [`target_forecast_fit`](@ref)
  - [`descriptor_scores`](@ref)
"""
function target_forecast_samples(S::Arr3Num, y::MatNum, w::MatNum, nt::Integer)
    Tf = promote_type(real(eltype(S)), real(eltype(y)))
    N = size(S, 2)
    K = size(S, 3)
    Sf = Matrix{Tf}(undef, nt * N, K)
    yf = Vector{Tf}(undef, nt * N)
    ok = falses(nt * N)
    j = 0
    for t in 1:nt, i in 1:N
        j += 1
        good = w[t, i] > zero(w[t, i]) && isfinite(y[t, i])
        for k in 1:K
            Sf[j, k] = S[t, i, k]
            good &= isfinite(Sf[j, k])
        end
        yf[j] = y[t, i]
        ok[j] = good
    end
    return Sf, yf, ok
end
"""
    target_forecast_fit(rfe::TargetReturnForecast, Sf::MatNum, yf::VecNum,
                        ok::AbstractVector{Bool}) -> Option

Fit the regression target of a [`TargetReturnForecast`](@ref) on the valid samples.

A window with no valid sample fits nothing and returns `nothing`. This is the warm-up of the member, which forecasts `NaN` until the target of one observation has matured.

# Algorithm

The methods of [`target_forecast_model`](@ref) that Julia selects are the algorithm.

 1. [`PrequentialCalibration`](@ref) and a [`LinearModel`](@ref) with no keyword argument: add every valid sample to the normal equations through [`normal_equations_add!`](@ref), in the order of [`target_forecast_samples`](@ref), giving a [`NormalEquationsFit`](@ref). The predictions of [`target_forecast_prequential`](@ref) add the same samples in the same order, so the model of the latest observation is the end of that fold to the last bit.
 2. Any other pair: fit the regression target on the valid samples through `StatsAPI.fit`.

# Arguments

  - `rfe`: Target Return Forecast Estimator. The function reads the regression target from its field rather than from an abstract argument, so the call of the third-party fit stays concrete.
  - `Sf`: The flattened design.
  - `yf`: The flattened target.
  - `ok`: The mask of the valid samples.

# Returns

  - `model::Option`: The fitted model, or `nothing` when no sample is valid.

# Related

  - [`TargetReturnForecast`](@ref)
  - [`target_forecast_samples`](@ref)
  - [`AbstractRegressionTarget`](@ref)
  - [`NormalEquationsFit`](@ref)
"""
function target_forecast_fit(rfe::TargetReturnForecast, Sf::MatNum, yf::VecNum,
                             ok::AbstractVector{Bool})
    return any(ok) ? target_forecast_model(rfe.cv, rfe.tgt, Sf, yf, ok) : nothing
end
"""
    target_forecast_model(cv::PrequentialCalibration, tgt::LinearModel{@NamedTuple{}},
                          Sf::MatNum, yf::VecNum, ok::AbstractVector{Bool}) -> NormalEquationsFit
    target_forecast_model(cv, tgt::AbstractRegressionTarget, Sf::MatNum, yf::VecNum,
                          ok::AbstractVector{Bool})

Fit the model of the latest observation of a [`TargetReturnForecast`](@ref) on the valid samples. [`target_forecast_fit`](@ref) states the two methods.

# Arguments

  - `cv`: The rule of the calibration, `rfe.cv`.
  - `tgt`: The regression target, `rfe.tgt`.
  - `Sf`: The flattened design.
  - `yf`: The flattened target.
  - `ok`: The mask of the valid samples. At least one sample is valid.

# Returns

  - `model`: The fitted model.

# Related

  - [`target_forecast_fit`](@ref)
  - [`NormalEquationsFit`](@ref)
"""
function target_forecast_model(::PrequentialCalibration, ::LinearModel{@NamedTuple{}},
                               Sf::MatNum, yf::VecNum,
                               ok::AbstractVector{Bool})::NormalEquationsFit
    Tf = promote_type(real(eltype(Sf)), real(eltype(yf)))
    K = size(Sf, 2)
    XtX = zeros(Tf, K, K)
    Xty = zeros(Tf, K)
    n = normal_equations_add!(XtX, Xty, Sf, yf, ok, eachindex(ok))
    return NormalEquationsFit(XtX, Xty, n)
end
function target_forecast_model(::Any, tgt::AbstractRegressionTarget, Sf::MatNum, yf::VecNum,
                               ok::AbstractVector{Bool})
    idx = findall(ok)
    return StatsAPI.fit(tgt, Sf[idx, :], yf[idx])
end
"""
    target_forecast_uncalibrated(cv::Nothing, rfe::TargetReturnForecast, model,
                                 Sf::MatNum, yf::VecNum, ok::AbstractVector{Bool},
                                 N::Integer) -> VecNum
    target_forecast_uncalibrated(cv::CrossValidationEstimator, rfe::TargetReturnForecast,
                                 model, Sf::MatNum, yf::VecNum, ok::AbstractVector{Bool},
                                 N::Integer) -> VecNum
    target_forecast_uncalibrated(cv::PrequentialCalibration, rfe::TargetReturnForecast,
                                 model, Sf::MatNum, yf::VecNum, ok::AbstractVector{Bool},
                                 N::Integer) -> VecNum

Predict the uncalibrated forecast of the samples a [`TargetReturnForecast`](@ref) trained on.

# Algorithm

The method that Julia selects is the algorithm.

 1. `nothing`: the fitted model predicts its own training samples through [`target_forecast_insample`](@ref), so the calibration runs in sample.
 2. A cross-validation estimator: [`Base.split`](@ref) splits the valid samples. For each split, the function fits a model on the training folds and predicts the test fold, so the calibration never reads a prediction of a sample the model trained on. A sample that no split tests keeps its `NaN`. Below two valid samples per fold, which is twice the split count that [`n_splits`](@ref) reports, the function fits no fold and returns the prediction of [`target_forecast_warmup`](@ref) under `rfe.warmup`. The default [`NaNWarmup`](@ref) gives `NaN` to every sample, so the calibration is in its warm-up. [`InSampleWarmup`](@ref) gives the in-sample prediction of method 1, with the bias that the split removes.
 3. [`PrequentialCalibration`](@ref): each observation reads the prediction of the model fitted on the valid samples whose target had matured at it, through [`target_forecast_prequential`](@ref). The method reads no model that `model` holds.

# Arguments

  - `cv`: The rule of the calibration. It is `rfe.cv`. The caller passes it as an argument of its own, so that the method Julia selects is the algorithm.
  - `rfe`: Target Return Forecast Estimator.
  - `model`: The model fitted on every valid sample.
  - `Sf`: The flattened design.
  - `yf`: The flattened target.
  - `ok`: The mask of the valid samples.
  - `N`: The number of assets, so that the samples of observation `t` are `(t - 1) * N + 1` to `t * N`.

# Returns

  - `p::VecNum`: The uncalibrated prediction of every sample, `NaN` where the sample is not valid, and everywhere in the warm-up of an out-of-fold calibration under [`NaNWarmup`](@ref).

# Related

  - [`TargetReturnForecast`](@ref)
  - [`target_forecast_fit`](@ref)
  - [`CrossValidationEstimator`](@ref)
  - [`PrequentialCalibration`](@ref)
"""
function target_forecast_uncalibrated(::Nothing, ::TargetReturnForecast, model, Sf::MatNum,
                                      ::VecNum, ok::AbstractVector{Bool}, ::Integer)::VecNum
    return target_forecast_insample(model, Sf, ok)
end
function target_forecast_uncalibrated(cv::CrossValidationEstimator,
                                      rfe::TargetReturnForecast, model, Sf::MatNum,
                                      yf::VecNum, ok::AbstractVector{Bool},
                                      ::Integer)::VecNum
    Tf = real(eltype(Sf))
    idx = findall(ok)
    m = length(idx)
    rdx = ReturnsResult(; nx = ["sample"], X = zeros(Tf, m, 1))
    if m < 2 * n_splits(cv, rdx)
        return target_forecast_warmup(rfe.warmup, rfe, model, Sf, yf, ok)
    end
    p = fill(Tf(NaN), length(ok))
    Sv = Sf[idx, :]
    yv = yf[idx]
    q = fill(Tf(NaN), m)
    fld = Base.split(cv, rdx)
    for k in eachindex(fld.train_idx)
        tr = fld.train_idx[k]
        te = fld.test_idx[k]
        q[te] = StatsAPI.predict(StatsAPI.fit(rfe.tgt, Sv[tr, :], yv[tr]), Sv[te, :])
    end
    p[idx] = q
    return p
end
function target_forecast_uncalibrated(::PrequentialCalibration, rfe::TargetReturnForecast,
                                      ::Any, Sf::MatNum, yf::VecNum,
                                      ok::AbstractVector{Bool}, N::Integer)::VecNum
    return target_forecast_prequential(rfe.tgt, rfe.lag + rfe.horizon - 1, Sf, yf, ok, N)
end
"""
    target_forecast_insample(model, Sf::MatNum, ok::AbstractVector{Bool}) -> VecNum

Predict the valid samples of a [`TargetReturnForecast`](@ref) with the model fitted on all of them, and give every other sample `NaN`.

# Arguments

  - `model`: The model fitted on every valid sample.
  - `Sf`: The flattened design.
  - `ok`: The mask of the valid samples.

# Returns

  - `p::VecNum`: The in-sample prediction of every sample.

# Related

  - [`target_forecast_uncalibrated`](@ref)
  - [`InSampleWarmup`](@ref)
"""
function target_forecast_insample(model, Sf::MatNum, ok::AbstractVector{Bool})::VecNum
    Tf = real(eltype(Sf))
    p = fill(Tf(NaN), length(ok))
    idx = findall(ok)
    p[idx] = StatsAPI.predict(model, Sf[idx, :])
    return p
end
"""
    target_forecast_prequential(tgt::LinearModel{@NamedTuple{}}, g::Integer, Sf::MatNum,
                                yf::VecNum, ok::AbstractVector{Bool}, N::Integer) -> VecNum
    target_forecast_prequential(tgt::AbstractRegressionTarget, g::Integer, Sf::MatNum,
                                yf::VecNum, ok::AbstractVector{Bool}, N::Integer) -> VecNum

Predict each valid sample of a [`TargetReturnForecast`](@ref) with the model fitted on the valid samples whose target had matured at its observation, the rule of [`PrequentialCalibration`](@ref).

# Algorithm

For each observation `t` in order:

 1. When `t > g`, the target of observation `t - g` matures at `t`. Add its valid samples to the fit.
 2. When the fit holds a valid sample, predict the valid samples of observation `t` with it. Otherwise they keep `NaN`.

The method that Julia selects states how the fit grows.

 1. A [`LinearModel`](@ref) with no keyword argument: add the samples to the normal equations through [`normal_equations_add!`](@ref), and solve them through [`normal_equations_coef`](@ref) when they change. [`target_forecast_fit`](@ref) adds the same samples in the same order, so the two agree to the last bit.
 2. Any other regression target: fit it again through `StatsAPI.fit` on every matured valid sample, when step 1 adds one.

# Arguments

  - `tgt`: The regression target, `rfe.tgt`.
  - `g`: The rows between an observation and the maturity of its target, `lag + horizon - 1`.
  - `Sf`: The flattened design, `nt · N × K`.
  - `yf`: The flattened target.
  - `ok`: The mask of the valid samples.
  - `N`: The number of assets.

# Returns

  - `p::VecNum`: The prequential prediction of every sample, `NaN` where the sample is not valid or no valid sample matured before its observation.

# Related

  - [`PrequentialCalibration`](@ref)
  - [`target_forecast_uncalibrated`](@ref)
  - [`NormalEquationsFit`](@ref)
"""
function target_forecast_prequential(::LinearModel{@NamedTuple{}}, g::Integer, Sf::MatNum,
                                     yf::VecNum, ok::AbstractVector{Bool},
                                     N::Integer)::VecNum
    Tf = promote_type(real(eltype(Sf)), real(eltype(yf)))
    K = size(Sf, 2)
    p = fill(Tf(NaN), length(ok))
    XtX = zeros(Tf, K, K)
    Xty = zeros(Tf, K)
    b = zeros(Tf, K)
    n = 0
    # No target matured before observation g + 1, so the first g observations keep NaN.
    for t in (g + 1):target_forecast_observations(ok, N)
        # The fit changes only when the matured observation adds a valid sample.
        m = normal_equations_add!(XtX, Xty, Sf, yf, ok, target_forecast_row(t - g, N))
        n += m
        b = m > 0 ? normal_equations_coef(XtX, Xty) : b
        for j in target_forecast_row(t, N)
            if n > 0 && ok[j]
                p[j] = LinearAlgebra.dot(view(Sf, j, :), b)
            end
        end
    end
    return p
end
function target_forecast_prequential(tgt::AbstractRegressionTarget, g::Integer, Sf::MatNum,
                                     yf::VecNum, ok::AbstractVector{Bool},
                                     N::Integer)::VecNum
    Tf = promote_type(real(eltype(Sf)), real(eltype(yf)))
    p = fill(Tf(NaN), length(ok))
    idx = Int[]
    model = nothing
    for t in (g + 1):target_forecast_observations(ok, N)
        m = length(idx)
        append!(idx, Iterators.filter(j -> ok[j], target_forecast_row(t - g, N)))
        if length(idx) > m
            model = StatsAPI.fit(tgt, Sf[idx, :], yf[idx])
        end
        if isnothing(model)
            continue
        end
        js = findall(view(ok, target_forecast_row(t, N))) .+ (t - 1) * N
        if !isempty(js)
            p[js] = StatsAPI.predict(model, Sf[js, :])
        end
    end
    return p
end
"""
    target_forecast_observations(ok::AbstractVector{Bool}, N::Integer) -> Integer

Count the observations of the flat samples of a [`TargetReturnForecast`](@ref).

[`target_forecast_samples`](@ref) writes the samples observation by observation, `N` to each. So the samples hold `length(ok) ÷ N` observations, or none when there is no asset.

# Arguments

  - `ok`: The mask of the valid samples.
  - `N`: The number of assets.

# Returns

  - `nt::Integer`: The number of observations.

# Related

  - [`target_forecast_row`](@ref)
  - [`target_forecast_prequential`](@ref)
"""
function target_forecast_observations(ok::AbstractVector{Bool}, N::Integer)::Integer
    return N > 0 ? length(ok) ÷ N : 0
end
"""
    target_forecast_row(t::Integer, N::Integer) -> UnitRange{Int}

Return the flat samples of observation `t` of a [`TargetReturnForecast`](@ref).

[`target_forecast_samples`](@ref) writes the samples observation by observation, `N` to each. So the samples of observation `t` are `(t - 1) * N + 1` to `t * N`.

# Arguments

  - `t`: Index of the observation.
  - `N`: The number of assets.

# Returns

  - `js::UnitRange{Int}`: The samples of observation `t`.

# Related

  - [`target_forecast_observations`](@ref)
  - [`target_forecast_prequential`](@ref)
"""
function target_forecast_row(t::Integer, N::Integer)::UnitRange{Int}
    return ((t - 1) * N + 1):(t * N)
end
"""
    target_forecast_scatter(p::VecNum, nt::Integer, N::Integer) -> Matrix{<:Real}

Read a flat sample vector of a [`TargetReturnForecast`](@ref) back as an `observations × assets` matrix.

It is the inverse of the layout that [`target_forecast_samples`](@ref) writes, so the two functions hold the flattening, and no caller computes an index.

# Arguments

  - `p`: The flat vector, of length `nt · N`.
  - `nt`: Number of observations whose target has matured.
  - `N`: Number of assets.

# Returns

  - `P::Matrix{<:Real}`: The same values, `nt × N`.

# Related

  - [`TargetReturnForecast`](@ref)
  - [`target_forecast_samples`](@ref)
  - [`target_forecast_uncalibrated`](@ref)
"""
function target_forecast_scatter(p::VecNum, nt::Integer, N::Integer)::Matrix{<:Real}
    Tf = real(eltype(p))
    P = Matrix{Tf}(undef, nt, N)
    j = 0
    for t in 1:nt, i in 1:N
        j += 1
        P[t, i] = p[j]
    end
    return P
end
"""
    target_forecast_calibration_design(P::MatNum, fwd::MatNum, vs::MatNum, w::MatNum,
                                       t::Integer) -> Tuple

Gather the calibration sample of one observation of a [`TargetReturnForecast`](@ref).

An asset enters when it carries a positive cross-sectional weight, a positive and finite idiosyncratic variance, a finite uncalibrated prediction and a finite forward return. A variance of zero would take an infinite weight, so it leaves the sample. A pair of leverage one comes in with a `NaN` variance from [`forecast_idiosyncratic_variances`](@ref), so it leaves the sample too. Its weight is the cross-sectional weight divided by its idiosyncratic variance. [`ExpWeightedReturnForecast`](@ref) weighs its regression the same way in the return unit.

# Arguments

  - `P`: The uncalibrated prediction, `matured observations × assets`.
  - `fwd`: Forward mean idiosyncratic returns, `observations × assets`.
  - `vs`: Idiosyncratic variance history, `observations × assets`.
  - `w`: Cross-sectional weights, `observations × assets`.
  - `t`: Index of the observation.

# Returns

  - `(a, y, wv)::Tuple`: The prediction, the forward return and the weight of every entering asset.

# Related

  - [`TargetReturnForecast`](@ref)
  - [`target_forecast_calibration`](@ref)
"""
function target_forecast_calibration_design(P::MatNum, fwd::MatNum, vs::MatNum, w::MatNum,
                                            t::Integer)
    Tf = promote_type(real(eltype(P)), real(eltype(fwd)), real(eltype(vs)), real(eltype(w)))
    idx = Int[]
    for i in axes(P, 2)
        if w[t, i] > zero(w[t, i]) &&
           zero(vs[t, i]) < vs[t, i] < Inf &&
           isfinite(P[t, i]) &&
           isfinite(fwd[t, i])
            push!(idx, i)
        end
    end
    return Tf.(view(P, t, idx)), Tf.(view(fwd, t, idx)),
           Tf.(view(w, t, idx)) ./ Tf.(view(vs, t, idx))
end
"""
    target_forecast_calibration(P::MatNum, fwd::MatNum, vs::MatNum, w::MatNum,
                                decay::Real, min_obs::Integer) -> Real

Fit the scalar calibration coefficient of a [`TargetReturnForecast`](@ref).

The member transforms the target of its fit, so the prediction is not in return units. One exponentially weighted scalar regression of the forward return on that prediction puts the prediction into return units, and the member multiplies its forecast by the slope. [`TargetReturnForecast`](@ref) states the slope as a closed form.

# Algorithm

 1. For each observation in turn, gather its calibration sample through [`target_forecast_calibration_design`](@ref). An observation with fewer than two entering assets states no slope, and the function skips it.
 2. Divide the weights by their mean over the entering assets, giving `u`. Take the weighted inner product of the prediction with itself, giving `on`, and with the forward return, giving `oc`. The function skips an observation whose products are not finite, or whose `on` is at or below `eps`.
 3. Advance the two exponentially weighted accumulators `an` and `ac` by `on` and `oc`. A skipped observation does not decay them.
 4. Divide `ac` by `an` plus a ridge of `1e-6` times the larger of `abs(an)` and `eps`, giving `calib`.
 5. Return `calib` when `min_obs` observations have advanced the accumulators, and `NaN` before then.

# Arguments

  - `P`: The uncalibrated prediction, `matured observations × assets`.
  - `fwd`: Forward mean idiosyncratic returns, `observations × assets`.
  - `vs`: Idiosyncratic variance history, `observations × assets`.
  - `w`: Cross-sectional weights, `observations × assets`.
  - `decay`: The decay factor.
  - `min_obs`: The warm-up, in observations that advanced the accumulators.

# Returns

  - `calib::Real`: The calibration coefficient, or `NaN` while the fit is in its warm-up.

# Related

  - [`TargetReturnForecast`](@ref)
  - [`target_forecast_calibration_design`](@ref)
  - [`return_forecast`](@ref)
"""
function target_forecast_calibration(P::MatNum, fwd::MatNum, vs::MatNum, w::MatNum,
                                     decay::Real, min_obs::Integer)::Real
    Tf = promote_type(real(eltype(P)), real(eltype(fwd)))
    an = zero(Tf)
    ac = zero(Tf)
    calib = Tf(NaN)
    n = 0
    for t in axes(P, 1)
        a, y, wv = target_forecast_calibration_design(P, fwd, vs, w, t)
        s = length(wv) < 2 ? zero(Tf) : sum(wv) / length(wv)
        if !(isfinite(s) && s > zero(s))
            continue
        end
        u = wv ./ s
        on = LinearAlgebra.dot(u, a .* a)
        oc = LinearAlgebra.dot(u, a .* y)
        if !(isfinite(on) && isfinite(oc) && on > eps(Tf))
            continue
        end
        an = decay * an + (one(Tf) - decay) * on
        ac = decay * ac + (one(Tf) - decay) * oc
        n += 1
        calib = ac / (an + Tf(1e-6) * max(abs(an), eps(Tf)))
    end
    return n >= min_obs ? calib : Tf(NaN)
end
"""
    target_forecast_latest(model::Nothing, S::Arr3Num) -> Matrix{<:Real}
    target_forecast_latest(model, S::Arr3Num) -> Matrix{<:Real}

Predict the uncalibrated forecast of the latest observation of a [`TargetReturnForecast`](@ref).

# Algorithm

The method that Julia selects is the algorithm.

 1. `nothing`: the member fitted no model, so every asset reads `NaN`.
 2. A fitted model: the model predicts the assets whose scores are all finite in one call, and the other assets read `NaN`.

# Arguments

  - `model`: The fitted model, or `nothing`.
  - `S`: The Descriptor scores, `observations × assets × descriptors`.

# Returns

  - `P::Matrix{<:Real}`: The uncalibrated forecast of the last observation, `1 × assets`.

# Related

  - [`TargetReturnForecast`](@ref)
  - [`target_forecast_fit`](@ref)
  - [`return_forecast`](@ref)
"""
function target_forecast_latest(::Nothing, S::Arr3Num)::Matrix{<:Real}
    Tf = real(eltype(S))
    return fill(Tf(NaN), 1, size(S, 2))
end
function target_forecast_latest(model, S::Arr3Num)::Matrix{<:Real}
    Tf = real(eltype(S))
    A = Tf.(view(S, size(S, 1), :, :))
    P = fill(Tf(NaN), 1, size(A, 1))
    idx = Int[]
    for i in axes(A, 1)
        ok = true
        for k in axes(A, 2)
            ok &= isfinite(A[i, k])
        end
        if ok
            push!(idx, i)
        end
    end
    if !isempty(idx)
        P[1, idx] = StatsAPI.predict(model, A[idx, :])
    end
    return P
end
"""
    target_forecast_latest_variances(vs::Nothing) -> Nothing
    target_forecast_latest_variances(vs::MatNum) -> MatNum

Return the idiosyncratic variances of the latest observation, as a one-row matrix.

[`forecast_return_units`](@ref) converts a history, and the member converts only the last row of one. So the function returns that row as a one-row matrix and not as a vector.

# Arguments

  - `vs`: Idiosyncratic variance history, `observations × assets`, or `nothing`.

# Returns

  - `vs::Option{<:MatNum}`: The last row as a `1 × assets` matrix, or `nothing`.

# Related

  - [`TargetReturnForecast`](@ref)
  - [`forecast_return_units`](@ref)
"""
function target_forecast_latest_variances(::Nothing)::Nothing
    return nothing
end
function target_forecast_latest_variances(vs::MatNum)::MatNum
    return vs[size(vs, 1):size(vs, 1), :]
end
"""
    target_forecast_alignment(whole_history::Bool, S::Arr3Num, eps::MatNum,
                              vs::Option{<:MatNum}, w::MatNum,
                              groups::Option{<:AbstractMatrix{<:Integer}},
                              rows::AbstractUnitRange) -> NamedTuple

Put the scores of a [`TargetReturnForecast`](@ref) and the histories it fits on one observation axis.

# Algorithm

 1. `whole_history` is set: the function keeps the scores, the weights and the group labels on the axis of the returns data. It places the two block histories into the block's rows through [`return_forecast_pad`](@ref). A row before the block carries a `NaN` idiosyncratic return, so it states a target only where its forward window reaches into the block.
 2. `whole_history` is not set: the function cuts the scores, the weights and the group labels to the block's rows through [`return_forecast_cut`](@ref). The two block histories are already on those rows.

# Arguments

  - $(arg_dict[:rf_whole_history])
  - `S`: The Descriptor scores, `observations × assets × descriptors`, on the axis of the returns data.
  - `eps`: Idiosyncratic returns of the block, `observations × assets`.
  - `vs`: Idiosyncratic variance history of the block, `observations × assets`, or `nothing`.
  - `w`: Cross-sectional weights, `observations × assets`, on the axis of the returns data.
  - `groups`: Group label matrix on the axis of the returns data, or `nothing`.
  - `rows`: The rows of the returns data the block lives on.

# Returns

  - `(S, eps, vs, w, groups)::NamedTuple`: The same five, on one observation axis.

# Related

  - [`TargetReturnForecast`](@ref)
  - [`return_forecast_pad`](@ref)
  - [`return_forecast_cut`](@ref)
  - [`return_forecast_rows`](@ref)
"""
function target_forecast_alignment(whole_history::Bool, S::Arr3Num, eps::MatNum,
                                   vs::Option{<:MatNum}, w::MatNum,
                                   groups::Option{<:AbstractMatrix{<:Integer}},
                                   rows::AbstractUnitRange)
    if whole_history
        T = size(S, 1)
        return (; S = S, eps = return_forecast_pad(eps, rows, T),
                vs = return_forecast_pad(vs, rows, T), w = w, groups = groups)
    end
    return (; S = return_forecast_cut(S, rows), eps = eps, vs = vs,
            w = return_forecast_cut(w, rows), groups = return_forecast_cut(groups, rows))
end
"""
    return_forecast(rfe::TargetReturnForecast, rd::ReturnsResult,
                    csfm::CrossSectionalFactorModel,
                    cre::Option{<:AbstractCrossSectionalRegressionEstimator} = nothing) -> TargetReturnForecastResult

Fit a Return Forecast with a regression target over every observation and asset at once.

The four-argument method also fits the calibration coefficient `κ⊥` of the orthogonal part of the prediction. [`OrthogonalPartCalibration`](@ref) of [`CrossSectionalFactorPrior`](@ref) asks for it, and the three-argument method, which a standalone caller and every other rule read, never pays for it.

# Algorithm

 1. Compute the Descriptor scores `S` over all the returns data through [`descriptor_scores`](@ref), and read the idiosyncratic returns off the block. Read the variances through [`target_forecast_variances`](@ref), which states when the member needs them.
 2. Put the scores and the two block histories on one observation axis through [`target_forecast_alignment`](@ref), which reads `whole_history`. Under `intercept`, append a slice of ones to the scores, so that the fit, the out-of-fold fits and the latest prediction all read the constant.
 3. Take the forward mean target `fwd` through [`forward_mean_returns`](@ref). Convert it to the Forecast Unit through [`forecast_unit_target`](@ref), and pass it through `target_outlier` and then `target_scoring`, giving `y`.
 4. Count the observations whose target has matured, all but the last `lag + horizon - 1`, giving `nt`. Flatten them into one sample per `(observation, asset)` pair through [`target_forecast_samples`](@ref), giving `Sf`, `yf` and `ok`.
 5. Fit the regression target on the valid samples through [`target_forecast_fit`](@ref), giving `model`.
 6. Compute the calibration coefficient `calib` and the uncalibrated prediction `P` of the matured observations through [`target_forecast_coefficient`](@ref). A row before the block has no idiosyncratic variance, so it can enter the fit and never enters the calibration. When `cre` is given, compute `κ⊥` from `P` through [`target_forecast_orthogonal_coefficient`](@ref).
 7. Predict the latest observation through [`target_forecast_latest`](@ref) and convert the row to return units, giving `P`. The conversion reads the last row of the variances of the block as they stand. A pair of leverage one is out of the fit and the calibration, and in the Sharpe unit its forecast is zero, because its variance is zero.
 8. Multiply `P` by `scale` and by the multiplier of [`target_forecast_multiplier`](@ref), giving `mu`.

# Arguments

  - `rfe`: Target Return Forecast Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `csfm`: The fitted factor-model block. It must carry the cross-sectional fit, its histories state the block's rows, and it must carry the idiosyncratic variance history under a calibration or under [`IdiosyncraticSharpeUnit`](@ref).
  - `cre`: Cross-Sectional Regression Estimator that splits each row of the prediction for `κ⊥`, or `nothing` to fit no `κ⊥`.

# Validation

  - The rules of [`descriptor_scores`](@ref), of [`forecast_idiosyncratic_returns`](@ref) and of [`target_forecast_variances`](@ref).
  - When `cre` is given, the rules of [`target_forecast_orthogonal_rows`](@ref).

# Returns

  - `rf::TargetReturnForecastResult`: The fitted forecast, the model, the calibration coefficient and, when `cre` is given, `κ⊥`.

# Related

  - [`TargetReturnForecast`](@ref)
  - [`TargetReturnForecastResult`](@ref)
  - [`target_forecast_alignment`](@ref)
  - [`target_forecast_samples`](@ref)
  - [`target_forecast_calibration`](@ref)
  - [`target_forecast_latest`](@ref)
  - [`forecast_return_units`](@ref)
"""
function return_forecast(rfe::TargetReturnForecast, rd::ReturnsResult,
                         csfm::CrossSectionalFactorModel,
                         cre::Option{<:AbstractCrossSectionalRegressionEstimator} = nothing)::TargetReturnForecastResult
    ds = rfe.scores
    (; S, rows) = descriptor_scores(ds, rd, csfm)
    al = target_forecast_alignment(rfe.whole_history, S,
                                   forecast_idiosyncratic_returns(csfm),
                                   target_forecast_variances(rfe.unit, csfm, rfe.calibrate),
                                   return_forecast_weights(rd),
                                   exposure_group_labels(rd, ds.group), rows)
    Sa = if rfe.intercept
        cat(al.S, fill(one(eltype(al.S)), size(al.S, 1), size(al.S, 2), 1); dims = 3)
    else
        al.S
    end
    vs = al.vs
    emsk = al.w
    groups = al.groups
    fwd = forward_mean_returns(al.eps, rfe.horizon, rfe.lag)
    y = exposure_transform(rfe.target_scoring,
                           exposure_transform(rfe.target_outlier,
                                              forecast_unit_target(rfe.unit, fwd, vs), emsk,
                                              groups), emsk, groups)
    nt = max(size(Sa, 1) - (rfe.lag + rfe.horizon - 1), 0)
    Sf, yf, ok = target_forecast_samples(Sa, y, emsk, nt)
    model = target_forecast_fit(rfe, Sf, yf, ok)
    cf = target_forecast_coefficient(rfe, model, Sf, yf, ok, fwd, vs, emsk, nt)
    # Under `whole_history` the rows of `P` are the rows of the returns data, and the block
    # starts at `first(rows)`. Otherwise they are the rows of the block.
    ocalib = target_forecast_orthogonal_coefficient(cre, cf.P, fwd, vs, emsk, rfe, csfm,
                                                    rfe.whole_history ? first(rows) - 1 : 0)
    # The fit reads `NaN` at a pair of leverage one, and the read-out converts with the
    # variance the block holds there. The last row of the block is the last row of `vs`.
    P = forecast_return_units(rfe.unit, target_forecast_latest(model, Sa),
                              target_forecast_latest_variances(csfm.vs))
    return TargetReturnForecastResult(;
                                      mu = vec(rfe.scale .*
                                               target_forecast_multiplier(rfe.calibrate,
                                                                          cf.calib) .* P),
                                      model = model, calib = cf.calib, ocalib = ocalib)
end
"""
    target_forecast_multiplier(calibrate::Bool, calib::Number) -> Number

Return the multiplier a [`TargetReturnForecast`](@ref) applies to its uncalibrated prediction.

A member that does not calibrate publishes the prediction as it stands, so its multiplier is one. The `NaN` in its Result then states that the member fitted no coefficient, and not that the forecast is missing.

# Arguments

  - `calibrate`: Whether the member calibrates.
  - `calib`: The calibration coefficient.

# Returns

  - `m::Number`: The multiplier, `calib` or one in the type of `calib`.

# Related

  - [`TargetReturnForecast`](@ref)
  - [`target_forecast_calibration`](@ref)
"""
function target_forecast_multiplier(calibrate::Bool, calib::Number)::Number
    return calibrate ? calib : one(calib)
end
"""
    target_forecast_coefficient(rfe::TargetReturnForecast, model, Sf::MatNum, yf::VecNum,
                                ok::AbstractVector{Bool}, fwd::MatNum,
                                vs::Option{<:MatNum}, w::MatNum, nt::Integer) -> NamedTuple

Return the calibration coefficient of a [`TargetReturnForecast`](@ref), or `NaN`, and the uncalibrated prediction it was fitted on.

This function alone decides whether the calibration runs. A member that does not calibrate, and a member that fits no model, each return `NaN` and predict nothing. The `NaN` takes the type of the flattened design and of the forward returns. The prediction is returned beside the coefficient, so [`target_forecast_orthogonal_coefficient`](@ref) reads the same out-of-fold prediction and runs no second cross-validation.

# Arguments

  - `rfe`: Target Return Forecast Estimator.
  - `model`: The fitted model, or `nothing`.
  - `Sf`: The flattened design.
  - `yf`: The flattened target.
  - `ok`: The mask of the valid samples.
  - `fwd`: Forward mean idiosyncratic returns, `observations × assets`.
  - `vs`: Idiosyncratic variance history, or `nothing`.
  - `w`: Cross-sectional weights, `observations × assets`.
  - `nt`: Number of observations whose target has matured.

# Returns

  - `calib::Real`: The calibration coefficient, or `NaN`.
  - `P::Option{<:MatNum}`: The uncalibrated prediction in return units, `matured observations × assets`, or `nothing` when the calibration does not run.

# Related

  - [`TargetReturnForecast`](@ref)
  - [`target_forecast_calibration`](@ref)
  - [`target_forecast_uncalibrated`](@ref)
  - [`target_forecast_orthogonal_coefficient`](@ref)
"""
function target_forecast_coefficient(rfe::TargetReturnForecast, model, Sf::MatNum,
                                     yf::VecNum, ok::AbstractVector{Bool}, fwd::MatNum,
                                     vs::Option{<:MatNum}, w::MatNum, nt::Integer)
    if !rfe.calibrate || isnothing(model) || isnothing(vs)
        return (; calib = promote_type(real(eltype(Sf)), real(eltype(fwd)))(NaN),
                P = nothing)
    end
    p = target_forecast_uncalibrated(rfe.cv, rfe, model, Sf, yf, ok, size(w, 2))
    P = forecast_return_units(rfe.unit, target_forecast_scatter(p, nt, size(w, 2)),
                              view(vs, 1:nt, :))
    return (; calib = target_forecast_calibration(P, fwd, vs, w, rfe.decay, rfe.min_obs),
            P = P)
end
"""
    target_forecast_orthogonal_coefficient(cre::Nothing, P::Option{<:MatNum}, fwd::MatNum,
                                           vs::Option{<:MatNum}, w::MatNum,
                                           rfe::TargetReturnForecast,
                                           csfm::CrossSectionalFactorModel,
                                           off::Integer) -> Nothing
    target_forecast_orthogonal_coefficient(cre::AbstractCrossSectionalRegressionEstimator,
                                           P::Option{<:MatNum}, fwd::MatNum,
                                           vs::Option{<:MatNum}, w::MatNum,
                                           rfe::TargetReturnForecast,
                                           csfm::CrossSectionalFactorModel,
                                           off::Integer) -> Real

Fit the calibration coefficient `κ⊥` of the orthogonal part of the prediction of a [`TargetReturnForecast`](@ref).

The member regresses the forward idiosyncratic return, which the cross-sectional fit of the block makes orthogonal to the Factor Exposures. So the part of the prediction that the exposures span predicts nothing of it, and `κ` of the whole prediction under-scales the orthogonal part by the share of its variance. `κ⊥` is the same regression on the orthogonal part alone, so it is the scale of that part.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{p}^{\\perp}_{t} &= \\boldsymbol{p}_{t} - \\mathbf{B}_{t} \\boldsymbol{g}_{t}\\,, \\\\
\\kappa_{\\perp} &= \\frac{C^{\\perp}_{n}}{(1 + \\varrho) A^{\\perp}_{n}}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{p}_{t}``: The uncalibrated prediction of observation ``t``, out of fold under a cross-validation estimator.
  - ``\\mathbf{B}_{t}``: The exposures of the estimated factors at observation ``t`` of the block.
  - ``\\boldsymbol{g}_{t}``: Coefficients of the regression of ``\\boldsymbol{p}_{t}`` on ``\\mathbf{B}_{t}`` under the regression weights of observation ``t`` of the block, as [`cross_sectional_alpha_split`](@ref) fits them.
  - ``A^{\\perp}_{n}``, ``C^{\\perp}_{n}``, ``\\varrho``: The accumulators and the ridge of [`TargetReturnForecast`](@ref), on ``\\boldsymbol{p}^{\\perp}_{t}`` in place of ``\\boldsymbol{p}_{t}``.

# Algorithm

The method that Julia selects is the algorithm.

 1. `nothing`: the caller asks for no `κ⊥`, so return `nothing`.
 2. A Cross-Sectional Regression Estimator: return `NaN` when the calibration did not run. Otherwise split each row of `P` with [`target_forecast_orthogonal_rows`](@ref), and run [`target_forecast_calibration`](@ref) on the orthogonal parts.

# Arguments

  - `cre`: Cross-Sectional Regression Estimator of the split, or `nothing`.
  - `P`: The uncalibrated prediction, `matured observations × assets`, or `nothing` when the calibration did not run.
  - `fwd`: Forward mean idiosyncratic returns, `observations × assets`.
  - `vs`: Idiosyncratic variance history, or `nothing`.
  - `w`: Cross-sectional weights, `observations × assets`.
  - `rfe`: Target Return Forecast Estimator, whose `decay` and `min_obs` the regression reads.
  - `csfm`: The fitted factor-model block.
  - `off`: Number of rows of `P` before the first row of the block.

# Validation

  - The rules of [`target_forecast_orthogonal_rows`](@ref).

# Returns

  - `ocalib::Option{<:Real}`: `κ⊥`, `NaN` while it is in its warm-up, or `nothing`.

# Related

  - [`TargetReturnForecast`](@ref)
  - [`OrthogonalPartCalibration`](@ref)
  - [`target_forecast_coefficient`](@ref)
  - [`target_forecast_calibration`](@ref)
"""
function target_forecast_orthogonal_coefficient(::Nothing, ::Option{<:MatNum}, ::MatNum,
                                                ::Option{<:MatNum}, ::MatNum,
                                                ::TargetReturnForecast,
                                                ::CrossSectionalFactorModel,
                                                ::Integer)::Nothing
    return nothing
end
function target_forecast_orthogonal_coefficient(cre::AbstractCrossSectionalRegressionEstimator,
                                                P::Option{<:MatNum}, fwd::MatNum,
                                                vs::Option{<:MatNum}, w::MatNum,
                                                rfe::TargetReturnForecast,
                                                csfm::CrossSectionalFactorModel,
                                                off::Integer)::Real
    if isnothing(P) || isnothing(vs)
        return real(eltype(fwd))(NaN)
    end
    return target_forecast_calibration(target_forecast_orthogonal_rows(cre, P, csfm, off),
                                       fwd, vs, w, rfe.decay, rfe.min_obs)
end
"""
    target_forecast_orthogonal_rows(cre::AbstractCrossSectionalRegressionEstimator,
                                    P::MatNum, csfm::CrossSectionalFactorModel,
                                    off::Integer) -> Matrix{<:Real}

Split each row of the uncalibrated prediction of a [`TargetReturnForecast`](@ref) against the exposures of its own observation, and keep the orthogonal part.

Row `t` of `P` is observation `t - off` of the block. It splits against the exposures of the estimated factors of that observation, under the regression weights of that observation, through [`cross_sectional_alpha_split`](@ref), which is the split that [`CrossSectionalFactorPrior`](@ref) runs at the latest observation. A row outside the block has no exposure, and a row with no finite prediction has nothing to split, so each of them stays `NaN`.

# Arguments

  - `cre`: Cross-Sectional Regression Estimator of the split.
  - `P`: The uncalibrated prediction, `rows × assets`.
  - `csfm`: The fitted factor-model block.
  - `off`: Number of rows of `P` before the first row of the block.

# Validation

  - `csfm.Ms` and `csfm.rw` are given. Raises an [`IsNothingError`](@ref).
  - The rules of [`cross_sectional_alpha_split`](@ref).

# Returns

  - `Q::Matrix{<:Real}`: The orthogonal part of each row, `rows × assets`, `NaN` where the row has no split.

# Related

  - [`target_forecast_orthogonal_coefficient`](@ref)
  - [`cross_sectional_alpha_split`](@ref)
  - [`estimated_factor_columns`](@ref)
"""
function target_forecast_orthogonal_rows(cre::AbstractCrossSectionalRegressionEstimator,
                                         P::MatNum, csfm::CrossSectionalFactorModel,
                                         off::Integer)::Matrix{<:Real}
    Ms = csfm.Ms
    rw = csfm.rw
    @argcheck(!isnothing(Ms),
              IsNothingError("the calibration of the orthogonal part splits each row of the prediction against the exposures of its observation, and the block carries no exposure history in Ms"))
    @argcheck(!isnothing(rw),
              IsNothingError("the calibration of the orthogonal part splits each row of the prediction under the regression weights of its observation, and the block carries no regression weight history in rw"))
    est = estimated_factor_columns(csfm)
    Tf = real(eltype(P))
    Q = fill(Tf(NaN), size(P))
    for t in axes(P, 1)
        tb = t - off
        if !(1 <= tb <= size(Ms, 1)) || !any(isfinite, view(P, t, :))
            continue
        end
        Q[t, :] = cross_sectional_alpha_split(cre, view(P, t, :), view(Ms, tb, :, est),
                                              view(rw, tb, :)).ap
    end
    return Q
end

export TargetReturnForecast, TargetReturnForecastResult, NaNWarmup, InSampleWarmup,
       PrequentialCalibration, NormalEquationsFit
public AbstractCalibrationWarmup, target_forecast_warmup
