"""
$(DocStringExtensions.TYPEDEF)

Estimates a point-in-time cross-sectional factor model from an Asset Panel, and lifts it onto the assets.

The estimator reads per-asset Panel Fields, builds a Factor Exposure from each one, regresses the returns of every observation on the lagged exposures across the assets, and returns the asset moments beside a [`CrossSectionalFactorModel`](@ref) block. It is the cross-sectional counterpart of [`FactorPrior`](@ref), which regresses the returns of each asset on a factor-return series over time.

The factor list can also hold observed factors, whose returns the caller observes and the fit does not estimate. A member of [`AbstractObservedExposureEstimator`](@ref) declares them: [`CurrencyExposure`](@ref) gives one Currency Factor per currency, and [`ObservedExposure`](@ref) gives one factor, such as a macro series or an observed market return. Each reads its return from the Exogenous Series `rd.E` by name. The regression reads the returns net of the observed factors: by default the fit derives them as `X - Z_obs * r_obs`, and `lx` can name a Panel Field of net returns the caller measured. [`cross_sectional_local_returns`](@ref) states both paths. Under Currency Factors the net returns are the local returns, and `rd.X` stays in the base currency. The observed factors then join the factor model after the estimated factors, whatever their place in the list, so the factor covariance, `mu`, the scenarios, the factor attribution and the exposure constraints see them, and the Neutralisation and the Factor Family Basis pass them through.

The warm-ups of a fit add up, so a caller who sizes a window must sum three of them rather than take the longest. The longest warm-up of the Descriptors sets the first observation of the factor-return history, and `lag` adds `lag` observations to it. `pe` then warms up over that history, and `ve` over the idiosyncratic returns beside it. The default `pe` and the default `ve` each need 40 observations, their `min_obs`, so a default fit needs 40 observations after the Descriptor warm-up and the lag. A window that covers the Descriptors alone can leave `pe` too few observations to state a factor covariance, and the fit then refuses with a message that names the cause. Cross-validation meets this most often. A fold gives the estimator its own rows alone, so the Descriptors warm up again in every fold, and a rolling train window never grows past the warm-up. Size the train window against the sum of the three warm-ups.

On the online step the prior takes one of two routes. Unwrapped, it folds as a carry: its first step of [`partial_fit!`](@ref) seeds a [`CrossSectionalCarryState`](@ref), and each step computes the exposures, the regression and the idiosyncratic variance of the new observations alone. Wrapped in [`Online`](@ref), it refits: each step records the returns, both masks and the Panel Fields in a sample buffer, the call with no data `prior(pe)` is the batch fit over the rows of the buffer, and a `max_history` on the wrapper bounds them. When a member reads the Exogenous Series, as [`reads_exogenous_series`](@ref) answers, the buffer records every column of the series too, so an observed factor takes the refit. The carry fold does not record the series yet, so it refuses such a prior. The automatic choice of a dropped member reads every row, so `choice` states whether each fit chooses again or the first fit pins it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CrossSectionalFactorPrior(; factors::Dict_VecPair,
                              neutralise::Option{<:Dict_VecPair} = nothing,
                              families::Option{<:Dict_VecPair} = nothing,
                              cre::AbstractCrossSectionalRegressionEstimator = CrossSectionalLinearRegression(),
                              wa::AbstractCrossSectionalWeightsAlgorithm = MarketCapWeights(),
                              pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(; me = ExpWeightedExpectedReturns(), ce = RegimeAdjustedExpWeightedCovariance(; centred = true)),
                              ve::AbstractCovarianceEstimator = RegimeAdjustedExpWeightedVariance(; centred = true),
                              ce::StatsBase.CovarianceEstimator = ExpWeightedCovariance(; centred = true),
                              f_mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                              mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                              th::Real = 0.0, bp::Real = 1.0,
                              mcap::AbstractString = "market_cap",
                              bw::AbstractString = "benchmark_weights", lag::Integer = 1,
                              minra::Option{<:Integer} = nothing,
                              rfe::Option{<:AbstractReturnForecastEstimator} = nothing,
                              lambda::Real = 1.0, c::Real = 1.0,
                              lx::Option{<:AbstractString} = nothing,
                              mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm} = nothing,
                              ex::FLoops.Transducers.Executor = ThreadedEx(),
                              choice::AbstractChoiceRule = BatchChoice(),
                              cache::Option{<:AbstractPartialFitState} = nothing) -> CrossSectionalFactorPrior

Keywords correspond to the struct's fields. `factors`, `neutralise` and `families` also take a dictionary, and the constructor collects each one into a vector of Pairs.

## Validation

  - `factors` is not empty and repeats no factor name.
  - No Factor Family holds both estimated and observed factors, and `families` constrains no observed family, as [`assert_cross_sectional_observed_families`](@ref) states.
  - `lx`, when it is stated, is not empty, and `factors` holds an observed factor, because the prior reads net returns only under observed factors.
  - `neutralise` and `families` repeat no key.
  - `th` lies in `[0, 1]`.
  - `bp` is finite and `>= 0`.
  - `mcap` and `bw` are not empty.
  - Every factor whose Exposure Estimator reads benchmark weights names `bw` in its own `bw`, as [`assert_cross_sectional_benchmark_field`](@ref) states.
  - `lag` is `> 0`.
  - `minra`, when it is stated, is `> 0`.
  - `lambda` and `c` lie in `[0, 1]`.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `cre`: Recursively updated via [`factory`](@ref).
  - `wa`: Recursively updated via [`factory`](@ref).
  - `pe`: Recursively updated via [`factory`](@ref).
  - `ve`: Recursively updated via [`factory`](@ref).
  - `ce`: Recursively updated via [`factory`](@ref).
  - `f_mp`: Recursively updated via [`factory`](@ref).
  - `mp`: Recursively updated via [`factory`](@ref).
  - `rfe`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `ve`: Recursively viewed via [`port_opt_view`](@ref).
  - `ce`: Recursively viewed via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> CrossSectionalFactorPrior(; factors = [\"mkt\" => ConstantExposure()], lag = 2).lag
2
```

# Related

  - [`AbstractLowOrderPriorEstimator_A`](@ref)
  - [`prior`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`FactorPrior`](@ref)
  - [`AssetPanel`](@ref)
  - [`factor_exposure`](@ref)
  - [`cross_sectional_regression`](@ref)
  - [`factor_family_basis`](@ref)
  - [`neutralise_exposures!`](@ref)
  - [`AbstractReturnForecastEstimator`](@ref)
  - [`CurrencyExposure`](@ref)
  - [`AbstractObservedExposureEstimator`](@ref)
  - [`ObservedExposure`](@ref)
  - [`cross_sectional_local_returns`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
  - [`Online`](@ref)
  - [`CrossSectionalCarryState`](@ref)
  - [`AbstractChoiceRule`](@ref)
"""
@propagatable @concrete struct CrossSectionalFactorPrior <: AbstractLowOrderPriorEstimator_A
    """
    Pairs of `factor name => Exposure Estimator`. The estimated factors take the factor axis in the order they are written, and the observed factors follow them in the order they are written. A one-hot member contributes one factor per level of its categorical Panel Field, and so does a [`CurrencyExposure`](@ref), so a Pair is not always one factor.
    """
    factors
    """
    Neutralisation, as Pairs of `key => targets` run in order, or `nothing`. A key names a factor or a Factor Family, and so does each target.
    """
    neutralise
    """
    Constrained Factor Families, as Pairs of `family label => dropped member`, or `nothing`. When the dropped member is `nothing`, [`factor_family_basis`](@ref) chooses it.
    """
    families
    """
    Cross-Sectional Regression Estimator of the fit, and of the Neutralisation. It must fit no intercept, because the prior states the moments through the factor returns alone, and the fit refuses an intercept. A `"market" => ConstantExposure()` factor states the common return instead.
    """
    @fprop cre
    """
    Weight policy of the cross-sectional fit. Its `p` is the power the regression weights raise the market capitalisation to.
    """
    @fprop wa
    """
    $(field_dict[:pe]) The fit gives it the reduced factor-return series, so a constrained Factor Family gives it a full-rank covariance. The default is an [`EmpiricalPrior`](@ref) of an [`ExpWeightedExpectedReturns`](@ref) and a [`RegimeAdjustedExpWeightedCovariance`](@ref) with `centred = true`. Its half-life is the 40 observations of the default `ve`, so the systematic part and the specific part of `sigma` answer on one horizon, and after a change of regime both of them move. The factor covariance is centred, which is the convention of `ce` and of `ve`.
    """
    @fprop pe
    """
    $(field_dict[:ve]) [`variance_series`](@ref) on it gives the idiosyncratic variance history, whose last row is the idiosyncratic risk of the latest observation. The default is a [`RegimeAdjustedExpWeightedVariance`](@ref) with `centred = true`, which measures the second moment of each idiosyncratic series about zero. The prior states the mean of an asset through the factors alone, so the model sets the mean of the idiosyncratic return to zero, and the specific risk is its second moment. A variance about a running mean would understate it, and would also move the regime multiplier.
    """
    @fprop @vprop ve
    """
    $(field_dict[:ce]) It estimates the covariance of the standardised idiosyncratic returns, and the fit reads it only when `th` is positive. [`gap_fill_value`](@ref) on it gives the value of the cell of an inactive asset. The default gives `NaN`, so the fit passes the gap and the active mask of the panel to the estimator. A plain moment estimator gives zero, and the fit writes zero into the cell.
    """
    @fprop @vprop ce
    """
    $(field_dict[:f_mp]) It processes the factor covariance that `pe` states, which is a different matrix from the asset covariance that `mp` processes. The factor covariance is on the factor axis, `pe` estimates it from the factor-return series, and a Factor Family that drops a member can leave it singular. [`cross_sectional_lift`](@ref) takes its square root under `mtx_sqrt` for the low-rank square root, so under the default `mtx_sqrt` a factor covariance that is not positive definite fails there rather than in the asset block.
    """
    @fprop f_mp
    """
    $(field_dict[:mp])
    """
    @fprop mp
    """
    Idiosyncratic correlation threshold. A value of zero leaves the idiosyncratic covariance diagonal, and a positive value keeps every correlation above it and zeroes the rest, so the block becomes a matrix.
    """
    th
    """
    Power the benchmark weights raise the market capitalisation to. A value of zero gives every asset of the estimation universe the same benchmark weight, and reads no market capitalisation.
    """
    bp
    """
    Name of the numeric Panel Field holding the market capitalisation.
    """
    mcap
    """
    Name of the numeric Panel Field the prior writes its benchmark weights onto, and the one every Exposure Estimator reads them from. The Asset Panel must not already hold a field of this name.
    """
    bw
    """
    Number of observations by which the exposures lag the returns.
    """
    lag
    """
    Smallest eligible asset count an observation may carry, or `nothing` for `max(2K, 30)` over the reduced factor count `K`.
    """
    minra
    """
    Return Forecast Estimator whose forecast enters `mu`, or `nothing`. It is fitted on the coverage universe, after the factor model, and its forecast is split against the latest Factor Exposures into the part they span and the part they do not.
    """
    @fprop rfe
    """
    Shrinkage of the expected factor returns towards the spanned part of the Return Forecast, in `[0, 1]`. A value of one keeps the fitted factor mean, and a value of zero takes the spanned forecast alone. With no Return Forecast Estimator the spanned part is zero, so the value shrinks the factor mean towards zero.
    """
    lambda
    """
    Confidence in the orthogonal part of the Return Forecast, in `[0, 1]`. It scales the part of the forecast the factors do not span, which the block carries in `b`. A value of zero discards it where it is finite. An asset whose forecast is not finite carries `NaN` in `b` whatever `c` is, because `0 * NaN` is `NaN`, so the prior states no expected return for it and it leaves the Investable Mask. A forecast that is not finite at every asset splits into two zero parts, and changes no moment.
    """
    c
    """
    Name of a numeric Panel Field of the returns net of the observed factors, or `nothing` to derive them as `X - Z_obs * r_obs`. The regression reads the named field unchanged. Under Currency Factors the net returns are the local returns. A caller names a field when some asset has no currency label, which the derived path gives a `NaN` net return, or when their own local returns are not `X - Z_obs * r_obs`, which includes every panel of simple returns because of the cross term, see [`currency_excess_index`](@ref).
    """
    lx
    """
    $(field_dict[:mtx_sqrt])
    """
    mtx_sqrt
    """
    $(field_dict[:ex]) It computes the Factor Exposures of one dependency layer, whose members read no exposure of each other. Each member writes its own columns of the exposure history, so every executor gives the same prior. The regression runs under the executor of `cre`, and a Return Forecast under the executor of its own Descriptor Scores.
    """
    ex
    """
    Choice Rule of the dropped member of each Factor Family whose member is `nothing` in `families`. Under [`BatchChoice`](@ref), the default, each fit chooses again over every observation. Under [`PinnedChoice`](@ref), the online step writes the choice of the first fit into `families` of the estimator that it returns. The two agree in a batch fit.
    """
    choice
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    function CrossSectionalFactorPrior(factors::AbstractVector{<:Pair},
                                       neutralise::Option{<:AbstractVector{<:Pair}},
                                       families::Option{<:AbstractVector{<:Pair}},
                                       cre::AbstractCrossSectionalRegressionEstimator,
                                       wa::AbstractCrossSectionalWeightsAlgorithm,
                                       pe::AbstractLowOrderPriorEstimator_A_AF,
                                       ve::AbstractCovarianceEstimator,
                                       ce::StatsBase.CovarianceEstimator,
                                       f_mp::AbstractMatrixProcessingEstimator,
                                       mp::AbstractMatrixProcessingEstimator, th::Real,
                                       bp::Real, mcap::AbstractString, bw::AbstractString,
                                       lag::Integer, minra::Option{<:Integer},
                                       rfe::Option{<:AbstractReturnForecastEstimator},
                                       lambda::Real, c::Real, lx::Option{<:AbstractString},
                                       mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm},
                                       ex::FLoops.Transducers.Executor,
                                       choice::AbstractChoiceRule,
                                       cache::Option{<:AbstractPartialFitState})
        assert_closed_unit_interval(th, :th)
        assert_finite(bp, :bp)
        assert_nonneg(bp, :bp)
        assert_panel_terms(mcap, :mcap)
        assert_panel_terms(bw, :bw)
        for (key, xe) in factors
            assert_cross_sectional_benchmark_field(key, xe, bw)
        end
        assert_cross_sectional_observed_families(factors, families)
        if !isnothing(lx)
            assert_panel_terms(lx, :lx)
            @argcheck(any(p -> isa(last(p), AbstractObservedExposureEstimator), factors),
                      ArgumentError("lx names the Panel Field of net returns \"$lx\", and the prior reads net returns only under observed factors. Add an observed member to factors, for example CurrencyExposure(), or drop lx"))
        end
        assert_gt0(lag, :lag)
        if !isnothing(minra)
            assert_gt0(minra, :minra)
        end
        assert_closed_unit_interval(lambda, :lambda)
        assert_closed_unit_interval(c, :c)
        return new{typeof(factors), typeof(neutralise), typeof(families), typeof(cre),
                   typeof(wa), typeof(pe), typeof(ve), typeof(ce), typeof(f_mp), typeof(mp),
                   typeof(th), typeof(bp), typeof(mcap), typeof(bw), typeof(lag),
                   typeof(minra), typeof(rfe), typeof(lambda), typeof(c), typeof(lx),
                   typeof(mtx_sqrt), typeof(ex), typeof(choice), typeof(cache)}(factors,
                                                                                neutralise,
                                                                                families,
                                                                                cre, wa, pe,
                                                                                ve, ce,
                                                                                f_mp, mp,
                                                                                th, bp,
                                                                                mcap, bw,
                                                                                lag, minra,
                                                                                rfe, lambda,
                                                                                c, lx,
                                                                                mtx_sqrt,
                                                                                ex, choice,
                                                                                cache)
    end
end
function CrossSectionalFactorPrior(; factors::Dict_VecPair,
                                   neutralise::Option{<:Dict_VecPair} = nothing,
                                   families::Option{<:Dict_VecPair} = nothing,
                                   cre::AbstractCrossSectionalRegressionEstimator = CrossSectionalLinearRegression(),
                                   wa::AbstractCrossSectionalWeightsAlgorithm = MarketCapWeights(),
                                   pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(;
                                                                                            me = ExpWeightedExpectedReturns(),
                                                                                            ce = RegimeAdjustedExpWeightedCovariance(;
                                                                                                                                     centred = true)),
                                   ve::AbstractCovarianceEstimator = RegimeAdjustedExpWeightedVariance(;
                                                                                                       centred = true),
                                   ce::StatsBase.CovarianceEstimator = ExpWeightedCovariance(;
                                                                                             centred = true),
                                   f_mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                                   mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                                   th::Real = 0.0, bp::Real = 1.0,
                                   mcap::AbstractString = "market_cap",
                                   bw::AbstractString = "benchmark_weights",
                                   lag::Integer = 1, minra::Option{<:Integer} = nothing,
                                   rfe::Option{<:AbstractReturnForecastEstimator} = nothing,
                                   lambda::Real = 1.0, c::Real = 1.0,
                                   lx::Option{<:AbstractString} = nothing,
                                   mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm} = nothing,
                                   ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                                   choice::AbstractChoiceRule = BatchChoice(),
                                   cache::Option{<:AbstractPartialFitState} = nothing)::CrossSectionalFactorPrior
    return CrossSectionalFactorPrior(cross_sectional_prior_pairs(factors, :factors),
                                     cross_sectional_prior_option(neutralise, :neutralise),
                                     cross_sectional_prior_option(families, :families), cre,
                                     wa, pe, ve, ce, f_mp, mp, th, bp, mcap, bw, lag, minra,
                                     rfe, lambda, c, lx, mtx_sqrt, ex, choice, cache)
end
"""
    cross_sectional_prior_option(x::Nothing, sym::Sym_Str) -> nothing
    cross_sectional_prior_option(x::Dict_VecPair, sym::Sym_Str) -> Vector{<:Pair}

Collect an optional list-valued argument of a [`CrossSectionalFactorPrior`](@ref).

The Neutralisation and the constrained Factor Families are each absent or a list, so the absent case is a method rather than a test.

# Arguments

  - `x`: The Pairs, the dictionary, or `nothing`.
  - `sym`: Name of the field, for the messages.

# Validation

  - The rules of [`cross_sectional_prior_pairs`](@ref).

# Returns

  - `pr::Option{<:Vector{<:Pair}}`: The collected Pairs, or `nothing`.

# Related

  - [`cross_sectional_prior_pairs`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
function cross_sectional_prior_option(::Nothing, ::Sym_Str)
    return nothing
end
function cross_sectional_prior_option(x::Dict_VecPair, sym::Sym_Str)
    return cross_sectional_prior_pairs(x, sym)
end
"""
    prior(pe::CrossSectionalFactorPrior, X::MatNum, F::Option{<:MatNum} = nothing,
          pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, ne::Option{<:VecStr} = nothing,
          E::Option{<:MatNum} = nothing, iv::Option{<:MatNum} = nothing,
          ivpa::Option{<:Num_VecNum} = nothing, strict::Bool = false,
          kwargs...) -> LowOrderPrior

Fit a cross-sectional factor model on an Asset Panel, and return the asset prior it lifts.

This is the returns-matrix method that every prior estimator implements, and the fit runs in it. The panel is the third positional argument, so a wrapping prior composes this estimator when it forwards the panel it received. The method below that takes a [`ReturnsResult`](@ref) passes its fields to it.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{f}_{t} &= \\underset{\\boldsymbol{f}}{\\arg\\min} \\left(\\boldsymbol{x}^{\\mathrm{loc}}_{t} - \\mathbf{Z}_{t - \\ell} \\boldsymbol{f}\\right)^\\intercal \\mathbf{Q}_{t} \\left(\\boldsymbol{x}^{\\mathrm{loc}}_{t} - \\mathbf{Z}_{t - \\ell} \\boldsymbol{f}\\right)\\,, \\\\
\\boldsymbol{\\varepsilon}_{t} &= \\boldsymbol{x}^{\\mathrm{loc}}_{t} - \\mathbf{Z}_{t - \\ell} \\boldsymbol{f}_{t}\\,, \\\\
\\boldsymbol{g} &= \\underset{\\boldsymbol{g}}{\\arg\\min} \\left(\\boldsymbol{\\alpha}_{T} - \\mathbf{Z}_{T} \\boldsymbol{g}\\right)^\\intercal \\mathbf{Q}_{T} \\left(\\boldsymbol{\\alpha}_{T} - \\mathbf{Z}_{T} \\boldsymbol{g}\\right)\\,, \\\\
\\tilde{\\boldsymbol{\\mu}}_{f} &= \\lambda \\hat{\\boldsymbol{\\mu}}_{f} + (1 - \\lambda) \\boldsymbol{g}\\,, \\\\
\\boldsymbol{\\mu}_{\\mathcal{S}} &= \\mathbf{Z}_{T, \\mathcal{S}} \\tilde{\\boldsymbol{\\mu}}_{f} + c \\left(\\boldsymbol{\\alpha}_{T} - \\mathbf{Z}_{T} \\boldsymbol{g}\\right)_{\\mathcal{S}}\\,, \\\\
\\mathbf{\\Sigma}_{\\mathcal{S} \\mathcal{S}} &= \\mathbf{Z}_{T, \\mathcal{S}} \\hat{\\mathbf{\\Sigma}}_{f} \\mathbf{Z}_{T, \\mathcal{S}}^\\intercal + \\mathbf{D}_{\\mathcal{S} \\mathcal{S}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:f_t_att])
  - $(math_dict[:x_t_obs])
  - ``\\boldsymbol{x}^{\\mathrm{loc}}_{t}``: Returns the regression reads at observation ``t``, ``N \\times 1``. They are ``\\boldsymbol{x}_{t}`` without Currency Factors, and the local returns of [`cross_sectional_local_returns`](@ref) with them.
  - ``\\mathbf{Z}_{t}``: Factor Exposures of observation ``t``, ``N \\times K``, after the Neutralisation and on the reduced factor axis. ``\\mathbf{Z}_{T, \\mathcal{S}}`` holds the rows of the assets of ``\\mathcal{S}``. Under Currency Factors the regression reads the estimated exposures alone, and every other line reads ``\\mathbf{Z}_{t}`` with the currency exposures appended as its last columns, ``\\boldsymbol{f}_{t}`` with the Currency Excess Returns appended, and ``\\boldsymbol{g}`` with zeros appended. The shrinkage ``\\lambda`` then reaches the estimated factors alone, and the expected return of a Currency Factor is the mean the nested factor prior states for it.
  - $(math_dict[:ell_lag_cs])
  - $(math_dict[:Q_t_att]) An asset that is not eligible at observation ``t`` takes a weight of zero.
  - $(math_dict[:eps_t_att])
  - ``\\boldsymbol{g}``: Coefficients of the part of the Return Forecast that the latest exposures span, ``K \\times 1``. It is zero when the prior states no Return Forecast Estimator.
  - $(math_dict[:alpha_t_fc]) The weight of an asset whose forecast or whose exposures are not finite is zero in the fit of ``\\boldsymbol{g}``.
  - ``\\hat{\\boldsymbol{\\mu}}_{f}``, ``\\hat{\\mathbf{\\Sigma}}_{f}``: Expected factor returns and factor covariance that the nested factor prior states over the factor returns ``\\boldsymbol{f}_{t}``.
  - ``\\tilde{\\boldsymbol{\\mu}}_{f}``: Blended expected factor returns, ``K \\times 1``.
  - ``\\lambda \\in [0, 1]``: Shrinkage of the expected factor returns towards ``\\boldsymbol{g}``.
  - ``c \\in [0, 1]``: Confidence in the part of the Return Forecast that the latest exposures do not span.
  - $(math_dict[:mu_er])
  - ``\\mathbf{\\Sigma}``: Asset covariance matrix, ``N \\times N``.
  - $(math_dict[:D_orth])
  - ``\\mathcal{S}``: Assets whose latest exposures are finite, whose systematic part the model states. An inactive asset has no finite exposure, so every asset of ``\\mathcal{S}`` is active at the latest observation.
  - ``\\mathcal{I}``: Investable assets, those that the Asset Panel activates at the latest observation and whose idiosyncratic variance and latest exposures are finite.
  - $(math_dict[:T])
  - $(math_dict[:N])
  - $(math_dict[:K])

The prior states an entry exactly when the model determines it. An asset of ``\\mathcal{S}`` outside ``\\mathcal{I}``, for example one in the warm-up of its idiosyncratic variance, has a finite expected return, and under a diagonal ``\\mathbf{D}`` a finite covariance with each other asset of ``\\mathcal{S}``, because the model sets the idiosyncratic covariance of two assets to zero. Its variance is `NaN`, and under a positive `th` so is each of its covariances, which read its variance. Every entry of ``\\boldsymbol{\\mu}`` and every row and column of ``\\mathbf{\\Sigma}`` outside ``\\mathcal{S}`` is `NaN`. The Investable Mask needs a finite mean and a finite variance, so an optimiser still reads ``\\mathcal{I}`` alone. An Empty Factor, an estimated factor whose exposure is zero at every pair of positive weight, has ``f_{tk} = 0`` at every observation, and ``\\hat{\\mathbf{\\Sigma}}_{f}`` holds a zero row and column for it. It keeps its place on every factor axis, so a sub-universe with no asset in one level of a one-hot factor still fits. Without a Return Forecast Estimator the forecast terms are zero, so ``\\boldsymbol{\\mu}_{\\mathcal{S}} = \\lambda \\mathbf{Z}_{T, \\mathcal{S}} \\hat{\\boldsymbol{\\mu}}_{f}``. At ``\\lambda = 0`` and ``c = 1`` the two parts of the forecast add up to ``\\boldsymbol{\\alpha}_{T}``, so the expected return of an asset of ``\\mathcal{S}`` with a finite forecast is that forecast.

# Algorithm

 1. Orient `X`, `F` and `E` by `dims`, rebuild the returns data that the Descriptors read, from `X`, `F`, `ne`, `E`, `pnl`, `iv` and `ivpa`, and take the two universe masks off `pnl` with [`cross_sectional_panel_masks`](@ref). Split the factor list into estimated and observed members with [`cross_sectional_factor_partition`](@ref).
 2. Build the benchmark weights `BW` with [`cross_sectional_cap_weights`](@ref), over the assets of the estimation universe whose return and market capitalisation are finite, and write them onto a copy of the Asset Panel with [`cross_sectional_benchmark_returns`](@ref). The universe and the warm-up read `X`, or the named net returns under `lx`, so a gap in an observed series does not move them. A benchmark power of zero reads no market capitalisation.
 3. Read the observed factors from the copy with [`cross_sectional_observed`](@ref), so an observed member that wraps a [`CompositeExposure`](@ref) reads the benchmark weights. Take the returns the regression reads, `Xl`, with [`cross_sectional_local_returns`](@ref). Build every estimated Factor Exposure with [`cross_sectional_exposure_history`](@ref), one dependency layer after another and the members of a layer under `ex`, giving `Ms`, `nf` and `fam`. Under observed factors the estimated members read `Xl` in place of `X`, so a Descriptor of the returns measures the returns the regression explains. The first `pe.lag` rows of the derived `Xl` have no lagged exposure, so the derivation takes the exposure of the same observation there. An observed member cannot read `Xl`, because `Xl` is derived from its exposures. So the observed members that can read returns read `X` net of the observed members that read none, as [`cross_sectional_observed`](@ref) states.
 4. Drop the leading observations the Descriptors warm up over, with [`cross_sectional_warmup`](@ref) on the estimated and the observed exposures together, giving the rows `rw`. [`cross_sectional_exposure_stage`](@ref) runs steps 1 to 4.
 5. Neutralise the exposures with [`cross_sectional_neutralise!`](@ref), under the benchmark weights and the prior's own regression estimator.
 6. Build the Factor Family Basis `fb` with [`cross_sectional_family_basis`](@ref), and reduce the exposures through it.
 7. Lag the reduced exposures and the market capitalisation by `pe.lag`, giving `Zl` and `mcl`. Trim the observed factors to the fitted observations with [`cross_sectional_observed_block`](@ref), which refuses a non-finite observed return among them. Take the eligibility mask `msk` of the fit with [`cross_sectional_eligible`](@ref) on `Xl`, and drop from it every pair whose lagged market capitalisation is not finite.
 8. Regress each observation's `Xl` on its lagged reduced exposures with [`cross_sectional_live_regression`](@ref), giving `csr` and the mask `lv` of the factors that are not empty, under the weights `W` of [`cs_weights_initial`](@ref). Refuse a `csr` that carries an intercept with [`assert_cross_sectional_no_intercept`](@ref). When [`needs_second_pass`](@ref) answers `true`, refine the weights with [`cs_weights_refine`](@ref) and regress again. The refinement reads the idiosyncratic variance under the two universe masks, as step 9 does.
 9. Take the idiosyncratic variance history `vs` with [`variance_series`](@ref), under the active mask and the estimation mask of the fitted observations. An estimator that reads the masks resets an asset that the active mask turns off, and measures its regime over the estimation universe alone. An estimator that reads no mask ignores them. Standardise the idiosyncratic returns by `vs` with [`cross_sectional_standardised_residuals`](@ref), giving `S`, and take the latest idiosyncratic covariance `esigma` with [`cross_sectional_idiosyncratic_covariance`](@ref), from the same residuals with no fill, so a gap stays a gap for the correlation. Record the degrees of freedom and the divisor of each variance with [`variance_count`](@ref) and [`cross_sectional_variance_counts`](@ref).
10. Append the observed factors after the estimated ones with [`cross_sectional_observed_append`](@ref): the observed returns after the factor returns, the observed exposures after the loadings and the exposure history, the names and the family labels, and pass-through factors on the Factor Family Basis. Fit `pe.pe` on the combined reduced factor returns of the factors that are not empty with [`cross_sectional_factor_moments`](@ref), giving `f_pr`, which refuses a non-finite factor moment and processes the factor covariance under `pe.f_mp`, the matrix processing estimator of the factor axis and not the asset one. An Observed Factor is never empty. The method passes `strict` to `pe.pe`, as [`FactorPrior`](@ref) does, because the slot admits [`BlackLittermanPrior`](@ref) and [`EntropyPoolingPrior`](@ref), which resolve view names against a universe. [`cross_sectional_assemble`](@ref) runs the standardisation and the counts of step 9 and the steps 11 to 17, and the call with no data of the carry fold runs it too.
11. Build the [`CrossSectionalFactorModel`](@ref) block `csfm`, with the raw exposures of the latest observation in `M`, the reduced ones `L` beside them, a zero `b`, and the observed returns in `fx`. Under a family re-basis, expand the combined factor returns onto the raw axis with [`cross_sectional_expand`](@ref), each row with the basis of its lagged exposures, and store them in `fr`.
12. Fit the Return Forecast with [`cross_sectional_return_forecast`](@ref), on the full returns data that the estimated members read, so that a Descriptor of the forecast warms up over every observation the panel has, giving the block `rr` with the orthogonal part in `b` and the Result in `rf`. Under observed factors the forecast thus reads `Xl`, and it forecasts the net return of each asset, which the split measures against loadings of the same returns. Blend the spanned part into the mean of the estimated factors with [`cross_sectional_forecast_mu`](@ref), and keep the mean of the observed factors, giving `f_mu`.
13. Expand the blended factor moments onto the raw factor axis with [`cross_sectional_expand`](@ref), so `fpr` states the distribution of the factors the caller named. Its factor returns are `fr`, or the combined factor returns when no family is constrained.
14. Take the investable assets `idx` with [`cross_sectional_investable`](@ref).
15. Rebuild the asset return scenarios `Xs` with [`cross_sectional_scenarios`](@ref).
16. Lift the reduced factor distribution onto the assets with [`cross_sectional_lift`](@ref), and add `b` to the expected return it answers. The lift states every entry the model determines: the investable block, and the mean and the systematic covariances of an asset with finite exposures and no idiosyncratic variance.
17. Assemble a [`LowOrderPrior`](@ref) over `Xs`, with the fitted base-currency returns under `o_X`, the three lifted moments, the factor prior's `w`, `ens`, `kld` and `ow`, the block under `rr`, and the expanded factor prior under `fpr`. The factor returns of `fpr` and the returns of `o_X` keep the last rows alone, as many as `Xs` holds, so a factor prior with a Scenario Cap pairs each scenario with its own observation.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:F]) The fit does not read it, because this estimator builds its own factors from the panel. The method writes it onto the rebuilt returns data only so that the returns data states what the caller held.
  - $(arg_dict[:pnl_prior]) This estimator reads it, and refuses without it.
  - `dims`: Dimension along which the observations lie.
  - `ne`: Names of the Exogenous Series, written onto the rebuilt returns data.
  - `E`: The Exogenous Series, oriented by `dims` as `X` is. The observed factors read their returns from it.
  - `iv`: Implied volatilities, written onto the rebuilt returns data.
  - `ivpa`: Implied-volatility risk-premium adjustment, written onto the rebuilt returns data.
  - $(arg_dict[:strict]) It is forwarded to the nested factor prior `pe.pe`.
  - `kwargs...`: Additional keyword arguments passed to the verbs of the algorithm.

# Validation

  - `pnl` is not `nothing`. Raises an [`IsNothingError`](@ref).
  - The Asset Panel is time-varying. Raises an `ArgumentError`.
  - The Asset Panel holds no Panel Field named `pe.bw`, because the fit writes its benchmark weights onto that name. Raises an `ArgumentError`.
  - The history is longer than the exposure lag. Raises an `ArgumentError`.
  - At least two observations are left after the Descriptor warm-up and the exposure lag, because a covariance of one observation is not a number. Raises an `ArgumentError`.
  - Every fitted observation carries at least `minra` eligible assets. Raises an `ArgumentError`.
  - Under observed factors, `E` carries a column for every observed series, and every observed return of the fitted observations is finite. Raises an [`IsNothingError`](@ref), an `ArgumentError` or an [`IsNonFiniteError`](@ref), from [`cross_sectional_observed`](@ref) and [`cross_sectional_observed_block`](@ref).
  - At least one estimated factor is not empty. Raises an `ArgumentError`.
  - The regression fits no intercept. The prior states the moments through the factor returns alone, so an intercept would leave its mean out of `mu` and its variance out of `sigma`. Raises an `ArgumentError`.
  - The factor prior states a finite factor mean and a finite factor covariance. Raises an [`IsNonFiniteError`](@ref).
  - At least one asset is investable at the latest observation. Raises an [`IsEmptyError`](@ref).
  - The rules of every verb the algorithm names.

# Returns

  - `pr::LowOrderPrior`: The prior on the full asset universe. An entry that the model does not determine is `NaN`: the whole entry of `mu` and row and column of `sigma` of an asset with no finite latest exposure, and the variance of an asset with no idiosyncratic variance, with its covariances under a positive `th`. `rr` is a [`CrossSectionalFactorModel`](@ref), and `fpr` is the factor prior on the raw factor axis.

# Related

  - [`CrossSectionalFactorPrior`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`investable_mask`](@ref)
  - [`cross_sectional_lift`](@ref)
  - [`cross_sectional_return_forecast`](@ref)
  - [`cross_sectional_forecast_mu`](@ref)
"""
function prior(pe::CrossSectionalFactorPrior, X::MatNum, F::Option{<:MatNum} = nothing,
               pnl::Option{<:AssetPanel} = nothing; dims::Int = 1,
               ne::Option{<:VecStr} = nothing, E::Option{<:MatNum} = nothing,
               iv::Option{<:MatNum} = nothing, ivpa::Option{<:Num_VecNum} = nothing,
               strict::Bool = false, kwargs...)
    X, F, E = dims_oriented(dims, X, F, E)
    @argcheck(!isnothing(pnl),
              IsNothingError("a Cross-Sectional Factor Prior reads its Factor Exposures off an Asset Panel, and the panel is nothing. Call prior(pe, rd) with a ReturnsResult whose `pnl` is the one asset_panel returns, or hand the panel to this method as its third positional argument."))
    st = cross_sectional_exposure_stage(pe, X, F, pnl; ne = ne, E = E, iv = iv, ivpa = ivpa)
    (; amsk, emsk, mcap, BW, cc, Xl, rde, Ms, nf, fam, rw) = st
    Msw = Ms[rw, :, :]
    Xw = X[rw, :]
    Xlw = Xl[rw, :]
    bww = BW[rw, :]
    mcw = cross_sectional_rows(mcap, rw)
    cross_sectional_neutralise!(pe.neutralise, Msw, pe.cre, bww, nf, fam)
    fb = cross_sectional_family_basis(pe.families, Msw, bww, nf, fam)
    @argcheck(length(rw) > pe.lag,
              ArgumentError("the exposures lag the returns by lag = $(pe.lag), so a fit needs more than $(pe.lag) observations after the Descriptor warm-up, and $(length(rw)) are left. Give more observations, shorten the warm-up of the Descriptors, or lower lag."))
    @argcheck(length(rw) - pe.lag >= 2,
              ArgumentError("the factor prior states a covariance of the factor returns, and a covariance of one observation is not a number, so a fit needs at least two observations after the Descriptor warm-up and the exposure lag of $(pe.lag), and $(length(rw) - pe.lag) is left. Give more observations, or shorten the warm-up of the Descriptors."))
    r = (pe.lag + 1):length(rw)
    Zl = fb.Ms[r .- pe.lag, :, :]
    Xr = Xlw[r, :]
    cb = cross_sectional_observed_block(cc, rw, r)
    amr = amsk[rw[r], :]
    bwr = bww[r, :]
    msk = cross_sectional_eligible(Xr, Zl, emsk[rw[r], :])
    mcl = cross_sectional_rows(mcw, r .- pe.lag)
    cross_sectional_cap_finite!(msk, mcl)
    assert_cross_sectional_coverage(msk, cross_sectional_minra(pe, size(Zl, 3)))
    W = cs_weights_initial(pe.wa, mcl, msk)
    (; csr, lv) = cross_sectional_live_regression(pe.cre, Zl, Xr, W)
    assert_cross_sectional_no_intercept(csr)
    # The idiosyncratic variance is a folding statistic, so it reads both universe masks: it
    # resets an asset the active mask turns off, and a regime-adjusted estimator measures its
    # regime over the estimation universe alone (ADR 0172). An estimator that takes no mask
    # ignores the two keywords.
    emr = emsk[rw[r], :]
    if needs_second_pass(pe.wa)
        W = cs_weights_refine(pe.wa, W, csr.eps, pe.ve, msk; estimation_mask = emr,
                              active_mask = amr)
        (; csr, lv) = cross_sectional_live_regression(pe.cre, Zl, Xr, W)
    end
    vs = variance_series(pe.ve, csr.eps; dims = 1, estimation_mask = emr, active_mask = amr)
    # `strict` reaches the nested factor prior for the reason it reaches `FactorPrior`'s:
    # the slot admits `BlackLittermanPrior` and `EntropyPoolingPrior`, whose views name
    # factors on an axis the caller declared, and a name the axis lacks is the caller's
    # error to hear about under `strict`.
    ca = cross_sectional_observed_append(cb, csr.f, fb.Ms[r[end], :, :], Msw[r, :, :], nf,
                                         fam, fb.fcb)
    # The factor covariance takes its own estimator for the reason the asset one takes
    # `pe.mp`: they are different matrices. This one is estimated from the factor-return
    # series over a factor axis a constrained Family has already reduced, and
    # `cross_sectional_lift` factorises it for the low-rank square root, so under the default
    # `mtx_sqrt` a covariance that is merely positive SEMI-definite -- a short warm-up, a
    # collinear Family -- raises a `PosDefException` out of the Cholesky rather than
    # answering. The default `pdm` is a no-op on a matrix that is already positive definite,
    # so a healthy fit is untouched.
    # An Empty Factor has no return to estimate a variance from, so the factor prior and
    # `pe.f_mp` read the other factors. An Observed Factor carries the return the caller
    # observed, so it is never empty.
    f_pr = cross_sectional_factor_moments(pe.pe, pe.f_mp, ca.f,
                                          vcat(lv, trues(size(ca.f, 2) - length(lv)));
                                          strict = strict, kwargs...)
    fit = (; csr = csr, W = W, vs = vs, cnt = variance_count(pe.ve, csr.eps), amr = amr,
           bwr = bwr, Xo = Xw[r, :], r = r)
    return cross_sectional_assemble(pe, f_pr, ca, fit, rde; kwargs...)
end
"""
    cross_sectional_assemble(pe::CrossSectionalFactorPrior, f_pr::NamedTuple, ca::NamedTuple,
                             fit::NamedTuple, rde::ReturnsResult; kwargs...) -> LowOrderPrior

Builds the Prior Result of a [`CrossSectionalFactorPrior`](@ref) from its fitted regression, its idiosyncratic variance history and its factor moments.

The returns-matrix method of [`prior`](@ref) calls it after the variance history. The call with no data of the carry fold calls it with the histories it carries and the factor moments of its folded factor prior. So the two routes build the result with one code, and they differ only in how they reach its inputs.

# Algorithm

 1. Record the degrees of freedom and the divisor of each variance with [`cross_sectional_variance_counts`](@ref), from the count `cnt`.
 2. Standardise the idiosyncratic returns by `vs` with [`cross_sectional_standardised_residuals`](@ref), giving `S`. Take the latest idiosyncratic covariance `esigma` with [`cross_sectional_idiosyncratic_covariance`](@ref), from the same residuals with no fill, so a gap stays a gap for the correlation.
 3. Build the [`CrossSectionalFactorModel`](@ref) block `csfm`, with the raw exposures of the latest observation in `M`, the reduced ones `L` beside them, and a zero `b`. Under a family re-basis, expand the factor returns onto the raw axis with [`cross_sectional_expand`](@ref), each row with the basis of its lagged exposures, and store them in `fr`.
 4. Fit the Return Forecast with [`cross_sectional_return_forecast`](@ref) on `rde`, giving the block `rr` with the orthogonal part in `b`. Blend the spanned part into the mean of the estimated factors with [`cross_sectional_forecast_mu`](@ref), and keep the mean of the observed factors, giving `f_mu`.
 5. Expand the blended factor moments onto the raw factor axis with [`cross_sectional_expand`](@ref).
 6. Take the investable assets `idx` with [`cross_sectional_investable`](@ref), and rebuild the asset return scenarios `Xs` with [`cross_sectional_scenarios`](@ref).
 7. Lift the reduced factor distribution onto the assets with [`cross_sectional_lift`](@ref), and add `b` to the expected return it answers.
 8. Assemble a [`LowOrderPrior`](@ref) over `Xs`, with the base-currency returns of the scenario rows under `o_X`, the three lifted moments, the factor prior's `w`, `ens`, `kld` and `ow`, the block under `rr`, and the expanded factor prior under `fpr`.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `f_pr`: The factor moments that [`cross_sectional_factor_moments`](@ref) states over the combined reduced factor returns.
  - `ca`: The estimated factors with the observed factors appended, from [`cross_sectional_observed_append`](@ref).
  - `fit`: The fitted observations: `csr`, the regression; `W`, its weights; `vs`, the idiosyncratic variance history; `cnt`, the answer of [`variance_count`](@ref) on the residuals; `amr`, the active mask; `bwr`, the benchmark weights; `Xo`, the base-currency returns; and `r`, the fitted rows among the rows after the Descriptor warm-up.
  - `rde`: The returns data that the estimated members read, which the Return Forecast reads.
  - `kwargs...`: Additional keyword arguments passed to [`cross_sectional_lift`](@ref).

# Validation

  - At least one asset is investable at the latest observation. Raises an [`IsEmptyError`](@ref).
  - The rules of every verb the algorithm names.

# Returns

  - `pr::LowOrderPrior`: The prior on the full asset universe, as [`prior`](@ref) states it.

# Related

  - [`prior`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function cross_sectional_assemble(pe::CrossSectionalFactorPrior, f_pr::NamedTuple,
                                  ca::NamedTuple, fit::NamedTuple, rde::ReturnsResult;
                                  kwargs...)
    (; csr, W, vs, cnt, amr, bwr, Xo, r) = fit
    (; edof, ediv) = cross_sectional_variance_counts(cnt, csr)
    S = cross_sectional_standardised_residuals(csr.eps, vs, amr)
    # The correlation reads the residuals with no fill: the fill of the scenarios writes the
    # mean of the other assets of an observation into a gap, and that value correlates the
    # asset with each of them (#1384). The threshold of zero reads no residual.
    Sc = if iszero(pe.th)
        S
    else
        cross_sectional_standardised_residuals(csr.eps, vs, amr; filled = false)
    end
    esigma = cross_sectional_idiosyncratic_covariance(pe.th, pe.ce, pe.mp.pdm, Sc,
                                                      vs[end, :], amr)
    fnow = cross_sectional_basis_now(ca.fcb, r)
    # The block's `fcb` covers its own rows alone, and the factor return of row `t` is stated
    # in the basis of row `t - lag`, so the raw-axis history is expanded here, where the basis
    # of the rows before the first fitted row still exists (#1422).
    fr = cross_sectional_expand(ca.fcb, r, pe.lag, ca.f)
    L = ca.L
    Msr = ca.Ms
    Tb = promote_type(real(eltype(L)), real(eltype(f_pr.mu)))
    csfm = CrossSectionalFactorModel(; M = Msr[end, :, :],
                                     L = cross_sectional_reduced_loadings(fnow, L),
                                     b = zeros(Tb, size(Xo, 2)), csr = csr, Ms = Msr,
                                     vs = vs, esigma = esigma, edof = edof, ediv = ediv,
                                     rw = W, bw = bwr, nf = ca.nf, fam = ca.fam, fcb = fnow,
                                     lag = pe.lag, fx = ca.fx, fr = fr)
    # The forecast reads the returns the estimated members read, so under observed factors it
    # forecasts the net return, which the split measures against loadings of the same returns.
    (; rr, g) = cross_sectional_return_forecast(pe.rfe, rde, csfm, pe.cre, pe.c)
    # The forecast spans the estimated factors alone, so the blend reaches their mean and
    # the mean of each observed factor is the one the factor prior states.
    Ke = size(csr.f, 2)
    f_mu = @views vcat(cross_sectional_forecast_mu(pe.lambda, f_pr.mu[1:Ke], g),
                       f_pr.mu[(Ke + 1):end])
    ex = cross_sectional_expand(ca.fcb, r, f_mu, f_pr.sigma)
    ev = vs[end, :]
    idx = cross_sectional_investable(@view(amr[end, :]), L, ev)
    @argcheck(!isempty(idx),
              IsEmptyError("no asset is investable at the latest observation: every asset is either inactive, or carries a non-finite idiosyncratic variance or Factor Exposure. Give more observations, or widen the active mask of the Asset Panel."))
    Xs = cross_sectional_scenarios(f_pr.X, L, S, ev)
    # A factor prior with a Scenario Cap carries its last rows alone, and the scenarios pair
    # the last rows of each history, so the factor returns and the original returns keep the
    # rows the scenarios keep (#1384).
    rs = (length(r) - size(Xs, 1) + 1):length(r)
    lift = cross_sectional_lift(pe.mp, L, f_mu, f_pr.sigma, esigma, idx, Xs;
                                mtx_sqrt = pe.mtx_sqrt, kwargs...)
    fpr = LowOrderPrior(; X = something(fr, ca.f)[rs, :], mu = ex.mu, sigma = ex.sigma,
                        w = f_pr.w, ens = f_pr.ens, kld = f_pr.kld, ow = f_pr.ow)
    return LowOrderPrior(; X = Xs, o_X = Xo[rs, :], mu = lift.mu + rr.b, sigma = lift.sigma,
                         chol = lift.chol, w = f_pr.w, ens = f_pr.ens, kld = f_pr.kld,
                         ow = f_pr.ow, rr = rr, fpr = fpr)
end
"""
    cross_sectional_exposure_stage(pe::CrossSectionalFactorPrior, X::MatNum,
                                   F::Option{<:MatNum}, pnl::AssetPanel;
                                   ne::Option{<:VecStr} = nothing,
                                   E::Option{<:MatNum} = nothing,
                                   iv::Option{<:MatNum} = nothing,
                                   ivpa::Option{<:Num_VecNum} = nothing) -> NamedTuple

Runs the steps of the fit of a [`CrossSectionalFactorPrior`](@ref) that read the whole history: the benchmark weights, the observed factors, the returns that the regression reads, the exposure history, and the rows after the Descriptor warm-up.

The returns-matrix method of [`prior`](@ref) and the online step under a [`PinnedChoice`](@ref) both read these steps. So the choice that the step pins is the choice that the fit over the same rows makes.

# Algorithm

 1. Rebuild the returns data, take the two universe masks and build the benchmark weights `BW` with [`cross_sectional_benchmark_stage`](@ref). Split the factor list into estimated and observed members with [`cross_sectional_factor_partition`](@ref).
 2. Read the observed factors with [`cross_sectional_observed`](@ref), take the returns the regression reads, `Xl`, with [`cross_sectional_local_returns`](@ref), and build every estimated Factor Exposure with [`cross_sectional_exposure_history`](@ref).
 3. Take the rows `rw` after the Descriptor warm-up with [`cross_sectional_warmup`](@ref), on the estimated and the observed exposures together.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - $(arg_dict[:X]) It holds one observation per row.
  - $(arg_dict[:F]) The steps do not read it. They write it onto the rebuilt returns data.
  - `pnl`: Time-varying Asset Panel of the rows of `X`.
  - `ne`: Names of the Exogenous Series.
  - `E`: The Exogenous Series, one observation per row. The observed factors read their returns from it.
  - `iv`: Implied volatilities, written onto the rebuilt returns data.
  - `ivpa`: Implied-volatility risk-premium adjustment, written onto the rebuilt returns data.

# Validation

  - The rules of every verb the algorithm names.

# Returns

  - `st::NamedTuple`: The fields `amsk` and `emsk`, the active mask and the estimation mask; `mcap`, the market capitalisation or `nothing`; `BW`, the benchmark weights; `cc`, the observed factors or `nothing`; `Xl`, the returns the regression reads; `rde`, the returns data that the estimated members read; `Ms`, `nf` and `fam`, the exposure history of the estimated factors with their names and Factor Family labels; and `rw`, the rows after the warm-up.

# Related

  - [`CrossSectionalFactorPrior`](@ref)
  - [`prior`](@ref)
  - [`cross_sectional_pinned_families`](@ref)
"""
function cross_sectional_exposure_stage(pe::CrossSectionalFactorPrior, X::MatNum,
                                        F::Option{<:MatNum}, pnl::AssetPanel;
                                        ne::Option{<:VecStr} = nothing,
                                        E::Option{<:MatNum} = nothing,
                                        iv::Option{<:MatNum} = nothing,
                                        ivpa::Option{<:Num_VecNum} = nothing)
    (; amsk, emsk, mcap, BW, rdb, Xu) = cross_sectional_benchmark_stage(pe, X, F, pnl;
                                                                        ne = ne, E = E,
                                                                        iv = iv,
                                                                        ivpa = ivpa)
    (; est, obs) = cross_sectional_factor_partition(pe.factors)
    cc = cross_sectional_observed(obs, rdb, pe.lag)
    Xl = cross_sectional_local_returns(pe.lx, cc, X, rdb, pe.lag)
    # The estimated members read the returns the regression explains. Under observed factors
    # those are the net returns, so a Descriptor of the returns measures the move of an asset
    # net of its currency, and not the move of the currency it holds.
    rde = if isnothing(cc)
        rdb
    else
        ReturnsResult(; nx = rdb.nx, X = Xl, nf = rdb.nf, F = rdb.F, ne = rdb.ne, E = rdb.E,
                      iv = rdb.iv, ivpa = rdb.ivpa, pnl = rdb.pnl)
    end
    (; Ms, nf, fam) = cross_sectional_exposure_history(est, rde, pe.ex)
    # An observed member warms up too: one that wraps a Descriptor gives no exposure over the
    # warm-up of the Descriptor, and its derived net return is then not finite.
    Mo = isnothing(cc) ? Ms : cat(Ms, cc.Z; dims = 3)
    rw = (cross_sectional_warmup(Xu, Mo, emsk) + 1):size(X, 1)
    return (; amsk = amsk, emsk = emsk, mcap = mcap, BW = BW, cc = cc, Xl = Xl, rde = rde,
            Ms = Ms, nf = nf, fam = fam, rw = rw)
end
"""
    cross_sectional_benchmark_stage(pe::CrossSectionalFactorPrior, X::MatNum,
                                    F::Option{<:MatNum}, pnl::AssetPanel;
                                    ne::Option{<:VecStr} = nothing,
                                    E::Option{<:MatNum} = nothing,
                                    iv::Option{<:MatNum} = nothing,
                                    ivpa::Option{<:Num_VecNum} = nothing) -> NamedTuple

Rebuilds the returns data of a [`CrossSectionalFactorPrior`](@ref), and writes its benchmark weights onto the Asset Panel.

[`cross_sectional_exposure_stage`](@ref) runs it first. The carry fold runs it on the rows that it carries, because each of its outputs is a value per row: the benchmark weights of a row read the market capitalisation and the returns of that row alone.

# Algorithm

 1. Rebuild the returns data that the Descriptors read, from `X`, `F`, `ne`, `E`, `pnl`, `iv` and `ivpa`. The names of the assets are the column numbers. Take the two universe masks off `pnl` with [`cross_sectional_panel_masks`](@ref).
 2. Take the returns `Xu` that the universe and the warm-up read: `X`, or the named net returns under `pe.lx`.
 3. Build the benchmark weights `BW` with [`cross_sectional_cap_weights`](@ref), over the assets of the estimation universe whose return and market capitalisation are finite. A benchmark power of zero reads no market capitalisation.
 4. Write `BW` onto a copy of the Asset Panel with [`cross_sectional_benchmark_returns`](@ref).

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - $(arg_dict[:X]) It holds one observation per row.
  - $(arg_dict[:F]) The steps do not read it. They write it onto the rebuilt returns data.
  - `pnl`: Time-varying Asset Panel of the rows of `X`.
  - `ne`: Names of the Exogenous Series.
  - `E`: The Exogenous Series, one observation per row.
  - `iv`: Implied volatilities, written onto the rebuilt returns data.
  - `ivpa`: Implied-volatility risk-premium adjustment, written onto the rebuilt returns data.

# Validation

  - The rules of every verb the algorithm names.

# Returns

  - `st::NamedTuple`: The fields `amsk` and `emsk`, the active mask and the estimation mask; `mcap`, the market capitalisation or `nothing`; `BW`, the benchmark weights; `rdb`, the returns data with the benchmark weights on its panel; and `Xu`, the returns that the universe and the warm-up read.

# Related

  - [`cross_sectional_exposure_stage`](@ref)
  - [`cross_sectional_cap_weights`](@ref)
  - [`cross_sectional_benchmark_returns`](@ref)
"""
function cross_sectional_benchmark_stage(pe::CrossSectionalFactorPrior, X::MatNum,
                                         F::Option{<:MatNum}, pnl::AssetPanel;
                                         ne::Option{<:VecStr} = nothing,
                                         E::Option{<:MatNum} = nothing,
                                         iv::Option{<:MatNum} = nothing,
                                         ivpa::Option{<:Num_VecNum} = nothing)
    # Every Descriptor of the fit reads its Panel Fields off the returns data, so the
    # returns data is rebuilt here rather than demanded from the caller: the panel is the
    # one field of it a wrapping prior can forward, and no verb of the fit reads `nx`,
    # `ts`, `nb` or `B`.
    #
    # A `ReturnsResult` holds names for its returns, and neither a matrix nor a panel
    # states any, so the names are the column numbers. They are read nowhere. Their length
    # is: it is the asset axis `check_asset_panel` binds the panel to, which is the check
    # this rebuild is worth making.
    rd = ReturnsResult(; nx = string.(1:size(X, 2)), X = X,
                       nf = isnothing(F) ? nothing : string.(1:size(F, 2)), F = F, ne = ne,
                       E = E, iv = iv, ivpa = ivpa, pnl = pnl)
    amsk, emsk = cross_sectional_panel_masks(pnl)
    # The warm-up and the benchmark universe read the returns the caller stated: `X`, or the
    # named net returns. The derived net returns hold the observed returns, so a gap in an
    # observed series over the warm-up would otherwise move it. Neither reads the observed
    # factors, so the benchmark weights are written before any member reads the panel: an
    # observed member that wraps a composite reads them too.
    Xu = isnothing(pe.lx) ? X : descriptor_field_values(rd, pe.lx)
    mcap = if cross_sectional_needs_market_cap(pe.bp, pe.wa)
        descriptor_field_values(rd, pe.mcap)
    else
        nothing
    end
    bmsk = isfinite.(Xu) .& emsk
    cross_sectional_cap_finite!(bmsk, mcap)
    BW = cross_sectional_cap_weights(pe.bp, mcap, bmsk)
    rdb = cross_sectional_benchmark_returns(rd, pe.bw, BW)
    return (; amsk = amsk, emsk = emsk, mcap = mcap, BW = BW, rdb = rdb, Xu = Xu)
end
"""
    prior(pe::CrossSectionalFactorPrior, rd::ReturnsResult; kwargs...) -> LowOrderPrior

Fit a Cross-Sectional Factor Prior from a [`ReturnsResult`](@ref).

The method passes the fields of `rd` to the returns-matrix method above, which runs the fit. The file defines this method instead of using [`prior(pe::AbstractPriorEstimator, rd::ReturnsResult)`](@ref), so that the refusal of an `rd` with no Asset Panel names `rd.pnl`, the field of `rd` that the caller built.

# Algorithm

 1. Check that `rd` carries asset returns.
 2. Call the returns-matrix method with `rd.X`, `rd.F` and `rd.pnl`, forwarding `rd.ne`, `rd.E`, `rd.iv` and `rd.ivpa` as keyword arguments alongside `kwargs`, and `dims = 1` last, because a `ReturnsResult` holds its observations along the rows.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - $(arg_dict[:rd]) It must carry asset returns in `rd.X` and a time-varying Asset Panel in `rd.pnl`.
  - `kwargs...`: Additional keyword arguments passed to the returns-matrix method.

# Validation

  - `rd.X` is not `nothing`. Raises an [`IsNothingError`](@ref).
  - `rd.pnl` is not `nothing`. Raises an [`IsNothingError`](@ref).
  - The rules of the returns-matrix method.

# Returns

  - `pr::LowOrderPrior`: The prior the returns-matrix method fitted.

# Related

  - [`CrossSectionalFactorPrior`](@ref)
  - [`prior`](@ref)
  - [`ReturnsResult`](@ref)
"""
function prior(pe::CrossSectionalFactorPrior, rd::ReturnsResult; kwargs...)
    @argcheck(!isnothing(rd.X),
              IsNothingError("a Cross-Sectional Factor Prior regresses asset returns on their Factor Exposures, and rd.X is nothing"))
    @argcheck(!isnothing(rd.pnl),
              IsNothingError("a Cross-Sectional Factor Prior reads its Factor Exposures off an Asset Panel, and rd.pnl is nothing. Build the ReturnsResult with the `pnl` that asset_panel returns."))
    return prior(pe, rd.X, rd.F, rd.pnl; ne = rd.ne, E = rd.E, iv = rd.iv, ivpa = rd.ivpa,
                 kwargs..., dims = 1)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a Cross-Sectional Factor Prior whose tree reads the Exogenous Series on the carry fold.

An observed factor reads its returns from the Exogenous Series, and so does an [`EWMacroSensitivity`](@ref) that names a column of it. The carry fold does not record the series yet, so a call with no data could not rebuild what those members read. The refit route records it, so it does not call this function. The check reads [`reads_exogenous_series`](@ref), which answers per type, so a Descriptor that reads the series is refused with the observed members.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.

# Validation

  - `reads_exogenous_series(pe)` is `false`. An `ArgumentError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`reads_exogenous_series`](@ref)
  - [`cross_sectional_carry_fold`](@ref)
"""
function assert_cross_sectional_online_factors(pe::CrossSectionalFactorPrior)::Nothing
    @argcheck(!reads_exogenous_series(pe),
              ArgumentError("the carry fold of a Cross-Sectional Factor Prior does not record the Exogenous Series that an observed factor or a macro sensitivity reads, so its call with no data could not rebuild what they read. Wrap the prior in `Online` to take the refit route, which records the series, or fit in batch."))
    return nothing
end
function refit_prior_step(pe::CrossSectionalFactorPrior, rd::ReturnsResult)
    return cross_sectional_pin_choice(pe.choice, refit_prior_fold(pe, rd))
end
"""
    cross_sectional_pin_choice(choice::BatchChoice, pe::CrossSectionalFactorPrior) -> CrossSectionalFactorPrior
    cross_sectional_pin_choice(choice::PinnedChoice, pe::CrossSectionalFactorPrior) -> CrossSectionalFactorPrior

Applies the Choice Rule of a Cross-Sectional Factor Prior after an online step.

A batch choice chooses again at each call with no data, so the step returns `pe` unchanged. A pinned choice writes the dropped member of each Factor Family into `families`, once the buffer holds the rows of a first fit.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`BatchChoice`](@ref): return `pe`.
 2. [`PinnedChoice`](@ref): take the families with [`cross_sectional_pinned_families`](@ref). Return `pe` when it answers `nothing`, and otherwise `pe` rebuilt with those families.

# Arguments

  - `choice`: The Choice Rule of `pe`.
  - `pe`: The prior after the fold of the step.

# Validation

  - The rules of [`cross_sectional_pinned_families`](@ref).

# Returns

  - `pe::CrossSectionalFactorPrior`: The prior that the step returns.

# Related

  - [`partial_fit!`](@ref)
  - [`cross_sectional_pinned_families`](@ref)
"""
function cross_sectional_pin_choice(::BatchChoice, pe::CrossSectionalFactorPrior)
    return pe
end
function cross_sectional_pin_choice(::PinnedChoice, pe::CrossSectionalFactorPrior)
    families = cross_sectional_pinned_families(pe)
    return isnothing(families) ? pe : rebuild_estimator(pe, (; families = families))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the Factor Families of a Cross-Sectional Factor Prior with the dropped member of each one named, as the fit over the rows of its buffer chooses it, or `nothing` when there is nothing to pin yet.

This is the step of a [`PinnedChoice`](@ref). It runs the steps of the fit that the choice reads, with [`cross_sectional_exposure_stage`](@ref), the Neutralisation and [`factor_family_basis`](@ref), over the rows of the buffer. So the name it writes is the member that the call with no data after the same step drops, and every later call with no data drops it too.

# Algorithm

 1. Answer `nothing` when `pe.families` is `nothing`, or names the dropped member of every family. Nothing is left to pin.
 2. Answer `nothing` when the buffer records no Asset Panel. The call with no data refuses that buffer, so no fit is made.
 3. Run [`cross_sectional_exposure_stage`](@ref) over the rows of the buffer, with the Exogenous Series that the buffer records. Answer `nothing` when fewer than `pe.lag + 2` rows remain after the Descriptor warm-up, because the fit refuses them, so no first fit is made yet.
 4. Neutralise the exposures after the warm-up with [`cross_sectional_neutralise!`](@ref), and build the Factor Family Basis with [`factor_family_basis`](@ref).
 5. Name the dropped member of each family in the order of `pe.families`.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator whose `cache` holds a [`SampleBufferState`](@ref).

# Validation

  - The rules of every verb the algorithm names.

# Returns

  - `families::Option{<:Vector{<:Pair}}`: Pairs of `family label => dropped member`, or `nothing`.

# Related

  - [`cross_sectional_pin_choice`](@ref)
  - [`PinnedChoice`](@ref)
  - [`factor_family_basis`](@ref)
"""
function cross_sectional_pinned_families(pe::CrossSectionalFactorPrior)
    if isnothing(pe.families) || all(p -> !isnothing(last(p)), pe.families)
        return nothing
    end
    state = pe.cache
    pnl = sample_buffer_panel(state)
    if isnothing(pnl)
        return nothing
    end
    (; BW, Ms, nf, fam, rw) = cross_sectional_exposure_stage(pe, sample_buffer(state),
                                                             nothing, pnl;
                                                             exogenous_buffer_kwargs(state)...)
    if length(rw) - pe.lag < 2
        return nothing
    end
    Msw = Ms[rw, :, :]
    bww = BW[rw, :]
    cross_sectional_neutralise!(pe.neutralise, Msw, pe.cre, bww, nf, fam)
    return cross_sectional_dropped_names(pe.families,
                                         factor_family_basis(pe.families, Msw, bww, nf,
                                                             fam), nf)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the cross-sectional factor axis a [`CrossSectionalFactorPrior`](@ref) will produce, before any fit.

The method forwards `pe.factors` to the Pairs method of [`cross_sectional_factor_axis`](@ref), with its observed members moved after the estimated ones by [`cross_sectional_axis_pairs`](@ref), so a caller who holds the estimator reads the axis off it rather than copying the Pairs. The observed factors are the last names of the axis, as they are in the fitted block. The method reads the levels of a one-hot member off the Asset Panel that `rd` carries, so the field index of the panel sets the answer, and the answer is the same in every fold.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `rd`: Returns data carrying the Asset Panel the one-hot levels are read from.

# Validation

  - The rules of the Pairs method of [`cross_sectional_factor_axis`](@ref).

# Returns

  - `nf::Vector{String}`: The raw factor names, in column order.
  - `fam::Vector{String}`: The Factor Family label of each name.

# Related

  - [`CrossSectionalFactorPrior`](@ref)
  - [`cross_sectional_factor_axis`](@ref)
  - [`cross_sectional_factor_sets`](@ref)
"""
function cross_sectional_factor_axis(pe::CrossSectionalFactorPrior, rd::ReturnsResult)
    return cross_sectional_factor_axis(cross_sectional_axis_pairs(pe.factors), rd)
end
"""
    cross_sectional_factor_sets(pe::CrossSectionalFactorPrior, rd::ReturnsResult,
                                sets::Option{<:UniverseSets} = nothing) -> UniverseSets

Declare the cross-sectional factor axis a [`CrossSectionalFactorPrior`](@ref) will produce, and its Factor Family groups, on a [`UniverseSets`](@ref).

The method forwards `pe.factors` to the Pairs method of [`cross_sectional_factor_sets`](@ref), with its observed members moved after the estimated ones by [`cross_sectional_axis_pairs`](@ref), so the observed factors take their names and a group under their family labels. A caller who writes a [`FactorSpace`](@ref) mandate against the estimator, in a [`Pipeline`](@ref) step or in a fold of a cross-validation, declares the universe from the estimator it will fit, so nobody types the list of one-hot levels by hand.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `rd`: Returns data carrying the Asset Panel the one-hot levels are read from, and the asset names a new sets declares.
  - `sets`: A declared universe to widen. When it is `nothing`, a new one is built over `rd.nx` with the default key prefixes.

# Validation

  - The rules of the Pairs method of [`cross_sectional_factor_sets`](@ref).

# Returns

  - `sets::UniverseSets`: The declared universe, carrying the cross-sectional factor axis and one group per Factor Family.

# Related

  - [`CrossSectionalFactorPrior`](@ref)
  - [`cross_sectional_factor_sets`](@ref)
  - [`cross_sectional_factor_axis`](@ref)
  - [`UniverseSets`](@ref)
  - [`FactorSpace`](@ref)
"""
function cross_sectional_factor_sets(pe::CrossSectionalFactorPrior, rd::ReturnsResult,
                                     sets::Option{<:UniverseSets} = nothing)::UniverseSets
    return cross_sectional_factor_sets(cross_sectional_axis_pairs(pe.factors), rd, sets)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the Pairs whose factor axis a [`CrossSectionalFactorPrior`](@ref) produces: its estimated members, then its observed members.

The fit puts the observed factors after the estimated factors, so the axis declared before the fit puts them in the same place.

# Arguments

  - `factors`: Pairs of `factor name => Exposure Estimator`.

# Returns

  - `pr::Vector{<:Pair}`: The Pairs of the axis.

# Related

  - [`cross_sectional_factor_partition`](@ref)
  - [`cross_sectional_factor_axis`](@ref)
  - [`cross_sectional_factor_sets`](@ref)
"""
function cross_sectional_axis_pairs(factors::AbstractVector{<:Pair})
    (; est, obs) = cross_sectional_factor_partition(factors)
    return vcat(est, obs)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a [`CrossSectionalFactorPrior`](@ref) except `cache`.

The state a `cache` holds is the running detail of an incremental fit, not the configuration a reader looks the type up for. Set `set_show_nothing_fields!(:CrossSectionalFactorPrior, true)` to render it.

# Arguments

  - `::CrossSectionalFactorPrior`: Prior estimator, read for its type alone.

# Returns

  - `fields::Tuple`: The field names to render, every field but `cache`.

# Related

  - [`CrossSectionalFactorPrior`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
function show_fields(::CrossSectionalFactorPrior)
    return (:factors, :neutralise, :families, :cre, :wa, :pe, :ve, :ce, :f_mp, :mp, :th,
            :bp, :mcap, :bw, :lag, :minra, :rfe, :lambda, :c, :lx, :mtx_sqrt, :ex, :choice)
end
function factor_residual_config(::CrossSectionalFactorPrior)
    # The declaration names a variance estimator that a consumer re-runs on the
    # reconstruction error to rebuild the residual block and subtract it (see
    # [`factor_residual_config`](@ref)). This estimator's block is not that quantity: it is
    # the last row of an idiosyncratic variance history, and under a positive `th` it is a
    # full matrix. The block it added is on the result, at `rr.esigma`, so a consumer reads
    # it there rather than rebuilding it. An explicit `nothing` would say that no block was
    # added, which is false, so the method refuses instead.
    return throw(ArgumentError("a Cross-Sectional Factor Prior states no residual declaration. The block it adds is the idiosyncratic covariance it measured, which the result carries at `rr.esigma`; it is not `var(ve, X - posterior_X)`, so a consumer that rebuilds the block from a variance estimator would subtract a different matrix."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the smallest eligible asset count of an observation of a Cross-Sectional Factor Prior.

The batch fit and the carry fold refuse an observation with fewer eligible assets, through [`assert_cross_sectional_coverage`](@ref).

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `K`: Number of factors on the reduced factor axis.

# Returns

  - `minra::Integer`: `pe.minra`, or `max(2K, 30)` when it is `nothing`.

# Related

  - [`CrossSectionalFactorPrior`](@ref)
  - [`assert_cross_sectional_coverage`](@ref)
"""
function cross_sectional_minra(pe::CrossSectionalFactorPrior, K::Integer)::Integer
    return isnothing(pe.minra) ? max(2 * K, 30) : pe.minra
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Regress each observation on the factors that are not empty, and give every Empty Factor a return of zero.

A factor is empty when its exposure is zero at every pair of positive weight, which are the pairs that the regression reads. A sub-universe with no asset in one level of a one-hot factor leaves the factor of that level empty, and a meta-optimiser gives its sub-problems such sub-universes. The data then state nothing about the return of the factor. The function regresses on the other factors and writes a zero return, so the answer does not depend on how the solve algorithm of `cre` treats a rank-deficient design.

# Mathematical definition

```math
\\begin{align}
\\mathcal{E} &= \\left\\{k : B_{tik} = 0 \\ \\forall (t, i) : u_{ti} > 0\\right\\}\\,, \\\\
f_{tk} &= 0\\,, \\quad k \\in \\mathcal{E}\\,,\\ t = 1, \\ldots, T\\,.
\\end{align}
```

Where:

  - ``\\mathcal{E}``: Empty Factors.
  - $(math_dict[:B_tik_cs])
  - $(math_dict[:u_ti_cs])
  - $(math_dict[:f_t_att])
  - $(math_dict[:T])

A zero column of the design adds nothing to the fit, so the returns of the other factors are the returns of the regression on every factor. The minimum-norm solution of that regression gives the same zero to an Empty Factor.

# Algorithm

The method with `lv` regresses on the factors that a caller marks. The carry fold of a [`CrossSectionalFactorPrior`](@ref) calls it on its new observations, with the mark of every observation it fitted, so a factor that is empty at the new observations alone keeps its column, as it does in the batch fit.

# Algorithm

 1. Without `lv`, mark the factors that are not empty with [`cross_sectional_live_factors`](@ref), giving `lv`.
 2. If every factor is in `lv`, regress on `Z` with [`cross_sectional_regression`](@ref) and return.
 3. Otherwise, regress on the columns of `Z` at `lv`, giving `csl`. Write its factor returns into the columns at `lv` of a zero matrix `f`, and keep its residuals, its counts and its intercept.

# Arguments

  - `cre`: Cross-sectional regression estimator.
  - `Z`: Exposure tensor `observations × assets × factors`.
  - `X`: Asset returns matrix `observations × assets`.
  - `W`: Cross-sectional weights matrix `observations × assets`.
  - `lv`: `true` at each factor to regress on.

# Validation

  - Without `lv`, at least one factor is not empty. Raises an `ArgumentError`.
  - The rules of [`cross_sectional_design_mask`](@ref) and of [`cross_sectional_regression`](@ref).

# Returns

  - `csr::CrossSectionalRegression`: The regression on every factor of `Z`.
  - `lv::BitVector`: `true` at each factor that is not empty.

# Related

  - [`cross_sectional_regression`](@ref)
  - [`cross_sectional_factor_moments`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
function cross_sectional_live_regression(cre::AbstractCrossSectionalRegressionEstimator,
                                         Z::Arr3Num, X::MatNum, W::MatNum)
    lv = cross_sectional_live_factors(Z, X, W)
    @argcheck(any(lv),
              ArgumentError("every one of the $(size(Z, 3)) factors is empty: no factor has a nonzero exposure at an (observation, asset) pair of positive weight, so the regression has nothing to fit. Widen the eligible cross-section, or give factors that the assets load on."))
    return cross_sectional_live_regression(cre, Z, X, W, lv)
end
function cross_sectional_live_regression(cre::AbstractCrossSectionalRegressionEstimator,
                                         Z::Arr3Num, X::MatNum, W::MatNum,
                                         lv::AbstractVector{Bool})
    if all(lv)
        return (; csr = cross_sectional_regression(cre, Z, X, W), lv = lv)
    end
    csl = cross_sectional_regression(cre, Z[:, :, lv], X, W)
    f = zeros(eltype(csl.f), size(csl.f, 1), size(Z, 3))
    f[:, lv] = csl.f
    return (;
            csr = CrossSectionalRegression(; f = f, eps = csl.eps, n = csl.n, b = csl.b,
                                           h1 = csl.h1), lv = lv)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Mark each factor that is not an Empty Factor over the observations of a design.

A factor is empty when its exposure is zero at every (observation, asset) pair of positive weight. [`cross_sectional_live_regression`](@ref) reads the mark over every observation of its design. The carry fold of a [`CrossSectionalFactorPrior`](@ref) reads it over its new observations, and joins it to the mark of the observations it fitted.

# Algorithm

 1. Take the pairs of positive weight with [`cross_sectional_design_mask`](@ref), giving `act`.
 2. Mark each factor whose exposure is not zero at a pair of `act`.

# Arguments

  - `Z`: Exposure tensor `observations × assets × factors`.
  - `X`: Asset returns matrix `observations × assets`.
  - `W`: Cross-sectional weights matrix `observations × assets`.

# Validation

  - The rules of [`cross_sectional_design_mask`](@ref).

# Returns

  - `lv::BitVector`: `true` at each factor that is not empty.

# Related

  - [`cross_sectional_live_regression`](@ref)
  - [`cross_sectional_design_mask`](@ref)
"""
function cross_sectional_live_factors(Z::Arr3Num, X::MatNum, W::MatNum)::BitVector
    act = findall(cross_sectional_design_mask(Z, X, W))
    return BitVector([any(c -> !iszero(Z[c[1], c[2], k]), act) for k in axes(Z, 3)])
end
function lookback(pe::CrossSectionalFactorPrior)::Option{<:Integer}
    L = lookback(map(last, pe.factors))
    lr = isnothing(pe.rfe) ? 1 : lookback(pe.rfe)
    return isnothing(L) || isnothing(lr) ? nothing : max(L + pe.lag, lr)
end

export CrossSectionalFactorPrior
