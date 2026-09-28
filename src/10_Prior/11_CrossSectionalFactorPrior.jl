"""
$(DocStringExtensions.TYPEDEF)

Estimates a point-in-time cross-sectional factor model from an Asset Panel, and lifts it onto the assets.

The estimator reads per-asset Panel Fields, builds a Factor Exposure from each one, regresses the returns of every observation on the lagged exposures across the assets, and returns the asset moments beside a [`CrossSectionalFactorModel`](@ref) block. It is the cross-sectional counterpart of [`FactorPrior`](@ref), which regresses the returns of each asset on a factor-return series over time.

The factor list can also hold observed factors, whose returns the caller observes and the fit does not estimate. A member of [`AbstractObservedExposureEstimator`](@ref) declares them: [`CurrencyExposure`](@ref) gives one Currency Factor per currency, and [`ObservedExposure`](@ref) gives one factor, such as a macro series or an observed market return. Each reads its return from the Exogenous Series `rd.E` by name. The regression reads the returns net of the observed factors: by default the fit derives them as `X - Z_obs * r_obs`, and `lx` can name a Panel Field of net returns the caller measured. [`cross_sectional_local_returns`](@ref) states both paths. Under Currency Factors the net returns are the local returns, and `rd.X` stays in the base currency. The observed factors then join the factor model after the estimated factors, whatever their place in the list, so the factor covariance, `mu`, the scenarios, the factor attribution and the exposure constraints see them, and the Neutralisation and the Factor Family Basis pass them through.

The warm-ups of a fit add up, so a caller who sizes a window must sum three of them rather than take the longest. The longest warm-up of the Descriptors sets the first observation of the factor-return history, and `lag` adds `lag` observations to it. `pe` then warms up over that history, and `ve` over the idiosyncratic returns beside it. The default `pe` and the default `ve` each need 40 observations, their `min_obs`, so a default fit needs 40 observations after the Descriptor warm-up and the lag. A window that covers the Descriptors alone can leave `pe` too few observations to state a factor covariance, and the fit then refuses with a message that names the cause. Cross-validation meets this most often. A fold gives the estimator its own rows alone, so the Descriptors warm up again in every fold, and a rolling train window never grows past the warm-up. Size the train window against the sum of the three warm-ups.

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
                              lx::Option{<:AbstractString} = nothing) -> CrossSectionalFactorPrior

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
    $(field_dict[:f_mp]) It processes the factor covariance that `pe` states, which is a different matrix from the asset covariance that `mp` processes. The factor covariance is on the factor axis, `pe` estimates it from the factor-return series, and a Factor Family that drops a member can leave it singular. [`cross_sectional_lift`](@ref) takes its Cholesky factor for the low-rank square root, so a factor covariance that is not positive definite fails there rather than in the asset block.
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
                                       lambda::Real, c::Real, lx::Option{<:AbstractString})
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
                   typeof(minra), typeof(rfe), typeof(lambda), typeof(c), typeof(lx)}(factors,
                                                                                      neutralise,
                                                                                      families,
                                                                                      cre,
                                                                                      wa,
                                                                                      pe,
                                                                                      ve,
                                                                                      ce,
                                                                                      f_mp,
                                                                                      mp,
                                                                                      th,
                                                                                      bp,
                                                                                      mcap,
                                                                                      bw,
                                                                                      lag,
                                                                                      minra,
                                                                                      rfe,
                                                                                      lambda,
                                                                                      c, lx)
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
                                   lx::Option{<:AbstractString} = nothing)::CrossSectionalFactorPrior
    return CrossSectionalFactorPrior(cross_sectional_prior_pairs(factors, :factors),
                                     cross_sectional_prior_option(neutralise, :neutralise),
                                     cross_sectional_prior_option(families, :families), cre,
                                     wa, pe, ve, ce, f_mp, mp, th, bp, mcap, bw, lag, minra,
                                     rfe, lambda, c, lx)
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
\\boldsymbol{\\mu}_{\\mathcal{I}} &= \\mathbf{Z}_{T, \\mathcal{I}} \\tilde{\\boldsymbol{\\mu}}_{f} + c \\left(\\boldsymbol{\\alpha}_{T} - \\mathbf{Z}_{T} \\boldsymbol{g}\\right)_{\\mathcal{I}}\\,, \\\\
\\mathbf{\\Sigma}_{\\mathcal{I} \\mathcal{I}} &= \\mathbf{Z}_{T, \\mathcal{I}} \\hat{\\mathbf{\\Sigma}}_{f} \\mathbf{Z}_{T, \\mathcal{I}}^\\intercal + \\mathbf{D}_{\\mathcal{I} \\mathcal{I}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:f_t_att])
  - $(math_dict[:x_t_obs])
  - ``\\boldsymbol{x}^{\\mathrm{loc}}_{t}``: Returns the regression reads at observation ``t``, ``N \\times 1``. They are ``\\boldsymbol{x}_{t}`` without Currency Factors, and the local returns of [`cross_sectional_local_returns`](@ref) with them.
  - ``\\mathbf{Z}_{t}``: Factor Exposures of observation ``t``, ``N \\times K``, after the Neutralisation and on the reduced factor axis. ``\\mathbf{Z}_{T, \\mathcal{I}}`` holds the rows of the investable assets. Under Currency Factors the regression reads the estimated exposures alone, and every other line reads ``\\mathbf{Z}_{t}`` with the currency exposures appended as its last columns, ``\\boldsymbol{f}_{t}`` with the Currency Excess Returns appended, and ``\\boldsymbol{g}`` with zeros appended. The shrinkage ``\\lambda`` then reaches the estimated factors alone, and the expected return of a Currency Factor is the mean the nested factor prior states for it.
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
  - ``\\mathcal{I}``: Investable assets, those that the Asset Panel activates at the latest observation and whose idiosyncratic variance and latest exposures are finite.
  - $(math_dict[:T])
  - $(math_dict[:N])
  - $(math_dict[:K])

Every entry of ``\\boldsymbol{\\mu}`` and every row and column of ``\\mathbf{\\Sigma}`` outside ``\\mathcal{I}`` is `NaN`. Without a Return Forecast Estimator the forecast terms are zero, so ``\\boldsymbol{\\mu}_{\\mathcal{I}} = \\lambda \\mathbf{Z}_{T, \\mathcal{I}} \\hat{\\boldsymbol{\\mu}}_{f}``. At ``\\lambda = 0`` and ``c = 1`` the two parts of the forecast add up to ``\\boldsymbol{\\alpha}_{T}``, so the expected return of an investable asset with a finite forecast is that forecast.

# Algorithm

 1. Orient `X`, `F` and `E` by `dims`, rebuild the returns data that the Descriptors read, from `X`, `F`, `ne`, `E`, `pnl`, `iv` and `ivpa`, and take the two universe masks off `pnl` with [`cross_sectional_panel_masks`](@ref). Split the factor list into estimated and observed members with [`cross_sectional_factor_partition`](@ref). Read the observed factors with [`cross_sectional_observed`](@ref), and the returns the regression reads, `Xl`, with [`cross_sectional_local_returns`](@ref). The warm-up reads `X`, or the named net returns under `lx`, so a gap in an observed series does not move it.
 2. Build the benchmark weights `BW` with [`cross_sectional_cap_weights`](@ref), over the assets of the estimation universe whose return and market capitalisation are finite, and write them onto a copy of the Asset Panel with [`cross_sectional_benchmark_returns`](@ref). A benchmark power of zero reads no market capitalisation.
 3. Build every Factor Exposure with [`cross_sectional_exposure_history`](@ref), in dependency order, giving `Ms`, `nf` and `fam`.
 4. Drop the leading observations the Descriptors warm up over, with [`cross_sectional_warmup`](@ref), giving the rows `rw`.
 5. Neutralise the exposures with [`cross_sectional_neutralise!`](@ref), under the benchmark weights and the prior's own regression estimator.
 6. Build the Factor Family Basis `fb` with [`cross_sectional_family_basis`](@ref), and reduce the exposures through it.
 7. Lag the reduced exposures and the market capitalisation by `pe.lag`, giving `Zl` and `mcl`. Trim the observed factors to the fitted observations with [`cross_sectional_observed_block`](@ref), which refuses a non-finite observed return among them. Take the eligibility mask `msk` of the fit with [`cross_sectional_eligible`](@ref) on `Xl`, and drop from it every pair whose lagged market capitalisation is not finite.
 8. Regress each observation's `Xl` on its lagged reduced exposures, giving `csr`, under the weights `W` of [`cs_weights_initial`](@ref). Refuse a `csr` that carries an intercept. When [`needs_second_pass`](@ref) answers `true`, refine the weights with [`cs_weights_refine`](@ref) and regress again.
 9. Take the idiosyncratic variance history `vs` with [`variance_series`](@ref), standardise the idiosyncratic returns by it with [`cross_sectional_standardised_residuals`](@ref), giving `S`, and take the latest idiosyncratic covariance `esigma` with [`cross_sectional_idiosyncratic_covariance`](@ref). Record the degrees of freedom and the divisor of each variance with [`variance_count`](@ref) and [`cross_sectional_variance_counts`](@ref).
10. Append the observed factors after the estimated ones with [`cross_sectional_observed_append`](@ref): the observed returns after the factor returns, the observed exposures after the loadings and the exposure history, the names and the family labels, and pass-through factors on the Factor Family Basis. Fit `pe.pe` on the combined reduced factor returns, giving `f_pr`, refuse a non-finite factor moment with [`assert_cross_sectional_factor_moments`](@ref), and process the factor covariance in place under `pe.f_mp`, which is the matrix processing estimator of the factor axis and not the asset one. The method passes `strict` to `pe.pe`, as [`FactorPrior`](@ref) does, because the slot admits [`BlackLittermanPrior`](@ref) and [`EntropyPoolingPrior`](@ref), which resolve view names against a universe.
11. Build the [`CrossSectionalFactorModel`](@ref) block `csfm`, with the raw exposures of the latest observation in `M`, the reduced ones `L` beside them, a zero `b`, and the observed returns in `fx`.
12. Fit the Return Forecast with [`cross_sectional_return_forecast`](@ref), on the full returns data, so that a Descriptor of the forecast warms up over every observation the panel has, giving the block `rr` with the orthogonal part in `b` and the Result in `rf`. Blend the spanned part into the mean of the estimated factors with [`cross_sectional_forecast_mu`](@ref), and keep the mean of the observed factors, giving `f_mu`.
13. Expand the blended factor moments onto the raw factor axis with [`cross_sectional_expand`](@ref), so `fpr` states the distribution of the factors the caller named.
14. Take the investable assets `idx` with [`cross_sectional_investable`](@ref).
15. Rebuild the asset return scenarios `Xs` with [`cross_sectional_scenarios`](@ref).
16. Lift the reduced factor distribution onto the investable assets with [`cross_sectional_lift`](@ref), and add `b` to the expected return it answers.
17. Assemble a [`LowOrderPrior`](@ref) over `Xs`, with the fitted base-currency returns under `o_X`, the three lifted moments, the factor prior's `w`, `ens`, `kld` and `ow`, the block under `rr`, and the expanded factor prior under `fpr`.

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
  - The regression fits no intercept. The prior states the moments through the factor returns alone, so an intercept would leave its mean out of `mu` and its variance out of `sigma`. Raises an `ArgumentError`.
  - The factor prior states a finite factor mean and a finite factor covariance. Raises an [`IsNonFiniteError`](@ref).
  - At least one asset is investable at the latest observation. Raises an [`IsEmptyError`](@ref).
  - The rules of every verb the algorithm names.

# Returns

  - `pr::LowOrderPrior`: The prior on the full asset universe. At an asset that the estimator states no moment for, the entry of `mu` and the row and the column of `sigma` are `NaN`. `rr` is a [`CrossSectionalFactorModel`](@ref), and `fpr` is the factor prior on the raw factor axis.

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
    (; est, obs) = cross_sectional_factor_partition(pe.factors)
    cc = cross_sectional_observed(obs, rd)
    Xl = cross_sectional_local_returns(pe.lx, cc, X, rd, pe.lag)
    # The warm-up and the benchmark universe read the returns the caller stated: `X`, or the
    # named net returns. The derived net returns hold the observed returns, so a gap in an
    # observed series over the warm-up would otherwise move it.
    Xu = isnothing(pe.lx) ? X : Xl
    mcap = if cross_sectional_needs_market_cap(pe.bp, pe.wa)
        panel_field_values(rd, pe.mcap)
    else
        nothing
    end
    bmsk = isfinite.(Xu) .& emsk
    cross_sectional_cap_finite!(bmsk, mcap)
    BW = cross_sectional_cap_weights(pe.bp, mcap, bmsk)
    (; Ms, nf, fam) = cross_sectional_exposure_history(est,
                                                       cross_sectional_benchmark_returns(rd,
                                                                                         pe.bw,
                                                                                         BW))
    rw = (cross_sectional_warmup(Xu, Ms, emsk) + 1):size(X, 1)
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
    assert_cross_sectional_coverage(msk, if isnothing(pe.minra)
                                        max(2 * size(Zl, 3), 30)
                                    else
                                        pe.minra
                                    end)
    W = cs_weights_initial(pe.wa, mcl, msk)
    csr = cross_sectional_regression(pe.cre, Zl, Xr, W)
    @argcheck(isnothing(csr.b),
              ArgumentError("a Cross-Sectional Factor Prior states its moments through the factor returns alone, and its regression estimator fitted an intercept, whose mean and variance the moments would leave out. Give cre an estimator with intercept = false, and state the common return as a factor, for example \"market\" => ConstantExposure()."))
    if needs_second_pass(pe.wa)
        W = cs_weights_refine(pe.wa, W, csr.eps, pe.ve, msk)
        csr = cross_sectional_regression(pe.cre, Zl, Xr, W)
    end
    vs = variance_series(pe.ve, csr.eps; dims = 1)
    (; edof, ediv) = cross_sectional_variance_counts(variance_count(pe.ve, csr.eps), csr)
    S = cross_sectional_standardised_residuals(csr.eps, vs, amr)
    esigma = cross_sectional_idiosyncratic_covariance(pe.th, pe.ce, pe.mp.pdm, S,
                                                      vs[end, :], amr)
    # `strict` reaches the nested factor prior for the reason it reaches `FactorPrior`'s:
    # the slot admits `BlackLittermanPrior` and `EntropyPoolingPrior`, whose views name
    # factors on an axis the caller declared, and a name the axis lacks is the caller's
    # error to hear about under `strict`.
    ca = cross_sectional_observed_append(cb, csr.f, fb.Ms[r[end], :, :], Msw[r, :, :], nf,
                                         fam, fb.fcb)
    f_pr = prior(pe.pe, ca.f; strict = strict)
    assert_cross_sectional_factor_moments(f_pr.mu, f_pr.sigma, length(r))
    # The factor covariance takes its own estimator for the reason the asset one takes
    # `pe.mp`: they are different matrices. This one is estimated from the factor-return
    # series over a factor axis a constrained Family has already reduced, and
    # `cross_sectional_lift` factorises it for the low-rank square root, so a covariance
    # that is merely positive SEMI-definite -- a short warm-up, a collinear Family -- raises
    # a `PosDefException` out of the Cholesky rather than answering. The default `pdm` is a
    # no-op on a matrix that is already positive definite, so a healthy fit is untouched,
    # and `f_pr` is local to this method: nothing outside it holds the matrix.
    matrix_processing!(pe.f_mp, f_pr.sigma, ca.f; kwargs...)
    fnow = cross_sectional_basis_now(ca.fcb, r)
    L = ca.L
    Msr = ca.Ms
    Tb = promote_type(real(eltype(L)), real(eltype(f_pr.mu)))
    csfm = CrossSectionalFactorModel(; M = Msr[end, :, :],
                                     L = cross_sectional_reduced_loadings(fnow, L),
                                     b = zeros(Tb, size(X, 2)), csr = csr, Ms = Msr,
                                     vs = vs, esigma = esigma, edof = edof, ediv = ediv,
                                     rw = W, bw = bwr, nf = ca.nf, fam = ca.fam, fcb = fnow,
                                     lag = pe.lag, fx = ca.fx)
    (; rr, g) = cross_sectional_return_forecast(pe.rfe, rd, csfm, pe.cre, pe.c)
    # The forecast spans the estimated factors alone, so the blend reaches their mean and
    # the mean of each observed factor is the one the factor prior states.
    Ke = size(csr.f, 2)
    f_mu = vcat(cross_sectional_forecast_mu(pe.lambda, f_pr.mu[1:Ke], g),
                f_pr.mu[(Ke + 1):end])
    ex = cross_sectional_expand(ca.fcb, r, pe.lag, ca.f, f_mu, f_pr.sigma)
    ev = vs[end, :]
    idx = cross_sectional_investable(@view(amr[end, :]), L, ev)
    @argcheck(!isempty(idx),
              IsEmptyError("no asset is investable at the latest observation: every asset is either inactive, or carries a non-finite idiosyncratic variance or Factor Exposure. Give more observations, or widen the active mask of the Asset Panel."))
    Xs = cross_sectional_scenarios(f_pr.X, L, S, ev)
    lift = cross_sectional_lift(pe.mp, L, f_mu, f_pr.sigma, esigma, idx, Xs; kwargs...)
    fpr = LowOrderPrior(; X = ex.f, mu = ex.mu, sigma = ex.sigma, w = f_pr.w,
                        ens = f_pr.ens, kld = f_pr.kld, ow = f_pr.ow)
    return LowOrderPrior(; X = Xs, o_X = Xw[r, :], mu = lift.mu + rr.b, sigma = lift.sigma,
                         chol = lift.chol, w = f_pr.w, ens = f_pr.ens, kld = f_pr.kld,
                         ow = f_pr.ow, rr = rr, fpr = fpr)
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

export CrossSectionalFactorPrior
