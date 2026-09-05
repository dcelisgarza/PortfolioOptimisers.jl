"""
$(DocStringExtensions.TYPEDEF)

Estimates a point-in-time cross-sectional factor model from an Asset Panel, and lifts it onto the assets.

The estimator reads per-asset Panel Fields, builds a Factor Exposure from each one, regresses every observation's returns on the **lagged** exposures across the assets, and returns the asset moments beside a [`CrossSectionalFactorModel`](@ref) block. It is the cross-sectional counterpart of [`FactorPrior`](@ref), which regresses each asset's returns on a factor-return series over time.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CrossSectionalFactorPrior(; factors::Dict_VecPair,
                              neutralise::Option{<:Dict_VecPair} = nothing,
                              families::Option{<:Dict_VecPair} = nothing,
                              cre::AbstractCrossSectionalRegressionEstimator = CrossSectionalLinearRegression(),
                              wa::AbstractCrossSectionalWeightsAlgorithm = MarketCapWeights(),
                              pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
                              ve::AbstractCovarianceEstimator = RegimeAdjustedExpWeightedVariance(),
                              ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                              mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                              th::Real = 0.0, bp::Real = 1.0,
                              mcap::AbstractString = "market_cap",
                              bw::AbstractString = "benchmark_weights", lag::Integer = 1,
                              minra::Option{<:Integer} = nothing)

## Validation

  - `factors` is not empty and repeats no factor name.
  - `neutralise` and `families` repeat no key.
  - `th` lies in `[0, 1]`.
  - `bp` is finite and `>= 0`.
  - `lag` is `> 0`.
  - `minra`, when it is stated, is `> 0`.

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
"""
@propagatable @concrete struct CrossSectionalFactorPrior <: AbstractLowOrderPriorEstimator_A
    """
    Pairs of `factor name => Exposure Estimator`, in the order they take on the factor axis. A one-hot member contributes one factor per level of its categorical Panel Field, so a Pair is not always one factor.
    """
    factors
    """
    Neutralisation, as Pairs of `key => targets` run in order, or `nothing`. A key names a factor or a Factor Family, and so does each target.
    """
    neutralise
    """
    Constrained Factor Families, as Pairs of `family label => dropped member`, or `nothing`. A `nothing` right lets [`factor_family_basis`](@ref) choose the member to drop.
    """
    families
    """
    Cross-Sectional Regression Estimator of the fit, and of the Neutralisation.
    """
    @fprop cre
    """
    Weight policy of the cross-sectional fit. Its `p` is the power the regression weights raise the market capitalisation to.
    """
    @fprop wa
    """
    $(field_dict[:pe]) It is fitted on the **reduced** factor-return series, so a constrained Factor Family gives it a full-rank covariance.
    """
    @fprop pe
    """
    $(field_dict[:ve]) [`variance_series`](@ref) on it gives the idiosyncratic variance history, whose last row is the idiosyncratic risk of the latest observation.
    """
    @fprop @vprop ve
    """
    $(field_dict[:ce]) It estimates the covariance of the standardised idiosyncratic returns, and it is read only when `th` is positive.
    """
    @fprop @vprop ce
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
    Name of the numeric Panel Field the prior writes its benchmark weights onto, and the one every Exposure Estimator reads them from.
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
    function CrossSectionalFactorPrior(factors::AbstractVector{<:Pair},
                                       neutralise::Option{<:AbstractVector{<:Pair}},
                                       families::Option{<:AbstractVector{<:Pair}},
                                       cre::AbstractCrossSectionalRegressionEstimator,
                                       wa::AbstractCrossSectionalWeightsAlgorithm,
                                       pe::AbstractLowOrderPriorEstimator_A_AF,
                                       ve::AbstractCovarianceEstimator,
                                       ce::StatsBase.CovarianceEstimator,
                                       mp::AbstractMatrixProcessingEstimator, th::Real,
                                       bp::Real, mcap::AbstractString, bw::AbstractString,
                                       lag::Integer, minra::Option{<:Integer})
        assert_closed_unit_interval(th, :th)
        assert_finite(bp, :bp)
        assert_nonneg(bp, :bp)
        assert_panel_terms(mcap, :mcap)
        assert_panel_terms(bw, :bw)
        assert_gt0(lag, :lag)
        if !isnothing(minra)
            assert_gt0(minra, :minra)
        end
        return new{typeof(factors), typeof(neutralise), typeof(families), typeof(cre),
                   typeof(wa), typeof(pe), typeof(ve), typeof(ce), typeof(mp), typeof(th),
                   typeof(bp), typeof(mcap), typeof(bw), typeof(lag), typeof(minra)}(factors,
                                                                                     neutralise,
                                                                                     families,
                                                                                     cre,
                                                                                     wa, pe,
                                                                                     ve, ce,
                                                                                     mp, th,
                                                                                     bp,
                                                                                     mcap,
                                                                                     bw,
                                                                                     lag,
                                                                                     minra)
    end
end
function CrossSectionalFactorPrior(; factors::Dict_VecPair,
                                   neutralise::Option{<:Dict_VecPair} = nothing,
                                   families::Option{<:Dict_VecPair} = nothing,
                                   cre::AbstractCrossSectionalRegressionEstimator = CrossSectionalLinearRegression(),
                                   wa::AbstractCrossSectionalWeightsAlgorithm = MarketCapWeights(),
                                   pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
                                   ve::AbstractCovarianceEstimator = RegimeAdjustedExpWeightedVariance(),
                                   ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                                   mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                                   th::Real = 0.0, bp::Real = 1.0,
                                   mcap::AbstractString = "market_cap",
                                   bw::AbstractString = "benchmark_weights",
                                   lag::Integer = 1,
                                   minra::Option{<:Integer} = nothing)::CrossSectionalFactorPrior
    return CrossSectionalFactorPrior(cross_sectional_prior_pairs(factors, :factors),
                                     cross_sectional_prior_option(neutralise, :neutralise),
                                     cross_sectional_prior_option(families, :families), cre,
                                     wa, pe, ve, ce, mp, th, bp, mcap, bw, lag, minra)
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
    prior(pe::CrossSectionalFactorPrior, rd::ReturnsResult; kwargs...) -> LowOrderPrior

Fit a cross-sectional factor model on an Asset Panel, and return the asset prior it lifts.

# Algorithm

 1. Read the returns and the two universe masks off the carrier, with [`cross_sectional_panel_masks`](@ref).
 2. Build the benchmark weights with [`cross_sectional_cap_weights`](@ref), over the assets of the estimation universe whose return is finite, and write them onto a copy of the Asset Panel with [`cross_sectional_benchmark_carrier`](@ref).
 3. Build every Factor Exposure with [`cross_sectional_exposure_history`](@ref), in dependency order, giving `Ms`, `nf` and `fam`.
 4. Drop the leading observations the Descriptors warm up over, with [`cross_sectional_warmup`](@ref).
 5. Neutralise the exposures with [`cross_sectional_neutralise!`](@ref), under the benchmark weights and the prior's own regression estimator.
 6. Build the Factor Family Basis with [`cross_sectional_family_basis`](@ref), and reduce the exposures through it.
 7. Lag the reduced exposures and the market capitalisation by `pe.lag`, and take the eligibility mask of the fit with [`cross_sectional_eligible`](@ref).
 8. Regress each observation's returns on its lagged reduced exposures, through [`cs_weights_initial`](@ref), [`needs_second_pass`](@ref) and [`cs_weights_refine`](@ref).
 9. Take the idiosyncratic variance history with [`variance_series`](@ref), standardise the idiosyncratic returns by it with [`cross_sectional_standardised_residuals`](@ref), and take the latest idiosyncratic covariance with [`cross_sectional_idiosyncratic_covariance`](@ref).
10. Fit `pe.pe` on the reduced factor returns, and expand its moments onto the raw factor axis with [`cross_sectional_expand`](@ref), so `fpr` states the distribution of the factors the caller named.
11. Rebuild the asset return scenarios with [`cross_sectional_scenarios`](@ref).
12. Lift the reduced factor distribution onto the investable assets with [`cross_sectional_lift`](@ref).

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - $(arg_dict[:rd]) It must carry asset returns in `rd.X` and a time-varying Asset Panel in `rd.pnl`.

# Validation

  - `rd.X` and `rd.pnl` are not `nothing`. Raises an [`IsNothingError`](@ref).
  - The Asset Panel is time-varying. Raises an `ArgumentError`.
  - The history is longer than the exposure lag. Raises an `ArgumentError`.
  - Every fitted observation carries at least `minra` eligible assets. Raises an `ArgumentError`.
  - At least one asset is investable at the latest observation. Raises an [`IsEmptyError`](@ref).
  - The rules of every verb the algorithm names.

# Returns

  - `pr::LowOrderPrior`: The prior on the **full** asset universe. `mu` and the diagonal of `sigma` are `NaN` at an asset the estimator states no moment for, `rr` is a [`CrossSectionalFactorModel`](@ref), and `fpr` is the factor prior on the raw factor axis.

# Related

  - [`CrossSectionalFactorPrior`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`investable_mask`](@ref)
  - [`cross_sectional_lift`](@ref)
"""
function prior(pe::CrossSectionalFactorPrior, rd::ReturnsResult; kwargs...)
    X = rd.X
    pnl = rd.pnl
    @argcheck(!isnothing(X),
              IsNothingError("a Cross-Sectional Factor Prior regresses asset returns on their Factor Exposures, and rd.X is nothing"))
    @argcheck(!isnothing(pnl),
              IsNothingError("a Cross-Sectional Factor Prior reads its Factor Exposures off an Asset Panel, and rd.pnl is nothing. Build the carrier with the `pnl` that asset_panel returns."))
    amsk, emsk = cross_sectional_panel_masks(pnl)
    mcap = if cross_sectional_needs_market_cap(pe.bp, pe.wa)
        panel_field_values(rd, pe.mcap)
    else
        nothing
    end
    bmsk = isfinite.(X) .& emsk
    cross_sectional_cap_finite!(bmsk, mcap)
    BW = cross_sectional_cap_weights(pe.bp, mcap, bmsk)
    (; Ms, nf, fam) = cross_sectional_exposure_history(pe.factors,
                                                       cross_sectional_benchmark_carrier(rd,
                                                                                         pe.bw,
                                                                                         BW))
    rw = (cross_sectional_warmup(X, Ms, emsk) + 1):size(X, 1)
    Msw = Ms[rw, :, :]
    Xw = X[rw, :]
    bww = BW[rw, :]
    mcw = cross_sectional_rows(mcap, rw)
    cross_sectional_neutralise!(pe.neutralise, Msw, pe.cre, bww, nf, fam)
    fb = cross_sectional_family_basis(pe.families, Msw, bww, nf, fam)
    @argcheck(length(rw) > pe.lag,
              ArgumentError("the exposures lag the returns by lag = $(pe.lag), so a fit needs more than $(pe.lag) observations after the Descriptor warm-up, and $(length(rw)) are left. Give more observations, shorten the warm-up of the Descriptors, or lower lag."))
    r = (pe.lag + 1):length(rw)
    Zl = fb.Ms[r .- pe.lag, :, :]
    Xr = Xw[r, :]
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
    if needs_second_pass(pe.wa)
        W = cs_weights_refine(pe.wa, W, csr.eps, pe.ve, msk)
        csr = cross_sectional_regression(pe.cre, Zl, Xr, W)
    end
    vs = variance_series(pe.ve, csr.eps; dims = 1)
    S = cross_sectional_standardised_residuals(csr.eps, vs, amr)
    esigma = cross_sectional_idiosyncratic_covariance(pe.th, pe.ce, pe.mp.pdm, S,
                                                      vs[end, :])
    f_pr = prior(pe.pe, csr.f)
    fnow = cross_sectional_basis_now(fb.fcb, r)
    ex = cross_sectional_expand(fb.fcb, r, pe.lag, csr.f, f_pr.mu, f_pr.sigma)
    L = fb.Ms[r[end], :, :]
    ev = vs[end, :]
    idx = cross_sectional_investable(amr[end, :], L, ev)
    @argcheck(!isempty(idx),
              IsEmptyError("no asset is investable at the latest observation: every asset is either inactive, or carries a non-finite idiosyncratic variance or Factor Exposure. Give more observations, or widen the active mask of the Asset Panel."))
    Xs = cross_sectional_scenarios(f_pr.X, L, S, ev)
    lift = cross_sectional_lift(pe.mp, L, f_pr.mu, f_pr.sigma, esigma, idx, Xs; kwargs...)
    Msr = Msw[r, :, :]
    rr = CrossSectionalFactorModel(; M = Msr[end, :, :],
                                   L = cross_sectional_reduced_loadings(fnow, L),
                                   b = zeros(eltype(lift.mu), size(X, 2)), csr = csr,
                                   Ms = Msr, vs = vs, esigma = esigma, rw = W, bw = bwr,
                                   nf = nf, fam = fam, fcb = fnow, lag = pe.lag)
    fpr = LowOrderPrior(; X = ex.f, mu = ex.mu, sigma = ex.sigma, w = f_pr.w,
                        ens = f_pr.ens, kld = f_pr.kld, ow = f_pr.ow)
    return LowOrderPrior(; X = Xs, o_X = Xr, mu = lift.mu, sigma = lift.sigma,
                         chol = lift.chol, w = f_pr.w, ens = f_pr.ens, kld = f_pr.kld,
                         ow = f_pr.ow, rr = rr, fpr = fpr,
                         pnl = port_opt_view(pnl, rw[r], :, false))
end
"""
    prior(pe::CrossSectionalFactorPrior, X::MatNum, args...; kwargs...) -> Union{}

Refuse a Cross-Sectional Factor Prior that is handed bare matrices.

Every other low order prior estimator is fitted from a returns matrix alone. This one reads per-asset Panel Fields and the two universe masks, and a matrix carries neither, so the shape a wrapper hands its nested estimator cannot fit it. The refusal names the entry point that works.

# Arguments

  - `pe`: Cross-Sectional Factor Prior estimator.
  - `X`: Asset returns.
  - `args...`: Ignored.

# Validation

  - Always raises an `ArgumentError`.

# Returns

  - Nothing is returned.

# Related

  - [`CrossSectionalFactorPrior`](@ref)
  - [`prior`](@ref)
  - [`ReturnsResult`](@ref)
"""
function prior(::CrossSectionalFactorPrior, ::MatNum, args...; kwargs...)
    return throw(ArgumentError("a Cross-Sectional Factor Prior is fitted on an Asset Panel, and a returns matrix carries neither the Panel Fields its Factor Exposures read nor the two universe masks it fits against. Call prior(pe, rd) with the ReturnsResult that carries the panel."))
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
