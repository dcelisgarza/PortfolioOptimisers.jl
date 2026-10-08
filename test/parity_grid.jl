#=
The fixture and the configurations of the prior's configuration grid (#1385, map #1375),
shared by the parity tests that fit them. It has no `test_` prefix, so the runner does not
discover it. Include it after `parity_harness.jl`.
=#
# The fixture of the harness, with three more numeric Panel Fields: `macro_beta`, the loading of
# each asset on the macro series; `local_returns`, the local return of each asset; and `custom_mu`,
# a custom forecast. The base-currency form sets the USD Currency Excess Return to zero, as it is
# for an investor whose base currency is USD, and states the base returns again from the local ones.
function grid_fixture(fx; base_usd::Bool = false)
    rd = fx.rd
    T, N = size(rd.X)
    rng = StableRNG(1385)
    amsk = fx.amsk
    nanout(A) = ifelse.(amsk, A, NaN)
    beta = nanout(repeat(randn(rng, 1, N), T))
    cmu = nanout(0.001 .* randn(rng, T, N))
    R = copy(fx.R)
    if base_usd
        R[:, 3] .= 0.0
    end
    X = nanout(fx.loc .+ R[:, fx.code])
    h = fx.at.holiday
    X[h...] = NaN
    loc = nanout(fx.loc)
    loc[h...] = NaN
    extra = asset_panel([NumericPanelInput(; name = "macro_beta", vals = beta,
                                           alg = ForwardPanelFill(; val = 0.0)),
                         NumericPanelInput(; name = "local_returns", vals = loc,
                                           alg = ForwardPanelFill(; val = 0.0)),
                         NumericPanelInput(; name = "custom_mu", vals = cmu,
                                           alg = ForwardPanelFill(; val = 0.0))];
                        amsk = amsk, emsk = fx.emsk)
    pnl = AssetPanel(; pf = [rd.pnl.pf; extra.pf], amsk = rd.pnl.amsk, emsk = rd.pnl.emsk)
    E = copy(rd.E)
    E[:, 1:3] = R
    return ReturnsResult(; nx = rd.nx, X = base_usd ? X : rd.X, ne = rd.ne, E = E,
                         pnl = pnl)
end

function grid_pass(f; family = "style")
    return CompositeExposure(; descriptors = [Passthrough(; field = f)], outlier = nothing,
                             scoring = nothing, family = family)
end
function grid_scored(f)
    return CompositeExposure(; descriptors = [Passthrough(; field = f)], family = "style")
end

# The factor prior and the idiosyncratic variance at the oracle's defaults: the regime multiplier
# clips to (0.7, 1.6), and the regime statistic reads an asset only when its variance exceeds
# 1e-12. That threshold is `min_val`; `min_val = 0.0` differs only for a variance in (0, 1e-12],
# and the one-member industry of the fixture puts a round-off variance there (`FamOne`).
const GRID_PE = EmpiricalPrior(; me = ExpWeightedExpectedReturns(),
                               ce = RegimeAdjustedExpWeightedCovariance(;
                                                                        centring = PreCentred(),
                                                                        debias = RawStatistic(),
                                                                        regime_lohi_mult = (0.7,
                                                                                            1.6)))
const GRID_VE = RegimeAdjustedExpWeightedVariance(; centring = PreCentred(),
                                                  debias = RawStatistic(),
                                                  regime_lohi_mult = (0.7, 1.6),
                                                  min_val = 1e-12)

function grid_config(name::AbstractString, rd::ReturnsResult)
    base = ["market" => ConstantExposure(), "style1" => grid_pass("style1"),
            "style2" => grid_pass("style2")]
    ind = ["market" => ConstantExposure(),
           "industry" => OneHotExposure(; field = "industry", family = "industry"),
           "style1" => grid_pass("style1"), "style2" => grid_pass("style2")]
    ds = DescriptorScores(;
                          descriptors = [Passthrough(; field = "net_income_ttm"),
                                         Passthrough(; field = "sales_ttm")])
    fc = (; lambda = 0.4, c = 0.6)
    kw = (; factors = base, minra = 5, pe = GRID_PE, ve = GRID_VE)
    cfg = if name == "Base"
        kw
    elseif name == "Lag2"
        (; kw..., lag = 2)
    elseif name == "Lag3"
        (; kw..., lag = 3)
    elseif name == "ScoredBp1"
        (; kw...,
         factors = ["market" => ConstantExposure(), "style1" => grid_scored("style1"),
                    "style2" => grid_scored("style2")])
    elseif name == "ScoredBp05"
        (; grid_config("ScoredBp1", rd)..., bp = 0.5)
    elseif name == "ScoredBp0"
        (; grid_config("ScoredBp1", rd)..., bp = 0.0)
    elseif name == "MinraNone"
        (; kw..., minra = nothing)
    elseif name == "Minra10"
        (; kw..., minra = 10)
    elseif name == "Minra11"
        (; kw..., minra = 11)
    elseif name == "Target"
        (; kw..., cre = CrossSectionalTargetRegression(; tgt = LinearModel()))
    elseif name == "Blend"
        (; kw..., wa = BlendedInverseVarianceWeights(; lambda = 0.5, ratio = 3.0))
    elseif name == "Neutralised"
        (; kw...,
         factors = vcat(["market" => ConstantExposure(),
                         "size" => CompositeExposure(; descriptors = [LogMarketCap()],
                                                     outlier = nothing, family = "style")],
                        base[2:3]),
         neutralise = ["style1" => ["size"], "style2" => ["size", "style1"]])
    elseif name == "FamOne"
        (; kw..., factors = ind, families = ["industry" => nothing])
    elseif name == "FamStated"
        (; kw..., factors = ind, families = ["industry" => "industry=Banks"])
    elseif name == "FamTwo"
        (; kw...,
         factors = [ind;
                    "region" => OneHotExposure(; field = "currency", family = "region")],
         families = ["industry" => nothing, "region" => nothing])
    elseif name == "NeutFam"
        (; kw..., factors = ind, families = ["industry" => nothing],
         neutralise = ["style" => ["industry"]])
    elseif name == "FcFixed"
        (; kw..., fc..., rfe = FixedWeightedReturnForecast(; scores = ds, scale = 0.02))
    elseif name == "FcFixedSharpe"
        (; kw..., fc...,
         rfe = FixedWeightedReturnForecast(; scores = ds, scale = 0.2,
                                           unit = IdiosyncraticSharpeUnit()))
    elseif name == "FcEW"
        # A fitted member: the oracle reads it as it stands, so the case states the oracle's
        # Orthogonal Forecast Fit (#1486). The library's default neutralises its scores.
        (; kw..., fc..., ofit = UnadjustedForecast(),
         rfe = ExpWeightedReturnForecast(; scores = ds, half_life = 10.0))
    elseif name == "FcTarget"
        # The oracle calibrates the target member on the out-of-fold predictions of a 5-fold
        # split in its batch fit (#1418), so the case states `KFold()`: the default of the
        # library is prequential since #1575. The oracle fits no intercept here.
        (; kw..., fc..., ofit = UnadjustedForecast(),
         rfe = TargetReturnForecast(; scores = ds, half_life = 10.0, cv = KFold()))
    elseif name == "FcTargetIntercept"
        # The oracle's default target member, which fits an intercept (#1419), under the
        # k-fold calibration of its batch fit (#1575).
        (; kw..., fc..., ofit = UnadjustedForecast(),
         rfe = TargetReturnForecast(; scores = ds, half_life = 10.0, intercept = true,
                                    cv = KFold()))
    elseif name == "FcTargetRaw"
        # The same, uncalibrated, so the fitted model alone is compared.
        (; kw..., fc..., ofit = UnadjustedForecast(),
         rfe = TargetReturnForecast(; scores = ds, half_life = 10.0, intercept = true,
                                    calibrate = false))
    elseif name == "FcCustom"
        (; kw..., fc...,
         rfe = CustomValueReturnForecast(;
                                         mu = PortfolioOptimisers.panel_field_values(rd,
                                                                                     "custom_mu")[end,
                                                                                                  :]))
    elseif name == "FcFamily"
        (; kw..., fc..., factors = ind, families = ["industry" => nothing],
         rfe = FixedWeightedReturnForecast(; scores = ds, scale = 0.02))
    elseif name == "Currency"
        (; kw..., factors = [base; "ccy" => CurrencyExposure()])
    elseif name == "CurrencyLx"
        # The local returns stated as a Panel Field, rather than derived from the base returns.
        (; kw..., factors = [base; "ccy" => CurrencyExposure()], lx = "local_returns")
    elseif name == "CurrencyBase"
        # The oracle has no base currency, and it refuses the fit when the excess return of
        # the base currency, which is zero, is a factor. A factor whose return is zero adds zero
        # to `mu` and to `sigma`, so the model without it is the same model: `base` states it,
        # and the stored case is the oracle's own fit with the base column dropped. Better.
        (; kw..., factors = [base; "ccy" => CurrencyExposure(; base = "USD")])
    elseif name == "Macro"
        # The oracle's observed-factor path takes a one-hot member alone. The stored case runs
        # that path on the continuous loading `macro_beta` and the series `MACRO`, with the
        # returns net of the macro factor, which is the model `ObservedExposure` states.
        (; kw...,
         factors = [base;
                    "macro" =>
                        ObservedExposure(; xe = grid_pass("macro_beta"; family = "macro"),
                                         series = "MACRO", family = "macro")])
    else
        throw(ArgumentError(name))
    end
    return cfg
end
function grid_prior(name::AbstractString, rd::ReturnsResult)
    return CrossSectionalFactorPrior(; lambda = 1, grid_config(name, rd)...)
end
