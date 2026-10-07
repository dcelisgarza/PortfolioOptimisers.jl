#=
Parity of the Return Forecast members and of the Descriptor Scores on the block of a fitted prior
(#1386, map #1375).

`test_08x_return_forecasts.jl` compares the members on a hand-built block. Here the block is the
one a `CrossSectionalFactorPrior` fits on the panels of `parity_harness.jl`: a market factor and
the two passthrough styles, the idiosyncratic returns and variances of the fit, and its exposure
history. The oracle reads the same panel with that block padded back onto the whole observation
axis, as its own prior pads it before it calls its forecast. The three Descriptors are two
passthrough fields and a momentum with a warm-up.

Every stored case under `test/assets/Parity_<Unit>_<Case>_<Output>.csv.gz` is the oracle output of
one row of the case tables below:

  - `DescriptorScores_Small<Case>_Scores<k>`: the scores of Descriptor `k` on the small panel.
  - `FixedWeightedReturnForecast_Small<Case>_Hist`, `ExpWeightedReturnForecast_Small<Case>_Hist`:
    the whole history on the small panel. `ExpWeightedReturnForecast_<Panel><Case>_Coef`: the
    latest coefficients.
  - `<Member>_Large<Case>_Mu`: the latest forecast on the large panel.
  - `TargetReturnForecast_<Panel><Case>_Mu`, `_Calib`: the latest forecast and the calibration
    coefficient.

The `Default` cases state no keyword but the Descriptors, the scale and the half-life, so they pin
our defaults against the oracle's: an out-of-fold calibration on five folds, and a forecast from
the first observation that advances the state (#1386). The oracle's integer `cv = 3` gives the
stored `Hz3` output bit for bit, so `KFold(; n = 3)` is the route of the integer short form, as
the `cv` docstring of `TargetReturnForecast` states (#1408).

A score, a residual of a Neutralisation and a forecast are sums whose small cells come from a
cancellation, so every matrix and every forecast compares against its largest entry
(`scale = :array`). Measured 1.0e-14 at most there on the small panel and 3.4e-15 on the large
one, and 5.8e-11 cell by cell on a cell near zero. A coefficient and a calibration coefficient
compare cell by cell, measured 4.8e-14 and 9.5e-16 at most. Every output has the oracle's `NaN`
pattern.

The block of an industry factor holds a one-member level, whose idiosyncratic variance is
round-off (1e-34) and whose inverse-variance weight then rules every weighted fit, on both sides.
That is the limit of a zero variance, and it measures nothing, so the block states no industry
factor, and the industry groups the transforms instead.
=#
include(joinpath(@__DIR__, "parity_harness.jl"))

@testset "The Return Forecast members on the block of a fit, at parity (#1386)" begin
    PO = PortfolioOptimisers
    mpass(f) = CompositeExposure(; descriptors = [Passthrough(; field = f)],
                                 outlier = nothing, scoring = nothing, family = "style")
    factors = ["market" => ConstantExposure(), "style1" => mpass("style1"),
               "style2" => mpass("style2")]
    pe = EmpiricalPrior(; me = ExpWeightedExpectedReturns(),
                        ce = RegimeAdjustedExpWeightedCovariance(; centring = PreCentred(),
                                                                 debias = RawStatistic(),
                                                                 regime_lohi_mult = (0.7,
                                                                                     1.6)))
    ve = RegimeAdjustedExpWeightedVariance(; centring = PreCentred(),
                                           debias = RawStatistic(),
                                           regime_lohi_mult = (0.7, 1.6), min_val = 0.0)
    fit(rd; kw...) = prior(CrossSectionalFactorPrior(; lambda = 1, factors = factors,
                                                     pe = pe, ve = ve, kw...), rd)
    desc = [Passthrough(; field = "net_income_ttm"), Passthrough(; field = "sales_ttm"),
            EWMomentum(; half_life = 5, skip = 3)]
    std2 = CrossSectionalStandardiser(; min_group_size = 2)
    # The reference neutralises with no intercept, so every case passes its rule (#1521).
    ds(; kw...) = DescriptorScores(; descriptors = desc,
                                   cre = CrossSectionalLinearRegression(;
                                                                        intercept = false),
                                   kw...)
    load(u, c, o) = parity_load(u, c, o)
    loadv(u, c, o) = vec(load(u, c, o))
    fxs = parity_small_panel()
    fxl = parity_large_panel()
    panels = (("Small", fxs.rd, fit(fxs.rd; minra = 5)), ("Large", fxl.rd, fit(fxl.rd)))

    scores = ["Default" => ds(), "Group" => ds(; scoring = std2, group = "industry"),
              "Tanh" => ds(; outlier = nothing, scoring = CrossSectionalTanhShrinker()),
              "Raw" => ds(; outlier = nothing, scoring = nothing),
              "NeutFamily" => ds(; neutralise = "style"),
              "NeutFactor" => ds(; neutralise = ["market"]),
              "NeutGroup" => ds(; neutralise = ["market", "style2"], scoring = std2,
                                group = "industry")]
    fixed = ["Equal" => FixedWeightedReturnForecast(; scores = ds(), scale = 0.02),
             "Signed" => FixedWeightedReturnForecast(; scores = ds(), scale = 0.03,
                                                     weights = [0.5, -0.3, 0.2],
                                                     min_coverage = 0.6),
             "Sharpe" => FixedWeightedReturnForecast(; scores = ds(; neutralise = "style"),
                                                     scale = 0.2, unit = IdiosyncraticSharpeUnit()),
             "One" => FixedWeightedReturnForecast(;
                                                  scores = DescriptorScores(;
                                                                            descriptors = desc[3:3],
                                                                            group = "industry",
                                                                            scoring = std2),
                                                  scale = 0.01, weights = [-2.0]),
             "Raw" => FixedWeightedReturnForecast(;
                                                  scores = DescriptorScores(;
                                                                            descriptors = desc[1:2],
                                                                            outlier = nothing,
                                                                            scoring = nothing),
                                                  scale = 1e-4, weights = [1.0, 2.0],
                                                  min_coverage = 1.0)]
    ew = ["Default" => ExpWeightedReturnForecast(; scores = ds()),
          "Hl5" => ExpWeightedReturnForecast(; scores = ds(), half_life = 5.0, horizon = 3,
                                             lag = 2),
          "Decay" => ExpWeightedReturnForecast(; scores = ds(), decay = 0.85),
          "Ridge0" => ExpWeightedReturnForecast(; scores = ds(), ridge = 0.0),
          "Ridge01" => ExpWeightedReturnForecast(; scores = ds(), ridge = 0.1),
          "NoNorm" =>
              ExpWeightedReturnForecast(; scores = ds(), normalise = false, scale = 2.5),
          "Sharpe" => ExpWeightedReturnForecast(;
                                                scores = ds(; neutralise = "style", scoring = std2,
                                                            group = "industry"), half_life = 10.0,
                                                unit = IdiosyncraticSharpeUnit())]
    target = ["NoCal" => TargetReturnForecast(; scores = ds(), calibrate = false),
              "Default" => TargetReturnForecast(; scores = ds(), half_life = 10.0),
              "Hz3" => TargetReturnForecast(; scores = ds(), horizon = 3, lag = 2,
                                            cv = KFold(; n = 3), scale = 2.0),
              "Sharpe" => TargetReturnForecast(; scores = ds(; neutralise = "style"),
                                               unit = IdiosyncraticSharpeUnit()),
              "TgtScore" => TargetReturnForecast(; scores = ds(), target_outlier = nothing,
                                                 target_scoring = CrossSectionalStandardiser())]
    # A matrix or a forecast compares against its largest entry, a coefficient cell by cell.
    same(a, b, n) = parity_compare(a, b; scale = :array, name = n).ok
    cell(a, b, n) = parity_compare(a, b; name = n).ok

    @testset "The Descriptor Scores, $(c)" for (c, d) in scores
        (_, rd, pr) = panels[1]
        S = descriptor_scores(d, rd, pr.rr).S
        for k in axes(S, 3)
            @test same(S[:, :, k], load("DescriptorScores", "Small$(c)", "Scores$(k)"),
                       "$(c) $(k)")
        end
    end

    @testset "The raw scores are the Panel Fields, bit for bit" begin
        (_, rd, pr) = panels[1]
        S = descriptor_scores(scores[4][2], rd, pr.rr).S
        for k in 1:2
            @test isequal(S[:, :, k], load("DescriptorScores", "SmallRaw", "Scores$(k)"))
        end
    end

    @testset "The fixed weighted member, $(p) $(c)" for (p, rd, pr) in panels,
                                                        (c, e) in fixed

        rf = return_forecast(e, rd, pr.rr)
        if p == "Small"
            rows = PO.return_forecast_rows(rd, pr.rr)
            # The oracle states the whole observation axis, and the member the block's rows.
            H = load("FixedWeightedReturnForecast", "Small$(c)", "Hist")
            @test same(rf.hist, H[rows, :], "$(c) hist")
        end
        @test same(rf.mu, loadv("FixedWeightedReturnForecast", "$(p)$(c)", "Mu"), "$(c) mu")
    end

    @testset "The exponentially weighted member, $(p) $(c)" for (p, rd, pr) in panels,
                                                                (c, e) in ew

        rf = return_forecast(e, rd, pr.rr)
        if p == "Small"
            rows = PO.return_forecast_rows(rd, pr.rr)
            H = load("ExpWeightedReturnForecast", "Small$(c)", "Hist")
            @test same(rf.hist, H[rows, :], "$(c) hist")
        end
        @test same(rf.mu, loadv("ExpWeightedReturnForecast", "$(p)$(c)", "Mu"), "$(c) mu")
        @test cell(rf.coef, loadv("ExpWeightedReturnForecast", "$(p)$(c)", "Coef"),
                   "$(c) coef")
    end

    @testset "The target member, $(p) $(c)" for (p, rd, pr) in panels, (c, e) in target
        rf = return_forecast(e, rd, pr.rr)
        @test same(rf.mu, loadv("TargetReturnForecast", "$(p)$(c)", "Mu"), "$(c) mu")
        @test cell([rf.calib], loadv("TargetReturnForecast", "$(p)$(c)", "Calib"),
                   "$(c) calib")
    end

    @testset "An in-sample calibration is ours only, and it is biased" begin
        # `cv = nothing` calibrates on the predictions of the model fitted to the very forward
        # returns the calibration regresses on them. The oracle has no such mode. On the small
        # panel its slope is 1.43, against 1.22 out of fold: the regression fitted its own
        # training targets, and the slope measures that fit too (#1418).
        (_, rd, pr) = panels[1]
        oof = return_forecast(target[2][2], rd, pr.rr)
        ins = return_forecast(TargetReturnForecast(; scores = ds(), half_life = 10.0,
                                                   cv = nothing), rd, pr.rr)
        @test isfinite(ins.calib) && isfinite(oof.calib)
        @test ins.calib > 1.1 * oof.calib
        # The uncalibrated prediction is the same model's, so the forecasts are proportional.
        f = isfinite.(oof.mu)
        @test isequal(isfinite.(ins.mu), f)
        @test isapprox(ins.mu[f] ./ ins.calib, oof.mu[f] ./ oof.calib; rtol = 1e-12)
    end

    @testset "The default forecasts from the first observation" begin
        # The coefficients after `n` observations are the weighted least squares of those
        # observations: the weight `1 - lambda^n` multiplies both accumulators and the ridge.
        # So the default publishes from the first one, as the oracle does, and a warm-up is a
        # keyword that holds back values the model determines.
        (_, rd, pr) = panels[1]
        r1 = return_forecast(ew[1][2], rd, pr.rr)
        r20 = return_forecast(ExpWeightedReturnForecast(; scores = ds(), min_obs = 20), rd,
                              pr.rr)
        @test isequal(r1.coef, r20.coef)
        @test count(isfinite, r20.hist) < count(isfinite, r1.hist)
        @test isequal(r1.mu, r20.mu)
    end
end
