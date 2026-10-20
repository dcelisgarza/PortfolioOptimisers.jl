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
our defaults against the oracle's, and a forecast starts from the first observation that advances
the state (#1386). The calibration is the one exception: the target `Default` case states
`cv = KFold()`, the out-of-fold calibration on five folds of the oracle's batch fit, because the
default of the library is prequential since #1575. The oracle's integer `cv = 3` gives the
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
    # The oracle neutralises with no intercept, so every case passes its rule (#1521).
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
    # The oracle calibrates on the out-of-fold predictions of a k-fold split in its batch fit,
    # so every calibrated case states `KFold()`. The default of the library is prequential
    # since #1575, and "NoCal" keeps it, so the oracle checks the model that it fits.
    target = ["NoCal" => TargetReturnForecast(; scores = ds(), calibrate = false),
              "Default" =>
                  TargetReturnForecast(; scores = ds(), half_life = 10.0, cv = KFold()),
              "Hz3" => TargetReturnForecast(; scores = ds(), horizon = 3, lag = 2,
                                            cv = KFold(; n = 3), scale = 2.0),
              "Sharpe" => TargetReturnForecast(; scores = ds(; neutralise = "style"),
                                               unit = IdiosyncraticSharpeUnit(), cv = KFold()),
              "TgtScore" => TargetReturnForecast(; scores = ds(), target_outlier = nothing,
                                                 target_scoring = CrossSectionalStandardiser(),
                                                 cv = KFold())]
    # ADR 0186: no field takes an integer as the short form of `KFold(; n)`.
    @test_throws TypeError TargetReturnForecast(; scores = ds(), cv = 3)
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

#=
The Cross-Sectional Factor Prior on the online step (#1468, map #1375, ADR 0193).

The prior refits over a sample buffer that records the returns, both masks and the Panel Fields,
so its call with no data equals the batch fit over the rows of the buffer. Its Choice Rule says how the
automatic dropped member of a Factor Family behaves on that seam: `BatchChoice()` chooses again at
each call with no data, and `PinnedChoice()` keeps the choice of the first fit.

Every `Parity_CrossSectionalFactorPrior_Online*` file is an output of the oracle's online update
over one stream of batches, rows 1-90, 91-170 and 171-250 of the exchange that
`parity_write(dir, grid_fixture(parity_large_panel()); filled = true)` writes. The panel holds
Panel Fields, and its estimation mask differs from its active mask on asset 6. Each file stacks
the three steps vertically: one row of `mu`, one 40 x 40 block of `sigma`, or the factor returns
of every row fitted so far. `Fold*` is the oracle's online update after each batch, and `Prefix*`
is its batch fit over the rows seen so far. The two cases are the grid configurations of
`parity_grid.jl`:

  - `Style`: `Base` with `families = ["style" => nothing]`. The oracle's update pins `style1` over
    the first batch, and its batch fit over 170 and 250 rows drops `style2`.
  - `Industry`: `FamOne`. Every fit drops `industry=Software`, so `Fold` and `Prefix` agree, and
    only `Fold` is stored.
  - `CurrencyStyle` (#1478): `Currency` with `families = ["style" => nothing]`. The oracle's update
    reads the Currency Excess Returns of each batch, and it pins `style1` as in `Style`. Only `Fold`
    is stored.

The refit records the Exogenous Series that an observed factor or a macro sensitivity reads
(#1478, ADR 0193), so those priors take the online step too.
=#
include(joinpath(@__DIR__, "parity_grid.jl"))

@testset "The Cross-Sectional Factor Prior on the online step (#1468)" begin
    po = PortfolioOptimisers
    fx = parity_large_panel()
    rd = grid_fixture(fx)
    N = size(rd.X, 2)
    edges = (0, 90, 170, 250)
    rows(r, i) = po.port_opt_view(r, i, :)
    online(est; kwargs...) = po.update_online_estimator(Online(est; kwargs...))
    # The prior after each step of the stream, and its call with no data.
    function stream(pe)
        e = online(pe)
        out = []
        for k in 1:3
            e = partial_fit!(e, rows(rd, (edges[k] + 1):edges[k + 1]))
            push!(out, (; pe = e, pr = prior(e)))
        end
        return out
    end
    batch(pe, k) = prior(pe, rows(rd, 1:edges[k + 1]))
    dropped(pr) = pr.rr.nf[po.dropped_factor_indices(pr.rr.fcb)]
    function message(f)
        return try
            f()
            ""
        catch err
            err.msg
        end
    end
    style = (; grid_config("Base", rd)..., families = ["style" => nothing])
    industry = grid_config("FamOne", rd)
    sb = stream(CrossSectionalFactorPrior(; lambda = 1, style...))
    sp = stream(CrossSectionalFactorPrior(; lambda = 1, style..., choice = PinnedChoice()))
    ib = stream(CrossSectionalFactorPrior(; lambda = 1, industry...))
    ip = stream(CrossSectionalFactorPrior(; lambda = 1, industry...,
                                          choice = PinnedChoice()))
    sbatch = [batch(CrossSectionalFactorPrior(; lambda = 1, style...), k) for k in 1:3]

    @testset "The refit equals the batch fit over the same rows" begin
        for k in 1:3
            pr, b = sb[k].pr, sbatch[k]
            @test isequal(pr.mu, b.mu) && isequal(pr.sigma, b.sigma)
            @test isequal(pr.fpr.X, b.fpr.X) && isequal(pr.rr.M, b.rr.M)
        end
        # A cap windows the whole fit: the call with no data is the batch fit over the last rows.
        e = online(CrossSectionalFactorPrior(; lambda = 1, style...); max_history = 120)
        for k in 1:3
            e = partial_fit!(e, rows(rd, (edges[k] + 1):edges[k + 1]))
        end
        b = prior(CrossSectionalFactorPrior(; lambda = 1, style...), rows(rd, 131:250))
        @test isequal(prior(e).mu, b.mu) && isequal(prior(e).sigma, b.sigma)
        # A view of a stepped prior slices its buffer to the selected assets.
        v = po.port_opt_view(sb[1].pe, 1:20)
        @test size(po.sample_buffer(v.cache)) == (90, 20)
        # The buffer is running state, and it does not print.
        s = sprint(show, sb[1].pe)
        @test !occursin("cache", s) && occursin("choice", s)
    end

    @testset "On the online step of an optimiser" begin
        # The step forwards each block to the prior, and the call with no data of the optimiser reads
        # the prior out of its buffer. A time-varying panel with Panel Fields and a narrower
        # estimation universe reaches the prior, and the weights equal the batch weights.
        for choice in (BatchChoice(), PinnedChoice())
            pe = CrossSectionalFactorPrior(; lambda = 1, style..., choice = choice)
            o = po.update_online_estimator(InverseVolatility(; pe = Online(pe)))
            for k in 1:3
                o = partial_fit!(o, rows(rd, (edges[k] + 1):edges[k + 1]))
                bo, brd = po.batch_from_state(o)
                ref = choice isa BatchChoice ? sb[k].pr : sp[k].pr
                @test isequal(bo.pe.mu, ref.mu) && isequal(bo.pe.sigma, ref.sigma)
                @test isequal(brd.X, rd.X[1:edges[k + 1], :])
            end
            if choice isa BatchChoice
                @test isequal(optimise(o).w, optimise(InverseVolatility(; pe = pe), rd).w)
            end
        end
    end

    @testset "The Choice Rule" begin
        # In a batch fit the two rules agree.
        for k in 1:3
            pp = batch(CrossSectionalFactorPrior(; lambda = 1, style...,
                                                 choice = PinnedChoice()), k)
            @test isequal(pp.sigma, sbatch[k].sigma) && isequal(pp.mu, sbatch[k].mu)
        end
        # A batch choice chooses again at each call with no data, and leaves `families` alone.
        @test [dropped(x.pr) for x in sb] == [["style1"], ["style2"], ["style2"]]
        @test all(x -> x.pe.families == ["style" => nothing], sb)
        # A pinned choice writes the choice of the first fit into `families`, and every
        # later call with no data drops it.
        @test all(x -> x.pe.families == ["style" => "style1"], sp)
        @test [dropped(x.pr) for x in sp] == [["style1"], ["style1"], ["style1"]]
        @test isequal(sp[1].pr.sigma, sb[1].pr.sigma)
        # Pinned, the call with no data is the batch fit with the member stated.
        stated = CrossSectionalFactorPrior(; lambda = 1, style...,
                                           families = ["style" => "style1"])
        @test isequal(sp[3].pr.sigma, batch(stated, 3).sigma)
        # The choice moves on this stream, so the two rules differ after the first step.
        @test maximum(abs, filter(isfinite, sp[3].pr.sigma - sb[3].pr.sigma)) > 1e-6
        # A first block that the fit refuses pins nothing. The next block pins the choice of
        # the fit over every row the buffer holds, the first fit.
        e = online(CrossSectionalFactorPrior(; lambda = 1, style...,
                                             choice = PinnedChoice()))
        e = partial_fit!(e, rows(rd, 1:2))
        @test e.families == ["style" => nothing]
        @test_throws ArgumentError prior(e)
        e = partial_fit!(e, rows(rd, 3:90))
        @test e.families == ["style" => "style1"]
        @test isequal(prior(e).sigma, sp[1].pr.sigma)
        # A stated member is not pinned again, and the step leaves `families` as it is.
        e = online(CrossSectionalFactorPrior(; lambda = 1, style...,
                                             choice = PinnedChoice(),
                                             families = ["style" => "style2"]))
        e = partial_fit!(e, rows(rd, 1:90))
        @test e.families == ["style" => "style2"]
        # So does a prior with no constrained family.
        e = online(CrossSectionalFactorPrior(; lambda = 1, grid_config("Base", rd)...,
                                             choice = PinnedChoice()))
        @test isnothing(partial_fit!(e, rows(rd, 1:90)).families)
        # The industry choice does not move, so the two rules agree at every step.
        @test all(x -> x.pe.families == ["industry" => "industry=Software"], ip)
        @test all(k -> isequal(ip[k].pr.sigma, ib[k].pr.sigma), 1:3)
    end

    @testset "Refusals" begin
        pe = CrossSectionalFactorPrior(; lambda = 1, style...)
        # An unwrapped prior takes the other route, the carry fold of #1471, which
        # test_12zb_parity_cs_prior_carry_fold.jl tests.
        @test isa(partial_fit!(pe, rows(rd, 1:90)).cache, po.CrossSectionalCarryState)
        m = message(() -> partial_fit!(online(pe), rd.X))
        @test occursin("the matrix form of `partial_fit!` carries none", m)
    end

    @testset "The refit records the Exogenous Series (#1478)" begin
        # An observed factor reads its return from the Exogenous Series, and an
        # `EWMacroSensitivity` that names a series reads its reference return there. The
        # buffer records every column of the series, so each of the three refits with no data
        # equals the batch fit over the same rows, bit for bit.
        base = grid_config("Base", rd)
        msens = "msens" => CompositeExposure(;
                                             descriptors = [EWMacroSensitivity(; series = "MACRO",
                                                                               half_life = 10)],
                                             outlier = nothing, scoring = nothing, family = "msens")
        cfgs = (; Currency = grid_config("Currency", rd), Macro = grid_config("Macro", rd),
                Sensitivity = (; base..., factors = [base.factors; msens]))
        for (name, cfg) in pairs(cfgs)
            pe = CrossSectionalFactorPrior(; lambda = 1, cfg...)
            @test po.reads_exogenous_series(pe)
            out = stream(pe)
            for k in 1:3
                b = batch(pe, k)
                @test isequal(out[k].pr.mu, b.mu) && isequal(out[k].pr.sigma, b.sigma)
                @test isequal(out[k].pr.fpr.X, b.fpr.X)
            end
            st = out[3].pe.cache
            @test st.ne == rd.ne && isequal(po.exogenous_buffer_kwargs(st).E, rd.E)
            # A cap windows the series with the rows.
            e = online(pe; max_history = 120)
            for k in 1:3
                e = partial_fit!(e, rows(rd, (edges[k] + 1):edges[k + 1]))
            end
            @test isequal(prior(e).sigma, prior(pe, rows(rd, 131:250)).sigma)
        end
        # The predicate answers per type, through the factor list, the Return Forecast
        # Estimator and the wrapper. A macro sensitivity that names no series reads none.
        @test !po.reads_exogenous_series(CrossSectionalFactorPrior(; lambda = 1, style...))
        @test !po.reads_exogenous_series(EWMacroSensitivity())
        @test po.reads_exogenous_series(Online(CrossSectionalFactorPrior(; lambda = 1,
                                                                         base...,
                                                                         factors = [base.factors;
                                                                                    msens])))
        ds = DescriptorScores(; descriptors = [EWMacroSensitivity(; series = "MACRO")])
        @test po.reads_exogenous_series(CrossSectionalFactorPrior(; lambda = 1, style...,
                                                                  rfe = FixedWeightedReturnForecast(;
                                                                                                    scores = ds,
                                                                                                    scale = 0.02)))
        # The same prior inside an optimiser: the prior's buffer owns the series, the Fold
        # Context keeps no copy and reads it back, and the weights equal the batch weights.
        ccy = CrossSectionalFactorPrior(; lambda = 1, grid_config("Currency", rd)...)
        o = po.update_online_estimator(InverseVolatility(; pe = Online(ccy)))
        for k in 1:3
            o = partial_fit!(o, rows(rd, (edges[k] + 1):edges[k + 1]))
        end
        @test isnothing(o.cache.E) && isnothing(o.cache.ne)
        bo, brd = po.batch_from_state(o)
        @test brd.ne == rd.ne && isequal(brd.E, rd.E)
        @test isequal(bo.pe.sigma, prior(ccy, rd).sigma)
        @test isequal(optimise(o).w, optimise(InverseVolatility(; pe = ccy), rd).w)
        # A prior that reads no series leaves it to the Fold Context.
        o = po.update_online_estimator(InverseVolatility(;
                                                         pe = Online(CrossSectionalFactorPrior(;
                                                                                               lambda = 1,
                                                                                               style...))))
        o = partial_fit!(o, rows(rd, 1:90))
        @test isnothing(o.pe.cache.E) && o.cache.ne == rd.ne
        @test isequal(po.batch_from_state(o)[2].E, rd.E[1:90, :])
        # Refusals. A step with no series, and a step with other names.
        nor = ReturnsResult(; nx = rd.nx, X = rd.X, pnl = rd.pnl)
        m = message(() -> partial_fit!(online(ccy), rows(nor, 1:90)))
        @test occursin("this step carries no `rd.E`", m)
        e = partial_fit!(online(ccy), rows(rd, 1:90))
        m = message(() -> partial_fit!(e, rows(nor, 91:170)))
        @test occursin("this step carries no `rd.E`", m)
        ren = ReturnsResult(; nx = rd.nx, X = rd.X, ne = [rd.ne[1:(end - 1)]; "OTHER"],
                            E = rd.E, pnl = rd.pnl)
        m = message(() -> partial_fit!(e, rows(ren, 91:170)))
        @test occursin("a later block must carry the same names", m)
        # An infinite value is refused on a row that the fit reads. A `NaN` is a gap of the
        # observed series (#1530), so the refit and the batch fit keep it alike. Row 1 has no
        # lagged exposure, so the regression never reads it.
        for (row, x, refused) in ((200, Inf, true), (200, NaN, false), (1, NaN, false))
            E = copy(rd.E)
            E[row, 1] = x
            rn = ReturnsResult(; nx = rd.nx, X = rd.X, ne = rd.ne, E = E, pnl = rd.pnl)
            e = online(ccy)
            for k in 1:3
                e = partial_fit!(e, rows(rn, (edges[k] + 1):edges[k + 1]))
            end
            if refused
                @test_throws IsNonFiniteError prior(e)
                @test_throws IsNonFiniteError prior(ccy, rn)
            else
                @test isequal(prior(e).sigma, prior(ccy, rn).sigma)
            end
        end
        # The carry fold records the series too (#1479), and test_12zb tests it.
        @test isa(partial_fit!(CrossSectionalFactorPrior(; lambda = 1, cfgs.Currency...),
                               rows(rd, 1:90)).cache, po.CrossSectionalCarryState)
    end

    @testset "Parity with the oracle's online update" begin
        load(case, o) = parity_load("CrossSectionalFactorPrior", "Online$(case)", o)
        # The prior of each step against the stored block of that step. `mu` compares cell by
        # cell. `sigma` compares against its largest entry: an off-diagonal entry of two assets
        # near zero correlation is a cancellation in `L F L'` (#1376), measured maxrel 2.7e-12
        # cell by cell. It was 1.3e-10 when the lift repaired the systematic block `L F L'`, which the oracle does not repair (#1576). The factor returns compare cell by cell with an absolute floor: the
        # industry level "Utilities" is empty after its only asset delists, so its factor return
        # is a round-off zero on both sides, measured maxrel 1.9 and maxabs 1.5e-16. The floor
        # is at most 6e-14 of the largest factor return, and every other cell measured maxrel
        # 5.7e-13.
        function check(out, case, src)
            mu = load(case, "$(src)Mu")
            S = load(case, "$(src)Sigma")
            F = load(case, "$(src)FactorReturns")
            r0 = 0
            for k in 1:3
                pr = out[k].pr
                n = edges[k + 1] - 1
                @test parity_compare(pr.mu, mu[k, :]; name = "$(case) $(src) mu $(k)").ok
                @test parity_compare(pr.sigma, S[((k - 1) * N + 1):(k * N), :];
                                     scale = :array, name = "$(case) $(src) sigma $(k)").ok
                @test parity_compare(pr.fpr.X, F[(r0 + 1):(r0 + n), :]; atol = 1e-15,
                                     name = "$(case) $(src) factor returns $(k)").ok
                r0 += n
            end
        end
        # The pinned choice reproduces the oracle's online update, which pins the dropped
        # member at its first call. Measured over the three steps: mu maxrel 1.7e-14, sigma
        # maxscaled 1.1e-15, factor returns maxrel 4.6e-14 on `Style`; mu maxrel 5.3e-13,
        # sigma maxscaled 1.2e-15, factor returns maxabs 1.5e-16 on `Industry`. The sigma
        # values were 4.2e-13 and 6.5e-13 when the lift repaired the systematic block `L F L'`, which the oracle does not repair (#1576).
        check(sp, "Style", "Fold")
        check(ip, "Industry", "Fold")
        # The pinned choice under currency factors (#1478): the buffer records the Exogenous
        # Series, and the update pins `style1` over the first batch while the batch fit over
        # 170 and 250 rows drops `style2`. Measured over the three steps: mu maxrel 2.3e-13,
        # sigma maxscaled 5.6e-16 (4.2e-13 before #1576), factor returns maxrel 9.3e-14.
        cpe = CrossSectionalFactorPrior(; lambda = 1, grid_config("Currency", rd)...,
                                        families = ["style" => nothing],
                                        choice = PinnedChoice())
        cp = stream(cpe)
        @test all(x -> x.pe.families == ["style" => "style1"], cp)
        check(cp, "CurrencyStyle", "Fold")
        # Deliberate difference: the batch choice is the oracle's batch fit over the rows
        # seen so far, not its online update. The automatic dropped member is a function of
        # the whole sample: the member with the largest sum of absolute benchmark-weighted
        # exposures. Under the batch choice the call with no data is the same function of the same
        # rows as the batch fit, whatever the split of the stream. The factor returns of the
        # full basis are the same under either member, but the factor prior fits the reduced
        # factor returns, whose time-varying ratios depend on the member, so its moments move
        # with the member: here sigma by 7.4e-5 at the second step. `PinnedChoice()` is the
        # oracle's rule, one keyword away. Measured against the oracle's batch fit: mu maxrel
        # 2.7e-14, sigma maxscaled 1.4e-15 (4.1e-13 before #1576), factor returns maxrel
        # 2.8e-13.
        check(sb, "Style", "Prefix")
        @test !parity_compare(sb[2].pr.sigma, load("Style", "FoldSigma")[(N + 1):(2N), :];
                              scale = :array, name = "Style batch vs fold sigma 2",
                              quiet = true).ok
    end
end
