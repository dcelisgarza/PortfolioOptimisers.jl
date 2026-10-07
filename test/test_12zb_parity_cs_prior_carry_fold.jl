#=
The carry fold of the Cross-Sectional Factor Prior (#1471, map #1375, ADR 0193).

An unwrapped prior folds as a carry: its first step seeds a `CrossSectionalCarryState`, and each
step computes the exposures, the regression and the idiosyncratic variance of the new
observations alone, from the panel rows that its Descriptors read. The factor prior and the
idiosyncratic variance fold, and the Return Forecast and the idiosyncratic correlation refit at the
call with no data. A batch choice that moves, or a factor that comes alive, fits every carried observation
again.

A step folds the member states in place, as every fold does, so each stream reads the prior out
right after its step.

Every `Parity_CrossSectionalFactorPrior_CarrySeed*` file is an output of the oracle's online update
over the stream of #1468, rows 1-90, 91-170 and 171-250 of the exchange that
`parity_write(dir, grid_fixture(parity_large_panel()).rd; filled = true)` writes. Its factor prior
is an exponentially weighted mean and covariance of half-life 20 and `min_observations = 5`, whose
window of 60 rows cuts the factor returns of the first call alone. Each file stacks the three steps
vertically, as the `Online*` files of test_12za do: one row of `mu`, one 40 x 40 block of `sigma`,
or the factor returns of every row fitted so far. The cases are `SeedStyle` (`Base` with
`families = ["style" => nothing]`) and `SeedIndustry` (`FamOne`). The pinned choice with no window
is checked against the `Online*Fold*` files of test_12za.

A prior whose tree reads the Exogenous Series folds on the carry route too (#1479, ADR 0193): the
buffer and the carried rows hold the series, and the state derives each row of the returns net of
the observed factors one time. Its pinned choice under currency factors is checked against the
`OnlineCurrencyStyle_Fold*` files of test_12za.

A slot that reads the Return Forecast history makes the state carry it (#1484): each step fits the
member once for each new observation, and the history and the resolved `lambda` and `c` equal those
of the batch fit over the same rows.
=#
include(joinpath(@__DIR__, "parity_harness.jl"))
include(joinpath(@__DIR__, "parity_grid.jl"))
include(joinpath(@__DIR__, "test06c_setup.jl"))

# Test-local rules of `lambda` and `c` that read the Return Forecast history (#1484). Each records
# the history it got, and its number moves with every row of it.
struct CarryHistoryShrinkage <:
       PortfolioOptimisers.AbstractSpannedShrinkageCalibrationAlgorithm
    seen::Vector{Any}
end
CarryHistoryShrinkage() = CarryHistoryShrinkage(Any[])
PortfolioOptimisers.reads_forecast_history(::CarryHistoryShrinkage) = true
function (r::CarryHistoryShrinkage)(key, pr, w, slv, ctx)
    h = ctx.cs.hist
    push!(r.seen, copy(h))
    return 1 / (1 + sum(abs, filter(isfinite, h)))
end
struct CarryHistoryScale <:
       PortfolioOptimisers.AbstractOrthogonalForecastScaleCalibrationAlgorithm
    seen::Vector{Any}
end
CarryHistoryScale() = CarryHistoryScale(Any[])
PortfolioOptimisers.reads_forecast_history(::CarryHistoryScale) = true
function (r::CarryHistoryScale)(key, pr, w, slv, ctx)
    h = ctx.cs.hist
    push!(r.seen, copy(h))
    return mean(abs, filter(isfinite, h)) * 100
end

@testset "The carry fold of the Cross-Sectional Factor Prior (#1471)" begin
    po = PortfolioOptimisers
    fx = parity_large_panel()
    rd = grid_fixture(fx)
    N = size(rd.X, 2)
    edges = (0, 90, 170, 250)
    rows(r, i) = po.port_opt_view(r, i, :)
    # Equal cell by cell, a `NaN` equal to a `NaN`.
    same(a, b) = size(a) == size(b) && all(((x, y),) -> isequal(x, y) || x == y, zip(a, b))
    function relerr(a, b)
        return maximum(abs, filter(isfinite, a - b); init = 0.0) /
               maximum(abs, filter(isfinite, b); init = 0.0)
    end
    # The prior after each step of a stream, and its call with no data at once.
    function stream(pe, r = rd, e = edges)
        out = []
        for k in 1:(length(e) - 1)
            pe = partial_fit!(pe, rows(r, (e[k] + 1):e[k + 1]))
            pr = try
                prior(pe)
            catch err
                err
            end
            push!(out, (; pe = pe, pr = pr))
        end
        return out
    end
    batch(pe, k, r = rd, e = edges) = prior(pe, rows(r, 1:e[k + 1]))
    function message(f)
        return try
            f()
            ""
        catch err
            sprint(showerror, err)
        end
    end
    function agrees(x, b)
        return same(x.mu, b.mu) &&
               same(x.sigma, b.sigma) &&
               same(x.X, b.X) &&
               same(x.o_X, b.o_X) &&
               same(x.fpr.X, b.fpr.X) &&
               same(x.rr.csr.f, b.rr.csr.f) &&
               same(x.rr.vs, b.rr.vs) &&
               same(x.rr.Ms, b.rr.Ms) &&
               same(x.rr.rw, b.rr.rw) &&
               isequal(x.rr.edof, b.rr.edof) &&
               x.rr.idx == b.rr.idx
    end
    dropped(pr) = pr.rr.nf[po.dropped_factor_indices(pr.rr.fcb)]
    style = (; grid_config("Base", rd)..., families = ["style" => nothing])
    industry = grid_config("FamOne", rd)

    @testset "The carry fold equals the batch fit at each step" begin
        # Measured: every configuration gives a difference of exactly zero, because each
        # output of an observation reads no later observation, and the step runs the code of
        # the batch fit on the new observations.
        for name in
            ("Base", "Lag2", "Blend", "Neutralised", "FamStated", "FamTwo", "FcFixed",
             "FcCustom")
            pe = CrossSectionalFactorPrior(; lambda = 1, grid_config(name, rd)...)
            for (k, x) in enumerate(stream(pe))
                @test agrees(x.pr, batch(pe, k))
            end
        end
        # One observation at a time after the first fit.
        pe = CrossSectionalFactorPrior(; lambda = 1, style...)
        e = (0, 90, 91, 92, 93, 250)
        for (k, x) in enumerate(stream(pe, rd, e))
            @test agrees(x.pr, batch(pe, k, rd, e))
        end
        # The state keeps the standardised idiosyncratic returns, and a threshold above zero
        # makes it keep them with no fill too, which the correlation reads. A step appends
        # the rows of its new observations to both (#1565).
        pe = CrossSectionalFactorPrior(; lambda = 1, th = 0.1, style...)
        for (k, x) in enumerate(stream(pe, rd, e))
            @test agrees(x.pr, batch(pe, k, rd, e))
            st = x.pe.cache
            @test size(st.S) == size(st.Sc) == size(st.csr.eps)
        end
    end

    @testset "A Calibration Rule resolves at each step (#1481)" begin
        # A rule that reads the number of fitted observations moves at each step. The carry
        # resolves it at the call with no data, as the batch fit over the same rows does.
        nobs(k, p, w, s, x) = size(x.cs.csfm.vs, 1) / 1000
        half(k, p, w, s, x) = 0.5
        pe = CrossSectionalFactorPrior(; lambda = 1,
                                       merge(grid_config("FcFixed", rd),
                                             (; lambda = half, c = nobs))...)
        cs = Float64[]
        for (k, x) in enumerate(stream(pe))
            b = batch(pe, k)
            @test x.pr.rr.c == b.rr.c == size(b.rr.vs, 1) / 1000
            @test x.pr.rr.lambda == 0.5
            @test agrees(x.pr, b)
            push!(cs, x.pr.rr.c)
        end
        @test allunique(cs)
    end

    @testset "The state carries the Return Forecast history (#1484)" begin
        # A member that publishes no history is fitted again at each block row. The state
        # carries those rows, so a step fits the member once for each new observation, and
        # the history equals the one the batch fit makes over the same rows.
        function history_case(cfg, e)
            rl, rc = CarryHistoryShrinkage(), CarryHistoryScale()
            bl, bc = CarryHistoryShrinkage(), CarryHistoryScale()
            pc = CrossSectionalFactorPrior(; cfg..., lambda = rl, c = rc)
            pb = CrossSectionalFactorPrior(; cfg..., lambda = bl, c = bc)
            xs = stream(pc, rd, e)
            for (k, x) in enumerate(xs)
                b = batch(pb, k, rd, e)
                @test x.pr.rr.lambda == b.rr.lambda
                @test x.pr.rr.c == b.rr.c
                @test agrees(x.pr, b)
                @test same(rl.seen[k], bl.seen[k])
                @test same(rc.seen[k], bc.seen[k])
                @test same(rl.seen[k], rc.seen[k])
            end
            return (; xs = xs, seen = rl.seen)
        end
        target = grid_config("FcTarget", rd)
        # Blocks of rows, then one observation at a time.
        for e in (edges, (0, 90, 91, 92, 93))
            (; xs, seen) = history_case(target, e)
            for (k, x) in enumerate(xs)
                H = x.pe.cache.hist
                # Every block row but the last, which the call with no data appends.
                @test size(H, 1) == size(seen[k], 1) - 1
                @test same(H, seen[k][1:(end - 1), :])
                # A step that fits the new observations alone keeps the carried rows.
                if k > 1
                    Hp = xs[k - 1].pe.cache.hist
                    @test same(H[1:size(Hp, 1), :], Hp)
                end
            end
            @test allunique([x.pr.rr.lambda for x in xs])
        end
        # A batch choice that moves fits every observation again, and makes every row again.
        # Measured: the move at the second step changes the old rows by up to 1.9e-17, so a
        # carry that kept them would not equal the batch fit.
        (; xs) = history_case((; target..., families = ["style" => nothing]), edges)
        @test [only(x.pe.cache.families).second for x in xs] ==
              ["style1", "style2", "style2"]
        H1, H2 = xs[1].pe.cache.hist, xs[2].pe.cache.hist
        @test !same(H1, H2[1:size(H1, 1), :])
        # A member that publishes its history carries no row: the call with no data reads the
        # history of the Result it fits.
        (; xs) = history_case(grid_config("FcEW", rd), edges)
        @test all(x -> isnothing(x.pe.cache.hist), xs)
        @test !PortfolioOptimisers.cross_sectional_carries_history(CrossSectionalFactorPrior(;
                                                                                             grid_config("FcEW",
                                                                                                         rd)...,
                                                                                             c = CarryHistoryScale()))
        # A prior whose slots read no history carries none.
        @test isnothing(last(stream(CrossSectionalFactorPrior(; target...))).pe.cache.hist)
        # The shipped rules that read the history split each carried row against the exposures
        # and the weights of its row, so they resolve on the carry fold as on the batch fit.
        rules = (; lambda = PrecisionBlend(; err = ForecastHistoryError()),
                 c = ForecastCalibrationSlope(; wu = ThresholdWarmUp(; min_obs = 2)))
        for cfg in (target, (; target..., families = ["style" => nothing]))
            pr_rules = CrossSectionalFactorPrior(; cfg..., rules...)
            for (k, x) in enumerate(stream(pr_rules))
                b = batch(pr_rules, k)
                @test x.pr.rr.lambda == b.rr.lambda
                @test x.pr.rr.c == b.rr.c
                @test 0 <= x.pr.rr.lambda <= 1
                @test x.pr.rr.c != 1
            end
        end
    end

    @testset "The carried panel rows" begin
        # The Passthrough exposures read one row and the lag is one, so two rows are kept.
        pe = CrossSectionalFactorPrior(; lambda = 1, style...)
        x = last(stream(pe))
        @test po.cross_sectional_carry_rows(pe) == 2 == size(x.pe.cache.win.X, 1)
        @test size(po.sample_buffer(po.returns_buffer(x.pe.cache))) == size(rd.X)
        # A Return Forecast that reads the panel keeps every row, a custom one keeps the
        # rows of the exposures.
        fixed = CrossSectionalFactorPrior(; lambda = 1, grid_config("FcFixed", rd)...)
        @test isnothing(po.cross_sectional_carry_rows(fixed))
        @test size(last(stream(fixed)).pe.cache.win.X, 1) == 250
        @test po.cross_sectional_carry_rows(CrossSectionalFactorPrior(; lambda = 1,
                                                                      grid_config("FcCustom",
                                                                                  rd)...)) ==
              2
        # A finite look-back keeps its last rows, an exponentially weighted Descriptor keeps
        # every row. The synthetic panel lists, delists and leaves gaps.
        syn = synthetic_asset_panel(; n_assets = 60, n_observations = 160, n_industries = 4,
                                    rng = StableRNG(1471)).rd
        bounded = ["market" => ConstantExposure(),
                   "reversal" => CompositeExposure(; descriptors = [Reversal()]),
                   "value" => CompositeExposure(; descriptors = [BookToPrice()]),
                   "industry" => OneHotExposure(; field = "industry", family = "industry")]
        ew = vcat(bounded, ["beta" => CompositeExposure(; descriptors = [EWMarketBeta()])])
        # The first block lies inside the Descriptor warm-up, and the next ones are too short
        # for the factor prior, so the fold refuses where the batch fit refuses, with the same
        # type of error. Inside the warm-up the batch fit refuses in its warm-up and the fold at
        # its call with no data, so the two messages differ there.
        e = (0, 10, 40, 41, 80, 81, 120, 160)
        for (factors, kept) in ((bounded, 22), (ew, 160))
            pe = CrossSectionalFactorPrior(; lambda = 1, factors = factors,
                                           families = ["industry" => nothing], minra = 5,
                                           pe = GRID_PE, ve = GRID_VE)
            for (k, x) in enumerate(stream(pe, syn, e))
                b = try
                    batch(pe, k, syn, e)
                catch err
                    err
                end
                if isa(b, LowOrderPrior)
                    # Measured 1.3e-15 on the bounded case: a rolling return over the cut
                    # rows is a difference of cumulative sums from another first row (#1470).
                    # Exactly zero on the unbounded case.
                    @test relerr(x.pr.mu, b.mu) < 1e-14 &&
                          relerr(x.pr.sigma, b.sigma) < 1e-14
                    @test isequal(isnan.(x.pr.sigma), isnan.(b.sigma))
                else
                    @test typeof(x.pr) == typeof(b)
                end
            end
            @test size(last(stream(pe, syn, e)).pe.cache.win.X, 1) == kept
        end
    end

    @testset "The Choice Rule" begin
        # A batch choice that moves fits every observation again, and equals the batch fit.
        sb = stream(CrossSectionalFactorPrior(; lambda = 1, style...))
        @test [dropped(x.pr) for x in sb] == [["style1"], ["style2"], ["style2"]]
        @test all(k -> agrees(sb[k].pr,
                              batch(CrossSectionalFactorPrior(; lambda = 1, style...), k)),
                  1:3)
        # A pinned choice is recorded in the state, and `families` stays as it is. Its
        # call with no data is the batch fit with the member stated.
        sp = stream(CrossSectionalFactorPrior(; lambda = 1, style...,
                                              choice = PinnedChoice()))
        @test all(x -> x.pe.families == ["style" => nothing], sp)
        @test all(x -> x.pe.cache.families == ["style" => "style1"], sp)
        stated = CrossSectionalFactorPrior(; lambda = 1, style...,
                                           families = ["style" => "style1"])
        @test all(k -> agrees(sp[k].pr, batch(stated, k)), 1:3)
        # A factor that is empty at the first fit and comes alive fits every observation
        # again.
        rdz = deepcopy(rd)
        po.panel_field(rdz.pnl, "style1").vals[1:120, :] .= 0.0
        pe = CrossSectionalFactorPrior(; lambda = 1, grid_config("Base", rdz)...)
        sz = stream(pe, rdz)
        @test [x.pe.cache.lv for x in sz] == [[true, false, true], trues(3), trues(3)]
        @test all(k -> agrees(sz[k].pr, batch(pe, k, rdz)), 1:3)
    end

    @testset "Folds the members that fold, and refits the rest" begin
        # A rolling window does not fold, so every step fits every observation again over
        # the carried histories, the second pass of the weights too.
        wv = WindowedVariance(;
                              ve = ExpWeightedVariance(; decay = 2.0^(-1 / 20),
                                                       min_obs = 5), window = 60)
        for cfg in (style, grid_config("Blend", rd))
            pe = CrossSectionalFactorPrior(; lambda = 1, cfg..., ve = wv)
            @test all(((k, x),) -> agrees(x.pr, batch(pe, k)), enumerate(stream(pe)))
        end
        # A factor prior that does not fold refits over the carried factor returns.
        pe = CrossSectionalFactorPrior(; lambda = 1, style...,
                                       pe = EntropyPoolingPrior(; pe = GRID_PE))
        sx = stream(pe)
        @test all(x -> isa(x.pe.cache.pe, EntropyPoolingPrior), sx)
        @test all(k -> agrees(sx[k].pr, batch(pe, k)), 1:3)
    end

    @testset "Parity with the oracle's online update" begin
        # The prior of each step against the stored block of that step. `mu` compares cell by
        # cell. `sigma` compares against its largest entry: an off-diagonal entry of two assets
        # near zero correlation is a cancellation in `L F L'` (#1376), measured maxrel 1.3e-10
        # cell by cell. The factor returns compare cell by cell with an absolute floor: the
        # industry level "Utilities" is empty after its only asset delists, so its factor return
        # is a round-off zero on both sides, measured maxrel 1.9 and maxabs 1.5e-16. The floor
        # is at most 6e-14 of the largest factor return, and every other cell measured maxrel
        # 5.7e-13.
        function check(out, unit_case, src)
            load(o) = parity_load("CrossSectionalFactorPrior", unit_case, "$(src)$(o)")
            mu, S, F = load("Mu"), load("Sigma"), load("FactorReturns")
            r0 = 0
            for k in 1:3
                pr = out[k].pr
                n = edges[k + 1] - 1
                @test parity_compare(pr.mu, mu[k, :]; name = "$(unit_case) mu $(k)").ok
                @test parity_compare(pr.sigma, S[((k - 1) * N + 1):(k * N), :];
                                     scale = :array, name = "$(unit_case) sigma $(k)").ok
                @test parity_compare(pr.fpr.X, F[(r0 + 1):(r0 + n), :]; atol = 1e-15,
                                     name = "$(unit_case) factor returns $(k)").ok
                r0 += n
            end
        end
        pinned(cfg) = stream(CrossSectionalFactorPrior(; lambda = 1, cfg...,
                                                       choice = PinnedChoice()))
        # The pinned choice reproduces the oracle's online update, which pins the dropped
        # member at its first call. Measured over the three steps: mu maxrel 1.7e-14, sigma
        # maxscaled 4.2e-13, factor returns maxrel 4.6e-14 on `Style`; mu maxrel 5.3e-13,
        # sigma maxscaled 6.5e-13, factor returns maxabs 1.5e-16 on `Industry`.
        check(pinned(style), "OnlineStyle", "Fold")
        check(pinned(industry), "OnlineIndustry", "Fold")
        # Currency factors on the carry route (#1479). The oracle's online update reads the
        # Currency Excess Returns of each batch, and it pins `style1` as in `Style`. Measured
        # over the three steps: mu maxrel 2.3e-13, sigma maxscaled 4.2e-13, factor returns
        # maxrel 9.3e-14.
        cp = pinned((; grid_config("Currency", rd)..., families = ["style" => nothing]))
        @test all(x -> x.pe.cache.families == ["style" => "style1"], cp)
        check(cp, "OnlineCurrencyStyle", "Fold")
        # A seed window on the factor prior cuts the factor returns of the first fit to the
        # last 60, and every later step folds every row, as the oracle's does. Measured: mu
        # maxrel 7.2e-15, sigma maxscaled 3.7e-13, factor returns maxrel 4.6e-14 on `Style`;
        # mu maxrel 1.4e-13, sigma maxscaled 7.3e-13, factor returns maxabs 1.5e-16 on
        # `Industry`.
        decay = 2.0^(-1 / 20)
        seed = EmpiricalPrior(;
                              me = WindowedExpectedReturns(;
                                                           me = ExpWeightedExpectedReturns(;
                                                                                           decay = decay,
                                                                                           min_obs = 5),
                                                           window = 60,
                                                           rule = SeedWindow()),
                              ce = WindowedCovariance(;
                                                      ce = ExpWeightedCovariance(;
                                                                                 decay = decay,
                                                                                 min_obs = 5,
                                                                                 centring = PreCentred()),
                                                      window = 60, rule = SeedWindow()))
        ss = pinned((; style..., pe = seed))
        check(ss, "CarrySeedStyle", "Fold")
        check(pinned((; industry..., pe = seed)), "CarrySeedIndustry", "Fold")
        # The first fit is the batch fit, and the window of the first fit alone moves the
        # later calls with no data away from it: sigma by 19 % of its largest entry at the last step.
        spe = CrossSectionalFactorPrior(; lambda = 1, style..., pe = seed)
        @test agrees(ss[1].pr, batch(spe, 1))
        @test relerr(ss[3].pr.sigma, batch(spe, 3).sigma) > 0.1
    end

    @testset "On the online step of an optimiser" begin
        pe = CrossSectionalFactorPrior(; lambda = 1, style...)
        o = po.update_online_estimator(InverseVolatility(; pe = pe))
        for k in 1:3
            o = partial_fit!(o, rows(rd, (edges[k] + 1):edges[k + 1]))
            bo, brd = po.batch_from_state(o)
            @test isa(o.pe.cache, po.CrossSectionalCarryState)
            @test agrees(bo.pe, batch(pe, k))
            @test isequal(brd.X, rd.X[1:edges[k + 1], :])
        end
        @test optimise(o).w == optimise(InverseVolatility(; pe = pe), rd).w
        # `partial_fit` copies the state, so the earlier prior answers as it did.
        e1 = partial_fit!(pe, rows(rd, 1:90))
        p1 = prior(e1)
        e2 = partial_fit(e1, rows(rd, 91:170))
        @test agrees(prior(e1), p1) && agrees(prior(e2), batch(pe, 2))
    end

    @testset "An observed factor (#1479)" begin
        # The carry derives each row of the returns net of the observed factors one time, from
        # the observed exposures of the row `lag` observations before it, and carries the row.
        # The batch fit derives the first `lag` rows of the sample from the exposure of the same
        # row, so a window that derived its first rows again would give other values.
        base = grid_config("Base", rd)
        ccy = grid_config("Currency", rd)
        msens = "msens" => CompositeExposure(;
                                             descriptors = [EWMacroSensitivity(; series = "MACRO",
                                                                               half_life = 10)],
                                             outlier = nothing, scoring = nothing, family = "msens")
        # An observed member over a Descriptor of the returns reads the returns net of the
        # Currency Factors, a second derived series.
        rev = "rev" => ObservedExposure(;
                                        xe = CompositeExposure(; descriptors = [Reversal()],
                                                               outlier = nothing, scoring = nothing,
                                                               family = "rev"), series = "MACRO",
                                        family = "rev")
        mixed = (; ccy..., factors = [ccy.factors; rev])
        rfr = FixedWeightedReturnForecast(;
                                          scores = DescriptorScores(;
                                                                    descriptors = [Reversal()]),
                                          scale = 0.02)
        wv = WindowedVariance(;
                              ve = ExpWeightedVariance(; decay = 2.0^(-1 / 20),
                                                       min_obs = 5), window = 60)
        exact = (; Currency = ccy, CurrencyLx = grid_config("CurrencyLx", rd),
                 Macro = grid_config("Macro", rd),
                 Sensitivity = (; base..., factors = [base.factors; msens]),
                 MixedForecast = (; mixed..., rfe = rfr),
                 Entropy = (; ccy..., pe = EntropyPoolingPrior(; pe = GRID_PE)),
                 Rolling = (; ccy..., ve = wv))
        # Measured: every case gives a difference of exactly zero at each step. The forecast
        # reads the net returns of every row, so the carry keeps every row and every derived row.
        for cfg in exact
            pe = CrossSectionalFactorPrior(; lambda = 1, cfg...)
            @test po.reads_exogenous_series(pe)
            @test all(((k, x),) -> agrees(x.pr, batch(pe, k)), enumerate(stream(pe)))
        end
        # The `Reversal` exposure reads its last 21 rows, so the carry keeps 22 rows, and a
        # rolling return over the cut rows is a difference of cumulative sums from another first
        # row (#1470). Measured over the three steps, relative to the largest entry: mu 4.6e-16,
        # sigma 4.3e-16, factor returns 1.8e-16; under `lag = 2`, 1.4e-16, 3.1e-16 and 1.8e-16.
        @test po.cross_sectional_carry_rows(CrossSectionalFactorPrior(; lambda = 1,
                                                                      mixed...)) == 22
        for cfg in (mixed, (; mixed..., lag = 2))
            pe = CrossSectionalFactorPrior(; lambda = 1, cfg...)
            for (k, x) in enumerate(stream(pe))
                b = batch(pe, k)
                @test relerr(x.pr.mu, b.mu) < 1e-12 &&
                      relerr(x.pr.sigma, b.sigma) < 1e-12 &&
                      relerr(x.pr.fpr.X, b.fpr.X) < 1e-12
                @test isequal(isnan.(x.pr.sigma), isnan.(b.sigma))
            end
        end
        # A macro sensitivity is a recursion from the first row and states no look-back, so the
        # carry keeps every row.
        @test isnothing(po.cross_sectional_carry_rows(CrossSectionalFactorPrior(;
                                                                                lambda = 1,
                                                                                exact.Sensitivity...)))
        # One observation at a time after the first fit.
        pe = CrossSectionalFactorPrior(; lambda = 1, ccy...)
        e = (0, 90, 91, 92, 93, 250)
        @test all(((k, x),) -> agrees(x.pr, batch(pe, k, rd, e)),
                  enumerate(stream(pe, rd, e)))
        # The window carries the series and the derived rows of its two rows, and the buffer
        # records every column of the series.
        st = last(stream(pe)).pe.cache
        @test st.win.ne == rd.ne && isequal(st.win.E, rd.E[249:250, :])
        @test size(st.der.Xl, 1) == 2 && isnothing(st.der.Xn)
        @test isequal(po.exogenous_buffer_kwargs(po.returns_buffer(st)).E, rd.E)
        # On the online step of an optimiser the prior's buffer owns the series, and the Fold
        # Context reads it back.
        o = po.update_online_estimator(InverseVolatility(; pe = pe))
        for k in 1:3
            o = partial_fit!(o, rows(rd, (edges[k] + 1):edges[k + 1]))
        end
        @test isa(o.pe.cache, po.CrossSectionalCarryState) && isnothing(o.cache.E)
        bo, brd = po.batch_from_state(o)
        @test brd.ne == rd.ne && isequal(brd.E, rd.E)
        @test agrees(bo.pe, batch(pe, 3))
        @test optimise(o).w == optimise(InverseVolatility(; pe = pe), rd).w
        # A step with no series, and a step with other names.
        nor = ReturnsResult(; nx = rd.nx, X = rd.X, pnl = rd.pnl)
        @test occursin("this step carries no `rd.E`",
                       message(() -> partial_fit!(pe, rows(nor, 1:90))))
        e1 = partial_fit!(pe, rows(rd, 1:90))
        @test occursin("this step carries no `rd.E`",
                       message(() -> partial_fit!(e1, rows(nor, 91:170))))
        ren = ReturnsResult(; nx = rd.nx, X = rd.X, ne = [rd.ne[1:(end - 1)]; "OTHER"],
                            E = rd.E, pnl = rd.pnl)
        @test occursin("a later block must carry the same names",
                       message(() -> partial_fit!(e1, rows(ren, 91:170))))
        # The carry keeps every fitted row, so the step that fits a row refuses an infinite
        # observed return on it, as the batch fit does. A NaN marks a gap, and the step fits
        # it as the batch fit does (#1530). Row 1 has no lagged exposure, so the regression
        # never reads it.
        for (row, v, refused) in ((200, Inf, true), (200, NaN, false), (1, NaN, false))
            E = copy(rd.E)
            E[row, 1] = v
            rn = ReturnsResult(; nx = rd.nx, X = rd.X, ne = rd.ne, E = E, pnl = rd.pnl)
            if refused
                e2 = partial_fit!(partial_fit!(pe, rows(rn, 1:90)), rows(rn, 91:170))
                err = try
                    partial_fit!(e2, rows(rn, 171:250))
                catch x
                    x
                end
                @test isa(err, IsNonFiniteError)
                @test occursin("first at observation 200 of the returns data",
                               sprint(showerror, err))
                @test_throws IsNonFiniteError batch(pe, 3, rn)
            else
                @test agrees(last(stream(pe, rn)).pr, batch(pe, 3, rn))
            end
        end
    end

    @testset "Refusals" begin
        pe = CrossSectionalFactorPrior(; lambda = 1, style...)
        m = message(() -> prior(partial_fit!(pe, rows(rd, 1:2))))
        @test occursin("holds 2 observation(s) after the Descriptor warm-up", m)
        @test occursin("lag + 2 = 3", m)
        m = message(() -> partial_fit!(pe, ReturnsResult(; nx = rd.nx, X = rd.X[1:5, :])))
        @test occursin("the step carries no panel", m)
        e = partial_fit!(pe, rows(rd, 1:90))
        m = message(() -> po.port_opt_view(e, 1:5))
        @test occursin("not a slice of the ones it carries", m)
        m = message(() -> po.merge_states(e.cache, partial_fit!(pe, rows(rd, 91:170)).cache))
        @test occursin("cannot merge two states fitted on disjoint blocks", m)
        @test occursin("the matrix form of `partial_fit!` carries none",
                       message(() -> partial_fit!(pe, rd.X)))
    end
end
