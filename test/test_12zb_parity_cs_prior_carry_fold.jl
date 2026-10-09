#=
The carry fold of the Cross-Sectional Factor Prior (#1471, map #1375, ADR 0193).

An unwrapped prior folds as a carry: its first step seeds a `CrossSectionalCarryState`, and each
step computes the exposures, the regression and the idiosyncratic variance of the new
observations alone, from the panel rows that its Descriptors read. The factor prior and the
idiosyncratic variance fold, and the Return Forecast and the idiosyncratic correlation refit at the
call with no data. A factor that comes alive fits every carried observation again. A batch choice
that moves folds the move, and solves again each observation whose answer depends on the basis
(#1605, #1613).

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
# A test-local rule of `c` that records the fit of the prior it reads (#1572).
struct ContextRecordScale <:
       PortfolioOptimisers.AbstractOrthogonalForecastScaleCalibrationAlgorithm
    seen::Vector{Any}
end
ContextRecordScale() = ContextRecordScale(Any[])
PortfolioOptimisers.reads_forecast_history(::ContextRecordScale) = true
function (r::ContextRecordScale)(key, pr, w, slv, ctx)
    push!(r.seen, ctx.cs)
    return 1.0
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
    # Equal to rounding, with the same `NaN` cells. The gap over the largest finite entry, and
    # zero when the two are equal, so a block with no finite entry other than zero compares.
    function close_to(p, q; tol = 1e-12)
        gap = maximum(abs, filter(isfinite, p - q); init = 0.0)
        return size(p) == size(q) &&
               isequal(isnan.(p), isnan.(q)) &&
               (iszero(gap) || relerr(p, q) < tol)
    end
    # A move of a batch choice keeps the factor returns of the old basis and selects their
    # columns, where the batch fit solves each row in the new basis, so the two differ by
    # rounding alone (#1605, #1613).
    function near(x, b; tol = 1e-12)
        return all(((p, q),) -> close_to(p, q; tol = tol),
                   ((x.mu, b.mu), (x.sigma, b.sigma), (x.X, b.X), (x.o_X, b.o_X),
                    (x.fpr.X, b.fpr.X), (x.rr.csr.f, b.rr.csr.f), (x.rr.vs, b.rr.vs),
                    (x.rr.Ms, b.rr.Ms), (x.rr.rw, b.rr.rw))) && x.rr.idx == b.rr.idx
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
        # One observation at a time after the first fit. The batch choice of `style` moves at
        # the last step, and the fold of the move equals the batch fit to rounding (#1605,
        # #1613), so the configurations of `style` compare with `near` from here on.
        pe = CrossSectionalFactorPrior(; lambda = 1, style...)
        e = (0, 90, 91, 92, 93, 250)
        for (k, x) in enumerate(stream(pe, rd, e))
            @test near(x.pr, batch(pe, k, rd, e))
        end
        # The state keeps the standardised idiosyncratic returns, and a step appends the rows
        # of its new observations (#1565). A threshold above zero makes the step fold the
        # rows with no fill into the `ce` of the prior, so the state keeps no row with no
        # fill. The fold of the default `ExpWeightedCovariance` is its batch recursion, so
        # the carry equals the batch fit to the last bit (#1594).
        pe = CrossSectionalFactorPrior(; lambda = 1, th = 0.1, style...)
        for (k, x) in enumerate(stream(pe, rd, e))
            @test near(x.pr, batch(pe, k, rd, e))
            st = x.pe.cache
            @test size(st.S) == size(st.csr.eps)
            @test isnothing(st.Sc)
            @test isa(st.ce.cache, po.ExpWeightedCovarianceState)
        end
        # A `ce` that does not fold makes the state keep the rows with no fill, and the call
        # with no data estimates the correlation over them again.
        pe = CrossSectionalFactorPrior(; lambda = 1, th = 0.1,
                                       ce = Covariance(; alg = SemiMoment()), style...)
        for (k, x) in enumerate(stream(pe, rd, e))
            @test near(x.pr, batch(pe, k, rd, e))
            st = x.pe.cache
            @test size(st.S) == size(st.Sc) == size(st.csr.eps)
            @test isnothing(st.ce)
        end
        # A `ce` with a finite fill value folds the filled rows with no mask, as the batch
        # fit estimates them. Its fold is Welford's recursion, which equals the batch fit up
        # to round-off.
        pe = CrossSectionalFactorPrior(; lambda = 1, th = 0.1, ce = Covariance(), style...)
        for (k, x) in enumerate(stream(pe, rd, e))
            b = batch(pe, k, rd, e)
            @test isnothing(x.pe.cache.Sc)
            @test relerr(x.pr.sigma, b.sigma) < 1e-12
            @test relerr(x.pr.mu, b.mu) < 1e-12
            @test close_to(x.pr.X, b.X)
        end
        # The batch fit and the read-out of the carry fold run the same lift, so the carry
        # fold equals the batch fit under each Systematic Repair rule (#1576).
        for srep in (NoSystematicRepair(), SystematicRepair())
            pe = CrossSectionalFactorPrior(; lambda = 1, srep = srep, style...)
            for (k, x) in enumerate(stream(pe, rd, e))
                @test near(x.pr, batch(pe, k, rd, e))
            end
        end
        # A member that no asset of an observation loads on leaves the zero-sum condition of
        # that observation, on a step as in the batch fit (#1606). The only asset in Utilities
        # delists after data row 225, and the step of row 226 fits that row alone.
        dl = parity_panel(; T = 300, N = 60, seed = 1601).rd
        de = (0, 200, 225, 226, 300)
        dlf = ["market" => ConstantExposure(),
               "industry" => OneHotExposure(; field = "industry", family = "industry"),
               "size" =>
                   CompositeExposure(; descriptors = [LogMarketCap()], family = "style")]
        for fam in (["industry" => "industry=Energy"], ["industry" => nothing])
            pe = CrossSectionalFactorPrior(; factors = dlf, families = fam, minra = 5)
            for (k, x) in enumerate(stream(pe, dl, de))
                @test agrees(x.pr, batch(pe, k, dl, de))
            end
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
        # A stream whose batch choice moves equals the batch fit to rounding, `exact = false`.
        function history_case(cfg, e; exact = true)
            rl, rc = CarryHistoryShrinkage(), CarryHistoryScale()
            bl, bc = CarryHistoryShrinkage(), CarryHistoryScale()
            pc = CrossSectionalFactorPrior(; cfg..., lambda = rl, c = rc)
            pb = CrossSectionalFactorPrior(; cfg..., lambda = bl, c = bc)
            xs = stream(pc, rd, e)
            eqn(p, q) = exact ? p == q : isapprox(p, q; rtol = 1e-12)
            eqh(p, q) = exact ? same(p, q) : close_to(p, q)
            for (k, x) in enumerate(xs)
                b = batch(pb, k, rd, e)
                @test eqn(x.pr.rr.lambda, b.rr.lambda)
                @test eqn(x.pr.rr.c, b.rr.c)
                @test exact ? agrees(x.pr, b) : near(x.pr, b)
                @test eqh(rl.seen[k], bl.seen[k])
                @test eqh(rc.seen[k], bc.seen[k])
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
        # A batch choice that moves folds the move, which keeps the old rows of the history
        # (#1605, #1613). Measured when the move fitted every observation again: the move at
        # the second step changes the old rows by up to 1.9e-17, so the carry equals the batch
        # fit to rounding.
        (; xs) = history_case((; target..., families = ["style" => nothing]), edges;
                              exact = false)
        @test [only(x.pe.cache.families).second for x in xs] ==
              ["style1", "style2", "style2"]
        H1, H2 = xs[1].pe.cache.hist, xs[2].pe.cache.hist
        @test same(H1, H2[1:size(H1, 1), :])
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

    @testset "A fitted forecast leaves a pair of leverage one out of its fit (#1571)" begin
        # The panel and the factors of map #1562. Asset 3 is the only member of its industry
        # on 627 of the 927 block rows, so its residual is zero by construction there and 588
        # of its variances are exactly zero. Before #1571 both units of
        # `TargetReturnForecast` raised a `DomainError` at `vs[40, 3]`.
        pp = parity_panel(; T = 1200, N = 40, seed = 1471).rd
        Np = size(pp.X, 2)
        sty(d) = CompositeExposure(; descriptors = [d], family = "style")
        cfg = (; minra = 5, pe = GRID_PE, ve = GRID_VE, families = ["industry" => nothing],
               factors = ["market" => ConstantExposure(),
                          "industry" =>
                              OneHotExposure(; field = "industry", family = "industry"),
                          "size" => sty(LogMarketCap()), "value" => sty(BookToPrice()),
                          "momentum" => sty(RollingMomentum()),
                          "reversal" => sty(Reversal())])
        blk = prior(CrossSectionalFactorPrior(; cfg...), pp).rr
        h1 = blk.csr.h1
        @test findall(i -> any(view(h1, :, i)), 1:Np) == [3]
        @test count(h1) == 627
        # The same block with `NaN` at every marked pair by hand, and no mask.
        e2 = copy(blk.csr.eps)
        e2[h1] .= NaN
        v2 = copy(blk.vs)
        v2[h1] .= NaN
        csr2 = CrossSectionalRegression(; f = blk.csr.f, eps = e2, n = blk.csr.n,
                                        b = blk.csr.b)
        blk2 = po.Accessors.setproperties(blk, (; csr = csr2, vs = v2))
        ds = DescriptorScores(;
                              descriptors = [Passthrough(; field = "net_income_ttm"),
                                             Passthrough(; field = "sales_ttm")])
        sharpe = IdiosyncraticSharpeUnit()
        # Each fit equals the fit on the block with the cells set by hand. The oracle passes
        # here only because its residual is rounding noise (3.6e-16, variance 7.4e-33), and
        # its fit reads that noise: its calibration is 1.629 in the return unit and 0.964 in
        # the Sharpe unit. With the cells of asset 3 set to `NaN`, the oracle gives 1.874 and
        # 0.445 (#1566), and these are the values below. So this is a deliberate difference
        # from the oracle as it stands.
        # The oracle calibrates out of fold in its batch fit, so the comparison states
        # `KFold()`. The prequential default of #1575 treats the two blocks alike too. It
        # gives -2.020 and -2.069 here: the two Passthrough scores anti-predict out of time.
        calibs = Dict("return" => 1.8743201813934023, "sharpe" => 0.44520892743434026)
        for (unit, uname) in ((IdiosyncraticReturnUnit(), "return"), (sharpe, "sharpe"))
            rfe = TargetReturnForecast(; scores = ds, half_life = 10.0, unit = unit,
                                       cv = KFold())
            a = return_forecast(rfe, pp, blk)
            b = return_forecast(rfe, pp, blk2)
            @test isequal(a.calib, b.calib)
            @test isapprox(a.calib, calibs[uname]; rtol = 1e-10)
            @test same(a.mu, b.mu)
            rfe = TargetReturnForecast(; scores = ds, half_life = 10.0, unit = unit)
            a = return_forecast(rfe, pp, blk)
            b = return_forecast(rfe, pp, blk2)
            @test isequal(a.calib, b.calib)
            @test same(a.mu, b.mu)
            rfe = ExpWeightedReturnForecast(; scores = ds, unit = unit)
            a = return_forecast(rfe, pp, blk)
            b = return_forecast(rfe, pp, blk2)
            @test isequal(a.coef, b.coef) && isequal(a.A, b.A) && isequal(a.c, b.c)
            @test a.n == b.n
            @test same(a.mu, b.mu)
            # The read-out converts with the variance the block holds. In the Sharpe unit a
            # marked pair forecasts zero, and the block set by hand forecasts `NaN` there.
            @test same(a.hist[.!h1], b.hist[.!h1])
            if unit === sharpe
                @test all(iszero, filter(isfinite, a.hist[h1 .& (blk.vs .== 0)]))
            end
        end
        # The block does not change.
        @test count(iszero, filter(isfinite, view(blk.vs, :, 3))) == 588
        # The carry fold equals the batch fit under each fitted forecast, and carries the same
        # mask. Measured: every step is exact since the rolling returns fold from their state
        # (#1583). Before it, the cut of `RollingMomentum` (#1563) moved the exposures by
        # 1.6e-15, and the four forecasts by up to 5.6e-15 in `mu` and 4.5e-14 in the
        # calibration or the coefficients.
        ep = (0, 1150, 1151, 1200)
        fk = (; lambda = 0.4, c = 0.6, ofit = UnadjustedForecast())
        for unit in (IdiosyncraticReturnUnit(), sharpe),
            rfe in (TargetReturnForecast(; scores = ds, half_life = 10.0, unit = unit),
                    ExpWeightedReturnForecast(; scores = ds, unit = unit))

            pe = CrossSectionalFactorPrior(; cfg..., fk..., rfe = rfe)
            for (k, x) in enumerate(stream(pe, pp, ep))
                b = batch(pe, k, pp, ep)
                @test x.pr.rr.csr.h1 == b.rr.csr.h1
                @test relerr(x.pr.mu, b.mu) <= 1e-12
                @test relerr(x.pr.sigma, b.sigma) <= 1e-12
                @test relerr(x.pr.rr.vs, b.rr.vs) <= 1e-12
                fc(rf) = rf isa TargetReturnForecastResult ? [rf.calib] : rf.coef
                @test relerr(fc(x.pr.rr.rf), fc(b.rr.rf)) <= 1e-12
            end
        end
        @testset "A forecast of a pair of leverage one counts no return twice (#1572)" begin
            # On rows 1-880 asset 3 is the only member of its industry at the latest row. Its
            # row of I - P is zero, so its orthogonal forecast is zero, and its whole forecast
            # enters through the spanned coefficients g. The prior blends g with the factor
            # mean convexly, and the factor mean of its industry already holds the
            # idiosyncratic returns of asset 3. So the forecast replaces a share 1 - lambda of
            # that mean, and adds nothing to it. Measured: |b[3]| is 1.7e-17 at lambda 0 and
            # 1.0e-17 at lambda 0.4, and the largest |b| is 0.018 and 0.011.
            p880 = rows(pp, 1:880)
            fx = FixedWeightedReturnForecast(; scores = ds, scale = 0.02)
            m0 = prior(CrossSectionalFactorPrior(; cfg..., lambda = 1, c = 0), p880).mu
            for (lambda, c) in ((0.0, 1.0), (0.4, 0.6))
                pr = prior(CrossSectionalFactorPrior(; cfg..., lambda = lambda, c = c,
                                                     rfe = fx), p880)
                @test pr.rr.csr.h1[end, 3]
                @test abs(pr.rr.b[3]) < 1e-15
                @test maximum(abs, filter(isfinite, pr.rr.b)) > 1e-3
                @test isapprox(pr.mu[3], lambda * m0[3] + (1 - lambda) * pr.rr.rf.mu[3];
                               rtol = 1e-12)
            end
            # A pair of the calibration slope is the orthogonal forecast of a row and the
            # idiosyncratic return of the next row. When an asset becomes a pair of leverage
            # one at the next row, its target is zero by construction and its forecast is
            # not, so the pair would pull the slope towards zero. The panels of map #1562
            # hold no such pair, because asset 3 is marked at every row that fits it. So the
            # test marks one pair of another asset by hand, as the fit marks it.
            rec = ContextRecordScale()
            prior(CrossSectionalFactorPrior(; cfg..., lambda = 0.4, c = rec, rfe = fx), pp)
            cs = only(rec.seen)
            csr = cs.csfm.csr
            ap = po.cross_sectional_split_history(cs).ap
            t, j = Tuple(findfirst(k -> k[1] < size(ap, 1) &&
                                        !csr.h1[k[1], k[2]] &&
                                        !csr.h1[k[1] + 1, k[2]] &&
                                        isfinite(ap[k]) &&
                                        !iszero(ap[k]) &&
                                        isfinite(csr.eps[k[1] + 1, k[2]]),
                                   CartesianIndices(ap)))
            function with_cell(h, e)
                eps = copy(csr.eps)
                eps[t + 1, j] = e
                h1 = copy(csr.h1)
                h1[t + 1, j] = h
                c2 = CrossSectionalRegression(; f = csr.f, eps = eps, n = csr.n, b = csr.b,
                                              h1 = h1)
                return merge(cs,
                             (; csfm = po.Accessors.setproperties(cs.csfm, (; csr = c2))))
            end
            marked = po.orthogonal_forecast_pairs(with_cell(true, 0.0))
            byhand = po.orthogonal_forecast_pairs(with_cell(false, NaN))
            pooled = po.orthogonal_forecast_pairs(with_cell(false, 0.0))
            @test isequal(marked, byhand)
            @test length(pooled.a) == length(marked.a) + 1
            slope(x) = po.forecast_calibration_slope(x.a, x.b, x.q)
            @test slope(pooled) != slope(marked)
        end
    end

    @testset "The carried panel rows" begin
        # The Passthrough exposures read one row and the lag is one, so two rows are kept.
        pe = CrossSectionalFactorPrior(; lambda = 1, style...)
        x = last(stream(pe))
        @test po.cross_sectional_carry_rows(pe) == 2 == size(x.pe.cache.win.X, 1)
        @test size(po.sample_buffer(po.returns_buffer(x.pe.cache))) == size(rd.X)
        # A Return Forecast that refits at the call with no data keeps every row. One that
        # computes its history one observation at a time keeps the rows of its look-back
        # (#1573, #1574), and a custom one keeps the rows of the exposures.
        for name in ("FcFixed", "FcEW")
            fold = CrossSectionalFactorPrior(; lambda = 1, grid_config(name, rd)...)
            @test po.cross_sectional_carry_rows(fold) ==
                  po.lookback(fold) ==
                  size(last(stream(fold)).pe.cache.win.X, 1) ==
                  2
        end
        @test isnothing(po.cross_sectional_carry_rows(CrossSectionalFactorPrior(;
                                                                                grid_config("FcTarget",
                                                                                            rd)...)))
        @test po.cross_sectional_carry_rows(CrossSectionalFactorPrior(; lambda = 1,
                                                                      grid_config("FcCustom",
                                                                                  rd)...)) ==
              2
        # A finite look-back keeps its last rows, and a rolling return and an exponentially
        # weighted beta keep none beyond the lag, because they fold from their state (#1583,
        # #1608). The synthetic panel lists, delists and leaves gaps.
        syn = synthetic_asset_panel(; n_assets = 60, n_observations = 160, n_industries = 4,
                                    rng = StableRNG(1471)).rd
        bounded = ["market" => ConstantExposure(),
                   "reversal" => CompositeExposure(; descriptors = [Reversal()]),
                   "value" => CompositeExposure(; descriptors = [BookToPrice()]),
                   "industry" => OneHotExposure(; field = "industry", family = "industry")]
        ew = vcat(bounded, ["beta" => CompositeExposure(; descriptors = [EWMarketBeta()])])
        # A step computes the exposures of its rows alone, each from the rows that its member
        # reads (#1563). A derived member reads the rows of its source that the step computes.
        derived = vcat(bounded,
                       ["value2" => DerivedExposure(; source = "value", f = x -> x .^ 2)])
        # The first block lies inside the Descriptor warm-up, and the next ones are too short
        # for the factor prior, so the fold refuses where the batch fit refuses, with the same
        # type of error. Inside the warm-up the batch fit refuses in its warm-up and the fold at
        # its call with no data, so the two messages differ there.
        e = (0, 10, 40, 41, 80, 81, 120, 160)
        for (factors, kept) in ((bounded, 2), (ew, 2), (derived, 2))
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
                    # Equal to the last bit: the rolling return folds from its cumulative
                    # sums since the first row (#1583). Before it, a step cut the rows of
                    # the member (#1563), and the cases moved by 6.9e-16 to 2.4e-15.
                    @test same(x.pr.rr.Ms, b.rr.Ms)
                    @test same(x.pr.mu, b.mu) && same(x.pr.sigma, b.sigma)
                else
                    @test typeof(x.pr) == typeof(b)
                end
            end
            @test size(last(stream(pe, syn, e)).pe.cache.win.X, 1) == kept
        end
    end

    @testset "The Choice Rule" begin
        # A batch choice that moves folds the move, and equals the batch fit to rounding.
        sb = stream(CrossSectionalFactorPrior(; lambda = 1, style...))
        @test [dropped(x.pr) for x in sb] == [["style1"], ["style2"], ["style2"]]
        @test all(k -> near(sb[k].pr,
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

    @testset "A move of a batch choice folds and solves no row of full rank again (#1605)" begin
        # The raw factor returns of an observation of full rank do not depend on the dropped
        # member, so a move selects their columns and folds the factor prior again. The
        # residuals of the old observations are the ones of the step before, bit for bit. Under
        # `SolvedUnseenMember()`, the rule of the `grid_config` cases, the move also solves each
        # observation with an Unseen Member again (#1613).
        for unseen in (ZeroUnseenMember(), SolvedUnseenMember()),
            pf in
            (GRID_PE, EmpiricalPrior(; me = SimpleExpectedReturns(), ce = Covariance()))

            pe = CrossSectionalFactorPrior(; lambda = 1, style..., unseen = unseen, pe = pf)
            sb = stream(pe)
            @test [dropped(x.pr) for x in sb] == [["style1"], ["style2"], ["style2"]]
            @test all(k -> dropped(batch(pe, k)) == dropped(sb[k].pr), 1:3)
            @test all(k -> near(sb[k].pr, batch(pe, k)), 1:3)
            # A refit leaves a plain `Matrix`, and a step appends to a view of a backing.
            @test sb[2].pe.cache.csr.f isa SubArray
            e1, e2 = sb[1].pe.cache.csr.eps, sb[2].pe.cache.csr.eps
            @test same(e2[1:size(e1, 1), :], e1)
            @test same(sb[2].pe.cache.W[1:size(e1, 1), :], sb[1].pe.cache.W)
        end
        @test CrossSectionalFactorPrior(; style...).unseen === SolvedUnseenMember()
        # A target that the library does not know can penalise its coefficients, so a move
        # under it fits every observation again.
        @test isnothing(po.cross_sectional_move_basis(ZeroUnseenMember(),
                                                      CrossSectionalTargetRegression(),
                                                      nothing, nothing))
        # The panel of #1606: the only asset in Utilities delists after data row 225, so row
        # 226 has an Unseen Member. A larger market cap of Energy moves the automatic member
        # from Software to Energy. From row 150 the move comes before the step of row 226, and
        # from row 229 it comes after it, so the move solves row 226 again under
        # `SolvedUnseenMember()`.
        dlf = ["market" => ConstantExposure(),
               "industry" => OneHotExposure(; field = "industry", family = "industry"),
               "size" =>
                   CompositeExposure(; descriptors = [LogMarketCap()], family = "style")]
        function energy_panel(from, x)
            d = parity_panel(; T = 300, N = 60, seed = 1601).rd
            lvl = ["Banks", "Energy", "Software"]
            ind = [i == 3 ? "Utilities" : lvl[mod(i - 1, 3) + 1] for i in 1:size(d.X, 2)]
            po.panel_field(d.pnl, "market_cap").vals[from:end, ind .== "Energy"] .*= x
            return d
        end
        unseen_rows(st) = [s
                           for s in axes(st.W, 1)
                           if !isnothing(po.unseen_member_change(st.fcb,
                                                                 view(st.Ms, s, :, :),
                                                                 view(st.W, s, :) .> 0, s))]
        for (dl, de, mv) in ((energy_panel(150, 4.0), (0, 150, 190, 225, 226, 300), 2),
                             (energy_panel(229, 100.0), (0, 228, 260, 300), 2)),
            (unseen, pf) in ((ZeroUnseenMember(), GRID_PE),
                             (ZeroUnseenMember(),
                              EmpiricalPrior(; me = SimpleExpectedReturns(), ce = Covariance())),
                             (SolvedUnseenMember(), GRID_PE),
                             (SolvedUnseenMember(),
                              EmpiricalPrior(; me = SimpleExpectedReturns(), ce = Covariance())))

            pe = CrossSectionalFactorPrior(; factors = dlf,
                                           families = ["industry" => nothing], minra = 5,
                                           unseen = unseen, pe = pf)
            xs = stream(pe, dl, de)
            @test [only(dropped(x.pr)) for x in xs] ==
                  ["industry=Software"; fill("industry=Energy", length(xs) - 1)]
            @test xs[mv].pe.cache.csr.f isa SubArray
            for (k, x) in enumerate(xs)
                b = batch(pe, k, dl, de)
                @test dropped(x.pr) == dropped(b)
                @test near(x.pr, b)
            end
        end
        # Row 226 is the fitted observation 225, and the step before the move fits it. Under
        # `SolvedUnseenMember()` its answer depends on the dropped member (#1613).
        pe = CrossSectionalFactorPrior(; factors = dlf, families = ["industry" => nothing],
                                       minra = 5, unseen = SolvedUnseenMember())
        dl, de = energy_panel(229, 100.0), (0, 228, 260, 300)
        xs = stream(pe, dl, de)
        s1 = xs[1].pe.cache
        @test unseen_rows(s1) == [225]
        b = batch(pe, 2, dl, de)
        # The columns of the old answer that the new basis keeps are not the batch answer at
        # that row, and the row that the move solves again is.
        Tf = size(s1.Ms, 1)
        fr = po.cross_sectional_expand(s1.fcb, (pe.lag + 1):Tf, pe.lag, s1.csr.f)
        old = fr[225, po.retained_factor_indices(b.rr.fcb)]
        @test relerr(old, b.rr.csr.f[225, :]) > 1e-3
        @test relerr(xs[2].pr.rr.csr.f[225, :], b.rr.csr.f[225, :]) < 1e-12
        @test same(xs[2].pe.cache.csr.eps[1:size(s1.csr.eps, 1), :], s1.csr.eps)
        @test same(xs[2].pe.cache.W[1:size(s1.W, 1), :], s1.W)
        # A beta that shrinks to the mean of its industry is a function of the industry
        # columns where every industry shrinks fully, so the design of such an observation has
        # a dependent factor set under every rule. Its answer of least norm reads the basis, so
        # the move solves it again in the new basis (#1616). Without that, the factor returns
        # differed by 1 %. The rows that the scan solves again are the rows that it writes
        # over `NaN`. A zero column has a return of zero in every basis, so under
        # `ZeroUnseenMember()` the stream with no beta solves no row again, and under
        # `SolvedUnseenMember()` it solves the rows with an Unseen Member alone.
        function solved(pe, st)
            f = fill(NaN, size(st.csr.f))
            po.cross_sectional_move_solve(pe, st, (; fcb = st.fcb, f = f, lv = st.lv))
            return findall(r -> all(isfinite, r), collect(eachrow(f)))
        end
        beta = "beta" => CompositeExposure(;
                                           descriptors = [EWMarketBeta(; half_life = 10, agg_obs = 5,
                                                                       group = "industry")],
                                           family = "style")
        dl, de = energy_panel(150, 4.0), (0, 150, 190, 225, 226, 300)
        for unseen in (ZeroUnseenMember(), SolvedUnseenMember()), fs in (dlf, [dlf; beta])
            pe = CrossSectionalFactorPrior(; factors = fs,
                                           families = ["industry" => nothing], minra = 5,
                                           unseen = unseen, pe = GRID_PE)
            xs = stream(pe, dl, de)
            @test only(dropped(xs[2].pr)) == "industry=Energy"
            @test xs[2].pe.cache.csr.f isa SubArray
            @test all(k -> near(xs[k].pr, batch(pe, k, dl, de)), eachindex(xs))
            st = xs[end].pe.cache
            rs = solved(pe, st)
            if fs === dlf
                @test rs == (unseen === ZeroUnseenMember() ? Int[] : unseen_rows(st))
            else
                @test unseen === SolvedUnseenMember() || length(rs) == 131
                @test issubset(unseen === ZeroUnseenMember() ? Int[] : unseen_rows(st), rs)
            end
        end
        # A new dropped member with a zero benchmark-weighted exposure at an observation has
        # no finite ratio, so the rebase answers `nothing`, and the fit of every observation
        # refuses the member as the batch fit does.
        fcb = FactorFamilyBasis(; fnm = ["ind"], fi = [[1, 2, 3]], di = [1],
                                ratios = [0.5 0.0; 0.25 2.0], K = 3)
        @test isnothing(po.cross_sectional_rebase(fcb, ["ind" => "c"], ["a", "b", "c"]))
        rb = po.cross_sectional_rebase(fcb, ["ind" => "b"], ["a", "b", "c"])
        @test rb.di == [2] && rb.ratios ≈ [2.0 0.0; 4.0 8.0]
        @test po.cross_sectional_rebase(fcb, ["ind" => "a"], ["a", "b", "c"]).ratios ==
              fcb.ratios
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
        @test all(k -> near(sx[k].pr, batch(pe, k)), 1:3)
    end

    @testset "Parity with the oracle's online update" begin
        # The prior of each step against the stored block of that step. `mu` compares cell by
        # cell. `sigma` compares against its largest entry: an off-diagonal entry of two assets
        # near zero correlation is a cancellation in `L F L'` (#1376), measured maxrel 2.7e-12
        # cell by cell. It was 1.3e-10 when the lift repaired the systematic block `L F L'`, which the oracle does not repair (#1576). The factor returns compare cell by cell with an absolute floor: the
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
        # maxscaled 1.1e-15, factor returns maxrel 4.6e-14 on `Style`; mu maxrel 5.3e-13,
        # sigma maxscaled 1.2e-15, factor returns maxabs 1.5e-16 on `Industry`. The sigma
        # values were 4.2e-13 and 6.5e-13 when the lift repaired the systematic block `L F L'`, which the oracle does not repair (#1576).
        check(pinned(style), "OnlineStyle", "Fold")
        check(pinned(industry), "OnlineIndustry", "Fold")
        # Currency factors on the carry route (#1479). The oracle's online update reads the
        # Currency Excess Returns of each batch, and it pins `style1` as in `Style`. Measured
        # over the three steps: mu maxrel 2.3e-13, sigma maxscaled 5.6e-16 (4.2e-13 before
        # #1576), factor returns maxrel 9.3e-14.
        cp = pinned((; grid_config("Currency", rd)..., families = ["style" => nothing]))
        @test all(x -> x.pe.cache.families == ["style" => "style1"], cp)
        check(cp, "OnlineCurrencyStyle", "Fold")
        # A seed window on the factor prior cuts the factor returns of the first fit to the
        # last 60, and every later step folds every row, as the oracle's does. Measured: mu
        # maxrel 7.2e-15, sigma maxscaled 1.5e-15 (3.7e-13 before #1576), factor returns maxrel
        # 4.6e-14 on `Style`; mu maxrel 1.4e-13, sigma maxscaled 6.1e-16 (7.3e-13 before
        # #1576), factor returns maxabs 1.5e-16 on `Industry`.
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
            @test near(bo.pe, batch(pe, k))
            @test isequal(brd.X, rd.X[1:edges[k + 1], :])
        end
        @test optimise(o).w == optimise(InverseVolatility(; pe = pe), rd).w
        # `partial_fit` copies the state, so the earlier prior answers as it did.
        e1 = partial_fit!(pe, rows(rd, 1:90))
        p1 = prior(e1)
        e2 = partial_fit(e1, rows(rd, 91:170))
        @test agrees(prior(e1), p1) && near(prior(e2), batch(pe, 2))
    end

    @testset "A step appends to the histories in place (#1564)" begin
        # A step writes its rows into the spare rows of the backing of each history, and each
        # state keeps the view of its own rows.
        pe = CrossSectionalFactorPrior(; lambda = 1, style...)
        e = (0, 90, 91, 92, 93)
        s = stream(pe, rd, e)
        st2, st4 = s[2].pe.cache, s[4].pe.cache
        @test st4.Ms isa SubArray && parent(st4.Ms) === parent(st2.Ms)
        @test parent(st4.csr.f) === parent(st2.csr.f) && parent(st4.vs) === parent(st2.vs)
        @test size(st2.Ms, 1) + 2 == size(st4.Ms, 1)
        # An earlier estimator reads its own rows after a later step. The factor prior folds
        # in place, so its moments read the later state, and the rows are compared alone.
        function rows_agree(x, b)
            return same(x.rr.csr.f, b.rr.csr.f) &&
                   same(x.rr.vs, b.rr.vs) &&
                   same(x.rr.Ms, b.rr.Ms) &&
                   same(x.rr.rw, b.rr.rw) &&
                   x.rr.idx == b.rr.idx
        end
        @test all(k -> rows_agree(prior(s[k].pe), s[k].pr), 1:4)
        # A second step of an earlier state copies its histories, so the later state and the
        # Result read out of it keep their rows.
        H = deepcopy((st4.Ms, st4.X, st4.csr.f, st4.vs, st4.W))
        rdz = deepcopy(rd)
        rdz.X[92:93, :] .*= 1.5
        b = partial_fit!(s[2].pe, rows(rdz, 92:93))
        @test parent(b.cache.Ms) !== parent(st4.Ms)
        @test isequal(H, (st4.Ms, st4.X, st4.csr.f, st4.vs, st4.W))
        @test agrees(s[4].pr, batch(pe, 4, rd, e))
    end

    @testset "A step joins the observed exposures of its new rows alone (#1589)" begin
        # Under observed factors the exposure history and the observed exposures are views of
        # one backing, the estimated factors first. A step appends its joined rows in place, and
        # the call with no data reads the joined history as a view of the backing.
        pe = CrossSectionalFactorPrior(; lambda = 1, grid_config("Currency", rd)...)
        e = (0, 90, 91, 92, 93)
        s = stream(pe, rd, e)
        st2, st4 = s[2].pe.cache, s[4].pe.cache
        P = parent(st4.Ms)
        @test P === parent(st4.obs.Z) && P === parent(st2.Ms)
        @test size(P, 3) == size(st4.Ms, 3) + size(st4.obs.Z, 3)
        @test parent(s[4].pr.rr.Ms) === P
        @test all(k -> agrees(s[k].pr, batch(pe, k, rd, e)), 1:4)
        # A copy keeps the two histories in one backing, and shares no array with the state.
        c = copy(st4)
        @test parent(c.Ms) === parent(c.obs.Z) && parent(c.Ms) !== P
        # A second step of an earlier state copies its histories, so the later Result keeps
        # its rows.
        H = deepcopy(s[4].pr.rr.Ms)
        rdz = deepcopy(rd)
        rdz.X[92:93, :] .*= 1.5
        b = partial_fit!(s[2].pe, rows(rdz, 92:93))
        @test parent(b.cache.Ms) === parent(b.cache.obs.Z) && parent(b.cache.Ms) !== P
        @test isequal(H, s[4].pr.rr.Ms)
        @test agrees(s[4].pr, batch(pe, 4, rd, e))
    end

    @testset "An observed member computes the exposures of its new rows alone (#1593)" begin
        # An `ObservedExposure` folds the member that it wraps, as an estimated member does, so
        # a Descriptor that carries a state reads the new rows alone, and the carry keeps the
        # rows of the lag alone. A Descriptor that carries no state computes the new rows from
        # the rows that it reads. The derivation of the net returns of a row reads the observed
        # exposures of the row `lag` observations before it, so the state carries those of the
        # last `lag` rows.
        ccy = grid_config("Currency", rd)
        function observed(d, nm)
            return nm => ObservedExposure(;
                                          xe = CompositeExposure(; descriptors = [d],
                                                                 outlier = nothing,
                                                                 scoring = nothing, family = nm),
                                          series = "MACRO", family = nm)
        end
        e = (0, 90, 91, 92, 93, 250)
        for (d, L) in ((Reversal(), 1), (EWMomentum(; half_life = 10, skip = 5), 1),
                       (GrowthRate(; field = "sales_ttm", lag = 5), 1),
                       (RollingMax(; window = 6), 6)), lag in (1, 2)

            pe = CrossSectionalFactorPrior(; lambda = 1, ccy...,
                                           factors = [ccy.factors; observed(d, "obs")],
                                           lag = lag)
            @test po.cross_sectional_carry_rows(pe) == L + lag
            s = stream(pe, rd, e)
            for (k, x) in enumerate(s)
                b = batch(pe, k, rd, e)
                @test relerr(x.pr.mu, b.mu) < 1e-12 &&
                      relerr(x.pr.sigma, b.sigma) < 1e-12 &&
                      relerr(x.pr.fpr.X, b.fpr.X) < 1e-12
                @test isequal(isnan.(x.pr.sigma), isnan.(b.sigma))
            end
            st = last(s).pe.cache
            @test size(st.win.X, 1) == L + lag
            @test size(st.der.Zo) == (lag, N, size(st.obs.Z, 3))
            @test isequal(st.der.Zo, st.obs.Z[(end - lag + 1):end, :, :])
            # The state keeps the folded member at its place in the factor list.
            @test [first(p) for p in st.xf] == [first(p) for p in pe.factors]
            if L == 1
                @test !isnothing(last(st.xf[end]).xe.descriptors[1].cache)
            end
        end
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
                 Entropy = (; ccy..., pe = EntropyPoolingPrior(; pe = GRID_PE)),
                 Rolling = (; ccy..., ve = wv))
        # Measured: every case gives a difference of exactly zero at each step.
        for cfg in exact
            pe = CrossSectionalFactorPrior(; lambda = 1, cfg...)
            @test po.reads_exogenous_series(pe)
            @test all(((k, x),) -> agrees(x.pr, batch(pe, k)), enumerate(stream(pe)))
        end
        # The `Reversal` exposure of the observed member folds from its carried state, so the
        # carry keeps the rows of the lag alone (#1593). A fixed weighted forecast scores the net
        # returns of the rows it reads, and its `Reversal` score folds too (#1587).
        @test po.cross_sectional_carry_rows(CrossSectionalFactorPrior(; lambda = 1,
                                                                      mixed...)) == 2
        for cfg in (mixed, (; mixed..., lag = 2), (; mixed..., rfe = rfr),
                    (; mixed..., rfe = rfr, lag = 2))
            pe = CrossSectionalFactorPrior(; lambda = 1, cfg...)
            for (k, x) in enumerate(stream(pe))
                b = batch(pe, k)
                @test relerr(x.pr.mu, b.mu) < 1e-12 &&
                      relerr(x.pr.sigma, b.sigma) < 1e-12 &&
                      relerr(x.pr.fpr.X, b.fpr.X) < 1e-12
                @test isequal(isnan.(x.pr.sigma), isnan.(b.sigma))
                @test isnothing(pe.rfe) || relerr(x.pr.rr.rf.hist, b.rr.rf.hist) < 1e-12
            end
        end
        # A constrained Family reduces the factor axis, and the observed factor returns fill its
        # trailing columns. So the carry of a forecast that folds, and of the history of one
        # that reads the panel, builds the model on the reduced axis, as the batch fit does
        # (#1585). Before, the seed threw a `DimensionMismatch`.
        famccy = [grid_config("FamOne", rd).factors; "ccy" => CurrencyExposure()]
        fold = (; grid_config("FcFamily", rd)..., factors = famccy)
        pe = CrossSectionalFactorPrior(; fold...)
        @test all(((k, x),) -> agrees(x.pr, batch(pe, k)), enumerate(stream(pe)))
        panel = (; grid_config("FcTarget", rd)..., factors = famccy,
                 families = ["industry" => nothing])
        pc = CrossSectionalFactorPrior(; panel..., lambda = CarryHistoryShrinkage())
        pb = CrossSectionalFactorPrior(; panel..., lambda = CarryHistoryShrinkage())
        @test all(((k, x),) -> agrees(x.pr, batch(pb, k)), enumerate(stream(pc)))
        # A macro sensitivity is a recursion from the first row, and it folds from its state, so
        # the carry keeps the rows of the lag alone (#1608).
        @test po.cross_sectional_carry_rows(CrossSectionalFactorPrior(; lambda = 1,
                                                                      exact.Sensitivity...)) ==
              2
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

    @testset "A fixed weighted Return Forecast folds its rows (#1573)" begin
        # A row of the history reads the scores and the block of its observation alone, so the
        # state carries the scores and the rows, and keeps the panel rows of the look-back.
        # The Reversal and the GrowthRate scores fold from their carried states (#1587,
        # #1598), so the carry keeps the rows of the lag of the prior alone.
        sharpe = IdiosyncraticSharpeUnit()
        ind = grid_config("FamOne", rd).factors
        scores(nm) = DescriptorScores(;
                                      descriptors = [Passthrough(;
                                                                 field = "net_income_ttm"),
                                                     Reversal(; window = 30),
                                                     GrowthRate(; field = "sales_ttm",
                                                                lag = 5)], neutralise = nm,
                                      nw = BlockRegressionWeights(), group = "industry")
        fw(nm, unit) = FixedWeightedReturnForecast(; scores = scores(nm), scale = 0.2,
                                                   weights = [1.0, -0.5, 0.25], unit = unit)
        fk = (; lambda = 0.4, c = 0.6, minra = 5, pe = GRID_PE, ve = GRID_VE)
        e = (0, 90, 91, 92, 93, 170, 250)
        wv = WindowedVariance(;
                              ve = ExpWeightedVariance(; decay = 2.0^(-1 / 20),
                                                       min_obs = 5), window = 60)
        # The batch choice of `style` moves at the second step, and a rolling window does not
        # fold, so both fit every observation again from the scores that the state carries.
        for (cfg, ee) in (((; fk..., factors = ind, families = ["industry" => nothing],
                            rfe = fw(["industry", "style1"], sharpe)), e),
                          ((; fk..., factors = ind, lag = 2,
                            rfe = fw(["industry", "style1"], IdiosyncraticReturnUnit())), e),
                          ((; style..., fk..., rfe = fw(["style1"], sharpe)), edges),
                          ((; style..., fk..., ve = wv, rfe = fw(["style1"], sharpe)), edges))
            pe = CrossSectionalFactorPrior(; cfg...)
            @test po.lookback(pe) == 30 && po.cross_sectional_carry_rows(pe) == pe.lag + 1
            for (k, x) in enumerate(stream(pe, rd, ee))
                b = batch(pe, k, rd, ee)
                st = x.pe.cache
                @test size(x.pe.cache.win.X, 1) == pe.lag + 1
                @test size(st.fh) == size(st.vs) == size(b.rr.rf.hist)
                @test size(st.fsc.S, 1) == size(st.Ms, 1)
                @test relerr(x.pr.mu, b.mu) < 1e-14 &&
                      relerr(x.pr.sigma, b.sigma) < 1e-14 &&
                      relerr(x.pr.rr.rf.hist, b.rr.rf.hist) < 1e-14 &&
                      x.pr.rr.rf.weights == b.rr.rf.weights
                @test isequal(isnan.(x.pr.rr.rf.hist), isnan.(b.rr.rf.hist))
            end
        end
    end

    @testset "An exponentially weighted Return Forecast folds its rows (#1574)" begin
        # The batch fit is a forward recursion over the rows whose target is known. A step
        # reads its new rows and the `lag + horizon - 1` rows before them, whose targets
        # mature at the step, from the carried state of the recursion. The Reversal and the
        # GrowthRate scores fold from their carried states (#1587, #1598), so the carry keeps
        # the rows of the lag of the prior alone.
        sharpe = IdiosyncraticSharpeUnit()
        ind = grid_config("FamOne", rd).factors
        scores(nm) = DescriptorScores(;
                                      descriptors = [Passthrough(;
                                                                 field = "net_income_ttm"),
                                                     Reversal(; window = 30),
                                                     GrowthRate(; field = "sales_ttm",
                                                                lag = 5)], neutralise = nm,
                                      nw = BlockRegressionWeights(), group = "industry")
        ew(nm, unit; kw...) = ExpWeightedReturnForecast(; scores = scores(nm),
                                                        half_life = 10, scale = 0.5,
                                                        unit = unit, kw...)
        fk = (; lambda = 0.4, c = 0.6, minra = 5, pe = GRID_PE, ve = GRID_VE)
        e = (0, 90, 91, 92, 93, 170, 250)
        # The first fit holds two rows, so the first steps carry fewer rows than the gap of
        # four. Passthrough scores and a short variance warm-up let the recursion advance in
        # them. The factor prior reads too few rows until the last step, so only the last
        # step reads out, and it reads the rows and the state of the early steps.
        e0 = (0, 3, 4, 5, 6, 7, 8, 120)
        early = (; grid_config("Base", rd)..., fk..., ofit = UnadjustedForecast(),
                 ve = ExpWeightedVariance(; decay = 2.0^(-1 / 20), min_obs = 2),
                 rfe = ExpWeightedReturnForecast(;
                                                 scores = DescriptorScores(;
                                                                           descriptors = [Passthrough(;
                                                                                                      field = "net_income_ttm"),
                                                                                          Passthrough(;
                                                                                                      field = "sales_ttm")]),
                                                 half_life = 10, horizon = 2, lag = 3))
        st = partial_fit!(CrossSectionalFactorPrior(; early...), rows(rd, 1:3))
        st = partial_fit!(partial_fit!(st, rows(rd, 4:4)), rows(rd, 5:8)).cache
        @test size(st.fh, 1) == 7 && st.fst.n == 2
        wv = WindowedVariance(;
                              ve = ExpWeightedVariance(; decay = 2.0^(-1 / 20),
                                                       min_obs = 5), window = 60)
        # The batch choice of `style` moves at the second step, and a rolling window does not
        # fold, so both fit every observation again from an empty state of the recursion.
        cases = (((; fk..., factors = ind, families = ["industry" => nothing],
                   rfe = ew(["industry", "style1"], sharpe)), e),
                 ((; fk..., factors = ind, lag = 2,
                   rfe = ew(["industry", "style1"], IdiosyncraticReturnUnit(); horizon = 3,
                            lag = 2, min_obs = 20, normalise = false)), e),
                 ((; style..., fk..., ofit = UnadjustedForecast(),
                   rfe = ew(nothing, sharpe; horizon = 2, lag = 3)), e), (early, e0),
                 ((; style..., fk..., ofit = ScoreNeutralisation(),
                   rfe = ew(nothing, IdiosyncraticReturnUnit())), edges),
                 ((; style..., fk..., ve = wv, rfe = ew(["style1"], sharpe; horizon = 2)),
                  edges))
        for (cfg, ee) in cases
            pe = CrossSectionalFactorPrior(; cfg...)
            @test po.lookback(pe) == (cfg === early ? 2 : 30)
            @test po.cross_sectional_carry_rows(pe) == pe.lag + 1
            for (k, x) in enumerate(stream(pe, rd, ee))
                # A block too short for a fit refuses on both sides.
                b = try
                    batch(pe, k, rd, ee)
                catch err
                    err
                end
                st = x.pe.cache
                if b isa Exception
                    @test x.pr isa Exception
                    continue
                end
                rf, rb = x.pr.rr.rf, b.rr.rf
                @test size(st.win.X, 1) <= 6
                @test size(st.fh) == size(st.vs) == size(rb.hist)
                @test relerr(x.pr.mu, b.mu) < 1e-13 &&
                      relerr(x.pr.sigma, b.sigma) < 1e-13 &&
                      relerr(rf.hist, rb.hist) < 1e-13
                @test isequal(isnan.(rf.hist), isnan.(rb.hist))
                # The state of the recursion equals the one of the batch fit.
                @test rf.n == rb.n == st.fst.n
                @test relerr(rf.A, rb.A) < 1e-13 && relerr(rf.c, rb.c) < 1e-13
                @test isequal(isnan.(rf.coef), isnan.(rb.coef)) &&
                      relerr(rf.coef, rb.coef) < 1e-12
            end
        end
        # A step copies the carried state, so the state of an earlier step stays as it is.
        pe = CrossSectionalFactorPrior(; first(first(cases))...)
        pe = partial_fit!(partial_fit!(pe, rows(rd, 1:90)), rows(rd, 91:91))
        A0 = copy(pe.cache.fst.A)
        st0 = pe.cache
        partial_fit!(pe, rows(rd, 92:92))
        @test st0.fst.A == A0
    end

    @testset "A Target Return Forecast folds one row from its prequential state (#1581)" begin
        # Under the prequential rule with a LinearModel of no keyword argument, the batch fit
        # and a step run one fold, `return_forecast_step`. The state carries the normal
        # equations, the two calibration regressions and the coefficients of the rows whose
        # target has not matured, so the carry keeps the panel rows of the look-back alone.
        # The Reversal and the GrowthRate scores fold from their carried states (#1587,
        # #1598), so the carry keeps the rows of the lag of the prior alone.
        sharpe = IdiosyncraticSharpeUnit()
        ind = grid_config("FamOne", rd).factors
        ds = DescriptorScores(;
                              descriptors = [Passthrough(; field = "net_income_ttm"),
                                             Reversal(; window = 30),
                                             GrowthRate(; field = "sales_ttm", lag = 5)],
                              group = "industry")
        tf(; kw...) = TargetReturnForecast(; scores = ds, half_life = 10.0, kw...)
        fk = (; lambda = 0.4, c = 0.6, minra = 5, pe = GRID_PE, ve = GRID_VE)
        wv = WindowedVariance(;
                              ve = ExpWeightedVariance(; decay = 2.0^(-1 / 20),
                                                       min_obs = 5), window = 60)
        coef(rf) = isnothing(rf.model) ? Float64[] : po.StatsAPI.coef(rf.model)
        function slope(a, b)
            return isequal(a, b) || isapprox(a, b; rtol = 1e-12)
        end
        # The carry equals the batch fit at each step, and returns the same refusal.
        function check(pe, ee, r = rd)
            xs = stream(pe, r, ee)
            for (k, x) in enumerate(xs)
                b = try
                    batch(pe, k, r, ee)
                catch err
                    err
                end
                if !isa(b, LowOrderPrior)
                    @test typeof(x.pr) == typeof(b)
                    continue
                end
                rf, rb = x.pr.rr.rf, b.rr.rf
                @test relerr(x.pr.mu, b.mu) < 1e-13 && relerr(x.pr.sigma, b.sigma) < 1e-13
                @test relerr(rf.mu, rb.mu) < 1e-13 && isequal(isnan.(rf.mu), isnan.(rb.mu))
                @test relerr(coef(rf), coef(rb)) < 1e-12
                @test slope(rf.calib, rb.calib)
                @test isnothing(rf.ocalib) == isnothing(rb.ocalib)
                @test isnothing(rf.ocalib) || slope(rf.ocalib, rb.ocalib)
                st = x.pe.cache
                @test size(st.fh, 1) == size(st.vs, 1)
                @test size(st.fst.B, 1) <= po.forecast_target_gap(pe.rfe)
            end
            return xs
        end
        e = (0, 90, 91, 92, 93, 170, 250)
        # The default, a Sharpe unit with an intercept, the calibration of the orthogonal
        # part, a member that reads the block alone with a gap of three rows, a batch choice
        # that moves at the second step, and a rolling variance that refits every step. The
        # last two fit every observation again from the scores that the state carries.
        for cfg in
            ((; fk..., factors = ind, families = ["industry" => nothing], rfe = tf()),
             (; fk..., factors = ind, rfe = tf(; unit = sharpe, intercept = true)),
             (; fk..., factors = ind, ofit = OrthogonalPartCalibration(), rfe = tf()),
             (; fk..., factors = ind, lag = 2, ofit = UnadjustedForecast(),
              rfe = tf(; whole_history = false, horizon = 2, lag = 2)),
             (; style..., fk..., rfe = tf()), (; style..., fk..., ve = wv, rfe = tf()))
            pe = CrossSectionalFactorPrior(; cfg...)
            @test po.folds_forecast_rows(pe.rfe)
            @test po.lookback(pe) == 30 && po.cross_sectional_carry_rows(pe) == pe.lag + 1
            xs = check(pe, e)
            @test all(x -> size(x.pe.cache.win.X, 1) == pe.lag + 1, xs)
        end
        # Under `whole_history` a row before the block trains the fit when its forward window
        # reaches into the block. With a gap of four rows the first row before the block
        # matures at the fourth block row, after the first fit, so a step reads it again.
        pas = DescriptorScores(;
                               descriptors = [Passthrough(; field = "net_income_ttm"),
                                              Passthrough(; field = "sales_ttm")])
        pe = CrossSectionalFactorPrior(; style..., fk...,
                                       rfe = TargetReturnForecast(; scores = pas, lag = 4,
                                                                  half_life = 10.0))
        @test po.cross_sectional_carry_rows(pe) == 2
        check(pe, (0, 3, 4, 5, 6, 7, 8, 12, 40))
        # A horizon above one under `whole_history` folds too (#1588), and the carry keeps the
        # rows of the look-back alone.
        h2 = CrossSectionalFactorPrior(; fk..., factors = ind, rfe = tf(; horizon = 2))
        @test po.folds_forecast_rows(h2.rfe)
        @test po.cross_sectional_carry_rows(h2) == 2
        # A slot that reads the history reads the rows of the fold: the row of an
        # observation is the forecast of a fit through that observation.
        rl, bl = CarryHistoryShrinkage(), CarryHistoryShrinkage()
        cfg = (; fk..., factors = ind, rfe = tf())
        pc = CrossSectionalFactorPrior(; cfg..., lambda = rl)
        pb = CrossSectionalFactorPrior(; cfg..., lambda = bl)
        for (k, x) in enumerate(stream(pc, rd, e))
            b = batch(pb, k, rd, e)
            @test relerr(rl.seen[k], bl.seen[k]) < 1e-13
            @test isequal(isnan.(rl.seen[k]), isnan.(bl.seen[k]))
            @test isapprox(x.pr.rr.lambda, b.rr.lambda; rtol = 1e-12)
            @test size(x.pe.cache.hist, 1) == size(x.pe.cache.fh, 1) - 1
        end
    end

    @testset "A RollingLogReturn folds one row from its carried state (#1583)" begin
        # The state takes the cumulative sums from the first observation with the arithmetic
        # of the batch call, so each folded row equals the batch call to the last bit, in
        # blocks of any size. A second step from the same state gives the same rows, because
        # a step copies the buffers of the state and changes no carried row.
        syn = synthetic_asset_panel(; n_assets = 60, n_observations = 160, n_industries = 4,
                                    rng = StableRNG(1583)).rd
        blocks = ((1, 1), (2, 30), (31, 31), (32, 100), (101, 160))
        for de in (RollingMomentum(; window = 40, skip = 5), Reversal(),
                   RollingLogReturn(; window = 10, sign = -1, exponentiate = true))
            full = descriptor(de, syn)
            st = de
            for (a, b) in blocks
                r = po.descriptor_step(st, rows(syn, a:b))
                @test same(r.D, full[a:b, :])
                @test same(po.descriptor_step(st, rows(syn, a:b)).D, r.D)
                @test same(partial_fit!(st, rows(syn, a:b)).cache.cs[end],
                           r.de.cache.cs[end])
                st = r.de
            end
            @test length(st.cache.cs) == de.window + de.skip + 1
            c = copy(st.cache)
            @test all(i -> c.cs[i] == st.cache.cs[i] && c.cs[i] !== st.cache.cs[i],
                      eachindex(c.cs))
            @test occursin("cannot merge two states fitted on disjoint blocks",
                           message(() -> po.merge_states(c, st.cache)))
            @test occursin("carries 60 assets, and the step brings 5",
                           message(() -> po.descriptor_step(st,
                                                            po.port_opt_view(rows(syn, 1:2),
                                                                             1:5))))
        end
        # On the carry fold the Descriptor counts as one row, so the carry keeps the rows of
        # the lag alone, and the exposures of every step equal those of the batch fit.
        momentum = ["market" => ConstantExposure(),
                    "momentum" => CompositeExposure(;
                                                    descriptors = [RollingMomentum(; window = 40,
                                                                                   skip = 5),
                                                                   BookToPrice()]),
                    "industry" => OneHotExposure(; field = "industry", family = "industry")]
        pe = CrossSectionalFactorPrior(; lambda = 1, factors = momentum,
                                       families = ["industry" => nothing], minra = 5,
                                       pe = GRID_PE, ve = GRID_VE)
        @test po.lookback(pe) == 46
        @test po.cross_sectional_carry_rows(pe) == 2
        e = (0, 50, 80, 81, 120, 121, 160)
        s = stream(pe, syn, e)
        fitted = 0
        for (k, x) in enumerate(s)
            b = try
                batch(pe, k, syn, e)
            catch err
                err
            end
            if isa(b, LowOrderPrior)
                fitted += 1
                @test same(x.pr.rr.Ms, b.rr.Ms)
                @test same(x.pr.mu, b.mu) && same(x.pr.sigma, b.sigma)
            else
                @test typeof(x.pr) == typeof(b)
            end
        end
        # The case reads out at least once, so the loop is not vacuous.
        @test fitted >= 3
        @test size(last(s).pe.cache.win.X, 1) == 2
        @test isa(last(last(s).pe.cache.xf[2]).descriptors[1].cache,
                  po.RollingLogReturnState)
    end
    @testset "An EW mean Descriptor folds one row from its carried state (#1586)" begin
        # The state runs the recursion from the first observation with the arithmetic of the
        # batch call, so each folded row equals the batch call to the last bit, in blocks of
        # any size. A skip reads the log returns of the ring of the state. A second step from
        # the same state gives the same rows, because a step copies the state.
        syn = synthetic_asset_panel(; n_assets = 60, n_observations = 160, n_industries = 4,
                                    rng = StableRNG(1586)).rd
        blocks = ((1, 1), (2, 30), (31, 31), (32, 100), (101, 160))
        for de in (EWMomentum(; half_life = 10, skip = 5),
                   EWMean(; decay = 0.9, min_obs = 3, exponentiate = true),
                   EWShareTurnover(; half_life = 5), EWAmihudIlliquidity(; half_life = 7),
                   DaysToCover(; half_life = 4),
                   EWVolumeRatio(; num = nothing, den = "adj_close", decay = 0.7, min_obs = 2))
            full = descriptor(de, syn)
            st = de
            for (a, b) in blocks
                r = po.descriptor_step(st, rows(syn, a:b))
                @test same(r.D, full[a:b, :])
                @test same(po.descriptor_step(st, rows(syn, a:b)).D, r.D)
                @test same(partial_fit!(st, rows(syn, a:b)).cache.s, r.de.cache.s)
                st = r.de
            end
            @test count(isfinite, full) > 8000
            c = copy(st.cache)
            @test c.s == st.cache.s && c.s !== st.cache.s && c.n == st.cache.n
            @test occursin("cannot merge two states fitted on disjoint blocks",
                           message(() -> po.merge_states(c, st.cache)))
            @test occursin("carries 60 assets, and the step brings 5",
                           message(() -> po.descriptor_step(st,
                                                            po.port_opt_view(rows(syn, 1:2),
                                                                             1:5))))
        end
        @test length(partial_fit!(EWMomentum(; skip = 5), syn).cache.buf) == 6
        # The state of a ratio holds no ring, so an EWMean refuses it.
        ratio = partial_fit!(DaysToCover(; half_life = 4), rows(syn, 1:5)).cache
        @test occursin("holds no ring of log returns",
                       message(() -> po.descriptor_step(EWMean(; decay = 0.5, min_obs = 1,
                                                               cache = ratio),
                                                        rows(syn, 6:7))))
        @test occursin("hold one entry per asset, got 2 and 1",
                       message(() -> po.EWMeanState(; s = [0.0, 0.0], n = [0])))
        @test po.show_fields(EWMean(; decay = 0.5, min_obs = 1)) ==
              (:decay, :min_obs, :skip, :exponentiate)
        # On the carry fold each Descriptor counts as one row, so the carry keeps the rows of
        # the lag alone, and the exposures of every step equal those of the batch fit.
        ew = ["market" => ConstantExposure(),
              "momentum" => CompositeExposure(;
                                              descriptors = [EWMomentum(; half_life = 10, skip = 5),
                                                             BookToPrice()]),
              "liquidity" => CompositeExposure(;
                                               descriptors = [EWShareTurnover(; half_life = 5),
                                                              DaysToCover(; half_life = 4)]),
              "industry" => OneHotExposure(; field = "industry", family = "industry")]
        pe = CrossSectionalFactorPrior(; lambda = 1, factors = ew,
                                       families = ["industry" => nothing], minra = 5,
                                       pe = GRID_PE, ve = GRID_VE)
        @test isnothing(po.lookback(pe))
        @test po.cross_sectional_carry_rows(pe) == 2
        e = (0, 50, 80, 81, 120, 121, 160)
        s = stream(pe, syn, e)
        fitted = 0
        for (k, x) in enumerate(s)
            b = try
                batch(pe, k, syn, e)
            catch err
                err
            end
            if isa(b, LowOrderPrior)
                fitted += 1
                @test same(x.pr.rr.Ms, b.rr.Ms)
                @test same(x.pr.mu, b.mu) && same(x.pr.sigma, b.sigma)
            else
                @test typeof(x.pr) == typeof(b)
            end
        end
        # The case reads out at least once, so the loop is not vacuous.
        @test fitted >= 3
        @test size(last(s).pe.cache.win.X, 1) == 2
        @test all(d -> isa(d.cache, po.EWMeanState),
                  last(last(s).pe.cache.xf[3]).descriptors)
    end

    @testset "A lag Descriptor folds one row from its carried state (#1598)" begin
        # The state carries the lagged quantity of the last `lag` observations with the bits
        # of the batch call, so each folded row equals the batch call to the last bit, in
        # blocks of any size, also blocks shorter than the lag. A second step from the same
        # state gives the same rows, because a step copies the state.
        syn = synthetic_asset_panel(; n_assets = 60, n_observations = 160, n_industries = 4,
                                    rng = StableRNG(1598)).rd
        blocks = ((1, 1), (2, 3), (4, 30), (31, 31), (32, 100), (101, 160))
        for de in (GrowthRate(; field = "sales_ttm", lag = 7), SalesGrowthRate(; lag = 1),
                   ChangeToScale(; field = "net_income_ttm", scale = "market_cap", lag = 5),
                   EarningsChangeToPrice(; lag = 40),
                   ChangeInIntensity(; field = "capex_ttm", scale = "total_assets", lag = 9),
                   CapexToAssetsChangeInIntensity(; lag = 3))
            full = descriptor(de, syn)
            st = de
            for (a, b) in blocks
                r = po.descriptor_step(st, rows(syn, a:b))
                @test same(r.D, full[a:b, :])
                @test same(po.descriptor_step(st, rows(syn, a:b)).D, r.D)
                @test same(partial_fit!(st, rows(syn, a:b)).cache.q[end], r.de.cache.q[end])
                st = r.de
            end
            @test count(isfinite, full) > 6000
            @test length(st.cache.q) == de.lag
            # A merge of the states of two consecutive blocks is the state of both blocks.
            a = partial_fit!(de, rows(syn, 1:50)).cache
            m = po.merge_states(a, partial_fit!(de, rows(syn, 51:52)).cache)
            @test all(((x, y),) -> same(x, y),
                      zip(m.q, partial_fit!(de, rows(syn, 1:52)).cache.q))
            c = copy(a)
            @test same(c.q[end], a.q[end]) && c.q[end] !== a.q[end]
            @test occursin("carries 60 assets, and the step brings 5",
                           message(() -> po.descriptor_step(st,
                                                            po.port_opt_view(rows(syn, 1:2),
                                                                             1:5))))
        end
        g = partial_fit!(GrowthRate(; field = "sales_ttm", lag = 4), rows(syn, 1:9))
        @test occursin("carries the rows of a lag of 4, and the lag of the Descriptor is 3",
                       message(() -> po.descriptor_step(GrowthRate(; field = "sales_ttm",
                                                                   lag = 3,
                                                                   cache = g.cache),
                                                        rows(syn, 10:11))))
        @test occursin("got the lags 4 and 3",
                       message(() -> po.merge_states(g.cache,
                                                     partial_fit!(GrowthRate(;
                                                                             field = "sales_ttm",
                                                                             lag = 3),
                                                                  rows(syn, 1:2)).cache)))
        q = po.DataStructures.CircularBuffer{Vector{Float64}}(2)
        push!(q, [0.0, 0.0])
        push!(q, [0.0])
        @test occursin("got rows of [2, 1] assets",
                       message(() -> po.LagDescriptorState(; q = q)))
        @test po.show_fields(EarningsChangeToPrice()) == (:field, :scale, :lag, :gt0)
        # On the carry fold each lag Descriptor counts as one row, so the carry keeps the
        # rows of the lag of the prior alone, and every step equals the batch fit.
        lagf = ["market" => ConstantExposure(),
                "growth" => CompositeExposure(;
                                              descriptors = [SalesGrowthRate(; lag = 20),
                                                             EarningsChangeToPrice(; lag = 15)]),
                "intensity" => CompositeExposure(;
                                                 descriptors = [CapexToAssetsChangeInIntensity(; lag = 25),
                                                                BookToPrice()]),
                "industry" => OneHotExposure(; field = "industry", family = "industry")]
        pe = CrossSectionalFactorPrior(; lambda = 1, factors = lagf,
                                       families = ["industry" => nothing], minra = 5,
                                       pe = GRID_PE, ve = GRID_VE)
        @test po.lookback(pe) == 27
        @test po.cross_sectional_carry_rows(pe) == 2
        e = (0, 50, 80, 81, 120, 121, 160)
        s = stream(pe, syn, e)
        fitted = 0
        for (k, x) in enumerate(s)
            b = try
                batch(pe, k, syn, e)
            catch err
                err
            end
            if isa(b, LowOrderPrior)
                fitted += 1
                @test same(x.pr.rr.Ms, b.rr.Ms)
                @test same(x.pr.mu, b.mu) && same(x.pr.sigma, b.sigma)
            else
                @test typeof(x.pr) == typeof(b)
            end
        end
        # The case reads out at least once, so the loop is not vacuous.
        @test fitted >= 3
        @test size(last(s).pe.cache.win.X, 1) == 2
        @test all(d -> isa(d.cache, po.LagDescriptorState),
                  last(last(s).pe.cache.xf[2]).descriptors)
        @test isa(first(last(last(s).pe.cache.xf[3]).descriptors).cache,
                  po.LagDescriptorState)
    end

    @testset "An EW volatility Descriptor folds one row from its carried state (#1607)" begin
        # The state carries the state of the variance estimator and, for a residual form, the
        # state of the beta recursion. Both run the arithmetic of the batch call from the first
        # observation, so each folded row equals the batch call to the last bit, in blocks of
        # any size, also with a regime adjustment and HAC lags. A second step from the same
        # state gives the same rows, because a step copies the state.
        syn = synthetic_asset_panel(; n_assets = 60, n_observations = 160, n_industries = 4,
                                    rng = StableRNG(1607)).rd
        blocks = ((1, 1), (2, 30), (31, 31), (32, 100), (101, 160))
        for de in (EWVolatility(; half_life = 10),
                   EWDownsideVolatility(; half_life = 5, mar = 0.001),
                   EWResidualVolatility(; half_life = 5, beta_half_life = 8),
                   EWResidualDownsideVolatility(; half_life = 7, beta_half_life = 3),
                   EWVolatility(;
                                ce = RegimeAdjustedExpWeightedVariance(; decay = 0.9, min_obs = 3,
                                                                       regime_min_obs = 4)),
                   EWResidualVolatility(;
                                        ce = RegimeAdjustedExpWeightedVariance(; decay = 0.8,
                                                                               min_obs = 2,
                                                                               hac_lags = 2)))
            full = descriptor(de, syn)
            st = de
            for (a, b) in blocks
                r = po.descriptor_step(st, rows(syn, a:b))
                @test same(r.D, full[a:b, :])
                @test same(po.descriptor_step(st, rows(syn, a:b)).D, r.D)
                @test same(partial_fit!(st, rows(syn, a:b)).cache.ve.variance,
                           r.de.cache.ve.variance)
                st = r.de
            end
            @test count(isfinite, full) > 8000
            c = copy(st.cache)
            @test c.ve.variance == st.cache.ve.variance &&
                  c.ve.variance !== st.cache.ve.variance
            @test isnothing(c.beta) == isnothing(st.cache.beta)
            @test occursin("cannot merge two states fitted on disjoint blocks",
                           message(() -> po.merge_states(c, st.cache)))
            # A residual form checks its beta state first.
            @test occursin(r"the state holds 60 assets, and `X` holds 5|the beta state carries 60 assets, and the step brings 5",
                           message(() -> po.descriptor_step(st,
                                                            po.port_opt_view(rows(syn, 1:2),
                                                                             1:5))))
        end
        # The beta recursion folds by itself too: a step from the carried state equals the
        # batch call, also where `min_obs` counts observations across the cut.
        rm = po.market_return_series(syn, "market_cap")
        B, Vm = po.ew_beta_series(syn.X, rm, 0.8, 7, 1e-12, syn.pnl.amsk)
        b1 = po.ew_beta_series!(po.ew_beta_state(nothing, syn.X, rm, 0.8), syn.X[1:4, :],
                                rm[1:4], 0.8, 7, 1e-12, syn.pnl.amsk[1:4, :])
        b2 = po.ew_beta_series!(po.ew_beta_state(b1.st, syn.X, rm, 0.8), syn.X[5:160, :],
                                rm[5:160], 0.8, 7, 1e-12, syn.pnl.amsk[5:160, :])
        @test same([b1.B; b2.B], B) && same([b1.Vm; b2.Vm], Vm) && b2.st.t == 160
        @test occursin("cannot merge two states fitted on disjoint blocks",
                       message(() -> po.merge_states(b1.st, b2.st)))
        @test occursin("the beta state carries 60 assets, and the step brings 5",
                       message(() -> po.ew_beta_state(b1.st, syn.X[:, 1:5], rm, 0.8)))
        @test occursin("hold one entry per asset",
                       message(() -> po.EWBetaState(; b = [0.0], mu = [0.0, 0.0],
                                                    cv = [0.0], n = [0], act = trues(1),
                                                    mu_m = 0.0, var_m = 0.0, t = 0)))
        # A state of the other form is refused.
        res = partial_fit!(EWResidualVolatility(), rows(syn, 1:5)).cache
        vol = partial_fit!(EWVolatility(), rows(syn, 1:5)).cache
        @test occursin("holds a beta state",
                       message(() -> po.descriptor_step(EWVolatility(; cache = res),
                                                        rows(syn, 6:7))))
        @test occursin("holds no beta state",
                       message(() -> po.descriptor_step(EWResidualVolatility(; cache = vol),
                                                        rows(syn, 6:7))))
        @test po.EWVolatilityState(; ve = vol.ve).ve === vol.ve
        @test po.show_fields(EWVolatility()) == (:ce, :alg, :mar)
        @test po.show_fields(EWResidualVolatility()) ==
              (:mcap, :ce, :beta_decay, :alg, :mar, :min_val)
        # A `ce` with no fold has no state: the Descriptor answers no step, and its batch
        # call is the variance series of `ce`.
        wv = EWVolatility(; ce = WindowedVariance(; window = 20))
        @test isnothing(po.descriptor_step(wv, rows(syn, 1:5)))
        @test isnothing(po.carry_lookback(wv))
        @test isnothing(po.ew_volatility_fold(wv, syn, nothing).st)
        # On the carry fold each Descriptor counts as one row, so the carry keeps the rows of
        # the lag alone, and every step equals the batch fit.
        vf = ["market" => ConstantExposure(),
              "volatility" => CompositeExposure(;
                                                descriptors = [EWVolatility(; half_life = 10),
                                                               EWDownsideVolatility(; half_life = 5)]),
              "residual" => CompositeExposure(;
                                              descriptors = [EWResidualVolatility(; half_life = 5,
                                                                                  beta_half_life = 8),
                                                             BookToPrice()]),
              "industry" => OneHotExposure(; field = "industry", family = "industry")]
        pe = CrossSectionalFactorPrior(; lambda = 1, factors = vf,
                                       families = ["industry" => nothing], minra = 5,
                                       pe = GRID_PE, ve = GRID_VE)
        @test isnothing(po.lookback(pe))
        @test po.cross_sectional_carry_rows(pe) == 2
        e = (0, 50, 80, 81, 120, 121, 160)
        s = stream(pe, syn, e)
        fitted = 0
        for (k, x) in enumerate(s)
            b = try
                batch(pe, k, syn, e)
            catch err
                err
            end
            if isa(b, LowOrderPrior)
                fitted += 1
                @test dropped(x.pr) == dropped(b)
                @test same(x.pr.rr.Ms, b.rr.Ms)
                # The dropped member of the industry family moves at the fourth step. A move
                # folds, and equals the batch fit to rounding alone (#1605).
                if k < 4
                    @test same(x.pr.mu, b.mu) && same(x.pr.sigma, b.sigma)
                else
                    @test near(x.pr, b)
                end
            else
                @test typeof(x.pr) == typeof(b)
            end
        end
        # The case reads out at least once, so the loop is not vacuous.
        @test fitted >= 3
        @test [only(dropped(x.pr)) for x in s] ==
              ["industry=Banks", "industry=Banks", "industry=Banks", "industry=Real Estate",
               "industry=Real Estate", "industry=Real Estate"]
        @test size(last(s).pe.cache.win.X, 1) == 2
        @test all(d -> isa(d.cache, po.EWVolatilityState),
                  last(last(s).pe.cache.xf[2]).descriptors)
        @test isa(first(last(last(s).pe.cache.xf[3]).descriptors).cache,
                  po.EWVolatilityState)
        # A `ce` with no fold makes the carry keep every row, and the strict carry rule
        # refuses it in the constructor.
        wf = ["market" => ConstantExposure(),
              "volatility" => CompositeExposure(; descriptors = [wv])]
        @test isnothing(po.cross_sectional_carry_rows(CrossSectionalFactorPrior(;
                                                                                factors = wf,
                                                                                minra = 5)))
        @test occursin("the factor \"volatility\" (CompositeExposure) has no finite look-back",
                       message(() -> CrossSectionalFactorPrior(; factors = wf, minra = 5,
                                                               carry = FoldOnly())))
    end

    @testset "An EW beta Descriptor folds one row from its carried state (#1608)" begin
        # The state runs each recursion from the first observation with the arithmetic of the
        # batch call, so each folded row equals the batch call to the last bit, in blocks of
        # any size. With `agg_obs > 1` the state carries the observations of the window that
        # is not complete, so a block that ends inside a window gives the same rows, and the
        # shrinkage of a group reads the labels and the weights of the row that closes the
        # window. A second step from the same state gives the same rows, because a step
        # copies the state.
        blocks = ((1, 1), (2, 3), (4, 30), (31, 31), (32, 33), (34, 100), (101, 101),
                  (102, 250))
        # A copy is equal and shares no array with the original, at every level.
        function copied(c, s)
            return all(fieldnames(typeof(c))) do f
                x, y = getfield(c, f), getfield(s, f)
                if isa(x, po.AbstractPartialFitState)
                    copied(x, y)
                else
                    isequal(x, y) && (!isa(x, AbstractArray) || x !== y)
                end
            end
        end
        for de in
            (EWMarketBeta(; half_life = 10), EWMarketBeta(; half_life = 10, agg_obs = 3),
             EWMarketBeta(; half_life = 10, group = "industry", min_group_size = 3),
             EWBeta(; decay = 0.8, min_obs = 30, agg_obs = 4, group = "industry",
                    min_group_size = 3), EWDownsideBeta(; half_life = 7, mar = 0.001),
             EWMacroSensitivity(; series = "MACRO", half_life = 8),
             EWMacroSensitivity(; series = "MACRO", half_life = 8, agg_obs = 3))
            full = descriptor(de, rd)
            st = de
            for (a, b) in blocks
                r = po.descriptor_step(st, rows(rd, a:b))
                @test same(r.D, full[a:b, :])
                @test same(po.descriptor_step(st, rows(rd, a:b)).D, r.D)
                @test copied(partial_fit!(st, rows(rd, a:b)).cache, r.de.cache)
                st = r.de
            end
            @test count(isfinite, full) > 4000
            @test copied(copy(st.cache), st.cache)
            @test occursin("cannot merge two states fitted on disjoint blocks",
                           message(() -> po.merge_states(st.cache, st.cache)))
            @test occursin("carries $N assets, and the step brings 5",
                           message(() -> po.descriptor_step(st,
                                                            po.port_opt_view(rows(rd, 1:2),
                                                                             1:5))))
        end
        # A step refuses the gap and the infinite value of the batch call, and names the same
        # observations, counted from the first one that the state folded.
        j = findfirst(==("MACRO"), rd.ne)
        withE(E) = ReturnsResult(; nx = rd.nx, X = rd.X, ne = rd.ne, E = E, pnl = rd.pnl)
        E = copy(rd.E)
        E[120, j] = NaN
        dm = EWMacroSensitivity(; series = "MACRO", half_life = 8)
        @test occursin("none at observation 120", message(() -> descriptor(dm, withE(E))))
        @test occursin("none at observation 120",
                       message(() -> partial_fit!(partial_fit!(dm, rows(withE(E), 1:100)),
                                                  rows(withE(E), 101:140))))
        E[120, j] = 0.0
        E[118:120, j] .= NaN
        d3 = EWMacroSensitivity(; series = "MACRO", half_life = 8, agg_obs = 3)
        @test occursin("none at observations 118 to 120",
                       message(() -> descriptor(d3, withE(E))))
        @test occursin("none at observations 118 to 120",
                       message(() -> partial_fit!(partial_fit!(d3, rows(withE(E), 1:119)),
                                                  rows(withE(E), 120:140))))
        # The window of observations 148 to 150 holds two finite values, so the gap is
        # accepted, and the infinite value is refused.
        E[118:120, j] .= 0.0
        E[150, j] = Inf
        @test occursin("infinite at observation 150",
                       message(() -> descriptor(d3, withE(E))))
        @test occursin("infinite at observation 150",
                       message(() -> partial_fit!(partial_fit!(d3, rows(withE(E), 1:100)),
                                                  rows(withE(E), 101:200))))
        # The state of the other Descriptor is refused.
        b = partial_fit!(EWMarketBeta(; half_life = 10), rows(rd, 1:20)).cache
        m = partial_fit!(dm, rows(rd, 1:20)).cache
        @test occursin("holds no reference return, so it belongs to another Descriptor",
                       message(() -> partial_fit!(EWMacroSensitivity(; series = "MACRO",
                                                                     cache = b),
                                                  rows(rd, 21:22))))
        @test occursin("holds a reference return, so it belongs to another Descriptor",
                       message(() -> partial_fit!(EWBeta(; decay = 0.9, min_obs = 5,
                                                         cache = m), rows(rd, 21:22))))
        @test occursin("hold one market return and one reference return each",
                       message(() -> po.EWBlockState(; st = b.st, X = zeros(1, 2),
                                                     m = Float64[])))
        @test occursin("hold one entry per asset, got 2 and 1",
                       message(() -> po.EWDownsideBetaState(; cd = [0.0, 0.0], n = [0],
                                                            vd = 0.0, t = 0)))
        @test po.show_fields(EWDownsideBeta()) == (:mcap, :decay, :min_obs, :mar, :min_val)
        # On the carry fold each Descriptor counts as one row, so the carry keeps the rows of
        # the lag alone, and the exposures of every step equal those of the batch fit.
        ewb = ["market" => ConstantExposure(),
               "beta" => CompositeExposure(;
                                           descriptors = [EWMarketBeta(; half_life = 10,
                                                                       agg_obs = 3,
                                                                       group = "industry",
                                                                       min_group_size = 3),
                                                          EWDownsideBeta(; half_life = 7)]),
               "macro" => CompositeExposure(;
                                            descriptors = [EWMacroSensitivity(; series = "MACRO",
                                                                              half_life = 8,
                                                                              agg_obs = 2),
                                                           BookToPrice()]),
               "industry" => OneHotExposure(; field = "industry", family = "industry")]
        pe = CrossSectionalFactorPrior(; lambda = 1, factors = ewb,
                                       families = ["industry" => nothing], minra = 5,
                                       pe = GRID_PE, ve = GRID_VE)
        @test isnothing(po.lookback(pe))
        @test po.cross_sectional_carry_rows(pe) == 2
        e = (0, 50, 80, 81, 120, 121, 160, 250)
        s = stream(pe, rd, e)
        fitted = 0
        for (k, x) in enumerate(s)
            b = try
                batch(pe, k, rd, e)
            catch err
                err
            end
            if isa(b, LowOrderPrior)
                fitted += 1
                @test same(x.pr.rr.Ms, b.rr.Ms)
                @test same(x.pr.mu, b.mu) && same(x.pr.sigma, b.sigma)
            else
                @test typeof(x.pr) == typeof(b)
            end
        end
        # The case reads out at least once, so the loop is not vacuous.
        @test fitted >= 3
        @test size(last(s).pe.cache.win.X, 1) == 2
        @test all(d -> isa(d.cache, Union{po.EWBlockState, po.EWDownsideBetaState}),
                  last(last(s).pe.cache.xf[2]).descriptors)
        @test isa(first(last(last(s).pe.cache.xf[3]).descriptors).cache, po.EWBlockState)
    end

    @testset "The Descriptor scores of a Return Forecast fold their stateful Descriptors (#1587)" begin
        # A Descriptor of the scores that carries a state reads the rows of the step alone, so
        # the carry keeps the rows of the lag alone, and every step equals the batch fit to the
        # last bit. The state carries the Descriptor Scores with the state of each Descriptor.
        ind = grid_config("FamOne", rd).factors
        fk = (; lambda = 0.4, c = 0.6, minra = 5, pe = GRID_PE, ve = GRID_VE,
              families = ["industry" => nothing])
        e = (0, 90, 91, 92, 170, 250)
        for de in (Reversal(), RollingMomentum(; window = 40, skip = 5))
            ds = DescriptorScores(;
                                  descriptors = [Passthrough(; field = "net_income_ttm"),
                                                 de], group = "industry")
            for rfe in (FixedWeightedReturnForecast(; scores = ds, scale = 0.02),
                        ExpWeightedReturnForecast(; scores = ds),
                        TargetReturnForecast(; scores = ds, half_life = 10.0))
                pe = CrossSectionalFactorPrior(; fk..., factors = ind, rfe = rfe)
                @test po.lookback(pe) == po.lookback(de)
                @test po.carry_lookback(rfe) == 1
                @test po.cross_sectional_carry_rows(pe) == 2
                for (k, x) in enumerate(stream(pe, rd, e))
                    b = batch(pe, k, rd, e)
                    st = x.pe.cache
                    @test size(st.win.X, 1) == 2
                    @test same(x.pr.mu, b.mu) && same(x.pr.sigma, b.sigma)
                    @test same(x.pr.rr.rf.mu, b.rr.rf.mu)
                    @test isa(st.fds.descriptors[2].cache, po.RollingLogReturnState)
                end
            end
        end
        # The first steps end inside the warm-up of the momentum factor, so they bring no
        # observation to score. Their rows still fold into the state of the Reversal score: a
        # state that started at a later row would sum from another first row, and its scores
        # would differ from the batch fit in the last bits.
        syn = synthetic_asset_panel(; n_assets = 60, n_observations = 160, n_industries = 4,
                                    rng = StableRNG(1583)).rd
        momentum = ["market" => ConstantExposure(),
                    "momentum" => CompositeExposure(;
                                                    descriptors = [RollingMomentum(; window = 40,
                                                                                   skip = 5)]),
                    "industry" => OneHotExposure(; field = "industry", family = "industry")]
        rev = DescriptorScores(; descriptors = [Reversal(; window = 10), BookToPrice()])
        pe = CrossSectionalFactorPrior(; lambda = 1, factors = momentum,
                                       families = ["industry" => nothing], minra = 5,
                                       pe = GRID_PE, ve = GRID_VE,
                                       rfe = FixedWeightedReturnForecast(; scores = rev,
                                                                         scale = 0.02))
        @test po.cross_sectional_carry_rows(pe) == 2
        e = (0, 10, 20, 50, 80, 81, 120, 160)
        s = stream(pe, syn, e)
        @test isnothing(s[1].pe.cache.fsc) && isnothing(s[2].pe.cache.Ms)
        @test isa(s[2].pe.cache.fds.descriptors[1].cache, po.RollingLogReturnState)
        fitted = 0
        for (k, x) in enumerate(s)
            b = try
                batch(pe, k, syn, e)
            catch err
                err
            end
            if isa(b, LowOrderPrior)
                fitted += 1
                @test same(x.pr.mu, b.mu) && same(x.pr.rr.rf.hist, b.rr.rf.hist)
            else
                @test typeof(x.pr) == typeof(b)
            end
        end
        # Two of the steps read out, so the loop is not vacuous.
        @test fitted == 2
        # A forecast that does not fold its rows reads the panel, so the state carries no
        # Descriptor Scores.
        kf = CrossSectionalFactorPrior(; fk..., factors = ind,
                                       rfe = TargetReturnForecast(;
                                                                  scores = DescriptorScores(;
                                                                                            descriptors = [Reversal()]),
                                                                  half_life = 10.0,
                                                                  cv = KFold()))
        @test !po.folds_forecast_rows(kf.rfe)
        @test isnothing(partial_fit!(kf, rows(rd, 1:90)).cache.fds)
    end

    @testset "A Target Return Forecast trains on rows of the warm-up and folds (#1588)" begin
        # Under `whole_history` the member trains on the last `lag + horizon - 1` rows before
        # the block, and the histories hold `pe.lag` of them. The state keeps the scores of the
        # other rows, which are rows of the warm-up of the momentum factor, before its first
        # row. A neutralised score is `NaN` before the block, so the rows train only under a
        # rule of `ofit` that does not neutralise. Before #1588 a horizon above one refitted
        # from every row, and a forecast lag above `pe.lag` lost the first of those rows: under
        # `UnadjustedForecast` its `mu` differed from the batch fit by 2 to 12 per cent.
        syn = synthetic_asset_panel(; n_assets = 60, n_observations = 160, n_industries = 4,
                                    rng = StableRNG(1583)).rd
        momentum = ["market" => ConstantExposure(),
                    "momentum" => CompositeExposure(;
                                                    descriptors = [RollingMomentum(; window = 40,
                                                                                   skip = 5)]),
                    "industry" => OneHotExposure(; field = "industry", family = "industry")]
        rev = DescriptorScores(; descriptors = [Reversal(; window = 10), BookToPrice()])
        tf(; kw...) = TargetReturnForecast(; scores = rev, half_life = 10.0, kw...)
        wv = WindowedVariance(;
                              ve = ExpWeightedVariance(; decay = 2.0^(-1 / 20),
                                                       min_obs = 5), window = 60)
        un = UnadjustedForecast()
        # Steps of one row cross the end of the warm-up, so the rows that the state keeps come
        # from several steps. The rolling variance fits every observation again at each step.
        e = (0, 20, 41, 42, 43, 44, 45, 46, 47, 80, 120, 160)
        for (cfg, a) in (((; ofit = un, rfe = tf(; horizon = 5)), 4),
                         ((; ofit = un, ve = wv, rfe = tf(; horizon = 5)), 4),
                         ((; ofit = OrthogonalPartCalibration(), rfe = tf(; horizon = 3, lag = 2)), 3),
                         ((; ofit = un, lag = 2, rfe = tf(; horizon = 2, lag = 4)), 3),
                         ((; ofit = un, rfe = tf(; lag = 3)), 2), ((; rfe = tf(; horizon = 5)), 4))
            pe = CrossSectionalFactorPrior(; lambda = 1, factors = momentum,
                                           families = ["industry" => nothing], minra = 5,
                                           pe = GRID_PE, ve = GRID_VE, cfg...)
            @test po.folds_forecast_rows(pe.rfe)
            @test po.cross_sectional_forecast_lead(pe.rfe, pe.lag) == a
            s = stream(pe, syn, e)
            fitted = 0
            for (k, x) in enumerate(s)
                b = try
                    batch(pe, k, syn, e)
                catch err
                    err
                end
                if isa(b, LowOrderPrior)
                    fitted += 1
                    rf, rb = x.pr.rr.rf, b.rr.rf
                    @test same(x.pr.mu, b.mu) && same(x.pr.sigma, b.sigma)
                    @test same(rf.mu, rb.mu) && rf.model.n == rb.model.n
                    @test isequal(rf.calib, rb.calib) && isequal(rf.ocalib, rb.ocalib)
                else
                    @test typeof(x.pr) == typeof(b)
                end
            end
            @test fitted == 2
            st = last(s).pe.cache
            @test size(st.fsc.S, 1) - size(st.Ms, 1) == a
            @test size(st.win.X, 1) == po.cross_sectional_carry_rows(pe)
        end
        # The strict carry rule accepts the member, because it folds.
        @test po.folds_forecast_rows(CrossSectionalFactorPrior(; factors = momentum,
                                                               carry = FoldOnly(),
                                                               rfe = tf(; horizon = 5)).rfe)
        # The member that reads the block alone keeps no row of the warm-up.
        @test po.cross_sectional_forecast_lead(tf(; horizon = 5, whole_history = false),
                                               1) == 0
        @test po.cross_sectional_forecast_lead(ExpWeightedReturnForecast(; scores = rev),
                                               1) == 0
        @test po.cross_sectional_forecast_lead(tf(; horizon = 2, lag = 1), 3) == 0
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
