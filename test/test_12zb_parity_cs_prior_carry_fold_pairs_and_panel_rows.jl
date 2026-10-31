#=
The checks of this subject are split over the files `test_12zb_parity_cs_prior_carry_fold*.jl`, so that each file of the
suite runs in about the same time. This file holds one part of them.

The carry fold of the Cross-Sectional Factor Prior (#1471, map #1375, ADR 0193).

An unwrapped prior folds as a carry: its first step seeds a `CrossSectionalCarryState`, and each
step computes the exposures, the regression and the idiosyncratic variance of the new
observations alone, from the panel rows that its Descriptors read. The factor prior and the
idiosyncratic variance fold, and the Return Forecast and the idiosyncratic correlation refit at the
call with no data. A factor that comes alive keeps every fitted observation, and folds the factor
prior again (#1627). A batch choice
that moves folds the move, and solves again each observation whose answer depends on the basis
(#1605, #1613). It folds the same way under a target that gives the same coefficients in every
basis, as `is_basis_invariant` answers (#1614), and under a `GeneralisedLinearModel` target (#1615).

A step folds the member states in place, as every fold does, so each stream reads the prior out
right after its step.

Every `Parity_CrossSectionalFactorPrior_CarrySeed*` file is an output of the oracle's online update
over the stream of #1468, rows 1-90, 91-170 and 171-250 of the exchange that
`parity_write(dir, grid_fixture(parity_large_panel()); filled = true)` writes. Its factor prior
is an exponentially weighted mean and covariance of half-life 20 and `min_observations = 5`, whose
window of 60 rows cuts the factor returns of the first call alone. Each file stacks the three steps
vertically, as the `Online*` files of test_12z do: one row of `mu`, one 40 x 40 block of `sigma`,
or the factor returns of every row fitted so far. The cases are `SeedStyle` (`Base` with
`families = ["style" => nothing]`) and `SeedIndustry` (`FamOne`). The pinned choice with no window
is checked against the `Online*Fold*` files of test_12z.

A prior whose tree reads the Exogenous Series folds on the carry route too (#1479, ADR 0193): the
buffer and the carried rows hold the series, and the state derives each row of the returns net of
the observed factors one time. Its pinned choice under currency factors is checked against the
`OnlineCurrencyStyle_Fold*` files of test_12z.

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
# A test-local target that fits as a `LinearModel` and keeps the default answer of
# `is_basis_invariant`, so a move of the dropped member under it fits every observation again
# (#1614).
struct BasisBoundTarget <: PortfolioOptimisers.AbstractUnpenalisedRegressionTarget end
function PortfolioOptimisers.factory(::BasisBoundTarget, w::PortfolioOptimisers.ObsWeights)
    return factory(LinearModel(), w)
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
        # The values of the prequential default, to the three decimals that ADR 0200 states.
        pcalibs = Dict("return" => -2.020, "sharpe" => -2.069)
        for (unit, uname) in ((IdiosyncraticReturnUnit(), "return"), (sharpe, "sharpe"))
            rfe = TargetReturnForecast(; scores = ds, half_life = 10.0, unit = unit,
                                       cv = KFold())
            a = return_forecast(rfe, pp, blk)
            b = return_forecast(rfe, pp, blk2)
            @test isequal(a.calib, b.calib)
            # Measured maxrel 1.3e-15 (return) and 1.7e-14 (Sharpe).
            @test parity_compare([a.calib], [calibs[uname]]; rtol = 1e-13,
                                 name = "calib $(uname)").ok
            @test same(a.mu, b.mu)
            rfe = TargetReturnForecast(; scores = ds, half_life = 10.0, unit = unit)
            a = return_forecast(rfe, pp, blk)
            b = return_forecast(rfe, pp, blk2)
            @test isequal(a.calib, b.calib)
            @test isapprox(a.calib, pcalibs[uname]; atol = 5e-4)
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

    @testset "The standardised residuals read a pair of leverage one as a gap (#1720)" begin
        # Asset 1 shares the level "A" with asset 2 until row 150, and asset 2 moves to "B"
        # at row 151. So asset 1 is a pair of leverage one from row 151 on, and its variance
        # there is positive, because it decays from the residuals of the rows before. Before
        # #1720 the prior standardised the zero residual of such a pair to a standardised
        # return of zero, which the scenarios and the idiosyncratic correlation read as data,
        # and gave the fill to a pair whose variance was zero.
        T, N, s = 300, 8, 151
        rng = StableRNG(1720)
        lvl = repeat(permutedims([i <= 2 ? "A" : (i <= 5 ? "B" : "C") for i in 1:N]), T)
        lvl[s:T, 2] .= "B"
        fi = Dict(k => 0.01 .* randn(rng, T) for k in ("A", "B", "C"))
        X = 0.01 .* randn(rng, T, N)
        for t in 1:T, i in 1:N
            X[t, i] += fi[lvl[t, i]][t]
        end
        mcap = exp.(log(1e9) .+ 0.5 .* randn(rng, 1, N) .+
                    0.01 .* cumsum(randn(rng, T, N); dims = 1))
        lp = ReturnsResult(; nx = ["a$i" for i in 1:N], X = X,
                           pnl = asset_panel([NumericPanelInput(; name = "market_cap",
                                                                vals = mcap),
                                              CategoricalPanelInput(; name = "industry",
                                                                    vals = lvl)]))
        lcfg = (; minra = 5, pe = GRID_PE, ve = GRID_VE, families = ["industry" => nothing],
                factors = ["market" => ConstantExposure(),
                           "industry" =>
                               OneHotExposure(; field = "industry", family = "industry")])
        @test CrossSectionalFactorPrior(; lcfg...).l1mask === MaskedLeverageOne()
        a = prior(CrossSectionalFactorPrior(; lcfg...), lp)
        b = prior(CrossSectionalFactorPrior(; lcfg..., l1mask = UnmaskedLeverageOne()), lp)
        h1 = a.rr.csr.h1
        vs = a.rr.vs
        r1 = findall(view(h1, :, 1))
        @test findall(i -> any(view(h1, :, i)), 1:N) == [1]
        @test r1 == s:(T - 1)
        @test all(>(0), vs[r1, 1])
        # The two rules differ only at the marked pairs. There the oracle's form reads a
        # standardised return of zero, and the default reads the mean standardised return of
        # the other assets of the observation. Measured: 2.7e-18.
        eps = a.rr.csr.eps
        zbar = [mean(filter(isfinite, eps[t, 2:N] ./ sqrt.(vs[t, 2:N]))) for t in r1]
        D = a.X .- b.X
        other = trues(size(D))
        other[r1, 1] .= false
        @test all(iszero, D[other])
        @test isapprox(D[r1, 1], zbar .* sqrt(vs[end, 1]); atol = 1e-15)
        @test maximum(abs, D[r1, 1]) > 1e-4
        # The pure function gives the same gap whatever the variance of the pair, and the
        # oracle's form gives zero or the gap by the variance.
        m = trues(1, 1)
        for v in (1e-6, 0.0)
            e = po.leverage_one_residuals(MaskedLeverageOne(), m, [0.0;;])
            z = po.cross_sectional_standardised_residuals(e, [v;;], m; filled = false)
            @test isnan(only(z))
            e = po.leverage_one_residuals(UnmaskedLeverageOne(), m, [0.0;;])
            z = po.cross_sectional_standardised_residuals(e, [v;;], m; filled = false)
            @test isequal(only(z), iszero(v) ? NaN : 0.0)
        end
        # `sigma` reads the residuals under a threshold above zero alone. There the row of
        # asset 1 of the idiosyncratic correlation moves. Measured: 6.4e-2 of the largest
        # entry of `sigma`.
        ta = prior(CrossSectionalFactorPrior(; lcfg..., th = 0.5), lp)
        tb = prior(CrossSectionalFactorPrior(; lcfg..., th = 0.5,
                                             l1mask = UnmaskedLeverageOne()), lp)
        @test a.sigma == b.sigma
        @test relerr(ta.sigma, tb.sigma) > 1e-2
        # The carry fold masks the same rows, so it equals the batch fit under each rule.
        le = (0, 200, 201, 300)
        for l1 in (MaskedLeverageOne(), UnmaskedLeverageOne()), th in (0.0, 0.5)
            pe = CrossSectionalFactorPrior(; lcfg..., th = th, l1mask = l1)
            for (k, x) in enumerate(stream(pe, lp, le))
                @test x.pe.cache isa po.CrossSectionalCarryState
                bk = batch(pe, k, lp, le)
                @test isequal(isnan.(x.pr.X), isnan.(bk.X))
                @test relerr(x.pr.X, bk.X) <= 1e-12
                @test relerr(x.pr.sigma, bk.sigma) <= 1e-12
            end
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
end
