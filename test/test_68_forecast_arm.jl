#=
The forecast-reading arm of the online portfolio selection family (ADR 0158, ADR 0165, #1176):
the price-level statistics and their fold, the composites the later papers add, the Prior
adapter, the fold-or-refit of a forecaster on the Rule State, and the rules that read a
forecast — the reversion step with its scale, the tracking step, the cost-aware step and the
sparse portfolio. Parity is measured against hand-written recursions and the worked examples
of the ADRs; the papers' defaults are asserted where they decide the shape of the answer.
=#
@testset "The forecast arm: statistics, the Prior adapter, the fold and the rules" begin
    using Test, PortfolioOptimisers, StableRNGs, LinearAlgebra, Statistics, Dates
    po = PortfolioOptimisers
    OPS = po.OnlinePortfolioSelection

    rng = StableRNG(11)
    T, N = 40, 4
    R = 0.02 .* randn(rng, T, N)
    X = 1 .+ R
    nx = ["A", "B", "C", "D"]
    ts = Date(2020, 1, 1) .+ Day.(0:(T - 1))
    rd = ReturnsResult(; nx = nx, X = R, ts = ts)
    rows(r, i) = po.port_opt_view(r, i, :)
    maxerr(a, b) = maximum(abs.(a .- b))
    # The price relative forecast of a statistic over a matrix of returns.
    xhat(alg, Y) = 1 .+ vec(mean(PriceLevelExpectedReturns(; alg = alg), Y))
    # Levels from a matrix of relatives, the last at one.
    function levels(Y)
        P = ones(size(Y, 1) + 1, size(Y, 2))
        for t in size(Y, 1):-1:1
            P[t, :] .= P[t + 1, :] ./ Y[t, :]
        end
        return P
    end

    @testset "The three-asset example of ADR 0158, and the windowed statistics" begin
        x2 = [0.90, 1.05, 1.00]
        x3 = [1.05, 1.00, 0.98]
        Y = [x2'; x3'] .- 1
        @test isapprox(xhat(MovingAverage(; window = 3), Y),
                       (1 .+ 1 ./ x3 .+ 1 ./ (x2 .* x3)) ./ 3; atol = 1e-15)
        @test isapprox(xhat(MovingAverage(; window = 3), Y), [1.0035, 0.9841, 1.0136];
                       atol = 5e-5)
        # The window truncates over the first rows: a wider window reads the same levels.
        @test xhat(MovingAverage(; window = 5), Y) == xhat(MovingAverage(; window = 3), Y)
        @test xhat(WindowPeak(; window = 5), Y) == xhat(WindowPeak(; window = 3), Y)
        # The peak is never below the last price, and it is one at a window peak.
        p = xhat(WindowPeak(; window = 3), Y)
        @test all(p .>= 1) && p[2] == 1
        @test p ≈ vec(maximum(levels([x2'; x3']); dims = 1))
        # The lagged price at one is the reciprocal of the last relative, at zero it is one.
        @test xhat(LaggedPrice(; lag = 1), Y) ≈ 1 ./ x3
        @test xhat(LaggedPrice(; lag = 2), Y) ≈ 1 ./ (x2 .* x3)
        @test xhat(LaggedPrice(; lag = 0), Y) == ones(3)
        @test xhat(LaggedPrice(; lag = 5), Y) == xhat(LaggedPrice(; lag = 2), Y)
        @test po.rows_needed(LaggedPrice(; lag = 3)) == 3
        @test po.rows_needed(WindowPeak(; window = 6)) == 5
        @test_throws DomainError LaggedPrice(; lag = -1)
        @test_throws DomainError WindowPeak(; window = 1)
        # `dims = 2` and the empty refusal.
        @test size(mean(PriceLevelExpectedReturns(; alg = WindowPeak()), transpose(Y);
                        dims = 2)) == (3, 1)
        @test_throws IsEmptyError mean(PriceLevelExpectedReturns(), zeros(0, 3))
    end

    @testset "The folding statistics: the hand recursion, the fold and the batch agree" begin
        # OLMAR-2's exponential moving average, seeded at the first price.
        alpha = 0.5
        ema = ones(N)
        for t in 1:T
            ema = alpha .+ (1 - alpha) .* ema ./ X[t, :]
        end
        me = PriceLevelExpectedReturns(; alg = ExponentialMovingAverage(; alpha = alpha))
        @test isapprox(xhat(me.alg, R), ema; atol = 1e-14)
        # The batch over levels is the same recursion.
        P = levels(X)
        ma = P[1, :]
        for t in 2:size(P, 1)
            ma = alpha .* P[t, :] .+ (1 - alpha) .* ma
        end
        @test isapprox(xhat(me.alg, R), ma; atol = 1e-14)
        # After one level the forecast is one; after one row it is `α + (1 − α) / x₁`.
        @test xhat(me.alg, R[1:1, :]) ≈ alpha .+ (1 - alpha) ./ X[1, :]
        # The fold, row by row, is the batch to the ulp, and the read-out reads the state.
        mf = me
        for t in 1:T
            mf = po.partial_fit!(mf, R[t, :])
        end
        @test mf.cache.n == T
        @test maxerr(vec(mean(mf)), vec(mean(me, R))) < 1e-15
        @test maxerr(vec(mean(po.partial_fit!(me, R))), vec(mean(me, R))) < 1e-15
        @test po.supports_partial_fit(me) &&
              !po.supports_partial_fit(PriceLevelExpectedReturns())
        @test po.rows_needed(me) == 1 && isnothing(po.window_rows(me.alg))
        @test_throws ArgumentError po.partial_fit!(PriceLevelExpectedReturns(), R[1, :])
        @test_throws ArgumentError mean(me)
        # The reweighted relative: seeded at the first relative, so the forecast after the
        # first row is one, and `theta` is the per-asset weight's strength.
        theta = 0.7
        phi = X[1, :]
        for t in 1:T
            g = theta .* X[t, :] ./ (theta .* X[t, :] .+ phi)
            phi = g .+ (1 .- g) .* phi ./ X[t, :]
        end
        rpr = PriceLevelExpectedReturns(; alg = ReweightedPriceRelative(; theta = theta))
        @test isapprox(xhat(rpr.alg, R), phi; atol = 1e-14)
        @test xhat(rpr.alg, R[1:1, :]) ≈ ones(N)
        rf = rpr
        for t in 1:T
            rf = po.partial_fit!(rf, R[t, :])
        end
        @test maxerr(vec(mean(rf)), vec(mean(rpr, R))) < 1e-15
        # The state pays the partial-fit interface: copy aliases nothing, a view slices.
        st = rf.cache
        c = copy(st)
        @test c.stat == st.stat && c.stat !== st.stat
        @test po.port_opt_view(st, [2, 4]).stat == st.stat[[2, 4]]
        @test po.port_opt_view(rf, [1, 3]).cache.stat == st.stat[[1, 3]]
        @test_throws ArgumentError po.merge_states(st, c)
        @test_throws DomainError ExponentialMovingAverage(; alpha = 0)
        @test_throws DomainError ReweightedPriceRelative(; theta = 0)
    end

    @testset "A folding statistic under an active mask: the reset, the count, the arms" begin
        # Two assets over six rows, the second unlisted over the first three.
        Rg = [0.010 -0.020
              -0.015 0.030
              0.020 -0.010
              -0.005 0.012
              0.008 -0.004
              -0.011 0.021]
        amskg = trues(6, 2)
        amskg[1:3, 2] .= false
        Rg[.!amskg] .= NaN
        pnlg = AssetPanel(; amsk = amskg, emsk = copy(amskg))
        emag = PriceLevelExpectedReturns(; alg = ExponentialMovingAverage(; alpha = 0.5))
        # The relisted asset reads its own rows alone, and the listed one reads every row.
        gf = po.partial_fit!(emag, Rg; active_mask = amskg)
        @test gf.cache.n == 6 && gf.cache.nu == [6, 3]
        @test vec(mean(gf))[2] ≈ vec(mean(emag, Rg[4:6, 2:2]))[1]
        @test vec(mean(gf))[1] ≈ vec(mean(emag, Rg[:, 1:1]))[1]
        # Below the warm-up the asset carries no forecast, so the step holds its leg.
        cold3 = po.partial_fit!(emag, Rg[1:3, :]; active_mask = amskg[1:3, :])
        @test cold3.cache.nu == [3, 0] && isnan(vec(mean(cold3))[2])
        @test po.flat_where_undefined(1 .+ vec(mean(cold3)))[2] == 1
        # The Asset Panel arm of the batch verb is that fold, so the two arms agree.
        @test isequal(vec(mean(emag, Rg, pnlg)), vec(mean(gf)))
        # With no mask a gap is a delisting, because nothing states otherwise.
        @test isequal(vec(mean(po.partial_fit!(emag, Rg))), vec(mean(gf)))
        # A windowed statistic takes the root of the seam instead: the Coverage Universe of
        # the window, `NaN` outside it, and no mask of its own.
        maw = PriceLevelExpectedReturns(; alg = MovingAverage(; window = 3))
        muw = vec(mean(maw, Rg, pnlg))
        @test isnan(muw[2]) && isfinite(muw[1])
        @test_throws ArgumentError mean(maw, Rg; active_mask = amskg)
        @test_throws DimensionMismatch po.partial_fit!(emag, Rg[1, :]; active_mask = [true])
        @test_throws DimensionMismatch mean(emag, Rg; active_mask = amskg[:, 1:1])
        # A holiday inside a listing is a flat level: the mask admits the asset, its return
        # is not there, the level did not move, and the fold warms nothing.
        hol = copy(Rg[4:6, :])
        hol[2, 1] = NaN
        hf = po.partial_fit!(emag, hol; active_mask = trues(3, 2))
        @test hf.cache.nu == [2, 3]
        flat = po.partial_fit!(po.partial_fit!(emag, hol[1:1, :]), zeros(1, 2))
        @test po.partial_fit!(emag, hol[1:2, :]; active_mask = trues(2, 2)).cache.stat[1] ≈
              flat.cache.stat[1]
        # A row on which no asset is active folds nothing and leaves every asset cold.
        dead = po.partial_fit!(emag, Rg[4:4, :]; active_mask = falses(1, 2))
        @test dead.cache.nu == [0, 0] && all(isnan, vec(mean(dead)))
        # The cold seed is the carried value that makes each recursion answer its own seed.
        x1 = 1 .+ Rg[4, :]
        @test po.cold_statistic(ExponentialMovingAverage(), x1) == ones(2)
        @test po.cold_statistic(ReweightedPriceRelative(), x1) == x1
        @test po.cold_statistic(KernelTrendPattern(), x1) == x1
        for alg in (ExponentialMovingAverage(), ReweightedPriceRelative())
            @test po.fold_statistic(alg, nothing, x1) ==
                  po.fold_statistic(alg, po.cold_statistic(alg, x1), x1)
        end
        # The reweighted relative seeds at the row's own relative, so a relisting that
        # started warm would read a different number.
        rprg = PriceLevelExpectedReturns(; alg = ReweightedPriceRelative(; theta = 0.7))
        @test vec(mean(po.partial_fit!(rprg, Rg; active_mask = amskg)))[2] ≈
              vec(mean(rprg, Rg[4:6, 2:2]))[1]
        # A statistic that couples the assets pools the live ones alone, so the fold over
        # the masked panel is the fold over the Coverage Universe of the row.
        ktp = PriceLevelExpectedReturns(; alg = KernelTrendPattern(; window = 2))
        kf = po.partial_fit!(ktp, Rg[1:3, :]; active_mask = amskg[1:3, :])
        @test isnan(vec(mean(kf))[2]) && all(isone, kf.cache.hist[:, 2])
        @test vec(mean(kf))[1] ≈ vec(mean(ktp, Rg[1:3, 1:1]))[1]
        kg = po.partial_fit!(ktp, Rg; active_mask = amskg)
        @test all(isfinite, vec(mean(kg))) && kg.cache.nu == [6, 3]
        @test isequal(vec(mean(ktp, Rg, pnlg)), vec(mean(kg)))
        # The count pays the state's interface beside the statistic and the memory.
        @test po.port_opt_view(gf.cache, [2]).nu == [3] && copy(gf.cache).nu == gf.cache.nu
        # A row on which no asset is active resets every asset, however warm the state was:
        # the recursion runs on nothing and the cold seed is what is left.
        warm = po.partial_fit!(emag, Rg[4:6, :]; active_mask = trues(3, 2))
        @test all(>(0), warm.cache.nu)
        wiped = po.partial_fit!(warm, Rg[4, :]; active_mask = falses(2))
        @test wiped.cache.nu == [0, 0] &&
              wiped.cache.stat == po.cold_statistic(emag.alg, ones(2))
        @test all(isnan, vec(mean(wiped)))
        # A panel carries its masks as `observations × assets` whatever `dims` says, so the
        # mask-aware arm reads the same answer from a transposed sample.
        @test isequal(vec(mean(emag, permutedims(Rg), pnlg; dims = 2)),
                      vec(mean(emag, Rg, pnlg; dims = 1)))
        @test isequal(vec(mean(ktp, permutedims(Rg), pnlg; dims = 2)),
                      vec(mean(ktp, Rg, pnlg; dims = 1)))
        @test isequal(vec(mean(maw, permutedims(Rg), pnlg; dims = 2)),
                      vec(mean(maw, Rg, pnlg; dims = 1)))
        # A composite is windowed even where it holds a folding member, so it keeps the
        # Coverage-Universe reading and holds the relisted asset the bare member re-admits.
        cmp = PriceLevelExpectedReturns(;
                                        alg = TrendSwitch(;
                                                          rising = ExponentialMovingAverage()))
        @test !po.folds(cmp.alg)
        @test isnan(vec(mean(cmp, Rg, pnlg))[2]) && isfinite(vec(mean(gf))[2])
    end

    @testset "The composite statistics of the later papers" begin
        P = levels(X[(end - 3):end, :])
        # The truncated exponential average, unnormalised, weights falling with age, and
        # never more than its window of levels.
        tema = TruncatedExponentialMovingAverage(; alpha = 0.5, window = 5)
        @test po.price_level_statistic(tema, P) ≈
              sum(0.5 * 0.5^k .* P[end - k, :] for k in 0:4)
        @test po.price_level_statistic(tema, P[end:end, :]) ≈ 0.5 .* P[end, :]
        @test po.price_level_statistic(tema, levels(X)) ≈ po.price_level_statistic(tema, P)
        # The Gaussian double estimate: nine levels at the paper's defaults, and the second
        # estimate replaces the current level by the previous first estimate.
        gw = GaussianWeightedDoubleEstimate()
        @test po.window_rows(gw) == 9 && po.rows_needed(gw) == 9
        P9 = levels(X[(end - 8):end, :])
        g = [exp(-k^2 / (2 * 2.8^2)) for k in 1:9]
        est(u) = sum(g[k] .* P9[u - k + 1, :] for k in 1:9) ./ sum(g)
        pp2 = (g[1] .* est(9) .+ sum(g[k] .* P9[10 - k + 1, :] for k in 2:9)) ./ sum(g)
        @test po.price_level_statistic(gw, P9) ≈ (est(10) .+ pp2) ./ 2
        @test_throws DomainError GaussianWeightedDoubleEstimate(; tau = 0.1, cutoff = 0.9)
        # A single level has no previous estimate, so the first estimate stands alone.
        @test po.price_level_statistic(gw, ones(1, 3)) == ones(3)
        # The trend tests on a rising, a flat and a falling asset.
        Q = [1.0 1.0 1.0; 1.1 1.0 0.9; 1.2 1.0 0.8; 1.3 1.0 0.7; 1.4 1.0 0.6]
        @test po.trend_sign(PairwiseSlopeSum(), Q) == [1, 0, -1]
        # The regression slope is `0.1` per period on the first asset, so the threshold
        # decides on either side of it, and a ridge weight pulls the slope below.
        @test po.trend_sign(RegressionSlope(; threshold = 0.05), Q) == [1, -1, -1]
        @test po.trend_sign(RegressionSlope(; threshold = 0.2), Q) == [-1, -1, -1]
        @test po.trend_sign(RegressionSlope(; threshold = 0.05, lambda = 100), Q) ==
              [-1, -1, -1]
        # The switch reads each branch on its own window.
        sw = TrendSwitch()
        s = po.price_level_statistic(sw, Q)
        @test s[1] ≈ po.price_level_statistic(sw.rising, Q)[1]
        @test s[2] ≈ Q[end, 2]
        @test s[3] ≈ maximum(Q[:, 3])
        @test po.rows_needed(sw) == 4
        @test isnothing(po.rows_needed(TrendSwitch(; flat = ExponentialMovingAverage())))
        # The composite: identical trends give the trend, and the back-test needs
        # `window - 1` rows beyond the widest trend.
        ct = CompositeTrend(; trends = [MovingAverage(), MovingAverage()])
        @test po.rows_needed(ct) == 8
        @test isnothing(po.rows_needed(CompositeTrend()))
        Pc = levels(X[(end - 8):end, :])
        @test po.price_level_statistic(ct, Pc) ≈
              po.price_level_statistic(MovingAverage(), Pc[6:10, :])
        # With a single level every trend forecasts one, and so does the composite.
        @test po.price_level_statistic(CompositeTrend(), ones(1, 3)) ≈ ones(3)
        # Three distinct trends: a level forecast between the trends' own.
        c3 = po.price_level_statistic(CompositeTrend(), Pc)
        lo = min.(po.price_level_statistic(MovingAverage(), Pc[6:10, :]),
                  po.price_level_statistic(ExponentialMovingAverage(), Pc),
                  po.price_level_statistic(WindowPeak(), Pc[6:10, :]))
        hi = max.(po.price_level_statistic(MovingAverage(), Pc[6:10, :]),
                  po.price_level_statistic(ExponentialMovingAverage(), Pc),
                  po.price_level_statistic(WindowPeak(), Pc[6:10, :]))
        @test all(lo .- 1e-12 .<= c3 .<= hi .+ 1e-12)
        @test_throws IsEmptyError CompositeTrend(; trends = MovingAverage[])
        @test_throws DomainError CompositeTrend(; sigma2 = 0)
    end

    @testset "The Prior adapter" begin
        pa = PriorExpectedReturns()
        @test vec(mean(pa, R)) ≈ vec(mean(R; dims = 1))
        @test size(mean(pa, transpose(R); dims = 2)) == (N, 1)
        @test vec(mean(pa, R, nothing)) ≈ vec(mean(R; dims = 1))
        @test mean(pa, R) == mean(pa, R; strict = false)
        # A price-level forecaster is a legal `me` of a prior.
        pr = prior(EmpiricalPrior(; me = PriceLevelExpectedReturns()), R)
        @test vec(pr.mu) ≈ vec(mean(PriceLevelExpectedReturns(), R))
        # A factor prior is refused at construction, a take-what-is-given prior is not.
        @test_throws ArgumentError PriorExpectedReturns(; pe = FactorPrior())
        @test isa(PriorExpectedReturns(; pe = HighOrderPriorEstimator()),
                  PriorExpectedReturns)
        # A Black–Litterman mean drives a reversion step.
        bl = BlackLittermanPrior(; sets = UniverseSets(; dict = Dict("nx" => nx)),
                                 views = LinearConstraintEstimator(; val = "A == 0.002"))
        blme = PriorExpectedReturns(; pe = bl)
        @test vec(mean(blme, R)) ≈ vec(prior(bl, R).mu)
        res = optimise(OPS(; alg = ForecastReversion(; me = blme)), rd)
        @test sum(res.w) ≈ 1 && all(res.w .>= 0)
        # The adapter folds when its prior does, and `EmpiricalPrior` carries rows instead.
        @test !po.supports_partial_fit(pa)
        @test isnothing(po.rows_needed(pa))
        @test po.forecast_min_rows(pa) == 2 &&
              po.forecast_min_rows(SimpleExpectedReturns()) == 1
        @test po.forecast_min_rows(VarianceExpectedReturns()) == 2
        @test po.forecast_min_rows(ShrunkExpectedReturns()) == 2
        @test po.forecast_min_rows(MedianExpectedReturns()) == 1
        @test !po.holds_second_moment(nothing)
        # A carrier over a forecaster that has folded nothing yet copies as it is.
        cold = po.ForecasterState(SimpleExpectedReturns())
        @test copy(cold).me === cold.me
        # A view forwards to the prior's own view.
        @test isa(po.port_opt_view(pa, [1, 2]), PriorExpectedReturns)
        # A constant column: the default prior's positive-definite repair throws although
        # `mu` is defined, and a covariance without the repair answers the sample mean.
        Rc = copy(R)
        Rc[:, 2] .= 0
        rdc = ReturnsResult(; nx = nx, X = Rc, ts = ts)
        @test_throws ArgumentError mean(pa, Rc)
        @test_throws ArgumentError optimise(OPS(; alg = ForecastReversion(; me = pa)), rdc)
        norep = PriorExpectedReturns(;
                                     pe = EmpiricalPrior(;
                                                         ce = PortfolioOptimisersCovariance(;
                                                                                            mp = MatrixProcessing(;
                                                                                                                  pdm = nothing))))
        @test vec(mean(norep, Rc)) ≈ vec(mean(Rc; dims = 1))
        res = optimise(OPS(; alg = ForecastReversion(; me = norep)), rdc)
        @test sum(res.w) ≈ 1 && all(isfinite, res.w)
        # A window bounds the rows the head holds for the adapter.
        wme = WindowedExpectedReturns(; me = pa, window = 10)
        @test po.rows_needed(ForecastReversion(; me = wme)) == 10
        @test vec(mean(wme, R)) ≈ vec(mean(pa, R[(end - 9):end, :]))
    end

    @testset "The fold-or-refit of a forecaster on the Rule State" begin
        # A folding forecaster is carried; the head holds the current row alone for it,
        # the row verbatim that the fold reads (ADR 0170).
        for me in (SimpleExpectedReturns(),
                   ExpWeightedExpectedReturns(; decay = 0.9, min_obs = 1),
                   PriceLevelExpectedReturns(; alg = ExponentialMovingAverage()))
            opt = OPS(; alg = ForecastReversion(; me = me))
            @test po.rows_needed(opt) == 1
            o = po.partial_fit!(opt, rows(rd, 1:9))
            @test o.cache.X.n == 1 && o.cache.X.max_history == 1
            @test isa(o.cache.st, po.ForecasterState)
            @test isnothing(po.online_entry_state(me))
            # The folded forecast equals the batch forecast over the same rows.
            folded = 1 .+ vec(mean(o.cache.st.me))
            batch = 1 .+ vec(mean(me, R[1:9, :]))
            @test maxerr(folded, batch) < 1e-13
            # The identity with the batch pass holds through the carrier.
            a = po.partial_fit!(o, rows(rd, 10:18))
            @test optimise(a).w == optimise(opt, rows(rd, 1:18)).w
            # A copy of the state aliases no array of the carrier.
            c = copy(o.cache)
            @test c.st.me.cache !== o.cache.st.me.cache
            # A view slices the carrier to the selected assets, where the forecaster has a
            # view of its own.
            v = po.port_opt_view(o.cache, [1, 3])
            if !isa(me, ExpWeightedExpectedReturns)
                @test length(vec(mean(v.st.me))) == 2
            end
            @test_throws ArgumentError po.merge_states(o.cache.st, c.st)
        end
        # A stateless forecaster is refit from the head's rows and carries nothing.
        for (me, need) in
            ((PriceLevelExpectedReturns(), 4), (MedianExpectedReturns(), nothing),
             (PriorExpectedReturns(), nothing))
            opt = OPS(; alg = ForecastReversion(; me = me))
            @test po.rows_needed(opt) == need
            o = po.partial_fit!(opt, rows(rd, 1:9))
            @test isnothing(o.cache.st)
            @test !isnothing(o.cache.X)
        end
        # A forecaster carrying a state at the door is refused: the head starts cold.
        warm = po.partial_fit!(SimpleExpectedReturns(), R[1:3, :])
        @test_throws ArgumentError optimise(OPS(; alg = ForecastReversion(; me = warm)), rd)
        # `Online(me)` in the slot is refused by name, on every forecast-reading rule.
        for f in (ForecastReversion, ForecastTracking, TransactionCostOptimisation,
                  ShortTermSparsePortfolio)
            @test_throws ArgumentError f(; me = Online(SimpleExpectedReturns()))
        end
        # A Prior refit on one row is undefined, so the first step holds; a flat forecast is
        # every rule's hold.
        one_row = optimise(OPS(; alg = ForecastReversion(; me = PriorExpectedReturns())),
                           rows(rd, 1:1))
        @test one_row.w ≈ fill(1 / N, N)
        @test po.flat_where_undefined([1.1, NaN, Inf]) == [1.1, 1.0, 1.0]
        var_me = OPS(; alg = ForecastReversion(; me = VarianceExpectedReturns()))
        @test optimise(var_me, rows(rd, 1:1)).w ≈ fill(1 / N, N)
        @test sum(optimise(var_me, rows(rd, 1:5)).w) ≈ 1
    end

    @testset "The reversion step, its scale and its constructors" begin
        # The scale is the identity when the scale statistic forecasts one everywhere.
        flat = ReturnsResult(; nx = nx, X = zeros(6, N))
        base = ForecastReversion(; me = SimpleExpectedReturns())
        scaled = ForecastReversion(; me = SimpleExpectedReturns(), scale = MovingAverage())
        @test optimise(OPS(; alg = base), flat).w == optimise(OPS(; alg = scaled), flat).w
        @test po.rows_needed(scaled) == 4 && po.rows_needed(base) == 1
        # On real data the preconditioner moves the step, and the answer stays feasible.
        ws = optimise(OPS(; alg = scaled), rd).w
        @test sum(ws) ≈ 1 && all(ws .>= 0)
        # One hand-written preconditioned step from the uniform start.
        w = fill(1 / N, N)
        set = po.resolve_allocation_set(BoundedAllocationSet(), N, false, Float64)
        rowsX = R[1:5, :]
        xh = 1 .+ vec(mean(SimpleExpectedReturns(), rowsX))
        D = 1 .+ vec(mean(PriceLevelExpectedReturns(), rowsX))
        dev = xh .- mean(xh)
        lam = max(0, (10 - dot(w, xh)) / sum(abs2, dev))
        q = w .+ lam .* D .* dev
        st = po.rule_state_seed(scaled, w)
        for t in 1:5
            st, w2 = po.online_update!(scaled, st, w, X[t, :], rows(rd, 1:t), set)
            t == 5 && @test w2 ≈ po.project_simplex(q)
        end
        # The constructors fill the paper's statistic and defaults.
        ema = ExponentialMovingAverageReversion()
        @test isa(ema.me.alg, ExponentialMovingAverage) &&
              ema.me.alg.alpha == 0.5 &&
              ema.eps == 10
        rprt = ReweightedPriceRelativeTracking()
        @test isa(rprt.me.alg, ReweightedPriceRelative) &&
              rprt.me.alg.theta == 0.7 &&
              rprt.eps == 50 &&
              rprt.scale == MovingAverage(; window = 5)
        gwr = GaussianWeightingReversion()
        @test isa(gwr.me.alg, GaussianWeightedDoubleEstimate) && gwr.eps == 50
        load = LocalAdaptiveLearning()
        @test isa(load.me.alg, TrendSwitch) &&
              isa(load.me.alg.test, RegressionSlope) &&
              isa(load.me.alg.rising, WindowPeak) &&
              load.me.alg.flat === load.me.alg.falling &&
              load.eps == 10 &&
              isnothing(po.rows_needed(load))
        @test isnothing(MovingAverageReversion().scale)
        @test_throws DomainError ForecastReversion(; eps = 0)
        # A view keeps the scale.
        @test po.port_opt_view(rprt, [1, 2]).scale == rprt.scale
    end

    @testset "The tracking step: one-hot at the paper's default, the ball below one" begin
        w = fill(1 / N, N)
        set = po.resolve_allocation_set(BoundedAllocationSet(), N, false, Float64)
        rowsX = R[1:5, :]
        xh = xhat(WindowPeak(), rowsX)
        dev = xh .- mean(xh)
        # The default saturates: all wealth on the asset with the largest centred forecast.
        ppt = PeakPriceTracking()
        @test ppt.eps == 100 && ppt.me.alg.window == 5
        _, w100 = po.online_update!(ppt, nothing, w, X[5, :], rows(rd, 1:5), set)
        @test w100 == (1:N .== argmax(dev))
        # Well below one the step has length `eps` before the projection and spreads.
        small = ForecastTracking(; eps = 0.05)
        _, ws = po.online_update!(small, nothing, w, X[5, :], rows(rd, 1:5), set)
        @test ws ≈ po.project_simplex(w .+ 0.05 .* dev ./ norm(dev))
        @test count(>(0), ws) > 1
        # A flat forecast holds.
        _, wh = po.online_update!(ForecastTracking(), nothing, w, X[5, :],
                                  ReturnsResult(; nx = nx, X = zeros(5, N)), set)
        @test wh == w
        # The constructors of the composite papers.
        aictr = AdaptiveInputCompositeTrend()
        @test isa(aictr.me.alg, CompositeTrend) &&
              aictr.eps == 1000 &&
              aictr.me.alg.sigma2 == 0.0025
        tppt = TrendPromotePriceTracking()
        @test isa(tppt.me.alg, TrendSwitch) &&
              isa(tppt.me.alg.test, PairwiseSlopeSum) &&
              isa(tppt.me.alg.rising, TruncatedExponentialMovingAverage) &&
              tppt.me.alg.flat == LaggedPrice(; lag = 0) &&
              tppt.eps == 100
        @test po.rows_needed(tppt) == 4
        @test_throws DomainError ForecastTracking(; eps = 0)
    end

    @testset "The cost-aware step and the sparse portfolio" begin
        w = [0.4, 0.3, 0.2, 0.1]
        set = po.resolve_allocation_set(BoundedAllocationSet(), N, false, Float64)
        x = X[5, :]
        what = po.price_adjusted_allocation(w, x)
        # TCO-1 reads the reciprocal of the last relative, and its soft threshold leaves
        # the drifted book alone when the cost is high.
        tco = TransactionCostOptimisation()
        @test isa(tco.me.alg, LaggedPrice) && tco.me.alg.lag == 1 && tco.eta == 10
        xh = 1 ./ x
        gg = xh ./ dot(what, xh)
        v = 10 .* (gg .- mean(gg))
        lam = 10 * 10 * 0.001
        q = what .+ sign.(v) .* max.(abs.(v) .- lam, 0)
        _, wt = po.online_update!(tco, nothing, w, x, rows(rd, 5:5), set)
        @test wt ≈ po.project_simplex(q)
        _, wexp = po.online_update!(TransactionCostOptimisation(; gamma = 1), nothing, w, x,
                                    rows(rd, 5:5), set)
        @test wexp ≈ what
        # TCO-2 reads the moving average.
        tco2 = TransactionCostOptimisation(; me = PriceLevelExpectedReturns())
        @test po.rows_needed(tco2) == 4
        @test_throws DomainError TransactionCostOptimisation(; gamma = -1)
        # The sparse portfolio: the paper's defaults, a feasible sparse answer, and the
        # iteration's own convergence.
        sspo = ShortTermSparsePortfolio()
        @test isa(sspo.me.alg, WindowPeak) &&
              sspo.lambda == 0.5 &&
              sspo.gamma == 0.01 &&
              sspo.eta == 0.005 &&
              sspo.zeta == 500 &&
              sspo.iters == 10_000 &&
              sspo.tol == 1e-4
        _, wsp = po.online_update!(sspo, nothing, w, x, rows(rd, 1:5), set)
        @test sum(wsp) ≈ 1 && all(wsp .>= 0)
        # The iterate solves the penalised programme's optimality up to the tolerance: the
        # scaled iterate projects to the asset with the largest generalised return.
        xw = xhat(WindowPeak(), R[1:5, :])
        @test argmax(wsp) == argmax(xw)
        @test_throws DomainError ShortTermSparsePortfolio(; zeta = 0)
        @test_throws DomainError po.online_update!(ShortTermSparsePortfolio(;
                                                                            me = CustomValueExpectedReturns(;
                                                                                                            val = fill(-2.0,
                                                                                                                       N))),
                                                   nothing, w, x, rows(rd, 1:5), set)
    end

    @testset "Every forecast-reading rule takes the head's verbs" begin
        for alg in (MovingAverageReversion(), ExponentialMovingAverageReversion(),
                    RobustMedianReversion(), ReweightedPriceRelativeTracking(),
                    GaussianWeightingReversion(), LocalAdaptiveLearning(), PeakPriceTracking(),
                    AdaptiveInputCompositeTrend(), TrendPromotePriceTracking(),
                    TransactionCostOptimisation(), ShortTermSparsePortfolio(),
                    ForecastReversion(; me = ExpWeightedExpectedReturns(; decay = 0.9, min_obs = 1)),
                    ForecastTracking(; me = SimpleExpectedReturns()))
            opt = OPS(; alg = alg)
            o = po.partial_fit!(opt, rows(rd, 1:10))
            o = po.partial_fit!(o, rows(rd, 11:17))
            o = po.partial_fit!(o, rows(rd, 18:18))
            a = optimise(o)
            b = optimise(opt, rows(rd, 1:18))
            @test a.w == b.w
            @test sum(a.w) ≈ 1 && all(a.w .>= -1e-12)
            @test isnothing(a.pr) && isa(a.retcode, OptimisationSuccess)
            # A bounded set is honoured in the rule's Euclidean geometry.
            capped = OPS(; alg = alg,
                         set = BoundedAllocationSet(;
                                                    wb = WeightBounds(; lb = 0, ub = 0.5)))
            wc = optimise(capped, rows(rd, 1:18)).w
            @test all(wc .<= 0.5 + 1e-12) && sum(wc) ≈ 1
            # A view of the head slices the rule and its carrier.
            v = po.port_opt_view(o, [1, 2, 4])
            @test length(optimise(v).w) == 3
            # A walk-forward at test_size = 1 through the online arm.
            cv = OnlineIndexWalkForward(5, 1)
            pred = cross_val_predict(opt, rows(rd, 1:12), cv)
            @test isapprox(pred.pred[1].res.w, optimise(opt, rows(rd, 1:5)).w; atol = 1e-14)
        end
    end

    @testset "The deviations from the papers, each pinned where the docstring states it" begin
        # The spatial median is not invariant to scaling each asset by its own price. The library
        # reads it on the path with every asset's last level at one; the paper's raw-price median
        # over the same window differs, and by more when one asset is quoted a hundred times
        # higher. The hand iteration is Vardi and Zhang's, written here.
        function l1median(Pw; tol = 1e-12, iters = 500)
            mu = vec(median(Pw; dims = 1))
            for _ in 1:iters
                num = zeros(size(Pw, 2))
                den = 0.0
                dir = zeros(size(Pw, 2))
                eta = 0
                for i in axes(Pw, 1)
                    d = norm(Pw[i, :] .- mu)
                    if iszero(d)
                        eta += 1
                    else
                        num .+= Pw[i, :] ./ d
                        den += 1 / d
                        dir .+= (Pw[i, :] .- mu) ./ d
                    end
                end
                iszero(den) && return mu
                g = norm(dir)
                mun = if iszero(eta)
                    num ./ den
                else
                    (if iszero(g)
                         mu
                     else
                         max(0, 1 - eta / g) .* (num ./ den) .+ min(1, eta / g) .* mu
                     end)
                end
                norm(mun .- mu) < tol && return mun
                mu = mun
            end
            return mu
        end
        Y = R[1:4, :]
        Pn = levels(X[1:4, :])                      # the normalised path, last row ones
        praw = cumprod(X[1:4, :]; dims = 1)         # raw prices from p_0 = 1
        Praw = vcat(ones(1, N), praw)
        xs = xhat(SpatialMedian(), Y)
        @test isapprox(xs, l1median(Pn); atol = 1e-8)
        @test maxerr(xs, l1median(Praw) ./ Praw[end, :]) > 1e-6
        S = [100.0, 1, 1, 1]
        @test maxerr(xs, l1median(Praw .* S') ./ (Praw[end, :] .* S)) > 1e-3
        # The iteration runs to its minimiser: the optimality residual is at machine precision,
        # where the paper's stop — a relative L1 change of 1e-3 — leaves it far from zero.
        resid(mu) = norm(sum((Pn[i, :] .- mu) ./ norm(Pn[i, :] .- mu) for i in axes(Pn, 1)))
        mlib = po.spatial_median(Pn, 100, 1e-8)
        @test resid(mlib) < 1e-6
        mu = vec(median(Pn; dims = 1))
        mpaper = mu
        for _ in 2:200
            # One Weiszfeld step from `mu`, then the paper's stop.
            num = zeros(N)
            den = 0.0
            for i in axes(Pn, 1)
                d = norm(Pn[i, :] .- mu)
                iszero(d) && continue
                num .+= Pn[i, :] ./ d
                den += 1 / d
            end
            mun = num ./ den
            mpaper = mun
            norm(mu .- mun, 1) <= 1e-3 * norm(mun, 1) && break
            mu = mun
        end
        @test resid(mpaper) > 1e-4 && maxerr(mpaper, mlib) > 1e-5
        # The moving-average reversion admits the two-level window and a threshold at or
        # below one, which the paper's algorithm excludes and the step is defined for.
        @test MovingAverageReversion(; window = 2, eps = 1).me.alg.window == 2
        @test_throws DomainError MovingAverageReversion(; window = 1)
        # The local adaptive learning takes the solution of the paper's programme, the
        # squared norm in the step length, not the printed one.
        w = fill(1 / N, N)
        set = po.resolve_allocation_set(BoundedAllocationSet(), N, false, Float64)
        load = LocalAdaptiveLearning(; eps = 1.05)
        xl = 1 .+ vec(mean(load.me, R[1:5, :]))
        dev = xl .- mean(xl)
        _, wl = po.online_update!(load, nothing, w, X[5, :], rows(rd, 1:5), set)
        @test wl ≈
              po.project_simplex(w .+ max(0, (1.05 - dot(w, xl)) / sum(abs2, dev)) .* dev)
        @test !(wl ≈
                po.project_simplex(w .+ max(0, (1.05 - dot(w, xl)) / norm(dev)) .* dev))
        # The sparse portfolio stops at the paper's tolerance on the budget residual, which the
        # iteration meets at a zero crossing of the residual long before its iterate settles:
        # the library's answer is the projection of that early iterate, which is still tenths
        # away from the iterate `iters` steps later. The iteration is written out here.
        w4 = [0.4, 0.3, 0.2, 0.1]
        xw = xhat(WindowPeak(), R[7:11, :])
        phi = -1.1 .* log.(xw) .- 1
        a = 0.5 / 0.01
        b = copy(w4)
        g = copy(w4)
        rho = 0.0
        bstop = nothing
        kstop = 0
        for o in 1:10_000
            rhs = a .* g .+ (0.005 - rho) .- phi
            b = rhs ./ a .- 0.005 * sum(rhs) / (a * (a + 0.005 * N))
            g = sign.(b) .* max.(abs.(b) .- 0.01, 0)
            res = sum(b) - 1
            rho += 0.005 * res
            if isnothing(bstop) && abs(res) < 1e-4
                bstop = copy(b)
                kstop = o
            end
        end
        _, wsp = po.online_update!(ShortTermSparsePortfolio(), nothing, w4, X[11, :],
                                   rows(rd, 7:11), set)
        @test wsp == po.project_simplex(500 .* bstop)
        @test kstop < 1000 && maxerr(bstop, b) > 0.5
    end

    @testset "Show and the search seam" begin
        @test occursin("ForecastReversion",
                       sprint(show, MIME("text/plain"), MovingAverageReversion()))
        @test !occursin("cache",
                        sprint(show, MIME("text/plain"), PriceLevelExpectedReturns()))
        @test occursin("scale",
                       sprint(show, MIME("text/plain"), ReweightedPriceRelativeTracking()))
        @test occursin("PriorExpectedReturns",
                       sprint(show, MIME("text/plain"), PriorExpectedReturns()))
        # The window is addressable as the constructor's docstring says.
        alg = MovingAverageReversion()
        @test alg.me.alg.window == 5
    end
end
