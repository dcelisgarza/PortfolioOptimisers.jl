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
        # #1172: the default is the value of the authors' code, which seeds the forecast at
        # one where the restating paper, and the library, seed it at the first relative. The
        # two differ by 0.024 after the first row and by less than 1e-4 after ten rows.
        @test ReweightedPriceRelative().theta == 0.8
        phi_code = ones(N)
        gaps = map(1:10) do t
            g = 0.8 .* X[t, :] ./ (0.8 .* X[t, :] .+ phi_code)
            phi_code = g .+ (1 .- g) .* phi_code ./ X[t, :]
            return maxerr(phi_code, xhat(ReweightedPriceRelative(), R[1:t, :]))
        end
        @test 0.02 < gaps[1] < 0.03 && gaps[10] < 1e-4
        # At `alpha = 1` the exponential moving average forecasts one in every asset.
        @test xhat(ExponentialMovingAverage(; alpha = 1), R) == ones(N)
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

    @testset "The composite statistics' docstrings, with numbers (#1188)" begin
        rng3 = StableRNG(1188)
        P = levels(1 .+ 0.02 .* randn(rng3, 30, 6))
        D = exp.(randn(rng3, 6))
        pls(alg, Q) = po.price_level_statistic(alg, Q)
        # Every statistic of the file reads each asset alone, so it scales with each asset.
        for alg in (TruncatedExponentialMovingAverage(), GaussianWeightedDoubleEstimate(),
                    TrendSwitch(),
                    TrendSwitch(; test = RegressionSlope(), rising = WindowPeak(),
                                flat = ExponentialMovingAverage(),
                                falling = ExponentialMovingAverage()), CompositeTrend())
            @test pls(alg, P .* D') ≈ pls(alg, P) .* D
        end
        # The truncated average: the weights sum to 0.96875 at the defaults, so a flat path
        # forecasts below one; at alpha = 1 the forecast is one; Float32 stays Float32.
        tema = TruncatedExponentialMovingAverage()
        @test pls(tema, ones(7, 2)) ≈ fill(0.96875, 2)
        @test pls(TruncatedExponentialMovingAverage(; alpha = 1.0), P) == P[end, :]
        @test eltype(pls(TruncatedExponentialMovingAverage(; alpha = 0.5f0), Float32.(P))) ==
              Float32
        # The Gaussian double estimate on a full window: eq. (3) gives l = 9, and the
        # statistic reads l + 1 levels.
        gw = GaussianWeightedDoubleEstimate()
        @test floor(Int, sqrt(-2 * 2.8^2 * log(0.005))) == 9
        g = [exp(-k^2 / (2 * 2.8^2)) for k in 1:9]
        K = size(P, 1)
        est(u) = sum(g[k] .* P[u - k + 1, :] for k in 1:9) ./ sum(g)
        p2 = (g[1] .* est(K - 1) .+ sum(g[k] .* P[K - k + 1, :] for k in 2:9)) ./ sum(g)
        @test pls(gw, P) ≈ (est(K) .+ p2) ./ 2
        @test pls(gw, P) == pls(gw, P[(end - 9):end, :])
        @test eltype(pls(GaussianWeightedDoubleEstimate(; tau = 2.8f0, cutoff = 0.005f0),
                         Float32.(P))) == Float32
        # The pairwise test sums ten slopes; the four slopes from the current level that
        # eq. (6) of the paper writes can have the other sign.
        ten(Q, i) = sign(sum((Q[b, i] - Q[a, i]) / (b - a) for a in 1:4 for b in (a + 1):5))
        four(Q, i) = sign(sum((Q[5, i] - Q[a, i]) / (5 - a) for a in 1:4))
        Qs = [levels(1 .+ 0.02 .* randn(rng3, 4, 1)) for _ in 1:200]
        @test all(Q -> po.trend_sign(PairwiseSlopeSum(), Q)[1] == ten(Q, 1), Qs)
        @test any(Q -> four(Q, 1) != ten(Q, 1), Qs)
        # The regression test is the least-squares slope with a free intercept at
        # lambda = 0; a slope equal to the threshold is flat, and one below it falls.
        Q = P[(end - 4):end, :]
        ols = [([ones(5) 1:5] \ Q[:, i])[2] for i in 1:6]
        @test po.trend_sign(RegressionSlope(; threshold = 0), Q) == sign.(ols)
        lin = reshape(collect(1.0:5.0), :, 1)
        @test po.trend_sign(RegressionSlope(; threshold = 1), lin) == [0]
        @test po.trend_sign(RegressionSlope(; threshold = 1.5), lin) == [-1]
        # Over one level no trend exists, and both tests return zero whatever the ridge
        # weight; a Float32 window gives a Float32 sign.
        for test in (PairwiseSlopeSum(), RegressionSlope(), RegressionSlope(; lambda = 1))
            @test po.trend_sign(test, P[end:end, :]) == zeros(6)
        end
        @test eltype(po.trend_sign(RegressionSlope(; threshold = 0.1f0, lambda = 0.0f0),
                                   Float32.(Q))) == Float32
        # The switch takes each branch per asset on the sign of its test.
        s = po.trend_sign(PairwiseSlopeSum(), Q)
        up = pls(TruncatedExponentialMovingAverage(), Q)
        pk = vec(maximum(Q; dims = 1))
        @test pls(TrendSwitch(), P) ≈
              [s[i] > 0 ? up[i] : s[i] < 0 ? pk[i] : Q[end, i] for i in 1:6]
        lal = TrendSwitch(; test = RegressionSlope(), rising = WindowPeak(),
                          flat = ExponentialMovingAverage(),
                          falling = ExponentialMovingAverage())
        sl = po.trend_sign(RegressionSlope(), Q)
        ema = pls(ExponentialMovingAverage(), P)
        @test pls(lal, P) ≈ [sl[i] > 0 ? pk[i] : ema[i] for i in 1:6]
        # The composite against its definition: the trend portfolio for period t - k is the
        # projected forecast made after the row of t - k - 1, scored on the relative of
        # t - k; the centre has the best worst return, and the weights are normalised.
        ct = CompositeTrend()
        function by_definition(Q)
            n = size(Q, 1)
            fc(l, u) = po.member_statistic(ct.trends[l], Q, u) ./ Q[u, :]
            xh = [fc(l, n) for l in 1:3]
            xt = po.project_simplex.(xh)
            # `scores`, not `R`: an assignment in a closure writes the testset's `R`.
            scores = [[dot(po.project_simplex(fc(l, n - k - 1)),
                           Q[n - k, :] ./ Q[n - k - 1, :]) for k in 0:4 if n - k - 1 >= 1]
                      for l in 1:3]
            star = isempty(scores[1]) ? 1 : argmax(minimum.(scores))
            phi = [exp(-sum(abs2, xt[star] .- xt[l]) / (2 * ct.sigma2)) for l in 1:3]
            return sum(phi[l] .* xh[l] for l in 1:3) ./ sum(phi) .* Q[n, :]
        end
        for n in (2, 3, 6, 15, 31)
            @test pls(ct, P[(end - n + 1):end, :]) ≈ by_definition(P[(end - n + 1):end, :])
        end
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

    @testset "The Prior adapter against its definition" begin
        # The adapter's mean is the mean of the prior fitted on the same rows, at both
        # orientations, for every family of prior it admits. The ensemble prior draws its
        # subsystems at each fit, so a fixed seed makes two fits comparable.
        h = 21
        for pe in (EmpiricalPrior(), EmpiricalPrior(; horizon = h),
                   EmpiricalPrior(; me = ShrunkExpectedReturns()), HighOrderPriorEstimator(),
                   EntropyPoolingPrior(), LowDimensionEnsemblePrior(; seed = 7))
            me = PriorExpectedReturns(; pe = pe)
            mu = vec(prior(pe, R).mu)
            @test vec(mean(me, R)) == mu
            @test maxerr(vec(mean(me, permutedims(R); dims = 2)), mu) < 1e-16
            @test vec(mean(me, R, nothing)) == mu
        end
        # Under the horizon arm, the mean is the arithmetic mean of the lognormal law at the
        # horizon, exp(h m + h s² / 2) - 1, with m and s² the moments of the log-returns.
        Rl = log1p.(R)
        lognormal = exp.(h .* vec(mean(Rl; dims = 1)) .+ h .* diag(cov(Rl)) ./ 2) .- 1
        @test maxerr(vec(mean(PriorExpectedReturns(; pe = EmpiricalPrior(; horizon = h)),
                              R)), lognormal) < 1e-15
        # A direct fold of the default prior, one row and then a block, reads the batch mean
        # of every row folded, although a host does not fold it.
        f = partial_fit!(partial_fit!(PriorExpectedReturns(), R[1, :]), R[2:end, :])
        @test isa(f, PriorExpectedReturns)
        @test size(mean(f)) == (1, N)
        @test maxerr(vec(mean(f)), vec(mean(R; dims = 1))) < 1e-16
        @test !po.supports_partial_fit(f)
        # The no-data form refuses a prior that folded nothing, and a factor leaf under an
        # optional-argument host is refused at construction.
        @test_throws ArgumentError mean(PriorExpectedReturns())
        @test_throws ArgumentError PriorExpectedReturns(;
                                                        pe = EntropyPoolingPrior(;
                                                                                 pe = FactorPrior()))
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
              rprt.me.alg.theta == 0.8 &&
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
        # The sparse portfolio: the paper's defaults on the rule and on each algorithm, and a
        # feasible sparse answer from each algorithm.
        sspo = ShortTermSparsePortfolio()
        @test isa(sspo.me.alg, WindowPeak) && sspo.alg == HuberOptimum() && sspo.zeta == 500
        hub = HuberOptimum()
        @test hub.lambda == 0.5 && hub.gamma == 0.01
        adm = AlternatingDirectionMethod()
        @test adm.lambda == 0.5 &&
              adm.gamma == 0.01 &&
              adm.eta == 0.005 &&
              adm.iters == 10_000 &&
              adm.tol == 1e-4
        # Below 1 / gamma assets every algorithm's scaled iterate projects to the asset with
        # the largest generalised return.
        xw = xhat(WindowPeak(), R[1:5, :])
        for alg in (L1Optimum(), hub, adm)
            _, wsp = po.online_update!(ShortTermSparsePortfolio(; alg = alg), nothing, w, x,
                                       rows(rd, 1:5), set)
            @test sum(wsp) ≈ 1 && all(wsp .>= 0)
            @test argmax(wsp) == argmax(xw)
        end
        @test po.port_opt_view(ShortTermSparsePortfolio(; alg = adm), [1, 2]).alg == adm
        @test_throws DomainError ShortTermSparsePortfolio(; zeta = 0)
        @test_throws DomainError HuberOptimum(; lambda = 0)
        @test_throws DomainError HuberOptimum(; gamma = 0)
        @test_throws DomainError AlternatingDirectionMethod(; lambda = 0)
        @test_throws DomainError AlternatingDirectionMethod(; gamma = 0)
        @test_throws DomainError AlternatingDirectionMethod(; eta = 0)
        @test_throws DomainError AlternatingDirectionMethod(; iters = 0)
        @test_throws DomainError AlternatingDirectionMethod(; tol = 0)
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
                    ShortTermSparsePortfolio(; alg = L1Optimum()),
                    ShortTermSparsePortfolio(; alg = AlternatingDirectionMethod()),
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
        # #1172: the numbers `SpatialMedian` states, over every five-level window of the
        # fixture. The authors' code rebases each asset to one on the first day; the paper
        # reads raw prices, here one asset quoted a hundred times higher.
        tight(P) = po.spatial_median(P, 100_000, 1e-15)
        wins = [levels(X[s:(s + 3), :]) for s in 1:(T - 4)]
        rebased(s, S) = wins[s] .* (vec(prod(X[1:(s + 3), :]; dims = 1)) .* S)'
        gap(S) = maximum(maxerr(tight(rebased(s, S)) ./ rebased(s, S)[end, :],
                                tight(wins[s])) for s in eachindex(wins))
        @test 1e-3 < gap(ones(N)) < 3e-3
        @test 5e-2 < gap([100.0, 1, 1, 1]) < 8e-2
        # The authors' code stops at a relative L1 change of 1e-9 against the old iterate, or
        # after 200 iterations; the library's defaults are as close to the minimiser.
        function code_median(P; tol = 1e-9, maxiter = 200)
            y = vec(median(P; dims = 1))
            for _ in 1:maxiter
                num, den, Rn, eta = po.weiszfeld_sums(P, y)
                Ty = if iszero(eta)
                    num ./ den
                else
                    max(0, 1 - eta / Rn) .* (num ./ den) .+ min(1, eta / Rn) .* y
                end
                stop = norm(Ty .- y, 1) <= tol * norm(y, 1)
                y = Ty
                stop && break
            end
            return y
        end
        @test maximum(maxerr(po.spatial_median(P, 100, 1e-8), tight(P)) for P in wins) <
              3e-8
        @test maximum(maxerr(code_median(P), tight(P)) for P in wins) < 3e-8
        # A median on a level is found exactly by the optimality test at the levels; the third
        # window's median is its second level, which the iteration alone approaches from 4e-7.
        m3 = po.spatial_median(wins[3], 100, 1e-8)
        @test any(k -> m3 == wins[3][k, :], 1:5)
        # A seed on a level that is not the median takes the modified step off the level.
        Pv = [0.0 0.0; 1.0 0.0; 0.0 1.0; 5.0 5.0; 0.2 0.1]
        @test vec(median(Pv; dims = 1)) == Pv[5, :]
        mv = po.spatial_median(Pv, 1000, 1e-14)
        @test po.weiszfeld_sums(Pv, mv)[3] < 1e-8 && mv != Pv[5, :]
        # A median very near a level but not on it converges slowly: the cap of 100 stops the
        # forecast about 1e-4 away while the sum of distances is within 1e-6 of its minimum.
        Ps = levels((1 .+ 0.02 .* randn(StableRNG(15), 40, 4))[6:9, :])
        cost(y) = sum(norm(Ps[i, :] .- y) for i in axes(Ps, 1))
        ms = po.spatial_median(Ps, 100, 1e-8)
        @test 1e-4 < maxerr(ms, tight(Ps)) < 2e-4
        @test cost(ms) - cost(tight(Ps)) < 1e-6
        # A step below `tol = 1e-8` stops it too, so a larger cap alone does not help.
        @test maxerr(po.spatial_median(Ps, 100_000, 1e-8), tight(Ps)) > 1e-6
        @test maxerr(po.spatial_median(Ps, 10_000, 1e-15), tight(Ps)) < 1e-8
        # The moving-average reversion admits the two-level window and a threshold at or
        # below one, which the paper's algorithm excludes and the step is defined for.
        @test MovingAverageReversion(; window = 2, eps = 1).me.alg.window == 2
        @test_throws DomainError MovingAverageReversion(; window = 1)
        # The local adaptive learning takes the step of the paper's eq. (12), whose step
        # length has the squared norm of the centred forecast, and not the plain norm.
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
        # The paper's iteration stops at the paper's tolerance on the budget residual, which
        # it meets at a zero crossing of the residual long before its iterate settles: its
        # answer is the projection of that early iterate, which is still tenths away from the
        # iterate `iters` steps later. The iteration is written out here.
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
        _, wsp = po.online_update!(ShortTermSparsePortfolio(;
                                                            alg = AlternatingDirectionMethod()),
                                   nothing, w4, X[11, :], rows(rd, 7:11), set)
        @test wsp == po.project_simplex(500 .* bstop)
        @test kstop < 1000 && maxerr(bstop, b) > 0.5
        # #1259: the fixed point is the optimum of the coupled programme the docstring states,
        # not of the paper's. Its closed form gives the asset at `min φ` the remainder and
        # every other asset `(λ − (φᵢ − min φ)) / (λ / γ)`.
        for _ in 10_001:1_000_000
            rhs = a .* g .+ (0.005 - rho) .- phi
            b = rhs ./ a .- 0.005 * sum(rhs) / (a * (a + 0.005 * N))
            g = sign.(b) .* max.(abs.(b) .- 0.01, 0)
            rho += 0.005 * (sum(b) - 1)
        end
        kmin = argmin(phi)
        b2 = (0.5 .- (phi .- phi[kmin])) ./ a
        b2[kmin] = 1 - (sum(b2) - b2[kmin])
        @test maximum(phi) - minimum(phi) <= 2 * 0.5
        @test isapprox(b, b2; atol = 1e-8)
        @test maxerr(b, g) ≈ 0.01
        # `HuberOptimum` is that fixed point in closed form, and the default rule projects it.
        bh = po.sparse_portfolio_iterate(HuberOptimum(), phi, w4)
        @test isapprox(bh, b; atol = 1e-8) && sum(bh) ≈ 1
        _, wh = po.online_update!(ShortTermSparsePortfolio(), nothing, w4, X[11, :],
                                  rows(rd, 7:11), set)
        @test wh == po.project_simplex(500 .* bh)
    end

    @testset "The sparse portfolio's three algorithms" begin
        # The stated programme's optimum is the vertex of the largest forecast, whatever the
        # held allocation, and no feasible point does better.
        phi = [-1.02, -1.10, -1.05, -1.01]
        w4 = [0.4, 0.3, 0.2, 0.1]
        e2 = [0.0, 1.0, 0.0, 0.0]
        @test po.sparse_portfolio_iterate(L1Optimum(), phi, w4) == e2
        p1(b) = dot(b, phi) + 0.5 * norm(b, 1)
        rng1 = StableRNG(4)
        for _ in 1:200
            v = randn(rng1, 4)
            v .+= (1 - sum(v)) / 4
            @test p1(v) >= p1(e2)
        end
        # Ties split the budget evenly; a flat objective is the equal weights for both closed
        # forms, and so is the fixed point.
        @test po.sparse_portfolio_iterate(L1Optimum(), [-1.1, -1.0, -1.1], w4[1:3]) ==
              [0.5, 0.0, 0.5]
        @test po.sparse_portfolio_iterate(L1Optimum(), fill(-1.0, 4), w4) == fill(0.25, 4)
        @test po.sparse_portfolio_iterate(HuberOptimum(), fill(-1.0, 4), w4) ≈ fill(0.25, 4)
        # Past the bound both programmes fall without end, and the closed forms take the
        # limit: the budget on the largest forecast.
        wide = [-1.0, -2.2, -1.5]
        @test maximum(wide) - minimum(wide) > 2 * 0.5
        @test po.sparse_portfolio_iterate(HuberOptimum(), wide, w4[1:3]) == [0.0, 1.0, 0.0]
        # With more than 1 / gamma assets the multiplier is a root inside its interval. The
        # optimality conditions of the coupled programme hold: a coordinate inside the clamp
        # reads the same multiplier off `a bᵢ = −(φᵢ + ν)`, and a clamped one sits on
        # `|φᵢ + ν| = λ` with its sign.
        function huber_kkt(alg, phi)
            b = po.sparse_portfolio_iterate(alg, phi, fill(1 / length(phi), length(phi)))
            a = alg.lambda / alg.gamma
            inside = abs.(b) .< alg.gamma - 1e-12
            nu = -(phi[inside] .+ a .* b[inside])
            nu0 = first(nu)
            atedge = .!inside
            return b, maxerr(nu, fill(nu0, length(nu))),
                   maximum(abs, -(phi[atedge] .+ nu0) .- alg.lambda .* sign.(b[atedge]);
                           init = 0.0), count(atedge)
        end
        rng2 = StableRNG(5)
        phin = -(1.1 .* log.(1 .+ max.(0, 0.05 .* randn(rng2, 150))) .+ 1)
        bn, dnu, dedge, nedge = huber_kkt(HuberOptimum(), phin)
        @test sum(bn) ≈ 1 && dnu < 1e-12 && dedge < 1e-12 && nedge == 0
        @test count(>(0), po.project_simplex(500 .* bn)) > 1
        @test po.project_simplex(500 .*
                                 po.sparse_portfolio_iterate(L1Optimum(), phin, phin)) ==
              Float64.(1:150 .== argmin(phin))
        # At the upper end of the interval the asset with the smallest forecast takes the
        # negative remainder, a short past `−gamma`.
        phih = vcat(zeros(149), 0.9)
        bh, dnu, dedge, nedge = huber_kkt(HuberOptimum(), phih)
        @test sum(bh) ≈ 1 && bh[end] < -0.01 && dnu < 1e-12 && dedge < 1e-12 && nedge == 1
        # The paper's iteration reads the held allocation: one iteration from two seeds
        # differs.
        one_step = AlternatingDirectionMethod(; iters = 1)
        @test po.sparse_portfolio_iterate(one_step, phi, w4) !=
              po.sparse_portfolio_iterate(one_step, phi, fill(0.25, 4))
    end

    @testset "The sparse portfolio's docstrings, with numbers (#1266)" begin
        # `HuberOptimum`: the minimum over `g` is the soft threshold, and it leaves the Huber
        # penalty `h`, whose derivative is `clamp(a b, −λ, λ)`.
        lam, gam = 0.5, 0.01
        a = lam / gam
        h(b) = abs(b) <= gam ? a / 2 * b^2 : lam * abs(b) - lam * gam / 2
        for b in range(-0.05, 0.05; length = 41)
            g = sign(b) * max(abs(b) - gam, 0)
            @test lam * abs(g) + a / 2 * (b - g)^2 ≈ h(b) atol = 1e-15
            @test all(lam * abs(u) + a / 2 * (b - u)^2 >= h(b) - 1e-15
                      for u in range(-0.1, 0.1; length = 201))
            # A central difference across the kink at `|b| = γ` is off by `a db / 4`.
            db = 1e-7
            @test (h(b + db) - h(b - db)) / (2 * db) ≈ clamp(a * b, -lam, lam) atol = 1e-5
        end
        # `AlternatingDirectionMethod`: the Sherman-Morrison line is the inverse.
        v = randn(StableRNG(3), 6)
        @test (a * I + 0.005 * ones(6, 6)) \ v ≈
              v ./ a .- 0.005 * sum(v) / (a * (a + 0.005 * 6))
        # The numbers of the docstring, on the fixture of this file from the uniform seed: the
        # stop, the sign changes before it, the distance there, and the iterations to within
        # 1e-6 of the fixed point, over the 36 windows of five rows.
        function adm_trace(phi, w)
            bh = po.sparse_portfolio_iterate(HuberOptimum(), phi, w)
            b = copy(w)
            g = copy(w)
            rho = 0.0
            prev = NaN
            nsign = 0
            kstop = 0
            dist = NaN
            kconv = 0
            o = 0
            while kstop == 0 || kconv == 0
                o += 1
                rhs = a .* g .+ (0.005 - rho) .- phi
                b = rhs ./ a .- 0.005 * sum(rhs) / (a * (a + 0.005 * length(w)))
                g = sign.(b) .* max.(abs.(b) .- gam, 0)
                res = sum(b) - 1
                rho += 0.005 * res
                if kstop == 0
                    nsign += !isnan(prev) && sign(res) != sign(prev)
                    prev = res
                    if abs(res) < 1e-4
                        kstop = o
                        dist = maxerr(b, bh)
                        # The library stops where the trace stops.
                        @test b ==
                              po.sparse_portfolio_iterate(AlternatingDirectionMethod(), phi,
                                                          w)
                    end
                end
                if kconv == 0 && maxerr(b, bh) < 1e-6
                    kconv = o
                end
            end
            return kstop, nsign, dist, kconv
        end
        traces = [adm_trace(-1.1 .* log.(xhat(WindowPeak(), R[(t - 4):t, :])) .- 1,
                            fill(0.25, N)) for t in 5:T]
        @test length(traces) == 36
        @test extrema(first.(traces)) == (359, 4619)
        @test extrema(getindex.(traces, 2)) == (2, 27)
        @test isapprox(maximum(getindex.(traces, 3)), 0.67; atol = 0.005)
        @test extrema(getindex.(traces, 4)) == (29461, 104386)
        # Two assets with almost equal forecasts: the stopped iterate and the fixed point
        # project to different assets.
        R1 = 0.02 .* randn(StableRNG(1), 60, 4)
        phi1 = -1.1 .* log.(xhat(WindowPeak(), R1[2:6, :])) .- 1
        pa = po.project_simplex(500 .*
                                po.sparse_portfolio_iterate(AlternatingDirectionMethod(),
                                                            phi1, fill(0.25, 4)))
        ph = po.project_simplex(500 .* po.sparse_portfolio_iterate(HuberOptimum(), phi1,
                                                                   fill(0.25, 4)))
        @test isapprox(pa, [0.46, 0, 0, 0.54]; atol = 0.005)
        @test ph == [0.0, 0.0, 0.0, 1.0]
        @test abs(phi1[1] - phi1[4]) < 1e-4
    end

    @testset "The forecast rules' docstrings, with numbers (#1190)" begin
        set = po.resolve_allocation_set(BoundedAllocationSet(), N, false, Float64)
        u = fill(1 / N, N)
        # `ForecastReversion`: without a preconditioner the raw step stays on the budget
        # hyperplane and reaches the target. With one it leaves both, and the projection
        # restores the budget.
        xh = 1 .+ vec(mean(SimpleExpectedReturns(), R[1:5, :]))
        D = 1 .+ vec(mean(PriceLevelExpectedReturns(), R[1:5, :]))
        dev = xh .- mean(xh)
        lam = max(0, (10 - dot(u, xh)) / sum(abs2, dev))
        q0 = u .+ lam .* dev
        q1 = u .+ lam .* D .* dev
        @test lam > 0
        @test sum(q0) ≈ 1 && dot(q0, xh) ≈ 10
        @test abs(sum(q1) - 1) > 1 && abs(dot(q1, xh) - 10) > 1
        # The reweighted price relative tracking takes the defaults of the authors' code. On
        # the fixture the step moves at every row, and from the second row on one asset holds
        # more than 0.01.
        rprt = ReweightedPriceRelativeTracking()
        @test rprt.me.alg.theta == 0.8 && rprt.eps == 50 && rprt.scale.window == 5
        w = copy(u)
        st = po.rule_state_seed(rprt, w)
        for t in 1:T
            rw = rows(rd, max(1, t - 3):t)
            _, xr = po.forecast_relative(rprt.me, deepcopy(st), X[t, :], rw)
            @test dot(w, xr) < rprt.eps
            st, w = po.online_update!(rprt, st, w, X[t, :], rw, set)
            t > 1 && @test count(>(0.01), w) == 1
        end
        # With fewer rows than its window, the preconditioner is the moving average of the
        # levels that the head holds.
        @test po.scale_relative(MovingAverage(), X[2, :], rows(rd, 1:2)) ≈
              xhat(MovingAverage(; window = 3), R[1:2, :])
        # `ForecastTracking`: the projection is one-hot whenever `eps` times the gap of the
        # unit direction is at least two, from any book on the simplex. From a book on the
        # second asset, a gap of 1.5 leaves two assets.
        rng = StableRNG(3)
        for _ in 1:500
            w0 = po.project_simplex(randn(rng, N))
            d = randn(rng, N)
            d .-= mean(d)
            d ./= norm(d)
            s = sort(d; rev = true)
            ep = 2 * (1 + rand(rng)) / (s[1] - s[2])
            ft = ForecastTracking(; me = CustomValueExpectedReturns(; val = d), eps = ep)
            _, w1 = po.online_update!(ft, nothing, w0, ones(N), nothing, set)
            @test w1 == (1:N .== argmax(d))
        end
        d = [0.6, 0.0, -0.3, -0.3]
        d ./= norm(d)
        ft = ForecastTracking(; me = CustomValueExpectedReturns(; val = d),
                              eps = 1.5 / (d[1] - d[2]))
        _, w1 = po.online_update!(ft, nothing, [0.0, 1.0, 0.0, 0.0], ones(N), nothing, set)
        @test w1 ≈ [0.75, 0.25, 0, 0]
        # `KernelTrendTracking`: the same bound on the kernel-scaled forecast is `2 / eta`.
        # From a book on the second asset, a gap of `1.5 / eta` leaves 0.75 and 0.25.
        wk = [0.0, 1.0, 0.0, 0.0]
        a = 0.00332336168084042
        dk = [a, 0.0, -a / 2, -a / 2]
        Kk = exp.(-abs.((wk .- mean(wk)) .- dk) .^ (1 / 6))
        @test isapprox(Kk[1] * dk[1] - Kk[2] * dk[2], 1.5e-3; atol = 1e-6)
        ktt = KernelTrendTracking(; me = CustomValueExpectedReturns(; val = dk))
        _, w2 = po.online_update!(ktt, nothing, wk, ones(N), nothing, set)
        @test isapprox(w2, [0.75, 0.25, 0, 0]; atol = 1e-4)
        rng = StableRNG(4)
        for _ in 1:500
            w0 = po.project_simplex(randn(rng, N))
            dk = 0.05 .* randn(rng, N)
            dk .-= mean(dk)
            sk = exp.(-abs.((w0 .- mean(w0)) .- dk) .^ (1 / 6)) .* dk
            s = sort(sk; rev = true)
            et = 2 * (1 + rand(rng)) / (s[1] - s[2])
            ktt = KernelTrendTracking(; me = CustomValueExpectedReturns(; val = dk),
                                      eta = et)
            _, w1 = po.online_update!(ktt, nothing, w0, ones(N), nothing, set)
            @test w1 == (1:N .== argmax(sk))
        end
        # `TransactionCostOptimisation`: the soft threshold at `10 eta gamma` minimises the
        # proximal programme with the budget multiplier at the mean gradient.
        wt = [0.4, 0.3, 0.2, 0.1]
        x = X[5, :]
        what = po.price_adjusted_allocation(wt, x)
        g = (1 ./ x) ./ dot(what, 1 ./ x)
        c = g .- mean(g)
        dt = sign.(10 .* c) .* max.(abs.(10 .* c) .- 10 * 0.01, 0)
        f(v) = -dot(c, v) + sum(abs2, v) / (2 * 10) + 0.01 * sum(abs, v)
        rng = StableRNG(5)
        @test all(f(dt .+ 1e-3 .* randn(rng, N)) >= f(dt) for _ in 1:500)
        _, wt2 = po.online_update!(TransactionCostOptimisation(), nothing, wt, x,
                                   rows(rd, 5:5), set)
        @test wt2 ≈ po.project_simplex(what .+ dt)
        # `ShortTermSparsePortfolio`: the fixed point gives every asset but the largest
        # forecast at most `gamma`.
        rng = StableRNG(6)
        for _ in 1:200
            n = rand(rng, 2:60)
            phi = -(1.1 .* log.(1 .+ 0.1 .* rand(rng, n)) .+ 1)
            b = po.sparse_portfolio_iterate(HuberOptimum(), phi, fill(1 / n, n))
            @test maximum(b[setdiff(1:n, argmin(phi))]) <= 0.01 + 1e-12
        end
        # The rows each constructor needs, as its docstring states.
        @test po.rows_needed(ExponentialMovingAverageReversion()) == 1
        @test po.rows_needed(KernelTrendPatternTracking()) == 1
        @test isnothing(po.rows_needed(AdaptiveInputCompositeTrend()))
        @test isnothing(po.rows_needed(LocalAdaptiveLearning()))
        # The defaults each docstring attributes to its paper.
        rmr = RobustMedianReversion()
        @test rmr.eps == 5 && rmr.me.alg.window == 5
        gwr = GaussianWeightingReversion()
        @test gwr.me.alg.tau == 2.8 && gwr.me.alg.cutoff == 0.005
        load = LocalAdaptiveLearning()
        @test load.me.alg.test.window == 5 &&
              load.me.alg.test.threshold == 0.1 &&
              load.me.alg.flat.alpha == 0.5
        aictr = AdaptiveInputCompositeTrend()
        @test aictr.me.alg.window == 5
        tppt = TrendPromotePriceTracking()
        @test tppt.me.alg.test.window == 5 && tppt.me.alg.rising.alpha == 0.5
        @test MovingAverageReversion().eps == 10 &&
              MovingAverageReversion().me.alg.window == 5
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
