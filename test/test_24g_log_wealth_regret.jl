#=
`log_wealth_regret` (ADR 0161): the verb over two prediction results with the same rows,
its Result, the hand example, the self-regret, the refusals, and the two Hindsight
Comparator recipes — `BestConstantRebalancedPortfolio` fit on the rows it is scored on,
and the best stock as a top-1 `ScoreSelector` composed with `EqualWeighted`. The one-argument
form compares a prediction result against the benchmark of its own rows (#1664).
=#
@testset "Log-wealth regret" begin
    using PortfolioOptimisers, Test, LinearAlgebra, StableRNGs, Statistics, Dates,
          Distributions
    po = PortfolioOptimisers
    # A prediction result over a bare return series, on the timestamps handed in.
    function bare_prediction(r, ts)
        res = po.NaiveOptimisationResult(; pr = nothing, wb = nothing,
                                         retcode = OptimisationSuccess(), w = [1.0],
                                         fb = nothing)
        return PredictionResult(; res = res,
                                rd = po.PredictionReturnsResult(; nx = ["P"], X = r,
                                                                ts = ts))
    end
    function newey_west(d, lags)
        M = length(d)
        e = d .- mean(d)
        g(k) = dot(e[(k + 1):M], e[1:(M - k)]) / M
        return g(0) + 2 * sum((1 - k / (lags + 1)) * g(k) for k in 1:lags; init = 0.0)
    end

    @testset "The hand example" begin
        a = [0.02, -0.01, 0.03]
        b = [0.025, 0.0, 0.03]
        ts = Date(2024, 1, 1) .+ Day.(0:2)
        reg = log_wealth_regret(bare_prediction(a, ts), bare_prediction(b, ts))
        @test isa(reg, LogWealthRegretResult)
        @test isapprox(reg.regret, 0.01494; atol = 1e-5)
        @test isapprox(reg.regret_per_period, 0.00498; atol = 1e-5)
        @test reg.difference ≈ log1p.(b) .- log1p.(a)
        @test reg.regret ≈ sum(reg.difference)
        @test reg.regret_per_period ≈ mean(reg.difference)
        # The running regret is summed once, so its last entry is the regret exactly.
        @test reg.cumulative[end] == reg.regret
        @test reg.regret_per_period == reg.regret / reg.n_periods
        @test reg.n_periods == 3
        @test reg.lags == 0
        @test reg.wealth_a ≈ prod(1 .+ a)
        @test reg.wealth_b ≈ prod(1 .+ b)
        @test reg.regret ≈ log(reg.wealth_b) - log(reg.wealth_a)
        # The test statistic against a hand Newey–West at zero lags, and at one lag.
        v0 = newey_west(reg.difference, 0)
        @test reg.variance ≈ v0
        @test reg.z ≈ sqrt(3) * mean(reg.difference) / sqrt(v0)
        @test reg.p ≈ 2 * ccdf(Normal(), abs(reg.z))
        reg1 = log_wealth_regret(bare_prediction(a, ts), bare_prediction(b, ts); lags = 1)
        @test reg1.variance ≈ newey_west(reg.difference, 1)
        @test reg1.lags == 1
        @test reg1.regret == reg.regret
        # The sign: swapping the two negates the regret.
        swap = log_wealth_regret(bare_prediction(b, ts), bare_prediction(a, ts))
        @test swap.regret ≈ -reg.regret
        @test swap.z ≈ -reg.z
        @test swap.p ≈ reg.p
    end

    @testset "A series against itself, and the refusals" begin
        a = [0.02, -0.01, 0.03]
        ts = Date(2024, 1, 1) .+ Day.(0:2)
        self = log_wealth_regret(bare_prediction(a, ts), bare_prediction(a, ts))
        @test self.regret == 0
        @test self.variance == 0
        @test isnan(self.z) && isnan(self.p)
        # Different timestamps compare nothing.
        @test_throws ArgumentError log_wealth_regret(bare_prediction(a, ts),
                                                     bare_prediction(a, ts .+ Day(1)))
        # Different lengths with no timestamps compare nothing either.
        @test_throws ArgumentError log_wealth_regret(bare_prediction(a, nothing),
                                                     bare_prediction(a[1:2], nothing))
        # The lag budget is bounded by the rows.
        @test_throws DomainError log_wealth_regret(bare_prediction(a, ts),
                                                   bare_prediction(a, ts); lags = 3)
        @test_throws DomainError log_wealth_regret(bare_prediction(a, ts),
                                                   bare_prediction(a, ts); lags = -1)
        # A return below -1 leaves a negative wealth, so it is refused with the return as
        # the value, in either series; a `NaN` return passes and reaches the regret.
        err = try
            log_wealth_regret(bare_prediction([0.01, -1.5, 0.0], ts),
                              bare_prediction(a, ts))
        catch e
            e
        end
        @test isa(err, DomainError) && err.val == -1.5
        @test_throws DomainError log_wealth_regret(bare_prediction(a, ts),
                                                   bare_prediction([NaN, -2.0, 0.0], ts))
        @test isnan(log_wealth_regret(bare_prediction([NaN, 0.0, 0.0], ts),
                                      bare_prediction(a, ts)).regret)
        # A return of -1 is a total loss: the regret is infinite and the test undefined.
        ruin = log_wealth_regret(bare_prediction([0.0, -1.0, 0.0], ts),
                                 bare_prediction(a, ts))
        @test ruin.regret == Inf && isnan(ruin.z) && isnan(ruin.p)
        # A difference that is a non-zero constant has no variance in exact arithmetic, so
        # the statistic is infinite; in floating point it is very large and `p` is zero.
        c = log_wealth_regret(bare_prediction(a, ts),
                              bare_prediction((1 .+ a) .* exp(0.001) .- 1, ts))
        @test isapprox(c.regret, 0.003; atol = 1e-12)
        @test c.variance < 1e-25 && abs(c.z) > 1e10 && c.p == 0
        # A population of paths reads its first path, as the summary does.
        b = [0.025, 0.0, 0.03]
        pop = bare_prediction([b, a], ts)
        @test log_wealth_regret(bare_prediction(a, ts), pop).regret ≈
              log_wealth_regret(bare_prediction(a, ts), bare_prediction(b, ts)).regret
    end

    # The interior volatility-pumping fixture of the head's own test file.
    function pumping_fixture(seed = 2150, T = 60)
        rng = StableRNG(seed)
        X = Matrix{Float64}(undef, T, 4)
        for t in 1:T
            up, dn = 0.4, -0.25
            X[t, 1] = (1 + (isodd(t) ? up : dn)) * exp(0.01 * randn(rng)) - 1
            X[t, 2] = (1 + (isodd(t) ? dn : up)) * exp(0.01 * randn(rng)) - 1
            X[t, 3] = 0.001 + 0.02 * randn(rng)
            X[t, 4] = -0.001 + 0.02 * randn(rng)
        end
        return X
    end
    X = pumping_fixture()
    T, N = size(X)
    nx = ["A", "B", "C", "D"]
    ts = Date(2024, 1, 1) .+ Day.(0:(T - 1))
    rd = ReturnsResult(; nx = nx, X = X, ts = ts)

    @testset "The best-stock recipe composes" begin
        pipe = Pipeline(;
                        steps = (ScoreSelector(; score = MeanReturn(; flag = true),
                                               rule = RankRule(; best = 1)),
                                 EqualWeighted()))
        res = fit(pipe, rd)
        best = argmax(vec(sum(log1p.(X); dims = 1)))
        # A pipeline's universe is fitted state (ADR 0028): the weights sit on the selected
        # assets, named by the pipeline's own returns, so the unit vector is on one asset.
        @test res.w == [1.0]
        @test res.ctx.returns.nx == [nx[best]]
        @test predict(res, rd).rd.X ≈ X[:, best]
        # The best `k` stocks equal-weighted is the same recipe at `best = k`.
        pipe2 = Pipeline(;
                         steps = (ScoreSelector(; score = MeanReturn(; flag = true),
                                                rule = RankRule(; best = 2)),
                                  EqualWeighted()))
        res2 = fit(pipe2, rd)
        top2 = sortperm(vec(sum(log1p.(X); dims = 1)); rev = true)[1:2]
        @test res2.w ≈ [0.5, 0.5]
        @test sort(res2.ctx.returns.nx) == sort(nx[top2])
        @test predict(res2, rd).rd.X ≈ vec(sum(X[:, top2]; dims = 2)) ./ 2
    end

    @testset "The Hindsight Comparator against the walk-forward" begin
        cv = IndexWalkForward(20, 1)
        ew = cross_val_predict(EqualWeighted(), rd, cv)
        @test isa(ew, MultiPeriodPredictionResult)
        rd_test = po.port_opt_view(rd, 21:T, :)
        @test rd_test.ts == ew.mrd.ts
        # Fit on the rows it is scored on, predicted in sample over the same rows.
        bcrp = predict(optimise(BestConstantRebalancedPortfolio(), rd_test), rd_test)
        reg = log_wealth_regret(ew, bcrp)
        # The hindsight best constant rebalanced portfolio beats every constant portfolio
        # on its own rows, the equal-weighted one included.
        @test reg.regret >= 0
        @test reg.n_periods == T - 20
        @test reg.regret ≈ sum(log1p, bcrp.rd.X) - sum(log1p, ew.mrd.X)
        @test reg.wealth_b >= reg.wealth_a
        # The mixed-type call reads a multi-period result and a single fold alike, and
        # the walk-forward's causal constant portfolio loses to the hindsight one too.
        causal = cross_val_predict(BestConstantRebalancedPortfolio(), rd, cv)
        @test log_wealth_regret(causal, bcrp).regret >= 0
        # The best stock in hindsight is a weaker comparator than the best constant
        # rebalanced portfolio on a pumping fixture, so the regret against it is lower.
        pipe = Pipeline(;
                        steps = (ScoreSelector(; score = MeanReturn(; flag = true),
                                               rule = RankRule(; best = 1)),
                                 EqualWeighted()))
        stock = predict(fit(pipe, rd_test), rd_test)
        @test log_wealth_regret(ew, stock).regret < reg.regret
        # A comparator over other rows is refused.
        other = predict(optimise(BestConstantRebalancedPortfolio(),
                                 po.port_opt_view(rd, 20:(T - 1), :)),
                        po.port_opt_view(rd, 20:(T - 1), :))
        @test_throws ArgumentError log_wealth_regret(ew, other)
    end

    @testset "The benchmark of the rows is the comparator (#1664)" begin
        # A bare prediction result that carries a benchmark series beside its returns.
        function bench_prediction(r, B, ts)
            res = po.NaiveOptimisationResult(; pr = nothing, wb = nothing,
                                             retcode = OptimisationSuccess(), w = [1.0],
                                             fb = nothing)
            return PredictionResult(; res = res,
                                    rd = po.PredictionReturnsResult(; nx = ["P"], X = r,
                                                                    B = B, ts = ts))
        end
        # Every field of the two results agrees, the `NaN` path length included.
        function same_regret(x, y)
            return all(f -> isequal(getproperty(x, f), getproperty(y, f)) ||
                            isapprox(getproperty(x, f), getproperty(y, f)),
                       fieldnames(LogWealthRegretResult))
        end
        # The comparator built by hand: a one-asset returns data whose one column is the
        # benchmark series, fit and predicted over the same rows.
        function hand_comparator(series, tss)
            r = ReturnsResult(; nx = ["M"], X = reshape(series, :, 1), ts = tss)
            return predict(optimise(EqualWeighted(), r), r)
        end
        a = [0.02, -0.01, 0.03]
        b = [0.025, 0.0, 0.03]
        tsh = Date(2024, 1, 1) .+ Day.(0:2)
        reg = log_wealth_regret(bench_prediction(a, b, tsh))
        @test same_regret(reg,
                          log_wealth_regret(bare_prediction(a, tsh),
                                            bare_prediction(b, tsh)))
        @test isnan(reg.path_length)
        @test same_regret(log_wealth_regret(bench_prediction(a, b, tsh); lags = 1),
                          log_wealth_regret(bare_prediction(a, tsh),
                                            bare_prediction(b, tsh); lags = 1))
        # A population reads its first path and its first benchmark series.
        @test same_regret(log_wealth_regret(bench_prediction([a, b], [b, a], tsh)), reg)
        # A multi-period result stacks the benchmark series of its folds.
        mp = MultiPeriodPredictionResult(;
                                         pred = [bench_prediction(a[1:2], b[1:2], tsh[1:2]),
                                                 bench_prediction(a[3:3], b[3:3], tsh[3:3])])
        @test same_regret(log_wealth_regret(mp), reg)
        # No benchmark, no comparator; the checks on the lags and the returns still hold.
        @test_throws ArgumentError log_wealth_regret(bare_prediction(a, tsh))
        @test_throws DomainError log_wealth_regret(bench_prediction(a, b, tsh); lags = 3)
        @test_throws DomainError log_wealth_regret(bench_prediction(a, [0.0, -1.5, 0.0],
                                                                    tsh))

        # A walk-forward keeps the benchmark of its test rows: a vector `B` as it is, a
        # matrix `B` collapsed by each fold's weights, in batch and online alike.
        Bv = 0.002 .+ 0.01 .* randn(StableRNG(1664), T)
        Bm = 0.01 .* randn(StableRNG(1665), T, N)
        rdv = ReturnsResult(; nx = nx, X = X, ts = ts, nb = ["M"], B = Bv)
        rdm = ReturnsResult(; nx = nx, X = X, ts = ts, nb = nx, B = Bm)
        tst = ts[21:T]
        cv = IndexWalkForward(20, 1)
        ewv = cross_val_predict(EqualWeighted(), rdv, cv)
        @test same_regret(log_wealth_regret(ewv),
                          log_wealth_regret(ewv, hand_comparator(Bv[21:T], tst)))
        ewm = cross_val_predict(EqualWeighted(), rdm, cv)
        @test same_regret(log_wealth_regret(ewm),
                          log_wealth_regret(ewm,
                                            hand_comparator(Bm[21:T, :] * fill(1 / N, N),
                                                            tst)))
        ocv = OnlineIndexWalkForward(20, 1)
        ops = po.OnlinePortfolioSelection(; alg = ExponentiatedGradient())
        ov = cross_val_predict(ops, rdv, ocv)
        @test same_regret(log_wealth_regret(ov),
                          log_wealth_regret(ov, hand_comparator(Bv[21:T], tst)))
        om = cross_val_predict(ops, rdm, ocv)
        # The head moves its weights at every fold, so the series is collapsed fold by fold.
        bom = [dot(Bm[20 + i, :], p.res.w) for (i, p) in enumerate(om.pred)]
        @test same_regret(log_wealth_regret(om),
                          log_wealth_regret(om, hand_comparator(bom, tst)))
    end
end

#=
The evaluation surface's second half (ADR 0165, #1184): `HindsightSplit`, the dynamic-regret
fields of `LogWealthRegretResult`, and the benchmark and turnover fields of
`performance_summary`. The three-row example is the ADR's with its first row moved to
`(1.2, 0.98)`, because at `(1.2, 0.8)` the leader over rows 1–2 is all-in on the first asset
and be-the-leader collapses onto the static comparator.
=#
@testset "Hindsight split and the dynamic-regret surface" begin
    using PortfolioOptimisers, Test, LinearAlgebra, Statistics, StatsBase, Dates, Clarabel
    po = PortfolioOptimisers
    # A prediction result over a bare return series, on the timestamps handed in, with the
    # weights and names a fold would carry.
    function bare_prediction(r, ts; w = [1.0], nx = ["P"], hw = nothing)
        res = po.NaiveOptimisationResult(; pr = nothing, wb = nothing,
                                         retcode = OptimisationSuccess(), w = w,
                                         fb = nothing)
        return PredictionResult(; res = res,
                                rd = po.PredictionReturnsResult(; nx = nx, X = r, ts = ts),
                                hw = hw)
    end
    ts3 = Date(2024, 1, 1) .+ Day.(0:2)
    # Price relatives `x_t`, as returns.
    X3 = [1.2 0.98; 0.9 1.1; 1.3 0.7] .- 1
    rd3 = ReturnsResult(; nx = ["A", "B"], X = X3, ts = ts3)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false))
    bcrp_mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv, ret = LogarithmicReturn()),
                       obj = MaximumReturn())
    best_stock = Pipeline(;
                          steps = (ScoreSelector(; score = MeanReturn(; flag = true),
                                                 rule = RankRule(; best = 1)),
                                   EqualWeighted()))

    @testset "The splitter" begin
        hs = HindsightSplit()
        @test hs.prefix && hs.start == 1
        @test isa(hs, po.WalkForwardEstimator)
        @test !po.folds_are_stepped(hs)
        @test po.fold_evaluation(hs) ==
              (; wd = nothing, pws = nothing, fa = nothing, store_weight_path = false,
               strict = false)
        @test po.folds_are_time_ordered(hs)
        @test_throws DomainError HindsightSplit(; start = 0)
        # Prefix folds: train `1:t`, test `t`, one fold per row.
        sp = split(hs, rd3)
        @test isa(sp, po.WalkForwardResult)
        @test sp.train_idx == [1:1, 1:2, 1:3]
        @test sp.test_idx == [1:1, 2:2, 3:3]
        @test n_splits(hs, rd3) == 3
        # Row-alone folds, from a later start.
        hs2 = HindsightSplit(; prefix = false, start = 2)
        sp2 = split(hs2, rd3)
        @test sp2.train_idx == [2:2, 3:3]
        @test sp2.test_idx == [2:2, 3:3]
        @test n_splits(hs2, rd3) == 2
        # `start` past the data is the data's rule, checked at the split.
        @test_throws DomainError split(HindsightSplit(; start = 4), rd3)
        # The rows of a `HindsightSplit(; start = k + 1)` are those of an
        # `IndexWalkForward(k, 1)`, so a comparator through it shares the strategy's rows.
        @test split(HindsightSplit(; start = 3), rd3).test_idx ==
              split(IndexWalkForward(2, 1), rd3).test_idx
        # The evaluation switches travel.
        hs3 = HindsightSplit(; wd = SelfFinancingDrift(), pws = DriftedWeights(),
                             store_weight_path = true, strict = true)
        fe = po.fold_evaluation(hs3)
        @test isa(fe.wd, SelfFinancingDrift) && isa(fe.pws, DriftedWeights)
        @test fe.store_weight_path && fe.strict
    end

    @testset "The three comparators on the three-row example" begin
        # The static comparator: all-in on the first asset, wealth 1.2 · 0.9 · 1.3.
        static = predict(optimise(BestConstantRebalancedPortfolio(), rd3), rd3)
        ew = predict(optimise(EqualWeighted(), rd3), rd3)
        reg_s = log_wealth_regret(ew, static)
        @test isapprox(reg_s.wealth_b, 1.404; atol = 1e-6)
        @test isnan(reg_s.path_length)
        @test reg_s.cumulative ≈ cumsum(reg_s.difference)
        @test reg_s.cumulative[end] ≈ reg_s.regret
        @test length(reg_s.cumulative) == 3
        # Be-the-leader: the best constant rebalanced portfolio over the rows through `t`,
        # played on row `t`. `u_1 = e_1`, `u_2` interior, `u_3 = e_1`. On two assets the
        # leader over rows 1–2 is the root of the first-order condition in `a = u_2[1]`,
        # `0.22 / (0.98 + 0.22 a) = 0.2 / (1.1 - 0.2 a)`, which is `a = 23 / 44`; the fixed
        # point at its default cap stops `3e-5` short of it.
        btl = cross_val_predict(BestConstantRebalancedPortfolio(), rd3, HindsightSplit())
        @test isa(btl, MultiPeriodPredictionResult)
        @test btl.mrd.ts == ts3
        u = [p.res.w for p in btl.pred]
        @test isapprox(u[1], [1.0, 0.0]; atol = 1e-8)
        @test isapprox(u[2], [23 / 44, 21 / 44]; atol = 1e-4)
        @test isapprox(u[3], [1.0, 0.0]; atol = 1e-8)
        reg_b = log_wealth_regret(ew, btl)
        @test isapprox(reg_b.wealth_b, 1.2 * dot(u[2], [0.9, 1.1]) * 1.3)
        @test isapprox(reg_b.wealth_b, 1.553; atol = 1e-3)
        @test isapprox(reg_b.path_length, norm(u[2] - u[1]) + norm(u[3] - u[2]))
        @test isapprox(reg_b.path_length, 1.350; atol = 1e-3)
        @test reg_b.wealth_b > reg_s.wealth_b
        # `MeanRisk` under a log return is the same leader from the second row, where its
        # covariance exists, to the solver's default tolerance.
        btl_mr = cross_val_predict(bcrp_mr, rd3, HindsightSplit(; start = 2))
        @test btl_mr.mrd.ts == ts3[2:3]
        @test isapprox(btl_mr.pred[1].res.w, u[2]; atol = 1e-4)
        @test isapprox(btl_mr.pred[2].res.w, u[3]; atol = 1e-4)
        # From the first row, the default prior's corrected covariance of one row is `NaN`
        # and its `Posdef` step throws. The prior the docstring of `HindsightSplit` names
        # fits one row, and the leader is then the same from row 1, and row by row. The
        # interior leader of row 2 lands `1.1e-4` in norm from `23 / 44` at Clarabel's
        # default tolerance.
        @test_throws ArgumentError cross_val_predict(bcrp_mr, rd3, HindsightSplit())
        pe1 = EmpiricalPrior(;
                             ce = PortfolioOptimisersCovariance(;
                                                                ce = Covariance(;
                                                                                ce = GeneralCovariance(;
                                                                                                       ce = StatsBase.SimpleCovariance())),
                                                                mp = MatrixProcessing(;
                                                                                      pdm = nothing)))
        bcrp_mr1 = MeanRisk(;
                            opt = JuMPOptimiser(; pe = pe1, slv = slv,
                                                ret = LogarithmicReturn()),
                            obj = MaximumReturn())
        btl_mr1 = cross_val_predict(bcrp_mr1, rd3, HindsightSplit())
        @test btl_mr1.mrd.ts == ts3
        @test isapprox(btl_mr1.pred[1].res.w, [1.0, 0.0]; atol = 1e-4)
        @test isapprox(btl_mr1.pred[2].res.w, [23 / 44, 21 / 44]; atol = 2e-4)
        @test isapprox(btl_mr1.pred[3].res.w, [1.0, 0.0]; atol = 1e-4)
        pp_mr1 = cross_val_predict(bcrp_mr1, rd3, HindsightSplit(; prefix = false))
        @test isapprox(pp_mr1.pred[1].res.w, [1.0, 0.0]; atol = 1e-4)
        @test isapprox(pp_mr1.pred[2].res.w, [0.0, 1.0]; atol = 1e-4)
        @test isapprox(pp_mr1.pred[3].res.w, [1.0, 0.0]; atol = 1e-4)
        # The per-period minimiser: one-hot on each row's best asset, wealth Π_t max_i x_t,i,
        # path length 2√2 for two switches between corners.
        pp = cross_val_predict(best_stock, rd3, HindsightSplit(; prefix = false))
        @test [p.rd.nx for p in pp.pred] == [["A"], ["B"], ["A"]]
        @test all(p.res.w == [1.0] for p in pp.pred)
        reg_p = log_wealth_regret(ew, pp)
        @test isapprox(reg_p.wealth_b, 1.2 * 1.1 * 1.3)
        @test isapprox(reg_p.wealth_b, 1.716)
        @test isapprox(reg_p.path_length, 2 * sqrt(2))
        @test reg_p.wealth_b > reg_b.wealth_b
        # The three regrets against one strategy order as the three wealths do.
        @test reg_s.regret < reg_b.regret < reg_p.regret
        # A one-fold multi-period result has one target and no path.
        one = cross_val_predict(BestConstantRebalancedPortfolio(), rd3,
                                HindsightSplit(; start = 3))
        ew3 = cross_val_predict(EqualWeighted(), rd3, IndexWalkForward(2, 1))
        @test ew3.mrd.ts == one.mrd.ts
        @test isnan(log_wealth_regret(ew3, one).path_length)
    end

    @testset "The online form is the prefix split fold for fold (ADR 0167)" begin
        # `OnlineHindsightSplit` is the prefix split wrapped in `Online`: the training
        # windows are nested prefixes, so the loop warms up on `1:start`, folds row `t`,
        # reads out and scores row `t`. The read-out after rows `1:t` equals the batch fit
        # over `1:t`, so the be-the-leader comparator is the batch one fold for fold, and
        # the Result carries the threaded estimator where the batch one carries none.
        for start in (1, 2)
            oh = OnlineHindsightSplit(; start = start)
            @test oh.est.prefix && oh.est.start == start
            @test split(oh, rd3).train_idx ==
                  split(HindsightSplit(; start = start), rd3).train_idx
            batch = cross_val_predict(BestConstantRebalancedPortfolio(), rd3,
                                      HindsightSplit(; start = start))
            online = cross_val_predict(BestConstantRebalancedPortfolio(), rd3, oh)
            @test online.mrd.ts == batch.mrd.ts
            @test length(online.pred) == length(batch.pred)
            for (po_, pb) in zip(online.pred, batch.pred)
                @test isapprox(po_.res.w, pb.res.w; atol = 1e-8)
                @test po_.rd.X == pb.rd.X
            end
            @test isnothing(batch.opt)
            @test !isnothing(online.opt)
        end
        # A JuMP leader steps too, from the row where its covariance exists.
        batch_mr = cross_val_predict(bcrp_mr, rd3, HindsightSplit(; start = 2))
        online_mr = cross_val_predict(bcrp_mr, rd3, OnlineHindsightSplit(; start = 2))
        for (po_, pb) in zip(online_mr.pred, batch_mr.pred)
            @test isapprox(po_.res.w, pb.res.w; atol = 1e-6)
        end
        # The row-alone split has no online form: `prefix` is not a keyword.
        @test_throws MethodError OnlineHindsightSplit(; prefix = false)
        @test_throws ArgumentError Online(HindsightSplit(; prefix = false))
    end

    @testset "The stacked fold weights embed by name" begin
        a = bare_prediction([0.01], ts3[1:1]; w = [1.0], nx = ["A"])
        b = bare_prediction([0.02], ts3[2:2]; w = [1.0], nx = ["B"])
        c = bare_prediction([0.03], ts3[3:3]; w = [0.25, 0.75], nx = ["B", "C"])
        mp = MultiPeriodPredictionResult(; pred = [a, b, c])
        W = po.stacked_fold_weights(mp, po.fold_target)
        @test W == [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.25 0.75]
        @test po.comparator_path_length(mp, Float64) ≈ sqrt(2) + norm([0.0, -0.75, 0.75])
        # A fold whose weights and names disagree is refused.
        bad = bare_prediction([0.01], ts3[1:1]; w = [0.5, 0.5], nx = ["A"])
        @test_throws DimensionMismatch po.stacked_fold_weights(MultiPeriodPredictionResult(;
                                                                                           pred = [bad,
                                                                                                   b]),
                                                               po.fold_target)
        # A population reads its first member.
        @test po.first_member([[0.2, 0.8], [0.5, 0.5]]) == [0.2, 0.8]
        @test po.first_member([0.2, 0.8]) == [0.2, 0.8]
    end

    @testset "The benchmark fields of the performance summary" begin
        ret = [0.02, -0.01, 0.03]
        bench = [0.01, 0.0, 0.01]
        ps = performance_summary(ret; ppy = 252, benchmark = bench)
        e = ret .- bench
        @test isapprox(ps.excess_ret, mean(e) * 252)
        @test isapprox(ps.excess_ret, 1.68; atol = 1e-6)
        @test isapprox(ps.tracking_error, std(e) * sqrt(252))
        @test isapprox(ps.tracking_error, 0.242; atol = 1e-3)
        @test isapprox(ps.information_ratio, ps.excess_ret / ps.tracking_error)
        @test isapprox(ps.information_ratio, 6.9; atol = 0.05)
        @test isnan(ps.turnover)
        # The seven released statistics do not move under a benchmark.
        ps0 = performance_summary(ret; ppy = 252)
        for f in (:ann_return, :ann_volatility, :sharpe, :sharpe_stderr, :sortino, :calmar,
                  :max_drawdown, :cvar, :n_periods, :ppy, :alpha, :compound)
            @test isequal(getproperty(ps, f), getproperty(ps0, f))
        end
        @test isnan(ps0.excess_ret) &&
              isnan(ps0.tracking_error) &&
              isnan(ps0.information_ratio) &&
              isnan(ps0.turnover)
        # A zero tracking error gives a `NaN` ratio, and a benchmark of the wrong length is
        # refused.
        pse = performance_summary(ret; benchmark = ret)
        @test pse.excess_ret == 0
        @test pse.tracking_error == 0 && isnan(pse.information_ratio)
        @test_throws DimensionMismatch performance_summary(ret; benchmark = bench[1:2])
        # Every route takes the keyword.
        X = [0.01 0.02; -0.02 0.01; 0.03 0.0]
        w = [0.5, 0.5]
        rd = ReturnsResult(; nx = ["A", "B"], X = X, ts = ts3)
        pw = performance_summary(w, X; benchmark = bench)
        @test isapprox(pw.excess_ret,
                       performance_summary(X * w; benchmark = bench).excess_ret)
        @test isapprox(performance_summary(w, rd; benchmark = bench).excess_ret,
                       pw.excess_ret)
        res = optimise(EqualWeighted(), rd)
        @test isapprox(performance_summary(res, rd; benchmark = bench).excess_ret,
                       pw.excess_ret)
        @test isapprox(performance_summary(predict(res, rd); benchmark = bench).excess_ret,
                       pw.excess_ret)
        @test_throws DimensionMismatch performance_summary(w, X; benchmark = bench[1:2])
    end

    @testset "The turnover of the held path" begin
        # The ADR's held path `[0.5, 0.5] → [0.6, 0.4] → [0.6, 0.4]`: the library's turnover
        # is the whole `sum(abs, Δw)`, `0.2` then `0`, so the mean is `0.1`.
        ws = [[0.5, 0.5], [0.6, 0.4], [0.6, 0.4]]
        folds = [bare_prediction([0.01], ts3[i:i]; w = ws[i], nx = ["A", "B"]) for i in 1:3]
        mp = MultiPeriodPredictionResult(; pred = folds)
        ps = performance_summary(mp)
        @test isapprox(ps.turnover, 0.1)
        @test isapprox(ps.turnover, mean(po.calc_turnover(ws)[2:end]))
        # A single fold and a one-fold result record no rebalance.
        @test isnan(performance_summary(folds[1]).turnover)
        @test isnan(performance_summary(MultiPeriodPredictionResult(; pred = folds[1:1])).turnover)
        @test isnothing(po.held_path_turnover(folds[1]))
        # The per-period minimiser switches corners on a one-name universe per fold: the
        # embedding by name makes every switch a full turnover of `2`.
        pp = cross_val_predict(best_stock, rd3, HindsightSplit(; prefix = false))
        @test isapprox(performance_summary(pp).turnover, 2.0)
        # With a drift the trade starts from the holding after the fold's last observation,
        # not from the target: fold 2 of be-the-leader drifts `u_2` on `x_2` before the
        # rebalance into `u_3`.
        btl = cross_val_predict(BestConstantRebalancedPortfolio(), rd3, HindsightSplit())
        btl_d = cross_val_predict(BestConstantRebalancedPortfolio(), rd3,
                                  HindsightSplit(; wd = SelfFinancingDrift(),
                                                 pws = DriftedWeights()))
        u = [p.res.w for p in btl.pred]
        held = [p.hw.w for p in btl_d.pred]
        @test all(!isnothing(p.hw) for p in btl_d.pred)
        @test isapprox(held[2], u[2] .* [0.9, 1.1] ./ dot(u[2], [0.9, 1.1]))
        tn_t = (sum(abs, u[2] - u[1]) + sum(abs, u[3] - u[2])) / 2
        tn_d = (sum(abs, u[2] - held[1]) + sum(abs, u[3] - held[2])) / 2
        @test isapprox(performance_summary(btl).turnover, tn_t)
        @test isapprox(performance_summary(btl_d).turnover, tn_d)
        @test tn_d > tn_t
        # The path length reads targets and ignores the drift.
        ew = predict(optimise(EqualWeighted(), rd3), rd3)
        @test isapprox(log_wealth_regret(ew, btl_d).path_length,
                       log_wealth_regret(ew, btl).path_length)
    end
end

#=
`BudgetedHindsightPath` (#1216): the best comparator sequence under a path-length budget,
the comparator of Zinkevich's (2003) Definition 7, as an estimator under the Hindsight
Comparator rule. At `L = 0` it is the static best constant rebalanced portfolio, at a slack
budget it is the per-period minimiser, and in between it beats be-the-leader at
be-the-leader's own path length, which is the gap the issue measured.
=#
@testset "Budgeted hindsight path" begin
    using PortfolioOptimisers, Test, LinearAlgebra, StableRNGs, Dates, Clarabel, Logging
    po = PortfolioOptimisers
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false))
    # The three-row example of the hindsight-split test, price relatives as returns.
    ts3 = Date(2024, 1, 1) .+ Day.(0:2)
    X3 = [1.2 0.98; 0.9 1.1; 1.3 0.7] .- 1
    rd3 = ReturnsResult(; nx = ["A", "B"], X = X3, ts = ts3)
    ew3 = predict(optimise(EqualWeighted(), rd3), rd3)
    path_weights(mp) = [p.res.w for p in mp.pred]
    # The Hindsight Comparator rule: fit on the rows, predict over them.
    function path_of(rd, L; kwargs...)
        return predict(optimise(BudgetedHindsightPath(; L = L, kwargs...), rd), rd)
    end

    @testset "The shape of the answer" begin
        bp = path_of(rd3, 1.0; slv = slv)
        @test isa(bp, MultiPeriodPredictionResult)
        @test length(bp.pred) == 3
        @test bp.mrd.ts == ts3
        @test all(p.rd.nx == ["A", "B"] for p in bp.pred)
        @test all(isa(p.res, po.NaiveOptimisationResult) for p in bp.pred)
        @test all(isa(p.res.retcode, OptimisationSuccess) for p in bp.pred)
        @test all(isnothing(p.hw) for p in bp.pred)
        # Each fold records its row, as a fold of a cross-validation records its test rows,
        # so a realised attribution of the path matches it to the block by position.
        @test [p.idx for p in bp.pred] == [1:1, 2:2, 3:3]
        u = path_weights(bp)
        @test all(isapprox(sum(ui), 1; atol = 1e-6) for ui in u)
        @test all(all(>=(-1e-6), ui) for ui in u)
        # Each fold's series is the row's return under that fold's allocation.
        @test all(isapprox(bp.pred[t].rd.X, [dot(u[t], X3[t, :])]; atol = 1e-8)
                  for t in 1:3)
        # The budget binds: the Euclidean path length the regret verb reads is `L`.
        reg = log_wealth_regret(ew3, bp)
        @test isapprox(reg.path_length, 1.0; atol = 1e-5)
        @test reg.n_periods == 3
        # A panel of one row has no step, so no budget is written: the path is the row's
        # best asset at any `L`, one fold, and no path length.
        rd1 = po.port_opt_view(rd3, 1:1, :)
        one = path_of(rd1, 0.0; slv = slv)
        @test length(one.pred) == 1
        @test isapprox(one.pred[1].res.w, [1.0, 0.0]; atol = 1e-5)
        ew1 = predict(optimise(EqualWeighted(), rd1), rd1)
        @test isnan(log_wealth_regret(ew1, one).path_length)
    end

    @testset "The two ends of the budget" begin
        # `L = 0`: a constant path, which is the best constant rebalanced portfolio in
        # hindsight, all-in on the first asset with wealth 1.2 · 0.9 · 1.3.
        static = path_of(rd3, 0.0; slv = slv)
        u = path_weights(static)
        @test all(isapprox(ui, [1.0, 0.0]; atol = 1e-5) for ui in u)
        reg_s = log_wealth_regret(ew3, static)
        @test isapprox(reg_s.wealth_b, 1.2 * 0.9 * 1.3; atol = 1e-5)
        @test isapprox(reg_s.path_length, 0; atol = 1e-5)
        bcrp = predict(optimise(BestConstantRebalancedPortfolio(), rd3), rd3)
        @test isapprox(reg_s.wealth_b, log_wealth_regret(ew3, bcrp).wealth_b; atol = 1e-5)
        # A slack budget: the per-period minimiser, one-hot on each row's best asset, wealth
        # Π_t max_i x_t,i and path length 2√2 for two switches between corners.
        slack = path_of(rd3, 10.0; slv = slv)
        u = path_weights(slack)
        @test isapprox(u[1], [1.0, 0.0]; atol = 1e-5)
        @test isapprox(u[2], [0.0, 1.0]; atol = 1e-5)
        @test isapprox(u[3], [1.0, 0.0]; atol = 1e-5)
        reg_p = log_wealth_regret(ew3, slack)
        @test isapprox(reg_p.wealth_b, 1.2 * 1.1 * 1.3; atol = 1e-5)
        @test isapprox(reg_p.path_length, 2 * sqrt(2); atol = 1e-5)
        # The budgeted optimum at be-the-leader's own path length, `1.35`, matches
        # be-the-leader's wealth `1.553` to solver tolerance on this example, as the issue
        # measured, and the budget binds.
        btl = cross_val_predict(BestConstantRebalancedPortfolio(), rd3, HindsightSplit())
        reg_b = log_wealth_regret(ew3, btl)
        bud = path_of(rd3, reg_b.path_length; slv = slv)
        reg_bud = log_wealth_regret(ew3, bud)
        @test reg_bud.wealth_b >= reg_b.wealth_b - 1e-6
        @test isapprox(reg_bud.wealth_b, reg_b.wealth_b; atol = 1e-4)
        @test isapprox(reg_bud.path_length, reg_b.path_length; atol = 1e-5)
        # The three regrets order as the budgets do.
        @test reg_s.regret <= reg_bud.regret + 1e-6 <= reg_p.regret + 2e-6
    end

    @testset "The norm of the budget" begin
        # Under the L1 norm the same two switches between corners cost 2 each, so the
        # budget the per-period minimiser needs is 4, and at 2 one switch is affordable.
        l1 = path_of(rd3, 4.0; p = 1, slv = slv)
        @test isapprox(log_wealth_regret(ew3, l1).wealth_b, 1.2 * 1.1 * 1.3; atol = 1e-5)
        l1_half = path_of(rd3, 2.0; p = 1, slv = slv)
        u = path_weights(l1_half)
        @test isapprox(sum(norm(u[t] - u[t - 1], 1) for t in 2:3), 2.0; atol = 1e-5)
        @test log_wealth_regret(ew3, l1_half).wealth_b < 1.2 * 1.1 * 1.3
        # The infinity norm charges a switch between corners one, and the power cone at
        # `p = 3` reads the same corners at `2^(1/3)` each; both reach the minimiser at a
        # slack budget.
        linf = path_of(rd3, 2.0; p = Inf, slv = slv)
        @test isapprox(log_wealth_regret(ew3, linf).wealth_b, 1.2 * 1.1 * 1.3; atol = 1e-5)
        l3 = path_of(rd3, 2 * 2^(1 / 3); p = 3, slv = slv)
        @test isapprox(log_wealth_regret(ew3, l3).wealth_b, 1.2 * 1.1 * 1.3; atol = 1e-4)
        # A binding budget under `p = 3` spends `L` in its own norm.
        l3_half = path_of(rd3, 1.0; p = 3, slv = slv)
        u = path_weights(l3_half)
        @test isapprox(sum(norm(u[t] - u[t - 1], 3) for t in 2:3), 1.0; atol = 1e-4)
    end

    @testset "The bounds" begin
        # A cap of 0.7 on every asset: no corner is reachable, so the slack-budget path is
        # `(0.7, 0.3)`, `(0.3, 0.7)`, `(0.7, 0.3)`.
        wb = WeightBounds(; lb = 0.0, ub = 0.7)
        capped = path_of(rd3, 10.0; wb = wb, slv = slv)
        u = path_weights(capped)
        @test isapprox(u[1], [0.7, 0.3]; atol = 1e-5)
        @test isapprox(u[2], [0.3, 0.7]; atol = 1e-5)
        @test isapprox(u[3], [0.7, 0.3]; atol = 1e-5)
        @test all(all(==(0.7), p.res.wb.ub) for p in capped.pred)
        # A constant capped path is the bounded best constant rebalanced portfolio.
        cap0 = path_of(rd3, 0.0; wb = wb, slv = slv)
        u = path_weights(cap0)
        @test all(isapprox(ui, [0.7, 0.3]; atol = 1e-5) for ui in u)
        # Named bounds through the sets.
        sets = UniverseSets(; dict = Dict("nx" => ["A", "B"]))
        wbe = WeightBoundsEstimator(; lb = Dict("B" => 0.2), ub = 1.0)
        named = path_of(rd3, 0.0; wb = wbe, sets = sets, slv = slv)
        u = path_weights(named)
        @test all(isapprox(ui, [0.8, 0.2]; atol = 1e-5) for ui in u)
    end

    @testset "The estimator, the result, and the refusals" begin
        est = BudgetedHindsightPath(; L = 1.0, slv = slv)
        @test isa(est, po.OptimisationEstimator)
        @test !isa(est, po.NonFiniteAllocationOptimisationEstimator)
        @test est.L == 1.0 && est.p == 2 && est.wb == WeightBounds() && isnothing(est.sets)
        @test !est.strict && est.slv === slv && isnothing(est.fb)
        # The constructor refuses a negative budget, a norm order below one, an empty
        # solver vector, and a bounds estimator with no sets to resolve on.
        @test_throws DomainError BudgetedHindsightPath(; L = -1.0, slv = slv)
        @test_throws DomainError BudgetedHindsightPath(; L = 1.0, p = 0.5, slv = slv)
        @test_throws po.IsEmptyError BudgetedHindsightPath(; L = 1.0, slv = Solver[])
        wbe = WeightBoundsEstimator(; lb = Dict("B" => 0.2), ub = 1.0)
        @test_throws po.IsNothingError BudgetedHindsightPath(; L = 1.0, slv = slv, wb = wbe)
        # The fit is the path itself, bound to its rows.
        res = optimise(est, rd3)
        @test isa(res, BudgetedHindsightPathResult)
        @test isa(res, po.OptimisationResult)
        @test size(res.w) == (3, 2)
        @test all(isapprox(sum(res.w[t, :]), 1; atol = 1e-6) for t in 1:3)
        @test isnothing(res.imsk) && res.nx == ["A", "B"] && res.ts == ts3
        @test isa(res.retcode, OptimisationSuccess) && isnothing(res.fb)
        @test all(res.wb.lb .== 0) && all(res.wb.ub .== 1)
        # `predict` over the fit's rows is the one-fold-per-row form, and its weights are
        # the rows of the path.
        pred = predict(res, rd3)
        @test isa(pred, MultiPeriodPredictionResult)
        @test path_weights(pred) == [res.w[t, :] for t in 1:3]
        # Other rows are another fit.
        @test_throws ArgumentError predict(res, po.port_opt_view(rd3, 1:2, :))
        @test_throws ArgumentError predict(res,
                                           ReturnsResult(; nx = ["A", "C"], X = X3,
                                                         ts = ts3))
        @test_throws ArgumentError predict(res,
                                           ReturnsResult(; nx = ["A", "B"], X = X3,
                                                         ts = ts3 .+ Day(1)))
        @test_throws po.IsNothingError optimise(est, ReturnsResult())
        @test optimise(est, rd3; dims = 2).w ≈ res.w
        # The estimator views its bounds and sets.
        sets = UniverseSets(; dict = Dict("nx" => ["A", "B"]))
        est_v = po.port_opt_view(BudgetedHindsightPath(; L = 1.0, slv = slv, sets = sets,
                                                       wb = WeightBounds(; lb = [0.0, 0.1],
                                                                         ub = 1.0)), 2:2)
        @test est_v.wb.lb == [0.1] && est_v.sets.dict["nx"] == ["B"]
        # A row with no asset in its Coverage Universe has no asset to hold, and the
        # refusal names the row.
        rdn = ReturnsResult(; nx = ["A", "B"], X = [0.1 0.2; NaN NaN; 0.0 0.1], ts = ts3)
        err = try
            path_of(rdn, 1.0; slv = slv)
        catch e
            e
        end
        @test isa(err, po.IsEmptyError)
        @test occursin("row 2", err.msg)
        # The restatement names the row for the empty universe alone; any other error of
        # the coverage rule passes through as it is.
        other = ArgumentError("x")
        @test po.row_universe_error(other, 2) === other
        @test occursin("row 7", po.row_universe_error(po.IsEmptyError("e"), 7).msg)
        # A bound no allocation meets makes the programme infeasible: the fit is `NaN`
        # with the failure, every fold carries both, and the regret verb reads a `NaN`
        # series.
        bad_est = BudgetedHindsightPath(; L = 1.0, wb = WeightBounds(; lb = 0.6, ub = 1.0),
                                        slv = slv)
        bad_res = optimise(bad_est, rd3)
        @test isa(bad_res.retcode, OptimisationFailure)
        @test all(isnan, bad_res.w)
        @test haskey(bad_res.retcode.res, :clarabel)
        bad = predict(bad_res, rd3)
        @test all(isa(p.res.retcode, OptimisationFailure) for p in bad.pred)
        @test all(all(isnan, p.res.w) for p in bad.pred)
        @test all(isnan, bad.mrd.X)
        @test isnan(log_wealth_regret(ew3, bad).regret)
        # The door walks the fallback chain: the infeasible fit falls back to the simplex
        # fit, and the result records the failed attempt.
        chained = optimise(BudgetedHindsightPath(; L = 1.0,
                                                 wb = WeightBounds(; lb = 0.6, ub = 1.0),
                                                 slv = slv, fb = est), rd3)
        @test isa(chained.retcode, OptimisationSuccess)
        @test chained.w ≈ res.w
        @test length(chained.fb) == 1
        @test isa(chained.fb[1][1], BudgetedHindsightPath) && chained.fb[1][1].wb.lb == 0.6
        @test isa(chained.fb[1][2].retcode, OptimisationFailure)
    end

    @testset "A point-in-time panel" begin
        # Asset B has no return on row 1 and asset A none on row 3: each is not investable
        # on that row, so the path is forced onto the other asset there, and the fold
        # carries the row's mask so `predict` never reads the gap.
        Xn = [1.2 NaN; 0.9 1.1; NaN 0.7] .- 1
        rdn = ReturnsResult(; nx = ["A", "B"], X = Xn, ts = ts3)
        pit_res = optimise(BudgetedHindsightPath(; L = 10.0, slv = slv), rdn)
        @test pit_res.imsk == BitMatrix([true false; true true; false true])
        @test pit_res.w[1, 2] == 0.0 && pit_res.w[3, 1] == 0.0
        pit = @test_logs min_level = Logging.Warn predict(pit_res, rdn)
        @test [p.rd.nx for p in pit.pred] == [["A"], ["A", "B"], ["B"]]
        @test pit.pred[1].res.imsk == BitVector([true, false])
        @test isnothing(pit.pred[2].res.imsk)
        @test pit.pred[3].res.imsk == BitVector([false, true])
        # The weights sit on the full universe with an exact zero at the gap.
        u = path_weights(pit)
        @test isapprox(u[1], [1.0, 0.0]; atol = 1e-6) && u[1][2] == 0.0
        @test isapprox(u[3], [0.0, 1.0]; atol = 1e-6) && u[3][1] == 0.0
        # A slack budget takes row 2's best asset, so the wealth is 1.2 · 1.1 · 0.7 and the
        # series is finite on every row.
        @test isapprox(u[2], [0.0, 1.0]; atol = 1e-5)
        @test all(isfinite, pit.mrd.X)
        # No asset is quoted throughout, so a covered strategy does not exist on this
        # panel; the regret is read against a bare cash series on the same timestamps.
        cash = PredictionResult(;
                                res = po.NaiveOptimisationResult(; pr = nothing,
                                                                 wb = nothing,
                                                                 retcode = OptimisationSuccess(),
                                                                 w = [1.0], fb = nothing),
                                rd = po.PredictionReturnsResult(; nx = ["P"], X = zeros(3),
                                                                ts = ts3))
        reg = log_wealth_regret(cash, pit)
        @test isapprox(reg.wealth_b, 1.2 * 1.1 * 0.7; atol = 1e-5)
        # The forced exit from A on row 3 is a step, charged to the path like any other:
        # `(1, 0) → (0, 1) → (0, 1)` embedded by name on the union of the folds' universes.
        @test isapprox(reg.path_length, sqrt(2); atol = 1e-5)
        # A zero budget cannot hold one allocation across the two masks, so the programme
        # is infeasible and every fold carries the failure.
        stuck_res = optimise(BudgetedHindsightPath(; L = 0.0, slv = slv), rdn)
        @test isa(stuck_res.retcode, OptimisationFailure)
        @test stuck_res.w[1, 2] == 0.0 && isnan(stuck_res.w[1, 1])
        stuck = predict(stuck_res, rdn)
        @test all(isa(p.res.retcode, OptimisationFailure) for p in stuck.pred)
        # The investable weights are `NaN`, and the gap keeps its exact zero.
        @test all(all(isnan, po.investable_weights_view(p.res.imsk, p.res.w))
                  for p in stuck.pred)
        @test stuck.pred[1].res.w[2] == 0.0
        @test all(isnan, stuck.mrd.X)
        # When the rows share assets, a zero budget holds the shared assets alone, and it is
        # the best constant rebalanced portfolio over them: B has no return on row 1, so both
        # fits drop it and go all-in on A.
        Xg = [1.2 NaN 1.01; 0.9 1.1 1.0; 1.3 0.7 1.02] .- 1
        rdg = ReturnsResult(; nx = ["A", "B", "C"], X = Xg, ts = ts3)
        zero_res = optimise(BudgetedHindsightPath(; L = 0.0, slv = slv), rdg)
        @test isa(zero_res.retcode, OptimisationSuccess)
        # B's gap on row 1 is an exact zero; the constant path holds B at the solver's zero.
        @test zero_res.w[1, 2] == 0.0
        @test all(isapprox.(zero_res.w[:, 2], 0; atol = 1e-6))
        bcrp_g = optimise(BestConstantRebalancedPortfolio(), rdg)
        @test all(isapprox(zero_res.w[t, :], bcrp_g.w; atol = 1e-5) for t in 1:3)
        # A vector lower bound is read on the investable assets alone, so a floor on the
        # missing asset does not make the row infeasible.
        floored = path_of(rdn, 10.0; wb = WeightBounds(; lb = [0.1, 0.1], ub = 1.0),
                          slv = slv)
        @test all(isa(p.res.retcode, OptimisationSuccess) for p in floored.pred)
        @test isapprox(path_weights(floored)[1], [1.0, 0.0]; atol = 1e-6)
        @test isapprox(path_weights(floored)[2], [0.1, 0.9]; atol = 1e-5)
        # The row's universe reads the Asset Panel's active mask too: asset B's return is
        # finite on row 2, but the panel says it is inactive there, so the row holds A
        # alone, as a `HindsightSplit(; prefix = false)` fold would.
        amsk = BitMatrix([true false; true false; false true])
        rdp = ReturnsResult(; nx = ["A", "B"], X = Xn, ts = ts3,
                            pnl = AssetPanel(; amsk = amsk, emsk = copy(amsk)))
        pnl_res = optimise(BudgetedHindsightPath(; L = 10.0, slv = slv), rdp)
        @test pnl_res.imsk == amsk
        @test isapprox(pnl_res.w[2, :], [1.0, 0.0]; atol = 1e-6) && pnl_res.w[2, 2] == 0.0
        @test [p.rd.nx for p in predict(pnl_res, rdp).pred] == [["A"], ["A"], ["B"]]
        # An inactive row over every asset is refused by name, finite returns or not.
        amsk0 = BitMatrix([true true; false false; true true])
        rdp0 = ReturnsResult(; nx = ["A", "B"], X = X3, ts = ts3,
                             pnl = AssetPanel(; amsk = amsk0, emsk = copy(amsk0)))
        @test_throws po.IsEmptyError optimise(BudgetedHindsightPath(; L = 1.0, slv = slv),
                                              rdp0)
    end

    # The interior volatility-pumping fixture of the regret test file, folds `21:60`, and
    # the gap the issue measured: the budgeted optimum at be-the-leader's own path length
    # beats be-the-leader, and at the per-period minimiser's it coincides with it.
    @testset "The gap on the pumping fixture" begin
        function pumping_fixture(seed = 2150, T = 60)
            rng = StableRNG(seed)
            X = Matrix{Float64}(undef, T, 4)
            for t in 1:T
                up, dn = 0.4, -0.25
                X[t, 1] = (1 + (isodd(t) ? up : dn)) * exp(0.01 * randn(rng)) - 1
                X[t, 2] = (1 + (isodd(t) ? dn : up)) * exp(0.01 * randn(rng)) - 1
                X[t, 3] = 0.001 + 0.02 * randn(rng)
                X[t, 4] = -0.001 + 0.02 * randn(rng)
            end
            return X
        end
        X = pumping_fixture()
        T = size(X, 1)
        rd = ReturnsResult(; nx = ["A", "B", "C", "D"], X = X,
                           ts = Date(2024, 1, 1) .+ Day.(0:(T - 1)))
        ew = cross_val_predict(EqualWeighted(), rd, IndexWalkForward(20, 1))
        rd_test = po.port_opt_view(rd, 21:T, :)
        # Be-the-leader over the test window's own prefixes, as the issue measured it.
        btl = cross_val_predict(BestConstantRebalancedPortfolio(), rd_test,
                                HindsightSplit())
        reg_b = log_wealth_regret(ew, btl)
        @test isapprox(reg_b.regret, 3.095; atol = 1e-2)
        @test isapprox(reg_b.path_length, 7.59; atol = 1e-2)
        bud = path_of(rd_test, reg_b.path_length; slv = slv)
        reg_bud = log_wealth_regret(ew, bud)
        @test isapprox(reg_bud.regret, 3.280; atol = 1e-2)
        @test reg_bud.regret > reg_b.regret + 0.1
        @test isapprox(reg_bud.path_length, reg_b.path_length; atol = 1e-4)
        best_stock = Pipeline(;
                              steps = (ScoreSelector(; score = MeanReturn(; flag = true),
                                                     rule = RankRule(; best = 1)),
                                       EqualWeighted()))
        pp = cross_val_predict(best_stock, rd_test, HindsightSplit(; prefix = false))
        reg_p = log_wealth_regret(ew, pp)
        @test isapprox(reg_p.regret, 12.014; atol = 1e-2)
        slack = path_of(rd_test, reg_p.path_length; slv = slv)
        reg_slack = log_wealth_regret(ew, slack)
        @test isapprox(reg_slack.regret, reg_p.regret; atol = 1e-3)
        @test isapprox(reg_slack.path_length, reg_p.path_length; atol = 1e-3)
    end
end

@testset "Base cross validation" begin
    using Test, PortfolioOptimisers, Clarabel, Dates, StableRNGs
    naive(w; retcode = OptimisationSuccess()) = NaiveOptimisationResult(; pr = nothing,
                                                                        wb = nothing,
                                                                        retcode = retcode,
                                                                        w = w, fb = nothing)
    fold(X) = PredictionResult(; res = naive([0.5, 0.5]),
                               rd = PredictionReturnsResult(; nx = ["p"], X = X))
    X2 = [[0.01, 0.02, 0.03], [0.0, 0.01, -0.01]]
    @testset "A split result splits to itself" begin
        rd = ReturnsResult(; nx = ["a", "b"], X = randn(StableRNG(1), 12, 2) / 100)
        res = split(KFold(; n = 3), rd)
        @test split(res, rd) === res
        @test n_splits(res) == n_splits(res, rd) == 3
    end
    @testset "PredictionReturnsResult checks a population's benchmark" begin
        @test PredictionReturnsResult(; X = X2, B = [[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]]).B ==
              [[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]]
        @test_throws DimensionMismatch PredictionReturnsResult(; X = X2,
                                                               B = [[0.1, 0.2],
                                                                    [0.0, 0.0, 0.0]])
        @test_throws ArgumentError PredictionReturnsResult(; X = X2, B = [0.1, 0.2, 0.3])
        @test_throws DimensionMismatch PredictionReturnsResult(; nf = ["f1", "f2"],
                                                               F = [1.0 2.0; 3.0 4.0],
                                                               B = [[0.1, 0.2, 0.3]],
                                                               ts = [Date(2020, 1, 1),
                                                                     Date(2020, 1, 2)])
        @test_throws DimensionMismatch PredictionReturnsResult(; X = X2,
                                                               B = [[0.1, 0.2], [0.0, 0.0]],
                                                               ts = [Date(2020, 1, i)
                                                                     for i in 1:3])
    end
    @testset "PredictionReturnsResult admits a population's iv without ivpa" begin
        iv = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
        prr = PredictionReturnsResult(; X = X2, iv = iv)
        @test prr.iv == iv
        @test isnothing(prr.ivpa)
        @test_throws DimensionMismatch PredictionReturnsResult(; X = X2, iv = iv,
                                                               ivpa = [1.0])
        @test_throws DomainError PredictionReturnsResult(; X = X2, iv = iv,
                                                         ivpa = [1.0, 0.0])
        @test_throws IsNothingError PredictionReturnsResult(; iv = [0.1, 0.2, 0.3])
        @test_throws IsNothingError PredictionReturnsResult(; iv = iv, ivpa = [1.0, 1.0])
    end
    @testset "A frontier predicts over implied volatilities without a premium" begin
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     check_sol = (; allow_local = true, allow_almost = true),
                     settings = "verbose" => false)
        rng = StableRNG(1)
        X = randn(rng, 60, 4) / 100 .+ 0.001
        iv = abs.(randn(rng, 60, 4)) / 10 .+ 0.1
        rd = ReturnsResult(; nx = ["a", "b", "c", "d"], X = X, iv = iv)
        mr = MeanRisk(;
                      opt = JuMPOptimiser(; slv = slv,
                                          ret = ArithmeticReturn(;
                                                                 settings = JuMPReturnsSettings(;
                                                                                                lb = Frontier(;
                                                                                                              N = 3)))))
        res = optimise(mr, rd)
        pred = predict(res, rd)
        @test isnothing(pred.rd.ivpa)
        @test length(pred.rd.iv) == 3
        for (ivi, wi) in zip(pred.rd.iv, res.w)
            @test isapprox(ivi, iv * (abs.(wi) / sum(abs, wi)))
        end
    end
    @testset "Population measures and the order of folds" begin
        mp = MultiPeriodPredictionResult(; pred = [fold([0.01, -0.02]), fold([0.03])])
        r = ConditionalValueatRisk()
        @test expected_risk(r, [mp, mp]) == fill(expected_risk(r, mp), 2)
        rd = ReturnsResult(; nx = ["a", "b"], X = randn(StableRNG(2), 12, 2) / 100)
        res = split(KFold(; n = 3), rd)
        preds = [fold([Float64(i)]) for i in 1:3]
        perm = sortperm(res.test_idx; by = first)
        @test PortfolioOptimisers.sort_predictions(res, preds) == preds[perm]
        shuffled = [[5, 6], [1, 2], [3, 4]]
        @test [p.rd.X[1] for p in PortfolioOptimisers.sort_predictions(shuffled, preds)] ==
              [2.0, 3.0, 1.0]
        @test [p.rd.X[1] for p in preds] == [1.0, 2.0, 3.0]
    end
    @testset "The fold-loop predicates" begin
        @test PortfolioOptimisers.folds_are_time_ordered(nothing) === true
        @test PortfolioOptimisers.folds_are_time_ordered(IndexWalkForward(10, 5)) === true
        @test PortfolioOptimisers.folds_are_time_ordered(KFold()) === false
        @test PortfolioOptimisers.folds_are_stepped(nothing) === false
        @test PortfolioOptimisers.folds_are_stepped(KFold()) === false
    end
    @testset "A population of single folds ranks to single folds" begin
        low = fold([0.01, -0.02, 0.03])
        high = fold([0.05, -0.20, 0.0])
        pop = PopulationPredictionResult(; pred = [high, low])
        @test pop.pred[1] === high
        r = ConditionalValueatRisk()
        sorted = sort_by_measure(pop, r)
        @test all(x -> isa(x, PredictionResult), sorted)
        @test sorted[1] === low
        @test sorted[2] === high
        @test PortfolioOptimisers.quantile_by_measure(pop, r, 1.0) === high
        @test PortfolioOptimisers.quantile_by_measure(pop, r, 0.0) === low
    end
    @testset "A failed member keeps its weights with no previous weights" begin
        s = OptimisationSuccess()
        f = OptimisationFailure(; res = nothing)
        W = [[0.5, 0.5], [NaN, NaN], [0.1, 0.9]]
        hsw = PortfolioOptimisers.held_start_weights
        @test isequal(hsw([s, f, s], W, nothing), W)
        @test hsw([s, f, s], W, [0.3, 0.7]) == [[0.5, 0.5], [0.3, 0.7], [0.1, 0.9]]
        @test hsw(f, W, [0.3, 0.7]) == fill([0.3, 0.7], 3)
        @test_throws DimensionMismatch hsw([s, f, s], W, [[1.0, 0.0]])
    end
end
