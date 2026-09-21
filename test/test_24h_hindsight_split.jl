#=
The evaluation surface's second half (ADR 0165, #1184): `HindsightSplit`, the dynamic-regret
fields of `LogWealthRegretResult`, and the benchmark and turnover fields of
`performance_summary`. The three-row example is the ADR's with its first row moved to
`(1.2, 0.98)`, because at `(1.2, 0.8)` the leader over rows 1–2 is all-in on the first asset
and be-the-leader collapses onto the static comparator.
=#
@testset "Hindsight split and the dynamic-regret surface" begin
    using PortfolioOptimisers, Test, LinearAlgebra, Statistics, Dates, Clarabel
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
        # played on row `t`. `u_1 = e_1`, `u_2` interior, `u_3 = e_1`.
        btl = cross_val_predict(BestConstantRebalancedPortfolio(), rd3, HindsightSplit())
        @test isa(btl, MultiPeriodPredictionResult)
        @test btl.mrd.ts == ts3
        u = [p.res.w for p in btl.pred]
        @test isapprox(u[1], [1.0, 0.0]; atol = 1e-8)
        @test isapprox(u[2], [0.522693, 0.477307]; atol = 1e-5)
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
        ps = performance_summary(ret; benchmark = bench)
        e = ret .- bench
        @test isapprox(ps.excess_ret, mean(e) * 252)
        @test isapprox(ps.excess_ret, 1.68; atol = 1e-6)
        @test isapprox(ps.tracking_error, std(e) * sqrt(252))
        @test isapprox(ps.tracking_error, 0.242; atol = 1e-3)
        @test isapprox(ps.information_ratio, ps.excess_ret / ps.tracking_error)
        @test isapprox(ps.information_ratio, 6.9; atol = 0.05)
        @test isnan(ps.turnover)
        # The seven released statistics do not move under a benchmark.
        ps0 = performance_summary(ret)
        for f in (:ann_return, :ann_volatility, :sharpe, :sharpe_stderr, :sortino, :calmar,
                  :max_drawdown, :cvar, :n_periods, :periods_per_year, :alpha, :compound)
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
