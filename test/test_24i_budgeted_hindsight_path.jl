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
