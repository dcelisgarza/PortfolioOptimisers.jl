#=
`budgeted_hindsight_path` (#1216): the best comparator sequence under a path-length budget,
the comparator of Zinkevich's (2003) Definition 7. At `L = 0` it is the static best constant
rebalanced portfolio, at a slack budget it is the per-period minimiser, and in between it
beats be-the-leader at be-the-leader's own path length, which is the gap the issue measured.
=#
@testset "Budgeted hindsight path" begin
    using PortfolioOptimisers, Test, LinearAlgebra, StableRNGs, Dates, Clarabel
    po = PortfolioOptimisers
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false))
    # The three-row example of the hindsight-split test, price relatives as returns.
    ts3 = Date(2024, 1, 1) .+ Day.(0:2)
    X3 = [1.2 0.98; 0.9 1.1; 1.3 0.7] .- 1
    rd3 = ReturnsResult(; nx = ["A", "B"], X = X3, ts = ts3)
    ew3 = predict(optimise(EqualWeighted(), rd3), rd3)
    path_weights(mp) = [p.res.w for p in mp.pred]

    @testset "The shape of the answer" begin
        bp = budgeted_hindsight_path(rd3, 1.0; slv = slv)
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
        one = budgeted_hindsight_path(rd1, 0.0; slv = slv)
        @test length(one.pred) == 1
        @test isapprox(one.pred[1].res.w, [1.0, 0.0]; atol = 1e-5)
        ew1 = predict(optimise(EqualWeighted(), rd1), rd1)
        @test isnan(log_wealth_regret(ew1, one).path_length)
    end

    @testset "The two ends of the budget" begin
        # `L = 0`: a constant path, which is the best constant rebalanced portfolio in
        # hindsight, all-in on the first asset with wealth 1.2 · 0.9 · 1.3.
        static = budgeted_hindsight_path(rd3, 0.0; slv = slv)
        u = path_weights(static)
        @test all(isapprox(ui, [1.0, 0.0]; atol = 1e-5) for ui in u)
        reg_s = log_wealth_regret(ew3, static)
        @test isapprox(reg_s.wealth_b, 1.2 * 0.9 * 1.3; atol = 1e-5)
        @test isapprox(reg_s.path_length, 0; atol = 1e-5)
        bcrp = predict(optimise(BestConstantRebalancedPortfolio(), rd3), rd3)
        @test isapprox(reg_s.wealth_b, log_wealth_regret(ew3, bcrp).wealth_b; atol = 1e-5)
        # A slack budget: the per-period minimiser, one-hot on each row's best asset, wealth
        # Π_t max_i x_t,i and path length 2√2 for two switches between corners.
        slack = budgeted_hindsight_path(rd3, 10.0; slv = slv)
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
        bud = budgeted_hindsight_path(rd3, reg_b.path_length; slv = slv)
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
        l1 = budgeted_hindsight_path(rd3, 4.0; p = 1, slv = slv)
        @test isapprox(log_wealth_regret(ew3, l1).wealth_b, 1.2 * 1.1 * 1.3; atol = 1e-5)
        l1_half = budgeted_hindsight_path(rd3, 2.0; p = 1, slv = slv)
        u = path_weights(l1_half)
        @test isapprox(sum(norm(u[t] - u[t - 1], 1) for t in 2:3), 2.0; atol = 1e-5)
        @test log_wealth_regret(ew3, l1_half).wealth_b < 1.2 * 1.1 * 1.3
        # The infinity norm charges a switch between corners one, and the power cone at
        # `p = 3` reads the same corners at `2^(1/3)` each; both reach the minimiser at a
        # slack budget.
        linf = budgeted_hindsight_path(rd3, 2.0; p = Inf, slv = slv)
        @test isapprox(log_wealth_regret(ew3, linf).wealth_b, 1.2 * 1.1 * 1.3; atol = 1e-5)
        l3 = budgeted_hindsight_path(rd3, 2 * 2^(1 / 3); p = 3, slv = slv)
        @test isapprox(log_wealth_regret(ew3, l3).wealth_b, 1.2 * 1.1 * 1.3; atol = 1e-4)
        # A binding budget under `p = 3` spends `L` in its own norm.
        l3_half = budgeted_hindsight_path(rd3, 1.0; p = 3, slv = slv)
        u = path_weights(l3_half)
        @test isapprox(sum(norm(u[t] - u[t - 1], 3) for t in 2:3), 1.0; atol = 1e-4)
    end

    @testset "The bounds" begin
        # A cap of 0.7 on every asset: no corner is reachable, so the slack-budget path is
        # `(0.7, 0.3)`, `(0.3, 0.7)`, `(0.7, 0.3)`.
        wb = WeightBounds(; lb = 0.0, ub = 0.7)
        capped = budgeted_hindsight_path(rd3, 10.0; wb = wb, slv = slv)
        u = path_weights(capped)
        @test isapprox(u[1], [0.7, 0.3]; atol = 1e-5)
        @test isapprox(u[2], [0.3, 0.7]; atol = 1e-5)
        @test isapprox(u[3], [0.7, 0.3]; atol = 1e-5)
        @test all(all(==(0.7), p.res.wb.ub) for p in capped.pred)
        # A constant capped path is the bounded best constant rebalanced portfolio.
        cap0 = budgeted_hindsight_path(rd3, 0.0; wb = wb, slv = slv)
        u = path_weights(cap0)
        @test all(isapprox(ui, [0.7, 0.3]; atol = 1e-5) for ui in u)
        # Named bounds through the sets.
        sets = UniverseSets(; dict = Dict("nx" => ["A", "B"]))
        wbe = WeightBoundsEstimator(; lb = Dict("B" => 0.2), ub = 1.0)
        named = budgeted_hindsight_path(rd3, 0.0; wb = wbe, sets = sets, slv = slv)
        u = path_weights(named)
        @test all(isapprox(ui, [0.8, 0.2]; atol = 1e-5) for ui in u)
        # An estimator with no sets to resolve on is refused by name.
        @test_throws po.IsNothingError budgeted_hindsight_path(rd3, 0.0; wb = wbe,
                                                               slv = slv)
    end

    @testset "The refusals and a failed solve" begin
        @test_throws DomainError budgeted_hindsight_path(rd3, -1.0; slv = slv)
        @test_throws DomainError budgeted_hindsight_path(rd3, 1.0; p = 0.5, slv = slv)
        @test_throws po.IsEmptyError budgeted_hindsight_path(rd3, 1.0; slv = Solver[])
        rdn = ReturnsResult(; nx = ["A", "B"], X = [0.1 NaN; 0.0 0.1], ts = ts3[1:2])
        @test_throws ArgumentError budgeted_hindsight_path(rdn, 1.0; slv = slv)
        @test_throws po.IsNothingError budgeted_hindsight_path(ReturnsResult(), 1.0;
                                                               slv = slv)
        # A bound no allocation meets makes the programme infeasible: every fold carries
        # `NaN` weights and the failure, and the regret verb reads a `NaN` series.
        bad = budgeted_hindsight_path(rd3, 1.0; wb = WeightBounds(; lb = 0.6, ub = 1.0),
                                      slv = slv)
        @test all(isa(p.res.retcode, OptimisationFailure) for p in bad.pred)
        @test all(all(isnan, p.res.w) for p in bad.pred)
        @test haskey(bad.pred[1].res.retcode.res, :clarabel)
        @test all(isnan, bad.mrd.X)
        @test isnan(log_wealth_regret(ew3, bad).regret)
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
        bud = budgeted_hindsight_path(rd_test, reg_b.path_length; slv = slv)
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
        slack = budgeted_hindsight_path(rd_test, reg_p.path_length; slv = slv)
        reg_slack = log_wealth_regret(ew, slack)
        @test isapprox(reg_slack.regret, reg_p.regret; atol = 1e-3)
        @test isapprox(reg_slack.path_length, reg_p.path_length; atol = 1e-3)
    end
end
