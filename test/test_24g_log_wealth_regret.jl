#=
`log_wealth_regret` (ADR 0161): the verb over two prediction results with the same rows,
its Result, the hand example, the self-regret, the refusals, and the two Hindsight
Comparator recipes — `BestConstantRebalancedPortfolio` fit on the rows it is scored on,
and the best stock as a top-1 `ScoreSelector` composed with `EqualWeighted`.
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
end
