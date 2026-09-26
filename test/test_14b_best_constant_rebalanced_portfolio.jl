#=
`BestConstantRebalancedPortfolio` (ADR 0161): Cover's fixed point as a naive head, parity
against `MeanRisk` under `LogarithmicReturn` on an interior fixture, the simplex-only
repair of a binding `wb`, and the online read-out identity of ADR 0137.

The fixture is a volatility-pumping pair: two assets whose price relatives alternate
`1.4, 0.75` in anti-phase, so each alone loses per two periods on average while the half
and half mix gains `(1.4 + 0.75) / 2 = 1.075` every period. A random fixture at `T = 60`
lands the best constant rebalanced portfolio on a corner, where the head and the programme
both pick the best single asset and the parity test says nothing about the fixed point.
=#
@testset "Best constant rebalanced portfolio" begin
    using PortfolioOptimisers, Test, LinearAlgebra, StableRNGs, Statistics, Dates, Clarabel
    po = PortfolioOptimisers
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
    log_wealth(w, X = X) = sum(log, (1 .+ X) * w)

    @testset "The fixed point on a hand example" begin
        # Two assets, anti-phase `1.4, 0.75`: the optimum is the half and half mix by
        # symmetry, and the kernel reaches it from the uniform start in one step.
        Xh = [1.4 0.75; 0.75 1.4; 1.4 0.75; 0.75 1.4]
        fp = po.cover_fixed_point(Xh, 100, 1e-12)
        @test fp.w ≈ [0.5, 0.5]
        @test fp.converged
        @test fp.log_wealth ≈ 4 * log(1.075)
        # On the symmetric example the uniform start is the fixed point already, so its
        # certificate is met before any step; on the fixture a budget of one takes one step
        # and stops short.
        fp1 = po.cover_fixed_point(Xh, 1, 1e-12)
        @test fp1.converged && fp1.iterations == 0
        @test fp1.gap <= 1e-12
        fpx = po.cover_fixed_point(1 .+ X, 1, 1e-12)
        @test !fpx.converged && fpx.iterations == 1
        # A budget of zero takes no step: the kernel certifies and returns the uniform start.
        fp0 = po.cover_fixed_point(1 .+ X, 0, 1e-12)
        @test !fp0.converged && fp0.iterations == 0
        @test fp0.w == fill(1 / N, N)
        @test fp0.log_wealth ≈ log_wealth(fill(1 / N, N))
        @test fp0.gap > 0
        # The certificate bounds the shortfall from the optimum from above, at every budget.
        fps = po.cover_fixed_point(1 .+ X, 100_000, 1e-14)
        for k in (1, 10, 100)
            fpk = po.cover_fixed_point(1 .+ X, k, 1e-12)
            @test fps.log_wealth - fpk.log_wealth <= fpk.gap
            @test fpk.gap > 0
        end
        # The type is derived from the data, never forced.
        fp32 = po.cover_fixed_point(Float32.(Xh), 100, 1.0f-6)
        @test eltype(fp32.w) === Float32
    end

    @testset "The head on the interior fixture" begin
        bcrp = BestConstantRebalancedPortfolio()
        res = optimise(bcrp, rd)
        @test isa(res, po.NaiveOptimisationResult)
        @test isa(res.retcode, OptimisationSuccess)
        @test res.retcode.res.converged
        @test 0 <= res.retcode.res.gap <= 1e-12 * max(1, abs(log_wealth(res.w)))
        @test sum(res.w) ≈ 1
        @test all(res.w .>= 0)
        # Interior: the pumping pair shares the book, the noise assets are dropped.
        @test all(res.w[1:2] .> 0.4)
        @test all(res.w[3:4] .< 1e-8)
        # The best constant rebalanced portfolio beats every single asset on its own rows.
        @test log_wealth(res.w) > maximum(log_wealth(Matrix(I, N, N)[:, i]) for i in 1:N)
        # A window every asset covers carries no mask, as the other prior-free heads do,
        # and the result carries the carrier it read.
        @test isnothing(res.imsk)
        @test res.pr.X === rd.X || res.pr.X == rd.X
        # A step budget that stops early is read off the return code, not guarded.
        r1 = optimise(BestConstantRebalancedPortfolio(; iters = 1), rd)
        @test isa(r1.retcode, OptimisationSuccess) && !r1.retcode.res.converged
        @test r1.retcode.res.iterations == 1
        @test sum(r1.w) ≈ 1
    end

    @testset "Parity with MeanRisk under LogarithmicReturn" begin
        tight = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                       settings = Dict("verbose" => false, "tol_gap_abs" => 1e-12,
                                       "tol_gap_rel" => 1e-12, "tol_feas" => 1e-12,
                                       "max_iter" => 500))
        default = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                         settings = Dict("verbose" => false))
        res = optimise(BestConstantRebalancedPortfolio(), rd)
        for (slv, tol) in ((tight, 1e-8), (default, 1e-4))
            mr = MeanRisk(; obj = MaximumReturn(),
                          opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv,
                                              ret = LogarithmicReturn(),
                                              wb = WeightBounds(0, 1), bgt = 1))
            mres = optimise(mr, rd)
            @test isapprox(res.w, mres.w; atol = tol)
            # The programme's objective is the fixed point's, per observation.
            @test isapprox(T * po.JuMP.objective_value(mres.model), log_wealth(mres.w);
                           atol = 100 * tol)
            @test isapprox(log_wealth(res.w), log_wealth(mres.w); atol = 100 * tol)
        end
    end

    @testset "Simplex only: a binding wb is the finaliser's repair" begin
        free = optimise(BestConstantRebalancedPortfolio(), rd)
        capped = optimise(BestConstantRebalancedPortfolio(; wb = WeightBounds(0, 0.3)), rd)
        @test all(capped.w .<= 0.3 + 1e-12)
        @test sum(capped.w) ≈ 1
        @test !isapprox(free.w, capped.w; atol = 1e-6)
        # The repair is worse than the unconstrained optimum, as a repair must be.
        @test log_wealth(capped.w) < log_wealth(free.w)
        # The default bounds are the simplex, and the finaliser leaves the fixed point alone.
        sets = UniverseSets(; dict = Dict("nx" => nx))
        est = optimise(BestConstantRebalancedPortfolio(;
                                                       wb = WeightBoundsEstimator(;
                                                                                  lb = "A" =>
                                                                                      0.0,
                                                                                  ub = nothing),
                                                       sets = sets), rd)
        @test est.w ≈ free.w
    end

    @testset "Validation" begin
        @test_throws DomainError BestConstantRebalancedPortfolio(; iters = 0)
        @test_throws DomainError BestConstantRebalancedPortfolio(; tol = 0.0)
        @test_throws po.IsNothingError BestConstantRebalancedPortfolio(;
                                                                       wb = WeightBoundsEstimator(;
                                                                                                  lb = "A" =>
                                                                                                      0.1,
                                                                                                  ub = nothing))
        @test optimise(BestConstantRebalancedPortfolio(), rd; dims = 2).w ≈
              optimise(BestConstantRebalancedPortfolio(), rd).w
        @test_throws po.IsNothingError optimise(BestConstantRebalancedPortfolio(),
                                                ReturnsResult())
    end

    @testset "The Coverage Universe of the window" begin
        Xn = copy(X)
        Xn[7, 3] = NaN
        rdn = ReturnsResult(; nx = nx, X = Xn, ts = ts)
        res = optimise(BestConstantRebalancedPortfolio(), rdn)
        @test res.imsk == [true, true, false, true]
        @test res.w[3] == 0
        @test sum(res.w) ≈ 1
        # The dropped column changes nothing for the pair, because it held nothing.
        full = optimise(BestConstantRebalancedPortfolio(), rd)
        @test isapprox(res.w[[1, 2, 4]], full.w[[1, 2, 4]]; atol = 1e-8)
    end

    @testset "The online read-out is the batch fit (ADR 0137)" begin
        opt = BestConstantRebalancedPortfolio()
        stepped = opt
        for t in 1:T
            stepped = partial_fit!(stepped, po.port_opt_view(rd, t:t, :))
        end
        @test isa(stepped.cache, po.ReturnsBufferState)
        @test po.held_timestamps(stepped) == ts
        o = optimise(stepped)
        b = optimise(opt, rd)
        @test o.w ≈ b.w
        @test po.returns_result(stepped).X == X
        # A head that has folded nothing has nothing to read out.
        @test_throws ArgumentError optimise(opt)
        # `Online` on the prior-free head windows the rows it keeps.
        w = 10
        onl = po.update_online_estimator(po.Online(opt; max_history = w))
        for t in 1:T
            onl = partial_fit!(onl, po.port_opt_view(rd, t:t, :))
        end
        @test size(po.returns_result(onl).X, 1) == w
        @test optimise(onl).w ≈ optimise(opt, po.port_opt_view(rd, (T - w + 1):T, :)).w
        # The cache is running state and is not rendered.
        @test !(:cache in po.show_fields(stepped))
        @test !occursin("cache", sprint(show, stepped))
    end

    @testset "Through the walk-forward, and as a Hindsight Comparator" begin
        cv = IndexWalkForward(20, 1)
        pred = cross_val_predict(BestConstantRebalancedPortfolio(), rd, cv)
        @test isa(pred, MultiPeriodPredictionResult)
        @test length(pred.pred) == T - 20
        # The causal constant portfolio is not the hindsight one: fit on the rows it is
        # scored on, the comparator beats it on those rows.
        rd_test = po.port_opt_view(rd, 21:T, :)
        hind = predict(optimise(BestConstantRebalancedPortfolio(), rd_test), rd_test)
        @test sum(log1p, hind.rd.X) >= sum(log1p, pred.mrd.X)
        # A fallback composes through the ordinary chain.
        fbres = optimise(BestConstantRebalancedPortfolio(; fb = EqualWeighted()), rd)
        @test isa(fbres.retcode, OptimisationSuccess)
        @test !po.needs_previous_weights(BestConstantRebalancedPortfolio())
        @test po.needs_previous_weights(BestConstantRebalancedPortfolio(;
                                                                        fb = PreviousWeights()))
    end
end
