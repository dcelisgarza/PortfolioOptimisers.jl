@testset "Naive optimisation tests" begin
    using PortfolioOptimisers, CSV, Test, TimeSeries, LinearAlgebra, StableRNGs
    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end])
    pr = prior(EmpiricalPrior(), rd)
    N = size(pr.X, 2)

    sets = UniverseSets(; dict = Dict("nx" => rd.nx))
    wb = WeightBoundsEstimator(; lb = "AAPL" => 0.07, ub = nothing)

    res = optimise(InverseVolatility(; pe = pr), rd)
    w = inv.(sqrt.(LinearAlgebra.diag(pr.sigma)))
    w /= sum(w)
    @test isapprox(res.w, w)

    res = optimise(EqualWeighted(), rd)
    @test isapprox(res.w, range(; start = inv(N), stop = inv(N), length = N))

    res = optimise(RandomWeighted(; rng = StableRNG(123456789)), rd)
    @test isapprox(sum(res.w), 1)

    @test optimise(RandomWeighted(; alpha = 5, rng = StableRNG(123456789), seed = 42),
                   rd).w == optimise(RandomWeighted(; alpha = fill(5, size(rd.X, 2)),
                              rng = StableRNG(123456789), seed = 42), rd).w

    res = optimise(RandomWeighted(), rd)
    @test isapprox(sum(res.w), 1)

    res = optimise(InverseVolatility(; wb = wb, sets = sets, pe = pr), rd)
    w = inv.(sqrt.(LinearAlgebra.diag(pr.sigma)))
    @test isapprox(res.w[1], 0.07)

    res = optimise(EqualWeighted(; wb = wb, sets = sets), rd)
    @test isapprox(res.w[1], 0.07)
    @test all(isapprox.(res.w[2:end], (1 - 0.07) / 19))

    res = optimise(RandomWeighted(; wb = wb, sets = sets, rng = StableRNG(123456789)), rd)
    @test isapprox(res.w[1], 0.07)
    @test isapprox(sum(res.w), 1)

    @testset "#1348: a carrier holds its observations along the rows, whatever `dims`" begin
        # A `ReturnsResult` is always observations × assets. No function that takes a
        # carrier has a `dims` keyword, and every bridge from a carrier to a matrix method
        # passes `dims = 1` after the caller's keywords, so a stray `dims` has no effect.
        # Before #1348, `MeanRisk` gave one weight per observation for `dims = 2`, and
        # `HierarchicalRiskParity` threw a `DimensionMismatch`.
        using Clarabel
        po = PortfolioOptimisers
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     settings = Dict("verbose" => false),
                     check_sol = (; allow_local = true, allow_almost = true))
        rds = po.port_opt_view(rd, :, 1:6)
        for opt in (EqualWeighted(), RandomWeighted(; rng = StableRNG(1), seed = 1348),
                    InverseVolatility(), BestConstantRebalancedPortfolio(),
                    MeanRisk(; opt = JuMPOptimiser(; slv = slv)), HierarchicalRiskParity())
            w1 = optimise(opt, rds).w
            @test length(w1) == 6
            for d in (1, 2, 3)
                @test optimise(opt, rds; dims = d).w ≈ w1
            end
        end
        pr1 = prior(EmpiricalPrior(), rds)
        @test prior(EmpiricalPrior(), rds; dims = 2).sigma ≈ pr1.sigma
        @test size(prior(EmpiricalPrior(), rds; dims = 2).sigma) == (6, 6)
        cle = ClustersEstimator()
        @test clusterise(cle, rds; dims = 2).res.order == clusterise(cle, rds).res.order
        @test clusterise(cle, pr1; dims = 2).res.order == clusterise(cle, pr1).res.order
        uc = NormalUncertaintySet(; rng = StableRNG(1), seed = 1348, n_sim = 20)
        @test mu_ucs(uc, rds; dims = 2).ub ≈ mu_ucs(uc, rds).ub
        @test po.coverage_reduction(rds)[2].X == rds.X
        # A matrix method keeps `dims`: it reads the transpose as the same sample.
        @test prior(EmpiricalPrior(), permutedims(rds.X); dims = 2).sigma ≈ pr1.sigma
    end

    @testset "The docstrings of 03_NaiveOptimisation.jl against numbers" begin
        using Dates, Statistics, Random
        po = PortfolioOptimisers
        rng = StableRNG(875)
        T, N = 200, 6
        X = 0.01 .* randn(rng, T, N) .* (1:N)'
        nx = ["A$i" for i in 1:N]
        ts = Date(2024, 1, 1) .+ Day.(0:(T - 1))
        rdx = ReturnsResult(; nx = nx, X = X, ts = ts)
        prx = prior(EmpiricalPrior(), rdx)
        s2 = LinearAlgebra.diag(prx.sigma)

        # InverseVolatility: the closed form for both `sq`, and the naive risk parity
        # sub-portfolio that HierarchicalRiskParity builds from the unitary risks.
        @test optimise(InverseVolatility(), rdx).w ≈ s2 .^ -0.5 / sum(s2 .^ -0.5)
        @test optimise(InverseVolatility(; sq = true), rdx).w ≈ inv.(s2) / sum(inv.(s2))
        rkv = po.unitary_expected_risks(Variance(; sigma = prx.sigma), X, nothing)
        rks = po.unitary_expected_risks(StandardDeviation(; sigma = prx.sigma), X, nothing)
        @test optimise(InverseVolatility(; sq = true), rdx).w ≈ inv.(rkv) / sum(inv.(rkv))
        @test optimise(InverseVolatility(), rdx).w ≈ inv.(rks) / sum(inv.(rks))
        # A precomputed prior needs no returns, and it is refused at the external door.
        @test optimise(InverseVolatility(; pe = prx), ReturnsResult()).w ≈
              s2 .^ -0.5 / sum(s2 .^ -0.5)
        @test_throws ArgumentError po.assert_external_optimiser(InverseVolatility(;
                                                                                  pe = prx))
        # `brt` fits the prior on the returns in excess of the benchmark.
        Bm = repeat(0.005 .* randn(StableRNG(2), T), 1, N)
        rdb = ReturnsResult(; nx = nx, X = X, ts = ts, nb = nx, B = Bm)
        sb = LinearAlgebra.diag(prior(EmpiricalPrior(),
                                      ReturnsResult(; nx = nx, X = X - Bm, ts = ts)).sigma)
        @test optimise(InverseVolatility(; brt = true), rdb).w ≈
              sb .^ -0.5 / sum(sb .^ -0.5)
        # A `ReturnsResult` is observations × assets, so a `dims` has no effect (#1348).
        @test optimise(InverseVolatility(), rdx; dims = 2).w ≈
              optimise(InverseVolatility(), rdx).w
        @test_throws ArgumentError optimise(InverseVolatility(;
                                                              pe = po.Online(EmpiricalPrior())),
                                            rdx)

        # An integer sample gives the weights of its float copy, in a float type, and a
        # Float32 sample keeps Float32. Free bounds, a bound estimator and the NaN of a
        # failed hold all threw InexactError before.
        Xi = round.(Int, X .* 1000)
        rdi = ReturnsResult(; nx = nx, X = Xi, ts = ts)
        rdf = ReturnsResult(; nx = nx, X = Float64.(Xi), ts = ts)
        sets = UniverseSets(; dict = Dict("nx" => nx))
        wbe = WeightBoundsEstimator(; lb = "A1" => 0.07, ub = nothing)
        for opt in (InverseVolatility(), EqualWeighted(), RandomWeighted(; seed = 1),
                    InverseVolatility(; wb = wbe, sets = sets),
                    EqualWeighted(; wb = wbe, sets = sets),
                    RandomWeighted(; wb = wbe, sets = sets, seed = 1))
            wi = optimise(opt, rdi).w
            @test eltype(wi) === Float64
            @test wi == optimise(opt, rdf).w
        end
        wpi = optimise(PreviousWeights(), rdi).w
        @test eltype(wpi) === Float64 && all(isnan, wpi) && length(wpi) == N
        Xb = rand(StableRNG(3), 0:1, 40, 3)
        rdbi = ReturnsResult(; nx = nx[1:3], X = Xb, ts = ts[1:40])
        rdbf = ReturnsResult(; nx = nx[1:3], X = Float64.(Xb), ts = ts[1:40])
        @test optimise(BestConstantRebalancedPortfolio(), rdbi).w ≈
              optimise(BestConstantRebalancedPortfolio(), rdbf).w
        rd32 = ReturnsResult(; nx = nx, X = Float32.(X), ts = ts)
        for opt in (InverseVolatility(), EqualWeighted(),
                    BestConstantRebalancedPortfolio(; tol = 1.0f-5), PreviousWeights())
            @test eltype(optimise(opt, rd32).w) === Float32
        end
        # The Dirichlet draw takes the float type of `alpha`, not of the data.
        @test eltype(optimise(RandomWeighted(; seed = 1), rd32).w) === Float64
        @test eltype(optimise(RandomWeighted(; alpha = 1.0f0, seed = 1), rdx).w) === Float32

        # The prior-free heads spread over the Coverage Universe and carry it as `imsk`.
        Xn = copy(X)
        Xn[5, 2] = NaN
        rdn = ReturnsResult(; nx = nx, X = Xn, ts = ts)
        ew = optimise(EqualWeighted(), rdn)
        @test ew.imsk == [true, false, true, true, true, true]
        @test ew.w == [0.2, 0.0, 0.2, 0.2, 0.2, 0.2]
        rw = optimise(RandomWeighted(; seed = 3), rdn)
        @test rw.imsk == ew.imsk && rw.w[2] == 0 && sum(rw.w) ≈ 1
        rdd = ReturnsResult(; nx = nx, X = fill(NaN, T, N), ts = ts)
        for opt in (EqualWeighted(), RandomWeighted(), BestConstantRebalancedPortfolio())
            @test_throws po.IsEmptyError optimise(opt, rdd)
        end

        # RandomWeighted: the Dirichlet mean, the slice of a vector alpha, its length check,
        # and the draws move nearer their mean as the concentration grows.
        alpha = collect(1.0:N)
        rga = StableRNG(9)
        wm = zeros(N)
        for _ in 1:5000
            wm .+= optimise(RandomWeighted(; alpha = alpha, rng = rga), rdx).w
        end
        @test isapprox(wm / 5000, alpha / sum(alpha); atol = 5e-3)
        kept = [1, 3, 4, 5, 6]
        @test optimise(RandomWeighted(; alpha = alpha, seed = 3), rdn).w[kept] ==
              optimise(RandomWeighted(; alpha = alpha[kept], seed = 3),
                       ReturnsResult(; nx = nx[kept], X = X[:, kept], ts = ts)).w
        @test_throws DimensionMismatch optimise(RandomWeighted(; alpha = alpha[1:5]), rdn)
        function spread(a)
            rgs = StableRNG(1)
            return mean(maximum(abs,
                                optimise(RandomWeighted(; alpha = a, rng = rgs), rdx).w .-
                                inv(N)) for _ in 1:500)
        end
        @test spread(1) > spread(10) > spread(100)

        # BestConstantRebalancedPortfolio: the multipliers keep the simplex, the log wealth
        # does not fall, a zero entry stays zero, the optimum meets the first-order
        # condition, and the certificate bounds the shortfall at every iterate.
        Xr = 1 .+ X
        mult(b) = Xr' * inv.(Xr * b) / T
        lwf(b) = sum(log, Xr * b)
        b = fill(1 / N, N)
        b[4] = 0
        b ./= sum(b)
        fp = po.cover_fixed_point(Xr, 200_000, 1e-14)
        lws = Float64[]
        for _ in 1:100
            g = mult(b)
            @test isapprox(sum(b .* g), 1; atol = 1e-14)
            @test fp.log_wealth - lwf(b) <= T * log(maximum(g)) + 1e-12
            push!(lws, lwf(b))
            b = b .* g
        end
        @test all(diff(lws) .>= -1e-12)
        @test b[4] == 0
        gs = mult(fp.w)
        @test all(gs .<= 1 + 1e-8)
        @test all(abs.(gs[fp.w .> 1e-6] .- 1) .< 1e-8)
        # At a corner the budget can stop first, and the certificate still bounds the
        # shortfall.
        Xc = 1 .+ (0.02 .* randn(StableRNG(7), 60, 4) .+ [0.002 0 -0.001 0.0])
        fpc = po.cover_fixed_point(Xc, 20_000, 1e-12)
        fpl = po.cover_fixed_point(Xc, 400_000, 1e-12)
        @test !fpc.converged && fpc.iterations == 20_000
        @test fpl.converged
        @test 0 <= fpl.log_wealth - fpc.log_wealth <= fpc.gap

        # PreviousWeights holds `w` as it is, a view slices it without a rescale, the
        # constructor refuses a non-finite entry, and a fit with no weights fails.
        wv = [0.1, 0.2, 0.3, 0.15, 0.15, 0.1]
        rp = optimise(PreviousWeights(; w = wv), rdx)
        @test rp.w === wv
        @test isnothing(rp.imsk) && isnothing(rp.wb) && rp.retcode isa OptimisationSuccess
        @test po.port_opt_view(PreviousWeights(; w = wv), [2, 4], X).w == [0.2, 0.15]
        @test_throws DomainError PreviousWeights(; w = [NaN, 1.0])
        rn = optimise(PreviousWeights(), rdx)
        @test rn.retcode isa OptimisationFailure && all(isnan, rn.w)
        @test isnothing(optimise(PreviousWeights(), ReturnsResult()).w)
        @test po.needs_previous_weights(PreviousWeights())
        @test po.factory(PreviousWeights(; fb = PreviousWeights()), wv).fb.w == wv

        # The keyword constructor of the result expands onto `imsk`; the positional one
        # does not.
        m = BitVector([1, 0, 1])
        @test po.NaiveOptimisationResult(; pr = nothing, wb = nothing,
                                         retcode = OptimisationSuccess(), w = [0.5, 0.5],
                                         imsk = m, fb = nothing).w == [0.5, 0.0, 0.5]
        @test po.NaiveOptimisationResult(nothing, nothing, nothing, OptimisationSuccess(),
                                         [0.5, 0.5], m, nothing).w == [0.5, 0.5]

        # A fee whose turnover is not fixed, and a PreviousWeights fallback, each need the
        # previous weights; a bare head does not.
        tn = Turnover(; w = fill(1 / N, N), val = 0.01)
        @test po.needs_previous_weights(EqualWeighted(; fees = Fees(; tn = tn)))
        @test po.needs_previous_weights(EqualWeighted(; fb = PreviousWeights()))
        @test !po.needs_previous_weights(EqualWeighted())
        @test !po.is_time_dependent(EqualWeighted())
    end
end

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
