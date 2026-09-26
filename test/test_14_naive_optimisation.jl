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

    # #885: `ReturnsResult` is always observations × assets, so `dims = 2` must be
    # refused, not answered by reading the observation axis as the asset count.
    res = optimise(EqualWeighted(), rd; dims = 1)
    @test isapprox(res.w, range(; start = inv(N), stop = inv(N), length = N))
    @test_throws PortfolioOptimisers.ConflictingArgumentError optimise(EqualWeighted(), rd;
                                                                       dims = 2)
    @test_throws PortfolioOptimisers.ConflictingArgumentError optimise(RandomWeighted(), rd;
                                                                       dims = 2)

    # The assertion splits the two refusals: a selector outside `(1, 2)` is malformed, and
    # `dims = 2` is a valid selector that conflicts with the fixed layout. The message names
    # the remedy and the argument the caller passed.
    @test isnothing(PortfolioOptimisers.assert_returns_result_dims(1))
    @test_throws DomainError PortfolioOptimisers.assert_returns_result_dims(3)
    err = try
        PortfolioOptimisers.assert_returns_result_dims(2, :d)
    catch e
        e
    end
    @test err isa PortfolioOptimisers.ConflictingArgumentError
    @test occursin("Pass d = 1", sprint(showerror, err))
    @test occursin("d => 2", sprint(showerror, err))

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
        # A `ReturnsResult` is observations × assets, so `dims = 2` is refused here too.
        @test_throws po.ConflictingArgumentError optimise(InverseVolatility(), rdx;
                                                          dims = 2)
        @test_throws DomainError optimise(InverseVolatility(), rdx; dims = 3)
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
