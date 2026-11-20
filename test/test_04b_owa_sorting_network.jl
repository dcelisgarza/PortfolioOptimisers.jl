@testset "OWA sorting-network formulation" begin
    using PortfolioOptimisers, Clarabel, StableRNGs, LinearAlgebra, JuMP, Test
    PO = PortfolioOptimisers

    #=
    `SortingNetworkOrderedWeightsArray` relaxes a sorting network. Each comparator becomes three
    linear rows, and for a nondecreasing weight vector the minimum of the weighted outputs
    is the OWA risk, so the formulation must agree with `ExactOrderedWeightsArray` to the
    solver tolerance for every convex weight builder, on both axes: the network (`merge`)
    and the order of its variables (`ordering`).
    =#
    @testset "Networks sort, orders permute" begin
        rng = StableRNG(7)
        for merge in (OddEvenMerge(), BitonicMerge()),
            T in (1, 2, 3, 5, 7, 8, 13, 64, 100, 257)

            c = PO.sorting_network(merge, T)
            @test all(((i, j),) -> 1 <= i <= T && 1 <= j <= T && i != j, c)
            for _ in 1:20
                x = randn(rng, T)
                for (i, j) in c
                    x[i], x[j] = min(x[i], x[j]), max(x[i], x[j])
                end
                @test issorted(x)
            end
            for ordering in (ComparatorOrdering(), WireOrdering())
                @test sort(PO.sorting_network_columns(ordering, c)) == 1:(2 * length(c))
            end
        end
        # The comparator counts the docstrings state.
        @test length(PO.sorting_network(OddEvenMerge(), 250)) == 3756
        @test length(PO.sorting_network(OddEvenMerge(), 500)) == 9511
        @test length(PO.sorting_network(BitonicMerge(), 250)) == 4405
        @test length(PO.sorting_network(BitonicMerge(), 500)) == 11026
        # At a power of two the counts are Batcher's closed forms, of order T log² T.
        for k in 1:10
            @test length(PO.sorting_network(OddEvenMerge(), 2^k)) ==
                  (k^2 - k + 4) * 2^k ÷ 4 - 1
            @test length(PO.sorting_network(BitonicMerge(), 2^k)) == k * (k + 1) * 2^k ÷ 4
        end
        # The wire order keeps the values of one position together, in the order of writing.
        c = PO.sorting_network(OddEvenMerge(), 8)
        cols = PO.sorting_network_columns(WireOrdering(), c)
        wire = collect(Iterators.flatten(c))
        @test issorted(wire[invperm(cols)])
        @test SortingNetworkOrderedWeightsArray() ===
              SortingNetworkOrderedWeightsArray(; merge = BitonicMerge(),
                                                ordering = ComparatorOrdering())
    end

    slv = Solver(; name = :c, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false, "tol_gap_abs" => 1e-10,
                                 "tol_gap_rel" => 1e-10, "tol_feas" => 1e-10),
                 check_sol = (; allow_local = true, allow_almost = true))
    X = 0.01 * randn(StableRNG(295), 120, 6) .+ 0.0005
    rd = ReturnsResult(; nx = string.('A':'F'), X = X)
    pr = prior(EmpiricalPrior(), rd)
    opt = JuMPOptimiser(; pe = pr, slv = slv)
    algs = (SortingNetworkOrderedWeightsArray(),
            SortingNetworkOrderedWeightsArray(; merge = OddEvenMerge(),
                                              ordering = WireOrdering()))

    @testset "Exact risk for every convex weight builder" begin
        ws = [owa_gmd, owa_cvar, owa_tg, owa_wr, owa_rg, owa_cvarrg, owa_tgrg,
              owa_l_moment_crm]
        for w in ws, alg in algs
            re = optimise(MeanRisk(;
                                   r = OrderedWeightsArray(; w = w,
                                                           alg = ExactOrderedWeightsArray()),
                                   obj = MinimumRisk(), opt = opt))
            ri = optimise(MeanRisk(; r = OrderedWeightsArray(; w = w, alg = alg),
                                   obj = MinimumRisk(), opt = opt))
            f = OrderedWeightsArray(; w = w)
            @test isapprox(f(X * ri.w), f(X * re.w); rtol = 1e-6)
            # The model's risk is the risk of its own weights, so it is exact, not a bound.
            @test isapprox(JuMP.value(ri.model[:sowa_risk_1]), f(X * ri.w); rtol = 1e-6)
        end
    end

    @testset "The risk at fixed weights is exact for any nondecreasing vector" begin
        # Not only the builders: a random sorted vector, of one sign or of mixed signs, at
        # weights that the bounds fix, so the model's risk is the risk of those weights.
        rng = StableRNG(11)
        for T in (7, 33), mixed in (false, true), alg in algs
            Xf = 0.01 * randn(rng, T, 4) .+ 0.0005
            w0 = rand(rng, 4)
            w0 ./= sum(w0)
            ow = sort(mixed ? randn(rng, T) : rand(rng, T))
            prf = prior(EmpiricalPrior(), ReturnsResult(; nx = string.('A':'D'), X = Xf))
            res = optimise(MeanRisk(; r = OrderedWeightsArray(; w = ow, alg = alg),
                                    obj = MinimumRisk(),
                                    opt = JuMPOptimiser(; pe = prf, slv = slv,
                                                        wb = WeightBounds(; lb = w0,
                                                                          ub = w0))))
            @test isapprox(JuMP.value(res.model[:sowa_risk_1]),
                           OrderedWeightsArray(; w = ow)(Xf * w0); rtol = 1e-6)
        end
    end

    @testset "A weight vector that decreases is refused" begin
        w = owa_l_moment(10, 4)
        @test !issorted(w)
        @test_throws DomainError OrderedWeightsArray(; w = w,
                                                     alg = SortingNetworkOrderedWeightsArray())
        # The range checks `w1 - w2` with `w2` as stored, which here is a decreasing vector.
        @test !issorted(reverse(owa_tg(10)) - zeros(10))
        @test_throws DomainError OrderedWeightsArrayRange(; w1 = reverse(owa_tg(10)),
                                                          w2 = zeros(10),
                                                          alg = SortingNetworkOrderedWeightsArray())
        # The exact formulation still takes the vector, so only the network refuses it.
        @test OrderedWeightsArray(; w = w, alg = ExactOrderedWeightsArray()).w === w
        # A callable is checked when the model is built.
        r = OrderedWeightsArray(; w = T -> owa_l_moment(T, 4),
                                alg = SortingNetworkOrderedWeightsArray())
        err = try
            optimise(MeanRisk(; r = r, obj = MinimumRisk(), opt = opt))
            nothing
        catch e
            e
        end
        while isa(err, TaskFailedException) || isa(err, CompositeException)
            err = isa(err, TaskFailedException) ? err.task.exception : first(err.exceptions)
        end
        @test isa(err, DomainError)
    end

    @testset "Bound row, ratio, and the fused range" begin
        re = optimise(MeanRisk(;
                               r = OrderedWeightsArray(; alg = ExactOrderedWeightsArray()),
                               obj = MinimumRisk(), opt = opt))
        ub = 1.2 * OrderedWeightsArray()(X * re.w)
        for alg in algs
            # A bound row on the risk of a maximum-return portfolio.
            rb = optimise(MeanRisk(;
                                   r = OrderedWeightsArray(;
                                                           settings = RiskMeasureSettings(;
                                                                                          ub = ub),
                                                           alg = alg),
                                   obj = MaximumReturn(), opt = opt))
            rbe = optimise(MeanRisk(;
                                    r = OrderedWeightsArray(;
                                                            settings = RiskMeasureSettings(;
                                                                                           ub = ub),
                                                            alg = ExactOrderedWeightsArray()),
                                    obj = MaximumReturn(), opt = opt))
            @test OrderedWeightsArray()(X * rb.w) <= ub * (1 + 1e-6)
            @test isapprox(dot(pr.mu, rb.w), dot(pr.mu, rbe.w); rtol = 1e-6)
            # The ratio objective homogenises the weights, and the network scales with them.
            rr = optimise(MeanRisk(; r = OrderedWeightsArray(; alg = alg),
                                   obj = MaximumRatio(), opt = opt))
            rre = optimise(MeanRisk(;
                                    r = OrderedWeightsArray(;
                                                            alg = ExactOrderedWeightsArray()),
                                    obj = MaximumRatio(), opt = opt))
            f = OrderedWeightsArray()
            @test isapprox(dot(pr.mu, rr.w) / f(X * rr.w), dot(pr.mu, rre.w) / f(X * rre.w);
                           rtol = 1e-5)
            # The range fuses its tails into one network over `w1 - w2`.
            rg = optimise(MeanRisk(; r = OrderedWeightsArrayRange(; alg = alg),
                                   obj = MinimumRisk(), opt = opt))
            rge = optimise(MeanRisk(;
                                    r = OrderedWeightsArrayRange(;
                                                                 alg = ExactOrderedWeightsArray()),
                                    obj = MinimumRisk(), opt = opt))
            g = OrderedWeightsArrayRange()
            @test isapprox(g(X * rg.w), g(X * rge.w); rtol = 1e-6)
            @test isapprox(JuMP.value(rg.model[:sowa_range_risk_1]), g(X * rg.w);
                           rtol = 1e-6)
        end
    end
end
