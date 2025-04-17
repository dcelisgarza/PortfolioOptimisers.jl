@safetestset "Risk Measure Tests" begin
    using PortfolioOptimisers, Test, Random, StableRNGs, CSV, DataFrames, StatsBase,
          Clarabel
    import PortfolioOptimisers: risk_measure_factory
    function find_tol(a1, a2; name1 = :a1, name2 = :a2)
        for rtol ∈
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; rtol = rtol)
                println("isapprox($name1, $name2, rtol = $(rtol))")
                break
            end
        end
    end
    @testset "Risk" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 20)
        A = rand(rng, 3, 20)
        B = rand(rng, 3)
        ew = eweights(1:1000, 1 / 1000; scale = true)
        pr1 = prior(HighOrderPriorEstimator(), X)

        slv = Solver(; name = :Clarabel, solver = Clarabel.Optimizer,
                     settings = Dict("verbose" => false))
        ucs1 = DeltaUncertaintySetEstimator(;)
        ucs2 = NormalUncertaintySetEstimator(;)
        settings = RiskMeasureSettings(; rke = false, scale = -1, ub = 3)
        formulation = RSOC()
        sigma = pr1.sigma * 1.5
        rc = LinearConstraintResult(; ineq = PartialLinearConstraintResult(; A = A, B = B))
        rs = [Variance(; settings = settings, formulation = formulation, sigma = sigma,
                       rc = rc), Variance(;)]
        r = risk_measure_factory(rs, pr1, Ref(slv), Ref(ucs2))
        @test r[1].settings === settings
        @test r[1].formulation === formulation
        @test r[1].sigma === sigma
        @test r[1].rc === rc
        @test r[2].sigma === pr1.sigma

        sigma = pr1.sigma * 2
        rs = [StandardDeviation(; settings = settings, sigma = sigma), StandardDeviation(;)]
        r = risk_measure_factory(rs, pr1, Ref(slv), Ref(ucs2))
        @test r[1].settings === settings
        @test r[1].sigma === sigma
        @test r[2].sigma === pr1.sigma

        sigma = pr1.sigma * 2.5
        rs = [UncertaintySetVariance(; settings = settings, ucs = ucs1, sigma = sigma),
              UncertaintySetVariance(;)]
        r = risk_measure_factory(rs, pr1, Ref(slv), Ref(ucs2))
        @test r[1].settings === settings
        @test r[1].ucs === ucs
        @test r[1].sigma === sigma
        @test r[2].sigma === pr1.sigma
        @test r[2].ucs === ucs2

        # rs = [LowOrderMoment(; settings = s, alg = FirstLowerMoment()),
        #       LowOrderMoment(; settings = s, alg = FirstLowerMoment(), target = fill(0.0, 20)),
        #       LowOrderMoment(; settings = s, alg = FirstLowerMoment(), target = nothing,
        #                      mu = fill(0.0, 20)),
        #       LowOrderMoment(; settings = s, alg = SemiDeviation()),
        #       LowOrderMoment(; settings = s, alg = SemiVariance()),
        #       LowOrderMoment(; settings = s, alg = MeanAbsoluteDeviation()),
        #       LowOrderMoment(; settings = s, alg = MeanAbsoluteDeviation(; w = ew))]
        # rs = [HighOrderMoment(; settings = s, alg = ThirdLowerMoment()),
        #       HighOrderMoment(; settings = s, alg = FourthLowerMoment()),
        #       HighOrderMoment(; settings = s, alg = FourthCentralMoment()),
        #       HighOrderMoment(; settings = s, alg = HighOrderDeviation(; alg = ThirdLowerMoment())),
        #       HighOrderMoment(; settings = s,
        #                       alg = HighOrderDeviation(; alg = FourthLowerMoment())),
        #       HighOrderMoment(; settings = s,
        #                       alg = HighOrderDeviation(; alg = FourthCentralMoment()))]
        # rs = [SquareRootKurtosis(; settings = s, alg = Full()),
        #       SquareRootKurtosis(; settings = s, w = ew),
        #       SquareRootKurtosis(; settings = s, mu = fill(0.0, 20)),
        #       SquareRootKurtosis(; settings = s, kt = pr1.skt),
        #       SquareRootKurtosis(; settings = s, alg = Semi())]
        #! TODO: actual tests.
    end
end
