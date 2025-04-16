@safetestset "Risk Measure Tests" begin
    using PortfolioOptimisers, Test, Random, StableRNGs, CSV, DataFrames
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
        pr1 = prior(HighOrderPriorEstimator(), X)
        rs = [Variance(), StandardDeviation(), UncertaintySetVariance(),
              LowOrderMoment(; alg = FirstLowerMoment()),
              LowOrderMoment(; alg = SemiDeviation()),
              LowOrderMoment(; alg = SemiVariance()),
              LowOrderMoment(; alg = MeanAbsoluteDeviation()),
              HighOrderMoment(; alg = ThirdLowerMoment()),
              HighOrderMoment(; alg = FourthLowerMoment()),
              HighOrderMoment(; alg = FourthCentralMoment()),
              HighOrderMoment(; alg = HighOrderDeviation(; alg = ThirdLowerMoment())),
              HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthLowerMoment())),
              HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthCentralMoment()))]
        #! TODO: actual tests.
        r = risk_measure_factory(rs, pr1)
    end
end
