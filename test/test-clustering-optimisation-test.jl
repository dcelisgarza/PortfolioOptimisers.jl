@safetestset "Clustering Optimistaion" begin
    using PortfolioOptimisers, CSV, DataFrames, Test, StableRNGs, Random, Clarabel
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
    @testset "Hierarchical Risk Parity" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 20)
        rd = ReturnsData(; nx = 1:size(X, 2), X = X)

        pm = prior(HighOrderPriorEstimator(), rd)
        clm = clusterise(ClusteringEstimator(), pm.X)
        opt = HierarchicalOptimiser(; pe = pm, cle = clm,
                                    slv = Solver(; name = :clarabel,
                                                 solver = Clarabel.Optimizer,
                                                 settings = Dict("max_step_fraction" => 0.75,
                                                                 "verbose" => false)))
        df = CSV.read(joinpath(@__DIR__, "./assets/HRP.csv"), DataFrame)
        risks = [StandardDeviation(), SemiStandardDeviation(), Variance(), SemiVariance(),
                 BrownianDistanceVariance(), MeanAbsoluteDeviation(),
                 NegativeQuadraticSkewness(), NegativeSkewness(),
                 NegativeQuadraticSemiSkewness(), NegativeSemiSkewness(),
                 SquareRootKurtosis(), SquareRootSemiKurtosis()]
        for (i, risk) ∈ enumerate(risks)
            w1 = PortfolioOptimisers.optimise!(HierarchicalRiskParity(; r = risk,
                                                                      opt = opt))
            res = isapprox(w1, df[:, i])
            if !res
                println("$risk failed")
                find_tol(w1, df[:, i]; name1 = :w1, name2 = :df)
            end
            @test res
        end
    end
end
