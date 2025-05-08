@safetestset "Risk Budgetting Optimisation" begin
    using PortfolioOptimisers, CSV, DataFrames, Test, StableRNGs, Random, Clarabel,
          StatsBase, LinearAlgebra, Pajarito, HiGHS, JuMP
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

    rng = StableRNG(123654789)
    X = randn(rng, 200, 10)
    F = X[:, [3, 8, 5]]
    rd = ReturnsResult(; nx = 1:size(X, 2), X = X, F = F, nf = 1:size(F, 2))
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = Dict("max_step_fraction" => 0.75, "verbose" => false))

    @testset "Asset Risk Budgetting" begin
        pr = prior(EmpiricalPriorEstimator(), rd)
        opt = JuMPOptimiser(; pe = pr, slv = slv)
        r = PortfolioOptimisers.risk_measure_factory(StandardDeviation(), pr)

        rbe = RiskBudgettingEstimator(; r = r, opt = opt)
        w = optimise!(rbe, rd).w
        rkc = PortfolioOptimisers.risk_contribution(r, w, pr.X)
        lo, hi = extrema(rkc)
        @test isapprox(hi / lo, 1, rtol = 5e-5)

        rbe = RiskBudgettingEstimator(; alg = AssetRiskBudgettingAlgorithm(; rkb = 1:10),
                                      r = r, opt = opt)
        w = optimise!(rbe, rd).w
        rkc = PortfolioOptimisers.risk_contribution(r, w, pr.X)
        lo, hi = extrema(rkc)
        @test isapprox(hi / lo, 10, rtol = 1e-4)
        @test argmin(rkc) == 1
        @test argmax(rkc) == 10

        rbe = RiskBudgettingEstimator(; alg = AssetRiskBudgettingAlgorithm(; rkb = 20:-2:2),
                                      r = r, opt = opt)
        w = optimise!(rbe, rd).w
        rkc = PortfolioOptimisers.risk_contribution(r, w, pr.X)
        lo, hi = extrema(rkc)
        @test isapprox(hi / lo, 10, rtol = 1e-4)
        @test argmin(rkc) == 10
        @test argmax(rkc) == 1
    end
end
