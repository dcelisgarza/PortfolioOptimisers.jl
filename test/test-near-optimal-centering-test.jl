@safetestset "NearOptimalCentering Optimisation" begin
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
    rf = 4.34 / 100 / 252
    rng = StableRNG(987456321)
    X = randn(rng, 200, 10)
    rd = ReturnsResult(; nx = 1:size(X, 2), X = X)
    @testset "Unconstrainted NearOptimalCentering" begin
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     check_sol = (; allow_local = true, allow_almost = true),
                     settings = Dict("max_step_fraction" => 0.75, "verbose" => false))
        pr = prior(EmpiricalPriorEstimator(), rd)
        opt = JuMPOptimiser(; pe = pr, slv = slv)
        objs = [MinimumRisk(), MaximumUtility(), MaximumRatio(), MaximumReturn()]
        bins = [1, 5, 10, 20, nothing, 50]
        w_min = optimise!(MeanRiskEstimator(; r = ConditionalValueatRisk(),
                                            obj = MinimumRisk(), opt = opt), rd).w
        w_max = optimise!(MeanRiskEstimator(; r = ConditionalValueatRisk(),
                                            obj = MaximumReturn(), opt = opt), rd).w
        df = CSV.read(joinpath(@__DIR__, "./assets/Unconstrained-NearOptimalCentering.csv"),
                      DataFrame)
        i = 1
        for obj ∈ objs
            for bin ∈ bins
                wt = df[!, i]
                noc1 = NearOptimalCenteringEstimator(; bins = bin,
                                                     r = ConditionalValueatRisk(),
                                                     obj = obj, opt = opt)
                w1 = optimise!(noc1, rd).w
                res1 = isapprox(w1, wt)
                if !res1
                    println("Iteration $i, compute everything in NOC failed.")
                    find_tol(w1, wt; name1 = :w1, name2 = :wt)
                end
                @test res1

                w_opt = optimise!(MeanRiskEstimator(; r = ConditionalValueatRisk(),
                                                    obj = obj, opt = opt), rd).w
                noc2 = NearOptimalCenteringEstimator(; w_min = w_min, w_max = w_max,
                                                     w_opt = w_opt, bins = bin,
                                                     r = ConditionalValueatRisk(),
                                                     obj = obj, opt = opt)
                w2 = optimise!(noc2, rd).w
                res2 = isapprox(w2, wt)
                if !res1
                    println("Iteration $i, initial vectors provided NOC failed.")
                    find_tol(w2, wt; name1 = :w2, name2 = :wt)
                end
                @test res2
                i += 1
            end
        end
    end
end
