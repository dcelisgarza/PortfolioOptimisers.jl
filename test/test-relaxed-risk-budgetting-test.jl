@safetestset "Relaxed Risk Budgetting Optimisation" begin
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
        for atol ∈
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; atol = atol)
                println("isapprox($name1, $name2, atol = $(atol))")
                break
            end
        end
    end
    rng = StableRNG(123654789)
    X = randn(rng, 200, 10)
    rd = ReturnsResult(; nx = 1:size(X, 2), X = X)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = Dict("max_step_fraction" => 0.75, "verbose" => false))

    pr = prior(EmpiricalPriorEstimator(), rd)
    opt = JuMPOptimiser(; pe = pr, slv = slv)
    r = PortfolioOptimisers.factory(StandardDeviation(), pr)
    algs = (BasicRelaxedRiskBudgettingAlgorithm(),
            RegularisationRelaxedRiskBudgettingAlgorithm(),
            RegularisationPenaltyRelaxedRiskBudgettingAlgorithm(),
            RegularisationPenaltyRelaxedRiskBudgettingAlgorithm(; p = 50))
    rkbs = (nothing, 1:10)
    df = CSV.read(joinpath(@__DIR__, "./assets/Relaxed-Risk-Budgetting.csv"), DataFrame)
    i = 1
    for alg ∈ algs
        for rkb ∈ rkbs
            rbe = RelaxedRiskBudgetting(; rkb = rkb, alg = alg, opt = opt)
            w = optimise!(rbe, rd).w
            wt = df[!, "$i"]
            res = isapprox(w, wt)
            if !res
                println("Iteration $i failed.")
                find_tol(w, wt; name1 = :w, name2 = :wt)
            end
            @test res
            i += 1
        end
    end
end
