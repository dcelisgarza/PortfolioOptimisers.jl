#=
@safetestset "Relaxed Risk Budgetting Optimisation" begin
    using PortfolioOptimisers, CSV, DataFrames, Test, Random, Clarabel, TimeSeries
    function find_tol(a1, a2; name1 = :a1, name2 = :a2)
        for rtol in
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; rtol = rtol)
                println("isapprox($name1, $name2, rtol = $(rtol))")
                break
            end
        end
        for atol in
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; atol = atol)
                println("isapprox($name1, $name2, atol = $(atol))")
                break
            end
        end
    end
    X = TimeArray(CSV.File(joinpath(@__DIR__, "./assets/asset_prices.csv"));
                  timestamp = :timestamp)
    F = TimeArray(CSV.File(joinpath(@__DIR__, "./assets/factor_prices.csv"));
                  timestamp = :timestamp)
    rd = prices_to_returns(X[(end - 252):end], F[(end - 252):end])
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = Dict("max_step_fraction" => 0.75, "verbose" => false))
    pr = prior(FactorPrior(; re = DimensionReductionRegression()), rd)
    opt = JuMPOptimiser(; pe = pr, slv = slv)
    algs = (BasicRelaxedRiskBudgetting(),
            RegularisedRelaxedRiskBudgetting(),
            RegularisedPenalisedRelaxedRiskBudgetting(),
            RegularisedPenalisedRelaxedRiskBudgetting(; p = 50))
    rkbs = (nothing, 1:30)
    df = CSV.read(joinpath(@__DIR__, "./assets/Relaxed-Risk-Budgetting.csv"), DataFrame)
    i = 1
    for alg in algs
        for rkb in rkbs
            rbe = RelaxedRiskBudgetting(; rkb = rkb, alg = alg, opt = opt)
            w = optimise!(rbe, rd).w
            wt = df[!, "$i"]
            res = isapprox(w, wt; rtol = 1e-6)
            if !res
                println("Iteration $i failed.")
                find_tol(w, wt; name1 = :w, name2 = :wt)
            end
            @test res
            i += 1
        end
    end
end
=#
