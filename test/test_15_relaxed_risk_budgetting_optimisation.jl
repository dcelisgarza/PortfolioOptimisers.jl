@safetestset "Relaxed Risk Budgeting" begin
    using Test, PortfolioOptimisers, DataFrames, CSV, TimeSeries, Clarabel, StatsBase,
          LinearAlgebra
    function find_tol(a1, a2; name1 = :lhs, name2 = :rhs)
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
    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end],
                           TimeArray(CSV.File(joinpath(@__DIR__, "./assets/Factors.csv.gz"));
                                     timestamp = :Date)[(end - 252):end])
    slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false)),
           Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.95)),
           Solver(; name = :clarabel3, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.9)),
           Solver(; name = :clarabel4, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.85)),
           Solver(; name = :clarabel5, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.80)),
           Solver(; name = :clarabel6, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.75)),
           Solver(; name = :clarabel7, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.7)),
           Solver(; name = :clarabel8, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.6,
                                  "max_iter" => 1500, "tol_gap_abs" => 1e-4,
                                  "tol_gap_rel" => 1e-4, "tol_ktratio" => 1e-3,
                                  "tol_feas" => 1e-4, "tol_infeas_abs" => 1e-4,
                                  "tol_infeas_rel" => 1e-4, "reduced_tol_gap_abs" => 1e-4,
                                  "reduced_tol_gap_rel" => 1e-4,
                                  "reduced_tol_ktratio" => 1e-3, "reduced_tol_feas" => 1e-4,
                                  "reduced_tol_infeas_abs" => 1e-4,
                                  "reduced_tol_infeas_rel" => 1e-4))]
    pr = prior(HighOrderPriorEstimator(), rd)
    clr = clusterise(ClustersEstimator(), pr)
    w0 = range(; start = inv(size(pr.X, 2)), stop = inv(size(pr.X, 2)),
               length = size(pr.X, 2))
    wp = StatsBase.pweights(range(; start = inv(size(pr.X, 1)), stop = inv(size(pr.X, 1)),
                                  length = size(pr.X, 1)))
    rf = 4.2 / 100 / 252
    algs = [BasicRelaxedRiskBudgeting(), RegularisedRelaxedRiskBudgeting(),
            RegularisedPenalisedRelaxedRiskBudgeting(),
            RegularisedPenalisedRelaxedRiskBudgeting(; p = 50)]
    @testset "Asset Risk Budgeting" begin
        df = CSV.read(joinpath(@__DIR__, "./assets/RelaxedAssetRiskBudgeting1.csv.gz"),
                      DataFrame)
        r = factory(StandardDeviation(), pr, slv)
        for (i, alg) in enumerate(algs)
            opt = JuMPOptimiser(; pr = pr, slv = slv)
            rb = RelaxedRiskBudgeting(; opt = opt, alg = alg)
            res = optimise(rb)
            @test isa(res.retcode, OptimisationSuccess)
            rkc = risk_contribution(r, res.w, pr.X)
            v1, m1 = findmin(rkc)
            v2, m2 = findmax(rkc)
            rtol = if i == 4
                0.5
            else
                5e-6
            end
            success = isapprox(v2 / v1, 1; rtol = rtol)
            if !success
                println("Extrema $i fails")
                find_tol(v2 / v1, 1)
            end
            @test success

            rtol = if Sys.isapple() && i ∈ (2, 5, 12)
                1e-4
            elseif Sys.isapple() && i == 17
                5e-3
            else
                1e-6
            end
            success = isapprox([res.w; rkc], df[!, "$i"]; rtol = rtol)
            if !success
                println("Weights and Contribution $i fails")
                find_tol([res.w; rkc], df[!, "$i"])
            end
            @test success
        end

        df = CSV.read(joinpath(@__DIR__, "./assets/RelaxedAssetRiskBudgeting2.csv.gz"),
                      DataFrame)
        for (i, alg) in enumerate(algs)
            opt = JuMPOptimiser(; pr = pr, slv = slv)
            rb = RelaxedRiskBudgeting(; opt = opt,
                                      rba = AssetRiskBudgeting(;
                                                               rkb = RiskBudgetResult(;
                                                                                      val = 20:-1:1)),
                                      alg = alg)
            res = optimise(rb)
            @test isa(res.retcode, OptimisationSuccess)
            rkc = risk_contribution(r, res.w, pr.X)
            v1, m1 = findmin(rkc)
            v2, m2 = findmax(rkc)
            rtol = if Sys.isapple() && i == 9
                5e-4
            elseif Sys.isapple() && i == 14
                1e-2
            else
                1e-6
            end
            success = isapprox([res.w; rkc], df[!, "$i"]; rtol = rtol)
            if !success
                println("Weights and Contribution $i fails")
                find_tol([res.w; rkc], df[!, "$i"])
            end
            @test success
        end

        res = optimise(RelaxedRiskBudgeting(; wi = w0,
                                            opt = JuMPOptimiser(; pr = pr,
                                                                slv = Solver(;
                                                                             solver = Clarabel.Optimizer,
                                                                             settings = ["verbose" => false,
                                                                                         "max_iter" => 1])),
                                            fb = InverseVolatility(; pe = pr)))
        @test isapprox(res.w, optimise(InverseVolatility(; pe = pr)).w)
    end
    @testset "Factor Risk Budgeting" begin
        r = factory(StandardDeviation(), pr, slv)
        df = CSV.read(joinpath(@__DIR__, "./assets/RelaxedFactorRiskBudgeting1.csv.gz"),
                      DataFrame)
        opt = JuMPOptimiser(; pr = pr, slv = slv,
                            sbgt = BudgetRange(; lb = 0, ub = nothing), bgt = 1,
                            wb = WeightBounds(; lb = nothing, ub = nothing))
        rr = regression(StepwiseRegression(), rd)
        for (i, alg) in enumerate(algs)
            rb = RelaxedRiskBudgeting(; opt = opt, rba = FactorRiskBudgeting(; re = rr))
            res = optimise(rb, rd)
            rkc = factor_risk_contribution(factory(r, pr, slv), res.w, pr.X;
                                           re = res.prb.rr)
            v1 = minimum(rkc[1:5])
            v2 = maximum(rkc[1:5])
            rtol = 0.25
            success = isapprox(v2 / v1, 1; rtol = rtol)
            if !success
                println("Extrema $i fails")
                find_tol(v2 / v1, 1)
            end
            @test success

            rtol = 5e-3
            success = isapprox([res.w; rkc], df[!, "$i"]; rtol = rtol)
            if !success
                println("Weights and Contribution $i fails")
                find_tol([res.w; rkc], df[!, "$i"])
            end
            @test success
        end
        df = CSV.read(joinpath(@__DIR__, "./assets/RelaxedFactorRiskBudgeting2.csv.gz"),
                      DataFrame)
        for (i, alg) in enumerate(algs)
            rb = RelaxedRiskBudgeting(; opt = opt,
                                      rba = FactorRiskBudgeting(; re = rr,
                                                                rkb = RiskBudgetResult(;
                                                                                       val = 1:5)))
            res = optimise(rb, rd)
            rkc = factor_risk_contribution(factory(r, pr, slv), res.w, pr.X;
                                           re = res.prb.rr)
            v1 = minimum(rkc[1:5])
            v2 = maximum(rkc[1:5])
            rtol = 1.1
            success = isapprox(v2 / v1, 5; rtol = rtol)
            if !success
                println("Extrema $i fails")
                find_tol(v2 / v1, 5)
            end
            @test success

            rtol = 5e-3
            success = isapprox([res.w; rkc], df[!, "$i"]; rtol = rtol)
            if !success
                println("Weights and Contribution $i fails")
                find_tol([res.w; rkc], df[!, "$i"])
            end
            @test success
        end
    end
end
