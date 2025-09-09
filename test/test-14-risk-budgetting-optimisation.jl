@safetestset "Risk Budgeting Optimisation" begin
    using Test, PortfolioOptimisers, DataFrames, CSV, TimeSeries, Clarabel, StatsBase
    function find_tol(a1, a2; name1 = :lhs, name2 = :rhs)
        for rtol in
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; rtol = rtol)
                println("isapprox($a1, $a2, rtol = $(rtol))")
                break
            end
        end
        for atol in
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; atol = atol)
                println("isapprox($a1, $a2, atol = $(atol))")
                break
            end
        end
    end
    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end],
                           TimeArray(CSV.File(joinpath(@__DIR__, "./assets/Factors.csv.gz"));
                                     timestamp = :Date)[(end - 252):end])
    slv = [Solver(; name = :clarabel6, solver = Clarabel.Optimizer,
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
    w0 = range(; start = inv(size(pr.X, 2)), stop = inv(size(pr.X, 2)),
               length = size(pr.X, 2))
    wp = pweights(range(; start = inv(size(pr.X, 1)), stop = inv(size(pr.X, 1)),
                        length = size(pr.X, 1)))
    rf = 4.2 / 100 / 252
    rs = [StandardDeviation(), Variance(), LowOrderMoment(),
          LowOrderMoment(;
                         alg = LowOrderDeviation(;
                                                 alg = SecondLowerMoment(;
                                                                         alg = SqrtRiskExpr()))),
          LowOrderMoment(; alg = LowOrderDeviation(; alg = SecondLowerMoment())),
          LowOrderMoment(;
                         alg = LowOrderDeviation(;
                                                 alg = SecondCentralMoment(;
                                                                           alg = SqrtRiskExpr()))),
          LowOrderMoment(; alg = LowOrderDeviation(; alg = SecondCentralMoment())),
          LowOrderMoment(; alg = MeanAbsoluteDeviation()), WorstRealisation(), Range(),
          ConditionalValueatRisk(), ConditionalValueatRiskRange(), EntropicValueatRisk(),
          EntropicValueatRiskRange(), RelativisticValueatRisk(),
          RelativisticValueatRiskRange(), MaximumDrawdown(), AverageDrawdown(),
          UlcerIndex(), ConditionalDrawdownatRisk(), EntropicDrawdownatRisk(),
          RelativisticDrawdownatRisk(), SquareRootKurtosis(; N = 2), SquareRootKurtosis(),
          OrderedWeightsArray(; alg = ExactOrderedWeightsArray()), OrderedWeightsArray(),
          OrderedWeightsArrayRange(), NegativeSkewness(),
          NegativeSkewness(; alg = QuadRiskExpr())]
    @testset "Asset Risk Budgeting" begin
        df = CSV.read(joinpath(@__DIR__, "./assets/RiskBudgeting1.csv.gz"), DataFrame)
        opt = JuMPOptimiser(; pe = pr, slv = slv)
        for (i, r) in enumerate(rs)
            r = factory(r, pr, slv)
            rb = RiskBudgeting(; r = r, opt = opt)
            res = optimise!(rb, rd)
            @test isa(res.retcode, OptimisationSuccess)
            rkc = risk_contribution(r, res.w, pr.X)
            v1 = minimum(rkc)
            v2 = maximum(rkc)
            rtol = if i ∈ (3, 8, 12, 27)
                5e-2
            elseif i ∈ (9, 15)
                2.5e-1
            elseif i ∈ (10, 18, 22)
                5e-1
            elseif i ∈ (11, 23)
                5e-3
            elseif i == 14
                5e-1
            elseif i ∈ (13, 17, 20, 21)
                1
            elseif i == 16
                2.5e-1
            elseif i == 26
                1e-2
            elseif i ∈ (28, 29)
                1e-3
            else
                5e-4
            end
            success = isapprox(v2 / v1, 1; rtol = rtol)
            if !success
                println("Extrema $i fails")
                find_tol(v2 / v1, 1)
            end
            @test success

            rtol = if i ∈ (7, 10, 19, 25) || Sys.isapple() && i ∈ (2, 5, 12)
                1e-4
            elseif i == 17
                5e-3
            elseif i ∈ (9, 11, 18, 24)
                5e-4
            elseif i == 20
                1e-3
            elseif i ∈ (13, 21, 14, 15, 16, 22)
                1e-2
            else
                5e-5
            end
            success = isapprox([res.w; rkc], df[!, "$i"]; rtol = rtol)
            if !success
                println("Weights and Contribution $i fails")
                find_tol([res.w; rkc], df[!, "$i"])
            end
            @test success
        end

        df = CSV.read(joinpath(@__DIR__, "./assets/RiskBudgeting2.csv.gz"), DataFrame)
        for (i, r) in enumerate(rs)
            r = factory(r, pr, slv)
            rb = RiskBudgeting(; r = r, opt = opt,
                               alg = AssetRiskBudgeting(;
                                                        rkb = RiskBudgetResult(;
                                                                               val = 1:20)))
            res = optimise!(rb, rd)
            @test isa(res.retcode, OptimisationSuccess)
            rkc = risk_contribution(r, res.w, pr.X)
            v1, m1 = findmin(rkc)
            v2, m2 = findmax(rkc)
            @test m1 == 1
            success = m2 == 20
            if !success
                success = m2 == 19 || m2 == 18
            end
            @test success
            rtol = if i ∈ (3, 24, 28)
                1e-3
            elseif i ∈ (8, 11, 13, 23, 27, 29)
                5e-3
            elseif i == 9
                5e-1
            elseif i ∈ (10, 19, 20, 21, 22)
                2.5e-1
            elseif i ∈ (14, 15, 17)
                1e-1
            elseif i ∈ (16, 18)
                5e-2
            else
                5e-4
            end
            success = isapprox(v2 / v1, 20; rtol = rtol)
            if !success
                println("Extrema $i fails")
                find_tol(v2 / v1, 20)
            end
            @test success

            rtol = if i == 11 || Sys.isapple() && i == 9
                5e-4
            elseif i == 17
                1e-3
            elseif i == 21 || Sys.isapple() && i == 14
                1e-2
            elseif i ∈ (14, 15, 16, 20, 22)
                5e-3
            elseif i == 18
                5e-4
            else
                1e-4
            end
            success = isapprox([res.w; rkc], df[!, "$i"]; rtol = rtol)
            if !success
                println("Weights and Contribution $i fails")
                find_tol([res.w; rkc], df[!, "$i"])
            end
            @test success
        end
    end
    @testset "Factor Risk Budgeting" begin
        df = CSV.read(joinpath(@__DIR__, "./assets/FactorRiskBudgeting1.csv.gz"), DataFrame)
        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            sbgt = BudgetRange(; lb = 0, ub = nothing), bgt = 1,
                            wb = WeightBounds(; lb = nothing, ub = nothing))
        rr = regression(StepwiseRegression(), rd)
        for (i, r) in enumerate(rs)
            if i == 25
                continue
            end
            rb = RiskBudgeting(; r = r, opt = opt, alg = FactorRiskBudgeting(; re = rr))
            res = optimise!(rb, rd)
            @test isa(res.retcode, OptimisationSuccess)
            rkc = factor_risk_contribution(factory(r, pr, slv), res.w, pr.X;
                                           re = res.prb.rr)
            v1 = minimum(rkc[1:5])
            v2 = maximum(rkc[1:5])
            rtol = if i ∈ (1, 2, 10, 17)
                5e-1
            elseif i ∈ (4, 5, 7, 13, 24, 28)
                5e-4
            elseif i == 6
                1e-4
            elseif i == 9
                1
            elseif i ∈ (11, 15, 18, 19, 20, 22)
                1e-1
            elseif i == 14
                2.5e-1
            elseif i == 26
                5e-3
            elseif i == 29
                5e-3
            else
                5e-2
            end
            success = isapprox(v2 / v1, 1; rtol = rtol)
            if !success
                println("Extrema $i fails")
                find_tol(v2 / v1, 1)
            end
            @test success

            rtol = if i ∈ (9, 15, 16, 17, 22, 27)
                5e-3
            elseif i == 14
                1e-2
            elseif Sys.iswindows() && i == 21
                5e-3
            elseif i ∈ (18, 19, 21)
                5e-4
            elseif i ∈ (10, 11)
                5e-4
            else
                1e-4
            end
            success = isapprox([res.w; rkc], df[!, "$i"]; rtol = rtol)
            if !success
                println("Weights and Contribution $i fails")
                find_tol([res.w; rkc], df[!, "$i"])
            end
            @test success
        end
        df = CSV.read(joinpath(@__DIR__, "./assets/FactorRiskBudgeting2.csv.gz"), DataFrame)
        rr = regression(StepwiseRegression(; alg = Backward()), rd)
        for (i, r) in enumerate(rs)
            if i == 25
                continue
            end
            opt = JuMPOptimiser(; pe = pr, slv = slv)
            rb = RiskBudgeting(; r = r, opt = opt,
                               alg = FactorRiskBudgeting(; flag = true, re = rr,
                                                         rkb = RiskBudgetResult(;
                                                                                val = 1:5)))
            res = optimise!(rb, rd)
            @test isa(res.retcode, OptimisationSuccess)
            rkc = factor_risk_contribution(factory(r, pr, slv), res.w, pr.X;
                                           re = res.prb.rr)
            v1, m1 = findmin(rkc[1:5])
            v2, m2 = findmax(rkc[1:5])
            @test m1 == 1
            success = m2 == 5
            if !success
                success = m2 == 4
            end
            @test success
            rtol = if i ∈ (1, 2, 6, 7)
                1e-2
            elseif i == 9
                1
            elseif i ∈ (10, 11, 14, 16)
                2.5e-1
            elseif i ∈ (13, 15, 17, 18, 19, 20, 21, 22, 28, 29)
                5e-1
            elseif i == 26
                1e-3
            else
                5e-2
            end
            success = isapprox(v2 / v1, 5; rtol = rtol)
            if !success
                println("Extrema $i fails")
                find_tol(v2 / v1, 5)
            end
            @test success

            rtol = if i == 22 || Sys.isapple() && i ∈ (18, 20) || Sys.iswindows() && i == 10
                1e-3
            elseif i ∈ (1, 10) || Sys.isapple() && i == 2
                5e-4
            elseif i ∈ (13, 15, 16, 17, 19)
                5e-3
            elseif i == 14
                1e-2
            elseif i ∈ (18, 20, 24, 27)
                5e-4
            elseif i == 21
                5e-2
            else
                1e-4
            end
            success = isapprox([res.w; rkc], df[!, "$i"]; rtol = rtol)
            if !success
                println("Weights and Contribution $i fails")
                find_tol([res.w; rkc], df[!, "$i"])
            end
            @test success
        end
    end
end
