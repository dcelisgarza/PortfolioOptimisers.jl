@safetestset "Clustering Optimistaion" begin
    using PortfolioOptimisers, CSV, DataFrames, Test, StableRNGs, Random, Clarabel,
          StatsBase
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
        X = randn(rng, 1000, 15)
        rd = ReturnsData(; nx = 1:size(X, 2), X = X)

        pm = prior(HighOrderPriorEstimator(), rd)
        clm = clusterise(ClusteringEstimator(), pm.X)
        opt = HierarchicalOptimiser(; pe = pm, cle = clm,
                                    slv = Solver(; name = :clarabel,
                                                 solver = Clarabel.Optimizer,
                                                 check_sol = (; allow_local = true,
                                                              allow_almost = true),
                                                 settings = Dict("max_step_fraction" => 0.75,
                                                                 "verbose" => false)))
        df = CSV.read(joinpath(@__DIR__, "./assets/HRP.csv"), DataFrame)
        ew = eweights(1:size(X, 1), inv(size(X, 1)); scale = true)
        risks = [ValueatRiskRange(), ValueatRiskRange(; alpha = 0.3, beta = 0.25),
                 ValueatRisk(), ValueatRisk(; alpha = 0.3), DrawdownatRisk(),
                 DrawdownatRisk(; alpha = 0.3), RelativeDrawdownatRisk(),
                 RelativeDrawdownatRisk(; alpha = 0.3), RelativeAverageDrawdown(),
                 RelativeAverageDrawdown(; w = ew), RelativeEntropicDrawdownatRisk(),
                 RelativeEntropicDrawdownatRisk(; alpha = 0.3), RelativeMaximumDrawdown(),
                 RelativeRelativisticDrawdownatRisk(),
                 RelativeRelativisticDrawdownatRisk(; alpha = 0.3), RelativeUlcerIndex(),
                 EqualRiskMeasure(), FirstLowerPartialMoment(),
                 FirstLowerPartialMoment(; target = 2e-4),
                 FirstLowerPartialMoment(; mu = pm.mu / 5),
                 FirstLowerPartialMoment(; w = ew), NegativeQuadraticSemiSkewness(),
                 NegativeSemiSkewness(), SemiStandardDeviation(), SemiVariance(),
                 BrownianDistanceVariance(), ConditionalValueatRiskRange(),
                 ConditionalValueatRiskRange(; alpha = 0.3, beta = 0.25),
                 EntropicValueatRiskRange(),
                 EntropicValueatRiskRange(; alpha = 0.18, beta = 0.31),
                 GiniMeanDifference(), MeanAbsoluteDeviation(),
                 MeanAbsoluteDeviation(; target = 2e-4),
                 MeanAbsoluteDeviation(; mu = pm.mu / 5), MeanAbsoluteDeviation(; w = ew),
                 MeanAbsoluteDeviation(; we = ew), NegativeQuadraticSkewness(),
                 NegativeSkewness(), Range(), RelativisticValueatRiskRange(),
                 RelativisticValueatRiskRange(; alpha = 0.14, kappa_a = 0.4, beta = 0.27,
                                              kappa_b = 0.1), SquareRootKurtosis(),
                 SquareRootSemiKurtosis(), StandardDeviation(), TailGiniRange(),
                 TailGiniRange(; alpha_i = 1e-6, alpha = 0.16, a_sim = 125, beta_i = 1e-6,
                               beta = 0.2, b_sim = 200), UncertaintySetVariance(),
                 Variance()]
        for (i, risk) ∈ enumerate(risks)
            w1 = optimise!(HierarchicalRiskParity(; r = risk, opt = opt))
            res = if i == 14
                isapprox(w1, df[:, i]; rtol = 1e-5)
            elseif i == 29
                if Sys.isapple()
                    isapprox(w1, df[:, i]; rtol = 1e-7)
                else
                    isapprox(w1, df[:, i]; rtol = 5e-8)
                end
            elseif i == 30
                isapprox(w1, df[:, i]; rtol = 5e-7)
            elseif i == 40
                isapprox(w1, df[:, i]; rtol = 5e-8)
            elseif i == 41
                if Sys.isapple()
                    isapprox(w1, df[:, i]; rtol = 5e-3)
                else
                    isapprox(w1, df[:, i]; rtol = 5e-4)
                end
            else
                isapprox(w1, df[:, i])
            end
            if !res
                println("$i\n$(string(risk)) failed")
                find_tol(w1, df[:, i]; name1 = :w1, name2 = :df)
            end
            @test res
        end

        opts = [HierarchicalOptimiser(; pe = pm, cle = clm, sce = SumScalariser(),
                                      slv = Solver(; name = :clarabel,
                                                   solver = Clarabel.Optimizer,
                                                   check_sol = (; allow_local = true,
                                                                allow_almost = true),
                                                   settings = Dict("max_step_fraction" => 0.75,
                                                                   "verbose" => false))),
                HierarchicalOptimiser(; pe = pm, cle = clm, sce = MaxScalariser(),
                                      slv = Solver(; name = :clarabel,
                                                   solver = Clarabel.Optimizer,
                                                   check_sol = (; allow_local = true,
                                                                allow_almost = true),
                                                   settings = Dict("max_step_fraction" => 0.75,
                                                                   "verbose" => false))),
                HierarchicalOptimiser(; pe = pm, cle = clm,
                                      sce = LogSumExpScalariser(; gamma = 1e-3),
                                      slv = Solver(; name = :clarabel,
                                                   solver = Clarabel.Optimizer,
                                                   check_sol = (; allow_local = true,
                                                                allow_almost = true),
                                                   settings = Dict("max_step_fraction" => 0.75,
                                                                   "verbose" => false))),
                HierarchicalOptimiser(; pe = pm, cle = clm,
                                      sce = LogSumExpScalariser(; gamma = 3),
                                      slv = Solver(; name = :clarabel,
                                                   solver = Clarabel.Optimizer,
                                                   check_sol = (; allow_local = true,
                                                                allow_almost = true),
                                                   settings = Dict("max_step_fraction" => 0.75,
                                                                   "verbose" => false)))]
        df = CSV.read(joinpath(@__DIR__, "./assets/HRP_vector_rm.csv"), DataFrame)
        risk = [MeanAbsoluteDeviation(; settings = RiskMeasureSettings(; scale = 2.3)),
                StandardDeviation(; settings = RiskMeasureSettings(; scale = 0.3))]
        for (i, opt) ∈ enumerate(opts)
            w1 = optimise!(HierarchicalRiskParity(; r = risk, opt = opt))
            res = isapprox(w1, df[:, i])
            if !res
                println("$i\n$(string(risk)) failed")
                find_tol(w1, df[:, i]; name1 = :w1, name2 = :df)
            end
            @test res
        end
    end
    @testset "Hierarchical Equal Risk Contribution" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 15)
        rd = ReturnsData(; nx = 1:size(X, 2), X = X)

        pm = prior(HighOrderPriorEstimator(), rd)
        clm = clusterise(ClusteringEstimator(), pm.X)
        opt = HierarchicalOptimiser(; pe = pm, cle = clm,
                                    slv = Solver(; name = :clarabel,
                                                 solver = Clarabel.Optimizer,
                                                 check_sol = (; allow_local = true,
                                                              allow_almost = true),
                                                 settings = Dict("max_step_fraction" => 0.75,
                                                                 "verbose" => false)))
        ew = eweights(1:size(X, 1), inv(size(X, 1)); scale = true)

        df = CSV.read(joinpath(@__DIR__, "./assets/HERC_same_measures.csv"), DataFrame)
        risks = [ValueatRiskRange(), ValueatRiskRange(; alpha = 0.3, beta = 0.25),
                 ValueatRisk(), ValueatRisk(; alpha = 0.3), DrawdownatRisk(),
                 DrawdownatRisk(; alpha = 0.3), RelativeDrawdownatRisk(),
                 RelativeDrawdownatRisk(; alpha = 0.3), RelativeAverageDrawdown(),
                 RelativeAverageDrawdown(; w = ew), RelativeEntropicDrawdownatRisk(),
                 RelativeEntropicDrawdownatRisk(; alpha = 0.3), RelativeMaximumDrawdown(),
                 RelativeRelativisticDrawdownatRisk(),
                 RelativeRelativisticDrawdownatRisk(; alpha = 0.3), RelativeUlcerIndex(),
                 EqualRiskMeasure(), FirstLowerPartialMoment(),
                 FirstLowerPartialMoment(; target = 2e-4),
                 FirstLowerPartialMoment(; mu = pm.mu / 5),
                 FirstLowerPartialMoment(; w = ew), NegativeQuadraticSemiSkewness(),
                 NegativeSemiSkewness(), SemiStandardDeviation(), SemiVariance(),
                 BrownianDistanceVariance(), ConditionalValueatRiskRange(),
                 ConditionalValueatRiskRange(; alpha = 0.3, beta = 0.25),
                 EntropicValueatRiskRange(),
                 EntropicValueatRiskRange(; alpha = 0.18, beta = 0.31),
                 GiniMeanDifference(), MeanAbsoluteDeviation(),
                 MeanAbsoluteDeviation(; target = 2e-4),
                 MeanAbsoluteDeviation(; mu = pm.mu / 5), MeanAbsoluteDeviation(; w = ew),
                 MeanAbsoluteDeviation(; we = ew), NegativeQuadraticSkewness(),
                 NegativeSkewness(), Range(), RelativisticValueatRiskRange(),
                 RelativisticValueatRiskRange(; alpha = 0.14, kappa_a = 0.4, beta = 0.27,
                                              kappa_b = 0.1), SquareRootKurtosis(),
                 SquareRootSemiKurtosis(), StandardDeviation(), TailGiniRange(),
                 TailGiniRange(; alpha_i = 1e-6, alpha = 0.16, a_sim = 125, beta_i = 1e-6,
                               beta = 0.2, b_sim = 200), UncertaintySetVariance(),
                 Variance()]
        for (i, risk) ∈ enumerate(risks)
            w1 = optimise!(HierarchicalEqualRiskContribution(; ri = risk, opt = opt))
            res = if i == 14
                isapprox(w1, df[:, i]; rtol = 1e-5)
            elseif i == 29
                if Sys.isapple()
                    isapprox(w1, df[:, i]; rtol = 1e-7)
                else
                    isapprox(w1, df[:, i]; rtol = 5e-8)
                end
            elseif i == 30
                isapprox(w1, df[:, i]; rtol = 5e-7)
            elseif i == 40
                isapprox(w1, df[:, i]; rtol = 1e-7)
            elseif i == 41
                if Sys.isapple()
                    isapprox(w1, df[:, i]; rtol = 5e-3)
                else
                    isapprox(w1, df[:, i]; rtol = 1e-7)
                end
            else
                isapprox(w1, df[:, i])
            end
            if !res
                println("$i\n$(string(risk)) failed")
                find_tol(w1, df[:, i]; name1 = :w1, name2 = :df)
            end
            @test res
        end

        df = CSV.read(joinpath(@__DIR__, "./assets/HERC_different_risk_measures.csv"),
                      DataFrame)
        for (i, (risk1, risk2)) ∈ enumerate(zip(risks, circshift(risks, 3)))
            w1 = optimise!(HierarchicalEqualRiskContribution(; ri = risk1, ro = risk2,
                                                             opt = opt))
            res = if i == 14
                isapprox(w1, df[:, i]; rtol = 1e-5)
            elseif i == 29
                if Sys.isapple()
                    isapprox(w1, df[:, i]; rtol = 1e-7)
                else
                    isapprox(w1, df[:, i]; rtol = 5e-8)
                end
            elseif i == 30
                isapprox(w1, df[:, i]; rtol = 1e-7)
            elseif i == 32 && Sys.isapple()
                isapprox(w1, df[:, i]; rtol = 1e-7)
            elseif i == 33
                isapprox(w1, df[:, i]; rtol = 1e-7)
            elseif i == 40
                isapprox(w1, df[:, i]; rtol = 5e-8)
            elseif i == 41
                if Sys.isapple()
                    isapprox(w1, df[:, i]; rtol = 5e-3)
                else
                    isapprox(w1, df[:, i]; rtol = 5e-8)
                end
            elseif i == 44
                if Sys.isapple()
                    isapprox(w1, df[:, i]; rtol = 5e-5)
                else
                    isapprox(w1, df[:, i]; rtol = 5e-8)
                end
            else
                isapprox(w1, df[:, i])
            end
            if !res
                println("$i\n$(string(risk1))\n$(string(risk2)) failed")
                find_tol(w1, df[:, i]; name1 = :w1, name2 = :df)
            end
            @test res
        end

        sces = [SumScalariser(), MaxScalariser(), LogSumExpScalariser(; gamma = 1e-3),
                LogSumExpScalariser(; gamma = 3)]
        base = [MeanAbsoluteDeviation(; settings = RiskMeasureSettings(; scale = 2.3)),
                StandardDeviation(; settings = RiskMeasureSettings(; scale = 4.2))]
        risks = [(r1 = base, r2 = base)
                 (r1 = base,
                  r2 = [ConditionalValueatRisk(;
                                               settings = RiskMeasureSettings(;
                                                                              scale = 3.2)),
                        WorstRealisation(; settings = RiskMeasureSettings(; scale = 0.2))])
                 (r1 = base,
                  r2 = ConditionalValueatRisk(;
                                              settings = RiskMeasureSettings(; scale = 3.2)))
                 (r1 = ConditionalValueatRisk(;
                                              settings = RiskMeasureSettings(; scale = 3.2)),
                  r2 = base)]
        df = CSV.read(joinpath(@__DIR__, "./assets/HERC_vector_rm.csv"), DataFrame)
        for (idx, sce) ∈ zip(1:4:16, sces)
            opt = HierarchicalOptimiser(; pe = pm, cle = clm, sce = sce,
                                        slv = Solver(; name = :clarabel,
                                                     solver = Clarabel.Optimizer,
                                                     check_sol = (; allow_local = true,
                                                                  allow_almost = true),
                                                     settings = Dict("max_step_fraction" => 0.75,
                                                                     "verbose" => false)))
            for (i, rs) ∈ enumerate(risks)
                w = PortfolioOptimisers.optimise!(HierarchicalEqualRiskContribution(;
                                                                                    ri = rs.r1,
                                                                                    ro = rs.r2,
                                                                                    opt = opt))
                res = isapprox(w, df[!, idx + i - 1])
                if !res
                    println("Failed on vector rm HERC\n$sce\n$i")
                    find_tol(w, df[!, idx + i - 1]; name1 = :w, name2 = :df)
                end
                @test res
            end
        end
    end
end
