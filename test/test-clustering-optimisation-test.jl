@safetestset "Clustering Optimisation" begin
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
        rng = StableRNG(987456321)
        X = randn(rng, 500, 10)
        rd = ReturnsResult(; nx = 1:size(X, 2), X = X)
        pr = prior(HighOrderPriorEstimator(), rd)
        clr = clusterise(ClusteringEstimator(), pr.X)
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     check_sol = (; allow_local = true, allow_almost = true),
                     settings = Dict("max_step_fraction" => 0.75, "verbose" => false))
        opt = HierarchicalOptimiser(; pe = pr, cle = clr, slv = slv)
        ew = eweights(1:size(X, 1), inv(size(X, 1)); scale = true)
        w1 = fill(inv(10), 10)
        rf = 4.34 / 100 / 252
        sigma = cov(GerberCovariance(), X)
        mu = vec(mean(ShrunkExpectedReturns(; ce = GerberCovariance()), X))
        sk, V = coskewness(Coskewness(; alg = Semi()), X; mean = transpose(mu))
        df = CSV.read(joinpath(@__DIR__, "./assets/HRP.csv"), DataFrame)
        risks = [Variance(; sigma = sigma),#
                 Variance(),# 2
                 ###
                 UncertaintySetVariance(; sigma = sigma),#
                 UncertaintySetVariance(),# 4
                 ###
                 StandardDeviation(; sigma = sigma),#
                 StandardDeviation(),# 6
                 ###
                 BrownianDistanceVariance(),# 7
                 ###
                 LowOrderMoment(; mu = mu),#
                 LowOrderMoment(; mu = rf),#
                 LowOrderMoment(; w = ew),#
                 LowOrderMoment(),# 11
                 ###
                 LowOrderMoment(;
                                alg = LowOrderDeviation(;
                                                        alg = SecondLowerMoment(;
                                                                                formulation = SqrtRiskExpr())),
                                mu = mu),#
                 LowOrderMoment(;
                                alg = LowOrderDeviation(;
                                                        alg = SecondLowerMoment(;
                                                                                formulation = SqrtRiskExpr())),
                                mu = rf),#
                 LowOrderMoment(;
                                alg = LowOrderDeviation(;
                                                        alg = SecondLowerMoment(;
                                                                                formulation = SqrtRiskExpr())),
                                w = ew),#
                 LowOrderMoment(;
                                alg = LowOrderDeviation(;
                                                        alg = SecondLowerMoment(;
                                                                                formulation = SqrtRiskExpr()))),# 15
                 ###
                 LowOrderMoment(; alg = LowOrderDeviation(; alg = SecondLowerMoment()),
                                mu = mu),#
                 LowOrderMoment(; alg = LowOrderDeviation(; alg = SecondLowerMoment()),
                                mu = rf),#
                 LowOrderMoment(; alg = LowOrderDeviation(; alg = SecondLowerMoment()),
                                w = ew),#
                 LowOrderMoment(; alg = LowOrderDeviation(; alg = SecondLowerMoment())),# 19
                 ###
                 LowOrderMoment(; alg = MeanAbsoluteDeviation(), mu = mu),#
                 LowOrderMoment(; alg = MeanAbsoluteDeviation(), mu = rf),#
                 LowOrderMoment(; alg = MeanAbsoluteDeviation(), w = ew),#
                 LowOrderMoment(; alg = MeanAbsoluteDeviation()),# 23
                 ###
                 HighOrderMoment(; mu = mu),#
                 HighOrderMoment(; mu = rf),#
                 HighOrderMoment(; w = ew),#
                 HighOrderMoment(),# 27
                 ###
                 HighOrderMoment(; alg = FourthLowerMoment(), mu = mu),#
                 HighOrderMoment(; alg = FourthLowerMoment(), mu = rf),#
                 HighOrderMoment(; alg = FourthLowerMoment(), w = ew),#
                 HighOrderMoment(; alg = FourthLowerMoment()),# 31
                 ###
                 HighOrderMoment(; alg = FourthCentralMoment(), mu = mu),#
                 HighOrderMoment(; alg = FourthCentralMoment(), mu = rf),#
                 HighOrderMoment(; alg = FourthCentralMoment(), w = ew),#
                 HighOrderMoment(; alg = FourthCentralMoment()),# 35
                 ###
                 HighOrderMoment(; alg = HighOrderDeviation(), mu = mu),#
                 HighOrderMoment(; alg = HighOrderDeviation(), mu = rf),#
                 HighOrderMoment(; alg = HighOrderDeviation(), w = ew),#
                 HighOrderMoment(; alg = HighOrderDeviation()),# 39
                 ###
                 HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthLowerMoment()),
                                 mu = mu),#
                 HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthLowerMoment()),
                                 mu = rf),#
                 HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthLowerMoment()),
                                 w = ew),#
                 HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthLowerMoment())),# 43
                 ###
                 HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthCentralMoment()),
                                 mu = mu),#
                 HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthCentralMoment()),
                                 mu = rf),#
                 HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthCentralMoment()),
                                 w = ew),#
                 HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthCentralMoment())),# 47
                 ###
                 SquareRootKurtosis(),#
                 SquareRootKurtosis(; mu = mu),#
                 SquareRootKurtosis(; alg = Semi()),#
                 SquareRootKurtosis(; alg = Semi(), mu = mu),# 51
                 ###
                 NegativeSkewness(),#
                 NegativeSkewness(; alg = QuadRiskExpr()),#
                 NegativeSkewness(; sk = sk, V = V),#
                 NegativeSkewness(; alg = QuadRiskExpr(), sk = sk, V = V),# 55
                 ###
                 ValueatRisk(),#
                 ValueatRiskRange(),#
                 DrawdownatRisk(),#
                 RelativeDrawdownatRisk(),# 59
                 ###
                 ConditionalValueatRisk(),#
                 DistributionallyRobustConditionalValueatRisk(),#
                 ConditionalValueatRiskRange(),#
                 ConditionalDrawdownatRisk(),#
                 RelativeConditionalDrawdownatRisk(),# 64
                 ###
                 EntropicValueatRisk(),#
                 EntropicValueatRiskRange(),#
                 EntropicDrawdownatRisk(),#
                 RelativeEntropicDrawdownatRisk(),# 68
                 ###
                 RelativisticValueatRisk(),#
                 RelativisticValueatRiskRange(),#
                 RelativisticDrawdownatRisk(),#
                 RelativeRelativisticDrawdownatRisk(),# 72
                 ###
                 OrderedWeightsArray(),#
                 OrderedWeightsArray(; w = owa_gmd(500)),#
                 OrderedWeightsArray(; w = owa_tg(500)),#
                 OrderedWeightsArray(; w = owa_tgrg(500)),# 76
                 ###
                 AverageDrawdown(),#
                 AverageDrawdown(; w = ew),#
                 RelativeAverageDrawdown(),#
                 RelativeAverageDrawdown(; w = ew),# 80
                 ###
                 UlcerIndex(),#
                 RelativeUlcerIndex(),# 82
                 ###
                 MaximumDrawdown(),#
                 RelativeMaximumDrawdown(),# 84
                 ###
                 WorstRealisation(),# 85
                 ###
                 Range(),# 86
                 ###
                 EqualRiskMeasure(),# 87
                 ###
                 TurnoverRiskMeasure(; w = w1),#
                 TrackingRiskMeasure(; tracking = WeightsTracking(; w = w1)),#
                 TrackingRiskMeasure(; tracking = ReturnsTracking(; w = pr.X * w1))]
        names = string.(risks)
        idx = [(findfirst(x -> x == '{', s) - 1) for s ∈ names]
        names = [n[1:i] for (n, i) ∈ zip(names, idx)]
        for (i, (risk, name)) ∈ enumerate(zip(risks, names))
            name = name * "_$(i)"
            w = optimise!(HierarchicalRiskParity(; r = risk, opt = opt)).w
            res = if i ∈ 65:72
                isapprox(w, df[!, name]; rtol = 5e-8)
            else
                isapprox(w, df[!, name])
            end
            if !res
                println("$name failed")
                find_tol(w, df[!, name]; name1 = :w1, name2 = :df)
            end
            @test res
        end

        rng = StableRNG(123456789)
        X = randn(rng, 1000, 15)
        rd = ReturnsResult(; nx = 1:size(X, 2), X = X)

        pr = prior(EmpiricalPriorEstimator(), rd)
        clr = clusterise(ClusteringEstimator(), pr.X)
        opts = [HierarchicalOptimiser(; pe = pr, cle = clr, sce = SumScalariser(),
                                      slv = slv),
                HierarchicalOptimiser(; pe = pr, cle = clr, sce = MaxScalariser(),
                                      slv = slv),
                HierarchicalOptimiser(; pe = pr, cle = clr,
                                      sce = LogSumExpScalariser(; gamma = 1e-3), slv = slv),
                HierarchicalOptimiser(; pe = pr, cle = clr,
                                      sce = LogSumExpScalariser(; gamma = 3), slv = slv)]
        df = CSV.read(joinpath(@__DIR__, "./assets/HRP_vector_rm.csv"), DataFrame)
        risk = [LowOrderMoment(; alg = MeanAbsoluteDeviation(),
                               settings = RiskMeasureSettings(; scale = 2.3)),
                StandardDeviation(; settings = RiskMeasureSettings(; scale = 0.3))]

        for (i, opt) ∈ enumerate(opts)
            w1 = optimise!(HierarchicalRiskParity(; r = risk, opt = opt)).w
            res = isapprox(w1, df[:, i])
            if !res
                println("$i\n$(string(risk)) failed")
                find_tol(w1, df[:, i]; name1 = :w1, name2 = :df)
            end
            @test res
        end
    end
    @testset "Hierarchical Equal Risk Contribution" begin
        rng = StableRNG(987456321)
        X = randn(rng, 500, 10)
        rd = ReturnsResult(; nx = 1:size(X, 2), X = X)
        pr = prior(HighOrderPriorEstimator(), rd)
        clr = clusterise(ClusteringEstimator(), pr.X)
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     check_sol = (; allow_local = true, allow_almost = true),
                     settings = Dict("max_step_fraction" => 0.75, "verbose" => false))
        opt = HierarchicalOptimiser(; pe = pr, cle = clr,
                                    cwf = JuMP_ClusteringWeightFiniliser(; slv = slv),
                                    slv = slv)
        ew = eweights(1:size(X, 1), inv(size(X, 1)); scale = true)
        w1 = fill(inv(10), 10)
        rf = 4.34 / 100 / 252
        sigma = cov(GerberCovariance(), X)
        mu = vec(mean(ShrunkExpectedReturns(; ce = GerberCovariance()), X))
        sk, V = coskewness(Coskewness(; alg = Semi()), X; mean = transpose(mu))
        df = CSV.read(joinpath(@__DIR__, "./assets/HERC_same_measures.csv"), DataFrame)
        risks = [Variance(; sigma = sigma),#
                 Variance(),# 2
                 ###
                 UncertaintySetVariance(; sigma = sigma),#
                 UncertaintySetVariance(),# 4
                 ###
                 StandardDeviation(; sigma = sigma),#
                 StandardDeviation(),# 6
                 ###
                 BrownianDistanceVariance(),# 7
                 ###
                 LowOrderMoment(; mu = mu),#
                 LowOrderMoment(; mu = rf),#
                 LowOrderMoment(; w = ew),#
                 LowOrderMoment(),# 11
                 ###
                 LowOrderMoment(;
                                alg = LowOrderDeviation(;
                                                        alg = SecondLowerMoment(;
                                                                                formulation = SqrtRiskExpr())),
                                mu = mu),#
                 LowOrderMoment(;
                                alg = LowOrderDeviation(;
                                                        alg = SecondLowerMoment(;
                                                                                formulation = SqrtRiskExpr())),
                                mu = rf),#
                 LowOrderMoment(;
                                alg = LowOrderDeviation(;
                                                        alg = SecondLowerMoment(;
                                                                                formulation = SqrtRiskExpr())),
                                w = ew),#
                 LowOrderMoment(;
                                alg = LowOrderDeviation(;
                                                        alg = SecondLowerMoment(;
                                                                                formulation = SqrtRiskExpr()))),# 15
                 ###
                 LowOrderMoment(; alg = LowOrderDeviation(; alg = SecondLowerMoment()),
                                mu = mu),#
                 LowOrderMoment(; alg = LowOrderDeviation(; alg = SecondLowerMoment()),
                                mu = rf),#
                 LowOrderMoment(; alg = LowOrderDeviation(; alg = SecondLowerMoment()),
                                w = ew),#
                 LowOrderMoment(; alg = LowOrderDeviation(; alg = SecondLowerMoment())),# 19
                 ###
                 LowOrderMoment(; alg = MeanAbsoluteDeviation(), mu = mu),#
                 LowOrderMoment(; alg = MeanAbsoluteDeviation(), mu = rf),#
                 LowOrderMoment(; alg = MeanAbsoluteDeviation(), w = ew),#
                 LowOrderMoment(; alg = MeanAbsoluteDeviation()),# 23
                 ###
                 HighOrderMoment(; mu = mu),#
                 HighOrderMoment(; mu = rf),#
                 HighOrderMoment(; w = ew),#
                 HighOrderMoment(),# 27
                 ###
                 HighOrderMoment(; alg = FourthLowerMoment(), mu = mu),#
                 HighOrderMoment(; alg = FourthLowerMoment(), mu = rf),#
                 HighOrderMoment(; alg = FourthLowerMoment(), w = ew),#
                 HighOrderMoment(; alg = FourthLowerMoment()),# 31
                 ###
                 HighOrderMoment(; alg = FourthCentralMoment(), mu = mu),#
                 HighOrderMoment(; alg = FourthCentralMoment(), mu = rf),#
                 HighOrderMoment(; alg = FourthCentralMoment(), w = ew),#
                 HighOrderMoment(; alg = FourthCentralMoment()),# 35
                 ###
                 HighOrderMoment(; alg = HighOrderDeviation(), mu = mu),#
                 HighOrderMoment(; alg = HighOrderDeviation(), mu = rf),#
                 HighOrderMoment(; alg = HighOrderDeviation(), w = ew),#
                 HighOrderMoment(; alg = HighOrderDeviation()),# 39
                 ###
                 HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthLowerMoment()),
                                 mu = mu),#
                 HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthLowerMoment()),
                                 mu = rf),#
                 HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthLowerMoment()),
                                 w = ew),#
                 HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthLowerMoment())),# 43
                 ###
                 HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthCentralMoment()),
                                 mu = mu),#
                 HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthCentralMoment()),
                                 mu = rf),#
                 HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthCentralMoment()),
                                 w = ew),#
                 HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthCentralMoment())),# 47
                 ###
                 SquareRootKurtosis(),#
                 SquareRootKurtosis(; mu = mu),#
                 SquareRootKurtosis(; alg = Semi()),#
                 SquareRootKurtosis(; alg = Semi(), mu = mu),# 51
                 ###
                 NegativeSkewness(),#
                 NegativeSkewness(; alg = QuadRiskExpr()),#
                 NegativeSkewness(; sk = sk, V = V),#
                 NegativeSkewness(; alg = QuadRiskExpr(), sk = sk, V = V),# 55
                 ###
                 ValueatRisk(),#
                 ValueatRiskRange(),#
                 DrawdownatRisk(),#
                 RelativeDrawdownatRisk(),# 59
                 ###
                 ConditionalValueatRisk(),#
                 DistributionallyRobustConditionalValueatRisk(),#
                 ConditionalValueatRiskRange(),#
                 ConditionalDrawdownatRisk(),#
                 RelativeConditionalDrawdownatRisk(),# 64
                 ###
                 EntropicValueatRisk(),#
                 EntropicValueatRiskRange(),#
                 EntropicDrawdownatRisk(),#
                 RelativeEntropicDrawdownatRisk(),# 68
                 ###
                 RelativisticValueatRisk(),#
                 RelativisticValueatRiskRange(),#
                 RelativisticDrawdownatRisk(),#
                 RelativeRelativisticDrawdownatRisk(),# 72
                 ###
                 OrderedWeightsArray(),#
                 OrderedWeightsArray(; w = owa_gmd(500)),#
                 OrderedWeightsArray(; w = owa_tg(500)),#
                 OrderedWeightsArray(; w = owa_tgrg(500)),# 76
                 ###
                 AverageDrawdown(),#
                 AverageDrawdown(; w = ew),#
                 RelativeAverageDrawdown(),#
                 RelativeAverageDrawdown(; w = ew),# 80
                 ###
                 UlcerIndex(),#
                 RelativeUlcerIndex(),# 82
                 ###
                 MaximumDrawdown(),#
                 RelativeMaximumDrawdown(),# 84
                 ###
                 WorstRealisation(),# 85
                 ###
                 Range(),# 86
                 ###
                 EqualRiskMeasure(),# 87
                 ###
                 TurnoverRiskMeasure(; w = w1),#
                 TrackingRiskMeasure(; tracking = WeightsTracking(; w = w1)),#
                 TrackingRiskMeasure(; tracking = ReturnsTracking(; w = pr.X * w1))]
        names = string.(risks)
        idx = [(findfirst(x -> x == '{', s) - 1) for s ∈ names]
        names = [n[1:i] for (n, i) ∈ zip(names, idx)]
        for (i, (risk, name)) ∈ enumerate(zip(risks, names))
            name = name * "_$(i)"
            w = optimise!(HierarchicalEqualRiskContribution(; ri = risk, opt = opt)).w
            res = if i ∈ 65:72
                isapprox(w, df[!, name]; rtol = 1e-7)
            else
                isapprox(w, df[!, name])
            end
            isapprox(w, df[!, name])
            if !res
                println("$name failed")
                find_tol(w, df[!, name]; name1 = :w1, name2 = :df)
            end
            @test res
        end

        rng = StableRNG(123456789)
        X = randn(rng, 1000, 15)
        rd = ReturnsResult(; nx = 1:size(X, 2), X = X)
        pr = prior(HighOrderPriorEstimator(), rd)
        clr = clusterise(ClusteringEstimator(), pr.X)
        opt = HierarchicalOptimiser(; pe = pr, cle = clr, slv = slv)
        sces = [SumScalariser(), MaxScalariser(), LogSumExpScalariser(; gamma = 1e-3),
                LogSumExpScalariser(; gamma = 3)]
        base = [LowOrderMoment(; alg = MeanAbsoluteDeviation(),
                               settings = RiskMeasureSettings(; scale = 2.3)),
                StandardDeviation(; settings = RiskMeasureSettings(; scale = 4.2))]
        risks = [(r1 = base, r2 = base);
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
                  r2 = base);
                 (r1 = base[1], r2 = base[2])]
        df = CSV.read(joinpath(@__DIR__, "./assets/HERC_vector_rm.csv"), DataFrame)
        for (idx, sce) ∈ zip(1:5:20, sces)
            opt = HierarchicalOptimiser(; pe = pr, cle = clr, sce = sce, slv = slv)
            for (i, rs) ∈ enumerate(risks)
                w = optimise!(HierarchicalEqualRiskContribution(; ri = rs.r1, ro = rs.r2,
                                                                opt = opt)).w
                res = isapprox(w, df[!, idx + i - 1])
                if !res
                    println("Failed on vector rm HERC\n$sce\n$i")
                    find_tol(w, df[!, idx + i - 1]; name1 = :w, name2 = :df)
                end
                @test res
            end
        end
    end
    @testset "Bounds tests" begin
        rng = StableRNG(987456321)
        X = randn(rng, 500, 10)
        rd = ReturnsResult(; nx = 1:size(X, 2), X = X)
        pr = prior(HighOrderPriorEstimator(), rd)
        clr = clusterise(ClusteringEstimator(), pr.X)
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     check_sol = (; allow_local = true, allow_almost = true),
                     settings = Dict("max_step_fraction" => 0.75, "verbose" => false))
        lb = [0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ub = [0.2, 1, 1, 1, 1, 1, 1, 1, 0.05, 0]
        opt = HierarchicalOptimiser(; pe = pr, cle = clr,
                                    wb = WeightBoundsResult(; lb = lb, ub = ub), slv = slv)
        r = [Variance(), ConditionalValueatRisk()]
        w = optimise!(HierarchicalRiskParity(; r = r, opt = opt)).w
        idx = (w - lb) .< 0
        if !isempty(w[idx])
            @test isapprox(w[idx], lb[idx])
        end
        @test all(w[.!idx] .>= lb[.!idx])

        idx = (ub - w) .< 0
        if !isempty(w[idx])
            @test isapprox(w[idx], ub[idx])
        end
        @test all(w[.!idx] .<= ub[.!idx])

        cwfs = [HeuristicClusteringWeightFiniliser(),
                JuMP_ClusteringWeightFiniliser(; slv = [slv]),
                JuMP_ClusteringWeightFiniliser(;
                                               alg = SquareRelativeErrorClusteringWeightFiniliser(),
                                               slv = [slv]),
                JuMP_ClusteringWeightFiniliser(;
                                               alg = AbsoluteErrorClusteringWeightFiniliser(),
                                               slv = [slv]),
                JuMP_ClusteringWeightFiniliser(;
                                               alg = SquareAbsoluteErrorClusteringWeightFiniliser(),
                                               slv = [slv])]
        for cwf ∈ cwfs
            opt = HierarchicalOptimiser(; pe = pr, cle = clr, cwf = cwf,
                                        wb = WeightBoundsResult(; lb = lb, ub = ub),
                                        slv = [slv])
            w = optimise!(HierarchicalEqualRiskContribution(; ri = r, opt = opt)).w
            idx = (w - lb) .< 0
            if !isempty(w[idx])
                @test isapprox(w[idx], lb[idx], atol = 1e-8)
            end
            @test all(w[.!idx] .>= lb[.!idx])

            idx = (ub - w) .< 0
            if !isempty(w[idx])
                @test isapprox(w[idx], ub[idx], atol = 1e-8)
            end
            @test all(w[.!idx] .<= ub[.!idx])
        end
    end
end
