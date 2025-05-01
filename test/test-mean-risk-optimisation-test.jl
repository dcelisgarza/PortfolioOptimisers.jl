@safetestset "MeanRisk Optimisation" begin
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
    @testset "MeanRisk measures" begin
        rng = StableRNG(987456321)
        X = randn(rng, 200, 10)
        rd = ReturnsResult(; nx = 1:size(X, 2), X = X)
        pr = prior(HighOrderPriorEstimator(), rd)
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     check_sol = (; allow_local = true, allow_almost = true),
                     settings = Dict("max_step_fraction" => 0.75, "verbose" => false))
        ew = eweights(1:size(X, 1), inv(size(X, 1)); scale = true)
        w1 = fill(inv(10), 10)
        rf = 4.34 / 100 / 252
        sigma = cov(GerberCovariance(), X)
        mu = vec(mean(ShrunkExpectedReturns(; ce = GerberCovariance()), X))
        kt = cokurtosis(Cokurtosis(; alg = Semi()), X; mean = transpose(mu))
        sk, V = coskewness(Coskewness(; alg = Semi()), X; mean = transpose(mu))
        ucs1 = sigma_ucs(NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                                       rng = rng,
                                                       alg = BoxUncertaintySetAlgorithm(),
                                                       seed = 987654321), X)
        ucs2 = sigma_ucs(NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                                       rng = rng,
                                                       alg = EllipseUncertaintySetAlgorithm(),
                                                       seed = 987654321), X)
        df = CSV.read(joinpath(@__DIR__, "./assets/MeanRisk.csv"), DataFrame)
        risks = [Variance(; sigma = sigma),#
                 Variance(),#
                 Variance(; formulation = Quad()),#
                 Variance(; formulation = RSOC()),# 4
                 ###
                 UncertaintySetVariance(; sigma = sigma, ucs = ucs1),#
                 UncertaintySetVariance(; ucs = ucs2),# 6
                 ###
                 StandardDeviation(; sigma = sigma),#
                 StandardDeviation(;),# 8
                 ###
                 LowOrderMoment(; mu = mu),#
                 LowOrderMoment(; mu = rf),#
                 LowOrderMoment(; w = ew),#
                 LowOrderMoment(),# 12
                 ###
                 LowOrderMoment(; alg = SemiDeviation(), mu = mu),#
                 LowOrderMoment(; alg = SemiDeviation(), mu = rf),#
                 LowOrderMoment(; alg = SemiDeviation(), w = ew),#
                 LowOrderMoment(; alg = SemiDeviation()),# 16
                 ###
                 LowOrderMoment(; alg = SemiVariance(), mu = mu),#
                 LowOrderMoment(; alg = SemiVariance(), mu = rf),#
                 LowOrderMoment(; alg = SemiVariance(), w = ew),#
                 LowOrderMoment(; alg = SemiVariance()),# 20
                 ###
                 LowOrderMoment(; alg = MeanAbsoluteDeviation(), mu = mu),#
                 LowOrderMoment(; alg = MeanAbsoluteDeviation(), mu = rf),#
                 LowOrderMoment(; alg = MeanAbsoluteDeviation(), w = ew),#
                 LowOrderMoment(; alg = MeanAbsoluteDeviation(; w = ew)),#
                 LowOrderMoment(; alg = MeanAbsoluteDeviation(;)),# 25
                 ###
                 SquareRootKurtosis(),#
                 SquareRootKurtosis(; kt = kt),#
                 SquareRootKurtosis(; N = 2),# 28
                 ###
                 NegativeSkewness(),#
                 NegativeSkewness(; alg = QuadraticNegativeSkewness()),#
                 NegativeSkewness(; sk = sk, V = V),#
                 NegativeSkewness(; alg = QuadraticNegativeSkewness(), sk = sk, V = V),# 32
                 ###
                 ConditionalValueatRisk(),#
                 DistributionallyRobustConditionalValueatRisk(; l = 2.5, r = 0.85),#
                 ConditionalValueatRiskRange(),#
                 ConditionalDrawdownatRisk(),# 36
                 ###
                 EntropicValueatRisk(),#
                 EntropicValueatRiskRange(),#
                 EntropicDrawdownatRisk(),# 39
                 ###
                 RelativisticValueatRisk(),#
                 RelativisticValueatRiskRange(),#
                 RelativisticDrawdownatRisk(),# 42
                 ###
                 OrderedWeightsArray(),#
                 OrderedWeightsArray(; w = owa_tg(200)),#
                 OrderedWeightsArrayRange(;),# 45
                 ###
                 OrderedWeightsArray(; formulation = ExactOrderedWeightsArray()),#
                 OrderedWeightsArray(; formulation = ExactOrderedWeightsArray(),
                                     w = owa_tg(200)),#
                 OrderedWeightsArrayRange(; formulation = ExactOrderedWeightsArray()),# 48
                 ###
                 AverageDrawdown(),#
                 AverageDrawdown(; w = ew),# 50
                 ###
                 UlcerIndex(),# 51
                 ###
                 MaximumDrawdown(),# 52
                 ###
                 WorstRealisation(),# 53
                 ###
                 Range(),# 54
                 ###
                 TurnoverRiskMeasure(; w = w1),#
                 TrackingRiskMeasure(; tracking = WeightsTracking(; w = w1)),#
                 TrackingRiskMeasure(; tracking = ReturnsTracking(; w = pr.X * w1))]
        objs = [MinimumRisk(), MaximumUtility(), MaximumUtility(; l = 1),
                MaximumRatio(; ohf = 1), MaximumRatio(;), MaximumRatio(; rf = rf, ohf = 1),
                MaximumRatio(; rf = rf), MaximumReturn()]
        rets = [ArithmeticReturn(), KellyReturn()]
        i = 1
        for r ∈ risks
            for obj ∈ objs
                for ret ∈ rets
                    opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
                    sol1 = optimise!(MeanRiskEstimator(; r = r, obj = obj, opt = opt))
                    sol2 = optimise!(MeanRiskEstimator(; r = [r, r], obj = obj, opt = opt))
                    if isa(sol1.retcode, OptimisationFailure) ||
                       isa(sol2.retcode, OptimisationFailure)
                        i += 1
                        continue
                    end
                    w1 = sol1.w
                    w2 = sol1.w
                    @test isapprox(w1, w2)
                    wt = df[!, i]
                    res = if i ∈ (92, 140, 142, 160, 364, 380, 396, 460, 474, 506, 510, 520,
                                  540, 556)
                        isapprox(w1, wt; rtol = 1e-4)
                    elseif i == 726
                        isapprox(w1, wt; rtol = 5e-4)
                    else
                        isapprox(w1, wt; rtol = 5e-5)
                    end
                    if !res
                        println("Iteration $i failed.")
                        find_tol(w1, wt; name1 = :w1, name2 = :wt)
                    end
                    @test res

                    i += 1
                end
            end
        end
    end
    @testset "Returns lower bounds" begin
        rng = StableRNG(987456321)
        X = randn(rng, 200, 10)
        rd = ReturnsResult(; nx = 1:size(X, 2), X = X)
        pr = prior(EmpiricalPriorEstimator(), rd)
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     check_sol = (; allow_local = true, allow_almost = true),
                     settings = Dict("max_step_fraction" => 0.75, "verbose" => false))
        rf = 4.34 / 100 / 252
        ucs1 = mu_ucs(NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                                    rng = rng,
                                                    alg = BoxUncertaintySetAlgorithm(),
                                                    seed = 987654321), X)
        ucs2 = mu_ucs(NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                                    rng = rng,
                                                    alg = EllipseUncertaintySetAlgorithm(),
                                                    seed = 987654321), X)

        ret = ArithmeticReturn()
        opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
        mre = MeanRiskEstimator(; obj = MaximumRatio(; rf = rf), opt = opt)
        res1 = optimise!(mre)

        ret = ArithmeticReturn(; lb = dot(res1.w, pr.mu))
        opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
        mre = MeanRiskEstimator(; obj = MinimumRisk(), opt = opt)
        res2 = optimise!(mre)
        @test abs(ret.lb - dot(res2.w, pr.mu)) <= 1e-9

        opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
        mre = MeanRiskEstimator(; obj = MaximumRatio(), opt = opt)
        res3 = optimise!(mre)
        @test abs(ret.lb - dot(res3.w, pr.mu)) <= 5e-7

        ret = KellyReturn()
        opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
        mre = MeanRiskEstimator(; obj = MaximumRatio(; rf = rf), opt = opt)
        res1 = optimise!(mre)

        ret = KellyReturn(; lb = mean(log1p.(pr.X * res1.w)))
        opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
        mre = MeanRiskEstimator(; obj = MinimumRisk(), opt = opt)
        res2 = optimise!(mre)
        @test abs(mean(log1p.(pr.X * res2.w)) - ret.lb) <= 1e-10

        opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
        mre = MeanRiskEstimator(; obj = MaximumRatio(), opt = opt)
        res3 = optimise!(mre)
        @test abs(mean(log1p.(pr.X * res3.w)) - ret.lb) <= 1e-8

        ucss = (ucs1, ucs2)
        objs = (MinimumRisk(), MaximumRatio(; rf = rf))
        df = CSV.read(joinpath(@__DIR__, "./assets/MeanRisk_ReturnUncertainty.csv"),
                      DataFrame)
        i = 1
        for ucs ∈ ucss
            for obj ∈ objs
                ret = ArithmeticReturn(; ucs = ucs)
                opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
                mre = MeanRiskEstimator(; obj = obj, opt = opt)
                res1 = optimise!(mre)
                w = res1.w
                wt = df[!, i]
                res = if i == 4
                    isapprox(w, wt; rtol = 1e-7)
                else
                    isapprox(w, wt)
                end
                if !res
                    println("Iteration $i failed.")
                    find_tol(w, wt; name1 = :w, name2 = :wt)
                end
                @test res
                df[!, "$(i)"] = res1.w
                i += 1
            end
        end
    end
    @testset "Budget range" begin
        rng = StableRNG(987456321)
        X = randn(rng, 200, 10)
        rd = ReturnsResult(; nx = 1:size(X, 2), X = X)
        pr = prior(EmpiricalPriorEstimator(), rd)
        rf = 4.34 / 100 / 252
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     check_sol = (; allow_local = true, allow_almost = true),
                     settings = Dict("max_step_fraction" => 0.75, "verbose" => false))
        opt = JuMPOptimiser(; pe = pr, slv = slv, bgt = BudgetRange(; lb = 0.8, ub = 1.5))
        mre = MeanRiskEstimator(; obj = MinimumRisk(), opt = opt)
        res = optimise!(mre, rd)
        @test 0.8 <= sum(res.w) <= 1.5

        mre = MeanRiskEstimator(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mre, rd)
        @test 0.8 <= sum(res.w) <= 1.5

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -0.1, ub = 0.15), bgt = 0.7,
                            sbgt = 0.2)
        mre = MeanRiskEstimator(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        @test isapprox(sum(w[w .< 0]), -0.2, rtol = 1e-6)
        @test isapprox(sum(w[w .> 0]), 0.7 + 0.2, rtol = 1e-6)
        @test isapprox(sum(w), 0.7, rtol = 1e-6)
        @test all(-0.1 - sqrt(eps()) .<= w .<= 0.15 + sqrt(eps()))

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -0.1, ub = 0.15),
                            bgt = BudgetRange(; lb = 0.6, ub = 0.8),
                            sbgt = BudgetRange(; lb = 0.1, ub = 0.3))
        mre = MeanRiskEstimator(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        @test -0.3 - sqrt(eps()) <= sum(w[w .< 0]) <= -0.1 + sqrt(eps())
        @test 0.6 + 0.1 - sqrt(eps()) <= sum(w[w .> 0]) <= 0.8 + 0.3 + sqrt(eps())
        @test 0.6 - sqrt(eps()) <= sum(w) <= 0.8 + sqrt(eps())
    end
    @testset "Cardinality" begin
        rng = StableRNG(987456321)
        X = randn(rng, 200, 10)
        rd = ReturnsResult(; nx = 1:size(X, 2), X = X)
        pr = prior(EmpiricalPriorEstimator(), rd)
        rf = 4.34 / 100 / 252
        slv = Solver(; name = :clarabel,
                     solver = solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                                 "verbose" => false,
                                                                 "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                          MOI.Silent() => true),
                                                                 "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                             "verbose" => false,
                                                                                                             "max_step_fraction" => 0.75)),
                     check_sol = (; allow_local = true, allow_almost = true))
        opt = JuMPOptimiser(; pe = pr, slv = slv, card = 3)
        mre = MeanRiskEstimator(; obj = MinimumRisk(), opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        @test length(w[w .> 1e-9]) <= 3

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -0.2, ub = 1), bgt = 1,
                            sbgt = 0.2, card = 5)
        mre = MeanRiskEstimator(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        @test length(w[abs.(w) .> 1e-9]) <= 5
    end
    @testset "Buy in threshold" begin
        rng = StableRNG(987456321)
        X = randn(rng, 100, 10)
        rd = ReturnsResult(; nx = 1:size(X, 2), X = X)
        pr = prior(EmpiricalPriorEstimator(), rd)
        rf = 4.34 / 100 / 252
        slv = Solver(; name = :clarabel,
                     solver = solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                                 "verbose" => false,
                                                                 "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                          MOI.Silent() => true),
                                                                 "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                             "verbose" => false,
                                                                                                             "max_step_fraction" => 0.75)),
                     check_sol = (; allow_local = true, allow_almost = true))

        opt = JuMPOptimiser(; pe = pr, slv = slv, lt = 0.2)
        mre = MeanRiskEstimator(; obj = MinimumRisk(), opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        @test all(isapprox.(w[abs.(w) .> 1e-9], 0.2))

        opt = JuMPOptimiser(; pe = pr, slv = slv, lt = 0.2)
        mre = MeanRiskEstimator(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        @test all(w[abs.(w) .> 1e-9] .- 0.2 .>= -sqrt(eps()))

        opt = JuMPOptimiser(; str_names = true, pe = pr, slv = slv, lt = 0.5, st = 0.15,
                            sbgt = 1, bgt = 0.7, wb = WeightBoundsResult(; lb = -1, ub = 1))
        mre = MeanRiskEstimator(; obj = MinimumRisk(), opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        w = w[abs.(w) .> 1e-9]
        @test all(w[w .< 0] .+ 0.15 .<= sqrt(eps()))
        @test all(w[w .>= 0] .- 0.5 .>= -sqrt(eps()))

        opt = JuMPOptimiser(; str_names = true, pe = pr, slv = slv, lt = 0.15, st = 0.2,
                            sbgt = 0.5, bgt = 1,
                            wb = WeightBoundsResult(; lb = -0.5, ub = 1))
        mre = MeanRiskEstimator(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        w = w[abs.(w) .> 1e-9]
        @test all(w[w .< 0] .+ 0.2 .<= sqrt(eps()))
        @test all(w[w .>= 0] .- 0.15 .>= -sqrt(eps()))
    end
    @testset "L1 and L2 penalties" begin
        rng = StableRNG(987456321)
        X = randn(rng, 100, 10)
        rd = ReturnsResult(; nx = 1:size(X, 2), X = X)
        pr = prior(EmpiricalPriorEstimator(), rd)
        rf = 4.34 / 100 / 252
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     check_sol = (; allow_local = true, allow_almost = true),
                     settings = Dict("max_step_fraction" => 0.75, "verbose" => false))
        opt = JuMPOptimiser(; pe = pr, slv = slv, l1 = 1.3385776e-1,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1)
        mre = MeanRiskEstimator(; obj = MaximumReturn(), opt = opt)
        res = optimise!(mre, rd)
        @test isapprox(res.w,
                       [-7.1115844550380404e-9, -0.6749708655857565, 2.095004012843859e-9,
                        0.9999999850136296, -1.005847682601429e-9, -1.8002229418234852e-7,
                        -2.203591023452724e-8, 1.3831254223699182e-9, 0.6749710887962451,
                        -1.526611091533063e-9], rtol = 0.01)

        opt = JuMPOptimiser(; pe = pr, slv = slv, l1 = 1,
                            wb = WeightBoundsResult(; lb = -0.2, ub = 1), sbgt = 0.2)
        mre = MeanRiskEstimator(; obj = MaximumRatio(; rf = rf, ohf = 1), opt = opt)
        res = optimise!(mre, rd)
        @test isapprox(res.w,
                       [-1.0686731420715847e-8, -0.10052767446696687, 0.21768831645458886,
                        0.44774450409664784, -3.6781525080429386e-9, -0.05620134434165095,
                        -0.043270941710922294, 0.1250365510849717, 0.40953061881510255,
                        -1.5566886983763398e-8])

        opt = JuMPOptimiser(; pe = pr, slv = slv, l2 = 1.5,
                            wb = WeightBoundsResult(; lb = -0.2, ub = 1), sbgt = 0.2)
        objs = (MinimumRisk(), MaximumUtility(), MaximumRatio(; rf = rf), MaximumReturn())
        l2s = CSV.read(joinpath(@__DIR__, "./assets/MeanRisk_L2.csv"), DataFrame)
        for (i, obj) ∈ enumerate(objs)
            mre = MeanRiskEstimator(; obj = MaximumReturn(), opt = opt)
            w = optimise!(mre, rd).w
            res = isapprox(w, l2s[!, i])
            if !res
                println("Iteration $i failed.")
                find_tol(w, l2s[!, i]; name1 = :w, name2 = :l2s)
            end
            @test res
        end
    end
    @testset "Fees" begin
        rng = StableRNG(987456321)
        X = randn(rng, 100, 10)
        rd = ReturnsResult(; nx = 1:size(X, 2), X = X)
        pr = prior(EmpiricalPriorEstimator(), rd)
        rf = 4.34 / 100 / 252
        slv = Solver(; name = :clarabel,
                     solver = solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                                 "verbose" => false,
                                                                 "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                          MOI.Silent() => true),
                                                                 "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                             "verbose" => false,
                                                                                                             "max_step_fraction" => 0.75)),
                     check_sol = (; allow_local = true, allow_almost = true))
        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            fees = PortfolioOptimisers.Fees(; long = 0.05, short = 0.03,
                                                            fixed_long = 0.002,
                                                            fixed_short = 0.0006),
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1)
        mre = MeanRiskEstimator(; obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(w,
                       [-4.3388784302049525e-11, -0.37047964581382725, 0.33550365102682617,
                        0.7750413188680487, -2.440133003012612e-11, -0.365013275656182,
                        -0.2645070750278348, 0.19954620198210546, 0.6899088247421615,
                        -5.350722819954533e-11])

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            fees = PortfolioOptimisers.Fees(; long = 0.05, short = 0.03,
                                                            fixed_long = 0.002),
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1)
        mre = MeanRiskEstimator(; obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(w,
                       [-0.044445488883193884, -0.33539095033394956, 0.33816334192422765,
                        0.751070537889643, -9.196473098683769e-12, -0.3313002556886578,
                        -0.2338934257213684, 0.220000605563935, 0.6907655130377879,
                        -0.05496987777922733])
    end
end
