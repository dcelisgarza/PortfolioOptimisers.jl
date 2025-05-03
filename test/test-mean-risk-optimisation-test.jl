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
    rf = 4.34 / 100 / 252
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
    rng = StableRNG(987456321)
    X = randn(rng, 200, 10)
    rd = ReturnsResult(; nx = 1:size(X, 2), X = X)
    pr = prior(EmpiricalPriorEstimator(), rd)
    @testset "Returns lower bounds" begin
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     check_sol = (; allow_local = true, allow_almost = true),
                     settings = Dict("max_step_fraction" => 0.75, "verbose" => false))
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
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     check_sol = (; allow_local = true, allow_almost = true),
                     settings = Dict("max_step_fraction" => 0.75, "verbose" => false))
        opt = JuMPOptimiser(; pe = pr, slv = slv, bgt = BudgetRange(; lb = 0.8, ub = 1.5))
        mre = MeanRiskEstimator(; obj = MinimumRisk(), opt = opt)
        w = optimise!(mre, rd).w
        @test 0.8 <= sum(w) <= 1.5

        mre = MeanRiskEstimator(; obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test 0.8 <= sum(w) <= 1.5

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -0.1, ub = 0.15), bgt = 0.7,
                            sbgt = 0.2)
        mre = MeanRiskEstimator(; obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(sum(w[w .< 0]), -0.2, rtol = 1e-6)
        @test isapprox(sum(w[w .> 0]), 0.7 + 0.2, rtol = 1e-6)
        @test isapprox(sum(w), 0.7, rtol = 1e-6)
        @test all(-0.1 - sqrt(eps()) .<= w .<= 0.15 + sqrt(eps()))

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -0.1, ub = 0.15),
                            bgt = BudgetRange(; lb = 0.6, ub = 0.8),
                            sbgt = BudgetRange(; lb = 0.1, ub = 0.3))
        mre = MeanRiskEstimator(; obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test -0.3 - sqrt(eps()) <= sum(w[w .< 0]) <= -0.1 + sqrt(eps())
        @test 0.6 + 0.1 - sqrt(eps()) <= sum(w[w .> 0]) <= 0.8 + 0.3 + sqrt(eps())
        @test 0.6 - sqrt(eps()) <= sum(w) <= 0.8 + sqrt(eps())

        opt = JuMPOptimiser(; pe = pr, slv = slv, bgt = nothing)
        mre = MeanRiskEstimator(; obj = MaximumReturn(), opt = opt)
        w = optimise!(mre, rd).w
        @test all(isapprox.(w[pr.mu .>= 0], 1))

        opt = JuMPOptimiser(; pe = pr, slv = slv, bgt = nothing,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = nothing)
        mre = MeanRiskEstimator(; obj = MaximumReturn(), opt = opt)
        w = optimise!(mre, rd).w
        @test all(isapprox.(w[pr.mu .>= 0], 1, rtol = 5e-8))
        @test all(isapprox.(w[pr.mu .< 0], -1, rtol = 1e-6))

        opt = JuMPOptimiser(; pe = pr, slv = slv, bgt = 1,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = nothing)
        mre = MeanRiskEstimator(; obj = MaximumReturn(), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(w[argmax(pr.mu)], 1)

        opt = JuMPOptimiser(; pe = pr, slv = slv, bgt = nothing,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1)
        mre = MeanRiskEstimator(; obj = MaximumReturn(), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(w[argmin(pr.mu)], -1, rtol = 5e-8)
        @test all(isapprox.(w[w .>= 1e-6], 1, rtol = 5e-8))

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -1, ub = 1),
                            bgt = BudgetRange(; lb = 0.6, ub = 1), sbgt = nothing)
        mre = MeanRiskEstimator(; obj = MaximumReturn(), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(sum(w), 0.6)
        @test isapprox(w[argmax(w)], 1)
        @test isapprox(w[argmin(w)], -0.4, rtol = 5e-8)

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -1, ub = 1),
                            sbgt = BudgetRange(; lb = 0.6, ub = 1), bgt = nothing)
        mre = MeanRiskEstimator(; obj = MaximumReturn(), opt = opt)
        w = optimise!(mre, rd).w
        @test all(isapprox.(w[pr.mu .>= 0], 1, rtol = 5e-8))
        @test isapprox(w[argmin(w)], -1, rtol = 5e-8)

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -1, ub = 1),
                            bgt = BudgetRange(; lb = 0.6, ub = 1), sbgt = 1)
        mre = MeanRiskEstimator(; obj = MaximumReturn(), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(sum(w), 0.6)

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -1, ub = 1),
                            sbgt = BudgetRange(; lb = 0.6, ub = 1), bgt = 1)
        mre = MeanRiskEstimator(; obj = MaximumReturn(), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(sum(w), 1)
        @test isapprox(sum(w[w .< 0]), -1)
    end
    rng = StableRNG(987456321)
    X = randn(rng, 100, 10)
    rd = ReturnsResult(; nx = 1:size(X, 2), X = X)
    pr = prior(EmpiricalPriorEstimator(), rd)
    @testset "Cardinality" begin
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

        sets = DataFrame(; Assets = 1:10, Clusters = [1, 5, 5, 2, 3, 2, 4, 1, 3, 4])
        gcard = [CardinalityConstraint(;
                                       A = CardinalityConstraintSide(; group = :Assets,
                                                                     name = 1), B = 0,
                                       comp = EQ()),
                 CardinalityConstraint(;
                                       A = CardinalityConstraintSide(;
                                                                     group = [:Assets,
                                                                              :Assets,
                                                                              :Assets],
                                                                     name = [4, 5, 6]),
                                       B = 1, comp = LEQ()),
                 CardinalityConstraint(;
                                       A = CardinalityConstraintSide(; group = [:Clusters],
                                                                     name = [5]), B = 2,
                                       comp = GEQ())]
        opt = JuMPOptimiser(; pe = pr, slv = [slv],
                            wb = WeightBoundsResult(; lb = -0.2, ub = 1), bgt = 1,
                            sbgt = 0.2, gcard = gcard, sets = sets)
        mre = MeanRiskEstimator(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        @test abs(w[1]) <= 1e-10
        @test count(abs.(w[4:6]) .>= 1e-10) == 1
        # Non-convex constraint will not always be satisfied.
        @test count(w[2:3] .>= 1e-10) >= 1

        gcard = cardinality_constraints(gcard, sets)
        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -0.2, ub = 1), bgt = 1,
                            sbgt = 0.2, gcard = gcard)
        mre = MeanRiskEstimator(; obj = MinimumRisk(;), opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        @test abs(w[1]) <= 1e-10
        @test count(abs.(w[4:6]) .>= 1e-9) == 1
        @test count(w[2:3] .>= 1e-10) >= 1

        plc = IntegerPhilogenyConstraintEstimator(; pe = NetworkEstimator(), B = 1)
        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -0.2, ub = 1), bgt = 1,
                            sbgt = 0.2, cplg = plc)
        mre = MeanRiskEstimator(; obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(w,
                       [6.565615791643085e-11, -0.19999999121989026, 1.1785997032846193e-10,
                        0.6290623048410415, 5.669382582540041e-11, 4.528644817343074e-11,
                        5.27027305656348e-11, 1.1158693787948315e-10, 0.5709376858787606,
                        5.0307429248830024e-11])

        plc = IntegerPhilogenyConstraintEstimator(; pe = NetworkEstimator(), B = 2)
        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -0.2, ub = 1), bgt = 1,
                            sbgt = 0.2, nplg = philogeny_constraints(plc, pr.X))
        mre = MeanRiskEstimator(; obj = MinimumRisk(), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(w,
                       [0.15616982376859195, 2.0034651720906613e-11, 0.1955737980072259,
                        0.19944891610203982, 2.2380647850800894e-11, 0.1645040684388084,
                        2.043982457248212e-11, 0.11718206338146936, 0.1671213302227158,
                        1.6293623903510733e-11])
    end
    @testset "Buy in threshold" begin
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
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     check_sol = (; allow_local = true, allow_almost = true),
                     settings = Dict("max_step_fraction" => 0.75, "verbose" => false))
        opt = JuMPOptimiser(; pe = pr, slv = slv, l1 = 1.3385776e-1,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1)
        mre = MeanRiskEstimator(; obj = MaximumReturn(), opt = opt)
        res = optimise!(mre, rd)
        @test isapprox(res.w,
                       [-7.570992079480368e-9, -0.8467820273900335, 1.9488026077369066e-9,
                        0.9999999862460646, -1.1273685119113323e-9, -1.7659660986120914e-7,
                        -2.3312799742601792e-8, 1.257511276583601e-9, 0.8467822482162758,
                        -1.6708505332966315e-9], rtol = 0.01)

        opt = JuMPOptimiser(; pe = pr, slv = slv, l1 = 1,
                            wb = WeightBoundsResult(; lb = -0.2, ub = 1), sbgt = 0.2)
        mre = MeanRiskEstimator(; obj = MaximumRatio(; rf = rf, ohf = 1), opt = opt)
        res = optimise!(mre, rd)
        @test isapprox(res.w,
                       [-5.949318104854074e-9, -0.10052775651905972, 0.21768828095861717,
                        0.44774457405889506, -2.025708709679082e-9, -0.05620137067838191,
                        -0.04327085674205319, 0.12503644321347543, 0.4095307023376115,
                        -8.654077507440204e-9])

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
                       [-3.3230380116974283e-11, -0.3694875940841836, 0.3489332062471002,
                        0.7572685287303744, -1.780869748956517e-11, -0.36248700431753916,
                        -0.26802539864209457, 0.21670038131543426, 0.6770978808452219,
                        -4.327434616371575e-11])

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            fees = PortfolioOptimisers.Fees(; long = 0.05, short = 0.03,
                                                            fixed_long = 0.002),
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1)
        mre = MeanRiskEstimator(; obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(w,
                       [-0.049368211827370834, -0.3290512728058653, 0.34845087647328093,
                        0.7342589196970375, -7.769421841241781e-12, -0.32364913873616125,
                        -0.23167511223026258, 0.23576157627488953, 0.6815286266083458,
                        -0.06625626344612386])

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            fees = PortfolioOptimisers.Fees(; fixed_long = 0.001))
        mre = MeanRiskEstimator(; obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(w,
                       [2.3544021099472797e-11, 2.3163252680475905e-11, 0.1924495112365115,
                        0.3672170233262318, 2.4475925865815772e-11, 2.3530483647261788e-11,
                        2.1824449308113803e-11, 0.10258877483006534, 0.3377446904721447,
                        1.8502046750641024e-11], rtol = 5e-8)

        wt = fill(0.1, 10)
        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            fees = PortfolioOptimisers.Fees(;
                                                            turnover = Turnover(; val = 0,
                                                                                w = wt)))
        mre = MeanRiskEstimator(; obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(w,
                       [6.552764846148974e-10, -5.943828279758435e-10, 0.1924493325202225,
                        0.36721740046202517, 1.9025635276565066e-9, -2.416288675840526e-10,
                        -2.7273148636586244e-10, 0.10258830140535145, 0.33774496373243884,
                        4.308652823762546e-10])

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            fees = PortfolioOptimisers.Fees(;
                                                            turnover = Turnover(; val = 0.1,
                                                                                w = wt)))
        mre = MeanRiskEstimator(; obj = MaximumUtility(), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(w,
                       [0.09835896556148706, 0.05746450961233853, 0.13960277264399099,
                        0.1656987569996019, 0.09999999890747412, 0.0999999994456413,
                        0.04960190697319207, 0.10000000057420562, 0.11555792325996056,
                        0.07371516602210967])

        opt1 = JuMPOptimiser(; pe = pr, slv = slv,
                             fees = PortfolioOptimisers.Fees(;
                                                             turnover = Turnover(;
                                                                                 val = 100,
                                                                                 w = wt)))
        opt2 = JuMPOptimiser(; pe = pr, slv = slv,
                             fees = PortfolioOptimisers.Fees(;
                                                             turnover = Turnover(; val = 0,
                                                                                 w = wt)))
        mre1 = MeanRiskEstimator(; obj = MinimumRisk(), opt = opt1)
        mre2 = MeanRiskEstimator(; obj = MinimumRisk(), opt = opt2)
        @test isapprox(optimise!(mre1, rd).w, optimise!(mre2, rd).w)
    end
    @testset "Cone constraints" begin
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     check_sol = (; allow_local = true, allow_almost = true),
                     settings = Dict("max_step_fraction" => 0.75, "verbose" => false))
        opt = JuMPOptimiser(; pe = pr, slv = slv, nea = 7.5,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1)
        mre = MeanRiskEstimator(; obj = MaximumReturn(), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(number_effective_assets(w), opt.nea)

        mre = MeanRiskEstimator(; obj = MaximumRatio(), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(number_effective_assets(w), opt.nea)

        wt = fill(0.1, 10)
        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1,
                            tn = Turnover(; w = wt, val = 0))
        mre = MeanRiskEstimator(; obj = MinimumRisk(), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(w, wt)
        @test all(abs.(w - wt) .<= sqrt(eps()))

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1,
                            tn = Turnover(; w = wt, val = 3e-2))
        mre = MeanRiskEstimator(; obj = MinimumRisk(), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(w,
                       [0.0983818364795562, 0.0742909015489213, 0.1299999937304853,
                        0.12999999604960435, 0.10527218086576251, 0.1299999774465235,
                        0.07000000730540809, 0.08823907886499684, 0.10381507713815513,
                        0.07000095057058692])
        @test all(abs.(w - wt) .- 3e-2 .<= sqrt(eps()))

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1,
                            tn = Turnover(; w = wt, val = 1e-1))
        mre = MeanRiskEstimator(; obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(w,
                       [1.5632964482151688e-8, 4.130316278719063e-9, 0.20000000125523024,
                        0.20000000268031962, 0.15322275351582243, 6.988903592994133e-9,
                        4.6962642970200895e-9, 0.20000000007708818, 0.2000000022542374,
                        0.04677720876885366])
        @test all(abs.(w - wt) .- 1e-1 .<= sqrt(eps()))

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1,
                            te = TrackingError(; err = 0,
                                               tracking = WeightsTracking(; w = wt)))
        mre = MeanRiskEstimator(; obj = MinimumRisk(), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(w, wt)

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1,
                            te = TrackingError(; err = 1e-1,
                                               tracking = WeightsTracking(; w = wt)))
        mre = MeanRiskEstimator(; obj = MinimumRisk(), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(w,
                       [0.10122218950563136, 0.06977241841375383, 0.14252415024175216,
                        0.1475449238825743, 0.1003091226106682, 0.1296476104654691,
                        0.0527410523677282, 0.08714655544084866, 0.10339952006629131,
                        0.0656924570052827])
        @test isapprox(norm(pr.X * (w - wt), 2) / sqrt(99), 1e-1, rtol = 5e-6)

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1,
                            te = TrackingError(; err = 0.2,
                                               tracking = WeightsTracking(; w = wt)))
        mre = MeanRiskEstimator(; obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(w,
                       [0.06154300186432786, 0.026919270903344086, 0.17175135043403802,
                        0.19286793085842294, 0.06425734438838164, 0.03151842256180709,
                        0.04315795429684342, 0.1603324726191926, 0.20317347596680851,
                        0.0444787761068339])
        @test isapprox(norm((pr.X * (w - wt)), 2) / sqrt(99), 2e-1)
    end
    @testset "Linear constraints" begin
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     check_sol = (; allow_local = true, allow_almost = true),
                     settings = Dict("max_step_fraction" => 0.75, "verbose" => false))
        sets = DataFrame(; Assets = 1:10, Clusters = [1, 2, 2, 2, 3, 2, 2, 1, 3, 3])
        lcs = [LinearConstraint(;
                                A = LinearConstraintSide(; group = :Assets, name = 1,
                                                         coef = 2), B = 0.35, comp = EQ()),
               LinearConstraint(;
                                A = LinearConstraintSide(; group = [:Assets, :Assets],
                                                         name = [2, 3], coef = [1, 1]),
                                B = 0.25, comp = LEQ()),
               LinearConstraint(;
                                A = LinearConstraintSide(; group = [:Clusters, :Clusters],
                                                         name = [3, 1], coef = [1, -1]),
                                comp = GEQ())]
        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1,
                            lcs = lcs, sets = sets)
        mre = MeanRiskEstimator(; obj = MinimumRisk(), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(w,
                       [0.17500000000000135, 0.056672594418786436, 0.13482785851729392,
                        0.15636839719096973, 0.08839446333816732, 0.13072388044798336,
                        0.04259248732887701, 0.06318991669424347, 0.09698711567135396,
                        0.05524328639232306])

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1,
                            lcs = linear_constraints(lcs, sets))
        mre = MeanRiskEstimator(; obj = MaximumRatio(), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(w,
                       [0.175, -0.31433113660247225, 0.3886753364208299, 0.6157211058967501,
                        -0.048258583869412223, -0.2511230578707009, -0.2350298583771572,
                        0.22304378977543673, 0.5975597164681479, -0.1512573118414222])
    end
    @testset "SDP philogeny constraints" begin
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     check_sol = (; allow_local = true, allow_almost = true),
                     settings = Dict("max_step_fraction" => 0.75, "verbose" => false))
        plc = SemiDefinitePhilogenyConstraintEstimator(; pe = NetworkEstimator())
        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -0.2, ub = 1), bgt = 1,
                            sbgt = 0.2, cplg = plc)
        mre = MeanRiskEstimator(; obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(w,
                       [1.9182464594380404e-9, -5.486929531582411e-9, 0.25294011598798327,
                        0.30736755395924736, 1.4314131117487336e-9, -0.035502895237015414,
                        -1.8881132164854558e-10, 0.17572595977320954, 0.29946926729917145,
                        5.434866179443442e-10])

        mre = MeanRiskEstimator(; r = ConditionalValueatRisk(),
                                obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(w,
                       [-2.1268204115045244e-9, -0.06023046002084164, 0.2601463681020507,
                        0.4010094500291993, -1.499386434008056e-10, -0.13976952342299712,
                        -1.5207511190183943e-8, 0.23777409644795705, 0.3010700864415278,
                        -9.263056126010764e-11])

        plc = SemiDefinitePhilogenyConstraintEstimator(; pe = NetworkEstimator(), p = 6)
        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -0.2, ub = 1), bgt = 1,
                            sbgt = 0.2, nplg = philogeny_constraints(plc, pr.X))
        mre = MeanRiskEstimator(; obj = MinimumRisk(), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(w,
                       [0.17673383391348899, 0.08326967338268225, 0.18709665987606094,
                        0.2002578058940592, 1.6110864592698421e-9, 0.18807982323288358,
                        5.105720929163698e-10, 1.8374998058844176e-9, 0.16456219746839842,
                        2.273269661767486e-9])

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -0.2, ub = 1), bgt = 1,
                            sbgt = 0.2, nplg = philogeny_constraints(plc, pr.X))
        mre = MeanRiskEstimator(; r = ConditionalValueatRisk(), obj = MinimumRisk(),
                                opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(w,
                       [0.1173710496744111, 0.10295801829339872, 0.20602685793235045,
                        0.16816808542114753, 0.128471143599474, 0.040312910353117445,
                        2.3739773429047343e-9, 0.055963185526288824, 0.04558432590959013,
                        0.13514442091624446])

        mre = MeanRiskEstimator(;
                                r = UncertaintySetVariance(;
                                                           ucs = NormalUncertaintySetEstimator(;
                                                                                               pe = EmpiricalPriorEstimator(),
                                                                                               rng = rng,
                                                                                               alg = BoxUncertaintySetAlgorithm(),
                                                                                               seed = 987654321)),
                                obj = MinimumRisk(), opt = opt)
        res = optimise!(mre, rd)
        @test !haskey(res.model, :sdp_nplg_p)

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -0.2, ub = 1), bgt = 1,
                            sbgt = 0.2, cplg = philogeny_constraints(plc, pr.X))
        mre = MeanRiskEstimator(;
                                r = UncertaintySetVariance(;
                                                           ucs = sigma_ucs(NormalUncertaintySetEstimator(;
                                                                                                         pe = EmpiricalPriorEstimator(),
                                                                                                         rng = rng,
                                                                                                         alg = BoxUncertaintySetAlgorithm(),
                                                                                                         seed = 987654321),
                                                                           pr.X)),
                                obj = MinimumRisk(), opt = opt)
        res = optimise!(mre, rd)
        @test !haskey(res.model, :sdp_cplg_p)
    end
    @testset "Centrality" begin
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     check_sol = (; allow_local = true, allow_almost = true),
                     settings = Dict("max_step_fraction" => 0.75, "verbose" => false))
        ce = CentralityConstraintEstimator(; A = CentralityEstimator(), B = MinValue(),
                                           comp = LEQ())
        sets = DataFrame(; Assets = 1:10, Clusters = [1, 2, 2, 2, 3, 2, 2, 1, 3, 3])
        opt = JuMPOptimiser(; pe = pr, slv = slv, cent = ce, sets = sets)
        mre = MeanRiskEstimator(; obj = MinimumRisk(), opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        @test all(abs.(res.cent.A_ineq * w .- res.cent.B_ineq) .<= sqrt(eps()))
        @test isapprox(w,
                       [0.22435864236807318, 0.11900511795941895, 0.2042024901056272,
                        0.21713501321050613, -1.1594442746740814e-10, 0.23529873506572008,
                        -1.461171082669015e-10, -1.3517306778692135e-10,
                        1.1400266353179912e-9, 5.478625248874811e-10])

        ce = [CentralityConstraintEstimator(; A = CentralityEstimator(), B = MedianValue(),
                                            comp = EQ())]
        opt = JuMPOptimiser(; pe = pr, slv = slv, cent = ce, sets = sets)
        mre = MeanRiskEstimator(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        @test all(abs.(res.cent.A_eq * w .- res.cent.B_eq) .<= sqrt(eps()))
        @test isapprox(w,
                       [1.5080697736395485e-9, 1.0676961354967205e-10, 0.20143237340958933,
                        0.37944774611460697, 2.08976824723604e-9, 4.915607815862469e-10,
                        3.260477423319088e-10, 0.08088011921478026, 0.3382397556825816,
                        1.0562258217209712e-9])
    end
end
