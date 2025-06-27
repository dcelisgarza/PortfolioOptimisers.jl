@safetestset "Risk Budgetting Optimisation" begin
    using PortfolioOptimisers, Test, StableRNGs, Random, Clarabel, Logging
    Logging.min_enabled_level(Logging.Error)
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
    F = X[:, [3, 8, 5]]
    rd = ReturnsResult(; nx = 1:size(X, 2), X = X, F = F, nf = 1:size(F, 2))
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = Dict("max_step_fraction" => 0.75, "verbose" => false))

    @testset "Asset Risk Budgetting" begin
        pr = prior(EmpiricalPriorEstimator(), rd)
        opt = JuMPOptimiser(; pe = pr, slv = slv)
        r = PortfolioOptimisers.factory(StandardDeviation(), pr)

        rbe = RiskBudgetting(; r = r, opt = opt)
        w = optimise!(rbe, rd).w
        @test isapprox(w,
                       [0.09649877977932793, 0.10231012728204829, 0.0999953050117097,
                        0.1086543394297094, 0.11531372351730931, 0.09541263578464912,
                        0.08302626448412825, 0.10775060344823295, 0.09310343083507516,
                        0.09793479042780956], rtol = 5e-5)
        rkc = PortfolioOptimisers.risk_contribution(r, w, pr.X)
        lo, hi = extrema(rkc)
        @test isapprox(hi / lo, 1, rtol = 5e-4)

        rbe = RiskBudgetting(; alg = AssetRiskBudgettingAlgorithm(; rkb = 1:10), r = r,
                             opt = opt)
        w = optimise!(rbe, rd).w
        @test isapprox(w,
                       [0.04184999027515667, 0.06053084151206798, 0.07442610546167501,
                        0.0957262273025942, 0.11606124389782511, 0.10742404789538053,
                        0.10431444315880778, 0.13553211044329608, 0.12485529857576533,
                        0.13927969147743116], rtol = 5e-5)
        rkc = PortfolioOptimisers.risk_contribution(r, w, pr.X)
        lo, hi = extrema(rkc)
        if Sys.iswindows()
            isapprox(hi / lo, 10; rtol = 5e-4)
        else
            @test isapprox(hi / lo, 10, rtol = 1e-4)
        end
        @test argmin(rkc) == 1
        @test argmax(rkc) == 10

        rbe = RiskBudgetting(; alg = AssetRiskBudgettingAlgorithm(; rkb = 20:-2:2), r = r,
                             opt = opt)
        w = optimise!(rbe, rd).w
        @test isapprox(w,
                       [0.1344025967977924, 0.13916273306889104, 0.12912481639960144,
                        0.12798620808491343, 0.1238529129777297, 0.09343419617075599,
                        0.06918961804851861, 0.08410566411445848, 0.05859315323841707,
                        0.04014810109892192], rtol = 5e-5)
        rkc = PortfolioOptimisers.risk_contribution(r, w, pr.X)
        lo, hi = extrema(rkc)
        @test isapprox(hi / lo, 10, rtol = 1e-4)
        @test argmin(rkc) == 10
        @test argmax(rkc) == 1
    end
    @testset "Factor Risk Budgetting" begin
        pr = prior(EmpiricalPriorEstimator(), rd)
        pr1 = prior(FactorPriorEstimator(), rd)
        opt = JuMPOptimiser(; bgt = 1, sbgt = 1, wb = WeightBoundsResult(; lb = -1, ub = 1),
                            pe = pr, slv = slv)
        r = PortfolioOptimisers.factory(StandardDeviation(), pr)
        rbe = RiskBudgetting(; alg = FactorRiskBudgettingAlgorithm(;), r = r, opt = opt)
        w = optimise!(rbe, rd).w
        @test isapprox(w,
                       [-0.002429290248856789, 0.016703717820814843, 0.3301898270479341,
                        -0.011597999465757413, 0.3168767982204206, 0.01030398678339704,
                        0.002791313936417602, 0.31577738714679515, 0.003837597212000095,
                        0.017546661546835148], rtol = 5e-5)
        rkc = PortfolioOptimisers.factor_risk_contribution(r, w, pr.X; rd = rd)
        lo, hi = extrema(rkc[1:3])
        rtol = if Sys.iswindows()
            1e-4
        else
            5e-5
        end
        @test isapprox(hi / lo, 1, rtol = rtol)

        rbe = RiskBudgetting(; alg = FactorRiskBudgettingAlgorithm(; rkb = 1:3), r = r,
                             opt = opt)
        w = optimise!(rbe, rd).w
        rtol = if Sys.islinux()
            1e-5
        else
            5e-5
        end
        @test isapprox(w,
                       [-0.0024299694588815622, 0.01854069887141735, 0.2310357404166038,
                        -0.013001802561148969, 0.40416631720507656, 0.018345407610834332,
                        0.0002641355009267967, 0.3266300163091142, -0.006013363196136585,
                        0.022462819302194334], rtol = rtol)
        rkc = PortfolioOptimisers.factor_risk_contribution(r, w, pr.X; rd = rd)
        lo, hi = extrema(rkc[1:3])
        if Sys.iswindows()
            rtol = 1e-4
        else
            rtol = 5e-5
        end
        @test isapprox(hi / lo, 3, rtol = rtol)
        @test argmin(rkc[1:3]) == 1
        @test argmax(rkc[1:3]) == 3

        rbe = RiskBudgetting(; alg = FactorRiskBudgettingAlgorithm(; rkb = 3:-1:1), r = r,
                             opt = opt)
        w = optimise!(rbe, rd).w
        rtol = if Sys.iswindows()
            1e-4
        else
            5e-5
        end
        @test isapprox(w,
                       [-0.002488601638213621, 0.015091302915909951, 0.42175518018986385,
                        -0.011102599327113884, 0.222890735414167, 0.0018962013902890556,
                        0.006026217057528843, 0.32005595687913146, 0.013540301399654144,
                        0.012335305718783359], rtol = 5e-5)
        rkc = PortfolioOptimisers.factor_risk_contribution(r, w, pr.X; rd = rd)
        lo, hi = extrema(rkc[1:3])
        rtol = if Sys.isapple()
            1e-4
        else
            5e-5
        end
        @test isapprox(hi / lo, 3, rtol = rtol)
        @test argmin(rkc[1:3]) == 3
        @test argmax(rkc[1:3]) == 1

        rbe = RiskBudgetting(;
                             alg = FactorRiskBudgettingAlgorithm(; re = pr1.loadings,
                                                                 flag = false), r = r,
                             opt = opt)
        w = optimise!(rbe, rd).w

        @test isapprox(w,
                       [0.01798148641044756, 0.02736012290038294, 0.3126099638502706,
                        -0.021438538716377662, 0.30620437320730104, -0.044304652226991766,
                        0.036530300291336554, 0.2912085672785147, 0.037991847232659366,
                        0.03585652977245686], rtol = 1.0e-5)
        rkc = PortfolioOptimisers.factor_risk_contribution(r, w, pr.X; re = pr1.loadings)
        lo, hi = extrema(rkc[1:3])
        @test isapprox(hi / lo, 1, rtol = 1e-4)

        rbe = RiskBudgetting(;
                             alg = FactorRiskBudgettingAlgorithm(; re = pr1.loadings,
                                                                 flag = false, rkb = 1:3),
                             r = r, opt = opt)
        w = optimise!(rbe, rd).w
        rtol = if Sys.iswindows()
            5e-5
        else
            1e-5
        end
        @test isapprox(w,
                       [0.013331858790916563, 0.020285380567765268, 0.23177571639816452,
                        -0.028444938923181767, 0.4062760437696709, -0.047136682614741805,
                        0.027084346308040037, 0.3098231251240554, 0.0404203521512489,
                        0.026584798428062356], rtol = rtol)
        rkc = PortfolioOptimisers.factor_risk_contribution(r, w, pr.X; re = pr1.loadings)
        lo, hi = extrema(rkc[1:3])
        rtol = if Sys.iswindows()
            5e-4
        else
            4e-5
        end
        @test isapprox(hi / lo, 3, rtol = rtol)
        @test argmin(rkc[1:3]) == 1
        @test argmax(rkc[1:3]) == 3

        rbe = RiskBudgetting(;
                             alg = FactorRiskBudgettingAlgorithm(; re = pr1.loadings,
                                                                 flag = false,
                                                                 rkb = 3:-1:1), r = r,
                             opt = opt)
        w = optimise!(rbe, rd).w
        @test isapprox(w,
                       [0.021932937302796402, 0.03337253920355779, 0.3813063381330762,
                        -0.014560230766499187, 0.20796223075612943, -0.04380647167640382,
                        0.04455787289512887, 0.2879340929944269, 0.03756465057450146,
                        0.04373604058328648], rtol = 5e-5)
        rkc = PortfolioOptimisers.factor_risk_contribution(r, w, pr.X; re = pr1.loadings)
        lo, hi = extrema(rkc[1:3])
        @test isapprox(hi / lo, 3, rtol = 1e-4)
        @test argmin(rkc[1:3]) == 3
        @test argmax(rkc[1:3]) == 1

        pr1 = prior(FactorPriorEstimator(; re = DimensionReductionRegression()), rd)
        opt = JuMPOptimiser(; bgt = 1, sbgt = 1, wb = WeightBoundsResult(; lb = -1, ub = 1),
                            pe = pr, slv = slv)
        r = PortfolioOptimisers.factory(StandardDeviation(), pr)

        rbe = RiskBudgetting(;
                             alg = FactorRiskBudgettingAlgorithm(; re = pr1.loadings,
                                                                 flag = true), r = r,
                             opt = opt)
        w = optimise!(rbe, rd).w
        @test isapprox(w,
                       [0.2430091043789357, 0.27949606219933204, -0.6807262742678152,
                        0.3166182498845791, 0.24565207961358468, 0.3094482425427256,
                        0.14887690697462394, -0.3192736708875129, 0.22229429893911315,
                        0.2346050006224304], rtol = 5e-5)
        rkc = PortfolioOptimisers.factor_risk_contribution(r, w, pr.X; re = pr1.loadings)
        lo, hi = extrema(rkc[1:3])
        @test isapprox(hi / lo, 2, rtol = 5e-3)

        rbe = RiskBudgetting(;
                             alg = FactorRiskBudgettingAlgorithm(; re = pr1.loadings,
                                                                 flag = true, rkb = 1:3),
                             r = r, opt = opt)
        w = optimise!(rbe, rd).w
        @test isapprox(w,
                       [0.21801307545615856, 0.25074697795599155, -0.6895968221557025,
                        0.28405075198369034, 0.4261051806723497, 0.2776182290676173,
                        0.13356331858369896, -0.31040312303139017, 0.19942898797317485,
                        0.21047342349436762], rtol = 5e-5)
        rkc = PortfolioOptimisers.factor_risk_contribution(r, w, pr.X; re = pr1.loadings)
        lo, hi = extrema(rkc[1:3])
        @test isapprox(hi / lo, 10, rtol = 5e-2)
        @test argmin(rkc[1:3]) == 1
        @test argmax(rkc[1:3]) == 3

        rbe = RiskBudgetting(;
                             alg = FactorRiskBudgettingAlgorithm(; re = pr1.loadings,
                                                                 rkb = 3:-1:1, flag = true),
                             r = r, opt = opt)
        w = optimise!(rbe, rd).w
        @test isapprox(w,
                       [0.26953851489929054, 0.310008760379444, -0.6509260338445255,
                        0.3511836348742858, 0.05412884411478991, 0.3432310420437536,
                        0.16512993469430032, -0.34907391545157906, 0.24656228328146979,
                        0.2602169350087556], rtol = 5e-5)
        rkc = PortfolioOptimisers.factor_risk_contribution(r, w, pr.X; re = pr1.loadings,
                                                           rd = rd)
        lo, hi = extrema(rkc[1:3])
        @test isapprox(hi / lo, 3, rtol = 1e-1)
        @test argmin(rkc[1:3]) == 3
        @test argmax(rkc[1:3]) == 1
    end
end
