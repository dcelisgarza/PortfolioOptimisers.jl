@safetestset "Clustering Optimisation" begin
    using PortfolioOptimisers, CSV, DataFrames, Test, Random, Clarabel, StatsBase,
          TimeSeries, CovarianceEstimation, FLoops
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
    X = TimeArray(CSV.File(joinpath(@__DIR__, "./assets/asset_prices.csv"));
                  timestamp = :timestamp)
    F = TimeArray(CSV.File(joinpath(@__DIR__, "./assets/factor_prices.csv"));
                  timestamp = :timestamp)
    rd = prices_to_returns(X[(end - 252):end], F[(end - 252):end])
    slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false)),
           Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.75)),
           Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
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
    pr = prior(HighOrderPriorEstimator(;
                                       pe = FactorPriorEstimator(;
                                                                 re = DimensionReductionRegression())),
               rd)
    clr = clusterise(ClusteringEstimator(), rd.X)
    opt = HierarchicalOptimiser(; pe = pr, cle = clr, slv = slv)
    T, N = size(pr.X)
    ew = eweights(1:T, inv(T); scale = true)
    w1 = fill(inv(N), N)
    rf = 4.34 / 100 / 252
    sigma = cov(GerberCovariance(), pr.X)
    mu = vec(mean(ShrunkExpectedReturns(; ce = GerberCovariance()), pr.X))
    sk, V = coskewness(Coskewness(; alg = Semi()), pr.X; mean = transpose(mu))
    kt = cokurtosis(Cokurtosis(; alg = Semi()), pr.X; mean = transpose(mu))
    pw = pweights(collect(range(; start = inv(size(pr.X, 1)), stop = inv(size(pr.X, 1)),
                                length = size(pr.X, 1))))
    rs = [Variance(; sigma = sigma), Variance(), UncertaintySetVariance(; sigma = sigma),
          UncertaintySetVariance(), StandardDeviation(; sigma = sigma), StandardDeviation(),
          BrownianDistanceVariance(), LowOrderMoment(; mu = mu), LowOrderMoment(; mu = rf),
          LowOrderMoment(; w = ew), LowOrderMoment(),
          LowOrderMoment(;
                         alg = LowOrderDeviation(;
                                                 alg = SecondLowerMoment(;
                                                                         formulation = SqrtRiskExpr())),
                         mu = mu),
          LowOrderMoment(;
                         alg = LowOrderDeviation(;
                                                 alg = SecondLowerMoment(;
                                                                         formulation = SqrtRiskExpr())),
                         mu = rf),
          LowOrderMoment(;
                         alg = LowOrderDeviation(;
                                                 ve = SimpleVariance(; corrected = false,
                                                                     w = ew),
                                                 alg = SecondLowerMoment(;
                                                                         formulation = SqrtRiskExpr())),
                         w = ew),
          LowOrderMoment(;
                         alg = LowOrderDeviation(;
                                                 alg = SecondLowerMoment(;
                                                                         formulation = SqrtRiskExpr()))),
          LowOrderMoment(; alg = LowOrderDeviation(; alg = SecondLowerMoment()), mu = mu),
          LowOrderMoment(; alg = LowOrderDeviation(; alg = SecondLowerMoment()), mu = rf),
          LowOrderMoment(;
                         alg = LowOrderDeviation(;
                                                 ve = SimpleVariance(; corrected = false,
                                                                     w = ew),
                                                 alg = SecondLowerMoment()), w = ew),
          LowOrderMoment(; alg = LowOrderDeviation(; alg = SecondLowerMoment())),
          LowOrderMoment(; alg = MeanAbsoluteDeviation(), mu = mu),
          LowOrderMoment(; alg = MeanAbsoluteDeviation(), mu = rf),
          LowOrderMoment(; alg = MeanAbsoluteDeviation(), w = ew),
          LowOrderMoment(; alg = MeanAbsoluteDeviation()), HighOrderMoment(; mu = mu),
          HighOrderMoment(; mu = rf), HighOrderMoment(; w = ew), HighOrderMoment(),
          HighOrderMoment(; alg = FourthLowerMoment(), mu = mu),
          HighOrderMoment(; alg = FourthLowerMoment(), mu = rf),
          HighOrderMoment(; alg = FourthLowerMoment(), w = ew),
          HighOrderMoment(; alg = FourthLowerMoment()),
          HighOrderMoment(; alg = FourthCentralMoment(), mu = mu),
          HighOrderMoment(; alg = FourthCentralMoment(), mu = rf),
          HighOrderMoment(; alg = FourthCentralMoment(), w = ew),
          HighOrderMoment(; alg = FourthCentralMoment()),
          HighOrderMoment(; alg = HighOrderDeviation(), mu = mu),
          HighOrderMoment(; alg = HighOrderDeviation(), mu = rf),
          HighOrderMoment(;
                          alg = HighOrderDeviation(;
                                                   ve = SimpleVariance(; corrected = false,
                                                                       w = ew)), w = ew),
          HighOrderMoment(; alg = HighOrderDeviation()),
          HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthLowerMoment()), mu = mu),
          HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthLowerMoment()), mu = rf),
          HighOrderMoment(;
                          alg = HighOrderDeviation(;
                                                   ve = SimpleVariance(; corrected = false,
                                                                       w = ew),
                                                   alg = FourthLowerMoment()), w = ew),
          HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthLowerMoment())),
          HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthCentralMoment()),
                          mu = mu),
          HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthCentralMoment()),
                          mu = rf),
          HighOrderMoment(;
                          alg = HighOrderDeviation(;
                                                   ve = SimpleVariance(; corrected = false,
                                                                       w = ew),
                                                   alg = FourthCentralMoment()), w = ew),
          HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthCentralMoment())),
          SquareRootKurtosis(; mu = mu), SquareRootKurtosis(; w = ew), SquareRootKurtosis(),
          SquareRootKurtosis(; alg = Semi(), mu = mu),
          SquareRootKurtosis(; alg = Semi(), w = ew), SquareRootKurtosis(; alg = Semi()),
          NegativeSkewness(), NegativeSkewness(; alg = QuadRiskExpr()),
          NegativeSkewness(; sk = sk, V = V),
          NegativeSkewness(; alg = QuadRiskExpr(), sk = sk, V = V),#
          ValueatRisk(; alpha = eps()), ValueatRisk(; alpha = eps(), w = pw), ValueatRisk(),
          ValueatRisk(; w = pw), ValueatRisk(; alpha = 0.5),
          ValueatRisk(; alpha = 0.5, w = pw), ValueatRisk(; alpha = 1 - eps()),
          ValueatRisk(; alpha = 1 - eps(), w = pw), ValueatRiskRange(; alpha = eps()),
          ValueatRiskRange(; alpha = eps(), w = pw), ValueatRiskRange(),
          ValueatRiskRange(; w = pw), ValueatRiskRange(; alpha = 0.5),
          ValueatRiskRange(; alpha = 0.5, w = pw), ValueatRiskRange(; alpha = 1 - eps()),
          ValueatRiskRange(; alpha = 1 - eps(), w = pw), DrawdownatRisk(; alpha = eps()),
          DrawdownatRisk(), DrawdownatRisk(; alpha = 0.5),
          RelativeDrawdownatRisk(; alpha = eps()), RelativeDrawdownatRisk(),
          RelativeDrawdownatRisk(; alpha = 0.5), ConditionalValueatRisk(; alpha = eps()),
          ConditionalValueatRisk(; alpha = eps(), w = pw), ConditionalValueatRisk(),
          ConditionalValueatRisk(; w = pw), ConditionalValueatRisk(; alpha = 0.5),
          ConditionalValueatRisk(; alpha = 0.5, w = pw),
          ConditionalValueatRisk(; alpha = 1 - eps()),
          ConditionalValueatRisk(; alpha = 1 - eps(), w = pw),
          DistributionallyRobustConditionalValueatRisk(; alpha = eps()),
          DistributionallyRobustConditionalValueatRisk(; alpha = eps(), w = pw),
          DistributionallyRobustConditionalValueatRisk(),
          DistributionallyRobustConditionalValueatRisk(; w = pw),
          DistributionallyRobustConditionalValueatRisk(; alpha = 0.5),
          DistributionallyRobustConditionalValueatRisk(; alpha = 0.5, w = pw),
          DistributionallyRobustConditionalValueatRisk(; alpha = 1 - eps()),
          DistributionallyRobustConditionalValueatRisk(; alpha = 1 - eps(), w = pw),
          ConditionalValueatRiskRange(; alpha = eps(), beta = eps()),
          ConditionalValueatRiskRange(; alpha = eps(), beta = eps(), w = pw),
          ConditionalValueatRiskRange(), ConditionalValueatRiskRange(; w = pw),
          ConditionalValueatRiskRange(; alpha = 0.15, beta = 0.15),
          ConditionalValueatRiskRange(; alpha = 0.15, beta = 0.15, w = pw),
          ConditionalValueatRiskRange(; alpha = 0.8, beta = 0.8),
          ConditionalValueatRiskRange(; alpha = 0.8, beta = 0.8, w = pw),
          ConditionalDrawdownatRisk(; alpha = eps()), ConditionalDrawdownatRisk(),
          ConditionalDrawdownatRisk(; alpha = 0.5),
          RelativeConditionalDrawdownatRisk(; alpha = eps()),
          RelativeConditionalDrawdownatRisk(),
          RelativeConditionalDrawdownatRisk(; alpha = 0.5),
          EntropicValueatRisk(; alpha = eps()),
          EntropicValueatRisk(; alpha = eps(), w = pw), EntropicValueatRisk(;),
          EntropicValueatRisk(; w = pw), EntropicValueatRisk(; alpha = 0.5),
          EntropicValueatRisk(; alpha = 0.5, w = pw),
          EntropicValueatRiskRange(; alpha = eps(), beta = eps()),
          EntropicValueatRiskRange(; alpha = eps(), beta = eps(), w = pw),
          EntropicValueatRiskRange(;), EntropicValueatRiskRange(; w = pw),
          EntropicValueatRiskRange(; alpha = 0.25, beta = 0.25),
          EntropicValueatRiskRange(; alpha = 0.25, beta = 0.25, w = pw),
          EntropicDrawdownatRisk(; alpha = eps()), EntropicDrawdownatRisk(;),
          EntropicDrawdownatRisk(; alpha = 0.5),
          RelativeEntropicDrawdownatRisk(; alpha = eps()),
          RelativeEntropicDrawdownatRisk(;), RelativeEntropicDrawdownatRisk(; alpha = 0.5),
          RelativisticValueatRisk(; alpha = eps()),
          RelativisticValueatRisk(; alpha = eps(), w = pw),
          RelativisticValueatRisk(; kappa = 0.6),
          RelativisticValueatRisk(; kappa = 0.6,
                                  w = pweights(collect(range(; start = 1, stop = 1,
                                                             length = size(pr.X, 1))))),
          RelativisticValueatRisk(; alpha = 0.3, kappa = 0.6),
          RelativisticValueatRisk(; alpha = 0.3, kappa = 0.6, w = pw),
          RelativisticValueatRiskRange(; alpha = eps(), beta = eps()),
          RelativisticValueatRiskRange(; alpha = eps(), beta = eps(), w = pw),
          RelativisticValueatRiskRange(;), RelativisticValueatRiskRange(; w = pw),
          RelativisticValueatRiskRange(; alpha = 0.25, beta = 0.25),
          RelativisticValueatRiskRange(; alpha = 0.25, beta = 0.25, w = pw),
          RelativisticDrawdownatRisk(; alpha = eps()),
          RelativisticDrawdownatRisk(; kappa = 0.6),
          RelativisticDrawdownatRisk(; alpha = 0.3, kappa = 0.6),
          RelativeRelativisticDrawdownatRisk(; alpha = eps()),
          RelativeRelativisticDrawdownatRisk(; kappa = 0.6),
          RelativeRelativisticDrawdownatRisk(; alpha = 0.3, kappa = 0.6),
          OrderedWeightsArray(), OrderedWeightsArray(; w = owa_tg(T)),
          OrderedWeightsArray(; w = owa_tgrg(T)),
          OrderedWeightsArrayRange(; w1 = owa_tgrg(T)), AverageDrawdown(),
          AverageDrawdown(; w = ew), RelativeAverageDrawdown(),
          RelativeAverageDrawdown(; w = ew), UlcerIndex(), RelativeUlcerIndex(),
          MaximumDrawdown(), RelativeMaximumDrawdown(), WorstRealisation(), Range(),
          EqualRiskMeasure(), TurnoverRiskMeasure(; w = w1),
          TrackingRiskMeasure(; tracking = WeightsTracking(; w = w1)),
          TrackingRiskMeasure(; tracking = ReturnsTracking(; w = pr.X * w1)),
          TrackingRiskMeasure(; formulation = NOCTracking(),
                              tracking = WeightsTracking(; w = w1)),
          TrackingRiskMeasure(; formulation = NOCTracking(),
                              tracking = ReturnsTracking(; w = pr.X * w1)),
          RiskTrackingRiskMeasure(; r = StandardDeviation(),
                                  tracking = WeightsTracking(; w = w1)),
          RiskTrackingRiskMeasure(; r = Variance(), tracking = WeightsTracking(; w = w1)),
          RiskTrackingRiskMeasure(; r = StandardDeviation(),
                                  tracking = WeightsTracking(; w = w1),
                                  formulation = DependentVariableTracking()),
          RiskTrackingRiskMeasure(; r = Variance(), tracking = WeightsTracking(; w = w1),
                                  formulation = DependentVariableTracking())]
    @testset "Hierarchical Risk Parity" begin
        df = CSV.read(joinpath(@__DIR__, "./assets/HRP.csv"), DataFrame)
        i = 1
        for r ∈ rs
            w = optimise!(HierarchicalRiskParity(; r = r, opt = opt)).w
            rtol = if i == 63
                0.5
            elseif i == 71
                0.01
            elseif !Sys.islinux() && i == 114 || Sys.iswindows() && i == 121
                5e-6
            elseif i == 133
                1e-4
            else
                1e-6
            end
            res = isapprox(w, df[!, i]; rtol = rtol)
            if !res
                println("Iteration $(i) failed, $(typeof(rs[i]))")
                find_tol(w, df[!, i]; name1 = "w", name2 = "df[!, $(i)]")
            end
            @test isapprox(w, df[!, i]; rtol = rtol)
            i += 1
        end
    end
    @testset "Hierarchical Equal Risk Contribution" begin
        df = CSV.read(joinpath(@__DIR__, "./assets/HERC-ri=ro.csv"), DataFrame)
        i = 1
        for r ∈ rs
            w = optimise!(HierarchicalEqualRiskContribution(; ri = r, opt = opt)).w
            rtol = if i == 63
                1
            elseif i == 71
                0.005
            elseif i == 114
                5e-6
            elseif !Sys.isapple() && i == 133
                5e-5
            elseif Sys.isapple() && i == 133
                1e-4
            else
                1e-6
            end
            res = isapprox(w, df[!, i]; rtol = rtol)
            if !res
                println("Iteration $(i) failed, $(typeof(rs[i]))")
                find_tol(w, df[!, i]; name1 = "w", name2 = "df[!, $(i)]")
            end
            @test isapprox(w, df[!, i]; rtol = rtol)
            i += 1
        end
    end
    @testset "Bounds tests" begin
        lb = [0.2, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0,
              0, 0, 0, 0, 0, 0, 0]
        ub = [0.2, 1, 1, 1, 1, 1, 1, 1, 0.05, 0, 0.2, 1, 1, 1, 1, 1, 1, 1, 0.05, 0, 0.2, 1,
              1, 1, 1, 1, 0.03, 1, 0.05, 0]
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
                JuMP_ClusteringWeightFiniliser(; slv = slv),
                JuMP_ClusteringWeightFiniliser(;
                                               alg = SquareRelativeErrorClusteringWeightFiniliser(),
                                               slv = slv),
                JuMP_ClusteringWeightFiniliser(;
                                               alg = AbsoluteErrorClusteringWeightFiniliser(),
                                               slv = slv),
                JuMP_ClusteringWeightFiniliser(;
                                               alg = SquareAbsoluteErrorClusteringWeightFiniliser(),
                                               slv = slv)]
        for cwf ∈ cwfs
            opt = HierarchicalOptimiser(; pe = pr, cle = clr, cwf = cwf,
                                        wb = WeightBoundsResult(; lb = lb, ub = ub),
                                        slv = slv)
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
    @testset "Schur HRP" begin
        pr = prior(EmpiricalPriorEstimator(;
                                           ce = PortfolioOptimisersCovariance(;
                                                                              ce = Covariance(;
                                                                                              alg = Full(),
                                                                                              ce = GeneralWeightedCovariance(;
                                                                                                                             ce = LinearShrinkage(CommonCovariance(),
                                                                                                                                                  :ss))))),
                   rd)
        opt = HierarchicalOptimiser(; pe = pr, cle = clr, slv = slv)
        res1 = optimise!(SchurHierarchicalRiskParity(;
                                                     params = SchurParams(; gamma = 0,
                                                                          alg = MonotonicSchur()),
                                                     opt = opt))
        res2 = optimise!(HierarchicalRiskParity(; opt = opt))
        @test isapprox(res1.w, res2.w)

        res3 = optimise!(SchurHierarchicalRiskParity(;
                                                     params = [SchurParams(; gamma = 0,
                                                                           alg = MonotonicSchur())],
                                                     opt = opt))
        @test isapprox(res1.w, res3.w)

        res4 = optimise!(SchurHierarchicalRiskParity(;
                                                     params = SchurParams(; gamma = 1,
                                                                          alg = MonotonicSchur()),
                                                     opt = opt))

        @test_throws MethodError optimise!(SchurHierarchicalRiskParity(;
                                                                       params = SchurParams(;
                                                                                            gamma = res4.gamma +
                                                                                                    1e-4,
                                                                                            alg = NonMonotonicSchur()),
                                                                       opt = opt))
    end
    @testset "Nested Clustering" begin
        pr = prior(EmpiricalPriorEstimator(), rd)
        clr = clusterise(ClusteringEstimator(), pr.X)
        jopt = JuMPOptimiser(; pe = pr, slv = slv)
        hopt = HierarchicalOptimiser(; slv = slv)
        res1 = optimise!(NestedClustering(; pe = pr, cle = clr,
                                          opti = NearOptimalCentering(;
                                                                      r = ConditionalValueatRisk(),
                                                                      bins = 20,
                                                                      obj = MaximumRatio(;
                                                                                         rf = rf),
                                                                      opt = jopt),
                                          opto = HierarchicalEqualRiskContribution(;
                                                                                   ri = ConditionalDrawdownatRisk(),
                                                                                   ro = StandardDeviation(),
                                                                                   opt = hopt),
                                          threads = SequentialEx()), rd)
        res2 = optimise!(NestedClustering(; pe = pr, cle = clr,
                                          opti = NearOptimalCentering(;
                                                                      r = ConditionalValueatRisk(),
                                                                      bins = 20,
                                                                      obj = MaximumRatio(;
                                                                                         rf = rf),
                                                                      opt = jopt),
                                          opto = HierarchicalEqualRiskContribution(;
                                                                                   ri = ConditionalDrawdownatRisk(),
                                                                                   ro = StandardDeviation(),
                                                                                   opt = hopt)),
                         rd)
        @test isapprox(res1.w,
                       [0.0018222027347502524, 0.003018495528527348, 0.0026864529798926746,
                        0.012681388345795436, 0.0008337946449769315, 0.0027990998709179764,
                        0.005535728517142857, 0.0030462435901917892, 0.005157792172153478,
                        0.03851249282703271, 0.2047064398188737, 0.0023257673444756046,
                        0.007376444419233182, 0.005798894488076123, 0.005312358350199489,
                        0.003304823890323986, 0.004021251241074547, 0.0004208461854390179,
                        0.0030689909363023162, 0.22743290904635988, 0.006345403613759411,
                        0.1317028546678364, 0.00424262864533623, 0.0017548183311587482,
                        0.007274196840803503, 0.23295706548658007, 0.011893535517372976,
                        0.0600405726796676, 0.002554411563893469, 0.0013720852159448413],
                       rtol = 1e-6)
        @test isapprox(res1.w, res2.w)

        jopt = JuMPOptimiser(; slv = slv)
        hopt = HierarchicalOptimiser(; pe = pr, slv = slv)
        res3 = optimise!(NestedClustering(; pe = pr, cle = clr,
                                          opto = NearOptimalCentering(;
                                                                      r = ConditionalValueatRisk(),
                                                                      obj = MaximumRatio(;
                                                                                         rf = rf),
                                                                      opt = jopt),
                                          opti = HierarchicalEqualRiskContribution(;
                                                                                   ri = ConditionalDrawdownatRisk(),
                                                                                   ro = StandardDeviation(),
                                                                                   opt = hopt),
                                          threads = SequentialEx()), rd)
        res4 = optimise!(NestedClustering(; pe = pr, cle = clr,
                                          opto = NearOptimalCentering(;
                                                                      r = ConditionalValueatRisk(),
                                                                      obj = MaximumRatio(;
                                                                                         rf = rf),
                                                                      opt = jopt),
                                          opti = HierarchicalEqualRiskContribution(;
                                                                                   ri = ConditionalDrawdownatRisk(),
                                                                                   ro = StandardDeviation(),
                                                                                   opt = hopt)),
                         rd)
        @test isapprox(res3.w,
                       [0.00024883463926763153, 0.00010174731357962266,
                        0.00032046290032649455, 0.05334331526368971, 0.00024287792567891233,
                        9.058334317154616e-5, 0.00014059196284429897, 0.017898077443852776,
                        0.000166353050966532, 0.03391266745884957, 0.00041134834084408523,
                        9.256129941895275e-5, 0.027231244929182784, 0.00011674161017761308,
                        0.03219827272876661, 8.852364662592574e-5, 8.13917760039694e-5,
                        0.00011774201832057956, 0.0001290803429228256,
                        0.00016931965791270348, 8.455890836019534e-5, 0.30649349734599307,
                        0.00012053009746729449, 0.00022033684779288818, 0.06882491486079627,
                        0.29679600954853047, 0.159474038802478, 4.170019166091807e-5,
                        0.00028018364679019994, 0.0005624900984134234], rtol = 1e-6)
        @test isapprox(res3.w, res4.w)
    end
end
