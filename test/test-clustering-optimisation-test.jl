@safetestset "Clustering Optimisation" begin
    using PortfolioOptimisers, CSV, DataFrames, Test, StableRNGs, Random, Clarabel,
          StatsBase, TimeSeries
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
        X = TimeArray(CSV.File(joinpath(@__DIR__, "./assets/asset_prices.csv"));
                      timestamp = :timestamp)
        F = TimeArray(CSV.File(joinpath(@__DIR__, "./assets/factor_prices.csv"));
                      timestamp = :timestamp)
        rd = prices_to_returns(X, F)
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
                                      "tol_infeas_rel" => 1e-4,
                                      "reduced_tol_gap_abs" => 1e-4,
                                      "reduced_tol_gap_rel" => 1e-4,
                                      "reduced_tol_ktratio" => 1e-3,
                                      "reduced_tol_feas" => 1e-4,
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

        rs = [Variance(; sigma = sigma), Variance(),
              UncertaintySetVariance(; sigma = sigma), UncertaintySetVariance(),
              StandardDeviation(; sigma = sigma), StandardDeviation(),
              BrownianDistanceVariance(), LowOrderMoment(; mu = mu),
              LowOrderMoment(; mu = rf), LowOrderMoment(; w = ew), LowOrderMoment(),
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
                                                     ve = SimpleVariance(;
                                                                         corrected = false,
                                                                         w = ew),
                                                     alg = SecondLowerMoment(;
                                                                             formulation = SqrtRiskExpr())),
                             w = ew),
              LowOrderMoment(;
                             alg = LowOrderDeviation(;
                                                     alg = SecondLowerMoment(;
                                                                             formulation = SqrtRiskExpr()))),
              LowOrderMoment(; alg = LowOrderDeviation(; alg = SecondLowerMoment()),
                             mu = mu),
              LowOrderMoment(; alg = LowOrderDeviation(; alg = SecondLowerMoment()),
                             mu = rf),
              LowOrderMoment(;
                             alg = LowOrderDeviation(;
                                                     ve = SimpleVariance(;
                                                                         corrected = false,
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
                                                       ve = SimpleVariance(;
                                                                           corrected = false,
                                                                           w = ew)),
                              w = ew), HighOrderMoment(; alg = HighOrderDeviation()),
              HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthLowerMoment()),
                              mu = mu),
              HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthLowerMoment()),
                              mu = rf),
              HighOrderMoment(;
                              alg = HighOrderDeviation(;
                                                       ve = SimpleVariance(;
                                                                           corrected = false,
                                                                           w = ew),
                                                       alg = FourthLowerMoment()), w = ew),
              HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthLowerMoment())),
              HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthCentralMoment()),
                              mu = mu),
              HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthCentralMoment()),
                              mu = rf),
              HighOrderMoment(;
                              alg = HighOrderDeviation(;
                                                       ve = SimpleVariance(;
                                                                           corrected = false,
                                                                           w = ew),
                                                       alg = FourthCentralMoment()),
                              w = ew),
              HighOrderMoment(; alg = HighOrderDeviation(; alg = FourthCentralMoment())),
              SquareRootKurtosis(; mu = mu), SquareRootKurtosis(; w = ew),
              SquareRootKurtosis(), SquareRootKurtosis(; alg = Semi(), mu = mu),
              SquareRootKurtosis(; alg = Semi(), w = ew),
              SquareRootKurtosis(; alg = Semi()), NegativeSkewness(),
              NegativeSkewness(; alg = QuadRiskExpr()), NegativeSkewness(; sk = sk, V = V),
              NegativeSkewness(; alg = QuadRiskExpr(), sk = sk, V = V),#
              ValueatRisk(; alpha = eps()),
              ValueatRisk(; alpha = eps(),
                          w = pweights(collect(range(; start = inv(size(pr.X, 1)),
                                                     stop = inv(size(pr.X, 1)),
                                                     length = size(pr.X, 1))))),
              ValueatRisk(),
              ValueatRisk(;
                          w = pweights(collect(range(; start = inv(size(pr.X, 1)),
                                                     stop = inv(size(pr.X, 1)),
                                                     length = size(pr.X, 1))))),
              ValueatRisk(; alpha = 0.5),
              ValueatRisk(; alpha = 0.5,
                          w = pweights(collect(range(; start = inv(size(pr.X, 1)),
                                                     stop = inv(size(pr.X, 1)),
                                                     length = size(pr.X, 1))))),
              ValueatRisk(; alpha = 1 - eps()),
              ValueatRisk(; alpha = 1 - eps(),
                          w = pweights(collect(range(; start = inv(size(pr.X, 1)),
                                                     stop = inv(size(pr.X, 1)),
                                                     length = size(pr.X, 1))))),
              ValueatRiskRange(; alpha = eps()),
              ValueatRiskRange(; alpha = eps(),
                               w = pweights(collect(range(; start = inv(size(pr.X, 1)),
                                                          stop = inv(size(pr.X, 1)),
                                                          length = size(pr.X, 1))))),
              ValueatRiskRange(),
              ValueatRiskRange(;
                               w = pweights(collect(range(; start = inv(size(pr.X, 1)),
                                                          stop = inv(size(pr.X, 1)),
                                                          length = size(pr.X, 1))))),
              ValueatRiskRange(; alpha = 0.5),
              ValueatRiskRange(; alpha = 0.5,
                               w = pweights(collect(range(; start = inv(size(pr.X, 1)),
                                                          stop = inv(size(pr.X, 1)),
                                                          length = size(pr.X, 1))))),
              ValueatRiskRange(; alpha = 1 - eps()),
              ValueatRiskRange(; alpha = 1 - eps(),
                               w = pweights(collect(range(; start = inv(size(pr.X, 1)),
                                                          stop = inv(size(pr.X, 1)),
                                                          length = size(pr.X, 1))))),
              DrawdownatRisk(; alpha = eps()), DrawdownatRisk(),
              DrawdownatRisk(; alpha = 0.5), RelativeDrawdownatRisk(; alpha = eps()),
              RelativeDrawdownatRisk(), RelativeDrawdownatRisk(; alpha = 0.5),
              ConditionalValueatRisk(; alpha = eps()),
              ConditionalValueatRisk(; alpha = eps(),
                                     w = pweights(collect(range(;
                                                                start = inv(size(pr.X, 1)),
                                                                stop = inv(size(pr.X, 1)),
                                                                length = size(pr.X, 1))))),
              ConditionalValueatRisk(),
              ConditionalValueatRisk(;
                                     w = pweights(collect(range(;
                                                                start = inv(size(pr.X, 1)),
                                                                stop = inv(size(pr.X, 1)),
                                                                length = size(pr.X, 1))))),
              ConditionalValueatRisk(; alpha = 0.5),
              ConditionalValueatRisk(; alpha = 0.5,
                                     w = pweights(collect(range(;
                                                                start = inv(size(pr.X, 1)),
                                                                stop = inv(size(pr.X, 1)),
                                                                length = size(pr.X, 1))))),
              ConditionalValueatRisk(; alpha = 1 - eps()),
              ConditionalValueatRisk(; alpha = 1 - eps(),
                                     w = pweights(collect(range(;
                                                                start = inv(size(pr.X, 1)),
                                                                stop = inv(size(pr.X, 1)),
                                                                length = size(pr.X, 1))))),
              DistributionallyRobustConditionalValueatRisk(; alpha = eps()),
              DistributionallyRobustConditionalValueatRisk(; alpha = eps(),
                                                           w = pweights(collect(range(;
                                                                                      start = inv(size(pr.X,
                                                                                                       1)),
                                                                                      stop = inv(size(pr.X,
                                                                                                      1)),
                                                                                      length = size(pr.X,
                                                                                                    1))))),
              DistributionallyRobustConditionalValueatRisk(),
              DistributionallyRobustConditionalValueatRisk(;
                                                           w = pweights(collect(range(;
                                                                                      start = inv(size(pr.X,
                                                                                                       1)),
                                                                                      stop = inv(size(pr.X,
                                                                                                      1)),
                                                                                      length = size(pr.X,
                                                                                                    1))))),
              DistributionallyRobustConditionalValueatRisk(; alpha = 0.5),
              DistributionallyRobustConditionalValueatRisk(; alpha = 0.5,
                                                           w = pweights(collect(range(;
                                                                                      start = inv(size(pr.X,
                                                                                                       1)),
                                                                                      stop = inv(size(pr.X,
                                                                                                      1)),
                                                                                      length = size(pr.X,
                                                                                                    1))))),
              DistributionallyRobustConditionalValueatRisk(; alpha = 1 - eps()),
              DistributionallyRobustConditionalValueatRisk(; alpha = 1 - eps(),
                                                           w = pweights(collect(range(;
                                                                                      start = inv(size(pr.X,
                                                                                                       1)),
                                                                                      stop = inv(size(pr.X,
                                                                                                      1)),
                                                                                      length = size(pr.X,
                                                                                                    1))))),
              ConditionalValueatRiskRange(; alpha = eps(), beta = eps()),
              ConditionalValueatRiskRange(; alpha = eps(), beta = eps(),
                                          w = pweights(collect(range(;
                                                                     start = inv(size(pr.X,
                                                                                      1)),
                                                                     stop = inv(size(pr.X,
                                                                                     1)),
                                                                     length = size(pr.X, 1))))),
              ConditionalValueatRiskRange(),
              ConditionalValueatRiskRange(;
                                          w = pweights(collect(range(;
                                                                     start = inv(size(pr.X,
                                                                                      1)),
                                                                     stop = inv(size(pr.X,
                                                                                     1)),
                                                                     length = size(pr.X, 1))))),
              ConditionalValueatRiskRange(; alpha = 0.15, beta = 0.15),
              ConditionalValueatRiskRange(; alpha = 0.15, beta = 0.15,
                                          w = pweights(collect(range(;
                                                                     start = inv(size(pr.X,
                                                                                      1)),
                                                                     stop = inv(size(pr.X,
                                                                                     1)),
                                                                     length = size(pr.X, 1))))),
              ConditionalValueatRiskRange(; alpha = 0.8, beta = 0.8),
              ConditionalValueatRiskRange(; alpha = 0.8, beta = 0.8,
                                          w = pweights(collect(range(;
                                                                     start = inv(size(pr.X,
                                                                                      1)),
                                                                     stop = inv(size(pr.X,
                                                                                     1)),
                                                                     length = size(pr.X, 1))))),
              ConditionalDrawdownatRisk(; alpha = eps()), ConditionalDrawdownatRisk(),
              ConditionalDrawdownatRisk(; alpha = 0.5),
              RelativeConditionalDrawdownatRisk(; alpha = eps()),
              RelativeConditionalDrawdownatRisk(),
              RelativeConditionalDrawdownatRisk(; alpha = 0.5),
              EntropicValueatRisk(; alpha = eps()),
              EntropicValueatRisk(; alpha = eps(),
                                  w = pweights(collect(range(; start = inv(size(pr.X, 1)),
                                                             stop = inv(size(pr.X, 1)),
                                                             length = size(pr.X, 1))))),
              EntropicValueatRisk(;),
              EntropicValueatRisk(;
                                  w = pweights(collect(range(; start = inv(size(pr.X, 1)),
                                                             stop = inv(size(pr.X, 1)),
                                                             length = size(pr.X, 1))))),
              EntropicValueatRisk(; alpha = 0.5),
              EntropicValueatRisk(; alpha = 0.5,
                                  w = pweights(collect(range(; start = inv(size(pr.X, 1)),
                                                             stop = inv(size(pr.X, 1)),
                                                             length = size(pr.X, 1))))),
              EntropicValueatRiskRange(; alpha = eps(), beta = eps()),
              EntropicValueatRiskRange(; alpha = eps(), beta = eps(),
                                       w = pweights(collect(range(;
                                                                  start = inv(size(pr.X, 1)),
                                                                  stop = inv(size(pr.X, 1)),
                                                                  length = size(pr.X, 1))))),
              EntropicValueatRiskRange(;),
              EntropicValueatRiskRange(;
                                       w = pweights(collect(range(;
                                                                  start = inv(size(pr.X, 1)),
                                                                  stop = inv(size(pr.X, 1)),
                                                                  length = size(pr.X, 1))))),
              EntropicValueatRiskRange(; alpha = 0.25, beta = 0.25),
              EntropicValueatRiskRange(; alpha = 0.25, beta = 0.25,
                                       w = pweights(collect(range(;
                                                                  start = inv(size(pr.X, 1)),
                                                                  stop = inv(size(pr.X, 1)),
                                                                  length = size(pr.X, 1))))),
              EntropicDrawdownatRisk(; alpha = eps()), EntropicDrawdownatRisk(;),
              EntropicDrawdownatRisk(; alpha = 0.5),
              RelativeEntropicDrawdownatRisk(; alpha = eps()),
              RelativeEntropicDrawdownatRisk(;),
              RelativeEntropicDrawdownatRisk(; alpha = 0.5),
              RelativisticValueatRisk(; alpha = eps()),
              RelativisticValueatRisk(; alpha = eps(),
                                      w = pweights(collect(range(;
                                                                 start = inv(size(pr.X, 1)),
                                                                 stop = inv(size(pr.X, 1)),
                                                                 length = size(pr.X, 1))))),
              RelativisticValueatRisk(; kappa = 0.6),
              RelativisticValueatRisk(; kappa = 0.6,
                                      w = pweights(collect(range(; start = 1, stop = 1,
                                                                 length = size(pr.X, 1))))),
              RelativisticValueatRisk(; alpha = 0.3, kappa = 0.6),
              RelativisticValueatRisk(; alpha = 0.3, kappa = 0.6,
                                      w = pweights(collect(range(;
                                                                 start = inv(size(pr.X, 1)),
                                                                 stop = inv(size(pr.X, 1)),
                                                                 length = size(pr.X, 1))))),
              RelativisticValueatRiskRange(; alpha = eps(), beta = eps()),
              RelativisticValueatRiskRange(; alpha = eps(), beta = eps(),
                                           w = pweights(collect(range(;
                                                                      start = inv(size(pr.X,
                                                                                       1)),
                                                                      stop = inv(size(pr.X,
                                                                                      1)),
                                                                      length = size(pr.X,
                                                                                    1))))),
              RelativisticValueatRiskRange(;),
              RelativisticValueatRiskRange(;
                                           w = pweights(collect(range(;
                                                                      start = inv(size(pr.X,
                                                                                       1)),
                                                                      stop = inv(size(pr.X,
                                                                                      1)),
                                                                      length = size(pr.X,
                                                                                    1))))),
              RelativisticValueatRiskRange(; alpha = 0.25, beta = 0.25),
              RelativisticValueatRiskRange(; alpha = 0.25, beta = 0.25,
                                           w = pweights(collect(range(;
                                                                      start = inv(size(pr.X,
                                                                                       1)),
                                                                      stop = inv(size(pr.X,
                                                                                      1)),
                                                                      length = size(pr.X,
                                                                                    1))))),
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
              TrackingRiskMeasure(; tracking = ReturnsTracking(; w = pr.X * w1))]
        df = CSV.read(joinpath(@__DIR__, "./assets/HRP.csv"), DataFrame)
        for i ∈ eachindex(rs)
            w = optimise!(HierarchicalRiskParity(; r = rs[i], opt = opt)).w
            rtol = if i ∈ (112, 114, 120, 129)
                5e-6
            elseif i == 128
                1e-5
            elseif i ∈ (128, 130, 131, 136, 137, 138, 139)
                0.005
            elseif i == 133
                5e-5
            elseif i ∈ (134, 135)
                5e-4
            elseif i == 140
                1e-4
            else
                1e-6
            end
            res = isapprox(w, df[!, i]; rtol = rtol)
            if !res
                println("Iteration $(i) failed, $(typeof(rs[i]))")
                find_tol(w, df[!, i]; name1 = "w", name2 = "df[!, $(i)]")
            end
            @test res
        end
        # df = CSV.read(joinpath(@__DIR__, "./assets/HRP-vector-risk-measure.csv"), DataFrame)
        # for (i, (r, rc)) ∈ enumerate(zip(rs, circshift(rs, 17)))
        #     w = optimise!(HierarchicalRiskParity(; r = [r, rc], opt = opt)).w
        #     rtol = 1e-6
        #     res = isapprox(w, df[!, i]; rtol = rtol)
        #     if !res
        #         println("Iteration $(i) failed,\n$(typeof(r))\n$(typeof(rc))")
        #         find_tol(w, df[!, i]; name1 = "w", name2 = "df[!, $(i)]")
        #     end
        #     @test res
        # end
    end
    #=
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
    =#
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
