@testset "Risk measures" begin
    using PortfolioOptimisers, Test, DataFrames, TimeSeries, CSV, Clarabel, StatsBase,
          LinearAlgebra, Optim
    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end])
    pr = prior(HighOrderPriorEstimator(; pe = EmpiricalPrior()), rd)
    w = fill(inv(size(rd.X, 2)), size(rd.X, 2))
    wt = StatsBase.pweights(fill(inv(size(rd.X, 1)), size(rd.X, 1)))
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
    @testset "X at Risk" begin
        rs = [(ValueatRisk(; alpha = 1e-6), ValueatRisk(; alpha = 1e-6, w = wt)),
              (ValueatRisk(;), ValueatRisk(; w = wt)),
              (ValueatRisk(; alpha = 1 - 1e-6), ValueatRisk(; alpha = 1 - 1e-6, w = wt)),
              (ValueatRiskRange(; alpha = 1e-6, beta = 1e-6),
               ValueatRiskRange(; alpha = 1e-6, beta = 1e-6, w = wt)),
              (ValueatRiskRange(;), ValueatRiskRange(; w = wt)),
              (ValueatRiskRange(; alpha = 1 - 1e-6, beta = 1 - 1e-6),
               ValueatRiskRange(; alpha = 1 - 1e-6, beta = 1 - 1e-6, w = wt)),
              (ConditionalValueatRisk(; alpha = 1e-6),
               ConditionalValueatRisk(; alpha = 1e-6, w = wt)),
              (ConditionalValueatRisk(;), ConditionalValueatRisk(; w = wt)),
              (ConditionalValueatRisk(; alpha = 1 - 1e-6),
               ConditionalValueatRisk(; alpha = 1 - 1e-6, w = wt)),
              (ConditionalValueatRiskRange(; alpha = 1e-6, beta = 1e-6),
               ConditionalValueatRiskRange(; alpha = 1e-6, beta = 1e-6, w = wt)),
              (ConditionalValueatRiskRange(;), ConditionalValueatRiskRange(; w = wt)),
              (ConditionalValueatRiskRange(; alpha = 1 - 1e-6, beta = 1 - 1e-6),
               ConditionalValueatRiskRange(; alpha = 1 - 1e-6, beta = 1 - 1e-6, w = wt)),
              (EntropicValueatRisk(; alpha = 1e-6, slv = slv),
               EntropicValueatRisk(; alpha = 1e-6, w = wt, slv = slv)),
              (EntropicValueatRisk(; slv = slv), EntropicValueatRisk(; w = wt, slv = slv)),
              (EntropicValueatRisk(; alpha = 1 - 1e-6, slv = slv),
               EntropicValueatRisk(; alpha = 1 - 1e-6, w = wt, slv = slv)),
              (EntropicValueatRiskRange(; alpha = 1e-6, beta = 1e-6, slv = slv),
               EntropicValueatRiskRange(; alpha = 1e-6, beta = 1e-6, w = wt, slv = slv)),
              (EntropicValueatRiskRange(; slv = slv),
               EntropicValueatRiskRange(; w = wt, slv = slv)),
              (EntropicValueatRiskRange(; alpha = 1 - 1e-6, beta = 1 - 1e-6, slv = slv),
               EntropicValueatRiskRange(; alpha = 1 - 1e-6, beta = 1 - 1e-6, w = wt,
                                        slv = slv)),
              (RelativisticValueatRisk(; alpha = 1e-6, slv = slv),
               RelativisticValueatRisk(; alpha = 1e-6, w = wt, slv = slv)),
              (RelativisticValueatRisk(; slv = slv),
               RelativisticValueatRisk(; w = wt, slv = slv)),
              (RelativisticValueatRisk(; alpha = 1 - 1e-6, slv = slv),
               RelativisticValueatRisk(; alpha = 1 - 1e-6, w = wt, slv = slv)),
              (RelativisticValueatRiskRange(; alpha = 1e-6, beta = 1e-6, slv = slv),
               RelativisticValueatRiskRange(; alpha = 1e-6, beta = 1e-6, w = wt, slv = slv)),
              (RelativisticValueatRiskRange(; slv = slv),
               RelativisticValueatRiskRange(; w = wt, slv = slv)),
              (RelativisticValueatRiskRange(; alpha = 1 - 1e-6, beta = 1 - 1e-6, slv = slv),
               RelativisticValueatRiskRange(; alpha = 1 - 1e-6, beta = 1 - 1e-6, w = wt,
                                            slv = slv)),
              (DrawdownatRisk(; alpha = 1e-6), DrawdownatRisk(; alpha = 1e-6, w = wt)),
              (DrawdownatRisk(;), DrawdownatRisk(; w = wt)),
              (DrawdownatRisk(; alpha = 1 - 1e-6),
               DrawdownatRisk(; alpha = 1 - 1e-6, w = wt)),
              (ConditionalDrawdownatRisk(; alpha = 1e-6),
               ConditionalDrawdownatRisk(; alpha = 1e-6, w = wt)),
              (ConditionalDrawdownatRisk(;), ConditionalDrawdownatRisk(; w = wt)),
              (ConditionalDrawdownatRisk(; alpha = 1 - 1e-6),
               ConditionalDrawdownatRisk(; alpha = 1 - 1e-6, w = wt)),
              (EntropicDrawdownatRisk(; alpha = 1e-6, slv = slv),
               EntropicDrawdownatRisk(; alpha = 1e-6, w = wt, slv = slv)),
              (EntropicDrawdownatRisk(; slv = slv),
               EntropicDrawdownatRisk(; w = wt, slv = slv)),
              (EntropicDrawdownatRisk(; alpha = 1 - 1e-6, slv = slv),
               EntropicDrawdownatRisk(; alpha = 1 - 1e-6, w = wt, slv = slv)),
              (RelativisticDrawdownatRisk(; alpha = 1e-6, slv = slv),
               RelativisticDrawdownatRisk(; alpha = 1e-6, w = wt, slv = slv)),
              (RelativisticDrawdownatRisk(; slv = slv),
               RelativisticDrawdownatRisk(; w = wt, slv = slv)),
              (RelativisticDrawdownatRisk(; alpha = 1 - 1e-6, slv = slv),
               RelativisticDrawdownatRisk(; alpha = 1 - 1e-6, w = wt, slv = slv)),
              (RelativeDrawdownatRisk(; alpha = 1e-6),
               RelativeDrawdownatRisk(; alpha = 1e-6, w = wt)),
              (RelativeDrawdownatRisk(;), RelativeDrawdownatRisk(; w = wt)),
              (RelativeDrawdownatRisk(; alpha = 1 - 1e-6),
               RelativeDrawdownatRisk(; alpha = 1 - 1e-6, w = wt)),
              (RelativeConditionalDrawdownatRisk(; alpha = 1e-6),
               RelativeConditionalDrawdownatRisk(; alpha = 1e-6, w = wt)),
              (RelativeConditionalDrawdownatRisk(;),
               RelativeConditionalDrawdownatRisk(; w = wt)),
              (RelativeConditionalDrawdownatRisk(; alpha = 1 - 1e-6),
               RelativeConditionalDrawdownatRisk(; alpha = 1 - 1e-6, w = wt)),
              (RelativeEntropicDrawdownatRisk(; alpha = 1e-6, slv = slv),
               RelativeEntropicDrawdownatRisk(; alpha = 1e-6, w = wt, slv = slv)),
              (RelativeEntropicDrawdownatRisk(; slv = slv),
               RelativeEntropicDrawdownatRisk(; w = wt, slv = slv)),
              (RelativeEntropicDrawdownatRisk(; alpha = 1 - 1e-6, slv = slv),
               RelativeEntropicDrawdownatRisk(; alpha = 1 - 1e-6, w = wt, slv = slv)),
              (RelativeRelativisticDrawdownatRisk(; alpha = 1e-6, slv = slv),
               RelativeRelativisticDrawdownatRisk(; alpha = 1e-6, w = wt, slv = slv)),
              (RelativeRelativisticDrawdownatRisk(; slv = slv),
               RelativeRelativisticDrawdownatRisk(; w = wt, slv = slv)),
              (RelativeRelativisticDrawdownatRisk(; alpha = 1 - 1e-6, slv = slv),
               RelativeRelativisticDrawdownatRisk(; alpha = 1 - 1e-6, w = wt, slv = slv)),
              (PowerNormValueatRisk(; slv = slv, alpha = 1e-6),
               PowerNormValueatRisk(; slv = slv, alpha = 1e-6, w = wt)),
              (PowerNormValueatRisk(; slv = slv),
               PowerNormValueatRisk(; slv = slv, w = wt)),
              (PowerNormValueatRisk(; slv = slv, alpha = 1 - 1e-6),
               PowerNormValueatRisk(; slv = slv, alpha = 1 - 1e-6, w = wt)),
              (PowerNormValueatRiskRange(; slv = slv, alpha = 1e-6, beta = 1e-6),
               PowerNormValueatRiskRange(; slv = slv, alpha = 1e-6, beta = 1e-6, w = wt)),
              (PowerNormValueatRiskRange(; slv = slv),
               PowerNormValueatRiskRange(; slv = slv, w = wt)),
              (PowerNormValueatRiskRange(; slv = slv, alpha = 1 - 1e-6, beta = 1 - 1e-6),
               PowerNormValueatRiskRange(; slv = slv, alpha = 1 - 1e-6, beta = 1 - 1e-6,
                                         w = wt)),
              (PowerNormDrawdownatRisk(; slv = slv, alpha = 1e-6),
               PowerNormDrawdownatRisk(; slv = slv, alpha = 1e-6, w = wt)),
              (PowerNormDrawdownatRisk(; slv = slv),
               PowerNormDrawdownatRisk(; slv = slv, w = wt)),
              (PowerNormDrawdownatRisk(; slv = slv, alpha = 1 - 1e-6),
               PowerNormDrawdownatRisk(; slv = slv, alpha = 1 - 1e-6, w = wt)),
              (RelativePowerNormDrawdownatRisk(; slv = slv, alpha = 1e-6),
               RelativePowerNormDrawdownatRisk(; slv = slv, alpha = 1e-6, w = wt)),
              (RelativePowerNormDrawdownatRisk(; slv = slv),
               RelativePowerNormDrawdownatRisk(; slv = slv, w = wt)),
              (RelativePowerNormDrawdownatRisk(; slv = slv, alpha = 1 - 1e-6),
               RelativePowerNormDrawdownatRisk(; slv = slv, alpha = 1 - 1e-6, w = wt))]
        df = CSV.read(joinpath(@__DIR__, "./assets/XatRisk.csv.gz"), DataFrame)
        for (i, r) in enumerate(rs)
            r1 = expected_risk(r[1], w, rd.X)
            r2 = expected_risk(r[2], w, rd.X)
            rtol = if i == 15
                5e-2
            elseif i in (18, 21)
                1e-1
            elseif i in (16, 35, 47, 51)
                5e-2
            elseif i in (20, 23, 24, 54)
                0.25
            elseif i in (33, 55)
                5e-5
            elseif i == 60
                5e-4
            elseif i == 57
                5e-4
            elseif i == 36
                5e-3
            elseif i in (49, 58)
                5e-6
            elseif i in (45, 48)
                1e-3
            else
                1e-6
            end
            success = isapprox(r1, r2; rtol = rtol)
            if !success
                println("Iteration $i fails")
                find_tol(r1, r2)
            end
            @test success

            success = isapprox(df[i, 1], r1; rtol = rtol)
            if !success
                println("Iteration $i r1 fails")
                find_tol(r1, df[i, 1])
            end
            @test success

            success = isapprox(df[i, 2], r2; rtol = rtol)
            if !success
                println("Iteration $i r2 fails")
                find_tol(r2, df[i, 2])
            end
            @test success
        end
    end
    @testset "Expected risk" begin
        r1 = factory(NegativeSkewness(), pr)
        r2 = factory(NegativeSkewness(; alg = SquaredSOCRiskExpr()), pr)
        r3 = factory(NegativeSkewness(; alg = QuadRiskExpr()), pr)
        @test isapprox(expected_risk(r1, w, rd.X), sqrt(expected_risk(r2, w, rd.X)))
        @test isapprox(expected_risk(r3, w, rd.X), expected_risk(r2, w, rd.X))
        @test isapprox(expected_risk(Kurtosis(; alg1 = SemiMoment()), w, rd.X),
                       0.0002291596657404573)
        @test isapprox(expected_risk(Kurtosis(;), w, rd.X),
                       expected_risk(Kurtosis(; mu = pr.mu), w, rd.X))
        @test isapprox(expected_risk(Kurtosis(; mu = LinearAlgebra.dot(w, pr.mu)), w, rd.X),
                       expected_risk(Kurtosis(;), w, rd.X))
        @test isapprox(expected_risk(Kurtosis(; w = wt), w, rd.X),
                       expected_risk(Kurtosis(;), w, rd.X))
        @test isapprox(expected_risk(LowOrderMoment(;
                                                    alg = SecondMoment(;
                                                                       alg1 = SemiMoment(),
                                                                       alg2 = SOCRiskExpr())),
                                     w, rd.X), 0.009123864007588172)
        @test isapprox(expected_risk(LowOrderMoment(; mu = LinearAlgebra.dot(w, pr.mu),
                                                    alg = SecondMoment(;
                                                                       alg1 = SemiMoment(),
                                                                       alg2 = SOCRiskExpr())),
                                     w, rd.X),
                       sqrt(expected_risk(LowOrderMoment(;
                                                         alg = SecondMoment(;
                                                                            alg1 = SemiMoment(),
                                                                            alg2 = QuadRiskExpr())),
                                          w, rd.X)))
        @test isapprox(expected_risk(LowOrderMoment(;
                                                    alg = SecondMoment(;
                                                                       alg2 = SOCRiskExpr())),
                                     w, rd.X), 0.012828296955991162)
        @test isapprox(expected_risk(LowOrderMoment(; mu = LinearAlgebra.dot(w, pr.mu),
                                                    alg = SecondMoment(;
                                                                       alg2 = SOCRiskExpr())),
                                     w, rd.X),
                       sqrt(expected_risk(LowOrderMoment(;
                                                         alg = SecondMoment(;
                                                                            alg2 = QuadRiskExpr())),
                                          w, rd.X)))
        @test isapprox(expected_risk(LowOrderMoment(; alg = MeanAbsoluteDeviation()), w,
                                     rd.X), 0.009807328313217291)
        @test isapprox(expected_risk(LowOrderMoment(; alg = MeanAbsoluteDeviation()), w,
                                     rd.X),
                       expected_risk(LowOrderMoment(; w = wt,
                                                    alg = MeanAbsoluteDeviation()), w,
                                     rd.X))
        @test isapprox(expected_risk(HighOrderMoment(;
                                                     alg = FourthMoment(;
                                                                        alg = SemiMoment())),
                                     w, rd.X), 5.251415240227812e-8)
        @test isapprox(expected_risk(HighOrderMoment(; mu = LinearAlgebra.dot(w, pr.mu),
                                                     alg = FourthMoment(;
                                                                        alg = SemiMoment())),
                                     w, rd.X),
                       expected_risk(HighOrderMoment(;
                                                     alg = FourthMoment(;
                                                                        alg = SemiMoment())),
                                     w, rd.X))

        @test isapprox(expected_risk(HighOrderMoment(; alg = FourthMoment()), w, rd.X),
                       9.793810102468416e-8)
        @test isapprox(expected_risk(HighOrderMoment(; mu = LinearAlgebra.dot(w, pr.mu),
                                                     alg = FourthMoment()), w, rd.X),
                       expected_risk(HighOrderMoment(; alg = FourthMoment()), w, rd.X))
        @test isapprox(expected_risk(HighOrderMoment(;
                                                     alg = StandardisedHighOrderMoment(;
                                                                                       alg = ThirdLowerMoment())),
                                     w, rd.X), 2.4944191180382487)
        @test isapprox(expected_risk(HighOrderMoment(; mu = LinearAlgebra.dot(w, pr.mu),
                                                     alg = StandardisedHighOrderMoment(;
                                                                                       alg = ThirdLowerMoment())),
                                     w, rd.X),
                       expected_risk(HighOrderMoment(;
                                                     alg = StandardisedHighOrderMoment(;
                                                                                       alg = ThirdLowerMoment())),
                                     w, rd.X))
        @test isapprox(expected_risk(HighOrderMoment(;
                                                     alg = StandardisedHighOrderMoment(;
                                                                                       alg = FourthMoment(;
                                                                                                          alg = SemiMoment()))),
                                     w, rd.X), 7.5781142136319515)
        @test isapprox(expected_risk(HighOrderMoment(; mu = LinearAlgebra.dot(w, pr.mu),
                                                     alg = StandardisedHighOrderMoment(;
                                                                                       alg = FourthMoment(;
                                                                                                          alg = SemiMoment()))),
                                     w, rd.X),
                       expected_risk(HighOrderMoment(;
                                                     alg = StandardisedHighOrderMoment(;
                                                                                       alg = FourthMoment(;
                                                                                                          alg = SemiMoment()))),
                                     w, rd.X))
        @test isapprox(expected_risk(HighOrderMoment(;
                                                     alg = StandardisedHighOrderMoment(;
                                                                                       alg = FourthMoment())),
                                     w, rd.X), 3.616393337050389)
        @test isapprox(expected_risk(HighOrderMoment(; mu = LinearAlgebra.dot(w, pr.mu),
                                                     alg = StandardisedHighOrderMoment(;
                                                                                       alg = FourthMoment())),
                                     w, rd.X),
                       expected_risk(HighOrderMoment(;
                                                     alg = StandardisedHighOrderMoment(;
                                                                                       alg = FourthMoment())),
                                     w, rd.X))
        @test isapprox(expected_risk(AverageDrawdown(), w, rd.X), 0.048143525862128035)
        @test isapprox(expected_risk(AverageDrawdown(), w, rd.X),
                       expected_risk(AverageDrawdown(; w = wt), w, rd.X))
        @test isapprox(expected_risk(RelativeAverageDrawdown(), w, rd.X),
                       0.05118499953858111)
        @test isapprox(expected_risk(RelativeAverageDrawdown(), w, rd.X),
                       expected_risk(RelativeAverageDrawdown(; w = wt), w, rd.X))
        @test isapprox(expected_risk(RelativeUlcerIndex(), w, rd.X), 0.06369337923112198)
        @test isapprox(expected_risk(RelativeMaximumDrawdown(), w, rd.X),
                       0.14712227931904298)
        @test isapprox(expected_risk(BrownianDistanceVariance(), w, rd.X),
                       0.0005291680154419391)

        @test isapprox(expected_risk(MedianAbsoluteDeviation(), w, rd.X),
                       0.011730101952145106)
        @test isapprox(expected_risk(MedianAbsoluteDeviation(; w = wt), w, rd.X),
                       0.011730101952145106)
        @test isapprox(expected_risk(MedianAbsoluteDeviation(; mu = MeanCentering()), w,
                                     rd.X), 0.011649020215754542)
        @test isapprox(expected_risk(MedianAbsoluteDeviation(; mu = MeanCentering(),
                                                             w = wt), w, rd.X),
                       0.011649020215754542)
        @test isapprox(expected_risk(MedianAbsoluteDeviation(; mu = zeros(size(pr.X, 2)),
                                                             w = wt), w, rd.X),
                       0.011807455957080073)
        @test isapprox(expected_risk(MedianAbsoluteDeviation(; mu = 0, w = wt), w, rd.X),
                       0.011807455957080073)
        @test isapprox(expected_risk(factory(ExpectedReturn(), pr), w, pr),
                       0.00014724061887437024)
        @test factory(ExpectedReturn()) === ExpectedReturn()
        ucs = DeltaUncertaintySet(;)
        r = factory(ExpectedReturn(), pr)
        @test r.rt.mu === pr.mu
        r = factory(ExpectedReturn(), ucs)
        @test r.rt.ucs === ucs
        r = factory(ExpectedReturn(), pr, ucs)
        @test r.rt.mu === pr.mu
        @test r.rt.ucs === ucs

        r = factory(ExpectedReturn(), ucs, pr)
        @test r.rt.mu === pr.mu
        @test r.rt.ucs === ucs

        @test isapprox(expected_risk(factory(ExpectedReturn(; rt = LogarithmicReturn()),
                                             pr), w, pr), 6.522750683623699e-5)

        mu_views = LinearConstraintEstimator(; val = "AAPL == 0.002")
        sets = UniverseSets(; dict = Dict("nx" => rd.nx))
        pr2 = prior(EntropyPoolingPrior(; sets = sets, mu_views = mu_views,
                                        opt = OptimEntropyPooling(;
                                                                  args = (Optim.Fminbox(;
                                                                                        mu0 = 1e-5),))),
                    rd)
        rf = 4.2 / 252 / 100
        Xret = rd.X * w

        r = factory(MeanReturn(; flag = true), pr2)
        @test r.w === pr2.w
        @test r.flag == true
        @test expected_risk(r, w, rd.X) == mean(log1p.(Xret), pr2.w)

        r = factory(MeanReturnRiskRatio(; rt = MeanReturn(; flag = true),
                                        rk = EntropicValueatRisk(), rf = rf), pr2, slv)
        @test r.rt.w === pr2.w
        @test r.rt.flag == true
        @test r.rk.w === pr2.w
        @test r.rk.slv === slv
        @test r.rf == rf
        @test expected_risk(r, w, rd.X) ==
              (mean(log1p.(Xret), pr2.w) - rf) /
              expected_risk(EntropicValueatRisk(; slv = slv, w = pr2.w), w, rd.X)

        r = factory(MeanReturn(), pr)
        @test r.w === pr.w
        @test r.flag == false
        @test expected_risk(r, w, rd.X) == mean(Xret)

        r = factory(MeanReturnRiskRatio(; rf = rf, rk = RelativisticValueatRisk()), pr, slv)
        @test r.rt.w === pr.w
        @test r.rt.flag == false
        @test r.rk.w === pr.w
        @test r.rk.slv === slv
        @test r.rf == rf
        @test expected_risk(r, w, rd.X) ==
              (mean(Xret) - rf) /
              expected_risk(RelativisticValueatRisk(; slv = slv), w, rd.X)

        # The rolling window is checked at the exported entry point, not left to the risk
        # kernel: a non-positive window indexed `X` out of bounds and surfaced as a bare
        # `BoundsError` from inside the measure, and a window longer than the sample returned
        # an empty vector that read as a legitimate result.
        T = size(rd.X, 1)
        rw = ConditionalValueatRisk()
        @test length(PortfolioOptimisers.rolling_window_measure(rw, w, rd.X, nothing, T)) ==
              1
        @test length(PortfolioOptimisers.rolling_window_measure(rw, w, rd.X, nothing, 20)) ==
              T - 19
        for bad in (0, -1, T + 1)
            @test_throws DomainError PortfolioOptimisers.rolling_window_measure(rw, w, rd.X,
                                                                                nothing,
                                                                                bad)
        end
    end
    @testset "Generic X at Risk Range" begin
        rs1=[GenericValueatRiskRange(; loss = ValueatRisk(), gain = ValueatRisk()),
             GenericValueatRiskRange(; loss = ConditionalValueatRisk(),
                                     gain = ConditionalValueatRisk()),
             GenericValueatRiskRange(; loss = EntropicValueatRisk(; slv = slv),
                                     gain = EntropicValueatRisk(; slv = slv)),
             GenericValueatRiskRange(; loss = RelativisticValueatRisk(; slv = slv),
                                     gain = RelativisticValueatRisk(; slv = slv)),
             GenericValueatRiskRange(; loss = WorstRealisation(),
                                     gain = WorstRealisation()),
             GenericValueatRiskRange(; loss = PowerNormValueatRisk(; slv = slv),
                                     gain = PowerNormValueatRisk(; slv = slv))]
        rs2=[ValueatRiskRange(), ConditionalValueatRiskRange(),
             EntropicValueatRiskRange(; slv = slv),
             RelativisticValueatRiskRange(; slv = slv), Range(),
             PowerNormValueatRiskRange(; slv = slv)]
        for (i, (r1, r2)) in enumerate(zip(rs1, rs2))
            res = isapprox(expected_risk(r1, w, rd.X), expected_risk(r2, w, rd.X))
            if !res
                println("Iteration $i fails")
                find_tol(expected_risk(r1, w, rd.X), expected_risk(r2, w, rd.X))
            end
        end
    end
    @testset "Weighted range functors do not mutate their inputs (#330)" begin
        # The weighted `*Range` functors used to `reverse!` **views** into the caller's
        # returns vector and into their own stored observation weights, so a measure reused
        # on a second vector returned a wrong number from the second call onward.
        xa = rd.X * w
        xb = rd.X[end:-1:1, :] * w
        @test xa != xb

        # `alpha`/`beta` are pinned rather than defaulted, and the pair matters. Against the
        # pre-fix code these assertions were checked at all 49 combinations of
        # (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)²: 42 return a wrong number and 7 do not,
        # because the scrambled weight order has to move the cumulative-weight crossing far
        # enough to land on a different observation. The **default** `0.05, 0.05` is one of
        # the 7 for `ValueatRiskRange`, so defaulting here would have tested nothing.
        # `0.01, 0.1` is wrong pre-fix for both types: 1.1% for `ValueatRiskRange`,
        # 2.4% for `ConditionalValueatRiskRange`.
        for ctor in (ValueatRiskRange, ConditionalValueatRiskRange)
            ow = StatsBase.pweights(range(0.5, 1.5; length = length(xa)))
            ow_before = collect(ow)
            xa_before = copy(xa)

            # A measure that has already been evaluated must agree with a fresh one.
            reused = ctor(; alpha = 0.01, beta = 0.1, w = ow)
            fresh = ctor(; alpha = 0.01, beta = 0.1,
                         w = StatsBase.pweights(range(0.5, 1.5; length = length(xa))))
            _ = reused(xa)

            @test xa == xa_before            # the caller's vector survives
            @test collect(ow) == ow_before   # the measure's own weights survive
            @test reused.w === ow            # and it really is the stored object

            # The assertion that catches the wrong number rather than only the corruption.
            @test reused(xb) == fresh(xb)
            @test reused(xa) == fresh(xa)
            @test xa == xa_before
            @test collect(ow) == ow_before
        end
    end
    @testset "A risk measure agrees between the model and the functor (#351)" begin
        # `VarianceSkewKurtosis`'s functor used to delegate to its children, so it reported
        # `Skewness`'s **standardised** third moment and, under the default `SOCRiskExpr`,
        # `Kurtosis`'s **square root** of the fourth. Its own `JuMP` model reads the raw
        # moments off the relaxation blocks, so the two disagreed in magnitude and in sign:
        # the functor returned about -0.0721 where the model computed about +1.45e-5.
        r = factory(VarianceSkewKurtosis(), pr)
        w2 = kron(w, w)
        mu2 = dot(w, pr.sigma, w)
        mu3 = dot(w, pr.sk, w2)
        mu4 = dot(w2, pr.kt, w2)
        @test isapprox(r(w, rd.X), mu2 - mu3 + mu4)

        # The scales are the model's per-child weights, applied the same way.
        r2 = factory(VarianceSkewKurtosis(;
                                          vr = Variance(;
                                                        settings = RiskMeasureSettings(;
                                                                                       scale = 2.0)),
                                          kt = Kurtosis(;
                                                        settings = RiskMeasureSettings(;
                                                                                       scale = 3.0))),
                     pr)
        @test isapprox(r2(w, rd.X), 2 * mu2 - mu3 + 3 * mu4)

        # A central moment does not move when `fees` shifts every observation by the same
        # per-period constant, and the model carries no fee term either.
        @test r(w, rd.X, Fees(; l = 0.001)) == r(w, rd.X)
    end
    @testset "The weighted even moment weights observations linearly (#351)" begin
        # `moment_risk` computed `norm(val .* r.w, 2p)`, which raises each observation
        # weight to the power `2p`. The `JuMP` model attains `(sum w_t d_t^(2p) / T_d)^(1/p)`,
        # a linear weighting, so the two disagreed by a factor of `T^(2p-1)` at uniform
        # weights. The suite missed it because every weighted case used uniform weights,
        # where the error is one constant factor and never moves an argmin.
        x = rd.X * w
        ow = StatsBase.pweights(range(0.5, 1.5; length = length(x)))
        for p in (2, 3), malg in (FullMoment(), SemiMoment())
            unweighted = LowOrderMoment(; mu = 0, alg = EvenMoment(; p = p, alg = malg))
            uniform = LowOrderMoment(; mu = 0, w = wt,
                                     alg = EvenMoment(; p = p, alg = malg))
            weighted = LowOrderMoment(; mu = 0, w = ow,
                                      alg = EvenMoment(; p = p, alg = malg))

            # Uniform observation weights must reproduce the unweighted value.
            @test isapprox(uniform(x), unweighted(x))

            # And a general weight vector must match the value the model attains. `ddof`
            # defaults to 0, so the effective sample size is `sum(ow)`.
            d = malg === FullMoment() ? x : min.(x, zero(eltype(x)))
            @test isapprox(weighted(x), (sum(ow .* d .^ (2p)) / sum(ow))^inv(p))
        end
    end
    @testset "The distribution Value-at-Risk agrees between the model and the functor (#351)" begin
        # `ValueatRisk`'s functor computed the empirical order statistic whatever `alg`
        # held, while the `JuMP` model under `DistributionValueatRisk` builds the
        # parametric quantile. `alg` selects the estimand, so on a 200 x 6 normal sample
        # the model reported 0.0059528 against the functor's 0.0063843, a gap of 7 %.
        # The functor now reads the same two moments the model reads, and a `MinimumRisk`
        # solve puts the two within 2e-9 of each other.
        r = factory(ValueatRisk(; alpha = 0.05, alg = DistributionValueatRisk()), pr)
        z = PortfolioOptimisers.compute_value_at_risk_z(r.alg.dist, r.alpha)
        sd = sqrt(dot(w, pr.sigma, w))
        @test isapprox(expected_risk(r, w, pr), -dot(pr.mu, w) + z * sd)

        # The weights, not the net return series, reach the parametric branch.
        @test PortfolioOptimisers.risk_input_kind(r) ==
              PortfolioOptimisers.WeightsReturnsFeesInput()

        # The empirical branch is untouched, and the two are different numbers.
        emp = expected_risk(ValueatRisk(; alpha = 0.05), w, pr)
        @test isapprox(emp, -partialsort(rd.X * w, ceil(Int, 0.05 * size(rd.X, 1))))
        @test !isapprox(emp, expected_risk(r, w, pr))

        # The range's two legs share one mean term, which cancels in their difference.
        rr = factory(ValueatRiskRange(; alpha = 0.05, beta = 0.05,
                                      alg = DistributionValueatRisk()), pr)
        z_h = PortfolioOptimisers.compute_value_at_risk_cz(rr.alg.dist, rr.beta)
        @test isapprox(expected_risk(rr, w, pr), (z - z_h) * sd)
    end
end
