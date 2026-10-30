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
            elseif i == 33
                # The two entropic solves of EDaR at alpha = 1 - 1e-6 differ by 5.5e-5 on some
                # hosts.
                1e-4
            elseif i == 55
                5e-5
            elseif i == 60
                5e-4
            elseif i == 57
                5e-4
            elseif i == 36
                5e-3
            elseif i in (49, 52, 58)
                # The power norm measures at alpha = 1e-6 are the largest loss, which `PRM`
                # returns exactly. The pinned values come from a solver, 5e-6 away from it.
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
        # The values of the returns that the functor of `Kurtosis` gave before #1846 are on
        # the moment measures, and the functor of `Kurtosis` reads its matrix.
        kt4 = HighOrderMoment(; alg = FourthMoment())
        @test isapprox(expected_risk(LowOrderMoment(;
                                                    alg = EvenMoment(; alg = SemiMoment())),
                                     w, rd.X), 0.0002291596657404573)
        @test isapprox(expected_risk(kt4, w, rd.X),
                       expected_risk(HighOrderMoment(; mu = pr.mu, alg = FourthMoment()), w,
                                     rd.X))
        @test isapprox(expected_risk(HighOrderMoment(; mu = LinearAlgebra.dot(w, pr.mu),
                                                     alg = FourthMoment()), w, rd.X),
                       expected_risk(kt4, w, rd.X))
        @test isapprox(expected_risk(HighOrderMoment(; w = wt, alg = FourthMoment()), w,
                                     rd.X), expected_risk(kt4, w, rd.X))
        @test isapprox(expected_risk(factory(Kurtosis(), pr), w, rd.X)^2,
                       expected_risk(kt4, w, rd.X))
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
        sv = SimpleVariance(; me = nothing)
        @test isapprox(expected_risk(HighOrderMoment(; alg = ThirdLowerMoment(), ve = sv),
                                     w, rd.X), 2.4944191180382487)
        @test isapprox(expected_risk(HighOrderMoment(; mu = LinearAlgebra.dot(w, pr.mu),
                                                     alg = ThirdLowerMoment(), ve = sv), w,
                                     rd.X),
                       expected_risk(HighOrderMoment(; alg = ThirdLowerMoment(), ve = sv),
                                     w, rd.X))
        @test isapprox(expected_risk(HighOrderMoment(;
                                                     alg = FourthMoment(;
                                                                        alg = SemiMoment()),
                                                     ve = sv), w, rd.X), 7.5781142136319515)
        @test isapprox(expected_risk(HighOrderMoment(; mu = LinearAlgebra.dot(w, pr.mu),
                                                     alg = FourthMoment(;
                                                                        alg = SemiMoment()),
                                                     ve = sv), w, rd.X),
                       expected_risk(HighOrderMoment(;
                                                     alg = FourthMoment(;
                                                                        alg = SemiMoment()),
                                                     ve = sv), w, rd.X))
        @test isapprox(expected_risk(HighOrderMoment(; alg = FourthMoment(), ve = sv), w,
                                     rd.X), 3.616393337050389)
        @test isapprox(expected_risk(HighOrderMoment(; mu = LinearAlgebra.dot(w, pr.mu),
                                                     alg = FourthMoment(), ve = sv), w,
                                     rd.X),
                       expected_risk(HighOrderMoment(; alg = FourthMoment(), ve = sv), w,
                                     rd.X))
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
        # The generic range with the two tails of a specific range has the value of that
        # range. The loop computed this and never asserted it.
        for (i, (r1, r2)) in enumerate(zip(rs1, rs2))
            res = isapprox(expected_risk(r1, w, rd.X), expected_risk(r2, w, rd.X))
            if !res
                println("Iteration $i fails")
                find_tol(expected_risk(r1, w, rd.X), expected_risk(r2, w, rd.X))
            end
            @test res
        end
        # Two different tails: the range is the sum of the loss tail on the returns and the
        # gain tail on the negated returns.
        x = rd.X * w
        g = GenericValueatRiskRange(; loss = ConditionalValueatRisk(; alpha = 0.03),
                                    gain = PowerNormValueatRisk(; alpha = 0.2, p = 3.0,
                                                                slv = slv))
        @test isapprox(g(x),
                       ConditionalValueatRisk(; alpha = 0.03)(x) +
                       PowerNormValueatRisk(; alpha = 0.2, p = 3.0, slv = slv)(-x))
        # A tail that reads the weights is refused on a returns vector, at every length. With
        # as many observations as assets, the functor used to return a wrong value with no
        # error.
        gd = GenericValueatRiskRange(;
                                     loss = ValueatRisk(; alg = DistributionValueatRisk()),
                                     gain = ConditionalValueatRisk())
        @test_throws ArgumentError gd(x)
        @test_throws ArgumentError gd(x[1:length(w)])
        # #1821: each tail reads its own input. A `DistributionValueatRisk` tail reads `w`
        # for the loss and `-w` for the gain, and a returns tail reads `x` and `-x`.
        vdist = ValueatRisk(; alg = DistributionValueatRisk())
        gdd = GenericValueatRiskRange(; loss = vdist, gain = vdist)
        @test PortfolioOptimisers.risk_input_kind(gdd) ===
              PortfolioOptimisers.WeightsReturnsFeesInput()
        @test PortfolioOptimisers.risk_input_kind(gd) ===
              PortfolioOptimisers.WeightsReturnsFeesInput()
        @test PortfolioOptimisers.risk_input_kind(GenericValueatRiskRange(;
                                                                          loss = ConditionalValueatRisk(),
                                                                          gain = vdist)) ===
              PortfolioOptimisers.WeightsReturnsFeesInput()
        @test !PortfolioOptimisers.supports_precomputed_returns(gd)
        @test PortfolioOptimisers.supports_precomputed_returns(g)
        @test isapprox(expected_risk(gdd, w, pr),
                       expected_risk(ValueatRiskRange(; alg = DistributionValueatRisk()), w,
                                     pr))
        # As many observations as assets: the value is the same as with every observation,
        # because a `DistributionValueatRisk` tail reads only the moments of the prior.
        N = length(w)
        rN = factory(gdd, pr)
        @test isapprox(rN(w, rd.X[1:N, :]), expected_risk(gdd, w, pr))
        # A mixed range, with a fee. The gain tail of the returns measure reads the negated
        # net returns of `w`, not the net returns of `-w`.
        gm = factory(GenericValueatRiskRange(; loss = vdist,
                                             gain = ConditionalValueatRisk(; alpha = 0.1)),
                     pr)
        fees = Fees(; l = 0.001)
        xf = calc_net_returns(w, pr.X, fees)
        vd = factory(vdist, pr)
        @test isapprox(expected_risk(gm, w, pr, fees),
                       vd(w) + ConditionalValueatRisk(; alpha = 0.1)(-xf))
        @test isapprox(gm(w, pr.X, fees), expected_risk(gm, w, pr, fees))
        # The gain tail of a `DistributionValueatRisk` is `dot(mu, w) + z * sigma(w)` for a law
        # symmetric about zero. `test_09n` checks a `Normal` with a nonzero mean (#1822).
        gg = factory(GenericValueatRiskRange(; loss = ConditionalValueatRisk(),
                                             gain = vdist), pr)
        z = PortfolioOptimisers.compute_value_at_risk_z(vdist.alg.dist, vdist.alpha)
        @test isapprox(gg(w, pr.X),
                       ConditionalValueatRisk()(pr.X * w) +
                       dot(pr.mu, w) +
                       z * sqrt(dot(w, pr.sigma, w)))
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
    @testset "`factory` keeps a `Skewness`'s settings (#1293)" begin
        # `factory(::Skewness, ...)` rebuilt the measure without its `settings`, so after
        # `factory` the skewness term of a `VarianceSkewKurtosis` had scale 1 and no floor.
        # The model used the stated scale, so the result stored a measure the model did not
        # use, and its functor reported the skewness term at the wrong scale.
        st = MaxRiskMeasureSettings(; scale = 2.0, lb = 0.5)
        @test factory(Skewness(; settings = st), pr).settings === st
        @test factory(Skewness(; settings = st, sk = pr.sk), pr.pr).settings === st

        # `VarianceSkewKurtosis` turns off each child's `rke`, and keeps its `scale` and `lb`.
        r = factory(VarianceSkewKurtosis(; sk = Skewness(; settings = st)), pr)
        @test r.sk.settings === MaxRiskMeasureSettings(; rke = false, scale = 2.0, lb = 0.5)
        w2 = kron(w, w)
        @test isapprox(r(w, rd.X),
                       dot(w, pr.sigma, w) - 2 * dot(w, pr.sk, w2) + dot(w2, pr.kt, w2))
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
        @test isapprox(emp, -partialsort(rd.X * w, floor(Int, 0.05 * size(rd.X, 1)) + 1))
        @test !isapprox(emp, expected_risk(r, w, pr))

        # The range's two legs share one mean term, which cancels in their difference.
        rr = factory(ValueatRiskRange(; alpha = 0.05, beta = 0.05,
                                      alg = DistributionValueatRisk()), pr)
        z_h = PortfolioOptimisers.compute_value_at_risk_cz(rr.alg.dist, rr.beta)
        @test isapprox(expected_risk(rr, w, pr), (z - z_h) * sd)
    end
    @testset "MaxRiskMeasureSettings validates its lower bound (#402)" begin
        # A vector bound is a sweep axis, one solve per entry, so an empty vector is a
        # sweep of zero points. `MaxRiskMeasureSettings(; lb = Float64[])` constructed,
        # and the optimisation that read it returned an empty result: no weights, no
        # return code, no warning. The sibling `RiskMeasureSettings` refuses the same
        # shape at construction.
        @test_throws PortfolioOptimisers.IsEmptyError MaxRiskMeasureSettings(;
                                                                             lb = Float64[])
        @test_throws DomainError MaxRiskMeasureSettings(; lb = Inf)
        @test_throws DomainError MaxRiskMeasureSettings(; lb = -Inf)
        @test_throws DomainError MaxRiskMeasureSettings(; lb = NaN)
        @test_throws DomainError MaxRiskMeasureSettings(; lb = [Inf, NaN])

        # The guard is `assert_nonempty_finite_val`, not the sibling's
        # `assert_nonempty_nonneg_finite_val`, because a negative floor is meaningful for
        # a quantity the optimisation maximises.
        @test MaxRiskMeasureSettings(; lb = -1e-8).lb == -1e-8
        @test MaxRiskMeasureSettings(; lb = [-1e-8, 1e-8]).lb == [-1e-8, 1e-8]

        # `nothing` and a `Frontier` reach the `args...` fallback unchanged.
        @test isnothing(MaxRiskMeasureSettings().lb)
        @test isa(MaxRiskMeasureSettings(; lb = Frontier(; N = 3)).lb, Frontier)
    end
    @testset "The unweighted tail statistics read the k smallest (#629)" begin
        # `partialsort!(v, k)` places the value at index `k` in the position it holds in a
        # fully sorted vector, and it promises nothing about the rest of the vector. Julia
        # switches selection strategy above a size threshold, and above it `v[1:k - 1]` is
        # not the `k - 1` smallest values. Four sites read that prefix as if it were, so
        # each returned a number that is not the statistic once `k` clears the threshold.
        # At `alpha = 0.5` the conditional value at risk came out negative, which is below
        # the value at risk it averages past.
        #
        # The series has to be long enough to reach the threshold. On a vector of 1008 the
        # prefix first goes wrong at `k = 117`, which is `alpha = 0.116`; on a vector of
        # 252 no `k` goes wrong at all, so the 252 rows the rest of this file reads would
        # have proved nothing. The levels below straddle 0.116 on purpose, and 0.05 is the
        # control that read correctly even before the fix.
        #
        # The weighted method of each measure sorts with `sortperm` and walks the
        # cumulative weights, so it rests on no prefix. Under uniform weights the two
        # methods price the same quantity, and that equality is what pins each site.
        rdl = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__,
                                                            "./assets/SP500.csv.gz"));
                                          timestamp = :Date))
        xl = rdl.X * fill(inv(size(rdl.X, 2)), size(rdl.X, 2))
        Tl = length(xl)
        @test Tl >= 500
        wl = StatsBase.pweights(fill(inv(Tl), Tl))

        # The definition: the mean of the losses past the value at risk, with the boundary
        # scenario carrying only the part of its mass the level leaves.
        function cvar_reference(x, alpha)
            xs = sort(x)
            aT = alpha * length(x)
            idx = ceil(Int, aT)
            return -(sum(view(xs, 1:(idx - 1))) + xs[idx] * (aT - (idx - 1))) / aT
        end

        for a in (0.05, 0.15, 0.2, 0.3, 0.5)
            # `ConditionalValueatRisk` and its distributionally robust twin are one method.
            cv = ConditionalValueatRisk(; alpha = a)(xl)
            @test isapprox(cv, cvar_reference(xl, a))
            @test isapprox(cv, ConditionalValueatRisk(; alpha = a, w = wl)(xl))
            # A conditional value at risk averages the losses past the value at risk, so it
            # can never sit below it.
            @test cv >= ValueatRisk(; alpha = a)(xl)
            @test isapprox(DistributionallyRobustConditionalValueatRisk(; alpha = a)(xl),
                           DistributionallyRobustConditionalValueatRisk(; alpha = a,
                                                                        w = wl)(xl))

            # Both ends of the range read a prefix, so the level sets `alpha` and `beta`.
            @test isapprox(ConditionalValueatRiskRange(; alpha = a, beta = a)(xl),
                           ConditionalValueatRiskRange(; alpha = a, beta = a, w = wl)(xl))
            @test isapprox(DistributionallyRobustConditionalValueatRiskRange(; alpha = a,
                                                                             beta = a)(xl),
                           DistributionallyRobustConditionalValueatRiskRange(; alpha = a,
                                                                             beta = a,
                                                                             w = wl)(xl))

            # `conditional_drawdown_at_risk` is the fourth site, and it reads a drawdown
            # series rather than the returns series.
            @test isapprox(ConditionalDrawdownatRisk(; alpha = a)(xl),
                           ConditionalDrawdownatRisk(; alpha = a, w = wl)(xl))
            @test isapprox(RelativeConditionalDrawdownatRisk(; alpha = a)(xl),
                           RelativeConditionalDrawdownatRisk(; alpha = a, w = wl)(xl))
        end
    end
    @testset "The realised-history rolling window measure reads a series (#770)" begin
        # One verb, two readings. The constant-weight reading re-scores one weight vector on
        # every window, so each number is a property of that vector. The realised-history
        # reading rolls a series that is already formed, so each number is a property of the
        # history that formed it. Over a constant-weight series the two price the same
        # quantity, which is what pins the new method to the old one.
        T = size(rd.X, 1)
        rw = ConditionalValueatRisk()
        ret = calc_net_returns(w, rd.X)

        # The definition, written out: each window is the measure on that window's rows.
        @test PortfolioOptimisers.rolling_window_measure(rw, ret, 20) ==
              [rw(view(ret, (t - 19):t)) for t in 20:T]
        # The two readings agree on a series no drift moved. They reduce the same rows in a
        # different order, so this is an `isapprox` rather than an equality.
        @test isapprox(PortfolioOptimisers.rolling_window_measure(rw, ret, 20),
                       PortfolioOptimisers.rolling_window_measure(rw, w, rd.X, nothing, 20))
        @test length(PortfolioOptimisers.rolling_window_measure(rw, ret, T)) == 1
        @test length(PortfolioOptimisers.rolling_window_measure(rw, ret, 20)) == T - 19

        # The window is refused at the same boundary as the constant-weight method, and the
        # message names the length of the series rather than the row count of a matrix.
        for bad in (0, -1, T + 1)
            @test_throws DomainError PortfolioOptimisers.rolling_window_measure(rw, ret,
                                                                                bad)
        end
        err = try
            PortfolioOptimisers.rolling_window_measure(rw, ret, T + 1)
        catch e
            e
        end
        @test occursin("observations in ret", sprint(showerror, err))

        # A vector of measures scalarises inside each window, so `sca` reaches the series
        # method exactly as it reaches the constant-weight one.
        rws = [ConditionalValueatRisk(), ValueatRisk()]
        @test isapprox(PortfolioOptimisers.rolling_window_measure(rws, ret, 20),
                       PortfolioOptimisers.rolling_window_measure(rws, w, rd.X, nothing,
                                                                  20))
        @test PortfolioOptimisers.rolling_window_measure(rws, ret, 20;
                                                         sca = MaxScalariser()) !=
              PortfolioOptimisers.rolling_window_measure(rws, ret, 20)

        # A measure that consumes weights is refused by name through the guard that
        # `expected_risk_from_returns` already carries. No new error type is added.
        @test !PortfolioOptimisers.supports_precomputed_returns(Variance())
        @test_throws ArgumentError PortfolioOptimisers.rolling_window_measure(Variance(),
                                                                              ret, 20)

        # A population of series is rolled one member at a time, which mirrors
        # `expected_risk_from_returns` on a vector of vectors.
        pop = PortfolioOptimisers.rolling_window_measure(rw, [ret, 2 * ret], 20)
        @test length(pop) == 2
        @test pop[1] == PortfolioOptimisers.rolling_window_measure(rw, ret, 20)
        @test pop[2] == PortfolioOptimisers.rolling_window_measure(rw, 2 * ret, 20)
    end
    @testset "The ending-weights rolling window measure reads a path (#769)" begin
        # The third reading of the one verb. The weight argument's type is the picker: a
        # vector is one target vector, and a matrix is a weight path whose row `t` holds the
        # weights carried through observation `t`. A window closing at row `t` is scored
        # against row `t` of the path — the weights held by the time the window closed.
        PO = PortfolioOptimisers
        T = size(rd.X, 1)
        rw = ConditionalValueatRisk()
        wdr = SelfFinancingDrift()
        U = PO.weight_path(wdr, w, rd.X)
        Uc = PO.weight_path(nothing, w, rd.X)

        # The definition, written out.
        @test PO.rolling_window_measure(rw, U, rd.X, nothing, 20) ==
              [rw(view(rd.X, (t - 19):t, :) * view(U, t, :)) for t in 20:T]

        # Every row of a constant path is the same vector, so the reading reproduces the
        # constant-weight method exactly rather than approximately.
        @test PO.rolling_window_measure(rw, Uc, rd.X, nothing, 20) ==
              PO.rolling_window_measure(rw, w, rd.X, nothing, 20)
        @test length(PO.rolling_window_measure(rw, U, rd.X, nothing, T)) == 1
        @test length(PO.rolling_window_measure(rw, U, rd.X, nothing, 20)) == T - 19

        # A drifted path moves the numbers, which is the whole point of the reading.
        @test PO.rolling_window_measure(rw, U, rd.X, nothing, 20) !=
              PO.rolling_window_measure(rw, w, rd.X, nothing, 20)

        # The window is refused at the same boundary as the two methods beside it.
        for bad in (0, -1, T + 1)
            @test_throws DomainError PO.rolling_window_measure(rw, U, rd.X, nothing, bad)
        end

        # A path shorter than the sample is refused by name, rather than surfacing as a
        # `BoundsError` from inside whichever measure `r` names.
        @test_throws DimensionMismatch PO.rolling_window_measure(rw, view(U, 1:(T - 1), :),
                                                                 rd.X, nothing, 20)
        err = try
            PO.rolling_window_measure(rw, view(U, 1:(T - 1), :), rd.X, nothing, 20)
        catch e
            e
        end
        @test occursin("size(w, 1) == size(X, 1)", sprint(showerror, err))

        # A vector of measures scalarises inside each window here too.
        rws = [ConditionalValueatRisk(), ValueatRisk()]
        @test PO.rolling_window_measure(rws, Uc, rd.X, nothing, 20) ==
              PO.rolling_window_measure(rws, w, rd.X, nothing, 20)
        @test PO.rolling_window_measure(rws, U, rd.X, nothing, 20; sca = MaxScalariser()) !=
              PO.rolling_window_measure(rws, U, rd.X, nothing, 20)
    end
    @testset "The power-norm measure is the largest loss under its condition (#1282)" begin
        rng = StableRNG(1282)
        r = PowerNormValueatRisk(; slv = slv)
        # alpha * T^(1/p) = 0.05 * sqrt(252) < 1: the value is the largest loss. The
        # docstring states the condition, and the measure does not warn.
        x = randn(rng, 252)
        v = @test_logs min_level = Logging.Warn r(x)
        @test isapprox(v, maximum(-x); rtol = 1e-5)
        # alpha * T^(1/p) = 0.05 * sqrt(401) > 1 with no tie: a smaller value.
        x = randn(rng, 401)
        @test r(x) < maximum(-x)
        # A tie of two observations at the largest loss doubles its probability, and
        # 0.05 * (800 / 2)^(1/2) = 1 is on the boundary.
        x = randn(rng, 800)
        x[1:2] .= minimum(x) - 1
        @test isapprox(r(x), maximum(-x); rtol = 1e-5)
        # A frequency weight of two on the largest loss acts as the same tie.
        x = randn(rng, 500)
        k = ones(Int, 500)
        k[argmin(x)] = 2
        rw = PowerNormValueatRisk(; slv = slv, w = fweights(k))
        @test isapprox(rw(x), maximum(-x); rtol = 1e-5)
        @test r(x) < maximum(-x)
        # An observation of zero weight carries no probability, so the second largest loss
        # sets the value.
        x = randn(rng, 252)
        q = ones(252)
        q[argmin(x)] = 0
        rw = PowerNormValueatRisk(; slv = slv, w = pweights(q))
        @test isapprox(rw(x), sort(-x; rev = true)[2]; rtol = 1e-5)
    end
end

# Verifies the `risk_input_kind` trait that replaced the `expected_risk` routing unions
# (ADR 0006). Two guards:
#   1. Equivalence — every measure that used to live in a routing union now declares the
#      matching kind, so the trait-based routing reproduces the old union-based routing.
#   2. Completeness — every concrete `AbstractBaseRiskMeasure` either declares a kind or is
#      a composite handled by an explicit `expected_risk` method, so a future measure added
#      without a kind fails here rather than at runtime.
using InteractiveUtils: InteractiveUtils
const PO = PortfolioOptimisers

# Snapshot of the OLD routing-union membership (pre-trait); the differential baseline.
# `ValueatRisk` and `ValueatRiskRange` split by formulation: `alg` selects the estimand, so
# the MIP branch scores a net return series and the parametric branch reads the prior's
# moments against the weights. Both branches are named, each in its own list. They are named
# by a CONCRETE type: `typeintersect` on two `alg`-bounded `UnionAll`s is a type with an
# uninhabited parameter rather than `Union{}`, so inference over a bound would report both
# kinds for either branch.
const _VAR_MIP = typeof(ValueatRisk(; alg = MIPValueatRisk()))
const _VAR_RANGE_MIP = typeof(ValueatRiskRange(; alg = MIPValueatRisk()))
const _VAR_DIST = typeof(ValueatRisk(; alg = DistributionValueatRisk()))
const _VAR_RANGE_DIST = typeof(ValueatRiskRange(; alg = DistributionValueatRisk()))
# `GenericValueatRiskRange` takes its kind from its tails (#1821). Two returns tails read the
# net returns. A `DistributionValueatRisk` tail on either side reads the weights.
const _GEN_RANGE_RET = typeof(GenericValueatRiskRange())
const _GEN_RANGE_DIST_LOSS = typeof(GenericValueatRiskRange(;
                                                            loss = ValueatRisk(;
                                                                               alg = DistributionValueatRisk())))
const _GEN_RANGE_DIST_GAIN = typeof(GenericValueatRiskRange(;
                                                            gain = ValueatRisk(;
                                                                               alg = DistributionValueatRisk())))

const _OLD_NETRETURNS = Any[WorstRealisation, _VAR_MIP, _VAR_RANGE_MIP,
                            ConditionalValueatRisk,
                            DistributionallyRobustConditionalValueatRisk,
                            DistributionallyRobustConditionalValueatRiskRange,
                            EntropicValueatRisk, EntropicValueatRiskRange,
                            RelativisticValueatRisk, RelativisticValueatRiskRange,
                            DrawdownatRisk, MaximumDrawdown, AverageDrawdown,
                            ConditionalDrawdownatRisk,
                            DistributionallyRobustConditionalDrawdownatRisk, UlcerIndex,
                            EntropicDrawdownatRisk, RelativisticDrawdownatRisk,
                            RelativeDrawdownatRisk, RelativeMaximumDrawdown,
                            RelativeAverageDrawdown, RelativeConditionalDrawdownatRisk,
                            RelativeUlcerIndex, RelativeEntropicDrawdownatRisk,
                            RelativeRelativisticDrawdownatRisk, Range,
                            ConditionalValueatRiskRange, OrderedWeightsArray,
                            OrderedWeightsArrayRange, BrownianDistanceVariance, MeanReturn,
                            PowerNormValueatRisk, PowerNormValueatRiskRange,
                            PowerNormDrawdownatRisk, RelativePowerNormDrawdownatRisk,
                            _GEN_RANGE_RET]
const _OLD_WEIGHTSRETURNSFEES = Any[LowOrderMoment, HighOrderMoment, TrackingRiskMeasure,
                                    RiskTrackingRiskMeasure, ThirdCentralMoment,
                                    MedianAbsoluteDeviation, VarianceSkewKurtosis,
                                    _VAR_DIST, _VAR_RANGE_DIST, _GEN_RANGE_DIST_LOSS,
                                    _GEN_RANGE_DIST_GAIN]
const _OLD_WEIGHTS = Any[StandardDeviation, NegativeSkewness, TurnoverRiskMeasure, Variance,
                         UncertaintySetVariance, EqualRisk, Kurtosis, Skewness]

# Composite / return-like measures handled by explicit `expected_risk` methods — they
# orchestrate other measures and intentionally declare no input kind.
const _EXPLICIT = Set{Any}([RiskRatio, NonOptimisationRiskRatio, MeanReturnRiskRatio,
                            ExpectedReturn, ExpectedReturnRiskRatio,
                            RiskTrackingRiskMeasure])

# The declared kind as a *type* — instance-free, since each method returns a singleton.
# Undeclared types hit the erroring default (inferred `Union{}`).
function declared_kind(@nospecialize(T))
    return reduce(typejoin, Base.return_types(PO.risk_input_kind, (T,)); init = Union{})
end

function all_concrete(@nospecialize(T))
    acc = Any[]
    for S in InteractiveUtils.subtypes(T)
        isabstracttype(S) ? append!(acc, all_concrete(S)) : push!(acc, S)
    end
    return acc
end

@testset "risk_input_kind — equivalence with old routing unions" begin
    # The three kinds partition the measures; no type may be classified two ways.
    @test allunique(vcat(_OLD_NETRETURNS, _OLD_WEIGHTSRETURNSFEES, _OLD_WEIGHTS))
    for T in _OLD_NETRETURNS
        @test declared_kind(T) === PO.NetReturnsInput
    end
    for T in _OLD_WEIGHTSRETURNSFEES
        @test declared_kind(T) === PO.WeightsReturnsFeesInput
    end
    for T in _OLD_WEIGHTS
        @test declared_kind(T) === PO.WeightsInput
    end
end

# Every kind a type can answer. A measure whose formulation selects the estimand answers
# more than one, and each must still be a concrete kind: an undeclared type infers
# `Union{}` from the erroring default and fails here.
function declared_kinds(@nospecialize(T))
    return unique(Base.return_types(PO.risk_input_kind, (T,)))
end

@testset "risk_input_kind — every concrete measure is classified" begin
    for T in all_concrete(PO.AbstractBaseRiskMeasure)
        if T in _EXPLICIT
            continue
        end
        ks = declared_kinds(T)
        @test !isempty(ks) && all(k -> isconcretetype(k) && k <: PO.RiskInputKind, ks)
    end
end

# ── ADR 0007: the precomputed-returns contract ────────────────────────────────────────────
# `expected_risk_from_returns(r, x)` evaluates a measure on an already-reduced net-return
# series. It is gated by `supports_precomputed_returns` so that an ineligible measure — a
# `WeightsInput` measure (whose `r(w)` shares the `r(::VecNum)` signature and would otherwise
# silently score the series as weights), a moment measure with a per-asset `mu` (whose target
# `dot(w, mu)` needs the absent weights), or a variance-carrying composite — throws an
# explanatory error rather than returning nonsense or hitting a raw `MethodError`.
const _x_series = [sinpi(2i / 64) * 0.1 + cospi(i / 32) * 0.03 for i in 1:64]

@testset "precomputed-returns contract — eligibility & differential" begin
    g = PO.expected_risk_from_returns
    # Eligible: NetReturnsInput measures, the weight-independent-target moment family, and
    # ratio composites whose constituents are themselves eligible.
    eligible = Any[ConditionalValueatRisk(), MaximumDrawdown(), ValueatRisk(),
                   WorstRealisation(), Range(), MeanReturn(), LowOrderMoment(),
                   HighOrderMoment(), MedianAbsoluteDeviation(), ThirdCentralMoment(),
                   ThirdCentralMoment(; ve = SimpleVariance()), LowOrderMoment(; mu = 0.01),
                   MedianAbsoluteDeviation(; mu = PO.MeanCentering()),
                   RiskRatio(; r1 = ConditionalValueatRisk(), r2 = MaximumDrawdown()),
                   MeanReturnRiskRatio(; rk = ConditionalValueatRisk()),
                   TrackingRiskMeasure(; tr = ReturnsTracking(; w = _x_series .* 0.5))]
    for r in eligible
        @test PO.supports_precomputed_returns(r)
        v = g(r, _x_series)
        @test v isa Number && isfinite(v)
    end

    # A `ReturnsTracking` measure is dispatched on the *second* type parameter, because
    # `settings` is the first. The declarations and the single-vector functor must bind on a
    # concrete instance, not only on the `UnionAll` the completeness test below inspects.
    let r = TrackingRiskMeasure(; tr = ReturnsTracking(; w = _x_series .* 0.5))
        @test g(r, _x_series) ≈ r([1], reshape(_x_series, :, 1))
        @test g(r, _x_series) > 0
    end

    # Differential oracle: for the moment family the single-vector form equals the one-asset
    # `(w, X, fees)` form — the same oracle the entropy-pooling tests use (ADR 0007).
    for r in (LowOrderMoment(), HighOrderMoment(), MedianAbsoluteDeviation(),
              ThirdCentralMoment(), ThirdCentralMoment(; ve = SimpleVariance()),
              LowOrderMoment(; mu = 0.01))
        @test g(r, _x_series) ≈ r([1], reshape(_x_series, :, 1))
    end

    # Ineligible: the gate throws the explanatory `ArgumentError` — no silent wrong answer,
    # no raw `MethodError`. The default `RiskRatio` is ineligible via its
    # weights-only `Variance` constituent.
    mu2 = [0.1, 0.2]
    ineligible = Any[EqualRisk(), TurnoverRiskMeasure(; w = fill(inv(64), 64)),
                     StandardDeviation(; sigma = [1.0 0.0; 0.0 1.0]),
                     Variance(; sigma = [1.0 0.0; 0.0 1.0]), NegativeSkewness(),
                     VarianceSkewKurtosis(), RiskRatio(), LowOrderMoment(; mu = mu2),
                     HighOrderMoment(; mu = mu2), Skewness(), Kurtosis(),
                     MedianAbsoluteDeviation(; mu = mu2), ThirdCentralMoment(; mu = mu2),
                     TrackingRiskMeasure(; tr = WeightsTracking(; w = [0.5, 0.5]))]
    for r in ineligible
        @test !PO.supports_precomputed_returns(r)
        @test_throws ArgumentError g(r, _x_series)
    end

    # The `WeightsTracking` functor is the second half of the same dispatch: a direct call
    # explains the refusal instead of raising a bare `MethodError`.
    @test_throws ArgumentError TrackingRiskMeasure(;
                                                   tr = WeightsTracking(; w = [0.5, 0.5]))(_x_series)
end

@testset "precomputed-returns contract — completeness" begin
    # EVERY concrete measure resolves `supports_precomputed_returns` to a `Bool` — a future
    # measure that fails to inherit an eligibility (e.g. via a missing `risk_input_kind`)
    # trips here rather than at runtime.
    #
    # The `_EXPLICIT` composites are NOT skipped here, and that is the whole point of the
    # loop being separate from the classification loop above. They are exempt from declaring
    # a *kind*, because an explicit `expected_risk` method routes each of them. They are not
    # exempt from resolving the *predicate*: `assert_scoreable` and
    # `expected_risk_from_returns` call it on whatever measure the user names, so a
    # composite that answers nothing raises the trait's internal "is not defined for" error
    # in a seam where the user needs a statement about their measure instead. #587 is that
    # defect: `ExpectedReturn` and `ExpectedReturnRiskRatio` reached `ScoreSelector` that
    # way, and the skip that used to stand here is why nothing caught it.
    for T in all_concrete(PO.AbstractBaseRiskMeasure)
        rt = reduce(typejoin, Base.return_types(PO.supports_precomputed_returns, (T,));
                    init = Union{})
        # `Bool <: rt` holds when the predicate resolves — `Bool`, or `Any` for a moment
        # UnionAll whose `mu` field is abstractly typed. It fails only on `Union{}`: an
        # undeclared measure whose `risk_input_kind` throws.
        @test Bool <: rt
    end
end

@testset "precomputed-returns contract — the composites that refuse a bare series" begin
    # #587. Three composites answered the predicate with the trait's internal "is not
    # defined for" error, because the skip above meant nothing asked them.
    #
    #   - A `PrRM` reads the prior result and contracts the expected returns it states with
    #     the weights. A bare series carries neither.
    #   - A `RiskTrackingRiskMeasure` tracks a benchmark held as a WEIGHT vector, so both of
    #     its functors need `w` as well.
    #
    # Each now answers `false`, and both seams that consult it then speak about the measure
    # the user named rather than about the trait.
    g = PO.expected_risk_from_returns
    for r in (ExpectedReturn(), ExpectedReturnRiskRatio(),
              RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = [0.5, 0.5])))
        @test !PO.supports_precomputed_returns(r)
        @test_throws ArgumentError g(r, _x_series)
        # The preselection seam is where the internal error used to reach the user.
        @test_throws ArgumentError ScoreSelector(; score = r,
                                                 rule = ThresholdRule(; lo = 0.0))
    end
end

@testset "the base-file selectors refuse a both-`nothing` call" begin
    # #595. `solver_selector` documented "Returns `nothing` if neither is available", and
    # the both-`nothing` method throws instead. The docstring now states the refusal, and
    # these two assertions hold it there. `sel` never reaches the solver refusal: a
    # both-`nothing` call routes on the operand types to `nothing_scalar_array_selector`.
    @test_throws ArgumentError PO.solver_selector(nothing, nothing)
    @test_throws ArgumentError PO.risk_measure_nothing_scalar_array_view(nothing, nothing,
                                                                         1)
    @test isnothing(PO.sel(nothing, nothing))
    @test isnothing(PO.nothing_scalar_array_selector(nothing, nothing))
end

# ── #773: the weight path is the picker for a scorer ──────────────────────────────────────
# Decision #772. A `VecNum` weight is one target vector and weighs every observation; a
# `MatNum` weight is a weight path, one row of weights per observation, which is what a fold
# scored under a Weight Drift held. `expected_risk`'s weight-taking family gains its `MatNum`
# mirror, and the three input kinds answer a path separately: a `NetReturnsInput` measure
# computes, and the two weight-reading kinds refuse by name.
const _path_X = [0.01 0.02; -0.02 0.01; 0.03 -0.01; 0.005 0.02]
const _path_w = [0.6, 0.4]

@testset "the weight path is the picker, and two kinds refuse it (#773)" begin
    wd = SelfFinancingDrift()
    U = PO.weight_path(wd, _path_w, _path_X)
    Uc = PO.weight_path(nothing, _path_w, _path_X)
    fees = Fees(; l = 0.001, fl = 0.002)
    @test U[1, :] == _path_w
    @test all(row == _path_w for row in eachrow(Uc))

    # A `NetReturnsInput` measure computes, and its answer is the measure over the series
    # the path produces. The kernel is untouched: `calc_net_returns` is what reads the type.
    rn = ConditionalValueatRisk()
    @test PO.risk_input_kind(rn) === PO.NetReturnsInput()
    @test expected_risk(rn, U, _path_X, fees) == rn(calc_net_returns(U, _path_X, fees))
    @test expected_risk(rn, U, _path_X) == rn(calc_net_returns(U, _path_X))
    # A drifted path moves the number, and a constant path reproduces the target reading.
    @test expected_risk(rn, U, _path_X, fees) != expected_risk(rn, _path_w, _path_X, fees)
    @test isapprox(expected_risk(rn, Uc, _path_X, fees),
                   expected_risk(rn, _path_w, _path_X, fees))

    # The prior route resolves the measure and the matrix once, then reads the path.
    rd = ReturnsResult(; nx = ["A", "B"], X = _path_X)
    rnr, Xr = PO.resolve_risk_inputs(rn, rd)
    @test expected_risk(rn, U, rd, fees) == expected_risk(rnr, U, Xr, fees)

    # A vector of measures scalarises over the path, `scale` is its combination weight, and
    # the vector's own prior route resolves once, exactly as the `VecNum` family does.
    rns = [ConditionalValueatRisk(), ConditionalValueatRisk(; alpha = 0.3)]
    @test expected_risk(rns, U, _path_X, fees) ==
          sum(expected_risk(r, U, _path_X, fees) * r.settings.scale for r in rns)
    rnsr, Xsr = PO.resolve_risk_inputs(rns, rd)
    @test expected_risk(rns, U, rd, fees) == expected_risk(rnsr, U, Xsr, fees)
    @test expected_risk(rns, U, _path_X, fees; sca = MaxScalariser()) !=
          expected_risk(rns, U, _path_X, fees)

    # The three ratio composites decompose onto the path as they do onto a vector.
    rr = RiskRatio(; r1 = ConditionalValueatRisk(), r2 = MaximumDrawdown())
    @test expected_risk(rr, U, _path_X, fees) ==
          expected_risk(rr.r1, U, _path_X, fees) / expected_risk(rr.r2, U, _path_X, fees)
    nrr = NonOptimisationRiskRatio(; r1 = ConditionalValueatRisk(), r2 = MaximumDrawdown())
    @test expected_risk(nrr, U, _path_X, fees) ==
          expected_risk(nrr.r1, U, _path_X, fees; sca = nrr.sca1) /
          expected_risk(nrr.r2, U, _path_X, fees; sca = nrr.sca2)
    mrr = MeanReturnRiskRatio(; rk = ConditionalValueatRisk())
    @test expected_risk(mrr, U, _path_X, fees) ==
          (expected_risk(mrr.rt, U, _path_X, fees) - mrr.rf) /
          expected_risk(mrr.rk, U, _path_X, fees; sca = mrr.sca)

    # `WeightsReturnsFeesInput` refuses a path by name. Its kernel reads `w` as one
    # cross-section, and a path gives one number per observation, which is a different
    # quantity rather than a wider input.
    rw = MedianAbsoluteDeviation()
    @test PO.risk_input_kind(rw) === PO.WeightsReturnsFeesInput()
    @test_throws ArgumentError expected_risk(rw, U, _path_X, fees)
    msg = try
        expected_risk(rw, U, _path_X, fees)
    catch e
        sprint(showerror, e)
    end
    @test occursin("MedianAbsoluteDeviation", msg)
    @test occursin("WeightsReturnsFeesInput", msg)
    # It still scores the target weights it does take.
    @test expected_risk(rw, _path_w, _path_X, fees) == rw(_path_w, _path_X, fees)

    # `WeightsInput` refuses on the same terms.
    rv = Variance(; sigma = [1.0 0.2; 0.2 1.0])
    @test PO.risk_input_kind(rv) === PO.WeightsInput()
    @test_throws ArgumentError expected_risk(rv, U, _path_X, fees)
    msg = try
        expected_risk(rv, U, _path_X, fees)
    catch e
        sprint(showerror, e)
    end
    @test occursin("Variance", msg)
    @test occursin("WeightsInput", msg)
    @test expected_risk(rv, _path_w, _path_X, fees) == rv(_path_w)

    # A refusal is taken by the kind and not by the argument count, so the no-fee call and
    # the prior route are refused too.
    @test_throws ArgumentError expected_risk(rw, U, _path_X)
    @test_throws ArgumentError expected_risk(rv, U, rd)
end

# The two kinds that read a return series refuse a `nothing` carrier by name (#1240). The
# call that lands here is the documented `expected_risk(r, res)` on a result that carries no
# carrier of its own, where the fallback to `res.pr` yields `nothing`. Before this the kind
# singleton reached no method and the caller met a `MethodError` naming neither the measure
# nor the missing argument. `WeightsInput` reads no series, so it still answers.
@testset "a `nothing` carrier is refused by name, and only by the two series kinds (#1240)" begin
    rn = ConditionalValueatRisk()
    rw = MedianAbsoluteDeviation()
    rv = Variance(; sigma = [1.0 0.0; 0.0 1.0])
    U = PO.weight_path(nothing, _path_w, _path_X)
    @test PO.risk_input_kind(rn) === PO.NetReturnsInput()
    @test PO.risk_input_kind(rw) === PO.WeightsReturnsFeesInput()
    @test PO.risk_input_kind(rv) === PO.WeightsInput()

    # Both series kinds refuse a carrier-free single target, at either arity.
    for r in (rn, rw)
        @test_throws IsNothingError expected_risk(r, _path_w)
        @test_throws IsNothingError expected_risk(r, _path_w, nothing)
        @test_throws IsNothingError expected_risk(r, _path_w, nothing, nothing)
    end

    # A weight path reaches the new refusal for `NetReturnsInput` alone. The other two
    # kinds refuse **any** path by name already (#773), and that refusal is the more
    # fundamental one, so a carrier-free path keeps it rather than naming the carrier.
    @test_throws IsNothingError expected_risk(rn, U)
    @test_throws IsNothingError expected_risk(rn, U, nothing, nothing)
    @test_throws ArgumentError expected_risk(rw, U)
    @test_throws ArgumentError expected_risk(rv, U)
    for r in (rw, rv)
        msg = try
            expected_risk(r, U)
        catch e
            sprint(showerror, e)
        end
        @test occursin("weight path", msg)
    end

    # The message names the measure and states the ways out.
    msg = try
        expected_risk(rn, _path_w)
    catch e
        sprint(showerror, e)
    end
    @test occursin("ConditionalValueatRisk", msg)
    @test occursin("expected_risk(r, res, X)", msg)
    @test occursin("expected_risk(r, res, pr)", msg)

    # A vector of measures is scalarised through the same door, so it refuses too.
    @test_throws IsNothingError expected_risk([rn], _path_w, nothing, nothing)

    # `WeightsInput` reads no series, so a carrier-free call answers as it always has.
    @test expected_risk(rv, _path_w) == rv(_path_w)
    @test expected_risk(rv, _path_w, nothing, nothing) == rv(_path_w)

    # Handing a carrier in answers, which is the way out the message names.
    @test expected_risk(rn, _path_w, _path_X) == rn(calc_net_returns(_path_w, _path_X))
end

# The drawdown series builders used to seed their running peak by mutating the caller's
# vector -- `pushfirst!(x, 0)`, compute, `popfirst!(x)`. That worked only because every
# caller handed them a freshly allocated `Vector` from `calc_net_returns`. Any other
# `AbstractVector` (a column view, a range) raised a `MethodError`, and an exception thrown
# between the two calls would have left the caller's vector one element longer than it
# started. The peak is now carried in a scalar, so `x` is read and never written.
@testset "Drawdown series are pure" begin
    using Test, PortfolioOptimisers, Random

    PO = PortfolioOptimisers
    Random.seed!(987654321)

    # the previous, mutating implementations, kept here as the behavioural reference
    function ref_absolute(x)
        x = copy(x)
        pushfirst!(x, zero(eltype(x)))
        cs = cumsum(x)
        peak = typemin(eltype(x))
        dd = similar(cs)
        for (idx, i) in pairs(cs)
            peak = ifelse(i > peak, i, peak)
            dd[idx] = i - peak
        end
        popfirst!(dd)
        return dd
    end
    function ref_relative(x)
        x = copy(x)
        pushfirst!(x, zero(eltype(x)))
        cs = cumprod(x .+ one(eltype(x)))
        peak = typemin(eltype(x))
        dd = similar(cs)
        for (idx, i) in pairs(cs)
            peak = ifelse(i > peak, i, peak)
            dd[idx] = i / peak - one(peak)
        end
        popfirst!(dd)
        return dd
    end

    @testset "identical to the previous implementation" begin
        for _ in 1:200
            x = 0.05 .* randn(rand(1:50))
            @test PO.absolute_drawdown_vec(x) ≈ ref_absolute(x)
            @test PO.relative_drawdown_vec(x) ≈ ref_relative(x)
        end
    end

    @testset "the input is not written" begin
        x = 0.05 .* randn(40)
        before = copy(x)
        PO.absolute_drawdown_vec(x)
        PO.relative_drawdown_vec(x)
        MaximumDrawdown()(x)
        @test x == before
        @test length(x) == 40
    end

    @testset "any AbstractVector works" begin
        X = 0.05 .* randn(30, 3)
        col = view(X, :, 2)
        @test PO.absolute_drawdown_vec(col) ≈ PO.absolute_drawdown_vec(X[:, 2])
        @test PO.relative_drawdown_vec(col) ≈ PO.relative_drawdown_vec(X[:, 2])
        @test MaximumDrawdown()(col) ≈ MaximumDrawdown()(X[:, 2])

        # a range is immutable; the old implementation could not take one at all
        r = 0.01:0.01:0.05
        @test PO.absolute_drawdown_vec(r) == zeros(5)   # monotone rise, no drawdown
        @test length(PO.relative_drawdown_vec(r)) == 5
    end

    @testset "the peak starts at the initial portfolio value" begin
        # a first observation below zero is already a drawdown, not the reference point
        @test PO.absolute_drawdown_vec([-0.1, 0.0]) ≈ [-0.1, -0.1]
        @test PO.relative_drawdown_vec([-0.1, 0.0]) ≈ [-0.1, -0.1]

        # and the drawdown series is never positive
        x = 0.05 .* randn(100)
        @test all(<=(0), PO.absolute_drawdown_vec(x))
        @test all(<=(0), PO.relative_drawdown_vec(x))
    end

    # `drawdowns` used to re-inline the two array builders and drop their `init`, so its
    # running peak started at the first observation. The JuMP path reads
    # `absolute_drawdown_arr` and the reporting path reads `drawdowns`, so the constraint
    # and the reported series disagreed on a portfolio that is under water at t = 1.
    @testset "the exported drawdowns agrees with the array builders" begin
        @test drawdowns([-0.1, 0.0]) ≈ [-0.1, -0.1]
        @test drawdowns([-0.1, 0.0], true) ≈ [-0.1, -0.1]
        @test all(<=(0), drawdowns(0.05 .* randn(100)))
        @test all(<=(0), drawdowns(0.05 .* randn(100), true))

        for _ in 1:50
            X = 0.05 .* randn(rand(2:40), rand(1:5))
            for dims in (1, 2)
                @test drawdowns(X; dims = dims) ≈ PO.absolute_drawdown_arr(X; dims = dims)
                @test drawdowns(X, true; dims = dims) ≈
                      PO.relative_drawdown_arr(X; dims = dims)
            end
            cXa = PO.absolute_cumulative_returns(X; dims = 1)
            cXr = PO.relative_cumulative_returns(X; dims = 1)
            @test drawdowns(cXa; cX = true) ≈ PO.absolute_drawdown_arr(cXa; cX = true)
            @test drawdowns(cXr, true; cX = true) ≈ PO.relative_drawdown_arr(cXr; cX = true)
        end

        # column by column, the array builder is the vector builder
        X = 0.05 .* randn(30, 4)
        for j in axes(X, 2)
            @test drawdowns(X)[:, j] ≈ PO.absolute_drawdown_vec(X[:, j])
            @test drawdowns(X, true)[:, j] ≈ PO.relative_drawdown_vec(X[:, j])
        end
    end
end

# The drawdown half of `src/14_NetReturnsDrawdowns.jl`, swept under issue #547. The
# running-peak convention, the `cX` short circuit and the `dims` axis are checked with
# numbers rather than read off the source.
@testset "The drawdown conventions" begin
    using Test, PortfolioOptimisers

    PO = PortfolioOptimisers

    @testset "the running peak starts at the initial capital" begin
        # A series under water from the first period reports a drawdown there. Were the
        # peak seeded at the first observation, both answers would open at `0.0`.
        @test drawdowns([-0.1, 0.05]) ≈ [-0.1, -0.05]
        @test drawdowns([-0.1, 0.05]) != zeros(2)
        @test PO.absolute_drawdown_arr([-0.1, 0.05]) ≈ [-0.1, -0.05]

        # the compound path is seeded at one and gives the matching relative answer
        @test drawdowns([-0.1, 0.05], true) ≈ [-0.09999999999999998, -0.05499999999999994]
        @test PO.relative_drawdown_arr([-0.1, 0.05]) ≈ drawdowns([-0.1, 0.05], true)

        # and the two agree to first order on a small return
        small = [-1e-6, 5e-7]
        @test maximum(abs, drawdowns(small) - drawdowns(small, true)) < 1e-12
    end

    @testset "cX = true skips the cumulation" begin
        x = [0.01, -0.02, 0.03, -0.015]
        @test drawdowns(PO.absolute_cumulative_returns(x); cX = true) == drawdowns(x)
        @test drawdowns(PO.relative_cumulative_returns(x), true; cX = true) ==
              drawdowns(x, true)

        # feeding the raw series with `cX = true` is a different answer, so the flag is read
        @test drawdowns(x; cX = true) != drawdowns(x)
    end

    @testset "dims = 2 transposes the answer of dims = 1" begin
        # A `dims` bug hides on a square matrix, so the fixture is 3x2.
        M = [0.1 -0.2; 0.0 0.0; 0.05 0.05]
        @test drawdowns(M; dims = 2) == permutedims(drawdowns(permutedims(M); dims = 1))
        @test drawdowns(M, true; dims = 2) ==
              permutedims(drawdowns(permutedims(M), true; dims = 1))
        @test cumulative_returns(M; dims = 2) ==
              permutedims(cumulative_returns(permutedims(M); dims = 1))
        @test cumulative_returns(M, true; dims = 2) ==
              permutedims(cumulative_returns(permutedims(M), true; dims = 1))

        # and the two axes genuinely disagree on this fixture
        @test drawdowns(M; dims = 1) != drawdowns(M; dims = 2)
        @test drawdowns(M; dims = 1) ≈ [0.0 -0.2; 0.0 -0.2; 0.0 -0.15]
        @test drawdowns(M; dims = 2) ≈ [0.0 -0.2; 0.0 0.0; 0.0 0.0]
    end

    @testset "cumulative_returns names the helper each branch reaches" begin
        x = [0.01, -0.02, 0.03, -0.015]
        @test cumulative_returns(x) == PO.absolute_cumulative_returns(x)
        @test cumulative_returns(x, true) == PO.relative_cumulative_returns(x)
        @test PO.absolute_cumulative_returns(x) == cumsum(x)
        @test PO.relative_cumulative_returns(x) == cumprod(1 .+ x)
    end
end

#=
Unit tests for the `scalarise` reduction combinator and its slot-wise helpers.
These are the shared reducers behind `hrp_scalarised_risk`,
`herc_intra_cluster_weights`, `herc_cluster_risk` and `near_optimal_centering_risks`; testing them
directly means the four reduction rules (sum, max, min, log-sum-exp) are checked
without needing an optimiser, prior, or solver.
=#
const PO = PortfolioOptimisers

@testset "scalarise scalars" begin
    xs = [1.0, 3.0, 2.0]
    @test PO.scalarise(identity, SumScalariser(), xs) == 6.0
    @test PO.scalarise(identity, MaxScalariser(), xs) == 3.0
    @test PO.scalarise(identity, MinScalariser(), xs) == 1.0
    gamma = 2.0
    lse = PO.scalarise(identity, LogSumExpScalariser(; gamma = gamma), xs)
    @test isapprox(lse, log(sum(exp.(gamma * xs))) / gamma)
    # Single element: every scalariser is the identity.
    for sca in (SumScalariser(), MaxScalariser(), MinScalariser(),
                LogSumExpScalariser(; gamma = gamma))
        @test isapprox(PO.scalarise(identity, sca, [4.2]), 4.2)
    end
end
@testset "scalarise arrays" begin
    xs = [1.0, 2.0]
    @test PO.scalarise(x -> [x, 2 * x], SumScalariser(), xs) == [3.0, 6.0]
    @test PO.scalarise(x -> [x, 3.0 - x], MaxScalariser(), xs) == [2.0, 2.0]
    @test PO.scalarise(x -> [x, 3.0 - x], MinScalariser(), xs) == [1.0, 1.0]
    gamma = 1.5
    lse = PO.scalarise(x -> [x, 3.0 - x], LogSumExpScalariser(; gamma = gamma), xs)
    @test isapprox(lse,
                   [log(sum(exp.(gamma * [1.0, 2.0]))) / gamma,
                    log(sum(exp.(gamma * [2.0, 1.0]))) / gamma])
end
@testset "scalarise tuples with mixed slots" begin
    # The NearOptimalCentering frontier shape: (scalar, vector, scalar).
    xs = [1.0, 2.0]
    f = x -> (x, [x, 3.0 - x], -x)
    s = PO.scalarise(f, SumScalariser(), xs)
    @test s[1] == 3.0 && s[2] == [3.0, 3.0] && s[3] == -3.0
    m = PO.scalarise(f, MaxScalariser(), xs)
    @test m[1] == 2.0 && m[2] == [2.0, 2.0] && m[3] == -1.0
    gamma = 2.0
    l = PO.scalarise(f, LogSumExpScalariser(; gamma = gamma), xs)
    @test isapprox(l[1], log(sum(exp.(gamma * xs))) / gamma)
    @test isapprox(l[2],
                   [log(sum(exp.(gamma * [1.0, 2.0]))) / gamma,
                    log(sum(exp.(gamma * [2.0, 1.0]))) / gamma])
    @test isapprox(l[3], log(sum(exp.(-gamma * xs))) / gamma)
end
@testset "scalarise keyed max/min" begin
    # `by` switches max/min from slot-wise to winner-take-all: the whole f
    # result with the extremal key is returned, as used by HRP (by = sum on the
    # (lrisk, rrisk) pair) and HERC's inner max/min (by = first on (key, value)).
    xs = [1.0, 3.0, 2.0]
    @test PO.scalarise(x -> (x, 10.0 * x), MaxScalariser(), xs; by = first) == (3.0, 30.0)
    @test PO.scalarise(x -> (x, 10.0 * x), MinScalariser(), xs; by = first) == (1.0, 10.0)
    @test PO.scalarise(x -> (x, 2.0 * x), MaxScalariser(), xs; by = sum) == (3.0, 6.0)
    # All keys tie: the earliest element wins.
    @test PO.scalarise(x -> (x, 10.0 - x), MaxScalariser(), xs; by = sum) == (1.0, 9.0)
    @test PO.scalarise(x -> (x, 10.0 - x), MinScalariser(), xs; by = sum) == (1.0, 9.0)
    # `by` is accepted and ignored by the accumulating scalarisers.
    @test PO.scalarise(identity, SumScalariser(), xs; by = sum) == 6.0
    gamma = 2.0
    @test isapprox(PO.scalarise(identity, LogSumExpScalariser(; gamma = gamma), xs;
                                by = sum), log(sum(exp.(gamma * xs))) / gamma)
end
@testset "scalarise helpers" begin
    @test PO.scalarise_combine(+, 1.0, 2.0) == 3.0
    @test PO.scalarise_combine(max, [1.0, 4.0], [2.0, 3.0]) == [2.0, 4.0]
    @test PO.scalarise_combine(+, (1.0, [1.0, 2.0]), (2.0, [3.0, 4.0])) == (3.0, [4.0, 6.0])
    @test PO.scalarise_map(x -> 2 * x, (1.0, [1.0, 2.0])) == (2.0, [2.0, 4.0])
    @test isapprox(PO.scalarise_logsumexp([1.0, 2.0]), log(exp(1.0) + exp(2.0)))
    @test isapprox(PO.scalarise_logsumexp([[1.0, 2.0], [2.0, 1.0]]),
                   [log(exp(1.0) + exp(2.0)), log(exp(2.0) + exp(1.0))])
    t = PO.scalarise_logsumexp([(1.0, [1.0, 2.0]), (2.0, [2.0, 1.0])])
    @test isapprox(t[1], log(exp(1.0) + exp(2.0)))
    @test isapprox(t[2], [log(exp(1.0) + exp(2.0)), log(exp(2.0) + exp(1.0))])
end

# The tracking risk measures of `src/16_RiskMeasures/16_TrackingRiskMeasure.jl`, checked
# against the closed forms their docstrings state: the five norms of `TrackingRiskMeasure`,
# the two modes of `RiskTrackingRiskMeasure`, the two modes of `RiskTrackingError` in an
# optimisation, and every propagation method of the three types.
using Clarabel, JuMP, Statistics

# Observation weights that resolve against the returns matrix. It lives at top level because
# `@testset` expands to a function body, which cannot hold a `struct`.
struct TrackedDecayWeights <: PortfolioOptimisers.DynamicAbstractWeights end
function PortfolioOptimisers.get_observation_weights(::TrackedDecayWeights,
                                                     X::PortfolioOptimisers.MatNum;
                                                     dims::Int = 1, kwargs...)
    return aweights(collect(range(1.0, 2.0; length = size(X, dims))))
end

@testset "Tracking risk measures" begin
    rng = StableRNG(1049)
    T, N = 60, 4
    X = 0.01 .* randn(rng, T, N) .+ 0.001
    w = [0.4, 0.3, 0.2, 0.1]
    wb = fill(0.25, N)
    pr = prior(EmpiricalPrior(), X)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = Dict("verbose" => false))
    fees = Fees(; l = 0.002)

    @testset "TrackingRiskMeasure states the five norms" begin
        d = X * w - X * wb
        for (alg, val) in ((L2Norm(), norm(d, 2) / sqrt(T - 1)),
                           (L2Norm(; ddof = 0), norm(d, 2) / sqrt(T)),
                           (SquaredL2Norm(), norm(d, 2)^2 / (T - 1)), (L1Norm(), norm(d, 1) / T),
                           (LpNorm(), norm(d, 3) / cbrt(T - 1)),
                           (LpNorm(; p = 4), norm(d, 4) / (T - 1)^(1 / 4)), (LInfNorm(), norm(d, Inf)))
            rr = TrackingRiskMeasure(; tr = ReturnsTracking(; w = X * wb), alg = alg)
            rw = TrackingRiskMeasure(; tr = WeightsTracking(; w = wb), alg = alg)
            @test isapprox(rr(w, X), val; rtol = 1e-12)
            @test isapprox(rw(w, X), val; rtol = 1e-12)
            @test isapprox(expected_risk(rw, w, X), val; rtol = 1e-12)
            # A return series tracks a return series; a weight benchmark refuses one.
            @test isapprox(rr(X * w), val; rtol = 1e-12)
            @test_throws ArgumentError rw(X * w)
        end
        # With d = 0 the L2 and L1 forms are Cajas's Equations 9.16 and 9.17.
        rt = WeightsTracking(; w = wb)
        @test isapprox(TrackingRiskMeasure(; tr = rt, alg = L2Norm(; ddof = 0))(w, X),
                       norm(X * wb - X * w, 2) / sqrt(T); rtol = 1e-12)
        @test isapprox(TrackingRiskMeasure(; tr = rt, alg = L1Norm())(w, X),
                       sum(abs, X * wb - X * w) / T; rtol = 1e-12)
    end

    @testset "TrackingRiskMeasure charges each fee on its own series" begin
        # `Fees(; l)` charges `l` times the long weight sum in every period.
        fb = Fees(; l = 0.001)
        r = TrackingRiskMeasure(; tr = WeightsTracking(; w = wb, fees = fb))
        x = X * w .- 0.002
        b = X * wb .- 0.001
        @test isapprox(r(w, X, fees), norm(x - b) / sqrt(T - 1); rtol = 1e-12)
        @test isapprox(r(w, X), norm(X * w - b) / sqrt(T - 1); rtol = 1e-12)
    end

    @testset "RiskTrackingRiskMeasure states the two modes" begin
        sd = factory(StandardDeviation(), pr)
        cvar = ConditionalValueatRisk()
        ri = RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = wb), r = sd)
        rd = RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = wb), r = cvar,
                                     alg = DependentVariableTracking())
        # The independent standard deviation is Cajas's Equation 9.18.
        @test isapprox(ri(w, X), sqrt(dot(w - wb, pr.sigma, w - wb)); rtol = 1e-12)
        @test isapprox(expected_risk(ri, w, X), ri(w, X); rtol = 1e-14)
        @test isapprox(rd(w, X),
                       abs(expected_risk(cvar, w, X) - expected_risk(cvar, wb, X));
                       rtol = 1e-12)
        @test iszero(ri(wb, X))
        @test iszero(rd(wb, X))
    end

    @testset "the independent functor charges the fee of the portfolio" begin
        # The model tracks the series `X(w - wb) - F(w)`, and the functor reads the same
        # series (#1316). Before the fix the functor charged `F(w - wb)`, zero at `wb`.
        cvar = ConditionalValueatRisk()
        x = X * (w - wb) .- 0.002 * sum(w)
        r = RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = wb), r = cvar)
        @test isapprox(r(w, X, fees), cvar(x); rtol = 1e-14)
        @test isapprox(r(wb, X, fees), 0.002; rtol = 1e-12)
        # Without a fee the two series are equal.
        @test r(w, X) == expected_risk(cvar, w - wb, X)
        sol = optimise(MeanRisk(; r = r, obj = MinimumRisk(),
                                opt = JuMPOptimiser(; pe = pr, slv = slv, fees = fees)))
        @test isa(sol.retcode, OptimisationSuccess)
        @test isapprox(sol.w, wb; atol = 1e-6)
        @test isapprox(JuMP.value(sol.model[:risk]), 0.002; rtol = 1e-6)
        @test isapprox(expected_risk(r, sol.w, X, fees), JuMP.value(sol.model[:risk]);
                       rtol = 1e-6)
        # A measure that reads the weights alone reads no fee.
        rs = factory(RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = wb),
                                             r = StandardDeviation()), pr)
        @test rs(w, X, fees) == rs(w, X)
        # The dependent functor charges each weight vector its own fee, as the model does.
        rd = RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = wb), r = cvar,
                                     alg = DependentVariableTracking())
        @test isapprox(rd(w, X, fees),
                       abs(expected_risk(cvar, w, X, fees) -
                           expected_risk(cvar, wb, X, fees)); rtol = 1e-12)
    end

    @testset "every tracked measure reads the series of the weight difference" begin
        cvar = ConditionalValueatRisk()
        wd = w - wb
        x = X * wd .- 0.002 * sum(w)
        wb2 = [0.1, 0.2, 0.3, 0.4]
        track(ri; b = wb, alg = IndependentVariableTracking()) = RiskTrackingRiskMeasure(;
                                                                                         tr = WeightsTracking(;
                                                                                                              w = b),
                                                                                         r = ri,
                                                                                         alg = alg)
        # A moment measure takes its target from the weight difference, net of the fee of the
        # portfolio, as the model does (#1320).
        lom = factory(LowOrderMoment(), pr)
        tgt = dot(wd, pr.mu) - 0.002 * sum(w)
        @test isapprox(track(lom)(w, X, fees), mean(max.(tgt .- x, 0)); rtol = 1e-12)
        @test isapprox(track(LowOrderMoment())(w, X, fees), mean(max.(mean(x) .- x, 0));
                       rtol = 1e-12)
        @test isapprox(track(MedianAbsoluteDeviation())(w, X, fees),
                       MedianAbsoluteDeviation()(x); rtol = 1e-12)
        # Observation weights that resolve against the data resolve first.
        ow = aweights(collect(range(1.0, 2.0; length = T)))
        lomd = LowOrderMoment(; w = TrackedDecayWeights(), mu = pr.mu)
        @test isapprox(track(lomd)(w, X, fees), mean(max.(tgt .- x, 0), ow); rtol = 1e-12)
        # Every moment family resolves its weights against `X`, down to the variance
        # estimator it holds, and agrees with the same measure built with the resolved
        # weights (#1319).
        sv = SimpleVariance(; me = nothing)
        for f in (ow -> LowOrderMoment(; w = ow, mu = pr.mu, alg = SecondMoment()),
                  ow -> HighOrderMoment(; w = ow, mu = pr.mu, ve = sv),
                  ow -> HighOrderMoment(; w = ow, mu = pr.mu, alg = FourthMoment()),
                  ow -> ThirdCentralMoment(; w = ow, mu = pr.mu, ve = SimpleVariance()))
            @test isapprox(track(f(TrackedDecayWeights()))(w, X, fees),
                           track(f(ow))(w, X, fees); rtol = 1e-12)
        end
        # A tracking error takes the norm of the series minus its own benchmark series.
        te = TrackingRiskMeasure(; tr = WeightsTracking(; w = wb2))
        @test isapprox(track(te)(w, X, fees), norm(x - X * wb2, 2) / sqrt(T - 1);
                       rtol = 1e-12)
        # A nested independent mode subtracts its benchmark weights. A nested dependent mode
        # subtracts the risk of its benchmark weights, which pay their own fee.
        @test isapprox(track(track(cvar; b = wb2))(w, X, fees),
                       cvar(X * (wd - wb2) .- 0.002 * sum(w)); rtol = 1e-12)
        @test isapprox(track(track(cvar; b = wb2, alg = DependentVariableTracking()))(w, X,
                                                                                      fees),
                       abs(cvar(x) - cvar(X * wb2 .- 0.002 * sum(wb2))); rtol = 1e-12)
        # A ratio divides the two results, and a composite of the weights reads no fee.
        cvar10 = ConditionalValueatRisk(; alpha = 0.1)
        @test isapprox(track(RiskRatio(; r1 = cvar, r2 = cvar10))(w, X, fees),
                       cvar(x) / cvar10(x); rtol = 1e-12)
        vsk = factory(VarianceSkewKurtosis(), prior(HighOrderPriorEstimator(), X))
        @test track(vsk)(w, X, fees) == vsk(wd, X)
        # Any other measure that reads weights and returns reads the series, or refuses.
        @test isapprox(PortfolioOptimisers.difference_risk(PortfolioOptimisers.WeightsReturnsFeesInput(),
                                                           LowOrderMoment(), wd, w, X,
                                                           fees), LowOrderMoment()(x);
                       rtol = 1e-12)
        @test_throws ArgumentError PortfolioOptimisers.difference_risk(PortfolioOptimisers.WeightsReturnsFeesInput(),
                                                                       te, wd, w, X, fees)
        # In an optimisation the functor reads back the value of the model.
        for ri in (lom, te, track(cvar; b = wb2))
            r = track(ri)
            sol = optimise(MeanRisk(; r = r, obj = MinimumRisk(),
                                    opt = JuMPOptimiser(; pe = pr, slv = slv, fees = fees)))
            @test isa(sol.retcode, OptimisationSuccess)
            # The net target lets the tracked moment reach zero at the benchmark (#1320).
            @test isapprox(expected_risk(factory(r, pr, slv), sol.w, X, fees),
                           JuMP.value(sol.model[:risk]); rtol = 1e-6, atol = 1e-9)
        end
    end

    @testset "in an optimisation the independent mode is exact and the dependent mode one-sided" begin
        cvar = ConditionalValueatRisk()
        rb = expected_risk(cvar, wb, X)
        head(tr) = MeanRisk(; r = cvar, obj = MinimumRisk(),
                            opt = JuMPOptimiser(; pe = pr, slv = slv, tr = tr))
        # The independent bound binds at `err`.
        err = 1e-3
        sol = optimise(head(RiskTrackingError(; tr = WeightsTracking(; w = wb), r = cvar,
                                              err = err)))
        @test isa(sol.retcode, OptimisationSuccess)
        @test isapprox(expected_risk(cvar, sol.w - wb, X), err; rtol = 1e-3)
        # The dependent bound holds from above only: the portfolio is less risky than the
        # benchmark by far more than `err` (#1317).
        err = 1e-5
        sol = optimise(head(RiskTrackingError(; tr = WeightsTracking(; w = wb), r = cvar,
                                              err = err, alg = DependentVariableTracking())))
        @test isa(sol.retcode, OptimisationSuccess)
        rw = expected_risk(cvar, sol.w, X)
        @test rw - rb <= err
        @test rb - rw > 10 * err
        # In the objective, the model reports zero for a portfolio that is less risky. This
        # holds for every measure whose model is an upper bound on the risk (#1317).
        for ri in (cvar, factory(StandardDeviation(), pr), factory(LowOrderMoment(), pr))
            rbi = expected_risk(ri, wb, X)
            r = RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = wb), r = ri,
                                        alg = DependentVariableTracking())
            sol = optimise(MeanRisk(; r = r, obj = MinimumRisk(),
                                    opt = JuMPOptimiser(; pe = pr, slv = slv)))
            @test isa(sol.retcode, OptimisationSuccess)
            @test abs(JuMP.value(sol.model[:risk])) < 1e-8
            @test rbi - expected_risk(ri, sol.w, X) > 1e-5
            @test isapprox(r(sol.w, X), rbi - expected_risk(ri, sol.w, X); rtol = 1e-12)
        end
    end

    @testset "the constructors" begin
        @test_throws DomainError RiskTrackingError(; tr = WeightsTracking(; w = wb),
                                                   err = -1.0)
        @test_throws DomainError RiskTrackingError(; tr = WeightsTracking(; w = wb),
                                                   err = Inf)
        # The tracked measure loses its own bound and objective flag.
        ub = RiskMeasureSettings(; ub = 0.1, rke = true)
        rte = RiskTrackingError(; tr = WeightsTracking(; w = wb),
                                r = StandardDeviation(; settings = ub))
        @test isnothing(rte.r.settings.ub)
        @test !rte.r.settings.rke
        rt = RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = wb),
                                     r = ConditionalValueatRisk(; settings = ub))
        @test isnothing(rt.r.settings.ub)
        @test !rt.r.settings.rke
        # The dependent mode warns for a measure whose model is a quadratic expression.
        @test_logs (:warn, r"not guaranteed to work") RiskTrackingRiskMeasure(;
                                                                              tr = WeightsTracking(;
                                                                                                   w = wb),
                                                                              alg = DependentVariableTracking())
        @test_logs RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = wb))
    end

    @testset "port_opt_view slices the benchmark and the tracked measure" begin
        i = [1, 3]
        sd = factory(StandardDeviation(), pr)
        rte = RiskTrackingError(; tr = WeightsTracking(; w = wb), r = sd, err = 0.01,
                                alg = DependentVariableTracking())
        v = PortfolioOptimisers.port_opt_view(rte, i, X)
        @test v.tr.w == wb[i]
        @test v.r.sigma == pr.sigma[i, i]
        @test v.err == rte.err
        @test v.alg === rte.alg
        rt = RiskTrackingRiskMeasure(; settings = RiskMeasureSettings(; scale = 2.0),
                                     tr = WeightsTracking(; w = wb), r = sd)
        v = PortfolioOptimisers.port_opt_view(rt, i, X)
        @test v.tr.w == wb[i]
        @test v.r.sigma == pr.sigma[i, i]
        @test v.settings === rt.settings
        @test v.alg === rt.alg
        tm = TrackingRiskMeasure(; tr = WeightsTracking(; w = wb))
        @test PortfolioOptimisers.port_opt_view(tm, i).tr.w == wb[i]
    end

    @testset "A data benchmark binds to the B of the rows it reads (#1657)" begin
        po = PortfolioOptimisers
        nx = ["A$i" for i in 1:N]
        bv = X * wb .+ 0.001
        Bm = 0.5 .* X
        rdv = ReturnsResult(; nx = nx, X = X, nb = ["BM"], B = bv)
        rdm = ReturnsResult(; nx = nx, X = X, nb = nx, B = Bm)
        rdn = ReturnsResult(; nx = nx, X = X)
        dt = DataBenchmarkTracking()
        @test isnothing(dt.B)
        # The binding reads the rows, and replaces a benchmark bound earlier.
        @test po.bind_data_benchmark(dt, rdv).B === rdv.B
        @test po.bind_data_benchmark(po.bind_data_benchmark(dt, rdv), rdm).B === rdm.B
        # An object with no data benchmark, and a reader with no returns data, give the
        # object itself.
        tw = TrackingError(; tr = WeightsTracking(; w = wb), err = 0.01)
        rs = [Variance(), TrackingRiskMeasure(; tr = WeightsTracking(; w = wb))]
        @test po.bind_data_benchmark(tw, rdv) === tw
        @test po.bind_data_benchmark(rs, rdv) === rs
        @test po.bind_data_benchmark(dt, nothing) === dt
        # A plain vector of numbers is no estimator, and reaches the reader as it is (#1912).
        @test po.bind_data_benchmark(wb, rdv) === wb
        # A tracking error, a measure, a vector, and the measure that a risk tracker tracks.
        tm = TrackingRiskMeasure(; tr = dt)
        @test po.bind_data_benchmark(TrackingError(; tr = dt, err = 0.01), rdv).tr.B ===
              rdv.B
        @test po.bind_data_benchmark([Variance(), tm], rdv)[2].tr.B === rdv.B
        rtr = RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = wb), r = tm)
        @test po.bind_data_benchmark(rtr, rdv).r.tr.B === rdv.B
        # The refusals: returns data with no benchmark, and a reader with no rows. The bind
        # takes no `brt`: the frame of a prior is refused in the constructor of its holder.
        @test_throws MethodError po.bind_data_benchmark(dt, rdv, true)
        @test_throws ArgumentError po.bind_data_benchmark(dt, rdn)
        @test_throws ArgumentError tm(w, X)
        # A vector benchmark is the series, so it equals a returns series of the same rows.
        bound = po.bind_data_benchmark(tm, rdv)
        @test bound(w, X) == TrackingRiskMeasure(; tr = ReturnsTracking(; w = bv))(w, X)
        @test expected_risk(tm, w, rdv) == bound(w, X)
        @test bound(X * w) == bound(w, X)
        @test po.supports_precomputed_returns(bound)
        # A matrix benchmark gives the series B * w, so the error is that of (X - B) w.
        mbound = po.bind_data_benchmark(tm, rdm)
        @test isapprox(mbound(w, X), po.norm_error(L2Norm(), (X - Bm) * w, zeros(T), T))
        @test expected_risk(tm, w, rdm) == mbound(w, X)
        @test !po.supports_precomputed_returns(mbound)
        @test !po.supports_precomputed_returns(tm)
        @test_throws ArgumentError mbound(X * w)
        @test_throws ArgumentError tm(X * w)
        # The benchmark must cover the rows and the assets that the reader holds.
        @test_throws ArgumentError bound(w, X[1:(T - 1), :])
        @test_throws ArgumentError mbound(w[1:3], X[:, 1:3])
        # A view keeps the rows of a vector and views the columns of a matrix.
        i = [1, 3]
        @test po.port_opt_view(bound.tr, i) === bound.tr
        @test po.port_opt_view(mbound.tr, i).B == Bm[:, i]
        @test po.port_opt_view(dt, i) === dt
    end

    @testset "factory advances the benchmark and resolves the tracked measure" begin
        w1 = [0.1, 0.2, 0.3, 0.4]
        moving = WeightsTracking(; w = wb)
        fixed = WeightsTracking(; w = wb, fixed = true)
        # RiskTrackingError.
        rte = RiskTrackingError(; tr = moving, r = StandardDeviation(), err = 0.01)
        f = factory(rte, pr, slv, nothing, w1)
        @test f.tr.w == w1
        @test f.r.sigma == pr.sigma
        @test f.err == rte.err && f.alg === rte.alg
        f = factory(rte, pr, slv, nothing)
        @test f.tr === rte.tr
        @test f.r.sigma == pr.sigma
        @test factory(rte, w1).tr.w == w1
        @test factory(RiskTrackingError(; tr = fixed, err = 0.01), w1).tr === fixed
        # TrackingRiskMeasure.
        tm = TrackingRiskMeasure(; tr = moving, alg = L1Norm())
        @test factory(tm, w1).tr.w == w1
        @test factory(tm, w1).alg === tm.alg
        @test factory(tm, pr, slv, nothing, w1).tr.w == w1
        @test factory(TrackingRiskMeasure(; tr = fixed), w1).tr === fixed
        ret = ReturnsTracking(; w = X * wb)
        @test factory(TrackingRiskMeasure(; tr = ret), w1).tr === ret
        # RiskTrackingRiskMeasure.
        rt = RiskTrackingRiskMeasure(; tr = moving, r = StandardDeviation())
        f = factory(rt, pr, slv)
        @test f.tr === rt.tr
        @test f.r.sigma == pr.sigma
        f = factory(rt, w1)
        @test f.tr.w == w1
        @test factory(RiskTrackingRiskMeasure(; tr = fixed), w1).tr === fixed
    end

    @testset "needs_previous_weights reads the benchmark and the tracked measure" begin
        moving = WeightsTracking(; w = wb)
        fixed = WeightsTracking(; w = wb, fixed = true)
        tn = TurnoverRiskMeasure(; w = wb)
        @test PortfolioOptimisers.needs_previous_weights(TrackingRiskMeasure(; tr = moving))
        @test !PortfolioOptimisers.needs_previous_weights(TrackingRiskMeasure(; tr = fixed))
        @test !PortfolioOptimisers.needs_previous_weights(TrackingRiskMeasure(;
                                                                              tr = ReturnsTracking(;
                                                                                                   w = X *
                                                                                                       wb)))
        for Tracker in (RiskTrackingError, RiskTrackingRiskMeasure)
            @test PortfolioOptimisers.needs_previous_weights(Tracker(; tr = moving))
            @test !PortfolioOptimisers.needs_previous_weights(Tracker(; tr = fixed))
            @test PortfolioOptimisers.needs_previous_weights(Tracker(; tr = fixed, r = tn))
        end
    end
end

# The relativistic risk measures of `src/16_RiskMeasures/06_XatRisk/04_RelativisticXatRisk.jl`,
# checked against the mathematics their docstrings state. The reference below solves the
# Kaniadakis ball of the dual with no conic solver, so it is independent of both programmes
# that `RRM` builds.
using Clarabel, JuMP

# The Kaniadakis logarithm, written out so the reference does not read `kappa_log`.
rl_lnk(u, k) = (u^k - u^(-k)) / (2k)
# Stationarity of `q ln_k(q / c)` at slope `y` has the root `q = c u^(1/k)` below.
function rl_q(L, c, k, lam, eta)
    y = (L .- eta) ./ lam
    u = (k .* y .+ sqrt.(k^2 .* y .^ 2 .+ (1 - k^2))) ./ (1 + k)
    return c .* u .^ (1 / k)
end
# The multiplier of `sum(q) = 1`, found by bisection, because `sum(q)` falls as it rises.
function rl_eta(L, c, k, lam)
    lo = minimum(L) - 1.0
    hi = maximum(L) + 1.0
    while sum(rl_q(L, c, k, lam, lo)) < 1
        lo -= hi - lo
    end
    while sum(rl_q(L, c, k, lam, hi)) > 1
        hi += hi - lo
    end
    for _ in 1:200
        m = (lo + hi) / 2
        sum(rl_q(L, c, k, lam, m)) > 1 ? (lo = m) : (hi = m)
    end
    return (lo + hi) / 2
end
# sup_q q'L subject to sum(q) = 1 and sum(q ln_k(q / (p T))) <= ln_k(1 / (alpha T)). The
# multiplier `lam` of the ball is found by bisection on its logarithm, because the
# divergence of the maximiser falls as `lam` rises. An observation of weight zero has
# q = 0 and adds nothing to either sum, so it is dropped, and T keeps every observation.
function rl_reference(x, alpha, k, w = nothing)
    T = length(x)
    p = isnothing(w) ? fill(1 / T, T) : w ./ sum(w)
    keep = p .> 0
    L = -x[keep]
    c = p[keep] .* T
    r = rl_lnk(1 / (alpha * T), k)
    divergence(lam) = (q = rl_q(L, c, k, lam, rl_eta(L, c, k, lam));
                       sum(q .* rl_lnk.(q ./ c, k)))
    llo, lhi = -30.0, 30.0
    for _ in 1:200
        m = (llo + lhi) / 2
        divergence(exp(m)) > r ? (llo = m) : (lhi = m)
    end
    lam = exp((llo + lhi) / 2)
    q = rl_q(L, c, k, lam, rl_eta(L, c, k, lam))
    return dot(q, L)
end
# A solver whose first construction fails, so the primal programme of `RRM` fails and the
# dual programme runs on the same solver.
function rl_fails_once()
    n = Ref(0)
    f = () -> (n[] += 1;
               n[] == 1 ? error("the first construction fails") : Clarabel.Optimizer())
    return Solver(; name = :fails_once, solver = f, settings = Dict("verbose" => false),
                  check_sol = (; allow_local = true, allow_almost = true))
end

@testset "Relativistic risk measures" begin
    rng = StableRNG(123)
    x = 0.01 .* randn(rng, 200) .+ 0.0005
    wt = rand(rng, 200) .+ 0.5
    W = pweights(wt)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false),
                 check_sol = (; allow_local = true, allow_almost = true))

    @testset "RRM solves the Kaniadakis ball" begin
        for (a, k) in ((0.05, 0.3), (0.05, 0.7), (0.1, 0.1), (0.2, 0.5))
            @test isapprox(PortfolioOptimisers.RRM(x, slv, a, k), rl_reference(x, a, k);
                           rtol = 1e-5)
            @test isapprox(PortfolioOptimisers.RRM(x, slv, a, k, W),
                           rl_reference(x, a, k, wt); rtol = 1e-5)
        end
        # Equal weights are the unweighted measure.
        @test isapprox(PortfolioOptimisers.RRM(x, slv, 0.2, 0.5, pweights(fill(3.0, 200))),
                       PortfolioOptimisers.RRM(x, slv, 0.2, 0.5); rtol = 1e-10)
        # With equal weights and alpha T <= 1 the ball holds every distribution.
        @test isapprox(PortfolioOptimisers.RRM(x, slv, 1e-3, 0.3), -minimum(x); rtol = 1e-6)
        @test_throws IsEmptyError PortfolioOptimisers.RRM(x, Solver[], 0.05, 0.3)
        for T in (RelativisticValueatRisk, RelativisticValueatRiskRange,
                  RelativisticDrawdownatRisk, RelativeRelativisticDrawdownatRisk)
            @test_throws IsEmptyError T(; slv = Solver[])
        end
    end

    @testset "RRM falls back to the dual programme" begin
        for (a, k) in ((0.1, 0.1), (0.2, 0.5))
            @test isapprox(PortfolioOptimisers.RRM(x, rl_fails_once(), a, k),
                           rl_reference(x, a, k); rtol = 1e-5)
            @test isapprox(PortfolioOptimisers.RRM(x, rl_fails_once(), a, k, W),
                           rl_reference(x, a, k, wt); rtol = 1e-5)
        end
        dead = Solver(; name = :dead, solver = () -> error("no solver"))
        @test isnan(PortfolioOptimisers.RRM(x, dead, 0.05, 0.3))
        @test isnan(PortfolioOptimisers.RRM(x, dead, 0.05, 0.3, W))
    end

    @testset "RelativisticValueatRisk sits between EVaR and the largest loss" begin
        a = 0.05
        cvar = ConditionalValueatRisk(; alpha = a)(x)
        evar = EntropicValueatRisk(; alpha = a, slv = slv)(x)
        rl = [RelativisticValueatRisk(; alpha = a, kappa = k, slv = slv)(x)
              for k in (0.01, 0.3, 0.7, 0.99)]
        @test issorted([cvar; evar; rl; -minimum(x)])
        @test isapprox(rl[1], evar; rtol = 1e-3)
        @test isapprox(rl[end], -minimum(x); rtol = 1e-6)
        # The functor needs a solver, and `factory` fills it.
        @test_throws MethodError RelativisticValueatRisk()(x)
        pr = prior(EmpiricalPrior(), hcat(x, 0.01 .* randn(StableRNG(7), 200)))
        @test factory(RelativisticValueatRisk(), pr, slv).slv === slv
    end

    @testset "The range adds the loss tail and the gain tail" begin
        r = RelativisticValueatRiskRange(; slv = slv, alpha = 0.2, kappa_a = 0.5,
                                         beta = 0.1, kappa_b = 0.1)
        @test isapprox(r(x), rl_reference(x, 0.2, 0.5) + rl_reference(-x, 0.1, 0.1);
                       rtol = 1e-5)
    end

    @testset "The drawdown measures read their drawdown series" begin
        c = cumsum(x)
        dd = c .- accumulate(max, [0.0; c])[2:end]
        C = cumprod(1 .+ x)
        rdd = C ./ accumulate(max, [1.0; C])[2:end] .- 1
        a, k = 0.2, 0.5
        rldar = RelativisticDrawdownatRisk(; alpha = a, kappa = k, slv = slv)(x)
        @test isapprox(rldar, rl_reference(dd, a, k); rtol = 1e-5)
        @test isapprox(RelativeRelativisticDrawdownatRisk(; alpha = a, kappa = k,
                                                          slv = slv)(x),
                       rl_reference(rdd, a, k); rtol = 1e-5)
        # DaR <= CDaR <= EDaR <= RLDaR <= MDD, Cajas's Equation 7.100.
        @test issorted([DrawdownatRisk(; alpha = a)(x),
                        ConditionalDrawdownatRisk(; alpha = a)(x),
                        EntropicDrawdownatRisk(; alpha = a, slv = slv)(x), rldar,
                        MaximumDrawdown()(x)])
    end

    @testset "A zero-weight loss does not stall RRM" begin
        # The worst loss sits on an observation of weight zero. Its rows have no effect on
        # the optimum, but a solver stalled on them, 3e-4 away from it.
        xz = copy(x)
        xz[1] = -0.5
        wz = [0.0; ones(199)]
        for (a, k) in ((0.2, 0.3), (0.05, 0.7))
            @test isapprox(PortfolioOptimisers.RRM(xz, slv, a, k, pweights(wz)),
                           rl_reference(xz, a, k, wz); rtol = 1e-5)
        end
    end
end

# The entropic risk measures of `src/16_RiskMeasures/06_XatRisk/03_EntropicXatRisk.jl`,
# checked against the definition their docstrings state. The reference minimises
# z ln(sum(p .* exp.(L ./ z)) / alpha) over z by a golden-section search on ln z, so it
# shares no code with the exponential cone programme of `ERM`.
function ev_reference(x, alpha, w = nothing)
    p = isnothing(w) ? fill(1 / length(x), length(x)) : w ./ sum(w)
    keep = p .> 0
    L = -x[keep]
    lp = log.(p[keep])
    function f(z)
        v = L ./ z .+ lp
        m = maximum(v)
        return z * (m + log(sum(exp.(v .- m))) - log(alpha))
    end
    lo, hi = -25.0, 10.0
    g = (sqrt(5) - 1) / 2
    for _ in 1:300
        c = hi - g * (hi - lo)
        d = lo + g * (hi - lo)
        f(exp(c)) < f(exp(d)) ? (hi = d) : (lo = c)
    end
    # The infimum at z -> 0 is the largest loss.
    return min(f(exp((lo + hi) / 2)), maximum(L))
end

@testset "Entropic risk measures" begin
    rng = StableRNG(123)
    x = 0.01 .* randn(rng, 200) .+ 0.0005
    wt = rand(rng, 200) .+ 0.5
    W = pweights(wt)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false),
                 check_sol = (; allow_local = true, allow_almost = true))
    dead = Solver(; name = :dead, solver = () -> error("no solver"))
    erm(args...) = PortfolioOptimisers.ERM(args...)

    @testset "ERM solves the Chernoff bound" begin
        for a in (0.01, 0.05, 0.2, 0.5, 0.9)
            @test isapprox(erm(x, slv, a), ev_reference(x, a); rtol = 1e-5)
            @test isapprox(erm(x, slv, a, W), ev_reference(x, a, wt); rtol = 2e-5)
        end
        # Equal weights are the unweighted measure.
        @test isapprox(erm(x, slv, 0.2, pweights(fill(3.0, 200))), erm(x, slv, 0.2);
                       rtol = 1e-5)
        # A zero-weight loss sets no risk. Before its rows were dropped, the solver stopped
        # 48 % above the optimum.
        xz = copy(x)
        xz[1] = -0.5
        wz = [0.0; ones(199)]
        @test isapprox(erm(xz, slv, 0.2, pweights(wz)), ev_reference(xz, 0.2, wz);
                       rtol = 1e-5)
        @test isapprox(erm(xz, slv, 0.2, pweights(wz)), erm(xz[2:end], slv, 0.2);
                       rtol = 1e-5)
    end

    @testset "The ends of the significance level" begin
        # With equal weights and alpha T <= 1, the EVaR is the largest loss.
        @test isapprox(erm(x, slv, 1 / 200), -minimum(x); rtol = 1e-5)
        @test isapprox(erm(x, slv, 1e-3), -minimum(x); rtol = 1e-5)
        # Level zero is the largest loss too, and builds no model.
        @test erm(x, dead, 0) == -minimum(x)
        # As alpha rises to one the EVaR falls to the mean loss, which it never reaches.
        evs = [erm(x, slv, a) for a in (0.01, 0.05, 0.2, 0.5, 0.9, 0.999, 0.9999)]
        @test issorted(evs; rev = true)
        @test evs[end] > -mean(x)
        @test isapprox(evs[end], -mean(x); atol = 2e-4)
        @test_throws DomainError EntropicValueatRisk(; alpha = 1)
    end

    @testset "EVaR bounds the conditional value at risk" begin
        a = 0.05
        @test issorted([ValueatRisk(; alpha = a)(x), ConditionalValueatRisk(; alpha = a)(x),
                        EntropicValueatRisk(; alpha = a, slv = slv)(x), -minimum(x)])
    end

    @testset "ERM tries each solver, and returns NaN when none solves" begin
        @test isnan(erm(x, dead, 0.05))
        @test isnan(erm(x, dead, 0.05, W))
        @test isnan(erm(x, [dead, dead], 0.05))
        @test isapprox(erm(x, [dead, slv], 0.05), ev_reference(x, 0.05); rtol = 1e-5)
        @test_throws IsEmptyError erm(x, Solver[], 0.05)
        @test_throws DimensionMismatch erm(x, slv, 0.05, pweights(ones(10)))
        for T in (EntropicValueatRisk, EntropicValueatRiskRange, EntropicDrawdownatRisk,
                  RelativeEntropicDrawdownatRisk)
            @test_throws IsEmptyError T(; slv = Solver[])
        end
        # The functor needs a solver, and `factory` fills it.
        @test_throws MethodError EntropicValueatRisk()(x)
        pr = prior(EmpiricalPrior(), hcat(x, 0.01 .* randn(StableRNG(7), 200)))
        @test factory(EntropicValueatRisk(), pr, slv).slv === slv
    end

    @testset "The range and the drawdown measures read their series" begin
        r = EntropicValueatRiskRange(; slv = slv, alpha = 0.2, beta = 0.1)
        @test isapprox(r(x), ev_reference(x, 0.2) + ev_reference(-x, 0.1); rtol = 1e-5)
        dd = cumsum(x) .- accumulate(max, cumsum(x); init = 0.0)
        rdd = cumprod(1 .+ x) ./ accumulate(max, cumprod(1 .+ x); init = 1.0) .- 1
        @test isapprox(EntropicDrawdownatRisk(; slv = slv, alpha = 0.2)(x),
                       ev_reference(dd, 0.2); rtol = 1e-5)
        @test isapprox(RelativeEntropicDrawdownatRisk(; slv = slv, alpha = 0.2)(x),
                       ev_reference(rdd, 0.2); rtol = 1e-5)
    end
end

# The power norm risk measures of `src/16_RiskMeasures/06_XatRisk/05_PowerNormXatRisk.jl`,
# checked against the definition their docstrings state. The reference minimises
# eta + (sum(q .* max.(L .- eta, 0) .^ p))^(1 / p) / alpha over eta by a golden-section search,
# because the objective is convex in eta, so it shares no code with the power cone programme
# of `PRM`.
function pn_reference(x, alpha, p, w = nothing)
    q = isnothing(w) ? fill(1 / length(x), length(x)) : w ./ sum(w)
    keep = q .> 0
    L = -x[keep]
    q = q[keep]
    f(eta) = eta + (sum(q .* max.(L .- eta, 0) .^ p))^(1 / p) / alpha
    lo, hi = minimum(L), maximum(L)
    g = (sqrt(5) - 1) / 2
    for _ in 1:400
        c = hi - g * (hi - lo)
        d = lo + g * (hi - lo)
        f(c) < f(d) ? (hi = d) : (lo = c)
    end
    return min(f((lo + hi) / 2), maximum(L))
end

@testset "Power norm risk measures" begin
    rng = StableRNG(603)
    x = 0.01 .* randn(rng, 1000) .+ 0.0005
    wt = rand(rng, 1000) .+ 0.5
    W = pweights(wt)
    # Clarabel can stall on a power cone programme, so the measures take a list of solvers,
    # each with a smaller step. None of them loosens the tolerances.
    slv = [Solver(; name = Symbol(:clarabel, k), solver = Clarabel.Optimizer,
                  settings = Dict("verbose" => false, "max_step_fraction" => f),
                  check_sol = (; allow_local = true, allow_almost = true))
           for (k, f) in enumerate((0.99, 0.9, 0.8, 0.7))]
    dead = Solver(; name = :dead, solver = () -> error("no solver"))
    prm(args...) = PortfolioOptimisers.PRM(args...)

    @testset "PRM solves the power norm minimum" begin
        for (a, p) in ((0.2, 2.0), (0.5, 3.0), (0.9, 2.0), (0.2, 1.1))
            @test isapprox(prm(x, slv, a, p), pn_reference(x, a, p); rtol = 1e-5)
            @test isapprox(prm(x, slv, a, p, W), pn_reference(x, a, p, wt); rtol = 1e-5)
        end
        # Equal weights are the unweighted measure.
        @test isapprox(prm(x, slv, 0.2, 2.0, pweights(fill(3.0, 1000))),
                       prm(x, slv, 0.2, 2.0); rtol = 1e-5)
        # A zero-weight loss sets no risk, and the model holds no row for it.
        xz = copy(x)
        xz[1] = -0.5
        wz = [0.0; ones(999)]
        @test isapprox(prm(xz, slv, 0.2, 2.0, pweights(wz)), pn_reference(xz, 0.2, 2.0, wz);
                       rtol = 1e-5)
        @test isapprox(prm(xz, slv, 0.2, 2.0, pweights(wz)), prm(xz[2:end], slv, 0.2, 2.0);
                       rtol = 1e-5)
    end

    @testset "The largest loss, without a model" begin
        # alpha * pi^(-1/p) <= 1: the value is the largest loss, and no solver is called.
        # 0.05 * 1000^(1/3) = 0.5.
        @test prm(x, dead, 0.05, 3.0) == -minimum(x)
        # 0.05 * 1000^(1/2) = 1.58 > 1: the model is built, and the dead solver fails.
        @test isnan(prm(x, dead, 0.05, 2.0))
        # p >= ln(pi) / ln(alpha) is the condition. At alpha = 0.2 the bound is 4.29.
        @test PowerNormValueatRisk(; alpha = 0.2, p = 4.3, slv = dead)(x) ==
              WorstRealisation()(x)
        @test pn_reference(x, 0.2, 4.2) < -minimum(x)
        # A tie of 25 observations raises pi to 0.025, and 0.05 * 0.025^(-1/2) = 0.32.
        xt = copy(x)
        xt[1:25] .= minimum(x) - 0.01
        @test prm(xt, dead, 0.05, 2.0) == -minimum(xt)
        # A weight carries the same share. Without it, the share is 1 / 1000.
        k = ones(Int, 1000)
        k[argmin(x)] = 25
        @test prm(x, dead, 0.05, 2.0, fweights(k)) == -minimum(x)
        # The largest loss on a zero-weight observation does not count.
        xz = copy(x)
        xz[1] = -0.5
        wz = [0.0; ones(999)]
        @test prm(xz, dead, 0.05, 3.0, pweights(wz)) == -minimum(xz[2:end])
    end

    @testset "The order of the power norm" begin
        # p = 1 is the Conditional Value-at-Risk of the definition, but the power cone of
        # order one is degenerate, so every constructor refuses it.
        @test isapprox(pn_reference(x, 0.2, 1.0), ConditionalValueatRisk(; alpha = 0.2)(x);
                       rtol = 1e-10)
        for T in (PowerNormValueatRisk, PowerNormDrawdownatRisk,
                  RelativePowerNormDrawdownatRisk)
            @test_throws DomainError T(; p = 1.0)
            @test_throws DomainError T(; p = 0.5)
        end
        @test_throws DomainError PowerNormValueatRiskRange(; pa = 1.0)
        @test_throws DomainError PowerNormValueatRiskRange(; pb = 1.0)
        # As p falls to one, the measure falls to the Conditional Value-at-Risk.
        cv = ConditionalValueatRisk(; alpha = 0.2)(x)
        vs = [prm(x, slv, 0.2, p) for p in (3.0, 2.0, 1.5, 1.1, 1.01, 1.001)]
        @test issorted(vs; rev = true)
        @test all(>(cv), vs)
        @test isapprox(vs[end], cv; rtol = 2e-3)
        # The order of the docstring, with the largest loss on top.
        @test issorted([cv, prm(x, slv, 0.2, 1.5), prm(x, slv, 0.2, 2.0),
                        prm(x, slv, 0.2, 3.0), -minimum(x)])
    end

    @testset "No order holds against the Entropic Value-at-Risk" begin
        ev(y) = EntropicValueatRisk(; alpha = 0.05, slv = slv)(y)
        # On a short sample the power norm measure is the largest loss, above the EVaR.
        xa = 0.01 .* randn(StableRNG(21), 200)
        @test ev(xa) < PowerNormValueatRisk(; alpha = 0.05, slv = slv)(xa) == -minimum(xa)
        # With a heavy tail and a small order, the EVaR is the larger.
        xb = 0.01 .* randn(StableRNG(22), 2000)
        xb[1:20] .= -0.2
        pb = prm(xb, slv, 0.05, 1.2)
        @test isapprox(pb, pn_reference(xb, 0.05, 1.2); rtol = 1e-5)
        @test pb < ev(xb)
    end

    @testset "PRM tries each solver, and returns NaN when none solves" begin
        @test isnan(prm(x, dead, 0.2, 2.0))
        @test isnan(prm(x, dead, 0.2, 2.0, W))
        @test isnan(prm(x, [dead, dead], 0.2, 2.0))
        @test isapprox(prm(x, [dead, slv[1]], 0.2, 2.0), pn_reference(x, 0.2, 2.0);
                       rtol = 1e-5)
        @test_throws IsEmptyError prm(x, Solver[], 0.2, 2.0)
        @test_throws DimensionMismatch prm(x, slv, 0.2, 2.0, pweights(ones(10)))
        for T in (PowerNormValueatRisk, PowerNormValueatRiskRange, PowerNormDrawdownatRisk,
                  RelativePowerNormDrawdownatRisk)
            @test_throws IsEmptyError T(; slv = Solver[])
        end
        # The functor needs a solver, and `factory` fills it.
        @test_throws MethodError PowerNormValueatRisk()(x)
        pr = prior(EmpiricalPrior(), hcat(x, 0.01 .* randn(StableRNG(7), 1000)))
        @test factory(PowerNormValueatRisk(), pr, slv).slv === slv
    end

    @testset "The range and the drawdown measures read their series" begin
        r = PowerNormValueatRiskRange(; slv = slv, alpha = 0.2, pa = 2.0, beta = 0.5,
                                      pb = 3.0)
        @test isapprox(r(x), pn_reference(x, 0.2, 2.0) + pn_reference(-x, 0.5, 3.0);
                       rtol = 2e-5)
        y = x[1:200]
        dd = cumsum(y) .- accumulate(max, cumsum(y); init = 0.0)
        rdd = cumprod(1 .+ y) ./ accumulate(max, cumprod(1 .+ y); init = 1.0) .- 1
        pd = PowerNormDrawdownatRisk(; slv = slv, alpha = 0.2)(y)
        @test isapprox(pd, pn_reference(dd, 0.2, 2.0); rtol = 1e-5)
        @test isapprox(RelativePowerNormDrawdownatRisk(; slv = slv, alpha = 0.2)(y),
                       pn_reference(rdd, 0.2, 2.0); rtol = 1e-5)
        @test issorted([ConditionalDrawdownatRisk(; alpha = 0.2)(y), pd,
                        MaximumDrawdown()(y)])
    end

    @testset "A zero-weight loss does not move the portfolio" begin
        # The crash row of weight zero has no rows in the model, so the optimum is the
        # optimum without the row. Before, the weights moved by 6e-4 at alpha = 0.1, p = 3.
        Xp = 0.01 .* randn(StableRNG(7), 200, 5) .+ 0.0005
        Xc = vcat([-0.5 -0.4 -0.45 -0.5 -0.3], Xp)
        wc = pweights([0.0; ones(200)])
        for (a, p) in ((0.2, 2.0), (0.1, 3.0))
            wA = optimise(MeanRisk(; r = PowerNormValueatRisk(; alpha = a, p = p, w = wc),
                                   opt = JuMPOptimiser(; pe = prior(EmpiricalPrior(), Xc),
                                                       slv = slv))).w
            wB = optimise(MeanRisk(; r = PowerNormValueatRisk(; alpha = a, p = p),
                                   opt = JuMPOptimiser(; pe = prior(EmpiricalPrior(), Xp),
                                                       slv = slv))).w
            @test isapprox(wA, wB; atol = 1e-6)
        end
    end
end

# The conditional tail measures of `src/16_RiskMeasures/06_XatRisk/02_ConditionalXatRisk.jl`,
# checked against the forms their docstrings state: every functor against a Rockafellar-Uryasev
# linear program and the sorted closed form, the weight validation of all seven constructors,
# and the two robust programs against Equation 27 of Mohajerin Esfahani and Kuhn.
using Clarabel, HiGHS, JuMP

# The weighted Rockafellar-Uryasev minimum, min over nu of -nu + sum(p .* max.(nu .- x, 0)) /
# (alpha * sum(p)), solved as a linear program.
function ru_lp(x, alpha, p = ones(length(x)))
    m = Model(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, nu)
    @variable(m, z[1:length(x)] >= 0)
    @constraint(m, z .>= nu .- x)
    @objective(m, Min, -nu + sum(p .* z) / (alpha * sum(p)))
    optimize!(m)
    return objective_value(m)
end
# The sorted closed form of the same minimum, without weights.
function sorted_tail(x, alpha)
    s = sort(x)
    aT = alpha * length(x)
    k = floor(Int, aT) + 1
    return -(sum(s[1:(k - 1)]) + (aT - (k - 1)) * s[k]) / aT
end
# The drawdown series of the definitions, with the running peaks starting at c_0 = 0 and C_0 = 1.
abs_dd(x) = cumsum(x) .- accumulate(max, cumsum(x); init = zero(eltype(x)))
rel_dd(x) = cumprod(1 .+ x) ./ accumulate(max, cumprod(1 .+ x); init = one(eltype(x))) .- 1
# The long-only minimum of mean(loss) + l * CVaR(loss) over the simplex, with the loss the
# negated portfolio return, or the portfolio drawdown when `dd` is true. It is the limit of the
# robust measures as the radius goes to zero.
function mean_cvar_lp(X, alpha, l, p; dd = false)
    T, N = size(X)
    m = Model(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, w[1:N] >= 0)
    @constraint(m, sum(w) == 1)
    @variable(m, nu)
    @variable(m, z[1:T] >= 0)
    loss = if dd
        C = cumsum(X; dims = 1)
        @variable(m, d[1:T])
        for t in 1:T
            @constraint(m, d[t] >= -dot(C[t, :], w))
            for s in 1:t
                @constraint(m, d[t] >= dot(C[s, :] - C[t, :], w))
            end
        end
        d
    else
        -X * w
    end
    @constraint(m, z .>= loss .- nu)
    sp = sum(p)
    @objective(m, Min, sum(p .* loss) / sp + l * (nu + sum(p .* z) / (alpha * sp)))
    optimize!(m)
    return objective_value(m)
end

@testset "Conditional tail measures" begin
    @testset "Each functor is the Rockafellar-Uryasev minimum" begin
        rng = StableRNG(1047)
        for T in (7, 20, 63), alpha in (0.01, 0.05, 0.33, 0.9)
            x = 0.02 .* randn(rng, T) .+ 0.001
            x0 = copy(x)
            p = rand(rng, T)
            p[1] = 0.0
            pw = pweights(p)
            cv = ConditionalValueatRisk(; alpha = alpha)
            @test isapprox(cv(x), ru_lp(x, alpha); atol = 1e-12)
            @test isapprox(cv(x), sorted_tail(x, alpha); atol = 1e-14)
            @test isapprox(ConditionalValueatRisk(; alpha = alpha, w = pw)(x),
                           ru_lp(x, alpha, p); atol = 1e-12)
            # Unit weights price the same quantity as no weights.
            @test isapprox(ConditionalValueatRisk(; alpha = alpha, w = pweights(ones(T)))(x),
                           cv(x); atol = 1e-14)
            @test DistributionallyRobustConditionalValueatRisk(; alpha = alpha)(x) == cv(x)
            for beta in (alpha, 0.2)
                rg = ConditionalValueatRiskRange(; alpha = alpha, beta = beta)
                rgw = ConditionalValueatRiskRange(; alpha = alpha, beta = beta, w = pw)
                @test isapprox(rg(x), ru_lp(x, alpha) + ru_lp(-x, beta); atol = 1e-12)
                @test isapprox(rgw(x), ru_lp(x, alpha, p) + ru_lp(-x, beta, p);
                               atol = 1e-12)
                @test DistributionallyRobustConditionalValueatRiskRange(; alpha = alpha,
                                                                        beta = beta)(x) ==
                      rg(x)
            end
            cd = ConditionalDrawdownatRisk(; alpha = alpha)
            @test isapprox(cd(x), ru_lp(abs_dd(x), alpha); atol = 1e-12)
            @test isapprox(ConditionalDrawdownatRisk(; alpha = alpha, w = pw)(x),
                           ru_lp(abs_dd(x), alpha, p); atol = 1e-12)
            @test DistributionallyRobustConditionalDrawdownatRisk(; alpha = alpha)(x) ==
                  cd(x)
            @test isapprox(RelativeConditionalDrawdownatRisk(; alpha = alpha)(x),
                           ru_lp(rel_dd(x), alpha); atol = 1e-12)
            @test isapprox(RelativeConditionalDrawdownatRisk(; alpha = alpha, w = pw)(x),
                           ru_lp(rel_dd(x), alpha, p); atol = 1e-12)
            # No functor reorders the caller's vector.
            @test x == x0
        end
    end

    @testset "The upper tail of the range is the CVaR of the negated returns, bit for bit" begin
        # The functor negates its copy and sorts it ascending, rather than sorting it with
        # `rev = true` (issue #1341). Negation is exact, so the range must equal the two plain
        # CVaRs exactly, ties and all, in every precision.
        rng = StableRNG(1341)
        for T in (Float32, Float64, BigFloat), n in (1, 7, 50), alpha in (0.05, 0.3, 0.99),
            beta in (0.05, 0.3, 0.99)

            x = T.(round.(randn(rng, n); digits = 1))
            x0 = copy(x)
            rg = ConditionalValueatRiskRange(; alpha = alpha, beta = beta)
            @test isequal(rg(x),
                          ConditionalValueatRisk(; alpha = alpha)(x) +
                          ConditionalValueatRisk(; alpha = beta)(-x))
            @test x == x0
        end
    end

    @testset "Every constructor refuses a negative weight" begin
        # `DistributionallyRobustConditionalValueatRisk` checked only that the weights were not
        # empty, and took a negative weight its six siblings refuse.
        bad = pweights([0.5, -0.1, 0.6])
        for R in (ConditionalValueatRisk, DistributionallyRobustConditionalValueatRisk,
                  ConditionalValueatRiskRange, DistributionallyRobustConditionalValueatRiskRange,
                  ConditionalDrawdownatRisk, DistributionallyRobustConditionalDrawdownatRisk,
                  RelativeConditionalDrawdownatRisk)
            @test_throws DomainError R(; w = bad)
            @test R(; w = pweights([0.5, 0.0, 0.5])) isa R
        end
    end

    rng = StableRNG(10471)
    X = 0.02 .* randn(rng, 40, 4) .+ [0.002 0.001 0.0 -0.001]
    pr = prior(EmpiricalPrior(), X)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = Dict("verbose" => false, "tol_gap_abs" => 1e-10,
                                 "tol_gap_rel" => 1e-10, "tol_feas" => 1e-10))
    min_risk(r) = optimise(MeanRisk(; r = r, obj = MinimumRisk(),
                                    opt = JuMPOptimiser(; pe = pr, slv = slv)))

    @testset "The robust program is Equation 27" begin
        # Values of Equation 27 over this sample, from an independent implementation of it:
        # (alpha, l, r) => (optimum, long-only weights).
        ref = ((0.05, 1.0, 1e-4) => (1.562371905272e-02,
                                     [4.934610126584e-01, 4.881094891810e-01, 0.0, 1.842949813185e-02]),
               (0.2, 3.0, 1e-4) => (2.864793597636e-02,
                                    [3.368775790342e-01, 2.049947764238e-01, 0.0, 4.581276439958e-01]),
               (0.05, 1.0, 0.02) => (1.238196688788e-01, fill(0.25, 4)),
               (0.2, 0.5, 0.1) => (9.027779962236e-02, fill(0.25, 4)))
        for ((alpha, l, r), (val, w)) in ref
            sol = min_risk(DistributionallyRobustConditionalValueatRisk(; alpha = alpha,
                                                                        l = l, r = r))
            @test isapprox(JuMP.value(sol.model[:risk]), val; rtol = 1e-8)
            @test isapprox(sol.w, w; atol = 1e-6)
        end
    end

    @testset "As the radius goes to zero the mean term stays" begin
        p = collect(range(0.2, 1.0; length = size(X, 1)))
        for (wo, pp) in ((nothing, ones(size(X, 1))), (pweights(p), p)), l in (0.5, 3.0),
            alpha in (0.05, 0.2)

            cvar = min_risk(DistributionallyRobustConditionalValueatRisk(; alpha = alpha,
                                                                         l = l, r = 1e-9,
                                                                         w = wo))
            @test isapprox(JuMP.value(cvar.model[:risk]), mean_cvar_lp(X, alpha, l, pp);
                           atol = 1e-7)
            cdar = min_risk(DistributionallyRobustConditionalDrawdownatRisk(; alpha = alpha,
                                                                            l = l, r = 1e-9,
                                                                            w = wo))
            @test isapprox(JuMP.value(cdar.model[:risk]),
                           mean_cvar_lp(X, alpha, l, pp; dd = true); atol = 1e-7)
        end
        # The mean terms of the two tails of the range cancel.
        for (alpha, beta, la, lb) in ((0.05, 0.1, 1.0, 2.0), (0.2, 0.05, 0.5, 3.0))
            sol = min_risk(DistributionallyRobustConditionalValueatRiskRange(;
                                                                             alpha = alpha,
                                                                             beta = beta,
                                                                             l_a = la,
                                                                             l_b = lb,
                                                                             r_a = 1e-9,
                                                                             r_b = 1e-9))
            m = Model(HiGHS.Optimizer)
            set_silent(m)
            T, N = size(X)
            @variable(m, w[1:N] >= 0)
            @constraint(m, sum(w) == 1)
            @variable(m, nu[1:2])
            @variable(m, z[1:T, 1:2] >= 0)
            @constraint(m, z[:, 1] .>= -X * w .- nu[1])
            @constraint(m, z[:, 2] .>= X * w .- nu[2])
            @objective(m, Min,
                       la * (nu[1] + sum(z[:, 1]) / (alpha * T)) +
                       lb * (nu[2] + sum(z[:, 2]) / (beta * T)))
            optimize!(m)
            @test isapprox(JuMP.value(sol.model[:risk]), objective_value(m); atol = 1e-7)
        end
    end

    @testset "The drawdown premium reads a bound on the portfolio drawdown" begin
        # For long-only weights the portfolio drawdown is at most -w' D_t, the linear function
        # of the asset drawdowns that the robust premium of the drawdown measure reads.
        D = PortfolioOptimisers.absolute_drawdown_arr(X)
        for _ in 1:50
            w = rand(rng, size(X, 2))
            w ./= sum(w)
            @test all(-PortfolioOptimisers.absolute_drawdown_vec(X * w) .<= -D * w .+ 1e-15)
        end
    end

    @testset "A measure whose slots hold numbers resolves to itself" begin
        for r in (DistributionallyRobustConditionalValueatRisk(),
                  DistributionallyRobustConditionalValueatRiskRange(),
                  DistributionallyRobustConditionalDrawdownatRisk())
            @test PortfolioOptimisers.resolve_deferred_quantities(r, pr) === r
        end
    end

    @testset "The radius is in the units of the series (#601)" begin
        # Scale the scenarios and every radius by c. While the support constraint is slack,
        # the dual program scales tau, s and the risk by c and keeps lambda and the weights, so
        # a radius read in any other units breaks the identity. At c = 10 the support binds on
        # this sample and the identity fails, because the support is in absolute returns.
        risk_w(X, r) = (sol = optimise(MeanRisk(; r = r, obj = MinimumRisk(),
                                                opt = JuMPOptimiser(;
                                                                    pe = prior(EmpiricalPrior(),
                                                                               X),
                                                                    slv = slv)));
                        (JuMP.value(sol.model[:risk]), sol.w))
        for c in (0.5, 2.0), (alpha, l, r) in ((0.05, 1.0, 1e-4), (0.05, 1.0, 0.02))
            for R in
                (rd -> DistributionallyRobustConditionalValueatRisk(; alpha = alpha, l = l,
                                                                    r = rd),
                 rd -> DistributionallyRobustConditionalDrawdownatRisk(; alpha = alpha,
                                                                       l = l, r = rd),
                 rd -> DistributionallyRobustConditionalValueatRiskRange(; alpha = alpha,
                                                                         l_a = l, r_a = rd,
                                                                         beta = 0.1,
                                                                         l_b = 2.0,
                                                                         r_b = 2 * rd))
                v, w = risk_w(X, R(r))
                vc, wc = risk_w(c .* X, R(c * r))
                @test isapprox(vc, c * v; rtol = 1e-6)
                @test isapprox(wc, w; atol = 1e-6)
            end
        end
        # A drawdown adds many returns, so on this sample the mean absolute drawdown of every
        # asset is larger than its mean absolute return.
        D = PortfolioOptimisers.absolute_drawdown_arr(X)
        @test all(vec(mean(abs, D; dims = 1)) .> vec(mean(abs, X; dims = 1)))
    end

    @testset "The tail means by hand at a fractional alpha T (#601)" begin
        # T = 10. At alpha = 0.2 the two largest losses, 0.04 and 0.03, form the tail. At
        # alpha = 0.25, alpha T = 2.5, so the third largest loss, 0.02, enters at half weight.
        x = [0.03, -0.01, 0.02, -0.04, 0.05, 0.0, -0.02, 0.01, -0.03, 0.04]
        @test ConditionalValueatRisk(; alpha = 0.2)(x) ≈ (0.04 + 0.03) / 2
        @test ConditionalValueatRisk(; alpha = 0.25)(x) ≈ (0.04 + 0.03 + 0.5 * 0.02) / 2.5
        # The path of the exact drawdown test of #600: the absolute drawdowns are
        # (0, 1/5, 3/20, 1/20, 3/20) and the relative ones (0, 1/5, 4/25, 19/250, 421/2500).
        # The deepest drawdown is 1/5 on both paths, and the second deepest differs.
        y = [0.1, -0.2, 0.05, 0.1, -0.1]
        for (alpha, cd, rcd) in
            ((0.2, 0.2, 0.2), (0.4, (0.2 + 0.15) / 2, (0.2 + 0.1684) / 2),
             (0.3, (0.2 + 0.5 * 0.15) / 1.5, (0.2 + 0.5 * 0.1684) / 1.5))
            @test ConditionalDrawdownatRisk(; alpha = alpha)(y) ≈ cd
            @test RelativeConditionalDrawdownatRisk(; alpha = alpha)(y) ≈ rcd
        end
    end
end

# A per-asset moment target centres a net series on its net mean (#1320). The series a moment
# measure reads is net of the fee, so a gross target `dot(w, mu)` shifted every deviation by the
# mean fee: the MAD model reported its functor plus the fee, and the Kurtosis functor disagreed
# with its fee-free model. Both levels now subtract the mean fee from a per-asset target.
using Clarabel, JuMP, Statistics

@testset "A per-asset moment target is the net mean (#1320)" begin
    rng = StableRNG(1049)
    T, N = 60, 4
    X = 0.01 .* randn(rng, T, N) .+ 0.001
    w = [0.4, 0.3, 0.2, 0.1]
    pr = prior(EmpiricalPrior(), X)
    prh = prior(HighOrderPriorEstimator(), X)
    mu = vec(mean(X; dims = 1))
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = Dict("verbose" => false))
    per = Fees(; l = 0.002)
    fixed = Fees(; l = 0.002, fl = 0.01)
    amort = Fees(; l = 0.002, fl = 0.01, fa = AmortisedFees())

    @testset "the target subtracts the mean fee per period" begin
        F(fees) = PortfolioOptimisers.term_fees(w, fees, T, true)
        mtf(m, fees) = PortfolioOptimisers.moment_target_fees(m, w, fees, T)
        # A per-asset target subtracts the mean fee; the one-off terms spread over `T`.
        @test mtf(mu, per) == F(per) == 0.002
        @test isapprox(mtf(mu, fixed), 0.002 + 4 * 0.01 / T; rtol = 1e-12)
        @test mtf(mu, amort) == mtf(mu, fixed)
        @test mtf(VecScalar(mu, 0.001), per) == F(per)
        @test iszero(mtf(mu, nothing))
        # A target stated on the net series takes no fee.
        @test iszero(mtf(nothing, per))
        @test iszero(mtf(0.001, per))
        @test iszero(mtf(MeanCentering(), per))
    end

    @testset "the prior mean agrees with the mean of the net series" begin
        # `mu` is the sample mean, so the net mean of the prior is the mean of the net series,
        # on either clock of the one-off terms.
        for fees in (nothing, per, fixed, amort)
            for r in (LowOrderMoment(), LowOrderMoment(; alg = MeanAbsoluteDeviation()),
                      LowOrderMoment(; alg = SecondMoment()),
                      LowOrderMoment(; alg = SecondMoment(; alg1 = SemiMoment())),
                      HighOrderMoment(; alg = ThirdLowerMoment()),
                      HighOrderMoment(; alg = FourthMoment()), ThirdCentralMoment())
                @test isapprox(factory(r, pr)(w, X, fees), r(w, X, fees); rtol = 1e-10)
            end
            @test isapprox(MedianAbsoluteDeviation(; mu = mu)(w, X, fees),
                           MedianAbsoluteDeviation(; mu = MeanCentering())(w, X, fees);
                           rtol = 1e-10)
        end
        # A per period fee cancels from every deviation of a per-asset target.
        r = factory(LowOrderMoment(; alg = MeanAbsoluteDeviation()), pr)
        @test isapprox(r(w, X, per), r(w, X); rtol = 1e-12)
        # A scalar target is a threshold on the net series, so the fee moves the value.
        x = X * w .- 0.002
        @test isapprox(LowOrderMoment(; mu = 0.0)(w, X, per), mean(max.(-x, 0));
                       rtol = 1e-12)
    end

    @testset "the model agrees with the functor" begin
        tn = Turnover(; w = fill(0.25, N), val = 0.001)
        for fees in (per, Fees(; l = 0.002, tn = tn))
            for (r, p) in ((LowOrderMoment(), pr),
                           (LowOrderMoment(; alg = MeanAbsoluteDeviation()), pr),
                           (LowOrderMoment(; alg = SecondMoment()), pr),
                           (LowOrderMoment(; alg = SecondMoment(; alg1 = SemiMoment())), pr),
                           (LowOrderMoment(; alg = MeanAbsoluteDeviation(), mu = mu), pr),
                           (LowOrderMoment(; mu = VecScalar(mu, 0.0)), pr),
                           (LowOrderMoment(; mu = 0.0), pr), (Kurtosis(), prh))
                sol = optimise(MeanRisk(; r = r, obj = MinimumRisk(),
                                        opt = JuMPOptimiser(; pe = p, slv = slv,
                                                            fees = fees)))
                @test isa(sol.retcode, OptimisationSuccess)
                @test isapprox(expected_risk(factory(r, p, slv), sol.w, X, fees),
                               JuMP.value(sol.model[:risk]); atol = 1e-8)
            end
        end
    end
end

# The second-moment measures of `src/16_RiskMeasures/02_Variance.jl`, checked against the forms
# their docstrings state: the functors against the closed forms, the models of the four encodings
# against the functors, the worst-case variance of every fitted set against an independent solve
# of the worst case over that set, and the factories, views and no-bounds copies of the measures.
using Clarabel, HiGHS, JuMP
import JuMP.MOI

# A Clarabel model with tolerances tight enough that a solve agrees with a closed form to 1e-8.
function tight_clarabel()
    m = Model(Clarabel.Optimizer)
    set_silent(m)
    for (k, v) in ("tol_gap_abs" => 1e-12, "tol_gap_rel" => 1e-12, "tol_feas" => 1e-12,
                   "max_iter" => 500)
        set_attribute(m, k, v)
    end
    return m
end

@testset "Variance measures" begin
    PO = PortfolioOptimisers
    rng = StableRNG(817)
    N = 4
    X = randn(rng, 250, N) *
        [1.0 0.3 0.0 0.1; 0.0 1.0 0.4 0.0; 0.0 0.0 1.0 0.2; 0.0 0.0 0.0 1.0] .* 0.01
    S = cov(X)
    w = [0.4, 0.3, 0.2, 0.1]
    W = w * transpose(w)
    rd = ReturnsResult(; X = X, nx = string.("A", 1:N))
    pr = prior(EmpiricalPrior(), rd)
    lb = S .- 0.3 .* abs.(S)
    ub = S .+ 0.2 .* abs.(S)
    box = BoxUncertaintySet(; lb = lb, ub = ub)
    Om = let B = randn(rng, N^2, N^2)
        B * transpose(B) / N^2 + I
    end
    ell = EllipsoidalUncertaintySet(; sigma = Om, k = 0.05,
                                    class = SigmaUncertaintySetClass())

    @testset "The functors are the closed forms" begin
        @test Variance(; sigma = S)(w) ≈ dot(w, S * w)
        @test StandardDeviation(; sigma = S)(w) ≈ sqrt(dot(w, S * w))
        # An estimator defines no set, so the functor gives the nominal variance.
        @test UncertaintySetVariance(; sigma = S)(w) ≈ dot(w, S * w)
        @test UncertaintySetVariance(; ucs = nothing, sigma = S)(w) ≈ dot(w, S * w)
        @test UncertaintySetVariance(; ucs = box, sigma = S)(w) ≈ PO.ucs_variance(box, S, w)
    end

    @testset "Each encoding reports the units of its functor" begin
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     settings = Dict("verbose" => false, "max_iter" => 500,
                                     "tol_gap_abs" => 1e-10, "tol_gap_rel" => 1e-10,
                                     "tol_feas" => 1e-10),
                     check_sol = (; allow_local = true, allow_almost = true))
        opt = JuMPOptimiser(; pe = pr, slv = slv)
        model_and_value(r) =
            let res = optimise(MeanRisk(; r = r, obj = MinimumRisk(), opt = opt), rd)
                (value(res.model[:risk]), expected_risk(factory(r, pr), res.w, X), res.w)
            end
        for alg in (QuadRiskExpr(), SquaredSOCRiskExpr())
            m, v, wo = model_and_value(Variance(; alg = alg))
            @test isapprox(m, v; rtol = 1e-6)
            @test v ≈ dot(wo, pr.sigma * wo)
        end
        m, v, wo = model_and_value(StandardDeviation())
        @test isapprox(m, v; rtol = 1e-6)
        @test v ≈ sqrt(dot(wo, pr.sigma * wo))
        # The three quadratic encodings report the second moment, and the cone encoding its
        # square root, in the model and in the functor alike.
        vals = Dict{Any, Float64}()
        for alg in (QuadRiskExpr(), SquaredSOCRiskExpr(), RSOCRiskExpr(), SOCRiskExpr())
            m, v, _ = model_and_value(LowOrderMoment(; alg = SecondMoment(; alg2 = alg)))
            @test isapprox(m, v; rtol = 1e-6)
            vals[typeof(alg)] = v
        end
        @test isapprox(vals[SOCRiskExpr], sqrt(vals[QuadRiskExpr]); rtol = 1e-6)
        @test isapprox(vals[RSOCRiskExpr], vals[QuadRiskExpr]; rtol = 1e-6)
    end

    @testset "ucs_variance is the worst case over the set" begin
        # Box: a linear programme over the symmetric entries between the bounds. A weight vector
        # with a short position reads the lower bound on its negative entries.
        for v in (w, [0.7, -0.2, 0.4, 0.1])
            m = Model(HiGHS.Optimizer)
            set_silent(m)
            @variable(m, lb[i, j] <= Sg[i = 1:N, j = 1:N] <= ub[i, j])
            @constraint(m, [i = 1:N, j = 1:N; i < j], Sg[i, j] == Sg[j, i])
            @objective(m, Max, dot(v * transpose(v), Sg))
            optimize!(m)
            @test isapprox(PO.ucs_variance(box, S, v), objective_value(m); rtol = 1e-10)
        end
        # The box value omits the condition that the matrix is positive semidefinite, as the
        # ellipsoid does. Corners that are not semidefinite put it above the worst case of the
        # set, and the model reaches that worst case through the slack of its lifted matrix.
        boxn = BoxUncertaintySet(; lb = [0.01 -0.9; -0.9 0.02], ub = [0.5 0.9; 0.9 0.5])
        wn = [0.5, 0.5]
        m = tight_clarabel()
        @variable(m, Sg[1:2, 1:2], PSD)
        @constraint(m, boxn.lb .<= Sg)
        @constraint(m, Sg .<= boxn.ub)
        @objective(m, Max, dot(wn * transpose(wn), Sg))
        optimize!(m)
        @test PO.ucs_variance(boxn, S[1:2, 1:2], wn) ≈ 0.7
        @test isapprox(objective_value(m), 0.5; rtol = 1e-8)
        slvn = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                      settings = Dict("verbose" => false, "max_iter" => 500,
                                      "tol_gap_abs" => 1e-10, "tol_gap_rel" => 1e-10,
                                      "tol_feas" => 1e-10),
                      check_sol = (; allow_local = true, allow_almost = true))
        resn = optimise(MeanRisk(; r = UncertaintySetVariance(; ucs = boxn),
                                 obj = MinimumRisk(),
                                 opt = JuMPOptimiser(; slv = slvn,
                                                     wb = WeightBounds(; lb = wn, ub = wn))),
                        ReturnsResult(; X = X[:, 1:2], nx = ["A1", "A2"]))
        @test isapprox(value(resn.model[:risk]), 0.5; rtol = 1e-6)
        # Ellipsoid: the value is the worst case without the condition that the matrix is
        # positive semidefinite, and it lies above the worst case with that condition.
        L = cholesky(Om).L
        function ell_primal(psd)
            m = tight_clarabel()
            @variable(m, u[1:(N ^ 2)])
            @constraint(m, [0.05; u] in SecondOrderCone())
            s = vec(S) .+ L * u
            psd && @constraint(m,
                               Symmetric(0.5 .* (reshape(s, N, N) .+ transpose(reshape(s, N, N)))) in
                               PSDCone())
            @objective(m, Max, dot(vec(W), s))
            optimize!(m)
            return objective_value(m)
        end
        @test isapprox(PO.ucs_variance(ell, S, w), ell_primal(false); rtol = 1e-7)
        @test PO.ucs_variance(ell, S, w) > ell_primal(true)
        # The set names its own centre, and `sigma` is the fallback.
        ellc = EllipsoidalUncertaintySet(; sigma = Om, k = 0.05,
                                         class = SigmaUncertaintySetClass(), val = 2 .* S)
        @test PO.ucs_variance(ellc, S, w) ≈ PO.ucs_variance(ell, 2 .* S, w)
        # Norm ball of orders 2, Inf and 1: the radius times the dual norm of the exposure.
        Lb = randn(rng, N^2, 3) .* 1e-3
        cones = Dict(2 => SecondOrderCone(), Inf => MOI.NormInfinityCone(4),
                     1 => MOI.NormOneCone(4))
        for p in (2, Inf, 1)
            nb = NormBallUncertaintySet(; kappa = 0.7, L = Lb, p = p,
                                        class = SigmaUncertaintySetClass())
            m = tight_clarabel()
            @variable(m, u[1:3])
            @constraint(m, [0.7; u] in cones[p])
            @objective(m, Max, dot(vec(W), vec(S) .+ Lb * u))
            optimize!(m)
            @test isapprox(PO.ucs_variance(nb, S, w), objective_value(m); rtol = 1e-7)
        end
        nb0 = NormBallUncertaintySet(; kappa = 0.7, L = zeros(N^2, 0), p = 2,
                                     class = SigmaUncertaintySetClass())
        @test PO.ucs_variance(nb0, S, w) ≈ dot(w, S * w)
        # Compact: the penalty is the squared norm of the part of `C w` outside the span of the
        # basis, for an orthonormal basis and for any other basis of the same span.
        Q = Matrix(qr(randn(rng, N, 2)).Q)[:, 1:2]
        C = [1.0, 0.5, 2.0, 1.5] .* 0.01
        P = I - Q * transpose(Q)
        closed = dot(w, S * w) + 3.0 * dot(C .* w, P * (C .* w))
        cpt = CompactCovarianceUncertaintySet(; kappa = 3.0, C = C, Q = Q)
        @test PO.ucs_variance(cpt, S, w) ≈ closed
        cpt2 = CompactCovarianceUncertaintySet(; kappa = 3.0, C = C,
                                               Q = Q * [2.0 1.0; 0.0 3.0])
        @test PO.ucs_variance(cpt2, S, w) ≈ closed
        cpt0 = CompactCovarianceUncertaintySet(; kappa = 3.0, C = C, Q = zeros(N, 0))
        @test PO.ucs_variance(cpt0, S, w) ≈ dot(w, S * w) + 3.0 * sum(abs2, C .* w)
    end

    @testset "The factories select a pair and keep a stated set" begin
        r = factory(Variance(), pr)
        @test r.sigma === pr.sigma && r.chol === pr.chol
        # A stated matrix is not paired with the factorisation of the prior.
        r = factory(StandardDeviation(; sigma = 2 .* S), pr)
        @test r.sigma == 2 .* S && isnothing(r.chol)
        # The set that the measure holds wins over the argument.
        usv = UncertaintySetVariance(; ucs = box)
        usv0 = UncertaintySetVariance(; ucs = nothing)
        @test factory(usv, pr, nothing, ell).ucs === box
        @test factory(usv, pr, ell).ucs === box
        @test factory(usv, ell, pr).ucs === box
        @test factory(usv0, pr, nothing, ell).ucs === ell
        @test factory(usv0, pr, ell).ucs === ell
        @test factory(usv0, pr).sigma === pr.sigma
        # Without a prior the uncertainty-set-first method keeps `sigma` as it stands.
        r = factory(usv0, ell)
        @test r.ucs === ell && isnothing(r.sigma)
        r = factory(usv0, ell, pr)
        @test r.ucs === ell && r.sigma === pr.sigma
        @test factory(UncertaintySetVariance(; ucs = nothing, sigma = S), ell, pr).sigma ===
              S
    end

    @testset "The views slice sigma on both axes and chol on its columns" begin
        G = Matrix(cholesky(S).U)
        for (T, i) in ((Variance, [1, 3]), (StandardDeviation, [2, 4]))
            rv = PO.port_opt_view(T(; sigma = S, chol = G), i)
            @test rv.sigma == S[i, i] && rv.chol == G[:, i]
        end
        rv = PO.port_opt_view(UncertaintySetVariance(; ucs = box, sigma = S), [1, 4])
        @test rv.sigma == S[[1, 4], [1, 4]] && rv.ucs.ub == ub[[1, 4], [1, 4]]
    end

    @testset "The no-bounds copies" begin
        usv = UncertaintySetVariance(; ucs = box, sigma = S,
                                     settings = RiskMeasureSettings(; ub = 1.0,
                                                                    scale = 2.0))
        # `flag` keeps the set, or drops it for the nominal Variance.
        for f in (nothing, Val(true))
            a = PO.no_bounds_risk_measure(usv, f)
            @test a isa UncertaintySetVariance && a.ucs === box && a.sigma === S
            @test isnothing(a.settings.ub) && a.settings.scale == 2 && a.settings.rke
            c = PO.no_bounds_no_risk_expr_risk_measure(usv, f)
            @test c isa UncertaintySetVariance && c.ucs === box
            @test isnothing(c.settings.ub) && isone(c.settings.scale) && !c.settings.rke
        end
        b = PO.no_bounds_risk_measure(usv, Val(false))
        @test b isa Variance && b.sigma === S
        @test isnothing(b.settings.ub) && b.settings.scale == 2 && b.settings.rke
        d = PO.no_bounds_no_risk_expr_risk_measure(usv, Val(false))
        @test d isa Variance && d.sigma === S && isnothing(d.rc)
        @test isnothing(d.settings.ub) && isone(d.settings.scale) && !d.settings.rke
        # The copy of an uncertainty-set variance matches the copy of every other measure.
        e = PO.no_bounds_no_risk_expr_risk_measure(Variance(; sigma = S,
                                                            settings = usv.settings))
        @test e.settings.scale == d.settings.scale && e.settings.rke == d.settings.rke
    end

    @testset "ucs_risk_measure fits the set on the returns" begin
        r = PO.ucs_risk_measure(UncertaintySetVariance(; sigma = S), rd)
        @test r.ucs isa BoxUncertaintySet && r.sigma === S
        # A set that reads a prior result has nothing to fit before the prior exists.
        est = NormalUncertaintySet(; pe = nothing)
        @test PO.ucs_risk_measure(UncertaintySetVariance(; ucs = est), rd).ucs === est
        rs = PO.ucs_risk_measure([UncertaintySetVariance(; ucs = box), Variance()], rd)
        @test rs[1].ucs === box && rs[2] isa Variance
    end

    @testset "An uncertainty set reads the source of its own prior (#1700)" begin
        Bv = 0.5 .* X[:, 1]
        rdb = ReturnsResult(; X = X, nx = string.("A", 1:N), nb = ["BM"], B = Bv)
        ux = DeltaUncertaintySet(; pe = EmpiricalPrior(; x_src = ExcessSource()))
        ud = DeltaUncertaintySet(; pe = EmpiricalPrior(; x_src = DataSource()))
        un = DeltaUncertaintySet(; pe = EmpiricalPrior())
        # The set gives the returns to its prior, and the prior states the source.
        @test sigma_ucs(ux, rdb).ub == sigma_ucs(un, X .- Bv).ub
        @test mu_ucs(ux, rdb).ub == mu_ucs(un, X .- Bv).ub
        @test ucs(ux, rdb)[2].ub == sigma_ucs(ux, rdb).ub
        @test sigma_ucs(ud, rdb).ub == sigma_ucs(un, rdb).ub == sigma_ucs(un, X).ub
        @test sigma_ucs(ux, rdb).ub != sigma_ucs(un, rdb).ub
        # The three-argument form and the pre-fit of the measure read the same source.
        @test sigma_ucs(ux, rdb, pr).ub == sigma_ucs(ux, rdb).ub
        @test mu_ucs(ux, rdb, pr).ub == mu_ucs(ux, rdb).ub
        @test ucs(ux, rdb, pr)[1].ub == mu_ucs(ux, rdb).ub
        @test PO.ucs_risk_measure(UncertaintySetVariance(; ucs = ux), rdb).ucs.ub ==
              sigma_ucs(ux, rdb).ub
        # An excess prior refuses returns data with no benchmark.
        @test_throws ArgumentError sigma_ucs(ux, rd)
        # No verb takes `brt`, and no `kwargs...` sink swallows it.
        for f in (ucs, mu_ucs, sigma_ucs)
            @test_throws MethodError f(un, rdb; brt = true)
            @test_throws MethodError f(un, rdb, pr; brt = true)
        end
        @test_throws MethodError PO.ucs_risk_measure(UncertaintySetVariance(; ucs = un),
                                                     rdb; brt = true)
        # The walk passes through the measure and the set to the prior of the set.
        @test PO.reader_frames(ux, "ucs", false).excess
        @test isempty(PO.reader_frames(ux, "ucs", false).raw)
        @test PO.reader_frames(un, "ucs", false).raw == ["ucs.pe"]
        noframes(fr) = !fr.excess && isempty(fr.raw)
        @test noframes(PO.reader_frames(ud, "ucs", false))
        usv = UncertaintySetVariance(; ucs = un)
        @test PO.reader_frames(usv, "r", false).raw == ["r.ucs.pe"]
        @test PO.reader_frames([Variance(), usv], "r", false).raw == ["r[2].ucs.pe"]
        rtr = RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = w), r = usv)
        @test PO.reader_frames(rtr, "r", false).raw == ["r.r.ucs.pe"]
        rte = PO.RiskTrackingError(; tr = WeightsTracking(; w = w), r = usv, err = 0.01)
        @test PO.reader_frames(rte, "tr", false).raw == ["tr.r.ucs.pe"]
        # A set that reads the Prior Result, a built set and a tracker with no reader add
        # no frames.
        for x in
            (DeltaUncertaintySet(; pe = nothing), box, UncertaintySetVariance(; ucs = box),
             TrackingRiskMeasure(; tr = WeightsTracking(; w = w)),
             TrackingError(; tr = WeightsTracking(; w = w)))
            @test noframes(PO.reader_frames(x, "x", false))
        end
    end

    @testset "A measure that holds a return term or a measure joins the walk (#1701, #1705)" begin
        noframes(fr) = !fr.excess && isempty(fr.raw)
        un = DeltaUncertaintySet(; pe = EmpiricalPrior())
        usv = UncertaintySetVariance(; ucs = un)
        art = ArithmeticReturn(; ucs = un)
        @test PO.reader_frames(ExpectedReturn(; rt = art), "r", false).raw ==
              ["r.rt.ucs.pe"]
        @test PO.reader_frames(ExpectedReturnRiskRatio(; rt = art, rk = usv), "r", false).raw ==
              ["r.rt.ucs.pe", "r.rk.ucs.pe"]
        @test PO.reader_frames(MeanReturnRiskRatio(; rk = usv), "r", false).raw ==
              ["r.rk.ucs.pe"]
        @test noframes(PO.reader_frames(ExpectedReturn(), "r", false))
        # The `pe` of a moment measure fits on the returns of the Prior Result that the
        # measure receives, so it states no source of its own. `Skewness` and `Kurtosis`
        # carry no `pe` since #1846.
        for T in (VarianceSkewKurtosis,), x_src in (DataSource(), ExcessSource())
            @test_throws ArgumentError T(; pe = EmpiricalPrior(; x_src = x_src))
            @test isa(T(; pe = EmpiricalPrior()), T)
        end
    end

    @testset "Validation" begin
        for T in (Variance, StandardDeviation, UncertaintySetVariance)
            @test_throws DimensionMismatch T(; sigma = rand(3, 4))
            @test_throws IsEmptyError T(; sigma = zeros(0, 0))
        end
        @test_throws IsEmptyError Variance(; sigma = S, chol = zeros(0, 0))
        @test_throws ArgumentError Variance(; chol = Matrix(cholesky(S).U))
    end
end

# The observation weights of the risk measures (#1621): the constructor and evaluation checks of
# `assert_observation_weights` and `checked_observation_weights`, the weighted Ulcer Index, and
# the significance level zero of the four tail measures, which is the largest loss among the
# observations with positive weight. The models are checked against the functors.
using Clarabel, JuMP

# A `DynamicAbstractWeights` that resolves to one weight too many, and one that resolves to a
# negative weight. Neither has values when the constructor runs, so evaluation must catch them.
struct LongObsWeights09p <: PortfolioOptimisers.DynamicAbstractWeights end
function PortfolioOptimisers.get_observation_weights(::LongObsWeights09p,
                                                     X::PortfolioOptimisers.VecNum_MatNum;
                                                     kwargs...)
    return pweights(ones(size(X, 1) + 1))
end
struct NegativeObsWeights09p <: PortfolioOptimisers.DynamicAbstractWeights end
function PortfolioOptimisers.get_observation_weights(::NegativeObsWeights09p,
                                                     X::PortfolioOptimisers.VecNum_MatNum;
                                                     kwargs...)
    return pweights([-1.0; ones(size(X, 1) - 1)])
end

@testset "Risk measure observation weights" begin
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = "verbose" => false)
    rng = StableRNG(7)
    x = 0.02 .* randn(rng, 40)
    w = pweights(rand(rng, 40))
    dd = PortfolioOptimisers.absolute_drawdown_vec(x)
    rdd = PortfolioOptimisers.relative_drawdown_vec(x)
    # The worst loss sits on the first observation, which carries no weight.
    xz = copy(x)
    xz[1] = -0.5
    wz = pweights([0.0; ones(39)])
    ddz = PortfolioOptimisers.absolute_drawdown_vec(xz)

    @testset "Constructors refuse invalid weights" begin
        @test_throws DomainError UlcerIndex(; w = pweights(zeros(40)))
        @test_throws DomainError RelativeUlcerIndex(; w = pweights([1.0, -1.0]))
        @test_throws DomainError ConditionalValueatRisk(; w = pweights(zeros(3)))
        @test_throws IsEmptyError AverageDrawdown(; w = pweights(Float64[]))
        @test_throws DomainError MeanReturn(; w = pweights([1.0, -2.0]))
        # `LowOrderMoment` checked only emptiness before.
        @test_throws DomainError LowOrderMoment(; w = pweights([1.0, -1.0]))
        @test_throws DomainError LowOrderMoment(; w = pweights(zeros(2)))
        @test_nowarn LowOrderMoment(; w = pweights([0.0, 1.0]))
        # A dynamic weight has no values yet, so the constructor passes it.
        @test_nowarn UlcerIndex(; w = NegativeObsWeights09p())
    end

    @testset "Evaluation refuses weights that do not fit the data" begin
        long = pweights(rand(StableRNG(1), 41))
        # The weighted quantile kernels read the weights through the sort permutation, so a
        # long weight vector used to pass in silence.
        @test_throws DimensionMismatch ConditionalValueatRisk(; w = long)(x)
        @test_throws DimensionMismatch ConditionalDrawdownatRisk(; w = long)(x)
        @test_throws DimensionMismatch ValueatRisk(; w = long)(x)
        @test_throws DimensionMismatch UlcerIndex(; w = long)(x)
        @test_throws DimensionMismatch AverageDrawdown(; w = long)(x)
        # Stored weights of a moment measure are checked against the data.
        @test_throws DimensionMismatch LowOrderMoment(; w = long)(x)
        @test_throws DimensionMismatch HighOrderMoment(; w = long)(x)
        @test_throws DimensionMismatch LowOrderMoment(; w = long)(fill(0.2, 5),
                                                                  randn(StableRNG(2), 40,
                                                                        5))
        # Dynamic weights are checked when they resolve.
        @test_throws DimensionMismatch UlcerIndex(; w = LongObsWeights09p())(x)
        @test_throws DomainError UlcerIndex(; w = NegativeObsWeights09p())(x)
        @test_throws DimensionMismatch LowOrderMoment(; w = LongObsWeights09p())(x)
        @test isnothing(PortfolioOptimisers.checked_observation_weights(nothing, x))
        @test PortfolioOptimisers.checked_observation_weights(w, x) === w
        @test isnothing(PortfolioOptimisers.assert_observation_weights(nothing, :w))
    end

    @testset "Every weighted measure refuses weights of the wrong length" begin
        # ADR 0202: every numeric site resolves its weights through
        # `checked_observation_weights`. The testset above samples the kernels; this one reaches
        # each measure that holds observation weights, one entry too many and one too few.
        long = pweights(rand(StableRNG(1), 41))
        short = pweights(rand(StableRNG(1), 39))
        ctors = (ValueatRisk, ValueatRiskRange, DrawdownatRisk, RelativeDrawdownatRisk,
                 ConditionalValueatRisk, DistributionallyRobustConditionalValueatRisk,
                 ConditionalValueatRiskRange,
                 DistributionallyRobustConditionalValueatRiskRange,
                 ConditionalDrawdownatRisk, DistributionallyRobustConditionalDrawdownatRisk,
                 RelativeConditionalDrawdownatRisk, AverageDrawdown,
                 RelativeAverageDrawdown, UlcerIndex, RelativeUlcerIndex, LowOrderMoment,
                 HighOrderMoment, MeanReturn, ThirdCentralMoment)
        slv_ctors = (EntropicValueatRisk, EntropicValueatRiskRange, EntropicDrawdownatRisk,
                     RelativeEntropicDrawdownatRisk, RelativisticValueatRisk,
                     RelativisticValueatRiskRange, RelativisticDrawdownatRisk,
                     RelativeRelativisticDrawdownatRisk, PowerNormValueatRisk,
                     PowerNormValueatRiskRange, PowerNormDrawdownatRisk,
                     RelativePowerNormDrawdownatRisk)
        for wv in (long, short)
            for ctor in ctors
                @test_throws DimensionMismatch ctor(; w = wv)(x)
            end
            for ctor in slv_ctors
                @test_throws DimensionMismatch ctor(; slv = slv, w = wv)(x)
            end
        end
    end

    @testset "Weighted Ulcer Index" begin
        @test UlcerIndex(; w = w)(x) ≈ sqrt(sum(w .* dd .^ 2) / sum(w))
        @test RelativeUlcerIndex(; w = w)(x) ≈ sqrt(sum(w .* rdd .^ 2) / sum(w))
        # Equal weights give the unweighted index.
        @test UlcerIndex(; w = pweights(fill(3.0, 40)))(x) ≈ UlcerIndex()(x)
        @test UlcerIndex()(x) ≈ sqrt(sum(dd .^ 2) / 40)
        @test RelativeUlcerIndex()(x) ≈ sqrt(sum(rdd .^ 2) / 40)
        # A zero weight drops the observation from the mean, not from the path.
        @test UlcerIndex(; w = wz)(xz) ≈ sqrt(sum(ddz[2:end] .^ 2) / 39)
        # The prior's weights reach a measure that states none.
        X = 0.01 .* randn(StableRNG(5), 40, 3)
        pr0 = prior(EmpiricalPrior(), X)
        pr = LowOrderPrior(; X = pr0.X, mu = pr0.mu, sigma = pr0.sigma, w = w)
        wp = fill(inv(3), 3)
        ddp = PortfolioOptimisers.absolute_drawdown_vec(X * wp)
        @test expected_risk(UlcerIndex(), wp, pr) ≈ sqrt(sum(w .* ddp .^ 2) / sum(w))
        @test expected_risk(UlcerIndex(), wp, pr0) ≈ sqrt(sum(ddp .^ 2) / 40)
    end

    @testset "Significance level zero" begin
        @test_throws DomainError ConditionalValueatRisk(; alpha = 1)
        @test_throws DomainError ConditionalValueatRisk(; alpha = -0.1)
        @test_throws DomainError ValueatRisk(; alpha = 0)
        @test_throws DomainError ConditionalValueatRiskRange(; alpha = 0)
        @test_throws DomainError PortfolioOptimisers.assert_half_open_unit_interval(1.0)
        @test isnothing(PortfolioOptimisers.assert_half_open_unit_interval(nothing))
        # Unweighted, level zero reads every observation.
        @test ConditionalValueatRisk(; alpha = 0)(xz) ≈ 0.5
        @test EntropicValueatRisk(; slv = slv, alpha = 0)(xz) ≈ 0.5
        @test ConditionalValueatRisk(; alpha = 0)(x) ≈ WorstRealisation()(x)
        @test ConditionalDrawdownatRisk(; alpha = 0)(x) ≈ MaximumDrawdown()(x)
        @test EntropicDrawdownatRisk(; slv = slv, alpha = 0)(x) ≈ MaximumDrawdown()(x)
        @test RelativeConditionalDrawdownatRisk(; alpha = 0)(x) ≈
              RelativeMaximumDrawdown()(x)
        @test RelativeEntropicDrawdownatRisk(; slv = slv, alpha = 0)(x) ≈
              RelativeMaximumDrawdown()(x)
        # Weighted, the zero-weight loss cannot set the risk, but it stays in the path.
        @test ConditionalValueatRisk(; alpha = 0, w = wz)(xz) ≈ -minimum(xz[2:end])
        @test EntropicValueatRisk(; slv = slv, alpha = 0, w = wz)(xz) ≈ -minimum(xz[2:end])
        @test ConditionalDrawdownatRisk(; alpha = 0, w = wz)(xz) ≈ -minimum(ddz[2:end])
        @test EntropicDrawdownatRisk(; slv = slv, alpha = 0, w = wz)(xz) ≈
              -minimum(ddz[2:end])
        @test -minimum(ddz[2:end]) > 0.4
        @test WorstRealisation()(xz) ≈ 0.5
        # Level zero bounds a small positive level from above.
        @test ConditionalValueatRisk(; alpha = 0, w = w)(x) >=
              ConditionalValueatRisk(; alpha = 0.05, w = w)(x)
    end

    @testset "JuMP models agree with the functors" begin
        X = 0.01 .* randn(StableRNG(11), 120, 4) .+ 0.0005
        # Every asset loses on the first observation, which carries no weight.
        X[1, :] .= -0.2
        rd = ReturnsResult(; nx = string.('A':'D'), X = X)
        pr = prior(EmpiricalPrior(), rd)
        wo = pweights(rand(StableRNG(3), 120))
        wo0 = pweights([0.0; ones(119)])
        opt = JuMPOptimiser(; pe = pr, slv = slv)
        function solved_risk(r)
            res = optimise(MeanRisk(; r = r, obj = MinimumRisk(), opt = opt), rd)
            @test isa(res.retcode, OptimisationSuccess)
            return JuMP.value(res.model[:risk]),
                   expected_risk(factory(r, pr, slv), res.w, pr)
        end
        for r in (UlcerIndex(), UlcerIndex(; w = wo), UlcerIndex(; w = wo0),
                  ConditionalValueatRisk(; alpha = 0),
                  ConditionalValueatRisk(; alpha = 0, w = wo0), EntropicValueatRisk(; alpha = 0),
                  EntropicValueatRisk(; alpha = 0, w = wo0),
                  ConditionalDrawdownatRisk(; alpha = 0),
                  ConditionalDrawdownatRisk(; alpha = 0, w = wo0),
                  EntropicDrawdownatRisk(; alpha = 0),
                  EntropicDrawdownatRisk(; alpha = 0, w = wo0))
            model_risk, functor_risk = solved_risk(r)
            @test isapprox(model_risk, functor_risk; rtol = 1e-6)
        end
        # Unweighted, level zero is the worst realisation and the maximum drawdown.
        @test isapprox(solved_risk(ConditionalValueatRisk(; alpha = 0))[1],
                       solved_risk(WorstRealisation())[1]; rtol = 1e-6)
        @test isapprox(solved_risk(ConditionalDrawdownatRisk(; alpha = 0))[1],
                       solved_risk(MaximumDrawdown())[1]; rtol = 1e-6)
        # The zero-weight loss of 0.2 sets the unweighted worst loss and no weighted one.
        @test solved_risk(ConditionalValueatRisk(; alpha = 0))[1] ≈ 0.2 atol = 1e-6
        @test solved_risk(ConditionalValueatRisk(; alpha = 0, w = wo0))[1] < 0.1
        @test solved_risk(EntropicValueatRisk(; alpha = 0, w = wo0))[1] < 0.1
        # At a positive level the zero-weight loss sets no risk either. Its cone rows have no
        # effect on the optimum, but the solver stalled on them before they were dropped.
        for r in (EntropicValueatRisk(; alpha = 0.2, w = wo0),
                  EntropicDrawdownatRisk(; alpha = 0.2, w = wo0),
                  RelativisticValueatRisk(; alpha = 0.2, w = wo0),
                  RelativisticDrawdownatRisk(; alpha = 0.2, w = wo0))
            model_risk, functor_risk = solved_risk(r)
            @test isapprox(model_risk, functor_risk; rtol = 1e-5)
        end
        res = optimise(MeanRisk(; r = EntropicValueatRisk(; alpha = 0.2, w = wo0),
                                obj = MinimumRisk(), opt = opt), rd)
        @test isapprox(JuMP.value(res.model[:risk]),
                       PortfolioOptimisers.ERM(X[2:end, :] * res.w, slv, 0.2); rtol = 1e-5)
        # Two Ulcer Index measures with different weights build two expressions.
        res = optimise(MeanRisk(;
                                r = [UlcerIndex(; w = wo),
                                     UlcerIndex(; w = wo0,
                                                settings = RiskMeasureSettings(;
                                                                               scale = 2.0))],
                                obj = MinimumRisk(), opt = opt), rd)
        @test isa(res.retcode, OptimisationSuccess)
        m = res.model
        @test haskey(m, :uci_risk_1) && haskey(m, :uci_risk_2)
        @test JuMP.value(m[:uci_risk_1]) ≈ UlcerIndex(; w = wo)(X * res.w) rtol = 1e-6
        @test JuMP.value(m[:uci_risk_2]) ≈ UlcerIndex(; w = wo0)(X * res.w) rtol = 1e-6
        # Weights of the wrong length are refused when the model is built.
        @test_throws DimensionMismatch optimise(MeanRisk(;
                                                         r = UlcerIndex(;
                                                                        w = pweights(ones(119))),
                                                         obj = MinimumRisk(), opt = opt),
                                                rd)
    end
end
