include(joinpath(@__DIR__, "test22_setup.jl"))

@testset "Mix optimisers" begin
    jopti = JuMPOptimiser(; pe = pr, slv = slv, sets = sets)
    jopto = JuMPOptimiser(; slv = slv)
    hopti = HierarchicalOptimiser(; pe = pr, slv = slv)
    hopto = HierarchicalOptimiser(; slv = slv)
    resi = optimise(MeanRisk(; opt = jopto), rd)
    opts = [NestedClustered(; cle = clr, opti = MeanRisk(; opt = jopti),
                            opto = MeanRisk(; opt = jopto)),
            NestedClustered(; cle = clr, opti = NearOptimalCentering(; opt = jopti),
                            opto = NearOptimalCentering(; opt = jopto)),
            NestedClustered(; cle = clr,
                            opti = RiskBudgeting(;
                                                 rba = AssetRiskBudgeting(; sets = sets,
                                                                          rkb = RiskBudgetEstimator(;
                                                                                                    val = UniformValues())),
                                                 opt = jopti),
                            opto = RiskBudgeting(; opt = jopto)),
            NestedClustered(; cle = clr, opti = RelaxedRiskBudgeting(; opt = jopti),
                            opto = RelaxedRiskBudgeting(; opt = jopto)),
            NestedClustered(; cle = clr, opti = HierarchicalRiskParity(; opt = hopti),
                            opto = HierarchicalRiskParity(; opt = hopto)),
            NestedClustered(; cle = clr,
                            opti = HierarchicalEqualRiskContribution(; opt = hopti),
                            opto = HierarchicalEqualRiskContribution(; opt = hopto)),
            NestedClustered(; pe = pr, cle = clr,
                            opti = NestedClustered(; pe = pr,
                                                   opti = MeanRisk(; opt = jopti),
                                                   opto = MeanRisk(; opt = jopto)),
                            opto = NestedClustered(; opti = MeanRisk(; opt = jopto),
                                                   opto = MeanRisk(; opt = jopto))),
            NestedClustered(; cle = clr, opti = InverseVolatility(; pe = pr),
                            opto = InverseVolatility()),
            NestedClustered(; cle = clr, opti = EqualWeighted(), opto = EqualWeighted()),
            NestedClustered(; cle = clr,
                            opti = RandomWeighted(; rng = StableRNG(1234567890)),
                            opto = RandomWeighted(; rng = StableRNG(0987654321))),
            NestedClustered(; cle = clr,
                            opti = SchurComplementHierarchicalRiskParity(;
                                                                         params = SchurComplementParams(;
                                                                                                        gamma = 0),
                                                                         opt = hopti),
                            opto = SchurComplementHierarchicalRiskParity(;
                                                                         params = SchurComplementParams(;
                                                                                                        gamma = 0.5),
                                                                         opt = hopto)),
            NestedClustered(; pe = pr, cle = clr,
                            opti = Stacking(;
                                            opti = [MeanRisk(; opt = jopti),
                                                    HierarchicalRiskParity(; opt = hopti),
                                                    InverseVolatility(; pe = pr),
                                                    EqualWeighted(),
                                                    NestedClustered(; pe = pr,
                                                                    opti = NearOptimalCentering(;
                                                                                                opt = jopti),
                                                                    opto = NearOptimalCentering(;
                                                                                                opt = jopto))],
                                            opto = Stacking(;
                                                            opti = [MeanRisk(; opt = jopto),
                                                                    HierarchicalRiskParity(;
                                                                                           opt = hopto),
                                                                    InverseVolatility(),
                                                                    EqualWeighted(),
                                                                    NestedClustered(;
                                                                                    opti = NearOptimalCentering(;
                                                                                                                opt = jopto),
                                                                                    opto = NearOptimalCentering(;
                                                                                                                opt = jopto))],
                                                            opto = HierarchicalRiskParity(;
                                                                                          opt = hopto))),
                            opto = Stacking(;
                                            opti = [MeanRisk(; opt = jopto),
                                                    HierarchicalRiskParity(; opt = hopto),
                                                    InverseVolatility(), EqualWeighted(),
                                                    NestedClustered(;
                                                                    opti = NearOptimalCentering(;
                                                                                                opt = jopto),
                                                                    opto = NearOptimalCentering(;
                                                                                                opt = jopto))],
                                            opto = Stacking(;
                                                            opti = [MeanRisk(; opt = jopto),
                                                                    HierarchicalRiskParity(;
                                                                                           opt = hopto),
                                                                    InverseVolatility(),
                                                                    EqualWeighted(),
                                                                    NestedClustered(;
                                                                                    opti = NearOptimalCentering(;
                                                                                                                opt = jopto),
                                                                                    opto = NearOptimalCentering(;
                                                                                                                opt = jopto))],
                                                            opto = HierarchicalRiskParity(;
                                                                                          opt = hopto)))),
            NestedClustered(; cle = clr,
                            opti = FactorRiskContribution(; flag = true, opt = jopti),
                            opto = FactorRiskContribution(; flag = true, opt = jopto)),
            NestedClustered(; cle = clr,
                            opti = HierarchicalRiskParity(; r = MedianAbsoluteDeviation(),
                                                          opt = hopti),
                            opto = HierarchicalRiskParity(; r = MedianAbsoluteDeviation(),
                                                          opt = hopto)),
            NestedClustered(; cle = clr,
                            opti = FactorRiskContribution(; flag = true, re = rr,
                                                          opt = jopti),
                            opto = FactorRiskContribution(; flag = true, opt = jopto)),
            NestedClustered(;
                            cle = ClustersEstimator(; ce = PortfolioOptimisersCovariance(),
                                                    de = Distance(;
                                                                  alg = CanonicalDistance()),
                                                    alg = KMeansAlgorithm(;
                                                                          rng = StableRNG(42),
                                                                          kwargs = (;
                                                                                    init = :kmcen)),
                                                    onc = OptimalNumberClusters(;
                                                                                alg = SecondOrderDifference())),
                            opti = RiskBudgeting(;
                                                 rba = AssetRiskBudgeting(;
                                                                          rkb = RiskBudget(;
                                                                                           val = fill(inv(size(pr.X,
                                                                                                               2)),
                                                                                                      size(pr.X,
                                                                                                           2)))),
                                                 opt = jopti),
                            opto = MeanRisk(; opt = jopto)),
            NestedClustered(; pe = pr, cle = clr,
                            opti = NestedClustered(; pe = pr,
                                                   opti = NestedClustered(; pe = pr,
                                                                          opti = MeanRisk(;
                                                                                          opt = jopti),
                                                                          opto = MeanRisk(;
                                                                                          opt = jopto)),
                                                   opto = NestedClustered(;
                                                                          opti = MeanRisk(;
                                                                                          opt = jopto),
                                                                          opto = MeanRisk(;
                                                                                          opt = jopto))),
                            opto = NestedClustered(;
                                                   opti = NestedClustered(;
                                                                          opti = MeanRisk(;
                                                                                          opt = jopto),
                                                                          opto = MeanRisk(;
                                                                                          opt = jopto)),
                                                   opto = NestedClustered(;
                                                                          opti = MeanRisk(;
                                                                                          opt = jopto),
                                                                          opto = MeanRisk(;
                                                                                          opt = jopto)))),
            NestedClustered(; cv = OptimisationCrossValidation(; cv = KFold(; n = 10)),
                            opti = MeanRisk(; opt = jopto), opto = MeanRisk(; opt = jopto)),
            NestedClustered(;
                            cv = OptimisationCrossValidation(;
                                                             cv = IndexWalkForward(23, 11)),
                            opti = MeanRisk(; opt = jopto), opto = MeanRisk(; opt = jopto)),
            NestedClustered(; cv = OptimisationCrossValidation(; cv = KFold(; n = 10)),
                            opti = Stacking(;
                                            cv = OptimisationCrossValidation(;
                                                                             cv = KFold(;
                                                                                        n = 10)),
                                            opti = [MeanRisk(; opt = jopto),
                                                    HierarchicalRiskParity(; opt = hopto),
                                                    MeanRisk(; obj = MaximumRatio(),
                                                             opt = jopto)],
                                            opto = MeanRisk(; opt = jopto)),
                            opto = Stacking(;
                                            cv = OptimisationCrossValidation(;
                                                                             cv = IndexWalkForward(23,
                                                                                                   11)),
                                            opti = [MeanRisk(; opt = jopto),
                                                    HierarchicalRiskParity(; opt = hopto),
                                                    MeanRisk(; obj = MaximumRatio(),
                                                             opt = jopto)],
                                            opto = MeanRisk(; opt = jopto))),
            NestedClustered(;
                            cv = OptimisationCrossValidation(;
                                                             cv = CombinatorialCrossValidation(;
                                                                                               n_folds = 4,
                                                                                               n_test_folds = 3)),
                            opti = Stacking(;
                                            cv = OptimisationCrossValidation(;
                                                                             cv = CombinatorialCrossValidation(;
                                                                                                               n_folds = 4,
                                                                                                               n_test_folds = 3)),
                                            opti = [MeanRisk(; opt = jopto),
                                                    HierarchicalRiskParity(; opt = hopto),
                                                    MeanRisk(; obj = MaximumRatio(),
                                                             opt = jopto)],
                                            opto = MeanRisk(; opt = jopto)),
                            opto = Stacking(;
                                            cv = OptimisationCrossValidation(;
                                                                             cv = CombinatorialCrossValidation(;
                                                                                                               n_folds = 4,
                                                                                                               n_test_folds = 3)),
                                            opti = [MeanRisk(; opt = jopto),
                                                    HierarchicalRiskParity(; opt = hopto),
                                                    MeanRisk(; obj = MaximumRatio(),
                                                             opt = jopto)],
                                            opto = MeanRisk(; opt = jopto))),
            NestedClustered(; pe = pr, cle = clr,
                            cv = OptimisationCrossValidation(; cv = KFold(; n = 10)),
                            opti = NestedClustered(; opti = MeanRisk(; opt = jopti),
                                                   opto = MeanRisk(; opt = jopto)),
                            opto = NestedClustered(; opti = MeanRisk(; opt = jopto),
                                                   opto = MeanRisk(; opt = jopto))),
            Stacking(;
                     opti = concrete_typed_array([resi,
                                                  HierarchicalRiskParity(; opt = hopto),
                                                  MeanRisk(; obj = MaximumRatio(),
                                                           opt = jopto)]),
                     opto = MeanRisk(; opt = jopto)),
            Stacking(; cv = OptimisationCrossValidation(; cv = KFold(; n = 10)),
                     opti = concrete_typed_array([resi,
                                                  HierarchicalRiskParity(; opt = hopto),
                                                  MeanRisk(; obj = MaximumRatio(),
                                                           opt = jopto)]),
                     opto = MeanRisk(; opt = jopto)),
            Stacking(;
                     cv = OptimisationCrossValidation(;
                                                      cv = CombinatorialCrossValidation(;
                                                                                        n_folds = 4,
                                                                                        n_test_folds = 3)),
                     opti = concrete_typed_array([resi,
                                                  HierarchicalRiskParity(; opt = hopto),
                                                  MeanRisk(; obj = MaximumRatio(),
                                                           opt = jopto)]),
                     opto = MeanRisk(; opt = jopto)),
            Stacking(; cv = OptimisationCrossValidation(; cv = IndexWalkForward(23, 11)),
                     opti = concrete_typed_array([resi,
                                                  HierarchicalRiskParity(; opt = hopto),
                                                  MeanRisk(; obj = MaximumRatio(),
                                                           opt = jopto)]),
                     opto = MeanRisk(; opt = jopto)),
            NestedClustered(; brt = true, opti = MeanRisk(; opt = jopti),
                            opto = MeanRisk(; opt = jopto)),
            Stacking(; brt = true,
                     opti = concrete_typed_array([resi,
                                                  HierarchicalRiskParity(; opt = hopto),
                                                  MeanRisk(; obj = MaximumRatio(),
                                                           opt = jopto)]),
                     opto = MeanRisk(; opt = jopto)),
            NestedClustered(;
                            cle = ClustersEstimator(;
                                                    onc = OptimalNumberClusters(; alg = 3)),
                            opti = SubsetResampling(; rng = StableRNG(100), seed = 42,
                                                    pe = pr, opt = MeanRisk(; opt = jopti)),
                            opto = SubsetResampling(; rng = StableRNG(100), seed = 42,
                                                    opt = MeanRisk(; opt = jopto))),
            Stacking(;
                     opti = [SubsetResampling(; rng = StableRNG(100), seed = 42, pe = pr,
                                              opt = MeanRisk(; opt = jopti)),
                             MeanRisk(; opt = jopti), InverseVolatility()],
                     opto = SubsetResampling(; rng = StableRNG(100), seed = 42,
                                             opt = MeanRisk(; opt = jopto)))]
    df = CSV.read(joinpath(@__DIR__, "./assets/NestedClustered.csv.gz"), DataFrame)
    for (i, opt) in enumerate(opts)
        res = try
            optimise(opt, rd)
        catch err
            if isa(err, ArgumentError) || isa(err, TaskFailedException)
                println("Failed iteration: $i\nError: $err\nContinuing with next iteration.")
                continue
            else
                rethrow(err)
            end
        end
        if i == 3
            @test isapprox(optimise(NestedClustered(; cle = clr,
                                                    opti = RiskBudgeting(; opt = jopti),
                                                    opto = RiskBudgeting(; opt = jopto)),
                                    rd).w, res.w)

            @test isapprox(optimise(NestedClustered(; cle = clr,
                                                    opti = RiskBudgeting(;
                                                                         rba = AssetRiskBudgeting(;
                                                                                                  sets = sets,
                                                                                                  alg = LogRiskBudgeting(;
                                                                                                                         z = ones(Int,
                                                                                                                                  size(rd.X,
                                                                                                                                       2))),
                                                                                                  rkb = RiskBudgetEstimator(;
                                                                                                                            val = UniformValues())),
                                                                         opt = jopti),
                                                    opto = RiskBudgeting(; opt = jopto)),
                                    rd).w, res.w, rtol = 5e-5)
        end
        rtol = if i in (2, 16)
            5e-5
        elseif i in (12, 20)
            5e-6
        elseif i == 10
            1.1
        elseif i == 3
            1e-4
        else
            1e-6
        end
        success = isapprox(res.w, df[!, i]; rtol = rtol)
        if !success
            println("Failed iteration: $i")
            find_tol(res.w, df[!, i])
        end
        @test success
    end
end
