@safetestset "NestedClustered" begin
    using PortfolioOptimisers, CSV, Test, TimeSeries, Clarabel, DataFrames, StableRNGs,
          Pajarito, HiGHS, JuMP, Clustering, NearestCorrelationMatrix
    function find_tol(a1, a2; name1 = :lhs, name2 = :rhs)
        for rtol in
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; rtol = rtol)
                println("isapprox($name1, $name2, rtol = $(rtol))")
                break
            end
        end
        for atol in
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; atol = atol)
                println("isapprox($name1, $name2, atol = $(atol))")
                break
            end
        end
    end
    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end],
                           TimeArray(CSV.File(joinpath(@__DIR__, "./assets/Factors.csv.gz"));
                                     timestamp = :Date)[(end - 252):end];
                           B = TimeArray(CSV.File(joinpath(@__DIR__,
                                                           "./assets/SP500_idx.csv.gz"));
                                         timestamp = :Date))
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
    mip_slv = [Solver(; name = :mip1,
                      solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                         "verbose" => false,
                                                         "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                  JuMP.MOI.Silent() => true),
                                                         "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                     "verbose" => false)),
                      check_sol = (; allow_local = true, allow_almost = true)),
               Solver(; name = :mip2,
                      solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                         "verbose" => false,
                                                         "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                  JuMP.MOI.Silent() => true),
                                                         "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                     "verbose" => false,
                                                                                                     "max_step_fraction" => 0.95)),
                      check_sol = (; allow_local = true, allow_almost = true)),
               Solver(; name = :mip3,
                      solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                         "verbose" => false,
                                                         "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                  JuMP.MOI.Silent() => true),
                                                         "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                     "verbose" => false,
                                                                                                     "max_step_fraction" => 0.90)),
                      check_sol = (; allow_local = true, allow_almost = true)),
               Solver(; name = :mip4,
                      solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                         "verbose" => false,
                                                         "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                  JuMP.MOI.Silent() => true),
                                                         "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                     "verbose" => false,
                                                                                                     "max_step_fraction" => 0.85)),
                      check_sol = (; allow_local = true, allow_almost = true)),
               Solver(; name = :mip5,
                      solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                         "verbose" => false,
                                                         "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                  JuMP.MOI.Silent() => true),
                                                         "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                     "verbose" => false,
                                                                                                     "max_step_fraction" => 0.80)),
                      check_sol = (; allow_local = true, allow_almost = true)),
               Solver(; name = :mip6,
                      solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                         "verbose" => false,
                                                         "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                  JuMP.MOI.Silent() => true),
                                                         "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                     "verbose" => false,
                                                                                                     "max_step_fraction" => 0.75)),
                      check_sol = (; allow_local = true, allow_almost = true)),
               Solver(; name = :mip7,
                      solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                         "verbose" => false,
                                                         "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                  JuMP.MOI.Silent() => true),
                                                         "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                     "verbose" => false,
                                                                                                     "max_step_fraction" => 0.7)),
                      check_sol = (; allow_local = true, allow_almost = true)),
               Solver(; name = :mip8,
                      solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                         "verbose" => false,
                                                         "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                  JuMP.MOI.Silent() => true),
                                                         "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                     "verbose" => false,
                                                                                                     "max_step_fraction" => 0.6,
                                                                                                     "max_iter" => 1500,
                                                                                                     "tol_gap_abs" => 1e-4,
                                                                                                     "tol_gap_rel" => 1e-4,
                                                                                                     "tol_ktratio" => 1e-3,
                                                                                                     "tol_feas" => 1e-4,
                                                                                                     "tol_infeas_abs" => 1e-4,
                                                                                                     "tol_infeas_rel" => 1e-4,
                                                                                                     "reduced_tol_gap_abs" => 1e-4,
                                                                                                     "reduced_tol_gap_rel" => 1e-4,
                                                                                                     "reduced_tol_ktratio" => 1e-3,
                                                                                                     "reduced_tol_feas" => 1e-4,
                                                                                                     "reduced_tol_infeas_abs" => 1e-4,
                                                                                                     "reduced_tol_infeas_rel" => 1e-4)),
                      check_sol = (; allow_local = true, allow_almost = true))]
    sets = AssetSets(;
                     dict = Dict("nx" => rd.nx, "group1" => rd.nx[1:2:end],
                                 "group2" => rd.nx[2:2:end],
                                 "nx_clusters1" => [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                                                    2, 3, 3, 3, 3, 3, 3],
                                 "nx_clusters2" => [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1,
                                                    2, 3, 1, 2, 3, 1, 2],
                                 "c1" => rd.nx[1:3:end], "c2" => rd.nx[2:3:end],
                                 "c3" => rd.nx[3:3:end],
                                 "nx_industries" => ["Technology", "Technology",
                                                     "Financials", "Consumer_Discretionary",
                                                     "Energy", "Industrials",
                                                     "Consumer_Discretionary", "Healthcare",
                                                     "Financials", "Consumer_Staples",
                                                     "Healthcare", "Healthcare",
                                                     "Technology", "Consumer_Staples",
                                                     "Healthcare", "Consumer_Staples",
                                                     "Energy", "Healthcare",
                                                     "Consumer_Staples", "Energy"],
                                 "ux_industries" => ["Technology", "Financials",
                                                     "Consumer_Discretionary", "Energy",
                                                     "Industrials", "Healthcare",
                                                     "Consumer_Staples"]))
    pr = prior(HighOrderPriorEstimator(), rd)
    rr = regression(DimensionReductionRegression(), rd)
    clr = clusterise(ClustersEstimator(), pr)
    w0 = fill(inv(size(pr.X, 2)), size(pr.X, 2))
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
                NestedClustered(; cle = clr, opti = EqualWeighted(),
                                opto = EqualWeighted()),
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
                                                        HierarchicalRiskParity(;
                                                                               opt = hopti),
                                                        InverseVolatility(; pe = pr),
                                                        EqualWeighted(),
                                                        NestedClustered(; pe = pr,
                                                                        opti = NearOptimalCentering(;
                                                                                                    opt = jopti),
                                                                        opto = NearOptimalCentering(;
                                                                                                    opt = jopto))],
                                                opto = Stacking(;
                                                                opti = [MeanRisk(;
                                                                                 opt = jopto),
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
                                                        HierarchicalRiskParity(;
                                                                               opt = hopto),
                                                        InverseVolatility(),
                                                        EqualWeighted(),
                                                        NestedClustered(;
                                                                        opti = NearOptimalCentering(;
                                                                                                    opt = jopto),
                                                                        opto = NearOptimalCentering(;
                                                                                                    opt = jopto))],
                                                opto = Stacking(;
                                                                opti = [MeanRisk(;
                                                                                 opt = jopto),
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
                                opti = HierarchicalRiskParity(;
                                                              r = MedianAbsoluteDeviation(),
                                                              opt = hopti),
                                opto = HierarchicalRiskParity(;
                                                              r = MedianAbsoluteDeviation(),
                                                              opt = hopto)),
                NestedClustered(; cle = clr,
                                opti = FactorRiskContribution(; flag = true, re = rr,
                                                              opt = jopti),
                                opto = FactorRiskContribution(; flag = true, opt = jopto)),
                NestedClustered(;
                                cle = ClustersEstimator(;
                                                        ce = PortfolioOptimisersCovariance(),
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
                                opti = MeanRisk(; opt = jopto),
                                opto = MeanRisk(; opt = jopto)),
                NestedClustered(;
                                cv = OptimisationCrossValidation(;
                                                                 cv = IndexWalkForward(23,
                                                                                       11)),
                                opti = MeanRisk(; opt = jopto),
                                opto = MeanRisk(; opt = jopto)),
                NestedClustered(; cv = OptimisationCrossValidation(; cv = KFold(; n = 10)),
                                opti = Stacking(;
                                                cv = OptimisationCrossValidation(;
                                                                                 cv = KFold(;
                                                                                            n = 10)),
                                                opti = [MeanRisk(; opt = jopto),
                                                        HierarchicalRiskParity(;
                                                                               opt = hopto),
                                                        MeanRisk(; obj = MaximumRatio(),
                                                                 opt = jopto)],
                                                opto = MeanRisk(; opt = jopto)),
                                opto = Stacking(;
                                                cv = OptimisationCrossValidation(;
                                                                                 cv = IndexWalkForward(23,
                                                                                                       11)),
                                                opti = [MeanRisk(; opt = jopto),
                                                        HierarchicalRiskParity(;
                                                                               opt = hopto),
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
                                                        HierarchicalRiskParity(;
                                                                               opt = hopto),
                                                        MeanRisk(; obj = MaximumRatio(),
                                                                 opt = jopto)],
                                                opto = MeanRisk(; opt = jopto)),
                                opto = Stacking(;
                                                cv = OptimisationCrossValidation(;
                                                                                 cv = CombinatorialCrossValidation(;
                                                                                                                   n_folds = 4,
                                                                                                                   n_test_folds = 3)),
                                                opti = [MeanRisk(; opt = jopto),
                                                        HierarchicalRiskParity(;
                                                                               opt = hopto),
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
                Stacking(;
                         cv = OptimisationCrossValidation(; cv = IndexWalkForward(23, 11)),
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
                                                        onc = OptimalNumberClusters(;
                                                                                    alg = 3)),
                                opti = SubsetResampling(; rng = StableRNG(100), seed = 42,
                                                        pe = pr,
                                                        opt = MeanRisk(; opt = jopti)),
                                opto = SubsetResampling(; rng = StableRNG(100), seed = 42,
                                                        opt = MeanRisk(; opt = jopto))),
                Stacking(;
                         opti = [SubsetResampling(; rng = StableRNG(100), seed = 42,
                                                  pe = pr, opt = MeanRisk(; opt = jopti)),
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
    @testset "Fees" begin
        fees = FeesEstimator(;
                             tn = TurnoverEstimator(; w = w0,
                                                    val = Dict("PG" => 0.5 / 252)),
                             l = Dict("MRK" => 0.3 / 252), s = "BAC" => 0.2 / 252,
                             fl = ["XOM" => 0.5 / 252], fs = "PFE" => 0.3 / 252)
        opti = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1,
                             wb = WeightBounds(; lb = -1, ub = 1), fees = fees, sets = sets)
        opto = JuMPOptimiser(; slv = slv, sbgt = 1, bgt = 1,
                             wb = WeightBounds(; lb = -1, ub = 1))
        res = optimise(NestedClustered(; cle = clr,
                                       opti = MeanRisk(; r = ConditionalDrawdownatRisk(),
                                                       opt = opti),
                                       opto = MeanRisk(; opt = opto)), rd)

        clusters = assignments(clr)
        idx = findfirst(x -> x == "PG", rd.nx)
        idxc = clusters[idx]
        idx = findall(x -> x == idxc, clusters)
        idx = findfirst(x -> x == "PG", rd.nx[idx])
        @test isapprox(res.resi[idxc].w[idx], 0.05)
        @test isapprox(res.w[findfirst(x -> x == "MRK", rd.nx)], 0)
        @test isapprox(res.w[findfirst(x -> x == "BAC", rd.nx)], 0)
        @test isapprox(res.w[findfirst(x -> x == "PFE", rd.nx)], 0)

        opti = JuMPOptimiser(; pe = pr, slv = mip_slv, sbgt = 1, bgt = 1,
                             wb = WeightBounds(; lb = -1, ub = 1),
                             fees = fees_constraints(fees, sets), sets = sets)
        @test isapprox(res.w,
                       optimise(NestedClustered(; cle = clr,
                                                opti = MeanRisk(;
                                                                r = ConditionalDrawdownatRisk(),
                                                                opt = opti),
                                                opto = MeanRisk(; opt = opto)), rd).w)
    end
    @testset "Advanced use" begin
        res = optimise(NestedClustered(; cle = clr,
                                       opti = MeanRisk(; r = ConditionalValueatRisk(),
                                                       opt = JuMPOptimiser(; pe = pr,
                                                                           slv = mip_slv,
                                                                           scard = [2, 1],
                                                                           smtx = concrete_typed_array([AssetSetsMatrixEstimator(;
                                                                                                                                 val = "nx_clusters1"),
                                                                                                        asset_sets_matrix(AssetSetsMatrixEstimator(;
                                                                                                                                                   val = "nx_clusters2"),
                                                                                                                          sets)]),
                                                                           sets = sets)),
                                       opto = MeanRisk(; opt = JuMPOptimiser(; slv = slv))),
                       rd)

        @test sum(.!iszero.([res.resi[1].w[res.resi[1].smtx[1][i, :]]
                             for i in axes(res.resi[1].smtx[1], 1)])) < 3
        @test sum(.!iszero.([res.resi[1].w[res.resi[1].smtx[2][i, :]]
                             for i in axes(res.resi[1].smtx[2], 1)])) < 2

        @test sum(.!iszero.([res.resi[2].w[res.resi[2].smtx[1][i, :]]
                             for i in axes(res.resi[2].smtx[1], 1)])) < 3
        @test sum(.!iszero.([res.resi[2].w[res.resi[2].smtx[2][i, :]]
                             for i in axes(res.resi[2].smtx[2], 1)])) < 2

        opt = NestedClustered(; cle = clr,
                              opti = MeanRisk(; r = ConditionalValueatRisk(),
                                              opt = JuMPOptimiser(; pe = pr, slv = mip_slv,
                                                                  lt = ThresholdEstimator(;
                                                                                          val = ["WMT" => 0.2,
                                                                                                 "group2" => 0.48]),
                                                                  sets = sets)),
                              opto = MeanRisk(; opt = JuMPOptimiser(; slv = slv)))
        res = optimise(opt, rd)
        clusters = assignments(clr)

        group2 = sets.dict["group2"]
        for i in 1:(clr.k)
            nx = rd.nx[findall(x -> x == i, clusters)]
            idx = findfirst(x -> x == "WMT", nx)
            if !isnothing(idx)
                @test isapprox(res.resi[i].w[idx], 0.2; rtol = 1e-6)
            end
            idx = [findfirst(x -> x == i, nx) for i in group2]
            filter!(!isnothing, idx)
            if !isempty(idx)
                for w in res.resi[i].w[idx]
                    if abs(w) > sqrt(20) * sqrt(eps(w))
                        @test w > 0.48 - sqrt(eps(w))
                    end
                end
            end
        end
        opt = NestedClustered(; cle = clr,
                              opti = MeanRisk(; r = ConditionalValueatRisk(),
                                              opt = JuMPOptimiser(; pe = pr, slv = mip_slv,
                                                                  lt = threshold_constraints(ThresholdEstimator(;
                                                                                                                val = ["WMT" => 0.2,
                                                                                                                       "group2" => 0.48]),
                                                                                             sets))),
                              opto = MeanRisk(; opt = JuMPOptimiser(; slv = slv)))
        @test isapprox(res.w, optimise(opt, rd).w)
    end
    @testset "Risk measure views" begin
        ucse = NormalUncertaintySet(; pe = EmpiricalPrior(), rng = StableRNG(987654321),
                                    alg = BoxUncertaintySetAlgorithm())
        ucs = sigma_ucs(ucse, rd.X)
        jopti = JuMPOptimiser(; pe = pr, slv = slv, sets = sets)
        jopto = JuMPOptimiser(; slv = slv,
                              pe = HighOrderPriorEstimator(;
                                                           ske = Coskewness(;
                                                                            mp = DenoiseDetoneAlgMatrixProcessing(;
                                                                                                                  pdm = nothing))))

        resa = optimise(NestedClustered(; cle = clr,
                                        opti = MeanRisk(; r = Kurtosis(; mu = pr.mu),
                                                        opt = jopti),
                                        opto = MeanRisk(; r = Kurtosis(), opt = jopto)), rd)
        resb = optimise(NestedClustered(; cle = clr,
                                        opti = MeanRisk(; r = Kurtosis(; kt = pr.kt),
                                                        opt = jopti),
                                        opto = MeanRisk(; r = Kurtosis(), opt = jopto)), rd)
        @test resa.w == resb.w

        resa = optimise(NestedClustered(; cle = clr,
                                        opti = MeanRisk(;
                                                        r = UncertaintySetVariance(;
                                                                                   ucs = ucse),
                                                        opt = jopti),
                                        opto = MeanRisk(;
                                                        r = UncertaintySetVariance(;
                                                                                   ucs = ucse),
                                                        opt = jopto)), rd)
        resb = optimise(NestedClustered(; cle = clr,
                                        opti = MeanRisk(;
                                                        r = UncertaintySetVariance(;
                                                                                   ucs = ucs),
                                                        opt = jopti),
                                        opto = MeanRisk(;
                                                        r = UncertaintySetVariance(;
                                                                                   ucs = ucse),
                                                        opt = jopto)), rd)
        @test resa.w != resb.w

        resa = optimise(NestedClustered(; cle = clr,
                                        opti = MeanRisk(;
                                                        r = LowOrderMoment(;
                                                                           alg = MeanAbsoluteDeviation()),
                                                        opt = jopti),
                                        opto = MeanRisk(;
                                                        r = LowOrderMoment(;
                                                                           alg = MeanAbsoluteDeviation()),
                                                        opt = jopto)), rd)
        resb = optimise(NestedClustered(; cle = clr,
                                        opti = MeanRisk(;
                                                        r = LowOrderMoment(; mu = pr.mu,
                                                                           alg = MeanAbsoluteDeviation()),
                                                        opt = jopti),
                                        opto = MeanRisk(;
                                                        r = LowOrderMoment(;
                                                                           alg = MeanAbsoluteDeviation()),
                                                        opt = jopto)), rd)
        @test resa.w == resb.w

        resa = optimise(NestedClustered(; cle = clr,
                                        opti = MeanRisk(; r = NegativeSkewness(;),
                                                        opt = jopti),
                                        opto = MeanRisk(; r = NegativeSkewness(;),
                                                        opt = jopto)), rd)
        resb = optimise(NestedClustered(; cle = clr,
                                        opti = MeanRisk(;
                                                        r = NegativeSkewness(; sk = pr.sk,
                                                                             V = pr.V),
                                                        opt = jopti),
                                        opto = MeanRisk(; r = NegativeSkewness(),
                                                        opt = jopto)), rd)
        @test resa.w == resb.w

        res = optimise(NestedClustered(; cle = clr,
                                       opti = MeanRisk(; r = ValueatRisk(),
                                                       opt = JuMPOptimiser(; pe = pr,
                                                                           slv = mip_slv,
                                                                           sets = sets)),
                                       opto = MeanRisk(; r = ValueatRisk(),
                                                       opt = JuMPOptimiser(; slv = mip_slv))),
                       rd)
        if Sys.isapple()
            @test isapprox(res.w,
                ([0.0, 0.0, 0.004821476046698469, 0.0010258936044387464, 0.0, 0.0, 0.006537279371877492, 0.18088433701440654, 0.005298988291077931, 0.3472528756459372, 0.0, 0.2402405305712111, 0.00033257997539444, 0.05000349773341354, 0.03242526200848679, 0.0, 0.0, 0.0, 0.1278243058653844, 0.0033529738716733854], rtol = 1e-6)
        else
            @test isapprox(res.w,
                           [0.0, 0.0, 0.005602412808720073, 0.0011920580781122523, 0.0, 0.0,
                            0.007596125612252459, 0.2375180560536035, 0.006157267937796995,
                            0.3435374666842433, 0.011940000819969961, 0.20302112082396714,
                            0.0003864481117456734, 0.0, 0.04878243497008711, 0.0, 0.0, 0.0,
                            0.1303705514595793, 0.0038960566399224694], rtol = 1e-6)
        end

        res = optimise(NestedClustered(; cle = clr,
                                       opti = MeanRisk(; r = DrawdownatRisk(),
                                                       opt = JuMPOptimiser(; pe = pr,
                                                                           slv = mip_slv,
                                                                           sets = sets)),
                                       opto = MeanRisk(; r = DrawdownatRisk(),
                                                       opt = JuMPOptimiser(; slv = mip_slv))),
                       rd)
        @test isapprox(res.w,
            [0.0012234700209226132, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0005385255722520739, 0.281628194040886, 0.0, 0.029236643246224468, 0.13170514517674964, 0.16228108868342814, 0.0, 0.23112812156090545, 0.0, 0.0, 0.0, 0.09158493164822426, 0.06963119505448694, 0.0010426849959224247], rtol = 1e-6)

        resa = optimise(NestedClustered(; cle = clr,
                                        opti = MeanRisk(;
                                                        r = ValueatRisk(;
                                                                        alg = DistributionValueatRisk()),
                                                        opt = JuMPOptimiser(; pe = pr,
                                                                            slv = slv,
                                                                            sets = sets)),
                                        opto = MeanRisk(;
                                                        r = ValueatRisk(;
                                                                        alg = DistributionValueatRisk()),
                                                        opt = JuMPOptimiser(; slv = slv))),
                        rd)
        resb = optimise(NestedClustered(; cle = clr,
                                        opti = MeanRisk(;
                                                        r = ValueatRisk(;
                                                                        alg = DistributionValueatRisk(;
                                                                                                      mu = pr.mu,
                                                                                                      sigma = pr.sigma)),
                                                        opt = JuMPOptimiser(; pe = pr,
                                                                            slv = slv,
                                                                            sets = sets)),
                                        opto = MeanRisk(;
                                                        r = ValueatRisk(;
                                                                        alg = DistributionValueatRisk()),
                                                        opt = JuMPOptimiser(; slv = slv))),
                        rd)
        @test resa.w == resb.w

        res = optimise(NestedClustered(; cle = clr,
                                       opti = MeanRisk(; r = ValueatRiskRange(),
                                                       opt = JuMPOptimiser(; pe = pr,
                                                                           slv = mip_slv,
                                                                           sets = sets)),
                                       opto = MeanRisk(; r = ValueatRiskRange(),
                                                       opt = JuMPOptimiser(; slv = mip_slv))),
                       rd)
        @test isapprox(res.w,
            [0.00021148772894721179, 0.0, 0.0, 5.503641791343221e-5, 0.0, 0.0003957305274730832, 0.0011574175816340929, 0.0, 0.002724254691574924, 0.19286877092707, 0.16396114472674364, 0.0, 0.0, 0.3005855067078481, 0.0, 0.0, 0.0, 0.0, 0.3352086932338209, 0.0028319574569745654], rtol = 1e-6)

        resa = optimise(NestedClustered(; cle = clr,
                                        opti = MeanRisk(;
                                                        r = ValueatRiskRange(;
                                                                             alg = DistributionValueatRisk()),
                                                        opt = JuMPOptimiser(; pe = pr,
                                                                            slv = slv,
                                                                            sets = sets)),
                                        opto = MeanRisk(;
                                                        r = ValueatRiskRange(;
                                                                             alg = DistributionValueatRisk()),
                                                        opt = JuMPOptimiser(; slv = slv))),
                        rd)
        resb = optimise(NestedClustered(; cle = clr,
                                        opti = MeanRisk(;
                                                        r = ValueatRiskRange(;
                                                                             alg = DistributionValueatRisk(;
                                                                                                           mu = pr.mu,
                                                                                                           sigma = pr.sigma)),
                                                        opt = JuMPOptimiser(; pe = pr,
                                                                            slv = slv,
                                                                            sets = sets)),
                                        opto = MeanRisk(;
                                                        r = ValueatRiskRange(;
                                                                             alg = DistributionValueatRisk()),
                                                        opt = JuMPOptimiser(; slv = slv))),
                        rd)
        @test resa.w == resb.w

        res = optimise(NestedClustered(; cle = clr,
                                       opti = MeanRisk(; r = TurnoverRiskMeasure(; w = w0),
                                                       opt = jopti),
                                       opto = MeanRisk(;
                                                       r = TurnoverRiskMeasure(;
                                                                               w = fill(1 /
                                                                                        2,
                                                                                        2)),
                                                       opt = jopto)), rd)
        @test isapprox(res.w,
                       [0.045454545461020436, 0.04545454545389754, 0.04545454545389754,
                        0.04545454545389754, 0.04545454545389754, 0.04545454545389754,
                        0.04545454545389754, 0.055555555478081366, 0.04545454545389754,
                        0.05555555556523971, 0.05555555556523971, 0.05555555556523971,
                        0.04545454545389754, 0.05555555556523971, 0.05555555556523971,
                        0.05555555556523971, 0.04545454545389754, 0.05555555556523971,
                        0.05555555556523971, 0.04545454545389754], rtol = 1e-6)
    end
    @testset "Efficient frontier" begin
        mr1 = MeanRisk(; opt = JuMPOptimiser(; pe = pr, slv = slv))
        mr2 = MeanRisk(; obj = MaximumRatio(), opt = JuMPOptimiser(; pe = pr, slv = slv))
        mr3 = MeanRisk(;
                       opt = JuMPOptimiser(;
                                           ret = ArithmeticReturn(;
                                                                  lb = Frontier(; N = 10)),
                                           slv = slv))
        nco = NestedClustered(; opti = mr1, opto = mr3)
        res = optimise(nco, rd)
        df = CSV.read(joinpath(@__DIR__,
                               "./assets/NestedClusteredEfficientFrontier.csv.gz"),
                      DataFrame)
        success = isapprox(Matrix(df), reduce(hcat, res.w); rtol = 1e-3)
        if !success
            find_tol(Matrix(df), reduce(hcat, res.w))
        end
        @test success

        st = Stacking(; opti = [mr1, mr2], opto = mr3)
        res = optimise(st, rd)
        df = CSV.read(joinpath(@__DIR__, "./assets/StackingEfficientFrontier.csv.gz"),
                      DataFrame)
        success = isapprox(Matrix(df), reduce(hcat, res.w); rtol = 5e-4)
        if !success
            find_tol(Matrix(df), reduce(hcat, res.w))
        end
        @test success
    end
end
