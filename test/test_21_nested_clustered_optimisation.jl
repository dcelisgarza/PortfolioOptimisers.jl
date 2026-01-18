@safetestset "NestedClustered" begin
    using PortfolioOptimisers, CSV, Test, TimeSeries, Clarabel, DataFrames, StableRNGs,
          Pajarito, HiGHS, JuMP, Clustering
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
                                     timestamp = :Date)[(end - 252):end])
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
        jopti = JuMPOptimiser(; pr = pr, slv = slv, sets = sets)
        jopto = JuMPOptimiser(; slv = slv)
        hopti = HierarchicalOptimiser(; pr = pr, slv = slv)
        hopto = HierarchicalOptimiser(; slv = slv)
        opts = [NestedClustered(; clr = clr, opti = MeanRisk(; opt = jopti),
                                opto = MeanRisk(; opt = jopto)),
                NestedClustered(; clr = clr, opti = NearOptimalCentering(; opt = jopti),
                                opto = NearOptimalCentering(; opt = jopto)),
                NestedClustered(; clr = clr,
                                opti = RiskBudgeting(;
                                                     rba = AssetRiskBudgeting(;
                                                                              rkb = RiskBudgetEstimator(;
                                                                                                        val = UniformValues())),
                                                     opt = jopti),
                                opto = RiskBudgeting(; opt = jopto)),
                NestedClustered(; clr = clr, opti = RelaxedRiskBudgeting(; opt = jopti),
                                opto = RelaxedRiskBudgeting(; opt = jopto)),
                NestedClustered(; clr = clr, opti = HierarchicalRiskParity(; opt = hopti),
                                opto = HierarchicalRiskParity(; opt = hopto)),
                NestedClustered(; clr = clr,
                                opti = HierarchicalEqualRiskContribution(; opt = hopti),
                                opto = HierarchicalEqualRiskContribution(; opt = hopto)),
                NestedClustered(; pr = pr, clr = clr,
                                opti = NestedClustered(; pr = pr,
                                                       opti = MeanRisk(; opt = jopti),
                                                       opto = MeanRisk(; opt = jopto)),
                                opto = NestedClustered(; opti = MeanRisk(; opt = jopto),
                                                       opto = MeanRisk(; opt = jopto))),
                NestedClustered(; clr = clr, opti = InverseVolatility(; pr = pr),
                                opto = InverseVolatility()),
                NestedClustered(; clr = clr, opti = EqualWeighted(),
                                opto = EqualWeighted()),
                NestedClustered(; clr = clr,
                                opti = RandomWeighted(; rng = StableRNG(1234567890)),
                                opto = RandomWeighted(; rng = StableRNG(0987654321))),
                NestedClustered(; clr = clr,
                                opti = SchurComplementHierarchicalRiskParity(;
                                                                             params = SchurComplementParams(;
                                                                                                            gamma = 0),
                                                                             opt = hopti),
                                opto = SchurComplementHierarchicalRiskParity(;
                                                                             params = SchurComplementParams(;
                                                                                                            gamma = 0.5),
                                                                             opt = hopto)),
                NestedClustered(; pr = pr, clr = clr,
                                opti = Stacking(;
                                                opti = [MeanRisk(; opt = jopti),
                                                        HierarchicalRiskParity(;
                                                                               opt = hopti),
                                                        InverseVolatility(; pr = pr),
                                                        EqualWeighted(),
                                                        NestedClustered(; pr = pr,
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
                NestedClustered(; clr = clr, opti = FactorRiskContribution(; opt = jopti),
                                opto = FactorRiskContribution(; opt = jopto)),
                NestedClustered(; clr = clr,
                                opti = HierarchicalRiskParity(;
                                                              r = MedianAbsoluteDeviation(),
                                                              opt = hopti),
                                opto = HierarchicalRiskParity(;
                                                              r = MedianAbsoluteDeviation(),
                                                              opt = hopto)),
                NestedClustered(; clr = clr,
                                opti = FactorRiskContribution(; re = rr, opt = jopti),
                                opto = FactorRiskContribution(; opt = jopto)),
                NestedClustered(;
                                clr = ClustersEstimator(;
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
                                                                              rkb = RiskBudgetResult(;
                                                                                                     val = fill(inv(size(pr.X,
                                                                                                                         2)),
                                                                                                                size(pr.X,
                                                                                                                     2)))),
                                                     opt = jopti),
                                opto = MeanRisk(; opt = jopto)),
                NestedClustered(; pr = pr, clr = clr,
                                opti = NestedClustered(; pr = pr,
                                                       opti = NestedClustered(; pr = pr,
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
                                                                                              opt = jopto))))]
        df = CSV.read(joinpath(@__DIR__, "./assets/NestedClustered.csv.gz"), DataFrame)
        for (i, opt) in enumerate(opts)
            res = optimise(opt, rd)
            if i == 3
                @test isapprox(optimise(NestedClustered(; clr = clr,
                                                        opti = RiskBudgeting(; opt = jopti),
                                                        opto = RiskBudgeting(; opt = jopto)),
                                        rd).w, res.w)
            end
            rtol = if i in (2, 3, 16)
                5e-5
            elseif i == 12
                5e-6
            elseif i == 10
                1.1
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
        opti = JuMPOptimiser(; pr = pr, slv = mip_slv, sbgt = 1, bgt = 1,
                             wb = WeightBounds(; lb = -1, ub = 1), fees = fees, sets = sets)
        opto = JuMPOptimiser(; slv = slv, sbgt = 1, bgt = 1,
                             wb = WeightBounds(; lb = -1, ub = 1))
        res = optimise(NestedClustered(; clr = clr,
                                       opti = MeanRisk(; r = ConditionalDrawdownatRisk(),
                                                       opt = opti),
                                       opto = MeanRisk(; opt = opto)), rd)

        clusters = get_clustering_indices(clr)
        idx = findfirst(x -> x == "PG", rd.nx)
        idxc = clusters[idx]
        idx = findall(x -> x == idxc, clusters)
        idx = findfirst(x -> x == "PG", rd.nx[idx])
        @test isapprox(res.resi[idxc].w[idx], 0.05)
        @test isapprox(res.w[findfirst(x -> x == "MRK", rd.nx)], 0)
        @test isapprox(res.w[findfirst(x -> x == "BAC", rd.nx)], 0)
        @test isapprox(res.w[findfirst(x -> x == "PFE", rd.nx)], 0)

        opti = JuMPOptimiser(; pr = pr, slv = mip_slv, sbgt = 1, bgt = 1,
                             wb = WeightBounds(; lb = -1, ub = 1),
                             fees = fees_constraints(fees, sets), sets = sets)
        @test isapprox(res.w,
                       optimise(NestedClustered(; clr = clr,
                                                opti = MeanRisk(;
                                                                r = ConditionalDrawdownatRisk(),
                                                                opt = opti),
                                                opto = MeanRisk(; opt = opto)), rd).w)
    end
    @testset "Advanced use" begin
        res = optimise(NestedClustered(; clr = clr,
                                       opti = MeanRisk(; r = ConditionalValueatRisk(),
                                                       opt = JuMPOptimiser(; pr = pr,
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

        @test sum(.!iszero.([res.resi[1].w[res.resi[1].smtx[1][i, :]] for i in axes(res.resi[1].smtx[1], 1)])) < 3
        @test sum(.!iszero.([res.resi[1].w[res.resi[1].smtx[2][i, :]] for i in axes(res.resi[1].smtx[2], 1)])) < 2

        @test sum(.!iszero.([res.resi[2].w[res.resi[2].smtx[1][i, :]] for i in axes(res.resi[2].smtx[1], 1)])) < 3
        @test sum(.!iszero.([res.resi[2].w[res.resi[2].smtx[2][i, :]] for i in axes(res.resi[2].smtx[2], 1)])) < 2

        opt = NestedClustered(; clr = clr,
                              opti = MeanRisk(; r = ConditionalValueatRisk(),
                                              opt = JuMPOptimiser(; pr = pr, slv = mip_slv,
                                                                  lt = ThresholdEstimator(;
                                                                                          val = ["WMT" => 0.2,
                                                                                                 "group2" => 0.48]),
                                                                  sets = sets)),
                              opto = MeanRisk(; opt = JuMPOptimiser(; slv = slv)))
        res = optimise(opt, rd)
        clusters = get_clustering_indices(clr)

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
        opt = NestedClustered(; clr = clr,
                              opti = MeanRisk(; r = ConditionalValueatRisk(),
                                              opt = JuMPOptimiser(; pr = pr, slv = mip_slv,
                                                                  lt = threshold_constraints(ThresholdEstimator(;
                                                                                                                val = ["WMT" => 0.2,
                                                                                                                       "group2" => 0.48]),
                                                                                             sets))),
                              opto = MeanRisk(; opt = JuMPOptimiser(; slv = slv)))
        @test isapprox(res.w, optimise(opt, rd).w)
    end
end
