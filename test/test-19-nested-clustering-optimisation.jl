@safetestset "NestedClustering" begin
    using PortfolioOptimisers, CSV, Test, TimeSeries, Clarabel, DataFrames, StableRNGs
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
    pr = prior(HighOrderPriorEstimator(), rd)
    clr = clusterise(ClusteringEstimator(), pr)
    @testset "Mix optimisers" begin
        jopti = JuMPOptimiser(; pe = pr, slv = slv)
        jopto = JuMPOptimiser(; slv = slv)
        hopti = HierarchicalOptimiser(; pe = pr, slv = slv)
        hopto = HierarchicalOptimiser(; slv = slv)
        opts = [NestedClustering(; cle = clr, opti = MeanRisk(; opt = jopti),
                                 opto = MeanRisk(; opt = jopto)),
                NestedClustering(; cle = clr, opti = NearOptimalCentering(; opt = jopti),
                                 opto = NearOptimalCentering(; opt = jopto)),
                NestedClustering(; cle = clr, opti = RiskBudgetting(; opt = jopti),
                                 opto = RiskBudgetting(; opt = jopto)),
                NestedClustering(; cle = clr, opti = RelaxedRiskBudgetting(; opt = jopti),
                                 opto = RelaxedRiskBudgetting(; opt = jopto)),
                NestedClustering(; cle = clr, opti = HierarchicalRiskParity(; opt = hopti),
                                 opto = HierarchicalRiskParity(; opt = hopto)),
                NestedClustering(; cle = clr,
                                 opti = HierarchicalEqualRiskContribution(; opt = hopti),
                                 opto = HierarchicalEqualRiskContribution(; opt = hopto)),
                NestedClustering(; pe = pr, cle = clr,
                                 opti = NestedClustering(; pe = pr,
                                                         opti = MeanRisk(; opt = jopti),
                                                         opto = MeanRisk(; opt = jopto)),
                                 opto = NestedClustering(; opti = MeanRisk(; opt = jopto),
                                                         opto = MeanRisk(; opt = jopto))),
                NestedClustering(; cle = clr, opti = InverseVolatility(; pe = pr),
                                 opto = InverseVolatility()),
                NestedClustering(; cle = clr, opti = EqualWeighted(),
                                 opto = EqualWeighted()),
                NestedClustering(; cle = clr,
                                 opti = RandomWeights(; rng = StableRNG(1234567890)),
                                 opto = RandomWeights(; rng = StableRNG(0987654321))),
                NestedClustering(; cle = clr,
                                 opti = SchurHierarchicalRiskParity(;
                                                                    params = SchurParams(;
                                                                                         gamma = 0),
                                                                    opt = hopti),
                                 opto = SchurHierarchicalRiskParity(;
                                                                    params = SchurParams(;
                                                                                         gamma = 0.5),
                                                                    opt = hopto)),
                NestedClustering(; pe = pr, cle = clr,
                                 opti = Stacking(;
                                                 opti = [MeanRisk(; opt = jopti),
                                                         HierarchicalRiskParity(;
                                                                                opt = hopti),
                                                         InverseVolatility(; pe = pr),
                                                         EqualWeighted(),
                                                         NestedClustering(; pe = pr,
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
                                                                         NestedClustering(;
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
                                                         NestedClustering(;
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
                                                                         NestedClustering(;
                                                                                          opti = NearOptimalCentering(;
                                                                                                                      opt = jopto),
                                                                                          opto = NearOptimalCentering(;
                                                                                                                      opt = jopto))],
                                                                 opto = HierarchicalRiskParity(;
                                                                                               opt = hopto))))]
        df = CSV.read(joinpath(@__DIR__, "./assets/NestedClustering.csv.gz"), DataFrame)
        for (i, opt) in enumerate(opts)
            res = optimise!(opt, rd)
            rtol = if i ∈ (2, 3)
                5e-5
            elseif i == 4 || Sys.isapple() && i == 12
                5e-6
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
        #=
        opt = NestedClustering(; pe = pr, cle = clr,
                               opti = NestedClustering(; pe = pr,
                                                       opti = NestedClustering(; pe = pr,
                                                                               opti = MeanRisk(;
                                                                                               opt = jopti),
                                                                               opto = MeanRisk(;
                                                                                               opt = jopto)),
                                                       opto = NestedClustering(;
                                                                               opti = MeanRisk(;
                                                                                               opt = jopto),
                                                                               opto = MeanRisk(;
                                                                                               opt = jopto))),
                               opto = NestedClustering(;
                                                       opti = NestedClustering(;
                                                                               opti = MeanRisk(;
                                                                                               opt = jopto),
                                                                               opto = MeanRisk(;
                                                                                               opt = jopto)),
                                                       opto = NestedClustering(;
                                                                               opti = MeanRisk(;
                                                                                               opt = jopto),
                                                                               opto = MeanRisk(;
                                                                                               opt = jopto))))
        res = optimise!(opt, rd)
        =#
    end
end
