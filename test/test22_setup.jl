# Shared fixtures for the test_22 nested-clustered/stacking split files.
# Not a test file (no `test_` prefix) so it is excluded from discovery;
# included by each split file. See ADR 0003.
using PortfolioOptimisers, CSV, Test, TimeSeries, Clarabel, DataFrames, StableRNGs,
      Pajarito, HiGHS, JuMP, Clustering, NearestCorrelationMatrix
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
                              "reduced_tol_gap_rel" => 1e-4, "reduced_tol_ktratio" => 1e-3,
                              "reduced_tol_feas" => 1e-4, "reduced_tol_infeas_abs" => 1e-4,
                              "reduced_tol_infeas_rel" => 1e-4))]
mip_slv = [Solver(; name = :mip1,
                  solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                     "oa_solver" =>
                                                         optimizer_with_attributes(HiGHS.Optimizer,
                                                                                   JuMP.MOI.Silent() =>
                                                                                       true),
                                                     "conic_solver" =>
                                                         optimizer_with_attributes(Clarabel.Optimizer,
                                                                                   "verbose" =>
                                                                                       false)),
                  check_sol = (; allow_local = true, allow_almost = true)),
           Solver(; name = :mip2,
                  solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                     "oa_solver" =>
                                                         optimizer_with_attributes(HiGHS.Optimizer,
                                                                                   JuMP.MOI.Silent() =>
                                                                                       true),
                                                     "conic_solver" =>
                                                         optimizer_with_attributes(Clarabel.Optimizer,
                                                                                   "verbose" =>
                                                                                       false,
                                                                                   "max_step_fraction" =>
                                                                                       0.95)),
                  check_sol = (; allow_local = true, allow_almost = true)),
           Solver(; name = :mip3,
                  solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                     "oa_solver" =>
                                                         optimizer_with_attributes(HiGHS.Optimizer,
                                                                                   JuMP.MOI.Silent() =>
                                                                                       true),
                                                     "conic_solver" =>
                                                         optimizer_with_attributes(Clarabel.Optimizer,
                                                                                   "verbose" =>
                                                                                       false,
                                                                                   "max_step_fraction" =>
                                                                                       0.90)),
                  check_sol = (; allow_local = true, allow_almost = true)),
           Solver(; name = :mip4,
                  solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                     "oa_solver" =>
                                                         optimizer_with_attributes(HiGHS.Optimizer,
                                                                                   JuMP.MOI.Silent() =>
                                                                                       true),
                                                     "conic_solver" =>
                                                         optimizer_with_attributes(Clarabel.Optimizer,
                                                                                   "verbose" =>
                                                                                       false,
                                                                                   "max_step_fraction" =>
                                                                                       0.85)),
                  check_sol = (; allow_local = true, allow_almost = true)),
           Solver(; name = :mip5,
                  solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                     "oa_solver" =>
                                                         optimizer_with_attributes(HiGHS.Optimizer,
                                                                                   JuMP.MOI.Silent() =>
                                                                                       true),
                                                     "conic_solver" =>
                                                         optimizer_with_attributes(Clarabel.Optimizer,
                                                                                   "verbose" =>
                                                                                       false,
                                                                                   "max_step_fraction" =>
                                                                                       0.80)),
                  check_sol = (; allow_local = true, allow_almost = true)),
           Solver(; name = :mip6,
                  solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                     "oa_solver" =>
                                                         optimizer_with_attributes(HiGHS.Optimizer,
                                                                                   JuMP.MOI.Silent() =>
                                                                                       true),
                                                     "conic_solver" =>
                                                         optimizer_with_attributes(Clarabel.Optimizer,
                                                                                   "verbose" =>
                                                                                       false,
                                                                                   "max_step_fraction" =>
                                                                                       0.75)),
                  check_sol = (; allow_local = true, allow_almost = true)),
           Solver(; name = :mip7,
                  solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                     "oa_solver" =>
                                                         optimizer_with_attributes(HiGHS.Optimizer,
                                                                                   JuMP.MOI.Silent() =>
                                                                                       true),
                                                     "conic_solver" =>
                                                         optimizer_with_attributes(Clarabel.Optimizer,
                                                                                   "verbose" =>
                                                                                       false,
                                                                                   "max_step_fraction" =>
                                                                                       0.7)),
                  check_sol = (; allow_local = true, allow_almost = true)),
           Solver(; name = :mip8,
                  solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                     "oa_solver" =>
                                                         optimizer_with_attributes(HiGHS.Optimizer,
                                                                                   JuMP.MOI.Silent() =>
                                                                                       true),
                                                     "conic_solver" =>
                                                         optimizer_with_attributes(Clarabel.Optimizer,
                                                                                   "verbose" =>
                                                                                       false,
                                                                                   "max_step_fraction" =>
                                                                                       0.6,
                                                                                   "max_iter" =>
                                                                                       1500,
                                                                                   "tol_gap_abs" =>
                                                                                       1e-4,
                                                                                   "tol_gap_rel" =>
                                                                                       1e-4,
                                                                                   "tol_ktratio" =>
                                                                                       1e-3,
                                                                                   "tol_feas" =>
                                                                                       1e-4,
                                                                                   "tol_infeas_abs" =>
                                                                                       1e-4,
                                                                                   "tol_infeas_rel" =>
                                                                                       1e-4,
                                                                                   "reduced_tol_gap_abs" =>
                                                                                       1e-4,
                                                                                   "reduced_tol_gap_rel" =>
                                                                                       1e-4,
                                                                                   "reduced_tol_ktratio" =>
                                                                                       1e-3,
                                                                                   "reduced_tol_feas" =>
                                                                                       1e-4,
                                                                                   "reduced_tol_infeas_abs" =>
                                                                                       1e-4,
                                                                                   "reduced_tol_infeas_rel" =>
                                                                                       1e-4)),
                  check_sol = (; allow_local = true, allow_almost = true))]
sets = AssetSets(;
                 dict = Dict("nx" => rd.nx, "group1" => rd.nx[1:2:end],
                             "group2" => rd.nx[2:2:end],
                             "nx_clusters1" =>
                                 [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
                                  3],
                             "nx_clusters2" =>
                                 [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1,
                                  2], "c1" => rd.nx[1:3:end], "c2" => rd.nx[2:3:end],
                             "c3" => rd.nx[3:3:end],
                             "nx_industries" => ["Technology", "Technology", "Financials",
                                                 "Consumer_Discretionary", "Energy", "Industrials",
                                                 "Consumer_Discretionary", "Healthcare", "Financials",
                                                 "Consumer_Staples", "Healthcare", "Healthcare",
                                                 "Technology", "Consumer_Staples", "Healthcare",
                                                 "Consumer_Staples", "Energy", "Healthcare",
                                                 "Consumer_Staples", "Energy"],
                             "ux_industries" =>
                                 ["Technology", "Financials", "Consumer_Discretionary",
                                  "Energy", "Industrials", "Healthcare",
                                  "Consumer_Staples"]))
fsets = AssetSets(; dict = Dict("nx" => rd.nf))
pr = prior(HighOrderPriorEstimator(), rd)
rr = regression(DimensionReductionRegression(), rd)
clr = clusterise(ClustersEstimator(), pr)
w0 = fill(inv(size(pr.X, 2)), size(pr.X, 2))
