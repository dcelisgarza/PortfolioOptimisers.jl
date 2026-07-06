# Shared fixtures for the test_16 risk-budgeting split files.
# Not a test file (no `test_` prefix) so it is excluded from discovery;
# included by each split file. See ADR 0003.
using Test, PortfolioOptimisers, DataFrames, CSV, TimeSeries, Clarabel, StatsBase, JuMP,
      Pajarito, HiGHS
rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                 timestamp = :Date)[(end - 252):end],
                       TimeArray(CSV.File(joinpath(@__DIR__, "./assets/Factors.csv.gz"));
                                 timestamp = :Date)[(end - 252):end])
slv = [Solver(; name = :clarabel6, solver = Clarabel.Optimizer,
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
pr = prior(HighOrderPriorEstimator(), rd)
w0 = range(; start = inv(size(pr.X, 2)), stop = inv(size(pr.X, 2)), length = size(pr.X, 2))
wp = StatsBase.pweights(range(; start = inv(size(pr.X, 1)), stop = inv(size(pr.X, 1)),
                              length = size(pr.X, 1)))
rf = 4.2 / 100 / 252
rs = [StandardDeviation(), Variance(), LowOrderMoment(),
      LowOrderMoment(; alg = SecondMoment(; alg1 = SemiMoment(), alg2 = SOCRiskExpr())),
      LowOrderMoment(; alg = SecondMoment(; alg1 = SemiMoment())),
      LowOrderMoment(; alg = SecondMoment(; alg2 = SOCRiskExpr())),
      LowOrderMoment(; alg = SecondMoment()),
      LowOrderMoment(; alg = MeanAbsoluteDeviation()), WorstRealisation(), Range(),
      ConditionalValueatRisk(), ConditionalValueatRiskRange(), EntropicValueatRisk(),
      EntropicValueatRiskRange(), RelativisticValueatRisk(), RelativisticValueatRiskRange(),
      MaximumDrawdown(), AverageDrawdown(), UlcerIndex(), ConditionalDrawdownatRisk(),
      EntropicDrawdownatRisk(), RelativisticDrawdownatRisk(), Kurtosis(; N = 2), Kurtosis(),
      OrderedWeightsArray(; alg = ExactOrderedWeightsArray()), OrderedWeightsArray(),
      OrderedWeightsArrayRange(), NegativeSkewness(),
      NegativeSkewness(; alg = SquaredSOCRiskExpr())]
sets = AssetSets(;
                 dict = Dict("nx" => rd.nx, "group1" => rd.nx[1:2:end],
                             "group2" => rd.nx[2:2:end],
                             "clusters1" =>
                                 [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
                                  3],
                             "clusters2" =>
                                 [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1,
                                  2], "c1" => rd.nx[1:3:end], "c2" => rd.nx[2:3:end],
                             "c3" => rd.nx[3:3:end]))
fsets = AssetSets(; dict = Dict("nx" => rd.nf))
