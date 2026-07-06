# Shared fixtures for the test_17 clustering-optimisation split files.
# Not a test file (no `test_` prefix) so it is excluded from discovery;
# included by each test_17*.jl split file. See ADR 0003.
using PortfolioOptimisers, CSV, Test, TimeSeries, Clarabel, DataFrames, FLoops
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
                              "reduced_tol_gap_rel" => 1e-4, "reduced_tol_ktratio" => 1e-3,
                              "reduced_tol_feas" => 1e-4, "reduced_tol_infeas_abs" => 1e-4,
                              "reduced_tol_infeas_rel" => 1e-4))]
pr = prior(HighOrderPriorEstimator(), rd)
clr = clusterise(ClustersEstimator(), pr)
w0 = range(; start = inv(size(pr.X, 2)), stop = inv(size(pr.X, 2)), length = size(pr.X, 2))
opt = HierarchicalOptimiser(; pe = pr, cle = clr, slv = slv)
rs = [EqualRisk(), Variance(), StandardDeviation(), UncertaintySetVariance(),
      LowOrderMoment(), HighOrderMoment(), Kurtosis(), NegativeSkewness(), ValueatRisk(),
      ValueatRiskRange(), ConditionalValueatRisk(),
      DistributionallyRobustConditionalValueatRisk(), ConditionalValueatRiskRange(),
      DistributionallyRobustConditionalValueatRiskRange(), EntropicValueatRisk(),
      EntropicValueatRiskRange(), EntropicDrawdownatRisk(),
      RelativeEntropicDrawdownatRisk(), RelativisticValueatRisk(),
      RelativisticValueatRiskRange(), RelativisticDrawdownatRisk(),
      RelativeRelativisticDrawdownatRisk(), AverageDrawdown(), RelativeAverageDrawdown(),
      TurnoverRiskMeasure(; w = w0), TrackingRiskMeasure(; tr = WeightsTracking(; w = w0)),
      RiskTrackingRiskMeasure(; r = StandardDeviation(), tr = WeightsTracking(; w = w0)),
      TrackingRiskMeasure(; tr = WeightsTracking(; w = w0), alg = L1Norm()),
      RiskTrackingRiskMeasure(; r = StandardDeviation(), tr = WeightsTracking(; w = w0),
                              alg = DependentVariableTracking()), RiskRatio(),
      MedianAbsoluteDeviation(),
      TrackingRiskMeasure(; tr = WeightsTracking(; w = w0), alg = LpNorm()),
      TrackingRiskMeasure(; tr = WeightsTracking(; w = w0), alg = LpNorm(; p = -10)),
      TrackingRiskMeasure(; tr = WeightsTracking(; w = w0), alg = LInfNorm(; pos = false)),
      TrackingRiskMeasure(; tr = WeightsTracking(; w = w0), alg = LpNorm(; p = 10)),
      TrackingRiskMeasure(; tr = WeightsTracking(; w = w0), alg = LInfNorm())]
