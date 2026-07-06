@testset "Factor risk contribution" begin
    using Test, PortfolioOptimisers, DataFrames, CSV, TimeSeries, Clarabel
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
    pr = prior(EmpiricalPrior(), rd)
    sets = AssetSets(; dict = Dict("nx" => rd.nf))
    lcs = LinearConstraintEstimator(; val = ["VLUE <= 0.74", "QUAL >= -0.07", "MTUM==0.09"])
    opt = JuMPOptimiser(; pe = pr, slv = slv)
    r = Variance(; rc = lcs)
    obj = MaximumRatio()
    frc = FactorRiskContribution(; r = r, obj = obj, opt = opt, sets = sets)
    res = optimise(frc, rd)
    rkc = factor_risk_contribution(factory(r, pr, slv), res.w, pr.X; rd = rd)
    rkc = rkc / sum(rkc)
    @test rkc[2] >= -0.07
    @test rkc[5] <= 0.74
    @test isapprox(rkc[1], 0.09, rtol = 5e-5)

    res = optimise(FactorRiskContribution(; r = [ConditionalValueatRisk(), Variance()],
                                          wi = range(; start = inv(size(rd.F, 2)),
                                                     stop = inv(size(rd.F, 2)),
                                                     length = size(rd.F, 2)),
                                          opt = JuMPOptimiser(; pe = pr,
                                                              slv = Solver(;
                                                                           solver = Clarabel.Optimizer,
                                                                           settings = ["verbose" =>
                                                                                           false,
                                                                                       "max_iter" =>
                                                                                           1])),
                                          fb = InverseVolatility(; pe = pr)), rd)
    @test isapprox(res.w, optimise(InverseVolatility(; pe = pr)).w)

    # A risk upper bound is not supported by FactorRiskContribution: it must warn
    # instead of silently ignoring the bound.
    logger = SimpleLogger()
    with_logger(logger) do
        @test_logs (:warn, r"Risk upper bound") match_mode = :any optimise(FactorRiskContribution(;
                                                                                                  r = ConditionalValueatRisk(;
                                                                                                                             settings = RiskMeasureSettings(;
                                                                                                                                                            ub = 1.0)),
                                                                                                  opt = JuMPOptimiser(;
                                                                                                                      pe = pr,
                                                                                                                      slv = slv)),
                                                                           rd)
    end
    res = optimise(FactorRiskContribution(;
                                          r = ConditionalValueatRisk(;
                                                                     settings = RiskMeasureSettings(;
                                                                                                    ub = 1.0)),
                                          opt = JuMPOptimiser(; pe = pr, slv = slv)), rd)
    @test isa(res.retcode, OptimisationSuccess)
end
