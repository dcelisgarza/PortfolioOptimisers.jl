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
    sets = UniverseSets(; dict = Dict("nx" => rd.nf))
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

@testset "Factor attribution reads the original returns matrix" begin
    using Test, PortfolioOptimisers, StableRNGs, LinearAlgebra, Clarabel

    rng = StableRNG(987654321)
    T, N, Nf = 400, 8, 3
    F = randn(rng, T, Nf) .* 0.02
    B = randn(rng, N, Nf)
    X = F * transpose(B) .+ randn(rng, T, N) .* 0.01
    rd = ReturnsResult(; X = X, nx = string.(1:N), F = F, nf = string.(1:Nf))
    pr = prior(FactorPrior(), rd)
    w = fill(inv(N), N)
    r = ConditionalValueatRisk()

    # `pr.X` is the reconstruction: rank `Nf + 1`, the factors plus the intercept, and no
    # residual. Reducing it leaves the off-factor term nothing to attribute but the
    # intercept's share, which came out negative.
    @test !isnothing(pr.o_X)
    @test pr.original_X === X
    @test rank(pr.X) == Nf + 1
    @test rank(pr.original_X) == N

    # The attribution reads `original_X`, so a prior and the caller's matrix agree.
    frc_pr = factor_risk_contribution(r, w, pr; rd = rd)
    frc_X = factor_risk_contribution(r, w, X; rd = rd)
    @test frc_pr ≈ frc_X
    @test frc_pr[end] > 0
    @test sum(frc_pr) ≈ expected_risk(r, w, X)

    # The price: the parts no longer sum to the risk the prior asserts.
    @test !isapprox(sum(frc_pr), expected_risk(r, w, pr))

    # The case the carrier is needed for: a precomputed `Regression` needs no data, so `rd`
    # is empty and `pr.original_X` is the only source of the caller's returns.
    rr = regression(StepwiseRegression(), rd)
    @test factor_risk_contribution(r, w, pr; re = rr) ≈ frc_X

    # With no `rd` and no precomputed result, the prior's own `rr` supplies the loadings.
    @test factor_risk_contribution(r, w, pr) ≈ frc_X
    @test size(pr.rr.M) == size(rr.M) == (N, Nf)

    # None of the three carriers holds loadings, so it refuses.
    @test_throws IsNothingError factor_risk_contribution(r, w, X)

    # A measure whose kernel reads a moment never touches the returns matrix, so its
    # attribution was already correct and does not move.
    v = Variance()
    @test factor_risk_contribution(v, w, pr; rd = rd) ≈
          factor_risk_contribution(factory(v, pr), w, X; rd = rd)
    @test factor_risk_contribution(v, w, pr; rd = rd)[end] > 0

    # Off a factor route `original_X === X`, so nothing moves.
    prE = prior(EmpiricalPrior(), rd)
    @test isnothing(prE.o_X)
    @test isnothing(prE.rr)
    @test factor_risk_contribution(r, w, prE; rd = rd) ≈
          factor_risk_contribution(r, w, prE.X; rd = rd)

    # The optimiser follows the same precedence, so a factor prior answers what used to be a
    # throw: no returns data at all, and a regression estimator.
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = Dict("verbose" => false))
    frc = FactorRiskContribution(; r = Variance(), obj = MinimumRisk(),
                                 opt = JuMPOptimiser(; pe = pr, slv = slv),
                                 sets = UniverseSets(; dict = Dict("nx" => rd.nf)))
    res_rd = optimise(frc, rd)
    res_no = optimise(frc, ReturnsResult())
    @test isa(res_no.retcode, OptimisationSuccess)
    @test res_rd.w ≈ res_no.w

    # A prior that carries no factor block still needs the data.
    frc_e = FactorRiskContribution(; r = Variance(), obj = MinimumRisk(),
                                   opt = JuMPOptimiser(; pe = prE, slv = slv),
                                   sets = UniverseSets(; dict = Dict("nx" => rd.nf)))
    @test_throws IsNothingError optimise(frc_e, ReturnsResult())
end
