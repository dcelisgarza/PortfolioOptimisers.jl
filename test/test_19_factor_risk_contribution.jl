@testset "Factor risk contribution" begin
    using Test, PortfolioOptimisers, DataFrames, CSV, TimeSeries, Clarabel, LinearAlgebra
    import JuMP
    rd = prices_to_returns(price_ingestion(PriceIngestion(),
                                           TimeArray(CSV.File(joinpath(@__DIR__,
                                                                       "./assets/SP500.csv.gz"));
                                                     timestamp = :Date)[(end - 252):end];
                                           F = TimeArray(CSV.File(joinpath(@__DIR__,
                                                                           "./assets/Factors.csv.gz"));
                                                         timestamp = :Date)[(end - 252):end]))
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

    # The rows sit on the semidefinite relaxation of formulation 16 of Cajas (2025, SSRN
    # 5097869). They bind the lifted matrix `frc_W`, and they bind the portfolio only where
    # `frc_W == w1 * w1'`. On this panel the minimum-risk and maximum-utility solves leave a
    # second eigenvalue in `frc_W`, so the realised shares miss the rows. The docstrings of
    # `Variance` and `FactorRiskContribution` state this. The paper's formulation, solved on
    # the same `sigma`, `mu` and loadings, returns the same weights and the same shares.
    for (obj, shares) in ((MinimumRisk(), [0.053, 0.0954, -0.6071, 0.9024, 0.5563]),
                          (MaximumUtility(), [0.1053, -0.1321, -0.4216, 0.6472, 0.8012]))
        resr = optimise(FactorRiskContribution(; r = r, obj = obj, opt = opt, sets = sets),
                        rd)
        W = JuMP.value.(resr.model[:frc_W])
        w1 = JuMP.value.(resr.model[:w1])
        b1 = pinv(transpose(resr.rr.L))
        Sb = transpose(b1) * pr.sigma * b1
        # The rows hold on the lifted matrix.
        lifted = diag(Sb * W) / tr(Sb * W)
        @test isapprox(lifted[1], 0.09; atol = 1e-4)
        @test lifted[2] >= -0.07 - 1e-4
        @test lifted[5] <= 0.74 + 1e-4
        # The lifted matrix is not of rank one.
        @test eigvals(Symmetric(W))[end - 1] > 1e-3
        # The realised factor shares are the shares of the rank-one matrix.
        fshares = factor_risk_contribution(factory(r, pr, slv), resr.w, pr.X; rd = rd)
        realised = fshares[1:(end - 1)] / sum(fshares)
        @test isapprox(realised, diag(Sb * w1 * transpose(w1)) / dot(w1, Sb, w1);
                       rtol = 1e-6)
        @test isapprox(realised, shares; atol = 2e-3)
        @test !isapprox(realised[1], 0.09; atol = 1e-2)
    end

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

@testset "A semidefinite factor phylogeny reads the marks of the assembled model" begin
    using Test, PortfolioOptimisers, StableRNGs, LinearAlgebra, Clarabel
    import JuMP

    rng = StableRNG(123)
    T, N, Nf = 300, 6, 3
    F = randn(rng, T, Nf) .* 0.02
    B = randn(rng, N, Nf)
    X = F * transpose(B) .+ randn(rng, T, N) .* 0.01 .+ 0.001
    rd = ReturnsResult(; X = X, nx = string.(1:N), F = F, nf = string.(1:Nf))
    slv = Solver(; solver = Clarabel.Optimizer, settings = Dict("verbose" => false),
                 check_sol = (; allow_local = true, allow_almost = true))
    opt = JuMPOptimiser(; pe = prior(EmpiricalPrior(), rd), slv = slv)
    A = [0 1 0; 1 0 0; 0 0 0]
    function build(r, obj; p = 0.05, ple = SemiDefinitePhylogeny(; A = A, p = p))
        res = optimise(FactorRiskContribution(; r = r, obj = obj, opt = opt, frc_ple = ple),
                       rd)
        @test isa(res.retcode, OptimisationSuccess)
        return res.model
    end
    k_of(m) = isa(m[:k], Number) ? m[:k] : JuMP.value(m[:k])

    # The head builds the factor phylogeny after the risk measures and the objective's mark,
    # so a variance that the objective minimises omits the `p·tr(frc_W)` penalty, as it
    # does for the asset phylogeny. It was built first, and the penalty was always added.
    m = build(Variance(), MinimumRisk())
    @test haskey(m, :variance_flag) && haskey(m, :risk_minimised)
    @test haskey(m, :frc_sdp_plg_1) && !haskey(m, :frc_sdp_plg_p_1)
    @test haskey(build(Variance(), MaximumReturn()), :frc_sdp_plg_p_1)
    @test haskey(build(ConditionalValueatRisk(), MinimumRisk()), :frc_sdp_plg_p_1)

    # The rows hold on the lifted matrix, and the PSD cone holds it above `w1·w1ᵀ / k`.
    for m in
        (build(Variance(), MinimumRisk()), build(ConditionalValueatRisk(), MinimumRisk()))
        W = JuMP.value.(m[:frc_W])
        w1 = JuMP.value.(m[:w1])
        @test maximum(abs, A .* W) < 1e-12
        @test eigmin(Symmetric(W - w1 * transpose(w1) / k_of(m))) > -1e-8
    end

    # The rows bind `frc_W`, not `w1·w1ᵀ`. With `p = 0` and no minimised variance, nothing
    # holds `frc_W` down, and the first two factors both carry weight. The penalty makes
    # `frc_W` of rank one here, and then the rows bind the factor weights.
    m = build(ConditionalValueatRisk(), MinimumRisk(); p = 0.0)
    W = JuMP.value.(m[:frc_W])
    w1 = JuMP.value.(m[:w1])
    @test maximum(abs, A .* W) < 1e-12
    @test maximum(abs, A .* (w1 * transpose(w1))) > 0.2
    @test eigvals(Symmetric(W))[end - 1] > 0.4
    m = build(ConditionalValueatRisk(), MinimumRisk(); p = 0.05)
    w1 = JuMP.value.(m[:w1])
    @test maximum(abs, A .* (w1 * transpose(w1))) < 1e-8

    # A vector skips an entry that is not semidefinite, and indexes the rows by position.
    # The integer entry adds rows of its own, which need a MIP solver, so the model is
    # built directly and not solved. The next testset solves it.
    m = JuMP.Model()
    PortfolioOptimisers.set_model_scales!(m, 1, 1)
    JuMP.@variable(m, w1[1:3])
    m[:k] = 1
    PortfolioOptimisers.set_sdp_frc_phylogeny_constraints!(m,
                                                           [IntegerPhylogeny(; A = A,
                                                                             B = 1),
                                                            SemiDefinitePhylogeny(; A = A,
                                                                                  p = 0.05)])
    @test !haskey(m, :frc_sdp_plg_1) && haskey(m, :frc_sdp_plg_2)
    @test haskey(m, :frc_sdp_plg_p_2)
    # The asset phylogeny skips the same way.
    m = JuMP.Model()
    PortfolioOptimisers.set_model_scales!(m, 1, 1)
    JuMP.@variable(m, w[1:3])
    m[:k] = 1
    PortfolioOptimisers.set_sdp_phylogeny_constraints!(m,
                                                       [IntegerPhylogeny(; A = A, B = 1),
                                                        SemiDefinitePhylogeny(; A = A,
                                                                              p = 0.05)])
    @test !haskey(m, :sdp_plg_1) && haskey(m, :sdp_plg_2) && haskey(m, :sdp_plg_p_2)

    # An estimator resolves on the factor returns, and a keyword that the head does not read
    # does not reach it.
    res = optimise(FactorRiskContribution(; opt = opt,
                                          frc_ple = SemiDefinitePhylogenyEstimator()), rd;
                   unread = 1)
    @test isa(res.retcode, OptimisationSuccess)
    @test isa(res.frc_plr, SemiDefinitePhylogeny)
    @test size(res.frc_plr.A) == (Nf, Nf)

    # The penalty is at least `p‖w‖² / k`, so it also spreads the weights. On the asset
    # phylogeny of this panel a large `p` holds both assets of a linked pair.
    Aa = zeros(Int, N, N)
    Aa[1, 2] = Aa[2, 1] = 1
    Aa[3, 4] = Aa[4, 3] = 1
    function asset_w(p)
        res = optimise(MeanRisk(; r = ConditionalValueatRisk(),
                                opt = JuMPOptimiser(; pe = prior(EmpiricalPrior(), rd),
                                                    slv = slv,
                                                    ple = SemiDefinitePhylogeny(; A = Aa,
                                                                                p = p))),
                       rd)
        return res.w
    end
    w = asset_w(0.05)
    @test maximum(abs, Aa .* (w * transpose(w))) < 1e-8
    w = asset_w(1.0)
    @test maximum(abs, Aa .* (w * transpose(w))) > 0.01
end
@testset "An integer factor phylogeny gates the factor weights" begin
    using Test, PortfolioOptimisers, StableRNGs, LinearAlgebra, Clarabel, HiGHS, Pajarito
    import JuMP

    rng = StableRNG(123)
    T, N, Nf = 300, 6, 3
    F = randn(rng, T, Nf) .* 0.02
    B = randn(rng, N, Nf)
    X = F * transpose(B) .+ randn(rng, T, N) .* 0.01 .+ 0.001
    rd = ReturnsResult(; X = X, nx = string.(1:N), F = F, nf = string.(1:Nf))
    cl = Solver(; solver = Clarabel.Optimizer, settings = Dict("verbose" => false),
                check_sol = (; allow_local = true, allow_almost = true))
    mip = Solver(;
                 solver = JuMP.optimizer_with_attributes(Pajarito.Optimizer,
                                                         "verbose" => false,
                                                         "oa_solver" =>
                                                             JuMP.optimizer_with_attributes(HiGHS.Optimizer,
                                                                                            JuMP.MOI.Silent() =>
                                                                                                true),
                                                         "conic_solver" =>
                                                             JuMP.optimizer_with_attributes(Clarabel.Optimizer,
                                                                                            "verbose" =>
                                                                                                false)),
                 check_sol = (; allow_local = true, allow_almost = true))
    pr = prior(EmpiricalPrior(), rd)
    A = [0 1 0; 1 0 0; 0 0 0]
    k_of(m) = isa(m[:k], Number) ? m[:k] : JuMP.value(m[:k])
    w1_of(m) = JuMP.value.(m[:w1]) / k_of(m)
    function solve(obj, slv; frc = IntegerPhylogeny(; A = A, B = 1), kwargs...)
        res = optimise(FactorRiskContribution(; obj = obj,
                                              opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                                  kwargs...),
                                              frc_ple = frc), rd)
        @test isa(res.retcode, OptimisationSuccess)
        return res
    end

    # Without the rows, factors 1 and 2 both carry weight. The rows let one of the two be
    # held, and the factor weights stay the loadings' image of the asset weights, which is
    # what the bounds of the gate are derived from.
    for obj in (MinimumRisk(), MaximumReturn(), MaximumRatio())
        w1 = w1_of(solve(obj, cl; frc = nothing).model)
        @test all(x -> abs(x) > 0.2, w1[1:2])
        res = solve(obj, mip)
        m = res.model
        w1 = w1_of(m)
        ib = round.(Int, JuMP.value.(m[:frc_ib]))
        @test ib == [1, 0, 1]
        @test abs(w1[2]) < 1e-9 && abs(w1[1]) > 1
        @test all(A * ib .<= 1)
        @test haskey(m, :frc_card_plg_1)
        @test maximum(abs, transpose(res.rr.L) * res.w - w1) < 1e-10
        @test isa(res.frc_plr, IntegerPhylogeny) && res.frc_plr.B == 1
        # A variable budget gates with the continuous product of the bits and `k`.
        @test haskey(m, :frc_ibf) == isa(obj, MaximumRatio)
    end

    # An estimator resolves on the factor returns. The network of the three factors gives
    # the row `[1 1 1]`, and with `B = 1` it would let one factor be held: no single factor
    # spans a long-only portfolio of this panel, so the solve would be infeasible. `B = 2`
    # holds two of the three.
    res = solve(MinimumRisk(), mip; frc = IntegerPhylogenyEstimator(; B = 2))
    @test isa(res.frc_plr, IntegerPhylogeny) && size(res.frc_plr.A, 2) == Nf
    @test haskey(res.model, :frc_card_plg_1)
    @test sum(round.(Int, JuMP.value.(res.model[:frc_ib]))) == 2
    @test count(x -> abs(x) < 1e-9, w1_of(res.model)) == 1
    # A vector indexes the rows by position, and the asset phylogeny keeps its own bits
    # beside the factor ones. Assets 4 and 5 are both held without the asset rows.
    Aa = zeros(Int, N, N)
    Aa[4, 5] = Aa[5, 4] = 1
    frc = [SemiDefinitePhylogeny(; A = A, p = 0.05), IntegerPhylogeny(; A = A, B = 1)]
    w = solve(MinimumRisk(), mip; frc = frc).w
    @test min(abs(w[4]), abs(w[5])) > 1e-3
    res = solve(MinimumRisk(), mip; frc = frc, ple = IntegerPhylogeny(; A = Aa, B = 1))
    m = res.model
    @test haskey(m, :frc_sdp_plg_1) && !haskey(m, :frc_card_plg_1)
    @test haskey(m, :frc_card_plg_2) && haskey(m, :card_plg_1)
    @test PortfolioOptimisers.held_bin(PortfolioOptimisers.mip_indicators(m)) === m[:ib]
    @test abs(w1_of(m)[2]) < 1e-9
    @test min(abs(res.w[4]), abs(res.w[5])) < 1e-9

    # The bounds are the image of the asset box under the loadings, widened to hold zero.
    Bt = [1.0 -2.0; 0.5 0.5]
    fwb = PortfolioOptimisers.factor_weight_bounds(WeightBounds(; lb = [-0.5, 0.0],
                                                                ub = [1.0, 0.25]), Bt)
    @test fwb.lb == [-1.0, -0.25] && fwb.ub == [1.0, 0.625]
    fwb = PortfolioOptimisers.factor_weight_bounds(WeightBounds(; lb = 0.25, ub = 1.0),
                                                   [1.0 1.0])
    @test fwb.lb == [0.0] && fwb.ub == [2.0]
    # A factor weight has no bound of its own, so the asset bounds must be finite.
    for wb in (nothing, WeightBounds(; lb = nothing, ub = 1.0),
               WeightBounds(; lb = 0.0, ub = [1.0, Inf]))
        @test_throws ArgumentError PortfolioOptimisers.factor_weight_bounds(wb, Bt)
    end
    # A phylogeny with no integer entry adds no bits.
    m = JuMP.Model()
    PortfolioOptimisers.set_frc_iplg_constraints!(m, SemiDefinitePhylogeny(; A = A),
                                                  nothing, Bt, nothing)
    @test !haskey(m, :frc_ib)
end
