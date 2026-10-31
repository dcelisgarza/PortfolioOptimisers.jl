using Clarabel, HiGHS, JuMP, Pajarito
const PO = PortfolioOptimisers

# The sweep of `src/17_Optimisation/05_JuMP/07_RiskBudgeting.jl`. Each formulation is checked
# against the risk budgeting condition of its original source, x_i (Sigma x)_i / x' Sigma x =
# b_i: the log constraint of bruderroncalli2012 equation 9, the orthant form of bai2016
# problem 2.13 with the general budget of mosek2023c problem 9.7, the mixed-integer form of
# mosek2023c problem 9.9, and the factor form of roncalliweisang2012 problem 8.

const RBS_X, RBS_F = let rng = StableRNG(689)
    T, N = 300, 5
    F = 0.01 .* randn(rng, T, 2)
    L = [0.9 0.2; 0.5 0.8; 1.1 -0.3; 0.3 0.6; 0.7 0.4]
    F * L' .+ 0.008 .* randn(rng, T, N) .+ 0.0005, F
end
const RBS_RD = ReturnsResult(; nx = ["a$i" for i in 1:5], X = RBS_X, nf = ["f1", "f2"],
                             F = RBS_F)
const RBS_PR = prior(EmpiricalPrior(), RBS_RD)
const RBS_B = [0.1, 0.15, 0.2, 0.25, 0.3]
const RBS_SLV = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                       check_sol = (; allow_local = true, allow_almost = true),
                       settings = Dict("verbose" => false, "tol_gap_abs" => 1e-11,
                                       "tol_gap_rel" => 1e-11, "tol_feas" => 1e-11))
const RBS_MIP = Solver(; name = :pajarito,
                       solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                          "verbose" => false,
                                                          "oa_solver" =>
                                                              optimizer_with_attributes(HiGHS.Optimizer,
                                                                                        JuMP.MOI.Silent() =>
                                                                                            true),
                                                          "conic_solver" =>
                                                              optimizer_with_attributes(Clarabel.Optimizer,
                                                                                        "verbose" =>
                                                                                            false)),
                       check_sol = (; allow_local = true, allow_almost = true))
# The share of the variance that each asset contributes. It is the same for the variance
# and for the standard deviation, because the two differ by a power.
rbs_share(w) = w .* (RBS_PR.sigma * w) ./ dot(w, RBS_PR.sigma * w)
function rbs_solve(alg; b = RBS_B, r = Variance(), slv = RBS_SLV, kwargs...)
    opt = JuMPOptimiser(; pe = RBS_PR, slv = slv, kwargs...)
    rba = AssetRiskBudgeting(; rkb = RiskBudget(; val = b), alg = alg)
    return optimise(RiskBudgeting(; opt = opt, r = r, rba = rba), RBS_RD)
end

@testset "RiskBudgeting sweep: the log constraint gives the budgeted contributions" begin
    for r in (Variance(), StandardDeviation())
        res = rbs_solve(LogRiskBudgeting(); r = r)
        @test isa(res.retcode, OptimisationSuccess)
        @test isapprox(rbs_share(res.w), RBS_B; atol = 5e-5)
        @test isapprox(sum(res.w), 1; atol = 1e-8)
        # Appendix A.2 of maillard2008: the solution cannot be zero, so the log constraint
        # binds, and the weights are the solution divided by its sum.
        y = value.(res.model[:w])
        @test isapprox(dot(RBS_B, log.(y)), 0; atol = 1e-6)
        @test isapprox(value(res.model[:k]), sum(y); rtol = 1e-8)
    end
    # Section 2.4 of bruderroncalli2012: any convex risk measure with the Euler decomposition.
    r = factory(ConditionalValueatRisk(), RBS_PR, RBS_SLV)
    res = rbs_solve(LogRiskBudgeting(); r = r)
    rc = risk_contribution(r, res.w, RBS_PR.X)
    @test isapprox(rc / sum(rc), RBS_B; atol = 5e-3)
end

@testset "RiskBudgeting sweep: every formulation meets a budget other than one" begin
    z = [1, 1, -1, 1, 1]
    rg = BudgetRange(; lb = 0.4, ub = 0.6)
    # With no short budget the long budget row fixes the sum of the long parts. Under the
    # mixed-integer form the parts are the weights, so a budget of 0.5 keeps the book long,
    # and a range bounds the shorts by 0.2 k, a row that binds.
    cases = ((LogRiskBudgeting(), RBS_SLV, (;), (0.5, rg)),
             (LogRiskBudgeting(; z = z), RBS_SLV,
              (; wb = WeightBounds(; lb = -1.0, ub = 1.0), sbgt = 1.0), (0.5, rg)),
             (MixedIntegerRiskBudgeting(), RBS_MIP, (; wb = nothing, so = 10_000), (0.5,)))
    for (alg, slv, kw, bgts) in cases, bgt in bgts
        res = rbs_solve(alg; slv = slv, bgt = bgt, kw...)
        @test isa(res.retcode, OptimisationSuccess)
        @test isapprox(rbs_share(res.w), RBS_B; atol = 5e-5)
        if isa(bgt, Number)
            @test isapprox(sum(res.w), bgt; atol = 1e-8)
        else
            @test bgt.lb - 1e-8 <= sum(res.w) <= bgt.ub + 1e-8
        end
    end
end

@testset "RiskBudgeting sweep: the orthant vector sets the signs of the weights" begin
    kw = (; wb = WeightBounds(; lb = -1.0, ub = 1.0), sbgt = 1.0)
    z = [1, 1, -1, 1, 1]
    res = rbs_solve(LogRiskBudgeting(; z = z); kw...)
    @test sign.(res.w) == z
    @test isapprox(rbs_share(res.w), RBS_B; atol = 5e-5)
    # The solution in the orthant of `z` has a negative sum, so the budget row needs k < 0,
    # which the short budget forbids. The solve returns a portfolio of that orthant that
    # misses the budget, or fails. Lemma 2.2 of bai2016: the normalised portfolio of that
    # orthant lies in the complementary orthant, and `-z` reaches it. This book has no
    # bound and a short budget that does not bind, so the budget holds there.
    kw = (; wb = WeightBounds(; lb = -Inf, ub = Inf), sbgt = 1000.0)
    z = [-1, 1, 1, -1, -1]
    res = rbs_solve(LogRiskBudgeting(; z = z); kw...)
    @test !isa(res.retcode, OptimisationSuccess) ||
          maximum(abs.(rbs_share(res.w) - RBS_B)) > 1e-2
    res = rbs_solve(LogRiskBudgeting(; z = -z); kw...)
    @test sign.(res.w) == -z
    # Measured gap 6.4e-5 on this book.
    @test isapprox(rbs_share(res.w), RBS_B; atol = 2e-4)
end

@testset "RiskBudgeting sweep: the mixed-integer form takes the orthant of least risk" begin
    S = RBS_PR.sigma
    kw = (; wb = nothing, bgt = nothing, sbgt = nothing)
    # The risk of the solution in each orthant, at the scale where b' ln|y| = 0. The first
    # sign is fixed, because the orthants z and -z hold the same solution up to its sign.
    zs = [vcat(1, collect(t)) for t in Iterators.product(fill((1, -1), 4)...)][:]
    risk = map(zs) do z
        res = rbs_solve(LogRiskBudgeting(; z = z); kw...)
        w = res.w
        # No row fixes `k` here, and `k >= 0` keeps the orthant of `z` (#1844).
        @test value(res.model[:k]) > 0
        @test sign.(w) == z
        @test isapprox(rbs_share(w), RBS_B; atol = 5e-5)
        u = w * exp(-dot(RBS_B, log.(abs.(w))))
        return dot(u, S * u)
    end
    res = rbs_solve(MixedIntegerRiskBudgeting(); slv = RBS_MIP, so = 10_000, kw...)
    @test isa(res.retcode, OptimisationSuccess)
    @test isapprox(rbs_share(res.w), RBS_B; atol = 5e-5)
    zm = Int.(sign.(res.w))
    @test zm * zm[1] == zs[argmin(risk)]
    # Under a budget, a short budget range leaves the short exposure free, so the search
    # reaches the same orthant.
    res = rbs_solve(MixedIntegerRiskBudgeting(); slv = RBS_MIP, so = 10_000,
                    wb = WeightBounds(; lb = -Inf, ub = Inf),
                    sbgt = BudgetRange(; lb = 0.0, ub = 10.0))
    @test isapprox(rbs_share(res.w), RBS_B; atol = 5e-5)
    @test isapprox(sum(res.w), 1; atol = 1e-8)
    @test Int.(sign.(res.w)) == zs[argmin(risk)]
end

@testset "RiskBudgeting sweep: the factor form gives the budgeted factor contributions" begin
    S = RBS_PR.sigma
    bf = [0.3, 0.7]
    for (flag, hedge) in ((false, false), (true, false), (true, true))
        opt = JuMPOptimiser(; pe = RBS_PR, slv = RBS_SLV,
                            wb = WeightBounds(; lb = -Inf, ub = Inf), sbgt = 1000.0)
        rba = FactorRiskBudgeting(; rkb = RiskBudget(; val = bf), flag = flag,
                                  hedge = hedge)
        res = optimise(RiskBudgeting(; opt = opt, rba = rba), RBS_RD)
        @test isa(res.retcode, OptimisationSuccess)
        w = res.w
        L = res.prb.rr.L
        # Theorem 1 of roncalliweisang2012: the contribution of factor j is y_j (L^+ dR/dx)_j,
        # with the factor exposures y = L' w.
        y = transpose(L) * w
        rc = y .* (pinv(L) * (S * w))
        @test isapprox(rc / sum(rc), bf; atol = 5e-5)
        @test isapprox(value.(res.model[:w1]) / value(res.model[:k]), y; rtol = 1e-8)
        # The off-factor weights are free, so at the optimum they carry no marginal risk:
        # the separation principle of section 4.1.1. With `flag = false` the weights lie in
        # the range of the loadings, and the factors take the whole variance.
        g = transpose(nullspace(Matrix(transpose(L)))) * (S * w)
        if flag
            @test norm(g) <= 1e-4 * norm(S * w)
        else
            @test isapprox(sum(rc), dot(w, S * w); rtol = 1e-8)
        end
    end
    # A long-only bound that binds keeps the off-factor weights from their free optimum. The
    # hedge rows hold the first-order condition all the same.
    opt = JuMPOptimiser(; pe = RBS_PR, slv = RBS_SLV)
    rba = FactorRiskBudgeting(; rkb = RiskBudget(; val = bf), hedge = true)
    res = optimise(RiskBudgeting(; opt = opt, rba = rba), RBS_RD)
    L = res.prb.rr.L
    g = transpose(nullspace(Matrix(transpose(L)))) * (S * res.w)
    @test norm(g) <= 1e-8 * norm(S * res.w)
end

@testset "RiskBudgeting sweep: the floor of k" begin
    # The derived floor is 1e-4 over the upper bound of the budget, and zero when no finite
    # budget ties `k` to the sum of the weights. A stated `kmin` replaces it.
    rb_of(; kmin = nothing, kw...) = RiskBudgeting(;
                                                   opt = JuMPOptimiser(; slv = RBS_SLV,
                                                                       kw...), kmin = kmin)
    floor_of(; kw...) = PO.risk_budgeting_scale_floor(rb_of(; kw...))
    @test floor_of() == 1e-4
    @test floor_of(; bgt = 0.5) == 2e-4
    @test floor_of(; bgt = BudgetRange(; lb = 0.4, ub = 0.8)) ≈ 1.25e-4
    @test iszero(floor_of(; bgt = nothing))
    @test iszero(floor_of(; bgt = BudgetRange(; lb = 0.4, ub = nothing)))
    @test floor_of(; kmin = 1e-2) == 1e-2
    @test floor_of(; kmin = 1e-2, bgt = nothing) == 1e-2
    # A stated zero turns the floor off and keeps only `k >= 0`.
    @test iszero(floor_of(; kmin = 0.0))
    @test_throws DomainError rb_of(; kmin = -1)
    @test PO.port_opt_view(rb_of(; kmin = 1e-3), 1:3, RBS_X).kmin == 1e-3
    # The solution in the orthant of `z` has a negative sum, and no weight bound or short
    # budget bounds the scale. No portfolio of that orthant meets a positive budget, so the
    # scale collapses onto the floor, and the weights miss the budget. Measured at the
    # tolerance of the solver alone, before the floor: k = 2.2e-6 and a gross exposure of
    # 5e6, with `OptimisationSuccess`.
    kw = (; wb = WeightBounds(; lb = -Inf, ub = Inf),
          sbgt = BudgetRange(; lb = 0.0, ub = nothing))
    z = [-1, 1, 1, -1, -1]
    for (bgt, kmin) in ((1.0, nothing), (0.5, nothing), (1.0, 1e-2))
        opt = JuMPOptimiser(; pe = RBS_PR, slv = RBS_SLV, bgt = bgt, kw...)
        rba = AssetRiskBudgeting(; rkb = RiskBudget(; val = RBS_B),
                                 alg = LogRiskBudgeting(; z = z))
        rb = RiskBudgeting(; opt = opt, r = Variance(), rba = rba, kmin = kmin)
        res = optimise(rb, RBS_RD)
        kf = PO.risk_budgeting_scale_floor(rb)
        @test lower_bound(res.model[:k]) == kf
        # Measured on the floor to 1.3e-5.
        @test isapprox(value(res.model[:k]), kf; rtol = 1e-4)
        @test sign.(res.w) == z
        @test isapprox(sum(res.w), bgt; rtol = 1e-6)
        @test maximum(abs.(rbs_share(res.w) - RBS_B)) > 1e-1
    end
    # The complementary orthant has a positive sum, so its scale stays far above the floor.
    # Its k = 0.459 is the smallest scale that a well-posed model fixed over the risk
    # budgeting tests.
    res = rbs_solve(LogRiskBudgeting(; z = -z); kw...)
    @test value(res.model[:k]) > 1e3 * floor_of()
    @test sign.(res.w) == -z
    @test isapprox(rbs_share(res.w), RBS_B; atol = 5e-5)
end

@testset "RiskBudgeting sweep: views, validation and the result" begin
    a = AssetRiskBudgeting(; alg = MixedIntegerRiskBudgeting())
    @test PO.port_opt_view(a, 1:3).alg === MixedIntegerRiskBudgeting()
    a = AssetRiskBudgeting(; alg = LogRiskBudgeting())
    @test PO.port_opt_view(a, 1:3).alg === a.alg
    a = AssetRiskBudgeting(; alg = LogRiskBudgeting(; z = [1, -1, 1, 1, -1]))
    @test PO.port_opt_view(a, [2, 5]).alg.z == [-1, -1]
    rb = RiskBudgeting(; opt = JuMPOptimiser(; slv = RBS_SLV),
                       rba = AssetRiskBudgeting(; alg = MixedIntegerRiskBudgeting()))
    @test isa(PO.port_opt_view(rb, 1:3, RBS_X).rba.alg, MixedIntegerRiskBudgeting)
    # A budget with no sets names no asset, so a non-investable list leaves it unchanged.
    rb2 = PO.non_investable_universe(rb, ["a2"])
    @test isnothing(rb2.rba.sets)
    @test isa(rb2.rba.alg, MixedIntegerRiskBudgeting)
    @test_throws IsEmptyError RiskBudgeting(; opt = JuMPOptimiser(; slv = RBS_SLV),
                                            r = PO.RiskMeasure[])
    @test_throws IsEmptyError LogRiskBudgeting(; z = Int[])
    @test_throws ArgumentError LogRiskBudgeting(; z = [1, 2])
    res = rbs_solve(LogRiskBudgeting())
    @test isa(res.prb, PO.ProcessedAssetRiskBudgetingAttributes)
    @test res.rkb.val == RBS_B
    res2 = PO.set_retcode(res, OptimisationFailure(; res = nothing))
    @test isa(res2.retcode, OptimisationFailure)
    @test res2.w == res.w
end
