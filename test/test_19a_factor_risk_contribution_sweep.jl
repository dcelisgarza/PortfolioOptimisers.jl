using Clarabel
using JuMP: JuMP
const PO = PortfolioOptimisers

# The sweep of `src/17_Optimisation/05_JuMP/05_FactorRiskContribution.jl`. The change of basis
# is Section 3 of roncalliweisang2012: `w = B1 * w1 + B2 * w2`, with `B1` the pseudoinverse of
# the transposed loadings and the columns of `B2` a basis of their null space (Equation 2,
# after Section 4 of meucci2007). Each test checks one consequence of it with numbers.

const FRS_N = 6
const FRS_DATA = let rng = StableRNG(687)
    T, K = 300, 2
    F = 0.01 .* randn(rng, T, K)
    X = 0.0005 .+ F * randn(rng, K, FRS_N) .+ 0.01 .* randn(rng, T, FRS_N) .+
        (0.0003 .* (1:FRS_N))'
    # A third factor whose loadings are zero, for the case of loadings of low rank.
    Fz = hcat(F, 0.01 .* randn(rng, T))
    (; X, F, Fz)
end
const FRS_RD = ReturnsResult(; nx = string.("a", 1:FRS_N), X = FRS_DATA.X,
                             nf = ["f1", "f2"], F = FRS_DATA.F)
const FRS_PR = prior(EmpiricalPrior(), FRS_RD)
const FRS_SLV = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                       check_sol = (; allow_local = true, allow_almost = true),
                       settings = Dict("verbose" => false, "tol_gap_abs" => 1e-11,
                                       "tol_gap_rel" => 1e-11, "tol_feas" => 1e-11))
# No weight bound and a short budget that never binds: the budget row alone constrains the
# weights, so the minimum-variance portfolio has a closed form.
function frs_free(; kwargs...)
    return JuMPOptimiser(; pe = FRS_PR, slv = FRS_SLV,
                         wb = WeightBounds(; lb = -Inf, ub = Inf), sbgt = 1000.0, kwargs...)
end
frs_floor(lb) = ArithmeticReturn(; settings = JuMPReturnsSettings(; lb = lb))

@testset "FactorRiskContribution sweep: the two forms of the change of basis" begin
    S = FRS_PR.sigma
    o = ones(FRS_N)
    v(w) = dot(w, S, w)
    res0 = optimise(FactorRiskContribution(; opt = frs_free()), FRS_RD)
    @test isa(res0.retcode, PO.OptimisationSuccess)
    # The loadings resolve through the result.
    B = res0.L
    @test B === res0.rr.L
    # `flag = false`: the weights lie in the column space of B, and the factor exposures are
    # the decision variables, B' w = w1.
    @test isapprox(transpose(B) * res0.w, JuMP.value.(res0.model[:w1]); atol = 1e-12)
    # The minimum variance over w = B z with 1' w = 1: z ∝ (B' Σ B)⁻¹ B' 1.
    z = (transpose(B) * S * B) \ (transpose(B) * o)
    @test isapprox(res0.w, B * z / dot(o, B * z); atol = 5e-5)
    # `flag = true`: the weights span every direction, so the minimum is the minimum-variance
    # portfolio Σ⁻¹ 1 / (1' Σ⁻¹ 1) of MeanRisk. The objective is flat near the minimum, so
    # the weights agree to the solver tolerance and the variance to a relative 1e-6.
    res1 = optimise(FactorRiskContribution(; opt = frs_free(), flag = true), FRS_RD)
    wmv = S \ o / dot(o, S \ o)
    @test isapprox(v(res1.w), v(wmv); rtol = 1e-6)
    @test isapprox(res1.w, wmv; atol = 1e-3)
    @test v(res1.w) < v(res0.w)
    # The Euler decomposition in the factor basis: the entries sum to the variance, and the
    # off-factor entry is zero when the weights lie in the column space of B.
    rc0 = factor_risk_contribution(Variance(), res0.w, FRS_PR; rd = FRS_RD)
    @test isapprox(sum(rc0), v(res0.w); rtol = 1e-6)
    @test isapprox(rc0[end], 0; atol = 1e-12)
    rc1 = factor_risk_contribution(Variance(), res1.w, FRS_PR; rd = FRS_RD)
    @test isapprox(sum(rc1), v(res1.w); rtol = 1e-6)
end

@testset "FactorRiskContribution sweep: an rc row is a share of the whole variance" begin
    sets = UniverseSets(; dict = Dict("nx" => FRS_RD.nf))
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = Dict("verbose" => false))
    opt = JuMPOptimiser(; pe = FRS_PR, slv = slv, wb = WeightBounds(; lb = -Inf, ub = Inf),
                        sbgt = 1000.0)
    for target in (0.6, 0.3)
        r = Variance(; rc = LinearConstraintEstimator(; val = ["f1 == $target"]))
        res = optimise(FactorRiskContribution(; opt = opt, r = r, sets = sets), FRS_RD)
        @test isa(res.retcode, PO.OptimisationSuccess)
        rc = factor_risk_contribution(factory(r, FRS_PR, slv), res.w, FRS_PR; rd = FRS_RD)
        share = rc / dot(res.w, FRS_PR.sigma, res.w)
        @test isapprox(share[1], target; atol = 1e-4)
        @test isapprox(share[1] + share[2], 1; atol = 1e-4)
    end
end

@testset "FactorRiskContribution sweep: the hedge rows" begin
    # With `w1` fixed, the rows B2' Σ w = 0 give the minimum-variance weights with the factor
    # exposures w1, so w' Σ w = w1' (B' Σ⁻¹ B)⁻¹ w1.
    S = FRS_PR.sigma
    res = optimise(FactorRiskContribution(; opt = frs_free()), FRS_RD)
    B = res.L
    reg = PO.Regression(; M = res.rr.M, L = B, b = res.rr.b)
    model = JuMP.Model(Clarabel.Optimizer)
    JuMP.set_silent(model)
    PO.set_model_scales!(model, 1, 1)
    b1, b2, _ = PO.set_factor_risk_contribution_constraints!(model, reg, FRS_RD, FRS_PR,
                                                             true, nothing; hedge = true)
    @test isapprox(b1, B / (transpose(B) * B); atol = 1e-12)
    @test isapprox(transpose(b2) * b2, I; atol = 1e-12)
    @test isapprox(transpose(B) * b2, zeros(2, FRS_N - 2); atol = 1e-12)
    y = [0.3, -0.2]
    JuMP.@constraint(model, model[:w1] .== y)
    JuMP.optimize!(model)
    w = JuMP.value.(model[:w])
    @test isapprox(transpose(B) * w, y; atol = 1e-10)
    @test isapprox(dot(w, S, w), dot(y, inv(transpose(B) * (S \ B)), y); rtol = 1e-8)
end

@testset "FactorRiskContribution sweep: loadings of low rank" begin
    # A factor with zero loadings leaves B of rank 2 in 3 columns, so the null space of B'
    # has N - 2 columns, not N - 3. `w2` takes its size from the null space.
    res = optimise(FactorRiskContribution(; opt = frs_free()), FRS_RD)
    Lz = hcat(res.L, zeros(FRS_N))
    reg = PO.Regression(; M = res.rr.M, L = Lz, b = res.rr.b)
    rd = ReturnsResult(; nx = FRS_RD.nx, X = FRS_RD.X, nf = ["f1", "f2", "f3"],
                       F = FRS_DATA.Fz)
    resz = optimise(FactorRiskContribution(; opt = frs_free(), re = reg, flag = true), rd)
    @test isa(resz.retcode, PO.OptimisationSuccess)
    @test length(resz.model[:w2]) == FRS_N - 2
    S = FRS_PR.sigma
    o = ones(FRS_N)
    @test isapprox(dot(resz.w, S, resz.w), 1 / dot(o, S \ o); rtol = 1e-6)
end

@testset "An optimiser that solves once refuses a frontier" begin
    opt = JuMPOptimiser(; pe = FRS_PR, slv = FRS_SLV, ret = frs_floor([0.001, 0.0019]))
    for est in (FactorRiskContribution(; opt = opt), RiskBudgeting(; opt = opt),
                RelaxedRiskBudgeting(; opt = opt))
        @test_throws ArgumentError optimise(est, FRS_RD)
    end
    optf = JuMPOptimiser(; pe = FRS_PR, slv = FRS_SLV, ret = frs_floor(Frontier(; N = 3)))
    @test_throws ArgumentError optimise(FactorRiskContribution(; opt = optf), FRS_RD)
    r = Variance(; settings = RiskMeasureSettings(; ub = [2e-5, 3e-5]))
    @test_throws ArgumentError optimise(RiskBudgeting(;
                                                      opt = JuMPOptimiser(; pe = FRS_PR,
                                                                          slv = FRS_SLV),
                                                      r = r), FRS_RD)
    # The two optimisers that sweep a frontier keep their sweep.
    @test length(optimise(MeanRisk(; opt = opt), FRS_RD).w) == 2
    @test length(optimise(NearOptimalCentering(; opt = opt), FRS_RD).w) == 2
    # A one-step projection returns one allocation, so the set refuses a frontier floor.
    @test_throws ArgumentError ProgrammeAllocationSet(; slv = FRS_SLV,
                                                      ret = frs_floor(Frontier(; N = 3)))
    @test_throws ArgumentError ProgrammeAllocationSet(; slv = FRS_SLV,
                                                      ret = [frs_floor(0.001),
                                                             frs_floor([0.001, 0.002])])
    @test isa(ProgrammeAllocationSet(; slv = FRS_SLV, ret = frs_floor(0.001)),
              ProgrammeAllocationSet)
end
