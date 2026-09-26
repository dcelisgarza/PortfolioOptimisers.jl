using Test, PortfolioOptimisers, JuMP, Clarabel, StableRNGs, LinearAlgebra, StatsBase

# The builders of a standard deviation, a variance and an uncertainty-set variance. Each
# testset solves a model and reads its entries, or the weights, against the closed form that
# the docstring of the builder states. The fixture is a small synthetic panel, so each solve
# takes well under a second.

const PO = PortfolioOptimisers

X_vc = 0.01 * randn(StableRNG(42), 150, 8) .+ 0.001 * (1:8)'
N_vc = size(X_vc, 2)
rd_vc = ReturnsResult(; X = X_vc, nx = string.("A", 1:N_vc))
pr_vc = prior(EmpiricalPrior(), rd_vc)
S_vc = pr_vc.sigma
slv_vc = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                settings = "verbose" => false,
                check_sol = (; allow_local = true, allow_almost = true))
opt_vc = JuMPOptimiser(; pe = pr_vc, slv = slv_vc)
key_vc(name, i = 1) = PO.state_key(Symbol(""), name, i)
value_vc(res, name, i = 1) = JuMP.value(res.model[key_vc(name, i)])
solve_vc(r, obj; opt = opt_vc) = optimise(MeanRisk(; r = r, obj = obj, opt = opt), rd_vc)
ub_vc(ub) = RiskMeasureSettings(; ub = ub)
w_mv_vc = solve_vc(Variance(), MinimumRisk()).w
var_mv_vc = dot(w_mv_vc, S_vc, w_mv_vc)
rc_vc = LinearConstraint(;
                         ineq = PartialLinearConstraint(; A = Matrix(1.0I, N_vc, N_vc),
                                                        B = fill(0.5, N_vc)))

@testset "covariance_factor: a Cholesky factor, an eigen factor, or a refusal" begin
    G = PO.covariance_factor(S_vc)
    @test G isa UpperTriangular
    @test isapprox(transpose(G) * G, S_vc; rtol = 1e-12)
    # A singular covariance has no Cholesky factor, and its eigen factor still reproduces it.
    v = randn(StableRNG(9), N_vc)
    S1 = v * transpose(v)
    @test !issuccess(cholesky(S1; check = false))
    G1 = PO.covariance_factor(S1)
    @test isapprox(transpose(G1) * G1, S1; atol = 1e-12)
    @test_throws PosDefException(-1) PO.covariance_factor([1.0 0.5; 0.4 1.0])
    @test_throws PosDefException(1) PO.covariance_factor([1.0 2.0; 2.0 1.0])
end

@testset "chol_sigma_selector: a stated factor, then a stated matrix, then the prior" begin
    model = JuMP.Model()
    Gs = rand(StableRNG(3), N_vc, N_vc)
    @test PO.chol_sigma_selector(model, pr_vc, Variance(; sigma = S_vc, chol = Gs)) === Gs
    @test !haskey(model, :G)
    G2 = PO.chol_sigma_selector(model, pr_vc, Variance(; sigma = 4 * S_vc))
    @test isapprox(transpose(G2) * G2, 4 * S_vc; rtol = 1e-12)
    @test !haskey(model, :G)
    # The prior's factor is registered once, and a second measure reads the same one.
    G3 = PO.chol_sigma_selector(model, pr_vc, Variance())
    @test isapprox(transpose(G3) * G3, S_vc; rtol = 1e-12)
    @test PO.chol_sigma_selector(model, pr_vc, StandardDeviation()) === model[:G]
end

@testset "variance_risk_bounds_val converts a bound into the units of its expression" begin
    @test PO.variance_risk_bounds_val(LinearBound(), 4.0) == 4.0
    @test PO.variance_risk_bounds_val(LinearBound(), [4.0, 9.0]) == [4.0, 9.0]
    @test PO.variance_risk_bounds_val(SquareRootBound(), 4.0) == 2.0
    @test PO.variance_risk_bounds_val(SquareRootBound(), [4.0, 9.0]) == [2.0, 3.0]
    @test PO.variance_risk_bounds_val(SquaredBound(), 3.0) == 9.0
    @test PO.variance_risk_bounds_val(SquaredBound(), [2.0, 3.0]) == [4.0, 9.0]
    @test isnothing(PO.variance_risk_bounds_val(SquareRootBound(), nothing))
    f = PO.variance_risk_bounds_val(SquareRootBound(), Frontier(; N = 7))
    @test f.N == 7 && f.factor == 1 && f.bound isa SquareRootBound
end

@testset "A standard deviation: its epigraph is tight, and its bound is exact" begin
    res = solve_vc(StandardDeviation(), MinimumRisk())
    @test isapprox(value_vc(res, :sd_risk_), sqrt(dot(res.w, S_vc, res.w)); rtol = 1e-6)
    @test haskey(res.model, key_vc(:csd_risk_soc_))
    ub = 1.2 * sqrt(var_mv_vc)
    for obj in (MaximumReturn(), MaximumRatio(), MaximumRatio(; ohf = 10.0))
        r = solve_vc(StandardDeviation(; settings = ub_vc(ub)), obj)
        @test isa(r.retcode, OptimisationSuccess)
        @test sqrt(dot(r.w, S_vc, r.w)) <= ub * (1 + 1e-6)
    end
end

@testset "A variance: each formulation states the variance, and its bound is exact" begin
    ub = 1.3 * var_mv_vc
    for alg in (SquaredSOCRiskExpr(), QuadRiskExpr())
        res = solve_vc(Variance(; alg = alg), MinimumRisk())
        @test isapprox(value_vc(res, :variance_risk_), dot(res.w, S_vc, res.w); rtol = 1e-5)
        @test haskey(res.model, key_vc(:dev_)) && haskey(res.model, key_vc(:cdev_soc_))
        @test !haskey(res.model, :W)
        # The bound acts on `dev_` in the units of a standard deviation, which has degree one,
        # so it holds under a ratio objective for every `ohf`.
        for obj in (MaximumReturn(), MaximumRatio(), MaximumRatio(; ohf = 10.0))
            r = solve_vc(Variance(; alg = alg, settings = ub_vc(ub)), obj)
            @test isa(r.retcode, OptimisationSuccess)
            @test dot(r.w, S_vc, r.w) <= ub * (1 + 1e-5)
        end
    end
    # The risk-contribution rows move the variance to the lifted trace, which is tight under
    # a minimised risk and bounded in the units of a variance.
    res = solve_vc(Variance(; rc = rc_vc), MinimumRisk())
    @test haskey(res.model, :W) && !haskey(res.model, key_vc(:dev_))
    @test isapprox(value_vc(res, :variance_risk_), dot(res.w, S_vc, res.w); rtol = 1e-3)
    for obj in (MaximumReturn(), MaximumRatio())
        r = solve_vc(Variance(; rc = rc_vc, settings = ub_vc(ub)), obj)
        @test isa(r.retcode, OptimisationSuccess)
        @test dot(r.w, S_vc, r.w) <= ub * (1 + 1e-5)
    end
end

@testset "A variance built before the one with rows keeps its cone" begin
    rcv = Variance(; rc = rc_vc)
    first_plain = solve_vc([Variance(), rcv], MinimumRisk())
    first_rows = solve_vc([rcv, Variance()], MinimumRisk())
    @test haskey(first_plain.model, key_vc(:dev_, 1))
    @test !haskey(first_rows.model, key_vc(:dev_, 2))
    @test haskey(first_rows.model, key_vc(:variance_risk_, 2))
end

@testset "An uncertainty-set variance states its worst case" begin
    # Two boxes share the dual matrices, and each still reaches its own worst case.
    b1 = BoxUncertaintySet(; lb = S_vc .- 0.5 * abs.(S_vc), ub = S_vc .+ 0.5 * abs.(S_vc))
    b2 = BoxUncertaintySet(; lb = S_vc .- 2.0 * abs.(S_vc), ub = S_vc .+ 0.1 * abs.(S_vc))
    res = solve_vc([UncertaintySetVariance(; ucs = b1), UncertaintySetVariance(; ucs = b2)],
                   MinimumRisk())
    @test isapprox(value_vc(res, :bucs_variance_risk_, 1), PO.ucs_variance(b1, S_vc, res.w);
                   rtol = 1e-4)
    @test isapprox(value_vc(res, :bucs_variance_risk_, 2), PO.ucs_variance(b2, S_vc, res.w);
                   rtol = 1e-4)
    e1 = EllipsoidalUncertaintySet(; sigma = Matrix(1.0I, N_vc^2, N_vc^2), k = 5e-3,
                                   class = SigmaUncertaintySetClass())
    res = solve_vc(UncertaintySetVariance(; ucs = e1), MinimumRisk())
    @test isapprox(value_vc(res, :eucs_variance_risk_), PO.ucs_variance(e1, S_vc, res.w);
                   rtol = 1e-4)
end

@testset "A compact set: its bound holds under a ratio objective, and a singular centre" begin
    Q = Matrix(qr(randn(StableRNG(7), N_vc, 2)).Q)[:, 1:2]
    C = 0.5 .+ rand(StableRNG(8), N_vc)
    ucs = CompactCovarianceUncertaintySet(; kappa = 2e-4, C = C, Q = Q)
    wc_min = PO.ucs_variance(ucs, S_vc,
                             solve_vc(UncertaintySetVariance(; ucs = ucs), MinimumRisk()).w)
    # A bound below the worst case of the unbounded maximum-ratio portfolio, so it binds.
    ub = 1.1 * wc_min
    r0 = solve_vc(UncertaintySetVariance(; ucs = ucs), MaximumRatio())
    @test PO.ucs_variance(ucs, S_vc, r0.w) > ub
    ws = map((nothing, 1.0, 10.0)) do ohf
        r = solve_vc(UncertaintySetVariance(; ucs = ucs, settings = ub_vc(ub)),
                     MaximumRatio(; ohf = ohf))
        @test isa(r.retcode, OptimisationSuccess)
        # The row bounds the square root, of degree one, so the bound does not move with k.
        @test isapprox(PO.ucs_variance(ucs, S_vc, r.w), ub; rtol = 1e-4)
        @test haskey(r.model, key_vc(:sd_cucs_)) && haskey(r.model, key_vc(:csd_cucs_soc_))
        r.w
    end
    @test isapprox(ws[1], ws[2]; rtol = 1e-3, atol = 1e-5)
    @test isapprox(ws[1], ws[3]; rtol = 1e-3, atol = 1e-5)
    r = solve_vc(UncertaintySetVariance(; ucs = ucs, settings = ub_vc(ub)), MaximumReturn())
    @test isapprox(PO.ucs_variance(ucs, S_vc, r.w), ub; rtol = 1e-5)
    # Without a bound the model carries no square root.
    @test !haskey(r0.model, key_vc(:sd_cucs_))
    # A singular centre has no Cholesky factor, and the compact set accepts it as a variance
    # does.
    v = randn(StableRNG(9), N_vc)
    S1 = v * transpose(v) + Diagonal([zeros(N_vc - 2); 1e-4; 1e-4])
    pr1 = PO.LowOrderPrior(; X = X_vc, mu = vec(mean(X_vc; dims = 1)), sigma = S1)
    opt1 = JuMPOptimiser(; pe = pr1, slv = slv_vc)
    r1 = solve_vc(UncertaintySetVariance(; ucs = ucs), MinimumRisk(); opt = opt1)
    @test isa(r1.retcode, OptimisationSuccess)
    @test isapprox(value_vc(r1, :cucs_variance_risk_), PO.ucs_variance(ucs, S1, r1.w);
                   rtol = 1e-4, atol = 1e-10)
end

@testset "The factor variance prices the off-factor weights under flag = true" begin
    F = 0.01 * randn(StableRNG(11), 150, 3)
    Xf = F * randn(StableRNG(12), 3, N_vc) + 0.005 * randn(StableRNG(13), 150, N_vc)
    rdf = ReturnsResult(; X = Xf, nx = string.("A", 1:N_vc), F = F, nf = ["F1", "F2", "F3"])
    prf = prior(EmpiricalPrior(), rdf)
    function frc_vc(; rc = nothing)
        r = optimise(FactorRiskContribution(; r = Variance(; rc = rc), obj = MinimumRisk(),
                                            flag = true,
                                            opt = JuMPOptimiser(; pe = prf, slv = slv_vc)),
                     rdf)
        @test isa(r.retcode, OptimisationSuccess)
        return r
    end
    # The lift covers the whole decision vector `[w1; w2]`, so the expression is the
    # variance of the returned weights (#1350). Without the off-factor block, eight
    # long-only weights in the span of three factors cannot sum to one on this panel, so
    # only the block is solved. The basis `[b1 b2]` reaches every weight vector, so the
    # minimum is the minimum variance of `MeanRisk`.
    r = frc_vc()
    @test size(r.model[:frc_W]) == (N_vc, N_vc)
    @test isapprox(value_vc(r, :variance_risk_), dot(r.w, prf.sigma, r.w); rtol = 1e-4)
    rm = optimise(MeanRisk(; r = Variance(), obj = MinimumRisk(),
                           opt = JuMPOptimiser(; pe = prf, slv = slv_vc)), rdf)
    @test isapprox(dot(r.w, prf.sigma, r.w), dot(rm.w, prf.sigma, rm.w); rtol = 1e-4)
    # A factor row states the Euler contribution of the factor as a share of the whole
    # variance, the share that `factor_risk_contribution` reports. Unconstrained, the first
    # factor carries about 2 % of the variance. A row that asks for 30 % binds, and at rank
    # one the share of the returned weights meets it.
    f0 = factor_risk_contribution(Variance(), frc_vc().w, prf; rd = rdf)
    @test f0[1] / sum(f0) < 0.1
    r = frc_vc(;
               rc = LinearConstraint(;
                                     ineq = PartialLinearConstraint(; A = [-1.0 0.0 0.0],
                                                                    B = [-0.3])))
    f = factor_risk_contribution(Variance(), r.w, prf; rd = rdf)
    sW = JuMP.value.(r.model[key_vc(:sigma_W_)])
    @test size(sW) == (3, N_vc)
    @test isapprox(diag(sW), f[1:3]; rtol = 1e-3, atol = 1e-9)
    @test isapprox(f[1] / sum(f), 0.3; rtol = 1e-3)
end
