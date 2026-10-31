using Clarabel, JuMP
const PO = PortfolioOptimisers

# The sweep of `src/17_Optimisation/05_JuMP/08_RelaxedRiskBudgeting.jl`. The three variants
# are Models (A), (B) and (C) of gambetakwon2020, and the oracle below states each one in the
# variables of the source, with the number of assets n: x' Sigma x <= n psi^2 for Model (A),
# and x' Sigma x <= n (psi^2 - rho^2) with x' Sigma x <= rho^2 or lambda x' Theta x <= rho^2
# for Models (B) and (C). The head writes the same models in variables that are sqrt(n)
# times larger, so the weights agree and the objectives differ by that factor.

const RRS_X, RRS_F = let rng = StableRNG(690)
    T, N = 300, 5
    F = 0.01 .* randn(rng, T, 2)
    L = [0.9 0.2; 0.5 0.8; 1.1 -0.3; 0.3 0.6; 0.7 0.4]
    F * L' .+ 0.008 .* randn(rng, T, N) .+ 0.0005 .* (1:N)', F
end
const RRS_RD = ReturnsResult(; nx = ["a$i" for i in 1:5], X = RRS_X, nf = ["f1", "f2"],
                             F = RRS_F)
const RRS_PR = prior(EmpiricalPrior(), RRS_RD)
const RRS_B = [0.1, 0.15, 0.2, 0.25, 0.3]
const RRS_SLV = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                       check_sol = (; allow_local = true, allow_almost = true),
                       settings = Dict("verbose" => false, "tol_gap_abs" => 1e-11,
                                       "tol_gap_rel" => 1e-11, "tol_feas" => 1e-11))
rrs_share(w) = w .* (RRS_PR.sigma * w) ./ dot(w, RRS_PR.sigma * w)
function rrs_oracle(model::Symbol; lambda = 1.0, R = nothing)
    S = RRS_PR.sigma
    n = size(S, 1)
    G = cholesky(Symmetric(S)).U
    m = JuMP.Model(Clarabel.Optimizer)
    set_silent(m)
    for k in ("tol_gap_abs", "tol_gap_rel", "tol_feas")
        set_attribute(m, k, 1e-11)
    end
    @variable(m, x[1:n] >= 0)
    @variable(m, zeta[1:n] >= 0)
    @variable(m, psi >= 0)
    @variable(m, gamma >= 0)
    @variable(m, rho >= 0)
    @constraint(m, zeta .== S * x)
    @constraint(m, sum(x) == 1)
    # x_i zeta_i >= gamma^2.
    @constraint(m, [i = 1:n], [x[i], zeta[i], sqrt(2) * gamma] in RotatedSecondOrderCone())
    if model == :A
        @constraint(m, [sqrt(n) * psi; G * x] in SecondOrderCone())
    else
        @constraint(m, [sqrt(n) * psi; G * x; sqrt(n) * rho] in SecondOrderCone())
        if model == :B
            @constraint(m, [rho; G * x] in SecondOrderCone())
        else
            @constraint(m, [rho; sqrt(lambda) * sqrt.(diag(S)) .* x] in SecondOrderCone())
        end
    end
    if !(isnothing(R))
        @constraint(m, dot(RRS_PR.mu, x) >= R)
    end
    @objective(m, Min, psi - gamma)
    optimize!(m)
    return value.(x)
end
function rrs_solve(alg; rba = AssetRiskBudgeting(), R = nothing, kwargs...)
    ret = if isnothing(R)
        ArithmeticReturn()
    else
        ArithmeticReturn(; settings = JuMPReturnsSettings(; lb = R))
    end
    opt = JuMPOptimiser(; pe = RRS_PR, slv = RRS_SLV, ret = ret, kwargs...)
    return optimise(RelaxedRiskBudgeting(; opt = opt, rba = rba, alg = alg), RRS_RD)
end

@testset "RelaxedRiskBudgeting sweep: Models (A), (B) and (C) of gambetakwon2020" begin
    mu_rp = dot(RRS_PR.mu, rrs_oracle(:A))
    cases = ((BasicRelaxedRiskBudgeting(), :A, 1.0),
             (RegularisedRelaxedRiskBudgeting(), :B, 1.0),
             (RegularisedPenalisedRelaxedRiskBudgeting(; p = 0.2), :C, 0.2),
             (RegularisedPenalisedRelaxedRiskBudgeting(), :C, 1.0),
             (RegularisedPenalisedRelaxedRiskBudgeting(; p = 5), :C, 5.0))
    # Without a target return every model gives the equal contributions. With the target
    # return of equation 16, Models (B) and (C) at lambda = 5 leave the portfolio of
    # Model (A), so a regulating term that misses the factor n fails here.
    for R in (nothing, 1.2 * mu_rp, 1.5 * mu_rp), (alg, model, lambda) in cases
        res = rrs_solve(alg; R = R)
        @test isa(res.retcode, OptimisationSuccess)
        @test isapprox(res.w, rrs_oracle(model; lambda = lambda, R = R); atol = 1e-5)
        if isnothing(R)
            @test isapprox(rrs_share(res.w), fill(0.2, 5); atol = 1e-5)
        end
    end
    wa = rrs_solve(BasicRelaxedRiskBudgeting(); R = 1.2 * mu_rp).w
    wb = rrs_solve(RegularisedRelaxedRiskBudgeting(); R = 1.2 * mu_rp).w
    wc = rrs_solve(RegularisedPenalisedRelaxedRiskBudgeting(; p = 5); R = 1.2 * mu_rp).w
    @test maximum(abs, wb - wa) > 1e-2
    @test maximum(abs, wc - wa) > 1e-2
end

@testset "RelaxedRiskBudgeting sweep: Model (A) meets a budget when no row binds" begin
    rba = AssetRiskBudgeting(; rkb = RiskBudget(; val = RRS_B))
    res = rrs_solve(BasicRelaxedRiskBudgeting(); rba = rba)
    @test isa(res.retcode, OptimisationSuccess)
    @test isapprox(rrs_share(res.w), RRS_B; atol = 1e-5)
    # The chain gamma^2 <= sum_i w_i zeta_i <= psi^2 holds the objective at zero.
    psi, gamma = value(res.model[:psi]), value(res.model[:gamma])
    @test isapprox(psi, sqrt(dot(res.w, RRS_PR.sigma, res.w)); rtol = 1e-6)
    @test isapprox(psi - gamma, 0; atol = 1e-8)
    rb = optimise(RiskBudgeting(; opt = JuMPOptimiser(; pe = RRS_PR, slv = RRS_SLV),
                                r = Variance(), rba = rba), RRS_RD)
    @test isapprox(res.w, rb.w; atol = 1e-5)
    # The cones read the shares of the budget, so its sum does not move the portfolio.
    res10 = rrs_solve(BasicRelaxedRiskBudgeting();
                      rba = AssetRiskBudgeting(; rkb = RiskBudget(; val = 10 * RRS_B)))
    @test isapprox(res10.w, res.w; atol = 1e-6)
end

@testset "RelaxedRiskBudgeting sweep: the factor forms" begin
    rr = regression(StepwiseRegression(), RRS_RD)
    b = [0.3, 0.7]
    kw = (; sbgt = BudgetRange(; lb = 0, ub = nothing), bgt = 1,
          wb = WeightBounds(; lb = nothing, ub = nothing))
    # The objective of RiskBudgeting is a variance of 2.4e-4, and Clarabel reads an
    # absolute gap on a cost below one. At so = 1 the solve stops 1.3e-5 short of the
    # budget under flag = false (#1906). At so = 1000 the largest share error is below
    # 5e-7 in each form. The warning of RiskBudgeting states the measured grid.
    opt = JuMPOptimiser(; pe = RRS_PR, slv = RRS_SLV, so = 1000, kw...)
    rv = factory(Variance(), RRS_PR, RRS_SLV)
    algs = (BasicRelaxedRiskBudgeting(), RegularisedRelaxedRiskBudgeting(),
            RegularisedPenalisedRelaxedRiskBudgeting())
    for (flag, hedge) in ((false, false), (true, false), (true, true))
        rba = FactorRiskBudgeting(; re = rr, rkb = RiskBudget(; val = b), flag = flag,
                                  hedge = hedge)
        rb = optimise(RiskBudgeting(; opt = opt, r = Variance(), rba = rba), RRS_RD)
        for alg in algs
            res = rrs_solve(alg; rba = rba, kw...)
            @test isa(res.retcode, OptimisationSuccess)
            rkc = factor_risk_contribution(rv, res.w, RRS_PR.X; re = res.prb.rr)
            sigma2 = dot(res.w, RRS_PR.sigma, res.w)
            x = value.(res.model[:w1])
            zeta = value.(res.model[:zeta])
            # The factor terms x_j zeta_j are the Euler contributions of the factors. The
            # row crrp holds to the feasibility tolerance of the solver, 6.2e-8 on an entry
            # of 9.2e-5 under hedge = true, so the shares agree to 4.7e-4 there.
            @test isapprox(x .* zeta / sigma2, rkc[1:2] / sigma2; atol = 1e-3)
            if flag && !hedge
                # The off-factor contribution c_2 takes no budget, so the objective of
                # Model (A) falls below zero, and the optimum is not the one of
                # RiskBudgeting.
                if isa(alg, BasicRelaxedRiskBudgeting)
                    psi, gamma = value(res.model[:psi]), value(res.model[:gamma])
                    @test psi - gamma < -1e-6
                    @test gamma^2 <= sigma2 - rkc[3] + 1e-12
                    @test maximum(abs, res.w - rb.w) > 1e-2
                end
            else
                # With no off-factor term, or with the hedge rows, the relaxation is exact.
                # One CI host kind misses the shares by a norm of 4.7e-5 and the weights of
                # RiskBudgeting by 2.8e-5, on 16 of 21 runs from 40939998281 to 41050705423
                # (G1 of #1817). The other host kind and the local host stay below 1e-5.
                @test isapprox(rkc[1:2] / sigma2, b; atol = 1e-4)
                @test isapprox(rkc[3], 0; atol = 1e-12)
                @test isapprox(res.w, rb.w; atol = 1e-4)
            end
        end
    end
end

@testset "RelaxedRiskBudgeting sweep: validation" begin
    @test_throws DomainError RegularisedPenalisedRelaxedRiskBudgeting(; p = 0)
    @test_throws DomainError RegularisedPenalisedRelaxedRiskBudgeting(; p = Inf)
    @test_throws IsEmptyError RelaxedRiskBudgeting(; opt = JuMPOptimiser(; slv = RRS_SLV),
                                                   wi = Float64[])
end
