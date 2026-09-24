include(joinpath(@__DIR__, "test18_setup.jl"))

# The regularisation penalties. Each testset reads the model back by the names that the
# `# JuMP formulation` of the builder states, and checks the entries against the closed form
# of its `# Mathematical definition`.

@testset "Every penalty reads back as its closed form" begin
    # Short positions and a utility objective make the 1-norm exceed one, so the L1 penalty
    # has something to price. All four builders act in one model, and their penalties add.
    l2s = [L2Regularisation(; val = 1e-3, alg = SOCRiskExpr()),
           L2Regularisation(; val = 2e-3, alg = SquaredSOCRiskExpr()),
           L2Regularisation(; val = 3e-3, alg = QuadRiskExpr()),
           L2Regularisation(; val = 4e-3, alg = RSOCRiskExpr())]
    lps = [LpRegularisation(; p = 1.5, val = 1e-3), LpRegularisation(; p = 3, val = 2e-3)]
    opt = JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = -1, ub = 1),
                        sbgt = 1, bgt = 1, l1 = 5e-4, linf = 2e-3, l2 = l2s, lp = lps)
    res = optimise(MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumUtility(; l = 2),
                            opt = opt); save = true)
    @test isa(res.retcode, OptimisationSuccess)
    m = res.model
    w = res.w
    n1, n2, ninf = norm(w, 1), norm(w, 2), norm(w, Inf)
    @test n1 > 1.5
    # The objective pulls every epigraph variable onto its norm.
    @test isapprox(value(m[:t_l1]), n1; atol = 1e-6)
    @test isapprox(value(m[:t_linf]), ninf; atol = 1e-6)
    @test isapprox(value(m[:t_l2_1]), n2; atol = 1e-6)
    @test isapprox(value(m[:t_l2_2]), n2; atol = 1e-6)
    @test isapprox(value(m[:t_l2_4]), n2^2; atol = 1e-6)
    @test isapprox(value(m[:t_lp_1]), norm(w, 1.5); atol = 1e-6)
    @test isapprox(value(m[:t_lp_2]), norm(w, 3); atol = 1e-6)
    @test isapprox(sum(value.(m[:r_lp_1])), value(m[:t_lp_1]); atol = 1e-8)
    # The rows carry the names the docstrings state, one set for each term.
    for key in
        (:cl1_noc, :clinf_nic, :cl2_soc_1, :cl2_soc_2, :cl2_rsoc_4, :clp_1, :cslp_1, :clp_2,
         :cslp_2)
        @test haskey(m, key)
    end
    @test length(m[:clp_1]) == length(w)
    # QuadRiskExpr penalises the weights directly and creates no epigraph variable.
    @test !haskey(m, :t_l2_3)
    @test !haskey(m, :t_lp_3)
    # Each expression is the penalty of its term, and `op` is their sum.
    pens = (l1 = 5e-4 * n1, linf = 2e-3 * ninf, l2_1 = 1e-3 * n2, l2_2 = 2e-3 * n2^2,
            l2_3 = 3e-3 * n2^2, l2_4 = 4e-3 * n2^2, lp_1 = 1e-3 * norm(w, 1.5),
            lp_2 = 2e-3 * norm(w, 3))
    for (key, pen) in pairs(pens)
        @test isapprox(value(m[key]), pen; atol = 1e-8)
    end
    @test isapprox(value(m[:op]), sum(pens); atol = 1e-8)
end

@testset "A long-only L1 penalty is a constant" begin
    # With no short positions and a budget of one, the 1-norm is one at every feasible point.
    # The optimum of the variance is flat, so the weights move within the solver tolerance
    # while the variance does not.
    S = pr.sigma
    w0 = optimise(MeanRisk(; r = Variance(), opt = JuMPOptimiser(; pe = pr, slv = slv))).w
    w1 = optimise(MeanRisk(; r = Variance(),
                           opt = JuMPOptimiser(; pe = pr, slv = slv, l1 = 1e-1))).w
    @test isapprox(norm(w1, 1), 1; atol = 1e-8)
    @test isapprox(dot(w1, S, w1), dot(w0, S, w0); rtol = 1e-5)
end

@testset "Under MaximumRatio the penalty acts on the homogenised weights" begin
    # The ratio's model weights are k * w. A penalty on them is the penalty of the returned
    # weights times k, in the form that bounds the return and in the form that bounds the risk.
    for (sr_risk, kw) in ((false, (;)),
                          (true,
                           (; ret = LogarithmicReturn(), wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1,
                            bgt = 1)))
        opt = JuMPOptimiser(; pe = pr, slv = slv, l1 = 1e-4, kw...)
        res = optimise(MeanRisk(; r = StandardDeviation(), obj = MaximumRatio(), opt = opt);
                       save = true)
        @test isa(res.retcode, OptimisationSuccess)
        m = res.model
        @test haskey(m, :sr_risk) == sr_risk
        k = value(m[:k])
        @test isapprox(value.(m[:w]), k * res.w; atol = 1e-10)
        @test isapprox(value(m[:op]), 1e-4 * k * norm(res.w, 1); atol = 1e-8)
    end
    # A penalty of power one scales with k as the ratio does, so the normalisation of the ratio
    # does not move the weights.
    ohf = mean(abs.(pr.mu))
    w_of(ohf) = optimise(MeanRisk(; r = StandardDeviation(),
                                  obj = MaximumRatio(; ohf = ohf),
                                  opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                      l2 = L2Regularisation(; val = 1e-4)))).w
    @test isapprox(w_of(ohf), w_of(10 * ohf); atol = 1e-4)
end

@testset "The builders refuse a coefficient that is not positive and finite" begin
    @test_throws DomainError PortfolioOptimisers.set_l1_regularisation!(JuMP.Model(), 0.0)
    @test_throws DomainError PortfolioOptimisers.set_linf_regularisation!(JuMP.Model(), Inf)
    @test_throws DomainError LpRegularisation(; p = 1)
    @test_throws IsNonFiniteError LpRegularisation(; p = Inf)
    @test_throws DomainError L2Regularisation(; val = -1.0)
end
