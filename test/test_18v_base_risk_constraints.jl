include(joinpath(@__DIR__, "test18_setup.jl"))

# The base layer of the risk measure constraints: the scalariser rows, the drawdown rows and the
# number bound. Each testset reads the model back and checks it against the closed form that
# the docstring of the builder states.

@testset "The scalarisers reduce risk_vec to the stated risk" begin
    r = [StandardDeviation(; settings = RiskMeasureSettings(; scale = 2.0)),
         ConditionalValueatRisk(; settings = RiskMeasureSettings(; scale = 0.5))]
    gamma = 50.0
    for (sca, f) in ((SumScalariser(), sum), (MaxScalariser(), maximum),
                     (LogSumExpScalariser(; gamma = gamma), x -> log(sum(exp.(gamma * x))) / gamma))
        opt = JuMPOptimiser(; pe = pr, slv = slv, sca = sca)
        res = optimise(MeanRisk(; r = r, opt = opt))
        @test isa(res.retcode, OptimisationSuccess)
        risk_vec = JuMP.value.(res.model[:risk_vec])
        risk = JuMP.value(res.model[:risk])
        # A minimum risk objective pulls `risk` down onto the scalarised value.
        @test isapprox(risk, f(risk_vec); rtol = 1e-6)
        # The rows hold `risk` at or above every entry.
        @test all(risk_vec .<= risk + 1e-8)
    end
end

@testset "A scalariser registers nothing when no measure reaches risk_vec" begin
    rt = optimise(MeanRisk(; r = StandardDeviation(), obj = MaximumReturn(),
                           opt = JuMPOptimiser(; pe = pr, slv = slv)))
    ub = expected_risk(factory(StandardDeviation(), pr), rt.w, pr) / 2
    r = StandardDeviation(; settings = RiskMeasureSettings(; rke = false, ub = ub))
    for sca in (SumScalariser(), MaxScalariser(), LogSumExpScalariser())
        opt = JuMPOptimiser(; pe = pr, slv = slv, sca = sca)
        res = optimise(MeanRisk(; r = r, obj = MaximumReturn(), opt = opt))
        @test isa(res.retcode, OptimisationSuccess)
        @test !haskey(res.model, :risk_vec)
        @test !haskey(res.model, :risk)
        # The measure reaches the model through its bound alone.
        @test expected_risk(factory(StandardDeviation(), pr), res.w, pr) <= ub * (1 + 1e-6)
    end
end

@testset "The drawdown rows bound the drawdown from above" begin
    opt = JuMPOptimiser(; pe = pr, slv = slv)
    drawdown(w) = begin
        c = [zero(eltype(w)); cumsum(pr.X * w)]
        [maximum(view(c, 1:t)) - c[t] for t in eachindex(c)]
    end
    # A sum over the entries pulls every entry onto the drawdown.
    res = optimise(MeanRisk(; r = AverageDrawdown(), opt = opt))
    dd = JuMP.value.(res.model[:dd])
    @test abs(dd[1]) < 1e-10
    @test isapprox(dd, drawdown(res.w); atol = 1e-8)
    # A maximum pulls on the largest entry alone. Every entry stays at or above the drawdown.
    res = optimise(MeanRisk(; r = MaximumDrawdown(), opt = opt))
    dd = JuMP.value.(res.model[:dd])
    DD = drawdown(res.w)
    @test all(dd .>= DD .- 1e-8)
    @test isapprox(maximum(dd), maximum(DD); rtol = 1e-6)
end

@testset "A number bound holds on the weights a ratio objective returns" begin
    opt = JuMPOptimiser(; pe = pr, slv = slv)
    rsd = factory(StandardDeviation(), pr)
    sdmin = expected_risk(rsd, optimise(MeanRisk(; r = StandardDeviation(), opt = opt)).w,
                          pr)
    ub = 1.05 * sdmin
    r = StandardDeviation(; settings = RiskMeasureSettings(; ub = ub))
    res = optimise(MeanRisk(; r = r, obj = MaximumRatio(), opt = opt))
    @test isa(res.retcode, OptimisationSuccess)
    # The row is `sc * (risk - ub * k) <= 0`, and the head divides the weights by `k`.
    @test !isapprox(JuMP.value(res.model[:k]), 1; rtol = 1e-3)
    @test isapprox(expected_risk(rsd, res.w, pr), ub; rtol = 1e-6)
end

using Test, PortfolioOptimisers, JuMP, Clarabel, StableRNGs, LinearAlgebra

# The weight norm ceilings and the dual norm epigraph. Each testset solves a model and reads
# the reported weights, or the model entries, against the closed form that the docstring of the
# builder states. The fixture is a small synthetic panel so the file runs in seconds.

const PO = PortfolioOptimisers

X_wn = 0.01 * randn(StableRNG(42), 150, 8) .+ 0.001 * (1:8)'
pr_wn = prior(EmpiricalPrior(), X_wn)
slv_wn = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                check_sol = (; allow_local = true, allow_almost = true),
                settings = "verbose" => false)
function solve_wn(head, kw; bgt = 1)
    opt = JuMPOptimiser(; pe = pr_wn, slv = slv_wn, bgt = bgt, kw...)
    est = head === :rb ? RiskBudgeting(; opt = opt) : MeanRisk(; obj = head, opt = opt)
    return optimise(est)
end
entry_names(res) = Set(keys(JuMP.object_dictionary(res.model)))

@testset "Weight norm ceilings bind on the reported weights" begin
    cases = ((; l2c = 0.4), x -> norm(x, 2), 0.4, (:l2c, :cl2c_soc, :cl2c)),
            ((; lpc = LpRegularisation(; p = 3, val = 0.3)), x -> norm(x, 3), 0.3,
             (:t_lpc_1, :r_lpc_1, :clpc_1, :cslpc_1, :clpc_bnd_1)),
            ((; linfc = 0.2), x -> norm(x, Inf), 0.2, (:t_linfc, :clinfc_nic, :clinfc))
    for (kw, f, val, names) in cases
        # Without the ceiling both objectives sit above it, so the ceiling binds.
        for obj in (MaximumReturn(), MaximumRatio())
            @test f(solve_wn(obj, (;)).w) > val + 1e-3
            res = solve_wn(obj, kw)
            @test isa(res.retcode, OptimisationSuccess)
            # MaximumRatio stops about 1.5e-5 below the ceiling, on the feasible side.
            @test f(res.w) <= val * (1 + 1e-6)
            @test isapprox(f(res.w), val; rtol = 1e-4)
            @test all(n -> n in entry_names(res), names)
        end
        # Under MaximumRatio `k` is not one, and the ceiling still holds on `w / k`.
        @test !isapprox(JuMP.value(solve_wn(MaximumRatio(), kw).model[:k]), 1; atol = 1e-2)
    end
end

@testset "The budget does not scale a weight norm ceiling" begin
    res = solve_wn(MaximumReturn(), (; l2c = 0.4); bgt = 0.5)
    @test isapprox(sum(res.w), 0.5; rtol = 1e-6)
    @test isapprox(norm(res.w, 2), 0.4; rtol = 1e-5)
end

@testset "RiskBudgeting: a weight norm ceiling holds on the renormalised weights" begin
    res = solve_wn(:rb, (; linfc = 0.14))
    @test JuMP.value(res.model[:k]) > 2
    @test isapprox(sum(res.w), 1; rtol = 1e-6)
    @test isapprox(norm(res.w, Inf), 0.14; rtol = 1e-5)
    # At the floor `N^(1/p - 1)` the only admitted weights are the equal weights.
    res = solve_wn(:rb, (; lpc = LpRegularisation(; p = 3, val = 8^(1 / 3 - 1))))
    @test isapprox(res.w, fill(1 / 8, 8); rtol = 1e-4)
end

@testset "A weight norm ceiling below the floor is infeasible" begin
    # Eight weights that sum to one have a 2-norm of at least `1 / sqrt(8) ≈ 0.354`.
    @test isa(solve_wn(MinimumRisk(), (; l2c = 0.34)).retcode, OptimisationFailure)
    @test isa(solve_wn(MinimumRisk(), (; linfc = 0.12)).retcode, OptimisationFailure)
end

@testset "A weight norm ceiling states a floor on the effective number of assets" begin
    m = 5
    res = solve_wn(MaximumReturn(), (; l2c = inv(sqrt(m))))
    @test number_effective_assets(res.w) >= m - 1e-6
    res = solve_wn(MaximumReturn(),
                   (; lpc = LpRegularisation(; p = 3, val = m^(1 / 3 - 1))))
    @test sum(abs.(res.w) .^ 3)^inv(1 - 3) >= m - 1e-6
    res = solve_wn(MaximumReturn(), (; linfc = 1 / m))
    @test inv(norm(res.w, Inf)) >= m - 1e-6
    @test count(>(1e-6), abs.(res.w)) >= m
end

@testset "Weight norm ceilings refuse a value that is not positive and finite" begin
    model = JuMP.Model()
    PO.set_model_scales!(model, 1.0, 1.0)
    for (f, name) in ((PO.set_weight_norm_2_constraints!, "l2c"),
                      (PO.set_weight_norm_inf_constraints!, "linfc")),
        val in (0.0, -1.0, Inf)

        err = try
            f(model, val)
            nothing
        catch e
            e
        end
        @test isa(err, DomainError)
        @test occursin(name, sprint(showerror, err))
    end
    # Any other argument adds nothing to the model.
    @test isnothing(PO.set_weight_norm_2_constraints!(model, nothing))
    @test isnothing(PO.set_weight_norm_p_constraints!(model, nothing))
    @test isnothing(PO.set_weight_norm_inf_constraints!(model, nothing))
    @test isempty(JuMP.all_variables(model))
end

@testset "The dual norm epigraph reaches the dual norm on every route" begin
    xv = [0.3, -0.7, 0.05, 1.2, -0.4]
    for p in (2, Inf, 1, 3, 1.5)
        model = JuMP.Model(Clarabel.Optimizer)
        JuMP.set_silent(model)
        # A constraint scale other than one leaves the feasible set unchanged.
        PO.set_model_scales!(model, 3.0, 1.0)
        JuMP.@variable(model, x[1:length(xv)])
        JuMP.fix.(x, xv)
        t = PO.norm_ball_dual_norm_epigraph!(model, :probe_, 1, x, p)
        JuMP.@objective(model, Min, t)
        JuMP.optimize!(model)
        q = PO.dual_norm_order(p)
        @test isapprox(JuMP.value(t), norm(xv, q); rtol = 1e-6)
        @test haskey(model, :probe_t_nbucs_1)
        @test haskey(model, :probe_nbucs_cone_1)
        power = !(q in (1, 2, Inf))
        @test haskey(model, :probe_r_nbucs_1) == power
        @test haskey(model, :probe_nbucs_cone_sum_1) == power
    end
end

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
    # QuadRiskExpr penalises the weights directly and creates no epigraph variable. With
    # k = 1 no squared formulation divides by k through a cone.
    @test !haskey(m, :t_l2_3)
    @test !haskey(m, :t_lp_3)
    @test !haskey(m, :t_l2_sq_2)
    @test !haskey(m, :t_l2_sq_3)
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

@testset "Under MaximumRatio a squared penalty is divided by k" begin
    # Issue #1318. The model penalises norm(k * w)^2 / k = k * norm(w)^2, which has degree one
    # in k as the standard deviation has, so the normalisation of the ratio does not move the
    # weights. Before the division the weights moved by 0.25 between ohf and 10 * ohf.
    ohf = mean(abs.(pr.mu))
    for (sr_risk, kw) in ((false, (;)), (true, (; ret = LogarithmicReturn())))
        w_of(alg, ohf) = optimise(MeanRisk(; r = StandardDeviation(),
                                           obj = MaximumRatio(; ohf = ohf),
                                           opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                               l2 = L2Regularisation(;
                                                                                     val = 1e-2,
                                                                                     alg = alg),
                                                               kw...)); save = true)
        w0 = optimise(MeanRisk(; r = StandardDeviation(), obj = MaximumRatio(; ohf = ohf),
                               opt = JuMPOptimiser(; pe = pr, slv = slv, kw...))).w
        ws = map((SquaredSOCRiskExpr(), QuadRiskExpr(), RSOCRiskExpr())) do alg
            res = w_of(alg, ohf)
            @test isa(res.retcode, OptimisationSuccess)
            m = res.model
            @test haskey(m, :sr_risk) == sr_risk
            k = value(m[:k])
            n2 = norm(res.w, 2)
            @test isapprox(value(m[:l2_1]), 1e-2 * k * n2^2; rtol = 1e-5)
            if isa(alg, RSOCRiskExpr)
                @test isapprox(value(m[:t_l2_1]), k * n2^2; rtol = 1e-5)
            else
                @test isapprox(value(m[:t_l2_sq_1]), k * n2^2; rtol = 1e-5)
                @test haskey(m, :cl2_sq_rsoc_1)
            end
            @test isapprox(res.w, w_of(alg, 10 * ohf).w; atol = 1e-4)
            res.w
        end
        # The penalty moves the weights, and the three squared formulations agree.
        @test norm(ws[1] - w0, Inf) > 0.1
        @test isapprox(ws[1], ws[2]; atol = 1e-4)
        @test isapprox(ws[1], ws[3]; atol = 1e-4)
    end
end

@testset "The builders refuse a coefficient that is not positive and finite" begin
    @test_throws DomainError PortfolioOptimisers.set_l1_regularisation!(JuMP.Model(), 0.0)
    @test_throws DomainError PortfolioOptimisers.set_linf_regularisation!(JuMP.Model(), Inf)
    @test_throws DomainError LpRegularisation(; p = 1)
    @test_throws IsNonFiniteError LpRegularisation(; p = Inf)
    @test_throws DomainError L2Regularisation(; val = -1.0)
end

using Test, PortfolioOptimisers, JuMP, Clarabel, HiGHS, StableRNGs, LinearAlgebra

# The fee builders of a JuMP model. Each testset solves a model and reads its entries, or the
# weights, against the closed form that the docstring of the builder states. The fixture is a
# small synthetic panel, so a MIP solve takes well under a second.

const PO = PortfolioOptimisers

X_fc = 0.01 * randn(StableRNG(42), 80, 6) .+ 0.001 * (1:6)'
pr_fc = prior(EmpiricalPrior(), X_fc)
T_fc, N_fc = size(X_fc)
hslv_fc = Solver(; name = :highs, solver = HiGHS.Optimizer,
                 settings = "log_to_console" => false,
                 check_sol = (; allow_local = true, allow_almost = true))
cslv_fc = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = "verbose" => false,
                 check_sol = (; allow_local = true, allow_almost = true))
function solve_fc(obj; r = ConditionalValueatRisk(), slv = hslv_fc, kw...)
    opt = JuMPOptimiser(; pe = pr_fc, slv = slv, kw...)
    return optimise(MeanRisk(; r = r, obj = obj, opt = opt))
end
entry_names_fc(res) = Set(keys(JuMP.object_dictionary(res.model)))
# CVaR of a return series, from its linear programme.
function cvar_fc(r, alpha)
    m = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(m)
    JuMP.@variable(m, t)
    JuMP.@variable(m, z[1:length(r)] >= 0)
    JuMP.@constraint(m, z .>= -r .- t)
    JuMP.@objective(m, Min, t + sum(z) / (alpha * length(r)))
    JuMP.optimize!(m)
    return JuMP.objective_value(m)
end

@testset "A fixed fee scales with k, and the ratio reaches its optimum" begin
    c = 0.002
    fees = Fees(; fl = c)
    alpha = ConditionalValueatRisk().alpha
    e1 = [1.0; zeros(T_fc - 1)]
    # The ratio the model states: the expected return pays the one-off fee spread over the
    # fit, and the CVaR reads the net series, whose first observation pays it whole.
    function ratio(w)
        C = c * count(>(1e-6), w)
        return (dot(pr_fc.mu, w) - C / T_fc) / cvar_fc(X_fc * w .- C .* e1, alpha)
    end
    # The optimum over every support. On a fixed support the fee is a constant, so each
    # support is one Charnes-Cooper linear programme.
    best = -Inf
    for mask in 1:(2 ^ N_fc - 1)
        S = [(mask >> (i - 1)) & 1 == 1 for i in 1:N_fc]
        C = c * sum(S)
        m = JuMP.Model(HiGHS.Optimizer)
        JuMP.set_silent(m)
        JuMP.@variable(m, y[1:N_fc] >= 0)
        JuMP.@variable(m, k >= 0)
        JuMP.@variable(m, t)
        JuMP.@variable(m, z[1:T_fc] >= 0)
        JuMP.@constraint(m, [i in 1:N_fc; !S[i]], y[i] == 0)
        JuMP.@constraint(m, sum(y) == k)
        JuMP.@constraint(m, z .>= -(X_fc * y .- C * k .* e1) .- t)
        JuMP.@constraint(m, t + sum(z) / (alpha * T_fc) <= 1)
        JuMP.@objective(m, Max, dot(pr_fc.mu, y) - C / T_fc * k)
        JuMP.optimize!(m)
        if JuMP.termination_status(m) == JuMP.MOI.OPTIMAL
            best = max(best, JuMP.objective_value(m))
        end
    end
    ws = map((1e-3, 1e-1, 1.0)) do ohf
        res = solve_fc(MaximumRatio(; ohf = ohf); fees = fees)
        @test isa(res.retcode, OptimisationSuccess)
        m = res.model
        # The fee is the rate times the gate, and the gate is k on each held position.
        k = JuMP.value(m[:k])
        @test isapprox(JuMP.value(m[:ffl]), c * k * sum(round.(JuMP.value.(m[:ib])));
                       rtol = 1e-6)
        @test isapprox(ratio(res.w), best; rtol = 1e-6)
        return res.w
    end
    # The weights do not depend on `ohf`.
    @test isapprox(ws[1], ws[2]; atol = 1e-6)
    @test isapprox(ws[1], ws[3]; atol = 1e-6)
    # The long-short bundle charges each side on its own gate.
    res = solve_fc(MaximumRatio(; ohf = 0.1); fees = Fees(; fl = 1e-4, fs = 2e-4),
                   wb = WeightBounds(; lb = -1.0, ub = 1.0), sbgt = 0.5)
    @test isa(res.retcode, OptimisationSuccess)
    m = res.model
    k = JuMP.value(m[:k])
    @test isapprox(JuMP.value(m[:ffl]), 1e-4 * k * sum(round.(JuMP.value.(m[:ilb])));
                   rtol = 1e-6)
    @test isapprox(JuMP.value(m[:ffs]), 2e-4 * k * sum(round.(JuMP.value.(m[:isb])));
                   rtol = 1e-6)
    # Under a constant k the gate is the binary itself.
    res = solve_fc(MinimumRisk(); fees = fees)
    @test isapprox(JuMP.value(res.model[:ffl]),
                   c * sum(round.(JuMP.value.(res.model[:ib]))))
end

@testset "A fixed fee alone reaches the net return series" begin
    for (fa, cf) in ((nothing, t -> t == 1 ? 1.0 : 0.0), (AmortisedFees(), t -> 1 / T_fc))
        res = solve_fc(MinimumRisk(); fees = Fees(; fl = 0.002, fa = fa))
        m = res.model
        @test !haskey(m, :fees)
        net = JuMP.value.(m[:net_X])
        gross = X_fc * JuMP.value.(m[:w])
        f_o = JuMP.value(m[:one_time_fees])
        @test f_o > 0
        @test all(isapprox(net[t], gross[t] - cf(t) * f_o; atol = 1e-12) for t in 1:T_fc)
        # A zero per period fee registers `:fees`, and it leaves the weights alone.
        res0 = solve_fc(MinimumRisk(); fees = Fees(; fl = 0.002, fa = fa, l = 0.0))
        @test haskey(res0.model, :fees)
        @test isapprox(res0.w, res.w; atol = 1e-6)
    end
end

@testset "The turnover fee is an epigraph, tight when the fee is read" begin
    wt = fill(1 / N_fc, N_fc)
    fees = Fees(; tn = Turnover(; w = wt, val = 0.01))
    res = solve_fc(MaximumReturn(); r = Variance(), slv = cslv_fc, fees = fees)
    @test isa(res.retcode, OptimisationSuccess)
    m = res.model
    @test all(n -> n in entry_names_fc(res), (:t_ftn, :x_ftn, :ftn, :cftn, :fees))
    x = JuMP.value.(m[:w]) .- wt .* JuMP.value(m[:k])
    @test isapprox(JuMP.value.(m[:x_ftn]), x; atol = 1e-10)
    @test isapprox(JuMP.value.(m[:t_ftn]), abs.(x); atol = 1e-7)
    @test isapprox(JuMP.value(m[:ftn]), 0.01 * sum(abs, x); atol = 1e-8)
    # A variance does not read the net series, so under a minimum risk objective nothing
    # pulls on the fee: a larger epigraph entry leaves the objective and the weights.
    res = solve_fc(MinimumRisk(); r = Variance(), slv = cslv_fc, fees = fees)
    m = res.model
    obj = JuMP.objective_value(m)
    w = JuMP.value.(m[:w])
    JuMP.set_lower_bound(m[:t_ftn][1], 1.0)
    JuMP.optimize!(m)
    @test isapprox(JuMP.objective_value(m), obj; rtol = 1e-5)
    @test isapprox(JuMP.value.(m[:w]), w; atol = 1e-4)
    @test JuMP.value(m[:ftn]) > 0.01 * sum(abs, w .- wt) + 1e-3
end

@testset "The proportional fees charge the parts of the weights" begin
    fees = Fees(; l = 0.001, s = 0.002)
    # Long only: the long part is the weights, and the model registers no short part.
    res = solve_fc(MaximumReturn(); slv = cslv_fc, fees = fees)
    @test haskey(res.model, :fl)
    @test !haskey(res.model, :fs)
    @test isapprox(JuMP.value(res.model[:fl]), 0.001 * sum(res.w); atol = 1e-9)
    # Long-short: the budget pins the sums of the parts. Without the pin the minimum risk
    # portfolio holds no short position and still pays both fees in full; with the pin the
    # fees agree with the weights.
    kw = (; wb = WeightBounds(; lb = -1.0, ub = 1.0), sbgt = 0.5, fees = fees)
    for (xbgt, gap_l, gap_s) in ((false, 0.0005, 0.001), (true, 0.0, 0.0))
        res = solve_fc(MinimumRisk(); slv = xbgt ? hslv_fc : cslv_fc, xbgt = xbgt, kw...)
        @test isa(res.retcode, OptimisationSuccess)
        m = res.model
        w = JuMP.value.(m[:w])
        @test isapprox(JuMP.value(m[:fl]) - 0.001 * sum(max.(w, 0)), gap_l; atol = 1e-8)
        @test isapprox(JuMP.value(m[:fs]) - 0.002 * sum(max.(-w, 0)), gap_s; atol = 1e-8)
    end
end

@testset "A short credit is charged exactly, against the weights" begin
    # #1518. A negative `s` is a credit. A slack short part would earn it on an exposure the
    # model does not hold, so the builder charges `max(l w, -s w)` per asset against `w`.
    for fees in (Fees(; l = 0.001, s = -0.001),
                 Fees(; l = fill(0.002, N_fc), s = [-0.002, -0.001, 0.0, 0.001, -0.002, 0.0]))
        kw = (; wb = WeightBounds(; lb = -1.0, ub = 1.0), sbgt = 0.5, fees = fees)
        res = solve_fc(MaximumReturn(); slv = cslv_fc, kw...)
        @test isa(res.retcode, OptimisationSuccess)
        m = res.model
        @test all(n -> n in entry_names_fc(res), (:t_fp, :cfp_l, :cfp_s, :fp, :fees))
        @test !haskey(m, :fl)
        @test !haskey(m, :fs)
        w = JuMP.value.(m[:w])
        @test any(<(-1e-6), w)
        # The objective pulls the fee down, so the epigraph is tight and the model's fee is
        # the fee of the realised weights.
        @test isapprox(JuMP.value(m[:fp]), PO.calc_periodic_fees(w, fees); atol = 1e-8)
        # Under minimum risk nothing pulls on the fee, and it still never falls below the
        # definition, whatever the parts hold.
        res = solve_fc(MinimumRisk(); slv = cslv_fc, kw...)
        w = JuMP.value.(res.model[:w])
        @test JuMP.value(res.model[:fp]) >= PO.calc_periodic_fees(w, fees) - 1e-8
    end
    # A long-only model holds no short part and no short position, so its part builders
    # stay exact and charge `l` alone.
    res = solve_fc(MaximumReturn(); slv = cslv_fc, fees = Fees(; l = 0.001, s = -0.001))
    @test haskey(res.model, :fl)
    @test !haskey(res.model, :fp)
    @test isapprox(JuMP.value(res.model[:fl]), 0.001 * sum(res.w); atol = 1e-9)
end

@testset "The liquidation charges are constants times k" begin
    lq = Turnover(; w = [0.1, -0.2, 0.0], val = [0.01, 0.02, 0.03])
    for k in (1, 2.5)
        m = JuMP.Model()
        m[:k] = k
        PO.set_liquidation_fees!(m, lq)
        PO.set_fixed_liquidation_fees!(m, lq, (; atol = 1e-8))
        # The proportional charge prices the traded amount, the fixed one each position sold.
        @test JuMP.value(m[:flq_prop]) ≈ (0.01 * 0.1 + 0.02 * 0.2) * k
        @test JuMP.value(m[:flq_fixed]) ≈ (0.01 + 0.02) * k
        @test JuMP.value(m[:fees]) ≈ JuMP.value(m[:flq_prop])
        @test JuMP.value(m[:one_time_fees]) ≈ JuMP.value(m[:flq_fixed])
        # A second term adds to the registered expression in place.
        JuMP.@variable(m, x)
        PO.add_to_fees!(m, 2x)
        PO.add_to_one_time_fees!(m, 3x)
        @test JuMP.coefficient(m[:fees], x) == 2
        @test JuMP.coefficient(m[:one_time_fees], x) == 3
        @test JuMP.constant(m[:fees]) ≈ (0.01 * 0.1 + 0.02 * 0.2) * k
    end
    m = JuMP.Model()
    @test isnothing(PO.set_liquidation_fees!(m, nothing))
    @test isnothing(PO.set_fixed_liquidation_fees!(m, nothing, (;)))
    @test !haskey(m, :fees)
    @test !haskey(m, :one_time_fees)
end

@testset "A fee refuses a rate of +Inf on a turnover carrier" begin
    tn = Turnover(; w = fill(1 / N_fc, N_fc), val = Inf)
    @test_throws DomainError Fees(; tn = tn)
    @test_throws DomainError Fees(; lq = tn)
    @test_throws DomainError Fees(; flq = tn)
    @test Fees(; tn = Turnover(; w = fill(1 / N_fc, N_fc), val = 0.01)).tn.val == 0.01
end

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

@testset "The default factor of a variance: a Cholesky factor, an eigen factor, or a refusal" begin
    factor(S) = PO.chol_sigma_selector(JuMP.Model(), nothing, Variance(; sigma = S))
    G = factor(S_vc)
    @test G isa UpperTriangular
    @test isapprox(transpose(G) * G, S_vc; rtol = 1e-12)
    # A singular covariance has no Cholesky factor, and its eigen factor still reproduces it.
    v = randn(StableRNG(9), N_vc)
    S1 = v * transpose(v)
    @test !issuccess(cholesky(S1; check = false))
    G1 = factor(S1)
    @test isapprox(transpose(G1) * G1, S1; atol = 1e-12)
    @test_throws PosDefException(-1) factor([1.0 0.5; 0.4 1.0])
    @test_throws PosDefException(1) factor([1.0 2.0; 2.0 1.0])
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

# The builders of the Relativistic Value-at-Risk, its range and the Relativistic
# Drawdown-at-Risk. The oracle is the dual programme of Equation 12 of `rlvar`, the
# supremum of the expected loss over the Kaniadakis ball, written here as its own power
# cone model. It shares no code with the library's primal or dual programme.

X_rl = 0.01 * randn(StableRNG(1791), 150, 8) .+ 0.001 * (1:8)'
rd_rl = ReturnsResult(; X = X_rl, nx = string.("A", 1:8))
pr_rl = prior(EmpiricalPrior(), rd_rl)
# `allow_almost = false`: an almost-solved point can break a power cone row by 1 %, so a
# bound looks breached that the formulation holds.
slv_rl = [Solver(; name = Symbol(:clarabel, j), solver = Clarabel.Optimizer,
                 settings = ["verbose" => false, "max_step_fraction" => f],
                 check_sol = (; allow_local = true, allow_almost = false))
          for (j, f) in enumerate((0.99, 0.95, 0.9, 0.8, 0.7))]
opt_rl = JuMPOptimiser(; pe = pr_rl, slv = slv_rl)
function solve_rl(r, obj = MinimumRisk())
    return optimise(MeanRisk(; r = r, obj = obj, opt = opt_rl), rd_rl)
end
value_rl(res, name, i = 1) = JuMP.value(res.model[PO.state_key(Symbol(""), name, i)])
function rlvar_dual_rl(x, alpha, kappa, p = fill(inv(length(x)), length(x)))
    T = length(x)
    c = p * T
    u = inv(alpha * T)
    lnk = (u^kappa - u^(-kappa)) / (2 * kappa)
    m = JuMP.Model(Clarabel.Optimizer)
    JuMP.set_silent(m)
    JuMP.@variable(m, q[1:T] >= 0)
    JuMP.@variable(m, nu[1:T])
    JuMP.@variable(m, tau[1:T])
    JuMP.@constraint(m, sum(q) == 1)
    # q ln_k(q / c) = (q^(1 + k) c^(-k) - q^(1 - k) c^k) / (2k), one cone for each term.
    JuMP.@constraint(m, [i = 1:T], [nu[i], c[i], q[i]] in MOI.PowerCone(inv(1 + kappa)))
    JuMP.@constraint(m, [i = 1:T], [q[i], c[i], tau[i]] in MOI.PowerCone(1 - kappa))
    JuMP.@constraint(m, sum(nu - tau) / (2 * kappa) <= lnk)
    JuMP.@objective(m, Max, -dot(q, x))
    JuMP.optimize!(m)
    return JuMP.objective_value(m)
end
function drawdown_rl(x)
    return (c = [zero(eltype(x)); cumsum(x)];
            [c[t] - maximum(view(c, 1:t)) for t in 2:length(c)])
end

@testset "The relativistic programme states the RLVaR of the net returns" begin
    res = solve_rl(RelativisticValueatRisk())
    @test isa(res.retcode, OptimisationSuccess)
    entries = Set(keys(JuMP.object_dictionary(res.model)))
    for k in (:t_rlvar_1, :z_rlvar_1, :omega_rlvar_1, :psi_rlvar_1, :theta_rlvar_1,
              :epsilon_rlvar_1, :rlvar_risk_1, :crlvar_pcone_a_1, :crlvar_pcone_b_1, :crlvar_1)
        @test k in entries
    end
    @test isapprox(value_rl(res, :rlvar_risk_), rlvar_dual_rl(X_rl * res.w, 0.05, 0.3);
                   rtol = 1e-5)
    # Observation weights enter the sum and the cones as `T p_i`, and the radius keeps
    # `1 / (alpha T)`.
    ow = eweights(150, 0.02)
    res = solve_rl(RelativisticValueatRisk(; w = ow, alpha = 0.1, kappa = 0.6))
    @test isapprox(value_rl(res, :rlvar_risk_),
                   rlvar_dual_rl(X_rl * res.w, 0.1, 0.6, ow / sum(ow)); rtol = 1e-4)
end

@testset "The RLDaR is the RLVaR of the drawdown series" begin
    res = solve_rl(RelativisticDrawdownatRisk())
    @test isa(res.retcode, OptimisationSuccess)
    @test isapprox(value_rl(res, :rldar_risk_),
                   rlvar_dual_rl(drawdown_rl(X_rl * res.w), 0.05, 0.3); rtol = 1e-5)
end

@testset "The RLVaR range pairs each tail with its own kappa" begin
    r = RelativisticValueatRiskRange(; alpha = 0.05, kappa_a = 0.3, beta = 0.1,
                                     kappa_b = 0.7)
    res = solve_rl(r)
    @test isa(res.retcode, OptimisationSuccess)
    x = X_rl * res.w
    risk = value_rl(res, :rlvar_range_risk_)
    @test isapprox(risk, rlvar_dual_rl(x, 0.05, 0.3) + rlvar_dual_rl(-x, 0.1, 0.7);
                   rtol = 1e-5)
    # The swapped pairing is 0.9 % away, so the check above reads the pairing.
    @test !isapprox(risk, rlvar_dual_rl(x, 0.05, 0.7) + rlvar_dual_rl(-x, 0.1, 0.3);
                    rtol = 1e-3)
    @test isapprox(value_rl(res, :rlvar_risk_, PO.nested_index(:loss_, 1)),
                   rlvar_dual_rl(x, 0.05, 0.3); rtol = 1e-5)
end

@testset "A relativistic bound is exact, and the risk expression lies above the measure" begin
    for (rm, key) in ((RelativisticValueatRisk, :rlvar_risk_),
                      (RelativisticDrawdownatRisk, :rldar_risk_))
        f = rm(; slv = slv_rl)
        lo = value_rl(solve_rl(rm()), key)
        hi = f(X_rl * solve_rl(rm(), MaximumRatio()).w)
        ub = (lo + hi) / 2
        ws = Vector{Float64}[]
        for obj in (MaximumReturn(), MaximumRatio(), MaximumRatio(; ohf = 10.0))
            res = solve_rl(rm(; settings = RiskMeasureSettings(; ub = ub)), obj)
            @test isa(res.retcode, OptimisationSuccess)
            @test isapprox(f(X_rl * res.w), ub; rtol = 1e-4)
            push!(ws, res.w)
        end
        @test isapprox(ws[2], ws[3]; rtol = 1e-3)
        # With no bound and a return objective nothing pulls the expression down.
        res = solve_rl(rm(), MaximumReturn())
        @test value_rl(res, key) >= f(X_rl * res.w) * (1 - 1e-6)
    end
end
