using Test, PortfolioOptimisers, JuMP, Clarabel, HiGHS, StableRNGs, LinearAlgebra,
      Statistics, StatsBase

# The return terms, their builders and the objectives. Each testset solves a model and reads the
# model entries against the closed form that the docstring of the builder states. The worst case
# of each robust set is checked against an oracle that minimises the characteristic over the set
# for the solved weights. The fixture is a small synthetic panel, so the file runs in seconds.

X_ro = 0.01 * randn(StableRNG(42), 150, 6) .+ 0.001 * (1:6)'
pr_ro = prior(EmpiricalPrior(), X_ro)
N_ro = size(X_ro, 2)
slv_ro = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                check_sol = (; allow_local = true, allow_almost = true),
                settings = ["verbose" => false, "tol_gap_abs" => 1e-10,
                            "tol_gap_rel" => 1e-10, "tol_feas" => 1e-10])
function solve_ro(obj, kw; r = StandardDeviation(), slv = slv_ro)
    opt = JuMPOptimiser(; pe = pr_ro, slv = slv, kw...)
    return optimise(MeanRisk(; obj = obj, r = r, opt = opt))
end
# A long-short book, so the worst cases read `|w|` and not `w`.
ls_ro = (; wb = WeightBounds(; lb = -1, ub = 1), sbgt = 1, bgt = 1)
rcap_ro = StandardDeviation(; settings = RiskMeasureSettings(; ub = 0.03))
entries_ro(res) = Set(keys(JuMP.object_dictionary(res.model)))

@testset "Each robust return term reports the worst case over its set" begin
    muh = pr_ro.mu
    Sig = cov(X_ro) / 150
    sd = sqrt.(diag(cov(X_ro)))
    Lm = [1.0 0.0; 0.5 1.0; 0.0 0.3; 0.2 0.2; 0.1 0.0; 0.0 0.4] * 0.002
    function oracle(build, w)
        m = Model(Clarabel.Optimizer)
        set_silent(m)
        mu = build(m)
        @objective(m, Min, dot(mu, w))
        optimize!(m)
        return objective_value(m)
    end
    cases = ((BoxUncertaintySet(; lb = zeros(N_ro), ub = fill(0.002, N_ro)),
              # The half-width of the box is centred on the characteristic.
              w -> oracle(m -> (@variable(m, -0.001 <= d[1:N_ro] <= 0.001); muh .+ d), w),
              (:bucs_w_1, :bucs_ret_1)),
             (EllipsoidalUncertaintySet(; sigma = Sig, k = 2.0,
                                        class = MuUncertaintySetClass()),
              w -> oracle(m -> begin
                              @variable(m, mu[1:N_ro])
                              G = cholesky(Symmetric(inv(Sig))).U
                              @constraint(m, [2.0; G * (mu .- muh)] in SecondOrderCone())
                              mu
                          end, w), (:x_eucs_w_1, :t_eucs_gw_1, :eucs_ret_1)),
             (L1UncertaintySet(; eps = 0.5, sd = sd),
              w -> oracle(m -> begin
                              @variable(m, e[1:N_ro])
                              @constraint(m, [0.5; e ./ sd] in MOI.NormOneCone(N_ro + 1))
                              muh .+ e
                          end, w), (:t_l1ucs_1, :l1ucs_ret_1)),
             (SignedL1UncertaintySet(; ep = 0.3, en = 0.6, sd = sd),
              w -> oracle(m -> begin
                              @variable(m, p[1:N_ro] >= 0)
                              @variable(m, q[1:N_ro] >= 0)
                              @constraint(m, sum(p) <= 0.3)
                              @constraint(m, sum(q) <= 0.6)
                              muh .+ sd .* (p .- q)
                          end, w),
              (:t_sl1ucs_p_1, :t_sl1ucs_m_1, :sl1ucs_ret_p_1, :sl1ucs_ret_m_1)),
             # Hölder's equality is the worst case of a norm ball, with 1/3 + 1/q = 1.
             (NormBallUncertaintySet(; kappa = 1.5, L = Lm, p = 3,
                                     class = MuUncertaintySetClass()),
              w -> dot(muh, w) - 1.5 * norm(Lm' * w, 3 / 2), (:x_nbucs_w_1,)))
    for (ucs, worst, names) in cases
        ret = ArithmeticReturn(; ucs = ucs)
        # A maximising objective pulls the epigraph down onto the norm it bounds.
        res = solve_ro(MaximumReturn(), (; ret = ret, ls_ro...); r = rcap_ro)
        @test isa(res.retcode, OptimisationSuccess)
        @test any(<(-1e-6), res.w)
        @test isapprox(value(res.model[:ret_1]), worst(res.w); atol = 1e-8)
        @test all(n -> n in entries_ro(res), names)
        # With nothing to pull it, the reported return lies below the worst case.
        res = solve_ro(MinimumRisk(), (; ret = ret, ls_ro...))
        @test value(res.model[:ret_1]) < worst(res.w) - 1e-5
        # A lower bound on the term still bounds the worst case.
        lb = worst(res.w) + 1e-4
        res = solve_ro(MinimumRisk(),
                       (;
                        ret = ArithmeticReturn(; ucs = ucs,
                                               settings = JuMPReturnsSettings(; lb = lb)),
                        ls_ro...))
        @test worst(res.w) >= lb - 1e-8
    end
    # A map with no column spans nothing, so the term raises no cone.
    nb0 = NormBallUncertaintySet(; kappa = 1.5, L = zeros(N_ro, 0), p = 3,
                                 class = MuUncertaintySetClass())
    res = solve_ro(MaximumUtility(), (; ret = ArithmeticReturn(; ucs = nb0)))
    @test value(res.model[:ret]) == dot(pr_ro.mu, res.w)
    @test !(:x_nbucs_w_1 in entries_ro(res))
end

@testset "The ratio takes its form from the robust cones and the aggregate" begin
    function form(ret; obj = MaximumRatio())
        res = solve_ro(obj, (; ret = ret))
        @test isa(res.retcode, OptimisationSuccess)
        d = entries_ro(res)
        return (:sr_ret in d, :sr_risk in d, res)
    end
    sd = sqrt.(diag(cov(X_ro)))
    box = BoxUncertaintySet(; lb = zeros(N_ro), ub = fill(0.002, N_ro))
    ell = EllipsoidalUncertaintySet(; sigma = cov(X_ro) / 150, k = 2.0,
                                    class = MuUncertaintySetClass())
    nb = NormBallUncertaintySet(; kappa = 1.5, L = Matrix(0.002I, N_ro, 2), p = 3,
                                class = MuUncertaintySetClass())
    nb0 = NormBallUncertaintySet(; kappa = 1.5, L = zeros(N_ro, 0), p = 3,
                                 class = MuUncertaintySetClass())
    for ret in (ArithmeticReturn(), ArithmeticReturn(; ucs = nb0))
        @test form(ret)[1:2] == (true, false)
    end
    for ret in (ArithmeticReturn(; ucs = box), ArithmeticReturn(; ucs = ell),
                ArithmeticReturn(; ucs = nb), LogarithmicReturn(),
                ArithmeticReturn(; ucs = L1UncertaintySet(; eps = 0.5, sd = sd)),
                ArithmeticReturn(; ucs = SignedL1UncertaintySet(; ep = 0.3, en = 0.6, sd = sd)))
        @test form(ret)[1:2] == (false, true)
    end
    # Each term alone is below the rate, and the aggregate is above it.
    rf = 1.1 * maximum(pr_ro.mu)
    @test form(ArithmeticReturn(); obj = MaximumRatio(; rf = rf))[1:2] == (false, true)
    @test form([ArithmeticReturn(), ArithmeticReturn()]; obj = MaximumRatio(; rf = rf))[1:2] ==
          (true, false)
end

@testset "A penalty or a charge forces the risk form (#1358)" begin
    # The characteristic that the form test reads carries no l1 penalty and no charge. Each of
    # these leaves no portfolio above the rate, so the return form had no solution.
    for kw in ((; ret = ArithmeticReturn(; ucs = L1UncertaintySet(; eps = 0.5))),
               (; ret = ArithmeticReturn(; ucs = SignedL1UncertaintySet(; ep = 0.5, en = 0.5))),
               (; ret = ArithmeticReturn(), fees = Fees(; l = 0.01)))
        res = solve_ro(MaximumRatio(), kw)
        @test isa(res.retcode, OptimisationSuccess)
        @test :sr_risk in entries_ro(res)
        @test !(:sr_ret in entries_ro(res))
        # No portfolio beats the rate, so `k` sits on the floor.
        @test isapprox(value(res.model[:k]), JuMP.lower_bound(res.model[:k]); rtol = 1e-3)
    end
    # A long fee on a long-only book is `l` per unit of budget, so it is a rate of `rf + l`.
    # The fee takes the risk form, the rate takes the return form, and both reach one maximiser.
    l = 0.3 * maximum(pr_ro.mu)
    res_fee = solve_ro(MaximumRatio(), (; fees = Fees(; l = l)))
    res_rf = solve_ro(MaximumRatio(; rf = l), (;))
    @test :sr_risk in entries_ro(res_fee)
    @test :sr_ret in entries_ro(res_rf)
    @test isapprox(res_fee.w, res_rf.w; atol = 1e-5)
    # A term with `fee = false` deducts nothing, so it keeps the return form.
    res = solve_ro(MaximumRatio(),
                   (;
                    ret = ArithmeticReturn(; settings = JuMPReturnsSettings(; fee = false)),
                    fees = Fees(; l = l)))
    @test :sr_ret in entries_ro(res)
    @test isapprox(res.w, solve_ro(MaximumRatio(), (;)).w; atol = 1e-6)
end

@testset "The ratio sizes ohf and the scale floor from the characteristic" begin
    ohf = min(1e3, max(1e-3, mean(abs, pr_ro.mu)))
    box = BoxUncertaintySet(; lb = zeros(N_ro), ub = fill(0.002, N_ro))
    res = solve_ro(MaximumRatio(), (; ret = ArithmeticReturn()))
    @test value(res.model[:ohf]) == ohf
    # The return form takes no floor unless the caller names one.
    @test JuMP.lower_bound(res.model[:k]) == 0
    w1 = res.w
    res = solve_ro(MaximumRatio(), (; ret = ArithmeticReturn(; ucs = box)))
    @test JuMP.lower_bound(res.model[:k]) ≈ 1e-4 * ohf / max(ohf, maximum(pr_ro.mu))
    res = solve_ro(MaximumRatio(; kmin = 0.5), (; ret = ArithmeticReturn()))
    @test JuMP.lower_bound(res.model[:k]) == 0.5
    # No portfolio beats the rate, so `k` lands on the floor.
    res = solve_ro(MaximumRatio(; rf = 1.0), (; ret = ArithmeticReturn()))
    @test isapprox(value(res.model[:k]), 1e-4; rtol = 1e-4)
    # `ohf` scales `k` and leaves the weights.
    res = solve_ro(MaximumRatio(; ohf = 2.0), (; ret = ArithmeticReturn()))
    @test value(res.model[:ohf]) == 2.0
    @test isapprox(res.w, w1; atol = 1e-5)
end

@testset "The logarithmic term is the mean log return" begin
    f(w) = mean(log1p.(X_ro * w))
    ow = pweights(collect(range(0.5, 1.5; length = size(X_ro, 1))))
    fw(w) = sum(ow .* log1p.(X_ro * w)) / sum(ow)
    res = solve_ro(MaximumReturn(), (; ret = LogarithmicReturn(), ls_ro...); r = rcap_ro)
    @test isapprox(value(res.model[:ret]), f(res.w); atol = 1e-9)
    @test isapprox(expected_return(LogarithmicReturn(), res.w, pr_ro), f(res.w))
    @test all(n -> n in entries_ro(res), (:t_elog_ret_1, :kret_1, :elog_ret_ret_1))
    res = solve_ro(MaximumReturn(), (; ret = LogarithmicReturn(; w = ow), ls_ro...);
                   r = rcap_ro)
    @test isapprox(value(res.model[:ret]), fw(res.w); atol = 1e-9)
    @test isapprox(expected_return(LogarithmicReturn(; w = ow), res.w, pr_ro), fw(res.w))
    # Under the ratio the cone bounds each entry by `k ln(1 + x' w / k)` on the scaled weights.
    res = solve_ro(MaximumRatio(), (; ret = LogarithmicReturn()))
    k = value(res.model[:k])
    @test !isapprox(k, 1; atol = 1e-2)
    @test isapprox(value(res.model[:ret]), k * f(res.w); atol = 1e-9)
end

@testset "A term nets its own flagged charges" begin
    function deduction(kw, settings; slv = slv_ro, r = StandardDeviation())
        res = solve_ro(MinimumRisk(),
                       (; ret = ArithmeticReturn(; settings = settings), kw...); r = r,
                       slv = slv)
        @test isa(res.retcode, OptimisationSuccess)
        return dot(pr_ro.mu, res.w) - value(res.model[:ret_1]), res.model
    end
    # A fixed fee needs a binary, so this case solves a linear programme with HiGHS.
    slv_h = Solver(; name = :highs, solver = HiGHS.Optimizer,
                   settings = "log_to_console" => false,
                   check_sol = (; allow_local = true, allow_almost = true))
    fees = (; fees = Fees(; l = 0.001, s = 0.002, fl = 0.0005), ls_ro...)
    d, m = deduction(fees, JuMPReturnsSettings(); slv = slv_h, r = ConditionalValueatRisk())
    charge = value(m[:fees]) + value(m[:one_time_fees]) / m[:T]
    @test value(m[:one_time_fees]) > 0
    @test isapprox(d, charge; atol = 1e-12)
    d, _ = deduction(fees, JuMPReturnsSettings(; fee = false); slv = slv_h,
                     r = ConditionalValueatRisk())
    @test isapprox(d, 0; atol = 1e-12)
    w0 = fill(1 / N_ro, N_ro)
    bmi = BudgetMarketImpact(; bgt = 1.0, w = w0, vp = 0.01, vn = 0.01, up = 0.5, un = 0.5)
    d, m = deduction((; bgt = bmi), JuMPReturnsSettings())
    @test value(m[:cost_bgt_expr]) > 0
    @test isapprox(d, value(m[:cost_bgt_expr]); atol = 1e-10)
    d, _ = deduction((; bgt = bmi), JuMPReturnsSettings(; mic = false))
    @test isapprox(d, 0; atol = 1e-12)
    # A plain budget cost constrains the budget and never reaches the return.
    bc = BudgetCosts(; bgt = 1.0, w = w0, vp = 0.01, vn = 0.01, up = 0.5, un = 0.5)
    d, _ = deduction((; bgt = bc), JuMPReturnsSettings())
    @test isapprox(d, 0; atol = 1e-12)
end

@testset "The bound, the flag and the sum of the return terms" begin
    r0 = dot(pr_ro.mu, solve_ro(MinimumRisk(), (; ret = ArithmeticReturn())).w)
    lb = r0 + 0.001
    res = solve_ro(MinimumRisk(),
                   (; ret = ArithmeticReturn(; settings = JuMPReturnsSettings(; lb = lb))))
    @test :ret_lb_1 in entries_ro(res)
    @test isapprox(dot(pr_ro.mu, res.w), lb; atol = 1e-10)
    # The row reads `ret - lb k`, so it binds on `w / k` under the ratio.
    res = solve_ro(MaximumRatio(),
                   (;
                    ret = ArithmeticReturn(;
                                           settings = JuMPReturnsSettings(;
                                                                          lb = lb + 0.001))))
    @test !isapprox(value(res.model[:k]), 1; atol = 1e-2)
    @test isapprox(dot(pr_ro.mu, res.w), lb + 0.001; atol = 1e-10)
    # A term out of the sum still binds its own bound.
    rets = [ArithmeticReturn(),
            ArithmeticReturn(; mu = 2 * pr_ro.mu,
                             settings = JuMPReturnsSettings(; rte = false, lb = 2 * lb))]
    res = solve_ro(MinimumRisk(), (; ret = rets))
    @test length(res.model[:ret_vec]) == 1
    @test value(res.model[:ret]) == dot(pr_ro.mu, res.w)
    @test isapprox(2 * dot(pr_ro.mu, res.w), 2 * lb; atol = 1e-10)
    # Several terms sum at their own scale, and one term drops its scale.
    rets = [ArithmeticReturn(; settings = JuMPReturnsSettings(; scale = 0.25)),
            ArithmeticReturn(; mu = reverse(pr_ro.mu),
                             settings = JuMPReturnsSettings(; scale = 0.75))]
    res = solve_ro(MaximumUtility(), (; ret = rets))
    @test isapprox(value(res.model[:ret]),
                   0.25 * dot(pr_ro.mu, res.w) + 0.75 * dot(reverse(pr_ro.mu), res.w);
                   atol = 1e-14)
    w1 = solve_ro(MaximumUtility(),
                  (;
                   ret = ArithmeticReturn(; settings = JuMPReturnsSettings(; scale = 3.0)))).w
    @test w1 == solve_ro(MaximumUtility(), (; ret = ArithmeticReturn())).w
end

@testset "Each objective sets its sense and its expression" begin
    rcap = StandardDeviation(; settings = RiskMeasureSettings(; ub = 0.006))
    box = BoxUncertaintySet(; lb = zeros(N_ro), ub = fill(0.002, N_ro))
    cases = ((MinimumRisk(), (;), StandardDeviation(), MOI.MIN_SENSE,
              (ret, risk, k, op) -> risk + op),
             (MaximumUtility(; l = 3.0), (;), StandardDeviation(), MOI.MAX_SENSE,
              (ret, risk, k, op) -> ret - 3.0 * risk - op),
             (MaximumReturn(), (;), rcap, MOI.MAX_SENSE, (ret, risk, k, op) -> ret - op),
             (MaximumRatio(), (;), StandardDeviation(), MOI.MIN_SENSE,
              (ret, risk, k, op) -> risk + op),
             (MaximumRatio(; rf = 0.0005), (; ret = ArithmeticReturn(; ucs = box)),
              StandardDeviation(), MOI.MAX_SENSE,
              (ret, risk, k, op) -> ret - 0.0005 * k - op))
    for (obj, kw, r, sense, expr) in cases
        res = solve_ro(obj, (; so = 2.0, l2 = L2Regularisation(; val = 0.01), kw...); r = r)
        m = res.model
        @test objective_sense(m) == sense
        @test value(m[:op]) > 0
        @test isapprox(objective_value(m),
                       2.0 *
                       expr(value(m[:ret]), value(m[:risk]), value(m[:k]), value(m[:op]));
                       atol = 1e-12)
    end
    # A variance of degree two gives the Sharpe ratio under the ratio objective.
    ws = solve_ro(MaximumRatio(), (;)).w
    @test isapprox(solve_ro(MaximumRatio(), (;); r = Variance()).w, ws; atol = 1e-4)
    @test isapprox(solve_ro(MaximumRatio(), (;); r = Variance(; alg = QuadRiskExpr())).w,
                   ws; atol = 1e-4)
end

@testset "A stated scalar characteristic must be finite" begin
    @test ArithmeticReturn(; mu = 0.1).mu == 0.1
    @test_throws IsNonFiniteError ArithmeticReturn(; mu = Inf)
end
