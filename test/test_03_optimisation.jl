@testset "Optimisation failure" begin
    using PortfolioOptimisers, JuMP, Test, Clarabel
    @testset "optimise_JuMP_model!" begin
        model = JuMP.Model()
        JuMP.set_optimizer(model, Clarabel.Optimizer)
        res = PortfolioOptimisers.optimise_JuMP_model!(JuMP.Model(),
                                                       [Solver(; name = :Clarabel,
                                                               solver = Clarabel.Optimizer,
                                                               settings = Dict("verbose" =>
                                                                                   false)),
                                                        Solver(; name = :Nothing,
                                                               solver = nothing,
                                                               settings = Dict("verbose" =>
                                                                                   false))])
        @test !res.success
        @test haskey(res.trials[:Clarabel], :optimize!)
        @test haskey(res.trials[:Nothing], :set_optimizer)
        res = PortfolioOptimisers.optimise_JuMP_model!(JuMP.Model(), Solver[])
        @test !res.success
    end
end

@testset "Solver chain records" begin
    using PortfolioOptimisers, JuMP, Test, Clarabel
    @testset "optimise_JuMP_model!" begin
        function qpmodel(lb)
            model = JuMP.Model()
            JuMP.@variable(model, w[1:3])
            JuMP.@constraint(model, sum(w) == 1)
            JuMP.@constraint(model, w .>= lb)
            JuMP.@objective(model, Min, sum(w .^ 2))
            return model
        end
        # An optimizer attribute reaches the solver, and a single `Solver` is a vector of one.
        model = qpmodel(0.0)
        res = PortfolioOptimisers.optimise_JuMP_model!(model,
                                                       Solver(; solver = Clarabel.Optimizer,
                                                              settings = MOI.Silent() =>
                                                                  true))
        @test res.success
        @test isempty(res.trials)
        @test JuMP.get_attribute(model, MOI.Silent())
        @test isapprox(JuMP.value.(model[:w]), fill(inv(3), 3); rtol = 1e-6)
        # An attribute that the solver refuses raises out of the loop, and no later
        # solver is tried.
        @test_throws Exception PortfolioOptimisers.optimise_JuMP_model!(qpmodel(0.0),
                                                                        [Solver(; name = :a,
                                                                                solver = Clarabel.Optimizer,
                                                                                settings = "no_such_attribute" =>
                                                                                    1),
                                                                         Solver(; name = :b,
                                                                                solver = Clarabel.Optimizer)])
        # The entry of a failed check also holds the settings of the solver.
        res = PortfolioOptimisers.optimise_JuMP_model!(qpmodel(0.9),
                                                       Solver(; name = :c,
                                                              solver = Clarabel.Optimizer,
                                                              settings = Dict("verbose" =>
                                                                                  false)))
        @test !res.success
        @test Set(keys(res.trials[:c])) ==
              Set([:assert_is_solved_and_feasible, :termination_status, :settings])
        @test res.trials[:c][:settings] == Dict("verbose" => false)
        @test res.trials[:c][:termination_status] == JuMP.MOI.INFEASIBLE
        # Two unnamed solvers share one entry, and two named solvers keep two.
        nosolver = () -> error("no solver")
        res = PortfolioOptimisers.optimise_JuMP_model!(qpmodel(0.0),
                                                       [Solver(; solver = nosolver),
                                                        Solver(; solver = nosolver)])
        @test length(res.trials) == 1
        res = PortfolioOptimisers.optimise_JuMP_model!(qpmodel(0.0),
                                                       [Solver(; name = :a,
                                                               solver = nosolver),
                                                        Solver(; name = :b,
                                                               solver = nosolver)])
        @test length(res.trials) == 2
    end
end

@testset "Solver fallback chain robustness" begin
    using PortfolioOptimisers, JuMP, Test, Clarabel
    function testmodel(lb)
        model = JuMP.Model()
        JuMP.@variable(model, w[1:3])
        JuMP.@variable(model, k)
        JuMP.@constraint(model, sum(w) == 1)
        JuMP.@constraint(model, w .>= lb)
        JuMP.@constraint(model, k == 1)
        JuMP.@objective(model, Min, sum(w .^ 2))
        return model
    end
    @testset "clean termination without primal does not abort the chain" begin
        # `w .>= 0.9` with `sum(w) == 1` is infeasible: solvers terminate without a
        # primal, which used to crash the unguarded `JuMP.value` reads.
        slv = [Solver(; name = :c1, solver = Clarabel.Optimizer,
                      settings = Dict("verbose" => false)),
               Solver(; name = :c2, solver = Clarabel.Optimizer,
                      check_sol = (; allow_local = true, allow_almost = true),
                      settings = Dict("verbose" => false))]
        mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv))
        logger = SimpleLogger()
        with_logger(logger) do
            @test_logs (:warn,
                        r"could not be solved satisfactorily(.|\n)*c1: termination_status → INFEASIBLE") PortfolioOptimisers.optimise_JuMP_model!(testmodel(0.9),
                                                                                                                                                  mr)
        end
        retcode, sol = PortfolioOptimisers.optimise_JuMP_model!(testmodel(0.9), mr)
        @test isa(retcode, OptimisationFailure)
        @test all(isnan, sol.w)
        # Both solvers must have been tried and their diagnostics merged: the
        # assertion error must not be overwritten by the solution summary.
        for name in (:c1, :c2)
            trial = retcode.res[name]
            @test haskey(trial, :assert_is_solved_and_feasible)
            @test haskey(trial, :err)
            @test haskey(trial, :settings)
            @test trial[:termination_status] == JuMP.MOI.INFEASIBLE
        end
    end
    @testset "chain advances past a failing solver to a successful one" begin
        slv = [Solver(; name = :limited, solver = Clarabel.Optimizer,
                      settings = Dict("verbose" => false, "max_iter" => 1)),
               Solver(; name = :ok, solver = Clarabel.Optimizer,
                      check_sol = (; allow_local = true, allow_almost = true),
                      settings = Dict("verbose" => false))]
        mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv))
        retcode, sol = PortfolioOptimisers.optimise_JuMP_model!(testmodel(0.0), mr)
        @test isa(retcode, OptimisationSuccess)
        @test haskey(retcode.res, :limited)
        @test haskey(retcode.res[:limited], :assert_is_solved_and_feasible)
        @test isapprox(sol.w, fill(inv(3), 3); rtol = 1e-6)
    end
end

@testset "Risk upper bound with unsupported optimiser" begin
    using PortfolioOptimisers, JuMP, Test, Clarabel
    slv = Solver(; name = :c1, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false))
    frc = FactorRiskContribution(; opt = JuMPOptimiser(; slv = slv))
    model = JuMP.Model()
    # A user-set `settings.ub` must warn instead of being silently dropped.
    logger = SimpleLogger()
    with_logger(logger) do
        @test_logs (:warn, r"Risk upper bound") PortfolioOptimisers.set_risk_upper_bound!(model,
                                                                                          frc,
                                                                                          1.0,
                                                                                          1.0,
                                                                                          :risk)
    end
    # No bound requested: no-op.
    @test isnothing(PortfolioOptimisers.set_risk_upper_bound!(model, frc, 1.0, nothing,
                                                              :risk))
end

@testset "L1 uncertainty set: robust best-characteristic portfolios" begin
    using PortfolioOptimisers, Test, HiGHS, StableRNGs, Random, Statistics, LinearAlgebra
    # Zhou & Palomar, "Understanding the Quintile Portfolio", IEEE TSP 68 (2020).
    # The paper's closed forms are exact, so these assert equality with them rather than
    # against stored regression output. Every model below is a MeanRisk recipe: the library
    # ships no quintile optimiser (ADR 0032).
    rng = StableRNG(987654321)
    N = 20
    mu_t = round.(sort(randn(rng, N) .* 0.02 .+ 0.05; rev = true); digits = 4)
    sd_t = round.(0.1 .+ 0.15 .* rand(rng, N); digits = 4)
    X = randn(rng, 2000, N)
    X .= (X .- mean(X; dims = 1)) ./ std(X; dims = 1) .* sd_t' .+ mu_t'
    rd = ReturnsResult(; X = X, nx = string.("A", 1:N))
    pr = prior(EmpiricalPrior(), rd.X)
    sd_hat = sqrt.(diag(pr.sigma))
    slv = Solver(; name = :highs, solver = HiGHS.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = Dict("log_to_console" => false))
    # Activation ladders of the closed forms.
    g(k) = sum(mu_t[i] - mu_t[k] for i in 1:k)
    gs(k) = sum((mu_t[i] - mu_t[k]) / sd_hat[i] for i in 1:k)
    f(m) = sum(mu_t[i] - mu_t[m] for i in 1:m) +
           sum(mu_t[N - m + 1] - mu_t[N - j + 1] for j in 1:m)
    quintile(ucs; kwargs...) = optimise(MeanRisk(; r = NoRisk(), obj = MaximumReturn(),
                                                 opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                                     ret = ArithmeticReturn(;
                                                                                            ucs = ucs),
                                                                     kwargs...)))
    lo(ucs) = quintile(ucs; bgt = 1.0, wb = WeightBounds(; lb = 0.0, ub = 1.0)).w
    @testset "Model 1 (eq. 9): 1/N and long-only quintile" begin
        # Lemma 2 / Corollary 4: exactly q active, each weighted 1/q, on the top q by mu.
        for q in (1, 4, 7)
            w = lo(L1UncertaintySet(; eps = (g(q) + g(q + 1)) / 2))
            @test count(>(1e-6), w) == q
            @test all(x -> isapprox(x, 1 / q; atol = 1e-6), w[w .> 1e-6])
            @test sort(findall(>(1e-6), w)) == collect(1:q)
            @test isapprox(sum(w), 1.0; atol = 1e-8)
        end
        # Corollary 3: a radius above the top of the ladder gives the 1/N portfolio.
        @test all(x -> isapprox(x, 1 / N; atol = 1e-6),
                  lo(L1UncertaintySet(; eps = g(N) * 1.5)))
    end
    @testset "Model 3 (eq. 21): long-only quintile-IV" begin
        # Lemma 9: active assets weighted proportionally to 1/sigma_i, not equally.
        w = lo(L1UncertaintySet(; eps = (gs(4) + gs(5)) / 2, sd = sd_hat))
        act = findall(>(1e-6), w)
        @test sort(act) == collect(1:4)
        @test isapprox(w[act], (1 ./ sd_hat[act]) ./ sum(1 ./ sd_hat[act]); atol = 1e-6)
        @test !all(x -> isapprox(x, 0.25; atol = 1e-3), w[act])
        # Corollary 10: a large radius gives the full inverse-volatility portfolio.
        @test isapprox(lo(L1UncertaintySet(; eps = gs(N) * 1.5, sd = sd_hat)),
                       (1 ./ sd_hat) ./ sum(1 ./ sd_hat); atol = 1e-6)
    end
    @testset "Model 2 (eq. 14): dollar-neutral long-short quintile" begin
        # Corollary 7: m long and m short, antisymmetric at +/- 1/(2m). Reached without the
        # paper's Lemma 5 pairing -- the LP composes with bgt/sbgt directly.
        for m in (2, 4)
            w = quintile(L1UncertaintySet(; eps = (f(m) + f(m + 1)) / 2); bgt = 0.0,
                         sbgt = 0.5, wb = WeightBounds(; lb = -1.0, ub = 1.0)).w
            @test sort(findall(>(1e-6), w)) == collect(1:m)
            @test sort(findall(<(-1e-6), w)) == collect((N - m + 1):N)
            @test all(x -> isapprox(abs(x), 1 / (2m); atol = 1e-6), w[abs.(w) .> 1e-6])
            @test isapprox(sum(w), 0.0; atol = 1e-8)
            @test isapprox(sum(abs, w), 1.0; atol = 1e-6)
        end
    end
    @testset "Model 4 (eq. 25): quintile-IV long-short, solved coupled" begin
        # Modelling A2's worst case directly keeps the problem coupled, so the legs cannot
        # overlap -- the complementary-support caveat of the paper's Remark 12 cannot arise.
        r = quintile(SignedL1UncertaintySet(; ep = 0.05, en = 0.05, sd = sd_hat); bgt = 0.0,
                     sbgt = 0.5, wb = WeightBounds(; lb = -1.0, ub = 1.0))
        w = r.w
        longs, shorts = findall(>(1e-6), w), findall(<(-1e-6), w)
        @test isa(r.jr.retcode, PortfolioOptimisers.OptimisationSuccess)
        @test isempty(intersect(longs, shorts))
        @test !isempty(longs) && !isempty(shorts)
        @test isapprox(sum(w), 0.0; atol = 1e-8)
        # Each leg is inverse-volatility weighted within itself (Corollary 13).
        @test isapprox(w[longs] ./ sum(w[longs]),
                       (1 ./ sd_hat[longs]) ./ sum(1 ./ sd_hat[longs]); atol = 1e-5)
        # The signed set is NOT the joint set with ep == em: it may hold a different number
        # of longs and shorts, which the paper's decoupled form cannot express.
        wj = quintile(L1UncertaintySet(; eps = 0.05, sd = sd_hat); bgt = 0.0, sbgt = 0.5,
                      wb = WeightBounds(; lb = -1.0, ub = 1.0)).w
        @test !isapprox(w, wj)
    end
    @testset "Model 5 (eq. 30): market exposure constraint" begin
        beta = round.(0.6 .+ 0.8 .* rand(rng, N); digits = 4)
        lc = LinearConstraint(; eq = PartialLinearConstraint(reshape(beta, 1, N), [1.0]))
        w = quintile(L1UncertaintySet(; eps = 0.002); bgt = nothing, lcse = lc,
                     wb = WeightBounds(; lb = 0.0, ub = Inf)).w
        @test isapprox(dot(beta, w), 1.0; atol = 1e-6)
        @test all(>=(-1e-9), w)
        # The paper ranks on the characteristic-beta ratio here, not on mu alone.
        act = findall(>(1e-6), w)
        @test sort(act) == sort(sortperm(mu_t ./ beta; rev = true)[1:length(act)])
    end
    @testset "Model 6 (eq. 33): market neutral -- gross pinned, net free" begin
        beta = round.(0.6 .+ 0.8 .* rand(rng, N); digits = 4)
        lc = LinearConstraint(; eq = PartialLinearConstraint(reshape(beta, 1, N), [0.0]))
        r = quintile(L1UncertaintySet(; eps = 0.002); bgt = nothing, gbgt = 1.0,
                     xbgt = true, lcse = lc, wb = WeightBounds(; lb = -1.0, ub = 1.0))
        w = r.w
        @test isa(r.jr.retcode, PortfolioOptimisers.OptimisationSuccess)
        @test isapprox(sum(abs, w), 1.0; atol = 1e-5)     # gross pinned
        @test isapprox(dot(beta, w), 0.0; atol = 1e-6)    # market neutral
        # The net is genuinely free -- this is the combination bgt/sbgt cannot express.
        @test !isapprox(sum(w), 0.0; atol = 1e-3)
    end
    @testset "Model 7 (eq. 35): zero active sector exposure" begin
        sector = repeat(1:4; inner = 5)
        A = zeros(4, N)
        for k in 1:4, j in findall(==(k), sector)
            A[k, j] = 1.0
        end
        w = quintile(L1UncertaintySet(; eps = 0.004); bgt = 0.0, sbgt = 0.5,
                     lcse = LinearConstraint(; eq = PartialLinearConstraint(A, zeros(4))),
                     wb = WeightBounds(; lb = -1.0, ub = 1.0)).w
        for k in 1:4
            @test isapprox(sum(w[sector .== k]), 0.0; atol = 1e-8)
        end
        @test isapprox(sum(abs, w), 1.0; atol = 1e-6)
        # Lemma 14: within a sector the i-th best pairs against the i-th worst.
        @test all(k -> isapprox(sort(w[sector .== k]), -reverse(sort(-w[sector .== k]));
                                atol = 1e-6), 1:4)
    end
    @testset "xbgt: budgets bound by default, pin when set" begin
        # Past f(floor(N/2)) the relaxation prefers cash while the paper holds a 50/50
        # corner at a negative worst-case objective. Cash is rejected as a structured
        # failure rather than returned silently.
        eps_big = f(N ÷ 2) * 1.5
        args = (; bgt = 0.0, sbgt = 0.5, wb = WeightBounds(; lb = -1.0, ub = 1.0))
        relaxed = quintile(L1UncertaintySet(; eps = eps_big); args...)
        @test isa(relaxed.jr.retcode, PortfolioOptimisers.OptimisationFailure)
        exact = quintile(L1UncertaintySet(; eps = eps_big); xbgt = true, args...)
        @test isa(exact.jr.retcode, PortfolioOptimisers.OptimisationSuccess)
        @test isapprox(sum(abs, exact.w), 1.0; atol = 1e-5)
        @test isapprox(sum(exact.w), 0.0; atol = 1e-6)
        # Inside the paper's range the two agree, so xbgt buys nothing there.
        eps_ok = (f(4) + f(5)) / 2
        @test isapprox(quintile(L1UncertaintySet(; eps = eps_ok); args...).w,
                       quintile(L1UncertaintySet(; eps = eps_ok); xbgt = true, args...).w;
                       atol = 1e-6)
    end
    @testset "xbgt reuses existing MIP binaries" begin
        # A sign bit and a held bit are different bits, so xbgt cannot share `card`'s `ib`.
        # It reuses the long/short pair when one exists and takes a lean single-bit branch
        # when nothing else needs binaries. Both must pin, and neither may leave a second,
        # unlinked indicator family behind. Deliberately at the quintile radius, not the
        # divergence one: this is about the variable structure, and eps_big turns these
        # into the hard MILP the docs warn about (minutes rather than seconds).
        nbin(r) = count(JuMP.is_binary, JuMP.all_variables(r.jr.model))
        args = (; bgt = 0.0, sbgt = 0.5, wb = WeightBounds(; lb = -1.0, ub = 1.0))
        eps_q = (f(4) + f(5)) / 2

        # Alone: the lean sign-bit branch, N binaries, no ilb/isb/ib in sight.
        alone = quintile(L1UncertaintySet(; eps = eps_q); xbgt = true, args...)
        @test nbin(alone) == N
        @test haskey(alone.jr.model, :xbgt_ib)
        @test !haskey(alone.jr.model, :ilb)
        @test !haskey(alone.jr.model, :ib)
        @test isapprox(sum(abs, alone.w), 1.0; atol = 1e-5)

        # With a threshold: rides ilb/isb, so xbgt costs nothing on top and adds no
        # indicator of its own.
        st = Threshold(; val = 0.05)
        without = quintile(L1UncertaintySet(; eps = eps_q); st = st, args...)
        with = quintile(L1UncertaintySet(; eps = eps_q); st = st, xbgt = true, args...)
        @test nbin(with) == nbin(without)
        @test !haskey(with.jr.model, :xbgt_ib)
        @test haskey(with.jr.model, :ilb) && haskey(with.jr.model, :isb)
        @test isapprox(sum(abs, with.w), 1.0; atol = 1e-5)

        # With cardinality: 2N is irreducible (held bit + sign bit), but `card` must still
        # count *holdings*, not longs -- the bug that sharing `ib` would have introduced.
        cw = quintile(L1UncertaintySet(; eps = eps_q); card = 6, xbgt = true, args...)
        @test nbin(cw) == 2 * N
        @test count(abs.(cw.w) .> 1e-10) <= 6
        @test count(cw.w .< -1e-10) > 0     # shorts survive and are counted
        @test isapprox(sum(abs, cw.w), 1.0; atol = 1e-5)
    end
    @testset "Calibrated radius drives the optimisation end to end" begin
        # The estimator form: no radius in sight, just a target active count.
        ue = CharacteristicUncertaintySet(; pe = EmpiricalPrior(),
                                          alg = L1UncertaintySetAlgorithm(;
                                                                          method = ActiveAssetsUncertaintyAlgorithm(;
                                                                                                                    active = 0.2)))
        r = optimise(MeanRisk(; r = NoRisk(), obj = MaximumReturn(),
                              opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv,
                                                  bgt = 1.0,
                                                  wb = WeightBounds(; lb = 0.0, ub = 1.0),
                                                  ret = ArithmeticReturn(; ucs = ue))), rd)
        @test count(>(1e-6), r.w) == 4
        @test all(x -> isapprox(x, 0.25; atol = 1e-6), r.w[r.w .> 1e-6])
    end
    @testset "Ranking on volatility (Table III) needs no new machinery" begin
        # The characteristic need not be a return: a prior carrying standard deviations in
        # the mean slot ranks on the Low Volatility factor.
        prv = prior(EmpiricalPrior(; me = StandardDeviationExpectedReturns()), rd.X)
        w = optimise(MeanRisk(; r = NoRisk(), obj = MaximumReturn(),
                              opt = JuMPOptimiser(; pe = prv, slv = slv, bgt = 1.0,
                                                  wb = WeightBounds(; lb = 0.0, ub = 1.0),
                                                  ret = ArithmeticReturn(;
                                                                         ucs = L1UncertaintySet(;
                                                                                                eps = 0.05))))).w
        # Ranks by volatility, so the active set is the highest-volatility assets.
        act = findall(>(1e-6), w)
        @test sort(act) == sort(sortperm(prv.mu; rev = true)[1:length(act)])
    end
    @testset "The set carries its centre, and that centre wins (ADR 0050)" begin
        # A set is a neighbourhood of the quantity it was calibrated on. When it names
        # one, the optimiser bounds *that* vector and not the outer prior's.
        q = 4
        eps_q = (g(q) + g(q + 1)) / 2
        # `pr.mu` is sorted descending, so reversing it is the same multiset under a
        # reversed ranking: the ladder, and therefore the radius, is unchanged.
        rev_mu = reverse(pr.mu)
        w_carried = lo(L1UncertaintySet(; eps = eps_q, mu = rev_mu))
        @test count(>(1e-6), w_carried) == q
        @test all(x -> isapprox(x, 1 / q; atol = 1e-6), w_carried[w_carried .> 1e-6])
        # The carried vector ranks the *last* q assets first, so the fallback's answer
        # (the first q) is not the one taken.
        @test sort(findall(>(1e-6), w_carried)) == collect((N - q + 1):N)
        @test sort(findall(>(1e-6), lo(L1UncertaintySet(; eps = eps_q)))) == collect(1:q)
        # It also wins over an explicit `ArithmeticReturn.mu`, which is the middle rung of
        # the precedence: carried, then estimator field, then prior.
        w_over_field = optimise(MeanRisk(; r = NoRisk(), obj = MaximumReturn(),
                                         opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                             bgt = 1.0,
                                                             wb = WeightBounds(; lb = 0.0,
                                                                               ub = 1.0),
                                                             ret = ArithmeticReturn(;
                                                                                    mu = pr.mu,
                                                                                    ucs = L1UncertaintySet(;
                                                                                                           eps = eps_q,
                                                                                                           mu = rev_mu))))).w
        @test isapprox(w_over_field, w_carried; atol = 1e-6)
        # The signed set resolves through its own field the same way.
        w_signed = quintile(SignedL1UncertaintySet(; ep = 1e-8, en = eps_q, mu = rev_mu);
                            bgt = 1.0, wb = WeightBounds(; lb = 0.0, ub = 1.0)).w
        @test sort(findall(>(1e-6), w_signed)) == collect((N - q + 1):N)
    end
end
@testset "Weight finalisation" begin
    using PortfolioOptimisers, JuMP, Test, Clarabel, LinearAlgebra, Dates, StableRNGs
    slv = Solver(; name = :Clarabel, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false))
    w0 = [0.5, 0.3, 0.15, 0.05]
    wb = WeightBounds(; lb = fill(0.1, 4), ub = fill(0.4, 4))
    @testset "The objective is the norm the docstring states" begin
        # The cone bounds the norm itself, so under a SquaredL2Norm the objective value is
        # the L2 norm and not its square; the minimiser is the same.
        rel = RelativeWeightDeviation()
        abs_dev = AbsoluteWeightDeviation()
        rel_res(w) = w ./ w0 .- one(eltype(w0))
        for (alg, obj) in ((ErrorWeightFinaliser(), w -> norm(rel_res(w), 1)),
                           (ErrorWeightFinaliser(; alg = L2Norm()), w -> norm(rel_res(w), 2)),
                           (ErrorWeightFinaliser(; alg = SquaredL2Norm()), w -> norm(rel_res(w), 2)),
                           (ErrorWeightFinaliser(; alg = LInfNorm()), w -> norm(rel_res(w), Inf)),
                           (ErrorWeightFinaliser(; alg = LpNorm(; p = 3)), w -> norm(rel_res(w), 3)),
                           (ErrorWeightFinaliser(; dev = abs_dev), w -> norm(w - w0, 1)),
                           (ErrorWeightFinaliser(; alg = L2Norm(), dev = abs_dev), w -> norm(w - w0, 2)),
                           (ErrorWeightFinaliser(; alg = LInfNorm(), dev = abs_dev), w -> norm(w - w0, Inf)),
                           (ErrorWeightFinaliser(; alg = LpNorm(; p = 3), dev = abs_dev),
                            w -> norm(w - w0, 3)))
            model = JuMP.Model()
            JuMP.@expression(model, sc, 1.0)
            JuMP.@expression(model, so, 1.0)
            JuMP.@variable(model, w[1:4])
            JuMP.@constraint(model, sc * (sum(w) - sum(w0)) == 0)
            JuMP.@constraint(model, sc * (w .- wb.lb) >= 0)
            JuMP.@constraint(model, sc * (w .- wb.ub) <= 0)
            PortfolioOptimisers.set_clustering_weight_finaliser_alg!(model, alg, copy(w0))
            JuMP.set_optimizer(model, Clarabel.Optimizer)
            JuMP.set_attribute(model, "verbose", false)
            JuMP.optimize!(model)
            @test isapprox(JuMP.objective_value(model), obj(JuMP.value.(model[:w]));
                           rtol = 1e-6)
        end
    end
    @testset "A repaired vector keeps its budget and lands in the bounds" begin
        for wf in (IterativeWeightFinaliser(), EuclideanWeightFinaliser(),
                   EntropicWeightFinaliser(), JuMPWeightFinaliser(; slv = slv))
            w = PortfolioOptimisers.opt_weight_bounds(wf, wb, copy(w0))
            @test isapprox(sum(w), sum(w0))
            @test all(wb.lb .- 1e-8 .<= w .<= wb.ub .+ 1e-8)
        end
        # Weights already inside the bounds come back untouched, without a solve.
        wok = [0.25, 0.25, 0.25, 0.25]
        @test PortfolioOptimisers.opt_weight_bounds(JuMPWeightFinaliser(; slv = slv), wb,
                                                    copy(wok)) == wok
    end
    @testset "An unsatisfiable bound set reports failure" begin
        # `Σ lb = 1.2` cannot hold a budget of one, so no finaliser can meet the bounds.
        wbi = WeightBounds(; lb = fill(0.3, 4), ub = fill(0.9, 4))
        for wf in (IterativeWeightFinaliser(), EuclideanWeightFinaliser(),
                   EntropicWeightFinaliser())
            retcode, w = PortfolioOptimisers.finalise_weight_bounds(wf, wbi, copy(w0))
            @test isa(retcode, OptimisationFailure)
            @test w ≈ wbi.lb
        end
        retcode, w = PortfolioOptimisers.finalise_weight_bounds(IterativeWeightFinaliser(),
                                                                wb, copy(w0))
        @test isa(retcode, OptimisationSuccess)
        @test !PortfolioOptimisers.weights_meet_bounds(wb, [NaN, 0.5, 0.25, 0.25], 1.0)
    end
    @testset "The iterative loop falls back to the projection when it cannot converge" begin
        opt = PortfolioOptimisers.opt_weight_bounds
        proj = PortfolioOptimisers.euclidean_weight_projection
        # A stall: after the clip no weight lies strictly inside its bounds, so the clipped
        # mass has nowhere to go, and the loop sat at `[0.5, 0.5, 0.0]` for any `iter`.
        # A long-short divergence: the loop spread mass over a sum of mixed signs. The third
        # case is the worst of a random census, where the loop reached `[277, -515, 933,
        # -694]`.
        cases = ((WeightBounds(; lb = 0.0, ub = 0.4), [0.6, 0.4, 0.0], [0.4, 0.4, 0.2]),
                 (WeightBounds(; lb = -0.3, ub = 0.5), [0.9, 0.6, -0.5], [0.5, 0.5, 0.0]),
                 (WeightBounds(; lb = [-0.026, -0.103, -0.1837, -0.1388],
                               ub = [0.0554, 0.44, 0.1866, 0.4171]),
                  [0.703, -0.5214, 1.3316, -0.5132], [0.0554, 0.3749, 0.1866, 0.3831]))
        for (wbc, wc, wp) in cases,
            wf in (IterativeWeightFinaliser(), IterativeWeightFinaliser(; iter = 10_000),
                   EuclideanWeightFinaliser(), EntropicWeightFinaliser())

            w = opt(wf, wbc, copy(wc))
            @test isapprox(w, wp; atol = 1e-4)
            @test w == proj(wc, wbc)
            @test isa(first(PortfolioOptimisers.finalise_weight_bounds(wf, wbc, copy(wc))),
                      OptimisationSuccess)
        end
        # A loop that reaches the bounds keeps its own answer, which keeps the ratio of the
        # free weights and so differs from the projection.
        wbr = WeightBounds(; lb = 0.0, ub = 0.5)
        @test opt(IterativeWeightFinaliser(), wbr, [0.9, 0.09, 0.01]) ≈ [0.5, 0.45, 0.05]
        @test opt(EuclideanWeightFinaliser(), wbr, [0.9, 0.09, 0.01]) ≈ [0.5, 0.29, 0.21]
    end
    @testset "A scalar bound is tested against every weight" begin
        # `map` over a scalar bound and a vector stopped after the first weight, so a first
        # weight inside its bounds returned the vector untouched.
        wbs = WeightBounds(; lb = 0.0, ub = 0.4)
        for wf in (IterativeWeightFinaliser(), EuclideanWeightFinaliser(),
                   EntropicWeightFinaliser(), JuMPWeightFinaliser(; slv = slv))
            w = PortfolioOptimisers.opt_weight_bounds(wf, wbs, [0.1, 0.8, 0.1])
            @test isapprox(sum(w), 1.0)
            @test all(-1e-8 .<= w .<= 0.4 + 1e-8)
        end
    end
    @testset "The Euclidean projection holds any budget and an absent bound" begin
        proj = PortfolioOptimisers.euclidean_weight_projection
        for s in (1.0, 0.5, 0.0, -0.3, 2.0)
            wbl = WeightBounds(; lb = [-0.5, -0.4, -0.3, -0.6], ub = [0.9, 0.6, 0.7, 0.8])
            wc = [1.5, -1.0, 0.4, -0.9] .+ (s - 0.0) / 4
            w = PortfolioOptimisers.opt_weight_bounds(EuclideanWeightFinaliser(), wbl, wc)
            @test isapprox(sum(w), s; atol = 1e-12)
            @test PortfolioOptimisers.weights_meet_bounds(wbl, w, s)
        end
        # The projection is the programme of the squared absolute error, in closed form.
        wbj = WeightBounds(; lb = fill(0.1, 4), ub = fill(0.3, 4))
        wj = [0.6, 0.25, 0.1, 0.05]
        wq = PortfolioOptimisers.opt_weight_bounds(JuMPWeightFinaliser(; slv = slv,
                                                                       alg = ErrorWeightFinaliser(;
                                                                                                  alg = L2Norm(),
                                                                                                  dev = AbsoluteWeightDeviation())),
                                                   wbj, copy(wj))
        @test proj(wj, wbj) ≈ [0.3, 0.3, 0.225, 0.175]
        # Clarabel solves the cone to about `5e-6`.
        @test isapprox(proj(wj, wbj), wq; atol = 1e-4)
        # An absent bound is no constraint on its side.
        @test proj([0.7, 0.4, -0.1], WeightBounds(; lb = nothing, ub = 0.5)) ≈
              [0.5, 0.5, 0.0]
        @test proj([0.7, 0.4, -0.1], WeightBounds(; lb = 0.0, ub = nothing)) ≈
              [0.65, 0.35, 0.0]
        wu = [0.25, 0.25, 0.5]
        @test PortfolioOptimisers.opt_weight_bounds(EuclideanWeightFinaliser(),
                                                    WeightBounds(; lb = 0.0, ub = 0.5),
                                                    wu) === wu
    end
    @testset "The entropic projection keeps the ratios of the free weights" begin
        opt = PortfolioOptimisers.opt_weight_bounds
        ent = PortfolioOptimisers.entropic_weight_projection
        euc = PortfolioOptimisers.euclidean_weight_projection
        # Where the loop reaches the bounds, the entropic projection is its exact limit.
        wbr = WeightBounds(; lb = 0.0, ub = 0.5)
        @test opt(EntropicWeightFinaliser(), wbr, [0.9, 0.09, 0.01]) ≈ [0.5, 0.45, 0.05]
        @test opt(EntropicWeightFinaliser(), wb, copy(w0)) ≈
              opt(IterativeWeightFinaliser(), wb, copy(w0))
        wq = [0.35, 0.3, 0.2, 0.1, 0.05]
        wbq = WeightBounds(; lb = fill(0.08, 5), ub = fill(0.28, 5))
        w = opt(EntropicWeightFinaliser(), wbq, wq)
        free = findall(wbq.lb .< w .< wbq.ub)
        @test length(free) >= 2
        @test all(isapprox.(w[free] ./ wq[free], w[free[1]] / wq[free[1]]))
        @test sum(w) ≈ 1
        # An absent bound is no constraint on its side.
        @test ent([0.7, 0.3], WeightBounds(; lb = nothing, ub = 0.5)) ≈ [0.5, 0.5]
        @test ent([0.9, 0.1], WeightBounds(; lb = 0.2, ub = nothing)) ≈ [0.8, 0.2]
        # Outside its domain it is the Euclidean projection: a zero weight that must take
        # mass (the stall of the loop), a negative weight, a negative lower bound, a budget
        # that is not positive, and a budget equal to `Σ lb`.
        for (wbc, wc) in ((WeightBounds(; lb = 0.0, ub = 0.4), [0.6, 0.4, 0.0]),
                          (WeightBounds(; lb = -0.3, ub = 0.5), [0.9, 0.6, -0.5]),
                          (WeightBounds(; lb = -0.3, ub = 0.5), [0.9, 0.1, 0.0]),
                          (WeightBounds(; lb = -0.5, ub = 0.5), [0.7, -0.2, -0.5]),
                          (WeightBounds(; lb = [0.5, 0.3, 0.2], ub = 0.8), [0.7, 0.2, 0.1]))
            @test ent(wc, wbc) == euc(wc, wbc)
            @test PortfolioOptimisers.weights_meet_bounds(wbc, ent(wc, wbc), sum(wc))
        end
        @test ent([0.6, 0.4, 0.0], WeightBounds(; lb = 0.0, ub = 0.4)) ≈ [0.4, 0.4, 0.2]
        # Under zero lower bounds the loop converges to the entropic point. Under two-sided
        # bounds it can stop elsewhere, farther from the input in relative entropy.
        kl(w, w0) = sum(w .* log.(w ./ w0))
        w3 = [0.5, 0.4, 0.1]
        wb3 = WeightBounds(; lb = 0.0, ub = 0.45)
        @test opt(IterativeWeightFinaliser(), wb3, copy(w3)) ≈ [0.45, 0.44, 0.11]
        @test opt(EntropicWeightFinaliser(), wb3, copy(w3)) ≈ [0.45, 0.44, 0.11]
        wb3 = WeightBounds(; lb = 0.3, ub = 0.45)
        wi = opt(IterativeWeightFinaliser(; iter = 10_000), wb3, copy(w3))
        we = opt(EntropicWeightFinaliser(), wb3, copy(w3))
        @test wi ≈ [0.4, 0.3, 0.3]
        @test we ≈ [3.5, 2.8, 2.7] ./ 9
        @test kl(we, w3) < kl(wi, w3)
    end
    @testset "An integer weight vector projects as its float copy does" begin
        # An absent bound was `typemin(Int)`, which is finite, so `w .- typemin` overflowed
        # and `[0, -5]` under `ub = 0` projected to `-9.2e18`. The iterative loop wrote a
        # float into an integer vector, and `weights_meet_bounds` had no `eps(Int)`.
        opt = PortfolioOptimisers.opt_weight_bounds
        euc = PortfolioOptimisers.euclidean_weight_projection
        ent = PortfolioOptimisers.entropic_weight_projection
        @test euc([0, -5], WeightBounds(; lb = nothing, ub = 0)) == [0.0, -5.0]
        @test ent([4, 1], WeightBounds(; lb = -3, ub = nothing)) == [4.0, 1.0]
        for (wbi, wbf) in
            ((WeightBounds(; lb = nothing, ub = 1), WeightBounds(; lb = nothing, ub = 1.0)),
             (WeightBounds(; lb = -1, ub = 1), WeightBounds(; lb = -1.0, ub = 1.0)),
             (WeightBounds(; lb = 0, ub = nothing), WeightBounds(; lb = 0.0, ub = nothing))),
            wf in (IterativeWeightFinaliser(), EuclideanWeightFinaliser(),
                   EntropicWeightFinaliser()), wi in ([3, -2, 0], [2, 1, 0], [-1, 1, 1])

            rf, wf_ = PortfolioOptimisers.finalise_weight_bounds(wf, wbf, Float64.(wi))
            ri, wi_ = PortfolioOptimisers.finalise_weight_bounds(wf, wbi, copy(wi))
            @test typeof(ri) == typeof(rf)
            @test isequal(wi_, wf_)
        end
    end
    @testset "A Rational weight vector meets its bounds exactly (#1349)" begin
        # `weights_meet_bounds` took `sqrt(eps(T))`, and `Rational` has no `eps`, so every
        # finaliser threw on `Rational` weights, also with no bounds at all.
        tolf = PortfolioOptimisers.weights_bound_tolerance
        meet = PortfolioOptimisers.weights_meet_bounds
        @test tolf(Float64) == sqrt(eps(Float64))
        @test tolf(Float32) === sqrt(eps(Float32))
        @test tolf(Rational{Int}) === 0 // 1
        w = fill(1 // 3, 3)
        wb = WeightBounds(; lb = 0 // 1, ub = 1 // 2)
        @test meet(wb, w, 1 // 1)
        @test meet(WeightBounds(; lb = nothing, ub = nothing), w, 1 // 1)
        # Exact arithmetic has no round-off, so the smallest miss fails.
        @test !meet(WeightBounds(; lb = 0 // 1, ub = 1 // 3 - 1 // 10^12), w, 1 // 1)
        @test !meet(wb, w, 1 // 1 + 1 // 10^12)
        # A finaliser that moves the weights lands on the bounds and the budget exactly.
        wr = [1 // 2, 3 // 10, 3 // 20, 1 // 20]
        wbr = WeightBounds(; lb = 1 // 10, ub = 2 // 5)
        for (wf, we) in ((IterativeWeightFinaliser(), [2 // 5, 1 // 3, 1 // 6, 1 // 10]),
                         (EuclideanWeightFinaliser(), [2 // 5, 13 // 40, 7 // 40, 1 // 10]),
                         (EntropicWeightFinaliser(), [2 // 5, 1 // 3, 1 // 6, 1 // 10]))
            retcode, wf_ = PortfolioOptimisers.finalise_weight_bounds(wf, wbr, copy(wr))
            @test isa(retcode, OptimisationSuccess)
            @test wf_ == we
            @test eltype(wf_) == Rational{Int}
        end
        X = Rational.(rand(StableRNG(1349), -20:20, 50, 3)) .// 1000
        rd = ReturnsResult(; nx = ["a", "b", "c"], X = X,
                           ts = Date(2024, 1, 1) .+ Day.(0:49))
        for opt in (EqualWeighted(), EqualWeighted(; wb = nothing))
            res = optimise(opt, rd)
            @test isa(res.retcode, OptimisationSuccess)
            @test res.w == fill(1 // 3, 3)
            @test eltype(res.w) == Rational{Int}
        end
    end
    @testset "The relative formulations divide by eps in a copy of a zero weight" begin
        # The formulation writes `eps` into a copy, so the caller's weights stay as they
        # are (#785), and the zero weights still reach the bounds.
        wz = [0.6, 0.4, 0.0, 0.0]
        wzc = copy(wz)
        w = PortfolioOptimisers.opt_weight_bounds(JuMPWeightFinaliser(; slv = slv),
                                                  WeightBounds(; lb = fill(0.05, 4),
                                                               ub = fill(0.5, 4)), wzc)
        @test wzc == wz
        @test all(x -> 0.05 - 1e-8 <= x <= 0.5 + 1e-8, w)
        @test isapprox(sum(w), 1; rtol = 1e-8)
    end
end
@testset "assert_special_nco_requirements over a vector" begin
    using PortfolioOptimisers, Test
    # The vector arity broadcast, so the `::Nothing` annotation could not convert the
    # `Vector{Nothing}` the broadcast returned, and every vector threw.
    mr = MeanRisk(; opt = JuMPOptimiser(; slv = Solver(; solver = nothing)))
    @test isnothing(PortfolioOptimisers.assert_special_nco_requirements(mr))
    @test isnothing(PortfolioOptimisers.assert_special_nco_requirements([mr, mr]))
    @test isnothing(PortfolioOptimisers.assert_special_nco_requirements(TimeDependent([mr,
                                                                                       mr];
                                                                                      default = mr)))
end
@testset "PreviousWeights holds the weights it was handed (#1021)" begin
    using PortfolioOptimisers, Test, StableRNGs
    PO = PortfolioOptimisers
    rd = ReturnsResult(; nx = ["A", "B", "C"], X = randn(StableRNG(3), 10, 3))
    # Verbatim, on the full universe, with no bounds and no mask.
    pw = PreviousWeights(; w = [0.6, 0.3, 0.1])
    res = optimise(pw, rd)
    @test isa(res, NaiveOptimisationResult)
    @test isa(res.retcode, OptimisationSuccess)
    @test res.w === pw.w
    @test isnothing(res.imsk) && isnothing(res.wb) && res.pr === rd
    # The fold-less entry and the read-out entry answer the same.
    @test optimise(pw).w === pw.w
    @test PO.optimise(PO.partial_fit!(pw, rd)).w === pw.w
    # No weights is a failure that names the field, so a chain walks on.
    none = optimise(PreviousWeights())
    @test isa(none.retcode, OptimisationFailure)
    @test occursin("`w` is `nothing`", none.retcode.res)
    @test isnothing(none.w)
    # With a carrier the failure carries `NaN` at every asset, as any failed solve does.
    none_rd = optimise(PreviousWeights(), rd)
    @test isa(none_rd.retcode, OptimisationFailure)
    @test length(none_rd.w) == 3 && all(isnan, none_rd.w)
    # A chain that reaches the leaf holds; one that reaches an empty leaf fails through.
    slv = Solver(; name = :none, solver = nothing)
    @test PO.needs_previous_weights(PreviousWeights())
    @test PO.needs_previous_weights(EqualWeighted(; fb = PreviousWeights()))
    @test !PO.needs_previous_weights(EqualWeighted())
    # The factory fills `w` and recurses into `fb`; a non-vector leaves the leaf as it is.
    filled = PO.factory(PreviousWeights(; fb = PreviousWeights()), [0.2, 0.8])
    @test filled.w == [0.2, 0.8] && filled.fb.w == [0.2, 0.8]
    @test PO.factory(PreviousWeights(), [[0.2, 0.8], [0.5, 0.5]]) == PreviousWeights()
    @test_throws DomainError PreviousWeights(; w = [0.5, Inf])
    @test_throws ArgumentError PreviousWeights(;
                                               fb = TimeDependent([EqualWeighted()],
                                                                  :nearest))
    @test PreviousWeights(; fb = TimeDependent([EqualWeighted()])) isa PreviousWeights
end
@testset "A result records the fallback chain that answered it (#1024)" begin
    using PortfolioOptimisers, Test, StableRNGs
    PO = PortfolioOptimisers
    rd = ReturnsResult(; nx = ["A", "B", "C"], X = randn(StableRNG(3), 10, 3))
    # Two failures before the answer: the chain holds both, in the order they ran, and
    # each failure carries no chain of its own.
    chain = PreviousWeights(; fb = PreviousWeights(; fb = EqualWeighted()))
    res = optimise(chain, rd)
    @test isa(res, NaiveOptimisationResult)
    @test isa(res.retcode, OptimisationSuccess)
    @test isa(res.fb, PO.FbChain)
    @test length(res.fb) == 2
    @test res.fb[1][1] === chain
    @test res.fb[2][1] === chain.fb
    @test all(isa(p[2], NaiveOptimisationResult) for p in res.fb)
    @test all(isa(p[2].retcode, OptimisationFailure) for p in res.fb)
    @test all(isnothing(p[2].fb) for p in res.fb)
    # The answer of a fallback-less estimator carries `nothing`.
    @test isnothing(optimise(EqualWeighted(), rd).fb)
    # A JuMP failure enters the chain like any other.
    slv = Solver(; name = :none, solver = nothing)
    mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv), fb = EqualWeighted())
    mr_res = optimise(mr, rd)
    @test isa(mr_res, NaiveOptimisationResult)
    @test length(mr_res.fb) == 1
    @test mr_res.fb[1][1] === mr
    @test isa(mr_res.fb[1][2], MeanRiskResult)
    @test isa(mr_res.fb[1][2].retcode, OptimisationFailure)
    # When every attempt fails, the last failure carries the ones before it.
    res = optimise(PreviousWeights(; fb = PreviousWeights()), rd)
    @test isa(res.retcode, OptimisationFailure)
    @test length(res.fb) == 1
    @test isa(res.fb[1][2].retcode, OptimisationFailure)
    # The rebuild admits the chain and the estimator alike, and `set_retcode` carries the
    # chain over.
    @test isnothing(PO.factory(res, nothing).fb)
    @test PO.factory(res, EqualWeighted()).fb == EqualWeighted()
    failed_mr = PO.factory(mr_res.fb[1][2], mr_res.fb)
    @test failed_mr.fb === mr_res.fb
    @test PO.set_retcode(failed_mr, OptimisationSuccess()).fb === mr_res.fb
end
@testset "A PreviousWeights fallback holds its slice under an asset-subset view (#1339)" begin
    using PortfolioOptimisers, Test, StableRNGs
    PO = PortfolioOptimisers
    slv = Solver(; name = :none, solver = nothing)
    X4 = randn(StableRNG(2), 10, 4)
    # The view slices `w`, and recurses into the fallback of the fallback.
    pw = PreviousWeights(; w = [0.4, 0.3, 0.2, 0.1],
                         fb = PreviousWeights(; w = fill(0.25, 4)))
    pwv = PO.port_opt_view(pw, [1, 3], X4)
    @test pwv.w == [0.4, 0.2]
    @test pwv.fb.w == [0.25, 0.25]
    @test isnothing(PO.port_opt_view(PreviousWeights(), [1, 3], X4).w)
    # Every holder views its fallback: a JuMP head, a hierarchical head and a naive head.
    fb = PreviousWeights(; w = fill(0.25, 4))
    mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv), fb = fb)
    @test PO.port_opt_view(mr, [1, 2], X4).fb.w == [0.25, 0.25]
    hrp = HierarchicalRiskParity(; fb = fb)
    @test PO.port_opt_view(hrp, [2, 4], X4).fb.w == [0.25, 0.25]
    ew = EqualWeighted(; fb = fb)
    @test PO.port_opt_view(ew, [1, 2, 3], X4).fb.w == fill(0.25, 3)
    # A precomputed fallback answers on the universe it was solved on, so a view keeps
    # it. A schedule that holds one still refuses a subset view.
    rd4 = ReturnsResult(; nx = string.('a':'d'), X = X4)
    res = optimise(EqualWeighted(), rd4)
    @test PO.port_opt_view(MeanRisk(; opt = JuMPOptimiser(; slv = slv), fb = res), [1, 2],
                           X4).fb === res
    @test PO.port_opt_view(EqualWeighted(; fb = res), [1, 2], X4).fb === res
    @test_throws ArgumentError PO.port_opt_view(MeanRisk(; opt = JuMPOptimiser(; slv = slv),
                                                         fb = TimeDependent([EqualWeighted(),
                                                                             res])), [1, 2],
                                                X4)
    # The reproduction of #1339: a cluster whose solve fails falls back to the held
    # book, sliced to the cluster, where it used to throw a `DimensionMismatch`.
    X8 = randn(StableRNG(2), 200, 8) ./ 100 .+ 0.001
    rd8 = ReturnsResult(; nx = string.('a':'h'), X = X8)
    w8 = collect(1.0:8.0) ./ 36
    inner = MeanRisk(; opt = JuMPOptimiser(; slv = slv), fb = PreviousWeights(; w = w8))
    nres = optimise(NestedClustered(; opti = inner, opto = EqualWeighted()), rd8)
    @test length(nres.w) == 8
    cls = [findall(==(k), PO.assignments(nres.clr)) for k in 1:(nres.clr.k)]
    @test all(nres.resi[k].w == w8[cls[k]] for k in eachindex(cls))
    @test all(isa(r.retcode, OptimisationSuccess) for r in nres.resi)
end

# Solver-free tests for the shared JuMP model-assembly pipeline (ADR 0008).
#
# These build a model through the per-optimiser head and `assemble_jump_model!`, then
# assert on the *constructed* model — without ever solving it. The interface under test is
# the assembler's keyword arguments (`r`, `l2c`/`l1`/… settings); the assertions check which
# Model-State keys the middle registers, so they exercise the routing rather than numerics.
@testset "JuMP model assembly (solver-free)" begin
    using Test, PortfolioOptimisers, CSV, TimeSeries, Clarabel, JuMP
    PO = PortfolioOptimisers
    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end])
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = "verbose" => false)

    # Run the head + assembler for `mr`, stop before solving, return the set of registered
    # Model-State key names. `r`/`b1`/`obj`/… default to the optimiser's own values but can
    # be overridden to probe a single branch in isolation.
    function assemble_keys(mr; r = mr.r, b1 = nothing, obj = mr.obj,
                           sdp_asset_phylogeny = true)
        attrs = PO.processed_jump_optimiser_attributes(mr.opt, rd)
        model = JuMP.Model()
        PO.set_model_scales!(model, mr.opt.sc, mr.opt.so)
        PO.set_model_observations!(model, size(attrs.pr.X, 1))
        # Reach the head's own seams, not a copy of them: `set_maximum_ratio_factor_variables!`
        # takes exactly one objective, and the budget group travels as `mr.opt`. A hand-copied
        # call used to pass three arguments here and to name `bgt`/`sbgt` alone, so the
        # `MaximumRatio` branch was never built and `gbgt` never reached the model.
        PO.set_maximum_ratio_factor_variables!(model, obj)
        PO.set_w!(model, attrs.pr.X, mr.wi)
        PO.set_weight_constraints!(model, attrs.wb, mr.opt)
        PO.assemble_jump_model!(model, mr, mr.opt, attrs, rd, r, obj, b1,
                                sdp_asset_phylogeny)
        return Set(string.(keys(JuMP.object_dictionary(model))))
    end

    base = MeanRisk(; r = Variance(), opt = JuMPOptimiser(; slv = slv))

    @testset "core state is established" begin
        ks = assemble_keys(base)
        @test "w" in ks
        @test "k" in ks
        @test "ret" in ks
    end

    # `k` has two spellings and the objective picks between them. A wrong-arity call to the
    # producer used to be absorbed by a variadic fallback, which registered the constant
    # under every objective — so this branch built but was never the one under test.
    @testset "the objective picks the spelling of `k`" begin
        model = JuMP.Model()
        PO.set_maximum_ratio_factor_variables!(model, MinimumRisk())
        @test model[:k] == 1
        @test !isa(model[:k], JuMP.VariableRef)

        rmodel = JuMP.Model()
        PO.set_maximum_ratio_factor_variables!(rmodel, MaximumRatio())
        @test isa(rmodel[:k], JuMP.VariableRef)
        @test JuMP.lower_bound(rmodel[:k]) == 0

        # The producer takes exactly one objective; a wrong-arity call must not be absorbed.
        @test_throws MethodError PO.set_maximum_ratio_factor_variables!(JuMP.Model(),
                                                                        nothing,
                                                                        MaximumRatio())
    end

    # The budget group travels as one object, so a head cannot read part of it. `gbgt` binds
    # only once the long/short decomposition exists, which a negative weight bound forces.
    @testset "the gross budget reaches the model through the optimiser" begin
        gopt = JuMPOptimiser(; slv = slv, wb = WeightBounds(; lb = -1.0, ub = 1.0),
                             bgt = nothing, sbgt = nothing, gbgt = 2.0)
        ks = assemble_keys(MeanRisk(; r = Variance(), opt = gopt))
        @test "gbgt" in ks
        @test "gbgt" ∉ assemble_keys(base)
    end

    @testset "risk is routed by the `r` kwarg" begin
        ks = assemble_keys(base)
        @test "variance_risk_1" in ks
        @test "variance_risk_2" ∉ ks

        # A vector of risk measures registers one indexed expression each.
        ksv = assemble_keys(MeanRisk(; r = [Variance(), Variance()],
                                     opt = JuMPOptimiser(; slv = slv)))
        @test "variance_risk_1" in ksv
        @test "variance_risk_2" in ksv

        # r = nothing → the risk + scalarise step is a no-op (the RelaxedRiskBudgeting path),
        # but the head and the return constraints are still applied.
        ksn = assemble_keys(base; r = nothing)
        @test "variance_risk_1" ∉ ksn
        @test "risk" ∉ ksn
        @test "w" in ksn
        @test "ret" in ksn
    end

    @testset "optional constraints toggle on their settings" begin
        @test "l2c" ∉ assemble_keys(base)
        @test "l2c" in assemble_keys(MeanRisk(; r = Variance(),
                                              opt = JuMPOptimiser(; slv = slv, l2c = 0.5)))

        @test "l1" ∉ assemble_keys(base)
        @test "l1" in assemble_keys(MeanRisk(; r = Variance(),
                                             opt = JuMPOptimiser(; slv = slv, l1 = 0.1)))
    end

    @testset "integer phylogeny gates on the held indicator, not a fixed key" begin
        # `set_iplg_constraints!` re-read `model[:ib]`, which only the long-only builder
        # registers. Anything that routes the model to the long-short builder — a short
        # threshold here, but fixed fees or `xbgt` do it too — registers `ilb`/`isb`/`i_mip`
        # and no `:ib`, so the phylogeny constraint threw `KeyError(:ib)` at assembly time.
        # The builder now registers its bundle in Model State (`set_mip_indicators!`) and the
        # late emitter reads it back with `mip_indicators` + `held_bin` — no key involved.
        mr = MeanRisk(; r = Variance(),
                      opt = JuMPOptimiser(; slv = slv, bgt = 1, sbgt = 1,
                                          wb = WeightBounds(; lb = -1, ub = 1),
                                          ple = IntegerPhylogenyEstimator(;
                                                                          pl = NetworkEstimator(),
                                                                          B = 1),
                                          st = Threshold(; val = 0.05)))
        ks = assemble_keys(mr)
        @test "card_plg_1" in ks       # the phylogeny constraint was emitted at all
        @test "i_mip" in ks            # ... gated on the long-short builder's held indicator
        @test "ib" ∉ ks                # ... which is not `:ib`; that key is never registered

        # The long-only builder still routes through the same call and keeps `:ib`.
        mrl = MeanRisk(; r = Variance(),
                       opt = JuMPOptimiser(; slv = slv, bgt = 1, sbgt = 1,
                                           wb = WeightBounds(; lb = -1, ub = 1),
                                           ple = IntegerPhylogenyEstimator(;
                                                                           pl = NetworkEstimator(),
                                                                           B = 1)))
        ksl = assemble_keys(mrl)
        @test "card_plg_1" in ksl
        @test "ib" in ksl
    end

    @testset "lp penalty and lpc constraint register distinct keys" begin
        # The p-norm weight constraint and the Lp regularisation penalty both build
        # PowerCone epigraphs. They must not register under the same Model-State names,
        # or whichever runs second silently clobbers the other's handle in the object
        # dictionary (JuMP's `model[k] = v` overwrites without complaint).
        ks = assemble_keys(MeanRisk(; r = Variance(),
                                    opt = JuMPOptimiser(; slv = slv,
                                                        lp = LpRegularisation(; p = 3,
                                                                              val = 0.1),
                                                        lpc = LpRegularisation(; p = 3,
                                                                               val = 0.5))))
        for k in ("clp_1", "cslp_1", "t_lp_1", "r_lp_1")       # Lp penalty
            @test k in ks
        end
        for k in ("clpc_1", "cslpc_1", "t_lpc_1", "r_lpc_1")   # p-norm constraint
            @test k in ks
        end
    end
end

# Tests for the custom JuMP objective/constraint extension points.
#
# `cobj`/`ccnt` on a `JuMPOptimiser` are the user-facing escape hatch: subtype
# `CustomJuMPObjective`/`CustomJuMPConstraint` and implement one builder method. Nothing else
# in the suite exercises them, so this file pins the contract:
#   - a single estimator reaches the model and moves the solution,
#   - a VECTOR of them applies every element (both vector paths were broken: a `ccnt` vector
#     silently hit the no-op `args...` fallback, a `cobj` vector raised a MethodError),
#   - the `get_k` idiom is load-bearing under a ratio objective,
#   - the hook still fires through `NearOptimalCentering`, which calls it from its own
#     objective builder rather than through `set_portfolio_objective_function!`,
#   - a subtype with no method defined RAISES rather than silently contributing nothing,
#   - the same term is sense-correct under a minimisation and a maximisation with no
#     sign dispatch by the implementer, and a QUADRATIC term works against an AFFINE
#     objective — both consequences of routing custom terms through the objective penalty
#     (ADR 0036).
#
# The custom types must be defined at module top level (a `struct` cannot be declared inside
# the `@testset` block).
using Test, PortfolioOptimisers, CSV, TimeSeries, Clarabel, JuMP, LinearAlgebra, Statistics,
      Logging
const PO = PortfolioOptimisers

# A soft reward on a per-asset score. Contributed as a NEGATIVE penalty: the library applies
# the factor matching each objective's optimisation sense, so this one definition rewards
# score exposure under a minimisation and a maximisation alike.
struct ScoreTilt{T1, T2} <: PO.CustomJuMPObjective
    score::T1
    lambda::T2
end
function PO.add_custom_objective_term!(model::JuMP.Model, obj, cobj::ScoreTilt, optimiser,
                                       attrs)
    w = PO.get_w(model)
    PO.add_to_objective_penalty!(model, -cobj.lambda * (cobj.score' * w))
    return nothing
end

# A quadratic concentration penalty. Before ADR 0036 this was impossible against an affine
# objective: the hook mutated `obj_expr` in place, and `add_to_expression!(::AffExpr,
# ::QuadExpr)` is a MethodError. The penalty accumulator promotes instead.
struct SpreadPenalty{T} <: PO.CustomJuMPObjective
    lambda::T
end
function PO.add_custom_objective_term!(model::JuMP.Model, obj, cobj::SpreadPenalty,
                                       optimiser, attrs)
    w = PO.get_w(model)
    PO.add_to_objective_penalty!(model, cobj.lambda * JuMP.@expression(model, dot(w, w)))
    return nothing
end

# A hard floor on the same score exposure, following the model idiom: scale by the constraint
# scale, and lift the constant bound to `k`.
struct ScoreFloor{T1, T2} <: PO.CustomJuMPConstraint
    score::T1
    floor::T2
end
function PO.add_custom_constraint!(model::JuMP.Model, ccnt::ScoreFloor, optimiser, attrs)
    w = PO.get_w(model)
    k = PO.get_k(model)
    sc = PO.get_constraint_scale(model)
    JuMP.@constraint(model, sc * (ccnt.score' * w - ccnt.floor * k) >= 0)
    return nothing
end

# The floor's mirror image, used to prove a `ccnt` vector applies *both* elements.
struct ScoreCap{T1, T2} <: PO.CustomJuMPConstraint
    score::T1
    cap::T2
end
function PO.add_custom_constraint!(model::JuMP.Model, ccnt::ScoreCap, optimiser, attrs)
    w = PO.get_w(model)
    k = PO.get_k(model)
    sc = PO.get_constraint_scale(model)
    JuMP.@constraint(model, sc * (ccnt.cap * k - ccnt.score' * w) >= 0)
    return nothing
end

# The same floor with the `* k` omitted — correct only where `k == 1`.
struct ScoreFloorNoK{T1, T2} <: PO.CustomJuMPConstraint
    score::T1
    floor::T2
end
function PO.add_custom_constraint!(model::JuMP.Model, ccnt::ScoreFloorNoK, optimiser, attrs)
    w = PO.get_w(model)
    sc = PO.get_constraint_scale(model)
    JuMP.@constraint(model, sc * (ccnt.score' * w - ccnt.floor) >= 0)
    return nothing
end

# Records the arguments the objective hook is handed, without touching `obj_expr`. Used to
# pin the calling convention on every path that invokes the hook.
struct SpyObjective{T} <: PO.CustomJuMPObjective
    log::T
end
function PO.add_custom_objective_term!(model::JuMP.Model, obj, cobj::SpyObjective,
                                       optimiser, attrs)
    push!(cobj.log, (; obj, optimiser, attrs))
    return nothing
end

# Subtypes that never define a builder method: they must hit the no-op fallback.
struct InertConstraint <: PO.CustomJuMPConstraint end
struct InertObjective <: PO.CustomJuMPObjective end

@testset "custom JuMP objective/constraint hooks" begin
    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end])
    pr = prior(EmpiricalPrior(), rd)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = "verbose" => false)
    rf = 4.2 / 100 / 252

    # Standardised trailing-63-day momentum: a continuous per-asset preference.
    score = let m = vec(sum(rd.X[(end - 62):end, :]; dims = 1))
        (m .- sum(m) / length(m)) ./ std(m)
    end
    exposure(res) = dot(score, res.w)

    minrisk(; kwargs...) = optimise(MeanRisk(; obj = MinimumRisk(),
                                             opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                                 kwargs...)))
    ratio(; kwargs...) = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                                           opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                               kwargs...)))

    base = minrisk()
    base_exp = exposure(base)

    @testset "objective hook: a single term reaches the objective" begin
        tilted = minrisk(; cobj = ScoreTilt(score, 1e-4))
        # The tilt rewards score exposure, so it must rise above the untilted baseline.
        @test exposure(tilted) > base_exp + 1e-3
    end

    @testset "objective hook: a vector composes additively" begin
        # Regression: a `cobj` vector used to raise a MethodError, because every
        # `set_portfolio_objective_function!` typed `cobj::Option{<:CustomJuMPObjective}`.
        # Two λ terms accumulate into the same expression, so they must equal one 2λ term.
        two = minrisk(; cobj = [ScoreTilt(score, 1e-4), ScoreTilt(score, 1e-4)])
        one = minrisk(; cobj = ScoreTilt(score, 2e-4))
        @test isapprox(exposure(two), exposure(one); rtol = 1e-5)
        @test exposure(two) > base_exp + 1e-3
    end

    @testset "constraint hook: a single constraint binds at its bound" begin
        f = base_exp + 0.5                     # above the free optimum, so it must bind
        res = minrisk(; ccnt = ScoreFloor(score, f))
        @test isapprox(exposure(res), f; atol = 1e-4)
    end

    @testset "constraint hook: a vector applies every element" begin
        # Regression: a `ccnt` vector used to hit the no-op `args...` fallback and be
        # SILENTLY DROPPED — the solve returned the unconstrained weights.
        lo = base_exp + 0.3
        hi = lo + 0.2
        res = minrisk(; ccnt = [ScoreFloor(score, lo), ScoreCap(score, hi)])
        e = exposure(res)
        @test e >= lo - 1e-4                   # the floor element was applied
        @test e <= hi + 1e-4                   # ... and so was the cap element
        @test e > base_exp + 1e-3              # the pair genuinely moved the solution
    end

    @testset "constraint hook: the `k` idiom is load-bearing under a ratio objective" begin
        # `MaximumRatio` solves in a rescaled space (`w_real == w / k`), so a constant bound
        # must be multiplied by `k`. Under `MinimumRisk` `k == 1` and both spellings agree —
        # which is exactly why omitting it hides.
        f_min = base_exp + 0.3
        @test isapprox(exposure(minrisk(; ccnt = ScoreFloor(score, f_min))),
                       exposure(minrisk(; ccnt = ScoreFloorNoK(score, f_min))); rtol = 1e-5)

        # Under the ratio objective only the k-scaled floor honours the requested level.
        ratio_exp = exposure(ratio())
        f = ratio_exp + 0.02                   # just above the free optimum: binds, stays feasible
        right = ratio(; ccnt = ScoreFloor(score, f))
        wrong = ratio(; ccnt = ScoreFloorNoK(score, f))
        @test isapprox(exposure(right), f; atol = 1e-4)
        @test !isapprox(exposure(wrong), f; atol = 1e-3)
    end

    @testset "objective hook: the same term is sense-correct without sign dispatch" begin
        # ADR 0036: a term contributed to the objective penalty is oriented by the library,
        # so ONE definition rewards score exposure under a minimisation and a maximisation
        # alike. Before this, the implementer had to dispatch a sign on the objective type
        # and get it right, and `MaximumRatio` had no correct fixed sign at all.
        util(; kwargs...) = optimise(MeanRisk(; obj = MaximumUtility(),
                                              opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                                  kwargs...)))
        @test exposure(util(; cobj = ScoreTilt(score, 5e-3))) > exposure(util()) + 1e-3
        # ... and the same term still rewards under the minimisation.
        @test exposure(minrisk(; cobj = ScoreTilt(score, 1e-4))) > base_exp + 1e-3
    end

    @testset "objective hook: a quadratic term works against an affine objective" begin
        # Regression: `MaximumReturn` builds an AFFINE `obj_expr`, so a quadratic term used to
        # be a MethodError from `add_to_expression!(::AffExpr, ::QuadExpr)` — a custom term
        # whose validity depended on the risk measure and objective. The penalty accumulator
        # promotes, so it now works everywhere.
        maxret(; kwargs...) = optimise(MeanRisk(; obj = MaximumReturn(),
                                                opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                                    kwargs...)))
        plain = maxret()
        spread = maxret(; cobj = SpreadPenalty(1e2))
        # An unconstrained max-return solve piles into the highest-mean asset; the quadratic
        # penalty prices concentration, so the book must spread out.
        @test dot(spread.w, spread.w) < dot(plain.w, plain.w) - 1e-4
    end

    @testset "objective hook is called with the documented argument order" begin
        # The contract is `(model, obj, cobj, optimiser, attrs)`. A caller that shifts `cobj`
        # out of position 3 means the user's method misses dispatch — which now RAISES via the
        # typed fallback rather than silently dropping the term. `NearOptimalCentering` invokes
        # the hook from its own objective builder rather than through
        # `set_portfolio_objective_function!`, so assert the convention holds on both paths.
        # Only the *constrained* NOC algorithm routes through that call.
        #
        # This is deliberately solver-free: the spy records the call during model assembly, so
        # it pins the calling convention even where the subsequent solve is ill-conditioned.
        function spy_log(build)
            log = Any[]
            with_logger(NullLogger()) do
                return build(SpyObjective(log))
            end
            return log
        end

        main = spy_log(spy -> minrisk(; cobj = spy))
        @test length(main) == 1
        @test main[1].obj isa PO.ObjectiveFunction
        @test main[1].optimiser isa MeanRisk
        @test main[1].attrs isa ProcessedJuMPOptimiserAttributes

        noc = spy_log(spy -> optimise(NearOptimalCentering(; r = StandardDeviation(),
                                                           obj = MinimumRisk(),
                                                           alg = ConstrainedNearOptimalCentering(),
                                                           opt = JuMPOptimiser(; pe = pr,
                                                                               slv = slv,
                                                                               cobj = spy))))
        # NOC also solves `MeanRisk` sub-problems for its reference points, and those reach the
        # hook through the main path — so a non-empty log proves nothing. The centring
        # objective's own call is the one under test, and it is the only invocation carrying
        # the `NearOptimalCentering` estimator in the `optimiser` slot. (Before ADR 0036 this
        # call was the odd one out for the wrong reason: it passed a `JuMPOptimiser` there,
        # a different branch of the type tree from every other caller.)
        @test any(e -> isa(e.optimiser, NearOptimalCentering), noc)
        @test all(e -> e.obj isa PO.ObjectiveFunction, noc)
        @test all(e -> e.optimiser isa PO.OptimisationEstimator, noc)
        @test all(e -> e.attrs isa ProcessedJuMPOptimiserAttributes, noc)
    end

    @testset "fallback: a subtype with no builder method raises" begin
        # ADR 0036: the untyped `args...` catch-all used to absorb these silently, which is how
        # a `ccnt` vector once vanished without a word — and is how every stale hook would have
        # vanished when this signature changed. A subtype that defines no method is a user
        # error, and now says so.
        @test_throws ArgumentError minrisk(; cobj = InertObjective())
        @test_throws ArgumentError minrisk(; ccnt = InertConstraint())
        # `nothing` — no custom term configured — remains the one legitimate no-op.
        @test isapprox(minrisk(; cobj = nothing, ccnt = nothing).w, base.w; atol = 1e-6)
    end
end

# The docstrings of `src/17_Optimisation/05_JuMP/01_Base_JuMPOptimisation/` checked against
# numbers: the type families, the custom hooks, the model accessors, the Model State verbs,
# the frontier builders and the model setup. Most testsets build a model by hand and never
# solve it; the last ones solve a five-asset panel with Clarabel.
using Test, PortfolioOptimisers, JuMP, Clarabel, HiGHS, StableRNGs, LinearAlgebra
using InteractiveUtils: subtypes

# A subtype of each hook with no method of its own. A struct is defined at the top level.
struct HookConstraint03d <: PortfolioOptimisers.CustomJuMPConstraint end
struct HookObjective03d <: PortfolioOptimisers.CustomJuMPObjective end

@testset "Base JuMP optimisation against its docstrings" begin
    PO = PortfolioOptimisers

    @testset "the objective and return families have the stated members" begin
        # Four classic objectives and the internal corner objective.
        @test Set(subtypes(PO.ObjectiveFunction)) ==
              Set([MinimumRisk, MaximumReturn, MaximumUtility, MaximumRatio,
                   PO.MaximumElementReturn])
        @test Set(subtypes(PO.JuMPReturnsEstimator)) ==
              Set([ArithmeticReturn, LogarithmicReturn, NoReturn])
    end

    @testset "a custom hook with no method raises, and nothing is the only no-op" begin
        model = JuMP.Model()
        @test isnothing(PO.add_custom_constraint!(model, nothing, nothing, nothing))
        @test isnothing(PO.add_custom_objective_term!(model, MinimumRisk(), nothing,
                                                      nothing, nothing))
        @test_throws ArgumentError PO.add_custom_constraint!(model, HookConstraint03d(),
                                                             nothing, nothing)
        @test_throws ArgumentError PO.add_custom_objective_term!(model, MinimumRisk(),
                                                                 HookObjective03d(),
                                                                 nothing, nothing)
        # The vector methods reach the per-type method of each entry.
        @test_throws ArgumentError PO.add_custom_constraint!(model, [HookConstraint03d()],
                                                             nothing, nothing)
        @test_throws ArgumentError PO.add_custom_objective_term!(model, MinimumRisk(),
                                                                 [HookObjective03d()],
                                                                 nothing, nothing)
        @test !PO.needs_previous_weights(HookConstraint03d())
        @test !PO.needs_previous_weights([HookConstraint03d(), HookConstraint03d()])
        @test !PO.needs_previous_weights(HookObjective03d())
        @test !PO.needs_previous_weights([HookObjective03d()])
        @test isnothing(PO.port_opt_view(HookConstraint03d(), 1:2))
        @test isnothing(PO.port_opt_view(HookObjective03d(), 1:2))
        @test_throws IsEmptyError PO.JuMPOptimisationSolution(; w = Float64[])
    end

    @testset "every accessor raises on a model that lacks its entry" begin
        model = JuMP.Model()
        for f in
            (PO.get_constraint_scale, PO.get_objective_scale, PO.get_T, PO.get_w, PO.get_k,
             PO.get_ret, PO.get_risk, PO.get_net_X)
            @test_throws ArgumentError f(model)
        end
        @test !PO.has_net_X(model)
        # The message names the builder that registers `:ret`.
        msg = try
            PO.get_ret(model)
        catch e
            e.msg
        end
        @test occursin("scalarise_return_expression!", msg)
        @test isnothing(PO.decomposition_contract(model))
        @test !PO.is_unit_budget(model)
    end

    @testset "scales, observation count, unit budget and decomposition contract" begin
        model = JuMP.Model()
        PO.set_model_scales!(model, 2.0, 3.0)
        PO.set_model_observations!(model, 7)
        # JuMP stores a number given to `@expressions` as the number itself.
        @test PO.get_constraint_scale(model) === 2.0
        @test PO.get_objective_scale(model) === 3.0
        @test PO.get_T(model) == 7
        JuMP.@variable(model, k)
        @test PO.effective_k(model) === k
        PO.set_unit_budget!(model)
        @test PO.is_unit_budget(model)
        @test PO.effective_k(model) == 1
        @test PO.get_k(model) === k
        # The first record stays.
        PO.set_decomposition_contract!(model, PO.WeightsFromParts())
        PO.set_decomposition_contract!(model, PO.PartsBoundWeights())
        @test PO.decomposition_contract(model) isa PO.WeightsFromParts
    end

    @testset "Model State: keys, the collision guard, build once and flags" begin
        model = JuMP.Model()
        p = Symbol("")
        @test PO.state_key(p, :ret_, 1) == :ret_1
        @test PO.state_key(:a_, :x) == :a_x
        # Two different sets of parts give one key, and the second registration raises.
        @test PO.state_key(p, :tr_dr_, 11) == PO.state_key(p, :tr_dr_1, 1)
        @test PO.state_set!(model, p, :tr_dr_1, 1, :first) == :first
        @test_throws ArgumentError PO.state_set!(model, p, :tr_dr_, 11, :second)
        @test model[:tr_dr_11] == :first
        @test PO.state_set!(model, :a_, :y, 5) == 5
        @test_throws ArgumentError PO.state_set!(model, :a_, :y, 6)
        @test PO.state_has(model, :a_, :y) && PO.state_has(model, p, :tr_dr_1, 1)
        @test PO.state_get(model, :a_, :y) == 5
        @test PO.state_get(model, p, :tr_dr_1, 1) == :first
        @test_throws ArgumentError PO.state_get(model, :a_, :z)
        @test_throws ArgumentError PO.state_get(model, :a_, :z, 1)
        # `f` runs once for each key.
        calls = Ref(0)
        f = () -> (calls[] += 1; calls[])
        @test PO.state_build!(f, model, :b_, :x) == 1
        @test PO.state_build!(f, model, :b_, :x) == 1
        @test PO.state_build!(f, model, :b_, :x, 2) == 2
        @test PO.state_build!(f, model, :b_, :x, 2) == 2
        @test calls[] == 2
        # A flag holds `true`, and a second mark has no effect.
        PO.mark_state!(model, :c_, :flag)
        PO.mark_state!(model, :c_, :flag)
        PO.mark_state!(model, :c_, :flag, 3)
        @test model[:c_flag] === true && model[:c_flag3] === true
        @test PO.nested_prefix(:a_, :tr_dr_) == :a_tr_dr_
        @test PO.nested_prefix(:a_, :tr_dr_, 3) == :a_tr_dr_3_
        @test PO.nested_index(:loss_, 2) == :loss_2
        @test PO.nested_index(:gain_, PO.nested_index(:loss_, 2)) == :gain_loss_2
        # The shared verbs refuse a name off the list, and `shared_set!` replaces an entry.
        @test :X ∉ PO.SHARED_STATE
        @test_throws ArgumentError PO.shared_get(model, :X)
        @test_throws ArgumentError PO.shared_has(model, :X)
        @test_throws ArgumentError PO.shared_set!(model, :X, 1)
        @test_throws ArgumentError PO.shared_get(model, :fees)
        @test PO.shared_set!(model, :fee_fa, nothing) === nothing
        @test PO.shared_set!(model, :fee_fa, PO.AmortisedFees()) isa PO.AmortisedFees
        @test PO.shared_get(model, :fee_fa) isa PO.AmortisedFees
    end

    @testset "frontier rows: the floor, the polarity and the count" begin
        model = JuMP.Model()
        PO.set_model_scales!(model, 2.0, 1.0)
        PO.set_maximum_ratio_factor_variables!(model, MinimumRisk())
        JuMP.@variable(model, x)
        @test PO.frontier_sweep_points(model) == (big(1), Pair{Symbol, Int}[])
        @test isnothing(PO.assert_frontier_sweep_cap(model))
        @test PO.frontier_point_count(Frontier(; N = 4)) == 4
        @test PO.frontier_point_count([1.0, 2.0]) == 2
        risk = [(:a_var, :a) => (1.0 * x, [1.0, 2.0], true, 0),
                (:b_var, :b) => (1.0 * x, [3.0], false, 0)]
        rax = PO.set_risk_frontier_parameters!(model, risk)
        # A `true` flag is the ceiling `x / m <= u k`, a `false` flag the floor `x / m >= u k`,
        # and `s_c = 2` multiplies both. `m` is the largest absolute point of the entry: 2 for
        # `a` and 3 for `b`. The expression takes the division, so the parameter keeps `s_c`.
        ca = JuMP.constraint_object(model[:a])
        cb = JuMP.constraint_object(model[:b])
        @test ca.set == JuMP.MOI.LessThan(0.0) && cb.set == JuMP.MOI.LessThan(0.0)
        @test JuMP.coefficient(ca.func, x) == 1.0
        @test JuMP.coefficient(ca.func, model[:a_var]) == -2.0
        @test JuMP.coefficient(cb.func, x) ≈ -2 / 3
        @test JuMP.coefficient(cb.func, model[:b_var]) == 2.0
        ret = [(:r_var, :r) => (3.0 * x, [0.5, 0.7, 0.9], 1)]
        tax = PO.set_ret_frontier_parameters!(model, ret)
        cr = JuMP.constraint_object(model[:r])
        @test cr.set == JuMP.MOI.GreaterThan(0.0)
        @test JuMP.coefficient(cr.func, x) ≈ 6 / 0.9
        @test JuMP.coefficient(cr.func, model[:r_var]) == -2.0
        # The product: two risk points, one risk point and three return points.
        pts = collect(PO.frontier_sweep_axes(tax, rax))
        @test length(pts) == 6
        @test length(collect(PO.frontier_sweep_axes(tax, nothing))) == 3
        @test length(collect(PO.frontier_sweep_axes(nothing, rax))) == 2
        @test length(collect(PO.frontier_sweep_axes(nothing, nothing))) == 1
        # The risk axis changes fastest. Each parameter holds its point in units of `m`.
        PO.set_frontier_point!(model, pts[2])
        @test JuMP.parameter_value(model[:a_var]) == 2.0 / 2
        @test JuMP.parameter_value(model[:r_var]) == 0.5 / 0.9
        PO.set_frontier_point!(model, pts[3])
        @test JuMP.parameter_value(model[:a_var]) == 1.0 / 2
        @test JuMP.parameter_value(model[:r_var]) == 0.7 / 0.9
        @test JuMP.parameter_value(model[:b_var]) == 3.0 / 3
    end

    @testset "scalar bound rows: the row is divided by the size of its bound" begin
        model = JuMP.Model()
        PO.set_model_scales!(model, 2.0, 1.0)
        PO.set_maximum_ratio_factor_variables!(model, MinimumRisk())
        JuMP.@variable(model, x)
        mr = MeanRisk(; opt = JuMPOptimiser(; slv = Solver(; solver = Clarabel.Optimizer)))
        # A ceiling of the size of a fourth moment: `s_c (x - u) / |u| <= 0`, so both terms
        # are of order one. JuMP moves the constant into the set: `1e7 x <= 2`.
        PO.set_risk_upper_bound!(model, mr, 1.0 * x, 2e-7, :kt_risk_1)
        cu = JuMP.constraint_object(PO.state_get(model, :kt_risk_1, :_ub))
        @test cu.set isa JuMP.MOI.LessThan
        @test cu.set.upper ≈ 2
        @test JuMP.coefficient(cu.func, x) ≈ 2 / 2e-7
        # A `false` flag turns the row into a floor, and a negative bound keeps the sense:
        # `-2 (x + 0.5) / 0.5 <= 0`, which is `-4 x <= 2`.
        PO.set_risk_upper_bound!(model, mr, 1.0 * x, -0.5, :sk_risk_1, false)
        cl = JuMP.constraint_object(PO.state_get(model, :sk_risk_1, :_ub))
        @test JuMP.coefficient(cl.func, x) ≈ -4
        @test cl.set.upper ≈ 2
        # A zero bound takes the scale one.
        PO.set_risk_upper_bound!(model, mr, 1.0 * x, 0.0, :zr_risk_1)
        cz = JuMP.constraint_object(PO.state_get(model, :zr_risk_1, :_ub))
        @test JuMP.coefficient(cz.func, x) == 2
        # A return floor: `s_c (ret - l) / |l| >= 0`, which is `12 x >= 2`.
        PO.set_return_bounds!(model, 1, 3.0 * x, 0.5)
        cr = JuMP.constraint_object(PO.state_get(model, Symbol(""), :ret_lb_, 1))
        @test cr.set isa JuMP.MOI.GreaterThan
        @test cr.set.lower ≈ 2
        @test JuMP.coefficient(cr.func, x) ≈ 12
    end

    @testset "weights, start values and the return series" begin
        X = [0.01 0.02; -0.01 0.03; 0.02 -0.02; 0.0 0.01]
        model = JuMP.Model()
        PO.set_w!(model, X, [0.25, 0.75])
        w = PO.get_w(model)
        @test length(w) == 2
        @test JuMP.start_value.(w) == [0.25, 0.75]
        @test !JuMP.has_lower_bound(w[1]) && !JuMP.has_upper_bound(w[1])
        @test_throws DimensionMismatch PO.set_initial_w!(w, [1.0])
        # A vector of weight vectors, as a sweep of anchors passes, sets no start value.
        model2 = JuMP.Model()
        PO.set_w!(model2, X, [[0.5, 0.5], [0.2, 0.8]])
        @test all(isnothing, JuMP.start_value.(PO.get_w(model2)))
        Xw = PO.set_portfolio_returns!(model, X)
        @test PO.set_portfolio_returns!(model, X) === Xw
        @test PO.state_get(model, Symbol(""), :X) === Xw
        wv = [0.6, 0.4]
        val(e) = JuMP.value(v -> wv[JuMP.index(v).value], e)
        @test val.(Xw) ≈ X * wv
        net = PO.set_net_portfolio_returns!(model, X)
        # With no fees the net series is the gross one.
        @test val.(net) ≈ X * wv
        @test PO.has_net_X(model) && PO.get_net_X(model) === net
    end

    @testset "the net series charges the one-off fee on its clock" begin
        X = [0.01 0.02; -0.01 0.03; 0.02 -0.02; 0.0 0.01]
        wv = [0.6, 0.4]
        function net_series(fa)
            model = JuMP.Model()
            PO.set_model_scales!(model, 1.0, 1.0)
            PO.set_model_observations!(model, 4)
            JuMP.@variable(model, w[1:2])
            PO.add_to_fees!(model, 0.001 * w[1])
            PO.add_to_one_time_fees!(model, 0.004 * w[2])
            PO.shared_set!(model, :fee_fa, fa)
            net = PO.set_net_portfolio_returns!(model, X)
            @test PO.set_net_portfolio_returns!(model, X) === net
            return [JuMP.value(v -> wv[JuMP.index(v).value], e) for e in net]
        end
        gross = X * wv .- 0.001 * 0.6
        one_off = 0.004 * 0.4
        # The first observation pays the whole one-off fee on the default clock.
        @test net_series(nothing) ≈ gross .- one_off .* [1, 0, 0, 0]
        @test net_series(PO.FirstObservationFees()) ≈ gross .- one_off .* [1, 0, 0, 0]
        # Each of the four observations pays a quarter on the amortised clock.
        @test net_series(PO.AmortisedFees()) ≈ gross .- one_off / 4
    end

    @testset "the plus-one matrices" begin
        X = [0.01 0.02; -0.03 0.03; 0.02 -0.02]
        model = JuMP.Model()
        a = PO.set_asset_returns_plus_one!(model, X)
        @test a == X .+ 1
        @test PO.set_asset_returns_plus_one!(model, X) === a
        @test PO.state_get(model, Symbol(""), :Xap1) === a
        d = PO.set_portfolio_drawdowns_plus_one!(model, X)
        # The drawdown of each column of the cumulative sum, the peak seeded at zero.
        c = cumsum(X; dims = 1)
        @test d ≈ 1 .+ c .- max.(0, accumulate(max, c; dims = 1))
        @test d ≈ [1.0 1.0; 0.97 1.0; 0.99 0.98]
        @test all(<=(1), d)
        @test PO.set_portfolio_drawdowns_plus_one!(model, X) === d
        @test PO.state_get(model, Symbol(""), :ddap1) === d
        # The prefix keeps a second build apart.
        @test PO.set_asset_returns_plus_one!(model, -X; prefix = :gain_) == 1 .- X
        @test PO.state_get(model, :gain_, :Xap1) == 1 .- X
        @test PO.state_get(model, Symbol(""), :Xap1) === a
    end

    @testset "the drawdown builder registers its variables" begin
        X = [0.01 0.02; -0.03 0.03; 0.02 -0.02]
        model = JuMP.Model()
        PO.set_model_scales!(model, 1.0, 1.0)
        PO.set_w!(model, X, nothing)
        @test !PO.state_has(model, Symbol(""), :dd)
        dd = PO.set_drawdown_constraints!(model, X)
        @test PO.state_get(model, Symbol(""), :dd) === dd
        # One more variable than observations.
        @test length(dd) == size(X, 1) + 1
    end

    X = 0.01 * randn(StableRNG(42), 120, 5) .+ 0.001 * (1:5)'
    rd = ReturnsResult(; nx = ["A", "B", "C", "D", "E"], X = X)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = "verbose" => false)
    bad = Solver(; name = :bad, solver = () -> error("no solver"),
                 check_sol = (; allow_local = true, allow_almost = true))

    @testset "the solver loop falls back, and the weights are divided by k" begin
        res = optimise(MeanRisk(; r = Variance(), obj = MaximumRatio(),
                                opt = JuMPOptimiser(; slv = [bad, slv])), rd)
        @test res.retcode isa OptimisationSuccess
        # Only the solver that failed has an entry.
        @test collect(keys(res.retcode.res)) == [:bad]
        k = JuMP.value(res.model[:k])
        @test k > 0 && !isapprox(k, 1)
        @test res.w ≈ JuMP.value.(res.model[:w]) / k
        @test sum(res.w) ≈ 1
        # The default `getproperty` forwards `w` and the fields of `pa` to `jr`.
        @test res.w === getfield(res, :jr).w
        @test res.r === getfield(res, :r)
        @test :w in propertynames(res) && :pa in propertynames(res)
        # Every solver fails, so the weights are `NaN`.
        all_bad = MeanRisk(; r = Variance(), opt = JuMPOptimiser(; slv = bad))
        failed = @test_logs (:warn, r"bad: set_optimizer → no solver") match_mode = :any optimise(all_bad,
                                                                                                  rd)
        @test failed.retcode isa OptimisationFailure
        @test length(failed.w) == 5 && all(isnan, failed.w)
    end

    @testset "a solver that raises in optimize! is recorded, and the next one solves" begin
        # HiGHS takes no conic row without bridges, so `optimize!` raises.
        hs = Solver(; name = :highs, solver = HiGHS.Optimizer, add_bridges = false,
                    settings = "log_to_console" => false,
                    check_sol = (; allow_local = true, allow_almost = true))
        res = optimise(MeanRisk(; r = Variance(), opt = JuMPOptimiser(; slv = [hs, slv])),
                       rd)
        @test res.retcode isa OptimisationSuccess
        @test collect(keys(res.retcode.res)) == [:highs]
        @test collect(keys(res.retcode.res[:highs])) == [:optimize!]
    end

    @testset "a reset removes every time-dependent field" begin
        mr = MeanRisk(; r = Variance(),
                      obj = TimeDependent([MinimumRisk(), MaximumRatio()]),
                      opt = JuMPOptimiser(; slv = slv))
        @test PO.is_time_dependent(mr)
        reset = PO.reset_time_dependent_estimator(mr)
        @test !PO.is_time_dependent(reset)
        @test reset.obj isa PO.ObjectiveFunction
        # A static optimiser comes back unchanged.
        @test PO.reset_time_dependent_estimator(reset) === reset
    end

    @testset "a return frontier sweeps its own points" begin
        ret = ArithmeticReturn(;
                               settings = PO.JuMPReturnsSettings(; lb = Frontier(; N = 3)))
        res = optimise(MeanRisk(; r = Variance(), obj = MinimumRisk(),
                                opt = JuMPOptimiser(; slv = slv, ret = ret)), rd)
        @test length(res.w) == 3
        mu = vec(sum(X; dims = 1)) / size(X, 1)
        rets = [dot(mu, w) for w in res.w]
        @test issorted(rets)
        # The last point binds its floor. The parameter holds the floor in units of the fixed
        # scale `m` of the row `s_c (ret / m - b) >= 0`, so the weight `w_1` of the row has the
        # coefficient `s_c mu_1 / m`.
        sc = PO.get_constraint_scale(res.model)
        @test JuMP.normalized_coefficient(res.model[:ret_lb_1], res.model[:ret_lb_var_1]) ==
              -sc
        m = sc * mu[1] / JuMP.normalized_coefficient(res.model[:ret_lb_1], res.model[:w][1])
        lb = m * JuMP.parameter_value(res.model[:ret_lb_var_1])
        @test rets[end] >= lb - 1e-8
        @test isapprox(rets[end], lb; rtol = 1e-6)
    end
end

# The docstrings of `src/17_Optimisation/01_Base_Optimisation/` checked against numbers: the
# type families and their fallbacks, the routing of Pipeline targets, the fold schedules, the
# weight finalisers, the panel collapse and the investable view of a result. The finaliser
# testsets solve three-asset programmes with Clarabel; the others build no model.
using Test, PortfolioOptimisers, Clarabel, LinearAlgebra
using InteractiveUtils: subtypes

# A result with no prior, no fees and no mask, which reaches every fallback of the family.
struct BareResult03e <: PortfolioOptimisers.OptimisationResult
    w::Vector{Float64}
end
# A keyword constructor that records each call, for the substitution check.
const SUB03E_LOG = Any[]
struct Sub03e
    a::Any
    b::Any
end
function Sub03e(; a, b)
    push!(SUB03E_LOG, (a, b))
    return Sub03e(a, b)
end

@testset "Base optimisation against its docstrings" begin
    PO = PortfolioOptimisers
    # The concrete subtypes of `T` that the package defines. Another test file in the same
    # process can define subtypes of its own, which `subtypes` also returns.
    function leaves(T)
        out = Any[]
        for S in subtypes(T)
            if isabstracttype(S)
                append!(out, leaves(S))
            elseif parentmodule(S) === PO
                push!(out, S)
            end
        end
        return out
    end
    rd = ReturnsResult(; nx = ["A", "B", "C"],
                       X = [0.01 0.02 -0.01; 0.0 0.01 0.02; -0.01 0.0 0.01; 0.02 -0.01 0.0])
    slv = Solver(; solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false, "tol_gap_abs" => 1e-10,
                                 "tol_gap_rel" => 1e-10, "tol_feas" => 1e-10))

    @testset "the result families have the stated shape" begin
        # `fb` is the last field of every concrete result, and `factory(res, fb)` needs it.
        for R in leaves(PO.OptimisationResult)
            @test last(fieldnames(R)) === :fb
        end
        # A return code has the one field `res`.
        for R in leaves(PO.OptimisationReturnCode)
            @test fieldnames(R) == (:res,)
        end
        # The hierarchical family holds the three results of a `HierarchicalOptimiser`, and
        # `NestedClustered` is one of the four clustering estimators but not a member.
        @test Set(leaves(PO.HierarchicalOptimisationResult)) ==
              Set([HierarchicalRiskParityResult, HierarchicalEqualRiskContributionResult,
                   SchurComplementHierarchicalRiskParityResult])
        @test length(leaves(PO.ClusteringOptimisationEstimator)) == 4
        @test NestedClustered in leaves(PO.ClusteringOptimisationEstimator)
        @test JuMPOptimiser <: PO.BaseOptimisationEstimator
        @test PO.HierarchicalOptimiser <: PO.BaseOptimisationEstimator
    end

    @testset "the fallbacks of the family" begin
        res = BareResult03e([0.5, 0.5])
        @test isnothing(PO.result_investable_mask(res))
        # One fallback of the reset serves every estimator and every result.
        @test PO.reset_time_dependent_estimator(res) === res
        ga = GreedyAllocation()
        @test PO.reset_time_dependent_estimator(ga) === ga
        # A result with no method of `set_retcode` throws, and names its type.
        nres = optimise(EqualWeighted(), rd)
        err = try
            PO.set_retcode(nres, OptimisationFailure())
        catch e
            e
        end
        @test isa(err, ArgumentError)
        @test occursin("NaiveOptimisationResult", err.msg)
        # A precomputed result in `fb` keeps its identity when `factory` gives the estimator
        # the previous weights.
        @test factory(nres, [0.2, 0.3, 0.5]) === nres
        mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv), fb = nres)
        @test factory(mr, [0.2, 0.3, 0.5]).fb === nres
        # `factory(res, fb)` replaces the last field and keeps the others.
        chain = [(EqualWeighted(), nres)]
        nres2 = factory(nres, chain)
        @test nres2.fb === chain
        @test nres2.w == nres.w
        # `needs_previous_weights` falls back to `false` for nothing, an estimator and an
        # algorithm.
        @test !PO.needs_previous_weights(nothing)
        @test !PO.needs_previous_weights(EqualWeighted())
        @test !PO.needs_previous_weights(nres)
        @test isnothing(PO.assert_special_nco_requirements(nres))
    end

    @testset "the routing of Pipeline targets" begin
        rb = RiskBudgeting(; opt = JuMPOptimiser(; slv = slv))
        @test PO.pipe_config_field(rb) === :opt
        @test isnothing(PO.pipe_config_field(EqualWeighted()))
        # `:rkb` lands one level down, in `rba.rkb`.
        rkb = PO.risk_budget_constraints(nothing; N = 3)
        @test PO.pipe_accepts(rb, Val(:rkb))
        @test PO.pipe_route(rb, Val(:rkb), rkb).rba.rkb === rkb
        # A schedule in `rba` has no `rkb`, so the optimiser refuses the target.
        rbt = RiskBudgeting(; opt = JuMPOptimiser(; slv = slv),
                            rba = TimeDependent([AssetRiskBudgeting(),
                                                 AssetRiskBudgeting()]))
        @test !PO.pipe_accepts(rbt, Val(:rkb))
        @test_throws ArgumentError PO.pipe_route(rbt, Val(:rkb), rkb)
        # A target with the name of a field lands in that field.
        @test PO.pipe_accepts(EqualWeighted(), Val(:wb))
        wb = WeightBounds(; lb = 0.0, ub = 0.6)
        @test PO.pipe_route(EqualWeighted(), Val(:wb), wb).wb === wb
    end

    @testset "a schedule resolves as its docstring states" begin
        ctx = TimeDependentContext(; i = 2, n = 3, rd = rd, train_idx = [1:2, 1:3, 1:4],
                                   test_idx = [3:3, 4:4, 4:4])
        # A vector reads entry `ctx.i`, a function and a wrapped function read the context.
        @test PO.time_dependent_value(TimeDependent([10, 20, 30]), ctx) == 20
        @test PO.time_dependent_value(TimeDependent(c -> 5 * c.i), ctx) == 10
        pwf = PreviousWeightsFunction(; f = c -> c.n)
        @test PO.time_dependent_value(TimeDependent(pwf), ctx) == 3
        @test PO.needs_previous_weights(pwf)
        @test PO.needs_previous_weights(TimeDependent(pwf))
        @test !PO.needs_previous_weights(TimeDependent(c -> 1))
        @test_throws DomainError TimeDependentContext(; i = 4, n = 3, rd = rd,
                                                      train_idx = [], test_idx = [])
        # The constructor refuses the stated values.
        @test_throws IsEmptyError TimeDependent([])
        @test_throws ArgumentError TimeDependent(TimeDependent([1, 2]))
        @test_throws ArgumentError TimeDependent(TimeDependent([1, 2]), :nearest)
        # The refusal of a nested schedule is no method of its own, so no call is ambiguous
        # with the positional constructor that `@concrete` writes.
        @test !any(p -> p[1].name === :TimeDependent || p[2].name === :TimeDependent,
                   Test.detect_ambiguities(PortfolioOptimisers))
        @test_throws ArgumentError TimeDependent([TimeDependent([1, 2]), 3])
        @test_throws ArgumentError TimeDependent([1, 2]; default = TimeDependent([1, 2]))
        @test_throws ArgumentError TimeDependent([1, 2], :innermost)
        # A vector of optimisers and results is stored as a `Vector{OptE_Opt}`.
        nres = optimise(EqualWeighted(), rd)
        mixed = TimeDependent(Any[EqualWeighted(), nres])
        @test eltype(mixed.val) == PO.OptE_Opt
        @test isa(mixed, PO.TD_OptE_Opt)
        # The slice of a schedule slices its entries and its default, and a function passes.
        td = TimeDependent([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]; default = [7.0, 8.0, 9.0])
        tv = PO.nothing_scalar_array_view(td, [1, 3])
        @test tv.val == [[1.0, 3.0], [4.0, 6.0]]
        @test tv.default == [7.0, 9.0]
        f = c -> 1
        @test PO.nothing_scalar_array_view(TimeDependent(f), [1]).val === f
        # `factory` reaches the entries and the default.
        tf = factory(TimeDependent([nres, nres]; default = nres), [0.2, 0.3, 0.5])
        @test all(x -> x === nres, tf.val)
        @test tf.default === nres
        # The known entries are the vector and the default, and a function has none.
        @test length(PO.time_dependent_entries(TimeDependent([1, 2]; default = 3))) == 3
        @test isempty(PO.time_dependent_entries(TimeDependent(f)))
    end

    @testset "the value outside every fold loop, and the stand-in" begin
        # The reset value: the default of the schedule, then the static default, then
        # `nothing` for a field that the defaults do not list.
        defaults = (; a = 5, b = NoDefault())
        @test PO.time_dependent_reset_value(TimeDependent([1, 2]; default = 9), defaults,
                                            :b, EqualWeighted()) == 9
        @test PO.time_dependent_reset_value(TimeDependent([1, 2]), defaults, :a,
                                            EqualWeighted()) == 5
        @test isnothing(PO.time_dependent_reset_value(TimeDependent([1, 2]), defaults, :c,
                                                      EqualWeighted()))
        @test_throws TimeDependentDefaultError PO.time_dependent_reset_value(TimeDependent([1,
                                                                                            2]),
                                                                             defaults, :b,
                                                                             EqualWeighted())
        # The stand-in: the default, the static default, the first entry, else nothing.
        @test something(PO.time_dependent_stand_in(TimeDependent([1, 2]; default = 9),
                                                   defaults, :b)) == 9
        @test something(PO.time_dependent_stand_in(TimeDependent([1, 2]), defaults, :a)) ==
              5
        @test something(PO.time_dependent_stand_in(TimeDependent([1, 2]), defaults, :b)) ==
              1
        @test isnothing(PO.time_dependent_stand_in(TimeDependent(c -> 1), defaults, :b))
        # The static defaults fall back to an empty tuple.
        @test PO.time_dependent_field_defaults(BareResult03e([1.0])) == (;)
        # A schedule in the place of an optimiser must carry a default.
        @test_throws TimeDependentDefaultError PO.reset_time_dependent_estimator(TimeDependent([EqualWeighted(),
                                                                                                EqualWeighted()]))
        ew = EqualWeighted()
        @test PO.reset_time_dependent_estimator(TimeDependent([ew, ew]; default = ew)) ===
              ew
    end

    @testset "the substitution check calls the constructor once for each entry" begin
        empty!(SUB03E_LOG)
        args = (; a = TimeDependent([1, 2]; default = 3), b = TimeDependent([4, 5]))
        @test isnothing(PO.assert_time_dependent_substitution(Sub03e, args, (; b = 6)))
        # `a` runs its two entries and its default with `b` at its static default 6, then
        # `b` runs its two entries with `a` at its default 3.
        @test SUB03E_LOG == [(1, 6), (2, 6), (3, 6), (3, 4), (3, 5)]
        # A function schedule in a required field has no stand-in, so nothing is called.
        empty!(SUB03E_LOG)
        args = (; a = TimeDependent(c -> 1), b = TimeDependent([4, 5]))
        @test isnothing(PO.assert_time_dependent_substitution(Sub03e, args,
                                                              (; a = NoDefault())))
        @test isempty(SUB03E_LOG)
        @test isnothing(PO.substitute_time_dependent_entries(Sub03e, (; a = 1), :a, 1))
        @test isempty(SUB03E_LOG)
    end

    @testset "the fold count and the bind checks" begin
        ew = EqualWeighted()
        td = TimeDependent([ew, ew])
        @test isnothing(PO.assert_time_dependent_fold_count(td, 2))
        @test_throws DimensionMismatch PO.assert_time_dependent_fold_count(td, 3)
        # A schedule with `:nearest` is left for the nearest loop when `all_binds` is false.
        tdn = TimeDependent([ew, ew], :nearest)
        @test isnothing(PO.assert_time_dependent_fold_count(tdn, 3, false))
        @test_throws DimensionMismatch PO.assert_time_dependent_fold_count(tdn, 3, true)
        ctx = TimeDependentContext(; i = 1, n = 2, rd = rd, train_idx = [1:2, 1:3],
                                   test_idx = [3:3, 4:4])
        @test PO.update_time_dependent_estimator(tdn, ctx, false) === tdn
        @test PO.update_time_dependent_estimator(tdn, ctx, true) === ew
        # A function in the place of an optimiser that returns no optimiser is refused.
        @test_throws ArgumentError PO.update_time_dependent_estimator(TimeDependent(c -> 1),
                                                                      ctx)
        # The guards of `:nearest` in a field with no inner fold loop, and in one with.
        @test_throws ArgumentError PO.assert_no_nearest_bind_optimiser_schedule(tdn, :fb,
                                                                                :MeanRisk)
        @test isnothing(PO.assert_no_nearest_bind_optimiser_schedule(td, :fb, :MeanRisk))
        @test_throws TimeDependentDefaultError PO.assert_nearest_optimiser_schedule(tdn,
                                                                                    :opti,
                                                                                    KFold(),
                                                                                    :NestedClustered)
        tdd = TimeDependent([ew, ew], :nearest; default = ew)
        @test_throws ArgumentError PO.assert_nearest_optimiser_schedule(tdd, :opti, nothing,
                                                                        :NestedClustered)
        @test isnothing(PO.assert_nearest_optimiser_schedule(tdd, :opti, KFold(),
                                                             :NestedClustered))
        # A configuration with no schedule has no candidate field.
        @test PO.time_dependent_candidate_fields(JuMPOptimiser(; slv = slv)) == ()
    end

    @testset "the JuMP formulations give their hand minimisers" begin
        # w0 = [0.7, 0.2, 0.1] under ub = 0.5 moves a mass of 0.2 to the two free weights.
        wb = WeightBounds(; lb = 0.0, ub = 0.5)
        w0 = [0.7, 0.2, 0.1]
        fin(alg) = PO.opt_weight_bounds(JuMPWeightFinaliser(; slv = slv, alg = alg), wb, w0)
        abs_dev = AbsoluteWeightDeviation()
        # The L2 norm of the absolute deviation shifts the free weights by one constant.
        @test isapprox(fin(ErrorWeightFinaliser(; alg = L2Norm(), dev = abs_dev)),
                       [0.5, 0.3, 0.2]; atol = 1e-6)
        # The L1 norm of the relative deviation moves the mass to the largest free weight.
        @test isapprox(fin(ErrorWeightFinaliser()), [0.5, 0.4, 0.1]; atol = 1e-6)
        # The L2 norm of the relative deviation gives w0 + c w0², with c = 4.
        @test isapprox(fin(ErrorWeightFinaliser(; alg = L2Norm())), [0.5, 0.36, 0.14];
                       atol = 1e-6)
        # The L1 norm of the absolute deviation is not unique, and its value is twice the
        # moved mass.
        wa = fin(ErrorWeightFinaliser(; dev = abs_dev))
        @test isapprox(sum(abs, wa - w0), 0.4; atol = 1e-6)
        @test isapprox(sum(wa), 1.0; atol = 1e-8)
        @test all(x -> -1e-8 <= x <= 0.5 + 1e-8, wa)
        # The Euclidean projection is the minimiser of the absolute L2 norm.
        @test isapprox(PO.opt_weight_bounds(EuclideanWeightFinaliser(), wb, w0),
                       [0.5, 0.3, 0.2]; atol = 1e-12)
        # A relative deviation does not change the weights of the caller.
        wz = [0.7, 0.3, 0.0]
        PO.opt_weight_bounds(JuMPWeightFinaliser(; slv = slv,
                                                 alg = ErrorWeightFinaliser(;
                                                                            alg = L2Norm())),
                             wb, wz)
        @test wz == [0.7, 0.3, 0.0]
        # A p-norm needs 1 < p for its power cone, so the constructor refuses p = 1.
        @test_throws DomainError ErrorWeightFinaliser(; alg = LpNorm(; p = 1))
        # Weights in the bounds come back unchanged, with no solve.
        wi = [0.4, 0.4, 0.2]
        @test PO.opt_weight_bounds(JuMPWeightFinaliser(; slv = slv), wb, wi) === wi
    end

    @testset "a failed solve and the iterative finaliser" begin
        wb = WeightBounds(; lb = 0.0, ub = 0.4)
        # The iterative loop stalls on this input, and step 9 returns the projection.
        @test PO.opt_weight_bounds(IterativeWeightFinaliser(), wb, [0.6, 0.4, 0.0]) ≈
              [0.4, 0.4, 0.2]
        # A solver that cannot start makes the JuMP finaliser warn and use the iterative one.
        bad = JuMPWeightFinaliser(; slv = Solver(; solver = () -> error("no solver")))
        w = @test_logs((:warn, r"IterativeWeightFinaliser"), match_mode = :any,
                       PO.opt_weight_bounds(bad, wb, [0.6, 0.4, 0.0]))
        @test w ≈ [0.4, 0.4, 0.2]
        # A set of bounds that cannot hold the budget fails.
        rc, w = PO.finalise_weight_bounds(IterativeWeightFinaliser(),
                                          WeightBounds(; lb = 0.3, ub = 1.0), fill(0.25, 4))
        @test isa(rc, OptimisationFailure)
        @test w ≈ fill(0.3, 4)
        rc, w = PO.finalise_weight_bounds(IterativeWeightFinaliser(), wb, [0.6, 0.4, 0.0])
        @test isa(rc, OptimisationSuccess)
        @test sum(w) ≈ 1.0
    end

    @testset "the collapse of the panel" begin
        # The normalised weights are |W| over the gross exposure of each column.
        @test PO.synthetic_asset_weights([0.5, -0.25, 0.25]) ≈ [0.5, 0.25, 0.25]
        Wn = PO.synthetic_asset_weights([0.5 0.0; -0.5 0.0; 1.0 0.0])
        @test Wn ≈ [0.25 0.0; 0.25 0.0; 0.5 0.0]
        @test PO.synthetic_asset_weights(zeros(3)) == zeros(3)
        W = PO.synthetic_asset_weights([0.5 0.0; 0.5 1.0; 0.0 0.0])
        @test PO.collapse_panel_numeric([1.0, 2.0, 4.0], W) ≈ [1.5, 2.0]
        # A mask collapses as its support, and a categorical mask repeats over the levels.
        @test PO.collapse_categorical_mask(BitVector([true, false, false]), W, 3,
                                           nothing) == Bool[1 1 1; 0 0 0]
        om = BitMatrix([true false false; false true false])
        cm = PO.collapse_categorical_mask(om, W, 2, trues(2, 3))
        @test size(cm) == (2, 2, 2)
        @test cm[:, :, 1] == Bool[1 0; 1 1]
        @test cm[:, :, 2] == cm[:, :, 1]
        # An observed member that is inactive does not count.
        cmi = PO.collapse_categorical_mask(om, W, 2,
                                           BitMatrix([true true true; true false true]))
        @test cmi[:, :, 1] == Bool[1 0; 0 0]
        @test isnothing(PO.collapse_categorical_mask(nothing, W, 2, nothing))
    end

    @testset "the investable view of a result" begin
        nres = optimise(EqualWeighted(), rd)
        # An explicit prior has priority, and a result with no prior throws.
        @test PO.extract_pr(nres, nres.pr) === nres.pr
        @test PO.extract_pr(nres) === nres.pr
        @test_throws ArgumentError PO.extract_pr(BareResult03e([1.0]))
        @test isnothing(PO.extract_fees(BareResult03e([1.0])))
        fees = Fees(; l = 0.001)
        @test PO.extract_fees(BareResult03e([1.0]), fees) === fees
        # A result whose mask is nothing views nothing.
        imsk, w, X, f, nx = PO.result_investable_view(BareResult03e([0.5, 0.5]),
                                                      [1.0 2.0; 3.0 4.0], nothing,
                                                      ["a", "b"])
        @test isnothing(imsk)
        @test w == [0.5, 0.5]
        @test X == [1.0 2.0; 3.0 4.0]
        @test nx == ["a", "b"]
    end
end
@testset "JuMP optimiser against its docstrings" begin
    using PortfolioOptimisers, Test, JuMP, HiGHS, StableRNGs
    PO = PortfolioOptimisers
    slv = Solver(; solver = nothing)
    N = 5
    nx = ["A", "B", "C", "D", "E"]
    rng = StableRNG(685)
    X = randn(rng, 200, N) .* 0.01 .+ [0.001 0.0005 -0.0003 0.0008 -0.0006]
    rd = ReturnsResult(; nx = nx, X = X)

    @testset "the constructor raises as its Validation list states" begin
        @test_throws IsEmptyError JuMPOptimiser(; slv = Solver[])
        @test_throws IsNonFiniteError JuMPOptimiser(; slv, bgt = Inf)
        @test_throws ConflictingArgumentError JuMPOptimiser(; slv,
                                                            bgt = BudgetCosts(;
                                                                              w = fill(1 /
                                                                                       3,
                                                                                       3)),
                                                            sbgt = 0.2)
        @test_throws DomainError JuMPOptimiser(; slv, sbgt = -0.1)
        @test_throws DomainError JuMPOptimiser(; slv, bgt = nothing, gbgt = -1.0,
                                               wb = WeightBounds(; lb = -1, ub = 1))
        # A gross budget needs a negative bound, and it is over-determined by two numbers.
        @test_throws ArgumentError JuMPOptimiser(; slv, bgt = nothing, gbgt = 1.5)
        @test_throws ArgumentError JuMPOptimiser(; slv, bgt = 1.0, sbgt = 0.3, gbgt = 1.6,
                                                 wb = WeightBounds(; lb = -1, ub = 1))
        # Two numbers pin the gross budget of the model, not the realised gross exposure,
        # which is smaller under `xbgt = false` (#685, R14 of #1817).
        @test_throws "pin the gross budget at bgt + 2 * sbgt" JuMPOptimiser(; slv,
                                                                            bgt = 1.0,
                                                                            sbgt = 0.3,
                                                                            gbgt = 1.6,
                                                                            wb = WeightBounds(;
                                                                                              lb = -1,
                                                                                              ub = 1))
        for (k, v) in
            ((:cte, CentralityConstraint[]), (:tn, Turnover[]), (:tr, TrackingError[]),
             (:l2, L2Regularisation[]), (:lp, LpRegularisation[]),
             (:lpc, LpRegularisation[]))
            @test_throws IsEmptyError JuMPOptimiser(; slv, k => v)
        end
        for k in (:card, :l2c, :linfc, :l1, :linf)
            @test_throws DomainError JuMPOptimiser(; slv, k => (k === :card ? 0 : 0.0))
        end
        # A schedule passes the direct check, and the substitution checks each entry.
        @test JuMPOptimiser(; slv, l1 = TimeDependent([0.1, 0.2])) isa JuMPOptimiser
        @test_throws DomainError JuMPOptimiser(; slv, card = TimeDependent([2, 0]))
        # A slot keyed by name, a fee estimator among them, needs `sets`.
        sets = UniverseSets(; dict = Dict("nx" => nx))
        for kw in ((; wb = WeightBoundsEstimator(; lb = Dict("A" => 0.1))),
                   (; fees = FeesEstimator(; l = Dict("A" => 0.01))),
                   (; smtx = [AssetSetsMatrixEstimator(; val = "x")],
                    slt = [ThresholdEstimator(; val = Dict("A" => 0.1))]))
            @test_throws IsNothingError JuMPOptimiser(; slv, kw...)
            @test JuMPOptimiser(; slv, sets, kw...) isa JuMPOptimiser
        end
    end

    @testset "two numbers fix the gross budget, and xbgt makes the weights reach it" begin
        hs = Solver(; solver = HiGHS.Optimizer, settings = Dict("log_to_console" => false))
        g = map((false, true)) do xb
            res = optimise(MeanRisk(; r = ConditionalValueatRisk(), obj = MinimumRisk(),
                                    opt = JuMPOptimiser(; slv = hs,
                                                        wb = WeightBounds(; lb = -1.0,
                                                                          ub = 1.0),
                                                        bgt = 1.0, sbgt = 0.3, xbgt = xb)),
                           rd)
            @test res.retcode isa OptimisationSuccess
            @test isapprox(sum(res.w), 1.0; atol = 1e-6)
            # The short budget of the model is pinned in both cases.
            @test isapprox(JuMP.value(sum(res.model[:sw]) / res.model[:k]), 0.3;
                           atol = 1e-6)
            sum(abs, res.w)
        end
        # Without `xbgt` the realised gross exposure is below bgt + 2 * sbgt.
        @test g[1] < 1.6 - 1e-3
        @test isapprox(g[2], 1.6; atol = 1e-6)
    end

    @testset "the previous weights, factory and the view" begin
        tnv = Turnover(; w = fill(0.2, N), val = 0.1)
        o = JuMPOptimiser(; slv, tn = tnv)
        @test !PO.needs_previous_weights(JuMPOptimiser(; slv))
        @test PO.needs_previous_weights(o)
        # A schedule that reads the previous weights counts, a plain vector does not.
        @test PO.needs_previous_weights(JuMPOptimiser(; slv,
                                                      bgt = TimeDependent(PreviousWeightsFunction(;
                                                                                                  f = ctx -> 1.0))))
        @test !PO.needs_previous_weights(JuMPOptimiser(; slv,
                                                       bgt = TimeDependent([1.0, 0.9])))
        w1 = [0.6, 0.1, 0.1, 0.1, 0.1]
        f = PO.factory(o, w1)
        @test f.tn.w == w1
        @test f.tn.val == 0.1
        @test all(n -> n in (:tn, :fees, :tr, :ccnt, :cobj) ||
                       getfield(f, n) === getfield(o, n), fieldnames(JuMPOptimiser))

        lt5 = Threshold(; val = [0.01, 0.02, 0.03, 0.04, 0.05])
        st5 = Threshold(; val = [0.11, 0.12, 0.13, 0.14, 0.15])
        sm = [1 1 0 0 0; 0 0 1 1 1]
        idx = [2, 4, 5]
        # Distinct objects in the pairs view one by one.
        v = PO.port_opt_view(JuMPOptimiser(; slv, smtx = sm, sgmtx = sm, slt = lt5,
                                           sglt = nothing, sst = st5,
                                           sgst = Threshold(; val = fill(0.2, N)),
                                           tn = Turnover(; w = fill(0.2, N),
                                                         val = collect(0.1:0.1:0.5))), idx,
                             X)
        @test v.slt.val == [0.02, 0.04, 0.05]
        @test isnothing(v.sglt)
        @test v.sst.val == [0.12, 0.14, 0.15]
        @test v.sgst.val == fill(0.2, 3)
        @test v.smtx === v.sgmtx
        @test v.smtx == [1 0 0; 0 1 1]
        @test v.tn.val == [0.2, 0.4, 0.5]
        # One object in a pair stays one object.
        vs = PO.port_opt_view(JuMPOptimiser(; slv, smtx = sm, sgmtx = sm, slt = lt5,
                                            sglt = lt5, sst = st5, sgst = st5), idx, X)
        @test vs.slt === vs.sglt
        @test vs.sst === vs.sgst
        vm = PO.port_opt_view(JuMPOptimiser(; slv, smtx = sm, sgmtx = [1 0 1 0 1],
                                            slt = lt5, sglt = lt5), idx, X)
        @test vm.smtx == [1 0 0; 0 1 1]
        @test vm.sgmtx == [0 0 1]
        # After the view, the optimiser declares the departed names on its own sets, and it
        # refuses a name that is still on the asset axis.
        kept = UniverseSets(; dict = Dict("nx" => ["A", "B", "D", "E"]))
        nv = PO.non_investable_universe(JuMPOptimiser(; slv, sets = kept), ["C"])
        @test nv.sets.dict["ni"] == ["C"]
        @test_throws ArgumentError PO.non_investable_universe(JuMPOptimiser(; slv,
                                                                            sets = kept),
                                                              ["A"])
    end

    @testset "the result expands the weights once" begin
        pr = prior(EmpiricalPrior(),
                   ReturnsResult(; nx = ["a", "b", "c"],
                                 X = [0.1 -0.2 0.0; -0.1 0.2 0.01; 0.05 0.1 -0.02]))
        function attrs(imsk)
            return ProcessedJuMPOptimiserAttributes(; pr = pr, wb = nothing, lt = nothing,
                                                    st = nothing, lcsr = nothing,
                                                    ctr = nothing, gcardr = nothing,
                                                    sgcardr = nothing, smtx = nothing,
                                                    sgmtx = nothing, slt = nothing,
                                                    sst = nothing, sglt = nothing,
                                                    sgst = nothing, tn = nothing,
                                                    fees = nothing, plr = nothing,
                                                    ret = ArithmeticReturn(),
                                                    sca = SumScalariser(), imsk = imsk)
        end
        pa = attrs(BitVector([1, 0, 1]))
        sol = PO.JuMPOptimisationSolution(; w = [0.4, 0.6])
        res = JuMPOptimisationResult(; pa, retcode = OptimisationSuccess(), sol,
                                     model = nothing)
        @test res.w == [0.4, 0.0, 0.6]
        @test res.imsk == BitVector([1, 0, 1])
        resv = JuMPOptimisationResult(; pa, retcode = OptimisationSuccess(),
                                      sol = [sol,
                                             PO.JuMPOptimisationSolution(; w = [1.0, 0.0])],
                                      model = nothing)
        @test resv.w == [[0.4, 0.0, 0.6], [1.0, 0.0, 0.0]]
        @test JuMPOptimisationResult(; pa = attrs(nothing), retcode = OptimisationSuccess(),
                                     sol, model = nothing).w == [0.4, 0.6]
        # `set_retcode` keeps the expanded weights, and a second expansion raises.
        @test PO.set_retcode(res, OptimisationFailure()).w == res.w
        @test_throws DimensionMismatch JuMPOptimisationResult(; pa,
                                                              retcode = OptimisationSuccess(),
                                                              sol = res.sol,
                                                              model = nothing)
    end

    @testset "the view of a head declares the departed names on its own sets" begin
        sets = UniverseSets(; dict = Dict("nx" => nx))
        rb = RiskBudgeting(;
                           rba = AssetRiskBudgeting(; sets,
                                                    rkb = RiskBudgetEstimator(;
                                                                              val = Dict("A" =>
                                                                                             0.5))),
                           opt = JuMPOptimiser(; slv))
        imsk = BitVector([1, 1, 0, 1, 1])
        rbv, rdv = PO.investable_view(rb, rd, prior(EmpiricalPrior(), rd), imsk)
        @test rbv.rba.sets.dict["ni"] == ["C"]
        @test rbv.rba.sets.dict["nx"] == ["A", "B", "D", "E"]
        @test rdv.nx == ["A", "B", "D", "E"]
    end
end
