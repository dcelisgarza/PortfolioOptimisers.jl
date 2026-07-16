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
            @test_logs (:warn, r"Failed to solve") PortfolioOptimisers.optimise_JuMP_model!(testmodel(0.9),
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
        r = quintile(SignedL1UncertaintySet(; ep = 0.05, em = 0.05, sd = sd_hat); bgt = 0.0,
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
end
