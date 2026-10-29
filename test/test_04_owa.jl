@testset "OWA" begin
    using PortfolioOptimisers, CSV, DataFrames, Clarabel, Test, Random, Logging, JuMP,
          StableRNGs, LinearAlgebra
    # Scope warning suppression to this task. `Logging.disable_logging` mutates a
    # process-global level that never restores and leaks into other test files sharing
    # a reused worker (see ParallelTestRunner worker pool), randomly emptying their
    # `@test_logs` captures. `with_logger` is task-local and restores on exit.
    with_logger(SimpleLogger(stderr, Logging.Error)) do
        @testset "OWA l-moments" begin
            slv = Solver(; solver = Clarabel.Optimizer,
                         settings = Dict("max_step_fraction" => 0.75, "verbose" => false))
            owa_t = CSV.read(joinpath(@__DIR__, "./assets/OWA_l_moment_weights.csv.gz"),
                             DataFrame)
            owas = [NormalisedConstantRelativeRiskAversion(; g = 0.75),
                    NormalisedConstantRelativeRiskAversion(),
                    NormalisedConstantRelativeRiskAversion(; g = 0.25),
                    # MaximumEntropy needs a small objective scale to solve (#1834), and
                    # the default `so = nothing` gives it 1e-3 (#1907).
                    OWAJuMP(; alg = MaximumEntropy(), max_phi = 0.75, slv = slv),
                    OWAJuMP(; alg = MaximumEntropy(), slv = slv),
                    OWAJuMP(; alg = MaximumEntropy(), max_phi = 0.25, slv = slv),
                    OWAJuMP(; alg = MinimumSumSquares(), max_phi = 0.75, slv = slv),
                    OWAJuMP(; alg = MinimumSumSquares(), slv = slv),
                    OWAJuMP(; alg = MinimumSumSquares(), max_phi = 0.25, slv = slv),
                    OWAJuMP(; alg = MinimumSquaredDistance(), max_phi = 0.75, slv = slv),
                    OWAJuMP(; alg = MinimumSquaredDistance(), slv = slv),
                    OWAJuMP(; alg = MinimumSquaredDistance(), max_phi = 0.25, slv = slv)]
            owas_entr = [OWAJuMP(; alg = MaximumEntropy(; alg = ExponentialConeEntropy()),
                                 max_phi = 0.75, slv = slv),
                         OWAJuMP(; alg = MaximumEntropy(; alg = ExponentialConeEntropy()),
                                 slv = slv),
                         OWAJuMP(; alg = MaximumEntropy(; alg = ExponentialConeEntropy()),
                                 max_phi = 0.25, slv = slv)]
            owas_sum_sq_sqsoc = [OWAJuMP(;
                                         alg = MinimumSumSquares(;
                                                                 alg = SquaredSOCRiskExpr()),
                                         max_phi = 0.75, slv = slv),
                                 OWAJuMP(;
                                         alg = MinimumSumSquares(;
                                                                 alg = SquaredSOCRiskExpr()),
                                         slv = slv),
                                 OWAJuMP(;
                                         alg = MinimumSumSquares(;
                                                                 alg = SquaredSOCRiskExpr()),
                                         max_phi = 0.25, slv = slv)]
            owas_sum_sq_rsoc = [OWAJuMP(; alg = MinimumSumSquares(; alg = RSOCRiskExpr()),
                                        max_phi = 0.75, slv = slv),
                                OWAJuMP(; alg = MinimumSumSquares(; alg = RSOCRiskExpr()),
                                        slv = slv),
                                OWAJuMP(; alg = MinimumSumSquares(; alg = RSOCRiskExpr()),
                                        max_phi = 0.25, slv = slv)]
            owas_min_sq_dist_sqsoc = [OWAJuMP(;
                                              alg = MinimumSquaredDistance(;
                                                                           alg = SquaredSOCRiskExpr()),
                                              max_phi = 0.75, slv = slv),
                                      OWAJuMP(;
                                              alg = MinimumSquaredDistance(;
                                                                           alg = SquaredSOCRiskExpr()),
                                              slv = slv),
                                      OWAJuMP(;
                                              alg = MinimumSquaredDistance(;
                                                                           alg = SquaredSOCRiskExpr()),
                                              max_phi = 0.25, slv = slv)]
            owas_min_sq_dist_rsoc = [OWAJuMP(;
                                             alg = MinimumSquaredDistance(;
                                                                          alg = RSOCRiskExpr()),
                                             max_phi = 0.75, slv = slv),
                                     OWAJuMP(;
                                             alg = MinimumSquaredDistance(;
                                                                          alg = RSOCRiskExpr()),
                                             slv = slv),
                                     OWAJuMP(;
                                             alg = MinimumSquaredDistance(;
                                                                          alg = RSOCRiskExpr()),
                                             max_phi = 0.25, slv = slv)]
            for i in eachindex(owas)
                owa = owa_l_moment_crm(200, owas[i]; k = 5)
                if i in 4:6
                    # The entropy of the coefficients has one maximiser, so the two
                    # encodings agree to solver tolerance: 1.6e-6 and 4.8e-6 measured (#1834).
                    rtol = i in 4:5 ? 2e-5 : 1e-6
                    owa_entr = owa_l_moment_crm(200, owas_entr[i - 3]; k = 5)
                    res = isapprox(owa, owa_entr; rtol = rtol)
                    if !res
                        println("Fails on OWA l-moments (entropy) iteration $i")
                        find_tol(owa, owa_entr)
                    end
                    @test res
                elseif i in 7:9
                    owa_qsoc = owa_l_moment_crm(200, owas_sum_sq_sqsoc[i - 6]; k = 5)
                    owa_rsoc = owa_l_moment_crm(200, owas_sum_sq_rsoc[i - 6]; k = 5)
                    if i in 7:8
                        rtol = 5e-3
                    else
                        rtol = 1e-6
                    end
                    res = isapprox(owa, owa_qsoc; rtol = rtol)
                    if !res
                        println("Fails on OWA l-moments (sq soc) iteration $i")
                        find_tol(owa, owa_qsoc)
                    end
                    @test res
                    res = isapprox(owa, owa_rsoc; rtol = rtol)
                    if !res
                        println("Fails on OWA l-moments (rsoc) iteration $i")
                        find_tol(owa, owa_rsoc)
                    end
                    res = isapprox(owa_qsoc, owa_rsoc; rtol = rtol)
                    if !res
                        println("Fails on OWA l-moments (rsoc, qsoc) iteration $i")
                        find_tol(owa_qsoc, owa_rsoc)
                    end
                    @test res
                elseif i in 10:12
                    owa_qsoc = owa_l_moment_crm(200, owas_min_sq_dist_sqsoc[i - 9]; k = 5)
                    owa_rsoc = owa_l_moment_crm(200, owas_min_sq_dist_rsoc[i - 9]; k = 5)
                    if i == 10
                        rtol = 5e-2
                    elseif i == 11
                        rtol = 5e-3
                    else
                        rtol = 1e-6
                    end
                    res = isapprox(owa, owa_qsoc; rtol = rtol)
                    if !res
                        println("Fails on OWA l-moments (sq soc) iteration $i")
                        find_tol(owa, owa_qsoc)
                    end
                    @test res
                    res = isapprox(owa, owa_rsoc; rtol = rtol)
                    if !res
                        println("Fails on OWA l-moments (rsoc) iteration $i")
                        find_tol(owa, owa_rsoc)
                    end
                    @test res
                    if i in 10:11
                        rtol = 5e-3
                    else
                        rtol = 1e-6
                    end
                    res = isapprox(owa_qsoc, owa_rsoc; rtol = rtol)
                    if !res
                        println("Fails on OWA l-moments (rsoc, qsoc) iteration $i")
                        find_tol(owa_qsoc, owa_rsoc)
                    end
                    @test res
                end
                rtol = if i in 4:5
                    # The entropy maximiser touches the rows θ_{t+1} ≥ θ_t tangentially, so
                    # the solver leaves more digits to the host than in the quadratic cases.
                    1e-5
                elseif i in (7, 8, 11)
                    # Iteration 11 measures rtol = 1e-7 on a Linux host and passes 5e-8 on CI.
                    # The solver sets the last digits, so the host does (#1709).
                    1e-6
                else
                    1e-9
                end
                res = isapprox(owa, owa_t[!, i]; rtol = rtol)
                if !res
                    println("Fails on OWA l-moments iteration $i")
                    find_tol(owa, owa_t[!, i]; name1 = :owa, name2 = :owa_t)
                end
                @test res
            end

            @test_throws IsEmptyError OWAJuMP(slv = Solver[])
        end
        @testset "OWA weight vectors" begin
            owa_t = CSV.read(joinpath(@__DIR__, "./assets/OWA_weights.csv.gz"), DataFrame)
            @test isapprox(owa_gmd(100), owa_t[!, 1])
            @test isapprox(owa_cvar(100), owa_t[!, 2])
            @test isapprox(owa_tg(100), owa_t[!, 3])
            @test isapprox(owa_wr(100), owa_t[!, 4])
            @test isapprox(owa_rg(100), owa_t[!, 5])
            @test isapprox(owa_cvarrg(100), owa_t[!, 6])
            @test isapprox(owa_tgrg(100), owa_t[!, 7])
            @test isapprox(owa_l_moment(100), owa_t[!, 8])
            @test isapprox(owa_l_moment(100, 3), owa_t[!, 9])
            @test isapprox(owa_l_moment(100, 4), owa_t[!, 10])
            @test isapprox(owa_l_moment(100, 10), owa_t[!, 11])

            w1 = owa_tgrg(100)
            alpha_i = 0.0001
            alpha = 0.05
            a_sim = 100
            alphas = range(; start = alpha_i, stop = alpha, length = a_sim)
            n = length(alphas)
            w = Vector{typeof(alpha)}(undef, n)

            w[1] = alphas[2] * alphas[1] / alphas[n]^2
            for i in 2:(n - 1)
                w[i] = (alphas[i + 1] - alphas[i - 1]) * alphas[i] / alphas[n]^2
            end
            w[n] = (alphas[n] - alphas[n - 1]) / alphas[n]
            w2 = owa_wcvarrg(100, alphas, w)
            @test isapprox(w1, w2)

            # No `w2` mirrors `w1`. A given `w2` is stored as given, so the caller reverses it.
            w = owa_tg(100)
            r = OrderedWeightsArrayRange(; w1 = w)
            @test r.w2 == reverse(r.w1)
            @test !hasfield(typeof(r), :rev)
            r = OrderedWeightsArrayRange(; w1 = w, w2 = w)
            @test r.w2 === w
            # The mirror and a caller's reversal give the same measure.
            x = randn(StableRNG(1896), 100) / 50
            @test OrderedWeightsArrayRange(; w1 = w)(x) ==
                  OrderedWeightsArrayRange(; w1 = w, w2 = reverse(w))(x)
            @test OrderedWeightsArrayRange(; w1 = owa_tg)(x) ≈
                  OrderedWeightsArrayRange(; w1 = owa_tg, w2 = reverse ∘ owa_tg)(x)
            # A positional rebuild from the stored fields reverses nothing a second time.
            rb = OrderedWeightsArrayRange(r.settings, r.w1, r.w2, r.alg)
            @test rb.w2 === r.w2

            # The constructor must not reverse the caller's vector in place, and building a
            # second measure from the same vector must give the same measure.
            w1 = owa_tg(8)
            w1_before = copy(w1)
            r1 = OrderedWeightsArrayRange(; w1 = w1)
            @test w1 == w1_before
            @test r1.w2 == reverse(w1_before)
            r2 = OrderedWeightsArrayRange(; w1 = w1)
            @test r1.w2 == r2.w2

            # The builder branch must not reverse a cached vector underneath the caller.
            cached = owa_cvar(8, 0.05)
            cached_before = copy(cached)
            r3 = OrderedWeightsArrayRange(; w1 = _ -> cached)
            @test r3.w2(8) == reverse(cached_before)
            @test cached == cached_before
            @test r3.w2(8) == reverse(cached_before)
        end
        @testset "OWA weight builders: closed forms and edge cases" begin
            x = [0.02, -0.01, 0.03, -0.04, 0.01, 0.005, -0.02]
            T = length(x)
            # The Gini mean difference over the ordered pairs, and its weights sum to zero.
            gmd = sum(abs(a - b) for a in x, b in x) / (T * (T - 1))
            @test isapprox(dot(owa_gmd(T), sort(x)), gmd)
            @test isapprox(sum(owa_gmd(T)), 0; atol = 1e-15)
            # The CVaR weights sum to -1 and give the value of the measure.
            @test isapprox(sum(owa_cvar(T, 0.3)), -1)
            @test isapprox(dot(owa_cvar(T, 0.3), sort(x)),
                           ConditionalValueatRisk(; alpha = 0.3)(x))
            # The quadrature weights of the tail Gini sum to one, and the range sums to zero.
            @test isapprox(sum(owa_tg(T; alpha_i = 0.01, alpha = 0.4, a_sim = 7)), -1)
            @test isapprox(sum(owa_cvarrg(T; alpha = 0.2, beta = 0.3)), 0; atol = 1e-15)
            @test isapprox(dot(owa_wr(T), sort(x)), -minimum(x))
            @test isapprox(dot(owa_rg(T), sort(x)), maximum(x) - minimum(x))
            # One observation has a range of zero.
            @test owa_rg(1) == [0.0]
            @test owa_rg(2) == [-1.0, 1.0]
            # The quadrature needs the two ends of its grid. One point raised an
            # ArgumentError from `range` at the call, after the constructor accepted it.
            @test_throws DomainError owa_tg(T; a_sim = 1)
            @test_throws DomainError OrderedWeightsArrayTailGini(; a_sim = 1)
            @test_throws DomainError OWA_TG_RG(; a_sim = 1)
            @test_throws DomainError OWA_TG_RG(; b_sim = 1)
            @test isapprox(OrderedWeightsArrayTailGini(; a_sim = 2)(T),
                           owa_tg(T; a_sim = 2))
            # Each level needs one weight. `zip` dropped the extra levels with no error.
            @test_throws DimensionMismatch owa_wcvar(T, [0.05, 0.1], [1.0])
            @test_throws DimensionMismatch owa_wcvarrg(T, [0.05], [1.0], [0.05, 0.1], [1.0])
        end
        @testset "OWA weight vectors against their measures and their hand forms" begin
            PO = PortfolioOptimisers
            x = randn(StableRNG(605), 37) / 50
            T = length(x)
            # The CVaR weights reproduce the measure at a fractional, an integer and a
            # sub-unit tail size.
            for alpha in (0.01, 0.05, 0.1, 0.27, 0.5, 0.9)
                @test isapprox(dot(owa_cvar(T, alpha), sort(x)),
                               ConditionalValueatRisk(; alpha = alpha)(x); atol = 1e-15)
            end
            @test isapprox(dot(owa_wr(T), sort(x)), WorstRealisation()(x); atol = 1e-15)
            @test isapprox(dot(owa_rg(T), sort(x)), Range()(x); atol = 1e-15)
            # The CVaR range is the CVaR of the losses plus the CVaR of the gains.
            @test isapprox(dot(owa_cvarrg(T; alpha = 0.1, beta = 0.2), sort(x)),
                           ConditionalValueatRisk(; alpha = 0.1)(x) +
                           ConditionalValueatRisk(; alpha = 0.2)(-x); atol = 1e-15)
            # The hand forms at T = 7.
            @test isapprox(owa_cvar(7, 0.3), [-1 / 2.1, -1 / 2.1, -1 + 2 / 2.1, 0, 0, 0, 0])
            @test owa_cvar(7, 0.1) == [-1.0, 0, 0, 0, 0, 0, 0]
            @test isapprox(owa_gmd(7), [2 * (2 * t - 1 - 7) / (7 * 6) for t in 1:7])
            # One observation has no pair of distinct observations, so the Gini mean difference
            # is refused. Before R13 of #1817, `owa_gmd(1)` returned the range `NaN:Inf:NaN`,
            # and a model on it stopped at a `NaN` coefficient.
            @test owa_gmd(2) == [-1.0, 1.0]
            @test_throws DomainError owa_gmd(1)
            @test_throws DomainError owa_gmd(0)
            @test_throws DomainError OrderedWeightsArray()([0.01])
            @test owa_wr(7) == [-1.0, 0, 0, 0, 0, 0, 0]
            @test owa_rg(7) == [-1.0, 0, 0, 0, 0, 0, 1]
            # Each range alias has the value of its fused vector, with the levels in their
            # tails. Asymmetric levels, so a swap of the two tails cannot pass.
            @test isa(OWA_CVaR_RG(), OrderedWeightsArrayRange)
            @test isapprox(OWA_CVaR_RG(; alpha = 0.1, beta = 0.2)(x),
                           dot(owa_cvarrg(T; alpha = 0.1, beta = 0.2), sort(x));
                           atol = 1e-15)
            @test isapprox(OWA_TG_RG(; alpha = 0.2, beta = 0.3, a_sim = 5, b_sim = 7)(x),
                           dot(owa_tgrg(T; alpha = 0.2, beta = 0.3, a_sim = 5, b_sim = 7),
                               sort(x)); atol = 1e-15)
            @test isapprox(OWA_TG_RG(; alpha_i = 0.01, alpha = 0.2, beta_i = 0.02,
                                     beta = 0.3)(x),
                           dot(owa_tgrg(T; alpha_i = 0.01, alpha = 0.2, beta_i = 0.02,
                                        beta = 0.3), sort(x)); atol = 1e-15)
            @test isapprox(OWA_RG()(x), dot(owa_rg(T), sort(x)); atol = 1e-15)
            # Each builder returns its closed form when it is called.
            @test LinearMoment(; k = 4)(T) == owa_l_moment_crm(T; k = 4)
            # A builder and its reversal hold calibration slots, and neither is a rule.
            @test !PO.is_calibration_rule(OrderedWeightsArrayTailGini())
            @test !PO.is_calibration_rule(reverse ∘ OrderedWeightsArrayTailGini())
            @test PO.is_calibration_rule(t -> t)
            # The L-moment weights against exact rational arithmetic, at a size whose
            # binomials overflowed an `Int` before the weights were computed in their own
            # element type.
            lm_exact(T, k) = [sum((-1)^j *
                                  binomial(big(k - 1), j) *
                                  binomial(big(i - 1), k - 1 - j) *
                                  binomial(big(T - i), j) for j in 0:(k - 1)) //
                              (k * binomial(big(T), k)) for i in 1:T]
            for (Ti, k) in ((7, 3), (100, 10), (1000, 8), (10000, 6))
                w = owa_l_moment(Ti, k)
                @test maximum(abs.(w - lm_exact(Ti, k))) * Ti < 1e-13
                # The first entry is (-1)^(k-1)/T and the last is 1/T.
                @test isapprox(w[1], (-1)^(k - 1) / Ti)
                @test isapprox(w[end], 1 / Ti)
            end
            @test owa_l_moment(5, 5) == [0.2, -0.8, 1.2, -0.8, 0.2]
            @test_throws DomainError owa_l_moment(3, 5)
            @test_throws DomainError owa_l_moment(3, 0)
            @test_throws DomainError LinearMoment(; k = 5)(3)
            @test eltype(owa_l_moment(big(20), 3)) == BigFloat
            # The fallback weights are the running maximum of the combination.
            W = reduce(hcat, [(-1)^i * owa_l_moment(40, i) for i in 2:6])
            phis = [prod(0.5 + j for j in 0:(i - 1)) / factorial(i + 1) for i in 1:5]
            @test PO.ncrra_weights(W, 0.5) ≈ accumulate(max, W * (phis / sum(phis)))
        end
        @testset "L-moment order: the first weight stays at -1/T" begin
            T = 60
            crm = [owa_l_moment_crm(T; k = k) for k in 2:8]
            @test crm[1] ≈ owa_l_moment(T, 2)
            @test all(w -> isapprox(w[1], -1 / T), crm)
            moves = [maximum(abs.(crm[k] - crm[k - 1])) * T for k in 2:7]
            @test isapprox(moves[1], 0.61; atol = 0.005)
            @test isapprox(moves[end], 0.053; atol = 0.0005)
            @test issorted(moves; rev = true)
        end
        @testset "L-moment programme: one feasible set, one objective each" begin
            slv = Solver(; solver = Clarabel.Optimizer, settings = Dict("verbose" => false))
            T, k = 200, 5
            W = reduce(hcat, [(-1)^i * owa_l_moment(T, i) for i in 2:k])
            entropy(phi) = -sum(p -> p > 0 ? p * log(p) : zero(p), phi)
            algs = (MaximumEntropy(), MaximumEntropy(; alg = ExponentialConeEntropy()),
                    MinimumSquaredDistance(),
                    MinimumSquaredDistance(; alg = SquaredSOCRiskExpr()),
                    MinimumSquaredDistance(; alg = RSOCRiskExpr()), MinimumSumSquares(),
                    MinimumSumSquares(; alg = SquaredSOCRiskExpr()),
                    MinimumSumSquares(; alg = RSOCRiskExpr()))
            thetas = [owa_l_moment_crm(T, OWAJuMP(; slv = slv, alg = alg); k = k)
                      for alg in algs]
            for theta in thetas
                # Every answer is in the feasible set of owa_model_setup.
                phi = W \ theta
                @test maximum(abs.(W * phi - theta)) < 1e-7
                @test isapprox(sum(phi), 1; atol = 1e-8)
                @test all(-1e-8 .<= phi .<= 0.5 + 1e-8)
                @test all(diff(phi) .<= 1e-8)
                # Twice the default feasibility tolerance of Clarabel, 1e-8.
                @test all(diff(theta) .>= -2e-8)
            end
            # The entropy of the coefficients has one maximiser, so both encodings reach it,
            # and no other objective reaches a larger entropy (#1834).
            hs = [entropy(W \ theta) for theta in thetas]
            @test isapprox(thetas[1], thetas[2]; rtol = 2e-5)
            @test maximum(hs[3:8]) < hs[1] - 1e-3
            # Each quadratic objective attains the least value of its own measure.
            ss = [sum(abs2, theta) for theta in thetas]
            sd = [sum(abs2, diff(theta)) for theta in thetas]
            @test argmin(ss) in 6:8
            # The entropy answer comes within 0.5 % of the least sum of squares.
            @test minimum(ss[6:8]) < (1 - 1e-3) * minimum(ss[1:5])
            @test argmin(sd) in 3:5
            @test minimum(sd[3:5]) < 0.95 * minimum(sd[[1, 2, 6, 7, 8]])
        end
        @testset "L-moment programme: each objective gives the default scales (#1907)" begin
            for alg in (MaximumEntropy(), MaximumEntropy(; alg = ExponentialConeEntropy()),
                        MinimumSquaredDistance(), MinimumSumSquares())
                @test PortfolioOptimisers.owa_constraint_scale(nothing, alg) == 1.0
                @test PortfolioOptimisers.owa_constraint_scale(2, alg) === 2
                @test PortfolioOptimisers.owa_objective_scale(nothing, alg) ==
                      (alg isa MaximumEntropy ? 1e-3 : 1.0)
                @test PortfolioOptimisers.owa_objective_scale(2, alg) === 2
            end
            slv = Solver(; solver = Clarabel.Optimizer, settings = Dict("verbose" => false))
            # The default OWAJuMP solved 4 of 15 programmes at so = 1 (#1907). Its default
            # scale now solves, and it returns the answer of the stated so = 1e-3.
            for (T, k) in ((200, 5), (500, 4), (1000, 6))
                w = @test_logs min_level = Logging.Warn owa_l_moment_crm(T,
                                                                         OWAJuMP(;
                                                                                 slv = slv);
                                                                         k = k)
                @test w ≈ owa_l_moment_crm(T, OWAJuMP(; slv = slv, so = 1e-3); k = k)
            end
            # A stated scale wins over the default of the objective.
            @test_logs (:warn, r"Reverting to ncrra_weights") match_mode = :any owa_l_moment_crm(200,
                                                                                                 OWAJuMP(;
                                                                                                         slv = slv,
                                                                                                         so = 1.0);
                                                                                                 k = 5)
        end
        @testset "L-moment programme: a failed solve warns and falls back" begin
            slv = Solver(; solver = Clarabel.Optimizer, settings = Dict("verbose" => false))
            ncrra(T, k) = PortfolioOptimisers.ncrra_weights(reduce(hcat,
                                                                   [(-1)^i *
                                                                    owa_l_moment(T, i)
                                                                    for i in 2:k]), 0.5)
            # At k = 2 the one coefficient must equal one, above max_phi, and the fallback
            # is the second L-moment. At k = 3 the rows leave phi = (0.5, 0.5), whose
            # weights decrease. At k = 4 the programme is infeasible up to 21 observations.
            for (T, k) in ((50, 2), (100, 3), (10, 4))
                method = OWAJuMP(; slv = slv, alg = MinimumSumSquares())
                w = @test_logs (:warn, r"Reverting to ncrra_weights") match_mode = :any owa_l_moment_crm(T,
                                                                                                         method;
                                                                                                         k = k)
                @test w ≈ ncrra(T, k)
            end
            @test owa_l_moment_crm(50, OWAJuMP(; slv = slv); k = 2) ≈ owa_l_moment(50, 2)
            w = @test_logs min_level = Logging.Warn owa_l_moment_crm(100,
                                                                     OWAJuMP(; slv = slv,
                                                                             alg = MinimumSumSquares());
                                                                     k = 4)
            @test !(w ≈ ncrra(100, 4))
            # A solver that fails is reported on every call, not swallowed.
            fail = Solver(; solver = () -> error("no solver"))
            for _ in 1:2
                w = @test_logs (:warn, r"Reverting to ncrra_weights") match_mode = :any owa_l_moment_crm(60,
                                                                                                         OWAJuMP(;
                                                                                                                 slv = fail);
                                                                                                         k = 5)
                @test w ≈ owa_l_moment_crm(60; k = 5)
            end
        end
        @testset "Approximate OWA formulation: an upper bound on the exact risk" begin
            slv = Solver(; name = :c, solver = Clarabel.Optimizer,
                         settings = Dict("verbose" => false, "tol_gap_abs" => 1e-10,
                                         "tol_gap_rel" => 1e-10, "tol_feas" => 1e-10),
                         check_sol = (; allow_local = true, allow_almost = true))
            X = 0.01 * randn(StableRNG(295), 120, 6) .+ 0.0005
            pr = prior(EmpiricalPrior(), ReturnsResult(; nx = string.('A':'F'), X = X))
            opt = JuMPOptimiser(; pe = pr, slv = slv)
            # The measured relative gap of the model risk over the exact minimum. Two-valued
            # weight vectors close it, as the `## Relaxation` block of the approximate
            # `set_risk_constraints!` method states. αT = 6 is an integer here.
            for (w, lo, hi) in
                ((owa_wr, -1e-7, 1e-7), (owa_cvar, -1e-7, 1e-7), (owa_gmd, 2e-4, 2e-3),
                 (T -> owa_l_moment_crm(T; k = 4), 0.02, 0.1))
                re = optimise(MeanRisk(;
                                       r = OrderedWeightsArray(; w = w,
                                                               alg = ExactOrderedWeightsArray()),
                                       obj = MinimumRisk(), opt = opt))
                ra = optimise(MeanRisk(;
                                       r = OrderedWeightsArray(; w = w,
                                                               alg = ApproxOrderedWeightsArray()),
                                       obj = MinimumRisk(), opt = opt))
                f = OrderedWeightsArray(; w = w)
                exact = JuMP.value(re.model[:owa_risk_1])
                approx = JuMP.value(ra.model[:aowa_risk_1])
                @test isapprox(exact, f(X * re.w); rtol = 1e-6)
                # model risk ≥ the exact risk of its weights ≥ the exact minimum.
                @test approx >= f(X * ra.w) * (1 - 1e-7)
                @test f(X * ra.w) >= exact * (1 - 1e-7)
                @test lo <= approx / exact - 1 <= hi
            end
        end
        @testset "Approximate OWA formulation: no p-norm cones for at most one inner entry (#1895)" begin
            # With at most one entry strictly inside the box, the redundant cones put the
            # optimum at the apex of every cone. Clarabel then stopped with SLOW_PROGRESS on
            # the first two ranges at these tolerances and with the defaults.
            slv = Solver(; name = :c, solver = Clarabel.Optimizer,
                         settings = Dict("verbose" => false, "tol_gap_abs" => 1e-10,
                                         "tol_gap_rel" => 1e-10, "tol_feas" => 1e-10),
                         check_sol = (; allow_local = true, allow_almost = true))
            T, N = 100, 8
            rng = StableRNG(295)
            X = 0.01 * (randn(rng, T, N) .* (1 .+ 0.5 * rand(rng, 1, N)) .+
                        0.3 * randn(rng, T, 1) .* rand(rng, 1, N)) .+ 0.0005
            X .= X .* (1 .+ (rand(rng, T, N) .< 0.05) .* 2)
            pr = prior(EmpiricalPrior(), ReturnsResult(; nx = string.('A':'H'), X = X))
            opt = JuMPOptimiser(; pe = pr, slv = slv)
            # αT = 7 + 1 ulp gives a third entry of -2.2e-16, 0.075 a fractional entry.
            for (a, b) in ((0.07, 0.07), (0.05, 0.1), (0.075, 0.13))
                w1 = OrderedWeightsArrayConditionalValueatRisk(; alpha = a)
                w2 = reverse ∘ OrderedWeightsArrayConditionalValueatRisk(; alpha = b)
                ra = optimise(MeanRisk(;
                                       r = OrderedWeightsArrayRange(; w1 = w1, w2 = w2,
                                                                    alg = ApproxOrderedWeightsArray()),
                                       obj = MinimumRisk(), opt = opt))
                re = optimise(MeanRisk(;
                                       r = OrderedWeightsArrayRange(; w1 = w1, w2 = w2,
                                                                    alg = ExactOrderedWeightsArray()),
                                       obj = MinimumRisk(), opt = opt))
                @test isa(ra.retcode, OptimisationSuccess)
                @test size(ra.model[:owa_epsilon_loss_1]) == (T, 0)
                @test size(ra.model[:owa_epsilon_gain_1]) == (T, 0)
                @test isapprox(JuMP.value(ra.model[:aowa_range_risk_1]),
                               JuMP.value(re.model[:owa_range_risk_1]); rtol = 1e-6)
            end
            # A vector with many inner entries keeps one cone block for each order of `p`.
            ra = optimise(MeanRisk(;
                                   r = OrderedWeightsArray(; w = owa_gmd,
                                                           alg = ApproxOrderedWeightsArray()),
                                   obj = MinimumRisk(), opt = opt))
            @test size(ra.model[:owa_epsilon_1]) == (T, 5)
        end
    end
end
