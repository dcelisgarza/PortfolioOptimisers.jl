# The conditional tail measures of `src/16_RiskMeasures/06_XatRisk/02_ConditionalXatRisk.jl`,
# checked against the forms their docstrings state: every functor against a Rockafellar-Uryasev
# linear program and the sorted closed form, the weight validation of all seven constructors,
# and the two robust programs against Equation 27 of Mohajerin Esfahani and Kuhn.
using Clarabel, HiGHS, JuMP

# The weighted Rockafellar-Uryasev minimum, min over nu of -nu + sum(p .* max.(nu .- x, 0)) /
# (alpha * sum(p)), solved as a linear program.
function ru_lp(x, alpha, p = ones(length(x)))
    m = Model(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, nu)
    @variable(m, z[1:length(x)] >= 0)
    @constraint(m, z .>= nu .- x)
    @objective(m, Min, -nu + sum(p .* z) / (alpha * sum(p)))
    optimize!(m)
    return objective_value(m)
end
# The sorted closed form of the same minimum, without weights.
function sorted_tail(x, alpha)
    s = sort(x)
    aT = alpha * length(x)
    k = ceil(Int, aT)
    return -(sum(s[1:(k - 1)]) + (aT - (k - 1)) * s[k]) / aT
end
# The drawdown series of the definitions, with the running peaks starting at c_0 = 0 and C_0 = 1.
abs_dd(x) = cumsum(x) .- accumulate(max, cumsum(x); init = zero(eltype(x)))
rel_dd(x) = cumprod(1 .+ x) ./ accumulate(max, cumprod(1 .+ x); init = one(eltype(x))) .- 1
# The long-only minimum of mean(loss) + l * CVaR(loss) over the simplex, with the loss the
# negated portfolio return, or the portfolio drawdown when `dd` is true. It is the limit of the
# robust measures as the radius goes to zero.
function mean_cvar_lp(X, alpha, l, p; dd = false)
    T, N = size(X)
    m = Model(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, w[1:N] >= 0)
    @constraint(m, sum(w) == 1)
    @variable(m, nu)
    @variable(m, z[1:T] >= 0)
    loss = if dd
        C = cumsum(X; dims = 1)
        @variable(m, d[1:T])
        for t in 1:T
            @constraint(m, d[t] >= -dot(C[t, :], w))
            for s in 1:t
                @constraint(m, d[t] >= dot(C[s, :] - C[t, :], w))
            end
        end
        d
    else
        -X * w
    end
    @constraint(m, z .>= loss .- nu)
    sp = sum(p)
    @objective(m, Min, sum(p .* loss) / sp + l * (nu + sum(p .* z) / (alpha * sp)))
    optimize!(m)
    return objective_value(m)
end

@testset "Conditional tail measures" begin
    @testset "Each functor is the Rockafellar-Uryasev minimum" begin
        rng = StableRNG(1047)
        for T in (7, 20, 63), alpha in (0.01, 0.05, 0.33, 0.9)
            x = 0.02 .* randn(rng, T) .+ 0.001
            x0 = copy(x)
            p = rand(rng, T)
            p[1] = 0.0
            pw = pweights(p)
            cv = ConditionalValueatRisk(; alpha = alpha)
            @test isapprox(cv(x), ru_lp(x, alpha); atol = 1e-12)
            @test isapprox(cv(x), sorted_tail(x, alpha); atol = 1e-14)
            @test isapprox(ConditionalValueatRisk(; alpha = alpha, w = pw)(x),
                           ru_lp(x, alpha, p); atol = 1e-12)
            # Unit weights price the same quantity as no weights.
            @test isapprox(ConditionalValueatRisk(; alpha = alpha, w = pweights(ones(T)))(x),
                           cv(x); atol = 1e-14)
            @test DistributionallyRobustConditionalValueatRisk(; alpha = alpha)(x) == cv(x)
            for beta in (alpha, 0.2)
                rg = ConditionalValueatRiskRange(; alpha = alpha, beta = beta)
                rgw = ConditionalValueatRiskRange(; alpha = alpha, beta = beta, w = pw)
                @test isapprox(rg(x), ru_lp(x, alpha) + ru_lp(-x, beta); atol = 1e-12)
                @test isapprox(rgw(x), ru_lp(x, alpha, p) + ru_lp(-x, beta, p);
                               atol = 1e-12)
                @test DistributionallyRobustConditionalValueatRiskRange(; alpha = alpha,
                                                                        beta = beta)(x) ==
                      rg(x)
            end
            cd = ConditionalDrawdownatRisk(; alpha = alpha)
            @test isapprox(cd(x), ru_lp(abs_dd(x), alpha); atol = 1e-12)
            @test isapprox(ConditionalDrawdownatRisk(; alpha = alpha, w = pw)(x),
                           ru_lp(abs_dd(x), alpha, p); atol = 1e-12)
            @test DistributionallyRobustConditionalDrawdownatRisk(; alpha = alpha)(x) ==
                  cd(x)
            @test isapprox(RelativeConditionalDrawdownatRisk(; alpha = alpha)(x),
                           ru_lp(rel_dd(x), alpha); atol = 1e-12)
            @test isapprox(RelativeConditionalDrawdownatRisk(; alpha = alpha, w = pw)(x),
                           ru_lp(rel_dd(x), alpha, p); atol = 1e-12)
            # No functor reorders the caller's vector.
            @test x == x0
        end
    end

    @testset "The upper tail of the range is the CVaR of the negated returns, bit for bit" begin
        # The functor negates its copy and sorts it ascending, rather than sorting it with
        # `rev = true` (issue #1341). Negation is exact, so the range must equal the two plain
        # CVaRs exactly, ties and all, in every precision.
        rng = StableRNG(1341)
        for T in (Float32, Float64, BigFloat), n in (1, 7, 50), alpha in (0.05, 0.3, 0.99),
            beta in (0.05, 0.3, 0.99)

            x = T.(round.(randn(rng, n); digits = 1))
            x0 = copy(x)
            rg = ConditionalValueatRiskRange(; alpha = alpha, beta = beta)
            @test isequal(rg(x),
                          ConditionalValueatRisk(; alpha = alpha)(x) +
                          ConditionalValueatRisk(; alpha = beta)(-x))
            @test x == x0
        end
    end

    @testset "Every constructor refuses a negative weight" begin
        # `DistributionallyRobustConditionalValueatRisk` checked only that the weights were not
        # empty, and took a negative weight its six siblings refuse.
        bad = pweights([0.5, -0.1, 0.6])
        for R in (ConditionalValueatRisk, DistributionallyRobustConditionalValueatRisk,
                  ConditionalValueatRiskRange, DistributionallyRobustConditionalValueatRiskRange,
                  ConditionalDrawdownatRisk, DistributionallyRobustConditionalDrawdownatRisk,
                  RelativeConditionalDrawdownatRisk)
            @test_throws DomainError R(; w = bad)
            @test R(; w = pweights([0.5, 0.0, 0.5])) isa R
        end
    end

    rng = StableRNG(10471)
    X = 0.02 .* randn(rng, 40, 4) .+ [0.002 0.001 0.0 -0.001]
    pr = prior(EmpiricalPrior(), X)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = Dict("verbose" => false, "tol_gap_abs" => 1e-10,
                                 "tol_gap_rel" => 1e-10, "tol_feas" => 1e-10))
    min_risk(r) = optimise(MeanRisk(; r = r, obj = MinimumRisk(),
                                    opt = JuMPOptimiser(; pe = pr, slv = slv)))

    @testset "The robust program is Equation 27" begin
        # Values of Equation 27 over this sample, from an independent implementation of it:
        # (alpha, l, r) => (optimum, long-only weights).
        ref = ((0.05, 1.0, 1e-4) => (1.562371905272e-02,
                                     [4.934610126584e-01, 4.881094891810e-01, 0.0, 1.842949813185e-02]),
               (0.2, 3.0, 1e-4) => (2.864793597636e-02,
                                    [3.368775790342e-01, 2.049947764238e-01, 0.0, 4.581276439958e-01]),
               (0.05, 1.0, 0.02) => (1.238196688788e-01, fill(0.25, 4)),
               (0.2, 0.5, 0.1) => (9.027779962236e-02, fill(0.25, 4)))
        for ((alpha, l, r), (val, w)) in ref
            sol = min_risk(DistributionallyRobustConditionalValueatRisk(; alpha = alpha,
                                                                        l = l, r = r))
            @test isapprox(JuMP.value(sol.model[:risk]), val; rtol = 1e-8)
            @test isapprox(sol.w, w; atol = 1e-6)
        end
    end

    @testset "As the radius goes to zero the mean term stays" begin
        p = collect(range(0.2, 1.0; length = size(X, 1)))
        for (wo, pp) in ((nothing, ones(size(X, 1))), (pweights(p), p)), l in (0.5, 3.0),
            alpha in (0.05, 0.2)

            cvar = min_risk(DistributionallyRobustConditionalValueatRisk(; alpha = alpha,
                                                                         l = l, r = 1e-9,
                                                                         w = wo))
            @test isapprox(JuMP.value(cvar.model[:risk]), mean_cvar_lp(X, alpha, l, pp);
                           atol = 1e-7)
            cdar = min_risk(DistributionallyRobustConditionalDrawdownatRisk(; alpha = alpha,
                                                                            l = l, r = 1e-9,
                                                                            w = wo))
            @test isapprox(JuMP.value(cdar.model[:risk]),
                           mean_cvar_lp(X, alpha, l, pp; dd = true); atol = 1e-7)
        end
        # The mean terms of the two tails of the range cancel.
        for (alpha, beta, la, lb) in ((0.05, 0.1, 1.0, 2.0), (0.2, 0.05, 0.5, 3.0))
            sol = min_risk(DistributionallyRobustConditionalValueatRiskRange(;
                                                                             alpha = alpha,
                                                                             beta = beta,
                                                                             l_a = la,
                                                                             l_b = lb,
                                                                             r_a = 1e-9,
                                                                             r_b = 1e-9))
            m = Model(HiGHS.Optimizer)
            set_silent(m)
            T, N = size(X)
            @variable(m, w[1:N] >= 0)
            @constraint(m, sum(w) == 1)
            @variable(m, nu[1:2])
            @variable(m, z[1:T, 1:2] >= 0)
            @constraint(m, z[:, 1] .>= -X * w .- nu[1])
            @constraint(m, z[:, 2] .>= X * w .- nu[2])
            @objective(m, Min,
                       la * (nu[1] + sum(z[:, 1]) / (alpha * T)) +
                       lb * (nu[2] + sum(z[:, 2]) / (beta * T)))
            optimize!(m)
            @test isapprox(JuMP.value(sol.model[:risk]), objective_value(m); atol = 1e-7)
        end
    end

    @testset "The drawdown premium reads a bound on the portfolio drawdown" begin
        # For long-only weights the portfolio drawdown is at most -w' D_t, the linear function
        # of the asset drawdowns that the robust premium of the drawdown measure reads.
        D = PortfolioOptimisers.absolute_drawdown_arr(X)
        for _ in 1:50
            w = rand(rng, size(X, 2))
            w ./= sum(w)
            @test all(-PortfolioOptimisers.absolute_drawdown_vec(X * w) .<= -D * w .+ 1e-15)
        end
    end

    @testset "A measure whose slots hold numbers resolves to itself" begin
        for r in (DistributionallyRobustConditionalValueatRisk(),
                  DistributionallyRobustConditionalValueatRiskRange(),
                  DistributionallyRobustConditionalDrawdownatRisk())
            @test PortfolioOptimisers.resolve_deferred_quantities(r, pr) === r
        end
    end
end
