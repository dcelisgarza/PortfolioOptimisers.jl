# The relativistic risk measures of `src/16_RiskMeasures/06_XatRisk/04_RelativisticXatRisk.jl`,
# checked against the mathematics their docstrings state. The reference below solves the
# Kaniadakis ball of the dual with no conic solver, so it is independent of both programmes
# that `RRM` builds.
using Clarabel, JuMP

# The Kaniadakis logarithm, written out so the reference does not read `kappa_log`.
rl_lnk(u, k) = (u^k - u^(-k)) / (2k)
# Stationarity of `q ln_k(q / c)` at slope `y` has the root `q = c u^(1/k)` below.
function rl_q(L, c, k, lam, eta)
    y = (L .- eta) ./ lam
    u = (k .* y .+ sqrt.(k^2 .* y .^ 2 .+ (1 - k^2))) ./ (1 + k)
    return c .* u .^ (1 / k)
end
# The multiplier of `sum(q) = 1`, found by bisection, because `sum(q)` falls as it rises.
function rl_eta(L, c, k, lam)
    lo = minimum(L) - 1.0
    hi = maximum(L) + 1.0
    while sum(rl_q(L, c, k, lam, lo)) < 1
        lo -= hi - lo
    end
    while sum(rl_q(L, c, k, lam, hi)) > 1
        hi += hi - lo
    end
    for _ in 1:200
        m = (lo + hi) / 2
        sum(rl_q(L, c, k, lam, m)) > 1 ? (lo = m) : (hi = m)
    end
    return (lo + hi) / 2
end
# sup_q q'L subject to sum(q) = 1 and sum(q ln_k(q / (p T))) <= ln_k(1 / (alpha T)). The
# multiplier `lam` of the ball is found by bisection on its logarithm, because the
# divergence of the maximiser falls as `lam` rises.
function rl_reference(x, alpha, k, w = nothing)
    T = length(x)
    L = -x
    p = isnothing(w) ? fill(1 / T, T) : w ./ sum(w)
    c = p .* T
    r = rl_lnk(1 / (alpha * T), k)
    divergence(lam) = (q = rl_q(L, c, k, lam, rl_eta(L, c, k, lam));
                       sum(q .* rl_lnk.(q ./ c, k)))
    llo, lhi = -30.0, 30.0
    for _ in 1:200
        m = (llo + lhi) / 2
        divergence(exp(m)) > r ? (llo = m) : (lhi = m)
    end
    lam = exp((llo + lhi) / 2)
    q = rl_q(L, c, k, lam, rl_eta(L, c, k, lam))
    return dot(q, L)
end
# A solver whose first construction fails, so the primal programme of `RRM` fails and the
# dual programme runs on the same solver.
function rl_fails_once()
    n = Ref(0)
    f = () -> (n[] += 1;
               n[] == 1 ? error("the first construction fails") : Clarabel.Optimizer())
    return Solver(; name = :fails_once, solver = f, settings = Dict("verbose" => false),
                  check_sol = (; allow_local = true, allow_almost = true))
end

@testset "Relativistic risk measures" begin
    rng = StableRNG(123)
    x = 0.01 .* randn(rng, 200) .+ 0.0005
    wt = rand(rng, 200) .+ 0.5
    W = pweights(wt)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false),
                 check_sol = (; allow_local = true, allow_almost = true))

    @testset "RRM solves the Kaniadakis ball" begin
        for (a, k) in ((0.05, 0.3), (0.05, 0.7), (0.1, 0.1), (0.2, 0.5))
            @test isapprox(PortfolioOptimisers.RRM(x, slv, a, k), rl_reference(x, a, k);
                           rtol = 1e-5)
            @test isapprox(PortfolioOptimisers.RRM(x, slv, a, k, W),
                           rl_reference(x, a, k, wt); rtol = 1e-5)
        end
        # Equal weights are the unweighted measure.
        @test isapprox(PortfolioOptimisers.RRM(x, slv, 0.2, 0.5, pweights(fill(3.0, 200))),
                       PortfolioOptimisers.RRM(x, slv, 0.2, 0.5); rtol = 1e-10)
        # With equal weights and alpha T <= 1 the ball holds every distribution.
        @test isapprox(PortfolioOptimisers.RRM(x, slv, 1e-3, 0.3), -minimum(x); rtol = 1e-6)
        @test_throws IsEmptyError PortfolioOptimisers.RRM(x, Solver[], 0.05, 0.3)
        for T in (RelativisticValueatRisk, RelativisticValueatRiskRange,
                  RelativisticDrawdownatRisk, RelativeRelativisticDrawdownatRisk)
            @test_throws IsEmptyError T(; slv = Solver[])
        end
    end

    @testset "RRM falls back to the dual programme" begin
        for (a, k) in ((0.1, 0.1), (0.2, 0.5))
            @test isapprox(PortfolioOptimisers.RRM(x, rl_fails_once(), a, k),
                           rl_reference(x, a, k); rtol = 1e-5)
            @test isapprox(PortfolioOptimisers.RRM(x, rl_fails_once(), a, k, W),
                           rl_reference(x, a, k, wt); rtol = 1e-5)
        end
        dead = Solver(; name = :dead, solver = () -> error("no solver"))
        @test isnan(PortfolioOptimisers.RRM(x, dead, 0.05, 0.3))
        @test isnan(PortfolioOptimisers.RRM(x, dead, 0.05, 0.3, W))
    end

    @testset "RelativisticValueatRisk sits between EVaR and the largest loss" begin
        a = 0.05
        cvar = ConditionalValueatRisk(; alpha = a)(x)
        evar = EntropicValueatRisk(; alpha = a, slv = slv)(x)
        rl = [RelativisticValueatRisk(; alpha = a, kappa = k, slv = slv)(x)
              for k in (0.01, 0.3, 0.7, 0.99)]
        @test issorted([cvar; evar; rl; -minimum(x)])
        @test isapprox(rl[1], evar; rtol = 1e-3)
        @test isapprox(rl[end], -minimum(x); rtol = 1e-6)
        # The functor needs a solver, and `factory` fills it.
        @test_throws MethodError RelativisticValueatRisk()(x)
        pr = prior(EmpiricalPrior(), hcat(x, 0.01 .* randn(StableRNG(7), 200)))
        @test factory(RelativisticValueatRisk(), pr, slv).slv === slv
    end

    @testset "The range adds the loss tail and the gain tail" begin
        r = RelativisticValueatRiskRange(; slv = slv, alpha = 0.2, kappa_a = 0.5,
                                         beta = 0.1, kappa_b = 0.1)
        @test isapprox(r(x), rl_reference(x, 0.2, 0.5) + rl_reference(-x, 0.1, 0.1);
                       rtol = 1e-5)
    end

    @testset "The drawdown measures read their drawdown series" begin
        c = cumsum(x)
        dd = c .- accumulate(max, [0.0; c])[2:end]
        C = cumprod(1 .+ x)
        rdd = C ./ accumulate(max, [1.0; C])[2:end] .- 1
        a, k = 0.2, 0.5
        rldar = RelativisticDrawdownatRisk(; alpha = a, kappa = k, slv = slv)(x)
        @test isapprox(rldar, rl_reference(dd, a, k); rtol = 1e-5)
        @test isapprox(RelativeRelativisticDrawdownatRisk(; alpha = a, kappa = k,
                                                          slv = slv)(x),
                       rl_reference(rdd, a, k); rtol = 1e-5)
        # DaR <= CDaR <= EDaR <= RLDaR <= MDD, Cajas's Equation 7.100.
        @test issorted([DrawdownatRisk(; alpha = a)(x),
                        ConditionalDrawdownatRisk(; alpha = a)(x),
                        EntropicDrawdownatRisk(; alpha = a, slv = slv)(x), rldar,
                        MaximumDrawdown()(x)])
    end
end
