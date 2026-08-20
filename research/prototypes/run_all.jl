# =============================================================================
# Driver — reproduces every number quoted in
# `research/PortfolioOptimisers_Improvements_8.md`.
#
# Run it from the repository root with the package environment, which already
# holds every dependency these prototypes use:
#
#     julia -t 1 --project=. research/prototypes/run_all.jl
#
# It loads no part of PortfolioOptimisers.jl. Each prototype is a self-
# contained module, so a failure here is a failure of the prototype and never
# of the library.
#
# Every random draw is seeded, so the output is reproducible.
# =============================================================================
using LinearAlgebra, Printf, Random, Statistics
BLAS.set_num_threads(1)

const HERE = @__DIR__

include(joinpath(HERE, "01_scenario_generation.jl"))
include(joinpath(HERE, "02_wasserstein_ambiguity.jl"))
include(joinpath(HERE, "03_model_population.jl"))
include(joinpath(HERE, "04_simulated_truth_calibration.jl"))
include(joinpath(HERE, "05_breakeven_view.jl"))
include(joinpath(HERE, "06_performance_summary.jl"))

using .ScenarioGeneration, .WassersteinAmbiguity, .ModelPopulation,
      .SimulatedTruthCalibration, .BreakevenView, .PerformanceSummary

banner(s) = (println(); println("=" ^ 78); println("  ", s); println("=" ^ 78))

"""
    build_world(rng, N) -> (mu, sigma)

Build a plausible daily world: annualised volatilities near 20 per cent, a flat
correlation of `0.35`, and per-asset annual Sharpe ratios in `[0.2, 0.6]`.
"""
function build_world(rng, N)
    sd = (0.20 / sqrt(252)) .* (0.8 .+ 0.4 .* rand(rng, N))
    C = fill(0.35, N, N)
    for i in 1:N
        C[i, i] = 1.0
    end
    sigma = Diagonal(sd) * C * Diagonal(sd)
    mu = (0.4 / sqrt(252)) .* sd .* (0.5 .+ rand(rng, N))
    return mu, Matrix(sigma)
end

# -----------------------------------------------------------------------------
banner("1. Scenario generation: does each generator reproduce what it claims?")
# -----------------------------------------------------------------------------
let rng = MersenneTwister(42)
    N = 6
    mu, sigma = build_world(rng, N)

    R = simulate_returns(GaussianScenarios(), mu, sigma; n_obs = 200_000, rng = rng)
    @printf("gaussian    : max |cov error| / max|sigma| = %.2e\n",
            maximum(abs, cov(R) .- sigma) / maximum(abs, sigma))

    Rt = simulate_returns(StudentTScenarios(; nu = 6.0), mu, sigma; n_obs = 400_000,
                          rng = rng)
    ex_kurt = mean(((Rt[:, 1] .- mean(Rt[:, 1])) ./ std(Rt[:, 1])) .^ 4) - 3
    @printf("student-t   : max |cov error| / max|sigma| = %.2e,  excess kurtosis = %.2f (theory %.2f)\n",
            maximum(abs, cov(Rt) .- sigma) / maximum(abs, sigma), ex_kurt, 6 / (6 - 4))

    X = simulate_returns(StudentTScenarios(; nu = 5.0), mu, sigma; n_obs = 1500, rng = rng)
    Rc = simulate_returns(GaussianCopulaScenarios(), X; n_obs = 100_000, rng = rng)
    sq(A) = [sortperm(sortperm(view(A, :, j))) for j in 1:size(A, 2)]
    rank_err = maximum(abs, cor(hcat(sq(Rc)...)) .- cor(hcat(sq(X)...)))
    @printf("copula      : max rank-correlation error = %.3f, 1%% marginal quantile error = %.2e\n",
            rank_err,
            maximum(abs,
                    [quantile(view(Rc, :, j), 0.01) - quantile(view(X, :, j), 0.01)
                     for j in 1:N]))

    Rb = simulate_returns(StationaryBootstrapScenarios(; block_size = 10.0), X;
                          n_obs = 50_000, rng = rng)
    @printf("bootstrap   : every simulated row is an observed row = %s\n",
            all(any(all(view(Rb, i, :) .== view(X, t, :)) for t in 1:size(X, 1))
                for i in 1:200))
end

# -----------------------------------------------------------------------------
banner("2. Wasserstein: is the closed form attained by the measure it names?")
# -----------------------------------------------------------------------------
let rng = MersenneTwister(7)
    T, N = 1000, 6
    mu, sigma = build_world(rng, N)
    X = randn(rng, T, N) * transpose(cholesky(Symmetric(sigma)).L) .+ transpose(mu)
    w = normalize(abs.(randn(rng, N)), 1)
    alpha = 0.05

    @printf("%-8s %-8s %14s %14s %10s\n", "ground", "dual", "closed form", "attained",
            "abs diff")
    for q in (1.0, 2.0, Inf)
        set = WassersteinAmbiguitySet(; radius = 0.002, order = 1, ground_norm = q)
        lhs = robust_cvar(set, X, w, alpha)
        Xw = worst_case_shifted_returns(set, X, w, alpha)
        rhs = empirical_cvar(-(Xw * w), alpha)
        @printf("q=%-6s p=%-6s %14.9f %14.9f %10.1e\n", q, dual_norm_order(set), lhs, rhs,
                abs(lhs - rhs))
    end

    set = WassersteinAmbiguitySet(; radius = 0.002, order = 1, ground_norm = Inf)
    Xw = worst_case_shifted_returns(set, X, w, alpha)
    cost = mean(norm(view(Xw, t, :) .- view(X, t, :), Inf) for t in 1:T)
    @printf("\ntransport cost of the worst case = %.6f, radius = %.6f\n", cost, set.radius)

    set2 = WassersteinAmbiguitySet(; radius = 0.004, order = 2, ground_norm = 2.0)
    S = cov(X)
    @printf("robust sd = %.6f = empirical sd %.6f + radius * ||w||_2 %.6f\n",
            robust_std(set2, S, w), sqrt(dot(w, S, w)), 0.004 * norm(w, 2))

    println("\ncalibrated radius, scale = 1% (the 1/sqrt(T) rate):")
    for TT in (250, 1000, 4000)
        @printf("  T = %5d  ->  delta = %.6f\n", TT,
                WassersteinAmbiguity.radius_from_confidence(TT, N; confidence = 0.95,
                                                            scale = 0.01))
    end
end

# -----------------------------------------------------------------------------
banner("3. Model population: the ambiguity decomposition")
# -----------------------------------------------------------------------------
let rng = MersenneTwister(11)
    N, M = 8, 6
    W = zeros(N, M)
    for m in 1:M
        v = abs.(randn(rng, N))
        v[rand(rng, 1:N)] = 0.0
        W[:, m] = v ./ sum(v)
    end
    pop = PortfolioPopulation(W; names = ["model $(m)" for m in 1:M])
    _, sigma = build_world(rng, N)
    d = ambiguity_decomposition(pop, sigma)
    @printf("average member variance = %.6e\n", d.mean_member)
    @printf("consensus variance      = %.6e\n", d.consensus)
    @printf("disagreement            = %.6e\n", d.disagreement)
    @printf("identity residual       = %.2e  (must be ~0)\n", d.residual)
    @printf("consensus is never worse: %s\n", d.consensus <= d.mean_member)
    @printf("mean pairwise active share = %.3f\n", mean_disagreement(pop))
    @printf("effective models = %.2f of a possible %d\n", effective_number_of_models(pop),
            M - 1)
    ss = support_stability(pop)
    @printf("mean Jaccard = %.2f, contested assets = %s\n", ss.mean_jaccard, ss.contested)
end

# -----------------------------------------------------------------------------
banner("4. Calibration against a known truth")
# -----------------------------------------------------------------------------
let
    rng = MersenneTwister(2026)
    N = 25
    mu_t, sig_t = build_world(rng, N)
    ests = ["sample" => sample_moments, "ledoit-wolf cov" => ledoit_wolf_identity,
            "bayes-stein mean" => bayes_stein_mean,
            "1/N (ignores the fit)" =>
                X -> (ones(size(X, 2)), Matrix(1.0I, size(X, 2), size(X, 2)))]
    for T in (60, 250, 1000)
        res = calibration_study(mu_t, sig_t; estimators = ests, T = T, n_trials = 400,
                                rng = MersenneTwister(1))
        @printf("\nT = %4d observations   oracle annual Sharpe = %.3f\n", T,
                res[1].oracle_sharpe * sqrt(252))
        @printf("  %-22s %9s %9s %10s %9s %9s\n", "estimator", "SR(ann)", "SR loss",
                "risk infl", "AS(orcl)", "leverage")
        for r in res
            @printf("  %-22s %9.3f %9.3f %10.2f %9.2f %9.2f\n", r.name,
                    r.sharpe_mean * sqrt(252), r.sharpe_loss * sqrt(252), r.risk_inflation,
                    r.active_share_from_oracle, r.gross_leverage)
        end
        @assert all(r -> r.sharpe_loss >= -1e-12, res) "an estimator beat the oracle"
    end
end

# -----------------------------------------------------------------------------
banner("5. The breakeven view")
# -----------------------------------------------------------------------------
let rng = MersenneTwister(3)
    T, N = 2000, 10
    mu, sigma = build_world(rng, N)
    X = randn(rng, T, N) * transpose(cholesky(Symmetric(sigma)).L) .+ transpose(mu)
    p = fill(1 / T, T)
    mu_h = vec(mean(X; dims = 1))
    S_h = cov(X)
    w_mv = Symmetric(S_h) \ ones(N)
    w_mv ./= sum(w_mv)
    w_ts = Symmetric(S_h) \ mu_h
    w_ts ./= sum(w_ts)
    w_eq = fill(1 / N, N)

    @printf("%-26s %10s %12s %8s %9s\n", "decision", "edge", "delta* (nats)", "ENS%",
            "p_detect")
    for (name, wa, wb) in
        (("tangency vs 1/N", w_ts, w_eq), ("1/N vs min-variance", w_eq, w_mv),
         ("tangency vs min-variance", w_ts, w_mv))
        c = X * (wa .- wb)
        r = breakeven_entropy(c, p)
        @printf("%-26s %10.2e %12.3e %8.1f %9.3f\n", name, r.prior_edge, r.divergence,
                100 * r.ens_fraction, detection_error_probability(r.divergence, T))
    end

    println("\nentropy cost of a stated view on asset 1's mean:")
    x1 = view(X, :, 1)
    for d in (0.0, 1e-4, 2e-4, 4e-4, 8e-4)
        r = view_entropy_cost(collect(x1), mean(x1) + d, p)
        @printf("  shift %+0.5f -> D = %.6f nats, ENS = %5.1f%%, p_detect = %.3f\n", d,
                r.divergence, 100 * r.ens_fraction,
                detection_error_probability(r.divergence, T))
    end
end

# -----------------------------------------------------------------------------
banner("6. The performance summary that the plot extension hides")
# -----------------------------------------------------------------------------
let rng = MersenneTwister(5)
    T = 2520
    r = 0.0004 .+ 0.01 .* randn(rng, T)
    r[500:520] .-= 0.02
    ps = performance_summary(r; periods_per_year = 252, alpha = 0.05, compound = false)
    @printf("annualised return    %8.4f\n", ps.ann_return)
    @printf("annualised vol       %8.4f\n", ps.ann_volatility)
    @printf("Sharpe               %8.4f  +/- %.4f  <- the plot cannot show this\n",
            ps.sharpe, ps.sharpe_stderr)
    @printf("Sortino              %8.4f\n", ps.sortino)
    @printf("Calmar               %8.4f\n", ps.calmar)
    @printf("maximum drawdown     %8.4f\n", ps.max_drawdown)
    @printf("Ulcer index          %8.4f\n", ps.ulcer_index)
    @printf("CVaR(5%%)             %8.4f\n", ps.cvar)
    @printf("skewness             %8.4f\n", ps.skewness)
    @printf("excess kurtosis      %8.4f\n", ps.excess_kurtosis)
    @printf("hit rate             %8.4f\n", ps.hit_rate)
    println()
    println("The Sharpe ratio is ", round(ps.sharpe / ps.sharpe_stderr; digits = 2),
            " standard errors from zero. A bar chart of the point estimate alone")
    println("invites a reader to rank this strategy against another. The Result does not.")
end

println()
println("All prototypes ran.")
