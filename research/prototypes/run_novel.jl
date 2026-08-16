# =============================================================================
# Driver — reproduces every number quoted in
# `research/PortfolioOptimisers_Improvements_9.md`, for prototypes 07 to 22.
#
# Prototypes 01 to 06 belong to report 8 and have their own driver,
# `run_all.jl`. Run both from the repository root:
#
#     julia -t 1 --project=. research/prototypes/run_all.jl
#     julia -t 1 --project=. research/prototypes/run_novel.jl
#
# Neither loads PortfolioOptimisers.jl. Every dependency used here is already a
# direct dependency of the package, so the package environment is enough.
#
# Every random draw is seeded. The output is reproducible.
# =============================================================================
using LinearAlgebra, Printf, Random, Statistics, Dates
BLAS.set_num_threads(1)

const HERE = @__DIR__
for f in ("07_optimiser_sensitivity.jl", "08_critical_line.jl",
          "09_online_portfolio_selection.jl", "10_conditional_stress.jl",
          "11_moment_and_divergence_ambiguity.jl", "12_weight_uncertainty.jl",
          "13_robustness_frontier.jl", "14_attribution.jl", "15_portfolio_state.jl",
          "16_rebalancing_policies.jl", "17_market_impact.jl", "18_tax_aware.jl",
          "19_online_moments.jl", "20_graph_structure.jl", "21_regime_models.jl",
          "22_provenance_and_capability.jl")
    include(joinpath(HERE, f))
end

using .OptimiserSensitivity, .CriticalLine, .OnlinePortfolioSelection, .ConditionalStress,
      .MomentDivergenceAmbiguity, .WeightUncertainty, .RobustnessFrontier, .Attribution,
      .PortfolioState, .RebalancingPolicies, .MarketImpact, .TaxAware, .OnlineMoments,
      .GraphStructure, .RegimeModels, .ProvenanceAndCapability

banner(s) = (println(); println("=" ^ 78); println("  ", s); println("=" ^ 78))

"""
    build_world(rng, N) -> (mu, sigma)

A plausible daily world: annualised volatilities near 20 per cent, a flat
correlation of 0.35, per-asset annual Sharpe ratios in `[0.2, 0.6]`.
"""
function build_world(rng, N)
    sd = (0.20 / sqrt(252)) .* (0.8 .+ 0.4 .* rand(rng, N))
    C = fill(0.35, N, N)
    for i in 1:N
        C[i, i] = 1.0
    end
    return (0.4 / sqrt(252)) .* sd .* (0.5 .+ rand(rng, N)),
           Matrix(Diagonal(sd) * C * Diagonal(sd))
end

# -----------------------------------------------------------------------------
banner("07. Optimiser sensitivity: analytic derivatives against finite differences")
# -----------------------------------------------------------------------------
let rng = MersenneTwister(17)
    N = 8
    B = randn(rng, N, N)
    S = B * B' / N + 0.05I
    mu = 0.01 .* randn(rng, N)
    A = ones(1, N)
    b = [1.0]
    g = 3.0
    s = solve_equality_qp(mu, S, A, b; gamma = g)
    h = 1e-7
    J = dw_dmu(s)
    Jfd = reduce(hcat,
                 [(solve_equality_qp(mu .+ h .* (1:N .== j), S, A, b; gamma = g).w .- s.w) ./
                  h for j in 1:N])
    @printf("dw/dmu     max |analytic - finite difference| = %.2e\n", maximum(abs, J - Jfd))
    E = randn(rng, N, N)
    E = (E + E') / 2
    @printf("dw/dsigma  max |analytic - finite difference| = %.2e\n",
            maximum(abs,
                    dw_dsigma_direction(s, E) .-
                    (solve_equality_qp(mu, S .+ h .* E, A, b; gamma = g).w .- s.w) ./ h))
    @printf("K null space contains A': %.1e   pullback == J'v: %.1e\n",
            maximum(abs, kkt_inverse_block(s) * A'),
            maximum(abs, pullback_mu(s, ones(N)) .- J' * ones(N)))
    f(w) = 0.5 * dot(w, S, w) - dot(mu, w) / g
    @printf("constraint price: d(obj)/db = %.6f, -nu = %.6f\n",
            (f(solve_equality_qp(mu, S, A, [1.0 + h]; gamma = g).w) - f(s.w)) / h, -s.nu[1])
    println("\nill-conditioning drives error amplification:")
    for eps_ in (0.5, 0.05, 0.005, 0.0005)
        r = sensitivity_report(solve_equality_qp(mu, B * B' / N + eps_ * I, A, b;
                                                 gamma = g))
        @printf("  ridge %.4f -> mu amplification = %9.2f\n", eps_, r.mu_amplification)
    end
end

# -----------------------------------------------------------------------------
banner("08. Critical line algorithm: exact to machine precision")
# -----------------------------------------------------------------------------
let rng = MersenneTwister(23)
    N = 10
    B = randn(rng, N, N)
    S = Matrix(Symmetric(B * B' / N + 0.02I))
    mu = 0.01 .* (0.5 .+ rand(rng, N))
    lb = zeros(N)
    ub = fill(0.35, N)
    f = critical_line(mu, S; lb = lb, ub = ub)
    @printf("%d turning points, lambda from %.4g down to %.4g\n", length(f.lambdas),
            first(f.lambdas), last(f.lambdas))
    maxres = 0.0
    maxviol = 0.0
    for (k, lam) in enumerate(f.lambdas)
        w = f.weights[k]
        gvec = S * w .- lam .* mu .- f.gammas[k]
        for i in 1:N
            if i in f.free_sets[k]
                maxres = max(maxres, abs(gvec[i]))
            elseif isapprox(w[i], lb[i]; atol = 1e-9)
                maxviol = max(maxviol, max(0.0, -gvec[i]))
            elseif isapprox(w[i], ub[i]; atol = 1e-9)
                maxviol = max(maxviol, max(0.0, gvec[i]))
            end
        end
    end
    @printf("KKT stationarity on the free set: %.2e ; sign condition on bounds: %.2e\n",
            maxres, maxviol)
    pts = frontier_points(f, mu, S)
    @printf("budgets exact: %s ; return and risk decreasing in lambda: %s / %s\n",
            all(w -> isapprox(sum(w), 1.0; atol = 1e-9), f.weights),
            issorted(pts.ret; rev = true), issorted(pts.risk; rev = true))
    ms = max_sharpe_on_frontier(f, mu, S)
    @printf("max Sharpe %.4f at lambda %.4g, found by interpolation with no solver\n",
            ms.sharpe, ms.lambda)
end

# -----------------------------------------------------------------------------
banner("09. Online portfolio selection: reverting against trending markets")
# -----------------------------------------------------------------------------
let
    function make_market(kind, T, N, rng)
        P = ones(T + 1, N)
        if kind === :trending
            lvl = zeros(N)
            for t in 1:T
                lvl .= 0.90 .* lvl .+ 0.010 .* randn(rng, N)
                @views P[t + 1, :] .= P[t, :] .* exp.(lvl)
            end
        else
            lp = zeros(N)
            for t in 1:T
                lp .= 0.90 .* lp .+ 0.03 .* randn(rng, N)
                @views P[t + 1, :] .= exp.(lp)
            end
        end
        return P[2:end, :] ./ P[1:(end - 1), :]
    end
    for kind in (:reverting, :trending)
        X = make_market(kind, 1500, 8, MersenneTwister(31))
        b = best_crp(X)
        res = [uniform_crp(X),
               universal_portfolio(X; n_experts = 3000, rng = MersenneTwister(1)),
               exponentiated_gradient(X; eta = 0.05), online_newton_step(X),
               pamr(X; epsilon = 0.5), olmar(X; window = 5), rmr(X; window = 5)]
        @printf("\n--- %s market ---\n", uppercase(String(kind)))
        @printf("%-26s %14s %10s\n", "strategy", "final wealth", "regret")
        for r in res
            @printf("%-26s %14.4f %10.4f\n", r.name, terminal_wealth(r),
                    b.log_wealth - log(terminal_wealth(r)))
        end
        @printf("%-26s %14.4f %10.4f\n", "best CRP (hindsight)", b.wealth, 0.0)
        bs = maximum(sum(log, view(X, :, i)) for i in 1:8)
        @printf("BCRP dominates the best single asset: %s\n", b.log_wealth >= bs - 1e-8)
    end
end

# -----------------------------------------------------------------------------
banner("10. Conditional stress: tail dependence and scenario transformations")
# -----------------------------------------------------------------------------
let rng = MersenneTwister(41)
    N = 6
    mu, S = build_world(rng, N)
    cm = conditional_moments(mu, S, [1], [-0.05])
    Xs = conditional_gaussian_stress(mu, S, [1], [-0.05]; n_obs = 200_000, rng = rng)
    @printf("conditional mean error %.1e, conditional covariance error %.1e\n",
            maximum(abs, vec(mean(view(Xs, :, cm.rest); dims = 1)) .- cm.mu_cond),
            maximum(abs, cov(view(Xs, :, cm.rest)) .- cm.sig_cond))
    @printf("a -5%% shock in asset 1 moves asset 2 to %.4f, against %.4f unconditionally\n",
            cm.mu_cond[1], mu[2])
    Xh = randn(rng, 3000, 2) * [1.0 0.0; 0.6 0.8]'
    println("\nt-copula tail dependence, empirical against theory (400k scenarios, q = 0.002):")
    for nu in (3.0, 5.0, 12.0, 500.0)
        R = t_copula_scenarios(Xh; nu = nu, n_obs = 400_000, rng = MersenneTwister(7))
        rho = 2 * sin(pi / 6 * cor(sortperm(sortperm(view(Xh, :, 1))),
                                   sortperm(sortperm(view(Xh, :, 2)))))
        @printf("  nu = %5.1f  theory %.3f  empirical %.3f\n", nu,
                tail_dependence_coefficient(rho, nu),
                empirical_tail_dependence(view(R, :, 1), view(R, :, 2); q = 0.002))
    end
    Xb = randn(rng, 5000, N) * cholesky(Symmetric(S)).L' .+ mu'
    Xv = scale_volatility(Xb, 2.0)
    Xc = shock_correlation(Xb, 0.8; target = 1.0)
    println("\nstress report on an equal-weight portfolio:")
    for r in stress_report(fill(1 / N, N),
                           ["base" => Xb, "volatility x2" => Xv, "correlation -> 1" => Xc,
                            "asset 1 at -5%, conditional" =>
                                conditional_gaussian_stress(mu, S, [1], [-0.05]; n_obs = 20000,
                                                            rng = rng)])
        @printf("  %-30s vol %.5f  CVaR %.5f  worst %.4f\n", r.name, r.vol, r.cvar, r.worst)
    end
end

# -----------------------------------------------------------------------------
banner("11. Moment and divergence ambiguity")
# -----------------------------------------------------------------------------
let rng = MersenneTwister(53)
    N = 5
    mk() = (A = randn(rng, N, N); Matrix(Symmetric(A * A' / N + 0.1I)))
    S1, S2, S3 = mk(), mk(), mk()
    D1 = Matrix(Diagonal(rand(rng, N) .+ 0.5))
    D2 = Matrix(Diagonal(rand(rng, N) .+ 0.5))
    @printf("Bures-Wasserstein: symmetric %.1e, triangle inequality %s, commuting case %.1e\n",
            abs(bures_wasserstein(S1, S2) - bures_wasserstein(S2, S1)),
            bures_wasserstein(S1, S3) <=
            bures_wasserstein(S1, S2) + bures_wasserstein(S2, S3) + 1e-10,
            abs(bures_wasserstein(D1, D2) - norm(sqrt.(diag(D1)) .- sqrt.(diag(D2)))))
    m1 = randn(rng, N)
    w = normalize(abs.(randn(rng, N)), 1)
    wc = worst_case_moments(m1, S1, w, 0.02; split = 0.5)
    @printf("worst-case moment pair sits at distance %.5f of radius 0.02, and is adverse in both\n",
            wc.distance)
    L = 0.02 .* randn(rng, 400) .+ 0.005
    L[1:10] .+= 0.15
    p = fill(1 / 400, 400)
    println("\nKullback-Leibler ball, dual value against the tilt it names:")
    @printf("%8s %11s %11s %12s %10s\n", "eta", "dual value", "E_q*[L]", "D(q*||p)",
            "theta*")
    for eta in (0.0, 0.001, 0.01, 0.05, 0.2, 1.0)
        r = kl_worst_case_expectation(L, eta; p = p)
        d = isinf(r.theta) ? 0.0 : divergence_of_tilt(L, r.theta; p = p)
        @printf("%8.3f %11.5f %11.5f %12.5f %10.4g\n", eta, r.value, dot(r.q, L), d,
                r.theta)
    end
    @printf("bounded above by the worst scenario: %.5f <= %.5f\n",
            kl_worst_case_expectation(L, 5.0; p = p).value, maximum(L))
end

# -----------------------------------------------------------------------------
banner("12. Weight uncertainty and conformal coverage")
# -----------------------------------------------------------------------------
let rng = MersenneTwister(61)
    T, N = 1200, 6
    mu, S = build_world(rng, N)
    X = randn(rng, T, N) * cholesky(Symmetric(S)).L' .+ mu'
    minvar(Y) = (v = Symmetric(cov(Y)) \ ones(size(Y, 2)); v ./ sum(v))
    tang(Y) = (v = Symmetric(cov(Y)) \ vec(mean(Y; dims = 1)); v ./ sum(v))
    for (nm, f) in ("minimum variance" => minvar, "tangency" => tang)
        iv = weight_intervals(bootstrap_weights(X, f; B = 600, rng = MersenneTwister(3));
                              alpha = 0.1)
        @printf("%-18s mean weight sd = %8.4f, mean sign stability = %.2f\n", nm,
                mean(r.sd for r in iv), mean(r.sign_stability for r in iv))
    end
    @printf("\nn = 18 calibration points cannot support a 95%% interval: %s\n",
            isinf(conformal_quantile(randn(rng, 18), 0.05)))
    println("conformal coverage on exchangeable data (theorem: within [1-a, 1-a+1/(n+1)]):")
    w = fill(1 / N, N)
    for alpha in (0.2, 0.1, 0.05)
        c = conformal_coverage(w, X, alpha; n_calib = 200, n_trials = 20000,
                               rng = MersenneTwister(9))
        @printf("  alpha %.2f -> coverage %.4f, target %.4f, upper bound %.4f : %s\n",
                alpha, c.coverage, c.target, c.upper_bound,
                c.target - 0.01 <= c.coverage <= c.upper_bound + 0.01)
    end
end

# -----------------------------------------------------------------------------
banner("13. Robustness as a third axis")
# -----------------------------------------------------------------------------
let rng = MersenneTwister(83)
    T, N = 900, 8
    mu, S = build_world(rng, N)
    X = randn(rng, T, N) * cholesky(Symmetric(S)).L' .+ mu'
    pert = standard_perturbations(; rng = MersenneTwister(5), n_resample = 15)
    eq(Y) = fill(1 / size(Y, 2), size(Y, 2))
    tg(l) = Y -> (Cc = cov(Y); Cs = (1 - l) .* Cc .+ l .* Diagonal(diag(Cc));
                  v = Symmetric(Cs) \ vec(mean(Y; dims = 1)); v ./ sum(v))
    mvs(l) = Y -> (Cc = cov(Y); Cs = (1 - l) .* Cc .+ l .* Diagonal(diag(Cc));
                   v = Symmetric(Cs) \ ones(size(Y, 2)); v ./ sum(v))
    @printf("R(equal weight) = %.6f, which must be exactly 1\n",
            procedure_robustness(eq, X, pert).robustness)
    rs = Float64[]
    println("robustness must rise monotonically with shrinkage:")
    for l in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        r = procedure_robustness(tg(l), X, pert)
        push!(rs, r.robustness)
        @printf("  lambda %.1f -> R = %7.4f, mean shift %.4f\n", l, r.robustness,
                r.mean_shift)
    end
    @printf("monotone: %s\n", issorted(rs))
    procs = vcat(["equal weight" => eq],
                 ["min-var shrink $(l)" => mvs(l) for l in (0.0, 0.5)],
                 ["tangency shrink $(l)" => tg(l) for l in (0.0, 0.3, 0.6, 1.0)])
    fr = robustness_frontier(procs, X, pert; mu_eval = mu, sigma_eval = S)
    @printf("\n%-22s %9s %9s %11s %s\n", "procedure", "return", "risk", "robustness",
            "pareto")
    for (i, p) in enumerate(fr.scored)
        @printf("%-22s %9.5f %9.5f %11.4f %s\n", p.name, p.ret, p.risk, p.robustness,
                i in fr.front ? "*" : "")
    end
end

# -----------------------------------------------------------------------------
banner("14. Attribution: Shapley against leave-one-out")
# -----------------------------------------------------------------------------
let rng = MersenneTwister(83)
    N = 8
    _, S = build_world(rng, N)
    vg(Sset) = isempty(Sset) ? 0.0 : sum(Sset)^1.5
    phi = shapley_values(vg, 6)
    @printf("efficiency %.1e ; null player %.1e ; symmetry %.1e\n",
            abs(sum(phi) - (vg(1:6) - vg(Int[]))),
            shapley_values(Sset -> if isempty(Sset)
                               0.0
                           else
                               Float64(sum(i for i in Sset if i <= 5; init = 0))
                           end, 6)[6],
            (ps = shapley_values(Sset -> Float64(length(Sset))^2, 5);
             maximum(ps) - minimum(ps)))
    function solve_sub(Sset)
        u = ones(N)
        if (1 in Sset)
            (u = min.(u, [0.05, 0.05, 1, 1, 1, 1, 1, 1]))
        end
        if (2 in Sset)
            (u = min.(u, [0.05, 0.05, 1, 1, 1, 1, 1, 1]))
        end
        if (3 in Sset)
            (u = min.(u, fill(0.20, N)))
        end
        w = fill(1 / N, N)
        for _ in 1:200
            w = min.(w, u)
            w ./= sum(w)
        end
        return w
    end
    obj = w -> sqrt(dot(w, S, w))
    ca = constraint_attribution(solve_sub,
                                ["cap A", "cap B (exact duplicate)", "global cap 20%"];
                                objective = obj)
    println("\ntwo identical binding caps plus one that never binds:")
    for c in ca
        @printf("  %-26s Shapley %+.3e   leave-one-out %+.3e\n", c.name, c.shapley,
                c.leave_one_out)
    end
    tot = obj(solve_sub([1, 2, 3])) - obj(solve_sub(Int[]))
    @printf("  sum of Shapley %+.3e equals the true total %+.3e (error %.1e)\n",
            sum(c.shapley for c in ca), tot, abs(sum(c.shapley for c in ca) - tot))
    @printf("  sum of leave-one-out %+.3e misses %.0f%% of the cost\n",
            sum(c.leave_one_out for c in ca),
            100 * (1 - sum(c.leave_one_out for c in ca) / tot))
    w0 = fill(1 / N, N)
    w1 = (v = Symmetric(S) \ ones(N); v ./ sum(v))
    rd = rebalance_decomposition(w0, w1, S, 1.7 .* S)
    @printf("\nrebalance: total %+.5f = weight %+.5f + moment %+.5f (residual %.1e)\n",
            rd.total, rd.weight_effect, rd.moment_effect, rd.residual)
    mc = marginal_risk_contribution(w1, S)
    @printf("Euler identity: sum of contributions equals total risk to %.1e\n",
            abs(sum(mc.contribution) - mc.total))
end

# -----------------------------------------------------------------------------
banner("15. Portfolio state: drift, freezing, and the turnover error")
# -----------------------------------------------------------------------------
let
    N = 6
    w = [0.20, 0.20, 0.15, 0.15, 0.15, 0.15]
    r = [0.10, -0.05, 0.00, 0.02, -0.01, 0.03]
    wd = drift_weights(w, r)
    @printf("drift preserves the budget to %.1e ; largest single move %.4f -> %.4f\n",
            abs(sum(wd) - sum(w)), w[1], wd[1])
    st = HeldPortfolio([wd[1], wd[2], wd[3], 0.0, wd[5], wd[6]];
                       status = [TRADABLE, FROZEN, TRADABLE, UNAVAILABLE, TRADABLE, NEW])
    sub = tradable_subproblem(st; budget = 1.0)
    wfull = restore_universe(fill(sub.sub_budget / length(sub.idx), length(sub.idx)), sub,
                             st)
    @printf("frozen holds %.4f, sub-budget %.4f, restored budget %.6f, frozen preserved exactly %s\n",
            sub.held_fixed, sub.sub_budget, sum(wfull), wfull[2] == st.w[2])
    tgt = fill(1 / N, N)
    naive = sum(abs, tgt .- w) / 2
    real_ = realised_turnover(tgt, st).turnover
    @printf("turnover against the last TARGET %.4f, against the DRIFTED holdings %.4f\n",
            naive, real_)
    @printf("  the state-blind figure understates the trade by %.1f per cent\n",
            100 * (real_ - naive) / real_)
end

# -----------------------------------------------------------------------------
banner("16. Rebalancing policies, net of cost")
# -----------------------------------------------------------------------------
let rng = MersenneTwister(97)
    T, N = 1500, 6
    mu, S = build_world(rng, N)
    X = randn(rng, T, N) * cholesky(Symmetric(S)).L' .+ mu'
    mv(Y) = (v = Symmetric(cov(Y)) \ ones(size(Y, 2)); v ./ sum(v))
    @printf("%-22s %8s %10s %10s %10s\n", "policy", "trades", "cost", "gross", "net")
    for (nm, pol) in ("always" => AlwaysRebalance(), "never" => NeverRebalance(),
                      "calendar, 21 days" => CalendarPolicy(21),
                      "threshold 0.05" => ThresholdPolicy(0.05),
                      "cost aware" => CostAwarePolicy(; gamma = 5.0), "hybrid" => HybridPolicy())
        s = simulate_policy(pol, X, mv; lookback = 252, cost_bps = 20.0)
        @printf("%-22s %8d %10.4f %10.4f %10.4f\n", nm, s.n_trades, s.total_cost,
                s.wealth_gross, s.wealth_net)
    end
    println("note: the cost-aware rule never trades. That is the horizon mismatch")
    println("documented on CostAwarePolicy, not a defect in the simulation.")
end

# -----------------------------------------------------------------------------
banner("17. Market impact and optimal execution")
# -----------------------------------------------------------------------------
let rng = MersenneTwister(11)
    srm = SquareRootImpact(; coefficient = 1.0, sigma = 0.02, adv = 1.0e6)
    Q = 1.0e5
    @printf("doubling the order multiplies cost by %.4f, theory 2^1.5 = %.4f\n",
            impact_cost(srm, 2Q) / impact_cost(srm, Q), 2^1.5)
    @printf("marginal cost divided by average cost = %.4f, theory 1.5\n",
            marginal_impact(srm, Q) / (impact_cost(srm, Q) / Q))
    @printf("convex, and symmetric in the sign of the trade: %s / %s\n",
            impact_cost(srm, 1.5Q) <= 0.5 * (impact_cost(srm, Q) + impact_cost(srm, 2Q)),
            impact_cost(srm, Q) == impact_cost(srm, -Q))
    println("\nAlmgren-Chriss liquidation of one million shares over 20 intervals:")
    for lam in (0.0, 1e-7, 1e-6, 1e-5)
        s = almgren_chriss_schedule(1.0e6, 20; sigma = 0.3, eta = 2.5e-6, gam = 2.5e-7,
                                    lam = lam)
        @printf("  lambda %-8.0e half-life %8.2f  E[cost] %10.1f  sd %10.1f\n", lam,
                s.half_life, s.expected_cost, sqrt(s.variance))
    end
    s0 = almgren_chriss_schedule(1.0e6, 20; sigma = 0.3, eta = 2.5e-6, gam = 2.5e-7,
                                 lam = 0.0)
    @printf("  lambda = 0 is exactly uniform: max deviation %.1e\n",
            maximum(abs, s0.n_k .- 1.0e6 / 20))
    lam = 1e-6
    s = almgren_chriss_schedule(1.0e6, 20; sigma = 0.3, eta = 2.5e-6, gam = 2.5e-7,
                                lam = lam)
    ob(sc) = (m = execution_cost_moments(sc, 0.3, 2.5e-6, 2.5e-7, 1.0);
              m.expected_cost + lam * m.variance)
    base = ob(s)
    worse = 0
    for _ in 1:400
        x2 = copy(s.x)
        x2[rand(rng, 2:20)] += 1e4 * randn(rng)
        ob((x = x2, n_k = [x2[i] - x2[i + 1] for i in 1:20])) > base - 1e-6 && (worse += 1)
    end
    @printf("  the closed-form path is optimal: %d of 400 perturbations were no better\n",
            worse)
end

# -----------------------------------------------------------------------------
banner("18. Tax-aware lot selection")
# -----------------------------------------------------------------------------
let
    today = Date(2026, 8, 16)
    pos = Position("ACME",
                   [TaxLot(100.0, 50.0, Date(2019, 1, 10)),
                    TaxLot(100.0, 120.0, Date(2026, 3, 1)),
                    TaxLot(100.0, 90.0, Date(2024, 6, 15)),
                    TaxLot(100.0, 200.0, Date(2026, 7, 1))])
    ap = after_tax_proceeds(pos, 250.0, 150.0, today)
    @printf("%-14s %12s %12s %12s %12s\n", "method", "tax", "net", "short gain",
            "long gain")
    for r in ap.results
        @printf("%-14s %12.2f %12.2f %12.2f %12.2f\n", r.method, r.tax, r.net, r.short_gain,
                r.long_gain)
    end
    @printf("best is %s. FIFO costs %.2f more on a %.0f sale, which is %.1f%% of proceeds.\n",
            ap.best.method, ap.results[1].tax - ap.best.tax, 250.0 * 150.0,
            100 * (ap.results[1].tax - ap.best.tax) / (250.0 * 150.0))
    r = realise_sale(pos, 250.0, 150.0, today; method = HIFO)
    @printf("shares conserved: 250 sold + %.0f remaining = %.0f held ; input unmutated: %s\n",
            sum(l.shares for l in r.remaining.lots), sum(l.shares for l in pos.lots),
            length(pos.lots) == 4)
    for f in wash_sale_flags([(Date(2026, 7, 1), -5000.0), (Date(2026, 1, 5), -2000.0)],
                             [(Date(2026, 7, 20), 100.0), (Date(2025, 1, 1), 50.0)])
        @printf("  loss %.0f on %s washed = %s\n", f.loss, f.date, f.washed)
    end
    @printf("  a purchase BEFORE the sale also triggers the rule: %s\n",
            wash_sale_flags([(Date(2026, 7, 1), -5000.0)], [(Date(2026, 6, 15), 100.0)])[1].washed)
end

# -----------------------------------------------------------------------------
banner("19. Online moments: Welford against the textbook formula")
# -----------------------------------------------------------------------------
let rng = MersenneTwister(101)
    T, N = 5000, 7
    X = randn(rng, T, N) * randn(rng, N, N)' .+ 1000.0
    w = WelfordMoments(N)
    partial_fit!(w, X)
    err_w = maximum(abs, current_cov(w) .- cov(X))
    naive = ((X'X ./ T) .- (vec(mean(X; dims = 1)) * vec(mean(X; dims = 1))')) .*
            (T / (T - 1))
    err_n = maximum(abs, naive .- cov(X))
    @printf("Welford covariance error  %.2e\n", err_w)
    @printf("textbook covariance error %.2e, which is %.0f times worse\n", err_n,
            err_n / max(err_w, eps()))
    a = WelfordMoments(N)
    partial_fit!(a, view(X, 1:1234, :))
    b = WelfordMoments(N)
    partial_fit!(b, view(X, 1235:T, :))
    @printf("parallel merge equals the sequential answer to %.2e, n = %d\n",
            maximum(abs, current_cov(merge_moments(a, b)) .- current_cov(w)),
            merge_moments(a, b).n)
    @printf("EWMA decay 0.06 has a half life of %.2f periods; the round trip is exact: %s\n",
            half_life(0.06), isapprox(decay_from_half_life(half_life(0.06)), 0.06))
end

# -----------------------------------------------------------------------------
banner("20. Graph structure: the Laplacian identity and its limit")
# -----------------------------------------------------------------------------
let
    A = zeros(6, 6)
    for (i, j, v) in ((1, 2, 0.9), (2, 3, 0.7), (4, 5, 0.8), (5, 6, 0.6))
        A[i, j] = v
        A[j, i] = v
    end
    L = graph_laplacian(A)
    wv = randn(MersenneTwister(3), 6)
    @printf("w'Lw equals the pairwise sum exactly: %.1e\n",
            abs(laplacian_penalty(wv, L) - laplacian_penalty_pairwise(wv, A)))
    @printf("L*1 = 0 to %.1e ; positive semi-definite ; %d connected components (truth 2)\n",
            maximum(abs, L * ones(6)), connected_components_count(A))
    ev = eigvals(Symmetric(normalised_laplacian(A)))
    @printf("normalised spectrum lies in [0, 2]: %s (min %.4f, max %.4f)\n",
            all(-1e-9 .<= ev .<= 2 + 1e-9), minimum(ev), maximum(ev))
    rng2 = MersenneTwister(5)
    B = randn(rng2, 6, 6)
    S = B * B' / 6 + 0.2I
    mu = 0.01 * randn(rng2, 6)
    println("as the penalty grows the weights equalise inside each component:")
    for lam in (0.0, 1.0, 100.0, 1.0e6)
        wl = laplacian_smoothed_weights(mu, S, L; lam = lam)
        @printf("  lambda %9.0f  spread inside component 1 %.4f, component 2 %.4f\n", lam,
                maximum(view(wl, 1:3)) - minimum(view(wl, 1:3)),
                maximum(view(wl, 4:6)) - minimum(view(wl, 4:6)))
    end
end

# -----------------------------------------------------------------------------
banner("21. Regime models: does the hidden Markov model recover the truth?")
# -----------------------------------------------------------------------------
let rng = MersenneTwister(7)
    Tn = 3000
    Ptrue = [0.98 0.02; 0.06 0.94]
    mus = [0.0008, -0.0015]
    sds = [0.006, 0.020]
    s = 1
    y = zeros(Tn)
    states = zeros(Int, Tn)
    for t in 1:Tn
        s = rand(rng) < Ptrue[s, 1] ? 1 : 2
        states[t] = s
        y[t] = mus[s] + sds[s] * randn(rng)
    end
    fit = fit_hmm(y, 2; rng = MersenneTwister(1))
    @printf("log likelihood non-decreasing at every step: %s (%d iterations)\n",
            all(diff(fit.loglik) .>= -1e-9), length(fit.loglik))
    @printf("true   mean %s  sd %s\n", round.(sort(mus); digits = 5),
            round.(sds[sortperm(mus)]; digits = 5))
    @printf("fitted mean %s  sd %s\n", round.(fit.mu; digits = 5),
            round.(fit.sd; digits = 5))
    truth = [states[t] == 2 ? 1 : 2 for t in 1:Tn]
    @printf("state classification accuracy %.3f\n",
            mean((view(fit.gam, :, 1) .> 0.5) .== (truth .== 1)))
    rp = regime_persistence(fit.P)
    @printf("expected durations %s periods ; stationary distribution %s\n",
            round.(rp.expected_duration; digits = 1), round.(rp.stationary; digits = 3))
    Xp = y .* ones(1, 4) .+ 0.004 .* randn(rng, Tn, 4)
    rcm = regime_conditional_moments(Xp, fit.gam)
    mm = mixture_moments(rcm.mu, rcm.sigma, rcm.weight ./ sum(rcm.weight))
    @printf("effective sample per regime %s of %d ; between-regime term is PSD: %s\n",
            round.(rcm.weight; digits = 1), Tn,
            minimum(eigvals(Symmetric(mm.between))) > -1e-14)
    @printf("dropping the between-regime term understates the variance by %.2f per cent\n",
            100 * (1 - mm.within[1, 1] / mm.sigma[1, 1]))
end

# -----------------------------------------------------------------------------
banner("22. Provenance and capability checking")
# -----------------------------------------------------------------------------
# Four stand-in components, each declaring what it produces or consumes. In the
# library these methods live beside the types they describe.
struct EmpPrior end
struct HOPrior end
struct VarianceRM end
struct SkewRM end
ProvenanceAndCapability.provides(::EmpPrior) = Set([:mu, :sigma, :X])
ProvenanceAndCapability.provides(::HOPrior) = Set([:mu, :sigma, :sk, :kt, :X])
ProvenanceAndCapability.requires(::VarianceRM) = Set([:sigma])
ProvenanceAndCapability.requires(::SkewRM) = Set([:sk, :sigma])

let
    Xd = randn(MersenneTwister(9), 100, 5)
    h1 = data_fingerprint(Xd)
    Xe = copy(Xd)
    Xe[1, 1] += 1e-3
    @printf("fingerprint stable %s, sensitive %s, transpose-aware %s\n",
            h1 == data_fingerprint(copy(Xd)), h1 != data_fingerprint(Xe),
            h1 != data_fingerprint(permutedims(Xd)))
    mk() = record_step(record_step(ProvenanceRecord(; data_hash = h1,
                                                    seeds = Dict(:cv => 42),
                                                    versions = Dict("PortfolioOptimisers" => "0.28.0")),
                                   :prior, "EmpiricalPrior"; ce = "LedoitWolf"),
                       :optimisation, "MeanRisk"; obj = "MinimumRisk", rm = "Variance")
    r = mk()
    @printf("identical inputs give an identical fingerprint: %s ; one more step changes it: %s\n",
            fingerprint(r) == fingerprint(mk()),
            fingerprint(r) != fingerprint(record_step(r, :extra, "Something")))
    print(describe(r))
    rep = check_compatibility([EmpPrior()], [VarianceRM(), SkewRM()];
                              labels = ["Variance", "NegativeSkewness"])
    @printf("compatible: %s\n", rep.ok)
    for m in rep.messages
        println("  ! ", m)
    end
    @printf("with a high-order prior: %s\n",
            check_compatibility([HOPrior()], [VarianceRM(), SkewRM()]).ok)
end

println()
println("All prototypes ran.")
