@testset "Plotting" begin
    using Test, PortfolioOptimisers, StatsPlots, GraphRecipes, Clarabel, CSV, TimeSeries,
          LinearAlgebra, Random, Statistics

    ## ── helpers ──────────────────────────────────────────────────────────────
    is_plot(x) = x isa Plots.Plot || x isa Plots.AbstractLayout

    ## ── shared test data ──────────────────────────────────────────────────────
    rng = MersenneTwister(42)
    T, N = 120, 5
    X = randn(rng, T, N) .* 0.01
    w = (1:N) ./ sum(1:N)
    mu = vec(mean(X; dims = 1))
    sigma = cov(X)
    nx = string.('A':'E')

    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end],
                           TimeArray(CSV.File(joinpath(@__DIR__, "./assets/Factors.csv.gz"));
                                     timestamp = :Date)[(end - 252):end])

    # w_rd matches the number of assets in rd (SP500 slice, typically 20)
    w_rd = fill(1 / size(rd.X, 2), size(rd.X, 2))

    pr  = prior(EmpiricalPrior(), rd)
    fpr = prior(FactorPrior(), rd)

    slv = [Solver(; name = :c1, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = "verbose" => false),
           Solver(; name = :c2, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = ["verbose" => false, "max_step_fraction" => 0.95])]

    r_cvr = ConditionalValueatRisk()   # return-based: works with raw X windows
    mr = MeanRisk(; r = r_cvr, opt = JuMPOptimiser(; slv = slv))
    res = optimise(mr, rd)

    @testset "Keyword validation" begin
        # alpha validated in plot_drawdowns, plot_histogram, plot_performance_summary
        @test_throws DomainError plot_drawdowns(w, X; alpha = 0.0)
        @test_throws DomainError plot_drawdowns(w, X; alpha = 1.0)
        @test_throws DomainError plot_histogram(w, X; alpha = 0.0)
        @test_throws DomainError plot_performance_summary(w, X; alpha = 1.0)
        # kappa validated in plot_drawdowns, plot_histogram
        @test_throws DomainError plot_drawdowns(w, X; kappa = 0.0)
        @test_throws DomainError plot_histogram(w, X; kappa = 1.0)
        # delta validated in plot_risk_contribution, plot_factor_risk_contribution
        @test_throws DomainError plot_risk_contribution(r_cvr, w, X; delta = 0.0)
        # points validated in plot_histogram
        @test_throws DomainError plot_histogram(w, X; points = -1)
        # rolling validated in plot_rolling_measure, plot_rolling_drawdowns
        @test_throws DomainError plot_rolling_measure(r_cvr, w, X; rolling = -1)
        @test_throws DomainError plot_rolling_drawdowns(w, X; rolling = -1)
    end

    @testset "plot_portfolio_cumulative_returns" begin
        @test is_plot(plot_portfolio_cumulative_returns(w, X))
        @test is_plot(plot_portfolio_cumulative_returns(w, X; ts = 1:T))
        @test is_plot(plot_portfolio_cumulative_returns(w, X; compound = true))
        @test is_plot(plot_portfolio_cumulative_returns(w_rd, rd))
        @test is_plot(plot_portfolio_cumulative_returns(w_rd, pr))
        @test is_plot(plot_portfolio_cumulative_returns(res, rd))
        @test is_plot(plot_portfolio_cumulative_returns(res))
    end

    @testset "plot_asset_cumulative_returns" begin
        @test is_plot(plot_asset_cumulative_returns(w, X))
        @test is_plot(plot_asset_cumulative_returns(w, X; nx = nx))
        @test is_plot(plot_asset_cumulative_returns(w_rd, rd))
    end

    @testset "plot_composition" begin
        @test is_plot(plot_composition(w))
        @test is_plot(plot_composition(w, nx))
    end

    @testset "plot_stacked_bar_composition" begin
        # VecVecNum: vector of weight vectors
        @test is_plot(plot_stacked_bar_composition([w, w], nx))
    end

    @testset "plot_stacked_area_composition" begin
        @test is_plot(plot_stacked_area_composition([w, w], nx))
    end

    @testset "plot_risk_contribution" begin
        @test is_plot(plot_risk_contribution(r_cvr, w, X))
        @test is_plot(plot_risk_contribution(r_cvr, w, X; nx = nx))
        @test is_plot(plot_risk_contribution(r_cvr, w_rd, rd))
    end

    @testset "plot_factor_risk_contribution" begin
        # rd supplies both asset returns (rd.X) and factor data for regression
        @test is_plot(plot_factor_risk_contribution(r_cvr, w_rd, rd.X; rd = rd))
    end

    @testset "plot_dendrogram" begin
        cle = ClustersEstimator()
        clr = clusterise(cle, pr.X)
        @test is_plot(plot_dendrogram(clr))
        @test is_plot(plot_dendrogram(cle, pr.X))
        @test is_plot(plot_dendrogram(cle, pr))
        @test is_plot(plot_dendrogram(cle, rd))
    end

    @testset "plot_clusters" begin
        cle = ClustersEstimator()
        clr = clusterise(cle, pr.X)
        @test is_plot(plot_clusters(clr))
        @test is_plot(plot_clusters(cle, pr.X))
        @test is_plot(plot_clusters(cle, pr, rd.nx))
        @test is_plot(plot_clusters(cle, rd))

        # A non-unit diagonal sends plot_clusters down its cov2cor! branch, which must not
        # rewrite the caller's stored similarity matrix.
        cov_clr = Clusters(; res = clr.res, S = 4 * clr.S, D = clr.D, k = clr.k)
        S0 = copy(cov_clr.S)
        @test is_plot(plot_clusters(cov_clr))
        @test cov_clr.S == S0
    end

    @testset "plot_drawdowns" begin
        @test is_plot(plot_drawdowns(w, X))
        @test is_plot(plot_drawdowns(w_rd, rd))
        @test is_plot(plot_drawdowns(w, X; compound = true))
    end

    @testset "plot_histogram" begin
        @test is_plot(plot_histogram(w, X))
        @test is_plot(plot_histogram(w, X; reference = false))
        @test is_plot(plot_histogram(w_rd, rd))
    end

    @testset "plot_network" begin
        ne = NetworkEstimator()
        @test is_plot(plot_network(ne, X, nx))
    end

    @testset "plot_centrality" begin
        cte = CentralityEstimator()
        @test is_plot(plot_centrality(cte, X, nx; N = 20, percentage = true))
    end

    @testset "plot_correlation" begin
        @test is_plot(plot_correlation(sigma, nx))
        C = sigma ./ (sqrt.(diag(sigma)) .* sqrt.(diag(sigma))')
        @test is_plot(plot_correlation(C, nx))
    end

    @testset "plot_mu" begin
        @test is_plot(plot_mu(mu))
        @test is_plot(plot_mu(pr.mu, rd.nx))
    end

    @testset "plot_sigma" begin
        @test is_plot(plot_sigma(sigma))
        @test is_plot(plot_sigma(pr.sigma, rd.nx))
        @test is_plot(plot_sigma(sigma; variance = true))
    end

    @testset "plot_factor_loadings" begin
        @test is_plot(plot_factor_loadings(fpr))
        @test is_plot(plot_factor_loadings(fpr, rd.nx, rd.nf))
        @test is_plot(plot_factor_loadings(fpr, rd))
    end

    @testset "plot_factor_sigma" begin
        @test is_plot(plot_factor_sigma(fpr.f_sigma, rd.nf))
        @test is_plot(plot_factor_sigma(fpr))
        @test is_plot(plot_factor_sigma(fpr, rd.nf))
        @test is_plot(plot_factor_sigma(fpr, rd))
    end

    @testset "plot_eigenspectrum" begin
        @test is_plot(plot_eigenspectrum(sigma))
        @test is_plot(plot_eigenspectrum(sigma; N_obs = T, reference = true))
    end

    @testset "plot_rolling_measure" begin
        # use return-based risk measure (CVaR), not covariance-based (Variance)
        @test is_plot(plot_rolling_measure(r_cvr, w, X))
        @test is_plot(plot_rolling_measure(r_cvr, w, X; rolling = 20))
        # The method rolls the series through `rolling_window_measure` rather than through a
        # second copy of the loop, so a `rolling` longer than the sample now raises. It used
        # to give an empty vector of risks and an empty plot, which the neighbouring
        # docstring already called a caller error (#770).
        @test is_plot(plot_rolling_measure(r_cvr, w, X; rolling = size(X, 1)))
        @test_throws DomainError plot_rolling_measure(r_cvr, w, X; rolling = size(X, 1) + 1)
    end

    @testset "plot_cv_scores" begin
        scores = [0.5, 0.6, 0.55, 0.52, 0.58]
        @test is_plot(plot_cv_scores(scores))
        @test is_plot(plot_cv_scores(scores, string.(1:5)))
    end

    @testset "plot_turnover" begin
        K = 6
        w_series = [normalize(abs.(randn(rng, N)), 1) for _ in 1:K]
        @test is_plot(plot_turnover(w_series))
        @test is_plot(plot_turnover(w_series; ts = 1:K))
    end

    @testset "plot_factor_mu" begin
        @test is_plot(plot_factor_mu(fpr.f_mu))
        @test is_plot(plot_factor_mu(fpr.f_mu, rd.nf))
        @test is_plot(plot_factor_mu(fpr))
        @test is_plot(plot_factor_mu(fpr, rd.nf))
        @test is_plot(plot_factor_mu(fpr, rd))
    end

    @testset "Factor entry points guard the factor block" begin
        # `pr` is an `EmpiricalPrior` result, so it has no factor block. Each of the six
        # prior-taking factor entry points must reach `assert_prior_regression` and throw
        # its message. This is a falsification witness for two distinct pre-fix behaviours:
        # `plot_factor_loadings` did check, but threw a bare `ArgumentError` whose message
        # named neither the cause nor the remedy; `plot_factor_sigma` and `plot_factor_mu`
        # had the check written but *unreachable*, because their optional axis-name argument
        # defaulted to a size taken off the missing block — `1:size(pr.f_sigma, 1)` and
        # `1:length(pr.f_mu)` — which Julia evaluates before the body, so the one-argument
        # form died on `size(::Nothing, ::Int64)` / `length(::Nothing)` instead.
        @test isnothing(pr.rr)
        @test isnothing(pr.fpr)
        for f in (plot_factor_loadings, plot_factor_sigma, plot_factor_mu)
            @test_throws PortfolioOptimisers.IsNothingError f(pr)
            @test_throws PortfolioOptimisers.IsNothingError f(pr, rd)
        end

        # The plotting lead names the entry point and its matrix-arity escape hatch; the
        # shared tail names the one cause and the one remedy, and is the same string the
        # estimator consumers get, so the two cannot drift apart on the diagnosis.
        err = try
            plot_factor_loadings(pr)
        catch e
            e
        end
        @test occursin("`plot_factor_loadings` draws the regression loadings `rr.M`",
                       err.msg)
        @test occursin("plot_factor_loadings(M, nx, nf)", err.msg)
        @test occursin(PortfolioOptimisers.prior_regression_remedy, err.msg)
        # …and it does *not* claim the caller is an estimator, which is the default lead.
        @test !occursin("this estimator projects factor moments", err.msg)

        err_mu = try
            plot_factor_mu(pr, rd)
        catch e
            e
        end
        @test occursin("`plot_factor_mu` draws the factor expected returns `fpr.mu`",
                       err_mu.msg)
        @test occursin(PortfolioOptimisers.prior_regression_remedy, err_mu.msg)
    end

    @testset "plot_benchmark" begin
        B_vec = randn(rng, T)
        @test is_plot(plot_benchmark(w, X, B_vec))
        # VecVecNum form: vector of benchmark return vectors
        @test is_plot(plot_benchmark(w, X, [B_vec, randn(rng, T)]))
    end

    @testset "plot_coskewness" begin
        sk = randn(rng, N, N^2)
        @test is_plot(plot_coskewness(sk))
        @test is_plot(plot_coskewness(sk, nx))
    end

    @testset "plot_cokurtosis" begin
        kt_raw = randn(rng, N^2, N^2)
        kt = kt_raw * kt_raw'
        @test is_plot(plot_cokurtosis(kt))
        @test is_plot(plot_cokurtosis(kt; heatmap = true))
    end

    @testset "plot_prior" begin
        @test is_plot(plot_prior(pr))
        @test is_plot(plot_prior(pr, rd))
    end

    @testset "Optimisation-based dispatch" begin
        @test !isnothing(res)

        @test is_plot(plot_asset_cumulative_returns(res, rd))
        @test is_plot(plot_composition(res, rd))
        @test is_plot(plot_composition(res, pr))
        @test is_plot(plot_risk_contribution(r_cvr, res, rd))
        @test is_plot(plot_risk_contribution(r_cvr, res, pr))
        @test is_plot(plot_drawdowns(res, rd))
        @test is_plot(plot_histogram(res, rd))
        @test is_plot(plot_rolling_measure(r_cvr, res, rd))
        @test is_plot(plot_prior(res))
        @test is_plot(plot_prior(res, rd))
        # default r=Variance() has no sigma; pass CVaR which computes from returns
        @test is_plot(plot_portfolio_dashboard(res, rd; r = r_cvr))
        @test is_plot(plot_measures([res], pr))
    end

    @testset "Cross-validation dispatch" begin
        mr    = MeanRisk(; opt = JuMPOptimiser(; slv = slv))
        mpred = cross_val_predict(mr, rd, IndexWalkForward(80, 40))

        @test is_plot(plot_portfolio_cumulative_returns(mpred))
        @test is_plot(plot_drawdowns(mpred))
        @test is_plot(plot_histogram(mpred))
        @test is_plot(plot_rolling_measure(r_cvr, mpred))
        @test is_plot(plot_composition(mpred))
        @test is_plot(plot_weight_stability(mpred))
        @test is_plot(plot_turnover(mpred))
        @test is_plot(plot_cv_dashboard(mpred))
    end

    @testset "plot_efficient_frontier" begin
        # Frontier result (VecVecNum weights)
        mr_f  = MeanRisk(; opt = JuMPOptimiser(; slv = slv, ret = ArithmeticReturn(; settings = JuMPReturnsSettings(; lb = Frontier(; N = 8)))))
        res_f = optimise(mr_f, rd)
        @test is_plot(plot_efficient_frontier(res_f, rd))
        @test is_plot(plot_efficient_frontier(res_f, pr))
        @test is_plot(plot_efficient_frontier(res_f, rd; min_risk = false,
                                              max_score = false))
        # Vector-of-results form
        mr1  = MeanRisk(; opt = JuMPOptimiser(; slv = slv))
        res1 = optimise(mr1, rd)
        mr2  = MeanRisk(; opt = JuMPOptimiser(; slv = slv), r = ConditionalValueatRisk())
        res2 = optimise(mr2, rd)
        @test is_plot(plot_efficient_frontier([res1, res2], rd))
    end

    @testset "plot_performance_summary" begin
        @test is_plot(plot_performance_summary(w, X))
        @test is_plot(plot_performance_summary(w_rd, rd))
        mr_p  = MeanRisk(; opt = JuMPOptimiser(; slv = slv))
        res_p = optimise(mr_p, rd)
        @test is_plot(plot_performance_summary(res_p, rd))
        mpred_p = cross_val_predict(mr_p, rd, IndexWalkForward(80, 40))
        @test is_plot(plot_performance_summary(mpred_p))
    end

    @testset "plot_rolling_drawdowns" begin
        @test is_plot(plot_rolling_drawdowns(w, X))
        @test is_plot(plot_rolling_drawdowns(w_rd, rd))
        @test is_plot(plot_rolling_drawdowns(w, X; rolling = 20))
        @test is_plot(plot_rolling_drawdowns(w, X; compound = true))
        mr_d  = MeanRisk(; opt = JuMPOptimiser(; slv = slv))
        res_d = optimise(mr_d, rd)
        @test is_plot(plot_rolling_drawdowns(res_d, rd))
        mpred_d = cross_val_predict(mr_d, rd, IndexWalkForward(80, 40))
        @test is_plot(plot_rolling_drawdowns(mpred_d))
    end

    @testset "Error dispatch – unsupported PredictionResult" begin
        mr   = MeanRisk(; opt = JuMPOptimiser(; slv = slv))
        raw  = cross_val_predict(mr, rd, KFold(; n = 2)).pred
        pred = isa(raw[1], PredictionResult) ? raw[1] : raw[1].pred[1]

        @test is_plot(plot_asset_cumulative_returns(pred))
        @test_throws ArgumentError plot_risk_contribution(r_cvr, pred)
        @test_throws ArgumentError plot_factor_risk_contribution(r_cvr, pred)
    end
    @testset "A drifted fold plots its risk contributions (#769)" begin
        # The blanket refusal above is lifted for a fold that carries a Held Weights
        # record: that record keeps the fold's asset returns, which is exactly what the
        # refusal said a `PredictionResult` had lost.
        mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv))
        raw = cross_val_predict(mr, rd, KFold(; n = 2, wd = SelfFinancingDrift())).pred
        pred = isa(raw[1], PredictionResult) ? raw[1] : raw[1].pred[1]

        @test !isnothing(pred.hw)
        @test is_plot(plot_risk_contribution(r_cvr, pred))
        @test is_plot(plot_factor_risk_contribution(r_cvr, pred))
        @test is_plot(plot_risk_contribution(r_cvr, pred; percentage = false))
    end
    @testset "Cross-sectional regression diagnostics (#798)" begin
        # Each figure draws one level-2 verb. The block is built by hand rather than fitted,
        # because the diagnostics read only the histories the block carries.
        rng_cs = MersenneTwister(798)
        Tc, Nc, Kc = 10, 8, 3
        Ms_cs = randn(rng_cs, Tc, Nc, Kc)
        csr_cs = CrossSectionalRegression(; f = 0.02 * randn(rng_cs, Tc, Kc),
                                          eps = 0.01 * randn(rng_cs, Tc, Nc),
                                          n = fill(Nc, Tc))
        csfm_cs = CrossSectionalFactorModel(; M = Ms_cs[Tc, :, :], b = zeros(Nc),
                                            csr = csr_cs, Ms = Ms_cs,
                                            rw = abs.(randn(rng_cs, Tc, Nc)) .+ 0.1,
                                            nf = ["value", "size", "momentum"], lag = 1)
        # `rr` and `fpr` are the factor block and travel together, so the factor prior is
        # built alongside the loadings even though these figures never read it.
        fpr_cs = LowOrderPrior(; X = randn(rng_cs, Tc, Kc), mu = zeros(Kc),
                               sigma = Matrix(1.0 * I, Kc, Kc))
        pr_cs = LowOrderPrior(; X = randn(rng_cs, Tc, Nc), mu = zeros(Nc),
                              sigma = Matrix(1.0 * I, Nc, Nc), rr = csfm_cs, fpr = fpr_cs)

        for plt in
            (plot_cs_regression_r2, plot_cs_regression_adjusted_r2, plot_cs_regression_aic,
             plot_cs_regression_bic, plot_exposure_condition_number)
            @test is_plot(plt(csfm_cs))
            @test is_plot(plt(pr_cs))
        end
        for plt in (plot_cs_regression_t_stats, plot_exposure_vif)
            @test is_plot(plt(csfm_cs))
            @test is_plot(plt(pr_cs))
            @test is_plot(plt(csfm_cs; nf = ["a", "b", "c"]))
        end
        @test is_plot(plot_cs_regression_t_stat_exceedance_rate(csfm_cs))
        @test is_plot(plot_cs_regression_t_stat_exceedance_rate(pr_cs))
        @test is_plot(plot_cs_regression_t_stat_exceedance_rate(csfm_cs; threshold = 1,
                                                                nf = ["a", "b", "c"]))
        # A block that names no factor is labelled by position.
        bare = CrossSectionalFactorModel(; M = Ms_cs[Tc, :, :], b = zeros(Nc), csr = csr_cs,
                                         Ms = Ms_cs, lag = 1)
        @test is_plot(plot_exposure_vif(bare))
        # A prior result that carries no factor block names the remedy.
        no_rr = LowOrderPrior(; X = randn(rng_cs, Tc, Nc), mu = zeros(Nc),
                              sigma = Matrix(1.0 * I, Nc, Nc))
        @test_throws PortfolioOptimisers.IsNothingError plot_exposure_vif(no_rr)
        @test_throws PortfolioOptimisers.IsNothingError plot_cs_regression_r2(no_rr)
    end
    @testset "The four factor attribution plots" begin
        # A hand-built factor model block, so the plots are exercised without a panel fit. The two
        # factors sit in two families, and the block keeps every history the realised attribution
        # and its standard errors read.
        rng_fa = StableRNG(782_101)
        Ms_fa = Array{Float64, 3}(undef, 4, 3, 2)
        for t in 1:4
            Ms_fa[t, :, :] = [1.0 0.5+0.1*t; 1.0 -0.5; 1.0 1.5]
        end
        f_fa = randn(rng_fa, 4, 2) ./ 50
        eps_fa = randn(rng_fa, 4, 3) ./ 200
        csr_fa = CrossSectionalRegression(; f = f_fa, eps = eps_fa, n = [3, 3, 3, 3])
        rw_fa = fill(1 / 3, 4, 3)
        vs_fa = fill(1.0e-4, 4, 3)
        rr_fa = CrossSectionalFactorModel(; M = Ms_fa[4, :, :], b = [0.001, 0.0005, 0.0012],
                                          csr = csr_fa, Ms = Ms_fa, vs = vs_fa,
                                          esigma = [1.2e-4, 1.8e-4, 1.6e-4], rw = rw_fa,
                                          bw = rw_fa, nf = ["market", "value"],
                                          fam = ["market", "style"], lag = 0)
        fpr_fa = LowOrderPrior(; X = f_fa, mu = vec(mean(f_fa; dims = 1)),
                               sigma = cov(f_fa))
        X_fa = f_fa * transpose(rr_fa.M) .+ eps_fa
        mu_fa = rr_fa.M * fpr_fa.mu .+ rr_fa.b
        sigma_fa = rr_fa.M * fpr_fa.sigma * transpose(rr_fa.M) + Diagonal(rr_fa.esigma)
        pr_fa = LowOrderPrior(; X = X_fa, mu = mu_fa, sigma = sigma_fa, rr = rr_fa,
                              fpr = fpr_fa)
        w_fa = [0.5, 0.3, 0.2]
        fa_r = factor_attribution(w_fa, pr_fa, X_fa; se = true)
        fa_p = factor_attribution(w_fa, pr_fa)
        rd_fa = ReturnsResult(; nx = ["a", "b", "c"], X = X_fa, nf = ["market", "value"],
                              F = f_fa)
        for plt in (plot_attribution_vol_contrib, plot_attribution_mu_contrib,
                    plot_attribution_exposure, plot_attribution_mu_vs_vol)
            # Both axes, both sides, with and without a row limit.
            @test is_plot(plt(fa_r))
            @test is_plot(plt(fa_p))
            @test is_plot(plt(fa_r; by_family = true))
            @test is_plot(plt(fa_p; by_family = true))
            @test is_plot(plt(fa_r; N = 1))
            @test is_plot(plt(fa_r; rd = rd_fa))
            @test is_plot(plt(fa_r; nf = ["one", "two"]))
            @test_throws DomainError plt(fa_r; N = 0)
        end
        @test_throws DomainError plot_attribution_mu_contrib(fa_r; z = -1)
        # A block that names no family has no family axis to draw.
        no_fam = CrossSectionalFactorModel(; M = rr_fa.M, b = rr_fa.b, csr = csr_fa,
                                           Ms = Ms_fa, esigma = rr_fa.esigma, lag = 0)
        pr_nf = LowOrderPrior(; X = X_fa, mu = mu_fa, sigma = sigma_fa, rr = no_fam,
                              fpr = fpr_fa)
        fa_nf = factor_attribution(w_fa, pr_nf)
        @test isnothing(fa_nf.fmbd)
        for plt in (plot_attribution_vol_contrib, plot_attribution_mu_contrib,
                    plot_attribution_exposure, plot_attribution_mu_vs_vol)
            @test_throws ArgumentError plt(fa_nf; by_family = true)
        end
    end
    @testset "Cross-sectional exposure diagnostics (#799)" begin
        # Each figure draws one level-2 verb. The block carries a benchmark weight history,
        # because the default weighting of the exposure group reads it.
        rng_ex = MersenneTwister(799)
        Te, Ne, Ke = 12, 8, 3
        Ms_ex = randn(rng_ex, Te, Ne, Ke)
        csr_ex = CrossSectionalRegression(; f = 0.02 * randn(rng_ex, Te, Ke),
                                          eps = 0.01 * randn(rng_ex, Te, Ne),
                                          n = fill(Ne, Te))
        csfm_ex = CrossSectionalFactorModel(; M = Ms_ex[Te, :, :], b = zeros(Ne),
                                            csr = csr_ex, Ms = Ms_ex,
                                            rw = abs.(randn(rng_ex, Te, Ne)) .+ 0.1,
                                            bw = fill(1 / Ne, Te, Ne),
                                            vs = abs.(randn(rng_ex, Te, Ne)) .+ 0.5,
                                            nf = ["value", "size", "momentum"], lag = 1)
        fpr_ex = LowOrderPrior(; X = randn(rng_ex, Te, Ke), mu = zeros(Ke),
                               sigma = Matrix(1.0 * I, Ke, Ke))
        pr_ex = LowOrderPrior(; X = randn(rng_ex, Te, Ne), mu = zeros(Ne),
                              sigma = Matrix(1.0 * I, Ne, Ne), rr = csfm_ex, fpr = fpr_ex)

        for plt in (plot_exposure_correlation, plot_exposure_dispersion)
            @test is_plot(plt(csfm_ex))
            @test is_plot(plt(pr_ex))
            @test is_plot(plt(csfm_ex; nf = ["a", "b", "c"]))
            @test is_plot(plt(csfm_ex; weighting = IdentityMetric()))
        end
        @test is_plot(plot_exposure_stability(csfm_ex; step = 3))
        @test is_plot(plot_exposure_stability(pr_ex; step = 3))
        @test is_plot(plot_exposure_stability(csfm_ex; step = 3, nf = ["a", "b", "c"],
                                              weighting = RegressionWeightMetric()))
        @test is_plot(plot_cumulative_exposure_ic(csfm_ex))
        @test is_plot(plot_cumulative_exposure_ic(pr_ex))
        @test is_plot(plot_cumulative_exposure_ic(csfm_ex; rank = false,
                                                  nf = ["a", "b", "c"]))
        @test is_plot(plot_exposure_distribution(csfm_ex))
        @test is_plot(plot_exposure_distribution(pr_ex; factor = 2))
        @test is_plot(plot_exposure_distribution(csfm_ex; factor = 3, observation = 4,
                                                 nf = ["a", "b", "c"]))
        # A block that names no factor is labelled by position.
        bare_ex = CrossSectionalFactorModel(; M = Ms_ex[Te, :, :], b = zeros(Ne),
                                            csr = csr_ex, Ms = Ms_ex, lag = 1)
        @test is_plot(plot_exposure_correlation(bare_ex; weighting = IdentityMetric()))
        @test is_plot(plot_exposure_distribution(bare_ex))
        # A re-based block labels the cumulative information coefficient on the reduced
        # axis when it is asked for the reduced answer.
        fcb_ex = FactorFamilyBasis(; fnm = ["industry"], fi = [[1, 2]], di = [2],
                                   ratios = reshape(collect(range(0.4, 0.9; length = Te)),
                                                    Te, 1), K = Ke)
        reb_ex = CrossSectionalFactorModel(; M = Ms_ex[Te, :, :],
                                           L = PortfolioOptimisers.reduce_loadings(fcb_ex,
                                                                                   Ms_ex[Te,
                                                                                         :,
                                                                                         :]),
                                           b = zeros(Ne), csr = csr_ex, Ms = Ms_ex,
                                           rw = abs.(randn(rng_ex, Te, Ne)) .+ 0.1,
                                           fcb = fcb_ex, nf = ["value", "size", "momentum"],
                                           lag = 1)
        @test is_plot(plot_cumulative_exposure_ic(reb_ex; reduced = true))
        # A prior result that carries no factor block names the remedy.
        no_rr_ex = LowOrderPrior(; X = randn(rng_ex, Te, Ne), mu = zeros(Ne),
                                 sigma = Matrix(1.0 * I, Ne, Ne))
        @test_throws PortfolioOptimisers.IsNothingError plot_exposure_correlation(no_rr_ex)
        @test_throws PortfolioOptimisers.IsNothingError plot_exposure_distribution(no_rr_ex)
    end
    @testset "The factor model summary and the factor forecast plots (#801)" begin
        # The summary figure draws the Result, and the two forecast figures draw the factor
        # covariance the prior carries.
        rng_fs = MersenneTwister(801)
        Tf, Nf, Kf = 12, 8, 3
        Ms_fs = randn(rng_fs, Tf, Nf, Kf)
        csr_fs = CrossSectionalRegression(; f = 0.02 * randn(rng_fs, Tf, Kf),
                                          eps = 0.01 * randn(rng_fs, Tf, Nf),
                                          n = fill(Nf, Tf))
        csfm_fs = CrossSectionalFactorModel(; M = Ms_fs[Tf, :, :], b = zeros(Nf),
                                            csr = csr_fs, Ms = Ms_fs,
                                            rw = abs.(randn(rng_fs, Tf, Nf)) .+ 0.1,
                                            bw = fill(1 / Nf, Tf, Nf),
                                            nf = ["value", "size", "momentum"], lag = 1)
        f_sigma_fs = Matrix(1.0 * I, Kf, Kf)
        f_sigma_fs[1, 2] = f_sigma_fs[2, 1] = 0.3
        fpr_fs = LowOrderPrior(; X = randn(rng_fs, Tf, Kf), mu = zeros(Kf),
                               sigma = f_sigma_fs)
        pr_fs = LowOrderPrior(; X = randn(rng_fs, Tf, Nf), mu = zeros(Nf),
                              sigma = Matrix(1.0 * I, Nf, Nf), rr = csfm_fs, fpr = fpr_fs)

        fs = factor_model_summary(csfm_fs; ppy = 252, step = 3)
        @test is_plot(plot_factor_model_summary(fs))
        @test is_plot(plot_factor_model_summary(fs; nf = ["a", "b", "c"]))
        @test is_plot(plot_factor_model_summary(csfm_fs; ppy = 252, step = 3))
        @test is_plot(plot_factor_model_summary(pr_fs; ppy = 252, step = 3))
        @test is_plot(plot_factor_model_summary(csfm_fs; ppy = 252, step = 3,
                                                nf = ["a", "b", "c"],
                                                weighting = IdentityMetric(),
                                                coverage_weighting = IdentityMetric()))
        # A block with no exposure history draws its four factor return columns alone.
        bare_fs = CrossSectionalFactorModel(; M = Ms_fs[Tf, :, :], b = zeros(Nf),
                                            csr = csr_fs, lag = 1)
        @test is_plot(plot_factor_model_summary(bare_fs; ppy = 252))

        @test is_plot(plot_factor_forecast_correlation(pr_fs))
        @test is_plot(plot_factor_forecast_correlation(pr_fs, ["a", "b", "c"]))
        @test is_plot(plot_factor_forecast_correlation(f_sigma_fs))
        @test is_plot(plot_factor_forecast_volatilities(pr_fs))
        @test is_plot(plot_factor_forecast_volatilities(pr_fs, ["a", "b", "c"]; ppy = 252))
        @test is_plot(plot_factor_forecast_volatilities(f_sigma_fs))

        @test is_plot(plot_factor_cumulative_returns(csfm_fs))
        @test is_plot(plot_factor_cumulative_returns(pr_fs))
        @test is_plot(plot_factor_cumulative_returns(csfm_fs; compound = true,
                                                     nf = ["a", "b", "c"]))
        # An absent factor return contributes nothing to the running sum.
        f_gap = copy(csr_fs.f)
        f_gap[4, 2] = NaN
        gap_fs = CrossSectionalFactorModel(; M = Ms_fs[Tf, :, :], b = zeros(Nf),
                                           csr = CrossSectionalRegression(; f = f_gap,
                                                                          eps = csr_fs.eps,
                                                                          n = csr_fs.n),
                                           Ms = Ms_fs, lag = 1)
        @test is_plot(plot_factor_cumulative_returns(gap_fs))
        # A prior result that carries no factor block names the remedy.
        no_rr_fs = LowOrderPrior(; X = randn(rng_fs, Tf, Nf), mu = zeros(Nf),
                                 sigma = Matrix(1.0 * I, Nf, Nf))
        @test_throws PortfolioOptimisers.IsNothingError plot_factor_model_summary(no_rr_fs)
        @test_throws PortfolioOptimisers.IsNothingError plot_factor_forecast_correlation(no_rr_fs)
        @test_throws PortfolioOptimisers.IsNothingError plot_factor_forecast_volatilities(no_rr_fs)
        @test_throws PortfolioOptimisers.IsNothingError plot_factor_cumulative_returns(no_rr_fs)
    end
    @testset "Cross-sectional idiosyncratic diagnostics (#800)" begin
        # Each figure draws one level-2 verb. The group reads `csr.eps` and `vs` alone, so
        # the block carries no exposure history and no weights.
        rng_id = MersenneTwister(800)
        Ti, Ni, Ki = 12, 8, 3
        vs_id = abs.(randn(rng_id, Ti, Ni)) .+ 0.5
        csr_id = CrossSectionalRegression(; f = 0.02 * randn(rng_id, Ti, Ki),
                                          eps = sqrt.(vs_id) .* randn(rng_id, Ti, Ni),
                                          n = fill(Ni, Ti))
        csfm_id = CrossSectionalFactorModel(; M = randn(rng_id, Ni, Ki), b = zeros(Ni),
                                            csr = csr_id, vs = vs_id, lag = 1)
        fpr_id = LowOrderPrior(; X = randn(rng_id, Ti, Ki), mu = zeros(Ki),
                               sigma = Matrix(1.0 * I, Ki, Ki))
        pr_id = LowOrderPrior(; X = randn(rng_id, Ti, Ni), mu = zeros(Ni),
                              sigma = Matrix(1.0 * I, Ni, Ni), rr = csfm_id, fpr = fpr_id)
        for plt in (plot_idio_calibration, plot_idio_kurtosis, plot_idio_skewness,
                    plot_idio_vol_ic, plot_idio_vol_residual_dependence)
            @test is_plot(plt(csfm_id))
            @test is_plot(plt(pr_id))
        end
        @test is_plot(plot_idio_tail_rate(csfm_id))
        @test is_plot(plot_idio_tail_rate(pr_id))
        @test is_plot(plot_idio_tail_rate(csfm_id; threshold = 2))
        # A block that carries no idiosyncratic variance history names the field.
        no_vs_id = CrossSectionalFactorModel(; M = randn(rng_id, Ni, Ki), b = zeros(Ni),
                                             csr = csr_id, lag = 1)
        @test_throws PortfolioOptimisers.IsNothingError plot_idio_calibration(no_vs_id)
        # A prior result that carries no factor block names the remedy.
        no_rr_id = LowOrderPrior(; X = randn(rng_id, Ti, Ni), mu = zeros(Ni),
                                 sigma = Matrix(1.0 * I, Ni, Ni))
        @test_throws PortfolioOptimisers.IsNothingError plot_idio_calibration(no_rr_id)
        @test_throws PortfolioOptimisers.IsNothingError plot_idio_vol_ic(no_rr_id)
    end

    @testset "A drawn plot keeps the frame, a computed plot reduces (#857)" begin
        # ADR 0118's plotting half. A heatmap or a bar chart of a Prior Result draws the
        # full universe and leaves a blank for a non-investable asset; a plot that computes
        # on the prior's blocks reduces to the Investable Mask first, because `eigvals` and
        # a plain moment estimator both refuse a `NaN`.
        #
        # The oracle for a computed plot is the same figure over the universe with the dead
        # asset removed **by hand**, which is the oracle `test/test_50_investable_reduction.jl`
        # uses for the optimiser side.
        rng_pv = MersenneTwister(857)
        T_pv, N_pv = 160, 5
        X_pv = randn(rng_pv, T_pv, N_pv) ./ 100 .+ 0.0005
        nx_pv = ["a", "b", "c", "d", "e"]
        k_pv, keep_pv = 3, [1, 2, 4, 5]
        rd_pv = ReturnsResult(; nx = nx_pv, X = X_pv)
        pr_pv = prior(EmpiricalPrior(), rd_pv)
        hpr_pv = prior(HighOrderPriorEstimator(), rd_pv)

        mu_pv = collect(pr_pv.mu)
        sigma_pv = collect(pr_pv.sigma)
        Xn_pv = collect(pr_pv.X)
        mu_pv[k_pv] = NaN
        sigma_pv[k_pv, :] .= NaN
        sigma_pv[:, k_pv] .= NaN
        Xn_pv[:, k_pv] .= NaN
        prn_pv = LowOrderPrior(; X = Xn_pv, mu = mu_pv, sigma = sigma_pv)
        prk_pv = LowOrderPrior(; X = pr_pv.X[:, keep_pv], mu = pr_pv.mu[keep_pv],
                               sigma = pr_pv.sigma[keep_pv, keep_pv])

        skn_pv = collect(hpr_pv.sk)
        ktn_pv = collect(hpr_pv.kt)
        skn_pv[k_pv, :] .= NaN
        ktn_pv[k_pv, :] .= NaN
        ktn_pv[:, k_pv] .= NaN
        hprn_pv = HighOrderPrior(; pr = prn_pv, kt = ktn_pv, D2 = hpr_pv.D2, L2 = hpr_pv.L2,
                                 S2 = hpr_pv.S2, sk = skn_pv, V = hpr_pv.V,
                                 skmp = hpr_pv.skmp)
        hprk_pv = PortfolioOptimisers.port_opt_view(hpr_pv, keep_pv)

        w_pv = [0.4, 0.3, 0.0, 0.2, 0.1]
        # A bar series is a polygon whose vertices are separated by `NaN`, so two figures are
        # compared with `isequal` rather than `==`.
        sy_pv(p, i = 1) = p.series_list[i][:y]
        same_pv(a, b) = length(a) == length(b) && all(isequal.(a, b))

        # ── the three ranking and colour rules ────────────────────────────────────────
        @test PortfolioOptimisers.finite_magnitudes([1.0, -2.0, NaN, 0.5]) ==
              [1.0, 2.0, 0.0, 0.5]
        @test PortfolioOptimisers.finite_magnitudes([1.0, -2.0]) == [1.0, 2.0]
        @test PortfolioOptimisers.finite_symmetric_clim([1.0 NaN; -3.0 2.0]) == (-3.0, 3.0)
        @test PortfolioOptimisers.finite_columns([1.0 NaN 3.0; 4.0 5.0 6.0], [1, 2, 3]) ==
              [1, 3]
        # The blank never takes a top slot from a live asset, and it ranks last.
        N_rel, idx_rel = PortfolioOptimisers.relevant_assets([0.4, 0.3, NaN, 0.2, 0.1], 5,
                                                             3)
        @test N_rel == 3
        @test idx_rel == [1, 2, 4, 5, 3]

        # ── the reduction verb ────────────────────────────────────────────────────────
        prv_pv, nxv_pv, wv_pv = PortfolioOptimisers.investable_plot_view(prn_pv, nx_pv,
                                                                         w_pv)
        @test length(prv_pv.mu) == length(keep_pv)
        @test collect(nxv_pv) == nx_pv[keep_pv]
        @test collect(wv_pv) == w_pv[keep_pv]
        # A complete prior and a returns result are the identity, objects included.
        @test PortfolioOptimisers.investable_plot_view(pr_pv, nx_pv, w_pv) ===
              (pr_pv, nx_pv, w_pv)
        @test PortfolioOptimisers.investable_plot_view(rd_pv, nx_pv, w_pv) ===
              (rd_pv, nx_pv, w_pv)

        # ── piece 1: the drawn plots keep the frame ───────────────────────────────────
        for p_pv in (plot_mu(prn_pv, nx_pv), plot_mu(prn_pv, nx_pv; N = 3),
                     plot_sigma(prn_pv, nx_pv), plot_sigma(prn_pv, nx_pv; N = 3),
                     plot_correlation(prn_pv, nx_pv), plot_prior(prn_pv, nx_pv),
                     plot_coskewness(hprn_pv, nx_pv),
                     plot_cokurtosis(hprn_pv, nx_pv; heatmap = true))
            @test is_plot(p_pv)
        end
        # The frame is the full universe: the axis carries every asset, gapped or not. The
        # bar itself is missing, which is what the backend draws at a `NaN`, so the gapped
        # figure holds fewer polygon vertices than the complete one.
        @test plot_mu(prn_pv, nx_pv)[1][:xaxis][:ticks][2] == nx_pv
        @test plot_sigma(prn_pv, nx_pv)[1][:xaxis][:ticks][2] == nx_pv
        @test length(sy_pv(plot_mu(prn_pv, nx_pv))) < length(sy_pv(plot_mu(pr_pv, nx_pv)))
        @test length(sy_pv(plot_sigma(prn_pv, nx_pv))) <
              length(sy_pv(plot_sigma(pr_pv, nx_pv)))
        # A blank never takes a top slot, so a truncated frame holds live assets alone.
        @test nx_pv[k_pv] ∉ plot_mu(prn_pv, nx_pv; N = 3)[1][:xaxis][:ticks][2]
        @test nx_pv[k_pv] ∉ plot_sigma(prn_pv, nx_pv; N = 3)[1][:xaxis][:ticks][2]
        @test size(plot_correlation(prn_pv, nx_pv).series_list[1][:z]) == (N_pv, N_pv)
        @test size(plot_coskewness(hprn_pv, nx_pv).series_list[1][:z]) == (N_pv, N_pv^2)
        @test size(plot_cokurtosis(hprn_pv, nx_pv; heatmap = true).series_list[1][:z]) ==
              (N_pv^2, N_pv^2)
        # A gapped loadings row leaves a blank, and the colour limit stays finite.
        M_pv = randn(rng_pv, N_pv, 3)
        M_pv[k_pv, :] .= NaN
        nf_pv = ["f1", "f2", "f3"]
        @test is_plot(plot_factor_loadings(M_pv, nx_pv, nf_pv))
        @test all(isfinite, PortfolioOptimisers.finite_symmetric_clim(M_pv))

        # ── piece 2: the computed plots reduce ────────────────────────────────────────
        cte_pv = CentralityEstimator()
        ne_pv = NetworkEstimator()
        @test same_pv(sy_pv(plot_eigenspectrum(prn_pv)), sy_pv(plot_eigenspectrum(prk_pv)))
        @test !same_pv(sy_pv(plot_eigenspectrum(prn_pv)), sy_pv(plot_eigenspectrum(pr_pv)))
        @test same_pv(sy_pv(plot_eigenspectrum(prn_pv, rd_pv)),
                      sy_pv(plot_eigenspectrum(prk_pv, rd_pv)))
        @test same_pv(sy_pv(plot_centrality(cte_pv, prn_pv, nx_pv)),
                      sy_pv(plot_centrality(cte_pv, prk_pv, nx_pv[keep_pv])))
        @test same_pv(sy_pv(plot_cokurtosis(hprn_pv, nx_pv)),
                      sy_pv(plot_cokurtosis(hprk_pv, nx_pv[keep_pv])))
        # The returns-result arity takes both sides through the same door.
        @test same_pv(sy_pv(plot_cokurtosis(hprn_pv, rd_pv)),
                      sy_pv(plot_cokurtosis(hprn_pv, nx_pv)))
        @test is_plot(plot_cokurtosis(hprn_pv, rd_pv; heatmap = true))
        @test is_plot(plot_network(ne_pv, prn_pv, w_pv))
        @test is_plot(plot_network(ne_pv, prn_pv))

        # ── piece 4: the "rest" aggregate excludes what it cannot value ───────────────
        p_ac_pv = plot_asset_cumulative_returns(w_pv, prn_pv; N = 2)
        @test all(s_pv -> all(isfinite, s_pv[:y]), p_ac_pv.series_list)

        # ── piece 3: a prediction is finite by the fold's filter ──────────────────────
        Xd_pv = copy(X_pv)
        Xd_pv[:, k_pv] .= NaN
        Xd_pv[140:end, 5] .= NaN
        rdd_pv = ReturnsResult(; nx = nx_pv, X = Xd_pv)
        res_pv = optimise(MeanRisk(; opt = JuMPOptimiser(; pe = prn_pv, slv = slv)), rd_pv)
        pred_pv = predict(res_pv, rdd_pv, collect(130:T_pv))
        @test all(isfinite, sy_pv(plot_portfolio_cumulative_returns(pred_pv)))
        @test is_plot(plot_drawdowns(pred_pv))

        # The optimisation-result arity of `plot_network` resolved its own prior through a
        # local that was not yet defined, so it raised an `UndefVarError` on every call.
        # Behind that stood a second defect: the result carries the prior of the universe it
        # solved, which ADR 0115 reduced, and weights the solution expanded onto the full
        # one, so the two axes must be paired. Issue #884 owns the rest of that class.
        @test PortfolioOptimisers.result_investable_mask(res_pv) ==
              BitVector([1, 1, 0, 1, 1])
        @test is_plot(plot_network(ne_pv, res_pv))
        @test is_plot(plot_network(ne_pv, res_pv; rd = rd_pv))
    end
end
