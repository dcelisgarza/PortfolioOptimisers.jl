@safetestset "Plotting" begin
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

    @testset "PlottingOptions" begin
        opts = PlottingOptions()
        @test opts.compound == false
        @test opts.alpha == 0.05
        @test opts.kappa == 0.3
        @test opts.flag == true
        @test opts.points == 0
        @test opts.rolling == 0

        @test PlottingOptions(; alpha = 0.1).alpha == 0.1
        @test_throws Exception PlottingOptions(; alpha = 0.0)
        @test_throws Exception PlottingOptions(; alpha = 1.0)
        @test_throws Exception PlottingOptions(; kappa = 0.0)
        @test_throws Exception PlottingOptions(; delta = 0.0)
        @test_throws Exception PlottingOptions(; points = -1)
        @test_throws Exception PlottingOptions(; rolling = -1)
        @test_throws Exception PlottingOptions(; N = -1.0)
    end

    @testset "plot_ptf_cumulative_returns" begin
        @test is_plot(plot_ptf_cumulative_returns(w, X))
        @test is_plot(plot_ptf_cumulative_returns(w, X; ts = 1:T))
        @test is_plot(plot_ptf_cumulative_returns(w, X;
                                                  opts = PlottingOptions(; compound = true)))
        @test is_plot(plot_ptf_cumulative_returns(w_rd, rd))
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
    end

    @testset "plot_drawdowns" begin
        @test is_plot(plot_drawdowns(w, X))
        @test is_plot(plot_drawdowns(w_rd, rd))
        @test is_plot(plot_drawdowns(w, X; opts = PlottingOptions(; compound = true)))
    end

    @testset "plot_histogram" begin
        @test is_plot(plot_histogram(w, X))
        @test is_plot(plot_histogram(w, X; opts = PlottingOptions(; flag = false)))
        @test is_plot(plot_histogram(w_rd, rd))
    end

    @testset "plot_network" begin
        ne = NetworkEstimator()
        @test is_plot(plot_network(ne, X, nx))
    end

    @testset "plot_centrality" begin
        cte = CentralityEstimator()
        @test is_plot(plot_centrality(cte, X, nx))
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
        @test is_plot(plot_sigma(sigma; opts = PlottingOptions(; percentage = true)))
    end

    @testset "plot_factor_loadings" begin
        @test is_plot(plot_factor_loadings(fpr))
        @test is_plot(plot_factor_loadings(fpr, rd.nx, rd.nf))
    end

    @testset "plot_factor_sigma" begin
        @test is_plot(plot_factor_sigma(fpr.f_sigma, rd.nf))
    end

    @testset "plot_eigenspectrum" begin
        @test is_plot(plot_eigenspectrum(sigma))
        @test is_plot(plot_eigenspectrum(sigma; N_obs = T,
                                         opts = PlottingOptions(; flag = true)))
    end

    @testset "plot_rolling_measure" begin
        # use return-based risk measure (CVaR), not covariance-based (Variance)
        @test is_plot(plot_rolling_measure(r_cvr, w, X))
        @test is_plot(plot_rolling_measure(r_cvr, w, X;
                                           opts = PlottingOptions(; rolling = 20)))
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
        @test is_plot(plot_cokurtosis(kt; opts = PlottingOptions(; percentage = true)))
    end

    @testset "plot_prior" begin
        @test is_plot(plot_prior(pr))
        @test is_plot(plot_prior(pr, rd))
    end

    @testset "Optimisation-based dispatch" begin
        mr  = MeanRisk(; opt = JuMPOptimiser(; slv = slv))
        res = optimise(mr, rd)
        @test !isnothing(res)

        @test is_plot(plot_ptf_cumulative_returns(res, rd))
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
        @test is_plot(plot_measures([res], rd))
    end

    @testset "Cross-validation dispatch" begin
        mr    = MeanRisk(; opt = JuMPOptimiser(; slv = slv))
        mpred = cross_val_predict(mr, rd, IndexWalkForward(80, 40))

        @test is_plot(plot_ptf_cumulative_returns(mpred))
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
        mr_f  = MeanRisk(; opt = JuMPOptimiser(; slv = slv, ret = ArithmeticReturn(; lb = Frontier(; N = 8))))
        res_f = optimise(mr_f, rd)
        @test is_plot(plot_efficient_frontier(res_f, rd))
        @test is_plot(plot_efficient_frontier(res_f, pr))
        @test is_plot(plot_efficient_frontier(res_f, rd; annotate_minrisk = false,
                                              annotate_maxsharpe = false))
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
        @test is_plot(plot_rolling_drawdowns(w, X; opts = PlottingOptions(; rolling = 20)))
        @test is_plot(plot_rolling_drawdowns(w, X;
                                             opts = PlottingOptions(; compound = true)))
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

        @test_throws ArgumentError plot_asset_cumulative_returns(pred)
        @test_throws ArgumentError plot_risk_contribution(r_cvr, pred)
        @test_throws ArgumentError plot_factor_risk_contribution(r_cvr, pred)
    end
end
