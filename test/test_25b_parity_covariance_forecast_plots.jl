#=
The figures of the covariance forecast evaluation (#1405, map #1375).

Each figure is the rolling form of `covariance_forecast_summary`: point t reads the steps of its
window that scored, with the weights the summary gives them. So a window as wide as the run ends
at the number of the summary, and the first testset pins that identity on every series.

THE PARITY. Every stored case under
`test/assets/Parity_CovarianceForecastPlots_<Case>_<Figure>.csv.gz` holds the traces of one oracle
figure: the first column is the first test row of each point, and each next column is one trace.
A band is two traces, the upper edge and then the lower edge, after its line. The oracle drew
each figure from our own per-step diagnostics, which `test_24e_parity_covariance_forecast_evaluation.jl`
pins against it. So these cases measure the window and the band, and nothing before them. The
oracle leaves out the first `window - 1` points, where we draw a blank `NaN`, so each case compares
the points the oracle drew. Every point compares at `rtol = 1e-12`, measured 2.6e-15 at most.

THE MAHALANOBIS RATIO WITH A LISTING (Deliberate difference, the oracle's rule one keyword away,
#1513). The oracle takes the plain mean of the per-step ratios in a window. We take the mean
weighted by the degrees of freedom of each step, as the summary does. The two agree while every step has the same active count and
horizon. They differ when an asset lists or delists (`Portfolios`). Under a Gaussian null
`m_t` is chi-squared on `ν_t` over `ν_t`, so its variance is `2 / ν_t`. The weights `ν_t` are the
inverse variances, so the weighted mean is the unbiased linear combination of least variance,
and it is itself chi-squared on `Σ ν_t` over `Σ ν_t`. The plain mean has a larger variance and no
chi-squared law. The test pins our weighted series, and shows that the plain mean of the same
steps is the oracle's series, so the weights are the whole difference. `step_weighting =
EqualStepWeighting()` draws the plain mean, and gives the oracle's summary of the same steps
(`Parity_CovarianceForecastPlots_Portfolios_Summary`: the rows Mahalanobis ratio, diagonal
ratio, standardised return and portfolio QLIKE of each evaluation, the columns mean, median,
standard deviation, p5 and p95).

THE STEP WITH NO ACTIVE ASSET (`Holiday`). The oracle drops the step, so its window counts the
scored steps. We keep the step in its place (#1389), so a window is the last `window` steps of
the walk-forward, and the unscored step adds nothing to it. The two windows are the same set of
steps when the window does not hold the unscored step, and the case compares those points. A
window that holds it is the mean of its other steps, pinned here by hand. `scored_steps = true`
windows over the scored steps instead, and `whole_windows = true` starts each series at its first
whole window, the oracle's layout. With both set, every figure equals the oracle's point for
point (#1513).
=#
include(joinpath(@__DIR__, "parity_harness.jl"))
using Dates, StatsPlots, GraphRecipes, Statistics

@testset "The covariance forecast evaluation figures (#1405)" begin
    PO = PortfolioOptimisers
    ext = Base.get_extension(PO, :PortfolioOptimisersPlotsExt)
    function cp_panel(; gap = false, holiday = 0)
        rng = StableRNG(20260929)
        T, N = 160, 8
        A = randn(rng, N, N) ./ 10 + I
        X = (randn(rng, T, N) * A') ./ 100 .+ 0.0003
        amsk = trues(T, N)
        if gap
            amsk[1:30, 3] .= false
            amsk[131:end, 5] .= false
        end
        X[.!amsk] .= NaN
        holiday > 0 && (X[holiday, :] .= NaN)
        return ReturnsResult(; nx = ["A$i" for i in 1:N], X = X,
                             ts = Date(2024, 1, 1) .+ Day.(0:(T - 1)),
                             pnl = AssetPanel(; amsk = amsk, emsk = copy(amsk)))
    end
    function cp_estimator(hl)
        return ExpWeightedCovariance(; decay = 2.0^(-1 / hl), min_obs = 10,
                                     centring = PreCentred())
    end
    W3 = [fill(1 / 8, 8), collect(1.0:8.0), collect(8.0:-1:1)]
    CP_CASES = (;
                Batch = (; online = false, gap = false, holiday = 0, test = 5, window = 5,
                         w = nothing, compare = false),
                Portfolios = (; online = false, gap = true, holiday = 0, test = 5,
                              window = 5, w = W3, compare = true),
                Holiday = (; online = true, gap = false, holiday = 100, test = 1,
                           window = 10, w = nothing, compare = true))
    function cp_evaluations(c; target = HorizonReturn())
        rd = cp_panel(; gap = c.gap, holiday = c.holiday)
        cv = if c.online
            OnlineIndexWalkForward(62, c.test; purged_size = 2)
        else
            IndexWalkForward(62, c.test; purged_size = 2)
        end
        ev(hl) = covariance_forecast_evaluation(cp_estimator(hl), rd, cv; target = target,
                                                w = c.w)
        return ev(20), c.compare ? ev(40) : nothing
    end
    # The traces of a figure in the oracle's layout: each line, and after a line with a band
    # its upper and then its lower edge. The dashed reference lines are not traces.
    function traces(p, n::Integer)
        cols = Vector{Vector{Float64}}()
        for s in p.series_list[1:n]
            push!(cols, s[:y])
            if s[:fillrange] isa Tuple
                lo, hi = s[:fillrange]
                push!(cols, hi, lo)
            end
        end
        return reduce(hcat, cols)
    end
    load(c, o) = parity_load("CovarianceForecastPlots", c, o)

    @testset "A window as wide as the run ends at the summary" begin
        # The realised covariance target, three test portfolios, a listing, a delisting and a
        # step with no active asset: every weight of the summary is in play.
        c = (; online = true, gap = true, holiday = 100, test = 1, window = 10, w = W3,
             compare = false)
        r, _ = cp_evaluations(c; target = RealisedCovariance())
        M = length(r.dates)
        @test count(iszero, r.n_valid) == 1
        s = covariance_forecast_summary(r)
        p = plot_covariance_calibration(r; window = M)
        @test p[1][:title] == "Rolling Covariance Calibration (window=$(M))"
        @test [x[:label] for x in p.series_list] ==
              ["Mahalanobis Ratio", "Diagonal Ratio", "Bias Statistic", ""]
        m, d, b = (p.series_list[k][:y] for k in 1:3)
        @test all(isnan, m[1:(M - 1)])
        @test m[M] ≈ s.mahalanobis_mean[1] rtol = 1e-14
        @test d[M] ≈ s.diagonal_mean[1] rtol = 1e-14
        @test b[M] ≈ s.bias_statistic[1] rtol = 1e-14
        lo, hi = p.series_list[3][:fillrange]
        @test lo[M] ≈ s.bias_p5[1] rtol = 1e-14
        @test hi[M] ≈ s.bias_p95[1] rtol = 1e-14
        @test all(==(1.0), p.series_list[4][:y])
        q = plot_covariance_qlike(r; window = M)
        @test q[1][:title] == "Rolling Portfolio QLIKE Loss (window=$(M))"
        @test q.series_list[1][:y][M] ≈ s.portfolio_qlike_mean[1] rtol = 1e-14
        e = plot_covariance_exceedance(r; window = M)
        @test e[1][:title] == "Rolling Exceedance Rate (window=$(M))"
        @test [e.series_list[k][:y][M] for k in 1:2] == vec(s.exceedance)
        @test [x[:label] for x in e.series_list] == ["q = 0.95", "q = 0.99", "", ""]
        # Each target rate is drawn at one less the level, in the colour of its series.
        @test all(y -> y ≈ 0.05, e.series_list[3][:y])
        @test e.series_list[3][:linecolor] == e.series_list[1][:linecolor]
        @test e.series_list[4][:linecolor] == e.series_list[2][:linecolor]
        # One test portfolio is drawn with no band.
        r1, _ = cp_evaluations(merge(c, (; w = nothing)); target = RealisedCovariance())
        p1 = plot_covariance_calibration(r1; window = 10, diagnostics = (:bias,))
        @test length(p1.series_list) == 2
        @test isnothing(p1.series_list[1][:fillrange])
        # A window of one step holds no pair of steps, so it draws no bias statistic.
        @test all(isnan,
                  plot_covariance_calibration(r1; window = 1, diagnostics = (:bias,)).series_list[1][:y])
    end

    @testset "The per-step statistics the figures read" begin
        r, _ = cp_evaluations(CP_CASES.Holiday)
        u = findfirst(iszero, r.n_valid)
        e = PO.covariance_exceedance(r, 0.95)
        @test isnan(e[u])
        @test all(x -> x == 0 || x == 1, e[r.n_valid .> 0])
        @test isnan(PO.covariance_diagonal_mean(r)[u])
        @test PO.covariance_forecast_names(nothing, 2) == ["forecast_1", "forecast_2"]
        @test_throws DimensionMismatch PO.covariance_forecast_names(["a"], 2)
        # A window that holds the unscored step is the mean of its other steps.
        W = 10
        p = plot_covariance_qlike(r; window = W)
        t = u + 3
        k = setdiff((t - W + 1):t, u)
        @test p.series_list[1][:y][t] ≈ mean(r.portfolio_qlike[k, 1]) rtol = 1e-14
    end

    @testset "The refusals" begin
        r, b = cp_evaluations(CP_CASES.Holiday)
        M = length(r.dates)
        @test_throws DomainError plot_covariance_calibration(r; window = 0)
        @test_throws DomainError plot_covariance_qlike(r; window = M + 1)
        # The default window of 50 steps is wider than the 19 steps of this run.
        rb, _ = cp_evaluations(CP_CASES.Batch)
        @test_throws DomainError plot_covariance_exceedance(rb)
        @test_throws DomainError plot_covariance_calibration(r; window = 10,
                                                             diagnostics = (:trace,))
        @test_throws DomainError plot_covariance_calibration(r; window = 10,
                                                             diagnostics = ())
        @test_throws DomainError plot_covariance_calibration(r; window = 10,
                                                             diagnostics = ("bias",))
        @test_throws DomainError plot_covariance_exceedance(r; window = 10, levels = (1.0,))
        @test_throws DomainError plot_covariance_exceedance([r, b]; window = 10,
                                                            levels = ())
        empty = typeof(r)[]
        @test_throws PO.IsEmptyError plot_covariance_calibration(empty)
        @test_throws PO.IsEmptyError plot_covariance_qlike(empty)
        @test_throws PO.IsEmptyError plot_covariance_exceedance(empty)
        @test_throws DimensionMismatch plot_covariance_qlike([r, b]; names = ["a"],
                                                             window = 10)
    end

    @testset "The comparison labels each series by its evaluation" begin
        r, b = cp_evaluations(CP_CASES.Holiday)
        p = plot_covariance_calibration([r, b]; names = ["a", "b"], window = 10,
                                        diagnostics = (:mahalanobis,))
        @test [x[:label] for x in p.series_list] == ["a", "b", ""]
        @test isequal(p.series_list[1][:y],
                      plot_covariance_calibration(r; window = 10,
                                                  diagnostics = (:mahalanobis,)).series_list[1][:y])
        p = plot_covariance_calibration([r, b]; window = 10,
                                        diagnostics = (:diagonal, :bias))
        @test [x[:label] for x in p.series_list][1:4] ==
              ["forecast_1 - Diagonal Ratio", "forecast_1 - Bias Statistic",
               "forecast_2 - Diagonal Ratio", "forecast_2 - Bias Statistic"]
        e = plot_covariance_exceedance([r, b]; window = 10, levels = (0.9, 0.95))
        @test [x[:label] for x in e.series_list] ==
              ["forecast_1 - q = 0.9", "forecast_1 - q = 0.95", "forecast_2 - q = 0.9",
               "forecast_2 - q = 0.95", "", ""]
        @test [x[:label] for x in plot_covariance_qlike([r, b]; window = 10).series_list] ==
              ["forecast_1", "forecast_2"]
    end

    @testset "$(case) at parity" for case in keys(CP_CASES)
        c = CP_CASES[case]
        W = c.window
        a, b = cp_evaluations(c)
        firsts = first.(a.test_idx)
        # The points the oracle drew are the steps whose window holds no unscored step.
        u = findall(iszero, a.n_valid)
        clean(x) = (t = findfirst(==(x), firsts); all(j -> !(j in u), (t - W + 1):t))
        figs = (; Cal = (plot_covariance_calibration(a; window = W), 3),
                Qlike = (plot_covariance_qlike(a; window = W), 1),
                Exc = (plot_covariance_exceedance(a; window = W), 2))
        if c.compare
            figs = (; figs...,
                    CmpCal = (plot_covariance_calibration([a, b]; window = W), 6),
                    CmpQlike = (plot_covariance_qlike([a, b]; window = W), 2),
                    CmpExc = (plot_covariance_exceedance([a, b]; window = W), 2))
        end
        for (fig, (p, n)) in pairs(figs)
            o = load(String(case), String(fig))
            keep = findall(clean, o[:, 1])
            @test length(keep) == (isempty(u) ? size(o, 1) : size(o, 1) - (W - 1))
            k = [findfirst(==(x), firsts) for x in o[keep, 1]]
            ours = traces(p, n)[k, :]
            oracle = o[keep, 2:end]
            # The Mahalanobis columns: the first of each evaluation's calibration traces.
            mcols = if fig === :Cal
                [1]
            elseif fig === :CmpCal
                [1, size(oracle, 2) ÷ 2 + 1]
            else
                Int[]
            end
            rest = setdiff(axes(oracle, 2), mcols)
            @test parity_compare(ours[:, rest], oracle[:, rest]; name = "$(case) $(fig)").ok
            for (j, ev) in zip(mcols, (a, b))
                scored = ev.n_valid .> 0
                plain = ext.covariance_rolling_mean(ev.mahalanobis_ratio, nothing, scored,
                                                    ext.covariance_windows(scored, W,
                                                                           false))
                @test parity_compare(plain[k], oracle[:, j];
                                     name = "$(case) $(fig) plain Mahalanobis").ok
                if allequal(ev.n_valid[scored])
                    @test parity_compare(ours[:, j], oracle[:, j];
                                         name = "$(case) $(fig) Mahalanobis").ok
                else
                    # A listing moves the degrees of freedom, and the weights move the mean.
                    @test !parity_compare(ours[:, j], oracle[:, j]; quiet = true).ok
                    nu = PO.target_dof.(Ref(ev.target), ev.n_valid, ev.horizon)
                    @test ours[:, j] ≈
                          [sum(nu[s] * ev.mahalanobis_ratio[s] for s in (t - W + 1):t) /
                           sum(nu[(t - W + 1):t]) for t in k] rtol = 1e-14
                end
            end
        end
    end

    @testset "The oracle's rules, one keyword away (#1513)" begin
        eq = EqualStepWeighting()
        # R56: the plain mean of the steps is the oracle's summary and its rolling mean.
        c = CP_CASES.Portfolios
        a, b = cp_evaluations(c)
        o = load("Portfolios", "Summary")
        for (ev, rows) in ((a, 1:4), (b, 5:8))
            @test !allequal(ev.n_valid[ev.n_valid .> 0])
            os = o[rows, :]
            se = covariance_forecast_summary(ev; step_weighting = eq)
            sd = covariance_forecast_summary(ev)
            @test parity_compare([se.mahalanobis_mean[1], se.mahalanobis_median[1],
                                  se.mahalanobis_p5[1], se.mahalanobis_p95[1],
                                  se.diagonal_mean[1], se.diagonal_median[1],
                                  se.diagonal_p5[1], se.diagonal_p95[1]],
                                 [os[1, 1], os[1, 2], os[1, 4], os[1, 5], os[2, 1],
                                  os[2, 2], os[2, 4], os[2, 5]];
                                 name = "Portfolios summary").ok
            @test !parity_compare([sd.mahalanobis_mean[1]], [os[1, 1]]; quiet = true).ok
        end
        firsts = first.(a.test_idx)
        for (fig, p, n) in
            ((:Cal, plot_covariance_calibration(a; window = c.window, step_weighting = eq),
              3),
             (:CmpCal,
              plot_covariance_calibration([a, b]; window = c.window, step_weighting = eq),
              6))
            oc = load("Portfolios", String(fig))
            k = [findfirst(==(x), firsts) for x in oc[:, 1]]
            ours = traces(p, n)[k, :]
            @test parity_compare(ours, oc[:, 2:end]; name = "Portfolios $(fig) equal").ok
        end
        # R88 and R91: with both keywords, every figure of the holiday case is the oracle's,
        # point for point, and the unscored step draws a blank point inside the series.
        c = CP_CASES.Holiday
        W = c.window
        a, b = cp_evaluations(c)
        kw = (; window = W, scored_steps = true, whole_windows = true)
        scored = a.n_valid .> 0
        u = findfirst(!, scored)
        s = findall(scored)
        t0 = s[W]
        figs = (; Cal = (plot_covariance_calibration(a; step_weighting = eq, kw...), 3),
                Qlike = (plot_covariance_qlike(a; kw...), 1),
                Exc = (plot_covariance_exceedance(a; kw...), 2),
                CmpCal = (plot_covariance_calibration([a, b]; step_weighting = eq, kw...),
                          6), CmpQlike = (plot_covariance_qlike([a, b]; kw...), 2),
                CmpExc = (plot_covariance_exceedance([a, b]; kw...), 2))
        for (fig, (p, n)) in pairs(figs)
            oc = load("Holiday", String(fig))
            @test p.series_list[1][:x] == Dates.value.(a.dates[t0:end])
            y = traces(p, n)
            @test all(isnan, y[u - t0 + 1, :])
            keep = scored[t0:end]
            @test first.(a.test_idx)[t0:end][keep] == oc[:, 1]
            @test parity_compare(y[keep, :], oc[:, 2:end]; name = "Holiday $(fig) oracle").ok
        end
        # A window over the scored steps stretches over the hole: the point after it reads
        # the last W scored steps.
        p = plot_covariance_qlike(a; window = W, scored_steps = true)
        t = u + 3
        j = findfirst(==(t), s)
        @test p.series_list[1][:y][t] ≈ mean(a.portfolio_qlike[s[(j - W + 1):j], 1]) rtol = 1e-14
        @test all(isnan, p.series_list[1][:y][1:(t0 - 1)])
        @test isnan(p.series_list[1][:y][u])
        # `whole_windows` alone starts at step W.
        q = plot_covariance_qlike(a; window = W, whole_windows = true)
        @test q.series_list[1][:x] == Dates.value.(a.dates[W:end])
        @test isequal(q.series_list[1][:y],
                      plot_covariance_qlike(a; window = W).series_list[1][:y][W:end])
        # The defaults keep every value.
        for (f, x) in
            ((plot_covariance_calibration, (; step_weighting = DofStepWeighting())),
             (plot_covariance_qlike, (;)), (plot_covariance_exceedance, (;)))
            d = f(a; window = W)
            e = f(a; window = W, scored_steps = false, whole_windows = false, x...)
            @test all(i -> isequal(d.series_list[i][:y], e.series_list[i][:y]),
                      eachindex(d.series_list))
        end
        # The window counts the scored steps under `scored_steps = true`.
        M = count(scored)
        @test_throws DomainError plot_covariance_qlike(a; window = M + 1,
                                                       scored_steps = true)
        @test length(plot_covariance_qlike(a; window = M, scored_steps = true,
                                           whole_windows = true).series_list[1][:y]) == 1
    end
end
