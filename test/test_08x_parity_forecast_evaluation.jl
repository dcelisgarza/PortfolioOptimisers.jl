#=
Parity of the Return Forecast evaluation, series by series and date by date (#1389, map #1375).

`test_08x_forecast_evaluation.jl` pins the summaries of the evaluation against literals. Here every
per-date series, every forward-window table and every summary is pinned against a stored oracle
output, on the fixture of that file. The oracle scores the same forecast history, the same target
history and the same estimation mask through its own public evaluation, so the forecast itself is
out of the comparison: a member's history is at parity since #1386.

Every stored case under `test/assets/Parity_ForecastEvaluation_<Case>_<Output>.csv.gz` is the
oracle output of one row of `FE_CASES`:

  - `Series`: one row per oracle date, the date, the Spearman and the Pearson coefficient, the
    returns and the turnovers of the rank and the z-score book, the quantile spreads, the count of
    scored assets and the coverage.
  - `Factor`: the contemporaneous correlation of the forecast with every exposure, on every row.
  - `Tables`: the holding-period table over the decay table, eleven columns each.
  - `Summary`: the coefficient, book, quantile, coverage and calibration summaries, in the layout
    of `fe_summary`.
  - `Curve`: the calibration curve, one row per bin.

WHAT THE ORACLE RANKS DIFFERENTLY. The oracle ranks a cross-section by an unstable sort, so it
ranks a tie in an arbitrary order. The forecasts of the fixture hold ties, because the winsoriser
clips both tails to one value. A rank statistic of a tie is its midrank by definition, and that is
our default `ties = :average` (#1332, Better since #1381). The stored oracle ran with its rank
replaced by the midrank, so it isolates everything else. `TiesOrdinal` ran with a stable ordinal
rank, which `ties = :ordinal` reproduces: the oracle's rule stays one keyword away.

THE TWO DIFFERENCES THAT ADR 0149 RULES.

  1. The t-statistic. The oracle scales the ratio by the root of the date count wherever it reads
     it. Our standard error reads the overlap of the forward windows (`forecast_ic_lags`), so the
     two agree where the windows are disjoint. Every oracle t-statistic is our ratio times the
     root of a whole count of dates, and that identity is asserted where the two differ.
  2. A date with no scorable asset. The oracle drops it from the grid. We keep it with `NaN`
     statistics, so `step` means the same thing everywhere. No statistic reads the kept date but
     the coverage, which counts it as a silenced date by design. The book carries its weights
     over such a date, because the forecast states nothing there: before #1389 the turnover out
     of it read a trade out of an empty book (Defect found).

The coefficients, the books, the spreads, the coverage, the calibration and the curve compare cell
by cell at `rtol = 1e-12`. Measured 3.9e-14 at most on a series and 5.5e-15 on a table. The factor
correlation compares against its largest entry (measured 5.6e-16 of it), because the one-hot columns give cells of round-off size, and the pooled mean of a
standardised forecast is round-off zero, so it compares against the pooled standard deviation.
=#
include(joinpath(@__DIR__, "parity_harness.jl"))
include(joinpath(@__DIR__, "test06c_setup.jl"))
include(joinpath(@__DIR__, "forecast_evaluation_fixture.jl"))

# The layout of a stored `Summary`: the coefficient summary (Spearman then Pearson, each mean,
# std, ratio, t-statistic, hit rate), the book summary (rank then z-score, each annualised mean,
# annualised volatility, ratio, hit rate, mean turnover), the quantile summary (per quantile the
# annualised mean, volatility, ratio and hit rate), the coverage summary (mean and least coverage,
# mean and least count of scored assets) and the calibration summary (slope, mean and std of the
# forecast, mean and std of the target, bins).
function fe_summary(s::ForecastSummaryResult)
    q = vec(permutedims(hcat(vec(s.spread_ann_return), vec(s.spread_ann_volatility),
                             vec(s.spread_sharpe), vec(s.spread_hit_rate))))
    return vcat(s.spearman_mean_ic, s.spearman_std_ic, s.spearman_ic_ir, s.spearman_t_stat,
                s.spearman_hit_rate, s.pearson_mean_ic, s.pearson_std_ic, s.pearson_ic_ir,
                s.pearson_t_stat, s.pearson_hit_rate, s.rank_ann_return,
                s.rank_ann_volatility, s.rank_sharpe, s.rank_hit_rate, s.rank_mean_turnover,
                s.zscore_ann_return, s.zscore_ann_volatility, s.zscore_sharpe,
                s.zscore_hit_rate, s.zscore_mean_turnover, q, s.mean_coverage,
                s.min_coverage, s.mean_n_scored, s.min_n_scored, s.calibration_slope,
                s.mean_alpha, s.std_alpha, s.mean_y, s.std_y, s.n_bins)
end

@testset "The Return Forecast evaluation, series by series, at parity (#1389)" begin
    PO = PortfolioOptimisers
    gap(a) = (a[10, :] .= NaN; a[20, 4:end] .= NaN; a)
    ties(a) = round.(a ./ 0.5) .* 0.5
    FE_CASES = (; Base = (;), H5 = (; h = 5, step = 5, nfp = 3),
                Overlap = (; h = 3, l = 2, step = 1, nfp = 3),
                Weighted = (; h = 2, step = 2, weighted = true),
                Spearman = (; h = 2, step = 1, rank = true), Gap = (; edit = gap),
                Refit = (; planted = false, member = :target, h = 2, step = 2),
                EW = (; planted = false, member = :ew, h = 2, step = 1, nfp = 3),
                Ties = (; edit = ties, h = 2, step = 2))
    fixtures = Dict(p => evaluation_fixture(; planted = p) for p in (true, false))
    function build(; planted = true, member = :fixed, h = 1, l = 1, step = h, nfp = 4,
                   weighted = false, rank = false, edit = identity)
        fx = fixtures[planted]
        rd, csfm, rows = fx.rd, fx.csfm, fx.rows
        alpha = if member === :fixed
            return_forecast(FixedWeightedReturnForecast(; scores = fx.scores, scale = 1.0,
                                                        weights = [0.4, 0.6]), rd, csfm).hist
        elseif member === :ew
            return_forecast(ExpWeightedReturnForecast(; scores = fx.scores,
                                                      half_life = 10.0, min_obs = 1,
                                                      horizon = h, lag = l), rd, csfm).hist
        else
            # The oracle's evaluation refuses a member with no history of its own; ours
            # refits it along the grid, and the oracle scores the history ours refitted.
            PO.forecast_history(TargetReturnForecast(; scores = fx.scores, horizon = h,
                                                     lag = l, calibrate = false), rd, csfm;
                                step = step)
        end
        alpha = edit(copy(alpha))
        amsk = rd.pnl.amsk[rows, :]
        # A weight outside the active universe is missing, as the oracle's panel needs it.
        W = if weighted
            ifelse.(amsk, PO.panel_field(rd.pnl, "market_cap").vals[rows, :], NaN)
        else
            nothing
        end
        return (; alpha, X = csfm.csr.eps, emsk = rd.pnl.emsk[rows, :], W, B = csfm.Ms, h,
                l, step, nfp, rank)
    end
    function score(c; ties = :average)
        y = PO.forward_mean_returns(c.X, c.h, c.l)
        fe = forecast_evaluation(c.alpha, y; umsk = c.emsk, horizon = c.h, lag = c.l,
                                 step = c.step, ties = ties)
        rk = forecast_portfolio(fe; kind = :rank)
        zs = forecast_portfolio(fe; kind = :zscore)
        series = hcat(fe.dates, forecast_ic(fe, c.W), rk.ret, zs.ret, rk.turnover,
                      zs.turnover,
                      forecast_quantile_spread(fe; quantiles = (0.1, 0.25)).spread,
                      PO.forecast_summary_scored(fe,
                                                 PO.forecast_ic_weights(fe.alpha, nothing)),
                      forecast_coverage(fe))
        return (; fe, series,
                summary = fe_summary(forecast_evaluation_summary(fe, c.W;
                                                                 quantiles = (0.1, 0.25))))
    end
    # Our series on the oracle's dates, and the dates the oracle dropped.
    function on_oracle_dates(ours, stored)
        k = [findfirst(==(t), ours[:, 1]) for t in stored[:, 1]]
        return ours[k, :], setdiff(ours[:, 1], stored[:, 1])
    end
    table(t) = hcat(t.spearman_mean_ic, t.spearman_ic_ir, t.spearman_t_stat,
                    t.pearson_mean_ic, t.pearson_ic_ir, t.pearson_t_stat, t.rank_ann_return,
                    t.rank_sharpe, t.zscore_ann_return, t.zscore_sharpe, t.mean_coverage)
    load(c, o) = parity_load("ForecastEvaluation", c, o)
    # The oracle's t-statistic is our ratio times the root of a whole count of dates.
    whole_root(t, ir) = all(i -> !isfinite(t[i] / ir[i]) ||
                                 abs((t[i] / ir[i])^2 - round((t[i] / ir[i])^2)) < 1e-9,
                            eachindex(t, ir))
    nonT = [1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    for (name, kw) in pairs(FE_CASES)
        case = String(name)
        @testset "$(case)" begin
            c = build(; kw...)
            o = score(c)
            stored = load(case, "Series")
            ours, dropped = on_oracle_dates(o.series, stored)
            # The dates: ours keeps a date with no scorable asset, the oracle drops it.
            @test isempty(dropped) == (case != "Gap")
            @test all(t -> all(isnan, o.series[findfirst(==(t), o.series[:, 1]), 2:3]),
                      dropped)
            @test parity_compare(ours, stored; name = "$(case) series").ok
            fc = forecast_factor_correlation(o.fe, c.B, c.W; rank = c.rank)
            @test parity_compare(fc, load(case, "Factor"); scale = :array,
                                 name = "$(case) factor").ok
            tabs = vcat(table(forecast_holding_period(o.fe, c.X, c.W; n = c.nfp)),
                        table(forecast_decay(o.fe, c.X, c.W; n = c.nfp)))
            st = load(case, "Tables")
            @test parity_compare(tabs[:, [1, 2, 4, 5, 7, 8, 9, 10, 11]],
                                 st[:, [1, 2, 4, 5, 7, 8, 9, 10, 11]];
                                 name = "$(case) tables").ok
            @test whole_root(st[:, 3], tabs[:, 2]) && whole_root(st[:, 6], tabs[:, 5])
            cb = forecast_calibration(o.fe, c.W)
            curve = hcat(cb.curve.bin, cb.curve.mean_alpha, cb.curve.mean_y, cb.curve.count)
            @test parity_compare(curve, load(case, "Curve"); name = "$(case) curve").ok
            ss = vec(load(case, "Summary"))
            s = o.summary
            n = length(s)
            @test n == length(ss)
            keep = setdiff(1:n, [4, 9, 29, 30, 31, 32, n - 4])
            @test parity_compare(s[keep], ss[keep]; name = "$(case) summary").ok
            @test abs(s[n - 4] - ss[n - 4]) <= 1e-12 * s[n - 3]
            ic = forecast_ic(o.fe, c.W)
            nsp, npe = count(isfinite, ic[:, 1]), count(isfinite, ic[:, 2])
            @test isapprox(ss[4], s[3] * sqrt(nsp); rtol = 1e-12)
            @test isapprox(ss[9], s[8] * sqrt(npe); rtol = 1e-12)
            if PO.forecast_ic_lags(o.fe) == 0
                @test isapprox(s[[4, 9]], ss[[4, 9]]; rtol = 1e-12)
            end
            # The coverage summary counts a silenced date; on the scored dates it is the
            # oracle's.
            cov = stored[:, end]
            nsc = stored[:, end - 1]
            @test parity_compare([sum(cov) / length(cov), minimum(cov),
                                  sum(nsc) / length(nsc), minimum(nsc)], ss[29:32]).ok
            if case == "Gap"
                @test s[30] == 0 && s[32] == 0
            else
                @test parity_compare(s[29:32], ss[29:32]).ok
            end
        end
    end

    @testset "TiesOrdinal: the oracle's rank is one keyword away" begin
        c = build(; FE_CASES.Ties...)
        o = score(c; ties = :ordinal)
        stored = load("TiesOrdinal", "Series")
        ours, _ = on_oracle_dates(o.series, stored)
        @test parity_compare(ours, stored; name = "TiesOrdinal series").ok
        ss = vec(load("TiesOrdinal", "Summary"))
        @test parity_compare(o.summary[nonT], ss[nonT]; name = "TiesOrdinal summary").ok
        # The midrank moves the rank statistics of a tied forecast, and nothing else.
        a = score(c)
        @test maximum(abs, filter(!isnan, a.series[:, 2] .- o.series[:, 2])) > 1e-3
        @test isequal(a.series[:, 3], o.series[:, 3])
    end

    @testset "The book carries its weights over a date with no scorable asset" begin
        c = build(; FE_CASES.Gap...)
        fe = score(c).fe
        for kind in (:rank, :zscore)
            p = forecast_portfolio(fe; kind = kind)
            k = findfirst(==(10), fe.dates)
            @test isnan(p.ret[k]) && isnan(p.turnover[k])
            @test p.w[k, :] == p.w[k - 1, :]
            @test p.turnover[k + 1] ≈ sum(abs, p.w[k + 1, :] .- p.w[k - 1, :])
        end
    end
end
