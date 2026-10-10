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

THE FORECAST IS STORED. `Alpha` holds the forecast that the oracle scored, once for each member
(`Base` the fixed member, `EW` and `Refit`), and a case edits it as it edits its own. The test
compares the forecast of this process with it at `rtol = 1e-12` of its largest entry (measured
8.3e-16 of it under the flags below), and then scores the stored one. A
process under `--check-bounds=yes`, as `Pkg.test` runs, sums in another order: its forecast moves by
an ulp, a tie of the winsoriser breaks, and a rank or a calibration bin moves with it (#1475).

THE TWO DIFFERENCES THAT ADR 0149 RULES.

  1. The t-statistic. The oracle scales the ratio by the root of the date count wherever it reads
     it. Our standard error reads the overlap of the forward windows (`forecast_ic_lags`), so the
     two agree where the windows are disjoint. Every oracle t-statistic is our ratio times the
     root of a whole count of dates, and that identity is asserted where the two differ.
  2. A date with no scorable asset. The oracle drops it from the grid. We keep it with `NaN`
     statistics, so `step` means the same thing everywhere. No statistic reads the kept date but
     the coverage, which counts it as a silenced date by design. The summary also reports the
     count of silenced dates and the coverage over the scored dates, which is the oracle's
     coverage (#1512). The book carries its weights
     over such a date, because the forecast states nothing there: before #1389 the turnover out
     of it read a trade out of an empty book (Defect found).

The coefficients, the books, the spreads, the coverage, the calibration and the curve compare cell
by cell at `rtol = 1e-12`. Measured 3.9e-14 at most on a series and 5.5e-15 on a table. The factor
correlation compares against its largest entry (measured 5.6e-16 of it), because the one-hot columns give cells of round-off size, and the pooled mean of a
standardised forecast is round-off zero, so it compares against the pooled standard deviation
(measured 1.3e-15 of it, #1558). The t-statistic that ADR 0149 rules is the oracle's at
`rtol = 1e-14` (measured 1.8e-15) and the square of its ratio to ours is a whole count to
`rtol = 1e-13` (measured 2.5e-14).
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
        # The forecast the oracle scored, stored once for each member.
        stored = edit(parity_load("ForecastEvaluation",
                                  (; fixed = "Base", ew = "EW", target = "Refit")[member],
                                  "Alpha"))
        @test parity_compare(alpha, stored; scale = :array, name = "$(member) alpha").ok
        alpha = stored
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
    # Measured 2.5e-14 at most under the flags of `Pkg.test` and 1.1e-14 without them (H5,
    # Pearson; #1558): the square of a ratio of the oracle's value and ours, each summed in
    # its own order, doubles the round-off of both.
    function whole_root(t, ir, name)
        r = filter(isfinite, (t ./ ir) .^ 2)
        return parity_compare(r, round.(r); rtol = 1e-13, name = "$(name) whole root").ok
    end
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
            @test whole_root(st[:, 3], tabs[:, 2], "$(case) spearman")
            @test whole_root(st[:, 6], tabs[:, 5], "$(case) pearson")
            # R89 (#1512): the oracle's tables are per period whatever its factor is. An
            # evaluation annualised at 12 gives them with `ppy = 1` on the two verbs, and its
            # default tables are the linear rescale of them.
            fe12 = PO.ForecastEvaluationResult(o.fe.alpha, o.fe.y, o.fe.umsk, o.fe.dates,
                                               o.fe.target, o.fe.horizon, o.fe.lag,
                                               o.fe.step, o.fe.min_count, o.fe.ties, 12)
            per = vcat(table(forecast_holding_period(fe12, c.X, c.W; n = c.nfp, ppy = 1)),
                       table(forecast_decay(fe12, c.X, c.W; n = c.nfp, ppy = 1)))
            @test parity_compare(per[:, 7:10], st[:, 7:10]; name = "$(case) per period").ok
            ann = vcat(table(forecast_holding_period(fe12, c.X, c.W; n = c.nfp)),
                       table(forecast_decay(fe12, c.X, c.W; n = c.nfp)))
            @test isequal(isnan.(ann), isnan.(per))
            f = isfinite.(per)
            # The mean rescales bit-equal; the ratio measures 4.0e-16 at most (#1558).
            @test parity_compare(ann[:, [7, 9]][f[:, [7, 9]]],
                                 12 .* per[:, [7, 9]][f[:, [7, 9]]]; rtol = 1e-14,
                                 name = "$(case) annual mean").ok
            @test parity_compare(ann[:, [8, 10]][f[:, [8, 10]]],
                                 sqrt(12) .* per[:, [8, 10]][f[:, [8, 10]]]; rtol = 1e-14,
                                 name = "$(case) annual ratio").ok
            @test isequal(ann[:, setdiff(1:11, 7:10)], per[:, setdiff(1:11, 7:10)])
            cb = forecast_calibration(o.fe, c.W)
            curve = hcat(cb.curve.bin, cb.curve.mean_alpha, cb.curve.mean_y, cb.curve.count)
            @test parity_compare(curve, load(case, "Curve"); name = "$(case) curve").ok
            ss = vec(load(case, "Summary"))
            s = o.summary
            n = length(s)
            @test n == length(ss)
            keep = setdiff(1:n, [4, 9, 29, 30, 31, 32, n - 4])
            @test parity_compare(s[keep], ss[keep]; name = "$(case) summary").ok
            # The pooled mean of a standardised forecast is round-off zero, so it compares
            # against the pooled standard deviation beside it: measured 1.3e-15 of it at most
            # under the flags of `Pkg.test` (Spearman) and 2.2e-16 without them. The t
            # identities below measure 1.8e-15 at most (H5), each a ratio over a root (#1558).
            @test parity_compare(s[[n - 4, n - 3]], ss[[n - 4, n - 3]]; rtol = 1e-14,
                                 scale = :array, name = "$(case) pooled mean").ok
            ic = forecast_ic(o.fe, c.W)
            nsp, npe = count(isfinite, ic[:, 1]), count(isfinite, ic[:, 2])
            @test parity_compare(ss[[4, 9]], [s[3] * sqrt(nsp), s[8] * sqrt(npe)];
                                 rtol = 1e-14, name = "$(case) oracle t").ok
            if PO.forecast_ic_lags(o.fe) == 0
                @test parity_compare(s[[4, 9]], ss[[4, 9]]; rtol = 1e-14,
                                     name = "$(case) disjoint t").ok
            end
            # The coverage summary counts a silenced date; on the scored dates it is the
            # oracle's. The library's own coverage on the oracle's dates gives it.
            cov = ours[:, end]
            nsc = ours[:, end - 1]
            @test parity_compare([sum(cov) / length(cov), minimum(cov),
                                  sum(nsc) / length(nsc), minimum(nsc)], ss[29:32]).ok
            if case == "Gap"
                @test s[30] == 0 && s[32] == 0
            else
                @test parity_compare(s[29:32], ss[29:32]).ok
            end
            # R87 (#1512): the coverage over the scored dates is the oracle's coverage, and
            # the silenced dates are the dates the oracle dropped. The two means satisfy the
            # identity of `forecast_summary_coverage`.
            fs = forecast_evaluation_summary(o.fe, c.W)
            @test fs.n_silenced == [length(dropped)]
            @test parity_compare([fs.mean_coverage_scored[1], fs.min_coverage_scored[1]],
                                 ss[29:30]; name = "$(case) scored coverage").ok
            nc = count(isfinite, forecast_coverage(o.fe, c.W))
            # Measured bit-equal in every case (#1558).
            @test parity_compare([fs.mean_coverage_scored[1]],
                                 [fs.mean_coverage[1] * nc / (nc - fs.n_silenced[1])];
                                 rtol = 1e-14, name = "$(case) coverage identity").ok
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
            # Measured 1.2e-16 (#1558).
            @test parity_compare([p.turnover[k + 1]],
                                 [sum(abs, p.w[k + 1, :] .- p.w[k - 1, :])]; rtol = 1e-14,
                                 name = "$(kind) turnover out of the gap").ok
        end
    end
end
