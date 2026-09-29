#=
Parity of the covariance forecast evaluation, step by step, in batch and online (#1389, map #1375).

`test_24e_covariance_forecast_evaluation.jl` pins the per-step kernel against literals. Here the
whole fold loop and the summary are pinned against stored oracle outputs. Both sides fit an
exponentially weighted covariance of half-life 20 with a warm-up of 10 observations, on data it
assumes centred, so our location is zero and the oracle's kernel, which centres nothing, reads the
same test rows. The target is the horizon return, which is the oracle's.

Every stored case under `test/assets/Parity_CovarianceForecastEvaluation_<Case>_<Output>.csv.gz`
is the oracle output of one row of `CE_CASES`:

  - `Steps`: one row per oracle step, the first test row, the count of active assets, the squared
    Mahalanobis distance, the Mahalanobis ratio, the diagonal ratio (a mean over the active
    assets), then the standardised return and the portfolio QLIKE of each test portfolio.
  - `Summary`: the oracle's summary (the Mahalanobis ratio, the diagonal ratio, the standardised
    return and the portfolio QLIKE, each mean, median, standard deviation, 5th and 95th
    percentile), the bias statistic of each portfolio, its percentiles (5, 25, 50, 75, 95), its
    mean and the portfolio count, and the exceedance rates at 0.95 and 0.99.
  - `Forecasts`: the oracle's forecast of every step, eight rows a step, after its first test row.

THE WINDOW. The oracle trains on `train` rows, purges `purged` rows and tests on the next `test`.
Our walk-forward counts the purge inside its training span, so the same folds are
`IndexWalkForward(train + purged, test; purged_size = purged)` and its online form.

THE STEP WITH NO ACTIVE ASSET. A market holiday inside a one-row test window leaves no asset with
a realised return. That is valid input, so the step has nothing to score and the evaluation must
go on. The oracle drops the step with no message. Before #1389 we refused the whole evaluation
(Defect found). The step is now kept and unscored, `n_valid = 0` and every diagnostic `NaN`, so
each step stays the step of its fold, and the summary and the comparison read the scored steps
only. The summaries then equal the oracle's.

THE PANEL WITH A LISTING AND A DELISTING. There the forecasts differ, because each pair of our
covariance ages on its common observations (#1420, Better). Fed the oracle's own forecast, our
kernel returns the oracle's steps, so the fold loop and the kernel are at parity and the whole
difference is the forecast.

Every step compares cell by cell at `rtol = 1e-12`. Measured 8.4e-14 at most, on a standardised
return of a one-row window. Every summary compares at `rtol = 1e-12`, measured 7.9e-16.
=#
include(joinpath(@__DIR__, "parity_harness.jl"))
using Dates

@testset "The covariance forecast evaluation, step by step, at parity (#1389)" begin
    PO = PortfolioOptimisers
    function panel(; gap = false, holiday = 0)
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
        # A market holiday: every asset stays active, and none has a return on the row.
        holiday > 0 && (X[holiday, :] .= NaN)
        return ReturnsResult(; nx = ["A$i" for i in 1:N], X = X,
                             ts = Date(2024, 1, 1) .+ Day.(0:(T - 1)),
                             pnl = AssetPanel(; amsk = amsk, emsk = copy(amsk)))
    end
    CE_CASES = (; CovBatch = (;), CovBatchExpand = (; expand = true),
                CovOnline = (; online = true),
                CovHoliday = (; online = true, holiday = 100, test = 1),
                CovHolidayBatch = (; holiday = 100, test = 1))
    ce = ExpWeightedCovariance(; decay = 2.0^(-1 / 20), min_obs = 10, centred = true)
    function evaluate(; online = false, expand = false, gap = false, holiday = 0,
                      train = 60, test = 5, purged = 2, w = nothing)
        cv = if online
            OnlineIndexWalkForward(train + purged, test; purged_size = purged)
        else
            IndexWalkForward(train + purged, test; purged_size = purged,
                             expand_train = expand)
        end
        rd = panel(; gap = gap, holiday = holiday)
        return rd,
               covariance_forecast_evaluation(ce, rd, cv; target = HorizonReturn(), w = w,
                                              store_forecasts = true)
    end
    # Our steps in the oracle's layout, on the steps the oracle kept.
    function steps(r, firsts)
        k = [findfirst(==(t), first.(r.test_idx)) for t in firsts]
        d = [sum(filter(isfinite, view(r.diagonal_ratio, i, :))) / r.n_valid[i] for i in k]
        return hcat(firsts, r.n_valid[k], r.n_valid[k] .* r.mahalanobis_ratio[k],
                    r.mahalanobis_ratio[k], d, r.standardised_return[k, :],
                    r.portfolio_qlike[k, :])
    end
    load(c, o) = parity_load("CovarianceForecastEvaluation", c, o)
    for (name, kw) in pairs(CE_CASES)
        case = String(name)
        @testset "$(case)" begin
            _, r = evaluate(; kw...)
            st = load(case, "Steps")
            unscored = first.(r.test_idx[r.n_valid .== 0])
            @test sort(vcat(st[:, 1], unscored)) == first.(r.test_idx)
            @test unscored == (haskey(kw, :holiday) ? [kw.holiday] : Int[])
            @test all(isnan, r.mahalanobis_ratio[r.n_valid .== 0])
            @test parity_compare(steps(r, st[:, 1]), st; name = "$(case) steps").ok
            s = covariance_forecast_summary(r)
            os = vec(load(case, "Summary"))
            k = r.n_valid .> 0
            b = r.standardised_return[k, 1]
            ours = [s.mahalanobis_mean[1], s.mahalanobis_median[1], s.mahalanobis_p5[1],
                    s.mahalanobis_p95[1], s.diagonal_mean[1], s.diagonal_median[1],
                    s.diagonal_p5[1], s.diagonal_p95[1], sum(b) / length(b),
                    s.bias_statistic[1], s.portfolio_qlike_mean[1], s.bias_p5[1],
                    s.bias_p25[1], s.bias_statistic[1], s.bias_p75[1], s.bias_p95[1],
                    s.exceedance[1, 1], s.exceedance[1, 2]]
            oracle = os[[1, 2, 4, 5, 6, 7, 9, 10, 11, 13, 16, 22, 23, 24, 25, 26, 29, 30]]
            @test parity_compare(ours, oracle; name = "$(case) summary").ok
            @test s.n_steps[1] == size(st, 1)
        end
    end

    @testset "CovPortfolios: the kernel on the oracle's forecast, and the forecast of #1420" begin
        W = [fill(1 / 8, 8), collect(1.0:8.0)]
        rd, r = evaluate(; gap = true, w = W)
        st = load("CovPortfolios", "Steps")
        F = load("CovPortfolios", "Forecasts")
        ko = reduce(vcat,
                    map(eachindex(st[:, 1])) do j
                        t = Int(st[j, 1])
                        rows = findall(==(t), F[:, 1])
                        s = covariance_forecast_step(F[rows, 2:end], rd.X[t:(t + 4), :],
                                                     zeros(8), W, HorizonReturn())
                        d = sum(filter(isfinite, s.diagonal_ratio)) / s.n_valid
                        permutedims(vcat(t, s.n_valid, s.n_valid * s.mahalanobis_ratio,
                                         s.mahalanobis_ratio, d, s.standardised_return,
                                         s.portfolio_qlike))
                    end)
        @test parity_compare(ko, st; name = "CovPortfolios kernel").ok
        # Our own forecast differs in the pairs of the asset that lists late and of the one
        # that delists. Their variances are the oracle's, so the diagonal ratio is too.
        ours = steps(r, st[:, 1])
        @test ours[:, 2] == st[:, 2]
        @test parity_compare(ours[:, 5], st[:, 5]; name = "CovPortfolios diagonal").ok
        @test !parity_compare(ours[:, 4], st[:, 4]).ok
    end

    @testset "The comparison pairs the steps both evaluations scored" begin
        rd = panel(; holiday = 100)
        cv = OnlineIndexWalkForward(62, 1; purged_size = 2)
        a = covariance_forecast_evaluation(ce, rd, cv; target = HorizonReturn())
        b = covariance_forecast_evaluation(ExpWeightedCovariance(; decay = 2.0^(-1 / 40),
                                                                 min_obs = 10,
                                                                 centred = true), rd, cv;
                                           target = HorizonReturn())
        c = covariance_forecast_compare(a, b; lags = 0)
        k = (a.n_valid .> 0) .& (b.n_valid .> 0)
        @test count(!, k) == 1
        @test c.n_steps == count(k)
        @test c.mean_difference[1] ≈ sum(a.qlike[k] .- b.qlike[k]) / count(k)
        @test all(isfinite, c.z)
    end
end
