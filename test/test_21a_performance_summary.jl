#=
`performance_summary` had no behavioural test -- only the plotting smoke tests in
`test_25_plotting.jl`, which assert that a figure comes back. Its standard error of the
Sharpe ratio was wrong because of that.

The source states the variance of the estimator as

    (1 - gamma_1 * SR + (gamma_2 - 1) / 4 * SR ^ 2) / (T - 1),

where `gamma_2` is the RAW fourth standardised moment. `StatsBase.kurtosis` returns the
EXCESS moment `g_2 = gamma_2 - 3`, so the coefficient is `(g_2 + 2) / 4`. The code wrote
`g_2 / 4`, which drops the whole `SR ^ 2 / 2` term the naive expression carries, so a normal
series was reported as more precise than it is.

The other statistics are pinned here too, because nothing else pins them.
=#

@testset "performance_summary: the statistics" begin
    ret = [0.01, -0.02, 0.03, 0.005, -0.01, 0.02, -0.005, 0.015]
    ps = performance_summary(ret; periods_per_year = 4, alpha = 0.25)
    T = length(ret)
    m = mean(ret)
    s = std(ret)
    @test ps.n_periods == T
    @test ps.periods_per_year == 4
    @test ps.alpha == 0.25
    @test !ps.compound
    @test isapprox(ps.ann_return, m * 4)
    @test isapprox(ps.ann_volatility, s * sqrt(4))
    @test isapprox(ps.sharpe, (m * 4) / (s * sqrt(4)))
    # The downside deviation divides by `T`, not by the count of losing periods.
    ddev = sqrt(mean(min.(ret, 0) .^ 2) * 4)
    @test isapprox(ps.sortino, (m * 4) / ddev)
    dd = PortfolioOptimisers.drawdowns(PortfolioOptimisers.cumulative_returns(ret, false),
                                       false; cX = true)
    @test isapprox(ps.max_drawdown, minimum(dd))
    @test isapprox(ps.calmar, (m * 4) / abs(minimum(dd)))
    # `cvar` is reported in return space, the opposite sign to the risk measure.
    @test isapprox(ps.cvar, -ConditionalValueatRisk(; alpha = 0.25)(ret))
    @test ps.cvar <= 0
end

@testset "performance_summary: the Sharpe ratio standard error" begin
    ret = [0.01, -0.02, 0.03, 0.005, -0.01, 0.02, -0.005, 0.015]
    ps = performance_summary(ret; periods_per_year = 4)
    T = length(ret)
    sr_p = mean(ret) / std(ret)
    g1 = StatsBase.skewness(ret)
    g2 = StatsBase.kurtosis(ret)
    # The `+ 2` converts the excess fourth moment to the raw one the source states.
    want = sqrt((1 - g1 * sr_p + (g2 + 2) / 4 * sr_p^2) / (T - 1) * 4)
    @test isapprox(ps.sharpe_stderr, want)
    # The form the code used to carry is a different number, so this is not a tie.
    wrong = sqrt((1 - g1 * sr_p + g2 / 4 * sr_p^2) / (T - 1) * 4)
    @test !isapprox(ps.sharpe_stderr, wrong)

    # On a normal series the expression must reduce to the naive `sqrt((1 + SR^2 / 2) / T)`
    # of Lo (2002). Only the corrected coefficient does: a Gaussian sample has `g2 = 0`, and
    # `g2 / 4` then leaves `1` where `(g2 + 2) / 4` leaves `1 + SR^2 / 2`.
    rng = StableRNG(99887766)
    T2 = 250
    nsim = 20_000
    srs = Vector{Float64}(undef, nsim)
    for i in 1:nsim
        r = randn(rng, T2) .+ 0.5          # per-period Sharpe ratio of 0.5
        srs[i] = mean(r) / std(r)
    end
    empirical = std(srs)
    SR = 0.5
    corrected = sqrt((1 + (0 + 2) / 4 * SR^2) / (T2 - 1))
    dropped = sqrt((1 + 0 / 4 * SR^2) / (T2 - 1))
    @test isapprox(corrected, empirical; rtol = 5e-3)
    @test !isapprox(dropped, empirical; rtol = 5e-3)
    @test abs(corrected - empirical) < abs(dropped - empirical)
end

@testset "performance_summary: validation and the degenerate series" begin
    ret = [0.01, -0.02, 0.03, 0.005]
    @test_throws DomainError performance_summary(ret; alpha = 0.0)
    @test_throws DomainError performance_summary(ret; alpha = 1.0)
    @test_throws DomainError performance_summary(ret; periods_per_year = 0)
    # A constant series has no drawdown and no losing period, so the ratios that divide by
    # them are `NaN`. The Sharpe ratio is not among them: `std` of a constant vector returns
    # `1.8e-18` rather than an exact zero, so the `ann_volatility > 0` guard does not fire and
    # the ratio comes back as a huge finite number. That is the arithmetic of `std`, not a
    # rule of this function, and a caller reads `ann_volatility` to tell the two apart.
    flat = performance_summary(fill(0.01, 10))
    @test isnan(flat.sortino)
    @test isnan(flat.calmar)
    @test iszero(flat.max_drawdown)
    @test flat.ann_volatility < 1e-15
    @test isfinite(flat.sharpe)
end

#=
Issue #549, conditions 2 and 3. `src/21_ExpectedReturns.jl` held 49 of child map 8's 66
misses, and `brinson_attribution` had no test in the whole suite. Every check below was run
with real numbers in the REPL first, and each quoted number is the one the REPL returned.
=#

using Dates

const PO = PortfolioOptimisers

# One fixture for the whole of #549. The prior is empirical, so `pr.mu` is the column mean
# of `X` and `pr.X` is `X` itself.
const ER_RNG = StableRNG(987654321)
const ER_X = randn(ER_RNG, 60, 4) .* 0.02 .+ 0.001
const ER_PR = prior(EmpiricalPrior(), ER_X)
const ER_W = [0.4, 0.3, 0.2, 0.1]
const ER_MU = [0.01, 0.02, 0.03, 0.04]

@testset "#549: the three expected returns" begin
    @testset "`ArithmeticReturn` contracts the weights with `mu`" begin
        r = ArithmeticReturn(; mu = ER_MU)
        # `dot(w, mu)` by hand: 0.4*0.01 + 0.3*0.02 + 0.2*0.03 + 0.1*0.04 = 0.02.
        @test expected_return(r, ER_W, ER_PR) == 0.02
        @test expected_return(r, ER_W, ER_PR) == dot(ER_W, ER_MU)
        # The term's own `mu` wins over the prior's, and an unstated one falls back.
        @test expected_return(ArithmeticReturn(), ER_W, ER_PR) == dot(ER_W, ER_PR.mu)
        @test expected_return(ArithmeticReturn(), ER_W, ER_PR) != 0.02
    end

    @testset "the fee is subtracted exactly once" begin
        r = ArithmeticReturn(; mu = ER_MU)
        fees = Fees(; l = 0.01)
        f = calc_fees(ER_W, fees)
        # `calc_fees` is 0.01 on a long-only unit budget, so the net figure is 0.01.
        @test isapprox(f, 0.01)
        @test isapprox(expected_return(r, ER_W, ER_PR, fees), 0.02 - f)
    end

    @testset "a Deferred Quantity in the `mu` slot resolves against the prior" begin
        # The prior is in hand, so `resolve_deferred_quantities` fits the estimator here.
        r = ArithmeticReturn(; mu = MedianExpectedReturns())
        mu_med = vec(PO.mean(MedianExpectedReturns(), ER_X))
        @test isapprox(expected_return(r, ER_W, ER_PR), dot(ER_W, mu_med))
        # 0.00346890140412576, and the empirical prior's own answer is a different number.
        @test !isapprox(expected_return(r, ER_W, ER_PR), dot(ER_W, ER_PR.mu))
    end

    @testset "`LogarithmicReturn` is the mean of the log, not the log of the mean" begin
        unweighted = expected_return(LogarithmicReturn(), ER_W, ER_PR)
        @test isapprox(unweighted, mean(log1p.(ER_X * ER_W)))
        # The observation weights reach `mean`, so the weighted form is a different number.
        ow = StatsBase.pweights(range(0.5, 1.5; length = 60))
        weighted = expected_return(LogarithmicReturn(; w = ow), ER_W, ER_PR)
        @test isapprox(weighted, mean(log1p.(ER_X * ER_W), ow))
        @test !isapprox(weighted, unweighted)
        # The whole point of the estimator: 0.002200065746599195 is not `log1p` of the
        # arithmetic answer.
        @test !isapprox(unweighted, log1p(dot(ER_W, ER_PR.mu)))
        # The fee reaches this term too.
        fees = Fees(; l = 0.01)
        @test isapprox(expected_return(LogarithmicReturn(), ER_W, ER_PR, fees),
                       unweighted - calc_fees(ER_W, fees))
    end

    @testset "`NoReturn` is a typed zero that charges no fee" begin
        fees = Fees(; l = 0.01)
        # `settings.fee` is `true` by default and the term still charges nothing.
        @test NoReturn().settings.fee
        @test iszero(expected_return(NoReturn(), ER_W, ER_PR, fees))
        @test iszero(expected_return(NoReturn(), ER_W, ER_PR))
        z = expected_return(NoReturn(), Float32[0.5, 0.5, 0, 0], ER_PR, fees)
        @test z === zero(Float32)
    end
end

@testset "#549: `term_fees` follows the fee flag alone" begin
    fees = Fees(; l = 0.01)
    @test isapprox(PO.term_fees(ER_W, fees, true), calc_fees(ER_W, fees))
    @test iszero(PO.term_fees(ER_W, fees, false))
    @test iszero(PO.term_fees(ER_W, nothing, true))
    # The gross figure comes back when the term's own flag is `false`.
    gross = ArithmeticReturn(; mu = ER_MU, settings = JuMPReturnsSettings(; fee = false))
    net = ArithmeticReturn(; mu = ER_MU, settings = JuMPReturnsSettings(; fee = true))
    @test expected_return(gross, ER_W, ER_PR, fees) == 0.02
    @test isapprox(expected_return(net, ER_W, ER_PR, fees), 0.02 - calc_fees(ER_W, fees))
end

@testset "#549: the aggregate is aggregate over aggregate" begin
    # Two terms, different scales, and the second one disabled.
    rA = ArithmeticReturn(; mu = ER_MU, settings = JuMPReturnsSettings(; scale = 2.0))
    rB = ArithmeticReturn(; mu = fill(0.05, 4),
                          settings = JuMPReturnsSettings(; scale = 3.0, rte = false))
    rts = [rA, rB]

    @testset "the sum runs over the enabled terms at their own scale" begin
        # 2.0 * 0.02 = 0.04. The disabled term would have added 3.0 * 0.05 = 0.15.
        @test expected_return(rts, ER_W, ER_PR) == 0.04
        @test expected_return([rA], ER_W, ER_PR) == 0.04
        @test expected_return([rB], ER_W, ER_PR) == 0.0
        # One expected return per weight vector on the `VecVecNum` method.
        wv = [ER_W, fill(0.25, 4)]
        @test expected_return(rts, wv, ER_PR) == [0.04, 0.05]
        # `settings.scale` is a combination weight of the AGGREGATE, so the singular
        # method never applies it: the same term alone gives 0.02 and not 2.0 * 0.02.
        @test expected_return(rA, wv, ER_PR) == [0.02, 0.025]
    end

    @testset "`sca` scalarises the risk axis and never the return axis" begin
        rks = [factory(Variance(), ER_PR),
               factory(ConditionalValueatRisk(; alpha = 0.1), ER_PR)]
        rk_sum = expected_risk(rks, ER_W, ER_PR; sca = SumScalariser())
        rk_max = expected_risk(rks, ER_W, ER_PR; sca = MaxScalariser())
        @test !isapprox(rk_sum, rk_max)
        rt = expected_return(rts, ER_W, ER_PR)
        rat_sum = expected_ratio(rks, rts, ER_W, ER_PR; sca = SumScalariser(), rf = 0.001)
        rat_max = expected_ratio(rks, rts, ER_W, ER_PR; sca = MaxScalariser(), rf = 0.001)
        @test isapprox(rat_sum, (rt - 0.001) / rk_sum)
        @test isapprox(rat_max, (rt - 0.001) / rk_max)
        @test !isapprox(rat_sum, rat_max)
        # The return axis is the same 0.04 under both scalarisers.
        for sca in (SumScalariser(), MaxScalariser())
            rk, rtt, rr = expected_risk_ret_ratio(rks, rts, ER_W, ER_PR; sca = sca,
                                                  rf = 0.001)
            @test rtt == 0.04
            @test isapprox(rr, (rtt - 0.001) / rk)
        end
    end

    @testset "the SRIC penalty is applied once, not per element" begin
        rks = [factory(Variance(), ER_PR),
               factory(ConditionalValueatRisk(; alpha = 0.1), ER_PR)]
        T, N = size(ER_PR.X)
        @test (T, N) == (60, 4)
        sr = expected_ratio(rks, rts, ER_W, ER_PR; sca = SumScalariser(), rf = 0.001)
        pen = PO.sric_penalty(sr, ER_PR)
        @test pen == N / (T * sr)
        sric = expected_sric(rks, rts, ER_W, ER_PR; sca = SumScalariser(), rf = 0.001)
        @test sric == sr - pen
        # Two measures and two terms, so a per-element penalty would be four times as
        # large and visibly different: 2.3314625367769004 against 2.4134055727315573.
        @test !isapprox(sric, sr - 2 * 2 * pen)
        rk, rtt, sric2 = expected_risk_ret_sric(rks, rts, ER_W, ER_PR;
                                                sca = SumScalariser(), rf = 0.001)
        @test sric2 == sric
        @test rtt == 0.04
        @test isapprox(rk, expected_risk(rks, ER_W, ER_PR; sca = SumScalariser()))
    end
end

@testset "#549: the two prior-reading measures" begin
    rA = ArithmeticReturn(; mu = ER_MU, settings = JuMPReturnsSettings(; scale = 2.0))
    rB = ArithmeticReturn(; mu = fill(0.05, 4),
                          settings = JuMPReturnsSettings(; scale = 3.0, rte = false))

    @testset "`ExpectedReturn` reports the aggregate return as its risk" begin
        fees = Fees(; l = 0.01)
        r = ExpectedReturn(; rt = ArithmeticReturn(; mu = ER_MU))
        @test expected_risk(r, ER_W, ER_PR) == 0.02
        @test isapprox(expected_risk(r, ER_W, ER_PR, fees), 0.02 - calc_fees(ER_W, fees))
        @test expected_risk(ExpectedReturn(; rt = [rA, rB]), ER_W, ER_PR) == 0.04
        @test keys(PO.deferred_slots(r)) == (:rt,)
        @test_throws PortfolioOptimisers.IsEmptyError ExpectedReturn(;
                                                                     rt = PO.JuMPReturnsEstimator[])
    end

    @testset "`ExpectedReturnRiskRatio` pins its own `sca` and `rf`" begin
        rks = [factory(Variance(), ER_PR),
               factory(ConditionalValueatRisk(; alpha = 0.1), ER_PR)]
        rts = [rA, rB]
        r = ExpectedReturnRiskRatio(; rt = rts, rk = rks, sca = SumScalariser(), rf = 0.001)
        want = expected_ratio(rks, rts, ER_W, ER_PR; sca = SumScalariser(), rf = 0.001)
        @test expected_risk(r, ER_W, ER_PR) == want
        # A `sca` and an `rf` at the call site lose to the fields.
        @test expected_risk(r, ER_W, ER_PR; sca = MaxScalariser(), rf = 0.5) == want
        @test keys(PO.deferred_slots(r)) == (:rt, :rk)
        @test !PO.needs_previous_weights(r)
        @test_throws PortfolioOptimisers.IsEmptyError ExpectedReturnRiskRatio(;
                                                                              rt = PO.JuMPReturnsEstimator[])
        @test_throws PortfolioOptimisers.IsEmptyError ExpectedReturnRiskRatio(;
                                                                              rk = PO.AbstractBaseRiskMeasure[])
        @test_throws PortfolioOptimisers.IsNonFiniteError ExpectedReturnRiskRatio(;
                                                                                  rf = NaN)
        @test_throws PortfolioOptimisers.IsNonFiniteError ExpectedReturnRiskRatio(;
                                                                                  rf = Inf)
    end

    @testset "a vector of weight vectors resolves the measure once" begin
        wv = [ER_W, fill(0.25, 4)]
        r = ExpectedReturn(; rt = ArithmeticReturn(; mu = MedianExpectedReturns()))
        mu_med = vec(PO.mean(MedianExpectedReturns(), ER_X))
        @test isapprox(expected_risk(r, wv, ER_PR), [dot(wi, mu_med) for wi in wv])
        rr = ExpectedReturnRiskRatio(; rk = Variance(), rf = 0.001, sca = MaxScalariser())
        @test isapprox(expected_risk(rr, wv, ER_PR),
                       [expected_risk(factory(rr, ER_PR), wi, ER_PR) for wi in wv])
    end

    @testset "the performance group answers that bigger is better" begin
        @test PO.bigger_is_better(ExpectedReturn())
        @test PO.bigger_is_better(ExpectedReturnRiskRatio())
        @test PO.bigger_is_better(MeanReturn())
        @test PO.bigger_is_better(MeanReturnRiskRatio())
    end
end

@testset "#549: a prior-reading measure refuses a prediction result" begin
    # A prediction result needs no solver: a `NaiveOptimisationResult` is already solved.
    res = NaiveOptimisationResult(; pr = nothing, wb = nothing,
                                  retcode = OptimisationSuccess(), w = [0.5, 0.5],
                                  fb = nothing)
    rdv = PredictionReturnsResult(; nx = ["A", "B"], X = [0.01, -0.02, 0.03, 0.005],
                                  ts = nothing)
    rdvv = PredictionReturnsResult(; nx = ["A", "B"], X = [[0.01, -0.02], [0.03, 0.005]],
                                   ts = nothing)
    predv = PredictionResult(; res = res, rd = rdv)
    predvv = PredictionResult(; res = res, rd = rdvv)
    mpred = MultiPeriodPredictionResult(; pred = [predv])
    ppred = PopulationPredictionResult(; pred = [predv])

    @testset "all four methods raise, and none returns a figure" begin
        for pred in (predv, predvv, mpred, ppred)
            @test_throws ArgumentError expected_risk(ExpectedReturn(), pred)
            @test_throws ArgumentError expected_risk(ExpectedReturnRiskRatio(), pred)
        end
    end

    @testset "the message names the type it refused and the replacement" begin
        msg = PO.prrm_prediction_message(ExpectedReturn(), predv)
        @test occursin("PredictionResult", msg)
        @test occursin("MeanReturn", msg)
        # A singular `rt` swaps cleanly, so the message carries no second sentence.
        @test !occursin("return terms", msg)

        msg_rr = PO.prrm_prediction_message(ExpectedReturnRiskRatio(), mpred)
        @test occursin("MultiPeriodPredictionResult", msg_rr)
        @test occursin("MeanReturnRiskRatio", msg_rr)

        # A vector `rt` cannot collapse to a measure that carries no return estimator.
        wide = ExpectedReturn(; rt = [ArithmeticReturn(), NoReturn()])
        msg_wide = PO.prrm_prediction_message(wide, predv)
        @test occursin("`rt` holds 2 return terms", msg_wide)
        @test occursin("Scalarise the terms yourself", msg_wide)
    end
end

@testset "#549: `brinson_attribution`" begin
    # Four assets in two classes. `X` holds PRICES, so the period return of an asset is the
    # ratio of its last value in the range to its first, less one.
    ts = [Date(2020, 1, 1), Date(2020, 1, 2), Date(2020, 1, 3), Date(2020, 1, 4)]
    P = [100.0 50.0 20.0 10.0
         102.0 51.0 21.0 10.5
         104.0 49.0 22.0 11.0
         110.0 56.0 23.0 12.5]
    Xta = TimeArray(ts, P, [:A, :B, :C, :D])
    ac = DataFrame(; Asset = ["A", "B", "C", "D"], Class = ["Eq", "Eq", "Bd", "Bd"])
    wp = [0.4, 0.2, 0.3, 0.1]
    wb = [0.25, 0.25, 0.25, 0.25]
    ret = [110.0 / 100 - 1, 56.0 / 50 - 1, 23.0 / 20 - 1, 12.5 / 10 - 1]

    @testset "`X` is prices, and the four terms are the hand-computed ones" begin
        df = brinson_attribution(Xta, wp, wb, ac, :Class)
        ret_b = dot(ret, wb)
        # 0.155, the benchmark return of the whole universe.
        @test isapprox(ret_b, 0.155)
        w_i = wp[1] + wp[2]
        wb_i = wb[1] + wb[2]
        ret_i = (wp[1] * ret[1] + wp[2] * ret[2]) / w_i
        ret_b_i = (wb[1] * ret[1] + wb[2] * ret[2]) / wb_i
        AA = (w_i - wb_i) * (ret_b_i - ret_b)
        SS = wb_i * (ret_i - ret_b_i)
        In = (w_i - wb_i) * (ret_i - ret_b_i)
        @test isapprox(df[1, "Eq"], AA)
        @test isapprox(df[2, "Eq"], SS)
        @test isapprox(df[3, "Eq"], In)
        @test isapprox(df[4, "Eq"], AA + SS + In)
        # -0.0045 and -0.0065 are the REPL's own numbers for the first and the last row.
        @test isapprox(df[1, "Eq"], -0.0045)
        @test isapprox(df[4, "Eq"], -0.0065)
        @test names(df) == ["index", "Eq", "Bd", "Total"]
    end

    @testset "the identity holds, per class and overall" begin
        df = brinson_attribution(Xta, wp, wb, ac, :Class)
        for class in ("Eq", "Bd", "Total")
            @test isapprox(df[4, class], df[1, class] + df[2, class] + df[3, class])
        end
        # The `Total` column is the row sum over the class columns.
        for i in 1:4
            @test isapprox(df[i, "Total"], df[i, "Eq"] + df[i, "Bd"])
        end
        # And the overall total excess return is the portfolio's excess over the benchmark.
        @test isapprox(df[4, "Total"], dot(wp, ret) - dot(wb, ret))
        @test isapprox(df[4, "Total"], -0.021)
    end

    @testset "both dates filter, and one date alone does not" begin
        df = brinson_attribution(Xta, wp, wb, ac, :Class, Date(2020, 1, 2),
                                 Date(2020, 1, 3))
        ret23 = [104.0 / 102 - 1, 49.0 / 51 - 1, 22.0 / 21 - 1, 11.0 / 10.5 - 1]
        @test isapprox(df[4, "Total"], dot(wp, ret23) - dot(wb, ret23))
        @test isapprox(df[4, "Total"], 0.00014005602240894574)
        # A series read as returns rather than prices gives a different number, which is
        # what settles the question the ticket asks.
        raw = vec(values(Xta[3]))
        @test !isapprox(df[4, "Total"], dot(wp, raw) - dot(wb, raw))
        whole = brinson_attribution(Xta, wp, wb, ac, :Class)
        @test brinson_attribution(Xta, wp, wb, ac, :Class, Date(2020, 1, 2), nothing) ==
              whole
        @test brinson_attribution(Xta, wp, wb, ac, :Class, nothing, Date(2020, 1, 3)) ==
              whole
    end

    @testset "a zero-weight class is a `NaN`, and neither division is guarded" begin
        # A class holding zero PORTFOLIO weight makes `ret_i` a `NaN`, so the three rows
        # that read it are `NaN`. `AA_i` reads only the benchmark, so it stays finite.
        dfz = brinson_attribution(Xta, [0.5, 0.5, 0.0, 0.0], wb, ac, :Class)
        @test isfinite(dfz[1, "Bd"])
        @test all(isnan, [dfz[2, "Bd"], dfz[3, "Bd"], dfz[4, "Bd"]])
        @test all(isnan, [dfz[2, "Total"], dfz[3, "Total"], dfz[4, "Total"]])
        @test isfinite(dfz[1, "Total"])
        # A class holding zero BENCHMARK weight makes `ret_b_i` a `NaN`, and that one
        # reaches all four rows because `AA_i` reads it too.
        dfb = brinson_attribution(Xta, wp, [0.5, 0.5, 0.0, 0.0], ac, :Class)
        @test all(isnan, [dfb[i, "Bd"] for i in 1:4])
        @test all(isnan, [dfb[i, "Total"] for i in 1:4])
    end
end

@testset "#549: every `performance_summary` route reaches the same statistics" begin
    ret = vec(ER_X * ER_W)
    base = performance_summary(ret; periods_per_year = 12)
    rd = ReturnsResult(; nx = ["A", "B", "C", "D"], X = ER_X)

    @testset "the weights-and-matrix routes" begin
        @test performance_summary(ER_W, ER_X; periods_per_year = 12).ann_return ==
              base.ann_return
        @test performance_summary(ER_W, rd; periods_per_year = 12).ann_return ==
              base.ann_return
        # The fee is charged in every period, so it moves the annualised return by
        # `periods_per_year * calc_fees`.
        fees = Fees(; l = 0.01)
        net = performance_summary(ER_W, ER_X, fees; periods_per_year = 12)
        @test isapprox(net.ann_return, base.ann_return - 12 * calc_fees(ER_W, fees))
    end

    @testset "the optimisation-result route" begin
        X2 = ER_X[:, 1:2]
        rd2 = ReturnsResult(; nx = ["A", "B"], X = X2)
        res = NaiveOptimisationResult(; pr = nothing, wb = nothing,
                                      retcode = OptimisationSuccess(), w = [0.5, 0.5],
                                      fb = nothing)
        @test performance_summary(res, rd2; periods_per_year = 12).ann_return ==
              performance_summary([0.5, 0.5], X2; periods_per_year = 12).ann_return
    end

    @testset "the prediction-result routes read the realised series" begin
        res = NaiveOptimisationResult(; pr = nothing, wb = nothing,
                                      retcode = OptimisationSuccess(), w = [0.5, 0.5],
                                      fb = nothing)
        rdv = PredictionReturnsResult(; nx = ["A", "B"], X = [0.01, -0.02, 0.03, 0.005],
                                      ts = nothing)
        rdvv = PredictionReturnsResult(; nx = ["A", "B"],
                                       X = [[0.01, -0.02], [0.03, 0.005]], ts = nothing)
        predv = PredictionResult(; res = res, rd = rdv)
        predvv = PredictionResult(; res = res, rd = rdvv)
        mpred = MultiPeriodPredictionResult(; pred = [predv])
        # 0.025 = mean([0.01, -0.02, 0.03, 0.005]) * 4.
        @test isapprox(performance_summary(predv; periods_per_year = 4).ann_return, 0.025)
        # A vector of vectors takes the first one: mean([0.01, -0.02]) * 4 = -0.02.
        @test isapprox(performance_summary(predvv; periods_per_year = 4).ann_return, -0.02)
        @test isapprox(performance_summary(mpred; periods_per_year = 4).ann_return, 0.025)
    end
end

@testset "#773: `performance_summary` reads a weight path, and the caveat is written" begin
    # Decision #772: the weight argument's type is the picker. `performance_summary` is
    # bound `ArrNum`, so it admits a matrix already, and the gap was one method deep at the
    # base verb. `calc_net_returns(U, X, fees)` closes it, so this scorer gains no method.
    PO = PortfolioOptimisers
    X = [0.01 0.02; -0.02 0.01; 0.03 -0.01; 0.005 0.02]
    w = [0.6, 0.4]
    fees = Fees(; l = 0.001, fl = 0.002)
    wd = SelfFinancingDrift()
    U = PO.weight_path(wd, w, X)
    Uc = PO.weight_path(nothing, w, X)

    # A path is scored as the series that path produces, on the method that already takes a
    # series. Every statistic agrees, not only the first.
    ps_path = performance_summary(U, X, fees; periods_per_year = 4)
    ps_ret = performance_summary(calc_net_returns(U, X, fees); periods_per_year = 4)
    for f in fieldnames(typeof(ps_path))
        @test isequal(getfield(ps_path, f), getfield(ps_ret, f))
    end

    # It is the series the drift route forms from the same window under the same drift.
    @test isapprox(ps_path.ann_return,
                   performance_summary(calc_net_returns(w, X, fees, wd);
                                       periods_per_year = 4).ann_return)

    # A drifted path moves the statistics, and a constant path reproduces the target read.
    ps_target = performance_summary(w, X, fees; periods_per_year = 4)
    @test ps_path.ann_return != ps_target.ann_return
    @test isapprox(performance_summary(Uc, X, fees; periods_per_year = 4).ann_return,
                   ps_target.ann_return)

    # The returns-result route forwards its weights to the same base verb.
    rd = ReturnsResult(; nx = ["A", "B"], X = X)
    @test performance_summary(U, rd, fees; periods_per_year = 4).ann_return ==
          ps_path.ann_return

    # The caveat rides in both of the places #772 named: the field text every renderer
    # inherits, and the docstring beside the non-normality paragraph. `sharpe_stderr`
    # corrects for the third and fourth moments and not for serial dependence, and a
    # drifted series is serially dependent through the weights it held.
    @test occursin("serial dependence", PO.field_dict[:ps_sharpe_stderr])
    doc = string(Base.Docs.doc(Base.Docs.Binding(PO, :performance_summary)))
    @test occursin("serial dependence", doc)
    @test occursin("Weight Drift", doc)
    @test occursin("long-run variance estimator", doc)
end
