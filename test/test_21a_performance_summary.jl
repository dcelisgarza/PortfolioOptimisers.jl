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
