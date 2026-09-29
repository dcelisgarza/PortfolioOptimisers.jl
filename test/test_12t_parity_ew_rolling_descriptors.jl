#=
Parity of map #1375 for the exponentially weighted and the rolling Descriptors (#1380). Every
`Parity_*` file this test reads is an output of the oracle, stored with the harness of #1376,
and each case in the table below states how it was made. The conventions of the harness are in
the comment that #1376 links; this file does not repeat them.

Every case reads the raw exchange: a Panel Field is `NaN` where it is not observed, because a
Descriptor reads its Panel Fields through the observed mask and never sees a fill value. The
small fixture holds a late listing, a delisting, a relisting, a holiday gap, an asset outside
the estimation mask and a tie, so each recursion meets each of them.

THE CLEAN FIXTURE. The fixture carries a negative volume and a negative short interest. The
oracle refuses both in `EWShareTurnover`, `EWAmihudIlliquidity` and `DaysToCover`, and since
#1380 the library refuses them too, through the `nonneg` guard that each of the three presets
sets on every Panel Field it reads. So their parity cases read `parity_1380_clean`, which
makes the two cells positive, and a testset below pins the refusal on the raw cells.

THE MEASURE. All 55 cases of the ticket agreed on the `NaN` pattern. The largest relative
difference of each family, over the cases stored here and the large-panel defaults measured
with them:

| Family | `maxrel` |
| --- | --- |
| `EWMean`, `EWMomentum` | 4.2e-13 (large panel, default) |
| `EWVolumeRatio`, `EWShareTurnover`, `EWAmihudIlliquidity`, `DaysToCover` | 0.0 |
| `EWVolatility`, `EWDownsideVolatility` | 3.3e-16 |
| `EWResidualVolatility`, `EWResidualDownsideVolatility` | 7.5e-16 |
| `EWBeta`, `EWMarketBeta` | 7.4e-14 |
| `EWMacroSensitivity` | 1.9e-13 |
| `EWDownsideBeta` | 1.1e-15 |
| `RollingLogReturn`, `RollingMomentum`, `Reversal` | 3.0e-12 cell by cell, 9.6e-17 against the largest entry |
| `RollingMax`, `MaxReturn` | 0.0 |

A rolling log return is the difference of two cumulative sums, on both sides. A window whose
sum is near zero keeps the round-off of the cumulative sums, so its relative error is large
while the error against the largest entry stays at the last bit. The rolling cases therefore
compare with `scale = :array`.
=#
include(joinpath(@__DIR__, "parity_harness.jl"))

# The fixture with its negative volume and its negative short interest made positive.
function parity_1380_clean(fx)
    rd = deepcopy(fx.rd)
    at = fx.at
    fv = panel_field(rd.pnl, "adj_volume").vals
    fv[at.negative_volume...] = abs(fv[at.negative_volume...])
    fs = panel_field(rd.pnl, "short_interest").vals
    fs[at.negative_short_interest...] = abs(fs[at.negative_short_interest...])
    return rd
end

# The MACRO series of the Exogenous Series, with a gap at observations 10, 30 and 31.
function parity_1380_gap(r::AbstractVector)
    g = copy(r)
    g[[10, 30, 31]] .= NaN
    return g
end

@testset "Parity: the exponentially weighted and the rolling Descriptors (#1380)" begin
    fxs = parity_small_panel()
    rds = fxs.rd
    rdsc = parity_1380_clean(fxs)
    rdl = parity_large_panel().rd
    macro_s = rds.E[:, findfirst(==("MACRO"), rds.ne)]
    # A volatility estimator whose warm-up is set apart from its half-life.
    vol(h, w) = RegimeAdjustedExpWeightedVariance(;
                                                  decay = PortfolioOptimisers.half_life_decay(h),
                                                  min_obs = w, centred = true,
                                                  regime_method = nothing)
    # Each case: the unit, the case, the returns data, and the Descriptor. The case names the
    # fixture (the small raw panel, its clean copy, or the large raw panel) and the
    # configuration. The output is the Descriptor, `observations × assets`.
    cases = [("EWMomentum", "SmallH5S3", rds,
              r -> descriptor(EWMomentum(; half_life = 5, skip = 3), r)),
             ("EWMomentum", "SmallH5Min8Exp", rds,
              r -> descriptor(EWMomentum(; half_life = 5, skip = 0, min_obs = 8,
                                         exponentiate = true), r)),
             ("EWMomentum", "LargeDefault", rdl, r -> descriptor(EWMomentum(), r)),
             # The general form at a decay no half-life of the presets names.
             ("EWMean", "SmallDecay085", rds,
              r -> descriptor(EWMean(; decay = 0.85, min_obs = 6, skip = 2), r)),
             ("EWShareTurnover", "SmallCleanDefault", rdsc,
              r -> descriptor(EWShareTurnover(), r)),
             ("EWShareTurnover", "SmallCleanH5", rdsc,
              r -> descriptor(EWShareTurnover(; half_life = 5), r)),
             ("EWVolumeRatio", "SmallCleanDecay08", rdsc,
              r -> descriptor(EWVolumeRatio(; num = "adj_volume",
                                            den = "adj_shares_outstanding", decay = 0.8,
                                            min_obs = 4), r)),
             ("EWAmihudIlliquidity", "SmallCleanDefault", rdsc,
              r -> descriptor(EWAmihudIlliquidity(), r)),
             ("EWAmihudIlliquidity", "SmallCleanH5", rdsc,
              r -> descriptor(EWAmihudIlliquidity(; half_life = 5), r)),
             ("DaysToCover", "SmallCleanDefault", rdsc, r -> descriptor(DaysToCover(), r)),
             ("DaysToCover", "SmallCleanH5Min2", rdsc,
              r -> descriptor(DaysToCover(; half_life = 5, min_obs = 2), r)),
             ("EWVolatility", "SmallDefault", rds, r -> descriptor(EWVolatility(), r)),
             ("EWVolatility", "SmallH5", rds,
              r -> descriptor(EWVolatility(; half_life = 5), r)),
             ("EWVolatility", "SmallH8Min3", rds,
              r -> descriptor(EWVolatility(; ce = vol(8, 3)), r)),
             ("EWDownsideVolatility", "SmallDefault", rds,
              r -> descriptor(EWDownsideVolatility(), r)),
             ("EWDownsideVolatility", "SmallH5Mar", rds,
              r -> descriptor(EWDownsideVolatility(; half_life = 5, mar = 0.001), r)),
             ("EWResidualVolatility", "SmallDefault", rds,
              r -> descriptor(EWResidualVolatility(), r)),
             ("EWResidualVolatility", "SmallH5B8", rds,
              r -> descriptor(EWResidualVolatility(; half_life = 5, beta_half_life = 8), r)),
             ("EWResidualDownsideVolatility", "SmallDefault", rds,
              r -> descriptor(EWResidualDownsideVolatility(), r)),
             ("EWResidualDownsideVolatility", "SmallH5B8Mar", rds,
              r -> descriptor(EWResidualDownsideVolatility(; half_life = 5,
                                                           beta_half_life = 8,
                                                           mar = -0.001), r)),
             ("EWMarketBeta", "SmallDefault", rds, r -> descriptor(EWMarketBeta(), r)),
             ("EWMarketBeta", "SmallH8", rds,
              r -> descriptor(EWMarketBeta(; half_life = 8), r)),
             # Groups of three or four assets: a group prior at `min_group_size = 3`, and the
             # global prior for every group at the default of five.
             ("EWMarketBeta", "SmallH8Group3", rds,
              r -> descriptor(EWMarketBeta(; half_life = 8, group = "industry",
                                           min_group_size = 3), r)),
             ("EWMarketBeta", "SmallH8Group5", rds,
              r -> descriptor(EWMarketBeta(; half_life = 8, group = "industry"), r)),
             ("EWMarketBeta", "SmallH4Agg3", rds,
              r -> descriptor(EWMarketBeta(; half_life = 4, agg_obs = 3), r)),
             ("EWMarketBeta", "SmallH4Agg3Group3", rds,
              r -> descriptor(EWMarketBeta(; half_life = 4, agg_obs = 3, group = "industry",
                                           min_group_size = 3, bounds = (0.2, 0.9)), r)),
             ("EWBeta", "SmallDecay09", rds,
              r -> descriptor(EWBeta(; decay = 0.9, min_obs = 6), r)),
             # The reference return through `ref` and through `series`.
             ("EWMacroSensitivity", "SmallDefault", rds,
              r -> descriptor(EWMacroSensitivity(), r; ref = macro_s)),
             ("EWMacroSensitivity", "SmallH8", rds,
              r -> descriptor(EWMacroSensitivity(; series = "MACRO", half_life = 8), r)),
             ("EWMacroSensitivity", "SmallH8Gap", rds,
              r -> descriptor(EWMacroSensitivity(; half_life = 8), r;
                              ref = parity_1380_gap(macro_s))),
             ("EWMacroSensitivity", "SmallH4Agg3Gap", rds,
              r -> descriptor(EWMacroSensitivity(; half_life = 4, agg_obs = 3), r;
                              ref = parity_1380_gap(macro_s))),
             ("EWDownsideBeta", "SmallDefault", rds, r -> descriptor(EWDownsideBeta(), r)),
             ("EWDownsideBeta", "SmallH8Mar", rds,
              r -> descriptor(EWDownsideBeta(; half_life = 8, mar = -0.002), r)),
             ("RollingMomentum", "SmallW10S3", rds,
              r -> descriptor(RollingMomentum(; window = 10, skip = 3), r)),
             ("RollingMomentum", "SmallW10Exp", rds,
              r -> descriptor(RollingMomentum(; window = 10, skip = 0, exponentiate = true),
                              r)),
             ("RollingMomentum", "LargeW120", rdl,
              r -> descriptor(RollingMomentum(; window = 120), r)),
             ("Reversal", "SmallDefault", rds, r -> descriptor(Reversal(), r)),
             ("Reversal", "SmallW5", rds, r -> descriptor(Reversal(; window = 5), r)),
             ("RollingLogReturn", "SmallW7S2NegExp", rds,
              r -> descriptor(RollingLogReturn(; window = 7, skip = 2, sign = -1,
                                               exponentiate = true), r)),
             ("MaxReturn", "SmallDefault", rds, r -> descriptor(MaxReturn(), r)),
             ("MaxReturn", "SmallW5", rds, r -> descriptor(MaxReturn(; window = 5), r)),
             ("RollingMax", "SmallW2", rds, r -> descriptor(RollingMax(; window = 2), r))]
    rolling = ("RollingMomentum", "Reversal", "RollingLogReturn")
    for (unit, case, rd, f) in cases
        @testset "$unit $case" begin
            D = f(rd)
            O = parity_load(unit, case, "D")
            scale = unit in rolling ? :array : :cell
            r = parity_compare(D, O; scale = scale, name = "$(unit)_$(case)")
            @test r.pattern
            @test r.ok
        end
    end

    @testset "An inactive cell is missing on both sides, so no recursion reads it (#719)" begin
        # The oracle refuses a finite return outside the active mask, and the library masks the
        # returns before a recursion starts. The cases above cross a delisting and a
        # relisting; here the delisted asset of the fixture is `NaN` to its last observation.
        (i, t) = fxs.at.delist
        for (unit, case, rd, f) in cases
            rd === rds || continue
            @test all(isnan, f(rd)[(t + 1):end, i])
        end
    end

    @testset "A rolling maximum over one observation is the return itself (#720)" begin
        # The oracle refuses `window = 1`, a window that is well defined: the maximum of one
        # return. The library gives it, so the unit is better than the oracle there.
        D = descriptor(RollingMax(; window = 1), rds)
        E = ifelse.(rds.pnl.amsk, rds.X, NaN)
        @test isequal(D, E)
        @test isequal(descriptor(MaxReturn(; window = 1), rds), E)
    end

    @testset "A negative volume or short interest is refused, as the oracle refuses it" begin
        # A volume, a share count, a price and a short interest cannot be negative, so the
        # presets refuse a negative value of every Panel Field they read. Before #1380 the
        # share turnover read the negative volume as a negative turnover, and the days to
        # cover gave a negative Descriptor.
        for de in (EWShareTurnover(; half_life = 5), EWAmihudIlliquidity(; half_life = 5),
                   DaysToCover(; half_life = 5))
            @test_throws DomainError descriptor(de, rds)
        end
        @test_throws r"short_interest" descriptor(DaysToCover(;
                                                              nonneg = ["short_interest"]),
                                                  rds)
        @test_throws r"adj_volume" descriptor(DaysToCover(; nonneg = ["adj_volume"]), rds)
        # With the guard off, a negative volume in a denominator holds the state, and a
        # negative volume in a numerator enters the mean.
        (t, i) = fxs.at.negative_volume
        Dc = descriptor(EWAmihudIlliquidity(; half_life = 5), rdsc)
        Da = descriptor(EWAmihudIlliquidity(; half_life = 5, nonneg = nothing), rds)
        @test Da[t, i] == Da[t - 1, i]
        @test Da[t, i] != Dc[t, i]
        Dt = descriptor(EWShareTurnover(; half_life = 5, nonneg = nothing), rds)
        @test Dt[t, i] < Dt[t - 1, i]
        (ts, is) = fxs.at.negative_short_interest
        Dd = descriptor(DaysToCover(; half_life = 5, nonneg = nothing), rds)
        @test Dd[ts, is] < 0
        # The guard reads the active observed cells alone (ADR 0108): a negative volume that is
        # observed on an inactive cell reaches no output, so it is not refused, and the
        # Descriptor equals the one of the clean panel.
        rdo = deepcopy(rdsc)
        (i3, t3) = fxs.at.delist
        f = panel_field(rdo.pnl, "adj_volume")
        f.vals[t3 + 5, i3] = -1.0
        f.omsk[t3 + 5, i3] = true
        for de in (EWShareTurnover(; half_life = 5), EWAmihudIlliquidity(; half_life = 5),
                   DaysToCover(; half_life = 5))
            @test isequal(descriptor(de, rdo), descriptor(de, rdsc))
        end
        # The generic ratio does not guard by default: a difference of Panel Fields can be
        # negative by design.
        @test isnothing(EWVolumeRatio(; num = ["adj_volume" => 1, "short_interest" => -1],
                                      den = "adj_shares_outstanding", decay = 0.8,
                                      min_obs = 2).nonneg)
        # A guard must name a Panel Field the ratio reads.
        @test_throws ArgumentError EWShareTurnover(; nonneg = ["adj_close"])
        @test_throws ArgumentError EWAmihudIlliquidity(;
                                                       nonneg = ["adj_shares_outstanding"])
        @test_throws ArgumentError DaysToCover(; nonneg = ["market_cap"])
        @test_throws PortfolioOptimisers.IsEmptyError DaysToCover(; nonneg = String[])
        @test PortfolioOptimisers.panel_term_names(nothing) == String[]
        @test PortfolioOptimisers.panel_term_names(["adj_close", "adj_volume"]) ==
              ["adj_close", "adj_volume"]
    end

    @testset "An infinite reference return is refused on both paths (#1365)" begin
        rf = copy(macro_s)
        rf[6] = Inf
        @test_throws PortfolioOptimisers.IsNonFiniteError descriptor(EWMacroSensitivity(;
                                                                                        half_life = 8),
                                                                     rds; ref = rf)
        E = copy(rds.E)
        E[6, findfirst(==("MACRO"), rds.ne)] = Inf
        rdi = ReturnsResult(; nx = rds.nx, X = rds.X, ne = rds.ne, E = E, pnl = rds.pnl)
        @test_throws PortfolioOptimisers.IsNonFiniteError descriptor(EWMacroSensitivity(;
                                                                                        series = "MACRO",
                                                                                        half_life = 8),
                                                                     rdi)
    end
end
