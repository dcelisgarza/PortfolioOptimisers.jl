#=
Check `lookback`, the number of trailing observations that the value of an estimator at one
observation reads. The generic function and its fallback are in
`src/05_Moments/32_CrossSectionalFactorModel/04_FactorExposures/01_Base_Descriptor.jl`, and each
family states its method next to its `descriptor` or `factor_exposure` method. Issue #1470,
map #1375, ADR 0193.

TWO KINDS OF PROBE.

1. THE VALUE. Each family returns the count its algorithm reads: one for a Descriptor of its
   own row, `lag + 1` for a lag Descriptor, `skip + window` and `window` for a rolling one, and
   `nothing` for a recursion from the first row or for an estimator that states no look-back.
   A list takes the largest count, and `nothing` wins.

2. THE CUT. For every finite member, the value at the last row of the synthetic panel,
   computed on the last `lookback` rows alone, equals the value computed on every row, `NaN`
   for `NaN`. It is equal bit for bit, except where a member reads a rolling log return. That
   Descriptor differences two cumulative sums that start at the first row of its input, so a
   cut input rounds the difference in a different way. Measured: at most 5e-15 relative, as a
   vector norm, at 320 and at 2520 observations, and 2.4e-13 on the worst single cell, which is
   near zero. So those members are compared under `rtol = 1e-12` on the vector. One row fewer
   loses the value of a lag or a rolling member, so the count is also the smallest one that
   holds.
=#
include(joinpath(@__DIR__, "test06c_setup.jl"))
const PO = PortfolioOptimisers

# A Descriptor that states no look-back of its own takes the fallback.
struct LookbackProbeDescriptor <: PO.AbstractDescriptorEstimator end

# The returns data cut to its last `L` observations, with the Asset Panel beside it.
function lookback_cut(rd::ReturnsResult, L::Integer)
    T = size(rd.X, 1)
    return PO.port_opt_view(rd, (T - L + 1):T, :)
end

# The value on the cut input against the value on the full input: the `NaN` pattern exactly,
# and the values bit for bit when `exact`, or under `rtol = 1e-12` on the vector otherwise.
function lookback_agrees(cut::AbstractVector, full::AbstractVector, exact::Bool)
    if exact
        return isequal(cut, full)
    end
    f = isfinite.(full)
    return isequal(isnan.(cut), isnan.(full)) && isapprox(cut[f], full[f]; rtol = 1e-12)
end

@testset "The look-back of each Descriptor family" begin
    @testset "A Descriptor of its own row reads one row" begin
        for de in (BookToPrice(), EarningsToPrice(), ShareholderYield(), LogMarketCap(),
                   PanelFieldLog(; field = "market_cap"), Passthrough(; field = "market_cap"))
            @test PO.lookback(de) === 1
        end
    end
    @testset "A lag Descriptor reads its lag and its own row" begin
        @test PO.lookback(GrowthRate(; field = "sales_ttm", lag = 5)) === 6
        @test PO.lookback(ChangeToScale(; field = "sales_ttm", scale = "market_cap",
                                        lag = 3)) === 4
        @test PO.lookback(ChangeInIntensity(; field = "capex_ttm", scale = "total_assets",
                                            lag = 1)) === 2
        @test PO.lookback(AssetsGrowthRate()) === 253
        de = EarningsChangeToPrice()
        @test PO.lookback(de) === de.lag + 1
    end
    @testset "A rolling Descriptor reads its skip and its window" begin
        @test PO.lookback(RollingMomentum()) === 273
        @test PO.lookback(Reversal()) === 21
        @test PO.lookback(MaxReturn()) === 21
        @test PO.lookback(RollingLogReturn(; window = 10, skip = 5)) === 15
        @test PO.lookback(RollingMax(; window = 7)) === 7
    end
    @testset "A recursion from the first row is unbounded" begin
        for de in (EWMean(; decay = 0.9, min_obs = 5), EWMomentum(),
                   EWVolumeRatio(; num = "short_interest", den = "adj_volume", decay = 0.9,
                                 min_obs = 5), EWShareTurnover(), EWAmihudIlliquidity(),
                   DaysToCover(), EWVolatility(), EWDownsideVolatility(), EWResidualVolatility(),
                   EWResidualDownsideVolatility(), EWBeta(; decay = 0.9, min_obs = 5),
                   EWMarketBeta(), EWMacroSensitivity(), EWDownsideBeta())
            @test isnothing(PO.lookback(de))
        end
    end
    @testset "A Descriptor with no method of its own takes the fallback" begin
        @test isnothing(PO.lookback(LookbackProbeDescriptor()))
    end
    @testset "A list takes the largest look-back, and nothing wins" begin
        @test PO.lookback(PO.AbstractDescriptorEstimator[]) === 1
        @test PO.lookback([Reversal(), RollingMomentum(), BookToPrice()]) === 273
        @test isnothing(PO.lookback([Reversal(), EWMomentum()]))
    end
end

@testset "The look-back of each Exposure and Return Forecast family" begin
    comp = CompositeExposure(;
                             descriptors = [Reversal(),
                                            GrowthRate(; field = "sales_ttm", lag = 30)])
    @testset "Exposures" begin
        @test PO.lookback(comp) === 31
        @test isnothing(PO.lookback(CompositeExposure(;
                                                      descriptors = [Reversal(),
                                                                     EWMomentum()])))
        @test PO.lookback(ConstantExposure()) === 1
        @test PO.lookback(OneHotExposure(; field = "industry", family = "industry")) === 1
        @test PO.lookback(CurrencyExposure()) === 1
        @test PO.lookback(DerivedExposure(; source = "size", f = x -> x .^ 2)) === 1
        @test PO.lookback(ObservedExposure(; xe = comp, series = "mkt")) === 31
        @test PO.lookback(ObservedExposure(; xe = ConstantExposure(), series = "mkt")) === 1
    end
    @testset "Return Forecasts" begin
        ds = DescriptorScores(; descriptors = [Reversal(), RollingMomentum()])
        @test PO.lookback(CustomValueReturnForecast(; mu = [0.1, 0.2])) === 1
        @test PO.lookback(FixedWeightedReturnForecast(; scores = ds, scale = 1.0)) === 273
        @test isnothing(PO.lookback(FixedWeightedReturnForecast(;
                                                                scores = DescriptorScores(;
                                                                                          descriptors = [EWMomentum()]),
                                                                scale = 1.0)))
        @test isnothing(PO.lookback(ExpWeightedReturnForecast(; scores = ds)))
        @test isnothing(PO.lookback(TargetReturnForecast(; scores = ds)))
    end
end

@testset "The look-back of a Cross-Sectional Factor Prior" begin
    factors = ["market" => ConstantExposure(),
               "momentum" => CompositeExposure(; descriptors = [RollingMomentum()]),
               "value" => CompositeExposure(; descriptors = [BookToPrice()])]
    @testset "The factors plus the lag" begin
        @test PO.lookback(CrossSectionalFactorPrior(; factors = factors)) === 274
        @test PO.lookback(CrossSectionalFactorPrior(; factors = factors, lag = 5)) === 278
        @test PO.lookback(CrossSectionalFactorPrior(;
                                                    factors = ["market" =>
                                                                   ConstantExposure()])) ===
              2
    end
    @testset "An unbounded factor makes the prior unbounded" begin
        ew = vcat(factors, ["beta" => CompositeExposure(; descriptors = [EWMarketBeta()])])
        @test isnothing(PO.lookback(CrossSectionalFactorPrior(; factors = ew)))
    end
    @testset "The Return Forecast counts when it reads further back" begin
        long = DescriptorScores(; descriptors = [RollingLogReturn(; window = 400)])
        short = DescriptorScores(; descriptors = [Reversal()])
        @test PO.lookback(CrossSectionalFactorPrior(; factors = factors,
                                                    rfe = FixedWeightedReturnForecast(;
                                                                                      scores = long,
                                                                                      scale = 1.0))) ===
              400
        @test PO.lookback(CrossSectionalFactorPrior(; factors = factors,
                                                    rfe = FixedWeightedReturnForecast(;
                                                                                      scores = short,
                                                                                      scale = 1.0))) ===
              274
        @test PO.lookback(CrossSectionalFactorPrior(; factors = factors,
                                                    rfe = CustomValueReturnForecast(;
                                                                                    mu = [0.1]))) ===
              274
        @test isnothing(PO.lookback(CrossSectionalFactorPrior(; factors = factors,
                                                              rfe = ExpWeightedReturnForecast(;
                                                                                              scores = short))))
    end
end

@testset "The last rows alone give the value at the last row" begin
    rd = synthetic_asset_panel(; n_assets = 20, n_observations = 320, n_industries = 3,
                               rng = StableRNG(1470)).rd
    @testset "Descriptors" begin
        for (de, exact) in
            ((BookToPrice(), true), (EarningsToPrice(), true), (LogMarketCap(), true),
             (Passthrough(; field = "market_cap"), true),
             (GrowthRate(; field = "sales_ttm", lag = 5), true),
             (ChangeToScale(; field = "net_income_ttm", scale = "market_cap", lag = 3),
              true),
             (ChangeInIntensity(; field = "capex_ttm", scale = "total_assets", lag = 4),
              true), (RollingMomentum(), false), (Reversal(), false), (MaxReturn(), true),
             (RollingLogReturn(; window = 10, skip = 5), false))
            L = PO.lookback(de)
            D = descriptor(de, rd)
            Dc = descriptor(de, lookback_cut(rd, L))
            @test size(Dc, 1) == L
            @test any(isfinite, D[end, :])
            @test lookback_agrees(Dc[end, :], D[end, :], exact)
            if L > 1
                @test all(isnan, descriptor(de, lookback_cut(rd, L - 1))[end, :])
            end
        end
    end
    @testset "Exposures" begin
        size_xe = CompositeExposure(; descriptors = [LogMarketCap()], bw = "market_cap")
        for (xe, exact) in ((CompositeExposure(;
                                               descriptors = [RollingMomentum(), BookToPrice(),
                                                              GrowthRate(; field = "sales_ttm", lag = 5)],
                                               bw = "market_cap"), false),
                            (CompositeExposure(; descriptors = [Reversal(), MaxReturn()],
                                               bw = "market_cap", group = "industry"), false),
                            (size_xe, true),
                            (OneHotExposure(; field = "industry", family = "industry"), true),
                            (ConstantExposure(), true),
                            (ObservedExposure(;
                                              xe = CompositeExposure(; descriptors = [Reversal()],
                                                                     bw = "market_cap"), series = "mkt"),
                             false))
            L = PO.lookback(xe)
            B = factor_exposure(xe, rd)
            Bc = factor_exposure(xe, lookback_cut(rd, L))
            @test size(Bc, 1) == L
            @test any(isfinite, selectdim(B, 1, size(B, 1)))
            @test lookback_agrees(vec(selectdim(Bc, 1, L)),
                                  vec(selectdim(B, 1, size(B, 1))), exact)
        end
        # A derived member reads the exposure of its source on the same rows, so the prior
        # counts the rows of the source.
        de = DerivedExposure(; source = "size", f = x -> x .^ 2, bw = "market_cap")
        L = max(PO.lookback(de), PO.lookback(size_xe))
        rc = lookback_cut(rd, L)
        B = factor_exposure(de, rd, factor_exposure(size_xe, rd))
        Bc = factor_exposure(de, rc, factor_exposure(size_xe, rc))
        @test any(isfinite, B[end, :])
        @test isequal(Bc[end, :], B[end, :])
    end
end
