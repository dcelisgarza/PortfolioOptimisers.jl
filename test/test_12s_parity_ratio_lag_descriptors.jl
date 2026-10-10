#=
Parity of map #1375 for the point-in-time ratio Descriptors and the lag Descriptors (#1379). Every
`Parity_ratio_lag_descriptors_*` file this test reads is an output of the oracle, stored with the
harness of #1376, and each testset states how its case was made.

The measure found one defect, which this ticket fixes: a quotient of two finite values that
overflows gave an infinity, and every cross-sectional transform refuses an infinity, so one cell
cost the whole fit. `positive_divide` now gives `NaN` there, as the oracle does.

A data error refuses, and an undefined ratio is `NaN` (ADR 0108). A field that is positive by
construction (a price, a market capitalisation, a share count, a total of assets) refuses a value
at or below zero through the `gt0` guard of its named constructor. A denominator that valid data
can make zero or negative (a book equity, an enterprise value) gives `NaN`. Three differences
from the oracle are deliberate, and ADR 0108 records them:

  - the ratios over `total_assets` and `MarketLeverage` refuse a non-positive total of assets or
    market capitalisation, where the oracle gives `NaN`; `GrossMargin` refuses a negative sales
    figure;
  - the archetypes `ChangeToScale` and `ChangeInIntensity` give `NaN` at a non-positive scale by
    default, where the oracle refuses; their named constructors refuse, as the oracle does;
  - the non-negative guard reads the active observed cells alone, where the oracle reads every
    finite cell, a cell outside the active mask included.

Every refusal keeps the oracle's output one keyword away: `parity_unguarded` rebuilds each named
Descriptor with its `gt0` fields moved to `pos` and its sign guards off, and that estimator equals
the oracle on every case.
=#
include(joinpath(@__DIR__, "parity_harness.jl"))

# The small fixture of the harness, with the Panel Fields that the named Descriptors read by
# default and the fixture does not hold. `kind` selects the case:
#
#   - `:clean`: every guarded field is valid. The negative short interest of the fixture is made
#     positive, so `ShortInterest` has a value to compare.
#   - `:negative`: a negative value, at one active observed cell, in each field that a
#     Descriptor refuses to read below zero.
#   - `:nonpositive`: a zero or a negative value in each denominator that the oracle refuses at
#     or below zero.
function parity_ratio_case(kind::Symbol = :clean; fx = parity_small_panel())
    pnl = fx.rd.pnl
    amsk = pnl.amsk
    T, N = size(amsk)
    rng = StableRNG(1379)
    raw(name) = parity_field_rows(panel_field(pnl, name),
                                  amsk .& something(panel_field(pnl, name).omsk, amsk))
    mcap = raw("market_cap")
    shares = raw("adj_shares_outstanding")
    assets = raw("total_assets")
    income = raw("net_income_ttm")
    sales = raw("sales_ttm")
    debt = 0.3 .* assets .* exp.(0.3 .* randn(rng, 1, N) .+ 0.05 .* randn(rng, T, N))
    ocf = income .+ 0.02 .* assets .* randn(rng, T, N)
    cogs = sales .* (0.7 .+ 0.15 .* randn(rng, T, N))
    ebitda = 0.12 .* assets .* (1 .+ 0.4 .* randn(rng, T, N))
    ev = mcap .+ debt .- 0.2 .* assets .* exp.(0.5 .* randn(rng, T, N))
    eps = income ./ shares .* (1 .+ 0.1 .* randn(rng, T, N))
    dps = max.(0.4 .* eps, 0.0)
    dividends = dps .* shares
    buybacks = 0.01 .* mcap .* randn(rng, T, N)
    eps_std = 0.2 .* abs.(eps) .* exp.(0.3 .* randn(rng, T, N))
    capex = 0.05 .* assets .* exp.(0.2 .* randn(rng, T, N))
    # An enterprise value at or below zero: the ratio over it is NaN on both sides.
    ev[50, 11] = -abs(ev[50, 11])
    ev[51, 12] = 0.0
    # A zero dividend is valid, and the ratio over the price is zero.
    dividends[30:32, 1] .= 0.0
    # Blanks in an active stretch of the new fields.
    debt[45, 5] = NaN
    capex[30, 1] = NaN
    si = raw("short_interest")
    nsi = fx.at.negative_short_interest
    si[nsi...] = abs(si[nsi...])
    close = raw("adj_close")
    if kind === :negative
        dividends[25, 2] = -1.0
        dps[21, 3] = -1.0e-3
        eps_std[22, 4] = -1.0e-3
        capex[23, 5] = -1.0
        assets[24, 6] = -assets[24, 6]
        sales[25, 7] = -sales[25, 7]
        shares[26, 8] = -shares[26, 8]
        si[nsi...] = -si[nsi...]
    elseif kind === :nonpositive
        mcap[60, 1] = -mcap[60, 1]
        mcap[61, 2] = 0.0
        close[62, 5] = 0.0
        shares[63, 6] = 0.0
        assets[64, 7] = 0.0
        sales[65, 9] = 0.0
    end
    numeric = ["market_cap" => mcap, "adj_close" => close,
               "adj_shares_outstanding" => shares, "short_interest" => si,
               "book_equity" => raw("book_equity"), "total_assets" => assets,
               "net_income_ttm" => income, "sales_ttm" => sales, "style1" => raw("style1"),
               "total_debt" => debt, "operating_cash_flow_ttm" => ocf,
               "cost_of_revenue_ttm" => cogs, "ebitda_ttm" => ebitda,
               "enterprise_value" => ev, "eps_ntm" => eps, "dps_ntm" => dps,
               "dividends_ttm" => dividends, "net_buybacks_ttm" => buybacks,
               "eps_ntm_std" => eps_std, "capex_ttm" => capex]
    for (_, A) in numeric
        A[.!amsk] .= NaN
    end
    pf = [NumericPanelInput(; name = k, vals = A, alg = ForwardPanelFill(; val = 0.0))
          for (k, A) in numeric]
    rd = ReturnsResult(; nx = fx.rd.nx, X = fx.rd.X,
                       pnl = asset_panel(pf; amsk = amsk, emsk = pnl.emsk))
    return (; rd, amsk)
end

# Each Descriptor of the ticket, under the name of its column in the stored files. A named
# constructor reads its default Panel Fields, which are the field names of the oracle. The lag
# Descriptors run at lag 1 and at lag 30, which reaches back across the inactive stretch of
# asset 4.
function parity_ratio_descriptors()
    ds = Pair{String, Any}["BookToPrice" => BookToPrice(),
                           "CashFlowToPrice" => CashFlowToPrice(),
                           "SalesToPrice" => SalesToPrice(),
                           "EarningsToPrice" => EarningsToPrice(),
                           "ForwardEarningsToPrice" => ForwardEarningsToPrice(),
                           "EbitdaToEnterpriseValue" => EbitdaToEnterpriseValue(),
                           "DividendToPrice" => DividendToPrice(),
                           "ForwardDividendToPrice" => ForwardDividendToPrice(),
                           "ShareholderYield" => ShareholderYield(),
                           "BookLeverage" => BookLeverage(),
                           "MarketLeverage" => MarketLeverage(),
                           "DebtToAssets" => DebtToAssets(),
                           "GrossProfitability" => GrossProfitability(),
                           "GrossMargin" => GrossMargin(),
                           "ReturnOnAssets" => ReturnOnAssets(),
                           "ReturnOnEquity" => ReturnOnEquity(),
                           "AssetTurnover" => AssetTurnover(),
                           "CashFlowToAssets" => CashFlowToAssets(),
                           "SalesToEnterpriseValue" => SalesToEnterpriseValue(),
                           "AccrualsCashFlow" => AccrualsCashFlow(),
                           "AnalystDispersionToPrice" => AnalystDispersionToPrice(),
                           "LogMarketCap" => LogMarketCap(),
                           "ShortInterest" => ShortInterest(),
                           "PassthroughStyle1" => Passthrough(; field = "style1"),
                           "PassthroughBookEquity" => Passthrough(; field = "book_equity")]
    for L in (1, 30)
        append!(ds,
                ["AssetsGrowthRateL$L" => AssetsGrowthRate(; lag = L),
                 "SalesGrowthRateL$L" => SalesGrowthRate(; lag = L),
                 "IssuanceGrowthRateL$L" => IssuanceGrowthRate(; lag = L),
                 "EarningsChangeToPriceL$L" => EarningsChangeToPrice(; lag = L),
                 "CapexToAssetsChangeInIntensityL$L" =>
                     CapexToAssetsChangeInIntensity(; lag = L),
                 "GrowthRateCapexL$L" => GrowthRate(; field = "capex_ttm", lag = L),
                 "ChangeToScaleSalesAssetsL$L" =>
                     ChangeToScale(; field = "sales_ttm", scale = "total_assets", lag = L),
                 "ChangeInIntensityDebtMcapL$L" =>
                     ChangeInIntensity(; field = "total_debt", scale = "market_cap",
                                       lag = L)])
    end
    return ds
end

# The route with no refusal: a ratio with its `gt0` fields moved to `pos` and no `nonneg`
# guard, a logarithm and a scale with `gt0 = false`. It gives NaN in every cell that a guard
# would refuse, which is the oracle's rule for a denominator it does not refuse.
function parity_unguarded(de::PanelFieldRatio)
    pos = unique([something(de.pos, String[]); something(de.gt0, String[])])
    return PanelFieldRatio(; num = de.num, den = de.den, pos = isempty(pos) ? nothing : pos)
end
parity_unguarded(de::PanelFieldLog) = PanelFieldLog(; field = de.field)
function parity_unguarded(de::ChangeToScale)
    return ChangeToScale(; field = de.field, scale = de.scale, lag = de.lag)
end
function parity_unguarded(de::ChangeInIntensity)
    return ChangeInIntensity(; field = de.field, scale = de.scale, lag = de.lag)
end
parity_unguarded(de) = de

# The stored oracle of a case: the clean outputs, with the cells of `Changes` replaced. A row of
# `Changes` is the index of the Descriptor in `parity_ratio_descriptors()`, the linear index of the
# cell, and the oracle's value of the cell in that case.
function parity_ratio_oracle(clean::AbstractMatrix, changes::AbstractMatrix, k::Integer)
    o = clean[:, k]
    for r in eachrow(changes)
        if Int(r[1]) == k
            o[Int(r[2])] = r[3]
        end
    end
    return o
end

@testset "Parity: the point-in-time ratio Descriptors and the lag Descriptors (#1379)" begin
    PO = PortfolioOptimisers
    ds = parity_ratio_descriptors()
    names = first.(ds)
    # The clean case: `parity_ratio_case(:clean)`, every Descriptor of `ds` with its default
    # Panel Fields, one column per Descriptor, `vec` of the 80 × 12 output. Measured
    # `maxrel = 0.0` on all 41: bit-equal, and the NaN pattern is equal.
    clean = parity_load("ratio_lag_descriptors", "SmallClean", "Descriptors")
    @testset "Clean panel" begin
        c = parity_ratio_case(:clean)
        @test size(clean) == (80 * 12, length(ds))
        for (k, (name, de)) in enumerate(ds)
            @test parity_compare(vec(descriptor(de, c.rd)), clean[:, k]; name = name).ok
        end
    end
    # The non-positive case: `parity_ratio_case(:nonpositive)`. The oracle refuses 19 Descriptors,
    # and the stored output of each is the oracle run on the panel with the non-positive cells of
    # its refused field blanked to NaN, which is the output under the oracle's own NaN rule. The
    # 22 named Descriptors below refuse the panel through `gt0`. The four generic lag archetypes
    # have `gt0 = false` and give NaN. Every Descriptor with no refusal, `parity_unguarded`,
    # equals the stored oracle. Measured `maxrel = 0.0`.
    @testset "A non-positive price, market cap, share count or total assets refuses" begin
        refused = ["BookToPrice", "CashFlowToPrice", "SalesToPrice", "EarningsToPrice",
                   "ForwardEarningsToPrice", "DividendToPrice", "ForwardDividendToPrice",
                   "ShareholderYield", "MarketLeverage", "DebtToAssets",
                   "GrossProfitability", "ReturnOnAssets", "AssetTurnover",
                   "CashFlowToAssets", "AccrualsCashFlow", "AnalystDispersionToPrice",
                   "LogMarketCap", "ShortInterest", "EarningsChangeToPriceL1",
                   "CapexToAssetsChangeInIntensityL1", "EarningsChangeToPriceL30",
                   "CapexToAssetsChangeInIntensityL30"]
        @test refused ⊆ names
        c = parity_ratio_case(:nonpositive)
        changes = parity_load("ratio_lag_descriptors", "SmallNonpositive", "Changes")
        for (k, (name, de)) in enumerate(ds)
            oracle = parity_ratio_oracle(clean, changes, k)
            if name in refused
                @test_throws DomainError descriptor(de, c.rd)
            else
                @test parity_compare(vec(descriptor(de, c.rd)), oracle; name = name).ok
            end
            @test parity_compare(vec(descriptor(parity_unguarded(de), c.rd)), oracle;
                                 name = "unguarded " * name).ok
        end
        # Without the guard, the cell of a non-positive market capitalisation is NaN, and its
        # neighbours are not.
        D = descriptor(parity_unguarded(BookToPrice()), c.rd)
        @test isnan(D[60, 1]) && isnan(D[61, 2])
        @test isfinite(D[59, 1]) && isfinite(D[62, 2])
        # A book equity at or below zero is a state of the firm, not a data error: the
        # fixture holds both, and the return on equity is NaN there with no refusal.
        R = descriptor(ReturnOnEquity(), c.rd)
        @test isnan(R[40, 7]) && all(isnan, R[41:50, 8])
    end
    # The negative case: `parity_ratio_case(:negative)`. Ours and the oracle refuse the same 13
    # Descriptors, whose negative field the oracle also refuses. Ours also refuses the nine
    # below, over a negative total of assets or a negative sales figure, where the oracle gives
    # NaN or refuses the scale; the stored output of each is the oracle's, and
    # `parity_unguarded` equals it. Measured `maxrel = 0.0`.
    @testset "Negative guarded fields refuse" begin
        both = ["DividendToPrice", "ForwardDividendToPrice", "ShareholderYield",
                "AnalystDispersionToPrice", "ShortInterest", "AssetsGrowthRateL1",
                "SalesGrowthRateL1", "IssuanceGrowthRateL1", "GrowthRateCapexL1",
                "AssetsGrowthRateL30", "SalesGrowthRateL30", "IssuanceGrowthRateL30",
                "GrowthRateCapexL30"]
        ours = ["DebtToAssets", "GrossProfitability", "GrossMargin", "ReturnOnAssets",
                "AssetTurnover", "CashFlowToAssets", "AccrualsCashFlow",
                "CapexToAssetsChangeInIntensityL1", "CapexToAssetsChangeInIntensityL30"]
        @test both ⊆ names && ours ⊆ names
        c = parity_ratio_case(:negative)
        changes = parity_load("ratio_lag_descriptors", "SmallNegative", "Changes")
        for (k, (name, de)) in enumerate(ds)
            if name in both
                @test_throws DomainError descriptor(de, c.rd)
                continue
            end
            oracle = parity_ratio_oracle(clean, changes, k)
            if name in ours
                @test_throws DomainError descriptor(de, c.rd)
            else
                @test parity_compare(vec(descriptor(de, c.rd)), oracle; name = name).ok
            end
            @test parity_compare(vec(descriptor(parity_unguarded(de), c.rd)), oracle;
                                 name = "unguarded " * name).ok
        end
    end
    # The non-negative guard reads the active observed cells alone (ADR 0108). A negative value
    # outside the active mask never reaches the output, so it does not refuse the panel.
    @testset "The non-negative guard reads the active cells" begin
        amsk = trues(4, 2)
        amsk[1:2, 2] .= false
        div = [1.0 -5.0; 1.0 -5.0; 1.0 2.0; 1.0 2.0]
        mcap = fill(10.0, 4, 2)
        pnl = asset_panel([NumericPanelInput(; name = "dividends_ttm", vals = div),
                           NumericPanelInput(; name = "market_cap", vals = mcap)];
                          amsk = amsk, emsk = amsk)
        rd = ReturnsResult(; nx = ["a", "b"], X = zeros(4, 2), pnl = pnl)
        @test panel_field(pnl, "dividends_ttm").vals[1, 2] == -5.0
        @test isequal(descriptor(DividendToPrice(), rd),
                      [0.1 NaN; 0.1 NaN; 0.1 0.2; 0.1 0.2])
        div[3, 2] = -5.0
        pnl = asset_panel([NumericPanelInput(; name = "dividends_ttm", vals = div),
                           NumericPanelInput(; name = "market_cap", vals = mcap)];
                          amsk = amsk, emsk = amsk)
        rd = ReturnsResult(; nx = ["a", "b"], X = zeros(4, 2), pnl = pnl)
        @test_throws DomainError descriptor(DividendToPrice(), rd)
    end
    # A quotient that overflows is NaN, as the oracle gives, and not an infinity that refuses the
    # exposure (the defect this ticket fixes).
    @testset "An overflowing quotient is NaN" begin
        c = parity_ratio_case(:clean)
        pnl = c.rd.pnl
        flds = map(pnl.pf) do f
            v = copy(f.vals)
            if f.name == "book_equity"
                v[50, 1] = 1.0e300
            elseif f.name == "market_cap"
                v[50, 1] = 1.0e-10
            end
            return NumericPanelInput(; name = f.name, vals = v)
        end
        rd = ReturnsResult(; nx = c.rd.nx, X = c.rd.X,
                           pnl = asset_panel(flds; amsk = pnl.amsk, emsk = pnl.emsk))
        D = descriptor(BookToPrice(), rd)
        @test isnan(D[50, 1])
        @test isequal(D[:, 2:end], descriptor(BookToPrice(), c.rd)[:, 2:end])
        E = factor_exposure(CompositeExposure(; descriptors = [BookToPrice()],
                                              bw = "market_cap"), rd)
        @test all(isfinite, E[50, 2:end])
    end
end
