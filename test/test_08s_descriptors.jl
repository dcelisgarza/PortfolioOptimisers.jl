#=
Check `src/05_Moments/32_CrossSectionalFactorModel/04_FactorExposures/01_Base_Descriptor.jl`,
`02_PanelFieldDescriptors.jl` and `03_LagDescriptors.jl` against the contract their docstrings
state, and against the oracle's own descriptor tests. Issue #717, map #643.

THREE CONVENTIONS SHAPE THE PROBES.

1. A BLANK NEVER REACHES A CARRIER. `asset_panel` resolves every blank to a fill value and
   records the resolution in an observed-mask column, so a Descriptor must read the mask back
   or it mistakes a fill for data. Every probe that plants a `NaN` in a raw field checks that
   the Descriptor is `NaN` there, not the fill.

2. AN INACTIVE CELL IS `NaN`, whatever its Panel Fields hold. The hand panels below carry a
   value on an inactive cell on purpose, so a Descriptor that forgets the active mask fails.

3. A DATA ERROR REFUSES, AN UNDEFINED RATIO IS `NaN` (#1379, ADR 0108). A field that is
   positive by construction (a price, a market capitalisation, a share count, a total of
   assets) refuses a value at or below zero through the `gt0` guard, and a field that is
   non-negative by construction refuses a negative value through `nonneg`. A denominator that
   valid data can make zero or negative (a book equity, an enterprise value) carries no guard,
   and its ratio is `NaN`. The archetypes with no guard give `NaN` in every such cell, so the
   probes of the `NaN` rule use them. `GrowthRate` refuses a negative field.

The synthetic panel of `test06c_setup.jl` carries every Panel Field the named descriptors
read, so the last testset runs all of them on it and checks each against the field arithmetic
written out by hand from `panel_feature_matrix(rd.pnl)[1]` and `panel_feature_matrix(rd.pnl)[2]`.
=#
include(joinpath(@__DIR__, "test06c_setup.jl"))

# A small hand panel. Every field takes a forward fill, so each earns an observed-mask column
# and a raw `NaN` reads back as `NaN` through the mask rather than as the fill value.
function descriptor_hand_panel(fields::AbstractVector{<:Pair{String, <:AbstractMatrix}};
                               amsk::AbstractMatrix{Bool} = trues(size(fields[1][2])...),
                               emsk::AbstractMatrix{Bool} = amsk)
    inputs = [NumericPanelInput(; name = n, vals = v, alg = ForwardPanelFill(; val = 0.0))
              for (n, v) in fields]
    res = asset_panel(inputs; amsk = amsk, emsk = emsk)
    T, N = size(amsk)
    return ReturnsResult(; nx = ["A" * string(i) for i in 1:N], X = zeros(T, N), pnl = res)
end

# Read one raw field and its observed mask back off the carrier by column name, so the
# expected value of a probe never goes through the function under test.
function descriptor_raw_field(rd::ReturnsResult, name::AbstractString)
    Z = panel_feature_matrix(rd.pnl)[2]
    col = findfirst(==(name), panel_feature_matrix(rd.pnl)[1])
    ocol = findfirst(==(name * "::observed"), panel_feature_matrix(rd.pnl)[1])
    V = Matrix{Float64}(Z[:, :, col])
    if !isnothing(ocol)
        V[iszero.(Z[:, :, ocol])] .= NaN
    end
    return V
end

# The expected ratio, written out by hand: `NaN` where the denominator is not strictly
# positive, where either side is unobserved, or where the asset is inactive.
function descriptor_expected_ratio(num::AbstractMatrix, den::AbstractMatrix,
                                   amsk::AbstractMatrix{Bool})
    return [amsk[k] && den[k] > 0 ? num[k] / den[k] : NaN for k in CartesianIndices(amsk)]
end

const DESCRIPTOR_RATIO_CONSTRUCTORS = (BookToPrice, CashFlowToPrice, SalesToPrice,
                                       EarningsToPrice, ForwardEarningsToPrice,
                                       EbitdaToEnterpriseValue, DividendToPrice,
                                       ForwardDividendToPrice, ShareholderYield,
                                       BookLeverage, MarketLeverage, DebtToAssets,
                                       GrossProfitability, GrossMargin, ReturnOnAssets,
                                       ReturnOnEquity, AssetTurnover, CashFlowToAssets,
                                       SalesToEnterpriseValue, AccrualsCashFlow,
                                       AnalystDispersionToPrice, ShortInterest)
const DESCRIPTOR_LAG_CONSTRUCTORS = (AssetsGrowthRate, SalesGrowthRate, IssuanceGrowthRate,
                                     EarningsChangeToPrice, CapexToAssetsChangeInIntensity)

@testset "Descriptor constructors and their refusals" begin
    @testset "Every named ratio is a PanelFieldRatio that reads its census fields" begin
        for C in DESCRIPTOR_RATIO_CONSTRUCTORS
            de = C()
            @test isa(de, PanelFieldRatio)
            @test isa(de, PortfolioOptimisers.AbstractDescriptorEstimator)
        end
        @test BookToPrice().num == "book_equity"
        @test BookToPrice().den == "market_cap"
        @test isnothing(BookToPrice().nonneg)
        @test isnothing(BookToPrice().pos)
        @test BookToPrice().gt0 == ["market_cap"]
        @test DividendToPrice().gt0 == ["market_cap"]
        @test ForwardEarningsToPrice().gt0 == ["adj_close"]
        @test ShortInterest().gt0 == ["adj_shares_outstanding"]
        @test DebtToAssets().gt0 == ["total_assets"]
        @test AccrualsCashFlow().gt0 == ["total_assets"]
        # A denominator that valid data can make zero or negative carries no guard.
        @test isnothing(ReturnOnEquity().gt0)
        @test isnothing(EbitdaToEnterpriseValue().gt0)
        @test isnothing(SalesToEnterpriseValue().gt0)
        @test isnothing(BookLeverage().gt0)
        @test BookLeverage().nonneg == ["total_debt"]
        @test GrossMargin().nonneg == ["sales_ttm"]
        @test isnothing(GrossMargin().gt0)
        @test isnothing(PanelFieldRatio(; num = "a", den = "b").gt0)
        @test DividendToPrice().nonneg == ["dividends_ttm"]
        @test ForwardDividendToPrice().nonneg == ["dps_ntm"]
        @test AnalystDispersionToPrice().nonneg == ["eps_ntm_std"]
        @test ShortInterest().nonneg == ["short_interest"]
        @test ShareholderYield().num == ["dividends_ttm" => 1, "net_buybacks_ttm" => 1]
        @test ShareholderYield().nonneg == ["dividends_ttm"]
        @test BookLeverage().den == ["total_debt" => 1, "book_equity" => 1]
        @test MarketLeverage().den == ["total_debt" => 1, "market_cap" => 1]
        @test isnothing(MarketLeverage().pos)
        @test MarketLeverage().gt0 == ["market_cap"]
        @test MarketLeverage().nonneg == ["total_debt"]
        @test GrossMargin().num == ["sales_ttm" => 1, "cost_of_revenue_ttm" => -1]
        @test GrossMargin().den == "sales_ttm"
        @test GrossProfitability().num == ["sales_ttm" => 1, "cost_of_revenue_ttm" => -1]
        @test AccrualsCashFlow().num ==
              ["net_income_ttm" => 1, "operating_cash_flow_ttm" => -1]
        @test isa(LogMarketCap(), PanelFieldLog)
        @test LogMarketCap().field == "market_cap"
        @test LogMarketCap().gt0
        @test !PanelFieldLog(; field = "market_cap").gt0
        @test Passthrough(; field = "eps_ntm").field == "eps_ntm"
    end
    @testset "A keyword override renames the field, and the guard follows the rename" begin
        de = BookToPrice(; num = "be", den = "mc")
        @test de.num == "be"
        @test de.den == "mc"
        @test DividendToPrice(; num = "div").nonneg == ["div"]
        @test MarketLeverage(; debt = "d", mcap = "m").den == ["d" => 1, "m" => 1]
        @test MarketLeverage(; debt = "d", mcap = "m").gt0 == ["m"]
        @test BookToPrice(; den = "mc").gt0 == ["mc"]
        @test GrossMargin(; sales = "s", cogs = "c").num == ["s" => 1, "c" => -1]
        @test GrossMargin(; sales = "s", cogs = "c").den == "s"
        @test LogMarketCap(; field = "cap").field == "cap"
    end
    @testset "Every named lag descriptor fixes its fields and a lag of 252" begin
        @test isa(AssetsGrowthRate(), GrowthRate)
        @test AssetsGrowthRate().field == "total_assets"
        @test SalesGrowthRate().field == "sales_ttm"
        @test IssuanceGrowthRate().field == "adj_shares_outstanding"
        @test isa(EarningsChangeToPrice(), ChangeToScale)
        @test EarningsChangeToPrice().field == "net_income_ttm"
        @test EarningsChangeToPrice().scale == "market_cap"
        @test isa(CapexToAssetsChangeInIntensity(), ChangeInIntensity)
        @test CapexToAssetsChangeInIntensity().field == "capex_ttm"
        @test CapexToAssetsChangeInIntensity().scale == "total_assets"
        @test EarningsChangeToPrice().gt0
        @test CapexToAssetsChangeInIntensity().gt0
        @test !ChangeToScale(; field = "a", scale = "b", lag = 1).gt0
        @test !ChangeInIntensity(; field = "a", scale = "b", lag = 1).gt0
        for C in DESCRIPTOR_LAG_CONSTRUCTORS
            @test C().lag == 252
            @test C(; lag = 21).lag == 21
        end
    end
    @testset "The constructors refuse what the docstrings say they refuse" begin
        @test_throws IsEmptyError PanelFieldRatio(; num = "", den = "a")
        @test_throws IsEmptyError PanelFieldRatio(; num = "a", den = "")
        @test_throws IsEmptyError PanelFieldRatio(; num = Pair{String, Int}[], den = "a")
        @test_throws IsEmptyError PanelFieldRatio(; num = ["" => 1], den = "a")
        @test_throws DomainError PanelFieldRatio(; num = ["a" => Inf], den = "b")
        @test_throws DomainError PanelFieldRatio(; num = ["a" => NaN], den = "b")
        # A guard must name a field the ratio reads, or a typo would pass in silence.
        @test_throws ArgumentError PanelFieldRatio(; num = "a", den = "b", nonneg = ["c"])
        @test_throws ArgumentError PanelFieldRatio(; num = "a", den = "b", pos = ["c"])
        @test_throws IsEmptyError PanelFieldRatio(; num = "a", den = "b", nonneg = String[])
        @test_throws IsEmptyError PanelFieldRatio(; num = "a", den = "b", pos = String[])
        @test_throws ArgumentError PanelFieldRatio(; num = "a", den = "b", gt0 = ["c"])
        @test_throws IsEmptyError PanelFieldRatio(; num = "a", den = "b", gt0 = String[])
        # A guard may name a denominator field, and it may name a field of a combination.
        @test PanelFieldRatio(; num = "a", den = ["a" => 1, "b" => 1], pos = ["b"]).pos ==
              ["b"]
        @test_throws IsEmptyError PanelFieldLog(; field = "")
        @test_throws IsEmptyError Passthrough(; field = "")
        @test_throws IsEmptyError GrowthRate(; field = "", lag = 1)
        @test_throws DomainError GrowthRate(; field = "a", lag = 0)
        @test_throws DomainError GrowthRate(; field = "a", lag = -3)
        @test_throws IsEmptyError ChangeToScale(; field = "a", scale = "", lag = 1)
        @test_throws DomainError ChangeToScale(; field = "a", scale = "b", lag = 0)
        @test_throws IsEmptyError ChangeInIntensity(; field = "", scale = "b", lag = 1)
        @test_throws DomainError ChangeInIntensity(; field = "a", scale = "b", lag = 0)
    end
    @testset "The term helpers" begin
        @test PortfolioOptimisers.panel_term_names("a") == ["a"]
        @test PortfolioOptimisers.panel_term_names(["a" => 1, "b" => -1]) == ["a", "b"]
        @test PortfolioOptimisers.positive_divide(1.0, 4.0) == 0.25
        @test isnan(PortfolioOptimisers.positive_divide(1.0, 0.0))
        @test isnan(PortfolioOptimisers.positive_divide(1.0, -2.0))
        @test isnan(PortfolioOptimisers.positive_divide(1.0, NaN))
        @test isnan(PortfolioOptimisers.positive_divide(NaN, 2.0))
        @test PortfolioOptimisers.positive_divide(1, 4) == 0.25
        # A quotient of two finite values that overflows is NaN, not an infinity (#1379).
        @test isnan(PortfolioOptimisers.positive_divide(1.0e300, 1.0e-10))
        @test isnan(PortfolioOptimisers.positive_divide(-1.0e300, 1.0e-10))
        @test PortfolioOptimisers.positive_divide(1.0e300, 1.0e10) == 1.0e290
    end
end

@testset "The shared read: a fill and an inactive cell read back as NaN (#1411)" begin
    amsk = [true true true; true false true; true true true]
    rd = descriptor_hand_panel(["mcap" => [1.0 2.0 3.0; NaN 5.0 6.0; 7.0 8.0 9.0],
                                "debt" => [0.5 1.0 1.5; 2.0 2.5 3.0; 3.5 4.0 4.5]];
                               amsk = amsk)
    V = PortfolioOptimisers.descriptor_field_values(rd, "mcap")
    @test isa(V, Matrix{Float64})
    @test isnan(V[2, 1])
    @test V[1, :] == [1.0, 2.0, 3.0]
    # Cell (2, 2) is inactive but observed, and it reads as NaN too: a lag or a recursion
    # that reaches it must not read the finite value the panel keeps there (#1411).
    @test isnan(V[2, 2])
    W = PortfolioOptimisers.descriptor_field_values(rd, ["mcap" => 1, "debt" => -2])
    @test W[1, :] == [0.0, 0.0, 0.0]
    @test W[3, 3] == 9.0 - 9.0
    @test isnan(W[2, 1])
    @test_throws IsEmptyError PortfolioOptimisers.descriptor_field_values(rd,
                                                                          Pair{String, Int}[])
    @test_throws KeyError PortfolioOptimisers.descriptor_field_values(rd, "mcp")
    # A carrier with no panel cannot be read, and a categorical field is not one number.
    bare = ReturnsResult(; nx = ["A1", "A2", "A3"], X = zeros(3, 3))
    @test_throws IsNothingError PortfolioOptimisers.descriptor_field_values(bare, "mcap")
    @test_throws IsNothingError descriptor(BookToPrice(), bare)
    res = asset_panel([NumericPanelInput(; name = "mcap", vals = ones(2, 2)),
                       CategoricalPanelInput(; name = "sector", vals = ["a" "b"; "a" "b"],
                                             levels = ["a", "b"])]; amsk = trues(2, 2),
                      emsk = trues(2, 2))
    rdc = ReturnsResult(; nx = ["A1", "A2"], X = zeros(2, 2), pnl = res)
    @test_throws ArgumentError PortfolioOptimisers.descriptor_field_values(rdc, "sector")
    @test_throws ArgumentError descriptor(Passthrough(; field = "sector"), rdc)
    # A field that cannot blank carries no observed-mask column, and reads back whole.
    @test PortfolioOptimisers.descriptor_field_values(rdc, "mcap") == ones(2, 2)
    # The read lands in the type a division of the field lands in, which is the one that
    # carries a NaN: an exact integer field reads in Float64, a Float32 field stays Float32,
    # and every archetype that reads it can then write NaN on an inactive or blank cell.
    ami = [true true; true false]
    pint = asset_panel([NumericPanelInput(; name = "shares", vals = [10 20; 30 40]),
                        NumericPanelInput(; name = "gap",
                                          vals = Union{Missing, Int}[10 missing; 30 40],
                                          alg = ConstantPanelFill(; val = 0)),
                        NumericPanelInput(; name = "f32", vals = Float32[1 2; 3 4])];
                       amsk = ami, emsk = ami)
    rdi = ReturnsResult(; nx = ["A1", "A2"], X = zeros(2, 2), pnl = pint)
    Vi = PortfolioOptimisers.descriptor_field_values(rdi, "shares")
    @test eltype(Vi) === Float64
    @test isequal(Vi, [10.0 20.0; 30.0 NaN])
    Vg = PortfolioOptimisers.descriptor_field_values(rdi, "gap")
    @test isnan(Vg[1, 2])
    @test eltype(PortfolioOptimisers.descriptor_field_values(rdi, "f32")) === Float32
    Dp = descriptor(Passthrough(; field = "shares"), rdi)
    @test isnan(Dp[2, 2])
    @test Dp[1, :] == [10.0, 20.0]
    Dl = descriptor(PanelFieldLog(; field = "shares"), rdi)
    @test Dl[1, 1] ≈ log(10)
    @test isnan(Dl[2, 2])
    @test eltype(descriptor(Passthrough(; field = "f32"), rdi)) === Float32

    D = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
    @test isnothing(PortfolioOptimisers.descriptor_active_fill!(D, rd.pnl))
    @test isnan(D[2, 2])
    @test count(isnan, D) == 1
    @test_throws DimensionMismatch PortfolioOptimisers.descriptor_active_fill!(zeros(2, 3),
                                                                               rd.pnl)
end

@testset "Point-in-time ratios on a hand panel" begin
    # Column 3 carries a zero and a negative market capitalisation, row 2 an unobserved book
    # equity, and cell (3, 2) is inactive with values in every field. A named constructor
    # refuses the market capitalisation of column 3, so the probes of the NaN rule read the
    # archetype with no guard, and `mp` is a market capitalisation that is positive
    # wherever it is active.
    amsk = [true true true; true true true; true false true]
    be = [2.0 3.0 4.0; NaN 6.0 7.0; 8.0 9.0 10.0]
    mcap = [4.0 6.0 0.0; 10.0 12.0 -1.0; 16.0 18.0 20.0]
    debt = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
    div = [0.1 0.2 0.3; 0.4 0.5 0.6; 0.7 0.8 0.9]
    bb = [0.05 -0.1 0.15; -0.2 0.25 -0.3; 0.35 0.4 -0.45]
    rd = descriptor_hand_panel(["book_equity" => be, "market_cap" => mcap,
                                "total_debt" => debt, "dividends_ttm" => div,
                                "net_buybacks_ttm" => bb]; amsk = amsk)
    mp = [4.0 6.0 2.0; 10.0 12.0 1.0; 16.0 -18.0 20.0]
    rdp = descriptor_hand_panel(["book_equity" => be, "market_cap" => mp,
                                 "total_debt" => debt, "dividends_ttm" => div,
                                 "net_buybacks_ttm" => bb]; amsk = amsk)
    btp = PanelFieldRatio(; num = "book_equity", den = "market_cap")
    @testset "A ratio equals the field arithmetic, and NaN where the docstring says so" begin
        D = descriptor(btp, rd)
        @test isa(D, Matrix{Float64})
        @test size(D) == (3, 3)
        @test D[1, 1] == 0.5
        @test D[1, 2] == 0.5
        @test isnan(D[1, 3])                         # zero denominator
        @test isnan(D[2, 3])                         # negative denominator
        @test isnan(D[2, 1])                         # unobserved numerator
        @test isnan(D[3, 2])                         # inactive cell
        @test D[3, 1] == 0.5
        @test D[3, 3] == 0.5
        @test isequal(D, descriptor_expected_ratio(be, mcap, amsk))
        # The named constructor refuses the zero and the negative market capitalisation.
        @test_throws DomainError descriptor(BookToPrice(), rd)
        # On `mp` the guard passes, the negative value of the inactive cell is not read, and
        # the named constructor equals the archetype.
        @test isequal(descriptor(BookToPrice(), rdp), descriptor(btp, rdp))
    end
    @testset "The gt0 guard refuses a non-positive field, and only an observed active one" begin
        for C in (BookToPrice, DividendToPrice, ShareholderYield, MarketLeverage)
            @test_throws DomainError descriptor(C(), rd)
        end
        @test_throws DomainError descriptor(LogMarketCap(), rd)
        err = try
            descriptor(BookToPrice(), rd)
        catch e
            e
        end
        @test occursin("strictly positive", err.msg)
        @test occursin("observation 1 for asset 3", err.msg)
        # An unobserved cell reads back as NaN and is not checked.
        m2 = copy(mp)
        m2[1, 1] = NaN
        rdu = descriptor_hand_panel(["book_equity" => be, "market_cap" => m2]; amsk = amsk)
        @test isnan(descriptor(BookToPrice(), rdu)[1, 1])
    end
    @testset "A combination numerator sums its terms with their coefficients" begin
        D = descriptor(PanelFieldRatio(;
                                       num = ["dividends_ttm" => 1,
                                              "net_buybacks_ttm" => 1], den = "market_cap",
                                       nonneg = ["dividends_ttm"]), rd)
        @test D[1, 1] == (0.1 + 0.05) / 4.0
        @test D[2, 2] == (0.5 + 0.25) / 12.0
        @test isnan(D[1, 3])
        @test isnan(D[3, 2])
    end
    @testset "A combination denominator, and the pos guard beyond it" begin
        # Book leverage: debt / (debt + book equity).
        D = descriptor(BookLeverage(), rd)
        @test D[1, 1] == 1.0 / 3.0
        @test isnan(D[2, 1])
        @test isnan(D[3, 2])
        # Market leverage with the pos guard: the total capital at (1, 3) is 3 > 0, but the
        # market capitalisation is 0, so the pos guard writes NaN where the denominator alone
        # would not. At (2, 3) the total is 5 > 0 and the capitalisation is -1.
        # `MarketLeverage()` refuses both cells through its gt0 guard instead.
        D = descriptor(PanelFieldRatio(; num = "total_debt",
                                       den = ["total_debt" => 1, "market_cap" => 1],
                                       pos = ["market_cap"]), rd)
        @test D[1, 1] == 1.0 / 5.0
        @test isnan(D[1, 3])
        @test isnan(D[2, 3])
        @test isnan(D[3, 2])
        # The same fields with the guard off return the ratio.
        de = PanelFieldRatio(; num = "total_debt",
                             den = ["total_debt" => 1, "market_cap" => 1])
        @test descriptor(de, rd)[1, 3] == 1.0
        @test descriptor(de, rd)[2, 3] == 6.0 / 5.0
    end
    @testset "The nonneg guard refuses a negative dividend, and only an observed active one" begin
        @test isa(descriptor(DividendToPrice(), rdp), Matrix{Float64})
        neg = copy(div)
        neg[2, 2] = -0.1
        rdn = descriptor_hand_panel(["dividends_ttm" => neg, "market_cap" => mp];
                                    amsk = amsk)
        @test_throws DomainError descriptor(DividendToPrice(), rdn)
        # A negative value on an inactive cell is not a data error the estimator reads.
        neg[2, 2] = 0.1
        neg[3, 2] = -0.1
        rdi = descriptor_hand_panel(["dividends_ttm" => neg, "market_cap" => mp];
                                    amsk = amsk)
        @test isa(descriptor(DividendToPrice(), rdi), Matrix{Float64})
        # Neither is an unobserved one.
        neg[3, 2] = 0.1
        neg[1, 1] = NaN
        rdu = descriptor_hand_panel(["dividends_ttm" => neg, "market_cap" => mp];
                                    amsk = amsk)
        @test isnan(descriptor(DividendToPrice(), rdu)[1, 1])
        # The buybacks may be negative: the guard names the dividends alone.
        @test isa(descriptor(ShareholderYield(), rdp), Matrix{Float64})
    end
    @testset "The logarithm and the passthrough" begin
        D = descriptor(PanelFieldLog(; field = "market_cap"), rd)
        @test D[1, 1] == log(4.0)
        @test isnan(D[1, 3])
        @test isnan(D[2, 3])
        @test isnan(D[3, 2])
        @test D[3, 3] == log(20.0)
        P = descriptor(Passthrough(; field = "market_cap"), rd)
        @test P[1, 3] == 0.0
        @test P[2, 3] == -1.0
        @test isnan(P[3, 2])
        P = descriptor(Passthrough(; field = "book_equity"), rd)
        @test isnan(P[2, 1])
        @test P[1, 1] == 2.0
    end
    @testset "The estimator holds no data: two carriers, one estimator" begin
        de = btp
        D1 = descriptor(de, rd)
        rd2 = descriptor_hand_panel(["book_equity" => 2 .* be, "market_cap" => mcap];
                                    amsk = amsk)
        D2 = descriptor(de, rd2)
        @test isequal(D2, 2 .* D1)
        @test isequal(descriptor(de, rd), D1)
    end
end

@testset "Lag descriptors on a hand panel" begin
    T, N = 8, 2
    sales = [100.0 50.0; 110.0 55.0; 121.0 60.5; 133.1 66.55; 146.41 73.205;
             161.051 80.5255;
             177.1561 88.57805; 194.87171 97.435855]
    mcap = fill(1000.0, T, N)
    @testset "GrowthRate equals x[lag+1:end] ./ x[1:end-lag] .- 1 exactly" begin
        for lag in (1, 3, 5)
            rd = descriptor_hand_panel(["sales_ttm" => sales, "market_cap" => mcap])
            D = descriptor(GrowthRate(; field = "sales_ttm", lag = lag), rd)
            @test size(D) == (T, N)
            @test all(isnan, D[1:lag, :])
            @test !any(isnan, D[(lag + 1):end, :])
            @test D[(lag + 1):end, :] ==
                  sales[(lag + 1):end, :] ./ sales[1:(end - lag), :] .- 1
        end
    end
    @testset "A NaN propagates to the current and to the lagged read, a zero base is NaN" begin
        s = copy(sales)
        s[5, 1] = NaN
        rd = descriptor_hand_panel(["sales_ttm" => s, "market_cap" => mcap])
        D = descriptor(GrowthRate(; field = "sales_ttm", lag = 3), rd)
        @test isnan(D[5, 1])
        @test isnan(D[8, 1])
        @test !isnan(D[6, 1])
        @test !isnan(D[5, 2])
        s = copy(sales)
        s[2, 1] = 0.0
        rd = descriptor_hand_panel(["sales_ttm" => s, "market_cap" => mcap])
        D = descriptor(GrowthRate(; field = "sales_ttm", lag = 3), rd)
        @test isnan(D[5, 1])
        @test D[4, 1] == sales[4, 1] / sales[1, 1] - 1
        # A zero current value over a positive base is a growth of minus one.
        @test D[2 + 3, 1] |> isnan
        s = copy(sales)
        s[5, 2] = 0.0
        rd = descriptor_hand_panel(["sales_ttm" => s, "market_cap" => mcap])
        @test descriptor(GrowthRate(; field = "sales_ttm", lag = 3), rd)[5, 2] == -1.0
    end
    @testset "GrowthRate refuses a negative field" begin
        s = copy(sales)
        s[5, 1] = -100.0
        rd = descriptor_hand_panel(["sales_ttm" => s, "market_cap" => mcap])
        @test_throws DomainError descriptor(GrowthRate(; field = "sales_ttm", lag = 3), rd)
        # The same value on an inactive cell is not read.
        amsk = trues(T, N)
        amsk[5, 1] = false
        rd = descriptor_hand_panel(["sales_ttm" => s, "market_cap" => mcap]; amsk = amsk)
        D = descriptor(GrowthRate(; field = "sales_ttm", lag = 3), rd)
        @test isnan(D[5, 1])
        # The negative base is not read as a refusal, and a non-positive base is NaN.
        @test isnan(D[8, 1])
        @test D[7, 1] == s[7, 1] / s[4, 1] - 1
    end
    @testset "A lag at or past the sample is all NaN, not an error" begin
        rd = descriptor_hand_panel(["sales_ttm" => sales, "market_cap" => mcap])
        @test all(isnan, descriptor(GrowthRate(; field = "sales_ttm", lag = T), rd))
        @test all(isnan, descriptor(GrowthRate(; field = "sales_ttm", lag = T + 5), rd))
        @test all(isnan,
                  descriptor(ChangeToScale(; field = "sales_ttm", scale = "market_cap",
                                           lag = T), rd))
        @test all(isnan,
                  descriptor(ChangeInIntensity(; field = "sales_ttm", scale = "market_cap",
                                               lag = T), rd))
    end
    @testset "ChangeToScale divides the change by the current scale" begin
        m = copy(mcap)
        m[6, 1] = 0.0
        m[7, 1] = -5.0
        ni = sales .- 120.0                     # a field that turns negative
        rd = descriptor_hand_panel(["net_income_ttm" => ni, "market_cap" => m])
        # The named constructor refuses the zero and the negative capitalisation; the
        # archetype with no guard gives NaN there.
        @test_throws DomainError descriptor(EarningsChangeToPrice(; lag = 2), rd)
        D = descriptor(ChangeToScale(; field = "net_income_ttm", scale = "market_cap",
                                     lag = 2), rd)
        @test all(isnan, D[1:2, :])
        @test D[3, 1] == (ni[3, 1] - ni[1, 1]) / m[3, 1]
        @test D[5, 2] == (ni[5, 2] - ni[3, 2]) / m[5, 2]
        @test isnan(D[6, 1])
        @test isnan(D[7, 1])
        @test D[8, 1] == (ni[8, 1] - ni[6, 1]) / m[8, 1]
        # A negative field is what this archetype exists for, so it raises nothing.
        @test ni[3, 2] < 0
        @test !isnan(D[3, 2])
    end
    @testset "ChangeInIntensity differences the ratio at both ends" begin
        capex = 0.02 .* sales
        ta = 2 .* sales
        ta[4, 2] = 0.0
        rd = descriptor_hand_panel(["capex_ttm" => capex, "total_assets" => ta])
        @test_throws DomainError descriptor(CapexToAssetsChangeInIntensity(; lag = 2), rd)
        D = descriptor(ChangeInIntensity(; field = "capex_ttm", scale = "total_assets",
                                         lag = 2), rd)
        @test all(isnan, D[1:2, :])
        @test D[3, 1] ≈ capex[3, 1] / ta[3, 1] - capex[1, 1] / ta[1, 1] atol = 1e-15
        @test isnan(D[4, 2])                    # the current scale is zero
        @test isnan(D[6, 2])                    # the lagged scale is zero
        @test !isnan(D[5, 2])
        # The capex grew with the assets it serves, so the intensity did not change.
        @test all(x -> isnan(x) || abs(x) < 1e-15, D)
    end
    @testset "An inactive cell is NaN in every lag descriptor" begin
        amsk = trues(T, N)
        amsk[6, 2] = false
        rd = descriptor_hand_panel(["sales_ttm" => sales, "market_cap" => mcap];
                                   amsk = amsk)
        for de in (GrowthRate(; field = "sales_ttm", lag = 1),
                   ChangeToScale(; field = "sales_ttm", scale = "market_cap", lag = 1),
                   ChangeInIntensity(; field = "sales_ttm", scale = "market_cap", lag = 1))
            D = descriptor(de, rd)
            @test isnan(D[6, 2])
            # Observation 7 lags into the inactive cell, which reads as missing, so it is
            # NaN too: the panel keeps a finite value there, and no lag may read it (#1411).
            @test isnan(D[7, 2])
            @test count(isnan, D) == N + 2
        end
    end
end

@testset "Every named descriptor runs on the synthetic panel" begin
    rng = StableRNG(717)
    rd = synthetic_asset_panel(; n_assets = 12, n_observations = 300, n_industries = 3,
                               rng = rng).rd
    amsk = rd.pnl.amsk
    T, N = size(amsk)
    field(name) = descriptor_raw_field(rd, name)
    @testset "Shape, the inactive fill, and finite-or-NaN everywhere" begin
        for C in (DESCRIPTOR_RATIO_CONSTRUCTORS..., LogMarketCap,
                  DESCRIPTOR_LAG_CONSTRUCTORS...)
            D = descriptor(C(), rd)
            @test size(D) == (T, N)
            @test all(isnan, D[.!amsk])
            @test all(x -> isnan(x) || isfinite(x), D)
        end
        P = descriptor(Passthrough(; field = "market_cap"), rd)
        @test all(isnan, P[.!amsk])
        @test !any(isnan, P[amsk])
    end
    @testset "Every ratio equals its census formula on the observed active cells" begin
        f = Dict(n => field(n)
                 for n in
                     ("book_equity", "market_cap", "operating_cash_flow_ttm", "sales_ttm",
                      "net_income_ttm", "eps_ntm", "adj_close", "ebitda_ttm",
                      "enterprise_value", "dividends_ttm", "dps_ntm", "net_buybacks_ttm",
                      "total_debt", "total_assets", "cost_of_revenue_ttm", "eps_ntm_std",
                      "short_interest", "adj_shares_outstanding"))
        checks = [BookToPrice() => (f["book_equity"], f["market_cap"]),
                  CashFlowToPrice() => (f["operating_cash_flow_ttm"], f["market_cap"]),
                  SalesToPrice() => (f["sales_ttm"], f["market_cap"]),
                  EarningsToPrice() => (f["net_income_ttm"], f["market_cap"]),
                  ForwardEarningsToPrice() => (f["eps_ntm"], f["adj_close"]),
                  EbitdaToEnterpriseValue() => (f["ebitda_ttm"], f["enterprise_value"]),
                  DividendToPrice() => (f["dividends_ttm"], f["market_cap"]),
                  ForwardDividendToPrice() => (f["dps_ntm"], f["adj_close"]),
                  ShareholderYield() =>
                      (f["dividends_ttm"] .+ f["net_buybacks_ttm"], f["market_cap"]),
                  BookLeverage() => (f["total_debt"], f["total_debt"] .+ f["book_equity"]),
                  DebtToAssets() => (f["total_debt"], f["total_assets"]),
                  GrossProfitability() =>
                      (f["sales_ttm"] .- f["cost_of_revenue_ttm"], f["total_assets"]),
                  GrossMargin() =>
                      (f["sales_ttm"] .- f["cost_of_revenue_ttm"], f["sales_ttm"]),
                  ReturnOnAssets() => (f["net_income_ttm"], f["total_assets"]),
                  ReturnOnEquity() => (f["net_income_ttm"], f["book_equity"]),
                  AssetTurnover() => (f["sales_ttm"], f["total_assets"]),
                  CashFlowToAssets() => (f["operating_cash_flow_ttm"], f["total_assets"]),
                  SalesToEnterpriseValue() => (f["sales_ttm"], f["enterprise_value"]),
                  AccrualsCashFlow() =>
                      (f["net_income_ttm"] .- f["operating_cash_flow_ttm"],
                       f["total_assets"]),
                  AnalystDispersionToPrice() => (f["eps_ntm_std"], f["adj_close"]),
                  ShortInterest() => (f["short_interest"], f["adj_shares_outstanding"])]
        for (de, (num, den)) in checks
            @test isequal(descriptor(de, rd), descriptor_expected_ratio(num, den, amsk))
        end
        # Market leverage carries the extra guard, and the synthetic capitalisation is
        # positive wherever it is observed, so the guard changes nothing here.
        ml = descriptor_expected_ratio(f["total_debt"], f["total_debt"] .+ f["market_cap"],
                                       amsk)
        @test isequal(descriptor(MarketLeverage(), rd), ml)
        lm = [amsk[k] && f["market_cap"][k] > 0 ? log(f["market_cap"][k]) : NaN
              for k in CartesianIndices(amsk)]
        @test isequal(descriptor(LogMarketCap(), rd), lm)
        # The panel blanks about one cell in a hundred, so a fill is read back somewhere.
        @test count(isnan, descriptor(BookToPrice(), rd)) > count(!, amsk)
    end
    @testset "Every growth descriptor equals its census formula" begin
        lag = 21
        for (C, name) in
            ((AssetsGrowthRate, "total_assets"), (SalesGrowthRate, "sales_ttm"),
             (IssuanceGrowthRate, "adj_shares_outstanding"))
            x = field(name)
            D = descriptor(C(; lag = lag), rd)
            E = fill(NaN, T, N)
            for i in 1:N, t in (lag + 1):T
                if amsk[t, i] && x[t - lag, i] > 0
                    E[t, i] = x[t, i] / x[t - lag, i] - 1
                end
            end
            @test isequal(D, E)
            @test all(isnan, D[1:lag, :])
            @test count(!isnan, D) > 0
        end
        ni, mc = field("net_income_ttm"), field("market_cap")
        D = descriptor(EarningsChangeToPrice(; lag = lag), rd)
        E = fill(NaN, T, N)
        for i in 1:N, t in (lag + 1):T
            if amsk[t, i] && mc[t, i] > 0
                E[t, i] = (ni[t, i] - ni[t - lag, i]) / mc[t, i]
            end
        end
        @test isequal(D, E)
        cx, ta = field("capex_ttm"), field("total_assets")
        D = descriptor(CapexToAssetsChangeInIntensity(; lag = lag), rd)
        E = fill(NaN, T, N)
        for i in 1:N, t in (lag + 1):T
            if amsk[t, i] && ta[t, i] > 0 && ta[t - lag, i] > 0
                E[t, i] = cx[t, i] / ta[t, i] - cx[t - lag, i] / ta[t - lag, i]
            end
        end
        @test isequal(D, E)
    end
    @testset "An asset view of the carrier gives the same descriptor as a slice of the whole" begin
        v = PortfolioOptimisers.port_opt_view(rd, [2, 5, 9])
        for de in
            (BookToPrice(), MarketLeverage(), LogMarketCap(), SalesGrowthRate(; lag = 21))
            @test isequal(descriptor(de, v), descriptor(de, rd)[:, [2, 5, 9]])
        end
    end
end
@testset "A widened numeric bound admits a number that is not Real (#909)" begin
    # These verbs neither order their values nor test them for finiteness, so their bounds
    # state `<:Number` rather than `<:Real`. The probe is a Complex argument: it carries no
    # order, so it reaches a widened method and is refused by a narrow one.
    pnl = AssetPanel(; pf = [NumericPanelField(; name = "a", vals = [1.0 2.0; 3.0 4.0])],
                     amsk = [true false; true true], emsk = [true false; true true])

    D = Complex{Float64}[1 2; 3 4]
    PortfolioOptimisers.descriptor_active_fill!(D, pnl)
    @test isnan(real(D[1, 2]))
    @test D[1, 1] == 1 && D[2, 1] == 3 && D[2, 2] == 4

    L = Complex{Float64}[1 2; 3 4]
    PortfolioOptimisers.exposure_active_fill!(L, pnl)
    @test isnan(real(L[1, 2]))

    L3 = fill(one(Complex{Float64}), 2, 2, 2)
    PortfolioOptimisers.exposure_active_fill!(L3, pnl)
    @test isnan(real(L3[1, 2, 1])) && L3[1, 1, 1] == 1

    B = fill(one(Complex{Float64}), 2, 2, 2)
    PortfolioOptimisers.one_hot_observed_fill!(B, [true false; true true])
    @test isnan(real(B[1, 2, 1])) && B[1, 1, 1] == 1
    PortfolioOptimisers.one_hot_observed_fill!(B, nothing)
    @test B[1, 1, 1] == 1

    X = Complex{Float64}[1 2; 3 4]
    @test isnan(real(PortfolioOptimisers.ew_active_returns(X, pnl)[1, 2]))

    Ba = Complex{Float64}[1 2; 3 4]
    @test PortfolioOptimisers.ew_beta_expand(Ba, 4, 2)[2, :] == Complex{Float64}[1, 2]

    v = Complex{Float64}[1, 2, 6]
    w = Complex{Float64}[1, 1, 3]
    msk = [true, false, true]
    @test PortfolioOptimisers.ew_masked_mean(v, msk) == 3.5
    @test PortfolioOptimisers.ew_masked_weighted_mean(v, w, msk) == 4.75

    mu = Complex{Float64}[1, 2]
    cv = Complex{Float64}[3, 4]
    PortfolioOptimisers.ew_beta_reset!([true false; true true], mu, cv, [1, 1],
                                       [true, true], 1)
    @test iszero(mu[2]) && iszero(cv[2])
    PortfolioOptimisers.ew_beta_reset!(nothing, mu, cv, [1, 1], [true, true], 1)

    @test PortfolioOptimisers.ew_volatility_input(FullMoment(), X, 0) === X

    Ms = zeros(Complex{Float64}, 2, 2, 2)
    PortfolioOptimisers.cross_sectional_exposure_write!(Ms, X, 1, 1, "f")
    @test Ms[:, :, 1] == X
    PortfolioOptimisers.cross_sectional_exposure_write!(Ms, reshape(X, 2, 2, 1), 2, 1, "f")
    @test Ms[:, :, 2] == X

    @test PortfolioOptimisers.target_forecast_multiplier(true, 2 + 0im) == 2 + 0im
    @test PortfolioOptimisers.target_forecast_multiplier(false, 2 + 0im) == 1
end

# A number type whose conversion from `NaN` raises an error other than `InexactError`, so the
# probe below sees that `nan_fill_value` refuses only the error it knows and passes the rest on.
struct NaNProbe726 <: Real
    x::Float64
end
Base.convert(::Type{NaNProbe726}, x::Float64) = throw(DomainError(x, "NaNProbe726"))

@testset "A fill refuses an element type that cannot hold NaN (#726)" begin
    # A fill changes its caller's array, so it cannot widen the element type. Each fill
    # converts `NaN` once, before it writes, and refuses a type that holds no `NaN` with an
    # `ArgumentError` rather than an `InexactError` at the first inactive cell.
    pnl = AssetPanel(; pf = [NumericPanelField(; name = "a", vals = [1.0 2.0; 3.0 4.0])],
                     amsk = [true false; true true], emsk = [true false; true true])
    omsk = [true false; true true]

    for T in (Int, Bool, Rational{Int}, Complex{Int})
        D = ones(T, 2, 2)
        @test_throws ArgumentError PortfolioOptimisers.nan_fill_value(D)
        @test_throws ArgumentError PortfolioOptimisers.descriptor_active_fill!(D, pnl)
        @test all(isone, D)
        @test_throws ArgumentError PortfolioOptimisers.exposure_active_fill!(ones(T, 2, 2),
                                                                             pnl)
        @test_throws ArgumentError PortfolioOptimisers.exposure_active_fill!(ones(T, 2, 2,
                                                                                  2), pnl)
        @test_throws ArgumentError PortfolioOptimisers.one_hot_observed_fill!(ones(T, 2, 2,
                                                                                   2), omsk)
        @test isnothing(PortfolioOptimisers.one_hot_observed_fill!(ones(T, 2, 2, 2),
                                                                   nothing))
    end
    for T in (Int, Rational{Int})
        @test_throws ArgumentError PortfolioOptimisers.one_hot_level_fill!(zeros(T, 2, 2,
                                                                                 2))
    end

    # The type of the array is kept: a Float32 array takes NaN32, a BigFloat array a BigFloat.
    for T in (Float32, BigFloat)
        D = ones(T, 2, 2)
        PortfolioOptimisers.descriptor_active_fill!(D, pnl)
        @test eltype(D) === T
        @test isnan(D[1, 2]) && count(isnan, D) == 1
    end

    @test_throws DomainError PortfolioOptimisers.nan_fill_value(fill(NaNProbe726(1.0), 1))
end

@testset "A combination of Panel Fields does not depend on the order of its terms (#726)" begin
    # The first term used to fix the element type, so a Float32 first term truncated a
    # Float64 second term. The sum now takes the type that every term promotes to.
    pnl = asset_panel([NumericPanelInput(; name = "f32", vals = Float32[1 2; 3 4]),
                       NumericPanelInput(; name = "f64", vals = [1e-9 0.0; 0.0 2.5])];
                      amsk = trues(2, 2), emsk = trues(2, 2))
    rd = ReturnsResult(; nx = ["A1", "A2"], X = zeros(2, 2), pnl = pnl)
    V1 = PortfolioOptimisers.descriptor_field_values(rd, ["f32" => 1, "f64" => 1])
    V2 = PortfolioOptimisers.descriptor_field_values(rd, ["f64" => 1, "f32" => 1])
    @test eltype(V1) === Float64 && eltype(V2) === Float64
    @test V1 == V2
    @test V1[1, 1] == 1.0 + 1e-9
    # One Float32 term with an integer coefficient stays Float32.
    V3 = PortfolioOptimisers.descriptor_field_values(rd, ["f32" => 2])
    @test eltype(V3) === Float32
    @test V3 == Float32[2 4; 6 8]
end

@testset "The log-return check names the first return at or below -1 (#726)" begin
    @test isnothing(PortfolioOptimisers.assert_log_returns([0.1 NaN; -0.5 Inf]))
    err = try
        PortfolioOptimisers.assert_log_returns([0.1 0.2; -1.0 -2.0])
    catch e
        e
    end
    @test err isa DomainError
    @test err.val == -1.0
    @test occursin("observation 2 for asset 1", err.msg)
end

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
        # An exponentially weighted forecast reads the look-back of its Descriptors too (#1574).
        @test PO.lookback(ExpWeightedReturnForecast(; scores = ds)) === 273
        # A Target Return Forecast reads the look-back of its Descriptors (#1581). Its fit is
        # a fold over a state that the carry keeps, not a read of the panel rows.
        @test PO.lookback(TargetReturnForecast(; scores = ds)) === 273
    end
end

@testset "The look-back of a Cross-Sectional Factor Prior" begin
    factors = ["market" => ConstantExposure(),
               "momentum" => CompositeExposure(; descriptors = [RollingMomentum()]),
               "value" => CompositeExposure(; descriptors = [BookToPrice()])]
    @testset "The factors plus the lag" begin
        @test PO.lookback(CrossSectionalFactorPrior(; lambda = 1, factors = factors)) ===
              274
        @test PO.lookback(CrossSectionalFactorPrior(; lambda = 1, factors = factors,
                                                    lag = 5)) === 278
        @test PO.lookback(CrossSectionalFactorPrior(; lambda = 1,
                                                    factors = ["market" =>
                                                                   ConstantExposure()])) ===
              2
    end
    @testset "An unbounded factor makes the prior unbounded" begin
        ew = vcat(factors, ["beta" => CompositeExposure(; descriptors = [EWMarketBeta()])])
        @test isnothing(PO.lookback(CrossSectionalFactorPrior(; lambda = 1, factors = ew)))
    end
    @testset "The Return Forecast counts when it reads further back" begin
        long = DescriptorScores(; descriptors = [RollingLogReturn(; window = 400)])
        short = DescriptorScores(; descriptors = [Reversal()])
        @test PO.lookback(CrossSectionalFactorPrior(; lambda = 1, factors = factors,
                                                    rfe = FixedWeightedReturnForecast(;
                                                                                      scores = long,
                                                                                      scale = 1.0))) ===
              400
        @test PO.lookback(CrossSectionalFactorPrior(; lambda = 1, factors = factors,
                                                    rfe = FixedWeightedReturnForecast(;
                                                                                      scores = short,
                                                                                      scale = 1.0))) ===
              274
        @test PO.lookback(CrossSectionalFactorPrior(; lambda = 1, factors = factors,
                                                    rfe = CustomValueReturnForecast(;
                                                                                    mu = [0.1]))) ===
              274
        @test PO.lookback(CrossSectionalFactorPrior(; lambda = 1, factors = factors,
                                                    rfe = ExpWeightedReturnForecast(;
                                                                                    scores = short))) ===
              274
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

#=
Check `src/05_Moments/32_CrossSectionalFactorModel/04_FactorExposures/07_RollingDescriptors.jl` against the contract its
docstrings state, and against the oracle's own rolling descriptors.
Issue #720, map #643.

FIVE CONVENTIONS SHAPE THE PROBES.

1. RETURNS ARE NOT A PANEL FIELD. A rolling Descriptor reads `rd.X`, and the active mask of
   the Asset Panel beside it. It reads no column of the feature matrix.

2. A WINDOW THAT IS NOT WHOLLY ACTIVE IS `NaN`. One rule covers three cases: the warm-up at
   the start of the sample, an asset that lists late, and a gap inside a listing. The hand
   panels below carry a return on an inactive cell on purpose, so a Descriptor that forgets
   the active mask fails.

3. A MISSING RETURN IS NOT A LOSS. Inside an active window, `RollingLogReturn` counts a
   missing return as a zero contribution, and `RollingMax` ignores it. A window whose
   returns are all missing is `NaN` for `RollingMax` and zero for `RollingLogReturn`.

4. A RETURN AT OR BELOW `-1` IS A REFUSAL, NOT A `NaN`. The logarithm is undefined there, so
   `RollingLogReturn` raises on the whole matrix. `RollingMax` takes no logarithm and needs
   no such rule.

5. AN INFINITE RETURN IS A REFUSAL, FOR BOTH ESTIMATORS. The window sum is one difference of
   two cumulative sums, so one infinite return would make every later window `NaN`, also a
   window that does not hold it. The oracle refuses it too (#741).

The last two testsets run the estimators on the synthetic panel of `test06c_setup.jl` and
compare them with the oracle. The parity is pinned twice: once as a
re-derivation written out by hand in plain Julia, and once as the oracle's own output at
five observations, stored as a literal.
=#
include(joinpath(@__DIR__, "parity_harness.jl"))

# A small hand panel of returns. The Asset Panel needs one Panel Field, and no rolling
# Descriptor reads it, so a constant one carries the masks and nothing else.
function rolling_hand_panel(X::AbstractMatrix{<:Real};
                            amsk::AbstractMatrix{Bool} = trues(size(X)...),
                            emsk::AbstractMatrix{Bool} = amsk)
    T, N = size(X)
    res = asset_panel([NumericPanelInput(; name = "market_cap", vals = ones(T, N))];
                      amsk = amsk, emsk = emsk)
    return ReturnsResult(; nx = ["A" * string(i) for i in 1:N], X = X, pnl = res)
end

# The window sum of log returns, written out by hand: a missing return contributes zero, and
# the value is `NaN` unless every observation of the window is active.
function rolling_expected_sum(X::AbstractMatrix{<:Real}, amsk::AbstractMatrix{Bool},
                              window::Integer, skip::Integer)
    T, N = size(X)
    E = fill(NaN, T, N)
    for i in 1:N, t in (skip + window):T
        rows = (t - skip - window + 1):(t - skip)
        if all(k -> amsk[k, i], rows)
            E[t, i] = sum(k -> isnan(X[k, i]) ? 0.0 : log1p(X[k, i]), rows)
        end
    end
    E[.!amsk] .= NaN
    return E
end

# The rolling maximum, written out by hand: a missing return is ignored, and the value is
# `NaN` unless every observation of the window is active and one return exists.
function rolling_expected_max(X::AbstractMatrix{<:Real}, amsk::AbstractMatrix{Bool},
                              window::Integer)
    T, N = size(X)
    E = fill(NaN, T, N)
    for i in 1:N, t in window:T
        rows = (t - window + 1):t
        vals = [X[k, i] for k in rows if !isnan(X[k, i])]
        if all(k -> amsk[k, i], rows) && !isempty(vals)
            E[t, i] = maximum(vals)
        end
    end
    E[.!amsk] .= NaN
    return E
end

# Compare a log-return Descriptor with the sum written out by hand. The NaN pattern must
# agree exactly. The values agree only to the last bit, because the estimator differences one
# cumulative sum where the re-derivation adds the window up directly, and the two orders of
# summation round differently. `RollingMax` takes no sum, so its probes compare exactly.
# The comparison is cell by cell (`parity_compare`). A difference of two cumulative sums
# carries the round-off of the sums, not of the window, so a cell near zero needs `atol`.
# Over every probe of this file, the worst cell measures maxabs 6.1e-16, and maxrel reaches
# 2.8e-11 on a cell near zero (maxscaled 1.2e-15). `atol = 2e-15` is about three times the
# worst cell.
function rolling_agrees(D::AbstractMatrix{<:Real}, E::AbstractMatrix{<:Real}, label = "";
                        rtol = 1e-14, atol = 2e-15)
    return parity_compare(D, E; rtol = rtol, atol = atol, name = label).ok
end

@testset "Rolling Descriptor constructors and their refusals" begin
    @testset "The three named Descriptors fix the census defaults" begin
        @test isa(RollingMomentum(), RollingLogReturn)
        @test isa(Reversal(), RollingLogReturn)
        @test isa(MaxReturn(), RollingMax)
        for de in (RollingMomentum(), Reversal(), MaxReturn())
            @test isa(de, PortfolioOptimisers.AbstractDescriptorEstimator)
        end
        @test RollingMomentum().window == 252
        @test RollingMomentum().skip == 21
        @test RollingMomentum().sign == 1
        @test RollingMomentum().exponentiate == false
        @test Reversal().window == 21
        @test Reversal().skip == 0
        @test Reversal().sign == -1
        @test Reversal().exponentiate == false
        @test MaxReturn().window == 21
    end
    @testset "Every keyword of a named Descriptor overrides its default" begin
        de = RollingMomentum(; window = 126, skip = 5, sign = -1, exponentiate = true)
        @test de.window == 126
        @test de.skip == 5
        @test de.sign == -1
        @test de.exponentiate
        rv = Reversal(; window = 5, skip = 2, sign = 1, exponentiate = true)
        @test rv.window == 5
        @test rv.skip == 2
        @test rv.sign == 1
        @test rv.exponentiate
        @test MaxReturn(; window = 5).window == 5
    end
    @testset "A window, a skip and a sign outside their domain are refused" begin
        @test_throws DomainError RollingLogReturn(; window = 0)
        @test_throws DomainError RollingLogReturn(; window = -3)
        @test_throws DomainError RollingLogReturn(; window = 2, skip = -1)
        @test_throws DomainError RollingLogReturn(; window = 2, sign = 2)
        @test_throws DomainError RollingLogReturn(; window = 2, sign = 0)
        @test_throws DomainError RollingLogReturn(; window = 2, sign = -0.5)
        @test_throws DomainError RollingMax(; window = 0)
        @test_throws DomainError RollingMax(; window = -1)
        # The two the sign accepts, in both the integer and the floating point spelling.
        @test PortfolioOptimisers.assert_rolling_sign(1) === nothing
        @test PortfolioOptimisers.assert_rolling_sign(-1.0) === nothing
    end
    @testset "A carrier without returns or without an Asset Panel is refused" begin
        @test_throws PortfolioOptimisers.IsNothingError descriptor(Reversal(; window = 1),
                                                                   ReturnsResult())
        @test_throws PortfolioOptimisers.IsNothingError descriptor(MaxReturn(; window = 1),
                                                                   ReturnsResult())
        rd = ReturnsResult(; nx = ["A", "B"], X = [0.1 0.1; 0.2 0.2])
        @test_throws PortfolioOptimisers.IsNothingError descriptor(Reversal(; window = 1),
                                                                   rd)
        @test_throws PortfolioOptimisers.IsNothingError descriptor(MaxReturn(; window = 1),
                                                                   rd)
    end
end

@testset "RollingLogReturn on a hand panel" begin
    X = [0.10 -0.05; 0.20 0.10; -0.10 0.05; 0.05 NaN; 0.02 0.03]
    T, N = size(X)
    @testset "The window sum equals the sum of log1p written out by hand" begin
        for (window, skip) in ((1, 0), (2, 0), (3, 0), (2, 1), (1, 3))
            de = RollingLogReturn(; window = window, skip = skip)
            rd = rolling_hand_panel(X)
            @test rolling_agrees(descriptor(de, rd),
                                 rolling_expected_sum(X, trues(T, N), window, skip))
        end
    end
    @testset "The skip moves the window back and lengthens the warm-up" begin
        rd = rolling_hand_panel(X)
        plain = descriptor(RollingLogReturn(; window = 2, skip = 0), rd)
        moved = descriptor(RollingLogReturn(; window = 2, skip = 1), rd)
        @test all(isnan, moved[1:2, :])
        @test isequal(moved[3:T, :], plain[2:(T - 1), :])
        # A skip that leaves no room for a whole window gives a Descriptor of NaN.
        @test all(isnan, descriptor(RollingLogReturn(; window = 2, skip = 4), rd))
    end
    @testset "The sign negates the sum, and Reversal is the negated momentum" begin
        rd = rolling_hand_panel(X)
        up = descriptor(RollingLogReturn(; window = 2, sign = 1), rd)
        down = descriptor(RollingLogReturn(; window = 2, sign = -1), rd)
        @test isequal(down, -up)
        @test isequal(descriptor(Reversal(; window = 2), rd), down)
        @test isequal(descriptor(RollingMomentum(; window = 2, skip = 0), rd), up)
    end
    @testset "The exponentiate flag reads the simple return of the window" begin
        rd = rolling_hand_panel(X)
        S = descriptor(RollingLogReturn(; window = 2), rd)
        E = descriptor(RollingLogReturn(; window = 2, exponentiate = true), rd)
        @test isequal(E, expm1.(S))
        # Two returns compound, so the simple return of the window is their product.
        @test E[2, 1] ≈ (1 + X[1, 1]) * (1 + X[2, 1]) - 1
        # The sign is applied before the exponentiation, as the oracle does.
        R = descriptor(Reversal(; window = 2, exponentiate = true), rd)
        @test isequal(R, expm1.(-S))
    end
    @testset "An active observation with a missing return contributes zero" begin
        rd = rolling_hand_panel(X)
        D = descriptor(RollingLogReturn(; window = 2), rd)
        @test D[4, 2] ≈ log1p(X[3, 2])
        @test D[5, 2] ≈ log1p(X[5, 2])
        # A window whose returns are all missing sums to zero, not to NaN.
        Y = [0.1 NaN; 0.2 NaN; 0.3 NaN]
        @test descriptor(RollingLogReturn(; window = 2), rolling_hand_panel(Y))[3, 2] == 0
    end
    @testset "A window that is not wholly active is NaN across a delisting" begin
        amsk = trues(T, N)
        amsk[3, 2] = false
        rd = rolling_hand_panel(X; amsk = amsk)
        D = descriptor(RollingLogReturn(; window = 2), rd)
        @test rolling_agrees(D, rolling_expected_sum(X, amsk, 2, 0))
        # The inactive observation itself, and the one after it, both lose their window.
        @test isnan(D[3, 2])
        @test isnan(D[4, 2])
        @test !isnan(D[5, 2])
        # A late listing loses the whole warm-up of the window.
        late = trues(T, N)
        late[1:2, 1] .= false
        Dl = descriptor(RollingLogReturn(; window = 2), rolling_hand_panel(X; amsk = late))
        @test all(isnan, Dl[1:3, 1])
        @test !isnan(Dl[4, 1])
    end
    @testset "A return at or below -1 is refused" begin
        for bad in (-1.0, -1.5)
            Y = [0.1 0.1; 0.2 bad; 0.3 0.3]
            @test_throws DomainError descriptor(Reversal(; window = 1),
                                                rolling_hand_panel(Y))
        end
        # The refusal reads the whole matrix, so a bad return outside every window still
        # raises, and an inactive cell does not exempt it.
        Y = [-1.0 0.1; 0.2 0.2; 0.3 0.3]
        amsk = trues(3, 2)
        amsk[1, 1] = false
        @test_throws DomainError descriptor(Reversal(; window = 1),
                                            rolling_hand_panel(Y; amsk = amsk))
        # RollingMax takes no logarithm, so it answers where RollingLogReturn refuses.
        @test descriptor(RollingMax(; window = 1),
                         rolling_hand_panel([0.1 0.1; 0.2 -1.0; 0.3 0.3]))[2, 2] == -1
    end
    @testset "With a skip, an inactive current observation is NaN" begin
        # The window of observation 4 with a skip of 2 is rows 1 and 2, which are active, so
        # only the active mask of observation 4 itself makes the value NaN.
        amsk = trues(T, N)
        amsk[4, 2] = false
        D = descriptor(RollingLogReturn(; window = 2, skip = 2),
                       rolling_hand_panel(X; amsk = amsk))
        @test isnan(D[4, 2])
        @test D[4, 1] ≈ log1p(X[1, 1]) + log1p(X[2, 1])
    end
end

@testset "The rolling Descriptors refuse an infinite return (#741)" begin
    # Before the refusal, one infinite return in row 2 made rows 4 and 5 of RollingLogReturn
    # NaN, though neither window holds row 2: the difference of two infinite cumulative sums
    # is NaN.
    for bad in (Inf, -Inf)
        Y = [0.1 0.1; bad 0.2; 0.3 0.3; 0.1 0.1; 0.2 0.2]
        for de in (RollingLogReturn(; window = 2), Reversal(; window = 2),
                   RollingMax(; window = 2), MaxReturn(; window = 2))
            @test_throws DomainError descriptor(de, rolling_hand_panel(Y))
        end
    end
    # The refusal reads the whole matrix, so an inactive cell does not exempt it.
    Y = [Inf 0.1; 0.2 0.2; 0.3 0.3]
    amsk = trues(3, 2)
    amsk[1, 1] = false
    @test_throws DomainError descriptor(RollingMax(; window = 1),
                                        rolling_hand_panel(Y; amsk = amsk))
    msg = try
        descriptor(MaxReturn(; window = 1), rolling_hand_panel([0.1 0.1; 0.2 Inf]))
        ""
    catch err
        sprint(showerror, err)
    end
    @test occursin("observation 2 for asset 2", msg)
end

@testset "The rolling Descriptors derive their number type from the returns (#741)" begin
    # An integer panel raised InexactError: Int64(NaN) at the first NaN of the warm-up. It
    # is now floated, and gives the answer of the same panel in Float64 bit for bit.
    Xi = [0 1; 2 0; 1 3; 4 2; 0 1]
    Xf = Float64.(Xi)
    for de in
        (RollingMax(; window = 2), MaxReturn(; window = 3), RollingLogReturn(; window = 2),
         Reversal(; window = 1))
        Di = descriptor(de, rolling_hand_panel(Xi))
        @test eltype(Di) == Float64
        @test isequal(Di, descriptor(de, rolling_hand_panel(Xf)))
    end
    # A Float32 panel keeps a Float32 Descriptor.
    X32 = Float32[0.1 0.2; 0.05 -0.1; 0.3 0.0]
    for de in (RollingLogReturn(; window = 2), RollingMax(; window = 2))
        @test eltype(descriptor(de, rolling_hand_panel(X32))) == Float32
    end
end

@testset "The named Descriptors agree with the oracle on a panel with gaps" begin
    #=
    The oracle ran on this 8 x 3 panel. Asset 1 misses the return of row 4 inside an active
    listing. Asset 2 is inactive in rows 3 and 4,
    a gap inside its listing. Asset 3 lists late, at row 3, and misses the return of row 6. The
    oracle refuses a finite value on an inactive cell, so those cells hold NaN here.
    `mom_w2_s2` pins the rule on the current observation: row 4 of asset 2 has an active
    window, rows 1 and 2, and it is NaN because observation 4 is inactive.
    =#
    X = [0.010 -0.020 0.030
         0.025 0.015 -0.010
         -0.030 0.040 0.020
         NaN -0.010 0.005
         0.015 0.020 -0.025
         0.040 -0.035 NaN
         -0.012 0.018 0.012
         0.022 0.030 -0.008]
    amsk = trues(8, 3)
    amsk[3:4, 2] .= false
    amsk[1:2, 3] .= false
    X[.!amsk] .= NaN
    rd = rolling_hand_panel(X; amsk = amsk)
    ref_mom_w3_s2 = [NaN NaN NaN
                     NaN NaN NaN
                     NaN NaN NaN
                     NaN NaN NaN
                     0.004183735958831036 NaN NaN
                     -0.005766594894337047 NaN NaN
                     -0.015570594990957892 NaN -0.0005276391770710903
                     0.05410932564703195 NaN -0.020330266473250803]
    ref_mom_w2_s2 = [NaN NaN NaN
                     NaN NaN NaN
                     NaN NaN NaN
                     0.03464294344353958 NaN NaN
                     -0.005766594894337047 NaN NaN
                     -0.030459207484708546 NaN 0.024790168807218786
                     0.014888612493750654 NaN -0.020330266473250803
                     0.05410932564703195 -0.015824550346971413 -0.025317807984289876]
    ref_mom_w2_s0_exp = [NaN NaN NaN
                         0.03525 -0.005300000000000001 NaN
                         -0.0057500000000000025 NaN NaN
                         -0.03 NaN 0.0251
                         0.015 NaN -0.020125
                         0.0556 -0.015700000000000002 -0.025
                         0.02752 -0.017630000000000003 0.012
                         0.009736000000000007 0.04854 0.0039039999999999995]
    ref_rev_w2 = [NaN NaN NaN
                  -0.03464294344353958 0.005314094823768795 NaN
                  0.005766594894337047 NaN NaN
                  0.030459207484708546 NaN -0.024790168807218786
                  -0.014888612493750654 NaN 0.020330266473250803
                  -0.05410932564703195 0.015824550346971413 0.025317807984289876
                  -0.02714813191901206 0.017787259514820127 -0.011928570865273802
                  -0.00968891054724346 -0.0473987203698754 -0.003896399168009542]
    ref_max_w3 = [NaN NaN NaN
                  NaN NaN NaN
                  0.025 NaN NaN
                  0.025 NaN NaN
                  0.015 NaN 0.02
                  0.04 NaN 0.005
                  0.04 0.02 0.012
                  0.04 0.03 0.012]
    cases = ((RollingMomentum(; window = 3, skip = 2), ref_mom_w3_s2),
             (RollingMomentum(; window = 2, skip = 2), ref_mom_w2_s2),
             (RollingMomentum(; window = 2, skip = 0, exponentiate = true),
              ref_mom_w2_s0_exp), (Reversal(; window = 2), ref_rev_w2),
             (MaxReturn(; window = 3), ref_max_w3))
    # Against the oracle, the worst cell of the four sums measures maxrel 4.4e-16 and
    # maxabs 6.9e-18, and the maximum is bit-equal. The cumulative sums of eight rows stay
    # small, so no cell needs `atol` here, and `rtol = 1e-14` is about twenty times the
    # worst value.
    for (i, (de, ref)) in enumerate(cases)
        @test rolling_agrees(descriptor(de, rd), ref, "gapped case $(i)"; atol = 0.0)
    end
    @test isequal(descriptor(MaxReturn(; window = 3), rd), ref_max_w3)
    @test isnan(ref_mom_w2_s2[4, 2])
end

@testset "RollingMax on a hand panel" begin
    X = [0.10 -0.05; 0.20 0.10; -0.10 0.05; 0.05 NaN; 0.02 0.03]
    T, N = size(X)
    @testset "The maximum equals the one written out by hand" begin
        for window in (1, 2, 3, 5)
            rd = rolling_hand_panel(X)
            @test isequal(descriptor(RollingMax(; window = window), rd),
                          rolling_expected_max(X, trues(T, N), window))
        end
    end
    @testset "A missing return is ignored, and an all-missing window is NaN" begin
        rd = rolling_hand_panel(X)
        D = descriptor(RollingMax(; window = 2), rd)
        @test D[4, 2] == X[3, 2]
        Y = [0.1 NaN; 0.2 NaN; 0.3 NaN]
        @test isnan(descriptor(RollingMax(; window = 2), rolling_hand_panel(Y))[3, 2])
    end
    @testset "A window that is not wholly active is NaN, and so is the warm-up" begin
        amsk = trues(T, N)
        amsk[3, 2] = false
        rd = rolling_hand_panel(X; amsk = amsk)
        D = descriptor(RollingMax(; window = 2), rd)
        @test isequal(D, rolling_expected_max(X, amsk, 2))
        @test isnan(D[3, 2])
        @test isnan(D[4, 2])
        @test all(isnan, D[1, :])
        # A window of one has no warm-up at all.
        @test !any(isnan, descriptor(RollingMax(; window = 1), rolling_hand_panel(X))[1, :])
    end
    @testset "A window of one is the return itself, which the oracle refuses" begin
        # The oracle raises for `window <= 1`. The port answers, which
        # adds a mode rather than removes one. Restore the refusal by raising the bound of
        # `assert_gt0(window, :window)` in the inner constructor of `RollingMax`.
        rd = rolling_hand_panel(X)
        D = descriptor(RollingMax(; window = 1), rd)
        @test isequal(D, X)
        @test D[1, 1] == X[1, 1]
        @test isnan(D[4, 2])
    end
end

@testset "Every rolling Descriptor runs on the synthetic panel" begin
    rng = StableRNG(717)
    rd = synthetic_asset_panel(; n_assets = 12, n_observations = 300, n_industries = 3,
                               rng = rng).rd
    amsk = rd.pnl.amsk
    T, N = size(amsk)
    X = Matrix{Float64}(rd.X)
    @testset "Shape, the inactive fill, and finite-or-NaN everywhere" begin
        for de in (RollingMomentum(; window = 63, skip = 21), Reversal(), MaxReturn(),
                   RollingMax(; window = 5))
            D = descriptor(de, rd)
            @test size(D) == (T, N)
            @test all(isnan, D[.!amsk])
            @test all(x -> isnan(x) || isfinite(x), D)
            @test count(!isnan, D) > 0
        end
    end
    @testset "Each estimator equals the formula written out by hand" begin
        for (window, skip) in ((63, 21), (21, 0), (252, 21), (1, 0))
            de = RollingLogReturn(; window = window, skip = skip)
            @test rolling_agrees(descriptor(de, rd),
                                 rolling_expected_sum(X, amsk, window, skip))
        end
        @test rolling_agrees(descriptor(Reversal(), rd),
                             -rolling_expected_sum(X, amsk, 21, 0))
        for window in (5, 21)
            @test isequal(descriptor(RollingMax(; window = window), rd),
                          rolling_expected_max(X, amsk, window))
        end
    end
    @testset "A window of 252 leaves only the last observations readable" begin
        D = descriptor(RollingMomentum(; window = 252, skip = 21), rd)
        @test all(isnan, D[1:272, :])
        @test count(!isnan, D) == 303
    end
    @testset "An asset view of the carrier gives the same Descriptor as a slice" begin
        v = PortfolioOptimisers.port_opt_view(rd, [2, 5, 9])
        for de in (RollingMomentum(; window = 63, skip = 21), Reversal(), MaxReturn())
            @test isequal(descriptor(de, v), descriptor(de, rd)[:, [2, 5, 9]])
        end
    end
end

@testset "The oracle's own output, pinned" begin
    #=
    The oracle was driven on the same synthetic panel, rebuilt as its own
    panel container with the same active and estimation masks, and its output at
    observations 296 to 300 is stored below. Asset 8 delists at observation 295, so its
    column pins the inactive rule as well as the arithmetic.

    Every NaN pattern agrees on all seven cases that were run. `RollingMax` agrees to the
    last bit. `RollingLogReturn` agrees to 5.6e-17 in absolute value, which is the residue
    of a one-bit difference in `log1p` itself on 9 of the 3474 returns of the panel; the
    two cumulative sums are taken in the same order.
    =#
    oracle_reversal_21 = [0.005157908733073507 0.028093695361967097 0.12535624416725344 0.026813586525631483 0.0343297254340056 0.0840484614301415 0.09754260674030271 NaN 0.13459419284319432 0.0634559227890284 0.1386150341782934 0.030867361361560597
                                   -0.006743816405957448 0.013834529481491276 0.11600843205164058 0.02436438201512081 0.03047133581177322 0.07996858781623134 0.08743823363933584 NaN 0.13363775359043986 0.05438198569666042 0.12405686794558396 0.035775406692541395
                                   0.0006776615676298559 0.030254850755471074 0.11219040135996705 0.01522123064251657 0.02705239073655294 0.08545514958512845 0.07628570799226408 NaN 0.11405493566344696 0.05991986944125616 0.11630365270684817 0.05049959289708941
                                   -0.02950449992547463 0.0019573302318608987 0.08019697436427603 -0.04584546850027882 0.022572876148533413 0.05568418435993244 0.05963487774185494 NaN 0.08728377563883488 0.06670118720530782 0.06762213916522058 0.0507463797793371
                                   -0.044203394546773106 -0.00048265885253762075 0.09359272043606551 0.0049821566375432 0.026374438266559413 0.05894600925303006 0.045928205707354064 NaN 0.09270316512248242 0.07705538088004404 0.09992704661917473 0.06624252677832965]
    oracle_max_return_21 = [0.047040239006114946 0.019845745986494884 0.015747046220745013 0.05305034898441107 0.023668269945923172 0.031145013351726942 0.03526007018851601 NaN 0.022958162687840997 0.03231014345231089 0.017469906096132946 0.014427781564077926
                                     0.047040239006114946 0.019845745986494884 0.015747046220745013 0.05305034898441107 0.023668269945923172 0.031145013351726942 0.03526007018851601 NaN 0.022958162687840997 0.03231014345231089 0.017469906096132946 0.014427781564077926
                                     0.047040239006114946 0.019845745986494884 0.015747046220745013 0.05305034898441107 0.023668269945923172 0.031145013351726942 0.03526007018851601 NaN 0.022958162687840997 0.03231014345231089 0.017469906096132946 0.014427781564077926
                                     0.047040239006114946 0.019845745986494884 0.018654034668595016 0.05305034898441107 0.023668269945923172 0.031145013351726942 0.03526007018851601 NaN 0.022958162687840997 0.03231014345231089 0.028353674929423155 0.014427781564077926
                                     0.047040239006114946 0.019845745986494884 0.018654034668595016 0.05305034898441107 0.023668269945923172 0.031145013351726942 0.03526007018851601 NaN 0.022958162687840997 0.03231014345231089 0.028353674929423155 0.014427781564077926]
    rng = StableRNG(717)
    rd = synthetic_asset_panel(; n_assets = 12, n_observations = 300, n_industries = 3,
                               rng = rng).rd
    rows = 296:300
    rev = descriptor(Reversal(), rd)[rows, :]
    mx = descriptor(MaxReturn(), rd)[rows, :]
    @test isequal(isnan.(rev), isnan.(oracle_reversal_21))
    @test isequal(isnan.(mx), isnan.(oracle_max_return_21))
    ok = .!isnan.(oracle_reversal_21)
    # Measured maxabs 8.3e-17. A reversal is a sum of returns, so a cell near zero carries a
    # large relative round-off, and the pin is absolute: one ulp of the largest entry, 0.05.
    @test parity_compare(rev[ok], oracle_reversal_21[ok]; rtol = 0.0, atol = 1e-16,
                         name = "Reversal 21").ok
    okm = .!isnan.(oracle_max_return_21)
    # The window selects one return of the history, and that return is itself the output of a
    # product whose reduction order is the BLAS thread count of the machine. A CI runner and a
    # developer machine therefore differ in the last bit, so the pin is one ulp wide, as the
    # `rev` pin above is.
    # Measured maxabs 3.5e-18.
    @test parity_compare(mx[okm], oracle_max_return_21[okm]; rtol = 0.0, atol = 1e-16,
                         name = "MaxReturn 21").ok
end

#=
Check `src/05_Moments/32_CrossSectionalFactorModel/04_FactorExposures/04_EWMeanDescriptors.jl` and
`05_EWVolatilityDescriptors.jl` against the contract their docstrings state, and against the
oracle. Issue #718, map #643.

FOUR CONVENTIONS SHAPE THE PROBES, and the first three are `test_08s_descriptors.jl`'s.

1. A BLANK NEVER REACHES A CARRIER, so a Descriptor reads its Panel Fields back through the
   observed-mask column and never sees a fill value as data.

2. AN INACTIVE CELL IS `NaN`, whatever its Panel Fields hold.

3. A NON-POSITIVE DENOMINATOR IS `NaN`, NEVER AN ERROR.

4. A GAP HOLDS THE RECURSION. An asset that is listed but carries no value at an observation
   neither advances its state nor resets it, and the observation does not count toward the
   warm-up. Only [`EWVolatility`](@ref) resets, and only where the asset turns inactive. Every
   hand recursion below writes that rule out separately from the code under test.

The two recursions the census pins are written out as literals: `EWMomentum(half_life = 5,
skip = 0)` and `EWVolatility(half_life = 5)`. The last testset diffs two cases against numbers
the oracle printed on the synthetic panel of `test06c_setup.jl`.
=#

# A small hand panel. Every field takes a forward fill, so each earns an observed-mask column
# and a raw `NaN` reads back as `NaN` through the mask rather than as the fill value.
function ew_hand_panel(fields::AbstractVector{<:Pair{String, <:AbstractMatrix}},
                       X::AbstractMatrix{<:Real};
                       amsk::AbstractMatrix{Bool} = trues(size(X)...),
                       emsk::AbstractMatrix{Bool} = amsk)
    inputs = [NumericPanelInput(; name = n, vals = v, alg = ForwardPanelFill(; val = 0.0))
              for (n, v) in fields]
    res = asset_panel(inputs; amsk = amsk, emsk = emsk)
    return ReturnsResult(; nx = ["A" * string(i) for i in 1:size(X, 2)], X = X, pnl = res)
end

# Two matrices agree when they hold `NaN` in the same cells and are close everywhere else.
function ew_same(A::AbstractMatrix, B::AbstractMatrix; atol::Real = 0, rtol::Real = 1e-12)
    if !(size(A) == size(B))
        return false
    end
    return all(CartesianIndices(A)) do k
        a, b = A[k], B[k]
        return isnan(a) ? isnan(b) : (!isnan(b) && isapprox(a, b; atol = atol, rtol = rtol))
    end
end

# The exponentially weighted mean recursion, written out by hand. A cell that is not finite
# holds the state and does not count toward the warm-up.
function ew_hand_mean(R::AbstractMatrix{<:Real}, decay::Real, min_obs::Integer)
    T, N = size(R)
    S = fill(NaN, T, N)
    for i in 1:N
        s, n = 0.0, 0
        for t in 1:T
            if isfinite(R[t, i])
                s = decay * s + (1 - decay) * R[t, i]
                n += 1
            end
            if n >= min_obs
                S[t, i] = s
            end
        end
    end
    return S
end

# The exponentially weighted volatility, written out by hand: uncentred, bias corrected by
# `1 / (1 - λ^n)`, and reset where the asset turns inactive. `mar` of `nothing` is the
# two-sided form.
function ew_hand_volatility(X::AbstractMatrix{<:Real}, amsk::AbstractMatrix{Bool},
                            decay::Real, min_obs::Integer; mar = nothing)
    T, N = size(X)
    D = fill(NaN, T, N)
    act, v, n = trues(N), zeros(N), zeros(Int, N)
    for t in 1:T, i in 1:N
        if act[i] && !amsk[t, i]
            v[i], n[i] = 0.0, 0
        end
        act[i] = amsk[t, i]
        x = X[t, i]
        if amsk[t, i] && isfinite(x)
            c = isnothing(mar) ? x : min(x - mar, 0.0)
            v[i] = decay * v[i] + (1 - decay) * c^2
            n[i] += 1
        end
        if n[i] >= min_obs
            D[t, i] = sqrt(v[i] / (1 - decay^n[i]))
        end
    end
    return D
end

# Read one raw field and its observed mask back off the carrier by column name.
function ew_raw_field(rd::ReturnsResult, name::AbstractString)
    Z = panel_feature_matrix(rd.pnl)[2]
    col = findfirst(==(name), panel_feature_matrix(rd.pnl)[1])
    ocol = findfirst(==(name * "::observed"), panel_feature_matrix(rd.pnl)[1])
    V = Matrix{Float64}(Z[:, :, col])
    if !isnothing(ocol)
        V[iszero.(Z[:, :, ocol])] .= NaN
    end
    return V
end

@testset "Exponentially weighted Descriptor constructors" begin
    @testset "Every named constructor fixes its census fields and half-life" begin
        mom = EWMomentum()
        @test isa(mom, EWMean)
        @test mom.decay ≈ exp2(-inv(87.0))
        @test mom.min_obs == 87
        @test mom.skip == 21
        @test mom.exponentiate == false

        trn = EWShareTurnover()
        @test isa(trn, EWVolumeRatio)
        @test trn.num == "adj_volume"
        @test trn.den == "adj_shares_outstanding"
        @test trn.decay ≈ exp2(-inv(21.0))
        @test trn.min_obs == 21

        ami = EWAmihudIlliquidity()
        @test isa(ami, EWVolumeRatio)
        @test isnothing(ami.num)
        @test ami.den == ["adj_close", "adj_volume"]
        @test ami.decay ≈ exp2(-inv(63.0))
        @test ami.min_obs == 63

        dtc = DaysToCover()
        @test isa(dtc, DaysToCover)
        @test dtc.num == "short_interest"
        @test dtc.den == "adj_volume"
        @test dtc.decay ≈ exp2(-inv(21.0))
        @test dtc.min_obs == 21

        vol = EWVolatility()
        @test isa(vol, EWVolatility)
        @test vol.ce.decay ≈ exp2(-inv(40.0))
        @test vol.ce.min_obs == 40
        @test isa(vol.ce.centring, PreCentred)
        @test isnothing(vol.ce.regime_method)
        @test isa(vol.alg, FullMoment)
        @test iszero(vol.mar)

        dvol = EWDownsideVolatility()
        @test isa(dvol, EWVolatility)
        @test dvol.ce.decay ≈ exp2(-inv(40.0))
        @test isa(dvol.alg, SemiMoment)
        @test iszero(dvol.mar)
    end
    @testset "A keyword override renames a field, and half_life fixes the two it converts" begin
        @test EWShareTurnover(; num = "vol", den = "shr").num == "vol"
        @test EWShareTurnover(; num = "vol", den = "shr").den == "shr"
        @test DaysToCover(; num = "si", den = "vol").num == "si"
        @test EWAmihudIlliquidity(; den = ["px", "vol"]).den == ["px", "vol"]
        @test EWMomentum(; half_life = 4).decay ≈ exp2(-inv(4.0))
        @test EWMomentum(; half_life = 4).min_obs == 4
        # A value passed for a converted field is used as it stands.
        @test EWMomentum(; half_life = 4, min_obs = 9).min_obs == 9
        @test DaysToCover(; half_life = 4, decay = 0.25).decay == 0.25
        @test EWVolatility(; half_life = 7).ce.min_obs == 7
        @test EWDownsideVolatility(; half_life = 7, mar = 0.01).mar == 0.01
    end
    @testset "The half-life conversions" begin
        @test PortfolioOptimisers.half_life_decay(1.0) == 0.5
        @test PortfolioOptimisers.half_life_decay(40.0) ≈ exp2(-inv(40.0))
        @test PortfolioOptimisers.half_life_min_obs(40.0) == 40
        @test PortfolioOptimisers.half_life_min_obs(40.5) == 41
        @test PortfolioOptimisers.half_life_min_obs(0.25) == 1
        @test_throws Exception PortfolioOptimisers.half_life_decay(0.0)
        @test_throws Exception PortfolioOptimisers.half_life_min_obs(-1.0)
    end
    @testset "The constructors refuse what the docstrings say they refuse" begin
        @test_throws DomainError EWMean(; decay = 0.0, min_obs = 1)
        @test_throws DomainError EWMean(; decay = 1.0, min_obs = 1)
        @test_throws DomainError EWMean(; decay = 0.5, min_obs = 1, skip = -1)
        @test_throws Exception EWMean(; decay = 0.5, min_obs = 0)
        @test_throws DomainError EWVolumeRatio(; num = "a", den = "b", decay = 1.5,
                                               min_obs = 1)
        @test_throws Exception EWVolumeRatio(; num = "", den = "b", decay = 0.5,
                                             min_obs = 1)
        @test_throws Exception EWVolumeRatio(; num = "a", den = String[], decay = 0.5,
                                             min_obs = 1)
        @test_throws Exception EWVolumeRatio(; num = "a", den = ["b", ""], decay = 0.5,
                                             min_obs = 1)
        @test_throws Exception EWVolumeRatio(; num = Pair{String, Float64}[], den = "b",
                                             decay = 0.5, min_obs = 1)
        @test_throws Exception DaysToCover(; num = "", den = "b")
        @test_throws Exception EWVolatility(; mar = NaN)
    end
    @testset "A Descriptor over the returns still refuses a carrier with no Asset Panel" begin
        rd = ReturnsResult(; nx = ["A", "B"], X = [0.01 0.02; 0.03 0.04])
        @test_throws Exception descriptor(EWMomentum(; half_life = 2, skip = 0), rd)
        @test_throws Exception descriptor(EWVolatility(; half_life = 2), rd)
    end
end

@testset "Exponentially weighted means on a hand panel" begin
    X = [0.10 0.20; -0.05 NaN; 0.04 0.03; NaN -0.02; 0.01 0.06]
    vol = [10.0 20.0; 30.0 40.0; 50.0 60.0; 70.0 80.0; 90.0 100.0]
    shr = [100.0 200.0; 100.0 200.0; 100.0 400.0; 100.0 400.0; 100.0 400.0]
    px = [5.0 6.0; 5.5 6.5; 6.0 7.0; 6.5 7.5; 7.0 8.0]
    si = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0; 9.0 10.0]
    rd = ew_hand_panel(["adj_volume" => copy(vol), "adj_shares_outstanding" => copy(shr),
                        "adj_close" => copy(px), "short_interest" => copy(si)], X)
    lam, mo = 0.5, 2
    @testset "EWMean equals the hand recursion over log1p, and a NaN holds the state" begin
        D = descriptor(EWMean(; decay = lam, min_obs = mo), rd)
        @test ew_same(D, ew_hand_mean(log1p.(X), lam, mo))
        # The second asset misses observation two, so its warm-up ends one row later.
        @test isnan(D[2, 2])
        @test !isnan(D[2, 1])
    end
    @testset "A skip delays the input, and the first skip rows read nothing" begin
        for skip in (1, 2, 3)
            R = fill(NaN, size(X))
            for i in axes(X, 2), t in (skip + 1):size(X, 1)
                R[t, i] = log1p(X[t - skip, i])
            end
            D = descriptor(EWMean(; decay = lam, min_obs = 1, skip = skip), rd)
            @test ew_same(D, ew_hand_mean(R, lam, 1))
            @test all(isnan, view(D, 1:skip, :))
        end
    end
    @testset "exponentiate returns the same signal in return units" begin
        de = EWMean(; decay = lam, min_obs = 1)
        D = descriptor(de, rd)
        E = descriptor(EWMean(; decay = lam, min_obs = 1, exponentiate = true), rd)
        @test ew_same(E, expm1.(D))
    end
    @testset "A return at or below -1 raises, and NaN does not" begin
        bad = ew_hand_panel(["adj_volume" => [1.0 2.0; 3.0 4.0]], [0.1 0.2; -1.0 0.3])
        @test_throws DomainError descriptor(EWMean(; decay = lam, min_obs = 1), bad)
        ok = ew_hand_panel(["adj_volume" => [1.0 2.0; 3.0 4.0]], [0.1 0.2; NaN 0.3])
        @test size(descriptor(EWMean(; decay = lam, min_obs = 1), ok)) == (2, 2)
    end
    @testset "EWVolumeRatio equals the recursion over the ratio of its two sides" begin
        D = descriptor(EWVolumeRatio(; num = "adj_volume", den = "adj_shares_outstanding",
                                     decay = lam, min_obs = mo), rd)
        @test ew_same(D, ew_hand_mean(vol ./ shr, lam, mo))
    end
    @testset "A nothing numerator reads the absolute returns, and a product denominator multiplies" begin
        D = descriptor(EWVolumeRatio(; num = nothing, den = ["adj_close", "adj_volume"],
                                     decay = lam, min_obs = 1), rd)
        @test ew_same(D, ew_hand_mean(abs.(X) ./ (px .* vol), lam, 1))
    end
    @testset "A combination side sums its terms with their coefficients" begin
        D = descriptor(EWVolumeRatio(; num = ["adj_volume" => 1, "short_interest" => -1],
                                     den = "adj_shares_outstanding", decay = lam,
                                     min_obs = 1), rd)
        @test ew_same(D, ew_hand_mean((vol .- si) ./ shr, lam, 1))
    end
    @testset "A denominator that is not strictly positive is NaN, and holds the state" begin
        zvol = copy(vol)
        zvol[2, 1] = 0.0
        nvol = copy(shr)
        nvol[3, 2] = -1.0
        rd2 = ew_hand_panel(["adj_volume" => zvol, "adj_shares_outstanding" => nvol], X)
        D = descriptor(EWVolumeRatio(; num = "adj_volume", den = "adj_shares_outstanding",
                                     decay = lam, min_obs = 1), rd2)
        R = [nvol[k] > 0 ? zvol[k] / nvol[k] : NaN for k in CartesianIndices(zvol)]
        @test ew_same(D, ew_hand_mean(R, lam, 1))
        @test D[3, 2] == D[2, 2]
    end
    @testset "A preset refuses a share count or a price at or below zero, not a zero volume" begin
        # A share count and a price are positive by construction, so the `gt0` guard of the
        # presets refuses a value at or below zero (#1379, ADR 0108). A volume is zero on a
        # day with no trade, so it is not guarded, and its ratio is NaN.
        @test EWShareTurnover().gt0 == ["adj_shares_outstanding"]
        @test EWAmihudIlliquidity().gt0 == ["adj_close"]
        @test EWAmihudIlliquidity(; den = ["p", "v"]).gt0 == ["p"]
        @test isnothing(EWVolumeRatio(; num = "a", den = "b", decay = lam, min_obs = 1).gt0)
        @test_throws ArgumentError EWShareTurnover(; gt0 = ["adj_close"])
        fields(; v = vol, s = shr, p = px) = ["adj_volume" => copy(v),
                                              "adj_shares_outstanding" => copy(s),
                                              "adj_close" => copy(p),
                                              "short_interest" => copy(si)]
        zshr = copy(shr)
        zshr[3, 2] = 0.0
        rdz = ew_hand_panel(fields(; s = zshr), X)
        @test_throws DomainError descriptor(EWShareTurnover(; half_life = 2), rdz)
        # With the guard off, the zero share count gives no ratio and holds the state.
        D = descriptor(EWShareTurnover(; half_life = 2, min_obs = 1, gt0 = nothing), rdz)
        @test D[3, 2] == D[2, 2]
        npx = copy(px)
        npx[2, 1] = -1.0
        rdn = ew_hand_panel(fields(; p = npx), X)
        @test_throws DomainError descriptor(EWAmihudIlliquidity(; half_life = 2), rdn)
        zvol = copy(vol)
        zvol[2, 1] = 0.0
        rdv = ew_hand_panel(fields(; v = zvol), X)
        @test size(descriptor(EWShareTurnover(; half_life = 2), rdv)) == size(X)
        @test size(descriptor(EWAmihudIlliquidity(; half_life = 2), rdv)) == size(X)
    end
    @testset "DaysToCover smooths the denominator alone, and only a positive value advances" begin
        zvol = copy(vol)
        zvol[3, 1] = 0.0
        rd2 = ew_hand_panel(["adj_volume" => zvol, "short_interest" => copy(si)], X)
        D = descriptor(DaysToCover(; decay = lam, min_obs = mo), rd2)
        V = [v > 0 ? v : NaN for v in zvol]
        @test ew_same(D, si ./ ew_hand_mean(V, lam, mo))
        # The zero volume neither advances the state nor counts toward the warm-up.
        @test D[4, 1] ≈ si[4, 1] / ew_hand_mean(V, lam, mo)[4, 1]
    end
    @testset "An inactive cell is NaN in every archetype" begin
        amsk = trues(size(X))
        amsk[4, 1] = false
        rd2 = ew_hand_panel(["adj_volume" => copy(vol),
                             "adj_shares_outstanding" => copy(shr),
                             "short_interest" => copy(si)], ifelse.(amsk, X, NaN);
                            amsk = amsk)
        for de in (EWMean(; decay = lam, min_obs = 1),
                   EWVolumeRatio(; num = "adj_volume", den = "adj_shares_outstanding",
                                 decay = lam, min_obs = 1),
                   DaysToCover(; decay = lam, min_obs = 1))
            @test isnan(descriptor(de, rd2)[4, 1])
        end
    end
    @testset "The estimator holds no data: two carriers, one estimator" begin
        de = EWVolumeRatio(; num = "adj_volume", den = "adj_shares_outstanding",
                           decay = lam, min_obs = 1)
        D1 = descriptor(de, rd)
        rd2 = ew_hand_panel(["adj_volume" => 2 .* vol,
                             "adj_shares_outstanding" => copy(shr)], X)
        D2 = descriptor(de, rd2)
        @test ew_same(D2, 2 .* D1)
        @test ew_same(D1, descriptor(de, rd))
    end
end

@testset "Exponentially weighted volatility on a hand panel" begin
    X = [0.10 0.20; -0.05 NaN; 0.04 0.03; -0.02 -0.06; 0.01 0.05]
    mcap = ones(size(X))
    rd = ew_hand_panel(["market_cap" => copy(mcap)], X)
    @testset "EWVolatility equals sqrt(S / (1 - lambda^n)) over the raw returns" begin
        for hl in (1.0, 2.0, 5.0)
            de = EWVolatility(; half_life = hl)
            lam, mo = exp2(-inv(hl)), max(1, ceil(Int, hl))
            @test ew_same(descriptor(de, rd),
                          ew_hand_volatility(X, trues(size(X)), lam, mo); rtol = 1e-10)
        end
    end
    @testset "EWDownsideVolatility clips every return above the target to zero" begin
        for mar in (0.0, 0.02)
            de = EWDownsideVolatility(; half_life = 2, mar = mar)
            @test ew_same(descriptor(de, rd),
                          ew_hand_volatility(X, trues(size(X)), exp2(-inv(2.0)), 2;
                                             mar = mar); rtol = 1e-10)
        end
    end
    @testset "The transform is the alg slot, and it reads the target" begin
        @test PortfolioOptimisers.ew_volatility_input(FullMoment(), X, 0.5) === X
        @test PortfolioOptimisers.ew_volatility_input(SemiMoment(), [0.1 -0.2], 0.0) ==
              [0.0 -0.2]
        @test PortfolioOptimisers.ew_volatility_input(SemiMoment(), [0.1 -0.2], 0.2) ==
              [-0.1 -0.4]
    end
    @testset "An asset that turns inactive restarts its recursion" begin
        amsk = trues(size(X))
        amsk[3, 2] = false
        rd2 = ew_hand_panel(["market_cap" => copy(mcap)], ifelse.(amsk, X, NaN);
                            amsk = amsk)
        de = EWVolatility(; half_life = 1)
        D = descriptor(de, rd2)
        @test ew_same(D, ew_hand_volatility(ifelse.(amsk, X, NaN), amsk, 0.5, 1);
                      rtol = 1e-10)
        @test isnan(D[3, 2])
        # The observation after the gap is the first of a fresh recursion, so it equals the
        # volatility of one observation.
        @test D[4, 2] ≈ abs(X[4, 2])
    end
    @testset "A caller may pass any variance estimator in the ce slot" begin
        de = EWVolatility(; ce = SimpleVariance(; corrected = false))
        D = descriptor(de, rd)
        @test size(D) == size(X)
        # The expanding-window fallback refits, so row `t` is the sample deviation of the
        # first `t` observations rather than a recursion.
        @test iszero(D[1, 1])
        @test D[3, 1] ≈ std(view(X, 1:3, 1); corrected = false)
    end
end

@testset "The two recursions the census pins" begin
    X = [0.010 0.020; -0.005 0.030; 0.004 -0.010; 0.007 0.002; -0.003 0.006; 0.002 -0.004;
         0.001 0.005; -0.002 0.003]
    rd = ew_hand_panel(["market_cap" => ones(size(X))], X)
    lam = exp2(-inv(5.0))
    @testset "EWMomentum(half_life = 5, skip = 0) is the recursion over log1p" begin
        S = zeros(2)
        expected = fill(NaN, size(X))
        for t in axes(X, 1), i in axes(X, 2)
            S[i] = lam * S[i] + (1 - lam) * log1p(X[t, i])
            if t >= 5
                expected[t, i] = S[i]
            end
        end
        @test ew_same(descriptor(EWMomentum(; half_life = 5, skip = 0), rd), expected)
    end
    @testset "EWVolatility(half_life = 5) is sqrt(var / (1 - lambda^n))" begin
        V = zeros(2)
        expected = fill(NaN, size(X))
        for t in axes(X, 1), i in axes(X, 2)
            V[i] = lam * V[i] + (1 - lam) * X[t, i]^2
            if t >= 5
                expected[t, i] = sqrt(V[i] / (1 - lam^t))
            end
        end
        @test ew_same(descriptor(EWVolatility(; half_life = 5), rd), expected; rtol = 1e-10)
    end
end

@testset "Every exponentially weighted Descriptor runs on the synthetic panel" begin
    res = synthetic_asset_panel(; n_assets = 12, n_observations = 300, n_industries = 3,
                                rng = StableRNGs.StableRNG(718))
    rd = res.rd
    amsk = rd.pnl.amsk
    des = ("EWMomentum" => EWMomentum(; half_life = 5, skip = 3),
           "EWShareTurnover" => EWShareTurnover(; half_life = 5),
           "EWAmihudIlliquidity" => EWAmihudIlliquidity(; half_life = 5),
           "DaysToCover" => DaysToCover(; half_life = 5),
           "EWVolatility" => EWVolatility(; half_life = 5),
           "EWDownsideVolatility" => EWDownsideVolatility(; half_life = 5))
    @testset "Shape, the inactive fill, and finite-or-NaN everywhere" begin
        for (name, de) in des
            D = descriptor(de, rd)
            @test size(D) == size(rd.X)
            @test all(isnan, D[.!amsk])
            @test all(x -> isnan(x) || isfinite(x), D)
            @test any(isfinite, D)
        end
    end
    @testset "Every one equals its census formula, written from panel_feature_matrix(rd.pnl)[1] and panel_feature_matrix(rd.pnl)[2]" begin
        lam, mo = exp2(-inv(5.0)), 5
        X = rd.X
        vol = ew_raw_field(rd, "adj_volume")
        shr = ew_raw_field(rd, "adj_shares_outstanding")
        px = ew_raw_field(rd, "adj_close")
        si = ew_raw_field(rd, "short_interest")

        R = fill(NaN, size(X))
        for i in axes(X, 2), t in 4:size(X, 1)
            R[t, i] = log1p(X[t - 3, i])
        end
        mom = ew_hand_mean(R, lam, mo)
        mom[.!amsk] .= NaN
        @test ew_same(descriptor(EWMomentum(; half_life = 5, skip = 3), rd), mom)

        trn = ew_hand_mean([b > 0 ? a / b : NaN for (a, b) in zip(vol, shr)], lam, mo)
        trn[.!amsk] .= NaN
        @test ew_same(descriptor(EWShareTurnover(; half_life = 5), rd), trn)

        amt = px .* vol
        ami = ew_hand_mean([b > 0 ? a / b : NaN for (a, b) in zip(abs.(X), amt)], lam, mo)
        ami[.!amsk] .= NaN
        @test ew_same(descriptor(EWAmihudIlliquidity(; half_life = 5), rd), ami)

        V = ew_hand_mean([v > 0 ? v : NaN for v in vol], lam, mo)
        dtc = [b > 0 ? a / b : NaN for (a, b) in zip(si, V)]
        dtc[.!amsk] .= NaN
        @test ew_same(descriptor(DaysToCover(; half_life = 5), rd), dtc)

        @test ew_same(descriptor(EWVolatility(; half_life = 5), rd),
                      ew_hand_volatility(X, amsk, lam, mo); rtol = 1e-10)
        @test ew_same(descriptor(EWDownsideVolatility(; half_life = 5), rd),
                      ew_hand_volatility(X, amsk, lam, mo; mar = 0.0); rtol = 1e-10)
    end
    @testset "An asset view of the carrier gives the same Descriptor as a slice of the whole" begin
        idx = [2, 5, 9]
        rdv = PortfolioOptimisers.port_opt_view(rd, idx)
        for (name, de) in des
            @test ew_same(descriptor(de, rdv), descriptor(de, rd)[:, idx]; rtol = 1e-10)
        end
    end
    #=
    The oracle was run on this exact panel, all thirteen cases of the two
    files. Every one agreed: no cell differed in whether it is `NaN`, and the largest relative
    difference over the finite cells was 4.3e-15, which is floating point noise. Four cases
    agreed to the last bit.

    Two rows of two cases are stored here as literals, so a regression is caught without the
    oracle. Each carries the skip, the exponentiate flag and a non-zero minimum acceptable
    return, which the hand recursions above exercise least. Asset two is delisted before the
    last observation, so its literal is `NaN`.
    =#
    @testset "Two cases against numbers the oracle printed" begin
        mom_250 = [-0.0062617311055179368, -0.0017340648519905755, -0.0014142152400465399,
                   -0.0066824380942661545, -0.003862538174442892, -0.0059118968616883802,
                   -0.0011177723894685181, -0.0030445356884902497, -0.0018186579722233619,
                   -0.0037452755978652427, -0.0025812232095826274, -0.0011207232842962492]
        mom_300 = [0.0026801832529124402, NaN, 0.0013041330563512584,
                   -0.00034676038877819767, 0.0021258058284273546, 0.0066697029825081421,
                   0.0001141682515419082, 0.0038148303524405849, -0.0035315191667812159,
                   0.0050263759405964302, -0.0050916758636027565, 0.00049559608707931669]
        D = descriptor(EWMomentum(; half_life = 5, skip = 3, exponentiate = true), rd)
        @test ew_same(reshape(view(D, 250, :), 1, :), reshape(mom_250, 1, :); rtol = 1e-12)
        @test ew_same(reshape(view(D, 300, :), 1, :), reshape(mom_300, 1, :); rtol = 1e-12)

        dvol_250 = [0.014740415979782184, 0.010935460514462909, 0.0078246811492014238,
                    0.011856636498505527, 0.011270136212473032, 0.016393160826384069,
                    0.0088465900408822348, 0.015734551356596065, 0.0082358432719374357,
                    0.013898674790251199, 0.0068745744622555538, 0.0088197316727841851]
        dvol_300 = [0.012527657490412247, NaN, 0.0074375526678808193, 0.010477183167212674,
                    0.0098730045114479772, 0.015458198225516948, 0.0070970854066652106,
                    0.014025551244518937, 0.0089478645922382478, 0.013239202598403822,
                    0.0069770549263295598, 0.008116416489161369]
        V = descriptor(EWDownsideVolatility(; half_life = 40, mar = 0.001), rd)
        @test ew_same(reshape(view(V, 250, :), 1, :), reshape(dvol_250, 1, :); rtol = 1e-12)
        @test ew_same(reshape(view(V, 300, :), 1, :), reshape(dvol_300, 1, :); rtol = 1e-12)
    end
end

@testset "The volatility Descriptors derive their number type from the returns" begin
    Xi = [1 2; -1 0; 1 1; 0 -2; 2 1]
    Xf = Float64.(Xi)
    rdi = ew_hand_panel(["market_cap" => ones(size(Xi))], Xi)
    rdf = ew_hand_panel(["market_cap" => ones(size(Xi))], Xf)
    # The default estimator built its state in the type of the returns, so an integer panel
    # raised an `InexactError` at the first `NaN` of the warm-up.
    for de in (EWVolatility(; half_life = 2), EWDownsideVolatility(; half_life = 2),
               EWResidualVolatility(; half_life = 2, beta_half_life = 2),
               EWResidualDownsideVolatility(; half_life = 2, beta_half_life = 2))
        @test isequal(descriptor(de, rdi), descriptor(de, rdf))
    end
    rd32 = ew_hand_panel(["market_cap" => ones(size(Xi))], Float32.(Xi) ./ 100)
    @test eltype(descriptor(EWVolatility(; half_life = 2), rd32)) == Float32
end

@testset "The default estimator states its recursion" begin
    # The divisor 1 - λ^n is the total weight of n observations, so a constant input is its own
    # variance from the first observation. A gap holds the state, so it holds that value too.
    Y = fill(0.03, 6, 2)
    Y[3, 2] = NaN
    V = PortfolioOptimisers.variance_series(PortfolioOptimisers.ew_variance_estimator(3.0,
                                                                                      1.0),
                                            Y)
    @test all(isapprox(0.03^2; rtol = 1e-12), V)
    # Each refusal names the argument the caller passed.
    @test_throws r"half_life" PortfolioOptimisers.ew_variance_estimator(0.0)
    @test_throws r"warm_up" PortfolioOptimisers.ew_variance_estimator(2.0, NaN)
    @test_throws DomainError EWVolatility(; half_life = -1)
    @test_throws DomainError EWDownsideVolatility(; half_life = Inf)
end

@testset "The mean Descriptors derive their number type from the returns" begin
    Xi = [0 1; 1 0; 2 1; 0 0; 1 2]
    Xf = Float64.(Xi)
    vol = [10.0 20.0; 30.0 40.0; 50.0 60.0; 70.0 80.0; 90.0 100.0]
    rdi = ew_hand_panel(["adj_volume" => copy(vol)], Xi)
    rdf = ew_hand_panel(["adj_volume" => copy(vol)], Xf)
    # `EWMean` built its delayed input, and `ew_mean_series` its series, in the type of the
    # returns, so an integer panel raised an `InexactError` at the first `NaN`.
    for de in
        (EWMean(; decay = 0.5, min_obs = 1), EWMean(; decay = 0.5, min_obs = 2, skip = 1),
         EWVolumeRatio(; num = nothing, den = "adj_volume", decay = 0.5, min_obs = 1))
        @test isequal(descriptor(de, rdi), descriptor(de, rdf))
    end
    @test isequal(PortfolioOptimisers.ew_mean_series(Xi, 0.5, 1),
                  PortfolioOptimisers.ew_mean_series(Xf, 0.5, 1))
    rd32 = ew_hand_panel(["adj_volume" => copy(vol)], Float32.(Xi) ./ 100)
    @test eltype(descriptor(EWMean(; decay = 0.5, min_obs = 1), rd32)) == Float32
end

#=
The oracle ran the four named Descriptors on this panel at `half_life = 3`,
and every cell agreed to the last bit, `NaN` pattern included. The panel carries what the
synthetic panel above carries least: a gap in every Panel Field, a zero volume, a missing share
count, a missing short interest, and an asset that is inactive for two rows. The literals below
are what the oracle printed.
=#
@testset "The four named Descriptors agree with the oracle on a panel with gaps" begin
    X = [0.01 0.02 -0.01; -0.02 NaN 0.03; 0.03 0.01 NaN; NaN -0.01 NaN; 0.00 0.02 0.01;
         0.01 -0.03 0.02; -0.01 0.01 -0.02; 0.02 NaN 0.01]
    vol = [10.0 20.0 30.0; 12.0 22.0 31.0; 14.0 0.0 NaN; 16.0 26.0 NaN; 0.0 28.0 34.0;
           20.0 NaN 35.0; 22.0 32.0 36.0; 24.0 34.0 37.0]
    px = [5.0 6.0 7.0; 5.1 6.1 7.1; 5.2 6.2 NaN; 5.3 6.3 NaN; 5.4 6.4 7.4; 5.5 6.5 7.5;
          5.6 6.6 7.6; 5.7 6.7 7.7]
    shr = [100.0 200.0 300.0; 100.0 200.0 300.0; 100.0 200.0 NaN; 100.0 200.0 NaN;
           100.0 200.0 300.0; 100.0 200.0 300.0; NaN 200.0 300.0; 100.0 200.0 300.0]
    si = [1.0 2.0 3.0; 1.5 2.5 3.5; 2.0 NaN NaN; 2.5 3.5 NaN; 3.0 4.0 5.0; 3.5 4.5 5.5;
          4.0 5.0 6.0; 4.5 5.5 6.5]
    amsk = trues(size(X))
    amsk[3:4, 3] .= false
    rd = ew_hand_panel(["adj_volume" => vol, "adj_close" => px,
                        "adj_shares_outstanding" => shr, "short_interest" => si], X;
                       amsk = amsk)
    n4 = fill(NaN, 4, 3)
    mom = vcat(n4,
               [0.004083124258934984 NaN NaN;
                0.004083124258934984 0.002129448022702344 NaN;
                0.0032407778719751347 0.005775415611009435 0.00558655940754667;
                0.004624955122877056 -0.0016997680748015746 0.008519326735545996])
    mom_exp = [NaN NaN NaN; NaN NaN NaN; 0.004091471567958498 NaN NaN;
               0.004091471567958498 0.0021317169073478514 NaN;
               0.0032460348699697933 0.0057921254770681675 0.005602193330274423;
               0.004635666735030029 -0.0016983242871980682 0.008555719473355195;
               0.0015987269168514903 0.0007038888198189767 0.002597353510921931;
               0.005367522558516728 0.0007038888198189767 0.004120060455043248]
    trn = [NaN NaN NaN; NaN NaN NaN; 0.06152677898136927 0.0310074526087766 NaN;
           0.08184175268216429 0.05142956306708006 NaN;
           0.06495784215129441 0.0697015976196999 0.05329645965700086;
           0.09281696828545455 0.0697015976196999 0.06636970003137363;
           0.09281696828545455 0.08833011053523193 0.0774336027062162;
           0.12318075031223083 0.1051785657747503 0.08690269299206196]
    ami = [NaN NaN NaN; NaN NaN NaN; 0.00016451542876188153 NaN NaN;
           0.00016451542876188153 NaN NaN;
           0.00016451542876188153 5.468091941871078e-5 3.6706135970056026e-5;
           0.00014933047997861418 5.468091941871078e-5 4.485173458939356e-5;
           0.00013526876768109943 5.3168242023617174e-5 5.0679216242495956e-5;
           0.0001375236338728997 5.3168242023617174e-5 4.7465239367207845e-5]
    dtc = [NaN NaN NaN; NaN NaN NaN; 0.32506171021980756 NaN NaN;
           0.30546755391576846 0.30263002317661414 NaN;
           0.36656106469892213 0.2674556324606383 0.31271620617819745;
           0.3295118216932188 0.3008875865182181 0.27623046849190186;
           0.30842551111616473 0.2706803719462379 0.2582857997177295;
           0.29518328704247854 0.25374393090195446 0.24932100399519014]
    @test ew_same(descriptor(EWMomentum(; half_life = 3, skip = 2), rd), mom)
    @test ew_same(descriptor(EWMomentum(; half_life = 3, skip = 0, exponentiate = true),
                             rd), mom_exp)
    @test ew_same(descriptor(EWShareTurnover(; half_life = 3), rd), trn)
    @test ew_same(descriptor(EWAmihudIlliquidity(; half_life = 3), rd), ami)
    @test ew_same(descriptor(DaysToCover(; half_life = 3), rd), dtc)
end

@testset "A named mean Descriptor refuses a half-life that is not positive and finite" begin
    @test_throws r"half_life" EWMomentum(; half_life = 0)
    @test_throws r"half_life" EWShareTurnover(; half_life = Inf)
    @test_throws r"half_life" EWAmihudIlliquidity(; half_life = -1)
    @test_throws r"half_life" DaysToCover(; half_life = NaN)
end

#=
Check `src/05_Moments/32_CrossSectionalFactorModel/04_FactorExposures/06_EWBetaDescriptors_a.jl` and `_b.jl`, the residual half of
`05_EWVolatilityDescriptors.jl`, and the market-return builder and beta recursion they share
in `01_Base_Descriptor.jl`, against the contract their docstrings state and against the
oracle. Issue #719, map #643.

FIVE CONVENTIONS SHAPE THE PROBES, and the first three are `test_08s_descriptors.jl`'s.

1. AN INACTIVE CELL IS `NaN`, whatever the Panel Fields hold.

2. A GAP HOLDS THE RECURSION. An asset that is listed but carries no return at an
   observation neither advances its state nor resets it, and the observation does not count
   toward its warm-up.

3. AN INACTIVE CELL IS ALSO A GAP. The oracle's own container refuses a
   return outside the active mask, so no member ever advances a state there. The library
   masks the returns through `ew_active_returns` before the recursion starts, which is the
   same rule stated on this side.

4. THE MARKET STATE IS ONE STATE FOR THE WHOLE PANEL, and it advances at every observation
   whatever any single asset does. Only [`EWMacroSensitivity`](@ref) freezes it, and only
   where the reference return is not finite, because a partial beta needs both series.

5. ONLY THE RESIDUAL VOLATILITY RESETS. Its beta and its variance both restart where an
   asset turns inactive. The three beta Descriptors freeze instead, so an asset that
   re-enters the universe resumes from the state it left.

THE STORED CASES ARE THE ORACLE'S OWN OUTPUT. `assets/EWMarketBeta.csv.gz`
is `EWMarketBeta(half_life = 5, group = "industry", min_group_size = 2,
bounds = (0.1, 0.9))`, and `assets/EWMacroSensitivity.csv.gz` is
`EWMacroSensitivity(half_life = 5, agg_obs = 3)`. Both were produced on the panel the last
testset rebuilds. They were chosen for the two paths the hand recursions below exercise
least: the James-Stein shrinkage under a clamped weight, and the bivariate partial beta on
an aggregated clock. All thirteen cases of the diff agreed with no difference in the `NaN`
pattern and a worst relative difference of 2.5e-13 over the finite cells.
=#

# A small hand panel carrying a market capitalisation beside the returns, and optionally an
# industry classification. Every numeric field takes a forward fill, so each earns an
# observed-mask column and a raw `NaN` reads back as `NaN` rather than as the fill value.
function ewb_hand_panel(X::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real};
                        amsk::AbstractMatrix{Bool} = trues(size(X)...),
                        emsk::AbstractMatrix{Bool} = amsk,
                        industry::Union{Nothing, <:AbstractMatrix{<:AbstractString}} = nothing)
    inputs = Any[NumericPanelInput(; name = "market_cap", vals = W,
                                   alg = ForwardPanelFill(; val = 0.0))]
    if !isnothing(industry)
        push!(inputs,
              CategoricalPanelInput(; name = "industry", vals = industry,
                                    levels = sort(unique(vec(industry)))))
    end
    res = asset_panel(identity.(inputs); amsk = amsk, emsk = emsk)
    return ReturnsResult(; nx = ["A" * string(i) for i in 1:size(X, 2)], X = X, pnl = res)
end

# Two matrices agree when they hold `NaN` in the same cells and are close everywhere else.
function ewb_same(A::AbstractMatrix, B::AbstractMatrix; atol::Real = 0, rtol::Real = 1e-12)
    if !(size(A) == size(B))
        return false
    end
    return all(CartesianIndices(A)) do k
        a, b = A[k], B[k]
        return isnan(a) ? isnan(b) : (!isnan(b) && isapprox(a, b; atol = atol, rtol = rtol))
    end
end

# The capitalisation-weighted market return, written out by hand. Only an estimable pair
# whose return and whose weight are both finite enters.
function ewb_hand_market(X::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real},
                         emsk::AbstractMatrix{Bool})
    T, N = size(X)
    rm = zeros(T)
    for t in 1:T
        s, w = 0.0, 0.0
        for i in 1:N
            if emsk[t, i] && isfinite(X[t, i]) && isfinite(W[t, i])
                s += W[t, i] * X[t, i]
                w += W[t, i]
            end
        end
        rm[t] = s / w
    end
    return rm
end

# The exponentially weighted market beta, written out by hand. Deviations are taken from the
# means of the previous step. A cell outside the active mask, or whose return is not finite,
# advances nothing; the beta of that cell is the one the asset last carried. `reset` says
# whether an asset that turns inactive restarts.
function ewb_hand_beta(X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                       amsk::AbstractMatrix{Bool}, decay::Real, min_obs::Integer,
                       min_val::Real; reset::Bool = false)
    T, N = size(X)
    B = fill(NaN, T, N)
    b, mu, cv, n, act = fill(NaN, N), zeros(N), zeros(N), zeros(Int, N), trues(N)
    mu_m, var_m = 0.0, 0.0
    for t in 1:T
        if reset
            for i in 1:N
                if act[i] && !amsk[t, i]
                    mu[i], cv[i], n[i] = 0.0, 0.0, 0
                end
                act[i] = amsk[t, i]
            end
        end
        dm = rm[t] - mu_m
        mu_m = decay * mu_m + (1 - decay) * rm[t]
        var_m = decay * var_m + (1 - decay) * dm * dm
        for i in 1:N
            if amsk[t, i] && isfinite(X[t, i])
                d = X[t, i] - mu[i]
                mu[i] = decay * mu[i] + (1 - decay) * X[t, i]
                cv[i] = decay * cv[i] + (1 - decay) * d * dm
                n[i] += 1
                if t >= min_obs && n[i] >= min_obs
                    b[i] = cv[i] / (var_m + min_val)
                end
            end
            B[t, i] = b[i]
        end
    end
    return B
end

# The exponentially weighted downside beta, written out by hand. The market's shortfall
# advances at every observation, and a ready asset reads it whether or not its own return is
# finite at that observation.
function ewb_hand_downside(X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                           amsk::AbstractMatrix{Bool}, decay::Real, min_obs::Integer,
                           mar::Real, min_val::Real)
    T, N = size(X)
    B = fill(NaN, T, N)
    cd, n, vd = zeros(N), zeros(Int, N), 0.0
    for t in 1:T
        dm = min(rm[t] - mar, 0.0)
        vd = decay * vd + (1 - decay) * dm * dm
        for i in 1:N
            if amsk[t, i] && isfinite(X[t, i])
                cd[i] = decay * cd[i] + (1 - decay) * min(X[t, i] - mar, 0.0) * dm
                n[i] += 1
            end
            if t >= min_obs && n[i] >= min_obs
                B[t, i] = cd[i] / (vd + min_val)
            end
        end
    end
    return B
end

# The exponentially weighted partial beta on a reference series, written out by hand through
# the Frisch-Waugh closed form. A reference return that is not finite freezes the whole
# state, market included.
function ewb_hand_macro(X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                        rf::AbstractVector{<:Real}, amsk::AbstractMatrix{Bool}, decay::Real,
                        min_obs::Integer, min_val::Real)
    T, N = size(X)
    B = fill(NaN, T, N)
    b, mu, cam, caf, n = fill(NaN, N), zeros(N), zeros(N), zeros(N), zeros(Int, N)
    mu_m, mu_f, var_m, var_f, cov_mf, c = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    for t in 1:T
        if isfinite(rf[t])
            c += 1
            dm, df = rm[t] - mu_m, rf[t] - mu_f
            mu_m = decay * mu_m + (1 - decay) * rm[t]
            mu_f = decay * mu_f + (1 - decay) * rf[t]
            var_m = decay * var_m + (1 - decay) * dm * dm
            var_f = decay * var_f + (1 - decay) * df * df
            cov_mf = decay * cov_mf + (1 - decay) * dm * df
            vm = var_m + min_val
            vf = var_f - cov_mf * cov_mf / vm + min_val
            for i in 1:N
                if amsk[t, i] && isfinite(X[t, i])
                    d = X[t, i] - mu[i]
                    mu[i] = decay * mu[i] + (1 - decay) * X[t, i]
                    cam[i] = decay * cam[i] + (1 - decay) * d * dm
                    caf[i] = decay * caf[i] + (1 - decay) * d * df
                    n[i] += 1
                    if c >= min_obs && n[i] >= min_obs
                        b[i] = (caf[i] - cam[i] * cov_mf / vm) / vf
                    end
                end
            end
        end
        B[t, :] = b
    end
    return B
end

# The exponentially weighted volatility of the market-model residual, written out by hand:
# uncentred, bias corrected by `1 / (1 - λ^n)`, and reset where the asset turns inactive.
function ewb_hand_residual(X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                           B::AbstractMatrix{<:Real}, amsk::AbstractMatrix{Bool},
                           decay::Real, min_obs::Integer; mar = nothing)
    T, N = size(X)
    D = fill(NaN, T, N)
    v, n, act = zeros(N), zeros(Int, N), trues(N)
    for t in 1:T, i in 1:N
        if act[i] && !amsk[t, i]
            v[i], n[i] = 0.0, 0
        end
        act[i] = amsk[t, i]
        if amsk[t, i] && isfinite(X[t, i])
            e = X[t, i] - B[t, i] * rm[t]
            c = isnothing(mar) ? e : min(e - mar, 0.0)
            v[i] = decay * v[i] + (1 - decay) * c * c
            n[i] += 1
        end
        if n[i] >= min_obs
            D[t, i] = sqrt(v[i] / (1 - decay^n[i]))
        end
    end
    return D
end

# The James-Stein shrinkage of one cross-section, written out from its closed form. The
# estimation set is every asset whose beta is not `NaN`, whose group is set and whose weight
# is finite and strictly positive. A group below `mgs` takes the prior of the whole set.
function ewb_hand_shrink(b::AbstractVector{<:Real}, bev::AbstractVector{<:Real},
                         L::AbstractVector{<:Integer}, w::AbstractVector{<:Real},
                         mgs::Integer, bounds::Tuple{<:Real, <:Real}, min_val::Real)
    ix = [i
          for i in eachindex(b)
          if !isnan(b[i]) &&
                 L[i] != PortfolioOptimisers.CS_MISSING_GROUP &&
                 isfinite(w[i]) &&
                 w[i] > 0]
    function prior(jx)
        W = sum(w[j] for j in jx)
        mu = sum(w[j] * b[j] for j in jx) / W
        tau2 = sum(w[j] * (b[j] - mu)^2 for j in jx) / W -
               sum(bev[j] for j in jx) / length(jx)
        return mu, max(tau2, 0.0)
    end
    s = copy(b)
    for i in ix
        gx = [j for j in ix if L[j] == L[i]]
        mu, tau2 = length(gx) < mgs ? prior(ix) : prior(gx)
        q = clamp(tau2 / (tau2 + bev[i] + min_val), bounds...)
        s[i] = q * b[i] + (1 - q) * mu
    end
    return s
end

@testset "EW beta descriptors: the market return" begin
    X = [0.10 0.20 -0.05
         -0.10 0.00 0.03
         0.05 0.05 0.01
         0.02 -0.03 0.04]
    W = [1.0 3.0 2.0
         2.0 2.0 2.0
         5.0 1.0 4.0
         1.0 1.0 1.0]
    emsk = [true true false
            true true true
            false true true
            true true true]
    rd = ewb_hand_panel(X, W; amsk = trues(4, 3), emsk = emsk)
    rm = PortfolioOptimisers.market_return_series(rd, "market_cap")
    @testset "It is the capitalisation-weighted mean over the estimation universe" begin
        @test rm ≈ ewb_hand_market(X, W, emsk)
        @test length(rm) == 4
        @test rm[1] ≈ (1.0 * 0.10 + 3.0 * 0.20) / 4.0
        @test rm[3] ≈ (1.0 * 0.05 + 4.0 * 0.01) / 5.0
    end
    @testset "A missing return and a missing weight both leave the estimate" begin
        Xn = copy(X)
        Xn[2, 1] = NaN
        rdn = ewb_hand_panel(Xn, W; amsk = trues(4, 3), emsk = emsk)
        @test PortfolioOptimisers.market_return_series(rdn, "market_cap")[2] ≈
              (2.0 * 0.00 + 2.0 * 0.03) / 4.0
        Wn = copy(W)
        Wn[2, 3] = NaN
        rdw = ewb_hand_panel(Xn, Wn; amsk = trues(4, 3), emsk = emsk)
        @test PortfolioOptimisers.market_return_series(rdw, "market_cap")[2] ≈ 0.0
    end
    @testset "An observation with no estimable weight raises" begin
        emz = copy(emsk)
        emz[3, :] .= false
        rdz = ewb_hand_panel(X, W; amsk = trues(4, 3), emsk = emz)
        @test_throws ArgumentError PortfolioOptimisers.market_return_series(rdz,
                                                                            "market_cap")
        @test_throws ArgumentError descriptor(EWMarketBeta(; half_life = 1), rdz)
    end
    @testset "A carrier with no Asset Panel raises" begin
        rdp = ReturnsResult(; nx = ["A1", "A2", "A3"], X = X)
        @test_throws PortfolioOptimisers.IsNothingError PortfolioOptimisers.market_return_series(rdp,
                                                                                                 "market_cap")
    end
end

@testset "EW beta descriptors: EWBeta against its hand recursion" begin
    X = [0.10 0.20 -0.05
         -0.10 0.00 0.03
         0.05 NaN 0.01
         0.02 -0.03 0.04
         -0.04 0.06 -0.02
         0.03 0.01 0.05]
    W = [1.0 3.0 2.0
         2.0 2.0 2.0
         5.0 1.0 4.0
         1.0 1.0 1.0
         2.0 3.0 1.0
         1.0 2.0 3.0]
    amsk = trues(6, 3)
    amsk[1, 3] = false
    amsk[5, 1] = false
    emsk = copy(amsk)
    rd = ewb_hand_panel(X, W; amsk = amsk, emsk = emsk)
    rm = ewb_hand_market(X, W, emsk)
    Xm = ifelse.(amsk, X, NaN)
    decay, min_obs, min_val = exp2(-inv(2.0)), 2, 1e-12
    @testset "The beta is the hand recursion, masked by the active mask" begin
        D = descriptor(EWMarketBeta(; half_life = 2), rd)
        B = ewb_hand_beta(Xm, rm, amsk, decay, min_obs, min_val)
        @test ewb_same(D, ifelse.(amsk, B, NaN))
        @test all(k -> !amsk[k] ? isnan(D[k]) : true, CartesianIndices(D))
    end
    @testset "A gap holds the beta rather than resetting it" begin
        D = descriptor(EWMarketBeta(; half_life = 2), rd)
        # Asset 2 has no return at observation 3, so it carries observation 2's beta there.
        @test D[3, 2] == D[2, 2]
        # Asset 1 leaves the universe at observation 5 and comes back at observation 6, so
        # its beta at observation 6 continues the state it left rather than restarting.
        @test isnan(D[5, 1])
        @test isfinite(D[6, 1])
    end
    @testset "The warm-up answers NaN before an asset has min_obs of its own" begin
        D = descriptor(EWMarketBeta(; half_life = 4), rd)
        @test all(isnan, D[1:3, :])
        @test isfinite(D[4, 1])
    end
    @testset "The recursion reads the decay and the warm-up, not the half-life" begin
        a = descriptor(EWBeta(; decay = decay, min_obs = min_obs), rd)
        b = descriptor(EWMarketBeta(; half_life = 2), rd)
        @test isequal(a, b)
        c = descriptor(EWBeta(; decay = decay, min_obs = 4), rd)
        @test all(isnan, c[1:3, :])
    end
end

@testset "EW beta descriptors: the aggregated clock" begin
    X = [0.10 0.20
         -0.10 0.00
         0.05 0.03
         0.02 -0.03
         -0.04 0.06
         0.03 0.01
         0.04 -0.01]
    W = ones(7, 2) .+ 0.5
    rd = ewb_hand_panel(X, W)
    rm = ewb_hand_market(X, W, trues(7, 2))
    D = descriptor(EWMarketBeta(; half_life = 2, agg_obs = 3), rd)
    @testset "It aggregates, then spreads the windows over their observations" begin
        Xa = PortfolioOptimisers.ew_agg_series(X, 3)
        rma = PortfolioOptimisers.ew_agg_vector(rm, 3)
        @test size(Xa) == (2, 2)
        @test Xa[1, 1] ≈ (0.10 - 0.10 + 0.05) / 3
        B = ewb_hand_beta(Xa, rma, trues(2, 2), exp2(-inv(2.0)), 2, 1e-12)
        @test all(isnan, D[1:5, :])
        @test isequal(D[6, :], B[2, :])
        @test isequal(D[7, :], B[2, :])
    end
    @testset "A window whose entries are all missing aggregates to NaN" begin
        A = [1.0 NaN; 3.0 NaN; 5.0 NaN]
        @test PortfolioOptimisers.ew_agg_series(A, 3) ≈ [3.0 NaN] nans = true
    end
    @testset "A tail shorter than one window is dropped" begin
        @test size(PortfolioOptimisers.ew_agg_series(X, 4)) == (1, 2)
        @test PortfolioOptimisers.ew_agg_vector([1.0, 3.0, 5.0], 2) ≈ [2.0]
    end
end

@testset "EW beta descriptors: the James-Stein shrinkage" begin
    @testset "A shrunk beta lies between its raw value and the group mean" begin
        b = [0.6, 1.4, 1.0, 2.0]
        bev = [0.01, 0.01, 0.01, 0.01]
        L = [1, 1, 1, 1]
        w = ones(4)
        s = PortfolioOptimisers.ew_beta_shrink(b, bev, L, w, 1, (0.0, 1.0), 1e-12)
        m = sum(b) / 4
        for i in eachindex(b)
            @test min(b[i], m) <= s[i] <= max(b[i], m)
        end
        @test sum(s) ≈ sum(b)
    end
    @testset "The bounds clamp the weight the raw beta keeps" begin
        # Twenty precise betas and one noisy one. The precise assets want the whole of
        # their raw value and the noisy one wants almost none of it, so a clamp of
        # `(0.25, 0.75)` binds at both ends at once.
        b = vcat(fill(0.6, 10), fill(1.4, 10), 2.0)
        bev = vcat(fill(1e-9, 20), 1.0)
        L = ones(Int, 21)
        w = ones(21)
        m = sum(b) / 21
        s = PortfolioOptimisers.ew_beta_shrink(b, bev, L, w, 1, (0.25, 0.75), 1e-12)
        @test s[1] ≈ 0.75 * b[1] + 0.25 * m
        @test s[21] ≈ 0.25 * b[21] + 0.75 * m
        # With no clamp the precise asset keeps almost the whole of its raw beta and the
        # noisy one moves almost the whole way to the mean, so it is the bounds that set
        # the two weights above and not the data.
        u = PortfolioOptimisers.ew_beta_shrink(b, bev, L, w, 1, (0.0, 1.0), 1e-12)
        @test abs(u[1] - b[1]) < abs(s[1] - b[1])
        @test abs(u[21] - m) < abs(s[21] - m)
    end
    @testset "An asset outside the estimation set keeps its raw beta" begin
        b = [0.6, 1.4, 2.0, 3.0]
        bev = fill(0.01, 4)
        L = [1, 1, PortfolioOptimisers.CS_MISSING_GROUP, 1]
        w = [1.0, 1.0, 1.0, 0.0]
        s = PortfolioOptimisers.ew_beta_shrink(b, bev, L, w, 1, (0.0, 1.0), 1e-12)
        @test s[3] == b[3]
        @test s[4] == b[4]
        @test s[1] != b[1]
        # A NaN beta is outside the set too, and it stays NaN.
        @test isnan(PortfolioOptimisers.ew_beta_shrink([NaN], [0.01], [1], [1.0], 1,
                                                       (0.0, 1.0), 1e-12)[1])
    end
    @testset "A group below min_group_size falls back on the whole cross-section" begin
        b = [0.5, 0.7, 1.5, 1.7, 3.0]
        bev = fill(0.01, 5)
        L = [1, 1, 2, 2, 3]
        w = ones(5)
        small = PortfolioOptimisers.ew_beta_shrink(b, bev, L, w, 3, (0.0, 1.0), 1e-12)
        whole = PortfolioOptimisers.ew_beta_shrink(b, bev, ones(Int, 5), w, 1, (0.0, 1.0),
                                                   1e-12)
        @test small ≈ whole
        grouped = PortfolioOptimisers.ew_beta_shrink(b, bev, L, w, 2, (0.0, 1.0), 1e-12)
        @test !(grouped ≈ whole)
        # The lone asset of group 3 is below min_group_size either way, so it takes the
        # whole cross-section's mean in both runs.
        @test grouped[5] ≈ whole[5]
    end
    @testset "No estimable asset leaves every beta raw" begin
        b = [0.6, 1.4]
        s = PortfolioOptimisers.ew_beta_shrink(b, [0.01, 0.01],
                                               fill(PortfolioOptimisers.CS_MISSING_GROUP,
                                                    2), ones(2), 1, (0.0, 1.0), 1e-12)
        @test s == b
    end
    @testset "The Descriptor shrinks only after the warm-up, and only with a group" begin
        X = [0.10 0.20 -0.05
             -0.10 0.00 0.03
             0.05 0.02 0.01
             0.02 -0.03 0.04
             -0.04 0.06 -0.02]
        W = fill(2.0, 5, 3)
        ind = repeat(["a" "a" "b"], 5, 1)
        rd = ewb_hand_panel(X, W; industry = ind)
        raw = descriptor(EWMarketBeta(; half_life = 2), rd)
        shr = descriptor(EWMarketBeta(; half_life = 2, group = "industry",
                                      min_group_size = 2), rd)
        @test isequal(raw[1, :], shr[1, :])
        @test !isequal(raw[3, :], shr[3, :])
        @test isequal(isnan.(raw), isnan.(shr))
    end
    @testset "The residual variance reads the previous observation's beta" begin
        X = [0.10 0.20; -0.10 0.00; 0.05 0.02; 0.02 -0.03]
        rm = [0.15, -0.05, 0.035, -0.005]
        B, _ = PortfolioOptimisers.ew_beta_series(X, rm, 0.5, 1, 1e-12)
        Vr = PortfolioOptimisers.ew_beta_residual_variance(X, rm, B, 0.5, 1)
        @test all(iszero, Vr[1, :])
        for t in 2:4, i in 1:2
            e = X[t, i] - B[t - 1, i] * rm[t]
            @test Vr[t, i] ≈ 0.5 * Vr[t - 1, i] + 0.5 * e * e
        end
    end
    @testset "The half-life is recovered from the decay exactly enough" begin
        for hl in (1.0, 5.0, 40.0, 60.0, 87.0)
            @test PortfolioOptimisers.decay_half_life(PortfolioOptimisers.half_life_decay(hl)) ≈
                  hl
        end
    end
end

@testset "EW beta descriptors: EWMacroSensitivity" begin
    X = [0.10 0.20 -0.05
         -0.10 0.00 0.03
         0.05 0.02 0.01
         0.02 -0.03 0.04
         -0.04 0.06 -0.02
         0.03 0.01 0.05]
    W = fill(2.0, 6, 3)
    amsk = trues(6, 3)
    amsk[1, 3] = false
    rd = ewb_hand_panel(X, W; amsk = amsk, emsk = amsk)
    rm = ewb_hand_market(X, W, amsk)
    ref = [0.02, -0.01, 0.03, 0.00, NaN, 0.01]
    Xm = ifelse.(amsk, X, NaN)
    @testset "It is the Frisch-Waugh partial beta of the hand recursion" begin
        D = descriptor(EWMacroSensitivity(; half_life = 2), rd; ref = ref)
        B = ewb_hand_macro(Xm, rm, ref, amsk, exp2(-inv(2.0)), 2, 1e-12)
        @test ewb_same(D, ifelse.(amsk, B, NaN))
    end
    @testset "A reference return that is not finite freezes the whole state" begin
        D = descriptor(EWMacroSensitivity(; half_life = 2), rd; ref = ref)
        @test isequal(D[5, :], D[4, :])
        # The freeze reaches the market too: a run whose fifth reference return is present
        # differs at every later observation.
        rf = copy(ref)
        rf[5] = 0.005
        Dn = descriptor(EWMacroSensitivity(; half_life = 2), rd; ref = rf)
        @test !isequal(Dn[6, :], D[6, :])
    end
    @testset "The reference series is required, and it is one entry per observation" begin
        @test_throws PortfolioOptimisers.IsNothingError descriptor(EWMacroSensitivity(;
                                                                                      half_life = 2),
                                                                   rd)
        @test_throws DimensionMismatch descriptor(EWMacroSensitivity(; half_life = 2), rd;
                                                  ref = ref[1:3])
    end
    @testset "The aggregated clock aggregates the reference series too" begin
        D = descriptor(EWMacroSensitivity(; half_life = 2, agg_obs = 3), rd; ref = ref)
        Xa = PortfolioOptimisers.ew_agg_series(Xm, 3)
        rma = PortfolioOptimisers.ew_agg_vector(rm, 3)
        rfa = PortfolioOptimisers.ew_agg_vector(ref, 3)
        @test rfa[2] ≈ (0.00 + 0.01) / 2
        B = ewb_hand_macro(Xa, rma, rfa, trues(2, 3), exp2(-inv(2.0)), 2, 1e-12)
        @test isequal(D[6, :], ifelse.(amsk[6, :], B[2, :], NaN))
        @test all(isnan, D[1:2, :])
    end
end

# Returns driven by the market and by one macro series, with the macro series in the Exogenous
# Series of the returns data under the name "FX" (#1365, ADR 0184).
function ewb_macro_fixture(; T::Integer = 120, N::Integer = 10, fx = nothing)
    rng = StableRNGs.StableRNG(1365)
    m = 0.01 .* randn(rng, T)
    f = isnothing(fx) ? 0.005 .* randn(rng, T) : fx
    bm = collect(range(0.6, 1.4; length = N))
    bf = collect(range(-1.0, 1.0; length = N))
    X = m .* transpose(bm) .+ ifelse.(isfinite.(f), f, 0.0) .* transpose(bf) .+
        0.002 .* randn(rng, T, N)
    W = 1.0 .+ rand(rng, T, N)
    pnl = asset_panel([NumericPanelInput(; name = "market_cap", vals = W)];
                      amsk = trues(T, N), emsk = trues(T, N))
    return ReturnsResult(; nx = ["A" * string(i) for i in 1:N], X = X, ne = ["SPX", "FX"],
                         E = hcat(m, f), pnl = pnl)
end

@testset "EW beta descriptors: EWMacroSensitivity reads the Exogenous Series (#1365)" begin
    PO = PortfolioOptimisers
    rd = ewb_macro_fixture()
    ref = rd.E[:, 2]
    @testset "The named column gives the partial beta of the keyword" begin
        for agg_obs in (1, 3)
            de = EWMacroSensitivity(; series = "FX", half_life = 5, agg_obs = agg_obs)
            dr = EWMacroSensitivity(; half_life = 5, agg_obs = agg_obs)
            @test isequal(descriptor(de, rd), descriptor(dr, rd; ref = ref))
        end
        # The column is found by its name, not by its place.
        rds = ReturnsResult(; nx = rd.nx, X = rd.X, ne = ["FX", "SPX"], E = rd.E[:, [2, 1]],
                            pnl = rd.pnl)
        @test isequal(descriptor(EWMacroSensitivity(; series = "FX", half_life = 5), rds),
                      descriptor(EWMacroSensitivity(; series = "FX", half_life = 5), rd))
    end
    @testset "A gap in the warm-up is accepted, and the recursion skips it" begin
        # The warm-up of a half-life of 5 is more than four observations, and three of its
        # values are missing.
        de = EWMacroSensitivity(; series = "FX", half_life = 5)
        f = copy(ref)
        f[[1, 2, 4]] .= NaN
        rdg = ewb_macro_fixture(; fx = f)
        @test isequal(descriptor(de, rdg),
                      descriptor(EWMacroSensitivity(; half_life = 5), rdg; ref = f))
        # With agg_obs = 3, a window of the warm-up averages its finite values.
        da = EWMacroSensitivity(; series = "FX", half_life = 5, agg_obs = 3)
        @test isequal(descriptor(da, rdg),
                      descriptor(EWMacroSensitivity(; half_life = 5, agg_obs = 3), rdg;
                                 ref = f))
        # With agg_obs = 7, the 17 complete windows end at observation 119, so the recursion
        # does not read observation 120, and a gap there is accepted.
        f = copy(ref)
        f[120] = NaN
        rdt = ewb_macro_fixture(; fx = f)
        dt = EWMacroSensitivity(; series = "FX", half_life = 5, agg_obs = 7)
        @test isequal(descriptor(dt, rdt),
                      descriptor(EWMacroSensitivity(; half_life = 5, agg_obs = 7), rdt;
                                 ref = f))
    end
    @testset "The refusals" begin
        de = EWMacroSensitivity(; series = "FX", half_life = 5)
        # Both sources of the series.
        @test_throws ArgumentError descriptor(de, rd; ref = ref)
        # Neither source.
        @test_throws PO.IsNothingError descriptor(EWMacroSensitivity(; half_life = 5), rd)
        # No Exogenous Series.
        rd0 = ReturnsResult(; nx = rd.nx, X = rd.X, pnl = rd.pnl)
        @test_throws PO.IsNothingError descriptor(de, rd0)
        # A name the Exogenous Series does not hold is named in the message.
        err = try
            descriptor(EWMacroSensitivity(; series = "JPY", half_life = 5), rd)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("\"JPY\"", err.msg)
        # A value that is not finite after the warm-up, at observation 50.
        f = copy(ref)
        f[50] = NaN
        rdn = ewb_macro_fixture(; fx = f)
        err = try
            descriptor(de, rdn)
        catch e
            e
        end
        @test err isa PO.IsNonFiniteError
        @test occursin("\"FX\"", err.msg)
        @test occursin("observation 50", err.msg)
        # With agg_obs = 3, the window of observations 49 to 51 still holds two finite
        # values, and the recursion reads their mean, so the gap is accepted.
        da = EWMacroSensitivity(; series = "FX", half_life = 5, agg_obs = 3)
        @test isequal(descriptor(da, rdn),
                      descriptor(EWMacroSensitivity(; half_life = 5, agg_obs = 3), rdn;
                                 ref = f))
        # A window with no finite value is refused, and the message names the window.
        f3 = copy(ref)
        f3[49:51] .= NaN
        err = try
            descriptor(da, ewb_macro_fixture(; fx = f3))
        catch e
            e
        end
        @test err isa PO.IsNonFiniteError
        @test occursin("observations 49 to 51", err.msg)
        # The keyword keeps the rule of the recursion: the state holds its value there.
        D = descriptor(EWMacroSensitivity(; half_life = 5), rdn; ref = f)
        @test isequal(D[50, :], D[49, :])
        # An infinite value is refused on both paths, in the warm-up too, because it is
        # not a gap.
        fi = copy(ref)
        fi[2] = -Inf
        rdi = ewb_macro_fixture(; fx = fi)
        @test_throws PO.IsNonFiniteError descriptor(de, rdi)
        @test_throws PO.IsNonFiniteError descriptor(EWMacroSensitivity(; half_life = 5),
                                                    rdi; ref = fi)
        # An empty name is refused by the constructor.
        err = try
            EWMacroSensitivity(; series = "")
        catch e
            e
        end
        @test err isa PO.IsEmptyError
        @test occursin("Exogenous Series", err.msg)
    end
    @testset "An observation view reads the rows of E that it keeps" begin
        de = EWMacroSensitivity(; series = "FX", half_life = 5)
        rows, cols = 21:100, [1, 3, 4, 8]
        rdv = PO.port_opt_view(rd, rows, cols)
        @test isequal(descriptor(de, rdv),
                      descriptor(EWMacroSensitivity(; half_life = 5), rdv; ref = ref[rows]))
    end
    @testset "Inside CrossSectionalFactorPrior" begin
        de = EWMacroSensitivity(; series = "FX", half_life = 5)
        xe = CompositeExposure(; descriptors = [de], outlier = nothing, scoring = nothing,
                               family = "macro")
        # The exposure is the Descriptor of a direct call with the same series. A direct call
        # of the member reads the benchmark weights from the panel, where the prior writes them.
        pnl = rd.pnl
        rdb = ReturnsResult(; nx = rd.nx, X = rd.X, ne = rd.ne, E = rd.E,
                            pnl = AssetPanel(;
                                             pf = [pnl.pf...,
                                                   NumericPanelField(;
                                                                     name = "benchmark_weights",
                                                                     vals = ones(size(rd.X)))],
                                             amsk = pnl.amsk, emsk = pnl.emsk))
        L = factor_exposure(xe, rdb)
        @test isequal(L, descriptor(EWMacroSensitivity(; half_life = 5), rd; ref = ref))
        for f in (xe, ObservedExposure(; xe = xe, series = "FX"))
            pe = CrossSectionalFactorPrior(; lambda = 1,
                                           factors = ["market" => ConstantExposure(),
                                                      "fx" => f], minra = 3, bp = 0,
                                           wa = MarketCapWeights(; p = 0))
            pr = prior(pe, rd)
            k = findfirst(==("fx"), pr.rr.nf)
            @test !isnothing(k)
            @test all(isfinite, pr.mu)
            @test all(isfinite, pr.sigma)
            @test pr.rr.M[:, k] ≈ L[end, :]
        end
    end
end

@testset "EW beta descriptors: EWDownsideBeta" begin
    X = [0.10 0.20 -0.05
         -0.10 -0.02 0.03
         0.05 NaN 0.01
         0.02 -0.03 0.04
         -0.04 0.06 -0.02]
    W = [1.0 3.0 2.0
         2.0 2.0 2.0
         5.0 1.0 4.0
         1.0 1.0 1.0
         2.0 3.0 1.0]
    amsk = trues(5, 3)
    amsk[1, 3] = false
    rd = ewb_hand_panel(X, W; amsk = amsk, emsk = amsk)
    rm = ewb_hand_market(X, W, amsk)
    Xm = ifelse.(amsk, X, NaN)
    @testset "It is the lower partial co-moment of the hand recursion" begin
        for mar in (0.0, 0.001, -0.01)
            D = descriptor(EWDownsideBeta(; half_life = 2, mar = mar), rd)
            B = ewb_hand_downside(Xm, rm, amsk, exp2(-inv(2.0)), 2, mar, 1e-12)
            @test ewb_same(D, ifelse.(amsk, B, NaN))
        end
    end
    @testset "A ready asset reads the market's shortfall through its own gap" begin
        D = descriptor(EWDownsideBeta(; half_life = 2), rd)
        # Asset 2 has no return at observation 3, and the market's downside variance still
        # advances there, so its beta moves even though its co-moment holds its value.
        @test isfinite(D[3, 2])
        @test D[3, 2] != D[2, 2]
    end
    @testset "A market that never falls short leaves every beta at zero" begin
        Xp = fill(0.01, 4, 2)
        rdp = ewb_hand_panel(Xp, ones(4, 2))
        D = descriptor(EWDownsideBeta(; half_life = 2, mar = 0.0), rdp)
        @test all(iszero, D[2:end, :])
    end
end

@testset "EW beta descriptors: EWResidualVolatility" begin
    X = [0.10 0.20 -0.05
         -0.10 0.00 0.03
         0.05 NaN 0.01
         0.02 -0.03 0.04
         -0.04 0.06 -0.02
         0.03 0.01 0.05
         0.01 -0.02 0.02]
    W = fill(2.0, 7, 3)
    amsk = trues(7, 3)
    amsk[4, 1] = false
    rd = ewb_hand_panel(X, W; amsk = amsk, emsk = amsk)
    rm = ewb_hand_market(X, W, amsk)
    Xm = ifelse.(amsk, X, NaN)
    bdecay = exp2(-inv(3.0))
    vdecay = exp2(-inv(2.0))
    B = ewb_hand_beta(Xm, rm, amsk, bdecay, 1, 1e-12; reset = true)
    @testset "It is the volatility of the hand residuals" begin
        D = descriptor(EWResidualVolatility(; half_life = 2, beta_half_life = 3), rd)
        E = ewb_hand_residual(Xm, rm, B, amsk, vdecay, 3)
        @test ewb_same(D, ifelse.(amsk, E, NaN); rtol = 1e-10)
    end
    @testset "The downside form clips every excess above the target" begin
        for mar in (0.0, 0.001)
            D = descriptor(EWResidualDownsideVolatility(; half_life = 2, beta_half_life = 3,
                                                        mar = mar), rd)
            E = ewb_hand_residual(Xm, rm, B, amsk, vdecay, 3; mar = mar)
            @test ewb_same(D, ifelse.(amsk, E, NaN); rtol = 1e-10)
        end
    end
    @testset "An asset that turns inactive restarts both recursions" begin
        D = descriptor(EWResidualVolatility(; half_life = 2, beta_half_life = 3), rd)
        @test isnan(D[4, 1])
        # Asset 1 restarts at observation 5, so it needs three more valid returns before it
        # answers again, and it answers at observation 7.
        @test isnan(D[5, 1])
        @test isnan(D[6, 1])
        @test isfinite(D[7, 1])
    end
    @testset "The warm-up is the longer of the two half-lives" begin
        a = EWResidualVolatility(; half_life = 2, beta_half_life = 9)
        b = EWResidualVolatility(; half_life = 9, beta_half_life = 2)
        @test a.ce.min_obs == 9
        @test b.ce.min_obs == 9
        @test a.ce.decay ≈ exp2(-inv(2.0))
        @test b.ce.decay ≈ exp2(-inv(9.0))
        @test PortfolioOptimisers.ew_variance_estimator(5.0).min_obs == 5
        @test PortfolioOptimisers.ew_variance_estimator(5.0, 8.0).min_obs == 8
    end
    @testset "A residual exists only where the asset is active and its return is finite" begin
        E = PortfolioOptimisers.ew_residual_returns([0.1 0.2; NaN 0.0], [0.1, -0.05],
                                                    [1.0 2.0; 1.0 2.0],
                                                    [true false; true true])
        @test isnan(E[1, 2])
        @test isnan(E[2, 1])
        @test E[1, 1] ≈ 0.0
        @test E[2, 2] ≈ 0.1
        # Integer arguments hold the `NaN` of a missing cell in their floating point type.
        Ei = PortfolioOptimisers.ew_residual_returns([1 2; 3 4], [1, 2], [1 1; 2 2],
                                                     [true false; true true])
        @test isequal(Ei, [0.0 NaN; -1.0 0.0])
    end
    @testset "At the first valid observation the residual is zero up to the floor" begin
        # The beta includes observation 1, so β = (1 - λ) x r_m / ((1 - λ) r_m² + min_val) and
        # the residual is x min_val / ((1 - λ) r_m² + min_val). A warm-up of one observation
        # makes the Descriptor its absolute value there. A large floor keeps the subtraction
        # well conditioned.
        mv = 1e-4
        de = EWResidualVolatility(; half_life = 2, beta_half_life = 3, min_val = mv,
                                  ce = PortfolioOptimisers.ew_variance_estimator(2.0, 1.0))
        D = descriptor(de, rd)
        expected = abs.(Xm[1, :] .* mv ./ ((1 - bdecay) * rm[1]^2 + mv))
        @test isapprox(D[1, :], expected; rtol = 1e-8)
    end
    @testset "A refusal names the half-life the caller passed" begin
        @test_throws r"beta_half_life" EWResidualVolatility(; beta_half_life = Inf)
        @test_throws r"beta_half_life" EWResidualDownsideVolatility(; beta_half_life = 0)
        @test_throws r"half_life" EWResidualVolatility(; half_life = -1)
    end
end

@testset "EW beta descriptors: constructors and their refusals" begin
    @testset "Every archetype refuses a decay outside the open unit interval" begin
        for f in
            (d -> EWBeta(; decay = d, min_obs = 2), d -> EWMacroSensitivity(; decay = d),
             d -> EWDownsideBeta(; decay = d), d -> EWResidualVolatility(; beta_decay = d))
            @test_throws DomainError f(0.0)
            @test_throws DomainError f(1.0)
            @test_throws DomainError f(-0.5)
        end
    end
    @testset "Every archetype refuses a warm-up, a group size or a floor below one" begin
        @test_throws DomainError EWBeta(; decay = 0.5, min_obs = 0)
        @test_throws DomainError EWBeta(; decay = 0.5, min_obs = 2, min_group_size = 0)
        @test_throws DomainError EWBeta(; decay = 0.5, min_obs = 2, min_val = 0.0)
        @test_throws DomainError EWMacroSensitivity(; min_obs = 0)
        @test_throws DomainError EWDownsideBeta(; min_obs = 0)
        @test_throws DomainError EWResidualVolatility(; min_val = 0.0)
    end
    @testset "The aggregation period is at least one observation" begin
        @test_throws DomainError EWBeta(; decay = 0.5, min_obs = 2, agg_obs = 0)
        @test_throws DomainError EWMacroSensitivity(; agg_obs = -1)
        @test EWBeta(; decay = 0.5, min_obs = 2, agg_obs = 1).agg_obs == 1
    end
    @testset "The shrinkage bounds are ordered and lie in the unit interval" begin
        for b in ((-0.1, 1.0), (0.0, 1.1), (0.7, 0.3))
            @test_throws DomainError EWBeta(; decay = 0.5, min_obs = 2, bounds = b)
        end
        @test EWBeta(; decay = 0.5, min_obs = 2, bounds = (0.1, 0.9)).bounds == (0.1, 0.9)
    end
    @testset "A minimum acceptable return must be finite" begin
        @test_throws DomainError EWDownsideBeta(; mar = NaN)
        @test_throws DomainError EWResidualVolatility(; mar = Inf)
    end
    @testset "The capitalisation Panel Field is named, and its name is not empty" begin
        @test_throws PortfolioOptimisers.IsEmptyError EWBeta(; decay = 0.5, min_obs = 2,
                                                             mcap = "")
        @test EWMarketBeta(; mcap = "cap").mcap == "cap"
        @test EWDownsideBeta(; mcap = "cap").mcap == "cap"
        @test EWMacroSensitivity(; mcap = "cap").mcap == "cap"
        @test EWResidualVolatility(; mcap = "cap").mcap == "cap"
    end
    @testset "A named constructor fixes the defaults its half-life converts to" begin
        @test EWMarketBeta().decay ≈ exp2(-inv(60.0))
        @test EWMarketBeta().min_obs == 60
        @test EWMacroSensitivity().decay ≈ exp2(-inv(60.0))
        @test EWDownsideBeta().decay ≈ exp2(-inv(60.0))
        @test EWDownsideBeta().mar == 0.0
        @test EWResidualVolatility().ce.decay ≈ exp2(-inv(40.0))
        @test EWResidualVolatility().ce.min_obs == 60
        @test EWResidualVolatility().beta_decay ≈ exp2(-inv(60.0))
        @test isa(EWResidualVolatility().alg, FullMoment)
        @test isa(EWResidualDownsideVolatility().alg, SemiMoment)
        # A value passed for a converted field is used as it stands.
        @test EWMarketBeta(; half_life = 2, decay = 0.9).decay == 0.9
        @test EWMarketBeta(; half_life = 2, min_obs = 7).min_obs == 7
    end
    @testset "Every member is a Descriptor Estimator" begin
        for de in
            (EWMarketBeta(), EWMacroSensitivity(), EWDownsideBeta(), EWResidualVolatility(),
             EWResidualDownsideVolatility())
            @test isa(de, PortfolioOptimisers.AbstractDescriptorEstimator)
        end
    end
end

@testset "EW beta descriptors: the stored oracle cases" begin
    res = synthetic_asset_panel(; n_assets = 12, n_observations = 300, n_industries = 3,
                                rng = StableRNGs.StableRNG(719))
    rd0 = res.rd
    pnl = rd0.pnl
    amsk = Matrix{Bool}(pnl.amsk)
    T, N = size(amsk)
    # The same forty gaps the oracle was driven on. The generator leaves
    # none inside the active mask, and a gap is what holds a recursion.
    Xg = Matrix{Float64}(rd0.X)
    let g = StableRNGs.StableRNG(7192), holes = 0
        while holes < 40
            t = rand(g, 1:T)
            i = rand(g, 1:N)
            if amsk[t, i] && isfinite(Xg[t, i])
                Xg[t, i] = NaN
                holes += 1
            end
        end
    end
    rd = ReturnsResult(; nx = rd0.nx, X = Xg, nf = rd0.nf, F = rd0.F, nb = rd0.nb,
                       B = rd0.B, ts = rd0.ts, iv = rd0.iv, pnl = rd0.pnl)
    ref = 0.01 .* randn(StableRNGs.StableRNG(7191), T)
    ref[7] = NaN
    ref[8] = NaN
    ref[123] = NaN
    @testset "Shape, the inactive fill, and finite-or-NaN everywhere" begin
        des = ("EWMarketBeta" => EWMarketBeta(; half_life = 5),
               "EWDownsideBeta" => EWDownsideBeta(; half_life = 5),
               "EWResidualVolatility" =>
                   EWResidualVolatility(; half_life = 5, beta_half_life = 8),
               "EWResidualDownsideVolatility" =>
                   EWResidualDownsideVolatility(; half_life = 5, beta_half_life = 8))
        for (name, de) in des
            D = descriptor(de, rd)
            @test size(D) == (T, N)
            @test all(k -> amsk[k] || isnan(D[k]), CartesianIndices(D))
            @test all(d -> isfinite(d) || isnan(d), D)
        end
        Dm = descriptor(EWMacroSensitivity(; half_life = 5), rd; ref = ref)
        @test size(Dm) == (T, N)
        @test all(k -> amsk[k] || isnan(Dm[k]), CartesianIndices(Dm))
    end
    @testset "The shrunk beta matches the stored case cell by cell" begin
        D = descriptor(EWMarketBeta(; half_life = 5, group = "industry", min_group_size = 2,
                                    bounds = (0.1, 0.9)), rd)
        E = Matrix(CSV.read(joinpath(@__DIR__, "assets/EWMarketBeta.csv.gz"), DataFrame))
        # rtol = 1e-12 cell by cell; measured maxrel 1.7e-15 (#1380).
        @test parity_compare(D, E; name = "EWMarketBeta").ok
    end
    @testset "The aggregated macro sensitivity matches the stored case cell by cell" begin
        D = descriptor(EWMacroSensitivity(; half_life = 5, agg_obs = 3), rd; ref = ref)
        E = Matrix(CSV.read(joinpath(@__DIR__, "assets/EWMacroSensitivity.csv.gz"),
                            DataFrame))
        # rtol = 1e-12 cell by cell; measured maxrel 3.1e-14 (#1380).
        @test parity_compare(D, E; name = "EWMacroSensitivity").ok
        # The same series named in the Exogenous Series gives the same case (#1365). Its
        # gap at observation 123 falls in a window that holds two finite values.
        rde = ReturnsResult(; nx = rd.nx, X = rd.X, ne = ["FX"], E = reshape(ref, :, 1),
                            pnl = rd.pnl)
        Ds = descriptor(EWMacroSensitivity(; series = "FX", half_life = 5, agg_obs = 3),
                        rde)
        @test parity_compare(Ds, E; name = "EWMacroSensitivity series").ok
    end
end

@testset "EW beta descriptors: the shrinkage is the closed form, written out" begin
    @testset "Unequal weights, a small group, a missing group and a NaN beta" begin
        b = [0.6, 1.4, 1.1, 0.9, 1.8, 0.7, 1.2, NaN]
        bev = [0.02, 0.05, 0.01, 0.03, 0.04, 0.02, 0.06, 0.01]
        L = [1, 1, 1, 2, 2, 3, PortfolioOptimisers.CS_MISSING_GROUP, 1]
        w = [3.0, 1.0, 2.0, 5.0, 1.0, 2.0, 4.0, 1.0]
        for bounds in ((0.0, 1.0), (0.2, 0.7))
            s = PortfolioOptimisers.ew_beta_shrink(b, bev, L, w, 2, bounds, 1e-12)
            @test ewb_same(s[:, :], ewb_hand_shrink(b, bev, L, w, 2, bounds, 1e-12)[:, :];
                           rtol = 1e-14)
        end
        # The asset with no group keeps its raw beta, and the NaN beta stays NaN.
        s = PortfolioOptimisers.ew_beta_shrink(b, bev, L, w, 2, (0.0, 1.0), 1e-12)
        @test s[7] == b[7]
        @test isnan(s[8])
    end
    @testset "The grouped Descriptor on an aggregated clock is the hand pipeline" begin
        X = [0.10 0.20 -0.05 0.04
             -0.10 0.00 0.03 -0.02
             0.05 NaN 0.01 0.06
             0.02 -0.03 0.04 -0.01
             -0.04 0.06 -0.02 0.03
             0.03 0.01 0.05 -0.04
             0.06 -0.02 0.02 0.01
             -0.01 0.04 -0.03 0.05]
        W = [1.0 3.0 2.0 1.5
             2.0 2.0 2.0 1.0
             5.0 1.0 4.0 2.0
             1.0 1.0 1.0 3.0
             2.0 3.0 1.0 1.0
             1.0 2.0 3.0 2.0
             3.0 1.0 2.0 1.0
             2.0 2.0 1.0 4.0]
        ind = repeat(["a" "a" "b" "b"], 8, 1)
        rd = ewb_hand_panel(X, W; industry = ind)
        D = descriptor(EWMarketBeta(; half_life = 2, agg_obs = 2, group = "industry",
                                    min_group_size = 2, bounds = (0.1, 0.9)), rd)
        decay, min_obs, min_val, en = exp2(-inv(2.0)), 2, 1e-12, 4.0
        # Each window is the mean of the finite entries of its two observations.
        rm = ewb_hand_market(X, W, trues(8, 4))
        Xa = [sum(filter(isfinite, X[(2k - 1):(2k), i])) /
              count(isfinite, X[(2k - 1):(2k), i]) for k in 1:4, i in 1:4]
        rma = [(rm[2k - 1] + rm[2k]) / 2 for k in 1:4]
        Ba = ewb_hand_beta(Xa, rma, trues(4, 4), decay, min_obs, min_val)
        # The market variance, and the residual variance against the previous window's beta.
        Vm, Vr = zeros(4), zeros(4, 4)
        let mu = 0.0, v = 0.0, r = zeros(4)
            for k in 1:4
                d = rma[k] - mu
                mu = decay * mu + (1 - decay) * rma[k]
                v = decay * v + (1 - decay) * d * d
                Vm[k] = v
                if k - 1 >= min_obs
                    for i in 1:4
                        e = Xa[k, i] - Ba[k - 1, i] * rma[k]
                        r[i] = decay * r[i] + (1 - decay) * e * e
                    end
                end
                Vr[k, :] = r
            end
        end
        E = fill(NaN, 8, 4)
        s = fill(NaN, 4)
        for t in 1:8
            k = div(t, 2)
            if iseven(t) && k >= min_obs
                s = ewb_hand_shrink(Ba[k, :], Vr[k, :] ./ (en * (Vm[k] + min_val)),
                                    [1, 1, 2, 2], W[t, :], 2, (0.1, 0.9), min_val)
            end
            if k >= min_obs
                E[t, :] = s
            elseif k >= 1
                E[t, :] = Ba[k, :]
            end
        end
        @test ewb_same(D, E; rtol = 1e-12)
        @test all(isfinite, D[4:8, :])
        # The shrinkage moved every beta of the last window.
        @test all(D[8, :] .!= Ba[4, :])
    end
end

@testset "EW beta descriptors: the number type comes from the data" begin
    Xi = [1 2
          -1 0
          1 1
          2 -1
          0 3]
    W = [1.0 2.0
         3.0 4.0
         5.0 6.0
         1.0 1.0
         2.0 1.0]
    amsk = trues(5, 2)
    amsk[3, 2] = false
    rdi = ewb_hand_panel(Xi, W; amsk = amsk)
    rdf = ewb_hand_panel(Float64.(Xi), W; amsk = amsk)
    @testset "Integer returns give the Descriptor of the same returns as floats" begin
        @test isequal(descriptor(EWMarketBeta(; half_life = 1), rdi),
                      descriptor(EWMarketBeta(; half_life = 1), rdf))
        @test isequal(descriptor(EWMarketBeta(; half_life = 1, agg_obs = 2), rdi),
                      descriptor(EWMarketBeta(; half_life = 1, agg_obs = 2), rdf))
        @test isequal(descriptor(EWDownsideBeta(; half_life = 1), rdi),
                      descriptor(EWDownsideBeta(; half_life = 1), rdf))
        @test isequal(descriptor(EWMacroSensitivity(; half_life = 1, agg_obs = 2), rdi;
                                 ref = [1, 0, -1, 2, 1]),
                      descriptor(EWMacroSensitivity(; half_life = 1, agg_obs = 2), rdf;
                                 ref = [1.0, 0.0, -1.0, 2.0, 1.0]))
    end
    @testset "The helpers float an integer and keep every other type" begin
        A = PortfolioOptimisers.ew_agg_series([1 2; 3 4], 2)
        @test A == [2.0 3.0]
        @test eltype(A) == Float64
        R = PortfolioOptimisers.ew_agg_series([1//2 1//3; 1//4 1//5], 2)
        @test R == [3//8 4//15]
        @test eltype(R) == Rational{Int}
        @test eltype(PortfolioOptimisers.ew_agg_series(Float32[1 2; 3 4], 2)) == Float32
        @test isequal(PortfolioOptimisers.ew_beta_expand([1 2; 3 4], 3, 2),
                      [NaN NaN; 1.0 2.0; 1.0 2.0])
        @test PortfolioOptimisers.ew_beta_expand([1//2 1//3; 1//4 1//5], 2, 1) ==
              [1//2 1//3; 1//4 1//5]
        # The kernels run from a new state, as the batch call does.
        function downside(X, rm, mar)
            st = PortfolioOptimisers.ew_downside_beta_state(nothing, X, rm, 0.5)
            return PortfolioOptimisers.ew_downside_beta_series!(st, X, rm, 0.5, 1, mar,
                                                                1e-12).B
        end
        B = downside([1 2; -1 0], [1, -1], 0)
        @test eltype(B) == Float64
        @test B ≈ downside([1.0 2.0; -1.0 0.0], [1.0, -1.0], 0.0) nans = true
        function partial(X, rm, rf)
            st = PortfolioOptimisers.ew_macro_sensitivity_state(X, rm, rf, 0.5)
            return PortfolioOptimisers.ew_macro_sensitivity_series!(st, X, rm, rf, 0.5, 1,
                                                                    1e-12).B
        end
        M = partial([1 2; -1 0; 2 1], [1, -1, 0], [1, 0, 2])
        @test isequal(M,
                      partial([1.0 2.0; -1.0 0.0; 2.0 1.0], [1.0, -1.0, 0.0],
                              [1.0, 0.0, 2.0]))
        Bi, Vi = PortfolioOptimisers.ew_beta_series([1 2; -1 0; 2 1], [1, -1, 0], 0.5, 1,
                                                    1e-12)
        Bf, Vf = PortfolioOptimisers.ew_beta_series([1.0 2.0; -1.0 0.0; 2.0 1.0],
                                                    [1.0, -1.0, 0.0], 0.5, 1, 1e-12)
        @test isequal(Bi, Bf)
        @test isequal(Vi, Vf)
        V = PortfolioOptimisers.ew_beta_residual_variance([1 2; -1 0; 2 1], [1, -1, 0],
                                                          [1 2; 1 1; 1 1], 0.5, 1)
        @test V[2, :] ≈ [0.5 * (-1 - 1 * -1)^2, 0.5 * (0 - 2 * -1)^2]
        @test eltype(V) == Float64
    end
end

# The beta series from its closed form in BigFloat: each mean, variance and covariance is a sum
# over the valid observations of its asset since the last restart, so it shares no loop with
# the recursion it checks. A restart happens where an asset turns inactive, and a cell that
# writes no new beta holds the last one.
function ewb_closed_beta(X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                         amsk::AbstractMatrix{Bool}, decay::Real, min_obs::Integer,
                         min_val::Real)
    lam = BigFloat(decay)
    T, N = size(X)
    r = BigFloat.(rm)
    # `mum[s]` is the market mean after `s - 1` observations, the mean a deviation at `s` uses.
    mum = [(1 - lam) * sum((lam^(s - 1 - u) * r[u] for u in 1:(s - 1)); init = zero(lam))
           for s in 1:T]
    Vm = [(1 - lam) * sum(lam^(t - s) * (r[s] - mum[s])^2 for s in 1:t) for t in 1:T]
    B = fill(BigFloat(NaN), T, N)
    for i in 1:N
        start = 1
        b = BigFloat(NaN)
        for t in 1:T
            if t > 1 && amsk[t - 1, i] && !amsk[t, i]
                start = t
            end
            v = [s for s in start:t if amsk[s, i] && isfinite(X[s, i])]
            k = length(v)
            if k > 0 && last(v) == t && t >= min_obs && k >= min_obs
                x = BigFloat.(X[v, i])
                mu = [(1 - lam) *
                      sum((lam^(j - 1 - l) * x[l] for l in 1:(j - 1)); init = zero(lam))
                      for j in 1:k]
                C = (1 - lam) *
                    sum(lam^(k - j) * (x[j] - mu[j]) * (r[v[j]] - mum[v[j]]) for j in 1:k)
                b = C / (Vm[t] + min_val)
            end
            B[t, i] = b
        end
    end
    return B, Vm
end

@testset "The beta series is the closed form of its recursion (#726)" begin
    # Asset 1 has a gap at observation 3, and asset 2 turns inactive at observation 4. The
    # warm-up of two observations delays the first beta and the first beta after the restart.
    X = [0.10 0.20
         -0.10 0.00
         NaN 0.05
         0.02 -0.03
         -0.04 0.06
         0.03 0.01]
    rm = [0.15, -0.05, 0.035, -0.005, 0.01, 0.02]
    amsk = [true true; true true; true true; true false; true true; true true]
    decay, min_obs, min_val = 0.6, 2, 1e-4
    B, Vm = PortfolioOptimisers.ew_beta_series(X, rm, decay, min_obs, min_val, amsk)
    Bc, Vc = ewb_closed_beta(X, rm, amsk, decay, min_obs, min_val)
    @test ewb_same(B, Float64.(Bc); rtol = 1e-12)
    @test Vm ≈ Float64.(Vc) rtol = 1e-12
    @test all(isnan, B[1, :])
    # A gap holds the beta, and a restart does not clear it: the first valid return after the
    # restart counts one observation, below the warm-up, so the old beta holds one more step.
    @test B[3, 1] == B[2, 1]
    @test B[5, 2] == B[3, 2]
    @test B[6, 2] != B[5, 2]
    # With no mask, no state restarts: the inactive cell is an ordinary finite return.
    B0, _ = PortfolioOptimisers.ew_beta_series(X, rm, decay, min_obs, min_val)
    Bc0, _ = ewb_closed_beta(X, rm, trues(size(X)), decay, min_obs, min_val)
    @test ewb_same(B0, Float64.(Bc0); rtol = 1e-12)
end

@testset "A negative capitalisation enters the market return as it stands (#726)" begin
    X = [0.10 0.20 -0.05; 0.02 -0.03 0.04]
    W = [1.0 -0.5 2.0; 1.0 1.0 1.0]
    rd = ewb_hand_panel(X, W)
    rm = PortfolioOptimisers.market_return_series(rd, "market_cap")
    @test rm[1] ≈ (1.0 * 0.10 - 0.5 * 0.20 + 2.0 * -0.05) / 2.5
    # A total weight of zero divides no mean, even where every pair is finite.
    Wz = [1.0 -1.0 0.0; 1.0 1.0 1.0]
    @test_throws ArgumentError PortfolioOptimisers.market_return_series(ewb_hand_panel(X,
                                                                                       Wz),
                                                                        "market_cap")
end
