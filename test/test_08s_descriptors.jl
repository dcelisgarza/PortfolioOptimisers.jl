#=
Check `src/08_Moments/42_FactorExposures/01_Base_Descriptor.jl`,
`02_PanelFieldDescriptors.jl` and `03_LagDescriptors.jl` against the contract their docstrings
state, and against the reference implementation's own descriptor tests. Issue #717, map #643.

THREE CONVENTIONS SHAPE THE PROBES.

1. A BLANK NEVER REACHES A CARRIER. `asset_panel` resolves every blank to a fill value and
   records the resolution in an observed-mask column, so a Descriptor must read the mask back
   or it mistakes a fill for data. Every probe that plants a `NaN` in a raw field checks that
   the Descriptor is `NaN` there, not the fill.

2. AN INACTIVE CELL IS `NaN`, whatever its Panel Fields hold. The hand panels below carry a
   value on an inactive cell on purpose, so a Descriptor that forgets the active mask fails.

3. A NON-POSITIVE DENOMINATOR IS `NaN`, NEVER AN ERROR. The reference implementation raises on
   some of them and returns `NaN` on others; the port answers `NaN` on every one, and the
   `nonneg` guard is the one refusal it keeps. `GrowthRate` refuses a negative field, as the
   reference does.

The synthetic panel of `test06c_setup.jl` carries every Panel Field the named descriptors
read, so the last testset runs all of them on it and checks each against the field arithmetic
written out by hand from `rd.nz` and `rd.Z`.
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
    return ReturnsResult(; nx = ["A" * string(i) for i in 1:N], X = zeros(T, N), res...)
end

# Read one raw field and its observed mask back off the carrier by column name, so the
# expected value of a probe never goes through the function under test.
function descriptor_raw_field(rd::ReturnsResult, name::AbstractString)
    Z = rd.Z
    col = findfirst(==(name), rd.nz)
    ocol = findfirst(==(name * "::observed"), rd.nz)
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
        @test DividendToPrice().nonneg == ["dividends_ttm"]
        @test ForwardDividendToPrice().nonneg == ["dps_ntm"]
        @test AnalystDispersionToPrice().nonneg == ["eps_ntm_std"]
        @test ShortInterest().nonneg == ["short_interest"]
        @test ShareholderYield().num == ["dividends_ttm" => 1, "net_buybacks_ttm" => 1]
        @test ShareholderYield().nonneg == ["dividends_ttm"]
        @test BookLeverage().den == ["total_debt" => 1, "book_equity" => 1]
        @test MarketLeverage().den == ["total_debt" => 1, "market_cap" => 1]
        @test MarketLeverage().pos == ["market_cap"]
        @test GrossMargin().num == ["sales_ttm" => 1, "cost_of_revenue_ttm" => -1]
        @test GrossMargin().den == "sales_ttm"
        @test GrossProfitability().num == ["sales_ttm" => 1, "cost_of_revenue_ttm" => -1]
        @test AccrualsCashFlow().num ==
              ["net_income_ttm" => 1, "operating_cash_flow_ttm" => -1]
        @test isa(LogMarketCap(), PanelFieldLog)
        @test LogMarketCap().field == "market_cap"
        @test Passthrough(; field = "eps_ntm").field == "eps_ntm"
    end
    @testset "A keyword override renames the field, and the guard follows the rename" begin
        de = BookToPrice(; num = "be", den = "mc")
        @test de.num == "be"
        @test de.den == "mc"
        @test DividendToPrice(; num = "div").nonneg == ["div"]
        @test MarketLeverage(; debt = "d", mcap = "m").den == ["d" => 1, "m" => 1]
        @test MarketLeverage(; debt = "d", mcap = "m").pos == ["m"]
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
    end
end

@testset "The shared read: a fill reads back as NaN, and the active mask fills" begin
    amsk = [true true true; true false true; true true true]
    rd = descriptor_hand_panel(["mcap" => [1.0 2.0 3.0; NaN 5.0 6.0; 7.0 8.0 9.0],
                                "debt" => [0.5 1.0 1.5; 2.0 2.5 3.0; 3.5 4.0 4.5]];
                               amsk = amsk)
    V = PortfolioOptimisers.panel_field_values(rd, "mcap")
    @test isa(V, Matrix{Float64})
    @test isnan(V[2, 1])
    @test V[1, :] == [1.0, 2.0, 3.0]
    # The read itself does not touch the active mask: cell (2, 2) is inactive but observed.
    @test V[2, 2] == 5.0
    W = PortfolioOptimisers.panel_field_values(rd, ["mcap" => 1, "debt" => -2])
    @test W[1, :] == [0.0, 0.0, 0.0]
    @test W[3, 3] == 9.0 - 9.0
    @test isnan(W[2, 1])
    @test_throws IsEmptyError PortfolioOptimisers.panel_field_values(rd,
                                                                     Pair{String, Int}[])
    @test_throws KeyError PortfolioOptimisers.panel_field_values(rd, "mcp")
    # A carrier with no panel cannot be read, and a categorical field is not one number.
    bare = ReturnsResult(; nx = ["A1", "A2", "A3"], X = zeros(3, 3))
    @test_throws IsNothingError PortfolioOptimisers.panel_field_values(bare, "mcap")
    @test_throws IsNothingError descriptor(BookToPrice(), bare)
    res = asset_panel([NumericPanelInput(; name = "mcap", vals = ones(2, 2)),
                       CategoricalPanelInput(; name = "sector", vals = ["a" "b"; "a" "b"],
                                             levels = ["a", "b"])]; amsk = trues(2, 2),
                      emsk = trues(2, 2))
    rdc = ReturnsResult(; nx = ["A1", "A2"], X = zeros(2, 2), res...)
    @test_throws ArgumentError PortfolioOptimisers.panel_field_values(rdc, "sector")
    @test_throws ArgumentError descriptor(Passthrough(; field = "sector"), rdc)
    # A field that cannot blank carries no observed-mask column, and reads back whole.
    @test PortfolioOptimisers.panel_field_values(rdc, "mcap") == ones(2, 2)

    D = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
    @test isnothing(PortfolioOptimisers.descriptor_active_fill!(D, rd.pnl))
    @test isnan(D[2, 2])
    @test count(isnan, D) == 1
    @test_throws DimensionMismatch PortfolioOptimisers.descriptor_active_fill!(zeros(2, 3),
                                                                               rd.pnl)
end

@testset "Point-in-time ratios on a hand panel" begin
    # Column 3 carries a zero and a negative market capitalisation, row 2 an unobserved book
    # equity, and cell (3, 2) is inactive with values in every field.
    amsk = [true true true; true true true; true false true]
    be = [2.0 3.0 4.0; NaN 6.0 7.0; 8.0 9.0 10.0]
    mcap = [4.0 6.0 0.0; 10.0 12.0 -1.0; 16.0 18.0 20.0]
    debt = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
    div = [0.1 0.2 0.3; 0.4 0.5 0.6; 0.7 0.8 0.9]
    bb = [0.05 -0.1 0.15; -0.2 0.25 -0.3; 0.35 0.4 -0.45]
    rd = descriptor_hand_panel(["book_equity" => be, "market_cap" => mcap,
                                "total_debt" => debt, "dividends_ttm" => div,
                                "net_buybacks_ttm" => bb]; amsk = amsk)
    @testset "A ratio equals the field arithmetic, and NaN where the docstring says so" begin
        D = descriptor(BookToPrice(), rd)
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
    end
    @testset "A combination numerator sums its terms with their coefficients" begin
        D = descriptor(ShareholderYield(), rd)
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
        # Market leverage: the total capital at (1, 3) is 3 > 0, but the market
        # capitalisation is 0, so the pos guard writes NaN where the denominator alone would
        # not. At (2, 3) the total is 5 > 0 and the capitalisation is -1.
        D = descriptor(MarketLeverage(), rd)
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
        @test isa(descriptor(DividendToPrice(), rd), Matrix{Float64})
        neg = copy(div)
        neg[2, 2] = -0.1
        rdn = descriptor_hand_panel(["dividends_ttm" => neg, "market_cap" => mcap];
                                    amsk = amsk)
        @test_throws DomainError descriptor(DividendToPrice(), rdn)
        # A negative value on an inactive cell is not a data error the estimator reads.
        neg[2, 2] = 0.1
        neg[3, 2] = -0.1
        rdi = descriptor_hand_panel(["dividends_ttm" => neg, "market_cap" => mcap];
                                    amsk = amsk)
        @test isa(descriptor(DividendToPrice(), rdi), Matrix{Float64})
        # Neither is an unobserved one.
        neg[3, 2] = 0.1
        neg[1, 1] = NaN
        rdu = descriptor_hand_panel(["dividends_ttm" => neg, "market_cap" => mcap];
                                    amsk = amsk)
        @test isnan(descriptor(DividendToPrice(), rdu)[1, 1])
        # The buybacks may be negative: the guard names the dividends alone.
        @test isa(descriptor(ShareholderYield(), rd), Matrix{Float64})
    end
    @testset "The logarithm and the passthrough" begin
        D = descriptor(LogMarketCap(), rd)
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
        de = BookToPrice()
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
    @testset "GrowthRate refuses a negative field, as the reference implementation does" begin
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
        D = descriptor(EarningsChangeToPrice(; lag = 2), rd)
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
        D = descriptor(CapexToAssetsChangeInIntensity(; lag = 2), rd)
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
            @test count(isnan, D) == N + 1
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
