#=
The public read of a Panel Field at a stated policy for its inactive and its unobserved cells
(#1411). The panel keeps a finite value in every cell (ADR 0102), and the caller picks, at read
time, what an inactive cell and an unobserved cell read as. The element type is derived from the
stored type and the two policies.
=#

@testset "panel_field_values: a policy for the inactive and the unobserved cells (#1411)" begin
    PO = PortfolioOptimisers
    # Asset 2 lists at observation 2, and asset 1 has a blank at observation 3.
    amsk = [true false; true true; true true]
    pnl = asset_panel([NumericPanelInput(; name = "mcap",
                                         vals = [1.0 NaN; 3.0 4.0; NaN 6.0],
                                         alg = ForwardPanelFill(; val = 0.0)),
                       NumericPanelInput(; name = "shares", vals = [10 20; 30 40; 50 60]),
                       NumericPanelInput(; name = "f32", vals = Float32[1 2; 3 4; 5 6]),
                       CategoricalPanelInput(; name = "sector",
                                             vals = ["a" "b"; "b" "b"; "a" "a"])];
                      amsk = amsk, emsk = amsk)
    stored = panel_field(pnl, "mcap").vals
    @testset "A float field" begin
        # The default keeps an inactive cell and blanks each cell the fill wrote. Cell (1, 2)
        # is both, and the default inactive policy keeps no value, so the fill reads NaN.
        @test isequal(panel_field_values(pnl, "mcap"), [1.0 NaN; 3.0 4.0; NaN 6.0])
        @test isequal(panel_field_values(pnl, "mcap"; unobserved = nothing),
                      [1.0 stored[1, 2]; 3.0 4.0; stored[3, 1] 6.0])
        @test isequal(panel_field_values(pnl, "mcap"; inactive = NaN),
                      [1.0 NaN; 3.0 4.0; NaN 6.0])
        @test isequal(panel_field_values(pnl, "mcap"; inactive = 0),
                      [1.0 0.0; 3.0 4.0; NaN 6.0])
        # With no unobserved policy the fill value reads back.
        @test panel_field_values(pnl, "mcap"; inactive = 0, unobserved = nothing) ==
              [1.0 0.0; 3.0 4.0; stored[3, 1] 6.0]
        @test panel_field_values(pnl, "mcap"; inactive = nothing, unobserved = nothing) ==
              stored
        @test eltype(panel_field_values(pnl, "mcap"; inactive = 0)) === Float64
    end
    @testset "A cell both inactive and unobserved takes the inactive policy" begin
        # Cell (1, 2) was blank in the input and is inactive, so it is both.
        @test !panel_field(pnl, "mcap").omsk[1, 2]
        @test panel_field_values(pnl, "mcap"; inactive = -1)[1, 2] == -1.0
        @test isnan(panel_field_values(pnl, "mcap"; inactive = nothing)[1, 2])
    end
    @testset "An integer field" begin
        # An integer policy keeps the integer type, and a NaN policy floats it.
        V = panel_field_values(pnl, "shares"; inactive = 0, unobserved = nothing)
        @test V == [10 0; 30 40; 50 60]
        @test eltype(V) === Int
        W = panel_field_values(pnl, "shares"; inactive = NaN)
        @test isequal(W, [10.0 NaN; 30.0 40.0; 50.0 60.0])
        @test eltype(W) === Float64
        # The default unobserved policy is NaN, so the default read is floated too.
        @test eltype(panel_field_values(pnl, "shares")) === Float64
        @test eltype(panel_field_values(pnl, "shares"; inactive = 0.5,
                                        unobserved = nothing)) === Float64
        # A Float32 field stays Float32 under a NaN policy.
        F = panel_field_values(pnl, "f32"; inactive = NaN)
        @test eltype(F) === Float32
        @test isequal(F, Float32[1 NaN; 3 4; 5 6])
    end
    @testset "A categorical field reads its codes" begin
        C = panel_field_values(pnl, "sector"; inactive = PO.CS_MISSING_GROUP,
                               unobserved = nothing)
        @test C == [1 PO.CS_MISSING_GROUP; 2 2; 1 1]
        @test eltype(C) === Int
        @test isequal(panel_field_values(pnl, "sector"; inactive = NaN),
                      [1.0 NaN; 2.0 2.0; 1.0 1.0])
        @test cross_sectional_groups(pnl, "sector") == C
    end
    @testset "A tensor field takes the policy on every label" begin
        tp = AssetPanel(;
                        pf = [TensorPanelField(; name = "b", axis = "factor",
                                               labels = ["x", "y"],
                                               vals = reshape(collect(1.0:12.0), 3, 2, 2))],
                        amsk = amsk, emsk = amsk)
        B = panel_field_values(tp, "b"; inactive = NaN)
        @test size(B) == (3, 2, 2)
        @test all(isnan, B[1, 2, :])
        @test count(isnan, B) == 2
        @test isequal(B[.!isnan.(B)], tp.pf[1].vals[.!isnan.(B)])
    end
    @testset "A static panel has no inactive cell" begin
        sp = AssetPanel(; pf = [NumericPanelField(; name = "m", vals = [1, 2, 3])])
        @test panel_field_values(sp, "m"; inactive = NaN) == [1.0, 2.0, 3.0]
        @test panel_field_values(sp, "m"; inactive = 0, unobserved = nothing) == [1, 2, 3]
    end
    @testset "A ReturnsResult reads its panel, and the read is a copy" begin
        rd = ReturnsResult(; nx = ["A", "B"], X = zeros(3, 2), pnl = pnl)
        @test isequal(panel_field_values(rd, "mcap"; inactive = NaN),
                      panel_field_values(pnl, "mcap"; inactive = NaN))
        V = panel_field_values(rd, "mcap"; unobserved = nothing)
        V .= -1.0
        @test panel_field(pnl, "mcap").vals == stored
        @test_throws IsNothingError panel_field_values(ReturnsResult(; nx = ["A", "B"],
                                                                     X = zeros(3, 2)),
                                                       "mcap")
        @test_throws KeyError panel_field_values(pnl, "mcp")
    end
    @testset "A policy the answer cannot hold refuses at the first cell it writes" begin
        rp = AssetPanel(;
                        pf = [NumericPanelField(; name = "r",
                                                vals = [1//2 1//3; 1//4 1//5])],
                        amsk = [true false; true true], emsk = [true false; true true])
        @test_throws InexactError panel_field_values(rp, "r"; inactive = NaN)
        @test panel_field_values(rp, "r"; inactive = 0) == [1//2 0//1; 1//4 1//5]
        @test eltype(panel_field_values(rp, "r"; inactive = 0)) === Rational{Int}
    end
end
