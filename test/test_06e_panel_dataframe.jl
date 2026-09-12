#=
The Asset Panel's DataFrame export (issue #929, map #643).

The library persists no panel of its own, so `panel_dataframe` is the whole of its interchange:
a caller writes the table with whatever they already use. The file tests the three shapes a
call can take -- one Panel Field laid out as it stands, the long layout and the wide layout --
across the panel's static and time-varying shapes, and the two selections and the categorical
decoding that narrow each of them.

A `TensorPanelField` is the one Panel Field the reference implementation's own export skips.
Here it is carried: its trailing axis spreads into one column per label, under the
`"<field>=<label>"` name it already takes in a derived Feature Matrix. Only the single-field
shape refuses it, because observations × assets has no room for a third axis.
=#
function panel_929_static()
    return AssetPanel(;
                      pf = [NumericPanelField(; name = "mcap", vals = [1.0, 2.0, 3.0]),
                            CategoricalPanelField(; name = "sector",
                                                  levels = ["Tech", "Energy"],
                                                  codes = [1, 2, 1],
                                                  omsk = [true, false, true]),
                            TensorPanelField(; name = "beta", axis = "factor",
                                             labels = ["size", "value"],
                                             vals = [1.0 2.0; 3.0 4.0; 5.0 6.0])])
end
function panel_929_timevarying()
    amsk = [true true true; true false true]
    emsk = [true true false; true false true]
    return AssetPanel(;
                      pf = [NumericPanelField(; name = "mcap",
                                              vals = [1.0 2.0 3.0; 4.0 5.0 6.0],
                                              omsk = amsk),
                            CategoricalPanelField(; name = "sector",
                                                  levels = ["Tech", "Energy"],
                                                  codes = [1 2 1; 2 2 1]),
                            TensorPanelField(; name = "beta", axis = "factor",
                                             labels = ["size", "value"],
                                             vals = reshape(collect(1.0:12.0), 2, 3, 2),
                                             omsk = reshape(trues(12), 2, 3, 2))],
                      amsk = amsk, emsk = emsk)
end
@testset "panel_frame_columns, the one place a table and a Feature Matrix differ" begin
    pnl = panel_929_static()
    num, cat, ten = pnl.pf

    # A numeric Panel Field is one column under its own name, in both renderings.
    cols = PortfolioOptimisers.panel_frame_columns(num, true)
    @test length(cols) == 1
    @test first(cols)[1] == "mcap"
    @test first(cols)[2] == [1.0, 2.0, 3.0]
    @test first(cols)[3] === nothing

    # A categorical Panel Field is ONE column here and one per level in a Feature Matrix,
    # because a table cell holds the level itself.
    cols = PortfolioOptimisers.panel_frame_columns(cat, true)
    @test length(cols) == 1
    @test length(PortfolioOptimisers.panel_field_labels(cat)) == 2
    @test first(cols)[1] == "sector"
    @test first(cols)[2] == ["Tech", "Energy", "Tech"]
    @test first(cols)[3] == [true, false, true]

    # `decode = false` hands the codes back instead.
    @test first(PortfolioOptimisers.panel_frame_columns(cat, false))[2] == [1, 2, 1]

    # A tensor Panel Field takes the name a Feature Matrix gives it, one column per label.
    cols = PortfolioOptimisers.panel_frame_columns(ten, true)
    @test [c[1] for c in cols] == PortfolioOptimisers.panel_field_labels(ten)
    @test [c[1] for c in cols] == ["beta=size", "beta=value"]
    @test cols[1][2] == [1.0, 3.0, 5.0]
    @test cols[2][2] == [2.0, 4.0, 6.0]
    @test all(isnothing(c[3]) for c in cols)

    # A tensor Panel Field that carries a mask slices the mask on the same axis.
    tvten = panel_929_timevarying().pf[3]
    cols = PortfolioOptimisers.panel_frame_columns(tvten, true)
    @test size(cols[1][2]) == (2, 3)
    @test cols[2][3] == trues(2, 3)
end
@testset "panel_frame_fields and panel_frame_assets, the two selections" begin
    pnl = panel_929_static()

    # `nothing` is every Panel Field, in panel order; a collection is the order it writes.
    @test PortfolioOptimisers.panel_frame_fields(pnl, nothing) === pnl.pf
    @test [f.name for f in PortfolioOptimisers.panel_frame_fields(pnl, ["beta", "mcap"])] ==
          ["beta", "mcap"]
    @test_throws KeyError PortfolioOptimisers.panel_frame_fields(pnl, ["mcapp"])

    # The panel does not name its assets, so the names are the caller's.
    nx = ["A", "B", "C"]
    @test PortfolioOptimisers.panel_frame_assets(nx, nothing) == [1, 2, 3]
    @test PortfolioOptimisers.panel_frame_assets(nx, "B") == [2]
    @test PortfolioOptimisers.panel_frame_assets(nx, ["C", "A"]) == [3, 1]
    @test_throws KeyError PortfolioOptimisers.panel_frame_assets(nx, ["D"])
    @test_throws ArgumentError PortfolioOptimisers.panel_frame_assets(nx, ["A", "A"])
end
@testset "The long layout of a static panel" begin
    pnl = panel_929_static()
    df = panel_dataframe(pnl; nx = ["A", "B", "C"])

    # A static panel has no observation axis and no mask, so the asset is the only key.
    @test names(df) ==
          ["asset", "mcap", "sector", "sector::observed", "beta=size", "beta=value"]
    @test DataFrames.nrow(df) == 3
    @test df.asset == ["A", "B", "C"]
    @test df.mcap == [1.0, 2.0, 3.0]
    @test df.sector == ["Tech", "Energy", "Tech"]
    @test df[!, "sector::observed"] == [true, false, true]
    @test df[!, "beta=size"] == [1.0, 3.0, 5.0]
    @test df[!, "beta=value"] == [2.0, 4.0, 6.0]

    # With no `nx` an asset is named by its position on the panel's asset axis.
    @test panel_dataframe(pnl).asset == ["1", "2", "3"]

    # `decode = false` keeps the storage the panel holds.
    @test panel_dataframe(pnl; decode = false).sector == [1, 2, 1]

    # Both selections narrow the table, each in the order it writes.
    df = panel_dataframe(pnl; nx = ["A", "B", "C"], fields = ["beta", "mcap"],
                         assets = ["C", "A"])
    @test names(df) == ["asset", "beta=size", "beta=value", "mcap"]
    @test df.asset == ["C", "A"]
    @test df.mcap == [3.0, 1.0]
    @test df[!, "beta=size"] == [5.0, 1.0]
end
@testset "The long layout of a time-varying panel is filtered by the active mask" begin
    pnl = panel_929_timevarying()
    df = panel_dataframe(pnl; nx = ["A", "B", "C"])

    # One row per (observation, asset) that the active mask holds: (2, B) is not in the
    # universe, so it has no row at all. The estimation mask stays as a column, because it
    # varies within the rows that survive.
    @test names(df) ==
          ["observation", "asset", "mcap", "mcap::observed", "sector", "beta=size",
           "beta=size::observed", "beta=value", "beta=value::observed", "emsk"]
    @test DataFrames.nrow(df) == 5
    @test df.observation == [1, 1, 1, 2, 2]
    @test df.asset == ["A", "B", "C", "A", "C"]
    @test df.mcap == [1.0, 2.0, 3.0, 4.0, 6.0]
    @test df.sector == ["Tech", "Energy", "Tech", "Energy", "Tech"]
    @test df.emsk == [true, true, false, true, true]

    # The tensor Panel Field lands on the same rows, one column per trailing-axis label.
    @test df[!, "beta=size"] == [1.0, 3.0, 5.0, 2.0, 6.0]
    @test df[!, "beta=value"] == [7.0, 9.0, 11.0, 8.0, 12.0]
    @test all(df[!, "beta=size::observed"])

    # `ts` labels the observations, and the asset order is the order `assets` writes.
    df = panel_dataframe(pnl; nx = ["A", "B", "C"], ts = ["d1", "d2"], assets = ["C", "A"],
                         fields = ["mcap"])
    @test names(df) == ["observation", "asset", "mcap", "mcap::observed", "emsk"]
    @test df.observation == ["d1", "d1", "d2", "d2"]
    @test df.asset == ["C", "A", "C", "A"]
    @test df.mcap == [3.0, 1.0, 6.0, 4.0]
end
@testset "The wide layout keeps every cell, and both universe masks" begin
    pnl = panel_929_timevarying()
    df = panel_dataframe(pnl; nx = ["A", "B", "C"], layout = :wide, fields = ["mcap"])

    # One row per observation, one column per (Panel Field column, asset). Nothing is
    # filtered, which is why the active mask comes with it as a block of its own.
    @test names(df) == ["observation", "mcap@A", "mcap@B", "mcap@C", "mcap::observed@A",
                        "mcap::observed@B", "mcap::observed@C", "amsk@A", "amsk@B", "amsk@C", "emsk@A",
                        "emsk@B", "emsk@C"]
    @test DataFrames.nrow(df) == 2
    @test df.observation == [1, 2]
    @test df[!, "mcap@B"] == [2.0, 5.0]
    @test df[!, "amsk@B"] == [true, false]
    @test df[!, "emsk@C"] == [false, true]

    # A tensor Panel Field spreads on both axes at once: label, then asset.
    df = panel_dataframe(pnl; nx = ["A", "B", "C"], layout = :wide, fields = ["beta"],
                         assets = ["A"])
    @test names(df) ==
          ["observation", "beta=size@A", "beta=size::observed@A", "beta=value@A",
           "beta=value::observed@A", "amsk@A", "emsk@A"]
    @test df[!, "beta=size@A"] == [1.0, 2.0]
    @test df[!, "beta=value@A"] == [7.0, 8.0]

    # A static panel has one implicit observation, so its wide table is one row.
    df = panel_dataframe(panel_929_static(); nx = ["A", "B", "C"], layout = :wide,
                         fields = ["mcap", "sector"])
    @test DataFrames.nrow(df) == 1
    @test names(df) == ["mcap@A", "mcap@B", "mcap@C", "sector@A", "sector@B", "sector@C",
                        "sector::observed@A", "sector::observed@B", "sector::observed@C"]
    @test df[!, "mcap@C"] == [3.0]
    @test df[!, "sector@B"] == ["Energy"]
end
@testset "One Panel Field is laid out as it stands, and a tensor field has no such shape" begin
    pnl = panel_929_timevarying()

    # A single name is the third shape: observations × assets, no layout, no mask.
    df = panel_dataframe(pnl; nx = ["A", "B", "C"], fields = "mcap", ts = ["d1", "d2"])
    @test names(df) == ["observation", "A", "B", "C"]
    @test df.observation == ["d1", "d2"]
    @test df.B == [2.0, 5.0]

    # It decodes like every other shape, and it takes the asset selection.
    df = panel_dataframe(pnl; nx = ["A", "B", "C"], fields = "sector", assets = "B")
    @test names(df) == ["observation", "B"]
    @test df.B == ["Energy", "Energy"]
    @test panel_dataframe(pnl; nx = ["A", "B", "C"], fields = "sector", decode = false).A ==
          [1, 2]

    # A static panel's single-field table is its one implicit observation.
    df = panel_dataframe(panel_929_static(); nx = ["A", "B", "C"], fields = "mcap")
    @test names(df) == ["A", "B", "C"]
    @test df.A == [1.0]

    # A tensor Panel Field carries a third axis, so it has no observations × assets table.
    # The message names the layouts that do carry it.
    @test_throws ArgumentError panel_dataframe(pnl; fields = "beta")
    msg = try
        panel_dataframe(pnl; fields = "beta")
    catch e
        sprint(showerror, e)
    end
    @test occursin("factor axis of 2 label(s)", msg)
    @test occursin("fields = [\"beta\"]", msg)
end
@testset "The axes and the layout are checked against the panel" begin
    pnl = panel_929_timevarying()
    @test_throws ArgumentError panel_dataframe(pnl; layout = :tall)
    @test_throws DimensionMismatch panel_dataframe(pnl; nx = ["A", "B"])
    @test_throws DimensionMismatch panel_dataframe(pnl; ts = [1])
    @test_throws KeyError panel_dataframe(pnl; fields = ["mcapp"])
    @test_throws KeyError panel_dataframe(pnl; nx = ["A", "B", "C"], assets = ["D"])

    # A static panel has no observation axis, so `ts` is ignored rather than checked.
    @test DataFrames.nrow(panel_dataframe(panel_929_static(); ts = [1, 2, 3, 4])) == 3
end
@testset "A panel view exports the universe it was cut down to" begin
    pnl = panel_929_timevarying()
    v = PortfolioOptimisers.port_opt_view(pnl, 1:2, [1, 3])
    df = panel_dataframe(v; nx = ["A", "C"], fields = ["mcap", "beta"])
    @test df.asset == ["A", "C", "A", "C"]
    @test df.mcap == [1.0, 3.0, 4.0, 6.0]
    @test df[!, "beta=value"] == [7.0, 11.0, 8.0, 12.0]
    @test DataFrames.nrow(panel_dataframe(v; nx = ["A", "C"], layout = :wide)) == 2
end
