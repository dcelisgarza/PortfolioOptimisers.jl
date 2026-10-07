#=
The round trip of an Asset Panel through a table (#1399, map #1375).

`panel_dataframe` writes the values and the masks of a panel, `panel_manifest` writes the parts that
a table column cannot hold, and `asset_panel(df, mf)` reads the panel back from the two tables. The
oracle saves its panel to a directory and loads it, so parity is a round trip. Every case here goes
through CSV files, which keep no type and no metadata, so a case that passes through CSV passes
through any table format.

The wide layout keeps every cell, so its round trip is exact. The long layout keeps only the cells
in the universe, so its round trip is exact on those cells, and a cell outside the universe comes
back as zero, the first level, and an unobserved cell.
=#
using Dates

include(joinpath(@__DIR__, "parity_harness.jl"))

function panel_1399_csv(df::DataFrame, mf::DataFrame; kwargs...)
    return mktempdir() do d
        p, m = joinpath(d, "panel.csv"), joinpath(d, "manifest.csv")
        CSV.write(p, df)
        CSV.write(m, mf)
        return asset_panel(DataFrame(CSV.File(p; types = Dict("asset" => String),
                                              validate = false)), DataFrame(CSV.File(m));
                           kwargs...)
    end
end
function panel_1399_trip(pnl; nx = nothing, ts = nothing, layout = :wide, decode = true,
                         kwargs...)
    return panel_1399_csv(panel_dataframe(pnl; nx, ts, layout, decode),
                          panel_manifest(pnl; nx, ts); decode, kwargs...)
end
# Every part of a Panel Field but the values, which a long table keeps on the active cells only.
function panel_1399_schema(f::NumericPanelField)
    return (NumericPanelField, f.name, isnothing(f.omsk))
end
function panel_1399_schema(f::CategoricalPanelField)
    return (CategoricalPanelField, f.name, isnothing(f.omsk), f.levels)
end
function panel_1399_schema(f::TensorPanelField)
    return (TensorPanelField, f.name, isnothing(f.omsk), f.axis, f.labels, f.groups)
end
panel_1399_values(f::CategoricalPanelField) = f.codes
panel_1399_values(f::PortfolioOptimisers.AbstractPanelField) = f.vals
# The cells of a Panel Field array where the active mask is true, one entry per trailing label.
function panel_1399_active(A::AbstractArray, amsk::AbstractMatrix{Bool})
    return [A[t, k, l...] for l in Iterators.product(axes(A)[3:end]...)
            for t in axes(A, 1), k in axes(A, 2) if amsk[t, k]]
end
function panel_1399_equal(p1::AssetPanel, p2::AssetPanel; active_only::Bool = false)
    ok = length(p1.pf) == length(p2.pf) && p1.amsk == p2.amsk && p1.emsk == p2.emsk
    for (f1, f2) in zip(p1.pf, p2.pf)
        ok &= panel_1399_schema(f1) == panel_1399_schema(f2)
        for (a1, a2) in
            ((panel_1399_values(f1), panel_1399_values(f2)), (f1.omsk, f2.omsk))
            if isnothing(a1)
                continue
            end
            ok &= if active_only
                panel_1399_active(a1, p1.amsk) == panel_1399_active(a2, p1.amsk)
            else
                a1 == a2
            end
        end
    end
    return ok
end
function panel_1399_static()
    return AssetPanel(;
                      pf = [NumericPanelField(; name = "mcap", vals = [1.5, 2.0, 3.25]),
                            CategoricalPanelField(; name = "sector",
                                                  levels = ["Tech", "Energy", "Unused"],
                                                  codes = [2, 2, 1],
                                                  omsk = [true, false, true]),
                            TensorPanelField(; name = "beta", axis = "factor",
                                             labels = ["size", "value"],
                                             groups = ["style", ""],
                                             vals = [1.0 2.0; 3.0 4.0; 5.0 6.0])])
end
function panel_1399_timevarying()
    # Asset 3 is never in the universe, and observation 3 holds no asset in the universe.
    amsk = Bool[1 1 0; 1 0 0; 0 0 0; 0 1 0]
    emsk = Bool[1 0 0; 1 0 0; 0 0 0; 0 1 0]
    rng = StableRNG(1399)
    return AssetPanel(;
                      pf = [NumericPanelField(; name = "mcap", vals = randn(rng, 4, 3),
                                              omsk = rand(rng, Bool, 4, 3)),
                            NumericPanelField(; name = "count",
                                              vals = rand(rng, 1:9, 4, 3)),
                            CategoricalPanelField(; name = "sector",
                                                  levels = ["Energy", "Tech", "Unused"],
                                                  codes = [2 1 2; 2 1 1; 1 1 1; 1 2 1]),
                            TensorPanelField(; name = "beta", axis = "factor",
                                             labels = ["size", "value", "mom"],
                                             groups = ["style", "style", "momentum"],
                                             vals = randn(rng, 4, 3, 3),
                                             omsk = rand(rng, Bool, 4, 3, 3)),
                            TensorPanelField(; name = "load", axis = "pc", labels = ["pc1"],
                                             vals = randn(rng, 4, 3, 1))], amsk = amsk,
                      emsk = emsk)
end
const TS_1399 = Date(2024, 1, 1) .+ Day.(0:3)
# The message of a `KeyError`, which `showerror` prints with its quotes escaped.
function panel_1399_keyerror(f)
    try
        f()
    catch e
        return e isa KeyError ? string(e.key) : ""
    end
    return ""
end
@testset "panel_manifest" begin
    pnl = panel_1399_timevarying()
    mf = panel_manifest(pnl; nx = ["A", "B", "C"], ts = TS_1399)
    @test names(mf) == ["kind", "field", "label", "group", "observed", "grouped"]
    @test mf.kind[1:8] ==
          ["panel", "observation", "observation", "observation", "observation", "asset",
           "asset", "asset"]
    @test mf.label[1:8] ==
          ["time-varying", "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "A", "B",
           "C"]
    # Each Panel Field in order; a level with no cell keeps its row.
    @test mf.kind[9:end] ==
          ["numeric", "numeric", "categorical", "level", "level", "level", "tensor",
           "label", "label", "label", "tensor", "label"]
    @test isequal(mf.label[12:14], ["Energy", "Tech", "Unused"])
    @test isequal(mf.observed[9:11], [true, false, false])
    @test isequal(mf.label[15], "factor") && isequal(mf.grouped[15], true)
    @test isequal(mf.group[16:18], ["style", "style", "momentum"])
    @test isequal(mf.grouped[19], false) && all(ismissing, mf.group[20:end])

    # A static panel has no observation row, and the selections narrow the manifest as they
    # narrow the table.
    smf = panel_manifest(panel_1399_static(); fields = "sector", assets = ["3", "1"])
    @test smf.kind == ["panel", "asset", "asset", "categorical", "level", "level", "level"]
    @test isequal(smf.label[1:3], ["static", "3", "1"])
    @test_throws DimensionMismatch panel_manifest(pnl; nx = ["A"])
    @test_throws DimensionMismatch panel_manifest(pnl; ts = TS_1399[1:2])
    @test_throws KeyError panel_manifest(pnl; fields = ["nope"])
end
@testset "asset_panel(df, mf): a static panel" begin
    pnl = panel_1399_static()
    for layout in (:long, :wide), decode in (true, false)
        p2 = panel_1399_trip(pnl; nx = ["A", "B", "C"], layout, decode)
        @test isnothing(p2.amsk) && isnothing(p2.emsk)
        @test panel_1399_equal(pnl, p2)
    end
    # The positions name the assets when the caller gives no names.
    @test panel_1399_equal(pnl, panel_1399_trip(pnl; layout = :long))
    # In memory, with no file.
    df = panel_dataframe(pnl)
    @test panel_1399_equal(pnl, asset_panel(df, panel_manifest(pnl)))
end
@testset "asset_panel(df, mf): a time-varying panel" begin
    pnl = panel_1399_timevarying()
    nx = ["A", "007", "C"]
    for decode in (true, false)
        # The wide layout keeps every cell, so the round trip is exact.
        w = panel_1399_trip(pnl; nx, ts = TS_1399, layout = :wide, decode)
        @test panel_1399_equal(pnl, w)
        @test eltype(panel_field(w, "count").vals) == Int
        # The long layout keeps the cells in the universe. The manifest restores the asset
        # that is never active and the observation with no active asset.
        l = panel_1399_trip(pnl; nx, ts = TS_1399, layout = :long, decode)
        @test size(l.amsk) == (4, 3)
        @test panel_1399_equal(pnl, l; active_only = true)
        # A cell outside the universe comes back as zero, the first level, and unobserved.
        @test all(iszero, panel_field(l, "mcap").vals[.!pnl.amsk])
        @test all(isone, panel_field(l, "sector").codes[.!pnl.amsk])
        @test !any(panel_field(l, "mcap").omsk[.!pnl.amsk])
        @test !panel_1399_equal(pnl, l)
    end
    # A subset of the Panel Fields, in the order of the call, from the whole pair.
    df = panel_dataframe(pnl; layout = :long)
    mf = panel_manifest(pnl)
    s = asset_panel(df, mf; fields = ["beta", "mcap"])
    @test [f.name for f in s.pf] == ["beta", "mcap"]
    @test panel_1399_schema(s.pf[1]) == panel_1399_schema(pnl.pf[4])
    @test [f.name for f in asset_panel(df, mf; fields = "sector").pf] == ["sector"]
    # A table with a subset of the Panel Fields reads with the same subset.
    sub = panel_dataframe(pnl; fields = ["load"], layout = :wide)
    @test panel_field(asset_panel(sub, mf; fields = ["load"]), "load").vals ==
          panel_field(pnl, "load").vals
    @test_throws KeyError asset_panel(sub, mf)
end
@testset "an observation with no active asset is valid input" begin
    # Better than the oracle (R76 of #1416, ADR 0102). The universe of an observation is a set
    # of assets, and the empty set is a set: before the first listing, or in a sub-universe
    # whose assets are all delisted. A view on assets intersects each row with the selection,
    # so the views of a panel are panels only when an empty row is allowed. The oracle refuses
    # an empty row, so it refuses its own selection of asset 1 below.
    amsk = Bool[1 0; 1 1; 0 1]
    pnl = AssetPanel(;
                     pf = [NumericPanelField(; name = "book",
                                             vals = [1.0 2.0; 3.0 4.0; 5.0 6.0],
                                             omsk = Bool[0 0; 1 1; 1 1])], amsk = amsk,
                     emsk = copy(amsk))
    # Every observation holds an active asset, and the view on asset 1 has none at
    # observation 3.
    v = PortfolioOptimisers.port_opt_view(pnl, :, [1])
    @test v.amsk == reshape(Bool[1, 1, 0], 3, 1)
    @test v.emsk == v.amsk
    # The alignment to `book` moves the start of asset 1 to observation 2, so observation 1
    # holds no active asset.
    res = panel_align_active(pnl, "book")
    @test res.n == 1
    @test res.pnl.amsk == Bool[0 0; 1 1; 0 1]
    @test !any(res.pnl.amsk[1, :])
    # The long table keeps no cell of that observation, and the manifest restores it.
    @test panel_1399_equal(res.pnl, panel_1399_trip(res.pnl; layout = :long);
                           active_only = true)
    # The count of active assets of each observation is 0, 2 and 1.
    c = PortfolioOptimisers.panel_mask_coverage(res.pnl.amsk)
    @test c.per_observation == (; min = 0, median = 1.0, max = 2)
end
@testset "asset_panel(df, mf): masks alone, a lifted field and the parity fixture" begin
    # A panel with no Panel Field, and one with no observation.
    for T in (3, 0)
        amsk = trues(T, 2)
        pnl = AssetPanel(; amsk = amsk, emsk = amsk)
        for layout in (:long, :wide)
            p2 = panel_1399_trip(pnl; layout)
            @test isempty(p2.pf) && p2.amsk == amsk && p2.emsk == amsk
        end
    end
    # A static input that a time-varying build lifts comes back as an ordinary array.
    amsk = Bool[1 1; 0 1; 1 1]
    pnl = asset_panel([NumericPanelInput(; name = "mcap",
                                         vals = [10.0 20.0; 11.0 21.0; 12.0 22.0]),
                       CategoricalPanelInput(; name = "sector", vals = ["Fin", "Tech"])];
                      amsk = amsk, emsk = amsk)
    @test panel_field(pnl, "sector").codes isa PortfolioOptimisers.RepeatedLeading
    p2 = panel_1399_trip(pnl)
    @test panel_field(p2, "sector").codes isa Matrix{Int}
    @test panel_1399_equal(pnl, p2)
    # The panel of the parity fixture, with its real names and dates.
    fx = parity_small_panel()
    ppnl = fx.rd.pnl
    ts = Date(2020, 1, 1) .+ Day.(0:(size(ppnl.amsk, 1) - 1))
    @test panel_1399_equal(ppnl, panel_1399_trip(ppnl; nx = fx.rd.nx, ts))
    @test panel_1399_equal(ppnl, panel_1399_trip(ppnl; nx = fx.rd.nx, ts, layout = :long);
                           active_only = true)
end
@testset "asset_panel(df, mf): refusals" begin
    pnl = panel_1399_timevarying()
    L = panel_dataframe(pnl; nx = ["A", "B", "C"], ts = TS_1399)
    W = panel_dataframe(pnl; nx = ["A", "B", "C"], ts = TS_1399, layout = :wide)
    mf = panel_manifest(pnl; nx = ["A", "B", "C"], ts = TS_1399)
    # Not a manifest.
    @test_throws KeyError asset_panel(L, mf[:, 1:5])
    @test_throws "one \"panel\" row" asset_panel(L, mf[2:end, :])
    bad = copy(mf)
    bad.kind[9] = "vector"
    @test_throws "a \"level\" or a \"label\" row follows" asset_panel(L, bad)
    bad = copy(mf)
    bad.field[12] = "industry"
    @test_throws "a \"level\" or a \"label\" row follows" asset_panel(L, bad)
    bad = copy(mf)
    bad.label[1] = "static"
    @test_throws "a static panel has no observation axis" asset_panel(L, bad)
    bad = copy(mf)
    bad.label[8] = "A"
    @test_throws "names an asset twice" asset_panel(W, bad)
    bad = copy(mf)
    bad.label[3] = "2024-01-01"
    @test_throws "names an observation twice" asset_panel(L, bad)
    @test_throws KeyError asset_panel(L, mf; fields = ["nope"])
    # The table does not match the manifest.
    @test_throws DimensionMismatch asset_panel(W[1:3, :], mf)
    @test_throws KeyError asset_panel(DataFrames.select(L, DataFrames.Not("beta=mom")), mf)
    r = copy(L)
    r.asset[1] = "D"
    @test occursin("names no asset \"D\"", panel_1399_keyerror(() -> asset_panel(r, mf)))
    r = copy(L)
    r.observation[1] = Date(1999, 1, 1)
    @test occursin("names no observation \"1999-01-01\"",
                   panel_1399_keyerror(() -> asset_panel(r, mf)))
    @test_throws "two rows of the long table" asset_panel(vcat(L, L[1:1, :]), mf)
    r = DataFrames.allowmissing(W)
    r[2, "mcap@B"] = missing
    @test_throws "no value in column \"mcap@B\" at row 2" asset_panel(r, mf)
    r = copy(L)
    r.sector[1] = "Retail"
    @test occursin("holds no level \"Retail\"",
                   panel_1399_keyerror(() -> asset_panel(r, mf)))
    @test_throws "decode = false" asset_panel(L, mf; decode = false)
    # A static long table holds every asset.
    spnl = panel_1399_static()
    @test_throws "one row per asset" asset_panel(panel_dataframe(spnl)[1:2, :],
                                                 panel_manifest(spnl))
    # A reader that types the asset "007" as the integer 7 loses the name; reading the key
    # column as text keeps it.
    tv = panel_1399_timevarying()
    mktempdir() do d
        p, m = joinpath(d, "panel.csv"), joinpath(d, "manifest.csv")
        CSV.write(p, panel_dataframe(tv; nx = ["007", "8", "9"]))
        CSV.write(m, panel_manifest(tv; nx = ["007", "8", "9"]))
        mf2 = DataFrame(CSV.File(m))
        @test occursin("read the key columns as text",
                       panel_1399_keyerror(() -> asset_panel(DataFrame(CSV.File(p)), mf2)))
        @test panel_1399_equal(tv,
                               asset_panel(DataFrame(CSV.File(p;
                                                              types = Dict("asset" =>
                                                                               String))),
                                           mf2); active_only = true)
    end
end
