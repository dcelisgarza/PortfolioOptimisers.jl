#=
The missing-share report, the info report and the active-mask alignment of an Asset Panel (#1400).

`describe` counts the cells that the builder filled, `panel_info` prints the dimensions, the mask
coverage and the level coverage, and `panel_align_active` moves the start of each asset to the
first active observation where each named Panel Field is observed. Each is checked by hand on a
small panel, and against the oracle on the parity fixture of map #1375 with an observed mask
that has leading gaps, a field that is never observed, a categorical field with unobserved
cells, and a tensor field.
=#
const PO = PortfolioOptimisers
include(joinpath(@__DIR__, "parity_harness.jl"))

function summary_small_panel()
    mcap = NumericPanelField(; name = "mcap",
                             vals = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0],
                             omsk = Bool[0 1 1; 1 1 0; 1 0 0])
    sector = CategoricalPanelField(; name = "sector", levels = ["Tech", "Energy"],
                                   codes = [1 2 1; 1 2 1; 1 2 1],
                                   omsk = Bool[1 1 1; 1 1 0; 1 1 1])
    beta = TensorPanelField(; name = "beta", axis = "factor", labels = ["size", "value"],
                            vals = zeros(3, 3, 2),
                            omsk = cat(Bool[1 1 1; 1 1 1; 1 1 1], Bool[0 1 1; 1 1 1; 1 1 0];
                                       dims = 3))
    amsk = Bool[1 1 0; 1 1 1; 1 1 1]
    emsk = Bool[1 0 0; 1 1 1; 1 1 1]
    return AssetPanel(; pf = [mcap, sector, beta], amsk = amsk, emsk = emsk)
end

# The parity fixture with observed masks that the alignment acts on. Asset 1 starts
# `book_equity` late; asset 2 lists late and starts it later still; asset 4 misses it at its
# relisting, which is not a leading cell; asset 12 never reports it. `sales_ttm` is observed on
# asset 7 at the first observation only and then from observation 10, while `book_equity`
# starts on it at observation 5, so the order of the two fields matters to a one-pass rule.
# `industry` is not observed on asset 5 for its first ten observations, and the tensor field
# misses one label on asset 3 for its first three.
function summary_parity_panel()
    fx = parity_small_panel()
    pnl = fx.rd.pnl
    T, N = size(pnl.amsk)
    book = panel_field(pnl, "book_equity")
    bo = copy(book.omsk)
    bo[1:6, 1] .= false
    bo[21:26, 2] .= false
    bo[41:43, 4] .= false
    bo[:, 12] .= false
    bo[1:4, 7] .= false
    sales = panel_field(pnl, "sales_ttm")
    so = copy(sales.omsk)
    so[2:9, 7] .= false
    so[1:2, 1] .= false
    ind = panel_field(pnl, "industry")
    io_ = trues(T, N)
    io_[1:10, 5] .= false
    rng = StableRNG(1400)
    eo = trues(T, N, 3)
    eo[1:3, 3, 2] .= false
    eo[60:62, 8, 1] .= false
    expo = TensorPanelField(; name = "exposure", axis = "factor",
                            labels = ["size", "value", "mom"], vals = randn(rng, T, N, 3),
                            omsk = eo)
    pf = [NumericPanelField(; name = "book_equity", vals = book.vals, omsk = bo),
          NumericPanelField(; name = "sales_ttm", vals = sales.vals, omsk = so),
          panel_field(pnl, "adj_close"),
          CategoricalPanelField(; name = "industry", levels = ind.levels, codes = ind.codes,
                                omsk = io_), panel_field(pnl, "currency"), expo]
    return AssetPanel(; pf = pf, amsk = pnl.amsk, emsk = pnl.emsk)
end

const SUMMARY_ALIGN_CASES = ["Book" => ["book_equity"],
                             "BookSales" => ["book_equity", "sales_ttm"],
                             "SalesBook" => ["sales_ttm", "book_equity"],
                             "Industry" => ["industry"]]

@testset "describe counts the filled cells of each Panel Field" begin
    pnl = summary_small_panel()
    df = describe(pnl)
    @test df.field == ["mcap", "sector", "beta"]
    @test df.kind == [:numeric, :categorical, :tensor]
    @test df.cells == [9, 9, 18]
    @test df.missing ≈ [4 / 9, 1 / 9, 2 / 18] rtol = 1e-15
    @test df.active_cells == [8, 8, 16]
    # mcap: active and missing at (1, 1), (2, 3), (3, 2) and (3, 3); (1, 3) is inactive.
    @test df.active_missing ≈ [4 / 8, 1 / 8, 2 / 16] rtol = 1e-15
    # Asset 3 is active at observations 2 and 3, and mcap is missing at both.
    @test df.assets_missing == [1, 0, 0]
    by = describe(pnl; by = "sector")
    @test by.field == ["mcap", "mcap", "beta", "beta"]
    @test by.level == ["Tech", "Energy", "Tech", "Energy"]
    # Tech: (1, 1), (2, 1), (3, 1), (3, 3); (2, 3) is not observed and (1, 3) is inactive.
    @test by.cells == [4, 3, 8, 6]
    @test by.missing ≈ [2 / 4, 1 / 3, 2 / 8, 0 / 6] rtol = 1e-15
    @test_throws ArgumentError describe(pnl; by = "mcap")
    @test_throws KeyError describe(pnl; by = "industry")
end

@testset "describe on a static panel, an empty level and a panel with no Panel Field" begin
    pnl = AssetPanel(;
                     pf = [NumericPanelField(; name = "mcap", vals = [1.0, 2.0, 3.0],
                                             omsk = Bool[1, 0, 1]),
                           CategoricalPanelField(; name = "sector",
                                                 levels = ["Tech", "Energy", "Retail"],
                                                 codes = [1, 2, 1])])
    df = describe(pnl)
    @test df.cells == [3, 3]
    @test df.active_cells == [3, 3]
    @test df.missing == df.active_missing
    @test df.assets_missing == [1, 0]
    by = describe(pnl; by = "sector")
    @test by.cells == [2, 1, 0]
    # A share over no cell is NaN.
    @test isnan(by.missing[3])
    @test by.missing[1:2] == [0.0, 1.0]
    empty = AssetPanel(; amsk = trues(2, 2), emsk = trues(2, 2))
    @test size(describe(empty)) == (0, 7)
end

@testset "panel_info prints the dimensions, the masks, the fields and the levels" begin
    pnl = summary_small_panel()
    s = sprint(io -> panel_info(io, pnl; ts = ["t1", "t2", "t3"]))
    @test occursin("observations : 3 (t1 to t3)", s)
    @test occursin("Panel Fields : 3", s)
    @test occursin("missing      : 19.4% of the cells, 21.9% of the active cells", s)
    @test occursin("in mask          : 8 / 9 cells (88.9%)", s)
    @test occursin("in mask          : 7 / 9 cells (77.8%)", s)
    @test occursin("sector: 2 levels", s)
    @test occursin("< 10    : 2 levels (Tech, Energy)", s)
    @test !occursin("equal to the active mask", s)
    @test_throws DimensionMismatch panel_info(devnull, pnl; ts = ["t1"])
    @test isnothing(panel_info(devnull, pnl))
    # The method with no stream prints to `stdout`, so the report goes to the test log.
    @test isnothing(panel_info(pnl))
    same = AssetPanel(; pf = pnl.pf, amsk = pnl.amsk, emsk = pnl.amsk)
    @test occursin("estimation mask: equal to the active mask", sprint(panel_info, same))
    static = AssetPanel(; pf = [NumericPanelField(; name = "mcap", vals = [1.0, 2.0])])
    s = sprint(io -> panel_info(io, static; ts = [1]))
    @test occursin("observations : none, the panel is static", s)
    @test occursin("active mask: none, the panel is static", s)
    @test !occursin("categorical", s)
    # A group of more than six levels prints four names and the count of the others.
    levels = ["L$k" for k in 1:8]
    many = AssetPanel(;
                      pf = [CategoricalPanelField(; name = "g", levels = levels,
                                                  codes = [1:8;])])
    @test occursin("< 10    : 8 levels (L1, L2, L3, L4, … +4 more)",
                   sprint(panel_info, many))
    # A panel with no row prints no spread.
    norow = AssetPanel(; amsk = falses(0, 2), emsk = falses(0, 2))
    s = sprint(panel_info, norow)
    @test !occursin("assets per obs.", s)
    @test !occursin("duration", s)
end

@testset "panel_align_active moves the start of each asset to its first observed cell" begin
    pnl = summary_small_panel()
    res = panel_align_active(pnl, "mcap")
    # Asset 1 starts at 2; asset 2 is observed at its first cell; asset 3 is never observed
    # while active, so it leaves the universe.
    @test res.n == 1 + 2
    @test res.pnl.amsk == Bool[0 1 0; 1 1 0; 1 1 0]
    @test res.pnl.emsk == res.pnl.emsk .& res.pnl.amsk
    @test res.pnl.emsk == Bool[0 0 0; 1 1 0; 1 1 0]
    @test res.pnl.pf === pnl.pf
    # The tensor field is observed at a pair when each label is. Its gap at (3, 3) follows
    # the first active cell of asset 3, which is observed, so it stays.
    res = panel_align_active(pnl, ["beta"])
    @test res.n == 1
    @test res.pnl.amsk == Bool[0 1 0; 1 1 1; 1 1 1]
    # The order of the fields does not change the result, and a second call removes nothing.
    a = panel_align_active(pnl, ["mcap", "beta", "sector"])
    b = panel_align_active(pnl, ["sector", "beta", "mcap"])
    @test a.pnl.amsk == b.pnl.amsk
    @test a.n == b.n
    @test panel_align_active(a.pnl, ["mcap", "beta", "sector"]).n == 0
    # No field, or a field that cannot blank, removes nothing.
    @test panel_align_active(pnl, String[]).n == 0
    lifted = AssetPanel(;
                        pf = [NumericPanelField(; name = "x",
                                                vals = PO.RepeatedLeading([1.0, 2.0, 3.0],
                                                                          3))],
                        amsk = pnl.amsk, emsk = pnl.emsk)
    @test panel_align_active(lifted, "x").n == 0
    @test_throws KeyError panel_align_active(pnl, "nope")
    static = AssetPanel(; pf = [NumericPanelField(; name = "mcap", vals = [1.0, 2.0])])
    @test_throws ArgumentError panel_align_active(static, "mcap")
end

@testset "describe equals the oracle on the parity fixture" begin
    pnl = summary_parity_panel()
    df = describe(pnl)
    # The oracle gives each share as a percentage; the stored file holds it over 100.
    ref = parity_load("AssetPanelSummary", "Small", "Describe")
    r = parity_compare([df.missing df.active_missing], ref; name = "describe")
    @test r.ok
    # The oracle's rows are level by level; ours are field by field.
    by = describe(pnl; by = "industry")
    ref = parity_load("AssetPanelSummary", "Small", "DescribeBy")
    names = [f.name for f in pnl.pf]
    lv = panel_field(pnl, "industry").levels
    ord = [findfirst(i -> by.field[i] == names[Int(row[1]) + 1] &&
                          by.level[i] == lv[Int(row[2]) + 1], eachindex(by.field))
           for row in eachrow(ref)]
    @test by.cells[ord] == Int.(ref[:, 3])
    r = parity_compare(by.missing[ord], ref[:, 4]; name = "describe by industry")
    @test r.ok
end

@testset "the numbers of panel_info equal the oracle on the parity fixture" begin
    pnl = summary_parity_panel()
    for (msk, name) in ((pnl.amsk, "InfoActive"), (pnl.emsk, "InfoEstimation"))
        c = PO.panel_mask_coverage(msk)
        ref = Int.(vec(parity_load("AssetPanelSummary", "Small", name)))
        # The oracle truncates a median to an integer.
        @test [c.in_mask, c.cells, c.per_observation.min,
               floor(Int, c.per_observation.median), c.per_observation.max, c.assets, c.N,
               floor(Int, c.durations.median), c.durations.min, c.durations.max] == ref
    end
    # The oracle reports the fully missing assets of a numeric or categorical field.
    df = describe(pnl)
    ref = parity_load("AssetPanelSummary", "Small", "InfoFullyMissing")
    @test df.assets_missing[Int.(ref[:, 1]) .+ 1] == Int.(ref[:, 2])
    # The level groups: each row is a field, a level and a group, all zero-based.
    ref = Int.(parity_load("AssetPanelSummary", "Small", "InfoLevels"))
    s = sprint(panel_info, pnl)
    groups = ["< 10", "10 - 19", "20 - 49", ">= 50"]
    for row in eachrow(ref)
        f = pnl.pf[row[1] + 1]
        line = only(filter(l -> occursin("    $(rpad(groups[row[3] + 1], 8)):", l),
                           split(s[findfirst("  $(f.name): ", s)[1]:end], '\n')[2:5]))
        @test occursin(f.levels[row[2] + 1], line)
    end
end

@testset "panel_align_active equals the oracle on the parity fixture" begin
    pnl = summary_parity_panel()
    for (case, fs) in SUMMARY_ALIGN_CASES
        res = panel_align_active(pnl, fs)
        n = Int.(vec(parity_load("AssetPanelSummary", "Align$case", "N")))
        # The oracle's second call; one call of ours gives it.
        amsk = parity_load("AssetPanelSummary", "Align$case", "ActiveMask2")
        @test res.pnl.amsk == (amsk .== 1)
        @test res.n == sum(n)
        @test res.pnl.emsk ==
              (parity_load("AssetPanelSummary", "Align$case", "EstimationMask") .== 1) .&
              res.pnl.amsk
        if case == "SalesBook"
            # Better: the oracle's first call walks the fields in turn, so it stops at
            # observation 5 on asset 7, where `sales_ttm` is not observed, and its second
            # call moves on to observation 10. The rule of the library is its fixed point.
            first = parity_load("AssetPanelSummary", "Align$case", "ActiveMask") .== 1
            @test findfirst(first[:, 7]) == 5
            @test findfirst(res.pnl.amsk[:, 7]) == 10
            @test n[2] > 0
        else
            @test n[2] == 0
        end
    end
    # Better: the oracle refuses a tensor field; the library aligns asset 3 past the three
    # observations where the label "value" is not observed. The gap of asset 8 is not a
    # leading one, so it stays.
    res = panel_align_active(pnl, "exposure")
    @test res.n == 3
    @test findfirst(res.pnl.amsk[:, 3]) == 4
end
