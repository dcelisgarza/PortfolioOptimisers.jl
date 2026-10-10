#=
The concatenation of Asset Panels along the observation axis.

`vcat` joins the parts of one universe. A panel split into two views and joined again is the
panel it came from, for each kind of Panel Field, with and without an observed mask. A part
whose schema differs from the first part is refused, one refusal per testset entry. A lazy
lift and an all-true mask stay lazy when every part holds the same one.
=#
const PO = PortfolioOptimisers

function concat_test_panel(T::Integer, N::Integer; rng = StableRNG(1401))
    num = NumericPanelField(; name = "mcap", vals = rand(rng, T, N),
                            omsk = rand(rng, T, N) .> 0.2)
    cat = CategoricalPanelField(; name = "sector", levels = ["Tech", "Energy", "Retail"],
                                codes = rand(rng, 1:3, T, N))
    ten = TensorPanelField(; name = "beta", axis = "factor",
                           labels = ["size", "value", "mom"],
                           groups = ["style", "style", "momentum"],
                           vals = randn(rng, T, N, 3), omsk = rand(rng, T, N, 3) .> 0.1)
    amsk = rand(rng, T, N) .> 0.1
    emsk = amsk .& (rand(rng, T, N) .> 0.2)
    return AssetPanel(; pf = [num, cat, ten], amsk = amsk, emsk = emsk)
end

function same_panel(a::AssetPanel, b::AssetPanel)
    return isequal(panel_feature_matrix(a), panel_feature_matrix(b)) &&
           isequal(Matrix(a.amsk), Matrix(b.amsk)) &&
           isequal(Matrix(a.emsk), Matrix(b.emsk))
end

function flat_omsk(f)
    if isnothing(f.omsk)
        return nothing
    end
    return Matrix(reshape(f.omsk, size(f.omsk, 1), :))
end

function tv_panel(pf; T::Integer, N::Integer)
    return AssetPanel(; pf = pf, amsk = trues(T, N), emsk = trues(T, N))
end

@testset "A split panel joins back to itself, for every kind of Panel Field" begin
    T, N = 9, 4
    pnl = concat_test_panel(T, N)
    parts = [PO.port_opt_view(pnl, r, :) for r in (1:3, 4:4, 5:T)]
    joined = vcat(parts...)
    @test same_panel(joined, pnl)
    @test PO.panel_axes(joined) == (T, N)
    @test [typeof(f).name.wrapper for f in joined.pf] ==
          [NumericPanelField, CategoricalPanelField, TensorPanelField]
    # Each field keeps its values and its observed mask cell for cell.
    for (f, g) in zip(joined.pf, pnl.pf)
        @test f.name == g.name
        @test isequal(flat_omsk(f), flat_omsk(g))
    end
    @test panel_field(joined, "sector").levels == ["Tech", "Energy", "Retail"]
    @test panel_field(joined, "sector").codes == panel_field(pnl, "sector").codes
    @test panel_field(joined, "beta").groups == ["style", "style", "momentum"]
    @test panel_field(joined, "beta").axis == "factor"
    @test panel_field(joined, "beta").vals == panel_field(pnl, "beta").vals
    # `reduce` gives the same panel, and so does a concatenation of one part.
    @test same_panel(reduce(vcat, parts), pnl)
    @test same_panel(vcat(pnl), pnl)
end

@testset "A part with no observed mask joins a part with one as observed cells" begin
    a = tv_panel([NumericPanelField(; name = "x", vals = [1.0 2.0; 3.0 4.0])]; T = 2, N = 2)
    b = tv_panel([NumericPanelField(; name = "x", vals = [5.0 6.0], omsk = [true false])];
                 T = 1, N = 2)
    c = vcat(a, b)
    @test c.pf[1].vals == [1.0 2.0; 3.0 4.0; 5.0 6.0]
    @test c.pf[1].omsk == [true true; true true; true false]
    # With no mask in any part, the result has none.
    @test isnothing(vcat(a, a).pf[1].omsk)
end

@testset "The element types promote as vcat promotes them" begin
    a = tv_panel([NumericPanelField(; name = "x", vals = Float32[1 2])]; T = 1, N = 2)
    b = tv_panel([NumericPanelField(; name = "x", vals = [3.0 4.0])]; T = 1, N = 2)
    @test eltype(vcat(a, b).pf[1].vals) === Float64
    @test eltype(vcat(a, a).pf[1].vals) === Float32
end

@testset "A panel with no Panel Field joins its masks" begin
    a = AssetPanel(; amsk = [true true; false true], emsk = [true false; false true])
    b = AssetPanel(; amsk = [true false], emsk = [true false])
    c = vcat(a, b)
    @test isempty(c.pf)
    @test c.pf === a.pf
    @test Matrix(c.amsk) == [true true; false true; true false]
    @test Matrix(c.emsk) == [true false; false true; true false]
end

@testset "A lazy lift and an all-true mask stay lazy" begin
    s = NumericPanelField(; name = "size", vals = [1.0, 2.0, 3.0])
    lifted(n) = AssetPanel(; pf = [PO.panel_field_lift(s, n)], amsk = PO.AllTrueMask(n, 3),
                           emsk = PO.AllTrueMask(n, 3))
    c = vcat(lifted(2), lifted(5))
    @test isa(c.pf[1].vals, PO.RepeatedLeading)
    @test size(c.pf[1].vals) == (7, 3)
    @test c.pf[1].vals == repeat([1.0 2.0 3.0], 7)
    @test isa(c.amsk, PO.AllTrueMask) && isa(c.emsk, PO.AllTrueMask)
    @test size(c.amsk) == (7, 3)
    # A lift of an equal array in a new object is the same lift.
    t = NumericPanelField(; name = "size", vals = [1.0, 2.0, 3.0])
    d = vcat(lifted(2),
             AssetPanel(; pf = [PO.panel_field_lift(t, 1)], amsk = PO.AllTrueMask(1, 3),
                        emsk = PO.AllTrueMask(1, 3)))
    @test isa(d.pf[1].vals, PO.RepeatedLeading)
    # A lift of another array makes the field time-varying, and a mask with a false cell
    # makes the mask dense.
    u = NumericPanelField(; name = "size", vals = [4.0, 5.0, 6.0])
    e = vcat(lifted(2),
             AssetPanel(; pf = [PO.panel_field_lift(u, 1)], amsk = trues(1, 3),
                        emsk = [true false true]))
    @test !isa(e.pf[1].vals, PO.RepeatedLeading)
    @test e.pf[1].vals == [1.0 2.0 3.0; 1.0 2.0 3.0; 4.0 5.0 6.0]
    @test !isa(e.amsk, PO.AllTrueMask)
    @test Matrix(e.emsk) == [true true true; true true true; true false true]
end

@testset "Equal static panels join to the first one" begin
    mk() = AssetPanel(;
                      pf = [NumericPanelField(; name = "mcap", vals = [1.0, 2.0]),
                            CategoricalPanelField(; name = "sector", levels = ["A", "B"],
                                                  codes = [2, 1])])
    a = mk()
    @test vcat(a, mk(), mk()) === a
    b = AssetPanel(;
                   pf = [NumericPanelField(; name = "mcap", vals = [1.0, 3.0]),
                         CategoricalPanelField(; name = "sector", levels = ["A", "B"],
                                               codes = [2, 1])])
    @test_throws "part 3 holds values that differ" vcat(a, mk(), b)
end

@testset "A part whose schema differs is refused" begin
    T, N = 2, 3
    base = concat_test_panel(T, N)
    num(name = "mcap"; n = N) = NumericPanelField(; name = name, vals = ones(T, n))
    function catf(levels = ["Tech", "Energy", "Retail"]; n = N)
        return CategoricalPanelField(; name = "sector", levels = levels,
                                     codes = ones(Int, T, n))
    end
    function ten(; axis = "factor", labels = ["size", "value", "mom"],
                 groups = ["style", "style", "momentum"], n = N)
        return TensorPanelField(; name = "beta", axis = axis, labels = labels,
                                groups = groups, vals = ones(T, n, 3))
    end
    part(pf; n = N) = tv_panel(pf; T = T, N = n)
    # The schema of `base` itself joins.
    @test PO.panel_axes(vcat(base, part([num(), catf(), ten()]))) == (2T, N)

    static = AssetPanel(; pf = [NumericPanelField(; name = "mcap", vals = ones(N))])
    @test_throws "the first part is time-varying and part 2 is static" vcat(base, static)
    @test_throws "the first part is static and part 2 is time-varying" vcat(static, base)
    narrow = part([num(; n = 2), catf(; n = 2), ten(; n = 2)]; n = 2)
    @test_throws DimensionMismatch vcat(base, narrow)
    @test_throws "got 2 assets in part 2 against 3" vcat(base, narrow)
    msg = "the same Panel Fields in the same order"
    @test_throws msg vcat(base, part([num(), catf()]))
    @test_throws msg vcat(base, part([catf(), num(), ten()]))
    @test_throws msg vcat(base, part([num("size"), catf(), ten()]))
    kind = NumericPanelField(; name = "sector", vals = ones(T, N))
    msg = "is a `NumericPanelField` in part 2 and a `CategoricalPanelField`"
    @test_throws msg vcat(base, part([num(), kind, ten()]))
    msg = "the codes of the two parts name different categories"
    @test_throws msg vcat(base, part([num(), catf(["Energy", "Tech", "Retail"]), ten()]))
    msg = "the tensor Panel Field \"beta\""
    @test_throws msg vcat(base, part([num(), catf(), ten(; axis = "loading")]))
    @test_throws msg vcat(base,
                          part([num(), catf(), ten(; labels = ["size", "mom", "value"])]))
    @test_throws msg vcat(base, part([num(), catf(), ten(; groups = ["a", "b", "c"])]))
    @test_throws msg vcat(base, part([num(), catf(), ten(; groups = nothing)]))
    # The message names the position of the part.
    @test_throws "in part 3" vcat(base, base,
                                  part([num(), catf(), ten(; axis = "loading")]))
end

@testset "Two blocks of returns data join their panels with every Panel Field" begin
    T, N = 8, 4
    pnl = concat_test_panel(T, N)
    rd = ReturnsResult(; nx = ["A", "B", "C", "D"], X = randn(StableRNG(7), T, N),
                       pnl = pnl)
    joined = PO.vcat_observations(PO.port_opt_view(rd, 1:5, :),
                                  PO.port_opt_view(rd, 6:T, :))
    @test joined.X == rd.X
    @test same_panel(joined.pnl, pnl)
end
