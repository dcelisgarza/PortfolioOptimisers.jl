#=
The census of #1508: every consumer of an Asset Panel ignores the placeholder of an unobserved cell.

A numeric Panel Field keeps a finite placeholder in a blank cell that no fill reaches. Its observed
mask marks the cell `false`, and its placeholder mask marks it `true` (ADR 0102, #1631). A cell
that a fill policy wrote is `false` in both masks: it holds data. The placeholder is a storage
convention, not an observation. A consumer that reads it adds a fabricated value to its sample, and its answer then
changes with the placeholder. This file finds such a consumer, as test_06j finds a consumer that
reads an inactive cell.

**The blanks.** The fixture of test_06j holds almost no active unobserved cell, so
`census_blank` marks about one active cell in twenty of every Panel Field as a placeholder. A tensor
Panel Field gets its blanks label by label, so a member can be observed at one label and not at
another, and the square field of the collapse fixture gets a mask that no pair of asset masks
gives. The blanks are the same in the clean and in the poisoned copy.

**The poison.** `census_unobserved_poison` copies a `ReturnsResult` and changes every active
placeholder of every Panel Field: a numeric or a tensor value becomes `1e6`, and a category
code moves to the next level. The two masks stay as they are. Each case runs one consumer on
the clean and on the poisoned copy, and needs the two answers equal cell by cell, `NaN` pattern
included.

**The census.** The lists of test_06j: every function with a method whose signature names an
`AssetPanel`, and every concrete Descriptor Estimator, exposure estimator, forecast unit and
forecast target. Each has a case or an exemption that states why no unobserved cell reaches its
answer. The cases are those of test_06j, and the cases of `feature_matrix`, which reads the
observed mask here and holds `NaN` at an unobserved cell.
=#
include(joinpath(@__DIR__, "panel_census.jl"))

# The blanks: about one active cell in twenty of every Panel Field becomes a placeholder, label
# by label on a tensor Panel Field. An inactive cell keeps its masks.
function census_blank_masks(f, amsk::AbstractMatrix{Bool}, sz::Tuple, rng)
    b = rand(rng, sz...) .< 0.05
    b .&= amsk
    o = (isnothing(f.omsk) ? trues(sz) : BitArray(f.omsk)) .& .!b
    p = (isnothing(f.pmsk) ? falses(sz) : BitArray(f.pmsk)) .| b
    return o, p
end
function census_blank_field(f::NumericPanelField, amsk, rng)
    return NumericPanelField(f.name, f.vals,
                             census_blank_masks(f, amsk, size(amsk), rng)...)
end
function census_blank_field(f::CategoricalPanelField, amsk, rng)
    return CategoricalPanelField(f.name, f.levels, f.codes,
                                 census_blank_masks(f, amsk, size(amsk), rng)...)
end
function census_blank_field(f::TensorPanelField, amsk, rng)
    return TensorPanelField(f.name, f.axis, f.labels, f.groups, f.vals,
                            census_blank_masks(f, amsk, size(f.vals), rng)...)
end
function census_blank(rd::ReturnsResult)
    pnl = rd.pnl
    rng = StableRNG(1508)
    pf = [census_blank_field(f, pnl.amsk, rng) for f in pnl.pf]
    return ReturnsResult(; nx = rd.nx, X = rd.X, nf = rd.nf, F = rd.F, nb = rd.nb, B = rd.B,
                         ne = rd.ne, E = rd.E, ts = rd.ts, iv = rd.iv, ivpa = rd.ivpa,
                         pnl = AssetPanel(pf, pnl.amsk, pnl.emsk))
end

# The poison: every active placeholder of every Panel Field changes, and stays a placeholder.
function census_unobserved_cells!(V::AbstractArray, p::AbstractArray{Bool},
                                  amsk::AbstractMatrix{Bool}, f)
    for k in CartesianIndices(p)
        if amsk[k[1], k[2]] && p[k]
            V[k] = f(V[k])
        end
    end
    return V
end
census_unobserved_field(f::CENSUS_PO.AbstractPanelField, ::Any, ::Nothing) = f
census_unobserved_field(f, amsk) = census_unobserved_field(f, amsk, f.pmsk)
function census_unobserved_field(f::NumericPanelField, amsk, p::AbstractArray{Bool})
    return NumericPanelField(f.name,
                             census_unobserved_cells!(Array(f.vals), p, amsk, Returns(1e6)),
                             f.omsk, p)
end
function census_unobserved_field(f::CategoricalPanelField, amsk, p::AbstractArray{Bool})
    nl = length(f.levels)
    return CategoricalPanelField(f.name, f.levels,
                                 census_unobserved_cells!(Array(f.codes), p, amsk,
                                                          c -> mod1(c + 1, nl)), f.omsk, p)
end
function census_unobserved_field(f::TensorPanelField, amsk, p::AbstractArray{Bool})
    return TensorPanelField(f.name, f.axis, f.labels, f.groups,
                            census_unobserved_cells!(Array(f.vals), p, amsk, Returns(1e6)),
                            f.omsk, p)
end
function census_unobserved_poison(rd::ReturnsResult)
    pnl = rd.pnl
    pf = [census_unobserved_field(f, pnl.amsk) for f in pnl.pf]
    return ReturnsResult(; nx = rd.nx, X = rd.X, nf = rd.nf, F = rd.F, nb = rd.nb, B = rd.B,
                         ne = rd.ne, E = rd.E, ts = rd.ts, iv = rd.iv, ivpa = rd.ivpa,
                         pnl = AssetPanel(pf, pnl.amsk, pnl.emsk))
end

# The exemptions of test_06j, with the reads of the stored cells that differ here: the stack of
# `feature_matrix` has cases below, and the export of every stored cell beside its mask reads the
# placeholder by design.
const CENSUS_UNOBSERVED_EXEMPT = let d = copy(CENSUS_EXEMPT)
    delete!(d, :feature_matrix)
    d[:panel_feature_matrix] = "an export of every stored cell beside its observed-mask column"
    d
end

@testset "Every consumer of an Asset Panel ignores the placeholder of an unobserved cell (#1508)" begin
    rd, rdp = census_fixture(census_unobserved_poison, census_blank)
    @test any(f -> !isnothing(f.pmsk) && any(rd.pnl.amsk .& f.pmsk), rd.pnl.pf)
    cases = census_cases(rd)
    append!(cases,
            Any[((:feature_matrix,), "every Panel Field", r -> feature_matrix(r.pnl)),
                ((:feature_matrix,), "a level, a label and a mask, over rows",
                 r -> feature_matrix(r.pnl,
                                     ["industry", "loadings" => "y", "style1",
                                      "style1" => :observed]; rows = 5:30)),
                ((:feature_matrix,), "a zero in place of a placeholder",
                 r -> feature_matrix(r.pnl, ["style1", "loadings"]; placeholder = 0))])
    rdk, rdkp = census_collapse_fixture(rd, census_unobserved_poison, census_blank)
    collapse_cases = census_collapse_cases(rdk)
    @testset "The census names each consumer once" begin
        census_names_test(cases, collapse_cases, CENSUS_UNOBSERVED_EXEMPT)
    end
    @testset "$(join(names, ", ")): $(label)" for (names, label, f) in cases
        @test census_equal(f(rd), f(rdp))
    end
    @testset "$(join(names, ", ")): $(label)" for (names, label, f) in collapse_cases
        @test census_equal(f(rdk), f(rdkp))
    end
end

# The answers of the three readers, on panels small enough to check by hand. Asset `y` holds a
# placeholder at the last row of the time-varying panel, and the pair (`x`, `y`) of the square
# field holds a placeholder.
@testset "The readers read the active cells that hold data (#1508, #1631)" begin
    nx = ["x", "y", "z"]
    o = trues(3, 3)
    o[3, 2] = false
    tv = AssetPanel(;
                    pf = [NumericPanelField(; name = "a", vals = [1.0 2 3; 4 5 6; 7 1e6 9],
                                            omsk = o, pmsk = .!o)], amsk = trues(3, 3),
                    emsk = trues(3, 3))
    rd = ReturnsResult(; nx = nx, X = zeros(3, 3), pnl = tv)
    euc = CENSUS_PO.Distances.Euclidean()
    fd(alg) = FeatureDistance(; sel = ["a"], alg = alg, metric = euc)
    @testset "feature_matrix holds NaN at a placeholder" begin
        Z = feature_matrix(tv)
        @test isnan(Z[3, 2, 1])
        @test count(isnan, Z) == 1
        @test feature_matrix(tv; placeholder = nothing)[3, 2, 1] == 1e6
        @test feature_matrix(tv; placeholder = 0)[3, 2, 1] == 0
        # An integer panel with no observed mask cannot blank, and keeps its type.
        pint = AssetPanel(; pf = [NumericPanelField(; name = "i", vals = [1, 2])])
        @test eltype(feature_matrix(pint)) === Int
    end
    @testset "FeatureDistance reads the last readable row" begin
        # x at row 3 (7), y at row 2 (5), z at row 3 (9).
        D = distance(fd(LastObservation(; alg = LastActiveRow())), nothing, rd.X; rd = rd)
        @test D ≈ [0 2 2; 2 0 4; 2 4 0]
        @test_throws "the assets [\"y\"]" distance(fd(LastObservation()), nothing, rd.X;
                                                   rd = rd)
        @test CENSUS_PO.feature_readable_mask(fd(LastObservation()), nothing, rd) ==
              BitVector([1, 0, 1])
    end
    @testset "A static asset with a placeholder is unreadable" begin
        st = AssetPanel(;
                        pf = [NumericPanelField(; name = "a", vals = [1.0, 1e6, 3.0],
                                                omsk = BitVector([1, 0, 1]),
                                                pmsk = BitVector([0, 1, 0]))])
        rds = ReturnsResult(; nx = nx, X = zeros(3, 3), pnl = st)
        @test_throws "the assets [\"y\"]" distance(fd(LastObservation()), nothing, rds.X;
                                                   rd = rds)
        @test CENSUS_PO.feature_readable_mask(fd(AggregateFeatures()), nothing, rds) ==
              BitVector([1, 0, 1])
        rdv = CENSUS_PO.port_opt_view(rds, [1, 3])
        @test distance(fd(LastObservation()), nothing, rdv.X; rd = rdv) ≈ [0 2; 2 0]
    end
    W = [0.5 0.0; 0.5 0.0; 0.0 1.0]
    @testset "The panel collapse divides by the weight of the members that hold data" begin
        f = only(CENSUS_PO.collapse_asset_panel(tv, W, nx, RenormaliseActive()).pf)
        @test f.vals ≈ [1.5 3; 4.5 6; 7 9]
        @test f.omsk == trues(3, 2)
        @test f.pmsk == falses(3, 2)
        g = only(CENSUS_PO.collapse_asset_panel(tv, W, nx, InactiveAsCash()).pf)
        @test g.vals ≈ [1.5 3; 4.5 6; 3.5 9]
    end
    @testset "A square field reads the observed pairs, and stays symmetric" begin
        os = trues(3, 3)
        os[1, 2] = os[2, 1] = false
        sq = AssetPanel(;
                        pf = [TensorPanelField(; name = "adj", axis = "asset", labels = nx,
                                               vals = [1.0 1e6 3; 1e6 5 6; 3 6 9],
                                               omsk = os, pmsk = .!os)])
        f = only(CENSUS_PO.collapse_asset_panel(sq, W, nx, RenormaliseActive()).pf)
        # The block of x and y reads the pairs (x, x) and (y, y) alone.
        @test f.vals ≈ [3 4.5; 4.5 9]
        @test f.omsk == trues(2, 2)
    end
end

# #1631: a fill policy writes data, and only a blank that no fill reaches holds a placeholder.
@testset "A filled cell is data, and only a placeholder is skipped (#1631)" begin
    nx = ["x", "y", "z"]
    euc = CENSUS_PO.Distances.Euclidean()
    @testset "The builder marks the placeholders of a directional fill alone" begin
        p = asset_panel([NumericPanelInput(; name = "a", vals = [NaN 1.0; 2.0 NaN; NaN 3.0],
                                           alg = ForwardPanelFill())])
        f = only(p.pf)
        @test f.vals == [0.0 1.0; 2.0 1.0; 2.0 3.0]
        @test f.omsk == Bool[0 1; 1 0; 0 1]
        @test f.pmsk == Bool[1 0; 0 0; 0 0]
        # A blank past the run of `lim` is a placeholder too, and so is a trailing blank of a
        # backward fill.
        q = asset_panel([NumericPanelInput(; name = "a", vals = [1.0; NaN; NaN; 4.0; NaN;;],
                                           alg = ForwardPanelFill(; lim = 1)),
                         NumericPanelInput(; name = "b", vals = [1.0; NaN; NaN; 4.0; NaN;;],
                                           alg = BackwardPanelFill())]; amsk = trues(5, 1))
        @test vec(q.pf[1].pmsk) == Bool[0, 0, 1, 0, 0]
        @test vec(q.pf[2].pmsk) == Bool[0, 0, 0, 0, 1]
        # A placeholder never crosses an inactive stretch: each listing starts its own walk.
        r = asset_panel([NumericPanelInput(; name = "a", vals = [1.0; NaN; NaN; NaN;;],
                                           alg = ForwardPanelFill())];
                        amsk = reshape(Bool[1, 1, 0, 1], 4, 1))
        @test vec(r.pf[1].pmsk) == Bool[0, 0, 1, 1]
        # A categorical directional fill marks its placeholders the same way.
        c = asset_panel([CategoricalPanelInput(; name = "s",
                                               vals = [missing; "a"; missing;;],
                                               alg = ForwardPanelFill(; val = "z"))];
                        amsk = trues(3, 1))
        @test vec(c.pf[1].pmsk) == Bool[1, 0, 0]
        # ConstantPanelFill writes the value that the absence means: no placeholder.
        s = asset_panel([NumericPanelInput(; name = "a", vals = [1.0, NaN, 3.0],
                                           alg = ConstantPanelFill(; val = 0.0))])
        @test only(s.pf).omsk == BitVector([1, 0, 1])
        @test isnothing(only(s.pf).pmsk)
    end
    @testset "The constructor refuses a placeholder at an observed cell" begin
        @test_throws ArgumentError NumericPanelField(; name = "a", vals = [1.0, 2.0],
                                                     omsk = BitVector([1, 0]),
                                                     pmsk = BitVector([1, 0]))
        @test_throws ArgumentError NumericPanelField(; name = "a", vals = [1.0, 2.0],
                                                     pmsk = BitVector([0, 1]))
        @test_throws DimensionMismatch NumericPanelField(; name = "a", vals = [1.0, 2.0],
                                                         omsk = BitVector([1, 0]),
                                                         pmsk = BitVector([0, 1, 0]))
        @test_throws ArgumentError CategoricalPanelField(; name = "s", levels = ["a"],
                                                         codes = [1, 1],
                                                         omsk = BitVector([1, 0]),
                                                         pmsk = BitVector([1, 0]))
        @test_throws ArgumentError TensorPanelField(; name = "t", axis = "k",
                                                    labels = ["u"], vals = [1.0; 2.0;;],
                                                    omsk = Bool[1; 0;;],
                                                    pmsk = Bool[1; 0;;])
    end
    @testset "FeatureDistance reads a filled cell (the docs example of #1631)" begin
        st = asset_panel([NumericPanelInput(; name = "a", vals = [1.0, NaN, 3.0],
                                            alg = ConstantPanelFill(; val = 0.0))])
        rds = ReturnsResult(; nx = nx, X = zeros(3, 3), pnl = st)
        fd = FeatureDistance(; sel = ["a"], metric = euc)
        @test distance(fd, nothing, rds.X; rd = rds) ≈ [0 1 2; 1 0 3; 2 3 0]
        @test isnothing(CENSUS_PO.feature_readable_mask(fd, nothing, rds))
        @test feature_matrix(st) == reshape([1.0, 0.0, 3.0], 3, 1)
        # A time-varying filled cell is read at the last row as well.
        o = trues(3, 3)
        o[3, 2] = false
        tv = AssetPanel(;
                        pf = [NumericPanelField(; name = "a",
                                                vals = [1.0 2 3; 4 5 6; 7 5 9], omsk = o)],
                        amsk = trues(3, 3), emsk = trues(3, 3))
        rdt = ReturnsResult(; nx = nx, X = zeros(3, 3), pnl = tv)
        @test distance(FeatureDistance(; sel = ["a"], metric = euc), nothing, rdt.X;
                       rd = rdt) ≈ [0 2 2; 2 0 4; 2 4 0]
    end
    @testset "The panel collapse reads a filled cell, and marks a cell with no data" begin
        W = [0.5 0.0; 0.5 0.0; 0.0 1.0]
        o = trues(3, 3)
        o[3, 2] = false
        filled = AssetPanel(;
                            pf = [NumericPanelField(; name = "a",
                                                    vals = [1.0 2 3; 4 5 6; 7 1 9],
                                                    omsk = o)], amsk = trues(3, 3),
                            emsk = trues(3, 3))
        f = only(CENSUS_PO.collapse_asset_panel(filled, W, nx, RenormaliseActive()).pf)
        @test f.vals ≈ [1.5 3; 4.5 6; 4 9]
        @test isnothing(f.pmsk)
        # Both members of the first sub-portfolio hold a placeholder at row 3.
        o2 = trues(3, 3)
        o2[3, 1:2] .= false
        held = AssetPanel(;
                          pf = [NumericPanelField(; name = "a",
                                                  vals = [1.0 2 3; 4 5 6; 7 1 9], omsk = o2,
                                                  pmsk = .!o2)], amsk = trues(3, 3),
                          emsk = trues(3, 3))
        g = only(CENSUS_PO.collapse_asset_panel(held, W, nx, RenormaliseActive()).pf)
        @test g.pmsk == Bool[0 0; 0 0; 1 0]
        @test g.omsk == Bool[1 1; 1 1; 0 1]
    end
    @testset "A lift, a view and a concatenation keep the masks" begin
        f = NumericPanelField(; name = "a", vals = [1.0, 0.0, 3.0],
                              omsk = BitVector([1, 0, 0]), pmsk = BitVector([0, 0, 1]))
        l = CENSUS_PO.panel_field_lift(f, 2)
        @test l.omsk == Bool[1 0 0; 1 0 0]
        @test l.pmsk == Bool[0 0 1; 0 0 1]
        # A static input that ConstantPanelFill filled keeps its observed mask in the lift.
        p = asset_panel([NumericPanelInput(; name = "a", vals = [1.0, NaN],
                                           alg = ConstantPanelFill())]; amsk = trues(2, 2))
        @test only(p.pf).omsk == Bool[1 0; 1 0]
        v = CENSUS_PO.panel_field_view(l, 1:1, [2, 3], nothing)
        @test v.pmsk == Bool[0 1;]
        a = AssetPanel(; pf = [l], amsk = trues(2, 3), emsk = trues(2, 3))
        b = AssetPanel(; pf = [NumericPanelField(; name = "a", vals = ones(1, 3))],
                       amsk = trues(1, 3), emsk = trues(1, 3))
        @test only(vcat(a, b).pf).pmsk == Bool[0 0 1; 0 0 1; 0 0 0]
    end
end
