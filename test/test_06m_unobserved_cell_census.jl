#=
The census of #1508: every consumer of an Asset Panel ignores the placeholder of an unobserved cell.

A numeric Panel Field keeps a finite placeholder in a blank cell that no fill reaches, and its
observed mask marks the cell `false` (ADR 0102). The placeholder is a storage convention, not an
observation. A consumer that reads it adds a fabricated value to its sample, and its answer then
changes with the placeholder. This file finds such a consumer, as test_06j finds a consumer that
reads an inactive cell.

**The blanks.** The fixture of test_06j holds almost no active unobserved cell, so
`census_blank` marks about one active cell in twenty of every Panel Field as unobserved. A tensor
Panel Field gets its blanks label by label, so a member can be observed at one label and not at
another, and the square field of the collapse fixture gets a mask that no pair of asset masks
gives. The blanks are the same in the clean and in the poisoned copy.

**The poison.** `census_unobserved_poison` copies a `ReturnsResult` and changes every active
unobserved cell of every Panel Field: a numeric or a tensor value becomes `1e6`, and a category
code moves to the next level. The observed mask stays `false`. Each case runs one consumer on
the clean and on the poisoned copy, and needs the two answers equal cell by cell, `NaN` pattern
included.

**The census.** The lists of test_06j: every function with a method whose signature names an
`AssetPanel`, and every concrete Descriptor Estimator, exposure estimator, forecast unit and
forecast target. Each has a case or an exemption that states why no unobserved cell reaches its
answer. The cases are those of test_06j, and the cases of `feature_matrix`, which reads the
observed mask here and holds `NaN` at an unobserved cell.
=#
include(joinpath(@__DIR__, "panel_census.jl"))

# The blanks: about one active cell in twenty of every Panel Field becomes unobserved, label by
# label on a tensor Panel Field. An inactive cell keeps its mask.
function census_blank_mask(o, amsk::AbstractMatrix{Bool}, sz::Tuple, rng)
    b = rand(rng, sz...) .< 0.05
    b .&= amsk
    return (isnothing(o) ? trues(sz) : BitArray(o)) .& .!b
end
function census_blank_field(f::NumericPanelField, amsk, rng)
    return NumericPanelField(f.name, f.vals,
                             census_blank_mask(f.omsk, amsk, size(amsk), rng))
end
function census_blank_field(f::CategoricalPanelField, amsk, rng)
    return CategoricalPanelField(f.name, f.levels, f.codes,
                                 census_blank_mask(f.omsk, amsk, size(amsk), rng))
end
function census_blank_field(f::TensorPanelField, amsk, rng)
    return TensorPanelField(f.name, f.axis, f.labels, f.groups, f.vals,
                            census_blank_mask(f.omsk, amsk, size(f.vals), rng))
end
function census_blank(rd::ReturnsResult)
    pnl = rd.pnl
    rng = StableRNG(1508)
    pf = [census_blank_field(f, pnl.amsk, rng) for f in pnl.pf]
    return ReturnsResult(; nx = rd.nx, X = rd.X, nf = rd.nf, F = rd.F, nb = rd.nb, B = rd.B,
                         ne = rd.ne, E = rd.E, ts = rd.ts, iv = rd.iv, ivpa = rd.ivpa,
                         pnl = AssetPanel(pf, pnl.amsk, pnl.emsk))
end

# The poison: every active unobserved cell of every Panel Field changes, and stays unobserved.
function census_unobserved_cells!(V::AbstractArray, o::AbstractArray{Bool},
                                  amsk::AbstractMatrix{Bool}, f)
    for k in CartesianIndices(o)
        if amsk[k[1], k[2]] && !o[k]
            V[k] = f(V[k])
        end
    end
    return V
end
census_unobserved_field(f::CENSUS_PO.AbstractPanelField, ::Any, ::Nothing) = f
census_unobserved_field(f, amsk) = census_unobserved_field(f, amsk, f.omsk)
function census_unobserved_field(f::NumericPanelField, amsk, o::AbstractArray{Bool})
    return NumericPanelField(f.name,
                             census_unobserved_cells!(Array(f.vals), o, amsk, Returns(1e6)),
                             o)
end
function census_unobserved_field(f::CategoricalPanelField, amsk, o::AbstractArray{Bool})
    nl = length(f.levels)
    return CategoricalPanelField(f.name, f.levels,
                                 census_unobserved_cells!(Array(f.codes), o, amsk,
                                                          c -> mod1(c + 1, nl)), o)
end
function census_unobserved_field(f::TensorPanelField, amsk, o::AbstractArray{Bool})
    return TensorPanelField(f.name, f.axis, f.labels, f.groups,
                            census_unobserved_cells!(Array(f.vals), o, amsk, Returns(1e6)),
                            o)
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
    @test any(f -> !isnothing(f.omsk) && any(rd.pnl.amsk .& .!f.omsk), rd.pnl.pf)
    cases = census_cases(rd)
    append!(cases,
            Any[((:feature_matrix,), "every Panel Field", r -> feature_matrix(r.pnl)),
                ((:feature_matrix,), "a level, a label and a mask, over rows",
                 r -> feature_matrix(r.pnl,
                                     ["industry", "loadings" => "y", "style1",
                                      "style1" => :observed]; rows = 5:30)),
                ((:feature_matrix,), "a zero in place of an unobserved cell",
                 r -> feature_matrix(r.pnl, ["style1", "loadings"]; unobserved = 0))])
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

# The answers of the three readers, on panels small enough to check by hand. Asset `y` is not
# observed at the last row of the time-varying panel, and the pair (`x`, `y`) of the square field
# is not observed.
@testset "The readers read the active observed cells (#1508)" begin
    nx = ["x", "y", "z"]
    o = trues(3, 3)
    o[3, 2] = false
    tv = AssetPanel(;
                    pf = [NumericPanelField(; name = "a", vals = [1.0 2 3; 4 5 6; 7 1e6 9],
                                            omsk = o)], amsk = trues(3, 3),
                    emsk = trues(3, 3))
    rd = ReturnsResult(; nx = nx, X = zeros(3, 3), pnl = tv)
    euc = CENSUS_PO.Distances.Euclidean()
    fd(alg) = FeatureDistance(; sel = ["a"], alg = alg, metric = euc)
    @testset "feature_matrix holds NaN at an unobserved cell" begin
        Z = feature_matrix(tv)
        @test isnan(Z[3, 2, 1])
        @test count(isnan, Z) == 1
        @test feature_matrix(tv; unobserved = nothing)[3, 2, 1] == 1e6
        @test feature_matrix(tv; unobserved = 0)[3, 2, 1] == 0
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
    @testset "A static asset with an unobserved cell is unreadable" begin
        st = AssetPanel(;
                        pf = [NumericPanelField(; name = "a", vals = [1.0, 1e6, 3.0],
                                                omsk = BitVector([1, 0, 1]))])
        rds = ReturnsResult(; nx = nx, X = zeros(3, 3), pnl = st)
        @test_throws "the assets [\"y\"]" distance(fd(LastObservation()), nothing, rds.X;
                                                   rd = rds)
        @test CENSUS_PO.feature_readable_mask(fd(AggregateFeatures()), nothing, rds) ==
              BitVector([1, 0, 1])
        rdv = CENSUS_PO.port_opt_view(rds, [1, 3])
        @test distance(fd(LastObservation()), nothing, rdv.X; rd = rdv) ≈ [0 2; 2 0]
    end
    W = [0.5 0.0; 0.5 0.0; 0.0 1.0]
    @testset "The panel collapse divides by the weight of the observed members" begin
        f = only(CENSUS_PO.collapse_asset_panel(tv, W, nx, RenormaliseActive()).pf)
        @test f.vals ≈ [1.5 3; 4.5 6; 7 9]
        @test f.omsk == trues(3, 2)
        g = only(CENSUS_PO.collapse_asset_panel(tv, W, nx, InactiveAsCash()).pf)
        @test g.vals ≈ [1.5 3; 4.5 6; 3.5 9]
    end
    @testset "A square field reads the observed pairs, and stays symmetric" begin
        os = trues(3, 3)
        os[1, 2] = os[2, 1] = false
        sq = AssetPanel(;
                        pf = [TensorPanelField(; name = "adj", axis = "asset", labels = nx,
                                               vals = [1.0 1e6 3; 1e6 5 6; 3 6 9],
                                               omsk = os)])
        f = only(CENSUS_PO.collapse_asset_panel(sq, W, nx, RenormaliseActive()).pf)
        # The block of x and y reads the pairs (x, x) and (y, y) alone.
        @test f.vals ≈ [3 4.5; 4.5 9]
        @test f.omsk == trues(2, 2)
    end
end
