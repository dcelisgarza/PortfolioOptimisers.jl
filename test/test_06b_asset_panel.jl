#=
The Asset Panel carrier, its builder and the view contract (issue #664, map #643, #809).

A Panel Field owns its values, so the panel *is* the feature data and nothing else on a
carrier holds a feature matrix. The file tests four seams: the three Panel Field types and
their rules, the panel's own invariant in the static and the time-varying shape, the builder
that resolves every blank, and the two `port_opt_view` arities that carry the panel in step
with `X`. The derived Feature Matrix is checked where it is derived, not where it is stored,
because nothing stores it.
=#
@testset "Panel Field types" begin
    num = NumericPanelField(; name = "mcap", vals = [1.0, 2.0, 3.0])
    @test num.name == "mcap"
    @test num.omsk === nothing
    @test PortfolioOptimisers.panel_field_labels(num) == ["mcap"]
    @test PortfolioOptimisers.panel_field_axes(num) == (3,)
    @test PortfolioOptimisers.panel_field_observed_labels(num) == ["mcap::observed"]

    cat = CategoricalPanelField(; name = "sector", levels = ["Tech", "Energy"],
                                codes = [1, 2, 1])
    @test cat.levels == ["Tech", "Energy"]
    @test cat.codes == [1, 2, 1]
    @test PortfolioOptimisers.panel_field_labels(cat) == ["sector=Tech", "sector=Energy"]
    @test PortfolioOptimisers.panel_field_axes(cat) == (3,)

    ten = TensorPanelField(; name = "beta", axis = "factor", labels = ["size", "value"],
                           groups = ["style", "style"], vals = [1.0 2.0; 3.0 4.0])
    @test ten.axis == "factor"
    @test ten.groups == ["style", "style"]
    @test PortfolioOptimisers.panel_field_labels(ten) == ["beta=size", "beta=value"]
    @test PortfolioOptimisers.panel_field_axes(ten) == (2,)
    @test PortfolioOptimisers.panel_field_observed_labels(ten) ==
          ["beta=size::observed", "beta=value::observed"]

    # A time-varying Panel Field prepends the observation axis to each of the three.
    @test PortfolioOptimisers.panel_field_axes(NumericPanelField(; name = "a",
                                                                 vals = ones(4, 3))) ==
          (4, 3)
    @test PortfolioOptimisers.panel_field_axes(TensorPanelField(; name = "b", axis = "f",
                                                                labels = ["a"],
                                                                vals = ones(4, 3, 1))) ==
          (4, 3)

    # The name, the label and the shape rules, each raising on its own condition.
    @test_throws IsEmptyError NumericPanelField(; name = "", vals = [1.0])
    @test_throws IsEmptyError NumericPanelField(; name = "a", vals = Float64[])
    @test_throws DimensionMismatch NumericPanelField(; name = "a", vals = ones(2, 2, 2))
    @test_throws DimensionMismatch NumericPanelField(; name = "a", vals = [1.0, 2.0],
                                                     omsk = [true])
    @test_throws IsEmptyError CategoricalPanelField(; name = "a", levels = String[],
                                                    codes = [1])
    @test_throws ArgumentError CategoricalPanelField(; name = "a", levels = ["a", ""],
                                                     codes = [1])
    @test_throws ArgumentError CategoricalPanelField(; name = "a", levels = ["a", "a"],
                                                     codes = [1])
    # A code indexes the levels, so it lies in `1:length(levels)`.
    @test_throws DomainError CategoricalPanelField(; name = "a", levels = ["x"],
                                                   codes = [2])
    @test_throws DomainError CategoricalPanelField(; name = "a", levels = ["x"],
                                                   codes = [0])
    @test_throws IsEmptyError TensorPanelField(; name = "a", axis = "", labels = ["a"],
                                               vals = ones(2, 1))
    @test_throws ArgumentError TensorPanelField(; name = "a", axis = "f",
                                                labels = ["a", "a"], vals = ones(2, 2))
    @test_throws DimensionMismatch TensorPanelField(; name = "a", axis = "f",
                                                    labels = ["a", "b"], groups = ["g"],
                                                    vals = ones(2, 2))
    @test_throws DimensionMismatch TensorPanelField(; name = "a", axis = "f",
                                                    labels = ["a", "b"], vals = ones(2, 3))
    @test_throws DimensionMismatch TensorPanelField(; name = "a", axis = "f",
                                                    labels = ["a"], vals = ones(2))
end
@testset "AssetPanel, its two shapes and its invariant" begin
    # The static shape: no observation axis, and no mask.
    st = AssetPanel(;
                    pf = [NumericPanelField(; name = "mcap", vals = [1.0, 2.0, 3.0]),
                          CategoricalPanelField(; name = "sector", levels = ["T", "E"],
                                                codes = [1, 2, 1])])
    @test PortfolioOptimisers.panel_is_static(st)
    @test st.amsk === nothing
    @test st.emsk === nothing
    @test isa(st, AssetPanel{<:Any, Nothing, Nothing})

    # The time-varying shape: both masks, matching the Panel Fields.
    amsk = [true true true; true true false]
    emsk = [true true false; true false false]
    pf = [NumericPanelField(; name = "mcap", vals = [1.0 2.0 3.0; 4.0 5.0 6.0],
                            omsk = [true true false; true true true]),
          CategoricalPanelField(; name = "sector", levels = ["T", "E"],
                                codes = [1 2 1; 2 2 1])]
    pnl = AssetPanel(; pf = pf, amsk = amsk, emsk = emsk)
    @test !PortfolioOptimisers.panel_is_static(pnl)
    @test length(pnl.pf) == 2
    @test PortfolioOptimisers.panel_field(pnl, "sector").levels == ["T", "E"]
    @test_throws KeyError PortfolioOptimisers.panel_field(pnl, "sectr")

    # An empty panel, a repeated name, and two Panel Fields of different shapes.
    @test_throws IsEmptyError AssetPanel(; pf = PortfolioOptimisers.AbstractPanelField[])
    @test_throws ArgumentError AssetPanel(;
                                          pf = [NumericPanelField(; name = "a",
                                                                  vals = [1.0]),
                                                NumericPanelField(; name = "a",
                                                                  vals = [2.0])])
    @test_throws DimensionMismatch AssetPanel(;
                                              pf = [NumericPanelField(; name = "a",
                                                                      vals = [1.0]),
                                                    NumericPanelField(; name = "b",
                                                                      vals = [1.0, 2.0])])
    # The masks are `nothing` if and only if the Panel Fields are static.
    @test_throws DimensionMismatch AssetPanel(;
                                              pf = [NumericPanelField(; name = "a",
                                                                      vals = ones(2, 3))])
    @test_throws DimensionMismatch AssetPanel(;
                                              pf = [NumericPanelField(; name = "a",
                                                                      vals = [1.0, 2.0])],
                                              amsk = trues(2, 2), emsk = trues(2, 2))
    # Given together or not at all.
    @test_throws DimensionMismatch AssetPanel(;
                                              pf = [NumericPanelField(; name = "a",
                                                                      vals = ones(2, 3))],
                                              amsk = trues(2, 3))
    @test_throws DimensionMismatch AssetPanel(; pf = pf, amsk = trues(2, 3),
                                              emsk = trues(2, 2))
    # The subset invariant is checked, not coerced.
    @test_throws ArgumentError AssetPanel(;
                                          pf = [NumericPanelField(; name = "a",
                                                                  vals = ones(2, 2))],
                                          amsk = [true false; true true],
                                          emsk = [true true; true true])
end
@testset "The derived Feature Matrix, and its inverse" begin
    pnl = AssetPanel(;
                     pf = [NumericPanelField(; name = "mcap", vals = [1.0 2.0; 3.0 4.0],
                                             omsk = [true false; true true]),
                           CategoricalPanelField(; name = "sector", levels = ["T", "E"],
                                                 codes = [1 2; 1 2]),
                           TensorPanelField(; name = "beta", axis = "factor",
                                            labels = ["size"],
                                            vals = reshape(1.0:4.0, 2, 2, 1))],
                     amsk = trues(2, 2), emsk = trues(2, 2))
    nz, Z = panel_feature_matrix(pnl)
    @test nz == ["mcap", "mcap::observed", "sector=T", "sector=E", "beta=size"]
    @test nz == PortfolioOptimisers.panel_feature_names(pnl)
    @test size(Z) == (2, 2, 5)
    @test Z[:, :, 1] == [1.0 2.0; 3.0 4.0]
    @test Z[:, :, 2] == [1.0 0.0; 1.0 1.0]
    # One-hot: exactly one 1 per cell across the level columns.
    @test Z[:, :, 3] == [1.0 0.0; 1.0 0.0]
    @test Z[:, :, 4] == [0.0 1.0; 0.0 1.0]
    @test Z[:, :, 5] == [1.0 3.0; 2.0 4.0]
    @test all(isfinite, Z)

    # A static panel derives an `assets × features` matrix.
    snz, sZ = panel_feature_matrix(AssetPanel(;
                                              pf = [NumericPanelField(; name = "mcap",
                                                                      vals = [1.0, 2.0]),
                                                    TensorPanelField(; name = "beta",
                                                                     axis = "f",
                                                                     labels = ["a", "b"],
                                                                     vals = [1.0 2.0;
                                                                             3.0 4.0],
                                                                     omsk = trues(2, 2))]))
    @test snz == ["mcap", "beta=a", "beta=b", "beta=a::observed", "beta=b::observed"]
    @test size(sZ) == (2, 5)
    @test sZ[:, 1] == [1.0, 2.0]
    @test sZ[:, 4] == [1.0, 1.0]

    # `feature_matrix_panel` is the exact inverse for a matrix of named columns.
    rt = feature_matrix_panel(nz, Z)
    @test panel_feature_matrix(rt) == (nz, Z)
    @test all(f -> isa(f, NumericPanelField), rt.pf)
    @test rt.amsk == trues(2, 2)
    srt = feature_matrix_panel(snz, sZ)
    @test PortfolioOptimisers.panel_is_static(srt)
    @test panel_feature_matrix(srt) == (snz, sZ)

    @test_throws ArgumentError feature_matrix_panel(["a", "a"], ones(2, 2))
    @test_throws DimensionMismatch feature_matrix_panel(["a"], ones(2, 2))
    @test_throws DimensionMismatch feature_matrix_panel(["a", "b"], ones(2, 2);
                                                        amsk = trues(2, 2),
                                                        emsk = trues(2, 2))
    @test PortfolioOptimisers.panel_feature_names(nothing) === nothing
end
@testset "AssetPanel views" begin
    amsk = [true true true; true true false]
    emsk = [true true false; true false false]
    pf = [NumericPanelField(; name = "mcap", vals = [1.0 2.0 3.0; 4.0 5.0 6.0],
                            omsk = [true true false; true true true]),
          CategoricalPanelField(; name = "sector", levels = ["T", "E"],
                                codes = [1 2 1; 2 2 1])]
    pnl = AssetPanel(; pf = pf, amsk = amsk, emsk = emsk)

    # The asset arity keeps every observation and slices the assets out of every value.
    v = PortfolioOptimisers.port_opt_view(pnl, 2:3)
    @test v.amsk == amsk[:, 2:3]
    @test v.emsk == emsk[:, 2:3]
    @test isa(v.amsk, SubArray)
    @test PortfolioOptimisers.panel_field(v, "mcap").vals == [2.0 3.0; 5.0 6.0]
    @test PortfolioOptimisers.panel_field(v, "mcap").omsk == [true false; true true]
    @test PortfolioOptimisers.panel_field(v, "sector").codes == [2 1; 2 1]
    @test PortfolioOptimisers.panel_field(v, "sector").levels === pf[2].levels

    # The observations-then-assets arity slices both axes.
    w = PortfolioOptimisers.port_opt_view(pnl, 1:1, 2:3)
    @test w.amsk == amsk[1:1, 2:3]
    @test w.emsk == emsk[1:1, 2:3]
    @test PortfolioOptimisers.panel_field(w, "mcap").vals == [2.0 3.0]

    # A static panel ignores the observation index and keeps its masks `nothing`.
    st = AssetPanel(;
                    pf = [NumericPanelField(; name = "mcap", vals = [1.0, 2.0, 3.0]),
                          TensorPanelField(; name = "beta", axis = "f", labels = ["a"],
                                           vals = reshape([1.0, 2.0, 3.0], 3, 1))])
    sv = PortfolioOptimisers.port_opt_view(st, 1:1, 2:3)
    @test PortfolioOptimisers.panel_is_static(sv)
    @test PortfolioOptimisers.panel_field(sv, "mcap").vals == [2.0, 3.0]
    @test PortfolioOptimisers.panel_field(sv, "beta").vals == reshape([2.0, 3.0], 2, 1)

    # `sq` says the Panel Fields *are* the assets, so the same index cuts both.
    sq = feature_matrix_panel(["A", "B", "C"], reshape(collect(1.0:18.0), 2, 3, 3))
    sqv = PortfolioOptimisers.port_opt_view(sq, 1:2, 2:3, true)
    @test PortfolioOptimisers.panel_feature_names(sqv) == ["B", "C"]
    @test length(sqv.pf) == 2
    @test PortfolioOptimisers.panel_field(sqv, "B").vals == [9.0 11.0; 10.0 12.0]
end
@testset "Fill policies" begin
    @test PortfolioOptimisers.is_panel_blank(missing)
    @test PortfolioOptimisers.is_panel_blank(NaN)
    @test !PortfolioOptimisers.is_panel_blank(1.0)
    @test !PortfolioOptimisers.is_panel_blank("a")

    @test ConstantPanelFill().val == 0.0
    @test ForwardPanelFill().lim === nothing
    @test BackwardPanelFill(; lim = 2).lim == 2
    @test_throws DomainError ConstantPanelFill(; val = Inf)
    @test_throws DomainError ForwardPanelFill(; val = NaN)
    @test_throws DomainError ForwardPanelFill(; lim = 0)
    @test_throws DomainError BackwardPanelFill(; lim = -1)

    v = [missing, 1.0, missing, missing, 2.0, missing]
    @test PortfolioOptimisers.panel_fill(ConstantPanelFill(; val = -1.0), v, "f") ==
          [-1.0, 1.0, -1.0, -1.0, 2.0, -1.0]
    # A forward fill carries the last observed value, and a leading blank falls to `val`.
    @test PortfolioOptimisers.panel_fill(ForwardPanelFill(; val = 0.0), v, "f") ==
          [0.0, 1.0, 1.0, 1.0, 2.0, 2.0]
    # The limit stops the carry after that many consecutive blanks.
    @test PortfolioOptimisers.panel_fill(ForwardPanelFill(; val = 0.0, lim = 1), v, "f") ==
          [0.0, 1.0, 1.0, 0.0, 2.0, 2.0]
    # A backward fill looks the other way, and its trailing blank falls to `val`.
    @test PortfolioOptimisers.panel_fill(BackwardPanelFill(; val = 0.0), v, "f") ==
          [1.0, 1.0, 2.0, 2.0, 2.0, 0.0]

    # NoPanelFill refuses a blank and passes a complete column through.
    @test_throws ArgumentError PortfolioOptimisers.panel_fill(NoPanelFill(), v, "f")
    @test PortfolioOptimisers.panel_fill(NoPanelFill(), [1.0, 2.0], "f") == [1.0, 2.0]
end
@testset "Panel Field inputs and their resolution" begin
    @test_throws IsEmptyError NumericPanelInput(; name = "", vals = ones(2, 2))
    @test_throws IsEmptyError NumericPanelInput(; name = "a", vals = zeros(0, 0))
    @test_throws DimensionMismatch NumericPanelInput(; name = "a", vals = ones(2, 2, 2))
    @test_throws ArgumentError CategoricalPanelInput(; name = "a", vals = ["x" "y"],
                                                     levels = ["a", "a"])
    @test_throws DimensionMismatch TensorPanelInput(; name = "a", vals = ones(2, 2, 2),
                                                    axis = "f", labels = ["a"])
    @test_throws DimensionMismatch TensorPanelInput(; name = "a", vals = ones(2),
                                                    axis = "f", labels = ["a"])

    num = NumericPanelInput(; name = "mcap", vals = [missing 2.0; 3.0 4.0],
                            alg = ForwardPanelFill(; val = -1.0))
    @test !PortfolioOptimisers.panel_input_is_static(num)
    v, o = PortfolioOptimisers.panel_resolve(num)
    @test v == [-1.0 2.0; 3.0 4.0]
    @test o == [false true; true true]
    nf = PortfolioOptimisers.panel_input_field(num, v, o)
    @test isa(nf, NumericPanelField)
    @test nf.omsk == o

    cats = CategoricalPanelInput(; name = "sector", vals = [missing "E"; "T" "E"],
                                 alg = ConstantPanelFill(; val = "U"))
    cv, co = PortfolioOptimisers.panel_resolve(cats)
    @test cv == ["U" "E"; "T" "E"]
    @test co == [false true; true true]
    # The levels are read off the resolved labels, so the fill's own level gets a code.
    cf = PortfolioOptimisers.panel_input_field(cats, cv, co)
    @test cf.levels == ["E", "T", "U"]
    @test cf.codes == [3 1; 2 1]
    # A label outside the declared levels is refused, with a suggestion.
    bad = CategoricalPanelInput(; name = "sector", vals = ["E" "E"; "T" "E"],
                                levels = ["T"])
    bv, bo = PortfolioOptimisers.panel_resolve(bad)
    @test_throws ArgumentError PortfolioOptimisers.panel_input_field(bad, bv, bo)

    ten = TensorPanelInput(; name = "beta", vals = reshape([1.0, NaN, 3.0, 4.0], 2, 1, 2),
                           axis = "factor", labels = ["a", "b"],
                           alg = ConstantPanelFill(; val = 0.0))
    tv, to = PortfolioOptimisers.panel_resolve(ten)
    @test tv == reshape([1.0, 0.0, 3.0, 4.0], 2, 1, 2)
    @test to == reshape([true, false, true, true], 2, 1, 2)
    @test PortfolioOptimisers.panel_input_field(ten, tv, to).labels == ["a", "b"]

    # A static input carries one axis fewer, and resolves cell by cell.
    sn = NumericPanelInput(; name = "mcap", vals = [1.0, missing, 3.0],
                           alg = ConstantPanelFill(; val = 0.0))
    @test PortfolioOptimisers.panel_input_is_static(sn)
    sv, so = PortfolioOptimisers.panel_resolve(sn)
    @test sv == [1.0, 0.0, 3.0]
    @test so == [true, false, true]
    st = TensorPanelInput(; name = "beta", vals = [1.0 2.0; 3.0 4.0], axis = "f",
                          labels = ["a", "b"])
    @test PortfolioOptimisers.panel_input_is_static(st)
    sc = CategoricalPanelInput(; name = "sector", vals = ["T", "E", "T"])
    @test PortfolioOptimisers.panel_input_is_static(sc)
    scv, sco = PortfolioOptimisers.panel_resolve(sc)
    @test PortfolioOptimisers.panel_input_field(sc, scv, sco).codes == [2, 1, 2]

    # An infinity is not a blank, so no policy resolves it and the field is named.
    @test_throws IsNonFiniteError PortfolioOptimisers.panel_resolve(NumericPanelInput(;
                                                                                      name = "bad",
                                                                                      vals = [Inf 1.0;
                                                                                              2.0 3.0]))
    # A directional fill has no observation axis to carry a value along on a static input.
    @test_throws ArgumentError PortfolioOptimisers.assert_panel_input_fill(NumericPanelInput(;
                                                                                             name = "a",
                                                                                             vals = [1.0,
                                                                                                     2.0],
                                                                                             alg = ForwardPanelFill()))
    @test isnothing(PortfolioOptimisers.assert_panel_input_fill(sn))
end
@testset "asset_panel, the build seam" begin
    pnl = asset_panel([NumericPanelInput(; name = "mcap", vals = [1.0 missing; 3.0 4.0],
                                         alg = ForwardPanelFill(; val = 0.0)),
                       CategoricalPanelInput(; name = "sector", vals = ["T" "E"; "T" "E"],
                                             levels = ["T", "E"]),
                       TensorPanelInput(; name = "beta",
                                        vals = reshape(collect(1.0:4.0), 2, 2, 1),
                                        axis = "factor", labels = ["size"])])
    nz, Z = panel_feature_matrix(pnl)
    @test nz == ["mcap", "mcap::observed", "sector=T", "sector=E", "beta=size"]
    @test size(Z) == (2, 2, 5)
    @test Z[:, :, 1] == [1.0 0.0; 3.0 4.0]
    @test Z[:, :, 2] == [1.0 0.0; 1.0 1.0]
    @test Z[:, :, 5] == [1.0 3.0; 2.0 4.0]
    @test all(isfinite, Z)
    # A NoPanelFill field carries no observed mask.
    @test PortfolioOptimisers.panel_field(pnl, "sector").omsk === nothing
    @test PortfolioOptimisers.panel_field(pnl, "mcap").omsk == [true false; true true]
    @test pnl.amsk == trues(2, 2)
    @test pnl.emsk == trues(2, 2)

    # The masks are taken as given, and the subset invariant reaches them.
    pnl2 = asset_panel([NumericPanelInput(; name = "a", vals = ones(2, 2))];
                       amsk = [true false; true true], emsk = [true false; false true])
    @test pnl2.amsk == [true false; true true]
    @test_throws ArgumentError asset_panel([NumericPanelInput(; name = "a",
                                                              vals = ones(2, 2))];
                                           amsk = [true false; true true],
                                           emsk = [true true; true true])

    # The static entry: the rank of the raw values, no mask, and no directional fill.
    sp = asset_panel([NumericPanelInput(; name = "mcap", vals = [1.0, missing, 3.0],
                                        alg = ConstantPanelFill(; val = 0.0)),
                      CategoricalPanelInput(; name = "sector", vals = ["T", "E", "T"])])
    @test PortfolioOptimisers.panel_is_static(sp)
    @test panel_feature_matrix(sp)[1] == ["mcap", "mcap::observed", "sector=E", "sector=T"]
    @test panel_feature_matrix(sp)[2][:, 1] == [1.0, 0.0, 3.0]
    @test_throws DimensionMismatch asset_panel([NumericPanelInput(; name = "a",
                                                                  vals = [1.0, 2.0])];
                                               amsk = trues(2, 2), emsk = trues(2, 2))
    @test_throws ArgumentError asset_panel([NumericPanelInput(; name = "a",
                                                              vals = [1.0, 2.0],
                                                              alg = BackwardPanelFill())])

    @test_throws IsEmptyError asset_panel(PortfolioOptimisers.AbstractPanelFieldInput[])
    @test_throws ArgumentError asset_panel([NumericPanelInput(; name = "a",
                                                              vals = ones(2, 2)),
                                            NumericPanelInput(; name = "a",
                                                              vals = ones(2, 2))])
    @test_throws DimensionMismatch asset_panel([NumericPanelInput(; name = "a",
                                                                  vals = ones(2, 2)),
                                                NumericPanelInput(; name = "b",
                                                                  vals = ones(3, 2))])
end
@testset "ReturnsResult carries the panel through both views" begin
    pnl = asset_panel([NumericPanelInput(; name = "mcap",
                                         vals = [1.0 2.0 3.0; 4.0 missing 6.0],
                                         alg = ForwardPanelFill(; val = 0.0))];
                      amsk = [true true false; true true true],
                      emsk = [true false false; true true true])
    nx = ["A", "B", "C"]
    X = [0.1 0.2 0.3; 0.4 0.5 0.6]
    rd = ReturnsResult(; nx = nx, X = X,
                       ts = [Dates.Date(2020, 1, 1), Dates.Date(2020, 1, 2)], pnl = pnl)
    @test rd.pnl.amsk == [true true false; true true true]
    @test size(panel_feature_matrix(rd.pnl)[2]) == (2, 3, 2)

    # The asset arity slices the assets out of every value, and keeps the feature axis.
    va = PortfolioOptimisers.port_opt_view(rd, 2:3)
    @test va.pnl.amsk == [true false; true true]
    @test panel_feature_matrix(va.pnl)[1] == panel_feature_matrix(rd.pnl)[1]
    @test PortfolioOptimisers.panel_field(va.pnl, "mcap").vals == [2.0 3.0; 2.0 6.0]

    # The observations-then-assets arity slices both axes.
    vb = PortfolioOptimisers.port_opt_view(rd, 2:2, 1:2)
    @test vb.pnl.amsk == [true true]
    @test vb.pnl.emsk == [true true]
    @test size(panel_feature_matrix(vb.pnl)[2]) == (1, 2, 2)

    # The panel describes the carrier's universe, so its axes must be the carrier's.
    @test_throws DimensionMismatch ReturnsResult(; nx = ["A", "B"], X = [0.1 0.2; 0.3 0.4],
                                                 pnl = pnl)
    @test_throws DimensionMismatch ReturnsResult(; nx = nx, X = [0.1 0.2 0.3], pnl = pnl)
    @test_throws IsNothingError ReturnsResult(; pnl = pnl)
    # A static panel binds the asset axis alone, so it needs no observation count.
    stat = feature_matrix_panel(["f1", "f2"], [1.0 2.0; 3.0 4.0; 5.0 6.0])
    @test ReturnsResult(; nx = nx, X = X, pnl = stat).pnl === stat
    @test_throws IsNothingError ReturnsResult(; nx = nx, pnl = pnl)
    # A carrier with no panel checks nothing.
    @test isnothing(ReturnsResult(; nx = nx, X = X).pnl)
end
@testset "prices_to_returns carries the panel" begin
    ts = [Dates.Date(2020, 1, i) for i in 1:3]
    P = TimeSeries.TimeArray(ts, [100.0 100.0 100.0; 110.0 90.0 100.0; 121.0 81.0 100.0],
                             [:A, :B, :C])
    pnl = asset_panel([NumericPanelInput(; name = "mcap", vals = reshape(1.0:9.0, 3, 3))];
                      amsk = trues(3, 3), emsk = trues(3, 3))
    rd = prices_to_returns(P; pnl = pnl)
    # Two returns rows survive the difference, so the panel is sliced to them.
    @test size(rd.X) == (2, 3)
    @test size(rd.pnl.amsk) == (2, 3)
    @test PortfolioOptimisers.panel_field(rd.pnl, "mcap").vals == [2.0 5.0 8.0; 3.0 6.0 9.0]
    @test panel_feature_matrix(rd.pnl)[1] == panel_feature_matrix(pnl)[1]

    # A static panel has no observation axis to recover, so it rides through whole.
    sp = asset_panel([NumericPanelInput(; name = "mcap", vals = [1.0, 2.0, 3.0])])
    @test PortfolioOptimisers.panel_field(prices_to_returns(P; pnl = sp).pnl, "mcap").vals ==
          [1.0, 2.0, 3.0]
end
