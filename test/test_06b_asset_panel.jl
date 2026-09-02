#=
The Asset Panel carrier, its builder and the view contract (issue #664, map #643).

The panel's structure rides `ReturnsResult.pnl`; its values ride `nz` and `Z`. So the file
tests three seams: the field index and the two masks, the builder that resolves every blank,
and the two `port_opt_view` arities that carry the masks in step with `X`.
=#
@testset "Panel Field kinds" begin
    @test PortfolioOptimisers.panel_field_labels(NumericPanelField(), "mcap") == ["mcap"]
    @test PortfolioOptimisers.panel_field_observables(NumericPanelField()) == 1

    cat = CategoricalPanelField(; levels = ["Tech", "Energy"])
    @test cat.levels == ["Tech", "Energy"]
    @test PortfolioOptimisers.panel_field_labels(cat, "sector") ==
          ["sector=Tech", "sector=Energy"]
    @test PortfolioOptimisers.panel_field_observables(cat) == 1

    ten = TensorPanelField(; axis = "factor", labels = ["size", "value"],
                           groups = ["style", "style"])
    @test ten.axis == "factor"
    @test ten.groups == ["style", "style"]
    @test PortfolioOptimisers.panel_field_labels(ten, "beta") == ["beta=size", "beta=value"]
    @test PortfolioOptimisers.panel_field_observables(ten) == 2

    # The three label rules, each raising on its own condition.
    @test_throws IsEmptyError CategoricalPanelField(; levels = String[])
    @test_throws ArgumentError CategoricalPanelField(; levels = ["a", ""])
    @test_throws ArgumentError CategoricalPanelField(; levels = ["a", "a"])
    @test_throws IsEmptyError TensorPanelField(; axis = "", labels = ["a"])
    @test_throws ArgumentError TensorPanelField(; axis = "f", labels = ["a", "a"])
    @test_throws DimensionMismatch TensorPanelField(; axis = "f", labels = ["a", "b"],
                                                    groups = ["g"])
end
@testset "PanelField, the index row" begin
    f = PanelField(; name = "mcap", kind = NumericPanelField(), cols = [1], ocols = [2])
    @test f.name == "mcap"
    @test f.cols == [1]
    @test f.ocols == [2]

    @test_throws IsEmptyError PanelField(; name = "", kind = NumericPanelField(),
                                         cols = [1])
    @test_throws IsEmptyError PanelField(; name = "a", kind = NumericPanelField(),
                                         cols = Int[])
    @test_throws DomainError PanelField(; name = "a", kind = NumericPanelField(),
                                        cols = [0])
    @test_throws ArgumentError PanelField(; name = "a", kind = NumericPanelField(),
                                          cols = [1, 1])
    # A numeric Panel Field claims exactly one column.
    @test_throws DimensionMismatch PanelField(; name = "a", kind = NumericPanelField(),
                                              cols = [1, 2])
    # A tensor Panel Field needs one observed-mask column per label.
    ten = TensorPanelField(; axis = "f", labels = ["a", "b"])
    @test_throws DimensionMismatch PanelField(; name = "t", kind = ten, cols = [1, 2],
                                              ocols = [3])
    # A column carries one meaning.
    @test_throws ArgumentError PanelField(; name = "a", kind = NumericPanelField(),
                                          cols = [1], ocols = [1])
end
@testset "AssetPanel, its invariant and its views" begin
    pf = [PanelField(; name = "mcap", kind = NumericPanelField(), cols = [1], ocols = [2]),
          PanelField(; name = "sector", kind = CategoricalPanelField(; levels = ["T", "E"]),
                     cols = [3, 4])]
    amsk = [true true true; true true false]
    emsk = [true true false; true false false]
    pnl = AssetPanel(; pf = pf, amsk = amsk, emsk = emsk)
    @test length(pnl.pf) == 2
    @test PortfolioOptimisers.panel_field(pnl, "sector").cols == [3, 4]
    @test_throws KeyError PortfolioOptimisers.panel_field(pnl, "sectr")

    # The asset arity slices the mask columns and keeps every observation.
    v = PortfolioOptimisers.port_opt_view(pnl, 2:3)
    @test size(v.amsk) == (2, 2)
    @test v.amsk == amsk[:, 2:3]
    @test v.emsk == emsk[:, 2:3]
    @test v.pf === pnl.pf
    @test isa(v.amsk, SubArray)

    # The observations-then-assets arity slices both axes, and ignores the factor index.
    w = PortfolioOptimisers.port_opt_view(pnl, 1:1, 2:3, :)
    @test size(w.amsk) == (1, 2)
    @test w.amsk == amsk[1:1, 2:3]
    @test w.emsk == emsk[1:1, 2:3]
    @test isa(w.emsk, SubArray)

    # An empty index, a repeated name, a column claimed twice, and a mask mismatch.
    @test_throws IsEmptyError AssetPanel(; pf = PanelField[], amsk = trues(2, 3),
                                         emsk = trues(2, 3))
    @test_throws ArgumentError AssetPanel(;
                                          pf = [PanelField(; name = "a",
                                                           kind = NumericPanelField(),
                                                           cols = [1]),
                                                PanelField(; name = "a",
                                                           kind = NumericPanelField(),
                                                           cols = [2])], amsk = trues(2, 3),
                                          emsk = trues(2, 3))
    @test_throws ArgumentError AssetPanel(;
                                          pf = [PanelField(; name = "a",
                                                           kind = NumericPanelField(),
                                                           cols = [1]),
                                                PanelField(; name = "b",
                                                           kind = NumericPanelField(),
                                                           cols = [1])], amsk = trues(2, 3),
                                          emsk = trues(2, 3))
    @test_throws IsEmptyError AssetPanel(; pf = pf, amsk = trues(0, 0), emsk = trues(0, 0))
    @test_throws DimensionMismatch AssetPanel(; pf = pf, amsk = trues(2, 3),
                                              emsk = trues(2, 2))
    # The subset invariant is checked, not coerced.
    @test_throws ArgumentError AssetPanel(; pf = pf, amsk = [true false; true true],
                                          emsk = [true true; true true])
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
    @test_throws ArgumentError CategoricalPanelInput(; name = "a", vals = ["x" "y"],
                                                     levels = ["a", "a"])
    @test_throws DimensionMismatch TensorPanelInput(; name = "a", vals = ones(2, 2, 2),
                                                    axis = "f", labels = ["a"])

    num = NumericPanelInput(; name = "mcap", vals = [missing 2.0; 3.0 4.0],
                            alg = ForwardPanelFill(; val = -1.0))
    v, o = PortfolioOptimisers.panel_resolve(num)
    @test v == [-1.0 2.0; 3.0 4.0]
    @test o == [false true; true true]
    @test PortfolioOptimisers.panel_input_kind(num, v) == NumericPanelField()

    cats = CategoricalPanelInput(; name = "sector", vals = [missing "E"; "T" "E"],
                                 alg = ConstantPanelFill(; val = "U"))
    cv, co = PortfolioOptimisers.panel_resolve(cats)
    @test cv == ["U" "E"; "T" "E"]
    @test co == [false true; true true]
    # The levels are read off the resolved labels, so the fill's own level gets a column.
    @test PortfolioOptimisers.panel_input_kind(cats, cv).levels == ["E", "T", "U"]

    ten = TensorPanelInput(; name = "beta", vals = reshape([1.0, NaN, 3.0, 4.0], 2, 1, 2),
                           axis = "factor", labels = ["a", "b"],
                           alg = ConstantPanelFill(; val = 0.0))
    tv, to = PortfolioOptimisers.panel_resolve(ten)
    @test tv == reshape([1.0, 0.0, 3.0, 4.0], 2, 1, 2)
    @test to == reshape([true, false, true, true], 2, 1, 2)
    @test PortfolioOptimisers.panel_input_kind(ten, tv).labels == ["a", "b"]

    # An infinity is not a blank, so no policy resolves it and the field is named.
    @test_throws IsNonFiniteError PortfolioOptimisers.panel_resolve(NumericPanelInput(;
                                                                                      name = "bad",
                                                                                      vals = [Inf 1.0;
                                                                                              2.0 3.0]))
    # A label outside the declared levels is refused, with a suggestion.
    kind = CategoricalPanelField(; levels = ["T"])
    Z = zeros(1, 1, 1)
    @test_throws ArgumentError PortfolioOptimisers.panel_write!(Z, kind, ["E";;], [1])

    # A single observable names its mask column after the field, several after the columns.
    @test PortfolioOptimisers.panel_observed_labels(NumericPanelField(), "mcap") ==
          ["mcap::observed"]
    @test PortfolioOptimisers.panel_observed_labels(TensorPanelField(; axis = "f",
                                                                     labels = ["a", "b"]),
                                                    "beta") ==
          ["beta=a::observed", "beta=b::observed"]
end
@testset "asset_panel, the build seam" begin
    res = asset_panel([NumericPanelInput(; name = "mcap", vals = [1.0 missing; 3.0 4.0],
                                         alg = ForwardPanelFill(; val = 0.0)),
                       CategoricalPanelInput(; name = "sector", vals = ["T" "E"; "T" "E"],
                                             levels = ["T", "E"]),
                       TensorPanelInput(; name = "beta",
                                        vals = reshape(collect(1.0:4.0), 2, 2, 1),
                                        axis = "factor", labels = ["size"])])
    @test res.nz == ["mcap", "mcap::observed", "sector=T", "sector=E", "beta=size"]
    @test size(res.Z) == (2, 2, 5)
    @test res.Z[:, :, 1] == [1.0 0.0; 3.0 4.0]
    @test res.Z[:, :, 2] == [1.0 0.0; 1.0 1.0]
    # One-hot: exactly one 1 per cell across the level columns.
    @test res.Z[:, :, 3] == [1.0 0.0; 1.0 0.0]
    @test res.Z[:, :, 4] == [0.0 1.0; 0.0 1.0]
    @test res.Z[:, :, 5] == [1.0 3.0; 2.0 4.0]
    @test all(isfinite, res.Z)
    # A NoPanelFill field contributes no observed-mask column.
    @test PortfolioOptimisers.panel_field(res.pnl, "sector").ocols === nothing
    @test PortfolioOptimisers.panel_field(res.pnl, "mcap").ocols == [2]
    @test res.pnl.amsk == trues(2, 2)
    @test res.pnl.emsk == trues(2, 2)

    # The masks are taken as given, and the subset invariant reaches them.
    res2 = asset_panel([NumericPanelInput(; name = "a", vals = ones(2, 2))];
                       amsk = [true false; true true], emsk = [true false; false true])
    @test res2.pnl.amsk == [true false; true true]
    @test_throws ArgumentError asset_panel([NumericPanelInput(; name = "a",
                                                              vals = ones(2, 2))];
                                           amsk = [true false; true true],
                                           emsk = [true true; true true])

    @test_throws IsEmptyError asset_panel(PortfolioOptimisers.AbstractPanelFieldInput[])
    @test_throws ArgumentError asset_panel([NumericPanelInput(; name = "a",
                                                              vals = ones(2, 2)),
                                            NumericPanelInput(; name = "a",
                                                              vals = ones(2, 2))])
    @test_throws DimensionMismatch asset_panel([NumericPanelInput(; name = "a",
                                                                  vals = ones(2, 2)),
                                                NumericPanelInput(; name = "b",
                                                                  vals = ones(3, 2))])
    # A field literally named after another's one-hot column collides on the feature axis.
    @test_throws ArgumentError asset_panel([CategoricalPanelInput(; name = "sector",
                                                                  vals = ["T" "T"; "T" "T"],
                                                                  levels = ["T"]),
                                            NumericPanelInput(; name = "sector=T",
                                                              vals = ones(2, 2))])
    @test PortfolioOptimisers.panel_column_owner(res.pnl.pf, 2) == "mcap"
    @test PortfolioOptimisers.panel_column_owner(res.pnl.pf, 99) == "?"
end
@testset "ReturnsResult carries the panel through both views" begin
    res = asset_panel([NumericPanelInput(; name = "mcap",
                                         vals = [1.0 2.0 3.0; 4.0 missing 6.0],
                                         alg = ForwardPanelFill(; val = 0.0))];
                      amsk = [true true false; true true true],
                      emsk = [true false false; true true true])
    nx = ["A", "B", "C"]
    X = [0.1 0.2 0.3; 0.4 0.5 0.6]
    rd = ReturnsResult(; nx = nx, X = X,
                       ts = [Dates.Date(2020, 1, 1), Dates.Date(2020, 1, 2)], res...)
    @test rd.pnl.amsk == [true true false; true true true]
    @test size(rd.Z) == (2, 3, 2)

    # The asset arity slices the mask columns; the field index is untouched.
    va = PortfolioOptimisers.port_opt_view(rd, 2:3)
    @test va.pnl.amsk == [true false; true true]
    @test va.nz == rd.nz
    @test va.pnl.pf === rd.pnl.pf

    # The observations-then-assets arity slices both mask axes.
    vb = PortfolioOptimisers.port_opt_view(rd, 2:2, 1:2)
    @test vb.pnl.amsk == [true true]
    @test vb.pnl.emsk == [true true]
    @test size(vb.Z) == (1, 2, 2)

    # A carrier holding a panel needs the feature axis and the time-varying feature matrix.
    @test_throws IsNothingError ReturnsResult(; nx = nx, X = X, pnl = res.pnl)
    @test_throws DimensionMismatch ReturnsResult(; nx = nx, X = X, nz = ["a", "b"],
                                                 Z = ones(3, 2), pnl = res.pnl)
    # The masks must match the feature matrix's first two axes.
    @test_throws DimensionMismatch ReturnsResult(; nx = ["A", "B"], X = [0.1 0.2; 0.3 0.4],
                                                 nz = res.nz, Z = ones(2, 2, 2),
                                                 pnl = res.pnl)
    # A field index naming a column the feature axis does not hold is refused.
    wide = AssetPanel(;
                      pf = [PanelField(; name = "mcap", kind = NumericPanelField(),
                                       cols = [9])], amsk = res.pnl.amsk,
                      emsk = res.pnl.emsk)
    @test_throws DimensionMismatch ReturnsResult(; nx = nx, X = X, nz = res.nz, Z = rd.Z,
                                                 pnl = wide)
    # A carrier with no panel checks nothing.
    @test isnothing(ReturnsResult(; nx = nx, X = X).pnl)
end
@testset "prices_to_returns carries the panel" begin
    ts = [Dates.Date(2020, 1, i) for i in 1:3]
    P = TimeSeries.TimeArray(ts, [100.0 100.0 100.0; 110.0 90.0 100.0; 121.0 81.0 100.0],
                             [:A, :B, :C])
    res = asset_panel([NumericPanelInput(; name = "mcap", vals = reshape(1.0:9.0, 3, 3))];
                      amsk = trues(3, 3), emsk = trues(3, 3))
    rd = prices_to_returns(P; res...)
    # Two returns rows survive the difference, so the panel is sliced to them.
    @test size(rd.X) == (2, 3)
    @test size(rd.pnl.amsk) == (2, 3)
    @test size(rd.Z) == (2, 3, 1)
    @test rd.pnl.pf === res.pnl.pf
    @test rd.nz == res.nz
end
