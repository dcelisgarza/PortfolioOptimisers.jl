#=
The Asset Panel carrier, its builder and the view contract (issue #664, map #643, #809).

A Panel Field owns its values, so the panel *is* the feature data and nothing else on a
carrier holds a feature matrix. The file tests four seams: the three Panel Field types and
their rules, the panel's own invariant in the static and the time-varying shape, the builder
that resolves every blank, and the two `port_opt_view` arities that carry the panel in step
with `X`. The derived Feature Matrix is checked where it is derived, not where it is stored,
because nothing stores it.
=#
include(joinpath(@__DIR__, "asset_panel_fixture.jl"))
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

    #=
    Two Panel Fields with different names can derive one Feature Matrix column name (#1335). The
    panel refuses it at construction, so no panel exists without a valid Feature Matrix, and the
    message names the column and the Panel Fields that give it.
    =#
    derived_name_error(pf) =
        try
            AssetPanel(; pf = pf)
            nothing
        catch err
            err
        end
    # A numeric Panel Field against a tensor label.
    err = derived_name_error([NumericPanelField(; name = "beta=size", vals = [1.0, 2.0]),
                              TensorPanelField(; name = "beta", axis = "f",
                                               labels = ["size"],
                                               vals = reshape([7.0, 8.0], 2, 1))])
    @test isa(err, ArgumentError)
    @test occursin("\"beta=size\" is derived twice, first by the Panel Field \"beta=size\" and then by \"beta\"",
                   err.msg)
    # A numeric Panel Field against another's observed-mask column.
    err = derived_name_error([NumericPanelField(; name = "x", vals = [1.0, 2.0],
                                                omsk = [true, false]),
                              NumericPanelField(; name = "x::observed", vals = [1.0, 0.0])])
    @test isa(err, ArgumentError)
    @test occursin("\"x::observed\" is derived twice, first by the Panel Field \"x\" and then by \"x::observed\"",
                   err.msg)
    # A numeric Panel Field against a categorical level, in either order.
    s = CategoricalPanelField(; name = "s", levels = ["a", "b"], codes = [1, 2])
    sa = NumericPanelField(; name = "s=a", vals = [1.0, 2.0])
    @test isa(derived_name_error([s, sa]), ArgumentError)
    err = derived_name_error([sa, s])
    @test occursin("\"s=a\" is derived twice, first by the Panel Field \"s=a\" and then by \"s\"",
                   err.msg)
    # One tensor Panel Field whose value column is another label's observed-mask column.
    err = derived_name_error([TensorPanelField(; name = "t", axis = "f",
                                               labels = ["a", "a::observed"],
                                               vals = ones(2, 2), omsk = trues(2, 2))])
    @test isa(err, ArgumentError)
    @test occursin("\"t=a::observed\" is derived twice, first by the Panel Field \"t\" and then by \"t\"",
                   err.msg)
    # Names that only look alike stay legal, and the Feature Matrix names stay unique.
    ok = AssetPanel(;
                    pf = [NumericPanelField(; name = "x", vals = [1.0, 2.0],
                                            omsk = [true, false]),
                          NumericPanelField(; name = "x:observed", vals = [1.0, 0.0]), s,
                          NumericPanelField(; name = "s=c", vals = [1.0, 2.0])])
    @test panel_feature_matrix(ok)[1] ==
          ["x", "x::observed", "x:observed", "s=a", "s=b", "s=c"]
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

    #=
    A panel of one numeric Panel Field per column derives the matrix it was built from. `#807`
    deleted the library verb that did this, because the collapse returns a panel and a producer
    builds its tensor field directly, so the round trip is a *test* helper now: it says that
    the derivation is faithful, which is what every consumer of `panel_feature_matrix` rests on.
    =#
    rt = matrix_panel(nz, Z)
    @test panel_feature_matrix(rt) == (nz, Z)
    @test all(f -> isa(f, NumericPanelField), rt.pf)
    @test rt.amsk == trues(2, 2)
    srt = matrix_panel(snz, sZ)
    @test PortfolioOptimisers.panel_is_static(srt)
    @test panel_feature_matrix(srt) == (snz, sZ)

    # A duplicated column name is still refused, by the panel itself.
    @test_throws ArgumentError matrix_panel(["a", "a"], ones(2, 2))
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

    #=
    The square case is derived at the view, by name: a tensor Panel Field whose labels are the
    carrier's asset names is cut on its label axis by the same asset index. Nothing records the
    fact, and the comparison is `features_are_assets`.
    =#
    Zt = reshape(collect(1.0:18.0), 2, 3, 3)
    nx = ["A", "B", "C"]
    sq = asset_panel([TensorPanelInput(; name = "prox", axis = "asset", labels = nx,
                                       vals = Zt)])
    @test PortfolioOptimisers.features_are_assets(PortfolioOptimisers.panel_field(sq,
                                                                                  "prox"),
                                                  nx)
    sqv = PortfolioOptimisers.port_opt_view(sq, 1:2, 2:3, nx)
    @test PortfolioOptimisers.panel_field(sqv, "prox").labels == ["B", "C"]
    @test PortfolioOptimisers.panel_feature_names(sqv) == ["prox=B", "prox=C"]
    @test PortfolioOptimisers.panel_field(sqv, "prox").vals == Zt[1:2, 2:3, 2:3]

    # The groups belong to the labels, so the square case cuts them by the same index (#807
    # decision 1, ADR 0102), and a field without groups keeps its `nothing`.
    @test isnothing(PortfolioOptimisers.panel_field(sqv, "prox").groups)
    sqg = asset_panel([TensorPanelInput(; name = "prox", axis = "asset", labels = nx,
                                        groups = ["g1", "g2", "g2"], vals = Zt)])
    sqgv = PortfolioOptimisers.port_opt_view(sqg, 1:2, [3, 1], nx)
    @test PortfolioOptimisers.panel_field(sqgv, "prox").labels == ["C", "A"]
    @test PortfolioOptimisers.panel_field(sqgv, "prox").groups == ["g2", "g1"]
    @test PortfolioOptimisers.panel_field(sqgv, "prox").vals == Zt[1:2, [3, 1], [3, 1]]
    # Outside the square case the groups travel whole with the labels.
    rectg = asset_panel([TensorPanelInput(; name = "beta", axis = "factor",
                                          labels = ["f1", "f2", "f3"],
                                          groups = ["s", "s", "m"], vals = Zt)])
    @test PortfolioOptimisers.panel_field(PortfolioOptimisers.port_opt_view(rectg, 1:2, 2:3,
                                                                            nx), "beta").groups ==
          ["s", "s", "m"]

    # A label vector that is not the asset names leaves the label axis whole.
    rect = asset_panel([TensorPanelInput(; name = "beta", axis = "factor",
                                         labels = ["f1", "f2", "f3"], vals = Zt)])
    @test !PortfolioOptimisers.features_are_assets(PortfolioOptimisers.panel_field(rect,
                                                                                   "beta"),
                                                   nx)
    rv = PortfolioOptimisers.port_opt_view(rect, 1:2, 2:3, nx)
    @test PortfolioOptimisers.panel_field(rv, "beta").labels == ["f1", "f2", "f3"]
    @test PortfolioOptimisers.panel_field(rv, "beta").vals == Zt[1:2, 2:3, :]

    # A numeric or categorical Panel Field has no trailing axis, so it is never square.
    @test !PortfolioOptimisers.features_are_assets(PortfolioOptimisers.panel_field(pnl,
                                                                                   "mcap"),
                                                   nx)
    # And a carrier that does not name its assets makes no claim.
    @test !PortfolioOptimisers.features_are_assets(PortfolioOptimisers.panel_field(sq,
                                                                                   "prox"),
                                                   nothing)
end
@testset "Fill policies" begin
    @test PortfolioOptimisers.is_panel_blank(missing)
    @test PortfolioOptimisers.is_panel_blank(NaN)
    @test PortfolioOptimisers.is_panel_blank(nothing)
    @test !PortfolioOptimisers.is_panel_blank(1.0)
    @test !PortfolioOptimisers.is_panel_blank("a")
    # A source with an explicit null writes `nothing`, and the build resolves it like a
    # `missing`; the field's type is read off the cells it keeps.
    pnull = asset_panel([NumericPanelInput(; name = "a",
                                           vals = Union{Nothing, Float64}[1.0, nothing],
                                           alg = ConstantPanelFill(; val = -1.0)),
                         CategoricalPanelInput(; name = "s", vals = Any["x", nothing],
                                               alg = ConstantPanelFill(; val = "x"))])
    @test PortfolioOptimisers.panel_field(pnull, "a").vals == [1.0, -1.0]
    @test eltype(PortfolioOptimisers.panel_field(pnull, "a").vals) === Float64
    @test PortfolioOptimisers.panel_field(pnull, "a").omsk == [true, false]
    @test PortfolioOptimisers.panel_field(pnull, "s").codes == [1, 1]
    @test PortfolioOptimisers.panel_field(pnull, "s").omsk == [true, false]
    @test_throws ArgumentError asset_panel([NumericPanelInput(; name = "a",
                                                              vals = Any[1.0, nothing])])
    # A categorical input is filled with a label, so a policy whose val is the numeric
    # default is refused at construction rather than minting a level named "0.0".
    @test_throws ArgumentError CategoricalPanelInput(; name = "s", vals = ["a", missing],
                                                     alg = ForwardPanelFill())
    @test_throws ArgumentError CategoricalPanelInput(; name = "s", vals = ["a", missing],
                                                     alg = ConstantPanelFill())
    @test_throws ArgumentError CategoricalPanelInput(; name = "s", vals = ["a", missing],
                                                     alg = BackwardPanelFill(; val = 1))
    @test isa(CategoricalPanelInput(; name = "s", vals = ["a", missing],
                                    alg = ForwardPanelFill(; val = "a")),
              CategoricalPanelInput)

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
# A test-local input and a test-local policy that define only the methods an extension written
# before #1413 defines: the builder reaches them through the fallbacks that drop the active mask.
struct Test06bPanelInput <: PortfolioOptimisers.AbstractPanelFieldInput
    name::String
    vals::Matrix{Float64}
    alg::Any
end
function test06b_numeric(inp::Test06bPanelInput)
    return NumericPanelInput(; name = inp.name, vals = inp.vals, alg = inp.alg)
end
PortfolioOptimisers.panel_input_is_static(::Test06bPanelInput) = false
function PortfolioOptimisers.panel_resolve(inp::Test06bPanelInput)
    return PortfolioOptimisers.panel_resolve(test06b_numeric(inp))
end
function PortfolioOptimisers.panel_input_field(inp::Test06bPanelInput, vals, obs)
    return PortfolioOptimisers.panel_input_field(test06b_numeric(inp), vals, obs)
end
struct Test06bPanelFill <: PortfolioOptimisers.AbstractPanelFillAlgorithm end
function PortfolioOptimisers.panel_fill(::Test06bPanelFill, v::AbstractVector,
                                        ::AbstractString)
    return PortfolioOptimisers.panel_fill(ForwardPanelFill(), v, "f")
end
@testset "A directional fill stops at the edge of an inactive stretch (#1413)" begin
    # Asset 1 delists after observation 4 and lists again at 8, whose first cell is blank.
    amsk = trues(12, 2)
    amsk[5:7, 1] .= false
    x = [collect(1.0:12.0) collect(21.0:32.0)]
    x[5:8, 1] .= NaN
    build(alg; m = amsk) = panel_field(asset_panel([NumericPanelInput(; name = "x",
                                                                      vals = x, alg = alg)];
                                                   amsk = m, emsk = m), "x")
    f = build(ForwardPanelFill(; val = -1.0))
    # Nothing crosses the stretch: cell 8 has nothing to carry on its own side of it, and the
    # inactive cells 5:7 have nothing either.
    @test f.vals[:, 1] ==
          [1.0, 2.0, 3.0, 4.0, -1.0, -1.0, -1.0, -1.0, 9.0, 10.0, 11.0, 12.0]
    @test f.vals[:, 2] == x[:, 2]
    @test f.omsk == .!isnan.(x)
    # The run toward the limit restarts in each stretch, so no limit reaches back over it.
    for lim in 1:5
        @test build(ForwardPanelFill(; val = -1.0, lim = lim)).vals[8, 1] == -1.0
    end
    # Without a mask the column is one stretch, and the fill carries as before.
    @test build(ForwardPanelFill(; val = -1.0); m = nothing).vals[5:8, 1] == fill(4.0, 4)
    @test build(ForwardPanelFill(; val = -1.0, lim = 3); m = nothing).vals[5:8, 1] ==
          [4.0, 4.0, 4.0, -1.0]
    # The backward fill is the mirror image: the last cell of the first listing is blank.
    y = copy(x)
    y[4:7, 1] .= NaN
    y[5:7, 1] .= [50.0, 60.0, 70.0]
    b = panel_field(asset_panel([NumericPanelInput(; name = "y", vals = y,
                                                   alg = BackwardPanelFill(; val = -1.0))];
                                amsk = amsk, emsk = amsk), "y")
    # An observed inactive cell carries within its own stretch, and never into the active one.
    @test b.vals[4:8, 1] == [-1.0, 50.0, 60.0, 70.0, 9.0]
    # A tensor Panel Field reads the active mask of its asset, the second axis, in every slice.
    t = cat(x, x .+ 100; dims = 3)
    tf = panel_field(asset_panel([TensorPanelInput(; name = "t", vals = t, axis = "f",
                                                   labels = ["a", "b"],
                                                   alg = ForwardPanelFill(; val = -1.0))];
                                 amsk = amsk, emsk = amsk), "t")
    @test tf.vals[5:8, 1, 1] == fill(-1.0, 4)
    @test tf.vals[5:8, 1, 2] == fill(-1.0, 4)
    @test tf.vals[:, 2, :] == t[:, 2, :]
    # The one-argument resolution reads no mask, so the column is one stretch.
    inp = NumericPanelInput(; name = "x", vals = x, alg = ForwardPanelFill(; val = -1.0))
    @test PortfolioOptimisers.panel_resolve(inp)[1][8, 1] == 4.0
    @test PortfolioOptimisers.panel_resolve(inp, amsk)[1][8, 1] == -1.0
    # A mask of the wrong shape is refused before the fill reads it.
    @test_throws DimensionMismatch PortfolioOptimisers.panel_resolve(inp, trues(11, 2))
    @test_throws DimensionMismatch asset_panel([inp]; amsk = trues(12, 3),
                                               emsk = trues(12, 3))
    # An extension's input and policy that know no mask keep working through the fallbacks.
    ext = Test06bPanelInput("x", x, ForwardPanelFill(; val = -1.0))
    @test panel_field(asset_panel([ext]; amsk = amsk, emsk = amsk), "x").vals[8, 1] == 4.0
    own = NumericPanelInput(; name = "x", vals = x, alg = Test06bPanelFill())
    @test PortfolioOptimisers.panel_resolve(own, amsk)[1][8, 1] == 4.0
    # A cell-wise policy reads no mask, and NoPanelFill names the position in the whole column.
    c = panel_field(asset_panel([NumericPanelInput(; name = "x", vals = x,
                                                   alg = ConstantPanelFill(; val = -1.0))];
                                amsk = amsk, emsk = amsk), "x")
    @test c.vals[5:8, 1] == fill(-1.0, 4)
    err = try
        asset_panel([NumericPanelInput(; name = "x", vals = x)]; amsk = amsk, emsk = amsk)
    catch e
        e
    end
    @test occursin("position 5", err.msg)
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
    #=
    Masks beside static inputs alone are a *request* for observations, not a contradiction:
    the panel couples the masks to the shape, so the build lifts each static input onto the
    masks' observation count. The lift is lazy, and it lifts the masks of the Panel Field too.
    This input cannot blank, so it carries no observed mask before or after the lift (#1631).
    =#
    lifted = asset_panel([NumericPanelInput(; name = "a", vals = [1.0, 2.0])];
                         amsk = trues(3, 2), emsk = trues(3, 2))
    @test !PortfolioOptimisers.panel_is_static(lifted)
    @test size(PortfolioOptimisers.panel_field(lifted, "a").vals) == (3, 2)
    @test PortfolioOptimisers.panel_field(lifted, "a").vals == [1.0 2.0; 1.0 2.0; 1.0 2.0]
    @test PortfolioOptimisers.panel_field(lifted, "a").omsk === nothing
    # The values are stored once.
    @test isa(PortfolioOptimisers.panel_field(lifted, "a").vals,
              PortfolioOptimisers.RepeatedLeading)
    @test PortfolioOptimisers.panel_field(lifted, "a").vals.parent == [1.0, 2.0]
    # A view of a lifted field is an ordinary `SubArray`, sliced like any other.
    lv = PortfolioOptimisers.port_opt_view(lifted, 2:3, [2])
    @test PortfolioOptimisers.panel_field(lv, "a").vals == reshape([2.0, 2.0], 2, 1)
    # A static input that meets a time-varying one is lifted the same way.
    mixed = asset_panel([NumericPanelInput(; name = "t", vals = [1.0 2.0; 3.0 4.0]),
                         CategoricalPanelInput(; name = "s", vals = ["T", "E"])])
    @test !PortfolioOptimisers.panel_is_static(mixed)
    @test PortfolioOptimisers.panel_field(mixed, "s").codes == [2 1; 2 1]
    @test PortfolioOptimisers.panel_feature_matrix(mixed)[1] == ["t", "s=E", "s=T"]
    # The lazy array owes `show` too.
    @test occursin("RepeatedLeading",
                   sprint(show, PortfolioOptimisers.panel_field(lifted, "a").vals))
    @test occursin("RepeatedLeading",
                   sprint(show, MIME"text/plain"(),
                          PortfolioOptimisers.panel_field(lifted, "a").vals))
    @test_throws DomainError PortfolioOptimisers.RepeatedLeading([1.0, 2.0], 0)
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
    stat = matrix_panel(["f1", "f2"], [1.0 2.0; 3.0 4.0; 5.0 6.0])
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
    rd = prices_to_returns(PricesResult(; X = P, pnl = pnl))
    # Two returns rows survive the difference, so the panel is sliced to them.
    @test size(rd.X) == (2, 3)
    @test size(rd.pnl.amsk) == (2, 3)
    @test PortfolioOptimisers.panel_field(rd.pnl, "mcap").vals == [2.0 5.0 8.0; 3.0 6.0 9.0]
    @test panel_feature_matrix(rd.pnl)[1] == panel_feature_matrix(pnl)[1]

    # A static panel has no observation axis to recover, so it rides through whole.
    sp = asset_panel([NumericPanelInput(; name = "mcap", vals = [1.0, 2.0, 3.0])])
    @test PortfolioOptimisers.panel_field(prices_to_returns(PricesResult(; X = P, pnl = sp)).pnl,
                                          "mcap").vals == [1.0, 2.0, 3.0]
end
@testset "PricesResult carries the panel through both views" begin
    ts = [Dates.Date(2020, 1, i) for i in 1:3]
    P = TimeSeries.TimeArray(ts, [100.0 100.0 100.0; 110.0 90.0 100.0; 121.0 81.0 100.0],
                             [:A, :B, :C])
    # vals is column-major, so vals == [1 4 7; 2 5 8; 3 6 9] and every entry names its own
    # position. The two masks differ from each other, so a slice that dropped one of them
    # cannot pass by matching the other.
    amsk = [true true false; true false true; false true true]
    emsk = [true false false; true false false; false true true]
    pnl = asset_panel([NumericPanelInput(; name = "mcap", vals = reshape(1.0:9.0, 3, 3))];
                      amsk = amsk, emsk = emsk)
    pr = PricesResult(; X = P, pnl = pnl)
    @test pr.pnl === pnl

    # The panel describes the carrier's universe, so both its axes are the price series'.
    @test_throws DimensionMismatch PricesResult(; X = P,
                                                pnl = asset_panel([NumericPanelInput(;
                                                                                     name = "mcap",
                                                                                     vals = ones(3,
                                                                                                 2))];
                                                                  amsk = trues(3, 2),
                                                                  emsk = trues(3, 2)))
    @test_throws DimensionMismatch PricesResult(; X = P,
                                                pnl = asset_panel([NumericPanelInput(;
                                                                                     name = "mcap",
                                                                                     vals = ones(2,
                                                                                                 3))];
                                                                  amsk = trues(2, 3),
                                                                  emsk = trues(2, 3)))
    # A carrier with no panel checks nothing.
    @test isnothing(PricesResult(; X = P).pnl)

    # The timestamp arity keeps every asset and slices both masks to the rows it keeps.
    vt = PortfolioOptimisers.port_opt_view(pr, ts[2:3])
    @test vt.pnl.amsk == amsk[2:3, :]
    @test vt.pnl.emsk == emsk[2:3, :]
    @test PortfolioOptimisers.panel_field(vt.pnl, "mcap").vals == [2.0 5.0 8.0; 3.0 6.0 9.0]

    # The observations-then-assets arity slices both mask columns as well.
    va = PortfolioOptimisers.port_opt_view(pr, ts[2:3], [1, 3])
    @test va.pnl.amsk == amsk[2:3, [1, 3]]
    @test va.pnl.emsk == emsk[2:3, [1, 3]]
    @test PortfolioOptimisers.panel_field(va.pnl, "mcap").vals == [2.0 8.0; 3.0 9.0]
    @test panel_feature_matrix(va.pnl)[1] == panel_feature_matrix(pr.pnl)[1]

    # The conversion costs the first observation, so the panel comes back one row shorter.
    rd = prices_to_returns(PricesToReturns(), pr)
    @test size(rd.X, 1) == size(values(P), 1) - 1
    @test rd.pnl.amsk == amsk[2:3, :]
    @test rd.pnl.emsk == emsk[2:3, :]
    @test PortfolioOptimisers.panel_field(rd.pnl, "mcap").vals == [2.0 5.0 8.0; 3.0 6.0 9.0]

    # A static panel binds the asset axis alone, so a timestamp window leaves it whole.
    sp = matrix_panel(["f1", "f2"], [1.0 2.0; 3.0 4.0; 5.0 6.0])
    prs = PricesResult(; X = P, pnl = sp)
    @test panel_feature_matrix(PortfolioOptimisers.port_opt_view(prs, ts[2:3]).pnl)[2] ==
          [1.0 2.0; 3.0 4.0; 5.0 6.0]
    @test panel_feature_matrix(PortfolioOptimisers.port_opt_view(prs, ts[2:3], [1, 3]).pnl)[2] ==
          [1.0 2.0; 5.0 6.0]
    @test isnothing(prices_to_returns(PricesToReturns(), prs).pnl.amsk)
end
@testset "A view selects observations by timestamp and assets and factors by name (#1402)" begin
    ts = [Dates.Date(2024, 1, 2), Dates.Date(2024, 1, 3), Dates.Date(2024, 1, 5),
          Dates.Date(2024, 1, 8)]
    nx = ["AAPL", "MSFT", "XOM"]
    pnl = asset_panel([NumericPanelInput(; name = "mcap", vals = reshape(1.0:12.0, 4, 3))];
                      amsk = trues(4, 3),
                      emsk = [true false true; true true true;
                              false true true; true true false])
    rd = ReturnsResult(; nx = nx, X = reshape(0.01:0.01:0.12, 4, 3), nf = ["mkt", "smb"],
                       F = reshape(0.1:0.1:0.8, 4, 2), B = reshape(0.5:0.5:6.0, 4, 3),
                       nb = nx, ts = ts, pnl = pnl)
    # A name or a timestamp is a spelling of a position, so every such view equals the view
    # by position, field for field and in the order the caller gives.
    function same_view(a, b)
        return a.nx == b.nx &&
               a.X == b.X &&
               a.nf == b.nf &&
               a.F == b.F &&
               a.B == b.B &&
               a.nb == b.nb &&
               a.ts == b.ts &&
               a.pnl.amsk == b.pnl.amsk &&
               a.pnl.emsk == b.pnl.emsk &&
               PortfolioOptimisers.panel_field(a.pnl, "mcap").vals ==
               PortfolioOptimisers.panel_field(b.pnl, "mcap").vals
    end
    PV = PortfolioOptimisers.port_opt_view
    @test same_view(PV(rd, ["XOM", "AAPL"]), PV(rd, [3, 1]))
    @test same_view(PV(rd, ts[[2, 4]], ["MSFT"]), PV(rd, [2, 4], [2]))
    @test same_view(PV(rd, :, ["MSFT", "XOM"], ["smb"]), PV(rd, :, [2, 3], [2]))
    # A calendar range selects the observations the returns hold, in clock order, as the
    # timestamp view of a PricesResult does; order and repeats of the timestamps do not
    # matter, and a timestamp off the clock selects no row.
    @test same_view(PV(rd, Dates.Date(2024, 1, 3):Dates.Day(1):Dates.Date(2024, 1, 7), :),
                    PV(rd, [2, 3], :))
    @test same_view(PV(rd, [ts[4], Dates.Date(2023, 12, 31), ts[1], ts[4]], :),
                    PV(rd, [1, 4], :))
    # A DateTime selects the Date at the same instant, because the two compare equal.
    @test same_view(PV(rd, [Dates.DateTime(2024, 1, 5)], :), PV(rd, [3], :))
    # Positions, ranges and colons pass through unchanged.
    @test same_view(PV(rd, 2:3, 1:2, :), PV(rd, [2, 3], [1, 2]))

    # A name that selects nothing is refused rather than dropped: a dropped ticker would
    # shrink the universe with no error.
    err = @test_throws ArgumentError PV(rd, ["AAPL", "APPL"])
    @test occursin("APPL", err.value.msg) && occursin("did you mean `AAPL`", err.value.msg)
    @test_throws ArgumentError PV(rd, :, ["MSFT", "IBM"])
    @test_throws ArgumentError PV(rd, :, :, ["hml"])
    # A timestamp window that holds no observation is empty, which the carrier refuses.
    @test_throws IsEmptyError PV(rd, [Dates.Date(2023, 1, 1)], :)
    # Each index keeps its axis: a timestamp never selects an asset, and a name never selects
    # an observation, so the reversed index order of the two methods cannot pass silently.
    err = @test_throws ArgumentError PV(rd, ts[1:2])
    @test occursin("observation index", err.value.msg)
    err = @test_throws ArgumentError PV(rd, ["AAPL"], :)
    @test occursin("second index", err.value.msg)
    # A carrier with no names or no timestamps has nothing to match. Asset returns always
    # carry their names, so only a carrier of factor returns alone has no asset names.
    fonly = ReturnsResult(; nf = ["mkt", "smb"], F = rd.F)
    err = @test_throws ArgumentError PV(fonly, :, ["AAPL"])
    @test occursin("no asset names", err.value.msg)
    bare = ReturnsResult(; nx = nx, X = rd.X)
    err = @test_throws ArgumentError PV(bare, ts[1:2], :)
    @test occursin("no timestamps", err.value.msg)
    @test_throws ArgumentError PV(bare, :, :, ["mkt"])

    P = TimeSeries.TimeArray(ts, reshape(100.0:111.0, 4, 3), Symbol.(nx))
    ppnl = asset_panel([NumericPanelInput(; name = "mcap", vals = reshape(1.0:12.0, 4, 3))];
                       amsk = trues(4, 3), emsk = trues(4, 3))
    pr = PricesResult(; X = P,
                      B = TimeSeries.TimeArray(ts, reshape(1.0:12.0, 4, 3), Symbol.(nx)),
                      pnl = ppnl)
    function same_prices(a, b)
        return values(a.X) == values(b.X) &&
               TimeSeries.colnames(a.X) == TimeSeries.colnames(b.X) &&
               TimeSeries.timestamp(a.X) == TimeSeries.timestamp(b.X) &&
               values(a.B) == values(b.B) &&
               PortfolioOptimisers.panel_field(a.pnl, "mcap").vals ==
               PortfolioOptimisers.panel_field(b.pnl, "mcap").vals
    end
    @test same_prices(PV(pr, ts[2:3], ["XOM", "AAPL"]), PV(pr, ts[2:3], [3, 1]))
    @test same_prices(PV(pr, 2:3, ["MSFT"]), PV(pr, 2:3, [2]))
    @test same_prices(PV(pr, :, ["MSFT"]), PV(pr, :, [2]))
    err = @test_throws ArgumentError PV(pr, :, ["MSFT", "MSTF"])
    @test occursin("MSTF", err.value.msg)
    @test_throws ArgumentError PV(pr, ts[2:3], ts[2:3])
end
@testset "The panel's numeric type comes from its values (#910)" begin
    # A field resolves in the type its own cells carry, so a Float32 panel stays Float32
    # and a blank-carrying Union{Missing, Float64} field still resolves in Float64.
    p32 = asset_panel([NumericPanelInput(; name = "a", vals = Float32[1, 2, 3]),
                       CategoricalPanelInput(; name = "s", vals = ["x", "y", "x"])])
    @test eltype(PortfolioOptimisers.panel_field(p32, "a").vals) === Float32

    pmiss = asset_panel([NumericPanelInput(; name = "a",
                                           vals = Union{Missing, Float64}[1.0, missing,
                                                                          3.0],
                                           alg = ConstantPanelFill(; val = 0.0))])
    @test eltype(PortfolioOptimisers.panel_field(pmiss, "a").vals) === Float64

    # An integer field stays exact, and a heterogeneous raw field promotes over its cells.
    pint = asset_panel([NumericPanelInput(; name = "a", vals = [1, 2, 3])])
    @test eltype(PortfolioOptimisers.panel_field(pint, "a").vals) === Int
    pany = asset_panel([NumericPanelInput(; name = "a", vals = Any[1, 2.0f0, 3])])
    @test eltype(PortfolioOptimisers.panel_field(pany, "a").vals) === Float32

    # A tensor field resolves the same way.
    ptn = asset_panel([TensorPanelInput(; name = "t", axis = "factor",
                                        labels = ["f1", "f2"],
                                        vals = Float32[1 2; 3 4; 5 6])])
    @test eltype(PortfolioOptimisers.panel_field(ptn, "t").vals) === Float32

    # The Feature Matrix takes the promotion over the Panel Fields it stacks. An indicator
    # carries no type of its own, so it does not widen the block it is stacked beside.
    @test eltype(feature_matrix(p32)) === Float32
    @test eltype(panel_feature_matrix(p32)[2]) === Float32
    @test eltype(feature_matrix(pmiss)) === Float64
    @test eltype(feature_matrix(pint)) === Int

    # A selection of indicators alone has nothing to derive from, and stacks in Float64.
    pcat = asset_panel([CategoricalPanelInput(; name = "s", vals = ["x", "y", "x"])])
    @test eltype(feature_matrix(pcat)) === Float64
    @test eltype(panel_feature_matrix(pcat)[2]) === Float64

    # A selection that names only a mask column derives nothing from the columns it
    # stacks, so it takes the panel's own type: the fallback is the panel's, and it falls
    # to Float64 only on a panel with no numeric or tensor field at all (Q6, #802).
    pmsk = asset_panel([NumericPanelInput(; name = "a",
                                          vals = Union{Missing, Float32}[1, missing, 3],
                                          alg = ConstantPanelFill(; val = 0.0f0))])
    @test eltype(feature_matrix(pmsk)) === Float32
    @test eltype(feature_matrix(pmsk, ["a" => :observed])) === Float32

    # The Panel Field kinds report the type they contribute.
    @test PortfolioOptimisers.panel_value_eltype(PortfolioOptimisers.panel_field(p32, "a")) ===
          Float32
    @test PortfolioOptimisers.panel_value_eltype(PortfolioOptimisers.panel_field(p32, "s")) ===
          Union{}
    @test PortfolioOptimisers.panel_value_eltype(PortfolioOptimisers.panel_field(ptn, "t")) ===
          Float32
    @test PortfolioOptimisers.panel_value_eltype(Float64[]) === Float64
    @test PortfolioOptimisers.panel_value_eltype(Any[]) === Union{}

    # An indicator is built rather than read, so its caller names the type of the block it
    # is stacked beside. The default is what a caller who stacks it beside nothing gets.
    onehot = PortfolioOptimisers.panel_onehot(PortfolioOptimisers.panel_field(p32, "s"))
    @test eltype(onehot) === Float64
    @test onehot == [1 0; 0 1; 1 0]
    @test eltype(PortfolioOptimisers.panel_onehot(PortfolioOptimisers.panel_field(p32, "s");
                                                  datatype = Float32)) === Float32

    # A directional fill keeps the column's own type where it writes nothing, and widens
    # to the fill value's type only where it wrote it.
    pfwd = asset_panel([NumericPanelInput(; name = "a", vals = Float32[1 2; 3 4],
                                          alg = ForwardPanelFill()),
                        NumericPanelInput(; name = "b", vals = [1 2; 3 4],
                                          alg = BackwardPanelFill())]; amsk = trues(2, 2))
    @test eltype(PortfolioOptimisers.panel_field(pfwd, "a").vals) === Float32
    @test eltype(PortfolioOptimisers.panel_field(pfwd, "b").vals) === Int
    pgap = asset_panel([NumericPanelInput(; name = "a",
                                          vals = Union{Missing, Float32}[1 missing; 3 4],
                                          alg = ForwardPanelFill(; val = 0.0f0)),
                        NumericPanelInput(; name = "b",
                                          vals = Union{Missing, Float32}[1 missing; 3 4],
                                          alg = ForwardPanelFill())]; amsk = trues(2, 2))
    @test eltype(PortfolioOptimisers.panel_field(pgap, "a").vals) === Float32
    @test eltype(PortfolioOptimisers.panel_field(pgap, "b").vals) === Float64
end

@testset "An estimation mask that is not given is the active mask" begin
    # An all-true default would break the subset rule at the first inactive cell, so a
    # build that states the active mask alone gets that mask as its estimation mask.
    amsk = [true true; true false; true true]
    pnl = asset_panel([NumericPanelInput(; name = "m", vals = ones(3, 2))]; amsk = amsk)
    @test pnl.amsk == amsk
    @test pnl.emsk == amsk
    # The estimation mask alone widens the active mask to all-true, which contains it.
    emsk = [true false; true false; false true]
    pe = asset_panel([NumericPanelInput(; name = "m", vals = ones(3, 2))]; emsk = emsk)
    @test pe.amsk == trues(3, 2)
    @test pe.emsk == emsk
    # Both given, both kept, and the subset rule still holds them apart.
    pb = asset_panel([NumericPanelInput(; name = "m", vals = ones(3, 2))]; amsk = amsk,
                     emsk = emsk .& amsk)
    @test pb.emsk == emsk .& amsk
    @test_throws ArgumentError asset_panel([NumericPanelInput(; name = "m",
                                                              vals = ones(3, 2))];
                                           amsk = amsk, emsk = trues(3, 2))
end

@testset "The lazy lift checks its bounds on every axis" begin
    R = PortfolioOptimisers.RepeatedLeading([1.0, 2.0], 3)
    @test size(R) == (3, 2)
    @test R[3, 2] == 2.0
    @test_throws BoundsError R[4, 1]
    @test_throws BoundsError R[0, 1]
    @test_throws BoundsError R[1, 3]
    M = PortfolioOptimisers.RepeatedLeading([1 2; 3 4], 2)
    @test M[2, 2, 1] == 3
    @test_throws BoundsError M[3, 1, 1]
end

#=
The synthetic Asset Panel generator (issue #656, map #643).

The generator draws a panel from a factor model it also returns, so every test here asks one
question: does the panel the caller receives still hold the model the generator drew? The
shapes, the two masks, the reconciliation `X = f B' + eps` and the reproducibility of one
seed are the four the acceptance tests of the prior stand on.

The generator is a toy fixture for this one acceptance test, not library functionality, so it
lives in `test06c_setup.jl` beside this file rather than under `src/`.
=#
include(joinpath(@__DIR__, "test06c_setup.jl"))
using Statistics, Dates

@testset "Synthetic panel shapes and carrier" begin
    res = synthetic_asset_panel(; n_assets = 8, n_observations = 60, n_industries = 3,
                                rng = StableRNG(987))
    rd, tr = res.rd, res.truth
    T, N, K = 60, 8, 1 + 3 + length(SYNTHETIC_STYLES)

    @test size(rd.X) == (T, N)
    @test length(rd.nx) == N
    @test rd.nx[1] == "A00001"
    @test length(rd.ts) == T
    @test allunique(rd.ts)
    @test all(d -> Dates.dayofweek(d) <= 5, rd.ts)
    @test issorted(rd.ts)

    # Twenty numeric Panel Fields, each with its observed-mask column, then the one-hot
    # industry block.
    @test size(panel_feature_matrix(rd.pnl)[2]) == (T, N, 43)
    @test length(panel_feature_matrix(rd.pnl)[1]) == 43
    @test panel_feature_matrix(rd.pnl)[1][1] == "adj_close"
    @test panel_feature_matrix(rd.pnl)[1][2] == "adj_close::observed"
    @test panel_feature_matrix(rd.pnl)[1][(end - 2):end] ==
          ["industry=Real Estate", "industry=Software", "industry=Banks"]
    @test all(isfinite, panel_feature_matrix(rd.pnl)[2])
    @test length(rd.pnl.pf) == 21
    @test isa(PortfolioOptimisers.panel_field(rd.pnl, "market_cap"), NumericPanelField)
    @test PortfolioOptimisers.panel_field(rd.pnl, "industry").levels ==
          ["Real Estate", "Software", "Banks"]

    @test size(tr.B) == (N, K)
    @test size(tr.f) == (T, K)
    @test size(tr.eps) == (T, N)
    @test length(tr.ivar) == N
    @test length(tr.nf) == K
    @test length(tr.fgrp) == K
    @test tr.nf[1] == "market"
    @test tr.fgrp == vcat("market", fill("industry", 3), fill("style", 10))
    @test tr.nf[5:end] == [s[1] for s in SYNTHETIC_STYLES]
    @test all(x -> x > 0, tr.ivar)
end

@testset "Synthetic panel masks" begin
    res = synthetic_asset_panel(; n_assets = 40, n_observations = 400, n_industries = 4,
                                rng = StableRNG(4242))
    amsk, emsk = res.rd.pnl.amsk, res.rd.pnl.emsk

    # The subset invariant the carrier checks, and the listing structure it describes.
    @test all(emsk .<= amsk)
    @test !all(amsk)
    @test all(view(amsk, :, 1))
    @test all(view(emsk, :, 1))
    @test count(!, emsk) > count(!, amsk)

    # A cell that is not active carries a `NaN` return and no observed value.
    @test all(isnan, res.rd.X[.!amsk])
    @test all(isfinite, res.rd.X[amsk])
    obs = panel_feature_matrix(res.rd.pnl)[2][:, :,
                                              findfirst(==("market_cap::observed"),
                                                        panel_feature_matrix(res.rd.pnl)[1])]
    @test all(iszero, obs[.!amsk])
    @test all(isone, obs[amsk])

    # Every active window is contiguous: an asset lists once and delists once.
    for i in axes(amsk, 2)
        col = findall(view(amsk, :, i))
        @test col == first(col):last(col)
    end

    # Both probabilities at zero leave every asset listed throughout.
    full = synthetic_asset_panel(; n_assets = 10, n_observations = 50, n_industries = 2,
                                 late_listing_proba = 0.0, delisting_proba = 0.0,
                                 rng = StableRNG(11))
    @test all(full.rd.pnl.amsk)
    @test all(isfinite, full.rd.X)

    # Both probabilities at one drive the late-listing and the delisting branches.
    thin = synthetic_asset_panel(; n_assets = 10, n_observations = 400, n_industries = 2,
                                 late_listing_proba = 1.0, delisting_proba = 1.0,
                                 rng = StableRNG(12))
    tmsk = thin.rd.pnl.amsk
    @test !any(view(tmsk, 1, 2:10))
    @test all(view(tmsk, :, 1))
end

@testset "Synthetic panel reconciles with its own model" begin
    svr = 0.4
    res = synthetic_asset_panel(; n_assets = 12, n_observations = 120, n_industries = 3,
                                systematic_variance_ratio = svr, rng = StableRNG(2024))
    rd, tr = res.rd, res.truth
    amsk = rd.pnl.amsk

    # The floor at -0.99 is part of the model, so the reconciliation carries it.
    recon = max.(tr.f * tr.B' .+ tr.eps, -0.99)
    @test isapprox(rd.X[amsk], recon[amsk]; rtol = 1e-12)

    # The market exposure is positive and bounded, and the industry block is one-hot.
    onehot = tr.B[:, 2:4]
    @test all(x -> 0.1 <= x <= 3.0, tr.B[:, 1])
    @test all(x -> x == 0 || x == 1, onehot)
    @test all(isone, sum(onehot; dims = 2))

    # The style exposures are standardised across the cross-section.
    styles = tr.B[:, 5:end]
    @test isapprox(vec(sum(styles; dims = 1)), zeros(10); atol = 1e-10)
    @test isapprox(vec(Statistics.std(styles; dims = 1, corrected = false)), ones(10);
                   rtol = 1e-10)

    # The systematic share of the cross-sectional variance is the caller's number.
    sysvar = Statistics.mean(Statistics.var(tr.f * tr.B'; dims = 2, corrected = false))
    @test isapprox(sysvar / (sysvar + Statistics.mean(tr.ivar)), svr; rtol = 1e-10)
end

@testset "Synthetic panel is reproducible from one seed" begin
    kwargs = (; n_assets = 7, n_observations = 45, n_industries = 2, missing_ratio = 0.05)
    a = synthetic_asset_panel(; kwargs..., rng = StableRNG(7))
    b = synthetic_asset_panel(; kwargs..., rng = StableRNG(7))
    c = synthetic_asset_panel(; kwargs..., rng = StableRNG(8))

    # `X` carries a `NaN` where an asset is not active, so equality is `isequal`.
    @test isequal(a.rd.X, b.rd.X)
    @test panel_feature_matrix(a.rd.pnl)[2] == panel_feature_matrix(b.rd.pnl)[2]
    @test panel_feature_matrix(a.rd.pnl)[1] == panel_feature_matrix(b.rd.pnl)[1]
    @test a.rd.ts == b.rd.ts
    @test a.rd.pnl.amsk == b.rd.pnl.amsk
    @test a.rd.pnl.emsk == b.rd.pnl.emsk
    @test a.truth.B == b.truth.B
    @test a.truth.f == b.truth.f
    @test a.truth.eps == b.truth.eps
    @test a.truth.ivar == b.truth.ivar
    @test !isequal(a.rd.X, c.rd.X)
end

@testset "Synthetic panel coverage and blanking" begin
    res = synthetic_asset_panel(; n_assets = 30, n_observations = 300, n_industries = 3,
                                missing_ratio = 0.2, rng = StableRNG(55))
    rd = res.rd
    amsk = rd.pnl.amsk
    obs_share = name -> begin
        col = panel_feature_matrix(rd.pnl)[2][:, :,
                                              findfirst(==(name * "::observed"),
                                                        panel_feature_matrix(rd.pnl)[1])]
        return sum(col[amsk]) / sum(amsk)
    end

    # A protected Panel Field is observed on every active cell; a named one sits at its
    # own coverage; an unnamed one sits at `1 - missing_ratio`.
    @test obs_share("adj_close") == 1
    @test obs_share("market_cap") == 1
    @test isapprox(obs_share("eps_ntm"), 0.8; atol = 0.05)
    @test isapprox(obs_share("ebitda_ttm"), 0.9; atol = 0.05)
    @test isapprox(obs_share("sales_ttm"), 0.8; atol = 0.05)

    # Nothing is blanked when the coverage is complete and no Panel Field is named below
    # one, so the observed columns of the unnamed fields are the active mask itself.
    none = synthetic_asset_panel(; n_assets = 6, n_observations = 40, n_industries = 2,
                                 missing_ratio = 0.0, rng = StableRNG(56))
    ncol = panel_feature_matrix(none.rd.pnl)[2][:, :,
                                                findfirst(==("sales_ttm::observed"),
                                                          panel_feature_matrix(none.rd.pnl)[1])]
    @test ncol == none.rd.pnl.amsk
end

@testset "Synthetic panel accounting identities" begin
    res = synthetic_asset_panel(; n_assets = 10, n_observations = 80, n_industries = 2,
                                missing_ratio = 0.0, rng = StableRNG(99))
    rd = res.rd
    amsk = rd.pnl.amsk
    field = name -> panel_feature_matrix(rd.pnl)[2][:, :,
                                                    findfirst(==(name),
                                                              panel_feature_matrix(rd.pnl)[1])]

    # The market capitalisation is the price times the share count, on every active cell.
    mcap = field("market_cap")[amsk]
    @test isapprox(mcap, (field("adj_close") .* field("adj_shares_outstanding"))[amsk];
                   rtol = 1e-10)

    # The levels every descriptor divides by stay strictly positive, and the gross margin
    # keeps the cost of revenue below the sales.
    @test all(x -> x > 0, mcap)
    @test all(x -> x > 0, field("adj_close")[amsk])
    @test all(x -> x > 0, field("total_assets")[amsk])
    @test all(x -> x >= 0, field("adj_volume")[amsk])
    @test all(x -> x >= 0, field("short_interest")[amsk])
    @test all(x -> x >= 0, field("eps_ntm_std")[amsk])
    @test all((field("cost_of_revenue_ttm") .< field("sales_ttm"))[amsk])
end

@testset "Synthetic panel helpers" begin
    # A constant cross-section standardises to zero rather than to a division by zero.
    @test synthetic_standardise(fill(3.0, 5)) == zeros(5)
    @test isapprox(synthetic_standardise([1.0, 2.0, 3.0]), [-sqrt(1.5), 0.0, sqrt(1.5)];
                   rtol = 1e-10)

    # The autoregressive filter keeps the stationary variance at one.
    rng = StableRNG(3)
    filt = synthetic_ar1_filter(randn(rng, 20000, 2), 0.9)
    @test isapprox(Statistics.var(filt[:, 1]), 1.0; rtol = 0.1)
    @test isapprox(Statistics.cor(filt[2:end, 1], filt[1:(end - 1), 1]), 0.9; rtol = 0.1)

    # The paths hit their stated annualised volatility and mean.
    paths = synthetic_ar1_paths(StableRNG(4), 20000, [0.2], [0.1], [0.0])
    @test isapprox(Statistics.std(paths[:, 1]) * sqrt(252), 0.2; rtol = 0.1)
    @test isapprox(Statistics.mean(paths[:, 1]) * 252, 0.1; atol = 0.05)

    # The fat-tailed draw is unit variance and heavier tailed than a normal.
    z = synthetic_fat_tailed_normal(StableRNG(5), 6.0, 40000)
    @test isapprox(Statistics.var(z), 1.0; rtol = 0.15)
    @test StatsBase.kurtosis(z) > 1
end

@testset "Synthetic panel statistical truths" begin
    res = synthetic_asset_panel(; n_assets = 25, n_observations = 1500, n_industries = 4,
                                rng = StableRNG(606))
    tr = res.truth

    # The idiosyncratic variances the generator reports are the ones it drew.
    realised = vec(Statistics.var(tr.eps; dims = 1, corrected = false))
    @test isapprox(Statistics.mean(realised ./ tr.ivar), 1.0; rtol = 0.15)

    # The idiosyncratic returns are close to uncorrelated with the factor returns.
    @test maximum(abs.(Statistics.cor(tr.f, tr.eps))) < 0.25
end

@testset "Synthetic panel refusals" begin
    @test_throws DomainError synthetic_asset_panel(; n_assets = 1)
    @test_throws DomainError synthetic_asset_panel(; n_observations = 1)
    @test_throws DomainError synthetic_asset_panel(; n_assets = 4, n_observations = 10,
                                                   n_industries = 0)
    @test_throws DomainError synthetic_asset_panel(; n_assets = 4, n_observations = 10,
                                                   n_industries = 17)
    @test_throws DomainError synthetic_asset_panel(; n_assets = 4, n_observations = 10,
                                                   systematic_variance_ratio = 0.0)
    @test_throws DomainError synthetic_asset_panel(; n_assets = 4, n_observations = 10,
                                                   systematic_variance_ratio = 1.0)
    @test_throws DomainError synthetic_asset_panel(; n_assets = 4, n_observations = 10,
                                                   late_listing_proba = 1.5)
    @test_throws DomainError synthetic_asset_panel(; n_assets = 4, n_observations = 10,
                                                   delisting_proba = -0.1)
    @test_throws DomainError synthetic_asset_panel(; n_assets = 4, n_observations = 10,
                                                   missing_ratio = 2.0)

    # One industry is the smallest classification the generator admits.
    one = synthetic_asset_panel(; n_assets = 4, n_observations = 20, n_industries = 1,
                                rng = StableRNG(1))
    @test one.truth.nf[2] == "industry=Real Estate"
    @test all(isone, one.truth.B[:, 2])
end

#=
The Asset Panel's DataFrame export (issue #929, map #643).

The library has no file format of its own, so `panel_dataframe` is its interchange: a caller
writes the table with whatever they already use. `asset_panel(df, mf)` reads the table back with
its Panel Manifest, and `test_06b_asset_panel.jl` tests that round trip (#1399). The file tests the three shapes a
call can take -- one Panel Field laid out as it stands, the long layout and the wide layout --
across the panel's static and time-varying shapes, and the two selections and the categorical
decoding that narrow each of them.

A `TensorPanelField` is the one Panel Field the oracle's own export skips.
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
@testset "The docstrings of 16_PanelDataFrame.jl against numbers" begin
    # panel_frame_columns returns the array of a numeric Panel Field, the codes of a
    # categorical one when decode = false, and a view of a tensor one, never a copy.
    pnl = panel_929_static()
    num, cat, ten = pnl.pf
    @test first(PortfolioOptimisers.panel_frame_columns(num, true))[2] === num.vals
    @test first(PortfolioOptimisers.panel_frame_columns(cat, false))[2] === cat.codes
    @test parent(first(PortfolioOptimisers.panel_frame_columns(ten, true))[2]) === ten.vals

    # An entry of `fields` or `assets` can be a Symbol, and a repeated name in `nx`
    # resolves to its first position.
    df = panel_dataframe(pnl; nx = ["A", "B", "C"], fields = [:mcap], assets = [:C, :A])
    @test names(df) == ["asset", "mcap"]
    @test df.mcap == [3.0, 1.0]
    @test PortfolioOptimisers.panel_frame_assets(["A", "B", "A"], ["A"]) == [1]

    # `layout` is checked in every call, also when `fields` names one Panel Field.
    @test_throws ArgumentError panel_dataframe(pnl; fields = "mcap", layout = :tall)

    # The long layout of a time-varying panel: before the filter, row (t - 1) n + k holds
    # observation t and asset k, and the filter keeps the rows where the active mask is true.
    tv = AssetPanel(;
                    pf = [NumericPanelField(; name = "x",
                                            vals = [11.0 12.0 13.0; 21.0 22.0 23.0])],
                    amsk = [true true true; true true true],
                    emsk = [true false true; false true true])
    df = panel_dataframe(tv; nx = ["A", "B", "C"], ts = ["t1", "t2"])
    @test df.observation == ["t1", "t1", "t1", "t2", "t2", "t2"]
    @test df.asset == ["A", "B", "C", "A", "B", "C"]
    @test df.x == [11.0, 12.0, 13.0, 21.0, 22.0, 23.0]
    @test df.emsk == [true, false, true, false, true, true]
    @test df.observation isa Vector{String}

    # A panel with no Panel Field and two masks gives the key and mask columns alone.
    mo = AssetPanel(; amsk = [true false; true true], emsk = [true false; false true])
    df = panel_dataframe(mo; nx = ["A", "B"])
    @test names(df) == ["observation", "asset", "emsk"]
    @test df.asset == ["A", "A", "B"]
    @test df.emsk == [true, false, true]
    df = panel_dataframe(mo; nx = ["A", "B"], layout = :wide)
    @test names(df) == ["observation", "amsk@A", "amsk@B", "emsk@A", "emsk@B"]
    @test df[!, "amsk@B"] == [false, true]

    # An empty asset selection gives a table with its columns and no row.
    df = panel_dataframe(tv; assets = String[])
    @test names(df) == ["observation", "asset", "x", "emsk"]
    @test DataFrames.nrow(df) == 0

    # The element type of the values stays as the panel holds it.
    @test eltype(panel_dataframe(AssetPanel(;
                                            pf = [NumericPanelField(; name = "x",
                                                                    vals = Float32[1, 2])])).x) ===
          Float32
    rp = AssetPanel(; pf = [NumericPanelField(; name = "x", vals = [1//2 1//3; 1//4 1//5])],
                    amsk = trues(2, 2), emsk = trues(2, 2))
    @test eltype(panel_dataframe(rp; layout = :wide)[!, "x@1"]) === Rational{Int}
    @test panel_dataframe(rp; fields = "x")[!, "2"] == [1//3, 1//5]

    # Every shape holds copies: a change to a cell of the table leaves the panel unchanged.
    # The static long layout shared the memory of the panel before, through the view that
    # `vec(permutedims(v))` returns for a vector.
    for (kw, col) in (((;), "mcap"), ((; decode = false), "sector"), ((;), "beta=size"),
                      ((; layout = :wide), "mcap@1"), ((; fields = "mcap"), "1"))
        p = panel_929_static()
        before = deepcopy(p)
        df = panel_dataframe(p; kw...)
        df[1, col] += 100
        @test p.pf[1].vals == before.pf[1].vals
        @test p.pf[2].codes == before.pf[2].codes
        @test p.pf[3].vals == before.pf[3].vals
    end
    p = panel_929_timevarying()
    df = panel_dataframe(p; layout = :wide)
    df[1, "amsk@1"] = false
    df[1, "mcap@1"] = -1.0
    @test p.amsk[1, 1]
    @test p.pf[1].vals[1, 1] == 1.0

    # Two columns with one name raise an error. Before, the second column replaced the first
    # and the table lost data with no error.
    mk(names...) = AssetPanel(;
                              pf = [NumericPanelField(; name = n,
                                                      vals = [1.0 2.0; 3.0 4.0])
                                    for n in names], amsk = [true true; true false],
                              emsk = [true false; true false])
    for (p, kw) in ((mk("asset"), (;)), (mk("observation"), (;)), (mk("emsk"), (;)),
                    (mk("amsk"), (; layout = :wide)), (mk("emsk"), (; layout = :wide)),
                    (mk("x"), (; nx = ["observation", "B"], fields = "x")),
                    (mk("x"), (; nx = ["A", "A"], layout = :wide)),
                    (AssetPanel(; pf = [NumericPanelField(; name = "asset", vals = [1.0, 2.0])]), (;)))
        @test_throws ArgumentError panel_dataframe(p; kw...)
    end
    # The AssetPanel constructor refuses two Panel Fields that derive one column name
    # (#1335), so these two panels never reach panel_dataframe.
    @test_throws ArgumentError AssetPanel(;
                                          pf = [NumericPanelField(; name = "x",
                                                                  vals = [1.0, 2.0],
                                                                  omsk = [true, false]),
                                                NumericPanelField(; name = "x::observed",
                                                                  vals = [5.0, 6.0])])
    @test_throws ArgumentError AssetPanel(;
                                          pf = [NumericPanelField(; name = "beta=size",
                                                                  vals = [1.0, 2.0]),
                                                TensorPanelField(; name = "beta",
                                                                 axis = "f",
                                                                 labels = ["size"],
                                                                 vals = reshape([7.0, 8.0],
                                                                                2, 1))])
    msg = try
        panel_dataframe(mk("asset"))
    catch e
        sprint(showerror, e)
    end
    @test occursin("already holds a column named \"asset\"", msg)
    df = DataFrames.DataFrame(; a = [1])
    @test_throws ArgumentError PortfolioOptimisers.panel_frame_column!(df, "a", [2])
    @test df.a == [1]
end

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
    return (NumericPanelField, f.name, isnothing(f.omsk), isnothing(f.pmsk))
end
function panel_1399_schema(f::CategoricalPanelField)
    return (CategoricalPanelField, f.name, isnothing(f.omsk), isnothing(f.pmsk), f.levels)
end
function panel_1399_schema(f::TensorPanelField)
    return (TensorPanelField, f.name, isnothing(f.omsk), isnothing(f.pmsk), f.axis,
            f.labels, f.groups)
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
            ((panel_1399_values(f1), panel_1399_values(f2)), (f1.omsk, f2.omsk),
             (f1.pmsk, f2.pmsk))
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
    @test names(mf) ==
          ["kind", "field", "label", "group", "observed", "placeholder", "grouped"]
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
    # A wide table is read by position, so a sorted table read back permuted with no error
    # (R7 of #1630). Its labels must be the ones of the manifest, in order.
    msg = "its \"observation\" column names them in another order or with other labels"
    @test_throws msg asset_panel(W[[2, 1, 3, 4], :], mf)
    r = copy(W)
    r.observation[4] = Date(1999, 1, 1)
    @test_throws msg asset_panel(r, mf)
    # A table with no "observation" column is still read by position.
    @test panel_1399_equal(pnl,
                           asset_panel(DataFrames.select(W, DataFrames.Not("observation")),
                                       mf))
    # A long table of a panel with no active cell has no row, and CSV.jl types each of its
    # columns as `Missing`. The reader names the columns and the remedy (R7 of #1630).
    z = AssetPanel(; pf = [NumericPanelField(; name = "x", vals = ones(2, 2))],
                   amsk = falses(2, 2), emsk = falses(2, 2))
    @test size(asset_panel(panel_dataframe(z; layout = :long), panel_manifest(z)).amsk) ==
          (2, 2)
    @test_throws "hold no value" panel_1399_csv(panel_dataframe(z; layout = :long),
                                                panel_manifest(z))
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

# #1631: a placeholder mask comes back from both layouts, and its manifest row records it.
@testset "asset_panel(df, mf): a placeholder mask survives the round trip (#1631)" begin
    amsk = Bool[1 1; 1 1; 0 1]
    pnl = asset_panel([NumericPanelInput(; name = "mcap",
                                         vals = [NaN 1.0; 2.0 NaN; 3.0 NaN],
                                         alg = ForwardPanelFill()),
                       CategoricalPanelInput(; name = "sector",
                                             vals = [missing "a"; "b" "a"; "b" missing],
                                             alg = ForwardPanelFill(; val = "z")),
                       TensorPanelInput(; name = "beta", axis = "factor",
                                        labels = ["u", "v"],
                                        vals = cat([NaN 1.0; 2.0 3.0; 4.0 5.0],
                                                   [1.0 NaN; 2.0 3.0; 4.0 5.0]; dims = 3),
                                        alg = ForwardPanelFill())]; amsk = amsk)
    @test panel_field(pnl, "mcap").pmsk == Bool[1 0; 0 0; 0 0]
    @test panel_field(pnl, "sector").pmsk == Bool[1 0; 0 0; 0 0]
    mf = panel_manifest(pnl; nx = ["A", "B"])
    @test isequal(mf.placeholder[mf.kind .∈ Ref(["numeric", "categorical", "tensor"])],
                  [true, true, true])
    for decode in (true, false)
        w = panel_1399_trip(pnl; nx = ["A", "B"], layout = :wide, decode)
        @test panel_1399_equal(pnl, w)
        l = panel_1399_trip(pnl; nx = ["A", "B"], layout = :long, decode)
        @test panel_1399_equal(pnl, l; active_only = true)
    end
end
