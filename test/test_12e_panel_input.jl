#=
The one bridge from a `UniverseSets` key to a Panel Field.

`#806` deleted the graded edge-authoring program and replaced it with `panel_input`. A key is one
vector over the asset axis, which is the static raw form of one Panel Field, so the bridge is a
dispatch on the key's element type and nothing more: a string-valued key is a classification and a
number-valued key is a quantity.

Every matrix the deleted grammar wrote is one static `assets × nodes` matrix, and a static
`TensorPanelInput` admits any such matrix as data, with none of the grammar's parser, resolution
order or documented hazards. The last testset pins that claim.
=#
const TAXD = Dict{String, Any}("nx" => ["A", "B", "C", "D"],
                               "nx_sector" => ["Fin", "Tech", "Fin", "Ind"],
                               "nx_rating" => [1.0, 2.0, 3.0, 4.0],
                               "nx_mixed" => Any["Fin", 2.0, "Fin", 4.0],
                               "sector" => ["Fin", "Tech", "Fin", "Ind"])
const TAXS = UniverseSets(; dict = TAXD)

@testset "A key becomes a Panel Field, by element type" begin
    # A string-valued key is a classification.
    ci = panel_input(TAXS, "nx_sector")
    @test isa(ci, CategoricalPanelInput)
    @test ci.vals == TAXD["nx_sector"]
    # A number-valued key is a quantity.
    ni = panel_input(TAXS, "nx_rating")
    @test isa(ni, NumericPanelInput)
    @test ni.vals == TAXD["nx_rating"]

    # The field is named by the key with the asset prefix and its underscore stripped,
    # because every Panel Field of a panel is asset-parallel by construction.
    @test ci.name == "sector"
    @test ni.name == "rating"
    # A key with no prefix keeps its name.
    @test panel_input(TAXS, "sector").name == "sector"
    # And `name` overrides the derivation.
    @test panel_input(TAXS, "nx_sector"; name = "gics").name == "gics"

    # A mixed element type is refused: neither reading is right, and guessing one is worse
    # than saying so. The message names both ways out.
    res = @test_throws ArgumentError panel_input(TAXS, "nx_mixed")
    @test occursin("CategoricalPanelInput", res.value.msg)
    @test occursin("NumericPanelInput", res.value.msg)

    # An absent key is a `KeyError` naming the consumer that asked for it.
    @test_throws KeyError panel_input(TAXS, "nope")
end

@testset "A pair forces the type, and there is no `kind` keyword" begin
    # A numeric rating read as a label rather than as a number.
    fi = panel_input(TAXS, "nx_rating" => CategoricalPanelInput)
    @test isa(fi, CategoricalPanelInput)
    @test fi.name == "rating"
    @test fi.vals == string.(TAXD["nx_rating"])

    # And the other way, which is how a mixed key resolves.
    @test isa(panel_input(TAXS, "nx_sector" => CategoricalPanelInput),
              CategoricalPanelInput)

    # The scalar form takes `levels` and `alg`; the numeric form declares no levels and says
    # so rather than ignoring the keyword.
    lv = panel_input(TAXS, "nx_sector"; levels = ["Tech", "Fin", "Ind"])
    @test lv.levels == ["Tech", "Fin", "Ind"]
    @test_throws ArgumentError panel_input(TAXS, "nx_rating"; levels = ["a"])
    @test isa(panel_input(TAXS, "nx_rating"; alg = ConstantPanelFill(; val = 0.0)).alg,
              ConstantPanelFill)
end

@testset "The vector form maps the rule, and takes no per-field keyword" begin
    inps = panel_input(TAXS, ["nx_sector", "nx_rating"])
    @test length(inps) == 2
    @test [i.name for i in inps] == ["sector", "rating"]
    @test isa(inps[1], CategoricalPanelInput)
    @test isa(inps[2], NumericPanelInput)

    # A pair entry forces the type in the vector form too.
    mixed = panel_input(TAXS, ["nx_sector", "nx_rating" => CategoricalPanelInput])
    @test isa(mixed[2], CategoricalPanelInput)

    @test_throws PortfolioOptimisers.IsEmptyError panel_input(TAXS, String[])
end

@testset "A taxonomy reaches the panel, and the distance measures it" begin
    pnl = asset_panel(panel_input(TAXS, ["nx_sector", "nx_rating"]))
    @test PortfolioOptimisers.panel_is_static(pnl)
    nz, Z = panel_feature_matrix(pnl)
    @test nz == ["sector=Fin", "sector=Ind", "sector=Tech", "rating"]
    @test Z[:, 1] == [1.0, 0.0, 1.0, 0.0]
    @test Z[:, 4] == TAXD["nx_rating"]

    # A caller who selected a taxonomy block by key now writes the field name, and a block of
    # levels is a paired entry away.
    rng = StableRNG(20260910)
    X = randn(rng, 50, 4) / 100
    rd = ReturnsResult(; nx = TAXD["nx"], X = X, pnl = pnl)
    de = FeatureDistance(; sel = ["sector" => ["Fin", "Tech"]], strict = true)
    @test size(feature_matrix(de, nothing, rd, X)) == (4, 2)
    @test clusterise(ClustersEstimator(; de = de), rd) isa
          PortfolioOptimisers.AbstractClusteringResult

    # A nested taxonomy is several keys, so several fields, all in one panel.
    two = asset_panel(vcat(panel_input(TAXS, ["nx_sector"]),
                           [panel_input(TAXS, "sector"; name = "gics")]))
    @test [f.name for f in two.pf] == ["sector", "gics"]
end

@testset "A static taxonomy joins a time-varying panel by the lazy lift" begin
    rng = StableRNG(20260911)
    T = 12
    mcap = abs.(randn(rng, T, 4)) .+ 1
    pnl = asset_panel([NumericPanelInput(; name = "mcap", vals = mcap),
                       panel_input(TAXS, "nx_sector")])
    @test !PortfolioOptimisers.panel_is_static(pnl)
    f = PortfolioOptimisers.panel_field(pnl, "sector")
    @test size(f.codes) == (T, 4)
    # Every row is the same classification, stored once.
    @test all(t -> f.codes[t, :] == f.codes[1, :], 1:T)
    @test isa(f.codes, PortfolioOptimisers.RepeatedLeading)
    # A lifted field carries no observed mask, because every cell was observed.
    @test isnothing(f.omsk)
    # The masks are the second lift signal.
    lifted = asset_panel([panel_input(TAXS, "nx_sector")]; amsk = trues(T, 4),
                         emsk = trues(T, 4))
    @test !PortfolioOptimisers.panel_is_static(lifted)
end

@testset "Every matrix the graded program wrote is a static tensor input" begin
    #=
    The round-trip argument that closed the grammar. A scaled block, a cross edge, an asset
    node, a mixed axis and an all-zero row are all cells of one `assets × nodes` matrix, and
    a static `TensorPanelInput` takes any such matrix as data.
    =#
    nodes = ["Fin", "Tech", "Ind", "A", "cross"]
    Zg = Float64[2.0 0.0 0.0 1.0 0.5
                 0.0 2.0 0.0 0.0 0.5
                 2.0 0.0 0.0 0.0 0.0
                 0.0 0.0 0.0 0.0 0.0]
    pnl = asset_panel([TensorPanelInput(; name = "program", axis = "node", labels = nodes,
                                        vals = Zg)])
    nz, Z = panel_feature_matrix(pnl)
    @test nz == ["program=" .* n for n in nodes]
    @test Z == Zg
    # An all-zero row survives, as the grammar's did.
    @test all(iszero, Z[4, :])

    # And a selector names a node directly, with no second namespace to order.
    de = FeatureDistance(; sel = ["program" => ["Fin", "cross"]], strict = true)
    rng = StableRNG(20260912)
    X = randn(rng, 40, 4) / 100
    rd = ReturnsResult(; nx = TAXD["nx"], X = X, pnl = pnl)
    @test feature_matrix(de, nothing, rd, X) == Zg[:, [1, 5]]
end
