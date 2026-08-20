include(joinpath(@__DIR__, "test12_setup.jl"))
using Clustering, LinearAlgebra

#=
`AssetSetsFeatures` is the map's third producer, and the only *exogenous* one: a sector or
country classification is structure the return correlations do not contain, which is what
`skfolio/skfolio` issue 241 asks for. Every other producer derives `Z` from the returns.

The taxonomy below is a nested GICS-shaped one over `test12_setup`'s real twenty-asset
universe: sector ⊃ industry ⊃ sub-industry, so two assets sharing a finer level share every
coarser one. That nesting is what makes `cos = shared / L` read as "how many classification
levels do these two agree on".
=#
const TAX = UniverseSets(; xkey = "nx",
                         dict = Dict("nx" => rd.nx,
                                     "nx_sector" =>
                                         ["Technology", "Technology", "Financials",
                                          "ConsumerDiscretionary", "Energy", "Industrials",
                                          "ConsumerDiscretionary", "HealthCare",
                                          "Financials", "ConsumerStaples", "HealthCare",
                                          "HealthCare", "Technology", "ConsumerStaples",
                                          "HealthCare", "ConsumerStaples", "Energy",
                                          "HealthCare", "ConsumerStaples", "Energy"],
                                     "nx_industry" =>
                                         ["Hardware", "Semiconductors", "Banks",
                                          "SpecialtyRetail", "IntegratedOil", "Aerospace",
                                          "SpecialtyRetail", "Pharma", "Banks", "Beverages",
                                          "Pharma", "Pharma", "Software", "Beverages",
                                          "Pharma", "HouseholdProducts",
                                          "ExplorationProduction", "ManagedCare",
                                          "FoodRetail", "IntegratedOil"],
                                     "nx_subindustry" =>
                                         ["ConsumerElectronics", "Semiconductors",
                                          "DiversifiedBanks", "ComputerRetail",
                                          "IntegratedOil", "AerospaceDefense",
                                          "HomeImprovementRetail", "Pharma",
                                          "DiversifiedBanks", "SoftDrinks", "Pharma",
                                          "Pharma", "SystemsSoftware", "SoftDrinks",
                                          "Pharma", "HouseholdProducts", "OilGasEP",
                                          "ManagedHealthCare", "Hypermarkets",
                                          "IntegratedOil"]))
const KEYS = ["nx_sector", "nx_industry", "nx_subindustry"]
const NA = length(rd.nx)
const IDX = Dict(n => i for (i, n) in pairs(rd.nx))

@testset "asset_sets_features: shape, dtype and the equal-norm identity" begin
    Z = asset_sets_features(KEYS, TAX)

    # `Float64`, never the `BitMatrix` `asset_sets_matrix` returns: `AngularDist` delegates
    # to Distances' gemm `_pairwise!(::CosineDist, …)`, which a bit matrix would forfeit.
    @test isa(Z, Matrix{Float64})
    @test all(z -> z in (0.0, 1.0), Z)
    # Assets-major, matching the carried layout. The feature count is the total number of
    # distinct group values across the three keys.
    @test size(Z, 1) == NA
    @test size(Z, 2) == sum(length(unique(TAX.dict[k])) for k in KEYS)

    # `asset_sets_matrix` builds its groups from `unique(all_sets)`, so every key is a
    # *partition*: each asset lands in exactly one group per key. Every row therefore has
    # exactly `L` ones and norm `sqrt(L)` -- the property that makes this the one producer
    # needing nothing from the transform question.
    L = length(KEYS)
    @test all(==(Float64(L)), sum(Z; dims = 2))
    @test all(≈(sqrt(L)), norm.(eachrow(Z)))

    # Equal norms make the cosine similarity exactly the shared-group count over `L`, and
    # the angular distance exactly `acos` of it -- an identity, not an approximation, so it
    # is asserted on every pair rather than on a hand-picked one.
    G = Z * transpose(Z)                      # G[i, j] == shared(i, j)
    D = distance(FeatureDistance(), Z)
    @test all(G .== round.(G))                # integer counts, since Z is 0/1
    @test all(0 .<= G .<= L)
    @test isapprox(cos.(pi .* D), G ./ L; atol = 1e-14)

    # `ReturnsResult` requires `nz` whenever `Z` is set, so the names are paired with the
    # columns rather than left to the caller to rebuild. They are qualified by key because a
    # *nested* taxonomy reuses its values across levels -- `IntegratedOil` is both an
    # industry and a sub-industry here -- and bare values would collide.
    nz = asset_sets_feature_names(KEYS, TAX)
    @test length(nz) == size(Z, 2)
    @test allunique(nz)
    @test !allunique(reduce(vcat, [unique(TAX.dict[k]) for k in KEYS]))
    @test all(startswith(nz[k], KEYS[1]) for k in 1:length(unique(TAX.dict[KEYS[1]])))
    # Name and column agree: column `k` is exactly the indicator of the group `nz[k]` names.
    for (k, name) in pairs(nz)
        key, grp = split(name, "="; limit = 2)
        @test Z[:, k] == Float64.(TAX.dict[key] .== grp)
    end

    # The two entry points are the same matrix; they differ only in which carrier holds it.
    @test asset_sets_features(KEYS, TAX) == Z
    # Column order follows key order, so a permuted `vals` is a column permutation -- which
    # every `Distances.jl` semimetric is invariant to.
    @test distance(FeatureDistance(), asset_sets_features(reverse(KEYS), TAX)) == D
end

@testset "A nested taxonomy grades relatedness by depth" begin
    Z = asset_sets_features(KEYS, TAX)
    D = distance(FeatureDistance(), Z)
    d(a, b) = D[IDX[a], IDX[b]]

    # Four pairs agreeing at 3, 2, 1 and 0 of the three levels. Nesting is what makes this
    # a *depth*: JNJ and LLY share the sub-industry, hence the industry, hence the sector.
    @test d("JNJ", "LLY") == 0.0                       # 3/3: same sub-industry
    @test d("BBY", "HD") ≈ acos(2 / 3) / pi            # 2/3: SpecialtyRetail, differing sub
    @test d("CVX", "RRC") ≈ acos(1 / 3) / pi           # 1/3: Energy, differing industry
    @test d("AAPL", "JPM") ≈ 0.5                       # 0/3: orthogonal rows

    @test d("JNJ", "LLY") < d("BBY", "HD") < d("CVX", "RRC") < d("AAPL", "JPM")

    # The grading is the *taxonomy's*, not a rescaling of the returns' own: the correlation
    # distance disagrees about which of these pairs is closest.
    Dc = distance(Distance(; alg = CanonicalDistance()), PortfolioOptimisersCovariance(),
                  rd.X)
    dc(a, b) = Dc[IDX[a], IDX[b]]
    @test dc("JNJ", "LLY") > 0
    @test argmin([d("JNJ", "LLY"), d("BBY", "HD"), d("CVX", "RRC"), d("AAPL", "JPM")]) !=
          argmin([dc("JNJ", "LLY"), dc("BBY", "HD"), dc("CVX", "RRC"), dc("AAPL", "JPM")])

    # A *crossed* taxonomy reads the same count differently -- independent attributes
    # shared rather than levels agreed -- and is a genuinely different matrix.
    @test asset_sets_features(["nx_sector", "nx_industry"], TAX) != Z[:, 1:size(Z, 2)]
end

@testset "Two keys is a floor, and duplicates are refused" begin
    # A single partition is one-hot. Every `Distances.jl` semimetric is invariant to
    # permuting coordinates, so its distance matrix takes at most two values for *any*
    # metric and clustering it returns the partition it was built from. The check is at the
    # kernel and at construction, from one encoding of the rule.
    @test_throws ArgumentError asset_sets_features(["nx_sector"], TAX)
    @test_throws ArgumentError asset_sets_features(String[], TAX)
    @test_throws ArgumentError AssetSetsFeatures(; vals = ["nx_sector"])
    @test_throws ArgumentError AssetSetsFeatures(; vals = String[])

    # And the proof of why: the one-hot matrix a single key would give is two-valued.
    one_hot = Float64.(transpose(asset_sets_matrix("nx_sector", TAX)))
    for de in (FeatureDistance(),
               FeatureDistance(; metric = PortfolioOptimisers.Distances.Jaccard()),
               FeatureDistance(; metric = PortfolioOptimisers.Distances.BrayCurtis()))
        @test length(unique(distance(de, one_hot))) == 2
    end

    # A repeated key doubles its block and silently reweights that partition against the
    # others, so it is refused rather than deduplicated.
    @test_throws ArgumentError asset_sets_features(["nx_sector", "nx_sector"], TAX)
    @test_throws ArgumentError AssetSetsFeatures(; vals = ["a", "b", "a"])

    # A key that is not in the taxonomy, or whose length does not match the universe, is
    # `asset_sets_matrix`'s existing job.
    @test_throws KeyError asset_sets_features(["nx_sector", "nx_nope"], TAX)

    # Both consumers of a caller-supplied taxonomy key read `sets.dict` through the one
    # `taxonomy_column` helper, so the miss names the consumer and suggests the spelling
    # instead of raising the bare `KeyError` a plain `sets.dict[key]` gives.
    for (f, need) in ((asset_sets_features, "asset_sets_matrix"),
                      (asset_sets_feature_names, "asset_sets_feature_names"))
        err = try
            f(["nx_sector", "nx_sectr"], TAX)
        catch e
            e
        end
        @test isa(err, KeyError)
        msg = sprint(showerror, err)
        @test occursin(need, msg)
        @test occursin("did you mean `nx_sector`?", msg)
    end
    short = UniverseSets(; xkey = "nx",
                         dict = Dict("nx" => rd.nx, "nx_sector" => TAX.dict["nx_sector"],
                                     "shortgroup" => ["a", "b"]))
    @test_throws AssertionError asset_sets_features(["nx_sector", "shortgroup"], short)
end

@testset "AssetSetsFeatures: the producer, and the feature axis is groups" begin
    ze = AssetSetsFeatures(; vals = KEYS)
    @test ze.vals == KEYS
    @test isa(ze, PortfolioOptimisers.AbstractFeatureMatrixEstimator)

    # The producer is the bare entry point with `FeaturePrior.sets` supplying the taxonomy,
    # which is what finally makes that field live -- it shipped unused with the carrier.
    pe = FeaturePrior(; ze = ze, sets = TAX)
    pr = prior(pe, rd)
    @test pr.Z == asset_sets_features(KEYS, TAX)

    # Purely additive: every moment is the wrapped estimator's, untouched.
    pr0 = prior(EmpiricalPrior(), rd)
    @test pr.mu == pr0.mu
    @test pr.sigma == pr0.sigma

    # The columns index groups even when the group count coincides with the asset count: a
    # coincidence of counts is not a claim about what they mean, and an asset view must not
    # truncate every row's feature vector on the strength of it.
    sq = UniverseSets(; xkey = "nx",
                      dict = Dict("nx" => rd.nx,
                                  "nx_pairA" => string.(repeat(1:(NA ÷ 2); inner = 2)),
                                  "nx_pairB" => string.(repeat(1:(NA ÷ 2); outer = 2))))
    prsq = prior(FeaturePrior(; ze = AssetSetsFeatures(; vals = ["nx_pairA", "nx_pairB"]),
                              sets = sq), rd)
    @test size(prsq.Z, 2) == NA
    i = [1, 3, 5]
    @test size(PortfolioOptimisers.port_opt_view(prsq, i).Z) == (length(i), NA)

    # The taxonomy is the whole input, so an absent `sets` is a missing argument rather than
    # a defaulted one: a returns-derived substitute would be the exact endogeneity this
    # producer exists to escape.
    res = @test_throws PortfolioOptimisers.IsNothingError prior(FeaturePrior(; ze = ze), rd)
    @test occursin("FeaturePrior.sets", res.value.msg)
end

@testset "Views: the producer is configuration, the taxonomy is the data" begin
    ze = AssetSetsFeatures(; vals = KEYS)
    pe = FeaturePrior(; ze = ze, sets = TAX)
    i = [1, 3, 5, 7, 9, 11]

    # The producer holds key *names*, so it is configuration and passes through unchanged --
    # the `RegressionFeatures` treatment, reached via `feature_estimator_view`'s delegation
    # to the universal `port_opt_view` leaf fallback.
    @test PortfolioOptimisers.feature_estimator_view(ze, i) === ze
    pev = PortfolioOptimisers.port_opt_view(pe, i)
    @test pev.ze === ze

    # The `sets` are the data, and they slice: every `key`-prefixed group follows `i`.
    @test pev.sets.dict["nx"] == rd.nx[i]
    for k in KEYS
        @test pev.sets.dict[k] == TAX.dict[k][i]
    end

    # So the viewed producer recomputes the taxonomy of the *subproblem*, rather than
    # carrying the full universe's columns into it. The feature axis is rebuilt, not
    # sliced: a group with no members left in the view disappears.
    Zv = prior(pev, rd.X[:, i]).Z
    @test size(Zv, 1) == length(i)
    @test size(Zv, 2) == sum(length(unique(TAX.dict[k][i])) for k in KEYS)
    @test all(==(Float64(length(KEYS))), sum(Zv; dims = 2))

    # A key not prefixed by `sets.xkey` is not sliced by the view, which does *not* fail
    # silently: `asset_sets_matrix`'s length check throws on the next call because the
    # sliced universe no longer matches the unsliced group. Name taxonomy keys with the
    # prefix if they are to survive a fold or an NCO cluster.
    bare = UniverseSets(; xkey = "nx",
                        dict = Dict("nx" => rd.nx, "sector" => TAX.dict["nx_sector"],
                                    "nx_industry" => TAX.dict["nx_industry"]))
    peb = FeaturePrior(; ze = AssetSetsFeatures(; vals = ["sector", "nx_industry"]),
                       sets = bare)
    @test_throws AssertionError prior(PortfolioOptimisers.port_opt_view(peb, i), rd.X[:, i])
end

@testset "A taxonomy drives a FeatureDistance end to end, from both carriers" begin
    Z = asset_sets_features(KEYS, TAX)
    fde = FeatureDistance()
    cde = Distance(; alg = CanonicalDistance())

    # Carrier one: the user's own data, under the default `z_src = :data`. This is why
    # `asset_sets_features` is public in its own right -- a producer alone runs inside
    # `prior(pe::FeaturePrior, …)` and could only ever feed the derived carrier.
    rdz = ReturnsResult(; nx = rd.nx, X = rd.X, nf = rd.nf, F = rd.F, ts = rd.ts,
                        nz = asset_sets_feature_names(KEYS, TAX), Z = Z)
    # Carrier two: derived, via the producer.
    pe = FeaturePrior(; ze = AssetSetsFeatures(; vals = KEYS), sets = TAX)
    pr = prior(pe, rd)
    @test pr.Z == Z

    # The two carriers agree on the matrix, hence on the distance -- they differ only in
    # what a fold does to them, not in what they say about the universe.
    @test distance(fde, pr.Z) == distance(fde, Z)

    cle_f = ClustersEstimator(; de = fde)
    cle_c = ClustersEstimator(; de = cde)
    clr_d = clusterise(cle_f, rdz)
    clr_p = clusterise(cle_f, pr; z_src = :prior)
    clr_c = clusterise(cle_c, rd)
    @test clr_d.D == distance(fde, Z)
    @test clr_p.D == clr_d.D

    # The headline claim: a taxonomy clusters a real universe *differently* from the
    # returns correlation. A test that passed on an ignored `Z` would be worthless.
    @test clr_d.D != clr_c.D
    @test clr_d.res.merges != clr_c.res.merges

    # And it drives a full solve, from either carrier, to weights that differ from the
    # correlation-driven ones.
    hopt_d = HierarchicalOptimiser(; cle = cle_f, slv = slv)
    hopt_p = HierarchicalOptimiser(; pe = pe, cle = cle_f, slv = slv, z_src = :prior)
    hopt_c = HierarchicalOptimiser(; cle = cle_c, slv = slv)
    for oe in (HierarchicalRiskParity, HierarchicalEqualRiskContribution)
        wd = optimise(oe(; opt = hopt_d), rdz)
        wp = optimise(oe(; opt = hopt_p), rd)
        wc = optimise(oe(; opt = hopt_c), rd)
        for res in (wd, wp, wc)
            @test isapprox(sum(res.w), 1)
            @test all(isfinite, res.w)
        end
        @test wd.clr.D == distance(fde, Z)
        @test wp.clr.D == distance(fde, Z)
        # Same matrix through both carriers, so the same weights -- the carriers are two
        # routes to one answer.
        @test isapprox(wd.w, wp.w)
        @test wd.w != wc.w
    end
end

#=
Everything below is the *graded* contract: `vals` as an ordered edge-authoring program over
a **declared** feature axis, rather than a list of keys whose partitions are stacked. The
two share a type by dispatch on element type, because the grammar strictly subsumes the key
list -- `"nx_sector" => 2.0` emits the same columns as `"nx_sector"`, only scaled.

`GSETS`/`GPROG` are the grilling's worked example, and `GEXP` is the matrix it was verified
against by hand before a line of this was written. A mismatch here is a bug in the
implementation, not in the table.
=#
const GSETS = UniverseSets(; xkey = "nx", zkey = "nz",
                           dict = Dict{String, Any}("nx" => ["A", "B", "C"],
                                                    "nz" =>
                                                        ["A", "B", "C", "Tech", "Finance",
                                                         "US", "UK", "esg"],
                                                    "nx_sector" =>
                                                        ["Tech", "Tech", "Finance"],
                                                    "nx_country" => ["US", "UK", "UK"],
                                                    "nx_esg" => [0.30, 0.80, 0.50],
                                                    "AC" => ["A", "C"],
                                                    "Tech_group" => ["A", "B"]))
const GPROG = ["nx_sector" => 2.0,                             # 1: diagonal, categorical
               "nx_country" => 1.0,                            # 2: diagonal, categorical
               "nx_esg" => Scale(1.3),                          # 3: diagonal, numeric
               "nx_country" => "UK" => 0.5,                     # 4: diagonal, restricted
               "C" => ["nx_country" => "US" => 0.3],            # 5: asset row scope
               "nx_sector" =>
                   "Tech" => ["nx_country" => "UK" => 0.4, "nx_country" => "US" => 0.1,
                              "nx_esg" => 0.7],      # 6: taxonomy row scope
               "Tech_group" => ["nx_country" => "UK" => 0.4, "nx_country" => "US" => 0.1,
                                "nx_esg" => 0.7],               # 7: group row scope
               "B" => ["nx_esg" => 0.9, "nx_sector" => "Finance" => 0.2],        # 8: asset row scope
               "AC" => ["nx_sector" => "Finance" => 0.5, "nx_country" => "UK" => 0.3],           # 9: group row scope
               "B" => "AC" => 0.2]                              # 10: group as target
const GEXP = [0.0 0.0 0.0 2.0 0.5 0.1 0.3 0.7
              0.2 0.0 0.2 2.0 0.2 0.1 0.4 0.9
              0.0 0.0 0.0 0.0 0.5 0.3 0.3 0.65]

@testset "The graded grammar reproduces the worked example, cell by cell" begin
    Z = asset_sets_features(GPROG, GSETS)
    @test isa(Z, Matrix{Float64})
    @test size(Z) == (3, 8)
    # Asserted whole rather than summarised: the three rows exercise different parts of the
    # grammar and the two easiest cells to get wrong are interior ones.
    @test Z == GEXP
    for i in axes(GEXP, 1), k in axes(GEXP, 2)
        @test Z[i, k] == GEXP[i, k]
    end

    nz = GSETS.dict["nz"]
    z(a, n) = Z[findfirst(==(a), GSETS.dict["nx"]), findfirst(==(n), nz)]

    # Entry 9 overrides entry 1 across a diagonal write and an explicit target: `C` is in
    # `Finance`, so entry 1 wrote 2.0 there, and entry 9's fully-qualified target replaced
    # it. Last wins, and it wins across productions, not only within one.
    @test z("C", "Finance") == 0.5

    # `Scale(1.3)` against a *numeric* key resolves per cell against that asset's own datum,
    # and `C`'s survives untouched because nothing later writes its `esg` node.
    @test z("C", "esg") == 1.3 * 0.50
    @test z("C", "esg") == 0.65

    # An absolute set beats both a `Scale` (entry 3) and an earlier explicit target
    # (entries 6/7), because every write is an overwrite and never a read-modify-write.
    @test z("B", "esg") == 0.9
    @test z("A", "esg") == 0.7

    # Entry 10: a *group* in target position expands to one asset node per member.
    @test z("B", "A") == 0.2
    @test z("B", "C") == 0.2
    @test z("B", "B") == 0.0

    # Every unwritten cell is zero -- the graded default, not the key path's 1.0.
    @test count(iszero, Z) == count(iszero, GEXP) == 8
end

@testset "Scale resolves against the natural value, and a cross edge is zero" begin
    @test resolve_feature_value(0.7, 0.5) == 0.7          # a bare number is absolute
    @test resolve_feature_value(Scale(1.3), 0.5) == 0.65  # a marker scales
    @test Scale(1.3).val == 1.3
    @test Scale().val == 1.0
    @test isa(Scale(1.0), PortfolioOptimisers.AbstractFeatureValue)
    @test_throws DomainError Scale(NaN)
    @test_throws DomainError Scale(Inf)

    nz = GSETS.dict["nz"]
    z(Z, a, n) = Z[findfirst(==(a), GSETS.dict["nx"]), findfirst(==(n), nz)]

    # Numeric key: the natural value is the asset's own datum.
    Zn = asset_sets_features(["nx_esg" => Scale(2.0)], GSETS)
    @test [z(Zn, a, "esg") for a in ("A", "B", "C")] == 2 .* [0.30, 0.80, 0.50]

    # One-hot key: the natural value is 1.0 for a member, so a Scale reads as a plain set --
    # which is exactly why nesting depth could not have carried the distinction.
    Zh = asset_sets_features(["nx_sector" => Scale(2.0)], GSETS)
    @test Zh == asset_sets_features(["nx_sector" => 2.0], GSETS)

    # A *cross* edge scales the row's membership of the node the target named, and `C` is
    # UK, so scaling its US membership gives zero. A documented hazard, not a defect: use a
    # bare number to set a cross edge.
    Zx = asset_sets_features(["C" => ["nx_country" => "US" => Scale(0.3)]], GSETS)
    @test z(Zx, "C", "US") == 0.0
    @test z(asset_sets_features(["C" => ["nx_country" => "US" => 0.3]], GSETS), "C",
            "US") == 0.3
    # Same for an asset-node target: the natural value is the 0.0 of "is B in {A}?".
    @test z(asset_sets_features(["B" => "A" => Scale(0.9)], GSETS), "B", "A") == 0.0
    # And 1.0 when the row *is* the node it names.
    @test z(asset_sets_features(["B" => "B" => Scale(0.9)], GSETS), "B", "B") == 0.9
end

@testset "Last wins, and a repeated key is the point" begin
    # `allunique` does not carry into graded mode: repeating a key is how a program refines
    # a broad stroke, and refusing it would refuse the grammar's whole ordering rule.
    Z = asset_sets_features(["nx_sector" => 1.0, "nx_sector" => 2.0, "nx_sector" => 3.0],
                            GSETS)
    @test Z == asset_sets_features(["nx_sector" => 3.0], GSETS)
    # Order is load-bearing, so reversing the program is a different matrix.
    p = ["nx_sector" => 1.0, "AC" => ["nx_sector" => "Finance" => 0.5]]
    @test asset_sets_features(p, GSETS) != asset_sets_features(reverse(p), GSETS)
    # ... unlike the key path, where a permuted `vals` is only a column permutation and a
    # repeated key is refused outright, because there it silently doubles a block.
    @test_throws ArgumentError AssetSetsFeatures(; vals = ["nx_sector", "nx_sector"])
    @test AssetSetsFeatures(; vals = ["nx_sector" => 1.0, "nx_sector" => 2.0]) isa
          AssetSetsFeatures
end

@testset "The graded path subsumes the key path, with the two name conventions" begin
    # Both contracts on one `UniverseSets`: the same universe, the same taxonomy, and the
    # declared axis added alongside. This is what "subsumes" means operationally -- an
    # all-`1.0` diagonal program is bit-identical to stacking the same keys.
    both = UniverseSets(; xkey = "nx", zkey = "nz",
                        dict = Dict{String, Any}("nx" => rd.nx,
                                                 "nx_sector" => TAX.dict["nx_sector"],
                                                 "nx_industry" => TAX.dict["nx_industry"],
                                                 "nz" => [unique(TAX.dict["nx_sector"]);
                                                          unique(TAX.dict["nx_industry"])]))
    keys_path = ["nx_sector", "nx_industry"]
    prog = ["nx_sector" => 1.0, "nx_industry" => 1.0]
    @test asset_sets_features(prog, both) == asset_sets_features(keys_path, both)

    # The names, however, differ by design. The key path *derives* its axis by stacking, so
    # it must qualify or a nested taxonomy collides with itself. The graded path's axis is
    # *authored*, so the names are whatever the caller wrote -- bare.
    @test asset_sets_feature_names(prog, both) == both.dict["nz"]
    @test all(occursin("=", n) for n in asset_sets_feature_names(keys_path, both))
    @test !any(occursin("=", n) for n in asset_sets_feature_names(prog, both))
    @test length(asset_sets_feature_names(prog, both)) ==
          size(asset_sets_features(prog, both), 2)

    # And the cost of bareness, documented as a graded-mode limitation: a nested taxonomy
    # with a repeated value is inexpressible, because both levels land on the one node. The
    # key path stays the tool for that case.
    @test !allunique([unique(TAX.dict["nx_industry"]);
                      unique(TAX.dict["nx_subindustry"])])
    @test allunique(asset_sets_feature_names(KEYS, TAX))
end

@testset "strict governs names only; nothing structural is refused" begin
    prog = ["nx_sector" => 2.0]

    # An unknown *node* is reported against the feature axis, with the key that declares it
    # -- a caller who put their nodes under `nx` would otherwise be told about assets.
    bad_node = ["A" => ["nx_sector" => "Financ" => 1.0]]
    @test_logs((:warn, r"not in feature universe"), match_mode = :any,
               asset_sets_features(bad_node, GSETS))
    res = @test_throws ArgumentError asset_sets_features(bad_node, GSETS; strict = true)
    @test occursin("feature universe", res.value.msg)
    @test occursin("did you mean `Finance`?", res.value.msg)

    # An unknown *asset or group* in row-selector position, with the pool widened past the
    # raw universe so a mistyped group can still be suggested.
    bad_row = ["Tech_grou" => ["nx_sector" => "Tech" => 1.0]]
    @test_logs((:warn, r"not in asset universe"), match_mode = :any,
               asset_sets_features(bad_row, GSETS))
    res = @test_throws ArgumentError asset_sets_features(bad_row, GSETS; strict = true)
    @test occursin("did you mean `Tech_group`?", res.value.msg)

    # A group value that matches no asset, reported against its own key's values.
    res = @test_throws ArgumentError asset_sets_features(["nx_country" => "U" => 1.0],
                                                         GSETS; strict = true)
    @test occursin("matches no asset", res.value.msg)

    # Nothing structural is refused. An all-zero row is legal -- and is the one genuine
    # silent-wrongness case grading opens, since the zero-norm convention makes zero rows
    # mutually *identical*, so forgotten assets cluster together at distance 0.
    Z0 = asset_sets_features(["A" => "A" => 1.0], GSETS)
    @test all(iszero, Z0[2, :])
    @test all(iszero, Z0[3, :])
    D0 = distance(FeatureDistance(), Z0)
    @test D0[2, 3] == 0

    # A one-column matrix is legal: `assert_feature_keys`' two-key floor is a property of
    # stacking partitions and does not carry here.
    onecol = UniverseSets(;
                          dict = Dict{String, Any}("nx" => ["A", "B"], "nz" => ["only"],
                                                   "nx_g" => ["only", "other"]))
    Z1 = @test_logs (:warn,) match_mode = :any asset_sets_features(["nx_g" => 1.0], onecol)
    @test size(Z1) == (2, 1)

    # Only non-emptiness is unconditional, at the kernel and at construction alike.
    @test_throws PortfolioOptimisers.IsEmptyError asset_sets_features(Pair[], GSETS)
    @test_throws PortfolioOptimisers.IsEmptyError AssetSetsFeatures(; vals = Pair[])

    # A malformed *term* throws regardless of `strict`: there is no reading of it to fall
    # back to, so it is a syntax error and not a name failure.
    @test_throws ArgumentError asset_sets_features(["A" => 0.5], GSETS)
    @test_throws ArgumentError asset_sets_features(["A" => ["nx_sector" => 0.5]], GSETS)
end

@testset "A factor-axis key is refused by name, in every position" begin
    # #224 made `nf`-prefixed keys reachable in `sets.dict`, and they are factor-length, so
    # they can index neither the rows (assets) nor the declared nodes. Letting one fall
    # through to the plain-group branch would fail later on a length mismatch naming neither
    # the axis nor the cause.
    fs = UniverseSets(; xkey = "nx", zkey = "nz",
                      dict = Dict{String, Any}("nx" => ["A", "B", "C"],
                                               "nf" => ["F1", "F2"],
                                               "nz" => ["Tech", "Finance"],
                                               "nx_sector" => ["Tech", "Tech", "Finance"],
                                               "nf_style" => ["Value", "Growth"]))
    for prog in (["nf_style" => ["nx_sector" => "Tech" => 1.0]],   # row selector
                 ["A" => ["nf_style" => "Value" => 1.0]],          # target, two-level
                 ["A" => ["nf_style" => 1.0]])                     # target, bare
        @test_logs((:warn, r"names the factor axis"), match_mode = :any,
                   asset_sets_features(prog, fs))
        res = @test_throws ArgumentError asset_sets_features(prog, fs; strict = true)
        @test occursin("names the factor axis", res.value.msg)
        # No suggestion: the name resolved perfectly well, on the wrong axis.
        @test !occursin("did you mean", res.value.msg)
    end
end

@testset "The declared axis is fold-invariant, and its zkey validation" begin
    # `port_opt_view` gains no branch for `zkey`, but for a *different* reason than the
    # factor axis: some of these nodes *are* assets. It passes through because the axis is
    # declared rather than derived -- the caller's coordinate system, not a summary of the
    # current universe.
    i = [1, 3]
    v = PortfolioOptimisers.port_opt_view(GSETS, i)
    @test v.zkey == GSETS.zkey
    @test v.dict["nz"] == GSETS.dict["nz"]
    @test v.dict["nx"] == GSETS.dict["nx"][i]
    @test v.dict["nx_sector"] == GSETS.dict["nx_sector"][i]

    # So `size(Z, 2)` is fold-invariant -- the exact opposite of the key path, where the
    # viewed producer rebuilds the axis and a group with no members left disappears.
    sub = ["nx_sector" => 2.0, "nx_country" => 1.0, "nx_esg" => Scale(1.3)]
    Zv = asset_sets_features(sub, v)
    Zf = asset_sets_features(sub, GSETS)
    @test size(Zv, 2) == size(Zf, 2) == length(GSETS.dict["nz"])
    @test size(Zv, 1) == length(i)
    @test Zv == Zf[i, :]
    # And the accepted cost: a node for an asset the view dropped survives as an all-zero
    # column. Benign for every blessed metric, since a zero column contributes nothing to a
    # dot product or a row norm.
    @test all(iszero, Zv[:, findfirst(==("B"), GSETS.dict["nz"])])

    # `zkey` is the fifth *name* field and joins the mutual-prefix loop, which now runs 20
    # ordered checks rather than 12.
    @test GSETS.zkey == "nz"
    @test UniverseSets(; dict = Dict("nx" => ["A"])).zkey == "nz"
    @test_throws ArgumentError UniverseSets(; zkey = "n", dict = Dict("nx" => ["A"]))
    @test_throws ArgumentError UniverseSets(; zkey = "nx", dict = Dict("nx" => ["A"]))
    # It carries no length rule and no unique-entry sibling -- nothing is written *over* the
    # feature axis -- but it does carry `allunique`, so `ReturnsResult`'s own uniqueness
    # check cannot be reached with a duplicate.
    @test_throws ArgumentError UniverseSets(;
                                            dict = Dict("nx" => ["A"], "nz" => ["a", "a"]))
    @test UniverseSets(; dict = Dict("nx" => ["A", "B"], "nz" => ["a", "b", "c", "d"])) isa
          UniverseSets
    # The entry is optional; a consumer that needs it diagnoses the absence at the point of
    # need, as `factor_universe` does for factors.
    nozk = UniverseSets(; dict = Dict("nx" => ["A", "B"], "nx_g" => ["a", "b"]))
    res = @test_throws KeyError asset_sets_features(["nx_g" => 1.0], nozk)
    @test occursin("the declared feature axis", string(res.value.key))
    @test occursin("sets.zkey", string(res.value.key))
end

@testset "AssetSetsFeatures carries both contracts, and strict is a field" begin
    ze = AssetSetsFeatures(; vals = GPROG)
    @test ze.vals === GPROG
    @test ze.strict == false
    @test AssetSetsFeatures(; vals = GPROG, strict = true).strict == true
    # The key path keeps its construction-time floor; the graded path does not inherit it.
    @test_throws ArgumentError AssetSetsFeatures(; vals = ["nx_sector"])
    @test AssetSetsFeatures(; vals = ["nx_sector" => 1.0]) isa AssetSetsFeatures

    # `strict` has to be a field: the producer interface is `feature_matrix(ze, pr, X, F,
    # sets)`, so there is nowhere to pass a keyword through.
    pe = FeaturePrior(; ze = ze, sets = GSETS)
    pr = prior(pe, randn(StableRNG(987654321), 64, 3))
    @test pr.Z == GEXP
    @test pr.mu == prior(EmpiricalPrior(), randn(StableRNG(987654321), 64, 3)).mu

    bad = FeaturePrior(;
                       ze = AssetSetsFeatures(; vals = ["Zz" => "A" => 1.0], strict = true),
                       sets = GSETS)
    @test_throws ArgumentError prior(bad, randn(StableRNG(987654321), 64, 3))
    lax = FeaturePrior(; ze = AssetSetsFeatures(; vals = ["Zz" => "A" => 1.0]),
                       sets = GSETS)
    @test_logs (:warn,) match_mode = :any prior(lax, randn(StableRNG(987654321), 64, 3))
end

@testset "A graded program drives a FeatureDistance end to end, from both carriers" begin
    # The real twenty-asset universe, with the declared axis alongside the taxonomy the key
    # path reads. An all-`1.0` program is the key path exactly, so the two routes are
    # comparable cell for cell -- and then the grading is added on top.
    nzz = [unique(TAX.dict["nx_sector"]); unique(TAX.dict["nx_industry"])]
    G = UniverseSets(; xkey = "nx", zkey = "nz",
                     dict = Dict{String, Any}("nx" => rd.nx,
                                              "nx_sector" => TAX.dict["nx_sector"],
                                              "nx_industry" => TAX.dict["nx_industry"],
                                              "nz" => nzz))
    # Sector counts double: the reweighting instance the transform question no longer has to
    # answer for this producer, because it is authored rather than applied.
    prog = ["nx_sector" => 2.0, "nx_industry" => 1.0]
    Z = asset_sets_features(prog, G)
    @test Z != asset_sets_features(["nx_sector" => 1.0, "nx_industry" => 1.0], G)

    fde = FeatureDistance()
    # Carrier one: the user's own data, under the default `z_src = :data`.
    rdz = ReturnsResult(; nx = rd.nx, X = rd.X, nf = rd.nf, F = rd.F, ts = rd.ts,
                        nz = asset_sets_feature_names(prog, G), Z = Z)
    @test rdz.nz == nzz
    # Carrier two: derived, via the producer.
    pe = FeaturePrior(; ze = AssetSetsFeatures(; vals = prog), sets = G)
    pr = prior(pe, rd)
    @test pr.Z == Z

    cle_f = ClustersEstimator(; de = fde)
    clr_d = clusterise(cle_f, rdz)
    clr_p = clusterise(cle_f, pr; z_src = :prior)
    @test clr_d.D == distance(fde, Z)
    @test clr_p.D == clr_d.D

    # The grading changes the answer, which is the whole point of authoring it: doubling the
    # sector weight is a different distance from the one-hot stack.
    clr_1 = clusterise(cle_f,
                       ReturnsResult(; nx = rd.nx, X = rd.X, ts = rd.ts, nz = nzz,
                                     Z = asset_sets_features(["nx_sector" => 1.0,
                                                              "nx_industry" => 1.0], G)))
    @test clr_d.D != clr_1.D
    # And it still differs from the returns correlation, so exogenous structure survives.
    clr_c = clusterise(ClustersEstimator(; de = Distance(; alg = CanonicalDistance())), rd)
    @test clr_d.D != clr_c.D

    hopt_d = HierarchicalOptimiser(; cle = cle_f, slv = slv)
    hopt_p = HierarchicalOptimiser(; pe = pe, cle = cle_f, slv = slv, z_src = :prior)
    for oe in (HierarchicalRiskParity, HierarchicalEqualRiskContribution)
        wd = optimise(oe(; opt = hopt_d), rdz)
        wp = optimise(oe(; opt = hopt_p), rd)
        for res in (wd, wp)
            @test isapprox(sum(res.w), 1)
            @test all(isfinite, res.w)
        end
        # Same matrix through both carriers, so the same weights.
        @test isapprox(wd.w, wp.w)
    end
end
