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
const TAX = AssetSets(; key = "nx",
                      dict = Dict("nx" => rd.nx,
                                  "nx_sector" => ["Technology", "Technology", "Financials",
                                                  "ConsumerDiscretionary", "Energy", "Industrials",
                                                  "ConsumerDiscretionary", "HealthCare", "Financials",
                                                  "ConsumerStaples", "HealthCare", "HealthCare",
                                                  "Technology", "ConsumerStaples", "HealthCare",
                                                  "ConsumerStaples", "Energy", "HealthCare",
                                                  "ConsumerStaples", "Energy"],
                                  "nx_industry" => ["Hardware", "Semiconductors", "Banks",
                                                    "SpecialtyRetail", "IntegratedOil", "Aerospace",
                                                    "SpecialtyRetail", "Pharma", "Banks", "Beverages",
                                                    "Pharma", "Pharma", "Software", "Beverages",
                                                    "Pharma", "HouseholdProducts",
                                                    "ExplorationProduction", "ManagedCare", "FoodRetail",
                                                    "IntegratedOil"],
                                  "nx_subindustry" =>
                                      ["ConsumerElectronics", "Semiconductors",
                                       "DiversifiedBanks", "ComputerRetail",
                                       "IntegratedOil", "AerospaceDefense",
                                       "HomeImprovementRetail", "Pharma",
                                       "DiversifiedBanks", "SoftDrinks", "Pharma", "Pharma",
                                       "SystemsSoftware", "SoftDrinks", "Pharma",
                                       "HouseholdProducts", "OilGasEP", "ManagedHealthCare",
                                       "Hypermarkets", "IntegratedOil"]))
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
    short = AssetSets(; key = "nx",
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
    sq = AssetSets(; key = "nx",
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

    # A key not prefixed by `sets.key` is not sliced by the view, which does *not* fail
    # silently: `asset_sets_matrix`'s length check throws on the next call because the
    # sliced universe no longer matches the unsliced group. Name taxonomy keys with the
    # prefix if they are to survive a fold or an NCO cluster.
    bare = AssetSets(; key = "nx",
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
