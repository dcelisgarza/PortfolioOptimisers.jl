using PortfolioOptimisers, Test, StableRNGs, StatsBase, LinearAlgebra, Clarabel
using PortfolioOptimisers.Distances: Euclidean, Cityblock, SqEuclidean, Minkowski,
                                     ChiSqDist, WeightedEuclidean, WeightedCityblock,
                                     CosineDist, Chebyshev, WeightedSqEuclidean,
                                     WeightedMinkowski, TotalVariation

const PO = PortfolioOptimisers

#=
A `FeatureDistance` reads each asset, or each pair of assets, only at its own active rows
(#1450, built by #1454). A window of an Asset Panel holds a finite value in every inactive
cell (ADR 0102), and that value is arbitrary: a fill value, a zero, or the last value before a
delisting. So a collapse that forgot the active mask gave a plausible distance from it.

The hand panel below has four rows and three assets `a`, `b`, `c`, with one feature `f`:

    row   a   b   c
     1    1   2   9*
     2    2   4   9*
     3    3   9*  5
     4    4   9*  7

A starred cell is inactive and holds `9`. So `a` is active on every row, `b` on rows 1 and 2,
and `c` on rows 3 and 4. The pair `(b, c)` shares no active row. Every expected value below is
worked by hand from this table, and a wrong read of a starred cell moves it.
=#
function hand_rd(; V = [1.0 2 9; 2 4 9; 3 9 5; 4 9 7], A = Bool[1 1 0; 1 1 0; 1 0 1; 1 0 1],
                 nx = ["a", "b", "c"])
    T, N = size(V)
    pnl = AssetPanel(; pf = [NumericPanelField(; name = "f", vals = V)], amsk = A, emsk = A)
    X = randn(StableRNG(1454), T, N)
    return ReturnsResult(; nx = nx, X = X, pnl = pnl)
end
function fdist(alg; metric = Euclidean(), kwargs...)
    return FeatureDistance(; metric = metric, alg = alg, sel = ["f"], kwargs...)
end
hand_distance(de, rd) = distance(de, nothing, rd.X; rd = rd)

@testset "A FeatureDistance reads each asset and each pair at its own active rows (#1454)" begin
    rd = hand_rd()
    @testset "LastObservation" begin
        @test LastObservation().alg === LastRow()
        # LastRow reads the last row, where `b` is inactive: a direct call refuses it by name.
        msg = "FeatureDistance has no value to read for the assets [\"b\"]: each is inactive at the last row of the window, or a value column of `sel` was not observed there."
        @test_throws msg hand_distance(fdist(LastObservation()), rd)
        # On a view without `b`, the last row is 4 against 7.
        D = hand_distance(fdist(LastObservation()), PO.port_opt_view(rd, [1, 3]))
        @test D == [0.0 3.0; 3.0 0.0]
        # LastActiveRow reads `a` at row 4 (4), `b` at row 2 (4), and `c` at row 4 (7).
        D = hand_distance(fdist(LastObservation(; alg = LastActiveRow())), rd)
        @test D == [0.0 0.0 3.0; 0.0 0.0 3.0; 3.0 3.0 0.0]
        # The stack holds the rows from the earliest last active row, row 2, to the last.
        @test PO.collapse_rows(LastObservation(; alg = LastActiveRow()), rd.pnl) == 2:4
        @test PO.collapse_rows(LastObservation(), rd.pnl) == 4:4
    end
    @testset "AggregateFeatures" begin
        # The mean of `a` is 2.5, of `b` 3, and of `c` 6.
        D = hand_distance(fdist(AggregateFeatures()), rd)
        @test D ≈ [0 0.5 3.5; 0.5 0 3; 3.5 3 0] rtol = 1e-15
        # The median of each asset over its active rows is the same here.
        D = hand_distance(fdist(AggregateFeatures(; alg = MedianCollapse())), rd)
        @test D ≈ [0 0.5 3.5; 0.5 0 3; 3.5 3 0] rtol = 1e-15
        # Weights 1:4 are restricted to each asset's rows and renormalised:
        # a = 30 / 10, b = 10 / 3, c = 43 / 7.
        D = hand_distance(fdist(AggregateFeatures(; w = StatsBase.Weights(1.0:4.0))), rd)
        z = [3, 10 / 3, 43 / 7]
        @test D ≈ abs.(z .- z') rtol = 1e-14
        # An asset whose active rows carry a zero weight has no value to read.
        msg = "the observation weights of those rows sum to zero for the assets [\"c\"]"
        de = fdist(AggregateFeatures(; w = StatsBase.Weights([1.0, 1, 0, 0])))
        @test_throws msg hand_distance(de, rd)
    end
    @testset "AggregateDistances" begin
        # (a, b) shares rows 1 and 2: |1 - 2| and |2 - 4| average to 1.5. (a, c) shares rows 3
        # and 4: |3 - 5| and |4 - 7| average to 2.5. (b, c) shares no row.
        msg = "FeatureDistance reads each pair of assets at the rows at which both are active, and the assets (\"b\", \"c\") share no active row."
        @test_throws msg hand_distance(fdist(AggregateDistances()), rd)
        msg = "DropFewerRows drops an asset at the entry of a fit"
        de = fdist(AggregateDistances(; pair = DropFewerRows()))
        @test_throws msg hand_distance(de, rd)
        # The fallback measures (b, c) by their means over their own rows, 3 against 6.
        D = hand_distance(fdist(AggregateDistances(; pair = FeatureFallback())), rd)
        @test D ≈ [0 1.5 2.5; 1.5 0 3; 2.5 3 0] rtol = 1e-15
        # Weights 1:4: (a, b) = (1 * 1 + 2 * 2) / 3, (a, c) = (3 * 2 + 4 * 3) / 7.
        D = hand_distance(fdist(AggregateDistances(; w = StatsBase.Weights(1.0:4.0),
                                                   pair = FeatureFallback())), rd)
        @test D[1, 2] ≈ 5 / 3 rtol = 1e-15
        @test D[1, 3] ≈ 18 / 7 rtol = 1e-15
        @test D[2, 3] ≈ 3 rtol = 1e-15
    end
    @testset "StackObservations rescales each metric by its method" begin
        # (a, b) stacks rows 1 and 2, differences (1, 2); (a, c) rows 3 and 4, (2, 3). The
        # window has four rows, so each pair shares half of it.
        fb = FeatureFallback()
        D = hand_distance(fdist(StackObservations(; pair = fb)), rd)
        @test D[1, 2] ≈ sqrt(5) * sqrt(2) rtol = 1e-15
        @test D[1, 3] ≈ sqrt(13) * sqrt(2) rtol = 1e-15
        # The fallback fills each row with the means 3 and 6: four rows of 3.
        @test D[2, 3] ≈ 6 rtol = 1e-15
        D = hand_distance(fdist(StackObservations(; pair = fb); metric = Cityblock()), rd)
        @test D ≈ [0 6 10; 6 0 12; 10 12 0] rtol = 1e-15
        D = hand_distance(fdist(StackObservations(; pair = fb); metric = SqEuclidean()), rd)
        @test D ≈ [0 10 26; 10 0 36; 26 36 0] rtol = 1e-15
        D = hand_distance(fdist(StackObservations(; pair = fb); metric = Minkowski(3)), rd)
        @test D[1, 2] ≈ cbrt(9) * cbrt(2) rtol = 1e-15
        D = hand_distance(fdist(StackObservations(; pair = fb); metric = ChiSqDist()), rd)
        @test D[1, 2] ≈ 2 * (1 / 3 + 4 / 6) rtol = 1e-15
        # A weighted form takes the ratio of its weight sums.
        w = [1.0, 2, 3, 4]
        D = hand_distance(fdist(StackObservations(; pair = fb);
                                metric = WeightedEuclidean(w)), rd)
        @test D[1, 2] ≈ sqrt(1 + 2 * 4) * sqrt(10 / 3) rtol = 1e-15
        D = hand_distance(fdist(StackObservations(; pair = fb);
                                metric = WeightedCityblock(w)), rd)
        @test D[1, 3] ≈ (3 * 2 + 4 * 3) * 10 / 7 rtol = 1e-15
        D = hand_distance(fdist(StackObservations(; pair = fb);
                                metric = WeightedSqEuclidean(w)), rd)
        @test D[1, 2] ≈ (1 + 2 * 4) * 10 / 3 rtol = 1e-15
        D = hand_distance(fdist(StackObservations(; pair = fb);
                                metric = WeightedMinkowski(w, 3)), rd)
        @test D[1, 2] ≈ cbrt(1 + 2 * 8) * cbrt(10 / 3) rtol = 1e-15
        D = hand_distance(fdist(StackObservations(; pair = fb); metric = TotalVariation()),
                          rd)
        @test D[1, 2] ≈ (1 + 2) / 2 * 2 rtol = 1e-15
        # A ratio metric takes no rescale.
        D = hand_distance(fdist(StackObservations(; pair = fb); metric = AngularDist()), rd)
        @test D[1, 3] == AngularDist()([3.0, 4], [5.0, 7])
        D = hand_distance(fdist(StackObservations(; pair = fb); metric = CosineDist()), rd)
        @test D[1, 2] ≈ CosineDist()([1.0, 2], [2.0, 4]) atol = 1e-15
        # A metric with no rescale method refuses a partial pair, and names the method to add.
        msg = "Add a method `PortfolioOptimisers.stack_rescale(metric::Chebyshev, d, idx, len)`"
        de = fdist(StackObservations(; pair = fb); metric = Chebyshev())
        @test_throws msg hand_distance(de, rd)
        # The empty pair refuses by default.
        @test_throws "share no active row" hand_distance(fdist(StackObservations()), rd)
    end
    @testset "A window with no inactive cell reads every row, as before" begin
        # With every cell active the mask changes nothing, bit for bit.
        rdf = hand_rd(; A = trues(4, 3))
        Z = feature_matrix(fdist(StackObservations()), nothing, rdf, rdf.X)
        for alg in (LastObservation(), LastObservation(; alg = LastActiveRow()),
                    AggregateFeatures(), AggregateDistances(), StackObservations())
            de = fdist(alg)
            @test hand_distance(de, rdf) ==
                  distance(de, feature_matrix(de, nothing, rdf, rdf.X))
        end
        @test hand_distance(fdist(StackObservations()), rdf) ==
              distance(fdist(StackObservations()), Z)
    end
    @testset "The raw 3-D entry takes the active mask as a keyword" begin
        Z = reshape([1.0 2 9; 2 4 9; 3 9 5; 4 9 7], 4, 3, 1)
        A = Bool[1 1 0; 1 1 0; 1 0 1; 1 0 1]
        de = FeatureDistance(; metric = Euclidean(),
                             alg = AggregateDistances(; pair = FeatureFallback()))
        @test distance(de, Z; amsk = A) ==
              hand_distance(fdist(AggregateDistances(; pair = FeatureFallback())), rd)
        # The same window with the assets on the last axis.
        @test distance(de, permutedims(Z, (1, 3, 2)); dims = 2, amsk = A) ==
              distance(de, Z; amsk = A)
        # With no names, a refusal quotes positions.
        msg = "the assets at the positions (2, 3)"
        @test_throws msg distance(fdist(AggregateDistances()), Z; amsk = A)
        msg = "no value to read for the assets at the positions [2]: each is inactive at the last row"
        @test_throws msg distance(fdist(LastObservation()), Z; amsk = A)
        # An asset with no active row has no value under any collapse.
        B = copy(A)
        B[:, 2] .= false
        msg = "no value to read for the assets at the positions [2]: each has no readable row in the window: no row at which it is active and every value column of `sel` was observed. Inside a fit"
        @test_throws msg distance(fdist(AggregateFeatures()), Z; amsk = B)
        # A mask whose last row is complete reads the last row under either rule.
        C = copy(A)
        C[4, :] .= true
        for alg in (LastObservation(), LastObservation(; alg = LastActiveRow()))
            @test distance(fdist(alg), Z; amsk = C) == [0 5 3; 5 0 2; 3 2 0]
        end
        # A pair whose shared rows hold a zero vector takes the zero-feature-vector convention.
        Zz = copy(Z)
        Zz[1:2, 2, 1] .= 0
        de = fdist(StackObservations(; pair = FeatureFallback());
                   metric = PortfolioOptimisers.Distances.CosineDist())
        @test distance(de, Zz; amsk = A)[1, 2] == 1
        @test_throws DimensionMismatch distance(de, Z; amsk = trues(3, 3))
    end
    @testset "The entry of a fit composes the readable assets with the Investable Mask" begin
        # LastRow cannot read `b`; the other collapses read every asset.
        @test PO.feature_readable_mask(fdist(LastObservation()), nothing, rd) ==
              BitVector([1, 0, 1])
        @test isnothing(PO.feature_readable_mask(fdist(AggregateFeatures()), nothing, rd))
        # The Investable Mask of the prior is kept.
        @test PO.feature_readable_mask(fdist(LastObservation()), BitVector([0, 1, 1]),
                                       rd) == BitVector([0, 0, 1])
        # DropFewerRows drops the asset of (b, c) with fewer rows; on a tie, the later one.
        de = fdist(AggregateDistances(; pair = DropFewerRows()))
        @test PO.feature_readable_mask(de, nothing, rd) == BitVector([1, 1, 0])
        A = Bool[1 1 0; 1 1 0; 1 0 1; 1 0 0]
        @test PO.feature_readable_mask(de, nothing, hand_rd(; A = A)) ==
              BitVector([1, 1, 0])
        A = Bool[1 1 0; 1 0 0; 1 0 1; 1 0 1]
        @test PO.feature_readable_mask(de, nothing, hand_rd(; A = A)) ==
              BitVector([1, 0, 1])
        # Every holder forwards, and a producer or a precomputed estimator reads no panel.
        cle = ClustersEstimator(; de = fdist(LastObservation()))
        @test PO.feature_readable_mask(cle, nothing, rd) == BitVector([1, 0, 1])
        @test PO.feature_readable_mask((nothing, [cle]), nothing, rd) ==
              BitVector([1, 0, 1])
        ne = NetworkEstimator(; de = fdist(LastObservation()))
        @test PO.feature_readable_mask(CentralityEstimator(; pl = ne), nothing, rd) ==
              BitVector([1, 0, 1])
        @test PO.feature_readable_mask(SemiDefinitePhylogenyEstimator(; pl = cle), nothing,
                                       rd) == BitVector([1, 0, 1])
        @test isnothing(PO.feature_readable_mask(ClustersEstimator(), nothing, rd))
        # Returns data with no panel, or with a static one, has no active rows to compose.
        imsk = BitVector([1, 0, 1])
        @test PO.feature_readable_mask(fdist(LastObservation()), imsk,
                                       ReturnsResult(; nx = rd.nx, X = rd.X)) === imsk
        spnl = AssetPanel(; pf = [NumericPanelField(; name = "f", vals = [1.0, 2, 3])])
        @test PO.feature_readable_mask(fdist(LastObservation()), imsk,
                                       ReturnsResult(; nx = rd.nx, X = rd.X, pnl = spnl)) ===
              imsk
        # A static panel has no observation axis, so LastActiveRow names every row.
        @test PO.collapse_rows(LastObservation(; alg = LastActiveRow()), spnl) === Colon()
        # A refusal quotes the names of the returns data, or of a ReturnsResult passed as
        # `pr`, and positions otherwise.
        @test PO.feature_asset_names(nothing, rd) == rd.nx
        @test PO.feature_asset_names(rd, nothing) == rd.nx
        @test isnothing(PO.feature_asset_names(nothing, nothing))
        @test isnothing(PO.feature_readable_mask(FeatureDistance(; ape = PhylogenyPanel()),
                                                 nothing, rd))
        # No asset left refuses.
        @test_throws IsEmptyError PO.feature_readable_mask(fdist(LastObservation()),
                                                           BitVector([0, 1, 0]), rd)
    end
    @testset "Indexed weights keep their kind, a range included" begin
        # StatsBase rebuilds indexed weights with the type of the stored values, which fails on
        # a range. The restriction to the active rows, and a fold of a static weight, rebuild
        # weights of the same kind instead.
        w = PO.nothing_scalar_array_getindex(StatsBase.Weights(1.0:4.0), [1, 3])
        @test w isa StatsBase.Weights && w == [1.0, 3.0]
        @test PO.nothing_scalar_array_getindex(StatsBase.fweights([1, 2, 3]), [1, 3]) isa
              StatsBase.FrequencyWeights
        ag = PO.obs_weights_view(AggregateFeatures(; w = StatsBase.pweights(1.0:10.0)),
                                 [1, 3])
        @test ag.w isa StatsBase.ProbabilityWeights && ag.w == [1.0, 3.0]
    end
end

#=
The entry of a fit. Asset `d` of `drop_rd` delists at row 21, and a Coverage Policy that
expires after 100 rows keeps it in the investable universe. The default collapse reads the last
row, where `d` is inactive, so every entry that clusters drops `d` as a non-investable asset.
=#
function drop_rd()
    rng = StableRNG(14542)
    T, N = 40, 5
    X = 0.01 .* randn(rng, T, N)
    A = trues(T, N)
    A[21:end, 4] .= false
    X[21:end, 4] .= NaN
    pnl = AssetPanel(; pf = [NumericPanelField(; name = "f", vals = randn(rng, T, N))],
                     amsk = A, emsk = A)
    return ReturnsResult(; nx = ["a", "b", "c", "d", "e"], X = X, pnl = pnl)
end
# Asset `d` is active on rows 1 to 20 and asset `e` on rows 21 to 40, so the pair (d, e) shares
# no active row. A covariance of the pair has no shared row either, so the fits take a prior
# fitted on the complete returns.
function pair_rd()
    rng = StableRNG(14541)
    T, N = 40, 5
    X = 0.01 .* randn(rng, T, N)
    A = trues(T, N)
    A[21:end, 4] .= false
    A[1:20, 5] .= false
    pnl = AssetPanel(; pf = [NumericPanelField(; name = "f", vals = randn(rng, T, N))],
                     amsk = A, emsk = A)
    nx = ["a", "b", "c", "d", "e"]
    return ReturnsResult(; nx = nx, X = X, pnl = pnl),
           prior(EmpiricalPrior(), ReturnsResult(; nx = nx, X = X))
end

@testset "The entry of a fit drops an asset that a FeatureDistance cannot read (#1454)" begin
    cvg = CoveragePolicy(; alg = ExpireCoverage(; after = 100))
    pe = EmpiricalPrior(; me = SimpleExpectedReturns(; cvg = cvg),
                        ce = PortfolioOptimisersCovariance(; ce = Covariance(; cvg = cvg)))
    rd = drop_rd()
    @test isnothing(PO.investable_mask(prior(pe, rd)))
    cle = ClustersEstimator(; de = FeatureDistance(; sel = ["f"]))
    ho = HierarchicalOptimiser(; pe = pe, cle = cle)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false))
    jo = JuMPOptimiser(; pe = pe, slv = slv,
                       ple = SemiDefinitePhylogenyEstimator(; pl = cle))
    msg = (:info, r"1 asset\(s\) left the investable universe .*\[\"d\"\]")
    for (name, opt) in ("HierarchicalRiskParity" => HierarchicalRiskParity(; opt = ho),
                        "HierarchicalEqualRiskContribution" =>
                            HierarchicalEqualRiskContribution(; opt = ho),
                        "SchurComplementHierarchicalRiskParity" =>
                            SchurComplementHierarchicalRiskParity(; opt = ho),
                        "NestedClustered" =>
                            NestedClustered(; pe = pe, cle = cle, opti = InverseVolatility(; pe = pe),
                                            opto = InverseVolatility(; pe = pe)),
                        "MeanRisk" => MeanRisk(; opt = jo),
                        "SubsetResampling" =>
                            SubsetResampling(; pe = pe, opt = HierarchicalRiskParity(; opt = ho),
                                             subset_size = 4, n_subsets = 3))
        @testset "$(name)" begin
            res = @test_logs msg match_mode = :any optimise(opt, rd)
            @test iszero(res.w[4])
            @test sum(res.w) ≈ 1 rtol = 1e-6
        end
    end
    # Stacking clusters inside its members alone: the clustering member drops `d`, and the
    # member that reads no feature keeps it.
    st = Stacking(; pe = pe,
                  opti = [HierarchicalRiskParity(; opt = ho), InverseVolatility(; pe = pe)],
                  opto = InverseVolatility(; pe = pe))
    res = optimise(st, rd)
    @test iszero(res.resi[1].w[4])
    @test res.resi[2].w[4] > 0
    # The masked result names `d` as non-investable.
    res = optimise(HierarchicalRiskParity(; opt = ho), rd)
    @test PO.result_investable_mask(res) == BitVector([1, 1, 1, 0, 1])
    # LastActiveRow reads `d` at row 20, so nothing drops.
    cla = ClustersEstimator(;
                            de = FeatureDistance(; sel = ["f"],
                                                 alg = LastObservation(;
                                                                       alg = LastActiveRow())))
    res = optimise(HierarchicalRiskParity(;
                                          opt = HierarchicalOptimiser(; pe = pe, cle = cla)),
                   rd)
    @test res.w[4] > 0
    @testset "The rule for an empty pair, in a fit" begin
        rdq, prq = pair_rd()
        hq(p) = HierarchicalRiskParity(;
                                       opt = HierarchicalOptimiser(; pe = prq,
                                                                   cle = ClustersEstimator(;
                                                                                           de = FeatureDistance(;
                                                                                                                sel = ["f"],
                                                                                                                alg = AggregateDistances(;
                                                                                                                                         pair = p)))))
        @test_throws "the assets (\"d\", \"e\") share no active row" optimise(hq(RefusePair()),
                                                                              rdq)
        # A tie of 20 rows each: the later asset, `e`, departs.
        res = @test_logs (:info, r"\[\"e\"\]") match_mode = :any optimise(hq(DropFewerRows()),
                                                                          rdq)
        @test iszero(res.w[5]) && all(>(0), res.w[1:4])
        res = optimise(hq(FeatureFallback()), rdq)
        @test all(>(0), res.w)
    end
    @testset "A default prior leaves the mask, and so the fit, unchanged" begin
        # The default prior drops `d` itself, so the composed mask equals the prior's mask.
        pr = prior(EmpiricalPrior(), rd)
        imsk = PO.investable_mask(pr)
        @test imsk == BitVector([1, 1, 1, 0, 1])
        @test PO.feature_readable_mask(cle, imsk, rd) == imsk
        @test PO.feature_readable_mask(cle, nothing, PO.port_opt_view(rd, [1, 2, 3, 5])) ===
              nothing
    end
end
