using PortfolioOptimisers, Test, StableRNGs, Statistics, LinearAlgebra, Random

const PO = PortfolioOptimisers

#=
This file owns preselection's read of the feature matrix: `ClusterGroups` clusters on `Z`
when its `cle` carries a `FeatureDistance`, and takes it straight off the data carrier.

Preselection is a *pre-prior* site. A selector is fitted by `fit_preprocessing` from returns
data alone, `maybe_inject_step` has no method for a preprocessing estimator, and a selector
writes `:returns`, which `PIPELINE_INVALIDATES` says invalidates `:prior`. So a prior is
unreachable from here by three independent mechanisms, `ClusterGroups` carries no source
selector, and the absence of that field is the statement — see ADR 0045.

Unlike the rest of the feature suite, these tests do *not* need the `RecordingDistance`
instrument. The observable here is the selected **universe**, not the weights, so a wrong
feature matrix changes the answer directly rather than being absorbed by a dendrogram's
leaf ordering.
=#

include(joinpath(@__DIR__, "asset_panel_fixture.jl"))
# A square block is one tensor Panel Field whose labels are the asset names.
function sqpanel(labels, vals)
    return asset_panel([TensorPanelInput(; name = "prox", axis = "asset", labels = labels,
                                         vals = vals)])
end
@testset "Feature-driven preselection" begin
    rng = StableRNG(987654321)
    T, N = 200, 12
    X = randn(rng, T, N) * 0.01
    nx = ["A$i" for i in 1:N]

    # A nested taxonomy: three sectors, each split into two industries. This is the
    # motivating case — keeping one representative per sector is a statement about the
    # classification, not about the sample correlation.
    sets = UniverseSets(; xkey = "nx",
                        dict = Dict("nx" => nx,
                                    "Sector" =>
                                        repeat(["Tech", "Energy", "Fin"], inner = 4),
                                    "Industry" =>
                                        repeat(["Semis", "Soft", "Oil", "Gas", "Bank",
                                                "Ins"], inner = 2)))
    tax_pnl = asset_panel(panel_input(sets, ["Sector", "Industry"]))
    nz, Z = panel_feature_matrix(tax_pnl)

    rd_tax = ReturnsResult(; nx = nx, X = X, pnl = tax_pnl)
    rd_bare = ReturnsResult(; nx = nx, X = X)

    sel_feat = RedundancySelector(;
                                  alg = ClusterGroups(;
                                                      cle = ClustersEstimator(;
                                                                              de = FeatureDistance())),
                                  score = SCM())
    sel_ret = RedundancySelector(; alg = ClusterGroups(), score = SCM())

    @testset "ClusterGroups carries no source selector" begin
        # A knob restricted to the data carrier has one legal position, and a throw needs
        # something to throw on, so the flag is omitted rather than restricted. The same
        # fact settles `x_src`: neither carrier selector exists on a selector.
        @test fieldnames(ClusterGroups) == (:cle,)
        @test !hasproperty(ClusterGroups(), :z_src)
        @test !hasproperty(RedundancySelector(; alg = ClusterGroups(), score = SCM()),
                           :z_src)
        # The `cle` reads `rd.X` as it arrived, so it states no source (#1698, #1705).
        for x_src in (DataSource(), ExcessSource(), PriorSource())
            @test_throws ArgumentError ClusterGroups(;
                                                     cle = ClustersEstimator(;
                                                                             x_src = x_src))
        end
        @test isnothing(ClusterGroups(; cle = ClustersEstimator(; x_src = nothing)).cle.x_src)
    end

    @testset "a taxonomy selects a different universe from the returns correlation" begin
        kept_feat = fit_preprocessing(sel_feat, rd_tax).nx
        kept_ret = fit_preprocessing(sel_ret, rd_tax).nx

        # the universe itself is the observable, so this is a direct assertion
        @test kept_feat != kept_ret
        @test !isempty(kept_feat)
        @test length(kept_feat) < N
        @test all(in(nx), kept_feat)

        # the taxonomy's own structure survives: one representative per sector, and no two
        # survivors share one
        sector = Dict(zip(nx, sets.dict["Sector"]))
        @test allunique(sector[a] for a in kept_feat)

        # the returns carrier is untouched by the feature read, so the correlation-driven
        # answer is the same whether or not `Z` is present
        @test fit_preprocessing(sel_ret, rd_bare).nx == kept_ret
    end

    @testset "a square carrier selects on its own neighbourhood structure" begin
        # `PhylogenyPanel`-shaped: the labels are the asset names, so an asset view
        # slices *both* axes
        Zsq = phylogeny_features(Proximity(), NetworkEstimator(), X)
        @test size(Zsq) == (N, N)

        rd_sq = ReturnsResult(; nx = nx, X = X, pnl = sqpanel(nx, Zsq))
        kept_sq = fit_preprocessing(sel_feat, rd_sq).nx

        @test kept_sq != fit_preprocessing(sel_ret, rd_sq).nx
        # a square carrier and a rectangular one are different feature spaces over the same
        # universe, so they need not agree either
        @test kept_sq != fit_preprocessing(sel_feat, rd_tax).nx
        @test !isempty(kept_sq)
        @test length(kept_sq) < N
    end

    @testset "the replay half slices Z through port_opt_view" begin
        res = fit_preprocessing(sel_feat, rd_tax)
        rdv = apply_preprocessing(res, rd_tax)

        k = length(res.nx)
        @test rdv.nx == res.nx
        @test size(rdv.X) == (T, k)
        # rectangular: the asset axis is sliced, the feature axis is not
        @test size(panel_feature_matrix(rdv.pnl)[2]) == (k, size(Z, 2))
        @test panel_feature_matrix(rdv.pnl)[1] == nz
        # and the rows are the fitted assets' own rows, in fitted order
        idx = [findfirst(==(a), nx) for a in res.nx]
        @test panel_feature_matrix(rdv.pnl)[2] == Z[idx, :]

        # the selection is decided on the *full* universe and sliced only afterwards, so
        # the fitted answer does not depend on the slice
        @test fit_preprocessing(sel_feat, rdv).nx ⊆ res.nx

        # square: both axes slice, and the labels slice with them
        Zsq = phylogeny_features(Proximity(), NetworkEstimator(), X)
        rd_sq = ReturnsResult(; nx = nx, X = X, pnl = sqpanel(nx, Zsq))
        res_sq = fit_preprocessing(sel_feat, rd_sq)
        rdv_sq = apply_preprocessing(res_sq, rd_sq)
        ksq = length(res_sq.nx)
        idx_sq = [findfirst(==(a), nx) for a in res_sq.nx]
        f = PO.panel_field(rdv_sq.pnl, "prox")
        @test size(f.vals) == (ksq, ksq)
        @test f.labels == res_sq.nx
        @test f.vals == Zsq[idx_sq, idx_sq]
    end

    @testset "the refusal names the two routes, and neither is a prior" begin
        # A prior is structurally unreachable from a selector, so the message must not offer
        # one: the two ways forward are a panel on the carrier, or a producer that reads no
        # prior.
        err = try
            fit_preprocessing(sel_feat, rd_bare)
            nothing
        catch e
            e
        end
        @test err isa PO.IsNothingError
        msg = sprint(showerror, err)
        @test occursin("asset_panel", msg)
        @test occursin("RegressionPanel", msg)
        @test !occursin("z_src", msg)

        # A producer that *does* read a prior is the other refusal, and it names the site.
        sel_reg = RedundancySelector(;
                                     alg = ClusterGroups(;
                                                         cle = ClustersEstimator(;
                                                                                 de = FeatureDistance(;
                                                                                                      ape = RegressionPanel()))),
                                     score = SCM())
        err = try
            fit_preprocessing(sel_reg, rd_tax)
            nothing
        catch e
            e
        end
        @test err isa PO.IsNothingError
        @test occursin("PhylogenyPanel", sprint(showerror, err))

        # And a producer that reads none runs here, which is what makes the site usable.
        sel_phy = RedundancySelector(;
                                     alg = ClusterGroups(;
                                                         cle = ClustersEstimator(;
                                                                                 de = FeatureDistance(;
                                                                                                      ape = PhylogenyPanel()))),
                                     score = SCM())
        @test !isempty(fit_preprocessing(sel_phy, rd_bare).nx)
    end

    @testset "the extra keywords are inert on the returns path" begin
        # The two carriers ride `kwargs...` to every clustering algorithm, and a present
        # but unused panel stays silent — matching `iv`/`ivpa`/`F`/`B`.
        for cle in (ClustersEstimator(), ClustersEstimator(; alg = DBHT()),
                    ClustersEstimator(; alg = KMeansAlgorithm()))
            sel = RedundancySelector(; alg = ClusterGroups(; cle = cle), score = SCM())
            # `KMeansAlgorithm` draws its starts from the global RNG, so the two calls must
            # begin from the same state or the comparison measures the RNG rather than the
            # presence of `Z`.
            with_tax = (Random.seed!(1); fit_preprocessing(sel, rd_tax).nx)
            without = (Random.seed!(1); fit_preprocessing(sel, rd_bare).nx)
            @test with_tax == without
        end
    end

    @testset "end to end through a Pipeline" begin
        pipe = Pipeline(; steps = (sel_feat, EmpiricalPrior(), EqualWeighted()))
        res = fit(pipe, rd_tax)
        kept = res.ctx.returns.nx
        @test kept == fit_preprocessing(sel_feat, rd_tax).nx
        @test length(res.ctx.opt.w) == length(kept)
        # the surviving window still carries a sliced feature matrix
        @test size(panel_feature_matrix(res.ctx.returns.pnl)[2]) ==
              (length(kept), size(Z, 2))
        @test predict(res, rd_tax) isa Any

        # a producer set on a downstream optimiser does not reach the selector: the
        # preselected universe is identical either way
        pipe_zs = Pipeline(;
                           steps = (sel_feat, EmpiricalPrior(),
                                    HierarchicalRiskParity(;
                                                           opt = HierarchicalOptimiser(;
                                                                                       cle = ClustersEstimator(;
                                                                                                               de = FeatureDistance(;
                                                                                                                                    ape = PhylogenyPanel()))))))
        @test fit(pipe_zs, rd_tax).ctx.returns.nx == kept
    end

    @testset "PredictionReturnsResult is refused, and no longer carries a Z at all" begin
        # #180 expected this carrier to become readable here. It never did: `X` is a
        # *portfolio* return vector — the asset axis is the thing the collapse removed — so
        # the type satisfies neither the old `{nx, X}` contract nor the widened `{nx, X, Z}`
        # one, and the refusal has nothing to do with `Z`. That independence is why the
        # refusal survives the carrier's `nz`/`Z` being deleted outright: the selectors
        # never depended on them.
        prd = PO.PredictionReturnsResult(; nx = nx, X = X * fill(1 / N, N))
        @test :Z ∉ fieldnames(PO.PredictionReturnsResult)
        @test size(prd.X, 2) == 1 != length(prd.nx)

        # Loud at every entry point, and never a distance over the wrong axis. The three
        # entry points below are the only ways in, and each refuses on the carrier's type
        # before any feature read happens. `redundancy_keep` itself is not reachable with
        # this carrier, and it is deliberately not gated a second time: the panel is
        # resolved inside the kernel now, so a structural miss there would be a duplicate of
        # the refusal the entry points already give.
        @test_throws MethodError fit_preprocessing(sel_feat, prd)
        @test_throws MethodError fit_preprocessing(sel_ret, prd)
        # even a selector that reads no feature matrix at all
        @test_throws MethodError fit_preprocessing(CompleteAssetSelector(), prd)
        # and the replay half refuses on the `port_opt_view` tripwire
        @test_throws ArgumentError apply_preprocessing(PO.AssetSelectorResult(nx[1:3]), prd)
    end

    #=
    A panel presents every slice as a feature, the observed masks and the one-hot levels
    included, so a redundancy selector that measured all of them would drop assets on a
    distance the caller never chose. The panel names its own columns, and this testset is
    the proof those names arrive: preselection passes `rd` alone, and the selector resolves
    against the names the panel derives.
    =#
    @testset "sel reaches the pre-prior site through the panel's own names" begin
        # The sector block is the first three columns: `Sector` has three distinct values
        # and `Industry` six, concatenated in the order of `vals`.
        rd_sec = ReturnsResult(; nx = nx, X = X, pnl = matrix_panel(nz[1:3], Z[:, 1:3]))
        # The cut is a real one, so the equality below is not two names for one matrix.
        @test distance(FeatureDistance(), Z; dims = 1) !=
              distance(FeatureDistance(), Z[:, 1:3]; dims = 1)

        # The whole `Sector` field, by name, and the same block written level by level.
        sel_key = RedundancySelector(;
                                     alg = ClusterGroups(;
                                                         cle = ClustersEstimator(;
                                                                                 de = FeatureDistance(;
                                                                                                      sel = ["Sector"]))),
                                     score = SCM())
        levels = PO.panel_field(rd_tax.pnl, "Sector").levels
        sel_lvl = RedundancySelector(;
                                     alg = ClusterGroups(;
                                                         cle = ClustersEstimator(;
                                                                                 de = FeatureDistance(;
                                                                                                      sel = ["Sector" =>
                                                                                                                 levels]))),
                                     score = SCM())
        # Selecting the sector block out of the full carrier is the same preselection as
        # carrying the sector block alone.
        @test fit_preprocessing(sel_key, rd_tax).nx ==
              fit_preprocessing(sel_feat, rd_sec).nx
        @test fit_preprocessing(sel_lvl, rd_tax).nx ==
              fit_preprocessing(sel_feat, rd_sec).nx
        # `ClusterGroups` carries no source selector and needs none, but it does need the
        # panel's names: a name that resolves against nothing still diagnoses here.
        sel_bad = RedundancySelector(;
                                     alg = ClusterGroups(;
                                                         cle = ClustersEstimator(;
                                                                                 de = FeatureDistance(;
                                                                                                      sel = ["nope"],
                                                                                                      strict = true))),
                                     score = SCM())
        @test_throws ArgumentError fit_preprocessing(sel_bad, rd_tax)
    end
end

using PortfolioOptimisers, Test, StableRNGs, StatsBase, LinearAlgebra, Clarabel
using PortfolioOptimisers.Distances: Euclidean, Cityblock, SqEuclidean, Minkowski,
                                     ChiSqDist, WeightedEuclidean, WeightedCityblock,
                                     CosineDist, Chebyshev, WeightedSqEuclidean,
                                     WeightedMinkowski, TotalVariation, BrayCurtis, Jaccard,
                                     CorrDist, MeanAbsDeviation, MeanSqDeviation,
                                     RMSDeviation

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
        msg = "FeatureDistance has no value to read for the assets [\"b\"]: each is inactive at the last row of the window, or a value column of `sel` holds a placeholder there."
        @test_throws msg hand_distance(fdist(LastObservation()), rd)
        # On a view without `b`, the last row is 4 against 7.
        D = hand_distance(fdist(LastObservation()), PO.port_opt_view(rd, [1, 3]))
        @test D == [0.0 3.0; 3.0 0.0]
        # LastActiveRow reads `a` at row 4 (4), `b` at row 2 (4), and `c` at row 4 (7).
        D = hand_distance(fdist(LastObservation(; alg = LastActiveRow())), rd)
        @test D == [0.0 0.0 3.0; 0.0 0.0 3.0; 3.0 3.0 0.0]
        # A direct `cor_and_dist` or `clusterise` call refuses `b` by name too (#1771).
        @test_throws msg cor_and_dist(fdist(LastObservation()), nothing, rd.X; rd = rd)
        @test_throws msg clusterise(ClustersEstimator(; de = fdist(LastObservation())), rd)
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
        # A direct `cor_and_dist` or `clusterise` call refuses the pair too (#1771).
        @test_throws msg cor_and_dist(fdist(AggregateDistances()), nothing, rd.X; rd = rd)
        @test_throws msg clusterise(ClustersEstimator(; de = fdist(AggregateDistances())),
                                    rd)
        msg = "DropFewerRows drops an asset at the entry of a fit"
        de = fdist(AggregateDistances(; pair = DropFewerRows()))
        @test_throws msg hand_distance(de, rd)
        @test_throws msg clusterise(ClustersEstimator(; de = de), rd)
        # The fallback measures (b, c) by their means over their own rows, 3 against 6.
        D = hand_distance(fdist(AggregateDistances(; pair = FeatureFallback())), rd)
        @test D ≈ [0 1.5 2.5; 1.5 0 3; 2.5 3 0] rtol = 1e-15
        # Weights 1:4: (a, b) = (1 * 1 + 2 * 2) / 3, (a, c) = (3 * 2 + 4 * 3) / 7.
        D = hand_distance(fdist(AggregateDistances(; w = StatsBase.Weights(1.0:4.0),
                                                   pair = FeatureFallback())), rd)
        @test D[1, 2] ≈ 5 / 3 rtol = 1e-15
        @test D[1, 3] ≈ 18 / 7 rtol = 1e-15
        @test D[2, 3] ≈ 3 rtol = 1e-15
        # The fallback takes its own collapse and weights (#1771). `b` is active on rows 1 to
        # 3, at 2, 4 and 9, and `c` on row 4 alone, at 7. The median of `b` is 4 and its mean
        # 5. Under weights 1:4 it is (2 + 8 + 27) / 6, and a stack of four equal rows of the
        # pair is twice the gap under the Euclidean metric.
        rd3 = hand_rd(; A = Bool[1 1 0; 1 1 0; 1 1 0; 1 0 1])
        w4 = StatsBase.Weights(1.0:4.0)
        fbd(fb) = hand_distance(fdist(AggregateDistances(; pair = fb)), rd3)[2, 3]
        @test fbd(FeatureFallback()) ≈ 2 rtol = 1e-15
        @test fbd(FeatureFallback(; alg = MedianCollapse())) ≈ 3 rtol = 1e-15
        @test fbd(FeatureFallback(; w = w4)) ≈ 7 - 37 / 6 rtol = 1e-14
        D = hand_distance(fdist(StackObservations(; pair = FeatureFallback(; w = w4))), rd3)
        @test D[2, 3] ≈ 2 * (7 - 37 / 6) rtol = 1e-14
        # An asset whose active rows carry a zero weight has no distance to read. The total of
        # `c` was `0 / 0` on the diagonal, which no pair rule reaches, and the refusal named
        # the pair (a, c) as one that shares no active row (R7 of #1630).
        w0 = StatsBase.Weights([1.0, 1, 0, 0])
        msg = "the observation weights of those rows sum to zero for the assets [\"c\"], so no distance of the asset has a value to read"
        for pair in (RefusePair(), DropFewerRows(), FeatureFallback())
            @test_throws msg hand_distance(fdist(AggregateDistances(; w = w0, pair = pair)),
                                           rd)
        end
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
        # So do the other ratio metrics and the mean deviations (#1771). On this panel (a, c)
        # shares rows 1, 3 and 4, at (1, 3, 4) and (9, 5, 7), and (a, b) rows 1 and 2. A
        # rescale of either pair would move it by a factor of 4 / 3 or 2.
        rd2 = hand_rd(; A = Bool[1 1 1; 1 1 0; 1 0 1; 1 0 1])
        for m in
            (CorrDist(), BrayCurtis(), Jaccard(), MeanAbsDeviation(), MeanSqDeviation(),
             RMSDeviation())
            D = hand_distance(fdist(StackObservations(; pair = fb); metric = m), rd2)
            @test D[1, 3] ≈ m([1.0, 3, 4], [9.0, 5, 7]) rtol = 1e-14
            @test D[1, 2] ≈ m([1.0, 2], [2.0, 4]) atol = 1e-15
        end
        # A stack of two features rescales over the coordinates of both, and the fallback
        # fills each feature with its own mean. `g` holds (0.5, 1, 2, 2), (1, 3, 8, 8) and
        # (3, 3, 1, 4) for `a`, `b` and `c`.
        G = [0.5 1 3; 1 3 3; 2 8 1; 2 8 4]
        pnl2 = AssetPanel(;
                          pf = [NumericPanelField(; name = "f", vals = rd.pnl.pf[1].vals),
                                NumericPanelField(; name = "g", vals = G)],
                          amsk = rd.pnl.amsk, emsk = rd.pnl.emsk)
        rdg = ReturnsResult(; nx = rd.nx, X = rd.X, pnl = pnl2)
        de2 = FeatureDistance(; metric = Euclidean(), alg = StackObservations(; pair = fb),
                              sel = ["f", "g"])
        D = hand_distance(de2, rdg)
        # (a, b): (1, 2, 0.5, 1) against (2, 4, 1, 3) on four of eight coordinates.
        @test D[1, 2] ≈ sqrt(1 + 4 + 0.25 + 4) * sqrt(2) rtol = 1e-15
        # (a, c): (3, 4, 2, 2) against (5, 7, 1, 4).
        @test D[1, 3] ≈ sqrt(4 + 9 + 1 + 4) * sqrt(2) rtol = 1e-15
        # (b, c): the means (3, 2) against (6, 2.5), on four rows.
        @test D[2, 3] ≈ 2 * sqrt(9 + 0.25) rtol = 1e-15
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
        msg = "no value to read for the assets at the positions [2]: each has no readable row in the window: no row at which it is active and every value column of `sel` holds data, observed or filled. Inside a fit"
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
    @testset "The checks read only the readable cells (#1734)" begin
        # The cell (1, 3) is inactive. Its value refuses nothing and moves no distance, under
        # each collapse that reads the window and each metric, in either layout.
        Z = reshape([1.0 2 9; 2 4 9; 3 9 5; 4 9 7], 4, 3, 1)
        A = Bool[1 1 0; 1 1 0; 1 0 1; 1 0 1]
        fb = FeatureFallback()
        metrics = (Euclidean(), AngularDist(), ChiSqDist(), BrayCurtis(), Jaccard())
        algs = (LastObservation(; alg = LastActiveRow()), AggregateFeatures(),
                AggregateDistances(; pair = fb), StackObservations(; pair = fb))
        for v in (-9.0, NaN, Inf), metric in metrics, alg in algs
            Zv = copy(Z)
            Zv[1, 3, 1] = v
            de = FeatureDistance(; metric = metric, alg = alg)
            @test distance(de, Zv; amsk = A) == distance(de, Z; amsk = A)
            @test distance(de, permutedims(Zv, (1, 3, 2)); dims = 2, amsk = A) ==
                  distance(de, Z; amsk = A)
            @test cor_and_dist(de, Zv; amsk = A) == cor_and_dist(de, Z; amsk = A)
        end
        # A fit reads the hand panel with -9 at the inactive cell as it reads the stored 9.
        rdn = hand_rd(; V = [1.0 2 -9; 2 4 9; 3 9 5; 4 9 7])
        for metric in (ChiSqDist(), BrayCurtis(), Jaccard()), alg in algs
            de = fdist(alg; metric = metric)
            @test hand_distance(de, rdn) == hand_distance(de, rd)
        end
        # A readable cell is still checked.
        Zv = copy(Z)
        Zv[1, 1, 1] = NaN
        @test_throws "all(isfinite, Z) must hold" distance(fdist(AggregateFeatures()), Zv;
                                                           amsk = A)
        Zv[1, 1, 1] = -1
        de = FeatureDistance(; metric = ChiSqDist(), alg = AggregateFeatures())
        @test_throws DomainError distance(de, Zv; amsk = A)
        # A static matrix refuses an unreadable asset as unreadable, whatever its cells hold.
        msg = "no value to read for the assets at the positions [2]"
        @test_throws msg distance(fdist(LastObservation()), [1.0 2; NaN 3; 4 5];
                                  amsk = Bool[1 0 1])
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
        # The rule repeats until no pair is empty (#1771). (a, b) is the first empty pair,
        # and `b` has fewer rows; then (a, c) is empty, and `c` has fewer rows. One pass
        # would keep `c`.
        A = Bool[1 0 0 1; 1 0 0 1; 1 0 0 1; 0 1 1 1]
        rd4 = hand_rd(; V = [1.0 2 9 3; 2 4 9 4; 3 9 5 5; 4 9 7 6], A = A,
                      nx = ["a", "b", "c", "d"])
        @test PO.feature_readable_mask(de, nothing, rd4) == BitVector([1, 0, 0, 1])
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
                        # The vector arms enter at sites of their own (#1771).
                        "HierarchicalRiskParity, a vector of measures" =>
                            HierarchicalRiskParity(; r = [Variance(), ConditionalValueatRisk()], opt = ho),
                        "SchurComplementHierarchicalRiskParity, a vector of params" =>
                            SchurComplementHierarchicalRiskParity(;
                                                                  params = [SchurComplementParams(),
                                                                            SchurComplementParams(;
                                                                                                  gamma = 0.5)],
                                                                  opt = ho),
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
