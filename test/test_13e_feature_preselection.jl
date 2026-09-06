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
