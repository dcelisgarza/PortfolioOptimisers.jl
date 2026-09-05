using PortfolioOptimisers, Test, StableRNGs, Statistics, LinearAlgebra, Random

const PO = PortfolioOptimisers

#=
This file owns preselection's read of the feature matrix: `ClusterGroups` clusters on `Z`
when its `cle` carries a `FeatureDistance`, and takes it straight off the data carrier.

Preselection is a *pre-prior* site. A selector is fitted by `fit_preprocessing` from returns
data alone, `maybe_inject_step` has no method for a preprocessing estimator, and a selector
writes `:returns`, which `PIPELINE_INVALIDATES` says invalidates `:prior`. So a prior is
unreachable from here by three independent mechanisms, `ClusterGroups` carries no `z_src`,
and the absence of that field is the statement — see ADR 0045.

Unlike the rest of the feature suite, these tests do *not* need the `RecordingDistance`
instrument. The observable here is the selected **universe**, not the weights, so a wrong
feature matrix changes the answer directly rather than being absorbed by a dendrogram's
leaf ordering.
=#

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
    vals = ["Sector", "Industry"]
    nz = asset_sets_feature_names(vals, sets)
    Z = asset_sets_features(vals, sets)

    rd_tax = ReturnsResult(; nx = nx, X = X, pnl = feature_matrix_panel(nz, Z))
    rd_bare = ReturnsResult(; nx = nx, X = X)

    sel_feat = RedundancySelector(;
                                  alg = ClusterGroups(;
                                                      cle = ClustersEstimator(;
                                                                              de = FeatureDistance())),
                                  score = SCM())
    sel_ret = RedundancySelector(; alg = ClusterGroups(), score = SCM())

    @testset "ClusterGroups carries no z_src" begin
        # A knob restricted to `:data` has one legal position, and a throw needs something
        # to throw on, so the flag is omitted rather than restricted. The same fact settles
        # `x_src`: neither carrier selector exists on a selector.
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
        # `PhylogenyFeatures`-shaped: `nz == nx`, so an asset view slices *both* axes
        Zsq = phylogeny_features(Proximity(), NetworkEstimator(), X)
        @test size(Zsq) == (N, N)

        rd_sq = ReturnsResult(; nx = nx, X = X, pnl = feature_matrix_panel(nx, Zsq))
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

        # square: both axes slice, and `nz` slices with them
        Zsq = phylogeny_features(Proximity(), NetworkEstimator(), X)
        rd_sq = ReturnsResult(; nx = nx, X = X, pnl = feature_matrix_panel(nx, Zsq))
        res_sq = fit_preprocessing(sel_feat, rd_sq)
        rdv_sq = apply_preprocessing(res_sq, rd_sq)
        ksq = length(res_sq.nx)
        idx_sq = [findfirst(==(a), nx) for a in res_sq.nx]
        @test size(panel_feature_matrix(rdv_sq.pnl)[2]) == (ksq, ksq)
        @test panel_feature_matrix(rdv_sq.pnl)[1] == res_sq.nx
        @test panel_feature_matrix(rdv_sq.pnl)[2] == Zsq[idx_sq, idx_sq]
    end

    @testset ":data_only names the pre-prior situation" begin
        # `:neither`'s remedy — "use a FeaturePrior" — is actively wrong here, so the
        # message must not offer it: a prior is structurally unreachable from a selector.
        err = try
            fit_preprocessing(sel_feat, rd_bare)
            nothing
        catch e
            e
        end
        @test err isa PO.IsNothingError
        msg = sprint(showerror, err)
        @test occursin("before any prior exists", msg)
        @test occursin("set `Z` on the `ReturnsResult`", msg)
        @test !occursin("FeaturePrior", msg)
        @test !occursin("z_src", msg)
    end

    @testset "the five diagnostics stay distinct" begin
        f = PO.assert_feature_matrix_supplied
        # a supplied Z never throws, whatever the diagnostic says
        @test isnothing(f(Z, :data_only))
        @test isnothing(f(Z, :neither))

        msgs = Dict(s => sprint(showerror, try
                                    f(nothing, s)
                                catch e
                                    e
                                end) for s in (:none, :data, :prior, :data_only, :neither))
        @test length(unique(values(msgs))) == 5
        @test occursin("raw returns matrix", msgs[:none])
        @test occursin("set `z_src = :prior`", msgs[:data])
        @test occursin("set `z_src = :data`", msgs[:prior])
        @test occursin("only the data carrier can supply one", msgs[:data_only])
        @test occursin("neither the returns result nor the prior result", msgs[:neither])

        # `:data_only` is an explicit branch, so an unrecognised symbol still falls through
        # to `:neither`'s text rather than being captured by it
        @test sprint(showerror, try
                         f(nothing, :DataOnly)
                     catch e
                         e
                     end) == msgs[:neither]
    end

    @testset "the extra keywords are inert on the returns path" begin
        # `Z` and `z_src` ride `kwargs...` to every clustering algorithm, and a present but
        # unused `Z` stays silent — matching `iv`/`ivpa`/`F`/`B`.
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

        # a `z_src` set on a downstream optimiser does not reach the selector: the
        # preselected universe is identical either way
        pipe_zs = Pipeline(;
                           steps = (sel_feat, EmpiricalPrior(),
                                    HierarchicalRiskParity(;
                                                           opt = HierarchicalOptimiser(;
                                                                                       z_src = :prior))))
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

        # loud at every entry point, and never a distance over the wrong axis. The direct
        # `redundancy_keep` call now fails one step earlier and one step more precisely:
        # with the field gone the carrier misses the widened `{nx, X, Z}` contract
        # *structurally* rather than on `X`'s shape, so it is a `FieldError` naming `Z`
        # rather than a `MethodError`. No reachable path changes — the three entry points
        # below still refuse first, and they are the only ways in.
        @test_throws FieldError PO.redundancy_keep(ClusterGroups(), prd, ones(N), true)
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
    distance the caller never chose. `panel_feature_matrix(rd.pnl)[1]` travels beside `panel_feature_matrix(rd.pnl)[2]` to this site for that
    reason, and this testset is the proof it arrives: preselection is the one `Z` consumer
    that reads the carrier directly rather than through `feature_matrix_picker`.
    =#
    @testset "sel reaches the pre-prior site through panel_feature_matrix(rd.pnl)[1]" begin
        # The sector block is the first three columns: `Sector` has three distinct values
        # and `Industry` six, concatenated in the order of `vals`.
        rd_sec = ReturnsResult(; nx = nx, X = X,
                               pnl = feature_matrix_panel(nz[1:3], Z[:, 1:3]))
        # The cut is a real one, so the equality below is not two names for one matrix.
        @test distance(FeatureDistance(), Z; dims = 1) !=
              distance(FeatureDistance(), Z[:, 1:3]; dims = 1)

        sel_key = RedundancySelector(;
                                     alg = ClusterGroups(;
                                                         cle = ClustersEstimator(;
                                                                                 de = FeatureDistance(;
                                                                                                      sel = ["Sector"],
                                                                                                      sets = sets))),
                                     score = SCM())
        sel_idx = RedundancySelector(;
                                     alg = ClusterGroups(;
                                                         cle = ClustersEstimator(;
                                                                                 de = FeatureDistance(;
                                                                                                      sel = [1,
                                                                                                             2,
                                                                                                             3]))),
                                     score = SCM())
        # Selecting the sector block out of the full carrier is the same preselection as
        # carrying the sector block alone.
        @test fit_preprocessing(sel_key, rd_tax).nx ==
              fit_preprocessing(sel_feat, rd_sec).nx
        @test fit_preprocessing(sel_idx, rd_tax).nx ==
              fit_preprocessing(sel_feat, rd_sec).nx
        # `ClusterGroups` carries no `z_src` and needs none, but it does need `nz`: a name
        # that resolves against nothing still diagnoses here.
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
