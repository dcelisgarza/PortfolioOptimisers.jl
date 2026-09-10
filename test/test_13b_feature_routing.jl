using PortfolioOptimisers, Test, Clustering, CSV, DataFrames, TimeSeries, StableRNGs,
      StatsBase, LinearAlgebra, Clarabel

# A `DynamicAbstractWeights` resolves against whatever window it is handed, so its length
# always matches the collapse's observation axis by construction — which is why it, and not
# a static `AbstractWeights`, is what cross-fold weighting requires. Declared here because
# the library ships no concrete subtype.
struct WindowLengthWeights <: PortfolioOptimisers.DynamicAbstractWeights end
function PortfolioOptimisers.get_observation_weights(::WindowLengthWeights,
                                                     X::PortfolioOptimisers.VecNum;
                                                     kwargs...)
    return aweights(collect(range(1, length(X)) ./ sum(1:length(X))))
end
function PortfolioOptimisers.get_observation_weights(::WindowLengthWeights,
                                                     X::PortfolioOptimisers.MatNum;
                                                     dims::Int = 1, kwargs...)
    T = size(X, dims)
    return aweights(collect(range(1, T) ./ sum(1:T)))
end

include(joinpath(@__DIR__, "asset_panel_fixture.jl"))
const PO = PortfolioOptimisers
@testset "Feature matrix routing" begin
    rd0 = prices_to_returns(price_ingestion(PriceIngestion(),
                                            TimeArray(CSV.File(joinpath(@__DIR__,
                                                                        "./assets/SP500.csv.gz"));
                                                      timestamp = :Date)[(end - 252):end];
                                            F = TimeArray(CSV.File(joinpath(@__DIR__,
                                                                            "./assets/Factors.csv.gz"));
                                                          timestamp = :Date)[(end - 252):end]))
    na = size(rd0.X, 2)
    rng = StableRNG(20260728)
    # The user-supplied carrier. Deliberately unrelated to the returns, so a distance
    # derived from it cannot coincide with a correlation distance by accident.
    Zd = abs.(randn(rng, na, 6))
    nzd = ["z$i" for i in 1:6]
    rd = ReturnsResult(; nx = rd0.nx, X = rd0.X, nf = rd0.nf, F = rd0.F, ts = rd0.ts,
                       pnl = matrix_panel(nzd, Zd))
    # The produced panel: factor loadings, `assets × factors`, built at the point of use.
    ape = RegressionPanel()
    pr_fac = prior(FactorPrior(), rd)
    Zp = PO.panel_field(PO.asset_panel(ape, pr_fac, rd, rd.X), "loadings").vals
    pr_noz = prior(EmpiricalPrior(), rd)
    # A carrier the ingestion layer did not build states no universe, and `pnl === nothing`
    # is the one meaning ADR 0132 gives that. `rd0` runs through the layer, so it carries a
    # panel — one with no Panel Field, which is the layer's common case — and the no-panel
    # fixture is therefore built without the layer rather than taken from it.
    rd_noz = ReturnsResult(; nx = rd0.nx, X = rd0.X, nf = rd0.nf, F = rd0.F, ts = rd0.ts)
    fde = FeatureDistance()
    pde = FeatureDistance(; ape = ape)
    cde = Distance(; alg = CanonicalDistance())
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = Dict("verbose" => false))

    @testset "The panel reaches the kernel, and the clusters differ" begin
        cle_f = ClustersEstimator(; de = fde)
        cle_c = ClustersEstimator(; de = cde)

        # `nothing` in the `ape` slot reads the panel the data carrier holds.
        clr_d = clusterise(cle_f, rd)
        @test clr_d.D == distance(fde, Zd)
        # And it is genuinely a different clustering from the correlation one, not a
        # relabelling of it — a test that passes on an ignored panel is worthless.
        clr_c = clusterise(cle_c, rd)
        @test clr_d.D != clr_c.D
        @test clr_d.res.merges != clr_c.res.merges

        # A producer is the other route, and the two give different answers, so the slot is
        # doing real work.
        clr_p = clusterise(ClustersEstimator(; de = pde), pr_fac; rd = rd)
        @test clr_p.D == distance(fde, Zp)
        @test clr_p.D != clr_d.D

        # A producer ignores the carrier's panel entirely: it builds its own.
        @test clusterise(ClustersEstimator(; de = pde), pr_fac; rd = rd_noz).D == clr_p.D
    end

    @testset "Every clustering and network consumer is reachable" begin
        # `cor_and_dist`: hierarchical, non-hierarchical, DBHT.
        for alg in (HClustAlgorithm(), KMeansAlgorithm(), DBHT())
            cle = ClustersEstimator(; de = fde, alg = alg)
            clr = clusterise(cle, rd)
            @test clr.D == distance(fde, Zd)
            @test size(clr.D) == (na, na)
            @test clr.k >= 1
        end
        # `distance`: the tree branch of `calc_adjacency`.
        nte_t = NetworkEstimator(; de = fde, alg = KruskalTree())
        A_f = phylogeny_matrix(nte_t, rd)
        A_c = phylogeny_matrix(NetworkEstimator(; de = cde, alg = KruskalTree()), rd)
        @test A_f.X != A_c.X
        # `cor_and_dist`: the similarity-matrix branch of `calc_adjacency`.
        nte_s = NetworkEstimator(; de = fde, alg = MaximumDistanceSimilarity())
        @test phylogeny_matrix(nte_s, rd).X != phylogeny_matrix(NetworkEstimator(; de = cde,
                                            alg = MaximumDistanceSimilarity()), rd).X
        # `NetworkClustersEstimator`, both branches.
        for nte in (nte_t, nte_s)
            clr = clusterise(NetworkClustersEstimator(; nte = nte), rd)
            @test clr.D == distance(fde, Zd)
        end
        # `logo!`'s `distance(je.de, S, X)`: the second positional is a *similarity
        # matrix*, which is why the three-argument method must not bound it. `logo!` is
        # called from matrix processing, which has no carrier, so a `FeatureDistance`
        # there is unreachable by construction — and says so rather than silently using
        # the returns.
        sigma = pr_noz.sigma
        S = StatsBase.cov2cor(sigma, sqrt.(diag(sigma)))
        @test distance(cde, S, rd0.X) isa Matrix
        @test_throws PortfolioOptimisers.IsNothingError distance(fde, S, rd0.X)
        @test_throws PortfolioOptimisers.IsNothingError PortfolioOptimisers.logo(LoGo(;
                                                                                      de = fde),
                                                                                 copy(sigma),
                                                                                 rd0.X)

        # Constraint generation gets its features for free, through the same bridge.
        plc = SemiDefinitePhylogenyEstimator(; pl = nte_t)
        @test phylogeny_constraints(plc, rd).A == A_f.X
        cte = CentralityEstimator(; pl = nte_t)
        @test centrality_vector(cte, rd).X != centrality_vector(CentralityEstimator(;
                                                pl = NetworkEstimator(; de = cde,
                                                                      alg = KruskalTree())),
                            rd).X
        @test asset_phylogeny(nte_t, fill(inv(na), na), rd) isa Number
        @test average_centrality(cte, fill(inv(na), na), rd) isa Number
    end

    @testset "An absent panel throws, and the message names the cause" begin
        cle = ClustersEstimator(; de = fde)

        # 1. No carrier at all: driven straight from a returns matrix.
        e = try
            clusterise(cle, rd0.X)
        catch err
            err
        end
        @test isa(e, PortfolioOptimisers.IsNothingError)
        @test occursin("ReturnsResult", e.msg)
        @test occursin("RegressionPanel", e.msg)

        # 2. A carrier that holds no panel, in either slot.
        for (pr, rdx) in ((rd_noz, nothing), (rd_noz, rd_noz))
            e = try
                isnothing(rdx) ? clusterise(cle, pr) : clusterise(cle, pr; rd = rdx)
            catch err
                err
            end
            @test isa(e, PortfolioOptimisers.IsNothingError)
            @test occursin("asset_panel", e.msg)
        end

        # 2b. A carrier the layer *did* build holds a panel with no Panel Field, which is
        # the layer's common case: a caller holding only prices has no market
        # capitalisation and no sector. Such a panel states a universe and carries no
        # feature data, and the refusal names that cause — the panel, the Feature Matrix
        # and `asset_panel` — rather than the emptiness of the matrix it would otherwise
        # have derived. Issue #1003.
        e = try
            clusterise(cle, rd0)
        catch err
            err
        end
        @test isa(e, PortfolioOptimisers.IsEmptyError)
        @test isa(rd0.pnl, AssetPanel)
        @test isempty(rd0.pnl.pf)
        @test occursin("Panel Field", e.msg)
        @test occursin("Feature Matrix", e.msg)
        @test occursin("asset_panel", e.msg)
        # The refusal is the resolution's, not the kernel's, so the labels a caller
        # rebuilds refuse with the same message.
        el = try
            feature_labels(rd0.pnl)
        catch err
            err
        end
        @test isa(el, PortfolioOptimisers.IsEmptyError)
        @test el.msg == e.msg
        # And it outranks the selector's own emptiness diagnostic: a `sel` naming a Panel
        # Field a fieldless panel cannot hold is refused for the panel, not for the entry.
        es = try
            feature_matrix(rd0.pnl, [nzd[1]])
        catch err
            err
        end
        @test isa(es, PortfolioOptimisers.IsEmptyError)
        @test es.msg == e.msg

        # 3. A prior result alone carries no panel at all, and the message says so.
        e = try
            clusterise(cle, pr_noz)
        catch err
            err
        end
        @test isa(e, PortfolioOptimisers.IsNothingError)
        @test occursin("a prior result carries no panel", e.msg)

        # 4. A producer that reads a prior, at a site with none.
        e = try
            clusterise(ClustersEstimator(; de = pde), rd)
        catch err
            err
        end
        @test isa(e, PortfolioOptimisers.IsNothingError)
        @test occursin("PhylogenyPanel", e.msg)

        # A panel that is present but unused stays silent, matching `iv`/`ivpa`/`F`/`B`.
        @test clusterise(ClustersEstimator(; de = cde), rd).D ==
              clusterise(ClustersEstimator(; de = cde), rd_noz).D
    end

    @testset "dims is ignored on the routed path" begin
        # The ambient `dims` describes `X`; a stacked Feature Matrix is canonically
        # assets-major, so the three-argument methods hardcode `dims = 1`.
        cle = ClustersEstimator(; de = fde)
        @test clusterise(cle, rd; dims = 1).D == distance(fde, Zd; dims = 1)
        Zsq = abs.(randn(rng, na, na))
        rd_sq = ReturnsResult(; nx = rd0.nx, X = rd0.X, ts = rd0.ts,
                              pnl = matrix_panel(["z$i" for i in 1:na], Zsq))
        # A square matrix is the only shape where a transposed read would not throw, so it is
        # the only one that can prove `dims` is not consulted.
        @test clusterise(cle, rd_sq; dims = 2).D == distance(fde, Zsq; dims = 1)
        @test clusterise(cle, rd_sq; dims = 2).D != distance(fde, Zsq; dims = 2)
    end

    @testset "The producer slot drives a full optimisation" begin
        hopt_d = HierarchicalOptimiser(; cle = ClustersEstimator(; de = fde), slv = slv)
        hopt_p = HierarchicalOptimiser(; pe = FactorPrior(),
                                       cle = ClustersEstimator(; de = pde), slv = slv)
        hopt_c = HierarchicalOptimiser(; cle = ClustersEstimator(; de = cde), slv = slv)
        # The source selector is gone: there is one carrier and one producer slot.
        @test !hasproperty(hopt_d, :z_src)

        for oe in (HierarchicalRiskParity, HierarchicalEqualRiskContribution)
            wd = optimise(oe(; opt = hopt_d), rd)
            wp = optimise(oe(; opt = hopt_p), rd)
            wc = optimise(oe(; opt = hopt_c), rd)
            for res in (wd, wp, wc)
                @test isapprox(sum(res.w), 1)
                @test all(isfinite, res.w)
            end
            # The clusters — hence the weights — actually come from the panel.
            @test wd.clr.D == distance(fde, Zd)
            @test wp.clr.D == distance(fde, Zp)
            @test wd.w != wc.w
            @test wp.w != wd.w
        end

        # `SchurComplementHierarchicalRiskParity` forwards the field too.
        ws = optimise(SchurComplementHierarchicalRiskParity(; opt = hopt_d), rd)
        @test ws.clr.D == distance(fde, Zd)

        jopt = JuMPOptimiser(; slv = slv)
        @test !hasproperty(jopt, :z_src)
        nco = NestedClustered(; cle = ClustersEstimator(; de = fde),
                              opti = MeanRisk(; opt = jopt), opto = MeanRisk(; opt = jopt))
        @test !hasproperty(nco, :z_src)
        wn = optimise(nco, rd)
        @test wn.clr.D == distance(fde, Zd)
        @test isapprox(sum(wn.w), 1)

        # The producer survives the constructor round trip through `port_opt_view`, which is
        # what keeps it stable across NCO clusters and cross-validation folds.
        i = [1, 3, 5, 7]
        @test PortfolioOptimisers.port_opt_view(hopt_p, i, rd.X).cle.de.ape === ape
    end

    @testset "A constraint-generating JuMPOptimiser routes the panel too" begin
        nte = NetworkEstimator(; de = fde, alg = KruskalTree())
        mr_f = MeanRisk(;
                        opt = JuMPOptimiser(; slv = slv,
                                            ple = SemiDefinitePhylogenyEstimator(;
                                                                                 pl = nte)))
        mr_c = MeanRisk(;
                        opt = JuMPOptimiser(; slv = slv,
                                            ple = SemiDefinitePhylogenyEstimator(;
                                                                                 pl = NetworkEstimator(;
                                                                                                       de = cde,
                                                                                                       alg = KruskalTree()))))
        wf = optimise(mr_f, rd)
        wc = optimise(mr_c, rd)
        @test isapprox(sum(wf.w), 1)
        @test wf.w != wc.w
    end

    @testset "collapse_weights rejects a window-length mismatch" begin
        Z3 = abs.(randn(rng, 6, 5, 4))
        # A static `AbstractWeights` is fixed at construction and outlives the fold. A
        # longer one used to be read positionally, silently giving the oldest weights to
        # the newest observations.
        wlong = pweights(fill(1.0, 8))
        wshort = pweights(fill(1.0, 4))
        wexact = pweights([1.0, 1, 1, 1, 1, 5])
        for alg in (AggregateDistances, AggregateFeatures)
            for w in (wlong, wshort)
                @test_throws DimensionMismatch distance(FeatureDistance(;
                                                                        alg = alg(; w = w)),
                                                        Z3)
            end
            @test distance(FeatureDistance(; alg = alg(; w = wexact)), Z3) isa Matrix
        end
        # A `DynamicAbstractWeights` resolves against the window it is handed, so it is
        # fold-local and correct without a length check firing.
        for alg in (AggregateDistances, AggregateFeatures)
            dw = FeatureDistance(; alg = alg(; w = WindowLengthWeights()))
            D6 = distance(dw, Z3)
            D3 = distance(dw, Z3[4:6, :, :])
            @test size(D6) == size(D3) == (5, 5)
            @test D6 != D3
        end
    end

    #=
    An Asset Panel that holds only features can be swallowed whole. One that holds anything
    else -- an observed mask, a one-hot level -- cannot: a distance measured over every
    column is one the caller did not ask for, and it is finite, symmetric and plausible, so
    nothing downstream reports it. `sel` is the cut, and these tests pin what each entry may
    name and what an entry that resolves against nothing does.
    =#
    @testset "sel names Panel Fields, levels, labels and masks" begin
        D_all = distance(fde, Zd; dims = 1)
        D_cut = distance(fde, Zd[:, 1:2]; dims = 1)
        # A panel carrying one of each kind, so every entry form has something to resolve
        # against. `mcap` blanks and `sector` does not, which is what separates the two
        # observed-mask cases.
        gnum = NumericPanelField(; name = "mcap", vals = [1.0, 2.0, 3.0],
                                 omsk = [true, false, true])
        gcat = CategoricalPanelField(; name = "sector", levels = ["T", "E"],
                                     codes = [1, 2, 1])
        gten = TensorPanelField(; name = "beta", axis = "factor", labels = ["mkt", "smb"],
                                vals = [1.0 2.0; 3.0 4.0; 5.0 6.0],
                                omsk = [true true; true false; true true])
        gpnl = AssetPanel(; pf = [gnum, gcat, gten])

        @testset "construction refuses what cannot be read" begin
            @test isnothing(FeatureDistance().sel)
            @test FeatureDistance().strict === false
            # An empty selection is refused rather than read as "every Panel Field":
            # `nothing` already says that, and a selection that silently widens to the whole
            # panel is the failure `sel` exists to remove.
            @test_throws PortfolioOptimisers.IsEmptyError FeatureDistance(; sel = String[])
            @test_throws ArgumentError FeatureDistance(; sel = ["z1", "z1"])
            # There is no integer entry: every field, level and label carries a name, so a
            # position has nothing to index.
            @test_throws ArgumentError FeatureDistance(; sel = [1, 2])
            @test_throws ArgumentError FeatureDistance(; sel = [""])
            @test_throws ArgumentError FeatureDistance(; sel = ["z1" => :nope])
            @test_throws ArgumentError FeatureDistance(; sel = ["z1" => String[]])
            @test_throws ArgumentError FeatureDistance(; sel = ["z1" => ["a", "a"]])
            @test_throws ArgumentError FeatureDistance(; sel = ["z1" => ["a", ""]])
            @test_throws ArgumentError FeatureDistance(; sel = [1 => "a"])
            @test_throws ArgumentError FeatureDistance(; sel = ["" => "a"])
            @test_throws ArgumentError FeatureDistance(; sel = ["z1" => 1])
        end

        @testset "a nothing selector stacks every Panel Field's values" begin
            @test feature_labels(rd.pnl) == ["z$i" for i in 1:6]
            @test feature_matrix(rd.pnl) == Zd
            @test distance(FeatureDistance(), nothing, rd.X; rd = rd) == D_all
            # A mask is never among them: a bare name, and an absent selector, are the
            # values alone.
            @test feature_labels(gpnl) ==
                  ["mcap", "sector" => "T", "sector" => "E", "beta" => "mkt",
                   "beta" => "smb"]
        end

        @testset "the four entry forms, and the order of sel is the column order" begin
            @test feature_labels(gpnl, ["sector"]) == ["sector" => "T", "sector" => "E"]
            @test feature_labels(gpnl, ["sector" => "E"]) == ["sector" => "E"]
            @test feature_labels(gpnl, ["beta" => ["smb", "mkt"]]) ==
                  ["beta" => "smb", "beta" => "mkt"]
            @test feature_labels(gpnl, ["mcap" => :observed]) == ["mcap" => :observed]
            # Mixed entries are admitted, and the vector's order is the column order.
            sel = ["beta" => "smb", "mcap", "sector" => ["E"], "mcap" => :observed]
            @test feature_labels(gpnl, sel) ==
                  ["beta" => "smb", "mcap", "sector" => "E", "mcap" => :observed]
            @test feature_matrix(gpnl, sel) ==
                  [2.0 1.0 0.0 1.0; 4.0 2.0 1.0 0.0; 6.0 3.0 0.0 1.0]
            @test feature_matrix(gpnl, reverse(sel)) ==
                  feature_matrix(gpnl, sel)[:, [4, 3, 2, 1]]
        end

        @testset "a label vector is a selector that rebuilds the same matrix" begin
            for sel in (nothing, ["mcap"], ["sector"], ["beta" => ["smb", "mkt"]],
                        ["mcap" => :observed, "beta" => :observed, "sector"])
                lab = feature_labels(gpnl, sel)
                @test feature_matrix(gpnl, lab) == feature_matrix(gpnl, sel)
                @test feature_labels(gpnl, lab) == lab
            end
        end

        @testset "a Panel Field contributes one observed-mask column" begin
            # One column whatever the kind, and whatever the number of value columns.
            @test size(feature_matrix(gpnl, ["beta" => :observed]), 2) == 1
            @test feature_matrix(gpnl, ["mcap" => :observed]) ==
                  reshape([1.0, 0.0, 1.0], 3, 1)
            # A tensor Panel Field's mask carries a label axis of its own, so its column
            # holds where every label of that asset was observed.
            @test feature_matrix(gpnl, ["beta" => :observed]) ==
                  reshape([1.0, 0.0, 1.0], 3, 1)
            # `omsk === nothing` means the Panel Field cannot blank, so every cell was
            # observed and the column is ones. That holds for a tensor Panel Field too,
            # whose mask would otherwise be reduced over its label axis.
            @test feature_matrix(gpnl, ["sector" => :observed]) == ones(3, 1)
            bare = TensorPanelField(; name = "raw", axis = "factor",
                                    labels = ["mkt", "smb"],
                                    vals = [1.0 2.0; 3.0 4.0; 5.0 6.0])
            @test feature_matrix(AssetPanel(; pf = [bare]), ["raw" => :observed]) ==
                  ones(3, 1)
        end

        @testset "names select the same columns through the routed entry point" begin
            de_cut = FeatureDistance(; sel = ["z1", "z2"])
            @test distance(de_cut, nothing, rd.X; rd = rd) == D_cut
            @test distance(de_cut, nothing, rd.X; rd = rd) != D_all
            S, D = cor_and_dist(de_cut, nothing, rd.X; rd = rd)
            @test D == D_cut
            @test S == PortfolioOptimisers.distance_to_similarity(fde.sim; D = D_cut)
            k = [4, 1, 6]
            @test feature_matrix(rd.pnl, ["z$i" for i in k]) == Zd[:, k]
        end

        @testset "an unresolvable field, level or label warns and drops, or throws" begin
            @test (@test_logs (:warn,) feature_matrix(rd.pnl, ["z1", "z2", "nope"])) ==
                  Zd[:, 1:2]
            @test_throws ArgumentError feature_matrix(rd.pnl, ["z1", "nope"]; strict = true)
            # A level and a label resolve in their own Panel Field's namespace.
            @test (@test_logs (:warn,) feature_labels(gpnl, ["sector" => "nope", "mcap"])) ==
                  ["mcap"]
            @test_throws ArgumentError feature_matrix(gpnl, ["beta" => ["nope"]];
                                                      strict = true)
            # A numeric Panel Field has no second namespace, so every key paired with it is
            # absent.
            @test_throws ArgumentError feature_matrix(gpnl, ["mcap" => "nope"];
                                                      strict = true)
            # Every entry dropping leaves nothing to measure. That is not a droppable
            # thing, so it throws whatever `strict` says.
            @test_throws PortfolioOptimisers.IsEmptyError feature_matrix(rd.pnl,
                                                                         ["no1", "no2"])
            @test_throws PortfolioOptimisers.IsEmptyError distance(FeatureDistance(;
                                                                                   sel = ["no1"]),
                                                                   nothing, rd.X; rd = rd)
            # Two entries that expand to one column double that column's contribution.
            @test_throws ArgumentError feature_matrix(gpnl, ["sector", "sector" => "T"])
        end

        @testset "dims names the asset axis at the raw-matrix entry point" begin
            # The routed entry point always stacks assets-major, so `dims` is meaningful
            # only where a caller hands a matrix in directly.
            @test distance(fde, permutedims(Zd[:, 1:2]); dims = 2) == D_cut
            Z3s = permutedims(cat(Zd, 2 * Zd; dims = 3), (3, 1, 2))
            @test distance(fde, Z3s[:, :, 1:2]; dims = 1) ==
                  distance(fde, Z3s[:, :, 1:2]; dims = 1)
        end

        @testset "the square case selects reference assets, not assets" begin
            # When the feature axis is the asset axis, a name is an asset used as a
            # reference column. Every row survives, so the matrix stays assets x assets.
            nxs = ["A", "B", "C"]
            Zsq = [1.0 0.2 0.1; 0.2 1.0 0.7; 0.1 0.7 1.0]
            sq_pnl = asset_panel([TensorPanelInput(; name = "prox", axis = "asset",
                                                   labels = nxs, vals = Zsq)])
            @test PO.features_are_assets(PO.panel_field(sq_pnl, "prox"), nxs)
            rd_sq3 = ReturnsResult(; nx = nxs, X = randn(rng, 40, 3) / 100, pnl = sq_pnl)
            D_sq = distance(FeatureDistance(; sel = ["prox" => ["A", "B"]]), nothing,
                            rd_sq3.X; rd = rd_sq3)
            @test size(D_sq) == (3, 3)
            @test D_sq == distance(fde, Zsq[:, 1:2]; dims = 1)
        end

        @testset "the panel travels whole, and a producer builds its own" begin
            # There is one carrier and one producer slot: the resolution reads either
            # carrier slot, and a producer ignores both panels and builds a fresh one.
            @test PO.asset_panel(nothing, pr_noz, rd, rd.X) === rd.pnl
            @test PO.asset_panel(nothing, rd, nothing, rd.X) === rd.pnl
            @test isdisjoint(feature_labels(PO.asset_panel(ape, pr_fac, rd, rd.X)),
                             feature_labels(rd.pnl))
        end

        @testset "the selector survives the whole routed path" begin
            de_sel = FeatureDistance(; sel = ["z1", "z2"])
            rd_cut = ReturnsResult(; nx = rd.nx, X = rd.X,
                                   pnl = matrix_panel(["z1", "z2"], Zd[:, 1:2]))
            pm_sel = phylogeny_matrix(NetworkEstimator(; de = de_sel), rd)
            pm_ref = phylogeny_matrix(NetworkEstimator(; de = fde), rd_cut)
            pm_all = phylogeny_matrix(NetworkEstimator(; de = fde), rd)
            @test pm_sel.X == pm_ref.X
            @test pm_sel.X != pm_all.X
            # A name the produced panel does not carry cannot resolve: `strict` names the
            # entry that failed, and the default drops every entry and reports an empty
            # selection.
            de_strict = FeatureDistance(; ape = ape, sel = ["z1", "z2"], strict = true)
            @test_throws ArgumentError phylogeny_matrix(NetworkEstimator(; de = de_strict),
                                                        pr_fac; rd = rd)
            @test_throws PortfolioOptimisers.IsEmptyError phylogeny_matrix(NetworkEstimator(;
                                                                                            de = FeatureDistance(;
                                                                                                                 ape = ape,
                                                                                                                 sel = ["z1",
                                                                                                                        "z2"])),
                                                                           pr_fac; rd = rd)
        end

        @testset "factory carries the new fields" begin
            de_sel = FeatureDistance(; ape = ape, sel = ["z1", "z2"], strict = true)
            f = factory(de_sel, pr_noz)
            @test f.sel == de_sel.sel
            @test f.ape == de_sel.ape
            @test f.strict === de_sel.strict
        end
    end
end
