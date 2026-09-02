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

@testset "Feature matrix routing" begin
    rd0 = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                      timestamp = :Date)[(end - 252):end],
                            TimeArray(CSV.File(joinpath(@__DIR__,
                                                        "./assets/Factors.csv.gz"));
                                      timestamp = :Date)[(end - 252):end])
    na = size(rd0.X, 2)
    rng = StableRNG(20260728)
    # The user-supplied carrier. Deliberately unrelated to the returns, so a distance
    # derived from it cannot coincide with a correlation distance by accident.
    Zd = abs.(randn(rng, na, 6))
    rd = ReturnsResult(; nx = rd0.nx, X = rd0.X, nf = rd0.nf, F = rd0.F, ts = rd0.ts,
                       nz = ["z$i" for i in 1:6], Z = Zd)
    # The derived carrier: factor loadings, `assets × factors`, computed by the prior.
    fpe = FeaturePrior(; pe = FactorPrior(), ze = RegressionFeatures())
    pr_z = prior(fpe, rd)
    pr_noz = prior(EmpiricalPrior(), rd)
    rd_noz = rd0
    fde = FeatureDistance()
    cde = Distance(; alg = CanonicalDistance())
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = Dict("verbose" => false))

    @testset "The routed Z reaches the kernel, and the clusters differ" begin
        cle_f = ClustersEstimator(; de = fde)
        cle_c = ClustersEstimator(; de = cde)

        # `z_src = :data` is the default: the raw returns result wins.
        clr_d = clusterise(cle_f, rd)
        @test clr_d.D == distance(fde, Zd)
        # And it is genuinely a different clustering from the correlation one, not a
        # relabelling of it — a test that passes on an ignored `Z` is worthless.
        clr_c = clusterise(cle_c, rd)
        @test clr_d.D != clr_c.D
        @test clr_d.res.merges != clr_c.res.merges

        # `z_src = :prior` selects the derived carrier instead, and the two carriers give
        # different answers, so the selector is doing real work.
        clr_p = clusterise(cle_f, pr_z; z_src = :prior)
        @test clr_p.D == distance(fde, pr_z.Z)
        @test clr_p.D != clr_d.D

        # With both carriers populated, `z_src` picks between them rather than falling back.
        @test clusterise(cle_f, pr_z; rd = rd, z_src = :data).D == clr_d.D
        @test clusterise(cle_f, pr_z; rd = rd, z_src = :prior).D == clr_p.D

        # A typo is a throw, not a silent selection of the other carrier.
        @test_throws ArgumentError clusterise(cle_f, pr_z; rd = rd, z_src = :Prior)
        @test_throws ArgumentError clusterise(cle_f, pr_z; rd = rd, z_src = :returns)
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

    @testset "A missing Z throws, and the message names the cause" begin
        cle = ClustersEstimator(; de = fde)

        # 1. No carrier at all: driven straight from a returns matrix.
        e = try
            clusterise(cle, rd0.X)
        catch err
            err
        end
        @test isa(e, PortfolioOptimisers.IsNothingError)
        @test occursin("supplied none", e.msg)
        @test occursin("distance(de, Z", e.msg)

        # 2. A carrier, but neither it nor the returns result holds a feature matrix.
        for (pr, rdx) in ((rd_noz, nothing), (pr_noz, nothing), (pr_noz, rd_noz))
            e = try
                isnothing(rdx) ? clusterise(cle, pr) : clusterise(cle, pr; rd = rdx)
            catch err
                err
            end
            @test isa(e, PortfolioOptimisers.IsNothingError)
            @test occursin("neither the returns result nor the prior result", e.msg)
        end

        # 3. The selector picked the empty carrier while the other one was populated —
        #    both directions, each naming the value to switch to.
        e = try
            clusterise(cle, pr_z; rd = rd_noz, z_src = :data)
        catch err
            err
        end
        @test isa(e, PortfolioOptimisers.IsNothingError)
        @test occursin("z_src = :data", e.msg)
        @test occursin("set `z_src = :prior`", e.msg)

        e = try
            clusterise(cle, pr_noz; rd = rd, z_src = :prior)
        catch err
            err
        end
        @test isa(e, PortfolioOptimisers.IsNothingError)
        @test occursin("z_src = :prior", e.msg)
        @test occursin("set `z_src = :data`", e.msg)

        # A `Z` that is present but unused stays silent, matching `iv`/`ivpa`/`F`/`B`.
        @test clusterise(ClustersEstimator(; de = cde), rd).D ==
              clusterise(ClustersEstimator(; de = cde), rd_noz).D
    end

    @testset "dims is ignored on the routed path" begin
        # The ambient `dims` describes `X`; a carried `Z` is canonically assets-major, so
        # the three-argument methods hardcode `dims = 1`.
        cle = ClustersEstimator(; de = fde)
        @test clusterise(cle, rd; dims = 1).D == distance(fde, Zd; dims = 1)
        Zsq = abs.(randn(rng, na, na))
        rd_sq = ReturnsResult(; nx = rd0.nx, X = rd0.X, ts = rd0.ts,
                              nz = ["z$i" for i in 1:na], Z = Zsq)
        # A square `Z` is the only shape where a transposed read would not throw, so it is
        # the only one that can prove `dims` is not consulted.
        @test clusterise(cle, rd_sq; dims = 2).D == distance(fde, Zsq; dims = 1)
        @test clusterise(cle, rd_sq; dims = 2).D != distance(fde, Zsq; dims = 2)
    end

    @testset "z_src is an optimiser field and drives a full optimisation" begin
        hopt_d = HierarchicalOptimiser(; cle = ClustersEstimator(; de = fde), slv = slv)
        hopt_p = HierarchicalOptimiser(; pe = fpe, cle = ClustersEstimator(; de = fde),
                                       slv = slv, z_src = :prior)
        hopt_c = HierarchicalOptimiser(; cle = ClustersEstimator(; de = cde), slv = slv)
        @test hopt_d.z_src === :data
        @test hopt_p.z_src === :prior
        @test_throws ArgumentError HierarchicalOptimiser(; slv = slv, z_src = :Data)

        for oe in (HierarchicalRiskParity, HierarchicalEqualRiskContribution)
            wd = optimise(oe(; opt = hopt_d), rd)
            wp = optimise(oe(; opt = hopt_p), rd)
            wc = optimise(oe(; opt = hopt_c), rd)
            for res in (wd, wp, wc)
                @test isapprox(sum(res.w), 1)
                @test all(isfinite, res.w)
            end
            # The clusters — hence the weights — actually come from `Z`.
            @test wd.clr.D == distance(fde, Zd)
            @test wp.clr.D == distance(fde, pr_z.Z)
            @test wd.w != wc.w
            @test wp.w != wd.w
        end

        # `SchurComplementHierarchicalRiskParity` forwards the field too.
        ws = optimise(SchurComplementHierarchicalRiskParity(; opt = hopt_d), rd)
        @test ws.clr.D == distance(fde, Zd)

        # `NestedClustered` carries its own `z_src`, not `HierarchicalOptimiser`'s.
        jopt = JuMPOptimiser(; slv = slv)
        @test jopt.z_src === :data
        @test_throws ArgumentError JuMPOptimiser(; slv = slv, z_src = :none)
        nco = NestedClustered(; cle = ClustersEstimator(; de = fde),
                              opti = MeanRisk(; opt = jopt), opto = MeanRisk(; opt = jopt))
        @test nco.z_src === :data
        wn = optimise(nco, rd)
        @test wn.clr.D == distance(fde, Zd)
        @test isapprox(sum(wn.w), 1)

        # `z_src` survives the constructor round trip through `port_opt_view`, which is
        # what keeps it stable across NCO clusters and cross-validation folds.
        i = [1, 3, 5, 7]
        @test PortfolioOptimisers.port_opt_view(hopt_p, i, rd.X).z_src === :prior
        @test PortfolioOptimisers.port_opt_view(nco, i, rd.X).z_src === :data
        @test PortfolioOptimisers.port_opt_view(jopt, i, rd.X).z_src === :data
    end

    @testset "A constraint-generating JuMPOptimiser routes Z too" begin
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
    A feature matrix that holds only features can be swallowed whole. One that holds
    anything else -- an observed mask, a one-hot level -- cannot: a distance measured over
    every column is one the caller did not ask for, and it is finite, symmetric and
    plausible, so nothing downstream reports it. `sel` is the cut, and these tests pin what
    each entry may name and what an entry that resolves against nothing does.
    =#
    @testset "sel cuts the feature axis, and names resolve against nz" begin
        nzd = rd.nz
        D_all = distance(fde, Zd; dims = 1)
        D_cut = distance(fde, Zd[:, 1:2]; dims = 1)

        @testset "construction refuses what cannot be read" begin
            @test isnothing(FeatureDistance().sel)
            @test isnothing(FeatureDistance().sets)
            @test FeatureDistance().strict === false
            # An empty selection is refused rather than read as "every column": `nothing`
            # already says that, and a selection that silently widens to the whole axis is
            # the failure `sel` exists to remove.
            @test_throws PortfolioOptimisers.IsEmptyError FeatureDistance(; sel = String[])
            @test_throws PortfolioOptimisers.IsEmptyError FeatureDistance(; sel = Int[])
            @test_throws ArgumentError FeatureDistance(; sel = ["z1", "z1"])
            @test_throws ArgumentError FeatureDistance(; sel = [1, 1])
            # `sets` is checked at construction rather than bounded by the field's type,
            # because `UniverseSets` is defined in a later file than `FeatureDistance`.
            @test_throws ArgumentError FeatureDistance(; sets = Dict("nx" => ["A"]))
        end

        @testset "a nothing selector is the behaviour every caller already had" begin
            @test distance(FeatureDistance(; sel = nothing), Zd; dims = 1, nz = nzd) ==
                  D_all
            # The passthrough builds no view at all.
            @test PortfolioOptimisers.select_features(fde, Zd, nzd, 1) === Zd
        end

        @testset "names and indices select the same columns" begin
            @test distance(FeatureDistance(; sel = ["z1", "z2"]), Zd; dims = 1, nz = nzd) ==
                  D_cut
            @test distance(FeatureDistance(; sel = [1, 2]), Zd; dims = 1) == D_cut
            @test distance(FeatureDistance(; sel = ["z1", "z2"]), Zd; dims = 1, nz = nzd) !=
                  D_all
            # `cor_and_dist` forwards `nz` through its own kwargs, so both halves agree.
            S, D = cor_and_dist(FeatureDistance(; sel = ["z1", "z2"]), Zd; dims = 1,
                                nz = nzd)
            @test D == D_cut
            @test S == PortfolioOptimisers.distance_to_similarity(fde.sim; D = D_cut)
        end

        @testset "the order of sel is the column order" begin
            k = [4, 1, 6]
            @test distance(FeatureDistance(; sel = k), Zd; dims = 1) ==
                  distance(fde, Zd[:, k]; dims = 1)
            @test distance(FeatureDistance(; sel = nzd[k]), Zd; dims = 1, nz = nzd) ==
                  distance(fde, Zd[:, k]; dims = 1)
        end

        @testset "an integer selector needs no names, and a name refuses without them" begin
            # `LowOrderPrior` carries `Z` without `nz`, so this is the whole of what a
            # selector can do under `z_src = :prior`.
            @test distance(FeatureDistance(; sel = [1, 2]), Zd; dims = 1, nz = nothing) ==
                  D_cut
            @test_throws PortfolioOptimisers.IsNothingError distance(FeatureDistance(;
                                                                                     sel = ["z1"]),
                                                                     Zd; dims = 1,
                                                                     nz = nothing)
        end

        @testset "an unresolvable name warns and drops, or throws under strict" begin
            de_warn = FeatureDistance(; sel = ["z1", "z2", "nope"])
            @test (@test_logs (:warn,) distance(de_warn, Zd; dims = 1, nz = nzd)) == D_cut
            @test_throws ArgumentError distance(FeatureDistance(; sel = ["z1", "nope"],
                                                                strict = true), Zd;
                                                dims = 1, nz = nzd)
            # Every entry dropping leaves nothing to measure. That is not a droppable
            # thing, so it throws whatever `strict` says.
            @test_throws PortfolioOptimisers.IsEmptyError distance(FeatureDistance(;
                                                                                   sel = ["no1",
                                                                                          "no2"]),
                                                                   Zd; dims = 1, nz = nzd)
            @test_throws DomainError distance(FeatureDistance(; sel = [1, 99]), Zd;
                                              dims = 1)
            @test_throws DomainError distance(FeatureDistance(; sel = [0]), Zd; dims = 1)
        end

        @testset "a taxonomy key expands to the block it produced" begin
            tax = UniverseSets(; xkey = "nx",
                               dict = Dict("nx" => ["A", "B", "C"],
                                           "nx_sector" => ["Tech", "Tech", "Fin"],
                                           "nx_country" => ["US", "UK", "UK"]))
            Zt = asset_sets_features(["nx_sector", "nx_country"], tax)
            nzt = asset_sets_feature_names(["nx_sector", "nx_country"], tax)
            # `asset_sets_features` and `asset_sets_feature_names` now read one traversal,
            # `taxonomy_feature_names`, so the block a key selects is the block it built.
            @test nzt ==
                  vcat(PortfolioOptimisers.taxonomy_feature_names(tax, "nx_sector", "test"),
                       PortfolioOptimisers.taxonomy_feature_names(tax, "nx_country",
                                                                  "test"))
            @test distance(FeatureDistance(; sel = ["nx_sector"], sets = tax), Zt; dims = 1,
                           nz = nzt) == distance(fde, Zt[:, 1:2]; dims = 1)
            # A key and a plain column name compose, and a key wins the name lookup.
            @test distance(FeatureDistance(; sel = ["nx_sector", "nx_country=US"],
                                           sets = tax), Zt; dims = 1, nz = nzt) ==
                  distance(fde, Zt[:, 1:3]; dims = 1)
            # A key that resolves but whose columns are not in `nz` is not a typo, so its
            # message says the matrix was built from another taxonomy instead.
            other = UniverseSets(; xkey = "nx",
                                 dict = Dict("nx" => ["A", "B", "C"],
                                             "nx_size" => ["Big", "Small", "Big"]))
            @test_throws ArgumentError distance(FeatureDistance(; sel = ["nx_size"],
                                                                sets = other,
                                                                strict = true), Zt;
                                                dims = 1, nz = nzt)
        end

        @testset "dims names the asset axis, so the feature axis follows it" begin
            @test distance(FeatureDistance(; sel = [1, 2]), permutedims(Zd); dims = 2) ==
                  D_cut
            Z3s = permutedims(cat(Zd, 2 * Zd; dims = 3), (3, 1, 2))
            @test distance(FeatureDistance(; sel = ["z1", "z2"]), Z3s; dims = 1,
                           nz = nzd) == distance(fde, Z3s[:, :, 1:2]; dims = 1)
        end

        @testset "the square case selects reference assets, not assets" begin
            # When the feature axis is the asset axis, a name is an asset used as a
            # reference column. Every row survives, so the matrix stays assets x assets.
            nxs = ["A", "B", "C"]
            Zsq = [1.0 0.2 0.1; 0.2 1.0 0.7; 0.1 0.7 1.0]
            @test PortfolioOptimisers.features_are_assets(nxs, nxs)
            D_sq = distance(FeatureDistance(; sel = ["A", "B"]), Zsq; dims = 1, nz = nxs)
            @test size(D_sq) == (3, 3)
            @test D_sq == distance(fde, Zsq[:, 1:2]; dims = 1)
        end

        @testset "the picker carries nz beside Z, and only the data carrier has it" begin
            Zp, nzp, zdiag = PortfolioOptimisers.feature_matrix_picker(pr_noz, rd, :data)
            @test Zp === rd.Z
            @test nzp === rd.nz
            @test zdiag === :data
            # `LowOrderPrior` holds `Z` without `nz`, so the prior carrier gives none.
            Zq, nzq, _ = PortfolioOptimisers.feature_matrix_picker(pr_z, rd, :prior)
            @test Zq === pr_z.Z
            @test isnothing(nzq)
        end

        @testset "the selector survives the whole routed path" begin
            de_sel = FeatureDistance(; sel = ["z1", "z2"])
            rd_cut = ReturnsResult(; nx = rd.nx, X = rd.X, nz = nzd[1:2], Z = Zd[:, 1:2])
            pm_sel = phylogeny_matrix(NetworkEstimator(; de = de_sel), pr_noz; rd = rd,
                                      z_src = :data)
            pm_ref = phylogeny_matrix(NetworkEstimator(; de = fde), pr_noz; rd = rd_cut,
                                      z_src = :data)
            pm_all = phylogeny_matrix(NetworkEstimator(; de = fde), pr_noz; rd = rd,
                                      z_src = :data)
            @test pm_sel.X == pm_ref.X
            @test pm_sel.X != pm_all.X
            # A name still cannot resolve on the derived carrier, which carries no names.
            @test_throws PortfolioOptimisers.IsNothingError phylogeny_matrix(NetworkEstimator(;
                                                                                              de = de_sel),
                                                                             pr_z; rd = rd,
                                                                             z_src = :prior)
        end

        @testset "factory carries the new fields" begin
            de_sel = FeatureDistance(; sel = ["z1", "z2"], strict = true)
            f = factory(de_sel, pr_noz)
            @test f.sel == de_sel.sel
            @test f.sets === de_sel.sets
            @test f.strict === de_sel.strict
        end
    end
end
