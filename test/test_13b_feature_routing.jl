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
end
