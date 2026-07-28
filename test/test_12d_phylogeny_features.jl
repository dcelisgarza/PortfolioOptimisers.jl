include(joinpath(@__DIR__, "test12_setup.jl"))
using Clustering, StableRNGs, LinearAlgebra

# A square graph source built once from the fixture universe. `Pex` stands in for an
# exogenous graph — a supply chain, a shared-ownership network — and is only *derived* here
# so the two source kinds can be checked against each other on identical input.
const NTE = NetworkEstimator(; n = 2)
const PEX = Matrix{Float64}(phylogeny_matrix(NTE, rd.X).X)
const NA = size(rd.X, 2)

@testset "phylogeny_features: the three kernels" begin
    Zb = phylogeny_features(BinaryNeighbourhood(), NTE, rd.X)
    Zg = phylogeny_features(GradedNeighbourhood(), NTE, rd.X)
    Zp = phylogeny_features(GradedNeighbourhood(), PhylogenyResult(; X = PEX))

    for Z in (Zb, Zg, Zp)
        @test isa(Z, Matrix{Float64})       # never Int or BitMatrix: the gemm path needs it
        @test size(Z) == (NA, NA)
        @test issymmetric(Z)
        @test all(isfinite, Z)
    end

    # Binary is exactly `phylogeny_matrix` with the diagonal restored.
    @test Zb == Matrix{Float64}(phylogeny_matrix(NTE, rd.X).X) + I
    @test all(z -> z in (0.0, 1.0), Zb)

    # Graded keeps the step count `clamp!(P, 0, 1)` destroys: a direct neighbour scores `n`,
    # the asset itself `n + 1`, anything past `n` hops zero.
    @test all(==(NTE.n + 1), diag(Zg))
    @test sort(unique(Zg)) == [0.0, 1.0, 2.0, 3.0]
    @test Zg != Zb
    # Their supports agree: grading changes the values, not which pairs are within `n` hops.
    @test (Zg .> 0) == (Zb .> 0)

    # A precomputed result is used as given, only the diagonal is set. `PEX` is this graph,
    # so the exogenous path reproduces the binary one exactly.
    @test Zp == PEX + I
    @test Zp == Zb
    # `alg` is inert on that path.
    @test phylogeny_features(BinaryNeighbourhood(), PhylogenyResult(; X = PEX)) == Zp

    # Weights survive rather than being binarised.
    W = [0.0 0.4 0.0; 0.4 0.0 2.5; 0.0 2.5 0.0]
    Zw = phylogeny_features(GradedNeighbourhood(), PhylogenyResult(; X = W))
    @test Zw == W + 2.5 * I
    @test sort(unique(Zw)) == [0.0, 0.4, 2.5]

    # An edgeless matrix falls back to the identity rather than to a degenerate all-zero
    # feature matrix: with no edges, no two assets share anything.
    @test phylogeny_features(GradedNeighbourhood(), PhylogenyResult(; X = zeros(4, 4))) ==
          Matrix(1.0I, 4, 4)
end

@testset "The diagonal picks between two different algorithms" begin
    # Three-node path 1 - 2 - 3. With a zero diagonal the two *non-adjacent* endpoints come
    # out identical and the adjacent pairs maximally far: that is structural equivalence,
    # the opposite of the proximity the type promises.
    A = Float64[0 1 0; 1 0 1; 0 1 0]
    de = FeatureDistance()
    Dopen = distance(de, A)
    @test Dopen[1, 3] == 0.0                     # two hops apart, yet identical
    @test Dopen[1, 2] == 0.5                     # adjacent, yet maximally distant
    @test Dopen[1, 3] < Dopen[1, 2]

    Dclosed = distance(de,
                       phylogeny_features(BinaryNeighbourhood(), PhylogenyResult(; X = A)))
    @test Dclosed[1, 2] < Dclosed[1, 3]          # monotone in hop count, as promised
    @test Dclosed[1, 2] == Dclosed[2, 3]
    @test all(>(0), Dclosed[i, k] for i in 1:3, k in 1:3 if i != k)

    # The fold argument: an asset view of a spanning tree routinely isolates the selected
    # vertices from each other. A zero diagonal makes those rows zero rows, and
    # `AngularDist`'s zero-vector convention then declares all of them identical.
    A1 = Matrix{Float64}(phylogeny_matrix(NetworkEstimator(; n = 1), rd.X).X)
    j = first(k
              for k in Iterators.product(1:NA, 1:NA, 1:NA)
              if k[1] < k[2] < k[3] && all(iszero, A1[collect(k), collect(k)]))
    sub = A1[collect(j), collect(j)]
    @test all(iszero, distance(de, sub))         # every isolated asset "identical"
    Dsub = distance(de,
                    phylogeny_features(GradedNeighbourhood(), PhylogenyResult(; X = sub)))
    @test all(iszero, diag(Dsub))
    @test all(==(0.5), Dsub[i, k] for i in 1:3, k in 1:3 if i != k)
end

@testset "A clustering source is rejected, and why" begin
    # `pl` is bound by `NwE_PlM`, so a partition never reaches the producer.
    @test_throws TypeError PhylogenyFeatures(; pl = ClustersEstimator())
    @test_throws TypeError PhylogenyFeatures(; pl = clusterise(ClustersEstimator(), rd.X))

    # The degeneracy that bound is protecting against: `P * transpose(P) - I` has row `i`
    # equal to the co-membership indicator of asset `i`, so the distance depends on nothing
    # but cluster size — and a size-two cluster's *within*-cluster distance equals its
    # across-cluster distance, because `- I` leaves each row a lone 1 pointing at the other
    # member, making the two rows orthogonal.
    asg = [1, 1, 1, 2, 2, 3, 3]
    P = Float64[(i != k && asg[i] == asg[k]) for i in eachindex(asg), k in eachindex(asg)]
    D = distance(FeatureDistance(), P)
    @test length(unique(round.(D; digits = 10))) == 3
    @test D[4, 5] == D[1, 4]                     # same cluster, yet as far as a different one
    @test D[4, 5] == maximum(D)
end

@testset "PhylogenyFeatures constructs, validates and is the only z_sq = true producer" begin
    ze = PhylogenyFeatures()
    @test isa(ze, PortfolioOptimisers.AbstractFeatureMatrixEstimator)
    @test isa(ze.pl, NetworkEstimator)
    @test ze.alg == GradedNeighbourhood()
    @test isa(GradedNeighbourhood(), PortfolioOptimisers.AbstractPhylogenyFeatureAlgorithm)
    @test isa(BinaryNeighbourhood(), PortfolioOptimisers.AbstractPhylogenyFeatureAlgorithm)

    # A non-square precomputed matrix cannot even be wrapped in a `PhylogenyResult`, so the
    # producer's own squareness check guards the path that can: a square matrix over the
    # wrong universe.
    @test_throws DimensionMismatch prior(FeaturePrior(;
                                                      ze = PhylogenyFeatures(;
                                                                             pl = PhylogenyResult(;
                                                                                                  X = PEX[1:5,
                                                                                                          1:5]))),
                                         rd)

    for alg in (BinaryNeighbourhood(), GradedNeighbourhood()),
        pl in (NTE, PhylogenyResult(; X = PEX))

        pr = prior(FeaturePrior(; ze = PhylogenyFeatures(; pl = pl, alg = alg)), rd)
        @test pr.z_sq                            # the only producer that says true
        @test size(pr.Z) == (NA, NA)
        @test pr.Z == phylogeny_features(alg, pl, rd.X)
    end

    # `RegressionFeatures` and a literal still say false, so `z_sq` is a real statement.
    @test !prior(FeaturePrior(; pe = FactorPrior(), ze = RegressionFeatures()), rd).z_sq
    @test !prior(FeaturePrior(; ze = rand(StableRNG(987654321), NA, 3)), rd).z_sq
end

@testset "z_sq = true slices both axes under an asset view" begin
    i = [1, 4, 7, 11, 15]
    de = FeatureDistance()
    for pl in (NTE, PhylogenyResult(; X = PEX))
        pr = prior(FeaturePrior(; ze = PhylogenyFeatures(; pl = pl)), rd)
        prv = PortfolioOptimisers.port_opt_view(pr, i)
        @test size(prv.Z) == (length(i), length(i))
        @test prv.Z == pr.Z[i, i]
        @test prv.z_sq
    end

    # `distance` does **not** commute with the view here, and that is the whole meaning of
    # `z_sq = true`. Slicing the feature axis truncates every row's feature vector, so a
    # subproblem is measured on its own neighbourhood structure — "related to asset k, for k
    # in this subproblem" — rather than on the full universe's.
    prs = prior(FeaturePrior(; ze = PhylogenyFeatures(; pl = NTE)), rd)
    @test distance(de, PortfolioOptimisers.port_opt_view(prs, i).Z) !=
          distance(de, prs.Z)[i, i]

    # A `z_sq = false` carrier is the contrast: its view slices rows only, every row's
    # feature vector survives intact, and the distance therefore does commute.
    prf = prior(FeaturePrior(; pe = FactorPrior(), ze = RegressionFeatures()), rd)
    @test !prf.z_sq
    @test distance(de, PortfolioOptimisers.port_opt_view(prf, i).Z) ==
          distance(de, prf.Z)[i, i]
end

@testset "A producer embedding data is sliced, not passed through" begin
    # `feature_estimator_view` delegates to `port_opt_view`, so a producer holding a
    # precomputed matrix indexed by the full universe slices it. Returning the estimator
    # unchanged would leave a full-universe matrix in a subproblem.
    i = [2, 5, 9, 13]
    pe = FeaturePrior(; ze = PhylogenyFeatures(; pl = PhylogenyResult(; X = PEX)))
    pev = PortfolioOptimisers.port_opt_view(pe, i)
    @test size(pev.ze.pl.X) == (length(i), length(i))
    @test pev.ze.pl.X == PEX[i, i]
    @test pev.ze.alg == pe.ze.alg

    rdv = ReturnsResult(; nx = rd.nx[i], X = rd.X[:, i])
    @test prior(pev, rdv).Z == prior(pe, rd).Z[i, i]

    # A network source is configuration and refits on the viewed returns instead.
    pen = FeaturePrior(; ze = PhylogenyFeatures(; pl = NTE))
    @test PortfolioOptimisers.port_opt_view(pen, i).ze.pl === NTE
    # A producer with nothing to slice is still passed through unchanged.
    @test PortfolioOptimisers.port_opt_view(FeaturePrior(; ze = RegressionFeatures()), i).ze ==
          RegressionFeatures()
end

@testset "port_opt_view on a PhylogenyResult" begin
    # Previously a `MethodError`: the universal fallback reaches `nothing_scalar_array_view`,
    # which has no method for a result type.
    i = [1, 3, 6]
    plr = PortfolioOptimisers.port_opt_view(PhylogenyResult(; X = PEX), i)
    @test isa(plr, PhylogenyResult)
    @test plr.X == PEX[i, i]
    @test issymmetric(plr.X) && all(iszero, diag(plr.X))

    v = PortfolioOptimisers.port_opt_view(PhylogenyResult(; X = collect(1.0:10.0)), i)
    @test v.X == [1.0, 3.0, 6.0]
end

@testset "A square feature matrix drives an optimisation end to end" begin
    mk(ze) = HierarchicalRiskParity(;
                                    opt = HierarchicalOptimiser(;
                                                                pe = FeaturePrior(;
                                                                                  ze = ze),
                                                                cle = ClustersEstimator(;
                                                                                        de = FeatureDistance()),
                                                                z_src = :prior))
    wb = optimise(mk(PhylogenyFeatures(; pl = NTE, alg = BinaryNeighbourhood())), rd).w
    wg = optimise(mk(PhylogenyFeatures(; pl = NTE, alg = GradedNeighbourhood())), rd).w
    we = optimise(mk(PhylogenyFeatures(; pl = PhylogenyResult(; X = PEX))), rd).w

    for w in (wb, wg, we)
        @test length(w) == NA
        @test sum(w) ≈ 1
        @test all(isfinite, w)
    end
    # Graded is not binary in disguise: the retained step count changes the hierarchy.
    @test wb != wg
    # `PEX` is this network's own phylogeny matrix, so the exogenous path reproduces it.
    @test we ≈ wb

    # How far neighbourhood overlap departs from the returns correlation depends on which
    # source it came from, and the two cases are genuinely different.
    Dr = distance(Distance(; alg = CanonicalDistance()), PortfolioOptimisersCovariance(),
                  rd.X)
    hc = Clustering.hclust(Dr; linkage = :ward)

    # Endogenous: the network is *filtered from* that same correlation, so topology is not
    # independent of it. The merge order and the weights differ, but the coarse cuts do not
    # — they agree at k = 2 and k = 3 and only diverge from k = 4 on. Recorded rather than
    # asserted away: it is what an endogenous source buys, and what it does not.
    pr = prior(FeaturePrior(; ze = PhylogenyFeatures(; pl = NTE)), rd)
    hf = Clustering.hclust(distance(FeatureDistance(), pr.Z); linkage = :ward)
    @test hf.merges != hc.merges
    @test Clustering.cutree(hf; k = 3) == Clustering.cutree(hc; k = 3)
    @test Clustering.cutree(hf; k = 4) != Clustering.cutree(hc; k = 4)

    # Exogenous: a graph that never passed through the returns. Two disjoint cliques over an
    # asset ordering unrelated to any correlation structure, which is the case the whole
    # `PhylogenyResult` path exists for. It disagrees with the correlation at every cut.
    grp = [isodd(k) for k in 1:NA]
    Pxo = Float64[(i != k && grp[i] == grp[k]) for i in 1:NA, k in 1:NA]
    prx = prior(FeaturePrior(; ze = PhylogenyFeatures(; pl = PhylogenyResult(; X = Pxo))),
                rd)
    hx = Clustering.hclust(distance(FeatureDistance(), prx.Z); linkage = :ward)
    @test Clustering.cutree(hx; k = 2) != Clustering.cutree(hc; k = 2)
    @test Clustering.cutree(hx; k = 2) == (grp .+ 1) ||
          Clustering.cutree(hx; k = 2) == (2 .- grp)
end

@testset "The recursion hazard fails loudly rather than looping" begin
    # A `FeatureDistance` inside the source's own `de` runs inside `prior(pe, X, F; …)`,
    # before `pr.Z` exists, so there is no feature matrix to find and none to recurse into.
    ze = PhylogenyFeatures(; pl = NetworkEstimator(; de = FeatureDistance()))
    @test_throws PortfolioOptimisers.IsNothingError prior(FeaturePrior(; ze = ze), rd)
    res = @test_throws PortfolioOptimisers.IsNothingError phylogeny_features(BinaryNeighbourhood(),
                                                                             NetworkEstimator(;
                                                                                              de = FeatureDistance()),
                                                                             rd.X)
    @test occursin("FeatureDistance requires a feature matrix", res.value.msg)
end
