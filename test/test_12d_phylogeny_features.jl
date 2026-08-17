include(joinpath(@__DIR__, "test12_setup.jl"))
using Clustering, StableRNGs, LinearAlgebra

# The two source kinds. Both are *estimators* -- `PhylogenyFeatures` holds no Result -- so
# both refit from whatever `X` they are handed. `PEX` is `NTE`'s graph materialised once, as
# the reference the kernels are checked against.
const NTE = NetworkEstimator(; sep = HopCount(; n = 2))
const CLE = ClustersEstimator()
const PEX = Matrix{Float64}(phylogeny_matrix(NTE, rd.X).X)
const NA = size(rd.X, 2)

@testset "phylogeny_features: the two kernels" begin
    Zb = phylogeny_features(Proximity(; decay = NoDecay()), NTE, rd.X)
    Zg = phylogeny_features(Proximity(), NTE, rd.X)
    Zc = phylogeny_features(Proximity(), CLE, rd.X)

    for Z in (Zb, Zg, Zc)
        @test isa(Z, Matrix{Float64})       # never Int or BitMatrix: the gemm path needs it
        @test size(Z) == (NA, NA)
        @test issymmetric(Z)
        @test all(isfinite, Z)
    end

    # The retirement gate. `BinaryNeighbourhood` *was* `phylogeny_matrix` with the diagonal
    # restored, so checking `NoDecay` against that expression is the retired type's own
    # implementation, not a paraphrase of it -- a flat decay under a budget that still cuts
    # is an indicator, which is exactly what the deleted member produced.
    @test Zb == Matrix{Float64}(phylogeny_matrix(NTE, rd.X).X) + I
    @test all(z -> z in (0.0, 1.0), Zb)
    # And it is the budget doing the cutting, not the decay: no decay emits a zero, yet the
    # matrix is full of them.
    @test any(iszero, Zb)
    @test all(==(1), separation_decay.(Ref(NoDecay()), 0:10, 2))

    # Graded keeps the step count `clamp!(P, 0, 1)` destroys: a direct neighbour scores `n`,
    # the asset itself `n + 1`, anything past `n` hops zero.
    @test all(==(NTE.sep.n + 1), diag(Zg))
    @test sort(unique(Zg)) == [0.0, 1.0, 2.0, 3.0]
    @test Zg != Zb
    # Their supports agree: grading changes the values, not which pairs are within `n` hops.
    @test (Zg .> 0) == (Zb .> 0)

    # A clustering source is a partition: co-membership with the diagonal restored, so the
    # matrix is 0/1 and every row is a cluster indicator.
    @test Zc == Matrix{Float64}(phylogeny_matrix(CLE, rd.X).X) + I
    @test all(z -> z in (0.0, 1.0), Zc)
    @test all(isone, diag(Zc))
    # A partition has no hop structure to decay, so `alg` is inert rather than an error --
    # the same treatment a static feature matrix gets from `FeatureDistance`'s collapse alg.
    @test phylogeny_features(Proximity(; decay = NoDecay()), CLE, rd.X) == Zc
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

    # `+ I` is exactly what every kernel does to the graph it builds; applying it here lets
    # the diagonal argument be made on a hand-built path rather than a fitted universe.
    Dclosed = distance(de, A + I)
    @test Dclosed[1, 2] < Dclosed[1, 3]          # monotone in hop count, as promised
    @test Dclosed[1, 2] == Dclosed[2, 3]
    @test all(>(0), Dclosed[i, k] for i in 1:3, k in 1:3 if i != k)

    # The fold argument: an asset view of a spanning tree routinely isolates the selected
    # vertices from each other. A zero diagonal makes those rows zero rows, and
    # `AngularDist`'s zero-vector convention then declares all of them identical.
    A1 = Matrix{Float64}(phylogeny_matrix(NetworkEstimator(; sep = HopCount(; n = 1)),
                                          rd.X).X)
    j = first(k
              for k in Iterators.product(1:NA, 1:NA, 1:NA)
              if k[1] < k[2] < k[3] && all(iszero, A1[collect(k), collect(k)]))
    sub = A1[collect(j), collect(j)]
    @test all(iszero, distance(de, sub))         # every isolated asset "identical"
    Dsub = distance(de, sub + I)
    @test all(iszero, diag(Dsub))
    @test all(==(0.5), Dsub[i, k] for i in 1:3, k in 1:3 if i != k)
end

@testset "A clustering source is admitted, and what it costs" begin
    # `pl` is bound by `NwE_ClE`: both source kinds, both estimators. A precomputed result
    # of either kind is rejected by the type -- an Estimator does not hold a Result.
    @test isa(PhylogenyFeatures(; pl = CLE), PhylogenyFeatures)
    @test_throws TypeError PhylogenyFeatures(; pl = clusterise(ClustersEstimator(), rd.X))
    @test_throws TypeError PhylogenyFeatures(; pl = PhylogenyResult(; X = PEX))

    # The cost of the partition source, which is why a graph is preferred: `P * transpose(P) - I` has row `i`
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

@testset "PhylogenyFeatures constructs, validates and produces a square matrix" begin
    ze = PhylogenyFeatures()
    @test isa(ze, PortfolioOptimisers.AbstractFeatureMatrixEstimator)
    @test isa(ze.pl, NetworkEstimator)
    @test ze.alg == Proximity()
    @test isa(Proximity(), PortfolioOptimisers.AbstractPhylogenyFeatureAlgorithm)
    @test isa(Proximity(; decay = NoDecay()),
              PortfolioOptimisers.AbstractPhylogenyFeatureAlgorithm)

    # Both sources refit, so the universe always matches by construction -- there is no
    # stored matrix left that could describe a different one.
    for alg in (Proximity(; decay = NoDecay()), Proximity()), pl in (NTE, CLE)
        pr = prior(FeaturePrior(; ze = PhylogenyFeatures(; pl = pl, alg = alg)), rd)
        @test size(pr.Z) == (NA, NA)              # the only producer whose axes coincide
        @test pr.Z == phylogeny_features(alg, pl, rd.X)
    end

    # Squareness is a property of the matrix, not a claim on the carrier: the other producers
    # are rectangular and the carrier says nothing about either case.
    @test size(prior(FeaturePrior(; pe = FactorPrior(), ze = RegressionFeatures()), rd).Z,
               2) != NA
    @test size(prior(FeaturePrior(; ze = rand(StableRNG(987654321), NA, 3)), rd).Z) ==
          (NA, 3)
end

@testset "A subproblem measures its own neighbourhood, by refitting" begin
    i = [1, 4, 7, 11, 15]
    de = FeatureDistance()

    # The carrier slices the asset axis only, square matrix or not: a derived `Z` carries no
    # squareness flag, because there is nothing here a producer could not recompute.
    for pl in (NTE, CLE)
        pr = prior(FeaturePrior(; ze = PhylogenyFeatures(; pl = pl)), rd)
        prv = PortfolioOptimisers.port_opt_view(pr, i)
        @test size(prv.Z) == (length(i), NA)
        @test prv.Z == pr.Z[i, :]
    end

    # The semantics survive the flag, reached by the better route. A subproblem is measured
    # on its own neighbourhood structure — "related to asset k, for k in this subproblem" —
    # because the producer **refits** on the subproblem's universe rather than having a
    # matrix describing a larger one cut down. `distance` therefore still does not commute
    # with the subselection, which was the whole content of the deleted flag.
    pen = FeaturePrior(; ze = PhylogenyFeatures(; pl = NTE))
    prs = prior(pen, rd)
    rdv = ReturnsResult(; nx = rd.nx[i], X = rd.X[:, i])
    Zr = prior(PortfolioOptimisers.port_opt_view(pen, i), rdv).Z
    @test size(Zr) == (length(i), length(i))
    @test distance(de, Zr) != distance(de, prs.Z)[i, i]

    # A rectangular producer is the contrast: its feature axis is not the asset axis, so a
    # view keeps every row's feature vector intact and the distance does commute.
    prf = prior(FeaturePrior(; pe = FactorPrior(), ze = RegressionFeatures()), rd)
    @test distance(de, PortfolioOptimisers.port_opt_view(prf, i).Z) ==
          distance(de, prf.Z)[i, i]
end

@testset "Every producer is configuration, so a view passes it through" begin
    # No producer embeds data any more: `PhylogenyFeatures` holds an estimator, so a view has
    # nothing to slice and the source refits on the viewed returns instead. That is what
    # makes `feature_estimator_view`'s delegation to `port_opt_view` a no-op for every
    # producer in the family.
    i = [2, 5, 9, 13]
    for pl in (NTE, CLE)
        pen = FeaturePrior(; ze = PhylogenyFeatures(; pl = pl))
        @test PortfolioOptimisers.port_opt_view(pen, i).ze.pl === pl
    end
    @test PortfolioOptimisers.port_opt_view(FeaturePrior(; ze = RegressionFeatures()), i).ze ==
          RegressionFeatures()

    # The viewed producer refits on the viewed universe, which is the whole point: the
    # feature matrix a subproblem sees describes the subproblem's assets.
    rdv = ReturnsResult(; nx = rd.nx[i], X = rd.X[:, i])
    pen = FeaturePrior(; ze = PhylogenyFeatures(; pl = NTE))
    Zv = prior(PortfolioOptimisers.port_opt_view(pen, i), rdv).Z
    @test size(Zv) == (length(i), length(i))
    @test Zv == phylogeny_features(Proximity(), NTE, rd.X[:, i])
end

@testset "A square feature matrix drives an optimisation end to end" begin
    mk(ze) = HierarchicalRiskParity(;
                                    opt = HierarchicalOptimiser(;
                                                                pe = FeaturePrior(;
                                                                                  ze = ze),
                                                                cle = ClustersEstimator(;
                                                                                        de = FeatureDistance()),
                                                                z_src = :prior))
    wb = optimise(mk(PhylogenyFeatures(; pl = NTE, alg = Proximity(; decay = NoDecay()))),
                  rd).w
    wg = optimise(mk(PhylogenyFeatures(; pl = NTE, alg = Proximity())), rd).w
    wc = optimise(mk(PhylogenyFeatures(; pl = CLE)), rd).w

    for w in (wb, wg, wc)
        @test length(w) == NA
        @test sum(w) ≈ 1
        @test all(isfinite, w)
    end
    # Graded is not binary in disguise: the retained step count changes the hierarchy.
    @test wb != wg
    # A partition is a different structure again, not a relabelling of either graph variant.
    @test wc != wb
    @test wc != wg

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

    # Both sources are now endogenous -- every square-producing source refits from the
    # returns, so there is no exogenous square route left in the family. A partition source
    # is endogenous too, and coarser: it recodes a clustering of the same returns, so it
    # agrees with the correlation hierarchy at the cut that defined it.
    prc = prior(FeaturePrior(; ze = PhylogenyFeatures(; pl = CLE)), rd)
    hxc = Clustering.hclust(distance(FeatureDistance(), prc.Z); linkage = :ward)
    @test size(prc.Z) == (NA, NA)
    @test length(unique(prc.Z)) == 2
end

@testset "The recursion hazard fails loudly rather than looping" begin
    # A `FeatureDistance` inside the source's own `de` runs inside `prior(pe, X, F; …)`,
    # before `pr.Z` exists, so there is no feature matrix to find and none to recurse into.
    ze = PhylogenyFeatures(; pl = NetworkEstimator(; de = FeatureDistance()))
    @test_throws PortfolioOptimisers.IsNothingError prior(FeaturePrior(; ze = ze), rd)
    res = @test_throws PortfolioOptimisers.IsNothingError phylogeny_features(Proximity(;
                                                                                       decay = NoDecay()),
                                                                             NetworkEstimator(;
                                                                                              de = FeatureDistance()),
                                                                             rd.X)
    @test occursin("FeatureDistance requires a feature matrix", res.value.msg)
end

# The separation family. Like the decay family it lives in `11_Phylogeny/01_Base_Phylogeny.jl`
# for include order -- `NetworkEstimator` carries it as a field -- so its tests live beside
# the consumer that grades what it measures.
@testset "HopCount carries the budget the network estimator used to" begin
    @test isa(HopCount(), PortfolioOptimisers.AbstractSeparationAlgorithm)
    @test HopCount().n == 1
    @test HopCount(; n = 4).n == 4
    # The budget moved off `NetworkEstimator` onto the member that states its unit, so the
    # old spelling is gone rather than deprecated.
    @test NetworkEstimator().sep == HopCount()
    @test !hasproperty(NetworkEstimator(), :n)
    # `n >= 1` came with it: the validation is on the member, not on the estimator.
    @test_throws DomainError HopCount(; n = 0)
    @test_throws DomainError HopCount(; n = -2)
    @test isa(NetworkEstimator(; sep = HopCount(; n = 3)), NetworkEstimator)
    @test_throws TypeError NetworkEstimator(; sep = 3)

    # `separation_matrix` is the hop matrix `phylogeny_features` used to build inline, one
    # `gdistances` per vertex, and the sentinel it reports is passed through unrepaired.
    G = PortfolioOptimisers.Graphs
    g = G.SimpleGraph(PortfolioOptimisers.calc_adjacency(NTE, rd.X; dims = 1))
    d = separation_matrix(NTE.sep, NTE, rd.X; dims = 1)
    @test isa(d, Matrix{Int})
    @test size(d) == (NA, NA)
    @test issymmetric(d)
    @test all(iszero, diag(d))
    @test d == reduce(hcat, G.gdistances(g, v) for v in 1:NA)

    # `separation_budget` is configured rather than observed for a hop count, so neither the
    # estimator nor the separations it produced can move it.
    @test separation_budget(NTE.sep, NTE, d) == NTE.sep.n == 2
    @test separation_budget(HopCount(; n = 7), NTE, d) == 7
    @test separation_budget(HopCount(; n = 7), NTE, zeros(Int, 2, 2)) == 7
end

# The separation decay family. It lives in `11_Phylogeny/01_Base_Phylogeny.jl` for include
# order rather than for ownership -- `Proximity` is its only consumer -- so its
# tests live beside that consumer.
@testset "The decay members honour the contract they are held to" begin
    dks = (LinearDecay(), ExponentialDecay(), ExponentialDecay(; rate = 0.25),
           ExponentialDecay(; rate = 4.0), ReciprocalDecay(),
           ReciprocalDecay(; power = 0.5), ReciprocalDecay(; power = 3.0))
    for dk in dks
        @test isa(dk, PortfolioOptimisers.AbstractSeparationDecayAlgorithm)
        for dmax in (1, 2, 5, 8)
            fs = [separation_decay(dk, d, dmax) for d in 0:dmax]
            @test all(isfinite, fs)
            @test fs[1] > 0                       # `f(0) > 0`
            @test all(<=(fs[1]), fs)              # and maximal
            @test issorted(fs; rev = true)        # monotone non-increasing
            # Truncation is `n`'s job alone, so nothing inside the budget may reach zero:
            # that is what makes a zero in `Z` mean "unreachable" and nothing else.
            @test all(>(0), fs)
        end
    end

    # `d` is a *real* separation, not a hop count. One family serves both variants, so the
    # members must not assume an integer -- and the halfway point must sit between.
    for dk in dks
        @test separation_decay(dk, 0.5, 3) < separation_decay(dk, 0, 3)
        @test separation_decay(dk, 0.5, 3) > separation_decay(dk, 1, 3)
    end

    # `dmax` is the budget in scope and only `LinearDecay` reads it -- the other members are
    # pinned at `f(0) = 1` and ignore it, which is what decouples contrast from reach.
    @test separation_decay(LinearDecay(), 0, 3) == 4
    @test separation_decay(LinearDecay(), 0, 7) == 8
    for dk in (ExponentialDecay(; rate = 2.0), ReciprocalDecay(; power = 2.0))
        @test separation_decay(dk, 0, 3) == 1
        @test separation_decay(dk, 2, 3) == separation_decay(dk, 2, 99)
    end

    # The closed forms, stated rather than paraphrased.
    @test separation_decay(LinearDecay(), 2, 5) == 5 + 1 - 2
    @test separation_decay(ExponentialDecay(; rate = 0.7), 3, 5) == exp(-0.7 * 3)
    @test separation_decay(ReciprocalDecay(; power = 1.5), 3, 5) == inv((1 + 3)^1.5)
    # Rate, not ratio: `ratio^d` is available as `rate = -log(ratio)`.
    @test separation_decay(ExponentialDecay(; rate = -log(0.6)), 4, 9) ≈ 0.6^4
    # The exponential falls off by a constant factor per unit of separation; the reciprocal
    # does not, which is the point of shipping both.
    ed, rd_ = ExponentialDecay(; rate = 0.8), ReciprocalDecay(; power = 0.8)
    es = [separation_decay(ed, d, 5) for d in 0:5]
    rs = [separation_decay(rd_, d, 5) for d in 0:5]
    @test all(≈(exp(-0.8)), es[2:end] ./ es[1:(end - 1)])
    @test !all(≈(rs[2] / rs[1]), rs[2:end] ./ rs[1:(end - 1)])
    @test rs[end] > es[end]                       # heavier tail

    # Field validation, on the members that carry a parameter.
    @test_throws DomainError ExponentialDecay(; rate = 0)
    @test_throws DomainError ExponentialDecay(; rate = -1.0)
    @test_throws DomainError ReciprocalDecay(; power = 0)
    @test_throws DomainError ReciprocalDecay(; power = -0.5)
    @test ExponentialDecay().rate == 1.0
    @test ReciprocalDecay().power == 1.0
end

@testset "NoDecay is the flat end of the family, and does not mean no truncation" begin
    @test isa(NoDecay(), PortfolioOptimisers.AbstractSeparationDecayAlgorithm)
    # Constant, so it is the one member that is not strictly decreasing -- which is why it
    # sits outside the loop above and why the contract asks for monotone *non*-increasing.
    for dmax in (1, 2, 5, 8), d in 0:dmax
        @test separation_decay(NoDecay(), d, dmax) == 1
    end
    @test separation_decay(NoDecay(), 0.5, 3) == separation_decay(NoDecay(), 0, 3)
    @test separation_decay(NoDecay(), 2, 3) == separation_decay(NoDecay(), 2, 99)
    # It satisfies the contract by construction, so it opts out of the probe with the other
    # shipped members -- and would pass it if it did not.
    @test isnothing(PortfolioOptimisers.assert_separation_decay(NoDecay(), 0:3, 3))
    @test isnothing(invoke(PortfolioOptimisers.assert_separation_decay,
                           Tuple{PortfolioOptimisers.AbstractSeparationDecayAlgorithm, Any,
                                 Number}, NoDecay(), 0:3, 3))

    # The name is about the fall-off alone: the budget still cuts, so what comes out is an
    # indicator of the neighbourhood the budget selects and not a matrix of ones.
    for n in (1, 2, 3)
        pl = NetworkEstimator(; sep = HopCount(; n = n))
        Z = phylogeny_features(Proximity(; decay = NoDecay()), pl, rd.X)
        @test sort(unique(Z)) == [0.0, 1.0]
        @test Z != ones(NA, NA)
        @test Z == Matrix{Float64}(phylogeny_matrix(pl, rd.X).X) + I
        # The support is the graded variant's, whatever the fall-off -- the budget is the
        # only thing that selects.
        @test (Z .> 0) == (phylogeny_features(Proximity(), pl, rd.X) .> 0)
    end
end

# A decay defined outside the library. `assert_separation_decay` is opt-out: the fallback on
# the abstract type probes, and only the shipped members turn it off.
struct DecayNoTop <: PortfolioOptimisers.AbstractSeparationDecayAlgorithm end
PortfolioOptimisers.separation_decay(::DecayNoTop, d, dmax) = d + 1
struct DecayZeroAtZero <: PortfolioOptimisers.AbstractSeparationDecayAlgorithm end
PortfolioOptimisers.separation_decay(::DecayZeroAtZero, d, dmax) = -float(d)
struct DecayNonMonotone <: PortfolioOptimisers.AbstractSeparationDecayAlgorithm end
PortfolioOptimisers.separation_decay(::DecayNonMonotone, d, dmax) = d == 1 ? 0.1 : 1.0
struct DecayInfinite <: PortfolioOptimisers.AbstractSeparationDecayAlgorithm end
PortfolioOptimisers.separation_decay(::DecayInfinite, d, dmax) = iszero(d) ? Inf : 1.0
struct DecayFine <: PortfolioOptimisers.AbstractSeparationDecayAlgorithm end
PortfolioOptimisers.separation_decay(::DecayFine, d, dmax) = inv(2.0^d)
# Goes negative at `d = 2`, so a probe over `0:3` walks straight into it.
struct DecayNegativeInLoop <: PortfolioOptimisers.AbstractSeparationDecayAlgorithm end
PortfolioOptimisers.separation_decay(::DecayNegativeInLoop, d, dmax) = 1.0 - d
# Non-negative everywhere `0:1` looks, negative by `d = 3`. Only the endpoint catches it.
struct DecayNegativeAtBudget <: PortfolioOptimisers.AbstractSeparationDecayAlgorithm end
PortfolioOptimisers.separation_decay(::DecayNegativeAtBudget, d, dmax) = 1.0 - d / 2

@testset "The contract is enforced on extensions, not merely documented" begin
    # `Proximity`'s diagonal comes out of the decay itself, so a member that is not
    # maximal at `d = 0` silently produces the structural-equivalence matrix the diagonal
    # exists to prevent -- see the diagonal testset above. Hence a *probing* fallback.
    @test_throws DomainError PortfolioOptimisers.assert_separation_decay(DecayNoTop(), 0:3,
                                                                         3)
    @test_throws DomainError PortfolioOptimisers.assert_separation_decay(DecayZeroAtZero(),
                                                                         0:3, 3)
    @test_throws DomainError PortfolioOptimisers.assert_separation_decay(DecayNonMonotone(),
                                                                         0:3, 3)
    @test_throws DomainError PortfolioOptimisers.assert_separation_decay(DecayInfinite(),
                                                                         0:3, 3)
    @test isnothing(PortfolioOptimisers.assert_separation_decay(DecayFine(), 0:3, 3))
    # The probe does not depend on the caller handing it sorted separations.
    @test_throws DomainError PortfolioOptimisers.assert_separation_decay(DecayNonMonotone(),
                                                                         [3, 0, 2, 1], 3)
    @test isnothing(PortfolioOptimisers.assert_separation_decay(DecayFine(), [3, 0, 2, 1],
                                                                3))
    # `0` is the unreachable sentinel, so a negative score inside the budget would put a
    # *reachable* pair below an *unreachable* one. Caught at a probed `d`...
    @test_throws DomainError PortfolioOptimisers.assert_separation_decay(DecayNegativeInLoop(),
                                                                         0:3, 3)
    # ...and caught at `dmax` when `ds` stops short of it, which is what makes the extra
    # out-of-loop evaluation load-bearing rather than redundant: every probed value here is
    # non-negative, and a weighted separation can never hand the probe an exhaustive `ds`.
    @test all(>=(0),
              PortfolioOptimisers.separation_decay.(Ref(DecayNegativeAtBudget()), 0:1, 3))
    @test_throws DomainError PortfolioOptimisers.assert_separation_decay(DecayNegativeAtBudget(),
                                                                         0:1, 3)
    # Non-negative, *not* strictly positive: a decay bottoming out at zero inside the budget
    # says "no relatedness", which is what an unreachable pair says too.
    @test PortfolioOptimisers.separation_decay(DecayNegativeInLoop(), 1, 1) == 0
    @test isnothing(PortfolioOptimisers.assert_separation_decay(DecayNegativeInLoop(), 0:1,
                                                                1))

    # The shipped members opt out, so the check costs nothing for what ships.
    for dk in (LinearDecay(), ExponentialDecay(), ReciprocalDecay())
        @test isnothing(PortfolioOptimisers.assert_separation_decay(dk, 0:3, 3))
    end
    # And they would pass the probe if they did not: `LinearDecay` is negative above
    # `dmax + 1`, but the clause is scoped to the budget, so nothing there is violated.
    @test PortfolioOptimisers.separation_decay(LinearDecay(), 5, 3) == -1
    for dk in (LinearDecay(), ExponentialDecay(), ReciprocalDecay()), dmax in (1, 3, 8)
        @test isnothing(invoke(PortfolioOptimisers.assert_separation_decay,
                               Tuple{PortfolioOptimisers.AbstractSeparationDecayAlgorithm,
                                     Any, Number}, dk, 0:dmax, dmax))
    end

    # It fires from the kernel too, before the `assets^2` loop rather than inside it.
    @test_throws DomainError phylogeny_features(Proximity(; decay = DecayNoTop()), NTE,
                                                rd.X)
    # And a well-behaved outside member just works -- the family is open.
    Zx = phylogeny_features(Proximity(; decay = DecayFine()), NTE, rd.X)
    @test all(==(1.0), diag(Zx))
    @test sort(unique(Zx)) == [0.0, 0.25, 0.5, 1.0]
end

@testset "Proximity carries the decay, and linear is unchanged" begin
    @test Proximity().decay == LinearDecay()
    @test Proximity(; decay = ExponentialDecay()).decay == ExponentialDecay()
    # The field is bound by type rather than checked at runtime, so a non-decay is refused
    # at construction.
    @test_throws TypeError Proximity(; decay = 2.0)

    # Linear reproduces the hardcoded fall-off `Proximity` shipped before the decay
    # family existed, re-derived here from the graph rather than paraphrased. Verified
    # bit-for-bit against the pre-change output over 17 matrices -- four graph algorithms,
    # budgets 1 to 8, full universe and a slice -- on issue #197.
    G = PortfolioOptimisers.Graphs
    for n in (1, 2, 3, 5)
        pl = NetworkEstimator(; sep = HopCount(; n = n))
        g = G.SimpleGraph(PortfolioOptimisers.calc_adjacency(pl, rd.X; dims = 1))
        ref = zeros(Float64, NA, NA)
        for v in 1:NA
            h = G.gdistances(g, v)
            for u in 1:NA
                ref[u, v] = ifelse(h[u] <= n, Float64(n + 1 - h[u]), 0.0)
            end
        end
        @test phylogeny_features(Proximity(), pl, rd.X) == ref
        @test phylogeny_features(Proximity(; decay = LinearDecay()), pl, rd.X) == ref
    end

    # A different decay changes the values, not which pairs are inside the budget: the
    # support is the binary variant's, whatever the fall-off.
    Zb = phylogeny_features(Proximity(; decay = NoDecay()), NTE, rd.X)
    for dk in
        (LinearDecay(), ExponentialDecay(; rate = 0.5), ExponentialDecay(; rate = 3.0),
         ReciprocalDecay(), ReciprocalDecay(; power = 2.0))
        Z = phylogeny_features(Proximity(; decay = dk), NTE, rd.X)
        @test isa(Z, Matrix{Float64})             # the gemm path still
        @test issymmetric(Z)
        @test (Z .> 0) == (Zb .> 0)
        @test all(==(separation_decay(dk, 0, NTE.sep.n)), diag(Z))
        @test maximum(Z) == separation_decay(dk, 0, NTE.sep.n)
    end

    # The exponential's fall-off, read off the matrix: the score at each hop level is
    # `exp(-rate * hops)`, and the diagonal is 1 rather than `n + 1`.
    Zg = phylogeny_features(Proximity(), NTE, rd.X)
    Ze = phylogeny_features(Proximity(; decay = ExponentialDecay(; rate = 0.9)), NTE, rd.X)
    for h in 0:(NTE.sep.n)
        m = Zg .== NTE.sep.n + 1 - h
        @test any(m)
        @test all(≈(exp(-0.9 * h)), Ze[m])
    end
    @test all(==(1.0), diag(Ze))
    # Sharper fall-off really is sharper: the same pairs, further apart in score.
    Zs = phylogeny_features(Proximity(; decay = ExponentialDecay(; rate = 3.0)), NTE, rd.X)
    far = Zg .== 1.0                              # exactly `n` hops away
    @test all(Zs[far] .< Ze[far])
end

@testset "An unreachable pair scores zero without evaluating the decay" begin
    # Two disjoint edges: 1-2 and 3-4. Every cross-component pair is unreachable, which
    # `gdistances` reports as `typemax(Int)`. Every structure a shipped estimator can build
    # is connected -- a spanning tree or a PMFG -- so a disconnected one is handed to the
    # kernels directly. That is what the graph-taking entry points are for, and it is why no
    # test double subtypes `AbstractNetworkEstimator` to answer `calc_adjacency` any more.
    G = PortfolioOptimisers.Graphs
    A = [0 1 0 0; 1 0 0 0; 0 0 0 1; 0 0 1 0]
    g = PortfolioOptimisers.separation_graph(HopCount(), G.SimpleGraph(A))
    @test G.gdistances(g, 1)[3] == typemax(Int)
    # `separation_matrix` passes the sentinel through unrepaired, which is what makes
    # `is_related` the guard rather than a tidy-up.
    dd = separation_matrix(HopCount(), g)
    @test dd[1, 3] == typemax(Int)
    @test dd[1, 2] == 1
    # The sentinel is not a separation, whatever the budget says. `isfinite` alone would
    # admit it, which is the whole reason the test belongs to the family.
    @test isfinite(dd[1, 3])
    @test !PortfolioOptimisers.is_reachable(HopCount(), dd[1, 3])
    @test !PortfolioOptimisers.is_related(HopCount(), dd[1, 3], typemax(Int))
    @test PortfolioOptimisers.is_related(HopCount(), dd[1, 2], 1)

    # The budget comparison must *short-circuit*, not merely select: `ifelse` evaluates both
    # branches, and `ReciprocalDecay` overflows `1 + d` at `typemax(Int)` -- for a fractional
    # power that is a `DomainError` rather than a discarded number.
    @test_throws DomainError inv((1 + typemax(Int))^0.5)
    dk = ReciprocalDecay(; power = 0.5)
    Zr = PortfolioOptimisers._proximity_features(Proximity(; decay = dk), HopCount(; n = 1),
                                                 dd, 1, Float64)
    @test Zr[1, 3] == 0.0
    @test Zr[1, 2] == separation_decay(dk, 1, 1) == inv(sqrt(2))
    @test all(==(1.0), diag(Zr))
    # And the same for a beyond-budget but reachable pair, which is the other zero.
    Ap = [0 1 0; 1 0 1; 0 1 0]
    dp = separation_matrix(HopCount(),
                           PortfolioOptimisers.separation_graph(HopCount(),
                                                                G.SimpleGraph(Ap)))
    Zp = PortfolioOptimisers._proximity_features(Proximity(; decay = dk), HopCount(; n = 1),
                                                 dp, 1, Float64)
    @test Zp[1, 3] == 0.0
    @test Zp[1, 2] == inv(sqrt(2))
end

# A weighted graph chosen by the test, driving both separations over the same structure --
# which is what lets them be compared at all. The kernels take the structure, so this is two
# helpers rather than an `AbstractNetworkEstimator` that answers internal generics: the
# weighted graph is what a `PathLength` measures over, and `separation_graph` binarises it
# for a `HopCount`. `NetworkEstimator()` stands in `separation_budget`'s inert slot.
const SWG = PortfolioOptimisers.SimpleWeightedGraphs
fixed_graph(::PathLength, W::Matrix{Float64}) = SWG.SimpleWeightedGraph(W)
function fixed_graph(sep::HopCount, W::Matrix{Float64})
    return PortfolioOptimisers.separation_graph(sep, SWG.SimpleWeightedGraph(W))
end
function fixed_features(alg::Proximity, sep, W::Matrix{Float64})
    g = fixed_graph(sep, W)
    d = separation_matrix(sep, g)
    return PortfolioOptimisers._proximity_features(alg, sep, d,
                                                   separation_budget(sep,
                                                                     NetworkEstimator(), d),
                                                   Float64)
end
# 1 - 2 - 3 - 4 with a long shortcut 1 - 3. Hops call `(1, 3)` adjacent and `(2, 4)` two
# apart; path lengths call `(2, 4)` the closer of the two. Constructed rather than found:
# over twenty real assets the two separations strictly invert 0.16% of pairs of pairs on a
# minimum spanning tree and *none at all* on a PMFG, so sampling would prove nothing.
const WX = [0.0 1.0 5.0 0.0; 1.0 0.0 1.0 0.0; 5.0 1.0 0.0 0.1; 0.0 0.0 0.1 0.0]

@testset "PathLength measures the structure the hop count counts" begin
    @test isa(PathLength(), PortfolioOptimisers.AbstractSeparationAlgorithm)
    @test isnothing(PathLength().dmax)
    @test PathLength(; dmax = 0.5).dmax == 0.5
    @test isa(NetworkEstimator(; sep = PathLength()), NetworkEstimator)
    # A budget of zero keeps nothing but the diagonal, so it is refused as `HopCount`'s is.
    @test_throws DomainError PathLength(; dmax = 0)
    @test_throws DomainError PathLength(; dmax = -1.5)
    @test_throws DomainError PathLength(; dmax = NaN)

    nt = NetworkEstimator(; sep = PathLength())
    np = NetworkEstimator(; alg = MaximumDistanceSimilarity(), sep = PathLength())
    dt = separation_matrix(nt.sep, nt, rd.X; dims = 1)
    @test isa(dt, Matrix{Float64})
    @test size(dt) == (NA, NA)
    @test issymmetric(dt)
    @test all(iszero, diag(dt))                   # the diagonal is zero, not the sentinel

    # The oracle for both branches: the path runs over the *distances* on the structure's
    # own edge set. On the tree branch the structure already carries them, so the two graph
    # builders return the same graph.
    G = PortfolioOptimisers.Graphs
    gt = PortfolioOptimisers.calc_distance_weighted_graph(nt, rd.X; dims = 1)
    @test gt == PortfolioOptimisers.calc_weighted_adjacency_graph(nt, rd.X; dims = 1)
    @test dt == G.floyd_warshall_shortest_paths(gt).dists

    # On the PMFG branch the structure is selected by similarity and then re-weighted: same
    # edges, different values. Both halves are checked, because either alone would pass on a
    # graph that had quietly changed shape.
    gp = PortfolioOptimisers.calc_distance_weighted_graph(np, rd.X; dims = 1)
    Ap = PortfolioOptimisers.calc_adjacency(np, rd.X; dims = 1)
    Wp = Matrix(G.adjacency_matrix(gp))
    @test (Wp .!= 0) == (Matrix(Ap) .!= 0)
    _, Dp = PortfolioOptimisers.cor_and_dist(np.de, np.ce, rd.X; dims = 1)
    @test all(Wp[i, k] == Dp[i, k] for i in 1:NA, k in 1:NA if Wp[i, k] != 0)

    # And it is emphatically *not* the PMFG's own weights. A path over similarities
    # minimises total similarity, so it prefers the route through the weakest links -- yet
    # it correlates 0.95 to 0.97 with the right answer, which is why the check is an
    # equality against `D` and not a plausibility test.
    gs = PortfolioOptimisers.calc_weighted_adjacency_graph(np, rd.X; dims = 1)
    Ws = Matrix(G.adjacency_matrix(gs))
    @test maximum(Wp) < minimum(Ws[Ws .!= 0])     # distances below every similarity here
    @test separation_matrix(np.sep, np, rd.X; dims = 1) !=
          G.floyd_warshall_shortest_paths(gs).dists
end

@testset "The two separations order the same graph differently" begin
    dp = separation_matrix(PathLength(), fixed_graph(PathLength(), WX))
    dh = separation_matrix(HopCount(), fixed_graph(HopCount(), WX))
    @test dp == [0.0 1.0 2.0 2.1; 1.0 0.0 1.0 1.1; 2.0 1.0 0.0 0.1; 2.1 1.1 0.1 0.0]
    @test dh == [0 1 1 2; 1 0 1 2; 1 1 0 1; 2 2 1 0]

    # The inversion, stated as the pair-of-pairs comparison it is: hops rank `(1, 3)` closer
    # than `(2, 4)`, path lengths rank them the other way round.
    @test dh[1, 3] < dh[2, 4]
    @test dp[1, 3] > dp[2, 4]
    # The shortcut is on the graph and is not the shortest path: 5.0 direct against 2.0
    # through vertex 2. Weights select the route, they do not merely label it.
    @test WX[1, 3] == 5.0
    @test dp[1, 3] == WX[1, 2] + WX[2, 3]

    # And the inversion survives into `Z`, which is the only place a consumer sees it.
    Zp = fixed_features(Proximity(), PathLength(), WX)
    Zh = fixed_features(Proximity(), HopCount(; n = 2), WX)
    @test Zp[1, 3] < Zp[2, 4]
    @test Zh[1, 3] > Zh[2, 4]
    # The scales are not comparable either: the budgets are in different units, so the
    # diagonal is `delta + 1` on one and `n + 1` on the other.
    @test all(==(2.1 + 1), diag(Zp))
    @test all(==(2 + 1), diag(Zh))
end

@testset "The budget is the observed diameter, and a chosen one is capped by it" begin
    nte = NetworkEstimator()
    d = separation_matrix(PathLength(), fixed_graph(PathLength(), WX))
    @test separation_budget(PathLength(), nte, d) == 2.1 == maximum(d)
    # A number below the diameter is used as given -- that is what buys fold-stability.
    @test separation_budget(PathLength(; dmax = 1.0), nte, d) == 1.0
    # Above it, the clamp bites. It truncates nothing, since no pair sits beyond 2.1; it
    # keeps `LinearDecay`'s scale top off a number the graph never reaches.
    @test separation_budget(PathLength(; dmax = 100.0), nte, d) == 2.1
    @test all(==(2.1 + 1),
              diag(fixed_features(Proximity(), PathLength(; dmax = 100.0), WX)))

    # `HopCount` is handed the same matrix and still pays nothing for it: its budget is
    # configured, so no reduction over `d` happens on that path at all.
    @test separation_budget(HopCount(; n = 3), nte, d) == 3

    # An epsilon-ball: the budget cuts, the decay does not fall off. Newly expressible, and
    # what `NoDecay` was kept for.
    Zb = fixed_features(Proximity(; decay = NoDecay()), PathLength(; dmax = 1.2), WX)
    @test sort(unique(Zb)) == [0.0, 1.0]
    @test Zb == Float64[(d[i, k] <= 1.2) for i in 1:4, k in 1:4]
    @test Zb[1, 3] == 0.0                         # 2.0 away, outside the ball
    @test Zb[2, 4] == 1.0                         # 1.1 away, inside it
end

@testset "An unreachable weighted pair is Inf, and the budget still guards the decay" begin
    # Two components, 1 - 2 and 3 - 4. `floyd_warshall_shortest_paths` reports `typemax(T)`,
    # which on `Float64` weights is `Inf` -- a sentinel the arithmetic tolerates, unlike the
    # hop count's `typemax(Int)`.
    Wd = [0.0 1.0 0.0 0.0; 1.0 0.0 0.0 0.0; 0.0 0.0 0.0 2.0; 0.0 0.0 2.0 0.0]
    nte = NetworkEstimator()
    d = separation_matrix(PathLength(), fixed_graph(PathLength(), Wd))
    @test d[1, 3] == Inf
    @test d[3, 4] == 2.0

    # The sentinel is excluded from the diameter rather than repaired into it. Taking it
    # would make the budget `Inf`, and `LinearDecay` then scores `Inf` at every separation.
    @test separation_budget(PathLength(), nte, d) == 2.0
    @test separation_decay(LinearDecay(), 1.0, Inf) == Inf

    # So the budget comparison is load-bearing under a weighted separation too, for the
    # opposite reason to the hop count's: `LinearDecay` at the sentinel is `-Inf`, not an
    # overflow, and `-Inf` would sort every unreachable pair below every reachable one.
    @test separation_decay(LinearDecay(), Inf, 2.0) == -Inf
    Z = fixed_features(Proximity(), PathLength(), Wd)
    @test Z[1, 3] == 0.0
    @test Z[1, 2] == 2.0 + 1 - 1.0
    @test all(==(3.0), diag(Z))

    # A graph with no edges at all is the degenerate end of the same rule: the diameter is
    # zero, every pair is unreachable, and the honest answer is the identity.
    Wn = zeros(Float64, 4, 4)
    gn = fixed_graph(PathLength(), Wn)
    @test separation_budget(PathLength(), nte, separation_matrix(PathLength(), gn)) == 0.0
    @test fixed_features(Proximity(), PathLength(), Wn) == Matrix{Float64}(I, 4, 4)
end

@testset "PathLength drives the whole feature-prior path" begin
    for (alg, sep) in Iterators.product((Proximity(), Proximity(; decay = NoDecay()),
                                         Proximity(; decay = ExponentialDecay(; rate = 0.8)),
                                         Proximity(; decay = ReciprocalDecay())),
                                        (PathLength(), PathLength(; dmax = 0.5)))
        pl = NetworkEstimator(; sep = sep)
        Z = phylogeny_features(alg, pl, rd.X)
        @test isa(Z, Matrix{Float64})             # the gemm path, as for every other kernel
        @test size(Z) == (NA, NA)
        @test issymmetric(Z)
        @test all(isfinite, Z)
        @test all(>(0), diag(Z))                  # the diagonal is the top of the scale
        @test maximum(Z) == first(diag(Z))
        pr = prior(FeaturePrior(; ze = PhylogenyFeatures(; pl = pl, alg = alg)), rd)
        @test pr.Z == Z
    end

    # `dmax = nothing` reaches the whole connected component, so a flat decay over a graph
    # that is connected by construction is the matrix of ones -- and a chosen budget is what
    # makes it selective again.
    pl = NetworkEstimator(; sep = PathLength())
    @test phylogeny_features(Proximity(; decay = NoDecay()), pl, rd.X) == ones(NA, NA)
    @test any(iszero,
              phylogeny_features(Proximity(; decay = NoDecay()),
                                 NetworkEstimator(; sep = PathLength(; dmax = 0.5)), rd.X))

    # `sep` is the estimator's, not the algorithm's, so a clustering source has no `sep` to
    # be inert -- swapping the separation cannot change a partition's answer because a
    # partition never reaches this kernel's separation branch.
    @test phylogeny_features(Proximity(), CLE, rd.X) ==
          Matrix{Float64}(phylogeny_matrix(CLE, rd.X).X) + I

    # `NoDecay` is the neighbourhood indicator, so it must equal the phylogeny matrix with
    # the diagonal restored -- on *both* separations. #238 established the equality for a
    # hop count; the radius ball now has to satisfy it too, and it is not a coincidence:
    # both sides threshold the same separations at the same budget.
    for alg in (KruskalTree(), MaximumDistanceSimilarity()),
        sep in (HopCount(; n = 2), PathLength(), PathLength(; dmax = 0.5))

        pl = NetworkEstimator(; alg = alg, sep = sep)
        @test phylogeny_features(Proximity(; decay = NoDecay()), pl, rd.X) ==
              Matrix{Float64}(phylogeny_matrix(pl, rd.X).X) + I
    end
end
