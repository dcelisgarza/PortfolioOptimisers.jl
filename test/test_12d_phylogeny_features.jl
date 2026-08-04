include(joinpath(@__DIR__, "test12_setup.jl"))
using Clustering, StableRNGs, LinearAlgebra

# The two source kinds. Both are *estimators* -- `PhylogenyFeatures` holds no Result -- so
# both refit from whatever `X` they are handed. `PEX` is `NTE`'s graph materialised once, as
# the reference the kernels are checked against.
const NTE = NetworkEstimator(; n = 2)
const CLE = ClustersEstimator()
const PEX = Matrix{Float64}(phylogeny_matrix(NTE, rd.X).X)
const NA = size(rd.X, 2)

@testset "phylogeny_features: the three kernels" begin
    Zb = phylogeny_features(BinaryNeighbourhood(), NTE, rd.X)
    Zg = phylogeny_features(GradedNeighbourhood(), NTE, rd.X)
    Zc = phylogeny_features(GradedNeighbourhood(), CLE, rd.X)

    for Z in (Zb, Zg, Zc)
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

    # A clustering source is a partition: co-membership with the diagonal restored, so the
    # matrix is 0/1 and every row is a cluster indicator.
    @test Zc == Matrix{Float64}(phylogeny_matrix(CLE, rd.X).X) + I
    @test all(z -> z in (0.0, 1.0), Zc)
    @test all(isone, diag(Zc))
    # A partition has no hop structure to decay, so `alg` is inert rather than an error --
    # the same treatment a static feature matrix gets from `FeatureDistance`'s collapse alg.
    @test phylogeny_features(BinaryNeighbourhood(), CLE, rd.X) == Zc
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
    A1 = Matrix{Float64}(phylogeny_matrix(NetworkEstimator(; n = 1), rd.X).X)
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
    @test ze.alg == GradedNeighbourhood()
    @test isa(GradedNeighbourhood(), PortfolioOptimisers.AbstractPhylogenyFeatureAlgorithm)
    @test isa(BinaryNeighbourhood(), PortfolioOptimisers.AbstractPhylogenyFeatureAlgorithm)

    # Both sources refit, so the universe always matches by construction -- there is no
    # stored matrix left that could describe a different one.
    for alg in (BinaryNeighbourhood(), GradedNeighbourhood()), pl in (NTE, CLE)
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
    @test Zv == phylogeny_features(GradedNeighbourhood(), NTE, rd.X[:, i])
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
    res = @test_throws PortfolioOptimisers.IsNothingError phylogeny_features(BinaryNeighbourhood(),
                                                                             NetworkEstimator(;
                                                                                              de = FeatureDistance()),
                                                                             rd.X)
    @test occursin("FeatureDistance requires a feature matrix", res.value.msg)
end

# The separation decay family. It lives in `11_Phylogeny/01_Base_Phylogeny.jl` for include
# order rather than for ownership -- `GradedNeighbourhood` is its only consumer -- so its
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
    # `GradedNeighbourhood`'s diagonal comes out of the decay itself, so a member that is not
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
    @test_throws DomainError phylogeny_features(GradedNeighbourhood(; decay = DecayNoTop()),
                                                NTE, rd.X)
    # And a well-behaved outside member just works -- the family is open.
    Zx = phylogeny_features(GradedNeighbourhood(; decay = DecayFine()), NTE, rd.X)
    @test all(==(1.0), diag(Zx))
    @test sort(unique(Zx)) == [0.0, 0.25, 0.5, 1.0]
end

@testset "GradedNeighbourhood carries the decay, and linear is unchanged" begin
    @test GradedNeighbourhood().decay == LinearDecay()
    @test GradedNeighbourhood(; decay = ExponentialDecay()).decay == ExponentialDecay()
    # The field is bound by type rather than checked at runtime, so a non-decay is refused
    # at construction.
    @test_throws TypeError GradedNeighbourhood(; decay = 2.0)

    # Linear reproduces the hardcoded fall-off `GradedNeighbourhood` shipped before the decay
    # family existed, re-derived here from the graph rather than paraphrased. Verified
    # bit-for-bit against the pre-change output over 17 matrices -- four graph algorithms,
    # budgets 1 to 8, full universe and a slice -- on issue #197.
    G = PortfolioOptimisers.Graphs
    for n in (1, 2, 3, 5)
        pl = NetworkEstimator(; n = n)
        g = G.SimpleGraph(PortfolioOptimisers.calc_adjacency(pl, rd.X; dims = 1))
        ref = zeros(Float64, NA, NA)
        for v in 1:NA
            h = G.gdistances(g, v)
            for u in 1:NA
                ref[u, v] = ifelse(h[u] <= n, Float64(n + 1 - h[u]), 0.0)
            end
        end
        @test phylogeny_features(GradedNeighbourhood(), pl, rd.X) == ref
        @test phylogeny_features(GradedNeighbourhood(; decay = LinearDecay()), pl, rd.X) ==
              ref
    end

    # A different decay changes the values, not which pairs are inside the budget: the
    # support is the binary variant's, whatever the fall-off.
    Zb = phylogeny_features(BinaryNeighbourhood(), NTE, rd.X)
    for dk in
        (LinearDecay(), ExponentialDecay(; rate = 0.5), ExponentialDecay(; rate = 3.0),
         ReciprocalDecay(), ReciprocalDecay(; power = 2.0))
        Z = phylogeny_features(GradedNeighbourhood(; decay = dk), NTE, rd.X)
        @test isa(Z, Matrix{Float64})             # the gemm path still
        @test issymmetric(Z)
        @test (Z .> 0) == (Zb .> 0)
        @test all(==(separation_decay(dk, 0, NTE.n)), diag(Z))
        @test maximum(Z) == separation_decay(dk, 0, NTE.n)
    end

    # The exponential's fall-off, read off the matrix: the score at each hop level is
    # `exp(-rate * hops)`, and the diagonal is 1 rather than `n + 1`.
    Zg = phylogeny_features(GradedNeighbourhood(), NTE, rd.X)
    Ze = phylogeny_features(GradedNeighbourhood(; decay = ExponentialDecay(; rate = 0.9)),
                            NTE, rd.X)
    for h in 0:(NTE.n)
        m = Zg .== NTE.n + 1 - h
        @test any(m)
        @test all(≈(exp(-0.9 * h)), Ze[m])
    end
    @test all(==(1.0), diag(Ze))
    # Sharper fall-off really is sharper: the same pairs, further apart in score.
    Zs = phylogeny_features(GradedNeighbourhood(; decay = ExponentialDecay(; rate = 3.0)),
                            NTE, rd.X)
    far = Zg .== 1.0                              # exactly `n` hops away
    @test all(Zs[far] .< Ze[far])
end

# A network estimator handing the kernel an adjacency matrix chosen by the test. Every
# estimator `calc_adjacency` can build is connected -- a spanning tree or a PMFG -- so this
# is the only way to drive the unreachable branch from the public kernel.
struct FixedAdjacency <: PortfolioOptimisers.AbstractNetworkEstimator
    A::Matrix{Int}
    n::Int
end
function PortfolioOptimisers.calc_adjacency(pl::FixedAdjacency, X; kwargs...)
    return pl.A
end

@testset "An unreachable pair scores zero without evaluating the decay" begin
    # Two disjoint edges: 1-2 and 3-4. Every cross-component pair is unreachable, which
    # `gdistances` reports as `typemax(Int)`.
    A = [0 1 0 0; 1 0 0 0; 0 0 0 1; 0 0 1 0]
    X = zeros(Float64, 3, 4)
    g = PortfolioOptimisers.Graphs.SimpleGraph(A)
    @test PortfolioOptimisers.Graphs.gdistances(g, 1)[3] == typemax(Int)

    # The budget comparison must *short-circuit*, not merely select: `ifelse` evaluates both
    # branches, and `ReciprocalDecay` overflows `1 + d` at `typemax(Int)` -- for a fractional
    # power that is a `DomainError` rather than a discarded number.
    @test_throws DomainError inv((1 + typemax(Int))^0.5)
    dk = ReciprocalDecay(; power = 0.5)
    Zr = phylogeny_features(GradedNeighbourhood(; decay = dk), FixedAdjacency(A, 1), X)
    @test Zr[1, 3] == 0.0
    @test Zr[1, 2] == separation_decay(dk, 1, 1) == inv(sqrt(2))
    @test all(==(1.0), diag(Zr))
    # And the same for a beyond-budget but reachable pair, which is the other zero.
    Ap = [0 1 0; 1 0 1; 0 1 0]
    Zp = phylogeny_features(GradedNeighbourhood(; decay = dk), FixedAdjacency(Ap, 1),
                            zeros(Float64, 3, 3))
    @test Zp[1, 3] == 0.0
    @test Zp[1, 2] == inv(sqrt(2))
end
