#=
The peel family: the rule that removes the assets of an undetermined pair at admission, so that an
available-case covariance carries no `NaN` inside its finite block. Decided on #1416 (row R86),
built by #1511.

An undetermined pair is a pair of admitted assets that each have a variance but share too few
observations for a covariance. The pairs make a graph, and a set of assets removes every pair
exactly when it is a vertex cover of that graph. `GreedyPeel` is the oracle's rule, max degree
first and the smallest index on a tie. `MinimalPeel` is the minimum cover, and it keeps the
greedy cover whenever that cover is minimum. `NoPeel` keeps the `NaN` for the matrix repair to
refuse.
=#
include(joinpath(@__DIR__, "parity_harness.jl"))

# A graph from a list of edges.
function peel_graph(n, edges)
    U = falses(n, n)
    for (i, j) in edges
        U[i, j] = U[j, i] = true
    end
    return U
end
function peel_is_cover(U, C)
    return all(!U[i, j] || i in C || j in C for i in axes(U, 1), j in axes(U, 2))
end
# The minimum vertex cover by enumeration, and the lexicographically smallest one of that size.
function peel_brute(U)
    n = size(U, 1)
    for k in 0:n
        cs = [findall(b -> (m >> (b - 1)) & 1 == 1, 1:n)
              for m in 0:(2 ^ n - 1) if count_ones(m) == k]
        cs = filter(c -> peel_is_cover(U, c), cs)
        if !isempty(cs)
            return minimum(cs)
        end
    end
end
# The returns of the parity cases. The values come from closed formulas and the gaps from fixed
# windows, so the generator of the stored oracle output and this test build the same matrix, and
# the stored `Input` pins it.
#
# `Counterexample`: five blocks of eight rows. X (asset 2) is valid in block 0 alone, A1 to A3
# (assets 3 to 5) skip block 0 and their own block, and L1 to L3 (assets 6 to 8) are valid in block
# 0 and their own block. The graph is X-(A1, A2, A3) and Ai-Li, and assets 1 and 9 are complete.
# `Windows`: each asset is valid on a window of rows, so a pair is undetermined when its two
# windows share fewer than two rows.
function peel_parity_returns(case::Symbol)
    if case == :Counterexample
        T, N = 40, 9
        blocks = Dict(2 => [0], 3 => [2, 3, 4], 4 => [1, 3, 4], 5 => [1, 2, 4], 6 => [0, 1],
                      7 => [0, 2], 8 => [0, 3])
        keep = [haskey(blocks, j) ? fld(t - 1, 8) in blocks[j] : true
                for t in 1:T, j in 1:N]
    else
        T, N = 30, 10
        windows = [1:30, 1:8, 6:14, 13:22, 20:30, 1:12, 9:20, 18:30, 1:30, 14:15]
        keep = [t in windows[j] for t in 1:T, j in 1:N]
    end
    X = [(sin(0.7 * t + 1.3 * j) * (1 + j / 10) + cos(0.3 * t * j) / 2) / 100
         for t in 1:T, j in 1:N]
    X[.!keep] .= NaN
    return X
end
function peel_covariance(peel; kwargs...)
    return Covariance(; cvg = CoveragePolicy(; peel = peel), kwargs...)
end
# Assets 3 and 4 have disjoint histories, and every other pair shares rows.
function peel_disjoint_returns()
    X = [(sin(0.9 * t + 2.1 * j) + cos(0.2 * t * j) / 3) / 100 for t in 1:40, j in 1:4]
    X[21:end, 3] .= NaN
    X[1:20, 4] .= NaN
    return X
end
struct PeelEveryEnd <: PortfolioOptimisers.AbstractPeel end
function PortfolioOptimisers.peel_assets(::PeelEveryEnd, U::AbstractMatrix{Bool})
    return findall(vec(any(U; dims = 2)))
end
@testset "Peel family" begin
    @testset "Graph rules" begin
        # The star X-(A1, A2, A3) whose leaves each carry one more edge Ai-Li. Greedy removes X
        # first, then one end of each pendant edge: four assets. Three suffice.
        U = peel_graph(7, ((1, 2), (1, 3), (1, 4), (2, 5), (3, 6), (4, 7)))
        @test PortfolioOptimisers.peel_assets(GreedyPeel(), U) == [1, 2, 3, 4]
        @test PortfolioOptimisers.peel_assets(MinimalPeel(), U) == [2, 3, 4]
        @test isempty(PortfolioOptimisers.peel_assets(NoPeel(), U))
        # A path: the greedy cover is minimum, so the minimal rule keeps it.
        P = peel_graph(5, ((1, 2), (2, 3), (3, 4), (4, 5)))
        @test PortfolioOptimisers.peel_assets(GreedyPeel(), P) == [2, 4]
        @test PortfolioOptimisers.peel_assets(MinimalPeel(), P) == [2, 4]
        # No edge, nothing to remove.
        @test isempty(PortfolioOptimisers.peel_assets(MinimalPeel(), falses(3, 3)))
        @test isempty(PortfolioOptimisers.peel_assets(GreedyPeel(), falses(3, 3)))
        # A cycle of five needs three, a cycle of four needs two, and they share no vertex.
        C = peel_graph(9,
                       ((1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (6, 7), (7, 8), (8, 9),
                        (9, 6)))
        @test PortfolioOptimisers.peel_cycle_cover(C, trues(9)) == 5
        @test PortfolioOptimisers.peel_cover_within(C, trues(9), 5)
        @test !PortfolioOptimisers.peel_cover_within(C, trues(9), 4)
        @test !PortfolioOptimisers.peel_cover_within(C, trues(9), -1)
        @test PortfolioOptimisers.peel_max_degree(C, falses(9)) == (0, 0)
        # Against enumeration: the minimal rule is a minimum cover, it is the greedy cover when
        # that cover is minimum, and it is the lexicographically smallest minimum cover otherwise.
        rng = StableRNG(1511)
        larger = 0
        for _ in 1:600
            n = rand(rng, 1:10)
            p = rand(rng) * 0.6
            G = falses(n, n)
            for i in 1:n, j in (i + 1):n
                if rand(rng) < p
                    G[i, j] = G[j, i] = true
                end
            end
            g = PortfolioOptimisers.peel_assets(GreedyPeel(), G)
            m = PortfolioOptimisers.peel_assets(MinimalPeel(), G)
            b = peel_brute(G)
            @test peel_is_cover(G, g)
            if length(g) == length(b)
                @test m == g
            else
                larger += 1
                @test m == b
            end
        end
        # The sample must reach the lexicographic arm, or the last test above checks nothing.
        @test larger > 0
    end
    @testset "Policy and interface" begin
        @test CoveragePolicy().peel === MinimalPeel()
        @test CoveragePolicy(; peel = GreedyPeel()).peel === GreedyPeel()
        @test all(T -> T <: PortfolioOptimisers.AbstractPeel,
                  (MinimalPeel, GreedyPeel, NoPeel))
        @test all(s -> Base.isexported(PortfolioOptimisers, s),
                  (:MinimalPeel, :GreedyPeel, :NoPeel))
        @test Base.ispublic(PortfolioOptimisers, :AbstractPeel)
        @test Base.ispublic(PortfolioOptimisers, :peel_assets)
        # A peel of the caller's own reaches the covariance through the interface.
        s = @test_logs (:warn, r"indices \[3, 4\]") cov(peel_covariance(PeelEveryEnd()),
                                                        peel_disjoint_returns())
        @test all(isnan, s[3, :]) && all(isnan, s[4, :]) && all(isfinite, s[1:2, 1:2])
    end
    @testset "Covariance peels at admission" begin
        X = peel_disjoint_returns()
        s0 = cov(peel_covariance(NoPeel()), X)
        # Today's available-case answer: a NaN cell between two assets with a variance.
        @test isnan(s0[3, 4]) && isnan(s0[4, 3])
        @test all(isfinite, LinearAlgebra.diag(s0))
        s1 = @test_logs (:warn, r"indices \[3\].*1 undetermined pair") cov(peel_covariance(MinimalPeel()),
                                                                           X)
        @test all(isnan, s1[3, :]) && all(isnan, s1[:, 3])
        keep = [1, 2, 4]
        @test s1[keep, keep] == s0[keep, keep]
        # The peel removes the asset, so the other cells are the fit on the panel without it.
        @test s1[keep, keep] == cov(peel_covariance(MinimalPeel()), X[:, keep])
        @test_throws ArgumentError cov(peel_covariance(MinimalPeel()), X; strict = true)
        @test_throws ArgumentError cor(peel_covariance(GreedyPeel()), X; strict = true)
        r1 = @test_logs (:warn,) cor(peel_covariance(GreedyPeel()), X)
        @test all(isnan, r1[3, :])
        @test r1[keep, keep] ≈ cor(peel_covariance(GreedyPeel()), X[:, keep])
        # The semi-covariance takes its admission from the same rule.
        sm = @test_logs (:warn, r"indices \[3\]") cov(peel_covariance(MinimalPeel();
                                                                      alg = SemiMoment()),
                                                      X)
        @test all(isnan, sm[3, :])
        @test sm[keep, keep] ==
              cov(peel_covariance(MinimalPeel(); alg = SemiMoment()), X[:, keep])
        @test_throws ArgumentError cov(peel_covariance(MinimalPeel(); alg = SemiMoment()),
                                       X; strict = true)
        # The fold reads the state through the same rule, and `strict` reaches it.
        ce = partial_fit!(peel_covariance(MinimalPeel()), X)
        sf = @test_logs (:warn,) cov(ce)
        @test isequal(sf, s1)
        @test_throws ArgumentError cov(ce; strict = true)
        @test_throws ArgumentError cor(ce; strict = true)
        # An estimator wrapped in `Online` reads its Sample Buffer through the batch verb, and
        # `strict` reaches that verb too.
        ow = partial_fit!(PortfolioOptimisers.update_online_estimator(Online(peel_covariance(MinimalPeel()))),
                          X)
        @test isequal((@test_logs (:warn,) cov(ow)), s1)
        @test_throws ArgumentError cov(ow; strict = true)
        @test_throws ArgumentError cor(ow; strict = true)
        # A NoPeel matrix still meets the refusal of the matrix repair.
        @test_throws PortfolioOptimisers.IsNonFiniteError PortfolioOptimisers.matrix_processing_block!(MatrixProcessing(),
                                                                                                       copy(s0),
                                                                                                       X)
        # A pair that shares exactly one row is undetermined under the Bessel correction alone.
        Y = peel_disjoint_returns()
        Y[20, 4] = 0.003
        @test_logs (:warn, r"indices \[3\]") cov(peel_covariance(MinimalPeel()), Y)
        su = @test_logs cov(peel_covariance(MinimalPeel();
                                            ce = StatsBase.SimpleCovariance(;
                                                                            corrected = false)),
                            Y)
        @test all(isfinite, su)
    end
    @testset "Peel beside an asset that admission refuses" begin
        # Asset 5 holds two rows, under the floor, so admission refuses it before the peel. The
        # peel then removes asset 3 from the mask that admission already cut.
        X = hcat(peel_disjoint_returns(), fill(NaN, 40))
        X[39:40, 5] .= [0.01, -0.02]
        cvg = CoveragePolicy(; min_coverage = 0.1)
        s = @test_logs (:warn, r"indices \[3\]") cov(Covariance(; cvg = cvg), X)
        @test all(isnan, s[3, :]) && all(isnan, s[5, :])
        @test all(isfinite, s[[1, 2, 4], [1, 2, 4]])
        # With no undetermined pair, the policy peels nothing and says nothing.
        @test all(isfinite, @test_logs cov(Covariance(; cvg = CoveragePolicy()), X[:, 1:3]))
    end
    @testset "Prior reads one universe" begin
        X = peel_disjoint_returns()
        function pe(peel; kwargs...)
            return EmpiricalPrior(; me = SimpleExpectedReturns(; cvg = CoveragePolicy()),
                                  ce = Covariance(; cvg = CoveragePolicy(; peel = peel)),
                                  kwargs...)
        end
        pr = @test_logs (:warn, r"indices \[3\]") match_mode = :any prior(pe(MinimalPeel()),
                                                                          X)
        @test PortfolioOptimisers.investable_mask(pr) == [true, true, false, true]
        keep = [1, 2, 4]
        pr0 = prior(pe(MinimalPeel()), X[:, keep])
        @test pr.sigma[keep, keep] == pr0.sigma
        @test pr.mu[keep] == pr0.mu
        @test_throws ArgumentError prior(pe(MinimalPeel()), X; strict = true)
        @test_throws PortfolioOptimisers.IsNonFiniteError prior(pe(NoPeel()), X)
        # A log-return prior passes `strict` to its covariance too.
        @test_throws ArgumentError prior(pe(MinimalPeel(); horizon = 1), X; strict = true)
        # The folded prior reads the same universe, and `strict` reaches its covariance.
        pf = partial_fit!(pe(MinimalPeel()), X)
        prf = @test_logs (:warn, r"indices \[3\]") match_mode = :any prior(pf)
        @test isequal(prf.sigma, pr.sigma)
        @test_throws ArgumentError prior(pf; strict = true)
        @test_throws ArgumentError prior(partial_fit!(pe(MinimalPeel(); horizon = 1), X);
                                         strict = true)
    end
    @testset "Parity: the oracle's greedy peel" begin
        # The oracle's covariance setter, with no nearest repair, peels the available-case
        # covariance of each case. `Input` pins the matrix it read, and `GreedyPeel` reaches
        # the matrix it wrote.
        for case in ("Counterexample", "Windows")
            X = peel_parity_returns(Symbol(case))
            s0 = cov(peel_covariance(NoPeel()), X)
            @test parity_compare(s0, parity_load("peel", case, "Input");
                                 name = "peel $(case) Input").ok
            sg = @test_logs (:warn,) cov(peel_covariance(GreedyPeel()), X)
            @test parity_compare(sg, parity_load("peel", case, "Sigma");
                                 name = "peel $(case) Sigma").ok
        end
        # On the counterexample the minimal peel keeps X, and removes A1 to A3 alone.
        sm = @test_logs (:warn, r"indices \[3, 4, 5\]") cov(peel_covariance(MinimalPeel()),
                                                            peel_parity_returns(:Counterexample))
        @test findall(!isfinite, LinearAlgebra.diag(sm)) == [3, 4, 5]
        # On the windows the greedy peel is minimum, so the minimal peel is the oracle's.
        sw = @test_logs (:warn,) cov(peel_covariance(MinimalPeel()),
                                     peel_parity_returns(:Windows))
        @test parity_compare(sw, parity_load("peel", "Windows", "Sigma");
                             name = "peel Windows minimal").ok
    end
end
