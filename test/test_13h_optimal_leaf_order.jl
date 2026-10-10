using PortfolioOptimisers, Test, LinearAlgebra, StableRNGs, Clustering

const PO = PortfolioOptimisers

#=
`branchorder = :optimal` is the exact optimal leaf ordering, #1494.

Clustering.jl maps `:optimal` and `:barjoseph` to one function, `orderbranches_barjoseph!`, a
heuristic that flips each merge once from its four outermost leaves. The library dispatches
`Val(branchorder)` before Clustering.jl reads the keyword: `:optimal` runs the dynamic
programme of `optimal_leaf_order!`, and `:barjoseph` keeps the heuristic under its own name.

Every expected value here is a brute-force minimum over each combination of flips of the
merges, never a value recorded from a run. A tree of `N` leaves has `N - 1` merges, so the
search has `2^(N - 1)` orders, and `N <= 9` keeps it under 256.
=#

# The leaves below the root, left child first, which is the order `Clustering.Hclust` stores.
function merge_order(ml, mr)
    below(c) = c < 0 ? [-c] : vcat(below(ml[c]), below(mr[c]))
    return below(length(ml))
end

function adjacent_sum(D, o)
    return sum(D[o[t], o[t + 1]] for t in 1:(length(o) - 1); init = zero(eltype(D)))
end

function brute_force_minimum(D, ml0, mr0)
    best = nothing
    for mask in 0:(2 ^ length(ml0) - 1)
        ml, mr = copy(ml0), copy(mr0)
        for v in eachindex(ml)
            if isodd(mask >> (v - 1))
                ml[v], mr[v] = mr[v], ml[v]
            end
        end
        cost = adjacent_sum(D, merge_order(ml, mr))
        best = isnothing(best) ? cost : min(best, cost)
    end
    return best
end

euclidean(X) = [norm(X[:, i] - X[:, j]) for i in axes(X, 2), j in axes(X, 2)]

@testset "Optimal leaf order" begin
    @testset "The four-leaf case where the heuristic misses" begin
        # The tree is `((1, 4), 3)` merged with `2`. The heuristic keeps `[3, 1, 4, 2]`,
        # 1.25 + 0.746 + 2.343 = 4.339. A flip of `(1, 4)` and a flip of its merge with `3`
        # give `[4, 1, 3, 2]`, 0.746 + 1.25 + 1.799 = 3.795, and no order the tree permits
        # is shorter.
        D = [0.0 1.653 1.25 0.746; 1.653 0.0 1.799 2.343; 1.25 1.799 0.0 1.833;
             0.746 2.343 1.833 0.0]
        h = PO.branch_ordered_hclust(D, :ward, Val(:barjoseph))
        r = PO.branch_ordered_hclust(D, :ward, Val(:optimal))
        @test h.order == [3, 1, 4, 2]
        @test r.order == [4, 1, 3, 2]
        @test adjacent_sum(D, r.order) ≈ 3.795 rtol = 1e-15
        @test adjacent_sum(D, r.order) ==
              brute_force_minimum(D, h.merges[:, 1], h.merges[:, 2])
        @test r.merges == [-4 -1; 1 -3; 2 -2]
    end

    @testset "Every linkage reaches the brute-force minimum" begin
        rng = StableRNG(1494)
        misses = 0
        for linkage in (:single, :average, :complete, :ward), _ in 1:40
            N = rand(rng, 4:9)
            D = euclidean(randn(rng, 3, N))
            h = PO.branch_ordered_hclust(D, linkage, Val(:barjoseph))
            r = PO.branch_ordered_hclust(D, linkage, Val(:optimal))
            best = brute_force_minimum(D, h.merges[:, 1], h.merges[:, 2])
            # The measured gap is 0 on every tree, so the tolerance holds round-off alone.
            @test adjacent_sum(D, r.order) ≈ best rtol = 1e-12
            @test adjacent_sum(D, r.order) <= adjacent_sum(D, h.order) + 1e-12
            misses += adjacent_sum(D, h.order) > best + 1e-12
            # A flip keeps the tree: each merge holds the same two children and height, so
            # every cut is the same.
            @test sort.(eachrow(r.merges)) == sort.(eachrow(h.merges))
            @test r.heights == h.heights
            @test r.linkage == h.linkage
            for k in 1:N
                @test cutree(r; k = k) == cutree(h; k = k)
            end
            # The stored order is the one the merges spell.
            @test r.order == merge_order(r.merges[:, 1], r.merges[:, 2])
        end
        # The heuristic misses the minimum on most of these trees, 89 of 160 when measured.
        @test misses > 0
    end

    @testset "`clusterise` takes the exact order by default" begin
        rng = StableRNG(2001)
        X = randn(rng, 200, 9)
        clr = clusterise(ClustersEstimator(), X)
        clh = clusterise(ClustersEstimator(), X; branchorder = :barjoseph)
        D = clr.D
        best = brute_force_minimum(D, clh.res.merges[:, 1], clh.res.merges[:, 2])
        @test adjacent_sum(D, clr.res.order) ≈ best rtol = 1e-12
        @test clr.k == clh.k
        @test clusterise(ClustersEstimator(), X; branchorder = :r).res.order ==
              hclust(D; linkage = :ward, branchorder = :r).order
        @test_throws ArgumentError clusterise(ClustersEstimator(), X;
                                              branchorder = :unknown)
    end

    @testset "DBHT dispatches the same names" begin
        rng = StableRNG(2012)
        for _ in 1:4
            X = randn(rng, 80, 9)
            C = cor(X)
            D = sqrt.(clamp.((1 .- C) ./ 2, 0, 1))
            S = 1 .- D .^ 2
            ro = PO.DBHTs(D, S; branchorder = :optimal)[end]
            rb = PO.DBHTs(D, S; branchorder = :barjoseph)[end]
            rd = PO.DBHTs(D, S; branchorder = :default)[end]
            best = brute_force_minimum(D, rb.merges[:, 1], rb.merges[:, 2])
            @test adjacent_sum(D, ro.order) ≈ best rtol = 1e-12
            @test sort.(eachrow(ro.merges)) == sort.(eachrow(rb.merges))
            @test ro.heights == rb.heights == rd.heights
            # `:barjoseph` is Clustering.jl's own heuristic on the unordered merges.
            hmer = Clustering.HclustMerges{Float64}(9)
            append!(hmer.mleft, rd.merges[:, 1])
            append!(hmer.mright, rd.merges[:, 2])
            append!(hmer.heights, rd.heights)
            Clustering.orderbranches_barjoseph!(hmer, D)
            @test rb.merges == hcat(hmer.mleft, hmer.mright)
        end
    end

    @testset "The numeric type comes from the distances" begin
        # Integer and rational distances keep their type, and the sum is exact.
        D = [0 4 1 3; 4 0 2 5; 1 2 0 6; 3 5 6 0]
        ml, mr = [-1, -2, 1], [-3, -4, 2]
        best = brute_force_minimum(D, ml, mr)
        o = PO.optimal_leaf_order!(copy(ml), copy(mr), D)
        @test adjacent_sum(D, o) == best
        Dq = D .// 3
        oq = PO.optimal_leaf_order!(copy(ml), copy(mr), Dq)
        @test adjacent_sum(Dq, oq) == best // 3
        @test adjacent_sum(Dq, oq) isa Rational{Int}
    end

    @testset "One and two leaves" begin
        @test PO.optimal_leaf_order!(Int[], Int[], zeros(1, 1)) == [1]
        ml, mr = [-2], [-1]
        @test PO.optimal_leaf_order!(ml, mr, [0.0 1.0; 1.0 0.0]) == [2, 1]
        # The root keeps its orientation: its left child stays first.
        @test (ml, mr) == ([-2], [-1])
    end
end
