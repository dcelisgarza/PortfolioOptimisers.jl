using PortfolioOptimisers, Test, SparseArrays, LinearAlgebra

const PO = PortfolioOptimisers

#=
This file drives the back half of the DBHT pipeline at its own seam, #1037.

`DBHTs` runs `PMFG_T2s`, `distance_wei` and `CliqHierarchyTree2s`, and then hands the
bubble structure to `BubbleHierarchy`, `DirectHb`, `BubbleCluster8s`, `BubbleMember`,
`LinkageFunction`, `DendroConstruct`, `build_link_and_dendro`, `HierarchyConstruct4s` and
`turn_into_Hclust_merges` in that order. `test_13_phylogeny.jl` reaches the first three
directly and the rest only through `DBHTs`, so a wrong dendrogram was diagnosed backwards
from the final clustering. Every function past `CliqHierarchyTree2s` reads matrices alone,
so each is driven here from a bubble structure small enough to work by hand, and the
expected answer of every assertion is derived in the comment above it, never recorded from
a run.

Two conventions the seam relies on and `DBHTs` supplies:

  - `Hb` is sparse. `DirectHb` cuts an edge out of a copy and calls `dropzeros!` on it,
    which has no method for a dense matrix. `BubbleHierarchy` returns a sparse `H`.
  - `Mv` is the vertex membership of the *whole* bubble, and `Mc` restricts it to one
    cluster. `BubbleMember` scores a vertex against the whole bubble through `Mv` and reads
    only the candidates off `Mc`.

The `BubbleMember` set is the one ADR 0049 asked for: its denominator is a total weight,
not a count, and the ADR records that signed cancellation drives it to zero or negative
with no defence at this step. `PMFG_T2s` refuses a negative and a `NaN` before `Rpm` is
built, and `test_13_phylogeny.jl`'s "`PMFG_T2s`' backstop" pins that refusal, so the tests
here pin the mechanism the guard upstream exists for.
=#

# A symmetric weight matrix from an edge list, so a fixture reads as a graph.
function symmetric_from_edges(N, edges)
    R = zeros(N, N)
    for (i, j, w) in edges
        R[i, j] = w
        R[j, i] = w
    end
    return R
end

@testset "The DBHT seam (#1037)" begin
    @testset "BubbleHierarchy from a hand-built clique hierarchy" begin
        # Clique 1 is the root, cliques 2 and 3 are its children, and clique 4 is the child
        # of clique 2. Only cliques 1 and 2 are separating. The first bubble holds the root
        # and its children, and clique 2 opens the second, so the two bubbles share clique
        # 2 and nothing else.
        H, Mb = PO.BubbleHierarchy([0, 1, 1, 2], [1, 1, 0, 0])
        @test Mb == [1 0; 1 1; 1 0; 0 1]
        @test H isa SparseMatrixCSC
        # `H` is symmetrised as `H + transpose(H)`, so a stored entry is `2` and not `1`.
        # Every consumer reads it as `Hb .!= 0`, so the pattern is what the seam carries.
        @test (H .!= 0) == [0 1; 1 0]
        @test iszero(LinearAlgebra.diag(H))
        # Two roots open one bubble that holds both before either root's own bubble. Clique
        # 3 hangs off root 1 and clique 4 off root 2, so bubbles 2 and 3 each share one root
        # with bubble 1 and nothing with each other.
        H, Mb = PO.BubbleHierarchy([0, 0, 1, 2], [0, 0, 0, 0])
        @test Mb == [1 1 0; 1 0 1; 0 1 0; 0 0 1]
        @test (H .!= 0) == [0 1 1; 1 0 0; 1 0 0]
    end

    #=
    A chain of three bubbles. Bubble 1 holds cliques {1,2,3} and {2,3,4}, bubble 2 holds
    {2,3,4}, {3,4,5} and {2,4,7}, and bubble 3 holds {3,4,5} and {4,5,6}. Clique {2,3,4}
    separates bubbles 1 and 2, and {3,4,5} separates bubbles 2 and 3. Vertex 7 is in
    bubble 2 alone, so it is in no converging bubble and is assigned by distance.
    =#
    CliqList = [1 2 3; 2 3 4; 3 4 5; 4 5 6; 2 4 7]
    Mb = [1 0 0; 1 1 0; 0 1 1; 0 0 1; 0 1 0]
    Hb = SparseArrays.sparse([0 1 0; 1 0 1; 0 1 0])
    Mv = zeros(Int, 7, 3)
    for b in axes(Mb, 2)
        Mv[sort!(unique(CliqList[Mb[:, b] .!= 0, :])), b] .= 1
    end
    Rpm = symmetric_from_edges(7,
                               [(1, 2, 2.0), (1, 3, 2.0), (2, 3, 0.5), (2, 4, 1.0),
                                (3, 4, 0.5), (3, 5, 0.1), (4, 5, 0.1), (4, 6, 3.0),
                                (5, 6, 3.0), (2, 7, 0.1), (4, 7, 0.1)])
    # Only row 7 of the shortest paths is read: every other vertex is in a converging
    # bubble. Vertex 7 is at mean distance 2 from {1, 2, 3} and 7/3 from {4, 5, 6}.
    Dpm = ones(7, 7) - LinearAlgebra.I
    Dpm[7, :] = Dpm[:, 7] = [3, 1, 2, 2, 1, 4, 0]

    @testset "DirectHb directs each edge towards the heavier side" begin
        # The vertex sets are the ones the derivation below reads.
        @test Mv == [1 0 0; 1 1 0; 1 1 1; 1 1 1; 0 1 1; 0 0 1; 0 1 0]
        # Edge 1|2 through {2,3,4}: the reached side is bubble 1, whose vertices less the
        # clique are {1}, and the cut side is {5, 6, 7}. The clique draws 2 + 2 = 4 from
        # the left and 0.1 + 0.1 + 3 + 0.1 + 0.1 = 3.4 from the right, so the edge points
        # from bubble 2 to bubble 1 and carries 4.
        # Edge 2|3 through {3,4,5}: the reached side is bubbles 1 and 2, whose vertices
        # less the clique are {1, 2, 7}, and the cut side is {6}. The clique draws
        # 2 + 0.5 + 1 + 0.1 = 3.6 from the left and 3 + 3 = 6 from the right, so the edge
        # points from bubble 2 to bubble 3 and carries 6.
        Hc, Sep = PO.DirectHb(Rpm, Hb, Mb, Mv, CliqList)
        @test Matrix(Hc) == [0 0 0; 4 0 6; 0 0 0]
        # Bubbles 1 and 3 have no outgoing edge, so they converge. Bubble 2 has no incoming
        # edge and two neighbours, so it diverges.
        @test Sep == [1, 2, 1]
        # A dense `Hb` is refused: the edge cut needs `dropzeros!`.
        @test_throws MethodError PO.DirectHb(Rpm, Matrix(Hb), Mb, Mv, CliqList)
    end

    @testset "BubbleCluster8s assigns a shared vertex by chi and a stray one by distance" begin
        Adjv, Tc = PO.BubbleCluster8s(Rpm, Dpm, Hb, Mb, Mv, CliqList)
        # Converging bubble 1 reaches bubble 2 against the arrows, so its non-discrete
        # cluster is {1, 2, 3, 4, 5, 7}; bubble 3 reaches bubble 2 too, so its cluster is
        # {2, 3, 4, 5, 6, 7}.
        @test Matrix(Adjv) == [1 0; 1 1; 1 1; 1 1; 1 1; 0 1; 1 1]
        # Vertices 3 and 4 are in both converging bubbles. Each has four vertices, so both
        # `chi` denominators are 3 * (4 - 2) = 6 and the numerators decide: vertex 3 draws
        # 2 + 0.5 + 0.5 = 3 from bubble 1 and 0.5 + 0.1 = 0.6 from bubble 3; vertex 4 draws
        # 1 + 0.5 = 1.5 from bubble 1 and 0.5 + 0.1 + 3 = 3.6 from bubble 3. Vertex 7 is in
        # neither, and is closer on average to the members of cluster 1.
        @test Tc == [1, 1, 1, 2, 2, 2, 1]
        # One converging bubble, or none, is one cluster and an empty `Adjv`.
        Hb1 = SparseArrays.sparse([0 1; 1 0])
        Adjv, Tc = PO.BubbleCluster8s(Rpm[1:5, 1:5], Dpm[1:5, 1:5], Hb1, [1 0; 1 1; 0 1],
                                      [1 0; 1 0; 1 1; 1 1; 0 1], [1 2 3; 2 3 4; 3 4 5])
        @test size(Adjv) == (0, 0)
        @test Tc == ones(Int, 5)
    end

    @testset "BubbleMember scores a shared vertex by its fraction of the bubble's weight" begin
        #=
        Two bubbles, {1, 2, 3} and {3, 4, 5}, share vertex 3, and the cluster is the whole
        graph, so `Mc == Mv`. The fraction `phi` of a bubble's internal weight that vertex
        3 draws decides its bubble. Bubble 1 carries 1 + 0.1 + 0.1 = 1.2 and vertex 3 draws
        0.2 of it; bubble 2 carries 3 and vertex 3 draws 2 of it. So `phi` is 1/6 against
        2/3, and bubble 2 wins.
        =#
        Mv2 = [1 0; 1 0; 1 1; 0 1; 0 1]
        Rpos = symmetric_from_edges(5,
                                    [(1, 2, 1.0), (1, 3, 0.1), (2, 3, 0.1), (3, 4, 1.0),
                                     (3, 5, 1.0), (4, 5, 1.0)])
        Mvv = PO.BubbleMember(Rpos, Mv2, Mv2)
        @test Mvv == [1 0; 1 0; 0 1; 0 1; 0 1]
        # A vertex in one bubble of the cluster is copied straight from `Mc`, and a vertex
        # outside the cluster gets no bubble.
        Mc = [1 0; 1 0; 0 0; 0 1; 0 1]
        @test PO.BubbleMember(Rpos, Mv2, Mc) == Mc

        @testset "The signed cancellation of ADR 0049" begin
            #=
            The denominator of `phi` is a total weight, not a count, so it has no floor.
            Flip the two edges vertex 3 has into bubble 1 to -0.1 and cut the third to 0.1.
            Bubble 1 now carries 0.1 - 0.1 - 0.1 = -0.1 and vertex 3 draws -0.2 of it, so
            `phi` is (-0.2) / (-0.1) = 2 against the unchanged 2/3, and `argmax` gives
            vertex 3 to the bubble it is repelled from. Nothing at this step notices.
            =#
            Rneg = symmetric_from_edges(5,
                                        [(1, 2, 0.1), (1, 3, -0.1), (2, 3, -0.1),
                                         (3, 4, 1.0), (3, 5, 1.0), (4, 5, 1.0)])
            @test PO.BubbleMember(Rneg, Mv2, Mv2)[3, :] == [1, 0]
            # Cancel bubble 1 exactly: 0 + 1 - 1 = 0 in total, and vertex 3 draws 1 - 1 = 0
            # of it. `phi` is `0 / 0 = NaN`, and `argmax` selects the `NaN`, so the same
            # wrong bubble wins for a second reason.
            Rzero = symmetric_from_edges(5,
                                         [(1, 2, 0.0), (1, 3, 1.0), (2, 3, -1.0),
                                          (3, 4, 1.0), (3, 5, 1.0), (4, 5, 1.0)])
            @test PO.BubbleMember(Rzero, Mv2, Mv2)[3, :] == [1, 0]
            # The defence is upstream: `Rpm` is `PMFG_T2s(S)[1]`, and `PMFG_T2s` refuses a
            # negative before any bubble is built. The two fixtures above are too small for
            # it, so the refusal is pinned on the smallest matrix it accepts.
            W = ones(9, 9)
            W[1, 2] = W[2, 1] = -0.1
            @test_throws DomainError PO.PMFG_T2s(W)
        end
    end

    @testset "LinkageFunction scores the union, not the cut" begin
        # Vertices 1 and 2 carry label 1, vertex 3 label 2 and vertex 4 label 3. Labels 1
        # and 2 are the closest across the cut, 0.1 on both edges, but the union
        # {1, 2, 3} carries the 0.9 between the two members of label 1, so its diameter is
        # 0.9. The union of labels 1 and 3 is 0.9 for the same reason, and the union of
        # labels 2 and 3 is the single edge 0.2, so that pair is selected.
        d = symmetric_from_edges(4,
                                 [(1, 2, 0.9), (1, 3, 0.1), (2, 3, 0.1), (1, 4, 0.5),
                                  (2, 4, 0.5), (3, 4, 0.2)])
        PairLink, dvu = PO.LinkageFunction(d, [1, 1, 2, 3])
        @test PairLink == [2, 3]
        @test dvu == 0.2
        # A union with no non-zero distance scores 0, the smallest score there is, and the
        # first such pair in label order is the one returned.
        PairLink, dvu = PO.LinkageFunction(zeros(3, 3), [1, 2, 3])
        @test PairLink == [1, 2]
        @test dvu == 0
    end

    @testset "DendroConstruct appends the two labels a merge joined" begin
        # Vertices 1 and 2 moved from labels 1 and 2 to label 4; vertex 3 did not move.
        Z = PO.DendroConstruct(zeros(0, 3), [1, 2, 3], [4, 4, 3], 0.5)
        @test Z == [1 2 0.5]
        # Rows accumulate, and a vector height is spliced in as its one entry.
        Z = PO.DendroConstruct(Z, [4, 4, 3], [5, 5, 5], [2])
        @test Z == [1 2 0.5; 3 4 2]
    end

    #=
    Two bubbles, {1, 2, 3} and {3, 4, 5}, share vertex 3. With the positive weights of
    the `BubbleMember` set inverted so that vertex 3 belongs to bubble 1, and the shortest
    paths below, the dendrogram follows by hand.
    =#
    Mv5 = [1 0; 1 0; 1 1; 0 1; 0 1]
    Rpm5 = symmetric_from_edges(5,
                                [(1, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0), (3, 4, 0.1),
                                 (3, 5, 0.1), (4, 5, 1.0)])
    Dpm5 = symmetric_from_edges(5,
                                [(1, 2, 0.2), (1, 3, 0.5), (2, 3, 0.6), (1, 4, 1.0),
                                 (1, 5, 1.0), (2, 4, 1.0), (2, 5, 1.0), (3, 4, 0.7),
                                 (3, 5, 0.8), (4, 5, 0.3)])

    @testset "build_link_and_dendro merges the closest pair and lowers the counter" begin
        # Bubble 1 alone: vertices {1, 2, 3}, two merges, and `nc` starts at 2. Pair
        # (1, 2) has the smallest diameter, 0.2, so it merges first at height 1/2 under the
        # fresh label 6. The only pair left is (3, 6), at height 1/1 under label 7.
        V = [1, 2, 3]
        LabelVec1 = collect(1:5)
        Z, nc, LabelVec1 = PO.build_link_and_dendro(1:2, Dpm5[V, V], LabelVec1[V],
                                                    LabelVec1, copy(LabelVec1), V, 2,
                                                    zeros(0, 3))
        @test Z == [1 2 0.5; 3 6 1.0]
        @test nc == 0
        @test LabelVec1 == [7, 7, 7, 4, 5]
        # An empty range is a no-op.
        Z, nc, LabelVec1 = PO.build_link_and_dendro(1:0, Dpm5[V, V], LabelVec1[V],
                                                    LabelVec1, copy(LabelVec1), V, 0, Z)
        @test Z == [1 2 0.5; 3 6 1.0]
        @test nc == 0
        @test LabelVec1 == [7, 7, 7, 4, 5]
    end

    @testset "HierarchyConstruct4s builds a known dendrogram" begin
        @testset "Two clusters, one bubble each" begin
            #=
            Cluster 1 is {1, 2, 3} and cluster 2 is {4, 5}. Restricted to cluster 1,
            bubble 2 holds vertex 3 alone, and `phi` gives vertex 3 to bubble 1 (2/3
            against 1/6), so cluster 1 is bubble 1 and `nc` starts at 2. Its two merges
            are the ones `build_link_and_dendro` made above: (1, 2) at 1/2, then (3, 6) at
            1. One bubble means no inter-bubble merge. Cluster 2 is bubble 2 with `nc = 1`,
            so (4, 5) merges at height 1 under label 8. The inter-cluster merge joins
            labels 7 and 8, and its height is the count of clusters joined, 1 + 1 = 2, not
            a distance.
            =#
            Z = PO.HierarchyConstruct4s(Rpm5, Dpm5, [1, 1, 1, 2, 2], Mv5)
            @test Z == [1 2 0.5; 3 6 1.0; 4 5 1.0; 7 8 2.0]
            # Every intra-cluster height is at most 1, and the inter-cluster one is above.
            @test all(<=(1), Z[1:3, 3])
            @test Z[4, 3] > 1
        end
        @testset "One cluster, two bubbles" begin
            #=
            The same graph as one cluster. Vertex 3 still goes to bubble 1, so bubble 1 is
            {1, 2, 3} and bubble 2 is {4, 5}, and `nc` starts at 4 for the whole cluster.
            Bubble 1 merges (1, 2) at 1/4 under label 6 and (3, 6) at 1/3 under label 7.
            Bubble 2 merges (4, 5) at 1/2 under label 8. The one inter-bubble merge joins
            7 and 8 at 1/1, and there is no inter-cluster merge.
            =#
            Z = PO.HierarchyConstruct4s(Rpm5, Dpm5, ones(Int, 5), Mv5)
            @test Z == [1 2 0.25; 3 6 1/3; 4 5 0.5; 7 8 1.0]
        end
        @testset "The chain fixture, from its clusters" begin
            #=
            `BubbleCluster8s` gave `Tc = [1, 1, 1, 2, 2, 2, 1]` above. In cluster 1,
            `phi` gives vertex 2 to bubble 2 (1.6/2.4 against 2.5/6) and vertex 3 to
            bubble 1 (3/6 against 1.1/2.4 and 0.6/6.7), so bubble 1 is {1, 3}, bubble 2 is
            {2, 7}, and `nc` starts at 3: (1, 3) at 1/3 under 8, (2, 7) at 1/2 under 9,
            then the inter-bubble (8, 9) at 1 under 10. In cluster 2, vertex 4 goes to
            bubble 2 (1.7/2.4) and vertex 5 to bubble 3 (3.2/6.7), so bubble 2 is {4} and
            bubble 3 is {5, 6}, with `nc = 2`: (5, 6) at 1/2 under 11, then (4, 11) at 1
            under 12. The inter-cluster merge joins 10 and 12 at height 2.
            =#
            Tc = [1, 1, 1, 2, 2, 2, 1]
            Z = PO.HierarchyConstruct4s(Rpm, Dpm, Tc, Mv)
            @test Z == [1 3 1/3; 2 7 0.5; 8 9 1.0; 5 6 0.5; 4 11 1.0; 10 12 2.0]
        end
    end

    @testset "turn_into_Hclust_merges renumbers leaves and rows, and counts sizes" begin
        # The two-cluster dendrogram above, N = 5. A label at most 5 is a leaf and is
        # negated; a label above 5 names the row that built it, less 5. Row 2 joins leaf 3
        # to the pair of row 1, so it holds 3; row 4 joins rows 2 and 3, so it holds 5.
        Z = [1 2 0.5; 3 6 1.0; 4 5 1.0; 7 8 2.0]
        H = PO.turn_into_Hclust_merges(Z)
        @test H == [-1 -2 0.5 2; -3 1 1.0 3; -4 -5 1.0 2; 2 3 2.0 5]
        # The heights are untouched, every leaf appears exactly once, every row is named
        # exactly once by a later row, and the last row holds every leaf.
        @test H[:, 3] == Z[:, 3]
        @test sort(filter(<(0), vec(H[:, 1:2]))) == -(5:-1:1)
        @test sort(filter(>(0), vec(H[:, 1:2]))) == 1:3
        @test H[end, 4] == 5
        # `DBHTs` reads the first two columns as `Int` and the third as the heights, so a
        # row that names a row must name an earlier one.
        for i in axes(H, 1), j in 1:2
            H[i, j] > 0 && @test H[i, j] < i
        end
        # The chain fixture, N = 7, on the same rules.
        Z = [1 3 1/3; 2 7 0.5; 8 9 1.0; 5 6 0.5; 4 11 1.0; 10 12 2.0]
        H = PO.turn_into_Hclust_merges(Z)
        @test H == [-1 -3 1/3 2; -2 -7 0.5 2; 1 2 1.0 4; -5 -6 0.5 2; -4 4 1.0 3; 3 5 2.0 7]
    end
end
