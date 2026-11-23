#=
The union of named Exogenous Graphs, `graph_union` and `GraphUnion` (#2208, the decision table of
#2203, map #2162). Each testset is one row of the table. The oracle has no union of graphs, so
there is no parity fixture: each check compares with a dense sum written out by hand.
=#
using Test, PortfolioOptimisers, LinearAlgebra, SparseArrays, Dates, StableRNGs

const GU = PortfolioOptimisers

# A rule of the user's own for a missing part, which never finds a graph in force.
struct GuNeverKnown <: PortfolioOptimisers.AbstractUnknownEntryRule end
PortfolioOptimisers.union_parts_in_force(::GuNeverKnown, ::AbstractVector{Bool}) = nothing

# Two relations over the assets A, B, C and the non-asset S, the sources in the rows.
gu_supply() = ExogenousGraph(; edges = [("S", "A", 2.0), ("S", "B", 1.0), ("A", "C", 1.0)])
gu_credit() = ExogenousGraph(; edges = [("B", "A", 1.0), ("S", "A", 1.0)])
function gu_rd(G; ng = ["supply", "credit"][1:length(G)], nx = ["A", "B", "C"], T = 12)
    return ReturnsResult(; nx = nx, X = randn(StableRNG(2208), T, length(nx)), ng = ng,
                         G = G)
end
# The dense adjacency of `g` over `nodes`, the sources in the rows.
function gu_dense(g, nodes)
    A = zeros(length(nodes), length(nodes))
    pos = Dict(n => k for (k, n) in enumerate(nodes))
    for (j, i, v) in zip(findnz(sparse(g.At))...)
        A[pos[g.nodes[i]], pos[g.nodes[j]]] += v
    end
    return A
end

@testset "Graph union (#2208)" begin
    @testset "Row 1: the function and the spec are two forms of one union" begin
        u = graph_union(gu_supply(), gu_credit())
        @test isa(u, ExogenousGraph)
        s = ExogenousGraphSeries(; ts = [1, 6], graphs = [gu_supply(), gu_credit()])
        @test isa(graph_union(s, gu_credit()), ExogenousGraphSeries)
        # The spec in a consumer gives the walk of the stored union.
        rd = gu_rd([gu_supply(), gu_credit()])
        ru = ReturnsResult(; nx = rd.nx, X = rd.X, ng = ["supply+credit"], G = [u])
        a = asset_panel(WalkProfilePanel(; ng = GraphUnion(; ng = ["supply", "credit"])),
                        nothing, rd, rd.X)
        b = asset_panel(WalkProfilePanel(; ng = "supply+credit"), nothing, ru, ru.X)
        @test a.pf[1].name == b.pf[1].name == "walk_out_supply+credit"
        @test a.pf[1].vals == b.pf[1].vals
        @test a.pf[1].labels == b.pf[1].labels
    end
    @testset "Row 2: the ng of each consumer on dev" begin
        rd = gu_rd([gu_supply(), gu_credit()])
        gu = GraphUnion(; ng = ["supply", "credit"], nrm = WeightShare())
        # A vector keeps each entry apart, a name or a union.
        ape = WalkProfilePanel(; ng = ["supply", gu], dir = BothEdges())
        pnl = asset_panel(ape, nothing, rd, rd.X)
        @test [f.name for f in pnl.pf] ==
              ["walk_out_supply", "walk_in_supply", "walk_out_supply+credit",
               "walk_in_supply+credit"]
        # Each single direction builds its own union, along its own direction.
        for (k, d) in ((3, OutEdges()), (4, InEdges()))
            g = graph_union(gu_supply(), gu_credit(); nrm = WeightShare(), dir = d)
            rg = ReturnsResult(; nx = rd.nx, X = rd.X, ng = ["u"], G = [g])
            q = asset_panel(WalkProfilePanel(; ng = "u", dir = d), nothing, rg, rg.X)
            @test pnl.pf[k].vals ≈ q.pf[1].vals
        end
        # WalkDistance takes the union too.
        de = WalkDistance(; ng = gu)
        @test de.ape.ng === gu
        # The sink C has no out-edge in the union, so the walk cannot read it.
        keep = PortfolioOptimisers.readable_assets(de.ape, rd)
        @test keep == [true, true, false]
        rv = GU.port_opt_view(rd, findall(keep))
        S, D = cor_and_dist(de, nothing, rv.X; rd = rv)
        @test size(D) == (2, 2)
        # A vector of unions alone, and a vector of type Any, are valid.
        @test WalkProfilePanel(; ng = [gu]).ng == [gu]
        @test WalkProfilePanel(; ng = Any["supply", gu]).ng[1] == "supply"
        # Anything else in the vector is refused.
        @test_throws ArgumentError WalkProfilePanel(; ng = Any["supply", 1])
        @test_throws IsEmptyError WalkProfilePanel(; ng = Any[])
        # The field takes the prefix of #2209.
        pp = asset_panel(WalkProfilePanel(; ng = gu, prefix = "dep"), nothing, rd, rd.X)
        @test pp.pf[1].name == "dep_out_supply+credit"
    end
    @testset "Row 3: the node set and the storage" begin
        u = graph_union(gu_supply(), gu_credit())
        @test u.nodes == ["S", "A", "B", "C"]
        @test issparse(u.At)
        d1 = ExogenousGraph(; nodes = ["A", "B"], A = [0.0 1.0; 0.0 0.0])
        d2 = ExogenousGraph(; nodes = ["B", "C"], A = [0.0 2.0; 0.0 0.0])
        ud = graph_union(d1, d2)
        @test ud.nodes == ["A", "B", "C"]
        @test isa(ud.At, Matrix)
        @test adjacency(ud) == [0 1 0; 0 0 2; 0 0 0]
        @test issparse(graph_union(d1, gu_credit()).At)
    end
    @testset "Row 4: weights, a shared edge and the normalisation" begin
        g1, g2 = gu_supply(), gu_credit()
        u = graph_union(g1, g2)
        nodes = u.nodes
        @test adjacency(u) == gu_dense(g1, nodes) + gu_dense(g2, nodes)
        # The shared edge S → A gets the sum of its weights.
        @test adjacency(u)[1, 2] == 3.0
        w = graph_union(g1, g2; w = [0.5, 2])
        @test adjacency(w) ≈ 0.5 * gu_dense(g1, nodes) + 2 * gu_dense(g2, nodes)
        # A part with weight 0 is skipped, its nodes too.
        z = graph_union(g1, ExogenousGraph(; edges = [("X", "Y", 1.0)]); w = [1, 0])
        @test z.nodes == g1.nodes
        @test_throws DimensionMismatch graph_union(g1, g2; w = [1.0])
        @test_throws DomainError graph_union(g1, g2; w = [1.0, -1.0])
        @test_throws DomainError graph_union(g1, g2; w = [1.0, NaN])
        @test_throws DomainError graph_union(g1, g2; w = [0.0, 0.0])
        @test_throws DomainError GraphUnion(; ng = ["a", "b"], w = [0, 0])
        # WeightShare divides each part by its column sums under OutEdges.
        share(A) = A * Diagonal([iszero(s) ? 0.0 : 1 / s for s in vec(sum(A; dims = 1))])
        o = graph_union(g1, g2; nrm = WeightShare())
        @test adjacency(o)' ≈ share(gu_dense(g1, nodes)') + share(gu_dense(g2, nodes)')
        # and by its row sums under InEdges.
        i = graph_union(g1, g2; nrm = WeightShare(), dir = InEdges())
        @test adjacency(i) ≈ share(gu_dense(g1, nodes)) + share(gu_dense(g2, nodes))
        @test adjacency(i)[1, 2] ≈ 1.5
        @test adjacency(i)[3, 2] ≈ 0.5
        @test_throws MethodError graph_union(g1, g2; nrm = WeightShare(), dir = BothEdges())
        @test graph_union(g1, g2; dir = BothEdges()) == u
    end
    @testset "Row 5: the time rule and a missing part" begin
        a1 = ExogenousGraph(; edges = [("S", "A", 1.0)])
        a2 = ExogenousGraph(; edges = [("S", "A", 2.0)])
        b1 = ExogenousGraph(; edges = [("B", "A", 1.0)])
        sa = ExogenousGraphSeries(; ts = [2, 7], graphs = [a1, a2])
        sb = ExogenousGraphSeries(; ts = [4, 9], graphs = [b1, nothing])
        # EntrywiseUnknown: the keys start at the latest first key, and a `nothing` in force
        # gives `nothing`.
        e = graph_union(sa, sb)
        @test e.ts == [4, 7, 9]
        @test e.graphs[1] == graph_union(a1, b1)
        @test e.graphs[2] == graph_union(a2, b1)
        @test isnothing(e.graphs[3])
        # ZeroUnknown: the keys start at the earliest first key, and a missing part is empty.
        z = graph_union(sa, sb; unknown = ZeroUnknown())
        @test z.ts == [2, 4, 7, 9]
        @test z.graphs[1] == a1
        @test z.graphs[4] == a2
        # A static part is in force at every row.
        st = graph_union(sa, b1)
        @test st.ts == [2, 7]
        zs = graph_union(sa, b1; unknown = ZeroUnknown())
        @test zs.ts == [1, 2, 7]
        @test zs.graphs[1] == b1
        # Date keys follow the same rule, and a static part takes the earliest date.
        d = ExogenousGraphSeries(; ts = [Date(2020, 1, 3), Date(2020, 1, 8)],
                                 graphs = [a1, a2])
        zd = graph_union(d, b1; unknown = ZeroUnknown())
        @test first(zd.ts) == typemin(Date)
        @test zd.ts[2:end] == d.ts
        # Integer keys and date keys cannot mix.
        @test_throws ArgumentError graph_union(sa, d)
        @test_throws ArgumentError GU.assert_consumer_entry(WalkProfilePanel(),
                                                            GraphUnion(; ng = ["a", "b"]),
                                                            ["a", "b"], [sa, d], "")
        # A rule of the user's own takes the second job through `union_parts_in_force`. A rule
        # that finds no graph in force at any key is refused.
        @test GU.union_parts_in_force(GuNeverKnown(), [true, true]) === nothing
        @test_throws ArgumentError graph_union(sa, sb; unknown = GuNeverKnown())
        # KindwiseUnknown has no method of the second job.
        @test_throws MethodError graph_union(sa, sb; unknown = KindwiseUnknown())
        @test_throws MethodError GraphUnion(; ng = ["a", "b"], unknown = KindwiseUnknown())
        # The resolver reads the snapshots in force at the last row, under the rule.
        rd = ReturnsResult(; nx = ["A", "B"], X = randn(StableRNG(1), 12, 2),
                           ng = ["a", "b"], G = [sa, sb])
        ape = WalkProfilePanel(; ng = GraphUnion(; ng = ["a", "b"]))
        v = GU.port_opt_view(rd, 1:8, :)
        @test GU.consumer_graph(ape, v, ape.ng, OutEdges()) == graph_union(a2, b1)
        # Before the latest first key, the message names the part that has not started. A
        # view of the rows 1:3 holds a `nothing` key at its first row in place of the later
        # key, so the core method reads the last row 3 of the full series.
        err = try
            GU.consumer_graph(ape, rd.ng, rd.G, ape.ng, 3, nothing, OutEdges())
        catch ex
            ex
        end
        @test isa(err, ArgumentError)
        @test occursin("\"b\" comes before its first key 4", err.msg)
        @test occursin("a+b", err.msg)
        early = GU.port_opt_view(rd, 1:3, :)
        @test_throws ArgumentError GU.consumer_graph(ape, early, ape.ng, OutEdges())
        # At a `nothing` key the message says so.
        err = try
            GU.consumer_graph(ape, rd, ape.ng, OutEdges())
        catch ex
            ex
        end
        @test occursin("holds `nothing` at its key 9", err.msg)
        # A series keyed by dates reads the timestamp of the last row.
        dts = Date(2020, 1, 1) .+ Day.(0:11)
        da = ExogenousGraphSeries(; ts = [dts[2], dts[7]], graphs = [a1, a2])
        db = ExogenousGraphSeries(; ts = [dts[4]], graphs = [b1])
        rdd = ReturnsResult(; nx = ["A", "B"], X = randn(StableRNG(2), 12, 2), ts = dts,
                            ng = ["a", "b"], G = [da, db])
        @test GU.consumer_graph(ape, rdd, ape.ng, OutEdges()) == graph_union(a2, b1)
        @test GU.consumer_graph(ape, GU.port_opt_view(rdd, 1:5, :), ape.ng, OutEdges()) ==
              graph_union(a1, b1)
        # ZeroUnknown reads the missing part as empty.
        apz = WalkProfilePanel(;
                               ng = GraphUnion(; ng = ["a", "b"], unknown = ZeroUnknown()))
        @test GU.consumer_graph(apz, early, apz.ng, OutEdges()) == a1
        @test GU.consumer_graph(apz, rd, apz.ng, OutEdges()) == a2
    end
    @testset "Row 6: one graph" begin
        @test_throws IsEmptyError GraphUnion(; ng = String[])
        @test_throws ArgumentError GraphUnion(; ng = ["a", "a"])
        err = try
            GraphUnion(; ng = ["a"])
        catch ex
            ex
        end
        @test isa(err, ArgumentError)
        @test occursin("ng = \"a\"", err.msg)
        @test GraphUnion(; ng = ["a"], nrm = WeightShare()).ng == ["a"]
        @test_throws IsEmptyError graph_union()
        @test_throws ArgumentError graph_union(gu_supply())
        g = graph_union(gu_supply(); nrm = WeightShare())
        @test adjacency(g)[1, 2] ≈ 2 / 3
    end
    @testset "Row 7: the name" begin
        @test GU.graph_entry_label(GraphUnion(; ng = ["a", "b"])) == "a+b"
        @test GU.graph_entry_label(GraphUnion(; ng = ["a", "b"], name = "ab")) == "ab"
        @test_throws IsEmptyError GraphUnion(; ng = ["a", "b"], name = "")
        rd = gu_rd([gu_supply(), gu_credit()])
        p = asset_panel(WalkProfilePanel(;
                                         ng = GraphUnion(; ng = ["supply", "credit"],
                                                         name = "both")), nothing, rd, rd.X)
        @test p.pf[1].name == "walk_out_both"
        # A name that collides with another entry is refused.
        @test_throws ArgumentError WalkProfilePanel(;
                                                    ng = Any["a+b",
                                                             GraphUnion(; ng = ["a", "b"])])
    end
    @testset "Row 8: the checks of the entry verb and the element type" begin
        bad = ExogenousGraph(; edges = [("S", "A", 1.0), ("S", "B", -1.0)])
        s = ExogenousGraphSeries(; ts = [1, 9], graphs = [gu_credit(), bad])
        rd = gu_rd([gu_supply(), s])
        de = WalkDistance(; ng = GraphUnion(; ng = ["supply", "credit"]))
        # Fold 1 cannot see the bad snapshot at row 9, and the entry check sees it.
        err = try
            GU.assert_graph_entry(de, rd)
        catch ex
            ex
        end
        @test isa(err, ArgumentError)
        @test occursin("\"credit\" at the key 9", err.msg)
        # A fit checks at its entry, before the first view.
        hrp(de) = HierarchicalRiskParity(;
                                         ser = HierarchicalSeriation(;
                                                                     cle = ClustersEstimator(;
                                                                                             de = de)))
        @test_throws r"cannot read the Exogenous Graph \"credit\"" optimise(hrp(de), rd)
        # A fit with a union drops the asset that the walk cannot read.
        ok = gu_rd([gu_supply(), gu_credit()])
        w = optimise(hrp(WalkDistance(; ng = GraphUnion(; ng = ["supply", "credit"]))), ok).w
        @test sum(w) ≈ 1
        @test iszero(w[3])
        # A name that the returns data does not hold.
        @test_throws ArgumentError GU.assert_graph_entry(WalkDistance(;
                                                                      ng = GraphUnion(;
                                                                                      ng = ["supply",
                                                                                            "debt"])),
                                                         gu_rd([gu_supply(), gu_credit()]))
        @test isnothing(GU.assert_graph_entry(WalkDistance(;
                                                           ng = ["supply",
                                                                 GraphUnion(;
                                                                            ng = ["supply",
                                                                                  "credit"])]),
                                              gu_rd([gu_supply(), gu_credit()])))
        # The element type promotes over the parts and `w`.
        f1 = ExogenousGraph(; nodes = ["A", "B"], A = Float32[0 1; 0 0])
        f2 = ExogenousGraph(; nodes = ["A", "B"], A = Float32[0 0; 1 0])
        @test eltype(graph_union(f1, f2).At) == Float32
        @test eltype(graph_union(f1, f2; w = Float32[1, 2]).At) == Float32
        @test eltype(graph_union(f1, f2; w = [1.0, 2.0]).At) == Float64
        @test eltype(graph_union(f1, f2; nrm = WeightShare()).At) == Float32
    end
end
