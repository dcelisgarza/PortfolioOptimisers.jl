#=
The Exogenous Graph and the Exogenous Graph Series on the returns data (issue #2171, decision
of #2165).

The file tests the two types and their refusals, the fields `ng` and `G` of the two data types,
the as-of lookup, the observation view and its rebase of integer keys, `prices_to_returns`,
`vcat_observations`, the round trip through the edge table and the manifest, and the check of
the requirements of a consumer at the entry of a user verb. The consumer `GraphReader` below
stands for the consumers of #2172 and #2173: it states a requirement and names its graphs.
=#
using Dates, SparseArrays, Random
const PO = PortfolioOptimisers

struct GraphReader{T} <: PO.AbstractEstimator
    ng::T
end
PO.graph_requirements(::GraphReader) = (NonNegativeWeights(),)
PO.graph_names(c::GraphReader) = c.ng
struct GraphHolder{A, B} <: PO.AbstractEstimator
    v::A
    p::B
end
# The graph in force at a key `t`, read by the definition: the last key at or before `t`.
function as_of(s::ExogenousGraphSeries, t)
    p = searchsortedlast(s.ts, t)
    return p == 0 ? nothing : s.graphs[p]
end
as_of(g::ExogenousGraph, ::Any) = g

const g1 = ExogenousGraph(; edges = [("S", "A", 0.4)])
const g2 = ExogenousGraph(; edges = [("S", "A", 0.5), ("S", "B", 0.5)])
const g3 = ExogenousGraph(; edges = [("S", "B", 1.0)])

@testset "ExogenousGraph: the three forms" begin
    # Square form: rows are sources, the stored layout has the sources in the columns.
    A = [0 0 0; 0 0 0; 0.4 0.6 0]
    g = ExogenousGraph(; nodes = ["A", "B", "S"], A = A)
    @test g.At == permutedims(A)
    @test isa(g.At, Matrix{Float64})
    @test adjacency(g) == A
    @test adjacency(g) isa LinearAlgebra.Transpose
    # A sparse input stays sparse, and drops its stored zeros.
    S = sparse([1, 3], [2, 1], [0.5, 0.0], 3, 3)
    gs = ExogenousGraph(; nodes = ["a", "b", "c"], A = S)
    @test isa(gs.At, SparseMatrixCSC)
    @test nnz(gs.At) == 1 && gs.At[2, 1] == 0.5
    # An integer or a Bool adjacency is floated once.
    @test eltype(ExogenousGraph(; nodes = ["a", "b"], A = [0 1; 1 0]).At) == Float64
    @test eltype(ExogenousGraph(; nodes = ["a", "b"], A = Bool[0 1; 0 0]).At) == Float64
    @test eltype(ExogenousGraph(; nodes = ["a", "b"], A = sparse([0 2; 0 0])).At) == Float64
    # A Float32 adjacency keeps its type.
    @test eltype(ExogenousGraph(; nodes = ["a", "b"], A = Float32[0 1; 0 0]).At) == Float32

    # Block form with `sources == targets` is the square form, and keeps a dense `A`.
    b = ExogenousGraph(; sources = ["a", "b"], targets = ["a", "b"], A = [0 0.3; 0 0])
    @test b == ExogenousGraph(; nodes = ["a", "b"], A = [0 0.3; 0 0])
    @test isa(b.At, Matrix)
    # Block form: nodes are the sources, then each target that is not a source. A name in
    # both lists is one node, so a supplier of a supplier is a row of the block.
    blk = ExogenousGraph(; sources = ["A", "B", "S1"], targets = ["S1", "S2"],
                         A = [0.5 0.0; 0.2 0.8; 0.0 0.3])
    @test blk.nodes == ["A", "B", "S1", "S2"]
    @test isa(blk.At, SparseMatrixCSC)
    @test adjacency(blk)[1, 3] == 0.5
    @test adjacency(blk)[2, 3] == 0.2
    @test adjacency(blk)[2, 4] == 0.8
    @test adjacency(blk)[3, 4] == 0.3
    @test nnz(blk.At) == 4

    # Edge list: nodes in the order of first use, or as given with isolated nodes.
    e = ExogenousGraph(; edges = [("S", "A", 0.4), ("S", "B", 0.6)])
    @test e.nodes == ["S", "A", "B"]
    @test isa(e.At, SparseMatrixCSC)
    @test e == ExogenousGraph(; sources = ["S"], targets = ["A", "B"], A = [0.4 0.6])
    en = ExogenousGraph(; nodes = ["Z", "S", "A"], edges = [("S", "A", 1)])
    @test en.nodes == ["Z", "S", "A"]
    @test eltype(en.At) == Float64
    @test adjacency(en)[2, 3] == 1.0
    # Mixed weight types promote.
    @test eltype(ExogenousGraph(; edges = [("a", "b", 1), ("b", "c", 0.5f0)]).At) == Float32
    # A graph with no edge is valid when the nodes are given.
    @test nnz(ExogenousGraph(; nodes = ["a"], edges = Tuple{String, String, Float64}[]).At) ==
          0
    # Negative weights, self-loops and cycles are accepted.
    lp = ExogenousGraph(; edges = [("a", "a", 0.2), ("a", "b", -0.3), ("b", "a", 1.0)])
    @test adjacency(lp)[1, 1] == 0.2 && adjacency(lp)[1, 2] == -0.3

    # The positional constructor takes the stored layout.
    @test ExogenousGraph(["x", "y"], [0.0 0.0; 1.0 0.0]) ==
          ExogenousGraph(; edges = [("x", "y", 1.0)])
end

@testset "ExogenousGraph: refusals" begin
    @test_throws IsEmptyError ExogenousGraph(; nodes = String[], A = zeros(0, 0))
    @test_throws ArgumentError ExogenousGraph(; nodes = ["a", ""], A = zeros(2, 2))
    @test_throws ArgumentError ExogenousGraph(; nodes = ["a", "a"], A = zeros(2, 2))
    @test_throws DimensionMismatch ExogenousGraph(; nodes = ["a", "b"], A = zeros(2, 3))
    @test_throws DimensionMismatch ExogenousGraph(; sources = ["a"], targets = ["b"],
                                                  A = zeros(2, 1))
    @test_throws ArgumentError ExogenousGraph(; sources = ["a", "a"], targets = ["b"],
                                              A = zeros(2, 1))
    @test_throws IsNonFiniteError ExogenousGraph(; nodes = ["a", "b"], A = [0 Inf; 0 0])
    @test_throws IsNonFiniteError ExogenousGraph(; edges = [("a", "b", NaN)])
    err = try
        ExogenousGraph(; edges = [("a", "b", 1.0), ("a", "b", 2.0)])
    catch e
        e
    end
    @test isa(err, ArgumentError) && occursin("\"a\" => \"b\"", err.msg)
    err = try
        ExogenousGraph(; nodes = ["a"], edges = [("a", "b", 1.0)])
    catch e
        e
    end
    @test isa(err, ArgumentError) && occursin("\"a\" => \"b\"", err.msg)
    @test_throws IsEmptyError ExogenousGraph(; edges = Tuple{String, String, Float64}[])
    # A set of keywords that is none of the three forms.
    for kw in ((;), (; A = zeros(1, 1)), (; nodes = ["a"]),
               (; nodes = ["a"], A = zeros(1, 1), edges = [("a", "a", 1.0)]),
               (; sources = ["a"], A = zeros(1, 1)))
        @test_throws ArgumentError ExogenousGraph(; kw...)
    end
    @test_throws DimensionMismatch ExogenousGraph(["a", "b"], zeros(3, 3))
end

@testset "ExogenousGraph: equality and the Graphs.jl conversions" begin
    d = ExogenousGraph(; nodes = ["S", "A", "B"], A = [0 0.4 0.6; 0 0 0; 0 0 0])
    s = ExogenousGraph(; edges = [("S", "A", 0.4), ("S", "B", 0.6)])
    @test d == s
    @test isequal(d, s)
    @test hash(d) == hash(s)
    @test d != ExogenousGraph(; nodes = ["S", "B", "A"], A = [0 0.6 0.4; 0 0 0; 0 0 0])
    D = PO.Graphs.SimpleDiGraph(s)
    @test PO.Graphs.nv(D) == 3 && PO.Graphs.ne(D) == 2
    @test PO.Graphs.has_edge(D, 1, 2) && !PO.Graphs.has_edge(D, 2, 1)
    W = PO.SimpleWeightedGraphs.SimpleWeightedDiGraph(d)
    @test PO.Graphs.weights(W)[1, 3] == 0.6
    @test PO.Graphs.has_edge(W, 1, 2) && !PO.Graphs.has_edge(W, 2, 1)
    # The weighted graph is mutable, and a change to it leaves the graph as it was.
    PO.Graphs.add_edge!(W, 2, 3, 9.0)
    @test adjacency(d)[2, 3] == 0.0
end

@testset "ExogenousGraphSeries: construction and refusals" begin
    s = ExogenousGraphSeries(; ts = [1, 4, 6], graphs = [g1, g2, nothing])
    @test s.ts == [1, 4, 6]
    @test s == ExogenousGraphSeries([1, 4, 6], [g1, g2, nothing])
    @test isequal(s, ExogenousGraphSeries([1, 4, 6], [g1, g2, nothing]))
    @test hash(s) == hash(ExogenousGraphSeries([1, 4, 6], [g1, g2, nothing]))
    @test s != ExogenousGraphSeries([1, 4, 6], [g1, g2, g3])
    @test ExogenousGraphSeries(; ts = [Date(2024, 1, 1)], graphs = [g1]).ts ==
          [Date(2024, 1, 1)]
    @test_throws IsEmptyError ExogenousGraphSeries(; ts = Int[], graphs = ExogenousGraph[])
    @test_throws ArgumentError ExogenousGraphSeries(; ts = Any[1, Date(2024, 1, 1)],
                                                    graphs = [g1, g2])
    @test_throws ArgumentError ExogenousGraphSeries(; ts = [true], graphs = [g1])
    @test_throws DomainError ExogenousGraphSeries(; ts = [0, 2], graphs = [g1, g2])
    @test_throws ArgumentError ExogenousGraphSeries(; ts = [3, 3], graphs = [g1, g2])
    @test_throws ArgumentError ExogenousGraphSeries(; ts = [3, 2], graphs = [g1, g2])
    @test_throws DimensionMismatch ExogenousGraphSeries(; ts = [1, 2], graphs = [g1])
end

@testset "The fields ng and G of the returns data" begin
    X = rand(StableRNG(1), 10, 2)
    s = ExogenousGraphSeries(; ts = [1, 4], graphs = [g1, g2])
    rd = ReturnsResult(; nx = ["A", "B"], X = X, ng = ["supply", "cust"], G = [s, g3])
    @test rd.ng == ["supply", "cust"]
    @test rd.G[2] === g3
    # A vector literal of a series and a static graph has the supertype of the family as its
    # element type, and the bound of `G` refuses any other element (#2395).
    @test eltype([s, g3]) == PO.AbstractExogenousGraph
    @test_throws TypeError ReturnsResult(; nx = ["A", "B"], X = X, ng = ["a"], G = [1.0])
    @test_throws TypeError PricesResult(;
                                        X = TimeArray(Date(2024, 1, 1):Day(1):Date(2024, 1,
                                                                                   10), X,
                                                      ["A", "B"]), ng = ["a"], G = Any[g1])
    @test_throws IsNothingError ReturnsResult(; nx = ["A", "B"], X = X, ng = ["supply"])
    @test_throws IsNothingError ReturnsResult(; nx = ["A", "B"], X = X, G = [g1])
    @test_throws ArgumentError ReturnsResult(; nx = ["A", "B"], X = X, ng = ["a", "a"],
                                             G = [g1, g2])
    @test_throws IsEmptyError ReturnsResult(; nx = ["A", "B"], X = X, ng = String[],
                                            G = ExogenousGraph[])
    @test_throws DimensionMismatch ReturnsResult(; nx = ["A", "B"], X = X, ng = ["a"],
                                                 G = [g1, g2])
    # An integer key after the last row.
    @test_throws DomainError ReturnsResult(; nx = ["A", "B"], X = X, ng = ["a"],
                                           G = [ExogenousGraphSeries([1, 11], [g1, g2])])
    # A date key with no timestamps, and a date type that does not compare with them.
    sd = ExogenousGraphSeries(; ts = [Date(2024, 1, 1)], graphs = [g1])
    @test_throws IsNothingError ReturnsResult(; nx = ["A", "B"], X = X, ng = ["a"],
                                              G = [sd])
    ts = collect(Date(2024, 1, 1):Day(1):Date(2024, 1, 10))
    @test ReturnsResult(; nx = ["A", "B"], X = X, ts = ts, ng = ["a"], G = [sd]).G[1] === sd
    st = ExogenousGraphSeries(; ts = [Time(1)], graphs = [g1])
    @test_throws ArgumentError ReturnsResult(; nx = ["A", "B"], X = X, ts = ts, ng = ["a"],
                                             G = [st])
    # A series with integer keys needs rows.
    @test_throws IsNothingError PO.check_exogenous_graphs(["a"], [s], nothing, nothing)
    # The asset view passes the graphs through.
    av = PO.port_opt_view(rd, 2:2)
    @test av.ng === rd.ng && av.G === rd.G
    # The price data checks the graphs on the rows of X.
    P = TimeArray(ts, 100 .+ X, ["A", "B"])
    @test PricesResult(; X = P, ng = ["a"], G = [sd]).G[1] === sd
    @test_throws DomainError PricesResult(; X = P, ng = ["a"],
                                          G = [ExogenousGraphSeries([11], [g1])])
end

@testset "The graph in force at the last row" begin
    X = rand(StableRNG(2), 10, 2)
    s = ExogenousGraphSeries(; ts = [2, 4, 5, 8], graphs = [g1, g2, g3, nothing])
    rd = ReturnsResult(; nx = ["A", "B"], X = X, ng = ["supply"], G = [s])
    @test graph_in_force(PO.port_opt_view(rd, 1:7, :)) == g3
    @test graph_in_force(PO.port_opt_view(rd, 1:4, :), "supply") == g2
    # A `nothing` key ends the graph before it.
    err = try
        graph_in_force(rd)
    catch e
        e
    end
    @test isa(err, ArgumentError) && occursin("the key 8 holds `nothing`", err.msg)
    # A row before the first key has no graph. A view on that row holds one `nothing` key.
    err = try
        graph_in_force(rd.ng, rd.G, nothing, 1, nothing)
    catch e
        e
    end
    @test isa(err, ArgumentError) && occursin("before the first key 2", err.msg)
    @test_throws ArgumentError graph_in_force(PO.port_opt_view(rd, 1:1, :))
    # The name of the graph.
    rd2 = ReturnsResult(; nx = ["A", "B"], X = X, ng = ["supply", "cust"], G = [g1, g2])
    @test graph_in_force(rd2, "cust") == g2
    @test_throws ArgumentError graph_in_force(rd2)
    @test_throws ArgumentError graph_in_force(rd2, "other")
    @test_throws IsNothingError graph_in_force(ReturnsResult(; nx = ["A", "B"], X = X))
    # Date keys read the last timestamp.
    ts = collect(Date(2024, 1, 1):Day(1):Date(2024, 1, 10))
    sd = ExogenousGraphSeries(; ts = [Date(2023, 12, 1), Date(2024, 1, 5)],
                              graphs = [g1, g2])
    rdd = ReturnsResult(; nx = ["A", "B"], X = X, ts = ts, ng = ["d"], G = [sd])
    @test graph_in_force(rdd) == g2
    @test graph_in_force(PO.port_opt_view(rdd, 1:4, :)) == g1
    P = TimeArray(ts, 100 .+ X, ["A", "B"])
    @test graph_in_force(PricesResult(; X = P, ng = ["d"], G = [sd])) == g2
end

@testset "The observation view of a series" begin
    s = ExogenousGraphSeries(; ts = [1, 4, 5, 8], graphs = [g1, g2, g3, nothing])
    rd = ReturnsResult(; nx = ["A", "B"], X = rand(StableRNG(3), 10, 2), ng = ["supply"],
                       G = [s])
    # The anchor becomes 1; keys 4 and 5 fall in the gap after row 3 and move to row 3 of the
    # view, where the last of them stays.
    v = PO.port_opt_view(rd, [2, 3, 6, 7], :).G[1]
    @test v.ts == [1, 3]
    @test v.graphs == [g1, g3]
    # A key after the window is dropped, so a training fold holds no graph of its test rows.
    @test PO.port_opt_view(rd, 1:3, :).G[1] == ExogenousGraphSeries([1], [g1])
    # No graph before the window and no key inside it: one `nothing` key.
    @test PO.graph_view(ExogenousGraphSeries([5], [g1]), 1:3, nothing) ==
          ExogenousGraphSeries([1], [nothing])
    # A key inside the window with no anchor keeps no anchor.
    @test PO.graph_view(ExogenousGraphSeries([5], [g1]), 3:7, nothing) ==
          ExogenousGraphSeries([3], [g1])
    # A Colon, and a static graph, pass through.
    @test PO.graph_view(s, :, nothing) === s
    @test PO.graph_view(g1, [3, 1], nothing) === g1
    # A logical mask of the view reads as its positions, and the graph view refuses a mask.
    mask = falses(10)
    mask[[2, 3, 6, 7]] .= true
    @test PO.port_opt_view(rd, mask, :).G[1] == v
    @test PO.port_opt_view(rd, mask, :).X == rd.X[mask, :]
    @test_throws ArgumentError PO.graph_view(s, mask, nothing)
    @test_throws ArgumentError PO.graph_view(s, [3, 1], nothing)
    @test_throws IsEmptyError PO.graph_view(s, Int[], nothing)
    # Date keys stay as they are.
    ts = collect(Date(2024, 1, 1):Day(1):Date(2024, 1, 10))
    sd = ExogenousGraphSeries(;
                              ts = [Date(2023, 12, 1), Date(2024, 1, 4), Date(2024, 1, 9)],
                              graphs = [g1, g2, g3])
    vd = PO.graph_view(sd, 3:6, ts)
    @test vd.ts == [Date(2023, 12, 1), Date(2024, 1, 4)]
    @test vd.graphs == [g1, g2]
    @test PO.graph_view(ExogenousGraphSeries([Date(2024, 2, 1)], [g1]), 3:6, ts) ==
          ExogenousGraphSeries([ts[3]], [nothing])
    # The graph in force at each row of a view is the graph at its parent row.
    rng = StableRNG(4)
    pool = [g1, g2, g3, nothing]
    for _ in 1:500
        T = rand(rng, 3:15)
        ks = sort!(randperm(rng, T)[1:rand(rng, 1:min(5, T))])
        si = ExogenousGraphSeries(ks,
                                  Union{Nothing, typeof(g1)}[rand(rng, pool) for _ in ks])
        rows = sort!(unique(rand(rng, 1:T, rand(rng, 1:T))))
        vs = PO.graph_view(si, rows, nothing)
        @test all(r -> isequal(as_of(vs, r), as_of(si, rows[r])), eachindex(rows))
    end
end

@testset "prices_to_returns and the views of the price data" begin
    ts = collect(Date(2024, 1, 1):Day(1):Date(2024, 1, 10))
    P = TimeArray(ts, 100 .+ cumsum(rand(StableRNG(5), 10, 2); dims = 1), ["A", "B"])
    si = ExogenousGraphSeries(; ts = [1, 4, 8], graphs = [g1, g2, g3])
    sd = ExogenousGraphSeries(; ts = [Date(2023, 12, 25), Date(2024, 1, 4)],
                              graphs = [g1, g2])
    pr = PricesResult(; X = P, ng = ["i", "d"], G = [si, sd])
    # A return row keeps the timestamp of its later price row.
    rr = prices_to_returns(pr)
    @test rr.ng == pr.ng
    @test rr.G[1].ts == [1, 3, 7]
    @test rr.G[2] == sd
    @test all(t -> isequal(as_of(rr.G[1], t), as_of(si, t + 1)), 1:9)
    rp = prices_to_returns(pr; padding = true)
    @test rp.G[1] == si
    # A timestamp view of the price data rebases the integer keys.
    pv = PO.port_opt_view(pr, 3:6)
    @test pv.G[1].ts == [1, 2]
    @test pv.G[2].ts == [Date(2023, 12, 25), Date(2024, 1, 4)]
    pa = PO.port_opt_view(pr, 3:6, ["B"])
    @test pa.G[1] == pv.G[1]
    @test PO.port_opt_view(pr, :, :) === pr
end

@testset "price_ingestion carries the graphs onto the emitted clock (#2395)" begin
    # 2024-01-01 is a Monday, so the 21 days are three weeks.
    ts = collect(Date(2024, 1, 1):Day(1):Date(2024, 1, 21))
    P = TimeArray(ts, 100 .+ cumsum(rand(StableRNG(6), 21, 2); dims = 1), ["A", "B"])
    si = ExogenousGraphSeries(; ts = [1, 9, 17], graphs = [g1, g2, nothing])
    sd = ExogenousGraphSeries(; ts = [Date(2024, 1, 3), Date(2024, 1, 10)],
                              graphs = [g1, g2])
    pr = PricesResult(; X = P, ng = ["i", "d", "s"], G = [si, sd, g3])
    # No collapse: the clock does not move, so the graphs do not change, and the returns
    # data reads them.
    p0 = price_ingestion(PriceIngestion(), pr)
    @test p0.ng == pr.ng
    @test p0.G[1] == si
    @test p0.G[2] == sd
    @test p0.G[3] === g3
    rd = prices_to_returns(p0)
    @test rd.ng == pr.ng
    @test graph_in_force(rd, "d") == g2
    @test graph_in_force(PO.port_opt_view(rd, 1:15, :), "i") == g2
    # The keywords of the table form.
    pk = price_ingestion(PriceIngestion(), P; ng = ["i"], G = [si])
    @test pk.ng == ["i"] && pk.G[1] == si
    # A weekly collapse named by the first day: each week takes the graph in force at the row
    # of its prices, its last day, also for a date key that falls inside the week.
    est = PriceIngestion(; collapse_args = (Dates.week, first, last))
    pw = price_ingestion(est, pr)
    tw = timestamp(pw.X)
    @test tw == [Date(2024, 1, 1), Date(2024, 1, 8), Date(2024, 1, 15)]
    @test pw.G[1] == ExogenousGraphSeries([1, 2, 3], [g1, g2, nothing])
    @test pw.G[2] == ExogenousGraphSeries(tw[1:2], [g1, g2])
    @test pw.G[3] === g3
    @test graph_in_force(PO.port_opt_view(pw, 1:1), "d") == g1
    @test prices_to_returns(pw).ng == pr.ng
    # An outer join adds two rows before the asset table: no integer key is in force there.
    F = TimeArray(Date(2023, 12, 30):Day(1):Date(2024, 1, 21), rand(StableRNG(7), 23, 1),
                  ["F1"])
    po = @test_logs (:warn,) match_mode = :any price_ingestion(PriceIngestion(;
                                                                              join_method = :outer),
                                                               PricesResult(; X = P, F = F,
                                                                            ng = ["i", "s"],
                                                                            G = [si, g3]))
    @test po.G[1] == ExogenousGraphSeries([3, 11, 19], [g1, g2, nothing])
    @test po.G[2] === g3
    # The graph in force at each emitted row is the graph in force at the row of its prices,
    # for random series of both kinds and random collapses.
    rng = StableRNG(8)
    pool = [g1, g2, g3, nothing]
    for _ in 1:200
        ks = sort!(randperm(rng, 21)[1:rand(rng, 1:5)])
        gs = Union{Nothing, typeof(g1)}[rand(rng, pool) for _ in ks]
        s1 = ExogenousGraphSeries(ks, gs)
        s2 = ExogenousGraphSeries(ts[ks] .- Day(rand(rng, 0:2)), gs)
        ca = rand(rng, ((), (Dates.week, first, last), (Dates.week, last, last)))
        e = PriceIngestion(; collapse_args = ca)
        out = price_ingestion(e, PricesResult(; X = P, ng = ["a", "b"], G = [s1, s2]))
        row = PO.ingestion_row_clock(ts, ca)
        to = timestamp(out.X)
        @test all(i -> isequal(as_of(out.G[1], i), as_of(s1, searchsortedlast(ts, row[i]))),
                  eachindex(row))
        @test all(i -> isequal(as_of(out.G[2], to[i]), as_of(s2, row[i])), eachindex(row))
    end
    # A date key after the last row is dropped, so no graph is in force at any emitted row.
    pl = price_ingestion(PriceIngestion(), P; ng = ["d"],
                         G = [ExogenousGraphSeries([Date(2024, 2, 1)], [g1])])
    @test pl.G[1] == ExogenousGraphSeries([ts[1]], [nothing])
    # The graphs are checked against the clock of the table the caller passed.
    @test_throws DomainError price_ingestion(PriceIngestion(), P; ng = ["i"],
                                             G = [ExogenousGraphSeries([22], [g1])])
    @test_throws IsNothingError price_ingestion(PriceIngestion(), P; ng = ["i"])
end

@testset "vcat_observations joins the graphs" begin
    ts = collect(Date(2024, 1, 1):Day(1):Date(2024, 1, 12))
    X = rand(StableRNG(6), 12, 2)
    si = ExogenousGraphSeries(; ts = [1, 4, 5, 9], graphs = [g1, g2, nothing, g3])
    sd = ExogenousGraphSeries(; ts = [Date(2023, 12, 1), Date(2024, 1, 7)],
                              graphs = [g1, g2])
    rd = ReturnsResult(; nx = ["A", "B"], X = X, ts = ts, ng = ["i", "d", "s"],
                       G = [si, sd, g3])
    for c in 1:11
        a = PO.port_opt_view(rd, 1:c, :)
        b = PO.port_opt_view(rd, (c + 1):12, :)
        j = PO.vcat_observations(a, b)
        @test j.G[3] === g3
        @test all(t -> isequal(as_of(j.G[1], t), as_of(si, t)), 1:12)
        @test all(t -> isequal(as_of(j.G[2], ts[t]), as_of(sd, ts[t])), 1:12)
    end
    # An anchor equal to the graph in force at the last row of `a` is dropped.
    a = PO.port_opt_view(rd, 1:5, :)
    b = PO.port_opt_view(rd, 6:12, :)
    @test PO.vcat_observations(a, b).G[1] == si
    # A block with no graph at its first row ends the graph of `a`.
    @test PO.graph_vcat(ExogenousGraphSeries([1], [g1]), ExogenousGraphSeries([2], [g2]),
                        "x", 3, nothing, 3, nothing) ==
          ExogenousGraphSeries([1, 4, 5], [g1, nothing, g2])
    # Refusals: another `ng`, two static graphs that differ, a static graph with a series,
    # keys of two kinds, and a date anchor of `b` that states another graph for a row of `a`.
    other = ReturnsResult(; nx = ["A", "B"], X = X[7:12, :], ts = ts[7:12],
                          ng = ["i", "d", "x"],
                          G = [ExogenousGraphSeries([1], [g1]), sd, g3])
    @test_throws ArgumentError PO.vcat_observations(a, other)
    @test_throws ArgumentError PO.graph_vcat(g1, g2, "x", 1, nothing, 1, nothing)
    @test PO.graph_vcat(g1, g1, "x", 1, nothing, 1, nothing) === g1
    @test_throws ArgumentError PO.graph_vcat(g1, si, "x", 1, nothing, 1, nothing)
    @test_throws ArgumentError PO.graph_vcat(si, sd, "x", 1, ts, 1, ts)
    @test_throws ArgumentError PO.graph_vcat(ExogenousGraphSeries([ts[1]], [g1]),
                                             ExogenousGraphSeries([ts[2]], [g2]), "x", 3,
                                             ts[1:3], 3, ts[4:6])
    # The price data joins its graphs too.
    P = TimeArray(ts, 100 .+ X, ["A", "B"])
    pr = PricesResult(; X = P, ng = ["i"], G = [si])
    pj = PO.vcat_observations(PO.port_opt_view(pr, 1:6), PO.port_opt_view(pr, 7:12))
    @test all(t -> isequal(as_of(pj.G[1], t), as_of(si, t)), 1:12)
end

@testset "The round trip through the edge table and the manifest" begin
    gd = ExogenousGraph(; nodes = ["S", "A", "C"], A = [0 0.4 0; 0 0 0; 0 0 0])
    sd = ExogenousGraphSeries(; ts = [Date(2024, 1, 1), Date(2024, 1, 5)],
                              graphs = [g1, g2])
    ng = ["static", "series", "dates"]
    G = [gd, ExogenousGraphSeries(; ts = [1, 3, 5], graphs = [g1, nothing, g2]), sd]
    df = graph_dataframe(ng, G)
    mf = graph_manifest(ng, G)
    @test names(df) == ["graph", "key", "source", "target", "weight"]
    @test names(mf) == ["kind", "graph", "key", "label"]
    @test DataFrames.nrow(df) == 1 + 1 + 2 + 1 + 2
    ngr, Gr = exogenous_graphs(df, mf)
    @test ngr == ng
    @test all(isequal.(Gr, G))
    @test isa(Gr[1].At, Matrix)
    # A reader that types each cell as text gives back the same graphs.
    text(d) = DataFrame([c => [ismissing(x) ? "" : string(x) for x in d[!, c]]
                         for c in names(d)])
    ngt, Gt = exogenous_graphs(text(df), text(mf))
    @test ngt == ng && all(isequal.(Gt, G))
    # The edge table alone: nodes in the order of first use, sparse storage, and a series for
    # each graph with keys. The `nothing` key and the isolated node are lost.
    ng1, G1 = exogenous_graphs(df)
    @test ng1 == ng
    @test G1[1].nodes == ["S", "A"] && isa(G1[1].At, SparseMatrixCSC)
    @test G1[2].ts == [1, 5]
    @test G1[3] == sd
    # The data types write their own graphs.
    rd = ReturnsResult(; nx = ["A", "B"], X = rand(StableRNG(7), 6, 2), ng = ng[1:2],
                       G = G[1:2])
    @test isequal(graph_dataframe(rd), graph_dataframe(ng[1:2], G[1:2]))
    @test isequal(graph_manifest(rd), graph_manifest(ng[1:2], G[1:2]))
    ts = collect(Date(2024, 1, 1):Day(1):Date(2024, 1, 6))
    pr = PricesResult(; X = TimeArray(ts, rand(StableRNG(8), 6, 2) .+ 1, ["A", "B"]),
                      ng = ng[1:2], G = G[1:2])
    @test isequal(graph_dataframe(pr), graph_dataframe(rd))
    @test isequal(graph_manifest(pr), graph_manifest(rd))
    @test_throws IsNothingError graph_dataframe(ReturnsResult(; nx = ["A"], X = ones(2, 1)))
    # Refusals of the reader.
    @test_throws ArgumentError exogenous_graphs(df[!, [:graph, :key, :source, :target]])
    @test_throws ArgumentError exogenous_graphs(df, mf[!, [:kind, :graph, :key]])
    mixed = copy(df)
    mixed.key = Union{Missing, Int, Date}[missing, 1, 1, 5, 5, missing, missing]
    @test_throws ArgumentError exogenous_graphs(mixed)
    @test_throws ArgumentError exogenous_graphs(df, mf[1:2, :])
    @test_throws ArgumentError exogenous_graphs(df, mf[[1, 3], :])
    @test_throws ArgumentError PO.graph_table_key("not a key")
    @test PO.graph_table_weight("0.5") == 0.5
    @test PO.graph_table_key(Int32(3)) === 3
    bad = copy(mf)
    bad[bad.kind .== "graph" .&& bad.graph .== "series", :label] .= "static"
    @test_throws ArgumentError exogenous_graphs(df, bad)
    none = copy(mf)
    none[none.kind .== "snapshot", :label] .= "none"
    @test_throws ArgumentError exogenous_graphs(df, none)
end

@testset "The requirements of a consumer at the entry of a user verb" begin
    gneg = ExogenousGraph(; edges = [("S", "A", 0.4), ("S", "B", -0.2)])
    sneg = ExogenousGraphSeries(; ts = [1, 6], graphs = [g1, gneg])
    rd = ReturnsResult(; nx = ["A", "B"], X = rand(StableRNG(9), 10, 2),
                       ng = ["supply", "cust"], G = [sneg, g2])
    @test PO.graph_requirement_breach(NonNegativeWeights(), gneg) == (1, 3)
    @test isnothing(PO.graph_requirement_breach(NonNegativeWeights(), g2))
    @test PO.graph_requirements(InverseVolatility()) == ()
    # The walk reaches a consumer in a vector, in a Pair and in a tuple, and names it.
    est = GraphHolder([1.0, 2.0], :x => [GraphReader("cust"), GraphReader("supply")])
    err = try
        PO.assert_graph_entry(est, rd)
    catch e
        e
    end
    @test isa(err, ArgumentError)
    @test occursin("`p[2]`", err.msg) && occursin("at the key 6", err.msg)
    @test occursin("\"S\" => \"B\" has the weight -0.2", err.msg)
    @test isnothing(PO.assert_graph_entry(GraphHolder((GraphReader(["cust"]),), nothing),
                                          rd))
    @test_throws ArgumentError PO.assert_graph_entry(GraphHolder(nothing,
                                                                 GraphReader(nothing)), rd)
    @test_throws IsNothingError PO.assert_graph_entry(GraphReader("cust"),
                                                      ReturnsResult(; nx = ["A"],
                                                                    X = ones(3, 1)))
    # The entry of a user verb reads every snapshot, so the bad snapshot of a later fold
    # fails before the first fold, also when the window of the first fold does not reach it.
    @test_throws ArgumentError PO.assert_graph_entry(GraphReader("supply"), rd)
    # A direct call checks the one snapshot that it reads.
    @test PO.consumer_graph(GraphReader("supply"), PO.port_opt_view(rd, 1:5, :)) == g1
    @test_throws ArgumentError PO.consumer_graph(GraphReader("supply"), rd)
    # The price data, and data with no graph, reach the walk too.
    ts = collect(Date(2024, 1, 1):Day(1):Date(2024, 1, 10))
    pr = PricesResult(; X = TimeArray(ts, rand(StableRNG(10), 10, 2) .+ 1, ["A", "B"]),
                      ng = ["supply"], G = [sneg])
    @test_throws ArgumentError PO.assert_graph_entry(GraphReader(nothing), pr)
    @test isnothing(PO.assert_graph_entry(GraphReader("cust"), rand(3, 2)))
end

@testset "The remaining branches of the views, the joins and the walk" begin
    # The view helpers that moved to the end of the directory, so the graph types load first.
    @test PO.label_positions(["A", "B"], 2, "asset") == 2:2
    @test_throws ArgumentError PO.label_positions(["A", "B"], true, "asset")
    @test PO.timestamp_positions(nothing, 3) == 3:3
    @test_throws ArgumentError PO.timestamp_positions(nothing, true)
    @test_throws ArgumentError PO.matched_row_indices(nothing, [1])
    # A Colon view keeps the graphs as they are.
    s = ExogenousGraphSeries(; ts = [1, 4], graphs = [g1, g2])
    rd = ReturnsResult(; nx = ["A", "B"], X = rand(StableRNG(12), 6, 2), ng = ["s"],
                       G = [s])
    @test PO.port_opt_view(rd, :, :).G === rd.G
    @test isnothing(PO.graphs_vcat(nothing, nothing, nothing, 1, nothing, 1, nothing))
    # A date block with no key at its first row ends the graph of the earlier block.
    ts = collect(Date(2024, 1, 1):Day(1):Date(2024, 1, 6))
    j = PO.graph_vcat(ExogenousGraphSeries([ts[1]], [g1]),
                      ExogenousGraphSeries([ts[5]], [g2]), "x", 3, ts[1:3], 3, ts[4:6])
    @test j == ExogenousGraphSeries([ts[1], ts[4], ts[5]], [g1, nothing, g2])
    # A consumer that states requirements and names no graph.
    @test_throws ArgumentError PO.graph_names(InverseVolatility())
    # The walk stops at a matrix of numbers.
    @test isnothing(PO.assert_graph_entry(GraphHolder(rand(2, 2), GraphReader("s")), rd))
end

@testset "The user verbs run on returns data with graphs" begin
    X = 0.01 .* randn(StableRNG(11), 60, 3)
    s = ExogenousGraphSeries(; ts = [1, 20, 40], graphs = [g1, g2, g3])
    rd = ReturnsResult(; nx = ["A", "B", "C"], X = X, ng = ["supply"], G = [s])
    plain = ReturnsResult(; nx = ["A", "B", "C"], X = X)
    @test optimise(InverseVolatility(), rd).w == optimise(InverseVolatility(), plain).w
    pred = cross_val_predict(InverseVolatility(), rd, KFold(; n = 3))
    ref = cross_val_predict(InverseVolatility(), plain, KFold(; n = 3))
    @test all(k -> pred.pred[k].res.w == ref.pred[k].res.w, eachindex(ref.pred))
end
