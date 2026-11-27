#=
The network readers over an Exogenous Graph (#2295, map #2162), the decision of #2289.

`ExogenousNetwork` reads a named graph of the returns data as a network of the assets. The
graph below has two nodes that are not assets, a reciprocal pair, a self-loop, a cycle and an
asset that is not a node. The separations are found by hand, and the centralities are the
calls of Graphs.jl on a graph that the test builds by itself.
=#
using Test, PortfolioOptimisers, LinearAlgebra, SparseArrays, StableRNGs, Statistics, Dates,
      Clarabel, HiGHS, Logging
using PortfolioOptimisers.Graphs: Graphs, AbstractGraph, SimpleGraph, SimpleDiGraph,
                                  add_edge!, vertices, nv, is_directed, gdistances,
                                  floyd_warshall_shortest_paths, degree_centrality,
                                  indegree_centrality, outdegree_centrality, pagerank,
                                  katz_centrality, betweenness_centrality,
                                  closeness_centrality, stress_centrality,
                                  radiality_centrality, eigenvector_centrality
using PortfolioOptimisers.SimpleWeightedGraphs: SimpleWeightedGraph, SimpleWeightedDiGraph
const PO = PortfolioOptimisers

const XN_NX = ["A", "B", "C", "D", "E"]
const XN_NODES = ["S", "D", "A", "C", "T", "B"]
const XN_EDGES = [("A", "B", 0.5), ("B", "A", 0.25), ("A", "A", 3.0), ("A", "S", 1.0),
                  ("S", "C", 0.5), ("C", "T", 0.25), ("T", "D", 0.5), ("D", "A", 0.25)]
const XN_X = randn(StableRNG(2295), 60, 5) .* 0.02 .+ 0.001
const XN_TS = Date(2024, 1, 1) .+ Day.(0:59)
const XN_DIRS = (OutEdges(), InEdges(), BothEdges(), AnyEdges())
const XN_QUIET = NullLogger()

function xn_matrix(T = Float64; edges = XN_EDGES)
    A = zeros(T, length(XN_NODES), length(XN_NODES))
    for (s, t, w) in edges
        A[findfirst(==(s), XN_NODES), findfirst(==(t), XN_NODES)] = w
    end
    return A
end
xn_graph(A = xn_matrix()) = ExogenousGraph(; nodes = XN_NODES, A = A)
function xn_rd(g = xn_graph(); X = XN_X, nx = XN_NX, kwargs...)
    return ReturnsResult(; nx = nx, X = X, ng = ["g"], G = [g], kwargs...)
end
const XN_RD = xn_rd()
xn_quiet(f) = with_logger(f, XN_QUIET)
function xn_sep(sep, rd = XN_RD; kwargs...)
    return xn_quiet() do
        return separation_matrix(sep, ExogenousNetwork(; kwargs...), rd.X; rd = rd)
    end
end

const XN_I = typemax(Int)
const XN_HOPS = Dict((OutEdges, AllNodes) =>
                         [0 1 2 4 XN_I; 1 0 3 5 XN_I; 3 4 0 2 XN_I; 1 2 3 0 XN_I;
                          XN_I XN_I XN_I XN_I 0],
                     (OutEdges, AssetNodes) =>
                         [0 1 XN_I XN_I XN_I; 1 0 XN_I XN_I XN_I; XN_I XN_I 0 XN_I XN_I;
                          1 2 XN_I 0 XN_I; XN_I XN_I XN_I XN_I 0],
                     (AnyEdges, AllNodes) =>
                         [0 1 2 1 XN_I; 1 0 3 2 XN_I; 2 3 0 2 XN_I; 1 2 2 0 XN_I;
                          XN_I XN_I XN_I XN_I 0],
                     (AnyEdges, AssetNodes) =>
                         [0 1 XN_I 1 XN_I; 1 0 XN_I 2 XN_I; XN_I XN_I 0 XN_I XN_I;
                          1 2 XN_I 0 XN_I; XN_I XN_I XN_I XN_I 0])
const XN_PATHS = Dict((OutEdges, AllNodes) =>
                          [0 2 3 9 Inf; 4 0 7 13 Inf; 10 12 0 6 Inf; 4 6 7 0 Inf;
                           Inf Inf Inf Inf 0],
                      (OutEdges, AssetNodes) =>
                          [0 2 Inf Inf Inf; 4 0 Inf Inf Inf; Inf Inf 0 Inf Inf;
                           4 6 Inf 0 Inf; Inf Inf Inf Inf 0],
                      (AnyEdges, AllNodes) =>
                          [0 2 3 4 Inf; 2 0 5 6 Inf; 3 5 0 6 Inf; 4 6 6 0 Inf;
                           Inf Inf Inf Inf 0],
                      (AnyEdges, AssetNodes) =>
                          [0 2 Inf 4 Inf; 2 0 Inf 6 Inf; Inf Inf 0 Inf Inf; 4 6 Inf 0 Inf;
                           Inf Inf Inf Inf 0])
for R in (XN_HOPS, XN_PATHS), s in (AllNodes, AssetNodes)
    o = R[(OutEdges, s)]
    R[(InEdges, s)] = permutedims(o)
    R[(BothEdges, s)] = min.(o, permutedims(o))
end

@testset "construction" begin
    n = ExogenousNetwork()
    @test isnothing(n.ng) && n.dir === AnyEdges() && n.scope === AllNodes()
    @test isnothing(n.nrm) && n.len == ReciprocalLength(; power = 1) && !n.strict
    @test n.sep == HopCount() && n isa PO.AbstractNetworkEstimator
    @test PO.graph_requirements(n) == (NonNegativeWeights(),)
    gu = GraphUnion(; ng = ["a", "b"])
    @test PO.graph_names(ExogenousNetwork(; ng = gu)) === gu
    @test isnothing(PO.graph_names(n)) && isnothing(PO.returns_source(n))
    @test_throws PO.IsEmptyError ExogenousNetwork(; ng = "")
    @test_throws TypeError ExogenousNetwork(; ng = ["a", "b"])
    @test_throws TypeError ExogenousNetwork(; len = 1)
    @test_throws DomainError ReciprocalLength(; power = -1)
    @test_throws DomainError ReciprocalLength(; power = Inf)
    # A view of the assets and a factory return the network: the graph is found by names.
    @test PO.port_opt_view(n, [1, 2]) === n && factory(n) === n
    # A reader of a network adds no frame when its network reads no returns.
    for x in (CentralityEstimator(; pl = n), SemiDefinitePhylogenyEstimator(; pl = n),
              IntegerPhylogenyEstimator(; pl = n),
              CentralityConstraint(; A = CentralityEstimator(; pl = n)))
        f = PO.reader_frames(x, "x", true)
        @test !f.excess && isempty(f.raw)
        @test !PO.plain_graph_reader(x)
        @test isnothing(PO.feature_readable_mask(x, nothing, XN_RD))
    end
end

@testset "separations by hand: $(nameof(T)), $k" for T in (Float32, Float64),
                                                     k in (:dense, :sparse)

    A = k === :dense ? xn_matrix(T) : sparse(xn_matrix(T))
    rd = xn_rd(xn_graph(A); X = T.(XN_X))
    for dir in XN_DIRS, scope in (AllNodes(), AssetNodes())
        key = (typeof(dir), typeof(scope))
        h = xn_sep(HopCount(), rd; dir = dir, scope = scope)
        @test h == XN_HOPS[key] && eltype(h) == Int
        d = xn_sep(PathLength(), rd; dir = dir, scope = scope)
        @test d == XN_PATHS[key] && eltype(d) == T
        # A unit length counts the edges, and a stored length is a length.
        u = xn_sep(PathLength(), rd; dir = dir, scope = scope,
                   len = ReciprocalLength(; power = 0))
        @test u == map(x -> x == XN_I ? Inf : x, h)
        inv_g = ExogenousGraph(; nodes = XN_NODES,
                               A = map(x -> iszero(x) ? x : inv(x), xn_matrix(T)))
        @test xn_sep(PathLength(), xn_rd(inv_g); dir = dir, scope = scope, len = nothing) ==
              XN_PATHS[key]
        # A self-loop changes nothing, under a normalisation too.
        g0 = xn_graph(xn_matrix(T; edges = XN_EDGES[[1, 2, 4, 5, 6, 7, 8]]))
        for nrm in (nothing, WeightShare())
            kw = (; dir = dir, scope = scope, nrm = nrm)
            @test xn_sep(PathLength(), xn_rd(g0); kw...) ==
                  xn_sep(PathLength(), xn_rd(xn_graph(xn_matrix(T))); kw...)
        end
    end
end

# The largest product of weights over the simple paths from `s`, by enumeration.
function xn_best_product!(best, M, s, v, p, seen)
    best[v] = max(best[v], p)
    for u in axes(M, 1)
        if u != v && !(u in seen) && !iszero(M[v, u])
            xn_best_product!(best, M, s, u, p * M[v, u], [seen; u])
        end
    end
    return best
end
function xn_products(M)
    return reduce(hcat,
                  [xn_best_product!(zeros(size(M, 1)), M, s, s, 1.0, [s])
                   for s in axes(M, 1)])'
end

@testset "lengths and normalisation" begin
    A = xn_matrix()
    A[diagind(A)] .= 0
    share(M) = M ./ max.(sum(M; dims = 2), eps())
    va = [findfirst(==(n), XN_NODES) for n in XN_NX[1:4]]
    P = xn_products(share(A))[va, va]
    d = xn_sep(PathLength(); dir = OutEdges(), nrm = WeightShare(),
               len = NegativeLogLength())
    @test exp.(-d[1:4, 1:4]) ≈ P
    # The in-edges take their own shares.
    Pin = xn_products(share(permutedims(A)))[va, va]
    din = xn_sep(PathLength(); dir = InEdges(), nrm = WeightShare(),
                 len = NegativeLogLength())
    @test exp.(-din[1:4, 1:4]) ≈ Pin
    # Over both directions, the pair takes the smallest separation, so it is symmetric.
    dbo = xn_sep(PathLength(); dir = BothEdges(), nrm = WeightShare(),
                 len = NegativeLogLength())
    @test dbo == min.(d, din, d', din')
    # Under AnyEdges, the shares are of the out-edges, and a pair takes its shorter length.
    W = share(A)
    U = max.(W, permutedims(W))
    Pany = xn_products(U)[va, va]
    dany = xn_sep(PathLength(); nrm = WeightShare(), len = NegativeLogLength())
    @test exp.(-dany[1:4, 1:4]) ≈ Pany && issymmetric(dany)
    # A union with a normalisation acts along the same direction.
    for dir in (OutEdges(), InEdges(), AnyEdges())
        g = xn_graph(A)
        gu = GraphUnion(; ng = ["g"], nrm = WeightShare())
        @test xn_sep(PathLength(), xn_rd(g); ng = gu, dir = dir) ≈
              xn_sep(PathLength(), xn_rd(g); ng = "g", dir = dir, nrm = WeightShare())
    end
    # A share of one has the length zero, which keeps its edge.
    @test xn_sep(PathLength(); dir = OutEdges(), nrm = WeightShare(),
                 len = NegativeLogLength())[3, 4] == 2 * nextfloat(0.0)
    @test_throws DomainError xn_sep(PathLength(), xn_rd(xn_graph(2 * xn_matrix()));
                                    len = NegativeLogLength())
    # The exponent of Opsahl, Agneessens and Skvoretz, on a single edge.
    @test xn_sep(PathLength(); dir = OutEdges(), len = ReciprocalLength(; power = 2))[1,
                                                                                      2] ==
          4
    # A product of shares that the rounding sums in two orders gives a symmetric matrix.
    R = rand(StableRNG(1), 6, 6)
    for dir in (AnyEdges(), BothEdges())
        dr = xn_sep(PathLength(), xn_rd(xn_graph(R)); dir = dir)
        @test issymmetric(dr)
        pl = ExogenousNetwork(; dir = dir, sep = PathLength(; dmax = FractionOf(0.5)))
        @test issymmetric(xn_quiet(() -> phylogeny_matrix(pl, xn_rd(xn_graph(R)))).X)
    end
end

@testset "budgets read the pairs of assets" begin
    # The largest finite separation of two assets, not of two nodes.
    nte = ExogenousNetwork(; dir = OutEdges(), sep = PathLength())
    d = xn_sep(PathLength(); dir = OutEdges())
    @test separation_budget(PathLength(), nte, d) == 13
    full = xn_quiet(() -> floyd_warshall_shortest_paths(first(PO.separation_graph(PathLength(),
                                                                                  nte, XN_X;
                                                                                  rd = XN_RD).gs)).dists)
    @test maximum(filter(isfinite, full)) > 13
    # A fraction is the quantile of the separations of the assets.
    h = xn_sep(HopCount(); dir = OutEdges())
    v = [h[i, j] for i in 1:5, j in 1:5 if i != j && h[i, j] < XN_I]
    r = xn_quiet(() -> resolve_separation(HopCount(; n = FractionOf(0.7)), nte, XN_X;
                                          rd = XN_RD))
    @test r.n.val == round(Int, quantile(v, 0.7))
    p = [d[i, j] for i in 1:5, j in 1:5 if i != j && isfinite(d[i, j])]
    r = xn_quiet(() -> resolve_separation(PathLength(; dmax = FractionOf(0.7)), nte, XN_X;
                                          rd = XN_RD))
    @test r.dmax ≈ quantile(p, 0.7)
    # A rule receives the whole graph: six nodes and the asset that is not a node.
    for dir in XN_DIRS
        seen = Ref{Any}()
        nt = ExogenousNetwork(; dir = dir)
        f(n, X, g; kwargs...) = (seen[] = g; 2)
        @test xn_quiet(() -> resolve_separation(HopCount(; n = f), nt, XN_X; rd = XN_RD)).n.val ==
              2
        @test seen[] isa AbstractGraph && nv(seen[]) == 7
        @test is_directed(seen[]) == (dir !== AnyEdges())
        @test xn_quiet(() -> resolve_separation(PathLength(; dmax = f), nt, XN_X;
                                                rd = XN_RD)).dmax == 2
        @test seen[] isa AbstractGraph && nv(seen[]) == 7
    end
    @test_throws ArgumentError xn_quiet(() -> resolve_separation(HopCount(;
                                                                          n = FractionOf(0.5)),
                                                                 nte, XN_X[:, 3:3];
                                                                 rd = xn_rd(; nx = ["C"],
                                                                            X = XN_X[:,
                                                                                     3:3])))
end

@testset "the phylogeny matrix and its readers" begin
    for dir in (AnyEdges(), BothEdges()), scope in (AllNodes(), AssetNodes())
        for (sep, R, b) in ((HopCount(; n = 2), XN_HOPS, 2), (HopCount(), XN_HOPS, 1),
                            (PathLength(; dmax = 5), XN_PATHS, 5))
            d = R[(typeof(dir), typeof(scope))]
            P = [Int(i != j && d[i, j] <= b) for i in 1:5, j in 1:5]
            pl = ExogenousNetwork(; dir = dir, scope = scope, sep = sep)
            got = xn_quiet(() -> phylogeny_matrix(pl, XN_RD))
            @test got.X == P && eltype(got.X) == Int
            # A prior result with the returns data beside it, and the transposed matrix.
            pr = prior(EmpiricalPrior(), XN_RD)
            @test xn_quiet(() -> phylogeny_matrix(pl, pr; rd = XN_RD)).X == P
            @test xn_quiet(() -> phylogeny_matrix(pl, permutedims(XN_X); dims = 2,
                                                  rd = XN_RD)).X == P
            w = [0.1, 0.2, 0.3, 0.25, 0.15]
            @test xn_quiet(() -> asset_phylogeny(pl, w, XN_RD)) ≈
                  dot(P, w * w') / sum(w * w')
            @test xn_quiet(() -> phylogeny_constraints(IntegerPhylogenyEstimator(; pl = pl),
                                                       pr; rd = XN_RD)).A ==
                  unique(P + I; dims = 1)
            @test xn_quiet(() -> phylogeny_constraints(SemiDefinitePhylogenyEstimator(;
                                                                                      pl = pl),
                                                       pr; rd = XN_RD)).A == P
        end
    end
    for dir in XN_DIRS, sep in (HopCount(; n = 2), PathLength())
        pl = ExogenousNetwork(; dir = dir, sep = sep)
        d = xn_sep(sep; dir = dir)
        b = separation_budget(sep, pl, d)
        Z = [PO.is_related(sep, x, b) ? PO.separation_decay(LinearDecay(), x, b) : 0.0
             for x in d]
        @test xn_quiet(() -> phylogeny_features(Proximity(), pl, XN_X; rd = XN_RD)) ≈ Z
        # The panel passes the returns data, from either slot, and labels its axis.
        ape = PhylogenyPanel(; pl = pl)
        for (pr, rd) in ((nothing, XN_RD), (XN_RD, nothing))
            f = only(xn_quiet(() -> PO.asset_panel(ape, pr, rd, XN_X)).pf)
            @test f.vals ≈ Z
        end
        @test only(xn_quiet(() -> PO.asset_panel(ape, nothing, XN_RD, XN_X)).pf).labels ==
              XN_NX
    end
end

# The graph that a centrality reads, built without the library.
function xn_plain(dir, scope, names; n = 1)
    g = SimpleDiGraph(length(names))
    for (s, t, _) in XN_EDGES
        i, j = findfirst(==(s), names), findfirst(==(t), names)
        if i != j && !isnothing(i) && !isnothing(j)
            dir isa InEdges ? add_edge!(g, j, i) : add_edge!(g, i, j)
        end
    end
    g = dir isa AnyEdges ? SimpleGraph(g) : g
    c = dir isa AnyEdges ? SimpleGraph(nv(g)) : SimpleDiGraph(nv(g))
    for v in vertices(g), u in vertices(g)
        if u != v && gdistances(g, v)[u] <= n
            add_edge!(c, v, u)
        end
    end
    return c
end
function xn_weighted(dir, names, f, op)
    n = length(names)
    M = zeros(n, n)
    for (s, t, w) in XN_EDGES
        i, j = findfirst(==(s), names), findfirst(==(t), names)
        if i != j && !isnothing(i) && !isnothing(j)
            M[i, j] = f(w)
        end
    end
    M = dir isa InEdges ? permutedims(M) : M
    if dir isa AnyEdges
        M = [iszero(M[i, j]) ? M[j, i] : iszero(M[j, i]) ? M[i, j] : op(M[i, j], M[j, i])
             for i in 1:n, j in 1:n]
        return SimpleWeightedGraph(M)
    end
    return SimpleWeightedDiGraph(M)
end

@testset "centralities: $(nameof(typeof(dir))), $(nameof(typeof(scope)))" for dir in
                                                                              (OutEdges(),
                                                                               InEdges(),
                                                                               AnyEdges()),
                                                                              scope in
                                                                              (AllNodes(),
                                                                               AssetNodes())

    names = scope isa AllNodes ? [XN_NODES; "E"] : XN_NX
    va = [findfirst(==(n), names) for n in XN_NX]
    function score(ct, rd = XN_RD; kwargs...)
        return xn_quiet() do
            return centrality_vector(CentralityEstimator(;
                                                         pl = ExogenousNetwork(; dir = dir,
                                                                               scope = scope,
                                                                               kwargs...),
                                                         ct = ct), rd).X
        end
    end
    for n in (1, 2)
        g = xn_plain(dir, scope, names; n = n)
        for (ct, f) in ((DegreeCentrality(), degree_centrality),
                        (DegreeCentrality(; kind = 1), indegree_centrality),
                        (DegreeCentrality(; kind = 2), outdegree_centrality),
                        (Pagerank(), g -> pagerank(g, 0.85, 100, 1e-6)),
                        (KatzCentrality(), g -> katz_centrality(g, 0.3)),
                        (BetweennessCentrality(; ov = TopologyOnly()), betweenness_centrality),
                        (ClosenessCentrality(; ov = TopologyOnly()), closeness_centrality),
                        (StressCentrality(; ov = TopologyOnly()), stress_centrality),
                        (RadialityCentrality(; ov = TopologyOnly()), radiality_centrality))
            @test isequal(score(ct; sep = HopCount(; n = n)), f(g)[va])
        end
        # A path budget in unit lengths gives the same closure.
        @test score(DegreeCentrality(); sep = PathLength(; dmax = n),
                    len = ReciprocalLength(; power = 0)) == degree_centrality(g)[va]
    end
    for (len, f) in ((ReciprocalLength(), inv), (ReciprocalLength(; power = 2), w -> w^-2),
                     (NegativeLogLength(), w -> isone(w) ? nextfloat(0.0) : -log(w)),
                     (nothing, identity))
        g = xn_weighted(dir, names, f, min)
        for (ct, h) in ((BetweennessCentrality(), betweenness_centrality),
                        (ClosenessCentrality(), closeness_centrality),
                        (StressCentrality(), stress_centrality),
                        (RadialityCentrality(), radiality_centrality))
            @test isequal(score(ct; len = len), h(g)[va])
        end
    end
    if scope isa AllNodes
        ev = eigenvector_centrality(xn_weighted(dir, names, identity, +))[va]
        @test score(EigenvectorCentrality()) ≈ ev atol = 1e-8
        # Stored lengths state no strength, so the algorithm reads the plain graph.
        pv = eigenvector_centrality(xn_plain(dir, scope, names))[va]
        @test score(EigenvectorCentrality(); len = nothing) ≈ pv atol = 1e-8
        @test score(EigenvectorCentrality(; ov = TopologyOnly())) ≈ pv atol = 1e-8
    end
    # The portfolio average, and the estimator form with the three arguments.
    w = [0.1, 0.2, 0.3, 0.25, 0.15]
    cte = CentralityEstimator(; pl = ExogenousNetwork(; dir = dir, scope = scope))
    @test xn_quiet(() -> average_centrality(cte, w, XN_RD)) ≈
          dot(score(DegreeCentrality()), w)
    @test xn_quiet(() -> centrality_vector(cte.pl, cte.ct, XN_RD)).X ==
          score(DegreeCentrality())
end

@testset "a length of a user" begin
    @eval struct XNHalfLength <: PO.AbstractEdgeLength end
    @eval PO.edge_length(::XNHalfLength, w::Number) = inv(2 * w)
    @test @eval(xn_sep(PathLength(); dir = OutEdges(), len = XNHalfLength())) ==
          XN_PATHS[(OutEdges, AllNodes)] ./ 2
    # A scope of a user, the nodes of the graph alone with the assets that are nodes.
    @eval struct XNFirstNodes <: PO.AbstractNetworkScope end
    @eval function PO.scope_vertices(::XNFirstNodes, nodes, nx)
        n, vn, va = PO.scope_vertices(AllNodes(), nodes, nx)
        return n, [nodes[k] == "S" ? 0 : vn[k] for k in eachindex(nodes)], va
    end
    @test @eval(xn_sep(HopCount(); dir = OutEdges(), scope = XNFirstNodes()))[1, 3] == XN_I
end

@testset "the graph in force" begin
    # An asset that is not a node.
    msg = "the assets [\"E\"] are not nodes of the Exogenous Graph \"g\""
    @test_logs (:warn, Regex("\\Q$msg\\E")) match_mode = :any phylogeny_matrix(ExogenousNetwork(),
                                                                               XN_RD)
    for scope in (AllNodes(), AssetNodes())
        e = try
            phylogeny_matrix(ExogenousNetwork(; strict = true, scope = scope), XN_RD)
        catch err
            err
        end
        @test e isa ArgumentError && occursin(msg, e.msg)
    end
    rd4 = xn_rd(; X = XN_X[:, 1:4], nx = XN_NX[1:4])
    @test_logs phylogeny_matrix(ExogenousNetwork(; strict = true), rd4)
    many = xn_rd(; X = randn(StableRNG(3), 10, 8), nx = string.("Z", 1:8))
    e = try
        phylogeny_matrix(ExogenousNetwork(; strict = true), many)
    catch err
        err
    end
    @test occursin("and 3 more", e.msg)
    # An asset that the prior drops changes no path over all the nodes.
    v = PO.port_opt_view(XN_RD, [2, 3, 4])
    for dir in XN_DIRS
        @test xn_sep(HopCount(), v; dir = dir) == XN_HOPS[(typeof(dir), AllNodes)][2:4, 2:4]
    end
    @test all(==(XN_I), xn_sep(HopCount(), v; scope = AssetNodes())[[2, 3, 4, 6, 7, 8]])
    # A union of two graphs, and the graph of the last row of a series.
    g1 = ExogenousGraph(; edges = [("A", "B", 1.0)])
    g2 = ExogenousGraph(; edges = [("B", "C", 1.0), ("D", "E", 1.0)])
    rd2 = ReturnsResult(; nx = XN_NX, X = XN_X, ng = ["a", "b"], G = [g1, g2])
    u = ExogenousNetwork(; ng = GraphUnion(; ng = ["a", "b"]), sep = HopCount(; n = 2))
    Pu = xn_quiet(() -> phylogeny_matrix(u, rd2)).X
    @test Pu[1, 2] == Pu[2, 3] == Pu[1, 3] == Pu[4, 5] == 1 && sum(Pu) == 8
    @test xn_quiet(() -> phylogeny_matrix(ExogenousNetwork(; ng = "a"), rd2)).X[1, 2] == 1
    @test_throws ArgumentError phylogeny_matrix(ExogenousNetwork(), rd2)
    s = xn_rd(ExogenousGraphSeries(; ts = XN_TS[[1, 35]], graphs = [g1, g2]); ts = XN_TS)
    early = PO.port_opt_view(s, 1:20, :)
    @test xn_quiet(() -> phylogeny_matrix(ExogenousNetwork(), early)).X[1, 2] == 1
    @test xn_quiet(() -> phylogeny_matrix(ExogenousNetwork(), s)).X[1, 2] == 0
    late = xn_rd(ExogenousGraphSeries(; ts = XN_TS[[35]], graphs = [g2]); ts = XN_TS)
    @test_throws ArgumentError phylogeny_matrix(ExogenousNetwork(),
                                                PO.port_opt_view(late, 1:20, :))
    # A gap in the returns changes nothing.
    Xg = copy(XN_X)
    Xg[3, 2] = NaN
    @test xn_quiet(() -> phylogeny_matrix(ExogenousNetwork(), xn_rd(; X = Xg))).X ==
          xn_quiet(() -> phylogeny_matrix(ExogenousNetwork(), XN_RD)).X
end

@testset "refusals" begin
    # No returns data, no asset names, and a matrix of another size.
    pr = prior(EmpiricalPrior(), XN_RD)
    for x in (pr, XN_X)
        @test_throws PO.IsNothingError phylogeny_matrix(ExogenousNetwork(), x)
        @test_throws PO.IsNothingError centrality_vector(CentralityEstimator(;
                                                                             pl = ExogenousNetwork()),
                                                         x)
    end
    @test_throws PO.IsNothingError phylogeny_matrix(ExogenousNetwork(),
                                                    ReturnsResult(; X = XN_X, ng = ["g"],
                                                                  G = [xn_graph()]))
    @test_throws DimensionMismatch phylogeny_matrix(ExogenousNetwork(), XN_X[:, 1:3];
                                                    rd = XN_RD)
    # The directions, in the constructor and at a direct call.
    for dir in (OutEdges(), InEdges())
        n = ExogenousNetwork(; dir = dir)
        @test_throws ArgumentError SemiDefinitePhylogenyEstimator(; pl = n)
        @test_throws ArgumentError IntegerPhylogenyEstimator(; pl = n)
        @test_throws ArgumentError phylogeny_matrix(n, XN_RD)
        @test_throws ArgumentError asset_phylogeny(n, ones(5) / 5, XN_RD)
        @test CentralityEstimator(; pl = n).pl === n && PhylogenyPanel(; pl = n).pl === n
    end
    b = ExogenousNetwork(; dir = BothEdges())
    @test_throws ArgumentError CentralityEstimator(; pl = b)
    for ct in (DegreeCentrality(), ClosenessCentrality(), EigenvectorCentrality())
        @test_throws ArgumentError centrality_vector(b, ct, XN_RD)
    end
    @test SemiDefinitePhylogenyEstimator(; pl = b).pl === b
    @test_throws ArgumentError NetworkClustersEstimator(; nte = ExogenousNetwork())
    # The outer problem and the factor phylogeny, at any depth.
    n = ExogenousNetwork()
    plain = InverseVolatility()
    cle(ape) = ClustersEstimator(; de = FeatureDistance(; ape = ape))
    hrp(ape) = HierarchicalRiskParity(; ser = HierarchicalSeriation(; cle = cle(ape)))
    slv = Solver(; name = :c, solver = Clarabel.Optimizer, settings = "verbose" => false)
    mr(; kwargs...) = MeanRisk(; opt = JuMPOptimiser(; slv = slv, kwargs...))
    for opto in (hrp(PhylogenyPanel(; pl = n)), hrp(WalkProfilePanel()),
                 mr(; ple = SemiDefinitePhylogenyEstimator(; pl = n)),
                 mr(; cte = CentralityConstraint(; A = CentralityEstimator(; pl = n))))
        @test_throws ArgumentError NestedClustered(; opti = plain, opto = opto)
        @test_throws ArgumentError Stacking(; opti = [plain], opto = opto)
        @test Stacking(; opti = [opto], opto = plain) isa Stacking
    end
    for ple in (SemiDefinitePhylogenyEstimator(; pl = n),
                SemiDefinitePhylogenyEstimator(; pl = cle(PhylogenyPanel(; pl = n))),
                [IntegerPhylogenyEstimator(; pl = n)])
        @test_throws ArgumentError FactorRiskContribution(;
                                                          opt = JuMPOptimiser(; slv = slv),
                                                          frc_ple = ple)
    end
    # A negative weight, before fold 1 although fold 1 does not read it.
    neg = ExogenousGraph(; edges = [("A", "B", -1.0)])
    s = xn_rd(ExogenousGraphSeries(; ts = XN_TS[[1, 55]], graphs = [xn_graph(), neg]);
              ts = XN_TS)
    for opt in (hrp(PhylogenyPanel(; pl = n)),
                mr(; ple = SemiDefinitePhylogenyEstimator(; pl = n)))
        e = try
            cross_val_predict(opt, s, IndexWalkForward(20, 10))
        catch err
            err
        end
        @test e isa ArgumentError &&
              occursin("each edge weight must be non-negative", e.msg)
    end
    @test_throws ArgumentError phylogeny_matrix(n, xn_rd(neg))
end

const XN_SLV = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                      check_sol = (; allow_local = true, allow_almost = true),
                      settings = "verbose" => false)
const XN_MIP = Solver(; name = :highs, solver = HiGHS.Optimizer,
                      check_sol = (; allow_local = true, allow_almost = true),
                      settings = "log_to_console" => false)

@testset "fits in a walk-forward" begin
    g1 = ExogenousGraph(; nodes = XN_NX, A = Float64[0 1 1 0 0; zeros(4, 5)])
    g2 = ExogenousGraph(; nodes = XN_NX, A = [zeros(3, 5); 0 0 0 0 1; zeros(1, 5)])
    rd = xn_rd(ExogenousGraphSeries(; ts = XN_TS[[1, 35]], graphs = [g1, g2]); ts = XN_TS)
    cv = IndexWalkForward(20, 10)
    # Fold k trains on the rows up to 10k + 10.
    gs = [10k + 10 >= 35 ? g2 : g1 for k in 1:4]
    n = ExogenousNetwork(; strict = true)
    pred = cross_val_predict(MeanRisk(;
                                      opt = JuMPOptimiser(; slv = XN_SLV,
                                                          ple = SemiDefinitePhylogenyEstimator(;
                                                                                               pl = n))),
                             rd, cv)
    P(g) = phylogeny_matrix(n, xn_rd(g)).X
    @test [f.res.plr.A for f in pred.pred] == P.(gs)
    pred = cross_val_predict(MeanRisk(; r = ConditionalValueatRisk(),
                                      opt = JuMPOptimiser(; slv = XN_MIP,
                                                          ple = IntegerPhylogenyEstimator(;
                                                                                          pl = n))),
                             rd, cv)
    for (f, g) in zip(pred.pred, gs)
        @test f.res.plr.A == unique(P(g) + I; dims = 1)
        @test all(f.res.plr.A * (collect(PO.fold_target(f)) .> 1e-6) .<= 1)
    end
    cc = CentralityConstraint(; A = CentralityEstimator(; pl = n), B = 0.1)
    pred = cross_val_predict(MeanRisk(; opt = JuMPOptimiser(; slv = XN_SLV, cte = cc)), rd,
                             cv)
    for (f, g) in zip(pred.pred, gs)
        c = centrality_vector(cc.A, xn_rd(g)).X
        @test vec(f.res.ctr.ineq.A) == c
        @test dot(c, PO.fold_target(f)) <= 0.1 + 1e-6
    end
    # HRP and HERC cluster over the graph.
    cle = ClustersEstimator(; de = FeatureDistance(; ape = PhylogenyPanel(; pl = n)))
    for opt in (HierarchicalRiskParity(; ser = HierarchicalSeriation(; cle = cle)),
                HierarchicalEqualRiskContribution(; cle = cle))
        pred = cross_val_predict(opt, rd, cv)
        for (f, g) in zip(pred.pred, gs)
            clr = opt isa HierarchicalRiskParity ? f.res.ser.clr : f.res.clr
            @test clr.D == clusterise(cle, xn_rd(g)).D
        end
    end
    # A gap that the prior reads. A plain graph reader leaves the asset out, and the network
    # reads no returns, so it keeps the asset.
    Xg = copy(XN_X)
    Xg[3, 2] = NaN
    rdg = xn_rd(g1; X = Xg)
    cvg = CoveragePolicy()
    pe = EmpiricalPrior(; me = SimpleExpectedReturns(; cvg = cvg),
                        ce = PortfolioOptimisersCovariance(; ce = Covariance(; cvg = cvg)))
    sdp(pl) = MeanRisk(;
                       opt = JuMPOptimiser(; pe = pe, slv = XN_SLV,
                                           ple = SemiDefinitePhylogenyEstimator(; pl = pl)))
    res = optimise(sdp(n), rdg)
    @test isnothing(res.imsk) || all(res.imsk)
    @test res.plr.A == P(g1)
    @test xn_quiet(() -> optimise(sdp(NetworkEstimator()), rdg)).imsk == [1, 0, 1, 1, 1]
    # The fit of a pipeline step.
    res = fit(Pipeline(;
                       steps = (SemiDefinitePhylogenyEstimator(; pl = n),
                                MeanRisk(; opt = JuMPOptimiser(; slv = XN_SLV)))),
              xn_rd(g1))
    @test res.ctx.constraints.A == P(g1)
    res = fit(Pipeline(; steps = (n, MeanRisk(; opt = JuMPOptimiser(; slv = XN_SLV)))),
              xn_rd(g1))
    @test res.ctx.phylogeny.X == P(g1)
    res = fit(Pipeline(; steps = (cc, MeanRisk(; opt = JuMPOptimiser(; slv = XN_SLV)))),
              xn_rd(g1))
    @test vec(res.ctx.constraints.res.ineq.A) == centrality_vector(cc.A, xn_rd(g1)).X
end
