#=
The Dependency Exposure (#2173, map #2162), at parity with the oracle at v1.14.4. The generator
of the stored cases is in the parity folder `graph2162/dep2173`, with a README.

Every stored case is `test/assets/Parity_Dep_<Case>_<Output>.csv.gz`. An exposure is one row
for each (row of weights, node) in the oracle's index order, with the columns weight, direct,
indirect and total.

  - `H<s>_A`, s = 0..3: a graph of 8 asset nodes `a1..a8` and 4 other nodes, with cycles, the
    sources in the rows. Seeds 2 and 3 hold self-loops. `H<s>_W`: three rows of weights.
    `H<s>_E`: 36 blocks of 12 nodes, for each row, each max_hops of (None, 1, 2, 3), and each
    of w, |w| and max(w, 0).
  - `N_E`: the nodes subset. `S_*`: a series with a missing graph and a node that one
    snapshot drops, with 2D and 1D weights, and a series with no graph. `D_*`: a dated
    portfolio with weight drift. `L_*`: a portfolio over 4 of the 6 rows of a series.

The verdicts:

  - Parity: every stored case, dense and sparse, the net, the gross on |w| and the long part
    on max(w, 0). The refusals of the oracle: a negative edge, an unknown node, a held asset
    that is not a node under `strict = true`, and no graph with no `nodes`.
  - Better: `(none)` keeps the weight of an asset that is not a node, one row for each fold of
    a path (the oracle gives the last row only), and the members of a frontier.
  - The improvements have tests of their own: `InEdges`, `BothEdges`, `AnyEdges`,
    `WalkShares`, a decay, the contributions, a union of graphs, the report section, and the
    relation with the taxonomy section.
=#
using Test, PortfolioOptimisers, LinearAlgebra, SparseArrays, StableRNGs, Statistics, Dates,
      DataFrames, Clarabel
const PO = PortfolioOptimisers
include(joinpath(@__DIR__, "parity_harness.jl"))

const DEP_NODES = vcat(["a$i" for i in 1:8], ["s$i" for i in 1:4])
const DEP_ASSETS = DEP_NODES[1:8]
const DEP_COLS = [:weight, :direct, :indirect, :total]

dep_load(case, out) = parity_load("Dep", case, out)
dep_table(res) = Matrix(DataFrame(res)[:, DEP_COLS])
dep_graph(A) = ExogenousGraph(; nodes = DEP_NODES, A = A)

# A dense reference of the hops of the shortest path from node i to node j, along `dir`.
function dep_hops(A, dir)
    B = A .!= 0
    B = dir == :out ? B : dir == :in ? permutedims(B) : B .| permutedims(B)
    n = size(B, 1)
    H = fill(typemax(Int), n, n)
    for i in 1:n
        H[i, i] = 0
        front = [i]
        h = 0
        while !isempty(front)
            h += 1
            next = Int[]
            for u in front, v in findall(view(B, u, :))
                if H[i, v] == typemax(Int)
                    H[i, v] = h
                    push!(next, v)
                end
            end
            front = next
        end
    end
    return H
end
# The reference exposure of the shortest hops: own weight, direct, indirect, over all nodes.
function dep_reference(A, w, dir; n = typemax(Int), f = h -> 1.0)
    H = dir == :both ? min.(dep_hops(A, :out), dep_hops(A, :in)) : dep_hops(A, dir)
    E = zeros(size(A, 1), 3)
    for (i, wi) in enumerate(w)
        E[i, 1] += wi
        for j in axes(A, 1)
            h = H[i, j]
            if h == 1 && h <= n
                E[j, 2] += wi * f(h)
            elseif 1 < h <= n && h < typemax(Int)
                E[j, 3] += wi * f(h)
            end
        end
    end
    return E
end

@testset "the hops are at parity with the oracle, dense and sparse" begin
    for s in 0:3
        A, W, E = dep_load("H$s", "A"), dep_load("H$s", "W"), dep_load("H$s", "E")
        b = 0
        for r in 1:3, H in (nothing, 1, 2, 3), v in 1:3
            b += 1
            ref = E[(12 * (b - 1) + 1):(12 * b), :]
            sep = isnothing(H) ? HopCount(; n = FractionOf(1.0)) : HopCount(H)
            de = DependencyExposure(; reach = ShortestHops(; sep = sep))
            for g in (dep_graph(A), dep_graph(sparse(A)))
                res = dependency_exposure(W[r, :], DEP_ASSETS, g, de)
                @test res.nodes == DEP_NODES
                df = DataFrame(res)
                # The oracle on w is our net, on |w| our gross, on max(w, 0) our long part.
                got = v == 1 ? dep_table(res) : reshape(v == 2 ? df.gross : df.long, :, 1)
                @test parity_compare(got, v == 1 ? ref : ref[:, 4:4]; rtol = 1e-13,
                                     atol = 1e-14, name = "H$s block $b").ok
            end
        end
    end
end

@testset "the nodes subset, and the refusals of the oracle" begin
    A, W = dep_load("H0", "A"), dep_load("H0", "W")
    g = dep_graph(A)
    res = dependency_exposure(W[1, :], DEP_ASSETS, g,
                              DependencyExposure(; nodes = ["s1", "a3", "s4"]))
    @test res.nodes == ["s1", "a3", "s4"]
    @test parity_compare(dep_table(res), dep_load("N", "E"); rtol = 1e-13, atol = 1e-14).ok
    @test_throws ArgumentError dependency_exposure(W[1, :], DEP_ASSETS, g,
                                                   DependencyExposure(; nodes = ["zz"]))
    An = copy(A)
    An[1, 10] = -0.5
    @test_throws ArgumentError dependency_exposure(W[1, :], DEP_ASSETS, dep_graph(An))
    # A held asset that is not a node: refused under `strict`, `(none)` otherwise.
    w = [W[1, :]; 0.1]
    nx = [DEP_ASSETS; "zz"]
    @test_throws ArgumentError dependency_exposure(w, nx, g,
                                                   DependencyExposure(; strict = true))
    res = @test_logs (:warn, r"\"zz\"") dependency_exposure(w, nx, g)
    @test res.nodes == [DEP_NODES; "(none)"]
    df = DataFrame(res)
    @test df.weight[end] == 0.1 && df.direct[end] == 0 && df.total[end] == 0.1
    @test parity_compare(dep_table(res)[1:12, :], dep_load("H0", "E")[1:12, :];
                         rtol = 1e-13, atol = 1e-14).ok
    @test sum(df.weight) ≈ sum(w)
    # An asset with a zero weight is never refused.
    res0 = dependency_exposure([W[1, :]; 0.0], nx, g, DependencyExposure(; strict = true))
    @test res0.nodes == DEP_NODES
    # A graph that holds a node named `(none)` gives the row the first free suffix.
    gn = ExogenousGraph(; edges = [("a", "(none)", 1.0), ("b", "(none)_1", 1.0)])
    rn = @test_logs (:warn,) dependency_exposure([0.5, 0.2, 0.3], ["a", "b", "c"], gn)
    @test last(rn.nodes) == "(none)_2"
end

@testset "a series with a missing graph, and a node that a snapshot drops" begin
    A1, A2, WS = dep_load("S", "A1"), dep_load("S", "A2"), dep_load("S", "W")
    g1 = dep_graph(A1)
    g2 = ExogenousGraph(; nodes = DEP_NODES[1:11], A = A2[1:11, 1:11])
    s = ExogenousGraphSeries(; ts = [1, 3, 4, 6], graphs = [g1, nothing, g2, g1])
    rd = ReturnsResult(; nx = DEP_ASSETS, X = zeros(6, 8), ng = ["g"], G = [s])
    r2 = dependency_exposure(WS, rd)
    @test parity_compare(dep_table(r2), dep_load("S", "E2"); rtol = 1e-13, atol = 1e-14).ok
    @test r2.ridx == [1, 1, 0, 2, 2, 1] && length(r2.Rd) == 2
    # The long table has the index of the oracle: the row, then the node.
    df = DataFrame(r2)
    @test names(df) ==
          ["row", "node", "weight", "direct", "indirect", "total", "long", "short", "gross"]
    @test df.row == repeat(1:6; inner = 12) && df.node == repeat(DEP_NODES, 6)
    @test all(iszero, df.total[(df.row .== 4) .& (df.node .== "s4")])
    r1 = dependency_exposure(WS[1, :], rd,
                             DependencyExposure(;
                                                reach = ShortestHops(; sep = HopCount(2))))
    @test parity_compare(dep_table(r1), dep_load("S", "E1"); rtol = 1e-13, atol = 1e-14).ok
    # No graph in force: refused with no nodes, NaN with nodes.
    rd0 = ReturnsResult(; nx = DEP_ASSETS, X = zeros(2, 8), ng = ["g"],
                        G = [ExogenousGraphSeries(; ts = [1], graphs = [nothing])])
    @test_throws ArgumentError dependency_exposure(WS[1, :], rd0)
    r0 = dependency_exposure(WS[1, :], rd0, DependencyExposure(; nodes = ["s1", "a2"]))
    @test isequal(dep_table(r0), dep_load("S", "E0"))
    @test isempty(r0.Rd) && r0.ridx == [0, 0]
    @test_throws ArgumentError exposure_contributions(r0, 1)
end

@testset "a row before the first key follows `strict`" begin
    g1 = dep_graph(dep_load("S", "A1"))
    rd = ReturnsResult(; nx = DEP_ASSETS, X = zeros(5, 8), ng = ["g"],
                       G = [ExogenousGraphSeries(; ts = [3], graphs = [g1])])
    w = fill(1 / 8, 8)
    @test_throws ArgumentError dependency_exposure(w, rd,
                                                   DependencyExposure(; strict = true))
    res = @test_logs (:warn, r"first key 3") dependency_exposure(w, rd)
    @test all(isnan, res.L[1:2, :, :]) && all(isfinite, res.L[3:5, :, :])
    # A date series needs the timestamps of the rows.
    ts = DateTime(2020, 1, 1) .+ Day.(0:4)
    sd = ExogenousGraphSeries(; ts = ts[[1]], graphs = [g1])
    @test_throws IsNothingError PO.exposure_row_key(sd, 1, nothing)
end

@testset "a portfolio with weight drift, and a series longer than the portfolio" begin
    A1, A2, X = dep_load("S", "A1"), dep_load("S", "A2"), dep_load("D", "X")
    g1 = dep_graph(A1)
    g2 = ExogenousGraph(; nodes = DEP_NODES[1:11], A = A2[1:11, 1:11])
    ts = collect(DateTime(2020, 1, 1):Day(1):DateTime(2020, 1, 6))
    s = ExogenousGraphSeries(; ts = ts[[1, 3, 6]], graphs = [g1, g2, g1])
    rd = ReturnsResult(; nx = DEP_ASSETS, X = X, ts = ts, ng = ["g"], G = [s])
    res = optimise(EqualWeighted(), rd)
    pred = predict(res, rd; wd = SelfFinancingDrift())
    @test pred.rd.ng == ["g"] && pred.rd.G == [s]
    rD = dependency_exposure(pred, DependencyExposure(; src = DriftedWeights()))
    @test parity_compare(rD.W, dep_load("D", "U"); rtol = 1e-13, atol = 1e-15).ok
    @test parity_compare(dep_table(rD), dep_load("D", "E"); rtol = 1e-13, atol = 1e-14).ok
    @test rD.rows == ts
    # One row at the target weights, with the graph in force at the first row.
    rT = dependency_exposure(pred)
    @test size(rT.L, 1) == 1 && rT.rows == ts[1:1]
    @test parity_compare(dep_table(rT), dep_table(rD)[1:12, :]; rtol = 1e-13).ok
    # A prediction made with no graph, and an explicit series of 6 rows: date keys, and integer
    # keys that count the rows of the prediction.
    rd4 = ReturnsResult(; nx = DEP_ASSETS, X = X[3:6, :], ts = ts[3:6])
    pred4 = predict(res, rd4)
    @test isnothing(pred4.rd.G)
    @test_throws IsNothingError dependency_exposure(pred4)
    de = DependencyExposure(; src = DriftedWeights())
    @test parity_compare(dep_table(dependency_exposure(pred4, s, de)), dep_load("L", "E");
                         rtol = 1e-13, atol = 1e-14).ok
    si = ExogenousGraphSeries(; ts = [1, 4], graphs = [g2, g1])
    @test parity_compare(dep_table(dependency_exposure(pred4, si, de)), dep_load("L", "E");
                         rtol = 1e-13, atol = 1e-14).ok
    # A portfolio built from weights, with no cross-validation.
    @test parity_compare(dep_table(dependency_exposure(res, rd4, g1)), dep_load("L", "E1");
                         rtol = 1e-13, atol = 1e-14).ok
    @test dep_table(dependency_exposure(res, rd)) == dep_table(rT)
    # The explicit graph replaces the lookup of `ng`.
    @test_throws ArgumentError dependency_exposure(pred4, s, DependencyExposure(; ng = "g"))
    @test_throws ArgumentError dependency_exposure(pred4, s,
                                                   DependencyExposure(;
                                                                      ng = GraphUnion(;
                                                                                      ng = ["g",
                                                                                            "h"])))
end

# A dense reference of the walk shares, the part at one hop and the part at two hops or more.
function dep_walk(A, w, dir; n, f = h -> 1.0, nrm = true, unit = false)
    At = permutedims(Float64.(A))
    M = if dir == :out
        At
    elseif dir == :in
        permutedims(At)
    else
        At + permutedims(At) - Diagonal(diag(At))
    end
    M = unit ? Float64.(M .!= 0) : M
    if nrm
        M = M * Diagonal([x == 0 ? 0.0 : 1 / x for x in vec(sum(M; dims = 1))])
    end
    E = zeros(size(A, 1), 3)
    for (i, wi) in enumerate(w)
        E[i, 1] += wi
        v = zeros(size(A, 1))
        v[i] = 1
        for h in 1:n
            v = M * v
            p = copy(v)
            p[i] = 0
            E[:, h == 1 ? 2 : 3] .+= wi * f(h) .* p
        end
    end
    return E
end
dep_parts(res) = Matrix(DataFrame(res)[:, [:weight, :direct, :indirect]])

@testset "the directions, the budget and the decay of the shortest hops" begin
    A, W = dep_load("H1", "A"), dep_load("H1", "W")
    w = [W[2, :]; zeros(4)]
    g = dep_graph(A)
    for (dir, key) in
        ((OutEdges(), :out), (InEdges(), :in), (BothEdges(), :both), (AnyEdges(), :any)),
        sp in (false, true)

        gg = sp ? dep_graph(sparse(A)) : g
        res = dependency_exposure(W[2, :], DEP_ASSETS, gg, DependencyExposure(; dir = dir))
        @test dep_parts(res) ≈ dep_reference(A, w, key)
        de = DependencyExposure(; dir = dir,
                                reach = ShortestHops(; sep = HopCount(2),
                                                     decay = ExponentialDecay(; rate = 0.5)))
        @test dep_parts(dependency_exposure(W[2, :], DEP_ASSETS, gg, de)) ≈
              dep_reference(A, w, key; n = 2, f = h -> exp(-0.5 * h))
    end
    # A FractionOf takes its quantile over the hops of the pairs of a held asset and a node.
    H = dep_hops(A, :out)
    v = [H[i, j] for i in 1:8 for j in 1:12 if i != j && H[i, j] < typemax(Int)]
    n = round(Int, quantile(v, 0.5))
    de = DependencyExposure(;
                            reach = ShortestHops(; sep = HopCount(; n = FractionOf(0.5)),
                                                 decay = LinearDecay()))
    @test dep_parts(dependency_exposure(W[2, :], DEP_ASSETS, g, de)) ≈
          dep_reference(A, w, :out; n = n, f = h -> n + 1 - h)
    # A held asset that reaches no other node: the exposure is its own weight.
    gi = ExogenousGraph(; edges = [("a", "b", 1.0)], nodes = ["a", "b", "c"])
    ri = dependency_exposure([0.0, 0.0, 1.0], ["a", "b", "c"], gi)
    @test dep_parts(ri) == [0 0 0; 0 0 0; 1 0 0]
    @test_throws ArgumentError PO.resolve_separation(HopCount(; n = FractionOf(0.5)),
                                                     ShortestHops(),
                                                     reshape([0, typemax(Int)], 2, 1),
                                                     [true])
end

@testset "the walk shares, and the contributions" begin
    A, W = dep_load("H2", "A"), dep_load("H2", "W")
    w = [W[1, :]; zeros(4)]
    for (dir, key) in ((OutEdges(), :out), (InEdges(), :in), (AnyEdges(), :any))
        de = DependencyExposure(; dir = dir, reach = WalkShares(; sep = HopCount(3)))
        res = dependency_exposure(W[1, :], DEP_ASSETS, dep_graph(sparse(A)), de)
        @test dep_parts(res) ≈ dep_walk(A, w, key; n = 3)
        C = exposure_contributions(res, 1)
        @test size(C) == (12, 8)
        @test vec(sum(C; dims = 2)) ≈ DataFrame(res).total
    end
    de = DependencyExposure(; dir = BothEdges(), reach = WalkShares(; sep = HopCount(2)))
    @test dep_parts(dependency_exposure(W[1, :], DEP_ASSETS, dep_graph(A), de)) ≈
          dep_walk(A, w, :out; n = 2) .+ dep_walk(A, w, :in; n = 2) .- [w zeros(12, 2)]
    de = DependencyExposure(;
                            reach = WalkShares(; sep = HopCount(2), nrm = nothing,
                                               wgt = UnitEdgeWeights(),
                                               decay = ExponentialDecay(; rate = 1.0)))
    @test dep_parts(dependency_exposure(W[1, :], DEP_ASSETS, dep_graph(A), de)) ≈
          dep_walk(A, w, :out; n = 2, nrm = false, unit = true, f = h -> exp(-h))
    # The default budget: the largest hop count of a shortest path of a held asset.
    H = dep_hops(A, :out)
    n = maximum(H[i, j] for i in 1:8 for j in 1:12 if i != j && H[i, j] < typemax(Int))
    @test dep_parts(dependency_exposure(W[1, :], DEP_ASSETS, dep_graph(A),
                                        DependencyExposure(; reach = WalkShares()))) ≈
          dep_walk(A, w, :out; n = n)
    # No reach gives zero parts.
    gi = ExogenousGraph(; edges = [("a", "b", 1.0)], nodes = ["a", "b", "c"])
    ri = dependency_exposure([0.0, 0.0, 1.0], ["a", "b", "c"], gi,
                             DependencyExposure(; reach = WalkShares()))
    @test dep_parts(ri) == [0 0 0; 0 0 0; 1 0 0]
end

struct DepUserSource <: PortfolioOptimisers.AbstractPreviousWeightsSource end

@testset "the configuration refuses what it cannot read" begin
    @test_throws ArgumentError ShortestHops(; sep = HopCount(; n = (args...; kw...) -> 2))
    @test_throws ArgumentError WalkShares(; sep = HopCount(; n = (args...; kw...) -> 2))
    @test_throws TypeError ShortestHops(; sep = PathLength())
    @test_throws PO.IsEmptyError DependencyExposure(; ng = "")
    @test_throws PO.IsEmptyError DependencyExposure(; nodes = String[])
    @test_throws ArgumentError DependencyExposure(; nodes = ["a", "a"])
    @test_throws MethodError WalkProfilePanel(; dir = AnyEdges())
    @test PO.graph_requirements(DependencyExposure()) == (NonNegativeWeights(),)
    @test PO.graph_names(DependencyExposure(; ng = "g")) == "g"
    @test PO.exposure_parts(AnyEdges()) == (AnyEdges(),)
    @test PO.exposure_parts(BothEdges()) == (OutEdges(), InEdges())
    g = dep_graph(dep_load("H0", "A"))
    @test_throws ArgumentError dependency_exposure(ones(8), DEP_ASSETS, g,
                                                   DependencyExposure(; ng = "g"))
    @test_throws DimensionMismatch dependency_exposure(ones(7), DEP_ASSETS, g)
    # Integer weights enter as floats, so a row with no graph can be NaN.
    @test eltype(dependency_exposure([1, 0, 0, 0, 0, 0, 0, 0], DEP_ASSETS, g).L) == Float64
    rd = ReturnsResult(; nx = DEP_ASSETS, X = zeros(3, 8), ng = ["g"], G = [g])
    W = ones(3, 8) ./ 8
    W[2, 1] = NaN
    @test_throws ArgumentError dependency_exposure(W, rd)
    @test_throws DimensionMismatch dependency_exposure(ones(2, 8), rd)
    @test_throws PO.IsNothingError dependency_exposure(ones(8),
                                                       ReturnsResult(; X = zeros(3, 8)))
    # A source of a user with no method of `exposure_weights` raises a `MethodError`.
    p = predict(optimise(EqualWeighted(), rd), rd)
    @test_throws MethodError dependency_exposure(p,
                                                 DependencyExposure(;
                                                                    src = DepUserSource()))
    # The Result checks its shapes.
    kw = (; nodes = ["a"], nx = ["x"], rows = [1], L = zeros(1, 1, 3), S = zeros(1, 1, 3),
          R0 = Matrix{Bool}[], Rd = Matrix{Float64}[], Ri = Matrix{Float64}[])
    @test_throws DimensionMismatch DependencyExposureResult(; kw..., W = zeros(2, 1),
                                                            ridx = [0])
    @test_throws DomainError DependencyExposureResult(; kw..., W = zeros(1, 1), ridx = [1])
end

@testset "a union of graphs in `ng`" begin
    g1 = dep_graph(dep_load("S", "A1"))
    g2 = dep_graph(dep_load("S", "A2"))
    rd = ReturnsResult(; nx = DEP_ASSETS, X = zeros(4, 8), ng = ["a", "b"], G = [g1, g2])
    w = fill(1 / 8, 8)
    gu = GraphUnion(; ng = ["a", "b"], nrm = WeightShare())
    for dir in (OutEdges(), InEdges())
        de = DependencyExposure(; ng = gu, dir = dir, reach = WalkShares())
        ref = dependency_exposure(w, DEP_ASSETS,
                                  graph_union(g1, g2; nrm = WeightShare(), dir = dir),
                                  DependencyExposure(; dir = dir, reach = WalkShares()))
        @test dep_table(dependency_exposure(w, rd, de))[1:12, :] ≈ dep_table(ref)
    end
    # BothEdges builds the union along each single direction.
    de = DependencyExposure(; ng = gu, dir = BothEdges())
    @test size(dependency_exposure(w, rd, de).L) == (4, 12, 3)
    @test_throws MethodError dependency_exposure(w, rd,
                                                 DependencyExposure(; ng = gu,
                                                                    dir = AnyEdges()))
    # The union of the parts with no normalisation reads `AnyEdges`.
    de = DependencyExposure(; ng = GraphUnion(; ng = ["a", "b"]), dir = AnyEdges())
    ref = dependency_exposure(w, DEP_ASSETS, graph_union(g1, g2),
                              DependencyExposure(; dir = AnyEdges()))
    @test dep_table(dependency_exposure(w, rd, de))[1:12, :] ≈ dep_table(ref)
    # A negative edge in a part is refused before any compute.
    An = dep_load("S", "A2")
    An[1, 2] = -1.0
    rdn = ReturnsResult(; nx = DEP_ASSETS, X = zeros(4, 8), ng = ["a", "b"],
                        G = [g1, dep_graph(An)])
    @test_throws ArgumentError dependency_exposure(w, rdn, DependencyExposure(; ng = gu))
end

const DEP_X = randn(StableRNG(987654321), 60, 4) .* 0.02 .+ 0.001
const DEP_TS = Date(2024, 1, 1) .+ Day.(0:59)
const DEP_G1 = ExogenousGraph(;
                              edges = [("A", "S", 1.0), ("B", "S", 2.0), ("S", "T", 1.0),
                                       ("C", "A", 1.0), ("D", "T", 1.0)])
const DEP_G2 = ExogenousGraph(; edges = [("A", "S", 1.0), ("D", "T", 2.0), ("S", "T", 1.0)])
const DEP_RD = ReturnsResult(; nx = ["A", "B", "C", "D"], X = DEP_X, ts = DEP_TS,
                             ng = ["supply"],
                             G = [ExogenousGraphSeries(; ts = DEP_TS[[1, 35]],
                                                       graphs = [DEP_G1, DEP_G2])])
const DEP_PATH = cross_val_predict(InverseVolatility(), DEP_RD,
                                   IndexWalkForward(20, 10; wd = SelfFinancingDrift()))
const DEP_POP = cross_val_predict(InverseVolatility(), DEP_RD,
                                  MultipleRandomised(IndexWalkForward(30, 10);
                                                     subset_size = 3, n_subsets = 3,
                                                     rng = StableRNG(11), seed = 7))

@testset "a path gives one row for each fold, and a population one row for each path" begin
    res = @test_logs (:warn,) (:warn,) match_mode = :any dependency_exposure(DEP_PATH)
    folds = DEP_PATH.pred
    @test res.rows == [f.rd.ts[1] for f in folds]
    # DEP_G2 holds no node B or C, so the folds that read it give the row `(none)`.
    @test res.nodes == ["A", "S", "B", "T", "C", "D", "(none)"]
    # Each row is the exposure of the target of its fold, with the graph of its first row.
    for (k, f) in enumerate(folds)
        # The test rows of fold k start at 10k + 11, and the key of DEP_G2 is row 35.
        g = 10 * k + 11 >= 35 ? DEP_G2 : DEP_G1
        ref = dependency_exposure(collect(PO.fold_target(f)), f.rd.nx, g)
        # A node that the graph of the fold does not hold has zero exposure at its row.
        for (c, n) in enumerate(res.nodes)
            r = findfirst(==(n), ref.nodes)
            @test res.L[k, c, :] ≈ (isnothing(r) ? zeros(3) : ref.L[1, r, :])
        end
    end
    # The oracle reads the initial weights of the last portfolio, our last row.
    @test res.ridx == [1, 1, 2, 2]
    rD = dependency_exposure(DEP_PATH, DependencyExposure(; src = DriftedWeights()))
    @test size(rD.L, 1) == 40 && rD.rows == reduce(vcat, [f.rd.ts for f in folds])
    @test rD.W[1, :] ≈ PO.fold_target(folds[1]) && rD.W[2, :] != rD.W[1, :]
    # A population: the mean of the rows of each path.
    rp = dependency_exposure(DEP_POP)
    @test rp.rows == [PO.population_path_key(p, k) for (k, p) in enumerate(DEP_POP.pred)]
    @test all(iszero, rp.ridx) && isempty(rp.Rd)
    r1 = dependency_exposure(DEP_POP.pred[1])
    c = [findfirst(==(n), rp.nodes) for n in r1.nodes]
    @test rp.L[1, c, :] ≈ dropdims(mean(r1.L; dims = 1); dims = 1)
    @test_throws ArgumentError exposure_contributions(rp, 1)
    @test size(dependency_exposure(DEP_POP, DEP_G1).L, 1) == 3
    @test_throws PO.IsEmptyError dependency_exposure(PopulationPredictionResult())
end

@testset "the members of a frontier" begin
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = Dict("verbose" => false))
    opt = MeanRisk(;
                   opt = JuMPOptimiser(; slv = slv,
                                       ret = ArithmeticReturn(;
                                                              settings = JuMPReturnsSettings(;
                                                                                             lb = Frontier(;
                                                                                                           N = 3)))))
    p = cross_val_predict(opt, DEP_RD, IndexWalkForward(30, 10))
    de = DependencyExposure(; nodes = ["S", "T"])
    for j in 1:3
        res = dependency_exposure(p, de, j)
        f = p.pred[end]
        ref = dependency_exposure(collect(PO.fold_target(f, j)), f.rd.nx, DEP_G2, de)
        @test res.L[end, :, :] ≈ ref.L[1, :, :]
    end
    @test dependency_exposure(p, de, 1).L != dependency_exposure(p, de, 3).L
    # Under a drift, each member reads its own weight path.
    pd = cross_val_predict(opt, DEP_RD, IndexWalkForward(30, 10; wd = SelfFinancingDrift()))
    dd = DependencyExposure(; nodes = ["S", "T"], src = DriftedWeights())
    for j in 1:3
        r = dependency_exposure(pd, dd, j)
        U = PO.path_member(PO.weight_path(pd.pred[1].hw, pd.pred[1].res.w), j)
        @test size(r.L, 1) == 30 && r.W[1:10, :] ≈ U
    end
end

function dep_section(e)
    return e.sections[findfirst(s -> s.title == "Dependency exposure", e.sections)]
end

@testset "the report section" begin
    one = predict(optimise(InverseVolatility(), DEP_RD), DEP_RD)
    de = DependencyExposure()
    r = @test_logs (:warn,) match_mode = :any portfolio_report([one, DEP_PATH, DEP_POP];
                                                               dependency = ["Supply" => de,
                                                                             "Explicit" =>
                                                                                 (DEP_G1,
                                                                                  DependencyExposure(;
                                                                                                     dir = InEdges()))])
    @test [s.title for s in r.entries[1].sections] ==
          ["Statistics", "Weights", "Dependency exposure"]
    s1 = dep_section(r.entries[1])
    @test [b.title for b in s1.blocks] == ["Supply", "Explicit"]
    t = only(s1.blocks[1].blocks)
    @test names(t.df) ==
          ["node", "weight", "direct", "indirect", "total", "long", "short", "gross"]
    @test issorted(abs.(t.df.total); rev = true)
    ref = DataFrame(dependency_exposure(one, de))
    @test sort(t.df.total) ≈ sort(ref.total)
    t2 = only(dep_section(r.entries[2]).blocks[1].blocks)
    @test names(t2.df) == ["node", "last", "count", "min", "q25", "median", "q75", "max"]
    @test issorted(abs.(t2.df.last); rev = true) && all(==(4), t2.df.count)
    t3 = only(dep_section(r.entries[3]).blocks[2].blocks)
    @test !("last" in names(t3.df)) && issorted(abs.(t3.df.median); rev = true)
    # `weight_threshold` hides a node with no exposure of at least the threshold.
    rt = portfolio_report(one;
                          dependency = ["Explicit" =>
                                            (DEP_G1, DependencyExposure(; dir = InEdges()))],
                          weight_threshold = 0.3)
    tt = only(only(dep_section(rt.entries[1]).blocks).blocks)
    @test all(x -> abs(x) >= 0.3, tt.df.total) && occursin("not shown", tt.caption)
    # One fold under `DriftedWeights` has many rows.
    held = DEP_PATH.pred[1]
    rh = portfolio_report(held;
                          dependency = ["Held" =>
                                            DependencyExposure(; src = DriftedWeights())])
    @test "last" in names(only(only(dep_section(rh.entries[1]).blocks).blocks).df)
    @test_throws ArgumentError portfolio_report(one; dependency = ["x" => 1])
    @test_throws ArgumentError portfolio_report(one; dependency = [1 => de])
    @test_throws ArgumentError portfolio_report(one; dependency = ["x" => (de, DEP_G1)])
    # A node of a row with no graph shows, because NaN is not less than the threshold.
    @test PO.dependency_shown(dependency_exposure(ones(4) ./ 4,
                                                  ReturnsResult(; nx = ["A", "B", "C", "D"],
                                                                X = zeros(2, 4), ng = ["g"],
                                                                G = [ExogenousGraphSeries(;
                                                                                          ts = [1,
                                                                                                2],
                                                                                          graphs = [DEP_G1,
                                                                                                    nothing])])),
                              1.0) == trues(6)
end

@testset "the taxonomy net equals `direct` over the bipartite graph of the labels" begin
    sets = UniverseSets(;
                        dict = Dict("nx" => ["A", "B", "C", "D"],
                                    "nx_sector" => ["x", "x", "y", "z"]))
    res = PO.NaiveOptimisationResult(; pr = nothing, wb = nothing,
                                     retcode = OptimisationSuccess(),
                                     w = [0.5, 0.7, -0.3, 0.1], fb = nothing)
    rd = ReturnsResult(; nx = ["A", "B", "C", "D"], X = DEP_X, ts = DEP_TS)
    p = predict(res, rd)
    r = portfolio_report(p; sets = sets, taxonomy = ["nx_sector"])
    ex = r.entries[1].sections[findfirst(s -> s.title == "Exposure", r.entries[1].sections)]
    tab = only(only(ex.blocks).blocks).df
    g = ExogenousGraph(;
                       edges = [(a, l, 1.0)
                                for (a, l) in zip(sets.dict["nx"], sets.dict["nx_sector"])])
    de = DependencyExposure(; reach = ShortestHops(; sep = HopCount(1)))
    df = DataFrame(dependency_exposure(p, g, de))
    for (grp, net) in zip(tab.group, tab.net)
        @test df.direct[df.node .== grp][1] ≈ net
    end
end

@testset "a prediction keeps the graphs of its fold" begin
    f = DEP_PATH.pred[3]
    @test f.rd.ng == ["supply"]
    @test graph_in_force(PO.ReturnsResult(; nx = f.rd.nx, X = zeros(10, 4), ts = f.rd.ts,
                                          ng = f.rd.ng, G = f.rd.G)) == DEP_G2
    @test isnothing(DEP_PATH.mrd.G)
    @test_throws DimensionMismatch PredictionReturnsResult(; X = zeros(3), ng = ["a", "b"],
                                                           G = [DEP_G1])
end
