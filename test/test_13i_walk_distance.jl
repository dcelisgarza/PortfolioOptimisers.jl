#=
The Walk Profile and the walk distance (#2172, map #2162), at parity with the oracle at v1.14.4.
The generator of the stored cases is in the parity folder `graph2162/walk2172`, with a README.

Every stored case is `test/assets/Parity_Walk_<Case>_<Output>.csv.gz`:

  - `G<s>_A`, s = 0..3: a graph of 8 asset nodes `a1..a8` and 5 other nodes, the sources in the
    rows. Seed 1: `a4` has no out-edge. Seed 2: `a4` and `a7`. Seed 3 adds an asset `zz` that is
    not a node.
  - `G<s>_D`, `G<s>_S`: the oracle's distance and codependence of 14 cases stacked in rows: the
    7 configurations of `WALK_CONFIGS`, each on a dense and on a sparse adjacency. A NaN row is
    an asset that the oracle cannot read.
  - `Herc_X`, `Herc<s>_W`: HERC with the oracle's graph distance and ward linkage at `k` = 2, 3,
    4 clusters, on the graphs of seeds 0 and 1. The oracle raises on seed 1, and its rows are
    NaN.

The verdicts:

  - Parity: D and S of every case, the set of unreadable assets, Float32 at its tolerance, and
    HERC through a fixed cut.
  - Better: where the oracle raises on an asset with no out-edge, a fit drops the asset at its
    entry and a direct call refuses it by name. The scaled walk does not overflow where the
    oracle raises.
  - The improvements have tests of their own: `InEdges`, `BothEdges`, `self`, `UnitEdgeWeights`,
    several graphs, the scaled walk, `LinearDecay` and `ReciprocalDecay`, a `FractionOf` and a
    rule budget, the readable mask in a fit, and HRP, Schur, NCO and HERC under a fold.
=#
using Test, PortfolioOptimisers, LinearAlgebra, SparseArrays, StableRNGs, Statistics, Dates
include(joinpath(@__DIR__, "parity_harness.jl"))

const WALK_NODES = vcat(["a$i" for i in 1:8], ["s$i" for i in 1:5])
# (max_hops, decay, normalize_adjacency) of the oracle.
const WALK_CONFIGS = [(1, 0.5, true), (2, 0.5, true), (5, 0.0, true), (4, 1.0, true),
                      (2, 0.5, false), (4, 1.0, false), (3, 0.5, false)]

walk_assets(s) = vcat(WALK_NODES[1:8], s == 3 ? ["zz"] : String[])
function walk_rd(A; nx = WALK_NODES[1:8], ng = ["g"],
                 X = randn(StableRNG(2172), 30, length(nx)))
    return ReturnsResult(; nx = nx, X = X, ng = ng,
                         G = [ExogenousGraph(; nodes = WALK_NODES, A = a) for a in A])
end
# Our configuration of an oracle configuration: alpha = 0 is one hop, alpha = 1 is no decay,
# and alpha^h is `ExponentialDecay` at the rate log(1 / alpha).
function walk_config(H, a, nrm; T = Float64)
    decay = a == 0 || a == 1 ? nothing : ExponentialDecay(; rate = log(one(T) / T(a)))
    return WalkDistance(; sep = HopCount(; n = a == 0 ? 1 : H), decay = decay,
                        nrm = nrm ? WeightShare() : nothing)
end
# The distance and similarity of `de` over the assets that its producer can read.
function walk_measure(de, rd)
    keep = PortfolioOptimisers.readable_assets(de.ape, rd)
    rv = PortfolioOptimisers.port_opt_view(rd, findall(keep))
    S, D = cor_and_dist(de, nothing, rv.X; rd = rv)
    return keep, S, D
end
# A dense reference of the Walk Profile, scaled to unit length, assets × nodes. `A` holds the
# sources in its rows. `ws[h + 1]` is the weight of a walk of h hops.
function walk_reference(A, nx, ws; dir = :out, nrm = true, self = false, unit = false)
    At = permutedims(Float64.(A))
    M = dir == :out ? At : permutedims(At)
    M = unit ? Float64.(M .!= 0) : M
    if nrm
        s = vec(sum(M; dims = 1))
        M = M * Diagonal([x == 0 ? 0.0 : 1 / x for x in s])
    end
    pos = Dict(n => k for (k, n) in enumerate(WALK_NODES))
    P = zeros(length(nx), length(WALK_NODES))
    for (i, a) in enumerate(nx)
        e = zeros(length(WALK_NODES))
        e[pos[a]] = 1
        p = self ? ws[1] * e : zeros(length(WALK_NODES))
        v = e
        for h in 1:(length(ws) - 1)
            v = M * v
            p += ws[h + 1] * v
        end
        P[i, :] = iszero(norm(p)) ? p : p / norm(p)
    end
    return P
end
# The orthant chord distance between the stacked rows of the profiles `Ps`.
function walk_reference_distance(Ps...)
    P = reduce(hcat, Ps)
    c(i, j) =
        if iszero(norm(P[i, :])) || iszero(norm(P[j, :]))
            0.0
        else
            dot(P[i, :], P[j, :]) / (norm(P[i, :]) * norm(P[j, :]))
        end
    return [i == j ? 0.0 : sqrt(clamp(1 - c(i, j), 0, 1))
            for i in axes(P, 1), j in axes(P, 1)]
end

@testset "1. D and S at parity with the oracle, and the unreadable set" begin
    worst = (0.0, 0.0)
    for s in 0:3
        A = parity_load("Walk", "G$s", "A")
        Do = parity_load("Walk", "G$s", "D")
        So = parity_load("Walk", "G$s", "S")
        na = length(walk_assets(s))
        for (c, (H, a, nrm)) in enumerate(WALK_CONFIGS),
            (k, kind) in enumerate((:dense, :sparse))

            rd = walk_rd([kind == :sparse ? sparse(A) : A]; nx = walk_assets(s))
            keep, S, D = walk_measure(walk_config(H, a, nrm), rd)
            r = (((c - 1) * 2 + k - 1) * na + 1):(((c - 1) * 2 + k) * na)
            # The oracle writes NaN for each asset that it cannot read, and we drop the same set.
            @test keep == [!all(isnan, Do[i, :]) for i in r]
            worst = (max(worst[1], maximum(abs, D - Do[r, :][keep, keep])),
                     max(worst[2], maximum(abs, S - So[r, :][keep, keep])))
        end
    end
    # 2.9e-15 and 5.6e-16 on 2026-11-23.
    @test worst[1] <= 1e-13
    @test worst[2] <= 1e-13
end

@testset "2. A Float32 graph walks in Float32, at its tolerance" begin
    A = parity_load("Walk", "G0", "A")
    Do = parity_load("Walk", "G0", "D")
    rd = ReturnsResult(; nx = WALK_NODES[1:8], X = randn(StableRNG(1), Float32, 30, 8),
                       ng = ["g"],
                       G = [ExogenousGraph(; nodes = WALK_NODES, A = Float32.(A))])
    _, S, D = walk_measure(walk_config(2, 0.5, true; T = Float32), rd)
    @test eltype(D) == Float32
    @test eltype(S) == Float32
    # The config of row block 2 is (2, 0.5, true), dense.
    @test isapprox(D, Do[17:24, :]; atol = 1e-5)
end

@testset "3. HERC at parity through a fixed cut, and the drop where the oracle raises" begin
    X = parity_load("Walk", "Herc", "X")
    cle = ClustersEstimator(; de = WalkDistance(),
                            alg = HClustAlgorithm(; linkage = WardLinkage()))
    rd0 = walk_rd([parity_load("Walk", "G0", "A")]; X = X)
    clr = clusterise(cle, rd0)
    Wo = parity_load("Walk", "Herc0", "W")
    for (i, k) in enumerate((2, 3, 4))
        clk = Clusters(; res = clr.res, S = clr.S, D = clr.D, k = k)
        w = optimise(HierarchicalEqualRiskContribution(; cle = clk), rd0).w
        # 5.6e-17 on 2026-11-23.
        @test isapprox(w, Wo[i, :]; atol = 1e-14)
    end
    # The oracle raises on seed 1, because a4 has no out-edge.
    @test all(isnan, parity_load("Walk", "Herc1", "W"))
    rd1 = walk_rd([parity_load("Walk", "G1", "A")]; X = X)
    @test PortfolioOptimisers.readable_assets(cle.de.ape, rd1) == (1:8 .!= 4)
    r1 = @test_logs (:info, r"\"a4\"") match_mode=:any optimise(HierarchicalEqualRiskContribution(;
                                                                                                  cle = cle),
                                                                rd1)
    @test r1.w[4] == 0
    rv = PortfolioOptimisers.port_opt_view(rd1, [1, 2, 3, 5, 6, 7, 8])
    @test r1.w[[1, 2, 3, 5, 6, 7, 8]] ==
          optimise(HierarchicalEqualRiskContribution(; cle = cle), rv).w
end

@testset "4. A direct call refuses an unreadable asset by name" begin
    de = WalkDistance()
    rd1 = walk_rd([parity_load("Walk", "G1", "A")])
    @test_throws r"\"a4\"" cor_and_dist(de, nothing, rd1.X; rd = rd1)
    rd3 = walk_rd([parity_load("Walk", "G3", "A")]; nx = walk_assets(3))
    @test PortfolioOptimisers.readable_assets(de.ape, rd3) == .!in.(1:9, Ref((4, 7, 9)))
    @test_throws r"\"zz\"" distance(de, nothing, rd3.X; rd = rd3)
    # `clusterise(cle, rd)` puts the returns data in the prior slot, and the walk reads it.
    rd0 = walk_rd([parity_load("Walk", "G0", "A")])
    pnl = asset_panel(de.ape, rd0, nothing, rd0.X)
    @test pnl.pf[1].name == "walk_out_g"
    @test pnl.pf[1].labels == WALK_NODES
    @test isnothing(pnl.pf[1].pmsk)
    @test_throws IsNothingError asset_panel(de.ape, nothing, nothing, rd0.X)
    @test_throws DimensionMismatch asset_panel(de.ape, nothing, rd0, rd0.X[:, 1:3])
    @test_throws IsNothingError PortfolioOptimisers.readable_assets(de.ape,
                                                                    ReturnsResult(;
                                                                                  X = rd0.X,
                                                                                  ng = ["g"],
                                                                                  G = rd0.G))
    # A producer that cannot tell before it fits keeps the mask.
    @test isnothing(PortfolioOptimisers.readable_assets(PhylogenyPanel(), rd0))
    @test PortfolioOptimisers.feature_readable_mask(FeatureDistance(;
                                                                    ape = PhylogenyPanel()),
                                                    nothing, rd0) === nothing
end

@testset "5. The directions, the self term and the unit weights against a dense reference" begin
    A = parity_load("Walk", "G0", "A")
    rd = walk_rd([sparse(A)])
    nx = rd.nx
    ws = [exp(-log(2) * h) for h in 0:2]
    for (dir, sym) in ((OutEdges(), :out), (InEdges(), :in))
        for self in (false, true),
            (wgt, unit) in ((nothing, false), (UnitEdgeWeights(), true))

            de = WalkDistance(; dir = dir, self = self, wgt = wgt)
            keep, _, D = walk_measure(de, rd)
            P = walk_reference(A, nx[keep], ws; dir = sym, self = self, unit = unit)
            @test isapprox(D, walk_reference_distance(P); atol = 1e-13)
        end
    end
    # InEdges walks the transposed graph along its out-edges.
    rdt = walk_rd([permutedims(A)])
    @test isapprox(last(walk_measure(WalkDistance(; dir = InEdges()), rd)),
                   last(walk_measure(WalkDistance(), rdt)); atol = 1e-14)
    # BothEdges gives the two profiles as two fields, and the distance of their stack.
    pnl = asset_panel(WalkProfilePanel(; dir = BothEdges()), nothing, rd, rd.X)
    @test [f.name for f in pnl.pf] == ["walk_out_g", "walk_in_g"]
    keep, _, D = walk_measure(WalkDistance(; dir = BothEdges()), rd)
    Po = walk_reference(A, nx[keep], ws; dir = :out)
    Pi = walk_reference(A, nx[keep], ws; dir = :in)
    @test isapprox(D, walk_reference_distance(Po, Pi); atol = 1e-13)
    # The self term keeps an asset with no out-edge readable.
    rd1 = walk_rd([parity_load("Walk", "G1", "A")])
    @test !PortfolioOptimisers.readable_assets(WalkProfilePanel(), rd1)[4]
    @test all(PortfolioOptimisers.readable_assets(WalkProfilePanel(; self = true), rd1))
    _, _, D1 = walk_measure(WalkDistance(; self = true), rd1)
    @test isapprox(D1,
                   walk_reference_distance(walk_reference(parity_load("Walk", "G1", "A"),
                                                          rd1.nx, ws; self = true));
                   atol = 1e-13)
    # A graph with no edge: under the self term each profile is the asset's own node.
    rde = walk_rd([zeros(13, 13)])
    for nrm in (WeightShare(), nothing)
        _, S, D = walk_measure(WalkDistance(; self = true, nrm = nrm), rde)
        @test D == ones(8, 8) - I
        @test S == Matrix{Float64}(I, 8, 8)
    end
end

@testset "6. Several graphs, the prefix of the fields, and a zero part" begin
    A0 = parity_load("Walk", "G0", "A")
    A2 = parity_load("Walk", "G2", "A")
    rd = walk_rd([A0, A2]; ng = ["g0", "g2"])
    ape = WalkProfilePanel(; ng = ["g0", "g2"], prefix = "dep")
    pnl = asset_panel(ape, nothing, rd, rd.X)
    @test [f.name for f in pnl.pf] == ["dep_out_g0", "dep_out_g2"]
    # a4 and a7 have no out-edge in g2, so that part is zero, and they stay readable.
    @test all(iszero, pnl.pf[2].vals[[4, 7], :])
    @test isnothing(pnl.pf[2].pmsk)
    @test all(PortfolioOptimisers.readable_assets(ape, rd))
    ws = [exp(-log(2) * h) for h in 0:2]
    _, _, D = walk_measure(WalkDistance(; ng = ["g0", "g2"]), rd)
    @test isapprox(D,
                   walk_reference_distance(walk_reference(A0, rd.nx, ws),
                                           walk_reference(A2, rd.nx, ws)); atol = 1e-13)
    # One name reads that graph, and `nothing` refuses when the data holds two graphs.
    @test asset_panel(WalkProfilePanel(; ng = "g2"), nothing, rd, rd.X).pf[1].name ==
          "walk_out_g2"
    @test_throws ArgumentError asset_panel(WalkProfilePanel(), nothing, rd, rd.X)
    # An asset that is not a node of one graph has a placeholder in its fields alone.
    # a8 is not a node of gz.
    kz = vcat(1:7, 9:13)
    gz = ExogenousGraph(; nodes = WALK_NODES[kz], A = A2[kz, kz])
    rdz = ReturnsResult(; nx = WALK_NODES[1:8], X = randn(StableRNG(3), 30, 8),
                        ng = ["g0", "gz"],
                        G = [ExogenousGraph(; nodes = WALK_NODES, A = A0), gz])
    pz = asset_panel(WalkProfilePanel(; ng = ["g0", "gz"]), nothing, rdz, rdz.X)
    @test isnothing(pz.pf[1].pmsk)
    @test pz.pf[2].pmsk == repeat(1:8 .== 8, 1, 12)
    @test pz.pf[2].omsk == .!pz.pf[2].pmsk
    @test PortfolioOptimisers.readable_assets(WalkProfilePanel(; ng = ["g0", "gz"]), rdz) ==
          (1:8 .!= 8)
end

@testset "7. The scaled walk does not overflow or underflow" begin
    A = parity_load("Walk", "G0", "A")
    X = randn(StableRNG(7), 30, 8)
    de = WalkDistance(; nrm = nothing, sep = HopCount(; n = 4), decay = nothing)
    for f in (1e250, 1e-250, 1.0)
        _, _, DF = walk_measure(de, walk_rd([A .* f]; X = X))
        _, _, DB = walk_measure(de, walk_rd([big.(A) .* big(f)]; X = big.(X)))
        # Weights of 1e250 reach 1e1000 at four hops, where the oracle raises.
        @test all(isfinite, DF)
        @test isapprox(DF, Float64.(DB); atol = 1e-13)
    end
    # A decaying weight and a dead walk, an asset with no out-edge under the self term.
    A1 = parity_load("Walk", "G1", "A")
    de = WalkDistance(; nrm = nothing, sep = HopCount(; n = 3), self = true)
    _, _, D = walk_measure(de, walk_rd([A1]))
    ws = [exp(-log(2) * h) for h in 0:3]
    @test isapprox(D,
                   walk_reference_distance(walk_reference(A1, WALK_NODES[1:8], ws;
                                                          nrm = false, self = true));
                   atol = 1e-13)
end

@testset "8. LinearDecay and ReciprocalDecay" begin
    A = parity_load("Walk", "G0", "A")
    rd = walk_rd([A])
    for (decay, ws) in ((LinearDecay(), [4, 3, 2, 1]),
                        (ReciprocalDecay(; power = 2.0), [inv((1 + h)^2.0) for h in 0:3]))
        for nrm in (WeightShare(), nothing)
            _, _, D = walk_measure(WalkDistance(; decay = decay, nrm = nrm,
                                                sep = HopCount(; n = 3)), rd)
            P = walk_reference(A, rd.nx, ws; nrm = !isnothing(nrm))
            @test isapprox(D, walk_reference_distance(P); atol = 1e-13)
        end
    end
end

@testset "9. A FractionOf budget and a rule resolve over the graph in force" begin
    A = parity_load("Walk", "G0", "A")
    rd = walk_rd([A])
    ape = WalkProfilePanel(; sep = HopCount(; n = FractionOf(0.5)))
    M = PortfolioOptimisers.walk_operator(ape, OutEdges(), permutedims(A))
    g = PortfolioOptimisers.walk_graph(M)
    v = [d for s in 1:8
         for (t, d) in enumerate(PortfolioOptimisers.Graphs.gdistances(g, s))
         if t != s && d < typemax(d)]
    n = round(Int, quantile(v, 0.5))
    @test PortfolioOptimisers.resolve_separation(ape.sep, ape, rd.X, M, collect(1:8)) ==
          HopCount(n)
    @test last(walk_measure(WalkDistance(; sep = HopCount(; n = FractionOf(0.5))), rd)) ==
          last(walk_measure(WalkDistance(; sep = HopCount(n)), rd))
    # A CountOf builds no graph, and returns the budget as it is.
    @test PortfolioOptimisers.resolve_separation(HopCount(3), ape, rd.X, M, collect(1:8)) ==
          HopCount(3)
    # A rule receives the producer and the graph of the steps.
    rule = (ape, X, g; dims = 1) -> PortfolioOptimisers.Graphs.nv(g) > 5 ? 3 : 1
    @test last(walk_measure(WalkDistance(; sep = HopCount(; n = rule)), rd)) ==
          last(walk_measure(WalkDistance(; sep = HopCount(3)), rd))
    @test_throws ArgumentError walk_measure(WalkDistance(;
                                                         sep = HopCount(;
                                                                        n = (args...; kwargs...) -> 2.5)),
                                            rd)
    # A fraction over a graph that the walk cannot leave refuses.
    @test_throws ArgumentError asset_panel(WalkProfilePanel(;
                                                            sep = HopCount(;
                                                                           n = FractionOf(0.5))),
                                           nothing, walk_rd([zeros(13, 13)]), rd.X)
end

@testset "10. The constructor and the graph requirements" begin
    @test_throws IsEmptyError WalkProfilePanel(; ng = "")
    @test_throws IsEmptyError WalkProfilePanel(; prefix = "")
    @test_throws IsEmptyError WalkProfilePanel(; ng = String[])
    @test_throws ArgumentError WalkProfilePanel(; ng = ["a", "a"])
    # A walk has no radius, so a path length fails at dispatch.
    @test_throws TypeError WalkProfilePanel(; sep = PathLength())
    # The factory does not forward the prefix.
    @test_throws MethodError WalkDistance(; prefix = "x")
    de = WalkDistance(; ng = "g", dir = BothEdges())
    @test de.metric == OrthantChordDist()
    @test de.sim == OrthantChordSimilarity()
    @test de.ape == WalkProfilePanel(; ng = "g", dir = BothEdges())
    @test PortfolioOptimisers.graph_requirements(de.ape) == (NonNegativeWeights(),)
    @test PortfolioOptimisers.graph_names(de.ape) == "g"
    # A negative weight in any snapshot fails at the entry of the user verb.
    A = parity_load("Walk", "G0", "A")
    An = copy(A)
    An[1, 9] = -0.5
    rd = walk_rd([An])
    cle = ClustersEstimator(; de = WalkDistance())
    @test_throws r"cannot read the Exogenous Graph" optimise(HierarchicalRiskParity(;
                                                                                    ser = HierarchicalSeriation(;
                                                                                                                cle = cle)),
                                                             rd)
end

@testset "11. HRP, Schur, NCO and HERC read the graph in force at each fold" begin
    g0 = ExogenousGraph(; nodes = WALK_NODES, A = parity_load("Walk", "G0", "A"))
    g1 = ExogenousGraph(; nodes = WALK_NODES, A = parity_load("Walk", "G1", "A"))
    rd = ReturnsResult(; nx = WALK_NODES[1:8], X = parity_load("Walk", "Herc", "X"),
                       ng = ["supply"],
                       G = [ExogenousGraphSeries(; ts = [1, 61], graphs = [g0, g1])])
    cle = ClustersEstimator(; de = WalkDistance(),
                            alg = HClustAlgorithm(; linkage = WardLinkage()))
    cv = IndexWalkForward(50, 20)
    opts = (HierarchicalRiskParity(; ser = HierarchicalSeriation(; cle = cle)),
            SchurComplementHierarchicalRiskParity(;
                                                  ser = HierarchicalSeriation(; cle = cle)),
            NestedClustered(; cle = cle, opti = InverseVolatility(),
                            opto = InverseVolatility()),
            HierarchicalEqualRiskContribution(; cle = cle))
    for opt in opts
        w4 = [x.res.w[4] for x in cross_val_predict(opt, rd, cv).pred]
        # The first fold ends before row 61 and reads g0. The others read g1, where a4 has no
        # out-edge.
        @test w4[1] > 0
        @test w4[2:end] == [0.0, 0.0]
    end
end

@testset "12. OrthantChordDist and OrthantChordSimilarity" begin
    m = OrthantChordDist()
    a, b = [1.0, 0.0, 2.0], [0.5, 1.0, 0.0]
    c = dot(a, b) / (norm(a) * norm(b))
    @test m(a, b) ≈ sqrt(1 - c)
    @test m(a, a) ≈ 0 atol = 1e-7
    @test m(zeros(3), zeros(3)) == 0
    @test m(zeros(3), a) == 1
    # A negative cosine clamps to the distance of two orthogonal vectors.
    @test m([1.0, 0.0], [-1.0, 0.0]) == 1
    @test m([1, 0], [0, 1]) isa Float64
    @test PortfolioOptimisers.Distances.result_type(m, Float32, Float32) == Float32
    Z = hcat(a, b, zeros(3), [-1.0, 0.0, -2.0])
    D = PortfolioOptimisers.Distances.pairwise(m, Z; dims = 2)
    @test D[1, 1] == 0
    @test D[3, 3] == 0
    @test D[1, 3] == D[3, 1] == 1
    @test D[1, 4] == 1
    @test D[1, 2] ≈ m(a, b)
    S = PortfolioOptimisers.distance_to_similarity(OrthantChordSimilarity(); D = D)
    @test S ≈ 1 .- D .^ 2
    @test PortfolioOptimisers.default_similarity(m) == OrthantChordSimilarity()
    @test OrthantChordSimilarity() isa
          PortfolioOptimisers.AbstractNonNegativeSimilarityMatrixAlgorithm
    @test isnothing(PortfolioOptimisers.assert_similarity_domain(OrthantChordSimilarity(),
                                                                 WalkDistance(), D))
    @test_throws DomainError PortfolioOptimisers.assert_similarity_domain(OrthantChordSimilarity(),
                                                                          WalkDistance(),
                                                                          [0.0 1.5;
                                                                           1.5 0.0])
end

@testset "13. An online fold reads the graph in force, as a batch fit does (#2286)" begin
    PO = PortfolioOptimisers
    T, N = 90, 6
    nx = ["a$i" for i in 1:N]
    X = randn(StableRNG(2286), T, N) ./ 100
    ts = Date(2024, 1, 1) .+ Day.(0:(T - 1))
    # The nodes that are not assets differ in each snapshot.
    function snapshot(seed, extra)
        n = N + length(extra)
        return ExogenousGraph(; nodes = vcat(nx, extra),
                              A = abs.(randn(StableRNG(seed), n, n)) .* (1 .- I(n)))
    end
    gs = [snapshot(1, ["s1", "s2"]), snapshot(2, ["s3"]), snapshot(3, ["s1", "s4", "s5"])]
    cle = ClustersEstimator(; de = WalkDistance(),
                            alg = HClustAlgorithm(; linkage = WardLinkage()))
    ser = HierarchicalSeriation(; cle = cle)
    function head(H, pe)
        opt = HierarchicalOptimiser(; pe = pe)
        return if H === HierarchicalEqualRiskContribution
            H(; opt = opt, cle = cle)
        else
            H(; opt = opt, ser = ser)
        end
    end
    rows(r, i) = PO.port_opt_view(r, i, :)
    G = (gs[1], ExogenousGraphSeries(; ts = [1, 41, 63], graphs = gs),
         ExogenousGraphSeries(; ts = ts[[1, 41, 63]], graphs = gs))
    for (g, t) in zip(G, (nothing, nothing, ts))
        rd = ReturnsResult(; nx = nx, X = X, ts = t, ng = ["supply"], G = [g])
        for H in (HierarchicalRiskParity, HierarchicalEqualRiskContribution,
                  SchurComplementHierarchicalRiskParity)
            opt = head(H, EmpiricalPrior())
            o = foldl((o, s) -> partial_fit!(o, rows(rd, s:s)), 31:70;
                      init = partial_fit!(opt, rows(rd, 1:30)))
            @test optimise(o).w ≈ optimise(opt, rows(rd, 1:70)).w rtol = 1e-10
            # Under a cap the window rolls over the change points.
            for (pe, train) in ((Online(EmpiricalPrior()), true),
                                (Online(EmpiricalPrior(); max_history = 40), false))
                on = cross_val_predict(head(H, pe), rd, OnlineIndexWalkForward(40, 10))
                ba = cross_val_predict(opt, rd,
                                       IndexWalkForward(40, 10; expand_train = train))
                @test length(on.pred) == length(ba.pred) == 5
                @test all(isapprox(a.res.w, b.res.w; rtol = 1e-10)
                          for (a, b) in zip(on.pred, ba.pred))
            end
        end
    end
end
