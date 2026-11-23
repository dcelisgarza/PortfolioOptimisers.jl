#=
The dependency exposure constraint (#2202, map #2162), row 13 of the decision of #2168.

`DependencySpace` is a constraint space, so `ExposureConstraintEstimator` takes rows in the
names of the nodes of an Exogenous Graph. A row `a` re-bases to `(Rᵀ a)ᵀ w`, with `R` the reach
of every asset at every node over the graph in force at the last training row. The oracle has
no such constraint, so the reference is the Dependency Exposure of #2173, which is at parity
with the oracle in `test_13j_dependency_exposure.jl`.
=#
using Test, PortfolioOptimisers, LinearAlgebra, SparseArrays, StableRNGs, Statistics, Dates,
      Clarabel, Logging
const PO = PortfolioOptimisers

const DS_X = randn(StableRNG(2202), 80, 4) .* 0.02 .+ [0.004 0.003 0.002 0.001]
const DS_NX = ["A", "B", "C", "D"]
const DS_TS = Date(2024, 1, 1) .+ Day.(0:79)
const DS_G1 = ExogenousGraph(;
                             edges = [("A", "S", 1.0), ("B", "S", 2.0), ("S", "T", 1.0),
                                      ("C", "A", 1.0), ("D", "T", 1.0)])
const DS_G2 = ExogenousGraph(; edges = [("A", "S", 1.0), ("D", "U", 2.0), ("S", "T", 1.0)])
const DS_SETS = UniverseSets(; dict = Dict("nx" => DS_NX, "hubs" => ["S", "U"]))
const DS_SLV = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                      check_sol = (; allow_local = true, allow_almost = true),
                      settings = Dict("verbose" => false))

ds_rd(g; kwargs...) = ReturnsResult(; nx = DS_NX, X = DS_X, ng = ["g"], G = [g], kwargs...)
const DS_RD = ds_rd(DS_G1)
const DS_SERIES = ds_rd(ExogenousGraphSeries(; ts = DS_TS[[1, 35]],
                                             graphs = [DS_G1, DS_G2]); ts = DS_TS)
function ds_ece(val; key = nothing, kwargs...)
    return ExposureConstraintEstimator(;
                                       lce = LinearConstraintEstimator(; key = key,
                                                                       val = val),
                                       space = DependencySpace(; kwargs...))
end
function ds_row(ece, rd; sets = DS_SETS, kwargs...)
    return vec(linear_constraints(ece, sets; rd = rd, kwargs...).ineq.A)
end
# The exposure that the row caps, on unit weights of each asset.
function ds_reach(node, rd; kwargs...)
    res = dependency_exposure(ones(4), DS_NX, rd, DependencyExposure(; kwargs...))
    return exposure_contributions(res, 1)[findfirst(==(node), res.nodes), :]
end

@testset "construction" begin
    s = DependencySpace()
    @test isnothing(s.ng) && s.reach === ShortestHops() && s.dir === OutEdges()
    @test DependencySpace <: PO.AbstractConstraintSpace
    @test PO.graph_requirements(s) == (NonNegativeWeights(),)
    @test isnothing(PO.graph_names(s)) && PO.graph_names(DependencySpace(; ng = "g")) == "g"
    @test_throws PO.IsEmptyError DependencySpace(; ng = "")
    @test_throws TypeError DependencySpace(; ng = ["g"])
    @test_throws TypeError DependencySpace(; reach = HopCount(1))
    # A view of the assets returns the space: the graph is found by names.
    ece = ds_ece("S <= 0.3"; ng = "g")
    @test PO.port_opt_view(ece, [1, 3]).space === ece.space
    # A column of a precomputed constraint has no name, so the space refuses it.
    lc = LinearConstraint(; ineq = PartialLinearConstraint(; A = [1.0 0.0 0.0], B = [0.3]))
    @test_throws ArgumentError ExposureConstraintEstimator(; lce = lc, space = s)
    @test_throws ArgumentError ExposureConstraintEstimator(; lce = PO.LcE_Lc[ece.lce, lc],
                                                           space = s)
    @test ExposureConstraintEstimator(; lce = [ece.lce, ece.lce], space = s).lce ==
          [ece.lce, ece.lce]
end

@testset "a row is the reach of the Dependency Exposure" begin
    A = [1.0 0 0 0 0.5 0 0; 0 0 2 0 1 0 0; 0 1 0 0 0 0 3; 0 0 0 0 0 1 0; 0 0 0 0 0 1 1;
         1 0 0 0 0 0 0; 0 0 0 0 1 0 0]
    nodes = [DS_NX; "S"; "T"; "U"]
    for M in (A, sparse(A)), dir in (OutEdges(), InEdges(), BothEdges(), AnyEdges())
        g = ExogenousGraph(; nodes = nodes, A = M)
        rd = ds_rd(g)
        for reach in (ShortestHops(), ShortestHops(; sep = HopCount(1)),
                      ShortestHops(; sep = HopCount(3), decay = ReciprocalDecay()),
                      WalkShares(; sep = HopCount(2)))
            kw = (; reach = reach, dir = dir)
            for node in nodes
                # A zero row is dropped, so a second node keeps each row.
                a = ds_row(ds_ece("2*$node - 3*B <= 0.1"; kw...), rd)
                @test a ≈ 2 * ds_reach(node, g; kw...) - 3 * ds_reach("B", g; kw...)
            end
        end
    end
    # The sign of `>=`, an equality, and the right-hand side.
    lc = linear_constraints(ds_ece(["S >= 0.2", "T == 0.5"]), DS_SETS; rd = DS_RD)
    @test vec(lc.ineq.A) == -ds_reach("S", DS_G1) && lc.ineq.B == [-0.2]
    @test vec(lc.eq.A) == ds_reach("T", DS_G1) && lc.eq.B == [0.5]
    @test ds_reach("S", DS_G1) == [1, 1, 1, 0]
    # A vector of wrapped constraints gives one result for each.
    v = linear_constraints(ExposureConstraintEstimator(;
                                                       lce = [LinearConstraintEstimator(;
                                                                                        val = "S <= 0.3"),
                                                              LinearConstraintEstimator(;
                                                                                        val = "T <= 0.3")],
                                                       space = DependencySpace()), DS_SETS;
                           rd = DS_RD)
    @test [vec(x.ineq.A) for x in v] == [ds_reach("S", DS_G1), ds_reach("T", DS_G1)]
end

@testset "the names of a row" begin
    # A node that is not an asset, an asset, and a group.
    @test ds_row(ds_ece("A <= 0.3"), DS_RD) == [1, 0, 1, 0]
    # The series holds U only from row 35, and the graph in force does not hold B and C.
    @test ds_row(ds_ece("hubs <= 0.3"), DS_SERIES) == [1, 0, 0, 1]
    @test ds_row(ds_ece("S + C <= 0.3"), DS_SERIES) == [1, 0, 0, 0]
    early = PO.port_opt_view(DS_SERIES, 1:20, :)
    # U is in no snapshot that the early rows see, so the row drops or raises.
    msg = "variable `U` not in node universe (6 nodes under key `g`)"
    @test isnothing(@test_logs (:warn, Regex("\\Q$msg\\E")) match_mode = :any linear_constraints(ds_ece("hubs <= 0.3"),
                                                                                                 DS_SETS;
                                                                                                 rd = early))
    @test_throws ArgumentError linear_constraints(ds_ece("hubs <= 0.3"), DS_SETS;
                                                  rd = early, strict = true)
    # The declared names keep the row, with zero exposure to U.
    sets = UniverseSets(;
                        dict = Dict("nx" => DS_NX, "hubs" => ["S", "U"],
                                    "nodes" => ["U", "T", "S"]))
    @test ds_row(ds_ece("hubs <= 0.3"; key = "nodes"), early; sets = sets, strict = true) ==
          [1, 1, 1, 0]
    @test ds_row(ds_ece("U + T <= 0.3"; key = "nodes"), DS_SERIES; sets = sets) ==
          [1, 0, 0, 1]
    @test_throws KeyError ds_row(ds_ece("T <= 0.3"; key = "nope"), early; sets = sets)
    # A node that no asset reaches.
    zmsg = r"resolved against the node universe \(3 nodes under key `nodes`\) but projected to an all-zero row over 4 assets: every matched node has zero loadings"
    @test isnothing(@test_logs (:warn, zmsg) match_mode = :any linear_constraints(ds_ece("U <= 0.3";
                                                                                         key = "nodes"),
                                                                                  sets;
                                                                                  rd = early))
    @test_throws IsZeroError linear_constraints(ds_ece("U <= 0.3"; key = "nodes"), sets;
                                                rd = early, strict = true)
    # An asset that left the universe can still be a node, so a group keeps it.
    red = PO.non_investable_sets(UniverseSets(;
                                              dict = Dict("nx" => ["B", "C", "D"],
                                                          "grp" => ["A", "S"])), ["A"])
    a = ds_row(ds_ece("grp <= 0.3"), DS_RD; sets = red)
    @test a == [1, 2, 0]
    # A node name with a space, through a group.
    spaced = ds_rd(ExogenousGraph(; edges = [("A", "The Bank", 1.0)]))
    bank = UniverseSets(; dict = Dict("nx" => DS_NX, "bank" => ["The Bank"]))
    @test ds_row(ds_ece("bank <= 0.3"), spaced; sets = bank) == [1, 0, 0, 0]
    # An asset that is not a node adds nothing.
    other = UniverseSets(; dict = Dict("nx" => ["A", "zz"]))
    @test ds_row(ds_ece("S <= 0.3"), DS_RD; sets = other) == [1, 0]
end

@testset "the graph of the returns data" begin
    ece = ds_ece("S <= 0.3")
    @test_throws PO.IsNothingError linear_constraints(ece, DS_SETS)
    @test_throws PO.IsNothingError linear_constraints(ece, DS_SETS;
                                                      rd = ReturnsResult(; nx = DS_NX,
                                                                         X = DS_X))
    @test_throws ArgumentError linear_constraints(ds_ece("S <= 0.3"; ng = "h"), DS_SETS;
                                                  rd = DS_RD)
    # A fit cannot continue with no graph in force.
    gap = ds_rd(ExogenousGraphSeries(; ts = [1, 20], graphs = [DS_G1, nothing]))
    @test_throws ArgumentError linear_constraints(ece, DS_SETS; rd = gap)
    @test ds_row(ece, PO.port_opt_view(gap, 1:15, :)) == [1, 1, 1, 0]
    # A negative weight is refused in a direct call and at the entry of the verb, at any key.
    neg = ExogenousGraph(; edges = [("A", "S", -1.0)])
    @test_throws ArgumentError linear_constraints(ece, DS_SETS; rd = ds_rd(neg))
    jopt = JuMPOptimiser(; slv = DS_SLV, sets = DS_SETS, lcse = ece)
    late = ds_rd(ExogenousGraphSeries(; ts = [1, 5], graphs = [neg, DS_G1]))
    err = try
        optimise(MeanRisk(; opt = jopt), late)
    catch e
        sprint(showerror, e)
    end
    @test occursin("DependencySpace at `opt.cbe.lcse.space`", err) ||
          occursin("DependencySpace at `opt.lcse.space`", err)
    @test occursin("at the key 1", err)
end

@testset "a union of graphs" begin
    g1 = ExogenousGraph(; edges = [("A", "S", 1.0), ("B", "S", 1.0)])
    g2 = ExogenousGraph(; edges = [("S", "Bank", 1.0), ("C", "Bank", 1.0)])
    rd = ReturnsResult(; nx = DS_NX, X = DS_X, ng = ["supply", "credit"], G = [g1, g2])
    gu = GraphUnion(; ng = ["supply", "credit"])
    # The path A → S → Bank changes relation at S.
    @test ds_row(ds_ece("Bank <= 0.3"; ng = gu), rd) == [1, 1, 1, 0]
    @test ds_row(ds_ece("Bank <= 0.3"; ng = "credit"), rd) == [0, 0, 1, 0]
    @test_throws ArgumentError linear_constraints(ds_ece("Bank <= 0.3"), DS_SETS; rd = rd)
    u = graph_union(g1, g2)
    for dir in (InEdges(), BothEdges(), AnyEdges())
        @test ds_row(ds_ece("A + Bank <= 0.3"; ng = gu, dir = dir), rd) ==
              ds_reach("A", u; dir = dir) + ds_reach("Bank", u; dir = dir)
    end
    # A normalised union normalises each part along each direction of the reach.
    gn = GraphUnion(; ng = ["supply", "credit"], nrm = WeightShare())
    kw = (; reach = WalkShares(; sep = HopCount(2)), dir = BothEdges())
    res = dependency_exposure(Matrix(1.0I, 80, 4), rd, DependencyExposure(; ng = gn, kw...))
    a = ds_row(ds_ece("Bank + S <= 0.3"; ng = gn, kw...), rd)
    @test a ≈ [sum(sum(res.L[i, findfirst(==(n), res.nodes), :]) for n in ("Bank", "S"))
               for i in 1:4]
end

function ds_opt(val; sets = DS_SETS, kwargs...)
    return MeanRisk(; obj = MaximumReturn(),
                    opt = JuMPOptimiser(; slv = DS_SLV, sets = sets,
                                        lcse = ds_ece(val; kwargs...)))
end
@testset "the optimised exposure is the requested one" begin
    res = optimise(ds_opt("S <= 0.3"), DS_RD)
    e = dependency_exposure(res, DS_RD)
    tot(e, n) = sum(e.L[1, findfirst(==(n), e.nodes), :] +
                    e.S[1, findfirst(==(n), e.nodes), :])
    @test isapprox(tot(e, "S"), 0.3; atol = 1e-6)
    @test vec(res.lcsr.ineq.A) == ds_reach("S", DS_G1)
    # Shorts: the cap is on the net exposure.
    sh = optimise(MeanRisk(; obj = MaximumReturn(),
                           opt = JuMPOptimiser(; slv = DS_SLV, sets = DS_SETS,
                                               wb = WeightBounds(-1, 1), sbgt = 0.5,
                                               lcse = ds_ece("S <= 0.2"))), DS_RD)
    @test tot(dependency_exposure(sh, DS_RD), "S") <= 0.2 + 1e-6
end

@testset "each fold reads the graph of its last training row" begin
    opt = ds_opt("S <= 0.4")
    pred = cross_val_predict(opt, DS_SERIES, IndexWalkForward(20, 10))
    rows = [vec(f.res.lcsr.ineq.A) for f in pred.pred]
    # Fold k trains on the rows up to 10k + 10, and the key of DS_G2 is row 35.
    @test rows == [ds_reach("S", 10k + 10 >= 35 ? DS_G2 : DS_G1) for k in 1:6]
    @test rows[1] != rows[end]
    for f in pred.pred
        w = collect(PO.fold_target(f))
        @test dot(vec(f.res.lcsr.ineq.A), w) <= 0.4 + 1e-6
    end
end

@testset "meta-optimisers" begin
    jo = MeanRisk(;
                  opt = JuMPOptimiser(; slv = DS_SLV, sets = DS_SETS,
                                      lcse = ds_ece("S <= 1")))
    plain = MeanRisk(; opt = JuMPOptimiser(; slv = DS_SLV))
    # An inner solve views the assets and reads the graph by their names. A cluster that the
    # node does not reach drops the row with a warning.
    nco = NestedClustered(; cle = ClustersEstimator(), opti = jo, opto = plain,
                          sets = DS_SETS)
    res = with_logger(NullLogger()) do
        return optimise(nco, DS_RD)
    end
    @test isapprox(sum(res.w), 1; atol = 1e-6)
    for (k, r) in enumerate(res.resi)
        cl = findall(==(k), PO.assignments(res.clr))
        a = ds_reach("S", DS_G1)[cl]
        @test all(iszero, a) ? isnothing(r.lcsr) : vec(r.lcsr.ineq.A) == a
    end
    # The names of an outer problem are synthetic, so the outer optimiser refuses the space.
    @test_throws ArgumentError NestedClustered(; cle = ClustersEstimator(), opti = plain,
                                               opto = jo)
    @test_throws ArgumentError Stacking(; opti = [plain], opto = jo)
    @test Stacking(; opti = [jo], opto = plain) isa Stacking
    # The walk reaches a vector, a schedule and an optimiser of an optimiser.
    ece = ds_ece("S <= 1")
    vece = PO.EcE_LcE_Lc[LinearConstraintEstimator(; val = "A >= 0.05"), ece]
    td = TimeDependent([ece, ece])
    for l in (vece, td)
        @test_throws ArgumentError Stacking(; opti = [plain],
                                            opto = MeanRisk(;
                                                            opt = JuMPOptimiser(;
                                                                                slv = DS_SLV,
                                                                                sets = DS_SETS,
                                                                                lcse = l)))
    end
    @test_throws ArgumentError Stacking(; opti = [plain],
                                        opto = SubsetResampling(; subset_size = 0.8,
                                                                opt = jo))
    @test PO.holds_outer_space(("a" => ece, 1)) && !PO.holds_outer_space([1.0, 2.0])
    @test !PO.holds_outer_space([1.0 2.0]) && !PO.holds_outer_space(plain)
end

@testset "a pipeline step needs no prior" begin
    ece = ds_ece("S <= 0.3")
    @test PO.pipe_reads(ece) == (:returns,)
    @test PO.pipe_reads(ExposureConstraintEstimator(; lce = ece.lce, space = FactorSpace())) ==
          (:returns, :prior)
    opt = MeanRisk(; obj = MaximumReturn(), opt = JuMPOptimiser(; slv = DS_SLV))
    direct = optimise(ds_opt("S <= 0.3"), DS_RD).w
    for steps in ((ece, opt), (EmpiricalPrior(), ece, opt))
        res = fit(Pipeline(; steps = steps), DS_RD)
        @test vec(res.ctx.constraints.ineq.A) == ds_reach("S", DS_G1)
        @test isapprox(res.w, direct; rtol = 1e-6)
    end
end

# A space of a user that reads an exposure table, with the names in the basis.
struct DSTableSpace{T} <: PO.AbstractConstraintSpace
    basis::T
end
PO.constraint_space_basis(s::DSTableSpace, sets, rr, rd) = (s.basis, nothing)
@testset "a space of a user returns a ConstraintBasis" begin
    cb = PO.ConstraintBasis(; M = [1 0; 1 0; 0 1; 0 2], nx = ["USD", "EUR"],
                            axis = "currency", key = "table")
    @test_throws DimensionMismatch PO.ConstraintBasis(; M = [1 0; 1 0], nx = ["USD"],
                                                      axis = "currency", key = "table")
    sp = DSTableSpace(cb)
    ece(l) = ExposureConstraintEstimator(; lce = l, space = sp)
    @test ds_row(ece(LinearConstraintEstimator(; val = "EUR + USD <= 0.2")), nothing) ==
          [1, 1, 1, 2]
    # A fixed table has fixed columns, so a precomputed constraint re-bases through it.
    lc = LinearConstraint(; ineq = PartialLinearConstraint(; A = [0.0 1.0], B = [0.3]))
    @test ds_row(ece(lc), nothing) == [0, 0, 1, 2]
    v = linear_constraints(ece(PO.LcE_Lc[lc, LinearConstraintEstimator(; val = "USD <= 1")]),
                           DS_SETS)
    @test [vec(x.ineq.A) for x in v] == [[0, 0, 1, 2], [1, 1, 0, 0]]
    @test PO.pipe_reads(ece(lc)) == (:returns, :prior)
end
