#=
The position constraint (#2297, map #2162), as the decision of #2290 states it.

`PositionConstraintEstimator` applies the rows of any entry of `lcse` to a reading of the
weights: gross, long, short or net. The reference of a cap on a node is the Dependency Exposure,
which is at parity with the oracle in `test_13j_dependency_exposure.jl`. The oracle has no such
constraint.
=#
using PortfolioOptimisers, Test, Clarabel, HiGHS, Pajarito, StableRNGs, JuMP, LinearAlgebra,
      Dates, Logging
@testset "Position constraint (#2297)" begin
    PO = PortfolioOptimisers
    cslv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false))
    hslv = Solver(; name = :highs, solver = HiGHS.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = "log_to_console" => false)
    nx = ["A", "B", "C", "D", "E", "F"]
    N = length(nx)
    T = 120
    # A drift that falls with the index, so a maximum return book is long A and short F.
    X = 0.01 .* randn(StableRNG(2297), T, N) .+ [0.004 0.003 0.0015 0.001 -0.001 -0.003]
    ts = Date(2024, 1, 1) .+ Day.(0:(T - 1))
    # S reaches A, B, E and F, so a hedged book has a small net and a large gross exposure.
    g1 = ExogenousGraph(;
                        edges = [("A", "S", 1.0), ("B", "S", 2.0), ("E", "S", 1.0),
                                 ("F", "S", 1.0), ("C", "T", 1.0), ("D", "C", 1.0)])
    g2 = ExogenousGraph(; edges = [("A", "S", 1.0), ("F", "S", 1.0), ("B", "T", 1.0)])
    rd = ReturnsResult(; nx = nx, X = X, ng = ["g"], G = [g1])
    rds = ReturnsResult(; nx = nx, X = X, ts = ts, ng = ["g"],
                        G = [ExogenousGraphSeries(; ts = ts[[1, 60]], graphs = [g1, g2])])
    sets = UniverseSets(;
                        dict = Dict("nx" => nx, "tech" => ["A", "F"], "mid" => ["C", "D"],
                                    "nf" => ["MTUM", "VLUE"]))
    s1 = [1, 1, 0, 0, 1, 1]
    ls = WeightBounds(; lb = -1.0, ub = 1.0)
    tol = 1e-5
    lce(val; kwargs...) = LinearConstraintEstimator(; val = val, kwargs...)
    function node(val; kwargs...)
        return ExposureConstraintEstimator(; lce = lce(val),
                                           space = DependencySpace(; kwargs...))
    end
    pce(l, pos = GrossPosition()) = PositionConstraintEstimator(; lce = l, pos = pos)
    function solve(lcse; slv = cslv, obj = MaximumReturn(), wb = ls, bgt = nothing,
                   data = rd, kwargs...)
        opt = JuMPOptimiser(; slv = slv, sets = sets, wb = wb, bgt = bgt, lcse = lcse,
                            kwargs...)
        return with_logger(NullLogger()) do
            return optimise(MeanRisk(; r = ConditionalValueatRisk(), obj = obj, opt = opt),
                            data)
        end
    end
    ok(res) = isa(res.retcode, OptimisationSuccess)
    long(w, a = s1) = dot(a, max.(w, 0))
    short(w, a = s1) = dot(a, max.(-w, 0))
    gross(w, a = s1) = dot(a, abs.(w))
    tech = [1, 0, 0, 0, 0, 1]

    @testset "Construction and resolution" begin
        p = PositionConstraintEstimator(; lce = lce("tech <= 0.4"))
        @test p.pos === GrossPosition()
        @test_throws PO.IsEmptyError PositionConstraintEstimator(;
                                                                 lce = LinearConstraintEstimator[])
        # A vector that mixes the shapes, and no wrapper in a wrapper.
        lc = LinearConstraint(;
                              ineq = PartialLinearConstraint(; A = [1.0 0 0 0 0 0],
                                                             B = [0.3]))
        mix = PositionConstraintEstimator(; lce = [lce("A <= 0.5"), lc, node("S <= 0.3")])
        @test eltype(mix.lce) <: PO.EcE_LcE_Lc
        @test_throws ArgumentError PositionConstraintEstimator(;
                                                               lce = [lce("A <= 0.5"), mix])
        @test_throws TypeError PositionConstraintEstimator(; lce = p)
        @test_throws TypeError PositionConstraintEstimator(; lce = lc, pos = :gross)
        # One Result for each row block, with the rows of the wrapped constraint.
        r = linear_constraints(p, sets)
        @test r isa PositionConstraint && r.pos === GrossPosition()
        @test r.lc.ineq.A == linear_constraints(p.lce, sets).ineq.A
        @test PositionConstraint(; lc = lc).pos === GrossPosition()
        rs = linear_constraints(mix, sets; rd = rd)
        @test length(rs) == 3 && all(x -> x isa PositionConstraint, rs)
        @test rs[2].lc === lc && vec(rs[3].lc.ineq.A) == s1
        # A precomputed constraint needs no names, alone, in the wrapper and in a vector.
        @test linear_constraints(r) === r
        @test linear_constraints(pce(lc), nothing).lc === lc
        @test linear_constraints(PO.PcE_Pc_EcE_LcE_Lc[pce(lc), r, lc], nothing)[2] === r
        vl = [lc, lc]
        @test linear_constraints(vl, nothing; datatype = Float64) === vl
        # A dropped row is nothing, and an entry that wraps a vector is flat in the slot.
        drop = with_logger(NullLogger()) do
            return linear_constraints(pce(lce("ZZ <= 1")), sets)
        end
        @test isnothing(drop)
        flat = PO.flat_linear_constraints
        @test isnothing(flat(nothing)) && flat(r) === r
        v = PO.Pc_Lc[r, lc]
        @test flat(v) === v
        @test flat([[r, lc], nothing, r]) == [r, lc, r]
        @test eltype(flat([[r, lc], nothing])) <: PO.Pc_Lc
        @test isnothing(flat([nothing, nothing]))
    end

    @testset "Views, widths and readers" begin
        lc = LinearConstraint(;
                              ineq = PartialLinearConstraint(; A = [1.0 0 0 0 0 0],
                                                             B = [0.3]))
        pc = PositionConstraint(; lc = lc, pos = ShortPosition())
        @test PO.port_opt_view(pc, 1:2) == pc
        fs = FactorSpace(; re = Regression(; M = Matrix(1.0I, N, 2)))
        pe = pce(ExposureConstraintEstimator(; lce = lce("MTUM <= 0.3"), space = fs))
        @test size(PO.port_opt_view(pe, 1:3).lce.space.re.M) == (3, 2)
        @test PO.port_opt_view(pe, 1:3).pos === pe.pos
        @test length(PO.port_opt_view(pce([lce("A <= 1"), pe.lce]), 1:3).lce) == 2
        w = PO.assert_investable_constraint_width
        @test isnothing(w(pc, N, "lcse")) && isnothing(w([pc, lc], N, "lcse"))
        @test_throws DimensionMismatch w(pc, N - 1, "lcse")
        @test_throws DimensionMismatch w([lc, pc], N - 1, "lcse")
        rr = pce(ExposureConstraintEstimator(; lce = lce("MTUM <= 0.3"),
                                             space = FactorSpace(;
                                                                 re = DimensionReductionRegression())))
        fr = PO.reader_frames(rr, "lcse", false)
        fl = PO.reader_frames(rr.lce, "lcse.lce", false)
        @test fr.excess == fl.excess && fr.raw == fl.raw
        @test PO.name_keyed(rr) && !PO.name_keyed(pce(lc))
        @test PO.exposure_keyed(rr) && !PO.exposure_keyed(pce(lce("A <= 1")))
        @test PO.exposure_keyed([pce([lce("A <= 1"), rr.lce])])
        # A fold reduction that drops an asset meets the check of the width.
        @test_throws DimensionMismatch optimise(MeanRisk(;
                                                         opt = JuMPOptimiser(; slv = cslv,
                                                                             lcse = pc)),
                                                ReturnsResult(; nx = nx[1:5],
                                                              X = X[:, 1:5]))
    end

    @testset "The readings" begin
        w = [0.5, -0.2, 0.0]
        lw, sw = max.(w, 0), max.(-w, 0)
        position_reading_ = PO.position_reading
        rdg(p) = position_reading_(p, w, lw, sw)
        @test rdg(GrossPosition()) == [0.5, 0.2, 0.0]
        @test rdg(NetPosition()) == w
        @test rdg(LongPosition()) == lw
        @test rdg(ShortPosition()) == sw
        for p in (GrossPosition(), NetPosition(), LongPosition())
            @test PO.long_only_reading(p, lw) === lw
        end
        @test PO.long_only_reading(ShortPosition(), lw) == zeros(3)
    end

    @testset "A gross cap holds on a long-short book, and the net cap does not" begin
        base = solve(nothing)
        @test ok(base) && gross(base.w) > 1 && short(base.w) > 0.5
        net = solve(node("S <= 0.3"))
        @test dot(s1, net.w) <= 0.3 + tol && gross(net.w) > 0.3 + 0.1
        for (l, a) in ((node("S <= 0.3"), s1), (lce("tech <= 0.3"), tech),
                       (lce("F <= 0.3"), [0, 0, 0, 0, 0, 1]))
            res = solve(pce(l))
            @test ok(res) && gross(res.w, a) <= 0.3 + tol
            @test haskey(res.model, :lcs_ineq_1) && haskey(res.model, :lcs_pos_1)
            @test res.lcsr isa PositionConstraint
        end
        # The cap binds.
        @test isapprox(gross(solve(pce(node("S <= 0.3"))).w), 0.3; atol = 1e-4)
        # The other budgets.
        for kw in ((; gbgt = BudgetRange(; lb = nothing, ub = 1.8)), (; gbgt = 1.8),
                   (; bgt = 1, sbgt = 0.4), (; bgt = 1, sbgt = BudgetRange(; lb = 0.0, ub = 0.4)),
                   (; gbgt = BudgetRange(; lb = nothing, ub = 1.8), xbgt = true, slv = hslv),
                   (; bgt = 1, sbgt = 0.4, obj = MaximumRatio()), (; obj = MaximumRatio()))
            res = solve(pce(node("S <= 0.3")); kw...)
            @test ok(res) && gross(res.w) <= 0.3 + tol
            res = solve(node("S <= 0.3"); kw...)
            @test ok(res) && gross(res.w) > 0.3 + 0.05
        end
    end

    @testset "The long and short caps, and the report" begin
        kw = (; bgt = 1, sbgt = BudgetRange(; lb = 0.0, ub = 0.6))
        for reach in (ShortestHops(), WalkShares(; sep = HopCount(2)))
            lc = node("S <= 0.2"; ng = "g", reach = reach)
            for (pos, part) in
                ((LongPosition(), e -> sum(e.L[1, findfirst(==("S"), e.nodes), :])),
                 (ShortPosition(), e -> -sum(e.S[1, findfirst(==("S"), e.nodes), :])),
                 (GrossPosition(),
                  e -> sum(e.L[1, findfirst(==("S"), e.nodes), :]) -
                       sum(e.S[1, findfirst(==("S"), e.nodes), :])))
                res = solve(pce(lc, pos); kw...)
                e = dependency_exposure(res, rd,
                                        DependencyExposure(; ng = "g", reach = reach))
                a = vec(res.lcsr.lc.ineq.A)
                f = isa(pos, LongPosition) ? long : isa(pos, ShortPosition) ? short : gross
                # The exposure of the weights is the part of the report.
                @test isapprox(f(res.w, a), part(e); atol = 1e-8)
                @test isapprox(part(e), 0.2; atol = 1e-4)
                un = dependency_exposure(solve(nothing; kw...), rd,
                                         DependencyExposure(; ng = "g", reach = reach))
                @test part(un) > 0.25
            end
        end
        # The net reading is the plain row.
        @test solve(pce(node("S <= 0.2"), NetPosition())).w == solve(node("S <= 0.2")).w
    end

    @testset "Long-only weights" begin
        wb = WeightBounds(; lb = 0.0, ub = 1.0)
        plain = solve(lce("tech <= 0.4"); wb = wb, bgt = 1)
        @test isapprox(dot(tech, plain.w), 0.4; atol = tol)
        nv = num_variables(plain.model)
        for pos in (GrossPosition(), LongPosition(), NetPosition())
            res = solve(pce(lce("tech <= 0.4"), pos); wb = wb, bgt = 1)
            @test res.w == plain.w && num_variables(res.model) == nv
        end
        free = solve(nothing; wb = wb, bgt = 1)
        res = solve(pce(lce("tech <= 0.4"), ShortPosition()); wb = wb, bgt = 1)
        @test isapprox(res.w, free.w; atol = 1e-6) && num_variables(res.model) == nv
        for s in (cslv, hslv)
            @test !ok(solve(pce(lce("tech >= 0.1"), ShortPosition()); wb = wb, bgt = 1,
                            slv = s))
        end
    end

    @testset "A floor and a signed row" begin
        # The assets of the floor have no weight, so no realised position meets it.
        wbz = WeightBounds(; lb = [-1.0, -1, 0, 0, -1, -1], ub = [1.0, 1, 0, 0, 1, 1])
        floor_mid = pce(lce("mid >= 0.4"))
        loose = solve(floor_mid; wb = wbz)
        @test ok(loose) && gross(loose.w, [0, 0, 1, 1, 0, 0]) <= tol
        @test !ok(solve(floor_mid; wb = wbz, xbgt = true, slv = hslv))
        mr = (; obj = MinimumRisk(), xbgt = true, slv = hslv)
        pinned = solve(pce(lce("tech >= 1.2")); mr...)
        @test ok(pinned) && gross(pinned.w, tech) >= 1.2 - tol
        # A factor with loadings of both signs.
        M = [1.0 0.0; 0.8 0.0; 0.0 1.0; 0.0 1.0; -1.0 0.0; -0.9 0.0]
        fac = ExposureConstraintEstimator(; lce = lce("MTUM <= 0.2"),
                                          space = FactorSpace(; re = Regression(; M = M)))
        f = solve(pce(fac); xbgt = true, slv = hslv)
        @test ok(f) && gross(f.w, M[:, 1]) <= 0.2 + tol
        @test ok(solve(pce(fac)))
        # A cap over a non-negative factor holds without the pin.
        vl = ExposureConstraintEstimator(; lce = lce("VLUE <= 0.2"),
                                         space = FactorSpace(; re = Regression(; M = M)))
        @test gross(solve(pce(vl)).w, M[:, 2]) <= 0.2 + tol
        # An equality of the gross position, with the pin.
        eq = solve(pce(lce("tech == 0.7")); xbgt = true, slv = hslv)
        @test ok(eq) && isapprox(gross(eq.w, tech), 0.7; atol = tol)
        @test haskey(eq.model, :lcs_eq_1)
    end

    @testset "A vector mixes plain, re-based and position rows" begin
        lc = LinearConstraint(;
                              ineq = PartialLinearConstraint(; A = [0.0 1 0 0 0 0],
                                                             B = [0.35]))
        v = [lce("C <= 0.25"), node("S <= 0.5"), pce(node("S <= 0.9")),
             pce([lce("tech <= 0.4"), lce("mid <= 0.6")], LongPosition()), lc,
             PositionConstraint(; lc = lc)]
        res = solve(v)
        @test ok(res) && length(res.lcsr) == 7
        w = res.w
        @test w[3] <= 0.25 + tol && dot(s1, w) <= 0.5 + tol && gross(w) <= 0.9 + tol
        @test long(w, tech) <= 0.4 + tol && long(w, [0, 0, 1, 1, 0, 0]) <= 0.6 + tol
        @test abs(w[2]) <= 0.35 + tol
        @test all(haskey(res.model, Symbol(:lcs_ineq_, i)) for i in 1:7)
        @test haskey(res.model, :lcs_pos_3) && !haskey(res.model, :lcs_pos_1)
        # A dropped entry leaves the slot, and the others stay.
        res = solve([lce("ZZ <= 1"), pce(lce("tech <= 0.4"))])
        @test length(res.lcsr) == 1 && gross(res.w, tech) <= 0.4 + tol
        @test isnothing(solve([lce("ZZ <= 1"), pce(lce("ZZ <= 1"))]).lcsr)
        # A precomputed Result enters as given, and needs no sets.
        pc = PositionConstraint(; lc = lc)
        res = optimise(MeanRisk(; obj = MaximumReturn(),
                                opt = JuMPOptimiser(; slv = cslv, wb = ls, bgt = nothing,
                                                    lcse = pc)), rd)
        @test res.lcsr === pc && abs(res.w[2]) <= 0.35 + tol
        # The Result maps back to a block, and the resolver gives its value back.
        cbe = ConstraintBlock(; lcse = [pc, lc])
        cbr = PO.resolve_constraint_block(cbe, nothing, nothing)
        @test cbr.lcsr == [pc, lc]
        @test PO.constraint_block_from_resolved(cbe, cbr).lcse == [pc, lc]
    end

    @testset "The slots that refuse the wrapper" begin
        p = pce(lce("tech <= 0.4"))
        pc = linear_constraints(p, sets)
        for slot in (:gcarde, :sgcarde, :cte), x in (p, pc, [p], [pc])
            @test_throws Union{TypeError, ArgumentError, MethodError} ConstraintBlock(;
                                                                                      sets = sets,
                                                                                      NamedTuple{(slot,)}((x,))...)
        end
        for x in (p, pc)
            @test_throws Union{TypeError, MethodError} Variance(; rc = x)
        end
        # `ConstraintBlock` names an entry that no shape of the slot admits.
        @test_throws ArgumentError ConstraintBlock(; sets = sets, lcse = Any[p, 1])
    end

    @testset "The parts of a block" begin
        function model2()
            m = JuMP.Model()
            PO.set_model_scales!(m, 1, 1)
            PO.set_model_observations!(m, T)
            JuMP.@expression(m, k, 1)
            JuMP.@variable(m, w[1:N])
            PO.state_set!(m, :b_, :w, PO.get_w(m))
            PO.state_set!(m, :b_, :w_owner, Symbol(""))
            return m
        end
        pc = linear_constraints(pce(lce("tech <= 0.4")), sets)
        rows! = PO.set_linear_weight_constraints!
        lo = WeightBounds(; lb = 0.0, ub = 1.0)
        # A block after a long-only block makes its parts under its prefix.
        m = model2()
        PO.set_weight_constraints!(m, lo, 1, nothing)
        rows!(m, pc, :lcs_ineq_, :lcs_eq_; wb = lo)
        @test num_variables(m) == N
        PO.set_weight_constraints!(m, ls, nothing, nothing; prefix = :b_)
        rows!(m, pc, :lcs_ineq_, :lcs_eq_; wb = ls, prefix = :b_)
        @test num_variables(m) == 3N && haskey(m, :b_lw) && haskey(m, :b_sw)
        @test haskey(m, :b_lcs_ineq_1) && haskey(m, :lcs_ineq_1) && !haskey(m, :sw)
        # A later entry of the block reads the same parts, and the net reading needs none.
        m = model2()
        PO.set_weight_constraints!(m, lo, 1, nothing)
        net = PositionConstraint(; lc = pc.lc, pos = NetPosition())
        rows!(m, net, :lcs_ineq_, :lcs_eq_; wb = ls, prefix = :b_)
        @test num_variables(m) == N && !haskey(m, :b_lw)
        # A long-only block after a block with parts adds no variable.
        m = model2()
        PO.set_weight_constraints!(m, ls, 1, nothing)
        rows!(m, pc, :lcs_ineq_, :lcs_eq_; wb = ls)
        rows!(m, pc, :lcs_ineq_, :lcs_eq_; wb = lo, prefix = :b_)
        @test num_variables(m) == 3N
        # A free bound admits a short position, however the user states it.
        for wb in
            (WeightBounds(; lb = -Inf, ub = 1.0), WeightBounds(; lb = nothing, ub = 1.0))
            res = solve(pce(lce("tech <= 0.4")); wb = wb, bgt = 1,
                        sbgt = BudgetRange(; lb = 0.0, ub = 0.5))
            @test ok(res) && gross(res.w, tech) <= 0.4 + tol && haskey(res.model, :sw)
        end
    end

    @testset "The heads" begin
        p = pce(lce("tech <= 0.4"))
        pslv = Solver(; name = :pajarito,
                      solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                         "verbose" => false,
                                                         "oa_solver" =>
                                                             optimizer_with_attributes(HiGHS.Optimizer,
                                                                                       MOI.Silent() =>
                                                                                           true),
                                                         "conic_solver" =>
                                                             optimizer_with_attributes(Clarabel.Optimizer,
                                                                                       "verbose" =>
                                                                                           false)),
                      check_sol = (; allow_local = true, allow_almost = true))
        # A mixed-integer risk budget derives the weights from its own parts.
        function rb(lcse)
            opt = JuMPOptimiser(; slv = pslv, sets = sets, wb = ls,
                                sbgt = BudgetRange(; lb = 0.0, ub = 0.3), lcse = lcse)
            return optimise(RiskBudgeting(; r = StandardDeviation(), opt = opt,
                                          rba = AssetRiskBudgeting(;
                                                                   alg = MixedIntegerRiskBudgeting())),
                            rd)
        end
        free = rb(nothing)
        @test ok(free) && haskey(free.model, :sw)
        cap = 0.8 * gross(free.w, tech)
        res = rb(pce(lce("tech <= $cap")))
        @test ok(res) && gross(res.w, tech) <= cap + 1e-4
        function noc(lcse)
            opt = JuMPOptimiser(; slv = cslv, sets = sets, wb = ls,
                                sbgt = BudgetRange(; lb = 0.0, ub = 0.5), lcse = lcse)
            return optimise(NearOptimalCentering(; r = StandardDeviation(), opt = opt,
                                                 alg = ConstrainedNearOptimalCentering()),
                            rd)
        end
        @test gross(noc(nothing).w, tech) > 0.2 + 1e-3
        @test gross(noc(pce(lce("tech <= 0.2"))).w, tech) <= 0.2 + tol
        # The projection of a programme set.
        function resolve(set)
            return PO.resolve_allocation_set(set, N, false, Float64)
        end
        q = [0.9, 0.3, 0.2, 0.1, 0.0, -0.5]
        wh = fill(1 / N, N)
        set(l) = ProgrammeAllocationSet(; slv = cslv, sets = sets, wb = ls, lcse = l)
        p0 = PO.project(EuclideanProjection(), resolve(set(nothing)), q, wh)
        p1 = PO.project(EuclideanProjection(), resolve(set(p)), q, wh)
        @test isapprox(p0, q; atol = 1e-4) && isapprox(gross(p1, tech), 0.4; atol = 1e-4)
        # The set is a block of its own after a long-only leader.
        lead = MeanRisk(; obj = MaximumReturn(),
                        opt = JuMPOptimiser(; slv = cslv, wb = WeightBounds(; ub = 0.6)))
        function ftl(s)
            return optimise(PO.OnlinePortfolioSelection(;
                                                        alg = FollowTheLeader(; opt = lead),
                                                        set = s), rd).w
        end
        @test ftl(set(nothing))[1] > 0.5
        @test gross(ftl(set(p)), tech) <= 0.4 + 1e-4
    end

    @testset "Folds, schedules and the graph in force" begin
        mr(l) = MeanRisk(; obj = MaximumReturn(),
                         opt = JuMPOptimiser(; slv = cslv, sets = sets, wb = ls,
                                             sbgt = BudgetRange(; lb = 0.0, ub = 0.5),
                                             lcse = l))
        cv = IndexWalkForward(40, 20)
        pred = cross_val_predict(mr(pce(node("S <= 0.3"))), rds, cv)
        for (k, f) in enumerate(pred.pred)
            # Fold k trains on the rows up to 20k + 20, and the key of g2 is row 60.
            a = 20k + 20 >= 60 ? [1, 0, 0, 0, 0, 1] : s1
            @test vec(f.res.lcsr.lc.ineq.A) == a
            @test gross(collect(PO.fold_target(f)), a) <= 0.3 + tol
        end
        # A schedule of entries.
        td = TimeDependent([pce(lce("tech <= $(0.1k)")) for k in eachindex(pred.pred)])
        for (k, f) in enumerate(cross_val_predict(mr(td), rd, cv).pred)
            @test gross(collect(PO.fold_target(f)), tech) <= 0.1k + tol
        end
        # The entry check of the graphs reaches the space through the wrapper.
        neg = ReturnsResult(; nx = nx, X = X, ng = ["g"],
                            G = [ExogenousGraph(; edges = [("A", "S", -1.0)])])
        @test_throws ArgumentError optimise(mr(pce(node("S <= 0.3"))), neg)
    end

    @testset "A pipeline step" begin
        p = pce(node("S <= 0.3"))
        @test PO.pipe_reads(p) == (:returns,)
        fs = ExposureConstraintEstimator(; lce = lce("MTUM <= 0.3"), space = FactorSpace())
        lc = LinearConstraint(;
                              ineq = PartialLinearConstraint(; A = [1.0 0 0 0 0 0],
                                                             B = [0.3]))
        @test PO.pipe_reads(pce(fs)) == (:returns, :prior)
        @test PO.pipe_reads(pce([lce("A <= 1"), fs])) == (:returns, :prior)
        @test PO.pipe_reads(pce(lc)) == (:returns,)
        @test PO.pipe_constraint_targets(p) == (:lcse,)
        mr = MeanRisk(; obj = MaximumReturn(),
                      opt = JuMPOptimiser(; slv = cslv, wb = ls, bgt = nothing))
        res = fit(Pipeline(; steps = (p, mr)), rd)
        @test res.ctx.constraints isa PositionConstraint
        @test isapprox(res.w, solve(p).w; rtol = 1e-5)
        # Two steps reach one slot, and a step that wraps a vector adds each constraint.
        two = fit(Pipeline(;
                           steps = (lce("A <= 0.2"),
                                    pce([lce("B <= 0.3"), lce("F <= 0.2")]), mr)), rd)
        @test length(two.ctx.constraints) == 3
        @test two.w[1] <= 0.2 + tol &&
              abs(two.w[2]) <= 0.3 + tol &&
              abs(two.w[6]) <= 0.2 + tol
        # A prior with loadings is the basis of a factor step.
        F = 0.01 .* randn(StableRNG(3), T, 2)
        rdf = ReturnsResult(; nx = nx, X = X, nf = ["MTUM", "VLUE"], F = F)
        rf = fit(Pipeline(; steps = (FactorPrior(), pce(fs), mr)), rdf)
        @test rf.ctx.constraints isa PositionConstraint
        @test rf.ctx.constraints.lc.ineq.A ==
              linear_constraints(fs, sets; rr = rf.ctx.prior.rr).ineq.A
    end

    @testset "Meta-optimisers" begin
        plain = MeanRisk(; opt = JuMPOptimiser(; slv = cslv))
        mr(l) = MeanRisk(;
                         opt = JuMPOptimiser(; slv = cslv, sets = sets, wb = ls, lcse = l))
        cle = ClustersEstimator()
        nco(; opti = plain, opto = plain) = NestedClustered(; cle = cle, opti = opti,
                                                            opto = opto, sets = sets)
        # The outer optimiser refuses a space over nodes, and a stated basis.
        stated = ExposureConstraintEstimator(; lce = lce("MTUM <= 0.3"),
                                             space = FactorSpace(;
                                                                 re = Regression(;
                                                                                 M = zeros(N,
                                                                                           2))))
        for l in (pce(node("S <= 1")), [pce(node("S <= 1"))],
                  pce([lce("A <= 1"), node("S <= 1")]), TimeDependent([pce(node("S <= 1"))]))
            @test_throws ArgumentError nco(; opto = mr(l))
            @test_throws ArgumentError Stacking(; opti = [plain], opto = mr(l))
        end
        @test PO.stated_constraint_space_basis(pce([lce("A <= 1"), stated]))
        @test_throws ArgumentError nco(; opto = mr(pce(stated)))
        # The inner optimiser refuses a precomputed constraint in each shape.
        lc = LinearConstraint(; ineq = PartialLinearConstraint(; A = ones(1, N), B = [0.3]))
        pc = PositionConstraint(; lc = lc)
        for l in (pc, pce(lc), [lce("A <= 1"), pc], pce([lce("A <= 1"), lc]),
                  TimeDependent([pce(lc)]), TimeDependent([lce("A <= 1")]; default = pc))
            @test_throws ArgumentError nco(; opti = mr(l))
        end
        @test !PO.holds_precomputed(pce(node("S <= 1")), LinearConstraint)
        # A wrapper of names is an inner constraint, and each cluster reads its assets.
        res = with_logger(NullLogger()) do
            return optimise(nco(; opti = mr([pce(node("S <= 1.1")), pce(lce("A <= 0.5"))])),
                            rd)
        end
        @test isapprox(sum(res.w), 1; atol = 1e-6)
        for (k, r) in enumerate(res.resi)
            cl = findall(==(k), PO.assignments(res.clr))
            a = 1 in cl ? findfirst(==(1), cl) : nothing
            @test isnothing(a) || abs(r.w[a]) <= 0.5 + tol
        end
    end

    @testset "A concentration limit reads the short positions" begin
        cl = ConcentrationLimit(; t = 0.1, c = 0.2, s = 0.3, pos = ShortPosition(),
                                alg = SumLargestConcentration())
        kw = (; bgt = 1, sbgt = BudgetRange(; lb = 0.0, ub = 0.6))
        @test maximum(x -> max(-x, 0), solve(nothing; kw...).w) > 0.3
        res = solve(nothing; conc = cl, kw...)
        @test ok(res) && maximum(x -> max(-x, 0), res.w) <= 0.2 + tol
        cm = concentration_margins(cl, res.w; tol = 1e-5)
        @test cm.x == max.(-res.w, 0) && all(cm.cap .<= tol) && cm.total <= tol
        # On a long-only book the rule sees no position.
        wb = WeightBounds(; lb = 0.0, ub = 1.0)
        lo = solve(nothing; wb = wb, bgt = 1, conc = cl)
        @test isapprox(lo.w, solve(nothing; wb = wb, bgt = 1).w; atol = 1e-5)
        @test maximum(lo.w) > 0.2 + tol
        gr = ConcentrationLimit(; t = 0.1, c = 0.4, s = 1.0,
                                alg = SumLargestConcentration())
        @test maximum(solve(nothing; wb = wb, bgt = 1, conc = gr).w) <= 0.4 + tol
        exact = ConcentrationLimit(; t = 0.1, c = 0.2, s = 0.3, pos = ShortPosition())
        @test ok(solve(nothing; wb = wb, bgt = 1, conc = exact, slv = hslv))
    end
end
