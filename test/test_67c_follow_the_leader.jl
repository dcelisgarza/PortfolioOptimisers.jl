#=
The third algorithm set of the online portfolio selection family (issue #1178 on map #1148):
`FollowTheLeader` over a Sample Selector, the seven selectors, the Allocation Set Constraint
through which a re-solve honours the head's set (ADR 0164), `FollowTheLeadingHistory`,
`RankOneCovariance` with the `ShortTermLossControlPortfolio` constructor, and the
pattern-matching and risk-aversion aggregations as configurations.

Parity: follow the leader equals a walk-forward at `test_size = 1` over `MeanRisk` under
`LogarithmicReturn` with an expanding window, fold for fold; the selectors are checked
against their papers' definitions on hand fixtures; the four tests ADR 0164 owes are here.
=#
using Test, PortfolioOptimisers, StableRNGs, LinearAlgebra, Statistics, Dates, Clarabel,
      HiGHS, Pajarito, JuMP
@testset "Online portfolio selection: the third set" begin
    po = PortfolioOptimisers
    OPS = po.OnlinePortfolioSelection
    rows(r, i) = po.port_opt_view(r, i, :)
    resolve(set, N) = po.resolve_allocation_set(set, N, false, Float64)

    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false, "tol_gap_abs" => 1e-12,
                                 "tol_gap_rel" => 1e-12, "tol_feas" => 1e-12,
                                 "max_iter" => 500),
                 check_sol = (; allow_local = true, allow_almost = true))
    mip = Solver(; name = :mip,
                 solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                    "oa_solver" =>
                                                        optimizer_with_attributes(HiGHS.Optimizer,
                                                                                  MOI.Silent() =>
                                                                                      true),
                                                    "conic_solver" =>
                                                        optimizer_with_attributes(Clarabel.Optimizer,
                                                                                  "verbose" =>
                                                                                      false)),
                 check_sol = (; allow_local = true, allow_almost = true))
    logopt(; kw...) = MeanRisk(; obj = MaximumReturn(),
                               opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv,
                                                   ret = LogarithmicReturn(), kw...))
    log_wealth(w, X) = sum(log.(X * w))

    rng = StableRNG(11)
    T, N = 40, 4
    R = 0.02 .* randn(rng, T, N)
    X = 1 .+ R
    nx = ["A", "B", "C", "D"]
    ts = Date(2020, 1, 1) .+ Day.(0:(T - 1))
    rd = ReturnsResult(; nx = nx, X = R, ts = ts)

    @testset "Sample Selectors on a hand fixture" begin
        @test_throws DomainError LastRows(; W = 0)
        @test_throws DomainError HistogramMatch(; window = 0)
        @test_throws ArgumentError HistogramMatch(; edges = [1, 1])
        @test_throws ArgumentError HistogramMatch(; edges = [2, 1])
        @test_throws Exception HistogramMatch(; edges = Float64[])
        @test_throws DomainError KernelMatch(; radius = 0)
        @test_throws DomainError KernelMatch(; radius = Inf)
        @test_throws DomainError NearestNeighbourMatch(; neighbours = 0)
        @test_throws DomainError NearestNeighbourMatch(; neighbours = 1.0)
        @test_throws DomainError CorrelationMatch(; rho = 1.5)
        @test_throws DomainError ClusterMatch(; window = 0)
        @test isnothing(po.rows_needed(Prefix()))
        @test po.rows_needed(LastRows(; W = 7)) == 7
        @test isnothing(po.rows_needed(CorrelationMatch()))
        # Six rows of two assets: the two-row windows before rows 3, 4, 5 and 6 are
        # "up, up", "up, down", "down, up" and "up, up" in asset 1, and the latest window
        # (rows 5 and 6) is "up, up" as well.
        H = [1.1 0.9; 1.2 1.0; 1.1 1.1; 0.8 1.1; 1.3 0.9; 1.2 1.1]
        @test po.select_rows(Prefix(), H) == 1:6
        @test po.select_rows(LastRows(; W = 4), H) == 3:6
        @test po.select_rows(LastRows(; W = 10), H) == 1:6
        # No candidate until window + 1 rows exist.
        @test isempty(po.select_rows(HistogramMatch(; window = 2), H[1:2, :]))
        @test isempty(po.select_rows(CorrelationMatch(; window = 2), H[1:2, :]))
        # The window compared for row i is the two rows before it, and the latest window
        # is the last two rows; the two never coincide. The histogram cell at edges = [1]
        # is the direction of every entry: the window before row 3 (rows 1, 2) is up-up in
        # asset 1 and down-flat in asset 2, as the latest (rows 5, 6) is; the window before
        # row 6 (rows 4, 5) is down-up and up-down. Row 3 alone shares the cell.
        hm = HistogramMatch(; window = 2)
        cell(v) = po.histogram_cell([1], v)
        @test cell(vec(H[1:2, :])) == cell(vec(H[5:6, :]))
        @test cell(vec(H[4:5, :])) != cell(vec(H[5:6, :]))
        @test po.select_rows(hm, H) == [3]
        # A finer partition separates rows 1-2 from 5-6 at 1.15.
        @test isempty(po.select_rows(HistogramMatch(; window = 2, edges = [1, 1.15]), H))
        # A history whose first window recurs at the end matches itself: rows 7 and 8
        # repeat rows 1 and 2, so the window before row 3 is the latest window.
        H8 = vcat(H, H[1:2, :])
        @test po.select_rows(hm, H8) == [3, 7]
        @test po.select_rows(HistogramMatch(; window = 2, edges = [1, 1.15]), H8) == [3]
        # The kernel: distances of the candidate windows from the latest one.
        latest = vec(H[5:6, :])
        d = [norm(vec(H[(i - 2):(i - 1), :]) - latest) for i in 3:6]
        @test po.select_rows(KernelMatch(; window = 2, radius = 0.3), H) == (3:6)[d .<= 0.3]
        @test isempty(po.select_rows(KernelMatch(; window = 2, radius = 1e-9), H))
        @test po.select_rows(KernelMatch(; window = 2, radius = 1e-9), H8) == [3]
        # Nearest neighbours by count and by fraction, the fraction floored.
        order = sortperm(d)
        @test po.select_rows(NearestNeighbourMatch(; window = 2, neighbours = 2), H) ==
              sort((3:6)[order[1:2]])
        @test po.select_rows(NearestNeighbourMatch(; window = 2, neighbours = 10), H) == 3:6
        @test po.select_rows(NearestNeighbourMatch(; window = 2, neighbours = 0.5), H) ==
              sort((3:6)[order[1:2]])
        @test isempty(po.select_rows(NearestNeighbourMatch(; window = 2, neighbours = 0.2),
                                     H))
        # Correlation of the concatenated windows; the recurring window correlates with
        # the latest at one, so a threshold near one keeps row 3 of H8 alone.
        c = [cor(vec(H[(i - 2):(i - 1), :]), latest) for i in 3:6]
        @test cor(vec(H8[1:2, :]), vec(H8[7:8, :])) ≈ 1
        @test po.select_rows(CorrelationMatch(; window = 2, rho = 0.999), H8) == [3]
        @test isempty(po.select_rows(CorrelationMatch(; window = 2, rho = 0.999), H))
        @test po.select_rows(CorrelationMatch(; window = 2, rho = 0.1), H) ==
              (3:6)[c .>= 0.1]
        @test po.select_rows(CorrelationMatch(; window = 2, rho = -1), H) == 3:6
        # A constant window has no correlation and never matches.
        Hc = copy(H)
        Hc[1:2, :] .= 1
        @test !(3 in po.select_rows(CorrelationMatch(; window = 2, rho = -1), Hc))
        # Clusters: fewer than two candidates is the empty selection; on a fixture of
        # two clear regimes the latest window's cluster is its own regime.
        @test isempty(po.select_rows(ClusterMatch(; window = 2), H[1:3, :]))
        up = [1.1 1.1; 1.2 1.1]
        dn = [0.9 0.9; 0.8 0.9]
        G = vcat(up, dn, up, dn, up, dn, up)
        cm = ClusterMatch(; window = 2,
                          clusterer = ClustersEstimator(; alg = KMeansAlgorithm(; seed = 1),
                                                        onc = OptimalNumberClusters(;
                                                                                    max_k = 2)))
        sel = po.select_rows(cm, G)
        @test !isempty(sel)
        @test 3 in sel || 7 in sel
    end

    @testset "Follow the leader over the solver-free optimiser" begin
        ftl = OPS(; alg = FollowTheLeader())
        @test FollowTheLeader().gamma == 0
        @test_throws DomainError FollowTheLeader(; gamma = 1.5)
        @test isnothing(po.rows_needed(FollowTheLeader()))
        @test po.rows_needed(FollowTheLeader(; sel = LastRows(; W = 5))) == 5
        # The Causal Pass after rows 1:t is the best constant rebalanced portfolio of
        # those rows, at every t, and the fold on one row is the leader of one row.
        for t in (1, 2, 5, 20, T)
            @test isapprox(optimise(ftl, rows(rd, 1:t)).w,
                           optimise(BestConstantRebalancedPortfolio(), rows(rd, 1:t)).w;
                           atol = 1e-14)
        end
        # The variable rebalanced portfolio: the last W rows, the window filling first.
        vrp = OPS(; alg = FollowTheLeader(; sel = LastRows(; W = 6)))
        @test isapprox(optimise(vrp, rows(rd, 1:3)).w,
                       optimise(BestConstantRebalancedPortfolio(), rows(rd, 1:3)).w;
                       atol = 1e-14)
        @test isapprox(optimise(vrp, rows(rd, 1:20)).w,
                       optimise(BestConstantRebalancedPortfolio(), rows(rd, 15:20)).w;
                       atol = 1e-14)
        # The batch–online identity, at test_size 1 and in a block.
        o = po.partial_fit!(vrp, rows(rd, 1:8))
        o = po.partial_fit!(o, rows(rd, 9:13))
        @test optimise(o).w == optimise(vrp, rows(rd, 1:13)).w
        @test po.observation_count(o) == 13
        # An empty selection answers the uniform portfolio: no pattern matches yet.
        pm = OPS(; alg = FollowTheLeader(; sel = CorrelationMatch(; window = 3)))
        @test optimise(pm, rows(rd, 1:3)).w == fill(0.25, 4)
        # ... projected onto the set in the rule's geometry under a cap.
        pmc = OPS(; alg = FollowTheLeader(; sel = CorrelationMatch(; window = 3)),
                  set = BoundedAllocationSet(; wb = WeightBounds(0, 0.2)))
        @test_throws ArgumentError optimise(pmc, rows(rd, 1:3))
        pmc = OPS(; alg = FollowTheLeader(; sel = CorrelationMatch(; window = 3)),
                  set = BoundedAllocationSet(; wb = WeightBounds(0.1, 0.4)))
        @test optimise(pmc, rows(rd, 1:3)).w == fill(0.25, 4)
        # The damping: the mix of the leader and the held allocation, unprojected on the
        # bounded set.
        wscrp = OPS(; alg = FollowTheLeader(; gamma = 0.3))
        w1 = optimise(ftl, rows(rd, 1:1)).w
        w2 = optimise(wscrp, rows(rd, 1:1)).w
        @test w2 ≈ 0.7 .* w1 .+ 0.3 .* fill(0.25, 4)
        # A bounded set goes into the fixed point's wb: the repaired fixed point.
        cap = OPS(; alg = FollowTheLeader(),
                  set = BoundedAllocationSet(; wb = WeightBounds(0, 0.5)))
        @test isapprox(optimise(cap, rd).w,
                       optimise(BestConstantRebalancedPortfolio(;
                                                                wb = WeightBounds(0, 0.5)),
                                rd).w; atol = 1e-14)
        @test maximum(optimise(cap, rd).w) <= 0.5 + 1e-12
        # A programme set is refused by name on the solver-free leader, and through a
        # mixture and a leading history over it.
        pset = ProgrammeAllocationSet(; slv = slv)
        @test_throws ArgumentError OPS(; alg = FollowTheLeader(), set = pset)
        @test_throws ArgumentError OPS(;
                                       alg = ExpertMixture(; experts = [FollowTheLeader()]),
                                       set = pset)
        @test_throws ArgumentError OPS(;
                                       alg = FollowTheLeadingHistory(;
                                                                     alg = FollowTheLeader()),
                                       set = pset)
        # Views slice the held optimiser.
        v = po.port_opt_view(OPS(; alg = FollowTheLeader(; opt = logopt())), [1, 2])
        @test v.alg.sel === Prefix()
        # The buffer holds returns; the selector reads relatives.
        st = po.partial_fit!(ftl, rows(rd, 1:5)).cache
        @test po.sample_buffer(st.X) == R[1:5, :]
    end

    @testset "Follow the leader over a JuMP head equals the walk-forward" begin
        ftl = OPS(; alg = FollowTheLeader(; opt = logopt()))
        k = 5
        cv = IndexWalkForward(k, 1; expand_train = true)
        pred = cross_val_predict(logopt(), rd, cv)
        for i in 1:4
            @test isapprox(pred.pred[i].res.w, optimise(ftl, rows(rd, 1:(k + i - 1))).w;
                           atol = 1e-6)
        end
        # ... and the head under the online step walks the same folds.
        cvo = IndexWalkForward(k, 1; expand_train = true, ff = OnlineStep())
        predo = cross_val_predict(ftl, rd, cvo)
        for i in 1:4
            @test isapprox(predo.pred[i].res.w, pred.pred[i].res.w; atol = 1e-6)
        end
        # The solver-free and the solved leader agree to the solver's tolerance on an
        # interior fixture: two assets alternating 1.4 and 0.75 in anti-phase, with noise.
        rngi = StableRNG(2150)
        Ti = 30
        Xi = hcat(ifelse.(isodd.(1:Ti), 1.4, 0.75), ifelse.(isodd.(1:Ti), 0.75, 1.4),
                  ones(Ti), ones(Ti)) .* (1 .+ 0.01 .* randn(rngi, Ti, 4))
        rdi = ReturnsResult(; nx = nx, X = Xi .- 1)
        wi = optimise(ftl, rdi).w
        @test all(wi[1:2] .> 0.2)
        @test isapprox(wi, optimise(OPS(; alg = FollowTheLeader()), rdi).w; atol = 1e-6)
        # A head that fits a covariance needs two rows: one row is the uniform portfolio.
        @test po.leader_min_rows(logopt()) == 2
        @test po.leader_min_rows(BestConstantRebalancedPortfolio()) == 1
        @test optimise(ftl, rows(rd, 1:1)).w == fill(0.25, 4)
        @test !(optimise(ftl, rows(rd, 1:2)).w == fill(0.25, 4))
        # The names the head pins reach the carrier: a set on the held optimiser by name.
        named = OPS(;
                    alg = FollowTheLeader(;
                                          opt = logopt(;
                                                       wb = WeightBoundsEstimator(;
                                                                                  ub = Dict("A" =>
                                                                                                0.1)),
                                                       sets = UniverseSets(;
                                                                           dict = Dict("nx" =>
                                                                                           nx)))))
        @test optimise(named, rd).w[1] <= 0.1 + 1e-8
        # Outside a step the carrier is named by column.
        @test po.leader_carrier(R[1:3, :]).nx == ["1", "2", "3", "4"]
    end

    @testset "The Allocation Set Constraint (ADR 0164)" begin
        # Sixty rows of three assets on a two-period cycle: the first two co-moving, the
        # second with the lower drift, the third in anti-phase. The unconstrained leader is
        # all-in on the first, because every other asset's mean ratio to it is below one.
        T2 = 60
        odd = isodd.(1:T2)
        R3 = hcat(ifelse.(odd, 0.05, -0.02), ifelse.(odd, 0.044, -0.026),
                  ifelse.(odd, -0.01, 0.03))
        X3 = 1 .+ R3
        rd3 = ReturnsResult(; nx = ["A", "B", "C"], X = R3)
        @test all(mean(X3[:, j] ./ X3[:, 1]) < 1 for j in 2:3)
        free = optimise(OPS(; alg = FollowTheLeader(; opt = logopt())), rd3).w
        @test free[1] > 0.99
        # 1. The constrained leader on the capped set equals MeanRisk with the bound on
        #    opt, and the set enters every fold's programme.
        cset = ProgrammeAllocationSet(; slv = slv, wb = WeightBounds(0, 0.5))
        capped = OPS(; alg = FollowTheLeader(; opt = logopt()), set = cset)
        wc = optimise(capped, rd3).w
        wo = optimise(logopt(; wb = WeightBounds(0, 0.5)), rd3).w
        @test isapprox(wc, wo; atol = 1e-6)
        @test maximum(wc) <= 0.5 + 1e-8
        @test wc[1] ≈ 0.5 atol = 1e-6
        # 2. The projected leader is not it: the projection spreads the excess evenly,
        #    the constrained solve moves it to the asset in anti-phase.
        wp = po.project(EuclideanProjection(),
                        resolve(BoundedAllocationSet(; wb = WeightBounds(0, 0.5)), 3), free,
                        free)
        @test isapprox(wp, [0.5, 0.25, 0.25]; atol = 1e-6)
        @test !isapprox(wc, wp; atol = 1e-2)
        @test wc[3] > wp[3] + 0.1
        @test log_wealth(wc, X3) > log_wealth(wp, X3)
        # Both homes hold: a bound on opt and a bound on the set intersect.
        both = OPS(; alg = FollowTheLeader(; opt = logopt(; wb = WeightBounds(0, 0.6))),
                   set = cset)
        @test isapprox(optimise(both, rd3).w, wc; atol = 1e-6)
        both2 = OPS(; alg = FollowTheLeader(; opt = logopt(; wb = WeightBounds(0, 0.4))),
                    set = cset)
        @test maximum(optimise(both2, rd3).w) <= 0.4 + 1e-8
        # 3. The mix repair on the turnover example: w_t = [0.5, 0.5, 0],
        #    x_t = [1.5, 0.5, 1], so the book is [0.75, 0.25, 0]; tn = 0.2; γ = 0.5. On
        #    rows whose leader is the second asset, the constrained leader sits at the
        #    ceiling, [0.55, 0.45, 0], the mix [0.525, 0.475, 0] does not, and the second
        #    projection repairs it.
        tset = resolve(ProgrammeAllocationSet(; slv = slv, tn = 0.2), 3)
        wt = [0.5, 0.5, 0.0]
        xt = [1.5, 0.5, 1.0]
        book = wt .* xt ./ dot(wt, xt)
        @test book ≈ [0.75, 0.25, 0]
        rowsT = R3[:, [3, 1, 2]]
        alg0 = FollowTheLeader(; opt = logopt())
        (_, w0), held0 = po.with_projection_step(() -> po.online_update!(alg0, nothing, wt,
                                                                         xt, rowsT, tset),
                                                 rowsT, 1)
        @test isempty(held0)
        @test isapprox(w0, [0.55, 0.45, 0]; atol = 1e-6)
        alg5 = FollowTheLeader(; opt = logopt(), gamma = 0.5)
        (_, w5), held5 = po.with_projection_step(() -> po.online_update!(alg5, nothing, wt,
                                                                         xt, rowsT, tset),
                                                 rowsT, 1)
        @test isempty(held5)
        mix = 0.5 .* w0 .+ 0.5 .* wt
        @test maximum(abs.(mix .- book)) > 0.2
        @test all(abs.(w5 .- book) .<= 0.2 + 1e-6)
        @test !isapprox(w5, mix; atol = 1e-3)
        @test sum(w5) ≈ 1
        # On the bounded set the mix is not projected, and at γ = 0 nothing is.
        bset = resolve(BoundedAllocationSet(), 3)
        (_, wb5), _ = po.with_projection_step(() -> po.online_update!(alg5, nothing, wt, xt,
                                                                      rowsT, bset), rowsT,
                                              1)
        (_, wb0), _ = po.with_projection_step(() -> po.online_update!(alg0, nothing, wt, xt,
                                                                      rowsT, bset), rowsT,
                                              1)
        @test isapprox(wb5, 0.5 .* wb0 .+ 0.5 .* wt; atol = 1e-10)
        # 4. Under card = 2 the leader holds two names and the mix of it with a
        #    two-name book is projected back to two. A MIP kind on the set needs a
        #    MIP-capable solver on the held optimiser, whose model it enters.
        mset = resolve(ProgrammeAllocationSet(; slv = mip, card = 2), 3)
        algm = FollowTheLeader(; opt = logopt(; slv = mip))
        (_, wm0), heldm = po.with_projection_step(() -> po.online_update!(algm, nothing, wt,
                                                                          xt, rowsT, mset),
                                                  rowsT, 1)
        @test isempty(heldm)
        @test count(x -> x > 1e-6, wm0) <= 2
        wq = [0.0, 0.5, 0.5]
        algq = FollowTheLeader(; opt = logopt(; slv = mip), gamma = 0.5)
        (_, wm5), _ = po.with_projection_step(() -> po.online_update!(algq, nothing, wq, xt,
                                                                      rowsT, mset), rowsT,
                                              1)
        @test count(x -> x > 1e-6, wm5) <= 2
        @test sum(wm5) ≈ 1 atol = 1e-6
        # ... and on the conic solver alone the MIP kind is a Held Step.
        (_, wmh), heldh = po.with_projection_step(() -> po.online_update!(alg0, nothing, wt,
                                                                          xt, rowsT, mset),
                                                  rowsT, 1)
        @test length(heldh) == 1 && wmh == book
        # The set's other kinds reach the leader's model through the adapter: a linear
        # constraint, a variance ceiling read on the head's rows, a tracking error in
        # every norm, and a turnover beside the head's own turnover, at a free index.
        lset = ProgrammeAllocationSet(; slv = slv,
                                      lcs = LinearConstraintEstimator(; val = :(A <= 0.3)),
                                      sets = UniverseSets(;
                                                          dict = Dict("nx" =>
                                                                          ["A", "B", "C"])))
        @test optimise(OPS(; alg = FollowTheLeader(; opt = logopt()), set = lset), rd3).w[1] <=
              0.3 + 1e-8
        S3 = cov(R3)
        vub = 0.5 * dot(free, S3, free)
        vset = ProgrammeAllocationSet(; slv = slv,
                                      r = Variance(;
                                                   settings = RiskMeasureSettings(;
                                                                                  ub = vub)))
        wv = optimise(OPS(; alg = FollowTheLeader(; opt = logopt()), set = vset), rd3).w
        @test dot(wv, S3, wv) <= vub * (1 + 1e-4)
        @test !isapprox(wv, free; atol = 1e-3)
        # An infeasible ceiling is a Held Step, not a weaker set: four noisy assets have
        # no zero-variance portfolio.
        vres = optimise(OPS(; alg = FollowTheLeader(; opt = logopt()),
                            set = ProgrammeAllocationSet(; slv = slv,
                                                         r = Variance(;
                                                                      settings = RiskMeasureSettings(;
                                                                                                     ub = 1e-12)))),
                        rows(rd, 1:20))
        @test isa(vres.retcode.res, po.HeldStep)
        # The tracking error in every norm: the deviation from the benchmark series over
        # the head's rows meets the norm's own ceiling, scaled as the head's builders
        # scale it, and binds.
        b3 = fill(1 / 3, 3)
        te_bound(::L1Norm, err, T) = err * T
        te_bound(alg::L2Norm, err, T) = err * sqrt(T - alg.ddof)
        te_bound(alg::SquaredL2Norm, err, T) = sqrt(err * (T - alg.ddof))
        te_bound(alg::LpNorm, err, T) = err * (T - alg.ddof)^(1 / alg.p)
        te_bound(alg::LInfNorm, err, T) = err * (T - alg.ddof)
        te_norm(::L1Norm, d) = sum(abs, d)
        te_norm(::Union{<:L2Norm, <:SquaredL2Norm}, d) = norm(d)
        te_norm(alg::LpNorm, d) = norm(d, alg.p)
        te_norm(::LInfNorm, d) = maximum(abs, d)
        for alg in (L1Norm(), L2Norm(), SquaredL2Norm(), LpNorm(; p = 3), LInfNorm())
            eset = ProgrammeAllocationSet(; slv = slv,
                                          te = TrackingError(; err = 1e-6, alg = alg,
                                                             tr = WeightsTracking(; w = b3)))
            res_e = optimise(OPS(; alg = FollowTheLeader(; opt = logopt()), set = eset),
                             rd3)
            @test !isa(res_e.retcode.res, po.HeldStep)
            d = R3 * (res_e.w .- b3)
            @test te_norm(alg, d) <= te_bound(alg, 1e-6, T2) * (1 + 1e-4)
            @test te_norm(alg, d) >= te_bound(alg, 1e-6, T2) * (1 - 1e-2)
            @test !isapprox(res_e.w, free; atol = 1e-2)
        end
        # A turnover on the held optimiser reads the same book the set's does, through
        # `factory`: both ceilings hold from the drifted previous target.
        tnopt = logopt(; tn = Turnover(; val = 0.05, w = b3))
        htn = OPS(; alg = FollowTheLeader(; opt = tnopt),
                  set = ProgrammeAllocationSet(; slv = slv, tn = 0.3))
        rtn = optimise(htn, rd3)
        @test !isa(rtn.retcode.res, po.HeldStep)
        prev = optimise(htn, rows(rd3, 1:(T2 - 1))).w
        bookT = prev .* X3[T2, :] ./ dot(prev, X3[T2, :])
        @test all(abs.(rtn.w .- bookT) .<= 0.05 + 1e-6)
        @test !isapprox(rtn.w, wc; atol = 1e-2)
        # A failed re-solve is a Held Step: the update answers the book, the head warns,
        # and the read-out carries the record as a success.
        stuck = MeanRisk(; obj = MaximumReturn(),
                         opt = JuMPOptimiser(; pe = EmpiricalPrior(),
                                             slv = Solver(; name = :one,
                                                          solver = Clarabel.Optimizer,
                                                          settings = Dict("verbose" =>
                                                                              false,
                                                                          "max_iter" => 1)),
                                             ret = LogarithmicReturn()))
        hopt = OPS(; alg = FollowTheLeader(; opt = stuck), fb = EqualWeighted())
        resh = @test_logs (:warn, r"Held Step at 2020-01-03") match_mode=:any optimise(hopt,
                                                                                       rows(rd,
                                                                                            1:3))
        @test isa(resh.retcode, OptimisationSuccess)
        @test isa(resh.retcode.res, po.HeldStep)
        @test occursin("follow-the-leader programme", resh.retcode.res.reason)
        @test haskey(resh.retcode.res.trials, :one)
        # Row 1 is the uniform portfolio (one row, no covariance); rows 2 and 3 are held,
        # so the answer is that portfolio drifted through both.
        wexp = fill(0.25, 4)
        for t in 2:3
            wexp = wexp .* X[t, :] ./ dot(wexp, X[t, :])
        end
        @test isapprox(resh.w, wexp; atol = 1e-12)
    end

    @testset "RankOneCovariance and the PSD factor" begin
        W5 = X[1:5, :]
        S = cov(RankOneCovariance(), R[1:5, :])
        @test size(S) == (4, 4) && issymmetric(S)
        @test rank(S) == 1
        @test minimum(eigvals(Symmetric(S))) >= -1e-12
        # The paper's algorithm 1 by hand: θ₁ = Ξ₁₁², u₁ the principal right singular
        # vector of the uncentred window, tr(D) the centred energy.
        F = svd(W5)
        theta = F.S[1]^2
        u = F.V[:, 1]
        Xc = W5 .- mean(W5; dims = 1)
        trD = sum(abs2, Xc)
        @test trD ≈ tr(Diagonal(F.S .^ 2) -
                       Diagonal(F.S) * F.U' * ones(5, 5) * F.U * Diagonal(F.S) / 5)
        zeta = theta * (trD / (4 * 4))^(-1 / 2)
        @test isapprox(S, zeta .* u * u'; atol = 1e-10)
        # The shift: zero decomposes the returns themselves, a different direction.
        S0 = cov(RankOneCovariance(; shift = 0), R[1:5, :])
        @test !isapprox(S0, S; rtol = 1e-3)
        @test_throws ArgumentError cov(RankOneCovariance(), R[1:1, :])
        @test_throws DomainError RankOneCovariance(; shift = Inf)
        C = cor(RankOneCovariance(), R[1:5, :])
        @test all(diag(C) .== 1)
        @test all(abs.(C) .<= 1)
        @test C[1, 2] == sign(S[1, 2])
        @test cov(RankOneCovariance(), permutedims(R[1:5, :]); dims = 2) ≈ S
        # The factor: bitwise the Cholesky where it exists, a PSD factor where it does not.
        A = cov(R)
        @test po.covariance_factor(A) == cholesky(A).U
        G = po.covariance_factor(S)
        @test isapprox(G' * G, S; atol = 1e-12)
        @test_throws LinearAlgebra.PosDefException po.covariance_factor([1.0 0.5; 0.4 1.0])
        @test_throws LinearAlgebra.PosDefException po.covariance_factor(-A)
    end

    @testset "ShortTermLossControlPortfolio" begin
        alg = ShortTermLossControlPortfolio(; slv = slv)
        @test alg.sel == LastRows(; W = 5)
        @test alg.gamma == 0
        @test isa(alg.opt.r[1], WorstRealisation)
        @test isa(alg.opt.r[2], Variance) && alg.opt.r[2].settings.scale == 0.025
        @test isa(alg.opt.opt.pe.ce, RankOneCovariance)
        @test_throws DomainError ShortTermLossControlPortfolio(; slv = slv, gamma = -1)
        @test_throws DomainError ShortTermLossControlPortfolio(; slv = slv, window = 0)
        # γ = 0 is the pure max-min: on two assets a line search over the simplex.
        W6 = R[10:15, 1:2]
        X6 = 1 .+ W6
        rd6 = ReturnsResult(; nx = ["A", "B"], X = W6)
        w0 = optimise(OPS(;
                          alg = ShortTermLossControlPortfolio(; window = 6, gamma = 0,
                                                              slv = slv)), rd6).w
        grid = 0:0.0005:1
        best = argmax([minimum(X6 * [b, 1 - b]) for b in grid])
        @test isapprox(w0[1], grid[best]; atol = 1e-3)
        # γ = 0.025 on a hand window: the paper's quadratic programme in z = [b; q].
        alg25 = ShortTermLossControlPortfolio(; window = 6, gamma = 0.025, slv = slv)
        w25 = optimise(OPS(; alg = alg25), rd6).w
        Sro = cov(RankOneCovariance(), W6)
        m = JuMP.Model(Clarabel.Optimizer)
        JuMP.set_silent(m)
        JuMP.@variable(m, 0 <= b[1:2] <= 1)
        JuMP.@variable(m, q)
        JuMP.@constraint(m, sum(b) == 1)
        JuMP.@constraint(m, X6 * b .>= q)
        JuMP.@objective(m, Max, q - 0.025 * dot(b, Sro, b))
        JuMP.optimize!(m)
        @test isapprox(w25, JuMP.value.(b); atol = 1e-5)
        @test !isapprox(w25, w0; atol = 1e-4)
        # The window fills before the rule reads five rows: one row is uniform, two rows
        # is the programme on two.
        @test optimise(OPS(; alg = alg25), rows(rd, 1:1)).w == fill(0.25, 4)
        @test sum(optimise(OPS(; alg = alg25), rows(rd, 1:2)).w) ≈ 1 atol = 1e-6
        # Under a cap the head's set enters the programme.
        wcap = optimise(OPS(; alg = alg25,
                            set = ProgrammeAllocationSet(; slv = slv,
                                                         wb = WeightBounds(0, 0.5))), rd6).w
        @test maximum(wcap) <= 0.5 + 1e-6
    end

    @testset "FollowTheLeadingHistory" begin
        @test_throws DomainError FollowTheLeadingHistory(; alpha = 0)
        @test_throws ArgumentError FollowTheLeadingHistory(;
                                                           eset = BoundedAllocationSet(;
                                                                                       wb = WeightBounds([0.0,
                                                                                                          0.0],
                                                                                                         [1.0,
                                                                                                          1.0])))
        # The lifetime rule: i = r 2^k lives through i + 2^(k + 2) + 1.
        @test all(po.expert_alive(1, t) for t in 1:6) && !po.expert_alive(1, 7)
        @test po.expert_alive(4, 21) && !po.expert_alive(4, 22)
        @test po.expert_alive(6, 15) && !po.expert_alive(6, 16)
        @test po.expert_alive(8, 41) && !po.expert_alive(8, 42)
        # One row by hand under an exponentiated-gradient base: the first expert steps
        # from the Start Allocation, the newcomer starts uniform, and the weights are
        # one half each.
        eg = ExponentiatedGradient(; eta = 0.5)
        flh = FollowTheLeadingHistory(; alg = eg, prune = false)
        u = fill(0.25, 4)
        x1 = X[1, :]
        (_, h1), _ = po.with_projection_step(() -> po.online_update!(eg,
                                                                     po.rule_state_seed(eg,
                                                                                        u),
                                                                     u, x1, nothing,
                                                                     resolve(BoundedAllocationSet(),
                                                                             4)), nothing,
                                             1)
        r1 = optimise(OPS(; alg = flh), rows(rd, 1:1))
        @test isapprox(r1.w, 0.5 .* h1 .+ 0.5 .* u; atol = 1e-14)
        st1 = po.partial_fit!(OPS(; alg = flh), rows(rd, 1:1)).cache.st
        @test st1.born == [1, 2] && st1.p == [0.5, 0.5]
        # Without pruning every expert lives; with it the working set is logarithmic.
        stf = po.partial_fit!(OPS(; alg = flh), rd).cache.st
        @test length(stf.h) == T + 1 && stf.born == 1:(T + 1)
        stp = po.partial_fit!(OPS(; alg = FollowTheLeadingHistory(; alg = eg)), rd).cache.st
        @test length(stp.h) < T + 1
        @test all(b -> po.expert_alive(b, T + 1), stp.born)
        @test sum(stp.p) ≈ 1
        # The newcomer's share is 1 / (t + 1) before the pruning renormalises.
        @test stf.p[end] ≈ 1 / (T + 1)
        # An Expert Set caps the trust in one copy.
        cape = FollowTheLeadingHistory(; alg = eg, prune = false,
                                       eset = BoundedAllocationSet(;
                                                                   wb = WeightBounds(0.1,
                                                                                     0.6)))
        stc = po.partial_fit!(OPS(; alg = cape), rows(rd, 1:3)).cache.st
        @test length(stc.p) == 4
        @test all(0.1 - 1e-12 .<= stc.p .<= 0.6 + 1e-12) && sum(stc.p) ≈ 1
        @test stc.p[end] > 1 / 4 - 1e-12
        # A bound no probability vector over the two experts of the first row meets is
        # refused there.
        cape4 = FollowTheLeadingHistory(; alg = eg, prune = false,
                                        eset = BoundedAllocationSet(;
                                                                    wb = WeightBounds(0,
                                                                                      0.4)))
        @test_throws ArgumentError po.partial_fit!(OPS(; alg = cape4), rows(rd, 1:1))
        # The blend meets the head's set once more under a programme set with a
        # cardinality, and the identity holds.
        o = po.partial_fit!(OPS(; alg = FollowTheLeadingHistory(; alg = eg)), rows(rd, 1:7))
        o = po.partial_fit!(o, rows(rd, 8:12))
        @test optimise(o).w ≈
              optimise(OPS(; alg = FollowTheLeadingHistory(; alg = eg)), rows(rd, 1:12)).w
        # Copy, view and merge of the carrier.
        @test copy(stp).h == stp.h && !(copy(stp).h[1] === stp.h[1])
        v = po.port_opt_view(stp, [1, 2])
        @test length(v.h[1]) == 2 && sum(v.h[1]) ≈ 1 && v.born == stp.born
        @test_throws ArgumentError po.merge_states(stp, stp)
        @test po.port_opt_view(FollowTheLeadingHistory(; alg = eg), [1, 2]).alg === eg
        # Under the Newton base and a capped set the recursion stays in the set.
        wn = optimise(OPS(; alg = FollowTheLeadingHistory(),
                          set = BoundedAllocationSet(; wb = WeightBounds(0, 0.4))), rd).w
        @test maximum(wn) <= 0.4 + 1e-10 && sum(wn) ≈ 1
    end

    @testset "Configurations: the pattern-matching aggregations" begin
        # The kernel paper's grid of (window, radius) experts under the wealth weighting:
        # the mixture's wealth is at least the best expert's over the count.
        experts = [FollowTheLeader(;
                                   sel = KernelMatch(; window = w, radius = 0.05 * 4 / l))
                   for w in 1:2 for l in 1:3]
        mix = OPS(; alg = ExpertMixture(; experts = experts))
        wm = optimise(mix, rd).w
        @test sum(wm) ≈ 1 && all(wm .>= -1e-12)
        pathw(alg) = [optimise(OPS(; alg = alg), rows(rd, 1:t)).w for t in 1:(T - 1)]
        wealth(path) = prod(dot(path[t], X[t + 1, :]) for t in 1:(T - 1))
        wealths = [wealth(pathw(e)) for e in experts]
        @test wealth(pathw(ExpertMixture(; experts = experts))) >=
              maximum(wealths) / length(experts) * (1 - 1e-10)
        # CORN-U: windows 1 to W at one threshold, mixed uniformly by wealth.
        cornu = ExpertMixture(;
                              experts = [FollowTheLeader(;
                                                         sel = CorrelationMatch(;
                                                                                window = w))
                                         for w in 1:3])
        @test sum(optimise(OPS(; alg = cornu), rd).w) ≈ 1
        # RACORN-K: correlation-matched samples under a standard-deviation penalty on
        # the log return, one expert per threshold, the top-k of them by wealth.
        sdopt = MeanRisk(; r = StandardDeviation(), obj = MaximumUtility(; l = 1),
                         opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv,
                                             ret = LogarithmicReturn()))
        racorn = ExpertMixture(;
                               experts = [FollowTheLeader(;
                                                          sel = CorrelationMatch(;
                                                                                 window = 2,
                                                                                 rho = rho),
                                                          opt = sdopt) for rho in 0:0.3:0.9],
                               alg = TopK(; k = 2))
        wr = optimise(OPS(; alg = racorn), rows(rd, 1:12)).w
        @test sum(wr) ≈ 1 atol = 1e-6
        @test all(wr .>= -1e-8)
        pr = po.partial_fit!(OPS(; alg = racorn), rows(rd, 1:12)).cache.st.p
        @test count(>(0), pr) == 2
        # A mixture of leaders under a programme set: every expert's programme takes the
        # set, and the blend is projected once more.
        wpm = optimise(OPS(;
                           alg = ExpertMixture(;
                                               experts = [FollowTheLeader(; opt = logopt()),
                                                          FollowTheLeader(;
                                                                          sel = LastRows(;
                                                                                         W = 5),
                                                                          opt = logopt())]),
                           set = ProgrammeAllocationSet(; slv = slv,
                                                        wb = WeightBounds(0, 0.5))),
                       rows(rd, 1:10)).w
        @test maximum(wpm) <= 0.5 + 1e-6
    end

    @testset "Docstrings and the public surface" begin
        docs = [string(@doc(FollowTheLeader)), string(@doc(FollowTheLeadingHistory)),
                string(@doc(Prefix)), string(@doc(LastRows)), string(@doc(HistogramMatch)),
                string(@doc(KernelMatch)), string(@doc(NearestNeighbourMatch)),
                string(@doc(CorrelationMatch)), string(@doc(ClusterMatch)),
                string(@doc(RankOneCovariance)),
                string(@doc(ShortTermLossControlPortfolio)),
                string(@doc(po.AllocationSetConstraint)),
                string(@doc(po.AbstractSampleSelector)), string(@doc(po.covariance_factor))]
        @test all(d -> !occursin("No documentation found", d), docs)
        @test occursin("follow-the-leader rule", docs[1])
    end
end
