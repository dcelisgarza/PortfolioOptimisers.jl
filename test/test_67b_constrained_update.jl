#=
The Constrained Update seam of the online portfolio selection family, as ADRs 0159, 0163 and
0164 rule it (issue #1162 on map #1148): the Gram Projection Geometry, the programme
Allocation Set, the bare-model programme in the three divergences, the shared set builders,
the Held Step, and the dispatch that skips a mixture's second projection on a bounded set.

The four parity tests ADR 0159 names: the Euclidean root under `(0, 1)` equals the sort on
every step of the rules; the entropic root under a loose cap equals normalisation; the
programme under a slack linear constraint equals the closed form in both geometries; the
Gram programme on the bare simplex matches a hand-written quadratic programme.
=#
using Test, PortfolioOptimisers, StableRNGs, LinearAlgebra, Statistics, Dates, Clarabel,
      HiGHS, Pajarito, JuMP
@testset "Online portfolio selection: the Constrained Update seam" begin
    po = PortfolioOptimisers
    OPS = po.OnlinePortfolioSelection

    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false, "tol_gap_abs" => 1e-12,
                                 "tol_gap_rel" => 1e-12, "tol_feas" => 1e-12),
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
    resolve(set, N) = po.resolve_allocation_set(set, N, false, Float64)
    bset = resolve(BoundedAllocationSet(), 3)
    pset = resolve(ProgrammeAllocationSet(; slv = slv), 3)
    q = [0.5, 0.3, 0.4]
    wh = [0.3, 0.3, 0.4]

    @testset "Construction: the geometry, the set and the refusals" begin
        @test GramProjection(; slv = slv).A === nothing
        @test_throws DimensionMismatch GramProjection(; slv = slv, A = ones(2, 3))
        @test_throws UndefKeywordError ProgrammeAllocationSet()
        @test_throws TypeError ProgrammeAllocationSet(; slv = nothing)
        @test_throws DomainError ProgrammeAllocationSet(; slv = slv, tn = -0.1)
        @test_throws DomainError ProgrammeAllocationSet(; slv = slv, card = 0)
        @test_throws ArgumentError ProgrammeAllocationSet(; slv = slv, r = Variance())
        @test_throws Exception ProgrammeAllocationSet(; slv = slv,
                                                      wb = WeightBoundsEstimator())
        @test_throws TypeError ProgrammeAllocationSet(; slv = slv,
                                                      te = TrackingError(;
                                                                         tr = ReturnsTracking(;
                                                                                              w = zeros(3))))
        # The NewtonStep slot admits the Gram geometry and no other rule's does.
        @test isa(NewtonStep(; proj = GramProjection(; slv = slv)).proj, GramProjection)
        @test_throws TypeError MirrorDescent(; proj = GramProjection(; slv = slv))
        @test_throws TypeError PassiveAggressiveMeanReversion(;
                                                              proj = GramProjection(;
                                                                                    slv = slv))
        # A negative lower bound under an entropic rule is refused where alg and set meet.
        @test_throws DomainError OPS(; alg = ExponentiatedGradient(),
                                     set = BoundedAllocationSet(;
                                                                wb = WeightBounds(;
                                                                                  lb = -0.1,
                                                                                  ub = 1)))
        @test_throws DomainError OPS(; alg = ExponentiatedGradient(),
                                     set = ProgrammeAllocationSet(; slv = slv,
                                                                  wb = WeightBounds(;
                                                                                    lb = -0.1,
                                                                                    ub = 1)))
        @test isa(OPS(; alg = PassiveAggressiveMeanReversion(),
                      set = ProgrammeAllocationSet(; slv = slv,
                                                   wb = WeightBounds(; lb = -0.1, ub = 1))),
                  OPS)
        # The Start Allocation of a Gram-projected Newton step is Euclidean: A_0 = I.
        @test po.projection_geometry(NewtonStep(; proj = GramProjection(; slv = slv))) ==
              EuclideanProjection()
        @test po.rows_needed(ProgrammeAllocationSet(; slv = slv)) == 0
        @test isnothing(po.rows_needed(ProgrammeAllocationSet(; slv = slv,
                                                              r = Variance(;
                                                                           settings = RiskMeasureSettings(;
                                                                                                          ub = 1e-3)))))
        @test po.rows_needed(ProgrammeAllocationSet(; slv = slv,
                                                    r = Variance(;
                                                                 sigma = Matrix{Float64}(I,
                                                                                         3,
                                                                                         3),
                                                                 settings = RiskMeasureSettings(;
                                                                                                ub = 1e-3)))) ==
              0
        @test isnothing(po.rows_needed(ProgrammeAllocationSet(; slv = slv,
                                                              te = TrackingError(;
                                                                                 tr = WeightsTracking(;
                                                                                                      w = fill(1 /
                                                                                                               3,
                                                                                                               3)),
                                                                                 err = 0.01))))
        # The head's need is the maximum over the rule tree and the set.
        @test isnothing(po.rows_needed(OPS(; alg = BuyAndHold(),
                                           set = ProgrammeAllocationSet(; slv = slv,
                                                                        te = TrackingError(;
                                                                                           tr = WeightsTracking(;
                                                                                                                w = fill(1 /
                                                                                                                         3,
                                                                                                                         3)))))))
        # A view slices what has an asset axis and carries the rest.
        set = ProgrammeAllocationSet(; slv = slv, tn = [0.1, 0.2, 0.3], card = 2)
        v = po.port_opt_view(set, [1, 3])
        @test v.tn == [0.1, 0.3] && v.card == 2 && v.slv === slv
        # Resolution turns the name-keyed constraints into values.
        rs = resolve(ProgrammeAllocationSet(; slv = slv,
                                            sets = UniverseSets(;
                                                                dict = Dict("nx" =>
                                                                                ["A", "B",
                                                                                 "C"])),
                                            lcs = LinearConstraintEstimator(;
                                                                            val = :(A + B <=
                                                                                    0.8))),
                     3)
        @test isa(rs.lcs, po.LinearConstraint)
    end

    @testset "Parity 1 and 2: the scalar roots against the sort and normalisation" begin
        rng = StableRNG(3)
        for _ in 1:20
            v = randn(rng, 5)
            @test po.project(EuclideanProjection(), resolve(BoundedAllocationSet(), 5), v,
                             v) == po.project_simplex(v)
            u = abs.(v)
            @test po.project(EntropicProjection(),
                             resolve(BoundedAllocationSet(;
                                                          wb = WeightBounds(; lb = 0,
                                                                            ub = 0.999)),
                                     5), u, u) ≈ u ./ sum(u)
        end
    end

    @testset "Parity 3: the programme under a slack constraint equals the closed form" begin
        # A linear constraint no step hits: the programme and the root agree in both
        # geometries, and under a binding cap too.
        slack = resolve(ProgrammeAllocationSet(; slv = slv,
                                               sets = UniverseSets(;
                                                                   dict = Dict("nx" =>
                                                                                   ["A",
                                                                                    "B",
                                                                                    "C"])),
                                               lcs = LinearConstraintEstimator(;
                                                                               val = :(A +
                                                                                       B <=
                                                                                       0.99))),
                        3)
        @test isapprox(po.project(EuclideanProjection(), slack, q, wh),
                       po.project(EuclideanProjection(), bset, q, wh); atol = 1e-7)
        @test isapprox(po.project(EntropicProjection(), slack, q, wh),
                       po.project(EntropicProjection(), bset, q, wh); atol = 1e-5)
        cap = WeightBounds(; lb = 0, ub = 0.4)
        qq = [0.462, 0.293, 0.245]
        @test isapprox(po.project(EntropicProjection(),
                                  resolve(ProgrammeAllocationSet(; slv = slv, wb = cap), 3),
                                  qq, wh), [0.400, 0.327, 0.273]; atol = 1e-3)
        @test isapprox(po.project(EntropicProjection(),
                                  resolve(ProgrammeAllocationSet(; slv = slv, wb = cap), 3),
                                  qq, wh),
                       po.project(EntropicProjection(),
                                  resolve(BoundedAllocationSet(; wb = cap), 3), qq, wh);
                       atol = 1e-7)
        @test isapprox(po.project(EuclideanProjection(),
                                  resolve(ProgrammeAllocationSet(; slv = slv, wb = cap), 3),
                                  qq, wh),
                       po.project(EuclideanProjection(),
                                  resolve(BoundedAllocationSet(; wb = cap), 3), qq, wh);
                       atol = 1e-7)
        # A binding linear constraint is honoured, and a zero entry of the raw step stays
        # zero under the entropic programme, as it does under the closed form.
        tight = resolve(ProgrammeAllocationSet(; slv = slv,
                                               sets = UniverseSets(;
                                                                   dict = Dict("nx" =>
                                                                                   ["A",
                                                                                    "B",
                                                                                    "C"])),
                                               lcs = LinearConstraintEstimator(;
                                                                               val = :(A <=
                                                                                       0.2))),
                        3)
        wt = po.project(EuclideanProjection(), tight, q, wh)
        @test wt[1] ≈ 0.2 atol = 1e-7
        @test sum(wt) ≈ 1 atol = 1e-8
        wz = po.project(EntropicProjection(), pset, [0.6, 0.0, 0.4], wh)
        @test wz[2] ≈ 0 atol = 1e-8
        @test isapprox(wz, [0.6, 0.0, 0.4]; atol = 1e-5)
        @test_throws DomainError po.project(EntropicProjection(), pset, [0.5, -0.1, 0.6],
                                            wh)
        @test_throws DomainError po.project(EntropicProjection(),
                                            resolve(ProgrammeAllocationSet(; slv = slv,
                                                                           wb = WeightBounds(;
                                                                                             lb = -0.1,
                                                                                             ub = 1)),
                                                    3), q, wh)
        # A negative lower bound under the Euclidean programme: a long-short answer.
        ls = resolve(ProgrammeAllocationSet(; slv = slv,
                                            wb = WeightBounds(; lb = -0.2, ub = 1)), 3)
        wls = po.project(EuclideanProjection(), ls, [0.9, -0.3, 0.4], wh)
        @test wls[2] ≈ -0.2 atol = 1e-7
        @test sum(wls) ≈ 1 atol = 1e-8
        @test isapprox(wls, [0.85, -0.2, 0.35]; atol = 1e-6)
        @test isapprox(wls,
                       po.project(EuclideanProjection(),
                                  resolve(BoundedAllocationSet(;
                                                               wb = WeightBounds(;
                                                                                 lb = -0.2,
                                                                                 ub = 1)),
                                          3), [0.9, -0.3, 0.4], wh); atol = 1e-6)
    end

    @testset "Parity 4: the Gram programme on the bare simplex" begin
        A = [2.0 0.3 0.1; 0.3 1.5 0.2; 0.1 0.2 1.2]
        g = GramProjection(; slv = slv, A = A)
        wg = po.project(g, bset, q, wh)
        m = JuMP.Model(Clarabel.Optimizer)
        JuMP.set_silent(m)
        JuMP.@variable(m, x[1:3] >= 0)
        JuMP.@constraint(m, sum(x) == 1)
        JuMP.@objective(m, Min, (x - q)' * A * (x - q))
        JuMP.optimize!(m)
        @test isapprox(wg, JuMP.value.(x); atol = 1e-6)
        # The Gram norm is not the Euclidean one: the answers differ off the identity.
        @test !isapprox(wg, po.project_simplex(q); atol = 1e-3)
        @test isapprox(po.project(GramProjection(; slv = slv, A = Matrix{Float64}(I, 3, 3)),
                                  bset, q, wh), po.project_simplex(q); atol = 1e-6)
        # The geometry carries its own solver, on a programme set too.
        @test isapprox(po.project(g, pset, q, wh), wg; atol = 1e-6)
        @test_throws ArgumentError po.project(GramProjection(; slv = slv), bset, q, wh)
        @test_throws ArgumentError po.projection_solver(EuclideanProjection(), bset)
        @test po.projection_solver(EuclideanProjection(), pset) === slv
        # The rule binds its Gram matrix at every step and runs solver-free at the default.
        rng = StableRNG(7)
        T, N = 20, 4
        R = 0.02 .* randn(rng, T, N)
        rd = ReturnsResult(; nx = ["A", "B", "C", "D"], X = R,
                           ts = Date(2020, 1, 1) .+ Day.(0:(T - 1)))
        res_g = optimise(OPS(; alg = NewtonStep(; proj = GramProjection(; slv = slv))), rd)
        res_e = optimise(OPS(; alg = NewtonStep()), rd)
        @test isa(res_g.retcode, OptimisationSuccess) && isnothing(res_g.retcode.res)
        @test sum(res_g.w) ≈ 1 atol = 1e-8
        @test !isapprox(res_g.w, res_e.w; atol = 1e-4)
        # The mix precedes the projection: a zero cap is honoured at eta > 0.
        capped = OPS(; alg = NewtonStep(; eta = 0.2),
                     set = BoundedAllocationSet(;
                                                wb = WeightBounds(; lb = 0,
                                                                  ub = [1, 0, 1, 1])))
        @test optimise(capped, rd).w[2] == 0
    end

    @testset "The turnover ceiling, the covariance cone, the tracking error and the MIP kinds" begin
        # The ceiling is per asset, measured from the Price-Adjusted Allocation.
        tset = resolve(ProgrammeAllocationSet(; slv = slv, tn = 0.1), 3)
        wt = po.project(EuclideanProjection(), tset, [1.0, 0.0, 0.0], wh)
        @test maximum(abs.(wt .- wh)) ≈ 0.1 atol = 1e-6
        @test isapprox(wt, [0.4, 0.3, 0.3]; atol = 1e-6)
        # A ceiling of zero pins the book: the projection is the book itself.
        @test isapprox(po.project(EuclideanProjection(),
                                  resolve(ProgrammeAllocationSet(; slv = slv, tn = 0.0), 3),
                                  [1.0, 0.0, 0.0], wh), wh; atol = 1e-6)
        # The covariance cone, from a matrix on `r`: the step meets the ceiling.
        S = [4.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0] .* 1e-4
        vset = resolve(ProgrammeAllocationSet(; slv = slv,
                                              r = StandardDeviation(; sigma = S,
                                                                    settings = RiskMeasureSettings(;
                                                                                                   ub = 0.011))),
                       3)
        wv = po.project(EuclideanProjection(), vset, [1.0, 0.0, 0.0], wh)
        @test sqrt(dot(wv, S, wv)) ≈ 0.011 atol = 1e-6
        vset2 = resolve(ProgrammeAllocationSet(; slv = slv,
                                               r = Variance(; sigma = S,
                                                            settings = RiskMeasureSettings(;
                                                                                           ub = 0.011^2))),
                        3)
        @test isapprox(po.project(EuclideanProjection(), vset2, [1.0, 0.0, 0.0], wh), wv;
                       atol = 1e-6)
        # The fitted covariance reads the head's rows: outside a step it is refused, and
        # inside one it holds the step until the head has two rows.
        fset = resolve(ProgrammeAllocationSet(; slv = slv,
                                              r = Variance(;
                                                           settings = RiskMeasureSettings(;
                                                                                          ub = 4e-4))),
                       4)
        @test_throws ArgumentError po.project(EuclideanProjection(), fset, fill(0.25, 4),
                                              fill(0.25, 4))
        rng = StableRNG(7)
        T, N = 30, 4
        R = 0.02 .* randn(rng, T, N)
        rd = ReturnsResult(; nx = ["A", "B", "C", "D"], X = R,
                           ts = Date(2020, 1, 1) .+ Day.(0:(T - 1)))
        fopt = OPS(; alg = PassiveAggressiveMeanReversion(),
                   set = ProgrammeAllocationSet(; slv = slv,
                                                r = Variance(;
                                                             settings = RiskMeasureSettings(;
                                                                                            ub = 4e-4))))
        r1 = @test_logs (:warn, r"Held Step at 2020-01-01.*covariance of one observation") match_mode=:any optimise(fopt,
                                                                                                                    po.port_opt_view(rd,
                                                                                                                                     1:1,
                                                                                                                                     :))
        @test isa(r1.retcode.res, po.HeldStep)
        @test r1.retcode.res.ts == Date(2020, 1, 1)
        @test isnothing(r1.retcode.res.trials)
        # The step traded nothing: the answer is the book after the row's move.
        x1 = 1 .+ R[1, :]
        @test isapprox(r1.w, 0.25 .* x1 ./ dot(fill(0.25, 4), x1); atol = 1e-12)
        # The row buffer is unbounded for a fitted set, and every later row solves.
        r2 = optimise(fopt, rd)
        @test isa(r2.retcode, OptimisationSuccess)
        @test dot(r2.w, cov(R), r2.w) <= 4e-4 * (1 + 1e-6)
        # The tracking error over the head's rows.
        topt = OPS(; alg = PassiveAggressiveMeanReversion(),
                   set = ProgrammeAllocationSet(; slv = slv,
                                                te = TrackingError(;
                                                                   tr = WeightsTracking(;
                                                                                        w = fill(0.25,
                                                                                                 4)),
                                                                   err = 0.003)))
        rt = optimise(topt, rd)
        @test isa(rt.retcode, OptimisationSuccess)
        @test sqrt(sum(abs2, R * (rt.w .- 0.25)) / (T - 1)) <= 0.003 * (1 + 1e-6)
        @test !isapprox(rt.w, optimise(OPS(; alg = PassiveAggressiveMeanReversion()), rd).w;
                        atol = 1e-6)
        # A cardinality on the projection: the step is one-hot at card = 1.
        cset = resolve(ProgrammeAllocationSet(; slv = mip, card = 1), 3)
        wc = po.project(EuclideanProjection(), cset, q, wh)
        @test count(x -> x > 1e-6, wc) == 1
        @test wc[1] ≈ 1 atol = 1e-6
        # The set builders add to a model a head did not build (ADR 0164).
        m = JuMP.Model()
        po.set_model_scales!(m, 1, 1)
        po.set_model_observations!(m, 0)
        JuMP.@expression(m, k, 1)
        JuMP.@variable(m, w[1:3])
        po.set_allocation_set_constraints!(m, tset, wh, nothing)
        @test JuMP.num_constraints(m; count_variable_in_set_constraints = true) > 2
        m2 = JuMP.Model()
        po.set_model_scales!(m2, 1, 1)
        po.set_model_observations!(m2, 0)
        JuMP.@expression(m2, k, 1)
        JuMP.@variable(m2, w[1:3])
        po.set_allocation_set_constraints!(m2, bset, wh, nothing)
        @test JuMP.num_constraints(m2; count_variable_in_set_constraints = true) == 3
    end

    @testset "The Held Step" begin
        # A cap of 0.4 and a ceiling of 0.01 from a book at 0.7 cannot both hold.
        hset = resolve(ProgrammeAllocationSet(; slv = slv, tn = 0.01,
                                              wb = WeightBounds(; lb = 0, ub = 0.4)), 3)
        book = [0.7, 0.2, 0.1]
        (wheld, held) = po.with_projection_step(() -> po.project(EuclideanProjection(),
                                                                 hset, q, book), nothing,
                                                Date(2020, 1, 1))
        @test wheld == book && !(wheld === book)
        @test length(held) == 1
        @test held[1].ts == Date(2020, 1, 1)
        @test occursin("trades nothing", held[1].reason)
        @test haskey(held[1].trials, :clarabel)
        # Outside a step the hold is a warning, and the answer is still the book.
        @test (@test_logs (:warn, r"Held Step outside") match_mode=:any po.project(EuclideanProjection(),
                                                                                   hset, q,
                                                                                   book)) ==
              book
        # Inside a fold: the carrier still absorbs the row, the head warns with the row's
        # timestamp, the read-out carries the record as a success, and the fallback never
        # runs.
        Rh = [0.5 -0.2 0.0; 0.01 0.02 -0.01; 0.0 0.0 0.0]
        rdh = ReturnsResult(; nx = ["A", "B", "C"], X = Rh,
                            ts = [Date(2021, 1, 1), Date(2021, 1, 2), Date(2021, 1, 3)])
        hopt = OPS(; alg = NewtonStep(), w0 = [0.4, 0.3, 0.3],
                   set = ProgrammeAllocationSet(; slv = slv, tn = 0.01,
                                                wb = WeightBounds(; lb = 0, ub = 0.4)),
                   fb = EqualWeighted())
        res1 = @test_logs (:warn, r"Held Step at 2021-01-01") match_mode=:any optimise(hopt,
                                                                                       po.port_opt_view(rdh,
                                                                                                        1:1,
                                                                                                        :))
        @test isa(res1.retcode, OptimisationSuccess)
        @test isa(res1.retcode.res, po.HeldStep)
        @test res1.retcode.res.ts == Date(2021, 1, 1)
        # The step traded nothing: the answer is the book after the row's move.
        x1 = 1 .+ Rh[1, :]
        @test isapprox(res1.w, [0.4, 0.3, 0.3] .* x1 ./ dot([0.4, 0.3, 0.3], x1);
                       atol = 1e-12)
        st = po.partial_fit!(hopt, po.port_opt_view(rdh, 1:1, :)).cache
        @test st.st.n == 1 && isa(st.hold, po.HeldStep)
        # The retcode is the last step's: a book the ceiling can never bring back under the
        # cap is held on every row, and the record names the last one.
        res3 = optimise(hopt, rdh)
        @test isa(res3.retcode, OptimisationSuccess)
        @test res3.retcode.res.ts == Date(2021, 1, 3)
        # The state carries the hold through a copy and a view.
        @test copy(st).hold === st.hold
        @test po.port_opt_view(st, [1, 2]).hold === st.hold
        # The batch–online identity holds through a held row.
        opt2 = po.partial_fit!(hopt, po.port_opt_view(rdh, 1:2, :))
        @test optimise(opt2).w ≈ optimise(hopt, po.port_opt_view(rdh, 1:2, :)).w
    end

    @testset "The mixture's second projection (ADR 0163)" begin
        experts = [ConstantRebalancedPortfolio(; w = [1.0, 0.0]),
                   ConstantRebalancedPortfolio(; w = [0.0, 1.0])]
        x = [1.2, 0.8]
        w = [0.5, 0.5]
        tn = resolve(ProgrammeAllocationSet(; slv = slv, tn = 0.1), 2)
        step(alg, set) = first(po.with_projection_step(() -> po.online_update!(ExpertMixture(;
                                                                                             experts = experts,
                                                                                             alg = alg),
                                                                               po.rule_state_seed(ExpertMixture(;
                                                                                                                experts = experts,
                                                                                                                alg = alg),
                                                                                                  w),
                                                                               w, x,
                                                                               nothing, set),
                                                       nothing, 1))[2]
        # 1. Under buy-and-hold the blend is the drifted book, and the projection is the
        #    identity under the ceiling.
        @test isapprox(step(BuyAndHold(), tn), [0.6, 0.4]; atol = 1e-8)
        # 2. Under an exponentiated-gradient weighting the blend [0.88, 0.12] leaves the
        #    ceiling and is repaired to the nearest point within 0.1 per asset of the book
        #    [0.6, 0.4]: [0.70, 0.30].
        b = step(ExponentiatedGradient(; eta = 5), resolve(BoundedAllocationSet(), 2))
        @test isapprox(b, [0.8808, 0.1192]; atol = 1e-4)
        @test isapprox(step(ExponentiatedGradient(; eta = 5), tn), [0.70, 0.30];
                       atol = 1e-6)
        # 3. Under card = 1 the blend becomes the one-hot on its largest entry.
        @test isapprox(step(BuyAndHold(),
                            resolve(ProgrammeAllocationSet(; slv = mip, card = 1), 2)),
                       [1.0, 0.0]; atol = 1e-6)
        # On a bounded set the second projection is skipped by dispatch: the blend is
        # returned as it is, aliasing nothing.
        qb = [0.55, 0.45]
        @test po.blend_projection(EuclideanProjection(), resolve(BoundedAllocationSet(), 2),
                                  qb, w) === qb
        @test po.blend_projection(EuclideanProjection(), tn, [0.9, 0.1], [0.6, 0.4]) ≈
              [0.7, 0.3] atol = 1e-6
    end

    @testset "Any risk measure as the ceiling (#1163)" begin
        # The set owns the shared builders' constraint: a measure the JuMP optimisers bound
        # is a ceiling on the projection, and the refusals are the ones the set cannot build.
        ub_of(r, ub) = RiskMeasureSettings(; ub = ub, scale = r.settings.scale,
                                           rke = r.settings.rke)
        @test_throws ArgumentError ProgrammeAllocationSet(; slv = slv,
                                                          r = ConditionalValueatRisk())
        @test_throws ArgumentError ProgrammeAllocationSet(; slv = slv,
                                                          r = ConditionalValueatRisk(;
                                                                                     settings = RiskMeasureSettings(;
                                                                                                                    ub = Frontier(;
                                                                                                                                  N = 3))))
        @test_throws ArgumentError ProgrammeAllocationSet(; slv = slv,
                                                          r = Variance(;
                                                                       rc = LinearConstraintEstimator(;
                                                                                                      val = :(A <=
                                                                                                              0.5)),
                                                                       settings = RiskMeasureSettings(;
                                                                                                      ub = 1e-3)))
        cvar = ConditionalValueatRisk(; settings = RiskMeasureSettings(; ub = 0.02))
        mdd = MaximumDrawdown(; settings = RiskMeasureSettings(; ub = 0.05))
        @test isnothing(po.rows_needed(ProgrammeAllocationSet(; slv = slv, r = cvar)))
        @test po.risk_reads_rows(cvar) && po.risk_reads_rows(Variance())
        @test !po.risk_reads_rows(nothing) && !po.risk_reads_rows(Variance(; sigma = I(3)))
        # A two-argument view, the read-out's, carries a tail measure unchanged.
        @test po.port_opt_view(ProgrammeAllocationSet(; slv = slv, r = cvar), [1, 3]).r ===
              cvar
        # The set is a Risk Constraint Owner: its solver, and no risk-contribution rows.
        cset = resolve(ProgrammeAllocationSet(; slv = slv, r = cvar), 4)
        @test isa(cset, po.AbstractProgrammeAllocationSet)
        @test isa(cset, po.RiskConstraintOwner) && isa(cset, po.RiskBoundOwner)
        @test po.risk_constraint_solver(cset) === slv
        @test isnothing(po.risk_contribution_constraints(Variance(), cset,
                                                         prior(EmpiricalPrior(),
                                                               randn(StableRNG(1), 8, 4))))
        # Inside a step the ceiling binds on the head's rows: the tail of the projected
        # allocation meets the number, and differs from the unconstrained projection.
        rng = StableRNG(19)
        T, N = 60, 4
        R = 0.02 .* randn(rng, T, N)
        R[:, 1] .*= 3
        q4 = [0.9, 0.05, 0.03, 0.02]
        w4 = fill(0.25, 4)
        free = po.project(EuclideanProjection(),
                          resolve(ProgrammeAllocationSet(; slv = slv), 4), q4, w4)
        @test expected_risk(cvar, free, R) > 0.02
        (wc, hc) = po.with_projection_step(() -> po.project(EuclideanProjection(), cset, q4,
                                                            w4), R, 1)
        @test isempty(hc)
        @test expected_risk(cvar, wc, R) <= 0.02 * (1 + 1e-4)
        @test expected_risk(cvar, wc, R) >= 0.02 * (1 - 1e-2)
        @test !isapprox(wc, free; atol = 1e-3)
        # A drawdown ceiling, and a variance built on the fitted prior through the same
        # route as the tail measures, which agrees with the cone the matrix route writes.
        mset = resolve(ProgrammeAllocationSet(; slv = slv, r = mdd), 4)
        (wm, hm) = po.with_projection_step(() -> po.project(EuclideanProjection(), mset, q4,
                                                            w4), R, 1)
        @test isempty(hm)
        @test expected_risk(mdd, wm, R) <= 0.05 * (1 + 1e-4)
        @test !isapprox(wm, free; atol = 1e-3)
        S = cov(R)
        vub = 0.5 * dot(free, S, free)
        vfit = resolve(ProgrammeAllocationSet(; slv = slv,
                                              r = Variance(;
                                                           settings = RiskMeasureSettings(;
                                                                                          ub = vub))),
                       4)
        vmat = resolve(ProgrammeAllocationSet(; slv = slv,
                                              r = Variance(; sigma = S,
                                                           settings = RiskMeasureSettings(;
                                                                                          ub = vub))),
                       4)
        (wf, _) = po.with_projection_step(() -> po.project(EuclideanProjection(), vfit, q4,
                                                           w4), R, 1)
        @test isapprox(wf, po.project(EuclideanProjection(), vmat, q4, w4); atol = 1e-5)
        @test dot(wf, S, wf) <= vub * (1 + 1e-4)
        # The measure's `rke` is cleared and its expression joins no objective: the model
        # holds the bound and no risk vector, whichever the measure states.
        rke = ConditionalValueatRisk(;
                                     settings = RiskMeasureSettings(; ub = 0.02,
                                                                    rke = true))
        m = JuMP.Model()
        po.set_model_scales!(m, 1, 1)
        po.set_model_observations!(m, T)
        JuMP.@expression(m, k, 1)
        JuMP.@variable(m, w[1:4])
        po.set_allocation_set_constraints!(m,
                                           resolve(ProgrammeAllocationSet(; slv = slv,
                                                                          r = rke), 4), w4,
                                           R)
        @test haskey(m, :cvar_risk_1_ub) && !haskey(m, :risk_vec)
        # A measure that needs a quantity the prior does not carry is refused by name.
        kset = resolve(ProgrammeAllocationSet(; slv = slv,
                                              r = Kurtosis(;
                                                           settings = RiskMeasureSettings(;
                                                                                          ub = 1e-3))),
                       4)
        @test_throws ArgumentError po.with_projection_step(() -> po.project(EuclideanProjection(),
                                                                            kset, q4, w4),
                                                           R, 1)
        # Through a head: the tail ceiling holds on every step, and the fold before two
        # rows is the Held Step every fitted ceiling takes.
        rd = ReturnsResult(; nx = ["A", "B", "C", "D"], X = R,
                           ts = Date(2020, 1, 1) .+ Day.(0:(T - 1)))
        copt = OPS(; alg = PassiveAggressiveMeanReversion(),
                   set = ProgrammeAllocationSet(; slv = slv, r = cvar))
        rc1 = @test_logs (:warn, r"Held Step at 2020-01-01.*covariance of one observation") match_mode=:any optimise(copt,
                                                                                                                     po.port_opt_view(rd,
                                                                                                                                      1:1,
                                                                                                                                      :))
        @test isa(rc1.retcode.res, po.HeldStep)
        rcT = optimise(copt, rd)
        @test isa(rcT.retcode, OptimisationSuccess)
        @test expected_risk(cvar, rcT.w, R) <= 0.02 * (1 + 1e-4)
        # On a JuMP leader's model the ceiling lives in its own namespace beside the head's
        # measure of the same kind, and both bind: the head's on its selection, the set's
        # on the head's rows.
        hcvar = ConditionalValueatRisk(; settings = RiskMeasureSettings(; ub = 0.03))
        lead = FollowTheLeader(;
                               opt = MeanRisk(; obj = MaximumReturn(), r = hcvar,
                                              opt = JuMPOptimiser(; pe = EmpiricalPrior(),
                                                                  slv = slv,
                                                                  ret = LogarithmicReturn())))
        lres = optimise(OPS(; alg = lead,
                            set = ProgrammeAllocationSet(; slv = slv, r = cvar)), rd)
        @test isa(lres.retcode, OptimisationSuccess)
        @test expected_risk(cvar, lres.w, R) <= 0.02 * (1 + 1e-4)
        @test !isapprox(lres.w,
                        optimise(OPS(; alg = lead,
                                     set = ProgrammeAllocationSet(; slv = slv)), rd).w;
                        atol = 1e-3)
        @test occursin("RiskConstraintOwner", string(@doc(po.RiskConstraintOwner)))
        @test occursin("any", string(@doc(ProgrammeAllocationSet)))
    end
    @testset "Docs and the search seam" begin
        @test occursin("Held Step", string(@doc(po.HeldStep)))
        @test occursin("per-asset", string(@doc(ProgrammeAllocationSet)))
        @test occursin("before", string(@doc(NewtonStep)))
        @test occursin("K + 1", string(@doc(ExpertMixture)))
        io = IOBuffer()
        show(io, MIME"text/plain"(), ProgrammeAllocationSet(; slv = slv, tn = 0.1))
        @test occursin("ProgrammeAllocationSet", String(take!(io)))
    end
end
