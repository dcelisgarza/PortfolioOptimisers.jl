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
# Two custom terms with the set as their owner (#1206).
struct FirstAtMost <: PortfolioOptimisers.CustomJuMPConstraint
    ub::Float64
end
function PortfolioOptimisers.add_custom_constraint!(model::JuMP.Model, c::FirstAtMost,
                                                    ::Any, ::Any)
    w = PortfolioOptimisers.get_w(model)
    k = PortfolioOptimisers.get_k(model)
    sc = PortfolioOptimisers.get_constraint_scale(model)
    JuMP.@constraint(model, sc * (w[1] - c.ub * k) <= 0)
    return nothing
end
struct FirstCosts <: PortfolioOptimisers.CustomJuMPObjective
    c::Float64
end
function PortfolioOptimisers.add_custom_objective_term!(model::JuMP.Model, ::Any,
                                                        t::FirstCosts, ::Any, ::Any)
    w = PortfolioOptimisers.get_w(model)
    PortfolioOptimisers.add_to_objective_penalty!(model,
                                                  JuMP.@expression(model, t.c * w[1]))
    return nothing
end
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
        @test_throws TypeError ProgrammeAllocationSet(; slv = slv, tn = -0.1)
        @test_throws DomainError ProgrammeAllocationSet(; slv = slv, card = 0)
        @test_throws ArgumentError ProgrammeAllocationSet(; slv = slv, r = Variance())
        @test_throws Exception ProgrammeAllocationSet(; slv = slv,
                                                      wb = WeightBoundsEstimator())
        @test isa(ProgrammeAllocationSet(; slv = slv,
                                         tr = TrackingError(;
                                                            tr = ReturnsTracking(;
                                                                                 w = zeros(3)))),
                  ProgrammeAllocationSet)
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
                                                              tr = TrackingError(;
                                                                                 tr = WeightsTracking(;
                                                                                                      w = fill(1 /
                                                                                                               3,
                                                                                                               3)),
                                                                                 err = 0.01))))
        # The head's need is the maximum over the rule tree and the set.
        @test isnothing(po.rows_needed(OPS(; alg = BuyAndHold(),
                                           set = ProgrammeAllocationSet(; slv = slv,
                                                                        tr = TrackingError(;
                                                                                           tr = WeightsTracking(;
                                                                                                                w = fill(1 /
                                                                                                                         3,
                                                                                                                         3)))))))
        # A view slices what has an asset axis and carries the rest.
        set = ProgrammeAllocationSet(; slv = slv,
                                     tn = Turnover(; w = zeros(3), val = [0.1, 0.2, 0.3]),
                                     card = 2)
        v = po.port_opt_view(set, [1, 3])
        @test v.tn.val == [0.1, 0.3] && v.card == 2 && v.slv === slv
        # Resolution turns the name-keyed constraints into values.
        rs = resolve(ProgrammeAllocationSet(; slv = slv,
                                            sets = UniverseSets(;
                                                                dict = Dict("nx" =>
                                                                                ["A", "B",
                                                                                 "C"])),
                                            lcse = LinearConstraintEstimator(;
                                                                             val = :(A +
                                                                                     B <=
                                                                                     0.8))),
                     3)
        @test isa(rs.lcse, po.LinearConstraint)
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
                                               lcse = LinearConstraintEstimator(;
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
                                               lcse = LinearConstraintEstimator(;
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

    @testset "A penalty weighs against the Bregman divergence in every geometry (#1215)" begin
        # At an interior minimiser of D_Ψ(w, q) + λ‖w‖ on the simplex,
        # ∇Ψ(w) − ∇Ψ(q) + λ w / ‖w‖ is the same in every entry. The Tsallis arm wrote
        # (1 − α) D_Ψ, which gave the penalty 1 / (1 − α) times its weight: at α = 1/2 the
        # spread below was 0.046.
        qp = [0.35, 0.25, 0.4]
        A = [2.0 0.3 0.1; 0.3 1.0 0.2; 0.1 0.2 1.5]
        cases = ((EuclideanProjection(), w -> w, 0.2, 1e-6),
                 (EntropicProjection(), w -> log.(w), 0.2, 1e-6),
                 (TsallisProjection(; alpha = 0.5), w -> w .^ -0.5 ./ -0.5, 0.2, 1e-6),
                 (TsallisProjection(; alpha = 0.3), w -> w .^ -0.7 ./ -0.7, 0.1, 1e-6),
                 (LogBarrierProjection(), w -> -1 ./ w, 0.2, 1e-5),
                 (GramProjection(; slv = slv, A = A), w -> A * w, 0.2, 1e-6))
        for (proj, grad, lam, tol) in cases
            set = resolve(ProgrammeAllocationSet(; slv = slv,
                                                 l2 = L2Regularisation(; val = lam)), 3)
            wp = po.project(proj, set, qp, wh)
            r = grad(wp) .- grad(qp) .+ lam .* wp ./ norm(wp)
            @test maximum(r) - minimum(r) < tol
            @test sum(wp) ≈ 1 atol = 1e-8
        end
    end

    @testset "The turnover ceiling, the covariance cone, the tracking error and the MIP kinds" begin
        # The ceiling is per asset, measured from the Price-Adjusted Allocation.
        tset = resolve(ProgrammeAllocationSet(; slv = slv,
                                              tn = Turnover(; w = zeros(3), val = 0.1)), 3)
        wt = po.project(EuclideanProjection(), tset, [1.0, 0.0, 0.0], wh)
        @test maximum(abs.(wt .- wh)) ≈ 0.1 atol = 1e-6
        @test isapprox(wt, [0.4, 0.3, 0.3]; atol = 1e-6)
        # A ceiling of zero pins the book: the projection is the book itself.
        @test isapprox(po.project(EuclideanProjection(),
                                  resolve(ProgrammeAllocationSet(; slv = slv,
                                                                 tn = Turnover(;
                                                                               w = zeros(3),
                                                                               val = 0.0)),
                                          3), [1.0, 0.0, 0.0], wh), wh; atol = 1e-6)
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
                                                tr = TrackingError(;
                                                                   tr = WeightsTracking(;
                                                                                        w = fill(0.25,
                                                                                                 4),
                                                                                        fixed = true),
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
        # A set prior that cannot price an asset the model trades is refused by name; one
        # that prices every asset, and a set that fits none, pass (ADR 0170).
        Rn = 0.02 .* randn(StableRNG(5), 12, 3)
        Rn[:, 1] .= NaN
        rdn = ReturnsResult(; nx = ["A", "B", "C"], X = Rn)
        prn = prior(EmpiricalPrior(), rdn)
        @test isnothing(po.assert_set_prior_priced(nothing, rdn))
        @test isnothing(po.assert_set_prior_priced(prn, nothing))
        rdn23 = po.port_opt_view(rdn, [2, 3])
        @test isnothing(po.assert_set_prior_priced(prior(EmpiricalPrior(), rdn23), rdn23))
        @test_throws ArgumentError po.assert_set_prior_priced(prn, rdn)
        err = try
            po.assert_set_prior_priced(prn, rdn)
        catch e
            e
        end
        @test occursin("\"A\"", err.msg) && occursin("CoveragePolicy", err.msg)
        @test isnothing(po.prior_investable_mask(nothing))
        @test po.prior_investable_mask(prn) == [false, true, true]
        @test_throws po.IsNothingError po.programme_investable_reduction(trues(3), bset,
                                                                         nothing, prn)
        @test_throws po.IsNothingError po.programme_investable_reduction(trues(3), bset,
                                                                         rdn, nothing)
        @test_throws po.IsNothingError po.resolve_allocation_set_rows(tset, prn, nothing)
    end

    @testset "The Held Step" begin
        # A cap of 0.4 and a ceiling of 0.01 from a book at 0.7 cannot both hold.
        hset = resolve(ProgrammeAllocationSet(; slv = slv,
                                              tn = Turnover(; w = zeros(3), val = 0.01),
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
                   set = ProgrammeAllocationSet(; slv = slv,
                                                tn = Turnover(; w = zeros(3), val = 0.01),
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
        # The Start Allocation meets the set as a given one does: the uniform start under a
        # cap `1/N` breaks is projected, so the allocation held during the first period lies
        # in the set, and a given start on a bounded set is the same projection.
        capped = BoundedAllocationSet(; wb = WeightBounds(0, [0.2, 1, 1, 1]))
        rd4 = ReturnsResult(; nx = ["A", "B", "C", "D"],
                            X = 0.02 .* randn(StableRNG(7), 6, 4),
                            ts = Date(2020, 1, 1) .+ Day.(0:5))
        seed(o, s) = po.online_selection_seed(o, rd4,
                                              po.resolve_allocation_set(s, 4, false,
                                                                        Float64)).w
        @test seed(OPS(; alg = BuyAndHold(), set = capped), capped) ≈
              [0.2, 0.8 / 3, 0.8 / 3, 0.8 / 3]
        @test seed(OPS(; alg = BuyAndHold(), w0 = fill(0.25, 4), set = capped), capped) ≈
              seed(OPS(; alg = BuyAndHold(), set = capped), capped)
        @test seed(OPS(; alg = BuyAndHold()), BoundedAllocationSet()) == fill(0.25, 4)
        # A set that reads the head's rows has nothing to project onto before the first row:
        # a given start is held as given, never refused, and the first update holds the step
        # as every fitted ceiling does.
        fitted = ProgrammeAllocationSet(; slv = slv,
                                        r = Variance(;
                                                     settings = RiskMeasureSettings(;
                                                                                    ub = 4e-4)))
        @test seed(OPS(; alg = PassiveAggressiveMeanReversion(), w0 = [0.7, 0.1, 0.1, 0.1],
                       set = fitted), fitted) == [0.7, 0.1, 0.1, 0.1]
        rf = @test_logs (:warn, r"Held Step at 2020-01-01.*covariance of one observation") match_mode=:any optimise(OPS(;
                                                                                                                        alg = PassiveAggressiveMeanReversion(),
                                                                                                                        w0 = [0.7,
                                                                                                                              0.1,
                                                                                                                              0.1,
                                                                                                                              0.1],
                                                                                                                        set = fitted),
                                                                                                                    po.port_opt_view(rd4,
                                                                                                                                     1:1,
                                                                                                                                     :))
        @test isa(rf.retcode.res, po.HeldStep)
        # A programme the start cannot meet — a book the turnover ceiling cannot bring under
        # the cap — holds the start as given and warns, as a row's Held Step does.
        tset0 = ProgrammeAllocationSet(; slv = slv,
                                       tn = Turnover(; w = zeros(4), val = 0.01),
                                       wb = WeightBounds(; lb = 0, ub = 0.4))
        @test (@test_logs (:warn, r"Held Step at the start") match_mode=:any seed(OPS(;
                                                                                      alg = BuyAndHold(),
                                                                                      w0 = [0.7,
                                                                                            0.2,
                                                                                            0.1,
                                                                                            0.0],
                                                                                      set = tset0),
                                                                                  tset0)) ==
              [0.7, 0.2, 0.1, 0.0]
    end

    @testset "The solver-free roots on raw steps across many binades (#1258)" begin
        cset = resolve(BoundedAllocationSet(; wb = WeightBounds(; lb = 0, ub = 0.4)), 4)
        book = fill(0.25, 4)
        dproj = po.DiagonalProjection([1.0, 2, 3, 4])
        # The kinks of the budget, not a bracket's width, fix the root, so a step whose
        # entries span hundreds of binades is exact.
        for e in (1e-55, 1e-60, 1e-62, 1e-70, 1e-300)
            v = [1, e, e, e]
            @test po.project(EntropicProjection(), cset, v, book) ≈ [0.4, 0.2, 0.2, 0.2]
            @test po.project(EuclideanProjection(), cset, v, book) ≈ [0.4, 0.2, 0.2, 0.2]
        end
        @test po.project(EuclideanProjection(), cset, [1e300, -1e300, 0.3, 0.4], book) ≈
              [0.4, 0.0, 0.25, 0.35]
        @test po.project(dproj, cset, [1e300, -1e300, 0.3, 0.4], book) ≈
              [0.4, 0.0, 0.3 - 0.4 / 7, 0.4 - 0.3 / 7]
        # An infinite bound is no kink: the budget is linear past the last finite one.
        free = resolve(BoundedAllocationSet(;
                                            wb = WeightBounds(; lb = nothing, ub = nothing)),
                       4)
        v = [0.9, 0.05, 0.03, 0.5]
        @test po.project(EuclideanProjection(), free, v, book) ≈ v .- (sum(v) - 1) / 4
        @test po.project(EntropicProjection(),
                         resolve(BoundedAllocationSet(;
                                                      wb = WeightBounds(; lb = 0,
                                                                        ub = nothing)), 4),
                         v, book) ≈ v ./ sum(v)
        @test po.project(EuclideanProjection(),
                         resolve(BoundedAllocationSet(;
                                                      wb = WeightBounds(; lb = -0.2,
                                                                        ub = nothing)), 4),
                         [0.9, 0.05, -1.0, 0.5], book) ≈
              [0.9 - 0.25 / 3, 0.05 - 0.25 / 3, -0.2, 0.5 - 0.25 / 3]
        # A zero stays at its floor under the entropic projection, so caps that cannot reach
        # the budget are refused, as the barrier arms refuse them.
        @test_throws DomainError po.project(EntropicProjection(), cset, [1.0, 0, 0, 0],
                                            book)
        @test_throws DomainError po.project(EntropicProjection(), cset, [1.0, 2, 0, 0],
                                            book)
        # A non-finite raw step has no projection.
        @test_throws DomainError po.project(EntropicProjection(), cset, [Inf, 1, 1, 1],
                                            book)
        @test_throws DomainError po.project(EuclideanProjection(), cset,
                                            [NaN, 0.3, 0.3, 0.4], book)
        @test_throws DomainError po.project(EuclideanProjection(),
                                            resolve(BoundedAllocationSet(), 4),
                                            [NaN, 0.3, 0.3, 0.4], book)
        @test_throws DomainError po.project(dproj, cset, [Inf, 0.3, 0.3, 0.4], book)
        # A barrier base of `1e20` cancels the multiplier that brings it to a share of the
        # budget: the root misses the budget, and the step is held, not returned off it.
        for (proj, v) in ((LogBarrierProjection(), [1, 1e-20, 1e-20, 1e-20]),
                          (LogBarrierProjection(), [1, 1e-70, 1e-70, 1e-70]),
                          (TsallisProjection(; alpha = 0.5), [1, 1e-70, 1e-70, 1e-70]))
            (wheld, held) = po.with_projection_step(() -> po.project(proj, cset, v, book),
                                                    nothing, Date(2020, 1, 1))
            @test wheld == book && !(wheld === book)
            @test length(held) == 1
            @test occursin("did not meet the budget", held[1].reason)
            @test isnothing(held[1].trials)
        end
        @test (@test_logs (:warn, r"Held Step outside") po.project(LogBarrierProjection(),
                                                                   cset,
                                                                   [1, 1e-20, 1e-20, 1e-20],
                                                                   book)) == book
        # The root is exact on an ordinary step, and no step is held.
        (wn, held) = po.with_projection_step(() -> po.project(LogBarrierProjection(), cset,
                                                              [0.5, 0.3, 0.1, 0.1], book),
                                             nothing, Date(2020, 1, 1))
        @test isempty(held) && sum(wn) ≈ 1 && all(0 .<= wn .<= 0.4)
        # The barrier bisection halves a bracket as often as its type has binades and bits.
        @test po.bisection_cap(Float64) == 2151
        @test po.bisection_cap(Float32) == 301
        @test po.bisection_cap(Float16) == 51
        @test po.bisection_cap(Rational{Int}) == 2151
        @test po.bisection_cap(Int) == 2151
    end

    @testset "The mixture's second projection (ADR 0163)" begin
        experts = [ConstantRebalancedPortfolio(; w = [1.0, 0.0]),
                   ConstantRebalancedPortfolio(; w = [0.0, 1.0])]
        x = [1.2, 0.8]
        w = [0.5, 0.5]
        tn = resolve(ProgrammeAllocationSet(; slv = slv,
                                            tn = Turnover(; w = zeros(2), val = 0.1)), 2)
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
        cvar = ConditionalValueatRisk(; settings = RiskMeasureSettings(; ub = 0.02))
        mdd = MaximumDrawdown(; settings = RiskMeasureSettings(; ub = 0.05))
        @test isnothing(po.rows_needed(ProgrammeAllocationSet(; slv = slv, r = cvar)))
        @test po.risk_reads_rows(cvar) && po.risk_reads_rows(Variance())
        @test !po.risk_reads_rows(nothing) && !po.risk_reads_rows(Variance(; sigma = I(3)))
        @test po.risk_reads_rows(Variance(; sigma = I(3),
                                          rc = LinearConstraintEstimator(;
                                                                         val = :(A <= 0.5))))
        @test po.risk_reads_rows([Variance(; sigma = I(3)), cvar])
        # A two-argument view, the read-out's, carries a tail measure unchanged.
        @test po.port_opt_view(ProgrammeAllocationSet(; slv = slv, r = cvar), [1, 3]).r ===
              cvar
        # The set is a Risk Constraint Owner: its solver, and its risk-contribution rows
        # resolved over its sets.
        cset = resolve(ProgrammeAllocationSet(; slv = slv, r = cvar), 4)
        @test isa(cset, po.AbstractProgrammeAllocationSet)
        @test isa(cset, po.RiskConstraintOwner) && isa(cset, po.RiskBoundOwner)
        @test po.risk_constraint_solver(cset) === slv
        @test isnothing(po.risk_contribution_constraints(Variance(), cset,
                                                         prior(EmpiricalPrior(),
                                                               randn(StableRNG(1), 8, 4))))
        # The bound seam is total over the owner: no bound is a no-op on the set as on an
        # optimiser, and a frontier or a per-asset vector is refused by name, not by a
        # missing method, for a programme set that reaches the builder without the
        # constructor's check.
        mb = JuMP.Model()
        JuMP.@variable(mb, tb)
        @test isnothing(po.set_risk_upper_bound!(mb, cset, tb, nothing, :x))
        @test_throws ArgumentError po.set_risk_upper_bound!(mb, cset, tb, Frontier(; N = 3),
                                                            :x)
        @test_throws ArgumentError po.set_risk_upper_bound!(mb, cset, tb, [0.1, 0.2], :x)
        # Inside a step the ceiling binds on the head's rows: the tail of the projected
        # allocation meets the number, and differs from the unconstrained projection.
        rng = StableRNG(19)
        T, N = 60, 4
        R = 0.02 .* randn(rng, T, N)
        R[:, 1] .*= 3
        rdR = ReturnsResult(; nx = ["A", "B", "C", "D"], X = R)
        q4 = [0.9, 0.05, 0.03, 0.02]
        w4 = fill(0.25, 4)
        free = po.project(EuclideanProjection(),
                          resolve(ProgrammeAllocationSet(; slv = slv), 4), q4, w4)
        @test expected_risk(cvar, free, R) > 0.02
        (wc, hc) = po.with_projection_step(() -> po.project(EuclideanProjection(), cset, q4,
                                                            w4), rdR, 1)
        @test isempty(hc)
        @test expected_risk(cvar, wc, R) <= 0.02 * (1 + 1e-4)
        @test expected_risk(cvar, wc, R) >= 0.02 * (1 - 1e-2)
        @test !isapprox(wc, free; atol = 1e-3)
        # A drawdown ceiling, and a variance built on the fitted prior through the same
        # route as the tail measures, which agrees with the variance that holds its matrix.
        mset = resolve(ProgrammeAllocationSet(; slv = slv, r = mdd), 4)
        (wm, hm) = po.with_projection_step(() -> po.project(EuclideanProjection(), mset, q4,
                                                            w4), rdR, 1)
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
                                                           w4), rdR, 1)
        @test isapprox(wf, po.project(EuclideanProjection(), vmat, q4, w4); atol = 1e-5)
        @test dot(wf, S, wf) <= vub * (1 + 1e-4)
        # A matrix ceiling takes the shared builder's factor. A cash leg makes the matrix
        # singular at its first pivot, where a Cholesky factor without a check is wrong and
        # the ceiling did not bind: the answer was the free projection at four times the
        # ceiling.
        S0 = zero(S)
        S0[2:4, 2:4] = S[2:4, 2:4]
        vub0 = 0.25 * dot(free, S0, free)
        w0v = po.project(EuclideanProjection(),
                         resolve(ProgrammeAllocationSet(; slv = slv,
                                                        r = Variance(; sigma = S0,
                                                                     settings = RiskMeasureSettings(;
                                                                                                    ub = vub0))),
                                 4), q4, w4)
        w0s = po.project(EuclideanProjection(),
                         resolve(ProgrammeAllocationSet(; slv = slv,
                                                        r = StandardDeviation(; sigma = S0,
                                                                              settings = RiskMeasureSettings(;
                                                                                                             ub = sqrt(vub0)))),
                                 4), q4, w4)
        @test isapprox(dot(w0v, S0, w0v), vub0; rtol = 1e-6)
        @test isapprox(w0v, w0s; atol = 1e-8)
        # A stated factor is the one the cone reads, and an indefinite matrix is refused by
        # the factor, as the shared builder refuses it.
        G = Matrix(qr(randn(StableRNG(3), 4, 4)).Q) * po.covariance_factor(S)
        wch = po.project(EuclideanProjection(),
                         resolve(ProgrammeAllocationSet(; slv = slv,
                                                        r = Variance(; sigma = S, chol = G,
                                                                     settings = RiskMeasureSettings(;
                                                                                                    ub = vub))),
                                 4), q4, w4)
        @test isapprox(wch, wf; atol = 1e-5)
        Sind = [1.0 0.9 0.0 0.0; 0.9 1.0 0.9 0.0; 0.0 0.9 1.0 0.0; 0.0 0.0 0.0 1.0] .* 1e-4
        @test_throws PosDefException po.project(EuclideanProjection(),
                                                resolve(ProgrammeAllocationSet(; slv = slv,
                                                                               r = Variance(;
                                                                                            sigma = Sind,
                                                                                            settings = RiskMeasureSettings(;
                                                                                                                           ub = 1e-5))),
                                                        4), q4, w4)
        # The refusals of a ceiling that is not one number.
        @test isnothing(po.assert_risk_ceiling(nothing))
        @test_throws ArgumentError po.assert_risk_ceiling(ConditionalValueatRisk())
        @test_throws IsEmptyError po.assert_risk_ceiling(po.RiskMeasure[])
        # The clip raises a negative entry to zero in the element type it is given.
        @test po.clip_at_zero([-1e-12, 0.5, 0.5]) == [0.0, 0.5, 0.5]
        @test po.clip_at_zero([-1 // 10, 11 // 10]) == [0 // 1, 11 // 10]
        @test eltype(po.clip_at_zero(Float32[-1.0f-7, 1])) === Float32
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
                                           rdR)
        @test haskey(m, :cvar_risk_1_ub) && !haskey(m, :risk_vec)
        # `scale` weights a measure only in the objective, so it does not move a ceiling.
        cvar5 = ConditionalValueatRisk(;
                                       settings = RiskMeasureSettings(; ub = 0.02,
                                                                      scale = 5))
        (wc5, _) = po.with_projection_step(() -> po.project(EuclideanProjection(),
                                                            resolve(ProgrammeAllocationSet(;
                                                                                           slv = slv,
                                                                                           r = cvar5),
                                                                    4), q4, w4), rdR, 1)
        @test isapprox(wc5, wc; atol = 1e-8)
        # A measure that needs a quantity the prior does not carry is refused by name.
        kset = resolve(ProgrammeAllocationSet(; slv = slv,
                                              r = Kurtosis(;
                                                           settings = RiskMeasureSettings(;
                                                                                          ub = 1e-3))),
                       4)
        @test_throws ArgumentError po.with_projection_step(() -> po.project(EuclideanProjection(),
                                                                            kset, q4, w4),
                                                           rdR, 1)
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
    @testset "The optimiser's vocabulary on the set (#1206)" begin
        # Every constraint kind of `JuMPOptimiser` under the optimiser's names and bounds,
        # the direct objective penalties, the semidefinite kinds, and the builder order of
        # `assemble_jump_model!` on both arms.
        sets3 = UniverseSets(;
                             dict = Dict("nx" => ["A", "B", "C"],
                                         "nf" => ["MTUM", "VLUE", "QUAL"]))
        tn3(v; fixed = false) = Turnover(; w = zeros(3), val = v, fixed = fixed)
        ub_of(ub) = RiskMeasureSettings(; ub = ub)
        cvar = ConditionalValueatRisk(; settings = ub_of(0.02))
        mdd = MaximumDrawdown(; settings = ub_of(0.05))
        rng = StableRNG(23)
        T = 40
        R = 0.02 .* randn(rng, T, 3)
        R[:, 1] .*= 3
        rdR = ReturnsResult(; nx = ["A", "B", "C"], X = R)
        q3 = [0.8, 0.15, 0.05]
        w3 = fill(1 / 3, 3)
        inside(f) = first(po.with_projection_step(f, rdR, 1))
        proj(set; q = q3, w = w3) = inside(() -> po.project(EuclideanProjection(), set, q,
                                                            w))
        free = proj(pset)
        @testset "Construction: the names, the bounds and the refusals" begin
            # A bare number is not a turnover, as it is not on the optimiser.
            @test_throws TypeError ProgrammeAllocationSet(; slv = slv, tn = 0.1)
            @test_throws MethodError ProgrammeAllocationSet(; slv = slv, lcs = nothing)
            @test_throws MethodError ProgrammeAllocationSet(; slv = slv, gcard = nothing)
            # Every new slot takes the optimiser's bound.
            full = ProgrammeAllocationSet(; slv = slv, sets = sets3,
                                          tn = [tn3(0.1), tn3([0.1, 0.2, 0.3])],
                                          tr = [TrackingError(;
                                                              tr = ReturnsTracking(;
                                                                                   w = zeros(T)),
                                                              err = 0.01),
                                                TrackingError(;
                                                              tr = WeightsTracking(;
                                                                                   w = w3),
                                                              err = 0.01)], r = [cvar, mdd],
                                          gbgt = BudgetRange(; lb = nothing, ub = 1.4),
                                          wb = WeightBounds(; lb = -0.2, ub = 1), scard = 1,
                                          smtx = [1.0 0.0 0.0; 0.0 1.0 1.0], l2c = 0.9,
                                          linfc = 0.9, l1 = 0.1,
                                          l2 = L2Regularisation(; val = 0.1),
                                          cte = CentralityConstraint(),
                                          ple = IntegerPhylogenyEstimator(;
                                                                          pl = NetworkEstimator(),
                                                                          B = 1),
                                          ret = ArithmeticReturn(;
                                                                 settings = JuMPReturnsSettings(;
                                                                                                lb = 0.0)))
            @test fieldnames(typeof(full))[1:4] == (:pe, :slv, :r, :wb)
            @test isnothing(po.rows_needed(full))
            @test po.rows_needed(ProgrammeAllocationSet(; slv = slv, sbgt = 0.2,
                                                        wb = WeightBounds(; lb = -0.2,
                                                                          ub = 1),
                                                        l2c = 0.9, l1 = 0.1)) == 0
            # A precomputed centrality or phylogeny row reads no rows; an estimator does.
            @test po.rows_needed(ProgrammeAllocationSet(; slv = slv,
                                                        ple = po.IntegerPhylogeny(;
                                                                                  A = [0 1 0;
                                                                                       1 0 0;
                                                                                       0 0 0],
                                                                                  B = 1))) ==
                  0
            @test isnothing(po.rows_needed(ProgrammeAllocationSet(; slv = slv,
                                                                  cte = CentralityConstraint())))
            # A Calibration Rule in a penalty reads the prior, so it reads the rows.
            @test isnothing(po.rows_needed(ProgrammeAllocationSet(; slv = slv,
                                                                  l1 = RateRadius())))
            # The name-keyed guard covers the turnover and the selection matrices.
            @test_throws po.IsNothingError ProgrammeAllocationSet(; slv = slv,
                                                                  tn = TurnoverEstimator(;
                                                                                         w = zeros(3),
                                                                                         val = 0.1))
            @test_throws po.IsNothingError ProgrammeAllocationSet(; slv = slv, scard = 1,
                                                                  smtx = AssetSetsMatrixEstimator(;
                                                                                                  val = "nf"))
            # The sub-group shape checks are the optimiser's: the sub-grouped matrix is
            # checked against `sgmtx`, the matrix its builder pairs it with (a defect the
            # optimiser's constructor carried, which read `smtx` there).
            two = linear_constraints(LinearConstraintEstimator(;
                                                               val = [:(A <= 1), :(B <= 1)]),
                                     sets3)
            @test isa(ProgrammeAllocationSet(; slv = slv, sgcarde = two,
                                             sgmtx = [1.0 0.0 0.0; 0.0 1.0 1.0]),
                      ProgrammeAllocationSet)
            @test_throws DimensionMismatch ProgrammeAllocationSet(; slv = slv,
                                                                  sgcarde = two,
                                                                  smtx = [1.0 0.0 0.0;
                                                                          0.0 1.0 1.0],
                                                                  sgmtx = Matrix{Float64}(I,
                                                                                          3,
                                                                                          3))
            @test_throws DimensionMismatch JuMPOptimiser(; slv = slv, sgcarde = two,
                                                         smtx = [1.0 0.0 0.0; 0.0 1.0 1.0],
                                                         sgmtx = Matrix{Float64}(I, 3, 3))
            @test_throws ArgumentError ProgrammeAllocationSet(; slv = slv, scard = 1)
            @test_throws po.IsEmptyError ProgrammeAllocationSet(; slv = slv,
                                                                r = RiskMeasure[])
            @test_throws po.IsEmptyError ProgrammeAllocationSet(; slv = slv,
                                                                tn = Turnover[])
            @test_throws po.IsEmptyError ProgrammeAllocationSet(; slv = Solver[])
            @test_throws Exception ProgrammeAllocationSet(; slv = slv, gbgt = -1,
                                                          wb = WeightBounds(; lb = -1,
                                                                            ub = 1))
            # The sub-group and the sub-grouped slots are viewed and resolved apart when
            # they differ, and once when they are the same object.
            apart = ProgrammeAllocationSet(; slv = slv, sets = sets3, scard = 1,
                                           smtx = [1.0 0.0 0.0; 0.0 1.0 1.0], sgcarde = two,
                                           sgmtx = [1.0 1.0 0.0; 0.0 0.0 1.0],
                                           slt = Threshold(0.05), sglt = Threshold(0.02),
                                           sst = Threshold(0.01), sgst = Threshold(0.03))
            va = po.port_opt_view(apart, [1, 3])
            @test va.smtx == [1.0 0.0; 0.0 1.0] && va.sglt.val == 0.02 && va.sst.val == 0.01
            ra = resolve(apart, 3)
            @test ra.sgmtx == [1.0 1.0 0.0; 0.0 0.0 1.0] &&
                  ra.smtx == [1.0 0.0 0.0; 0.0 1.0 1.0]
            @test ra.slt.val == 0.05 && ra.sglt.val == 0.02 && ra.sgst.val == 0.03
            @test_throws Exception ProgrammeAllocationSet(; slv = slv, l2c = -1)
            # The gross budget is refused where the bounds forbid shorts, as on the
            # optimiser.
            @test_throws Exception ProgrammeAllocationSet(; slv = slv,
                                                          gbgt = BudgetRange(; lb = nothing,
                                                                             ub = 1.4))
            # A view slices every per-asset slot and carries the rest.
            v = po.port_opt_view(ProgrammeAllocationSet(; slv = slv,
                                                        tn = tn3([0.1, 0.2, 0.3]), card = 2,
                                                        l2c = 0.9), [1, 3])
            @test v.tn.val == [0.1, 0.3] && v.card == 2 && v.l2c == 0.9
            # Resolution turns every name-keyed slot into a value, the turnover included.
            rs = resolve(ProgrammeAllocationSet(; slv = slv, sets = sets3,
                                                tn = TurnoverEstimator(; w = zeros(3),
                                                                       val = ["A" => 0.1],
                                                                       dval = 0.5),
                                                scard = 1,
                                                smtx = AssetSetsMatrixEstimator(;
                                                                                val = "nf"),
                                                lcse = LinearConstraintEstimator(;
                                                                                 val = :(A +
                                                                                         B <=
                                                                                         0.8))),
                         3)
            @test isa(rs.tn, Turnover) && rs.tn.val == [0.1, 0.5, 0.5]
            @test isa(rs.smtx, AbstractMatrix) && isa(rs.lcse, po.LinearConstraint)
        end
        @testset "The turnover and the tracking kinds" begin
            # A turnover object: the reference is the step's book unless fixed.
            @test isapprox(proj(resolve(ProgrammeAllocationSet(; slv = slv, tn = tn3(0.1)),
                                        3); q = [1.0, 0.0, 0.0], w = wh), [0.4, 0.3, 0.3];
                           atol = 1e-6)
            fixed = resolve(ProgrammeAllocationSet(; slv = slv,
                                                   tn = Turnover(; w = [0.6, 0.2, 0.2],
                                                                 val = 0.05, fixed = true)),
                            3)
            wfx = proj(fixed; q = [1.0, 0.0, 0.0], w = wh)
            @test maximum(abs.(wfx .- [0.6, 0.2, 0.2])) <= 0.05 + 1e-6
            @test maximum(abs.(wfx .- wh)) > 0.05
            # A vector of turnovers: every ceiling binds.
            two = resolve(ProgrammeAllocationSet(; slv = slv,
                                                 tn = [tn3([0.1, 1.0, 1.0]),
                                                       tn3([1.0, 0.05, 1.0])]), 3)
            wtwo = proj(two; q = [1.0, 0.0, 0.0], w = wh)
            @test abs(wtwo[1] - wh[1]) <= 0.1 + 1e-6 && abs(wtwo[2] - wh[2]) <= 0.05 + 1e-6
            # A returns benchmark: the tracking error over the head's rows.
            bench = R * [0.2, 0.3, 0.5]
            rtr = resolve(ProgrammeAllocationSet(; slv = slv,
                                                 tr = TrackingError(;
                                                                    tr = ReturnsTracking(;
                                                                                         w = bench),
                                                                    err = 0.002)), 3)
            wr = proj(rtr)
            @test sqrt(sum(abs2, R * wr - bench) / (T - 1)) <= 0.002 * (1 + 1e-6)
            @test !isapprox(wr, free; atol = 1e-4)
            # A weights benchmark that is not fixed tracks the step's book; a fixed one
            # tracks the caller's.
            wt_fixed = proj(resolve(ProgrammeAllocationSet(; slv = slv,
                                                           tr = TrackingError(;
                                                                              tr = WeightsTracking(;
                                                                                                   w = [0.2,
                                                                                                        0.3,
                                                                                                        0.5],
                                                                                                   fixed = true),
                                                                              err = 0.002)),
                                    3))
            @test isapprox(wt_fixed, wr; atol = 1e-5)
            wt_book = proj(resolve(ProgrammeAllocationSet(; slv = slv,
                                                          tr = TrackingError(;
                                                                             tr = WeightsTracking(;
                                                                                                  w = [0.2,
                                                                                                       0.3,
                                                                                                       0.5]),
                                                                             err = 0.002)),
                                   3))
            @test sqrt(sum(abs2, R * (wt_book - w3)) / (T - 1)) <= 0.002 * (1 + 1e-6)
            @test !isapprox(wt_book, wt_fixed; atol = 1e-4)
            # A vector of tracking errors, and a risk tracking error with the set as the
            # owner of its inner builder.
            wv = proj(resolve(ProgrammeAllocationSet(; slv = slv,
                                                     tr = [TrackingError(;
                                                                         tr = ReturnsTracking(;
                                                                                              w = bench),
                                                                         err = 0.002),
                                                           TrackingError(;
                                                                         tr = WeightsTracking(;
                                                                                              w = [0.2,
                                                                                                   0.3,
                                                                                                   0.5],
                                                                                              fixed = true),
                                                                         err = 0.003,
                                                                         alg = LInfNorm())]),
                              3))
            @test sqrt(sum(abs2, R * wv - bench) / (T - 1)) <= 0.002 * (1 + 1e-6)
            @test maximum(abs, R * (wv - [0.2, 0.3, 0.5])) <= 0.003 * (1 + 1e-6)
            rte = RiskTrackingError(; r = Variance(), tr = WeightsTracking(; w = w3),
                                    err = 1e-5)
            wrt = proj(resolve(ProgrammeAllocationSet(; slv = slv, tr = rte), 3))
            @test dot(wrt - w3, cov(R), wrt - w3) <= 1e-5 * (1 + 1e-4)
            @test !isapprox(wrt, free; atol = 1e-3)
        end
        @testset "The ceilings, the budgets, the norm ceilings and the return floor" begin
            # A vector of ceilings: every one binds.
            wc = proj(resolve(ProgrammeAllocationSet(; slv = slv, r = [cvar, mdd]), 3))
            @test expected_risk(cvar, wc, R) <= 0.02 * (1 + 1e-4)
            @test expected_risk(mdd, wc, R) <= 0.05 * (1 + 1e-4)
            # Two matrix ceilings on a set that reads no rows, each its own cone.
            S = cov(R)
            wm = po.project(EuclideanProjection(),
                            resolve(ProgrammeAllocationSet(; slv = slv,
                                                           r = [Variance(; sigma = S,
                                                                         settings = ub_of(0.5 *
                                                                                          dot(free,
                                                                                              S,
                                                                                              free))),
                                                                StandardDeviation(;
                                                                                  sigma = S,
                                                                                  settings = ub_of(0.9 *
                                                                                                   sqrt(dot(free,
                                                                                                            S,
                                                                                                            free))))]),
                                    3), q3, w3)
            @test dot(wm, S, wm) <= 0.5 * dot(free, S, free) * (1 + 1e-4)
            # The short and gross budgets under a negative bound.
            ls = resolve(ProgrammeAllocationSet(; slv = slv,
                                                wb = WeightBounds(; lb = -1, ub = 1),
                                                gbgt = BudgetRange(; lb = nothing,
                                                                   ub = 1.2)), 3)
            wls = proj(ls; q = [1.2, 0.3, -0.5])
            @test sum(abs, wls) <= 1.2 * (1 + 1e-6)
            @test sum(wls) ≈ 1 atol = 1e-8
            wsb = proj(resolve(ProgrammeAllocationSet(; slv = slv,
                                                      wb = WeightBounds(; lb = -1, ub = 1),
                                                      sbgt = BudgetRange(; lb = nothing,
                                                                         ub = 0.05)), 3);
                       q = [1.2, 0.3, -0.5])
            @test sum(x -> max(-x, 0), wsb) <= 0.05 * (1 + 1e-6)
            # The norm ceilings: a cap on the largest weight, on the 2-norm and on a p-norm.
            @test maximum(proj(resolve(ProgrammeAllocationSet(; slv = slv, linfc = 0.5), 3))) <=
                  0.5 * (1 + 1e-6)
            @test norm(proj(resolve(ProgrammeAllocationSet(; slv = slv, l2c = 0.7), 3))) <=
                  0.7 * (1 + 1e-6)
            @test norm(proj(resolve(ProgrammeAllocationSet(; slv = slv,
                                                           lpc = LpRegularisation(; p = 3,
                                                                                  val = 0.6)),
                                    3)), 3) <= 0.6 * (1 + 1e-6)
            # The return floor: the prior's expected return of the answer meets the number.
            mu = vec(mean(R; dims = 1))
            lb = 0.5 * (maximum(mu) + dot(free, mu))
            wret = proj(resolve(ProgrammeAllocationSet(; slv = slv,
                                                       ret = ArithmeticReturn(;
                                                                              settings = JuMPReturnsSettings(;
                                                                                                             lb = lb))),
                                3))
            @test dot(wret, mu) >= lb * (1 - 1e-6)
            @test !isapprox(wret, free; atol = 1e-4)
        end
        @testset "The penalties, the custom terms and the phylogeny kinds" begin
            # An l2 penalty pulls toward uniform; an l1 penalty is a constant on the
            # long-only simplex.
            wl2 = proj(resolve(ProgrammeAllocationSet(; slv = slv,
                                                      l2 = L2Regularisation(; val = 0.5)),
                               3))
            @test maximum(wl2) < maximum(free) && minimum(wl2) > minimum(free)
            @test sum(wl2) ≈ 1 atol = 1e-8
            @test isapprox(proj(resolve(ProgrammeAllocationSet(; slv = slv, l1 = 0.3), 3)),
                           free; atol = 1e-5)
            @test !isapprox(proj(resolve(ProgrammeAllocationSet(; slv = slv,
                                                                lp = LpRegularisation(;
                                                                                      p = 3,
                                                                                      val = 0.5)),
                                         3)), free; atol = 1e-4)
            @test !isapprox(proj(resolve(ProgrammeAllocationSet(; slv = slv, linf = 0.5),
                                         3)), free; atol = 1e-4)
            # A custom constraint and a custom objective term, the set as their owner.
            wcc = proj(resolve(ProgrammeAllocationSet(; slv = slv, ccnt = FirstAtMost(0.1)),
                               3))
            @test wcc[1] <= 0.1 + 1e-6
            wco = proj(resolve(ProgrammeAllocationSet(; slv = slv, cobj = FirstCosts(0.2)),
                               3))
            @test wco[1] < free[1] && sum(wco) ≈ 1
            # Centrality rows fitted on the rows.
            wce = proj(resolve(ProgrammeAllocationSet(; slv = slv,
                                                      cte = CentralityConstraint(; B = 0.3,
                                                                                 comp = <=)),
                               3))
            @test sum(wce) ≈ 1 atol = 1e-8
            # The integer phylogeny: at most one asset per linked pair.
            wip = inside(() -> po.project(EuclideanProjection(),
                                          resolve(ProgrammeAllocationSet(; slv = mip,
                                                                         ple = IntegerPhylogenyEstimator(;
                                                                                                         pl = NetworkEstimator(),
                                                                                                         B = 1)),
                                                  3), q3, w3))
            @test count(x -> x > 1e-6, wip) <= 2
            # The semidefinite phylogeny: the lifted `W` and its rows on the projection,
            # pinned by the penalty, and the variance ceiling under it takes the
            # semidefinite form.
            sdp = SemiDefinitePhylogenyEstimator(; pl = ClustersEstimator(), p = 10.0)
            m = JuMP.Model()
            po.set_model_scales!(m, 1, 1)
            po.set_model_observations!(m, T)
            JuMP.@expression(m, k, 1)
            JuMP.@variable(m, w[1:3])
            inside(() -> po.set_allocation_set_constraints!(m,
                                                            resolve(ProgrammeAllocationSet(;
                                                                                           slv = slv,
                                                                                           ple = sdp),
                                                                    3), w3, rdR))
            @test haskey(m, :W) && haskey(m, :sdp_plg_1) && haskey(m, :op)
            # With every pair linked `W` is diagonal and `p · tr(W) = p (Σ|w|)²`, a constant
            # on the long-only simplex as `‖w‖₁` is: the step is the free one. A phylogeny
            # that links one pair pins `W₁₂` alone, and the penalty then prices holding
            # both, so the smaller leg shrinks.
            wsdp = proj(resolve(ProgrammeAllocationSet(; slv = slv, ple = sdp), 3))
            @test isapprox(wsdp, free; atol = 1e-5)
            wpair = proj(resolve(ProgrammeAllocationSet(; slv = slv,
                                                        ple = po.SemiDefinitePhylogeny(;
                                                                                       A = [0 1 0;
                                                                                            1 0 0;
                                                                                            0 0 0],
                                                                                       p = 10.0)),
                                 3))
            @test sum(wpair) ≈ 1 atol = 1e-6
            @test wpair[2] < free[2] - 1e-3
            m2 = JuMP.Model()
            po.set_model_scales!(m2, 1, 1)
            po.set_model_observations!(m2, 0)
            JuMP.@expression(m2, k, 1)
            JuMP.@variable(m2, w[1:3])
            po.set_allocation_set_constraints!(m2,
                                               resolve(ProgrammeAllocationSet(; slv = slv,
                                                                              r = Variance(;
                                                                                           sigma = cov(R),
                                                                                           settings = ub_of(1e-3)),
                                                                              ple = po.SemiDefinitePhylogeny(;
                                                                                                             A = [0 1 0;
                                                                                                                  1 0 0;
                                                                                                                  0 0 0],
                                                                                                             p = 10.0)),
                                                       3), w3, nothing)
            # A matrix ceiling goes through the shared builder with no prior: the semidefinite
            # rows on the one `W`, and the variance marks `variance_flag`, so the phylogeny
            # adds no penalty, as in a head (#1303).
            @test haskey(m2, :W) &&
                  haskey(m2, :variance_risk_1_ub) &&
                  haskey(m2, :sdp_plg_1) &&
                  haskey(m2, :variance_flag)
            @test !haskey(m2, :dev_1) && !haskey(m2, :sdp_plg_p_1)
            # The source of the matrix does not change the formulation: the ceiling that
            # holds its matrix and the ceiling fitted on the prior give one projection, with
            # the phylogeny and without it (#1303).
            S3 = po.prior(EmpiricalPrior(), rdR).sigma
            vub3 = 0.3 * dot(free, S3, free)
            pair = po.SemiDefinitePhylogeny(; A = [0 1 0; 1 0 0; 0 0 0], p = 10.0)
            ceil3(; kw...) = Variance(; settings = ub_of(vub3), kw...)
            mk3(r, ple) = resolve(ProgrammeAllocationSet(; slv = slv, r = r, ple = ple), 3)
            wsd = proj(mk3(ceil3(; sigma = S3), pair))
            @test isapprox(wsd, proj(mk3(ceil3(), pair)); atol = 1e-6)
            # The semidefinite row bounds tr(ΣW), which lies above the variance, so the
            # variance of the answer does not exceed the ceiling.
            @test dot(wsd, S3, wsd) <= vub3 * (1 + 1e-6)
            wsoc = proj(mk3(ceil3(; sigma = S3), nothing))
            @test isapprox(wsoc, proj(mk3(ceil3(), nothing)); atol = 1e-6)
            @test isapprox(dot(wsoc, S3, wsoc), vub3; rtol = 1e-6)
            # Without the phylogeny both routes write the shared builder's cone.
            m3 = JuMP.Model()
            po.set_model_scales!(m3, 1, 1)
            po.set_model_observations!(m3, 0)
            JuMP.@expression(m3, k, 1)
            JuMP.@variable(m3, w[1:3])
            po.set_allocation_set_constraints!(m3, mk3(ceil3(; sigma = S3), nothing), w3,
                                               nothing)
            @test haskey(m3, :dev_1) && haskey(m3, :cdev_soc_1) && haskey(m3, :dev_1_ub)
            @test !haskey(m3, :W)
            # A matrix ceiling after a variance with risk-contribution rows takes the
            # semidefinite form on the same `W`, as the second variance of a head does.
            rc3 = Variance(; rc = LinearConstraintEstimator(; val = :(A <= 0.5)),
                           settings = ub_of(vub3))
            m4 = JuMP.Model()
            po.set_model_scales!(m4, 1, 1)
            po.set_model_observations!(m4, T)
            JuMP.@expression(m4, k, 1)
            JuMP.@variable(m4, w[1:3])
            inside(() -> po.set_allocation_set_constraints!(m4,
                                                            resolve(ProgrammeAllocationSet(;
                                                                                           slv = slv,
                                                                                           sets = sets3,
                                                                                           r = [rc3,
                                                                                                ceil3(;
                                                                                                      sigma = S3)]),
                                                                    3), w3, rdR))
            @test haskey(m4, :rc_variance) && haskey(m4, :variance_risk_2_ub)
            @test !haskey(m4, :dev_2)
            # In a leader's model both routes build the ceiling on the leader's `W` and mark
            # the leader's `variance_flag`; only the set's rows take the prefix.
            for r in (ceil3(; sigma = S3), ceil3())
                ml = JuMP.Model()
                po.set_model_scales!(ml, 1, 1)
                po.set_model_observations!(ml, T)
                JuMP.@expression(ml, k, 1)
                JuMP.@variable(ml, w[1:3])
                inside(() -> po.add_allocation_set_constraints!(ml, mk3(r, pair), w3, rdR))
                @test haskey(ml, :W) && !haskey(ml, :aset_W)
                @test haskey(ml, :variance_flag) && !haskey(ml, :aset_variance_flag)
                @test haskey(ml, :aset_variance_risk_1_ub) && haskey(ml, :aset_sdp_plg_1)
                @test !haskey(ml, :aset_sdp_plg_p_1)
                @test po.weights_prefix(ml, :aset_) === Symbol("")
            end
            # A prefix without the set's mark owns itself, as a tracking build's does, even
            # when it holds the model's own `w`.
            mt = JuMP.Model()
            JuMP.@variable(mt, w[1:3])
            mt[:tr_w] = w
            @test po.weights_prefix(mt, :tr_) === :tr_
            @test po.weights_prefix(mt, Symbol("")) === Symbol("")
            po.mark_state!(mt, :tr_, :w_shared)
            @test po.weights_prefix(mt, :tr_) === Symbol("")
            # A Variance with risk-contribution rows reads the rows and builds through the
            # shared semidefinite builder.
            rcv = Variance(; rc = LinearConstraintEstimator(; val = :(A <= 0.5)),
                           settings = ub_of(1e-3))
            @test isnothing(po.rows_needed(ProgrammeAllocationSet(; slv = slv, r = rcv,
                                                                  sets = sets3)))
            wrc = proj(resolve(ProgrammeAllocationSet(; slv = slv, r = rcv, sets = sets3),
                               3))
            @test sum(wrc) ≈ 1 atol = 1e-6
            # An exposure row through pinned loadings: the rows carry no factor returns.
            M = [1.0 0.0 0.2; 0.5 0.5 0.0; 0.0 1.0 0.7]
            ece = ExposureConstraintEstimator(;
                                              lce = LinearConstraintEstimator(;
                                                                              val = "MTUM <= 0.3"),
                                              space = FactorSpace(;
                                                                  re = Regression(; M = M)))
            eset = ProgrammeAllocationSet(; slv = slv, sets = sets3, lcse = ece)
            @test isnothing(po.rows_needed(eset))
            @test po.exposure_keyed(eset.lcse) && !po.exposure_keyed(nothing)
            wex = proj(resolve(eset, 3))
            @test dot(M[:, 1], wex) <= 0.3 + 1e-6
            @test !isapprox(wex, free; atol = 1e-4)
        end
        @testset "The order and the leader's arm" begin
            # The bare model names its entries as a head's model does, and the builders
            # ran in the head's order: the tracking term precedes the risk ceiling.
            m = JuMP.Model()
            po.set_model_scales!(m, 1, 1)
            po.set_model_observations!(m, T)
            JuMP.@expression(m, k, 1)
            JuMP.@variable(m, w[1:3])
            inside(() -> po.set_allocation_set_constraints!(m,
                                                            resolve(ProgrammeAllocationSet(;
                                                                                           slv = slv,
                                                                                           tn = tn3(0.1),
                                                                                           tr = TrackingError(;
                                                                                                              tr = WeightsTracking(;
                                                                                                                                   w = w3),
                                                                                                              err = 0.01),
                                                                                           r = cvar,
                                                                                           l2c = 0.9,
                                                                                           l1 = 0.1),
                                                                    3), w3, rdR))
            @test haskey(m, :tn_1) && haskey(m, :t_tr_1) && haskey(m, :cvar_risk_1_ub)
            @test haskey(m, :t_l1) && haskey(m, :op)
            # Variables are indexed in the order they were created: the tracking term's
            # precedes the ceiling's.
            vidx(v::JuMP.VariableRef) = JuMP.index(v).value
            vidx(v::AbstractArray) = vidx(first(v))
            @test vidx(m[:t_tn_1]) < vidx(m[:t_tr_1]) < vidx(m[:t_l1]) < vidx(m[:var_1])
            # The leader's arm writes the same kinds beside the head's own, under the
            # `:aset_` prefix where a builder takes one and at free indices otherwise.
            rd = ReturnsResult(; nx = ["A", "B", "C"], X = R,
                               ts = Date(2020, 1, 1) .+ Day.(0:(T - 1)))
            lead = FollowTheLeader(;
                                   opt = MeanRisk(; obj = MaximumReturn(),
                                                  opt = JuMPOptimiser(;
                                                                      pe = EmpiricalPrior(),
                                                                      slv = slv,
                                                                      tn = tn3(0.5),
                                                                      ret = LogarithmicReturn())))
            aset = ProgrammeAllocationSet(; slv = slv, tn = tn3(0.5),
                                          tr = TrackingError(;
                                                             tr = ReturnsTracking(;
                                                                                  w = R *
                                                                                      [0.2,
                                                                                       0.3,
                                                                                       0.5]),
                                                             err = 0.003),
                                          r = ConditionalValueatRisk(;
                                                                     settings = ub_of(0.03)),
                                          wb = WeightBounds(; lb = -0.2, ub = 1),
                                          gbgt = BudgetRange(; lb = nothing, ub = 1.2),
                                          l2 = L2Regularisation(; val = 0.1),
                                          ple = SemiDefinitePhylogenyEstimator(;
                                                                               pl = ClustersEstimator(),
                                                                               p = 10.0))
            lres = optimise(OPS(; alg = lead, set = aset), rd)
            @test isa(lres.retcode, OptimisationSuccess)
            @test isnothing(lres.retcode.res)
            @test expected_risk(cvar, lres.w, R) <= 0.03 * (1 + 1e-4)
            @test sum(abs, lres.w) <= 1.2 * (1 + 1e-6)
            @test sqrt(sum(abs2, R * lres.w - R * [0.2, 0.3, 0.5]) / (T - 1)) <=
                  0.003 * (1 + 1e-4)
            # A benchmark series over the fold is cut to the rows folded so far.
            cut = po.prefix_tracking_benchmark(TrackingError(;
                                                             tr = ReturnsTracking(;
                                                                                  w = collect(1.0:5.0)),
                                                             err = 0.1), 3)
            @test cut.tr.w == [1.0, 2.0, 3.0] && cut.err == 0.1
            @test po.prefix_tracking_benchmark(nothing, 3) === nothing
            @test !isapprox(lres.w,
                            optimise(OPS(; alg = lead,
                                         set = ProgrammeAllocationSet(; slv = slv)), rd).w;
                            atol = 1e-3)
            # A leader that builds its own long and short parts: the set's budgets are
            # written over them, a floored range included.
            lsopt = MeanRisk(; obj = MaximumReturn(),
                             opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv,
                                                 wb = WeightBounds(; lb = -1, ub = 1),
                                                 sbgt = BudgetRange(; lb = nothing,
                                                                    ub = 0.5),
                                                 ret = LogarithmicReturn()))
            lsres = optimise(OPS(; alg = FollowTheLeader(; opt = lsopt),
                                 set = ProgrammeAllocationSet(; slv = slv,
                                                              wb = WeightBounds(; lb = -1,
                                                                                ub = 1),
                                                              sbgt = BudgetRange(;
                                                                                 lb = 0.05,
                                                                                 ub = 0.2))),
                             rd)
            @test isa(lsres.retcode, OptimisationSuccess) && isnothing(lsres.retcode.res)
            short = sum(x -> max(-x, 0), lsres.w)
            @test 0.05 * (1 - 1e-4) <= short <= 0.2 * (1 + 1e-4)
        end
    end
    @testset "Docs and the search seam" begin
        @test occursin("Held Step", string(@doc(po.HeldStep)))
        @test occursin("per-asset", string(@doc(ProgrammeAllocationSet)))
        @test occursin("before", string(@doc(NewtonStep)))
        @test occursin("K + 1", string(@doc(ExpertMixture)))
        io = IOBuffer()
        show(io, MIME"text/plain"(),
             ProgrammeAllocationSet(; slv = slv, tn = Turnover(; w = zeros(3), val = 0.1)))
        @test occursin("ProgrammeAllocationSet", String(take!(io)))
    end
end
