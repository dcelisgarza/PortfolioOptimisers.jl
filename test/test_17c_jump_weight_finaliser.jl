using Clarabel, HiGHS, JuMP, Statistics

#=
The weight finaliser that projects the weights of an optimiser onto the feasible set of a
Constraint Block (#2257). The oracle of each testset is the rows of the block on the finalised
book, a programme written by hand, or the book of a JuMP head under the same block.
=#

const PO = PortfolioOptimisers

hslv = Solver(; name = :highs, solver = HiGHS.Optimizer,
              settings = "log_to_console" => false)
cslv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false),
              check_sol = (; allow_local = true, allow_almost = true))
bad = Solver(; solver = () -> error("no solver"))
N = 10
nx = ["a$i" for i in 1:N]
X = 0.01 .* randn(StableRNG(123), 160, N) .+ 0.0005
rd = ReturnsResult(; nx = nx, X = X)
sets = UniverseSets(;
                    dict = Dict("nx" => nx,
                                "nx_group" => [i <= 5 ? "g1" : "g2" for i in 1:N]))
tol = 1e-6
success(res) = isa(res.retcode, OptimisationSuccess)
held(w) = count(x -> abs(x) > tol, w)
hopt(; kwargs...) = HierarchicalOptimiser(; kwargs...)
hrp(; kwargs...) = HierarchicalRiskParity(; opt = hopt(; kwargs...))

@testset "Every head keeps the cardinality and the concentration limit of its finaliser" begin
    cl = ConcentrationLimit(; t = 0.15, c = 0.3, s = 0.6)
    wf = JuMPWeightFinaliser(; slv = hslv, card = 5, conc = cl)
    # The fixed point holds weights near zero, which a relative deviation divides by.
    wfa = JuMPWeightFinaliser(; slv = hslv, card = 5, conc = cl,
                              alg = ErrorWeightFinaliser(; dev = AbsoluteWeightDeviation()))
    wb = WeightBounds(; ub = 0.3)
    prs = [SchurComplementParams(), SchurComplementParams(; gamma = 0.2)]
    heads = [hrp(; wb, wf),
             HierarchicalRiskParity(; r = [Variance(), ConditionalValueatRisk()],
                                    opt = hopt(; wb, wf)),
             HierarchicalEqualRiskContribution(; opt = hopt(; wb, wf)),
             SchurComplementHierarchicalRiskParity(; opt = hopt(; wb, wf)),
             SchurComplementHierarchicalRiskParity(; params = prs, opt = hopt(; wb, wf)),
             InverseVolatility(; wb, wf), EqualWeighted(; wb, wf),
             RandomWeighted(; wb, wf, seed = 3, rng = StableRNG(1)),
             BestConstantRebalancedPortfolio(; wb, wf = wfa),
             NestedClustered(; wb, wf, opti = InverseVolatility(),
                             opto = InverseVolatility()),
             Stacking(; wb, wf, opti = [InverseVolatility(), EqualWeighted()],
                      opto = InverseVolatility()),
             # An asset in no subset holds a zero, which a relative deviation divides by.
             SubsetResampling(; wb, wf = wfa, opt = InverseVolatility(), subset_size = 6,
                              n_subsets = 4, rng = StableRNG(7))]
    for h in heads
        res = optimise(h, rd)
        @test success(res)
        @test held(res.w) <= 5
        @test isapprox(sum(res.w), 1; atol = tol)
        cm = concentration_margins(cl, res.w)
        @test all(cm.cap .<= tol)
        @test cm.total <= tol
        # Without the block the book breaks the cardinality.
        free = optimise(PO.rebuild_struct(h,
                                          if hasproperty(h, :opt) &&
                                             h.opt isa HierarchicalOptimiser
                                              (; opt = hopt(; wb))
                                          else
                                              (; wf = IterativeWeightFinaliser())
                                          end), rd)
        @test held(free.w) > 5 || h isa BestConstantRebalancedPortfolio
    end
    # The result of the fixed point keeps its record.
    res = optimise(BestConstantRebalancedPortfolio(; wb, wf = wfa), rd)
    @test hasproperty(res.retcode.res, :converged)
end

@testset "A finaliser with no slots is the programme of the bounds" begin
    w0 = [0.5, 0.2, 0.15, 0.05, 0.1]
    ws = [0.6, 0.5, -0.4, 0.2, 0.1]
    wb = WeightBounds(; lb = 0.05, ub = 0.3)
    wbs = WeightBounds(; lb = fill(-0.2, 5), ub = fill(0.3, 5))
    for norm in (L1Norm(), L2Norm(), SquaredL2Norm(), LInfNorm(), LpNorm(; p = 3)),
        dev in (RelativeWeightDeviation(), AbsoluteWeightDeviation()),
        (wbi, wi) in ((wb, w0), (wbs, ws))

        alg = ErrorWeightFinaliser(; alg = norm, dev = dev)
        model = JuMP.Model()
        JuMP.@expression(model, sc, 1.0)
        JuMP.@expression(model, so, 1.0)
        JuMP.@variable(model, w[1:5])
        JuMP.@constraint(model, sum(w) - sum(wi) == 0)
        JuMP.@constraint(model, w .- wbi.lb >= 0)
        JuMP.@constraint(model, w .- wbi.ub <= 0)
        PO.set_clustering_weight_finaliser_alg!(model, alg, wi)
        JuMP.set_optimizer(model, Clarabel.Optimizer)
        JuMP.set_silent(model)
        JuMP.optimize!(model)
        wf = JuMPWeightFinaliser(; slv = cslv, alg = alg)
        @test isapprox(PO.opt_weight_bounds(wf, wbi, copy(wi)), JuMP.value.(w); atol = 1e-8)
    end
    # Weights that break no bound come back with no solve, and a slot solves for them too.
    wok = fill(0.2, 5)
    @test PO.opt_weight_bounds(JuMPWeightFinaliser(; slv = bad), wb, wok) === wok
    wc = PO.opt_weight_bounds(JuMPWeightFinaliser(; slv = hslv, card = 4),
                              WeightBounds(; ub = 0.3), wok)
    @test held(wc) <= 4
    # An entry that is not finite reaches no solver, and it is a failure.
    for wf in
        (JuMPWeightFinaliser(; slv = bad), JuMPWeightFinaliser(; slv = bad, card = 2))
        rc, wn = PO.finalise_weight_bounds(wf, wb, [0.9, NaN])
        @test isa(rc, OptimisationFailure)
        @test isequal(wn, [0.9, NaN])
    end
end

@testset "The constructors refuse what the block cannot hold" begin
    @test_throws r"State `wb` on the optimiser" JuMPWeightFinaliser(; slv = cslv,
                                                                    wb = WeightBounds())
    @test_throws r"State `wb` on the optimiser" JuMPWeightFinaliser(; slv = cslv,
                                                                    cbe = ConstraintBlock(;
                                                                                          card = 3))
    @test_throws r"State `sets` on the optimiser" JuMPWeightFinaliser(; slv = cslv,
                                                                      sets = sets)
    @test_throws r"Schedule the finaliser" JuMPWeightFinaliser(; slv = cslv,
                                                               card = TimeDependent([2, 3]))
    @test_throws r"has no rule that binds its matrix" JuMPWeightFinaliser(; slv = cslv,
                                                                          alg = GramProjection(;
                                                                                               slv = cslv))
    @test_throws IsEmptyError JuMPWeightFinaliser(; slv = Solver[])
    @test_throws DomainError JuMPWeightFinaliser(; slv = cslv, sc = 0)
    @test_throws DomainError JuMPWeightFinaliser(; slv = cslv, so = -1)
    @test_throws MethodError JuMPWeightFinaliser(; slv = cslv, notaslot = 1)
    # A given block, and a flat keyword that replaces its slot.
    wf = JuMPWeightFinaliser(; slv = cslv, cbe = ConstraintBlock(; wb = nothing, card = 3),
                             card = 4)
    @test wf.card == 4
    @test isnothing(wf.wb) && isnothing(wf.sets)
    # A slot keyed by name needs the sets of the head, and each head checks it.
    lcse = LinearConstraintEstimator(; val = "a1 >= 0.2")
    wfn = JuMPWeightFinaliser(; slv = cslv, lcse = lcse)
    @test PO.name_keyed(wfn)
    @test !PO.name_keyed(wf)
    iv = InverseVolatility()
    for f in (() -> hopt(; wf = wfn), () -> InverseVolatility(; wf = wfn),
              () -> EqualWeighted(; wf = wfn), () -> RandomWeighted(; wf = wfn),
              () -> BestConstantRebalancedPortfolio(; wf = wfn),
              () -> NestedClustered(; opti = iv, opto = iv, wf = wfn),
              () -> Stacking(; opti = [iv], opto = iv, wf = wfn),
              () -> SubsetResampling(; opt = iv, wf = wfn),
              () -> hopt(; wf = TimeDependent([wfn])),
              () -> JuMPOptimiser(; slv = cslv, lcse = lcse),
              () -> ProgrammeAllocationSet(; slv = cslv, lcse = lcse))
        @test_throws IsNothingError f()
    end
    @test ConstraintBlock(; lcse = lcse).lcse === lcse
    # A head that fits no prior refuses a slot that reads one, in a schedule too.
    wfp = JuMPWeightFinaliser(; slv = cslv,
                              tr = TrackingError(; tr = WeightsTracking(; w = fill(0.1, N)),
                                                 err = 0.1))
    for f in (() -> EqualWeighted(; wf = wfp), () -> RandomWeighted(; wf = wfp),
              () -> BestConstantRebalancedPortfolio(; wf = wfp),
              () -> EqualWeighted(; wf = TimeDependent([wfp])))
        @test_throws r"reads a prior" f()
    end
    for kw in ((; cte = CentralityConstraint()), (; ple = IntegerPhylogenyEstimator()),
               (;
                lcse = ExposureConstraintEstimator(;
                                                   lce = LinearConstraintEstimator(;
                                                                                   val = "f1 <= 0.1"),
                                                   space = FactorSpace(; re = StepwiseRegression()))),
               (; l2 = L2Regularisation(; val = ConcentrationRadius())))
        @test PO.block_reads_prior(JuMPWeightFinaliser(; slv = cslv, kw...).cbe)
    end
    @test !PO.block_reads_prior(wf.cbe)
    @test_throws r"reads a prior" PO.opt_weight_bounds(wfp, WeightBounds(), fill(0.1, N))
    # A benchmark of the returns data with a prior that already subtracts it.
    wfd = JuMPWeightFinaliser(; slv = cslv,
                              tr = TrackingError(; tr = DataBenchmarkTracking(), err = 0.1))
    pex = EmpiricalPrior(; x_src = ExcessSource())
    for f in
        (() -> hopt(; pe = pex, wf = wfd), () -> InverseVolatility(; pe = pex, wf = wfd),
         () -> NestedClustered(; pe = pex, opti = iv, opto = iv, wf = wfd),
         () -> Stacking(; pe = pex, opti = [iv], opto = iv, wf = wfd),
         () -> SubsetResampling(; pe = pex, opt = iv, wf = wfd))
        @test_throws ArgumentError f()
    end
end

@testset "A failed solve with a slot is a failure, and the fallback runs" begin
    wb = WeightBounds(; ub = 0.12)
    w0 = [0.3; fill(0.7 / 9, 9)]
    iter = PO.opt_weight_bounds(IterativeWeightFinaliser(), wb, w0)
    rc, w = PO.finalise_weight_bounds(JuMPWeightFinaliser(; slv = bad), wb, w0)
    @test isa(rc, OptimisationSuccess)
    @test w == iter
    wfc = JuMPWeightFinaliser(; slv = bad, card = 3)
    rc, w = PO.finalise_weight_bounds(wfc, wb, w0)
    @test isa(rc, OptimisationFailure)
    @test isa(rc.res, AbstractDict)
    @test w == iter
    @test PO.opt_weight_bounds(wfc, wb, w0) == iter
    # A cardinality of three at 12 % holds 36 %, so the programme is infeasible.
    res = optimise(HierarchicalRiskParity(;
                                          opt = hopt(; wb,
                                                     wf = JuMPWeightFinaliser(; slv = hslv,
                                                                              card = 3)),
                                          fb = EqualWeighted()), rd)
    @test success(res)
    @test res.w ≈ fill(0.1, N)
    # A slot whose programme solves keeps the success.
    @test success(optimise(hrp(; wb = WeightBounds(; ub = 0.5),
                               wf = JuMPWeightFinaliser(; slv = hslv, card = 3)), rd))
end

@testset "A Projection Geometry is the objective" begin
    w0 = optimise(hrp(), rd).w
    wb = WeightBounds(; ub = 0.105)
    # The entropic projection onto a cap keeps the ratios of the free weights.
    we = PO.opt_weight_bounds(JuMPWeightFinaliser(; slv = cslv, alg = EntropicProjection()),
                              wb, w0)
    free = we .< 0.105 - 1e-5
    @test 1 < count(free) < N
    r = we[free] ./ w0[free]
    @test isapprox(r, fill(r[1], length(r)); rtol = 1e-4)
    @test isapprox(sum(we), 1; atol = tol)
    @test isapprox(we, PO.opt_weight_bounds(EntropicWeightFinaliser(), wb, w0); atol = 1e-5)
    @test isapprox(PO.opt_weight_bounds(JuMPWeightFinaliser(; slv = cslv,
                                                            alg = EuclideanProjection()),
                                        wb, w0),
                   PO.opt_weight_bounds(EuclideanWeightFinaliser(), wb, w0); atol = 1e-5)
    @test_throws DomainError PO.opt_weight_bounds(JuMPWeightFinaliser(; slv = cslv,
                                                                      alg = EntropicProjection()),
                                                  WeightBounds(; lb = -1.0, ub = 0.1), w0)
    # The Gram projection minimises the tracking variance to the input, on its own solver.
    A = prior(EmpiricalPrior(), rd).sigma
    lin = LinearConstraint(;
                           ineq = PartialLinearConstraint(; A = [-ones(1, 3) zeros(1, 7)],
                                                          B = [-0.45]))
    wg = PO.opt_weight_bounds(JuMPWeightFinaliser(; slv = bad, lcse = lin,
                                                  alg = GramProjection(; slv = cslv, A = A)),
                              WeightBounds(), w0)
    model = JuMP.Model(Clarabel.Optimizer)
    JuMP.set_silent(model)
    JuMP.@variable(model, 0 <= w[1:N] <= 1)
    JuMP.@constraint(model, sum(w) == 1)
    JuMP.@constraint(model, sum(w[1:3]) >= 0.45)
    JuMP.@objective(model, Min, 1e4 * (w - w0)' * A * (w - w0))
    JuMP.optimize!(model)
    @test isapprox(wg, JuMP.value.(w); atol = 1e-4)
    @test sum(wg[1:3]) >= 0.45 - tol
    wd = PO.opt_weight_bounds(JuMPWeightFinaliser(; slv = cslv, lcse = lin,
                                                  alg = PO.DiagonalProjection(ones(N))),
                              WeightBounds(), w0)
    @test isapprox(wd,
                   PO.opt_weight_bounds(JuMPWeightFinaliser(; slv = cslv, lcse = lin,
                                                            alg = EuclideanProjection()),
                                        WeightBounds(), w0); atol = 1e-5)
end

@testset "A penalty of the block weighs against the deviation" begin
    w0 = [0.4, 0.3, 0.2, 0.1]
    # Half the squared distance plus c / 2 of the squared norm has the closed form
    # w = (w0 + c / N) / (1 + c) on the budget of one.
    wf = JuMPWeightFinaliser(; slv = cslv, alg = EuclideanProjection(),
                             l2 = L2Regularisation(; val = 0.5, alg = QuadRiskExpr()))
    w = PO.opt_weight_bounds(wf, WeightBounds(), w0)
    @test isapprox(w, (w0 .+ 1.0 / 4) ./ 2; atol = 1e-5)
    wq = JuMPWeightFinaliser(; slv = cslv,
                             alg = ErrorWeightFinaliser(; alg = SquaredL2Norm(),
                                                        dev = AbsoluteWeightDeviation()),
                             l2 = L2Regularisation(; val = 1.0, alg = QuadRiskExpr()))
    @test isapprox(PO.opt_weight_bounds(wq, WeightBounds(), w0), (w0 .+ 0.25) ./ 2;
                   atol = 1e-5)
end

@testset "A budget that is not one, and a long-short book" begin
    w0 = [0.5, 0.45, 0.25, 0.1, 0.2]
    wb = WeightBounds(; ub = 0.6)
    # A meta head can give a budget that is not one.
    W = [w0 reverse(w0)]
    ok = optimise(EqualWeighted(), rd)
    wf = JuMPWeightFinaliser(; slv = hslv, card = 3)
    ctx = (nothing, nothing, nothing, nothing, false)
    rc, w = PO.outer_optimisation_finaliser(wb, wf, [ok, ok], OptimisationSuccess(),
                                            [0.5, 0.5], W, ctx)
    @test isa(rc, OptimisationSuccess)
    @test isapprox(sum(w), 1.5; atol = tol)
    @test held(w) <= 3
    rcs, ws = PO.outer_optimisation_finaliser(wb, wf, [ok, ok],
                                              [OptimisationSuccess(),
                                               OptimisationSuccess()],
                                              [[0.5, 0.5], [1.0, 0.0]], W, ctx)
    @test all(x -> isapprox(sum(x), 1.5; atol = tol) && held(x) <= 3, ws)
    # A short stays without a short budget, and `sbgt` and `gbgt` cap it.
    ws0 = [0.6, 0.5, -0.4, 0.2, 0.1]
    wbs = WeightBounds(; lb = fill(-0.5, 5), ub = fill(0.6, 5))
    alg = ErrorWeightFinaliser(; alg = L2Norm(), dev = AbsoluteWeightDeviation())
    fin(; kw...) = PO.opt_weight_bounds(JuMPWeightFinaliser(; slv = cslv, alg = alg, kw...),
                                        wbs, ws0)
    short(w) = sum(max.(-w, 0))
    @test isapprox(fin(; l2c = 10.0), ws0; atol = 1e-5)
    w = fin(; sbgt = 0.2)
    @test short(w) <= 0.2 + tol
    @test isapprox(sum(w), 1; atol = tol)
    w = fin(; gbgt = BudgetRange(; lb = nothing, ub = 1.4))
    @test sum(abs, w) <= 1.4 + tol
    @test isapprox(sum(w), 1; atol = tol)
    @test short(w) > 0.1
    # A sum that passes the ceiling of the gross budget by a rounding error is that ceiling.
    @test PO.finaliser_budget([1.0, eps()], 1) == 1.0
    @test PO.finaliser_budget([-1.0, -eps()], 1) == -1.0
    @test PO.finaliser_budget([1.0, 0.1], 1) == 1.1
    @test PO.finaliser_budget([0.5, 0.25], nothing) == 0.75
    @test_throws ArgumentError fin(; gbgt = 0.5)
end

@testset "Names, views and the Investable Mask" begin
    # A precomputed prior and a panel where asset 4 has no price.
    Xn = copy(X)
    Xn[1:40, 4] .= NaN
    amsk = isfinite.(Xn)
    rdn = ReturnsResult(; nx = nx, X = Xn,
                        pnl = AssetPanel(; pf = nothing, amsk = amsk, emsk = amsk))
    keep = setdiff(1:N, 4)
    wt = collect(range(0.05, 0.15; length = N))
    cl = ConcentrationLimitEstimator(; t = 0.2, c = 0.7, s = 1.0,
                                     set = AssetSetsMatrixEstimator(; val = "nx_group"))
    A = prior(EmpiricalPrior(), rd).sigma
    for wf in (JuMPWeightFinaliser(; slv = hslv, tn = Turnover(; w = wt, val = 0.03),
                                   lcse = LinearConstraintEstimator(;
                                                                    val = ["g1 >= 0.35",
                                                                           "a4 <= 0.05"],
                                                                    key = nothing),
                                   gcarde = LinearConstraintEstimator(; val = "g2 <= 5"),
                                   conc = cl),
               JuMPWeightFinaliser(; slv = cslv, alg = GramProjection(; slv = cslv, A = A),
                                   tn = TurnoverEstimator(; w = wt, val = Dict("a1" => 0.01),
                                                          dval = 0.2)),
               JuMPWeightFinaliser(; slv = cslv, alg = PO.DiagonalProjection(wt), lt = nothing,
                                   l2c = 0.5))
        for h in (hrp(; wf, sets, wb = WeightBounds(; ub = 0.3)),
                  EqualWeighted(; wf, sets, wb = WeightBounds(; ub = 0.3)),
                  InverseVolatility(; wf, sets, wb = WeightBounds(; ub = 0.3)))
            res = optimise(h, rdn)
            @test success(res)
            @test iszero(res.w[4])
            @test isapprox(sum(res.w), 1; atol = tol)
            if isa(wf.tn, Turnover)
                @test all(abs.(res.w[keep] .- wt[keep]) .<= 0.03 + tol)
                @test sum(res.w[1:5]) >= 0.35 - tol
            elseif isa(wf.tn, TurnoverEstimator)
                @test abs(res.w[1] - wt[1]) <= 0.01 + tol
            end
        end
    end
    wf = JuMPWeightFinaliser(; slv = cslv, alg = GramProjection(; slv = cslv, A = A),
                             tn = Turnover(; w = wt, val = 0.1))
    v = PO.port_opt_view(wf, [1, 3], X)
    @test v.alg.A == A[[1, 3], [1, 3]]
    @test v.tn.w == wt[[1, 3]]
    @test PO.port_opt_view(GramProjection(; slv = cslv), 1:2).A === nothing
    # A nested head views the finaliser of its own and of its inner optimisers.
    st = Stacking(; opti = [hrp(; wf)], opto = InverseVolatility(), wf = wf)
    sv = PO.port_opt_view(st, [1, 3], X)
    @test sv.wf.tn.w == wt[[1, 3]]
    @test sv.opti[1].opt.wf.tn.w == wt[[1, 3]]
    nv = PO.port_opt_view(NestedClustered(; opti = hrp(; wf), opto = InverseVolatility(),
                                          wf = wf), [1, 3], X)
    @test nv.wf.tn.w == wt[[1, 3]] == nv.opti.opt.wf.tn.w
    @test PO.port_opt_view(SubsetResampling(; opt = InverseVolatility(), wf = wf), [1, 3],
                           X).wf.tn.w == wt[[1, 3]]
end

@testset "The weights of the previous fold reach the turnover ceiling" begin
    wf = JuMPWeightFinaliser(; slv = cslv, alg = EuclideanProjection(),
                             tn = Turnover(; w = fill(0.1, N), val = 0.02))
    iv = InverseVolatility()
    heads = [hrp(; wf), InverseVolatility(; wf), EqualWeighted(; wf),
             NestedClustered(; opti = iv, opto = iv, wf),
             Stacking(; opti = [iv], opto = iv, wf), SubsetResampling(; opt = iv, wf)]
    for h in heads
        @test PO.needs_previous_weights(h)
        hw = PO.factory(h, fill(0.2, N))
        wfw = hw isa HierarchicalRiskParity ? hw.opt.wf : hw.wf
        @test wfw.tn.w == fill(0.2, N)
    end
    @test !PO.needs_previous_weights(hrp())
    cv = IndexWalkForward(60, 30)
    # Each fold measures from the book of the fold before it, and a fixed reference stays.
    function books(fixed)
        wft = JuMPWeightFinaliser(; slv = cslv, alg = EuclideanProjection(),
                                  tn = Turnover(; w = fill(0.1, N), val = 0.002,
                                                fixed = fixed))
        return [r.res.w for r in cross_val_predict(hrp(; wf = wft), rd, cv).pred]
    end
    ws = books(false)
    @test length(ws) == 3
    @test all(i -> maximum(abs, ws[i] - ws[i - 1]) <= 0.002 + 1e-5, 2:3)
    @test maximum(abs, ws[3] .- 0.1) > 0.003
    @test all(w -> maximum(abs, w .- 0.1) <= 0.002 + 1e-5, books(true))
    # A schedule of the finaliser gives each fold its own cardinality.
    td = TimeDependent([JuMPWeightFinaliser(; slv = hslv, card = c) for c in (2, 3, 4)])
    res = cross_val_predict(hrp(; wf = td, wb = WeightBounds()), rd, cv)
    @test [held(r.res.w) for r in res.pred] == [2, 3, 4]
end

@testset "The readers of the block join the mask and the frames of the head" begin
    rng = StableRNG(987654321)
    T = 60
    X0 = randn(rng, T, 5) ./ 100 .+ 0.0005
    X0[1:30, 4] .= NaN
    amsk = isfinite.(X0)
    rdy = ReturnsResult(; nx = ["a", "b", "c", "d", "e"], X = X0,
                        pnl = AssetPanel(; pf = nothing, amsk = amsk, emsk = amsk))
    pe = EmpiricalPrior(; me = ExpWeightedExpectedReturns(; decay = 0.9, min_obs = 2),
                        ce = ExpWeightedCovariance(; decay = 0.9, min_obs = 2,
                                                   centring = PreCentred()), fill_limit = 1)
    # The prior keeps the young asset `d`, and the reader of the finaliser refuses it.
    sce = PortfolioOptimisersCovariance(;
                                        ce = Covariance(;
                                                        cvg = CoveragePolicy(;
                                                                             min_coverage = 0.6)))
    nte = NetworkEstimator(; ce = sce)
    cte = CentralityConstraint(; A = CentralityEstimator(; pl = nte), B = MeanValue(),
                               comp = <=)
    cle = ClustersEstimator(;
                            ce = ExpWeightedCovariance(; decay = 0.9, min_obs = 2,
                                                       centring = PreCentred()))
    iv = InverseVolatility(; pe = pe)
    for (wf, msg) in ((JuMPWeightFinaliser(; slv = cslv, cte = cte),
                       r"distance reader of this optimiser cannot estimate the correlation of 1 asset\(s\) that the prior kept: \[\"d\"\]"),
                      # A plain reader meets the gap of `d` at the entry.
                      (JuMPWeightFinaliser(; slv = cslv,
                                           ple = SemiDefinitePhylogenyEstimator(; pl = NetworkEstimator())),
                       r"graph reader of this optimiser.*\[\"d\"\]"))
        ho(s) = hopt(; pe, cle, wf, strict = s)
        prs = [SchurComplementParams(), SchurComplementParams(; gamma = 0.2)]
        heads = [s -> HierarchicalRiskParity(; opt = ho(s)),
                 s -> HierarchicalRiskParity(; r = [Variance(), StandardDeviation()],
                                             opt = ho(s)),
                 s -> HierarchicalEqualRiskContribution(; opt = ho(s)),
                 s -> SchurComplementHierarchicalRiskParity(; opt = ho(s)),
                 s -> SchurComplementHierarchicalRiskParity(; params = prs, opt = ho(s)),
                 s -> InverseVolatility(; pe, wf, strict = s),
                 s -> NestedClustered(; pe, cle, wf, strict = s, opti = iv,
                                      opto = InverseVolatility()),
                 s -> Stacking(; pe, wf, strict = s, opti = [iv, iv],
                               opto = InverseVolatility()),
                 s -> SubsetResampling(; pe, wf, strict = s, opt = iv, subset_size = 3,
                                       n_subsets = 2)]
        for h in heads
            res = @test_logs (:warn, msg) match_mode = :any optimise(h(false), rdy)
            @test res.imsk == BitVector([1, 1, 1, 0, 1])
            @test iszero(res.w[4])
            @test isapprox(sum(res.w), 1; atol = 1e-5)
            @test_throws IsNonFiniteError optimise(h(true), rdy)
        end
    end
    # The centrality row holds on the book of a full universe.
    wfc = JuMPWeightFinaliser(; slv = cslv,
                              cte = CentralityConstraint(; B = MeanValue(), comp = <=))
    res = optimise(hrp(; wf = wfc), rd)
    c = centrality_vector(CentralityEstimator(), rd).X
    @test success(res)
    @test dot(c, res.w) <= mean(c) + 1e-5
    # The reader is in the group of the head, at the path of its slot.
    wfr = JuMPWeightFinaliser(; slv = cslv, ple = SemiDefinitePhylogenyEstimator())
    fr = PO.group_frames(hopt(; wf = wfr))
    @test any(p -> occursin("wf.cbe.ple", p), fr.raw)
    # A view of the observations cuts the weights inside the block.
    w = StatsBase.pweights(fill(1.0, 160))
    tr = TrackingError(; tr = WeightsTracking(; w = fill(0.1, N)), err = 0.1)
    wfo = JuMPWeightFinaliser(; slv = cslv, tr = tr,
                              ple = SemiDefinitePhylogenyEstimator(;
                                                                   pl = NetworkEstimator(;
                                                                                         ce = Covariance(;
                                                                                                         ce = GeneralCovariance(;
                                                                                                                                w = w)))))
    @test length(PO.obs_weights_view(hopt(; wf = wfo), 1:50).wf.ple.pl.ce.ce.w) == 50
end

@testset "Tracking errors and the fees of the head" begin
    pr = prior(EmpiricalPrior(), rd)
    wb0 = [0.3; 0.3; fill(0.05, 8)]
    b = X * wb0
    fees = Fees(; l = 0.0005)
    te(w, fees = nothing) = norm(calc_net_returns(w, X, fees) - b, 2) / sqrt(size(X, 1) - 1)
    err = 0.0008
    tr = TrackingError(; tr = ReturnsTracking(; w = b), err = err, alg = L2Norm(; ddof = 1))
    alg = EuclideanProjection()
    w = optimise(hrp(; wf = JuMPWeightFinaliser(; slv = cslv, alg, tr)), rd).w
    @test isapprox(te(w), err; rtol = 1e-3)
    # Under a fee the tracked series is net of the fee.
    res = optimise(hrp(; fees, wf = JuMPWeightFinaliser(; slv = cslv, alg, tr)), rd)
    @test success(res)
    @test isapprox(te(res.w, fees), err; rtol = 1e-3)
    @test te(res.w) < err
    # A fixed fee makes no programme mixed-integer without a tracking error.
    ff = Fees(; fl = 0.001)
    @test success(optimise(hrp(; fees = ff, wb = WeightBounds(; ub = 0.2),
                               wf = JuMPWeightFinaliser(; slv = cslv, alg, l2c = 0.4)), rd))
    # A benchmark of the returns data binds to the rows of the head.
    rdb = ReturnsResult(; nx = nx, X = X, nb = ["b"], B = b)
    wfd = JuMPWeightFinaliser(; slv = cslv, alg,
                              tr = TrackingError(; tr = DataBenchmarkTracking(), err = err,
                                                 alg = L2Norm(; ddof = 1)))
    @test isapprox(optimise(hrp(; wf = wfd), rdb).w, w; atol = 1e-6)
    # A tracking error of a risk measure takes the finaliser as the owner.
    sd(w) = sqrt(dot(w, pr.sigma, w))
    for rt in (IndependentVariableTracking(), DependentVariableTracking())
        te2 = RiskTrackingError(; tr = WeightsTracking(; w = wb0), r = StandardDeviation(),
                                err = 0.0002, alg = rt)
        res = optimise(hrp(; wf = JuMPWeightFinaliser(; slv = cslv, alg, tr = te2)), rd)
        @test success(res)
        v = rt isa IndependentVariableTracking ? sd(res.w - wb0) : sd(res.w) - sd(wb0)
        @test v <= 0.0002 + 1e-7
    end
    # The finaliser has no sets, so the rows of a tracked variance are values.
    wf = JuMPWeightFinaliser(; slv = cslv, alg)
    @test isnothing(PO.risk_contribution_constraints(Variance(), wf, pr))
    @test_throws r"keyed by name" PO.risk_contribution_constraints(Variance(;
                                                                            rc = LinearConstraintEstimator(;
                                                                                                           val = "a1 <= 0.1")),
                                                                   wf, pr)
    # A tracked risk has one bound, so the owner refuses a frontier.
    @test_throws ArgumentError PO.set_risk_upper_bound!(JuMP.Model(), wf, nothing,
                                                        Frontier(; N = 3), :r)
end

@testset "A JuMP head charges its fees before its tracking error registers the net returns" begin
    # Before, the tracking row registered the gross returns as `net_X`, and each measure of
    # the head read them.
    b = X * fill(0.1, N)
    fees = Fees(; l = 0.001)
    loose = TrackingError(; tr = ReturnsTracking(; w = b), err = 1e3)
    function fit(tr, fees)
        return optimise(MeanRisk(; r = ConditionalValueatRisk(),
                                 opt = JuMPOptimiser(; slv = cslv, fees = fees, tr = tr)),
                        rd)
    end
    res = fit(loose, fees)
    net = JuMP.value.(res.model[:net_X])
    @test isapprox(net, X * res.w .- 0.001; atol = 1e-8)
    @test isapprox(JuMP.objective_value(res.model),
                   JuMP.objective_value(fit(nothing, fees).model); rtol = 1e-6)
    # A fixed fee beside a tracking error solves, and its clock is registered.
    rf = optimise(MeanRisk(; r = ConditionalValueatRisk(),
                           opt = JuMPOptimiser(; slv = hslv, fees = Fees(; fl = 0.0001),
                                               wb = WeightBounds(; ub = 0.3),
                                               tr = TrackingError(;
                                                                  tr = ReturnsTracking(;
                                                                                       w = b),
                                                                  err = 1e3,
                                                                  alg = L1Norm()))), rd)
    @test success(rf)
end
