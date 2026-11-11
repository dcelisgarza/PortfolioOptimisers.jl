@testset "Score weight bounds" begin
    using Test, PortfolioOptimisers, StableRNGs, Statistics, Clarabel
    PO = PortfolioOptimisers

    rng = StableRNG(20061)
    T, N = 80, 5
    X = 0.01 .* randn(rng, T, N) .+ 0.0005
    Q = 100 .* rand(rng, T, N)
    Q[:, 5] .= NaN # an asset with no observed volume
    pnl = asset_panel([NumericPanelInput(; name = "volume", vals = Q,
                                         alg = ForwardPanelFill())]; amsk = trues(T, N),
                      emsk = trues(T, N))
    nx = string.('A':'E')
    rd = ReturnsResult(; nx = nx, X = X, pnl = pnl)
    sets = UniverseSets(; dict = Dict("nx" => nx))
    s = PanelFieldScorer(; fields = "volume")
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false),
                 check_sol = (; allow_local = true, allow_almost = true))
    # The bound by hand: the median volume of each asset over `rows`, times `v`.
    function hand_cap(rows, v; nf = 0.0)
        return [(q = filter(isfinite, Q[rows, i]); isempty(q) ? nf : v * median(q))
                for i in 1:N]
    end

    @testset "construction and validation" begin
        swb = ScoreWeightBounds(; score = s)
        @test swb isa PO.AbstractWeightBoundsEstimator
        @test swb.val == 1 && isnothing(swb.dval) && swb.wb == WeightBounds()
        @test swb.symmetric && swb.nonfinite == 0
        # Issue #2008: a number or a vector of numbers in `val` names no asset, so the
        # estimator is keyed by name only through a named `val` or a named static bound.
        @test !PO.name_keyed(swb)
        @test !PO.name_keyed(ScoreWeightBounds(; score = s, val = [1.0, 2.0], wb = nothing))
        named = ScoreWeightBounds(; score = s, val = Dict("A" => 0.1))
        @test PO.name_keyed(named)
        @test PO.name_keyed(ScoreWeightBounds(; score = s, val = "A" => 0.1))
        @test PO.name_keyed(ScoreWeightBounds(; score = s,
                                              wb = WeightBoundsEstimator(;
                                                                         ub = Dict("A" =>
                                                                                       0.5))))
        @test PO.name_keyed(ScoreWeightBounds(; score = s, wb = named))
        @test ScoreWeightBounds(; score = SCM()).score == SCM()
        @test_throws ArgumentError ScoreWeightBounds(; score = Variance())
        @test_throws DomainError ScoreWeightBounds(; score = s, val = -1.0)
        @test_throws DomainError ScoreWeightBounds(; score = s, val = [1.0, NaN])
        @test_throws PO.IsEmptyError ScoreWeightBounds(; score = s, val = Float64[])
        @test_throws DomainError ScoreWeightBounds(; score = s, dval = -1.0)
        @test ScoreWeightBounds(; score = s, dval = Inf).dval == Inf
        @test_throws DomainError ScoreWeightBounds(; score = s, nonfinite = -0.1)
        @test isnothing(ScoreWeightBounds(; score = s, nonfinite = nothing).nonfinite)
        # Every holder of a batch fit needs `sets` for a named bound, because `val` resolves
        # against it.
        @test_throws PO.IsNothingError EqualWeighted(; wb = named)
        @test_throws PO.IsNothingError InverseVolatility(; wb = named)
        @test_throws PO.IsNothingError RandomWeighted(; wb = named)
        @test_throws PO.IsNothingError BestConstantRebalancedPortfolio(; wb = named)
        @test_throws PO.IsNothingError JuMPOptimiser(; slv = slv, wb = named)
        @test_throws PO.IsNothingError HierarchicalOptimiser(; wb = named)
        @test_throws PO.IsNothingError NestedClustered(; wb = named, opti = EqualWeighted(),
                                                       opto = EqualWeighted())
        @test_throws PO.IsNothingError Stacking(; wb = named,
                                                opti = [EqualWeighted(),
                                                        InverseVolatility()],
                                                opto = EqualWeighted())
        @test_throws PO.IsNothingError SubsetResampling(; wb = named, opt = EqualWeighted())
        @test_throws PO.IsNothingError BudgetedHindsightPath(; L = 0.1, p = 1, wb = named,
                                                             slv = slv)
        # An Allocation Set of an online head resolves once for a block of rows, so a
        # bound that reads the returns of the block would put a look-ahead in the pass.
        @test_throws TypeError BoundedAllocationSet(; wb = swb, sets = sets)
        @test_throws TypeError ProgrammeAllocationSet(; wb = swb, sets = sets, slv = slv)
    end

    @testset "score_cap and score_bounds" begin
        @test PO.score_cap(0.5, 2.0, 0) === 1.0
        @test PO.score_cap(0.5, NaN, 0) === 0.0
        @test PO.score_cap(0.5, Inf, Inf) === Inf
        @test PO.score_cap(Inf, 0.0, 0) === Inf
        @test PO.score_cap(0.5f0, 2.0f0, 0) === 1.0f0
        wb = WeightBounds([-1.0, 0.0, -0.1], [1.0, 0.2, 1.0])
        c = [0.5, 0.25, 0.3]
        r = PO.score_bounds(wb, c, true)
        @test r.lb == [-0.5, 0.0, -0.1] && r.ub == [0.5, 0.2, 0.3]
        r = PO.score_bounds(wb, c, false)
        @test r.lb == wb.lb && r.ub == [0.5, 0.2, 0.3]
        @test_throws DomainError PO.score_bounds(wb, [0.5, -0.25, 0.3], true)
        @test_throws DomainError PO.score_bounds(WeightBounds([0.0, 0.3], [1.0, 1.0]),
                                                 [0.5, 0.25], true)
        # R42 of #1817: under `symmetric = true`, a static upper bound below `-c` admits no
        # weight. The check names the score bound, where the constructor of `WeightBounds`
        # raised a message that names neither side.
        short = WeightBounds([-1.0, -1.0], [-0.5, 1.0])
        err = try
            PO.score_bounds(short, [0.2, 0.2], true)
        catch e
            e
        end
        @test err isa DomainError
        @test occursin("below the negative score bound at 1 assets", err.msg)
        r = PO.score_bounds(short, [0.2, 0.2], false)
        @test r.lb == [-1.0, -1.0] && r.ub == [-0.5, 0.2]
        r = PO.score_bounds(WeightBounds([-1.0, -1.0], [-0.2, 1.0]), [0.2, 0.2], true)
        @test r.lb == [-0.2, -0.2] && r.ub == [-0.2, 0.2]
    end

    @testset "weight_bounds_constraints reads the returns of the fit" begin
        swb = ScoreWeightBounds(; score = s, val = 0.003)
        wb = weight_bounds_constraints(swb, sets; rd = rd)
        @test wb.ub ≈ min.(1.0, hand_cap(1:T, 0.003))
        @test wb.ub[5] == 0
        @test all(iszero, wb.lb)
        @test_throws PO.IsNothingError weight_bounds_constraints(swb, sets)
        @test_throws DimensionMismatch weight_bounds_constraints(swb, sets; rd = rd, N = 3)
        # No bound from the score at an asset with no score, and none at the assets that a
        # dictionary `val` does not name when `dval = Inf`.
        d = ScoreWeightBounds(; score = s, val = Dict("A" => 0.001), dval = Inf,
                              nonfinite = nothing, wb = nothing)
        wd = weight_bounds_constraints(d, sets; rd = rd)
        @test wd.ub[1] ≈ hand_cap(1:T, 0.001)[1]
        @test all(isinf, wd.ub[2:5])
        @test wd.lb[1] ≈ -wd.ub[1] && all(isinf, wd.lb[2:5])
        # An asymmetric bound keeps the static lower side, here a short position.
        a = ScoreWeightBounds(; score = s, val = 0.003, symmetric = false,
                              wb = WeightBounds(; lb = -1.0, ub = 1.0))
        wa = weight_bounds_constraints(a, sets; rd = rd)
        @test all(==(-1.0), wa.lb) && wa.ub ≈ wb.ub
        # A name-keyed static bound and a nested score bound resolve through `sets`.
        n = ScoreWeightBounds(; score = s, val = 0.003,
                              wb = WeightBoundsEstimator(; lb = 0.0, ub = Dict("A" => 0.1)))
        wn = weight_bounds_constraints(n, sets; rd = rd)
        @test wn.ub[1] == min(0.1, hand_cap(1:T, 0.003)[1])
        nn = ScoreWeightBounds(; score = s, val = 0.001, wb = swb)
        @test weight_bounds_constraints(nn, sets; rd = rd).ub ≈
              min.(wb.ub, hand_cap(1:T, 0.001))
        # A vector `val` and a risk measure as the score.
        v = ScoreWeightBounds(; score = SCM(), val = [1.0, 2.0, 3.0, 4.0, 5.0])
        wv = weight_bounds_constraints(v, sets; rd = rd)
        @test wv.ub ≈ min.(1.0, [1.0, 2.0, 3.0, 4.0, 5.0] .* PO.asset_scores(SCM(), X))
        # A static lower bound above the score bound admits no weight.
        @test_throws DomainError weight_bounds_constraints(ScoreWeightBounds(; score = s,
                                                                             val = 0.003,
                                                                             wb = WeightBounds(;
                                                                                               lb = 0.1,
                                                                                               ub = 1.0)),
                                                           sets; rd = rd)
        # A view slices a vector `val` and the static bound with it.
        vv = PO.port_opt_view(ScoreWeightBounds(; score = s,
                                                val = [1.0, 2.0, 3.0, 4.0, 5.0],
                                                wb = WeightBounds(; lb = 0.0,
                                                                  ub = [0.1, 0.2, 0.3, 0.4,
                                                                        0.5])), [1, 3])
        @test vv.val == [1.0, 3.0] && vv.wb.ub == [0.1, 0.3] && vv.score === s
        # Issue #2008: a bound that names no asset resolves with no sets, and gives the
        # bounds that it gives with sets.
        @test weight_bounds_constraints(swb, nothing; rd = rd).ub ≈ wb.ub
        @test weight_bounds_constraints(v, nothing; rd = rd).ub ≈ wv.ub
        @test weight_bounds_constraints(a, nothing; rd = rd).lb == wa.lb
        @test_throws DimensionMismatch weight_bounds_constraints(ScoreWeightBounds(;
                                                                                   score = s,
                                                                                   val = [1.0,
                                                                                          2.0]),
                                                                 nothing; rd = rd)
        @test_throws PO.IsNothingError weight_bounds_constraints(d, nothing; rd = rd)
        @test_throws PO.IsNothingError weight_bounds_constraints(n, nothing; rd = rd)
    end

    @testset "an optimiser resolves the bound on its own fit" begin
        swb = ScoreWeightBounds(; score = s, val = 0.006)
        cap = min.(1.0, hand_cap(1:T, 0.006))
        ho = HierarchicalOptimiser(; wb = swb, sets = sets)
        # One optimiser for each site that resolves `wb`, each with the returns of its fit.
        opts = (EqualWeighted(; wb = swb, sets = sets),
                InverseVolatility(; wb = swb, sets = sets),
                RandomWeighted(; wb = swb, sets = sets, seed = 1),
                BestConstantRebalancedPortfolio(; wb = swb, sets = sets),
                HierarchicalRiskParity(; opt = ho),
                HierarchicalRiskParity(; opt = ho, r = [Variance(), Variance()]),
                HierarchicalEqualRiskContribution(; opt = ho),
                SchurComplementHierarchicalRiskParity(; opt = ho),
                SchurComplementHierarchicalRiskParity(; opt = ho,
                                                      params = [SchurComplementParams(),
                                                                SchurComplementParams(;
                                                                                      gamma = 0.2)]),
                NestedClustered(; wb = swb, sets = sets, opti = EqualWeighted(),
                                opto = EqualWeighted()),
                Stacking(; wb = swb, sets = sets,
                         opti = [EqualWeighted(), InverseVolatility()],
                         opto = EqualWeighted()),
                SubsetResampling(; wb = swb, sets = sets, opt = EqualWeighted(), seed = 1),
                BudgetedHindsightPath(; L = 0.5, p = 1, wb = swb, sets = sets, slv = slv))
        for opt in opts
            res = optimise(opt, rd)
            @test res.wb.ub ≈ cap
        end
        res = optimise(MeanRisk(; opt = JuMPOptimiser(; slv = slv, wb = swb, sets = sets)),
                       rd)
        @test res.pa.wb.ub ≈ cap
        @test all(res.w .<= cap .+ 1e-6)
        # Issue #2008: a bound that names no asset needs no sets in any holder.
        hn = HierarchicalOptimiser(; wb = swb)
        for opt in (EqualWeighted(; wb = swb), InverseVolatility(; wb = swb),
                    RandomWeighted(; wb = swb, seed = 1),
                    BestConstantRebalancedPortfolio(; wb = swb),
                    HierarchicalRiskParity(; opt = hn),
                    NestedClustered(; wb = swb, opti = EqualWeighted(), opto = EqualWeighted()),
                    Stacking(; wb = swb, opti = [EqualWeighted(), InverseVolatility()],
                             opto = EqualWeighted()),
                    SubsetResampling(; wb = swb, opt = EqualWeighted(), seed = 1),
                    BudgetedHindsightPath(; L = 0.5, p = 1, wb = swb, slv = slv))
            @test optimise(opt, rd).wb.ub ≈ cap
        end
        res = optimise(MeanRisk(; opt = JuMPOptimiser(; slv = slv, wb = swb)), rd)
        @test res.pa.wb.ub ≈ cap
        # The two online sets still take a bound that reads only names.
        wbe = WeightBoundsEstimator(; ub = Dict("A" => 0.5))
        @test BoundedAllocationSet(; wb = wbe, sets = sets).wb === wbe
        @test ProgrammeAllocationSet(; wb = wbe, sets = sets, slv = slv).wb === wbe
        # Each fold of a walk-forward resolves the bound on its own training rows, so no
        # fold reads the scores of the rows after it.
        cv = IndexWalkForward(40, 20)
        p = cross_val_predict(EqualWeighted(; wb = swb, sets = sets), rd, cv)
        tr = split(cv, rd).train_idx
        for (i, pred) in pairs(p.pred)
            @test pred.res.wb.ub ≈ min.(1.0, hand_cap(tr[i], 0.006))
        end
    end

    @testset "a Pipeline step resolves the bound on the returns of its window" begin
        # R42 of #1817: the step lands on `:wb`, and it scores the assets of the returns of
        # the context, as the `wb` field of an optimiser does. A step that had a target and
        # no step value built, and then raised a `MethodError` at the fit.
        swb = ScoreWeightBounds(; score = s, val = 0.006)
        cap = min.(1.0, hand_cap(1:T, 0.006))
        res = fit(Pipeline(; steps = (swb, EqualWeighted())), rd)
        @test res.ctx.constraints.ub ≈ cap
        @test PO.inject_context(EqualWeighted(), res.ctx).wb.ub ≈ cap
        @test all(res.w .<= cap .+ 1e-12)
        # Each fold of a walk-forward runs the step on its own training rows.
        cv = IndexWalkForward(40, 20)
        p = cross_val_predict(Pipeline(; steps = (swb, EqualWeighted())), rd, cv)
        tr = split(cv, rd).train_idx
        for (i, pred) in pairs(p.pred)
            @test pred.res.wb.ub ≈ min.(1.0, hand_cap(tr[i], 0.006))
        end
    end
end
