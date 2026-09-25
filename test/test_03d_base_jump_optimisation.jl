# The docstrings of `src/17_Optimisation/05_JuMP/01_Base_JuMPOptimisation/` checked against
# numbers: the type families, the custom hooks, the model accessors, the Model State verbs,
# the frontier builders and the model setup. Most testsets build a model by hand and never
# solve it; the last ones solve a five-asset panel with Clarabel.
using Test, PortfolioOptimisers, JuMP, Clarabel, HiGHS, StableRNGs, LinearAlgebra
using InteractiveUtils: subtypes

# A subtype of each hook with no method of its own. A struct is defined at the top level.
struct HookConstraint03d <: PortfolioOptimisers.CustomJuMPConstraint end
struct HookObjective03d <: PortfolioOptimisers.CustomJuMPObjective end

@testset "Base JuMP optimisation against its docstrings" begin
    PO = PortfolioOptimisers

    @testset "the objective and return families have the stated members" begin
        # Four classic objectives and the internal corner objective.
        @test Set(subtypes(PO.ObjectiveFunction)) ==
              Set([MinimumRisk, MaximumReturn, MaximumUtility, MaximumRatio,
                   PO.MaximumElementReturn])
        @test Set(subtypes(PO.JuMPReturnsEstimator)) ==
              Set([ArithmeticReturn, LogarithmicReturn, NoReturn])
    end

    @testset "a custom hook with no method raises, and nothing is the only no-op" begin
        model = JuMP.Model()
        @test isnothing(PO.add_custom_constraint!(model, nothing, nothing, nothing))
        @test isnothing(PO.add_custom_objective_term!(model, MinimumRisk(), nothing,
                                                      nothing, nothing))
        @test_throws ArgumentError PO.add_custom_constraint!(model, HookConstraint03d(),
                                                             nothing, nothing)
        @test_throws ArgumentError PO.add_custom_objective_term!(model, MinimumRisk(),
                                                                 HookObjective03d(),
                                                                 nothing, nothing)
        # The vector methods reach the per-type method of each entry.
        @test_throws ArgumentError PO.add_custom_constraint!(model, [HookConstraint03d()],
                                                             nothing, nothing)
        @test_throws ArgumentError PO.add_custom_objective_term!(model, MinimumRisk(),
                                                                 [HookObjective03d()],
                                                                 nothing, nothing)
        @test !PO.needs_previous_weights(HookConstraint03d())
        @test !PO.needs_previous_weights([HookConstraint03d(), HookConstraint03d()])
        @test !PO.needs_previous_weights(HookObjective03d())
        @test !PO.needs_previous_weights([HookObjective03d()])
        @test isnothing(PO.port_opt_view(HookConstraint03d(), 1:2))
        @test isnothing(PO.port_opt_view(HookObjective03d(), 1:2))
        @test_throws IsEmptyError PO.JuMPOptimisationSolution(; w = Float64[])
    end

    @testset "every accessor raises on a model that lacks its entry" begin
        model = JuMP.Model()
        for f in
            (PO.get_constraint_scale, PO.get_objective_scale, PO.get_T, PO.get_w, PO.get_k,
             PO.get_ret, PO.get_risk, PO.get_X, PO.get_net_X, PO.get_Xap1, PO.get_ddap1,
             PO.get_dd)
            @test_throws ArgumentError f(model)
        end
        for f in (PO.has_X, PO.has_net_X, PO.has_Xap1, PO.has_ddap1, PO.has_dd)
            @test !f(model)
        end
        # The message names the builder that registers `:ret`.
        msg = try
            PO.get_ret(model)
        catch e
            e.msg
        end
        @test occursin("scalarise_return_expression!", msg)
        @test isnothing(PO.decomposition_contract(model))
        @test !PO.is_unit_budget(model)
    end

    @testset "scales, observation count, unit budget and decomposition contract" begin
        model = JuMP.Model()
        PO.set_model_scales!(model, 2.0, 3.0)
        PO.set_model_observations!(model, 7)
        # JuMP stores a number given to `@expressions` as the number itself.
        @test PO.get_constraint_scale(model) === 2.0
        @test PO.get_objective_scale(model) === 3.0
        @test PO.get_T(model) == 7
        JuMP.@variable(model, k)
        @test PO.effective_k(model) === k
        PO.set_unit_budget!(model)
        @test PO.is_unit_budget(model)
        @test PO.effective_k(model) == 1
        @test PO.get_k(model) === k
        # The first record stays.
        PO.set_decomposition_contract!(model, PO.WeightsFromParts())
        PO.set_decomposition_contract!(model, PO.PartsBoundWeights())
        @test PO.decomposition_contract(model) isa PO.WeightsFromParts
    end

    @testset "Model State: keys, the collision guard, build once and flags" begin
        model = JuMP.Model()
        p = Symbol("")
        @test PO.state_key(p, :ret_, 1) == :ret_1
        @test PO.state_key(:a_, :x) == :a_x
        # Two different sets of parts give one key, and the second registration raises.
        @test PO.state_key(p, :tr_dr_, 11) == PO.state_key(p, :tr_dr_1, 1)
        @test PO.state_set!(model, p, :tr_dr_1, 1, :first) == :first
        @test_throws ArgumentError PO.state_set!(model, p, :tr_dr_, 11, :second)
        @test model[:tr_dr_11] == :first
        @test PO.state_set!(model, :a_, :y, 5) == 5
        @test_throws ArgumentError PO.state_set!(model, :a_, :y, 6)
        @test PO.state_has(model, :a_, :y) && PO.state_has(model, p, :tr_dr_1, 1)
        @test PO.state_get(model, :a_, :y) == 5
        @test PO.state_get(model, p, :tr_dr_1, 1) == :first
        @test_throws ArgumentError PO.state_get(model, :a_, :z)
        @test_throws ArgumentError PO.state_get(model, :a_, :z, 1)
        # `f` runs once for each key.
        calls = Ref(0)
        f = () -> (calls[] += 1; calls[])
        @test PO.state_build!(f, model, :b_, :x) == 1
        @test PO.state_build!(f, model, :b_, :x) == 1
        @test PO.state_build!(f, model, :b_, :x, 2) == 2
        @test PO.state_build!(f, model, :b_, :x, 2) == 2
        @test calls[] == 2
        # A flag holds `true`, and a second mark has no effect.
        PO.mark_state!(model, :c_, :flag)
        PO.mark_state!(model, :c_, :flag)
        PO.mark_state!(model, :c_, :flag, 3)
        @test model[:c_flag] === true && model[:c_flag3] === true
        @test PO.nested_prefix(:a_, :tr_dr_) == :a_tr_dr_
        @test PO.nested_prefix(:a_, :tr_dr_, 3) == :a_tr_dr_3_
        @test PO.nested_index(:loss_, 2) == :loss_2
        @test PO.nested_index(:gain_, PO.nested_index(:loss_, 2)) == :gain_loss_2
        # The shared verbs refuse a name off the list, and `shared_set!` replaces an entry.
        @test :X ∉ PO.SHARED_STATE
        @test_throws ArgumentError PO.shared_get(model, :X)
        @test_throws ArgumentError PO.shared_has(model, :X)
        @test_throws ArgumentError PO.shared_set!(model, :X, 1)
        @test_throws ArgumentError PO.shared_get(model, :fees)
        @test PO.shared_set!(model, :fee_fa, nothing) === nothing
        @test PO.shared_set!(model, :fee_fa, PO.AmortisedFees()) isa PO.AmortisedFees
        @test PO.shared_get(model, :fee_fa) isa PO.AmortisedFees
    end

    @testset "frontier rows: the floor, the polarity and the count" begin
        model = JuMP.Model()
        PO.set_model_scales!(model, 2.0, 1.0)
        PO.set_maximum_ratio_factor_variables!(model, MinimumRisk())
        JuMP.@variable(model, x)
        @test PO.frontier_sweep_points(model) == (big(1), Pair{Symbol, Int}[])
        @test isnothing(PO.assert_frontier_sweep_cap(model))
        @test PO.frontier_point_count(Frontier(; N = 4)) == 4
        @test PO.frontier_point_count([1.0, 2.0]) == 2
        risk = [(:a_var, :a) => (1.0 * x, [1.0, 2.0], true, 0),
                (:b_var, :b) => (1.0 * x, [3.0], false, 0)]
        rax = PO.set_risk_frontier_parameters!(model, risk)
        # A `true` flag is the ceiling `x <= u k`, a `false` flag the floor `x >= u k`,
        # and `s_c = 2` multiplies both.
        ca = JuMP.constraint_object(model[:a])
        cb = JuMP.constraint_object(model[:b])
        @test ca.set == JuMP.MOI.LessThan(0.0) && cb.set == JuMP.MOI.LessThan(0.0)
        @test JuMP.coefficient(ca.func, x) == 2.0
        @test JuMP.coefficient(ca.func, model[:a_var]) == -2.0
        @test JuMP.coefficient(cb.func, x) == -2.0
        @test JuMP.coefficient(cb.func, model[:b_var]) == 2.0
        ret = [(:r_var, :r) => (3.0 * x, [0.5, 0.7, 0.9], 1)]
        tax = PO.set_ret_frontier_parameters!(model, ret)
        cr = JuMP.constraint_object(model[:r])
        @test cr.set == JuMP.MOI.GreaterThan(0.0)
        @test JuMP.coefficient(cr.func, x) == 6.0
        @test JuMP.coefficient(cr.func, model[:r_var]) == -2.0
        # The product: two risk points, one risk point and three return points.
        pts = collect(PO.frontier_sweep_axes(tax, rax))
        @test length(pts) == 6
        @test length(collect(PO.frontier_sweep_axes(tax, nothing))) == 3
        @test length(collect(PO.frontier_sweep_axes(nothing, rax))) == 2
        @test length(collect(PO.frontier_sweep_axes(nothing, nothing))) == 1
        # The risk axis changes fastest.
        PO.set_frontier_point!(model, pts[2])
        @test JuMP.parameter_value(model[:a_var]) == 2.0
        @test JuMP.parameter_value(model[:r_var]) == 0.5
        PO.set_frontier_point!(model, pts[3])
        @test JuMP.parameter_value(model[:a_var]) == 1.0
        @test JuMP.parameter_value(model[:r_var]) == 0.7
        @test JuMP.parameter_value(model[:b_var]) == 3.0
    end

    @testset "weights, start values and the return series" begin
        X = [0.01 0.02; -0.01 0.03; 0.02 -0.02; 0.0 0.01]
        model = JuMP.Model()
        PO.set_w!(model, X, [0.25, 0.75])
        w = PO.get_w(model)
        @test length(w) == 2
        @test JuMP.start_value.(w) == [0.25, 0.75]
        @test !JuMP.has_lower_bound(w[1]) && !JuMP.has_upper_bound(w[1])
        @test_throws DimensionMismatch PO.set_initial_w!(w, [1.0])
        # A vector of weight vectors, as a sweep of anchors passes, sets no start value.
        model2 = JuMP.Model()
        PO.set_w!(model2, X, [[0.5, 0.5], [0.2, 0.8]])
        @test all(isnothing, JuMP.start_value.(PO.get_w(model2)))
        Xw = PO.set_portfolio_returns!(model, X)
        @test PO.set_portfolio_returns!(model, X) === Xw
        @test PO.has_X(model) && PO.get_X(model) === Xw
        wv = [0.6, 0.4]
        val(e) = JuMP.value(v -> wv[JuMP.index(v).value], e)
        @test val.(Xw) ≈ X * wv
        net = PO.set_net_portfolio_returns!(model, X)
        # With no fees the net series is the gross one.
        @test val.(net) ≈ X * wv
        @test PO.has_net_X(model) && PO.get_net_X(model) === net
    end

    @testset "the net series charges the one-off fee on its clock" begin
        X = [0.01 0.02; -0.01 0.03; 0.02 -0.02; 0.0 0.01]
        wv = [0.6, 0.4]
        function net_series(fa)
            model = JuMP.Model()
            PO.set_model_scales!(model, 1.0, 1.0)
            PO.set_model_observations!(model, 4)
            JuMP.@variable(model, w[1:2])
            PO.add_to_fees!(model, 0.001 * w[1])
            PO.add_to_one_time_fees!(model, 0.004 * w[2])
            PO.shared_set!(model, :fee_fa, fa)
            net = PO.set_net_portfolio_returns!(model, X)
            @test PO.set_net_portfolio_returns!(model, X) === net
            return [JuMP.value(v -> wv[JuMP.index(v).value], e) for e in net]
        end
        gross = X * wv .- 0.001 * 0.6
        one_off = 0.004 * 0.4
        # The first observation pays the whole one-off fee on the default clock.
        @test net_series(nothing) ≈ gross .- one_off .* [1, 0, 0, 0]
        @test net_series(PO.FirstObservationFees()) ≈ gross .- one_off .* [1, 0, 0, 0]
        # Each of the four observations pays a quarter on the amortised clock.
        @test net_series(PO.AmortisedFees()) ≈ gross .- one_off / 4
    end

    @testset "the plus-one matrices" begin
        X = [0.01 0.02; -0.03 0.03; 0.02 -0.02]
        model = JuMP.Model()
        a = PO.set_asset_returns_plus_one!(model, X)
        @test a == X .+ 1
        @test PO.set_asset_returns_plus_one!(model, X) === a
        @test PO.has_Xap1(model) && PO.get_Xap1(model) === a
        b = PO.set_asset_neg_returns_plus_one!(model, X)
        @test b == 1 .- X
        @test PO.set_asset_neg_returns_plus_one!(model, X) === b
        d = PO.set_portfolio_drawdowns_plus_one!(model, X)
        # The drawdown of each column of the cumulative sum, the peak seeded at zero.
        c = cumsum(X; dims = 1)
        @test d ≈ 1 .+ c .- max.(0, accumulate(max, c; dims = 1))
        @test d ≈ [1.0 1.0; 0.97 1.0; 0.99 0.98]
        @test all(<=(1), d)
        @test PO.set_portfolio_drawdowns_plus_one!(model, X) === d
        @test PO.has_ddap1(model) && PO.get_ddap1(model) === d
        # The prefix keeps a second build apart.
        @test PO.set_asset_returns_plus_one!(model, -X; prefix = :gain_) == 1 .- X
        @test PO.get_Xap1(model, :gain_) == 1 .- X
    end

    @testset "the drawdown accessors read the drawdown builder" begin
        X = [0.01 0.02; -0.03 0.03; 0.02 -0.02]
        model = JuMP.Model()
        PO.set_model_scales!(model, 1.0, 1.0)
        PO.set_w!(model, X, nothing)
        @test !PO.has_dd(model)
        dd = PO.set_drawdown_constraints!(model, X)
        @test PO.has_dd(model) && PO.get_dd(model) === dd
        # One more variable than observations.
        @test length(dd) == size(X, 1) + 1
    end

    X = 0.01 * randn(StableRNG(42), 120, 5) .+ 0.001 * (1:5)'
    rd = ReturnsResult(; nx = ["A", "B", "C", "D", "E"], X = X)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = "verbose" => false)
    bad = Solver(; name = :bad, solver = () -> error("no solver"),
                 check_sol = (; allow_local = true, allow_almost = true))

    @testset "the solver loop falls back, and the weights are divided by k" begin
        res = optimise(MeanRisk(; r = Variance(), obj = MaximumRatio(),
                                opt = JuMPOptimiser(; slv = [bad, slv])), rd)
        @test res.retcode isa OptimisationSuccess
        # Only the solver that failed has an entry.
        @test collect(keys(res.retcode.res)) == [:bad]
        k = JuMP.value(res.model[:k])
        @test k > 0 && !isapprox(k, 1)
        @test res.w ≈ JuMP.value.(res.model[:w]) / k
        @test sum(res.w) ≈ 1
        # The default `getproperty` forwards `w` and the fields of `pa` to `jr`.
        @test res.w === getfield(res, :jr).w
        @test res.r === getfield(res, :r)
        @test :w in propertynames(res) && :pa in propertynames(res)
        # Every solver fails, so the weights are `NaN`.
        all_bad = MeanRisk(; r = Variance(), opt = JuMPOptimiser(; slv = bad))
        failed = @test_logs (:warn, r"Failed to solve") match_mode = :any optimise(all_bad,
                                                                                   rd)
        @test failed.retcode isa OptimisationFailure
        @test length(failed.w) == 5 && all(isnan, failed.w)
    end

    @testset "a solver that raises in optimize! is recorded, and the next one solves" begin
        # HiGHS takes no conic row without bridges, so `optimize!` raises.
        hs = Solver(; name = :highs, solver = HiGHS.Optimizer, add_bridges = false,
                    settings = "log_to_console" => false,
                    check_sol = (; allow_local = true, allow_almost = true))
        res = optimise(MeanRisk(; r = Variance(), opt = JuMPOptimiser(; slv = [hs, slv])),
                       rd)
        @test res.retcode isa OptimisationSuccess
        @test collect(keys(res.retcode.res)) == [:highs]
        @test collect(keys(res.retcode.res[:highs])) == [:optimize!]
    end

    @testset "a reset removes every time-dependent field" begin
        mr = MeanRisk(; r = Variance(),
                      obj = TimeDependent([MinimumRisk(), MaximumRatio()]),
                      opt = JuMPOptimiser(; slv = slv))
        @test PO.is_time_dependent(mr)
        reset = PO.reset_time_dependent_estimator(mr)
        @test !PO.is_time_dependent(reset)
        @test reset.obj isa PO.ObjectiveFunction
        # A static optimiser comes back unchanged.
        @test PO.reset_time_dependent_estimator(reset) === reset
    end

    @testset "a return frontier sweeps its own points" begin
        ret = ArithmeticReturn(;
                               settings = PO.JuMPReturnsSettings(; lb = Frontier(; N = 3)))
        res = optimise(MeanRisk(; r = Variance(), obj = MinimumRisk(),
                                opt = JuMPOptimiser(; slv = slv, ret = ret)), rd)
        @test length(res.w) == 3
        mu = vec(sum(X; dims = 1)) / size(X, 1)
        rets = [dot(mu, w) for w in res.w]
        @test issorted(rets)
        # The last point binds its floor.
        lb = JuMP.parameter_value(res.model[:ret_lb_var_1])
        @test rets[end] >= lb - 1e-8
        @test isapprox(rets[end], lb; rtol = 1e-6)
    end
end
