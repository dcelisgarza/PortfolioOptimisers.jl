# The docstrings of `src/17_Optimisation/01_Base_Optimisation/` checked against numbers: the
# type families and their fallbacks, the routing of Pipeline targets, the fold schedules, the
# weight finalisers, the panel collapse and the investable view of a result. The finaliser
# testsets solve three-asset programmes with Clarabel; the others build no model.
using Test, PortfolioOptimisers, Clarabel, LinearAlgebra
using InteractiveUtils: subtypes

# A result with no prior, no fees and no mask, which reaches every fallback of the family.
struct BareResult03e <: PortfolioOptimisers.OptimisationResult
    w::Vector{Float64}
end
# A keyword constructor that records each call, for the substitution check.
const SUB03E_LOG = Any[]
struct Sub03e
    a::Any
    b::Any
end
function Sub03e(; a, b)
    push!(SUB03E_LOG, (a, b))
    return Sub03e(a, b)
end

@testset "Base optimisation against its docstrings" begin
    PO = PortfolioOptimisers
    # The concrete subtypes of `T` that the package defines. Another test file in the same
    # process can define subtypes of its own, which `subtypes` also returns.
    function leaves(T)
        out = Any[]
        for S in subtypes(T)
            if isabstracttype(S)
                append!(out, leaves(S))
            elseif parentmodule(S) === PO
                push!(out, S)
            end
        end
        return out
    end
    rd = ReturnsResult(; nx = ["A", "B", "C"],
                       X = [0.01 0.02 -0.01; 0.0 0.01 0.02; -0.01 0.0 0.01; 0.02 -0.01 0.0])
    slv = Solver(; solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false, "tol_gap_abs" => 1e-10,
                                 "tol_gap_rel" => 1e-10, "tol_feas" => 1e-10))

    @testset "the result families have the stated shape" begin
        # `fb` is the last field of every concrete result, and `factory(res, fb)` needs it.
        for R in leaves(PO.OptimisationResult)
            @test last(fieldnames(R)) === :fb
        end
        # A return code has the one field `res`.
        for R in leaves(PO.OptimisationReturnCode)
            @test fieldnames(R) == (:res,)
        end
        # The hierarchical family holds the three results of a `HierarchicalOptimiser`, and
        # `NestedClustered` is one of the four clustering estimators but not a member.
        @test Set(leaves(PO.HierarchicalOptimisationResult)) ==
              Set([HierarchicalRiskParityResult, HierarchicalEqualRiskContributionResult,
                   SchurComplementHierarchicalRiskParityResult])
        @test length(leaves(PO.ClusteringOptimisationEstimator)) == 4
        @test NestedClustered in leaves(PO.ClusteringOptimisationEstimator)
        @test JuMPOptimiser <: PO.BaseOptimisationEstimator
        @test PO.HierarchicalOptimiser <: PO.BaseOptimisationEstimator
    end

    @testset "the fallbacks of the family" begin
        res = BareResult03e([0.5, 0.5])
        @test isnothing(PO.result_investable_mask(res))
        # One fallback of the reset serves every estimator and every result.
        @test PO.reset_time_dependent_estimator(res) === res
        ga = GreedyAllocation()
        @test PO.reset_time_dependent_estimator(ga) === ga
        # A result with no method of `set_retcode` throws, and names its type.
        nres = optimise(EqualWeighted(), rd)
        err = try
            PO.set_retcode(nres, OptimisationFailure())
        catch e
            e
        end
        @test isa(err, ArgumentError)
        @test occursin("NaiveOptimisationResult", err.msg)
        # A precomputed result in `fb` keeps its identity when `factory` gives the estimator
        # the previous weights.
        @test factory(nres, [0.2, 0.3, 0.5]) === nres
        mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv), fb = nres)
        @test factory(mr, [0.2, 0.3, 0.5]).fb === nres
        # `factory(res, fb)` replaces the last field and keeps the others.
        chain = [(EqualWeighted(), nres)]
        nres2 = factory(nres, chain)
        @test nres2.fb === chain
        @test nres2.w == nres.w
        # `needs_previous_weights` falls back to `false` for nothing, an estimator and an
        # algorithm.
        @test !PO.needs_previous_weights(nothing)
        @test !PO.needs_previous_weights(EqualWeighted())
        @test !PO.needs_previous_weights(nres)
        @test isnothing(PO.assert_special_nco_requirements(nres))
    end

    @testset "the routing of Pipeline targets" begin
        rb = RiskBudgeting(; opt = JuMPOptimiser(; slv = slv))
        @test PO.pipe_config_field(rb) === :opt
        @test isnothing(PO.pipe_config_field(EqualWeighted()))
        # `:rkb` lands one level down, in `rba.rkb`.
        rkb = PO.risk_budget_constraints(nothing; N = 3)
        @test PO.pipe_accepts(rb, Val(:rkb))
        @test PO.pipe_route(rb, Val(:rkb), rkb).rba.rkb === rkb
        # A schedule in `rba` has no `rkb`, so the optimiser refuses the target.
        rbt = RiskBudgeting(; opt = JuMPOptimiser(; slv = slv),
                            rba = TimeDependent([AssetRiskBudgeting(),
                                                 AssetRiskBudgeting()]))
        @test !PO.pipe_accepts(rbt, Val(:rkb))
        @test_throws ArgumentError PO.pipe_route(rbt, Val(:rkb), rkb)
        # A target with the name of a field lands in that field.
        @test PO.pipe_accepts(EqualWeighted(), Val(:wb))
        wb = WeightBounds(; lb = 0.0, ub = 0.6)
        @test PO.pipe_route(EqualWeighted(), Val(:wb), wb).wb === wb
    end

    @testset "a schedule resolves as its docstring states" begin
        ctx = TimeDependentContext(; i = 2, n = 3, rd = rd, train_idx = [1:2, 1:3, 1:4],
                                   test_idx = [3:3, 4:4, 4:4])
        # A vector reads entry `ctx.i`, a function and a wrapped function read the context.
        @test PO.time_dependent_value(TimeDependent([10, 20, 30]), ctx) == 20
        @test PO.time_dependent_value(TimeDependent(c -> 5 * c.i), ctx) == 10
        pwf = PreviousWeightsFunction(; f = c -> c.n)
        @test PO.time_dependent_value(TimeDependent(pwf), ctx) == 3
        @test PO.needs_previous_weights(pwf)
        @test PO.needs_previous_weights(TimeDependent(pwf))
        @test !PO.needs_previous_weights(TimeDependent(c -> 1))
        @test_throws DomainError TimeDependentContext(; i = 4, n = 3, rd = rd,
                                                      train_idx = [], test_idx = [])
        # The constructor refuses the stated values.
        @test_throws IsEmptyError TimeDependent([])
        @test_throws ArgumentError TimeDependent(TimeDependent([1, 2]))
        @test_throws ArgumentError TimeDependent([TimeDependent([1, 2]), 3])
        @test_throws ArgumentError TimeDependent([1, 2]; default = TimeDependent([1, 2]))
        @test_throws ArgumentError TimeDependent([1, 2], :innermost)
        # A vector of optimisers and results is stored as a `Vector{OptE_Opt}`.
        nres = optimise(EqualWeighted(), rd)
        mixed = TimeDependent(Any[EqualWeighted(), nres])
        @test eltype(mixed.val) == PO.OptE_Opt
        @test isa(mixed, PO.TD_OptE_Opt)
        # The slice of a schedule slices its entries and its default, and a function passes.
        td = TimeDependent([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]; default = [7.0, 8.0, 9.0])
        tv = PO.nothing_scalar_array_view(td, [1, 3])
        @test tv.val == [[1.0, 3.0], [4.0, 6.0]]
        @test tv.default == [7.0, 9.0]
        f = c -> 1
        @test PO.nothing_scalar_array_view(TimeDependent(f), [1]).val === f
        # `factory` reaches the entries and the default.
        tf = factory(TimeDependent([nres, nres]; default = nres), [0.2, 0.3, 0.5])
        @test all(x -> x === nres, tf.val)
        @test tf.default === nres
        # The known entries are the vector and the default, and a function has none.
        @test length(PO.time_dependent_entries(TimeDependent([1, 2]; default = 3))) == 3
        @test isempty(PO.time_dependent_entries(TimeDependent(f)))
    end

    @testset "the value outside every fold loop, and the stand-in" begin
        # The reset value: the default of the schedule, then the static default, then
        # `nothing` for a field that the defaults do not list.
        defaults = (; a = 5, b = NoDefault())
        @test PO.time_dependent_reset_value(TimeDependent([1, 2]; default = 9), defaults,
                                            :b, EqualWeighted()) == 9
        @test PO.time_dependent_reset_value(TimeDependent([1, 2]), defaults, :a,
                                            EqualWeighted()) == 5
        @test isnothing(PO.time_dependent_reset_value(TimeDependent([1, 2]), defaults, :c,
                                                      EqualWeighted()))
        @test_throws TimeDependentDefaultError PO.time_dependent_reset_value(TimeDependent([1,
                                                                                            2]),
                                                                             defaults, :b,
                                                                             EqualWeighted())
        # The stand-in: the default, the static default, the first entry, else nothing.
        @test something(PO.time_dependent_stand_in(TimeDependent([1, 2]; default = 9),
                                                   defaults, :b)) == 9
        @test something(PO.time_dependent_stand_in(TimeDependent([1, 2]), defaults, :a)) ==
              5
        @test something(PO.time_dependent_stand_in(TimeDependent([1, 2]), defaults, :b)) ==
              1
        @test isnothing(PO.time_dependent_stand_in(TimeDependent(c -> 1), defaults, :b))
        # The static defaults fall back to an empty tuple.
        @test PO.time_dependent_field_defaults(BareResult03e([1.0])) == (;)
        # A schedule in the place of an optimiser must carry a default.
        @test_throws TimeDependentDefaultError PO.reset_time_dependent_estimator(TimeDependent([EqualWeighted(),
                                                                                                EqualWeighted()]))
        ew = EqualWeighted()
        @test PO.reset_time_dependent_estimator(TimeDependent([ew, ew]; default = ew)) ===
              ew
    end

    @testset "the substitution check calls the constructor once for each entry" begin
        empty!(SUB03E_LOG)
        args = (; a = TimeDependent([1, 2]; default = 3), b = TimeDependent([4, 5]))
        @test isnothing(PO.assert_time_dependent_substitution(Sub03e, args, (; b = 6)))
        # `a` runs its two entries and its default with `b` at its static default 6, then
        # `b` runs its two entries with `a` at its default 3.
        @test SUB03E_LOG == [(1, 6), (2, 6), (3, 6), (3, 4), (3, 5)]
        # A function schedule in a required field has no stand-in, so nothing is called.
        empty!(SUB03E_LOG)
        args = (; a = TimeDependent(c -> 1), b = TimeDependent([4, 5]))
        @test isnothing(PO.assert_time_dependent_substitution(Sub03e, args,
                                                              (; a = NoDefault())))
        @test isempty(SUB03E_LOG)
        @test isnothing(PO.substitute_time_dependent_entries(Sub03e, (; a = 1), :a, 1))
        @test isempty(SUB03E_LOG)
    end

    @testset "the fold count and the bind checks" begin
        ew = EqualWeighted()
        td = TimeDependent([ew, ew])
        @test isnothing(PO.assert_time_dependent_fold_count(td, 2))
        @test_throws DimensionMismatch PO.assert_time_dependent_fold_count(td, 3)
        # A schedule with `:nearest` is left for the nearest loop when `all_binds` is false.
        tdn = TimeDependent([ew, ew], :nearest)
        @test isnothing(PO.assert_time_dependent_fold_count(tdn, 3, false))
        @test_throws DimensionMismatch PO.assert_time_dependent_fold_count(tdn, 3, true)
        ctx = TimeDependentContext(; i = 1, n = 2, rd = rd, train_idx = [1:2, 1:3],
                                   test_idx = [3:3, 4:4])
        @test PO.update_time_dependent_estimator(tdn, ctx, false) === tdn
        @test PO.update_time_dependent_estimator(tdn, ctx, true) === ew
        # A function in the place of an optimiser that returns no optimiser is refused.
        @test_throws ArgumentError PO.update_time_dependent_estimator(TimeDependent(c -> 1),
                                                                      ctx)
        # The guards of `:nearest` in a field with no inner fold loop, and in one with.
        @test_throws ArgumentError PO.assert_no_nearest_bind_optimiser_schedule(tdn, :fb,
                                                                                :MeanRisk)
        @test isnothing(PO.assert_no_nearest_bind_optimiser_schedule(td, :fb, :MeanRisk))
        @test_throws TimeDependentDefaultError PO.assert_nearest_optimiser_schedule(tdn,
                                                                                    :opti,
                                                                                    KFold(),
                                                                                    :NestedClustered)
        tdd = TimeDependent([ew, ew], :nearest; default = ew)
        @test_throws ArgumentError PO.assert_nearest_optimiser_schedule(tdd, :opti, nothing,
                                                                        :NestedClustered)
        @test isnothing(PO.assert_nearest_optimiser_schedule(tdd, :opti, KFold(),
                                                             :NestedClustered))
        # A configuration with no schedule has no candidate field.
        @test PO.time_dependent_candidate_fields(JuMPOptimiser(; slv = slv)) == ()
    end

    @testset "the four JuMP formulations give their hand minimisers" begin
        # w0 = [0.7, 0.2, 0.1] under ub = 0.5 moves a mass of 0.2 to the two free weights.
        wb = WeightBounds(; lb = 0.0, ub = 0.5)
        w0 = [0.7, 0.2, 0.1]
        fin(alg) = PO.opt_weight_bounds(JuMPWeightFinaliser(; slv = slv, alg = alg), wb, w0)
        # The L2 norm of the absolute deviation shifts the free weights by one constant.
        @test isapprox(fin(SquaredAbsoluteErrorWeightFinaliser()), [0.5, 0.3, 0.2];
                       atol = 1e-6)
        # The L1 norm of the relative deviation moves the mass to the largest free weight.
        @test isapprox(fin(RelativeErrorWeightFinaliser()), [0.5, 0.4, 0.1]; atol = 1e-6)
        # The L2 norm of the relative deviation gives w0 + c w0², with c = 4.
        @test isapprox(fin(SquaredRelativeErrorWeightFinaliser()), [0.5, 0.36, 0.14];
                       atol = 1e-6)
        # The L1 norm of the absolute deviation is not unique, and its value is twice the
        # moved mass.
        wa = fin(AbsoluteErrorWeightFinaliser())
        @test isapprox(sum(abs, wa - w0), 0.4; atol = 1e-6)
        @test isapprox(sum(wa), 1.0; atol = 1e-8)
        @test all(x -> -1e-8 <= x <= 0.5 + 1e-8, wa)
        # The Euclidean projection is the minimiser of the absolute L2 norm.
        @test isapprox(PO.opt_weight_bounds(EuclideanWeightFinaliser(), wb, w0),
                       [0.5, 0.3, 0.2]; atol = 1e-12)
        # A relative formulation does not change the weights of the caller.
        wz = [0.7, 0.3, 0.0]
        PO.opt_weight_bounds(JuMPWeightFinaliser(; slv = slv,
                                                 alg = SquaredRelativeErrorWeightFinaliser()),
                             wb, wz)
        @test wz == [0.7, 0.3, 0.0]
        # Weights in the bounds come back unchanged, with no solve.
        wi = [0.4, 0.4, 0.2]
        @test PO.opt_weight_bounds(JuMPWeightFinaliser(; slv = slv), wb, wi) === wi
    end

    @testset "a failed solve and the iterative finaliser" begin
        wb = WeightBounds(; lb = 0.0, ub = 0.4)
        # The iterative loop stalls on this input, and step 9 returns the projection.
        @test PO.opt_weight_bounds(IterativeWeightFinaliser(), wb, [0.6, 0.4, 0.0]) ≈
              [0.4, 0.4, 0.2]
        # A solver that cannot start makes the JuMP finaliser warn and use the iterative one.
        bad = JuMPWeightFinaliser(; slv = Solver(; solver = () -> error("no solver")))
        w = @test_logs((:warn, r"IterativeWeightFinaliser"), match_mode = :any,
                       PO.opt_weight_bounds(bad, wb, [0.6, 0.4, 0.0]))
        @test w ≈ [0.4, 0.4, 0.2]
        # A set of bounds that cannot hold the budget fails.
        rc, w = PO.finalise_weight_bounds(IterativeWeightFinaliser(),
                                          WeightBounds(; lb = 0.3, ub = 1.0), fill(0.25, 4))
        @test isa(rc, OptimisationFailure)
        @test w ≈ fill(0.3, 4)
        rc, w = PO.finalise_weight_bounds(IterativeWeightFinaliser(), wb, [0.6, 0.4, 0.0])
        @test isa(rc, OptimisationSuccess)
        @test sum(w) ≈ 1.0
    end

    @testset "the collapse of the panel" begin
        # The normalised weights are |W| over the gross exposure of each column.
        @test PO.synthetic_asset_weights([0.5, -0.25, 0.25]) ≈ [0.5, 0.25, 0.25]
        Wn = PO.synthetic_asset_weights([0.5 0.0; -0.5 0.0; 1.0 0.0])
        @test Wn ≈ [0.25 0.0; 0.25 0.0; 0.5 0.0]
        @test PO.synthetic_asset_weights(zeros(3)) == zeros(3)
        W = PO.synthetic_asset_weights([0.5 0.0; 0.5 1.0; 0.0 0.0])
        @test PO.collapse_panel_numeric([1.0, 2.0, 4.0], W) ≈ [1.5, 2.0]
        # A mask collapses as its support, and a categorical mask repeats over the levels.
        @test PO.collapse_categorical_mask(BitVector([true, false, false]), W, 3) ==
              Bool[1 1 1; 0 0 0]
        cm = PO.collapse_categorical_mask(BitMatrix([true false false; false true false]),
                                          W, 2)
        @test size(cm) == (2, 2, 2)
        @test cm[:, :, 1] == Bool[1 0; 1 1]
        @test cm[:, :, 2] == cm[:, :, 1]
        @test isnothing(PO.collapse_categorical_mask(nothing, W, 2))
    end

    @testset "the investable view of a result" begin
        nres = optimise(EqualWeighted(), rd)
        # An explicit prior has priority, and a result with no prior throws.
        @test PO.extract_pr(nres, nres.pr) === nres.pr
        @test PO.extract_pr(nres) === nres.pr
        @test_throws ArgumentError PO.extract_pr(BareResult03e([1.0]))
        @test isnothing(PO.extract_fees(BareResult03e([1.0])))
        fees = Fees(; l = 0.001)
        @test PO.extract_fees(BareResult03e([1.0]), fees) === fees
        # A result whose mask is nothing views nothing.
        imsk, w, X, f, nx = PO.result_investable_view(BareResult03e([0.5, 0.5]),
                                                      [1.0 2.0; 3.0 4.0], nothing,
                                                      ["a", "b"])
        @test isnothing(imsk)
        @test w == [0.5, 0.5]
        @test X == [1.0 2.0; 3.0 4.0]
        @test nx == ["a", "b"]
    end
end
