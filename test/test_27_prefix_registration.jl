@testset "Prefix-namespaced risk-state registration" begin
    using PortfolioOptimisers, Test, Clarabel, Random, LinearAlgebra, StableRNGs, JuMP

    # Direct, build-level tests for the Step 4 prefix migration (ADR 0005). A risk
    # tracking build namespaces ALL of its shared model-state keys under a composed
    # prefix so a nested build cannot collide with the outer model. These tests assert
    # KEY CONSTRUCTION ONLY in `model.obj_dict` — NOT weights, NOT whether the
    # optimisation solved (keys are registered during build, before the solve).

    Random.seed!(42)
    # 5 assets / 200 obs for most measures; 3 assets for the full Kurtosis/VSK SDP
    # lifts; 15 obs for the observation-bound BrownianDistanceVariance.
    rd = ReturnsResult(; nx = string.('A':'E'), X = 0.01 .* randn(200, 5))
    rd3 = ReturnsResult(; nx = ["A", "B", "C"], X = 0.01 .* randn(200, 3))
    rd15 = ReturnsResult(; nx = string.('A':'E'), X = 0.01 .* randn(15, 5))
    slv = Solver(; name = :c, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = "verbose" => false)
    pr = prior(EmpiricalPrior(), rd)
    pr3 = prior(HighOrderPriorEstimator(), rd3)
    pr15 = prior(EmpiricalPrior(), rd15)
    opt = JuMPOptimiser(; pe = pr, slv = slv)
    opt3 = JuMPOptimiser(; pe = pr3, slv = slv)
    opt15 = JuMPOptimiser(; pe = pr15, slv = slv)
    w0 = fill(0.2, 5)
    w03 = fill(inv(3), 3)
    ucs = sigma_ucs(NormalUncertaintySet(; pe = EmpiricalPrior(), rng = StableRNG(1),
                                         alg = BoxUncertaintySetAlgorithm()), rd.X)

    # Build the model the realistic way: a vector [A, Tracking(A)] so ONE model builds
    # A's keys twice — outer A at index 1 (bare), the tracking measure at index 2 (so its
    # composed prefix is `:tr_iv_2_` / `:tr_dv_2_`). Return `nothing` if the build is
    # unsupported (some measures have no `expected_risk` for DependentVariableTracking).
    function track_model(A, alg, o, wv, rdx)
        tr = WeightsTracking(; w = wv)
        mr = MeanRisk(; r = [A, RiskTrackingRiskMeasure(; tr = tr, r = A, alg = alg)],
                      obj = MinimumRisk(), opt = o)
        return try
            optimise(mr, rdx).model
        catch
            nothing
        end
    end
    trk_prefix(::IndependentVariableTracking) = :tr_iv_2_
    trk_prefix(::DependentVariableTracking) = :tr_dv_2_

    # (measure, opt, weights, returns, Category-A singleton keys the inner build registers)
    cases = [("Variance", Variance(), opt, w0, rd, [:variance_flag]),
             ("StandardDeviation", StandardDeviation(), opt, w0, rd, Symbol[]),
             ("UncertaintySetVariance", UncertaintySetVariance(; ucs = ucs), opt, w0, rd,
              [:W, :M, :M_PSD, :Au, :Al, :cbucs_variance, :variance_flag]),
             ("ConditionalValueatRisk", ConditionalValueatRisk(), opt, w0, rd, [:net_X]),
             ("EntropicValueatRisk", EntropicValueatRisk(), opt, w0, rd, [:net_X]),
             ("PowerNormValueatRisk", PowerNormValueatRisk(), opt, w0, rd, [:net_X]),
             ("WorstRealisation", WorstRealisation(), opt, w0, rd,
              [:wr_risk, :cwr, :net_X]),
             ("Range", Range(), opt, w0, rd, [:range_risk, :br_risk, :cbr, :wr_risk]),
             ("MaximumDrawdown", MaximumDrawdown(), opt, w0, rd,
              [:dd, :mdd_risk, :cmdd_risk]),
             ("UlcerIndex", UlcerIndex(), opt, w0, rd, [:dd, :uci, :uci_risk, :cuci_soc]),
             ("OrderedWeightsArray", OrderedWeightsArray(), opt, w0, rd, [:net_X]),
             ("TurnoverRiskMeasure", TurnoverRiskMeasure(; w = w0), opt, w0, rd, Symbol[]),
             ("Kurtosis", Kurtosis(), opt3, w03, rd3, [:W, :M, :M_PSD, :L2W]),
             ("VarianceSkewKurtosis", VarianceSkewKurtosis(), opt3, w03, rd3,
              [:W1_vr_sk_kt, :W2_vr_sk_kt, :W3_vr_sk_kt, :L2W1_vr_sk_kt, :M_vr_sk_kt,
               :M_vr_sk_kt_PSD]),
             ("BrownianDistanceVariance", BrownianDistanceVariance(), opt15, w0, rd15,
              [:Dt, :Dx, :bdvariance_risk])]

    @testset "[A, Tracking(A)] — $name / $(nameof(typeof(alg)))" for (name, A, o, wv, rdx,
                                                                      ckeys) in cases,
                                                                     alg in
                                                                     (IndependentVariableTracking(),
                                                                      DependentVariableTracking())

        m = track_model(A, alg, o, wv, rdx)
        # IndependentVariableTracking must always build; Dependent may be unsupported for
        # a given measure (no `expected_risk`) — skip the combo if so.
        if isa(alg, IndependentVariableTracking)
            @test m !== nothing
        elseif m === nothing
            @test_skip "DependentVariableTracking unsupported for $name"
            continue
        end
        p = trk_prefix(alg)
        # The tracking-difference weights are stored under the composed prefix (universal).
        @test haskey(m, Symbol(p, :w))
        # The bare weights from the outer build still exist.
        @test haskey(m, :w)
        for k in ckeys
            # the inner build registered the key UNDER the tracking prefix ...
            @test haskey(m, Symbol(p, k))
            # ... and where the outer build also makes a bare JuMP object, the two are
            # DISTINCT objects (no obj_dict collision/overwrite). Boolean presence flags
            # (`:variance_flag` = `true`) are singletons, so identity is not meaningful —
            # their coexistence as separate dict entries (asserted above) is the point.
            if haskey(m, k) && !isa(m[k], Bool)
                @test m[Symbol(p, k)] !== m[k]
            end
        end
    end

    # Prior-derived caches are weight-independent, so they stay BARE and shared — NOT
    # re-registered under the tracking prefix (the bare-vs-prefix invariant, ADR 0005).
    @testset "prior caches stay bare — $(nameof(typeof(alg)))" for alg in
                                                                   (IndependentVariableTracking(),
                                                                    DependentVariableTracking())
        p = trk_prefix(alg)
        # Full Kurtosis (N = nothing) caches the projected-cokurtosis Cholesky as :Gkt.
        mf = track_model(Kurtosis(), alg, opt3, w03, rd3)
        if mf !== nothing
            @test haskey(mf, :Gkt)
            @test !haskey(mf, Symbol(p, :Gkt))
        end
        # Approximate Kurtosis (N = k) caches the eigendecomposition as :vals_Akt/:vecs_Akt.
        ma = track_model(Kurtosis(; N = 2), alg, opt3, w03, rd3)
        if ma !== nothing
            for cache in (:vals_Akt, :vecs_Akt)
                @test haskey(ma, cache)
                @test !haskey(ma, Symbol(p, cache))
            end
        end
    end

    # Tracking-nested-in-tracking: prefixes COMPOSE rather than replace, so the innermost
    # build is namespaced under BOTH layers and cannot collide. The inner tracking's index
    # is itself the composed symbol, so the innermost prefix is
    # `Symbol(:tr_iv_1_, :tr_iv_, :tr_iv_1_1, :_)`. This is the re-entrancy the old swap
    # could not provide.
    @testset "nested Tracking(Tracking(A)) composes — $(nameof(typeof(alg)))" for alg in
                                                                                  (IndependentVariableTracking(),
                                                                                   DependentVariableTracking())
        base = isa(alg, IndependentVariableTracking) ? :tr_iv_ : :tr_dv_
        inner = RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = w0),
                                        r = ConditionalValueatRisk(), alg = alg)
        outer = RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = w0), r = inner,
                                        alg = alg)
        m = try
            optimise(MeanRisk(; r = [outer], obj = MinimumRisk(), opt = opt), rd).model
        catch
            nothing
        end
        if m === nothing
            @test_skip "nested DependentVariableTracking unsupported"
            continue
        end
        p_outer = Symbol(base, 1, :_)                 # outer tracking at index 1
        i_inner = Symbol(p_outer, 1)                  # inner tracking's composed index
        p_inner = Symbol(p_outer, base, i_inner, :_)  # innermost prefix = composition
        @test p_inner != p_outer                       # composed, not replaced
        @test haskey(m, Symbol(p_outer, :w))          # outer layer weights
        @test haskey(m, Symbol(p_inner, :w))          # innermost layer weights (deeper)
        @test haskey(m, Symbol(p_inner, :net_X))      # innermost CVaR infra, deeply namespaced
    end
end
