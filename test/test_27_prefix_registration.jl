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
    # The entries that belong to the weights rather than to the build, `weights_prefix`.
    weights_owned = (:W, :M, :M_PSD, :variance_flag, :rc_variance)

    # (measure, opt, weights, returns, Category-A singleton keys the inner build registers)
    cases = [("Variance", Variance(), opt, w0, rd, Symbol[]),
             ("StandardDeviation", StandardDeviation(), opt, w0, rd, Symbol[]),
             ("UncertaintySetVariance", UncertaintySetVariance(; ucs = ucs), opt, w0, rd,
              [:W, :M, :M_PSD, :Au, :Al, :cbucs_variance]),
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

    tr = WeightsTracking(; w = w03)
    mr = MeanRisk(;
                  r = [VarianceSkewKurtosis(),
                       RiskTrackingRiskMeasure(; tr = tr, r = VarianceSkewKurtosis(),
                                               alg = DependentVariableTracking())],
                  obj = MinimumRisk(), opt = opt3)

    Skewness{MaxRiskMeasureSettings{Float64, Nothing, Bool},
             SimpleVariance{SimpleExpectedReturns{Nothing}, Nothing, Bool}, Matrix{Float64},
             Nothing, Vector{Float64}}

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
        # The tracking constructor clears the inner `rke`, so an inner variance is no term
        # of the objective and marks no `variance_flag` anywhere (#1305).
        @test !haskey(m, Symbol(p, :variance_flag))
        # The tracking-difference weights are stored under the composed prefix (universal).
        @test haskey(m, Symbol(p, :w))
        # The bare weights from the outer build still exist.
        @test haskey(m, :w)
        for k in ckeys
            # A dependent build tracks the head's own weights, so the lifted matrix and the
            # variance marks belong to the head (ADR 0005 amendment of 2026-09-24, #1305):
            # the inner build reuses the bare entries and registers none under its prefix.
            if isa(alg, DependentVariableTracking) && k in weights_owned
                @test !haskey(m, Symbol(p, k))
                @test haskey(m, k)
                continue
            end
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
        # FullMoment Kurtosis (N = nothing) caches the projected-cokurtosis Cholesky as :Gkt.
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
    # build is namespaced under BOTH layers and cannot collide. This is the re-entrancy the
    # old swap could not provide.
    #
    # The inner tracking builds at measure index 1 — its own position in the nested build —
    # NOT at an index seeded with the enclosing prefix. Separation across builds is the
    # prefix's job alone, so the innermost prefix is `Symbol(:tr_iv_1_, :tr_iv_, 1, :_)`.
    # Before ADR 0037's amendment the index carried the prefix too, and the same fact was
    # spelled on both axes.
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
        p_outer = Symbol(base, 1, :_)            # outer tracking at index 1
        p_inner = Symbol(p_outer, base, 1, :_)   # innermost prefix = composition
        @test p_inner != p_outer                 # composed, not replaced
        @test haskey(m, Symbol(p_outer, :w))     # outer layer weights
        @test haskey(m, Symbol(p_inner, :w))     # innermost layer weights (deeper)
        @test haskey(m, Symbol(p_inner, :net_X)) # innermost CVaR infra, deeply namespaced
        # The measure index is NOT seeded with the prefix: the innermost CVaR's own scratch
        # sits at index 1 under the innermost prefix, on the other axis entirely.
        @test haskey(m, Symbol(p_inner, :cvar_risk_1))
        @test !haskey(m, Symbol(:cvar_risk_, p_inner, 1))
    end

    # #1305: a dependent build registers the weights of its enclosing build, so it records
    # their owner and reads the owner's lifted matrix. A semidefinite phylogeny on the head
    # then constrains the inner variance. An independent build shifts the weights, so it
    # keeps its own `W`, and a dependent build inside it reads that one. The phylogeny's
    # `p·tr(W)` penalty is omitted only for a variance that the objective minimises: a
    # ceiling and a tracking variance put no price on the growth of `W`.
    @testset "DependentVariableTracking reads the lifted matrix of its weights (#1305)" begin
        A5 = zeros(Int, 5, 5)
        A5[1, 2] = A5[2, 1] = 1
        optp = JuMPOptimiser(; pe = pr, slv = slv,
                             ple = SemiDefinitePhylogeny(; A = A5, p = 0.05))
        dv(r) = RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = w0), r = r,
                                        alg = DependentVariableTracking())
        iv(r) = RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = w0), r = r,
                                        alg = IndependentVariableTracking())
        build(rs; obj = MinimumRisk()) = optimise(MeanRisk(; r = rs, obj = obj, opt = optp),
                                                  rd).model
        reads(m, expr, W) = issubset(keys(m[expr].terms), Set(vec(m[W])))

        m = build([Variance(), dv(Variance())])
        @test m[:tr_dv_2_w_owner] === Symbol("")
        @test haskey(m, :W) && !haskey(m, :tr_dv_2_W)
        @test reads(m, :tr_dv_2_variance_risk_1, :W)
        @test haskey(m, :variance_flag) && !haskey(m, :tr_dv_2_variance_flag)

        # The penalty is omitted only for a variance that the objective minimises. A head
        # without one keeps it, and so does a head whose variance is a tracking variance.
        @test haskey(build([ConditionalValueatRisk()]), :sdp_plg_p_1)
        m = build([Variance()])
        @test haskey(m, :variance_flag) && haskey(m, :risk_minimised)
        @test !haskey(m, :sdp_plg_p_1)
        m = build([dv(Variance())])
        @test haskey(m, :sdp_plg_1) && !haskey(m, :variance_flag)
        @test haskey(m, :sdp_plg_p_1)
        m = build([dv(UncertaintySetVariance(; ucs = ucs))])
        @test !haskey(m, :variance_flag) && !haskey(m, :tr_dv_1_variance_flag)
        @test !haskey(m, :tr_dv_1_W) && haskey(m, :sdp_plg_p_1)

        # The role is read per variance and per objective. A ceiling, a variance at zero
        # scale, and a variance that the objective does not minimise keep the penalty.
        ceiling = Variance(; settings = RiskMeasureSettings(; ub = 1.0, rke = false))
        m = build([ConditionalValueatRisk(), ceiling])
        @test haskey(m, :variance_risk_2_ub) && !haskey(m, :variance_flag)
        @test haskey(m, :sdp_plg_p_1)
        m = build([ConditionalValueatRisk(),
                   Variance(; settings = RiskMeasureSettings(; scale = 0.0))])
        @test !haskey(m, :variance_flag) && haskey(m, :sdp_plg_p_1)
        m = build([Variance()]; obj = MaximumReturn())
        @test haskey(m, :variance_flag) && !haskey(m, :risk_minimised)
        @test haskey(m, :sdp_plg_p_1)
        @test haskey(build([Variance()]; obj = MaximumUtility(; l = 0)), :sdp_plg_p_1)
        @test !haskey(build([Variance()]; obj = MaximumUtility()), :sdp_plg_p_1)
        # `MaximumRatio` minimises the risk in its return form; its risk form, `sr_risk`,
        # bounds it.
        m = build([Variance()]; obj = MaximumRatio(; rf = 0))
        @test haskey(m, :sdp_plg_p_1) == haskey(m, :sr_risk)
        m = build([Variance()]; obj = MaximumRatio(; rf = 1))
        @test haskey(m, :sr_risk) && haskey(m, :sdp_plg_p_1)

        # A dependent build inside an independent one records the independent build as the
        # owner, so it reads the shifted `W` and leaves the head's penalty in place.
        m = build([iv(dv(Variance()))])
        @test !haskey(m, :tr_iv_1_w_owner)
        @test m[:tr_iv_1_tr_dv_1_w_owner] === :tr_iv_1_
        @test haskey(m, :tr_iv_1_W) && !haskey(m, :tr_iv_1_tr_dv_1_W)
        @test m[:tr_iv_1_W] !== m[:W]
        @test reads(m, :tr_iv_1_tr_dv_1_variance_risk_1, :tr_iv_1_W)
        @test !haskey(m, :tr_iv_1_variance_flag) && !haskey(m, :variance_flag)
        @test haskey(m, :sdp_plg_p_1)
    end
end
