# The tracking risk measures of `src/16_RiskMeasures/16_TrackingRiskMeasure.jl`, checked
# against the closed forms their docstrings state: the five norms of `TrackingRiskMeasure`,
# the two modes of `RiskTrackingRiskMeasure`, the two modes of `RiskTrackingError` in an
# optimisation, and every propagation method of the three types.
using Clarabel, JuMP, Statistics

# Observation weights that resolve against the returns matrix. It lives at top level because
# `@testset` expands to a function body, which cannot hold a `struct`.
struct TrackedDecayWeights <: PortfolioOptimisers.DynamicAbstractWeights end
function PortfolioOptimisers.get_observation_weights(::TrackedDecayWeights,
                                                     X::PortfolioOptimisers.MatNum;
                                                     dims::Int = 1, kwargs...)
    return aweights(collect(range(1.0, 2.0; length = size(X, dims))))
end

@testset "Tracking risk measures" begin
    rng = StableRNG(1049)
    T, N = 60, 4
    X = 0.01 .* randn(rng, T, N) .+ 0.001
    w = [0.4, 0.3, 0.2, 0.1]
    wb = fill(0.25, N)
    pr = prior(EmpiricalPrior(), X)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = Dict("verbose" => false))
    fees = Fees(; l = 0.002)

    @testset "TrackingRiskMeasure states the five norms" begin
        d = X * w - X * wb
        for (alg, val) in ((L2Norm(), norm(d, 2) / sqrt(T - 1)),
                           (L2Norm(; ddof = 0), norm(d, 2) / sqrt(T)),
                           (SquaredL2Norm(), norm(d, 2)^2 / (T - 1)), (L1Norm(), norm(d, 1) / T),
                           (LpNorm(), norm(d, 3) / cbrt(T - 1)),
                           (LpNorm(; p = 4), norm(d, 4) / (T - 1)^(1 / 4)), (LInfNorm(), norm(d, Inf)))
            rr = TrackingRiskMeasure(; tr = ReturnsTracking(; w = X * wb), alg = alg)
            rw = TrackingRiskMeasure(; tr = WeightsTracking(; w = wb), alg = alg)
            @test isapprox(rr(w, X), val; rtol = 1e-12)
            @test isapprox(rw(w, X), val; rtol = 1e-12)
            @test isapprox(expected_risk(rw, w, X), val; rtol = 1e-12)
            # A return series tracks a return series; a weight benchmark refuses one.
            @test isapprox(rr(X * w), val; rtol = 1e-12)
            @test_throws ArgumentError rw(X * w)
        end
        # With d = 0 the L2 and L1 forms are Cajas's Equations 9.16 and 9.17.
        rt = WeightsTracking(; w = wb)
        @test isapprox(TrackingRiskMeasure(; tr = rt, alg = L2Norm(; ddof = 0))(w, X),
                       norm(X * wb - X * w, 2) / sqrt(T); rtol = 1e-12)
        @test isapprox(TrackingRiskMeasure(; tr = rt, alg = L1Norm())(w, X),
                       sum(abs, X * wb - X * w) / T; rtol = 1e-12)
    end

    @testset "TrackingRiskMeasure charges each fee on its own series" begin
        # `Fees(; l)` charges `l` times the long weight sum in every period.
        fb = Fees(; l = 0.001)
        r = TrackingRiskMeasure(; tr = WeightsTracking(; w = wb, fees = fb))
        x = X * w .- 0.002
        b = X * wb .- 0.001
        @test isapprox(r(w, X, fees), norm(x - b) / sqrt(T - 1); rtol = 1e-12)
        @test isapprox(r(w, X), norm(X * w - b) / sqrt(T - 1); rtol = 1e-12)
    end

    @testset "RiskTrackingRiskMeasure states the two modes" begin
        sd = factory(StandardDeviation(), pr)
        cvar = ConditionalValueatRisk()
        ri = RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = wb), r = sd)
        rd = RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = wb), r = cvar,
                                     alg = DependentVariableTracking())
        # The independent standard deviation is Cajas's Equation 9.18.
        @test isapprox(ri(w, X), sqrt(dot(w - wb, pr.sigma, w - wb)); rtol = 1e-12)
        @test isapprox(expected_risk(ri, w, X), ri(w, X); rtol = 1e-14)
        @test isapprox(rd(w, X),
                       abs(expected_risk(cvar, w, X) - expected_risk(cvar, wb, X));
                       rtol = 1e-12)
        @test iszero(ri(wb, X))
        @test iszero(rd(wb, X))
    end

    @testset "the independent functor charges the fee of the portfolio" begin
        # The model tracks the series `X(w - wb) - F(w)`, and the functor reads the same
        # series (#1316). Before the fix the functor charged `F(w - wb)`, zero at `wb`.
        cvar = ConditionalValueatRisk()
        x = X * (w - wb) .- 0.002 * sum(w)
        r = RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = wb), r = cvar)
        @test isapprox(r(w, X, fees), cvar(x); rtol = 1e-14)
        @test isapprox(r(wb, X, fees), 0.002; rtol = 1e-12)
        # Without a fee the two series are equal.
        @test r(w, X) == expected_risk(cvar, w - wb, X)
        sol = optimise(MeanRisk(; r = r, obj = MinimumRisk(),
                                opt = JuMPOptimiser(; pe = pr, slv = slv, fees = fees)))
        @test isa(sol.retcode, OptimisationSuccess)
        @test isapprox(sol.w, wb; atol = 1e-6)
        @test isapprox(JuMP.value(sol.model[:risk]), 0.002; rtol = 1e-6)
        @test isapprox(expected_risk(r, sol.w, X, fees), JuMP.value(sol.model[:risk]);
                       rtol = 1e-6)
        # A measure that reads the weights alone reads no fee.
        rs = factory(RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = wb),
                                             r = StandardDeviation()), pr)
        @test rs(w, X, fees) == rs(w, X)
        # The dependent functor charges each weight vector its own fee, as the model does.
        rd = RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = wb), r = cvar,
                                     alg = DependentVariableTracking())
        @test isapprox(rd(w, X, fees),
                       abs(expected_risk(cvar, w, X, fees) -
                           expected_risk(cvar, wb, X, fees)); rtol = 1e-12)
    end

    @testset "every tracked measure reads the series of the weight difference" begin
        cvar = ConditionalValueatRisk()
        wd = w - wb
        x = X * wd .- 0.002 * sum(w)
        wb2 = [0.1, 0.2, 0.3, 0.4]
        track(ri; b = wb, alg = IndependentVariableTracking()) = RiskTrackingRiskMeasure(;
                                                                                         tr = WeightsTracking(;
                                                                                                              w = b),
                                                                                         r = ri,
                                                                                         alg = alg)
        # A moment measure takes its target from the weight difference, as the model does.
        lom = factory(LowOrderMoment(), pr)
        @test isapprox(track(lom)(w, X, fees), mean(max.(dot(wd, pr.mu) .- x, 0));
                       rtol = 1e-12)
        @test isapprox(track(LowOrderMoment())(w, X, fees), mean(max.(mean(x) .- x, 0));
                       rtol = 1e-12)
        @test isapprox(track(MedianAbsoluteDeviation())(w, X, fees),
                       MedianAbsoluteDeviation()(x); rtol = 1e-12)
        # Observation weights that resolve against the data resolve first.
        ow = aweights(collect(range(1.0, 2.0; length = T)))
        lomd = LowOrderMoment(; w = TrackedDecayWeights(), mu = pr.mu)
        @test isapprox(track(lomd)(w, X, fees), mean(max.(dot(wd, pr.mu) .- x, 0), ow);
                       rtol = 1e-12)
        # Every moment family resolves its weights against `X`, down to the variance
        # estimator it holds, and agrees with the same measure built with the resolved
        # weights (#1319).
        sha = PortfolioOptimisers.StandardisedHighOrderMoment
        for f in (ow -> LowOrderMoment(; w = ow, mu = pr.mu, alg = SecondMoment()),
                  ow -> HighOrderMoment(; w = ow, mu = pr.mu, alg = sha()),
                  ow -> Kurtosis(; w = ow, mu = pr.mu), ow -> Skewness(; w = ow, mu = pr.mu))
            @test isapprox(track(f(TrackedDecayWeights()))(w, X, fees),
                           track(f(ow))(w, X, fees); rtol = 1e-12)
        end
        # A tracking error takes the norm of the series minus its own benchmark series.
        te = TrackingRiskMeasure(; tr = WeightsTracking(; w = wb2))
        @test isapprox(track(te)(w, X, fees), norm(x - X * wb2, 2) / sqrt(T - 1);
                       rtol = 1e-12)
        # A nested independent mode subtracts its benchmark weights. A nested dependent mode
        # subtracts the risk of its benchmark weights, which pay their own fee.
        @test isapprox(track(track(cvar; b = wb2))(w, X, fees),
                       cvar(X * (wd - wb2) .- 0.002 * sum(w)); rtol = 1e-12)
        @test isapprox(track(track(cvar; b = wb2, alg = DependentVariableTracking()))(w, X,
                                                                                      fees),
                       abs(cvar(x) - cvar(X * wb2 .- 0.002 * sum(wb2))); rtol = 1e-12)
        # A ratio divides the two results, and a composite of the weights reads no fee.
        cvar10 = ConditionalValueatRisk(; alpha = 0.1)
        @test isapprox(track(RiskRatio(; r1 = cvar, r2 = cvar10))(w, X, fees),
                       cvar(x) / cvar10(x); rtol = 1e-12)
        vsk = factory(VarianceSkewKurtosis(), prior(HighOrderPriorEstimator(), X))
        @test track(vsk)(w, X, fees) == vsk(wd, X)
        # Any other measure that reads weights and returns reads the series, or refuses.
        @test isapprox(PortfolioOptimisers.difference_risk(PortfolioOptimisers.WeightsReturnsFeesInput(),
                                                           LowOrderMoment(), wd, w, X,
                                                           fees), LowOrderMoment()(x);
                       rtol = 1e-12)
        @test_throws ArgumentError PortfolioOptimisers.difference_risk(PortfolioOptimisers.WeightsReturnsFeesInput(),
                                                                       te, wd, w, X, fees)
        # In an optimisation the functor reads back the value of the model.
        for ri in (lom, te, track(cvar; b = wb2))
            r = track(ri)
            sol = optimise(MeanRisk(; r = r, obj = MinimumRisk(),
                                    opt = JuMPOptimiser(; pe = pr, slv = slv, fees = fees)))
            @test isa(sol.retcode, OptimisationSuccess)
            @test isapprox(expected_risk(factory(r, pr, slv), sol.w, X, fees),
                           JuMP.value(sol.model[:risk]); rtol = 1e-6)
        end
    end

    @testset "in an optimisation the independent mode is exact and the dependent mode one-sided" begin
        cvar = ConditionalValueatRisk()
        rb = expected_risk(cvar, wb, X)
        head(tr) = MeanRisk(; r = cvar, obj = MinimumRisk(),
                            opt = JuMPOptimiser(; pe = pr, slv = slv, tr = tr))
        # The independent bound binds at `err`.
        err = 1e-3
        sol = optimise(head(RiskTrackingError(; tr = WeightsTracking(; w = wb), r = cvar,
                                              err = err)))
        @test isa(sol.retcode, OptimisationSuccess)
        @test isapprox(expected_risk(cvar, sol.w - wb, X), err; rtol = 1e-3)
        # The dependent bound holds from above only: the portfolio is less risky than the
        # benchmark by far more than `err` (#1317).
        err = 1e-5
        sol = optimise(head(RiskTrackingError(; tr = WeightsTracking(; w = wb), r = cvar,
                                              err = err, alg = DependentVariableTracking())))
        @test isa(sol.retcode, OptimisationSuccess)
        rw = expected_risk(cvar, sol.w, X)
        @test rw - rb <= err
        @test rb - rw > 10 * err
        # In the objective, the model reports zero for a portfolio that is less risky.
        r = RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = wb), r = cvar,
                                    alg = DependentVariableTracking())
        sol = optimise(MeanRisk(; r = r, obj = MinimumRisk(),
                                opt = JuMPOptimiser(; pe = pr, slv = slv)))
        @test isa(sol.retcode, OptimisationSuccess)
        @test abs(JuMP.value(sol.model[:risk])) < 1e-8
        @test rb - expected_risk(cvar, sol.w, X) > 1e-5
        @test isapprox(r(sol.w, X), rb - expected_risk(cvar, sol.w, X); rtol = 1e-12)
    end

    @testset "the constructors" begin
        @test_throws DomainError RiskTrackingError(; tr = WeightsTracking(; w = wb),
                                                   err = -1.0)
        @test_throws DomainError RiskTrackingError(; tr = WeightsTracking(; w = wb),
                                                   err = Inf)
        # The tracked measure loses its own bound and objective flag.
        ub = RiskMeasureSettings(; ub = 0.1, rke = true)
        rte = RiskTrackingError(; tr = WeightsTracking(; w = wb),
                                r = StandardDeviation(; settings = ub))
        @test isnothing(rte.r.settings.ub)
        @test !rte.r.settings.rke
        rt = RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = wb),
                                     r = ConditionalValueatRisk(; settings = ub))
        @test isnothing(rt.r.settings.ub)
        @test !rt.r.settings.rke
        # The dependent mode warns for a measure whose model is a quadratic expression.
        @test_logs (:warn, r"not guaranteed to work") RiskTrackingRiskMeasure(;
                                                                              tr = WeightsTracking(;
                                                                                                   w = wb),
                                                                              alg = DependentVariableTracking())
        @test_logs RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = wb))
    end

    @testset "port_opt_view slices the benchmark and the tracked measure" begin
        i = [1, 3]
        sd = factory(StandardDeviation(), pr)
        rte = RiskTrackingError(; tr = WeightsTracking(; w = wb), r = sd, err = 0.01,
                                alg = DependentVariableTracking())
        v = PortfolioOptimisers.port_opt_view(rte, i, X)
        @test v.tr.w == wb[i]
        @test v.r.sigma == pr.sigma[i, i]
        @test v.err == rte.err
        @test v.alg === rte.alg
        rt = RiskTrackingRiskMeasure(; settings = RiskMeasureSettings(; scale = 2.0),
                                     tr = WeightsTracking(; w = wb), r = sd)
        v = PortfolioOptimisers.port_opt_view(rt, i, X)
        @test v.tr.w == wb[i]
        @test v.r.sigma == pr.sigma[i, i]
        @test v.settings === rt.settings
        @test v.alg === rt.alg
        tm = TrackingRiskMeasure(; tr = WeightsTracking(; w = wb))
        @test PortfolioOptimisers.port_opt_view(tm, i).tr.w == wb[i]
    end

    @testset "factory advances the benchmark and resolves the tracked measure" begin
        w1 = [0.1, 0.2, 0.3, 0.4]
        moving = WeightsTracking(; w = wb)
        fixed = WeightsTracking(; w = wb, fixed = true)
        # RiskTrackingError.
        rte = RiskTrackingError(; tr = moving, r = StandardDeviation(), err = 0.01)
        f = factory(rte, pr, slv, nothing, w1)
        @test f.tr.w == w1
        @test f.r.sigma == pr.sigma
        @test f.err == rte.err && f.alg === rte.alg
        f = factory(rte, pr, slv, nothing)
        @test f.tr === rte.tr
        @test f.r.sigma == pr.sigma
        @test factory(rte, w1).tr.w == w1
        @test factory(RiskTrackingError(; tr = fixed, err = 0.01), w1).tr === fixed
        # TrackingRiskMeasure.
        tm = TrackingRiskMeasure(; tr = moving, alg = L1Norm())
        @test factory(tm, w1).tr.w == w1
        @test factory(tm, w1).alg === tm.alg
        @test factory(tm, pr, slv, nothing, w1).tr.w == w1
        @test factory(TrackingRiskMeasure(; tr = fixed), w1).tr === fixed
        ret = ReturnsTracking(; w = X * wb)
        @test factory(TrackingRiskMeasure(; tr = ret), w1).tr === ret
        # RiskTrackingRiskMeasure.
        rt = RiskTrackingRiskMeasure(; tr = moving, r = StandardDeviation())
        f = factory(rt, pr, slv)
        @test f.tr === rt.tr
        @test f.r.sigma == pr.sigma
        f = factory(rt, w1)
        @test f.tr.w == w1
        @test factory(RiskTrackingRiskMeasure(; tr = fixed), w1).tr === fixed
    end

    @testset "needs_previous_weights reads the benchmark and the tracked measure" begin
        moving = WeightsTracking(; w = wb)
        fixed = WeightsTracking(; w = wb, fixed = true)
        tn = TurnoverRiskMeasure(; w = wb)
        @test PortfolioOptimisers.needs_previous_weights(TrackingRiskMeasure(; tr = moving))
        @test !PortfolioOptimisers.needs_previous_weights(TrackingRiskMeasure(; tr = fixed))
        @test !PortfolioOptimisers.needs_previous_weights(TrackingRiskMeasure(;
                                                                              tr = ReturnsTracking(;
                                                                                                   w = X *
                                                                                                       wb)))
        for Tracker in (RiskTrackingError, RiskTrackingRiskMeasure)
            @test PortfolioOptimisers.needs_previous_weights(Tracker(; tr = moving))
            @test !PortfolioOptimisers.needs_previous_weights(Tracker(; tr = fixed))
            @test PortfolioOptimisers.needs_previous_weights(Tracker(; tr = fixed, r = tn))
        end
    end
end
