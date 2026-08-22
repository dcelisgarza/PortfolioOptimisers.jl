#=
Strictness of observation weight resolution.

`get_observation_weights` returns `nothing` to mean *no weights were requested*, never
*weights were unavailable*. Every `isnothing` branch downstream reads it the first way and
computes an unweighted result, so a `DynamicAbstractWeights` that resolved to `nothing`
would silently produce a plausible-looking but unweighted answer. It raises instead.

See docs/adr/0043-nothing-observation-weights-means-unweighted-not-unavailable.md and issue
#177.

The weight types live at top level because `@testset` expands to a function body, which
cannot hold a `struct`.
=#

# Implements both documented arities.
struct CompleteObsWeights{T} <: PortfolioOptimisers.DynamicAbstractWeights
    half_life::T
end
function PortfolioOptimisers.get_observation_weights(w::CompleteObsWeights,
                                                     X::PortfolioOptimisers.VecNum;
                                                     kwargs...)
    return aweights(collect(eweights(1:length(X), 2^(-inv(w.half_life)); scale = true)))
end
function PortfolioOptimisers.get_observation_weights(w::CompleteObsWeights,
                                                     X::PortfolioOptimisers.MatNum;
                                                     dims::Int = 1, kwargs...)
    return aweights(collect(eweights(1:size(X, dims), 2^(-inv(w.half_life)); scale = true)))
end

# Implements the vector arity ONLY. Handing this a matrix is the trap under test: before
# the split it resolved to `nothing` and every call site quietly went unweighted.
struct VectorOnlyObsWeights <: PortfolioOptimisers.DynamicAbstractWeights end
function PortfolioOptimisers.get_observation_weights(::VectorOnlyObsWeights,
                                                     X::PortfolioOptimisers.VecNum;
                                                     kwargs...)
    return aweights(fill(inv(length(X)), length(X)))
end

const PO = PortfolioOptimisers

@testset "Observation weight resolution" begin
    using Test, PortfolioOptimisers, StableRNGs, StatsBase, Statistics

    rng = StableRNG(123456789)
    X = randn(rng, 60, 5)
    x = view(X, :, 1)

    @testset "Resolver contract" begin
        # `nothing` in, `nothing` out: no weights were requested.
        @test isnothing(PO.get_observation_weights(nothing, X; dims = 1))
        @test isnothing(PO.get_observation_weights(nothing, x))

        # Pre-computed weights pass straight through, untouched.
        ew = eweights(1:60, inv(60); scale = true)
        @test PO.get_observation_weights(ew, X; dims = 1) === ew

        # A complete dynamic type resolves on both arities, honouring `dims`.
        cw = CompleteObsWeights(10)
        @test length(PO.get_observation_weights(cw, X; dims = 1)) == 60
        @test length(PO.get_observation_weights(cw, X; dims = 2)) == 5
        @test length(PO.get_observation_weights(cw, x)) == 60

        # The vector arity still works for a partially-implemented type...
        @test length(PO.get_observation_weights(VectorOnlyObsWeights(), x)) == 60
        # ...but the shape it never implemented raises rather than silently unweighting.
        @test_throws PO.ObservationWeightsError PO.get_observation_weights(VectorOnlyObsWeights(),
                                                                           X; dims = 1)
    end

    @testset "ObservationWeightsError" begin
        @test PO.ObservationWeightsError <: PO.PortfolioOptimisersError

        err = try
            PO.get_observation_weights(VectorOnlyObsWeights(), X; dims = 1)
        catch e
            e
        end
        msg = sprint(showerror, err)
        # The message must name the offending type, the shape it was handed, and both
        # signatures to implement - it is the only guidance a user gets.
        @test occursin("ObservationWeightsError", msg)
        @test occursin("VectorOnlyObsWeights", msg)
        @test occursin("2-dimensional input of size (60, 5)", msg)
        @test occursin("X::VecNum", msg)
        @test occursin("X::MatNum", msg)
        @test occursin("DynamicAbstractWeights", msg)
    end

    @testset "kwargs.weights constructor guards" begin
        # Three constructors screen a `weights` entry that they forward to a third-party
        # kernel. The check itself always worked. Its failure path did not: ArgCheck builds
        # the exception with `T(msg::String)`, and `Core.TypeError` has no such
        # constructor, so a wrong type raised `MethodError: no method matching
        # TypeError(::String)` and named neither the keyword nor the type received.
        bad = (; weights = 1.0)
        bad_tgt = LinearModel(; kwargs = bad)
        # `Clustering.kmeans` weights its points, which are the assets, so the k-means
        # message names a point weight. The two regression kernels weight observations.
        for (sym, quantity, f) in
            (("kwargs.weights", "point weights", () -> KMeansAlgorithm(; kwargs = bad)),
             ("tgt.kwargs.weights", "observation weights",
              () -> StepwiseRegression(; tgt = bad_tgt)),
             ("retgt.kwargs.weights", "observation weights",
              () -> DimensionReductionRegression(; retgt = bad_tgt)))
            @test_throws ArgumentError f()
            msg = sprint(showerror, try
                             f()
                         catch e
                             e
                         end)
            # The message must name the offending keyword and the type received.
            @test occursin(sym, msg)
            @test occursin(quantity, msg)
            @test occursin("Float64", msg)
        end

        # The emptiness guard beside it is unaffected: its exception takes a message.
        empty_tgt = LinearModel(; kwargs = (; weights = aweights(Float64[])))
        @test_throws PO.IsEmptyError KMeansAlgorithm(; kwargs = (; weights = Float64[]))
        @test_throws PO.IsEmptyError StepwiseRegression(; tgt = empty_tgt)
        @test_throws PO.IsEmptyError DimensionReductionRegression(; retgt = empty_tgt)

        # A well-formed `weights` entry still constructs.
        good = (; weights = aweights(fill(inv(60), 60)))
        good_tgt = LinearModel(; kwargs = good)
        @test KMeansAlgorithm(; kwargs = good).kwargs === good
        @test StepwiseRegression(; tgt = good_tgt).tgt.kwargs === good
        @test DimensionReductionRegression(; retgt = good_tgt).retgt.kwargs === good
    end

    @testset "moment_window_and_weights" begin
        idx = collect(21:30)

        # All four methods resolve a complete type and reject a partial one. The two
        # windowed methods subset first, so they must still raise on the sliced view.
        cw = CompleteObsWeights(10)
        Xw, w = PO.moment_window_and_weights(X, cw; dims = 1)
        @test size(Xw) == (60, 5)
        @test length(w) == 60
        Xw, w = PO.moment_window_and_weights(X, cw, idx; dims = 1)
        @test size(Xw) == (10, 5)
        @test length(w) == 10
        xw, w = PO.moment_window_and_weights(x, cw)
        @test length(w) == 60
        xw, w = PO.moment_window_and_weights(x, cw, idx)
        @test length(w) == 10

        vw = VectorOnlyObsWeights()
        @test_throws PO.ObservationWeightsError PO.moment_window_and_weights(X, vw;
                                                                             dims = 1)
        @test_throws PO.ObservationWeightsError PO.moment_window_and_weights(X, vw, idx;
                                                                             dims = 1)

        # `nothing` weights remain permissive through the same helper: the strictness must
        # not leak into the legitimately-unweighted path.
        Xw, w = PO.moment_window_and_weights(X, nothing; dims = 1)
        @test isnothing(w)
        Xw, w = PO.moment_window_and_weights(X, nothing, idx; dims = 1)
        @test isnothing(w)
    end

    @testset "08_Moments estimators" begin
        vw = VectorOnlyObsWeights()
        cw = CompleteObsWeights(10)

        # The weights genuinely reach the estimator: weighted != unweighted.
        @test !isapprox(cor(GeneralCovariance(), X), cor(GeneralCovariance(; w = cw), X))
        @test !isapprox(std(SimpleVariance(), X), std(SimpleVariance(; w = cw), X))
        @test !isapprox(mean(SimpleExpectedReturns(), X),
                        mean(SimpleExpectedReturns(; w = cw), X))

        # `cov(::GeneralCovariance, ...)` used to pass `ce.w` to `robust_cov` raw, never
        # resolving it, so ANY dynamic weights type - even a complete one - died with a
        # MethodError while its sibling `cor` worked. It now resolves like `cor` does, and
        # agrees with passing the equivalent static weights directly.
        @test isapprox(cov(GeneralCovariance(; w = cw), X),
                       cov(GeneralCovariance(;
                                             w = PO.get_observation_weights(cw, X;
                                                                            dims = 1)), X))

        # An unresolvable type raises instead of returning an unweighted moment.
        @test_throws PO.ObservationWeightsError cor(GeneralCovariance(; w = vw), X)
        @test_throws PO.ObservationWeightsError cov(GeneralCovariance(; w = vw), X)
        @test_throws PO.ObservationWeightsError std(SimpleVariance(; w = vw), X)
        @test_throws PO.ObservationWeightsError var(SimpleVariance(; w = vw), X)
        @test_throws PO.ObservationWeightsError mean(SimpleExpectedReturns(; w = vw), X)

        # Unweighted estimators are untouched.
        @test isa(cor(GeneralCovariance(), X), Matrix)
        @test isa(mean(SimpleExpectedReturns(), X), AbstractArray)
    end

    @testset "19_RiskMeasures resolve-then-rebuild" begin
        # These measures dispatch on the weights type, resolve, then rebuild themselves and
        # re-dispatch. A `nothing` resolution used to rebuild as an unweighted measure and
        # silently re-dispatch to the unweighted method - the same defect, one layer deeper.
        w = fill(inv(5), 5)
        cw = CompleteObsWeights(10)
        vw = VectorOnlyObsWeights()

        @test isa(LowOrderMoment(; w = cw)(w, X), Number)
        @test isa(HighOrderMoment(; w = cw)(w, X), Number)
        @test_throws PO.ObservationWeightsError LowOrderMoment(; w = vw)(w, X)
        @test_throws PO.ObservationWeightsError HighOrderMoment(; w = vw)(w, X)

        # Unweighted and pre-weighted measures behave exactly as before.
        @test isa(LowOrderMoment()(w, X), Number)
        @test isa(LowOrderMoment(; w = eweights(1:60, inv(60); scale = true))(w, X), Number)

        # All five central-moment measures rebuild through one generic, so the rule is
        # asserted on all five rather than on the two that used to be covered. Only
        # `LowOrderMoment` and `HighOrderMoment` were tested here, which is how the other
        # three drifted: `Skewness` dropped `settings` from its rebuild, and `Kurtosis`
        # bound its rebuild to `SemiMoment`, so the default `FullMoment` matched no method.
        rw = PO.get_observation_weights(cw, X; dims = 1)
        for (dyn, sta) in ((LowOrderMoment(; w = cw), LowOrderMoment(; w = rw)),
                           (HighOrderMoment(; w = cw), HighOrderMoment(; w = rw)),
                           (Kurtosis(; w = cw), Kurtosis(; w = rw)),
                           (Kurtosis(; w = cw, alg1 = PO.SemiMoment()),
                            Kurtosis(; w = rw, alg1 = PO.SemiMoment())),
                           (Skewness(; w = cw), Skewness(; w = rw)),
                           (PO.ThirdCentralMoment(; w = cw), PO.ThirdCentralMoment(; w = rw)))
            # Resolving the weights is all the rebuild does: the answer is the one the
            # measure gives when it is handed the same weights directly.
            @test dyn(w, X) == sta(w, X)
            @test dyn(X * w) == sta(X * w)
            # An unresolvable type still raises rather than going quietly unweighted.
            dyn_vw = PO.Accessors.@set dyn.w = vw
            @test_throws PO.ObservationWeightsError dyn_vw(w, X)
        end

        # The rebuild replaces `w` and copies every other field, so a non-default setting
        # survives it. The ten hand-written rebuilds named their fields one by one.
        for r in (LowOrderMoment(; w = cw, settings = RiskMeasureSettings(; scale = 3)),
                  Kurtosis(; w = cw, settings = RiskMeasureSettings(; scale = 3), N = 7),
                  Skewness(; w = cw, settings = MaxRiskMeasureSettings(; scale = 3)))
            rebuilt = PO.Accessors.@set r.w = PO.get_observation_weights(r.w, X)
            @test isa(rebuilt.w, StatsBase.AbstractWeights)
            @test all(f -> f === :w || getfield(rebuilt, f) === getfield(r, f),
                      fieldnames(typeof(r)))
        end
    end
end
#=
The conic tail measures resolve their observation weights on both sides of the
returns/drawdown twin.

A tail measure and its drawdown twin are one programme under the substitution
`net_X -> -dd[2:T+1]` (`risk_series`). They used to disagree about what they handed
`get_observation_weights`: the returns tails passed `net_X`, a vector of JuMP expressions
that matches neither documented arity, so a `DynamicAbstractWeights` ALWAYS raised there;
the drawdown twins passed `pr.X` and resolved. One substitution, two answers. Both sides now
pass `pr.X`, which is the documented `MatNum` arity.
=#
@testset "Observation weights resolve on both sides of the returns/drawdown twin" begin
    using Test, PortfolioOptimisers, StableRNGs, StatsBase, Clarabel

    rng = StableRNG(987654321)
    X = randn(rng, 40, 5) ./ 100
    rd = ReturnsResult(; nx = string.('A':'E'), X = X)
    cw = CompleteObsWeights(8)
    sw = PO.get_observation_weights(cw, X; dims = 1)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = "verbose" => false)

    # Build the model without solving it: the resolved weights are visible in the
    # constraints, and a solve would only add tolerance to the comparison.
    function build(r)
        mr = MeanRisk(; r = r, opt = JuMPOptimiser(; slv = slv))
        attrs = PO.processed_jump_optimiser_attributes(mr.opt, rd; dims = 1)
        model = PO.JuMP.Model()
        PO.JuMP.set_string_names_on_creation(model, false)
        PO.set_model_scales!(model, mr.opt.sc, mr.opt.so)
        PO.set_maximum_ratio_factor_variables!(model, mr.obj)
        PO.set_w!(model, attrs.pr.X, mr.wi)
        PO.set_weight_constraints!(model, attrs.wb, mr.opt)
        PO.assemble_jump_model!(model, mr, mr.opt, attrs, rd, mr.r, mr.obj)
        return sprint(print, model)
    end

    # Resolving the weights is all the dynamic type buys: the model is the one the measure
    # builds when it is handed the same weights directly.
    for (dyn, sta) in ((ValueatRisk(; w = cw, alg = MIPValueatRisk()),
                        ValueatRisk(; w = sw, alg = MIPValueatRisk())),
                       (DrawdownatRisk(; w = cw), DrawdownatRisk(; w = sw)),
                       (ConditionalValueatRisk(; w = cw), ConditionalValueatRisk(; w = sw)),
                       (ConditionalDrawdownatRisk(; w = cw), ConditionalDrawdownatRisk(; w = sw)),
                       (DistributionallyRobustConditionalValueatRisk(; w = cw),
                        DistributionallyRobustConditionalValueatRisk(; w = sw)),
                       (DistributionallyRobustConditionalDrawdownatRisk(; w = cw),
                        DistributionallyRobustConditionalDrawdownatRisk(; w = sw)),
                       (EntropicValueatRisk(; w = cw), EntropicValueatRisk(; w = sw)),
                       (EntropicDrawdownatRisk(; w = cw), EntropicDrawdownatRisk(; w = sw)),
                       (RelativisticValueatRisk(; w = cw), RelativisticValueatRisk(; w = sw)),
                       (RelativisticDrawdownatRisk(; w = cw), RelativisticDrawdownatRisk(; w = sw)),
                       (PowerNormValueatRisk(; w = cw), PowerNormValueatRisk(; w = sw)),
                       (PowerNormDrawdownatRisk(; w = cw), PowerNormDrawdownatRisk(; w = sw)))
        @test build(dyn) == build(sta)
    end

    # A type that cannot resolve the matrix arity still raises rather than going quietly
    # unweighted, and it now raises on both sides rather than only one.
    vw = VectorOnlyObsWeights()
    @test_throws PO.ObservationWeightsError build(ConditionalValueatRisk(; w = vw))
    @test_throws PO.ObservationWeightsError build(ConditionalDrawdownatRisk(; w = vw))

    @testset "DimensionReductionRegression standardises and recovers with one scale" begin
        # `prep_dim_red_reg` used to standardise with `StatsBase.ZScoreTransform` -- the
        # plain mean and the corrected standard deviation -- while `regression` divided the
        # recovered coefficients by `re.ve` and centred the intercept with `re.ve.me`. The
        # two agreed for the default `ve` and for nothing else, so every weighted path
        # recovered a coefficient in no unit at all. See issue #398.
        #
        # The invariant is the one Equations 4.13, 4.15 and 4.20 of the source impose: the
        # prediction in the original factor space must equal the prediction in the reduced
        # space. It holds only when the divisor is the scale that standardised.
        rng2 = StableRNG(987654321)
        F = randn(rng2, 250, 6) ./ 100
        Y = F * randn(rng2, 6, 3) .+ randn(rng2, 250, 3) ./ 500
        pw = pweights(range(; start = 1, stop = 5, length = 250))

        function prediction_gap(re, Y, F)
            reg = regression(re, Y, F)
            f1, Vp, mu, sigma = PO.prep_dim_red_reg(re, F)
            # The two statistics are the ones `regression` recovers with.
            @test isapprox(mu, vec(PO.Statistics.mean(re.ve.me, F; dims = 1)))
            @test isapprox(sigma, vec(PO.Statistics.std(re.ve, F; dims = 1)))
            gap = zero(eltype(F))
            for i in axes(Y, 2)
                y = view(Y, :, i)
                coefs = PO.StatsAPI.coef(PO.StatsAPI.fit(re.retgt, f1, y))
                pred = reg.b[i] .+ F * view(reg.M, i, :)
                gap = max(gap, maximum(abs, f1 * coefs .- pred))
            end
            return gap
        end

        # `ve` is `@fprop`-tagged, so `factory` puts the incoming weights into it.
        re_w = factory(DimensionReductionRegression(), pw)
        @test !isnothing(re_w.ve.w)
        @test prediction_gap(DimensionReductionRegression(), Y, F) < 1e-12
        @test prediction_gap(re_w, Y, F) < 1e-12
        # `corrected = false` is the other way the two scales used to part.
        re_c = DimensionReductionRegression(; ve = SimpleVariance(; corrected = false))
        @test prediction_gap(re_c, Y, F) < 1e-12

        # A constant factor has zero standard deviation. The floor at `eps` keeps the
        # division finite rather than answering `NaN`.
        Fc = copy(F)
        Fc[:, 3] .= 0.01
        reg_c = regression(DimensionReductionRegression(), Y, Fc)
        @test all(isfinite, reg_c.M)
        @test all(isfinite, reg_c.b)
    end
end
