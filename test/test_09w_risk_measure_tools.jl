# The sweep of the risk measure tools (#610): the five helpers that rewrite the settings of a
# measure, `measure_label`, `resolve_observation_weights` and the two functors of
# `MomentRiskMeasures`. Three defects: `no_bounds_no_risk_expr_risk_measure` turned the `scale`
# of every `RiskMeasure` and of `Skewness` into an `Int`, the helpers raised a `MethodError` for
# every `NonOptimisationRiskMeasure` (`unit_scale_risk_measure` also for `Skewness`), and the
# vector form of `bounds_risk_measure` broadcast a one-element `r` into several measures.
using InteractiveUtils: InteractiveUtils
using LinearAlgebra, StableRNGs, StatsBase

const PO = PortfolioOptimisers

const _X610 = randn(StableRNG(610), 200, 5) * 0.01 .+ 0.001
const _w610 = [0.3, 0.25, 0.2, 0.15, 0.1]
const _x610 = _X610 * _w610
const _ramp610 = StatsBase.pweights(collect(1.0:size(_X610, 1)))

# A dynamic observation weight: the ramp `1, 2, …, T`.
struct Ramp610 <: PortfolioOptimisers.DynamicAbstractWeights end
function PortfolioOptimisers.get_observation_weights(::Ramp610, X::AbstractVector;
                                                     kwargs...)
    return StatsBase.pweights(collect(1.0:length(X)))
end
function PortfolioOptimisers.get_observation_weights(::Ramp610, X::AbstractMatrix;
                                                     dims::Int = 1, kwargs...)
    return StatsBase.pweights(collect(1.0:size(X, dims)))
end

# Every concrete measure that the package defines. A test file can define its own measure, so
# the walk keeps the types of the package alone.
function _leaves610(T)
    out = Any[]
    for S in InteractiveUtils.subtypes(T)
        if isabstracttype(S)
            append!(out, _leaves610(S))
        elseif parentmodule(S) === PortfolioOptimisers
            push!(out, S)
        end
    end
    return out
end

# The default instance of each leaf. Three leaves have a keyword with no default.
function _build610(T)
    nm = nameof(T)
    wb = fill(0.2, 5)
    if nm === :RiskTrackingRiskMeasure || nm === :TrackingRiskMeasure
        return T(; tr = WeightsTracking(; w = wb))
    elseif nm === :TurnoverRiskMeasure
        return T(; w = wb)
    end
    return T()
end

# The same measure with a bound, `rke = true` and a scale of 2.5, where its settings hold them.
function _bounded610(r)
    if !hasproperty(r, :settings)
        return r
    end
    s = r.settings
    if s isa RiskMeasureSettings
        return PO.rebuild_estimator(r,
                                    (;
                                     settings = RiskMeasureSettings(; rke = true, ub = 0.7,
                                                                    scale = 2.5)))
    elseif s isa PO.MaxRiskMeasureSettings
        return PO.rebuild_estimator(r,
                                    (;
                                     settings = PO.MaxRiskMeasureSettings(; rke = true,
                                                                          lb = 0.7,
                                                                          scale = 2.5)))
    end
    return r
end

# The paths of the fields of `b` that differ from those of `a`. The walk enters a measure and
# a settings object, and compares every other field with `isequal`.
function _moved610(a, b, path = "")
    walk = a isa PO.AbstractBaseRiskMeasure || a isa PO.AbstractRiskMeasureSettings
    if walk && nameof(typeof(a)) === nameof(typeof(b))
        out = String[]
        for f in fieldnames(typeof(a))
            p = isempty(path) ? string(f) : string(path, ".", f)
            append!(out, _moved610(getfield(a, f), getfield(b, f), p))
        end
        return out
    end
    return isequal(a, b) ? String[] : [path]
end

const _helpers610 = (; no_bounds = r -> no_bounds_risk_measure(r),
                     no_bounds_no_rke = r -> no_bounds_no_risk_expr_risk_measure(r),
                     no_rke = r -> no_risk_expr_risk_measure(r),
                     unit_scale = r -> unit_scale_risk_measure(r),
                     bounds = r -> bounds_risk_measure(r, 0.3))

# The fields that each helper changes, by the family of the measure. The tables of the
# docstrings state the same.
const _changed610 = (; no_bounds = (; rm = ["settings.ub"], sk = ["settings.lb"]),
                     no_bounds_no_rke = (;
                                         rm = ["settings.scale", "settings.ub",
                                               "settings.rke"],
                                         sk = ["settings.scale", "settings.lb",
                                               "settings.rke"]),
                     no_rke = (; rm = ["settings.rke"], sk = ["settings.rke"]),
                     unit_scale = (; rm = ["settings.scale"], sk = ["settings.scale"]),
                     bounds = (; rm = ["settings.ub"], sk = ["settings.lb"]))

@testset "Risk measure tools" begin
    leaves = _leaves610(PO.AbstractBaseRiskMeasure)
    measures = [_build610(T) for T in leaves]

    @testset "measure_label names every concrete measure and joins a vector" begin
        @test length(measures) == length(leaves)
        for r in measures
            @test PO.measure_label(r) == string(nameof(typeof(r)))
        end
        @test PO.measure_label([Variance(), ConditionalValueatRisk()]) ==
              "Variance + ConditionalValueatRisk"
        @test PO.measure_label([Variance()]) == "Variance"
    end

    @testset "Each helper changes the fields its table names, and no other field" begin
        for r0 in measures, h in keys(_helpers610)
            r = _bounded610(r0)
            r_new = _helpers610[h](r)
            moved = sort(_moved610(r, r_new))
            if r isa RiskMeasure
                @test moved == sort(_changed610[h].rm)
                # The new settings keep the type of the old scale.
                @test typeof(r_new.settings.scale) === typeof(r.settings.scale)
            elseif r isa Skewness
                @test moved == sort(_changed610[h].sk)
                @test typeof(r_new.settings.scale) === typeof(r.settings.scale)
            else
                # A hierarchical measure and a non-optimisation measure have no bound and no
                # risk expression in a model, so each helper returns the measure itself.
                @test r isa Union{HierarchicalRiskMeasure, PO.NonOptimisationRiskMeasure}
                @test r_new === r
            end
        end
    end

    @testset "A unit scale keeps the number type of the scale" begin
        for sc in (3 // 2, big(2.5), 2.5f0)
            r = ConditionalValueatRisk(;
                                       settings = RiskMeasureSettings(; ub = 0.1,
                                                                      scale = sc))
            for f in (no_bounds_no_risk_expr_risk_measure, unit_scale_risk_measure)
                s = f(r).settings.scale
                @test isone(s)
                @test typeof(s) === typeof(sc)
            end
            sk = Skewness(; settings = PO.MaxRiskMeasureSettings(; lb = 0.1, scale = sc))
            for f in (no_bounds_no_risk_expr_risk_measure, unit_scale_risk_measure)
                s = f(sk).settings.scale
                @test isone(s)
                @test typeof(s) === typeof(sc)
            end
        end
    end

    @testset "A measure that the helper would not change returns as itself" begin
        r = ConditionalValueatRisk()
        @test no_bounds_risk_measure(r) === r
        @test unit_scale_risk_measure(r) === r
        r = ConditionalValueatRisk(; settings = RiskMeasureSettings(; ub = 0.1))
        @test bounds_risk_measure(r, 0.1) === r
        r = ConditionalValueatRisk(; settings = RiskMeasureSettings(; rke = false))
        @test no_risk_expr_risk_measure(r) === r
        @test no_bounds_no_risk_expr_risk_measure(r) === r
        sk = Skewness()
        @test unit_scale_risk_measure(sk) === sk
        sk = Skewness(;
                      settings = PO.MaxRiskMeasureSettings(; lb = 0.1, scale = 2.0,
                                                           rke = false))
        sk1 = unit_scale_risk_measure(sk)
        @test (sk1.settings.lb, sk1.settings.rke, sk1.settings.scale) == (0.1, false, 1.0)
    end

    @testset "The vector forms apply the method of each element" begin
        rs = [ConditionalValueatRisk(;
                                     settings = RiskMeasureSettings(; ub = 0.1,
                                                                    scale = 2.0)),
              MaximumDrawdown(; settings = RiskMeasureSettings(; ub = 0.2)),
              Skewness(; settings = PO.MaxRiskMeasureSettings(; lb = 0.05)), MeanReturn()]
        @test isnothing(no_bounds_risk_measure(rs)[1].settings.ub)
        @test isnothing(no_bounds_risk_measure(rs)[3].settings.lb)
        @test no_bounds_risk_measure(rs)[1].settings.scale == 2.0
        @test no_bounds_risk_measure(rs)[4] === rs[4]
        @test [r.settings.rke for r in no_bounds_no_risk_expr_risk_measure(rs)[1:3]] ==
              [false, false, false]
        @test [r.settings.ub for r in no_risk_expr_risk_measure(rs)[1:2]] == [0.1, 0.2]
        nb = bounds_risk_measure(rs, [0.05, 0.3, -0.1, 0.4])
        @test (nb[1].settings.ub, nb[2].settings.ub, nb[3].settings.lb) == (0.05, 0.3, -0.1)
        @test nb[4] === rs[4]
        # Each measure takes one bound. A broadcast would stretch a one-element side, and a
        # one-element `r` would then come back as several measures.
        @test_throws DimensionMismatch bounds_risk_measure(rs[1:2], [0.15])
        @test_throws DimensionMismatch bounds_risk_measure(rs[1:1], [0.1, 0.2, 0.3])
        @test_throws DimensionMismatch bounds_risk_measure(rs, [0.1, 0.2])
        @test_throws DomainError bounds_risk_measure(rs[1:1], [-0.1])
        @test_throws DomainError bounds_risk_measure(rs[1], Inf)
        # `flag` reaches an `UncertaintySetVariance` through the vector form.
        ucs = UncertaintySetVariance(; settings = RiskMeasureSettings(; ub = 0.1),
                                     ucs = BoxUncertaintySet(; lb = zeros(2, 2),
                                                             ub = ones(2, 2)))
        @test no_bounds_risk_measure([ucs], Val(false))[1] isa Variance
        @test no_bounds_risk_measure([ucs], Val(true))[1] isa UncertaintySetVariance
        @test no_bounds_no_risk_expr_risk_measure([ucs], Val(false))[1] isa Variance
    end

    @testset "A dynamic weight gives the risk of its resolved weights" begin
        makers = (r -> LowOrderMoment(; w = r),
                  r -> LowOrderMoment(; w = r, alg = SecondMoment()),
                  r -> HighOrderMoment(; w = r),
                  r -> HighOrderMoment(; w = r, ve = SimpleVariance(; me = nothing)),
                  r -> HighOrderMoment(; w = r, alg = FourthMoment()),
                  r -> ThirdCentralMoment(; w = r, ve = SimpleVariance()),
                  r -> ThirdCentralMoment(; w = r))
        for mk in makers
            dyn = mk(Ramp610())
            sto = mk(_ramp610)
            @test dyn(_w610, _X610) == sto(_w610, _X610)
            @test dyn(_x610) == sto(_x610)
            @test isapprox(dyn(_w610, _X610), dyn(_x610); rtol = 1e-12)
            # Stored weights do not rebuild.
            @test PO.resolve_observation_weights(sto, _X610) === sto
            # Weights of the wrong length raise.
            @test_throws DimensionMismatch PO.resolve_observation_weights(sto,
                                                                          _X610[1:10, :])
        end
        # The rebuild passes the resolved weights to the variance estimator of the measure.
        r = PO.resolve_observation_weights(LowOrderMoment(; w = Ramp610(),
                                                          alg = SecondMoment()), _X610)
        @test r.w == _ramp610
        @test r.alg.ve.w == _ramp610
        r = PO.resolve_observation_weights(HighOrderMoment(; w = Ramp610(),
                                                           ve = SimpleVariance(;
                                                                               me = nothing)),
                                           _X610)
        @test r.ve.w == _ramp610
        @test PO.resolve_observation_weights(ThirdCentralMoment(; w = Ramp610(),
                                                                ve = SimpleVariance()),
                                             _X610).ve.w == _ramp610
        # A measure outside the union, and a member with no weights, return as themselves.
        r = Variance()
        @test PO.resolve_observation_weights(r, _X610) === r
        r = LowOrderMoment()
        @test PO.resolve_observation_weights(r, _X610) === r
        # The first lower moment with the ramp weights, by hand.
        mu = sum(_ramp610 .* _x610) / sum(_ramp610)
        hand = sum(_ramp610 .* max.(mu .- _x610, 0)) / sum(_ramp610)
        @test isapprox(LowOrderMoment(; w = Ramp610())(_x610), hand; rtol = 1e-12)
    end
end
