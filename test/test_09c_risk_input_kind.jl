# Verifies the `risk_input_kind` trait that replaced the `expected_risk` routing unions
# (ADR 0006). Two guards:
#   1. Equivalence — every measure that used to live in a routing union now declares the
#      matching kind, so the trait-based routing reproduces the old union-based routing.
#   2. Completeness — every concrete `AbstractBaseRiskMeasure` either declares a kind or is
#      a composite handled by an explicit `expected_risk` method, so a future measure added
#      without a kind fails here rather than at runtime.
using InteractiveUtils: InteractiveUtils
const PO = PortfolioOptimisers

# Snapshot of the OLD routing-union membership (pre-trait); the differential baseline.
const _OLD_NETRETURNS = Any[WorstRealisation, ValueatRisk, ValueatRiskRange,
                            ConditionalValueatRisk,
                            DistributionallyRobustConditionalValueatRisk,
                            DistributionallyRobustConditionalValueatRiskRange,
                            EntropicValueatRisk, EntropicValueatRiskRange,
                            RelativisticValueatRisk, RelativisticValueatRiskRange,
                            DrawdownatRisk, MaximumDrawdown, AverageDrawdown,
                            ConditionalDrawdownatRisk,
                            DistributionallyRobustConditionalDrawdownatRisk, UlcerIndex,
                            EntropicDrawdownatRisk, RelativisticDrawdownatRisk,
                            RelativeDrawdownatRisk, RelativeMaximumDrawdown,
                            RelativeAverageDrawdown, RelativeConditionalDrawdownatRisk,
                            RelativeUlcerIndex, RelativeEntropicDrawdownatRisk,
                            RelativeRelativisticDrawdownatRisk, Range,
                            ConditionalValueatRiskRange, OrderedWeightsArray,
                            OrderedWeightsArrayRange, BrownianDistanceVariance, MeanReturn,
                            PowerNormValueatRisk, PowerNormValueatRiskRange,
                            PowerNormDrawdownatRisk, RelativePowerNormDrawdownatRisk]
const _OLD_WEIGHTSRETURNSFEES = Any[LowOrderMoment, HighOrderMoment, TrackingRiskMeasure,
                                    RiskTrackingRiskMeasure, Kurtosis, ThirdCentralMoment,
                                    Skewness, MedianAbsoluteDeviation, VarianceSkewKurtosis]
const _OLD_WEIGHTS = Any[StandardDeviation, NegativeSkewness, TurnoverRiskMeasure, Variance,
                         UncertaintySetVariance, EqualRiskMeasure]

# Composite / return-like measures handled by explicit `expected_risk` methods — they
# orchestrate other measures and intentionally declare no input kind.
const _EXPLICIT = Set{Any}([RiskRatioRiskMeasure, NonOptimisationRiskRatioRiskMeasure,
                            MeanReturnRiskRatio, ExpectedReturn, ExpectedReturnRiskRatio])

# The declared kind as a *type* — instance-free, since each method returns a singleton.
# Undeclared types hit the erroring default (inferred `Union{}`).
function declared_kind(@nospecialize(T))
    return reduce(typejoin, Base.return_types(PO.risk_input_kind, (T,)); init = Union{})
end

function all_concrete(@nospecialize(T))
    acc = Any[]
    for S in InteractiveUtils.subtypes(T)
        isabstracttype(S) ? append!(acc, all_concrete(S)) : push!(acc, S)
    end
    return acc
end

@testset "risk_input_kind — equivalence with old routing unions" begin
    # The three kinds partition the measures; no type may be classified two ways.
    @test allunique(vcat(_OLD_NETRETURNS, _OLD_WEIGHTSRETURNSFEES, _OLD_WEIGHTS))
    for T in _OLD_NETRETURNS
        @test declared_kind(T) === PO.NetReturnsInput
    end
    for T in _OLD_WEIGHTSRETURNSFEES
        @test declared_kind(T) === PO.WeightsReturnsFeesInput
    end
    for T in _OLD_WEIGHTS
        @test declared_kind(T) === PO.WeightsInput
    end
end

@testset "risk_input_kind — every concrete measure is classified" begin
    for T in all_concrete(PO.AbstractBaseRiskMeasure)
        if T in _EXPLICIT
            continue
        end
        k = declared_kind(T)
        @test isconcretetype(k) && k <: PO.RiskInputKind
    end
end

# ── ADR 0007: the precomputed-returns contract ────────────────────────────────────────────
# `expected_risk_from_returns(r, x)` evaluates a measure on an already-reduced net-return
# series. It is gated by `supports_precomputed_returns` so that an ineligible measure — a
# `WeightsInput` measure (whose `r(w)` shares the `r(::VecNum)` signature and would otherwise
# silently score the series as weights), a moment measure with a per-asset `mu` (whose target
# `dot(w, mu)` needs the absent weights), or a variance-carrying composite — throws an
# explanatory error rather than returning nonsense or hitting a raw `MethodError`.
const _x_series = [sinpi(2i / 64) * 0.1 + cospi(i / 32) * 0.03 for i in 1:64]

@testset "precomputed-returns contract — eligibility & differential" begin
    g = PO.expected_risk_from_returns
    # Eligible: NetReturnsInput measures, the weight-independent-target moment family, and
    # ratio composites whose constituents are themselves eligible.
    eligible = Any[ConditionalValueatRisk(), MaximumDrawdown(), ValueatRisk(),
                   WorstRealisation(), Range(), MeanReturn(), LowOrderMoment(),
                   HighOrderMoment(), Skewness(), Kurtosis(), MedianAbsoluteDeviation(),
                   ThirdCentralMoment(), LowOrderMoment(; mu = 0.01),
                   MedianAbsoluteDeviation(; mu = PO.MeanCentering()),
                   RiskRatioRiskMeasure(; r1 = ConditionalValueatRisk(),
                                        r2 = MaximumDrawdown()),
                   MeanReturnRiskRatio(; rk = ConditionalValueatRisk())]
    for r in eligible
        @test PO.supports_precomputed_returns(r)
        v = g(r, _x_series)
        @test v isa Number && isfinite(v)
    end

    # Differential oracle: for the moment family the single-vector form equals the one-asset
    # `(w, X, fees)` form — the same oracle the entropy-pooling tests use (ADR 0007).
    for r in (LowOrderMoment(), HighOrderMoment(), Skewness(), Kurtosis(),
              MedianAbsoluteDeviation(), ThirdCentralMoment(), LowOrderMoment(; mu = 0.01))
        @test g(r, _x_series) ≈ r([1], reshape(_x_series, :, 1))
    end

    # Ineligible: the gate throws the explanatory `ArgumentError` — no silent wrong answer,
    # no raw `MethodError`. The default `RiskRatioRiskMeasure` is ineligible via its
    # weights-only `Variance` constituent.
    mu2 = [0.1, 0.2]
    ineligible = Any[EqualRiskMeasure(), TurnoverRiskMeasure(; w = fill(inv(64), 64)),
                     StandardDeviation(; sigma = [1.0 0.0; 0.0 1.0]),
                     Variance(; sigma = [1.0 0.0; 0.0 1.0]), NegativeSkewness(),
                     VarianceSkewKurtosis(), RiskRatioRiskMeasure(),
                     LowOrderMoment(; mu = mu2), HighOrderMoment(; mu = mu2),
                     Skewness(; mu = mu2), Kurtosis(; mu = mu2),
                     MedianAbsoluteDeviation(; mu = mu2), ThirdCentralMoment(; mu = mu2)]
    for r in ineligible
        @test !PO.supports_precomputed_returns(r)
        @test_throws ArgumentError g(r, _x_series)
    end
end

@testset "precomputed-returns contract — completeness" begin
    # Every kind-classified concrete measure resolves `supports_precomputed_returns` to a
    # `Bool` — a future measure that fails to inherit an eligibility (e.g. via a missing
    # `risk_input_kind`) trips here rather than at runtime. The `_EXPLICIT` composites are
    # covered behaviourally above.
    for T in all_concrete(PO.AbstractBaseRiskMeasure)
        if T in _EXPLICIT
            continue
        end
        rt = reduce(typejoin, Base.return_types(PO.supports_precomputed_returns, (T,));
                    init = Union{})
        # `Bool <: rt` holds when the predicate resolves — `Bool`, or `Any` for a moment
        # UnionAll whose `mu` field is abstractly typed. It fails only on `Union{}`: an
        # undeclared measure whose `risk_input_kind` throws.
        @test Bool <: rt
    end
end

#=
I think the failure is due to running the tests/script when the test environment was out of sync with the worktree. When i started a session with

```bash
julia --project
```

and ran the init code in `test/runtests.jl` in the REPL followed by `test_09c` also in the REPL, i got this

```julia
julia> @testset "risk_input_kind — equivalence with old routing unions" begin
           # The three kinds partition the measures; no type may be classified two ways.
           @test allunique(vcat(_OLD_NETRETURNS, _OLD_WEIGHTSRETURNSFEES, _OLD_WEIGHTS))
           for T in _OLD_NETRETURNS
               @test declared_kind(T) === PO.NetReturnsInput
           end
           for T in _OLD_WEIGHTSRETURNSFEES
               @test declared_kind(T) === PO.WeightsReturnsFeesInput
           end
           for T in _OLD_WEIGHTS
               @test declared_kind(T) === PO.WeightsInput
           end
       end
Test Summary:                                         | Pass  Total  Time
risk_input_kind — equivalence with old routing unions |   51     51  1.1s
Test.DefaultTestSet("risk_input_kind — equivalence with old routing unions", Any[], 51, false, false, true, 1.781186434598269e9, 1.781186435665745e9, false, "REPL[19]", Random.Xoshiro(0x8738dd43ca12461a, 0x15592c6ca14d5bb6, 0xe256aaf2f4cfe209, 0x0ea8a7bd9ede2ed8, 0x99924f77ce1d0b51))

julia> @testset "risk_input_kind — every concrete measure is classified" begin
           for T in all_concrete(PO.AbstractBaseRiskMeasure)
               if T in _EXPLICIT
                   continue
               end
               k = declared_kind(T)
               @test isconcretetype(k) && k <: PO.RiskInputKind
           end
       end
Test Summary:                                          | Pass  Total  Time
risk_input_kind — every concrete measure is classified |   50     50  0.2s
Test.DefaultTestSet("risk_input_kind — every concrete measure is classified", Any[], 50, false, false, true, 1.78118643752396e9, 1.781186437731832e9, false, "REPL[20]", Random.Xoshiro(0x8738dd43ca12461a, 0x15592c6ca14d5bb6, 0xe256aaf2f4cfe209, 0x0ea8a7bd9ede2ed8, 0x99924f77ce1d0b51))
```

I don't know how to fix this, but the reason is clear: the test environment is not synced to the worktree environment. The fix is to properly syncronise it, but i don't know how.
=#
