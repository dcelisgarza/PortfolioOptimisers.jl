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
