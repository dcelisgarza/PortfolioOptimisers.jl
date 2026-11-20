# The sweep of the drawdown and range scalars (#606): `AverageDrawdown`, `UlcerIndex`,
# `MaximumDrawdown` and their relative pairs, `WorstRealisation` and `Range`. One short path
# answers every measure against a hand computation, and the same path separates each absolute
# measure from its relative pair. The degenerate paths, the limits of the tail measures and the
# ordered-weights forms of the worst realisation and the range follow. Two defects: the weighted
# `average_drawdown` called `Statistics.mean(dd, w)`, which reads a plain weight vector as a
# function, and `owa_rg(1)` returned `[1]`, so the ordered-weights range of one observation was
# that observation and not zero.
using LinearAlgebra, Statistics, StatsBase

const PO = PortfolioOptimisers

# Every partial sum and every partial product of this path is exact in binary.
const _x606 = [0.125, -0.25, 0.0625, -0.125, 0.375]
const _d606 = [0.0, -0.25, -0.1875, -0.3125, 0.0]
const _rd606 = [0.0, -0.25, -0.203125, -0.302734375, -0.041259765625]
const _w606 = pweights([1.0, 1.0, 1.0, 1.0, 4.0])

@testset "Drawdown and range risk measures" begin
    @testset "One path, every measure against a hand computation" begin
        x = _x606
        @test PO.absolute_drawdown_vec(x) == _d606
        @test PO.relative_drawdown_vec(x) == _rd606
        @test AverageDrawdown()(x) == 0.15
        @test AverageDrawdown()(x) ≈ -mean(_d606)
        @test UlcerIndex()(x) ≈ sqrt(0.1953125 / 5)
        @test UlcerIndex()(x) ≈ sqrt(mean(abs2, _d606))
        @test MaximumDrawdown()(x) == 0.3125
        @test WorstRealisation()(x) == 0.25
        @test Range()(x) == 0.625
        @test RelativeAverageDrawdown()(x) == 0.159423828125
        @test RelativeAverageDrawdown()(x) ≈ -mean(_rd606)
        @test RelativeUlcerIndex()(x) ≈ sqrt(mean(abs2, _rd606))
        @test RelativeMaximumDrawdown()(x) == 0.302734375
    end
    @testset "Absolute against relative" begin
        x = _x606
        # The last return lifts the sum to a new peak, but not the compounded wealth.
        @test last(PO.absolute_drawdown_vec(x)) == 0
        @test last(PO.relative_drawdown_vec(x)) < 0
        @test RelativeAverageDrawdown()(x) > AverageDrawdown()(x)
        @test RelativeUlcerIndex()(x) > UlcerIndex()(x)
        # The deepest point is shallower when compounded: the fall from 1.125 is 30.27 %.
        @test RelativeMaximumDrawdown()(x) < MaximumDrawdown()(x)
        # One fall from the start is the same in both series.
        for (a, r) in ((AverageDrawdown(), RelativeAverageDrawdown()),
                       (UlcerIndex(), RelativeUlcerIndex()),
                       (MaximumDrawdown(), RelativeMaximumDrawdown()))
            @test a([-0.1]) ≈ 0.1
            @test r([-0.1]) ≈ 0.1
        end
        # The wealth stays positive when every return is above -1, so the relative
        # maximum drawdown stays below one, where the uncompounded one does not.
        @test RelativeMaximumDrawdown()([-0.9, -0.9]) ≈ 0.99
        @test MaximumDrawdown()([-0.9, -0.9]) ≈ 1.8
    end
    @testset "Observation weights" begin
        x = _x606
        @test AverageDrawdown(; w = _w606)(x) == 0.09375
        @test AverageDrawdown(; w = _w606)(x) ≈ -sum(_w606 .* _d606) / sum(_w606)
        @test UlcerIndex(; w = _w606)(x) == 0.15625
        @test RelativeAverageDrawdown(; w = _w606)(x) ≈ -sum(_w606 .* _rd606) / sum(_w606)
        @test RelativeUlcerIndex(; w = _w606)(x) ≈
              sqrt(sum(_w606 .* _rd606 .^ 2) / sum(_w606))
        # A zero weight drops the period from the mean, but its return still moves every
        # later drawdown.
        w0 = pweights([1.0, 0.0, 1.0, 1.0, 1.0])
        @test AverageDrawdown(; w = w0)(x) ≈ (0.1875 + 0.3125) / 4
        @test AverageDrawdown(; w = w0)(x) != AverageDrawdown()(x[[1, 3, 4, 5]])
        # The kernels take any weight vector: a plain one gave a MethodError in
        # `average_drawdown`.
        wv = collect(_w606)
        for dd in (_d606, _rd606)
            @test PO.average_drawdown(dd, wv) ≈ PO.average_drawdown(dd, _w606)
            @test PO.ulcer_index(dd, wv) ≈ PO.ulcer_index(dd, _w606)
            @test PO.average_drawdown(dd, ones(5)) ≈ PO.average_drawdown(dd, nothing)
            @test PO.ulcer_index(dd, ones(5)) ≈ PO.ulcer_index(dd, nothing)
        end
        # A root mean square is never below the mean of the same nonnegative values.
        for w in (nothing, _w606, w0)
            @test UlcerIndex(; w = w)(x) >= AverageDrawdown(; w = w)(x)
            @test RelativeUlcerIndex(; w = w)(x) >= RelativeAverageDrawdown(; w = w)(x)
        end
    end
    @testset "Degenerate paths" begin
        dd_rms = (AverageDrawdown(), RelativeAverageDrawdown(), UlcerIndex(),
                  RelativeUlcerIndex(), MaximumDrawdown(), RelativeMaximumDrawdown())
        # A path that never falls below its running peak has no drawdown.
        for x in ([0.01, 0.02, 0.0, 0.03], zeros(4), [0.1])
            for r in dd_rms
                @test iszero(r(x))
            end
        end
        @test WorstRealisation()([0.01, 0.02, 0.0, 0.03]) == 0
        @test WorstRealisation()([0.1, 0.2]) == -0.1
        @test iszero(WorstRealisation()(zeros(4)))
        @test Range()([0.01, 0.02, 0.0, 0.03]) == 0.03
        # One observation, or equal returns, has no range.
        @test Range()([-0.1]) == 0
        @test Range()([0.1]) == 0
        @test Range()(fill(0.02, 3)) == 0
        @test WorstRealisation()([-0.1]) == 0.1
    end
    @testset "The limits of the tail measures" begin
        x = _x606
        lo, hi = 1e-12, 1 - 1e-12
        @test ConditionalDrawdownatRisk(; alpha = hi)(x) ≈ AverageDrawdown()(x)
        @test ConditionalDrawdownatRisk(; alpha = lo)(x) ≈ MaximumDrawdown()(x)
        @test DrawdownatRisk(; alpha = lo)(x) ≈ MaximumDrawdown()(x)
        @test RelativeConditionalDrawdownatRisk(; alpha = hi)(x) ≈
              RelativeAverageDrawdown()(x)
        @test RelativeConditionalDrawdownatRisk(; alpha = lo)(x) ≈
              RelativeMaximumDrawdown()(x)
        @test ValueatRisk(; alpha = lo)(x) ≈ WorstRealisation()(x)
        @test ConditionalValueatRisk(; alpha = lo)(x) ≈ WorstRealisation()(x)
        @test ValueatRiskRange(; alpha = lo, beta = lo)(x) ≈ Range()(x)
        @test ConditionalValueatRiskRange(; alpha = lo, beta = lo)(x) ≈ Range()(x)
        # The maximum drawdown is the worst realisation of the drawdown series.
        @test WorstRealisation()(PO.absolute_drawdown_vec(x)) == MaximumDrawdown()(x)
        @test WorstRealisation()(PO.relative_drawdown_vec(x)) ==
              RelativeMaximumDrawdown()(x)
        @test Range()(x) ≈ WorstRealisation()(x) + maximum(x)
    end
    @testset "Range and WorstRealisation against their ordered weights" begin
        @test owa_rg(1) == [0.0]
        @test owa_rg(2) == [-1.0, 1.0]
        @test owa_rg(4) == [-1.0, 0.0, 0.0, 1.0]
        for x in (_x606, [-0.1], [0.3, -0.2], [0.05, 0.05, 0.05], 0.01 .* sin.(1:40))
            T = length(x)
            @test OrderedWeightsArray(; w = owa_rg(T))(x) ≈ Range()(x) atol = 1e-15
            @test OWA_RG()(x) ≈ Range()(x) atol = 1e-15
            @test OrderedWeightsArray(; w = owa_wr(T))(x) ≈ WorstRealisation()(x)
            @test OWA_WR()(x) ≈ WorstRealisation()(x)
        end
    end
end
