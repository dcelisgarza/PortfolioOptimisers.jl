# The observation weights of the risk measures (#1621): the constructor and evaluation checks of
# `assert_observation_weights` and `checked_observation_weights`, the weighted Ulcer Index, and
# the significance level zero of the four tail measures, which is the largest loss among the
# observations with positive weight. The models are checked against the functors.
using Clarabel, JuMP

# A `DynamicAbstractWeights` that resolves to one weight too many, and one that resolves to a
# negative weight. Neither has values when the constructor runs, so evaluation must catch them.
struct LongObsWeights09p <: PortfolioOptimisers.DynamicAbstractWeights end
function PortfolioOptimisers.get_observation_weights(::LongObsWeights09p,
                                                     X::PortfolioOptimisers.VecNum_MatNum;
                                                     kwargs...)
    return pweights(ones(size(X, 1) + 1))
end
struct NegativeObsWeights09p <: PortfolioOptimisers.DynamicAbstractWeights end
function PortfolioOptimisers.get_observation_weights(::NegativeObsWeights09p,
                                                     X::PortfolioOptimisers.VecNum_MatNum;
                                                     kwargs...)
    return pweights([-1.0; ones(size(X, 1) - 1)])
end

@testset "Risk measure observation weights" begin
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = "verbose" => false)
    rng = StableRNG(7)
    x = 0.02 .* randn(rng, 40)
    w = pweights(rand(rng, 40))
    dd = PortfolioOptimisers.absolute_drawdown_vec(x)
    rdd = PortfolioOptimisers.relative_drawdown_vec(x)
    # The worst loss sits on the first observation, which carries no weight.
    xz = copy(x)
    xz[1] = -0.5
    wz = pweights([0.0; ones(39)])
    ddz = PortfolioOptimisers.absolute_drawdown_vec(xz)

    @testset "Constructors refuse invalid weights" begin
        @test_throws DomainError UlcerIndex(; w = pweights(zeros(40)))
        @test_throws DomainError RelativeUlcerIndex(; w = pweights([1.0, -1.0]))
        @test_throws DomainError ConditionalValueatRisk(; w = pweights(zeros(3)))
        @test_throws IsEmptyError AverageDrawdown(; w = pweights(Float64[]))
        @test_throws DomainError MeanReturn(; w = pweights([1.0, -2.0]))
        # `LowOrderMoment` checked only emptiness before.
        @test_throws DomainError LowOrderMoment(; w = pweights([1.0, -1.0]))
        @test_throws DomainError LowOrderMoment(; w = pweights(zeros(2)))
        @test_nowarn LowOrderMoment(; w = pweights([0.0, 1.0]))
        # A dynamic weight has no values yet, so the constructor passes it.
        @test_nowarn UlcerIndex(; w = NegativeObsWeights09p())
    end

    @testset "Evaluation refuses weights that do not fit the data" begin
        long = pweights(rand(StableRNG(1), 41))
        # The weighted quantile kernels read the weights through the sort permutation, so a
        # long weight vector used to pass in silence.
        @test_throws DimensionMismatch ConditionalValueatRisk(; w = long)(x)
        @test_throws DimensionMismatch ConditionalDrawdownatRisk(; w = long)(x)
        @test_throws DimensionMismatch ValueatRisk(; w = long)(x)
        @test_throws DimensionMismatch UlcerIndex(; w = long)(x)
        @test_throws DimensionMismatch AverageDrawdown(; w = long)(x)
        # Stored weights of a moment measure are checked against the data.
        @test_throws DimensionMismatch LowOrderMoment(; w = long)(x)
        @test_throws DimensionMismatch HighOrderMoment(; w = long)(x)
        @test_throws DimensionMismatch LowOrderMoment(; w = long)(fill(0.2, 5),
                                                                  randn(StableRNG(2), 40,
                                                                        5))
        # Dynamic weights are checked when they resolve.
        @test_throws DimensionMismatch UlcerIndex(; w = LongObsWeights09p())(x)
        @test_throws DomainError UlcerIndex(; w = NegativeObsWeights09p())(x)
        @test_throws DimensionMismatch LowOrderMoment(; w = LongObsWeights09p())(x)
        @test isnothing(PortfolioOptimisers.checked_observation_weights(nothing, x))
        @test PortfolioOptimisers.checked_observation_weights(w, x) === w
        @test isnothing(PortfolioOptimisers.assert_observation_weights(nothing, :w))
    end

    @testset "Weighted Ulcer Index" begin
        @test UlcerIndex(; w = w)(x) ≈ sqrt(sum(w .* dd .^ 2) / sum(w))
        @test RelativeUlcerIndex(; w = w)(x) ≈ sqrt(sum(w .* rdd .^ 2) / sum(w))
        # Equal weights give the unweighted index.
        @test UlcerIndex(; w = pweights(fill(3.0, 40)))(x) ≈ UlcerIndex()(x)
        @test UlcerIndex()(x) ≈ sqrt(sum(dd .^ 2) / 40)
        @test RelativeUlcerIndex()(x) ≈ sqrt(sum(rdd .^ 2) / 40)
        # A zero weight drops the observation from the mean, not from the path.
        @test UlcerIndex(; w = wz)(xz) ≈ sqrt(sum(ddz[2:end] .^ 2) / 39)
        # The prior's weights reach a measure that states none.
        X = 0.01 .* randn(StableRNG(5), 40, 3)
        pr0 = prior(EmpiricalPrior(), X)
        pr = LowOrderPrior(; X = pr0.X, mu = pr0.mu, sigma = pr0.sigma, w = w)
        wp = fill(inv(3), 3)
        ddp = PortfolioOptimisers.absolute_drawdown_vec(X * wp)
        @test expected_risk(UlcerIndex(), wp, pr) ≈ sqrt(sum(w .* ddp .^ 2) / sum(w))
        @test expected_risk(UlcerIndex(), wp, pr0) ≈ sqrt(sum(ddp .^ 2) / 40)
    end

    @testset "Significance level zero" begin
        @test_throws DomainError ConditionalValueatRisk(; alpha = 1)
        @test_throws DomainError ConditionalValueatRisk(; alpha = -0.1)
        @test_throws DomainError ValueatRisk(; alpha = 0)
        @test_throws DomainError ConditionalValueatRiskRange(; alpha = 0)
        @test_throws DomainError PortfolioOptimisers.assert_half_open_unit_interval(1.0)
        @test isnothing(PortfolioOptimisers.assert_half_open_unit_interval(nothing))
        # Unweighted, level zero reads every observation.
        @test ConditionalValueatRisk(; alpha = 0)(xz) ≈ 0.5
        @test EntropicValueatRisk(; slv = slv, alpha = 0)(xz) ≈ 0.5
        @test ConditionalValueatRisk(; alpha = 0)(x) ≈ WorstRealisation()(x)
        @test ConditionalDrawdownatRisk(; alpha = 0)(x) ≈ MaximumDrawdown()(x)
        @test EntropicDrawdownatRisk(; slv = slv, alpha = 0)(x) ≈ MaximumDrawdown()(x)
        @test RelativeConditionalDrawdownatRisk(; alpha = 0)(x) ≈
              RelativeMaximumDrawdown()(x)
        @test RelativeEntropicDrawdownatRisk(; slv = slv, alpha = 0)(x) ≈
              RelativeMaximumDrawdown()(x)
        # Weighted, the zero-weight loss cannot set the risk, but it stays in the path.
        @test ConditionalValueatRisk(; alpha = 0, w = wz)(xz) ≈ -minimum(xz[2:end])
        @test EntropicValueatRisk(; slv = slv, alpha = 0, w = wz)(xz) ≈ -minimum(xz[2:end])
        @test ConditionalDrawdownatRisk(; alpha = 0, w = wz)(xz) ≈ -minimum(ddz[2:end])
        @test EntropicDrawdownatRisk(; slv = slv, alpha = 0, w = wz)(xz) ≈
              -minimum(ddz[2:end])
        @test -minimum(ddz[2:end]) > 0.4
        @test WorstRealisation()(xz) ≈ 0.5
        # Level zero bounds a small positive level from above.
        @test ConditionalValueatRisk(; alpha = 0, w = w)(x) >=
              ConditionalValueatRisk(; alpha = 0.05, w = w)(x)
    end

    @testset "JuMP models agree with the functors" begin
        X = 0.01 .* randn(StableRNG(11), 120, 4) .+ 0.0005
        # Every asset loses on the first observation, which carries no weight.
        X[1, :] .= -0.2
        rd = ReturnsResult(; nx = string.('A':'D'), X = X)
        pr = prior(EmpiricalPrior(), rd)
        wo = pweights(rand(StableRNG(3), 120))
        wo0 = pweights([0.0; ones(119)])
        opt = JuMPOptimiser(; pe = pr, slv = slv)
        function solved_risk(r)
            res = optimise(MeanRisk(; r = r, obj = MinimumRisk(), opt = opt), rd)
            @test isa(res.retcode, OptimisationSuccess)
            return JuMP.value(res.model[:risk]),
                   expected_risk(factory(r, pr, slv), res.w, pr)
        end
        for r in (UlcerIndex(), UlcerIndex(; w = wo), UlcerIndex(; w = wo0),
                  ConditionalValueatRisk(; alpha = 0),
                  ConditionalValueatRisk(; alpha = 0, w = wo0), EntropicValueatRisk(; alpha = 0),
                  EntropicValueatRisk(; alpha = 0, w = wo0),
                  ConditionalDrawdownatRisk(; alpha = 0),
                  ConditionalDrawdownatRisk(; alpha = 0, w = wo0),
                  EntropicDrawdownatRisk(; alpha = 0),
                  EntropicDrawdownatRisk(; alpha = 0, w = wo0))
            model_risk, functor_risk = solved_risk(r)
            @test isapprox(model_risk, functor_risk; rtol = 1e-6)
        end
        # Unweighted, level zero is the worst realisation and the maximum drawdown.
        @test isapprox(solved_risk(ConditionalValueatRisk(; alpha = 0))[1],
                       solved_risk(WorstRealisation())[1]; rtol = 1e-6)
        @test isapprox(solved_risk(ConditionalDrawdownatRisk(; alpha = 0))[1],
                       solved_risk(MaximumDrawdown())[1]; rtol = 1e-6)
        # The zero-weight loss of 0.2 sets the unweighted worst loss and no weighted one.
        @test solved_risk(ConditionalValueatRisk(; alpha = 0))[1] ≈ 0.2 atol = 1e-6
        @test solved_risk(ConditionalValueatRisk(; alpha = 0, w = wo0))[1] < 0.1
        @test solved_risk(EntropicValueatRisk(; alpha = 0, w = wo0))[1] < 0.1
        # Two Ulcer Index measures with different weights build two expressions.
        res = optimise(MeanRisk(;
                                r = [UlcerIndex(; w = wo),
                                     UlcerIndex(; w = wo0,
                                                settings = RiskMeasureSettings(;
                                                                               scale = 2.0))],
                                obj = MinimumRisk(), opt = opt), rd)
        @test isa(res.retcode, OptimisationSuccess)
        m = res.model
        @test haskey(m, :uci_risk_1) && haskey(m, :uci_risk_2)
        @test JuMP.value(m[:uci_risk_1]) ≈ UlcerIndex(; w = wo)(X * res.w) rtol = 1e-6
        @test JuMP.value(m[:uci_risk_2]) ≈ UlcerIndex(; w = wo0)(X * res.w) rtol = 1e-6
        # Weights of the wrong length are refused when the model is built.
        @test_throws DimensionMismatch optimise(MeanRisk(;
                                                         r = UlcerIndex(;
                                                                        w = pweights(ones(119))),
                                                         obj = MinimumRisk(), opt = opt),
                                                rd)
    end
end
