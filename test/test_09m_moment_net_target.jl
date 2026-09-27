# A per-asset moment target centres a net series on its net mean (#1320). The series a moment
# measure reads is net of the fee, so a gross target `dot(w, mu)` shifted every deviation by the
# mean fee: the MAD model reported its functor plus the fee, and the Kurtosis functor disagreed
# with its fee-free model. Both levels now subtract the mean fee from a per-asset target.
using Clarabel, JuMP, Statistics

@testset "A per-asset moment target is the net mean (#1320)" begin
    rng = StableRNG(1049)
    T, N = 60, 4
    X = 0.01 .* randn(rng, T, N) .+ 0.001
    w = [0.4, 0.3, 0.2, 0.1]
    pr = prior(EmpiricalPrior(), X)
    prh = prior(HighOrderPriorEstimator(), X)
    mu = vec(mean(X; dims = 1))
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = Dict("verbose" => false))
    per = Fees(; l = 0.002)
    fixed = Fees(; l = 0.002, fl = 0.01)
    amort = Fees(; l = 0.002, fl = 0.01, fa = AmortisedFees())

    @testset "the target subtracts the mean fee per period" begin
        F(fees) = PortfolioOptimisers.term_fees(w, fees, T, true)
        mtf(m, fees) = PortfolioOptimisers.moment_target_fees(m, w, fees, T)
        # A per-asset target subtracts the mean fee; the one-off terms spread over `T`.
        @test mtf(mu, per) == F(per) == 0.002
        @test isapprox(mtf(mu, fixed), 0.002 + 4 * 0.01 / T; rtol = 1e-12)
        @test mtf(mu, amort) == mtf(mu, fixed)
        @test mtf(VecScalar(mu, 0.001), per) == F(per)
        @test iszero(mtf(mu, nothing))
        # A target stated on the net series takes no fee.
        @test iszero(mtf(nothing, per))
        @test iszero(mtf(0.001, per))
        @test iszero(mtf(MeanCentering(), per))
    end

    @testset "the prior mean agrees with the mean of the net series" begin
        # `mu` is the sample mean, so the net mean of the prior is the mean of the net series,
        # on either clock of the one-off terms.
        for fees in (nothing, per, fixed, amort)
            for r in (LowOrderMoment(), LowOrderMoment(; alg = MeanAbsoluteDeviation()),
                      LowOrderMoment(; alg = SecondMoment()),
                      LowOrderMoment(; alg = SecondMoment(; alg1 = SemiMoment())),
                      HighOrderMoment(; alg = ThirdLowerMoment()),
                      HighOrderMoment(; alg = FourthMoment()), ThirdCentralMoment())
                @test isapprox(factory(r, pr)(w, X, fees), r(w, X, fees); rtol = 1e-10)
            end
            @test isapprox(factory(Kurtosis(), prh)(w, X, fees), Kurtosis()(w, X, fees);
                           rtol = 1e-10)
            @test isapprox(MedianAbsoluteDeviation(; mu = mu)(w, X, fees),
                           MedianAbsoluteDeviation(; mu = MeanCentering())(w, X, fees);
                           rtol = 1e-10)
        end
        # A per period fee cancels from every deviation of a per-asset target.
        r = factory(LowOrderMoment(; alg = MeanAbsoluteDeviation()), pr)
        @test isapprox(r(w, X, per), r(w, X); rtol = 1e-12)
        # A scalar target is a threshold on the net series, so the fee moves the value.
        x = X * w .- 0.002
        @test isapprox(LowOrderMoment(; mu = 0.0)(w, X, per), mean(max.(-x, 0));
                       rtol = 1e-12)
    end

    @testset "the model agrees with the functor" begin
        tn = Turnover(; w = fill(0.25, N), val = 0.001)
        for fees in (per, Fees(; l = 0.002, tn = tn))
            for (r, p) in ((LowOrderMoment(), pr),
                           (LowOrderMoment(; alg = MeanAbsoluteDeviation()), pr),
                           (LowOrderMoment(; alg = SecondMoment()), pr),
                           (LowOrderMoment(; alg = SecondMoment(; alg1 = SemiMoment())), pr),
                           (LowOrderMoment(; alg = MeanAbsoluteDeviation(), mu = mu), pr),
                           (LowOrderMoment(; mu = VecScalar(mu, 0.0)), pr),
                           (LowOrderMoment(; mu = 0.0), pr), (Kurtosis(), prh))
                sol = optimise(MeanRisk(; r = r, obj = MinimumRisk(),
                                        opt = JuMPOptimiser(; pe = p, slv = slv,
                                                            fees = fees)))
                @test isa(sol.retcode, OptimisationSuccess)
                @test isapprox(expected_risk(factory(r, p, slv), sol.w, X, fees),
                               JuMP.value(sol.model[:risk]); atol = 1e-8)
            end
        end
    end
end
