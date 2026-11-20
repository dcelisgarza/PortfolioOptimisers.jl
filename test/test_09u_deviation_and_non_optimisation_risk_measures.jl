# The sweep of the deviation measures (#609): `MedianAbsoluteDeviation` with its two centring
# functions, and the measures that no optimiser takes, `MeanReturn`, `ThirdCentralMoment` and
# `MeanReturnRiskRatio`. Two defects: observation weights moved the centre of a median absolute
# deviation and not its median, and `ThirdCentralMoment` answered `bigger_is_better = false`
# where `Skewness`, the same moment over the cube of the standard deviation, answers `true`.
# `ThirdCentralMoment` also took a non-finite `mu`, which the two sibling measures refuse.
using Clarabel, Distributions, LinearAlgebra, Statistics, StatsBase, StableRNGs

const PO = PortfolioOptimisers

# A skewed sample: its median 0.01 and its mean 0.36 / 7 differ.
const _x609 = [0.01, -0.02, 0.03, 0.10, -0.01, 0.00, 0.25]
const _X609 = randn(StableRNG(609), 120, 4) * 0.01 .+ 0.001
const _w609 = [0.1, 0.2, 0.3, 0.4]
const _rd609 = ReturnsResult(; nx = ["a", "b", "c", "d"], X = _X609)

@testset "Deviation and non-optimisation risk measures" begin
    @testset "The two centrings of a median absolute deviation, by hand" begin
        x = _x609
        X = reshape(x, :, 1)
        w = [1.0]
        @test median(x) == 0.01
        @test isapprox(mean(x), 0.36 / 7)
        # About the median, |x - 0.01| sorted has the median 0.02.
        @test isapprox(MedianAbsoluteDeviation(; flag = false)(w, X), 0.02)
        # About the mean, |x - 0.36/7| sorted has the median 0.36/7, the deviation of 0.
        @test isapprox(MedianAbsoluteDeviation(; mu = MeanCentering(), flag = false)(w, X),
                       0.36 / 7)
        # The precomputed form agrees with the form of the weights and the returns.
        x2 = _X609 * _w609
        for mu in (MedianCentering(), MeanCentering(), 0.001)
            r = MedianAbsoluteDeviation(; mu = mu)
            @test r(x2) == r(_w609, _X609)
        end
    end

    @testset "flag = true scales by the consistency constant 1 / Φ⁻¹(3/4)" begin
        c = inv(quantile(Normal(), 3 / 4))
        @test isapprox(c, 1.4826022185056018)
        @test isapprox(MedianAbsoluteDeviation()(_x609) /
                       MedianAbsoluteDeviation(; flag = false)(_x609), c)
        # The unweighted value is the one of `StatsBase.mad`, bit for bit.
        y = randn(StableRNG(3), 1001)
        @test MedianAbsoluteDeviation()(y) ==
              StatsBase.mad(y .- median(y); center = 0.0, normalize = true)
        # On a large Gaussian draw, the scaled measure estimates the standard deviation, and
        # the unscaled measure misses it by the factor.
        g = randn(StableRNG(1), 200_000) * 0.02
        @test isapprox(MedianAbsoluteDeviation()(g), std(g); rtol = 1e-2)
        @test isapprox(MedianAbsoluteDeviation(; flag = false)(g) * c, std(g); rtol = 1e-2)
        @test !isapprox(MedianAbsoluteDeviation(; flag = false)(g), std(g); rtol = 0.2)
        # The element type of the data stays.
        @test isa(MedianAbsoluteDeviation()(Float32.(_x609)), Float32)
    end

    @testset "Observation weights weight both the centre and the median" begin
        x = _x609
        ow = pweights([1.0, 1, 1, 1, 1, 1, 10])
        c = median(x, ow)
        @test c == 0.1375
        # The weighted median of |x - 0.1375|. Before the fix the code took the unweighted
        # median of the same deviations, 0.1275.
        @test isapprox(MedianAbsoluteDeviation(; w = ow, flag = false)(x), 0.11075)
        @test isapprox(MedianAbsoluteDeviation(; w = ow, flag = false)(x),
                       median(abs.(x .- c), ow))
        cm = mean(x, ow)
        @test isapprox(MedianAbsoluteDeviation(; w = ow, mu = MeanCentering(),
                                               flag = false)(x), median(abs.(x .- cm), ow))
        # Equal weights give the unweighted value.
        eq = pweights(fill(1.0, length(x)))
        @test MedianAbsoluteDeviation(; w = eq, flag = false)(x) ==
              MedianAbsoluteDeviation(; flag = false)(x)
        # The weighted form of the weights and the returns agrees with the precomputed form.
        ow2 = pweights(collect(range(0.5, 1.5; length = size(_X609, 1))))
        r = MedianAbsoluteDeviation(; w = ow2)
        @test r(_w609, _X609) == r(_X609 * _w609)
    end

    @testset "A stated centre of a median absolute deviation" begin
        x = _X609 * _w609
        mu = vec(mean(_X609; dims = 1))
        s = 0.0003
        k = inv(quantile(Normal(), 3 / 4))
        @test isapprox(MedianAbsoluteDeviation(; mu = VecScalar(; v = mu, s = s))(_w609,
                                                                                  _X609),
                       median(abs.(x .- (dot(_w609, mu) + s))) * k)
        @test isapprox(MedianAbsoluteDeviation(; mu = mu)(_w609, _X609),
                       median(abs.(x .- dot(_w609, mu))) * k)
        @test isapprox(MedianAbsoluteDeviation(; mu = 0.001)(_w609, _X609),
                       median(abs.(x .- 0.001)) * k)
        @test PO.weight_independent_target(MedianCentering()) === true
        @test PO.weight_independent_target(MeanCentering()) === true
        # A centring function has nothing to slice, so a view of a subset keeps it.
        @test PO.nothing_scalar_array_view(MedianCentering(), [1, 2]) === MedianCentering()
        @test PO.port_opt_view(MedianAbsoluteDeviation(; mu = MeanCentering()), [1, 2]).mu ===
              MeanCentering()
        @test PO.port_opt_view(MedianAbsoluteDeviation(; mu = mu), [1, 2]).mu == mu[[1, 2]]
        @test PO.supports_precomputed_returns(MedianAbsoluteDeviation())
        @test PO.supports_precomputed_returns(MedianAbsoluteDeviation(; mu = 0.0))
        @test !PO.supports_precomputed_returns(MedianAbsoluteDeviation(; mu = mu))
        @test !PO.supports_precomputed_returns(MedianAbsoluteDeviation(;
                                                                       mu = VecScalar(;
                                                                                      v = mu,
                                                                                      s = s)))
    end

    @testset "MeanReturn by hand" begin
        x = _x609
        T = length(x)
        ow = pweights([1.0, 2, 1, 1, 3, 1, 1])
        @test isapprox(MeanReturn()(x), sum(x) / T)
        @test isapprox(MeanReturn(; flag = true)(x), sum(log1p.(x)) / T)
        @test isapprox(MeanReturn(; w = ow)(x), 0.032)
        @test isapprox(MeanReturn(; w = ow, flag = true)(x), sum(ow .* log1p.(x)) / sum(ow))
        @test PO.bigger_is_better(MeanReturn())
    end

    @testset "ThirdCentralMoment: divisor T, not standardised" begin
        x = _x609
        T = length(x)
        m = mean(x)
        m3 = sum((x .- m) .^ 3) / T
        @test isapprox(ThirdCentralMoment()(x), m3)
        # The skewness coefficient of the same sample is a different number.
        @test isapprox(m3 / std(x; corrected = false)^3, 1.4435733989503812)
        @test !isapprox(ThirdCentralMoment()(x), m3 / std(x; corrected = false)^3)
        ow = pweights([1.0, 2, 1, 1, 3, 1, 1])
        mw = sum(ow .* x) / sum(ow)
        @test isapprox(ThirdCentralMoment(; w = ow)(x), sum(ow .* (x .- mw) .^ 3) / sum(ow))
        @test isapprox(ThirdCentralMoment(; mu = 0.0)(x), sum(x .^ 3) / T)
        # A stated centre, through the weights and the returns.
        xp = _X609 * _w609
        mu = vec(mean(_X609; dims = 1))
        s = 0.0003
        @test isapprox(ThirdCentralMoment(; mu = VecScalar(; v = mu, s = s))(_w609, _X609),
                       mean((xp .- (dot(_w609, mu) + s)) .^ 3))
        @test isapprox(ThirdCentralMoment(; mu = mu)(_w609, _X609),
                       mean((xp .- dot(_w609, mu)) .^ 3))
        @test isapprox(ThirdCentralMoment(; mu = 0.001)(_w609, _X609),
                       mean((xp .- 0.001) .^ 3))
        @test PO.supports_precomputed_returns(ThirdCentralMoment())
        @test !PO.supports_precomputed_returns(ThirdCentralMoment(; mu = mu))
        # The constructor refuses a non-finite centre, as its siblings do.
        @test_throws PO.IsNonFiniteError ThirdCentralMoment(; mu = Inf)
        @test_throws PO.IsNonFiniteError ThirdCentralMoment(; mu = [0.1, NaN])
        @test_throws PO.IsEmptyError ThirdCentralMoment(; mu = Float64[])
    end

    @testset "ThirdCentralMoment is better when larger, as Skewness is" begin
        @test PO.bigger_is_better(ThirdCentralMoment()) === true
        @test PO.bigger_is_better(Skewness()) === true
        @test PO.bigger_is_better(ThirdCentralMoment(; ve = SimpleVariance())) === true
        # A vector of the two now agrees on its polarity.
        @test PO.bigger_is_better([Skewness(), ThirdCentralMoment()]) === true
    end

    @testset "MeanReturnRiskRatio: both sides, and the Sharpe ratio" begin
        X = [_x609 2 * _x609]
        w = [0.25, 0.75]
        rf = 0.001
        r = MeanReturnRiskRatio(; rf = rf)
        num = expected_risk(MeanReturn(), w, X) - rf
        den = expected_risk(ConditionalValueatRisk(), w, X)
        @test isapprox(expected_risk(r, w, X), num / den)
        @test isapprox(r(X * w), num / den)
        @test PO.bigger_is_better(r)
        # With a standard deviation in `rk`, the ratio is the reward-to-variability ratio.
        pr = prior(EmpiricalPrior(), _rd609)
        sr = factory(MeanReturnRiskRatio(; rk = StandardDeviation(), rf = 0.0002), pr)
        x = _X609 * _w609
        @test isapprox(expected_risk(sr, _w609, _X609), (mean(x) - 0.0002) / std(x))
        # A zero risk follows IEEE division.
        z = fill(0.5, 8)
        rk = LowOrderMoment(; alg = MeanAbsoluteDeviation())
        @test iszero(rk(z))
        @test MeanReturnRiskRatio(; rk = rk)(z) == Inf
        @test isnan(MeanReturnRiskRatio(; rk = rk, rf = 0.5)(z))
        @test MeanReturnRiskRatio(; rk = rk, rf = 1.0)(z) == -Inf
        @test_throws PO.IsEmptyError MeanReturnRiskRatio(; rk = ConditionalValueatRisk[])
    end

    @testset "needs_previous_weights of the ratio answers as rk does" begin
        tn = TurnoverRiskMeasure(; w = fill(0.25, 4))
        @test !PO.needs_previous_weights(MeanReturnRiskRatio())
        @test PO.needs_previous_weights(MeanReturnRiskRatio(; rk = tn))
        @test PO.needs_previous_weights(MeanReturnRiskRatio(;
                                                            rk = [ConditionalValueatRisk(),
                                                                  tn]))
        @test !PO.needs_previous_weights(MeanReturnRiskRatio(;
                                                             rk = TurnoverRiskMeasure(;
                                                                                      w = fill(0.25,
                                                                                               4),
                                                                                      fixed = true)))
        @test !PO.needs_previous_weights(MeanReturn())
        @test !PO.needs_previous_weights(ThirdCentralMoment())
        @test !PO.needs_previous_weights(MedianAbsoluteDeviation())
    end

    @testset "The optimisers refuse the measures they cannot take" begin
        slv = Solver(; solver = Clarabel.Optimizer, settings = Dict("verbose" => false))
        opt = JuMPOptimiser(; slv = slv)
        for r in (MeanReturn(), ThirdCentralMoment(), MeanReturnRiskRatio(),
                  MedianAbsoluteDeviation())
            err = try
                MeanRisk(; r = r, opt = opt)
                nothing
            catch e
                e
            end
            @test isa(err, TypeError)
            @test occursin("RiskMeasure", sprint(showerror, err))
        end
        for r in (MeanReturn(), ThirdCentralMoment(), MeanReturnRiskRatio())
            for mk in (r -> HierarchicalRiskParity(; r = r),
                       r -> HierarchicalEqualRiskContribution(; ri = r, ro = r))
                err = try
                    mk(r)
                    nothing
                catch e
                    e
                end
                @test isa(err, TypeError)
                @test occursin("OptimisationRiskMeasure", sprint(showerror, err))
            end
        end
        # A hierarchical optimiser takes the median absolute deviation.
        for o in (HierarchicalRiskParity(; r = MedianAbsoluteDeviation()),
                  HierarchicalEqualRiskContribution(; ri = MedianAbsoluteDeviation(),
                                                    ro = MedianAbsoluteDeviation()))
            res = optimise(o, _rd609)
            @test isa(res.retcode, OptimisationSuccess)
            @test isapprox(sum(res.w), 1)
            @test all(>(0), res.w)
        end
    end
end
