# The sweep of the wrapper measures (#607): `NoRisk` and its two guards, `RiskRatio` and
# `NonOptimisationRiskRatio`, `EqualRisk`, and the homogeneity correction of the risk
# contributions. Three defects: a ratio with a return in its numerator answered
# `bigger_is_better = false`, a ratio whose degree is not one and `VarianceSkewKurtosis` gave
# risk contributions that did not sum to the risk, and `EqualRisk` returned a `Float64` for
# every weight type.
using Clarabel, HiGHS, JuMP, LinearAlgebra, Statistics, StableRNGs

const PO = PortfolioOptimisers

const _X607 = randn(StableRNG(987456), 200, 5) * 0.01 .+ 0.0005
const _pr607 = prior(EmpiricalPrior(), _X607)
const _w607 = [0.3, 0.25, 0.2, 0.15, 0.1]
const _x607 = [0.125, -0.25, 0.0625, -0.125, 0.375]
const _clar607 = Solver(; solver = Clarabel.Optimizer, settings = Dict("verbose" => false))
const _highs607 = Solver(; solver = HiGHS.Optimizer,
                         settings = Dict("log_to_console" => false))

# The message of the `ArgumentError` that `f` throws.
function _msg607(f)
    try
        f()
    catch e
        if !(e isa ArgumentError)
            rethrow()
        end
        return e.msg
    end
    return error("no ArgumentError")
end

@testset "Wrapper risk measures" begin
    @testset "NoRisk measures zero in the element type of the returns" begin
        @test NoRisk()(_x607) === 0.0
        @test NoRisk()(Float32[0.1, -0.2]) === 0.0f0
        @test NoRisk()(Rational{Int}[1 // 2, -1 // 3]) === 0 // 1
        @test expected_risk(NoRisk(), _w607, _pr607) == 0
        @test iszero(risk_contribution(NoRisk(), _w607, _pr607))
        # The bound has no effect, because the settings refuse a negative bound.
        @test_throws DomainError RiskMeasureSettings(; ub = -1.0)
    end

    @testset "MeanRisk accepts NoRisk under the objectives that ignore the risk" begin
        wb = WeightBounds(; lb = 0.0, ub = 0.4)
        res = Dict(map((MaximumReturn(), MaximumUtility())) do obj
                       nameof(typeof(obj)) => optimise(MeanRisk(; r = NoRisk(), obj = obj,
                                                                opt = JuMPOptimiser(; pe = _pr607,
                                                                                    slv = _highs607, wb = wb)))
                   end)
        for (nm, r) in res
            @test isa(r.retcode, OptimisationSuccess)
            @test iszero(r.model[:risk])
            # Only affine rows: a linear programme, which HiGHS solves.
            @test all(t -> t[1] <: Union{JuMP.AffExpr, Vector{JuMP.AffExpr}},
                      JuMP.list_of_constraint_types(r.model))
        end
        # The utility loses its risk term, so the two objectives agree.
        @test isapprox(res[:MaximumUtility].w, res[:MaximumReturn].w; atol = 1e-8)
        # The weights fill the three largest means up to the bound.
        @test isapprox(res[:MaximumReturn].w,
                       [i in sortperm(_pr607.mu; rev = true)[1:2] ? 0.4 : 0.0 for i in 1:5] .+
                       [i == sortperm(_pr607.mu; rev = true)[3] ? 0.2 : 0.0 for i in 1:5];
                       atol = 1e-8)
    end

    @testset "The refusals name the cause" begin
        o = JuMPOptimiser(; pe = _pr607, slv = _clar607)
        vf = Variance(; settings = RiskMeasureSettings(; rke = false))
        no = "a `NoRisk` measure is present and it measures nothing"
        rk = "every risk measure carries `settings.rke = false`"
        for obj in (MinimumRisk(), MaximumRatio())
            msg = _msg607(() -> MeanRisk(; r = NoRisk(), obj = obj, opt = o))
            @test occursin("incompatible with $(nameof(typeof(obj)))", msg)
            @test occursin(no, msg)
            @test occursin(rk, _msg607(() -> MeanRisk(; r = vf, obj = obj, opt = o)))
        end
        # The mixed vector is refused too, with the NoRisk cause.
        @test occursin(no,
                       _msg607(() -> MeanRisk(; r = [NoRisk(), Variance()],
                                              obj = MinimumRisk(), opt = o)))
        for (T, f) in ((:RiskBudgeting, () -> RiskBudgeting(; r = NoRisk(), opt = o)),
                       (:FactorRiskContribution, () -> FactorRiskContribution(; r = NoRisk(), opt = o)),
                       (:NearOptimalCentering, () -> NearOptimalCentering(; r = NoRisk(), opt = o)),
                       (:HierarchicalRiskParity, () -> HierarchicalRiskParity(; r = NoRisk())),
                       (:HierarchicalEqualRiskContribution,
                        () -> HierarchicalEqualRiskContribution(; ri = NoRisk())),
                       (:HierarchicalEqualRiskContribution,
                        () -> HierarchicalEqualRiskContribution(; ro = NoRisk())))
            msg = _msg607(f)
            @test startswith(msg, "$T needs a non-zero risk expression, but $no")
        end
        # A risk expression that is not zero passes under every objective.
        @test isnothing(PO.assert_no_risk_objective_compatibility(Variance(),
                                                                  MinimumRisk()))
        @test isnothing(PO.assert_risk_measure_required(Variance(), :RiskBudgeting))
        # The JuMP optimisers refuse `rke = false`, the hierarchical ones do not read it.
        @test occursin(rk, _msg607(() -> RiskBudgeting(; r = vf, opt = o)))
        @test HierarchicalRiskParity(; r = vf) isa HierarchicalRiskParity
    end

    @testset "The two tests of a zero risk expression" begin
        vf = Variance(; settings = RiskMeasureSettings(; rke = false))
        @test PO.norisk_flag(NoRisk())
        @test PO.norisk_flag([Variance(), NoRisk()])
        @test !PO.norisk_flag(vf)
        @test !PO.norisk_flag(Variance())
        @test PO.zero_risk_expression_flag(NoRisk())
        @test PO.zero_risk_expression_flag(vf)
        @test PO.zero_risk_expression_flag([vf, vf])
        @test PO.zero_risk_expression_flag([NoRisk(), Variance()])
        # A measure that bounds the risk and stays out of the objective is not refused.
        @test !PO.zero_risk_expression_flag([Variance(),
                                             Variance(;
                                                      settings = RiskMeasureSettings(;
                                                                                     rke = false,
                                                                                     ub = 0.1))])
        @test !PO.zero_risk_expression_flag(Variance[])
        @test PO.zero_risk_expression_cause([NoRisk()]) ==
              "a `NoRisk` measure is present and it measures nothing"
        @test startswith(PO.zero_risk_expression_cause([vf]), "every risk measure carries")
        # The cause of the table of `norisk_flag`: the risk of NoRisk is zero, so the
        # minimum of the pair is zero and HRP divides zero by zero.
        @test iszero(PO.unitary_expected_risks(NoRisk(), _X607))
        @test iszero(expected_risk([NoRisk(), Variance()], _w607, _pr607;
                                   sca = MinScalariser()))
        @test expected_risk([NoRisk(), Variance()], _w607, _pr607; sca = MaxScalariser()) ==
              expected_risk(Variance(), _w607, _pr607)
    end

    @testset "A ratio is the quotient of its two sides" begin
        sd, cv = StandardDeviation(), ConditionalValueatRisk()
        @test expected_risk(RiskRatio(; r1 = sd, r2 = cv), _w607, _pr607) ==
              expected_risk(sd, _w607, _pr607) / expected_risk(cv, _w607, _pr607)
        r = NonOptimisationRiskRatio(; r1 = [sd, cv], sca1 = MaxScalariser(),
                                     r2 = [Variance(), WorstRealisation()],
                                     sca2 = SumScalariser())
        @test expected_risk(r, _w607, _pr607) ≈
              max(expected_risk(sd, _w607, _pr607), expected_risk(cv, _w607, _pr607)) /
              (expected_risk(Variance(), _w607, _pr607) +
               expected_risk(WorstRealisation(), _w607, _pr607))
        @test RiskRatio(; r1 = Range(), r2 = WorstRealisation())(_x607) == 2.5
        # A zero denominator follows IEEE division: a constant asset held alone. The constant
        # is a power of two, so every partial sum is exact, and the variance is zero on every
        # host. A constant of 0.001 gave a variance of 4.3e-37 on the CI host.
        Xz = hcat(fill(2.0^-10, 50), randn(StableRNG(1), 50, 2) * 0.01)
        prz = prior(EmpiricalPrior(), Xz)
        wz = [1.0, 0.0, 0.0]
        @test iszero(expected_risk(Variance(), wz, prz))
        @test expected_risk(RiskRatio(; r1 = cv, r2 = Variance()), wz, prz) == -Inf
        @test isnan(expected_risk(RiskRatio(; r1 = Variance(), r2 = sd), wz, prz))
    end

    @testset "The numerator decides the polarity of a ratio" begin
        bib = PO.bigger_is_better
        @test bib(NonOptimisationRiskRatio(; r1 = MeanReturn(), r2 = Variance()))
        @test !bib(NonOptimisationRiskRatio(; r1 = Variance(), r2 = MeanReturn()))
        @test !bib(NonOptimisationRiskRatio())
        @test !bib(RiskRatio())
        @test bib(NonOptimisationRiskRatio(; r1 = [MeanReturn(), MeanReturn()],
                                           r2 = Variance()))
        @test bib(MeanReturnRiskRatio())
        @test_throws ArgumentError bib(NonOptimisationRiskRatio(;
                                                                r1 = [MeanReturn(),
                                                                      Variance()]))
        # The export of `RiskRatio` lives in the file that defines it, and nowhere else.
        srcdir = joinpath(dirname(pathof(PortfolioOptimisers)), "16_RiskMeasures")
        owners = filter(readdir(srcdir; join = true)) do f
            return isfile(f) &&
                   any(l -> startswith(l, "export") && occursin(r"\bRiskRatio\b", l),
                       eachline(f))
        end
        @test basename.(owners) == ["19_RatioRiskMeasure.jl"]
    end

    @testset "EqualRisk splits every cluster evenly" begin
        @test EqualRisk()([0.5, 0.3, 0.2, 0.0]) === 0.25
        @test EqualRisk()(Float32[0.6, 0.4]) === 0.5f0
        @test EqualRisk()(Rational{Int}[1, 2, 3]) === 1 // 3
        @test EqualRisk()([1, 2, 3, 4]) === 0.25
        @test expected_risk(EqualRisk(), _w607, _pr607) == 0.2
        hrp = Dict(4 => fill(0.25, 4), 5 => [0.25, 0.25, 0.125, 0.125, 0.25],
                   6 => [0.25, 0.125, 0.25, 0.125, 0.125, 0.125])
        for N in (4, 5, 6)
            pr = prior(EmpiricalPrior(), randn(StableRNG(11), 300, N) * 0.01)
            opt = HierarchicalOptimiser(; pe = pr)
            @test optimise(HierarchicalRiskParity(; r = EqualRisk(), opt = opt)).w ≈ hrp[N]
            w = optimise(HierarchicalEqualRiskContribution(; ri = EqualRisk(),
                                                           ro = EqualRisk(), opt = opt)).w
            @test sum(w) ≈ 1
            # Each cluster splits evenly, and its assets share its weight equally.
            @test all(v -> any(c -> isapprox(v, c; atol = 1e-12),
                               (1 / 2, 1 / 4, 1 / 6, 1 / 8)), w)
        end
    end

    @testset "The degree of homogeneity of each measure" begin
        hd = PO.homogeneity_degree
        @test hd(ConditionalValueatRisk()) == 1
        @test hd(StandardDeviation()) == 1
        @test hd(NoRisk()) == 1
        @test hd(Variance()) == 2
        @test hd(Kurtosis(; alg = SOCRiskExpr())) == 2
        @test hd(LowOrderMoment(; alg = EvenMoment(; p = 3))) == 2
        @test hd(ThirdCentralMoment()) == 3
        @test hd(Skewness()) == 3
        @test hd(HighOrderMoment(; alg = ThirdLowerMoment())) == 3
        @test hd(HighOrderMoment(; alg = FourthMoment())) == 4
        @test hd(Kurtosis(; alg = QuadRiskExpr())) == 4
        @test hd(EqualRisk()) == 0
        @test hd(ThirdCentralMoment(; ve = SimpleVariance())) == 0
        # A variance estimator in `ve` moves the measure out of the cubed and the fourth power
        # unions, whose `alg` it shares, and into the scale-free union.
        sv = SimpleVariance(; me = nothing)
        @test hd(HighOrderMoment(; alg = FourthMoment(), ve = sv)) == 0
        @test hd(HighOrderMoment(; alg = FourthMoment(; alg = SemiMoment()), ve = sv)) == 0
        @test hd(HighOrderMoment(; alg = ThirdLowerMoment(), ve = sv)) == 0
        @test hd(RiskRatio()) == 1
        @test hd(RiskRatio(; r1 = StandardDeviation())) == 0
        @test hd(RiskRatio(; r1 = Kurtosis(; alg = QuadRiskExpr()))) == 3
        @test hd(NonOptimisationRiskRatio(; r1 = ConditionalValueatRisk(), r2 = Variance())) ==
              -1
        @test hd(NonOptimisationRiskRatio(; r1 = [Variance(), Variance()])) == 1
        @test isnothing(hd(NonOptimisationRiskRatio(;
                                                    r1 = [Variance(), StandardDeviation()])))
        @test isnothing(hd(NonOptimisationRiskRatio(; r1 = [Variance(), Variance()],
                                                    sca1 = LogSumExpScalariser())))
        @test hd(MeanReturnRiskRatio()) == 0
        @test isnothing(hd(MeanReturnRiskRatio(; rf = 0.01)))
        @test isnothing(hd(VarianceSkewKurtosis()))
        @test PO.adjust_risk_contribution(Variance(), 1 // 2) === 1 // 4
        @test PO.adjust_risk_contribution(ThirdCentralMoment(), 0.75) == 0.25
        @test PO.adjust_risk_contribution(EqualRisk(), 0.5) === 0.5
        @test PO.adjust_risk_contribution(VarianceSkewKurtosis(), 0.5) === 0.5
        @test PO.adjust_risk_contribution(NonOptimisationRiskRatio(;
                                                                   r1 = ConditionalValueatRisk(),
                                                                   r2 = Variance()),
                                          0.5) === -0.5
    end

    @testset "The contributions sum to the risk of a homogeneous measure" begin
        # By hand: the variance has the contributions `w .* (Σ * w)`.
        S = _pr607.sigma
        @test risk_contribution(Variance(), _w607, _pr607) ≈ _w607 .* (S * _w607)
        prh = prior(HighOrderPriorEstimator(), _X607)
        for r in (Variance(), StandardDeviation(), ConditionalValueatRisk(),
                  ThirdCentralMoment(), Skewness(), HighOrderMoment(; alg = FourthMoment()),
                  Kurtosis(; alg = SOCRiskExpr()), Kurtosis(; alg = QuadRiskExpr()),
                  LowOrderMoment(; alg = EvenMoment(; p = 3)),
                  NegativeSkewness(; alg = QuadRiskExpr()),
                  RiskRatio(; r1 = Kurtosis(; alg = QuadRiskExpr())),
                  NonOptimisationRiskRatio(; r1 = ConditionalValueatRisk(), r2 = Variance()),
                  VarianceSkewKurtosis(), [Variance(), VarianceSkewKurtosis()])
            rr = factory(r, prh)
            @test sum(risk_contribution(rr, _w607, prh)) ≈ expected_risk(rr, _w607, prh) rtol = 1e-6
        end
        # The skewness term of the composite has degree 3, by hand.
        v, s, k = PO.variance_skew_kurtosis_terms(factory(VarianceSkewKurtosis(), prh),
                                                  _w607)
        @test v ≈ dot(_w607, prh.sigma, _w607)
        @test s ≈ dot(_w607, prh.sk, kron(_w607, _w607))
        @test k ≈ dot(kron(_w607, _w607), prh.kt, kron(_w607, _w607))
        # The contributions of a measure of degree 0 sum to zero, and not to its risk. Each
        # one is not zero, except for EqualRisk, whose risk does not read the weights.
        for r in (ThirdCentralMoment(; ve = SimpleVariance()),
                  RiskRatio(; r1 = StandardDeviation()),
                  HighOrderMoment(; alg = FourthMoment(), ve = SimpleVariance(; me = nothing)))
            rr = factory(r, prh)
            rc = risk_contribution(rr, _w607, prh)
            @test !iszero(expected_risk(rr, _w607, prh))
            @test abs(sum(rc)) < 1e-8
            @test maximum(abs, rc) > 1e-3
        end
        @test iszero(risk_contribution(EqualRisk(), _w607, prh))
    end

    @testset "A measure with a fixed target has no degree" begin
        # `|x - c|` with a fixed `c ≠ 0` is not homogeneous in `w` (#1909). Each pair names a
        # measure and its degree. The loop checks the degree by its definition,
        # `R(c w) = c^d R(w)` at `c = 2` and `c = 3`. A measure with no degree fails the
        # definition for every `d`.
        hd = PO.homogeneity_degree
        w0 = fill(0.2, 5)
        mu = vec(mean(_X607; dims = 1))
        prh = prior(HighOrderPriorEstimator(), _X607)
        sv = SimpleVariance(; me = nothing)
        cases = [LowOrderMoment(; mu = 0.01) => nothing, LowOrderMoment(; mu = 0.0) => 1,
                 LowOrderMoment(; mu = mu) => 1,
                 LowOrderMoment(; mu = VecScalar(; v = mu, s = 0.01)) => nothing,
                 LowOrderMoment(; mu = VecScalar(; v = mu, s = 0.0)) => 1,
                 LowOrderMoment(; mu = 0.01, alg = SecondMoment()) => nothing,
                 LowOrderMoment(; mu = 0.01, alg = EvenMoment(; p = 3)) => nothing,
                 HighOrderMoment(; mu = 0.01) => nothing,
                 HighOrderMoment(; mu = 0.01, alg = FourthMoment()) => nothing,
                 HighOrderMoment(; mu = 0.01, ve = sv) => nothing,
                 HighOrderMoment(; mu = mu, alg = FourthMoment()) => 4,
                 ThirdCentralMoment(; mu = 0.01) => nothing,
                 ThirdCentralMoment(; mu = 0.01, ve = SimpleVariance()) => nothing,
                 ThirdCentralMoment(; mu = mu) => 3,
                 MedianAbsoluteDeviation(; mu = 0.01) => nothing,
                 MedianAbsoluteDeviation(; mu = mu) => 1, MedianAbsoluteDeviation() => 1,
                 TrackingRiskMeasure(; tr = WeightsTracking(; w = w0)) => nothing,
                 TrackingRiskMeasure(; tr = WeightsTracking(; w = w0),
                                     alg = SquaredL2Norm()) => nothing,
                 TrackingRiskMeasure(; tr = WeightsTracking(; w = zeros(5)),
                                     alg = SquaredL2Norm()) => 2,
                 TrackingRiskMeasure(; tr = ReturnsTracking(; w = _X607 * w0)) => nothing,
                 TrackingRiskMeasure(; tr = ReturnsTracking(; w = zeros(200))) => 1,
                 TrackingRiskMeasure(; tr = DataBenchmarkTracking(; B = _X607 * w0)) =>
                     nothing,
                 # A matrix benchmark gives the series `B * w`, which scales with `w`.
                 TrackingRiskMeasure(; tr = DataBenchmarkTracking(; B = 0.5 * _X607)) => 1,
                 RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = w0)) => nothing,
                 # With zero benchmark weights, the measure tracks its inner measure.
                 RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = zeros(5))) => 2,
                 RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = zeros(5)),
                                         r = ConditionalValueatRisk(),
                                         alg = DependentVariableTracking()) => 1,
                 TurnoverRiskMeasure(; w = w0) => nothing,
                 TurnoverRiskMeasure(; w = zeros(5)) => 1,
                 # A relative drawdown compounds the returns.
                 RelativeMaximumDrawdown() => nothing, RelativeAverageDrawdown() => nothing,
                 RelativeConditionalDrawdownatRisk() => nothing,
                 RelativeDrawdownatRisk() => nothing, RelativeUlcerIndex() => nothing,
                 MaximumDrawdown() => 1, ExpectedReturn() => 1,
                 ExpectedReturn(; rt = LogarithmicReturn()) => nothing,
                 ExpectedReturn(; rt = [ArithmeticReturn(), LogarithmicReturn()]) =>
                     nothing, ExpectedReturnRiskRatio() => -1,
                 ExpectedReturnRiskRatio(; rk = StandardDeviation()) => 0,
                 ExpectedReturnRiskRatio(; rf = 0.01) => nothing,
                 ExpectedReturnRiskRatio(; rt = LogarithmicReturn()) => nothing,
                 NonOptimisationRiskRatio(; r1 = Kurtosis(; alg = QuadRiskExpr()),
                                          r2 = TurnoverRiskMeasure(; w = w0)) => nothing]
        for (r, d) in cases
            rr = factory(r, prh)
            @test isequal(hd(r), d)
            @test isequal(hd(rr), d)
            r1 = expected_risk(rr, _w607, prh)
            q2 = expected_risk(rr, 2 * _w607, prh) / r1
            q3 = expected_risk(rr, 3 * _w607, prh) / r1
            if isnothing(d)
                d2 = log(abs(q2)) / log(2)
                d3 = log(abs(q3)) / log(3)
                @test !(isapprox(d2, d3; atol = 1e-6) &&
                        isapprox(d2, round(d2); atol = 1e-6))
            else
                @test q2 ≈ 2.0^d rtol = 1e-6
                @test q3 ≈ 3.0^d rtol = 1e-6
            end
        end
        # The solver measures of the relative drawdowns need no evaluation for their degree.
        for r in (RelativeEntropicDrawdownatRisk(), RelativePowerNormDrawdownatRisk(),
                  RelativeRelativisticDrawdownatRisk())
            @test isnothing(hd(r))
        end
        # An unbound data benchmark has no known target, so it claims no degree.
        @test PO.has_fixed_target(DataBenchmarkTracking())
        @test !PO.has_fixed_target(DataBenchmarkTracking(; B = zeros(200)))
        @test !PO.has_fixed_target(MeanCentering())
        # A ratio with no degree gives the plain marginal contributions. The old degree
        # `4 - 1 = 3` divided them by 3.
        r = factory(NonOptimisationRiskRatio(; r1 = Kurtosis(; alg = QuadRiskExpr()),
                                             r2 = TurnoverRiskMeasure(; w = w0)), prh)
        rc = risk_contribution(r, _w607, prh)
        @test rc ≈ _w607 .* risk_gradient(r, _w607, prh) rtol = 1e-6
        @test sum(rc) ≈ dot(_w607, risk_gradient(r, _w607, prh)) rtol = 1e-6
    end

    @testset "A fee that does not scale with the weights takes the degree" begin
        # A fixed fee, a turnover fee against weights that are not zero, and a liquidation
        # charge move the net returns by an amount that does not scale with `w` (#1910). A
        # fee that every period pays cancels from a measure that a constant shift does not
        # change. A one-off fee on the first observation cancels from no measure. Each
        # triple names a measure, a fee and the degree. The loop checks the degree by its
        # definition, and checks the contributions against the risk or the gradient.
        hd = PO.homogeneity_degree
        w0 = fill(0.2, 5)
        mu = vec(mean(_X607; dims = 1))
        prh = prior(HighOrderPriorEstimator(), _X607)
        tn = Fees(; tn = Turnover(; w = w0, val = 0.001))
        tn0 = Fees(; tn = Turnover(; w = zeros(5), val = 0.001))
        fl = Fees(; fl = 0.0001)
        fla = Fees(; fl = 0.0001, fa = AmortisedFees())
        l = Fees(; l = fill(0.001, 5))
        m2 = LowOrderMoment(; alg = SecondMoment())
        m20 = LowOrderMoment(; alg = SecondMoment(), mu = 0.0)
        cases = [(m2, tn, 2), (m2, fla, 2), (m2, fl, nothing), (m20, tn, nothing),
                 (m20, fla, nothing), (m20, l, 2), (m20, tn0, 2),
                 (LowOrderMoment(; alg = SecondMoment(), mu = mu), tn, 2),
                 (ThirdCentralMoment(), tn, 3), (ThirdCentralMoment(), fl, nothing),
                 (HighOrderMoment(; alg = FourthMoment()), tn, 4),
                 (BrownianDistanceVariance(), tn, 2),
                 (BrownianDistanceVariance(), fl, nothing), (Variance(), fl, 2),
                 (Variance(), tn, 2), (ConditionalValueatRisk(), tn, nothing),
                 (ConditionalValueatRisk(), tn0, 1), (MaximumDrawdown(), tn, nothing),
                 (TrackingRiskMeasure(; tr = WeightsTracking(; w = zeros(5)),
                                      alg = SquaredL2Norm()), tn, nothing),
                 (RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = zeros(5)), r = m2),
                  tn, 2),
                 (RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = zeros(5)), r = m2),
                  fl, nothing),
                 (RiskRatio(; r1 = ConditionalValueatRisk(), r2 = Variance()), tn, nothing),
                 (RiskRatio(; r1 = ConditionalValueatRiskRange(), r2 = Variance()), tn, -1),
                 (OrderedWeightsArray(; w = owa_gmd(200)), tn, 1),
                 (OrderedWeightsArray(; w = owa_cvar(200, 0.05)), tn, nothing),
                 (RiskRatio(; r1 = OrderedWeightsArray(), r2 = Variance()), tn, -1),
                 (RiskRatio(; r1 = OrderedWeightsArrayRange(), r2 = Variance()), tn, -1),
                 (RiskRatio(;
                            r1 = OrderedWeightsArray(;
                                                     w = OrderedWeightsArrayConditionalValueatRisk()),
                            r2 = Variance()), tn, nothing), (ExpectedReturn(), fl, nothing),
                 (ExpectedReturn(;
                                 rt = ArithmeticReturn(;
                                                       settings = JuMPReturnsSettings(;
                                                                                      fee = false))),
                  fl, 1)]
        T = size(_X607, 1)
        for (r, f, d) in cases
            rr = factory(r, prh)
            @test isequal(hd(r, f, T), d)
            @test isequal(hd(rr, f, T), d)
            r1 = expected_risk(rr, _w607, prh, f)
            q2 = expected_risk(rr, 2 * _w607, prh, f) / r1
            q3 = expected_risk(rr, 3 * _w607, prh, f) / r1
            if isnothing(d)
                d2 = log(abs(q2)) / log(2)
                d3 = log(abs(q3)) / log(3)
                @test !(isapprox(d2, d3; atol = 1e-6) &&
                        isapprox(d2, round(d2); atol = 1e-6))
            else
                @test q2 ≈ 2.0^d rtol = 1e-6
                @test q3 ≈ 3.0^d rtol = 1e-6
            end
        end
        # A measure with a degree under the fee decomposes its risk. A measure with none
        # gives the plain marginal contributions, which the old degree 2 divided by 2.
        for (r, f) in ((m2, tn), (m2, fla), (ThirdCentralMoment(), tn),
                       (RiskRatio(; r1 = ConditionalValueatRiskRange(), r2 = Variance()), tn))
            rr = factory(r, prh)
            @test sum(risk_contribution(rr, _w607, prh, f)) ≈
                  expected_risk(rr, _w607, prh, f) rtol = 1e-6
        end
        # An OWA measure sorts the returns, so its finite difference misses by 1.2e-6 here.
        # With no degree, the sum is minus the risk.
        rr = factory(RiskRatio(; r1 = OrderedWeightsArray(), r2 = Variance()), prh)
        @test sum(risk_contribution(rr, _w607, prh, tn)) ≈ expected_risk(rr, _w607, prh, tn) rtol = 1e-5
        for (r, f) in ((m2, fl), (m20, tn),
                       (RiskRatio(; r1 = ConditionalValueatRisk(), r2 = Variance()), tn))
            rr = factory(r, prh)
            g = risk_gradient(rr, _w607, prh, f)
            rc = risk_contribution(rr, _w607, prh, f)
            @test rc ≈ _w607 .* g rtol = 1e-6
            @test sum(rc) ≈ dot(_w607, g) rtol = 1e-6
        end
        # A liquidation charge does not read the weights, so it falls on each period.
        lq = Fees(; lq = Turnover(; w = [0.25], val = 0.01))
        @test PO.has_fixed_fee(lq, false)
        @test !PO.has_fixed_fee(lq, true)
        @test PO.has_fixed_fee(Fees(; flq = Turnover(; w = [0.25], val = 0.01)), true)
        @test !PO.has_fixed_fee(Fees(; tn = Turnover(; w = w0, val = 0.0)), false)
        @test !PO.has_fixed_fee(nothing, false)
        for f in (owa_gmd, owa_rg, owa_cvarrg, owa_tgrg, owa_l_moment), t in (20, 57, 200)
            @test PO.sums_to_zero(PO.resolve_owa_weights(f, t))
        end
        @test !PO.sums_to_zero(PO.resolve_owa_weights(owa_wr, T))
        @test PO.cancels_constant_fee(OrderedWeightsArray(), T)
        @test !PO.cancels_constant_fee(OrderedWeightsArray(;
                                                           w = OrderedWeightsArrayConditionalValueatRisk()),
                                       T)
        @test PO.cancels_constant_fee(OrderedWeightsArrayRange(; w1 = owa_tg(200)))
        @test PO.cancels_constant_fee(OrderedWeightsArrayRange(;
                                                               w1 = OrderedWeightsArrayTailGini()),
                                      T)
        # `owa_tg` and `owa_wr` both sum to -1, so their range does not move. Without `T`,
        # the sum of a builder is not known.
        @test PO.cancels_constant_fee(OrderedWeightsArrayRange(; w1 = owa_tg, w2 = owa_wr),
                                      T)
        @test !PO.cancels_constant_fee(OrderedWeightsArrayRange(; w1 = owa_tg,
                                                                w2 = owa_gmd), T)
        @test !PO.cancels_constant_fee(OrderedWeightsArrayRange())
        @test PO.cancels_constant_fee(GenericValueatRiskRange())
        @test !PO.reads_fees(ValueatRisk(; alg = DistributionValueatRisk()))
        @test !PO.reads_fees(VarianceSkewKurtosis())
        # A range reads the fees when one of its tails does.
        gr = GenericValueatRiskRange()
        @test PO.reads_fees(gr) == (PO.reads_fees(gr.loss) || PO.reads_fees(gr.gain))
        dvar = ValueatRisk(; alg = DistributionValueatRisk())
        @test !PO.reads_fees(GenericValueatRiskRange(; loss = dvar, gain = dvar))
        @test PO.sums_to_zero([1 // 2, -1 // 2])
        @test !PO.sums_to_zero([1 // 2, -1 // 3])
    end

    @testset "An OWA weight builder sums at the observation count of the data" begin
        # The sum of the weights of a builder can change with `T`, so `adjusted_risk` passes
        # the row count of the returns to the degree (#1911). A user closure whose weights sum
        # to zero keeps degree -1 on a side of a ratio under a fixed periodic fee, and its
        # contributions sum to the risk. `LinearMoment` at `k = 3` sums to 0.00332 at `T` =
        # 200, so the fee does not cancel, and the ratio has no degree.
        hd = PO.homogeneity_degree
        T = size(_X607, 1)
        prh = prior(HighOrderPriorEstimator(), _X607)
        tn = Fees(; tn = Turnover(; w = fill(0.2, 5), val = 0.001))
        rr = factory(RiskRatio(; r1 = OrderedWeightsArray(; w = t -> owa_gmd(t)),
                               r2 = Variance()), prh)
        @test isnothing(hd(rr, tn))
        @test hd(rr, tn, T) == -1
        @test sum(risk_contribution(rr, _w607, prh, tn)) ≈ expected_risk(rr, _w607, prh, tn) rtol = 1e-5
        @test PO.cancels_constant_fee(OrderedWeightsArray(; w = LinearMoment()), T)
        lm3 = OrderedWeightsArray(; w = LinearMoment(; k = 3))
        @test !PO.cancels_constant_fee(lm3, T)
        rr = factory(RiskRatio(; r1 = lm3, r2 = Variance()), prh)
        @test isnothing(hd(rr, tn, T))
        g = risk_gradient(rr, _w607, prh, tn)
        @test risk_contribution(rr, _w607, prh, tn) ≈ _w607 .* g rtol = 1e-6
        # A range of two builders reads both tails at `T`, so a vector tail and a builder
        # tail with the same sum cancel.
        @test PO.cancels_constant_fee(OrderedWeightsArrayRange(; w1 = owa_tg(T),
                                                               w2 = t -> reverse(owa_tg(t))),
                                      T)
    end
end
