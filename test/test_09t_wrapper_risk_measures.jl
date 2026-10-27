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
end
