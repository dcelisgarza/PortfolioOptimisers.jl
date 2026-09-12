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
# `ValueatRisk` and `ValueatRiskRange` split by formulation: `alg` selects the estimand, so
# the MIP branch scores a net return series and the parametric branch reads the prior's
# moments against the weights. Both branches are named, each in its own list. They are named
# by a CONCRETE type: `typeintersect` on two `alg`-bounded `UnionAll`s is a type with an
# uninhabited parameter rather than `Union{}`, so inference over a bound would report both
# kinds for either branch.
const _VAR_MIP = typeof(ValueatRisk(; alg = MIPValueatRisk()))
const _VAR_RANGE_MIP = typeof(ValueatRiskRange(; alg = MIPValueatRisk()))
const _VAR_DIST = typeof(ValueatRisk(; alg = DistributionValueatRisk()))
const _VAR_RANGE_DIST = typeof(ValueatRiskRange(; alg = DistributionValueatRisk()))

const _OLD_NETRETURNS = Any[WorstRealisation, _VAR_MIP, _VAR_RANGE_MIP,
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
                                    Skewness, MedianAbsoluteDeviation, VarianceSkewKurtosis,
                                    _VAR_DIST, _VAR_RANGE_DIST]
const _OLD_WEIGHTS = Any[StandardDeviation, NegativeSkewness, TurnoverRiskMeasure, Variance,
                         UncertaintySetVariance, EqualRisk]

# Composite / return-like measures handled by explicit `expected_risk` methods — they
# orchestrate other measures and intentionally declare no input kind.
const _EXPLICIT = Set{Any}([RiskRatio, NonOptimisationRiskRatio, MeanReturnRiskRatio,
                            ExpectedReturn, ExpectedReturnRiskRatio,
                            RiskTrackingRiskMeasure])

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

# Every kind a type can answer. A measure whose formulation selects the estimand answers
# more than one, and each must still be a concrete kind: an undeclared type infers
# `Union{}` from the erroring default and fails here.
function declared_kinds(@nospecialize(T))
    return unique(Base.return_types(PO.risk_input_kind, (T,)))
end

@testset "risk_input_kind — every concrete measure is classified" begin
    for T in all_concrete(PO.AbstractBaseRiskMeasure)
        if T in _EXPLICIT
            continue
        end
        ks = declared_kinds(T)
        @test !isempty(ks) && all(k -> isconcretetype(k) && k <: PO.RiskInputKind, ks)
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
                   RiskRatio(; r1 = ConditionalValueatRisk(), r2 = MaximumDrawdown()),
                   MeanReturnRiskRatio(; rk = ConditionalValueatRisk()),
                   TrackingRiskMeasure(; tr = ReturnsTracking(; w = _x_series .* 0.5))]
    for r in eligible
        @test PO.supports_precomputed_returns(r)
        v = g(r, _x_series)
        @test v isa Number && isfinite(v)
    end

    # A `ReturnsTracking` measure is dispatched on the *second* type parameter, because
    # `settings` is the first. The declarations and the single-vector functor must bind on a
    # concrete instance, not only on the `UnionAll` the completeness test below inspects.
    let r = TrackingRiskMeasure(; tr = ReturnsTracking(; w = _x_series .* 0.5))
        @test g(r, _x_series) ≈ r([1], reshape(_x_series, :, 1))
        @test g(r, _x_series) > 0
    end

    # Differential oracle: for the moment family the single-vector form equals the one-asset
    # `(w, X, fees)` form — the same oracle the entropy-pooling tests use (ADR 0007).
    for r in (LowOrderMoment(), HighOrderMoment(), Skewness(), Kurtosis(),
              MedianAbsoluteDeviation(), ThirdCentralMoment(), LowOrderMoment(; mu = 0.01))
        @test g(r, _x_series) ≈ r([1], reshape(_x_series, :, 1))
    end

    # Ineligible: the gate throws the explanatory `ArgumentError` — no silent wrong answer,
    # no raw `MethodError`. The default `RiskRatio` is ineligible via its
    # weights-only `Variance` constituent.
    mu2 = [0.1, 0.2]
    ineligible = Any[EqualRisk(), TurnoverRiskMeasure(; w = fill(inv(64), 64)),
                     StandardDeviation(; sigma = [1.0 0.0; 0.0 1.0]),
                     Variance(; sigma = [1.0 0.0; 0.0 1.0]), NegativeSkewness(),
                     VarianceSkewKurtosis(), RiskRatio(), LowOrderMoment(; mu = mu2),
                     HighOrderMoment(; mu = mu2), Skewness(; mu = mu2),
                     Kurtosis(; mu = mu2), MedianAbsoluteDeviation(; mu = mu2),
                     ThirdCentralMoment(; mu = mu2),
                     TrackingRiskMeasure(; tr = WeightsTracking(; w = [0.5, 0.5]))]
    for r in ineligible
        @test !PO.supports_precomputed_returns(r)
        @test_throws ArgumentError g(r, _x_series)
    end

    # The `WeightsTracking` functor is the second half of the same dispatch: a direct call
    # explains the refusal instead of raising a bare `MethodError`.
    @test_throws ArgumentError TrackingRiskMeasure(;
                                                   tr = WeightsTracking(; w = [0.5, 0.5]))(_x_series)
end

@testset "precomputed-returns contract — completeness" begin
    # EVERY concrete measure resolves `supports_precomputed_returns` to a `Bool` — a future
    # measure that fails to inherit an eligibility (e.g. via a missing `risk_input_kind`)
    # trips here rather than at runtime.
    #
    # The `_EXPLICIT` composites are NOT skipped here, and that is the whole point of the
    # loop being separate from the classification loop above. They are exempt from declaring
    # a *kind*, because an explicit `expected_risk` method routes each of them. They are not
    # exempt from resolving the *predicate*: `assert_scoreable` and
    # `expected_risk_from_returns` call it on whatever measure the user names, so a
    # composite that answers nothing raises the trait's internal "is not defined for" error
    # in a seam where the user needs a statement about their measure instead. #587 is that
    # defect: `ExpectedReturn` and `ExpectedReturnRiskRatio` reached `ScoreSelector` that
    # way, and the skip that used to stand here is why nothing caught it.
    for T in all_concrete(PO.AbstractBaseRiskMeasure)
        rt = reduce(typejoin, Base.return_types(PO.supports_precomputed_returns, (T,));
                    init = Union{})
        # `Bool <: rt` holds when the predicate resolves — `Bool`, or `Any` for a moment
        # UnionAll whose `mu` field is abstractly typed. It fails only on `Union{}`: an
        # undeclared measure whose `risk_input_kind` throws.
        @test Bool <: rt
    end
end

@testset "precomputed-returns contract — the composites that refuse a bare series" begin
    # #587. Three composites answered the predicate with the trait's internal "is not
    # defined for" error, because the skip above meant nothing asked them.
    #
    #   - A `PrRM` reads the prior result and contracts the expected returns it states with
    #     the weights. A bare series carries neither.
    #   - A `RiskTrackingRiskMeasure` tracks a benchmark held as a WEIGHT vector, so both of
    #     its functors need `w` as well.
    #
    # Each now answers `false`, and both seams that consult it then speak about the measure
    # the user named rather than about the trait.
    g = PO.expected_risk_from_returns
    for r in (ExpectedReturn(), ExpectedReturnRiskRatio(),
              RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = [0.5, 0.5])))
        @test !PO.supports_precomputed_returns(r)
        @test_throws ArgumentError g(r, _x_series)
        # The preselection seam is where the internal error used to reach the user.
        @test_throws ArgumentError ScoreSelector(; score = r,
                                                 rule = ThresholdRule(; lo = 0.0))
    end
end

@testset "the base-file selectors refuse a both-`nothing` call" begin
    # #595. `solver_selector` documented "Returns `nothing` if neither is available", and
    # the both-`nothing` method throws instead. The docstring now states the refusal, and
    # these two assertions hold it there. `sel` never reaches the solver refusal: a
    # both-`nothing` call routes on the operand types to `nothing_scalar_array_selector`.
    @test_throws ArgumentError PO.solver_selector(nothing, nothing)
    @test_throws ArgumentError PO.risk_measure_nothing_scalar_array_view(nothing, nothing,
                                                                         1)
    @test isnothing(PO.sel(nothing, nothing))
    @test isnothing(PO.nothing_scalar_array_selector(nothing, nothing))
end

# ── #773: the weight path is the picker for a scorer ──────────────────────────────────────
# Decision #772. A `VecNum` weight is one target vector and weighs every observation; a
# `MatNum` weight is a weight path, one row of weights per observation, which is what a fold
# scored under a Weight Drift held. `expected_risk`'s weight-taking family gains its `MatNum`
# mirror, and the three input kinds answer a path separately: a `NetReturnsInput` measure
# computes, and the two weight-reading kinds refuse by name.
const _path_X = [0.01 0.02; -0.02 0.01; 0.03 -0.01; 0.005 0.02]
const _path_w = [0.6, 0.4]

@testset "the weight path is the picker, and two kinds refuse it (#773)" begin
    wd = SelfFinancingDrift()
    U = PO.weight_path(wd, _path_w, _path_X)
    Uc = PO.weight_path(nothing, _path_w, _path_X)
    fees = Fees(; l = 0.001, fl = 0.002)
    @test U[1, :] == _path_w
    @test all(row == _path_w for row in eachrow(Uc))

    # A `NetReturnsInput` measure computes, and its answer is the measure over the series
    # the path produces. The kernel is untouched: `calc_net_returns` is what reads the type.
    rn = ConditionalValueatRisk()
    @test PO.risk_input_kind(rn) === PO.NetReturnsInput()
    @test expected_risk(rn, U, _path_X, fees) == rn(calc_net_returns(U, _path_X, fees))
    @test expected_risk(rn, U, _path_X) == rn(calc_net_returns(U, _path_X))
    # A drifted path moves the number, and a constant path reproduces the target reading.
    @test expected_risk(rn, U, _path_X, fees) != expected_risk(rn, _path_w, _path_X, fees)
    @test isapprox(expected_risk(rn, Uc, _path_X, fees),
                   expected_risk(rn, _path_w, _path_X, fees))

    # The prior route resolves the measure and the matrix once, then reads the path.
    rd = ReturnsResult(; nx = ["A", "B"], X = _path_X)
    rnr, Xr = PO.resolve_risk_inputs(rn, rd)
    @test expected_risk(rn, U, rd, fees) == expected_risk(rnr, U, Xr, fees)

    # A vector of measures scalarises over the path, `scale` is its combination weight, and
    # the vector's own prior route resolves once, exactly as the `VecNum` family does.
    rns = [ConditionalValueatRisk(), ConditionalValueatRisk(; alpha = 0.3)]
    @test expected_risk(rns, U, _path_X, fees) ==
          sum(expected_risk(r, U, _path_X, fees) * r.settings.scale for r in rns)
    rnsr, Xsr = PO.resolve_risk_inputs(rns, rd)
    @test expected_risk(rns, U, rd, fees) == expected_risk(rnsr, U, Xsr, fees)
    @test expected_risk(rns, U, _path_X, fees; sca = MaxScalariser()) !=
          expected_risk(rns, U, _path_X, fees)

    # The three ratio composites decompose onto the path as they do onto a vector.
    rr = RiskRatio(; r1 = ConditionalValueatRisk(), r2 = MaximumDrawdown())
    @test expected_risk(rr, U, _path_X, fees) ==
          expected_risk(rr.r1, U, _path_X, fees) / expected_risk(rr.r2, U, _path_X, fees)
    nrr = NonOptimisationRiskRatio(; r1 = ConditionalValueatRisk(), r2 = MaximumDrawdown())
    @test expected_risk(nrr, U, _path_X, fees) ==
          expected_risk(nrr.r1, U, _path_X, fees; sca = nrr.sca1) /
          expected_risk(nrr.r2, U, _path_X, fees; sca = nrr.sca2)
    mrr = MeanReturnRiskRatio(; rk = ConditionalValueatRisk())
    @test expected_risk(mrr, U, _path_X, fees) ==
          (expected_risk(mrr.rt, U, _path_X, fees) - mrr.rf) /
          expected_risk(mrr.rk, U, _path_X, fees; sca = mrr.sca)

    # `WeightsReturnsFeesInput` refuses a path by name. Its kernel reads `w` as one
    # cross-section, and a path gives one number per observation, which is a different
    # quantity rather than a wider input.
    rw = MedianAbsoluteDeviation()
    @test PO.risk_input_kind(rw) === PO.WeightsReturnsFeesInput()
    @test_throws ArgumentError expected_risk(rw, U, _path_X, fees)
    msg = try
        expected_risk(rw, U, _path_X, fees)
    catch e
        sprint(showerror, e)
    end
    @test occursin("MedianAbsoluteDeviation", msg)
    @test occursin("WeightsReturnsFeesInput", msg)
    # It still scores the target weights it does take.
    @test expected_risk(rw, _path_w, _path_X, fees) == rw(_path_w, _path_X, fees)

    # `WeightsInput` refuses on the same terms.
    rv = Variance(; sigma = [1.0 0.2; 0.2 1.0])
    @test PO.risk_input_kind(rv) === PO.WeightsInput()
    @test_throws ArgumentError expected_risk(rv, U, _path_X, fees)
    msg = try
        expected_risk(rv, U, _path_X, fees)
    catch e
        sprint(showerror, e)
    end
    @test occursin("Variance", msg)
    @test occursin("WeightsInput", msg)
    @test expected_risk(rv, _path_w, _path_X, fees) == rv(_path_w)

    # A refusal is taken by the kind and not by the argument count, so the no-fee call and
    # the prior route are refused too.
    @test_throws ArgumentError expected_risk(rw, U, _path_X)
    @test_throws ArgumentError expected_risk(rv, U, rd)
end
