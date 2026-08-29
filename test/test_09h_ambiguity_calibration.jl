#=
An **Ambiguity Radius** is the radius of the ball of probability measures a distributionally
robust model prices, and an **Esfahani-Kuhn tail weight** is the weight of the tail term of
the loss it minimises. Twelve slots across six types hold one or the other, and each of them
takes a **Calibration Rule** in place of the number.

`test_09f_calibration_slot.jl` covers the mechanism and `test_09g_calibration_rules.jl`
covers the three rules of the two older families. This file covers the two families this
ticket adds, the two radius rules it ships, and the twelve slots.

The two families take the same shape as the two older ones: an abstract family under
`AbstractCalibrationAlgorithm`, a `Func_` bound for the `alg` field that admits a plain
`Function`, one role type, and a `Num_` bound that pairs the role with `Number`. A rule is
run by CALLING it, so a callable struct and a plain function are the same thing to the
resolver.

A radius names no end of the distribution, so each family carries ONE role rather than the
two that a tail probability carries, and `mirror_role` has nothing to carry across.

Issue #584 widens the slots. Issue #311 settled the design, and its correction binds: `r` is
the radius and `l` is the tail weight, not the other way round.
=#
using Clarabel, JuMP, Distributions, InteractiveUtils
const PO = PortfolioOptimisers

const RNG = StableRNG(246813579)
const X60 = randn(RNG, 60, 4)
const PR60 = prior(EmpiricalPrior(), X60)
const X120 = randn(RNG, 120, 4)
const PR120 = prior(EmpiricalPrior(), X120)
const WTS = pweights(range(; start = 1, stop = 2, length = 60))
# A two-asset universe is the corner where `DimensionalRateRadius` reduces to the
# square-root rate, and a twenty-asset one at two record lengths is where it parts
# company with `RateRadius`.
const X60_2 = X60[:, 1:2]
const PR60_2 = prior(EmpiricalPrior(), X60_2)
const X125_20 = randn(RNG, 125, 20)
const PR125_20 = prior(EmpiricalPrior(), X125_20)
const X250_20 = randn(RNG, 250, 20)
const PR250_20 = prior(EmpiricalPrior(), X250_20)

# A radius rule with no type at all. A closure over a caller's own number is the case that
# cannot be given one, and it is why the `alg` bound admits a bare `Function`.
probe_radius(::Symbol, pr::PO.AbstractPriorResult, ::Any, ::Any) = 3 / size(pr.X, 1)

# The tail-weight family ships no rule, so a caller's own function is the whole of its
# population. This one reports the key, so the resolver's argument order is asserted.
const TWT_SEEN = Ref{Any}(nothing)
function probe_tail_weight(key::Symbol, pr::PO.AbstractPriorResult, w, slv)
    TWT_SEEN[] = (; key = key, weighted = !isnothing(w), solved = !isnothing(slv))
    return 2 / sqrt(size(pr.X, 1))
end

@testset "Ambiguity calibration: the two families join the calibration root" begin
    # Both families sit under the one root, beside the two the mechanism already carried.
    @test PO.AbstractAmbiguityRadiusCalibrationAlgorithm <: PO.AbstractCalibrationAlgorithm
    @test PO.AbstractAmbiguityTailWeightCalibrationAlgorithm <:
          PO.AbstractCalibrationAlgorithm

    # Neither family is the other, and neither is a significance or a deformation family.
    @test !(PO.AbstractAmbiguityRadiusCalibrationAlgorithm <:
            PO.AbstractAmbiguityTailWeightCalibrationAlgorithm)
    @test !(PO.AbstractAmbiguityRadiusCalibrationAlgorithm <:
            PO.AbstractSignificanceCalibrationAlgorithm)
    @test !(PO.AbstractAmbiguityTailWeightCalibrationAlgorithm <:
            PO.AbstractDeformationCalibrationAlgorithm)

    # Both roles are Estimators, not Algorithms. #593 split the taxonomy so that a role
    # inside another role's `alg` field is refused by the field's bound.
    @test AmbiguityRadiusCalibration <: PO.AbstractCalibrationEstimator
    @test AmbiguityTailWeightCalibration <: PO.AbstractCalibrationEstimator
    @test !(AmbiguityRadiusCalibration <: PO.AbstractCalibrationAlgorithm)
    @test !(AmbiguityTailWeightCalibration <: PO.AbstractCalibrationAlgorithm)

    # Neither abstract type is exported: an export is public API, and the convention is
    # that an abstract type is not one.
    @test :AbstractAmbiguityRadiusCalibrationAlgorithm ∉ names(PortfolioOptimisers)
    @test :AbstractAmbiguityTailWeightCalibrationAlgorithm ∉ names(PortfolioOptimisers)

    # The four concrete names are exported, on the same terms as the roles and rules of the
    # two older families.
    for sym in
        (:AmbiguityRadiusCalibration, :AmbiguityTailWeightCalibration, :ConcentrationRadius,
         :RateRadius, :DimensionalRateRadius)
        @test sym ∈ names(PortfolioOptimisers)
    end

    # The two rules that ship both compute a radius, and neither joins the tail-weight
    # family. Nothing computes an Esfahani-Kuhn tail weight, and that is deliberate.
    @test ConcentrationRadius <: PO.AbstractAmbiguityRadiusCalibrationAlgorithm
    @test RateRadius <: PO.AbstractAmbiguityRadiusCalibrationAlgorithm
    @test DimensionalRateRadius <: PO.AbstractAmbiguityRadiusCalibrationAlgorithm
    @test isempty(filter(t -> t !== AmbiguityTailWeightCalibration,
                         subtypes(PO.AbstractAmbiguityTailWeightCalibrationAlgorithm)))
end

@testset "Ambiguity calibration: the roles, the bounds and the family split" begin
    rrule = RateRadius(; c = 0.2)
    rrole = AmbiguityRadiusCalibration(; alg = rrule)
    trole = AmbiguityTailWeightCalibration(; alg = probe_tail_weight)

    # The `alg` bound admits the family's own rules and a plain function, and refuses the
    # other family's rule. A function carries no family, so no bound can refuse it.
    @test isa(rrule, PO.Func_AmbRadCal)
    @test isa(ConcentrationRadius(), PO.Func_AmbRadCal)
    @test isa(probe_radius, PO.Func_AmbRadCal)
    @test isa(probe_tail_weight, PO.Func_AmbTwtCal)
    @test !isa(rrule, PO.Func_AmbTwtCal)
    @test !isa(ScenarioCount(; n = 25), PO.Func_AmbRadCal)
    @test !isa(rrule, PO.Func_SigCal)
    @test !isa(rrule, PO.Func_DefCal)

    # The role is the whole of the type: the rule lives in `alg`.
    @test rrole.alg === rrule
    @test trole.alg === probe_tail_weight

    # A rule of the wrong family in an `alg` field is refused at construction, by the
    # bound. No guard method is written for it.
    @test_throws TypeError AmbiguityRadiusCalibration(; alg = ScenarioCount(; n = 25))
    @test_throws TypeError AmbiguityTailWeightCalibration(; alg = rrule)
    @test_throws TypeError AmbiguityRadiusCalibration(; alg = 0.02)

    # A role is not a rule, so a role inside an `alg` field is refused on the same terms.
    @test !isa(rrole, PO.Func_AmbRadCal)
    @test !isa(trole, PO.Func_AmbTwtCal)
    @test_throws TypeError AmbiguityRadiusCalibration(; alg = rrole)
    @test_throws TypeError AmbiguityTailWeightCalibration(; alg = trole)

    # The slot bound names one role and no other, so a tail-weight role in a radius slot
    # fails the constructor's signature.
    @test isa(rrole, PO.Num_AmbRadCal)
    @test isa(0.02, PO.Num_AmbRadCal)
    @test !isa(trole, PO.Num_AmbRadCal)
    @test isa(trole, PO.Num_AmbTwtCal)
    @test !isa(rrole, PO.Num_AmbTwtCal)
    @test !isa(SignificanceTailCalibration(; alg = ScenarioCount(; n = 25)),
               PO.Num_AmbRadCal)

    # A radius names no end of the distribution, so neither role mirrors: `mirror_role`
    # carries a number across and knows nothing else.
    @test PO.mirror_role(0.02) == 0.02
    @test_throws MethodError PO.mirror_role(rrole)
    @test_throws MethodError PO.mirror_role(trole)

    # `sel` keeps a role rather than falling back to the prior's value, which is what lets
    # a rule survive the selection that runs before the resolution.
    @test PO.sel(rrole, 0.02) === rrole
    @test PO.sel(trole, 1.0) === trole
end

@testset "ConcentrationRadius: the chi-squared form" begin
    # A stated scale is used as given, and the chi-squared factor is dimensionless.
    alg = ConcentrationRadius(; confidence = 0.95, scale = 0.5)
    q = Distributions.cquantile(Distributions.Chisq(4), 0.05)
    @test alg(:r, PR60, nothing, nothing) ≈ 0.5 * sqrt(q / 60)

    # A longer sample shrinks the ball, at the square-root rate.
    @test alg(:r, PR120, nothing, nothing) ≈ 0.5 * sqrt(q / 120)
    @test alg(:r, PR60, nothing, nothing) / alg(:r, PR120, nothing, nothing) ≈ sqrt(2)

    # A higher confidence level buys a larger ball.
    @test ConcentrationRadius(; confidence = 0.99, scale = 0.5)(:r, PR60, nothing,
                                                                nothing) >
          alg(:r, PR60, nothing, nothing)

    # `scale = nothing` reads the average asset volatility off the prior result, so the
    # radius carries the units of the returns without the caller naming a number.
    auto = ConcentrationRadius()
    scale = mean(sqrt, diag(PR60.sigma))
    @test auto(:r, PR60, nothing, nothing) ≈ scale * sqrt(q / 60)
    @test auto(:r, PR60, nothing, nothing) != auto(:r, PR120, nothing, nothing)

    # The key never selects the value, so one rule in two slots resolves to one number.
    @test alg(:r, PR60, nothing, nothing) == alg(:r_a, PR60, nothing, nothing)
    @test alg(:r, PR60, nothing, nothing) == alg(:val, PR60, nothing, nothing)

    # Stated weights move `T` to Kish's effective sample size, which is smaller than the
    # row count for any weights that are not equal, so the ball widens.
    kish = sum(WTS)^2 / sum(abs2, WTS)
    @test kish < 60
    @test alg(:r, PR60, WTS, nothing) ≈ 0.5 * sqrt(q / kish)
    @test alg(:r, PR60, WTS, nothing) > alg(:r, PR60, nothing, nothing)

    # The rule needs no solver, so it ignores the one the resolution threads.
    @test alg(:r, PR60, nothing, Solver(; solver = nothing)) ==
          alg(:r, PR60, nothing, nothing)

    # Construction validation. The confidence level is a probability; the scale is a
    # positive, finite number when it is stated at all.
    @test_throws DomainError ConcentrationRadius(; confidence = 1.0)
    @test_throws DomainError ConcentrationRadius(; confidence = 0.0)
    @test_throws DomainError ConcentrationRadius(; confidence = 0.95, scale = -1.0)
    @test_throws DomainError ConcentrationRadius(; confidence = 0.95, scale = Inf)
    @test ConcentrationRadius(; scale = 0.1).scale == 0.1
    @test isnothing(ConcentrationRadius().scale)
    @test ConcentrationRadius().confidence == 0.95
end

@testset "RateRadius: the square-root rate" begin
    alg = RateRadius(; c = 0.2)
    @test alg(:r, PR60, nothing, nothing) ≈ 0.2 / sqrt(60)
    @test alg(:r, PR120, nothing, nothing) ≈ 0.2 / sqrt(120)
    @test RateRadius().c == 1
    @test RateRadius()(:r, PR60, nothing, nothing) ≈ inv(sqrt(60))

    # The rule reads the raw row count, so stated weights change nothing. That is the one
    # place it parts company with `ConcentrationRadius`.
    @test alg(:r, PR60, WTS, nothing) == alg(:r, PR60, nothing, nothing)

    # The key and the solver never select the value.
    @test alg(:r_b, PR60, nothing, Solver(; solver = nothing)) ==
          alg(:r, PR60, nothing, nothing)

    @test_throws DomainError RateRadius(; c = 0.0)
    @test_throws DomainError RateRadius(; c = -1.0)
    @test_throws DomainError RateRadius(; c = Inf)
end

@testset "DimensionalRateRadius: the dimensional rate" begin
    # The exponent is floored at one half, so a two-asset universe is the corner this rule
    # shares with `RateRadius`, and the radius there is a hand-computed square-root rate.
    alg = DimensionalRateRadius(; confidence = 0.95, scale = 0.5)
    lq = log(1 / 0.05)
    @test size(PR60_2.X, 2) == 2
    @test alg(:r, PR60_2, nothing, nothing) ≈ 0.5 * sqrt(lq / 60)

    # A wide universe flattens the rate, and that difference is the whole content of the
    # rule. Over a doubling of the record `RateRadius` shrinks by `sqrt(2)` and this rule
    # shrinks by `2^(1/20)`.
    @test size(PR125_20.X, 2) == 20
    @test size(PR250_20.X, 2) == 20
    dim_ratio = alg(:r, PR125_20, nothing, nothing) / alg(:r, PR250_20, nothing, nothing)
    rate = RateRadius(; c = 0.5)
    rate_ratio = rate(:r, PR125_20, nothing, nothing) / rate(:r, PR250_20, nothing, nothing)
    @test dim_ratio ≈ 2^(1 / 20)
    @test rate_ratio ≈ sqrt(2)
    @test dim_ratio < rate_ratio

    # `T^(-1/20)` at `T = 250` is `0.76`, the number the docstring names.
    @test isapprox(250^(-1 / 20), 0.76; atol = 5e-3)

    # A higher confidence level buys a larger ball, through `log(1 / (1 - confidence))`.
    @test DimensionalRateRadius(; confidence = 0.99, scale = 0.5)(:r, PR60_2, nothing,
                                                                  nothing) >
          alg(:r, PR60_2, nothing, nothing)

    # `scale = nothing` reads the average asset volatility off the prior result, and a
    # stated scale overrides it. That is the pair `ConcentrationRadius` is tested on.
    auto = DimensionalRateRadius()
    scale = mean(sqrt, diag(PR60_2.sigma))
    @test auto(:r, PR60_2, nothing, nothing) ≈ scale * sqrt(lq / 60)
    @test auto(:r, PR60_2, nothing, nothing) != alg(:r, PR60_2, nothing, nothing)
    @test isnothing(auto.scale)
    @test DimensionalRateRadius(; scale = 0.1).scale == 0.1
    @test DimensionalRateRadius().confidence == 0.95

    # The key never selects the value, so one rule in two slots resolves to one number.
    @test alg(:r, PR60_2, nothing, nothing) == alg(:r_a, PR60_2, nothing, nothing)
    @test alg(:r, PR60_2, nothing, nothing) == alg(:val, PR60_2, nothing, nothing)

    # Stated weights move `T` to Kish's effective sample size, on the same terms as
    # `ConcentrationRadius` and unlike `RateRadius`. This rule is a concentration
    # statement, so it prices the record that Kish's count measures.
    kish = sum(WTS)^2 / sum(abs2, WTS)
    @test kish < 60
    @test alg(:r, PR60_2, WTS, nothing) ≈ 0.5 * sqrt(lq / kish)
    @test alg(:r, PR60_2, WTS, nothing) > alg(:r, PR60_2, nothing, nothing)

    # The rule needs no solver, so it ignores the one the resolution threads.
    @test alg(:r, PR60_2, nothing, Solver(; solver = nothing)) ==
          alg(:r, PR60_2, nothing, nothing)

    # A four-asset universe already leaves the square-root corner.
    @test alg(:r, PR60, nothing, nothing) ≈ 0.5 * (lq / 60)^(1 / 4)
    @test alg(:r, PR60, nothing, nothing) != alg(:r, PR60_2, nothing, nothing)

    # Construction validation, on the terms `ConcentrationRadius` writes.
    @test_throws DomainError DimensionalRateRadius(; confidence = 1.0)
    @test_throws DomainError DimensionalRateRadius(; confidence = 0.0)
    @test_throws DomainError DimensionalRateRadius(; confidence = 0.95, scale = -1.0)
    @test_throws DomainError DimensionalRateRadius(; confidence = 0.95, scale = Inf)

    # The rule joins the radius family and its bounds, and no other.
    @test DimensionalRateRadius <: PO.AbstractAmbiguityRadiusCalibrationAlgorithm
    @test isa(alg, PO.Func_AmbRadCal)
    @test !isa(alg, PO.Func_AmbTwtCal)
    @test isa(AmbiguityRadiusCalibration(; alg = alg), PO.Num_AmbRadCal)
    @test_throws TypeError AmbiguityTailWeightCalibration(; alg = alg)
end

@testset "Ambiguity calibration: the resolver runs a rule by calling it" begin
    rrole = AmbiguityRadiusCalibration(; alg = RateRadius(; c = 0.2))
    trole = AmbiguityTailWeightCalibration(; alg = probe_tail_weight)

    # The role is unwrapped and the rule is called, so the role never reaches the rule.
    @test PO.resolve_calibration_slot(rrole, :r, PR60, nothing) ≈ 0.2 / sqrt(60)

    # A stated number passes through unchanged, and so does `nothing`.
    @test PO.resolve_calibration_slot(0.02, :r, PR60, nothing) == 0.02
    @test isnothing(PO.resolve_calibration_slot(nothing, :l1, PR60, nothing))

    # The four arguments arrive in the stated order, and the last two are the effective
    # observation weights and the effective solver.
    TWT_SEEN[] = nothing
    slv = Solver(; name = :probe, solver = nothing)
    @test PO.resolve_calibration_slot(trole, :l, PR60, WTS, slv) ≈ 2 / sqrt(60)
    @test TWT_SEEN[] == (; key = :l, weighted = true, solved = true)

    # A plain function in the `alg` field is a rule on the same terms.
    @test PO.resolve_calibration_slot(AmbiguityRadiusCalibration(; alg = probe_radius), :r,
                                      PR60, nothing) ≈ 3 / 60
end

@testset "Ambiguity calibration: the six distributionally robust slots" begin
    rrole = AmbiguityRadiusCalibration(; alg = RateRadius(; c = 0.2))
    trole = AmbiguityTailWeightCalibration(; alg = probe_tail_weight)
    r_num = 0.2 / sqrt(60)
    l_num = 2 / sqrt(60)

    # --- the scalar value-at-risk measure ----------------------------------------------
    m = DistributionallyRobustConditionalValueatRisk(; l = trole, r = rrole)
    # `alpha` joined the declaration when #583 widened the significance slots, and it
    # stands first because the declaration follows field order.
    @test PO.calibration_slots(m) == (; alpha = m.alpha, l = trole, r = rrole)
    out = PO.resolve_deferred_quantities(m, PR60)
    @test out.r ≈ r_num
    @test out.l ≈ l_num
    @test out.alpha == m.alpha

    # A measure whose slots both hold numbers is returned unchanged, so the common case
    # allocates nothing.
    plain = DistributionallyRobustConditionalValueatRisk()
    @test PO.resolve_deferred_quantities(plain, PR60) === plain

    # The generated `factory` reaches the same resolution, which is the channel the
    # optimisation itself uses.
    @test PO.factory(m, PR60).r ≈ r_num

    # The rebuild goes through the keyword constructor, so the positivity check re-runs on
    # the calibrated number: a rule that returns a value the slot does not admit is
    # refused at fold time, by the check a caller's own number meets.
    neg = AmbiguityRadiusCalibration(; alg = (args...) -> -1.0)
    @test_throws DomainError PO.resolve_deferred_quantities(DistributionallyRobustConditionalValueatRisk(;
                                                                                                         r = neg),
                                                            PR60)

    # --- the range measure --------------------------------------------------------------
    rg = DistributionallyRobustConditionalValueatRiskRange(; l_a = trole, r_a = rrole,
                                                           l_b = trole, r_b = rrole)
    @test PO.calibration_slots(rg) ==
          (; alpha = rg.alpha, l_a = trole, r_a = rrole, beta = rg.beta, l_b = trole,
           r_b = rrole)
    rgo = PO.resolve_deferred_quantities(rg, PR60)
    @test rgo.r_a ≈ r_num
    @test rgo.r_b ≈ r_num
    @test rgo.l_a ≈ l_num
    @test rgo.l_b ≈ l_num
    @test rgo.alpha == rg.alpha
    @test rgo.beta == rg.beta
    plain_rg = DistributionallyRobustConditionalValueatRiskRange()
    @test PO.resolve_deferred_quantities(plain_rg, PR60) === plain_rg

    # Each tail keeps its own pair, so two different rules give two different numbers.
    mixed = DistributionallyRobustConditionalValueatRiskRange(; r_a = rrole,
                                                              r_b = AmbiguityRadiusCalibration(;
                                                                                               alg = RateRadius(;
                                                                                                                c = 0.9)))
    mout = PO.resolve_deferred_quantities(mixed, PR60)
    @test mout.r_a ≈ 0.2 / sqrt(60)
    @test mout.r_b ≈ 0.9 / sqrt(60)

    # --- the drawdown measure -----------------------------------------------------------
    dd = DistributionallyRobustConditionalDrawdownatRisk(; l = trole, r = rrole)
    @test PO.calibration_slots(dd) == (; alpha = dd.alpha, l = trole, r = rrole)
    ddo = PO.resolve_deferred_quantities(dd, PR60)
    @test ddo.r ≈ r_num
    @test ddo.l ≈ l_num
    plain_dd = DistributionallyRobustConditionalDrawdownatRisk()
    @test PO.resolve_deferred_quantities(plain_dd, PR60) === plain_dd

    # --- the role split holds at the slot ----------------------------------------------
    # A radius role in a tail-weight slot, and the reverse, are both refused at
    # construction. The bound is the whole of the validation.
    @test_throws TypeError DistributionallyRobustConditionalValueatRisk(; l = rrole)
    @test_throws TypeError DistributionallyRobustConditionalValueatRisk(; r = trole)
    @test_throws TypeError DistributionallyRobustConditionalValueatRiskRange(; l_a = rrole)
    @test_throws TypeError DistributionallyRobustConditionalValueatRiskRange(; r_b = trole)
    @test_throws TypeError DistributionallyRobustConditionalDrawdownatRisk(; l = rrole)

    # A significance role is refused too: it belongs to neither family.
    @test_throws TypeError DistributionallyRobustConditionalValueatRisk(;
                                                                        r = SignificanceTailCalibration(;
                                                                                                        alg = ScenarioCount(;
                                                                                                                            n = 3)))

    # --- the value-level entry point refuses a rule ------------------------------------
    # `expected_risk` given a bare returns matrix has no prior result to resolve against,
    # so it names the slot and the way out rather than failing several frames down.
    w = fill(0.25, 4)
    err = try
        expected_risk(m, w, X60)
        nothing
    catch e
        e
    end
    @test isa(err, ArgumentError)
    @test occursin("DistributionallyRobustConditionalValueatRisk.l", err.msg)
    @test occursin("AmbiguityTailWeightCalibration", err.msg)
    @test occursin("factory", err.msg)

    # Given the prior result the measure is resolved first, so the same call succeeds.
    @test isfinite(expected_risk(m, w, PR60))

    # A stated number still reaches the functor untouched, so nothing existing changed.
    @test expected_risk(plain, w, X60) ≈ expected_risk(plain, w, PR60)
end

@testset "Ambiguity calibration: the two regularisation coefficients" begin
    rrole = AmbiguityRadiusCalibration(; alg = RateRadius(; c = 0.2))
    r_num = 0.2 / sqrt(60)

    # --- L2Regularisation ---------------------------------------------------------------
    l2 = L2Regularisation(; val = rrole)
    @test PO.calibration_slots(l2) == (; val = rrole)
    @test PO.factory(l2, PR60).val ≈ r_num
    @test isa(PO.factory(l2, PR60).alg, SOCRiskExpr)

    # A term holding a number is returned unchanged, and a vector of terms is resolved
    # element by element by the generic vector method.
    plain_l2 = L2Regularisation(; val = 0.3)
    @test PO.factory(plain_l2, PR60) === plain_l2
    @test [x.val for x in PO.factory([l2, plain_l2], PR60)] ≈ [r_num, 0.3]

    # The guard: a radius is the coefficient of the un-squared norm, so a rule beside a
    # formulation that penalises the square has no reading and is refused at construction.
    for alg in (SquaredSOCRiskExpr(), QuadRiskExpr(), RSOCRiskExpr())
        @test_throws ArgumentError L2Regularisation(; val = rrole, alg = alg)
    end

    # A plain number stays legal with every formulation, so nothing existing breaks.
    for alg in (SOCRiskExpr(), SquaredSOCRiskExpr(), QuadRiskExpr(), RSOCRiskExpr())
        @test L2Regularisation(; val = 0.3, alg = alg).val == 0.3
    end

    # The guard is a set of methods, so a formulation refuses a radius by adding one. The
    # permissive fallback is what every other pair reaches.
    @test isnothing(PO.assert_ambiguity_radius_formulation(0.3, SquaredSOCRiskExpr()))
    @test isnothing(PO.assert_ambiguity_radius_formulation(rrole, SOCRiskExpr()))
    @test occursin("norm(w, 2)^2", PO.squared_norm_radius_msg(QuadRiskExpr()))
    @test occursin("QuadRiskExpr", PO.squared_norm_radius_msg(QuadRiskExpr()))

    # The rebuild re-runs the positivity check on the calibrated number.
    neg = L2Regularisation(; val = AmbiguityRadiusCalibration(; alg = (args...) -> -1.0))
    @test_throws DomainError PO.factory(neg, PR60)

    # --- LpRegularisation ---------------------------------------------------------------
    lp = LpRegularisation(; p = 3, val = rrole)
    @test PO.calibration_slots(lp) == (; val = rrole)
    @test PO.factory(lp, PR60).val ≈ r_num
    @test PO.factory(lp, PR60).p == 3
    plain_lp = LpRegularisation(; val = 0.4)
    @test PO.factory(plain_lp, PR60) === plain_lp

    # No formulation slot, so no pairing can be wrong and no guard runs.
    @test LpRegularisation(; p = 2.5, val = rrole).val === rrole

    # A tail-weight role is refused in a radius slot, on both estimators.
    trole = AmbiguityTailWeightCalibration(; alg = probe_tail_weight)
    @test_throws TypeError L2Regularisation(; val = trole)
    @test_throws TypeError LpRegularisation(; val = trole)
end

@testset "Ambiguity calibration: the optimiser's l1 and linf reach the model" begin
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = "verbose" => false)
    rd = ReturnsResult(; nx = string.(1:4), X = X60)
    rrole = AmbiguityRadiusCalibration(; alg = RateRadius(; c = 0.2))
    r_num = 0.2 / sqrt(60)

    # The four coefficients all take a rule, and the bound admits it.
    opt = JuMPOptimiser(; slv = slv, pe = PR60, l1 = rrole, linf = rrole,
                        l2 = L2Regularisation(; val = rrole),
                        lp = LpRegularisation(; val = rrole))
    @test isa(opt.l1, AmbiguityRadiusCalibration)
    @test isa(opt.linf, AmbiguityRadiusCalibration)

    # Neither the weights factory nor the cluster slice holds a prior result, so both carry
    # a rule through untouched. That is right: the rule resolves against the cluster's own
    # prior when the model is assembled.
    @test PO.factory(opt, fill(0.25, 4)).l1 === rrole
    @test PO.port_opt_view(opt, 1:3, X60).linf === rrole

    # Assembly resolves all four against the optimisation's own prior result, so the
    # coefficient the model carries is the calibrated number and not the rule.
    mr = MeanRisk(; r = Variance(), opt = opt)
    attrs = PO.processed_jump_optimiser_attributes(mr.opt, rd)
    model = JuMP.Model()
    PO.set_model_scales!(model, mr.opt.sc, mr.opt.so)
    PO.set_maximum_ratio_factor_variables!(model, mr.obj)
    PO.set_w!(model, attrs.pr.X, mr.wi)
    PO.set_weight_constraints!(model, attrs.wb, mr.opt)
    PO.assemble_jump_model!(model, mr, mr.opt, attrs, rd, mr.r, mr.obj)
    @test JuMP.coefficient(model[:l1], model[:t_l1]) ≈ r_num
    @test JuMP.coefficient(model[:linf], model[:t_linf]) ≈ r_num

    # A stated number reaches the same coefficient, so the widening changed nothing for a
    # caller who names one.
    optn = JuMPOptimiser(; slv = slv, pe = PR60, l1 = r_num)
    mrn = MeanRisk(; r = Variance(), opt = optn)
    attrsn = PO.processed_jump_optimiser_attributes(mrn.opt, rd)
    modeln = JuMP.Model()
    PO.set_model_scales!(modeln, mrn.opt.sc, mrn.opt.so)
    PO.set_maximum_ratio_factor_variables!(modeln, mrn.obj)
    PO.set_w!(modeln, attrsn.pr.X, mrn.wi)
    PO.set_weight_constraints!(modeln, attrsn.wb, mrn.opt)
    PO.assemble_jump_model!(modeln, mrn, mrn.opt, attrsn, rd, mrn.r, mrn.obj)
    @test JuMP.coefficient(modeln[:l1], modeln[:t_l1]) ≈ r_num

    # A tail weight is not a radius, so the tail-weight role is refused in all four slots.
    trole = AmbiguityTailWeightCalibration(; alg = probe_tail_weight)
    @test_throws TypeError JuMPOptimiser(; slv = slv, l1 = trole)
    @test_throws TypeError JuMPOptimiser(; slv = slv, linf = trole)
end

#=
The two deferral channels can now meet in one field: `l1` and `linf` are bounded
`TD_Option{<:Num_AmbRadCal}`, so a `TimeDependent` may wrap a Calibration Rule. ADR 0030
never considered a second channel, and the map that owns this work keeps the ORDER in its
*Not yet specified* section.

This testset records what the code does today rather than ratifying a design. The two verbs
run at two different points of the pipeline and neither knows about the other: the period
selection runs in `update_time_dependent_fields`, before any prior is fitted, and the
calibration resolution runs at assembly, against the prior of the period that was selected.
So the order falls out of the pipeline and nothing had to invent it.
=#
@testset "Ambiguity calibration: the time-dependent wrapper selects, then the rule runs" begin
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = "verbose" => false)
    rrole = AmbiguityRadiusCalibration(; alg = RateRadius(; c = 0.2))
    rd = ReturnsResult(; nx = string.(1:4), X = X120)

    # The widened bound admits a schedule over a rule and a number.
    td = TimeDependent([rrole, 0.5]; default = 0.5)
    @test isa(td, PO.TD_Option{<:PO.Num_AmbRadCal})
    opt = JuMPOptimiser(; slv = slv, l1 = td)
    @test PO.time_dependent_fields(opt) == (:l1,)

    # The selection carries the schedule's occupant out unchanged. It does not resolve it,
    # and it has no prior result with which it could.
    ctx1 = TimeDependentContext(; i = 1, n = 2, rd = rd, train_idx = [1:60, 1:90],
                                test_idx = [61:90, 91:120])
    ctx2 = TimeDependentContext(; i = 2, n = 2, rd = rd, train_idx = [1:60, 1:90],
                                test_idx = [61:90, 91:120])
    @test PO.time_dependent_value(td, ctx1) === rrole
    @test PO.time_dependent_value(td, ctx2) == 0.5

    # The selected rule then resolves against whichever prior the fold produced, so a
    # schedule and a rule compose rather than fight.
    sel1 = PO.time_dependent_value(td, ctx1)
    @test PO.resolve_calibration_slot(sel1, :l1, PR60, nothing) ≈ 0.2 / sqrt(60)
    @test PO.resolve_calibration_slot(sel1, :l1, PR120, nothing) ≈ 0.2 / sqrt(120)

    # Outside every fold loop the schedule falls back to its own default, which is a
    # number here and reaches the model as one.
    @test PO.reset_time_dependent_fields(opt).l1 == 0.5
end
