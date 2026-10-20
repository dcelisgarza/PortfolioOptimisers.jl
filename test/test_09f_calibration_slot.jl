#=
A risk-measure slot may hold a **Calibration Rule** — the rule that computes the tail
probability or the deformation parameter — instead of the number itself. The rule resolves
against the optimisation's own prior, so the struct that reaches a kernel always holds a
plain number.

The mechanism is parallel to the Deferred Quantity one, not shared with it, and
`test_09e_deferred_quantity.jl` covers that other half. A Deferred Quantity is fitted and a
quantity is read off the fit; a rule fits nothing and reads the sample size, the moments and
the effective observation weights. So the rule types stay out of the `DeferredQuantity`
union, and the resolver, the declaration and the refusal each take their own verb.

A rule is run by CALLING it. A callable struct and a plain function are therefore the same
thing to the resolver, and both bounds admit both.

Issue #581 builds the mechanism. The rules themselves are #582, and the slots that hold them
are #583 and #584, so every rule below is a probe defined here.
=#
using Clarabel, JuMP

const PO = PortfolioOptimisers

# A significance rule that reports what it was handed, so the resolver's argument order is
# asserted rather than assumed. `n / T` is the shape `ScenarioCount` will ship under #582.
struct ProbeScenarioCount{T} <: PO.AbstractSignificanceCalibrationAlgorithm
    n::T
end
function (alg::ProbeScenarioCount)(key::Symbol, pr::PO.AbstractPriorResult, w, slv,
                                   ctx::PO.CalibrationContext)
    return (; key = key, alpha = alg.n / size(pr.X, 1), weighted = !isnothing(w),
            solved = !isnothing(slv), series = ctx.series)
end

# A deformation rule, so that the family bound is checked against a real second family.
struct ProbeEntropyBudget{T} <: PO.AbstractDeformationCalibrationAlgorithm
    target::T
end
function (alg::ProbeEntropyBudget)(::Symbol, ::PO.AbstractPriorResult, ::Any, ::Any,
                                   ::PO.CalibrationContext)
    return alg.target
end

# The same rule with no type at all. A closure over the caller's own data is the case that
# cannot be given one, and it is why both `alg` bounds admit a bare `Function`.
function probe_rate(::Symbol, pr::PO.AbstractPriorResult, ::Any, ::Any,
                    ::PO.CalibrationContext)
    return inv(sqrt(size(pr.X, 1)))
end

# Stands in for a measure whose tail slot has been widened. #583 widens the real ones.
struct CalibratedProbe{T1, T2} <: PO.AbstractAlgorithm
    alpha::T1
    child::T2
    function CalibratedProbe(alpha::PO.Num_SigCal, child)
        return new{typeof(alpha), typeof(child)}(alpha, child)
    end
end
function CalibratedProbe(; alpha::PO.Num_SigCal = 0.05, child = nothing)
    return CalibratedProbe(alpha, child)
end
function PortfolioOptimisers.calibration_slots(x::CalibratedProbe)
    return (; alpha = x.alpha, child = x.child)
end

# Stands in for a head slot. Its bound is the whole of the rule validation.
struct HeadProbe{T} <: PO.AbstractAlgorithm
    beta::T
    function HeadProbe(beta::PO.Num_SigCal)
        return new{typeof(beta)}(beta)
    end
end

# Stands in for a deformation slot.
struct DeformationProbe{T} <: PO.AbstractAlgorithm
    kappa::T
    function DeformationProbe(kappa::PO.Num_DefCal)
        return new{typeof(kappa)}(kappa)
    end
end

# A rule that reports the context it was handed, as a number the slot bounds admit. The four
# states of the pair map to 0.1, 0.3, 0.5 and 0.7, so the derived reading of the effective
# observation weights and the effective solver is asserted rather than assumed.
struct ProbeContext <: PO.AbstractSignificanceCalibrationAlgorithm end
function (::ProbeContext)(::Symbol, ::PO.AbstractPriorResult, w, slv,
                          ::PO.CalibrationContext)
    return (1 + 2 * !isnothing(w) + 4 * !isnothing(slv)) / 10
end

# Carries both context fields, so the derivation settles each against the caller's.
struct ContextProbe{T1, T2, T3} <: PO.AbstractAlgorithm
    alpha::T1
    w::T2
    slv::T3
end
function ContextProbe(; alpha::PO.Num_SigCal = 0.05, w = nothing, slv = nothing)
    return ContextProbe(alpha, w, slv)
end
PortfolioOptimisers.calibration_slots(x::ContextProbe) = (; alpha = x.alpha)

# Carries neither, so the derivation hands the rule `pr.w` and the caller's own solver.
struct BareProbe{T1} <: PO.AbstractAlgorithm
    alpha::T1
end
BareProbe(; alpha::PO.Num_SigCal = 0.05) = BareProbe(alpha)
PortfolioOptimisers.calibration_slots(x::BareProbe) = (; alpha = x.alpha)

const RULE = ProbeScenarioCount(25)
const KRULE = ProbeEntropyBudget(0.3)
const CTX = ProbeContext()

@testset "Calibration slot: the taxonomy" begin
    # ONE root. A rule is an Algorithm, and the slot that holds it names the quantity, so
    # nothing stands between the caller and the rule. #641 withdrew the seven rules that
    # used to, and the Estimator root they stood under with them.
    @test PO.AbstractCalibrationAlgorithm <: PO.AbstractAlgorithm
    @test !(PO.AbstractCalibrationAlgorithm <: PO.AbstractEstimator)
    @test PO.AbstractSignificanceCalibrationAlgorithm <: PO.AbstractCalibrationAlgorithm
    @test PO.AbstractDeformationCalibrationAlgorithm <: PO.AbstractCalibrationAlgorithm

    # The two families are disjoint, which is what makes a family bound a real check.
    @test !(PO.AbstractSignificanceCalibrationAlgorithm <:
            PO.AbstractDeformationCalibrationAlgorithm)
    @test !(PO.AbstractDeformationCalibrationAlgorithm <:
            PO.AbstractSignificanceCalibrationAlgorithm)

    # A rule stays OUT of the Deferred Quantity union. The two mechanisms are parallel, and
    # a rule reaching `resolve_slot` would be fitted, which it cannot be.
    @test !isa(RULE, PO.DeferredQuantity)
    @test !isa(KRULE, PO.DeferredQuantity)
    @test !isa(probe_rate, PO.DeferredQuantity)

    # The abstract types stay unexported; the rules themselves are caller-facing. `names`
    # lists a `public` binding too, so the export check reads `Base.isexported`.
    @test !Base.isexported(PortfolioOptimisers, :AbstractCalibrationAlgorithm)
    @test !Base.isexported(PortfolioOptimisers, :AbstractSignificanceCalibrationAlgorithm)
    @test !Base.isexported(PortfolioOptimisers, :AbstractDeformationCalibrationAlgorithm)
    @test Base.isexported(PortfolioOptimisers, :ScenarioCount)
    @test Base.isexported(PortfolioOptimisers, :EntropyBudget)

    # The seven rules, their root, and the verb that put them on are all gone.
    for name in (:AbstractCalibrationEstimator, :SignificanceTailCalibration,
                 :SignificanceHeadCalibration, :DeformationTailCalibration,
                 :DeformationHeadCalibration, :AmbiguityRadiusCalibration,
                 :AmbiguityTailWeightCalibration, :NormCeilingCalibration, :bind_role, :Func_SigCal,
                 :Func_DefCal, :Func_AmbRadCal, :Func_AmbTwtCal, :Func_NormCeilCal)
        @test !isdefined(PortfolioOptimisers, name)
    end
end

@testset "Calibration slot: the field bounds" begin
    # Every bound admits the number, which is what keeps a stated value legal everywhere.
    @test isa(0.05, PO.Num_SigCal)
    @test isa(0.05, PO.Num_DefCal)

    # Each bound names one rule family and no other, so a rule of the wrong family is
    # refused. The slot names the end of the distribution, so ONE bound serves the tail
    # slot and the head slot alike, and the two names the rules needed are gone.
    @test isa(RULE, PO.Num_SigCal) && !isa(RULE, PO.Num_DefCal)
    @test isa(KRULE, PO.Num_DefCal) && !isa(KRULE, PO.Num_SigCal)

    # A plain FUNCTION is admitted by both, because a function carries no family and the
    # name of the slot is what states the quantity there.
    @test isa(probe_rate, PO.Num_SigCal) && isa(probe_rate, PO.Num_DefCal)

    # The refusal lands at construction, in the constructor's signature, and not at fold
    # time. This is the claim that lets the mechanism carry no guard at all. The three
    # probes are plain structs, so the positional route is bounded here too.
    @test HeadProbe(RULE).beta === RULE
    @test HeadProbe(0.05).beta == 0.05
    @test_throws MethodError HeadProbe(KRULE)
    @test DeformationProbe(KRULE).kappa === KRULE
    @test_throws MethodError DeformationProbe(RULE)
    @test_throws TypeError CalibratedProbe(; alpha = KRULE)
    @test_throws MethodError CalibratedProbe(KRULE, nothing)
end

@testset "Calibration slot: the slot stores the rule the caller wrote" begin
    # The slot names the quantity and the end of the distribution, so the caller states the
    # rule alone and the slot stores it. Nothing wraps it on the way in.
    bare = ConditionalValueatRisk(; alpha = RULE)
    @test bare.alpha === RULE

    # A stated number crosses unchanged, and so does a plain function.
    @test ConditionalValueatRisk(; alpha = 0.05).alpha === 0.05
    @test ConditionalValueatRisk(; alpha = probe_rate).alpha === probe_rate
    @test ConditionalValueatRisk(; alpha = CTX).alpha === CTX

    # One rule reaches both ends of a Range measure, and each slot stores it as it stands.
    both = ConditionalValueatRiskRange(; alpha = RULE, beta = RULE)
    @test both.alpha === RULE && both.beta === RULE

    # A head slot that defaults from its tail slot reads the same rule.
    defaulted = ConditionalValueatRiskRange(; alpha = RULE)
    @test defaulted.beta === RULE

    # A measure with two families takes one rule from each.
    rvar = RelativisticValueatRisk(; alpha = RULE, kappa = KRULE)
    @test rvar.alpha === RULE && rvar.kappa === KRULE

    # A slot that names no end of the distribution stores its rule the same way.
    drcvar = DistributionallyRobustConditionalValueatRisk(; alpha = RULE,
                                                          l = TailTermParity(),
                                                          r = ConcentrationRadius())
    @test isa(drcvar.l, TailTermParity)
    @test isa(drcvar.r, ConcentrationRadius)

    # The one dual-use slot admits both families, and the route refuses the one that has no
    # reading on it.
    @test isa(LpRegularisation(; val = ConcentrationRadius()).val, ConcentrationRadius)
    @test isa(LpRegularisation(; val = EffectiveAssetFloor()).val, EffectiveAssetFloor)
    @test isa(L2Regularisation(; val = ConcentrationRadius()).val, ConcentrationRadius)

    # An optimiser norm slot stores its rule on the same terms, under the time-dependent
    # wrapper.
    oslv = Solver(; name = :probe, solver = nothing)
    opt = JuMPOptimiser(; slv = oslv, l1 = ConcentrationRadius(),
                        l2c = EffectiveAssetFloor())
    @test isa(opt.l1, ConcentrationRadius)
    @test isa(opt.l2c, EffectiveAssetFloor)
    @test isnothing(JuMPOptimiser(; slv = oslv).l1)

    # The wrong family is refused by the keyword constructor's bound.
    @test_throws TypeError ConditionalValueatRisk(; alpha = KRULE)
    @test_throws TypeError RelativisticValueatRisk(; kappa = RULE)
    @test_throws TypeError LpRegularisation(; val = RULE)

    # The dual-use slot is the one slot that admits no plain function, because a function
    # names no family and the two guards of that slot read the family.
    @test_throws TypeError LpRegularisation(; val = probe_rate)
    @test isa(ConditionalValueatRisk(; alpha = probe_rate).alpha, Function)

    # The stored rule is what every reader of the slot sees.
    @test_throws ArgumentError PO.assert_calibrated_slots(bare)
    @test PO.calibration_slots(bare).alpha === bare.alpha
end

@testset "Calibration slot: the resolver is parallel to `resolve_slot`" begin
    rng = StableRNG(987654321)
    X = randn(rng, 60, 4)
    pr = prior(EmpiricalPrior(), X)

    # A stated number is returned unchanged, whatever the key and the weights are. That is
    # the whole of the second state, and it is why nothing existing moves.
    @test PO.resolve_calibration_slot(0.05, :alpha, pr, nothing) == 0.05
    @test PO.resolve_calibration_slot(0.05, :alpha, pr, pweights(fill(1 / 60, 60))) == 0.05
    @test isnothing(PO.resolve_calibration_slot(nothing, :kappa, pr, nothing))

    # A rule type is unwrapped and its rule is CALLED. The rule sees the key, the prior and
    # the weights, in that order, and it never sees the rule it was placed in.
    res = PO.resolve_calibration_slot(RULE, :alpha, pr, nothing)
    @test res.key === :alpha
    @test res.alpha == 25 / 60
    @test !res.weighted

    # The head rule carries the same rule, so it resolves to the same number.
    res = PO.resolve_calibration_slot(RULE, :beta, pr, pweights(fill(1 / 60, 60)))
    @test res.key === :beta
    @test res.alpha == 25 / 60
    @test res.weighted

    # The deformation family resolves through the same verb.
    @test PO.resolve_calibration_slot(KRULE, :kappa, pr, nothing) == 0.3
    @test PO.resolve_calibration_slot(KRULE, :kappa_b, pr, nothing) == 0.3

    # A plain function in `alg` resolves by the same call, with no adapter in between.
    @test PO.resolve_calibration_slot(probe_rate, :alpha, pr, nothing) == inv(sqrt(60))
    @test PO.resolve_calibration_slot(probe_rate, :kappa_b, pr, nothing) == inv(sqrt(60))

    # The solver reaches the rule, so a rule may call `ERM` or `RRM`. It is the fifth
    # argument and it defaults to `nothing`, so a caller that carries none states nothing.
    slv = Solver(; name = :probe, solver = nothing)
    res = PO.resolve_calibration_slot(RULE, :alpha, pr, nothing, slv)
    @test res.solved
    @test !PO.resolve_calibration_slot(RULE, :alpha, pr, nothing).solved
    @test PO.resolve_calibration_slot(0.05, :alpha, pr, nothing, slv) == 0.05

    # The rule refits when the sample moves, which is the entire reason for a rule over a
    # number: the same rule gives a different `alpha` on a shorter fold.
    fold = prior(EmpiricalPrior(), X[1:30, :])
    @test PO.resolve_calibration_slot(RULE, :alpha, fold, nothing).alpha == 25 / 30
end

@testset "Calibration slot: the declaration defaults to empty" begin
    # A type with no calibration slot needs no method, which is what keeps the widening
    # local to the 33 types that take it.
    @test PO.calibration_slots(0.05) == (;)
    @test PO.calibration_slots(nothing) == (;)
    # `Variance` is one of them: its `sigma` slot takes a Deferred Quantity, which is the
    # other mechanism, and it states no quantity a rule could compute.
    @test PO.calibration_slots(Variance()) == (;)
    @test PO.calibration_slots(MaximumDrawdown()) == (;)

    # A type that names its slots gets the values back under the field names.
    probe = CalibratedProbe(; alpha = RULE)
    slots = PO.calibration_slots(probe)
    @test keys(slots) === (:alpha, :child)
    @test slots.alpha === probe.alpha
    @test isnothing(slots.child)
end

@testset "Calibration slot: a value-level entry point refuses a rule" begin
    # Nothing to refuse: a stated number resolves nowhere and passes.
    @test isnothing(PO.assert_calibrated_slots(CalibratedProbe(; alpha = 0.05)))
    @test isnothing(PO.assert_calibrated_slots(0.05))

    # A rule that reached here has no prior to resolve against, so it is refused, and the
    # message names the slot, the rule standing in it and the way out.
    probe = CalibratedProbe(; alpha = RULE)
    @test_throws ArgumentError PO.assert_calibrated_slots(probe)
    msg = try
        PO.assert_calibrated_slots(probe)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("CalibratedProbe.alpha", msg)
    @test occursin("ProbeScenarioCount", msg)
    @test occursin("factory(r, pr)", msg)

    # The check recurses into a child, so a container is covered by its children's
    # declarations rather than by a forwarding method of its own.
    nested = CalibratedProbe(; alpha = 0.05, child = probe)
    @test_throws ArgumentError PO.assert_calibrated_slots(nested)

    # A slot that holds a vector of children is recursed element by element.
    @test isnothing(PO.assert_calibrated_slots([CalibratedProbe(; alpha = 0.05),
                                                CalibratedProbe(; alpha = 0.1)]))
    @test_throws ArgumentError PO.assert_calibrated_slots([CalibratedProbe(; alpha = 0.05),
                                                           probe])
end

@testset "Calibration slot: every container refuses a rule its child holds" begin
    # A container holds risk measures, and one of them may hold a rule. Two routes reach
    # that rule. A container whose own evaluation runs each child through `expected_risk`
    # is covered by the child's entry point. A container that calls its children directly
    # is covered by its own `calibration_slots` declaration, and by nothing else.
    #
    # `GenericValueatRiskRange` is the second kind: its functor calls `r.loss(x)` and
    # `r.gain(-x)`. It declared `deferred_slots` alone, so `assert_calibrated_slots` walked
    # an empty tuple and the rule reached the kernel, where `alpha * length(x)` raised a
    # `MethodError` naming neither the slot nor the rule. The declaration below is the fix.
    rng = StableRNG(987654321)
    X = randn(rng, 200, 4) / 100
    w = fill(0.25, 4)
    rule = RULE
    cvar = ConditionalValueatRisk(; alpha = rule)

    # The declaration names both children, so the walk reaches the rule in either of them.
    # The constructor strips each child's risk-expression contribution, so the slots are
    # the fields the container holds and not the measures the caller wrote.
    gvr = GenericValueatRiskRange(; loss = cvar, gain = ConditionalValueatRisk())
    @test PO.calibration_slots(gvr) == (; loss = gvr.loss, gain = gvr.gain)
    @test PO.calibration_slots(gvr) == PO.deferred_slots(gvr)
    @test isnothing(PO.assert_calibrated_slots(GenericValueatRiskRange()))
    @test_throws ArgumentError PO.assert_calibrated_slots(GenericValueatRiskRange(;
                                                                                  loss = cvar))
    @test_throws ArgumentError PO.assert_calibrated_slots(GenericValueatRiskRange(;
                                                                                  gain = cvar))

    # The message the caller reads names the child's slot, the rule standing in it and the
    # way out, on the same terms as a leaf measure's.
    function refusal(r)
        return try
            expected_risk(r, w, X)
            ""
        catch e
            sprint(showerror, e)
        end
    end
    owacvar = OrderedWeightsArrayConditionalValueatRisk(; alpha = rule)
    for (r, slot) in
        ((GenericValueatRiskRange(; loss = cvar), "ConditionalValueatRisk.alpha"),
         (RiskRatio(; r1 = cvar, r2 = Variance()), "ConditionalValueatRisk.alpha"),
         (NonOptimisationRiskRatio(; r1 = cvar, r2 = StandardDeviation()),
          "ConditionalValueatRisk.alpha"),
         (MeanReturnRiskRatio(; rk = cvar), "ConditionalValueatRisk.alpha"),
         (RiskTrackingRiskMeasure(; r = cvar, tr = WeightsTracking(; w = w)),
          "ConditionalValueatRisk.alpha"),
         (OrderedWeightsArray(; w = owacvar),
          "OrderedWeightsArrayConditionalValueatRisk.alpha"),
         (OrderedWeightsArrayRange(; w1 = owacvar,
                                   w2 = OrderedWeightsArrayConditionalValueatRisk()),
          "OrderedWeightsArrayConditionalValueatRisk.alpha"))
        msg = refusal(r)
        @test occursin("Calibration Rule", msg)
        @test occursin(slot, msg)
        @test occursin("factory(r, pr)", msg)
    end
end

@testset "Calibration slot: every Range head slot defaults from its tail slot" begin
    # One rule for a head slot, and every Range type states it the same way: the head slot
    # defaults to its tail counterpart. A stated number crosses unchanged,
    # so the default pair of each type is the pair it carried before the rule became one.
    sig = RULE
    def = KRULE
    ranges = (ValueatRiskRange(), ConditionalValueatRiskRange(),
              DistributionallyRobustConditionalValueatRiskRange(),
              EntropicValueatRiskRange(), RelativisticValueatRiskRange(),
              PowerNormValueatRiskRange(), OrderedWeightsArrayConditionalValueatRiskRange(),
              OrderedWeightsArrayTailGiniRange())
    for r in ranges
        @test r.alpha === 0.05
        @test r.beta === 0.05
    end

    # A stated number reaches the head slot, where six of the eight held a literal of their
    # own. The head slot of a Range measure is now the tail slot's occupant in all eight.
    @test ValueatRiskRange(; alpha = 0.01).beta === 0.01
    @test ConditionalValueatRiskRange(; alpha = 0.01).beta === 0.01
    @test DistributionallyRobustConditionalValueatRiskRange(; alpha = 0.01).beta === 0.01
    @test EntropicValueatRiskRange(; alpha = 0.01).beta === 0.01
    @test RelativisticValueatRiskRange(; alpha = 0.01).beta === 0.01
    @test PowerNormValueatRiskRange(; alpha = 0.01).beta === 0.01
    @test OrderedWeightsArrayConditionalValueatRiskRange(; alpha = 0.01).beta === 0.01
    @test OrderedWeightsArrayTailGiniRange(; alpha = 0.01).beta === 0.01

    # A stated tail RULE crosses unchanged to the head slot, in all eight.
    for r in (ValueatRiskRange(; alpha = sig), ConditionalValueatRiskRange(; alpha = sig),
              DistributionallyRobustConditionalValueatRiskRange(; alpha = sig),
              EntropicValueatRiskRange(; alpha = sig),
              RelativisticValueatRiskRange(; alpha = sig),
              PowerNormValueatRiskRange(; alpha = sig),
              OrderedWeightsArrayConditionalValueatRiskRange(; alpha = sig),
              OrderedWeightsArrayTailGiniRange(; alpha = sig))
        @test r.beta === RULE
    end

    # The deformation mirror has a caller in `src/`, and this is it. The gain-side pair of
    # the relativistic Range measure defaults to the loss-side pair, both halves of it.
    rl = RelativisticValueatRiskRange(; alpha = sig, kappa_a = def)
    @test rl.kappa_b === KRULE
    @test RelativisticValueatRiskRange(; kappa_a = 0.5).kappa_b === 0.5

    # The two norm orders take the same rule. They stay numbers, so they mirror by plain
    # assignment, the way `beta_i` mirrors `alpha_i`.
    @test PowerNormValueatRiskRange(; pa = 3.0).pb === 3.0
    @test OrderedWeightsArrayTailGiniRange(; alpha_i = 1e-3).beta_i === 1e-3

    # An ambiguity rule names no end of the distribution, so neither ambiguity slot of the
    # range defaults from the other, and both keep the numbers they declare.
    dr = DistributionallyRobustConditionalValueatRiskRange(; l_a = 2.0, r_a = 0.05)
    @test dr.l_b === 1.0
    @test dr.r_b === 0.02

    # A stated head slot wins over the mirror, and the two ends then stand apart.
    @test ValueatRiskRange(; alpha = 0.01, beta = 0.02).beta === 0.02
    @test RelativisticValueatRiskRange(; kappa_a = 0.5, kappa_b = 0.6).kappa_b === 0.6
end

@testset "Calibration slot: `sel` keeps a rule rather than filling from the prior" begin
    # A slot the caller filled with the rule that computes the value is a STATED slot, so
    # the prior must not fill it. The resolution that follows replaces it with the number
    # the rule produced. This arm is reachable because the `@propagatable` prior `factory`
    # selects BEFORE it resolves, so a widened slot that also carries `@pprop` arrives here
    # still holding a rule.
    @test PO.sel(RULE, 0.05) === RULE
    @test PO.sel(KRULE, 0.3) === KRULE
    @test PO.sel(probe_rate, 0.05) === probe_rate

    # A stated number still wins, and an unstated slot still falls back, both unchanged.
    @test PO.sel(0.01, 0.05) == 0.01
    @test PO.sel(nothing, 0.05) == 0.05
end

#=
The generated `factory` of a `@propagatable` type resolves Deferred Quantities LAST, after
every selection. #581 moved it there so that a Calibration Rule, which resolves on the same
pass, sees the solver the optimiser threaded and may call `ERM` or `RRM`. The probe below
records the state of the struct at the moment the resolution runs, so the ORDER is asserted
rather than assumed: under the previous order `slv` would still be the measure's own
`nothing`.
=#
const ORDER_SEEN = Ref{Any}(nothing)
PortfolioOptimisers.@propagatable struct OrderProbe{T1, T2, T3} <: PO.AbstractAlgorithm
    @cprop slv::T1
    @pprop w::T2
    alpha::T3
end
function OrderProbe(; slv = nothing, w = nothing, alpha = nothing)
    return OrderProbe(slv, w, alpha)
end
PortfolioOptimisers.deferred_slots(x::OrderProbe) = (; alpha = x.alpha)
function PortfolioOptimisers.resolve_deferred_quantities(x::OrderProbe,
                                                         pr::PO.AbstractPriorResult,
                                                         ::Any = nothing)
    ORDER_SEEN[] = (; slv = x.slv, w = x.w)
    return OrderProbe(; slv = x.slv, w = x.w, alpha = PO.resolve_slot(x.alpha, :mu, pr))
end

@testset "Calibration slot: the Deferred-Quantity resolution runs last" begin
    rng = StableRNG(123456789)
    X = randn(rng, 40, 3)
    pr = prior(EmpiricalPrior(), X)
    slv = Solver(; name = :probe, solver = nothing)

    ORDER_SEEN[] = nothing
    out = PO.factory(OrderProbe(; alpha = SimpleExpectedReturns()), pr, slv)

    # The context selection ran BEFORE the resolution, so the resolution saw the
    # optimiser's solver and not the measure's own `nothing`.
    @test ORDER_SEEN[].slv === slv
    @test out.slv === slv

    # The deferred slot still resolved, so moving the call did not skip it.
    @test isa(out.alpha, AbstractVector)
    @test length(out.alpha) == 3

    # A measure that states its own solver keeps it, which the selection has always done.
    own = Solver(; name = :own, solver = nothing)
    ORDER_SEEN[] = nothing
    out = PO.factory(OrderProbe(; slv = own, alpha = SimpleExpectedReturns()), pr, slv)
    @test ORDER_SEEN[].slv === own
    @test out.slv === own
end

#=
Issue #591. A measure resolves on two routes. The clustering route calls
`factory(hrp.r, pr, hrp.opt.slv)`, whose `@cprop` selection puts the effective solver on the
struct before the resolution runs. The `JuMP` risk-constraint route calls
`resolve_deferred_quantities` directly and calls no `factory`, so no selection runs and the
measure still holds the solver the caller stated — `nothing` in the common case, because a
measure's solver normally comes from `opt.slv`. A rule that reads the solver therefore
resolved against two different solvers, and inside `MeanRisk` the measure the result records
and the constraint the model was built from would disagree.

`set_risk_constraints!` now reads the estimator's own `opt.opt.slv` and threads it into the
resolution, and the resolution carries it as a third positional argument. The owner settles
it locally as `sel(x.slv, slv)`, beside the observation weights it already settles that way.

No shipped measure carries both a `@cprop slv` field and a calibration slot yet: #583 writes
the per-type methods for the twelve that carry the solver. So the wiring is gated on the
probe below, which carries both halves and builds a zero risk expression.
=#
const SLV_SEEN = Ref{Any}(nothing)
const BUILT_ALPHA = Ref{Any}(nothing)

# A rule whose value depends on the solver it is handed, so the number a route produced
# names the solver that route resolved against.
function probe_solver_rule(key::Symbol, pr::PO.AbstractPriorResult, w, slv,
                           ::PO.CalibrationContext)
    SLV_SEEN[] = (; key = key, weighted = !isnothing(w), slv = slv)
    return isnothing(slv) ? 0.1 : 0.05
end

PortfolioOptimisers.@propagatable struct SolverProbe{T1, T2, T3, T4} <: PO.RiskMeasure
    settings::T1
    @cprop slv::T2
    @pprop w::T3
    alpha::T4
end
function SolverProbe(; settings::RiskMeasureSettings = RiskMeasureSettings(), slv = nothing,
                     w = nothing, alpha = 0.05)
    return SolverProbe(settings, slv, w, alpha)
end
(::SolverProbe)(x::PO.VecNum) = sqrt(sum(abs2, x) / length(x))
PortfolioOptimisers.risk_input_kind(::SolverProbe) = PO.NetReturnsInput()
PortfolioOptimisers.calibration_slots(x::SolverProbe) = (; alpha = x.alpha)
function PortfolioOptimisers.resolve_deferred_quantities(x::SolverProbe,
                                                         pr::PO.AbstractPriorResult,
                                                         slv = nothing)
    # The two locals are the pair the ticket puts side by side: the effective observation
    # weights, which every owner already settled locally, and the effective solver.
    ws = PO.sel(x.w, pr.w)
    sv = PO.sel(x.slv, slv)
    return SolverProbe(; settings = x.settings, slv = x.slv, w = x.w,
                       alpha = PO.resolve_calibration_slot(x.alpha, :alpha, pr, ws, sv))
end
function PortfolioOptimisers.set_risk_constraints!(model::JuMP.Model, ::Any, r::SolverProbe,
                                                   opt::PO.RiskJuMPOptimisationEstimator,
                                                   pr::PO.AbstractPriorResult, args...;
                                                   prefix::Symbol = Symbol(""), kwargs...)
    # The builder records what it was handed, so the constraint's own number is in hand and
    # can be compared with the number the result records.
    BUILT_ALPHA[] = r.alpha
    return PO.state_build!(model, prefix, :probe_risk) do
        probe_risk = JuMP.@expression(model, zero(JuMP.AffExpr))
        PO.set_risk_bounds_and_expression!(model, opt, probe_risk, r.settings, :probe_risk;
                                           prefix = prefix)
        return probe_risk
    end
end

@testset "Calibration slot: the JuMP route resolves against the optimiser's solver" begin
    rng = StableRNG(192837465)
    X = randn(rng, 60, 4)
    pr = prior(EmpiricalPrior(), X)
    rd = ReturnsResult(; nx = string.(1:4), X = X)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = "verbose" => false)
    rule = probe_solver_rule

    # The caller states no solver on the measure, which is the common case: a measure's
    # solver comes from the optimiser.
    r = SolverProbe(; alpha = rule)
    @test isnothing(r.slv)

    opt = JuMPOptimiser(; slv = slv, pe = pr)
    mr = MeanRisk(; r = r, opt = opt)
    attrs = PO.processed_jump_optimiser_attributes(mr.opt, rd)
    model = JuMP.Model()
    PO.set_model_scales!(model, mr.opt.sc, mr.opt.so)
    PO.set_model_observations!(model, size(attrs.pr.X, 1))
    PO.set_maximum_ratio_factor_variables!(model, mr.obj)
    PO.set_w!(model, attrs.pr.X, mr.wi)
    PO.set_weight_constraints!(model, attrs.wb, mr.opt)
    SLV_SEEN[] = nothing
    BUILT_ALPHA[] = nothing
    PO.assemble_jump_model!(model, mr, mr.opt, attrs, rd, mr.r, mr.obj)

    # The rule saw the optimiser's own solver, and not the measure's `nothing`. This is the
    # assertion the ticket exists for: before the fix `slv` was `nothing` here.
    @test SLV_SEEN[].slv === slv
    @test SLV_SEEN[].key == :alpha
    @test SLV_SEEN[].weighted == false
    @test BUILT_ALPHA[] == 0.05

    # A vector of measures takes the same route, and the second overload threads the same
    # solver.
    mrv = MeanRisk(; r = [SolverProbe(; alpha = rule)], opt = opt)
    attrsv = PO.processed_jump_optimiser_attributes(mrv.opt, rd)
    modelv = JuMP.Model()
    PO.set_model_scales!(modelv, mrv.opt.sc, mrv.opt.so)
    PO.set_model_observations!(modelv, size(attrsv.pr.X, 1))
    PO.set_maximum_ratio_factor_variables!(modelv, mrv.obj)
    PO.set_w!(modelv, attrsv.pr.X, mrv.wi)
    PO.set_weight_constraints!(modelv, attrsv.wb, mrv.opt)
    SLV_SEEN[] = nothing
    PO.assemble_jump_model!(modelv, mrv, mrv.opt, attrsv, rd, mrv.r, mrv.obj)
    @test SLV_SEEN[].slv === slv

    # A measure that states its own solver keeps it, which is what `sel` has always done for
    # the weights beside it.
    own = Solver(; name = :own, solver = Clarabel.Optimizer)
    mro = MeanRisk(; r = SolverProbe(; slv = own, alpha = rule), opt = opt)
    attrso = PO.processed_jump_optimiser_attributes(mro.opt, rd)
    modelo = JuMP.Model()
    PO.set_model_scales!(modelo, mro.opt.sc, mro.opt.so)
    PO.set_model_observations!(modelo, size(attrso.pr.X, 1))
    PO.set_maximum_ratio_factor_variables!(modelo, mro.obj)
    PO.set_w!(modelo, attrso.pr.X, mro.wi)
    PO.set_weight_constraints!(modelo, attrso.wb, mro.opt)
    SLV_SEEN[] = nothing
    PO.assemble_jump_model!(modelo, mro, mro.opt, attrso, rd, mro.r, mro.obj)
    @test SLV_SEEN[].slv === own
end

@testset "Calibration slot: the two routes resolve against one solver" begin
    rng = StableRNG(192837465)
    X = randn(rng, 60, 4)
    pr = prior(EmpiricalPrior(), X)
    rd = ReturnsResult(; nx = string.(1:4), X = X)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = "verbose" => false)
    rule = probe_solver_rule
    r = SolverProbe(; alpha = rule)

    # The clustering route's call, verbatim: `factory(hrp.r, pr, hrp.opt.slv)`. The
    # selection puts the solver on the struct, and the resolution reads it there.
    SLV_SEEN[] = nothing
    fout = PO.factory(r, pr, slv)
    @test SLV_SEEN[].slv === slv
    @test fout.alpha == 0.05

    # The whole clustering route agrees with the constraint route, measure for measure.
    SLV_SEEN[] = nothing
    hrp = HierarchicalRiskParity(; r = r, opt = HierarchicalOptimiser(; slv = slv, pe = pr))
    res = optimise(hrp, rd)
    @test SLV_SEEN[].slv === slv
    @test isapprox(sum(res.w), 1)

    # A rule resolved with no solver at all still produces its own number, so the two
    # numbers above are the solver's doing and not the rule's.
    SLV_SEEN[] = nothing
    @test PO.resolve_deferred_quantities(r, pr).alpha == 0.1
    @test isnothing(SLV_SEEN[].slv)
end

@testset "Calibration slot: the MeanRisk result records the measure the model was built from" begin
    rng = StableRNG(192837465)
    X = randn(rng, 60, 4)
    pr = prior(EmpiricalPrior(), X)
    rd = ReturnsResult(; nx = string.(1:4), X = X)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = "verbose" => false)
    rule = probe_solver_rule

    # `_optimise` builds the model from `mr.r` and records `factory(mr.r, pr, mr.opt.slv)`.
    # The two calls are two routes over one measure, so the number the result carries must
    # be the number the constraint was built from.
    BUILT_ALPHA[] = nothing
    mr = MeanRisk(; r = SolverProbe(; alpha = rule),
                  opt = JuMPOptimiser(; slv = slv, pe = pr))
    res = optimise(mr, rd)
    @test isa(res.jr.retcode, PO.OptimisationSuccess)
    @test res.r.alpha == BUILT_ALPHA[]
    @test res.r.alpha == 0.05
end

#=
A declared calibration slot and the resolution that serves it are two separate statements,
and until now nothing paired them. The Deferred-Quantity channel owns
`assert_declared_slot_resolver` for exactly this failure: a type that declared the slot and
wrote no way to resolve it. The calibration channel had no counterpart, so a rule reached
the `JuMP` builders and was multiplied as though it were a number.

`assert_declared_calibration_resolver` is that counterpart. It walks the declaration the way
`assert_calibrated_slots` does, and it carries its own message: that one names a caller who
reached a value-level entry point, and this one names the library's own missing resolution.
=#

# A risk measure that declares its slot and resolves it nowhere. It is the failure the
# pairing exists to catch, and no shipped type is in this state.
#
# The declaration alone no longer reaches that state: `resolve_calibration_slots` derives the
# resolution from it, so a type that writes nothing is resolved rather than refused. The
# probe therefore opts out, which is what `L2Regularisation` and `LpRegularisation` do for
# their own reasons, and it then writes none of the resolution the opt-out promises.
PortfolioOptimisers.@propagatable struct UnpairedProbe{T1, T2, T3} <: PO.RiskMeasure
    settings::T1
    @pprop w::T2
    alpha::T3
end
function UnpairedProbe(; settings::RiskMeasureSettings = RiskMeasureSettings(), w = nothing,
                       alpha = 0.05)
    return UnpairedProbe(settings, w, alpha)
end
(::UnpairedProbe)(x::PO.VecNum) = sqrt(sum(abs2, x) / length(x))
PortfolioOptimisers.risk_input_kind(::UnpairedProbe) = PO.NetReturnsInput()
PortfolioOptimisers.calibration_slots(x::UnpairedProbe) = (; alpha = x.alpha)
function PortfolioOptimisers.resolve_calibration_slots(::UnpairedProbe,
                                                       ::PO.AbstractPriorResult,
                                                       ::Any = nothing)
    return (;)
end
function PortfolioOptimisers.set_risk_constraints!(model::JuMP.Model, ::Any,
                                                   r::UnpairedProbe,
                                                   opt::PO.RiskJuMPOptimisationEstimator,
                                                   pr::PO.AbstractPriorResult, args...;
                                                   prefix::Symbol = Symbol(""), kwargs...)
    return PO.state_build!(model, prefix, :unpaired_risk) do
        unpaired_risk = JuMP.@expression(model, zero(JuMP.AffExpr))
        PO.set_risk_bounds_and_expression!(model, opt, unpaired_risk, r.settings,
                                           :unpaired_risk; prefix = prefix)
        return unpaired_risk
    end
end

@testset "Calibration slot: the declaration is paired with its resolver" begin
    stated = CalibratedProbe(; alpha = 0.05)
    stated2 = CalibratedProbe(; alpha = 0.1)

    # Nothing to pair: a stated number needs no resolution and passes.
    @test isnothing(PO.assert_declared_calibration_resolver(stated))
    @test isnothing(PO.assert_declared_calibration_resolver(0.05))
    @test isnothing(PO.assert_declared_calibration_resolver(nothing))

    # A rule that survived the resolution names a declaration with no resolver.
    probe = CalibratedProbe(; alpha = RULE)
    @test_throws ArgumentError PO.assert_declared_calibration_resolver(probe)
    msg = try
        PO.assert_declared_calibration_resolver(probe)
        ""
    catch e
        sprint(showerror, e)
    end

    # The message names the type, the slot, the rule and the resolution to write beside the
    # declaration.
    @test occursin("CalibratedProbe.alpha", msg)
    @test occursin("ProbeScenarioCount", msg)
    @test occursin("calibration_slots", msg)
    @test occursin("resolve_calibration_slot(x.alpha, :alpha, pr, pr.w, slv)", msg)

    # The two halves carry two messages, so one failure never reads as the other. The
    # value-level half tells a caller to pass the prior result; this half tells the library
    # to write the resolution.
    vmsg = try
        PO.assert_calibrated_slots(probe)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("factory(r, pr)", vmsg)
    @test !occursin("factory(r, pr)", msg)
    @test !occursin("calibration_slots", vmsg)

    # The walk recurses into a child, so a container is paired through its children.
    nested = CalibratedProbe(; alpha = 0.05, child = probe)
    @test_throws ArgumentError PO.assert_declared_calibration_resolver(nested)

    # A slot that holds a vector of children is walked element by element.
    @test isnothing(PO.assert_declared_calibration_resolver([stated, stated2]))
    @test_throws ArgumentError PO.assert_declared_calibration_resolver([stated, probe])
end

@testset "Calibration slot: every shipped declaration names a type with a resolver" begin
    # The census reads the method table, so a declaration written tomorrow joins it on the
    # day it is written. Only the package's own methods are counted: the probes above and
    # in the sibling test files declare slots too, and they are not the library's.
    declared = Any[]
    for m in methods(PO.calibration_slots)
        m.module === PortfolioOptimisers || continue
        T = Base.unwrap_unionall(m.sig).parameters[2]
        T === Any && continue
        push!(declared, T)
    end
    @test !isempty(declared)

    # A declaration is served by the derivation unless the type opts out, so the census reads
    # the opt-outs rather than the declarations. `resolve_calibration_slots` has one untyped
    # method that does the work, and every other method of it is a type saying that its slot
    # is resolved somewhere else.
    optouts = Any[]
    for m in methods(PO.resolve_calibration_slots)
        m.module === PortfolioOptimisers || continue
        T = Base.unwrap_unionall(m.sig).parameters[2]
        T === Any && continue
        push!(optouts, T)
    end

    # A type that opts out names its own resolver. `factory` and `norm_ceiling_factory` are
    # the two routes ADR 0097 names, and `resolve_deferred_quantities` is the one an ordered
    # measure writes beside its declaration.
    resolvers = Any[]
    for f in (PO.resolve_deferred_quantities, PO.factory, PO.norm_ceiling_factory)
        for m in methods(f)
            m.module === PortfolioOptimisers || continue
            sig = Base.unwrap_unionall(m.sig).parameters
            length(sig) >= 2 || continue
            R = sig[2]
            R === Any && continue
            push!(resolvers, R)
        end
    end
    unpaired = [T for T in optouts if !any(R -> T <: R, resolvers)]
    @test isempty(unpaired)

    # The two opt-outs that ship are the two regularisation terms, and both are declared
    # types rather than an accident of the walk.
    @test all(T -> any(D -> D <: T || T <: D, declared), optouts)
end

@testset "Calibration slot: an unreachable slot fails loudly on the JuMP route" begin
    rng = StableRNG(1029384756)
    X = randn(rng, 60, 4)
    pr = prior(EmpiricalPrior(), X)
    rd = ReturnsResult(; nx = string.(1:4), X = X)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = "verbose" => false)
    rule = probe_rate

    # A `JuMP` builder reads the slot raw, so before the pairing this rule reached the
    # builder and the failure landed several frames down, inside a kernel that expected a
    # number. It is refused at the resolution instead, and the message names the type and
    # the slot.
    mr = MeanRisk(; r = UnpairedProbe(; alpha = rule),
                  opt = JuMPOptimiser(; slv = slv, pe = pr))
    @test_throws ArgumentError optimise(mr, rd)
    msg = try
        optimise(mr, rd)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("UnpairedProbe.alpha", msg)
    @test occursin("probe_rate", msg)

    # A vector of measures is walked the same way, so a lone measure and an aggregate are
    # refused alike.
    mrs = MeanRisk(; r = [Variance(), UnpairedProbe(; alpha = rule)],
                   opt = JuMPOptimiser(; slv = slv, pe = pr))
    @test_throws ArgumentError optimise(mrs, rd)

    # A stated number needs no resolution, so the same measure optimises when its slot
    # holds one. The refusal is about the rule and not about the declaration.
    ok = MeanRisk(; r = UnpairedProbe(; alpha = 0.05),
                  opt = JuMPOptimiser(; slv = slv, pe = pr))
    @test isa(optimise(ok, rd).retcode, PO.OptimisationSuccess)
end

@testset "Calibration slot: the resolution is derived from the declaration" begin
    rng = StableRNG(192837465)
    X = randn(rng, 60, 4)
    pr = prior(EmpiricalPrior(), X)
    ws = pweights(range(1, 2; length = 60))
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = "verbose" => false)

    # The declaration is the whole statement. A type that writes no resolution of its own is
    # resolved from `calibration_slots`, and the rebuild carries the number the rule gave.
    @test PO.resolve_deferred_quantities(ContextProbe(; alpha = CTX), pr).alpha == 0.1

    # The effective observation weights and the effective solver are read off the two fields
    # the library names everywhere. A type that carries the field settles it against the
    # caller's; a type that carries none takes the prior's and the caller's as they came.
    @test PO.resolve_deferred_quantities(ContextProbe(; alpha = CTX, w = ws), pr).alpha ==
          0.3
    @test PO.resolve_deferred_quantities(ContextProbe(; alpha = CTX, slv = slv), pr).alpha ==
          0.5
    @test PO.resolve_deferred_quantities(ContextProbe(; alpha = CTX, w = ws, slv = slv),
                                         pr).alpha == 0.7

    # A type with neither field reads both off its caller, which is the OWA weight builders'
    # state.
    @test PO.resolve_deferred_quantities(BareProbe(; alpha = CTX), pr).alpha == 0.1
    @test PO.resolve_deferred_quantities(BareProbe(; alpha = CTX), pr, slv).alpha == 0.5

    # A stated number resolves to itself, so the walk rebuilds nothing and returns the object
    # the caller handed in.
    stated = ContextProbe(; alpha = 0.05)
    @test PO.resolve_deferred_quantities(stated, pr) === stated

    # Every shipped measure whose slots carry no order between them takes the same route.
    # `ConditionalValueatRisk` carries `w` and no solver, and the four `PowerNorm` measures
    # carry both.
    @test PO.resolve_deferred_quantities(ConditionalValueatRisk(; alpha = CTX), pr).alpha ==
          0.1
    @test PO.resolve_deferred_quantities(ConditionalValueatRisk(; alpha = CTX, w = ws),
                                         pr).alpha == 0.3
    @test PO.resolve_deferred_quantities(PowerNormValueatRisk(; alpha = CTX, w = ws,
                                                              slv = slv), pr).alpha == 0.7

    # A weight builder carries neither field, so it reads the prior's weights and the solver
    # its container was given.
    @test PO.resolve_deferred_quantities(OrderedWeightsArrayTailGini(; alpha = CTX), pr,
                                         slv).alpha == 0.5

    # The container reaches the builder through the deferred recursion, and the calibration
    # walk then sees a resolved child and rebuilds nothing a second time.
    owa = OrderedWeightsArray(; w = OrderedWeightsArrayTailGini(; alpha = CTX))
    @test PO.resolve_deferred_quantities(owa, pr, slv).w.alpha == 0.5

    # The rebuild runs the inner constructor, so a joint bound that construction could not
    # check runs again on the calibrated number. `0 < alpha_i < alpha < 1` is that bound, and
    # a rule that returns a value at or below the stated `alpha_i` is refused at fold time
    # rather than stored.
    @test_throws DomainError PO.resolve_deferred_quantities(OrderedWeightsArrayTailGini(;
                                                                                        alpha_i = 0.2,
                                                                                        alpha = CTX),
                                                            pr)
end

@testset "Calibration slot: the three regularisation keys each reach the declaration" begin
    rng = StableRNG(5647382910)
    X = randn(rng, 60, 4)
    pr = prior(EmpiricalPrior(), X)

    # One field, one declaration naming `val`, and three resolution keys that are three
    # different quantities. ADR 0097 settles why the keys stay apart; the pairing is what
    # ties each of them back to the declaration that owns the slot.
    rrole = RateRadius(; c = 0.2)
    crole = EffectiveAssetFloor(; fraction = 0.5)

    l2 = factory(L2Regularisation(; val = rrole), pr)
    @test isa(l2.val, Number)
    @test isnothing(PO.assert_declared_calibration_resolver(l2))

    lp = factory(LpRegularisation(; p = 3, val = rrole), pr)
    @test isa(lp.val, Number)
    @test isnothing(PO.assert_declared_calibration_resolver(lp))

    lpc = PO.norm_ceiling_factory(LpRegularisation(; p = 3, val = crole), pr)
    @test isa(lpc.val, Number)
    @test isnothing(PO.assert_declared_calibration_resolver(lpc))

    # A vector of terms goes through the same verb, and each term is paired.
    lpcs = PO.norm_ceiling_factory([LpRegularisation(; p = 2, val = crole),
                                    LpRegularisation(; p = 3, val = crole)], pr)
    @test all(x -> isa(x.val, Number), lpcs)
    @test isnothing(PO.assert_declared_calibration_resolver(lpcs))

    # A stated number is carried through, and it is paired too.
    plain = factory(L2Regularisation(; val = 0.1), pr)
    @test plain.val == 0.1
    @test isnothing(PO.assert_declared_calibration_resolver(plain))
end

#=
Five **Calibration Rules** stand in the `alg` field of a rule type, and each one computes the
quantity of the slot its rule addresses. `test_09f_calibration_slot.jl` covers the mechanism
that carries them; this file covers the rules themselves.

`ScenarioCount` and `RateSignificance` compute a significance level, and `EntropyBudget`,
`HillTailDecay` and `RadialTailDecay` compute a Kaniadakis deformation parameter. The first
reads the effective observation weights, the second reads the raw row count, and the last
three read the probability of their own slot's end off the `CalibrationContext`.

No rule carries a range check of its own. Each returns the quantity of the slot it stands in,
so the slot owner's constructor is the whole validation and a value outside the slot's range
is refused there, at fold time. The three deformation rules carry the only checks any of them
carry, and each is a different claim: that the quantity exists at all. A target the band does
not reach leaves `EntropyBudget`'s sweep at an end of the interval, where the parameter is
far too small or too large to be the answer and yet still inside the range the slot owner
admits. A pool with no Hill estimate leaves `HillTailDecay` with nothing to invert, and a
covariance matrix that states no whitening leaves `RadialTailDecay` with no series to read.

Two of the five read the SHAPE of a series, and a slot owner tells them which series that is.
A drawdown measure prices the drawdown series of the portfolio and resolves the key `:kappa`,
which the value-at-risk twin resolves as well, so the key names no quantity and the marker
travels through the rule itself. `calibration_series` is the trait the owner answers and
`CalibrationContext.series` is the field that carries it, beside the significance level and
the norm order the same context carries. The reading itself does not move: the same standardisation, the same count and
the same estimator run over the drawdown series of each column, in place of the columns.

Issue #582 ships the first three rules, #611 ships `HillTailDecay` and #612 ships
`RadialTailDecay`. Issue #583 widens the slots that hold them, so every call below states the
resolution by hand.

The last two testsets read all ELEVEN rules rather than these five, because both gate a rule of
the whole family: a name states the method and carries the quantity only where the bare word is
taken, and a rule constructs bare unless the keyword it takes is the whole content of the rule.
ADR 0095 owns both lists.
=#
const PO = PortfolioOptimisers
using Distributions

# Two samples of different length, so that every rule is stated at two values of `T`.
const RNG = StableRNG(246813579)
const X60 = randn(RNG, 60, 4)
const PR60 = prior(EmpiricalPrior(), X60)
const PR70 = prior(EmpiricalPrior(), randn(RNG, 70, 4))
const PR120 = prior(EmpiricalPrior(), randn(RNG, 120, 4))

@testset "Calibration rules: the five rules join the two families" begin
    # Each rule subtypes the family whose quantity it computes, and no rule subtypes both.
    @test ScenarioCount <: PO.AbstractSignificanceCalibrationAlgorithm
    @test RateSignificance <: PO.AbstractSignificanceCalibrationAlgorithm
    @test EntropyBudget <: PO.AbstractDeformationCalibrationAlgorithm
    @test HillTailDecay <: PO.AbstractDeformationCalibrationAlgorithm
    @test RadialTailDecay <: PO.AbstractDeformationCalibrationAlgorithm
    @test !(EntropyBudget <: PO.AbstractSignificanceCalibrationAlgorithm)
    @test !(HillTailDecay <: PO.AbstractSignificanceCalibrationAlgorithm)
    @test !(RadialTailDecay <: PO.AbstractSignificanceCalibrationAlgorithm)
    @test !(ScenarioCount <: PO.AbstractDeformationCalibrationAlgorithm)

    # The family is what the slot bound checks, so each rule is admitted by one bound only.
    @test isa(ScenarioCount(; n = 25), PO.Num_SigCal)
    @test isa(RateSignificance(), PO.Num_SigCal)
    @test isa(EntropyBudget(; target = -1.3), PO.Num_DefCal)
    @test isa(HillTailDecay(), PO.Num_DefCal)
    @test isa(RadialTailDecay(), PO.Num_DefCal)
    @test !isa(ScenarioCount(; n = 25), PO.Num_DefCal)
    @test !isa(EntropyBudget(; target = -1.3), PO.Num_SigCal)
    @test !isa(HillTailDecay(), PO.Num_SigCal)
    @test !isa(RadialTailDecay(), PO.Num_SigCal)

    # A slot stores the rule the caller wrote, and both ends of a Range measure take it.
    @test ConditionalValueatRisk(; alpha = ScenarioCount(; n = 25)).alpha ==
          ScenarioCount(; n = 25)
    @test ConditionalValueatRiskRange(; alpha = RateSignificance()).beta ==
          RateSignificance()
    @test RelativisticValueatRisk(; kappa = EntropyBudget(; target = -1.3)).kappa ==
          EntropyBudget(; target = -1.3)
    @test RelativisticValueatRiskRange(; kappa_a = HillTailDecay()).kappa_b ==
          HillTailDecay()
    @test RelativisticValueatRisk(; kappa = RadialTailDecay()).kappa == RadialTailDecay()

    # The wrong family is refused at construction, by the slot's own bound.
    @test_throws TypeError ConditionalValueatRisk(; alpha = EntropyBudget(; target = -1))
    @test_throws TypeError RelativisticValueatRisk(; kappa = ScenarioCount(; n = 25))
    @test_throws TypeError ConditionalValueatRisk(; alpha = HillTailDecay())
    @test_throws TypeError ConditionalValueatRisk(; alpha = RadialTailDecay())

    # The five rules are caller-facing, because a caller states one directly.
    exported = names(PortfolioOptimisers)
    @test :ScenarioCount in exported
    @test :RateSignificance in exported
    @test :EntropyBudget in exported
    @test :HillTailDecay in exported
    @test :RadialTailDecay in exported

    # The positional inner constructor is the route a rebuild takes. Each rule holds its
    # own parameters and nothing of the site's, so the five constructors are short.
    @test ScenarioCount(25).n == 25
    @test RateSignificance(2).c == 2
    @test EntropyBudget(-1.3).target == -1.3
    @test HillTailDecay(12).kmin == 12
    @test RadialTailDecay(12).kmin == 12
    @test fieldnames(HillTailDecay) == (:kmin,)
    @test fieldnames(RadialTailDecay) == (:kmin,)
    @test fieldnames(EntropyBudget) == (:target,)

    # The series defaults to the returns, which is what every rule read before the marker
    # existed. It lives on the context, so a site that names no other reads what it read
    # then.
    @test isa(CalibrationContext().series, ReturnsSeries)
    @test isnothing(CalibrationContext().alpha)
    @test isnothing(CalibrationContext().p)
    @test isa(CalibrationContext(-1, AbsoluteDrawdownSeries(), 2, nothing).series,
              AbsoluteDrawdownSeries)
    # `cs` belongs to the Cross-Sectional Factor Prior, so a site that names none leaves it
    # unset.
    @test isnothing(CalibrationContext().cs)

    # `kmin` is a count of order statistics, so a rule that states none is refused at
    # construction. It is a check on the rule's OWN parameter, and not on the parameter the
    # rule returns.
    @test_throws DomainError HillTailDecay(; kmin = 0)
    @test_throws DomainError RadialTailDecay(; kmin = 0)

    # The same rule holds for the scalar of every other rule, so all eleven refuse a value
    # that no sample can make sensible. `ScenarioCount.n` is a count of observations and
    # `RateSignificance.c` is a rate coefficient, so both are positive and finite.
    # `EntropyBudget.target` may take either sign, because the band it must land in follows
    # the sample, so only the finiteness is a question the constructor can answer.
    @test_throws DomainError ScenarioCount(; n = 0)
    @test_throws DomainError ScenarioCount(; n = -1)
    @test_throws DomainError ScenarioCount(; n = Inf)
    @test_throws DomainError RateSignificance(; c = 0)
    @test_throws DomainError RateSignificance(; c = -1.0)
    @test_throws DomainError RateSignificance(; c = Inf)
    @test_throws DomainError EntropyBudget(; target = NaN)
    @test_throws DomainError EntropyBudget(; target = -Inf)
    @test EntropyBudget(; target = -0.5).target == -0.5

    # `RateSignificance.c` and `RateRadius.c` carry the same name, the same default and the
    # same closed form, so they refuse the same value in the same place.
    @test_throws DomainError RateRadius(; c = -1.0)
end

@testset "Calibration rules: `ScenarioCount` fixes the count, not the probability" begin
    rule = ScenarioCount(; n = 15)

    # The tail holds `n` observations, so `alpha * T` is the count the caller stated and
    # `ceil(alpha * T)` returns it whatever `T` is.
    a60 = rule(:alpha, PR60, nothing, nothing, CalibrationContext())
    a120 = rule(:alpha, PR120, nothing, nothing, CalibrationContext())
    @test a60 == 15 / 60
    @test a120 == 15 / 120
    @test ceil(a60 * 60) == 15
    @test ceil(a120 * 120) == 15

    # The sample doubled, so the probability halved. This is the whole reason for the rule:
    # a stated `alpha` would have left half as many observations in the tail.
    @test a120 == a60 / 2

    # The key does not change the count, so a tail slot and a head slot that carry one rule
    # resolve to one number.
    @test rule(:beta, PR60, nothing, nothing, CalibrationContext()) == a60

    # Uniform weights carry the same information as the unweighted sample, so Kish's
    # effective sample size is the row count and the answer does not move.
    @test rule(:alpha, PR60, pweights(fill(1 / 60, 60)), nothing, CalibrationContext()) ≈
          a60

    # Uneven weights carry less, so the effective sample is shorter and the tail must take a
    # larger share of it to hold the same 15 observations.
    uneven = pweights([fill(3.0, 30); fill(1.0, 30)])
    aw = rule(:alpha, PR60, uneven, nothing, CalibrationContext())
    @test aw ≈ 15 / (sum(uneven)^2 / sum(abs2, uneven))
    @test aw > a60

    # The solver reaches the rule and this one ignores it, which is the fourth argument's
    # whole contract here.
    @test rule(:alpha, PR60, nothing, Solver(; name = :probe, solver = nothing),
               CalibrationContext()) == a60

    # The rule carries no range check, so a count larger than the sample returns a value the
    # slot owner refuses rather than one the rule refuses.
    @test ScenarioCount(; n = 90)(:alpha, PR60, nothing, nothing, CalibrationContext()) ==
          1.5
end

@testset "Calibration rules: `RateSignificance` falls with the square root of `T`" begin
    rule = RateSignificance()

    # The default coefficient is the plain `1/sqrt(T)` rate.
    @test rule.c == 1
    a60 = rule(:alpha, PR60, nothing, nothing, CalibrationContext())
    a120 = rule(:alpha, PR120, nothing, nothing, CalibrationContext())
    @test a60 == inv(sqrt(60))
    @test a120 == inv(sqrt(120))

    # The sample doubled, so the probability fell by `sqrt(2)` and the tail's expected count
    # ROSE by the same factor. A longer sample buys a further tail, not only a fuller one.
    @test a120 ≈ a60 / sqrt(2)
    @test a120 * 120 ≈ sqrt(2) * (a60 * 60)

    # The coefficient scales the rate.
    @test RateSignificance(; c = 2)(:alpha, PR60, nothing, nothing, CalibrationContext()) ==
          2 * a60

    # The rule reads the RAW row count, so the weights do not move it. That is the stated
    # difference from `ScenarioCount`: a rate speaks of the length of the record.
    @test rule(:alpha, PR60, pweights([fill(3.0, 30); fill(1.0, 30)]), nothing,
               CalibrationContext()) == a60

    # The key does not change the rate either.
    @test rule(:beta, PR60, nothing, nothing, CalibrationContext()) == a60
end

@testset "Calibration rules: `EntropyBudget` inverts the Kaniadakis logarithm" begin
    target = -1.3
    rule = EntropyBudget(; target = target)

    # The returned parameter meets the target: it is the coefficient `RRM` multiplies its
    # dual variable by, and the rule solves for the deformation that prices it.
    k60 = rule(:kappa, PR60, nothing, nothing, CalibrationContext(; alpha = 0.05))
    @test 0 < k60 < 1
    @test PO.kappa_log(inv(0.05 * 60), k60) ≈ target

    # The same budget on a longer sample. The band the logarithm reaches moves with `T`, so
    # the two samples are 60 and 70 rather than 60 and 120.
    k70 = rule(:kappa, PR70, nothing, nothing, CalibrationContext(; alpha = 0.05))
    @test 0 < k70 < 1
    @test PO.kappa_log(inv(0.05 * 70), k70) ≈ target

    # A longer sample reaches the same budget with LESS deformation, which is the direction
    # the rule states: the price is held and the shape moves.
    @test k70 < k60

    # A larger `alpha` widens the tail and moves the answer on one sample.
    @test EntropyBudget(; target = target)(:kappa, PR60, nothing, nothing,
                                           CalibrationContext(; alpha = 0.06)) != k60

    # The key does not change the budget, and the solver is not needed: the inversion is a
    # scalar one, so this rule never reaches `RRM` itself.
    @test rule(:kappa_a, PR60, nothing, nothing, CalibrationContext(; alpha = 0.05)) == k60
    @test rule(:kappa, PR60, nothing, Solver(; name = :probe, solver = nothing),
               CalibrationContext(; alpha = 0.05)) == k60
    @test rule(:kappa, PR60, pweights(fill(1 / 60, 60)), nothing,
               CalibrationContext(; alpha = 0.05)) == k60

    # The band the coefficient reaches over `(0, 1)` runs from `log(u)` to `sinh(log(u))`,
    # and it moves with the sample. A target past either end has no root, so the rule
    # refuses it rather than returning the end of the sweep. That value would be far too
    # small or too large to be the answer, and yet still inside the range the slot owner
    # admits, so the slot owner cannot catch it.
    band = (PO.kappa_log(inv(0.05 * 60), 1), log(inv(0.05 * 60)))
    @test band[1] < target < band[2]
    @test_throws DomainError EntropyBudget(; target = -0.5)(:kappa, PR60, nothing, nothing,
                                                            CalibrationContext(;
                                                                               alpha = 0.05))
    @test_throws DomainError EntropyBudget(; target = -5.0)(:kappa, PR60, nothing, nothing,
                                                            CalibrationContext(;
                                                                               alpha = 0.05))
    msg = try
        EntropyBudget(; target = -0.5)(:kappa, PR60, nothing, nothing,
                                       CalibrationContext(; alpha = 0.05))
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("EntropyBudget.target", msg)
    @test occursin("T = 60", msg)

    # The band moves with the sample, so the target that suits 60 and 70 observations lies
    # outside the band of 120. That is what the message warns about.
    @test_throws DomainError rule(:kappa, PR120, nothing, nothing,
                                  CalibrationContext(; alpha = 0.05))

    # A rule whose sibling never arrived cannot form `inv(alpha * T)`, and it says so.
    unbound = EntropyBudget(; target = target)
    @test !hasfield(EntropyBudget, :alpha)
    @test_throws PO.IsNothingError unbound(:kappa, PR60, nothing, nothing,
                                           CalibrationContext())
    msg = try
        unbound(:kappa, PR60, nothing, nothing, CalibrationContext())
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("CalibrationContext.alpha", msg)
    @test occursin("EntropyBudget", msg)
end

@testset "Calibration rules: the context hands the sibling over" begin
    rule = EntropyBudget(; target = -1.3)
    ctx = CalibrationContext(; alpha = 0.05)

    # The number lives in the context and nowhere else. The rule holds no field for it, so
    # there is no copy to keep in step and nothing to rebuild on the way in.
    @test !hasfield(EntropyBudget, :alpha)
    @test ctx.alpha == 0.05
    @test rule(:kappa, PR60, nothing, nothing, ctx) ==
          EntropyBudget(; target = -1.3)(:kappa, PR60, nothing, nothing, ctx)

    # The resolver hands the context to the rule, so the SLOT is what a per-type method
    # passes and the occupant never moves. One rule serves the tail slot and the head slot
    # alike, because the slot is what names the end.
    @test PO.resolve_calibration_slot(rule, :kappa, PR60, nothing, nothing, ctx) ==
          rule(:kappa, PR60, nothing, nothing, ctx)
    @test PO.resolve_calibration_slot(rule, :kappa_b, PR60, nothing, nothing, ctx) ==
          rule(:kappa_b, PR60, nothing, nothing, ctx)

    # The second rule that reads a sibling reads the same field. A head slot's number is
    # the `beta` of its end, and the context names the reading rather than the spelling.
    hill = HillTailDecay(; kmin = 3)
    @test !hasfield(HillTailDecay, :alpha)
    @test hill.kmin == 3
    hctx = CalibrationContext(; alpha = 0.05)
    @test PO.resolve_calibration_slot(hill, :kappa_b, PR60, nothing, nothing, hctx) ==
          hill(:kappa_b, PR60, nothing, nothing, hctx)

    # A stated number passes through untouched. That is what lets the slot owner build one
    # context for every `kappa` slot rather than for some of them.
    @test PO.resolve_calibration_slot(0.3, :kappa, PR60, nothing, nothing, ctx) === 0.3
    @test PO.resolve_calibration_slot(nothing, :kappa, PR60, nothing, nothing, ctx) ===
          nothing

    # A rule that reads no sibling ignores the field, whether it is typed or a plain
    # function, and so does the rule it stands in.
    probe(::Symbol, ::PO.AbstractPriorResult, ::Any, ::Any, ::PO.CalibrationContext) = 0.3
    @test PO.resolve_calibration_slot(probe, :kappa, PR60, nothing, nothing, ctx) == 0.3
    @test PO.resolve_calibration_slot(probe, :kappa, PR60, nothing, nothing,
                                      CalibrationContext()) == 0.3

    # No significance rule reads a sibling, so the whole family answers the same number
    # under a context that names one and under the default.
    rule = ScenarioCount(; n = 25)
    @test PO.resolve_calibration_slot(rule, :alpha, PR60, nothing, nothing, ctx) ==
          PO.resolve_calibration_slot(rule, :alpha, PR60, nothing, nothing)
end

@testset "Calibration rules: the resolver runs each rule" begin
    # This is the shape #583's per-type method writes. A significance slot resolves on its
    # own; the deformation slot takes the number the significance slot produced.
    slot = ScenarioCount(; n = 3)
    alpha = PO.resolve_calibration_slot(slot, :alpha, PR60, nothing)
    @test alpha == 3 / 60

    kslot = EntropyBudget(; target = -1.3)
    kappa = PO.resolve_calibration_slot(kslot, :kappa, PR60, nothing, nothing,
                                        CalibrationContext(; alpha = alpha))
    @test 0 < kappa < 1
    @test PO.kappa_log(inv(alpha * 60), kappa) ≈ -1.3

    # The pair refits on a shorter fold, which is the whole point of a rule over a number.
    fold = prior(EmpiricalPrior(), X60[1:30, :])
    alpha30 = PO.resolve_calibration_slot(slot, :alpha, fold, nothing)
    @test alpha30 == 3 / 30
    @test alpha30 != alpha

    # A rule reached through a plain `Function` needs no wrapper either, and the rate rule
    # resolves through the same verb.
    @test PO.resolve_calibration_slot(RateSignificance(), :beta, PR60, nothing) ==
          inv(sqrt(60))
end

#=
`HillTailDecay` is the rule of issue #611, and it reads a quantity the other three do not:
the SHAPE of the sample's own tail. The inverse of `kappa_log` is the κ-exponential, whose
tail is a power law of index `1/κ`, so κ is a reciprocal tail index and the rule returns
`1 / â` for Hill's `â`. A Student-t of ν degrees of freedom has tail index ν, so a t draw is
a sample whose answer is known before the rule runs.

The estimate is taken over a POOL. Every column is centred, divided by its own dispersion,
signed to the end the slot prices, and the `T * N` values are pooled. `alpha` fixes the count
`k = ceil(alpha * T * N)` AND the depth the tail is read at, and a Hill estimate at a shallow
depth is biased upward on a Student-t, which approaches its power law slowly. The draws below
are read at `alpha = 0.01` for that reason. Each band is wide enough to hold that bias and
narrow enough to separate the three draws.
=#
const PRT3 = prior(EmpiricalPrior(),
                   rand(StableRNG(30001), Distributions.TDist(3), 1000, 20))
const PRT4 = prior(EmpiricalPrior(),
                   rand(StableRNG(40001), Distributions.TDist(4), 1000, 20))
const PRT4S = prior(EmpiricalPrior(),
                    rand(StableRNG(40002), Distributions.TDist(4), 500, 20))
const PRT6 = prior(EmpiricalPrior(),
                   rand(StableRNG(60001), Distributions.TDist(6), 1000, 20))

# A sample whose two tails carry two different indices. `-abs(t3)` reaches only the loss
# side and `+abs(t9)` only the gain side, so the left tail is a t(3) and the right a t(9).
const XSKEW = let rng = StableRNG(97531)
    -abs.(rand(rng, Distributions.TDist(3), 800, 12)) .+
    abs.(rand(rng, Distributions.TDist(9), 800, 12))
end
const PRSKEW = prior(EmpiricalPrior(), XSKEW)

@testset "Calibration rules: `HillTailDecay` returns the reciprocal tail index" begin
    rule = HillTailDecay(; kmin = 30)

    # The answer is `1 / ν`, and the band holds the Hill bias, which is upward at this
    # depth on a t draw.
    k3 = rule(:kappa, PRT3, nothing, nothing, CalibrationContext(; alpha = 0.01))
    k4 = rule(:kappa, PRT4, nothing, nothing, CalibrationContext(; alpha = 0.01))
    k6 = rule(:kappa, PRT6, nothing, nothing, CalibrationContext(; alpha = 0.01))
    for (kap, nu) in ((k3, 3), (k4, 4), (k6, 6))
        @test 0 < kap < 1
        @test 0.75 / nu <= kap <= 1.6 / nu
    end

    # The ORDER carries no bias: a heavier tail returns a larger κ, and the three draws
    # separate. This is the reading the rule claims, stated without a tolerance.
    @test k3 > k4 > k6

    # The same tail index at two sample lengths, which is the pair `test_09f` and #582 are
    # stated at. `T` moves the pool and the count with it, and both readings hold the band.
    k4s = rule(:kappa, PRT4S, nothing, nothing, CalibrationContext(; alpha = 0.01))
    @test 0.75 / 4 <= k4s <= 1.6 / 4
    @test k4s != k4

    # `alpha` sets the DEPTH as well as the count. A shallower reading takes in more of the
    # body, which a Student-t makes look heavier than its tail is, so the deeper reading
    # sits closer to the truth.
    shallow = HillTailDecay(; kmin = 30)(:kappa, PRT4, nothing, nothing,
                                         CalibrationContext(; alpha = 0.05))
    @test shallow > k4
    @test abs(k4 - 1 / 4) < abs(shallow - 1 / 4)

    # The rule reads `pr.X` ALONE. A prior whose covariance matrix is scaled a hundredfold
    # returns the same number, because a column's dispersion comes from that column. This
    # is the line that separates the rule from one that whitens with `sigma`.
    scaled = LowOrderPrior(; X = PRT4.X, mu = PRT4.mu, sigma = 100 .* PRT4.sigma)
    @test rule(:kappa, scaled, nothing, nothing, CalibrationContext(; alpha = 0.01)) == k4

    # A tail index is a statement about the shape of a series rather than about the count of
    # observations behind it, so the rule ignores the observation weights. `ScenarioCount`
    # reads Kish's effective sample size and `RateSignificance` reads the raw row count;
    # this rule reads neither.
    @test rule(:kappa, PRT4, pweights([fill(3.0, 500); fill(1.0, 500)]), nothing,
               CalibrationContext(; alpha = 0.01)) == k4

    # The solver reaches the rule and this one needs none: the estimate is a closed form.
    @test rule(:kappa, PRT4, nothing, Solver(; name = :probe, solver = nothing),
               CalibrationContext(; alpha = 0.01)) == k4
end

@testset "Calibration rules: `HillTailDecay` answers per end" begin
    rule = HillTailDecay(; kmin = 20)

    # `key` says which end the slot prices, so the two ends of a skewed sample resolve to
    # two different numbers. This is the OPPOSITE of the other three rules, and the reason
    # is that a tail index is a statement about ONE tail.
    ka = rule(:kappa_a, PRSKEW, nothing, nothing, CalibrationContext(; alpha = 0.01))
    kb = rule(:kappa_b, PRSKEW, nothing, nothing, CalibrationContext(; alpha = 0.01))
    @test ka > kb
    @test 0.75 / 3 <= ka <= 1.6 / 3
    @test 0.75 / 9 <= kb <= 1.6 / 9

    # `:kappa` is a loss key, so the scalar measures read the loss tail.
    @test rule(:kappa, PRSKEW, nothing, nothing, CalibrationContext(; alpha = 0.01)) == ka

    # `EntropyBudget` states the opposite in its own docstring: a budget is a price the
    # model pays, so it is one number for both ends of the same sample.
    budget = EntropyBudget(; target = -1.3)
    @test budget(:kappa_a, PR60, nothing, nothing, CalibrationContext(; alpha = 0.05)) ==
          budget(:kappa_b, PR60, nothing, nothing, CalibrationContext(; alpha = 0.05))
end

@testset "Calibration rules: `HillTailDecay` gives a Range measure two ends" begin
    # The travelling pair costs nothing new: `RelativisticValueatRiskRange` already puts
    # `alpha` in the tail slot's context and `beta` in the head slot's, so the rule reads
    # one field and holds none.
    rg = RelativisticValueatRiskRange(; alpha = 0.01, kappa_a = HillTailDecay(; kmin = 20),
                                      beta = 0.02, kappa_b = HillTailDecay(; kmin = 20))
    og = PO.resolve_deferred_quantities(rg, PRSKEW)
    @test og.kappa_a ≈ HillTailDecay(; kmin = 20)(:kappa_a, PRSKEW, nothing, nothing,
                                                  CalibrationContext(; alpha = 0.01))
    @test og.kappa_b ≈ HillTailDecay(; kmin = 20)(:kappa_b, PRSKEW, nothing, nothing,
                                                  CalibrationContext(; alpha = 0.02))
    @test og.kappa_a != og.kappa_b
    @test 0 < og.kappa_a < 1
    @test 0 < og.kappa_b < 1

    # The head end really reads its OWN probability. The same rule bound to the tail's
    # `alpha` answers differently, so the pairing is not an accident of the two ends holding
    # one number.
    @test og.kappa_b != HillTailDecay(; kmin = 20)(:kappa_b, PRSKEW, nothing, nothing,
                                                   CalibrationContext(; alpha = 0.01))
end

#=
The refusals. Each one states that the estimate EXISTS, which is the shape of the check
`EntropyBudget` carries. None of them is a range check on the returned κ: the slot owner's
constructor keeps that job.
=#
# 90 zeros and 10 ones per column. Centred, the column holds 90 values below its mean and 10
# above, so the loss pool holds 360 positive values out of 400 and the 381st is negative.
const XSPIKE = repeat([zeros(90); ones(10)], 1, 4)
const PRSPIKE = prior(EmpiricalPrior(), XSPIKE)
# A tail index below one, which is a tail with no finite mean.
const PRHEAVY = prior(EmpiricalPrior(),
                      rand(StableRNG(13579), Distributions.TDist(0.7), 300, 4))

@testset "Calibration rules: `HillTailDecay` refuses an estimate it cannot form" begin
    # A rule whose sibling never arrived cannot form the count, and it says so. This is the
    # message `EntropyBudget` carries, in the words of this rule's own count.
    unbound = HillTailDecay()
    @test !hasfield(typeof(unbound), :alpha)
    @test unbound.kmin == 30
    @test_throws PO.IsNothingError unbound(:kappa, PR60, nothing, nothing,
                                           CalibrationContext())
    msg = try
        unbound(:kappa, PR60, nothing, nothing, CalibrationContext())
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("CalibrationContext.alpha", msg)
    @test occursin("HillTailDecay", msg)

    # The floor. `PR60` pools 240 values, so `alpha = 0.05` leaves 12 order statistics and
    # the default floor of 30 refuses them. A Hill estimate on 12 points moves from fold to
    # fold for no reason in the data.
    @test_throws DomainError HillTailDecay()(:kappa, PR60, nothing, nothing,
                                             CalibrationContext(; alpha = 0.05))
    msg = try
        HillTailDecay()(:kappa, PR60, nothing, nothing, CalibrationContext(; alpha = 0.05))
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("kmin", msg)
    @test occursin("floor at 30", msg)
    @test occursin("= 12", msg)

    # A floor the sample clears returns a number, so the refusal is the floor's and not the
    # sample's.
    @test 0 <
          HillTailDecay(; kmin = 5)(:kappa, PR60, nothing, nothing,
                                    CalibrationContext(; alpha = 0.05)) <
          1

    # A probability that takes the whole sample leaves no `k + 1`-th value to divide by.
    @test_throws DomainError HillTailDecay(; kmin = 5)(:kappa, PR60, nothing, nothing,
                                                       CalibrationContext(; alpha = 0.999))
    msg = try
        HillTailDecay(; kmin = 5)(:kappa, PR60, nothing, nothing,
                                  CalibrationContext(; alpha = 0.999))
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("241 pooled values", msg)
    @test occursin("holds 240", msg)

    # The pool holds fewer positive values than the count asks for, so there is no Hill
    # estimate at all. A sample with fewer losses than `k + 1` is the case that produces it.
    @test_throws DomainError HillTailDecay(; kmin = 5)(:kappa, PRSPIKE, nothing, nothing,
                                                       CalibrationContext(; alpha = 0.95))
    msg = try
        HillTailDecay(; kmin = 5)(:kappa, PRSPIKE, nothing, nothing,
                                  CalibrationContext(; alpha = 0.95))
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("381-th largest", msg)
    @test occursin("not positive", msg)

    # The gain end of the same sample runs out as well, and for the mirror reason: the pool
    # holds 40 values above the column mean and 360 below it, so neither count reaches 381.
    # The refusal is the count's, and the floor case above is where a sample that clears it
    # returns a number.
    @test_throws DomainError HillTailDecay(; kmin = 5)(:kappa_b, PRSPIKE, nothing, nothing,
                                                       CalibrationContext(; alpha = 0.95))

    # An estimate at or below one is a tail with no finite mean, and no admissible κ reads
    # it. The band `(0, 1)` the slot admits IS the condition `â > 1`, so the refusal is the
    # reading rather than a guard bolted onto it.
    @test_throws DomainError HillTailDecay(; kmin = 5)(:kappa, PRHEAVY, nothing, nothing,
                                                       CalibrationContext(; alpha = 0.05))
    msg = try
        HillTailDecay(; kmin = 5)(:kappa, PRHEAVY, nothing, nothing,
                                  CalibrationContext(; alpha = 0.05))
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("tail index of 0.6", msg)
    @test occursin("(0, 1)", msg)
end

#=
`RadialTailDecay` is the rule of issue #612, and it is the sibling of `HillTailDecay`. It
reads the same quantity, the reciprocal of a tail index, off a different series. Every row of
`pr.X` is centred on `pr.mu` and whitened by the factor of `pr.sigma`, and the norm of the
whitened row is that observation's Mahalanobis distance. The `T` distances are ONE series,
and under an elliptical scale mixture that series carries the mixture's tail index. So a
multivariate Student-t draw of ν degrees of freedom is a sample whose answer is known before
the rule runs, on the same terms as the univariate draws above.

The series has no sign, so the rule answers ONE number for both ends of a Range measure where
its sibling answers two. It also holds `T` entries where the pool of the sibling holds `T N`,
so the same `alpha` leaves far fewer order statistics and the draws below are long.
=#
function mvt_prior(seed, nu, T, N)
    rng = StableRNG(seed)
    A = randn(rng, N, N)
    S = A * A' / N + 0.5I
    d = Distributions.MvTDist(float(nu), zeros(N), Matrix(S))
    return prior(EmpiricalPrior(), permutedims(rand(rng, d, T)))
end
const PRMVT3 = mvt_prior(1003, 3, 4000, 8)
const PRMVT4 = mvt_prior(1004, 4, 4000, 8)
const PRMVT4S = mvt_prior(1004, 4, 2000, 8)
const PRMVT6 = mvt_prior(1006, 6, 4000, 8)

@testset "Calibration rules: `RadialTailDecay` returns the reciprocal tail index" begin
    rule = RadialTailDecay()

    # The answer is `1 / ν`, and the band holds the Hill bias. It is the band the sibling's
    # own draws are read in, and both rules estimate the same quantity.
    k3 = rule(:kappa, PRMVT3, nothing, nothing, CalibrationContext(; alpha = 0.05))
    k4 = rule(:kappa, PRMVT4, nothing, nothing, CalibrationContext(; alpha = 0.05))
    k6 = rule(:kappa, PRMVT6, nothing, nothing, CalibrationContext(; alpha = 0.05))
    for (kap, nu) in ((k3, 3), (k4, 4), (k6, 6))
        @test 0 < kap < 1
        @test 0.75 / nu <= kap <= 1.6 / nu
    end

    # The ORDER carries no bias: a heavier tail returns a larger κ, and the three draws
    # separate. This is the reading the rule claims, stated without a tolerance.
    @test k3 > k4 > k6

    # The same tail index at a second sample length. `T` moves the series and the count with
    # it, and both readings hold the band.
    k4s = rule(:kappa, PRMVT4S, nothing, nothing, CalibrationContext(; alpha = 0.05))
    @test 0.75 / 4 <= k4s <= 1.6 / 4
    @test k4s != k4

    # The rule IS the Hill estimate of the Mahalanobis distances, so the number comes back
    # from `X`, `mu` and `sigma` by hand. This is the whole of what the rule computes.
    U = cholesky(PRMVT4.sigma).U
    d = [norm(transpose(U) \ (PRMVT4.X[t, :] .- PRMVT4.mu)) for t in axes(PRMVT4.X, 1)]
    k = ceil(Int, 0.05 * length(d))
    u = sort(d; rev = true)
    ahat = k / sum(log(u[i] / u[k + 1]) for i in 1:k)
    @test k4 ≈ inv(ahat)
    @test k == 200

    # A tail index is a statement about the shape of a series rather than about the count of
    # observations behind it, so the rule ignores the observation weights.
    @test rule(:kappa, PRMVT4, pweights([fill(3.0, 2000); fill(1.0, 2000)]), nothing,
               CalibrationContext(; alpha = 0.05)) == k4

    # The solver reaches the rule and this one needs none: the estimate is a closed form.
    @test rule(:kappa, PRMVT4, nothing, Solver(; name = :probe, solver = nothing),
               CalibrationContext(; alpha = 0.05)) == k4

    # `alpha` sets the DEPTH as well as the count, on the same terms as the sibling.
    @test RadialTailDecay()(:kappa, PRMVT4, nothing, nothing,
                            CalibrationContext(; alpha = 0.01)) != k4
end

@testset "Calibration rules: `RadialTailDecay` reads the covariance matrix" begin
    rule = RadialTailDecay(; kmin = 5)

    # The line that separates this rule from `HillTailDecay`: the shape of the covariance
    # matrix reaches the answer. The sibling standardises each column on its own, so a
    # covariance matrix with the same diagonal and no off-diagonal term leaves it unmoved.
    diagonal = LowOrderPrior(; X = PRMVT4.X, mu = PRMVT4.mu,
                             sigma = Matrix(Diagonal(diag(PRMVT4.sigma))))
    @test rule(:kappa, diagonal, nothing, nothing, CalibrationContext(; alpha = 0.05)) !=
          rule(:kappa, PRMVT4, nothing, nothing, CalibrationContext(; alpha = 0.05))
    @test HillTailDecay(; kmin = 5)(:kappa, diagonal, nothing, nothing,
                                    CalibrationContext(; alpha = 0.05)) ==
          HillTailDecay(; kmin = 5)(:kappa, PRMVT4, nothing, nothing,
                                    CalibrationContext(; alpha = 0.05))

    # `chol` and `sigma` state the same factorisation, so they give the same number.
    both = LowOrderPrior(; X = PRMVT4.X, mu = PRMVT4.mu, sigma = PRMVT4.sigma,
                         chol = cholesky(PRMVT4.sigma).U)
    @test rule(:kappa, both, nothing, nothing, CalibrationContext(; alpha = 0.05)) ==
          rule(:kappa, PRMVT4, nothing, nothing, CalibrationContext(; alpha = 0.05))

    # **`chol` takes precedence over `sigma`**, which is the rule the `chol` field states. A
    # carrier whose two fields disagree is read by its `chol`, and the identity `sigma` it
    # also carries reaches nothing.
    A = randn(StableRNG(99), 8, 8)
    other = A * A' + 8I
    precedence = LowOrderPrior(; X = PRMVT4.X, mu = PRMVT4.mu, sigma = Matrix(1.0I, 8, 8),
                               chol = cholesky(other).U)
    stated = LowOrderPrior(; X = PRMVT4.X, mu = PRMVT4.mu, sigma = other)
    identity_sigma = LowOrderPrior(; X = PRMVT4.X, mu = PRMVT4.mu,
                                   sigma = Matrix(1.0I, 8, 8))
    @test rule(:kappa, precedence, nothing, nothing, CalibrationContext(; alpha = 0.05)) ==
          rule(:kappa, stated, nothing, nothing, CalibrationContext(; alpha = 0.05))
    @test rule(:kappa, precedence, nothing, nothing, CalibrationContext(; alpha = 0.05)) !=
          rule(:kappa, identity_sigma, nothing, nothing, CalibrationContext(; alpha = 0.05))

    # A TALL `chol` states a covariance matrix all the same, and the `R` factor of its QR
    # factorisation is the square factor of that same matrix. `chol` is checked against the
    # length of `mu` alone, so a carrier can hold one.
    F = randn(StableRNG(4242), 20, 8)
    tall = LowOrderPrior(; X = PRMVT4.X, mu = PRMVT4.mu, sigma = F' * F, chol = F)
    square = LowOrderPrior(; X = PRMVT4.X, mu = PRMVT4.mu, sigma = F' * F)
    @test rule(:kappa, tall, nothing, nothing, CalibrationContext(; alpha = 0.05)) ≈
          rule(:kappa, square, nothing, nothing, CalibrationContext(; alpha = 0.05))

    # The verb that picks the factor is separate, and it returns the prior's own factor
    # untouched when there is one.
    @test PO.whitening_factor(both) === both.chol
    @test PO.whitening_factor(PRMVT4) == cholesky(PRMVT4.sigma).U
    @test size(PO.whitening_factor(tall)) == (8, 8)
end

@testset "Calibration rules: `RadialTailDecay` answers one number for both ends" begin
    rule = RadialTailDecay(; kmin = 5)

    # A distance has no sign, so every key gives the same number. The sibling gives two on
    # the same sample, which is the whole difference between what the two rules say.
    ka = rule(:kappa_a, PRSKEW, nothing, nothing, CalibrationContext(; alpha = 0.01))
    @test rule(:kappa, PRSKEW, nothing, nothing, CalibrationContext(; alpha = 0.01)) == ka
    @test rule(:kappa_b, PRSKEW, nothing, nothing, CalibrationContext(; alpha = 0.01)) == ka
    @test rule(:anything_at_all, PRSKEW, nothing, nothing,
               CalibrationContext(; alpha = 0.01)) == ka
    hill = HillTailDecay(; kmin = 5)
    @test hill(:kappa_a, PRSKEW, nothing, nothing, CalibrationContext(; alpha = 0.01)) !=
          hill(:kappa_b, PRSKEW, nothing, nothing, CalibrationContext(; alpha = 0.01))

    # A head slot that defaults from its tail slot therefore holds the same rule, and that
    # rule answers the head key with the tail's number.
    mctx = CalibrationContext(; alpha = 0.01)
    @test PO.resolve_calibration_slot(rule, :kappa_b, PRSKEW, nothing, nothing, mctx) ==
          PO.resolve_calibration_slot(rule, :kappa_a, PRSKEW, nothing, nothing, mctx)

    # Through a Range measure the two ends read their OWN probabilities, so the two numbers
    # part when the two probabilities differ. The count `k` moves, not the end.
    rg = RelativisticValueatRiskRange(; alpha = 0.01, kappa_a = RadialTailDecay(; kmin = 5),
                                      beta = 0.02, kappa_b = RadialTailDecay(; kmin = 5))
    og = PO.resolve_deferred_quantities(rg, PRSKEW)
    @test og.kappa_a == ka
    @test og.kappa_b == RadialTailDecay(; kmin = 5)(:kappa_b, PRSKEW, nothing, nothing,
                                                    CalibrationContext(; alpha = 0.02))
    @test og.kappa_a != og.kappa_b
    @test 0 < og.kappa_a < 1
    @test 0 < og.kappa_b < 1

    # The two ends of one probability DO agree, which is the statement the rule makes.
    eq = RelativisticValueatRiskRange(; alpha = 0.01, kappa_a = RadialTailDecay(; kmin = 5),
                                      beta = 0.01, kappa_b = RadialTailDecay(; kmin = 5))
    oeq = PO.resolve_deferred_quantities(eq, PRSKEW)
    @test oeq.kappa_a == oeq.kappa_b

    # The context reaches the rule through the resolver and through a direct call alike.
    # The rule is never rebuilt, so the slot holds the object the caller put there.
    rad = RadialTailDecay(; kmin = 7)
    c03 = CalibrationContext(; alpha = 0.03)
    @test PO.resolve_calibration_slot(rad, :kappa, PRSKEW, nothing, nothing, c03) ==
          rad(:kappa, PRSKEW, nothing, nothing, c03)
    @test RelativisticValueatRisk(; kappa = rad).kappa === rad
    @test RelativisticValueatRiskRange(; kappa_a = rad).kappa_b.kmin == 7
end

#=
The refusals of `RadialTailDecay`. Three of them are the sibling's, stated in the units of a
radial series. Three more belong to the covariance matrix, and each names the field it read:
`pr.sigma` when the prior carries no factor, and `pr.chol` when it does. A refusal that names
the field is the answer here, and a numerical guard on a near-singular covariance matrix is
not: the whitening then follows the sample's smallest eigen-direction, which is the
covariance matrix speaking rather than a defect.
=#
# 104 rows that sit exactly at the mean, and 96 rows of ±1 square waves of four periods.
# Every column holds 48 of each sign, so `mu` is exactly zero and 104 of the 200 radial
# distances are exactly zero. The four periods divide 96, so the sign block is orthogonal and
# the covariance matrix is a positive multiple of the identity.
const XZERO = let S = [iseven((t - 1) >> (j - 1)) ? 1.0 : -1.0 for t in 1:96, j in 1:4]
    [zeros(104, 4); S]
end
const PRZERO = prior(EmpiricalPrior(), XZERO)
# A radial series with a tail index at or below one, which is a tail with no finite mean.
const PRMVHEAVY = let rng = StableRNG(13579)
    d = Distributions.MvTDist(0.7, zeros(4), Matrix(1.0I, 4, 4))
    prior(EmpiricalPrior(), permutedims(rand(rng, d, 600)))
end
# A carrier whose covariance matrix is singular, whose factor is wide, and whose factor
# carries a zero on its diagonal. The three are stated by hand, because a fitted prior
# repairs what it can.
const XSING = randn(StableRNG(24680), 200, 4)
const PRSING = LowOrderPrior(; X = XSING, mu = zeros(4),
                             sigma = cov([XSING[:, 1:3] XSING[:, 1]]))
const PRWIDE = LowOrderPrior(; X = XSING, mu = zeros(4), sigma = Matrix(1.0I, 4, 4),
                             chol = randn(StableRNG(1), 2, 4))
const PRZDIAG = LowOrderPrior(; X = XSING, mu = zeros(4), sigma = Matrix(1.0I, 4, 4),
                              chol = UpperTriangular([1.0 0.2 0.3 0.4; 0 1.0 0.1 0.2;
                                                      0 0 0.0 0.5; 0 0 0 1.0]))

@testset "Calibration rules: `RadialTailDecay` refuses an estimate it cannot form" begin
    # A rule whose sibling never arrived cannot form the count, and it says so.
    unbound = RadialTailDecay()
    @test !hasfield(typeof(unbound), :alpha)
    @test unbound.kmin == 30
    @test_throws PO.IsNothingError unbound(:kappa, PR60, nothing, nothing,
                                           CalibrationContext())
    msg = try
        unbound(:kappa, PR60, nothing, nothing, CalibrationContext())
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("CalibrationContext.alpha", msg)
    @test occursin("RadialTailDecay", msg)

    # The floor. The series holds ONE entry per observation, so `PR60` leaves three order
    # statistics at `alpha = 0.05` where the pool of the sibling leaves twelve. The same
    # floor therefore binds harder here, and the message says which count it refused.
    @test_throws DomainError RadialTailDecay()(:kappa, PR60, nothing, nothing,
                                               CalibrationContext(; alpha = 0.05))
    msg = try
        RadialTailDecay()(:kappa, PR60, nothing, nothing,
                          CalibrationContext(; alpha = 0.05))
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("kmin", msg)
    @test occursin("floor at 30", msg)
    @test occursin("= 3` of the 60", msg)

    # A floor the sample clears returns a number, so the refusal is the floor's and not the
    # sample's.
    @test 0 <
          RadialTailDecay(; kmin = 3)(:kappa, PR60, nothing, nothing,
                                      CalibrationContext(; alpha = 0.05)) <
          1

    # A probability that takes the whole sample leaves no `k + 1`-th distance to divide by.
    @test_throws DomainError RadialTailDecay(; kmin = 5)(:kappa, PR60, nothing, nothing,
                                                         CalibrationContext(;
                                                                            alpha = 0.999))
    msg = try
        RadialTailDecay(; kmin = 5)(:kappa, PR60, nothing, nothing,
                                    CalibrationContext(; alpha = 0.999))
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("needs 61 radial distances", msg)
    @test occursin("which is 60", msg)

    # The series holds fewer positive entries than the count asks for, so there is no Hill
    # estimate at all. A sample that sits exactly at its own mean produces it.
    @test PRZERO.mu == zeros(4)
    @test_throws DomainError RadialTailDecay(; kmin = 5)(:kappa, PRZERO, nothing, nothing,
                                                         CalibrationContext(; alpha = 0.6))
    msg = try
        RadialTailDecay(; kmin = 5)(:kappa, PRZERO, nothing, nothing,
                                    CalibrationContext(; alpha = 0.6))
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("121-th largest radial distance is 0.0", msg)
    @test occursin("not positive", msg)

    # An estimate at or below one is a tail with no finite mean, and no admissible κ reads
    # it. The band `(0, 1)` the slot admits IS the condition `â > 1`, so the refusal is the
    # reading rather than a guard bolted onto it.
    @test_throws DomainError RadialTailDecay(; kmin = 5)(:kappa, PRMVHEAVY, nothing,
                                                         nothing,
                                                         CalibrationContext(; alpha = 0.05))
    msg = try
        RadialTailDecay(; kmin = 5)(:kappa, PRMVHEAVY, nothing, nothing,
                                    CalibrationContext(; alpha = 0.05))
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("tail index of 0.8", msg)
    @test occursin("(0, 1)", msg)

    # A covariance matrix that is not positive definite states no whitening, and the message
    # names the field it was read off.
    @test_throws DomainError RadialTailDecay(; kmin = 5)(:kappa, PRSING, nothing, nothing,
                                                         CalibrationContext(; alpha = 0.05))
    msg = try
        PO.whitening_factor(PRSING)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("`pr.sigma`", msg)
    @test occursin("not positive definite", msg)

    # A wide factor states a singular covariance matrix. Dropping to `pr.sigma` would state
    # something the prior does not, because `chol` takes precedence at every consumer.
    @test_throws DimensionMismatch RadialTailDecay(; kmin = 5)(:kappa, PRWIDE, nothing,
                                                               nothing,
                                                               CalibrationContext(;
                                                                                  alpha = 0.05))
    msg = try
        PO.whitening_factor(PRWIDE)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("`pr.chol` is 2 × 4", msg)
    @test occursin("takes precedence", msg)

    # A zero on the diagonal of a triangular factor is a rank statement, so it is refused
    # rather than solved against.
    @test_throws DomainError RadialTailDecay(; kmin = 5)(:kappa, PRZDIAG, nothing, nothing,
                                                         CalibrationContext(; alpha = 0.05))
    msg = try
        PO.whitening_factor(PRZDIAG)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("Entry 3 of the diagonal", msg)
    @test occursin("`pr.chol`", msg)
end

#=
The **series** a rule reads. A drawdown measure prices the drawdown series of the portfolio,
and a rule forms no portfolio, so it reads the drawdown series of each COLUMN in place of the
column. Nothing else in either reading moves. `CalibrationContext.series` carries the marker
from the owner to the rule, and it is the only place the marker is ever written, because the
quantity belongs to the measure and a rule cannot know which measure it reached.

The fixture is a drifted Student-t draw. The drift matters to what a drawdown series says: a
drawdown is a running functional, so a sample whose drift is strong enough for the drawdown
process to settle reads heavier than its own returns, and a weakly drifted one reads the
range of its path over the record. Both are readings of the series the measure prices, and
the tests below state the MECHANISM rather than the direction the two readings part in.
=#
const XDD = 0.0005 .+ 0.01 .* rand(StableRNG(515151), Distributions.TDist(4), 3000, 6)
const PRDD = prior(EmpiricalPrior(), XDD)

@testset "Calibration series: the family, the trait and the two series" begin
    # Two markers name a drawdown and one names the returns, so a rule asks the family
    # rather than the marker whether it holds a path functional.
    @test ReturnsSeries <: PO.AbstractCalibrationSeries
    @test AbsoluteDrawdownSeries <: PO.AbstractDrawdownSeries
    @test RelativeDrawdownSeries <: PO.AbstractDrawdownSeries
    @test PO.AbstractDrawdownSeries <: PO.AbstractCalibrationSeries
    @test !(ReturnsSeries <: PO.AbstractDrawdownSeries)

    # The markers are caller-facing, because a caller states one to run a rule by hand.
    exported = names(PortfolioOptimisers)
    @test :ReturnsSeries in exported
    @test :AbsoluteDrawdownSeries in exported
    @test :RelativeDrawdownSeries in exported

    # The trait. The default is the returns, so a type that prices the return distribution
    # writes no method, and the two drawdown measures do.
    @test isa(PO.calibration_series(0.3), ReturnsSeries)
    @test isa(PO.calibration_series(RelativisticValueatRisk()), ReturnsSeries)
    @test isa(PO.calibration_series(RelativisticValueatRiskRange()), ReturnsSeries)
    @test isa(PO.calibration_series(RelativisticDrawdownatRisk()), AbsoluteDrawdownSeries)
    @test isa(PO.calibration_series(RelativeRelativisticDrawdownatRisk()),
              RelativeDrawdownSeries)

    # A DRAWDOWN SERIES HOLDS ONE ENTRY PER OBSERVATION, so the change of series takes no
    # observation away and the count a rule forms on `X` is the count it reads.
    col = view(XDD, :, 2)
    @test PO.calibration_series_vec(ReturnsSeries(), col) === col
    @test PO.calibration_series_vec(AbsoluteDrawdownSeries(), col) ==
          PO.absolute_drawdown_vec(col)
    @test PO.calibration_series_vec(RelativeDrawdownSeries(), col) ==
          PO.relative_drawdown_vec(col)
    for s in (AbsoluteDrawdownSeries(), RelativeDrawdownSeries())
        @test length(PO.calibration_series_vec(s, col)) == length(col)
        @test size(PO.calibration_series_matrix(s, XDD)) == size(XDD)
    end

    # The matrix is the array builder of its own convention, and the returns matrix is
    # passed through.
    @test PO.calibration_series_matrix(ReturnsSeries(), XDD) === XDD
    D = PO.calibration_series_matrix(AbsoluteDrawdownSeries(), XDD)
    @test D == PO.absolute_drawdown_arr(XDD; dims = 1)
    @test PO.calibration_series_matrix(RelativeDrawdownSeries(), XDD) ==
          PO.relative_drawdown_arr(XDD; dims = 1)

    # The matrix reading and the vector reading state one definition of a drawdown, so the
    # column of the matrix is the vector of the column.
    @test all(j -> view(D, :, j) == PO.absolute_drawdown_vec(view(XDD, :, j)), axes(XDD, 2))
    @test all(j -> view(PO.calibration_series_matrix(RelativeDrawdownSeries(), XDD), :,
                        j) == PO.relative_drawdown_vec(view(XDD, :, j)), axes(XDD, 2))

    # A drawdown series is non-positive and the returns are not, which is the whole reason
    # the two readings differ.
    @test all(<=(0), D)
    @test any(>(0), XDD)
end

@testset "Calibration series: the context carries the owner's series" begin
    rule = HillTailDecay(; kmin = 12)
    abs_ctx = CalibrationContext(; alpha = 0.02, series = AbsoluteDrawdownSeries())
    rel_ctx = CalibrationContext(; alpha = 0.02, series = RelativeDrawdownSeries())
    ret_ctx = CalibrationContext(; alpha = 0.02)

    # The marker reaches the rule, and through both rules of the family. The rule is not
    # rebuilt, so the occupant of the slot is the object the caller put there.
    @test PO.resolve_calibration_slot(rule, :kappa, PRDD, nothing, nothing, abs_ctx) ==
          rule(:kappa, PRDD, nothing, nothing, abs_ctx)
    @test PO.resolve_calibration_slot(rule, :kappa, PRDD, nothing, nothing, rel_ctx) ==
          rule(:kappa, PRDD, nothing, nothing, rel_ctx)
    @test isa(abs_ctx.series, AbsoluteDrawdownSeries)
    @test isa(rel_ctx.series, RelativeDrawdownSeries)

    # The three markers name three different quantities of one sample, so a rule that reads
    # the shape of a series answers three different numbers.
    @test rule(:kappa, PRDD, nothing, nothing, abs_ctx) !=
          rule(:kappa, PRDD, nothing, nothing, ret_ctx)
    @test rule(:kappa, PRDD, nothing, nothing, rel_ctx) !=
          rule(:kappa, PRDD, nothing, nothing, ret_ctx)

    # THE OWNER'S SERIES IS THE ONLY SERIES. No rule carries a marker of its own, so there
    # is nothing for the owner's answer to overwrite and no precedence to state. This is
    # the reading the norm order takes too, and the significance level with it.
    @test !hasfield(HillTailDecay, :series)
    @test !hasfield(RadialTailDecay, :series)
    @test !hasfield(EffectiveAssetFloor, :p)

    # The default is `ReturnsSeries`, which is what `calibration_series` answers for every
    # owner that names no other.
    @test isa(CalibrationContext().series, ReturnsSeries)
    @test rule(:kappa, PRDD, nothing, nothing,
               CalibrationContext(; alpha = 0.02, series = ReturnsSeries())) ==
          rule(:kappa, PRDD, nothing, nothing, ret_ctx)

    # Everything that reads no series ignores the field: a stated number, a rule of another
    # reading, a plain function, and the rule each of them stands in.
    @test PO.resolve_calibration_slot(0.3, :kappa, PRDD, nothing, nothing, abs_ctx) === 0.3
    @test PO.resolve_calibration_slot(nothing, :kappa, PRDD, nothing, nothing, abs_ctx) ===
          nothing
    # The band moves with the sample, and PRDD is long, so the target is one that lies
    # inside the band at this length.
    budget = EntropyBudget(; target = -10.0)
    bctx = CalibrationContext(; alpha = 0.05, series = AbsoluteDrawdownSeries())
    @test budget(:kappa, PRDD, nothing, nothing, bctx) ==
          budget(:kappa, PRDD, nothing, nothing, CalibrationContext(; alpha = 0.05))
    probe = (key, pr, w, slv, ctx) -> 0.4
    @test PO.resolve_calibration_slot(probe, :kappa, PRDD, nothing, nothing, abs_ctx) == 0.4
    @test PO.resolve_calibration_slot(RateSignificance(), :alpha, PRDD, nothing, nothing,
                                      abs_ctx) ==
          PO.resolve_calibration_slot(RateSignificance(), :alpha, PRDD, nothing, nothing)
end

@testset "Calibration series: `HillTailDecay` pools the drawdowns" begin
    alpha = 0.05
    k = ceil(Int, alpha * length(XDD))

    # The count does not move with the series: the pool holds `T * N` entries under every
    # marker, because a drawdown series holds one entry per observation.
    @test k == ceil(Int, alpha * size(XDD, 1) * size(XDD, 2))

    # The reading is the sibling's, over the drawdown series of each column. The pool is
    # built here by hand, so the whole chain is checked and not merely its two ends.
    function hill_by_hand(f, X, k)
        pool = Float64[]
        for j in axes(X, 2)
            s = f(view(X, :, j))
            append!(pool, (s .- mean(s)) ./ std(s))
        end
        u = sort(pool)
        return k / sum(log(u[i] / u[k + 1]) for i in 1:k)
    end
    for (s, f) in ((AbsoluteDrawdownSeries(), PO.absolute_drawdown_vec),
                   (RelativeDrawdownSeries(), PO.relative_drawdown_vec), (ReturnsSeries(), identity))
        rule = HillTailDecay(; kmin = 20)
        @test rule(:kappa, PRDD, nothing, nothing,
                   CalibrationContext(; alpha = alpha, series = s)) ≈
              inv(hill_by_hand(f, XDD, k))
    end

    # The three markers name three series of one sample, so they answer three numbers.
    ka = HillTailDecay(; kmin = 20)(:kappa, PRDD, nothing, nothing,
                                    CalibrationContext(; alpha = alpha,
                                                       series = ReturnsSeries()))
    kb = HillTailDecay(; kmin = 20)(:kappa, PRDD, nothing, nothing,
                                    CalibrationContext(; alpha = alpha,
                                                       series = AbsoluteDrawdownSeries()))
    kc = HillTailDecay(; kmin = 20)(:kappa, PRDD, nothing, nothing,
                                    CalibrationContext(; alpha = alpha,
                                                       series = RelativeDrawdownSeries()))
    @test ka != kb
    @test kb != kc
    @test all(k -> 0 < k < 1, (ka, kb, kc))

    # A DRAWDOWN SERIES HAS ONE END, so the head key names nothing on it and is refused. No
    # drawdown Range measure ships, so only a caller who runs the rule by hand reaches this.
    @test PO.series_end_sign(ReturnsSeries(), :kappa_b) == 1
    @test PO.series_end_sign(ReturnsSeries(), :kappa) == -1
    @test PO.series_end_sign(AbsoluteDrawdownSeries(), :kappa) == -1
    @test PO.series_end_sign(RelativeDrawdownSeries(), :kappa_a) == -1
    @test_throws ArgumentError PO.series_end_sign(AbsoluteDrawdownSeries(), :kappa_b)
    @test_throws ArgumentError HillTailDecay(; kmin = 20)(:kappa_b, PRDD, nothing, nothing,
                                                          CalibrationContext(;
                                                                             alpha = alpha,
                                                                             series = AbsoluteDrawdownSeries()))
    msg = try
        PO.series_end_sign(AbsoluteDrawdownSeries(), :kappa_b)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("has no gain end", msg)
    @test occursin("AbsoluteDrawdownSeries", msg)
end

@testset "Calibration series: `RadialTailDecay` whitens the drawdowns" begin
    alpha = 0.05
    rule = RadialTailDecay(; kmin = 20)

    # The reading is the returns reading over the drawdown sample: the rows of that sample
    # are centred on ITS column means and whitened by the factor of ITS covariance matrix,
    # because a prior result states no drawdown moment.
    D = PO.calibration_series_matrix(AbsoluteDrawdownSeries(), XDD)
    mu = vec(mean(D; dims = 1))
    U = cholesky(cov(D)).U
    d = [norm(transpose(U) \ (D[t, :] .- mu)) for t in axes(D, 1)]
    k = ceil(Int, alpha * length(d))
    u = sort(d; rev = true)
    ahat = k / sum(log(u[i] / u[k + 1]) for i in 1:k)
    @test rule(:kappa, PRDD, nothing, nothing,
               CalibrationContext(; alpha = alpha, series = AbsoluteDrawdownSeries())) ≈
          inv(ahat)
    @test 0 <
          rule(:kappa, PRDD, nothing, nothing,
               CalibrationContext(; alpha = alpha, series = AbsoluteDrawdownSeries())) <
          1

    # The verb that picks the three inputs is separate, and it states which reading is which.
    Y, m, F = PO.radial_series_inputs(AbsoluteDrawdownSeries(), PRDD)
    @test Y == D
    @test m == mu
    @test F == U
    Yr, mr, Fr = PO.radial_series_inputs(ReturnsSeries(), PRDD)
    @test Yr === PRDD.X
    @test mr === PRDD.mu
    @test Fr == PO.whitening_factor(PRDD)

    # `pr.sigma` IS THE COVARIANCE MATRIX OF THE RETURNS, so it reaches nothing under a
    # drawdown marker: a change of it leaves the drawdown reading exactly where it stands.
    scaled = LowOrderPrior(; X = PRDD.X, mu = PRDD.mu, sigma = 100 .* PRDD.sigma)
    @test rule(:kappa, scaled, nothing, nothing,
               CalibrationContext(; alpha = alpha, series = AbsoluteDrawdownSeries())) ==
          rule(:kappa, PRDD, nothing, nothing,
               CalibrationContext(; alpha = alpha, series = AbsoluteDrawdownSeries()))

    # The returns reading DOES read `pr.sigma`, but only through the SHAPE of the whitening.
    # A hundredfold scaling of the covariance matrix scales every radial distance alike, and
    # a Hill estimate reads the ratios `u[i] / u[k + 1]` alone, so the scale cancels and the
    # returns reading holds to rounding. Do not tighten this to an equality: the two sides
    # take different roundings through the Cholesky factorisation, and CI has returned both
    # the same last bit and a different one.
    ret = RadialTailDecay(; kmin = 20)
    @test ret(:kappa, scaled, nothing, nothing, CalibrationContext(; alpha = alpha)) ≈
          ret(:kappa, PRDD, nothing, nothing, CalibrationContext(; alpha = alpha))

    # A per-asset stretch is not a uniform scaling: it moves the relative dispersion of the
    # columns, so the whitening reorders the distances and the returns reading moves. This
    # is the line that separates the returns reading from the drawdown reading.
    #
    # Build the congruence with a BROADCAST rather than with `Diagonal(d) * sigma * Diagonal(d)`.
    # The matrix product rounds the `(i, j)` entry as `(d[i] * sigma[i, j]) * d[j]` and the
    # `(j, i)` entry as `(d[j] * sigma[j, i]) * d[i]`, which is the same three factors in a
    # different order, so the product is not exactly symmetric on every host. `cholesky`
    # then reports `info = -1`, which is a matrix that is not Hermitian rather than a failed
    # pivot, and `whitening_factor` refuses it. The broadcast multiplies each entry by
    # `d[i] * d[j]` once, and a scalar product commutes exactly, so the stretch is symmetric
    # on every host.
    d6 = collect(1.0:6.0)
    stretched = LowOrderPrior(; X = PRDD.X, mu = PRDD.mu,
                              sigma = PRDD.sigma .* (d6 * transpose(d6)))
    @test issymmetric(stretched.sigma)
    @test ret(:kappa, stretched, nothing, nothing, CalibrationContext(; alpha = alpha)) !=
          ret(:kappa, PRDD, nothing, nothing, CalibrationContext(; alpha = alpha))
    @test rule(:kappa, stretched, nothing, nothing,
               CalibrationContext(; alpha = alpha, series = AbsoluteDrawdownSeries())) ==
          rule(:kappa, PRDD, nothing, nothing,
               CalibrationContext(; alpha = alpha, series = AbsoluteDrawdownSeries()))

    # The distance still has no sign, so the rule answers one number for every key. The
    # marker moves the sample the distance is taken over, and not what a distance is.
    @test rule(:kappa_a, PRDD, nothing, nothing,
               CalibrationContext(; alpha = alpha, series = AbsoluteDrawdownSeries())) ==
          rule(:kappa, PRDD, nothing, nothing,
               CalibrationContext(; alpha = alpha, series = AbsoluteDrawdownSeries()))
    @test rule(:kappa_b, PRDD, nothing, nothing,
               CalibrationContext(; alpha = alpha, series = AbsoluteDrawdownSeries())) ==
          rule(:kappa, PRDD, nothing, nothing,
               CalibrationContext(; alpha = alpha, series = AbsoluteDrawdownSeries()))

    # A drawdown sample states its own covariance matrix, so a sample that states a singular
    # one is refused there rather than in `whitening_factor`. Two columns with one path
    # between them are what produce it.
    twice = [XDD[:, 1] XDD[:, 1]]
    prtwice = LowOrderPrior(; X = twice, mu = vec(mean(twice; dims = 1)),
                            sigma = Matrix(1.0I, 2, 2))
    @test_throws DomainError rule(:kappa, prtwice, nothing, nothing,
                                  CalibrationContext(; alpha = alpha,
                                                     series = AbsoluteDrawdownSeries()))
    msg = try
        PO.radial_series_inputs(AbsoluteDrawdownSeries(), prtwice)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("AbsoluteDrawdownSeries", msg)
    @test occursin("not positive definite", msg)
    # The same carrier passes the returns reading, because `pr.sigma` states a whitening.
    @test 0 <
          ret(:kappa, prtwice, nothing, nothing, CalibrationContext(; alpha = alpha)) <
          1
end

@testset "Calibration series: a measure hands its own series over" begin
    # The four owners resolve one key, `:kappa`, and three of them price three series. The
    # marker is the only thing that separates the three readings.
    hill = HillTailDecay(; kmin = 20)
    rlvar = PO.resolve_deferred_quantities(RelativisticValueatRisk(; alpha = 0.05,
                                                                   kappa = hill), PRDD)
    rldar = PO.resolve_deferred_quantities(RelativisticDrawdownatRisk(; alpha = 0.05,
                                                                      kappa = hill), PRDD)
    rrldar = PO.resolve_deferred_quantities(RelativeRelativisticDrawdownatRisk(;
                                                                               alpha = 0.05,
                                                                               kappa = hill),
                                            PRDD)
    @test rlvar.kappa == HillTailDecay(; kmin = 20)(:kappa, PRDD, nothing, nothing,
                                                    CalibrationContext(; alpha = 0.05,
                                                                       series = ReturnsSeries()))
    @test rldar.kappa == HillTailDecay(; kmin = 20)(:kappa, PRDD, nothing, nothing,
                                                    CalibrationContext(; alpha = 0.05,
                                                                       series = AbsoluteDrawdownSeries()))
    @test rrldar.kappa == HillTailDecay(; kmin = 20)(:kappa, PRDD, nothing, nothing,
                                                     CalibrationContext(; alpha = 0.05,
                                                                        series = RelativeDrawdownSeries()))
    @test rlvar.kappa != rldar.kappa
    @test rldar.kappa != rrldar.kappa

    # The radial rule travels the same way, and its drawdown reading is its own.
    radial = RadialTailDecay(; kmin = 20)
    rdar = PO.resolve_deferred_quantities(RelativisticDrawdownatRisk(; alpha = 0.05,
                                                                     kappa = radial), PRDD)
    @test rdar.kappa == RadialTailDecay(; kmin = 20)(:kappa, PRDD, nothing, nothing,
                                                     CalibrationContext(; alpha = 0.05,
                                                                        series = AbsoluteDrawdownSeries()))
    @test rdar.kappa != rldar.kappa

    # NO MARKER CAN BE STATED ON THE RULE, so the measure that resolves it names the only
    # one there is and no caller can put a second one anywhere.
    @test !hasfield(HillTailDecay, :series)
    o = PO.resolve_deferred_quantities(RelativisticValueatRisk(; alpha = 0.05,
                                                               kappa = hill), PRDD)
    @test o.kappa == rlvar.kappa

    # Both ends of a Range measure price ONE series, where each end prices its own
    # probability. So the marker reaching the two `kappa` slots is the same marker.
    rg = PO.resolve_deferred_quantities(RelativisticValueatRiskRange(; alpha = 0.05,
                                                                     kappa_a = hill,
                                                                     beta = 0.05,
                                                                     kappa_b = hill), PRDD)
    @test rg.kappa_a == rlvar.kappa
    @test 0 < rg.kappa_b < 1
end

@testset "Calibration rules: the two Hill verbs read the k largest (#629)" begin
    # `partialsort!(v, k)` places the value at index `k` in the position it holds in a fully
    # sorted vector, and it promises nothing about the rest of the vector. Julia switches
    # selection strategy above a size threshold, and above it `v[1:k - 1]` is not the
    # `k - 1` smallest values. Both verbs sum `log(v[i] / v[k + 1])` over that prefix, so
    # each returned a number that is not the estimate once `k` cleared the threshold.
    #
    # The counts below straddle the threshold on purpose. `hill_tail_index` pools `T * N`
    # values, so a pool of 4000 reaches it at a few hundred; `radial_tail_index` reads one
    # distance per observation, so a series of 1000 reaches it at 117. The samples the rest
    # of this file reads are 60 to 120 rows long, where no count goes wrong at all, so a
    # short sample would have proved nothing. Each pair below is one count over the
    # threshold and one control under it.
    #
    # The reference sorts the whole series, so it rests on no prefix.
    X629 = randn(StableRNG(987654321), 1000, 4)
    PR629 = prior(EmpiricalPrior(), X629)

    function hill_reference(v, k)
        vs = sort(v)
        return k / sum(i -> log(vs[i] / vs[k + 1]), 1:k)
    end

    # `hill_tail_index` pools the standardised columns, and the sign puts the end it prices
    # in the lower tail of the pool.
    function hill_pool(X, s)
        T, N = size(X)
        pool = Vector{Float64}(undef, T * N)
        for j in axes(X, 2)
            col = view(X, :, j)
            m = mean(col)
            sd = std(col; mean = m)
            pool[((j - 1) * T + 1):(j * T)] .= -s .* (col .- m) ./ sd
        end
        return pool
    end

    for s in (1, -1), k in (600, 100)
        @test isapprox(PO.hill_tail_index(ReturnsSeries(), X629, s, k),
                       hill_reference(hill_pool(X629, s), k))
    end

    # `radial_tail_index` whitens the sample and negates the norm of each row, so the same
    # reading serves it.
    Y629, mu629, U629 = PO.radial_series_inputs(ReturnsSeries(), PR629)
    Z629 = transpose(U629) \ transpose(Y629 .- transpose(mu629))
    d629 = [-norm(view(Z629, :, t)) for t in axes(Y629, 1)]
    for k in (200, 50)
        @test isapprox(PO.radial_tail_index(Y629, mu629, U629, k), hill_reference(d629, k))
    end

    # The two rules reach the verbs through a count of their own, and the count clears the
    # threshold at a level a slot admits. `alpha = 0.15` reads 600 of the 4000 pooled values
    # and 150 of the 1000 distances.
    skappa = PO.series_end_sign(ReturnsSeries(), :kappa)
    @test isapprox(HillTailDecay(; kmin = 30)(:kappa, PR629, nothing, nothing,
                                              CalibrationContext(; alpha = 0.15)),
                   inv(hill_reference(hill_pool(X629, skappa), 600)))
    @test isapprox(RadialTailDecay(; kmin = 30)(:kappa, PR629, nothing, nothing,
                                                CalibrationContext(; alpha = 0.15)),
                   inv(hill_reference(d629, 150)))
end

@testset "Calibration rules: a name states the method, and the quantity where the word is taken" begin
    #=
    An ergonomics pass read the eleven names as one family and found a convention in four
    of them: `ConcentrationRadius`, `RateRadius`, `DimensionalRateRadius` and
    `DualNormRadius` each end in the quantity the rule computes, and six of the remaining
    seven do not. ADR 0015 is the naming Authority and says the reading is the wrong one:
    a name is the bare concept word, and a suffix is added only to earn back clarity the
    bare word would lose, never as a category marker.

    So a rule is named for the METHOD it runs, and carries the quantity only where the
    bare method word is already claimed. ADR 0095 holds the list and the reason for each
    entry, and this census gates it. A rule whose name ends in the word for a quantity of
    this channel and stands outside the list reds the census, and so does a list entry
    that no rule carries.
    =#
    earns_the_quantity = Dict(:RateSignificance => "`Rate` is one method over two quantities.",
                              :RateRadius => "The same collision, read from the other end.",
                              :DimensionalRateRadius => "The same `Rate` stem under a prefix.",
                              :ConcentrationRadius => "`Concentration` names the weight concentration of a portfolio.",
                              :DualNormRadius => "`DualNorm` names a mathematical object.",
                              :SteinShrinkage => "`BayesStein` and `JamesStein` name algorithms of `ShrunkExpectedReturns`.")

    # The words the five families name their quantities with. A rule that ends in one of
    # them states the quantity, whatever method word stands in front of it.
    quantity_words = ("Significance", "Deformation", "Radius", "TailWeight", "NormCeiling",
                      "Shrinkage", "ForecastScale")

    # `traverse_concrete_subtypes` walks the whole root, and `parentmodule` drops the probe
    # rules that `test_09f_calibration_slot.jl` defines, which reach this census whenever
    # the two files share a process.
    rule_types = filter(T -> parentmodule(T) === PortfolioOptimisers,
                        PO.traverse_concrete_subtypes(PO.AbstractCalibrationAlgorithm))
    rule_names = Set(nameof.(rule_types))
    @test ("the rules this census reads", length(rule_names)) ==
          ("the rules this census reads", 14)

    stated = Set(n for n in rule_names if any(w -> endswith(string(n), w), quantity_words))
    named = Set(keys(earns_the_quantity))

    # Both directions are named, so a failure says which name moved and which way. A rule
    # added with a quantity in its name states in ADR 0095 what claims the bare word, and
    # adds the row here.
    unearned = sort!(collect(setdiff(stated, named)))
    @test ("names that state the quantity and earn it nowhere", unearned) ==
          ("names that state the quantity and earn it nowhere", Symbol[])
    stale = sort!(collect(setdiff(named, stated)))
    @test ("list entries that no rule carries", stale) ==
          ("list entries that no rule carries", Symbol[])

    # The other eight name a method and stop there, which is the default the rule states.
    @test ("rules named for the method alone",
           sort!(collect(setdiff(rule_names, stated)))) ==
          ("rules named for the method alone",
           [:EffectiveAssetFloor, :EntropyBudget, :ForecastCalibrationSlope, :HillTailDecay,
            :PrecisionBlend, :RadialTailDecay, :ScenarioCount, :TailTermParity])
end

@testset "Calibration rules: twelve construct bare, and two name the quantity they refuse" begin
    #=
    A caller who meets nine rules that construct bare learns that a rule constructs bare.
    Two cannot, because the keyword each takes is the whole content of the rule, and a
    mandatory keyword then fails with an `UndefKeywordError` that names the keyword and
    nothing else -- no quantity, no reason, no value to start from. The keyword of each of
    the two stands at `nothing` instead, and the `@argcheck` that refuses it says all
    three. ADR 0095 owns the rule.

    A zero-argument method carrying the message was written first and does not work: Julia
    generates that method for a keyword constructor itself, and overwriting it is an ERROR
    during precompilation. The sentinel keeps the type refusal, because `Option{<:Number}`
    admits a number and `nothing` and nothing else.
    =#
    for R in
        (RateSignificance, HillTailDecay, RadialTailDecay, ConcentrationRadius, RateRadius,
         DimensionalRateRadius, DualNormRadius, TailTermParity, EffectiveAssetFloor,
         PrecisionBlend, SteinShrinkage, ForecastCalibrationSlope)
        @test isa(R(), R)
    end

    @test_throws ArgumentError ScenarioCount()
    @test_throws ArgumentError EntropyBudget()

    function refusal_text(f)
        return try
            f()
            ""
        catch e
            sprint(showerror, e)
        end
    end

    # The message names the quantity, the reason there is no default, a value to start
    # from, and the rule of the same family that does construct bare.
    scmsg = refusal_text(ScenarioCount)
    @test occursin("number of observations the tail is to hold", scmsg)
    @test occursin("no count suits every sample", scmsg)
    @test occursin("ScenarioCount(; n = 15)", scmsg)
    @test occursin("RateSignificance", scmsg)

    ebmsg = refusal_text(EntropyBudget)
    @test occursin("value of the Kaniadakis logarithm", ebmsg)
    @test occursin("moves with the sample", ebmsg)
    @test occursin("EntropyBudget(; target = -1.3)", ebmsg)
    @test occursin("HillTailDecay", ebmsg)

    # The sentinel widens the keyword to `Option{<:Number}` and nothing further, so a value
    # that is neither a number nor `nothing` is refused where it always was.
    @test ScenarioCount(; n = 15).n == 15
    @test EntropyBudget(; target = -1.3).target == -1.3
    @test_throws TypeError ScenarioCount(; n = "a")
    @test_throws TypeError EntropyBudget(; target = "a")

    # A caller who states the sentinel by hand meets the same refusal, so the keyword
    # route never puts it in the field.
    @test_throws ArgumentError ScenarioCount(; n = nothing)
    @test_throws ArgumentError EntropyBudget(; target = nothing)

    # The POSITIONAL route is not refused, and that is not this decision's doing: the
    # constructor `ConcreteStructs.@concrete` emits is broader than the hand-written inner
    # one on every type in the library, so it bypasses the inner `::Number` as it bypasses
    # every other bound. Issue #264 carries the hole and ADR 0095's Consequences record it.
    @test ScenarioCount(15).n == 15
    @test EntropyBudget(-1.3).target == -1.3
end

@testset "Calibration rules: a Scenario Cap states its count in `ens`, and the count readers take it (#1015)" begin
    # ADR 0138. `EmpiricalPrior(; max_scenarios = w)` fits the moments over every observation
    # and carries the last `w` rows, so a count read off the shape would price `w`
    # observations for moments fitted over `t`. The result states `t` in `ens` exactly when
    # the cap cuts, and `effective_sample_size` reads it before the shape.
    X1015 = randn(StableRNG(1015), 1250, 5) ./ 100
    pu = prior(EmpiricalPrior(), X1015)
    pc = prior(EmpiricalPrior(; max_scenarios = 250), X1015)
    ctx = CalibrationContext()

    @testset "The cap states the count, in batch and after t folds, and only when it cuts" begin
        @test isnothing(pu.ens)
        @test pc.ens == 1250
        @test size(pc.X, 1) == 250
        # A cap at or above the number of observations cuts nothing and states nothing.
        @test isnothing(prior(EmpiricalPrior(; max_scenarios = 1250), X1015).ens)
        @test isnothing(prior(EmpiricalPrior(; max_scenarios = 5000), X1015).ens)
        # The folded read-out states the same count off the same rows.
        o = EmpiricalPrior(; max_scenarios = 250)
        for i in 1:1250
            o = partial_fit!(o, X1015[i, :])
        end
        pf = prior(o)
        @test pf.ens == 1250
        @test size(pf.X, 1) == 250
        @test pf.mu ≈ pc.mu
        # The horizon arm states it too.
        @test prior(EmpiricalPrior(; horizon = 21, max_scenarios = 250), X1015).ens == 1250
        @test isnothing(prior(EmpiricalPrior(; horizon = 21), X1015).ens)
        # The verb itself: `nothing` when nothing is cut.
        @test isnothing(PO.scenario_ens(nothing, X1015))
        @test isnothing(PO.scenario_ens(1250, X1015))
        @test PO.scenario_ens(1249, X1015) == 1250
    end

    @testset "The count readers price t, not the rows carried" begin
        # The one verb, and its three arms in order.
        @test PO.effective_sample_size(pu, nothing) == 1250
        @test PO.effective_sample_size(pc, nothing) == 1250
        @test PO.effective_sample_size(pc, StatsBase.pweights(ones(250))) == 250
        # `ScenarioCount` and the two rate rules read the count behind the moments.
        @test ScenarioCount(; n = 25)(:alpha, pc, nothing, nothing, ctx) == 25 / 1250
        @test RateSignificance()(:alpha, pc, nothing, nothing, ctx) ==
              RateSignificance()(:alpha, pu, nothing, nothing, ctx)
        @test RateRadius()(:k, pc, nothing, nothing, ctx) ==
              RateRadius()(:k, pu, nothing, nothing, ctx)
        actx = CalibrationContext(; alpha = 0.05)
        @test EntropyBudget(; target = -10.0)(:kappa, pc, nothing, nothing, actx) ==
              EntropyBudget(; target = -10.0)(:kappa, pu, nothing, nothing, actx)
        # The normal set's `T` reads `ens`, so its box over the capped prior is the box
        # over the uncapped one: the ellipsoid is not `√5 ≈ 2.24` times too wide.
        ue = NormalUncertaintySet(; pe = nothing)
        @test PO.choose_scaling_parameter(ue, pc) == 1250
        @test mu_ucs(ue, pc).ub == mu_ucs(ue, pu).ub
        # The bootstrap can resample nothing but the rows carried, so it reads the cap.
        ab = ARCHUncertaintySet(; pe = nothing, seed = 3, n_sim = 20)
        @test mu_ucs(ab, pc).ub != mu_ucs(ab, pu).ub
    end

    @testset "An `ens` bound to a weighting is a diagnostic of it, and is not the stated count" begin
        # An entropy-pooling prior writes `exp(entropy(w))` beside `w`. A rule handed the
        # weights reads Kish's count of them, as before; a rule that reads the raw row
        # count still does, because a diagnostic of weights it ignores is not a count it
        # reads. Neither released number moves.
        w = StatsBase.pweights(vcat(fill(0.5, 60), fill(1.5, 60)))
        pw = LowOrderPrior(; X = pu.X[1:120, :], mu = pu.mu, sigma = pu.sigma, w = w,
                           ens = exp(StatsBase.entropy(w / sum(w))))
        kish = sum(w)^2 / sum(abs2, w)
        @test PO.effective_sample_size(pw, pw.w) == kish
        @test PO.effective_sample_size(pw, nothing) == 120
        @test ScenarioCount(; n = 25)(:alpha, pw, pw.w, nothing, ctx) == 25 / kish
        @test RateSignificance()(:alpha, pw, pw.w, nothing, ctx) ==
              RateSignificance()(:alpha, pu, nothing, nothing, ctx) * sqrt(1250 / 120)
        # The normal set reads `pr.ens` as it always did.
        @test PO.choose_scaling_parameter(NormalUncertaintySet(; pe = nothing), pw) ==
              pw.ens
    end
end

#=
An **Ambiguity Radius** is the radius of the ball of probability measures a distributionally
robust model prices, and an **Esfahani-Kuhn tail weight** is the weight of the tail term of
the loss it minimises. Twelve slots across six types hold one or the other, and each of them
takes a **Calibration Rule** in place of the number.

`test_09f_calibration_slot.jl` covers the mechanism and `test_09f_calibration_slot.jl`
covers the rules of the two older families. This file covers the ambiguity families, the
four radius rules and the one tail-weight rule they ship, and the twelve slots.

The two families take the same shape as the two older ones: an abstract family under
`AbstractCalibrationAlgorithm`, a `Func_` bound for the `alg` field that admits a plain
`Function`, one rule type, and a `Num_` bound that pairs the rule with `Number`. A rule is
run by CALLING it, so a callable struct and a plain function are the same thing to the
resolver.

A radius names no end of the distribution, so each family carries ONE rule rather than the
two that a tail probability carries, and no ambiguity slot defaults from another.

Issue #584 widens the slots. Issue #311 settled the design, and its correction binds: `r` is
the radius and `l` is the tail weight, not the other way round.
=#
using Clarabel, JuMP, Distributions, InteractiveUtils
const PO = PortfolioOptimisers

const RNG_H = StableRNG(246813579)
const X60_H = randn(RNG_H, 60, 4)
const PR60_H = prior(EmpiricalPrior(), X60_H)
const X120_H = randn(RNG_H, 120, 4)
const PR120_H = prior(EmpiricalPrior(), X120_H)
const WTS = pweights(range(; start = 1, stop = 2, length = 60))
# A two-asset universe is the corner where `DimensionalRateRadius` reduces to the
# square-root rate, and a twenty-asset one at two record lengths is where it parts
# company with `RateRadius`.
const X60_2 = X60_H[:, 1:2]
const PR60_2 = prior(EmpiricalPrior(), X60_2)
const X125_20 = randn(RNG_H, 125, 20)
const PR125_20 = prior(EmpiricalPrior(), X125_20)
const X250_20 = randn(RNG_H, 250, 20)
const PR250_20 = prior(EmpiricalPrior(), X250_20)

# `TailTermParity` reads a mean against a tail, so it needs a sample that carries a real
# expected return. `XDAY` is ten years of daily returns, and `XMON` sums its rows in blocks
# of 21. A mean scales with the horizon and a dispersion scales with its square root, so
# the two samples carry two DIFFERENT mean-to-CVaR ratios. That is the defect the rule
# fixes: one stated `l` is two trade-offs across the pair.
const XDAY = 0.0005 .+ 0.01 * randn(RNG_H, 2520, 4)
const PRDAY = prior(EmpiricalPrior(), XDAY)
const XMON = reduce(vcat,
                    [sum(view(XDAY, ((k - 1) * 21 + 1):(k * 21), :); dims = 1)
                     for k in 1:120])
const PRMON = prior(EmpiricalPrior(), XMON)
const WDAY = pweights(range(; start = 1, stop = 2, length = 2520))

# A sample of strictly positive returns has no loss in the worst `alpha` of any column, so
# its tail-term scale is not positive. It is one of the rule's two refusals.
const XPOS = 0.01 .+ 0.001 * abs.(randn(RNG_H, 60, 3))
const PRPOS = prior(EmpiricalPrior(), XPOS)

# Every column of `XZERO_H` sums to EXACTLY zero, so its pooled mean loss is exactly zero and
# the sample states no exchange rate between the two terms of the loss. The three columns
# carry three different patterns, so the covariance matrix is not singular.
const XZERO_H = hcat(repeat([0.02, -0.02], 30), repeat([0.03, 0.01, -0.03, -0.01], 15),
                     repeat([-0.05, 0.05], 30))
const PRZERO_H = prior(EmpiricalPrior(), XZERO_H)

# The CVaR of a return column, written out: the mean of the worst `alpha * T` of the
# sample, negated, with the boundary observation carrying its fractional part. The rule
# reads the measure's own CVaR, so this second encoding is what checks it.
function hand_cvar(x, alpha)
    T = length(x)
    s = sort(collect(x))
    aT = alpha * T
    k = ceil(Int, aT)
    return (-sum(view(s, 1:(k - 1))) - s[k] * (aT - (k - 1))) / aT
end

# A radius rule with no type at all. A closure over a caller's own number is the case that
# cannot be given one, and it is why the `alg` bound admits a bare `Function`.
function probe_radius(::Symbol, pr::PO.AbstractPriorResult, ::Any, ::Any,
                      ::PO.CalibrationContext)
    return 3 / size(pr.X, 1)
end

# The tail-weight family ships `TailTermParity`, and a caller's own function stands beside
# it. This one reports the key, so the resolver's argument order is asserted.
const TWT_SEEN = Ref{Any}(nothing)
function probe_tail_weight(key::Symbol, pr::PO.AbstractPriorResult, w, slv,
                           ctx::PO.CalibrationContext)
    TWT_SEEN[] = (; key = key, weighted = !isnothing(w), solved = !isnothing(slv),
                  series = ctx.series)
    return 2 / sqrt(size(pr.X, 1))
end

# A radius rule that answers a stated number. A rule of the family is what the resolver's
# range check dispatches on, so a test of that check needs a family and not a closure.
struct ProbeRadiusValue{T} <: PO.AbstractAmbiguityRadiusCalibrationAlgorithm
    val::T
end
function (alg::ProbeRadiusValue)(::Symbol, ::PO.AbstractPriorResult, ::Any, ::Any,
                                 ::PO.CalibrationContext)
    return alg.val
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

    # #641 withdrew the two rules that used to place a rule in these slots, so nothing
    # stands between the caller and the rule.
    @test !isdefined(PortfolioOptimisers, :AmbiguityRadiusCalibration)
    @test !isdefined(PortfolioOptimisers, :AmbiguityTailWeightCalibration)

    # Neither abstract type is exported: an export is public API, and the convention is
    # that an abstract type is not one. Both are `public`, which `names` also lists, so
    # the check reads `Base.isexported`.
    @test !Base.isexported(PortfolioOptimisers,
                           :AbstractAmbiguityRadiusCalibrationAlgorithm)
    @test !Base.isexported(PortfolioOptimisers,
                           :AbstractAmbiguityTailWeightCalibrationAlgorithm)

    # The five concrete names are exported, on the same terms as the rules of the two
    # older families.
    for sym in (:ConcentrationRadius, :RateRadius, :DimensionalRateRadius, :DualNormRadius,
                :TailTermParity)
        @test sym ∈ names(PortfolioOptimisers)
    end

    # Four rules compute a radius, and one computes an Esfahani-Kuhn tail weight. Neither
    # family holds a member of the other, and the abstract types stay unexported.
    @test ConcentrationRadius <: PO.AbstractAmbiguityRadiusCalibrationAlgorithm
    @test RateRadius <: PO.AbstractAmbiguityRadiusCalibrationAlgorithm
    @test DimensionalRateRadius <: PO.AbstractAmbiguityRadiusCalibrationAlgorithm
    @test DualNormRadius <: PO.AbstractAmbiguityRadiusCalibrationAlgorithm
    @test TailTermParity <: PO.AbstractAmbiguityTailWeightCalibrationAlgorithm
    @test subtypes(PO.AbstractAmbiguityTailWeightCalibrationAlgorithm) == [TailTermParity]
end

@testset "Ambiguity calibration: the bounds and the family split" begin
    rrule = RateRadius(; c = 0.2)

    # The slot bound admits the family's own rules, a plain function and the number, and
    # refuses the other family's rule. A function carries no family, so no bound refuses
    # one and the name of the slot is what states the quantity there.
    @test isa(rrule, PO.Num_AmbRadCal)
    @test isa(ConcentrationRadius(), PO.Num_AmbRadCal)
    @test isa(probe_radius, PO.Num_AmbRadCal)
    @test isa(0.02, PO.Num_AmbRadCal)
    @test isa(probe_tail_weight, PO.Num_AmbTwtCal)
    @test !isa(rrule, PO.Num_AmbTwtCal)
    @test !isa(TailTermParity(), PO.Num_AmbRadCal)
    @test !isa(ScenarioCount(; n = 25), PO.Num_AmbRadCal)
    @test !isa(rrule, PO.Num_SigCal)
    @test !isa(rrule, PO.Num_DefCal)

    # A rule of the wrong family in a slot is refused at construction, by the bound. No
    # guard method is written for it.
    @test_throws TypeError DistributionallyRobustConditionalValueatRisk(;
                                                                        r = TailTermParity())
    @test_throws TypeError DistributionallyRobustConditionalValueatRisk(; l = rrule)

    # A radius names no end of the distribution, so no ambiguity slot of a Range measure
    # defaults from another: each of the four states its own number.
    dr = DistributionallyRobustConditionalValueatRiskRange(; l_a = 2.0, r_a = 0.05)
    @test dr.l_b === 1.0
    @test dr.r_b === 0.02

    # `sel` keeps a rule rather than falling back to the prior's value, which is what lets
    # a rule survive the selection that runs before the resolution.
    @test PO.sel(rrule, 0.02) === rrule
    @test PO.sel(probe_tail_weight, 1.0) === probe_tail_weight
end

@testset "ConcentrationRadius: the chi-squared form" begin
    # A stated scale is used as given, and the chi-squared factor is dimensionless.
    alg = ConcentrationRadius(; confidence = 0.95, scale = 0.5)
    q = Distributions.cquantile(Distributions.Chisq(4), 0.05)
    @test alg(:r, PR60_H, nothing, nothing, CalibrationContext()) ≈ 0.5 * sqrt(q / 60)

    # A longer sample shrinks the ball, at the square-root rate.
    @test alg(:r, PR120_H, nothing, nothing, CalibrationContext()) ≈ 0.5 * sqrt(q / 120)
    @test alg(:r, PR60_H, nothing, nothing, CalibrationContext()) /
          alg(:r, PR120_H, nothing, nothing, CalibrationContext()) ≈ sqrt(2)

    # A higher confidence level buys a larger ball.
    @test ConcentrationRadius(; confidence = 0.99, scale = 0.5)(:r, PR60_H, nothing,
                                                                nothing,
                                                                CalibrationContext()) >
          alg(:r, PR60_H, nothing, nothing, CalibrationContext())

    # `scale = nothing` reads the average asset volatility off the prior result, so the
    # radius carries the units of the returns without the caller naming a number.
    auto = ConcentrationRadius()
    scale = mean(sqrt, diag(PR60_H.sigma))
    @test auto(:r, PR60_H, nothing, nothing, CalibrationContext()) ≈ scale * sqrt(q / 60)
    @test auto(:r, PR60_H, nothing, nothing, CalibrationContext()) !=
          auto(:r, PR120_H, nothing, nothing, CalibrationContext())

    # The key never selects the value, so one rule in two slots resolves to one number.
    @test alg(:r, PR60_H, nothing, nothing, CalibrationContext()) ==
          alg(:r_a, PR60_H, nothing, nothing, CalibrationContext())
    @test alg(:r, PR60_H, nothing, nothing, CalibrationContext()) ==
          alg(:l2reg_val, PR60_H, nothing, nothing, CalibrationContext())

    # Stated weights move `T` to Kish's effective sample size, which is smaller than the
    # row count for any weights that are not equal, so the ball widens.
    kish = sum(WTS)^2 / sum(abs2, WTS)
    @test kish < 60
    @test alg(:r, PR60_H, WTS, nothing, CalibrationContext()) ≈ 0.5 * sqrt(q / kish)
    @test alg(:r, PR60_H, WTS, nothing, CalibrationContext()) >
          alg(:r, PR60_H, nothing, nothing, CalibrationContext())

    # The rule needs no solver, so it ignores the one the resolution threads.
    @test alg(:r, PR60_H, nothing, Solver(; solver = nothing), CalibrationContext()) ==
          alg(:r, PR60_H, nothing, nothing, CalibrationContext())

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
    @test alg(:r, PR60_H, nothing, nothing, CalibrationContext()) ≈ 0.2 / sqrt(60)
    @test alg(:r, PR120_H, nothing, nothing, CalibrationContext()) ≈ 0.2 / sqrt(120)
    @test RateRadius().c == 1
    @test RateRadius()(:r, PR60_H, nothing, nothing, CalibrationContext()) ≈ inv(sqrt(60))

    # The rule reads the raw row count, so stated weights change nothing. That is the one
    # place it parts company with `ConcentrationRadius`.
    @test alg(:r, PR60_H, WTS, nothing, CalibrationContext()) ==
          alg(:r, PR60_H, nothing, nothing, CalibrationContext())

    # The key and the solver never select the value.
    @test alg(:r_b, PR60_H, nothing, Solver(; solver = nothing), CalibrationContext()) ==
          alg(:r, PR60_H, nothing, nothing, CalibrationContext())

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
    @test alg(:r, PR60_2, nothing, nothing, CalibrationContext()) ≈ 0.5 * sqrt(lq / 60)

    # A wide universe flattens the rate, and that difference is the whole content of the
    # rule. Over a doubling of the record `RateRadius` shrinks by `sqrt(2)` and this rule
    # shrinks by `2^(1/20)`.
    @test size(PR125_20.X, 2) == 20
    @test size(PR250_20.X, 2) == 20
    dim_ratio = alg(:r, PR125_20, nothing, nothing, CalibrationContext()) /
                alg(:r, PR250_20, nothing, nothing, CalibrationContext())
    rate = RateRadius(; c = 0.5)
    rate_ratio = rate(:r, PR125_20, nothing, nothing, CalibrationContext()) /
                 rate(:r, PR250_20, nothing, nothing, CalibrationContext())
    @test dim_ratio ≈ 2^(1 / 20)
    @test rate_ratio ≈ sqrt(2)
    @test dim_ratio < rate_ratio

    # `T^(-1/20)` at `T = 250` is `0.76`, the number the docstring names.
    @test isapprox(250^(-1 / 20), 0.76; atol = 5e-3)

    # A higher confidence level buys a larger ball, through `log(1 / (1 - confidence))`.
    @test DimensionalRateRadius(; confidence = 0.99, scale = 0.5)(:r, PR60_2, nothing,
                                                                  nothing,
                                                                  CalibrationContext()) >
          alg(:r, PR60_2, nothing, nothing, CalibrationContext())

    # `scale = nothing` reads the average asset volatility off the prior result, and a
    # stated scale overrides it. That is the pair `ConcentrationRadius` is tested on.
    auto = DimensionalRateRadius()
    scale = mean(sqrt, diag(PR60_2.sigma))
    @test auto(:r, PR60_2, nothing, nothing, CalibrationContext()) ≈ scale * sqrt(lq / 60)
    @test auto(:r, PR60_2, nothing, nothing, CalibrationContext()) !=
          alg(:r, PR60_2, nothing, nothing, CalibrationContext())
    @test isnothing(auto.scale)
    @test DimensionalRateRadius(; scale = 0.1).scale == 0.1
    @test DimensionalRateRadius().confidence == 0.95

    # The key never selects the value, so one rule in two slots resolves to one number.
    @test alg(:r, PR60_2, nothing, nothing, CalibrationContext()) ==
          alg(:r_a, PR60_2, nothing, nothing, CalibrationContext())
    @test alg(:r, PR60_2, nothing, nothing, CalibrationContext()) ==
          alg(:val, PR60_2, nothing, nothing, CalibrationContext())

    # Stated weights move `T` to Kish's effective sample size, on the same terms as
    # `ConcentrationRadius` and unlike `RateRadius`. This rule is a concentration
    # statement, so it prices the record that Kish's count measures.
    kish = sum(WTS)^2 / sum(abs2, WTS)
    @test kish < 60
    @test alg(:r, PR60_2, WTS, nothing, CalibrationContext()) ≈ 0.5 * sqrt(lq / kish)
    @test alg(:r, PR60_2, WTS, nothing, CalibrationContext()) >
          alg(:r, PR60_2, nothing, nothing, CalibrationContext())

    # The rule needs no solver, so it ignores the one the resolution threads.
    @test alg(:r, PR60_2, nothing, Solver(; solver = nothing), CalibrationContext()) ==
          alg(:r, PR60_2, nothing, nothing, CalibrationContext())

    # A four-asset universe already leaves the square-root corner.
    @test alg(:r, PR60_H, nothing, nothing, CalibrationContext()) ≈ 0.5 * (lq / 60)^(1 / 4)
    @test alg(:r, PR60_H, nothing, nothing, CalibrationContext()) !=
          alg(:r, PR60_2, nothing, nothing, CalibrationContext())

    # Construction validation, on the terms `ConcentrationRadius` writes.
    @test_throws DomainError DimensionalRateRadius(; confidence = 1.0)
    @test_throws DomainError DimensionalRateRadius(; confidence = 0.0)
    @test_throws DomainError DimensionalRateRadius(; confidence = 0.95, scale = -1.0)
    @test_throws DomainError DimensionalRateRadius(; confidence = 0.95, scale = Inf)

    # The rule joins the radius family and its bounds, and no other.
    @test DimensionalRateRadius <: PO.AbstractAmbiguityRadiusCalibrationAlgorithm
    @test isa(alg, PO.Num_AmbRadCal)
    @test !isa(alg, PO.Num_AmbTwtCal)
    @test_throws TypeError DistributionallyRobustConditionalValueatRisk(; l = alg)
end

@testset "TailTermParity: the tail term at a stated multiple of the mean term" begin
    alg = TailTermParity(; ratio = 1)
    l = alg(:l, PRDAY, nothing, nothing, CalibrationContext(; alpha = 0.05))

    # The two scales, each read off the sample without the rule. The tail-term scale is
    # written out here rather than taken off the measure, so the two encodings are checked
    # against one another.
    m = -mean(XDAY)
    c = mean(j -> hand_cvar(view(XDAY, :, j), 0.05), axes(XDAY, 2))
    @test m < 0
    @test c > 0
    @test l ≈ abs(m) / c

    # `ratio = 1` is PARITY: the tail term and the mean term are equal in magnitude on the
    # sample the prior carries. That is the whole content of the rule.
    @test l * c ≈ abs(m)

    # `ratio` is the caller's preference, and it scales the weight exactly.
    @test TailTermParity(; ratio = 2)(:l, PRDAY, nothing, nothing,
                                      CalibrationContext(; alpha = 0.05)) ≈ 2 * l
    @test TailTermParity(; ratio = 0.5)(:l, PRDAY, nothing, nothing,
                                        CalibrationContext(; alpha = 0.05)) ≈ l / 2

    # The tail-term scale is the MEAN of the per-column CVaR and not the POOLED one. A
    # pooled tail is drawn from the worst columns, so a universe of unequal volatilities
    # gives a pooled CVaR half again as large and a weight a third smaller.
    het = XDAY .* [1.0 2.0 4.0 8.0]
    c_het = mean(j -> hand_cvar(view(het, :, j), 0.05), axes(het, 2))
    pooled_het = hand_cvar(vec(het), 0.05)
    @test pooled_het > 1.4 * c_het
    @test abs(-mean(het)) / pooled_het < abs(-mean(het)) / c_het

    # The key never selects the value: the scales are read off the asset columns, so the
    # two ends of a Range measure part company through their two probabilities alone.
    @test alg(:l_a, PRDAY, nothing, nothing, CalibrationContext(; alpha = 0.05)) == l
    @test alg(:l_b, PRDAY, nothing, nothing, CalibrationContext(; alpha = 0.05)) == l

    # `alpha` fixes the depth of the tail, so a wider probability reads a smaller scale
    # and returns a larger weight.
    @test TailTermParity(; ratio = 1)(:l, PRDAY, nothing, nothing,
                                      CalibrationContext(; alpha = 0.2)) > l

    # The rule needs no solver, so it ignores the one the resolution threads.
    @test alg(:l, PRDAY, nothing, Solver(; solver = nothing),
              CalibrationContext(; alpha = 0.05)) == l

    # Both scales are sample statistics rather than counts, so stated weights move them.
    # That is where the rule parts company with `RateRadius`, which reads a raw row count.
    mw = -sum(j -> dot(view(XDAY, :, j), WDAY), axes(XDAY, 2)) / (4 * sum(WDAY))
    cw = mean(j -> ConditionalValueatRisk(; alpha = 0.05, w = WDAY)(view(XDAY, :, j)),
              axes(XDAY, 2))
    @test alg(:l, PRDAY, WDAY, nothing, CalibrationContext(; alpha = 0.05)) ≈ abs(mw) / cw
    @test alg(:l, PRDAY, WDAY, nothing, CalibrationContext(; alpha = 0.05)) != l

    # Construction validation, on the terms `RateRadius` writes.
    @test TailTermParity().ratio == 1
    @test !hasfield(TailTermParity, :alpha)
    @test_throws DomainError TailTermParity(; ratio = 0.0)
    @test_throws DomainError TailTermParity(; ratio = -1.0)
    @test_throws DomainError TailTermParity(; ratio = Inf)

    # The rule joins the tail-weight family and its bounds, and no other.
    @test TailTermParity <: PO.AbstractAmbiguityTailWeightCalibrationAlgorithm
    @test isa(alg, PO.Num_AmbTwtCal)
    @test !isa(alg, PO.Num_AmbRadCal)
    @test_throws TypeError DistributionallyRobustConditionalValueatRisk(; r = alg)
end

@testset "TailTermParity: one stated tail weight is two trade-offs" begin
    alg = TailTermParity(; ratio = 1)
    l_day = alg(:l, PRDAY, nothing, nothing, CalibrationContext(; alpha = 0.05))
    l_mon = alg(:l, PRMON, nothing, nothing, CalibrationContext(; alpha = 0.05))

    # The monthly sample sums 21 daily rows. A mean scales with the horizon and a
    # dispersion scales with its square root, so the parity weight rises by about
    # `sqrt(21)`. The rule follows the sample, and that is the defect it fixes.
    @test l_mon > l_day
    @test isapprox(l_mon / l_day, sqrt(21); rtol = 0.15)

    # At `l = 1` the daily sample prices the tail term about forty times the mean term,
    # and the monthly one under ten times. Those are the numbers the docstring names.
    @test 40 < inv(l_day) < 50
    @test 8 < inv(l_mon) < 10

    # A STATED number is one number on both samples. The same slot holding the rule gives
    # two, and each is the trade-off its own sample earns.
    @test PO.resolve_calibration_slot(1.0, :l, PRDAY, nothing) ==
          PO.resolve_calibration_slot(1.0, :l, PRMON, nothing)
    rule = alg
    lctx = CalibrationContext(; alpha = 0.05)
    @test PO.resolve_calibration_slot(rule, :l, PRDAY, nothing, nothing, lctx) !=
          PO.resolve_calibration_slot(rule, :l, PRMON, nothing, nothing, lctx)
end

@testset "TailTermParity: `alpha` and `l` travel together" begin
    alg = TailTermParity(; ratio = 1)
    rule = alg

    # The rule cannot run until the probability reaches it, and the message names the verb
    # that carries it.
    err = try
        alg(:l, PRDAY, nothing, nothing, CalibrationContext())
        nothing
    catch e
        e
    end
    @test isa(err, PO.IsNothingError)
    @test occursin("CalibrationContext.alpha", err.msg)

    # The context carries the number, and the resolver hands it through the RULE as well as
    # to a bare rule. A stated number and a plain function ignore the field, so no existing
    # behaviour moved.
    c05 = CalibrationContext(; alpha = 0.05)
    @test !hasfield(TailTermParity, :alpha)
    @test TailTermParity(; ratio = 3).ratio == 3
    @test PO.resolve_calibration_slot(rule, :l, PRDAY, nothing, nothing, c05) ≈
          alg(:l, PRDAY, nothing, nothing, c05)
    @test PO.resolve_calibration_slot(1.0, :l, PRDAY, nothing, nothing, c05) === 1.0
    @test PO.resolve_calibration_slot(probe_tail_weight, :l, PRDAY, nothing, nothing, c05) ≈
          2 / sqrt(size(PRDAY.X, 1))

    # The scalar measure resolves `alpha` first, so the slot reads the measure's own
    # probability and two measures give two weights.
    m1 = DistributionallyRobustConditionalValueatRisk(; alpha = 0.05, l = rule)
    m2 = DistributionallyRobustConditionalValueatRisk(; alpha = 0.2, l = rule)
    @test PO.resolve_deferred_quantities(m1, PRDAY).l ≈
          TailTermParity(; ratio = 1)(:l, PRDAY, nothing, nothing,
                                      CalibrationContext(; alpha = 0.05))
    @test PO.resolve_deferred_quantities(m2, PRDAY).l ≈
          TailTermParity(; ratio = 1)(:l, PRDAY, nothing, nothing,
                                      CalibrationContext(; alpha = 0.2))
    @test PO.resolve_deferred_quantities(m1, PRDAY).l !=
          PO.resolve_deferred_quantities(m2, PRDAY).l

    # Each end of a Range measure reads its OWN probability, which is the case the six-slot
    # resolution has to get right.
    rg = DistributionallyRobustConditionalValueatRiskRange(; alpha = 0.05, beta = 0.2,
                                                           l_a = rule, l_b = rule)
    out = PO.resolve_deferred_quantities(rg, PRDAY)
    @test out.l_a ≈ TailTermParity(; ratio = 1)(:l_a, PRDAY, nothing, nothing,
                                                CalibrationContext(; alpha = 0.05))
    @test out.l_b ≈ TailTermParity(; ratio = 1)(:l_b, PRDAY, nothing, nothing,
                                                CalibrationContext(; alpha = 0.2))
    @test out.l_a != out.l_b
    @test out.alpha == 0.05
    @test out.beta == 0.2

    # A rule in the `alpha` slot beside a rule in the `l` slot resolves in the right order:
    # the tail weight reads the RESOLVED probability and not the slot's occupant.
    srole = RateSignificance(; c = 0.5)
    both = DistributionallyRobustConditionalValueatRisk(; alpha = srole, l = rule)
    bout = PO.resolve_deferred_quantities(both, PRDAY)
    @test bout.alpha ≈ 0.5 / sqrt(2520)
    @test bout.l ≈ TailTermParity(; ratio = 1)(:l, PRDAY, nothing, nothing,
                                               CalibrationContext(; alpha = bout.alpha))

    # The drawdown measure carries the same pair, and it hands its own SERIES over beside
    # the probability. #623 corrected this: the number was the asset-column one, and the
    # tail term this measure prices is a CDaR of the drawdown series.
    dd = DistributionallyRobustConditionalDrawdownatRisk(; alpha = 0.05, l = rule)
    ddl = PO.resolve_deferred_quantities(dd, PRDAY).l
    @test ddl ≈ TailTermParity(; ratio = 1)(:l, PRDAY, nothing, nothing,
                                            CalibrationContext(; alpha = 0.05,
                                                               series = AbsoluteDrawdownSeries()))
    @test ddl != TailTermParity(; ratio = 1)(:l, PRDAY, nothing, nothing,
                                             CalibrationContext(; alpha = 0.05))

    # A measure whose slots all hold numbers is still returned unchanged, so the context
    # costs nothing where no rule reads a sibling.
    plain = DistributionallyRobustConditionalValueatRisk()
    @test PO.resolve_deferred_quantities(plain, PRDAY) === plain
    plain_rg = DistributionallyRobustConditionalValueatRiskRange()
    @test PO.resolve_deferred_quantities(plain_rg, PRDAY) === plain_rg
end

@testset "TailTermParity: the two refusals" begin
    alg = TailTermParity(; ratio = 1)

    # A sample of strictly positive returns holds no loss in the worst `alpha` of any
    # column, so it has no tail-term scale and no weight prices one term against the other.
    err = try
        alg(:l, PRPOS, nothing, nothing, CalibrationContext(; alpha = 0.05))
        nothing
    catch e
        e
    end
    @test isa(err, DomainError)
    @test occursin("tail-term scale", err.msg)

    # A sample whose pooled mean loss is zero states no exchange rate between the two
    # terms: every weight prices a mean term of zero alike.
    @test mean(XZERO_H) == 0
    err = try
        alg(:l, PRZERO_H, nothing, nothing, CalibrationContext(; alpha = 0.05))
        nothing
    catch e
        e
    end
    @test isa(err, DomainError)
    @test occursin("mean-term scale", err.msg)
end

@testset "Ambiguity calibration: the resolver runs a rule by calling it" begin
    rrole = RateRadius(; c = 0.2)
    trole = probe_tail_weight

    # The rule is unwrapped and the rule is called, so the rule never reaches the rule.
    @test PO.resolve_calibration_slot(rrole, :r, PR60_H, nothing) ≈ 0.2 / sqrt(60)

    # A stated number passes through unchanged, and so does `nothing`.
    @test PO.resolve_calibration_slot(0.02, :r, PR60_H, nothing) == 0.02
    @test isnothing(PO.resolve_calibration_slot(nothing, :l1, PR60_H, nothing))

    # The five arguments arrive in the stated order, and the last three are the effective
    # observation weights, the effective solver and the site context.
    TWT_SEEN[] = nothing
    slv = Solver(; name = :probe, solver = nothing)
    @test PO.resolve_calibration_slot(trole, :l, PR60_H, WTS, slv) ≈ 2 / sqrt(60)
    @test TWT_SEEN[] ==
          (; key = :l, weighted = true, solved = true, series = ReturnsSeries())

    # A plain function in the `alg` field is a rule on the same terms.
    @test PO.resolve_calibration_slot(probe_radius, :r, PR60_H, nothing) ≈ 3 / 60
end

@testset "Ambiguity calibration: the six distributionally robust slots" begin
    rrole = RateRadius(; c = 0.2)
    trole = probe_tail_weight
    r_num = 0.2 / sqrt(60)
    l_num = 2 / sqrt(60)

    # --- the scalar value-at-risk measure ----------------------------------------------
    m = DistributionallyRobustConditionalValueatRisk(; l = trole, r = rrole)
    # `alpha` joined the declaration when #583 widened the significance slots, and it
    # stands first because the declaration follows field order.
    @test PO.calibration_slots(m) == (; alpha = m.alpha, l = trole, r = rrole)
    out = PO.resolve_deferred_quantities(m, PR60_H)
    @test out.r ≈ r_num
    @test out.l ≈ l_num
    @test out.alpha == m.alpha

    # A measure whose slots both hold numbers is returned unchanged, so the common case
    # allocates nothing.
    plain = DistributionallyRobustConditionalValueatRisk()
    @test PO.resolve_deferred_quantities(plain, PR60_H) === plain

    # The generated `factory` reaches the same resolution, which is the channel the
    # optimisation itself uses.
    @test PO.factory(m, PR60_H).r ≈ r_num

    # The resolver checks the number the rule returned, so a rule that returns a value the
    # slot does not admit is refused at fold time, by the check a caller's own number
    # meets. The rebuild through the keyword constructor states the same rule again.
    neg = (args...) -> -1.0
    @test_throws DomainError PO.resolve_deferred_quantities(DistributionallyRobustConditionalValueatRisk(;
                                                                                                         r = neg),
                                                            PR60_H)

    # --- the range measure --------------------------------------------------------------
    rg = DistributionallyRobustConditionalValueatRiskRange(; l_a = trole, r_a = rrole,
                                                           l_b = trole, r_b = rrole)
    @test PO.calibration_slots(rg) ==
          (; alpha = rg.alpha, l_a = trole, r_a = rrole, beta = rg.beta, l_b = trole,
           r_b = rrole)
    rgo = PO.resolve_deferred_quantities(rg, PR60_H)
    @test rgo.r_a ≈ r_num
    @test rgo.r_b ≈ r_num
    @test rgo.l_a ≈ l_num
    @test rgo.l_b ≈ l_num
    @test rgo.alpha == rg.alpha
    @test rgo.beta == rg.beta
    plain_rg = DistributionallyRobustConditionalValueatRiskRange()
    @test PO.resolve_deferred_quantities(plain_rg, PR60_H) === plain_rg

    # Each tail keeps its own pair, so two different rules give two different numbers.
    mixed = DistributionallyRobustConditionalValueatRiskRange(; r_a = rrole,
                                                              r_b = RateRadius(; c = 0.9))
    mout = PO.resolve_deferred_quantities(mixed, PR60_H)
    @test mout.r_a ≈ 0.2 / sqrt(60)
    @test mout.r_b ≈ 0.9 / sqrt(60)

    # --- the drawdown measure -----------------------------------------------------------
    dd = DistributionallyRobustConditionalDrawdownatRisk(; l = trole, r = rrole)
    @test PO.calibration_slots(dd) == (; alpha = dd.alpha, l = trole, r = rrole)
    ddo = PO.resolve_deferred_quantities(dd, PR60_H)
    @test ddo.r ≈ r_num
    @test ddo.l ≈ l_num
    plain_dd = DistributionallyRobustConditionalDrawdownatRisk()
    @test PO.resolve_deferred_quantities(plain_dd, PR60_H) === plain_dd

    # --- the rule split holds at the slot ----------------------------------------------
    # A radius rule in a tail-weight slot, and the reverse, are both refused at
    # construction. The bound is the whole of the validation.
    @test_throws TypeError DistributionallyRobustConditionalValueatRisk(; l = rrole)
    @test_throws TypeError DistributionallyRobustConditionalValueatRisk(;
                                                                        r = TailTermParity())
    @test_throws TypeError DistributionallyRobustConditionalValueatRiskRange(; l_a = rrole)
    @test_throws TypeError DistributionallyRobustConditionalValueatRiskRange(;
                                                                             r_b = TailTermParity())
    @test_throws TypeError DistributionallyRobustConditionalDrawdownatRisk(; l = rrole)

    # A significance rule is refused too: it belongs to neither family.
    @test_throws TypeError DistributionallyRobustConditionalValueatRisk(;
                                                                        r = ScenarioCount(;
                                                                                          n = 3))

    # --- the value-level entry point refuses a rule ------------------------------------
    # `expected_risk` given a bare returns matrix has no prior result to resolve against,
    # so it names the slot and the way out rather than failing several frames down.
    w = fill(0.25, 4)
    err = try
        expected_risk(m, w, X60_H)
        nothing
    catch e
        e
    end
    @test isa(err, ArgumentError)
    @test occursin("DistributionallyRobustConditionalValueatRisk.l", err.msg)
    @test occursin("probe_tail_weight", err.msg)
    @test occursin("factory", err.msg)

    # Given the prior result the measure is resolved first, so the same call succeeds.
    @test isfinite(expected_risk(m, w, PR60_H))

    # A stated number still reaches the functor untouched, so nothing existing changed.
    @test expected_risk(plain, w, X60_H) ≈ expected_risk(plain, w, PR60_H)
end

@testset "Ambiguity calibration: the two regularisation coefficients" begin
    rrole = RateRadius(; c = 0.2)
    r_num = 0.2 / sqrt(60)

    # --- L2Regularisation ---------------------------------------------------------------
    l2 = L2Regularisation(; val = rrole)
    @test PO.calibration_slots(l2) == (; val = rrole)
    @test PO.factory(l2, PR60_H).val ≈ r_num
    @test isa(PO.factory(l2, PR60_H).alg, SOCRiskExpr)

    # A term holding a number is returned unchanged, and a vector of terms is resolved
    # element by element by the generic vector method.
    plain_l2 = L2Regularisation(; val = 0.3)
    @test PO.factory(plain_l2, PR60_H) === plain_l2
    @test [x.val for x in PO.factory([l2, plain_l2], PR60_H)] ≈ [r_num, 0.3]

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

    # The resolver checks the calibrated number, and the rebuild states the rule again.
    neg = L2Regularisation(; val = (args...) -> -1.0)
    @test_throws DomainError PO.factory(neg, PR60_H)

    # --- LpRegularisation ---------------------------------------------------------------
    lp = LpRegularisation(; p = 3, val = rrole)
    @test PO.calibration_slots(lp) == (; val = rrole)
    @test PO.factory(lp, PR60_H).val ≈ r_num
    @test PO.factory(lp, PR60_H).p == 3
    plain_lp = LpRegularisation(; val = 0.4)
    @test PO.factory(plain_lp, PR60_H) === plain_lp

    # No formulation slot, so no pairing can be wrong and no guard runs.
    @test LpRegularisation(; p = 2.5, val = rrole).val === rrole

    # A tail-weight rule is refused in a radius slot, on both estimators.
    @test_throws TypeError L2Regularisation(; val = TailTermParity())
    @test_throws TypeError LpRegularisation(; val = TailTermParity())

    # `L2Regularisation.val` names one quantity, so it admits a plain function. The dual-use
    # slot of `LpRegularisation` names two, so it is the one slot that refuses one.
    @test L2Regularisation(; val = probe_radius).val === probe_radius
    @test_throws TypeError LpRegularisation(; val = probe_radius)
end

@testset "Ambiguity calibration: the optimiser's l1 and linf reach the model" begin
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = "verbose" => false)
    rd = ReturnsResult(; nx = string.(1:4), X = X60_H)
    rrole = RateRadius(; c = 0.2)
    r_num = 0.2 / sqrt(60)

    # The four coefficients all take a rule, and the bound admits it.
    opt = JuMPOptimiser(; slv = slv, pe = PR60_H, l1 = rrole, linf = rrole,
                        l2 = L2Regularisation(; val = rrole),
                        lp = LpRegularisation(; val = rrole))
    @test opt.l1 === rrole
    @test opt.linf === rrole

    # Neither the weights factory nor the cluster slice holds a prior result, so both carry
    # a rule through untouched. That is right: the rule resolves against the cluster's own
    # prior when the model is assembled.
    @test PO.factory(opt, fill(0.25, 4)).l1 === rrole
    @test PO.port_opt_view(opt, 1:3, X60_H).linf === rrole

    # Assembly resolves all four against the optimisation's own prior result, so the
    # coefficient the model carries is the calibrated number and not the rule.
    mr = MeanRisk(; r = Variance(), opt = opt)
    attrs = PO.processed_jump_optimiser_attributes(mr.opt, rd)
    model = JuMP.Model()
    PO.set_model_scales!(model, mr.opt.sc, mr.opt.so)
    PO.set_model_observations!(model, size(attrs.pr.X, 1))
    PO.set_maximum_ratio_factor_variables!(model, mr.obj)
    PO.set_w!(model, attrs.pr.X, mr.wi)
    PO.set_weight_constraints!(model, attrs.wb, mr.opt)
    PO.assemble_jump_model!(model, mr, mr.opt, attrs, rd, mr.r, mr.obj)
    @test JuMP.coefficient(model[:l1], model[:t_l1]) ≈ r_num
    @test JuMP.coefficient(model[:linf], model[:t_linf]) ≈ r_num

    # A stated number reaches the same coefficient, so the widening changed nothing for a
    # caller who names one.
    optn = JuMPOptimiser(; slv = slv, pe = PR60_H, l1 = r_num)
    mrn = MeanRisk(; r = Variance(), opt = optn)
    attrsn = PO.processed_jump_optimiser_attributes(mrn.opt, rd)
    modeln = JuMP.Model()
    PO.set_model_scales!(modeln, mrn.opt.sc, mrn.opt.so)
    PO.set_model_observations!(modeln, size(attrsn.pr.X, 1))
    PO.set_maximum_ratio_factor_variables!(modeln, mrn.obj)
    PO.set_w!(modeln, attrsn.pr.X, mrn.wi)
    PO.set_weight_constraints!(modeln, attrsn.wb, mrn.opt)
    PO.assemble_jump_model!(modeln, mrn, mrn.opt, attrsn, rd, mrn.r, mrn.obj)
    @test JuMP.coefficient(modeln[:l1], modeln[:t_l1]) ≈ r_num

    # A tail weight is not a radius, so the tail-weight rule is refused in all four slots.
    @test_throws TypeError JuMPOptimiser(; slv = slv, l1 = TailTermParity())
    @test_throws TypeError JuMPOptimiser(; slv = slv, linf = TailTermParity())
end

#=
The two deferral channels can now meet in one field: `l1` and `linf` are bounded
`TD_Option{<:Num_AmbRadCal}`, so a `TimeDependent` may wrap a Calibration Rule. ADR 0030
never considered a second channel, and the amendment issues #617, #619, #620 and #621 added
to ADR 0095 settles it: a schedule reaches the HOST that holds the slot and no further. A
rule is never standalone, so the host is what a schedule swaps, and no channel is missing.

The two verbs run at two different points of the pipeline and neither knows about the
other: the period selection runs in `update_time_dependent_fields`, before any prior is
fitted, and the calibration resolution runs at assembly, against the prior of the period
that was selected. So the order falls out of the pipeline and nothing had to invent it.
=#
@testset "Ambiguity calibration: the time-dependent wrapper selects, then the rule runs" begin
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = "verbose" => false)
    rrole = RateRadius(; c = 0.2)
    rd = ReturnsResult(; nx = string.(1:4), X = X120_H)

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
    @test PO.resolve_calibration_slot(sel1, :l1, PR60_H, nothing) ≈ 0.2 / sqrt(60)
    @test PO.resolve_calibration_slot(sel1, :l1, PR120_H, nothing) ≈ 0.2 / sqrt(120)

    # Outside every fold loop the schedule falls back to its own default, which is a
    # number here and reaches the model as one.
    @test PO.reset_time_dependent_fields(opt).l1 == 0.5
end

#=
`DualNormRadius` is the first rule of the family for which `key` carries meaning. The eight
radius slots do not measure distance in one norm, so a rule that returns one number for
every key gives one coefficient to two penalties whose natural scales differ. This rule
reads the key, picks the ground metric of that slot, and returns the sampling error of the
empirical measure in it.

Issue #614 ships it, and it settles the `:val` collision by widening the two regularisation
keys to `:l2reg_val` and `:lpreg_val`. The two symbols are the two names `field_dict`
already uses for the two slots, so a caller's own function placed in one of those slots
now receives one of those keys in place of `:val`.

The `:lpreg_val` key still names no norm order, because `p` lives on the owner. The order
travels to the rule in the `CalibrationContext` the penalty site builds, so the caller states
the order once and on the penalty.

The prior below carries a DIAGONAL covariance matrix, so every norm of the per-asset error
vector has a closed form and no test reads a number off the implementation.
=#
const DNR_SD = [0.2, 0.3, 0.4, 0.5]
const PRDIAG = LowOrderPrior(; X = X60_H, mu = zeros(4),
                             sigma = Matrix(Diagonal(DNR_SD .^ 2)))
const DNR_Z = Distributions.quantile(Distributions.Normal(), 0.95)
const DNR_KISH = sum(WTS)^2 / sum(abs2, WTS)

@testset "DualNormRadius: the ground metric of the slot selects the norm" begin
    # The rule joins the radius family, and it is exported beside the two that ship.
    @test DualNormRadius <: PO.AbstractAmbiguityRadiusCalibrationAlgorithm
    @test !(DualNormRadius <: PO.AbstractAmbiguityTailWeightCalibrationAlgorithm)
    @test :DualNormRadius ∈ names(PortfolioOptimisers)
    @test !isa(DualNormRadius(), PO.Num_AmbTwtCal)
    @test isa(DualNormRadius(), PO.Num_AmbRadCal)

    # The defaults: a per-coordinate 95% level, and no norm order, which serves every slot
    # but `:lpreg_val`.
    @test DualNormRadius().confidence == 0.95
    @test !hasfield(DualNormRadius, :p)
    @test isnothing(CalibrationContext().p)

    alg = DualNormRadius()
    e = DNR_SD / sqrt(60)

    # --- the four closed forms, on a diagonal covariance matrix -------------------------
    # The `l1` coefficient multiplies the 1-norm of the weights, so its ground metric is
    # the ∞-norm and the radius is the largest per-asset error.
    @test alg(:l1, PRDIAG, nothing, nothing, CalibrationContext()) ≈ DNR_Z * maximum(e)

    # The `linf` coefficient and the three DR radii all multiply the ∞-norm, so their
    # ground metric is the 1-norm and the radius is the sum of the per-asset errors.
    @test alg(:linf, PRDIAG, nothing, nothing, CalibrationContext()) ≈ DNR_Z * sum(e)
    @test alg(:r, PRDIAG, nothing, nothing, CalibrationContext()) ≈ DNR_Z * sum(e)

    # The L2 penalty is its own dual, so the radius is the 2-norm.
    @test alg(:l2reg_val, PRDIAG, nothing, nothing, CalibrationContext()) ≈
          DNR_Z * norm(e, 2)

    # The Lp penalty's ground metric is the type-`q` metric with `1/p + 1/q = 1`. The
    # penalty site states `p` in the context, and a context a caller builds runs the rule
    # on its own, outside that site.
    @test DualNormRadius()(:lpreg_val, PRDIAG, nothing, nothing,
                           CalibrationContext(; p = 3)) ≈ DNR_Z * norm(e, 1.5)
    @test DualNormRadius()(:lpreg_val, PRDIAG, nothing, nothing,
                           CalibrationContext(; p = 1.25)) ≈ DNR_Z * norm(e, 5)

    # --- the defect the rule fixes ------------------------------------------------------
    # One rule, two slots, two numbers. That is the whole point: `ConcentrationRadius` and
    # `RateRadius` cannot tell the two apart, and this rule can.
    @test alg(:l1, PRDIAG, nothing, nothing, CalibrationContext()) !=
          alg(:linf, PRDIAG, nothing, nothing, CalibrationContext())
    @test alg(:linf, PRDIAG, nothing, nothing, CalibrationContext()) >
          alg(:l1, PRDIAG, nothing, nothing, CalibrationContext())
    @test alg(:l2reg_val, PRDIAG, nothing, nothing, CalibrationContext()) !=
          alg(:l1, PRDIAG, nothing, nothing, CalibrationContext()) !=
          alg(:linf, PRDIAG, nothing, nothing, CalibrationContext())

    # A radius names no end of the distribution, so the two ends of a Range twin resolve to
    # one number and both agree with the scalar measure's own slot.
    @test alg(:r_a, PRDIAG, nothing, nothing, CalibrationContext()) ==
          alg(:r_b, PRDIAG, nothing, nothing, CalibrationContext())
    @test alg(:r_a, PRDIAG, nothing, nothing, CalibrationContext()) ==
          alg(:r, PRDIAG, nothing, nothing, CalibrationContext())

    # --- the confidence level -----------------------------------------------------------
    # A higher level buys a larger ball, and the level scales the whole vector, so the
    # ratio of two levels is the ratio of the two quantiles in every ground metric.
    hi = DualNormRadius(; confidence = 0.99)
    z99 = Distributions.quantile(Distributions.Normal(), 0.99)
    @test hi(:l1, PRDIAG, nothing, nothing, CalibrationContext()) >
          alg(:l1, PRDIAG, nothing, nothing, CalibrationContext())
    @test hi(:linf, PRDIAG, nothing, nothing, CalibrationContext()) /
          alg(:linf, PRDIAG, nothing, nothing, CalibrationContext()) ≈ z99 / DNR_Z

    # --- the sample size ----------------------------------------------------------------
    # With no weights the rule reads the raw row count, and stated weights move it to
    # Kish's effective sample size, which is smaller for any weights that are not equal.
    @test DNR_KISH < 60
    @test alg(:l1, PRDIAG, WTS, nothing, CalibrationContext()) ≈
          DNR_Z * maximum(DNR_SD) / sqrt(DNR_KISH)
    @test alg(:l1, PRDIAG, WTS, nothing, CalibrationContext()) >
          alg(:l1, PRDIAG, nothing, nothing, CalibrationContext())
    @test alg(:linf, PRDIAG, WTS, nothing, CalibrationContext()) ≈
          DNR_Z * sum(DNR_SD) / sqrt(DNR_KISH)
    @test alg(:linf, PRDIAG, WTS, nothing, CalibrationContext()) /
          alg(:linf, PRDIAG, nothing, nothing, CalibrationContext()) ≈ sqrt(60 / DNR_KISH)

    # The rule needs no solver, so it ignores the one the resolution threads.
    @test alg(:l1, PRDIAG, nothing, Solver(; solver = nothing), CalibrationContext()) ==
          alg(:l1, PRDIAG, nothing, nothing, CalibrationContext())

    # --- the two refusals ---------------------------------------------------------------
    # An unrecognised key is the one refusal the rule owes: a caller who writes their own
    # measure hits it first, so the message names the key it got and the keys it serves.
    err = try
        alg(:alpha, PRDIAG, nothing, nothing, CalibrationContext())
        nothing
    catch e
        e
    end
    @test isa(err, ArgumentError)
    @test occursin(":alpha", err.msg)
    for k in (":l1", ":linf", ":r", ":r_a", ":r_b", ":l2reg_val", ":lpreg_val")
        @test occursin(k, err.msg)
    end

    # `:lpreg_val` with no norm order cannot name its ground metric. The penalty site fills
    # the field, so a `nothing` here means the rule ran outside that site, and the message
    # names both the field and the verb that fills it.
    perr = try
        alg(:lpreg_val, PRDIAG, nothing, nothing, CalibrationContext())
        nothing
    catch e
        e
    end
    @test isa(perr, ArgumentError)
    @test occursin("lpreg_val", perr.msg)
    @test occursin("CalibrationContext.p", perr.msg)

    # The scale function is the whole of the key's meaning, and it is reachable on its own.
    # It takes the order rather than the rule, because the order is the site's.
    @test PO.dual_norm_radius_scale(:l1, nothing, DNR_SD) == maximum(DNR_SD)
    @test PO.dual_norm_radius_scale(:linf, nothing, DNR_SD) == sum(DNR_SD)
    @test_throws ArgumentError PO.dual_norm_radius_scale(:beta, nothing, DNR_SD)

    # --- validation ---------------------------------------------------------------------
    # The confidence level is a probability, and it is the rule's own. The norm order is
    # the site's, so the rule that READS it is what refuses one with no conjugate.
    @test_throws DomainError DualNormRadius(; confidence = 1.0)
    @test_throws DomainError DualNormRadius(; confidence = 0.0)
    @test_throws DomainError DualNormRadius()(:lpreg_val, PRDIAG, nothing, nothing,
                                              CalibrationContext(; p = 1.0))
    @test_throws DomainError DualNormRadius()(:lpreg_val, PRDIAG, nothing, nothing,
                                              CalibrationContext(; p = 0.5))
    @test_throws DomainError DualNormRadius()(:lpreg_val, PRDIAG, nothing, nothing,
                                              CalibrationContext(; p = Inf))
end

@testset "DualNormRadius: the three distributionally robust radius slots" begin
    rule = DualNormRadius()
    # The measures resolve against the empirical prior, so the number is read from the same
    # rule rather than written out twice.
    r_num = DualNormRadius()(:r, PR60_H, nothing, nothing, CalibrationContext())
    @test r_num > 0

    m = DistributionallyRobustConditionalValueatRisk(; r = rule)
    @test PO.resolve_deferred_quantities(m, PR60_H).r ≈ r_num
    @test PO.factory(m, PR60_H).r ≈ r_num

    # The drawdown owner hands its own series over, so the error scale comes off the
    # drawdown sample. #623 corrected this: the number was the asset-return one.
    dd = DistributionallyRobustConditionalDrawdownatRisk(; r = rule)
    ddr = PO.resolve_deferred_quantities(dd, PR60_H).r
    @test ddr ≈ DualNormRadius()(:r, PR60_H, nothing, nothing,
                                 CalibrationContext(; series = AbsoluteDrawdownSeries()))
    @test ddr != r_num

    # Both ends of the Range twin carry the same ground metric, so one rule on both ends
    # gives one number. The two ends of a radius are not two tails.
    rg = DistributionallyRobustConditionalValueatRiskRange(; r_a = rule, r_b = rule)
    rgo = PO.resolve_deferred_quantities(rg, PR60_H)
    @test rgo.r_a ≈ r_num
    @test rgo.r_b ≈ r_num
    @test rgo.r_a == rgo.r_b
end

@testset "DualNormRadius: the two regularisation keys are two names, not one" begin
    # `:val` named both slots, and the two carry two different ground metrics, so route 1
    # of #614 widened them to the two names `field_dict` already used.
    l2rule = DualNormRadius()
    l2 = L2Regularisation(; val = l2rule)
    @test PO.factory(l2, PRDIAG).val ≈ DNR_Z * norm(DNR_SD / sqrt(60), 2)

    # The Lp term reads the type-`q` metric of its own norm order. The order belongs to the
    # penalty, so the site states it in the context and the caller states nothing: no rule
    # carries a `p` of its own.
    lprole = DualNormRadius()
    lp = LpRegularisation(; p = 3, val = lprole)
    @test PO.factory(lp, PRDIAG).val ≈ DNR_Z * norm(DNR_SD / sqrt(60), 1.5)
    @test PO.factory(lp, PRDIAG).p == 3

    # A second term of a different norm order gets a different number from the same rule,
    # which is the whole reason the order travels rather than sitting on the rule.
    lp2 = LpRegularisation(; p = 1.25, val = lprole)
    @test PO.factory(lp2, PRDIAG).val ≈ DNR_Z * norm(DNR_SD / sqrt(60), 5)
    @test PO.factory(lp, PRDIAG).val != PO.factory(lp2, PRDIAG).val

    # The two keys reach two different numbers from one rule, which is what the widening
    # bought. A rule that read `:val` for both could not tell them apart.
    @test PO.factory(l2, PRDIAG).val != PO.factory(lp, PRDIAG).val

    # The order reaches the rule through the context, and leaves everything else where it
    # was: a number, and a rule that reads no order, both cross whole and nothing is
    # rebuilt on the way in.
    c3 = CalibrationContext(; p = 3)
    @test l2rule.confidence == DualNormRadius().confidence
    @test PO.resolve_calibration_slot(0.3, :lpreg_val, PRDIAG, nothing, nothing, c3) == 0.3
    cr_rule = ConcentrationRadius()
    @test PO.resolve_calibration_slot(cr_rule, :r, PRDIAG, nothing, nothing, c3) ==
          cr_rule(:r, PRDIAG, nothing, nothing, c3)

    # The three older rules read no key, so the widening moved nothing for them.
    cr = ConcentrationRadius(; scale = 0.5)
    @test cr(:l2reg_val, PRDIAG, nothing, nothing, CalibrationContext()) ==
          cr(:lpreg_val, PRDIAG, nothing, nothing, CalibrationContext())
end

@testset "DualNormRadius: l1 and linf reach the model with two coefficients" begin
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = "verbose" => false)
    rd = ReturnsResult(; nx = string.(1:4), X = X60_H)
    rule = DualNormRadius()

    # One rule, stated on both coefficients of one optimiser.
    opt = JuMPOptimiser(; slv = slv, pe = PR60_H, l1 = rule, linf = rule)
    mr = MeanRisk(; r = Variance(), opt = opt)
    attrs = PO.processed_jump_optimiser_attributes(mr.opt, rd)
    model = JuMP.Model()
    PO.set_model_scales!(model, mr.opt.sc, mr.opt.so)
    PO.set_model_observations!(model, size(attrs.pr.X, 1))
    PO.set_maximum_ratio_factor_variables!(model, mr.obj)
    PO.set_w!(model, attrs.pr.X, mr.wi)
    PO.set_weight_constraints!(model, attrs.wb, mr.opt)
    PO.assemble_jump_model!(model, mr, mr.opt, attrs, rd, mr.r, mr.obj)

    c_l1 = JuMP.coefficient(model[:l1], model[:t_l1])
    c_linf = JuMP.coefficient(model[:linf], model[:t_linf])

    # The defect #614 fixes: the two penalties bound distance in two different norms, so
    # one rule must give them two coefficients. Under either older rule these are equal.
    @test c_l1 != c_linf
    @test c_l1 ≈ DualNormRadius()(:l1, PR60_H, PR60_H.w, slv, CalibrationContext())
    @test c_linf ≈ DualNormRadius()(:linf, PR60_H, PR60_H.w, slv, CalibrationContext())
    @test c_linf > c_l1

    # The gap grows with the universe, because the 1-norm of the error vector sums over the
    # assets and the ∞-norm does not.
    @test c_linf / c_l1 > 1
end

#=
Issue #623. Three rules of the two ambiguity families read a RETURNS quantity when their
slot owner prices a DRAWDOWN, and `DimensionalRateRadius` reads the same one. The mechanism
that closes the gap is the one commit 7e51431c45 built for the deformation family:
`calibration_series(x)` is the trait the owner answers, and `CalibrationContext.series`
carries the answer into the rule.

The reading is not a matter of taste. `set_risk_constraints!` for
`DistributionallyRobustConditionalDrawdownatRisk` measures the transport cost of its own
programme against `set_portfolio_drawdowns_plus_one!(model, pr.X)`, which is the per-asset
absolute drawdown sample plus one. So the ball is a ball over DRAWDOWN scenarios, the radius
is a distance between two such vectors, and the tail term the weight prices is a CDaR. The
first testset below pins that matrix against `calibration_series_matrix`, so the reading and
the model cannot part company.
=#
const XDDA = 0.0005 .+ 0.01 .* rand(StableRNG(717171), Distributions.TDist(4), 1500, 5)
const PRDDA = prior(EmpiricalPrior(), XDDA)
const DDA = PO.calibration_series_matrix(AbsoluteDrawdownSeries(), XDDA)

@testset "Ambiguity calibration: the model measures its ball over the drawdowns" begin
    # The matrix the transport cost is measured against IS the drawdown sample, shifted by
    # the support offset. This is the whole of the argument that a radius under this owner
    # carries drawdown units.
    model = JuMP.Model()
    ddap1 = PO.set_portfolio_drawdowns_plus_one!(model, XDDA)
    @test JuMP.value.(ddap1) ≈ DDA .+ 1
    @test DDA ≈ PO.absolute_drawdown_arr(XDDA)

    # A drawdown series is non-positive and its dispersion is the LARGER of the two, so
    # every rule below returns a larger number under the drawdown marker.
    @test all(<=(0), DDA)
    sr = PO.calibration_series_dispersion(ReturnsSeries(), PRDDA)
    sd = PO.calibration_series_dispersion(AbsoluteDrawdownSeries(), PRDDA)
    @test sr ≈ sqrt.(diag(PRDDA.sigma))
    @test sd ≈ vec(std(DDA; dims = 1))
    @test all(sd .> sr)

    # The relative marker is a third number, not a rescaling of the absolute one.
    sdr = PO.calibration_series_dispersion(RelativeDrawdownSeries(), PRDDA)
    @test sdr != sd
    @test length(sdr) == size(XDDA, 2)
end

@testset "Ambiguity calibration: the three radius rules read the owner's series" begin
    # Each rule multiplies ONE scale by a factor that no marker touches, so the ratio of
    # the two readings is the ratio of the two scales, and it is one number for all three.
    sr = mean(PO.calibration_series_dispersion(ReturnsSeries(), PRDDA))
    sd = mean(PO.calibration_series_dispersion(AbsoluteDrawdownSeries(), PRDDA))

    for alg in (ConcentrationRadius(), DimensionalRateRadius())
        r = alg(:r, PRDDA, nothing, nothing, CalibrationContext())
        d = alg(:r, PRDDA, nothing, nothing,
                PO.CalibrationContext(; series = AbsoluteDrawdownSeries()))
        @test d > r
        @test d / r ≈ sd / sr
    end

    # `DualNormRadius` takes a NORM of the error vector rather than its mean, so it moves
    # with the vector and not with the mean. The ground metric does not move with the
    # series: `:r` is the 1-norm under both markers.
    e_r = PO.calibration_series_dispersion(ReturnsSeries(), PRDDA) / sqrt(size(XDDA, 1))
    e_d = PO.calibration_series_dispersion(AbsoluteDrawdownSeries(), PRDDA) /
          sqrt(size(XDDA, 1))
    z = Distributions.quantile(Distributions.Normal(), 0.95)
    @test DualNormRadius()(:r, PRDDA, nothing, nothing, CalibrationContext()) ≈
          z * norm(e_r, 1)
    ddctx = PO.CalibrationContext(; series = AbsoluteDrawdownSeries())
    @test DualNormRadius()(:r, PRDDA, nothing, nothing, ddctx) ≈ z * norm(e_d, 1)
    dn = DualNormRadius()
    @test dn(:l1, PRDDA, nothing, nothing, ddctx) ≈ z * norm(e_d, Inf)
    @test dn(:r, PRDDA, nothing, nothing, ddctx) > dn(:l1, PRDDA, nothing, nothing, ddctx)

    # A STATED scale is the whole of the units, so it overrides the reading and the marker
    # then changes nothing.
    for alg in (ConcentrationRadius(; scale = 0.03), DimensionalRateRadius(; scale = 0.03))
        @test alg(:r, PRDDA, nothing, nothing,
                  PO.CalibrationContext(; series = AbsoluteDrawdownSeries())) ==
              alg(:r, PRDDA, nothing, nothing, CalibrationContext())
    end

    # One context carries the marker and the order together, and neither displaces the
    # other. The rule keeps its own fields, because nothing rebuilds it.
    a = ConcentrationRadius(; confidence = 0.9, scale = 0.02)
    @test a.confidence == 0.9
    @test a.scale == 0.02
    @test !hasfield(ConcentrationRadius, :series)
    c = CalibrationContext(; series = AbsoluteDrawdownSeries(), p = 3)
    @test c.p == 3
    @test c.series == AbsoluteDrawdownSeries()
    @test isnothing(c.alpha)
    @test DualNormRadius(; confidence = 0.9)(:lpreg_val, PRDDA, nothing, nothing, c) ≈
          Distributions.quantile(Distributions.Normal(), 0.9) *
          norm(PO.calibration_series_dispersion(AbsoluteDrawdownSeries(), PRDDA) /
               sqrt(size(XDDA, 1)), 1.5)
end

@testset "TailTermParity: the drawdown owner prices a drawdown against a CDaR" begin
    alpha = 0.05
    ret = TailTermParity()
    dd_ctx = CalibrationContext(; alpha = alpha, series = AbsoluteDrawdownSeries())
    dd = (key, pr, w, slv, ctx = dd_ctx) -> ret(key, pr, w, slv, ctx)

    # The tail term of the drawdown reading IS the mean of the per-column CDaR. The CVaR
    # kernel over a non-positive drawdown column is that measure's own reading, so the rule
    # carries no second encoding of it.
    cdar = mean(j -> ConditionalDrawdownatRisk(; alpha = alpha)(view(XDDA, :, j)),
                axes(XDDA, 2))
    cvar_of_dd = mean(j -> ConditionalValueatRisk(; alpha = alpha)(view(DDA, :, j)),
                      axes(DDA, 2))
    @test cdar ≈ cvar_of_dd

    m = -mean(DDA)
    @test dd(:l, PRDDA, nothing, nothing) ≈ abs(m) / cdar

    # The defect #623 names: the returns reading UNDERSTATES the weight on this owner,
    # because the drawdown mean term is the far larger of the two relative to its own tail.
    @test dd(:l, PRDDA, nothing, nothing) >
          ret(:l, PRDDA, nothing, nothing, CalibrationContext(; alpha = alpha))

    # `ratio` scales the answer under every marker.
    @test TailTermParity(; ratio = 3)(:l, PRDDA, nothing, nothing, dd_ctx) ≈
          3 * dd(:l, PRDDA, nothing, nothing)

    # The two markers name two different series of one column, so the relative reading is
    # its own number.
    rel = ret(:l, PRDDA, nothing, nothing,
              CalibrationContext(; alpha = alpha, series = RelativeDrawdownSeries()))
    @test rel != dd(:l, PRDDA, nothing, nothing)

    # One context carries the significance level and the marker together, so there are no
    # two verbs to compose and no order between them to get wrong.
    bare = TailTermParity(; ratio = 2)
    both = CalibrationContext(; alpha = alpha, series = AbsoluteDrawdownSeries())
    @test bare.ratio == 2
    @test both.alpha == alpha
    @test both.series == AbsoluteDrawdownSeries()
    @test bare(:l, PRDDA, nothing, nothing, both) ≈ 2 * dd(:l, PRDDA, nothing, nothing)

    # The weighted reading moves to the drawdown sample too.
    w = pweights(range(; start = 1, stop = 2, length = size(XDDA, 1)))
    @test dd(:l, PRDDA, w, nothing) != dd(:l, PRDDA, nothing, nothing)
end

@testset "Ambiguity calibration: the drawdown owner hands its own series over" begin
    @test PO.calibration_series(DistributionallyRobustConditionalDrawdownatRisk()) ==
          AbsoluteDrawdownSeries()
    @test PO.calibration_series(DistributionallyRobustConditionalValueatRisk()) ==
          ReturnsSeries()

    lrole = TailTermParity()
    rrole = ConcentrationRadius()

    dr = PO.resolve_deferred_quantities(DistributionallyRobustConditionalDrawdownatRisk(;
                                                                                        alpha = 0.05,
                                                                                        l = lrole,
                                                                                        r = rrole),
                                        PRDDA)
    vr = PO.resolve_deferred_quantities(DistributionallyRobustConditionalValueatRisk(;
                                                                                     alpha = 0.05,
                                                                                     l = lrole,
                                                                                     r = rrole),
                                        PRDDA)

    @test dr.l ≈ TailTermParity()(:l, PRDDA, nothing, nothing,
                                  CalibrationContext(; alpha = 0.05,
                                                     series = AbsoluteDrawdownSeries()))
    @test dr.r ≈ ConcentrationRadius()(:r, PRDDA, nothing, nothing,
                                       CalibrationContext(; series = AbsoluteDrawdownSeries()))
    @test vr.l ≈
          TailTermParity()(:l, PRDDA, nothing, nothing, CalibrationContext(; alpha = 0.05))
    @test vr.r ≈ ConcentrationRadius()(:r, PRDDA, nothing, nothing, CalibrationContext())

    # The two owners resolve the SAME two keys and part company on the series alone.
    @test dr.l != vr.l
    @test dr.r != vr.r

    # NO MARKER CAN BE STATED ON THE RULE, so the measure that resolves it is the only
    # thing that names one and there is no precedence to get wrong.
    @test !hasfield(ConcentrationRadius, :series)
    o = PO.resolve_deferred_quantities(DistributionallyRobustConditionalValueatRisk(;
                                                                                    alpha = 0.05,
                                                                                    r = rrole),
                                       PRDDA)
    @test o.r ≈ vr.r

    # The Range measure prices ONE series over its two ends, so all four slots take the
    # same marker and the two ends part company through their probabilities alone.
    rg = PO.resolve_deferred_quantities(DistributionallyRobustConditionalValueatRiskRange(;
                                                                                          alpha = 0.05,
                                                                                          l_a = lrole,
                                                                                          r_a = rrole,
                                                                                          beta = 0.1,
                                                                                          l_b = lrole,
                                                                                          r_b = rrole),
                                        PRDDA)
    @test rg.r_a ≈ vr.r
    @test rg.r_b ≈ vr.r
    @test rg.l_a ≈ vr.l
    @test rg.l_b ≈
          TailTermParity()(:l_b, PRDDA, nothing, nothing, CalibrationContext(; alpha = 0.1))
    @test rg.l_a != rg.l_b

    # A measure whose slots all hold numbers is returned unchanged, so the new call
    # allocates no rebuild in the common case.
    plain = DistributionallyRobustConditionalDrawdownatRisk()
    @test PO.resolve_deferred_quantities(plain, PRDDA) === plain
end

#=
Issues #619, #620 and #621 sweep the three rules of this file that shipped after the first
two. Each ticket asks the same kind of question: whether a reading the rule takes is the
right one, and whether a correction or a guard belongs in the rule. The answers are in the
three docstrings, and the numbers that back them are here.

Every case below is a reading the sweep SETTLED rather than a mechanism it added, so the
tests pin numbers and refusals rather than types.
=#
@testset "Sweep #619, #620, #621: the readings the three rules settle on" begin
    lq = log(1 / 0.05)
    alpha = 0.05

    # -- #619. The exponent is floored at one half, so a ONE-asset universe returns the
    # same square-root rate a two-asset one does. The bound states no rate above the floor.
    X60_1 = X60_H[:, 1:1]
    PR60_1 = prior(EmpiricalPrior(), X60_1)
    dim = DimensionalRateRadius(; confidence = 0.95, scale = 0.5)
    @test size(PR60_1.X, 2) == 1
    @test dim(:r, PR60_1, nothing, nothing, CalibrationContext()) ≈ 0.5 * sqrt(lq / 60)
    @test dim(:r, PR60_1, nothing, nothing, CalibrationContext()) ≈
          dim(:r, PR60_2, nothing, nothing, CalibrationContext())

    # -- #619. The radius is positive and finite at both ends of the admissible confidence
    # interval, so nothing re-checks it after resolution.
    for c in (1e-8, 0.5, 1 - 1e-8)
        v = DimensionalRateRadius(; confidence = c, scale = 1)(:r, PR60_H, nothing, nothing,
                                                               CalibrationContext())
        @test v > 0
        @test isfinite(v)
    end

    # -- #620. A sample of NO dispersion gives a ball of no width, which is the right
    # answer for it: the empirical measure is the only measure such a sample supports. In
    # floating point the number is many orders below the smallest one a caller would state
    # rather than an exact zero, so the slot's `> 0` check passes and the model prices it.
    Xflat = repeat([0.01 0.02 0.03], 60)
    PRFLAT = prior(EmpiricalPrior(), Xflat)
    vflat = DualNormRadius()(:r, PRFLAT, nothing, nothing, CalibrationContext())
    @test vflat >= 0
    @test vflat < 1e-12
    @test isa(DistributionallyRobustConditionalValueatRisk(; r = vflat).r, Number)

    # -- #620. `confidence` is a per-coordinate level and the rule does not correct it for
    # the number of assets. The correction is a whole number of per cent rather than a
    # rounding, which is why it stays with the caller and is named in prose.
    N = size(X60_H, 2)
    z_coord = Distributions.quantile(Distributions.Normal(), 0.95)
    z_vec = Distributions.quantile(Distributions.Normal(), 1 - 0.05 / N)
    @test z_vec / z_coord > 1.3
    @test DualNormRadius(; confidence = 0.95)(:l1, PR60_H, nothing, nothing,
                                              CalibrationContext()) * z_vec / z_coord ≈
          DualNormRadius(; confidence = 1 - 0.05 / N)(:l1, PR60_H, nothing, nothing,
                                                      CalibrationContext())

    # -- #620. The 1-norm arm sums the per-asset errors, so it prices them as if they moved
    # together. That is the LARGEST of the seven readings, and the ∞-norm arm is the
    # smallest, so the conservative reading is the one the rule takes.
    e = sqrt.(diag(PR60_H.sigma)) ./ sqrt(60)
    @test DualNormRadius()(:r, PR60_H, nothing, nothing, CalibrationContext()) ≈
          z_coord * sum(e)
    @test DualNormRadius()(:r, PR60_H, nothing, nothing, CalibrationContext()) >
          DualNormRadius()(:l2reg_val, PR60_H, nothing, nothing, CalibrationContext())
    @test DualNormRadius()(:l2reg_val, PR60_H, nothing, nothing, CalibrationContext()) >
          DualNormRadius()(:l1, PR60_H, nothing, nothing, CalibrationContext())
    @test DualNormRadius()(:l1, PR60_H, nothing, nothing, CalibrationContext()) ≈
          z_coord * maximum(e)

    # -- #621. `c` is the mean of the per-column CVaR and NOT the pooled one. A pooled tail
    # is drawn from the worst columns, so on a universe of unequal volatilities it is the
    # larger number and it would give the SMALLER weight.
    Xuneq = hcat(0.0005 .+ 0.002 * randn(RNG_H, 2520), 0.0005 .+ 0.01 * randn(RNG_H, 2520),
                 0.0005 .+ 0.05 * randn(RNG_H, 2520))
    PRUNEQ = prior(EmpiricalPrior(), Xuneq)
    c_col = mean(hand_cvar(view(Xuneq, :, j), alpha) for j in axes(Xuneq, 2))
    c_pool = hand_cvar(vec(Xuneq), alpha)
    @test c_pool > c_col
    l_col = TailTermParity(; ratio = 1)(:l, PRUNEQ, nothing, nothing,
                                        CalibrationContext(; alpha = alpha))
    @test l_col ≈ abs(-mean(Xuneq)) / c_col
    @test l_col > abs(-mean(Xuneq)) / c_pool

    # -- #621. The parity claim, against the loss the measure prices. At the weight the
    # rule returns, one tail term is worth exactly `ratio` mean terms.
    m = -mean(XDAY)
    c = mean(hand_cvar(view(XDAY, :, j), alpha) for j in axes(XDAY, 2))
    for ratio in (0.5, 1.0, 2.0)
        l = TailTermParity(; ratio = ratio)(:l, PRDAY, nothing, nothing,
                                            CalibrationContext(; alpha = alpha))
        @test l * c / abs(m) ≈ ratio
    end

    # -- #621. A stated `l = 1.0` is two trade-offs across the frequency pair: about forty
    # mean terms on the daily sample and under ten on the monthly one, the two numbers the
    # docstring names.
    l_day = TailTermParity(; ratio = 1)(:l, PRDAY, nothing, nothing,
                                        CalibrationContext(; alpha = alpha))
    l_mon = TailTermParity(; ratio = 1)(:l, PRMON, nothing, nothing,
                                        CalibrationContext(; alpha = alpha))
    @test 35 < inv(l_day) < 50
    @test inv(l_mon) < 10
end

#=
The effective sample size the four record-reading rules divide by. It was written out four
times, and each copy summed `w` itself rather than resolving it. The `w` a rule receives is
`sel(x.w, pr.w)`, and both fields are bounded `Option{<:ObsWeights}`, so a
`DynamicAbstractWeights` reaches the rule. `sum` states no method for one, so the four
copies raised a bare `MethodError` where `TailTermParity`, which resolves its weights,
raised the library's own `ObservationWeightsError`. `effective_sample_size` is the one
definition, and it resolves the weights.
=#

# A `DynamicAbstractWeights` that states the vector arity, which is the arity a column of
# the sample takes.
struct CalibrationEWeights <: PortfolioOptimisers.DynamicAbstractWeights end
function PortfolioOptimisers.get_observation_weights(::CalibrationEWeights,
                                                     X::PortfolioOptimisers.VecNum;
                                                     kwargs...)
    return pweights(range(; start = 1, stop = 2, length = length(X)))
end

# A `DynamicAbstractWeights` that states no arity at all, which is the refusal under test.
struct CalibrationNoWeights <: PortfolioOptimisers.DynamicAbstractWeights end

@testset "effective_sample_size: one definition, and the weights union read once" begin
    # No weights is the row count of the record, and stated weights are Kish's count.
    @test PO.effective_sample_size(PR60_H, nothing) == 60
    @test PO.effective_sample_size(PR120_H, nothing) == 120
    @test PO.effective_sample_size(PR60_H, WTS) ≈ sum(WTS)^2 / sum(abs2, WTS)
    @test PO.effective_sample_size(PR60_H, WTS) < 60

    # -- The four rules read the one definition, so each returns what it returned before.
    q = Distributions.cquantile(Distributions.Chisq(4), 0.05)
    kish = sum(WTS)^2 / sum(abs2, WTS)
    @test ConcentrationRadius(; confidence = 0.95, scale = 0.5)(:r, PR60_H, WTS, nothing,
                                                                CalibrationContext()) ≈
          0.5 * sqrt(q / kish)
    @test ScenarioCount(; n = 6)(:alpha, PR60_H, WTS, nothing, CalibrationContext()) ≈
          6 / kish
    @test DimensionalRateRadius(; confidence = 0.95, scale = 0.5)(:r, PR60_H, WTS, nothing,
                                                                  CalibrationContext()) ≈
          0.5 * (log(inv(0.05)) / kish)^inv(4)
    @test DualNormRadius(; confidence = 0.95)(:r, PR60_H, WTS, nothing,
                                              CalibrationContext()) >
          DualNormRadius(; confidence = 0.95)(:r, PR60_H, nothing, nothing,
                                              CalibrationContext())

    # -- A `DynamicAbstractWeights` that states the vector arity now resolves, in every one
    # of the four rules, to the count its weights carry.
    dyn = CalibrationEWeights()
    dyn_kish = sum(WTS)^2 / sum(abs2, WTS)
    @test PO.effective_sample_size(PR60_H, dyn) ≈ dyn_kish
    @test ScenarioCount(; n = 6)(:alpha, PR60_H, dyn, nothing, CalibrationContext()) ≈
          6 / dyn_kish
    @test ConcentrationRadius(; confidence = 0.95, scale = 0.5)(:r, PR60_H, dyn, nothing,
                                                                CalibrationContext()) ≈
          0.5 * sqrt(q / dyn_kish)
    @test DimensionalRateRadius(; confidence = 0.95, scale = 0.5)(:r, PR60_H, dyn, nothing,
                                                                  CalibrationContext()) ≈
          0.5 * (log(inv(0.05)) / dyn_kish)^inv(4)
    @test DualNormRadius(; confidence = 0.95)(:r, PR60_H, dyn, nothing,
                                              CalibrationContext()) ≈
          DualNormRadius(; confidence = 0.95)(:r, PR60_H, WTS, nothing,
                                              CalibrationContext())

    # -- The defect. A `DynamicAbstractWeights` that states no arity raises the library's
    # own named error in all four rules. Before this change, only `TailTermParity` did, and
    # the four copies raised a bare `MethodError` off `sum`.
    none = CalibrationNoWeights()
    @test_throws PO.ObservationWeightsError PO.effective_sample_size(PR60_H, none)
    @test_throws PO.ObservationWeightsError ScenarioCount(; n = 6)(:alpha, PR60_H, none,
                                                                   nothing,
                                                                   CalibrationContext())
    @test_throws PO.ObservationWeightsError ConcentrationRadius(; confidence = 0.95,
                                                                scale = 0.5)(:r, PR60_H,
                                                                             none, nothing,
                                                                             CalibrationContext())
    @test_throws PO.ObservationWeightsError DimensionalRateRadius(; confidence = 0.95,
                                                                  scale = 0.5)(:r, PR60_H,
                                                                               none,
                                                                               nothing,
                                                                               CalibrationContext())
    @test_throws PO.ObservationWeightsError DualNormRadius(; confidence = 0.95)(:r, PR60_H,
                                                                                none,
                                                                                nothing,
                                                                                CalibrationContext())
    err = try
        ScenarioCount(; n = 6)(:alpha, PR60_H, none, nothing, CalibrationContext())
    catch e
        e
    end
    @test occursin("CalibrationNoWeights is a DynamicAbstractWeights", err.msg)
end

#=
The two bare radius fields of the optimiser, `l1` and `linf`, hold the number itself rather
than a term, so no rebuild re-runs a constructor's check on a calibrated radius. The check
therefore belongs to `resolve_calibration_slot`, which is the one place the rule runs, and
the rule's own method there refuses the values a stated radius is refused. It is paid on
the calibrated path alone: a stated number takes the fallback method and meets no check.
=#
@testset "Ambiguity calibration: a calibrated radius meets the stated radius's check" begin
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = "verbose" => false)
    rd = ReturnsResult(; nx = string.(1:4), X = X60_H)
    function assemble_with(opt)
        mr = MeanRisk(; r = Variance(), opt = opt)
        attrs = PO.processed_jump_optimiser_attributes(mr.opt, rd)
        model = JuMP.Model()
        PO.set_model_scales!(model, mr.opt.sc, mr.opt.so)
        PO.set_model_observations!(model, size(attrs.pr.X, 1))
        PO.set_maximum_ratio_factor_variables!(model, mr.obj)
        PO.set_w!(model, attrs.pr.X, mr.wi)
        PO.set_weight_constraints!(model, attrs.wb, mr.opt)
        PO.assemble_jump_model!(model, mr, mr.opt, attrs, rd, mr.r, mr.obj)
        return model
    end

    # A stated radius of each shape is refused where the caller wrote it.
    @test_throws DomainError JuMPOptimiser(; slv = slv, pe = PR60_H, l1 = -1.0)
    @test_throws DomainError JuMPOptimiser(; slv = slv, pe = PR60_H, l1 = 0.0)
    @test_throws DomainError JuMPOptimiser(; slv = slv, pe = PR60_H, linf = NaN)
    @test_throws DomainError JuMPOptimiser(; slv = slv, pe = PR60_H, linf = Inf)

    # The same four numbers, computed by a rule of the radius family, are refused by the
    # resolver, under the key of the slot the rule stands in.
    for bad in (-1.0, 0.0, NaN, Inf)
        rrule = ProbeRadiusValue(bad)
        for key in (:l1, :linf, :r, :l2reg_val, :lpreg_val)
            @test_throws DomainError PO.resolve_calibration_slot(rrule, key, PR60_H,
                                                                 PR60_H.w, slv)
        end
        err = try
            PO.resolve_calibration_slot(rrule, :l1, PR60_H, PR60_H.w, slv)
        catch e
            e
        end
        @test occursin("l1", sprint(showerror, err))

        # So the whole route refuses it too, at assembly and at the term's own factory.
        @test_throws DomainError assemble_with(JuMPOptimiser(; slv = slv, pe = PR60_H,
                                                             l1 = rrule))
        @test_throws DomainError assemble_with(JuMPOptimiser(; slv = slv, pe = PR60_H,
                                                             linf = rrule))
        @test_throws DomainError PO.factory(L2Regularisation(; val = rrule), PR60_H)
        @test_throws DomainError PO.factory(LpRegularisation(; val = rrule), PR60_H)

        # A rule written as a plain function names no family, so no method of the resolver
        # can read the quantity it computes and the resolver returns the number unchecked.
        # `assemble_jump_model!` is what states the range of the four slots that reach the
        # model raw, so the route refuses it there instead.
        fn = (key, pr, w, slv, ctx) -> bad
        @test PO.resolve_calibration_slot(fn, :l1, PR60_H, PR60_H.w, slv) === bad
        @test_throws DomainError assemble_with(JuMPOptimiser(; slv = slv, pe = PR60_H,
                                                             l1 = fn))
        @test_throws DomainError assemble_with(JuMPOptimiser(; slv = slv, pe = PR60_H,
                                                             linf = fn))
    end

    # A stated number reaches the fallback method, which checks nothing: the cost is paid
    # on the calibrated path alone, and a number the constructor already admitted is not
    # measured twice.
    @test PO.resolve_calibration_slot(1e-3, :l1, PR60_H, PR60_H.w, slv) == 1e-3

    # A rule that answers a legal radius still reaches the model, so the check refuses
    # nothing a caller could have stated by hand.
    good = (key, pr, w, slv, ctx) -> 1e-3
    model = assemble_with(JuMPOptimiser(; slv = slv, pe = PR60_H, l1 = good, linf = good))
    @test JuMP.coefficient(model[:l1], model[:t_l1]) ≈ 1e-3
    @test JuMP.coefficient(model[:linf], model[:t_linf]) ≈ 1e-3
end

#=
A **Norm Ceiling** is the upper bound on a norm of the weight vector, the quantity the
`l2c`, `lpc` and `linfc` fields of `JuMPOptimiser` hold. It is NOT an **Ambiguity Radius**.
A radius is the coefficient of a norm penalty in the objective; a ceiling bounds that norm
in a constraint. The reciprocal of a ceiling is a floor on the effective number of assets,
which is a diversification statement rather than a statement about the set of measures the
model prices.

ADR 0095 rules that the rule names the quantity, so the ceiling takes its own family, its
own rule and its own bound rather than borrowing the ambiguity-radius family.
`test_09f_calibration_slot.jl` covers the radius family this one sits beside.

One field carries both readings. `LpRegularisation` is a penalty in `JuMPOptimiser.lp` and
a norm constraint in `JuMPOptimiser.lpc`, so its `val` cannot be bounded to one rule. The
bound admits both, and the FIELD THAT HOLDS THE TERM refuses the rule that has no reading
there. That is the one place in the library where a rule is settled after construction.
=#
using Clarabel, JuMP, InteractiveUtils
const PO = PortfolioOptimisers

const RNG_N = StableRNG(246813579)
const X60_N = randn(RNG_N, 60, 4)
const PR60_N = prior(EmpiricalPrior(), X60_N)
const RD60 = ReturnsResult(; nx = string.(1:4), X = X60_N)
const SLV = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                   settings = "verbose" => false)

# A ceiling rule with no type at all. Every ceiling slot admits a bare `Function` for
# exactly this case, and this one reports its arguments so the resolver's order is asserted.
const CEIL_SEEN = Ref{Any}(nothing)
function probe_ceiling(key::Symbol, pr::PO.AbstractPriorResult, w, slv,
                       ctx::PO.CalibrationContext)
    CEIL_SEEN[] = (; key = key, weighted = !isnothing(w), solved = !isnothing(slv),
                   order = ctx.p)
    return 1 / sqrt(size(pr.X, 2))
end

# A ceiling rule that answers a stated number. A rule of the family is what the resolver's
# range check dispatches on, so a test of that check needs a family and not a closure.
struct ProbeCeilingValue{T} <: PO.AbstractNormCeilingCalibrationAlgorithm
    val::T
end
function (alg::ProbeCeilingValue)(::Symbol, ::PO.AbstractPriorResult, ::Any, ::Any,
                                  ::PO.CalibrationContext)
    return alg.val
end

@testset "Norm ceiling calibration: the family joins the calibration root" begin
    # The family sits under the one root, beside the three the mechanism already carried.
    @test PO.AbstractNormCeilingCalibrationAlgorithm <: PO.AbstractCalibrationAlgorithm

    # It is none of the other three. A ceiling is a different quantity from a radius, and
    # that is the whole reason it takes a family of its own.
    @test !(PO.AbstractNormCeilingCalibrationAlgorithm <:
            PO.AbstractAmbiguityRadiusCalibrationAlgorithm)
    @test !(PO.AbstractAmbiguityRadiusCalibrationAlgorithm <:
            PO.AbstractNormCeilingCalibrationAlgorithm)
    @test !(PO.AbstractNormCeilingCalibrationAlgorithm <:
            PO.AbstractSignificanceCalibrationAlgorithm)

    # #641 withdrew the rule that used to place a rule in a ceiling slot, so nothing
    # stands between the caller and the rule.
    @test !isdefined(PortfolioOptimisers, :NormCeilingCalibration)
    @test EffectiveAssetFloor <: PO.AbstractNormCeilingCalibrationAlgorithm

    # The abstract type is not exported: an export is public API, and the convention is
    # that an abstract type is not one. It is `public`, which `names` also lists, so the
    # check reads `Base.isexported`.
    @test !Base.isexported(PortfolioOptimisers, :AbstractNormCeilingCalibrationAlgorithm)

    # The rule is exported, on the same terms as the radius family's.
    @test :EffectiveAssetFloor ∈ names(PortfolioOptimisers)

    # A ceiling names no end of a distribution, so no ceiling slot defaults from another:
    # each of the three states its own occupant.
    slv = Solver(; name = :probe, solver = nothing)
    @test isnothing(JuMPOptimiser(; slv = slv, l2c = EffectiveAssetFloor()).linfc)
end

@testset "Norm ceiling calibration: the bounds refuse the wrong family" begin
    crule = EffectiveAssetFloor()
    rrule = RateRadius(; c = 0.2)

    # The slot bound pairs `Number` with ONE rule family and a plain function, so the
    # refusal is at construction and no guard method is written for it.
    @test isa(1.0, PO.Num_NormCeilCal)
    @test isa(crule, PO.Num_NormCeilCal)
    @test isa(probe_ceiling, PO.Num_NormCeilCal)
    @test !isa(rrule, PO.Num_NormCeilCal)
    @test !isa(crule, PO.Num_AmbRadCal)

    # A radius rule in a ceiling slot is refused, and a ceiling rule in a radius slot too.
    slv = Solver(; name = :probe, solver = nothing)
    @test_throws TypeError JuMPOptimiser(; slv = slv, l2c = rrule)
    @test_throws TypeError JuMPOptimiser(; slv = slv, l1 = crule)

    # `LpRegularisation.val` is the one dual-use slot, so its bound admits both families.
    # It is also the one slot that admits no plain function: a function names no family,
    # and the two guards of that slot read the family.
    @test isa(crule, PO.Num_AmbRadNormCeilCal)
    @test isa(rrule, PO.Num_AmbRadNormCeilCal)
    @test isa(1.0, PO.Num_AmbRadNormCeilCal)
    @test !isa(probe_ceiling, PO.Num_AmbRadNormCeilCal)
    @test_throws TypeError LpRegularisation(; val = probe_ceiling)
end

@testset "Norm ceiling calibration: EffectiveAssetFloor reads the universe" begin
    # `fraction` is checked, and it is a fraction rather than a count.
    @test EffectiveAssetFloor().fraction == 0.5
    @test !hasfield(EffectiveAssetFloor, :p)
    @test_throws DomainError EffectiveAssetFloor(; fraction = 0)
    @test_throws DomainError EffectiveAssetFloor(; fraction = -0.1)
    @test_throws DomainError EffectiveAssetFloor(; fraction = 1.5)
    @test_throws DomainError EffectiveAssetFloor(; fraction = Inf)
    @test EffectiveAssetFloor(; fraction = 1).fraction == 1

    # The ceiling guard names the quantity, the value and the range, rather than printing
    # the source expression it was written from.
    msg = try
        EffectiveAssetFloor(; fraction = 1.5)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("EffectiveAssetFloor.fraction", msg)
    @test occursin("1.5", msg)
    @test occursin("(0, 1]", msg)
    @test !occursin("one(fraction)", msg)

    # A norm order below 1 is not a norm order. The order lives in the context now, so
    # the rule that reads it is what refuses one.
    @test_throws DomainError EffectiveAssetFloor(; fraction = 0.5)(:l2c, PR60_N, nothing,
                                                                   nothing,
                                                                   CalibrationContext(;
                                                                                      p = 0.5))
    @test EffectiveAssetFloor(; fraction = 0.5)(:lpc, PR60_N, nothing, nothing,
                                                CalibrationContext(; p = 3)) ≈ 2^(1 / 3 - 1)

    # The prior carries four assets, so half of the universe is two effective assets. The
    # ceiling is the order-`p` effective-asset reading the constraint's own docstring
    # states: `m^(1/p - 1)` for a finite order, and `1/m` for the infinite one.
    f2 = EffectiveAssetFloor(; fraction = 0.5)
    f3 = EffectiveAssetFloor(; fraction = 0.5)
    fi = EffectiveAssetFloor(; fraction = 0.5)
    @test f2(:l2c, PR60_N, PR60_N.w, nothing, CalibrationContext(; p = 2)) ≈ 2^(-1 / 2)
    @test f3(:lpc, PR60_N, PR60_N.w, nothing, CalibrationContext(; p = 3)) ≈ 2^(1 / 3 - 1)
    @test fi(:linfc, PR60_N, PR60_N.w, nothing, CalibrationContext(; p = Inf)) ≈ 1 / 2

    # The whole universe is the whole universe, so an equally weighted portfolio is the
    # only point the 2-norm ceiling admits.
    @test EffectiveAssetFloor(; fraction = 1)(:l2c, PR60_N, nothing, nothing,
                                              CalibrationContext(; p = 2)) ≈ 1 / 2

    # The floor moves with the universe rather than with a count the caller pinned. That is
    # the whole reason a rule beats a number.
    pr2 = prior(EmpiricalPrior(), randn(RNG_N, 60, 9))
    @test f2(:l2c, pr2, nothing, nothing, CalibrationContext(; p = 2)) ≈ (0.5 * 9)^(-1 / 2)

    # A universe count is not a sample count, so the rule ignores the observation weights.
    wts = pweights(range(; start = 1, stop = 2, length = 60))
    @test f2(:l2c, PR60_N, wts, nothing, CalibrationContext(; p = 2)) ==
          f2(:l2c, PR60_N, nothing, nothing, CalibrationContext(; p = 2))

    # The order belongs to the constraint. A rule resolved against a context that names
    # no order says which field it wanted.
    @test_throws ArgumentError EffectiveAssetFloor()(:l2c, PR60_N, nothing, nothing,
                                                     CalibrationContext())
end

#=
Issue #617 sweeps the family, and the exponent of the finite arm is what it corrected. The
first draft read the count as `1 / ||w||_p^p`, which is the same number as
`number_effective_assets` at `p = 2` and is not an effective count anywhere else: an
equal-weight portfolio over ten assets reports ONE HUNDRED at `p = 3`, a count above the
size of the universe. The order-`p` count is `(sum |w_i|^p)^(1/(1 - p))`, and its ceiling
is `m^(1/p - 1)`.

The three properties below are what separate the two readings, and no one of them alone
does it: the first holds for both at `p = 2`, and the second is what the third then joins
to the infinite arm.
=#
@testset "Norm ceiling calibration: the ceiling is the order-p effective count" begin
    # The order-`p` effective number of assets, written out. It is the reading
    # `number_effective_assets` states, taken to an arbitrary order.
    ena(w, p) = isinf(p) ? inv(maximum(abs, w)) : sum(abs.(w) .^ p)^inv(1 - p)

    # 1. An equal-weight portfolio over `m` assets reports EXACTLY `m`, at every order.
    for p in (1.5, 2.0, 3.0, 5.0, Inf), m in (2, 4, 5)
        @test ena(fill(1 / m, m), p) ≈ m
    end

    # 2. The rule's ceiling is the point where that count equals the floor, so an
    #    equal-weight portfolio over exactly `m` assets sits ON the ceiling, and one over
    #    fewer breaks it. The universe is four assets and the fraction is one half, so
    #    `m = 2` here and `pr9` below carries nine.
    pr9 = prior(EmpiricalPrior(), randn(RNG_N, 60, 9))
    for p in (1.5, 2.0, 3.0, Inf)
        alg = EffectiveAssetFloor(; fraction = 1 / 3)
        val = alg(:lpc, pr9, nothing, nothing, CalibrationContext(; p = p))
        @test ena(fill(1 / 3, 3), p) ≈ 3
        @test LinearAlgebra.norm(fill(1 / 3, 3), p) ≈ val
        @test LinearAlgebra.norm(fill(1 / 2, 2), p) > val
        @test LinearAlgebra.norm(fill(1 / 4, 4), p) < val
    end

    # 3. The infinite arm is the LIMIT of the finite one rather than a second reading, so
    #    the ceiling moves towards `1/m` as the order grows instead of away from it.
    m = 2
    fi = EffectiveAssetFloor(; fraction = 0.5)(:linfc, PR60_N, nothing, nothing,
                                               CalibrationContext(; p = Inf))
    @test fi ≈ inv(m)
    prev = Inf
    for p in (2.0, 5.0, 20.0, 100.0, 1000.0)
        val = EffectiveAssetFloor(; fraction = 0.5)(:lpc, PR60_N, nothing, nothing,
                                                    CalibrationContext(; p = p))
        @test val > fi
        @test val < prev
        prev = val
    end
    far = EffectiveAssetFloor(; fraction = 0.5)(:lpc, PR60_N, nothing, nothing,
                                                CalibrationContext(; p = 1e6))
    @test isapprox(far, fi; rtol = 1e-5)

    # At `p = 2` the two readings are one number, which is why the defect survived the
    # first draft: every test of the family stood at that order.
    @test EffectiveAssetFloor(; fraction = 0.5)(:l2c, PR60_N, nothing, nothing,
                                                CalibrationContext(; p = 2)) ≈ 2^(-1 / 2)
    @test 2^(1 / 2 - 1) ≈ 2^(-1 / 2)

    # A 1-norm ceiling is one, which is the budget itself: the 1-norm of a fully invested
    # long-only portfolio is one whatever it holds, so no floor on it binds.
    @test EffectiveAssetFloor(; fraction = 0.5)(:lpc, PR60_N, nothing, nothing,
                                                CalibrationContext(; p = 1)) ≈ 1
end

@testset "Norm ceiling calibration: the context carries the constraint's order" begin
    crule = EffectiveAssetFloor(; fraction = 0.5)
    ctx3 = CalibrationContext(; p = 3)

    # A stated number ignores the context, and so does a caller's own plain function that
    # reads no order. Nothing is rebuilt on the way in.
    @test PO.resolve_calibration_slot(0.4, :l2c, PR60_N, nothing, nothing, ctx3) == 0.4
    @test isnothing(PO.resolve_calibration_slot(nothing, :l2c, PR60_N, nothing, nothing,
                                                ctx3))
    # The order reaches the rule through the context, and the rule itself never moves:
    # the resolver calls it and hands the context on.
    @test PO.resolve_calibration_slot(crule, :lpc, PR60_N, nothing, nothing, ctx3) ≈
          crule(:lpc, PR60_N, nothing, nothing, ctx3)
    @test crule.fraction == 0.5
    @test !hasfield(EffectiveAssetFloor, :p)

    # No rule carries an order of its own, so the site's order is the only one there is.
    # One rule therefore serves the three sites, and each reads it at its own order.
    eaf = EffectiveAssetFloor(; fraction = 0.5)
    @test eaf(:l2c, PR60_N, nothing, nothing, CalibrationContext(; p = 2)) !=
          eaf(:linfc, PR60_N, nothing, nothing, CalibrationContext(; p = Inf))

    # Resolving against the context is the whole of the channel, and the key and the order
    # both reach the rule.
    CEIL_SEEN[] = nothing
    @test PO.resolve_calibration_slot(probe_ceiling, :l2c, PR60_N, PR60_N.w, SLV,
                                      CalibrationContext(; p = 2)) ≈ 1 / 2
    @test CEIL_SEEN[].key == :l2c
    @test CEIL_SEEN[].solved
    @test CEIL_SEEN[].order == 2
end

@testset "Norm ceiling calibration: the LpRegularisation route settles the reading" begin
    crule = EffectiveAssetFloor(; fraction = 0.5)
    rrule = RateRadius(; c = 0.2)

    # One term, one bound, two readings. The constructor admits both rules because it
    # cannot know which field the term is about to land in.
    @test isa(LpRegularisation(; p = 3, val = crule).val,
              PO.AbstractNormCeilingCalibrationAlgorithm)
    @test isa(LpRegularisation(; p = 3, val = rrule).val,
              PO.AbstractAmbiguityRadiusCalibrationAlgorithm)

    # A ceiling has no reading as a penalty coefficient, so the penalty route refuses it.
    # A number stays legal on both routes.
    @test isnothing(PO.assert_penalty_coefficient_role(1e-3))
    @test isnothing(PO.assert_penalty_coefficient_role(LpRegularisation(; val = rrule)))
    @test isnothing(PO.assert_penalty_coefficient_role([LpRegularisation(; val = rrule)]))
    @test_throws ArgumentError PO.assert_penalty_coefficient_role(crule)
    @test_throws ArgumentError PO.assert_penalty_coefficient_role(LpRegularisation(;
                                                                                   val = crule))
    @test_throws ArgumentError PO.assert_penalty_coefficient_role([LpRegularisation(;
                                                                                    val = crule)])

    # A radius has no reading as a ceiling, so the constraint route refuses it.
    @test isnothing(PO.assert_norm_ceiling_role(1e-3))
    @test isnothing(PO.assert_norm_ceiling_role(LpRegularisation(; val = crule)))
    @test isnothing(PO.assert_norm_ceiling_role([LpRegularisation(; val = crule)]))
    @test_throws ArgumentError PO.assert_norm_ceiling_role(rrule)
    @test_throws ArgumentError PO.assert_norm_ceiling_role(LpRegularisation(; val = rrule))
    @test_throws ArgumentError PO.assert_norm_ceiling_role([LpRegularisation(; val = rrule)])

    # `norm_ceiling_factory` is the constraint route's own verb. It binds the TERM's order,
    # so one rule serves several terms that carry different orders.
    terms = [LpRegularisation(; p = 2, val = crule), LpRegularisation(; p = 3, val = crule)]
    out = PO.norm_ceiling_factory(terms, PR60_N, SLV)
    @test out[1].val ≈ 2^(-1 / 2)
    @test out[2].val ≈ 2^(1 / 3 - 1)
    @test out[1].p == 2
    @test out[2].p == 3

    # A stated number is carried through by identity rather than rebuilt, and the fallback
    # returns anything else unchanged.
    stated = LpRegularisation(; p = 3, val = 0.6)
    @test PO.norm_ceiling_factory(stated, PR60_N, SLV) === stated
    @test isnothing(PO.norm_ceiling_factory(nothing, PR60_N, SLV))

    # The penalty route refuses on its own, so a term that reaches the objective by another
    # path is caught there too.
    @test_throws ArgumentError PO.factory(LpRegularisation(; val = crule), PR60_N, SLV)
    @test PO.factory(LpRegularisation(; p = 3, val = rrule), PR60_N, SLV).val ≈
          0.2 / sqrt(60)
end

@testset "Norm ceiling calibration: JuMPOptimiser refuses the wrong rule per field" begin
    crule = EffectiveAssetFloor()
    rrule = RateRadius(; c = 0.2)

    # `l2c` and `linfc` are bounded to the ceiling rule alone, so a radius is refused by the
    # signature itself, at the point where the caller wrote it.
    @test_throws TypeError JuMPOptimiser(; slv = SLV, l2c = rrule)
    @test_throws TypeError JuMPOptimiser(; slv = SLV, linfc = rrule)

    # A plain function carries no family, so the bound admits one and the name of the slot
    # is what states the quantity there.
    @test JuMPOptimiser(; slv = SLV, l2c = probe_ceiling).l2c === probe_ceiling

    # `lp` and `lpc` share one term type, so the FIELD refuses the rule that has no reading
    # in it. This is the first point at which the reading is known.
    @test_throws ArgumentError JuMPOptimiser(; slv = SLV,
                                             lp = LpRegularisation(; val = crule))
    @test_throws ArgumentError JuMPOptimiser(; slv = SLV,
                                             lpc = LpRegularisation(; val = rrule))
    @test_throws ArgumentError JuMPOptimiser(; slv = SLV,
                                             lp = [LpRegularisation(; val = crule)])

    # Every correct pairing is accepted, and the rules reach the fields untouched.
    opt = JuMPOptimiser(; slv = SLV, pe = PR60_N, l2c = crule, linfc = crule,
                        lpc = LpRegularisation(; p = 3, val = crule),
                        lp = LpRegularisation(; p = 3, val = rrule), l1 = rrule)
    @test opt.l2c === crule
    @test opt.linfc === crule
    @test opt.lpc.val === crule

    # Neither the weights factory nor the cluster slice holds a prior result, so both carry
    # a rule through untouched: it resolves against the cluster's own prior at assembly.
    @test PO.factory(opt, fill(0.25, 4)).l2c === crule
    @test PO.port_opt_view(opt, 1:3, X60_N).linfc === crule
end

@testset "Norm ceiling calibration: assembly reaches the stated number" begin
    crule = EffectiveAssetFloor(; fraction = 0.5)
    function build(opt)
        mr = MeanRisk(; r = Variance(), opt = opt)
        attrs = PO.processed_jump_optimiser_attributes(mr.opt, RD60)
        model = JuMP.Model()
        PO.set_model_scales!(model, mr.opt.sc, mr.opt.so)
        PO.set_model_observations!(model, size(attrs.pr.X, 1))
        PO.set_maximum_ratio_factor_variables!(model, mr.obj)
        PO.set_w!(model, attrs.pr.X, mr.wi)
        PO.set_weight_constraints!(model, attrs.wb, mr.opt)
        PO.assemble_jump_model!(model, mr, mr.opt, attrs, RD60, mr.r, mr.obj)
        return model
    end

    # The rule reaches the three constraints as the number a caller would have written by
    # hand. Four assets and a half-universe floor make two effective assets.
    mrule = build(JuMPOptimiser(; slv = SLV, pe = PR60_N, l2c = crule, linfc = crule,
                                lpc = LpRegularisation(; p = 3, val = crule)))
    mnum = build(JuMPOptimiser(; slv = SLV, pe = PR60_N, l2c = 2^(-1 / 2), linfc = 1 / 2,
                               lpc = LpRegularisation(; p = 3, val = 2^(1 / 3 - 1))))
    @test JuMP.normalized_rhs(mrule[:cl2c]) ≈ JuMP.normalized_rhs(mnum[:cl2c])
    @test JuMP.normalized_rhs(mrule[:clinfc]) ≈ JuMP.normalized_rhs(mnum[:clinfc])
    @test JuMP.normalized_rhs(mrule[:clpc_bnd_1]) ≈ JuMP.normalized_rhs(mnum[:clpc_bnd_1])
    @test JuMP.normalized_rhs(mrule[:cl2c]) ≈ 2^(-1 / 2)
    @test JuMP.normalized_rhs(mrule[:clinfc]) ≈ 1 / 2
    @test JuMP.normalized_rhs(mrule[:clpc_bnd_1]) ≈ 2^(1 / 3 - 1)

    # A model that names no ceiling carries no ceiling constraint, so the widening added no
    # constraint to a caller who states nothing.
    plain = build(JuMPOptimiser(; slv = SLV, pe = PR60_N))
    @test !haskey(JuMP.object_dictionary(plain), :cl2c)
    @test !haskey(JuMP.object_dictionary(plain), :clinfc)

    # The whole model still solves, so the calibrated ceilings are feasible together.
    res = optimise(MeanRisk(; r = Variance(),
                            opt = JuMPOptimiser(; slv = SLV, pe = PR60_N, l2c = crule)))
    @test isa(res.w, AbstractVector)
    @test sum(res.w) ≈ 1
    @test norm(res.w, 2) <= 2^(-1 / 2) + sqrt(eps())
end

#=
`l2c` and `linfc` are bounded `TD_Option{<:Num_NormCeilCal}`, so one field now carries two
deferral channels. ADR 0030 never considered the pair, and the amendment issue #617 added
to ADR 0095 settles it: a schedule reaches the HOST that holds the slot and no further,
because a rule is never standalone and the host already carries the channel.

The two verbs run at different points and neither knows about the other: the period
selection runs in `update_time_dependent_fields`, before any prior is fitted, and the
calibration resolution runs at assembly, against the prior of the period that was selected.
=#
@testset "Norm ceiling calibration: the schedule selects, then the rule runs" begin
    crule = EffectiveAssetFloor(; fraction = 0.5)
    td = TimeDependent([crule, 0.6]; default = 0.6)
    @test isa(td, PO.TD_Option{<:PO.Num_NormCeilCal})

    opt = JuMPOptimiser(; slv = SLV, l2c = td)
    @test PO.time_dependent_fields(opt) == (:l2c,)

    # The selection carries the schedule's occupant out unchanged. It does not resolve it,
    # and it has no prior result with which it could.
    ctx1 = TimeDependentContext(; i = 1, n = 2, rd = RD60, train_idx = [1:20, 1:40],
                                test_idx = [21:40, 41:60])
    ctx2 = TimeDependentContext(; i = 2, n = 2, rd = RD60, train_idx = [1:20, 1:40],
                                test_idx = [21:40, 41:60])
    @test PO.time_dependent_value(td, ctx1) === crule
    @test PO.time_dependent_value(td, ctx2) == 0.6

    # The selected rule then resolves against whichever prior the fold produced, so a
    # schedule and a rule compose rather than fight.
    sel = PO.time_dependent_value(td, ctx1)
    @test PO.resolve_calibration_slot(sel, :l2c, PR60_N, nothing, nothing,
                                      CalibrationContext(; p = 2)) ≈ 2^(-1 / 2)

    # A schedule is not a rule, so the constructor's range check skips it and the field
    # falls back to its own default outside every fold loop.
    @test PO.reset_time_dependent_fields(opt).l2c == 0.6
end

#=
Issue #618 sweeps the three units that #616 added to this file: the two rule guards and
`norm_ceiling_factory`. The ticket asks four questions, and each is answered below with a
run rather than a read.

ADR 0095 rules that a slot bound pairs `Number` with ONE rule family, and that the bound
is the whole of the family validation. `LpRegularisation.val` is the one slot that breaks
the rule, because one type serves two readings. The first section is the ratchet that keeps
the exception on that one slot: a second slot of the same shape would be a second exception,
and ADR 0095 grants none.
=#
@testset "Sweep #618: the two role guards on LpRegularisation" begin
    crule = EffectiveAssetFloor(; fraction = 0.5)
    rrule = RateRadius(; c = 0.2)
    root = pkgdir(PortfolioOptimisers)
    srcfiles = String[]
    for (d, _, fs) in walkdir(joinpath(root, "src")), f in fs
        endswith(f, ".jl") && push!(srcfiles, joinpath(d, f))
    end

    # -- The one exception. Each alias that pairs `Number` with a calibration rule family,
    # against the number of families it admits. An alias that admits the whole root names
    # no family, so it bounds no slot and is skipped.
    families = filter(T -> parentmodule(T) === PO,
                      InteractiveUtils.subtypes(PO.AbstractCalibrationAlgorithm))
    # Seven families: the five of the risk measures and the optimiser, and the Spanned Shrinkage
    # and the Orthogonal Forecast Scale of the Cross-Sectional Factor Prior.
    @test length(families) == 7
    counted = Dict{Symbol, Int}()
    for n in names(PO; all = true)
        # `names(; all = true)` also answers the compiler's own bindings, and a gensym
        # such as `#6593#val` is bound to the very alias its keyword annotates. It is not
        # a slot bound and counting it would count each alias twice, which is what Julia
        # 1.13 started doing here.
        Base.isgensym(n) && continue
        isdefined(PO, n) || continue
        v = getfield(PO, n)
        isa(v, Type) || continue
        isa(Base.unwrap_unionall(v), Union) || continue
        Number <: v || continue
        PO.AbstractCalibrationAlgorithm <: v && continue
        c = count(R -> R <: v, families)
        if c >= 1
            counted[n] = c
        end
    end
    @test length(counted) == 8
    @test sort([k for (k, c) in counted if c > 1]) == [:Num_AmbRadNormCeilCal]
    @test counted[:Num_AmbRadNormCeilCal] == 2
    @test all(==(1), [c for (k, c) in counted if k !== :Num_AmbRadNormCeilCal])

    # The dual-use alias bounds one slot, and that slot is `LpRegularisation.val`. The
    # three sites are the struct's field list and its two constructors.
    bound = [relpath(f, root) for f in srcfiles
             for l in eachline(f) if occursin("::Num_AmbRadNormCeilCal", l)]
    @test length(bound) == 3
    @test unique(bound) ==
          [joinpath("src", "17_Optimisation", "05_JuMP", "02_JuMPConstraints",
                    "12_RegularisationConstraints.jl")]

    # -- Both entry points, and no third path. Each builder is called from two sites, the
    # optimiser's and the programme Allocation Set's (ADR 0159), and each site wraps the
    # term in its own route's verb, so no term reaches a builder unchecked. The optimiser
    # wraps at the call. The set wraps in `resolve_allocation_set_rows`, which binds the
    # wrapped term to the name its call passes; its `::Nothing` arm runs when no slot reads
    # the head's rows, so the term it passes is already a value. A definition line and a
    # docstring signature both carry `::`, so the filter keeps the calls alone.
    calls(tok) = [strip(l) for f in srcfiles
                  for l in eachline(f) if occursin(tok, l) && !occursin("::", l)]
    lp_calls = calls("set_lp_regularisation!(model")
    lpc_calls = calls("set_weight_norm_p_constraints!(model")
    @test length(lp_calls) == 2
    @test length(lpc_calls) == 2
    @test count(l -> occursin("factory(opt.lp,", l), lp_calls) == 1
    @test count(l -> occursin("norm_ceiling_factory(opt.lpc,", l), lpc_calls) == 1
    @test count(==("set_lp_regularisation!(model, lp)"), lp_calls) == 1
    @test count(==("set_weight_norm_p_constraints!(model, lpc)"), lpc_calls) == 1
    @test length(calls("lp = factory(set.lp,")) == 1
    @test length(calls("lpc = norm_ceiling_factory(set.lpc,")) == 1

    # -- The `TimeDependent` case. The constructor calls the two guards with no wrapper
    # check, because a schedule is not a term and meets the permissive fallback. It is the
    # test-substitution that carries each entry back through the same constructor, so a
    # wrong rule inside a schedule is refused where a wrong rule in a bare field is.
    @test isnothing(PO.assert_penalty_coefficient_role(TimeDependent([LpRegularisation(;
                                                                                       val = crule)];
                                                                     default = LpRegularisation(;
                                                                                                val = rrule))))
    @test_throws ArgumentError JuMPOptimiser(; slv = SLV,
                                             lp = TimeDependent([LpRegularisation(;
                                                                                  val = crule),
                                                                 LpRegularisation(;
                                                                                  val = rrule)];
                                                                default = LpRegularisation(;
                                                                                           val = rrule)))
    @test_throws ArgumentError JuMPOptimiser(; slv = SLV,
                                             lp = TimeDependent([LpRegularisation(;
                                                                                  val = rrule)];
                                                                default = LpRegularisation(;
                                                                                           val = crule)))
    @test_throws ArgumentError JuMPOptimiser(; slv = SLV,
                                             lpc = TimeDependent([LpRegularisation(;
                                                                                   val = rrule),
                                                                  LpRegularisation(;
                                                                                   val = crule)];
                                                                 default = LpRegularisation(;
                                                                                            val = crule)))
    @test_throws ArgumentError JuMPOptimiser(; slv = SLV,
                                             lpc = TimeDependent([LpRegularisation(;
                                                                                   val = crule)];
                                                                 default = LpRegularisation(;
                                                                                            val = rrule)))

    # A schedule whose entry is a VECTOR of terms is substituted the same way, so the guard
    # reaches the term through the vector method rather than through a wrapper of its own.
    @test_throws ArgumentError JuMPOptimiser(; slv = SLV,
                                             lp = TimeDependent([[LpRegularisation(;
                                                                                   val = crule)]];
                                                                default = [LpRegularisation(;
                                                                                            val = rrule)]))

    # The correct pairing crosses, so the guard refuses the rule and not the schedule.
    opt = JuMPOptimiser(; slv = SLV,
                        lp = TimeDependent([LpRegularisation(; val = rrule),
                                            LpRegularisation(; val = 1e-3)];
                                           default = LpRegularisation(; val = rrule)),
                        lpc = TimeDependent([LpRegularisation(; val = crule),
                                             LpRegularisation(; val = 0.6)];
                                            default = LpRegularisation(; val = crule)))
    @test isa(opt.lp, TimeDependent)
    @test isa(opt.lpc, TimeDependent)
    @test PO.reset_time_dependent_fields(opt).lp.val == rrule

    # -- The two verbs differ in the guard and in the key, and in nothing else. Both bind
    # the term's own order, so a rule that reads the order gives a different number at a
    # different `p`. `DualNormRadius` refuses an unbound order, so a penalty route that
    # bound nothing would throw here instead of returning a number.
    dnr = DualNormRadius()
    @test_throws ArgumentError PO.resolve_calibration_slot(dnr, :lpreg_val, PR60_N,
                                                           PR60_N.w, SLV)
    e = sqrt.(diag(PR60_N.sigma)) ./ sqrt(60)
    p3 = PO.factory(LpRegularisation(; p = 3, val = dnr), PR60_N, SLV)
    p5 = PO.factory(LpRegularisation(; p = 5, val = dnr), PR60_N, SLV)
    @test p3.val != p5.val
    @test p3.val / p5.val ≈ norm(e, 3 / 2) / norm(e, 5 / 4)

    # A term whose `val` is already a number is returned by identity on this route too,
    # so the penalty verb and the ceiling verb agree on the case that resolves nothing.
    stated = LpRegularisation(; p = 3, val = 1e-3)
    @test PO.factory(stated, PR60_N, SLV) === stated

    # The keys are two readings rather than two names for one. A radius rule given an
    # order still has no reading under the ceiling key, which is why a `factory` call on
    # `lpc` would be wrong even after the rule guard.
    @test_throws ArgumentError PO.resolve_calibration_slot(dnr, :lpc, PR60_N, PR60_N.w, SLV,
                                                           CalibrationContext(; p = 3))
end

#=
Issue #626. `JuMPOptimiser` refused an empty `lp` and accepted an empty `l2` and an empty
`lpc`. The three fields hold the same shape of value, a term or a vector of terms, and each
builder iterates the terms, so an empty vector builds the model that `nothing` builds. The
three now agree: a vector must carry at least one term.
=#
@testset "Issue #626: the three regularisation fields refuse an empty vector" begin
    # -- The static case. One message per field, and the field names itself.
    @test_throws PortfolioOptimisers.IsEmptyError JuMPOptimiser(; slv = SLV,
                                                                l2 = L2Regularisation[])
    @test_throws PortfolioOptimisers.IsEmptyError JuMPOptimiser(; slv = SLV,
                                                                lp = LpRegularisation[])
    @test_throws PortfolioOptimisers.IsEmptyError JuMPOptimiser(; slv = SLV,
                                                                lpc = LpRegularisation[])

    # A vector that carries one term is admitted, so the guard refuses emptiness alone.
    opt = JuMPOptimiser(; slv = SLV, l2 = [L2Regularisation(; val = 5e-6)],
                        lp = [LpRegularisation(; val = 1e-3)],
                        lpc = [LpRegularisation(; val = 0.6)])
    @test length(opt.l2) == 1
    @test length(opt.lp) == 1
    @test length(opt.lpc) == 1

    # `nothing` is the way to build no term, and it stays admitted. That is the state the
    # empty vector reached by accident, and it now has one spelling rather than two.
    plain = JuMPOptimiser(; slv = SLV)
    @test isnothing(plain.l2)
    @test isnothing(plain.lp)
    @test isnothing(plain.lpc)

    # -- The scheduled case. A `TimeDependent` is not a vector of terms, so it meets no
    # guard itself. Its entries are test-substituted through this same constructor, so an
    # empty entry is refused there.
    @test_throws PortfolioOptimisers.IsEmptyError JuMPOptimiser(; slv = SLV,
                                                                l2 = TimeDependent([L2Regularisation[]];
                                                                                   default = [L2Regularisation(;
                                                                                                               val = 5e-6)]))
    @test_throws PortfolioOptimisers.IsEmptyError JuMPOptimiser(; slv = SLV,
                                                                lpc = TimeDependent([LpRegularisation[]];
                                                                                    default = [LpRegularisation(;
                                                                                                                val = 0.6)]))
end

#=
A rule states no number at construction, so the constructor guard on `l2c` and `linfc` reads
a rule, selects the permissive fallback of the validator, and checks nothing. Neither field
holds a term, so no rebuild states the range downstream either. The check therefore belongs
to `resolve_calibration_slot`, which is the one place the rule runs, and the rule's own
method there refuses the values a stated ceiling is refused. It is paid on the calibrated
path alone: a stated ceiling takes the fallback method and meets no check.
=#
@testset "Norm ceiling calibration: a calibrated ceiling meets the stated ceiling's check" begin
    function assemble_with(opt)
        mr = MeanRisk(; r = Variance(), opt = opt)
        attrs = PO.processed_jump_optimiser_attributes(mr.opt, RD60)
        model = JuMP.Model()
        PO.set_model_scales!(model, mr.opt.sc, mr.opt.so)
        PO.set_model_observations!(model, size(attrs.pr.X, 1))
        PO.set_maximum_ratio_factor_variables!(model, mr.obj)
        PO.set_w!(model, attrs.pr.X, mr.wi)
        PO.set_weight_constraints!(model, attrs.wb, mr.opt)
        PO.assemble_jump_model!(model, mr, mr.opt, attrs, RD60, mr.r, mr.obj)
        return model
    end

    # A stated ceiling of each shape is refused where the caller wrote it.
    @test_throws DomainError JuMPOptimiser(; slv = SLV, pe = PR60_N, l2c = -1.0)
    @test_throws DomainError JuMPOptimiser(; slv = SLV, pe = PR60_N, l2c = 0.0)
    @test_throws DomainError JuMPOptimiser(; slv = SLV, pe = PR60_N, linfc = NaN)
    @test_throws DomainError JuMPOptimiser(; slv = SLV, pe = PR60_N, linfc = Inf)

    # The same four numbers, computed by a rule of the ceiling family, are refused by the
    # resolver, under the key of the slot the rule stands in.
    for bad in (-1.0, 0.0, NaN, Inf)
        crule = ProbeCeilingValue(bad)
        for key in (:l2c, :linfc, :lpc)
            @test_throws DomainError PO.resolve_calibration_slot(crule, key, PR60_N,
                                                                 PR60_N.w, SLV,
                                                                 CalibrationContext(;
                                                                                    p = 2))
        end
        err = try
            PO.resolve_calibration_slot(crule, :l2c, PR60_N, PR60_N.w, SLV,
                                        CalibrationContext(; p = 2))
        catch e
            e
        end
        @test occursin("l2c", sprint(showerror, err))

        # So the whole route refuses it too, at assembly and at the term's own factory.
        @test_throws DomainError assemble_with(JuMPOptimiser(; slv = SLV, pe = PR60_N,
                                                             l2c = crule))
        @test_throws DomainError assemble_with(JuMPOptimiser(; slv = SLV, pe = PR60_N,
                                                             linfc = crule))
        @test_throws DomainError PO.norm_ceiling_factory(LpRegularisation(; p = 3,
                                                                          val = crule),
                                                         PR60_N)

        # A rule written as a plain function names no family, so no method of the resolver
        # can read the quantity it computes and the resolver returns the number unchecked.
        # `assemble_jump_model!` is what states the range of the four slots that reach the
        # model raw, so the route refuses it there instead.
        fn = (key, pr, w, slv, ctx) -> bad
        @test PO.resolve_calibration_slot(fn, :l2c, PR60_N, PR60_N.w, SLV,
                                          CalibrationContext(; p = 2)) === bad
        @test_throws DomainError assemble_with(JuMPOptimiser(; slv = SLV, pe = PR60_N,
                                                             l2c = fn))
        @test_throws DomainError assemble_with(JuMPOptimiser(; slv = SLV, pe = PR60_N,
                                                             linfc = fn))
    end

    # A stated ceiling reaches the fallback method, which checks nothing: the cost is paid
    # on the calibrated path alone, and a number the constructor already admitted is not
    # measured twice.
    @test PO.resolve_calibration_slot(1 / 2, :l2c, PR60_N, PR60_N.w, SLV,
                                      CalibrationContext(; p = 2)) == 1 / 2

    # A rule that answers a legal ceiling still reaches the model, so the check refuses
    # nothing a caller could have stated by hand.
    good = (key, pr, w, slv, ctx) -> 1 / 2
    model = assemble_with(JuMPOptimiser(; slv = SLV, pe = PR60_N, l2c = good, linfc = good))
    @test JuMP.normalized_rhs(model[:cl2c]) ≈ 1 / 2
    @test JuMP.normalized_rhs(model[:clinfc]) ≈ 1 / 2
end

#=
A **significance level** is the tail probability a risk measure prices, and a **deformation
parameter** is the shape the relativistic family reads that tail under. Forty slots across
twenty-seven types hold one or the other, and each of them takes a **Calibration Rule** in
place of the number.

`test_09f_calibration_slot.jl` covers the mechanism, `test_09f_calibration_slot.jl` covers
the three rules, and `test_09f_calibration_slot.jl` covers the ambiguity families. This
file covers the forty slots of issue #583 and the resolution written beside each of them.

Three of the forty-three surveyed slots do NOT widen. `alpha_i` on `OrderedWeightsArrayTailGini`
and on its Range twin, and `beta_i` on the Range twin, are the starting points of the inner
tail-Gini integration. They are not quantities to estimate, so they keep `::Number`.

The joint `0 < alpha_i < alpha < 1` bound is the whole of the ordering validation. Only
`alpha` moves, so a rule that returns a value at or below the stated `alpha_i` is refused
when the rebuild runs, at fold time, and no new guard exists.
=#
using Clarabel, JuMP, FLoops
const PO = PortfolioOptimisers

const RNG_S = StableRNG(135792468)
const X60_S = randn(RNG_S, 60, 4)
const PR60_S = prior(EmpiricalPrior(), X60_S)
const X120_S = randn(RNG_S, 120, 4)
const PR120_S = prior(EmpiricalPrior(), X120_S)
const W4 = fill(0.25, 4)

# A significance rule with no type at all. A closure over a caller's own number is the case
# that cannot be given one, and it is why the `alg` bound admits a bare `Function`. It
# reports what it was handed, so the resolver's argument order is asserted.
const SIG_SEEN = Ref{Any}(nothing)
function probe_significance(key::Symbol, pr::PO.AbstractPriorResult, w, slv,
                            ::PO.CalibrationContext)
    SIG_SEEN[] = (; key = key, weighted = !isnothing(w), solved = !isnothing(slv))
    return 4 / size(pr.X, 1)
end

# The two rules the file reuses. `n = 3` gives `alpha = 0.05` at `T = 60`, which is the
# library default, so a fold of that length is the one case where a rule and the default
# agree — every assertion below that separates them uses a second length.
#
# One rule serves both ends of its family, because the slot is what names the end. The four
# names below are therefore two rules under two spellings each, and each spelling names the
# end the reader is looking at.
sig_tail(n = 3) = ScenarioCount(; n = n)
const sig_head = sig_tail
def_tail(t = -1.2) = EntropyBudget(; target = t)
const def_head = def_tail

#=
The census. One row per type: the slots it declares, and the rule family of each. The four
rule families are disjoint, so the row also states which rule each slot refuses.
=#
const CENSUS = [(; T = ValueatRisk, tail = (:alpha,), head = (), dtail = (), dhead = ()),
                (; T = ValueatRiskRange, tail = (:alpha,), head = (:beta,), dtail = (),
                 dhead = ()),
                (; T = DrawdownatRisk, tail = (:alpha,), head = (), dtail = (), dhead = ()),
                (; T = RelativeDrawdownatRisk, tail = (:alpha,), head = (), dtail = (),
                 dhead = ()),
                (; T = ConditionalValueatRisk, tail = (:alpha,), head = (), dtail = (),
                 dhead = ()),
                (; T = DistributionallyRobustConditionalValueatRisk, tail = (:alpha,),
                 head = (), dtail = (), dhead = ()),
                (; T = ConditionalValueatRiskRange, tail = (:alpha,), head = (:beta,),
                 dtail = (), dhead = ()),
                (; T = DistributionallyRobustConditionalValueatRiskRange, tail = (:alpha,),
                 head = (:beta,), dtail = (), dhead = ()),
                (; T = ConditionalDrawdownatRisk, tail = (:alpha,), head = (), dtail = (),
                 dhead = ()),
                (; T = DistributionallyRobustConditionalDrawdownatRisk, tail = (:alpha,),
                 head = (), dtail = (), dhead = ()),
                (; T = RelativeConditionalDrawdownatRisk, tail = (:alpha,), head = (),
                 dtail = (), dhead = ()),
                (; T = EntropicValueatRisk, tail = (:alpha,), head = (), dtail = (),
                 dhead = ()),
                (; T = EntropicValueatRiskRange, tail = (:alpha,), head = (:beta,),
                 dtail = (), dhead = ()),
                (; T = EntropicDrawdownatRisk, tail = (:alpha,), head = (), dtail = (),
                 dhead = ()),
                (; T = RelativeEntropicDrawdownatRisk, tail = (:alpha,), head = (),
                 dtail = (), dhead = ()),
                (; T = RelativisticValueatRisk, tail = (:alpha,), head = (),
                 dtail = (:kappa,), dhead = ()),
                (; T = RelativisticValueatRiskRange, tail = (:alpha,), head = (:beta,),
                 dtail = (:kappa_a,), dhead = (:kappa_b,)),
                (; T = RelativisticDrawdownatRisk, tail = (:alpha,), head = (),
                 dtail = (:kappa,), dhead = ()),
                (; T = RelativeRelativisticDrawdownatRisk, tail = (:alpha,), head = (),
                 dtail = (:kappa,), dhead = ()),
                (; T = OrderedWeightsArrayConditionalValueatRisk, tail = (:alpha,),
                 head = (), dtail = (), dhead = ()),
                (; T = OrderedWeightsArrayTailGini, tail = (:alpha,), head = (), dtail = (),
                 dhead = ()),
                (; T = OrderedWeightsArrayConditionalValueatRiskRange, tail = (:alpha,),
                 head = (:beta,), dtail = (), dhead = ()),
                (; T = OrderedWeightsArrayTailGiniRange, tail = (:alpha,), head = (:beta,),
                 dtail = (), dhead = ()),
                (; T = PowerNormValueatRisk, tail = (:alpha,), head = (), dtail = (),
                 dhead = ()),
                (; T = PowerNormValueatRiskRange, tail = (:alpha,), head = (:beta,),
                 dtail = (), dhead = ()),
                (; T = PowerNormDrawdownatRisk, tail = (:alpha,), head = (), dtail = (),
                 dhead = ()),
                (; T = RelativePowerNormDrawdownatRisk, tail = (:alpha,), head = (),
                 dtail = (), dhead = ())]

# The nine rows of the census that write a resolution of their own. Two readings put a type
# here and nothing else does. Seven carry an order between their slots, which a `bind_` verb
# states: a deformation rule reads the significance level of a sibling, and a tail weight
# reads the probability of its own end. Two carry a second mechanism beside the calibration,
# a formulation child that the Deferred-Quantity recursion resolves. Every other row is
# derived from its declaration alone.
const ORDERED = [ValueatRisk, ValueatRiskRange,
                 DistributionallyRobustConditionalValueatRisk,
                 DistributionallyRobustConditionalValueatRiskRange,
                 DistributionallyRobustConditionalDrawdownatRisk, RelativisticValueatRisk,
                 RelativisticValueatRiskRange, RelativisticDrawdownatRisk,
                 RelativeRelativisticDrawdownatRisk]

kw(slot, val) = NamedTuple{(slot,)}((val,))

@testset "Significance calibration: the census is twenty-seven types and forty slots" begin
    # The counts #352 surveyed, re-derived from the rows above rather than restated.
    @test length(CENSUS) == 27
    @test sum(row -> length(row.tail), CENSUS) == 27
    @test sum(row -> length(row.head), CENSUS) == 8
    @test sum(row -> length(row.dtail), CENSUS) == 4
    @test sum(row -> length(row.dhead), CENSUS) == 1
    @test sum(row -> length(row.tail) +
                     length(row.head) +
                     length(row.dtail) +
                     length(row.dhead), CENSUS) == 40

    for row in CENSUS
        slots = (row.tail..., row.head..., row.dtail..., row.dhead...)
        # Every type declares its slots, and the declaration names exactly them. The
        # ambiguity slots of the three distributionally robust measures are declared beside
        # them, so those rows are a subset rather than an equality.
        declared = keys(PO.calibration_slots(row.T()))
        @test issubset(slots, declared)
        @test all(s -> hasproperty(row.T(), s), slots)
        # A resolution stands beside every declaration, and it takes the effective solver as
        # a third positional argument. It is the derivation for a type whose slots carry no
        # order between them, and a per-type method for one whose slots do, so the census
        # reads that a method applies rather than which of the two it is. The testset
        # "the resolution of every slot" then runs each row and reads the number back.
        ms = methods(PO.resolve_deferred_quantities, (row.T, PO.AbstractPriorResult, Any))
        @test !isempty(ms)
        # A per-type method exists where, and only where, a `bind_` verb carries a sibling's
        # value into the slot. The travelling pair is the whole of that set.
        own = Base.unwrap_unionall(first(ms).sig).parameters[2] !== Any
        @test own == (row.T in ORDERED)
    end

    # The three inner starting points do NOT widen, so no rule may stand in them.
    @test_throws TypeError OrderedWeightsArrayTailGini(; alpha_i = sig_tail())
    @test_throws TypeError OrderedWeightsArrayTailGiniRange(; alpha_i = sig_tail())
    @test_throws TypeError OrderedWeightsArrayTailGiniRange(; beta_i = sig_head())
    # They are not declared either, so nothing recurses into them.
    @test PO.calibration_slots(OrderedWeightsArrayTailGini()) == (; alpha = 0.05)
    @test keys(PO.calibration_slots(OrderedWeightsArrayTailGiniRange())) == (:alpha, :beta)
end

@testset "Significance calibration: each slot's bound refuses the other family" begin
    rules = (; tail = sig_tail(), head = sig_head(), dtail = def_tail(), dhead = def_head())
    # The slot names the end of the distribution, so one bound serves both ends of a
    # family. What a bound refuses is the OTHER family's rule.
    other_family = (; tail = :dtail, head = :dhead, dtail = :tail, dhead = :head)
    for row in CENSUS
        for (family, slots) in
            pairs((; tail = row.tail, head = row.head, dtail = row.dtail,
                   dhead = row.dhead))
            for slot in slots
                # The rule the slot admits builds.
                @test isa(getproperty(row.T(; kw(slot, rules[family])...), slot),
                          typeof(rules[family]))
                # The other family is refused at construction, by the bound and by no guard.
                @test_throws TypeError row.T(; kw(slot, rules[other_family[family]])...)
                # A plain number still builds, so nothing existing moved.
                @test getproperty(row.T(; kw(slot, 0.07)...), slot) == 0.07
            end
        end
    end
end

@testset "Significance calibration: the resolution of every slot" begin
    # `ScenarioCount(n = 4)` at `T = 120` gives `alpha = 1/30`, which is neither the library
    # default nor the value the same rule gives at `T = 60`. So a number that matches it can
    # only have come from the rule and from this sample.
    want = 4 / 120
    rules = (; tail = ScenarioCount(; n = 4), head = ScenarioCount(; n = 4),
             dtail = def_tail(-1.6), dhead = def_head(-1.6))
    for row in CENSUS
        pairs_ = ((:tail, row.tail), (:head, row.head), (:dtail, row.dtail),
                  (:dhead, row.dhead))
        kws = NamedTuple()
        for (family, slots) in pairs_, slot in slots
            kws = merge(kws, kw(slot, rules[family]))
        end
        x = row.T(; kws...)
        out = PO.resolve_deferred_quantities(x, PR120_S)
        for slot in (row.tail..., row.head...)
            @test getproperty(out, slot) ≈ want
        end
        # A deformation slot resolves to a number in `(0, 1)`, which is what its own bound
        # admits. The value is the inversion of the Kaniadakis logarithm, not `want`.
        for slot in (row.dtail..., row.dhead...)
            k = getproperty(out, slot)
            @test isa(k, Number)
            @test 0 < k < 1
        end
        # A measure whose slots all hold numbers is returned unchanged, so the common case
        # allocates nothing.
        plain = row.T()
        @test PO.resolve_deferred_quantities(plain, PR120_S) === plain
    end
end

@testset "Significance calibration: a plain function is a rule, and it sees five arguments" begin
    SIG_SEEN[] = nothing
    rule = probe_significance
    @test isa(probe_significance, PO.Num_SigCal)
    r = ConditionalValueatRisk(; alpha = rule)
    out = PO.resolve_deferred_quantities(r, PR60_S)
    @test out.alpha ≈ 4 / 60
    @test SIG_SEEN[] == (; key = :alpha, weighted = false, solved = false)

    # The measure's own observation weights reach the rule, and so does the solver of a
    # measure that carries one.
    SIG_SEEN[] = nothing
    ws = pweights(range(; start = 1, stop = 2, length = 60))
    PO.resolve_deferred_quantities(ConditionalValueatRisk(; alpha = rule, w = ws), PR60_S)
    @test SIG_SEEN[].weighted

    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = "verbose" => false)
    SIG_SEEN[] = nothing
    PO.resolve_deferred_quantities(EntropicValueatRisk(; alpha = rule, slv = slv), PR60_S)
    @test SIG_SEEN[].solved
    # On the `JuMP` route the measure states no solver and the caller threads one, and the
    # owner settles the two with `sel`.
    SIG_SEEN[] = nothing
    PO.resolve_deferred_quantities(EntropicValueatRisk(; alpha = rule), PR60_S, slv)
    @test SIG_SEEN[].solved
    # A measure that carries no solver at all gives the rule the one it was handed.
    SIG_SEEN[] = nothing
    PO.resolve_deferred_quantities(ConditionalValueatRisk(; alpha = rule), PR60_S, slv)
    @test SIG_SEEN[].solved
end

@testset "Significance calibration: the value-at-risk formulation resolves beside alpha" begin
    # `ValueatRisk` and `ValueatRiskRange` declare `alg` in `deferred_slots` and had no
    # resolver of their own. A per-type resolver is more specific than the derived
    # recursion, so it must carry the child itself or the formulation stops resolving.
    ce = PortfolioOptimisersCovariance()
    rule = ScenarioCount(; n = 4)
    v = PO.resolve_deferred_quantities(ValueatRisk(; alpha = rule,
                                                   alg = DistributionValueatRisk(;
                                                                                 sigma = ce)),
                                       PR120_S)
    @test v.alpha ≈ 4 / 120
    @test v.alg.sigma ≈ PR120_S.sigma

    vr = PO.resolve_deferred_quantities(ValueatRiskRange(; alpha = rule,
                                                         beta = ScenarioCount(; n = 6),
                                                         alg = DistributionValueatRisk(;
                                                                                       sigma = ce)),
                                        PR120_S)
    @test vr.alpha ≈ 4 / 120
    @test vr.beta ≈ 6 / 120
    @test vr.alg.sigma ≈ PR120_S.sigma

    # A stated number beside a deferred formulation still resolves the formulation alone.
    v2 = PO.resolve_deferred_quantities(ValueatRisk(;
                                                    alg = DistributionValueatRisk(;
                                                                                  sigma = ce)),
                                        PR120_S)
    @test v2.alpha == 0.05
    @test v2.alg.sigma ≈ PR120_S.sigma
end

@testset "Significance calibration: alpha and kappa are a travelling pair" begin
    # `EntropyBudget` reads the significance level of its sibling slot, so the owner
    # resolves `alpha` first and states the number in the context. Without it the rule has
    # nothing to invert and refuses by name.
    lone = EntropyBudget(; target = -1.2)
    @test !hasfield(EntropyBudget, :alpha)
    err = try
        lone(:kappa, PR60_S, nothing, nothing, CalibrationContext())
        nothing
    catch e
        e
    end
    @test isa(err, PO.IsNothingError)
    @test occursin("CalibrationContext.alpha", sprint(showerror, err))

    # Inside the measure the pair resolves together, and the κ it produces is the one the
    # rule gives when the caller states the same `alpha` themselves.
    r = RelativisticValueatRisk(; alpha = sig_tail(), kappa = def_tail(-1.2))
    out = PO.resolve_deferred_quantities(r, PR60_S)
    @test out.alpha ≈ 3 / 60
    bound = EntropyBudget(; target = -1.2)
    @test out.kappa ≈
          bound(:kappa, PR60_S, nothing, nothing, CalibrationContext(; alpha = out.alpha))
    @test 0 < out.kappa < 1

    # The Range type carries one pair per tail, and neither side reads the other's number.
    # The band the target must lie in moves with the level, so the two ends take two
    # targets: `alpha = 0.05` reaches `(-1.334, -1.099)` and `beta = 0.1` reaches
    # `(-2.917, -1.792)`. A single target that suits both ends does not exist here, which is
    # itself the reason the rule refuses a target outside the band rather than clamping.
    rg = RelativisticValueatRiskRange(; alpha = sig_tail(3), kappa_a = def_tail(-1.2),
                                      beta = sig_head(6), kappa_b = def_head(-2.0))
    og = PO.resolve_deferred_quantities(rg, PR60_S)
    @test og.alpha ≈ 3 / 60
    @test og.beta ≈ 6 / 60
    @test og.kappa_a ≈ EntropyBudget(; target = -1.2)(:kappa_a, PR60_S, nothing, nothing,
                                                      CalibrationContext(; alpha = og.alpha))
    @test og.kappa_b ≈ EntropyBudget(; target = -2.0)(:kappa_b, PR60_S, nothing, nothing,
                                                      CalibrationContext(; alpha = og.beta))
    @test og.kappa_a != og.kappa_b

    # The head end really reads its own `beta`. The head's target lies outside the band the
    # tail's level reaches, so the same target bound to `alpha` has no answer at all: the
    # pairing is not an accident of the two ends holding one number.
    @test_throws DomainError EntropyBudget(; target = -2.0)(:kappa_b, PR60_S, nothing,
                                                            nothing,
                                                            CalibrationContext(;
                                                                               alpha = og.alpha))

    # The context reaches the rule through the resolver, and a stated number ignores it.
    # No significance rule reads a sibling, so the field is dead weight for that family.
    ctx = CalibrationContext(; alpha = 0.05)
    @test PO.resolve_calibration_slot(0.3, :kappa, PR60_S, nothing, nothing, ctx) == 0.3
    st = sig_tail()
    @test PO.resolve_calibration_slot(st, :alpha, PR60_S, nothing, nothing, ctx) ==
          PO.resolve_calibration_slot(st, :alpha, PR60_S, nothing, nothing)
    # The rule is not rebuilt on the way in: the resolver calls it and hands the context
    # to it, so the occupant a per-type method compares against never moves.
    dt = def_tail(-1.2)
    @test !hasfield(typeof(dt), :alpha)
    @test PO.resolve_calibration_slot(dt, :kappa, PR60_S, nothing, nothing, ctx) ≈
          dt(:kappa, PR60_S, nothing, nothing, ctx)
    @test PO.resolve_calibration_slot(def_head(-1.2), :kappa_b, PR60_S, nothing, nothing,
                                      ctx) ≈
          EntropyBudget(; target = -1.2)(:kappa_b, PR60_S, nothing, nothing, ctx)
end

@testset "Significance calibration: the ordered-weights builders and their containers" begin
    rule = ScenarioCount(; n = 4)

    # `beta = alpha` is the default on the two Range types: a number crosses unchanged, and
    # so does a rule, because the slot is what names the end.
    cr = OrderedWeightsArrayConditionalValueatRiskRange(; alpha = rule)
    @test cr.beta === rule
    tg = OrderedWeightsArrayTailGiniRange(; alpha = rule)
    @test tg.beta === rule
    # `beta_i = alpha_i` is untouched, because neither side widens.
    @test tg.beta_i == tg.alpha_i
    # A stated number still defaults across, so no existing default moved.
    @test OrderedWeightsArrayConditionalValueatRiskRange(; alpha = 0.07).beta == 0.07
    # A caller who states `beta` gets their own occupant.
    @test OrderedWeightsArrayConditionalValueatRiskRange(; alpha = rule, beta = 0.09).beta ==
          0.09

    # A builder carries no observation weights of its own, so the rule reads `pr.w`.
    b = PO.resolve_deferred_quantities(OrderedWeightsArrayConditionalValueatRisk(;
                                                                                 alpha = rule),
                                       PR120_S)
    @test b.alpha ≈ 4 / 120

    # The container declares the builder, so the derived recursion reaches it and neither
    # the builder nor the container needs `@propagatable`.
    owa = OrderedWeightsArray(; w = OrderedWeightsArrayTailGini(; alpha = rule))
    @test PO.deferred_slots(owa) == (; w = owa.w)
    @test PO.calibration_slots(owa) == (; w = owa.w)
    o = PO.factory(owa, PR120_S)
    @test o.w.alpha ≈ 4 / 120
    @test o.w.alpha_i == 1e-4
    @test isa(PO.resolve_deferred_quantities(owa, PR120_S).w.alpha, Number)

    # A weight vector and a plain function defer nothing, so both are carried through.
    plain = OrderedWeightsArray()
    @test PO.factory(plain, PR120_S) === plain
    vec = OrderedWeightsArray(; w = owa_gmd(120))
    @test PO.factory(vec, PR120_S) === vec

    # The Range container wraps `w2` as `reverse ∘ w2`, so the builder a rule sits in is the
    # composition's inner half. Without a method for the composition the gain-side rule
    # would never resolve while the loss-side one did.
    rg = OrderedWeightsArrayRange(; w1 = OrderedWeightsArrayTailGini(; alpha = rule),
                                  w2 = OrderedWeightsArrayTailGini(; alpha = rule))
    @test isa(rg.w2, ComposedFunction)
    @test PO.calibration_slots(rg) == (; w1 = rg.w1, w2 = rg.w2)
    @test PO.calibration_slots(rg.w2) == (; inner = rg.w2.inner)
    org = PO.factory(rg, PR120_S)
    @test org.w1.alpha ≈ 4 / 120
    @test isa(org.w2, ComposedFunction)
    @test org.w2.inner.alpha ≈ 4 / 120
    # `rev` is stored as a done flag, so the positional rebuild reverses nothing twice.
    @test org.rev === true
    @test org.w2(120) ≈ reverse(org.w1(120))

    # The resolved container evaluates, and it reports the number a caller who states the
    # calibrated `alpha` themselves gets.
    stated = OrderedWeightsArray(; w = OrderedWeightsArrayTailGini(; alpha = 4 / 120))
    @test expected_risk(owa, W4, PR120_S) ≈ expected_risk(stated, W4, X120_S)
end

@testset "Significance calibration: the tail-Gini ordering is refused at fold time" begin
    # The joint bound is the whole of the ordering validation. A rule states no value at
    # construction, so the pair is checked when the rebuild runs.
    rule = sig_tail(3)
    ok = OrderedWeightsArrayTailGini(; alpha_i = 0.01, alpha = rule)
    @test ok.alpha === rule
    @test PO.resolve_deferred_quantities(ok, PR60_S).alpha ≈ 0.05

    # `alpha_i` above the number the rule returns builds, and is refused at fold time by the
    # same constructor a caller's own number meets. At `T = 60` the rule gives `0.05` and
    # the pair holds; at `T = 120` it gives `0.025` and the pair does not.
    late = OrderedWeightsArrayTailGini(; alpha_i = 0.04, alpha = rule)
    @test PO.resolve_deferred_quantities(late, PR60_S).alpha ≈ 0.05
    @test_throws DomainError PO.resolve_deferred_quantities(late, PR120_S)

    # An `alpha_i` outside the unit interval is still refused at construction, so the rule
    # branch checks the inner bound on its own rather than skipping it.
    @test_throws DomainError OrderedWeightsArrayTailGini(; alpha_i = 1.5, alpha = rule)
    @test_throws DomainError OrderedWeightsArrayTailGiniRange(; beta_i = 1.5, alpha = rule)

    # The Range twin refuses on whichever end breaks the pair.
    lateb = OrderedWeightsArrayTailGiniRange(; alpha_i = 1e-4, alpha = rule, beta_i = 0.04,
                                             beta = sig_head(3))
    @test PO.resolve_deferred_quantities(lateb, PR60_S).beta ≈ 0.05
    @test_throws DomainError PO.resolve_deferred_quantities(lateb, PR120_S)

    # A stated pair keeps the check it always had.
    @test_throws DomainError OrderedWeightsArrayTailGini(; alpha_i = 0.2, alpha = 0.1)
end

@testset "Significance calibration: the value-level entry point refuses a rule" begin
    # A bare returns matrix carries no sample size, no moments and no observation weights,
    # so a rule cannot be run against it. The refusal names the slot and the way out.
    r = ConditionalValueatRisk(; alpha = sig_tail())
    err = try
        expected_risk(r, W4, X60_S)
        nothing
    catch e
        e
    end
    @test isa(err, ArgumentError)
    @test occursin("ConditionalValueatRisk.alpha", err.msg)
    @test occursin("ScenarioCount", err.msg)
    @test occursin("factory", err.msg)

    # The refusal recurses into a container's children, so a rule inside a weight builder is
    # named by the builder rather than by the container.
    owa = OrderedWeightsArray(;
                              w = OrderedWeightsArrayConditionalValueatRisk(;
                                                                            alpha = sig_tail()))
    err2 = try
        expected_risk(owa, W4, X60_S)
        nothing
    catch e
        e
    end
    @test isa(err2, ArgumentError)
    @test occursin("OrderedWeightsArrayConditionalValueatRisk.alpha", err2.msg)

    # Given the prior result the measure is resolved first, so the same call succeeds and
    # reports the number a caller who states the calibrated level themselves gets.
    @test expected_risk(r, W4, PR120_S) ≈
          expected_risk(ConditionalValueatRisk(; alpha = 3 / 120), W4, X120_S)
    # A stated number reaches the functor untouched on both routes.
    plain = ConditionalValueatRisk()
    @test expected_risk(plain, W4, X60_S) ≈ expected_risk(plain, W4, PR60_S)
end

@testset "Significance calibration: the JuMP route prices the calibrated level" begin
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = "verbose" => false)
    rd = ReturnsResult(; nx = string.(1:4), X = X120_S)

    # `set_risk_constraints!` resolves the measure before it builds, so the model prices the
    # calibrated level. `n = 5` at `T = 120` gives `5/120`, which is not the default.
    cal = MeanRisk(; r = ConditionalValueatRisk(; alpha = sig_tail(5)),
                   opt = JuMPOptimiser(; slv = slv))
    stated = MeanRisk(; r = ConditionalValueatRisk(; alpha = 5 / 120),
                      opt = JuMPOptimiser(; slv = slv))
    dflt = MeanRisk(; r = ConditionalValueatRisk(; alpha = 0.05),
                    opt = JuMPOptimiser(; slv = slv))
    wc = optimise(cal, rd).w
    @test isapprox(wc, optimise(stated, rd).w; rtol = 1e-5)
    @test !isapprox(wc, optimise(dflt, rd).w; rtol = 1e-5)

    # A weight builder reaches the same route through its container.
    owa_cal = MeanRisk(;
                       r = OrderedWeightsArray(;
                                               w = OrderedWeightsArrayConditionalValueatRisk(;
                                                                                             alpha = sig_tail(5))),
                       opt = JuMPOptimiser(; slv = slv))
    owa_stated = MeanRisk(;
                          r = OrderedWeightsArray(;
                                                  w = OrderedWeightsArrayConditionalValueatRisk(;
                                                                                                alpha = 5 /
                                                                                                        120)),
                          opt = JuMPOptimiser(; slv = slv))
    @test isapprox(optimise(owa_cal, rd).w, optimise(owa_stated, rd).w; rtol = 1e-5)
end

@testset "Significance calibration: a cross-validation refits the level per fold" begin
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = "verbose" => false)
    rd = ReturnsResult(; nx = string.(1:4), X = X120_S)
    cv = IndexWalkForward(60, 30; expand_train = true)
    # Two folds of different training lengths, which is what makes the refit observable.
    @test split(cv, rd).train_idx == [1:60, 1:90]

    mr = MeanRisk(; r = ConditionalValueatRisk(; alpha = sig_tail(3)),
                  opt = JuMPOptimiser(; slv = slv))
    res = cross_val_predict(mr, rd, cv; ex = FLoops.SequentialEx())

    # Each fold holds a plain number, not the rule, so the resolution really ran per fold.
    alphas = [p.res.r.alpha for p in res.pred]
    Ts = [size(p.res.pa.pr.X, 1) for p in res.pred]
    @test Ts == [60, 90]
    @test all(isa.(alphas, Number))
    @test alphas ≈ [3 / 60, 3 / 90]
    @test alphas[1] != alphas[2]

    # The count the tail holds is the count the rule states, at both fold lengths. That is
    # the whole point of `ScenarioCount`: a stated probability would leave `3` scenarios in
    # the first fold and `4.5` in the second.
    @test [ceil(Int, a * T) for (a, T) in zip(alphas, Ts)] == [3, 3]
    @test ceil(Int, 0.05 * Ts[2]) != 3

    # Each fold's weights are the ones a caller gets by stating that fold's own level, so
    # the refit is not merely recorded on the struct but priced by the model.
    for (i, idx) in enumerate([1:60, 1:90])
        rdi = ReturnsResult(; nx = string.(1:4), X = X120_S[idx, :])
        mri = MeanRisk(; r = ConditionalValueatRisk(; alpha = alphas[i]),
                       opt = JuMPOptimiser(; slv = slv))
        @test isapprox(res.pred[i].res.w, optimise(mri, rdi).w; rtol = 1e-5)
    end
end

@testset "Significance calibration: the rebuild keeps every field the rule did not move" begin
    # The rebuild is derived from the type, so a field the resolution never names survives
    # it. The two Range builders carry four such fields between them, and each was retyped
    # by hand at the rebuild before the channel went through `rebuild_with_slots`.
    rule = sig_tail(3)
    tg = OrderedWeightsArrayTailGiniRange(; alpha_i = 1e-4, alpha = rule, beta_i = 2e-4,
                                          beta = sig_head(3), a_sim = 77, b_sim = 88)
    out = PO.resolve_deferred_quantities(tg, PR60_S)
    @test out.alpha ≈ 0.05
    @test out.beta ≈ 0.05
    @test out.alpha_i === tg.alpha_i
    @test out.beta_i === tg.beta_i
    @test out.a_sim === tg.a_sim
    @test out.b_sim === tg.b_sim

    # A measure whose slots all resolved to themselves is the object the caller passed in,
    # so the common case allocates nothing.
    stated = OrderedWeightsArrayTailGiniRange(; alpha = 0.05, beta = 0.05)
    @test PO.resolve_deferred_quantities(stated, PR60_S) === stated
end
