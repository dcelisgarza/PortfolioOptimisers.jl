#=
A risk-measure slot may hold a **Calibration Rule** — the rule that computes the tail
probability or the deformation parameter — instead of the number itself. The rule resolves
against the optimisation's own prior, so the struct that reaches a kernel always holds a
plain number.

The mechanism is parallel to the Deferred Quantity one, not shared with it, and
`test_09e_deferred_quantity.jl` covers that other half. A Deferred Quantity is fitted and a
quantity is read off the fit; a rule fits nothing and reads the sample size, the moments and
the effective observation weights. So the role types stay out of the `DeferredQuantity`
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
    function CalibratedProbe(alpha::PO.Num_SigTailCal, child)
        return new{typeof(alpha), typeof(child)}(alpha, child)
    end
end
function CalibratedProbe(; alpha::PO.Num_SigTailCal = 0.05, child = nothing)
    return CalibratedProbe(alpha, child)
end
function PortfolioOptimisers.calibration_slots(x::CalibratedProbe)
    return (; alpha = x.alpha, child = x.child)
end

# Stands in for a head slot. Its bound is the whole of the role validation.
struct HeadProbe{T} <: PO.AbstractAlgorithm
    beta::T
    function HeadProbe(beta::PO.Num_SigHeadCal)
        return new{typeof(beta)}(beta)
    end
end

# Stands in for a deformation slot.
struct DeformationProbe{T} <: PO.AbstractAlgorithm
    kappa::T
    function DeformationProbe(kappa::PO.Num_DefTailCal)
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
function ContextProbe(; alpha::PO.Num_SigTailCal = 0.05, w = nothing, slv = nothing)
    return ContextProbe(alpha, w, slv)
end
PortfolioOptimisers.calibration_slots(x::ContextProbe) = (; alpha = x.alpha)

# Carries neither, so the derivation hands the rule `pr.w` and the caller's own solver.
struct BareProbe{T1} <: PO.AbstractAlgorithm
    alpha::T1
end
BareProbe(; alpha::PO.Num_SigTailCal = 0.05) = BareProbe(alpha)
PortfolioOptimisers.calibration_slots(x::BareProbe) = (; alpha = x.alpha)

const RULE = ProbeScenarioCount(25)
const KRULE = ProbeEntropyBudget(0.3)
const CTX = SignificanceTailCalibration(; alg = ProbeContext())

@testset "Calibration slot: the taxonomy" begin
    # TWO roots, which is what #593 settled. A rule is an Algorithm, and the role that
    # places a rule in a slot is configuration that holds an algorithm, so it is an
    # Estimator. The two families sit under the algorithm root; the roles sit under
    # NEITHER of them, and that is the whole of the refusal below.
    @test PO.AbstractCalibrationAlgorithm <: PO.AbstractAlgorithm
    @test PO.AbstractCalibrationEstimator <: PO.AbstractEstimator
    @test !(PO.AbstractCalibrationEstimator <: PO.AbstractAlgorithm)
    @test !(PO.AbstractCalibrationAlgorithm <: PO.AbstractEstimator)
    @test PO.AbstractSignificanceCalibrationAlgorithm <: PO.AbstractCalibrationAlgorithm
    @test PO.AbstractDeformationCalibrationAlgorithm <: PO.AbstractCalibrationAlgorithm
    for T in (SignificanceTailCalibration, SignificanceHeadCalibration,
              DeformationTailCalibration, DeformationHeadCalibration)
        @test T <: PO.AbstractCalibrationEstimator
        @test !(T <: PO.AbstractCalibrationAlgorithm)
    end

    # The two families are disjoint, which is what makes a family bound a real check.
    @test !(PO.AbstractSignificanceCalibrationAlgorithm <:
            PO.AbstractDeformationCalibrationAlgorithm)
    @test !(PO.AbstractDeformationCalibrationAlgorithm <:
            PO.AbstractSignificanceCalibrationAlgorithm)

    # The role types stay OUT of the Deferred Quantity union. The two mechanisms are
    # parallel, and a rule reaching `resolve_slot` would be fitted, which it cannot be.
    @test !isa(SignificanceTailCalibration(; alg = RULE), PO.DeferredQuantity)
    @test !isa(SignificanceHeadCalibration(; alg = RULE), PO.DeferredQuantity)
    @test !isa(DeformationTailCalibration(; alg = KRULE), PO.DeferredQuantity)
    @test !isa(DeformationHeadCalibration(; alg = KRULE), PO.DeferredQuantity)

    # The abstract types stay unexported; the role types are caller-facing.
    exported = names(PortfolioOptimisers)
    @test !(:AbstractCalibrationAlgorithm in exported)
    @test !(:AbstractCalibrationEstimator in exported)
    @test !(:AbstractSignificanceCalibrationAlgorithm in exported)
    @test !(:AbstractDeformationCalibrationAlgorithm in exported)
    @test :SignificanceTailCalibration in exported
    @test :SignificanceHeadCalibration in exported
    @test :DeformationTailCalibration in exported
    @test :DeformationHeadCalibration in exported
end

@testset "Calibration slot: the rule lives in `alg`, and the family bounds it" begin
    # The rule is the whole content of the role type, and it survives the wrapping.
    @test SignificanceTailCalibration(; alg = RULE).alg === RULE
    @test SignificanceHeadCalibration(; alg = RULE).alg === RULE
    @test DeformationTailCalibration(; alg = KRULE).alg === KRULE
    @test DeformationHeadCalibration(; alg = KRULE).alg === KRULE

    # The positional inner constructor is the one a rebuild calls, so it takes the same rule.
    @test SignificanceTailCalibration(RULE).alg === RULE
    @test SignificanceHeadCalibration(RULE).alg === RULE
    @test DeformationTailCalibration(KRULE).alg === KRULE
    @test DeformationHeadCalibration(KRULE).alg === KRULE

    # A plain function is a rule too, in every role, because a rule is run by calling it.
    @test isa(probe_rate, PO.Func_SigCal) && isa(probe_rate, PO.Func_DefCal)
    @test SignificanceTailCalibration(; alg = probe_rate).alg === probe_rate
    @test SignificanceHeadCalibration(; alg = probe_rate).alg === probe_rate
    @test DeformationTailCalibration(; alg = probe_rate).alg === probe_rate
    @test DeformationHeadCalibration(; alg = probe_rate).alg === probe_rate

    # A typed rule names its family, so the wrong family is refused at construction by the
    # keyword constructor's bound. No guard method is written for it, and the raise is a
    # `TypeError` because the keyword carries the annotation.
    #
    # The POSITIONAL route is not checked here, and it is not a hole this mechanism opened:
    # `@concrete` emits an unconstrained `T(f1::__T_f1, ...) where {...}` for every struct
    # in the library, so a type-invalid positional call falls through every hand-written
    # bound. Issue #264 carries that, and it reaches every `@concrete` type equally.
    @test isa(RULE, PO.Func_SigCal) && !isa(RULE, PO.Func_DefCal)
    @test isa(KRULE, PO.Func_DefCal) && !isa(KRULE, PO.Func_SigCal)
    @test_throws TypeError SignificanceTailCalibration(; alg = KRULE)
    @test_throws TypeError SignificanceHeadCalibration(; alg = KRULE)
    @test_throws TypeError DeformationTailCalibration(; alg = RULE)
    @test_throws TypeError DeformationHeadCalibration(; alg = RULE)

    # A number is not a rule, so it never reaches an `alg` field.
    @test_throws TypeError SignificanceTailCalibration(; alg = 0.05)
    @test_throws TypeError DeformationTailCalibration(; alg = 0.05)

    # A ROLE is not a rule either, so a role inside another role's `alg` field is refused
    # at construction. Before #593 split the taxonomy a role subtyped its own rule family,
    # so `Func_SigCal` admitted it and the nesting type-checked. The refusal is the bound's,
    # and no guard method is written for it.
    tail = SignificanceTailCalibration(; alg = RULE)
    head = SignificanceHeadCalibration(; alg = RULE)
    ktail = DeformationTailCalibration(; alg = KRULE)
    @test !isa(tail, PO.Func_SigCal) && !isa(head, PO.Func_SigCal)
    @test !isa(ktail, PO.Func_DefCal)
    @test_throws TypeError SignificanceTailCalibration(; alg = head)
    @test_throws TypeError SignificanceTailCalibration(; alg = tail)
    @test_throws TypeError SignificanceHeadCalibration(; alg = tail)
    @test_throws TypeError DeformationTailCalibration(; alg = ktail)
    @test_throws TypeError DeformationHeadCalibration(; alg = ktail)
end

@testset "Calibration slot: the four field bounds" begin
    tail = SignificanceTailCalibration(; alg = RULE)
    head = SignificanceHeadCalibration(; alg = RULE)
    ktail = DeformationTailCalibration(; alg = KRULE)
    khead = DeformationHeadCalibration(; alg = KRULE)

    # Every bound admits the number, which is what keeps a stated value legal everywhere.
    @test isa(0.05, PO.Num_SigTailCal) && isa(0.05, PO.Num_SigHeadCal)
    @test isa(0.05, PO.Num_DefTailCal) && isa(0.05, PO.Num_DefHeadCal)

    # Each bound admits its own role and no other, so a role in the wrong slot is refused.
    @test isa(tail, PO.Num_SigTailCal)
    @test !isa(head, PO.Num_SigTailCal) && !isa(ktail, PO.Num_SigTailCal)
    @test isa(head, PO.Num_SigHeadCal)
    @test !isa(tail, PO.Num_SigHeadCal) && !isa(khead, PO.Num_SigTailCal)
    @test isa(ktail, PO.Num_DefTailCal)
    @test !isa(khead, PO.Num_DefTailCal) && !isa(tail, PO.Num_DefTailCal)
    @test isa(khead, PO.Num_DefHeadCal)
    @test !isa(ktail, PO.Num_DefHeadCal) && !isa(head, PO.Num_DefHeadCal)

    # A slot takes a role, never a bare rule and never a bare function. The role is what
    # names the end of the distribution the slot addresses.
    @test !isa(RULE, PO.Num_SigTailCal) && !isa(probe_rate, PO.Num_SigTailCal)

    # The refusal lands at construction, in the constructor's signature, and not at fold
    # time. This is the claim that lets the mechanism carry no role guard at all. The three
    # probes are plain structs, so the positional route is bounded here too.
    @test HeadProbe(head).beta === head
    @test HeadProbe(0.05).beta == 0.05
    @test_throws MethodError HeadProbe(tail)
    @test DeformationProbe(ktail).kappa === ktail
    @test_throws MethodError DeformationProbe(khead)
    @test_throws TypeError CalibratedProbe(; alpha = head)
    @test_throws MethodError CalibratedProbe(head, nothing)
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

    # A role type is unwrapped and its rule is CALLED. The rule sees the key, the prior and
    # the weights, in that order, and it never sees the role it was placed in.
    res = PO.resolve_calibration_slot(SignificanceTailCalibration(; alg = RULE), :alpha, pr,
                                      nothing)
    @test res.key === :alpha
    @test res.alpha == 25 / 60
    @test !res.weighted

    # The head role carries the same rule, so it resolves to the same number.
    res = PO.resolve_calibration_slot(SignificanceHeadCalibration(; alg = RULE), :beta, pr,
                                      pweights(fill(1 / 60, 60)))
    @test res.key === :beta
    @test res.alpha == 25 / 60
    @test res.weighted

    # The deformation family resolves through the same verb.
    @test PO.resolve_calibration_slot(DeformationTailCalibration(; alg = KRULE), :kappa, pr,
                                      nothing) == 0.3
    @test PO.resolve_calibration_slot(DeformationHeadCalibration(; alg = KRULE), :kappa_b,
                                      pr, nothing) == 0.3

    # A plain function in `alg` resolves by the same call, with no adapter in between.
    @test PO.resolve_calibration_slot(SignificanceTailCalibration(; alg = probe_rate),
                                      :alpha, pr, nothing) == inv(sqrt(60))
    @test PO.resolve_calibration_slot(DeformationHeadCalibration(; alg = probe_rate),
                                      :kappa_b, pr, nothing) == inv(sqrt(60))

    # The solver reaches the rule, so a rule may call `ERM` or `RRM`. It is the fifth
    # argument and it defaults to `nothing`, so a caller that carries none states nothing.
    slv = Solver(; name = :probe, solver = nothing)
    res = PO.resolve_calibration_slot(SignificanceTailCalibration(; alg = RULE), :alpha, pr,
                                      nothing, slv)
    @test res.solved
    @test !PO.resolve_calibration_slot(SignificanceTailCalibration(; alg = RULE), :alpha,
                                       pr, nothing).solved
    @test PO.resolve_calibration_slot(0.05, :alpha, pr, nothing, slv) == 0.05

    # The rule refits when the sample moves, which is the entire reason for a rule over a
    # number: the same role gives a different `alpha` on a shorter fold.
    fold = prior(EmpiricalPrior(), X[1:30, :])
    @test PO.resolve_calibration_slot(SignificanceTailCalibration(; alg = RULE), :alpha,
                                      fold, nothing).alpha == 25 / 30
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
    probe = CalibratedProbe(; alpha = SignificanceTailCalibration(; alg = RULE))
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
    # message names the slot, the role standing in it and the way out.
    probe = CalibratedProbe(; alpha = SignificanceTailCalibration(; alg = RULE))
    @test_throws ArgumentError PO.assert_calibrated_slots(probe)
    msg = try
        PO.assert_calibrated_slots(probe)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("CalibratedProbe.alpha", msg)
    @test occursin("SignificanceTailCalibration", msg)
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
    # an empty tuple and the role reached the kernel, where `alpha * length(x)` raised a
    # `MethodError` naming neither the slot nor the role. The declaration below is the fix.
    rng = StableRNG(987654321)
    X = randn(rng, 200, 4) / 100
    w = fill(0.25, 4)
    role = SignificanceTailCalibration(; alg = RULE)
    cvar = ConditionalValueatRisk(; alpha = role)

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

    # The message the caller reads names the child's slot, the role standing in it and the
    # way out, on the same terms as a leaf measure's.
    function refusal(r)
        return try
            expected_risk(r, w, X)
            ""
        catch e
            sprint(showerror, e)
        end
    end
    owacvar = OrderedWeightsArrayConditionalValueatRisk(; alpha = role)
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
        @test occursin("Calibration Role", msg)
        @test occursin(slot, msg)
        @test occursin("factory(r, pr)", msg)
    end
end

@testset "Calibration slot: `mirror_role` carries the tail across to the head" begin
    # A number crosses unchanged, which is what keeps `beta = alpha` alive with no widening.
    @test PO.mirror_role(0.05) === 0.05
    @test PO.mirror_role(1) === 1

    # A tail role crosses as the head role of the SAME family, holding the SAME rule.
    mirrored = PO.mirror_role(SignificanceTailCalibration(; alg = RULE))
    @test isa(mirrored, SignificanceHeadCalibration)
    @test mirrored.alg === RULE

    kmirrored = PO.mirror_role(DeformationTailCalibration(; alg = KRULE))
    @test isa(kmirrored, DeformationHeadCalibration)
    @test kmirrored.alg === KRULE

    # A function-valued rule crosses on the same terms.
    @test PO.mirror_role(SignificanceTailCalibration(; alg = probe_rate)).alg === probe_rate

    # The mirrored value is admitted by the head bound, which is the point of the carry.
    @test isa(mirrored, PO.Num_SigHeadCal)
    @test isa(kmirrored, PO.Num_DefHeadCal)
end

@testset "Calibration slot: every Range head slot defaults through `mirror_role`" begin
    # One rule for a head slot, and every Range type states it the same way: the head slot
    # defaults to `mirror_role` of its tail counterpart. A stated number crosses unchanged,
    # so the default pair of each type is the pair it carried before the rule became one.
    sig = SignificanceTailCalibration(; alg = RULE)
    def = DeformationTailCalibration(; alg = KRULE)
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

    # A stated tail ROLE crosses as the head role of the same family, in all eight.
    for r in (ValueatRiskRange(; alpha = sig), ConditionalValueatRiskRange(; alpha = sig),
              DistributionallyRobustConditionalValueatRiskRange(; alpha = sig),
              EntropicValueatRiskRange(; alpha = sig),
              RelativisticValueatRiskRange(; alpha = sig),
              PowerNormValueatRiskRange(; alpha = sig),
              OrderedWeightsArrayConditionalValueatRiskRange(; alpha = sig),
              OrderedWeightsArrayTailGiniRange(; alpha = sig))
        @test isa(r.beta, SignificanceHeadCalibration)
        @test r.beta.alg === RULE
    end

    # The deformation mirror has a caller in `src/`, and this is it. The gain-side pair of
    # the relativistic Range measure defaults to the loss-side pair, both halves of it.
    rl = RelativisticValueatRiskRange(; alpha = sig, kappa_a = def)
    @test isa(rl.kappa_b, DeformationHeadCalibration)
    @test rl.kappa_b.alg === KRULE
    @test RelativisticValueatRiskRange(; kappa_a = 0.5).kappa_b === 0.5

    # The two norm orders take the same rule. They stay numbers, so they mirror by plain
    # assignment, the way `beta_i` mirrors `alpha_i`.
    @test PowerNormValueatRiskRange(; pa = 3.0).pb === 3.0
    @test OrderedWeightsArrayTailGiniRange(; alpha_i = 1e-3).beta_i === 1e-3

    # An ambiguity role names no end of the distribution, so `mirror_role` has no method for
    # one and the two ambiguity slots of the range keep the numbers they declare.
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
    role = SignificanceTailCalibration(; alg = RULE)
    @test PO.sel(role, 0.05) === role
    @test PO.sel(DeformationHeadCalibration(; alg = KRULE), 0.3) isa
          DeformationHeadCalibration

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
    role = SignificanceTailCalibration(; alg = probe_solver_rule)

    # The caller states no solver on the measure, which is the common case: a measure's
    # solver comes from the optimiser.
    r = SolverProbe(; alpha = role)
    @test isnothing(r.slv)

    opt = JuMPOptimiser(; slv = slv, pe = pr)
    mr = MeanRisk(; r = r, opt = opt)
    attrs = PO.processed_jump_optimiser_attributes(mr.opt, rd)
    model = JuMP.Model()
    PO.set_model_scales!(model, mr.opt.sc, mr.opt.so)
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
    mrv = MeanRisk(; r = [SolverProbe(; alpha = role)], opt = opt)
    attrsv = PO.processed_jump_optimiser_attributes(mrv.opt, rd)
    modelv = JuMP.Model()
    PO.set_model_scales!(modelv, mrv.opt.sc, mrv.opt.so)
    PO.set_maximum_ratio_factor_variables!(modelv, mrv.obj)
    PO.set_w!(modelv, attrsv.pr.X, mrv.wi)
    PO.set_weight_constraints!(modelv, attrsv.wb, mrv.opt)
    SLV_SEEN[] = nothing
    PO.assemble_jump_model!(modelv, mrv, mrv.opt, attrsv, rd, mrv.r, mrv.obj)
    @test SLV_SEEN[].slv === slv

    # A measure that states its own solver keeps it, which is what `sel` has always done for
    # the weights beside it.
    own = Solver(; name = :own, solver = Clarabel.Optimizer)
    mro = MeanRisk(; r = SolverProbe(; slv = own, alpha = role), opt = opt)
    attrso = PO.processed_jump_optimiser_attributes(mro.opt, rd)
    modelo = JuMP.Model()
    PO.set_model_scales!(modelo, mro.opt.sc, mro.opt.so)
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
    role = SignificanceTailCalibration(; alg = probe_solver_rule)
    r = SolverProbe(; alpha = role)

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
    role = SignificanceTailCalibration(; alg = probe_solver_rule)

    # `_optimise` builds the model from `mr.r` and records `factory(mr.r, pr, mr.opt.slv)`.
    # The two calls are two routes over one measure, so the number the result carries must
    # be the number the constraint was built from.
    BUILT_ALPHA[] = nothing
    mr = MeanRisk(; r = SolverProbe(; alpha = role),
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
wrote no way to resolve it. The calibration channel had no counterpart, so a role reached
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

    # A role that survived the resolution names a declaration with no resolver.
    probe = CalibratedProbe(; alpha = SignificanceTailCalibration(; alg = RULE))
    @test_throws ArgumentError PO.assert_declared_calibration_resolver(probe)
    msg = try
        PO.assert_declared_calibration_resolver(probe)
        ""
    catch e
        sprint(showerror, e)
    end

    # The message names the type, the slot, the role and the resolution to write beside the
    # declaration.
    @test occursin("CalibratedProbe.alpha", msg)
    @test occursin("SignificanceTailCalibration", msg)
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
    role = SignificanceTailCalibration(; alg = probe_rate)

    # A `JuMP` builder reads the slot raw, so before the pairing this role reached the
    # builder and the failure landed several frames down, inside a kernel that expected a
    # number. It is refused at the resolution instead, and the message names the type and
    # the slot.
    mr = MeanRisk(; r = UnpairedProbe(; alpha = role),
                  opt = JuMPOptimiser(; slv = slv, pe = pr))
    @test_throws ArgumentError optimise(mr, rd)
    msg = try
        optimise(mr, rd)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("UnpairedProbe.alpha", msg)
    @test occursin("SignificanceTailCalibration", msg)

    # A vector of measures is walked the same way, so a lone measure and an aggregate are
    # refused alike.
    mrs = MeanRisk(; r = [Variance(), UnpairedProbe(; alpha = role)],
                   opt = JuMPOptimiser(; slv = slv, pe = pr))
    @test_throws ArgumentError optimise(mrs, rd)

    # A stated number needs no resolution, so the same measure optimises when its slot
    # holds one. The refusal is about the role and not about the declaration.
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
    rrole = AmbiguityRadiusCalibration(; alg = RateRadius(; c = 0.2))
    crole = NormCeilingCalibration(; alg = EffectiveAssetFloor(; fraction = 0.5))

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
