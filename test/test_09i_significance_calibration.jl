#=
A **significance level** is the tail probability a risk measure prices, and a **deformation
parameter** is the shape the relativistic family reads that tail under. Forty slots across
twenty-seven types hold one or the other, and each of them takes a **Calibration Rule** in
place of the number.

`test_09f_calibration_slot.jl` covers the mechanism, `test_09g_calibration_rules.jl` covers
the three rules, and `test_09h_ambiguity_calibration.jl` covers the ambiguity families. This
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

const RNG = StableRNG(135792468)
const X60 = randn(RNG, 60, 4)
const PR60 = prior(EmpiricalPrior(), X60)
const X120 = randn(RNG, 120, 4)
const PR120 = prior(EmpiricalPrior(), X120)
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
        out = PO.resolve_deferred_quantities(x, PR120)
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
        @test PO.resolve_deferred_quantities(plain, PR120) === plain
    end
end

@testset "Significance calibration: a plain function is a rule, and it sees five arguments" begin
    SIG_SEEN[] = nothing
    rule = probe_significance
    @test isa(probe_significance, PO.Num_SigCal)
    r = ConditionalValueatRisk(; alpha = rule)
    out = PO.resolve_deferred_quantities(r, PR60)
    @test out.alpha ≈ 4 / 60
    @test SIG_SEEN[] == (; key = :alpha, weighted = false, solved = false)

    # The measure's own observation weights reach the rule, and so does the solver of a
    # measure that carries one.
    SIG_SEEN[] = nothing
    ws = pweights(range(; start = 1, stop = 2, length = 60))
    PO.resolve_deferred_quantities(ConditionalValueatRisk(; alpha = rule, w = ws), PR60)
    @test SIG_SEEN[].weighted

    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = "verbose" => false)
    SIG_SEEN[] = nothing
    PO.resolve_deferred_quantities(EntropicValueatRisk(; alpha = rule, slv = slv), PR60)
    @test SIG_SEEN[].solved
    # On the `JuMP` route the measure states no solver and the caller threads one, and the
    # owner settles the two with `sel`.
    SIG_SEEN[] = nothing
    PO.resolve_deferred_quantities(EntropicValueatRisk(; alpha = rule), PR60, slv)
    @test SIG_SEEN[].solved
    # A measure that carries no solver at all gives the rule the one it was handed.
    SIG_SEEN[] = nothing
    PO.resolve_deferred_quantities(ConditionalValueatRisk(; alpha = rule), PR60, slv)
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
                                       PR120)
    @test v.alpha ≈ 4 / 120
    @test v.alg.sigma ≈ PR120.sigma

    vr = PO.resolve_deferred_quantities(ValueatRiskRange(; alpha = rule,
                                                         beta = ScenarioCount(; n = 6),
                                                         alg = DistributionValueatRisk(;
                                                                                       sigma = ce)),
                                        PR120)
    @test vr.alpha ≈ 4 / 120
    @test vr.beta ≈ 6 / 120
    @test vr.alg.sigma ≈ PR120.sigma

    # A stated number beside a deferred formulation still resolves the formulation alone.
    v2 = PO.resolve_deferred_quantities(ValueatRisk(;
                                                    alg = DistributionValueatRisk(;
                                                                                  sigma = ce)),
                                        PR120)
    @test v2.alpha == 0.05
    @test v2.alg.sigma ≈ PR120.sigma
end

@testset "Significance calibration: alpha and kappa are a travelling pair" begin
    # `EntropyBudget` reads the significance level of its sibling slot, so the owner
    # resolves `alpha` first and states the number in the context. Without it the rule has
    # nothing to invert and refuses by name.
    lone = EntropyBudget(; target = -1.2)
    @test !hasfield(EntropyBudget, :alpha)
    err = try
        lone(:kappa, PR60, nothing, nothing, CalibrationContext())
        nothing
    catch e
        e
    end
    @test isa(err, PO.IsNothingError)
    @test occursin("CalibrationContext.alpha", sprint(showerror, err))

    # Inside the measure the pair resolves together, and the κ it produces is the one the
    # rule gives when the caller states the same `alpha` themselves.
    r = RelativisticValueatRisk(; alpha = sig_tail(), kappa = def_tail(-1.2))
    out = PO.resolve_deferred_quantities(r, PR60)
    @test out.alpha ≈ 3 / 60
    bound = EntropyBudget(; target = -1.2)
    @test out.kappa ≈
          bound(:kappa, PR60, nothing, nothing, CalibrationContext(; alpha = out.alpha))
    @test 0 < out.kappa < 1

    # The Range type carries one pair per tail, and neither side reads the other's number.
    # The band the target must lie in moves with the level, so the two ends take two
    # targets: `alpha = 0.05` reaches `(-1.334, -1.099)` and `beta = 0.1` reaches
    # `(-2.917, -1.792)`. A single target that suits both ends does not exist here, which is
    # itself the reason the rule refuses a target outside the band rather than clamping.
    rg = RelativisticValueatRiskRange(; alpha = sig_tail(3), kappa_a = def_tail(-1.2),
                                      beta = sig_head(6), kappa_b = def_head(-2.0))
    og = PO.resolve_deferred_quantities(rg, PR60)
    @test og.alpha ≈ 3 / 60
    @test og.beta ≈ 6 / 60
    @test og.kappa_a ≈ EntropyBudget(; target = -1.2)(:kappa_a, PR60, nothing, nothing,
                                                      CalibrationContext(; alpha = og.alpha))
    @test og.kappa_b ≈ EntropyBudget(; target = -2.0)(:kappa_b, PR60, nothing, nothing,
                                                      CalibrationContext(; alpha = og.beta))
    @test og.kappa_a != og.kappa_b

    # The head end really reads its own `beta`. The head's target lies outside the band the
    # tail's level reaches, so the same target bound to `alpha` has no answer at all: the
    # pairing is not an accident of the two ends holding one number.
    @test_throws DomainError EntropyBudget(; target = -2.0)(:kappa_b, PR60, nothing,
                                                            nothing,
                                                            CalibrationContext(;
                                                                               alpha = og.alpha))

    # The context reaches the rule through the resolver, and a stated number ignores it.
    # No significance rule reads a sibling, so the field is dead weight for that family.
    ctx = CalibrationContext(; alpha = 0.05)
    @test PO.resolve_calibration_slot(0.3, :kappa, PR60, nothing, nothing, ctx) == 0.3
    st = sig_tail()
    @test PO.resolve_calibration_slot(st, :alpha, PR60, nothing, nothing, ctx) ==
          PO.resolve_calibration_slot(st, :alpha, PR60, nothing, nothing)
    # The rule is not rebuilt on the way in: the resolver calls it and hands the context
    # to it, so the occupant a per-type method compares against never moves.
    dt = def_tail(-1.2)
    @test !hasfield(typeof(dt), :alpha)
    @test PO.resolve_calibration_slot(dt, :kappa, PR60, nothing, nothing, ctx) ≈
          dt(:kappa, PR60, nothing, nothing, ctx)
    @test PO.resolve_calibration_slot(def_head(-1.2), :kappa_b, PR60, nothing, nothing,
                                      ctx) ≈
          EntropyBudget(; target = -1.2)(:kappa_b, PR60, nothing, nothing, ctx)
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
                                       PR120)
    @test b.alpha ≈ 4 / 120

    # The container declares the builder, so the derived recursion reaches it and neither
    # the builder nor the container needs `@propagatable`.
    owa = OrderedWeightsArray(; w = OrderedWeightsArrayTailGini(; alpha = rule))
    @test PO.deferred_slots(owa) == (; w = owa.w)
    @test PO.calibration_slots(owa) == (; w = owa.w)
    o = PO.factory(owa, PR120)
    @test o.w.alpha ≈ 4 / 120
    @test o.w.alpha_i == 1e-4
    @test isa(PO.resolve_deferred_quantities(owa, PR120).w.alpha, Number)

    # A weight vector and a plain function defer nothing, so both are carried through.
    plain = OrderedWeightsArray()
    @test PO.factory(plain, PR120) === plain
    vec = OrderedWeightsArray(; w = owa_gmd(120))
    @test PO.factory(vec, PR120) === vec

    # The Range container wraps `w2` as `reverse ∘ w2`, so the builder a rule sits in is the
    # composition's inner half. Without a method for the composition the gain-side rule
    # would never resolve while the loss-side one did.
    rg = OrderedWeightsArrayRange(; w1 = OrderedWeightsArrayTailGini(; alpha = rule),
                                  w2 = OrderedWeightsArrayTailGini(; alpha = rule))
    @test isa(rg.w2, ComposedFunction)
    @test PO.calibration_slots(rg) == (; w1 = rg.w1, w2 = rg.w2)
    @test PO.calibration_slots(rg.w2) == (; inner = rg.w2.inner)
    org = PO.factory(rg, PR120)
    @test org.w1.alpha ≈ 4 / 120
    @test isa(org.w2, ComposedFunction)
    @test org.w2.inner.alpha ≈ 4 / 120
    # `rev` is stored as a done flag, so the positional rebuild reverses nothing twice.
    @test org.rev === true
    @test org.w2(120) ≈ reverse(org.w1(120))

    # The resolved container evaluates, and it reports the number a caller who states the
    # calibrated `alpha` themselves gets.
    stated = OrderedWeightsArray(; w = OrderedWeightsArrayTailGini(; alpha = 4 / 120))
    @test expected_risk(owa, W4, PR120) ≈ expected_risk(stated, W4, X120)
end

@testset "Significance calibration: the tail-Gini ordering is refused at fold time" begin
    # The joint bound is the whole of the ordering validation. A rule states no value at
    # construction, so the pair is checked when the rebuild runs.
    rule = sig_tail(3)
    ok = OrderedWeightsArrayTailGini(; alpha_i = 0.01, alpha = rule)
    @test ok.alpha === rule
    @test PO.resolve_deferred_quantities(ok, PR60).alpha ≈ 0.05

    # `alpha_i` above the number the rule returns builds, and is refused at fold time by the
    # same constructor a caller's own number meets. At `T = 60` the rule gives `0.05` and
    # the pair holds; at `T = 120` it gives `0.025` and the pair does not.
    late = OrderedWeightsArrayTailGini(; alpha_i = 0.04, alpha = rule)
    @test PO.resolve_deferred_quantities(late, PR60).alpha ≈ 0.05
    @test_throws DomainError PO.resolve_deferred_quantities(late, PR120)

    # An `alpha_i` outside the unit interval is still refused at construction, so the rule
    # branch checks the inner bound on its own rather than skipping it.
    @test_throws DomainError OrderedWeightsArrayTailGini(; alpha_i = 1.5, alpha = rule)
    @test_throws DomainError OrderedWeightsArrayTailGiniRange(; beta_i = 1.5, alpha = rule)

    # The Range twin refuses on whichever end breaks the pair.
    lateb = OrderedWeightsArrayTailGiniRange(; alpha_i = 1e-4, alpha = rule, beta_i = 0.04,
                                             beta = sig_head(3))
    @test PO.resolve_deferred_quantities(lateb, PR60).beta ≈ 0.05
    @test_throws DomainError PO.resolve_deferred_quantities(lateb, PR120)

    # A stated pair keeps the check it always had.
    @test_throws DomainError OrderedWeightsArrayTailGini(; alpha_i = 0.2, alpha = 0.1)
end

@testset "Significance calibration: the value-level entry point refuses a rule" begin
    # A bare returns matrix carries no sample size, no moments and no observation weights,
    # so a rule cannot be run against it. The refusal names the slot and the way out.
    r = ConditionalValueatRisk(; alpha = sig_tail())
    err = try
        expected_risk(r, W4, X60)
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
        expected_risk(owa, W4, X60)
        nothing
    catch e
        e
    end
    @test isa(err2, ArgumentError)
    @test occursin("OrderedWeightsArrayConditionalValueatRisk.alpha", err2.msg)

    # Given the prior result the measure is resolved first, so the same call succeeds and
    # reports the number a caller who states the calibrated level themselves gets.
    @test expected_risk(r, W4, PR120) ≈
          expected_risk(ConditionalValueatRisk(; alpha = 3 / 120), W4, X120)
    # A stated number reaches the functor untouched on both routes.
    plain = ConditionalValueatRisk()
    @test expected_risk(plain, W4, X60) ≈ expected_risk(plain, W4, PR60)
end

@testset "Significance calibration: the JuMP route prices the calibrated level" begin
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = "verbose" => false)
    rd = ReturnsResult(; nx = string.(1:4), X = X120)

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
    rd = ReturnsResult(; nx = string.(1:4), X = X120)
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
        rdi = ReturnsResult(; nx = string.(1:4), X = X120[idx, :])
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
    out = PO.resolve_deferred_quantities(tg, PR60)
    @test out.alpha ≈ 0.05
    @test out.beta ≈ 0.05
    @test out.alpha_i === tg.alpha_i
    @test out.beta_i === tg.beta_i
    @test out.a_sim === tg.a_sim
    @test out.b_sim === tg.b_sim

    # A measure whose slots all resolved to themselves is the object the caller passed in,
    # so the common case allocates nothing.
    stated = OrderedWeightsArrayTailGiniRange(; alpha = 0.05, beta = 0.05)
    @test PO.resolve_deferred_quantities(stated, PR60) === stated
end
