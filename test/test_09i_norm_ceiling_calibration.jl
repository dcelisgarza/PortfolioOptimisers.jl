#=
A **Norm Ceiling** is the upper bound on a norm of the weight vector, the quantity the
`l2c`, `lpc` and `linfc` fields of `JuMPOptimiser` hold. It is NOT an **Ambiguity Radius**.
A radius is the coefficient of a norm penalty in the objective; a ceiling bounds that norm
in a constraint. The reciprocal of a ceiling is a floor on the effective number of assets,
which is a diversification statement rather than a statement about the set of measures the
model prices.

ADR 0095 rules that the role names the quantity, so the ceiling takes its own family, its
own role and its own bound rather than borrowing `AmbiguityRadiusCalibration`.
`test_09h_ambiguity_calibration.jl` covers the radius family this one sits beside.

One field carries both readings. `LpRegularisation` is a penalty in `JuMPOptimiser.lp` and
a norm constraint in `JuMPOptimiser.lpc`, so its `val` cannot be bounded to one role. The
bound admits both, and the FIELD THAT HOLDS THE TERM refuses the role that has no reading
there. That is the one place in the library where a role is settled after construction.
=#
using Clarabel, JuMP
const PO = PortfolioOptimisers

const RNG = StableRNG(246813579)
const X60 = randn(RNG, 60, 4)
const PR60 = prior(EmpiricalPrior(), X60)
const RD60 = ReturnsResult(; nx = string.(1:4), X = X60)
const SLV = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                   settings = "verbose" => false)

# A ceiling rule with no type at all. The `alg` bound admits a bare `Function` for exactly
# this case, and this one reports its arguments so the resolver's order is asserted.
const CEIL_SEEN = Ref{Any}(nothing)
function probe_ceiling(key::Symbol, pr::PO.AbstractPriorResult, w, slv)
    CEIL_SEEN[] = (; key = key, weighted = !isnothing(w), solved = !isnothing(slv))
    return 1 / sqrt(size(pr.X, 2))
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

    # The role is an Estimator, not an Algorithm, so a role inside another role's `alg`
    # field is refused by that field's bound.
    @test NormCeilingCalibration <: PO.AbstractCalibrationEstimator
    @test !(NormCeilingCalibration <: PO.AbstractCalibrationAlgorithm)
    @test EffectiveAssetFloor <: PO.AbstractNormCeilingCalibrationAlgorithm

    # The abstract type is not exported: an export is public API, and the convention is
    # that an abstract type is not one.
    @test :AbstractNormCeilingCalibrationAlgorithm ∉ names(PortfolioOptimisers)

    # The two concrete names are exported, on the same terms as the radius family's.
    @test :NormCeilingCalibration ∈ names(PortfolioOptimisers)
    @test :EffectiveAssetFloor ∈ names(PortfolioOptimisers)

    # A ceiling names no end of a distribution, so the family carries one role and
    # `mirror_role` has nothing to carry across.
    @test_throws MethodError PO.mirror_role(NormCeilingCalibration(; alg = probe_ceiling))
end

@testset "Norm ceiling calibration: the bounds refuse the wrong role" begin
    crole = NormCeilingCalibration(; alg = EffectiveAssetFloor())
    rrole = AmbiguityRadiusCalibration(; alg = RateRadius(; c = 0.2))

    # The `alg` bound admits a rule of the family and a plain function, and nothing else.
    # A role placed inside another role's `alg` is refused by the same route.
    @test isa(NormCeilingCalibration(; alg = EffectiveAssetFloor()).alg,
              EffectiveAssetFloor)
    @test isa(NormCeilingCalibration(; alg = probe_ceiling).alg, Function)
    @test_throws TypeError NormCeilingCalibration(; alg = RateRadius(; c = 0.2))
    @test_throws TypeError NormCeilingCalibration(; alg = crole)
    @test_throws TypeError AmbiguityRadiusCalibration(; alg = EffectiveAssetFloor())

    # The slot bound pairs `Number` with ONE role, so the refusal is at construction and
    # no guard method is written for it.
    @test isa(1.0, PO.Num_NormCeilCal)
    @test isa(crole, PO.Num_NormCeilCal)
    @test !isa(rrole, PO.Num_NormCeilCal)
    @test !isa(crole, PO.Num_AmbRadCal)

    # `LpRegularisation.val` is the one dual-use slot, so its bound admits both roles.
    @test isa(crole, PO.Num_AmbRadNormCeilCal)
    @test isa(rrole, PO.Num_AmbRadNormCeilCal)
    @test isa(1.0, PO.Num_AmbRadNormCeilCal)
    @test !isa(AmbiguityTailWeightCalibration(; alg = probe_ceiling),
               PO.Num_AmbRadNormCeilCal)
end

@testset "Norm ceiling calibration: EffectiveAssetFloor reads the universe" begin
    # `fraction` is checked, and it is a fraction rather than a count.
    @test EffectiveAssetFloor().fraction == 0.5
    @test isnothing(EffectiveAssetFloor().p)
    @test_throws DomainError EffectiveAssetFloor(; fraction = 0)
    @test_throws DomainError EffectiveAssetFloor(; fraction = -0.1)
    @test_throws DomainError EffectiveAssetFloor(; fraction = 1.5)
    @test_throws DomainError EffectiveAssetFloor(; fraction = Inf)
    @test EffectiveAssetFloor(; fraction = 1).fraction == 1

    # A norm order below 1 is not a norm order.
    @test_throws DomainError EffectiveAssetFloor(; fraction = 0.5, p = 0.5)
    @test EffectiveAssetFloor(; fraction = 0.5, p = 3).p == 3

    # The prior carries four assets, so half of the universe is two effective assets. The
    # ceiling is the reciprocal form the constraint's own docstring states.
    f2 = EffectiveAssetFloor(; fraction = 0.5, p = 2)
    f3 = EffectiveAssetFloor(; fraction = 0.5, p = 3)
    fi = EffectiveAssetFloor(; fraction = 0.5, p = Inf)
    @test f2(:l2c, PR60, PR60.w, nothing) ≈ 2^(-1 / 2)
    @test f3(:lpc, PR60, PR60.w, nothing) ≈ 2^(-1 / 3)
    @test fi(:linfc, PR60, PR60.w, nothing) ≈ 1 / 2

    # The whole universe is the whole universe, so an equally weighted portfolio is the
    # only point the 2-norm ceiling admits.
    @test EffectiveAssetFloor(; fraction = 1, p = 2)(:l2c, PR60, nothing, nothing) ≈ 1 / 2

    # The floor moves with the universe rather than with a count the caller pinned. That is
    # the whole reason a rule beats a number.
    pr2 = prior(EmpiricalPrior(), randn(RNG, 60, 9))
    @test f2(:l2c, pr2, nothing, nothing) ≈ (0.5 * 9)^(-1 / 2)

    # A universe count is not a sample count, so the rule ignores the observation weights.
    wts = pweights(range(; start = 1, stop = 2, length = 60))
    @test f2(:l2c, PR60, wts, nothing) == f2(:l2c, PR60, nothing, nothing)

    # The order belongs to the constraint. A rule resolved where nothing bound one names
    # the verb that should have filled it.
    @test_throws ArgumentError EffectiveAssetFloor()(:l2c, PR60, nothing, nothing)
end

@testset "Norm ceiling calibration: bind_norm_order carries the constraint's order" begin
    crole = NormCeilingCalibration(; alg = EffectiveAssetFloor(; fraction = 0.5))

    # A stated number crosses unchanged, and so does a caller's own plain function: neither
    # reads an order, and the default is the identity.
    @test PO.bind_norm_order(0.4, 2) == 0.4
    @test isnothing(PO.bind_norm_order(nothing, 2))
    fn_role = NormCeilingCalibration(; alg = probe_ceiling)
    @test PO.bind_norm_order(fn_role, 2).alg === probe_ceiling

    # The role is rebuilt around the bound rule, so the rule never sees the role it stands
    # in and the role never has to know what the rule reads.
    bound = PO.bind_norm_order(crole, 3)
    @test isa(bound, NormCeilingCalibration)
    @test bound.alg.p == 3
    @test bound.alg.fraction == 0.5

    # The constraint's order WINS. A rule that already carries one has it replaced, because
    # the rule cannot know which of the three sites it reached.
    @test PO.bind_norm_order(EffectiveAssetFloor(; p = 2), Inf).p == Inf

    # Binding then resolving is the whole of the channel, and the key reaches the rule.
    CEIL_SEEN[] = nothing
    @test PO.resolve_calibration_slot(PO.bind_norm_order(fn_role, 2), :l2c, PR60, PR60.w,
                                      SLV) ≈ 1 / 2
    @test CEIL_SEEN[].key == :l2c
    @test CEIL_SEEN[].solved
end

@testset "Norm ceiling calibration: the LpRegularisation route settles the reading" begin
    crole = NormCeilingCalibration(; alg = EffectiveAssetFloor(; fraction = 0.5))
    rrole = AmbiguityRadiusCalibration(; alg = RateRadius(; c = 0.2))

    # One term, one bound, two readings. The constructor admits both roles because it
    # cannot know which field the term is about to land in.
    @test isa(LpRegularisation(; p = 3, val = crole).val, NormCeilingCalibration)
    @test isa(LpRegularisation(; p = 3, val = rrole).val, AmbiguityRadiusCalibration)

    # A ceiling has no reading as a penalty coefficient, so the penalty route refuses it.
    # A number stays legal on both routes.
    @test isnothing(PO.assert_penalty_coefficient_role(1e-3))
    @test isnothing(PO.assert_penalty_coefficient_role(LpRegularisation(; val = rrole)))
    @test isnothing(PO.assert_penalty_coefficient_role([LpRegularisation(; val = rrole)]))
    @test_throws ArgumentError PO.assert_penalty_coefficient_role(crole)
    @test_throws ArgumentError PO.assert_penalty_coefficient_role(LpRegularisation(;
                                                                                   val = crole))
    @test_throws ArgumentError PO.assert_penalty_coefficient_role([LpRegularisation(;
                                                                                    val = crole)])

    # A radius has no reading as a ceiling, so the constraint route refuses it.
    @test isnothing(PO.assert_norm_ceiling_role(1e-3))
    @test isnothing(PO.assert_norm_ceiling_role(LpRegularisation(; val = crole)))
    @test isnothing(PO.assert_norm_ceiling_role([LpRegularisation(; val = crole)]))
    @test_throws ArgumentError PO.assert_norm_ceiling_role(rrole)
    @test_throws ArgumentError PO.assert_norm_ceiling_role(LpRegularisation(; val = rrole))
    @test_throws ArgumentError PO.assert_norm_ceiling_role([LpRegularisation(; val = rrole)])

    # `norm_ceiling_factory` is the constraint route's own verb. It binds the TERM's order,
    # so one rule serves several terms that carry different orders.
    terms = [LpRegularisation(; p = 2, val = crole), LpRegularisation(; p = 3, val = crole)]
    out = PO.norm_ceiling_factory(terms, PR60, SLV)
    @test out[1].val ≈ 2^(-1 / 2)
    @test out[2].val ≈ 2^(-1 / 3)
    @test out[1].p == 2
    @test out[2].p == 3

    # A stated number is carried through by identity rather than rebuilt, and the fallback
    # returns anything else unchanged.
    stated = LpRegularisation(; p = 3, val = 0.6)
    @test PO.norm_ceiling_factory(stated, PR60, SLV) === stated
    @test isnothing(PO.norm_ceiling_factory(nothing, PR60, SLV))

    # The penalty route refuses on its own, so a term that reaches the objective by another
    # path is caught there too.
    @test_throws ArgumentError PO.factory(LpRegularisation(; val = crole), PR60, SLV)
    @test PO.factory(LpRegularisation(; p = 3, val = rrole), PR60, SLV).val ≈ 0.2 / sqrt(60)
end

@testset "Norm ceiling calibration: JuMPOptimiser refuses the wrong role per field" begin
    crole = NormCeilingCalibration(; alg = EffectiveAssetFloor())
    rrole = AmbiguityRadiusCalibration(; alg = RateRadius(; c = 0.2))

    # `l2c` and `linfc` are bounded to the ceiling role alone, so a radius is refused by the
    # signature itself, at the point where the caller wrote it.
    @test_throws TypeError JuMPOptimiser(; slv = SLV, l2c = rrole)
    @test_throws TypeError JuMPOptimiser(; slv = SLV, linfc = rrole)
    @test_throws TypeError JuMPOptimiser(; slv = SLV,
                                         l2c = AmbiguityTailWeightCalibration(;
                                                                              alg = probe_ceiling))

    # `lp` and `lpc` share one term type, so the FIELD refuses the role that has no reading
    # in it. This is the first point at which the reading is known.
    @test_throws ArgumentError JuMPOptimiser(; slv = SLV,
                                             lp = LpRegularisation(; val = crole))
    @test_throws ArgumentError JuMPOptimiser(; slv = SLV,
                                             lpc = LpRegularisation(; val = rrole))
    @test_throws ArgumentError JuMPOptimiser(; slv = SLV,
                                             lp = [LpRegularisation(; val = crole)])

    # Every correct pairing is accepted, and the roles reach the fields untouched.
    opt = JuMPOptimiser(; slv = SLV, pe = PR60, l2c = crole, linfc = crole,
                        lpc = LpRegularisation(; p = 3, val = crole),
                        lp = LpRegularisation(; p = 3, val = rrole), l1 = rrole)
    @test isa(opt.l2c, NormCeilingCalibration)
    @test isa(opt.linfc, NormCeilingCalibration)
    @test isa(opt.lpc.val, NormCeilingCalibration)

    # Neither the weights factory nor the cluster slice holds a prior result, so both carry
    # a rule through untouched: it resolves against the cluster's own prior at assembly.
    @test PO.factory(opt, fill(0.25, 4)).l2c === crole
    @test PO.port_opt_view(opt, 1:3, X60).linfc === crole
end

@testset "Norm ceiling calibration: assembly reaches the stated number" begin
    crole = NormCeilingCalibration(; alg = EffectiveAssetFloor(; fraction = 0.5))
    function build(opt)
        mr = MeanRisk(; r = Variance(), opt = opt)
        attrs = PO.processed_jump_optimiser_attributes(mr.opt, RD60)
        model = JuMP.Model()
        PO.set_model_scales!(model, mr.opt.sc, mr.opt.so)
        PO.set_maximum_ratio_factor_variables!(model, mr.obj)
        PO.set_w!(model, attrs.pr.X, mr.wi)
        PO.set_weight_constraints!(model, attrs.wb, mr.opt)
        PO.assemble_jump_model!(model, mr, mr.opt, attrs, RD60, mr.r, mr.obj)
        return model
    end

    # The rule reaches the three constraints as the number a caller would have written by
    # hand. Four assets and a half-universe floor make two effective assets.
    mrule = build(JuMPOptimiser(; slv = SLV, pe = PR60, l2c = crole, linfc = crole,
                                lpc = LpRegularisation(; p = 3, val = crole)))
    mnum = build(JuMPOptimiser(; slv = SLV, pe = PR60, l2c = 2^(-1 / 2), linfc = 1 / 2,
                               lpc = LpRegularisation(; p = 3, val = 2^(-1 / 3))))
    @test JuMP.normalized_rhs(mrule[:cl2c]) ≈ JuMP.normalized_rhs(mnum[:cl2c])
    @test JuMP.normalized_rhs(mrule[:clinfc]) ≈ JuMP.normalized_rhs(mnum[:clinfc])
    @test JuMP.normalized_rhs(mrule[:clpc_bnd_1]) ≈ JuMP.normalized_rhs(mnum[:clpc_bnd_1])
    @test JuMP.normalized_rhs(mrule[:cl2c]) ≈ 2^(-1 / 2)
    @test JuMP.normalized_rhs(mrule[:clinfc]) ≈ 1 / 2
    @test JuMP.normalized_rhs(mrule[:clpc_bnd_1]) ≈ 2^(-1 / 3)

    # A model that names no ceiling carries no ceiling constraint, so the widening added no
    # constraint to a caller who states nothing.
    plain = build(JuMPOptimiser(; slv = SLV, pe = PR60))
    @test !haskey(JuMP.object_dictionary(plain), :cl2c)
    @test !haskey(JuMP.object_dictionary(plain), :clinfc)

    # The whole model still solves, so the calibrated ceilings are feasible together.
    res = optimise(MeanRisk(; r = Variance(),
                            opt = JuMPOptimiser(; slv = SLV, pe = PR60, l2c = crole)))
    @test isa(res.w, AbstractVector)
    @test sum(res.w) ≈ 1
    @test norm(res.w, 2) <= 2^(-1 / 2) + sqrt(eps())
end

#=
`l2c` and `linfc` are bounded `TD_Option{<:Num_NormCeilCal}`, so one field now carries two
deferral channels. This is the same open case ADR 0030 never considered for `l1` and
`linf`, and this testset records what the code does rather than ratifying a design.

The two verbs run at different points and neither knows about the other: the period
selection runs in `update_time_dependent_fields`, before any prior is fitted, and the
calibration resolution runs at assembly, against the prior of the period that was selected.
=#
@testset "Norm ceiling calibration: the schedule selects, then the rule runs" begin
    crole = NormCeilingCalibration(; alg = EffectiveAssetFloor(; fraction = 0.5))
    td = TimeDependent([crole, 0.6]; default = 0.6)
    @test isa(td, PO.TD_Option{<:PO.Num_NormCeilCal})

    opt = JuMPOptimiser(; slv = SLV, l2c = td)
    @test PO.time_dependent_fields(opt) == (:l2c,)

    # The selection carries the schedule's occupant out unchanged. It does not resolve it,
    # and it has no prior result with which it could.
    ctx1 = TimeDependentContext(; i = 1, n = 2, rd = RD60, train_idx = [1:20, 1:40],
                                test_idx = [21:40, 41:60])
    ctx2 = TimeDependentContext(; i = 2, n = 2, rd = RD60, train_idx = [1:20, 1:40],
                                test_idx = [21:40, 41:60])
    @test PO.time_dependent_value(td, ctx1) === crole
    @test PO.time_dependent_value(td, ctx2) == 0.6

    # The selected rule then resolves against whichever prior the fold produced, so a
    # schedule and a rule compose rather than fight.
    sel = PO.bind_norm_order(PO.time_dependent_value(td, ctx1), 2)
    @test PO.resolve_calibration_slot(sel, :l2c, PR60, nothing) ≈ 2^(-1 / 2)

    # A schedule is not a role, so the constructor's range check skips it and the field
    # falls back to its own default outside every fold loop.
    @test PO.reset_time_dependent_fields(opt).l2c == 0.6
end
