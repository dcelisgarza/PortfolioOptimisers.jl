#=
Three **Calibration Rules** stand in the `alg` field of a role type, and each one computes the
quantity of the slot its role addresses. `test_09f_calibration_slot.jl` covers the mechanism
that carries them; this file covers the rules themselves.

`ScenarioCount` and `RateSignificance` compute a significance level, and `EntropyBudget`
computes a Kaniadakis deformation parameter. The first reads the effective observation
weights, the second reads the raw row count, and the third reads its sibling `alpha` through
`bind_alpha`.

No rule carries a range check of its own. Each returns the quantity of the slot it stands in,
so the slot owner's constructor is the whole validation and a value outside the slot's range
is refused there, at fold time. `EntropyBudget` carries the one check any of them carries,
and it is a different claim: that the equation it inverts has a root at all. A target the
band does not reach leaves the sweep at an end of the interval, where the parameter is far
too small or too large to be the answer and yet still inside the range the slot owner admits.

Issue #582 ships the rules. Issue #583 widens the slots that hold them, so every call below
states the resolution by hand.
=#
const PO = PortfolioOptimisers

# Two samples of different length, so that every rule is stated at two values of `T`.
const RNG = StableRNG(246813579)
const X60 = randn(RNG, 60, 4)
const PR60 = prior(EmpiricalPrior(), X60)
const PR70 = prior(EmpiricalPrior(), randn(RNG, 70, 4))
const PR120 = prior(EmpiricalPrior(), randn(RNG, 120, 4))

@testset "Calibration rules: the three rules join the two families" begin
    # Each rule subtypes the family whose quantity it computes, and no rule subtypes both.
    @test ScenarioCount <: PO.AbstractSignificanceCalibrationAlgorithm
    @test RateSignificance <: PO.AbstractSignificanceCalibrationAlgorithm
    @test EntropyBudget <: PO.AbstractDeformationCalibrationAlgorithm
    @test !(EntropyBudget <: PO.AbstractSignificanceCalibrationAlgorithm)
    @test !(ScenarioCount <: PO.AbstractDeformationCalibrationAlgorithm)

    # The family is what the `alg` bound checks, so each rule is admitted by one bound only.
    @test isa(ScenarioCount(; n = 25), PO.Func_SigCal)
    @test isa(RateSignificance(), PO.Func_SigCal)
    @test isa(EntropyBudget(; target = -1.3), PO.Func_DefCal)
    @test !isa(ScenarioCount(; n = 25), PO.Func_DefCal)
    @test !isa(EntropyBudget(; target = -1.3), PO.Func_SigCal)

    # A rule goes inside a role, and both roles of its family take it.
    @test SignificanceTailCalibration(; alg = ScenarioCount(; n = 25)).alg ==
          ScenarioCount(; n = 25)
    @test isa(SignificanceHeadCalibration(; alg = RateSignificance()).alg, RateSignificance)
    @test isa(DeformationTailCalibration(; alg = EntropyBudget(; target = -1.3)).alg,
              EntropyBudget)
    @test isa(DeformationHeadCalibration(; alg = EntropyBudget(; target = -1.3)).alg,
              EntropyBudget)

    # The wrong family is refused at construction, by the role's own bound.
    @test_throws TypeError SignificanceTailCalibration(; alg = EntropyBudget(; target = -1))
    @test_throws TypeError DeformationTailCalibration(; alg = ScenarioCount(; n = 25))

    # The three rules are caller-facing, because a caller states one directly.
    exported = names(PortfolioOptimisers)
    @test :ScenarioCount in exported
    @test :RateSignificance in exported
    @test :EntropyBudget in exported

    # The positional inner constructor is the route a rebuild takes.
    @test ScenarioCount(25).n == 25
    @test RateSignificance(2).c == 2
    @test EntropyBudget(-1.3, 0.05).alpha == 0.05
end

@testset "Calibration rules: `ScenarioCount` fixes the count, not the probability" begin
    rule = ScenarioCount(; n = 15)

    # The tail holds `n` observations, so `alpha * T` is the count the caller stated and
    # `ceil(alpha * T)` returns it whatever `T` is.
    a60 = rule(:alpha, PR60, nothing, nothing)
    a120 = rule(:alpha, PR120, nothing, nothing)
    @test a60 == 15 / 60
    @test a120 == 15 / 120
    @test ceil(a60 * 60) == 15
    @test ceil(a120 * 120) == 15

    # The sample doubled, so the probability halved. This is the whole reason for the rule:
    # a stated `alpha` would have left half as many observations in the tail.
    @test a120 == a60 / 2

    # The key does not change the count, so a tail slot and a head slot that carry one rule
    # resolve to one number.
    @test rule(:beta, PR60, nothing, nothing) == a60

    # Uniform weights carry the same information as the unweighted sample, so Kish's
    # effective sample size is the row count and the answer does not move.
    @test rule(:alpha, PR60, pweights(fill(1 / 60, 60)), nothing) ≈ a60

    # Uneven weights carry less, so the effective sample is shorter and the tail must take a
    # larger share of it to hold the same 15 observations.
    uneven = pweights([fill(3.0, 30); fill(1.0, 30)])
    aw = rule(:alpha, PR60, uneven, nothing)
    @test aw ≈ 15 / (sum(uneven)^2 / sum(abs2, uneven))
    @test aw > a60

    # The solver reaches the rule and this one ignores it, which is the fourth argument's
    # whole contract here.
    @test rule(:alpha, PR60, nothing, Solver(; name = :probe, solver = nothing)) == a60

    # The rule carries no range check, so a count larger than the sample returns a value the
    # slot owner refuses rather than one the rule refuses.
    @test ScenarioCount(; n = 90)(:alpha, PR60, nothing, nothing) == 1.5
end

@testset "Calibration rules: `RateSignificance` falls with the square root of `T`" begin
    rule = RateSignificance()

    # The default coefficient is the plain `1/sqrt(T)` rate.
    @test rule.c == 1
    a60 = rule(:alpha, PR60, nothing, nothing)
    a120 = rule(:alpha, PR120, nothing, nothing)
    @test a60 == inv(sqrt(60))
    @test a120 == inv(sqrt(120))

    # The sample doubled, so the probability fell by `sqrt(2)` and the tail's expected count
    # ROSE by the same factor. A longer sample buys a further tail, not only a fuller one.
    @test a120 ≈ a60 / sqrt(2)
    @test a120 * 120 ≈ sqrt(2) * (a60 * 60)

    # The coefficient scales the rate.
    @test RateSignificance(; c = 2)(:alpha, PR60, nothing, nothing) == 2 * a60

    # The rule reads the RAW row count, so the weights do not move it. That is the stated
    # difference from `ScenarioCount`: a rate speaks of the length of the record.
    @test rule(:alpha, PR60, pweights([fill(3.0, 30); fill(1.0, 30)]), nothing) == a60

    # The key does not change the rate either.
    @test rule(:beta, PR60, nothing, nothing) == a60
end

@testset "Calibration rules: `EntropyBudget` inverts the Kaniadakis logarithm" begin
    target = -1.3
    rule = EntropyBudget(; target = target, alpha = 0.05)

    # The returned parameter meets the target: it is the coefficient `RRM` multiplies its
    # dual variable by, and the rule solves for the deformation that prices it.
    k60 = rule(:kappa, PR60, nothing, nothing)
    @test 0 < k60 < 1
    @test PO.kappa_log(inv(0.05 * 60), k60) ≈ target

    # The same budget on a longer sample. The band the logarithm reaches moves with `T`, so
    # the two samples are 60 and 70 rather than 60 and 120.
    k70 = rule(:kappa, PR70, nothing, nothing)
    @test 0 < k70 < 1
    @test PO.kappa_log(inv(0.05 * 70), k70) ≈ target

    # A longer sample reaches the same budget with LESS deformation, which is the direction
    # the rule states: the price is held and the shape moves.
    @test k70 < k60

    # A larger `alpha` widens the tail and moves the answer on one sample.
    @test EntropyBudget(; target = target, alpha = 0.06)(:kappa, PR60, nothing, nothing) !=
          k60

    # The key does not change the budget, and the solver is not needed: the inversion is a
    # scalar one, so this rule never reaches `RRM` itself.
    @test rule(:kappa_a, PR60, nothing, nothing) == k60
    @test rule(:kappa, PR60, nothing, Solver(; name = :probe, solver = nothing)) == k60
    @test rule(:kappa, PR60, pweights(fill(1 / 60, 60)), nothing) == k60

    # The band the coefficient reaches over `(0, 1)` runs from `log(u)` to `sinh(log(u))`,
    # and it moves with the sample. A target past either end has no root, so the rule
    # refuses it rather than returning the end of the sweep. That value would be far too
    # small or too large to be the answer, and yet still inside the range the slot owner
    # admits, so the slot owner cannot catch it.
    band = (PO.kappa_log(inv(0.05 * 60), 1), log(inv(0.05 * 60)))
    @test band[1] < target < band[2]
    @test_throws DomainError EntropyBudget(; target = -0.5, alpha = 0.05)(:kappa, PR60,
                                                                          nothing, nothing)
    @test_throws DomainError EntropyBudget(; target = -5.0, alpha = 0.05)(:kappa, PR60,
                                                                          nothing, nothing)
    msg = try
        EntropyBudget(; target = -0.5, alpha = 0.05)(:kappa, PR60, nothing, nothing)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("EntropyBudget.target", msg)
    @test occursin("T = 60", msg)

    # The band moves with the sample, so the target that suits 60 and 70 observations lies
    # outside the band of 120. That is what the message warns about.
    @test_throws DomainError rule(:kappa, PR120, nothing, nothing)

    # A rule whose sibling never arrived cannot form `inv(alpha * T)`, and it says so.
    unbound = EntropyBudget(; target = target)
    @test isnothing(unbound.alpha)
    @test_throws PO.IsNothingError unbound(:kappa, PR60, nothing, nothing)
    msg = try
        unbound(:kappa, PR60, nothing, nothing)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("EntropyBudget.alpha", msg)
    @test occursin("bind_alpha", msg)
end

@testset "Calibration rules: `bind_alpha` hands the sibling over" begin
    rule = EntropyBudget(; target = -1.3)

    # The rule itself takes the number, and nothing else about it moves.
    bound = PO.bind_alpha(rule, 0.05)
    @test isa(bound, EntropyBudget)
    @test bound.alpha == 0.05
    @test bound.target == rule.target

    # A role is rebuilt around the bound rule, so the verb takes the SLOT and the caller
    # unwraps nothing. Both deformation roles carry it, because a head slot holds a rule too.
    tail = PO.bind_alpha(DeformationTailCalibration(; alg = rule), 0.05)
    @test isa(tail, DeformationTailCalibration)
    @test tail.alg.alpha == 0.05
    head = PO.bind_alpha(DeformationHeadCalibration(; alg = rule), 0.05)
    @test isa(head, DeformationHeadCalibration)
    @test head.alg.alpha == 0.05

    # The default is the identity, so a stated number passes through untouched. That is what
    # lets the slot owner call the verb on every `kappa` slot rather than on some of them.
    @test PO.bind_alpha(0.3, 0.05) === 0.3
    @test PO.bind_alpha(nothing, 0.05) === nothing

    # A rule that reads no sibling passes through as well, whether it is typed or a plain
    # function, and so does the role it stands in.
    probe(::Symbol, ::PO.AbstractPriorResult, ::Any, ::Any) = 0.3
    @test PO.bind_alpha(probe, 0.05) === probe
    @test PO.bind_alpha(DeformationTailCalibration(; alg = probe), 0.05).alg === probe

    # The significance family needs no method of its own: no significance rule reads a
    # sibling, so the identity is already the right answer for it.
    role = SignificanceTailCalibration(; alg = ScenarioCount(; n = 25))
    @test PO.bind_alpha(role, 0.05) === role
end

@testset "Calibration rules: the resolver runs each rule" begin
    # This is the shape #583's per-type method writes. A significance slot resolves on its
    # own; the deformation slot takes the number the significance slot produced.
    slot = SignificanceTailCalibration(; alg = ScenarioCount(; n = 3))
    alpha = PO.resolve_calibration_slot(slot, :alpha, PR60, nothing)
    @test alpha == 3 / 60

    kslot = DeformationTailCalibration(; alg = EntropyBudget(; target = -1.3))
    kappa = PO.resolve_calibration_slot(PO.bind_alpha(kslot, alpha), :kappa, PR60, nothing)
    @test 0 < kappa < 1
    @test PO.kappa_log(inv(alpha * 60), kappa) ≈ -1.3

    # The pair refits on a shorter fold, which is the whole point of a rule over a number.
    fold = prior(EmpiricalPrior(), X60[1:30, :])
    alpha30 = PO.resolve_calibration_slot(slot, :alpha, fold, nothing)
    @test alpha30 == 3 / 30
    @test alpha30 != alpha

    # A rule reached through a plain `Function` needs no wrapper either, and the rate rule
    # resolves through the same verb.
    @test PO.resolve_calibration_slot(SignificanceHeadCalibration(;
                                                                  alg = RateSignificance()),
                                      :beta, PR60, nothing) == inv(sqrt(60))
end
