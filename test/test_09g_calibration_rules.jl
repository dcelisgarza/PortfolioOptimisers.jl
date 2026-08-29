#=
Four **Calibration Rules** stand in the `alg` field of a role type, and each one computes the
quantity of the slot its role addresses. `test_09f_calibration_slot.jl` covers the mechanism
that carries them; this file covers the rules themselves.

`ScenarioCount` and `RateSignificance` compute a significance level, and `EntropyBudget` and
`HillTailDecay` compute a Kaniadakis deformation parameter. The first reads the effective
observation weights, the second reads the raw row count, and the last two read the
probability of their own slot's end through `bind_alpha`.

No rule carries a range check of its own. Each returns the quantity of the slot it stands in,
so the slot owner's constructor is the whole validation and a value outside the slot's range
is refused there, at fold time. The two deformation rules carry the only checks any of them
carry, and each is a different claim: that the quantity exists at all. A target the band does
not reach leaves `EntropyBudget`'s sweep at an end of the interval, where the parameter is
far too small or too large to be the answer and yet still inside the range the slot owner
admits. A pool with no Hill estimate leaves `HillTailDecay` with nothing to invert.

Issue #582 ships the first three rules, and #611 ships `HillTailDecay`. Issue #583 widens the
slots that hold them, so every call below states the resolution by hand.
=#
const PO = PortfolioOptimisers
using Distributions

# Two samples of different length, so that every rule is stated at two values of `T`.
const RNG = StableRNG(246813579)
const X60 = randn(RNG, 60, 4)
const PR60 = prior(EmpiricalPrior(), X60)
const PR70 = prior(EmpiricalPrior(), randn(RNG, 70, 4))
const PR120 = prior(EmpiricalPrior(), randn(RNG, 120, 4))

@testset "Calibration rules: the four rules join the two families" begin
    # Each rule subtypes the family whose quantity it computes, and no rule subtypes both.
    @test ScenarioCount <: PO.AbstractSignificanceCalibrationAlgorithm
    @test RateSignificance <: PO.AbstractSignificanceCalibrationAlgorithm
    @test EntropyBudget <: PO.AbstractDeformationCalibrationAlgorithm
    @test HillTailDecay <: PO.AbstractDeformationCalibrationAlgorithm
    @test !(EntropyBudget <: PO.AbstractSignificanceCalibrationAlgorithm)
    @test !(HillTailDecay <: PO.AbstractSignificanceCalibrationAlgorithm)
    @test !(ScenarioCount <: PO.AbstractDeformationCalibrationAlgorithm)

    # The family is what the `alg` bound checks, so each rule is admitted by one bound only.
    @test isa(ScenarioCount(; n = 25), PO.Func_SigCal)
    @test isa(RateSignificance(), PO.Func_SigCal)
    @test isa(EntropyBudget(; target = -1.3), PO.Func_DefCal)
    @test isa(HillTailDecay(), PO.Func_DefCal)
    @test !isa(ScenarioCount(; n = 25), PO.Func_DefCal)
    @test !isa(EntropyBudget(; target = -1.3), PO.Func_SigCal)
    @test !isa(HillTailDecay(), PO.Func_SigCal)

    # A rule goes inside a role, and both roles of its family take it.
    @test SignificanceTailCalibration(; alg = ScenarioCount(; n = 25)).alg ==
          ScenarioCount(; n = 25)
    @test isa(SignificanceHeadCalibration(; alg = RateSignificance()).alg, RateSignificance)
    @test isa(DeformationTailCalibration(; alg = EntropyBudget(; target = -1.3)).alg,
              EntropyBudget)
    @test isa(DeformationHeadCalibration(; alg = EntropyBudget(; target = -1.3)).alg,
              EntropyBudget)
    @test isa(DeformationTailCalibration(; alg = HillTailDecay()).alg, HillTailDecay)
    @test isa(DeformationHeadCalibration(; alg = HillTailDecay()).alg, HillTailDecay)

    # The wrong family is refused at construction, by the role's own bound.
    @test_throws TypeError SignificanceTailCalibration(; alg = EntropyBudget(; target = -1))
    @test_throws TypeError DeformationTailCalibration(; alg = ScenarioCount(; n = 25))
    @test_throws TypeError SignificanceHeadCalibration(; alg = HillTailDecay())

    # The three rules are caller-facing, because a caller states one directly.
    exported = names(PortfolioOptimisers)
    @test :ScenarioCount in exported
    @test :RateSignificance in exported
    @test :EntropyBudget in exported
    @test :HillTailDecay in exported

    # The positional inner constructor is the route a rebuild takes.
    @test ScenarioCount(25).n == 25
    @test RateSignificance(2).c == 2
    @test EntropyBudget(-1.3, 0.05).alpha == 0.05
    @test HillTailDecay(12, 0.05).kmin == 12
    @test HillTailDecay(12, 0.05).alpha == 0.05

    # `kmin` is a count of order statistics, so a rule that states none is refused at
    # construction. It is a check on the rule's OWN parameter, and not on the parameter the
    # rule returns.
    @test_throws DomainError HillTailDecay(; kmin = 0)
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

    # The second rule that reads a sibling takes the number the same way, and keeps its
    # own field. A head slot's number is the `beta` of its end, and the field is named for
    # the sibling's ROLE rather than for the sibling's spelling.
    hill = PO.bind_alpha(HillTailDecay(; kmin = 12), 0.02)
    @test isa(hill, HillTailDecay)
    @test hill.alpha == 0.02
    @test hill.kmin == 12
    @test PO.bind_alpha(DeformationHeadCalibration(; alg = HillTailDecay()), 0.02).alg.alpha ==
          0.02

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
    rule = HillTailDecay(; kmin = 30, alpha = 0.01)

    # The answer is `1 / ν`, and the band holds the Hill bias, which is upward at this
    # depth on a t draw.
    k3 = rule(:kappa, PRT3, nothing, nothing)
    k4 = rule(:kappa, PRT4, nothing, nothing)
    k6 = rule(:kappa, PRT6, nothing, nothing)
    for (kap, nu) in ((k3, 3), (k4, 4), (k6, 6))
        @test 0 < kap < 1
        @test 0.75 / nu <= kap <= 1.6 / nu
    end

    # The ORDER carries no bias: a heavier tail returns a larger κ, and the three draws
    # separate. This is the reading the rule claims, stated without a tolerance.
    @test k3 > k4 > k6

    # The same tail index at two sample lengths, which is the pair `test_09f` and #582 are
    # stated at. `T` moves the pool and the count with it, and both readings hold the band.
    k4s = rule(:kappa, PRT4S, nothing, nothing)
    @test 0.75 / 4 <= k4s <= 1.6 / 4
    @test k4s != k4

    # `alpha` sets the DEPTH as well as the count. A shallower reading takes in more of the
    # body, which a Student-t makes look heavier than its tail is, so the deeper reading
    # sits closer to the truth.
    shallow = HillTailDecay(; kmin = 30, alpha = 0.05)(:kappa, PRT4, nothing, nothing)
    @test shallow > k4
    @test abs(k4 - 1 / 4) < abs(shallow - 1 / 4)

    # The rule reads `pr.X` ALONE. A prior whose covariance matrix is scaled a hundredfold
    # returns the same number, because a column's dispersion comes from that column. This
    # is the line that separates the rule from one that whitens with `sigma`.
    scaled = LowOrderPrior(; X = PRT4.X, mu = PRT4.mu, sigma = 100 .* PRT4.sigma)
    @test rule(:kappa, scaled, nothing, nothing) == k4

    # A tail index is a statement about the shape of a series rather than about the count of
    # observations behind it, so the rule ignores the observation weights. `ScenarioCount`
    # reads Kish's effective sample size and `RateSignificance` reads the raw row count;
    # this rule reads neither.
    @test rule(:kappa, PRT4, pweights([fill(3.0, 500); fill(1.0, 500)]), nothing) == k4

    # The solver reaches the rule and this one needs none: the estimate is a closed form.
    @test rule(:kappa, PRT4, nothing, Solver(; name = :probe, solver = nothing)) == k4
end

@testset "Calibration rules: `HillTailDecay` answers per end" begin
    rule = HillTailDecay(; kmin = 20, alpha = 0.01)

    # `key` says which end the slot prices, so the two ends of a skewed sample resolve to
    # two different numbers. This is the OPPOSITE of the other three rules, and the reason
    # is that a tail index is a statement about ONE tail.
    ka = rule(:kappa_a, PRSKEW, nothing, nothing)
    kb = rule(:kappa_b, PRSKEW, nothing, nothing)
    @test ka > kb
    @test 0.75 / 3 <= ka <= 1.6 / 3
    @test 0.75 / 9 <= kb <= 1.6 / 9

    # `:kappa` is a loss key, so the scalar measures read the loss tail.
    @test rule(:kappa, PRSKEW, nothing, nothing) == ka

    # `EntropyBudget` states the opposite in its own docstring: a budget is a price the
    # model pays, so it is one number for both ends of the same sample.
    budget = EntropyBudget(; target = -1.3, alpha = 0.05)
    @test budget(:kappa_a, PR60, nothing, nothing) ==
          budget(:kappa_b, PR60, nothing, nothing)
end

@testset "Calibration rules: `HillTailDecay` gives a Range measure two ends" begin
    # The travelling pair costs nothing new: `RelativisticValueatRiskRange` already binds
    # `alpha` to the tail slot and `beta` to the head slot, so the rule needs one `alpha`
    # field and one `bind_alpha` method.
    rg = RelativisticValueatRiskRange(; alpha = 0.01,
                                      kappa_a = DeformationTailCalibration(;
                                                                           alg = HillTailDecay(;
                                                                                               kmin = 20)),
                                      beta = 0.02,
                                      kappa_b = DeformationHeadCalibration(;
                                                                           alg = HillTailDecay(;
                                                                                               kmin = 20)))
    og = PO.resolve_deferred_quantities(rg, PRSKEW)
    @test og.kappa_a ≈
          HillTailDecay(; kmin = 20, alpha = 0.01)(:kappa_a, PRSKEW, nothing, nothing)
    @test og.kappa_b ≈
          HillTailDecay(; kmin = 20, alpha = 0.02)(:kappa_b, PRSKEW, nothing, nothing)
    @test og.kappa_a != og.kappa_b
    @test 0 < og.kappa_a < 1
    @test 0 < og.kappa_b < 1

    # The head end really reads its OWN probability. The same rule bound to the tail's
    # `alpha` answers differently, so the pairing is not an accident of the two ends holding
    # one number.
    @test og.kappa_b !=
          HillTailDecay(; kmin = 20, alpha = 0.01)(:kappa_b, PRSKEW, nothing, nothing)
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
    @test isnothing(unbound.alpha)
    @test unbound.kmin == 30
    @test_throws PO.IsNothingError unbound(:kappa, PR60, nothing, nothing)
    msg = try
        unbound(:kappa, PR60, nothing, nothing)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("HillTailDecay.alpha", msg)
    @test occursin("bind_alpha", msg)

    # The floor. `PR60` pools 240 values, so `alpha = 0.05` leaves 12 order statistics and
    # the default floor of 30 refuses them. A Hill estimate on 12 points moves from fold to
    # fold for no reason in the data.
    @test_throws DomainError HillTailDecay(; alpha = 0.05)(:kappa, PR60, nothing, nothing)
    msg = try
        HillTailDecay(; alpha = 0.05)(:kappa, PR60, nothing, nothing)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("kmin", msg)
    @test occursin("floor at 30", msg)
    @test occursin("= 12", msg)

    # A floor the sample clears returns a number, so the refusal is the floor's and not the
    # sample's.
    @test 0 < HillTailDecay(; kmin = 5, alpha = 0.05)(:kappa, PR60, nothing, nothing) < 1

    # A probability that takes the whole sample leaves no `k + 1`-th value to divide by.
    @test_throws DomainError HillTailDecay(; kmin = 5, alpha = 0.999)(:kappa, PR60, nothing,
                                                                      nothing)
    msg = try
        HillTailDecay(; kmin = 5, alpha = 0.999)(:kappa, PR60, nothing, nothing)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("241 pooled values", msg)
    @test occursin("holds 240", msg)

    # The pool holds fewer positive values than the count asks for, so there is no Hill
    # estimate at all. A sample with fewer losses than `k + 1` is the case that produces it.
    @test_throws DomainError HillTailDecay(; kmin = 5, alpha = 0.95)(:kappa, PRSPIKE,
                                                                     nothing, nothing)
    msg = try
        HillTailDecay(; kmin = 5, alpha = 0.95)(:kappa, PRSPIKE, nothing, nothing)
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
    @test_throws DomainError HillTailDecay(; kmin = 5, alpha = 0.95)(:kappa_b, PRSPIKE,
                                                                     nothing, nothing)

    # An estimate at or below one is a tail with no finite mean, and no admissible κ reads
    # it. The band `(0, 1)` the slot admits IS the condition `â > 1`, so the refusal is the
    # reading rather than a guard bolted onto it.
    @test_throws DomainError HillTailDecay(; kmin = 5, alpha = 0.05)(:kappa, PRHEAVY,
                                                                     nothing, nothing)
    msg = try
        HillTailDecay(; kmin = 5, alpha = 0.05)(:kappa, PRHEAVY, nothing, nothing)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("tail index of 0.6", msg)
    @test occursin("(0, 1)", msg)
end
