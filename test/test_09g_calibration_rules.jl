#=
Five **Calibration Rules** stand in the `alg` field of a role type, and each one computes the
quantity of the slot its role addresses. `test_09f_calibration_slot.jl` covers the mechanism
that carries them; this file covers the rules themselves.

`ScenarioCount` and `RateSignificance` compute a significance level, and `EntropyBudget`,
`HillTailDecay` and `RadialTailDecay` compute a Kaniadakis deformation parameter. The first
reads the effective observation weights, the second reads the raw row count, and the last
three read the probability of their own slot's end through `bind_alpha`.

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
`bind_series` is the verb that carries it, on the shape `bind_alpha` and `bind_norm_order`
already carry. The reading itself does not move: the same standardisation, the same count and
the same estimator run over the drawdown series of each column, in place of the columns.

Issue #582 ships the first three rules, #611 ships `HillTailDecay` and #612 ships
`RadialTailDecay`. Issue #583 widens the slots that hold them, so every call below states the
resolution by hand.
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

    # The family is what the `alg` bound checks, so each rule is admitted by one bound only.
    @test isa(ScenarioCount(; n = 25), PO.Func_SigCal)
    @test isa(RateSignificance(), PO.Func_SigCal)
    @test isa(EntropyBudget(; target = -1.3), PO.Func_DefCal)
    @test isa(HillTailDecay(), PO.Func_DefCal)
    @test isa(RadialTailDecay(), PO.Func_DefCal)
    @test !isa(ScenarioCount(; n = 25), PO.Func_DefCal)
    @test !isa(EntropyBudget(; target = -1.3), PO.Func_SigCal)
    @test !isa(HillTailDecay(), PO.Func_SigCal)
    @test !isa(RadialTailDecay(), PO.Func_SigCal)

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
    @test isa(DeformationTailCalibration(; alg = RadialTailDecay()).alg, RadialTailDecay)
    @test isa(DeformationHeadCalibration(; alg = RadialTailDecay()).alg, RadialTailDecay)

    # The wrong family is refused at construction, by the role's own bound.
    @test_throws TypeError SignificanceTailCalibration(; alg = EntropyBudget(; target = -1))
    @test_throws TypeError DeformationTailCalibration(; alg = ScenarioCount(; n = 25))
    @test_throws TypeError SignificanceHeadCalibration(; alg = HillTailDecay())
    @test_throws TypeError SignificanceTailCalibration(; alg = RadialTailDecay())

    # The five rules are caller-facing, because a caller states one directly.
    exported = names(PortfolioOptimisers)
    @test :ScenarioCount in exported
    @test :RateSignificance in exported
    @test :EntropyBudget in exported
    @test :HillTailDecay in exported
    @test :RadialTailDecay in exported

    # The positional inner constructor is the route a rebuild takes.
    @test ScenarioCount(25).n == 25
    @test RateSignificance(2).c == 2
    @test EntropyBudget(-1.3, 0.05).alpha == 0.05
    @test HillTailDecay(12, 0.05, ReturnsSeries()).kmin == 12
    @test HillTailDecay(12, 0.05, ReturnsSeries()).alpha == 0.05
    @test isa(HillTailDecay(12, 0.05, AbsoluteDrawdownSeries()).series,
              AbsoluteDrawdownSeries)
    @test RadialTailDecay(12, 0.05, ReturnsSeries()).kmin == 12
    @test RadialTailDecay(12, 0.05, ReturnsSeries()).alpha == 0.05
    @test isa(RadialTailDecay(12, 0.05, RelativeDrawdownSeries()).series,
              RelativeDrawdownSeries)

    # The series defaults to the returns, which is what every rule read before the marker
    # existed. A rule that never leaves it reads what it read then.
    @test isa(HillTailDecay().series, ReturnsSeries)
    @test isa(RadialTailDecay().series, ReturnsSeries)

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
    rule = RadialTailDecay(; alpha = 0.05)

    # The answer is `1 / ν`, and the band holds the Hill bias. It is the band the sibling's
    # own draws are read in, and both rules estimate the same quantity.
    k3 = rule(:kappa, PRMVT3, nothing, nothing)
    k4 = rule(:kappa, PRMVT4, nothing, nothing)
    k6 = rule(:kappa, PRMVT6, nothing, nothing)
    for (kap, nu) in ((k3, 3), (k4, 4), (k6, 6))
        @test 0 < kap < 1
        @test 0.75 / nu <= kap <= 1.6 / nu
    end

    # The ORDER carries no bias: a heavier tail returns a larger κ, and the three draws
    # separate. This is the reading the rule claims, stated without a tolerance.
    @test k3 > k4 > k6

    # The same tail index at a second sample length. `T` moves the series and the count with
    # it, and both readings hold the band.
    k4s = rule(:kappa, PRMVT4S, nothing, nothing)
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
    @test rule(:kappa, PRMVT4, pweights([fill(3.0, 2000); fill(1.0, 2000)]), nothing) == k4

    # The solver reaches the rule and this one needs none: the estimate is a closed form.
    @test rule(:kappa, PRMVT4, nothing, Solver(; name = :probe, solver = nothing)) == k4

    # `alpha` sets the DEPTH as well as the count, on the same terms as the sibling.
    @test RadialTailDecay(; alpha = 0.01)(:kappa, PRMVT4, nothing, nothing) != k4
end

@testset "Calibration rules: `RadialTailDecay` reads the covariance matrix" begin
    rule = RadialTailDecay(; kmin = 5, alpha = 0.05)

    # The line that separates this rule from `HillTailDecay`: the shape of the covariance
    # matrix reaches the answer. The sibling standardises each column on its own, so a
    # covariance matrix with the same diagonal and no off-diagonal term leaves it unmoved.
    diagonal = LowOrderPrior(; X = PRMVT4.X, mu = PRMVT4.mu,
                             sigma = Matrix(Diagonal(diag(PRMVT4.sigma))))
    @test rule(:kappa, diagonal, nothing, nothing) != rule(:kappa, PRMVT4, nothing, nothing)
    @test HillTailDecay(; kmin = 5, alpha = 0.05)(:kappa, diagonal, nothing, nothing) ==
          HillTailDecay(; kmin = 5, alpha = 0.05)(:kappa, PRMVT4, nothing, nothing)

    # `chol` and `sigma` state the same factorisation, so they give the same number.
    both = LowOrderPrior(; X = PRMVT4.X, mu = PRMVT4.mu, sigma = PRMVT4.sigma,
                         chol = cholesky(PRMVT4.sigma).U)
    @test rule(:kappa, both, nothing, nothing) == rule(:kappa, PRMVT4, nothing, nothing)

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
    @test rule(:kappa, precedence, nothing, nothing) ==
          rule(:kappa, stated, nothing, nothing)
    @test rule(:kappa, precedence, nothing, nothing) !=
          rule(:kappa, identity_sigma, nothing, nothing)

    # A TALL `chol` states a covariance matrix all the same, and the `R` factor of its QR
    # factorisation is the square factor of that same matrix. `chol` is checked against the
    # length of `mu` alone, so a carrier can hold one.
    F = randn(StableRNG(4242), 20, 8)
    tall = LowOrderPrior(; X = PRMVT4.X, mu = PRMVT4.mu, sigma = F' * F, chol = F)
    square = LowOrderPrior(; X = PRMVT4.X, mu = PRMVT4.mu, sigma = F' * F)
    @test rule(:kappa, tall, nothing, nothing) ≈ rule(:kappa, square, nothing, nothing)

    # The verb that picks the factor is separate, and it returns the prior's own factor
    # untouched when there is one.
    @test PO.whitening_factor(both) === both.chol
    @test PO.whitening_factor(PRMVT4) == cholesky(PRMVT4.sigma).U
    @test size(PO.whitening_factor(tall)) == (8, 8)
end

@testset "Calibration rules: `RadialTailDecay` answers one number for both ends" begin
    rule = RadialTailDecay(; kmin = 5, alpha = 0.01)

    # A distance has no sign, so every key gives the same number. The sibling gives two on
    # the same sample, which is the whole difference between what the two rules say.
    ka = rule(:kappa_a, PRSKEW, nothing, nothing)
    @test rule(:kappa, PRSKEW, nothing, nothing) == ka
    @test rule(:kappa_b, PRSKEW, nothing, nothing) == ka
    @test rule(:anything_at_all, PRSKEW, nothing, nothing) == ka
    hill = HillTailDecay(; kmin = 5, alpha = 0.01)
    @test hill(:kappa_a, PRSKEW, nothing, nothing) !=
          hill(:kappa_b, PRSKEW, nothing, nothing)

    # `mirror_role` is therefore trivially correct for this rule: the head role it builds
    # holds the same rule, and the same rule answers the head key with the tail's number.
    tail = DeformationTailCalibration(; alg = rule)
    head = PO.mirror_role(tail)
    @test isa(head, DeformationHeadCalibration)
    @test head.alg === rule
    @test PO.resolve_calibration_slot(head, :kappa_b, PRSKEW, nothing) ==
          PO.resolve_calibration_slot(tail, :kappa_a, PRSKEW, nothing)

    # Through a Range measure the two ends read their OWN probabilities, so the two numbers
    # part when the two probabilities differ. The count `k` moves, not the end.
    rg = RelativisticValueatRiskRange(; alpha = 0.01,
                                      kappa_a = DeformationTailCalibration(;
                                                                           alg = RadialTailDecay(;
                                                                                                 kmin = 5)),
                                      beta = 0.02,
                                      kappa_b = DeformationHeadCalibration(;
                                                                           alg = RadialTailDecay(;
                                                                                                 kmin = 5)))
    og = PO.resolve_deferred_quantities(rg, PRSKEW)
    @test og.kappa_a == ka
    @test og.kappa_b ==
          RadialTailDecay(; kmin = 5, alpha = 0.02)(:kappa_b, PRSKEW, nothing, nothing)
    @test og.kappa_a != og.kappa_b
    @test 0 < og.kappa_a < 1
    @test 0 < og.kappa_b < 1

    # The two ends of one probability DO agree, which is the statement the rule makes.
    eq = RelativisticValueatRiskRange(; alpha = 0.01,
                                      kappa_a = DeformationTailCalibration(;
                                                                           alg = RadialTailDecay(;
                                                                                                 kmin = 5)),
                                      beta = 0.01,
                                      kappa_b = DeformationHeadCalibration(;
                                                                           alg = RadialTailDecay(;
                                                                                                 kmin = 5)))
    oeq = PO.resolve_deferred_quantities(eq, PRSKEW)
    @test oeq.kappa_a == oeq.kappa_b

    # `bind_alpha` reaches the rule through both roles of the family, and through the rule
    # itself.
    @test PO.bind_alpha(RadialTailDecay(; kmin = 7), 0.03) ==
          RadialTailDecay(; kmin = 7, alpha = 0.03)
    @test PO.bind_alpha(DeformationTailCalibration(; alg = RadialTailDecay(; kmin = 7)),
                        0.03).alg.alpha == 0.03
    @test PO.bind_alpha(DeformationHeadCalibration(; alg = RadialTailDecay(; kmin = 7)),
                        0.03).alg.kmin == 7
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
    @test isnothing(unbound.alpha)
    @test unbound.kmin == 30
    @test_throws PO.IsNothingError unbound(:kappa, PR60, nothing, nothing)
    msg = try
        unbound(:kappa, PR60, nothing, nothing)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("RadialTailDecay.alpha", msg)
    @test occursin("bind_alpha", msg)

    # The floor. The series holds ONE entry per observation, so `PR60` leaves three order
    # statistics at `alpha = 0.05` where the pool of the sibling leaves twelve. The same
    # floor therefore binds harder here, and the message says which count it refused.
    @test_throws DomainError RadialTailDecay(; alpha = 0.05)(:kappa, PR60, nothing, nothing)
    msg = try
        RadialTailDecay(; alpha = 0.05)(:kappa, PR60, nothing, nothing)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("kmin", msg)
    @test occursin("floor at 30", msg)
    @test occursin("= 3` of the 60", msg)

    # A floor the sample clears returns a number, so the refusal is the floor's and not the
    # sample's.
    @test 0 < RadialTailDecay(; kmin = 3, alpha = 0.05)(:kappa, PR60, nothing, nothing) < 1

    # A probability that takes the whole sample leaves no `k + 1`-th distance to divide by.
    @test_throws DomainError RadialTailDecay(; kmin = 5, alpha = 0.999)(:kappa, PR60,
                                                                        nothing, nothing)
    msg = try
        RadialTailDecay(; kmin = 5, alpha = 0.999)(:kappa, PR60, nothing, nothing)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("needs 61 radial distances", msg)
    @test occursin("which is 60", msg)

    # The series holds fewer positive entries than the count asks for, so there is no Hill
    # estimate at all. A sample that sits exactly at its own mean produces it.
    @test PRZERO.mu == zeros(4)
    @test_throws DomainError RadialTailDecay(; kmin = 5, alpha = 0.6)(:kappa, PRZERO,
                                                                      nothing, nothing)
    msg = try
        RadialTailDecay(; kmin = 5, alpha = 0.6)(:kappa, PRZERO, nothing, nothing)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("121-th largest radial distance is 0.0", msg)
    @test occursin("not positive", msg)

    # An estimate at or below one is a tail with no finite mean, and no admissible κ reads
    # it. The band `(0, 1)` the slot admits IS the condition `â > 1`, so the refusal is the
    # reading rather than a guard bolted onto it.
    @test_throws DomainError RadialTailDecay(; kmin = 5, alpha = 0.05)(:kappa, PRMVHEAVY,
                                                                       nothing, nothing)
    msg = try
        RadialTailDecay(; kmin = 5, alpha = 0.05)(:kappa, PRMVHEAVY, nothing, nothing)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("tail index of 0.8", msg)
    @test occursin("(0, 1)", msg)

    # A covariance matrix that is not positive definite states no whitening, and the message
    # names the field it was read off.
    @test_throws DomainError RadialTailDecay(; kmin = 5, alpha = 0.05)(:kappa, PRSING,
                                                                       nothing, nothing)
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
    @test_throws DimensionMismatch RadialTailDecay(; kmin = 5, alpha = 0.05)(:kappa, PRWIDE,
                                                                             nothing,
                                                                             nothing)
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
    @test_throws DomainError RadialTailDecay(; kmin = 5, alpha = 0.05)(:kappa, PRZDIAG,
                                                                       nothing, nothing)
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
column. Nothing else in either reading moves. `bind_series` carries the marker from the owner
to the rule, and the owner's marker wins, because the quantity belongs to the measure and a
rule cannot know which measure it reached.

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

@testset "Calibration series: `bind_series` carries the owner's series" begin
    rule = HillTailDecay(; kmin = 12, alpha = 0.02)

    # The verb reaches the rule, and through both roles of the family.
    @test isa(PO.bind_series(rule, AbsoluteDrawdownSeries()).series, AbsoluteDrawdownSeries)
    @test isa(PO.bind_series(DeformationTailCalibration(; alg = rule),
                             RelativeDrawdownSeries()).alg.series, RelativeDrawdownSeries)
    @test isa(PO.bind_series(DeformationHeadCalibration(; alg = rule),
                             AbsoluteDrawdownSeries()).alg.series, AbsoluteDrawdownSeries)
    @test isa(PO.bind_series(RadialTailDecay(; kmin = 7), AbsoluteDrawdownSeries()).series,
              AbsoluteDrawdownSeries)

    # It carries the rule's other fields over untouched, and it commutes with `bind_alpha`:
    # the two verbs fill two different fields.
    bound = PO.bind_series(rule, AbsoluteDrawdownSeries())
    @test bound.kmin == 12
    @test bound.alpha == 0.02
    @test PO.bind_alpha(PO.bind_series(HillTailDecay(; kmin = 12),
                                       AbsoluteDrawdownSeries()), 0.02) == bound
    @test PO.bind_series(PO.bind_alpha(HillTailDecay(; kmin = 12), 0.02),
                         AbsoluteDrawdownSeries()) == bound

    # THE OWNER'S SERIES WINS. A rule that already carries a marker has it replaced, because
    # the quantity belongs to the measure. This is the reading `bind_norm_order` states for
    # a norm order, and it is the opposite of `bind_alpha`, whose number no rule holds.
    stated = HillTailDecay(; kmin = 12, alpha = 0.02, series = RelativeDrawdownSeries())
    @test isa(PO.bind_series(stated, ReturnsSeries()).series, ReturnsSeries)

    # The default is the identity, so everything that reads no series crosses unchanged.
    @test PO.bind_series(0.3, AbsoluteDrawdownSeries()) === 0.3
    @test PO.bind_series(nothing, AbsoluteDrawdownSeries()) === nothing
    budget = EntropyBudget(; target = -1.3, alpha = 0.05)
    @test PO.bind_series(budget, AbsoluteDrawdownSeries()) === budget
    probe = (key, pr, w, slv) -> 0.4
    @test PO.bind_series(probe, AbsoluteDrawdownSeries()) === probe
    @test PO.bind_series(DeformationTailCalibration(; alg = probe),
                         AbsoluteDrawdownSeries()).alg === probe
    @test PO.bind_series(SignificanceTailCalibration(; alg = RateSignificance()),
                         AbsoluteDrawdownSeries()).alg == RateSignificance()
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
        rule = HillTailDecay(; kmin = 20, alpha = alpha, series = s)
        @test rule(:kappa, PRDD, nothing, nothing) ≈ inv(hill_by_hand(f, XDD, k))
    end

    # The three markers name three series of one sample, so they answer three numbers.
    ka = HillTailDecay(; kmin = 20, alpha = alpha, series = ReturnsSeries())(:kappa, PRDD,
                                                                             nothing,
                                                                             nothing)
    kb = HillTailDecay(; kmin = 20, alpha = alpha, series = AbsoluteDrawdownSeries())(:kappa,
                                                                                      PRDD,
                                                                                      nothing,
                                                                                      nothing)
    kc = HillTailDecay(; kmin = 20, alpha = alpha, series = RelativeDrawdownSeries())(:kappa,
                                                                                      PRDD,
                                                                                      nothing,
                                                                                      nothing)
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
    @test_throws ArgumentError HillTailDecay(; kmin = 20, alpha = alpha,
                                             series = AbsoluteDrawdownSeries())(:kappa_b,
                                                                                PRDD,
                                                                                nothing,
                                                                                nothing)
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
    rule = RadialTailDecay(; kmin = 20, alpha = alpha, series = AbsoluteDrawdownSeries())

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
    @test rule(:kappa, PRDD, nothing, nothing) ≈ inv(ahat)
    @test 0 < rule(:kappa, PRDD, nothing, nothing) < 1

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
    @test rule(:kappa, scaled, nothing, nothing) == rule(:kappa, PRDD, nothing, nothing)

    # The returns reading DOES read `pr.sigma`, but only through the SHAPE of the whitening.
    # A hundredfold scaling of the covariance matrix scales every radial distance alike, and
    # a Hill estimate reads the ratios `u[i] / u[k + 1]` alone, so the scale cancels and the
    # returns reading holds to rounding. Do not tighten this to an equality: the two sides
    # take different roundings through the Cholesky factorisation, and CI has returned both
    # the same last bit and a different one.
    ret = RadialTailDecay(; kmin = 20, alpha = alpha)
    @test ret(:kappa, scaled, nothing, nothing) ≈ ret(:kappa, PRDD, nothing, nothing)

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
    @test ret(:kappa, stretched, nothing, nothing) != ret(:kappa, PRDD, nothing, nothing)
    @test rule(:kappa, stretched, nothing, nothing) == rule(:kappa, PRDD, nothing, nothing)

    # The distance still has no sign, so the rule answers one number for every key. The
    # marker moves the sample the distance is taken over, and not what a distance is.
    @test rule(:kappa_a, PRDD, nothing, nothing) == rule(:kappa, PRDD, nothing, nothing)
    @test rule(:kappa_b, PRDD, nothing, nothing) == rule(:kappa, PRDD, nothing, nothing)

    # A drawdown sample states its own covariance matrix, so a sample that states a singular
    # one is refused there rather than in `whitening_factor`. Two columns with one path
    # between them are what produce it.
    twice = [XDD[:, 1] XDD[:, 1]]
    prtwice = LowOrderPrior(; X = twice, mu = vec(mean(twice; dims = 1)),
                            sigma = Matrix(1.0I, 2, 2))
    @test_throws DomainError rule(:kappa, prtwice, nothing, nothing)
    msg = try
        PO.radial_series_inputs(AbsoluteDrawdownSeries(), prtwice)
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("AbsoluteDrawdownSeries", msg)
    @test occursin("not positive definite", msg)
    # The same carrier passes the returns reading, because `pr.sigma` states a whitening.
    @test 0 < ret(:kappa, prtwice, nothing, nothing) < 1
end

@testset "Calibration series: a measure hands its own series over" begin
    # The four owners resolve one key, `:kappa`, and three of them price three series. The
    # marker is the only thing that separates the three readings.
    hill = DeformationTailCalibration(; alg = HillTailDecay(; kmin = 20))
    rlvar = PO.resolve_deferred_quantities(RelativisticValueatRisk(; alpha = 0.05,
                                                                   kappa = hill), PRDD)
    rldar = PO.resolve_deferred_quantities(RelativisticDrawdownatRisk(; alpha = 0.05,
                                                                      kappa = hill), PRDD)
    rrldar = PO.resolve_deferred_quantities(RelativeRelativisticDrawdownatRisk(;
                                                                               alpha = 0.05,
                                                                               kappa = hill),
                                            PRDD)
    @test rlvar.kappa ==
          HillTailDecay(; kmin = 20, alpha = 0.05, series = ReturnsSeries())(:kappa, PRDD,
                                                                             nothing,
                                                                             nothing)
    @test rldar.kappa ==
          HillTailDecay(; kmin = 20, alpha = 0.05, series = AbsoluteDrawdownSeries())(:kappa,
                                                                                      PRDD,
                                                                                      nothing,
                                                                                      nothing)
    @test rrldar.kappa ==
          HillTailDecay(; kmin = 20, alpha = 0.05, series = RelativeDrawdownSeries())(:kappa,
                                                                                      PRDD,
                                                                                      nothing,
                                                                                      nothing)
    @test rlvar.kappa != rldar.kappa
    @test rldar.kappa != rrldar.kappa

    # The radial rule travels the same way, and its drawdown reading is its own.
    radial = DeformationTailCalibration(; alg = RadialTailDecay(; kmin = 20))
    rdar = PO.resolve_deferred_quantities(RelativisticDrawdownatRisk(; alpha = 0.05,
                                                                     kappa = radial), PRDD)
    @test rdar.kappa ==
          RadialTailDecay(; kmin = 20, alpha = 0.05, series = AbsoluteDrawdownSeries())(:kappa,
                                                                                        PRDD,
                                                                                        nothing,
                                                                                        nothing)
    @test rdar.kappa != rldar.kappa

    # A MARKER STATED ON THE RULE IS OVERWRITTEN by the measure that resolves it. It serves
    # a caller who runs the rule by hand, and nothing else.
    wrong = DeformationTailCalibration(;
                                       alg = HillTailDecay(; kmin = 20,
                                                           series = AbsoluteDrawdownSeries()))
    o = PO.resolve_deferred_quantities(RelativisticValueatRisk(; alpha = 0.05,
                                                               kappa = wrong), PRDD)
    @test o.kappa == rlvar.kappa

    # Both ends of a Range measure price ONE series, where each end prices its own
    # probability. So the marker reaching the two `kappa` slots is the same marker.
    rg = PO.resolve_deferred_quantities(RelativisticValueatRiskRange(; alpha = 0.05,
                                                                     kappa_a = hill,
                                                                     beta = 0.05,
                                                                     kappa_b = PO.mirror_role(hill)),
                                        PRDD)
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
    @test isapprox(HillTailDecay(; kmin = 30, alpha = 0.15)(:kappa, PR629, nothing,
                                                            nothing),
                   inv(hill_reference(hill_pool(X629, skappa), 600)))
    @test isapprox(RadialTailDecay(; kmin = 30, alpha = 0.15)(:kappa, PR629, nothing,
                                                              nothing),
                   inv(hill_reference(d629, 150)))
end
