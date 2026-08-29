#=
An **Ambiguity Radius** is the radius of the ball of probability measures a distributionally
robust model prices, and an **Esfahani-Kuhn tail weight** is the weight of the tail term of
the loss it minimises. Twelve slots across six types hold one or the other, and each of them
takes a **Calibration Rule** in place of the number.

`test_09f_calibration_slot.jl` covers the mechanism and `test_09g_calibration_rules.jl`
covers the rules of the two older families. This file covers the ambiguity families, the
four radius rules and the one tail-weight rule they ship, and the twelve slots.

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

# `TailTermParity` reads a mean against a tail, so it needs a sample that carries a real
# expected return. `XDAY` is ten years of daily returns, and `XMON` sums its rows in blocks
# of 21. A mean scales with the horizon and a dispersion scales with its square root, so
# the two samples carry two DIFFERENT mean-to-CVaR ratios. That is the defect the rule
# fixes: one stated `l` is two trade-offs across the pair.
const XDAY = 0.0005 .+ 0.01 * randn(RNG, 2520, 4)
const PRDAY = prior(EmpiricalPrior(), XDAY)
const XMON = reduce(vcat,
                    [sum(view(XDAY, ((k - 1) * 21 + 1):(k * 21), :); dims = 1)
                     for k in 1:120])
const PRMON = prior(EmpiricalPrior(), XMON)
const WDAY = pweights(range(; start = 1, stop = 2, length = 2520))

# A sample of strictly positive returns has no loss in the worst `alpha` of any column, so
# its tail-term scale is not positive. It is one of the rule's two refusals.
const XPOS = 0.01 .+ 0.001 * abs.(randn(RNG, 60, 3))
const PRPOS = prior(EmpiricalPrior(), XPOS)

# Every column of `XZERO` sums to EXACTLY zero, so its pooled mean loss is exactly zero and
# the sample states no exchange rate between the two terms of the loss. The three columns
# carry three different patterns, so the covariance matrix is not singular.
const XZERO = hcat(repeat([0.02, -0.02], 30), repeat([0.03, 0.01, -0.03, -0.01], 15),
                   repeat([-0.05, 0.05], 30))
const PRZERO = prior(EmpiricalPrior(), XZERO)

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
probe_radius(::Symbol, pr::PO.AbstractPriorResult, ::Any, ::Any) = 3 / size(pr.X, 1)

# The tail-weight family ships `TailTermParity`, and a caller's own function stands beside
# it. This one reports the key, so the resolver's argument order is asserted.
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
         :RateRadius, :DimensionalRateRadius, :DualNormRadius, :TailTermParity)
        @test sym ∈ names(PortfolioOptimisers)
    end

    # Four rules compute a radius, and one computes an Esfahani-Kuhn tail weight. Neither
    # family holds a member of the other, and the abstract types stay unexported.
    @test ConcentrationRadius <: PO.AbstractAmbiguityRadiusCalibrationAlgorithm
    @test RateRadius <: PO.AbstractAmbiguityRadiusCalibrationAlgorithm
    @test DimensionalRateRadius <: PO.AbstractAmbiguityRadiusCalibrationAlgorithm
    @test DualNormRadius <: PO.AbstractAmbiguityRadiusCalibrationAlgorithm
    @test TailTermParity <: PO.AbstractAmbiguityTailWeightCalibrationAlgorithm
    @test filter(t -> t !== AmbiguityTailWeightCalibration,
                 subtypes(PO.AbstractAmbiguityTailWeightCalibrationAlgorithm)) ==
          [TailTermParity]
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
    @test alg(:r, PR60, nothing, nothing) == alg(:l2reg_val, PR60, nothing, nothing)

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

@testset "TailTermParity: the tail term at a stated multiple of the mean term" begin
    alg = TailTermParity(; ratio = 1, alpha = 0.05)
    l = alg(:l, PRDAY, nothing, nothing)

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
    @test TailTermParity(; ratio = 2, alpha = 0.05)(:l, PRDAY, nothing, nothing) ≈ 2 * l
    @test TailTermParity(; ratio = 0.5, alpha = 0.05)(:l, PRDAY, nothing, nothing) ≈ l / 2

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
    @test alg(:l_a, PRDAY, nothing, nothing) == l
    @test alg(:l_b, PRDAY, nothing, nothing) == l

    # `alpha` fixes the depth of the tail, so a wider probability reads a smaller scale
    # and returns a larger weight.
    @test TailTermParity(; ratio = 1, alpha = 0.2)(:l, PRDAY, nothing, nothing) > l

    # The rule needs no solver, so it ignores the one the resolution threads.
    @test alg(:l, PRDAY, nothing, Solver(; solver = nothing)) == l

    # Both scales are sample statistics rather than counts, so stated weights move them.
    # That is where the rule parts company with `RateRadius`, which reads a raw row count.
    mw = -sum(j -> dot(view(XDAY, :, j), WDAY), axes(XDAY, 2)) / (4 * sum(WDAY))
    cw = mean(j -> ConditionalValueatRisk(; alpha = 0.05, w = WDAY)(view(XDAY, :, j)),
              axes(XDAY, 2))
    @test alg(:l, PRDAY, WDAY, nothing) ≈ abs(mw) / cw
    @test alg(:l, PRDAY, WDAY, nothing) != l

    # Construction validation, on the terms `RateRadius` writes.
    @test TailTermParity().ratio == 1
    @test isnothing(TailTermParity().alpha)
    @test_throws DomainError TailTermParity(; ratio = 0.0)
    @test_throws DomainError TailTermParity(; ratio = -1.0)
    @test_throws DomainError TailTermParity(; ratio = Inf)

    # The rule joins the tail-weight family and its bounds, and no other.
    @test TailTermParity <: PO.AbstractAmbiguityTailWeightCalibrationAlgorithm
    @test isa(alg, PO.Func_AmbTwtCal)
    @test !isa(alg, PO.Func_AmbRadCal)
    @test isa(AmbiguityTailWeightCalibration(; alg = alg), PO.Num_AmbTwtCal)
    @test_throws TypeError AmbiguityRadiusCalibration(; alg = alg)
end

@testset "TailTermParity: one stated tail weight is two trade-offs" begin
    alg = TailTermParity(; ratio = 1, alpha = 0.05)
    l_day = alg(:l, PRDAY, nothing, nothing)
    l_mon = alg(:l, PRMON, nothing, nothing)

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
    role = AmbiguityTailWeightCalibration(; alg = alg)
    @test PO.resolve_calibration_slot(role, :l, PRDAY, nothing) !=
          PO.resolve_calibration_slot(role, :l, PRMON, nothing)
end

@testset "TailTermParity: `alpha` and `l` travel together" begin
    alg = TailTermParity(; ratio = 1)
    role = AmbiguityTailWeightCalibration(; alg = alg)

    # The rule cannot run until the probability reaches it, and the message names the verb
    # that carries it.
    err = try
        alg(:l, PRDAY, nothing, nothing)
        nothing
    catch e
        e
    end
    @test isa(err, PO.IsNothingError)
    @test occursin("bind_alpha", err.msg)

    # `bind_alpha` fills the field, and it takes the ROLE as well as the rule. A stated
    # number and a plain function pass through untouched, so no existing behaviour moved.
    @test PO.bind_alpha(alg, 0.05).alpha == 0.05
    @test PO.bind_alpha(TailTermParity(; ratio = 3), 0.05).ratio == 3
    @test PO.bind_alpha(role, 0.05).alg.alpha == 0.05
    @test PO.bind_alpha(1.0, 0.05) === 1.0
    @test PO.bind_alpha(probe_tail_weight, 0.05) === probe_tail_weight

    # The scalar measure resolves `alpha` first, so the slot reads the measure's own
    # probability and two measures give two weights.
    m1 = DistributionallyRobustConditionalValueatRisk(; alpha = 0.05, l = role)
    m2 = DistributionallyRobustConditionalValueatRisk(; alpha = 0.2, l = role)
    @test PO.resolve_deferred_quantities(m1, PRDAY).l ≈
          TailTermParity(; ratio = 1, alpha = 0.05)(:l, PRDAY, nothing, nothing)
    @test PO.resolve_deferred_quantities(m2, PRDAY).l ≈
          TailTermParity(; ratio = 1, alpha = 0.2)(:l, PRDAY, nothing, nothing)
    @test PO.resolve_deferred_quantities(m1, PRDAY).l !=
          PO.resolve_deferred_quantities(m2, PRDAY).l

    # Each end of a Range measure reads its OWN probability, which is the case the six-slot
    # resolution has to get right.
    rg = DistributionallyRobustConditionalValueatRiskRange(; alpha = 0.05, beta = 0.2,
                                                           l_a = role, l_b = role)
    out = PO.resolve_deferred_quantities(rg, PRDAY)
    @test out.l_a ≈ TailTermParity(; ratio = 1, alpha = 0.05)(:l_a, PRDAY, nothing, nothing)
    @test out.l_b ≈ TailTermParity(; ratio = 1, alpha = 0.2)(:l_b, PRDAY, nothing, nothing)
    @test out.l_a != out.l_b
    @test out.alpha == 0.05
    @test out.beta == 0.2

    # A rule in the `alpha` slot beside a rule in the `l` slot resolves in the right order:
    # the tail weight reads the RESOLVED probability and not the slot's occupant.
    srole = SignificanceTailCalibration(; alg = RateSignificance(; c = 0.5))
    both = DistributionallyRobustConditionalValueatRisk(; alpha = srole, l = role)
    bout = PO.resolve_deferred_quantities(both, PRDAY)
    @test bout.alpha ≈ 0.5 / sqrt(2520)
    @test bout.l ≈
          TailTermParity(; ratio = 1, alpha = bout.alpha)(:l, PRDAY, nothing, nothing)

    # The drawdown measure carries the same pair, and it hands its own SERIES over beside
    # the probability. #623 corrected this: the number was the asset-column one, and the
    # tail term this measure prices is a CDaR of the drawdown series.
    dd = DistributionallyRobustConditionalDrawdownatRisk(; alpha = 0.05, l = role)
    ddl = PO.resolve_deferred_quantities(dd, PRDAY).l
    @test ddl ≈ PO.bind_series(TailTermParity(; ratio = 1, alpha = 0.05),
                               AbsoluteDrawdownSeries())(:l, PRDAY, nothing, nothing)
    @test ddl != TailTermParity(; ratio = 1, alpha = 0.05)(:l, PRDAY, nothing, nothing)

    # A measure whose slots all hold numbers is still returned unchanged, so the binding
    # costs nothing where no rule reads a sibling.
    plain = DistributionallyRobustConditionalValueatRisk()
    @test PO.resolve_deferred_quantities(plain, PRDAY) === plain
    plain_rg = DistributionallyRobustConditionalValueatRiskRange()
    @test PO.resolve_deferred_quantities(plain_rg, PRDAY) === plain_rg
end

@testset "TailTermParity: the two refusals" begin
    alg = TailTermParity(; ratio = 1, alpha = 0.05)

    # A sample of strictly positive returns holds no loss in the worst `alpha` of any
    # column, so it has no tail-term scale and no weight prices one term against the other.
    err = try
        alg(:l, PRPOS, nothing, nothing)
        nothing
    catch e
        e
    end
    @test isa(err, DomainError)
    @test occursin("tail-term scale", err.msg)

    # A sample whose pooled mean loss is zero states no exchange rate between the two
    # terms: every weight prices a mean term of zero alike.
    @test mean(XZERO) == 0
    err = try
        alg(:l, PRZERO, nothing, nothing)
        nothing
    catch e
        e
    end
    @test isa(err, DomainError)
    @test occursin("mean-term scale", err.msg)
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

#=
`DualNormRadius` is the first rule of the family for which `key` carries meaning. The eight
radius slots do not measure distance in one norm, so a rule that returns one number for
every key gives one coefficient to two penalties whose natural scales differ. This rule
reads the key, picks the ground metric of that slot, and returns the sampling error of the
empirical measure in it.

Issue #614 ships it, and it settles the `:val` collision by widening the two regularisation
keys to `:l2reg_val` and `:lpreg_val`. The two symbols are the two names `field_dict`
already uses for the two slots, so a caller's own function placed in an
`AmbiguityRadiusCalibration` now receives one of those in place of `:val`.

The `:lpreg_val` key still names no norm order, because `p` lives on the owner. The order
travels to the rule through `bind_norm_order`, which #615's sibling ticket already built
for the norm-ceiling family, so the caller states the order once and on the penalty.

The prior below carries a DIAGONAL covariance matrix, so every norm of the per-asset error
vector has a closed form and no test reads a number off the implementation.
=#
const DNR_SD = [0.2, 0.3, 0.4, 0.5]
const PRDIAG = LowOrderPrior(; X = X60, mu = zeros(4),
                             sigma = Matrix(Diagonal(DNR_SD .^ 2)))
const DNR_Z = Distributions.quantile(Distributions.Normal(), 0.95)
const DNR_KISH = sum(WTS)^2 / sum(abs2, WTS)

@testset "DualNormRadius: the ground metric of the slot selects the norm" begin
    # The rule joins the radius family, and it is exported beside the two that ship.
    @test DualNormRadius <: PO.AbstractAmbiguityRadiusCalibrationAlgorithm
    @test !(DualNormRadius <: PO.AbstractAmbiguityTailWeightCalibrationAlgorithm)
    @test :DualNormRadius ∈ names(PortfolioOptimisers)
    @test isa(DualNormRadius(), PO.Func_AmbRadCal)
    @test !isa(DualNormRadius(), PO.Func_AmbTwtCal)
    @test isa(AmbiguityRadiusCalibration(; alg = DualNormRadius()), PO.Num_AmbRadCal)

    # The defaults: a per-coordinate 95% level, and no norm order, which serves every slot
    # but `:lpreg_val`.
    @test DualNormRadius().confidence == 0.95
    @test isnothing(DualNormRadius().p)
    @test DualNormRadius(; p = 3).p == 3

    alg = DualNormRadius()
    e = DNR_SD / sqrt(60)

    # --- the four closed forms, on a diagonal covariance matrix -------------------------
    # The `l1` coefficient multiplies the 1-norm of the weights, so its ground metric is
    # the ∞-norm and the radius is the largest per-asset error.
    @test alg(:l1, PRDIAG, nothing, nothing) ≈ DNR_Z * maximum(e)

    # The `linf` coefficient and the three DR radii all multiply the ∞-norm, so their
    # ground metric is the 1-norm and the radius is the sum of the per-asset errors.
    @test alg(:linf, PRDIAG, nothing, nothing) ≈ DNR_Z * sum(e)
    @test alg(:r, PRDIAG, nothing, nothing) ≈ DNR_Z * sum(e)

    # The L2 penalty is its own dual, so the radius is the 2-norm.
    @test alg(:l2reg_val, PRDIAG, nothing, nothing) ≈ DNR_Z * norm(e, 2)

    # The Lp penalty's ground metric is the type-`q` metric with `1/p + 1/q = 1`. The
    # penalty site fills `p` through `bind_norm_order`, and a stated `p` runs the rule on
    # its own, outside that site.
    @test DualNormRadius(; p = 3)(:lpreg_val, PRDIAG, nothing, nothing) ≈
          DNR_Z * norm(e, 1.5)
    @test DualNormRadius(; p = 1.25)(:lpreg_val, PRDIAG, nothing, nothing) ≈
          DNR_Z * norm(e, 5)

    # --- the defect the rule fixes ------------------------------------------------------
    # One rule, two slots, two numbers. That is the whole point: `ConcentrationRadius` and
    # `RateRadius` cannot tell the two apart, and this rule can.
    @test alg(:l1, PRDIAG, nothing, nothing) != alg(:linf, PRDIAG, nothing, nothing)
    @test alg(:linf, PRDIAG, nothing, nothing) > alg(:l1, PRDIAG, nothing, nothing)
    @test alg(:l2reg_val, PRDIAG, nothing, nothing) !=
          alg(:l1, PRDIAG, nothing, nothing) !=
          alg(:linf, PRDIAG, nothing, nothing)

    # A radius names no end of the distribution, so the two ends of a Range twin resolve to
    # one number and both agree with the scalar measure's own slot.
    @test alg(:r_a, PRDIAG, nothing, nothing) == alg(:r_b, PRDIAG, nothing, nothing)
    @test alg(:r_a, PRDIAG, nothing, nothing) == alg(:r, PRDIAG, nothing, nothing)

    # --- the confidence level -----------------------------------------------------------
    # A higher level buys a larger ball, and the level scales the whole vector, so the
    # ratio of two levels is the ratio of the two quantiles in every ground metric.
    hi = DualNormRadius(; confidence = 0.99)
    z99 = Distributions.quantile(Distributions.Normal(), 0.99)
    @test hi(:l1, PRDIAG, nothing, nothing) > alg(:l1, PRDIAG, nothing, nothing)
    @test hi(:linf, PRDIAG, nothing, nothing) / alg(:linf, PRDIAG, nothing, nothing) ≈
          z99 / DNR_Z

    # --- the sample size ----------------------------------------------------------------
    # With no weights the rule reads the raw row count, and stated weights move it to
    # Kish's effective sample size, which is smaller for any weights that are not equal.
    @test DNR_KISH < 60
    @test alg(:l1, PRDIAG, WTS, nothing) ≈ DNR_Z * maximum(DNR_SD) / sqrt(DNR_KISH)
    @test alg(:l1, PRDIAG, WTS, nothing) > alg(:l1, PRDIAG, nothing, nothing)
    @test alg(:linf, PRDIAG, WTS, nothing) ≈ DNR_Z * sum(DNR_SD) / sqrt(DNR_KISH)
    @test alg(:linf, PRDIAG, WTS, nothing) / alg(:linf, PRDIAG, nothing, nothing) ≈
          sqrt(60 / DNR_KISH)

    # The rule needs no solver, so it ignores the one the resolution threads.
    @test alg(:l1, PRDIAG, nothing, Solver(; solver = nothing)) ==
          alg(:l1, PRDIAG, nothing, nothing)

    # --- the two refusals ---------------------------------------------------------------
    # An unrecognised key is the one refusal the rule owes: a caller who writes their own
    # measure hits it first, so the message names the key it got and the keys it serves.
    err = try
        alg(:alpha, PRDIAG, nothing, nothing)
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
        alg(:lpreg_val, PRDIAG, nothing, nothing)
        nothing
    catch e
        e
    end
    @test isa(perr, ArgumentError)
    @test occursin("lpreg_val", perr.msg)
    @test occursin("DualNormRadius.p", perr.msg)
    @test occursin("bind_norm_order", perr.msg)

    # The scale function is the whole of the key's meaning, and it is reachable on its own.
    @test PO.dual_norm_radius_scale(alg, :l1, DNR_SD) == maximum(DNR_SD)
    @test PO.dual_norm_radius_scale(alg, :linf, DNR_SD) == sum(DNR_SD)
    @test_throws ArgumentError PO.dual_norm_radius_scale(alg, :beta, DNR_SD)

    # --- construction validation --------------------------------------------------------
    # The confidence level is a probability, and the norm order meets the check its owner
    # makes: finite and greater than one.
    @test_throws DomainError DualNormRadius(; confidence = 1.0)
    @test_throws DomainError DualNormRadius(; confidence = 0.0)
    @test_throws DomainError DualNormRadius(; p = 1.0)
    @test_throws DomainError DualNormRadius(; p = 0.5)
    @test_throws PO.IsNonFiniteError DualNormRadius(; p = Inf)
end

@testset "DualNormRadius: the three distributionally robust radius slots" begin
    role = AmbiguityRadiusCalibration(; alg = DualNormRadius())
    # The measures resolve against the empirical prior, so the number is read from the same
    # rule rather than written out twice.
    r_num = DualNormRadius()(:r, PR60, nothing, nothing)
    @test r_num > 0

    m = DistributionallyRobustConditionalValueatRisk(; r = role)
    @test PO.resolve_deferred_quantities(m, PR60).r ≈ r_num
    @test PO.factory(m, PR60).r ≈ r_num

    # The drawdown owner hands its own series over, so the error scale comes off the
    # drawdown sample. #623 corrected this: the number was the asset-return one.
    dd = DistributionallyRobustConditionalDrawdownatRisk(; r = role)
    ddr = PO.resolve_deferred_quantities(dd, PR60).r
    @test ddr ≈
          PO.bind_series(DualNormRadius(), AbsoluteDrawdownSeries())(:r, PR60, nothing,
                                                                     nothing)
    @test ddr != r_num

    # Both ends of the Range twin carry the same ground metric, so one rule on both ends
    # gives one number. The two ends of a radius are not two tails.
    rg = DistributionallyRobustConditionalValueatRiskRange(; r_a = role, r_b = role)
    rgo = PO.resolve_deferred_quantities(rg, PR60)
    @test rgo.r_a ≈ r_num
    @test rgo.r_b ≈ r_num
    @test rgo.r_a == rgo.r_b
end

@testset "DualNormRadius: the two regularisation keys are two names, not one" begin
    # `:val` named both slots, and the two carry two different ground metrics, so route 1
    # of #614 widened them to the two names `field_dict` already used.
    l2role = AmbiguityRadiusCalibration(; alg = DualNormRadius())
    l2 = L2Regularisation(; val = l2role)
    @test PO.factory(l2, PRDIAG).val ≈ DNR_Z * norm(DNR_SD / sqrt(60), 2)

    # The Lp term reads the type-`q` metric of its own norm order. The order belongs to the
    # penalty, so the site hands it over with `bind_norm_order` and the caller states
    # nothing: the rule below carries no `p` of its own.
    lprole = AmbiguityRadiusCalibration(; alg = DualNormRadius())
    lp = LpRegularisation(; p = 3, val = lprole)
    @test PO.factory(lp, PRDIAG).val ≈ DNR_Z * norm(DNR_SD / sqrt(60), 1.5)
    @test PO.factory(lp, PRDIAG).p == 3

    # A second term of a different norm order gets a different number from the same rule,
    # which is the whole reason the order travels rather than sitting on the rule.
    lp2 = LpRegularisation(; p = 1.25, val = lprole)
    @test PO.factory(lp2, PRDIAG).val ≈ DNR_Z * norm(DNR_SD / sqrt(60), 5)
    @test PO.factory(lp, PRDIAG).val != PO.factory(lp2, PRDIAG).val

    # The order the site holds wins, so a rule that already carries one has it replaced.
    stated = LpRegularisation(; p = 3,
                              val = AmbiguityRadiusCalibration(;
                                                               alg = DualNormRadius(;
                                                                                    p = 1.25)))
    @test PO.factory(stated, PRDIAG).val ≈ PO.factory(lp, PRDIAG).val

    # The two keys reach two different numbers from one rule, which is what the widening
    # bought. A rule that read `:val` for both could not tell them apart.
    @test PO.factory(l2, PRDIAG).val != PO.factory(lp, PRDIAG).val

    # `bind_norm_order` fills the order through the role, and leaves everything else where
    # it was: a number, a plain function, and a rule that reads no order all cross whole.
    @test PO.bind_norm_order(l2role, 3).alg.p == 3
    @test PO.bind_norm_order(l2role, 3).alg.confidence == l2role.alg.confidence
    @test PO.bind_norm_order(0.3, 3) == 0.3
    cr_role = AmbiguityRadiusCalibration(; alg = ConcentrationRadius())
    @test PO.bind_norm_order(cr_role, 3).alg === cr_role.alg

    # The three older rules read no key, so the widening moved nothing for them.
    cr = ConcentrationRadius(; scale = 0.5)
    @test cr(:l2reg_val, PRDIAG, nothing, nothing) ==
          cr(:lpreg_val, PRDIAG, nothing, nothing)
end

@testset "DualNormRadius: l1 and linf reach the model with two coefficients" begin
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = "verbose" => false)
    rd = ReturnsResult(; nx = string.(1:4), X = X60)
    role = AmbiguityRadiusCalibration(; alg = DualNormRadius())

    # One rule, stated on both coefficients of one optimiser.
    opt = JuMPOptimiser(; slv = slv, pe = PR60, l1 = role, linf = role)
    mr = MeanRisk(; r = Variance(), opt = opt)
    attrs = PO.processed_jump_optimiser_attributes(mr.opt, rd)
    model = JuMP.Model()
    PO.set_model_scales!(model, mr.opt.sc, mr.opt.so)
    PO.set_maximum_ratio_factor_variables!(model, mr.obj)
    PO.set_w!(model, attrs.pr.X, mr.wi)
    PO.set_weight_constraints!(model, attrs.wb, mr.opt)
    PO.assemble_jump_model!(model, mr, mr.opt, attrs, rd, mr.r, mr.obj)

    c_l1 = JuMP.coefficient(model[:l1], model[:t_l1])
    c_linf = JuMP.coefficient(model[:linf], model[:t_linf])

    # The defect #614 fixes: the two penalties bound distance in two different norms, so
    # one rule must give them two coefficients. Under either older rule these are equal.
    @test c_l1 != c_linf
    @test c_l1 ≈ DualNormRadius()(:l1, PR60, PR60.w, slv)
    @test c_linf ≈ DualNormRadius()(:linf, PR60, PR60.w, slv)
    @test c_linf > c_l1

    # The gap grows with the universe, because the 1-norm of the error vector sums over the
    # assets and the ∞-norm does not.
    @test c_linf / c_l1 > 1
end

#=
Issue #623. Three rules of the two ambiguity families read a RETURNS quantity when their
slot owner prices a DRAWDOWN, and `DimensionalRateRadius` reads the same one. The mechanism
that closes the gap is the one commit 7e51431c45 built for the deformation family:
`calibration_series(x)` is the trait the owner answers, and `bind_series(slot, series)`
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
        r = alg(:r, PRDDA, nothing, nothing)
        d = PO.bind_series(alg, AbsoluteDrawdownSeries())(:r, PRDDA, nothing, nothing)
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
    @test DualNormRadius()(:r, PRDDA, nothing, nothing) ≈ z * norm(e_r, 1)
    @test PO.bind_series(DualNormRadius(), AbsoluteDrawdownSeries())(:r, PRDDA, nothing,
                                                                     nothing) ≈
          z * norm(e_d, 1)
    dn = PO.bind_series(DualNormRadius(), AbsoluteDrawdownSeries())
    @test dn(:l1, PRDDA, nothing, nothing) ≈ z * norm(e_d, Inf)
    @test dn(:r, PRDDA, nothing, nothing) > dn(:l1, PRDDA, nothing, nothing)

    # A STATED scale is the whole of the units, so it overrides the reading and the marker
    # then changes nothing.
    for alg in (ConcentrationRadius(; scale = 0.03), DimensionalRateRadius(; scale = 0.03))
        @test PO.bind_series(alg, AbsoluteDrawdownSeries())(:r, PRDDA, nothing, nothing) ==
              alg(:r, PRDDA, nothing, nothing)
    end

    # `bind_series` keeps every other field, and `bind_norm_order` keeps the marker.
    a = ConcentrationRadius(; confidence = 0.9, scale = 0.02)
    b = PO.bind_series(a, RelativeDrawdownSeries())
    @test b.confidence == a.confidence
    @test b.scale == a.scale
    @test b.series == RelativeDrawdownSeries()
    c = PO.bind_norm_order(PO.bind_series(DualNormRadius(; confidence = 0.9),
                                          AbsoluteDrawdownSeries()), 3)
    @test c.p == 3
    @test c.series == AbsoluteDrawdownSeries()
    @test c.confidence == 0.9
end

@testset "TailTermParity: the drawdown owner prices a drawdown against a CDaR" begin
    alpha = 0.05
    ret = TailTermParity(; alpha = alpha)
    dd = PO.bind_series(ret, AbsoluteDrawdownSeries())

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
    @test dd(:l, PRDDA, nothing, nothing) > ret(:l, PRDDA, nothing, nothing)

    # `ratio` scales the answer under every marker.
    @test PO.bind_series(TailTermParity(; ratio = 3, alpha = alpha),
                         AbsoluteDrawdownSeries())(:l, PRDDA, nothing, nothing) ≈
          3 * dd(:l, PRDDA, nothing, nothing)

    # The two markers name two different series of one column, so the relative reading is
    # its own number.
    rel = PO.bind_series(ret, RelativeDrawdownSeries())(:l, PRDDA, nothing, nothing)
    @test rel != dd(:l, PRDDA, nothing, nothing)

    # `bind_alpha` and `bind_series` compose in either order, and neither drops the other's
    # field.
    bare = TailTermParity(; ratio = 2)
    both = PO.bind_series(PO.bind_alpha(bare, alpha), AbsoluteDrawdownSeries())
    @test both.ratio == 2
    @test both.alpha == alpha
    @test both.series == AbsoluteDrawdownSeries()
    @test PO.bind_alpha(PO.bind_series(bare, AbsoluteDrawdownSeries()), alpha) == both

    # The weighted reading moves to the drawdown sample too.
    w = pweights(range(; start = 1, stop = 2, length = size(XDDA, 1)))
    @test dd(:l, PRDDA, w, nothing) != dd(:l, PRDDA, nothing, nothing)
end

@testset "Ambiguity calibration: the drawdown owner hands its own series over" begin
    @test PO.calibration_series(DistributionallyRobustConditionalDrawdownatRisk()) ==
          AbsoluteDrawdownSeries()
    @test PO.calibration_series(DistributionallyRobustConditionalValueatRisk()) ==
          ReturnsSeries()

    lrole = AmbiguityTailWeightCalibration(; alg = TailTermParity())
    rrole = AmbiguityRadiusCalibration(; alg = ConcentrationRadius())

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

    @test dr.l ≈
          PO.bind_series(TailTermParity(; alpha = 0.05), AbsoluteDrawdownSeries())(:l,
                                                                                   PRDDA,
                                                                                   nothing,
                                                                                   nothing)
    @test dr.r ≈ PO.bind_series(ConcentrationRadius(), AbsoluteDrawdownSeries())(:r, PRDDA,
                                                                                 nothing, nothing)
    @test vr.l ≈ TailTermParity(; alpha = 0.05)(:l, PRDDA, nothing, nothing)
    @test vr.r ≈ ConcentrationRadius()(:r, PRDDA, nothing, nothing)

    # The two owners resolve the SAME two keys and part company on the series alone.
    @test dr.l != vr.l
    @test dr.r != vr.r

    # A MARKER STATED ON THE RULE IS OVERWRITTEN by the measure that resolves it.
    wrong = AmbiguityRadiusCalibration(;
                                       alg = ConcentrationRadius(;
                                                                 series = AbsoluteDrawdownSeries()))
    o = PO.resolve_deferred_quantities(DistributionallyRobustConditionalValueatRisk(;
                                                                                    alpha = 0.05,
                                                                                    r = wrong),
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
    @test rg.l_b ≈ TailTermParity(; alpha = 0.1)(:l_b, PRDDA, nothing, nothing)
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
    X60_1 = X60[:, 1:1]
    PR60_1 = prior(EmpiricalPrior(), X60_1)
    dim = DimensionalRateRadius(; confidence = 0.95, scale = 0.5)
    @test size(PR60_1.X, 2) == 1
    @test dim(:r, PR60_1, nothing, nothing) ≈ 0.5 * sqrt(lq / 60)
    @test dim(:r, PR60_1, nothing, nothing) ≈ dim(:r, PR60_2, nothing, nothing)

    # -- #619. The radius is positive and finite at both ends of the admissible confidence
    # interval, so nothing re-checks it after resolution.
    for c in (1e-8, 0.5, 1 - 1e-8)
        v = DimensionalRateRadius(; confidence = c, scale = 1)(:r, PR60, nothing, nothing)
        @test v > 0
        @test isfinite(v)
    end

    # -- #620. A sample of NO dispersion gives a ball of no width, which is the right
    # answer for it: the empirical measure is the only measure such a sample supports. In
    # floating point the number is many orders below the smallest one a caller would state
    # rather than an exact zero, so the slot's `> 0` check passes and the model prices it.
    Xflat = repeat([0.01 0.02 0.03], 60)
    PRFLAT = prior(EmpiricalPrior(), Xflat)
    vflat = DualNormRadius()(:r, PRFLAT, nothing, nothing)
    @test vflat >= 0
    @test vflat < 1e-12
    @test isa(DistributionallyRobustConditionalValueatRisk(; r = vflat).r, Number)

    # -- #620. `confidence` is a per-coordinate level and the rule does not correct it for
    # the number of assets. The correction is a whole number of per cent rather than a
    # rounding, which is why it stays with the caller and is named in prose.
    N = size(X60, 2)
    z_coord = Distributions.quantile(Distributions.Normal(), 0.95)
    z_vec = Distributions.quantile(Distributions.Normal(), 1 - 0.05 / N)
    @test z_vec / z_coord > 1.3
    @test DualNormRadius(; confidence = 0.95)(:l1, PR60, nothing, nothing) * z_vec /
          z_coord ≈ DualNormRadius(; confidence = 1 - 0.05 / N)(:l1, PR60, nothing, nothing)

    # -- #620. The 1-norm arm sums the per-asset errors, so it prices them as if they moved
    # together. That is the LARGEST of the seven readings, and the ∞-norm arm is the
    # smallest, so the conservative reading is the one the rule takes.
    e = sqrt.(diag(PR60.sigma)) ./ sqrt(60)
    @test DualNormRadius()(:r, PR60, nothing, nothing) ≈ z_coord * sum(e)
    @test DualNormRadius()(:r, PR60, nothing, nothing) >
          DualNormRadius()(:l2reg_val, PR60, nothing, nothing)
    @test DualNormRadius()(:l2reg_val, PR60, nothing, nothing) >
          DualNormRadius()(:l1, PR60, nothing, nothing)
    @test DualNormRadius()(:l1, PR60, nothing, nothing) ≈ z_coord * maximum(e)

    # -- #621. `c` is the mean of the per-column CVaR and NOT the pooled one. A pooled tail
    # is drawn from the worst columns, so on a universe of unequal volatilities it is the
    # larger number and it would give the SMALLER weight.
    Xuneq = hcat(0.0005 .+ 0.002 * randn(RNG, 2520), 0.0005 .+ 0.01 * randn(RNG, 2520),
                 0.0005 .+ 0.05 * randn(RNG, 2520))
    PRUNEQ = prior(EmpiricalPrior(), Xuneq)
    c_col = mean(hand_cvar(view(Xuneq, :, j), alpha) for j in axes(Xuneq, 2))
    c_pool = hand_cvar(vec(Xuneq), alpha)
    @test c_pool > c_col
    l_col = TailTermParity(; ratio = 1, alpha = alpha)(:l, PRUNEQ, nothing, nothing)
    @test l_col ≈ abs(-mean(Xuneq)) / c_col
    @test l_col > abs(-mean(Xuneq)) / c_pool

    # -- #621. The parity claim, against the loss the measure prices. At the weight the
    # rule returns, one tail term is worth exactly `ratio` mean terms.
    m = -mean(XDAY)
    c = mean(hand_cvar(view(XDAY, :, j), alpha) for j in axes(XDAY, 2))
    for ratio in (0.5, 1.0, 2.0)
        l = TailTermParity(; ratio = ratio, alpha = alpha)(:l, PRDAY, nothing, nothing)
        @test l * c / abs(m) ≈ ratio
    end

    # -- #621. A stated `l = 1.0` is two trade-offs across the frequency pair: about forty
    # mean terms on the daily sample and under ten on the monthly one, the two numbers the
    # docstring names.
    l_day = TailTermParity(; ratio = 1, alpha = alpha)(:l, PRDAY, nothing, nothing)
    l_mon = TailTermParity(; ratio = 1, alpha = alpha)(:l, PRMON, nothing, nothing)
    @test 35 < inv(l_day) < 50
    @test inv(l_mon) < 10
end
