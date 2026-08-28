#=
# Calibrated risk measures: a rule in place of a number

`alpha = 0.05` is a statement about the *probability* of the tail. It is not a statement about
the number of observations the tail holds. Over one year of daily data the 5% tail holds about
13 observations; over five years it holds about 63. The number a caller states once therefore
means something different on every sample it meets, and a cross-validation over folds of
unequal length meets a different sample on every fold.

A **calibration slot** takes either kind of statement. It takes the number itself, and it takes
a **Calibration Rule**, which computes the number from the prior result of the sample in front
of it. The rule runs inside [`factory`](@ref), the verb an optimiser calls once per fit, so a
cross-validation refits the quantity on every fold and no other part of the model moves.

Every `alpha`, `beta`, `kappa`, ambiguity radius and Esfahani-Kuhn tail weight the library carries
is a calibration slot.

This example shows the slot from the caller's side.

 1. A stated number and a rule side by side on the same measure.
 2. The refit per fold, over folds of unequal length.
 3. The three rules that ship, and the reading that makes each one the right choice.
 4. The travelling pair, where `alpha` resolves first and its number reaches the ``\kappa`` rule.
 5. A plain function as a rule, which is the case that has no type.
 6. The role bounds, which refuse a head rule in a tail slot at construction.
 7. The ambiguity radius and the tail weight of the distributionally robust measure.

The regularisation coefficients `l1`, `linf`, [`L2Regularisation`](@ref) and
[`LpRegularisation`](@ref) are ambiguity radii too, and they take the same role. They belong to
the [regularisation example](../4_constraints_costs/07_Regularisation.md), which owns those
slots; this example stays on the slots that sit on a risk measure.
=#

using PortfolioOptimisers, PrettyTables, DataFrames, StatsBase, Statistics

## Format for pretty tables.
numfmt = (v, i, j) -> begin
    return isa(v, AbstractFloat) ? round(v; sigdigits = 4) : v
end;
resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;

#=
## 1. Setting up

Five years of daily data give enough length for the folds to differ from each other by a
meaningful amount, which is the whole point of the demonstration.
=#

using CSV, TimeSeries, Clarabel

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252 * 5):end]
rd = prices_to_returns(X)
println("size(rd.X) = $(size(rd.X))")

slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.95),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel3, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.9),
              check_sol = (; allow_local = true, allow_almost = true))];

#=
## 2. A stated number and a rule side by side

[`ConditionalValueatRisk`](@ref) takes `alpha = 0.05`, and it takes a
[`SignificanceTailCalibration`](@ref) holding a rule. The role names the end of the distribution
the slot addresses, and the rule it carries in `alg` is what computes the number. Nothing else on
the measure changes.
=#

cvar_stated = ConditionalValueatRisk(; alpha = 0.05)

cvar_rule = ConditionalValueatRisk(;
                                   alpha = SignificanceTailCalibration(;
                                                                       alg = ScenarioCount(;
                                                                                           n = 25)))

#=
[`ScenarioCount`](@ref) states the tail's population rather than its probability: `alpha = n / T`
leaves `n` observations in the tail whatever the sample length is.

A rule needs a prior result, because it reads the sample size and the moments off one. So the
number appears when [`factory`](@ref) runs, which is the verb the optimiser calls on the measure
once it has fitted the prior.
=#

pr = prior(EmpiricalPrior(), rd)
println("resolved alpha over the whole sample = $(factory(cvar_rule, pr).alpha)")

#=
The value-level entry point has no prior result to resolve against, so it refuses the rule and
names the way out rather than guessing a number.
=#

w0 = fill(inv(size(rd.X, 2)), size(rd.X, 2))
try
    expected_risk(cvar_rule, w0, rd.X)
catch e
    println(sprint(showerror, e))
end

#=
Passing the prior result instead resolves the rule and evaluates the measure.
=#

println("calibrated risk = $(expected_risk(cvar_rule, w0, pr))")
println("stated risk     = $(expected_risk(cvar_stated, w0, pr))")

#=
## 3. The refit per fold

This is the reason the slot widened. [`IndexWalkForward`](@ref) with `expand_train = true` grows
the training window one test block at a time, so the folds are of unequal length by construction.
=#

iwf = IndexWalkForward(252, 63; expand_train = true)
iwf_res = split(iwf, rd)
println("number of folds = $(length(iwf_res.train_idx))")

#=
`factory` is the verb the optimiser itself calls, so resolving a measure against a fold's own
prior reports the measure that fold optimises. This is the shortest honest way to read the
number, and the library needs no accessor for it.
=#

fold_prior(idx) = prior(EmpiricalPrior(), rd.X[idx, :])
resolved(r, idx, key) = getproperty(factory(r, fold_prior(idx)), key)

#=
The stated `alpha` is one number for every fold. The rule's `alpha` falls as the window grows,
and the tail's population is what stays fixed.
=#

fold_table = DataFrame(:fold => 1:length(iwf_res.train_idx),
                       :T => length.(iwf_res.train_idx),
                       :alpha_stated => fill(cvar_stated.alpha, length(iwf_res.train_idx)),
                       :tail_count_stated => cvar_stated.alpha * length.(iwf_res.train_idx),
                       :alpha_rule =>
                           [resolved(cvar_rule, idx, :alpha) for idx in iwf_res.train_idx])
fold_table.tail_count_rule = fold_table.alpha_rule .* fold_table.T
pretty_table(fold_table; formatters = [numfmt])

#=
The measure goes into a [`MeanRisk`](@ref) unchanged, and the cross-validation refits it per fold
without being told that anything is being calibrated.
=#

mr_rule = MeanRisk(; r = cvar_rule, opt = JuMPOptimiser(; slv = slv))
mr_stated = MeanRisk(; r = cvar_stated, opt = JuMPOptimiser(; slv = slv))

pred_rule = cross_val_predict(mr_rule, rd, iwf)
pred_stated = cross_val_predict(mr_stated, rd, iwf)

#=
Both run to the same folds, so the out-of-sample series are comparable.
=#

var_rm = LowOrderMoment(; alg = SecondMoment())
println("calibrated out-of-sample variance = $(expected_risk(var_rm, pred_rule))")
println("stated out-of-sample variance     = $(expected_risk(var_rm, pred_stated))")

#=
Fold by fold, the level the rule produced sits beside the out-of-sample risk of each run. The two
runs differ only in the measure the folds priced.
=#

fold_risk = DataFrame(:fold => 1:length(iwf_res.train_idx),
                      :T => length.(iwf_res.train_idx),
                      :alpha_rule => fold_table.alpha_rule,
                      :risk_rule => expected_risk.(Ref(var_rm), pred_rule.pred),
                      :risk_stated => expected_risk.(Ref(var_rm), pred_stated.pred))
pretty_table(fold_risk; formatters = [numfmt])

#=
The weights the calibrated run held, one column per fold.
=#

pretty_table(hcat(DataFrame(:tickers => rd.nx),
                  DataFrame(reduce(hcat, getproperty.(pred_rule.res, :w)),
                            Symbol.(1:length(pred_rule.res)))); formatters = [resfmt])

#=
## 4. The three rules

Three rules ship, and each answers a different question about the sample.

  - [`ScenarioCount`](@ref) answers *how many observations must the tail hold*. It reads Kish's
    effective sample size when observation weights are stated, because a weighted tail holds
    fewer independent observations than its row count suggests.
  - [`RateSignificance`](@ref) answers *how fast may the tail move outwards*. `alpha = c / sqrt(T)`
    leaves `c * sqrt(T)` observations in the tail, which grows with the sample but more slowly
    than the sample does. That is the rate at which a sample mean's own error falls. It reads the
    raw row count, because a rate is a statement about the length of the record.
  - [`EntropyBudget`](@ref) answers *what may the deformation cost*. It is the deformation family's
    rule, and section 5 puts it on a ``\kappa`` slot.
=#

count_rule = ConditionalValueatRisk(;
                                    alpha = SignificanceTailCalibration(;
                                                                        alg = ScenarioCount(;
                                                                                            n = 25)))
rate_rule = ConditionalValueatRisk(;
                                   alpha = SignificanceTailCalibration(;
                                                                       alg = RateSignificance(;
                                                                                              c = 1.5)))

rule_table = DataFrame(:fold => 1:length(iwf_res.train_idx),
                       :T => length.(iwf_res.train_idx),
                       :scenario_count =>
                           [resolved(count_rule, idx, :alpha) for idx in iwf_res.train_idx],
                       :rate =>
                           [resolved(rate_rule, idx, :alpha) for idx in iwf_res.train_idx])
rule_table.count_tail = rule_table.scenario_count .* rule_table.T
rule_table.rate_tail = rule_table.rate .* rule_table.T
pretty_table(rule_table; formatters = [numfmt])

#=
The two columns of tail populations are the difference between the two readings. The scenario
count holds its population flat, and the rate lets it grow with the square root of the record.

Observation weights separate the two rules a second time. A measure that carries `w` hands those
weights to the rule, and [`ScenarioCount`](@ref) divides by Kish's effective sample size rather
than by the row count. The effective size is the smaller of the two, so the weighted level is the
higher.
=#

obs_w = pweights(range(; start = 1, stop = 2, length = length(iwf_res.train_idx[1])))
count_weighted = ConditionalValueatRisk(; alpha = count_rule.alpha, w = obs_w)
rate_weighted = ConditionalValueatRisk(; alpha = rate_rule.alpha, w = obs_w)

first_idx = iwf_res.train_idx[1]
weight_table = DataFrame(:rule => ["ScenarioCount", "RateSignificance"],
                         :unweighted => [resolved(count_rule, first_idx, :alpha),
                                         resolved(rate_rule, first_idx, :alpha)],
                         :weighted => [resolved(count_weighted, first_idx, :alpha),
                                       resolved(rate_weighted, first_idx, :alpha)])
pretty_table(weight_table; formatters = [numfmt])

#=
The rate is unchanged, because it never reads the weights.

## 5. The travelling pair

[`RelativisticValueatRisk`](@ref) carries two calibration slots, `alpha` and `kappa`.
[`EntropyBudget`](@ref) states the price of the deformation directly: [`RRM`](@ref) multiplies its
dual variable by `kappa_log(inv(alpha * T), kappa)`, and the rule returns the ``\kappa`` that
meets a stated value of that coefficient.

The rule therefore reads its sibling `alpha`. `alpha` resolves first, and the number it produced
travels to the ``\kappa`` rule, so the pair resolves in one pass over the measure.
=#

rlvar_rule = RelativisticValueatRisk(;
                                     alpha = SignificanceTailCalibration(;
                                                                         alg = ScenarioCount(;
                                                                                             n = 25)),
                                     kappa = DeformationTailCalibration(;
                                                                        alg = EntropyBudget(;
                                                                                            target = -6.0)))
rlvar_stated = RelativisticValueatRisk(; alpha = 0.05,
                                       kappa = DeformationTailCalibration(;
                                                                          alg = EntropyBudget(;
                                                                                              target = -6.0)))

pair_table = DataFrame(:fold => 1:length(iwf_res.train_idx),
                       :T => length.(iwf_res.train_idx),
                       :alpha =>
                           [resolved(rlvar_rule, idx, :alpha) for idx in iwf_res.train_idx],
                       :kappa =>
                           [resolved(rlvar_rule, idx, :kappa) for idx in iwf_res.train_idx],
                       :kappa_stated_alpha => [resolved(rlvar_stated, idx, :kappa)
                                               for idx in iwf_res.train_idx])
pretty_table(pair_table; formatters = [numfmt])

#=
The `kappa` column is flat and the `kappa_stated_alpha` column is not, and the reason is the
handover. The coefficient reads `inv(alpha * T)`, and a scenario count fixes `alpha * T` at the
count itself, so the budget buys the same deformation on every fold. A stated `alpha` lets
`alpha * T` grow with the window, so the same budget buys a different deformation each time.

The rule carries one check, and it is not a range check on the ``\kappa`` it returns. The
coefficient reaches only the band between ``\ln(u)`` and ``\sinh(\ln(u))``, so a target outside
that band has no root at all. The band moves with `alpha` and with the sample, and the refusal
names both.
=#

try
    resolved(RelativisticValueatRisk(; alpha = 0.05,
                                     kappa = DeformationTailCalibration(;
                                                                        alg = EntropyBudget(;
                                                                                            target = -1.5))),
             first_idx, :kappa)
catch e
    println(sprint(showerror, e))
end

#=
## 6. A plain function as a rule

A rule is run by calling it, so a callable struct and a plain function are the same thing to the
resolver. A closure over a caller's own data is the case that has no type, and it is the shortest
way to state a one-off rule. The signature is `(key, pr, w, slv)`:

  - `key`: name of the slot being resolved;
  - `pr`: prior result the rule reads the sample size and the moments off;
  - `w`: effective observation weights, or `nothing`;
  - `slv`: effective solver, or `nothing`.

`key` earns its keep on a Range measure, where one function serves both ends and reads its own
budget for each.
=#

tail_budget = Dict(:alpha => 25, :beta => 50)
budgeted(key, pr, w, slv) = tail_budget[key] / size(pr.X, 1)

vrr = ValueatRiskRange(; alpha = SignificanceTailCalibration(; alg = budgeted),
                       beta = SignificanceHeadCalibration(; alg = budgeted))
vrr_res = factory(vrr, fold_prior(first_idx))
println("alpha = $(vrr_res.alpha), beta = $(vrr_res.beta)")

#=
The two ordered-weights Range measures default `beta` to `alpha`, and [`mirror_role`](@ref) keeps
that default alive: a tail role crosses over as the head role holding the same rule.
=#

owa_range = OrderedWeightsArrayConditionalValueatRiskRange(; alpha = count_rule.alpha)
println("beta is a $(typeof(owa_range.beta).name.name), same rule = $(owa_range.beta.alg === count_rule.alpha.alg)")

#=
## 7. The role bounds refuse a mismatch

The role is what the slot's type bound admits, and each bound names one role and no other. A head
rule in a tail slot is therefore refused at construction, before any data is in sight, and no
guard method is written for it.
=#

try
    ConditionalValueatRisk(;
                           alpha = SignificanceHeadCalibration(;
                                                               alg = ScenarioCount(;
                                                                                   n = 25)))
catch e
    println(sprint(showerror, e))
end

#=
A role inside another role's `alg` field is refused by the same mechanism. A role is configuration
that carries an algorithm, so the `alg` bound admits rules and functions only.
=#

try
    SignificanceTailCalibration(;
                                alg = SignificanceTailCalibration(;
                                                                  alg = ScenarioCount(;
                                                                                      n = 25)))
catch e
    println(sprint(showerror, e))
end

#=
## 8. The ambiguity radius and the tail weight

[`DistributionallyRobustConditionalValueatRisk`](@ref) prices a ball of probability measures
around the empirical one. Its `r` is the radius of that ball and its `l` is the weight of the
tail term, and both are calibration slots beside `alpha`. So one measure can refit all three
quantities per fold.

Two radius rules ship.

  - [`ConcentrationRadius`](@ref) is the Blanchet-Kang-Murthy form: a scale in the units of the
    returns times the square root of a chi-squared quantile over the sample size. A wider universe
    buys a wider ball at a fixed confidence level, and a longer sample shrinks it. `scale = nothing`
    reads the average asset volatility off the prior result.
  - [`RateRadius`](@ref) is `c / sqrt(T)`. The rate is the part of the form to trust and `c` is the
    part to calibrate, so a cross-validation over `c` is the honest route to a radius.

The tail-weight family ships no rule. A caller's own function is the whole of its population, which
is the case section 6 covers.
=#

drcvar = DistributionallyRobustConditionalValueatRisk(; alpha = count_rule.alpha,
                                                      r = AmbiguityRadiusCalibration(;
                                                                                     alg = ConcentrationRadius(;
                                                                                                               confidence = 0.95)),
                                                      l = AmbiguityTailWeightCalibration(;
                                                                                         alg = (key, pr, w, slv) -> 1.5))
drcvar_rate = DistributionallyRobustConditionalValueatRisk(;
                                                           r = AmbiguityRadiusCalibration(;
                                                                                          alg = RateRadius(;
                                                                                                           c = 0.02)))

amb_table = DataFrame(:fold => 1:length(iwf_res.train_idx),
                      :T => length.(iwf_res.train_idx),
                      :alpha =>
                          [resolved(drcvar, idx, :alpha) for idx in iwf_res.train_idx],
                      :concentration_r =>
                          [resolved(drcvar, idx, :r) for idx in iwf_res.train_idx],
                      :rate_r =>
                          [resolved(drcvar_rate, idx, :r) for idx in iwf_res.train_idx],
                      :l => [resolved(drcvar, idx, :l) for idx in iwf_res.train_idx])
pretty_table(amb_table; formatters = [numfmt])

#=
The rate radius falls with every fold, because the window only grows. The concentration radius does
not, and the reason is its scale: `scale = nothing` reads the average asset volatility off the
fold's own prior, so a window that takes in a more volatile period buys a wider ball even though it
is longer. A radius is in the units of the returns, and this is what reading those units off the
sample looks like.

The measure optimises in the same way the calibrated CVaR did.
=#

mr_drcvar = MeanRisk(; r = drcvar, opt = JuMPOptimiser(; slv = slv))
pred_drcvar = cross_val_predict(mr_drcvar, rd, iwf)
println("robust out-of-sample variance = $(expected_risk(var_rm, pred_drcvar))")

#=
## 9. What to take away

  - A calibration slot takes a number or a rule, and nothing else about the measure changes.
  - The rule resolves inside [`factory`](@ref), so a cross-validation refits it per fold and
    `factory(r, pr)` is how a caller reads the number a fold produced.
  - The rule states the question. A scenario count fixes the tail's population, a rate lets it grow
    with the square root of the record, and an entropy budget fixes the price of a deformation.
  - `alpha` and ``\kappa`` travel together, so a scenario count on `alpha` holds the entropy band
    still and a stated `alpha` does not.
  - A plain function of `(key, pr, w, slv)` is a rule, which covers the one-off case and the
    tail-weight family that ships no rule of its own.
  - A role names an end of the distribution, and its slot's type bound refuses the other end at
    construction.
=#
