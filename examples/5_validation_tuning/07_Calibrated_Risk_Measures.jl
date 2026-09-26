#=
```@meta
Description = "Calibrated risk measures in PortfolioOptimisers.jl: a rule that computes alpha from the sample it is given instead of a fixed number."
```

# [Calibrated risk measures: a rule in place of a number](@id example-calibrated-risk-measures-a-rule-in-place-of-a-number)

`alpha = 0.05` is a statement about the *probability* of the tail. It says nothing about the
number of observations the tail holds. Over one year of daily data the 5% tail holds about 13
observations, and over five years it holds about 63. A number you state once means something
different on every sample it meets, and a cross-validation over folds of unequal length meets a
different sample on every fold.

A calibration slot takes either statement. It takes the number itself, and it takes a
calibration rule, an object that computes the number from the prior result of the sample it is
given. The rule runs inside [`factory`](@ref), the function an optimiser calls once per fit, so
a cross-validation recomputes the quantity on every fold, and the rest of the model stays the same.

Every `alpha`, `beta` and `kappa` of a risk measure, and every ambiguity radius and Esfahani-Kuhn
tail weight, is a calibration slot.

This example shows the slot as you use it. Section 1 loads the data, and each section
after it shows one use of a slot.

 2. A stated number and a rule side by side on the same measure.
 3. The refit per fold, over folds of unequal length.
 4. Three of the rules the library has, and the question that makes each one the right choice.
 5. The two slots that resolve together, where `alpha` resolves first and its number reaches the
    ``\kappa`` rule.
 6. The two tail-decay rules, which estimate the tail of one sample per end and for both ends.
 7. A plain function as a rule, which needs no library type.
 8. The slot bounds, which refuse a rule of the wrong family at construction.
 9. The ambiguity radius and the tail weight of the distributionally robust measure.

The regularisation coefficients `l1`, `linf`, [`L2Regularisation`](@ref) and
[`LpRegularisation`](@ref) are ambiguity radii too, and they take the same rule family. They
belong to the [regularisation example](@ref example-regularisation), which covers
those slots. The three norm ceilings `l2c`, `lpc` and `linfc` of [`JuMPOptimiser`](@ref) bound a
norm rather than price one, so they are a different quantity and take a family of their own,
[`AbstractNormCeilingCalibrationAlgorithm`](@ref). That example runs both families, and this one
stays on the slots that sit on a risk measure.
=#

using PortfolioOptimisers, PrettyTables, DataFrames, StatsBase, Statistics

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

Five years of daily data give the training windows enough room to differ in length. A rule uses
the length, so it gives a different number on each fold.
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

[`ConditionalValueatRisk`](@ref) takes `alpha = 0.05`, and it takes a rule that computes the
number. The slot names the quantity and the end of the distribution, so the rule states the
method alone. The other fields of the measure stay the same.
=#

cvar_stated = ConditionalValueatRisk(; alpha = 0.05)

cvar_rule = ConditionalValueatRisk(; alpha = ScenarioCount(; n = 25))

#=
The slot stores the rule itself, so the field gives back what you wrote.
=#

cvar_rule.alpha

#=
[`ScenarioCount`](@ref) states the tail's population rather than its probability: `alpha = n / T`
leaves `n` observations in the tail whatever the sample length is.

A rule needs a prior result, because it takes the sample size and the moments from one. The
number therefore appears when [`factory`](@ref) runs, which is the function the optimiser calls on
the measure once it has fitted the prior.
=#

pr = prior(EmpiricalPrior(), rd)
println("resolved alpha over the whole sample = $(factory(cvar_rule, pr).alpha)")

#=
Call `expected_risk` on a matrix of returns and the rule has no prior result to read, so the call
refuses the rule and names what to pass instead of guessing a number.
=#

w0 = fill(inv(size(rd.X, 2)), size(rd.X, 2))
try
    expected_risk(cvar_rule, w0, rd.X)
catch e
    println(sprint(showerror, e))
end

#=
Pass the prior result instead. The rule resolves, and the measure evaluates.
=#

println("calibrated risk = $(expected_risk(cvar_rule, w0, pr))")
println("stated risk     = $(expected_risk(cvar_stated, w0, pr))")

#=
## 3. The refit per fold

[`IndexWalkForward`](@ref) with `expand_train = true` grows the training window one test block at
a time, so every fold trains on a window of a different length.
=#

iwf = IndexWalkForward(252, 63; expand_train = true)
iwf_res = split(iwf, rd)
println("number of folds = $(length(iwf_res.train_idx))")

#=
`factory` is the function the optimiser itself calls, so resolving a measure against a fold's own
prior gives you the measure that fold optimises. This is how you get the number, because the
library has no other function for it.
=#

fold_prior(idx) = prior(EmpiricalPrior(), rd.X[idx, :])
resolved(r, idx, key) = getproperty(factory(r, fold_prior(idx)), key)

#=
The stated `alpha` is one number for every fold. The rule's `alpha` falls as the window grows,
and the count of observations in the tail stays fixed.
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
We put the measure into a [`MeanRisk`](@ref) unchanged. The cross-validation refits it on every
fold, and nothing tells the optimiser that a slot holds a rule.
=#

mr_rule = MeanRisk(; r = cvar_rule, opt = JuMPOptimiser(; slv = slv))
mr_stated = MeanRisk(; r = cvar_stated, opt = JuMPOptimiser(; slv = slv))

pred_rule = cross_val_predict(mr_rule, rd, iwf)
pred_stated = cross_val_predict(mr_stated, rd, iwf)

#=
Both runs use the same folds, so the out-of-sample series are comparable.
=#

var_rm = LowOrderMoment(; alg = SecondMoment())
println("calibrated out-of-sample variance = $(expected_risk(var_rm, pred_rule))")
println("stated out-of-sample variance     = $(expected_risk(var_rm, pred_stated))")

#=
We put the level the rule produced next to the out-of-sample risk of each run, fold by fold. The two runs differ only in the measure the folds priced.
=#

fold_risk = DataFrame(:fold => 1:length(iwf_res.train_idx),
                      :T => length.(iwf_res.train_idx),
                      :alpha_rule => fold_table.alpha_rule,
                      :risk_rule => expected_risk.(Ref(var_rm), pred_rule.pred),
                      :risk_stated => expected_risk.(Ref(var_rm), pred_stated.pred))
pretty_table(fold_risk; formatters = [numfmt])

#=
We print the weights of the calibrated run, one column per fold.
=#

pretty_table(hcat(DataFrame(:tickers => rd.nx),
                  DataFrame(reduce(hcat, getproperty.(pred_rule.res, :w)),
                            Symbol.(1:length(pred_rule.res)))); formatters = [resfmt])

#=
## 4. Three of the rules

The library has eleven rules over the five families. Five of them compute a significance level or
a deformation parameter, and this section runs three of those five. Each answers a different
question about the sample.

  - [`ScenarioCount`](@ref) answers *how many observations must the tail hold*. It uses Kish's
    effective sample size when observation weights are stated, because a weighted tail has
    fewer independent observations than its row count suggests.
  - [`RateSignificance`](@ref) answers *how fast may the tail move outwards*. `alpha = c / sqrt(T)`
    leaves `c * sqrt(T)` observations in the tail, which grows with the sample but more slowly
    than the sample does. That is the rate at which a sample mean's own error falls. It uses the
    raw row count, because a rate is a statement about the length of the record.
  - [`EntropyBudget`](@ref) answers *what may the deformation cost*. Section 5 puts it on a
    ``\kappa`` slot.

The deformation family has two more rules, and both answer a different question:
*how fast does this sample's tail decay*. Each estimates a tail index and returns its
reciprocal. [`HillTailDecay`](@ref) standardises every column by its own dispersion and keeps
the sign of the end, so a skewed sample gives one number for the loss end and another for the
gain end. [`RadialTailDecay`](@ref) whitens each observation with the covariance matrix and
measures a distance, so it returns one number for both ends. Both fit any ``\kappa`` slot, as
[`EntropyBudget`](@ref) does, and section 6 runs both.
=#

count_rule = ConditionalValueatRisk(; alpha = ScenarioCount(; n = 25))
rate_rule = ConditionalValueatRisk(; alpha = RateSignificance(; c = 1.5))

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
Under the scenario count the tail population `count_tail` is flat, and under the rate
`rate_tail` grows with the square root of `T`.

Observation weights separate the two rules a second time. A measure that has `w` passes those
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
The rate is unchanged, because it does not use the weights.

## 5. The two slots that resolve together

[`RelativisticValueatRisk`](@ref) has two calibration slots, `alpha` and `kappa`.
[`EntropyBudget`](@ref) states the price of the deformation directly: [`RRM`](@ref) multiplies its
dual variable by `kappa_log(inv(alpha * T), kappa)`, and the rule returns the ``\kappa`` that
meets a stated value of that coefficient.

The rule therefore needs the `alpha` of the same measure. `alpha` resolves first, and the
``\kappa`` rule uses the number it produced, so both slots resolve in one pass over the measure.
=#

rlvar_rule = RelativisticValueatRisk(; alpha = ScenarioCount(; n = 25),
                                     kappa = EntropyBudget(; target = -6.0))
rlvar_stated = RelativisticValueatRisk(; alpha = 0.05,
                                       kappa = EntropyBudget(; target = -6.0))

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
The `kappa` column is flat and the `kappa_stated_alpha` column is not. The coefficient depends on
`inv(alpha * T)`, and a scenario count fixes `alpha * T` at the count itself, so the budget gives
the same deformation on every fold. A stated `alpha` lets `alpha * T` grow with the window, so
the same budget gives a different deformation each time.

The rule has one check, and it is not a range check on the ``\kappa`` it returns. With
``u = 1/(\alpha T)``, the first argument of `kappa_log` above, the coefficient reaches only the
band between ``\ln(u)`` and ``\sinh(\ln(u))``, so no ``\kappa`` meets a target outside that band.
The band moves with `alpha` and with the sample, and the refusal names both.
=#

try
    resolved(RelativisticValueatRisk(; alpha = 0.05,
                                     kappa = EntropyBudget(; target = -1.5)), first_idx,
             :kappa)
catch e
    println(sprint(showerror, e))
end

#=
## 6. The two tail-decay rules

[`HillTailDecay`](@ref) and [`RadialTailDecay`](@ref) ask the same question, *how fast does this
sample's tail decay*, and they measure two different quantities to answer it. Each estimates a tail
index and returns its reciprocal, which is the ``\kappa`` whose deformed exponential decays at
that rate.

[`RelativisticValueatRiskRange`](@ref) has a ``\kappa`` slot at each end, so one measure gives
both answers. Each end resolves a pair of its own. `kappa_a` uses `alpha`, and `kappa_b` uses
`beta`.
=#

hill_range = RelativisticValueatRiskRange(; kappa_a = HillTailDecay(),
                                          kappa_b = HillTailDecay())
radial_range = RelativisticValueatRiskRange(; kappa_a = RadialTailDecay(),
                                            kappa_b = RadialTailDecay())

hill_res = factory(hill_range, pr)
radial_res = factory(radial_range, pr)

decay_table = DataFrame(:rule => ["HillTailDecay", "RadialTailDecay"],
                        :kappa_a => [hill_res.kappa_a, radial_res.kappa_a],
                        :kappa_b => [hill_res.kappa_b, radial_res.kappa_b])
pretty_table(decay_table; formatters = [numfmt])

#=
The Hill row has two numbers, because
the rule keeps the sign of the end and estimates the loss tail under `kappa_a` and the gain tail
under `kappa_b`. The Radial row has one number twice, because the rule whitens each observation
and measures a distance, and a distance has no sign. The two Hill numbers differ because the
sample is skewed. A sample with no skew has one index at both ends, and two estimates of it then
differ by their own noise alone.
=#

println("pooled skewness = $(skewness(vec(pr.X)))")

#=
The Hill number is the tail index that the standardised columns share, estimated from the pool of
every column's values, and the Radial number is the index of the whole cross-section's radius.

The rule refits per fold, as every other rule does. The two ends move apart by a different
amount on every window, and which end has the heavier tail is a property of the
window, not of the whole record.
=#

decay_fold = DataFrame(:fold => 1:length(iwf_res.train_idx),
                       :T => length.(iwf_res.train_idx),
                       :kappa_a => [resolved(hill_range, idx, :kappa_a)
                                    for idx in iwf_res.train_idx],
                       :kappa_b => [resolved(hill_range, idx, :kappa_b)
                                    for idx in iwf_res.train_idx])
pretty_table(decay_fold; formatters = [numfmt])

#=
The Radial rule refuses the first fold, because of the count. Both rules estimate the tail from
the `k` most extreme values of a pool, and both refuse a `k` below `kmin`. The Hill pool holds
`T * N` standardised values, and the radial pool holds `T` distances. The same minimum therefore
binds `N` times harder on the radial side, and a one-year fold at `alpha = 0.05` leaves it 13
distances.
=#

try
    resolved(radial_range, first_idx, :kappa_a)
catch e
    println(sprint(showerror, e))
end

#=
## 7. A plain function as a rule

The library runs a rule by calling it, so a callable struct and a plain function work the same
way. A plain function is the rule that needs no library type, and it is the shortest way to state
a one-off rule. The signature is `(key, pr, w, slv, ctx)`:

  - `key`: name of the slot being resolved;
  - `pr`: prior result the rule takes the sample size and the moments from;
  - `w`: effective observation weights, or `nothing`;
  - `slv`: effective solver, or `nothing`;
  - `ctx`: a [`CalibrationContext`](@ref), which holds what the slot's owner knows and `key`
    does not: the significance level of a sibling slot, the series the owner prices, and the norm
    order of the constraint the quantity stands in. A rule that uses none of the three names the
    type and ignores it, as this one does.

`key` matters on a Range measure, where one function serves both ends and looks up its own budget
for each.
=#

tail_budget = Dict(:alpha => 25, :beta => 50)
budgeted(key, pr, w, slv, ctx) = tail_budget[key] / size(pr.X, 1)

vrr = ValueatRiskRange(; alpha = budgeted, beta = budgeted)
vrr_res = factory(vrr, fold_prior(first_idx))
println("alpha = $(vrr_res.alpha), beta = $(vrr_res.beta)")

#=
Every Range measure that has a `beta` defaults it to `alpha`. The rule states the method and the
slot states the end, so one rule serves both ends, and `beta` is the same object as `alpha`.
=#

owa_range = OrderedWeightsArrayConditionalValueatRiskRange(; alpha = count_rule.alpha)
println("beta is a $(typeof(owa_range.beta).name.name), same rule = $(owa_range.beta === count_rule.alpha)")

#=
## 8. The slot bounds refuse a rule of the wrong family

Each slot's type bound names the one rule family that computes the quantity the slot holds. A
deformation rule in a significance slot therefore fails at construction, before any data reaches
the measure.
=#

try
    ConditionalValueatRisk(; alpha = EntropyBudget(; target = -6.0))
catch e
    println(sprint(showerror, e))
end

#=
A radius and a tail weight are two quantities as well, so each has a family of its own and
each slot refuses the other's rule.
=#

try
    DistributionallyRobustConditionalValueatRisk(; r = TailTermParity(; ratio = 1))
catch e
    println(sprint(showerror, e))
end

#=
## 9. The ambiguity radius and the tail weight

[`DistributionallyRobustConditionalValueatRisk`](@ref) prices a ball of probability measures
around the empirical one. Its `r` is the radius of that ball and its `l` is the weight of the
tail term, and both are calibration slots beside `alpha`. So one measure can refit all three
quantities per fold.

The library has four radius rules, and this section runs the two below.

  - [`ConcentrationRadius`](@ref) is the Blanchet-Kang-Murthy form: a scale in the units of the
    returns times the square root of a chi-squared quantile over the sample size. A wider universe
    gives a wider ball at a fixed confidence level, and a longer sample shrinks it. `scale = nothing`
    takes the average asset volatility from the prior result.
  - [`RateRadius`](@ref) is `c / sqrt(T)`. The rate is the part of the form to trust and `c` is the
    part to calibrate, so you set a radius by cross-validating over `c`.

The other two answer a question these two do not. [`DimensionalRateRadius`](@ref) shrinks the
ball at the rate the number of assets sets rather than at the square-root rate of the sample
length, which is far slower over a wide universe. [`DualNormRadius`](@ref) uses the slot's own
key, picks the ground metric that slot names, and returns the sampling error in it, so two
slots of two different norms get two different numbers. Both fit any radius slot, as the two
above do, and the [regularisation example](@ref example-regularisation) runs
them, because the slots that separate them are the four penalty coefficients of
[`JuMPOptimiser`](@ref).

The tail-weight family has one rule, [`TailTermParity`](@ref), which prices the tail term of
the loss at a stated multiple of its mean term. A function you write works in the slot too, in
the form section 7 shows.
=#

drcvar = DistributionallyRobustConditionalValueatRisk(; alpha = count_rule.alpha,
                                                      r = ConcentrationRadius(;
                                                                              confidence = 0.95),
                                                      l = TailTermParity(; ratio = 1))
drcvar_rate = DistributionallyRobustConditionalValueatRisk(; r = RateRadius(; c = 0.02))

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
not, because of its scale. `scale = nothing` takes the average asset volatility from the fold's
own prior, so a window that takes in a more volatile period gives a wider ball even though it is
longer. A radius is in the units of the returns, and the rule takes those units from the fold it
is given.

The `l` column is a ratio of two scales the rule measures on the fold, so it moves with both. The
numerator is the mean loss of the pooled cross-section, and the denominator is the mean
per-column CVaR at the fold's own `alpha`. `ratio = 1` therefore prices one tail term at one mean
term. A fold whose mean return is near zero has a numerator near zero, and so a small `l`.

The measure optimises in the same way the calibrated CVaR did.
=#

mr_drcvar = MeanRisk(; r = drcvar, opt = JuMPOptimiser(; slv = slv))
pred_drcvar = cross_val_predict(mr_drcvar, rd, iwf)
println("robust out-of-sample variance = $(expected_risk(var_rm, pred_drcvar))")

#=
## 10. What to take away

  - A calibration slot takes a number or a rule, and the other fields of the measure stay the
    same.
  - The rule resolves inside [`factory`](@ref), so a cross-validation refits it per fold, and you
    get the number a fold produced with `factory(r, pr)`.
  - The rule states the question. A scenario count fixes the count of observations in the tail, a
    rate lets that count grow with the square root of the record, and an entropy budget fixes the
    price of a deformation.
  - `alpha` resolves before ``\kappa``, in one pass. A scenario count on `alpha` fixes
    `alpha * T`, so the same entropy budget gives the same ``\kappa`` on every fold, and a stated
    `alpha` does not.
  - The two tail-decay rules measure two different quantities. [`HillTailDecay`](@ref) answers per
    end, and [`RadialTailDecay`](@ref) answers once for both.
  - A plain function of `(key, pr, w, slv, ctx)` is a rule, which covers the one-off case in every
    family.
  - A slot names its quantity, and its type bound refuses a rule of another family at
    construction.
  - The [regularisation example](@ref example-regularisation) runs the other two
    radius rules and the three norm ceilings of `JuMPOptimiser`. `DualNormRadius` gives a
    different radius to penalties of different norms, and the four penalty coefficients of
    `JuMPOptimiser` are where those norms differ.
=#
