---
status: accepted
---

# The calibration channel is parallel to the Deferred Quantity, and the role is in the bound

## Context

ADR 0070 answered [#311](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/311): no
ambiguity set becomes a type, and a radius slot admits the rule that computes it instead. It widened
twelve slots across six types and left two holes. Nothing computed a Kaniadakis `kappa`, and the
tail-probability slots of the risk measures — the surface
[#352](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/352) charted — were still bare
numbers.

Map [#580](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/580) built the rest. It
inherited two mechanisms that already carry a value the caller does not state.

**The Deferred-Quantity channel of ADR 0051.** `deferred_slots(x)` declares the slots, the per-type
`resolve_deferred_quantities(x, pr, slv)` resolves them and rebuilds once, `resolve_slot` resolves
one slot, and `assert_resolved_slots` refuses at a value-level entry point. `resolve_slot`'s whole
body is

```julia
resolve_slot(dq::DeferredQuantity, key::Symbol, pr::AbstractPriorResult) =
    deferred_quantity(fit_deferred_quantity(dq, pr), key)
```

Fit the estimator, then read the quantity off the fit.

**The tag table of ADR 0061.** A propagation channel is a row of `PROP_TAG_CHANNELS` plus a stub
macro, and a `@pprop` field is filled from the same-named field of the prior result:
`sel(getfield(x, :f), getproperty(pr, :f))`.

A calibration rule fits nothing and reads no same-named field. It reads the sample length, the
moments and the effective observation weights that the prior result already carries, it may read the
effective solver, and it returns **one number** that the slot owner's own constructor must then
validate.

## Decision

### The channel is parallel, and a role stays out of the `DeferredQuantity` union

Four verbs are new. Each names its counterpart in
[`src/19_RiskMeasures/01_Base_RiskMeasures.jl`](../../src/19_RiskMeasures/01_Base_RiskMeasures.jl),
and the calibration half of each pair lives in
[`src/14_UncertaintySets/06_CalibrationRules.jl`](../../src/14_UncertaintySets/06_CalibrationRules.jl).

| Calibration | Deferred Quantity | What it does |
| :--- | :--- | :--- |
| `resolve_calibration_slot(slot, key, pr, w, slv = nothing)` | `resolve_slot(slot, key, pr)` | resolve one slot |
| `calibration_slots(x)` | `deferred_slots(x)` | declare the slots, as a `NamedTuple` |
| `assert_calibrated_slots(x)` | `assert_resolved_slots(x)` | refuse at a value-level entry point |
| `assert_declared_calibration_resolver(x, slots)` | `assert_declared_slot_resolver(x, slots)` | refuse a declared slot that no resolver reaches |

`resolve_calibration_slot` runs a rule by **calling** it, `r.alg(key, pr, w, slv)`, and its fallback
method returns a stated number unchanged. Three facts make it a separate verb rather than a widening
of `resolve_slot`:

1. **Fit-then-extract does not describe a rule.** There is no fit to hold, and `key` would select a
   quantity of a fit that never happened.
2. **`resolve_slot` carries neither the observation weights nor the solver**, and a rule reads both.
   `ScenarioCount` reads Kish's effective sample size off `w`, and a rule may call `ERM` or `RRM`.
3. **A refusal must name the mechanism the caller wrote.** `assert_calibrated_slots` tells a caller
   who reached `expected_risk(r, w, X)` that the slot holds a Calibration Role and that a bare
   returns matrix carries no sample for the rule to read. The Deferred-Quantity message names a fit
   instead, which is the wrong instruction.

**The declaration is paired with its resolver.** `calibration_slots` and the resolution beside it
are two statements, and a type that writes the first and forgets the second hands a role to the
model builders. `assert_declared_calibration_resolver` refuses that, and it is the fourth pair
because the Deferred-Quantity channel already refuses the same failure. It runs where a resolved
measure meets a consumer that cannot resolve: `set_risk_constraints!` on the `JuMP` route, and the
three regularisation factories. `assert_calibrated_slots` covers the value-level route already, so
each route now names the failure in its own words.

**The per-type entry point is shared.** A type's own `resolve_deferred_quantities` method resolves
both kinds of slot and rebuilds once, because a measure that carries both must not be rebuilt twice
and because the two resolutions have an order between them. Only the declaration and the resolver
are parallel, and ADR 0051's statement that the entry point resolves the deferred state and nothing
else is amended there.

### A tag row was refused

ADR 0061 makes a channel cheap: a row of `PROP_TAG_CHANNELS`, a stub macro, and every
`@propagatable` type gains the method. It was considered and not taken, so the two mechanisms stay
separate end to end.

A tag row emits **one generated method per struct** that rewrites each tagged field from a source of
the channel's choosing. A calibration slot has no such source. Its value comes from calling a rule
the caller put in the slot, and the order in which two slots of one type resolve is a property of
the rules rather than of the fields: a deformation rule reads the significance level of a sibling
slot, so `alpha` must resolve before `kappa`. A generated method cannot know that order, and a
per-type method is exactly what states it. That is the reason ADR 0051 already gives for writing
`resolve_deferred_quantities` per type rather than deriving it from a field walk.

### The role names the quantity, and the bound is the whole validation

A **Calibration Rule** computes a number. A **Calibration Role** places that rule in the slot of one
quantity and names the quantity. Seven roles ship, over five rule families:

| Role | Rule family | Slots it stands in |
| :--- | :--- | :--- |
| `SignificanceTailCalibration` | `AbstractSignificanceCalibrationAlgorithm` | `alpha` |
| `SignificanceHeadCalibration` | `AbstractSignificanceCalibrationAlgorithm` | `beta` |
| `DeformationTailCalibration` | `AbstractDeformationCalibrationAlgorithm` | `kappa`, `kappa_a` |
| `DeformationHeadCalibration` | `AbstractDeformationCalibrationAlgorithm` | `kappa_b` |
| `AmbiguityRadiusCalibration` | `AbstractAmbiguityRadiusCalibrationAlgorithm` | `r`, `r_a`, `r_b`, `val`, `l1`, `linf` |
| `AmbiguityTailWeightCalibration` | `AbstractAmbiguityTailWeightCalibrationAlgorithm` | `l`, `l_a`, `l_b` |
| `NormCeilingCalibration` | `AbstractNormCeilingCalibrationAlgorithm` | `l2c`, `linfc`, `val` |

[#593](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/593) split the root in two: a
role is an Estimator under `AbstractCalibrationEstimator`, and only a rule is an Algorithm under
`AbstractCalibrationAlgorithm`. Two bound families follow from that split, and together they are the
whole of the validation.

- A **slot** bound is `Num_SigTailCal`, `Num_SigHeadCal`, `Num_DefTailCal`, `Num_DefHeadCal`,
  `Num_AmbRadCal`, `Num_AmbTwtCal` or `Num_NormCeilCal`, each pairing `Number` with **one** concrete
  role. A head role in a tail slot, or an ambiguity role in a significance slot, is refused **at
  construction**.
- An **`alg`** bound is `Func_SigCal`, `Func_DefCal`, `Func_AmbRadCal`, `Func_AmbTwtCal` or
  `Func_NormCeilCal`, each pairing `Function` with one rule family. No role subtypes a rule family,
  so a role nested inside another role's `alg` is refused by the same route.

Refusal by the bound is **earlier than any `assert_` method could be**: it fires where the caller
wrote the mistake, not at the fold where the value is read. So no guard method is written for either
mismatch, and neither refusal has a message that must be kept in step with a bound.

**One bound names two roles, and two `assert_` methods part them.** `LpRegularisation.val` is a
penalty coefficient in `JuMPOptimiser.lp` and a norm ceiling in `JuMPOptimiser.lpc`, so the field
cannot name its reading from the type alone. `Num_AmbRadNormCeilCal` admits both roles, and
`assert_penalty_coefficient_role` and `assert_norm_ceiling_role` settle the reading per field. Both
run in `JuMPOptimiser`'s own constructor, so the refusal still fires where the caller wrote the
field. ADR 0097 carries that decision, and every other bound names one role.

### No ordering guard, and no reordering

`OrderedWeightsArrayTailGini` and its Range twin carry the joint check
`@argcheck(0 < alpha_i < alpha < 1, …)` in the constructor. Only `alpha` widens — `alpha_i` and
`beta_i` are the starting points of the inner tail-Gini integration and not quantities to estimate —
and every rebuild goes through the ordinary keyword constructor. So a rule whose number reaches or
crosses the stated `alpha_i` is refused **by the check that already exists**, at fold time, and the
docstring says so.

Nothing silently corrects such a pair. Reordering a caller's stated `alpha_i` to accommodate a
calibrated `alpha` answers a question the caller did not ask, and the pair carries no information
about which of the two the caller meant to hold still.

### A rule is a callable, and it gets the solver on both routes

A rule is run by calling it, so a callable Estimator and a plain `Function` of `(key, pr, w, slv)`
are the same thing to the resolver. Every `alg` bound admits a `Function`, and there is no
`calibrate` verb. Eleven rules ship, over the five families:

| Rule family | Rules |
| :--- | :--- |
| `AbstractSignificanceCalibrationAlgorithm` | `ScenarioCount`, `RateSignificance` |
| `AbstractDeformationCalibrationAlgorithm` | `EntropyBudget`, `HillTailDecay`, `RadialTailDecay` |
| `AbstractAmbiguityRadiusCalibrationAlgorithm` | `ConcentrationRadius`, `RateRadius`, `DimensionalRateRadius`, `DualNormRadius` |
| `AbstractAmbiguityTailWeightCalibrationAlgorithm` | `TailTermParity` |
| `AbstractNormCeilingCalibrationAlgorithm` | `EffectiveAssetFloor` |

A rule sees the effective solver on both of the routes that resolve a measure. On the `factory`
route `@propagatable` runs every selection before the resolution, so the solver is already on the
struct; ADR 0096 records that ordering. On the `JuMP` route no selection runs, so
`set_risk_constraints!` threads `opt.opt.slv` into `resolve_deferred_quantities` and the owner
settles it as `sel(x.slv, slv)`
([#591](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/591)). A rule may therefore call
`ERM` or `RRM`, which the map originally ruled out.

A rule gets **no portfolio**. A prior result carries no weight vector, so the "the `alpha` whose CVaR
meets a target loss" candidate stays refused.

### A slot key names no quantity, so the owner hands its series over

`RelativisticValueatRisk` and `RelativisticDrawdownatRisk` both resolve the key `:kappa`, and the two
price two different series. A rule that reads the **shape** of a series therefore cannot tell from
the key which quantity it stands in front of. `calibration_series(x)` is the trait each owner
answers, and `bind_series(slot, series)` carries the answer into the rule. The owner's series
**overwrites** one the rule carries: the quantity belongs to the measure, and a rule cannot know
which measure it reached.

`AbstractCalibrationSeries` names three markers: `ReturnsSeries`, and `AbsoluteDrawdownSeries` and
`RelativeDrawdownSeries` under `AbstractDrawdownSeries`. Six rules carry a `series` field —
`HillTailDecay`, `RadialTailDecay`, `TailTermParity`, `ConcentrationRadius`, `DimensionalRateRadius`
and `DualNormRadius` — and the marker moves the sample each of them reads.

**The reading does not move, only the sample it reads.** `HillTailDecay` pools the drawdown series of
each column in place of the columns, with the same standardisation, the same count and the same
estimator. `RadialTailDecay` whitens the rows of the drawdown sample in place of the rows of `pr.X`.
`TailTermParity` substitutes `calibration_series_matrix(series, pr.X)` for `pr.X` and nothing else,
because the `ConditionalValueatRisk` kernel over a non-positive drawdown column **is** the
`ConditionalDrawdownatRisk` of that column, so the rule still carries no second encoding of the
measure it calibrates. No rule forms a portfolio, and none needs one: a drawdown series is formed per
column, and it holds one entry per observation, so every count a rule forms on `pr.X` is the count it
reads.

**A drawdown sample carries its own moments.** `pr.mu` and `pr.sigma` are moments of the returns, and
no scaling of them states the moments of a drawdown. So under a drawdown marker `RadialTailDecay`
centres on the column means of the drawdown sample and whitens by the Cholesky factor of its
covariance matrix, through `radial_series_inputs`, and the three radius rules read their per-asset
dispersion through `calibration_series_dispersion`: `sqrt.(diag(pr.sigma))` on a returns series, and
the sample dispersion of the drawdown columns on a drawdown marker. The precedence of `pr.chol` over
`pr.sigma`, which [#612](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/612) records,
therefore governs the returns reading alone.

**The ambiguity families take the marker too, and the programme decides why.** A radius is the
coefficient of a norm penalty on the weight vector, so whether an Esfahani-Kuhn radius under a
drawdown owner belongs on the asset-return scale or on a drawdown scale is not a matter of taste.
`set_risk_constraints!` for `DistributionallyRobustConditionalDrawdownatRisk` answers it. That method
measures the transport cost of its own programme against
`set_portfolio_drawdowns_plus_one!(model, pr.X)`, which is `absolute_drawdown_arr(X) .+ 1`, and that
matrix is `calibration_series_matrix` under `AbsoluteDrawdownSeries` shifted by the support offset.
So the scenarios the ball is drawn around are the **per-asset drawdowns**, the radius is a distance
between two such vectors, and it carries drawdown units.

**The ground metric does not move with the series.** `DualNormRadius` reads `key` for the ground
metric and `series` for the vector it takes that norm of. The two are independent: `:r` is the 1-norm
under every marker, and only the error vector moves.

**A drawdown series has one end.** It is non-positive, so `:kappa_b` names nothing on it and
`series_end_sign` refuses that key under `AbstractDrawdownSeries`. No drawdown Range measure ships,
so only a caller who runs a rule by hand reaches the refusal.

**The drawdown error scale is a floor, and a rule says so rather than correcting it.** A drawdown is
a running functional, so its entries are dependent down a column and `s / sqrt(T_e)` prices a record
of independent draws that the sample does not hold. A correction needs a model of that dependence,
and the sample states none. This is the same refusal `DualNormRadius` already makes for the number of
assets.

**Five rules read no series.** `EntropyBudget` reads the sample length and its sibling `alpha`, and
neither moves with the series, so the identity default of `bind_series` serves it. `ScenarioCount`,
`RateSignificance`, `RateRadius` and `EffectiveAssetFloor` need no method either.

### Four verbs carry what no derivation can find

- **`mirror_role(x)`** is the default of the head slot on every Range type. A number crosses
  unchanged and a tail role crosses as the head role of the same family holding the same `alg`, so
  a rule stated on one end reaches both and no stated number moves. The two ordered-weights Range
  types already read `beta = alpha`, and the widening kept that default alive. The six that held a
  literal of their own now read the same default, which is the same number at the default
  arguments and the tail slot's occupant otherwise. `RelativisticValueatRiskRange` is what gives
  the deformation method a caller: its gain-side pair defaults to its loss-side pair, both halves.
- **`bind_alpha(slot, alpha)`** carries a **travelling pair**. `EntropyBudget` reads its sibling
  `alpha`, and `resolve_calibration_slot` carries a `Symbol` and no number, so the number travels
  through the rule itself: the owner resolves `alpha`, calls `bind_alpha` on the `kappa` slot, and
  resolves the result. `TailTermParity` takes the same pair, because its tail-term scale is a CVaR at
  the slot owner's own significance level. The default is the identity, and the significance and
  radius families need no method because no rule of either reads a sibling.
- **`bind_norm_order(slot, p)`** carries a constraint's norm order into the rule that computes its
  ceiling, on the shape `bind_alpha` gives. It differs in one respect: an order is a property of the
  constraint rather than of a sibling slot, so the constraint site's order overwrites one the rule
  already carries.
- **`bind_series(slot, series)`** carries the owner's series marker, and overwrites one the rule
  carries, for the reason `bind_norm_order` does.

Every method of the three `bind_` verbs preserves every other field of the slot it rebuilds, so the
rebuilds compose in either order and a call site may nest them freely. `mirror_role` is the one that
changes the type, and it does so by design: it carries the `alg` across and nothing else.

### A schedule reaches the host, and no further

A `TimeDependent` wrapping a rule is refused, and no channel is missing.

**A rule is never standalone.** It stands in a slot of a host, so the host is the thing a schedule
swaps, and the host already carries the channel. Where the host is a `JuMPOptimiser`, the four norm
fields are themselves schedulable, and a schedule over one of them selects a rule per fold. Where the
host is a risk measure, the slot's own bound admits no schedule, and the caller varies the whole
measure instead, through the schedulable risk-measure field of the optimiser.

**The two run at two points of the pipeline, and neither knows about the other.** The selection runs
in `update_time_dependent_fields`, before any prior is fitted. The resolution runs at assembly,
against the prior of the period that was selected. So a schedule and a rule compose, and the order
falls out of the pipeline rather than being invented for it.

So the `Num_` and `Func_` bounds of the seven roles stay free of `TimeDependent`. A generic
resolution is possible in principle: nothing in the mechanism stops a schedule from being resolved
wherever it stands. It buys no reading that the host's own channel does not already give, so it is
not built.

## Rejected alternatives

**Widening `DeferredQuantity` to admit the roles.** One union, one resolver, one declaration verb.
Rejected because `resolve_slot` would then need a branch on what it holds, two more arguments that
half its domain ignores, and a refusal message that names a fit for a value that is never fitted.
The two mechanisms would read as one and behave as two.

**A `PROP_TAG_CHANNELS` row plus a stub macro.** The cheapest possible wiring, and ADR 0061 exists to
make it cheap. Rejected for the reason above: a generated method rewrites a field from a source, and
a calibration slot has a rule and an order instead of a source.

**One role type per family, with a field naming the end.** `SignificanceCalibration(; end = :tail)`
halves the type count. Rejected because the end would then be data rather than type, so a tail rule
in a head slot could only be refused by a guard method at resolution time, which is later and needs
a message.

**A guard method per slot, with every slot bound at `Union{Number, AbstractCalibrationEstimator}`.**
Rejected on the same grounds. The bound already refuses every mismatch it can see, and a guard would
duplicate the refusal without widening what is caught.

**An ordering guard on the calibrated pair, or a silent reorder.** Rejected: the joint `@argcheck` is
the whole validation and it already runs on the rebuilt struct, and a reorder invents an intent.

**A stated `scale` field in place of the series marker.** `DimensionalRateRadius` reads
`mean(sqrt, diag(pr.sigma))` on the same terms as `ConcentrationRadius`, and its `scale` field was
the workaround a drawdown slot needed. Rejected once the programme settled the units: a caller who
must state a scale is stating what the model already knows. A stated `scale` survives as the way to
price a ball whose units are neither.

**A schedule inside a rule.** Rejected: it would name a fold the rule cannot see, and it would give a
second channel for what the host already varies.

## Consequences

- **Thirty-two types declare `calibration_slots`**, across the six `XatRisk` and ordered-weights
  files of `src/19_RiskMeasures/` and the two regularisation estimators. A container declares the
  child it wraps rather than a quantity, which is how `reverse ∘ owa_tg` reaches its inner builder.
- **`JuMPOptimiser` declares none.** Its `l1`, `linf`, `l2c` and `linfc` resolve at their constraint
  site, through `resolve_calibration_slot(opt.l1, :l1, pr, pr.w, opt.slv)` and its siblings. The
  optimiser has no value-level entry point, so `assert_calibrated_slots` has nothing to say about it.
- **`resolve_deferred_quantities` is no longer only about Deferred Quantities.** The name is now
  narrower than the method, and a reader who takes it literally will miss the calibration resolution
  beside it. ADR 0051 carries the amendment.
- **The root did not move under `src/01_Base.jl`, and the two uncertainty families were not
  re-parented.** `AbstractCalibrationAlgorithm` lives in
  `src/14_UncertaintySets/06_CalibrationRules.jl`, so ADR 0070's re-parenting of
  `AbstractUncertaintyKAlgorithm` and `AbstractUncertaintyEpsAlgorithm` has not shipped: both
  still subtype `AbstractAlgorithm` directly. Re-parenting them needs the root in
  `src/01_Base.jl` first.
- **A field bound is enforced on the keyword route only.** `ConcreteStructs.@concrete` emits a
  positional constructor that is strictly broader than the hand-written inner one, so a positional
  call bypasses every bound this decision rests on. That hole reaches every `@concrete` type in the
  library and is not created here;
  [#264](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/264) carries it, together with
  the measurement that made a parametric constraint too expensive to ship.
- **Three inner slots keep `::Number`.** `alpha_i` and `beta_i` on the two tail-Gini types are
  starting points, not estimates. A reading under which an inner starting point is itself calibrated
  would have to say what the joint check means when both sides move, and nothing has examined it.
- **A `TD_` wrapper holding a rule is the host's own channel.** `JuMPOptimiser.l1`, `.linf`, `.l2c`
  and `.linfc` are `TD_Option{<:Num_AmbRadCal}` or `TD_Option{<:Num_NormCeilCal}`, so one field
  carries two deferral channels and ADR 0030 never considered a second. The two resolve at two points
  of the pipeline, so they compose rather than compete.
- **One asymmetry in the dispersion reading is left open, and it is filed.** Under a returns marker
  the dispersion comes off `pr.sigma`, so a shrunk or a robust covariance reaches it. Under a
  drawdown marker it comes off `Statistics.std` of the drawdown sample, so the caller's own estimator
  does not. `radial_series_inputs` carries the same asymmetry. Closing it needs a rule that holds a
  **prior estimator** and fits it to the drawdown sample, which is a design the maintainer has raised
  and which this ADR does not decide.
