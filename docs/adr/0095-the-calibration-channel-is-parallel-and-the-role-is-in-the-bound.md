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

Five verbs are new. Each names its counterpart in
[`src/19_RiskMeasures/01_Base_RiskMeasures.jl`](../../src/19_RiskMeasures/01_Base_RiskMeasures.jl),
and the calibration half of each pair lives in
[`src/14_UncertaintySets/06_CalibrationRules.jl`](../../src/14_UncertaintySets/06_CalibrationRules.jl).

| Calibration | Deferred Quantity | What it does |
| :--- | :--- | :--- |
| `resolve_calibration_slot(slot, key, pr, w, slv = nothing, ctx = CalibrationContext())` | `resolve_slot(slot, key, pr)` | resolve one slot |
| `calibration_slots(x)` | `deferred_slots(x)` | declare the slots, as a `NamedTuple` |
| `resolve_calibration_slots(x, pr, slv = nothing)` | `resolve_deferred_quantities(x, pr, slv = nothing)` | derive the resolution from the declaration |
| `assert_calibrated_slots(x)` | `assert_resolved_slots(x)` | refuse at a value-level entry point |
| `assert_declared_calibration_resolver(x, slots)` | `assert_declared_slot_resolver(x, slots)` | refuse a declared slot that no resolver reaches |

`resolve_calibration_slot` runs a rule by **calling** it, `r.alg(key, pr, w, slv, ctx)`, and its fallback
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

**The entry point is shared.** `resolve_deferred_quantities` resolves both kinds of slot, and
ADR 0051's statement that it resolves the deferred state and nothing else is amended there. The
deferred step runs first and the calibration step runs on its result, which is what lets a container
name one key in both declarations: the child resolves through the deferred recursion, and the
calibration walk then sees a resolved child and rebuilds nothing.

**The resolution is derived from the declaration.** For most types the declaration is the whole
statement. `resolve_calibration_slots` reads `calibration_slots`, settles the effective observation
weights and the effective solver off the two fields the library names everywhere, resolves each slot
under its own name, and rebuilds. Eighteen of the twenty-seven calibrated measures write nothing at
all, and a nineteenth added tomorrow writes nothing either.

Two readings put a type outside the derivation, and nothing else does.

1. **An order between the slots.** A `CalibrationContext` built from a sibling's resolved number is
   what such an order looks like: the slot reads that number off the context. A derivation cannot
   know which sibling, so those types write their own method. The travelling pair below is the whole
   of that set.
2. **A key that is not the field's name.** The three regularisation keys are three quantities under
   one field, which ADR 0097 settles, so a derivation that reads the field name would hand the rule
   the wrong key. Those two terms declare a `resolve_calibration_slots` method returning `x`, and
   their own factories stay the one route.

### A tag row was refused

ADR 0061 makes a channel cheap: a row of `PROP_TAG_CHANNELS`, a stub macro, and every
`@propagatable` type gains the method. It was considered and not taken, so the two mechanisms stay
separate end to end.

A tag row emits **one generated method per struct** that rewrites each tagged field from a source of
the channel's choosing. A calibration slot has no such source. Its value comes from calling a rule
the caller put in the slot, and a tag row would emit the method whether or not the type needed one.

`resolve_calibration_slots` carries the same saving without the row. It derives the resolution from
the declaration the type already writes, it is one method rather than one per struct, and a type
that needs an order of its own overrides it. Where a tag row would have to state the order it cannot
know, a method is simply shadowed by the more specific one beside the declaration.

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

A rule is run by calling it, so a callable Estimator and a plain `Function` of `(key, pr, w, slv, ctx)`
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

### A rule is named for its method, and it constructs bare where it can

An ergonomics pass read the eleven names as one family and found a convention in four of them:
`ConcentrationRadius`, `RateRadius`, `DimensionalRateRadius` and `DualNormRadius` each end in the
quantity the rule computes, and six of the remaining seven do not. Read that way the family is
inconsistent in six places. The reading is the wrong one, and ADR 0015 is the Authority that says so:
a name is the bare concept word, and a role suffix is added only to earn back clarity the bare word
would lose, never as a blanket category marker. A quantity on every rule is that marker, and it
spells `EffectiveAssetFloorNormCeiling`.

**A rule is named for the method it runs.** It carries the name of the quantity only where the bare
method word is already claimed. Six rules name a method and stop there: `ScenarioCount`,
`EntropyBudget`, `HillTailDecay`, `RadialTailDecay`, `TailTermParity` and `EffectiveAssetFloor`.
Five carry the quantity, and each of the five earns it.

| Rule | What claims the bare word |
| :--- | :--- |
| `RateSignificance` | `Rate` is one method over two quantities. This rule and `RateRadius` share the closed form `c` over the square root of the sample length, so neither may hold the word. |
| `RateRadius` | The same collision, read from the other end. |
| `DimensionalRateRadius` | The same `Rate` stem under a prefix, and a prefix does not free the stem. |
| `ConcentrationRadius` | `Concentration` names the weight concentration of a portfolio in this library's own prose, so the bare word would read as a property of the portfolio rather than as a ball read off a concentration inequality. |
| `DualNormRadius` | `DualNorm` names a mathematical object, and the library already names norms in `LpRegularisation` and in the three norm-ceiling slots, so the bare word would read as the object rather than as the rule. |

The four ambiguity radius rules carry the suffix for four separate reasons, and one cause stands
behind all four: a radius is a distance, and each method that computes one is named after a standard
mathematical construct whose word already names that construct. The six bare rules are named by a
descriptive phrase that reads as a rule on its own. `EffectiveAssetFloor` is the sharpest case of the
reading, and it is correct under it. The rule's method **is** a floor on the effective number of
assets, the norm ceiling is what that floor converts into, and ADR 0097 is where the two are held
apart.

`test_09g_calibration_rules.jl` gates the list. A rule whose name ends in the word for a quantity of
this channel and stands outside the table reds the census, and so does a name in the table that no
rule carries.

**A rule constructs bare where it can.** Nine of the eleven state a default for every keyword, so a
bare call constructs and a caller reads the rule's shape before choosing a value. `ScenarioCount`
and `EntropyBudget` state none, because the keyword each takes is the whole content of the rule: a
scenario count that suits every sample does not exist, and the band an entropy budget must land in
moves with the sample. Both keep the keyword mandatory rather than invent a value a caller would
inherit without reading it.

Neither refusal is left to `UndefKeywordError`, which names the keyword and nothing else. The keyword
of each of the two stands at `nothing` and the constructor refuses that with an `@argcheck`, so the
message names the quantity, the reason there is no default, a value to start from, and the rule of
the same family that does construct bare. The bound is not weakened by the sentinel:
`Option{<:Number}` admits a number and `nothing`, so `ScenarioCount(; n = "a")` still raises
`TypeError` where it always did, and the field itself never holds `nothing`, because the inner
constructor takes `n::Number`.

A zero-argument method carrying the message was written first and does not work. Julia generates the
zero-argument method of a keyword constructor itself, a second definition of it overwrites the
generated one, and **method overwriting is an error during precompilation**. The sentinel is the
shape that survives the load, and it is the shape `dual_norm_radius_scale` already uses for the
sibling failure: a site that states no norm order reaches the same `@argcheck` with the same kind of
message.

### A slot key names no quantity, so the owner hands its series over

`RelativisticValueatRisk` and `RelativisticDrawdownatRisk` both resolve the key `:kappa`, and the two
price two different series. A rule that reads the **shape** of a series therefore cannot tell from
the key which quantity it stands in front of. `calibration_series(x)` is the trait each owner
answers, and the `series` field of the `CalibrationContext` carries the answer into the rule. **No
rule holds a series of its own**: the quantity belongs to the measure, a rule cannot know which
measure it reached, so there is nothing for the owner's answer to overwrite.

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
neither moves with the series, so it never reads `ctx.series`. `ScenarioCount`,
`RateSignificance`, `RateRadius` and `EffectiveAssetFloor` need no method either.

### Two carriers hold what no derivation can find

- **`mirror_role(x)`** is the default of the head slot on every Range type. A number crosses
  unchanged and a tail role crosses as the head role of the same family holding the same `alg`, so
  a rule stated on one end reaches both and no stated number moves. The two ordered-weights Range
  types already read `beta = alpha`, and the widening kept that default alive. The six that held a
  literal of their own now read the same default, which is the same number at the default
  arguments and the tail slot's occupant otherwise. `RelativisticValueatRiskRange` is what gives
  the deformation method a caller: its gain-side pair defaults to its loss-side pair, both halves.
- **`CalibrationContext(; alpha, series, p)`** is the second, and it carries everything the site
  knows that `key` does not. `resolve_calibration_slot` takes it as a sixth argument and hands it
  to the rule, which is called as `alg(key, pr, w, slv, ctx)`.
  - `alpha` is a **travelling pair**. `EntropyBudget` reads its sibling `alpha`, so the owner
    resolves `alpha` first and states it in the context of the `kappa` slot. `TailTermParity` takes
    the same pair, because its tail-term scale is a CVaR at the slot owner's own significance
    level. The significance and radius families never read it, because no rule of either reads a
    sibling.
  - `series` is the owner's marker, which `calibration_series(x)` answers.
  - `p` is the norm order of the constraint or of the penalty the quantity stands in.

**No rule holds a field for any of the three.** Each belongs to the site, and a rule cannot know
which site it reached, so there is no value on the rule for the site to overwrite, no precedence
between the two to state, and no occupant to rebuild on the way in. A caller who runs a rule
outside a measure builds the context the site would have built. `mirror_role` is therefore the one
verb that changes a value, and it does so by design: it carries the `alg` across and nothing
else.

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
a calibration slot has a rule instead of a source. `resolve_calibration_slots` gives the same saving
from one method, and a type that needs its own order shadows it.

**One hand-written resolution per calibrated type.** The shape the channel shipped in, and the shape
ADR 0051 gives the Deferred-Quantity side. Rejected once the numbers were counted: eighteen of the
twenty-seven bodies stated the slot list a second and a third time, beside the declaration that
already held it, and none of them stated an order. A field added to such a type and missed at one of
the three sites moved the number the measure priced and broke no test.

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
