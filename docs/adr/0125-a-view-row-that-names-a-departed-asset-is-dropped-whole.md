---
status: accepted
---

# A view row that names a departed asset is dropped whole, and a group sheds its departed members

## Context

[ADR 0115](0115-every-optimisation-estimator-reduces-once-at-its-entry-and-its-result-carries-the-investable-mask.md)
reduces every optimisation to the Investable Mask at its entry, and
[ADR 0124](0124-a-door-mints-a-non-investable-axis-and-every-view-drops-it.md) gave the door a way
to say who left: it mints the **Non-Investable Axis** on the sets it hands the problem, so a
name-keyed estimator resolving after the door can tell a departed asset from a typo and skip it in
silence rather than refuse it.

That fixed the **value-keyed** family alone. `name_to_val!` and `estimator_to_val` take the
counterpart axis, and `test_50_investable_reduction.jl` pins a weight bound, a threshold, a
turnover rate and a risk budget resolving under `strict = true` on a name the mask dropped. The
**row** family was not touched. `get_linear_constraints`
(`12_ConstraintGeneration/02_LinearConstraintGeneration.jl`) takes no counterpart axis, so `lcse`,
`gcarde` and `sgcarde` still report a departed name through `strict_diagnostic` — a warning under
`strict = false`, an `ArgumentError` under `strict = true`. ADR 0124's consequence, that such a
name "no longer refuses under `strict = true`, in any family, at any door", is therefore true of
the value-keyed family and not of the row family; it is corrected in place there, this ADR being
what makes it true.

Ticket [#919](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/919) then measured what
a departure does to the nine **wrapping priors** — the four Black–Litterman estimators, the two
entropy pooling estimators, opinion pooling and the high-order factor prior. One mechanism explains
all of them: a dense linear form over the full asset axis multiplies the departed asset's `NaN` by a
structural zero, and `0 * NaN = NaN`, so a view naming only *live* assets is poisoned as thoroughly
as one naming the departed asset. Black–Litterman is silently wrong — every entry of its posterior
comes back `NaN`, so the Investable Mask of its result is empty and the failure surfaces one layer
later naming the wrong cause — and seven of the others refuse with a message naming LAPACK, a bounds
check or a solver.

The reduction that repairs this is the optimisers' own, and #919 proved it: `port_opt_view` of the
wrapped carrier at `findall(investable_mask(pr))` reproduces the hand-reduced oracle bit-exactly,
for the low-order and the high-order carrier alike, and the whole Black–Litterman path over it
matches at `0.0`. Build tickets [#921](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/921)
and [#922](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/922) carry it. **What they
cannot proceed without is what the reduced view builder then does with the name of an asset that is
no longer on its axis**, and that is what this ADR decides.

A view is not a bound, and the difference is what makes the answer its own decision. A bound, a
threshold or a rate is keyed by one asset: when that asset leaves, the entry has nothing left to
apply to and dropping it loses nothing. A view is a **row** — a joint statement over several
assets, with one right-hand side. `"a + c == 0.05"` says something about `a` and `c` together.
Dropping `c`'s term leaves `a == 0.05`, which is a different and stronger claim, and one the caller
never wrote. The row is the unit the caller authored, so the row is the unit that must survive or go.

## Decision

### A name on the Non-Investable Axis drops the whole row, and the door says so once

A view row naming an asset the Investable Mask left out is dropped entire — the term is not struck
from the row and the row is not fitted without it. The dropped row joins `excl`, which
`get_black_litterman_views` already maintains, so `remove_excl_views` drops the matching entry of a
per-view confidence vector and the surviving rows keep their order and their confidences.

The drop never **refuses**, under either setting of `strict`, and this is the same reasoning ADR
0124 gave for the value-keyed family: `strict` exists to catch a caller's typo, a departed name is
the opposite of a typo — it was correct over the universe the caller was handed, and the data moved
it — and no caller can foresee which asset a prior will fail to estimate. Refusing reports something
nobody can act on.

It is not, however, **silent**. A caller who wrote three views and got one fitted has to be able to
learn it, and the departed asset's name alone does not say what its leaving cost. So the drop site
records what it dropped into a **ledger** the door owns, and the door reads the ledger once and
reports both together: who left, and which rows and group members went with them. `strict` still
governs the *typo*, which is reported where it happens, as it always was.

The unit of reporting is the **door**, not the drop. A view set is resolved once per fit and a fit
runs once per window, so reporting at each site would say the same thing once per row per window of
a walk-forward — a walk-forward over 250 windows with three view rows would print 750 lines of news
that happened once. One line per fit is the same information at the granularity a reader can use.

`record_non_investable_drop!` is the ledger's only writer and takes `nothing` for "nobody is
collecting", so a caller who assembles constraints outside a door pays nothing for a ledger it does
not keep, and the branch is dispatch rather than a condition.

**A name on neither axis keeps today's behaviour**, and that is the whole of the distinction: it is
reported through `strict_diagnostic`, so it refuses under `strict = true` and warns otherwise. What
changes for it is only the unit — under `strict = false` the row it appears in is now dropped whole
rather than fitted without its term, because a row the builder cannot assemble as written is a row
the caller did not ask for either way.

### A group sheds its departed members before its coefficient is spread

`port_opt_view(::UniverseSets, i)` slices the `xkey`- and `uxkey`-prefixed entries and carries every
**plain** group through bit-identical, so a group still names the assets that left. And
`replace_group_by_assets` spreads the coefficient at expansion time, before any name is resolved:
the Black–Litterman expansion divides by the member count `k`, so striking the departed member's
term afterwards would leave `k - 1` legs of `c/k` against an unchanged right-hand side — a row
asserting a mean it no longer computes.

So the shed happens **first**. `replace_group_by_assets` strikes the members that name the
Non-Investable Axis, and only then spreads the coefficient: the Black–Litterman mean divides by the
**surviving** count, and the entropy pooling sum runs over the survivors. The row survives, and the
shed goes into the door's ledger, which counts it rather than naming the member — the departed
assets are named once, by the door, and repeating them per group would make the report longer than
what it reports. A pair group of a correlation view sheds **jointly**: a position survives only when
both of its names did, which is both what a pair means and what keeps the two lists the same length.

This is not the rule above contradicted but the rule above applied. A group is a **description the
data resolves**, not a term the caller chose: `"tech"` means the technology assets of this problem,
and when one of them delists the description still names the rest. A name written out in the
equation is the caller pointing at one asset, and pointing at an asset that is not there is what
takes the row with it. A group that loses **every** member contributes nothing, and the row is then
dropped by the existing empty-row path.

### The prior's entry mints the axis, and the row builders learn the counterpart

The wrapping prior does at its entry exactly what an optimisation door does at its own: derive
`imsk = investable_mask(pr)` from the wrapped fit, take `port_opt_view` of the carrier and of
`pe.sets` at `findall(imsk)`, and declare the departed names with `non_investable_sets`. The mint is
the existing verb, unchanged, and `investable_mask` returning `nothing` when every asset is
investable keeps the gap-free path at its current cost.

`get_black_litterman_views`, `get_linear_constraints` and `replace_group_by_assets` learn the
counterpart axis the way `estimator_to_val` already does: they **read it off the sets they were
handed**, with `counterpart_axis_names`, rather than take it as an argument. They hold `sets` and
the key already, so an argument would be a second way to say what `UniverseSets` says, over the same
data, and one a caller could forget to pass. One mechanism therefore serves the priors and the
optimisation side alike, and the `lcse`, `gcarde` and `sgcarde` gap named in the Context closes with
it rather than in a separate repair.

The prior **does** announce, naming itself. `announce_non_investable` therefore takes the `process`
it is speaking for — `"optimisation"` at an optimiser's door, `"entropy pooling fit"` at a wrapping
prior's — and the `consequence` that door's departure carries, because a forced-liquidation carrier
is priced by an optimisation and by nothing else. Hard-coding `"optimisation"` made the message
wrong for every door but the first, and left a standalone `prior(pe, X)` call saying nothing at all.

A prior fitted *inside* an optimisation therefore reports twice, and the two are not duplicates:
the prior's names the view rows its departures cost, and the door's names the constraints and the
liquidation carriers. Each is the news of its own layer, and neither can state the other's.

The mint is local to the fit. A Prior Result carries no sets, so the axis does not travel out of the
prior, and `port_opt_view` drops it, so an optimiser that reduces afterwards mints its own. ADR
0124's invariant — a `UniverseSets` carrying the axis was reduced by exactly one door, for exactly
one problem — is unchanged, with the wrapping prior's own entry now among the doors.

**Where the report is emitted.** At the *end* of the fit, not at the reduction. A staged entropy
pooling algorithm interleaves its view builders with its solves, so the ledger is only complete once
the last stage has stated its views; reporting at the reduction would report who left and nothing
about what it cost, and reporting per stage would be three messages for one departure.

### A view set with no row left is a fit with no view, whatever emptied it

A view set whose every row has been dropped leaves nothing to assemble, and
[#852](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/852) settled what that means
today: `get_black_litterman_views` returns `nothing`, and `bl_preroll` turns it into a named error.
Under this ADR a delisting can empty the set, and refusing then would put back exactly the
unforeseeable refusal every rule above removes.

A delisting is not the only way, and that is what decides the rule. The row-whole unit above means a
view set empties whenever **every** row carries at least one name the builder cannot resolve, and
three quite different things produce that: a departure, a **sub-universe** — a cluster of a nested
optimisation, a subset of a resampling — in which the caller's names sit outside the axis this
particular fit was handed, and a typo. The second is the one that made the refusal untenable. A
`NestedClustered` fit takes `port_opt_view` of its estimator per cluster, so a view naming an asset
in another cluster is neither a departure nor a mistake; it is a correct name over the universe the
caller stated it against, met by a fit that only holds part of that universe.

So `bl_preroll` answers `nothing` whenever no row survived, and the fit proceeds with no view. A
Black–Litterman posterior with no view is the distribution it was going to update, and an entropy
pooling fit with no view is its prior probabilities, which is what that family already does — the
two families now agree, where the refusal was the one place they did not.

Nothing is lost by dropping the refusal, because it was the **second** report of a fact already
reported. An unresolved name goes through `strict_diagnostic` where it is met, naming the name and
the universe it failed against: an `ArgumentError` under `strict = true`, a warning otherwise. That
report names the cause. The refusal named only the symptom, one layer down, and could not tell a
typo from a sub-universe — so it killed the second along with the first, and `strict = true` remains
the way to make a typo fatal.

It is not silent. The fit reports that it went view-free, and the two ways it can happen are
reported differently because they are different news. When the **ledger** holds a departure
casualty, the door's one report is raised from `@info` to `@warn` and speaks in the departure's
words: every view named an asset that left, so the posterior is the unconditioned one. When the
ledger is empty nobody departed, so the departure report is silent — it has no name to carry — and
the view-free fit gets a warning of its own instead. Either way the caller learns it, because
handing back an unconditioned answer where views were asked for changes the answer rather than
trimming it.

### What "no view" means differs by member, and one of them has no prior to hand back

Three of the four members answer the distribution the views were going to update. `bl_posteriors`
is that verb: `BlackLittermanPrior` forwards the wrapped prior's moment pair,
`FactorBlackLittermanPrior` forwards the prior **factor** pair and lifts it exactly as it lifts a
posterior one, and `AugmentedBlackLittermanPrior` takes the joint prior when — and only when — both
halves of its stack are empty, since either half alone still leaves the joint posterior conditioned
by the other. In none of them is the empty-view *algebra* the answer: `vanilla_posteriors` adds the
estimation-error term ``[(\tau\Sigma)^{-1}]^{-1} = \tau\Sigma``, so an empty ``P`` would widen the
covariance to ``(1 + \tau)\Sigma`` on the strength of views that are not there.

`BayesianBlackLittermanPrior` is the exception, and it is an exception in both directions. Its
update is a **precision sum**, ``\hat{\Sigma}^{-1} = \Sigma_f^{-1} + P^\intercal\Omega^{-1}P``, so a
``0 \times K`` ``P`` contributes exactly zero and carries no estimation-error term to inflate: the
empty block is safe there, where it is wrong everywhere else. But the answer it produces is not that
member's wrapped prior. With ``\hat{\Sigma} = \Sigma_f`` and ``\hat{\mu} = \mu_f`` the asset block
collapses, by Woodbury, to ``\Sigma_a + M\Sigma_f M^\intercal`` with mean ``M\mu_f + b`` — the
moments its factor model implies. That member transforms the prior it wraps whether or not a view is
stated, so it has no unadjusted answer to fall back to, and the empty-block algebra is the honest
one. `bl_view_block` is the verb, shared with the augmented member's halves.

### A precomputed view matrix over a gapped universe is refused by name

Every rule above works by resolving the caller's view **names** against the universe that survives.
A `BlackLittermanViews` passed in ready-made resolves none: `black_litterman_views` hands it back
untouched, because it was assembled against whatever universe the caller held and nothing
downstream can re-check it. So its `P` cannot be reduced — there is no way to tell which of its rows
the departed asset belonged to — and there is nothing to reduce it *by*, because an estimator whose
views are precomputed is permitted to carry no `sets` at all.

Left alone this is the defect the rest of this ADR removes, unfixed for one configuration and
**silent**: the update forms `P * sigma * transpose(P)` over the full universe, `0 * NaN` is `NaN`,
and every entry of the posterior comes back `NaN`. Under the default matrix processing that
surfaces one layer later as `ArgumentError: matrix contains Infs or NaNs`, naming the wrong cause;
under a matrix processing estimator that does nothing — a legitimate configuration — the fit
*succeeds* and the caller is handed an empty universe.

So a Black–Litterman estimator whose `sets` is `nothing` refuses, by name, when the wrapped prior
left any asset out. The message states why the pair cannot be reduced and names the two ways out:
state the views as a `LinearConstraintEstimator` over a `UniverseSets`, which resolves names and
drops only the rows a departure actually took, or fit over a universe in which every asset is
investable. This is the map's rule where it has to be — handle it, or throw a named error — and it
is the only place in the family where the second half applies.

The refusal is scoped to the two members whose views land on the **asset** axis. A precomputed
*factor* view matrix is unaffected by an asset departure, so `BayesianBlackLittermanPrior` and
`FactorBlackLittermanPrior` reduce their asset side and carry a precomputed `P` through unchanged.

## Consequences

- A view naming an asset that delists no longer poisons a posterior, refuses under `strict = true`,
  or silently strengthens a neighbouring view. The row goes, and the rest of the view set is fitted.
- The unit of a drop differs by shape, and the two are now stated: a **value** keyed by a departed
  name is skipped, and a **row** naming one is dropped whole. Neither refuses.
- A fit reports its departures itself. A standalone `prior(pe, X)` over a gapped panel used to say
  nothing at all, because only an optimisation door announced; it now names who left and what their
  leaving cost the view set, in one line per fit.
- `unknown_variable_msg` gained a `consequence`, because "term dropped" became wrong wherever the
  row is the unit. `get_linear_constraints` says "row dropped" and `name_to_val!` still says "term
  dropped".
- One diagnostic disappeared rather than moved: a row whose name missed the universe used to be
  reported twice, once for the term and once for the row it then emptied. The row now goes at the
  name, so the second report cannot fire, and the empty-row message is left to the row that
  resolved and was annihilated anyway — by loadings, or by its own cancelling coefficients.
- A group view survives a delisting among its members, at a mean or a sum over the survivors. A
  caller who wants the departed member to count against the group's total cannot express that, and
  should state the view over the members by name.
- `lcse`, `gcarde` and `sgcarde` stop refusing a departed name under `strict = true`, which is what
  ADR 0124 said of them and did not yet do.
- Under `strict = false` a row carrying a genuinely unknown name is dropped rather than fitted
  without its term. This is a behaviour change beyond the point-in-time universe, and it is
  deliberate: the previous behaviour fitted a row the caller did not write.
- A walk-forward whose views name an asset that delists mid-sample runs to the end, and the windows
  after the delisting are fitted on fewer views than the windows before it. The per-window record is
  the only place that difference is visible.
- `bl_preroll` no longer refuses a view set with no row left, so #852's named error is gone and a
  Black–Litterman fit can no longer die because its views did not resolve. A **sub-universe** fit is
  what this buys: a `NestedClustered` optimisation whose per-cluster views name assets in other
  clusters now fits each cluster on the views that cluster holds, and on none where it holds none,
  rather than raising. That case reached CI as the failure this rule was written from.
- A caller who *wants* an unresolved view to be fatal says `strict = true`, and gets an
  `ArgumentError` naming the name and the universe. That was always the better report and is now the
  only one; a caller who leaves `strict = false` gets a warning per unresolved name and a second
  warning saying the fit went view-free.
- A view-free fit is reported two ways, because it happens two ways: a departure that took the last
  view raises the door's own report to `@warn`, and every other cause gets a warning of its own,
  since the door has no departed name to report and would otherwise say nothing at all.
- A **precomputed** asset-side view matrix over a gapped universe is refused by name rather than
  reduced. It is the one configuration in the family that cannot be reduced, and the one place this
  ADR's rule falls to the second half of the map's: handle it, or throw a named error.
- The Black–Litterman family expands where the entropy pooling family does not, and the asymmetry is
  structural rather than a difference of care. An entropy pooling row runs over **observations**, so
  its solved probabilities carry no asset axis; a Black–Litterman posterior is a moment pair over
  the reduced assets, and the contract says a prior result lives on the full universe. So the four
  Black–Litterman members write theirs back with `expand_moment`, and `FactorBlackLittermanPrior`
  and `AugmentedBlackLittermanPrior` write their reconstruction and their regression back too. A
  Cholesky factor is the one thing dropped rather than expanded: a `NaN` frame has no factorisation.
- The `w` a prior estimator carries is per-asset configuration and is sliced to the reduced axis
  with `investable_weights_view`, the verb the optimisation side already had. Without that,
  `FactorBlackLittermanPrior` and `AugmentedBlackLittermanPrior` met a narrower loadings matrix with
  a full-universe weight vector and raised a bare `DimensionMismatch` naming no asset.

## Alternatives refused

- **Drop the departed term and keep the row**, extending ADR 0124's rule verbatim. Refused because a
  row is a joint statement: `"a + c == 0.05"` fitted as `"a == 0.05"` asserts something the caller
  never wrote, and does it in silence.
- **Refuse a departed name under `strict = true`, as a typo is refused.** Refused because a caller
  cannot foresee which asset a prior will fail to estimate: a strict walk-forward would die in the
  first window that delists, and this map's closing test is exactly that walk-forward.
- **Announce each dropped row where it is dropped.** Refused because a departure is one event, and a
  message per row repeats it per row per window of a walk-forward. What the caller needs — *what* the
  departure cost — is kept, by the ledger, and said once at the door.
- **Say nothing about the casualties, only who left.** Refused because the departed asset's name
  does not tell a caller that two of their three views went with it, and nothing else will.
- **Carry the ledger in a scoped value**, so no signature changes. Refused because the data flow
  would then be invisible at every site that writes it, and because the doors that would set it do
  not wrap the work in a dynamic extent — an optimisation door reduces and returns, leaving nothing
  to scope over.
- **Let the group keep its original denominator**, so the row still reads as the mean over the group
  as it stood. Refused because the right-hand side does not move with it, so the fitted row asserts a
  mean it does not compute.
- **Drop the whole row when a group loses any member.** Refused because a fifty-name sector view
  would die for one delisting, in every window that delists.
- **Hand the builders the departed names as an argument**, leaving `nikey` a thing only optimisation
  doors mint. Refused as a second way to say what `UniverseSets` already says, over the same data.
- **Build the pair over the full universe and drop the departed columns afterwards**, keeping the
  diagnostics closest to what the caller wrote. Refused because the builder still needs the mask to
  shed a group's departed members, so the axis is required either way, and the assembled pair would
  come out at a width the rest of the reduced path does not use.
- **Keep `bl_preroll`'s refusal for a view set that was empty or mistyped from the start**, letting
  it stand wherever no departure wrote into the ledger. Refused because the ledger does not separate
  the cases it would have to: a sub-universe — a cluster, a resampled subset — writes into it
  exactly as little as a typo does, so the refusal would kill a view set that is correct over the
  universe the caller stated it against. It was also the second report of a fact `strict_diagnostic`
  already reports at the name, and the weaker of the two: it named the symptom, one layer below the
  cause. `strict = true` remains the way to make an unresolved name fatal.
- **Reduce a precomputed `P` by slicing its columns at the mask**, dropping the rows left with no
  nonzero coefficient. Refused because it guesses at what the caller wrote: a row can be annihilated
  by its own coefficients rather than by a departure, and the two are indistinguishable once the
  names are gone. A caller who wants a departure handled has a way to say so — resolve the views by
  name — and the refusal names it.
- **Answer the empty-view algebra rather than the wrapped prior**, letting a `0 × N` view matrix run
  through the master equations. Refused because the covariance term adds the estimation-error block
  ``\tau\mathbf{\Sigma}``, so an empty ``\mathbf{P}`` widens the posterior to
  ``(1 + \tau)\mathbf{\Sigma}`` — a covariance produced by views that no longer exist. Two members
  take the empty block anyway, and neither contradicts this: the augmented member's *halves* do,
  because a half that emptied still leaves the joint posterior conditioned by the other one, and
  `BayesianBlackLittermanPrior` does because its update is a precision sum with no estimation-error
  term for an empty ``\mathbf{P}`` to inflate.
