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

### A name on the Non-Investable Axis drops the whole row, in silence

A view row naming an asset the Investable Mask left out is dropped entire — the term is not struck
from the row and the row is not fitted without it. The dropped row joins `excl`, which
`get_black_litterman_views` already maintains, so `remove_excl_views` drops the matching entry of a
per-view confidence vector and the surviving rows keep their order and their confidences.

The drop is silent under **both** settings of `strict`, and this is the same reasoning ADR 0124
gave for the value-keyed family: `strict` exists to catch a caller's typo, a departed name is the
opposite of a typo — it was correct over the universe the caller was handed, and the data moved it —
and no caller can foresee which asset a prior will fail to estimate. Refusing, or warning once per
row per window of a walk-forward, reports something nobody can act on. The departure itself is
announced once, by the door that derived the mask.

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
**surviving** count, and the entropy pooling sum runs over the survivors. The row survives, in
silence.

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

`get_black_litterman_views`, `get_linear_constraints` and `replace_group_by_assets` gain the
counterpart-axis argument `name_to_val!` already carries, read with `counterpart_axis_names`. One
mechanism therefore serves the priors and the optimisation side alike, and the `lcse`, `gcarde` and
`sgcarde` gap named in the Context closes with it rather than in a separate repair.

The prior does **not** call `announce_non_investable`. A prior fitted inside an optimisation is
fitted before the door derives the mask, so the door announces the same names immediately
afterwards, and two `@info` per window is one too many. The departure is one event and the door
owns it.

The mint is local to the fit. A Prior Result carries no sets, so the axis does not travel out of the
prior, and `port_opt_view` drops it, so an optimiser that reduces afterwards mints its own. ADR
0124's invariant — a `UniverseSets` carrying the axis was reduced by exactly one door, for exactly
one problem — is unchanged, with the wrapping prior's own entry now among the doors.

### When a departure drops the last view, the fit proceeds view-free and says so once

A view set whose every row has been dropped leaves nothing to assemble, and
[#852](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/852) settled what that means
today: `get_black_litterman_views` returns `nothing`, and `bl_preroll` turns it into a named error.
Under this ADR a delisting can empty the set, and refusing then would put back exactly the
unforeseeable refusal every rule above removes.

So when the last surviving row was dropped by a **departure**, the fit proceeds with no views. A
Black–Litterman posterior with no view is its wrapped prior, and an entropy pooling fit with no view
is its prior probabilities, which is what that family already does. It emits **one** warning, naming
the departed assets that cost the last view. This is the one place silence is too quiet: handing
back the wrapped prior where the caller asked for views changes the answer, and the caller has no
other way to learn it.

`bl_preroll`'s named error is kept for the case it was written for: a view set that was empty, or
mistyped, from the start.

## Consequences

- A view naming an asset that delists no longer poisons a posterior, refuses under `strict = true`,
  or silently strengthens a neighbouring view. The row goes, and the rest of the view set is fitted.
- The unit of a drop differs by shape, and the two are now stated: a **value** keyed by a departed
  name is skipped, and a **row** naming one is dropped whole. Both are silent.
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

## Alternatives refused

- **Drop the departed term and keep the row**, extending ADR 0124's rule verbatim. Refused because a
  row is a joint statement: `"a + c == 0.05"` fitted as `"a == 0.05"` asserts something the caller
  never wrote, and does it in silence.
- **Refuse a departed name under `strict = true`, as a typo is refused.** Refused because a caller
  cannot foresee which asset a prior will fail to estimate: a strict walk-forward would die in the
  first window that delists, and this map's closing test is exactly that walk-forward.
- **Announce each dropped row.** Refused because a departure is one event, announced once by the
  door; a message per row repeats it per row per window. The one exception is the row that leaves
  the view set empty, which changes the fitted model rather than trimming it.
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
- **Announce the departure at the prior too**, so a standalone fit reports it. Refused because the
  door announces the same names immediately after every fit that happens inside an optimisation.
- **Keep `bl_preroll`'s refusal when every view drops.** Refused for the reason the strict refusal
  was: the caller could not have known, and the window is the one place the refusal is least
  actionable.
