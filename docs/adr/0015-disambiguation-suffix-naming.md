---
status: accepted
---

# Role suffixes are for disambiguation: bare when a name is unambiguous

## Context

A type's public name is the first signal a caller reads. The library names most
concepts *bare* — `Variance`, `ConditionalValueatRisk`, `MaximumDrawdown`,
`EmpiricalPrior`, `Covariance`, `MeanRisk` — letting the word carry the meaning. But
some types append a *role suffix* naming the abstraction they belong to: `RiskMeasure`,
`Estimator`, `Result`.

An ergonomics review flagged the `RiskMeasure` suffix as applied inconsistently: most
risk measures are bare, while `TurnoverRiskMeasure`, `TrackingRiskMeasure`,
`RiskTrackingRiskMeasure`, and `NonOptimisationRiskRatioRiskMeasure` carry the suffix,
alongside two that did not fit any rule — `EqualRisk` and `RiskRatioRiskMeasure`.
Said aloud, `Variance` and `TurnoverRiskMeasure` are the same kind of thing (a risk
measure), so a caller cannot predict the name or trust autocomplete — the convention
teaches one pattern but breaks it part of the time.

Looking across the surface, the suffixed measures share one property: each has a
**same-concept sibling that is not a risk measure**.

- `Turnover` (the turnover constraint spec) and `TurnoverEstimator` already own the
  word "turnover"; the measure form must therefore be `TurnoverRiskMeasure`.
- The `AbstractTracking` family — `WeightsTracking`, `ReturnsTracking`, `TrackingError`,
  `RiskTrackingError` — owns "tracking"; the measure forms are `TrackingRiskMeasure` and
  `RiskTrackingRiskMeasure`. `RiskTrackingError <: AbstractTracking` and
  `RiskTrackingRiskMeasure <: RiskMeasure` live in the same file and the same graph
  community, differing only in which abstraction they serve — exactly the case the suffix
  exists to separate.

The two misfits had no such sibling: nothing else claims "equal risk" or "risk ratio",
so `EqualRisk` → `EqualRisk` and `RiskRatioRiskMeasure` → `RiskRatio` were renamed
to bare. That left one violation: `NonOptimisationRiskRatioRiskMeasure` keeps the suffix,
but its only same-stem siblings (`RiskRatio`, `ExpectedReturnRiskRatio`,
`MeanReturnRiskRatio`) are all risk measures — no non-measure construct competes for the
name.

The `Estimator`/`Result` suffixes complicate any "suffix means collision" reading. Many
`…Estimator` types do disambiguate a sibling (`TurnoverEstimator`/`Turnover`,
`WeightBoundsEstimator`/`WeightBounds`, `LinearConstraintEstimator`/`LinearConstraint`,
`ClustersEstimator`/`Clusters` — ten such pairs). But several carry `Estimator` with no
bare sibling at all: `NetworkEstimator`, `CentralityEstimator`, `AssetSetsMatrixEstimator`,
`NetworkClustersEstimator`, `HighOrderFactorPriorEstimator`. There, the suffix is doing a
*second* job — turning a generic domain noun ("network", "centrality") into a name that
reads as the type rather than the concept. So no single mechanical rule governs all three
suffixes.

## Decision

A shared spirit, refined per role.

**Shared spirit — bare when unambiguous.** A type is named by the bare concept word when
that word is unambiguously the type. A role suffix (`RiskMeasure`, `Estimator`, `Result`)
is added only to earn back clarity the bare word would lose — never as a blanket category
marker. (Most risk measures and many estimators are bare; this is the default, not the
exception.)

**`RiskMeasure` — disambiguation only.** A risk-measure type takes the `RiskMeasure`
suffix *if and only if* its concept word already names a non-risk-measure construct (a
constraint, tracking, or estimator type), so the bare word is not free for the measure to
claim. Concretely:

- Keep the suffix: `TurnoverRiskMeasure` (vs `Turnover`/`TurnoverEstimator`),
  `TrackingRiskMeasure` and `RiskTrackingRiskMeasure` (vs the `AbstractTracking` family
  incl. `RiskTrackingError`).
- Drop the suffix: `EqualRisk` → `EqualRisk`, `RiskRatioRiskMeasure` → `RiskRatio`
  (done), and `NonOptimisationRiskRatioRiskMeasure` → `NonOptimisationRiskRatio` — none
  has a non-measure namesake; the `NonOptimisation` prefix already separates it from the
  hierarchical `RiskRatio`.

  **Why** disambiguation only, not a universal marker: a universal `RiskMeasure` suffix
  would rename the entire bare majority (`VarianceRiskMeasure`, …), which is exactly the
  verbosity the bare convention avoids. Keying the suffix to an actual naming collision
  makes the name predictable — a caller can ask "does another non-measure thing own this
  word?" and know the answer from the name.

  **Consequence:** `NonOptimisationRiskRatioRiskMeasure` is a breaking rename, consistent
  with the project's established practice of direct `Breaking:` renames without aliases
  (e.g. `LxTracking` → `LxNorm`). After it, the `RiskMeasure` suffix has zero exceptions.

**`Estimator`/`Result` — role markers, applied for collision *or* generic-noun clarity.**
These keep the suffix when a same-stem sibling exists (the disambiguation case) **or** when
the bare name is a generic domain noun that does not read as the type on its own
(`NetworkEstimator`, `CentralityEstimator`, `AssetSetsMatrixEstimator`). This is looser
than the `RiskMeasure` rule and deliberately so — it describes existing, deliberate
practice rather than forcing renames of correct names.

**No suffix stacking.** A type carries at most one role suffix, chosen for the role it is
confusable in. There is no `TurnoverRiskMeasureEstimator`.

## Alternatives considered

- **One strict universal disambiguation rule for all three suffixes.** Rejected: the
  `Estimator` suffix already violates it (`NetworkEstimator` et al. have no sibling), so a
  single strict rule would either mislabel those as errors or demand a wave of renames
  outside this decision's scope.
- **Scope the ADR to `RiskMeasure` only.** Rejected: it leaves the `Estimator`/`Result`
  conventions undocumented, inviting a future review to re-litigate why `NetworkEstimator`
  is suffixed but `Variance` is not.
- **Keep `NonOptimisationRiskRatioRiskMeasure` as a documented exception.** Rejected in
  favour of the rename: a rule with zero exceptions is worth one breaking change in a
  pre-1.0 package that already renames freely.

## Rollout

`EqualRisk` → `EqualRisk` and `RiskRatioRiskMeasure` → `RiskRatio` are already
done. Remaining: rename `NonOptimisationRiskRatioRiskMeasure` → `NonOptimisationRiskRatio`
(type, constructors, exports, tests, docs, and the `CONTEXT.md` §5 entry). Hard rename, no
alias, under a `Breaking:` commit. The `Estimator`/`Result` names are already compliant and
need no change.
