---
status: proposed
---

# An expert mixture weights its experts on an expert set of its own, and projects its blend onto the allocation set once more

## Context

[ADR 0156](0156-every-online-selection-algorithm-ships-as-a-closed-form-rule-an-expert-mixture-or-a-follow-the-leader-over-a-selected-sample.md)
made `ExpertMixture` a rule over rules: `K` experts, a weight vector `p` over them moved by a
weighting that is itself an Online Selection Rule applied to the expert-return vector
`r_t = (⟨h_k(t), x_t⟩)_k`, and the answer `Σ_k p_{t+1,k} h_k(t+1)`.
[ADR 0159](0159-a-constrained-online-update-projects-onto-an-allocation-set-in-the-rules-own-geometry-and-the-default-set-needs-no-solver.md)
gave every rule a `proj` slot, admitted a turnover ceiling and the MIP kinds on the head's
Allocation Set, and said of the mixture that its weighting rules are bound to `EntropicProjection`
and that "a convex combination of feasible experts is feasible, so the experts' rules carry the
geometry and the mixture projects nothing". An audit of the map found three things that sentence
does not survive;
[issue #1167](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1167) asks them.

- **The weighting has nothing of dimension `K` to project onto.** The weighting is a rule, and a
  rule's step is projected onto an Allocation Set in its `proj` geometry; but the head's `set`
  constrains `N` asset weights and `p` has `K` entries. The same ADR bounds `BuyAndHold` to
  `EuclideanProjection` and `NewtonStep` to `Union{EuclideanProjection, GramProjection}`, so
  "the weighting rules bound the slot to `EntropicProjection`" cannot hold for the default
  weighting or the Newton one, and `TopK` is no projection at all.
- **A blend honours the convex kinds and nothing else.** Weight bounds, asset-space linear
  constraints, a variance ceiling and a tracking error are convex, so a blend of feasible experts
  is feasible. A cardinality, a threshold or a semi-continuous bound is not: two corner experts
  `e₁ = [1, 0]`, `e₂ = [0, 1]` are each feasible under `card = 1`, and their blend at
  `p = [0.6, 0.4]` names two assets. The turnover ceiling is convex but *anchored*: each expert
  trades at most `tn` from its own Price-Adjusted Allocation `ŵ_k`
  ([ADR 0160](0160-an-online-update-starts-from-its-own-allocation-and-its-trade-is-measured-from-the-price-adjusted-one.md)),
  and the mixture's own trade is measured from `ŵ_mix = w_mix .* x / ⟨w_mix, x⟩`. Under
  `BuyAndHold` the two agree, because `p′ ∝ p .* r` is exactly the mixture's drift:
  `Σ_k p′_k ŵ_k = ŵ_mix`, so `‖w_mix,t+1 − ŵ_mix‖₁ ≤ Σ_k p′_k ‖h_k(t+1) − ŵ_k‖₁ ≤ tn`. Under any
  other weighting `p_{t+1} ≠ p′` and the re-weighting of the experts is a trade the ceiling does
  not see. On the two corner experts with `p = [½, ½]`, `x = [1.2, 0.8]` and `tn = 0.1`:
  `ŵ_mix = [0.6, 0.4]`, `BuyAndHold` answers `[0.6, 0.4]` with zero trade, and
  `ExponentiatedGradient(; eta = 5)` answers `[0.88, 0.12]` with a trade of `0.56`.
- **The literature runs its mixtures unconstrained, and its one constrained shape projects the
  blend alone.** Cover's universal portfolio and the pattern-matching aggregations run on the
  simplex; the meta-aggregators of the online-convex-optimisation literature run every expert
  unconstrained and project the blend once onto the feasible set, so none of them holds
  constrained experts. No paper bounds the weight an expert may carry; the fixed-share weighting
  of the switching portfolio keeps every expert alive by mixing in a uniform share, which is the
  weighting's own rule and not a bound.

## Decision

### The weighting projects onto an Expert Set the mixture holds, bare by default

`ExpertMixture` gains `eset::Option{<:BoundedAllocationSet}`, `nothing` by default. The **Expert
Set** is the Allocation Set over the `K` experts that the weighting's step is projected onto, in
the weighting rule's own Projection Geometry — its `proj` slot, bounded as ADR 0159 bounds every
rule's. `nothing` is the bare `K`-simplex, `p ≥ 0`, `Σ p = 1`: a no-op for the multiplicative
weightings (`BuyAndHold`, `ExponentiatedGradient`, `AggregatingAlgorithm`), the Euclidean scalar
root for a `NewtonStep` weighting, and nothing for `TopK`, which selects rather than projects. A
given `BoundedAllocationSet` is honoured the same way: a scalar bound broadcasts over the experts,
and a vector bound has one entry per expert. The Expert Set is always closed form, because the
bounded kind is the only one it admits; a programme over the experts is out of the family's
literature and refused by the field's bound. ADR 0159's sentence on the weightings' bound is
rewritten in place: each weighting's `proj` bound is its own rule's, and the set it projects onto
is the Expert Set.

### The blend is projected onto the Allocation Set once more, in the Euclidean geometry

`ExpertMixture` gains `proj::EuclideanProjection`, and its Online Update has one more step: the
experts each run their own Constrained Update against the head's `set` as ADR 0159 rules, the
weighting moves `p` on the Expert Set, and the blend `q = Σ_k p_{t+1,k} h_k(t+1)` is then
projected onto the head's `set` in the Euclidean geometry with the mixture's own Price-Adjusted
Allocation as the turnover reference, `project(proj, set, q, ŵ_mix)`. Where the blend is already
feasible the projection returns it: on every convex kind under `BuyAndHold`, and on every convex
kind but the turnover ceiling under any weighting. Where it is not, the projection is the repair:
`[0.88, 0.12]` becomes `[0.65, 0.35]` on the example above, and a blend of corner experts under
`card = 1` becomes the one-hot on its largest entry. On a `BoundedAllocationSet` a blend of
bounded allocations is bounded, so the second projection is skipped by dispatch on the set's type
and the default configuration solves nothing, as ADR 0159 promises. On a `ProgrammeAllocationSet`
the mixture pays `K + 1` programmes per period, the experts' and its own; the docs say so beside
the cost of the Newton weighting. The geometry is Euclidean only: the entropic projection cannot
produce the zeros a MIP kind needs, and the mixture has no Gram matrix.

The exact `log K` regret bound the mixture's docstring states
([ADR 0161](0161-log-wealth-regret-is-a-verb-over-two-prediction-results-and-a-hindsight-comparator-is-an-estimator-fit-on-the-rows-it-is-scored-on.md))
holds wherever the second projection is the identity; where it repairs, the docstring says the
bound is not claimed. ADR 0159's sentence "the mixture projects nothing" is rewritten to this.

### Follow the leading history inherits both

`FollowTheLeadingHistory`, whose expert set grows by one each period and is pruned, is a mixture
over a changing `K`: its weighting projects onto the Expert Set of the current experts, and its
blend meets the head's `set` once more, exactly as above. The one consequence of a changing `K` is
that its `eset` admits scalar bounds only, which broadcast over whatever experts are live; a
vector bound has no length to match and the set-3 build refuses it by name. The set-3 ADR cites
this one and adds nothing to it.

## Considered options

1. **The weighting on the bare `K`-simplex with no slot.** Rejected: the bare simplex is the
   default either way, and the slot buys a floor or a cap on an expert's share for one field that
   is `nothing` when unused.
2. **Refuse a mixture on a set carrying a turnover ceiling or a MIP kind**, by a type bound on the
   set's `Nothing`-typed fields. Rejected: it buys nothing a caller can run, and takes the universal
   portfolio and CORN-K off every turnover-bounded and cardinality set.
3. **Experts on the bare simplex, only the blend projected** — the meta-aggregators' shape.
   Rejected: one programme per period instead of `K + 1`, but on a convex programme set the answer
   is no longer a blend of feasible experts, the sampled constant rebalanced portfolios sit outside
   the set a caller could hold, and the `log K` bound against the best constrained expert is lost.
4. **Project the blend only when the set carries a turnover ceiling or a MIP kind**, by a
   value-level check on the set's fields. Folded into the decision as a build detail: the skip on
   `BoundedAllocationSet` is by type, and a `ProgrammeAllocationSet` whose kinds are all `Nothing`
   may skip by the same type-level fact; a runtime `isnothing` sweep is not the mechanism.
5. **The blend in the entropic geometry, or the weighting's geometry.** Rejected: the entropic
   projection cannot zero an entry, so no MIP kind is reachable; and the weighting's geometry is a
   statement about `p`, not about `w`.

## Consequences

- ADR 0159 is rewritten in place at its two sentences on the mixture: the weightings' bound and
  "the mixture projects nothing". ADR 0156's constructor line becomes
  `ExpertMixture(; experts, alg, eset, proj)`.
- `CONTEXT.md` gains *Expert Set*; *Expert Mixture* gains the second projection and the `K + 1`
  cost.
- The head's first build,
  [#1161](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1161), owes the two slots
  on `ExpertMixture`, the weighting's projection onto the Expert Set, and a test that a scalar
  bound on `eset` binds the trust vector.
- The Constrained Update build,
  [#1162](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1162), owes the second
  projection on a `ProgrammeAllocationSet`, the dispatch skip on `BoundedAllocationSet`, and three
  tests: the `BuyAndHold` identity under `tn`, the `ExponentiatedGradient` repair on the example
  above, and the one-hot under `card = 1`.
- The set-3 build owes `FollowTheLeadingHistory`'s scalar-only `eset` refusal.
