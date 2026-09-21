---
status: proposed
---

# A follow-the-leader rule solves its programme on the allocation set, and projects only its damped mix

## Context

[ADR 0156](0156-every-online-selection-algorithm-ships-as-a-closed-form-rule-an-expert-mixture-or-a-follow-the-leader-over-a-selected-sample.md)
made `FollowTheLeader(; sel, opt, gamma)` a rule whose update re-solves an optimisation estimator
on the rows its Sample Selector names and answers `(1 − γ) w⋆_t + γ w_t`, and left the meeting
with the constraint vocabulary to the constrained-update ticket with the words "a constraint is a
constraint on `opt`".
[ADR 0159](0159-a-constrained-online-update-projects-onto-an-allocation-set-in-the-rules-own-geometry-and-the-default-set-needs-no-solver.md)
then put every constraint of the family on the head's Allocation Set and every rule's step
through `project(proj, set, q, w)`, and does not mention the solved rows. An audit of the map
found the two sentences give one constraint two homes;
[issue #1168](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1168) asks which.

- **The projected optimum is not the leader.** Follow the leader is `argmax_{w ∈ K} Σ_s log⟨w, x_s⟩`
  over the decision set `K`; its regret theorems are stated on `K`. On sixty rows of three assets
  on a two-period cycle — returns `+0.05 / −0.02`, `+0.044 / −0.026` and `−0.01 / +0.03` on odd
  and even rows, so the first two co-move, the second with the lower drift, and the third moves
  against them — and a cap of `0.5`, the unconstrained leader is `[1, 0, 0]`; the leader solved on
  the capped set is `[0.5, 0, 0.5]`, log wealth `0.744` over the rows; the unconstrained leader
  projected onto the capped set in the Euclidean geometry is `[0.5, 0.25, 0.25]`, log wealth
  `0.717`. The projection spreads the excess equally where the constrained solve moves it all to
  the asset in anti-phase, and the projected answer grows less over every pair of rows. The fixture
  is the one `test/test_67c_follow_the_leader.jl` pins, and the two numbers are its.
- **Every JuMP head already admits an extra constraint object.** The shared model assembly of the
  JuMP heads ends with `add_custom_constraint!(model, opt.ccnt, optimiser, attrs)`
  (`src/17_Optimisation/05_JuMP/03_JuMPOptimiser.jl`), so a `CustomJuMPConstraint` on
  `JuMPOptimiser.ccnt` — one or a vector — receives the model mid-assembly, its weight variable,
  the homogenisation variable `k` and the constraint scale, and adds what it likes, binaries
  included. The Allocation Set's own builders are the ones the constrained-update build assembles
  a bare model from, so they run on the leader's model unchanged.
- **Translation field for field is not one mechanism.** The head's risk ceiling `r` is a field of
  `MeanRisk` and not of `JuMPOptimiser`, the tracking error is `tr`, the linear constraints are
  `lcse`; a translation needs a rebuild per head type and a rule for a field the caller also set on
  `opt`. And it would move the set's variance ceiling onto the leader's own `pe`, fitted on the
  selected rows, where ADR 0159 fits the set's `pe` on the head's rows.
- **A set refused on a solved rule breaks the mixture.** ADR 0156 makes the pattern-matching
  aggregation an `ExpertMixture` over `FollowTheLeader` rules, and
  [ADR 0163](0163-an-expert-mixture-weights-its-experts-on-an-expert-set-of-its-own-and-projects-its-blend-onto-the-allocation-set-once-more.md)
  projects every expert and the blend onto the head's set; an expert that refuses the set cannot
  sit under a blend that meets it. The turnover ceiling's reference, the Price-Adjusted Allocation
  of [ADR 0160](0160-an-online-update-starts-from-its-own-allocation-and-its-trade-is-measured-from-the-price-adjusted-one.md),
  lives on the set alone.
- **The damped mix can leave the set on two kinds.** `w⋆_t` is feasible by construction and `w_t`
  was projected onto the set a period ago, so on every static convex kind — bounds, linear,
  variance ceiling, tracking error — a convex mix of the two is feasible. A turnover ceiling is
  measured from `ŵ_t`, which moves with the day's price relative: with `w_t = [0.5, 0.5, 0]`,
  `x_t = [1.5, 0.5, 1]`, so `ŵ_t = [0.75, 0.25, 0]`, and `tn = 0.2`, the leader
  `w⋆ = [0.65, 0.25, 0.10]` sits exactly `0.2` from `ŵ_t`, `w_t` sits `0.5` from it, and the mix at
  `γ = 0.5`, `[0.575, 0.375, 0.05]`, sits `0.35` from it. Under `card = 2`, `w⋆ = [0.6, 0, 0.4]`
  and `w_t = [0.5, 0.5, 0]` mix to three names. ADR 0163 met the same fact on the mixture's blend.
- **The solver-free `opt` has no model to add to.** `BestConstantRebalancedPortfolio`
  ([ADR 0161](0161-log-wealth-regret-is-a-verb-over-two-prediction-results-and-a-hindsight-comparator-is-an-estimator-fit-on-the-rows-it-is-scored-on.md))
  runs Cover's fixed point on the simplex; its `wb` is the finaliser's repair after the fixed point,
  stated in its docstring, and it has no field for any programme kind.

The two implementations the maintainer holds in local memory give a caller neither: one solves
every pattern-matching programme on the bare simplex and admits no other constraint; the other
projects a linearised leader step onto its constraint set, which is its whole mechanism and not a
re-solve.

## Decision

### The set enters the re-solve as a constraint of the programme, and the optimum is the answer

A `FollowTheLeader` update forms the step's Price-Adjusted Allocation `ŵ_t`, wraps the head's
resolved Allocation Set, `ŵ_t` and the head's rows into an `AllocationSetConstraint <:
CustomJuMPConstraint`, appends it to the held optimiser's `ccnt` for that solve, threads `ŵ_t`
through `factory(opt, ŵ_t)` as the fold loop does, and runs `optimise` on the rows the Sample
Selector names. The programme's feasible region is the Allocation Set intersected with whatever
the held optimiser carries itself; the optimum is the leader and is taken as is — no projection.
`add_custom_constraint!` on the adapter calls the same builders the bare projection model of ADR
0159 is assembled from, in the same order, so the set's turnover ceiling measures from `ŵ_t`, its
risk ceilings and tracking errors read the set's own `pe` and the head's rows, its MIP kinds add
their binaries to the leader's model, its semidefinite phylogeny reuses the leader's lifted `W`,
and its objective penalties — `l1`, `l2`, `lp`, `linf`, `cobj` — fold into the Objective Penalty
the leader's objective builder folds in, the door the semidefinite phylogeny's own `p · tr(W)`
already takes from inside a constraint builder. The adapter is one door, not two. A turnover or a fee the caller placed on `opt` itself reads the
same book, because `factory` gave it `ŵ_t`.

Both homes hold. A caller who wants one home leaves `opt` bare of constraints and writes them on
the head's `set`; the docstring says so, and a bound set in both places is a redundant row, not a
clash. On the default set — the simplex — the adapter adds what a bare `JuMPOptimiser` already
has, so a default configuration solves the same programme it would have solved.

`BestConstantRebalancedPortfolio` as `opt`, the default, has no `ccnt`. A `BoundedAllocationSet`
goes into its `wb` and `sets`, so the answer is the repaired fixed point ADR 0161 already
documents and not the constrained leader; a `ProgrammeAllocationSet` is refused by name where the
head's `alg` and `set` meet, the site of ADR 0159's negative-bound refusal. The rule's docstring
states that the constrained leader under a programme set is `MeanRisk` under `LogarithmicReturn`.

### The damped mix is projected once more, in the Euclidean geometry

`FollowTheLeader` carries `proj::EuclideanProjection`, bound to that geometry, and its answer is
`project(proj, set, (1 − γ) w⋆_t + γ w_t, ŵ_t)`: the identity on every static convex kind, the
repair under a turnover ceiling on a day the drift exceeds it and under every MIP kind, skipped by
type on `BoundedAllocationSet` as the mixture's second projection is. On a programme set a period
whose mix leaves the set costs two programmes; at `γ = 0` the projection is the identity on every
set and is skipped. An empty selection answers the uniform portfolio projected onto the set in the
same geometry — the Start Allocation's rule of
[ADR 0162](0162-an-online-selection-head-buffers-returns-and-starts-from-a-given-allocation-or-a-uniform-one-over-the-pinned-universe.md).

### A failed re-solve is a Held Step, and a mixture inherits both

The held optimiser's own fallback chain runs first, as it would anywhere, and its answer is the
leader's; a result that is not a success is ADR 0159's Held Step, unchanged. `FollowTheLeadingHistory`
and an `ExpertMixture` over `FollowTheLeader` experts apply both rulings per expert and then ADR
0163's blend projection.

## Considered options

1. **The set translated field for field into the held optimiser's constraint fields.** Rejected:
   the same weights as the decision on a bounded set, but a rebuild per head type, a precedence
   rule for a field the caller also set, and the set's variance ceiling moved onto the leader's
   `pe` over the selected rows against ADR 0159.
2. **The solved answer projected like any rule's step.** Rejected on the example: `[0.5, 0.25,
   0.25]` at `−0.063` is not the leader, and the theorem is on `K`.
3. **A set refused on a solved rule, the caller constraining `opt` alone.** Rejected: incoherent
   with a mixture over follow-the-leader experts under ADR 0163, and the turnover reference of
   ADR 0160 lives on the set.
4. **`gamma > 0` refused on a set carrying a turnover ceiling or a MIP kind.** Rejected: it takes
   the damping away from the turnover-conscious caller, the one who wants both, and the mix is
   fine on every static convex kind, so the refusal would read the set's kinds.
5. **The mix left unprojected and the violation documented.** Rejected: the head's docstring says
   every allocation of the recursion lies in the set, and the violation lands on the day a
   ceiling binds.
6. **The mix as a proximal term inside the programme.** Rejected: that is a regularised leader,
   a different rule from the papers' damping of the solved answer.
7. **`BestConstantRebalancedPortfolio` refused on any non-simplex set.** Rejected for now: the
   repaired fixed point under a bound is what ADR 0161 already documents for the head itself, and
   the caller who wants the constrained leader has `MeanRisk`.

## Consequences

- ADR 0156's consequence line "where a constraint is a constraint on `opt`" is rewritten in place
  to point here; ADR 0159's mechanism sentence gains that a solved rule's programme takes the set
  as its feasible region and does not project its optimum; ADR 0161's roster line on the
  solver-free `opt` gains "on the default set".
- `CONTEXT.md`'s *Constrained Update* loses "a solve of the rule's own objective, which no member
  of the family does" and states the solved rule's meeting; *Sample Selector* and the
  `BestConstantRebalancedPortfolio` roster line gain the set's route; a new term *Allocation Set
  Constraint* names the adapter.
- The third-set build,
  [#1178](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1178), owes
  `AllocationSetConstraint` and its `add_custom_constraint!`, the `factory` threading of `ŵ_t`,
  the `proj::EuclideanProjection` slot on `FollowTheLeader` and its skip by type, the refusal of
  a programme set on the solver-free `opt`, and four tests: the constrained leader on the capped
  example above equals `MeanRisk` under `LogarithmicReturn` with the same bound on `opt`; the
  projected leader is not it; the mix repair on the turnover example; and the `card = 2` mix.
- The Constrained Update build,
  [#1162](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1162), writes its set
  builders so that they take a model and add to it, since the adapter calls them on a model it did
  not build.
