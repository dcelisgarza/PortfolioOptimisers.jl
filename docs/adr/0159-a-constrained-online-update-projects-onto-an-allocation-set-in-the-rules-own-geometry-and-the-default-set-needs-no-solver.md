---
status: proposed
---

# A constrained online update projects onto an allocation set in the rule's own geometry, and the default set needs no solver

## Context

[ADR 0155](0155-an-online-portfolio-selection-head-is-a-naive-optimiser-whose-batch-verb-is-a-causal-pass-and-whose-read-out-is-its-own-recursion.md)
made online portfolio selection one naive head with the rule on `alg`;
[ADR 0157](0157-an-online-selection-state-is-a-rule-state-beside-the-rows-held-once-and-a-block-of-rows-is-that-many-single-row-updates.md)
fixed the verb `(st′, w′) = online_update!(alg, st, w, x, rows)` with `w′` always a fresh vector,
"because the projection allocates one";
[ADR 0158](0158-a-forecast-reading-rule-holds-an-expected-returns-estimator-and-a-covariance-enters-on-the-constraint-that-reads-it.md)
ruled that a covariance enters *on the constraint object that reads it*, fitted on the head's rows,
and left that object to this decision.
[Issue #1155](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1155) asks how the
constrained update is solved: in which geometry, onto which constraints, by which mechanism, and how
the family's promise of *no prior, no covariance, no solver* survives a constraint that needs one.

Every online update has the same shape: an unconstrained step, then a projection back onto the
feasible set. The prototype projects every rule onto the simplex with the sorting algorithm of
Duchi, Shalev-Shwartz, Singer and Chandra (2008), the Euclidean projection. That one call is what
keeps the portfolio a portfolio and what makes the reversion rules sparse. It admits one feasible
set, `Σw = 1, w ≥ 0`, and nothing else: no bound, no group cap, no turnover ceiling, no variance
ceiling. Six facts measured on `dev` at `a44e79c271` shaped the decision.

- **The geometry is the theorem's, not decoration.** Exponentiated gradient's `O(√(T log N))`
  regret (Helmbold, Schapire, Singer and Warmuth 1998) is a relative-entropy argument, and its
  multiplicative update followed by normalisation *is* the exact Kullback–Leibler projection onto
  the simplex. The reversion rules (Li, Zhao, Hoi and Gopalkrishnan 2012; Li and Hoi 2012) are
  Euclidean, and their sparsity is the Euclidean projection zeroing entries. The online Newton
  step's logarithmic regret (Agarwal, Hazan, Kale and Schapire 2006) is proved for the projection
  in the norm of its Gram matrix `A_t`; the prototype substitutes the Euclidean projection and
  calls that "the standard simplification".
- **Under weight bounds alone, every geometry but the Gram norm is a scalar root.** The
  Euclidean projection onto `{Σw = 1, lb ≤ w ≤ ub}` is `w = clip(q − θ, lb, ub)` for the `θ`
  that restores the budget; the entropic projection is `w = clip(q / Z, lb, ub)` for the `Z` that
  does. On `w = [⅓, ⅓, ⅓]`, `x = [1.5, 1.0, 0.8]`, `η = 1`, cap `0.4`, the raw exponentiated
  gradient step is `[0.462, 0.293, 0.245]`; the entropic root gives `[0.400, 0.327, 0.273]`, the
  uncapped ratio `w₂/w₃ = 1.199` preserved, and the Euclidean root `[0.400, 0.324, 0.276]`, the
  uncapped difference `w₂ − w₃ = 0.049` preserved. Neither needs a solver. The Gram-norm
  projection is a quadratic programme even on the bare simplex.
- **The library's one projection object is a repair, not a step.** `JuMPWeightFinaliser` builds a
  bare model — `w`, `Σw = Σwᵢ`, `lb`, `ub` — and dispatches the objective on a typed formulation
  slot; `IterativeWeightFinaliser` clips and redistributes. Both return the input unchanged when
  the bounds already hold and both preserve the input's sum, not a budget, so neither is the
  simplex projection, and neither takes a linear constraint or a turnover.
- **The constraint builders are reusable on a bare model, except two.**
  `set_weight_constraints!`, `set_linear_weight_constraints!` on `linear_constraints(lcs, sets)`,
  `_set_turnover_constraints!` and the MIP builders take `model` plus the constraint object and
  read `w`, `k` and the scales off the model. The risk-measure builders take the optimiser in
  their `opt` slot and read three things off it: the solver a Deferred Quantity is resolved
  against, the `sets` and `strict` a `Variance`'s risk-contribution rows are resolved over, and
  the dispatch that decides whether `settings.ub` is honoured. Nothing else of the optimiser
  reaches a builder; `pr` carries the returns and the moments. The tracking-error builders take a
  prior result. A variance bound on its own is a ten-line second-order cone, `[ub; G w] ∈ SOC`
  with `G` the Cholesky factor of `sigma`, and a tracking error over the head's rows of returns is
  `‖X w − b‖ ≤ err √T`.
- **A budget is a cash column.** The library's wealth recursion at any budget is `1 + ⟨w, r⟩`;
  under `Σw = 1` it is `⟨w, x⟩`, the form every rule's formula uses — the gradient `x / ⟨w, x⟩`,
  the loss `⟨w, x⟩ ≤ ε`, the Newton gradient. A budget below one with idle cash is exactly a
  column with price relative `1`, and the column is the more capable of the two because the rule
  then decides the cash weight each period. The one case a budget field alone covers, leveraged
  long-only under the entropic geometry, takes every rule off its paper.
- **A programme can fail; a scalar root cannot.** A turnover ceiling and a cap can be jointly
  infeasible on the day a relisting re-enters at the recursion's weight (ADR 0157). The fold loop's
  convention on a failed fit is to hold and report (`held_start_weights`), and the finaliser's is
  to warn and fall back to a weaker answer.

## Decision

### The rule owns a Projection Geometry slot, bounded to what its theorem covers

Every Online Selection Rule carries a `proj` slot holding a Projection Geometry, a concrete
subtype of the unexported `AbstractProjectionGeometry`: `EuclideanProjection`,
`EntropicProjection` (Kullback–Leibler), `GramProjection(; slv)` (the norm of the rule's Gram
matrix), and, from ADR 0165, `TsallisProjection(; alpha)`, `LogBarrierProjection` and
`DiagonalProjection`, each a scalar root on the default set. The default is the paper's geometry, and **the slot's type bound on each rule names the
geometries admissible for that rule**, so a combination no theorem covers fails at construction and
not at run time. For the first set: `MirrorDescent` bounds the slot to every scalar-root map
(`ExponentiatedGradient` is its entropic constructor, ADR 0165); `NewtonStep` to `Union{EuclideanProjection, GramProjection}` with
`EuclideanProjection()` the default, which is the prototype's answer and needs no solver;
`PassiveAggressiveMeanReversion`, `ForecastReversion`, `ConstantRebalancedPortfolio` and
`BuyAndHold` to `EuclideanProjection`. A rule used as an Expert Mixture's weighting keeps its own
bound and projects onto the mixture's Expert Set, the bare simplex over the experts by default. The
mixture's experts each project onto the head's set in their own geometry, and the mixture then
projects its blend onto the same set once more in the Euclidean geometry, the identity wherever the
blend is already feasible
([ADR 0163](0163-an-expert-mixture-weights-its-experts-on-an-expert-set-of-its-own-and-projects-its-blend-onto-the-allocation-set-once-more.md)).
A later set states its bound in its own ADR under the same rule: the paper's geometry, plus any the
literature has run the rule under.

The sparsity follows from the geometry and is documented, not chosen: a Euclidean projection zeroes
every entry below its threshold, an entropic projection cannot zero a positive entry, and a
`GramProjection` zeroes where the quadratic programme does.

### The head holds one Allocation Set, and the default has no solver field

`OnlinePortfolioSelection` holds `set::AbstractAllocationSet` (unexported), with two concrete types.

- **`BoundedAllocationSet(; wb, sets)`** — weight bounds, or a `WeightBoundsEstimator` resolved over
  `sets`, and nothing else. It has **no `slv` field**, because every projection onto it is closed
  form: the Euclidean and the entropic scalar roots above, and the sort of Duchi and co-authors
  when the bounds are `(0, 1)`. It is the default, `BoundedAllocationSet()`, the simplex.
- **`ProgrammeAllocationSet(; wb, sets, lcs, tn, r, pe, te, card, …, slv)`** — every admissible
  kind below, with **`slv` required by its field bound**, so a set that needs a solver cannot be
  built without one. The projection is a bare JuMP model in the finaliser's idiom — `w`, `k = 1`,
  `Σw = 1`, the constraint scale and the objective scale — assembled by the shared builders and
  given the objective of the rule's geometry: `½‖w − q‖²` for `EuclideanProjection`,
  `Σ w log(w / q)` for `EntropicProjection` (an exponential-cone programme), `(w − q)ᵀ A (w − q)`
  for `GramProjection`.

`GramProjection` carries its own `slv` because it has no closed form on any set. The mechanism is
one dispatch, `project(proj, set, q, w)`: `(EuclideanProjection | EntropicProjection) ×
BoundedAllocationSet` is the scalar root; every other pair is the programme. A rule whose update is
itself a programme does not project its optimum: the set enters that programme as a constraint
and the optimum is the answer, and only its damped mix is projected
([ADR 0164](0164-a-follow-the-leader-rule-solves-its-programme-on-the-allocation-set-and-projects-only-its-damped-mix.md)).
That is the whole of the solver-free promise: it is a fact of the types, not a check, and the default configuration —
any rule at its default geometry on the default set — solves nothing.

### The admissible constraints, and the refusals

The programme set admits the kinds whose builders take a model and an object, plus the cone it
writes itself: weight bounds (`wb`, `sets`); linear constraints in the asset basis (`lcs` over
`sets`); a turnover ceiling (`tn`), the library's per-asset `|w_i − ŵ_i| ≤ tn_i`, whose reference
is the Price-Adjusted Allocation `ŵ_t = w_t .* x_t / ⟨w_t, x_t⟩` of the update, the book the step
trades from, computed in-step on that row — the executed trade, never the distance between two
targets (ADR 0160); a risk ceiling (`r`) under any `RiskMeasure`, its `settings.ub` the number,
built by the measure's own JuMP builder on the prior result of `pe` fitted on the head's rows as
ADR 0158 rules; a tracking error (`te`) over the head's rows; and the MIP kinds — cardinality,
group cardinality, thresholds and semi-continuous bounds — through the same
`model + wb + card + MIPSpace` builders `JuMPOptimiser` uses, under a MIP-capable `slv`.

The risk ceiling reaches the shared builders because the builders' `opt` slot takes a **Risk
Constraint Owner**, `RiskConstraintOwner = Union{<:RiskJuMPOptimisationEstimator,
<:AbstractProgrammeAllocationSet}`, and reads it through three methods that are the whole of what
a builder asks of an optimiser: `risk_constraint_solver` for the solver a Deferred Quantity is
resolved against, `risk_contribution_constraints` for a `Variance`'s risk-contribution rows, and
`set_risk_upper_bound!` for the bound, honoured for a `RiskBoundOwner` (the optimisers outside
factor risk contribution, and the programme set). `ProgrammeAllocationSet` subtypes
`AbstractProgrammeAllocationSet`, an Allocation Set whose projection is a programme, which lives in
`01_Base` with `AbstractAllocationSet` so the builders can name it. The set answers its own `slv`,
and answers a `Variance`'s `rc` as it is, because its constructor refuses one. Every existing
`set_risk_constraints!` method keeps its argument list; the `opt` annotation widened from the
optimiser type to the union, and no caller changed. The three optimiser reads had been written
inline in the builders (`opt.opt.slv` at three sites, `opt.opt.sets` and `opt.opt.strict` at one);
this is where they became methods.

The set materialises the measure against its prior through `factory` before the build, so a moment
the measure carries itself is the one it is built on and the model's shared caches — `:G`, `:Gkt`,
`:GV`, filled once per model from the first prior a builder saw — are never read for the ceiling.
This is what lets the ceiling join a JuMP leader's model beside the head's own measures, which the
head built from another prior; there the ceiling's entries live under the `:aset_` namespace with
the head's `w` registered under it, so a ceiling of the same kind as a head measure never meets it.
The measure's `rke` is cleared: the projection's objective is the geometry's divergence, and the
ceiling is a constraint alone. A `Variance` or `StandardDeviation` holding its matrix reads no rows
and no prior, and there is no prior result without rows, so that one ceiling stays the set's own
second-order cone — the same cone the shared builder writes — and keeps the ceiling from row one.

It refuses, by having no field for them: the SDP kinds, because a lifted `W = wwᵀ` has no meaning
in a one-step projection — a `Variance` with risk-contribution rows on `rc` is one, refused at
construction; a frontier or a per-asset `ub`, because a ceiling is one number; exposure constraints
in a factor Constraint Space, because their loadings come from a factor prior the family does not
read (the map's factor-forecast fog item); fees, which are a cost and not a constraint, and belong
to the start-weights-and-drift decision; and a budget, below.

A MIP projection is not unique, so the batch–online identity of ADR 0155 is stated precisely: it
is a claim about the *code path* — the Causal Pass and the Recursion Read-out run the same solves
in the same order on the same rows — and with a deterministic solver it holds exactly; it is not a
claim that the optimum is unique.

### The budget is one, and a bound may be negative where the geometry admits it

There is no budget field. `Σw = 1` on every set, so `⟨w, x⟩` is the wealth factor every rule's
formula assumes, and cash is an asset with price relative `1`, which the rule allocates like any
other. A negative lower bound is admitted under `EuclideanProjection` and `GramProjection`, so a
long-short reversion costs nothing, and refused at the head's construction under
`EntropicProjection`, whose `log w` is undefined below zero. The wealth factor of a leveraged
allocation can reach zero on an extreme day, where the log wealth and the next gradient are
undefined; that is documented on the set, not guarded.

### A failed programme holds the allocation: the Held Step

When the programme fails to solve — infeasible, timed out, or a MIP at its limit — the projection
answers the allocation it received, the step's Price-Adjusted Allocation, so the fund trades
nothing that period; the head warns once with the row's timestamp and continues, the rule's
carrier still absorbing the row. A programme whose constraints cannot be formed on the rows the
step holds — a covariance fitted on one observation — is held the same way, before any solve. The
hold is the family's own decision, so the read-out carries an `OptimisationSuccess` whose `res` is
the `HeldStep` record — the row's timestamp, the reason, and the solver trials — and the fallback
chain never runs on it (ADR 0160). A step never throws on a failed solve and never falls back to a
weaker set, because a constraint that is silently dropped on the day it binds is not a constraint.
The rows and the record travel outside `project`'s four-argument signature, on a task-scoped
Projection Step the head opens around each Online Update, so no rule threads them.

### The verification is the build's, not a prototype's

No prototype precedes the build. The projection build ticket carries four parity tests: the
Euclidean scalar root under `(0, 1)` equals the prototype's sort on every step of the seven rules;
the entropic root under a loose cap equals plain normalisation; the programme under a slack linear
constraint equals the closed form; and the Gram programme on the bare simplex matches a hand-written
quadratic programme.

## Considered options

1. **The rule's geometry derived, never stated — no slot.** Rejected: the Newton step has a real
   second choice, the Euclidean projection that keeps it solver-free, and a derived geometry has
   nowhere to put it.
2. **One Euclidean projection for all.** Rejected: exponentiated gradient under a binding bound is
   then no longer the paper's algorithm, and the mixture weighting loses its multiplicative form,
   for no saving — the entropic scalar root is as cheap as the Euclidean one.
3. **A caller slot with no bound.** Rejected: it admits combinations no theorem covers; the bound
   is what makes the slot a statement rather than a knob.
4. **Bare constraint fields on the head and a runtime `needs_solver` check.** Rejected: the refusal
   becomes a check instead of a bound, and the Gram norm's solver would sit on the head while its
   choice sits on the rule.
5. **Extend `WeightFinaliser` with an entropic formulation and `lcs`/`tn`.** Rejected: a finaliser
   returns the input unchanged when the bounds hold and preserves the input's sum, so it is a
   repair and not the step; its formulations are error norms, not divergences; and every naive head
   would inherit fields that mean nothing to it.
6. **Convex kinds only, no MIP.** Rejected by the maintainer, who wants the full constraint
   vocabulary; the cost is stated above as the precise form of the identity.
7. **Bounds and linear constraints only, no covariance on the set.** Rejected: ADR 0158 already
   placed the covariance here, and a variance ceiling is the one risk bound the constrained
   variants in the literature use.
8. **`lb ≥ 0` required everywhere.** Rejected: it refuses a long-short reversion the Euclidean
   rules compute correctly, and a `WeightBoundsEstimator` can yield a negative bound the caller
   never typed.
9. **A budget field, with the wealth factor `1 − Σw + ⟨w, x⟩` in every rule.** Rejected: a cash
   column gives the same allocations with every formula untouched, and a budget *range* would have
   the projection choose a budget as a side effect of its divergence.
10. **Throw on a failed programme.** Rejected: one infeasible day would end a walk-forward and a
    hyperparameter search; the loop's own convention is to hold and report.
11. **Fall back to the bounds-only closed form on a failed programme.** Rejected: the stated
    constraint is then violated on exactly the days it binds.
12. **A prototype ticket before the build.** Rejected: every question was answered without one;
    what remains is verification, which is a test.

## Consequences

- ADR 0155's field list for the head is rewritten in place: `wb`, `sets` and `wf` leave the head
  for the Allocation Set, and the projection replaces the Weight Finaliser as the enforcement.
  ADR 0158's restatement of that list follows.
- `CONTEXT.md` gains *Allocation Set*, *Projection Geometry*, *Constrained Update* and *Held
  Step*, and the *Online Update* entry names the Allocation Set where it said "the constrained
  set once a constraint is stated".
- The projection build ticket,
  [#1162](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1162), owes: the two abstract types and their five concrete members, the
  two scalar roots and the bare-model programme with three objectives, the `proj` bound on every
  first-set rule, the construction-time refusal of a negative bound under `EntropicProjection`, the
  Held Step, `rows_needed(set)` for the covariance and tracking-error kinds, and the four parity
  tests. The head's first build carries `set` from the start.
- [Issue #1163](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1163), outside the
  map, asked whether the risk-measure builders could be widened past the optimiser union so a
  programme set may bound any risk measure. They were, as *The admissible constraints* records:
  the builders take a Risk Constraint Owner, and the set is one.
- The map's build sequence is now phrasable: the constraint route was the last decision it waited
  on. The start-weights-and-drift decision, ADR 0160, ruled that the loop threads nothing into the
  update and that the turnover ceiling measures against the Price-Adjusted Allocation.
