---
status: proposed
---

# An online update starts from its own allocation, and its trade is measured from the price-adjusted one

## Context

[ADR 0155](0155-an-online-portfolio-selection-head-is-a-naive-optimiser-whose-batch-verb-is-a-causal-pass-and-whose-read-out-is-its-own-recursion.md)
made online portfolio selection one naive head whose batch verb is a Causal Pass and whose read-out
is its own recursion;
[ADR 0157](0157-an-online-selection-state-is-a-rule-state-beside-the-rows-held-once-and-a-block-of-rows-is-that-many-single-row-updates.md)
fixed the verb `(st′, w′) = online_update!(alg, st, w, x, rows)`, made a fold of `k` rows `k`
single-row updates (the Block Step), and left to this decision whether the recursion should read
the fold's held weights mid-block;
[ADR 0159](0159-a-constrained-online-update-projects-onto-an-allocation-set-in-the-rules-own-geometry-and-the-default-set-needs-no-solver.md)
admitted a turnover ceiling on the Allocation Set with "the `w` the update receives" as its
reference, refused fees on the set as a cost and not a constraint, made a failed programme a Held
Step, and left to this decision what the fold loop threads into that `w`, what the ceiling measures
against, where a fee lives, and what a Held Step's retcode is.
[Issue #1156](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1156) asks all of that
in one question: which weights does the update start from, and what do Weight Drift, the
Previous-Weights Source, turnover and fees mean for a family that trades every row.

Every paper writes one recursion, `w_{t+1} = f(w_t, x_t)`, and one wealth, `S_T = Π_t ⟨w_t, x_t⟩`:
the portfolio is rebalanced to its target at the start of every period and the rebalance is free.
The library's walk-forward is not that: a fold holds one target for `test_size` rows, the held
weights drift under `SelfFinancingDrift`, the next fold reads either the previous target or the
drifted book through the Previous-Weights Source, and the fold charges the Result's fee. Seven facts
measured on `dev` at `4591c0c6b1` shaped the decision.

- **The loop cannot feed the recursion.** The previous fold's weights reach an estimator through
  `factory(x, w)` on the per-fold copy (`12_OnlineFoldLoop.jl:102–113`), *after* `partial_fit!`
  ran on the threaded estimator, and the batch Causal Pass has no loop at all. Anything the loop
  wrote into the recursion would make the batch fit over the folded rows and the online read-out
  disagree, and ADR 0157 holds them exactly equal at every `test_size`.
- **The Previous-Weights Source changes what turnover measures and nothing else.** `CONTEXT.md`
  says so, and no optimiser's decision reads it: `previous_weights` (`01_Base_CrossValidation.jl`)
  is read by `Turnover`, `TurnoverEstimator`, the tracking constraints, a `PreviousWeights`
  fallback and the turnover fee, through `factory`.
- **One row's drift is a closed form of the update's own inputs.** At `Σw = 1`, the book at the end
  of period `t` is `ŵ_t = w_t .* x_t / ⟨w_t, x_t⟩` — one row of `SelfFinancingDrift`
  (`drift_position_values`, `14_NetReturnsDrawdowns.jl`) — and `prev.hw.w` under a source at
  `test_size = 1` is exactly that vector. The literature reads it by name: transaction-cost
  optimisation (Li, Wang, Huang and Hoi 2018) updates from the price-adjusted portfolio, the
  standard cost model of the field (Blum and Kalai 1999) charges `‖w_{t+1} − ŵ_t‖₁`, and
  `BuyAndHold`'s update *is* `w_{t+1} = ŵ_t`. The family already contains the drift as a member.
- **Target-to-target turnover is backwards for every member.** On `w_1 = [½, ½]` and price
  relatives `[1.2, 0.9]`, `[0.9, 1.1]`, `[1.1, 1.0]`: `BuyAndHold` shows `‖w_{t+1} − w_t‖₁ =
  0.143, 0.099, 0.047` and executes no trade; `ConstantRebalancedPortfolio` shows `0, 0, 0` and
  trades `0.143, 0.100, 0.048` every row. A ceiling on the target-to-target distance would force
  buy-and-hold to trade back toward yesterday's weights and leave constant rebalancing unbounded; a
  fee on it would charge the one and spare the other.
- **No naive head charges a fee.** The fold charges `res.fees` alone (`predict`,
  `01_Base_CrossValidation.jl:1784`, `extract_fees(res, nothing)`, which reads
  `hasproperty(res, :fees)`), and `NaiveOptimisationResult` has `pr`, `wb`, `retcode`, `w`,
  `imsk`, `fb`. The four naive heads — `InverseVolatility`, `EqualWeighted`, `RandomWeighted`,
  `PreviousWeights` — score fee-free in every walk-forward, and the post hoc route (`fold_fees`
  with a caller's `Fees`) cannot thread a per-fold book into `tn.w`. The solver-free head that does
  carry a fee is `HierarchicalRiskParity`: `fees` on the estimator, `fees` on `HierarchicalResult`
  (`01_Base_ClusteringOptimisation.jl:136`), charged by the fold.
- **The head and the walk-forward never meet in a constructor.** `cross_val_predict(opt, rd, cv)`
  (`06_Validation.jl:28`) takes them as separate arguments; the first place both are in scope is
  the loop's entry, where `assert_online_entry` (`09_OnlineOptimisation.jl:464`) already refuses
  by name before any fold.
- **A failure retcode is read by three released seams.** `optimise`
  (`01_Base_Optimisation.jl:2947`) runs the fallback chain on one; `threads_weights(nothing, ·)`
  reads `fold_solved(retcode)` and skips the fold; `held_start_weights(::OptimisationFailure, w,
  w_prev)` starts the fold's drift from `w_prev`. ADR 0145's invariant is that the target weights
  are finite exactly when the retcode is a success. A Held Step's allocation is finite by
  construction, so a failure code on it would be mis-read by all three: on a block whose last step
  held, the fund would drift the previous target, or the fallback's answer, while the recursion
  continued from the held allocation, and the two would never reconverge. `OptimisationSuccess`
  carries a `res` field, `nothing` by default.

## Decision

### The update starts from its own allocation, and no flag changes that

An Online Update reads the allocation the recursion chose, `w_t`, as every paper writes it. The
loop's `w_prev` reaches the head through `factory` only — into a turnover fee and a
`PreviousWeights` fallback — and never into the recursion. A block of `k` rows runs its `k`
updates on the recursion's own allocations, and the fund's held book, which is the block-start
target drifted for `k` rows, is the Block Step's stated fiction and stays it. Under the
walk-forward's default source the two coincide at `test_size = 1` by construction, because the
previous fold's target *is* the recursion's allocation.

No flag anchors the update at the drifted book. The in-step form — gradient at `w_t`, anchor at
`ŵ_t` — is a different algorithm per rule with no paper behind it and no regret bound (on the
example above, exponentiated gradient at `η = 1` answers `[0.638, 0.362]` instead of
`[0.571, 0.429]` after one row); the paper that does anchor there, transaction-cost optimisation,
derives its whole update from that choice and is a member. The loop-threaded form needs a new seam
before `partial_fit!` and breaks the batch–online identity. What either would buy — a correct
reading of the executed trade — the next two rulings give without touching the rule.

### The Price-Adjusted Allocation is the reference of the family's trade

The **Price-Adjusted Allocation** of an update is `ŵ_t = w_t .* x_t / ⟨w_t, x_t⟩`: what the
fund holds at the end of period `t` before it trades, one row of `SelfFinancingDrift` at budget
one, computed in-step through the library's drift verb on that row. It is the reference of the
Allocation Set's turnover ceiling, always: the ceiling bounds `‖w_{t+1} − ŵ_t‖₁`, the trade the
step executes, so `BuyAndHold` has zero turnover and `ConstantRebalancedPortfolio` has the trade it
makes. ADR 0159's sentence "whose reference is the `w` the update receives" is rewritten in place
to this. Nothing on the set reads `w_prev`; `factory` reaches the head's fee and fallback alone.
The cost-aware rules read the same vector by their own definition.

There is no switch on the set. A position that measured against `w_t` would bound the distance
between two decisions, which every rule's own regulariser already is and which `Turnover` does not
name; and a knob one position of which is backwards for every member is worse than no knob.

### A fee lives on the head, its Result carries it, and a turnover fee requires a source

The head takes `fees::Option{<:FeesE_Fees}`, threaded by `factory` as every fee is, and
`NaiveOptimisationResult` gains `fees`, `nothing` by default, so the fold charges it through the
seam it already has. This is the hierarchical precedent field for field, and it is the
generalisation the family teaches: the four existing naive heads gain the same field in the same
build, because a walk-forward on any of them is a backtest the fund pays for and today none can be
charged a turnover fee at all. `PreviousWeights` under a source then prices the rebalance from the
drifted book back to the held target, which is a real trade, and under no source prices nothing,
which is right.

A turnover fee on this family is measured against the held book or it is measured against nothing
the family does. It is therefore **valid only under a Previous-Weights Source**: the loop's entry
refuses, by name and before any fold, an `OnlinePortfolioSelection` head whose `fees` carry a `tn`
term when the walk-forward's `pws` is `nothing`. `DriftedWeights()` is the whole configuration —
it implies its drift — and under it the fee and the ceiling read one number at `test_size = 1`.
A head with no `tn` fee reads nothing from the source and is not checked. The pairing spans two
structs, so no type bound and no constructor can refuse it; the loop's door is the earliest point
it exists.

The fee the fold charges and the cost a rule believes are two things. Transaction-cost
optimisation's `γ` shapes its trade and is a rule parameter with the paper's default; the head's
`fees` is what the series is charged. Either may be set without the other, both may be set, and
there is no double charge: a penalty inside the step is not a charge on the wealth.

### A Held Step is a success that records its hold

A Held Step's read-out carries `OptimisationSuccess` whose `res` records the hold — the
programme's termination status and the row's timestamp — and the warning still fires. The hold is
the family's own decision under a failed programme, so the fallback chain never runs on it, the
fold threads and drifts the held allocation, and ADR 0145's seams read it unchanged. ADR 0159's
sentence "the head's Result carries the retcode of the step that produced the Next-Period
Allocation" is rewritten in place to this. The head's `fb` runs on a genuine failure alone: a
non-finite allocation a rule cannot recover from.

### The batch pass is the papers', and the loop's sequencing follows the fields

Outside a fold loop `optimise(opt, rd)` is the Causal Pass of ADR 0155: rebalanced every row,
free. `predict(res, rd)` on its Result charges the head's fees with no drift, as it does for every
optimiser. The batch answer and a walk-forward's held path differ by the drift and by the Block
Step, and the head's docstring says so; the difference is documented, not tested.

`needs_previous_weights(head)` is the naive family's derived rule: true exactly when `fees` or
`fb` needs them. The online arm is sequential by construction; the batch walk-forward runs the
folds in parallel otherwise.

## Considered options

1. **The update reads the loop's `w_prev` when a source threads it.** Rejected: unreachable
   without a loop change, breaks the batch–online identity, and turns every rule into an algorithm
   no paper defines under a source.
2. **A flag anchoring the update at the drifted book, in-step.** Rejected: a Boolean that makes a
   family of unnamed variants with no bound; the executed-trade reading it would buy is the
   ceiling's and the fee's already.
3. **The ceiling against `w_t`, as ADR 0159 wrote.** Rejected: backwards for every member, as the
   example shows.
4. **A Previous-Weights Source switch on the Allocation Set.** Rejected: its `w_t` position is
   option 3, and a knob that cannot be right is worse than none; the walk-forward's switch governs
   the fee, which is the one place the loop measures.
5. **The head writes its own drifted book into `fees.tn.w`, ignoring `w_prev`.** Rejected: exact
   at `test_size = 1` only; at `k > 1` the fund's book is the block-start target drifted over `k`
   rows, which the head would re-derive from a buffer `rows_needed(alg)` may cap, while a
   walk-forward field went silently unread.
6. **Accept a `tn` fee under `pws = nothing` because every optimiser does.** Rejected by the
   maintainer: valid by the library's default, mispriced for every member of this family.
7. **Refuse any `tn` term on the head by the type bound `Fees{Nothing}`.** Rejected: expressible,
   and it forbids the one turnover fee the literature charges.
8. **A new Result type for the family.** Rejected: the type ADR 0158 withdrew when `pr` was its
   only reason; one keyword on `NaiveOptimisationResult` is non-breaking and serves the four naive
   heads too.
9. **The fee on the walk-forward.** Rejected: a second fee source beside `res.fees`, needing a
   precedence rule against a JuMP head that priced its own; a library-wide redesign no ledger row
   asks for.
10. **A Held Step with a failure retcode, ADR 0159 as written.** Rejected: the fallback replaces a
    hold that was the design, the fund and the recursion diverge, and ADR 0145's three seams need
    a family-specific case.
11. **A Held Step as a success recorded in the log alone.** Rejected: the Result could not say a
    hold happened.

## Consequences

- ADR 0159 is rewritten in place at the two sentences named above.
- `CONTEXT.md` gains *Price-Adjusted Allocation*; *Previous-Weights Source*, *Weight Drift* and
  *Held Step* say what they mean for the family.
- The head's first build,
  [#1161](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1161), owes: `fees` on
  the head and on `NaiveOptimisationResult`, the same field on the four naive heads with a fold
  test each, the entry refusal of a `tn` fee without a source, `needs_previous_weights` by the
  derived rule, and the docstring sentence on the batch pass. The projection build,
  [#1162](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1162), owes the ceiling
  against the Price-Adjusted Allocation and the Held Step's success record.
- The map's build sequence waits on no further decision of this ticket's; the evaluation surface
  is the last decision open.
