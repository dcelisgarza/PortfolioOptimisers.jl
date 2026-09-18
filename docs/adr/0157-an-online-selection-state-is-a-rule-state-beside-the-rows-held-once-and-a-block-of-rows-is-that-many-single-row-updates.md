---
status: proposed
---

# An online selection state is a rule state beside the rows held once, and a block of rows is that many single-row updates

## Context

[ADR 0155](0155-an-online-portfolio-selection-head-is-a-naive-optimiser-whose-batch-verb-is-a-causal-pass-and-whose-read-out-is-its-own-recursion.md)
fixed the shape of online portfolio selection in this library: one head,
`OnlinePortfolioSelection <: NaiveOptimisationEstimator`, the update rule on `alg`, a state
`OnlinePortfolioSelectionState` on the head's `cache` holding the allocation `w_t` and the
rule's private carriers, a batch verb that is the Causal Pass, and a read-out that is a
Recursion Read-out. It left to
[issue #1152](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1152) the state's
contract: the one verb every rule writes, the state's field list, what a fold of several rows
means, what a step does with a non-finite return, how the state answers a view by
asset, and what pins the Fold Context a `Resume` reads.
[ADR 0156](0156-every-online-selection-algorithm-ships-as-a-closed-form-rule-an-expert-mixture-or-a-follow-the-leader-over-a-selected-sample.md)
added the constraint that the state must admit a carrier that is a window or a prefix of rows,
because every follow-the-leader and pattern-matching rule re-solves on past rows and every
reversion rule forecasts from a window of them.

Four facts from the two ledgers and the code shaped the decision.

- **The fold loop folds blocks and views once.** `thread_online_folds!` folds each fold's new
  rows as one `partial_fit!(est, rows)` and reads out **one** target held over the fold's
  `test_size` rows. Under a multiple-randomised path the estimator is viewed at the path's
  asset subset **once, before the warm-up**, so a live state is never re-viewed by the loop;
  the one live view is `investable_reduction` at the read-out, which views the head — and
  through `@vprop cache` its state — at the Investable Mask.
- **A window of price levels is a window of rows.** The reversion forecast `x̂ = MA(P) / P_t`
  is invariant to scaling each asset's levels by a constant, so `window − 1` price relatives
  reconstruct it exactly; the prototype's `T + 1 × N` matrix of levels is a convenience of a
  closure, not a carrier the rule needs.
- **The library has the buffer.** `SampleBufferState` holds rows verbatim with a cap and
  answers `copy` and `port_opt_view`; it is what the `Online` wrapper seeds and what a Prior
  Result carries.
- **A missing return already has a meaning.** The Held Gap rule zeroes a missing return at a
  held weight once — the position sits in cash — warning by default and refusing under
  `strict`; and `step_active_mask` refuses `iv`, a panel with fields and `emsk ≠ amsk` at
  every host's step. Every paper assumes a full finite matrix and rebalances every period.

## Decision

### One verb: `(st′, w′) = online_update!(alg, st, w, x, rows)`

Every rule writes one method. It takes the rule, the rule's private carrier `st`, the
allocation `w` held during the period, the finite price relative `x` of that period, and the
rows the head holds through that period; it answers the carrier and the allocation for the
next period. The carrier's arrays are written in place where the rule can, and `nothing` is a
legal carrier, so a rule that carries nothing — buy-and-hold, the constant rebalanced
portfolio, exponentiated gradient, passive-aggressive reversion — needs no mutable object.
`w′` is always a new vector, because the projection allocates one, so a `w` that is a view
after `port_opt_view` is never written. This is `partial_fit!`'s own promise
([ADR 0107](0107-partial-fit-bang-is-the-cheapest-exact-fold-and-partial-fit-is-value-semantic.md)):
in place where it can, the value returned is the truth.

### The state is a Rule State beside the rows held once

`OnlinePortfolioSelectionState` holds:

| field | what it is |
| :--- | :--- |
| `w` | the allocation held during the current period, over the full pinned universe |
| `st` | the rule's private carrier, or `nothing` |
| `X` | the rows any rule of the tree reads, a `SampleBufferState` of returns, or `nothing` |
| `nx`, `pnl` | the asset names and the static Asset Panel, pinned by the first step |
| `ts` | every timestamp folded, in order, unbounded |

The pair `(st, w)` is a **Rule State**, and it is the unit the family recurses over: the head
holds one, and an Expert Mixture's carrier is a vector of Rule States — one per expert — beside
the Rule State of its weighting, whose `w` is the weight vector `p` over the experts and whose
`st` is the weighting's own carrier (nothing for buy-and-hold, a `K × K` Gram for the Newton
weighting). The mixture's update is the head's recursion applied twice.

The rows are held **once, on the head**, and never inside a rule's carrier. A per-rule fact,
`rows_needed(alg)`, declares how many: `0` for a carrier-free or Gram-carrying rule, `window − 1`
for a reversion over `window` price levels, `W` for `LastRows(W)`, `2 · window` for
anti-correlation, unbounded for a prefix and for every pattern-matching selector, and the
maximum over the experts for a mixture. The head sizes `X`'s cap from that answer and pushes
`x` before it calls the rule, so the rows the rule reads run *through* the current period. A
mixture of thirty pattern-matching experts over two thousand rows therefore holds the rows once
where a per-rule carrier would hold them thirty times. This rewrites ADR 0155's sentence *a
rule's private carrier may be the rows it re-solves on* to *the rows any rule of the tree reads
are held once on the head's state*; ADR 0155 is a draft on `dev` and is rewritten in place.

The rows buffer and the timestamps are two things because they serve two readers: the rows
serve the rule and are capped by its need, the timestamps serve
[ADR 0144](0144-an-online-runs-result-carries-the-threaded-estimator-and-resume-re-enters-the-fold-loop-from-the-folds-it-holds.md)'s
`Resume` and are never capped, so a carrier-free rule still resumes. `held_timestamps(head)`
answers `cache.ts`, the pin-and-check reuses `assert_pinned_context`, and nothing in
`15_Resume.jl` moves.

### A block of rows is that many single-row updates

`partial_fit!(head, rd)` applies the Online Update to each row of `rd` in order, and so does
the Causal Pass; the state never learns the loop's cadence. A fold of `k` rows is `k` Online
Updates followed by one read-out, so a walk-forward with `test_size = k` holds the Next-Period
Allocation as of the block's end for `k` periods and the `k − 1` allocations the state formed
inside the block are computed and never held. This is the **Block Step**, and it keeps ADR
0155's identity — `optimise(opt)` after folding rows `1:t` equals `optimise(opt, rd[1:t])` —
exact at every `test_size`, because both arms take the same `t` single-row updates. A user who
wants a rule that genuinely rebalances weekly resamples the returns to weekly and runs
`test_size = 1`: the cadence is the data's, never the loop's. Whether the recursion's `w` should
read the *held* (drifted) weights mid-block is the start-weights, drift, turnover and fees
ticket's question.

### A non-finite return is filled with zero, once, before the buffer and the rule

The head fills every non-finite cell of the row `r` with `0` — the Held Gap's own number, the
position sits in cash — before it pushes the row into `X` and before it forms the price relative
`x = 1 .+ r` the rule reads
([ADR 0162](0162-an-online-selection-head-buffers-returns-and-starts-from-a-given-allocation-or-a-uniform-one-over-the-pinned-universe.md)).
A cell outside the panel's active mask is filled silently, because an inactive asset has no
return by definition; a cell inside it is a Held Gap and is named through the head's `strict`: a
warning by default, a refusal under `strict`. The rule and the rows buffer therefore see finite
rows over the full pinned universe, a re-solve on the rows sees a zero return at the gap exactly
as `predict` scores one, and masks act at the read-out and never inside the recursion. `step_active_mask`'s
three refusals — `iv`, a panel with fields, `emsk ≠ amsk` — are reused verbatim.

A consequence is that the state's `w` never carries a forced zero. An asset that is unlisted
for a span earns what a cash-like leg earns under the rule, and when it relists its weight is
the recursion's own — not zero, which a multiplicative rule would hold forever, and not `1/N`.
No re-entry policy exists anywhere in the family.

### A view slices every per-asset axis and renormalises every allocation

`port_opt_view(state, i)` is the state of the same observations over the selected assets: `w`
sliced and renormalised, `X` through `SampleBufferState`'s own view, `nx` and `pnl` viewed, `ts`
copied, and `st` forwarded to `port_opt_view(alg, st, i)`, which the rule writes — a Gram slices
to `A[i, i]` and `b[i]`, a mixture slices every expert's Rule State by the same verb and keeps
`p` unchanged — and whose fallback refuses by name, so a rule whose carrier cannot slice is
honest. The read-out's Investable Mask is the panel's active mask at the last folded timestamp,
`nothing` under a static panel; `investable_reduction` takes this view, the reduced `w` is
finalised, and `NaiveOptimisationResult` expands it with a zero at every non-investable asset.
The Causal Pass reduces by the same last-row mask, so the identity holds under a time-varying
panel too. The recursion keeps the full `w`: the view is a copy for the read-out.

### `merge_states` refuses uniformly, and `copy` is deep

ADR 0155's refusal stands and is one method on the head's state, not one per rule: even a
carrier that is a sufficient statistic for its block — a Gram sum, a prefix of rows — sits
beside an allocation that is not, and the family's parallel route is the Causal Pass, which is
`O(N)` a row. `copy` deep-copies `w`, `st`, `X` and `ts`, aliasing no array, so the value form
`partial_fit` and `copy_states` under `Resume` rest on it.

## Considered options

1. **`w` inside every rule's state, `online_update!(alg, st, x)`.** Rejected: some twenty-five
   state types each repeating `w`, the head's `w` a forward to `st.w`, and a carrier-free rule
   still needing a mutable struct.
2. **`w` written in place, the verb returning `st` alone.** Rejected: the projection allocates
   and then copies; a `w` that is a view cannot be written; the mixture must pre-allocate every
   expert's `w`.
3. **Each rule's carrier holds its own rows.** Rejected: a mixture duplicates the rows per
   expert, and every windowed rule re-implements a ring.
4. **Price levels in the reversion rules, rows in the solved rules**, as the prototype has.
   Rejected: two carrier shapes for one thing, and a `cumprod` of levels drifts numerically
   over a long run where a window of rows does not.
5. **One update per block on the compounded price relative.** Rejected: a different algorithm
   per rule at `k > 1` — weekly exponentiated gradient is not exponentiated gradient on weekly
   data unless the gradient is redefined — a `block` field on the head to keep the identity,
   and the paper's regret bound no longer applying as stated.
6. **Refuse `test_size > 1` by name.** Rejected: the head could be evaluated only at its
   data's cadence, so a turnover-and-fees comparison against a monthly `MeanRisk` would need
   resampled data on both sides.
7. **Refuse a non-finite cell by name.** Rejected: no point-in-time universe would reach the
   family, and one gap in a long history would refuse a whole run.
8. **Run the update on the row's active sub-universe and expand with zeros.** Rejected: a
   per-rule re-entry policy (every multiplicative rule holds `0` forever, the Newton step and
   the additive reversions re-enter by themselves), and every carrier slicing and re-growing
   as the universe changes size between steps.
9. **A view keeps `w` and drops the carrier.** Rejected: a step after a view would restart the
   recursion, so the view would not be the same observations over the selected assets, which
   every other state's view is.
10. **Refuse the view for any rule with a carrier.** Rejected: the read-out under a
    time-varying panel would fail for the Newton step, every reversion and every mixture.
11. **Cap `ts` with the rows buffer.** Rejected: a carrier-free rule would hold no timestamps,
    so `Resume` would refuse it unless the cap grew a floor that names nothing.
12. **No timestamps; `Resume` refused for the family.** Rejected: a deployment could never
    re-enter the fold loop, and ADR 0155's `Resume` sentence would be withdrawn.
13. **`merge_states` per rule**, answered where the carrier is sufficient. Rejected: the `w`
    beside it never is, and the head's uniform refusal is one method.

## Consequences

- ADR 0155's state paragraph is rewritten in the one sentence named above.
- `CONTEXT.md` gains *Rule State* and *Block Step*; *Online Update* gains the rows argument;
  *Recursion Read-out* and *Sample Selector* say the rows are held once on the head.
- The Prior-slot ticket meets a head whose rows buffer already holds what a price-level
  forecast reads, so a forecast estimator over the rows is a reader of `X`, and `OLMAR-2`'s
  exponential average is a carrier of its own on `st`.
- The constrained-update ticket meets a `w′` that is always a fresh vector, so a projection
  in any geometry replaces the simplex projection in one place.
- The start-weights, drift, turnover and fees ticket decides whether the recursion's `w`
  reads the held weights mid-block; this ADR leaves it the recursion's own.
- The build tickets graduate: the head, its state and the verb; the rows buffer and
  `rows_needed`; the view and the read-out mask under a time-varying panel; the Block Step's
  identity test at `test_size > 1`; the fill rule's Held Gap warning and refusal.
- `test/test_24b_optimiser_partial_fit.jl` gains the identity at `test_size = 5`;
  `test/test_62_partial_fit_state_interface_census.jl` gains the state; the Held Gap tests
  gain the step's warning and refusal.
