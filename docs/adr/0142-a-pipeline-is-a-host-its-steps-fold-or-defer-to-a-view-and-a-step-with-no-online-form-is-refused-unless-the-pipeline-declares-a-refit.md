---
status: accepted
---

# A Pipeline is a host: its steps fold or defer to a view, and a step with no online form is refused unless the Pipeline declares a refit

## Context

[Map #861](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/861) wants every layer
above the moments to take the online step, and the `Pipeline` is the outermost one. The fold loop's
online arm exists
([ADR 0140](0140-a-walk-forward-declares-its-fold-fit-and-the-online-arm-threads-the-estimator-from-a-cold-start.md)):
a walk-forward that declares `ff = OnlineStep()` warms one estimator up, folds each fold's new rows
into it, and hands the callback a `Fold` whose `train` is `nothing`, *the estimator holds its
window*. The optimiser takes that step
([ADR 0137](0137-an-optimiser-forwards-to-its-prior-alone-and-a-read-out-reconstitutes-the-carrier-and-runs-the-batch-path.md)):
`partial_fit!(opt, rd)` forwards to the prior alone, and `optimise(opt)` with no returns rebuilds
the carrier from a Fold Context and runs the ordinary batch path, so every consumer above the
prior is batch by construction. The search takes it
([ADR 0141](0141-a-search-scores-every-candidate-through-the-one-fold-loop-online-and-batch-alike.md)).
The Pipeline refuses it at its three doors by name, pending
[#872](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/872), which decided what a
fold with no training window means to a Pipeline's `fit`.

A Pipeline differs from an optimiser in one respect: **steps run before the row owner and change
the rows it folds.** Its preprocessing steps and its selectors are fitted on the window and
replayed, and the batch expanding walk-forward refits every one of them every fold
(`fit_and_predict(pipe, data; train_idx, …)` calls `fit(pipe, data_train)`, `03_Pipeline.jl:983`),
so the universe a selector picks and the values a fill writes both move fold to fold. The oracle
of the map is that the online run equals that batch run fold for fold. Four measurements shaped
the decision.

1. **What each fitted state does when refitted over one more row** sorts the steps into three
   classes. `PricesToReturns` reads rows `t - 1` and `t` and holds nothing; `PriceGapFill` with
   `CarriedPrice()` or a constant seeds `obs[end]` or the constant and carries the last observed
   price forward within the span (`08_PriceGapFill.jl:260`, `:300`); `MissingDataFilter`'s row
   filter at its default `row_thr = 1.0` drops nothing. These are **row-local**: a longer window
   leaves every earlier row's transform unchanged. `MissingDataFilter`'s column filter fits `nx`,
   the columns whose missing share is at most `col_thr` (`09_PriceFilters.jl:116`), and the three
   asset selectors fit `nx`, the columns kept (`13_AssetSelection.jl:173`). These are
   **universe-only**: a longer window changes which columns survive and nothing else.
   `PriceGapFill` with `MeanValue()`, `MedianValue()` or a function fits a per-asset statistic
   over the window (`:263`) and re-prices every earlier gap when the window grows; `MissingDataFilter`
   with `row_thr < 1` drops a price row whose missing share *among the fitted columns* exceeds
   the threshold, so a row kept at `t` is dropped at `t + 1` when a column leaves, and the returns
   either side of it move. These are **window-valued**: no state folded at `t` can be corrected,
   so no exact online form exists.
2. **A populated state is viewable by asset.** `test_08r_partial_fit.jl:240` pins
   `cov(port_opt_view(cv, i)) == cov(Covariance(), X[:, i])` on a state folded over every column,
   and `ReturnsBufferState`, `PriorCarryState` and `SampleBufferState` each carry
   `port_opt_view`. A state fitted on the full universe, viewed to a column set and read out, is
   the batch fit over that column set — which is what
   [#866](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/866)'s *a state lives on
   the full universe and never re-slices* needs to serve a universe that moves.
3. **The optimiser's `pe` is dead configuration in a Pipeline with a prior step.**
   `inject_context` routes `ctx.prior` into the optimiser's `pe` before the step runs
   (`03_Pipeline.jl:629`), so the optimiser's own prior is never fitted there. The row owner of a
   Pipeline is therefore not fixed by the optimiser's type but by the steps.
4. **The reference refuses every Pipeline.** skfolio raises
   `TypeError("Pipeline is not supported")` in `online_predict`, `online_score` and
   `OnlineGridSearch` (`model_selection/_online/_validation.py:862`), because scikit-learn 1.9.0's
   `Pipeline` defines no `partial_fit`; no selector of the reference has one; and the width of
   every online estimator is pinned at the first call. A caller of the reference cannot run a
   Pipeline online, and the reference never meets a re-selection between two steps. Any route here
   adds capability and removes none.

Two earlier rulings of the map pull apart on the window-valued class.
[#704](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/704): *a host folds every
member that folds and runs the batch verb over its own rows for every member that does not, so a
caller writes the estimator they would write in batch.*
[#870](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/870): *refused at the door by
name rather than run as a refit in silence, because a caller who declared the step would otherwise
read a batch answer as an online one.* And
[#865](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/865) fixed that *the `Online`
wrapper is the declaration that seeds a refit buffer*, whose cap
[#997](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/997) made the window.

## Decision

**A Pipeline is a host.** `partial_fit!(pipe, data::Prices_RR)` walks the steps in order, hands
each the new rows and takes back what it emits, and returns the rebuilt Pipeline, as every host
does. `fit(pipe)` with no data reads it out, mirroring `optimise(opt)`, and
`fit_and_predict(pipe, data; train_idx = nothing, test_idx)` is the fold loop's door. The read-out
returns an ordinary `PipelineResult`, so `predict`, `assert_universe_aligned` and the search
consume it unchanged. Its arity mirrors the batch verb's: `fit(pipe, data)` takes a carrier, and
so does the step.

**The row owner is found by a walk, not chosen.** It is the prior step when there is one, because
`inject_context` overrides the optimiser's `pe` with it; else the optimiser step, which is then a
host of ADR 0137 and carries its own Fold Context; else a prior-less head, the bottom of its own
chain. The Pipeline holds a Fold Context of its own — `cache::ReturnsBufferState`, the benchmark
and timestamp columns and the static panel — **only when a prior step owns the rows**, because
then no host below it holds them. The Fold Context takes the owner's cap, as ADR 0137's do.

**Each step before the row owner takes the step in the form its class allows.**

- A **row-local** step folds. `PricesToReturns`, `PriceGapFill` and `MissingDataFilter` gain
  `cache::Option{<:AbstractPartialFitState} = nothing` and a `partial_fit!` of their own, and read
  out to today's Results. `PricesToReturns` keeps the last price row, and under `CatchUpGapReturn`
  the last observed price per column, and emits the new return rows exactly; `PriceGapFill` keeps
  the carry per column and emits the filled rows; `MissingDataFilter` keeps the missing count per
  column and the row count, passes every row through, and defers its column filter to the
  read-out. The state lives on the step, as ADR 0106 places every Partial Fit State, so a step's
  online form is tested alone against its batch fit as every moment's is, and a caller's own
  preprocessing estimator joins the host route by writing the same two methods.
- A **universe-only** step folds nothing and defers. A selector's `partial_fit!` is the identity
  on the rows. At read-out the step's batch verb runs over the owner's carried rows, its Result is
  applied to `ctx.returns` as today (`apply_fitted_step`), and the *owner's state* is viewed to
  the columns that survive through `port_opt_view` before it is read out. The rows are therefore
  folded on the Pipeline's input universe at warm-up width, and a universe that moves between two
  steps is expressed as a view, never by re-slicing a state and never by freezing the selection.
  A selector re-ranks per step exactly as the batch loop re-fits it per fold.
- A **window-valued** step, or any step that writes a Data Slot before the row owner and has no
  online form — `PriceGapFill` with a statistic fill, `MissingDataFilter` with `row_thr < 1`, a
  callable `PipelineStep` writing `:prices` or `:returns`, a caller's preprocessing estimator
  without a `partial_fit!` — **is refused at warm-up by name**, and the message names both
  routes: give the step a `partial_fit!`, or declare a refit with `Online(pipe)`. The predicate is
  `supports_partial_fit`, reused by its readers' meaning, which reads a type and a field; the two
  library configurations answer `false` by the fill's type and by `row_thr`.

**`Online(pipe; max_history = w)` is the declared refit.** It seeds an input-carrier buffer into
`pipe.cache`, folds no member, and reads out by `fit(pipe, buffer)`, so it is exact for every
configuration at batch cost, and with a cap it is a rolling Pipeline online, equal to the rolling
batch walk-forward. The state's type is the route: a `ReturnsBufferState` in `pipe.cache` is the
host route's Fold Context, an input-carrier buffer is the refit route. This is one stated exception
to ADR 0137's *`Online` wraps the prior, never the host*: the reason there was two copies of the
rows under two caps, and under the refit route no member folds, so there is one copy and one cap.
An `Online` member below an `Online(pipe)` is refused by name, extending `Online(::Online)`'s
refusal (`15_Online.jl:1289`) to the tree, because a wrapper below the refit has no fold to seed
and its cap would be silently ignored. An `Online(pe)` inside an **unwrapped** Pipeline is the
ordinary host-route case: the owner's window is the Pipeline's, and its Fold Context takes the
owner's cap.

A plain Pipeline handed a scheme with `ff = OnlineStep()` takes the host route with no wrapper; the
refusal reaches the window-valued configurations and the unknown data-slot steps alone.

**The `PipelineContext` is not threaded.** The read-out is a fit, and the context is that fit's
blackboard, built per read-out; the ticket's third question collapses.

### What the step refuses

- A state at entry anywhere in the steps — the Pipeline's `cache`, a step's `cache` — by name,
  `online_entry_state` walking the steps. The loop starts cold (ADR 0140).
- A `TimeDependent` optimiser step **only when the optimiser is the row owner**, at warm-up, as
  `assert_stateless_schedule` refuses a schedule on a stateful field one layer down. When a prior
  step precedes it, the schedule swaps a stateless step and composes: the prior step owns the
  rows, the Pipeline holds the Fold Context, and the optimiser's `pe` is overridden. The
  `PricesToReturns → EmpiricalPrior → TimeDependent([…])` Pipeline of the time-dependent example
  runs online unchanged.
- A window-valued step and an unknown data-slot step, as above.
- An `Online` member below an `Online(pipe)`, as above.
- A `TrainTestSplit` and a finite allocation, as today: `assert_no_holdout` refuses the first at
  every cross-validation door, and the arm's type bound excludes the second.

A nested Pipeline step recurses: its steps are steps, and a nested prior step is the outer's row
owner. A callable or a constraint, uncertainty-set or phylogeny step writing a derived slot runs
at read-out over the reconstituted context, holds no state and is refused nowhere.

### The identities the build pins

1. `partial_fit!` over rows `1:t` then `fit(pipe)` equals `fit(pipe, data[1:t])` step for step —
   the filter's and the selector's `nx`, the fill's seeds, the prior — and in weights at
   ADR 0137's tolerances.
2. Each row-local online form alone against its batch fit, exactly: the conversion, the carried
   fill, the missing counts.
3. A re-selection between two steps: a selector that picks `{A, B}` at step `t` and `{A, C}` at
   step `t + 1` reaches the batch fold's weights, and the state was never re-sliced.
4. `IndexWalkForward(w, t; ff = OnlineStep())` over a gapped panel with a listing and a delisting,
   through a JuMP optimiser and a hierarchical optimiser, and with a `TimeDependent` optimiser
   step after a prior step, equals `expand_train = true` fold for fold — the map's closing test,
   through the Pipeline.
5. `Online(pipe; max_history = w)` equals the rolling batch walk-forward with warm-up
   `w + purged_size`, and `Online(pipe)` equals the unwrapped host route.
6. Every refusal, by message.
7. The Pipeline's search door lifts its refusal and picks the batch candidate (ADR 0141's
   identity).

## Considered options

1. **The Pipeline is a host (chosen).** Row-local steps fold, universe steps defer to a view, the
   row owner folds, the read-out runs the batch tail. The destination holds in the Pipeline as in
   every other layer, the moments fold at `O(N²)`, and a schedule composes after a prior step.
   Costs three per-type online forms at the price level and a `cache` on `Pipeline`.
2. **The Pipeline is a refit member.** `partial_fit!` appends the input carrier to one buffer and
   `fit(pipe)` is `fit(pipe, buffer)`. Exact everywhere with no census, at batch cost for every
   step. Rejected as the *only* route because it makes the Pipeline the one layer where the
   destination's rule does not hold — `Pipeline(PricesToReturns(), EmpiricalPrior(), MeanRisk())`
   would refit the prior every step while `MeanRisk(; pe = EmpiricalPrior())` folds it — and it
   holds two copies of the rows when the prior carries. It survives as the declared route,
   `Online(pipe)`.
3. **Freeze the selection at warm-up.** Diverges from the batch expanding walk-forward, which
   refits the selector every fold; out by the oracle. The glossary's *Universe Policy* replays a
   fitted universe from a fold's training window onto its test window, not from one fold onto the
   next.
4. **Re-slice the state to the new selection.** An asset that joins has no history; out by #866.
5. **Fall back to the refit automatically** for a window-valued step. #704's host rule verbatim,
   and nothing refuses. Rejected because a caller who declared `OnlineStep()` would read a batch
   refit under an online banner, which #870 refused by name; because the Pipeline would become a
   second row owner unasked; and because a caller who wants a window needs `Online(pipe)` anyway,
   so the automatic switch is the declared route plus a hidden one.
6. **Refuse the window-valued class with no refit route.** A predicate and a message, no price
   buffer, no `Online(::Pipeline)`. Rejected because two configurations of released steps would
   have no online step at all, and a search grid holding `PriceGapFill(MeanValue())` would refuse
   a candidate at its warm-up.
7. **One Pipeline-level state** holding every step's replay values beside the Fold Context.
   Rejected because the Pipeline would hold state that belongs to its members and know each
   member's internals — the leak ADR 0106's field bound exists to prevent — and a caller's own
   step could not join without a method on the Pipeline.
8. **Fold through an unknown data-slot step as if row-local**, applying the warm-up's fitted
   object to the new rows. Rejected as silently wrong for a window-valued transform and as the
   freeze of option 3 in another place.

## Consequences

- `PricesToReturns`, `PriceGapFill`, `MissingDataFilter` and `Pipeline` gain a `cache` keyword
  defaulting to `nothing`; no constructor call and no doctest moves, `show_fields` rendering it
  only where set. `fit(pipe, data)` is untouched. No released number moves: the route is reached
  only through a scheme's `ff = OnlineStep()`, which the doors refuse today.
- The three Pipeline doors — `cross_val_predict`, its `MultipleRandomised` form and
  `search_cross_validation` — drop `assert_batch_fold_fit`, and `assert_online_entry(::Pipeline)`
  becomes the walk over the steps.
- A capped owner (`EmpiricalPrior(; max_scenarios = w)`) makes a selector rank over `w` rows at
  read-out, which is the divergence the Scenario Cap already documents (ADR 0136); the Pipeline
  adds no rule.
- `CONTEXT.md`: *Pipeline* states its online form, *Fold Context* names the Pipeline as a holder,
  and *Sample Buffer* names the input-carrier buffer `Online(pipe)` seeds.
- The build is one ticket of map #861, blocked by nothing open; the Pipeline's search door lifts
  its refusal in the same build, after
  [#1020](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1020) lands the
  optimiser's.
