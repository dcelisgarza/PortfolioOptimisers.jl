---
status: accepted
---

# An optimiser forwards to its prior alone, and a read-out reconstitutes the carrier and runs the batch path

## Context

[Map #861](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/861) lifts every layer above
the moments onto the online seam. [ADR 0106](0106-a-partial-fit-state-is-the-one-result-an-estimator-holds.md)
fixed where a Partial Fit State lives,
[ADR 0107](0107-the-update-seam-has-two-verbs-and-a-view-slices-by-asset-and-drops-by-observation.md)
fixed its two verbs, and
[ADR 0136](0136-a-prior-folds-and-carries-the-buffer-is-owned-once-and-a-cap-is-either-a-scenario-cap-or-a-window.md)
put the prior family on it: a prior folds and carries, and the returns are owned once, at the
bottom of the chain. [Issue #867](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/867)
decided the optimiser layer, and
[issue #1007](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1007) built it.

An optimiser is the layer where the seam meets everything that is not a moment. A `JuMPOptimiser`
holds a prior, a solver, weight bounds, budgets, cardinalities, turnover, fees, tracking, `N`
return terms and `M` risk measures each with its own uncertainty set; a `HierarchicalOptimiser`
holds a clustering estimator beside its prior; a meta-optimiser holds inner optimisers of its own.
Two facts about that layer decided the shape. `_optimise(sr::SubsetResampling, rd)` hands each
subset `port_opt_view(rd, idx)` — a view of the **caller's `ReturnsResult`**, not of the outer
prior's result — and every meta-optimiser does the same. And every `UniverseSets` constraint
resolves **by asset name**, which lives in `rd.nx`; a `LowOrderPrior` carries `X` and no names. A
read-out that handed the batch path a bare matrix would hand its inner optimisers an empty panel
and its constraint estimators no names.

### What the reference does

Its optimiser's `partial_fit` folds *and* solves in one call — its own docstring says the problem
is solved fresh on each call — so a warm-up of `T` observations costs it `T` solves, and a failed
solve strands an estimator whose state has advanced past the last answer it can give, which is why
it refuses an estimator chain as a fallback and falls back to the previous weights alone. Its
observation flows straight into `prior_estimator.partial_fit(X, y)`, and everything downstream
reads the buffer the prior result carries; the names ride on the estimator, pinned at the first
call.

### What #968 left on the table

Issue #867 §3 specified one `ReturnsBufferState` holding every per-observation column of the carrier,
carried by `EmpiricalPrior` as pass-through columns it never reads. #968 landed before that
correction and built the prior's states without it — a `PriorCarryState` holding the rows, a
`SampleBufferState` for a refit, a `FactorSampleBufferState` for a prior whose batch verb requires
`F` — and the prior's step keeps the arity of its batch verb, `partial_fit!(pe, x)` or
`partial_fit!(pe, x, f)`, with no slot for a benchmark or a timestamp. So the pass-through columns
have no home at the prior layer, and the state #867 named has to live somewhere else.

## Decision

### Two verbs and one forward

`partial_fit!(opt, rd)` folds the observations of a carrier and returns the estimator; it solves
nothing. `optimise(opt)` with no returns reads the state out and solves once. The step **forwards
the observations to the prior alone**, in the prior's own arity — `X` and `F` to a prior that
requires factor returns, `X` to every other, and the active mask of a time-varying Asset Panel as
the keyword the prior's step takes — and to nothing else. Every optimiser has a `pe`, the three
meta-optimisers included, so one named forward per family reaches the whole library and the rule
for every optimiser added later is stated once: **an optimiser forwards the observation to its
prior and to nothing else, and everything it holds beside the prior takes its ordinary batch fit at
read-out, from the reconstituted carrier.**

The arity mirrors each layer's batch verb. The optimiser's is `optimise(opt, rd::ReturnsResult)`,
so its step takes a carrier — one row for a step of a walk-forward, a block for a warm-up — and
unpacks it on the way down. A JuMP head forwards to the `JuMPOptimiser` it holds and a
hierarchical head to its `HierarchicalOptimiser`, because the bundle is what holds the prior.

### The read-out reconstitutes the carrier and runs the batch path

`optimise(opt)` rebuilds a complete `ReturnsResult` from the state, swaps the folded prior for its
read-out `prior(pe)` — a Prior Result, which the batch path does not refit, so the solve reads the
prior's exact fold and the step stays quadratic in the assets — drops the state, and calls
`optimise(opt′, rd′)`. The clustering estimator, the constraint estimators, all `N + M` uncertainty
sets and a meta-optimiser's inner optimisers are therefore identical to batch **by construction**,
from one carrier and with no state of their own. No uncertainty set gains a method; the reference
reaches the same place by recomputing its one online set from the prior result on every call.

The read-out is **pure**. The state moves at `partial_fit!` and nothing else writes it, so
`optimise(opt)` is callable any number of times for the same answer, and a failed solve leaves the
state where the last fold put it. The ordinary fallback chain therefore walks unchanged: each
fallback is handed the reconstituted carrier and fits from it as it would from the caller's, and
`factory(res, fb)` records the chain. The reference's refusal of an estimator chain rests on a
premise — a state advanced past the last answer it can give — that splitting the verbs removed.

The fold-less entry `optimise(opt)` is now one method on the root, chosen by dispatch on the
host's `cache`. A host that has taken a step is read out; a host whose prior is already a result
is handed back with an empty carrier, which is the entry it has always had; any other host is
refused by name. The nine per-family `optimise` signatures that defaulted `rd` to an empty carrier
lose the default, because a default argument is a second one-argument method that would shadow the
read-out.

### The Fold Context lives on the host, and the returns stay owned once

The state #867 named is a `ReturnsBufferState`, and it lives in a `cache` field on the eight hosts
that hold a prior or read the observations directly: `JuMPOptimiser`, `HierarchicalOptimiser`,
`InverseVolatility`, `NestedClustered`, `Stacking`, `SubsetResampling`, `EqualWeighted` and
`RandomWeighted`. It carries what the prior does not: the factor and benchmark columns as
`SampleBufferState`s of their own, a single-column benchmark and the timestamps as plain vectors,
and the context pinned by the first step and checked at every step after it — `nx`, `nf`, `nb` and
a static Asset Panel. The returns stay **owned once, by the prior**; the context reads them out of
the prior's own buffer through `prior_returns_buffer`, which walks a wrapping prior down to the
prior that holds the rows. Only `EqualWeighted` and `RandomWeighted`, which hold no prior but
derive the Coverage Universe of their own window, keep the rows in their context, because with no
prior beneath them they are the bottom of the chain and ADR 0136's rule is unbent.

The context takes its cap from the prior's buffer, so a wrapper's `max_history` windows the whole
carrier and the read-out's rows line up with the prior's. `Online` therefore wraps the **prior**,
never the host; wrapping a host that holds a prior is refused by name, and the two prior-less
heads take the wrapper and seed a context carrying its cap.

A time-varying Asset Panel is never pinned. Its active mask rides into the returns buffer beside
the rows it explains — the prior's buffer, or the head's own — and the read-out rebuilds the panel
from it. A static panel is context and is pinned like the names.

### What the step refuses, and why

Six refusals, each an `ArgumentError` naming the field or the type.

- **A finite allocation.** `DiscreteAllocation` and `GreedyAllocation` convert a weight vector
  and prices into share counts and read no returns window, so they have no online step and need
  none; the message names `optimise(da, w, p)`.
- **A `TimeDependent` on `pe`, and a schedule of optimisers.** A schedule swaps the estimator that
  carries the state, and a member that never saw the folded rows cannot be handed them. A fold
  loop resolves the schedule before it steps. Every *other* schedule passes: `wb`, `bgt`, `lt`,
  `card` and their siblings hold values, and the read-out runs the batch path with whatever the
  schedule currently holds.
- **A Prior Result in `pe`.** It has no state to fold into; it is batch configuration, and the
  fold-less entry still serves it.
- **An implied-volatility surface.** The prior's step folds the returns and the factors alone, so
  a step carrying `iv` would fold a covariance that reads the surface without it and read out an
  answer a batch fit would not give. The refusal is loud where the alternative was silent.
- **A time-varying Panel Field.** A time-varying field's rows are sample, and the seam's buffers
  hold numbers, so the step has nowhere to keep a categorical or a tensor field and the read-out
  could not rebuild the panel. The masks-only panel of the ingestion layer passes.
- **An estimation mask narrower than the active mask.** The exact folds of the moment layer take
  an active mask and no estimation mask, and only the two regime-adjusted families read one, so an
  estimation universe narrower than the active one cannot be honoured online. The reconstituted
  panel carries the active mask as both.

**No frontier refusal.** A frontier read-out is an ordinary frontier over the observations folded
so far, which is meaningful; what has no meaning is turnover *across* those frontiers, and that is
[#1004](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1004)'s, in batch and online
alike. **No previous-weights threading.** The loop threads them and the step does not, which is the
reference line for line and correct today for every non-frontier optimiser.

## Considered options

1. **Widen the prior's state to the `ReturnsBufferState` #867 §3 drew** — one state, on the
   prior, holding every column. Rejected because #968 had landed three prior states with the
   prior's batch arity, and the prior's step has no slot for a benchmark or a timestamp: the
   pass-through columns would have to travel a verb whose signature is `prior(pe, X, F, pnl)`, and
   every one of the three states, every wrapper seed and every family that narrows its `cache`
   would have to move.
2. **Hand the batch path a bare matrix at read-out.** Rejected by the two facts in the context:
   the meta-optimisers view the caller's carrier and the constraints resolve by name.
3. **Fold and solve in one verb, as the reference does.** Rejected: a warm-up of `T` observations
   is `T` solves, and a failed solve strands the state, which forces the reference's refusal of
   an estimator chain.
4. **The context on the head rather than on the bundle** — `MeanRisk.cache` rather than
   `JuMPOptimiser.cache`. Rejected: fourteen hosts instead of eight, and the bundle is what holds
   the prior the step forwards to.
5. **Forward the estimation mask, and let a plain member refuse it.** Rejected: the refusal would
   be a `MethodError` from the moment layer on the first row where the masks differ, mid-run.

## Consequences

- The seam's shape holds at a third layer, unbent: `cov(ce, X)` / `partial_fit!(ce, x)` … `cov(ce)`;
  `prior(pe, rd)` / `partial_fit!(pe, rd₁)` … `prior(pe)`; `optimise(opt, rd)` /
  `partial_fit!(opt, rd₁)` … `optimise(opt)`. After `t` observations, `optimise(opt)` equals
  `optimise(opt, rd[1:t])` — exactly for the carrier the read-out rebuilds, and to the moment
  layer's own tolerance for the weights. `test/test_24b_optimiser_partial_fit.jl` pins the three
  identities of #867 §10.
- Eight hosts gain a `cache` field that `show_fields` hides, so no doctest moves. `factory` carries
  it and `port_opt_view` slices it, because a fold loop hands a folded optimiser through both.
- The `pe` slot of the six prior-holding hosts admits an `Online`, through the `Onl` alias ADR
  0136 introduced for the prior hosts, and the optimisers write the `update_online_estimator`
  recursion that reaches it; the two wrapping priors write the same recursion for the prior they
  embed, so a wrapper two levels down is seeded at warm-up.
- `iv`, a time-varying Panel Field and a narrowed estimation mask do not travel the step, and each
  is refused by name. Carrying any of them is a decision for a later ticket of map #861, and the
  refusals mark where it lands.
- A factor prior under `Online` holds `F` twice: once in its own pair of buffers, once in the
  context, so that the read-out's `rd′.F` comes from one place whatever prior sits beneath it. The
  duplicate is a matrix of factors, not of assets, and it is the price of one uniform
  reconstitution.
