---
status: proposed
---

# An online selection head buffers returns, and starts from a given allocation or a uniform one over the pinned universe

## Context

[ADR 0157](0157-an-online-selection-state-is-a-rule-state-beside-the-rows-held-once-and-a-block-of-rows-is-that-many-single-row-updates.md)
made the head's rows buffer `X` a `SampleBufferState` capped by `rows_needed`, filled a
non-finite cell with `1` before the push, and ruled that the allocation never carries a forced
zero and that a relisting asset re-enters at the recursion's own weight.
[ADR 0158](0158-a-forecast-reading-rule-holds-an-expected-returns-estimator-and-a-covariance-enters-on-the-constraint-that-reads-it.md)
made every forecaster an `AbstractExpectedReturnsEstimator` read through `mean(me, X, pnl)` as
`mu`, and
[ADR 0159](0159-a-constrained-online-update-projects-onto-an-allocation-set-in-the-rules-own-geometry-and-the-default-set-needs-no-solver.md)
fitted the Allocation Set's covariance and tracking error on the same rows. No ADR said where the
recursion's first allocation comes from. An audit of the map found the two ADRs fixing the
buffer's unit differently, and the start left to the reader;
[issue #1166](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1166) asks both.
Four facts measured on `dev` at `9f8b41d395` shaped the decision.

- **Every reader of the buffer but the rule reads returns.** `mean(me, X, pnl)` on any
  `AbstractExpectedReturnsEstimator`, `prior(pe, X, nothing, pnl).mu` through
  `PriorExpectedReturns`, the covariance a `ProgrammeAllocationSet`'s `pe` fits, the tracking
  error, and a follow-the-leader's `optimise(opt, rows)` are all written over returns. Handed
  price relatives, `SimpleExpectedReturns` answers `mu ≈ 1` and the Price Relative Forecast
  `1 .+ mu` is `≈ 2`. ADR 0158 already says `PriceLevelExpectedReturns` "reconstructs the levels
  from the *returns* it is handed" and, six paragraphs later, that "the head's rows buffer stays a
  buffer of price relatives" — one document, two units. The rule is the one reader that wants
  `x = 1 .+ r`, and it is one broadcast at its door.
- **The rows arrive as returns.** `ReturnsResult.X` is a returns matrix; every
  `SampleBufferState` in the library — the one `Online(pe)` seeds, the one a Prior Result
  carries — holds rows of it verbatim. A relatives buffer would be the only one whose rows are not
  what the Returns Result holds, and the Held Gap's own fill is a zero return.
- **The recursion needs a first allocation, and the prototype takes one.** Every paper starts at
  `1/N`. The prototype's driver takes `w0`, uniform if absent, and projects a given `w0` onto the
  simplex rather than refusing it. A live deployment starts from its book: on three assets under
  exponentiated gradient at `η = 0.05` and `x₁ = [1.10, 1.00, 0.95]`, a uniform start answers
  `≈ [0.336, 0.333, 0.331]` and a book `[0.6, 0.3, 0.1]` answers `≈ [0.604, 0.298, 0.098]` — a
  fund forced to start uniform trades 27 % of its book on the first row for nothing the paper
  wants. The loop's `w_prev` cannot serve: it reaches the head through `factory` alone, into a fee
  and a fallback, never into the recursion
  ([ADR 0160](0160-an-online-update-starts-from-its-own-allocation-and-its-trade-is-measured-from-the-price-adjusted-one.md)).
  `w0` is already the library's name for the first row of a held-weights path
  (`HeldWeightsResult.w0`, `14_NetReturnsDrawdowns.jl`).
- **Under a point-in-time universe `1/N` has two readings, and one of them is ADR 0157's.** On
  five pinned assets with the fifth unlisted for rows 1–20, a uniform start over the pinned
  universe parks `0.2` of the recursion in a leg that sees `x = 1` every row — a cash-like leg —
  so the read-out renormalises the four listed legs to `≈ 0.25` each with the recursion's tilts
  scaled by `0.8`, and at row 21 the fifth asset relists holding the `≈ 0.2` the recursion drifted
  it to. A uniform start over the first row's active mask puts the fifth leg at `0`, which a
  multiplicative rule holds forever — the forced zero ADR 0157 ruled out — and which only a
  re-entry policy, which the same ADR refused, could lift. The multiplicative gradient
  `x / ⟨w, x⟩` is the same scale either way, because `⟨w, x⟩ ≈ 1` in both.

## Decision

### The buffer holds returns, and the rule reads `1 .+ r` at its door

The head's rows buffer `X` is a `SampleBufferState` of **returns**, the rows of `ReturnsResult.X`
verbatim, capped by `rows_needed`. The head forms the period's price relative `x = 1 .+ r` once
per row and hands it to the Online Update; every other reader of the buffer — a forecaster's
`mean`, a Prior's `mu`, the Allocation Set's covariance, the tracking error over the rows, a
follow-the-leader's re-solve — reads returns as it is written and needs no conversion. A
non-finite cell is filled with `0` before the push, the Held Gap's own number, silently at an
inactive asset and as a Held Gap at an active one, so the rule sees `x = 1` there exactly as ADR
0157 intended. ADR 0157's table row and fill section, ADR 0158's closing sentence on the buffer,
and ADR 0159's tracking error are rewritten in place: `‖(X − 1) w − b‖` becomes `‖X w − b‖`.

### The head takes a Start Allocation, `w0`, and uniform is its absence

The head takes `w0::Option{<:AbstractVector}`, `nothing` by default. The **Start Allocation** is
the allocation held during the first period of the recursion: a given `w0` is over the pinned
asset names, is pinned with them, is sliced by a view as `w` is, and is projected once onto the
Allocation Set in the rule's own Projection Geometry at the first step, so a start outside the set
is made feasible as the prototype does and never refused. A zero entry under the entropic geometry
stays zero for the run, which is the caller's own choice and the same fact ADR 0159 states for a
negative bound. Absent, the Start Allocation is `1/N` over the pinned universe, and it meets the
set the same way, so the allocation held during the first period lies in the set under a cap that
`1/N` breaks. A set that reads the head's rows — a fitted risk ceiling, a tracking error — has no
constraints to form before the first row: the start is held as given, and the first Online Update
projects it, as that update holds every step until the head has two rows. A programme that fails
at the start holds the start as given and warns, as a row's Held Step does.

`w0` reaches the first Online Update as its `w`, and that is the whole of its meaning. A rule that
reads `w` — `BuyAndHold`, `ExponentiatedGradient`, `PassiveAggressive`, `ForecastReversion`
through the Price-Adjusted Allocation — continues from it; a rule that does not —
`ConstantRebalancedPortfolio` answers its own `w`, `ExpertMixture` answers its experts' mix, the
Newton step reads its Gram — replaces it at that update, so the fund holds `w0` for one period
and the rule's allocation from the second. That is what a live book migrating to such a rule does,
the period-one trade is measured from `ŵ₁` as ADR 0160 rules, and no rule refuses `w0`. The
docstring says the one-period case in one sentence.

### The uniform start is over the pinned universe

When `w0` is absent the recursion starts at `1/N` over every pinned name, unlisted ones included,
as ADR 0157's fill already implies. An unlisted leg earns a zero return until it lists, the read-out
slices it away and renormalises, and when it relists it holds the recursion's own weight. The cost
is stated, not hidden: on a universe where `k` of `N` assets are unlisted at the first row, the
recursion parks `k/N` of its weight in cash-like legs until the rule moves it, and the fund's
tilts are scaled by `(N − k)/N` for as long as that lasts. The head's docstring states it, and a
caller who wants the recursion to be the fund from row one gives `w0` over the listed assets with
zeros elsewhere and accepts what their rule's geometry does with a zero.

## Considered options

1. **A relatives buffer, every consumer subtracting one on entry.** Rejected: five `.− 1` sites
   today and one per future consumer, a consumer that forgets answers a silent `x̂ ≈ 2`, and the
   buffer would be the library's one `SampleBufferState` whose rows are not the Returns Result's.
2. **No `w0`, always uniform.** Rejected: a live deployment cannot start from its book, and the
   only route — a `ConstantRebalancedPortfolio` start row — is not the recursion's start.
3. **`w0` a vector or a naive estimator**, `Union{Nothing, AbstractVector,
   NaiveOptimisationEstimator}`. Rejected: no rows exist at the Causal Pass's first row, so any
   estimator but a row-free one refuses, and the union buys a spelling of uniform.
4. **A rule that does not read `w` refuses a given `w0` by name.** Rejected: a refusal per rule,
   and a live constant-rebalanced or mixture deployment could not declare its book; one meaning —
   held for period one, then the rule's — needs no refusal.
5. **Uniform over the first row's active mask.** Rejected: a multiplicative rule never holds a
   late lister, which reverses ADR 0157's no-forced-zero ruling or reopens the re-entry policy it
   closed; the reading is available to a caller through `w0` at their own risk.
6. **Refuse a `w0` outside the Allocation Set.** Rejected: the prototype projects, the first
   Online Update projects anyway, and a book that violates a new constraint is the ordinary case
   a constrained start exists for.

## Consequences

- ADR 0157 is rewritten in place at its state table row for `X`, its fill section and the two
  consequence lines that name the fill; ADR 0158 at its closing sentence on the buffer; ADR 0159
  at the tracking-error formula; ADR 0155 at the head's field list, which gains `w0`.
- `CONTEXT.md` gains *Start Allocation*; *Rule State* says the buffer holds returns and the fill
  is zero.
- The head's first build,
  [#1161](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1161), owes: the returns
  buffer with `x = 1 .+ r` formed at the step, the zero fill and its Held Gap warning, `w0` on
  the head with its pin, view and first-step projection, the one-period test on
  `ConstantRebalancedPortfolio`, the uniform start over the pinned universe with a relisting test,
  and the two docstring sentences.
- The parity test of ADR 0158 reads its price-relative fixture as `1 .+ r` first; the ledger's
  moving-average example is unchanged in value.
