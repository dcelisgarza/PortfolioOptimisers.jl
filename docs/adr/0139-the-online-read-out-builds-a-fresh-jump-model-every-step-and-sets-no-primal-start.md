---
status: accepted
---

# The online read-out builds a fresh JuMP model every step, and sets no primal start

## Context

[Map #861](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/861) names four
capabilities in scope on a best-effort basis, and the first is JuMP model reuse across steps: the
optimiser keeps its model between two online steps, updates the moment parameters inside it, and
warm-starts from the previous weights. A ticket that finds a capability cannot be justified records
the refusal and closes.
[Issue #869](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/869) is that ticket, and
this ADR is the record.

[ADR 0137](0137-an-optimiser-forwards-to-its-prior-alone-and-a-read-out-reconstitutes-the-carrier-and-runs-the-batch-path.md)
fixed the read-out: `optimise(opt)` rebuilds the carrier from the state and runs the ordinary batch
path, so every constraint, every uncertainty set and every inner optimiser is identical to batch by
construction, and the read-out is pure. A kept model is a second path beside that one. Whether it
earns its place is a measurement, not an argument, and the research of
[issue #863](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/863) had already narrowed
what the measurement could find: a `Parameter` is a fixed variable to every solver the library
tests; in-place modification reaches the solver only for HiGHS, while Clarabel and SCS rebuild from
JuMP's cache on every solve; and a scenario row cannot be appended to a vector constraint, so a new
observation is a new variable, a new constraint and `T + 1` coefficient rewrites.

### What the reference does

Its optimiser's `partial_fit` "is solved fresh on each call". Its expression cache lives inside one
fit and is cleared after every solve, and nothing keeps a problem across steps or sets a start. So
the capability has no parity target; it is an improvement or it is nothing.

### What was measured

Every number is `julia -t 1`, one BLAS thread, a `StableRNG` panel, `MeanRisk` over an
`EmpiricalPrior`, the minimum of five runs, in milliseconds. The batch step is what the read-out of
ADR 0137 costs today.

The build is a small share of a step under every solver:

| problem | prior | build | solve | build / step |
| --- | --- | --- | --- | --- |
| `Variance`, Clarabel, `N = 100`, `T = 1000` | 1.7 | 1.5 | 10.6 | 10 % |
| `ConditionalValueatRisk`, Clarabel, `N = 100`, `T = 1000` | 1.2 | 55 | 391 | 12 % |
| `ConditionalValueatRisk`, HiGHS, `N = 100`, `T = 1000` | 1.2 | 56 | 248 | 18 % |
| `ConditionalValueatRisk`, Clarabel, `N = 100`, `T = 3000` | 3 | 232 | 3389 | 6 % |
| `ConditionalValueatRisk`, HiGHS, `N = 100`, `T = 3000` | 3 | 186 | 2201 | 8 % |

A kept `ConditionalValueatRisk` model stepped in place from `T` to `T + 1` — one new exceedance
variable, one new exceedance row, `T + 1` objective-coefficient rewrites, one re-solve — reaches the
cold `T + 1` weights to `1e-15`, and costs:

| `N`, `T` | cold build and solve at `T + 1` | kept step | kept / cold |
| --- | --- | --- | --- |
| 20, 1000, HiGHS | 72 | 4.8 | 0.07 |
| 100, 1000, HiGHS | 409 | 15 | 0.04 |
| 100, 3000, HiGHS | 3098 | 39 | 0.01 |
| 20, 1000, Clarabel | 80 | 51 | 0.65 |
| 100, 1000, Clarabel | 478 | 357 | 0.75 |
| 100, 3000, Clarabel | 3502 | 2641 | 0.75 |

Under HiGHS the kept model re-solves by dual simplex from its own basis, and the step is one to
seven percent of the batch step. Under Clarabel the interior point runs in full from a fresh
`copy_to`, and the kept model saves the build and the attach alone.

A primal start does not pay under either. `set_start_value` **throws**
`UnsupportedAttribute{VariablePrimalStart}` on a model attached to Clarabel — it is silently
dropped only when set before the optimizer is attached, which the research had read as costing
nothing. Under HiGHS the start is accepted and the solve is *slower* than cold (565 ms against
248 ms at `N = 100`, `T = 1000`), because a primal point is crossed over where a basis is reused.
The warm start that pays is the kept basis, which comes with the kept model and with nothing else.

### What the seam would cost

The read-out of ADR 0137 has one path. A kept model is a second encoding of every builder whose
rows read the sample — fourteen families and about thirty-five concrete builders under
`src/20_Optimisation/20_RiskMeasureConstraints/` — each needing an append twin that knows its row
form, its `1 / T` scaling, its scalariser's epigraph and its `MaximumRatio` homogenisation, beside
an append for `X * w` and for the drawdown chain, a rewrite for the `ArithmeticReturn` mean, a
rewrite for each of the previous-weight sites `factory(opt, w)` writes (`tn`, `fees`, `tr`; `ccnt`
and `cobj` are caller callables and rebuild), and a per-type fingerprint naming which fields are
per-step so that a change to any other invalidates the model. Every future change to a builder is
made twice, and a drift between the twins is caught only by the family the identity test covers.
The kept model is also one mutated object: aliased across the results a loop stores, or hidden from
them, or copied at the build's own cost.

## Decision

**The online read-out builds a fresh `JuMP.Model` on every step, and no primal start is set.**
Capability 1 of map #861 is refused. The read-out of ADR 0137 stands as the only path, and
`optimise(opt)` after `t` folds costs what `optimise(opt, rd[1:t])` costs.

The numbers above are the threshold a future proposal must beat. The gain is 25 to 35 percent per
step under Clarabel on a scenario measure and at most 10 percent on a variance one, and 25 to 100
times under HiGHS on an LP scenario measure in a long expanding walk-forward. That corner is real
and narrow, and a caller in it already bounds `T` by capping the buffer, the composition
[ADR 0136](0136-a-prior-folds-and-carries-the-buffer-is-owned-once-and-a-cap-is-either-a-scenario-cap-or-a-window.md)
names, with no second path.

The warm-start half is closed for good, not deferred: a start throws under the default solver and
costs more than it saves under the one that accepts it.

## Considered options

1. **Keep the model, scoped to what pays** — append twins for the sample-reading LP families and
   the mean term, a rebuild for every other change, under any solver. Rejected on the
   maintainability lens: the second encoding of thirty-five builders is more surface than the whole
   online seam built so far, for 25 to 35 percent under the solver most of the library runs on.
2. **Keep the model, gated to solvers with the incremental interface** — the same twins, reuse only
   under HiGHS. Rejected: the same cost, a smaller gain, and a solver-dependent step.
3. **A partial-fit state on the JuMP host** as the model's home, as
   [ADR 0106](0106-a-partial-fit-state-is-the-one-result-an-estimator-holds.md) permits. Not chosen
   because the capability is refused; recorded because it was costed. The one exception in ADR 0106
   rests on no consumer reading a partial-fit state, and a `JuMP.Model` is exactly what consumers
   read through `JuMPOptimisationResult.model`, so the read-out would have to hide or copy it, and
   the purity of ADR 0137 would become idempotence.
4. **A Result the loop threads** as the model's home — `optimise(opt; prev = res)`, on the channel
   the loop already threads for the previous weights. Not chosen for the same reason, and it was
   the better home of the two: the estimator stays configuration, the read-out stays pure, and a
   stale hand-in is detected by a fingerprint rather than trusted.
5. **A `Parameter` per coefficient, or ParametricOptInterface.** Rejected by the research: a
   parameter times a variable is quadratic to JuMP, the scenario rows grow with `T` and are not
   parametric, and the inner solver rebuilds under Clarabel whichever way a change is pushed.

## Consequences

- `optimise(opt)` stays one method that runs the batch path, and
  `src/20_Optimisation/25_OnlineOptimisation.jl` gains nothing. The three identities of ADR 0137
  remain the whole contract of the online optimiser.
- The build of a scenario measure — 55 ms at `N = 100`, `T = 1000`, 232 ms at `T = 3000` under
  Clarabel, in `assemble_jump_model!` — is the only reuse-free saving on the table. It is a batch
  cost that the online step inherits, so it is a batch performance item outside map #861.
- The two open points of the research that this decision made moot — whether the frontier bound
  `p * k` under `MaximumRatio` passes Clarabel's bridges, and the route by which Pajarito loads its
  sub-solvers — stay unmeasured here, because nothing this decision leaves in place reads them.
