---
status: accepted
---

# A return expression is a weighted sum of terms

## Context

[`JuMPOptimiser`](../../src/20_Optimisation/10_JuMPOptimiser.jl)'s `ret` field held **one**
returns estimator. The risk side of the same struct had held a vector for far longer:
`MeanRisk.r` takes one risk measure or several, each carrying a
[`RiskMeasureSettings`](../../src/19_RiskMeasures/01_Base_RiskMeasures.jl) bundle, and the
optimiser's `sca` field names the **Scalariser** that collapses them into the model's single
`:risk` expression.

Widening `ret` the same way raised two questions at once, and they are the same question seen
from two sides.

1. **What is it that became plural?** The charting called it a *characteristic vector* — the
   per-asset quantity ADR 0032 minted for the ℓ1 uncertainty sets, and which `CONTEXT.md`
   §3.9 defines. If that is the plural noun, then §3.9's definition has to widen, and its
   *Avoid* line — which holds a Characteristic Vector apart from a **Feature** (§2) on the
   grounds that one is a *single* per-asset quantity and the other is *one column of many* —
   stops working, because *k* characteristic vectors side by side are an `assets × k` matrix.
2. **How do the terms combine?** Symmetry with the risk side says: add a second scalariser
   field, `rt_sca`, and rename `sca` to `rk_sca` on the optimiser so the two can coexist.
   That is the shape a reader arrives expecting, so refusing it needs a reason on record —
   otherwise it is added back for symmetry by whoever asks next.

## Decision

**A return expression is a weighted sum of return terms.** Both halves of that sentence are
decisions.

### The plural noun is the *term*, not the characteristic

An element of the vector is **any**
[`JuMPReturnsEstimator`](../../src/20_Optimisation/09_JuMPConstraints/02_Returns_and_ObjectiveFunctions.jl),
and two admissible elements have no characteristic at all:

- `LogarithmicReturn` holds `settings` and `w`. There is **no per-asset quantity on it**, so
  "the vector of characteristics the optimiser maximises" is false of a legal configuration on
  day one.
- A term whose `settings.rte` is `false` contributes **nothing** to the return expression,
  while its own `settings.lb` still binds. Such a **constraint-only** term shapes the feasible
  set and is never maximised. It is not a characteristic under any reading either.

So the concept `CONTEXT.md` §3.9 defines is **untouched by multiplicity**. Each set-bearing
term still has exactly one per-asset quantity its ball is built around, and ADR 0050 already
put that quantity **on the set**. An optimisation now simply holds several Characteristic
Vectors, one per set-bearing term.

That does cost §3.9 its *Avoid* line's argument, and the loss is total rather than partial:
the count argument does not weaken, it **inverts**. Two grounds replace it, both already
load-bearing elsewhere, so neither is a new assertion:

- **Use.** A Feature's rows become a **distance**. A Characteristic Vector enters the
  objective as `mu'w` and gets an uncertainty budget around it. Neither is derivable from the
  other's shape.
- **Side of ADR 0045's line.** A Feature Matrix is **data**: it rides on `ReturnsResult` /
  `LowOrderPrior` and is sliced by `port_opt_view`. Characteristics ride on **estimators**,
  which are configuration, and are resolved per term at solve time. So *k* characteristics
  **never accumulate into a Feature Matrix** — the two families cannot meet, structurally.

ADR 0045 needs no amendment: checked, and it never used the count argument. Its decision rests
on the naming collision itself and on the `Z`/`F` symbol split, both untouched.

### The terms combine by a weighted sum, and there is no scalariser on this side

```math
\mathrm{ret} = \sum_{i \,:\, \mathrm{rte}_i} s_i \, \mathrm{ret}_i
```

There is no `rt_sca` field, `sca` keeps its name, and there is no configuration in which a
return expression is combined any other way.

The reason is not symmetry-breaking for its own sake. **This package's scalarisers follow
cvxpy's `scalarize` transforms**, and cvxpy is the authority on what a scalariser is:

- cvxpy's `max` and `log_sum_exp` transforms **discard the objective's sense** and always
  return a `Minimize`. Applied to a **maximised concave** expression, which is what a return
  expression is, they fail DCP. cvxpy raises rather than reach for binaries.
- cvxpy ships **no `min` transform** at all. `MinScalariser` is this package's own extension,
  and it stays a `HierarchicalScalariser`.

Sense-normalisation would rescue all three at a stroke — store `-ret`, minimise, and every
transform becomes available and conic. It is **barred**. `:ret` is a model-global name that
[`get_ret`](../../src/20_Optimisation/08_Base_JuMPOptimisation.jl) serves to the objective, to
the return bounds, to the `MaximumRatio` numerator and to
[`NearOptimalCentering`](../../src/20_Optimisation/13_NearOptimalCentering.jl). A stored
`-ret` leads every one of them astray, and the cost of that is spread across the whole model
rather than paid at one site.

One thing this decision does **not** rest on is impossibility, and the record needs to be
straight about it: the **soft-min** is concave, and is the risk side's exponential cone with
one sign flipped. A smooth scalariser was affordable here. It was declined on **convention**
— the same name must mean the same formula on both sides, and cvxpy does not ship it — not on
cost.

The per-element **`scale` survives** the scalariser's absence, and is not a leftover. It is
exactly cvxpy's `weights` argument, and it is the only route to a weighted
`LogarithmicReturn`, whose struct holds `settings` and `w` alone.

## Consequences

**The shape that shipped.** Every return term carries a
[`JuMPReturnsSettings`](../../src/20_Optimisation/09_JuMPConstraints/02_Returns_and_ObjectiveFunctions.jl)
bundle — `scale`, `lb`, `rte`, `fee`, `mic` — in a field called `settings` placed **first**,
matching `Variance(settings, sigma, chol, rc, alg)` on the risk side. Each term's builder
registers index-suffixed names (`ret_1`, `t_l1ucs_2`), `set_return_expression!` pushes
`scaleᵢ · retᵢ` onto the `:ret_vec` Model State entry, and `scalarise_return_expression!` sums
that vector into `:ret`. The API break is two things and no more: `lb` **moved out of the
estimator into the bundle**, and `settings` took the **first slot**, which `@concrete`'s
positional constructor makes a breaking change rather than a cosmetic one.

**The single-term case is unchanged, and the proof is structural rather than numerical.** With
one term at `scale = 1`, `model[:ret] == model[:ret_1]` as affine functions, and a bare
`ArithmeticReturn` adds **no rows and no variables** — the only new Model State key is
`:ret_vec`, a plain Julia vector. Same rows plus same objective is the same problem, for every
input rather than for the ones a test happens to try.

**No binaries enter anywhere**, on either side of the decision. ADR 0032's claim that an ℓ1
term keeps the model an LP whenever the rest of the problem is one therefore **stands
unamended**: a sum of linear terms is linear, and `set_ucs_return_constraints!`'s docstring,
which advertises the same property, needed no correction either. This is worth stating because
the retired version of this decision — a scalariser whose non-conic formulas would "ship with
binaries" — would have falsified both.

**An empty return expression is legal but not universally.** Every term may opt out through
`rte = false`, which gives a zero return expression rather than an error, because that
configuration is meaningful for every objective except `MaximumRatio` — an empty numerator,
which is refused separately, mirroring `NoRisk` + `MaximumRatio`.

**The weight is not normalised, and that is visible in the charges.** A term's flagged fees are
charged once per flagged term, so the multiplier is `Σ_{i: fee} scaleᵢ`: a blend of two terms
at `scale = 0.5` charges the fees once, while two terms at `scale = 1` charge them twice.
There is no check, no warning and no divisor. It is a statement about the configuration the
caller wrote, and stating it here is cheaper than a guess at the intent.

**The asymmetry is now the documented answer to an obvious question.** A reader who finds
`opt.sca` beside a vector-valued `opt.ret` and looks for its twin finds this document instead
of an inconsistency, which is the debt [#267](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/267)
closed owing.

## Amendment (2026-08-16)

**The single-term invariance widens from `scale = 1` to *any* lone `scale`.**

The Consequences section above states `model[:ret] == model[:ret_1]` for one term **at
`scale = 1`**. That qualifier described the code accurately and hid a defect. The singular
route,
[`set_return_constraints!(model, pret::JuMPReturnsEstimator, …)`](../../src/20_Optimisation/09_JuMPConstraints/02_Returns_and_ObjectiveFunctions.jl),
carried no step that dropped the weight, so a lone term at `scale = 2` built
`model[:ret] == 2 * model[:ret_1]`.

That is observable. It moves `MaximumUtility`'s return/risk trade-off and Near Optimal
Centering's return barrier. It is *not* observable under `MinimumRisk`, `MaximumReturn` or
`MaximumRatio`, all of which are argmax-invariant under a positive rescale of `:ret`, which is
why it survived unnoticed. Measured on ten assets, a lone term moved `MaximumUtility`'s weights
by **0.268** between `scale = 1` and `scale = 50`, while the other three objectives moved by
about `1e-6`.

The rule is now stated on its own terms in ADR 0053: `scale` is a **Combination Weight**, and
one element is not a combination. The singular route drops the weight through
`unit_scale_returns_estimator`, the return-axis twin of `unit_scale_risk_measure`. So the
invariance above now holds for **every** lone `scale`, not only for `1`.

**The fee multiplier is untouched.** The Consequences section's `Σ_{i: fee} scaleᵢ` statement
is about *relative* weights among **several** terms, and several terms are a combination. It
stands exactly as written. Exempting the return axis on the ground that `scale` has an absolute
meaning there was considered and rejected: with one term, `0.5·(μ'w − fees)` against
`1·(μ'w − fees)` is a pure rescale of `:ret`, which is the situation the risk axis had already
resolved. An asymmetry between the two axes had no justification.

**`lb` is unaffected.** It binds on the term's own expression, before `scale` is applied, so
dropping a lone weight cannot move a bound. `rte`, `fee` and `mic` are not weights either, and
all four survive the drop.
