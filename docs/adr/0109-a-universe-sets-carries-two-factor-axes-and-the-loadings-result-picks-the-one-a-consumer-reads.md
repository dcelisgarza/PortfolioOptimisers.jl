---
status: accepted
---

# A Universe Sets carries two factor axes, and the loadings result picks the one a consumer reads

## Context

`UniverseSets` declared one factor axis, `fkey`, with a unique-entry sibling `ufkey`. Three groups
of consumers resolve factor names against it: the exposure constraints through
`constraint_space_basis`, `FactorRiskBudgeting` through `risk_budget_universe_key`, and the three
Black-Litterman priors that write views in factor names. Every one of them read `sets.fkey`
directly.

One axis was enough while one thing was a factor. A time-series regression fits one model per asset
over the observations, so its factors are the columns of a returns result's `F`. A caller copies
`rd.nf` into the dict under `fkey`, and `assert_universe_axis_order` holds the two lists equal name
for name, so the axis and the loadings agree by construction.

[Issue #679](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/679) built the
cross-sectional regression family, and
[issue #706](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/706) built
`CrossSectionalFactorModel`, a second member of `AbstractLoadingsRegressionResult`.
[Issue #707](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/707) widened
`LowOrderPrior.rr` to that root, so the prior's loadings slot now holds either family. A
cross-sectional model is fitted per observation across the assets, and its factors are the Factor
Exposures the fit was built from. Those exposures exist only inside the fitted block: no
`ReturnsResult` carries them, and `ReturnsResult` refuses `nf` without `F`, so they cannot be put
where `rd.nf` goes.

That is the collision. One key would have to name two different lists on one problem, and a problem
may carry both families at once — a time-series prior wrapped by an optimiser whose exposure
mandate is written against a cross-sectional model, for example. Both lists are factor names, and
both are the right length for `size(rr.M, 2)`, so a mismatch is not a length error. A constraint
written for one family and resolved against the other's names still assembles and still solves. It
constrains the wrong factors, silently.

[Issue #651](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/651) decided the shape in
its decisions 6 and 7, and this ADR records it.
[Issue #723](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/723) built it.

## Decision

### Two factor axes, seven key prefixes

`fkey` and `ufkey` are renamed to **`tfkey`** and **`utfkey`**, the time-series pair, and **`cfkey`**
and **`ucfkey`** join them as the cross-sectional pair. The defaults are `tfkey = "nf"`,
`utfkey = "uf"`, `cfkey = "ncf"`, `ucfkey = "ucf"`. `"nf"` stays the time-series default, because
`pipeline_asset_sets` writes `"nf" => rd.nf` and every sets object already written declares the
time-series axis under it.

The new pair obeys the old pair's two rules exactly. A `cfkey`-prefixed key is a partition of the
cross-sectional axis and has that axis' length; a `ucfkey`-prefixed key is a unique-entry group and
names the `cfkey`-prefixed partition it draws from. The rules are written **once**, in
`assert_factor_partition` and `assert_factor_unique_group`, and called twice with the axis named in
prose. Two copies of the same rule would drift, and the diagnostic has to say *which* axis is
missing: the key value alone does not, because a caller who declared the time-series axis and wrote
a cross-sectional partition would otherwise be told to add a key they believe they already have.

The two axes are **never validated against each other**. Each is optional on its own, so a problem
may declare one, both, or neither. A cross-sectional mandate with no time-series model is a
complete, valid object.

The prefix grammar grows from five keys to seven, so the ordered non-prefix check grows from 20
pairs to 42. `port_opt_view` carries both axes through unchanged, for one reason that covers both:
an asset index has no meaning on either.

### A consumer picks the axis by dispatch on the loadings result

`factor_axis_key(sets, rr)` is unexported and has three methods. A `Regression` answers
`sets.tfkey`, a `CrossSectionalFactorModel` answers `sets.cfkey`, and an
`AbstractTimeSeriesRegressionEstimator` — a specification whose verb produces a `Regression` —
answers `sets.tfkey`. The three cover `RegE_Reg` exactly, which is the alias every consumer of a
loadings source binds.

There is deliberately **no fallback on `AbstractLoadingsRegressionResult`**. A future member of the
root would inherit whichever axis the fallback named, and the failure mode of a wrong axis is the
silent one described above. A missing method is a `MethodError` naming the type.

`factor_universe` gains an explicit `key` argument for the same reason: it cannot read the axis off
`sets` any more, because `sets` carries two. A caller that holds a loadings result passes
`factor_axis_key`; a caller written against the returns data's own `F` passes `sets.tfkey`, the axis
those columns live on.

### The three options, and why dispatch won

`UniverseSets` carrying two axes was not in question — the two lists are different data. What was in
question is how a consumer says which one it reads. Three ways were weighed.

| Option | How a consumer names the axis | Why it was not taken |
| --- | --- | --- |
| **Dispatch on `rr`** | `factor_axis_key(sets, rr)` reads the key off the block that carries `M`. | Taken. |
| **An `axis` field per consumer** | `FactorSpace(; axis = :cfkey)`, and the same field on `FactorRiskBudgeting`. | The field can disagree with the loadings the consumer is handed, and nothing can check it: both lists are factor names of the right length. It also costs a field on every present and future consumer of the axis, and a caller must know a fact the loadings already state. |
| **Whichever key `dict` carries** | Read `cfkey` when present, else `tfkey`. | A problem that declares both axes has no answer, and that is the case the split exists for. It also makes a mandate's meaning depend on an unrelated key elsewhere in the dict, so adding a cross-sectional axis for one estimator would silently re-point another's names. |

The taken option needs no field, admits no disagreement, and puts the choice where the information
already is. The key follows the block that carries `M`, so a caller cannot name the wrong axis.

### The Black-Litterman axis symbols widen, and the rename is a hard break

`bl_preroll`'s `axis` argument accepts `(:xkey, :tfkey, :cfkey)` rather than `(:xkey, :fkey)`. Every
member in the library today writes views against a factor distribution whose columns are the columns
of `F`, so every one passes `:tfkey`; `:cfkey` is accepted because the axis exists and a view can
land on it, not because a member reaches it yet.

The rename is a **hard break**, as [#705](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/705)
was. The library carries no `@deprecate`, and a shim keyword that silently wrote `tfkey` would leave
a caller believing they had named an axis that the split gave a new meaning. A caller who declared
only the time-series axis under the default key sees no behaviour change at all.

## What the build measured

- **The rename touched 14 source files, six test files, three examples, the user guide and
  `CONTEXT.md`.** `ufkey` contains `fkey`, so one substitution produced both `tfkey` and `utfkey`.
- **Eight doctest blocks re-indented, not five.** The pretty-printer right-aligns field names to the
  widest field in the block, and `ucfkey` is one character wider than `utfkey` was, so every line of
  every block that prints a `UniverseSets` moved — nested blocks included. Documenter's `fix = true`
  rewrote none of them: it reports `could not find code block in source file` for a doctest written
  inside a `"""` docstring with escaped quotes. The expected output was spliced from a plain run's
  own report instead.
- **Two test files asserted the old single axis, and both assertions were the point of the split.**
  `test_12i`'s *M readers* testset held that a cross-sectional model and a regression answer the
  same key; it now holds that they answer `cfkey` and `tfkey` respectively, and that a sets
  declaring only the time-series axis serves the regression and raises a `KeyError` naming `ncf` for
  the model. `test_12e` held the exact wording of the refusal a graded feature program gives a
  factor-axis name; the message now names all four prefixes and both new ones are refused in all
  three positions.
- **`risk_budget_universe_key` had no `rr` in its signature and needed none.** `FactorRiskBudgeting`
  carries `re::RegE_Reg`, which is the source of its own loadings, so the verb reads the axis off
  the field it already holds.
- **The optimiser's name check stays on the time-series axis.** `assert_universe_axis_order`
  reconciles a declared axis against `rd.nf`, and `rd.nf` is the time-series axis by definition, so
  it reads `sets.tfkey` outright rather than dispatching.
- **`pipeline_asset_sets` stays on the time-series axis** for the same reason: it builds its sets
  from the returns' own names, and a returns result carries no cross-sectional exposures. A step
  whose loadings are a cross-sectional model therefore finds no `cfkey` entry and says so through
  `factor_universe`; such a mandate needs sets a caller supplies.

## Consequences

- A caller who declared only the time-series axis renames two keywords and changes nothing else.
- A mandate written for one factor family can no longer resolve against the other family's names.
  The failure is a `KeyError` or a `DimensionMismatch` naming the axis, at the point of need.
- `factor_universe` states its key, so a new consumer of a factor axis must decide which one it
  reads. That is the intended cost: the decision is no longer implicit.
- A new member of `AbstractLoadingsRegressionResult` owes a `factor_axis_key` method. Without one it
  raises rather than resolving names on an axis nobody chose for it.
- The two axes are independent, so nothing stops a caller declaring `cfkey` names that duplicate
  `tfkey` names. They are different lists in different keys, and a consumer reads exactly one.
