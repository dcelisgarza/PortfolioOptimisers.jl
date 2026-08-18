---
status: accepted
---

# Wrapping priors forward by default, and document every drop

## Context

Most prior estimators wrap another one. [`BlackLittermanPrior`](../../src/13_Prior/06_BlackLittermanPrior.jl)
takes a `pe`, fits it, and returns a carrier built from the result; so do
[`BayesianBlackLittermanPrior`](../../src/13_Prior/07_BayesianBlackLittermanPrior.jl),
[`EntropyPoolingPrior`](../../src/13_Prior/10_EntropyPoolingPrior.jl),
[`OpinionPoolingPrior`](../../src/13_Prior/11_OpinionPoolingPrior.jl) and
[`FeaturePrior`](../../src/13_Prior/13_FeaturePrior.jl). Each one decides, field by field, which of
the wrapped carrier's thirteen fields to carry across — and each one decided independently, by
writing out a `LowOrderPrior(; …)` call with whichever keywords its author thought applied.

The result was four Black-Litterman members forwarding four *different* subsets of the same carrier,
with no stated rule to appeal to. Nothing in the library detected a disagreement, because a carrier
with fewer fields populated is a perfectly valid carrier. Issue
[#181](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/181) traced the consequences and
found three defects of one class — a value the caller explicitly computed, silently discarded on the
way through a wrapper, with a plausible number coming out the other end:

- `BlackLittermanPrior(; pe = EntropyPoolingPrior(…))` dropped `w`, so the pooling posterior weights
  never reached the 28 `@pprop w` sites and the optimisation ran unweighted.
- The same drop took `ens` with it.
  [`choose_scaling_parameter`](../../src/14_UncertaintySets/03_NormalUncertaintySets.jl) falls back to
  `size(pr.X, 1)` when `ens` is `nothing`, so every uncertainty set was sized off a sample count
  measured at `ens = 225.9` against `T = 250` — about 10% too large.
- `FactorPrior(; pe = EntropyPoolingPrior(…))` forwarded `w` but not `ens` or `kld`, so the same
  defect reached factor priors by a second route.

Two framings were rejected before the rule was chosen. **"Forward everything"** is wrong because some
fields become false when forwarded: a `chol` carried past a covariance update is not a stale
diagnostic, it is a wrong answer (see below). **"Justify every forward"** is what produced the four
subsets — it makes the safe direction the expensive one to write, so the cheap thing to write is a
narrower carrier, and narrowing is exactly how the three defects arose.

## Decision

**Forward when forwarding is correct; drop only where forwarding would state something false;
document every drop in the estimator's docstring.**

Consistency of the returned result is the criterion. Destroying data the caller explicitly computed
is **not** an acceptable way to buy that consistency: where a drop is genuinely required it is
documented rather than silent.

The rule gets teeth in code as well as in prose, because prose alone is what the library already had.
[`forward_prior`](../../src/13_Prior/01_Base_Prior.jl) forwards the whole wrapped carrier and takes
the deviations as keywords:

```julia
# BlackLittermanPrior — `chol` the only explicit drop
return forward_prior(prior_model; mu = posterior_mu, sigma = posterior_sigma, chol = nothing)
```

Forwarding is now the default that costs nothing to write, and every deviation is spelled at the call
site — a replacement as `field = value`, a drop as `field = nothing`. The set of drops becomes
greppable and reviewable rather than implicit in which keywords a thirteen-field constructor call
happened to list.

### The two bindings that make the rule mechanical

Applying the rule needs no case-by-case judgement for two groups of fields, because they are *bound*
to another field rather than being independent. Forwarding a bound field past a change to the field
it describes is the definition of stating something false. `forward_prior` therefore **refuses** such
a forward, with a [`ConflictingArgumentError`](../../src/01_Base.jl) naming the remedy — the caller
must pass either a rebuilt value or `nothing`.

**`chol` is bound to `sigma`.** This is not a caching nicety. `chol` *takes precedence over* `sigma`
at every consumer:
[`02_VarianceConstraints.jl`](../../src/20_Optimisation/20_RiskMeasureConstraints/02_VarianceConstraints.jl)
reads `G = isnothing(pr.chol) ? LinearAlgebra.cholesky(pr.sigma).U : pr.chol`, and `@pprop chol`
selects it into [`Variance`](../../src/19_RiskMeasures/02_Variance.jl), `StandardDeviation` and
`DistributionValueatRisk`. A forwarded stale `chol` therefore makes the optimisation use the *prior*
covariance and silently ignore the posterior — the worst available failure mode, since the objective
is quietly built from the wrong matrix. Note that `chol` is not merely a cache: a prior may compute
one with deliberately better sparsity than a fresh factorisation would have. **Syncing still wins
over sparsity.** Dropping it costs one factorisation at the consumer, which is recomputed from a
`sigma` that is correct by construction.

**`w` is bound to the observation axis, and diagnostics follow their weights.** `w` holds
*observation* weights — `@pprop w` resolves it against a length-`T` return series via
`get_observation_weights` — so it stays true exactly as long as the rows of the returned `X` are the
rows it was computed over. Black-Litterman never touches that axis (`posterior_X === prior_model.X`),
so forwarding `w` through it states nothing false, and dropping it is not neutral: the fallback is
the *unweighted* empirical distribution, strictly further from the caller's intent than the weights
they computed. `ens`, `kld` and `ow` are diagnostics *of* `w` and travel as one bundle with it — a
carrier holding weights whose provenance has been discarded cannot be interrogated, and a carrier
holding another weighting's `ens` mis-sizes every uncertainty set built on it.

What the constructor already enforces is left to the constructor: `rr`, `f_mu` and `f_sigma` must be
supplied together or not at all, and `w`, `chol` and `Z` are re-checked against the shapes of `X` and
`mu` on every reconstruction.

### Documenting the drops

The rule's third clause is not decoration. A drop that is correct is still a surprise to a caller who
computed the dropped value, so each wrapping estimator's docstring lists the fields it drops and why,
and [`LowOrderPrior`](../../src/13_Prior/01_Base_Prior.jl) carries the rule itself so there is one
place to read it. Two drops need a docstring *warning* rather than a list entry, because they hand
back a carrier that is internally inconsistent by design:

- A Black-Litterman result pairs a posterior `mu`/`sigma` with the **wrapped prior's** observation
  weighting. That split is inherent to Black-Litterman — the returned `mu`/`sigma` are already not
  the moments of the returned `X` — but a caller reading `pr.w` should know which distribution it
  describes.
- Forwarding `rr`, `f_mu` and `f_sigma` through Black-Litterman leaves `mu != M * f_mu + b`. The
  factor block remains *structurally* true (the regression is over data BL does not modify) while
  becoming *distributionally* inconsistent with the asset block.

### Reconstruction goes through the carrier's ordinary keyword constructor

`forward_prior` rebuilds the carrier by calling the same `LowOrderPrior(; …)` /
`HighOrderPrior(; …)` constructor a construction site would write by hand — so **every `@argcheck`
runs on every forward**. A wrapper cannot produce an internally inconsistent carrier by forwarding
one half of a paired field group and replacing the other; it throws exactly as the hand-written call
would.

The split that makes this work is between the field **list** and the constructor **name**:

- The list is *derived*. `prior_field_values` returns `NamedTuple{fieldnames(T)}` read through
  `getfield`, so a carrier that gains a field needs no edit here. Enumerating the thirteen keywords
  by hand would have re-encoded the field list in the one helper whose job is to stop wrappers
  disagreeing about it — the same defect class, relocated.
- The name is *written*, one two-line `reconstruct_prior` method per carrier. Recovering it
  generically needs either `Base.typename(T).wrapper` — internal `Base` reflection of exactly the
  kind this repo has been removing — or a dependency on `ConstructionBase` for `constructorof`.
  Neither buys anything: two methods is the whole cost, and a future carrier without one gets a
  `MethodError` naming `reconstruct_prior` rather than being reconstructed by machinery that has
  never seen it.

`ConstructionBase.setproperties` was the first choice and is **rejected**, which is worth recording
because it looks like the obvious fit. It refuses any type whose `propertynames` differs from its
`fieldnames`, since it cannot tell which properties are settable — and
[`HighOrderPrior`](../../src/13_Prior/01_Base_Prior.jl) forwards the whole of its `pr`, so `mu` and
`sigma` are properties of it without being fields. It therefore errors out on the *carrier* rather
than on any mistake the caller made, and the carrier redesign that gives `LowOrderPrior` a
`@forward_properties` block of its own would extend that to both carriers. Working around it means
defining `setproperties` for `AbstractPriorResult` and taking `constructorof` and `getfields` from a
new direct dependency — a lot of surface, on another package's dispatch, to obtain a constructor call
we can simply write.

That property/field distinction survives regardless of the mechanism, and `forward_prior` enforces it
directly: only a carrier's own **fields** may be named, because a forwarded or computed property is a
*view* of a nested value, so setting it could only ever mean setting the field that value came from.
Naming `mu` on a `HighOrderPrior` is refused, with the field list in the message; the patch goes
through `pr`.

### Where the helper does not apply

Three estimators are not forwarding a single wrapped result along its own axis, and are not forced
through the helper:

- [`FactorPrior`](../../src/13_Prior/03_FactorPrior.jl) and
  [`FactorBlackLittermanPrior`](../../src/13_Prior/08_FactorBlackLittermanPrior.jl) **lift** a
  factor-axis prior into an asset-axis result, reconstructing `X` as `F * transpose(M) .+
  transpose(b)`. Almost every field changes meaning across that hop, so there is nothing to forward
  by default.
- [`AugmentedBlackLittermanPrior`](../../src/13_Prior/09_AugmentedBlackLittermanPrior.jl) **merges
  two** priors and has to choose a source for each field.

The rule still governs them — it is a rule about correctness, not about a function call — and
`forward_prior` still applies to the *factor block* the first two build, which is an ordinary forward
of the factor prior along its own axis.

## Consequences

- **No estimator's behaviour changes when this lands.** The helper exists and the rule is written;
  applying it at the construction sites is deliberately separated, so the sites are edited once,
  after the carrier redesign, rather than twice.
- **A wrapper can no longer forward a stale `chol` or an orphaned `ens`/`kld`/`ow` by omission.** Both
  bindings throw at the point of the forward. The failure is loud and names the remedy, where
  previously the optimisation ran and returned a number.
- **No new dependency, and no reflection.** The helper is `merge` on a named tuple, a keyword
  constructor call, and two `haskey` checks. `Accessors.@set` on a prior result keeps erroring as it
  always has; making it work is a separate decision, and taking it would not obviate `forward_prior`,
  since a lens patches fields without enforcing either binding.
- **A new prior result type needs a `reconstruct_prior` method** before `forward_prior` works on it.
  This is a deliberate two lines rather than an accident: the `MethodError` names the function to
  define, and a carrier the helper has never seen is not silently reconstructed.
- **A drop is now a docstring obligation.** Adding a field to `LowOrderPrior` means every wrapper that
  cannot forward it must say so; the reviewer's question at a construction site becomes "why is this
  keyword here?" rather than "which keywords are missing?".
- **Third-party prior estimators get the same guarantee.** `forward_prior` is marked `public` (not
  exported, following `port_opt_view` and `assert_prior_regression`), so an estimator defined outside
  the package composes correctly by default instead of having to rediscover the field-by-field
  contract.

## Amendment (2026-07-30): the factor block is one field, `fpr`

`LowOrderPrior`'s factor block is no longer the three flat fields `f_mu`, `f_sigma` and `f_w`. It is a
single nested prior result, `fpr::Option{<:LowOrderPrior}`, whose `X` is the factor returns matrix
over the same observations as the asset block. Two statements above are therefore read with `fpr` in
place of `f_mu`/`f_sigma`:

- What the constructor enforces is now that **`rr` and `fpr`** are supplied together or not at all,
  plus `size(fpr.X, 1) == size(X, 1)` — the shared observation axis, which nothing checked while the
  factor moments were flat fields. Everything internal to the block, including its own `w` against
  its own `X`, is validated by its own constructor rather than restated here.
- The docstring warning about forwarding the factor block through Black-Litterman is unchanged in
  substance — `mu != M * fpr.mu + b` — but it now covers the whole block in one field rather than a
  pair of fields that could drift apart.

Two consequences for the rule's mechanics:

- **The question "which factor fields does a wrapper forward?" no longer exists.** A wrapper forwards
  the factor block or it does not; there is no subset to disagree about, which was the failure mode
  this ADR exists to end. `f_ens`, `f_kld` and `f_ow` come with it at no storage cost.
- **`forward_prior` refuses the flat names**, because they are properties rather than fields — the
  same refusal `mu` gets on a `HighOrderPrior`. A construction site names `fpr`.

The flat names remain readable — `pr.f_mu`, `pr.f_sigma`, `pr.f_w` and now `pr.f_ens`, `pr.f_kld`,
`pr.f_ow` are computed properties of the nested block, returning `nothing` when there is no factor
block, exactly as the fields did. Whether they or `pr.fpr.mu` are the idiomatic public read is not
settled here.

## Amendment (2026-07-30): `HighOrderPrior` nests too, and the shared inner prior is enforced

`HighOrderPrior` gets the same treatment as `LowOrderPrior`: `f_kt`, `f_sk` and `f_V` are replaced by
a single nested `fpr::Option{<:HighOrderPrior}` over the factors, taking the carrier from eleven
fields to nine. As before, the flat names survive as computed properties, and `f_D2`, `f_L2`, `f_S2`
and `f_skmp` come with them at no storage cost. The direction is one-way: factor co-moments require a
low order factor block, but a low order factor block with no co-moments over it is ordinary, so
`fpr === nothing` is always allowed.

Nesting at two orders makes the factor low order prior reachable by two routes — `hop.fpr.pr`, the
nested high order block's own prior, and `hop.pr.fpr`, the wrapped low order carrier's factor block.
**They must be the same object, and the constructor enforces it** (`fpr.pr === inner`), rather than
leaving the two free to drift. The alternative — deriving one from the other so there is only one
route — was rejected: `fpr` has to be a `HighOrderPrior` for its own constructor to validate the
factor co-moment shapes, and a `HighOrderPrior` has a `pr`. The redundancy is inherent to the shape;
what is avoidable is letting it be inconsistent.

Two decisions follow from that:

- **The check is `===`, not `==`.** Both carriers are immutable, so `===` is field-wise egality with
  arrays compared by identity: it accepts a nested block rebuilt around the very same arrays and
  refuses one refit to numerically equal values. "Refit and it happened to agree" is exactly the case
  worth refusing, because the two routes would then be two computations rather than one distribution.
- **The error message is the whole user-facing surface of this design**, so it distinguishes the two
  ways to get it wrong. Nesting over a prior with no factor block at all raises `IsNothingError`
  naming `pr.fpr === nothing` and pointing at `FactorPrior`; nesting a block whose inner prior
  differs raises `ConflictingArgumentError` naming both routes, saying what would go wrong
  (`hop.fpr.mu` and `hop.f_mu` disagreeing with no way to tell which is right), and giving the fix
  (`HighOrderPrior(; pr = pr.fpr, kt = ...)`).

`forward_prior` gains no new binding for this. Patching `pr` on a `HighOrderPrior` without patching
`fpr` is caught by the constructor with the message above, which is more specific than a binding
error could be — the same reason `rr`/`fpr` togetherness is left to the constructor.

One name shifts meaning: `hop.fpr` is now the carrier's own field, the *high* order factor block,
where before it resolved through `forward(pr)` to the low order one. Reads through it are unaffected,
because the nested carrier forwards to its own `pr` and the invariant pins that to `pr.fpr` — so
`hop.fpr.mu` is the factor mean either way, and `hop.fpr` is simply "the factor prior at this order".

## Amendment (2026-07-30): the rule is applied at every construction site

The Consequences above open with "no estimator's behaviour changes when this lands", which was true
of the helper on its own. Applying the rule at the sites is where the behaviour changes, and it has
now landed. This records what changed and what it fixed.

Two sites became a `forward_prior` call, because they wrap one prior along its own axis and change
only the asset moments:

- [`BlackLittermanPrior`](../../src/13_Prior/06_BlackLittermanPrior.jl) — was forwarding `X`, `mu`,
  `sigma` and `Z`, and dropping the other seven fields. Now `chol` is its only drop.
- [`BayesianBlackLittermanPrior`](../../src/13_Prior/07_BayesianBlackLittermanPrior.jl) — same, and
  it was already forwarding `rr` and the factor block.

[`FeaturePrior`](../../src/13_Prior/13_FeaturePrior.jl) also collapses to one, with `Z` as the single
deviation; it was already forwarding everything by hand, so the change is that the hand-written list
can no longer drift from the carrier's field list.

Three sites keep a direct constructor call, as the Decision section says they should, and gained the
diagnostics that were missing from the `w` they already forwarded:

- [`FactorPrior`](../../src/13_Prior/03_FactorPrior.jl) and
  [`FactorBlackLittermanPrior`](../../src/13_Prior/08_FactorBlackLittermanPrior.jl) — `w =
  f_prior.w` was correct and stays; `ens`/`kld`/`ow` now travel with it.
- [`AugmentedBlackLittermanPrior`](../../src/13_Prior/09_AugmentedBlackLittermanPrior.jl) — the asset
  slot takes `a_prior`'s `w` and diagnostics, the factor block is `f_prior` whole, so the two
  weightings stay distinguishable. `chol` is dropped.

### The three silent defects this closed

Each has a regression test in `test/test_12g_forwarding_rule.jl` that carries its own falsification
witness — a carrier rebuilt exactly as the pre-fix site built it, asserted to show the old behaviour.

1. **Pooled observation weights never reached the risk-measure layer.**
   `BlackLittermanPrior(; pe = EntropyPoolingPrior(…))` dropped `w`, so the 29 `@pprop w` sites saw
   nothing and the optimisation ran unweighted, silently.
2. **Every uncertainty set built on such a prior was sized off the wrong sample count.**
   `choose_scaling_parameter` prefers `pr.ens` and falls back to `size(pr.X, 1)`; with `ens` dropped
   the set scaled by `T/ens`, measured at `1008/499 ≈ 2.0x` too large on the test fixture.
3. **The same defect through `FactorPrior(; pe = EntropyPoolingPrior(…))`**, which forwarded `w`
   without the `ens` that describes it.

### One behaviour change, deliberate

`HighOrderFactorPriorEstimator(; pe = BlackLittermanPrior(; pe = FactorPrior(…)))` went from throwing
`IsNothingError` to returning numbers: `rr` is structural — the regression of `X` on `F`, over data
Black-Litterman does not modify — so it is now forwarded and the factor block travels with it. The
higher co-moments project through `rr.M` while `mu`/`sigma` carry the views. Black-Litterman makes no
claim about third and fourth moments, so the factor projection is the only estimate available.

Two consequences of that follow through the prose. `assert_prior_regression` used to name two causes;
there is now only one — nothing in the chain ever computed a regression — because discarding one is no
longer possible. And `RegressionFeatures` no longer has to be nested inside a Black-Litterman prior:
its docstring instructed callers to "nest the other way round instead", and that instruction is gone.

## Amendment (2026-07-30): the factor-space plotting entry points guard through the same helper

`assert_prior_regression` now serves two kinds of consumer, not one. `plot_factor_loadings`,
`plot_factor_sigma` and `plot_factor_mu` each carried their own bare `ArgumentError` naming neither the
cause nor the remedy — and the latter two never reached it, because their optional axis-name argument
defaulted to a size taken off the block they were checking for (`1:size(pr.f_sigma, 1)`,
`1:length(pr.f_mu)`), which Julia evaluates before the method body. The one-argument form therefore died
on `size(::Nothing, ::Int64)` / `length(::Nothing)` with the guard sitting unreachable below.

All six prior-taking arities now call `assert_prior_regression`, and their axis-name arguments default to
`nothing`. Only the opening sentence differs between consumers, so it moves to a `lead` keyword; the
diagnosis — one cause, one remedy — is the `prior_regression_remedy` constant, shared verbatim. The
default `lead` stays estimator-framed, because the half of it that explains why the *field type* did not
catch this is only true of an estimator field.

This makes the plotting guards depend on an invariant recorded above: `rr` and `fpr` are provided
together or not at all, so checking `rr` covers a block whose `mu` and `sigma` the caller reaches through
the `f_mu`/`f_sigma` virtual reads. Relaxing that binding would silently stop these three guards covering
what they guard.

## Amendment (2026-08-03): `pr.fpr.mu` is the idiomatic read, and the flat set is frozen

Nesting the factor block left two ways to read the same value, and no rule saying which docs,
examples and library code should use. **`pr.fpr.mu` is the public read.** The flat `f_`-prefixed
names remain, as a compatibility surface for code written against the pre-nesting shape and for
the occasional value-or-`nothing` read.

The argument is not stylistic. The flat surface is **partial**, and always was: `LowOrderPrior`
has eleven fields and six flat names, so `fpr.X` — the factor returns matrix — along with
`fpr.Z`, `fpr.chol` and `fpr.rr` have no flat spelling at all. A surface that cannot express the
whole block cannot be the way to read it. This also settles the question the first amendment above
left open.

**The set is frozen at thirteen** — six on `LowOrderPrior`, seven on `HighOrderPrior`. A field
added to either carrier in future is reachable as `pr.fpr.<name>` and gains no `f_` counterpart, so
"the factor block gains every field for free" stays true of the storage *and* of the read, instead
of becoming an obligation to declare a fourteenth `compute` out of symmetry. Recording the freeze is
what makes a future reviewer's answer "no" rather than "why not?".

The two reads are not interchangeable, and the difference is where the block is absent: `pr.f_mu`
returns `nothing`, `pr.fpr.mu` throws. That is not an argument for the flat names, because the
consumers that read the block are already guarded — `assert_prior_regression` checks `rr`, which
covers `fpr` by the togetherness invariant. Guard, then read through `fpr`.

Applied at the only production reads that were still flat: `plot_factor_sigma` and `plot_factor_mu`
in the Plots extension, whose bodies took `pr.f_sigma`/`pr.f_mu` while their own error messages
named `fpr.sigma`/`fpr.mu`. Everything else in `src/` already read `fpr` directly.

## Amendment (2026-08-03): two estimators report a posterior factor block

The first amendment's warning — "forwarding the factor block through Black-Litterman leaves
`mu != M * fpr.mu + b`" — was written as though the choice were between forwarding a factor block
and dropping it. At two estimators there was a third option that neither the ADR nor the survey had
noticed: **the posterior factor moments were computed and then discarded**, and the prior ones
reported in their place. That is the same defect class this ADR exists to close — a value computed
and silently not carried — arriving from the opposite direction, so the rule applies and both now
report the posterior block.

- [`BayesianBlackLittermanPrior`](../../src/13_Prior/07_BayesianBlackLittermanPrior.jl) builds
  `mu_hat` and `sigma_hat` — the posterior factor mean and *precision* — uses them to reach the
  assets, and previously forwarded the wrapped prior's factor block untouched. It now forwards
  `mu_hat` and `inv(sigma_hat)`. The estimator gains an **`f_mp` field**, defaulting to
  `MatrixProcessing()`, mirroring `FactorBlackLittermanPrior`: the factor posterior covariance is a
  new matrix and wants its own processing, and `pe.mp` runs on the asset block.
- [`AugmentedBlackLittermanPrior`](../../src/13_Prior/09_AugmentedBlackLittermanPrior.jl) solves one
  augmented system over `[assets; factors]` and truncated it to the asset half, discarding a factor
  half that *is* the posterior factor distribution. It now reports that half. No second processing
  pass: `aug_posterior_sigma` is processed as a whole, and a principal submatrix of the result is
  already processed. No `rf` shift and no `b` either — the intercept is the regression's, hence
  asset-only.

**The consistency warning does not survive uniformly, which is the point of the change.**

| estimator | `mu == rr.M * fpr.mu + rr.b` | why |
| --- | --- | --- |
| `FactorBlackLittermanPrior` | exact (`0.0`) | `posterior_mu` *is* `M * f_posterior_mu + b` |
| `BayesianBlackLittermanPrior` | exact (`1.0e-16`) | views update the factors; the assets are their projection |
| `AugmentedBlackLittermanPrior` | no (`3.9e-4`, was `2.6e-2`) | see below |
| `BlackLittermanPrior` | no | views are on the assets; no posterior factor distribution exists |

`BayesianBlackLittermanPrior`'s warning is therefore **deleted**, not softened.
`AugmentedBlackLittermanPrior`'s is **rewritten**, because the first explanation reached for was
wrong. It is not that the asset views enter the asset block directly: muting either view set leaves
a gap of the same order, and the two *priors* satisfy the identity to `2.4e-18` (least squares with
an intercept reproduces the mean). The cause is **idiosyncratic variance**. The augmented covariance
stacks the full `sigma_a`, factor and residual, against a cross-covariance `M * sigma_f` that is pure
factor; the update moves the asset half by `tau * sigma_a * P'(…)` and the factor half by
`tau * sigma_f * M' * P'(…)`, and the two stay related by `M` only if `sigma_a == M * sigma_f * M'`.
Refitting the same fixture on an exact factor model closes the gap to `3.6e-14`.

`BlackLittermanPrior` keeps both warnings unchanged: the observation-weighting split, and a factor
block that is structurally true while distributionally inconsistent. It has nothing better to report,
and each estimator's docstring now names where it sits in that table rather than stating the
inconsistency as though it were uniform across the family.

## Amendment (2026-08-11)

`LowOrderPrior` gains a twelfth field, `o_X`, and a computed read, `original_X`. Issue #288 raised
it: three estimators overwrite `X`, and nothing on the carrier said so.

### What the field records

`FactorPrior`, `FactorBlackLittermanPrior` and `AugmentedBlackLittermanPrior` each set
`X = F * transpose(M) .+ transpose(b)`. Their `X` is therefore a **posterior** matrix — the return
distribution the prior asserts — and not the returns the caller supplied. `o_X` holds the returns the
caller supplied. It is `nothing` on every other route, where `X` already is them.

The two are not interchangeable. Measured on an 8-asset, 3-factor fixture:

| covariance fit on | `rank` | distance from `pr.sigma` |
| --- | --- | --- |
| `pr.X`, the reconstruction | 3 | 7.0e-5 |
| the caller's `X` | 8 | 1.0e-5 |

The reconstruction spans only the factors and carries no residual, so it is singular whenever there
are more assets than factors, and it is *further* from the covariance the carrier reports than the
honest fit is. The Deferred Quantity kernel (#280) was fitting on it. That is now corrected: all
three `fit_deferred_quantity` methods read `original_X`.

### Why two names

`o_X` is the storage and `original_X` is the read. The read is always a matrix — the field where
there is one, `X` where there is not — so no consumer writes a fallback and no consumer can forget
one. The field carries the `nothing`, and not the property, because `forward_prior` rebuilds through
the keyword constructor with **every field named**. A `nothing` is inert under that rebuild. An
always-populated field would be carried past a change to `X` and would then name a matrix that is no
longer the original.

`isnothing(getfield(pr, :o_X))` is the test for whether a carrier reconstructed `X`. The constructor
refuses `o_X === X` so that this state has exactly one encoding.

### The forwarding rule

`o_X` travels **on its own**, and not with the factor block. It needs no plumbing:
`prior_field_values` and `forward_prior` both enumerate `fieldnames`, so a new field forwards
automatically. Two producers rebuild their carrier by explicit destructure rather than through
`forward_prior` — `EntropyPoolingPrior` at two sites and `OpinionPoolingPrior` at one — and those
three were edited by hand. Without that edit, entropy pooling over a factor prior would drop the
original silently.

`o_X` gains **no binding to `X`**, unlike `sigma`-binds-`chol` and `w`-binds-the-diagnostics. The two
existing bindings guard a *stated value* going stale. Here the staleness would be inverted: it is the
**absence** of `o_X` that would become false if a wrapper replaced `X` on a non-factor route. No site
in `src` forwards `X` today, so the case is hypothetical, and a binding whose helper cannot be
`bound_field_is_stale` — which answers `false` for a `nothing` field — is not worth inventing for it.
Add the guard when the first such wrapper arrives.

It does gain a **third binding**, `rr` binds `o_X`, which the existing test suite found rather than
review did. `test_12f_forward_prior.jl` asserts that dropping the whole factor block at once is
valid, and on a factor carrier that now leaves `o_X` standing with nothing to explain it — the
constructor's `rr` requirement refuses it. The refusal is correct, and it belongs at the rule rather
than surfacing from the constructor, so `forward_prior` raises it with the other two:

> Naming `rr` requires naming `o_X`, unless `pr.o_X` is already `nothing`.

Dropping the factor block therefore drops the original with it. The binding is inert on every
carrier that has no `o_X`, which includes each of the factor priors the Black-Litterman estimators
forward — they call `forward_prior` on the *factor* block, which is fit on factors and has no
original of its own.

### The `rr` requirement, and what would relax it

The constructor requires `rr` whenever `o_X` is set. Every estimator that overwrites `X` today does
so by projecting a factor prior through regression loadings, so the loadings are always in hand and a
carrier claiming a reconstruction it cannot explain is a bug.

**This is a present-tense constraint, not a law of the domain.** A future prior that transforms `X`
without a regression — a bootstrap prior, a simulation prior — is the case that relaxes it, and it
must relax it deliberately rather than by discovering the `@argcheck` in the way. This paragraph is
the record that the coupling was chosen with that future in view.

The requirement runs **one way only**. A carrier may hold `rr` with no original, which is the shape
every hand-built factor-block test fixture takes, and `test_12f_forward_prior.jl` pins it. The
biconditional was rejected for exactly that reason: it would refuse every such fixture, and the two
fields answer different questions — `rr` records a projection, `o_X` records what was projected.

### Scope

The factor-risk-contribution path is **not** changed here. It reads its returns from the caller's
`ReturnsResult` rather than from the carrier, so `o_X` simplifies nothing there. It does have a
defect of the same family — `factor_risk_contribution` handed a factor prior reduces `pr.X`, which
has no residual, and reports a **negative** idiosyncratic share — and that has its own ticket.

## Amendment (2026-08-12) — from [#287](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/287)

The 2026-08-11 amendment above does not record where the twelfth field sits. `o_X` was inserted at
**position 2**, next to the `X` it explains, and not appended:

```julia
LowOrderPrior(X, o_X, mu, sigma, chol, w, ens, kld, ow, rr, fpr, Z)
```

So the positional constructor of an exported Result type shifted by one from `mu` onward. This is
an API break. It breaks **loudly** — the old call binds a `VecNum` `mu` to `o_X::Option{<:MatNum}`
and is a `MethodError` — and no call site in the repo, the tests, the examples or the user guide
uses the positional form. The keyword constructor is unchanged.

The position is the right one to keep. `o_X` is meaningless away from the `X` it qualifies, and a
reader of the field list has to see the pair together.

## Amendment (2026-08-12) — from [#289](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/289)

The Scope section of the 2026-08-11 amendment above is superseded. The factor-risk-contribution
path **is** changed, and it reads `original_X`. That amendment's claim that it "reads its returns
from the caller's `ReturnsResult`" is wrong: `factor_risk_contribution` takes the returns from its
own positional argument, and `rd` supplies only the regression.

### The split is by what the consumer does with the matrix

The 2026-08-11 amendment left one reader of `original_X`, the Deferred Quantity kernel, and one
rule: everything that evaluates portfolio returns keeps reading `X`. That rule is too coarse. The
factor attribution evaluates portfolio returns and still needs `original_X`, so the line is drawn
somewhere else.

> A consumer that **evaluates** the return distribution reads `X`. A consumer that **decomposes**
> risk into a factor part and a residual part reads `original_X`.

The reconstruction has rank `size(F, 2)` and carries no residual, so the second consumer has
nothing to attribute to the residual term and reports noise in its place. Measured on a
`FactorPrior` over 8 assets and 3 factors with a `ConditionalValueatRisk` at equal weights:

| `factor_risk_contribution` handed | idiosyncratic share | total risk |
| :-- | --: | --: |
| the `FactorPrior` result | **−0.38 %** | 0.024516 |
| the caller's `X` | +2.47 % | 0.025480 |

The off-factor term was picking up the **intercept's** share. `rank(pr.X)` is `Nf + 1` — the
factors plus the intercept — and for a series-reducing measure the intercept shifts the quantile,
so it lands in the nullspace block. That is why the number is not merely small but signed wrong.

### The price, stated

Under a factor prior the parts now sum to the risk on the caller's returns, and **not** to
`expected_risk(r, w, pr)`. The two were equal before, exactly, and that identity is what was
traded away. It was worth trading: `FactorPrior.rsd` defaults to `true`, so `pr.sigma` already
carries the residual block that `pr.X` cannot. The carrier's own covariance says the idiosyncratic
risk is real, and the reconstruction contradicted it.

The change reaches **only** measures whose kernel reduces the returns series. `Variance`,
`StandardDeviation` and `DistributionValueatRisk` read a moment and never touch `X`, so their
attribution was already correct and is byte-identical either way.

### Two seams, not one argument

`resolve_risk_inputs` keeps returning `pr.X`; `resolve_factor_risk_inputs` returns
`original_returns(X)` beside it, over a three-method accessor that answers for a matrix, a
`ReturnsResult` and a Prior Result alike. A source argument on the one seam was rejected: both
answers are correct, neither is a default of the other, and a new caller picking the wrong arm
would be silent.

### Loadings follow the returns

`resolve_factor_regression(re, rd, pr)` is one precedence, shared by the value-level function and
by `set_factor_risk_contribution_constraints!`:

 1. `re` when it is already a `Regression` result — the caller stating the answer.
 2. `pr.rr` when the prior carries a factor block.
 3. `regression(re, rd)` otherwise.

The prior outranks a refit because `pr.rr` was fitted on `pr.o_X`, which is the matrix the risk is
now measured on, so the pair is matched by construction rather than by the caller's care. The cost
is stated in both docstrings: **a stated regression *estimator* loses to a prior that carries
loadings**, silently. A precomputed `Regression` is the way to override it. The refusal variant was
considered and rejected — it needs a way to spell "`re` was not stated", which a defaulted keyword
does not have.

`pr.rr` and `regression(re, rd)` are dimensionally interchangeable: `M` and `L` are `N × Nf` with
zeros for dropped factors, so the factor axis and the constraint names stay aligned.

### The optimiser takes the prior

`set_factor_risk_contribution_constraints!` gains a `pr` parameter between `rd` and `flag`, and its
`IsNothingError` moves into `resolve_factor_regression`, which now names all three carriers rather
than two. Three call sites — `12_FactorRiskContribution.jl`, `14_RiskBudgeting.jl`,
`15_RelaxedRiskBudgeting.jl` — and each already held the prior. `set_risk_budgeting_constraints!`'s
factor method had the prior as an unnamed `::Any`; it is now `pr::AbstractPriorResult`, which is
what its three sibling methods already declared.

The optimiser is included because the library must say one thing about where the factor model comes
from, and because the basis the weights are re-based onto should be the one the moments were
projected through. It also removes a throw the carrier can answer: `optimise(frc, ReturnsResult())`
under a factor prior now solves, and gives the weights the `rd`-supplied route gives, spread `0.0`.

### What did not change

`plot_factor_risk_contribution` needed no edit. Its value-level arm forwards straight to
`factor_risk_contribution` and inherits everything; its `(r, res, rd)` arm passes `rd.X`, which is
already the caller's returns.

No fixture moved. No test or example of the factor-risk-contribution or risk-budgeting families
uses `FactorPrior`, `FactorBlackLittermanPrior` or `AugmentedBlackLittermanPrior`, so every one has
`pr.rr === nothing` and `original_X === X`.

## Amendment (2026-08-17) — the factor lift gets a home, from candidate F of the 2026-08-16 architecture review

The section *Where the helper does not apply* stands. `forward_prior` is still the wrong shape for
the three factor-axis sites, for the reason recorded there: almost every field changes meaning
across the hop, so there is nothing to forward by default.

What that section did not say is that the hop itself is one algorithm, and that it had no owner.
The same ~22 lines were written three times — a regression, a reconstruction of `X`, a projection of
the factor moments through the loadings, and an optional diagonal residual block. Three copies is
three chances to drift, and one had already drifted.

### The lift is now two functions

The lift splits where [`FactorBlackLittermanPrior`](../../src/13_Prior/08_FactorBlackLittermanPrior.jl)
needs it to split. Its views land on the *factor* distribution, so it must reconstruct `X` before it
has the moments to project.

- `factor_reconstruction(re, X, F)` fits the loadings and returns `(rr, F * transpose(M) .+
  transpose(b))`. All three sites call it.
- `factor_lift(mp, ve, rsd, rr, f_mu, f_sigma, X, posterior_X; kwargs...)` projects the moments,
  processes the covariance, adds the residual block when `rsd` is `true`, and returns
  `(; mu, sigma, chol)`. [`FactorPrior`](../../src/13_Prior/03_FactorPrior.jl) and
  `FactorBlackLittermanPrior` call it. Which factor moments arrive is the only thing that differs
  between them: the wrapped prior's, or the Black-Litterman posterior's.

[`AugmentedBlackLittermanPrior`](../../src/13_Prior/09_AugmentedBlackLittermanPrior.jl) calls the
first and not the second. Its asset moments come out of the augmented system, not out of a
projection, which is the same reason it merges rather than forwards.

### `factor_residual_config` replaces a reach past a type bound

[`HighOrderFactorPriorEstimator`](../../src/13_Prior/12_HighOrderFactorPriorEstimator.jl) needs the
*systematic* covariance for its residual cokurtosis correction, so a residual block the wrapped
estimator added has to come back off. It used to read `pe.pe.ve` and `pe.pe.mp.pdm` directly. Its
`pe` slot is bounded `AbstractLowOrderPriorEstimator_F_AF`, and only `FactorPrior` and
`FactorBlackLittermanPrior` carry those fields, so the read was a live `FieldError` for every other
member of the bound:

```julia
prior(HighOrderFactorPriorEstimator(; pe = FeaturePrior(; pe = FactorPrior(),
                                                        ze = RegressionFeatures())), X, F)
# FieldError: type FeaturePrior has no field `ve`
```

`factor_residual_config(pe)` is a per-type declaration that every prior estimator answers. The two
estimators that own a residual block report `(; ve, pdm, rsd)`; a wrapper over one of them
(`BlackLittermanPrior`, `BayesianBlackLittermanPrior`, `EntropyPoolingPrior`, `FeaturePrior`)
forwards the declaration; everything else — including `OpinionPoolingPrior`, which pools several
priors and has no single one — reports `nothing`. The type bound now guarantees an answer.

### One behaviour change, deliberate

The old guard was `isnothing(pr.chol)`, which is a proxy for "this is `AugmentedBlackLittermanPrior`"
rather than for "a residual block was added". It is wrong whenever the wrapped estimator sets
`rsd = false`: the block was never added, and the correction subtracted it anyway, feeding
`cokurtosis_residuals` a covariance too small by the residual variances. `factor_residual_config`
reports `rsd`, so the correction now leaves the covariance alone. With `FactorPrior(; rsd = false)`
underneath, the posterior cokurtosis changes; it is now identical to the `rsd = true` case, which is
the point — the systematic covariance is the same either way.

### The drifted copy

`FactorPrior`'s `prior` called `prior(pe.pe, F)` and dropped `strict`, while
`FactorBlackLittermanPrior` and `AugmentedBlackLittermanPrior` both passed `strict = strict`.
`FactorPrior.pe` admits `BlackLittermanPrior` and `EntropyPoolingPrior`, both of which resolve view
names against a universe and honour `strict`, so the flag was silently inert on that one route.
`prior(pe::FactorPrior, …)` now declares `strict::Bool = false` and passes it down, which also stops
`strict` from reaching `matrix_processing!` in the `kwargs...` bag — the behaviour the other two
sites already had.

## Amendment (2026-08-17) — `factor_residual_config` has no default, from finding 4 of the 2026-08-17 security review

The amendment above made `factor_residual_config` *total* by giving
`::AbstractPriorEstimator` a fallback that returns `nothing`. The review asked which reading of
"total" the ADR meant: *answers for every type*, or *throws for an undeclared type*. This amendment
settles it as the second.

A `nothing` fallback cannot separate the two statements it is asked to carry:

- this estimator adds no residual block, so leave the covariance alone;
- the author of this estimator forgot the method.

Both read as "leave the covariance alone", so a new estimator that lifts factors and adds a residual
block silently drops that block out of `HighOrderFactorPriorEstimator`'s systematic covariance. The
correction then runs on a covariance that is too large by the residual variances, and the result is
a wrong number rather than an error. Nothing in the codebase notices, because the surface is an
extension author's, not an untrusted input.

The declaration now has the polarity `range_tails` already uses for a per-type declaration whose
absence is a defect rather than an answer:

```julia
function factor_residual_config(pe::AbstractPriorEstimator)
    return throw(ArgumentError("`factor_residual_config` is not defined for `$(nameof(typeof(pe)))`. …"))
end
```

Every one of the eleven concrete prior estimators declares beside its own definition:

| Estimator                       | Declaration          | Why                                        |
| ------------------------------- | -------------------- | ------------------------------------------ |
| `FactorPrior`                   | `(; ve, pdm, rsd)`   | owns the residual block                    |
| `FactorBlackLittermanPrior`     | `(; ve, pdm, rsd)`   | owns the residual block                    |
| `BlackLittermanPrior`           | forwards `pe.pe`     | wrapper                                    |
| `BayesianBlackLittermanPrior`   | forwards `pe.pe`     | wrapper                                    |
| `EntropyPoolingPrior`           | forwards `pe.pe`     | wrapper                                    |
| `FeaturePrior`                  | forwards `pe.pe`     | wrapper                                    |
| `HighOrderPriorEstimator`       | forwards `pe.pe`     | wrapper                                    |
| `HighOrderFactorPriorEstimator` | forwards `pe.pe`     | wrapper                                    |
| `EmpiricalPrior`                | `nothing`            | no factor lift, so no residual block       |
| `AugmentedBlackLittermanPrior`  | `nothing`            | moments come out of the augmented system   |
| `OpinionPoolingPrior`           | `nothing`            | pools several priors, no single wrapped one|

The two high-order estimators declared nothing before this change and reached the old fallback.
They are wrappers, and the low-order block of the result they build *is* the wrapped estimator's
own, residual block and all, so they forward. No live call path changes: the one consumer calls
`factor_residual_config(pe.pe)` with `pe.pe` bounded `AbstractLowOrderPriorEstimator_F_AF`, and no
high-order estimator is a member of that bound.

### The shape is checked before the property access

The consumer reads `ve`, `pdm` and `rsd` off the answer by property access, which has no shape check
of its own. `assert_factor_residual_config(pe, cfg)` now checks the two shapes the contract admits —
`nothing`, or a `NamedTuple` carrying the three keys — and names the estimator that answered, so a
wrong declaration fails at the declaration rather than as a `FieldError` inside the cokurtosis
correction.

### What this does not decide

The load-time preferences channel of finding 1 and the `state_key` collision class of finding 3 are
separate, and are settled in ADR 0041 and ADR 0037 respectively.

## Amendment (2026-08-18) — `OpinionPoolingPrior` forwards `pe2`, from finding 5 of the 2026-08-17 architecture review

The table in the amendment above puts `OpinionPoolingPrior` in the `nothing` row, on the stated
grounds that it "pools several priors, no single wrapped one". The second half of that sentence does
not match the code. `prior(pe::OpinionPoolingPrior, …)` takes **every** moment of its result from the
refit `pe.pe2`:

```julia
pe2 = factory(pe.pe2, w)
(; X, o_X, mu, sigma, chol, rr, fpr, Z) = prior(pe2, X, F; strict = strict, kwargs...)
```

The pooled `pe.pes` contribute observation weights alone — the loop over them reads `pr.w` and
nothing else. So there **is** a single wrapped estimator, it is `pe2`, and the correct declaration is
`factor_residual_config(pe.pe2)`.

The row is corrected to:

| Estimator             | Declaration           | Why                                            |
| --------------------- | --------------------- | ---------------------------------------------- |
| `OpinionPoolingPrior` | forwards `pe.pe2`     | every moment comes from the refit `pe2`        |

### This one changed a live number

`pe2` is bounded `AbstractLowOrderPriorEstimator_A_F_AF`, which admits `FactorPrior`, and
`OpinionPoolingPrior <: AbstractLowOrderPriorEstimator_AF` is itself a member of
`HighOrderFactorPriorEstimator.pe`'s bound `AbstractLowOrderPriorEstimator_F_AF`. Both ends of the
call path are reachable, so the exemption was the class ADR 0046 exists to close: a covariance
carrying a residual block kept it, and the residual cokurtosis correction ran on a covariance too
large by the residual variances.

The witness, on a 200×8 sample lifted off three factors, with
`HighOrderFactorPriorEstimator(; pe = OpinionPoolingPrior(; pes = [EntropyPoolingPrior(), EntropyPoolingPrior()], pe2 = FactorPrior()), rsd = true)`:

| Quantity              | Before      | After       |
| --------------------- | ----------- | ----------- |
| `sum(abs, pr.kt)`     | 8904.51     | 6161.93     |
| largest element moved | —           | 190.78      |
| `pr.sigma`            | unchanged   | unchanged   |

Only the cokurtosis moves. The correction is the sole consumer of the declaration, and it writes
`kt` alone.

### The lesson is about the word *wrapper*

A pooling estimator is a wrapper for this purpose. The count of estimators it *pools* is not the
count of estimators it *wraps*, and the declaration follows the second. The general rule, restated:
an estimator forwards the declaration of whichever estimator its moments come from, however many
priors it also consults for something else.

The load-time half of the review's finding is already closed. `factor_residual_config` throws for an
undeclared type (the 2026-08-17 amendment above), and the census at the end of
`test/test_12g_forwarding_rule.jl` enumerates every concrete `AbstractPriorEstimator` the package
ships and asserts that each declares. What the census cannot catch is a declaration that is present
and wrong, which is what this amendment corrects; that failure mode needs a per-type assertion, and
the file now carries one for `OpinionPoolingPrior`.
