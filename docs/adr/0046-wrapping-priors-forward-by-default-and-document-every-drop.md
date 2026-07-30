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
