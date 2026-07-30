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
