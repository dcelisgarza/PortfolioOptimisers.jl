---
status: accepted
---

# The risk-free rate is one round trip, and it is added once

## Context

Four estimators carry an `rf` field and run a Black-Litterman update:
[`BlackLittermanPrior`](../../src/13_Prior/06_BlackLittermanPrior.jl),
[`BayesianBlackLittermanPrior`](../../src/13_Prior/07_BayesianBlackLittermanPrior.jl),
[`FactorBlackLittermanPrior`](../../src/13_Prior/08_FactorBlackLittermanPrior.jl) and
[`AugmentedBlackLittermanPrior`](../../src/13_Prior/09_AugmentedBlackLittermanPrior.jl). All
four documented the field with one line: "`rf`: Risk-free rate."

A Black-Litterman update is written in **excess** returns. The view returns in `Q` are excess
returns, and the equilibrium mean `l \Sigma w` is a risk premium, so it is one by construction.
A mean taken from a wrapped prior estimator is a total return, so it must lose the rate before
it can be blended against either of them, and get it back afterwards. That is one round trip,
and the field names it.

The round trip was never written down, and the code did not agree on it. The shared kernel
`vanilla_posteriors` took `rf` and added it to the posterior mean, so the *add* was owned by
the kernel and no member could opt out of it. Each member then wrote arithmetic of its own on
top:

| member                         | `l === nothing`                                    | `l` set                             |
| ------------------------------ | -------------------------------------------------- | ----------------------------------- |
| `BlackLittermanPrior`          | the kernel's add, once                             | no `l` field                        |
| `BayesianBlackLittermanPrior`  | its own add, once, no kernel call                  | no `l` field                        |
| `FactorBlackLittermanPrior`    | `.- rf` on the wrapped mean, then the kernel's add | the kernel's add, once              |
| `AugmentedBlackLittermanPrior` | `.- rf`, the kernel's add, then a second add       | the kernel's add, then a second add |

Three defects follow.

1. **The rate is added twice.** `AugmentedBlackLittermanPrior` adds it inside the kernel and
   again on the asset half. With `l` set nothing subtracts either add, so the posterior carries
   `2 rf`.
2. **The conversion is invisible.** The `.- rf` in the two factor members is the *correct*
   move — it is what puts the prior mean on the excess scale the views are written on — but it
   is unnamed, uncommented, and absent from the other two members and from the field
   documentation. A reader cannot tell a deliberate conversion from a stray subtraction, and a
   later reader deleted it as one.
3. **The rate is projected rather than added.** Applying `rf` to a *factor* posterior and then
   lifting it through the loadings gives `M (rf \mathbf{1})`, which is a projection of the rate
   and not a rate.

The default `rf = 0.0` hid all three. No shipped test, example or user guide sets a non-zero
`rf` on a Black-Litterman prior; the `rf` the examples do set belongs to `MaximumRatio`.

A second expression had the same shape. The equilibrium mean `l \Sigma w` was written three
times — in `EquilibriumExpectedReturns`, `FactorBlackLittermanPrior` and
`AugmentedBlackLittermanPrior` — with two different equal-weight fallbacks (`fill` and
`range`) and two different validation regimes. `FactorBlackLittermanPrior.w` also lacked the
`@vprop` tag its sibling `AugmentedBlackLittermanPrior.w` carries, on the same
`field_dict[:eqw]` field, so a `port_opt_view` passed full-universe weights into a subset
problem.

## Decision

**The risk-free rate is one round trip: off the prior mean before the update, back on the
posterior asset mean after it, once each.**

Two named functions own the two directions, and nothing else touches the field.

- `remove_rf(rf, mu)` is the only site that subtracts. It converts a total-return mean to an
  excess-return mean.
- `apply_rf(rf, mu)` is the only site that adds. It converts back, on the asset expected
  returns the estimator returns.

Three properties follow, and all three are contracts of the family:

- **Added once.** No member adds the rate twice, and the add is the last thing that happens to
  the returned mean.
- **Converted only where a conversion is due.** `remove_rf` is reached by
  `FactorBlackLittermanPrior` and `AugmentedBlackLittermanPrior`, and only where `l` is
  `nothing`. Those two are the members that can build the equilibrium mean instead, so their
  prior mean must reach the kernel on one known scale either way; where `l` is set the mean is
  a risk premium and needs no conversion. `BlackLittermanPrior` and
  `BayesianBlackLittermanPrior` have no equilibrium branch and take the wrapped mean on the
  scale it is given.
- **Asset-axis.** The rate is a property of the asset axis, so the factor block of a result
  never carries it. Where the rate came off a factor mean, nothing puts it back: the factor
  block is reported on the excess scale the update ran on. `FactorBlackLittermanPrior`
  therefore adds the rate *after* the lift, not before it.

A prior stays **isolated** under this rule. The round trip returns a wrapped prior's mean to
the scale it arrived on, so a rate an inner estimator applied internally is never undone, and
the rate that comes back on is the one that came off. Nesting is composition, not negotiation.

`vanilla_posteriors` carries no rate. Its signature drops the `rf` argument, so the kernel is
the Black-Litterman update and nothing else, and each member owns both ends of its own round
trip.

`equilibrium_mu(l, sigma, w)` is the single owner of `l \Sigma w`. `sigma` is a covariance
*block* whose columns are the assets the weights are written over, which is what lets the two
factor members build a prior mean over factors from asset weights through the same function.
The equal-weight fallback is `fill(inv(N), N)` and the length check is `length(w) ==
size(sigma, 2)`, both stated once. `FactorBlackLittermanPrior.w` gains the `@vprop` tag.

## Consequences

`rf` is learned once rather than four times, and the docstring field text is one shared
`field_dict[:bl_rf]` entry that states the round trip.

**The round trip is exact on one axis and not on two.** This is the sharpest consequence and
is asserted in the tests both ways.

- `AugmentedBlackLittermanPrior` takes the rate off the asset rows of its augmented prior and
  puts it back on the same rows. Under views whose every `P` row sums to zero — relative
  views — the view residual `Q - P \mu` is unchanged by the shift, so the rate cancels
  completely and `mu(r) == mu(0)` exactly.
- `FactorBlackLittermanPrior` takes the rate off the *factor* axis and adds it on the *asset*
  axis, after the lift. The two moves are not inverses: the subtraction reaches the assets
  through the loadings. Writing `s` for the row sums of `rr.M`, its answer moves by
  `rf (\mathbf{1} - s)` against the same estimator at `rf = 0`, and cancels only for an asset
  whose loadings sum to one. This is inherent in the field meaning a rate on the asset axis
  while the update runs on factors; it is recorded rather than removed.

**This changes numbers.** Only for a non-zero `rf`, which nothing shipped sets on these
estimators:

- `AugmentedBlackLittermanPrior` no longer doubles the rate.
- With `l` set, both factor members shift by exactly `rf`, where the augmented one previously
  shifted by `2 rf`.
- The factor block of a result no longer has the rate added to it. Where `l` is `nothing` it
  now sits `rf` *below* the scale its factor prior was supplied on, because the conversion is
  one-way on that axis.
- The internal-consistency identity the carriers document reads
  `mu == rr.M * fpr.mu + rr.b + rf`. At the default `rf = 0.0` it is the plain identity the
  tests already assert.

The `l \Sigma w` consolidation changes no numbers. `EquilibriumExpectedReturns` keeps its
`fill` fallback, so its output is unchanged bit for bit. The two factor members move from
`range` to `fill` on a path that a non-`nothing` `l` is the only way to reach.

The convention is enforced by a text census in `test/test_12b_prior_core.jl`: each of the four
sources reads `pe.rf` the number of times its row of the table above allows, every read is an
`apply_rf` or a `remove_rf` call rather than hand-written arithmetic, both verbs are declared
once, none of the members writes `pe.l *`, and `vanilla_posteriors` declares no `rf`.

## Related

- ADR 0046 governs what a prior *forwards*. This ADR governs the arithmetic a prior applies to
  its own posterior; the two do not overlap.

## Amendment (2026-08-27)

**The update runs on the scale the views are written on, so a level the prior mean lacks is
added before the blend and never after it.** This reverses the direction of the decision above
for the two members that can build an equilibrium mean, and it closes issue #570.

### Why the original direction was wrong

A Black-Litterman update blends the prior mean against the view returns by forming the residual
`q - P \mu`. The caller writes those view returns about the assets and factors as the data
reports them, so `q` is a total return, and `\mu` must be on the same scale when the residual is
formed. Converting `\mu` to an excess scale, blending, and converting back does not undo itself,
because the update is affine and not a translation: a shift of `-r_f` on the whole stack reaches
the asset half as `-(I - GP)r_f \mathbf{1}`, not as `-r_f \mathbf{1}`. The decision above
recorded that non-cancellation as a property to document. It is better read as evidence that the
conversion sat on the wrong side of the blend.

The same argument applies to the regression intercept, and there it was doing visible damage.
`AugmentedBlackLittermanPrior` added `rr.b` to the asset half *after* the update, on both
branches. On the `l === nothing` branch the prior asset mean is the mean of `X`, and least
squares with an intercept forces `mean(X) = M \mu_f + b`, so the intercept was in that mean
already. The estimator counted it twice, and the returned `mu` sat `b` above the mean the
augmented system produced — the same order as the expected returns themselves on daily data.
Issue #570 carries the reproduction. The published reference has the same asymmetry: its
equilibrium branch builds a premium by reverse optimisation and then adds the loadings constant,
which is right, and its historical branch adds the constant to a mean that already contains it.

### Decision

**Every prior mean reaches `vanilla_posteriors` as a total return, on the axis it lives on, and
nothing is added to the answer afterwards.**

- A mean taken from a wrapped prior estimator is one already, and reaches the update untouched.
  `remove_rf` loses its last caller and is deleted.
- An equilibrium mean is a bare risk premium, so the branch that builds one adds what it lacks
  before the update: `pe.rf` through `apply_rf`, and, in `AugmentedBlackLittermanPrior`, the
  asset-side intercept `rr.b`.
- `apply_rf` stays the single site that reads the field. `BlackLittermanPrior` and
  `BayesianBlackLittermanPrior` have no equilibrium branch and so have nothing to convert. They
  keep adding the rate to the posterior asset mean, last, where the field is a plain shift of
  the answer.

### Consequences

**`rf` no longer reaches the answer on the default branch of the two factor members.** With
`l === nothing`, `FactorBlackLittermanPrior` and `AugmentedBlackLittermanPrior` read the field
nowhere: their prior mean is the wrapped one, which needs no conversion, and there is no
posterior-side add. Two fits differing only in `rf` now agree bit for bit. This is a deliberate
narrowing of what the field does, and it is the honest consequence of the field naming a
conversion rather than a shift. Widening it again — by giving those two a posterior-side add
like their siblings' — is a separate decision, and it would reintroduce a rate the views were
not blended against.

**This changes numbers at the default `rf = 0.0`,** which the ADR above did not.

- `AugmentedBlackLittermanPrior` with `l === nothing` drops the duplicate intercept, so its
  `mu` moves by `rr.b`. On the shipped `AugmentedBlackLitterman` fixture that is `1.6e-3`
  against a largest mean of `2.8e-3`.
- `AugmentedBlackLittermanPrior` with `l` set carries the intercept through the blend as
  `(I - GP)b` rather than adding it whole afterwards, so its `mu` moves by `7.0e-4` on the same
  fixture against a largest mean of `2.0e-3`.
- No covariance moves. The update's covariance path does not read the prior mean, and both
  columns of the fixture match to `0.0` after the change.
- `FactorBlackLittermanPrior` moves only for a non-zero `rf`.

**It departs from the published reference on `mu`.** `test/assets/AugmentedBlackLitterman.csv.gz`
is that reference, and the library matched it bit for bit before this amendment. The golden test
keeps it: it still checks `sigma` against it exactly, states the `l === nothing` departure as the
exact vector `reference - rr.b`, and pins the `l` departure as one measured number. A synthetic
exact factor model, where the identity below is analytic, carries the proof of the mechanism.

**The identity `mu == rr.M * fpr.mu + rr.b` gains a branch and loses a cause.** The
`AugmentedBlackLittermanPrior` warning named two causes for the gap. The intercept cause is
gone on both branches, and idiosyncratic variance is the only one left. On an exact factor
model, where the augmented prior stack satisfies the identity and the update is affine, it now
holds to machine precision whatever the intercept and whatever `l`.

### Related

- Issue #570 reported the double-counted intercept, and sweep ticket #536 documented it before
  this amendment reversed it.
