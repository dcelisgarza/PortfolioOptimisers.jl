---
status: accepted
---

# A factor attribution's predicted totals anchor on the prior result, and the gap goes to the remainder

## Context

A factor model states two identities over the assets it describes:

```text
sigma = M F M' + D
mu    = M mu_f + b
```

`M` is the loadings, `F` the factor covariance, `D` the idiosyncratic covariance, `mu_f` the
expected factor returns and `b` the factor-orthogonal expected return. A predicted factor
attribution decomposes a portfolio's volatility and expected return through those two identities:
the systematic part is what the factors explain, and the idiosyncratic part is what the assets'
own residuals explain.

The two identities hold on a prior result the factor model itself produced. They do **not** hold on
a prior result that a **wrapping** estimator produced. Under
[ADR 0046](0046-wrapping-priors-forward-by-default-and-document-every-drop.md) every wrapping prior
forwards `rr` and `fpr` unchanged while it replaces `mu` and `sigma`:
`EntropyPoolingPrior` and `OpinionPoolingPrior` re-weight the observations and recompute both
moments, and `AugmentedBlackLittermanPrior` breaks the identity for the two causes its own
docstring measures. The block that reaches the attribution therefore describes a distribution the
carrier no longer holds.

So a predicted attribution has two candidates for its total, and they disagree:

1. **The model's own total**, `w' (M F M' + D) w` and `w' (M mu_f + b)`. The systematic and
   idiosyncratic parts then sum to it exactly, and there is nothing left over.
2. **The carrier's total**, `w' pr.sigma w` and `w' pr.mu`. This is what the optimiser saw, what
   `expected_return` reports and what `expected_risk(Variance(), w, pr)` reports.

The reference implementation takes the first, because its predicted attribution takes the model's
five arrays as arguments and never sees a carrier. Its Result has no place to put a difference, so
it cannot show one.

## Decision

**The predicted totals anchor on the prior result, and the two gaps land in the unattributed
remainder.**

`factor_attribution(w, pr)` reports:

```text
total.vol_contrib = sqrt(w' pr.sigma w)                    = sigma_P
total.mu_contrib  = w' pr.mu
sys.vol_contrib   = (M' w)' F (M' w) / sigma_P
idio.vol_contrib  = w' D w / sigma_P
sys.mu_contrib    = (M' w)' mu_f
idio.mu_contrib   = w' b
unattr.vol_contrib = w' (pr.sigma - M F M' - D) w / sigma_P
unattr.mu_contrib  = w' (pr.mu - M mu_f - b)
```

The four components therefore sum to the total by construction, on the predicted side exactly as
they do on the realised side. The remainder is at rounding level on a plain fit, because the model
reproduces its own carrier, and it is the measured gap under a wrapper.

Three consequences follow from the anchoring, and each is deliberate.

- **The remainder is present on the predicted side.** The reference's predicted Result carries no
  remainder at all. This one does, and it is the reader's only signal that the block and the
  carrier have parted company.
- **No guard reports the gap.** A threshold on the remainder would be a numerical guard on a
  quantity whose size is a property of the caller's prior, not of the arithmetic. The docstring
  names every source of the gap and states that a large `pct_var` on the remainder means the model
  does not explain the portfolio. The reader draws the conclusion.
- **The remainder has no series, so it has no volatility.** `unattr.vol` and `unattr.corr` are
  `NaN` on the predicted side, because the gap is a difference between two moments and not a return
  series. On the realised side both are finite, because there the remainder **is** a series.

**The asset axis decomposes the model, not the anchors.** Each asset row reads `M F M' + D` and
`M mu_f + b`, so the systematic rows sum to the systematic component, the idiosyncratic rows to the
idiosyncratic component, and their sum to the two together. The asset rows therefore do **not**
reach the total: the difference is the remainder, which is a property of the portfolio and admits
no per-asset split. The realised asset axis satisfies the same identity, so a reader compares the
two sides row by row.

## Alternatives rejected

- **Anchor on the model and report no remainder**, as the reference does. Refused because the
  numbers would then disagree with `expected_return` and `expected_risk` on the same weights and
  the same prior, with nothing in the Result to say why. A reader who tabulates an attribution
  beside a performance summary would find two different portfolio returns.
- **Route the gap into the idiosyncratic component.** Refused because the per-asset idiosyncratic
  rows would no longer sum to it, and the asset axis would lose the one identity that makes it
  readable.
- **Refuse a carrier whose block does not reproduce it.** Refused because that is a numerical guard
  on a threshold nobody can set: the gap of an entropy-pooling tilt is large and correct, and the
  attribution of such a carrier is exactly what a caller who tilted their prior wants to read.
- **Split the asset axis's share of the remainder over the assets.** Refused because there is no
  such split. The remainder is what the model does not explain, and the model is what assigns a
  quantity to an asset.

## Consequences

A reader of a predicted attribution reads four components rather than three, and the fourth is at
rounding level whenever the prior is a plain factor fit. A reader of a wrapped prior sees the
wrapper's gap as a number rather than as a silent disagreement with the optimiser's own figures.

The two sides of the verb now state one identity each, and they are the same identity: the four
components sum to the total, and the asset rows sum to the two components the model explains.

[Issue #782](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/782) built the verb, and
[issue #708](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/708) decided it. The rule
is decision 5 of that issue's resolution.

## Amendment (2026-09-06)

Issue #844 re-examined the asset axis and kept the decision. It corrects one sentence of the
reasoning, which was too strong.

The fourth rejected alternative says that the asset axis's share of the remainder has no split,
because "there is no such split". That is false as a statement of the mathematics. The remainder
is a quadratic form, `w' (pr.sigma - M F M' - D) w / sigma_P`, and Euler's identity splits any
quadratic form over the assets exactly: the sum over `i` of `w_i ((pr.sigma - M F M' - D) w)_i`
is that form. Anchoring the asset axis on `pr.sigma` would therefore make the rows reach the total,
and each asset's `vol_contrib` would then equal the Euler risk contribution to the variance that
`risk_contribution` reports.

The split exists, and the library still refuses it, for the reason the decision gives elsewhere:
the realised asset axis has no carrier moments to read, so anchoring the predicted axis on
`pr.sigma` would make the two sides state different identities, and a reader who compares a
predicted attribution against a realised one over the same prior would find the asset rows summing
to different things. What the carrier split also lacks is an interpretation: under an
entropy-pooling tilt the difference matrix is a change of observation weights, and an asset's
Euler share of it is a well-defined number that describes no property of the asset.

The choice changes numbers only when the carrier's moments differ from the block's model. On a
plain fit with the default matrix processing the two anchors agree to rounding. The gap opens
under a wrapping prior, under a matrix processing that denoises or detones the lifted covariance,
and under a `FactorPrior` with `rsd = false`, where the block carries no residual variance. When
it is open, the totals, the four components, the factor rows, the family rows, the per-asset
systematic and idiosyncratic rows and the asset-by-factor matrices are the same under both anchors.
Only the per-asset `vol_contrib`, `pct_var` and `mu_contrib` move, by that asset's Euler share of
the remainder.
