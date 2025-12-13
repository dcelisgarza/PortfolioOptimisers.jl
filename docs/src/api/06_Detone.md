# Detone

Financial data is often responds to broad market conditions. This market-wide behaviour can obscure specific correlation signals. By removing the largest `n` eigenvalues, the idiosyncratic relationships between assets are allowed to shine through [mlp1](@cite).

Detoned matrices may be non-positive definite, so they can be unsuitable for traditional optimisations, but they can be quite effective for clustering ones.

```@docs
AbstractDetoneEstimator
Detone
detone!
detone
```
