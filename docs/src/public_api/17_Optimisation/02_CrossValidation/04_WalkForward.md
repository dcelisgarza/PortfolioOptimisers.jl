```@meta
Description = "WalkForward, public API of PortfolioOptimisers.jl: WalkForwardResult, IndexWalkForward, DateWalkForward, DateAdjusterEstimator, HindsightSplit, …"
```

# WalkForward

```@docs
WalkForwardResult
IndexWalkForward
DateWalkForward
PortfolioOptimisers.DateAdjusterEstimator
HindsightSplit
OnlineIndexWalkForward
OnlineDateWalkForward
OnlineHindsightSplit
Base.split(iwf::IndexWalkForward, rd::Prices_RR)
Base.split(dwf::DateWalkForward, rd::Prices_RR)
Base.split(hs::HindsightSplit, rd::Prices_RR)
n_splits
n_splits(dwf::DateWalkForward, rd::Prices_RR)
Base.split(o::Online{<:WalkForwardEstimator}, rd::Prices_RR)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
