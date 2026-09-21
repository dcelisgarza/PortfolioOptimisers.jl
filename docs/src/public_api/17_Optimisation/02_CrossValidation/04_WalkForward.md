```@meta
Description = "WalkForward, public API of PortfolioOptimisers.jl: WalkForwardResult, IndexWalkForward, DateWalkForward, HindsightSplit, OnlineIndexWalkForward, …"
```

# WalkForward

```@docs
WalkForwardResult
IndexWalkForward
DateWalkForward
HindsightSplit
OnlineIndexWalkForward
OnlineDateWalkForward
OnlineHindsightSplit
Base.split(iwf::IndexWalkForward, rd::Prices_RR)
Base.split(dwf::DateWalkForward{<:Integer}, rd::Prices_RR)
Base.split(dwf::DateWalkForward{<:Any}, rd::Prices_RR)
Base.split(hs::HindsightSplit, rd::Prices_RR)
n_splits
n_splits(dwf::DateWalkForward{<:Integer}, rd::ReturnsResult)
n_splits(dwf::DateWalkForward{<:Any}, rd::ReturnsResult)
Base.split(o::Online{<:WalkForwardEstimator}, rd::Prices_RR)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
