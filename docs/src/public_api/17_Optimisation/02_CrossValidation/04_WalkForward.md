```@meta
Description = "WalkForward, public API of PortfolioOptimisers.jl: WalkForwardResult, OnlineStep, IndexWalkForward, DateWalkForward, HindsightSplit, Base.split, n_splits."
```

# WalkForward

```@docs
WalkForwardResult
OnlineStep
IndexWalkForward
DateWalkForward
HindsightSplit
Base.split(iwf::IndexWalkForward, rd::Prices_RR)
Base.split(dwf::DateWalkForward{<:Integer}, rd::Prices_RR)
Base.split(dwf::DateWalkForward{<:Any}, rd::Prices_RR)
Base.split(hs::HindsightSplit, rd::Prices_RR)
n_splits
n_splits(dwf::DateWalkForward{<:Integer}, rd::ReturnsResult)
n_splits(dwf::DateWalkForward{<:Any}, rd::ReturnsResult)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
