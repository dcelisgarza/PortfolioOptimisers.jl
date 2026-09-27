```@meta
Description = "Base finite allocation, public API of PortfolioOptimisers.jl: AbstractCollateralAlgorithm, ProceedsCollateral, CashCollateral, FiniteAllocationInput, …"
```

# Base finite allocation

A new collateral algorithm subtypes `AbstractCollateralAlgorithm` and adds the two call methods that its docstring states.

```@docs
PortfolioOptimisers.AbstractCollateralAlgorithm
ProceedsCollateral
CashCollateral
FiniteAllocationInput
factory(res::FiniteAllocationOptimisationResult, fb::Option{<:FOptE_FOpt_FbChain})
```
