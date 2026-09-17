```@meta
Description = "Base clustering optimisation, public API of PortfolioOptimisers.jl: HierarchicalResult, HierarchicalRiskParityResult, …"
```

# Base clustering optimisation

```@docs
HierarchicalResult
HierarchicalRiskParityResult
HierarchicalEqualRiskContributionResult
HierarchicalOptimiser
factory(res::HierarchicalRiskParityResult, fb::Option{<:OptE_Opt_FbChain})
factory(res::HierarchicalEqualRiskContributionResult, fb::Option{<:OptE_Opt_FbChain})
```
