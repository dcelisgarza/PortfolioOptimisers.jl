```@meta
Description = "MIP Constraints, private API of PortfolioOptimisers.jl: AbstractMIPSpace, AbstractMIPIndicators, AssetMIPSpace, SubsetMIPSpace, HeldIndicators, …"
```

# [MIP Constraints: private API](@id private-api-mip-constraints)

```@docs
AbstractMIPSpace
AbstractMIPIndicators
AssetMIPSpace
SubsetMIPSpace
HeldIndicators
LongShortIndicators
SignIndicators
get_mip_ss
set_mip_ss_expr!
mip_key
mip_wx!
mip_bounds
use_direct_mip_indicators
held
held_bin
set_mip_indicators!
mip_indicators
lb_gate
ub_gate
long_gate
short_gate
long_bin
short_bin
mip_wb
declare_held_indicators!
declare_long_short_indicators!
declare_sign_indicators!
short_mip_threshold_constraints
mip_constraints
sign_mip_constraints
run_mip_builder!
set_mip_constraints!
```
