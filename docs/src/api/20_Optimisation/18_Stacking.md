# Stacking

```@docs
BaseStackingOptimisationEstimator
StackingResult
factory(res::StackingResult, fb::Option{<:OptE_Opt})
Stacking
needs_previous_weights(opt::Stacking)
factory(st::Stacking, w::AbstractVector)
port_opt_view(st::Stacking, i, X::MatNum, args...)
predict_outer_st_estimator_returns
assert_special_nco_requirements(opt::Stacking)
optimise(st::Stacking{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                               <:Any, <:Any, Nothing}, rd::ReturnsResult; dims::Int = 1,
                  branchorder::Symbol = :optimal, str_names::Bool = false,
                  save::Bool = true, kwargs...)
```
