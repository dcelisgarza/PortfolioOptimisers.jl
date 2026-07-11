# Subset resampling

```@docs
BaseSubsetResamplingOptimisationEstimator
SubsetResamplingResult
factory(sr::SubsetResamplingResult, fb::Option{<:OptE_Opt})
SubsetResampling
needs_previous_weights(opt::SubsetResampling)
is_time_dependent(opt::SubsetResampling)
update_time_dependent_estimator(opt::SubsetResampling, ctx::TimeDependentContext, all_binds::Bool = true)
reset_time_dependent_estimator(opt::SubsetResampling)
factory(sr::SubsetResampling, w::AbstractVector)
port_opt_view(sr::SubsetResampling, i, X::MatNum, args...)
optimise(sr::SubsetResampling{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  branchorder::Symbol = :optimal, str_names::Bool = false,
                  save::Bool = true, kwargs...)
subset_resampling_finaliser
subset_resampling_retcode
```
