# Entropy Pooling

```@docs
RhoParsingResult
H0_EntropyPooling
H1_EntropyPooling
H2_EntropyPooling
LogEntropyPooling
ExpEntropyPooling
CVaREntropyPooling
OptimEntropyPooling
JuMPEntropyPooling
EntropyPoolingPrior
prior(pe::EntropyPoolingPrior{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any,
                                       <:StagedEP},
               X::MatNum, F::Option{<:MatNum} = nothing;
               dims::Int = 1, strict::Bool = false, kwargs...)
prior(pe::EntropyPoolingPrior{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                              <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                              <:H0_EntropyPooling}, X::MatNum;
      F::Option{<:MatNum} = nothing, dims::Int = 1, strict::Bool = false,
      kwargs...)
AbstractEntropyPoolingOptimiser
AbstractEntropyPoolingAlgorithm
AbstractEntropyPoolingOptAlgorithm
add_ep_constraint!
replace_prior_views
get_pr_value
ep_mu_views!
fix_mu!
ep_var_views!
entropy_pooling
ep_cvar_views_solve!
ep_sigma_views!
fix_sigma!
ep_rho_views!
ep_sk_views!
ep_kt_views!
```
