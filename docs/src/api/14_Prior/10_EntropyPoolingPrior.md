# Entropy Pooling

```@docs
H0_EntropyPooling
H1_EntropyPooling
H2_EntropyPooling
LogEntropyPooling
ExpEntropyPooling
CVaREntropyPooling
OptimEntropyPooling
JuMPEntropyPooling
EntropyPoolingPrior
prior(pe::EntropyPoolingPrior{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                  <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                  <:H0_EntropyPooling}, X::NumMat;
          F::Union{Nothing, <:NumMat} = nothing, dims::Int = 1, strict::Bool = false, kwargs...)
prior(pe::EntropyPoolingPrior{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                              <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                              <:Union{<:H1_EntropyPooling, <:H2_EntropyPooling}}, X::NumMat;
      F::Union{Nothing, <:NumMat} = nothing, dims::Int = 1, strict::Bool = false, kwargs...)
PortfolioOptimisers.AbstractEntropyPoolingOptimiser
PortfolioOptimisers.AbstractEntropyPoolingAlgorithm
PortfolioOptimisers.AbstractEntropyPoolingOptAlgorithm
PortfolioOptimisers.get_epw
PortfolioOptimisers.add_ep_constraint!
PortfolioOptimisers.replace_prior_views
PortfolioOptimisers.get_pr_value
PortfolioOptimisers.ep_mu_views!
PortfolioOptimisers.fix_mu!
PortfolioOptimisers.ep_var_views!
PortfolioOptimisers.entropy_pooling
PortfolioOptimisers.ep_cvar_views_solve!
PortfolioOptimisers.ep_sigma_views!
PortfolioOptimisers.fix_sigma!
PortfolioOptimisers.ep_rho_views!
PortfolioOptimisers.ep_sk_views!
PortfolioOptimisers.ep_kt_views!
```
