```@meta
Description = "Entropy Pooling, private API of PortfolioOptimisers.jl: StagedEP, NonCVaREP, AbstractEntropyPoolingOptimiser, AbstractEntropyPoolingAlgorithm, …"
```

# Entropy Pooling: private API

```@docs
StagedEP
NonCVaREP
AbstractEntropyPoolingOptimiser
AbstractEntropyPoolingAlgorithm
AbstractEntropyPoolingOptAlgorithm
AbstractEntropyPoolingViewEstimator
VV_VecVV
AbstractEntropyPoolingViewFormulation
AbstractConditionalValueatRiskViewFormulation
AbstractEntropicValueatRiskViewFormulation
AbstractRelativisticValueatRiskViewFormulation
CVaRVF_VecCVaRVF
EVaRVF_VecEVaRVF
RLVaRVF_VecRLVaRVF
AbstractEntropyPoolingTailViewEstimator
CVV_VecCVV
EVV_VecEVV
RVV_VecRVV
AbstractEntropyPoolingTailView
VecEPTV
add_ep_constraint!
announce_ep_departures
replace_prior_views
replace_coprior_views
get_pr_value
ep_mu_views!
fix_mu!
ep_var_views!
ep_prior_probabilities
entropy_pooling
ep_jump_entropy_pooling
ep_refine_iters
ep_refine_tail_view(tv::AbstractEntropyPoolingTailView, ::VecNum)
ep_check_tail_window(::AbstractEntropyPoolingTailView, ::VecNum)
ep_sigma_views!
fix_sigma!
ep_cov_views!
ep_rho_views!
ep_sk_views!
ep_kt_views!
ep_jump_views!
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
