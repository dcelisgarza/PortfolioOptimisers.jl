# Entropy Pooling

```@docs
RhoParsingResult
H0_EntropyPooling
H1_EntropyPooling
H2_EntropyPooling
StagedEP
LogEntropyPooling
ExpEntropyPooling
ConditionalValueatRiskEntropyPooling
OptimEntropyPooling
JuMPEntropyPooling
NonCVaREP
MeucciEntropyPoolingPrior
VecMeucciEP
prior(pe::MeucciEntropyPoolingPrior, X::MatNum, F::Option{<:MatNum} = nothing;
      dims::Int = 1, strict::Bool = false, kwargs...)
ep_prior(alg::StagedEP, pe::MeucciEntropyPoolingPrior, X::MatNum, F::Option{<:MatNum};
         strict::Bool = false, kwargs...)
ep_prior(alg::H0_EntropyPooling, pe::MeucciEntropyPoolingPrior, X::MatNum,
         F::Option{<:MatNum}; strict::Bool = false, kwargs...)
AbstractEntropyPoolingOptimiser
AbstractEntropyPoolingAlgorithm
AbstractEntropyPoolingOptAlgorithm
add_ep_constraint!
replace_prior_views
replace_coprior_views
get_pr_value
ep_mu_views!
fix_mu!
AbstractEntropyPoolingViewEstimator
ValueatRiskView
VV_VecVV
AbstractEntropyPoolingViewFormulation
AbstractConditionalValueatRiskViewFormulation
AbstractEntropicValueatRiskViewFormulation
LinearConditionalValueatRiskView
IntegerConditionalValueatRiskView
ConicEntropicValueatRiskView
GridEntropicValueatRiskView
CVaRVF_VecCVaRVF
EVaRVF_VecEVaRVF
AbstractEntropyPoolingTailViewEstimator
ConditionalValueatRiskView
EntropicValueatRiskView
CVV_VecCVV
EVV_VecEVV
ep_var_views!
entropy_pooling
ep_cvar_views_solve!
ep_sigma_views!
fix_sigma!
ep_cov_views!
ep_rho_views!
ep_sk_views!
ep_kt_views!
```

## Tail views

```@docs
AbstractEntropyPoolingViewFormulation
AbstractConditionalValueatRiskViewFormulation
AbstractEntropicValueatRiskViewFormulation
LinearConditionalValueatRiskView
IntegerConditionalValueatRiskView
ConicEntropicValueatRiskView
GridEntropicValueatRiskView
CVaRVF_VecCVaRVF
EVaRVF_VecEVaRVF
AbstractEntropyPoolingTailView
LinearConditionalValueatRiskViewConstraint
IntegerConditionalValueatRiskViewConstraint
ConicEntropicValueatRiskViewConstraint
GridEntropicValueatRiskViewConstraint
VecEPTV
ep_evar
ep_evar_grid_row
ep_view_terms
ep_normalise_view_term
ep_view_formulations
ep_sbar
ep_assert_reachable_view
ep_cvar_formulation
ep_evar_formulation
ep_add_cvar_view!
ep_add_evar_view!
ep_cvar_views!
ep_evar_views!
add_ep_tail_view!
ep_jump_views!
EntropyPoolingPrior
VecEP
prior(pe::EntropyPoolingPrior, X::MatNum, F::Option{<:MatNum} = nothing;
      dims::Int = 1, strict::Bool = false, kwargs...)
ep_prior(alg::StagedEP, pe::EntropyPoolingPrior, X::MatNum, F::Option{<:MatNum};
         strict::Bool = false, kwargs...)
ep_prior(alg::H0_EntropyPooling, pe::EntropyPoolingPrior, X::MatNum,
         F::Option{<:MatNum}; strict::Bool = false, kwargs...)
```
