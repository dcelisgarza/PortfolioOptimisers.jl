```@meta
Description = "Cross-Sectional Factor Prior internals, private API of PortfolioOptimisers.jl: cross_sectional_prior_pairs, cross_sectional_benchmark_carrier, …"
```

# Cross-Sectional Factor Prior internals: private API

The functions below are the steps of a [`CrossSectionalFactorPrior`](@ref) fit. They compute these parts of it.

- The benchmark weights that its exposure estimators read, which the prior computes from the market capitalisation.
- The history of each factor exposure, in an order that puts each exposure after the exposures it is derived from.
- The active mask and the estimation mask of the asset panel.
- The idiosyncratic covariance of the latest observation, and the degrees of freedom and the divisor of each idiosyncratic variance.
- The return scenarios of the assets.
- The moments of the assets, from the moments of the factors and the loadings.
- The forecast of the expected returns, when the prior has a return forecast estimator.

```@docs
PortfolioOptimisers.cross_sectional_prior_pairs
PortfolioOptimisers.cross_sectional_benchmark_carrier
PortfolioOptimisers.cross_sectional_exposure_order
PortfolioOptimisers.cross_sectional_exposure_widths
PortfolioOptimisers.cross_sectional_exposure_write!
PortfolioOptimisers.cross_sectional_exposure_history
PortfolioOptimisers.cross_sectional_exposures_finite
PortfolioOptimisers.cross_sectional_warmup
PortfolioOptimisers.cross_sectional_eligible
PortfolioOptimisers.assert_cross_sectional_coverage
PortfolioOptimisers.assert_cross_sectional_factor_moments
PortfolioOptimisers.cross_sectional_idiosyncratic_covariance
PortfolioOptimisers.cross_sectional_variance_counts
PortfolioOptimisers.cross_sectional_finite_mean
PortfolioOptimisers.cross_sectional_standardised_residuals
PortfolioOptimisers.cross_sectional_scenarios
PortfolioOptimisers.cross_sectional_investable
PortfolioOptimisers.cross_sectional_panel_masks
PortfolioOptimisers.cross_sectional_cap_finite!
PortfolioOptimisers.cross_sectional_needs_market_cap
PortfolioOptimisers.cross_sectional_rows
PortfolioOptimisers.cross_sectional_reduced_loadings
PortfolioOptimisers.cross_sectional_neutralise!
PortfolioOptimisers.cross_sectional_family_basis
PortfolioOptimisers.cross_sectional_basis_now
PortfolioOptimisers.cross_sectional_expand
PortfolioOptimisers.cross_sectional_residual_block
PortfolioOptimisers.cross_sectional_lift
PortfolioOptimisers.cross_sectional_alpha_split
PortfolioOptimisers.cross_sectional_return_forecast
PortfolioOptimisers.cross_sectional_forecast_mu
```
