```@meta
Description = "A Result resumes an online run, public API of PortfolioOptimisers.jl: Resume, cross_val_predict, search_cross_validation."
```

# A Result resumes an online run

An online walk-forward's [`MultiPeriodPredictionResult`](@ref) carries the estimator the fold loop threaded, folded through the last training end, and [`Resume`](@ref) hands that Result back to the loop over the full history extended: the folds up to the one whose training window ends where the state stopped are skipped, the ordinary delta is folded into a copy of the state, and the loop continues from the fold after them, reaching the weights of the one-shot run over the longer history fold for fold. The Result returned holds the new folds only, and `vcat` stacks a run and its resumes for scoring. ADR 0144 records the decision.

```@docs
Resume
cross_val_predict(r::PortfolioOptimisers.OptimiserResume, rd::ReturnsResult, cv::CVER)
search_cross_validation(::Resume, ::PortfolioOptimisers.AbstractSearchCrossValidationEstimator, ::Any)
```
