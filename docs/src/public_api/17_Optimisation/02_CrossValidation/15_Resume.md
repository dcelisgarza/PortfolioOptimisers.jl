```@meta
Description = "A Result resumes an online run, public API of PortfolioOptimisers.jl: Resume, cross_val_predict, search_cross_validation."
```

# A Result resumes an online run

The [`MultiPeriodPredictionResult`](@ref) of an online walk-forward stores the estimator as it was after the last training window. To continue the run, pass [`Resume`](@ref)`(res)` to `cross_val_predict` in place of the estimator, with the same walk-forward and the full history with the new rows at its end. The walk-forward skips every fold up to the one whose training window ends at the last row of the stored state. It adds the rows after that point to a copy of the state, and continues from the next fold. Each fold then gets the same weights that one run over the longer history gives. The result holds the new folds only, and `vcat` stacks a run and its resumes, so you can score them together.

```@docs
Resume
cross_val_predict(r::PortfolioOptimisers.OptimiserResume, rd::ReturnsResult, cv::CVER)
search_cross_validation(::Resume, ::PortfolioOptimisers.AbstractSearchCrossValidationEstimator, ::Any)
```
