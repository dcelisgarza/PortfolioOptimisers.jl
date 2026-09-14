# A Result resumes an online run

An online walk-forward's [`MultiPeriodPredictionResult`](@ref) carries the estimator the fold loop threaded, folded through the last training end, and [`Resume`](@ref) hands that Result back to the loop over the full history extended: the folds up to the one whose training window ends where the state stopped are skipped, the ordinary delta is folded into a copy of the state, and the loop continues from the fold after them, reaching the weights of the one-shot run over the longer history fold for fold. The Result returned holds the new folds only, and `vcat` stacks a run and its resumes for scoring. ADR 0144 records the decision.

```@docs
Resume
PortfolioOptimisers.OptimiserResume
cross_val_predict(r::PortfolioOptimisers.OptimiserResume, rd::ReturnsResult, cv::CVER)
PortfolioOptimisers.resume_fold_count
PortfolioOptimisers.assert_resume_scheme
PortfolioOptimisers.assert_resume_folds
PortfolioOptimisers.assert_resume_full_fold
PortfolioOptimisers.cv_resume_info
PortfolioOptimisers.copy_states
PortfolioOptimisers.copy_state
PortfolioOptimisers.carrier_timestamps
PortfolioOptimisers.context_timestamps
is_time_dependent(r::Resume)
search_cross_validation(::Resume, ::PortfolioOptimisers.AbstractSearchCrossValidationEstimator, ::Any)
Base.vcat(a::MultiPeriodPredictionResult, b::MultiPeriodPredictionResult)
```
