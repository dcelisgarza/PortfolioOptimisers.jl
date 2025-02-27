struct PortfolioOptimisersCovariance{T1 <: StatsBase.CovarianceEstimator,
                                     T2 <: FixNonPositiveDefiniteMatrix} <:
       PortfolioOptimisersCovarianceEstimator
    ce::T1
    fix_non_pos_def::T2
end

export PortfolioOptimisersCovariance, robust_cor
