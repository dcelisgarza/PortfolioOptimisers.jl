abstract type POCovarianceEstimator <: StatsBase.CovarianceEstimator end
abstract type PearsonCovarianceEstimator <: POCovarianceEstimator end
abstract type RankCovarianceEstimator <: POCovarianceEstimator end
abstract type POVarianceEstimator <: POCovarianceEstimator end
