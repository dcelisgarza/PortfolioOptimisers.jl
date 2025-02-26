abstract type PortfolioOptimisersCovarianceEstimator <: StatsBase.CovarianceEstimator end
abstract type PearsonCovarianceEstimator <: PortfolioOptimisersCovarianceEstimator end
abstract type RankCovarianceEstimator <: PortfolioOptimisersCovarianceEstimator end
abstract type POVarianceEstimator <: PortfolioOptimisersCovarianceEstimator end
abstract type GerberCovariance <: PortfolioOptimisersCovarianceEstimator end
abstract type BaseGerberCovariance <: GerberCovariance end
abstract type SmythBrobyCovariance <: GerberCovariance end
abstract type SmythBrobyGerberCovariance <: GerberCovariance end
