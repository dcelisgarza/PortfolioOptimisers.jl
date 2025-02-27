abstract type PortfolioOptimisersDistanceMetric <: Distances.Metric end
abstract type PortfolioOptimisersAbsoluteDistanceMetric <: PortfolioOptimisersDistanceMetric end

abstract type PortfolioOptimisersDistanceDistanceMetric <: Distances.Metric end
abstract type PortfolioOptimisersAbsoluteDistanceDistanceMetric <:
              PortfolioOptimisersDistanceDistanceMetric end

function distance end

export distance
