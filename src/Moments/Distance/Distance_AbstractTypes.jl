abstract type PortfolioOptimisersDistanceMetric <: Distances.Metric end
abstract type PortfolioOptimisersAbsoluteDistanceMetric <: PortfolioOptimisersDistanceMetric end

function distance end

export distance
