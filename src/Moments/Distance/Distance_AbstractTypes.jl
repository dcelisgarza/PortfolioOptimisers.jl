abstract type PortfolioOptimisersUnionDistanceMetric <: Distances.Metric end
abstract type PortfolioOptimisersDistanceMetric <: PortfolioOptimisersUnionDistanceMetric end
abstract type PortfolioOptimisersAbsoluteDistanceMetric <: PortfolioOptimisersDistanceMetric end
abstract type PortfolioOptimisersDistanceDistanceMetric <:
              PortfolioOptimisersUnionDistanceMetric end
abstract type PortfolioOptimisersAbsoluteDistanceDistanceMetric <:
              PortfolioOptimisersDistanceDistanceMetric end

function distance end

export distance
