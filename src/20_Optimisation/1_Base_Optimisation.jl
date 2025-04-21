abstract type OptimisationEstimator <: AbstractEstimator end
abstract type PortfolioResult <: AbstractResult end

function optimise! end

export optimise!
