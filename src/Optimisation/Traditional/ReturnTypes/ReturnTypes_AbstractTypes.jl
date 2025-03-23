abstract type PortfolioReturnType end
function cluster_return_factory(r::PortfolioReturnType; kwargs...)
    return r
end
