module PortfolioOptimisersGraphRecipesExt

# `plot_network` is the one plot function that draws with GraphRecipes, so it has an extension
# of its own. GraphRecipes brings GeometryTypes, whose methods invalidate much of the compiled
# code of a caller. A session that draws no network then loads neither package (#2186).
using PortfolioOptimisers, GraphRecipes

import PortfolioOptimisers: MatNum, VecNum, Option, Pr_RR, NwE_ClE_Cl, OptimisationResult,
                            investable_plot_view, result_investable_view

function PortfolioOptimisers.plot_network(pl::NwE_ClE_Cl, X::MatNum,
                                          nx::AbstractVector = 1:size(X, 2),
                                          w::Option{<:VecNum} = nothing;
                                          threshold::Number = 0, kwargs...)
    plr = phylogeny_matrix(pl, X)
    A = copy(plr.X)
    A[abs.(A) .<= threshold] .= 0
    node_size = if isnothing(w)
        fill(1, size(X, 2))
    else
        abs.(w) ./ maximum(abs, w)
    end
    return graphplot(A; names = nx, node_weights = node_size, title = "Asset Network",
                     kwargs...)
end
function PortfolioOptimisers.plot_network(pl::NwE_ClE_Cl, pr::Pr_RR,
                                          w::Option{<:VecNum} = nothing;
                                          nx::AbstractVector = 1:size(pr.X, 2), kwargs...)
    if isa(pr, ReturnsResult) && !isnothing(pr.nx)
        nx = pr.nx
    end
    pr_i, nx_i, w_i = investable_plot_view(pr, nx, w)
    return PortfolioOptimisers.plot_network(pl, pr_i.X, nx_i, w_i; kwargs...)
end
function PortfolioOptimisers.plot_network(pl::NwE_ClE_Cl, res::OptimisationResult;
                                          rd::Option{<:Pr_RR} = nothing,
                                          nx::AbstractVector = 1:length(res.w), kwargs...)
    # The method of the prior arity receives a viewed prior, whose mask is already
    # `nothing`, so the pairing of the result's two universes happens here.
    _, w, pr, _, nx = result_investable_view(res, rd, nothing, nx)
    return PortfolioOptimisers.plot_network(pl, pr, w; nx = nx, kwargs...)
end

end
