"""
    plot_ptf_cumulative_returns(w, X[, fees]; kwargs...)

Plot the cumulative returns of a portfolio with weights `w` over the data matrix `X`.

Implemented by the `PortfolioOptimisersPlotsExt` extension (requires `StatsPlots` and `GraphRecipes`).
"""
function plot_ptf_cumulative_returns end
"""
    plot_asset_cumulative_returns(w, X[, fees]; kwargs...)

Plot the cumulative returns of individual assets for a portfolio with weights `w`.

Implemented by the `PortfolioOptimisersPlotsExt` extension (requires `StatsPlots` and `GraphRecipes`).
"""
function plot_asset_cumulative_returns end
"""
    plot_composition(w[, nx]; kwargs...)

Plot the portfolio composition as a bar chart of asset weights `w`.

Implemented by the `PortfolioOptimisersPlotsExt` extension (requires `StatsPlots` and `GraphRecipes`).
"""
function plot_composition end
"""
    plot_stacked_bar_composition(w[, nx]; kwargs...)

Plot the portfolio composition as a stacked bar chart.

Implemented by the `PortfolioOptimisersPlotsExt` extension (requires `StatsPlots` and `GraphRecipes`).
"""
function plot_stacked_bar_composition end
"""
    plot_stacked_area_composition(w[, nx]; kwargs...)

Plot the portfolio composition as a stacked area chart.

Implemented by the `PortfolioOptimisersPlotsExt` extension (requires `StatsPlots` and `GraphRecipes`).
"""
function plot_stacked_area_composition end
"""
    plot_dendrogram(clr[, nx]; kwargs...)

Plot a dendrogram from a hierarchical clustering result.

Implemented by the `PortfolioOptimisersPlotsExt` extension (requires `StatsPlots` and `GraphRecipes`).
"""
function plot_dendrogram end
"""
    plot_clusters(clr, X[, nx]; kwargs...)

Plot asset clusters from a clustering result on the data matrix `X`.

Implemented by the `PortfolioOptimisersPlotsExt` extension (requires `StatsPlots` and `GraphRecipes`).
"""
function plot_clusters end
"""
    plot_drawdowns(w, X[, fees]; kwargs...)

Plot portfolio drawdowns over time.

Implemented by the `PortfolioOptimisersPlotsExt` extension (requires `StatsPlots` and `GraphRecipes`).
"""
function plot_drawdowns end
"""
    plot_risk_contribution(w, r, X[, fees]; kwargs...)

Plot the risk contribution of each asset in the portfolio.

Implemented by the `PortfolioOptimisersPlotsExt` extension (requires `StatsPlots` and `GraphRecipes`).
"""
function plot_risk_contribution end
"""
    plot_factor_risk_contribution(w, r, X[, fees]; kwargs...)

Plot the factor risk contribution for a factor risk model.

Implemented by the `PortfolioOptimisersPlotsExt` extension (requires `StatsPlots` and `GraphRecipes`).
"""
function plot_factor_risk_contribution end
"""
    plot_measures(w, rs, X[, fees]; kwargs...)

Plot multiple risk measures for the portfolio.

Implemented by the `PortfolioOptimisersPlotsExt` extension (requires `StatsPlots` and `GraphRecipes`).
"""
function plot_measures end
"""
    plot_histogram(w, X[, fees]; kwargs...)

Plot a histogram of portfolio returns.

Implemented by the `PortfolioOptimisersPlotsExt` extension (requires `StatsPlots` and `GraphRecipes`).
"""
function plot_histogram end

export plot_ptf_cumulative_returns, plot_asset_cumulative_returns, plot_composition,
       plot_stacked_bar_composition, plot_stacked_area_composition, plot_dendrogram,
       plot_clusters, plot_drawdowns, plot_risk_contribution, plot_factor_risk_contribution,
       plot_measures, plot_histogram
