## ──────────────────────────────────────────────────────────────────────────────
## Cumulative returns
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_portfolio_cumulative_returns(
        w::VecNum_VecVecNum,
        X::MatNum,
        fees::Option{<:Fees} = nothing;
        ts::AbstractVector = 1:size(X, 1),
        compound::Bool = false,
        kwargs...
    ) -> Plot
    plot_portfolio_cumulative_returns(
        w::VecNum_VecVecNum,
        pr::Pr_RR,
        fees::Option{<:Fees} = nothing;
        ts::AbstractVector = 1:size(pr.X, 1),
        compound::Bool = false,
        kwargs...
    ) -> Plot
    plot_portfolio_cumulative_returns(
        res::OptimisationResult,
        pr::Pr_RR;
        fees::Option{<:Fees} = nothing,
        compound::Bool = false,
        kwargs...
    ) -> Plot
    plot_portfolio_cumulative_returns(
        res::OptimisationResult;
        fees::Option{<:Fees} = nothing,
        compound::Bool = false,
        kwargs...
    ) -> Plot
    plot_portfolio_cumulative_returns(
        pred::Union{<:PredictionResult, <:MultiPeriodPredictionResult};
        compound::Bool = false,
        kwargs...
    ) -> Plot
    plot_portfolio_cumulative_returns(
        ppred::PopulationPredictionResult;
        compound::Bool = false,
        kwargs...
    ) -> Plot
    plot_portfolio_cumulative_returns(
        net_ret::VecNum_VecVecNum;
        ts::AbstractVector,
        compound::Bool = false,
        kwargs...
    ) -> Plot

Plot the cumulative returns of a portfolio against time.

The curve is [`cumulative_returns`](@ref) of the net portfolio returns that [`calc_net_returns`](@ref) forms from `w`, `X` and `fees`. A vector of weight vectors draws one curve per portfolio, labelled `Portfolio i`. The method that takes `net_ret` draws a return series the caller already formed. A population draws the curve of each member on one figure. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

A gapped panel reaches this figure only through the returns that the caller gives it. A prediction arity is finite already, because the fold set the Held Gap to zero once in [`predict`](@ref). A `w, X` or `w, pr` arity computes on the matrix that the caller holds and checks nothing, and `X * w` is `NaN` at a gap even where the weight is zero. Score a gapped panel through `predict(res, rd)` and plot the prediction.

# Arguments

  - `w`: Portfolio weights vector, or a vector of weight vectors.
  - `X`: Asset returns matrix (observations × assets).
  - `fees`: Transaction fees, or `nothing`.
  - `ts`: Labels of the time axis. A [`ReturnsResult`](@ref) that carries `ts` replaces the keyword with its own.
  - `compound`: Compound the returns when `true`, and sum them otherwise.
  - `pr`: Prior result or [`ReturnsResult`](@ref). The figure reads its `X`.
  - `res`: Optimisation result. The figure reads its weights, and its fees when `fees` is `nothing`, on its investable universe. Without `pr`, it reads the returns of the prior it was fitted on.
  - `pred`: A prediction, whose return series the figure draws.
  - `ppred`: A population of predictions.
  - `net_ret`: Net portfolio returns, or a vector of such series.

# Returns

  - `plt::Plots.Plot`: A line plot with one curve per portfolio.

# Related

  - [`calc_net_returns`](@ref)
  - [`cumulative_returns`](@ref)
"""
function plot_portfolio_cumulative_returns end

## ──────────────────────────────────────────────────────────────────────────────
## Asset cumulative returns
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_asset_cumulative_returns(
        w::VecNum,
        X::MatNum,
        fees::Option{<:Fees} = nothing;
        ts::AbstractVector = 1:size(X, 1),
        nx::AbstractVector = 1:size(X, 2),
        compound::Bool = false,
        N::Option{<:Number} = nothing,
        kwargs...
    ) -> Plot
    plot_asset_cumulative_returns(
        w::VecNum,
        pr::Pr_RR,
        fees::Option{<:Fees} = nothing;
        ts::AbstractVector = 1:size(pr.X, 1),
        nx::AbstractVector = 1:size(pr.X, 2),
        compound::Bool = false,
        N::Option{<:Number} = nothing,
        kwargs...
    ) -> Plot
    plot_asset_cumulative_returns(
        res::OptimisationResult[, pr::Pr_RR];
        fees::Option{<:Fees} = nothing,
        compound::Bool = false,
        N::Option{<:Number} = nothing,
        kwargs...
    ) -> Plot
    plot_asset_cumulative_returns(pred::PredictionResult; compound, N, kwargs...) -> Plot
    plot_asset_cumulative_returns(ppred::PopulationPredictionResult; compound, N, kwargs...) -> Plot

Plot the cumulative contribution of each asset to the portfolio return, one line per asset.

The line of asset ``i`` is [`cumulative_returns`](@ref) of its weighted return ``w_i x_{t,i}`` net of its own fee, which [`calc_net_asset_returns`](@ref) forms. It is not the cumulative return of the asset alone. [`relevant_assets`](@ref) ranks the assets by ``\\lvert w_i \\rvert`` and chooses how many lines to draw. The rest of the universe is one more line, `Others`, the cumulative net return of the sub-portfolio of the assets that the ranking left out. A population draws the figure of each member on one plot, and each member must be a single fold. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

A per-asset line **keeps its gap**, which the backend draws as a break, and that break is the delisting. Read against a prediction the same line is flat, because the fold set the Held Gap to zero before it formed the series. The `Others` line cannot keep a gap. One `NaN` makes every date of the sum `NaN`, at any weight, so it leaves out each column that is not finite throughout, through [`finite_columns`](@ref). The line therefore describes the rest of the universe without the assets that it cannot value.

A result arity draws the investable universe of the result, and it drops the charge of a liquidation, because a liquidated asset has no column on that universe for its exit charge. [`plot_portfolio_cumulative_returns`](@ref) charges it.

# Arguments

  - `w`: Portfolio weights vector.
  - `X`: Asset returns matrix (observations × assets).
  - `fees`: Transaction fees, or `nothing`.
  - `ts`: Labels of the time axis. A [`ReturnsResult`](@ref) that carries `ts` replaces the keyword with its own.
  - `nx`: Asset names. A [`ReturnsResult`](@ref) that carries `nx` replaces the keyword with its own.
  - `compound`: Compound the returns when `true`, and sum them otherwise.
  - `N`: The count or the share of the lines to draw, or `nothing` for the effective number of assets. [`relevant_assets`](@ref) states the rule.
  - `pr`: Prior result or [`ReturnsResult`](@ref). The figure reads its `X`.
  - `res`: Optimisation result. The figure reads its weights, and its fees when `fees` is `nothing`, on its investable universe. Without `pr`, it reads the returns of the prior that the result was fitted on.
  - `pred`: A prediction. The figure draws `pred.res` without `pr`, so it reads the returns the fold was fitted on and not the returns of its test window.
  - `ppred`: A population of single-fold predictions.

# Returns

  - `plt::Plots.Plot`: A line plot with one line per drawn asset, and an `Others` line when the count truncates.

# Related

  - [`number_effective_assets`](@ref)
  - [`calc_net_returns`](@ref)
  - [`cumulative_returns`](@ref)
"""
function plot_asset_cumulative_returns end

## ──────────────────────────────────────────────────────────────────────────────
## Composition
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_composition(
        w::VecNum,
        nx::AbstractVector = 1:length(w);
        N::Option{<:Number} = nothing,
        kwargs...
    ) -> Plot
    plot_composition(res::OptimisationResult, rd::ReturnsResult; N, kwargs...) -> Plot
    plot_composition(res::OptimisationResult, pr::AbstractPriorResult; N, kwargs...) -> Plot
    plot_composition(pred::PredictionResult; N, kwargs...) -> Plot
    plot_composition(mpred::MultiPeriodPredictionResult; N, kwargs...) -> Plot
    plot_composition(ppred::PopulationPredictionResult; N, kwargs...) -> Plot

Plot the composition of a portfolio as a bar chart of its asset weights.

[`relevant_assets`](@ref) ranks the assets by ``\\lvert w_i \\rvert`` and chooses how many bars to draw. The drawn bars keep the order of the asset axis, and the assets that the ranking left out become one more bar, `Others`, whose height is the signed sum of their weights. A walk-forward path and a population draw a stacked bar chart through [`plot_stacked_bar_composition`](@ref), one bar per fold or per member, and they accept `N` and do not read it. A member that is a walk-forward path takes the mean of its fold weights. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

# Arguments

  - `w`: Portfolio weights vector.
  - `nx`: Asset names.
  - `N`: The count or the share of the bars to draw, or `nothing` for the effective number of assets. [`relevant_assets`](@ref) states the rule.
  - `res`: Optimisation result. The figure reads its weights on the universe of the caller.
  - `rd`: Returns result. The figure reads its `nx`, and numbers the assets when it has none.
  - `pr`: Prior result. The figure numbers the assets and reads nothing else of it.
  - `pred`: A prediction. The figure reads the weights of its result on its investable universe, under the names of the fold.
  - `mpred`: A walk-forward path.
  - `ppred`: A population of predictions.

# Returns

  - `plt::Plots.Plot`: A bar chart, with an `Others` bar when the count truncates, or a stacked bar chart for a path or a population.

# Related

  - [`number_effective_assets`](@ref)
  - [`plot_stacked_bar_composition`](@ref)
"""
function plot_composition end

## ──────────────────────────────────────────────────────────────────────────────
## Stacked compositions
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_stacked_bar_composition(
        w::VecNum_VecVecNum,
        nx::AbstractVector = 1:(isa(w, VecVecNum) ? length(first(w)) : length(w));
        kwargs...
    ) -> Plot
    plot_stacked_bar_composition(
        res_vec::AbstractVector{<:OptimisationResult},
        rd::ReturnsResult;
        kwargs...
    ) -> Plot

Plot the compositions of several portfolios as a stacked bar chart, one bar per portfolio.

Each bar stacks the asset weights of one portfolio, one segment per asset. A single weight vector draws one bar. The method that takes `res_vec` reads the weights of each result, and the names of `rd`, or numbers the assets when `rd` has none. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

# Arguments

  - `w`: Portfolio weights vector, or a vector of weight vectors of equal length.
  - `nx`: Asset names, one per weight of a portfolio.
  - `res_vec`: Optimisation results.
  - `rd`: Returns result. The figure reads its `nx`.

# Returns

  - `plt::Plots.Plot`: A stacked bar chart.

# Related

  - [`plot_stacked_area_composition`](@ref)
  - [`plot_composition`](@ref)
"""
function plot_stacked_bar_composition end

"""
    plot_stacked_area_composition(
        w::VecNum_VecVecNum,
        nx::AbstractVector = 1:(isa(w, VecVecNum) ? length(first(w)) : length(w));
        kwargs...
    ) -> Plot
    plot_stacked_area_composition(
        res_vec::AbstractVector{<:OptimisationResult},
        rd::ReturnsResult;
        kwargs...
    ) -> Plot
    plot_stacked_area_composition(mpred::MultiPeriodPredictionResult; N, kwargs...) -> Plot
    plot_stacked_area_composition(ppred::PopulationPredictionResult; N, kwargs...) -> Plot

Plot the compositions of several portfolios as a stacked area chart, one band per asset.

The horizontal axis is the position of the portfolio, and the band of an asset is its weight in each portfolio. A walk-forward path draws one position per fold, and a population one per member, whose weights are the mean of its fold weights when it is a walk-forward path. Those two methods accept `N` and do not read it. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

# Arguments

  - `w`: Portfolio weights vector, or a vector of weight vectors of equal length.
  - `nx`: Asset names, one per weight of a portfolio.
  - `res_vec`: Optimisation results.
  - `rd`: Returns result. The figure reads its `nx`.
  - `mpred`: A walk-forward path.
  - `ppred`: A population of predictions.

# Returns

  - `plt::Plots.Plot`: A stacked area chart.

# Related

  - [`plot_stacked_bar_composition`](@ref)
  - [`plot_composition`](@ref)
"""
function plot_stacked_area_composition end

## ──────────────────────────────────────────────────────────────────────────────
## Risk contribution
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_risk_contribution(
        r::BaseRM_VecBaseRM,
        w::VecNum,
        X::MatNum_Pr,
        fees::Option{<:Fees} = nothing;
        nx::AbstractVector = 1:length(w),
        delta::Number = 1e-6,
        marginal::Bool = false,
        percentage::Bool = true,
        erc::Bool = true,
        N::Option{<:Number} = nothing,
        sca::Scalariser = SumScalariser(),
        kwargs...
    ) -> Plot
    plot_risk_contribution(r, w::VecNum, rd::ReturnsResult, fees = nothing; delta, marginal, percentage, erc, N, sca, kwargs...) -> Plot
    plot_risk_contribution(r, res::OptimisationResult, rd::ReturnsResult; delta, marginal, percentage, erc, N, sca, kwargs...) -> Plot
    plot_risk_contribution(r, res::OptimisationResult, pr::AbstractPriorResult; nx, delta, marginal, percentage, erc, N, sca, kwargs...) -> Plot
    plot_risk_contribution(r, pred::PredictionResult, fees = nothing; delta, marginal, percentage, erc, N, sca, kwargs...) -> Plot

Plot the risk contribution of each asset as a bar chart.

The bars are [`risk_contribution`](@ref) of `w`, rescaled to shares when `percentage` is `true`, and a horizontal line marks their mean when `erc` is `true`. [`plot_composition`](@ref) draws the bars, so an `N` that truncates keeps the order of the asset axis and adds an `Others` bar. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

The bars decompose **one** number, the scalarised aggregate of `r`, so they sum to the aggregate and not to any one element. Under [`MaxScalariser`](@ref) and [`MinScalariser`](@ref) the decomposition is exact only where the argmax is **unique**. At a near tie between two scaled measures the subgradient is a set, and the bars are one admissible answer of several.

The method that takes a fold reads the target weights of the fold and the asset returns that its Held Weights record kept, and it settles the fee against the length of the fold, as [`predict`](@ref) did. Under a Weight Drift the bars are exact to **first order in the drift** only, for the reason [`risk_contribution`](@ref) states. A fold whose scheme set neither `wd` nor `pws` keeps no record and so no asset returns.

# Mathematical definition

```math
\\begin{align}
\\tilde{c}_i &= \\frac{c_i}{\\sum_{j=1}^{N} c_j}\\,, \\\\
\\bar{c} &= \\frac{1}{N} \\sum_{i=1}^{N} \\tilde{c}_i\\,.
\\end{align}
```

Where:

  - ``c_i``: Risk contribution of asset ``i``, as [`risk_contribution`](@ref) returns it.
  - ``\\tilde{c}_i``: Height of the bar of asset ``i``. It is ``c_i`` itself when `percentage` is `false`.
  - ``\\bar{c}``: Height of the reference line, the mean over every asset before any truncation. Under `percentage` it is ``1/N``, and when `marginal` is `false` it is the height of each bar under an equal risk contribution.
  - $(math_dict[:N])

# Arguments

  - `r`: Risk measure, or a vector of them that `sca` combines.
  - `w`: Portfolio weights vector.
  - `X`: Asset returns matrix (observations × assets), or a prior result.
  - `rd`: Returns result. The figure reads its `X` and its `nx`.
  - `pr`: Prior result. The figure reads its `X`.
  - `fees`: Transaction fees, or `nothing`.
  - `nx`: Asset names.
  - `delta`: Finite difference step of [`risk_contribution`](@ref).
  - `marginal`: Draw the marginal contribution when `true`, and the component contribution otherwise.
  - `percentage`: Divide the contributions by their sum when `true`, so that the bars sum to one.
  - `erc`: Draw the reference line when `true`.
  - `N`: The count or the share of the bars to draw, or `nothing` for the effective number of the contributions. [`relevant_assets`](@ref) states the rule.
  - `sca`: Scalariser that combines the measures in `r`. It has no effect when `r` is one measure. Pass `res.sca` to draw the figure that the optimisation ran under.
  - `res`: Optimisation result. The figure reads its weights and its fees on its investable universe.
  - `pred`: A fold, from [`predict`](@ref).

# Validation

  - `delta > 0`, else a `DomainError` is raised.
  - If `N` is not `nothing`, `N > 0`, else a `DomainError` is raised.
  - A fold carries a Held Weights record, else an `ArgumentError` is raised.
  - The rules of [`risk_contribution`](@ref).

# Returns

  - `plt::Plots.Plot`: A bar chart, with the reference line when `erc` is `true`.

# Related

  - [`risk_contribution`](@ref)
  - [`plot_composition`](@ref)
  - [`PredictionResult`](@ref)
  - [`HeldWeightsResult`](@ref)
"""
function plot_risk_contribution end

## ──────────────────────────────────────────────────────────────────────────────
## Factor risk contribution
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_factor_risk_contribution(
        r::BaseRM_VecBaseRM,
        w::VecNum,
        X::MatNum_Pr,
        fees::Option{<:Fees} = nothing;
        re::RegE_Reg = StepwiseRegression(),
        rd::ReturnsResult = ReturnsResult(),
        nf::Option{<:AbstractVector} = nothing,
        delta::Number = 1e-6,
        N::Option{<:Number} = nothing,
        percentage::Bool = true,
        erc::Bool = true,
        sca::Scalariser = SumScalariser(),
        kwargs...
    ) -> Plot
    plot_factor_risk_contribution(r, res::OptimisationResult, rd::ReturnsResult; re, delta, N, percentage, erc, sca, kwargs...) -> Plot
    plot_factor_risk_contribution(
        r,
        pred::PredictionResult,
        fees = nothing;
        rd::Option{<:ReturnsResult} = nothing,
        re, delta, N, percentage, erc, sca, kwargs...
    ) -> Plot

Plot the risk contribution of each factor as a bar chart.

The bars are [`factor_risk_contribution`](@ref) of `w`, rescaled to shares when `percentage` is `true`, and a horizontal line marks their mean when `erc` is `true`. The last bar, `Off-factor`, is the contribution of the part of the weights that has no exposure to any factor. It is not a regression intercept. [`plot_composition`](@ref) draws the bars, so an `N` that truncates keeps the order of the factor axis and adds an `Others` bar. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

The bars take the names in `nf`, or else in `rd.nf`, only when the names count the columns of the loadings. A dimension reduction regression fits fewer columns than there are factors, so its bars are numbered.

The method that takes a fold is the twin of the fold method of [`plot_risk_contribution`](@ref). It fits the loadings on a returns result that it builds from the fold, the asset returns of its Held Weights record beside the factor block the fold carried, unless the caller gives `rd`, which it views at the mask of the fold. The first order caveat of that method holds here too.

# Mathematical definition

```math
\\begin{align}
\\tilde{c}_k &= \\frac{c_k}{\\sum_{j=1}^{K+1} c_j}\\,, \\\\
\\bar{c} &= \\frac{1}{K+1} \\sum_{k=1}^{K+1} \\tilde{c}_k\\,.
\\end{align}
```

Where:

  - ``c_k``: Risk contribution of factor ``k``, and of the off-factor part when ``k = K + 1``, as [`factor_risk_contribution`](@ref) returns it.
  - ``\\tilde{c}_k``: Height of bar ``k``. It is ``c_k`` itself when `percentage` is `false`.
  - ``\\bar{c}``: Height of the reference line, the mean over every bar before any truncation. Under `percentage` it is ``1/(K+1)``, the height of each bar under an equal risk contribution.
  - $(math_dict[:K])

# Arguments

  - `r`: Risk measure, or a vector of them that `sca` combines.
  - `w`: Portfolio weights vector.
  - `X`: Asset returns matrix (observations × assets), or a prior result.
  - `fees`: Transaction fees, or `nothing`.
  - `re`: Factor regression estimator.
  - `rd`: Returns result that carries the factor returns the loadings are fitted on, and the factor names in `rd.nf`.
  - `nf`: Factor names. They replace `rd.nf` when given.
  - `delta`: Finite difference step of [`factor_risk_contribution`](@ref).
  - `N`: The count or the share of the bars to draw, or `nothing` for the effective number of the contributions. [`relevant_assets`](@ref) states the rule.
  - `percentage`: Divide the contributions by their sum when `true`, so that the bars sum to one.
  - `erc`: Draw the reference line when `true`.
  - `sca`: Scalariser that combines the measures in `r`. It has no effect when `r` is one measure. Pass `res.sca` to draw the figure that the optimisation ran under.
  - `res`: Optimisation result. The figure reads its weights and its fees on its investable universe.
  - `pred`: A fold, from [`predict`](@ref).

# Validation

  - `delta > 0`, else a `DomainError` is raised.
  - If `N` is not `nothing`, `N > 0`, else a `DomainError` is raised.
  - A fold carries a Held Weights record, else an `ArgumentError` is raised.
  - The rules of [`factor_risk_contribution`](@ref).

# Returns

  - `plt::Plots.Plot`: A bar chart, with the reference line when `erc` is `true`.

# Related

  - [`factor_risk_contribution`](@ref)
  - [`plot_composition`](@ref)
  - [`PredictionResult`](@ref)
  - [`HeldWeightsResult`](@ref)
"""
function plot_factor_risk_contribution end

## ──────────────────────────────────────────────────────────────────────────────
## Dendrogram
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_dendrogram(
        clr::AbstractClusteringResult,
        nx::AbstractVector = 1:length(clr.res.order);
        dend_theme::Symbol = :Spectral,
        kwargs...
    ) -> Plot
    plot_dendrogram(cle::HClE_HCl, X::MatNum, nx = 1:size(X, 2); dims::Integer = 1, kwargs...) -> Plot
    plot_dendrogram(cle::HClE_HCl, pr::Pr_RR, nx = 1:size(pr.X, 2); kwargs...) -> Plot

Plot the dendrogram of a hierarchical clustering, with one shaded region per cluster.

The leaves follow the order of the clustering. Each cluster takes one colour of `dend_theme`, and its region reaches ``\\min(1.1 h, 1)``, where ``h`` is the largest merge height inside the cluster, so the shading never rises above a height of one. The estimator arities call [`clusterise`](@ref) first. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

A non-investable asset is **not drawn**. A clustering is fitted by a plain moment estimator, which refuses a gapped sample, so the prior arity reduces itself and its names to the Investable Mask at its entry through [`investable_plot_view`](@ref). The bare `X` arity computes on the matrix that the caller holds.

# Arguments

  - `clr`: Clustering result.
  - `cle`: Clustering estimator. It clusters `X`, or `pr.X`.
  - `X`: Asset returns matrix (observations × assets).
  - `pr`: Prior result or [`ReturnsResult`](@ref). A returns result that carries `nx` replaces the argument with its own.
  - `nx`: Asset names.
  - `dend_theme`: Colour palette of the cluster regions.
  - `dims`: Dimension of `X` that holds the observations, as [`clusterise`](@ref) reads it. Only the `X` arity takes it. The `X` of a prior or a returns result holds the observations along the rows, so the prior arity passes `dims = 1`.

# Validation

  - The rules of [`clusterise`](@ref).

# Returns

  - `plt::Plots.Plot`: A dendrogram of 600 × 600 pixels.

# Related

  - [`clusterise`](@ref)
"""
function plot_dendrogram end

## ──────────────────────────────────────────────────────────────────────────────
## Cluster heatmap
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_clusters(
        clr::AbstractClusteringResult,
        nx::AbstractVector = 1:size(clr.S, 1);
        dend_theme::Symbol = :Spectral,
        hmap_theme::Symbol = :Spectral,
        color_func = x -> any(x .< 0) ? (-1, 1) : (0, 1),
        line_color = :black,
        line_width = 3,
        kwargs...
    ) -> Plot
    plot_clusters(cle::HClE_HCl, X::MatNum, nx = 1:size(X, 2); dims::Integer = 1, kwargs...) -> Plot
    plot_clusters(cle::HClE_HCl, pr::Pr_RR, nx = 1:size(pr.X, 2); kwargs...) -> Plot

Plot the similarity matrix of a clustering as a heatmap in the order of the clustering, with a dendrogram on two sides and a box round each cluster.

The heatmap always draws a correlation. When a diagonal entry of `clr.S` is not one, the figure rescales a copy of it to a correlation first, as [`plot_correlation`](@ref) defines it. `color_func` maps that correlation to the colour limits. The dendrograms shade each cluster up to its largest merge height, with no cap. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

A non-investable asset is **not drawn**, for the reason [`plot_dendrogram`](@ref) states, and the prior arity reduces to the Investable Mask at its entry through [`investable_plot_view`](@ref).

# Arguments

  - `clr`: Clustering result.
  - `cle`: Clustering estimator. It clusters `X`, or `pr.X`.
  - `X`: Asset returns matrix (observations × assets).
  - `pr`: Prior result or [`ReturnsResult`](@ref). A returns result that carries `nx` replaces the argument with its own.
  - `nx`: Asset names.
  - `dend_theme`: Colour palette of the cluster regions of the dendrograms.
  - `hmap_theme`: Colour gradient of the heatmap.
  - `color_func`: Function that maps the correlation matrix to the colour limits `(lo, hi)`. The default gives `(-1, 1)` when an entry is negative, and `(0, 1)` otherwise.
  - `line_color`: Colour of the cluster boxes.
  - `line_width`: Width of the cluster boxes.
  - `dims`: Dimension of `X` that holds the observations, as [`clusterise`](@ref) reads it. Only the `X` arity takes it. The `X` of a prior or a returns result holds the observations along the rows, so the prior arity passes `dims = 1`.

# Validation

  - The rules of [`clusterise`](@ref).

# Returns

  - `plt::Plots.Plot`: A layout of 600 × 600 pixels, with the heatmap below one dendrogram and beside the other.

# Related

  - [`clusterise`](@ref)
  - [`prior`](@ref)
"""
function plot_clusters end

## ──────────────────────────────────────────────────────────────────────────────
## Drawdowns
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_drawdowns(
        w::ArrNum,
        X::MatNum,
        fees::Option{<:Fees} = nothing;
        slv::Option{<:Slv_VecSlv} = nothing,
        ts::AbstractVector = 1:size(X, 1),
        compound::Bool = false,
        alpha::Number = 0.05,
        kappa::Number = 0.3,
        rw = nothing,
        kwargs...
    ) -> Plot
    plot_drawdowns(w, rd::ReturnsResult, fees = nothing; slv, compound, alpha, kappa, rw, kwargs...) -> Plot
    plot_drawdowns(res::OptimisationResult, rd::ReturnsResult; slv, compound, alpha, kappa, rw, kwargs...) -> Plot
    plot_drawdowns(pred::PredictionResult; slv, compound, alpha, kappa, rw, kwargs...) -> Plot
    plot_drawdowns(mpred::MultiPeriodPredictionResult; slv, compound, alpha, kappa, rw, kwargs...) -> Plot
    plot_drawdowns(ppred::PopulationPredictionResult; slv, compound, alpha, kappa, rw, kwargs...) -> Plot
    plot_drawdowns(ret::VecNum; ts::AbstractVector, slv, compound, alpha, kappa, rw, kwargs...) -> Plot

Plot the cumulative returns of a portfolio above its drawdown, in two panels that share the time axis.

The upper panel is [`cumulative_returns`](@ref) of the net portfolio returns. The lower panel is their [`drawdowns`](@ref) in percent, with one horizontal line for each of these measures of the returns, negated and in percent: [`AverageDrawdown`](@ref), [`UlcerIndex`](@ref), [`DrawdownatRisk`](@ref), [`ConditionalDrawdownatRisk`](@ref) and [`MaximumDrawdown`](@ref). With `slv`, [`EntropicDrawdownatRisk`](@ref) and [`RelativisticDrawdownatRisk`](@ref) add two more lines. Under `compound` each measure is its relative form, such as [`RelativeMaximumDrawdown`](@ref). A population draws the mean of the return series of its members, which must be of one length, on the time axis of the first member. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

A gapped panel reaches this figure only through the returns that the caller gives it. A prediction arity is finite already, because the fold set the Held Gap to zero once in [`predict`](@ref). A `w, X` or `w, rd` arity computes on the matrix that the caller holds and checks nothing, and `X * w` is `NaN` at a gap even where the weight is zero. Score a gapped panel through `predict(res, rd)` and plot the prediction.

# Arguments

  - `w`: Portfolio weights.
  - `X`: Asset returns matrix (observations × assets).
  - `fees`: Transaction fees, or `nothing`.
  - `slv`: Solver of the entropic and relativistic lines. `nothing` leaves out the two lines.
  - `ts`: Labels of the time axis. A returns result or a prediction gives its own.
  - `compound`: Compound the returns and use the relative drawdown measures when `true`.
  - `alpha`: Significance level of the drawdown at risk lines.
  - `kappa`: Deformation parameter of the relativistic line.
  - `rw`: Observation weights of the average drawdown line, or `nothing`.
  - `res`: Optimisation result. The figure reads its weights and its fees on its investable universe.
  - `pred`, `mpred`, `ppred`: A prediction, a walk-forward path or a population, whose return series the figure draws.
  - `ret`: Net portfolio returns.

# Validation

  - `0 < alpha < 1`, else a `DomainError` is raised, whether or not `slv` is given.
  - `0 < kappa < 1`, else a `DomainError` is raised, whether or not `slv` is given.

# Returns

  - `plt::Plots.Plot`: A layout of two panels, the cumulative returns above the drawdown.

# Related

  - [`drawdowns`](@ref)
"""
function plot_drawdowns end

## ──────────────────────────────────────────────────────────────────────────────
## Risk/return scatter
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_measures(
        w::VecNum_VecVecNum,
        pr::Pr_RR,
        fees::Option{<:Fees} = nothing;
        x::BaseRM_VecBaseRM = Variance(),
        y::BaseRM_VecBaseRM = ExpectedReturn(),
        z::Option{<:BaseRM_VecBaseRM} = nothing,
        c::BaseRM_VecBaseRM = ExpectedReturnRiskRatio(; rk=x, rt=ArithmeticReturn(), rf=0),
        slv::Option{<:Slv_VecSlv} = nothing,
        factory::Bool = true,
        kwargs...
    ) -> Plot
    plot_measures(
        res_vec::AbstractVector{<:OptimisationResult},
        pr::Option{<:Pr_RR} = nothing;
        x, y, z, c, slv, fees, factory, kwargs...
    ) -> Plot
    plot_measures(
        pred::Union{<:PredictionResult, <:MultiPeriodPredictionResult, <:PopulationPredictionResult};
        x::BaseRM_VecBaseRM = ConditionalValueatRisk(),
        y::BaseRM_VecBaseRM = MeanReturn(),
        z::Option{<:BaseRM_VecBaseRM} = nothing,
        c::BaseRM_VecBaseRM = MeanReturnRiskRatio(; rk = x, rt = MeanReturn(), rf = 0),
        slv::Option{<:Slv_VecSlv} = nothing,
        factory::Bool = true,
        plt = nothing,
        kwargs...
    ) -> Plot

Plot a scatter of two or three measures over a collection of portfolios, coloured by a fourth.

Each point is one portfolio. Its coordinates are [`expected_risk`](@ref) of `x`, `y` and `z`, and its colour is [`expected_risk`](@ref) of `c`. The default colour is the ratio of the return of `x` to its risk. When `factory` is `true`, [`factory`](@ref) binds each measure to the prior and to `slv` first. The prediction arity reads the return series that the fold kept, so its defaults are measures that read a series: [`ConditionalValueatRisk`](@ref), [`MeanReturn`](@ref) and [`MeanReturnRiskRatio`](@ref). A prediction and a walk-forward path draw one point, and a population draws one point per member. With `plt`, the arity draws into that plot. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

Each axis takes one measure or a **vector** of them, and a vector becomes the one number that the axis plots. The figure names no scalariser, so a **bare vector on an axis takes [`SumScalariser`](@ref)**, each element weighted by its own `settings.scale`. A scalariser reaches an axis only through the `sca` field of a wrapper, [`ExpectedReturnRiskRatio`](@ref), [`MeanReturnRiskRatio`](@ref) or [`NonOptimisationRiskRatio`](@ref), and each axis can name its own. Those wrappers are all ratios, so the figure cannot apply a [`MaxScalariser`](@ref) to a **plain** vector of risks. Scalarise the vector with [`expected_risk`](@ref) and plot the numbers for that.

[`expected_risk`](@ref) reduces a Prior Result to the Investable Mask at its own entry, so a gapped prior needs no step here. A prediction arity is finite already, because the fold set the Held Gap to zero once in [`predict`](@ref).

# Arguments

  - `w`: Portfolio weights vector, or a vector of weight vectors.
  - `pr`: Prior result or returns result. The result arity scores each result against its own prior when `pr` is `nothing`.
  - `fees`: Transaction fees, or `nothing`.
  - `x`: Measure of the horizontal axis, or a vector of them.
  - `y`: Measure of the vertical axis, or a vector of them.
  - `z`: Measure of the third axis, or a vector of them, or `nothing` for a flat scatter.
  - `c`: Measure of the colour, or a vector of them.
  - `slv`: Solver that [`factory`](@ref) binds to each measure.
  - `factory`: Bind each measure through [`factory`](@ref) before the figure scores it.
  - `res_vec`: Optimisation results.
  - `pred`: A prediction, a walk-forward path or a population.
  - `plt`: A plot to draw into, or `nothing` for a new one.

# Validation

  - The rules of [`expected_risk`](@ref) for each measure.

# Returns

  - `plt::Plots.Plot`: A scatter, in three dimensions when `z` is given.

# Related

  - [`expected_risk`](@ref)
  - [`Scalariser`](@ref)
"""
function plot_measures end

## ──────────────────────────────────────────────────────────────────────────────
## Return histogram
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_histogram(
        w::ArrNum,
        X::MatNum,
        fees::Option{<:Fees} = nothing;
        slv::Option{<:Slv_VecSlv} = nothing,
        alpha::Number = 0.05,
        kappa::Number = 0.3,
        rw = nothing,
        points::Integer = 0,
        reference::Bool = true,
        kwargs...
    ) -> Plot
    plot_histogram(w, rd::ReturnsResult, fees = nothing; slv, alpha, kappa, rw, points, reference, kwargs...) -> Plot
    plot_histogram(res::OptimisationResult, rd::ReturnsResult; slv, alpha, kappa, rw, points, reference, kwargs...) -> Plot
    plot_histogram(pred::PredictionResult; slv, alpha, kappa, rw, points, reference, kwargs...) -> Plot
    plot_histogram(mpred::MultiPeriodPredictionResult; slv, alpha, kappa, rw, points, reference, kwargs...) -> Plot
    plot_histogram(ppred::PopulationPredictionResult; slv, alpha, kappa, rw, points, reference, kwargs...) -> Plot
    plot_histogram(ret::VecNum; slv, alpha, kappa, rw, points, reference, kwargs...) -> Plot

Plot the histogram of the returns of a portfolio, with a vertical line for each of several statistics and an optional fitted Normal density.

The histogram is normalised to a density. Vertical lines mark, in this order: the mean ``\\bar{r}``, and ``\\bar{r}`` less the standard deviation, less the mean absolute deviation, and less the Gini mean difference; the negated value at risk, conditional value at risk and tail Gini; with `slv`, the negated entropic and relativistic values at risk; and the worst return. A population draws the mean of the return series of its members. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

A gapped panel reaches this figure only through the returns that the caller gives it. A prediction arity is finite already, because the fold set the Held Gap to zero once in [`predict`](@ref). A `w, X` or `w, rd` arity computes on the matrix that the caller holds and checks nothing, and `X * w` is `NaN` at a gap even where the weight is zero. Score a gapped panel through `predict(res, rd)` and plot the prediction.

# Mathematical definition

```math
\\begin{align}
P &= \\begin{cases} \\lceil 4 \\sqrt{T} \\rceil & p = 0\\,, \\\\ p & p > 0\\,, \\end{cases} \\\\
\\hat{\\mu} &= \\frac{1}{T} \\sum_{t=1}^{T} r_t\\,, \\\\
\\hat{\\sigma}^2 &= \\frac{1}{T} \\sum_{t=1}^{T} (r_t - \\hat{\\mu})^2\\,.
\\end{align}
```

Where:

  - ``r_t``: Net portfolio return at observation ``t``.
  - ``p``: The `points` keyword.
  - ``P``: Number of points of the grid, evenly spaced from the worst to the best return, on which the figure evaluates the Normal density.
  - ``\\hat{\\mu}``, ``\\hat{\\sigma}^2``: Maximum likelihood mean and variance of the fitted Normal law. The divisor is ``T``, so ``\\hat{\\sigma}`` is not the standard deviation of the line, whose divisor is ``T - 1``.
  - $(math_dict[:T])

# Arguments

  - `w`: Portfolio weights.
  - `X`: Asset returns matrix (observations × assets).
  - `fees`: Transaction fees, or `nothing`.
  - `slv`: Solver of the entropic and relativistic lines. `nothing` leaves out the two lines.
  - `alpha`: Significance level of the tail lines.
  - `kappa`: Deformation parameter of the relativistic line.
  - `rw`: Observation weights of the mean absolute deviation line and of the four value at risk lines (plain, conditional, entropic and relativistic), or `nothing`. The mean, the standard deviation, the Gini mean difference and the tail Gini lines read no weights.
  - `points`: Number of points of the density grid, or `0` for the default.
  - `reference`: Draw the fitted Normal density when `true`.
  - `res`: Optimisation result. The figure reads its weights and its fees on its investable universe.
  - `pred`, `mpred`, `ppred`: A prediction, a walk-forward path or a population, whose return series the figure draws.
  - `ret`: Net portfolio returns.

# Validation

  - `0 < alpha < 1`, else a `DomainError` is raised.
  - `0 < kappa < 1`, else a `DomainError` is raised.
  - `points >= 0`, else a `DomainError` is raised.

# Returns

  - `plt::Plots.Plot`: A density histogram with its vertical lines, and the Normal density when `reference` is `true`.

# Related

  - [`calc_net_returns`](@ref)
  - [`ConditionalValueatRisk`](@ref)
  - [`plot_drawdowns`](@ref)
"""
function plot_histogram end

## ──────────────────────────────────────────────────────────────────────────────
## Network / phylogeny
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_network(
        pl::NwE_ClE_Cl,
        X::MatNum,
        nx::AbstractVector = 1:size(X, 2),
        w::Option{<:VecNum} = nothing;
        threshold::Number = 0,
        kwargs...
    ) -> Plot
    plot_network(pl, pr::Pr_RR, w::Option{<:VecNum} = nothing; nx = 1:size(pr.X, 2), threshold, kwargs...) -> Plot
    plot_network(pl, res::OptimisationResult; rd::Option{<:Pr_RR} = nothing, nx = 1:length(res.w), threshold, kwargs...) -> Plot

Plot the network of the assets as a graph, with `GraphRecipes.graphplot`.

The graph is the matrix that [`phylogeny_matrix`](@ref) fits with `pl`: a tree, a planar graph, or the adjacency of a clustering. An entry whose magnitude is at most `threshold` draws no edge. Every node has one size, unless `w` is given, which scales the node of asset ``i`` by ``\\lvert w_i \\rvert / \\max_j \\lvert w_j \\rvert``. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots` and `GraphRecipes`.

A non-investable asset is **not drawn**. The figure fits a phylogeny matrix on `pr.X`, and a plain moment estimator refuses a gapped sample, so the prior arity reduces itself, its names and its weights to the Investable Mask at its entry through [`investable_plot_view`](@ref). The graph is the investable universe, and no edge of a dead asset is drawn. The bare `X` and [`ReturnsResult`](@ref) arities carry no moments, so no mask exists to derive, and they compute on the matrix that the caller holds.

# Arguments

  - `pl`: Network or clustering estimator, or a clustering result.
  - `X`: Asset returns matrix (observations × assets).
  - `nx`: Asset names. A [`ReturnsResult`](@ref) that carries `nx` replaces the argument with its own.
  - `w`: Portfolio weights that size the nodes, or `nothing` for one size.
  - `threshold`: Largest magnitude of an entry that draws no edge.
  - `pr`: Prior result or [`ReturnsResult`](@ref). The figure reads its `X`.
  - `res`: Optimisation result. The figure reads its weights, which size the nodes, on its investable universe.
  - `rd`: Prior result or returns result, or `nothing` for the prior that `res` was fitted on.

# Validation

  - The rules of [`phylogeny_matrix`](@ref).

# Returns

  - `plt::Plots.Plot`: A graph plot.

# Related

  - [`phylogeny_matrix`](@ref)
"""
function plot_network end

## ──────────────────────────────────────────────────────────────────────────────
## Centrality
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_centrality(
        cte::AbstractCentralityEstimator,
        X::MatNum,
        nx::AbstractVector = 1:size(X, 2);
        N::Option{<:Number} = nothing,
        percentage::Bool = true,
        kwargs...
    ) -> Plot
    plot_centrality(cte, pr::AbstractPriorResult, nx = 1:size(pr.X, 2); N, percentage, kwargs...) -> Plot
    plot_centrality(cte, rd::ReturnsResult; N, percentage, kwargs...) -> Plot
    plot_centrality(cte, res::OptimisationResult, rd::ReturnsResult; N, percentage, kwargs...) -> Plot

Plot the centrality score of each asset as a bar chart, sorted from the largest.

The scores are [`centrality_vector`](@ref) of `X`. [`relevant_assets`](@ref) chooses how many bars to draw, and the assets that it leaves out draw no bar. The figure has no `Others` bar. Under `percentage` every score is divided by the sum over **all** assets, so the drawn bars sum to less than one when the count truncates. The result arity reads `rd` alone, and the result gives only the count of the default names. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

A non-investable asset is **not drawn**. The figure fits a centrality vector on `pr.X`, and a plain moment estimator refuses a gapped sample, so the prior arity reduces to the Investable Mask at its entry through [`investable_plot_view`](@ref). The axis is the investable universe. The bare `X` and [`ReturnsResult`](@ref) arities carry no moments, so no mask exists to derive, and they compute on the matrix that the caller holds.

# Arguments

  - $(arg_dict[:cte])
  - `X`: Asset returns matrix (observations × assets).
  - `nx`: Asset names.
  - `N`: The count or the share of the bars to draw, or `nothing` for the effective number of the scores. [`relevant_assets`](@ref) states the rule.
  - `percentage`: Divide the scores by their sum over all assets when `true`.
  - `pr`: Prior result. The figure reads its `X`.
  - `rd`: Returns result. The figure reads its `X` and its `nx`.
  - `res`: Optimisation result.

# Validation

  - The rules of [`centrality_vector`](@ref).

# Returns

  - `plt::Plots.Plot`: A bar chart.

# Related

  - [`centrality_vector`](@ref)
"""
function plot_centrality end

## ──────────────────────────────────────────────────────────────────────────────
## Correlation / covariance heatmap
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_correlation(X::MatNum, nx::AbstractVector = 1:size(X, 1); kwargs...) -> Plot
    plot_correlation(pr::AbstractPriorResult, nx = 1:size(pr.sigma, 1); kwargs...) -> Plot
    plot_correlation(pr::AbstractPriorResult, rd::ReturnsResult; kwargs...) -> Plot
    plot_correlation(res::OptimisationResult[, rd::ReturnsResult]; kwargs...) -> Plot
    plot_correlation(pred::PredictionResult[, rd::ReturnsResult]; kwargs...) -> Plot

Plot a correlation matrix as a heatmap, with no clustering and no dendrogram.

When a diagonal entry of the matrix is not one, the figure takes it for a covariance and rescales a copy of it to a correlation, whose colour limits are `(-1, 1)`. A matrix whose diagonal is all ones is drawn unchanged, between its smallest and its largest entry. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

A non-investable asset of a prior that the caller gives is drawn rather than removed. It carries `NaN` down its row and its column of `sigma`, so those cells are blank, and the blank shows the gap. Its diagonal entry is `NaN` and not one, so the figure rescales and the colour limits stay `(-1, 1)`, so one gap does not flatten the scale. A result arity draws the prior that the result was fitted on, which is already reduced to its investable universe, under the names of those assets.

# Mathematical definition

```math
\\begin{align}
\\rho_{ij} = \\frac{S_{ij}}{\\sqrt{S_{ii} S_{jj}}}\\,.
\\end{align}
```

Where:

  - ``S_{ij}``: Entry of the matrix that the caller gives.
  - ``\\rho_{ij}``: Entry that the heatmap draws when a diagonal entry of ``\\mathbf{S}`` is not one.

# Arguments

  - `X`: Covariance or correlation matrix.
  - `nx`: Asset names.
  - `pr`: Prior result. The figure reads its `sigma`.
  - `rd`: Returns result. The figure reads its `nx`.
  - `res`: Optimisation result. The figure reads the prior it was fitted on.
  - `pred`: A prediction. The figure reads the prior of its result.

# Returns

  - `plt::Plots.Plot`: A heatmap.

# Related

  - [`plot_clusters`](@ref)
"""
function plot_correlation end

## ──────────────────────────────────────────────────────────────────────────────
## Expected returns bar chart
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_mu(
        mu::VecNum,
        nx::AbstractVector = 1:length(mu);
        N::Option{<:Number} = nothing,
        kwargs...
    ) -> Plot
    plot_mu(pr::AbstractPriorResult[, nx]; N, kwargs...) -> Plot
    plot_mu(pr::AbstractPriorResult, rd::ReturnsResult; N, kwargs...) -> Plot
    plot_mu(res::OptimisationResult[, rd::ReturnsResult]; N, kwargs...) -> Plot
    plot_mu(pred::PredictionResult[, rd::ReturnsResult]; N, kwargs...) -> Plot

Plot the expected return of each asset as a bar chart.

[`relevant_assets`](@ref) ranks the assets by ``\\lvert \\mu_i \\rvert`` and chooses how many bars to draw. The drawn bars keep the order of the asset axis, and the assets that it leaves out draw no bar. With no `N`, the count is the effective number of the magnitudes, so it can be smaller than the universe; pass `N` equal to the number of assets to draw every one. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

A non-investable asset of a prior that the caller gives is drawn rather than removed. It carries `NaN` in the moments of the prior, so its bar is blank, and that blank shows the gap. The ranking reads [`finite_magnitudes`](@ref), so the blank never takes a top slot from a live asset, and the frame is unchanged when the count does not truncate. A result arity draws the prior that the result was fitted on, which is already reduced to its investable universe, under the names of those assets.

# Arguments

  - `mu`: Expected returns vector.
  - `nx`: Asset names.
  - `N`: The count or the share of the bars to draw, or `nothing` for the effective number of the magnitudes. [`relevant_assets`](@ref) states the rule.
  - `pr`: Prior result. The figure reads its `mu`.
  - `rd`: Returns result. The figure reads its `nx`.
  - `res`: Optimisation result. The figure reads the prior it was fitted on.
  - `pred`: A prediction. The figure reads the prior of its result.

# Returns

  - `plt::Plots.Plot`: A bar chart.

# Related

  - [`plot_sigma`](@ref)
"""
function plot_mu end

## ──────────────────────────────────────────────────────────────────────────────
## Asset volatility bar chart
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_sigma(
        sigma::MatNum,
        nx::AbstractVector = 1:size(sigma, 1);
        variance::Bool = false,
        N::Option{<:Number} = nothing,
        kwargs...
    ) -> Plot
    plot_sigma(pr::AbstractPriorResult[, nx]; variance, N, kwargs...) -> Plot
    plot_sigma(pr::AbstractPriorResult, rd::ReturnsResult; variance, N, kwargs...) -> Plot
    plot_sigma(res::OptimisationResult[, rd::ReturnsResult]; variance, N, kwargs...) -> Plot
    plot_sigma(pred::PredictionResult[, rd::ReturnsResult]; variance, N, kwargs...) -> Plot

Plot the volatility of each asset, the square root of the diagonal of the covariance, as a bar chart.

With no `N` the figure draws every asset. An `N` is a count, ``\\min(\\max(\\lceil N \\rceil, 1), M)`` for ``M`` assets, of the assets with the largest values, drawn in the order of the asset axis. Unlike [`plot_mu`](@ref), an `N` in `(0, 1]` is not a share, and it draws one bar. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

A non-investable asset of a prior that the caller gives is drawn rather than removed. It carries `NaN` on the diagonal of `sigma`, so its bar is blank, and that blank shows the gap. The ranking reads [`finite_magnitudes`](@ref), so the blank never takes a top slot from a live asset, and the frame is unchanged when the count does not truncate. A result arity draws the prior that the result was fitted on, which is already reduced to its investable universe, under the names of those assets.

# Arguments

  - `sigma`: Covariance matrix.
  - `nx`: Asset names.
  - `variance`: Draw the diagonal itself, the variances, when `true`.
  - `N`: The count of the bars to draw, or `nothing` for every asset.
  - `pr`: Prior result. The figure reads its `sigma`.
  - `rd`: Returns result. The figure reads its `nx`.
  - `res`: Optimisation result. The figure reads the prior it was fitted on.
  - `pred`: A prediction. The figure reads the prior of its result.

# Returns

  - `plt::Plots.Plot`: A bar chart.

# Related

  - [`plot_mu`](@ref)
  - [`plot_correlation`](@ref)
"""
function plot_sigma end

## ──────────────────────────────────────────────────────────────────────────────
## Factor loadings heatmap
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_factor_loadings(
        M::MatNum,
        nx::AbstractVector = 1:size(M, 1),
        nf::AbstractVector = 1:size(M, 2);
        kwargs...
    ) -> Plot
    plot_factor_loadings(pr::AbstractPriorResult, nx = nothing, nf = nothing; kwargs...) -> Plot
    plot_factor_loadings(pr::AbstractPriorResult, rd::ReturnsResult; kwargs...) -> Plot
    plot_factor_loadings(res::OptimisationResult[, rd::ReturnsResult]; kwargs...) -> Plot
    plot_factor_loadings(pred::PredictionResult[, rd::ReturnsResult]; kwargs...) -> Plot

Plot the factor loadings matrix, assets by factors, as a heatmap on a diverging colour scale centred at zero.

The prior arities draw `pr.rr.M`, the loadings of the regression that the prior carries. The matrix arity draws `M` and needs no prior. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

A non-investable asset of a prior that the caller gives is drawn rather than removed. It carries `NaN` down its row of the loadings, so that row is blank cells, and the blank shows the gap. The colour limits are [`finite_symmetric_clim`](@ref), so one gap does not flatten the scale. The gap is on the asset axis alone, so the factor axis is unaffected. A result arity draws the prior that the result was fitted on, which is already reduced to its investable universe.

# Arguments

  - `M`: Loadings matrix (assets × factors).
  - `nx`: Asset names, or `nothing` to number them.
  - `nf`: Factor names, or `nothing` to number them.
  - `pr`: Prior result that carries a factor block.
  - `rd`: Returns result. The figure reads its `nx` and its `nf`.
  - `res`: Optimisation result. The figure reads the prior it was fitted on.
  - `pred`: A prediction. The figure reads the prior of its result.

# Validation

  - `pr.rr` is not `nothing`, else [`assert_prior_regression`](@ref) raises an `IsNothingError`.

# Returns

  - `plt::Plots.Plot`: A heatmap.

# Related

  - [`plot_factor_sigma`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`assert_prior_regression`](@ref)
"""
function plot_factor_loadings end

## ──────────────────────────────────────────────────────────────────────────────
## Factor covariance heatmap
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_factor_sigma(
        f_sigma::MatNum,
        nf::AbstractVector = 1:size(f_sigma, 1);
        kwargs...
    ) -> Plot
    plot_factor_sigma(pr::AbstractPriorResult, nf = nothing; kwargs...) -> Plot
    plot_factor_sigma(pr::AbstractPriorResult, rd::ReturnsResult; kwargs...) -> Plot
    plot_factor_sigma(res::OptimisationResult[, rd::ReturnsResult]; kwargs...) -> Plot
    plot_factor_sigma(pred::PredictionResult[, rd::ReturnsResult]; kwargs...) -> Plot

Plot the correlation of the factor returns as a heatmap, through [`plot_correlation`](@ref).

The prior arities draw the factor covariance of the low order factor block that [`factor_plot_prior`](@ref) finds, rescaled to a correlation by the rule of [`plot_correlation`](@ref). The matrix arity draws `f_sigma` and needs no prior. [`plot_factor_forecast_correlation`](@ref) draws the same correlation, always with the colour limits `(-1, 1)`. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

The figure draws the **factor** axis, and a non-investable asset is on the asset axis, so a gap never reaches it. It needs neither a blank nor a reduction.

# Arguments

  - `f_sigma`: Factor covariance matrix (factors × factors).
  - `nf`: Factor names, or `nothing` to number them.
  - `pr`: Prior result that carries a factor block.
  - `rd`: Returns result. The figure reads its `nf`.
  - `res`: Optimisation result. The figure reads the prior it was fitted on.
  - `pred`: A prediction. The figure reads the prior of its result.

# Validation

  - `pr.rr` is not `nothing`, else [`assert_prior_regression`](@ref) raises an `IsNothingError`. The regression and the low order factor block are present together, so the check establishes both.

# Returns

  - `plt::Plots.Plot`: A heatmap.

# Related

  - [`plot_factor_loadings`](@ref)
  - [`plot_correlation`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`assert_prior_regression`](@ref)
"""
function plot_factor_sigma end

## ──────────────────────────────────────────────────────────────────────────────
## Eigenvalue spectrum
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_eigenspectrum(
        sigma::MatNum;
        N_obs::Option{<:Integer} = nothing,
        reference::Bool = true,
        kwargs...
    ) -> Plot
    plot_eigenspectrum(pr::AbstractPriorResult; reference, kwargs...) -> Plot
    plot_eigenspectrum(pr::AbstractPriorResult, rd::ReturnsResult; reference, kwargs...) -> Plot
    plot_eigenspectrum(res::OptimisationResult[, rd::ReturnsResult]; reference, kwargs...) -> Plot
    plot_eigenspectrum(pred::PredictionResult[, rd::ReturnsResult]; reference, kwargs...) -> Plot

Plot the eigenvalues of a covariance or correlation matrix as a bar chart, sorted from the largest.

With a number of observations, the figure draws the edges of the Marčenko–Pastur bulk as reference lines: the upper edge always, and the lower edge when there are fewer assets than observations. An arity that takes `rd` reads the number of observations off `rd.X`. The others reach it only through the `N_obs` keyword, and without it they draw no line. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

A non-investable asset is **not drawn**. `eigvals(Symmetric(sigma))` refuses a `NaN`, so the prior arity reduces to the Investable Mask at its entry through [`investable_plot_view`](@ref), and the spectrum is the spectrum of the investable block. The bare `sigma` arity computes on the matrix that the caller holds, and a gap in it raises rather than reduces.

# Mathematical definition

```math
\\begin{align}
q &= \\frac{N}{T}\\,, \\\\
\\bar{\\sigma}^2 &= \\frac{\\mathrm{tr}(\\mathbf{\\Sigma})}{N}\\,, \\\\
\\lambda_{\\pm} &= \\bar{\\sigma}^2 \\left(1 \\pm \\sqrt{q}\\right)^2\\,.
\\end{align}
```

Where:

  - ``\\mathbf{\\Sigma}``: The matrix whose spectrum the figure draws.
  - $(math_dict[:N])
  - $(math_dict[:T])
  - ``q``: Ratio of the number of assets to the number of observations.
  - ``\\bar{\\sigma}^2``: Mean variance, the mean eigenvalue.
  - ``\\lambda_{+}``, ``\\lambda_{-}``: Upper and lower edges of the Marčenko–Pastur bulk. The figure draws ``\\lambda_{-}`` only when ``q < 1``.

# Arguments

  - `sigma`: Covariance or correlation matrix.
  - `N_obs`: Number of observations, or `nothing` for no reference line.
  - `reference`: Draw the reference lines when `true` and the number of observations is known.
  - `pr`: Prior result. The figure reads its `sigma`.
  - `rd`: Returns result. The figure reads the number of rows of its `X`.
  - `res`: Optimisation result. The figure reads the prior it was fitted on.
  - `pred`: A prediction. The figure reads the prior of its result.

# Validation

  - Every entry of `sigma` is finite, else `eigvals` raises.

# Returns

  - `plt::Plots.Plot`: A bar chart, with the reference lines when they apply.

# Related

  - [`plot_correlation`](@ref)
"""
function plot_eigenspectrum end

## ──────────────────────────────────────────────────────────────────────────────
## Rolling risk/return measure
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_rolling_measure(
        r::BaseRM_VecBaseRM,
        w::VecNum,
        X::MatNum,
        fees::Option{<:Fees} = nothing;
        ts::AbstractVector = 1:size(X, 1),
        rolling::Integer = 0,
        sca::Scalariser = SumScalariser(),
        kwargs...
    ) -> Plot
    plot_rolling_measure(r, w, rd::ReturnsResult, fees = nothing; rolling, sca, kwargs...) -> Plot
    plot_rolling_measure(r, res::OptimisationResult, rd::ReturnsResult; rolling, sca, kwargs...) -> Plot
    plot_rolling_measure(r, pred::PredictionResult; rolling, sca, kwargs...) -> Plot
    plot_rolling_measure(r, mpred::MultiPeriodPredictionResult; rolling, sca, kwargs...) -> Plot
    plot_rolling_measure(r, ppred::PopulationPredictionResult; rolling, sca, kwargs...) -> Plot
    plot_rolling_measure(r, ret::VecNum; ts::AbstractVector, rolling, sca, kwargs...) -> Plot

Plot a risk or return measure of the portfolio returns over a rolling window, against the last date of each window.

[`rolling_window_measure`](@ref) scores each window of the net portfolio returns. Each window plots **one** number, so a vector of measures becomes its scalarised aggregate, not several lines. A population draws the mean of the return series of its members, on the time axis of the first member. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

A gapped panel reaches this figure only through the returns that the caller gives it. A prediction arity is finite already, because the fold set the Held Gap to zero once in [`predict`](@ref). A `w, X` or `w, rd` arity computes on the matrix that the caller holds and checks nothing, and `X * w` is `NaN` at a gap even where the weight is zero. Score a gapped panel through `predict(res, rd)` and plot the prediction.

# Mathematical definition

```math
\\begin{align}
W &= \\begin{cases} \\lceil \\sqrt{T} \\rceil & \\text{rolling} = 0\\,, \\\\ \\text{rolling} & \\text{otherwise}\\,, \\end{cases} \\\\
R_t &= \\rho\\left(r_{t-W+1}, \\ldots, r_t\\right)\\,, \\quad t = W, \\ldots, T\\,.
\\end{align}
```

Where:

  - $(math_dict[:W_roll])
  - $(math_dict[:R_t_roll])
  - ``\\rho``: The measure `r`, or the aggregate that `sca` forms from a vector of them.
  - ``r_t``: Net portfolio return at observation ``t``.
  - $(math_dict[:T])

# Arguments

  - `r`: Risk or return measure, or a vector of them that `sca` combines. A measure can carry its own solver, such as `EntropicValueatRisk(; slv = ...)`.
  - `w`: Portfolio weights.
  - `X`: Asset returns matrix (observations × assets).
  - `fees`: Transaction fees, or `nothing`.
  - `ts`: Labels of the time axis. A returns result or a prediction gives its own.
  - `rolling`: Length of the window, or `0` for the default.
  - `sca`: Scalariser that combines the measures in `r`. It has no effect when `r` is one measure.
  - `res`: Optimisation result. The figure reads its weights and its fees on its investable universe.
  - `pred`, `mpred`, `ppred`: A prediction, a walk-forward path or a population, whose return series the figure draws.
  - `ret`: Net portfolio returns.

# Validation

  - `rolling >= 0`, else a `DomainError` is raised.
  - ``W \\leq T``, else [`rolling_window_measure`](@ref) raises a `DomainError`.
  - Each measure reads a return series, else [`expected_risk_from_returns`](@ref) raises an `ArgumentError`.

# Returns

  - `plt::Plots.Plot`: A line plot of ``T - W + 1`` points.

# Related

  - [`expected_risk`](@ref)
  - [`rolling_window_measure`](@ref)
"""
function plot_rolling_measure end

## ──────────────────────────────────────────────────────────────────────────────
## Weight stability across folds
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_weight_stability(
        mpred::MultiPeriodPredictionResult;
        N::Option{<:Number} = nothing,
        kwargs...
    ) -> Plot
    plot_weight_stability(ppred::PopulationPredictionResult; N, kwargs...) -> Plot

Plot the distribution of the weight of each asset over the folds of a path or the members of a population, as one box per asset.

The weights are the target weights of each fold, or of each member. A member that is a walk-forward path gives one weight vector, the mean of its fold weights. [`relevant_assets`](@ref) ranks the assets by their mean absolute weight and chooses how many boxes to draw, in the order of the asset axis. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

# Mathematical definition

```math
\\begin{align}
m_i = \\frac{1}{K} \\sum_{k=1}^{K} \\lvert w_{i,k} \\rvert\\,.
\\end{align}
```

Where:

  - ``w_{i,k}``: Weight of asset ``i`` in fold or member ``k``.
  - ``K``: Number of folds or members.
  - ``m_i``: Mean absolute weight of asset ``i``, the key of the ranking.

# Arguments

  - `mpred`: A walk-forward path.
  - `ppred`: A population of predictions.
  - `N`: The count or the share of the boxes to draw, or `nothing` for the effective number of the mean absolute weights. [`relevant_assets`](@ref) states the rule.

# Returns

  - `plt::Plots.Plot`: A box plot.

# Related

  - [`MultiPeriodPredictionResult`](@ref)
  - [`PopulationPredictionResult`](@ref)
"""
function plot_weight_stability end

## ──────────────────────────────────────────────────────────────────────────────
## Cross-validation scores
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_cv_scores(
        scores::AbstractVector{<:Number},
        labels::AbstractVector = 1:length(scores);
        kwargs...
    ) -> Plot
    plot_cv_scores(
        r::BaseRM_VecBaseRM,
        mpred::MultiPeriodPredictionResult;
        sca::Scalariser = SumScalariser(),
        kwargs...
    ) -> Plot
    plot_cv_scores(
        r::BaseRM_VecBaseRM,
        ppred::PopulationPredictionResult;
        sca::Scalariser = SumScalariser(),
        kwargs...
    ) -> Plot

Plot the score of each fold or population member as a bar chart.

The score of a fold or a member is [`expected_risk`](@ref) of `r` on the return series that it kept. Each bar is **one** number, so a vector of measures becomes its aggregate under `sca`, not several bars. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

# Arguments

  - `scores`: The scores to draw.
  - `labels`: Label of each bar.
  - `r`: Risk or return measure, or a vector of them that `sca` combines.
  - `sca`: Scalariser that combines the measures in `r`. It has no effect when `r` is one measure.
  - `mpred`: A walk-forward path.
  - `ppred`: A population of predictions.

# Validation

  - The rules of [`expected_risk`](@ref) on a prediction.

# Returns

  - `plt::Plots.Plot`: A bar chart with one bar per fold or member.

# Related

  - [`expected_risk`](@ref)
  - [`Scalariser`](@ref)
  - [`MultiPeriodPredictionResult`](@ref)
  - [`PopulationPredictionResult`](@ref)
"""
function plot_cv_scores end

## ──────────────────────────────────────────────────────────────────────────────
## Portfolio turnover
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_turnover(
        w_series::AbstractVector{<:VecNum};
        ts::AbstractVector = 1:length(w_series),
        kwargs...
    ) -> Plot
    plot_turnover(mpred::MultiPeriodPredictionResult; kwargs...) -> Plot

Plot the turnover of a sequence of portfolios, the ``L_1`` change of the weights from one portfolio to the next.

[`calc_turnover`](@ref) computes the turnover. The first portfolio has no predecessor, so the verb answers `NaN` there and the figure drops it. A walk-forward path reads the target weights of each fold, not the held or drifted weights, against the last date of each fold, or against the position of the fold when the path carries no dates. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

# Mathematical definition

```math
\\begin{align}
\\tau_k = \\sum_{i=1}^{N} \\lvert w_{k,i} - w_{k-1,i} \\rvert\\,, \\quad k = 2, \\ldots, K\\,.
\\end{align}
```

Where:

  - ``w_{k,i}``: Weight of asset ``i`` in portfolio ``k``.
  - ``\\tau_k``: Turnover into portfolio ``k``.
  - ``K``: Number of portfolios.
  - $(math_dict[:N])

# Arguments

  - `w_series`: Weight vectors, one per portfolio, of equal length.
  - `ts`: Label of each portfolio on the time axis.
  - `mpred`: A walk-forward path.

# Returns

  - `plt::Plots.Plot`: A line plot of ``K - 1`` points.

# Related

  - [`MultiPeriodPredictionResult`](@ref)
  - [`calc_turnover`](@ref)
"""
function plot_turnover end

## ──────────────────────────────────────────────────────────────────────────────
## Composite prior dashboard
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_prior(
        pr::AbstractPriorResult,
        nx::AbstractVector = 1:length(pr.mu);
        N::Option{<:Number} = nothing,
        kwargs...
    ) -> Plot
    plot_prior(pr::AbstractPriorResult, rd::ReturnsResult; N, kwargs...) -> Plot
    plot_prior(res::OptimisationResult[, rd::ReturnsResult]; N, kwargs...) -> Plot
    plot_prior(pred::PredictionResult[, rd::ReturnsResult]; N, kwargs...) -> Plot

Plot a prior result as three panels side by side: the expected returns, the volatilities and the correlation.

 1. The expected returns, through [`plot_mu`](@ref).
 2. The volatilities, through [`plot_sigma`](@ref).
 3. The correlation, through [`plot_correlation`](@ref).

`N` reaches the first two panels, and the two read it by their own rules: [`plot_mu`](@ref) ranks by ``\\lvert \\mu_i \\rvert`` and with no `N` draws the effective number of assets, while [`plot_sigma`](@ref) ranks by volatility and with no `N` draws every asset. So the two panels can draw different assets. The extra keywords reach the outer layout of 1800 × 500 pixels and no panel. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

Every panel draws, so a non-investable asset of a prior that the caller gives is drawn rather than removed. Each panel states what its blank means. A result arity draws the prior that the result was fitted on, which is already reduced to its investable universe.

# Arguments

  - `pr`: Prior result. The figure reads its `mu` and its `sigma`.
  - `nx`: Asset names.
  - `N`: The count or the share of the bars of the first two panels.
  - `rd`: Returns result. The figure reads its `nx`.
  - `res`: Optimisation result. The figure reads the prior it was fitted on.
  - `pred`: A prediction. The figure reads the prior of its result.

# Returns

  - `plt::Plots.Plot`: A layout of three panels.

# Related

  - [`plot_mu`](@ref)
  - [`plot_sigma`](@ref)
  - [`plot_correlation`](@ref)
  - [`LowOrderPrior`](@ref)
"""
function plot_prior end

## ──────────────────────────────────────────────────────────────────────────────
## Factor expected returns bar chart
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_factor_mu(
        f_mu::VecNum,
        nf::AbstractVector = 1:length(f_mu);
        N::Option{<:Number} = nothing,
        kwargs...
    ) -> Plot
    plot_factor_mu(pr::AbstractPriorResult, nf = nothing; N, kwargs...) -> Plot
    plot_factor_mu(pr::AbstractPriorResult, rd::ReturnsResult; N, kwargs...) -> Plot
    plot_factor_mu(res::OptimisationResult[, rd::ReturnsResult]; N, kwargs...) -> Plot
    plot_factor_mu(pred::PredictionResult[, rd::ReturnsResult]; N, kwargs...) -> Plot

Plot the expected return of each factor as a bar chart.

The prior arities draw the factor expected returns of the low order factor block that [`factor_plot_prior`](@ref) finds. The vector arity draws `f_mu` and needs no prior. [`relevant_assets`](@ref) ranks the factors by ``\\lvert \\mu_{f,k} \\rvert`` and chooses how many bars to draw, in the order of the factor axis. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

The figure draws the **factor** axis, and a non-investable asset is on the asset axis, so a gap never reaches it. It needs neither a blank nor a reduction.

# Arguments

  - `f_mu`: Factor expected returns vector.
  - `nf`: Factor names, or `nothing` to number them.
  - `N`: The count or the share of the bars to draw, or `nothing` for the effective number of the magnitudes. [`relevant_assets`](@ref) states the rule.
  - `pr`: Prior result that carries a factor block.
  - `rd`: Returns result. The figure reads its `nf`.
  - `res`: Optimisation result. The figure reads the prior it was fitted on.
  - `pred`: A prediction. The figure reads the prior of its result.

# Validation

  - `pr.rr` is not `nothing`, else [`assert_prior_regression`](@ref) raises an `IsNothingError`.

# Returns

  - `plt::Plots.Plot`: A bar chart.

# Related

  - [`plot_mu`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`assert_prior_regression`](@ref)
"""
function plot_factor_mu end

## ──────────────────────────────────────────────────────────────────────────────
## Benchmark overlay
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_benchmark(
        w::ArrNum,
        X::MatNum,
        B::VecNum_VecVecNum,
        fees::Option{<:Fees} = nothing;
        ts::AbstractVector = 1:size(X, 1),
        nb::Option{<:AbstractVector} = nothing,
        compound::Bool = false,
        kwargs...
    ) -> Plot
    plot_benchmark(w, rd::ReturnsResult, fees = nothing; compound, kwargs...) -> Plot
    plot_benchmark(res::OptimisationResult, rd::ReturnsResult; compound, kwargs...) -> Plot
    plot_benchmark(pred::PredictionResult; compound, kwargs...) -> Plot
    plot_benchmark(mpred::MultiPeriodPredictionResult; compound, kwargs...) -> Plot
    plot_benchmark(
        net_ret::VecNum,
        B::VecNum_VecVecNum;
        ts::AbstractVector,
        nb::Option{<:AbstractVector} = nothing,
        compound::Bool = false,
        kwargs...
    ) -> Plot

Plot the cumulative returns of a portfolio beside those of one or more benchmarks.

Each curve is [`cumulative_returns`](@ref) of its series: the net portfolio returns as a solid line, and each benchmark as a dashed line. A benchmark without a name in `nb` is labelled `Benchmark i`. A returns result carries its benchmark in `B`, and the figure reads it only when it is one series. A matrix `B` is a benchmark per asset, and this figure does not draw it. A prediction whose series is a vector of series draws the first of them. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

# Arguments

  - `w`: Portfolio weights.
  - `X`: Asset returns matrix (observations × assets).
  - `B`: Benchmark return series, or a vector of series.
  - `fees`: Transaction fees, or `nothing`.
  - `ts`: Labels of the time axis. A returns result or a prediction gives its own.
  - `nb`: Benchmark names, at least one per benchmark, or `nothing`.
  - `compound`: Compound the returns of the portfolio and of every benchmark when `true`, and sum them otherwise.
  - `rd`: Returns result. The figure reads its `X`, `B`, `ts` and `nb`.
  - `res`: Optimisation result. The figure reads its weights and its fees on its investable universe.
  - `pred`, `mpred`: A prediction or a walk-forward path, whose return series and benchmark the figure draws.
  - `net_ret`: Net portfolio returns.

# Validation

  - The `B` of the returns result or of the prediction is not `nothing`, else an `ArgumentError` is raised.

# Returns

  - `plt::Plots.Plot`: A line plot with one portfolio curve and one curve per benchmark.

# Related

  - [`plot_portfolio_cumulative_returns`](@ref)
  - [`ReturnsResult`](@ref)
"""
function plot_benchmark end

## ──────────────────────────────────────────────────────────────────────────────
## Coskewness heatmap
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_coskewness(
        sk::MatNum,
        nx::AbstractVector = 1:size(sk, 1);
        kwargs...
    ) -> Plot
    plot_coskewness(pr::HighOrderPrior, nx = nothing; kwargs...) -> Plot
    plot_coskewness(pr::HighOrderPrior, rd::ReturnsResult; kwargs...) -> Plot
    plot_coskewness(res::OptimisationResult[, rd::ReturnsResult]; kwargs...) -> Plot
    plot_coskewness(pred::PredictionResult[, rd::ReturnsResult]; kwargs...) -> Plot

Plot the coskewness matrix, ``N \\times N^2``, of a [`HighOrderPrior`](@ref) as a heatmap on a diverging colour scale centred at zero.

A row is an asset and a column is a pair of assets. The column labels name the pair when there are at most ten assets, and give the position of the column otherwise, on at most about twenty ticks. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

A non-investable asset of a prior that the caller gives is drawn rather than removed. It carries `NaN` down its row and every column of a pair it belongs to, so those cells are blank, and the blank shows the gap. The colour limits are [`finite_symmetric_clim`](@ref), so one gap does not flatten the scale. A result arity draws the prior that the result was fitted on, which is already reduced to its investable universe.

# Arguments

  - `sk`: Coskewness matrix (assets × assets²).
  - `nx`: Asset names, or `nothing` to number them.
  - `pr`: High order prior result. The figure reads its `sk`.
  - `rd`: Returns result. The figure reads its `nx`.
  - `res`: Optimisation result. The figure reads the prior it was fitted on.
  - `pred`: A prediction. The figure reads the prior of its result.

# Validation

  - `pr.sk` is not `nothing`, else an `ArgumentError` is raised.
  - The prior of a result is a [`HighOrderPrior`](@ref), else an `ArgumentError` is raised.

# Returns

  - `plt::Plots.Plot`: A heatmap.

# Related

  - [`plot_cokurtosis`](@ref)
  - [`HighOrderPrior`](@ref)
"""
function plot_coskewness end

## ──────────────────────────────────────────────────────────────────────────────
## Cokurtosis eigenspectrum / heatmap
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_cokurtosis(
        kt::MatNum,
        nx::AbstractVector = 1:isqrt(size(kt, 1));
        heatmap::Bool = false,
        reference::Bool = true,
        kwargs...
    ) -> Plot
    plot_cokurtosis(pr::HighOrderPrior, nx = nothing; heatmap, reference, kwargs...) -> Plot
    plot_cokurtosis(pr::HighOrderPrior, rd::ReturnsResult; heatmap, reference, kwargs...) -> Plot
    plot_cokurtosis(res::OptimisationResult[, rd::ReturnsResult]; reference, kwargs...) -> Plot
    plot_cokurtosis(pred::PredictionResult[, rd::ReturnsResult]; reference, kwargs...) -> Plot

Plot the eigenvalues of the cokurtosis matrix, ``N^2 \\times N^2``, of a [`HighOrderPrior`](@ref) as a bar chart sorted from the largest, or the matrix itself as a heatmap.

The bar chart marks the mean eigenvalue with a reference line. The heatmap carries no axis labels, so `nx` has no effect on either figure. A result arity passes `heatmap` through its extra keywords. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

The two figures take opposite sides of one rule, because one draws and the other computes. Under `heatmap = true` the figure **draws**. A non-investable asset keeps its rows and its columns, they are blank cells, and the colour limits are [`finite_symmetric_clim`](@ref), so one gap does not flatten the scale. Under the default the figure **computes**. `eigvals(Symmetric(kt))` refuses a `NaN`, so the prior arity reduces to the Investable Mask at its entry through [`investable_plot_view`](@ref), and the spectrum is that of the investable block, of length ``N_i^2`` for ``N_i`` investable assets. The bare `kt` arity computes on the matrix that the caller holds, and a gap in it raises rather than reduces.

# Mathematical definition

```math
\\begin{align}
\\bar{\\lambda} = \\frac{1}{N^2} \\sum_{j=1}^{N^2} \\lambda_j = \\frac{\\mathrm{tr}(\\mathbf{K})}{N^2}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{K}``: The cokurtosis matrix whose spectrum the figure draws.
  - ``\\lambda_j``: Eigenvalue ``j`` of ``\\mathbf{K}``.
  - ``\\bar{\\lambda}``: Height of the reference line, the mean eigenvalue.
  - $(math_dict[:N])

# Arguments

  - `kt`: Cokurtosis matrix (assets² × assets²).
  - `nx`: Asset names, or `nothing`. Neither figure draws them.
  - `heatmap`: Draw the matrix as a heatmap when `true`, and its spectrum otherwise. A heatmap is readable for a small universe only.
  - `reference`: Draw the mean eigenvalue when `true`.
  - `pr`: High order prior result. The figure reads its `kt`.
  - `rd`: Returns result.
  - `res`: Optimisation result. The figure reads the prior it was fitted on.
  - `pred`: A prediction. The figure reads the prior of its result.

# Validation

  - `pr.kt` is not `nothing`, else an `ArgumentError` is raised.
  - The prior of a result is a [`HighOrderPrior`](@ref), else an `ArgumentError` is raised.
  - Under the spectrum, every entry of `kt` is finite, else `eigvals` raises.

# Returns

  - `plt::Plots.Plot`: A bar chart with the reference line when `reference` is `true`, or a heatmap.

# Related

  - [`plot_coskewness`](@ref)
  - [`HighOrderPrior`](@ref)
"""
function plot_cokurtosis end

## ──────────────────────────────────────────────────────────────────────────────
## Portfolio dashboard (multi-panel composite)
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_portfolio_dashboard(
        res::OptimisationResult,
        rd::Pr_RR;
        ts = 1:size(rd.X, 1),
        nx = 1:size(rd.X, 2),
        r::BaseRM_VecBaseRM = Variance(),
        slv::Option{<:Slv_VecSlv} = nothing,
        compound::Bool = false,
        N::Option{<:Number} = nothing,
        delta::Number = 1e-6,
        marginal::Bool = false,
        percentage::Bool = true,
        alpha::Number = 0.05,
        kappa::Number = 0.3,
        rw = nothing,
        sca::Scalariser = SumScalariser(),
        kwargs...
    ) -> Plot

Plot one optimisation result as four panels in a 2 × 2 layout: its composition, its cumulative returns, its risk contributions and its drawdown.

 1. The composition, through [`plot_composition`](@ref).
 2. The cumulative returns, through [`plot_portfolio_cumulative_returns`](@ref).
 3. The risk contribution of each asset under `r`, through [`plot_risk_contribution`](@ref).
 4. The drawdown, through [`plot_drawdowns`](@ref), which draws its own fixed set of drawdown measures and does not read `r`.

The figure reads the weights and the fees of `res`, the returns of `rd`, and the names on the investable universe of the result. A returns result replaces `nx` and `ts` with its own when it carries them. Panel 3 reads a prior whole, so a measure with an unset moment, the default `Variance()` among them, takes the moment from the prior, and it reads the matrix of a returns result. The extra keywords reach the outer layout of 1200 × 800 pixels and no panel. The package extension `PortfolioOptimisersPlotsExt` implements the method, and it loads with `StatsPlots`.

Each panel takes its own side of the draw and compute rule, and this figure adds no rule of its own. The figure views the result at its Investable Mask, so every panel draws the investable universe of the result alone.

# Arguments

  - `res`: Optimisation result.
  - `rd`: Returns result or prior result.
  - `ts`: Labels of the time axis.
  - `nx`: Asset names.
  - `r`: Risk measure of panel 3, or a vector of them that `sca` combines.
  - `sca`: Scalariser that combines the measures in `r`. It has no effect when `r` is one measure. Pass `res.sca` to match the optimisation.
  - `slv`: Solver of the entropic and relativistic drawdown lines of panel 4.
  - `compound`: Compound the returns of panels 2 and 4 when `true`.
  - `N`: The count or the share of the bars of panels 1 and 3.
  - `delta`: Finite difference step of panel 3.
  - `marginal`: Draw the marginal contribution in panel 3 when `true`, and the component contribution otherwise.
  - `percentage`: Divide the contributions of panel 3 by their sum when `true`.
  - `alpha`: Significance level of the drawdown lines of panel 4.
  - `kappa`: Deformation parameter of the relativistic line of panel 4.
  - `rw`: Observation weights of the average drawdown line of panel 4, or `nothing`.

# Validation

  - The rules of [`plot_risk_contribution`](@ref) and of [`plot_drawdowns`](@ref).

# Returns

  - `plt::Plots.Plot`: A layout of four panels.

# Related

  - [`plot_composition`](@ref)
  - [`plot_portfolio_cumulative_returns`](@ref)
  - [`plot_risk_contribution`](@ref)
  - [`plot_drawdowns`](@ref)
"""
function plot_portfolio_dashboard end

## ──────────────────────────────────────────────────────────────────────────────
## Cross-validation dashboard (multi-panel composite)
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_cv_dashboard(
        mpred::MultiPeriodPredictionResult;
        N::Option{<:Number} = nothing,
        compound::Bool = false,
        kwargs...
    ) -> Plot

Plot a walk-forward path as four panels in a 2 × 2 layout: its fold compositions, its cumulative returns, its turnover and the stability of its weights.

 1. The composition of each fold as a stacked bar, through [`plot_composition`](@ref).
 2. The cumulative returns of the whole path as one line, through [`plot_portfolio_cumulative_returns`](@ref).
 3. The turnover from one fold to the next, through [`plot_turnover`](@ref).
 4. The distribution of each weight over the folds, through [`plot_weight_stability`](@ref).

The extra keywords reach the outer layout of 1200 × 800 pixels and no panel. The package extension `PortfolioOptimisersPlotsExt` implements the method, and it loads with `StatsPlots`.

Every panel reads a walk-forward prediction, whose folds set their Held Gaps to zero once in [`predict`](@ref) and whose weights are full length at every fold. So the four panels share one frame over the folds, and this figure adds no rule of its own.

# Arguments

  - `mpred`: A walk-forward path.
  - `N`: The count or the share of the boxes of panel 4. Panel 1 accepts it and does not read it.
  - `compound`: Compound the returns of panel 2 when `true`.

# Returns

  - `plt::Plots.Plot`: A layout of four panels.

# Related

  - [`MultiPeriodPredictionResult`](@ref)
  - [`plot_turnover`](@ref)
  - [`plot_weight_stability`](@ref)
"""
function plot_cv_dashboard end

## ──────────────────────────────────────────────────────────────────────────────
## Efficient frontier
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_efficient_frontier(
        res_vec::AbstractVector{<:OptimisationResult},
        pr::Pr_RR;
        x::BaseRM_VecBaseRM = Variance(),
        y::BaseRM_VecBaseRM = ExpectedReturn(),
        c::BaseRM_VecBaseRM = ExpectedReturnRiskRatio(; rk=x, rt=ArithmeticReturn(), rf=0),
        slv::Option{<:Slv_VecSlv} = nothing,
        fees::Option{<:Fees} = nothing,
        min_risk::Bool = true,
        max_score::Bool = true,
        factory::Bool = true,
        kwargs...
    ) -> Plot
    plot_efficient_frontier(res_vec::AbstractVector{<:OptimisationResult}, rd::ReturnsResult; kwargs...) -> Plot
    plot_efficient_frontier(w::VecVecNum, pr::Pr_RR; x, y, c, slv, fees, min_risk, max_score, factory, kwargs...) -> Plot
    plot_efficient_frontier(res::OptimisationResult, pr::Pr_RR; fees, kwargs...) -> Plot
    plot_efficient_frontier(res::OptimisationResult, rd::ReturnsResult; fees, kwargs...) -> Plot

Plot a collection of portfolios as a line ordered by risk, coloured by a score, to trace an efficient frontier.

Each point is [`expected_risk`](@ref) of `x` and `y` for one portfolio, and its colour is [`expected_risk`](@ref) of `c`. The default colour is the ratio of the arithmetic return to the risk of `x`. Under the default `Variance()` that is the return over the variance, not a Sharpe ratio. The line joins the points in ascending order of `x`. A blue star marks the portfolio of least `x`, and a red star the portfolio of largest `c`. When `factory` is `true`, [`factory`](@ref) binds each measure to the prior and to `slv` first. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

A [`ReturnsResult`](@ref) in the second position is a tag and not data. Each result is scored against the prior it was fitted on, on its own investable universe, as [`plot_measures`](@ref) scores a vector of results. A prior in that position scores every portfolio against that one prior.

Each axis takes one measure or a **vector** of them, and a vector becomes the one number that the axis plots. As in [`plot_measures`](@ref) the figure names no scalariser, so a **bare vector on an axis takes [`SumScalariser`](@ref)**, each element weighted by its own `settings.scale`. A scalariser reaches an axis only through the `sca` field of a ratio wrapper, so the figure cannot apply a [`MaxScalariser`](@ref) to a plain vector of risks. The order of the line and the two stars read the scalarised number, so a vector `x` orders the frontier by the aggregate and not by one measure. The axis label joins the element measures with `+`, whichever scalariser formed the number.

# Arguments

  - `res_vec`: Optimisation results.
  - `w`: Weight vectors, one per portfolio.
  - `res`: One optimisation result, drawn as a frontier of one point.
  - `pr`: Prior result or returns result that the portfolios are scored against.
  - `rd`: Returns result, the tag that asks for the prior of each result.
  - `x`: Measure of the horizontal axis, or a vector of them.
  - `y`: Measure of the vertical axis, or a vector of them.
  - `c`: Measure of the colour, or a vector of them.
  - `slv`: Solver that [`factory`](@ref) binds to each measure.
  - `fees`: Transaction fees, or `nothing`.
  - `min_risk`: Mark the portfolio of least `x` when `true`.
  - `max_score`: Mark the portfolio of largest `c` when `true`.
  - `factory`: Bind each measure through [`factory`](@ref) before the figure scores it.

# Validation

  - The rules of [`expected_risk`](@ref) for each measure.

# Returns

  - `plt::Plots.Plot`: A line plot with markers, and the stars that `min_risk` and `max_score` ask for.

# Related

  - [`plot_measures`](@ref)
  - [`expected_risk`](@ref)
  - [`Scalariser`](@ref)
"""
function plot_efficient_frontier end

## ──────────────────────────────────────────────────────────────────────────────
## Performance summary
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_performance_summary(ps::PerformanceSummaryResult; kwargs...) -> Plot
    plot_performance_summary(
        w::ArrNum,
        X::MatNum,
        fees::Option{<:Fees} = nothing;
        periods_per_year::Number = 252,
        alpha::Number = 0.05,
        compound::Bool = false,
        benchmark::Option{<:VecNum} = nothing,
        kwargs...
    ) -> Plot
    plot_performance_summary(ret::VecNum; periods_per_year, alpha, compound, benchmark, kwargs...) -> Plot
    plot_performance_summary(w, rd::ReturnsResult, fees = nothing; periods_per_year, alpha, compound, benchmark, kwargs...) -> Plot
    plot_performance_summary(res::OptimisationResult, rd::ReturnsResult; periods_per_year, alpha, compound, benchmark, kwargs...) -> Plot
    plot_performance_summary(pred::PredRes_MultiPredRes; periods_per_year, alpha, compound, benchmark, kwargs...) -> Plot

Plot the performance statistics of a portfolio as a bar chart of eleven bars, blue at a value of zero or more and red below it.

Every method but the first computes a [`PerformanceSummaryResult`](@ref) with [`performance_summary`](@ref) and draws it, so it takes every arity that verb takes. The verb defines and validates the statistics, so a caller who wants the numbers and not the bars calls it directly and needs no plotting package. The figure draws the bars in the order below, seven of them in percent. The last four bars are `NaN`, drawn empty, when the summary has no benchmark, and the turnover bar is `NaN` on every arity that holds no path, which is every arity but a prediction. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

A gapped panel reaches this figure only through the returns that the caller gives it. A prediction arity is finite already, because the fold set the Held Gap to zero once in [`predict`](@ref). A `w, X` or `w, rd` arity computes on the matrix that the caller holds and checks nothing, and `X * w` is `NaN` at a gap even where the weight is zero. Score a gapped panel through `predict(res, rd)` and plot the prediction.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{v} = \\left(100 R_a,\\ 100 \\sigma_a,\\ \\mathrm{SR},\\ \\mathrm{So},\\ \\mathrm{Ca},\\ 100\\, \\mathrm{MDD},\\ 100\\, \\mathrm{CVaR},\\ 100\\, \\mathrm{ER},\\ 100\\, \\mathrm{TE},\\ \\mathrm{IR},\\ 100\\, \\mathrm{TO}\\right)\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{v}``: The heights of the eleven bars, in order.
  - ``R_a``, ``\\sigma_a``: Annualised return and annualised volatility, the fields `ann_return` and `ann_volatility`.
  - ``\\mathrm{SR}``, ``\\mathrm{So}``, ``\\mathrm{Ca}``: Sharpe, Sortino and Calmar ratios, the fields `sharpe`, `sortino` and `calmar`.
  - ``\\mathrm{MDD}``: Maximum drawdown, the field `max_drawdown`, which is at most zero.
  - ``\\mathrm{CVaR}``: The field `cvar`, the negated conditional value at risk, so its bar is negative for a loss.
  - ``\\mathrm{ER}``, ``\\mathrm{TE}``, ``\\mathrm{IR}``: Excess return, tracking error and information ratio against the benchmark, the fields `excess_ret`, `tracking_error` and `information_ratio`.
  - ``\\mathrm{TO}``: Turnover, the field `turnover`.

# Arguments

  - `ps`: The summary to draw.
  - `ret`: Net portfolio returns.
  - `w`: Portfolio weights.
  - `X`: Asset returns matrix (observations × assets).
  - `fees`: Transaction fees, or `nothing`.
  - `periods_per_year`: Number of periods in one year, which annualises the statistics.
  - `alpha`: Tail probability of the conditional value at risk.
  - `compound`: Compound the returns of the maximum drawdown when `true`.
  - `benchmark`: Benchmark return series of the three excess statistics, or `nothing`.
  - `res`: Optimisation result.
  - `rd`: Returns result.
  - `pred`: A prediction or a walk-forward path, whose held path gives the turnover bar.

# Validation

  - The rules of [`performance_summary`](@ref): `0 < alpha < 1` and `periods_per_year > 0`, else a `DomainError` is raised, and a `benchmark` of the length of the returns, else a `DimensionMismatch` is raised.

# Returns

  - `plt::Plots.Plot`: A bar chart of eleven bars.

# Related

  - [`performance_summary`](@ref)
  - [`PerformanceSummaryResult`](@ref)
  - [`calc_net_returns`](@ref)
"""
function plot_performance_summary end

## ──────────────────────────────────────────────────────────────────────────────
## Rolling drawdown evolution
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_rolling_drawdowns(
        w::ArrNum,
        X::MatNum,
        fees::Option{<:Fees} = nothing;
        ts::AbstractVector = 1:size(X, 1),
        rolling::Integer = 0,
        compound::Bool = false,
        kwargs...
    ) -> Plot
    plot_rolling_drawdowns(w, rd::ReturnsResult, fees = nothing; rolling, compound, kwargs...) -> Plot
    plot_rolling_drawdowns(res::OptimisationResult, rd::ReturnsResult; rolling, compound, kwargs...) -> Plot
    plot_rolling_drawdowns(pred::PredictionResult; rolling, compound, kwargs...) -> Plot
    plot_rolling_drawdowns(mpred::MultiPeriodPredictionResult; rolling, compound, kwargs...) -> Plot
    plot_rolling_drawdowns(ret::VecNum; ts::AbstractVector, rolling, compound, kwargs...) -> Plot

Plot the maximum drawdown of the portfolio over a rolling window, in percent, against the last date of each window.

Each window is drawn down from its own entry value. The running peak starts at the capital at the start of the window, not at the capital at inception, so a window that only rises reads zero. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

A gapped panel reaches this figure only through the returns that the caller gives it. A prediction arity is finite already, because the fold set the Held Gap to zero once in [`predict`](@ref). A `w, X` or `w, rd` arity computes on the matrix that the caller holds and checks nothing, and `X * w` is `NaN` at a gap even where the weight is zero. Score a gapped panel through `predict(res, rd)` and plot the prediction.

# Mathematical definition

```math
\\begin{align}
W &= \\begin{cases} \\lceil \\sqrt{T} \\rceil & \\text{rolling} = 0\\,, \\\\ \\text{rolling} & \\text{otherwise}\\,, \\end{cases} \\\\
D_t &= 100 \\min_{s = t-W+1, \\ldots, t} d_{t,s}\\,, \\quad t = W, \\ldots, T\\,.
\\end{align}
```

Where:

  - $(math_dict[:W_roll])
  - $(math_dict[:T])
  - ``d_{t,s}``: [`drawdowns`](@ref) at observation ``s`` of the returns ``r_{t-W+1}, \\ldots, r_t`` of the window that ends at ``t``, absolute or relative as `compound` selects.
  - ``D_t``: The value that the figure draws at observation ``t``, the maximum drawdown of that window in percent, at most zero.

# Arguments

  - `w`: Portfolio weights.
  - `X`: Asset returns matrix (observations × assets).
  - `fees`: Transaction fees, or `nothing`.
  - `ts`: Labels of the time axis. A returns result or a prediction gives its own.
  - `rolling`: Length of the window, or `0` for the default.
  - `compound`: Compound the returns and take relative drawdowns when `true`.
  - `res`: Optimisation result. The figure reads its weights and its fees on its investable universe.
  - `pred`, `mpred`: A prediction or a walk-forward path, whose return series the figure draws.
  - `ret`: Net portfolio returns.

# Validation

  - `rolling >= 0`, else a `DomainError` is raised.
  - ``W \\leq T``, else a `DomainError` is raised.

# Returns

  - `plt::Plots.Plot`: A line plot of ``T - W + 1`` points.

# Related

  - [`drawdowns`](@ref)
  - [`plot_drawdowns`](@ref)
  - [`plot_rolling_measure`](@ref)
"""
function plot_rolling_drawdowns end

## ────────────────────────────────────────────────────────────────────────────
## Cross-sectional regression diagnostics
## ────────────────────────────────────────────────────────────────────────────
"""
    plot_cs_regression_r2(csfm::CrossSectionalFactorModel; kwargs...) -> Plot
    plot_cs_regression_r2(pr::AbstractPriorResult; kwargs...) -> Plot

Plot the weighted cross-sectional coefficient of determination of every cross-sectional fit.

The figure draws what [`cs_regression_r2`](@ref) returns and computes nothing of its own. A prior result gives its `rr` field, the block that the fit produced. The horizontal axis counts the fits of the lagged sample, so position ``j`` is observation ``j`` plus the lag of the block. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `pr`: A prior result whose `rr` is such a block.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - `pr.rr` is not `nothing`, else [`assert_prior_regression`](@ref) raises an `IsNothingError`.
  - `pr.rr` is a [`CrossSectionalFactorModel`](@ref), else the verb raises a `MethodError`: a time-series [`Regression`](@ref) carries no exposure history.
  - The rules of [`cs_regression_r2`](@ref).

# Returns

  - `plt::Plots.Plot`: A line plot with one point per fit of the lagged sample.

# Related

  - [`cs_regression_r2`](@ref)
  - [`plot_cs_regression_adjusted_r2`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function plot_cs_regression_r2 end
"""
    plot_cs_regression_adjusted_r2(csfm::CrossSectionalFactorModel; kwargs...) -> Plot
    plot_cs_regression_adjusted_r2(pr::AbstractPriorResult; kwargs...) -> Plot

Plot the adjusted cross-sectional coefficient of determination of every cross-sectional fit.

The figure draws what [`cs_regression_adjusted_r2`](@ref) returns and computes nothing of its own. A prior result gives its `rr` field, the block that the fit produced. The horizontal axis counts the fits of the lagged sample, so position ``j`` is observation ``j`` plus the lag of the block. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `pr`: A prior result whose `rr` is such a block.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - `pr.rr` is not `nothing`, else [`assert_prior_regression`](@ref) raises an `IsNothingError`.
  - `pr.rr` is a [`CrossSectionalFactorModel`](@ref), else the verb raises a `MethodError`: a time-series [`Regression`](@ref) carries no exposure history.
  - The rules of [`cs_regression_adjusted_r2`](@ref).

# Returns

  - `plt::Plots.Plot`: A line plot with one point per fit of the lagged sample.

# Related

  - [`cs_regression_adjusted_r2`](@ref)
  - [`plot_cs_regression_r2`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function plot_cs_regression_adjusted_r2 end
"""
    plot_cs_regression_aic(csfm::CrossSectionalFactorModel; kwargs...) -> Plot
    plot_cs_regression_aic(pr::AbstractPriorResult; kwargs...) -> Plot

Plot the Akaike information criterion of every cross-sectional fit.

The figure draws what [`cs_regression_aic`](@ref) returns and computes nothing of its own. A prior result gives its `rr` field, the block that the fit produced. The horizontal axis counts the fits of the lagged sample, so position ``j`` is observation ``j`` plus the lag of the block. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `pr`: A prior result whose `rr` is such a block.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - `pr.rr` is not `nothing`, else [`assert_prior_regression`](@ref) raises an `IsNothingError`.
  - `pr.rr` is a [`CrossSectionalFactorModel`](@ref), else the verb raises a `MethodError`: a time-series [`Regression`](@ref) carries no exposure history.
  - The rules of [`cs_regression_aic`](@ref).

# Returns

  - `plt::Plots.Plot`: A line plot with one point per fit of the lagged sample.

# Related

  - [`cs_regression_aic`](@ref)
  - [`plot_cs_regression_bic`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function plot_cs_regression_aic end
"""
    plot_cs_regression_bic(csfm::CrossSectionalFactorModel; kwargs...) -> Plot
    plot_cs_regression_bic(pr::AbstractPriorResult; kwargs...) -> Plot

Plot the Bayesian information criterion of every cross-sectional fit.

The figure draws what [`cs_regression_bic`](@ref) returns and computes nothing of its own. A prior result gives its `rr` field, the block that the fit produced. The horizontal axis counts the fits of the lagged sample, so position ``j`` is observation ``j`` plus the lag of the block. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `pr`: A prior result whose `rr` is such a block.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - `pr.rr` is not `nothing`, else [`assert_prior_regression`](@ref) raises an `IsNothingError`.
  - `pr.rr` is a [`CrossSectionalFactorModel`](@ref), else the verb raises a `MethodError`: a time-series [`Regression`](@ref) carries no exposure history.
  - The rules of [`cs_regression_bic`](@ref).

# Returns

  - `plt::Plots.Plot`: A line plot with one point per fit of the lagged sample.

# Related

  - [`cs_regression_bic`](@ref)
  - [`plot_cs_regression_aic`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function plot_cs_regression_bic end
"""
    plot_cs_regression_t_stats(
        csfm::CrossSectionalFactorModel;
        nf::Option{<:AbstractVector} = nothing,
        kwargs...
    ) -> Plot
    plot_cs_regression_t_stats(
        pr::AbstractPriorResult;
        nf::Option{<:AbstractVector} = nothing,
        kwargs...
    ) -> Plot

Plot the t-statistic of every factor return, one series per factor.

The figure draws what [`cs_regression_t_stats`](@ref) returns and computes nothing of its own. The series are labelled by the factor names of the axis of the answer, which [`cs_diagnostic_factor_names`](@ref) resolves, so a block that carries a family re-basis is labelled on the reduced axis. The horizontal axis counts the fits of the lagged sample. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `pr`: A prior result whose `rr` is such a block.
  - `nf`: Factor names of the answer's axis. `nothing` reads them off the block, and falls back to the position of the factor.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - `pr.rr` is not `nothing`, else [`assert_prior_regression`](@ref) raises an `IsNothingError`.
  - `pr.rr` is a [`CrossSectionalFactorModel`](@ref), else the verb raises a `MethodError`: a time-series [`Regression`](@ref) carries no exposure history.
  - The rules of [`cs_regression_t_stats`](@ref).

# Returns

  - `plt::Plots.Plot`: A line plot with one series per factor, and one point per fit of the lagged sample in each.

# Related

  - [`cs_regression_t_stats`](@ref)
  - [`plot_cs_regression_t_stat_exceedance_rate`](@ref)
  - [`cs_diagnostic_factor_names`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function plot_cs_regression_t_stats end
"""
    plot_cs_regression_t_stat_exceedance_rate(
        csfm::CrossSectionalFactorModel;
        nf::Option{<:AbstractVector} = nothing,
        threshold::Number = 2,
        kwargs...
    ) -> Plot
    plot_cs_regression_t_stat_exceedance_rate(
        pr::AbstractPriorResult;
        nf::Option{<:AbstractVector} = nothing,
        threshold::Number = 2,
        kwargs...
    ) -> Plot

Plot the fraction of observations at which the t-statistic of each factor exceeds a threshold, one bar per factor.

The bars are what [`cs_regression_t_stat_exceedance_rate`](@ref) returns. A dashed line marks the rate that a factor of no explanatory power reaches, the one quantity the figure computes. The factors are labelled on the axis of the answer, as [`plot_cs_regression_t_stats`](@ref) labels them. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

# Mathematical definition

```math
\\begin{align}
q_0 = 2 \\Phi(-c)\\,.
\\end{align}
```

Where:

  - ``c``: The threshold, `threshold`.
  - ``\\Phi``: Distribution function of the standard normal law.
  - ``q_0``: Height of the reference line, the probability that a standard normal t-statistic exceeds ``c`` in magnitude. It is about `0.0455` at the default threshold.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `pr`: A prior result whose `rr` is such a block.
  - `nf`: Factor names of the answer's axis. `nothing` reads them off the block, and falls back to the position of the factor.
  - `threshold`: Absolute t-statistic above which an observation counts as significant.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - `pr.rr` is not `nothing`, else [`assert_prior_regression`](@ref) raises an `IsNothingError`.
  - `pr.rr` is a [`CrossSectionalFactorModel`](@ref), else the verb raises a `MethodError`: a time-series [`Regression`](@ref) carries no exposure history.
  - The rules of [`cs_regression_t_stat_exceedance_rate`](@ref).

# Returns

  - `plt::Plots.Plot`: A bar chart with one bar per factor, and the reference line.

# Related

  - [`cs_regression_t_stat_exceedance_rate`](@ref)
  - [`plot_cs_regression_t_stats`](@ref)
  - [`cs_diagnostic_factor_names`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function plot_cs_regression_t_stat_exceedance_rate end
"""
    plot_exposure_vif(
        csfm::CrossSectionalFactorModel;
        nf::Option{<:AbstractVector} = nothing,
        kwargs...
    ) -> Plot
    plot_exposure_vif(
        pr::AbstractPriorResult;
        nf::Option{<:AbstractVector} = nothing,
        kwargs...
    ) -> Plot

Plot the variance inflation factor of every factor, one series per factor.

The figure draws what [`exposure_vif`](@ref) returns and computes nothing of its own. A dashed line marks `1`, the value that an orthogonal design reaches. The series are labelled on the axis of the answer, as [`plot_cs_regression_t_stats`](@ref) labels them. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `pr`: A prior result whose `rr` is such a block.
  - `nf`: Factor names of the answer's axis. `nothing` reads them off the block, and falls back to the position of the factor.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - `pr.rr` is not `nothing`, else [`assert_prior_regression`](@ref) raises an `IsNothingError`.
  - `pr.rr` is a [`CrossSectionalFactorModel`](@ref), else the verb raises a `MethodError`: a time-series [`Regression`](@ref) carries no exposure history.
  - The rules of [`exposure_vif`](@ref).

# Returns

  - `plt::Plots.Plot`: A line plot with one series per factor, and the reference line.

# Related

  - [`exposure_vif`](@ref)
  - [`plot_exposure_condition_number`](@ref)
  - [`cs_diagnostic_factor_names`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function plot_exposure_vif end
"""
    plot_exposure_condition_number(csfm::CrossSectionalFactorModel; kwargs...) -> Plot
    plot_exposure_condition_number(pr::AbstractPriorResult; kwargs...) -> Plot

Plot the condition number of the cross-sectional design of every observation. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

The figure draws what [`exposure_condition_number`](@ref) returns and computes nothing of its own. The vertical axis is logarithmic, because the condition number of a nearly collinear design is many orders of magnitude above that of a well conditioned one.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `pr`: A prior result whose `rr` is such a block.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - `pr.rr` is not `nothing`, else [`assert_prior_regression`](@ref) raises an `IsNothingError`.
  - `pr.rr` is a [`CrossSectionalFactorModel`](@ref), else the verb raises a `MethodError`: a time-series [`Regression`](@ref) carries no exposure history.
  - The rules of [`exposure_condition_number`](@ref).

# Returns

  - `plt::Plots.Plot`: A line plot on a logarithmic vertical axis.

# Related

  - [`exposure_condition_number`](@ref)
  - [`plot_exposure_vif`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function plot_exposure_condition_number end
"""
    plot_exposure_correlation(
        csfm::CrossSectionalFactorModel;
        nf::Option{<:AbstractVector} = nothing,
        weighting = BenchmarkWeightMetric(),
        kwargs...
    ) -> Plot
    plot_exposure_correlation(
        pr::AbstractPriorResult;
        nf::Option{<:AbstractVector} = nothing,
        weighting = BenchmarkWeightMetric(),
        kwargs...
    ) -> Plot

Plot the time-averaged correlation between every pair of factor exposures as a heatmap.

The figure draws what [`exposure_correlation`](@ref) returns and computes nothing of its own. The colour limits are fixed at `(-1, 1)`, so two figures of two models are read against the same scale. A pair that the verb cannot correlate is `NaN`, and its cell is blank. The exposure figures answer on the **raw** factor axis, because they read the exposure history as the panel wrote it, so the labels come from the names of the block and not from [`cs_diagnostic_factor_names`](@ref). The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `pr`: A prior result whose `rr` is such a block.
  - `nf`: Factor names of the raw factor axis. `nothing` reads them off the block, and falls back to the position of the factor.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref). It names the weight history the block is read with.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - `pr.rr` is not `nothing`, else [`assert_prior_regression`](@ref) raises an `IsNothingError`.
  - `pr.rr` is a [`CrossSectionalFactorModel`](@ref), else the verb raises a `MethodError`: a time-series [`Regression`](@ref) carries no exposure history.
  - The rules of [`exposure_correlation`](@ref), which raises an `IsNothingError` when the block carries no exposure history or no weight history of the kind `weighting` names.

# Returns

  - `plt::Plots.Plot`: A heatmap of factors by factors.

# Related

  - [`exposure_correlation`](@ref)
  - [`plot_exposure_vif`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function plot_exposure_correlation end
"""
    plot_cumulative_exposure_ic(
        csfm::CrossSectionalFactorModel;
        nf::Option{<:AbstractVector} = nothing,
        rank::Bool = true,
        reduced::Bool = false,
        ties::Symbol = :average,
        kwargs...
    ) -> Plot
    plot_cumulative_exposure_ic(
        pr::AbstractPriorResult;
        nf::Option{<:AbstractVector} = nothing,
        rank::Bool = true,
        reduced::Bool = false,
        ties::Symbol = :average,
        kwargs...
    ) -> Plot

Plot the running sum of the information coefficient of every factor exposure, one series per factor.

The figure draws the running sum of what [`exposure_ic`](@ref) returns at a horizon of one observation, and computes nothing else of its own. An observation whose coefficient is not finite adds nothing to the sum, so one missing cross-section breaks no series. A series that rises through the sample is an exposure that forecast the return, a flat series is one that carried no forecast, and a falling series is one whose forecast had the opposite sign. The series are labelled on the raw factor axis, and on the reduced axis under `reduced`, as [`plot_cs_regression_t_stats`](@ref) labels them. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

A risk factor with an information coefficient near zero is not a bad risk factor. A risk factor is built to explain the covariance and not to predict the mean, so read [`plot_exposure_stability`](@ref) and the variance that the factor contributes before you judge one.

# Mathematical definition

```math
\\begin{align}
C_{t,k} = \\sum_{s=1}^{t} \\mathbb{1}\\left[\\mathrm{IC}_{s,k} \\text{ is finite}\\right] \\mathrm{IC}_{s,k}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{IC}_{s,k}``: Information coefficient of factor ``k`` at observation ``s``, as [`exposure_ic`](@ref) returns it.
  - ``C_{t,k}``: Point ``t`` of the series of factor ``k``.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `pr`: A prior result whose `rr` is such a block.
  - `nf`: Factor names of the axis of the answer. `nothing` reads them off the block, and falls back to the position of the factor.
  - `rank`: Take the rank correlation when `true`, and the weighted correlation otherwise.
  - `reduced`: Map the exposures through the family re-basis of the block before the correlation.
  - $(arg_dict[:cs_ties]) The weighted correlation reads no rank, so it ignores `ties`.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - `pr.rr` is not `nothing`, else [`assert_prior_regression`](@ref) raises an `IsNothingError`.
  - `pr.rr` is a [`CrossSectionalFactorModel`](@ref), else the verb raises a `MethodError`: a time-series [`Regression`](@ref) carries no exposure history.
  - The rules of [`exposure_ic`](@ref) at a horizon of one observation.

# Returns

  - `plt::Plots.Plot`: A line plot with one series per factor.

# Related

  - [`exposure_ic`](@ref)
  - [`exposure_ic_summary`](@ref)
  - [`plot_exposure_stability`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function plot_cumulative_exposure_ic end
"""
    plot_exposure_distribution(
        csfm::CrossSectionalFactorModel;
        factor::Integer = 1,
        observation::Option{<:Integer} = nothing,
        nf::Option{<:AbstractVector} = nothing,
        kwargs...
    ) -> Plot
    plot_exposure_distribution(
        pr::AbstractPriorResult;
        factor::Integer = 1,
        observation::Option{<:Integer} = nothing,
        nf::Option{<:AbstractVector} = nothing,
        kwargs...
    ) -> Plot

Plot the cross-sectional distribution of one factor exposure as a histogram.

The figure draws one slice of the exposure history of the block and computes nothing of its own. `observation` selects one observation, and `nothing` pools every observation into one figure. The entries that are not finite are dropped, so the count of the figure is the coverage of the factor. The factor is named on the raw factor axis. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `pr`: A prior result whose `rr` is such a block.
  - `factor`: Position of the factor on the raw factor axis of the block.
  - `observation`: Position of the observation, or `nothing` to pool every observation.
  - `nf`: Factor names of the raw axis. `nothing` reads them off the block, and falls back to the position of the factor.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - `pr.rr` is not `nothing`, else [`assert_prior_regression`](@ref) raises an `IsNothingError`.
  - `pr.rr` is a [`CrossSectionalFactorModel`](@ref), else the verb raises a `MethodError`: a time-series [`Regression`](@ref) carries no exposure history.
  - The block carries an exposure history, else an `IsNothingError` is raised.
  - `factor` and `observation` index the history, else a `BoundsError` is raised.

# Returns

  - `plt::Plots.Plot`: A histogram of the finite exposures of the slice.

# Related

  - [`exposure_dispersion`](@ref)
  - [`exposure_coverage`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function plot_exposure_distribution end
"""
    plot_exposure_dispersion(
        csfm::CrossSectionalFactorModel;
        nf::Option{<:AbstractVector} = nothing,
        weighting = BenchmarkWeightMetric(),
        kwargs...
    ) -> Plot
    plot_exposure_dispersion(
        pr::AbstractPriorResult;
        nf::Option{<:AbstractVector} = nothing,
        weighting = BenchmarkWeightMetric(),
        kwargs...
    ) -> Plot

Plot the weighted cross-sectional standard deviation of every factor exposure, one series per factor.

The figure draws what [`exposure_dispersion`](@ref) returns and computes nothing of its own. Read the series and not its level, because the level follows the standardisation the exposures were built under, and the series shows the observation at which the panel changed. The series are labelled on the raw factor axis, as [`plot_exposure_correlation`](@ref) labels them. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `pr`: A prior result whose `rr` is such a block.
  - `nf`: Factor names of the raw factor axis. `nothing` reads them off the block, and falls back to the position of the factor.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref). It names the weight history the block is read with.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - `pr.rr` is not `nothing`, else [`assert_prior_regression`](@ref) raises an `IsNothingError`.
  - `pr.rr` is a [`CrossSectionalFactorModel`](@ref), else the verb raises a `MethodError`: a time-series [`Regression`](@ref) carries no exposure history.
  - The rules of [`exposure_dispersion`](@ref), which raises an `IsNothingError` when the block carries no exposure history or no weight history of the kind `weighting` names.

# Returns

  - `plt::Plots.Plot`: A line plot with one series per factor.

# Related

  - [`exposure_dispersion`](@ref)
  - [`plot_exposure_distribution`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function plot_exposure_dispersion end
"""
    plot_exposure_stability(
        csfm::CrossSectionalFactorModel;
        nf::Option{<:AbstractVector} = nothing,
        step::Integer = 21,
        weighting = BenchmarkWeightMetric(),
        kwargs...
    ) -> Plot
    plot_exposure_stability(
        pr::AbstractPriorResult;
        nf::Option{<:AbstractVector} = nothing,
        step::Integer = 21,
        weighting = BenchmarkWeightMetric(),
        kwargs...
    ) -> Plot

Plot the stability of every factor exposure, one series per factor.

The figure draws what [`exposure_stability`](@ref) returns and computes nothing of its own. The stability is a weighted correlation of the exposures at two cross-sections `step` observations apart, so a dashed line marks `1`, the value of two cross-sections that are an exact positive affine map of each other. An exposure that keeps the order of the assets and changes their spacing stays below the line. The series are labelled on the raw factor axis, as [`plot_exposure_correlation`](@ref) labels them. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `pr`: A prior result whose `rr` is such a block.
  - `nf`: Factor names of the raw factor axis. `nothing` reads them off the block, and falls back to the position of the factor.
  - `step`: Number of observations between the two cross-sections.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref). It names the weight history the block is read with.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - `pr.rr` is not `nothing`, else [`assert_prior_regression`](@ref) raises an `IsNothingError`.
  - `pr.rr` is a [`CrossSectionalFactorModel`](@ref), else the verb raises a `MethodError`: a time-series [`Regression`](@ref) carries no exposure history.
  - The rules of [`exposure_stability`](@ref): `step >= 1`, else a `DomainError` is raised, and fewer than `step` observations raise a `DimensionMismatch`.

# Returns

  - `plt::Plots.Plot`: A line plot with one series per factor, and the reference line.

# Related

  - [`exposure_stability`](@ref)
  - [`plot_exposure_dispersion`](@ref)
  - [`plot_cumulative_exposure_ic`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function plot_exposure_stability end

## ────────────────────────────────────────────────────────────────────────────
## Cross-sectional idiosyncratic diagnostics
## ────────────────────────────────────────────────────────────────────────────
"""
    plot_idio_calibration(csfm::CrossSectionalFactorModel; kwargs...) -> Plot
    plot_idio_calibration(pr::AbstractPriorResult; kwargs...) -> Plot

Plot the cross-sectional standard deviation of the standardised idiosyncratic returns against the observation axis. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

The figure draws what [`idio_calibration`](@ref) returns and computes nothing of its own. A dashed line marks the Gaussian reference of `1`. A series that sits above the line is a fit whose specific risk is too small, and one that sits below it is a fit whose specific risk is too large.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `pr`: A prior result whose `rr` is such a block.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - `pr.rr` is not `nothing`, else [`assert_prior_regression`](@ref) raises an `IsNothingError`.
  - `pr.rr` is a [`CrossSectionalFactorModel`](@ref), else the verb raises a `MethodError`: a time-series [`Regression`](@ref) carries no exposure history.
  - The rules of [`idio_calibration`](@ref), which raises an `IsNothingError` when the block carries no residuals or no predicted specific variances.

# Returns

  - `plt::Plots.Plot`: A line plot with the reference line.

# Related

  - [`idio_calibration`](@ref)
  - [`plot_idio_tail_rate`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function plot_idio_calibration end
"""
    plot_idio_tail_rate(
        csfm::CrossSectionalFactorModel;
        threshold::Real = 3,
        kwargs...
    ) -> Plot
    plot_idio_tail_rate(
        pr::AbstractPriorResult;
        threshold::Real = 3,
        kwargs...
    ) -> Plot

Plot the share of assets whose standardised idiosyncratic return exceeds a threshold, against the observation axis. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

The figure draws what [`idio_tail_rate`](@ref) returns and computes nothing of its own. A dashed line marks the Gaussian reference ``2 \\Phi(-c)``, where ``c`` is `threshold` and ``\\Phi`` the distribution function of the standard normal law, as [`plot_cs_regression_t_stat_exceedance_rate`](@ref) defines it. It is about `0.0027` at the default threshold. A series above the line is a fit whose standardised returns carry heavier tails than the normal law implies, which is ordinary for an equity universe.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `pr`: A prior result whose `rr` is such a block.
  - `threshold`: Absolute standardised return above which an asset enters the rate.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - `pr.rr` is not `nothing`, else [`assert_prior_regression`](@ref) raises an `IsNothingError`.
  - `pr.rr` is a [`CrossSectionalFactorModel`](@ref), else the verb raises a `MethodError`: a time-series [`Regression`](@ref) carries no exposure history.
  - The rules of [`idio_tail_rate`](@ref), which raises an `IsNothingError` when the block carries no residuals or no predicted specific variances.

# Returns

  - `plt::Plots.Plot`: A line plot with the reference line.

# Related

  - [`idio_tail_rate`](@ref)
  - [`plot_idio_kurtosis`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function plot_idio_tail_rate end
"""
    plot_idio_kurtosis(csfm::CrossSectionalFactorModel; kwargs...) -> Plot
    plot_idio_kurtosis(pr::AbstractPriorResult; kwargs...) -> Plot

Plot the cross-sectional excess kurtosis of the standardised idiosyncratic returns against the observation axis. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

The figure draws what [`idio_kurtosis`](@ref) returns and computes nothing of its own. A dashed line marks the Gaussian reference of `0`. A positive series is a cross-section whose tails are heavier than the normal law implies.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `pr`: A prior result whose `rr` is such a block.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - `pr.rr` is not `nothing`, else [`assert_prior_regression`](@ref) raises an `IsNothingError`.
  - `pr.rr` is a [`CrossSectionalFactorModel`](@ref), else the verb raises a `MethodError`: a time-series [`Regression`](@ref) carries no exposure history.
  - The rules of [`idio_kurtosis`](@ref), which raises an `IsNothingError` when the block carries no residuals or no predicted specific variances.

# Returns

  - `plt::Plots.Plot`: A line plot with the reference line.

# Related

  - [`idio_kurtosis`](@ref)
  - [`plot_idio_skewness`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function plot_idio_kurtosis end
"""
    plot_idio_skewness(csfm::CrossSectionalFactorModel; kwargs...) -> Plot
    plot_idio_skewness(pr::AbstractPriorResult; kwargs...) -> Plot

Plot the cross-sectional skewness of the standardised idiosyncratic returns against the observation axis. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

The figure draws what [`idio_skewness`](@ref) returns and computes nothing of its own. A dashed line marks the Gaussian reference of `0`. A series that stays on one side of the line is a residual that carries a direction the factors did not take.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `pr`: A prior result whose `rr` is such a block.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - `pr.rr` is not `nothing`, else [`assert_prior_regression`](@ref) raises an `IsNothingError`.
  - `pr.rr` is a [`CrossSectionalFactorModel`](@ref), else the verb raises a `MethodError`: a time-series [`Regression`](@ref) carries no exposure history.
  - The rules of [`idio_skewness`](@ref), which raises an `IsNothingError` when the block carries no residuals or no predicted specific variances.

# Returns

  - `plt::Plots.Plot`: A line plot with the reference line.

# Related

  - [`idio_skewness`](@ref)
  - [`plot_idio_kurtosis`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function plot_idio_skewness end
"""
    plot_idio_vol_ic(csfm::CrossSectionalFactorModel; kwargs...) -> Plot
    plot_idio_vol_ic(pr::AbstractPriorResult; kwargs...) -> Plot

Plot the information coefficient of the predicted idiosyncratic volatility against the observation axis. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

The figure draws what [`idio_vol_ic`](@ref) returns and computes nothing of its own. It carries no reference line, because the series has no Gaussian reference, and a caller reads its level and its sign. A series that stays high is a fit that ranks specific risk across the assets well. The volatility predicted at one observation is scored against the return of the next, so the horizontal axis starts at the second observation.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `pr`: A prior result whose `rr` is such a block.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - `pr.rr` is not `nothing`, else [`assert_prior_regression`](@ref) raises an `IsNothingError`.
  - `pr.rr` is a [`CrossSectionalFactorModel`](@ref), else the verb raises a `MethodError`: a time-series [`Regression`](@ref) carries no exposure history.
  - The rules of [`idio_vol_ic`](@ref), which needs at least two observations.

# Returns

  - `plt::Plots.Plot`: A line plot with one point per observation from the second.

# Related

  - [`idio_vol_ic`](@ref)
  - [`plot_idio_vol_residual_dependence`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function plot_idio_vol_ic end
"""
    plot_idio_vol_residual_dependence(csfm::CrossSectionalFactorModel; kwargs...) -> Plot
    plot_idio_vol_residual_dependence(pr::AbstractPriorResult; kwargs...) -> Plot

Plot the residual dependence of the standardised idiosyncratic returns on the predicted volatility, against the observation axis. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

The figure draws what [`idio_vol_residual_dependence`](@ref) returns and computes nothing of its own. A dashed line marks the reference of `0`, which a well calibrated fit sits on. The volatility predicted at one observation is read against the return of the next, so the horizontal axis starts at the second observation. Read it beside [`plot_idio_vol_ic`](@ref). A fit that ranks well and leaves no residual dependence carries a high information coefficient and a dependence near `0`.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `pr`: A prior result whose `rr` is such a block.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - `pr.rr` is not `nothing`, else [`assert_prior_regression`](@ref) raises an `IsNothingError`.
  - `pr.rr` is a [`CrossSectionalFactorModel`](@ref), else the verb raises a `MethodError`: a time-series [`Regression`](@ref) carries no exposure history.
  - The rules of [`idio_vol_residual_dependence`](@ref), which needs at least two observations.

# Returns

  - `plt::Plots.Plot`: A line plot with one point per observation from the second, and the reference line.

# Related

  - [`idio_vol_residual_dependence`](@ref)
  - [`plot_idio_vol_ic`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function plot_idio_vol_residual_dependence end

## ────────────────────────────────────────────────────────────────────────────
## Factor attribution
## ────────────────────────────────────────────────────────────────────────────
"""
    plot_attribution_vol_contrib(
        fa::FactorAttributionResult;
        by_family::Bool = false,
        rd::ReturnsResult = ReturnsResult(),
        nf::Option{<:AbstractVector} = nothing,
        N::Option{<:Number} = nothing,
        kwargs...
    ) -> Plot

Plot the volatility contribution of each factor, or of each factor family, as a bar chart. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

The plot reads one [`FactorAttributionResult`](@ref) and computes nothing. The rows it draws are `fa.fbd.vol_contrib`, or `fa.fmbd.vol_contrib` under `by_family`, and it selects and orders the rows it shows. A family axis is present only when the factor model block names families, so `by_family = true` on a Result whose `fmbd` is `nothing` raises.

# Arguments

  - `fa`: Factor attribution result.
  - `by_family`: Whether to draw the family axis rather than the factor axis.
  - `rd`: Returns result providing the factor names through `rd.nf`.
  - `nf`: Factor names, one per row of the factor axis. They replace `rd.nf` when given. They have no effect under `by_family`, whose labels the Result carries.
  - `N`: The count of the rows to draw, ``\\min(\\max(\\lceil N \\rceil, 1), M)`` of the ``M`` rows, or `nothing` for every row. The rows are those of the largest finite magnitude of the value drawn, so a `NaN` row never takes a slot from a finite one. The rows shown keep the order of the axis, and the rest are not drawn.

# Validation

  - If `by_family` is `true`, `fa.fmbd` is not `nothing`, else an `ArgumentError` is raised.
  - If `N` is not `nothing`, `N > 0`, else a `DomainError` is raised.

# Returns

  - `plt::Plots.Plot`: A bar chart.

# Related

  - [`factor_attribution`](@ref)
  - [`FactorAttributionResult`](@ref)
  - [`plot_attribution_mu_contrib`](@ref)
  - [`plot_attribution_exposure`](@ref)
  - [`plot_attribution_mu_vs_vol`](@ref)
"""
function plot_attribution_vol_contrib end
"""
    plot_attribution_mu_contrib(
        fa::FactorAttributionResult;
        by_family::Bool = false,
        rd::ReturnsResult = ReturnsResult(),
        nf::Option{<:AbstractVector} = nothing,
        N::Option{<:Number} = nothing,
        z::Number = 1.96,
        kwargs...
    ) -> Plot

Plot the mean return contribution of each factor, or of each factor family, as a bar chart with error bars. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

The plot reads one [`FactorAttributionResult`](@ref) and computes nothing beyond the half-width `z * se` it draws. The bars are `fa.fbd.mu_contrib`, or `fa.fmbd.mu_contrib` under `by_family`, and the half-width of the error bar of row ``k`` is ``z \\, \\mathrm{se}_k``, from the standard errors the Result carries. A row whose standard error is `NaN`, such as a currency family, draws no error bar. A Result fitted without `se = true` carries none, and the plot then draws the bars alone.

# Arguments

  - `fa`: Factor attribution result.
  - `by_family`: Whether to draw the family axis rather than the factor axis.
  - `rd`: Returns result providing the factor names through `rd.nf`.
  - `nf`: Factor names, one per row of the factor axis. They replace `rd.nf` when given. They have no effect under `by_family`, whose labels the Result carries.
  - `N`: The count of the rows to draw, ``\\min(\\max(\\lceil N \\rceil, 1), M)`` of the ``M`` rows, or `nothing` for every row. The rows are those of the largest finite magnitude of the value drawn, so a `NaN` row never takes a slot from a finite one. The rows shown keep the order of the axis, and the rest are not drawn.
  - `z`: Half-width of the error bar, in standard errors.

# Validation

  - If `by_family` is `true`, `fa.fmbd` is not `nothing`, else an `ArgumentError` is raised.
  - If `N` is not `nothing`, `N > 0`, else a `DomainError` is raised.
  - `z >= 0`, else a `DomainError` is raised.

# Returns

  - `plt::Plots.Plot`: A bar chart, with error bars when the Result carries standard errors.

# Related

  - [`factor_attribution`](@ref)
  - [`FactorAttributionResult`](@ref)
  - [`plot_attribution_vol_contrib`](@ref)
  - [`plot_attribution_exposure`](@ref)
  - [`plot_attribution_mu_vs_vol`](@ref)
"""
function plot_attribution_mu_contrib end
"""
    plot_attribution_exposure(
        fa::FactorAttributionResult;
        by_family::Bool = false,
        rd::ReturnsResult = ReturnsResult(),
        nf::Option{<:AbstractVector} = nothing,
        N::Option{<:Number} = nothing,
        kwargs...
    ) -> Plot

Plot the portfolio's exposure to each factor, or to each factor family, as a bar chart with its spread. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

The plot reads one [`FactorAttributionResult`](@ref) and computes nothing. The bars are `fa.fbd.exposure`, or `fa.fmbd.exposure` under `by_family`, and each error bar is one standard deviation of the exposure over the observations, on either side of the bar. A predicted attribution reads one exposure and no history, so it carries no spread and the plot draws the bars alone.

# Arguments

  - `fa`: Factor attribution result.
  - `by_family`: Whether to draw the family axis rather than the factor axis.
  - `rd`: Returns result providing the factor names through `rd.nf`.
  - `nf`: Factor names, one per row of the factor axis. They replace `rd.nf` when given. They have no effect under `by_family`, whose labels the Result carries.
  - `N`: The count of the rows to draw, ``\\min(\\max(\\lceil N \\rceil, 1), M)`` of the ``M`` rows, or `nothing` for every row. The rows are those of the largest finite magnitude of the value drawn, so a `NaN` row never takes a slot from a finite one. The rows shown keep the order of the axis, and the rest are not drawn.

# Validation

  - If `by_family` is `true`, `fa.fmbd` is not `nothing`, else an `ArgumentError` is raised.
  - If `N` is not `nothing`, `N > 0`, else a `DomainError` is raised.

# Returns

  - `plt::Plots.Plot`: A bar chart, with error bars when the Result carries a history of the exposure.

# Related

  - [`factor_attribution`](@ref)
  - [`FactorAttributionResult`](@ref)
  - [`plot_attribution_vol_contrib`](@ref)
  - [`plot_attribution_mu_contrib`](@ref)
  - [`plot_attribution_mu_vs_vol`](@ref)
"""
function plot_attribution_exposure end
"""
    plot_attribution_mu_vs_vol(
        fa::FactorAttributionResult;
        by_family::Bool = false,
        rd::ReturnsResult = ReturnsResult(),
        nf::Option{<:AbstractVector} = nothing,
        N::Option{<:Number} = nothing,
        kwargs...
    ) -> Plot

Plot the mean return contribution of each factor against its volatility contribution, as a labelled scatter. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

The plot reads one [`FactorAttributionResult`](@ref) and computes nothing. Each point is one row of the factor axis, or of the family axis under `by_family`, placed at its volatility contribution and its mean return contribution. It is where a reader sees which factor paid for the risk it carried.

# Arguments

  - `fa`: Factor attribution result.
  - `by_family`: Whether to draw the family axis rather than the factor axis.
  - `rd`: Returns result providing the factor names through `rd.nf`.
  - `nf`: Factor names, one per row of the factor axis. They replace `rd.nf` when given. They have no effect under `by_family`, whose labels the Result carries.
  - `N`: The count of the rows to draw, ``\\min(\\max(\\lceil N \\rceil, 1), M)`` of the ``M`` rows, or `nothing` for every row. The rows are those of the largest finite magnitude of the volatility contribution. The rest are not drawn.

# Validation

  - If `by_family` is `true`, `fa.fmbd` is not `nothing`, else an `ArgumentError` is raised.
  - If `N` is not `nothing`, `N > 0`, else a `DomainError` is raised.

# Returns

  - `plt::Plots.Plot`: A scatter with one labelled point per drawn row.

# Related

  - [`factor_attribution`](@ref)
  - [`FactorAttributionResult`](@ref)
  - [`plot_attribution_vol_contrib`](@ref)
  - [`plot_attribution_mu_contrib`](@ref)
  - [`plot_attribution_exposure`](@ref)
"""
function plot_attribution_mu_vs_vol end

## ────────────────────────────────────────────────────────────────────────────
## Internal helpers (no Plots.jl dependency)
## ────────────────────────────────────────────────────────────────────────────
"""
    attribution_plot_axis(fa::FactorAttributionResult, by_family::Bool, rd::ReturnsResult,
                          nf::Option{<:AbstractVector})

Return the axis of a [`FactorAttributionResult`](@ref) a plot draws, and the labels of its rows.

The four attribution plots share one choice: the factor axis or the family axis, and where the labels of that axis come from. The family axis carries its own labels, because they are derived from the block. The factor axis carries none, because the factor names are carried input, so they come from `nf`, else from `rd.nf`, else from the position of the row.

# Arguments

  - `fa`: Factor attribution result.
  - `by_family`: Whether to draw the family axis rather than the factor axis.
  - `rd`: Returns result that gives the factor names through `rd.nf`. It has no effect on the family axis.
  - `nf`: Factor names. They replace `rd.nf` when given, and they have no effect on the family axis. Nothing checks that they count the rows.

# Validation

  - If `by_family` is `true`, `fa.fmbd` is not `nothing`, else an `ArgumentError` is raised.

# Returns

  - `bd::AttributionBreakdown`: The axis the plot draws.
  - `labels::AbstractVector`: The label of each row of that axis.

# Related

  - [`plot_attribution_vol_contrib`](@ref)
  - [`plot_attribution_mu_contrib`](@ref)
  - [`plot_attribution_exposure`](@ref)
  - [`plot_attribution_mu_vs_vol`](@ref)
"""
function attribution_plot_axis(fa::FactorAttributionResult, by_family::Bool,
                               rd::ReturnsResult, nf::Option{<:AbstractVector})
    if by_family
        @argcheck(!isnothing(fa.fmbd),
                  ArgumentError("this attribution has no family axis: the factor model block it decomposes names no factor family, so `fa.fmbd` is nothing. Fit the prior with families, or plot the factor axis"))
        return fa.fmbd, fa.fmbd.labels
    end
    K = length(fa.fbd.vol_contrib)
    labels = !isnothing(nf) ? nf : !isnothing(rd.nf) ? rd.nf : 1:K
    return fa.fbd, labels
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Rank the entries of a per-asset vector by magnitude, and choose how many of them a top-N figure draws.

The ranking reads [`finite_magnitudes`](@ref), so a non-finite entry ranks as a zero, below every live entry, and among the zeros by position. The count reads the shares of the magnitudes, so it does not change when `w` is scaled, so daily and annual expected returns of one universe give one count. A figure that draws the full axis when the count does not truncate keeps its frame.

# Mathematical definition

```math
\\begin{align}
p_i &= \\frac{m_i}{\\sum_{j=1}^{M} m_j}\\,, \\\\
N_{\\mathrm{eff}} &= \\left(\\sum_{i=1}^{M} p_i^2\\right)^{-1}\\,, \\\\
N &= \\begin{cases}
\\min\\left\\{k : \\sum_{j=1}^{k} p_{(j)} > 1 - n\\right\\} & 0 < n \\leq 1\\,, \\\\
\\min\\left(\\max\\left(\\lceil n \\rceil, 1\\right), M\\right) & \\text{otherwise}\\,.
\\end{cases}
\\end{align}
```

Where:

  - ``m_i``: Magnitude of entry ``i``, ``\\lvert w_i \\rvert`` when ``w_i`` is finite and zero otherwise.
  - ``p_i``: Share of entry ``i`` in the total magnitude, and ``p_{(j)}`` the ``j``-th largest share.
  - ``N_{\\mathrm{eff}}``: Effective number of entries. It lies in ``[1, M]``, and it is ``M`` when every magnitude is equal.
  - ``n``: The count or the share that the caller gives, or ``N_{\\mathrm{eff}}`` when the caller gives none. A value in ``(0, 1]`` is a share: ``N`` is the least number of the largest entries whose share **exceeds** ``1 - n``, so ``n = 0.1`` keeps more than 90 % of the magnitude.
  - ``N``: The number of entries the figure draws.
  - ``M``: The number of entries.

# Algorithm

 1. Map `w` to its finite magnitudes, `abs_w`.
 2. Sort `abs_w` in descending order, stably, giving the ranking `idx`.
 3. Divide `abs_w` by its sum, giving the shares `abs_w_norm`.
 4. Take `N_eff` as `N_opt`, or as [`number_effective_assets`](@ref) of `abs_w_norm` when `N_opt` is `nothing`.
 5. When `N_eff` is `NaN`, which it is when every magnitude is zero, take `N = M`.
 6. When `0 < N_eff ≤ 1`, take `N` as the first `k` at which the cumulative share `cw` of `abs_w_norm[idx]` exceeds `1 - N_eff`, or `M` when none does.
 7. Otherwise take `N = clamp(ceil(Int, N_eff), 1, M)`.

# Arguments

  - `w`: The per-asset vector the figure ranks by, such as a weight, an expected return, a centrality score or a mean absolute weight.
  - `M`: The number of entries, the largest count.
  - `N_opt`: The count or the share that the caller gives, or `nothing` for the effective number.

# Returns

  - `N::Int`: The number of entries to draw.
  - `idx::Vector{Int}`: The ranking of **every** entry, largest magnitude first. The first `N` entries are the ones to draw.

# Related

  - [`number_effective_assets`](@ref)
  - [`finite_magnitudes`](@ref)
"""
function relevant_assets(w::VecNum, M::Integer, N_opt::Option{<:Number} = nothing)
    abs_w = finite_magnitudes(w)
    idx = sortperm(abs_w; rev = true)
    abs_w_norm = abs_w ./ sum(abs_w)
    N_eff = isnothing(N_opt) ? number_effective_assets(abs_w_norm) : N_opt
    N = if isnan(N_eff)
        M
    elseif one(N_eff) >= N_eff > zero(N_eff)
        cw = cumsum(view(abs_w_norm, idx))
        k = findfirst(x -> one(x) - x < N_eff, cw)
        isnothing(k) ? M : k
    else
        clamp(ceil(Int, N_eff), 1, M)
    end
    return N, idx
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Take the ranking magnitudes of a per-asset vector, with a non-finite entry ranked as a zero.

A figure that shows the top assets sorts by the size of a per-asset number. A non-investable asset carries `NaN` in that number, and `NaN` sorts **first** under `rev = true`, so a blank bar would take the top slot from a live asset. The same `NaN` makes every reduction of the ranking `NaN`: `sum(v)` and `dot(v, v)` are both `NaN`, and `ceil(Int, NaN)` throws an `InexactError`.

Mapping a non-finite entry to zero settles the three for a vector with at least one live entry. The entry ranks below every non-zero entry, it counts for nothing, and the frame is unchanged when the count does not truncate. A vector whose entries are all finite ranks exactly as its absolute values do. A vector with no non-zero finite entry gives all zeros, and [`relevant_assets`](@ref) then draws every entry.

# Arguments

  - `v`: A per-asset number the figure ranks by, such as an expected return, a volatility, a centrality score or a weight.

# Returns

  - `m::Vector`: A new vector, ``\\lvert v_i \\rvert`` at every finite entry and zero at every other.

# Related

  - [`relevant_assets`](@ref)
  - [`finite_symmetric_clim`](@ref)
  - [`investable_mask`](@ref)
"""
function finite_magnitudes(v::VecNum)
    return map(vi -> isfinite(vi) ? abs(vi) : zero(vi), v)
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Take the symmetric colour limits of a heatmap over its finite entries alone.

`maximum(abs, A)` is `NaN` when any entry of `A` is, and a `NaN` colour limit paints every cell one colour. A drawn plot of a Prior Result keeps the full universe, so it does meet a `NaN`, because a non-investable asset carries one down its row and its column. The limits are therefore taken over the finite entries, and the blank cell is drawn against the scale the live cells set.

# Arguments

  - `A`: The matrix the heatmap draws.

# Validation

  - `A` is not empty, else `maximum` raises an `ArgumentError`.

# Returns

  - `clim::Tuple`: `(-c, c)`, where `c` is the largest finite absolute entry of `A`, or zero when `A` holds no finite non-zero entry.

# Related

  - [`finite_magnitudes`](@ref)
  - [`investable_mask`](@ref)
"""
function finite_symmetric_clim(A::MatNum)
    c = maximum(x -> isfinite(x) ? abs(x) : zero(x), A)
    return (-c, c)
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Keep the columns of `idx` whose column of `X` is finite throughout.

A figure that draws one line per asset keeps a gapped asset, and the break in its line is the delisting. A figure that sums several assets into **one** aggregate line cannot. A single `NaN` in the sum makes every date of it `NaN`, at any weight, even the zero weight that an optimiser gives a non-investable asset. Such a series draws nothing at all, so the aggregate excludes the asset instead of carrying it.

The rule reads the column rather than a mask, because the aggregate is formed from a returns matrix the caller holds and no prior travels with it.

# Arguments

  - `X`: The matrix whose columns the aggregate would sum (observations × assets), such as asset returns.
  - `idx`: Column indices the aggregate would sum.

# Validation

  - Every entry of `idx` indexes a column of `X`, else a `BoundsError` is raised.

# Returns

  - `keep::VecInt`: The entries of `idx` whose column of `X` holds no non-finite entry, in the order of `idx`.

# Related

  - [`plot_asset_cumulative_returns`](@ref)
  - [`investable_mask`](@ref)
"""
function finite_columns(X::MatNum, idx::AbstractVector{<:Integer})
    return [i for i in idx if all(isfinite, view(X, :, i))]
end

"""
    factor_plot_prior(pr::AbstractPriorResult)
    factor_plot_prior(pr::HighOrderPrior)

Return the low order factor prior whose `mu` and `sigma` a factor figure draws.

A prior result that carries a factor block keeps its low order factor moments in `fpr`. A [`HighOrderPrior`](@ref) is the exception. Its own `fpr` field holds the factor co-moments, and that field is `nothing` when the estimator formed none, although the prior it wraps still carries the low order factor block. The method for it falls back to the wrapped prior, so a factor figure draws the same block on both carriers.

# Arguments

  - `pr`: A prior result that carries a factor block. [`assert_prior_regression`](@ref) checks it before the call.

# Returns

  - `fpr`: The factor prior. Its `mu` is the vector of factor expected returns, and its `sigma` is the factor covariance matrix.

# Related

  - [`assert_prior_regression`](@ref)
  - [`plot_factor_mu`](@ref)
  - [`plot_factor_sigma`](@ref)
  - [`plot_factor_forecast_correlation`](@ref)
  - [`plot_factor_forecast_volatilities`](@ref)
"""
function factor_plot_prior(pr::AbstractPriorResult)
    return pr.fpr
end
function factor_plot_prior(pr::HighOrderPrior)
    return isnothing(pr.fpr) ? factor_plot_prior(pr.pr) : pr.fpr
end

"""
    investable_plot_view(
        pr::AbstractPriorResult,
        nx::Option{<:AbstractVector} = nothing,
        w::Option{<:VecNum} = nothing
    )
    investable_plot_view(
        rd::AbstractReturnsResult,
        nx::Option{<:AbstractVector} = nothing,
        w::Option{<:VecNum} = nothing
    )
    investable_plot_view(imsk::Nothing, pr::AbstractPriorResult, nx, w)
    investable_plot_view(imsk::BitVector, pr::AbstractPriorResult, nx, w)

Reduce a prior result, the axis names and the weights a computed figure draws to the Investable Mask.

A **drawn** plot keeps the frame. A heatmap or a bar chart of a Prior Result shows the full universe, and the backend leaves a blank cell and a missing bar where the asset is not investable. A **computed** plot has no such option. `eigvals(Symmetric(sigma))` refuses a `NaN`, and a phylogeny or a centrality score is fitted by a plain moment estimator, which refuses one too. Such a figure reduces here instead, and it draws the investable universe alone.

The reduction is the one [`port_opt_view`](@ref) the prior's owner already writes, so a new block cannot be forgotten and the reduced `pr.X` carries no dead column. The names and the weights ride the asset axis, so they take the mask directly.

A [`ReturnsResult`](@ref) carries no moments, so no mask exists to derive and it passes through. So one function states the reduction once, and dispatch decides whether it happens.

No diagnostic is emitted. A figure is a drawing rather than a number a caller acts on, and the docstring of each computed plot says that a non-investable asset is not drawn.

# Algorithm

 1. Return the three arguments unchanged when the carrier is a returns result.
 2. Derive the Investable Mask once with [`investable_mask`](@ref).
 3. Return the three unchanged when every asset is investable.
 4. Otherwise return a [`port_opt_view`](@ref) of the prior at `findall(imsk)`, and the views of the names and the weights at the mask.

# Arguments

  - `pr`: Prior result, or [`ReturnsResult`](@ref).
  - `nx`: Asset names of the axis, or `nothing`.
  - `w`: Portfolio weights the figure sizes by, or `nothing`.
  - `imsk`: The Investable Mask, or `nothing` when every asset is investable.

# Validation

  - At least one asset is investable, else [`investable_mask`](@ref) raises an `IsEmptyError`.
  - `nx` and `w` have one entry per asset, else the view raises a `BoundsError`.

# Returns

  - `(pr, nx, w)`: The three reduced to the Investable Mask, or unchanged.

# Related

  - [`investable_mask`](@ref)
  - [`investable_reduction`](@ref)
  - [`port_opt_view`](@ref)
  - [`plot_eigenspectrum`](@ref)
  - [`plot_cokurtosis`](@ref)
  - [`plot_network`](@ref)
  - [`plot_centrality`](@ref)
  - [`plot_dendrogram`](@ref)
  - [`plot_clusters`](@ref)
"""
function investable_plot_view(pr::AbstractPriorResult,
                              nx::Option{<:AbstractVector} = nothing,
                              w::Option{<:VecNum} = nothing)
    return investable_plot_view(investable_mask(pr), pr, nx, w)
end
function investable_plot_view(rd::AbstractReturnsResult,
                              nx::Option{<:AbstractVector} = nothing,
                              w::Option{<:VecNum} = nothing)
    return rd, nx, w
end
function investable_plot_view(::Nothing, pr::AbstractPriorResult,
                              nx::Option{<:AbstractVector}, w::Option{<:VecNum})
    return pr, nx, w
end
function investable_plot_view(imsk::BitVector, pr::AbstractPriorResult,
                              nx::Option{<:AbstractVector}, w::Option{<:VecNum})
    return port_opt_view(pr, findall(imsk)), nothing_scalar_array_view(nx, imsk),
           nothing_scalar_array_view(w, imsk)
end

"""
    plot_factor_model_summary(
        fs::FactorSummaryResult;
        nf::Option{<:AbstractVector} = nothing,
        kwargs...
    ) -> Plot
    plot_factor_model_summary(
        csfm::CrossSectionalFactorModel;
        nf::Option{<:AbstractVector} = nothing,
        ppy::Number = 1,
        threshold::Number = 2,
        step::Integer = 21,
        weighting = BenchmarkWeightMetric(),
        coverage_weighting = RegressionWeightMetric(),
        kwargs...
    ) -> Plot
    plot_factor_model_summary(pr::AbstractPriorResult; kwargs...) -> Plot

Plot the columns of a factor model summary as a grouped bar chart, one group per column and one bar per factor. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

The figure draws what [`factor_model_summary`](@ref) returns and computes nothing of its own. A column the summary carries as `nothing` is not drawn, and the title says so, so a block with no exposure history draws its four factor return columns alone.

The columns are not on one scale, and the figure rescales none of them. Read a column against its own factors and not against the column beside it.

# Arguments

  - `fs`: A factor model summary.
  - `csfm`: A cross-sectional factor model block, which the figure summarises first.
  - `pr`: A prior result whose `rr` is such a block.
  - `nf`: Factor names of the raw factor axis. `nothing` reads them off the block for the `csfm` and `pr` methods, and numbers the factors for the `fs` method. A list of the wrong length raises a `DimensionMismatch`.
  - `ppy`: Periods per year the summary annualises with.
  - `threshold`: Absolute t-statistic the exceedance rate counts against.
  - `step`: Number of observations between the two cross-sections the stability reads.
  - `weighting`: The [`AbstractOrthogonalityMetric`](@ref) the stability reads.
  - `coverage_weighting`: The [`AbstractOrthogonalityMetric`](@ref) whose positive weights are the universe of the coverage.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - `pr.rr` is not `nothing`, else [`assert_prior_regression`](@ref) raises an `IsNothingError`.
  - `pr.rr` is a [`CrossSectionalFactorModel`](@ref), else the verb raises a `MethodError`: a time-series [`Regression`](@ref) carries no exposure history.
  - The rules of [`factor_model_summary`](@ref).

# Returns

  - `plt::Plots.Plot`: A grouped bar chart, one group per drawn column and one bar per factor.

# Related

  - [`factor_model_summary`](@ref)
  - [`FactorSummaryResult`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function plot_factor_model_summary end
"""
    plot_factor_forecast_correlation(
        f_sigma::MatNum,
        nf::AbstractVector = 1:size(f_sigma, 1);
        kwargs...
    ) -> Plot
    plot_factor_forecast_correlation(
        pr::AbstractPriorResult,
        nf::Option{<:AbstractVector} = nothing;
        kwargs...
    ) -> Plot

Plot the forecast correlation of the factor returns as a heatmap, with the colour limits `(-1, 1)`.

The figure reads the factor covariance of the low order factor block that [`factor_plot_prior`](@ref) finds, and always rescales a copy of it to a correlation, by the formula of [`plot_correlation`](@ref). It is the forecast that the prior carries, and not the realised correlation of the fitted factor return series, so it answers on the factor axis of the factor prior. [`plot_factor_sigma`](@ref) draws the same correlation, and its colour limits span the entries when the matrix is already a correlation. A factor of zero variance gives a `NaN` row and column, drawn blank. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

# Arguments

  - `f_sigma`: Factor covariance matrix `factors × factors`.
  - `pr`: A prior result that carries a factor block.
  - `nf`: Factor names. `nothing` falls back to the position of the factor.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - `pr.rr` is not `nothing`, else [`assert_prior_regression`](@ref) raises an `IsNothingError`.

# Returns

  - `plt::Plots.Plot`: A heatmap of factors by factors.

# Related

  - [`plot_factor_sigma`](@ref)
  - [`plot_factor_forecast_volatilities`](@ref)
  - [`factor_plot_prior`](@ref)
"""
function plot_factor_forecast_correlation end
"""
    plot_factor_forecast_volatilities(
        f_sigma::MatNum,
        nf::AbstractVector = 1:size(f_sigma, 1);
        ppy::Number = 1,
        kwargs...
    ) -> Plot
    plot_factor_forecast_volatilities(
        pr::AbstractPriorResult,
        nf::Option{<:AbstractVector} = nothing;
        ppy::Number = 1,
        kwargs...
    ) -> Plot

Plot the forecast volatility of every factor return as a horizontal bar chart, ordered from the smallest.

The figure reads the factor covariance of the low order factor block that [`factor_plot_prior`](@ref) finds. It is the forecast that the prior carries, and not the realised volatility that [`factor_model_summary`](@ref) reports. A `NaN` volatility sorts last, so its bar is the top one. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

# Mathematical definition

```math
\\begin{align}
\\sigma_k = \\sqrt{p\\, \\Sigma_{f,kk}}\\,.
\\end{align}
```

Where:

  - ``\\Sigma_{f,kk}``: Diagonal entry ``k`` of the factor covariance, the variance of factor ``k``.
  - ``p``: The periods per year, `ppy`.
  - ``\\sigma_k``: Length of the bar of factor ``k``.

# Arguments

  - `f_sigma`: Factor covariance matrix `factors × factors`.
  - `pr`: A prior result that carries a factor block.
  - `nf`: Factor names. `nothing` falls back to the position of the factor.
  - `ppy`: Periods per year the volatility is annualised with.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - `pr.rr` is not `nothing`, else [`assert_prior_regression`](@ref) raises an `IsNothingError`.
  - `ppy >= 0`, else `sqrt` raises a `DomainError`.

# Returns

  - `plt::Plots.Plot`: A horizontal bar chart.

# Related

  - [`plot_factor_forecast_correlation`](@ref)
  - [`factor_model_summary`](@ref)
"""
function plot_factor_forecast_volatilities end
"""
    plot_factor_cumulative_returns(
        csfm::CrossSectionalFactorModel;
        nf::Option{<:AbstractVector} = nothing,
        compound::Bool = false,
        kwargs...
    ) -> Plot
    plot_factor_cumulative_returns(
        pr::AbstractPriorResult;
        nf::Option{<:AbstractVector} = nothing,
        compound::Bool = false,
        kwargs...
    ) -> Plot

Plot the cumulative return of every factor, one series per factor.

The figure draws [`cumulative_returns`](@ref) of each column of the factor return history `csr.f`, on the raw factor axis: the running sum of the returns, or their running product when `compound` is `true`. The figure sets a factor return that is not finite to zero first, so that observation holds the series flat and one absent cross-section breaks no series. [`plot_cumulative_exposure_ic`](@ref) follows the same convention. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `pr`: A prior result whose `rr` is such a block.
  - `nf`: Factor names of the raw factor axis. `nothing` reads them off the block, and falls back to the position of the factor.
  - `compound`: Whether the cumulative series compounds.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - `pr.rr` is not `nothing`, else [`assert_prior_regression`](@ref) raises an `IsNothingError`.
  - `pr.rr` is a [`CrossSectionalFactorModel`](@ref), else the verb raises a `MethodError`: a time-series [`Regression`](@ref) carries no exposure history.
  - `csfm.csr` is not `nothing`, else an `IsNothingError` is raised.

# Returns

  - `plt::Plots.Plot`: A line plot with one series per factor.

# Related

  - [`cumulative_returns`](@ref)
  - [`plot_cumulative_exposure_ic`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function plot_factor_cumulative_returns end
"""
    plot_forecast_cumulative_ic(
        fe::ForecastEvaluationResult,
        w::Option{<:MatNum} = nothing;
        min_count::Integer = fe.min_count,
        kwargs...
    ) -> Plot
    plot_forecast_cumulative_ic(
        fe::ForecastEvaluationResult,
        csfm::CrossSectionalFactorModel;
        weighting = IdentityMetric(),
        min_count::Integer = fe.min_count,
        kwargs...
    ) -> Plot
    plot_forecast_cumulative_ic(
        fes::AbstractVector{<:ForecastEvaluationResult},
        w::Option{<:MatNum} = nothing;
        names = nothing,
        rank::Bool = true,
        kwargs...
    ) -> Plot
    plot_forecast_cumulative_ic(
        fes::AbstractVector{<:ForecastEvaluationResult},
        csfm::CrossSectionalFactorModel;
        weighting = IdentityMetric(),
        names = nothing,
        rank::Bool = true,
        kwargs...
    ) -> Plot

Plot the running sum of the information coefficient of a Return Forecast: both coefficients of one forecast, or one coefficient of several forecasts overlaid.

The figure draws the running sum of what [`forecast_ic`](@ref) returns. An evaluation date that carries no coefficient adds nothing to the sum, so one thin cross-section breaks no series. A series that rises through the sample is a forecast that ordered the cross-section, a flat series is one that carried no ordering, and a falling series is one whose ordering had the opposite sign. The Spearman coefficient reads no weights, so `w`, `csfm` and `weighting` move the Pearson series alone. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

The vector method is the comparison. A mean hides whether an edge was continuous or came from three dates, and a comparison of two means hides it twice. So the vector method draws **one series per evaluation** on the dates that the evaluations share, with a zero reference line, and the reader sees where in the sample each forecast earned its coefficient. It is the figure of which the length-2 [`ForecastSummaryResult`](@ref) is the table. It refuses evaluations that are not comparable with [`forecast_summary_assert_comparable`](@ref), as [`forecast_evaluation_summary`](@ref) does, because two forecasts on different dates overlay nothing. It draws one coefficient, which `rank` names, so that a figure of four forecasts carries four series and not eight. Each evaluation reads its coefficient at its own `min_count`, for the reason [`forecast_summary_row`](@ref) states, and the comparability check makes it one number.

The figures of this group take the [`ForecastEvaluationResult`](@ref), where the cross-sectional diagnostics take the block. Those verbs read a block that is already fitted, so a figure that calls one costs nothing. Here the Return Forecast history can cost a rolling refit through [`forecast_history`](@ref), so a caller pairs once with [`forecast_evaluation`](@ref), and every figure reads that pairing. A weighting still reaches the figure as it reaches the level-2 verbs: as a bare weight history in the second position, or as the block whose [`AbstractOrthogonalityMetric`](@ref) resolves one.

# Mathematical definition

```math
\\begin{align}
C_j = \\sum_{i=1}^{j} \\mathbb{1}\\left[\\mathrm{IC}_i \\text{ is finite}\\right] \\mathrm{IC}_i\\,.
\\end{align}
```

Where:

  - ``\\mathrm{IC}_i``: Information coefficient of the forecast at evaluation date ``i``, as [`forecast_ic`](@ref) returns it.
  - ``C_j``: Point ``j`` of the series.

# Arguments

  - `fe`: The evaluation to draw, from [`forecast_evaluation`](@ref).
  - `fes`: The evaluations to overlay, at least one, from [`forecast_evaluation`](@ref).
  - `w`: Cross-sectional weight history `observations × assets`, on the axis of the forecast, or `nothing` for equal weights.
  - `csfm`: A cross-sectional factor model block, whose weight history `weighting` names.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref). It names the weight history the block is read with.
  - `min_count`: Least number of assets a cross-section needs before a coefficient of it is reported.
  - `names`: One name per evaluation, or `nothing` to number them.
  - `rank`: Overlay the Spearman coefficient when `true`, and the Pearson coefficient otherwise.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - The rules of [`forecast_ic`](@ref).
  - The vector method: the rules of [`forecast_summary_assert_comparable`](@ref) and [`forecast_summary_names`](@ref).

# Returns

  - `plt::Plots.Plot`: A line plot with one series per coefficient, or one series per evaluation and a zero reference line.

# Related

  - [`forecast_ic`](@ref)
  - [`forecast_ic_summary`](@ref)
  - [`forecast_summary_assert_comparable`](@ref)
  - [`forecast_summary_names`](@ref)
  - [`plot_forecast_rolling_ic`](@ref)
  - [`plot_forecast_cumulative_returns`](@ref)
  - [`plot_forecast_evaluation_summary`](@ref)
  - [`plot_cumulative_exposure_ic`](@ref)
  - [`ForecastEvaluationResult`](@ref)
"""
function plot_forecast_cumulative_ic end
"""
    plot_forecast_rolling_ic(
        fe::ForecastEvaluationResult,
        w::Option{<:MatNum} = nothing;
        rolling::Integer = 0,
        min_count::Integer = fe.min_count,
        kwargs...
    ) -> Plot
    plot_forecast_rolling_ic(
        fe::ForecastEvaluationResult,
        csfm::CrossSectionalFactorModel;
        weighting = IdentityMetric(),
        rolling::Integer = 0,
        min_count::Integer = fe.min_count,
        kwargs...
    ) -> Plot

Plot the running mean of both information coefficients of a Return Forecast over a window.

The figure draws the mean of the last ``W`` evaluation dates of what [`forecast_ic`](@ref) returns, one series per coefficient. Where [`plot_forecast_cumulative_ic`](@ref) shows what the forecast earned over the whole sample, this shows where in the sample it earned it. An evaluation date that carries no coefficient is left out of the mean of every window it falls in, so the series is the mean of the dates that scored. The first ``W - 1`` points, and a window with no finite coefficient, are `NaN` and drawn blank. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

The figure takes the evaluation and not the block, for the reason [`plot_forecast_cumulative_ic`](@ref) states.

# Mathematical definition

```math
\\begin{align}
W &= \\begin{cases} \\lceil \\sqrt{T} \\rceil & \\text{rolling} = 0\\,, \\\\ \\text{rolling} & \\text{otherwise}\\,, \\end{cases} \\\\
m_t &= \\frac{1}{\\lvert F_t \\rvert} \\sum_{i \\in F_t} \\mathrm{IC}_i\\,, \\quad t = W, \\ldots, T\\,.
\\end{align}
```

Where:

  - ``T``: Number of evaluation dates, `length(fe.dates)`.
  - $(math_dict[:W_roll])
  - ``F_t``: The dates of ``t - W + 1, \\ldots, t`` whose coefficient is finite.
  - ``\\mathrm{IC}_i``: Information coefficient at evaluation date ``i``, as [`forecast_ic`](@ref) returns it.
  - ``m_t``: Point ``t`` of a series.

# Arguments

  - `fe`: The evaluation to draw, from [`forecast_evaluation`](@ref).
  - `w`: Cross-sectional weight history `observations × assets`, on the axis of the forecast, or `nothing` for equal weights.
  - `csfm`: A cross-sectional factor model block, whose weight history `weighting` names.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref). It names the weight history the block is read with.
  - `rolling`: Number of evaluation dates in the window, or `0` for the square root of their number.
  - `min_count`: Least number of assets a cross-section needs before a coefficient of it is reported.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - ``W`` lies in `1:length(fe.dates)`, else a `DomainError` is raised.
  - The rules of [`forecast_ic`](@ref).

# Returns

  - `plt::Plots.Plot`: A line plot with one series per coefficient.

# Related

  - [`forecast_ic`](@ref)
  - [`plot_forecast_cumulative_ic`](@ref)
  - [`plot_rolling_measure`](@ref)
  - [`ForecastEvaluationResult`](@ref)
"""
function plot_forecast_rolling_ic end
"""
    plot_forecast_cumulative_returns(
        fe::ForecastEvaluationResult;
        kinds = (:rank, :zscore),
        compound::Bool = false,
        kwargs...
    ) -> Plot
    plot_forecast_cumulative_returns(
        fes::AbstractVector{<:ForecastEvaluationResult};
        names = nothing,
        kind::Symbol = :rank,
        compound::Bool = false,
        kwargs...
    ) -> Plot

Plot the cumulative return of the books a Return Forecast states on its own: both books of one forecast, or one book of several forecasts overlaid.

The figure draws [`cumulative_returns`](@ref) of the return series of [`forecast_portfolio`](@ref): their running sum, or their running product when `compound` is `true`. It computes nothing else of its own. The reference line of the vector method is the break-even level: zero for a running sum, and one under `compound`. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

Both books are centred and rescaled to the same gross exposure, so the two series are read against each other and against the same series of another forecast. An evaluation date whose cross-section carried no book is held flat, which is what [`plot_factor_cumulative_returns`](@ref) does to an absent factor return.

The vector method is the comparison, for the reason [`plot_forecast_cumulative_ic`](@ref) states: one series per evaluation on the dates the evaluations share, a zero reference line, and a refusal of evaluations that are not comparable with [`forecast_summary_assert_comparable`](@ref). One book is drawn, named by `kind`, so that a figure of four forecasts carries four series and not eight.

The figure takes the evaluation and not the block, for the reason [`plot_forecast_cumulative_ic`](@ref) states.

# Arguments

  - `fe`: The evaluation to draw, from [`forecast_evaluation`](@ref).
  - `fes`: The evaluations to overlay, at least one, from [`forecast_evaluation`](@ref).
  - `kinds`: The books to draw, each `:rank` or `:zscore`, as [`forecast_portfolio_weights`](@ref) names them.
  - `kind`: The one book to overlay, `:rank` or `:zscore`.
  - `names`: One name per evaluation, or `nothing` to number them.
  - `compound`: Whether the cumulative series compounds.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - `kinds` is not empty, else a `BoundsError` is raised.
  - The rules of [`forecast_portfolio`](@ref).
  - The vector method: the rules of [`forecast_summary_assert_comparable`](@ref) and [`forecast_summary_names`](@ref).

# Returns

  - `plt::Plots.Plot`: A line plot with one series per book, or one series per evaluation and a zero reference line.

# Related

  - [`forecast_portfolio`](@ref)
  - [`forecast_portfolio_weights`](@ref)
  - [`forecast_summary_assert_comparable`](@ref)
  - [`forecast_summary_names`](@ref)
  - [`cumulative_returns`](@ref)
  - [`plot_forecast_cumulative_ic`](@ref)
  - [`plot_forecast_quantile_returns`](@ref)
  - [`plot_forecast_evaluation_summary`](@ref)
  - [`ForecastEvaluationResult`](@ref)
"""
function plot_forecast_cumulative_returns end
"""
    plot_forecast_quantile_returns(
        fe::ForecastEvaluationResult;
        quantiles = (0.1,),
        compound::Bool = false,
        kwargs...
    ) -> Plot

Plot the cumulative top-minus-bottom spread of a Return Forecast, one series per quantile.

The figure draws [`cumulative_returns`](@ref) of the spread series of [`forecast_quantile_spread`](@ref): their running sum, or their running product when `compound` is `true`. It computes nothing else of its own. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

A spread that keeps rising as the tail narrows is a forecast whose ordering is sharpest at its ends, and one that flattens is a forecast whose ordering is spread across the whole cross-section. An evaluation date whose cross-section carried no spread is held flat.

The figure takes the evaluation and not the block, for the reason [`plot_forecast_cumulative_ic`](@ref) states.

# Arguments

  - `fe`: The evaluation to draw, from [`forecast_evaluation`](@ref).
  - `quantiles`: Tail fractions the spreads are cut at, each in `(0, 0.5]`.
  - `compound`: Whether the cumulative series compounds.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - The rules of [`forecast_quantile_spread`](@ref).

# Returns

  - `plt::Plots.Plot`: A line plot with one series per quantile.

# Related

  - [`forecast_quantile_spread`](@ref)
  - [`forecast_tail_spread`](@ref)
  - [`cumulative_returns`](@ref)
  - [`plot_forecast_cumulative_returns`](@ref)
  - [`ForecastEvaluationResult`](@ref)
"""
function plot_forecast_quantile_returns end
"""
    plot_forecast_calibration(
        fe::ForecastEvaluationResult,
        w::Option{<:MatNum} = nothing;
        bins::Integer = 10,
        kwargs...
    ) -> Plot
    plot_forecast_calibration(
        fe::ForecastEvaluationResult,
        csfm::CrossSectionalFactorModel;
        weighting = IdentityMetric(),
        bins::Integer = 10,
        kwargs...
    ) -> Plot

Plot the calibration curve of a Return Forecast against the slope fitted through it. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

The figure draws the curve of [`forecast_calibration`](@ref) as a scatter of the mean realised target of each bin against the mean forecast of that bin, and lays the zero-intercept slope of the same call over it. The two answer different questions. The curve says whether the relation is a straight line, and the slope says what multiplier maps the forecast onto realised units. A curve that sits below the slope at its right end is a forecast whose largest values are the least believable.

This is the one reading of a forecast that a rescaling moves. The coefficients correlate and the books are rescaled to a fixed gross exposure, so both are invariant to a positive rescaling of the units of the forecast. The slope is not, and that is what it is for.

The figure takes the evaluation and not the block, for the reason [`plot_forecast_cumulative_ic`](@ref) states.

# Arguments

  - `fe`: The evaluation to draw, from [`forecast_evaluation`](@ref).
  - `w`: Cross-sectional weight history `observations × assets`, on the axis of the forecast, or `nothing` for equal weights.
  - `csfm`: A cross-sectional factor model block, whose weight history `weighting` names.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref). It names the weight history the block is read with.
  - `bins`: Number of quantile bins the curve is cut into.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

The weights reach the slope alone, because the curve is the unweighted mean of each bin.

# Validation

  - The rules of [`forecast_calibration`](@ref).

# Returns

  - `plt::Plots.Plot`: A scatter of the curve, with the fitted slope as a dashed line.

# Related

  - [`forecast_calibration`](@ref)
  - [`forecast_calibration_curve`](@ref)
  - [`forecast_calibration_slope`](@ref)
  - [`ForecastEvaluationResult`](@ref)
"""
function plot_forecast_calibration end
"""
    plot_forecast_ic_by_holding_period(
        fe::ForecastEvaluationResult,
        X::MatNum,
        w::Option{<:MatNum} = nothing;
        n::Integer = 10,
        min_count::Integer = fe.min_count,
        kwargs...
    ) -> Plot
    plot_forecast_ic_by_holding_period(
        fe::ForecastEvaluationResult,
        rd::ReturnsResult,
        csfm::CrossSectionalFactorModel;
        weighting = IdentityMetric(),
        n::Integer = 10,
        min_count::Integer = fe.min_count,
        kwargs...
    ) -> Plot

Plot both mean information coefficients of a Return Forecast against the holding period. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

The figure draws two columns of the table of [`forecast_holding_period`](@ref) against its period and computes nothing of its own. A column that holds as the window lengthens is a forecast about a slow quantity, and one that falls away is a forecast whose book must be turned over to capture it.

Every row of the table is read on one date set, so the figure is internally comparable and is **not** comparable to the same figure at another `n`. A caller who compares two depths draws the deeper one.

The figure takes the evaluation and not the block, for the reason [`plot_forecast_cumulative_ic`](@ref) states. The target history is the second argument for the same reason the verb takes it, because a re-windowing needs the history the target was built from, and the evaluation carries only the matured target.

# Arguments

  - `fe`: The evaluation to draw, from [`forecast_evaluation`](@ref).
  - `X`: Target history `observations × assets`, on the axis of the forecast, from [`forecast_target_history`](@ref).
  - `rd`: The carrier the target history is built from. It carries an Asset Panel in `rd.pnl`.
  - `w`: Cross-sectional weight history `observations × assets`, on the axis of the forecast, or `nothing` for equal weights.
  - `csfm`: A cross-sectional factor model block, whose weight history `weighting` names.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref). It names the weight history the block is read with.
  - `n`: Number of holding periods the table reaches.
  - `min_count`: Least number of assets a cross-section needs before a statistic of it is reported. It also chooses the dates that every row of the table is read on.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - The rules of [`forecast_holding_period`](@ref).

# Returns

  - `plt::Plots.Plot`: A line plot with one series per column, against the holding period.

# Related

  - [`forecast_holding_period`](@ref)
  - [`plot_forecast_portfolio_by_holding_period`](@ref)
  - [`plot_forecast_ic_decay`](@ref)
  - [`ForecastEvaluationResult`](@ref)
"""
function plot_forecast_ic_by_holding_period end
"""
    plot_forecast_portfolio_by_holding_period(
        fe::ForecastEvaluationResult,
        X::MatNum,
        w::Option{<:MatNum} = nothing;
        n::Integer = 10,
        min_count::Integer = fe.min_count,
        kwargs...
    ) -> Plot
    plot_forecast_portfolio_by_holding_period(
        fe::ForecastEvaluationResult,
        rd::ReturnsResult,
        csfm::CrossSectionalFactorModel;
        weighting = IdentityMetric(),
        n::Integer = 10,
        min_count::Integer = fe.min_count,
        kwargs...
    ) -> Plot

Plot the annualised return and the Sharpe ratio of both books against the holding period. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

The figure draws four columns of the table of [`forecast_holding_period`](@ref) against its period and computes nothing of its own. It is the book-level reading of what [`plot_forecast_ic_by_holding_period`](@ref) shows at the coefficient level, and the two are read together. A coefficient that survives a longer window is only worth holding if the book built on it does too.

The four columns carry two units, so the axis is labelled `Value` and read column by column, which is what [`plot_factor_model_summary`](@ref) does with the same mix.

Every row of the table is read on one date set, so the figure is internally comparable and is **not** comparable to the same figure at another `n`.

The figure takes the evaluation and not the block, for the reason [`plot_forecast_cumulative_ic`](@ref) states.

# Arguments

  - `fe`: The evaluation to draw, from [`forecast_evaluation`](@ref).
  - `X`: Target history `observations × assets`, on the axis of the forecast, from [`forecast_target_history`](@ref).
  - `rd`: The carrier the target history is built from. It carries an Asset Panel in `rd.pnl`.
  - `w`: Cross-sectional weight history `observations × assets`, on the axis of the forecast, or `nothing` for equal weights.
  - `csfm`: A cross-sectional factor model block, whose weight history `weighting` names.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref). It names the weight history the block is read with.
  - `n`: Number of holding periods the table reaches.
  - `min_count`: Least number of assets a cross-section needs before a statistic of it is reported. It also chooses the dates that every row of the table is read on.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - The rules of [`forecast_holding_period`](@ref).

# Returns

  - `plt::Plots.Plot`: A line plot with one series per column, against the holding period.

# Related

  - [`forecast_holding_period`](@ref)
  - [`plot_forecast_ic_by_holding_period`](@ref)
  - [`plot_forecast_portfolio_decay`](@ref)
  - [`ForecastEvaluationResult`](@ref)
"""
function plot_forecast_portfolio_by_holding_period end
"""
    plot_forecast_ic_decay(
        fe::ForecastEvaluationResult,
        X::MatNum,
        w::Option{<:MatNum} = nothing;
        n::Integer = 10,
        min_count::Integer = fe.min_count,
        kwargs...
    ) -> Plot
    plot_forecast_ic_decay(
        fe::ForecastEvaluationResult,
        rd::ReturnsResult,
        csfm::CrossSectionalFactorModel;
        weighting = IdentityMetric(),
        n::Integer = 10,
        min_count::Integer = fe.min_count,
        kwargs...
    ) -> Plot

Plot both mean information coefficients of a Return Forecast against the forward window. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

The figure draws two columns of the table of [`forecast_decay`](@ref) against its period and computes nothing of its own. The windows are disjoint rather than cumulative, so the figure answers how long the forecast keeps forecasting. The first period is the horizon it was paired at, and a later period is the same forecast scored against a window it never saw.

Every row of the table is read on one date set, so the figure is internally comparable and is **not** comparable to the same figure at another `n`.

The figure takes the evaluation and not the block, for the reason [`plot_forecast_cumulative_ic`](@ref) states.

# Arguments

  - `fe`: The evaluation to draw, from [`forecast_evaluation`](@ref).
  - `X`: Target history `observations × assets`, on the axis of the forecast, from [`forecast_target_history`](@ref).
  - `rd`: The carrier the target history is built from. It carries an Asset Panel in `rd.pnl`.
  - `w`: Cross-sectional weight history `observations × assets`, on the axis of the forecast, or `nothing` for equal weights.
  - `csfm`: A cross-sectional factor model block, whose weight history `weighting` names.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref). It names the weight history the block is read with.
  - `n`: Number of forward windows the table reaches.
  - `min_count`: Least number of assets a cross-section needs before a statistic of it is reported. It also chooses the dates that every row of the table is read on.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - The rules of [`forecast_decay`](@ref).

# Returns

  - `plt::Plots.Plot`: A line plot with one series per column, against the forward window.

# Related

  - [`forecast_decay`](@ref)
  - [`plot_forecast_portfolio_decay`](@ref)
  - [`plot_forecast_ic_by_holding_period`](@ref)
  - [`ForecastEvaluationResult`](@ref)
"""
function plot_forecast_ic_decay end
"""
    plot_forecast_portfolio_decay(
        fe::ForecastEvaluationResult,
        X::MatNum,
        w::Option{<:MatNum} = nothing;
        n::Integer = 10,
        min_count::Integer = fe.min_count,
        kwargs...
    ) -> Plot
    plot_forecast_portfolio_decay(
        fe::ForecastEvaluationResult,
        rd::ReturnsResult,
        csfm::CrossSectionalFactorModel;
        weighting = IdentityMetric(),
        n::Integer = 10,
        min_count::Integer = fe.min_count,
        kwargs...
    ) -> Plot

Plot the annualised return and the Sharpe ratio of both books against the forward window. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

The figure draws four columns of the table of [`forecast_decay`](@ref) against its period and computes nothing of its own. It is the book-level reading of what [`plot_forecast_ic_decay`](@ref) shows at the coefficient level.

The four columns carry two units, so the axis is labelled `Value` and read column by column.

Every row of the table is read on one date set, so the figure is internally comparable and is **not** comparable to the same figure at another `n`.

The figure takes the evaluation and not the block, for the reason [`plot_forecast_cumulative_ic`](@ref) states.

# Arguments

  - `fe`: The evaluation to draw, from [`forecast_evaluation`](@ref).
  - `X`: Target history `observations × assets`, on the axis of the forecast, from [`forecast_target_history`](@ref).
  - `rd`: The carrier the target history is built from. It carries an Asset Panel in `rd.pnl`.
  - `w`: Cross-sectional weight history `observations × assets`, on the axis of the forecast, or `nothing` for equal weights.
  - `csfm`: A cross-sectional factor model block, whose weight history `weighting` names.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref). It names the weight history the block is read with.
  - `n`: Number of forward windows the table reaches.
  - `min_count`: Least number of assets a cross-section needs before a statistic of it is reported. It also chooses the dates that every row of the table is read on.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - The rules of [`forecast_decay`](@ref).

# Returns

  - `plt::Plots.Plot`: A line plot with one series per column, against the forward window.

# Related

  - [`forecast_decay`](@ref)
  - [`plot_forecast_ic_decay`](@ref)
  - [`plot_forecast_portfolio_by_holding_period`](@ref)
  - [`ForecastEvaluationResult`](@ref)
"""
function plot_forecast_portfolio_decay end
"""
    plot_forecast_factor_correlation(
        fe::ForecastEvaluationResult,
        B::Arr3Num,
        w::Option{<:MatNum} = nothing;
        nf::Option{<:AbstractVector} = nothing,
        dates::AbstractVector{<:Integer} = axes(fe.alpha, 1),
        rank::Bool = false,
        min_count::Integer = fe.min_count,
        kwargs...
    ) -> Plot
    plot_forecast_factor_correlation(
        fe::ForecastEvaluationResult,
        csfm::CrossSectionalFactorModel;
        nf::Option{<:AbstractVector} = nothing,
        weighting = IdentityMetric(),
        kwargs...
    ) -> Plot

Plot the contemporaneous correlation of a Return Forecast with every factor exposure. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

The figure draws what [`forecast_factor_correlation`](@ref) returns and computes nothing of its own, one series per factor, on the rows `dates` names. Nothing here is forward looking, because the correlation is taken on the date the forecast is stated, so the figure says what the forecast **is**, not what it earned, and it is drawn on every observation by default rather than on the evaluation grid; `dates = fe.dates` draws the grid. A series that sits near one is a forecast that restates an exposure the risk model already holds, and the return it earns is that factor's return under another name.

A neutralised forecast does not read zero here. The cross-sectional fit that neutralises it carries no intercept, so its residual is orthogonal to its target in the uncentred sense and keeps a correlation with it.

The figure takes the evaluation and not the block, for the reason [`plot_forecast_cumulative_ic`](@ref) states.

# Arguments

  - `fe`: The evaluation to draw, from [`forecast_evaluation`](@ref).
  - `B`: Exposure history `observations × assets × factors`, on the axis of the forecast.
  - `w`: Cross-sectional weight history `observations × assets`, on the axis of the forecast, or `nothing` for equal weights.
  - `csfm`: A cross-sectional factor model block, whose exposure history and weight history are read.
  - `nf`: Factor names of the axis of the answer. `nothing` reads them off the block for the `csfm` method, and numbers the factors for the `B` method.
  - `dates`: Row indices of `fe.alpha` the correlation is read and drawn on. The default is every observation; `fe.dates` reads the evaluation grid.
  - `rank`: Take the rank correlation when `true`, and the weighted correlation otherwise.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref). It names the weight history the block is read with.
  - `min_count`: Least number of assets a cross-section needs before a correlation of it is reported.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend. The `csfm` method passes `dates`, `rank` and `min_count` through them to the `B` method.

# Validation

  - The rules of [`forecast_factor_correlation`](@ref).

# Returns

  - `plt::Plots.Plot`: A line plot with one series per factor.

# Related

  - [`forecast_factor_correlation`](@ref)
  - [`forecast_factor_exposures`](@ref)
  - [`exposure_ic_summary`](@ref)
  - [`ForecastEvaluationResult`](@ref)
"""
function plot_forecast_factor_correlation end
"""
    plot_forecast_evaluation_summary(fs::ForecastSummaryResult; kwargs...) -> Plot
    plot_forecast_evaluation_summary(
        fes::AbstractVector{<:ForecastEvaluationResult},
        w::Option{<:MatNum} = nothing;
        names = nothing,
        bins::Integer = 10,
        quantiles = nothing,
        kwargs...
    ) -> Plot
    plot_forecast_evaluation_summary(
        fes::AbstractVector{<:ForecastEvaluationResult},
        csfm::CrossSectionalFactorModel;
        weighting = IdentityMetric(),
        names = nothing,
        bins::Integer = 10,
        quantiles = nothing,
        kwargs...
    ) -> Plot
    plot_forecast_evaluation_summary(
        fe::ForecastEvaluationResult,
        w::Option{<:MatNum} = nothing;
        kwargs...
    ) -> Plot
    plot_forecast_evaluation_summary(
        fe::ForecastEvaluationResult,
        csfm::CrossSectionalFactorModel;
        kwargs...
    ) -> Plot

Plot a [`ForecastSummaryResult`](@ref) as a grouped bar chart, one series per forecast. The package extension `PortfolioOptimisersPlotsExt` implements the methods, and it loads with `StatsPlots`.

The figure draws the ten headline columns of the summary, with the statistic on the axis and the forecast as the series, which is the shape [`plot_factor_model_summary`](@ref) takes. The ten are both mean information coefficients and their information ratios, the annualised return and the Sharpe ratio of each book, the calibration slope and the mean coverage. A single evaluation is the length-1 case and a set of them **is** the comparison, as the Result is.

The quantile spread block is drawn when the summary carries one, as one bar group per quantile, and each group is the Sharpe ratio of that spread; the other columns of the block are not drawn. A summary built with no quantile carries the block as `nothing`, and then the block is **not drawn** and the title says so.

The methods that summarise do not align the evaluations, so a set of evaluations on different dates raises. Align them first with [`forecast_evaluation_align`](@ref), or summarise them with [`forecast_evaluation_summary`](@ref) under `align = true` and draw the summary.

The columns carry several units, so the axis is labelled `Value` and read group by group. The columns the summary carries that this figure does not draw are read off the Result, which prints all of them.

# Arguments

  - `fs`: The summary to draw, from [`forecast_evaluation_summary`](@ref).
  - `fes`: The evaluations to summarise and draw, at least one.
  - `fe`: One evaluation. It is drawn as the length-1 case.
  - `w`: Cross-sectional weight history `observations × assets`, on the axis of the forecasts, or `nothing` for equal weights.
  - `csfm`: A cross-sectional factor model block, whose weight history `weighting` names.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref). It names the weight history the block is read with.
  - `names`: One name per evaluation, or `nothing` to number them.
  - `bins`: Number of quantile bins the calibration curve cuts.
  - `quantiles`: Tail fractions the quantile spreads are cut at, each in `(0, 0.5]`, or `nothing` for none.
  - `kwargs...`: Additional keyword arguments passed to the plotting backend.

# Validation

  - The rules of [`forecast_evaluation_summary`](@ref).

# Returns

  - `plt::Plots.Plot`: A grouped bar chart, one group per statistic and one bar per forecast.

# Related

  - [`forecast_evaluation_summary`](@ref)
  - [`ForecastSummaryResult`](@ref)
  - [`plot_factor_model_summary`](@ref)
  - [`ForecastEvaluationResult`](@ref)
"""
function plot_forecast_evaluation_summary end
export plot_portfolio_cumulative_returns, plot_asset_cumulative_returns, plot_composition,
       plot_stacked_bar_composition, plot_stacked_area_composition, plot_dendrogram,
       plot_clusters, plot_drawdowns, plot_risk_contribution, plot_factor_risk_contribution,
       plot_measures, plot_histogram, plot_network, plot_centrality, plot_correlation,
       plot_mu, plot_sigma, plot_factor_loadings, plot_factor_sigma, plot_eigenspectrum,
       plot_rolling_measure, plot_weight_stability, plot_cv_scores, plot_turnover,
       plot_prior, plot_factor_mu, plot_benchmark, plot_coskewness, plot_cokurtosis,
       plot_portfolio_dashboard, plot_cv_dashboard, plot_efficient_frontier,
       plot_performance_summary, plot_rolling_drawdowns, plot_cs_regression_r2,
       plot_cs_regression_adjusted_r2, plot_cs_regression_aic, plot_cs_regression_bic,
       plot_cs_regression_t_stats, plot_cs_regression_t_stat_exceedance_rate,
       plot_attribution_vol_contrib, plot_attribution_mu_contrib, plot_attribution_exposure,
       plot_attribution_mu_vs_vol, plot_exposure_vif, plot_exposure_condition_number,
       plot_exposure_correlation, plot_cumulative_exposure_ic, plot_exposure_distribution,
       plot_exposure_dispersion, plot_exposure_stability, plot_idio_calibration,
       plot_idio_tail_rate, plot_idio_kurtosis, plot_idio_skewness, plot_idio_vol_ic,
       plot_idio_vol_residual_dependence, plot_factor_model_summary,
       plot_factor_forecast_correlation, plot_factor_forecast_volatilities,
       plot_factor_cumulative_returns, plot_forecast_cumulative_ic,
       plot_forecast_rolling_ic, plot_forecast_cumulative_returns,
       plot_forecast_quantile_returns, plot_forecast_calibration,
       plot_forecast_ic_by_holding_period, plot_forecast_portfolio_by_holding_period,
       plot_forecast_ic_decay, plot_forecast_portfolio_decay,
       plot_forecast_factor_correlation, plot_forecast_evaluation_summary
