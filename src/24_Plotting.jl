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
        pred::Union{
                        <:PredictionResult,
                        <:MultiPeriodPredictionResult,
                        <:PopulationPredictionResult
                    };
        compound::Bool = false,
        kwargs...
    ) -> Plot

Plot the cumulative returns of a portfolio.

# Arguments

  - `w`: Portfolio weights vector or vector of weight vectors.
  - `X`: Asset returns matrix (observations × assets).
  - `fees::Option{<:Fees} = nothing`: Optional transaction fees.
  - `ts::AbstractVector = 1:size(X, 1)`: Time axis labels.
  - `compound::Bool = false`: If `true`, compound cumulative returns; otherwise use simple cumulative sums.
  - `pr`: Prior or returns result; extracts `X`, `ts`, and `nx` automatically when available.
  - `res::OptimisationResult`: Extracts `w` and `fees` when available.
  - `pred`: Predicted portfolio results.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

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
    plot_asset_cumulative_returns(w::VecNum, pr::Pr_RR, fees = nothing; compound, N, kwargs...) -> Plot
    plot_asset_cumulative_returns(res::OptimisationResult[, pr]; compound, N, kwargs...) -> Plot
    plot_asset_cumulative_returns(pred; compound, N, kwargs...) -> Plot

Plot the cumulative returns of individual assets, selecting the most relevant via `N`.
Assets beyond the top `N` are aggregated into an "Others" series.

# Arguments

  - `w`: Portfolio weights vector.
  - `X`: Asset returns matrix (observations × assets).
  - `fees::Option{<:Fees} = nothing`: Optional transaction fees.
  - `ts::AbstractVector = 1:size(X, 1)`: Time axis labels.
  - `nx::AbstractVector = 1:size(X, 2)`: Asset names.
  - `compound::Bool = false`: If `true`, compound cumulative returns.
  - `N::Option{<:Number} = nothing`: Maximum number of assets to display individually.
    `nothing` auto-selects via [`number_effective_assets`](@ref).
    A value in `(0, 1]` is treated as a cumulative weight threshold; a value `> 1` as an asset count.
  - `pr`: Prior or returns result; extracts `X`, `ts`, and `nx` automatically.
  - `res::OptimisationResult`: Extracts `w` and `fees` when available.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

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
    plot_composition(res::OptimisationResult, rd; N, kwargs...) -> Plot
    plot_composition(res::OptimisationResult, pr; N, kwargs...) -> Plot
    plot_composition(pred::PredictionResult; N, kwargs...) -> Plot
    plot_composition(mpred::MultiPeriodPredictionResult; N, kwargs...) -> Plot
    plot_composition(ppred::PopulationPredictionResult; N, kwargs...) -> Plot

Plot portfolio composition as a bar chart of asset weights. Assets beyond the top `N`
are collapsed into an "Others" bar.

`mpred` / `ppred` overloads produce stacked-bar fold compositions via
[`plot_stacked_bar_composition`](@ref); `N` is accepted but not applied in those overloads.

# Arguments

  - `w`: Portfolio weights vector.
  - `nx::AbstractVector = 1:length(w)`: Asset names.
  - `N::Option{<:Number} = nothing`: Maximum number of assets to display.
    `nothing` auto-selects via [`number_effective_assets`](@ref).
    A value in `(0, 1]` is treated as a cumulative weight threshold; a value `> 1` as an asset count.
  - `res::OptimisationResult`: Extracts `w` when available.
  - `rd::ReturnsResult`: Extracts `nx` when available.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

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
        nx::AbstractVector = 1:size(w, 1);
        kwargs...
    ) -> Plot
    plot_stacked_bar_composition(res_vec::AbstractVector{<:OptimisationResult}, rd; kwargs...) -> Plot

Plot portfolio composition as a stacked bar chart. Accepts a matrix, a `VecVecNum`, or a
vector of [`OptimisationResult`](@ref).

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`plot_stacked_area_composition`](@ref)
"""
function plot_stacked_bar_composition end

"""
    plot_stacked_area_composition(
        w::VecNum_VecVecNum,
        nx::AbstractVector = 1:size(w, 1);
        kwargs...
    ) -> Plot
    plot_stacked_area_composition(res_vec::AbstractVector{<:OptimisationResult}, rd; kwargs...) -> Plot

Plot portfolio composition as a stacked area chart. Accepts a matrix, `VecVecNum`, or a
vector of [`OptimisationResult`](@ref).

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`plot_stacked_bar_composition`](@ref)
"""
function plot_stacked_area_composition end

## ──────────────────────────────────────────────────────────────────────────────
## Risk contribution
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_risk_contribution(
        r::AbstractBaseRiskMeasure,
        w::VecNum,
        X::MatNum_Pr,
        fees::Option{<:Fees} = nothing;
        nx::AbstractVector = 1:length(w),
        delta::Number = 1e-6,
        marginal::Bool = false,
        percentage::Bool = false,
        N::Option{<:Number} = nothing,
        kwargs...
    ) -> Plot
    plot_risk_contribution(r, w, rd::ReturnsResult, fees = nothing; delta, marginal, percentage, N, kwargs...) -> Plot
    plot_risk_contribution(r, res::OptimisationResult, rd; delta, marginal, percentage, N, kwargs...) -> Plot
    plot_risk_contribution(r, res::OptimisationResult, pr; nx, delta, marginal, percentage, N, kwargs...) -> Plot

Plot per-asset risk contribution as a bar chart.

# Arguments

  - `r`: Risk measure.
  - `w`: Portfolio weights vector.
  - `X` / `rd` / `pr`: Asset returns or prior result.
  - `fees::Option{<:Fees} = nothing`: Optional transaction fees.
  - `nx::AbstractVector = 1:length(w)`: Asset names.
  - `delta::Number = 1e-6`: Finite-difference step size for [`risk_contribution`](@ref). Must be `> 0`.
  - `marginal::Bool = false`: If `true`, compute marginal risk contribution; otherwise component.
  - `percentage::Bool = false`: If `true`, normalise contributions to percentages.
  - `N::Option{<:Number} = nothing`: Maximum number of assets to display.

# Validation

  - `delta > 0`.
  - If `N` is not `nothing`, `N > 0`.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`risk_contribution`](@ref)
  - [`plot_composition`](@ref)
"""
function plot_risk_contribution end

## ──────────────────────────────────────────────────────────────────────────────
## Factor risk contribution
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_factor_risk_contribution(
        r::AbstractBaseRiskMeasure,
        w::VecNum,
        X::MatNum_Pr,
        fees::Option{<:Fees} = nothing;
        re::RegE_Reg = StepwiseRegression(),
        rd::ReturnsResult = ReturnsResult(),
        nf::Option{<:AbstractVector} = nothing,
        delta::Number = 1e-6,
        N::Option{<:Number} = nothing,
        kwargs...
    ) -> Plot
    plot_factor_risk_contribution(r, res::OptimisationResult, rd; re, delta, N, kwargs...) -> Plot

Plot per-factor risk contribution as a bar chart, including the constant (idiosyncratic) term.

# Arguments

  - `r`: Risk measure.
  - `w`: Portfolio weights vector.
  - `X` / `rd`: Asset returns or returns result.
  - `fees::Option{<:Fees} = nothing`: Optional transaction fees.
  - `re::RegE_Reg = StepwiseRegression()`: Factor regression estimator.
  - `rd::ReturnsResult = ReturnsResult()`: Returns result providing factor names via `rd.nf`.
  - `nf::Option{<:AbstractVector} = nothing`: Factor names; overrides `rd.nf` when provided.
  - `delta::Number = 1e-6`: Finite-difference step size. Must be `> 0`.
  - `N::Option{<:Number} = nothing`: Maximum number of factors to display.

# Validation

  - `delta > 0`.
  - If `N` is not `nothing`, `N > 0`.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`factor_risk_contribution`](@ref)
  - [`plot_composition`](@ref)
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
    plot_dendrogram(cle::HClE_HCl, X::MatNum, nx = 1:size(X,2); dims, kwargs...) -> Plot
    plot_dendrogram(cle::HClE_HCl, pr::Pr_RR, nx = 1:size(pr.X,2); dims, kwargs...) -> Plot

Plot a hierarchical clustering dendrogram with coloured cluster regions.

# Arguments

  - `clr::AbstractClusteringResult`: Precomputed clustering result.
  - `cle::HClE_HCl`: Clustering estimator; computes clustering from `X` or `pr.X`.
  - `pr`: [`AbstractPriorResult`](@ref); extracts `X` and optionally `nx`.
  - `nx`: Asset names.
  - `dend_theme::Symbol = :Spectral`: Colour palette for cluster regions.
  - `dims::Integer = 1`: Dimension passed to [`clusterise`](@ref).

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

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
    plot_clusters(cle::HClE_HCl, X::MatNum, nx = 1:size(X,2); dims, kwargs...) -> Plot
    plot_clusters(cle::HClE_HCl, pr::Pr_RR, nx = 1:size(pr.X,2); dims, kwargs...) -> Plot

Plot a reordered correlation/covariance heatmap with flanking dendrograms and coloured
cluster boxes.

# Arguments

  - `clr::AbstractClusteringResult`: Precomputed clustering result.
  - `cle::HClE_HCl`: Clustering estimator.
  - `pr`: [`AbstractPriorResult`](@ref); extracts `X` and optionally `nx`.
  - `nx`: Asset names.
  - `dend_theme::Symbol = :Spectral`: Colour palette for dendrogram cluster fills.
  - `hmap_theme::Symbol = :Spectral`: Colour gradient for the heatmap.
  - `color_func`: Function mapping the matrix to a colour range `(lo, hi)`.
  - `line_color = :black`: Colour of the cluster-border lines on the heatmap.
  - `line_width = 3`: Width of the cluster-border lines.
  - `dims::Integer = 1`: Dimension passed to [`clusterise`](@ref).

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

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
    plot_drawdowns(res::OptimisationResult, rd; slv, compound, alpha, kappa, rw, kwargs...) -> Plot
    plot_drawdowns(pred; slv, compound, alpha, kappa, rw, kwargs...) -> Plot
    plot_drawdowns(mpred::MultiPeriodPredictionResult; slv, compound, alpha, kappa, rw, kwargs...) -> Plot
    plot_drawdowns(ppred::PopulationPredictionResult; slv, compound, alpha, kappa, rw, kwargs...) -> Plot

Plot portfolio drawdown over time with horizontal lines for AverageDrawdown, UlcerIndex,
DaR, CDaR, MaximumDrawdown, and — when `slv` is provided — EDaR and RLDaR.

# Arguments

  - `w`: Portfolio weights.
  - `X`: Asset returns matrix (observations × assets).
  - `fees::Option{<:Fees} = nothing`: Optional transaction fees.
  - `slv::Option{<:Slv_VecSlv} = nothing`: Solver for EDaR / RLDaR lines (omitted when `nothing`).
  - `ts::AbstractVector = 1:size(X, 1)`: Time axis labels.
  - `compound::Bool = false`: If `true`, use compound drawdowns.
  - `alpha::Number = 0.05`: Confidence level for DaR / CDaR / EDaR / RLDaR lines. Must satisfy `0 < alpha < 1`.
  - `kappa::Number = 0.3`: Relativistic deformation parameter for RLDaR. Must satisfy `0 < kappa < 1`.
  - `rw`: Optional observation weights for AverageDrawdown and UlcerIndex.

# Validation

  - `0 < alpha < 1`.
  - `0 < kappa < 1`.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

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
        x::AbstractBaseRiskMeasure = Variance(),
        y::AbstractBaseRiskMeasure = ExpectedReturn(),
        z::Option{<:AbstractBaseRiskMeasure} = nothing,
        c::AbstractBaseRiskMeasure = ExpectedReturnRiskRatio(; rk=x, rt=ArithmeticReturn(), rf=0),
        slv::Option{<:Slv_VecSlv} = nothing,
        factory::Bool = true,
        kwargs...
    ) -> Plot
    plot_measures(res_vec::AbstractVector{<:OptimisationResult}, pr = nothing; x, y, z, c, slv, fees, factory, kwargs...) -> Plot
    plot_measures(ppred::PopulationPredictionResult; x, y, z, c, slv, factory, kwargs...) -> Plot

Scatter plot of risk/return measures across a collection of portfolio weight vectors.

# Arguments

  - `w`: Portfolio weights or vector of weight vectors.
  - `pr`: Prior or returns result.
  - `x`: Risk/return measure for the horizontal axis (default `Variance()`).
  - `y`: Risk/return measure for the vertical axis (default `ExpectedReturn()`).
  - `z`: Optional third measure for 3-D scatter.
  - `c`: Colour-coding measure (default Sharpe ratio derived from `x`).
  - `slv::Option{<:Slv_VecSlv} = nothing`: Solver passed to `factory`.
  - `factory::Bool = true`: If `true`, call [`factory`](@ref) on measures before evaluating.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`expected_risk`](@ref)
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
    plot_histogram(res::OptimisationResult, rd; slv, alpha, kappa, rw, points, reference, kwargs...) -> Plot
    plot_histogram(pred; slv, alpha, kappa, rw, points, reference, kwargs...) -> Plot
    plot_histogram(mpred::MultiPeriodPredictionResult; slv, alpha, kappa, rw, points, reference, kwargs...) -> Plot
    plot_histogram(ppred::PopulationPredictionResult; slv, alpha, kappa, rw, points, reference, kwargs...) -> Plot

Plot a histogram of portfolio returns with vertical risk-measure lines and an optional
fitted Normal distribution.

# Arguments

  - `w`: Portfolio weights.
  - `X`: Asset returns matrix (observations × assets).
  - `fees::Option{<:Fees} = nothing`: Optional transaction fees.
  - `slv::Option{<:Slv_VecSlv} = nothing`: Solver for EVaR / RLVaR lines (omitted when `nothing`).
  - `alpha::Number = 0.05`: Tail confidence level. Must satisfy `0 < alpha < 1`.
  - `kappa::Number = 0.3`: Relativistic deformation parameter for RLVaR. Must satisfy `0 < kappa < 1`.
  - `rw`: Optional observation weights for MAD, GMD, VaR, and CVaR.
  - `points::Integer = 0`: Number of PDF evaluation points. `0` auto-detects as `ceil(Int, 4√T)`.
  - `reference::Bool = true`: If `true`, overlay a fitted Normal distribution curve.

# Validation

  - `0 < alpha < 1`.
  - `0 < kappa < 1`.
  - `points >= 0`.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).
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
    plot_network(pl, pr::Pr_RR, w = nothing; nx, kwargs...) -> Plot
    plot_network(pl, res::OptimisationResult; rd, nx, kwargs...) -> Plot

Plot the asset network (MST, PMFG, TMFG, or adjacency) as a graph using
`GraphRecipes.graphplot`. Node size is uniform by default; pass `w` to scale node
area proportionally to portfolio weight.

# Arguments

  - `pl`: Network or clustering estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `nx`: Asset names.
  - `w::Option{<:VecNum} = nothing`: Optional portfolio weights for node sizing.
  - `threshold::Number = 0`: Adjacency entries with absolute value ≤ this are zeroed.
  - `pr`: Prior or returns result; extracts `X` and optionally `nx`.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots` and `GraphRecipes`).

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
        percentage::Bool = false,
        kwargs...
    ) -> Plot
    plot_centrality(cte, pr::AbstractPriorResult, nx = 1:size(pr.X,2); N, percentage, kwargs...) -> Plot
    plot_centrality(cte, rd::ReturnsResult; N, percentage, kwargs...) -> Plot
    plot_centrality(cte, res::OptimisationResult, rd; N, percentage, kwargs...) -> Plot

Bar chart of asset centrality scores, sorted in descending order.

# Arguments

  - `cte`: Centrality estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `nx`: Asset names.
  - `N::Option{<:Number} = nothing`: Maximum number of assets to display.
    `nothing` auto-selects via [`number_effective_assets`](@ref).
    A value in `(0, 1]` is treated as a cumulative score threshold; a value `> 1` as an asset count.
  - `percentage::Bool = false`: If `true`, normalise scores to sum to one.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`centrality_vector`](@ref)
"""
function plot_centrality end

## ──────────────────────────────────────────────────────────────────────────────
## Correlation / covariance heatmap
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_correlation(X::MatNum, nx::AbstractVector = 1:size(X, 1); kwargs...) -> Plot
    plot_correlation(pr::AbstractPriorResult, nx = 1:size(pr.sigma,1); kwargs...) -> Plot
    plot_correlation(pr::AbstractPriorResult, rd::ReturnsResult; kwargs...) -> Plot
    plot_correlation(res::OptimisationResult[, rd]; kwargs...) -> Plot
    plot_correlation(pred::PredictionResult[, rd]; kwargs...) -> Plot

Standalone correlation (or covariance) heatmap without clustering or dendrograms.
If the input contains a covariance matrix, it is normalised to a correlation matrix before plotting.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

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
    plot_mu(res::OptimisationResult[, rd]; N, kwargs...) -> Plot
    plot_mu(pred::PredictionResult[, rd]; N, kwargs...) -> Plot

Bar chart of per-asset expected returns (μ vector).

# Arguments

  - `mu`: Expected returns vector.
  - `nx`: Asset names.
  - `N::Option{<:Number} = nothing`: Maximum number of assets to display.
    `nothing` auto-selects via [`number_effective_assets`](@ref).
    A value in `(0, 1]` is treated as a cumulative return threshold; a value `> 1` as an asset count.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

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
    plot_sigma(res::OptimisationResult[, rd]; variance, N, kwargs...) -> Plot
    plot_sigma(pred::PredictionResult[, rd]; variance, N, kwargs...) -> Plot

Bar chart of per-asset volatility (√diag(Σ)).

# Arguments

  - `sigma`: Covariance (or correlation) matrix.
  - `nx`: Asset names.
  - `variance::Bool = false`: If `true`, show variance (diag(Σ)) instead of standard deviation.
  - `N::Option{<:Number} = nothing`: Maximum number of assets to display.
    `nothing` auto-selects the top assets by volatility magnitude.
    A value `> 1` is treated as an asset count.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

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
    plot_factor_loadings(pr::AbstractPriorResult[, nx, nf]; kwargs...) -> Plot
    plot_factor_loadings(pr::AbstractPriorResult, rd::ReturnsResult; kwargs...) -> Plot
    plot_factor_loadings(res::OptimisationResult[, rd]; kwargs...) -> Plot
    plot_factor_loadings(pred::PredictionResult[, rd]; kwargs...) -> Plot

Heatmap of the factor loadings matrix B (assets × factors) from a prior with a regression
model. Uses a diverging colour scale centred at zero.

Requires that `pr.rr` is not `nothing` (i.e. the prior was estimated with a factor model).

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`plot_factor_sigma`](@ref)
  - [`LowOrderPrior`](@ref)
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
    plot_factor_sigma(pr::AbstractPriorResult[, nf]; kwargs...) -> Plot
    plot_factor_sigma(pr::AbstractPriorResult, rd::ReturnsResult; kwargs...) -> Plot
    plot_factor_sigma(res::OptimisationResult[, rd]; kwargs...) -> Plot
    plot_factor_sigma(pred::PredictionResult[, rd]; kwargs...) -> Plot

Correlation/covariance heatmap of the factor covariance matrix (`pr.f_sigma`). Behaves
identically to [`plot_correlation`](@ref) but operates on the factor space.

Requires that `pr.f_sigma` is not `nothing`.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`plot_factor_loadings`](@ref)
  - [`plot_correlation`](@ref)
  - [`LowOrderPrior`](@ref)
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
    plot_eigenspectrum(res::OptimisationResult[, rd]; reference, kwargs...) -> Plot
    plot_eigenspectrum(pred::PredictionResult[, rd]; reference, kwargs...) -> Plot

Bar chart of eigenvalues of the covariance/correlation matrix, sorted in descending order.

# Arguments

  - `sigma`: Covariance or correlation matrix.
  - `N_obs::Option{<:Integer} = nothing`: Number of observations; enables Marchenko-Pastur overlay.
  - `reference::Bool = true`: If `true` and `N_obs` is provided, overlays the Marchenko-Pastur
    bulk upper bound `λ₊ = σ̄²(1 + √(N/T))²` as a reference line.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`plot_correlation`](@ref)
"""
function plot_eigenspectrum end

## ──────────────────────────────────────────────────────────────────────────────
## Rolling risk/return measure
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_rolling_measure(
        r::AbstractBaseRiskMeasure,
        w::VecNum,
        X::MatNum,
        fees::Option{<:Fees} = nothing;
        ts::AbstractVector = 1:size(X, 1),
        rolling::Integer = 0,
        kwargs...
    ) -> Plot
    plot_rolling_measure(r, w, rd::ReturnsResult, fees = nothing; rolling, kwargs...) -> Plot
    plot_rolling_measure(r, res::OptimisationResult, rd; rolling, kwargs...) -> Plot
    plot_rolling_measure(r, pred; rolling, kwargs...) -> Plot
    plot_rolling_measure(r, mpred::MultiPeriodPredictionResult; rolling, kwargs...) -> Plot
    plot_rolling_measure(r, ppred::PopulationPredictionResult; rolling, kwargs...) -> Plot

Line plot of a risk or return measure evaluated over a rolling window of portfolio returns.

# Arguments

  - `r`: Risk or return measure. May embed its own solver (e.g. `EntropicValueatRisk(; slv=...)`).
  - `w`: Portfolio weights.
  - `X`: Asset returns matrix (observations × assets).
  - `fees::Option{<:Fees} = nothing`: Optional transaction fees.
  - `ts::AbstractVector = 1:size(X, 1)`: Time axis labels.
  - `rolling::Integer = 0`: Rolling window size. `0` auto-detects as `⌈√T⌉`. Must be `>= 0`.

# Validation

  - `rolling >= 0`.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`expected_risk`](@ref)
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

Box plot of per-asset weight distributions across cross-validation folds or population members.

# Arguments

  - `mpred::MultiPeriodPredictionResult`: Walk-forward prediction result.
  - `ppred::PopulationPredictionResult`: Population prediction result; pools weights from all members.
  - `N::Option{<:Number} = nothing`: Maximum number of assets to display by mean absolute weight.
    `nothing` shows all assets.
    A value `> 1` is treated as an asset count.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

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
    plot_cv_scores(r::AbstractBaseRiskMeasure, mpred::MultiPeriodPredictionResult; kwargs...) -> Plot
    plot_cv_scores(r::AbstractBaseRiskMeasure, ppred::PopulationPredictionResult; kwargs...) -> Plot

Bar chart of cross-validation scores (one bar per fold or population member).

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`expected_risk`](@ref)
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

Line plot of portfolio turnover (L1 weight change) over time.

Turnover at step `t` is defined as `∑ |w_t − w_{t−1}|`.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`MultiPeriodPredictionResult`](@ref)
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
    plot_prior(res::OptimisationResult[, rd]; N, kwargs...) -> Plot
    plot_prior(pred::PredictionResult[, rd]; N, kwargs...) -> Plot

Three-panel composite plot summarising a prior result:

 1. Expected returns bar chart ([`plot_mu`](@ref)).
 2. Asset volatility bar chart ([`plot_sigma`](@ref)).
 3. Correlation heatmap ([`plot_correlation`](@ref)).

# Arguments

  - `pr::AbstractPriorResult`: Prior result containing `mu` and `sigma`.
  - `nx`: Asset names.
  - `N::Option{<:Number} = nothing`: Forwarded to [`plot_mu`](@ref) and [`plot_sigma`](@ref) to limit displayed assets.
  - `rd::ReturnsResult`: Provides asset names via `rd.nx` when given.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

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
    plot_factor_mu(pr::AbstractPriorResult[, nf]; N, kwargs...) -> Plot
    plot_factor_mu(pr::AbstractPriorResult, rd::ReturnsResult; N, kwargs...) -> Plot
    plot_factor_mu(res::OptimisationResult[, rd]; N, kwargs...) -> Plot
    plot_factor_mu(pred::PredictionResult[, rd]; N, kwargs...) -> Plot

Bar chart of per-factor expected returns (f_μ vector from a factor model prior).

Requires that the prior was estimated with a factor model (`pr.f_mu` is not `nothing`).

# Arguments

  - `f_mu`: Factor expected returns vector.
  - `nf`: Factor names.
  - `N::Option{<:Number} = nothing`: Maximum number of factors to display.
    `nothing` auto-selects via [`number_effective_assets`](@ref).

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`plot_mu`](@ref)
  - [`LowOrderPrior`](@ref)
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
    plot_benchmark(res::OptimisationResult, rd; compound, kwargs...) -> Plot
    plot_benchmark(pred::PredictionResult; compound, kwargs...) -> Plot
    plot_benchmark(mpred::MultiPeriodPredictionResult; compound, kwargs...) -> Plot

Overlay portfolio cumulative returns against one or more benchmark return series from `rd.B`.

# Arguments

  - `w`: Portfolio weights.
  - `X`: Asset returns matrix (observations × assets).
  - `B`: Benchmark return series or vector of series.
  - `fees::Option{<:Fees} = nothing`: Optional transaction fees.
  - `ts::AbstractVector = 1:size(X, 1)`: Time axis labels.
  - `nb::Option{<:AbstractVector} = nothing`: Benchmark names.
  - `compound::Bool = false`: If `true`, compound cumulative returns for both portfolio and benchmarks.
  - `rd::ReturnsResult`: Provides `B`, `ts`, and `nb`; throws `ArgumentError` if `rd.B` is `nothing`.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

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
    plot_coskewness(pr::HighOrderPrior[, nx]; kwargs...) -> Plot
    plot_coskewness(pr::HighOrderPrior, rd::ReturnsResult; kwargs...) -> Plot
    plot_coskewness(res::OptimisationResult[, rd]; kwargs...) -> Plot
    plot_coskewness(pred::PredictionResult[, rd]; kwargs...) -> Plot

Heatmap of the coskewness matrix (N × N²) from a [`HighOrderPrior`](@ref).
Uses a diverging colour scale centred at zero.

Requires that `pr.sk` is not `nothing` (i.e. the prior was estimated with higher moments).

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

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
    plot_cokurtosis(pr::HighOrderPrior[, nx]; heatmap, reference, kwargs...) -> Plot
    plot_cokurtosis(pr::HighOrderPrior, rd::ReturnsResult; heatmap, reference, kwargs...) -> Plot
    plot_cokurtosis(res::OptimisationResult[, rd]; heatmap, reference, kwargs...) -> Plot
    plot_cokurtosis(pred::PredictionResult[, rd]; heatmap, reference, kwargs...) -> Plot

Eigenvalue spectrum of the cokurtosis matrix (N² × N²) from a [`HighOrderPrior`](@ref).

# Arguments

  - `kt`: Cokurtosis matrix (N² × N²).
  - `nx`: Asset names.
  - `heatmap::Bool = false`: If `true`, show the raw heatmap instead (only recommended for small N).
  - `reference::Bool = true`: If `true`, overlays the mean eigenvalue as a reference line.

Requires that `pr.kt` is not `nothing`.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

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
        r::AbstractBaseRiskMeasure = Variance(),
        slv::Option{<:Slv_VecSlv} = nothing,
        compound::Bool = false,
        N::Option{<:Number} = nothing,
        delta::Number = 1e-6,
        marginal::Bool = false,
        percentage::Bool = false,
        alpha::Number = 0.05,
        kappa::Number = 0.3,
        rw = nothing,
        kwargs...
    ) -> Plot

Four-panel composite plot for a single optimisation result:

 1. Portfolio composition ([`plot_composition`](@ref)).
 2. Cumulative returns ([`plot_portfolio_cumulative_returns`](@ref)).
 3. Asset risk contribution ([`plot_risk_contribution`](@ref)).
 4. Drawdowns ([`plot_drawdowns`](@ref)).

`r` selects the risk measure for panels 3 and 4 (default `Variance()`).

Note: panel 3 requires raw asset returns. Pass `rd::ReturnsResult` (original returns),
not a `PredictionResult`, for the risk contribution panel.

# Arguments

  - `res::OptimisationResult`: Optimisation result.
  - `rd::Pr_RR`: Returns result or prior; extracts `nx` and `ts` from `rd` when `ReturnsResult`.
  - `r`: Risk measure for panels 3 and 4.
  - `slv`: Solver for EDaR / RLDaR drawdown lines.
  - `compound::Bool = false`: If `true`, compound cumulative returns and drawdowns.
  - `N::Option{<:Number} = nothing`: Forwarded to composition and risk contribution panels.
  - `delta::Number = 1e-6`: Finite-difference step for risk contribution. Must be `> 0`.
  - `marginal::Bool = false`: Marginal vs component risk contribution.
  - `percentage::Bool = false`: Normalise risk contributions to percentages.
  - `alpha::Number = 0.05`: Confidence level for drawdown risk lines.
  - `kappa::Number = 0.3`: Relativistic parameter for RLDaR.
  - `rw`: Observation weights for drawdown risk measures.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

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

Four-panel composite plot for a walk-forward cross-validation result:

 1. Stacked-bar fold compositions ([`plot_composition`](@ref)).
 2. Fold-shaded cumulative returns ([`plot_portfolio_cumulative_returns`](@ref)).
 3. Turnover per fold ([`plot_turnover`](@ref)).
 4. Weight stability box plot ([`plot_weight_stability`](@ref)).

# Arguments

  - `mpred::MultiPeriodPredictionResult`: Walk-forward prediction result.
  - `N::Option{<:Number} = nothing`: Forwarded to [`plot_weight_stability`](@ref).
  - `compound::Bool = false`: Forwarded to [`plot_portfolio_cumulative_returns`](@ref).

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

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
        x::AbstractBaseRiskMeasure = Variance(),
        y::AbstractBaseRiskMeasure = ExpectedReturn(),
        c::AbstractBaseRiskMeasure = ExpectedReturnRiskRatio(; rk=x, rt=ArithmeticReturn(), rf=0),
        slv::Option{<:Slv_VecSlv} = nothing,
        fees::Option{<:Fees} = nothing,
        min_risk::Bool = true,
        max_score::Bool = true,
        factory::Bool = true,
        kwargs...
    ) -> Plot
    plot_efficient_frontier(res_vec, rd::ReturnsResult; kwargs...) -> Plot
    plot_efficient_frontier(w::VecVecNum, pr::Pr_RR; x, y, c, slv, fees, min_risk, max_score, factory, kwargs...) -> Plot
    plot_efficient_frontier(res::OptimisationResult, pr::Pr_RR; fees, kwargs...) -> Plot
    plot_efficient_frontier(res::OptimisationResult, rd::ReturnsResult; kwargs...) -> Plot

Sort a collection of portfolio results by risk (`x`), connect them with a line to
trace the efficient frontier, and optionally annotate the minimum-risk and maximum-score portfolios.

# Arguments

  - `res_vec` / `w`: Portfolio results or weight vectors.
  - `pr` / `rd`: Prior or returns result for risk evaluation.
  - `x`: Risk measure for the horizontal axis (default `Variance()`).
  - `y`: Return measure for the vertical axis (default `ExpectedReturn()`).
  - `c`: Colour-coding measure (default Sharpe ratio derived from `x`).
  - `slv::Option{<:Slv_VecSlv} = nothing`: Solver passed to `factory`.
  - `fees::Option{<:Fees} = nothing`: Optional transaction fees.
  - `min_risk::Bool = true`: Overlay a star marker at the minimum-risk portfolio.
  - `max_score::Bool = true`: Overlay a star marker at the portfolio that maximises `c`.
  - `factory::Bool = true`: If `true`, call [`factory`](@ref) on measures before evaluating.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`plot_measures`](@ref)
  - [`expected_risk`](@ref)
"""
function plot_efficient_frontier end

## ──────────────────────────────────────────────────────────────────────────────
## Performance summary
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_performance_summary(
        w::ArrNum,
        X::MatNum,
        fees::Option{<:Fees} = nothing;
        periods_per_year::Number = 252,
        alpha::Number = 0.05,
        compound::Bool = false,
        kwargs...
    ) -> Plot
    plot_performance_summary(w, rd::ReturnsResult, fees = nothing; alpha, compound, kwargs...) -> Plot
    plot_performance_summary(res::OptimisationResult, rd; alpha, compound, kwargs...) -> Plot
    plot_performance_summary(pred; alpha, compound, kwargs...) -> Plot
    plot_performance_summary(mpred::MultiPeriodPredictionResult; alpha, compound, kwargs...) -> Plot

Bar chart of annualised portfolio performance metrics:
annualised return, annualised volatility, Sharpe ratio, Sortino ratio, Calmar ratio,
maximum drawdown %, and CVaR %.

# Arguments

  - `w`: Portfolio weights.
  - `X`: Asset returns matrix (observations × assets).
  - `fees::Option{<:Fees} = nothing`: Optional transaction fees.
  - `periods_per_year::Number = 252`: Trading periods per year used for annualisation.
  - `alpha::Number = 0.05`: Tail probability for CVaR. Must satisfy `0 < alpha < 1`.
  - `compound::Bool = false`: If `true`, use compound cumulative returns for max drawdown.

# Validation

  - `0 < alpha < 1`.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

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
    plot_rolling_drawdowns(res::OptimisationResult, rd; rolling, compound, kwargs...) -> Plot
    plot_rolling_drawdowns(pred; rolling, compound, kwargs...) -> Plot
    plot_rolling_drawdowns(mpred::MultiPeriodPredictionResult; rolling, compound, kwargs...) -> Plot

Line plot of the rolling maximum drawdown over a sliding window.

# Arguments

  - `w`: Portfolio weights.
  - `X`: Asset returns matrix (observations × assets).
  - `fees::Option{<:Fees} = nothing`: Optional transaction fees.
  - `ts::AbstractVector = 1:size(X, 1)`: Time axis labels.
  - `rolling::Integer = 0`: Window size. `0` auto-detects as `⌈√T⌉`. Must be `>= 0`.
  - `compound::Bool = false`: If `true`, use compound drawdowns.

# Validation

  - `rolling >= 0`.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`drawdowns`](@ref)
  - [`plot_drawdowns`](@ref)
  - [`plot_rolling_measure`](@ref)
"""
function plot_rolling_drawdowns end

## ────────────────────────────────────────────────────────────────────────────
## Internal helpers (no Plots.jl dependency)
## ────────────────────────────────────────────────────────────────────────────
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Select the top-N assets from a weight vector by absolute weight magnitude.

# Arguments

  - `w::VecNum`: Portfolio weight vector.
  - `M::Integer`: Total number of assets (upper bound for `N`).
  - `N_opt::Option{<:Number} = nothing`: Asset-count specification.
    `nothing` auto-selects via [`number_effective_assets`](@ref).

# Returns

  - `Tuple{Int, Vector{Int}}`: `(N, idx)` where `N` is the number of selected assets and
    `idx` is a permutation vector sorted descending by `|w|`.

# Details

  - When `N_opt` is `nothing`, `N_eff = number_effective_assets(w)`.
  - When `0 < N_eff ≤ 1`, `N_eff` is treated as a concentration threshold: `N` is the
    smallest index such that the cumulative normalised absolute weight covers at least
    `1 - N_eff` of the total; falls back to `M` if no such index exists.
  - Otherwise `N_eff` is treated as a count: `N = clamp(ceil(Int, N_eff), 1, M)`.

# Related

  - [`number_effective_assets`](@ref)
"""
function relevant_assets(w::VecNum, M::Integer, N_opt::Option{<:Number} = nothing)
    N_eff = isnothing(N_opt) ? number_effective_assets(w) : N_opt
    abs_w = abs.(w)
    idx = sortperm(abs_w; rev = true)
    abs_w_norm = abs_w ./ sum(abs_w)
    N = if one(N_eff) >= N_eff > zero(N_eff)
        cw = cumsum(view(abs_w_norm, idx))
        k = findfirst(x -> one(x) - x < N_eff, cw)
        isnothing(k) ? M : k
    else
        clamp(ceil(Int, N_eff), 1, M)
    end
    return N, idx
end

export plot_portfolio_cumulative_returns, plot_asset_cumulative_returns, plot_composition,
       plot_stacked_bar_composition, plot_stacked_area_composition, plot_dendrogram,
       plot_clusters, plot_drawdowns, plot_risk_contribution, plot_factor_risk_contribution,
       plot_measures, plot_histogram, plot_network, plot_centrality, plot_correlation,
       plot_mu, plot_sigma, plot_factor_loadings, plot_factor_sigma, plot_eigenspectrum,
       plot_rolling_measure, plot_weight_stability, plot_cv_scores, plot_turnover,
       plot_prior, plot_factor_mu, plot_benchmark, plot_coskewness, plot_cokurtosis,
       plot_portfolio_dashboard, plot_cv_dashboard, plot_efficient_frontier,
       plot_performance_summary, plot_rolling_drawdowns
