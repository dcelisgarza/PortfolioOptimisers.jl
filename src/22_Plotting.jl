"""
$(DocStringExtensions.TYPEDEF)

Configuration struct for plotting functions in `PortfolioOptimisers.jl`.

`PlottingOptions` encapsulates semantic plotting parameters (confidence levels, display modes,
numeric tolerances) that are shared across multiple `plot_*` functions and is passed via an
`opts` keyword argument. Plot-specific boolean flags (e.g. `variance` for
[`plot_sigma`](@ref), `heatmap` for [`plot_cokurtosis`](@ref), `min_risk`/`max_score` for
[`plot_efficient_frontier`](@ref)) are passed directly as `kwargs...` on those functions.
Plots.jl formatting options (titles, axis labels, colours, etc.) are also passed through
`kwargs...`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PlottingOptions(;
        alpha::Number    = 0.05,
        kappa::Number    = 0.3,
        N                = nothing,
        delta::Number    = 1e-6,
        points::Integer  = 0,
        rw               = nothing,
        rolling::Integer = 0,
        compound::Bool   = false,
        marginal::Bool   = false,
        percentage::Bool = false,
        reference::Bool  = true,
        factory::Bool    = true,
    ) -> PlottingOptions

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha < 1`.
  - `0 < kappa < 1`.
  - `delta > 0`.
  - `points >= 0`.
  - `rolling >= 0`.
  - If `N` is not `nothing`, `N > 0`.

# Examples

```jldoctest
julia> opts = PlottingOptions()
PlottingOptions
       alpha ┼ Float64: 0.05
       kappa ┼ Float64: 0.3
           N ┼ nothing
       delta ┼ Float64: 1.0e-6
      points ┼ Int64: 0
          rw ┼ nothing
     rolling ┼ Int64: 0
    compound ┼ Bool: false
    marginal ┼ Bool: false
  percentage ┼ Bool: false
   reference ┼ Bool: true
     factory ┴ Bool: true
```

# Related

  - [`AbstractEstimator`](@ref)
  - [`plot_ptf_cumulative_returns`](@ref)
  - [`plot_drawdowns`](@ref)
  - [`plot_histogram`](@ref)
  - [`plot_risk_contribution`](@ref)
  - [`plot_rolling_measure`](@ref)
"""
@concrete struct PlottingOptions <: AbstractEstimator
    """
    Confidence level for tail risk measures (VaR, CVaR, DaR, …). Must satisfy `0 < alpha < 1`.
    """
    alpha
    """
    Relativistic deformation parameter for RLVaR / RLDaR. Must satisfy `0 < kappa < 1`.
    """
    kappa
    """
    Maximum number of assets/factors to display. `nothing` auto-selects via [`number_effective_assets`](@ref). A value in `(0, 1]` is treated as a cumulative weight threshold; a value `> 1` is treated as an asset count.
    """
    N
    """
    Finite-difference step size for [`risk_contribution`](@ref). Must be `> 0`.
    """
    delta
    """
    Number of histogram bins / PDF evaluation points. `0` means auto-detect as `ceil(Int, 4*sqrt(T))`.
    """
    points
    """
    Optional observation weights for risk measures that accept [`ObsWeights`](@ref).
    """
    rw
    """
    Rolling window size (≥ 0). Used by [`plot_rolling_measure`](@ref); `0` = auto (⌈√T⌉).
    """
    rolling
    """
    If `true`, compound cumulative returns; otherwise use simple cumulative sums.
    """
    compound
    """
    If `true`, compute marginal risk contribution; otherwise component risk contribution.
    """
    marginal
    """
    If `true`, normalise asset risk contributions to percentages.
    """
    percentage
    """
    If `true`, overlay a fitted Normal distribution in [`plot_histogram`](@ref), a reference
    bound/line in eigenvalue plots, or a mean-eigenvalue line in [`plot_cokurtosis`](@ref).
    """
    reference
    """
    If `true`, call [`factory`](@ref) on plot measures before evaluation in
    [`plot_measures`](@ref) and [`plot_efficient_frontier`](@ref).
    """
    factory
    function PlottingOptions(alpha::Number, kappa::Number, N::Option{<:Number},
                             delta::Number, points::Integer, rw::Option{<:ObsWeights},
                             rolling::Integer, compound::Bool, marginal::Bool,
                             percentage::Bool, reference::Bool, factory::Bool)
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(zero(kappa) < kappa < one(kappa))
        @argcheck(delta > zero(delta))
        @argcheck(points >= 0)
        @argcheck(rolling >= 0)
        if !isnothing(N)
            @argcheck(N > zero(N))
        end
        return new{typeof(alpha), typeof(kappa), typeof(N), typeof(delta), typeof(points),
                   typeof(rw), typeof(rolling), typeof(compound), typeof(marginal),
                   typeof(percentage), typeof(reference), typeof(factory)}(alpha, kappa, N,
                                                                           delta, points,
                                                                           rw, rolling,
                                                                           compound,
                                                                           marginal,
                                                                           percentage,
                                                                           reference,
                                                                           factory)
    end
end
function PlottingOptions(; alpha::Number = 0.05, kappa::Number = 0.3,
                         N::Option{<:Number} = nothing, delta::Number = 1e-6,
                         points::Integer = 0, rw::Option{<:ObsWeights} = nothing,
                         rolling::Integer = 0, compound::Bool = false,
                         marginal::Bool = false, percentage::Bool = false,
                         reference::Bool = true, factory::Bool = true)::PlottingOptions
    return PlottingOptions(alpha, kappa, N, delta, points, rw, rolling, compound, marginal,
                           percentage, reference, factory)
end

## ──────────────────────────────────────────────────────────────────────────────
## Cumulative returns
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_ptf_cumulative_returns(
        w::VecNum_VecVecNum,
        X::MatNum,
        fees::Option{<:Fees} = nothing;
        ts::AbstractVector = 1:size(X, 1),
        compound::Bool = false,
        kwargs...
    ) -> Plot
    plot_ptf_cumulative_returns(
        w::VecNum_VecVecNum,
        pr::Pr_RR,
        fees::Option{<:Fees} = nothing;
        ts::AbstractVector = 1:size(pr.X,1),
        compound::Bool = false,
        kwargs...
    ) -> Plot
    plot_ptf_cumulative_returns(
        res::OptimisationResult;
        pr::Option{<:Pr_RR} = nothing,
        fees::Option{<:Fees} = nothing,
        compound::Bool = false,
        kwargs...
    ) -> Plot
    plot_ptf_cumulative_returns(
        pred::Union{
                        <:PredictionResult,
                        <:MultiPeriodPredictionResult,
                        <:PopulationPredictionResult
                    };
        compound::Bool = false,
        kwargs...
    ) -> Plot

Plot the cumulative returns of a portfolio. If a keyword is provided, it takes precedence over the data in the input objects.

# Arguments

  - `w`: Portfolio weights vector.
  - `X`: Asset returns matrix (observations × assets).
  - `fees`: Optional transaction fees.
  - `ts`: Time axis.
  - `compound`: Whether to compound returns.
  - `pr`: Prior or returns result, extracts `X`, `ts`, `nx` automatically if available.
  - `res`: Extracts `w`, and `fees` if available.
  - `pred`: Predicted portfolio results.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`PlottingOptions`](@ref)
  - [`calc_net_returns`](@ref)
  - [`cumulative_returns`](@ref)
"""
function plot_ptf_cumulative_returns end

## ──────────────────────────────────────────────────────────────────────────────
## Asset cumulative returns
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_asset_cumulative_returns(w, X[, fees]; ts, nx, opts, kwargs...)
    plot_asset_cumulative_returns(w, rd[, fees]; opts, kwargs...)
    plot_asset_cumulative_returns(res, rd; opts, kwargs...)
    plot_asset_cumulative_returns(pred; opts, kwargs...)

Plot the cumulative returns of individual assets, selecting the most relevant via
[`PlottingOptions`](@ref) `N`. Assets beyond the top `N` are aggregated into "Others".

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`PlottingOptions`](@ref)
  - [`number_effective_assets`](@ref)
"""
function plot_asset_cumulative_returns end

## ──────────────────────────────────────────────────────────────────────────────
## Composition
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_composition(w[, nx]; opts, kwargs...)
    plot_composition(res, rd; opts, kwargs...)
    plot_composition(res, pr; opts, kwargs...)
    plot_composition(pred; opts, kwargs...)
    plot_composition(mpred; opts, kwargs...)
    plot_composition(ppred; opts, kwargs...)

Plot portfolio composition as a bar chart of asset weights. Assets beyond the top `N`
(from [`PlottingOptions`](@ref)) are collapsed into an "Others" bar.

`mpred` / `ppred` produce stacked-bar fold compositions (delegates to
[`plot_stacked_bar_composition`](@ref)).

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`PlottingOptions`](@ref)
  - [`number_effective_assets`](@ref)
  - [`plot_stacked_bar_composition`](@ref)
"""
function plot_composition end

## ──────────────────────────────────────────────────────────────────────────────
## Stacked compositions
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_stacked_bar_composition(w[, nx]; opts, kwargs...)
    plot_stacked_bar_composition(res_vec, rd; opts, kwargs...)

Plot portfolio composition as a stacked bar chart. Accepts a matrix, a `VecVecNum`, or a
vector of [`NonFiniteAllocationOptimisationResult`](@ref).

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`PlottingOptions`](@ref)
  - [`plot_stacked_area_composition`](@ref)
"""
function plot_stacked_bar_composition end

"""
    plot_stacked_area_composition(w[, nx]; opts, kwargs...)
    plot_stacked_area_composition(res_vec, rd; opts, kwargs...)

Plot portfolio composition as a stacked area chart. Accepts a matrix, `VecVecNum`, or a
vector of [`NonFiniteAllocationOptimisationResult`](@ref).

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`PlottingOptions`](@ref)
  - [`plot_stacked_bar_composition`](@ref)
"""
function plot_stacked_area_composition end

## ──────────────────────────────────────────────────────────────────────────────
## Risk contribution
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_risk_contribution(r, w, X[, fees]; nx, opts, kwargs...)
    plot_risk_contribution(r, w, rd[, fees]; opts, kwargs...)
    plot_risk_contribution(r, res, rd; opts, kwargs...)
    plot_risk_contribution(r, res, pr; opts, kwargs...)
    plot_risk_contribution(r, pred; opts, kwargs...)

Plot per-asset risk contribution as a bar chart.

  - `opts.delta`: finite-difference step.
  - `opts.marginal`: marginal vs component contribution.
  - `opts.percentage`: normalise to percentages.
  - `opts.N`: top-N assets.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`risk_contribution`](@ref)
  - [`PlottingOptions`](@ref)
"""
function plot_risk_contribution end

## ──────────────────────────────────────────────────────────────────────────────
## Factor risk contribution
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_factor_risk_contribution(r, w, X[, fees]; re, rd, nf, opts, kwargs...)
    plot_factor_risk_contribution(r, res, rd; re, opts, kwargs...)
    plot_factor_risk_contribution(r, pred; re, opts, kwargs...)

Plot per-factor risk contribution as a bar chart, including the constant (idiosyncratic) term.

  - `opts.delta`: finite-difference step.
  - `opts.N`: top-N factors.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`factor_risk_contribution`](@ref)
  - [`PlottingOptions`](@ref)
"""
function plot_factor_risk_contribution end

## ──────────────────────────────────────────────────────────────────────────────
## Dendrogram
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_dendrogram(clr[, nx]; opts, kwargs...)
    plot_dendrogram(clr, rd; opts, kwargs...)
    plot_dendrogram(cle, X[, nx]; dims, opts, kwargs...)
    plot_dendrogram(cle, pr[, nx]; dims, opts, kwargs...)

Plot a hierarchical clustering dendrogram with coloured cluster regions.

  - `clr`: [`AbstractClusteringResult`](@ref) (already computed).
  - `cle`: Clustering estimator — computes clustering from `X` or `pr.X`.
  - `pr`: [`AbstractPriorResult`](@ref) — extracts `X`.
  - `rd`: [`ReturnsResult`](@ref) — extracts `nx` automatically.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`clusterise`](@ref)
  - [`PlottingOptions`](@ref)
"""
function plot_dendrogram end

## ──────────────────────────────────────────────────────────────────────────────
## Cluster heatmap
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_clusters(pe, cle[, rd]; dims, opts, kwargs...)
    plot_clusters(clr, X[, nx]; opts, kwargs...)
    plot_clusters(cle, pr[, nx]; dims, opts, kwargs...)

Plot a reordered correlation/covariance heatmap with flanking dendrograms and coloured
cluster boxes.

  - `pe` + `cle` + `rd`: compute prior and clustering from returns data.
  - `clr` + `X`: use precomputed clustering result and raw returns.
  - `cle` + `pr`: cluster from a prior result.

The `dend_theme`, `hmap_theme`, `color_func`, `line_color`, and `line_width` parameters
are passed as `kwargs...`.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`clusterise`](@ref)
  - [`prior`](@ref)
  - [`PlottingOptions`](@ref)
"""
function plot_clusters end

## ──────────────────────────────────────────────────────────────────────────────
## Drawdowns
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_drawdowns(w, X[, fees]; slv, ts, opts, kwargs...)
    plot_drawdowns(w, rd[, fees]; slv, opts, kwargs...)
    plot_drawdowns(res, rd; slv, opts, kwargs...)
    plot_drawdowns(pred; slv, opts, kwargs...)
    plot_drawdowns(mpred; slv, opts, kwargs...)

Plot portfolio drawdown over time with horizontal lines for AverageDrawdown, UlcerIndex,
DaR, CDaR, MaximumDrawdown, and — when `slv` is provided — EDaR and RLDaR at confidence
level `opts.alpha`.

  - `opts.compound`: compound vs simple drawdowns.
  - `opts.alpha`: confidence level for risk lines.
  - `opts.kappa`: RLDaR shape parameter.
  - `opts.rw`: observation weights.
  - `slv`: solver for EDaR / RLDaR (default `nothing` — those lines are omitted).

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`drawdowns`](@ref)
  - [`PlottingOptions`](@ref)
"""
function plot_drawdowns end

## ──────────────────────────────────────────────────────────────────────────────
## Risk/return scatter
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_measures(w, pr[, fees]; x, y, z, c, slv, opts, kwargs...)
    plot_measures(res_vec, rd; x, y, z, c, slv, opts, kwargs...)
    plot_measures(mpred; x, y, z, c, slv, opts, kwargs...)
    plot_measures(ppred; x, y, z, c, slv, opts, kwargs...)

Scatter plot of risk/return measures across a collection of portfolio weight vectors (e.g.,
an efficient frontier or population).

  - `x`, `y`: risk/return measures for the axes (default `Variance()` and `ExpectedReturn()`).
  - `z`: optional third measure for 3-D scatter.
  - `c`: colour-coding measure (default `ExpectedReturnRiskRatio`).
  - `opts.factory`: if `true`, call `factory` on the measures before evaluating.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`expected_risk`](@ref)
  - [`PlottingOptions`](@ref)
"""
function plot_measures end

## ──────────────────────────────────────────────────────────────────────────────
## Return histogram
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_histogram(w, X[, fees]; slv, opts, kwargs...)
    plot_histogram(w, rd[, fees]; slv, opts, kwargs...)
    plot_histogram(res, rd; slv, opts, kwargs...)
    plot_histogram(pred; slv, opts, kwargs...)
    plot_histogram(mpred; slv, opts, kwargs...)

Plot a histogram of portfolio returns with vertical lines for common tail risk measures and
an optional fitted Normal distribution.

  - `opts.alpha`: confidence level.
  - `opts.kappa`: RLVaR shape parameter.
  - `opts.reference`: overlay fitted Normal.
  - `opts.points`: number of PDF evaluation points (0 = auto).
  - `opts.rw`: observation weights.
  - `slv`: solver for EVaR / RLVaR (default `nothing` — those lines are omitted).

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`PlottingOptions`](@ref)
"""
function plot_histogram end

## ──────────────────────────────────────────────────────────────────────────────
## Network / phylogeny
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_network(pl, X[, nx, w]; threshold, opts, kwargs...)
    plot_network(pl, pr[, nx, w]; opts, kwargs...)
    plot_network(pl, rd[, w]; opts, kwargs...)
    plot_network(pl, res, rd; opts, kwargs...)

Plot the asset network (MST, PMFG, TMFG, or adjacency from a clustering result) as a graph
using `GraphRecipes.graphplot`. Node size is uniform by default; pass `w` to scale node
area proportionally to portfolio weight.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots` and `GraphRecipes`).

# Related

  - [`phylogeny_matrix`](@ref)
  - [`PlottingOptions`](@ref)
"""
function plot_network end

## ──────────────────────────────────────────────────────────────────────────────
## Centrality
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_centrality(cte, X[, nx]; opts, kwargs...)
    plot_centrality(cte, pr[, nx]; opts, kwargs...)
    plot_centrality(cte, rd; opts, kwargs...)
    plot_centrality(cte, res, rd; opts, kwargs...)

Bar chart of asset centrality scores, sorted in descending order.

  - `opts.N`: show top-N assets.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`centrality_vector`](@ref)
  - [`PlottingOptions`](@ref)
"""
function plot_centrality end

## ──────────────────────────────────────────────────────────────────────────────
## Correlation / covariance heatmap
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_correlation(X[, nx]; opts, kwargs...)
    plot_correlation(pr[, nx]; opts, kwargs...)
    plot_correlation(pr, rd; opts, kwargs...)

Standalone correlation (or covariance) heatmap without clustering or dendrograms. If the
input contains a covariance matrix, it is normalised to a correlation matrix before plotting.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`plot_clusters`](@ref)
  - [`PlottingOptions`](@ref)
"""
function plot_correlation end

## ──────────────────────────────────────────────────────────────────────────────
## Expected returns bar chart
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_mu(mu[, nx]; opts, kwargs...)
    plot_mu(pr[, nx]; opts, kwargs...)
    plot_mu(pr, rd; opts, kwargs...)
    plot_mu(res; opts, kwargs...)

Bar chart of per-asset expected returns (μ vector).

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`PlottingOptions`](@ref)
  - [`plot_sigma`](@ref)
"""
function plot_mu end

## ──────────────────────────────────────────────────────────────────────────────
## Asset volatility bar chart
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_sigma(sigma[, nx]; opts, kwargs...)
    plot_sigma(pr[, nx]; opts, kwargs...)
    plot_sigma(pr, rd; opts, kwargs...)

Bar chart of per-asset volatility (√diag(Σ)).

  - `variance::Bool = false` (kwarg): if `true`, show variance (diag(Σ)) instead of standard deviation.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`PlottingOptions`](@ref)
  - [`plot_mu`](@ref)
  - [`plot_correlation`](@ref)
"""
function plot_sigma end

## ──────────────────────────────────────────────────────────────────────────────
## Factor loadings heatmap
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_factor_loadings(M[, nx, nf]; opts, kwargs...)
    plot_factor_loadings(pr[, nx, nf]; opts, kwargs...)
    plot_factor_loadings(pr, rd; opts, kwargs...)

Heatmap of the factor loadings matrix B (assets × factors) from a prior with a regression
model. Uses a diverging colour scale centred at zero.

Requires that `pr.rr` is not `nothing` (i.e. the prior was estimated with a factor model).

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`PlottingOptions`](@ref)
  - [`plot_factor_sigma`](@ref)
  - [`LowOrderPrior`](@ref)
"""
function plot_factor_loadings end

## ──────────────────────────────────────────────────────────────────────────────
## Factor covariance heatmap
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_factor_sigma(f_sigma[, nf]; opts, kwargs...)
    plot_factor_sigma(pr[, nf]; opts, kwargs...)
    plot_factor_sigma(pr, rd; opts, kwargs...)

Correlation/covariance heatmap of the factor covariance matrix (`pr.f_sigma`). Behaves
identically to [`plot_correlation`](@ref) but operates on the factor space.

Requires that `pr.f_sigma` is not `nothing`.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`PlottingOptions`](@ref)
  - [`plot_factor_loadings`](@ref)
  - [`plot_correlation`](@ref)
  - [`LowOrderPrior`](@ref)
"""
function plot_factor_sigma end

## ──────────────────────────────────────────────────────────────────────────────
## Eigenvalue spectrum
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_eigenspectrum(sigma; opts, kwargs...)
    plot_eigenspectrum(pr; opts, kwargs...)
    plot_eigenspectrum(pr, rd; opts, kwargs...)

Bar chart of eigenvalues of the covariance/correlation matrix, sorted in descending order.

When `opts.reference = true` and `N_obs` is provided, overlays the Marchenko-Pastur bulk
upper bound `λ+ = σ̄²(1 + √(N/T))²` as a horizontal line, highlighting noise eigenvalues.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`PlottingOptions`](@ref)
  - [`plot_correlation`](@ref)
"""
function plot_eigenspectrum end

## ──────────────────────────────────────────────────────────────────────────────
## Rolling risk/return measure
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_rolling_measure(r, w, X[, fees]; ts, opts, kwargs...)
    plot_rolling_measure(r, w, rd[, fees]; opts, kwargs...)
    plot_rolling_measure(r, res, rd; opts, kwargs...)
    plot_rolling_measure(r, pred; opts, kwargs...)
    plot_rolling_measure(r, mpred; opts, kwargs...)

Line plot of a risk or return measure evaluated over a rolling window of portfolio returns.

Window size is controlled by `opts.rolling` (0 = auto: ⌈√T⌉).

The risk measure `r` may embed its own solver (e.g. `EntropicValueatRisk(; slv=...)`), so
no separate solver argument is needed here.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`PlottingOptions`](@ref)
  - [`expected_risk`](@ref)
"""
function plot_rolling_measure end

## ──────────────────────────────────────────────────────────────────────────────
## Weight stability across folds
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_weight_stability(mpred; opts, kwargs...)
    plot_weight_stability(ppred; opts, kwargs...)

Box plot of per-asset weight distributions across cross-validation folds.

  - `opts.N`: top-N assets by mean absolute weight (default: all).
  - For `PopulationPredictionResult`, pools weights from all population members.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`PlottingOptions`](@ref)
  - [`MultiPeriodPredictionResult`](@ref)
  - [`PopulationPredictionResult`](@ref)
"""
function plot_weight_stability end

## ──────────────────────────────────────────────────────────────────────────────
## Cross-validation scores
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_cv_scores(scores[, labels]; opts, kwargs...)
    plot_cv_scores(r, mpred; opts, kwargs...)
    plot_cv_scores(r, ppred; opts, kwargs...)

Bar chart of cross-validation scores (one bar per fold or population member).

  - For `MultiPeriodPredictionResult`: one score per fold.
  - For `PopulationPredictionResult`: one score per population member.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`PlottingOptions`](@ref)
  - [`expected_risk`](@ref)
  - [`MultiPeriodPredictionResult`](@ref)
  - [`PopulationPredictionResult`](@ref)
"""
function plot_cv_scores end

## ──────────────────────────────────────────────────────────────────────────────
## Portfolio turnover
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_turnover(w_series[; ts, opts, kwargs...])
    plot_turnover(mpred; opts, kwargs...)

Line plot of portfolio turnover (L1 weight change) over time.

Turnover at step `t` is defined as `∑ |w_t − w_{t−1}|`.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`PlottingOptions`](@ref)
  - [`MultiPeriodPredictionResult`](@ref)
"""
function plot_turnover end

## ──────────────────────────────────────────────────────────────────────────────
## Composite prior dashboard
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_prior(pr; opts, kwargs...)
    plot_prior(pr, rd; opts, kwargs...)

Three-panel composite plot summarising a prior result:

 1. Expected returns bar chart ([`plot_mu`](@ref)).
 2. Asset volatility bar chart ([`plot_sigma`](@ref)).
 3. Correlation heatmap ([`plot_correlation`](@ref)).

When `rd` is provided, asset names (`rd.nx`) are used throughout.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`PlottingOptions`](@ref)
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
    plot_factor_mu(f_mu[, nf]; opts, kwargs...)
    plot_factor_mu(pr[, nf]; opts, kwargs...)
    plot_factor_mu(pr, rd; opts, kwargs...)
    plot_factor_mu(res[, rd]; opts, kwargs...)
    plot_factor_mu(pred[, rd]; opts, kwargs...)

Bar chart of per-factor expected returns (f_μ vector from a factor model prior).

Requires that the prior was estimated with a factor model (`pr.f_mu` is not `nothing`).

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`PlottingOptions`](@ref)
  - [`plot_mu`](@ref)
  - [`LowOrderPrior`](@ref)
"""
function plot_factor_mu end

## ──────────────────────────────────────────────────────────────────────────────
## Benchmark overlay
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_benchmark(w, X, B[, fees]; ts, nb, opts, kwargs...)
    plot_benchmark(w, rd[, fees]; opts, kwargs...)
    plot_benchmark(res, rd; opts, kwargs...)
    plot_benchmark(pred; opts, kwargs...)
    plot_benchmark(mpred; opts, kwargs...)

Overlay portfolio cumulative returns against one or more benchmark return series from `rd.B`.

`opts.compound` controls compounding for both the portfolio and benchmark series.
Throws `ArgumentError` if `rd.B` is `nothing`.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`PlottingOptions`](@ref)
  - [`plot_ptf_cumulative_returns`](@ref)
  - [`ReturnsResult`](@ref)
"""
function plot_benchmark end

## ──────────────────────────────────────────────────────────────────────────────
## Coskewness heatmap
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_coskewness(sk[, nx]; opts, kwargs...)
    plot_coskewness(pr[, nx]; opts, kwargs...)
    plot_coskewness(pr, rd; opts, kwargs...)
    plot_coskewness(res[, rd]; opts, kwargs...)
    plot_coskewness(pred[, rd]; opts, kwargs...)

Heatmap of the coskewness matrix (N × N²) from a [`HighOrderPrior`](@ref).

Uses a diverging colour scale centred at zero.
Requires that `pr.sk` is not `nothing` (i.e. the prior was estimated with higher moments).

The factor variant uses `pr.f_sk` when available.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`PlottingOptions`](@ref)
  - [`plot_cokurtosis`](@ref)
  - [`HighOrderPrior`](@ref)
"""
function plot_coskewness end

## ──────────────────────────────────────────────────────────────────────────────
## Cokurtosis eigenspectrum / heatmap
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_cokurtosis(kt[, nx]; opts, kwargs...)
    plot_cokurtosis(pr[, nx]; opts, kwargs...)
    plot_cokurtosis(pr, rd; opts, kwargs...)
    plot_cokurtosis(res[, rd]; opts, kwargs...)
    plot_cokurtosis(pred[, rd]; opts, kwargs...)

Eigenvalue spectrum of the cokurtosis matrix (N² × N²) from a [`HighOrderPrior`](@ref).

When `opts.reference = true`, overlays the Marchenko-Pastur noise bulk bound.
Pass `heatmap = true` as a kwarg to show the raw heatmap instead (only recommended for small N).

Requires that `pr.kt` is not `nothing`.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`PlottingOptions`](@ref)
  - [`plot_coskewness`](@ref)
  - [`HighOrderPrior`](@ref)
"""
function plot_cokurtosis end

## ──────────────────────────────────────────────────────────────────────────────
## Portfolio dashboard (multi-panel composite)
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_portfolio_dashboard(res, rd; r, slv, opts, kwargs...)
    plot_portfolio_dashboard(pred; r, slv, opts, kwargs...)

Four-panel composite plot for a single optimisation result:

 1. Portfolio composition ([`plot_composition`](@ref)).
 2. Cumulative returns ([`plot_ptf_cumulative_returns`](@ref)).
 3. Asset risk contribution ([`plot_risk_contribution`](@ref)).
 4. Drawdowns ([`plot_drawdowns`](@ref)).

`r` selects the risk measure for panels 3 and 4 (default `Variance()`).

Note: panel 3 requires raw asset returns. Pass `rd::ReturnsResult` (original returns),
not a `PredictionResult`, for the risk contribution panel.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`PlottingOptions`](@ref)
  - [`plot_composition`](@ref)
  - [`plot_ptf_cumulative_returns`](@ref)
  - [`plot_risk_contribution`](@ref)
  - [`plot_drawdowns`](@ref)
"""
function plot_portfolio_dashboard end

## ──────────────────────────────────────────────────────────────────────────────
## Cross-validation dashboard (multi-panel composite)
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_cv_dashboard(mpred; r, opts, kwargs...)

Four-panel composite plot for a walk-forward cross-validation result:

 1. Stacked-bar fold compositions ([`plot_composition`](@ref)).
 2. Fold-shaded cumulative returns ([`plot_ptf_cumulative_returns`](@ref)).
 3. Turnover per fold ([`plot_turnover`](@ref)).
 4. Weight stability box plot ([`plot_weight_stability`](@ref)).

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`PlottingOptions`](@ref)
  - [`MultiPeriodPredictionResult`](@ref)
  - [`plot_turnover`](@ref)
  - [`plot_weight_stability`](@ref)
"""
function plot_cv_dashboard end

## ──────────────────────────────────────────────────────────────────────────────
## Efficient frontier
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_efficient_frontier(res_vec, pr[, fees]; x, y, c, slv, opts, kwargs...)
    plot_efficient_frontier(res_vec, rd; ...)

Sort a collection of portfolio results by risk (`x`), connect them with a line to
trace the efficient frontier, and optionally annotate the **minimum-risk** and
**maximum-Sharpe** portfolios.

  - `x`: risk measure for the horizontal axis (default `Variance()`).
  - `y`: return measure for the vertical axis (default `ExpectedReturn()`).
  - `c`: colour-coding measure (default Sharpe ratio derived from `x`).
  - `opts.factory`: if `true` (default), call `factory` on the measures before evaluating.
  - `min_risk::Bool = true` (kwarg): overlay a star marker at the minimum-risk portfolio.
  - `max_score::Bool = true` (kwarg): overlay a star marker at the portfolio that maximises the colour-coding measure.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`plot_measures`](@ref)
  - [`PlottingOptions`](@ref)
  - [`expected_risk`](@ref)
"""
function plot_efficient_frontier end

## ──────────────────────────────────────────────────────────────────────────────
## Performance summary
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_performance_summary(w, X[, fees]; periods_per_year, opts, kwargs...)
    plot_performance_summary(w, rd[, fees]; opts, kwargs...)
    plot_performance_summary(res, rd; opts, kwargs...)
    plot_performance_summary(pred; opts, kwargs...)
    plot_performance_summary(mpred; opts, kwargs...)

Bar chart of annualised portfolio performance metrics:
annualised return, annualised volatility, Sharpe ratio, Sortino ratio, Calmar ratio,
maximum drawdown %, and CVaR % (at `opts.alpha`).

  - `periods_per_year`: trading periods per year used for annualisation (default `252`).
  - `opts.alpha`: tail probability for CVaR (default `0.05`).

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`PlottingOptions`](@ref)
  - [`calc_net_returns`](@ref)
"""
function plot_performance_summary end

## ──────────────────────────────────────────────────────────────────────────────
## Rolling drawdown evolution
## ──────────────────────────────────────────────────────────────────────────────
"""
    plot_rolling_drawdowns(w, X[, fees]; ts, opts, kwargs...)
    plot_rolling_drawdowns(w, rd[, fees]; opts, kwargs...)
    plot_rolling_drawdowns(res, rd; opts, kwargs...)
    plot_rolling_drawdowns(pred; opts, kwargs...)
    plot_rolling_drawdowns(mpred; opts, kwargs...)

Line plot of the rolling maximum drawdown over a sliding window.

Window size is controlled by `opts.rolling` (0 = auto: ⌈√T⌉).
`opts.compound` selects compound vs simple drawdowns.

Implemented by `PortfolioOptimisersPlotsExt` (requires `StatsPlots`).

# Related

  - [`PlottingOptions`](@ref)
  - [`drawdowns`](@ref)
  - [`plot_drawdowns`](@ref)
  - [`plot_rolling_measure`](@ref)
"""
function plot_rolling_drawdowns end

## ────────────────────────────────────────────────────────────────────────────
## Internal helpers (no Plots.jl dependency)
## ────────────────────────────────────────────────────────────────────────────
function _pred_rd_to_matrix(rd::PredictionReturnsResult{<:Any, <:VecNum})
    return SingletonVector{Int}(), reshape(rd.X, :, 1)
end
function _pred_rd_to_matrix(rd::PredictionReturnsResult{<:Any, <:VecVecNum})
    return SingletonVector{Int}(), [reshape(ret, :, 1) for ret in rd.X]
end
# Select top-N assets by absolute weight magnitude.
# Returns (N_selected, sorted_idx) where idx is sorted descending by |w|.
function _relevant_assets(w::VecNum, M::Integer, N_opt::Option{<:Number})
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
# Rolling window evaluation of a risk measure.
# Returns a vector of length T - window + 1 for t = window:T.
function _rolling_window_measure(r::AbstractBaseRiskMeasure, w::VecNum, X::MatNum,
                                 fees::Option{<:Fees}, window::Integer)
    T = size(X, 1)
    return [expected_risk(r, w, view(X, (t - window + 1):t, :), fees) for t in window:T]
end

export PlottingOptions, plot_ptf_cumulative_returns, plot_asset_cumulative_returns,
       plot_composition, plot_stacked_bar_composition, plot_stacked_area_composition,
       plot_dendrogram, plot_clusters, plot_drawdowns, plot_risk_contribution,
       plot_factor_risk_contribution, plot_measures, plot_histogram, plot_network,
       plot_centrality, plot_correlation, plot_mu, plot_sigma, plot_factor_loadings,
       plot_factor_sigma, plot_eigenspectrum, plot_rolling_measure, plot_weight_stability,
       plot_cv_scores, plot_turnover, plot_prior, plot_factor_mu, plot_benchmark,
       plot_coskewness, plot_cokurtosis, plot_portfolio_dashboard, plot_cv_dashboard,
       plot_efficient_frontier, plot_performance_summary, plot_rolling_drawdowns
