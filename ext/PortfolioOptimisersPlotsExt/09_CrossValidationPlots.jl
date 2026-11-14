## plot_cv_scores
function PortfolioOptimisers.plot_cv_scores(r::PortfolioOptimisers.BaseRM_VecBaseRM,
                                            mpred::MultiPeriodPredictionResult;
                                            sca::Option{<:Scalariser} = nothing, kwargs...)
    scores = [expected_risk(sca, r, p) for p in mpred.pred]
    return PortfolioOptimisers.plot_cv_scores(scores, 1:length(scores); kwargs...)
end
function PortfolioOptimisers.plot_cv_scores(r::PortfolioOptimisers.BaseRM_VecBaseRM,
                                            ppred::PopulationPredictionResult;
                                            sca::Option{<:Scalariser} = nothing, kwargs...)
    scores = [expected_risk(sca, r, m) for m in ppred.pred]
    return PortfolioOptimisers.plot_cv_scores(scores, 1:length(scores); kwargs...)
end
## plot_turnover
function PortfolioOptimisers.plot_turnover(mpred::MultiPeriodPredictionResult; kwargs...)
    folds = mpred.pred
    w_series = getproperty.(getproperty.(folds, :res), :w)
    ts = if isnothing(folds[1].rd.ts)
        1:length(w_series)
    else
        [f.rd.ts[end] for f in folds]
    end
    return PortfolioOptimisers.plot_turnover(w_series; ts = ts, kwargs...)
end
## ────────────────────────────────────────────────────────────────────────────
## Weight stability across folds
## ────────────────────────────────────────────────────────────────────────────

# The rows of `prediction_weights` sit on the union of the names of the folds or the paths, so
# a fold that holds other assets than the first fold boxes each weight under its own name. A
# row that is `NaN` at every name is a portfolio that failed to solve, and it holds no weight,
# so it is absent from every box. With no row left, the figure draws the axes alone.
function weight_stability_boxplot(nx::AbstractVector, W::MatNum, N::Option{<:Number},
                                  title::AbstractString; kwargs...)
    W = W[[!all(isnan, r) for r in eachrow(W)], :]
    if isempty(W)
        return plot(; title = title, ylabel = "Weight", legend = false, kwargs...)
    end
    M = size(W, 2)
    mean_abs = vec(mean(abs, W; dims = 1))
    N, idx = relevant_assets(mean_abs, M, N)
    top_idx = sort(view(idx, 1:N))
    return boxplot(W[:, top_idx]; xticks = (1:N, string.(nx[top_idx])), title = title,
                   ylabel = "Weight", legend = false, xrotation = 60, kwargs...)
end
function PortfolioOptimisers.plot_weight_stability(mpred::MultiPeriodPredictionResult;
                                                   N::Option{<:Number} = nothing, kwargs...)
    nx, W, _ = prediction_weights(mpred)
    return weight_stability_boxplot(nx, W, N, "Weight Stability"; kwargs...)
end
function PortfolioOptimisers.plot_weight_stability(ppred::PopulationPredictionResult;
                                                   N::Option{<:Number} = nothing, kwargs...)
    nx, W, _ = prediction_weights(ppred)
    return weight_stability_boxplot(nx, W, N, "Population Weight Stability"; kwargs...)
end
## ────────────────────────────────────────────────────────────────────────────
## Cross-validation scores
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_cv_scores(scores::AbstractVector{<:Number},
                                            labels::AbstractVector = 1:length(scores);
                                            kwargs...)
    n = length(scores)
    return bar(scores; xticks = (1:n, string.(labels)), title = "CV Scores",
               ylabel = "Score", xlabel = "Fold / Member", legend = false, xrotation = 45,
               kwargs...)
end
## ────────────────────────────────────────────────────────────────────────────
## Portfolio turnover
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_turnover(w_series::AbstractVector{<:VecNum};
                                           ts::AbstractVector = 1:length(w_series),
                                           kwargs...)
    turnover = view(PortfolioOptimisers.calc_turnover(w_series), 2:length(w_series))
    return plot(@view(ts[2:end]), turnover; title = "Portfolio Turnover",
                ylabel = "Turnover (∑|Δw|)", xlabel = "Date", legend = false, linewidth = 2,
                kwargs...)
end
## ────────────────────────────────────────────────────────────────────────────
## Cross-validation dashboard (multi-panel composite)
## ────────────────────────────────────────────────────────────────────────────

function PortfolioOptimisers.plot_cv_dashboard(mpred::MultiPeriodPredictionResult;
                                               N::Option{<:Number} = nothing,
                                               compound::Bool = false, kwargs...)
    p1 = PortfolioOptimisers.plot_composition(mpred; N = N)
    p2 = PortfolioOptimisers.plot_portfolio_cumulative_returns(mpred; compound = compound)
    p3 = PortfolioOptimisers.plot_turnover(mpred)
    p4 = PortfolioOptimisers.plot_weight_stability(mpred; N = N)
    return plot(p1, p2, p3, p4; layout = (2, 2), size = (1200, 800), kwargs...)
end
