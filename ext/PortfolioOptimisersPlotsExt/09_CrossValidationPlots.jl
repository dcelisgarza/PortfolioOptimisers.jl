## plot_cv_scores
function PortfolioOptimisers.plot_cv_scores(r::PortfolioOptimisers.BaseRM_VecBaseRM,
                                            mpred::MultiPeriodPredictionResult; kwargs...)
    scores = [expected_risk(r, p) for p in mpred.pred]
    return PortfolioOptimisers.plot_cv_scores(scores, 1:length(scores); kwargs...)
end
function PortfolioOptimisers.plot_cv_scores(r::PortfolioOptimisers.BaseRM_VecBaseRM,
                                            ppred::PopulationPredictionResult; kwargs...)
    scores = [expected_risk(r, m) for m in ppred.pred]
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

function PortfolioOptimisers.plot_weight_stability(mpred::MultiPeriodPredictionResult;
                                                   N::Option{<:Number} = nothing, kwargs...)
    folds = mpred.pred
    w_mat = hcat(getproperty.(getproperty.(folds, :res), :w)...)
    nx = let rd1 = folds[1].rd
        isnothing(rd1.nx) ? string.(1:size(w_mat, 1)) : string.(rd1.nx)
    end
    M, K = size(w_mat)
    mean_abs = vec(mean(abs, w_mat; dims = 2))
    N, idx = relevant_assets(mean_abs, M, N)
    top_idx = sort(view(idx, 1:N))
    labels = nx[top_idx]
    data = w_mat[top_idx, :]
    return boxplot(data'; xticks = (1:N, labels), title = "Weight Stability",
                   ylabel = "Weight", legend = false, xrotation = 60, kwargs...)
end
function PortfolioOptimisers.plot_weight_stability(ppred::PopulationPredictionResult;
                                                   N::Option{<:Number} = nothing, kwargs...)
    members = ppred.pred
    w_mat = hcat(map(members) do m
                     if isa(m, PredictionResult)
                         m.res.w
                     else
                         vec(mean(hcat(getproperty.(getproperty.(m.pred, :res), :w)...);
                                  dims = 2))
                     end
                 end...)
    nx = let rd1 = isa(members[1], PredictionResult) ? members[1].rd : members[1].pred[1].rd
        isnothing(rd1.nx) ? string.(1:size(w_mat, 1)) : string.(rd1.nx)
    end
    M, K = size(w_mat)
    mean_abs = vec(mean(abs, w_mat; dims = 2))
    N, idx = relevant_assets(mean_abs, M, N)
    top_idx = sort(view(idx, 1:N))
    labels = nx[top_idx]
    data = w_mat[top_idx, :]
    return boxplot(data'; xticks = (1:N, labels), title = "Population Weight Stability",
                   ylabel = "Weight", legend = false, xrotation = 60, kwargs...)
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
