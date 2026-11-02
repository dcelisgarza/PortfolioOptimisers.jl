## default_report_figures
function PortfolioOptimisers.default_report_figures()
    return (;
            overview = ["Cumulative returns" => report_cumulative_returns,
                        "Risk and return" => report_risk_return],
            entry = ["Composition" => report_composition, "Drawdowns" => report_drawdowns,
                     "Histogram of the returns" => report_histogram,
                     "Risk contribution of the CVaR" => report_risk_contribution])
end
# The return series of a prediction, or of the first path of a population. Its type says
# whether the optimiser gave one portfolio, a vector, or many members, a vector of vectors.
function report_series(pred::PredRes_MultiPredRes)
    return first(regret_series(pred))
end
function report_series(ppred::PopulationPredictionResult)
    return report_series(first(ppred.pred))
end
# A figure of one portfolio: drawn for one member, and `nothing` for many, because the plots
# of a prediction read only its first member.
function member_figure(f, pred::AbstractPredictionResult, ::VecNum)
    return f(pred)
end
function member_figure(f, ::AbstractPredictionResult, ::VecVecNum)
    return nothing
end
function report_composition(pred::AbstractPredictionResult)
    return member_figure(plot_composition, pred, report_series(pred))
end
function report_drawdowns(pred::AbstractPredictionResult)
    return member_figure(plot_drawdowns, pred, report_series(pred))
end
function report_histogram(pred::AbstractPredictionResult)
    return member_figure(plot_histogram, pred, report_series(pred))
end
# The risk contribution reads the asset returns of a single fold, which only a Held Weights
# record keeps.
function report_risk_contribution(pred::PredictionResult)
    return risk_contribution_figure(pred, pred.hw)
end
function report_risk_contribution(::AbstractPredictionResult)
    return nothing
end
function risk_contribution_figure(::PredictionResult, ::Nothing)
    return nothing
end
function risk_contribution_figure(pred::PredictionResult, ::HeldWeightsResult)
    return member_figure(p -> plot_risk_contribution(ConditionalValueatRisk(), p), pred,
                         report_series(pred))
end
# Every series of a prediction with its timestamps: one for each member, and for a
# population the series of each path in turn.
function report_series_list(pred::PredRes_MultiPredRes)
    X, ts = regret_series(pred)
    return series_members(X, something(ts, 1:length(member_vector(X, 1))))
end
function report_series_list(ppred::PopulationPredictionResult)
    return reduce(vcat, [report_series_list(p) for p in ppred.pred])
end
function series_members(X::VecNum, ts::AbstractVector)
    return [(X, ts)]
end
function series_members(X::VecVecNum, ts::AbstractVector)
    return [(x, ts) for x in X]
end
# The candidates can cover different rows, so each series is drawn on its own timestamps.
# One colour and one legend label for each candidate, and its paths and members share both.
function report_cumulative_returns(preds::AbstractVector{<:AbstractPredictionResult},
                                   names::AbstractVector{<:AbstractString})
    plt = plot(; title = "Cumulative Returns", xlabel = "Date",
               ylabel = "Simple Cumulative Returns", legend = :outertopright)
    for k in eachindex(preds, names)
        for (i, (ret, ts)) in enumerate(report_series_list(preds[k]))
            plot!(plt, ts, cumulative_returns(ret, false); seriescolor = k,
                  label = i == 1 ? names[k] : "", linealpha = i == 1 ? 1.0 : 0.4)
        end
    end
    return plt
end
# One point for each series of a candidate, through `expected_risk` of the prediction, so a
# population gives one point for each path.
function report_risk_return(preds::AbstractVector{<:AbstractPredictionResult},
                            names::AbstractVector{<:AbstractString})
    plt = plot(; title = "Risk and Return", xlabel = "CVaR", ylabel = "Mean return",
               legend = :outertopright)
    for k in eachindex(preds, names)
        scatter!(plt, risk_points(ConditionalValueatRisk(), preds[k]),
                 risk_points(MeanReturn(), preds[k]); seriescolor = k, label = names[k])
    end
    return plt
end
# The value of a measure on each series of a prediction, as one flat vector: a fold gives one
# value, a population one for each path, and a prediction with many members one for each.
function risk_points(r, pred::AbstractPredictionResult)
    return collect(Iterators.flatten(vcat(expected_risk(r, pred))))
end
