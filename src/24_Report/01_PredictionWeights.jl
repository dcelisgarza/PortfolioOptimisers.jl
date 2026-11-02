"""
    prediction_weights(pred::PredictionResult, j::Integer = 1)
    prediction_weights(pred::MultiPeriodPredictionResult, j::Integer = 1)
    prediction_weights(ppred::PopulationPredictionResult, j::Integer = 1)

The weights of member `j` of a prediction, on asset names, with one row for each portfolio that the prediction holds.

A single fold holds one portfolio, its target. A path holds one portfolio for each fold, its target at the start of the fold. A population holds one portfolio for each path, the mean of the fold targets of the path. Every target comes from [`fold_target`](@ref), on the names `rd.nx` of its fold, so the weights sit on the universe that [`performance_summary`](@ref) reads for `n_eff` and `turnover`. The folds of a path and the paths of a population can hold different assets, so [`weights_by_name`](@ref) puts the rows on the union of their names, with a zero where a row does not hold an asset. A fold whose returns data carries no names takes the positions of its weights as names.

[`portfolio_report`](@ref) writes the weights table of an entry from this function. The composition plots of a path and of a population draw the same rows.

# Algorithm

 1. For a single fold, read the target `w` of member `j`, and give one row with the label `"weight"`.
 2. For a path, read the target of member `j` of each fold, and embed the targets on the union of the fold names. The label of a row is the first timestamp of its fold, or `"fold k"` when the fold carries no timestamps.
 3. For a population, read the rows of each path as in steps 1 and 2, and take their mean over the rows. Embed the means on the union of the path names. The label of a row is the key of its path, [`population_path_key`](@ref).

# Arguments

  - `pred`, `ppred`: The prediction result.
  - `j`: The index of the member.

# Validation

  - A population holds at least one path. Otherwise the function raises an [`IsEmptyError`](@ref).

# Returns

  - `names::Vector`: The asset names of the columns.
  - `W::MatNum`: One row for each portfolio, one column for each name.
  - `labels::Vector{String}`: The label of each row.

# Related

  - [`fold_target`](@ref)
  - [`weights_by_name`](@ref)
  - [`prediction_members`](@ref)
  - [`portfolio_report`](@ref)
"""
function prediction_weights(pred::PredictionResult, j::Integer = 1)
    w = fold_target(pred, j)
    return something(pred.rd.nx, string.(eachindex(w))), permutedims(w), ["weight"]
end
function prediction_weights(pred::MultiPeriodPredictionResult, j::Integer = 1)
    folds = pred.pred
    ws = [fold_target(f, j) for f in folds]
    nxs = [something(folds[k].rd.nx, string.(eachindex(ws[k]))) for k in eachindex(folds)]
    names, W = weights_by_name(ws, nxs)
    return names, W, [fold_label(folds[k].rd.ts, k) for k in eachindex(folds)]
end
function prediction_weights(ppred::PopulationPredictionResult, j::Integer = 1)
    paths = ppred.pred
    @argcheck(!isempty(paths),
              IsEmptyError("the population holds no path, so it holds no weights"))
    rows = [prediction_weights(p, j) for p in paths]
    names, W = weights_by_name([vec(mean(r[2]; dims = 1)) for r in rows], first.(rows))
    return names, W, [string(population_path_key(paths[k], k)) for k in eachindex(paths)]
end
"""
    fold_label(ts::Nothing, k::Integer)
    fold_label(ts::AbstractVector, k::Integer)

The label of fold `k` of a path: the first timestamp of the fold, or `"fold k"` when the fold carries no timestamps.

# Arguments

  - `ts`: The timestamps of the fold, or `nothing`.
  - `k`: The position of the fold in the path.

# Returns

  - `label::String`: The label.

# Related

  - [`prediction_weights`](@ref)
"""
function fold_label(::Nothing, k::Integer)
    return "fold $(k)"
end
function fold_label(ts::AbstractVector, ::Integer)
    return string(first(ts))
end
"""
    prediction_members(pred::PredictionResult)
    prediction_members(pred::MultiPeriodPredictionResult)
    prediction_members(ppred::PopulationPredictionResult)

The number of members of a prediction: the number of portfolios that its optimiser gives at each fold.

An optimiser that gives one portfolio has one member. An optimiser that gives many portfolios, such as a frontier, holds one weight vector for each member. The folds of a path and the paths of a population come from one optimiser, so the first fold gives the count.

# Arguments

  - `pred`, `ppred`: The prediction result.

# Validation

  - A population holds at least one path. Otherwise the function raises an [`IsEmptyError`](@ref).

# Returns

  - `n::Int`: The number of members.

# Related

  - [`prediction_weights`](@ref)
  - [`member_vector`](@ref)
"""
function prediction_members(pred::PredictionResult)
    return length(held_weight_members(pred.res.w))
end
function prediction_members(pred::MultiPeriodPredictionResult)
    return prediction_members(first(pred.pred))
end
function prediction_members(ppred::PopulationPredictionResult)
    @argcheck(!isempty(ppred.pred),
              IsEmptyError("the population holds no path, so it holds no members"))
    return prediction_members(first(ppred.pred))
end
