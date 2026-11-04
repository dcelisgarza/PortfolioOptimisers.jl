"""
    portfolio_report(preds::AbstractVector{<:AbstractPredictionResult};
                     names::Option{<:AbstractVector{<:AbstractString}} = nothing,
                     title::AbstractString = "Portfolio report",
                     groups::Option{<:AbstractVector} = nothing,
                     meta::Option{<:AbstractVector} = nothing,
                     allocs::Option{<:AbstractVector} = nothing,
                     overview_figures = (), entry_figures = (),
                     weight_threshold::Real = 1e-6, kwargs...) -> Report
    portfolio_report(pred::AbstractPredictionResult; name::AbstractString = "Portfolio",
                     group::Option{<:AbstractString} = nothing,
                     meta::Option{<:AbstractVector{<:Pair}} = nothing, alloc = nothing,
                     kwargs...) -> Report

Build a Portfolio Report that compares prediction results: one entry for each prediction, and an overview table of their statistics.

Each element of `preds` is one candidate, and one entry of the report. A candidate can be a single fold, a [`PredictionResult`](@ref), a walk-forward path, a [`MultiPeriodPredictionResult`](@ref), or a population of paths, a [`PopulationPredictionResult`](@ref). The builder does not split a candidate: the paths of a population and the members of a prediction whose optimiser gives many portfolios are rows inside its entry. The second method builds the report of one prediction.

# The overview

The overview holds the table of [`DataFrames.DataFrame(ps::PerformanceSummaryResult)`](@ref) of `performance_summary(preds; names, kwargs...)`, with one row for each candidate, path and member. Every keyword of [`performance_summary`](@ref) other than `names`, such as `measures`, `rf`, `ppy` and `benchmark`, passes through `kwargs`. The report computes no figure of its own, so each statistic has one source. With `groups`, a `group` column comes first. A click on a row of the table in the HTML file shows the entry of its candidate. `overview_figures` adds one figure for each of its pairs, from the vector of predictions and their names.

# The sections of an entry

 1. **Statistics.** A candidate with one row of the overview gives a table of two columns, the figure and its value. A candidate with many rows gives its rows as a sortable table. A population also gives the distribution of each figure over its paths: the count of finite values, the minimum, the 25 % quantile, the median, the 75 % quantile and the maximum. The distribution describes the columns of the summary, and it adds no statistic.
 2. **Weights.** One table for each member, from [`prediction_weights`](@ref). A single fold gives one column of weights. A path gives one column for each fold, and a population one column for each path, after a `mean` column. The rows are sorted by the absolute value of the first column of weights. A row whose largest absolute weight is less than `weight_threshold` is not shown, and the caption gives the count of such rows.
 3. **Allocation**, when the entry has a finite allocation result: the shares, the cost and the realised weight of each asset, with the cash that is left and the fees in the caption. A finite allocation is the allocation of one portfolio. For a path, it is the allocation of the target of the last fold, the portfolio to hold now.
 4. **Figures**, when `entry_figures` gives one: one figure for each of its pairs, from the prediction.

# Figures

`overview_figures` and `entry_figures` are collections of pairs `caption => f`. An overview figure is `f(preds, names)`, so a legend can name the predictions, and an entry figure is `f(pred)`. Each gives a figure, or `nothing`, which adds no figure. A figure is any object that [`ReportFigure`](@ref) takes. The function takes the prediction as it is, so it can dispatch on its type. Both collections are empty by default, so a report with no plotting package is complete. With the plotting extension loaded, [`default_report_figures`](@ref) gives the default figures. A function that throws stops the build with its own error, and the builder does not hide the failure.

# Arguments

  - `preds`: The predictions, one for each candidate.
  - `pred`: One prediction.
  - `names`: One name for each prediction, or `nothing` for `"Portfolio 1"`, `"Portfolio 2"`, …. The names must be unique.
  - `name`: The name of the prediction of the second method.
  - `title`: The title of the report.
  - `groups`, `group`: The group of each prediction, an `AbstractString` or `nothing`, or `nothing` for no groups. A group is a label of the caller.
  - `meta`: The labels of each entry, each a vector of pairs `label => value` or `nothing`, or `nothing` for none. The second method takes the labels of its one entry.
  - `allocs`, `alloc`: The finite allocation result of each prediction, a [`FiniteAllocationOptimisationResult`](@ref) or `nothing`, or `nothing` for none.
  - `overview_figures`, `entry_figures`: The figures, as stated in `# Figures`.
  - `weight_threshold`: The least absolute weight that a row of a weights table shows.
  - `kwargs...`: The keywords of [`performance_summary`](@ref).

# Validation

  - `preds` is not empty. Otherwise the function raises an [`IsEmptyError`](@ref).
  - `names`, `groups`, `meta` and `allocs` have the length of `preds`. Otherwise the function raises a `DimensionMismatch` that names the keyword.
  - The names are unique. Otherwise the function raises an `ArgumentError`.
  - Each element of `overview_figures` and `entry_figures` is a pair whose first part is an `AbstractString`. Otherwise the function raises an `ArgumentError`.
  - `weight_threshold` is finite and not negative. Otherwise the function raises a `DomainError`.
  - The length of the shares of an allocation is the number of assets of the portfolio that it allocates. Otherwise the function raises a `DimensionMismatch`.
  - A population has no allocation, because it holds no single portfolio. Otherwise the function raises an `ArgumentError`.
  - The rules of [`performance_summary`](@ref).

# Returns

  - `r::Report`: The report.

# Related

  - [`Report`](@ref)
  - [`write_report`](@ref)
  - [`performance_summary`](@ref)
  - [`prediction_weights`](@ref)
  - [`default_report_figures`](@ref)
"""
function portfolio_report(preds::AbstractVector{<:AbstractPredictionResult};
                          names::Option{<:AbstractVector{<:AbstractString}} = nothing,
                          title::AbstractString = "Portfolio report",
                          groups::Option{<:AbstractVector} = nothing,
                          meta::Option{<:AbstractVector} = nothing,
                          allocs::Option{<:AbstractVector} = nothing, overview_figures = (),
                          entry_figures = (), weight_threshold::Real = 1e-6,
                          kwargs...)::Report
    @argcheck(!isempty(preds),
              IsEmptyError("a report needs at least one prediction, and the vector is empty"))
    ps = collect(preds)
    n = length(ps)
    nm = forecast_summary_names(names, n, "Portfolio")
    assert_report_length(groups, n, :groups)
    assert_report_length(meta, n, :meta)
    assert_report_length(allocs, n, :allocs)
    @argcheck(isfinite(weight_threshold) && weight_threshold >= zero(weight_threshold),
              DomainError(weight_threshold,
                          "weight_threshold must be finite and not negative"))
    rep = unique(filter(x -> count(==(x), nm) > 1, nm))
    @argcheck(isempty(rep),
              ArgumentError("the names of the predictions of a report must be unique, and these names repeat: $(join(repr.(rep), ", "))"))
    ofigs = report_figure_pairs(overview_figures, :overview_figures)
    efigs = report_figure_pairs(entry_figures, :entry_figures)
    df = DataFrames.DataFrame(performance_summary(ps; names = nm, kwargs...))
    entries = [ReportEntry(nm[k],
                           entry_sections(ps[k], df[df.name .== nm[k], :],
                                          report_item(allocs, k), weight_threshold, efigs);
                           group = report_item(groups, k), meta = report_item(meta, k))
               for k in 1:n]
    overview = AbstractReportBlock[ReportTable(overview_table(df, nm, groups);
                                               caption = "One row for each portfolio, path and member. A click on a row shows its portfolio.",
                                               sortable = true, key = :name)]
    append!(overview, report_figures(ofigs, ps, nm))
    return Report(title, entries; overview = overview)
end
function portfolio_report(pred::AbstractPredictionResult;
                          name::AbstractString = "Portfolio",
                          group::Option{<:AbstractString} = nothing,
                          meta::Option{<:AbstractVector{<:Pair}} = nothing, alloc = nothing,
                          kwargs...)::Report
    # A keyword does not dispatch: a report of one ungrouped entry has no `group` column.
    groups = isnothing(group) ? nothing : [group]
    return portfolio_report([pred]; names = [name], groups = groups, meta = [meta],
                            allocs = [alloc], kwargs...)
end
"""
    assert_report_length(x::Nothing, n::Integer, name::Symbol)
    assert_report_length(x::AbstractVector, n::Integer, name::Symbol)

Check that a keyword of [`portfolio_report`](@ref) gives one value for each prediction.

# Arguments

  - `x`: The values of the keyword, or `nothing`.
  - `n`: The number of predictions.
  - `name`: The name of the keyword.

# Validation

  - `length(x) == n`. Otherwise the function raises a `DimensionMismatch` that names the keyword.

# Returns

  - `nothing`.

# Related

  - [`portfolio_report`](@ref)
  - [`report_item`](@ref)
"""
function assert_report_length(::Nothing, ::Integer, ::Symbol)
    return nothing
end
function assert_report_length(x::AbstractVector, n::Integer, name::Symbol)
    @argcheck(length(x) == n,
              DimensionMismatch("`$(name)` carries $(length(x)) entries and the report carries $(n) prediction(s)"))
    return nothing
end
"""
    report_item(x::Nothing, k::Integer)
    report_item(x::AbstractVector, k::Integer)

The value of a keyword of [`portfolio_report`](@ref) for prediction `k`: entry `k` of the keyword, or `nothing` when the keyword gives none.

# Related

  - [`assert_report_length`](@ref)
"""
function report_item(::Nothing, ::Integer)
    return nothing
end
function report_item(x::AbstractVector, k::Integer)
    return x[k]
end
"""
    report_figure_pairs(figs, name::Symbol) -> Vector

Check the figures of [`portfolio_report`](@ref), and collect them into a vector of pairs `caption => f`.

# Arguments

  - `figs`: A collection of pairs `caption => f`.
  - `name`: The name of the keyword.

# Validation

  - Each element is a pair whose first part is an `AbstractString`. Otherwise the function raises an `ArgumentError` that names the keyword.

# Returns

  - `figs::Vector`: The pairs, in order.

# Related

  - [`report_figures`](@ref)
"""
function report_figure_pairs(figs, name::Symbol)
    out = collect(figs)
    for p in out
        @argcheck(p isa Pair && first(p) isa AbstractString,
                  ArgumentError("each element of `$(name)` must be a pair `caption::AbstractString => f`, and one element is $(repr(p))"))
    end
    return out
end
"""
    report_figures(figs::AbstractVector, args...) -> Vector{AbstractReportBlock}

The figures of a report: one [`ReportFigure`](@ref) for each pair `caption => f` of `figs` whose `f(args...)` is not `nothing`.

A function that throws stops the build with its own error.

# Arguments

  - `figs`: The pairs, from [`report_figure_pairs`](@ref).
  - `args...`: The prediction of an entry, or the vector of predictions of the overview and their names.

# Returns

  - `blocks::Vector{AbstractReportBlock}`: The figures, in the order of `figs`.

# Related

  - [`portfolio_report`](@ref)
  - [`ReportFigure`](@ref)
"""
function report_figures(figs::AbstractVector, args...)
    blocks = AbstractReportBlock[]
    for p in figs
        fig = last(p)(args...)
        isnothing(fig) || push!(blocks, ReportFigure(fig; caption = first(p)))
    end
    return blocks
end
"""
    overview_table(df::DataFrames.AbstractDataFrame, nm::AbstractVector,
                   groups::Option{<:AbstractVector}) -> DataFrames.DataFrame

The table of the overview of a Portfolio Report: the summary table, with a `group` column first when the caller gives groups.

# Arguments

  - `df`: The summary table.
  - `nm`: The name of each prediction.
  - `groups`: The group of each prediction, or `nothing`.

# Returns

  - `tab::DataFrames.DataFrame`: The table.

# Related

  - [`portfolio_report`](@ref)
"""
function overview_table(df::DataFrames.AbstractDataFrame, ::AbstractVector, ::Nothing)
    return df
end
function overview_table(df::DataFrames.AbstractDataFrame, nm::AbstractVector,
                        groups::AbstractVector)
    pos = Dict(nm[k] => k for k in eachindex(nm))
    return DataFrames.insertcols(df, 1, :group => [groups[pos[x]] for x in df.name])
end
"""
    entry_sections(pred::AbstractPredictionResult, rows::DataFrames.AbstractDataFrame,
                   alloc, threshold::Real, figs::AbstractVector) -> Vector{ReportSection}

The sections of the entry of one prediction, in the order that [`portfolio_report`](@ref) states.

# Arguments

  - `pred`: The prediction.
  - `rows`: The rows of the summary table of the prediction.
  - `alloc`: The finite allocation result of the prediction, or `nothing`.
  - `threshold`: The least absolute weight that a weights table shows.
  - `figs`: The figures of an entry, from [`report_figure_pairs`](@ref).

# Returns

  - `sections::Vector{ReportSection}`: Statistics, Weights, then Allocation and Figures when the entry has them.

# Related

  - [`report_statistics`](@ref)
  - [`report_weights`](@ref)
  - [`report_allocation`](@ref)
  - [`report_figures`](@ref)
"""
function entry_sections(pred::AbstractPredictionResult, rows::DataFrames.AbstractDataFrame,
                        alloc, threshold::Real, figs::AbstractVector)
    sections = [report_statistics(pred, rows), report_weights(pred, threshold)]
    a = report_allocation(pred, alloc)
    if !(isnothing(a))
        push!(sections, a)
    end
    blocks = report_figures(figs, pred)
    if !(isempty(blocks))
        push!(sections, ReportSection("Figures", blocks))
    end
    return sections
end
"""
    statistics_table(rows::DataFrames.AbstractDataFrame) -> ReportTable

The table of the statistics of one prediction.

The `name` column is not shown, and the `path` and `member` columns are shown only when they hold a key. One row gives a table of two columns, the figure and its value. Many rows give a sortable table, one row for each path and member.

# Arguments

  - `rows`: The rows of the summary table of the prediction.

# Returns

  - `tab::ReportTable`: The table.

# Related

  - [`report_statistics`](@ref)
"""
function statistics_table(rows::DataFrames.AbstractDataFrame)
    cols = [c
            for c in names(rows)
            if c != "name" && !(c in ("path", "member") && all(isnothing, rows[!, c]))]
    tab = rows[:, cols]
    if DataFrames.nrow(tab) == 1
        return ReportTable(DataFrames.DataFrame(; figure = cols,
                                                value = [tab[1, c] for c in cols]))
    end
    return ReportTable(tab; caption = "One row for each path and member.", sortable = true)
end
"""
    distribution_table(rows::DataFrames.AbstractDataFrame) -> ReportTable

The distribution of each figure of a population over its paths.

A figure is a column of the summary table other than the keys and the dates. For each figure, the table gives the count of finite values, and the minimum, the 25 % quantile, the median, the 75 % quantile and the maximum of those values. A figure with no finite value shows empty cells.

# Arguments

  - `rows`: The rows of the summary table of the population.

# Returns

  - `tab::ReportTable`: The table, one row for each figure.

# Related

  - [`report_statistics`](@ref)
  - [`performance_summary`](@ref)
"""
function distribution_table(rows::DataFrames.AbstractDataFrame)
    figs = [c for c in names(rows) if !(c in ("name", "path", "member", "date0", "date1"))]
    q = [0, 0.25, 0.5, 0.75, 1]
    stats = map(figs) do c
        v = [x for x in rows[!, c] if x isa Real && isfinite(x)]
        return if isempty(v)
            (length(v), ntuple(_ -> missing, 5))
        else
            (length(v), Tuple(Statistics.quantile(v, q)))
        end
    end
    return ReportTable(DataFrames.DataFrame(; figure = figs, count = first.(stats),
                                            min = [s[2][1] for s in stats],
                                            q25 = [s[2][2] for s in stats],
                                            median = [s[2][3] for s in stats],
                                            q75 = [s[2][4] for s in stats],
                                            max = [s[2][5] for s in stats]);
                       caption = "The distribution of each figure over the paths of the population.")
end
"""
    report_statistics(pred::PredRes_MultiPredRes, rows::DataFrames.AbstractDataFrame)
    report_statistics(pred::PopulationPredictionResult, rows::DataFrames.AbstractDataFrame)

The Statistics section of the entry of one prediction: its table of statistics, and for a population the distribution of each figure over its paths.

# Arguments

  - `pred`: The prediction.
  - `rows`: The rows of the summary table of the prediction.

# Returns

  - `s::ReportSection`: The section.

# Related

  - [`statistics_table`](@ref)
  - [`distribution_table`](@ref)
"""
function report_statistics(::PredRes_MultiPredRes, rows::DataFrames.AbstractDataFrame)
    return ReportSection("Statistics", AbstractReportBlock[statistics_table(rows)])
end
function report_statistics(::PopulationPredictionResult, rows::DataFrames.AbstractDataFrame)
    return ReportSection("Statistics",
                         AbstractReportBlock[statistics_table(rows),
                                             distribution_table(rows)])
end
"""
    report_weights(pred::AbstractPredictionResult, threshold::Real) -> ReportSection

The Weights section of the entry of one prediction, one table for each member.

# Arguments

  - `pred`: The prediction.
  - `threshold`: The least absolute weight that a row shows.

# Returns

  - `s::ReportSection`: The section.

# Related

  - [`prediction_weights`](@ref)
  - [`prediction_members`](@ref)
  - [`weights_table`](@ref)
"""
function report_weights(pred::AbstractPredictionResult, threshold::Real)
    m = prediction_members(pred)
    tables = AbstractReportBlock[weights_table(prediction_weights(pred, j)..., threshold,
                                               m > 1 ? "Member $(j). " : "") for j in 1:m]
    return ReportSection("Weights", tables)
end
"""
    weights_table(names::AbstractVector, W::MatNum, labels::AbstractVector,
                  threshold::Real, prefix::AbstractString) -> ReportTable

The weights table of one member of a prediction, from the rows of [`prediction_weights`](@ref).

One row of `W` gives one column of weights. Many rows give a `mean` column, the mean over the rows, then one column for each row. The rows of the table are sorted by the absolute value of the first column of weights. A row whose largest absolute weight is less than `threshold` is not shown. A `NaN` weight is not less than `threshold`, so its row shows.

# Arguments

  - `names`: The asset names.
  - `W`: The weights, one row for each portfolio of the prediction.
  - `labels`: The label of each row of `W`.
  - `threshold`: The least absolute weight that a row shows.
  - `prefix`: The start of the caption, which names the member.

# Returns

  - `tab::ReportTable`: The table.

# Related

  - [`report_weights`](@ref)
  - [`prediction_weights`](@ref)
"""
function weights_table(names::AbstractVector, W::MatNum, labels::AbstractVector,
                       threshold::Real, prefix::AbstractString)
    cols = if size(W, 1) == 1
        [labels[1] => view(W, 1, :)]
    else
        ["mean" => vec(mean(W; dims = 1)); [labels[i] => view(W, i, :) for i in axes(W, 1)]]
    end
    keep = [!(maximum(abs, view(W, :, i)) < threshold) for i in axes(W, 2)]
    df = DataFrames.DataFrame("asset" => collect(names), cols...; makeunique = true)[keep,
                                                                                     :]
    df = df[sortperm(abs.(df[!, 2]); rev = true), :]
    dropped = count(!, keep)
    caption = string(prefix, "Sorted by the absolute value of the column `", first(cols[1]),
                     "`.",
                     if dropped > 0
                         " $(dropped) asset(s) with no absolute weight of at least $(threshold) are not shown."
                     else
                         ""
                     end)
    return ReportTable(df; caption = caption, sortable = true)
end
"""
    report_allocation(pred::AbstractPredictionResult, alloc::Nothing)
    report_allocation(pred::PredictionResult, alloc::FiniteAllocationOptimisationResult)
    report_allocation(pred::MultiPeriodPredictionResult, alloc::FiniteAllocationOptimisationResult)
    report_allocation(pred::PopulationPredictionResult, alloc::FiniteAllocationOptimisationResult)

The Allocation section of the entry of one prediction, or `nothing` when the entry has no allocation.

A finite allocation is the allocation of one portfolio. A single fold reads the names of its weights. A path reads the names of its last fold, the portfolio to hold now. A population holds no single portfolio, so it has no allocation.

# Arguments

  - `pred`: The prediction.
  - `alloc`: The finite allocation result, or `nothing`.

# Validation

  - `length(alloc.shares)` is the number of assets of the portfolio. Otherwise the function raises a `DimensionMismatch`.
  - A population has no allocation. Otherwise the function raises an `ArgumentError`.

# Returns

  - `s::Option{<:ReportSection}`: The section, or `nothing`.

# Related

  - [`portfolio_report`](@ref)
  - [`GreedyAllocationResult`](@ref)
  - [`DiscreteAllocationResult`](@ref)
"""
function report_allocation(::AbstractPredictionResult, ::Nothing)
    return nothing
end
function report_allocation(pred::PredictionResult,
                           alloc::FiniteAllocationOptimisationResult)
    names = first(prediction_weights(pred))
    @argcheck(length(alloc.shares) == length(names),
              DimensionMismatch("the allocation carries $(length(alloc.shares)) shares, and the portfolio that it allocates holds $(length(names)) assets"))
    df = DataFrames.DataFrame(; asset = collect(names), shares = alloc.shares,
                              cost = alloc.cost, weight = alloc.w)
    caption = string("Cash left: ", report_cell(alloc.cash, 6).text, ". Fees: ",
                     report_cell(alloc.fees, 6).text, ".")
    return ReportSection("Allocation",
                         AbstractReportBlock[ReportTable(df; caption = caption,
                                                         sortable = true)])
end
function report_allocation(pred::MultiPeriodPredictionResult,
                           alloc::FiniteAllocationOptimisationResult)
    return report_allocation(last(pred.pred), alloc)
end
function report_allocation(::PopulationPredictionResult,
                           ::FiniteAllocationOptimisationResult)
    return throw(ArgumentError("a population holds no single portfolio to hold, so it has no allocation: allocate one of its paths, and report that path"))
end

export portfolio_report
