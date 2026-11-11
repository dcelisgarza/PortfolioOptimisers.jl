"""
    portfolio_report(preds::AbstractVector{<:AbstractPredictionResult};
                     names::Option{<:AbstractVector{<:AbstractString}} = nothing,
                     title::AbstractString = "Portfolio report",
                     groups::Option{<:AbstractVector} = nothing,
                     meta::Option{<:AbstractVector} = nothing,
                     allocs::Option{<:AbstractVector} = nothing,
                     sets::Union{Nothing, UniverseSets, AbstractVector} = nothing,
                     taxonomy::AbstractVector{<:AbstractString} = String[],
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
 3. **Exposure**, when the entry has a [`UniverseSets`](@ref) and `taxonomy` names at least one taxonomy column: the sums of the weights of each group, as `# The Exposure section` states.
 4. **Allocation**, when the entry has a finite allocation result: the shares, the cost and the realised weight of each asset, with the cash that is left and the fees in the caption. A finite allocation is the allocation of one portfolio. For a path, it is the allocation of the target of the last fold, the portfolio to hold now.
 5. **Figures**, when `entry_figures` gives one: one figure for each of its pairs, from the prediction.

# The Exposure section

A taxonomy column is a key of `sets.dict`, such as `"nx_sector"`, that holds one group label for each asset of `sets.dict[sets.xkey]`, in that order. [`asset_sets_matrix`](@ref) reads the same column. Each name of `taxonomy` gives one subsection, and each member of the prediction gives one table in it. The section reads the rows of [`prediction_weights`](@ref), and it maps an asset to its label by name, so a fold that holds a part of the universe is correct. An asset that the universe does not name, or whose label is `missing`, counts in the group `(none)`, so the groups of a row sum to its weights. The section computes no statistic other than the sums.

  - **One row of weights**, a single fold: one row for each group, with the columns `long`, the sum of the positive weights, `short`, the sum of the negative weights, and `net`, their sum. The rows are sorted by the absolute value of `net`, and the caption gives the gross exposure. With a finite allocation, the columns `alloc_long`, `alloc_short` and `alloc_net` sum the realised weights of the allocation in the same way.
  - **Many rows of weights**, a path or a population: one row for each fold of the path or each path of the population, with the column `portfolio` that holds the label of the row, one column of net exposure for each group, and the column `gross`. A group whose label is `portfolio` or `gross` takes the column name of its label with the suffix `_1`. With a finite allocation of a path, the last row `allocation` sums the realised weights of the allocation.

The allocation is the allocation of the first member, as in the Allocation section. A `NaN` weight gives a `NaN` sum for its group, on both sides.

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
  - `sets`: The [`UniverseSets`](@ref) of every prediction, or the sets of each prediction, a `UniverseSets` or `nothing`, or `nothing` for no Exposure section.
  - `taxonomy`: The names of the taxonomy columns of `sets`, as stated in `# The Exposure section`. An empty vector gives no Exposure section.
  - `overview_figures`, `entry_figures`: The figures, as stated in `# Figures`.
  - `weight_threshold`: The least absolute weight that a row of a weights table shows.
  - `kwargs...`: The keywords of [`performance_summary`](@ref).

# Validation

  - `preds` is not empty. Otherwise the function raises an [`IsEmptyError`](@ref).
  - `names`, `groups`, `meta` and `allocs` have the length of `preds`, and so does `sets` when it is a vector. Otherwise the function raises a `DimensionMismatch` that names the keyword.
  - Each name of `taxonomy` is a key of the sets of an entry that has sets. Otherwise the function raises a `KeyError`, through [`taxonomy_column`](@ref).
  - A taxonomy column has the length of `sets.dict[sets.xkey]`. Otherwise the function raises a `DimensionMismatch`.
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
  - [`UniverseSets`](@ref)
"""
function portfolio_report(preds::AbstractVector{<:AbstractPredictionResult};
                          names::Option{<:AbstractVector{<:AbstractString}} = nothing,
                          title::AbstractString = "Portfolio report",
                          groups::Option{<:AbstractVector} = nothing,
                          meta::Option{<:AbstractVector} = nothing,
                          allocs::Option{<:AbstractVector} = nothing,
                          sets::Union{Nothing, UniverseSets, AbstractVector} = nothing,
                          taxonomy::AbstractVector{<:AbstractString} = String[],
                          overview_figures = (), entry_figures = (),
                          weight_threshold::Real = 1e-6, kwargs...)::Report
    @argcheck(!isempty(preds),
              IsEmptyError("a report needs at least one prediction, and the vector is empty"))
    ps = collect(preds)
    n = length(ps)
    nm = forecast_summary_names(names, n, "Portfolio")
    assert_report_length(groups, n, :groups)
    assert_report_length(meta, n, :meta)
    assert_report_length(allocs, n, :allocs)
    assert_report_length(sets, n, :sets)
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
                                          report_item(allocs, k), weight_threshold, efigs,
                                          report_item(sets, k), taxonomy);
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
    assert_report_length(x::Option{<:UniverseSets}, n::Integer, name::Symbol)
    assert_report_length(x::AbstractVector, n::Integer, name::Symbol)

Check that a keyword of [`portfolio_report`](@ref) gives one value for each prediction.

One [`UniverseSets`](@ref) is the sets of every prediction, so it has no length to check.

# Arguments

  - `x`: The values of the keyword, one `UniverseSets`, or `nothing`.
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
function assert_report_length(::Option{<:UniverseSets}, ::Integer, ::Symbol)
    return nothing
end
function assert_report_length(x::AbstractVector, n::Integer, name::Symbol)
    @argcheck(length(x) == n,
              DimensionMismatch("`$(name)` carries $(length(x)) entries and the report carries $(n) prediction(s)"))
    return nothing
end
"""
    report_item(x::Option{<:UniverseSets}, k::Integer)
    report_item(x::AbstractVector, k::Integer)

The value of a keyword of [`portfolio_report`](@ref) for prediction `k`: entry `k` of the keyword, the one [`UniverseSets`](@ref) of every prediction, or `nothing` when the keyword gives none.

# Related

  - [`assert_report_length`](@ref)
"""
function report_item(x::Option{<:UniverseSets}, ::Integer)
    return x
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
                   alloc, threshold::Real, figs::AbstractVector,
                   sets::Option{<:UniverseSets},
                   taxonomy::AbstractVector{<:AbstractString}) -> Vector{ReportSection}

The sections of the entry of one prediction, in the order that [`portfolio_report`](@ref) states.

The Allocation section is built before the Exposure section, so the checks of the allocation run before the Exposure section reads it.

# Arguments

  - `pred`: The prediction.
  - `rows`: The rows of the summary table of the prediction.
  - `alloc`: The finite allocation result of the prediction, or `nothing`.
  - `threshold`: The least absolute weight that a weights table shows.
  - `figs`: The figures of an entry, from [`report_figure_pairs`](@ref).
  - `sets`: The [`UniverseSets`](@ref) of the prediction, or `nothing`.
  - `taxonomy`: The names of the taxonomy columns of `sets`.

# Returns

  - `sections::Vector{ReportSection}`: Statistics, Weights, then Exposure, Allocation and Figures when the entry has them.

# Related

  - [`report_statistics`](@ref)
  - [`report_weights`](@ref)
  - [`report_exposure`](@ref)
  - [`report_allocation`](@ref)
  - [`report_figures`](@ref)
"""
function entry_sections(pred::AbstractPredictionResult, rows::DataFrames.AbstractDataFrame,
                        alloc, threshold::Real, figs::AbstractVector,
                        sets::Option{<:UniverseSets},
                        taxonomy::AbstractVector{<:AbstractString})
    sections = [report_statistics(pred, rows), report_weights(pred, threshold)]
    a = report_allocation(pred, alloc)
    x = report_exposure(pred, sets, taxonomy, alloc)
    if !(isnothing(x))
        push!(sections, x)
    end
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
    report_exposure(pred::AbstractPredictionResult, sets::Nothing,
                    taxonomy::AbstractVector, alloc)
    report_exposure(pred::AbstractPredictionResult, sets::UniverseSets,
                    taxonomy::AbstractVector{<:AbstractString}, alloc)

The Exposure section of the entry of one prediction, or `nothing` when the entry has no sets or `taxonomy` is empty.

The section holds one subsection for each name of `taxonomy`, from [`exposure_section`](@ref), and each subsection holds one table for each member. [`portfolio_report`](@ref) states the content of the tables.

# Arguments

  - `pred`: The prediction.
  - `sets`: The [`UniverseSets`](@ref) of the prediction, or `nothing`.
  - `taxonomy`: The names of the taxonomy columns of `sets`.
  - `alloc`: The finite allocation result of the prediction, or `nothing`. Its checks ran in [`report_allocation`](@ref).

# Returns

  - `s::Option{<:ReportSection}`: The section, or `nothing`.

# Related

  - [`exposure_section`](@ref)
  - [`allocation_weights`](@ref)
  - [`prediction_weights`](@ref)
"""
function report_exposure(::AbstractPredictionResult, ::Nothing, ::AbstractVector, ::Any)
    return nothing
end
function report_exposure(pred::AbstractPredictionResult, sets::UniverseSets,
                         taxonomy::AbstractVector{<:AbstractString}, alloc)
    if isempty(taxonomy)
        return nothing
    end
    ws = [prediction_weights(pred, j) for j in 1:prediction_members(pred)]
    aw = allocation_weights(pred, alloc)
    return ReportSection("Exposure",
                         AbstractReportBlock[exposure_section(sets, key, ws, aw)
                                             for key in taxonomy])
end
"""
    exposure_section(sets::UniverseSets, key::AbstractString, ws::AbstractVector,
                     aw::Option{<:Tuple}) -> ReportSection

The subsection of the Exposure section for one taxonomy column: one table for each member, from [`exposure_table`](@ref). The allocation is the allocation of the first member, so only its table reads `aw`.

# Arguments

  - `sets`: The [`UniverseSets`](@ref).
  - `key`: The name of the taxonomy column, and the title of the subsection.
  - `ws`: The rows of [`prediction_weights`](@ref) of each member.
  - `aw`: The realised weights of the allocation, from [`allocation_weights`](@ref), or `nothing`.

# Returns

  - `s::ReportSection`: The subsection.

# Related

  - [`report_exposure`](@ref)
  - [`exposure_table`](@ref)
"""
function exposure_section(sets::UniverseSets, key::AbstractString, ws::AbstractVector,
                          aw::Option{<:Tuple})
    m = length(ws)
    tabs = AbstractReportBlock[exposure_table(sets, key, ws[j]..., j == 1 ? aw : nothing,
                                              m > 1 ? "Member $(j). " : "") for j in 1:m]
    return ReportSection(key, tabs)
end
"""
    allocation_weights(pred::AbstractPredictionResult, alloc::Nothing)
    allocation_weights(pred::PredictionResult, alloc::FiniteAllocationOptimisationResult)
    allocation_weights(pred::MultiPeriodPredictionResult, alloc::FiniteAllocationOptimisationResult)

The realised weights of a finite allocation on the names of the portfolio that it allocates, or `nothing` when the entry has no allocation.

A single fold reads the names of its weights, and a path reads the names of its last fold, as [`report_allocation`](@ref) does.

# Arguments

  - `pred`: The prediction.
  - `alloc`: The finite allocation result, or `nothing`.

# Returns

  - `aw::Option{<:Tuple}`: The pair `(names, w)`, or `nothing`.

# Related

  - [`report_exposure`](@ref)
  - [`report_allocation`](@ref)
"""
function allocation_weights(::AbstractPredictionResult, ::Nothing)
    return nothing
end
function allocation_weights(pred::PredictionResult,
                            alloc::FiniteAllocationOptimisationResult)
    return first(prediction_weights(pred)), alloc.w
end
function allocation_weights(pred::MultiPeriodPredictionResult,
                            alloc::FiniteAllocationOptimisationResult)
    return allocation_weights(last(pred.pred), alloc)
end
"""
    exposure_labels(sets::UniverseSets, key::AbstractString,
                    names::AbstractVector) -> Vector{String}

The group label of each asset of `names`, from the taxonomy column `sets.dict[key]`.

An asset maps to its label by name, through `sets.dict[sets.xkey]`. An asset that the universe does not name, or whose label is `missing`, takes the label `(none)`.

# Arguments

  - `sets`: The [`UniverseSets`](@ref).
  - `key`: The name of the taxonomy column.
  - `names`: The asset names.

# Validation

  - `haskey(sets.dict, key)`, through [`taxonomy_column`](@ref).
  - `length(sets.dict[key]) == length(sets.dict[sets.xkey])`. Otherwise the function raises a `DimensionMismatch`.

# Returns

  - `labels::Vector{String}`: One label for each name.

# Related

  - [`exposure_table`](@ref)
  - [`taxonomy_column`](@ref)
"""
function exposure_labels(sets::UniverseSets, key::AbstractString, names::AbstractVector)
    col = taxonomy_column(sets, key, "the Exposure section of `portfolio_report`")
    nx = sets.dict[sets.xkey]
    @argcheck(length(col) == length(nx),
              DimensionMismatch("the taxonomy column `$(key)` holds $(length(col)) labels, and the asset universe `$(sets.xkey)` holds $(length(nx)) assets"))
    pos = Dict(nx[i] => col[i] for i in eachindex(nx, col))
    return [let v = get(pos, a, missing)
                ismissing(v) ? "(none)" : string(v)
            end
            for a in names]
end
"""
    exposure_sums(labels::AbstractVector{<:AbstractString}, w::AbstractVector,
                  groups::AbstractVector{<:AbstractString}) -> Tuple

The long and the short sums of the weights `w` in each group of `groups`.

The long sum adds `max(x, 0)` and the short sum adds `min(x, 0)` for each weight `x` of the group, so a short shows on the short side, and a `NaN` weight gives a `NaN` on both sides. The sums take the element type of `w`.

# Arguments

  - `labels`: The group label of each weight.
  - `w`: The weights.
  - `groups`: The groups, which hold every label of `labels`.

# Returns

  - `L::Vector`: The long sum of each group.
  - `S::Vector`: The short sum of each group, which is not positive.

# Related

  - [`exposure_table`](@ref)
"""
function exposure_sums(labels::AbstractVector{<:AbstractString}, w::AbstractVector,
                       groups::AbstractVector{<:AbstractString})
    gi = Dict(groups[k] => k for k in eachindex(groups))
    L = zeros(eltype(w), length(groups))
    S = zeros(eltype(w), length(groups))
    for i in eachindex(labels, w)
        k = gi[labels[i]]
        x = w[i]
        L[k] += max(x, zero(x))
        S[k] += min(x, zero(x))
    end
    return L, S
end
"""
    exposure_table(sets::UniverseSets, key::AbstractString, names::AbstractVector,
                   W::MatNum, labels::AbstractVector, aw::Option{<:Tuple},
                   prefix::AbstractString) -> ReportTable

The Exposure table of one member of a prediction for one taxonomy column, from the rows of [`prediction_weights`](@ref).

The groups are the labels of the assets of `names` and of the allocation, in order of first appearance, with `(none)` last. One row of `W` gives the table of [`exposure_group_table`](@ref), and many rows give the table of [`exposure_row_table`](@ref).

# Arguments

  - `sets`: The [`UniverseSets`](@ref).
  - `key`: The name of the taxonomy column.
  - `names`: The asset names of the columns of `W`.
  - `W`: The weights, one row for each portfolio of the prediction.
  - `labels`: The label of each row of `W`.
  - `aw`: The realised weights of the allocation, from [`allocation_weights`](@ref), or `nothing`.
  - `prefix`: The start of the caption, which names the member.

# Returns

  - `tab::ReportTable`: The table.

# Related

  - [`report_exposure`](@ref)
  - [`exposure_labels`](@ref)
  - [`exposure_sums`](@ref)
"""
function exposure_table(sets::UniverseSets, key::AbstractString, names::AbstractVector,
                        W::MatNum, labels::AbstractVector, aw::Option{<:Tuple},
                        prefix::AbstractString)
    gl = exposure_labels(sets, key, names)
    al = aw === nothing ? String[] : exposure_labels(sets, key, first(aw))
    # A stable sort on a `Bool` moves `(none)` last and keeps the order of the others.
    groups = sort!(unique([gl; al]); by = ==("(none)"))
    sums = [exposure_sums(gl, view(W, i, :), groups) for i in axes(W, 1)]
    asum = aw === nothing ? nothing : exposure_sums(al, last(aw), groups)
    lead = string(prefix, "The sums of the weights of each group of `", key, "`")
    if length(sums) == 1
        return exposure_group_table(groups, only(sums), asum, lead)
    end
    return exposure_row_table(groups, sums, labels, asum, lead)
end
"""
    exposure_group_table(groups::AbstractVector, s::Tuple, asum::Option{<:Tuple},
                         lead::AbstractString) -> ReportTable

The Exposure table of one row of weights: one row for each group, with the columns `long`, `short` and `net`, and the same three columns of the allocation after the prefix `alloc_`.

The rows are sorted by the absolute value of `net`, and the caption gives the gross exposure.

# Arguments

  - `groups`: The groups.
  - `s`: The long and the short sums of the weights, from [`exposure_sums`](@ref).
  - `asum`: The long and the short sums of the realised weights of the allocation, or `nothing`.
  - `lead`: The start of the caption.

# Returns

  - `tab::ReportTable`: The table.

# Related

  - [`exposure_table`](@ref)
  - [`exposure_row_table`](@ref)
"""
function exposure_group_table(groups::AbstractVector, s::Tuple, asum::Option{<:Tuple},
                              lead::AbstractString)
    L, S = s
    df = DataFrames.DataFrame(; group = groups, long = L, short = S, net = L .+ S)
    caption = string(lead,
                     ", long and short apart. Sorted by the absolute value of `net`. Gross exposure: ",
                     report_cell(sum(L) - sum(S), 6).text, ".")
    if asum !== nothing
        AL, AS = asum
        df.alloc_long = AL
        df.alloc_short = AS
        df.alloc_net = AL .+ AS
        caption = string(caption,
                         " The columns `alloc_long`, `alloc_short` and `alloc_net` sum the realised weights of the allocation, with a gross exposure of ",
                         report_cell(sum(AL) - sum(AS), 6).text, ".")
    end
    return ReportTable(df[sortperm(abs.(df.net); rev = true), :]; caption = caption,
                       sortable = true)
end
"""
    exposure_row_table(groups::AbstractVector, sums::AbstractVector,
                       labels::AbstractVector, asum::Option{<:Tuple},
                       lead::AbstractString) -> ReportTable

The Exposure table of many rows of weights: one row for each portfolio of the prediction, with the column `portfolio`, one column of net exposure for each group, and the column `gross`.

With an allocation, the last row `allocation` sums its realised weights. A group whose label is `portfolio` or `gross` takes the column name of its label with the suffix `_1`, so the two fixed columns keep their names.

# Arguments

  - `groups`: The groups.
  - `sums`: The long and the short sums of each row of weights, from [`exposure_sums`](@ref).
  - `labels`: The label of each row of weights.
  - `asum`: The long and the short sums of the realised weights of the allocation, or `nothing`.
  - `lead`: The start of the caption.

# Returns

  - `tab::ReportTable`: The table.

# Related

  - [`exposure_table`](@ref)
  - [`exposure_group_table`](@ref)
"""
function exposure_row_table(groups::AbstractVector, sums::AbstractVector,
                            labels::AbstractVector, asum::Option{<:Tuple},
                            lead::AbstractString)
    rows = collect(sums)
    rlabels = collect(String, labels)
    caption = string(lead,
                     ": the net exposure of each group and the gross exposure, one row for each portfolio of the prediction.")
    if asum !== nothing
        push!(rows, asum)
        push!(rlabels, "allocation")
        caption = string(caption,
                         " The row `allocation` sums the realised weights of the allocation.")
    end
    cols = [groups[k] => [first(r)[k] + last(r)[k] for r in rows]
            for k in eachindex(groups)]
    # The two fixed columns come first, so `makeunique` renames a group whose label is
    # `portfolio` or `gross`, and not the fixed column. Then the groups move between them.
    df = DataFrames.DataFrame("portfolio" => rlabels,
                              "gross" => [sum(first(r)) - sum(last(r)) for r in rows],
                              cols...; makeunique = true)
    return ReportTable(df[:, [1; 3:DataFrames.ncol(df); 2]]; caption = caption,
                       sortable = true)
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
