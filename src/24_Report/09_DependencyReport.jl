"""
    report_dependency_pairs(dep) -> Vector

Check the keyword `dependency` of [`portfolio_report`](@ref), and collect it into a vector of pairs.

# Arguments

  - `dep`: A collection of pairs `title => de` or `title => (G, de)`, with `de` a [`DependencyExposure`](@ref) and `G` an [`ExogenousGraph`](@ref) or an [`ExogenousGraphSeries`](@ref).

# Validation

  - Each element is such a pair, with an `AbstractString` title. Otherwise the function raises an `ArgumentError` that names the element.

# Returns

  - `pairs::Vector`: The pairs, in order.

# Related

  - [`report_dependency`](@ref)
"""
function report_dependency_pairs(dep)
    out = collect(dep)
    for p in out
        @argcheck(p isa Pair &&
                  first(p) isa AbstractString &&
                  (last(p) isa DependencyExposure ||
                   last(p) isa Tuple{AbstractExogenousGraph, DependencyExposure}),
                  ArgumentError("each element of `dependency` must be a pair `title::AbstractString => de::DependencyExposure` or `title => (G, de)` with G an ExogenousGraph or an ExogenousGraphSeries, and one element is $(repr(p))"))
    end
    return out
end
"""
    report_exposure_of(pred::AbstractPredictionResult, de::DependencyExposure, j::Integer) -> DependencyExposureResult
    report_exposure_of(pred::AbstractPredictionResult, spec::Tuple, j::Integer) -> DependencyExposureResult

Compute the Dependency Exposure of member `j` of a prediction for one pair of the keyword `dependency`: from the graphs of the folds for a configuration, and from the explicit graph for a tuple `(G, de)`.

# Algorithm

The method that Julia selects is the algorithm.

# Related

  - [`dependency_section`](@ref)
  - [`dependency_exposure`](@ref)
"""
function report_exposure_of(pred::AbstractPredictionResult, de::DependencyExposure,
                            j::Integer)
    return dependency_exposure(pred, de, j)
end
function report_exposure_of(pred::AbstractPredictionResult, spec::Tuple, j::Integer)
    return dependency_exposure(pred, spec[1], spec[2], j)
end
"""
    report_dependency(pred::AbstractPredictionResult, pairs::AbstractVector, threshold::Real) -> Option{<:ReportSection}

The section "Dependency exposure" of the entry of one prediction, or `nothing` when `pairs` is empty.

The section holds one subsection for each pair, from [`dependency_section`](@ref), and each subsection holds one table for each member. [`portfolio_report`](@ref) states the content of the tables.

# Arguments

  - `pred`: The prediction.
  - `pairs`: The pairs of the keyword `dependency`, from [`report_dependency_pairs`](@ref).
  - `threshold`: The least exposure that a row of a table shows.

# Returns

  - `s::Option{<:ReportSection}`: The section, or `nothing`.

# Related

  - [`dependency_section`](@ref)
  - [`dependency_exposure`](@ref)
"""
function report_dependency(pred::AbstractPredictionResult, pairs::AbstractVector,
                           threshold::Real)
    if isempty(pairs)
        return nothing
    end
    return ReportSection("Dependency exposure",
                         AbstractReportBlock[dependency_section(pred, first(p), last(p),
                                                                threshold) for p in pairs])
end
"""
    dependency_section(pred::AbstractPredictionResult, title::AbstractString, spec, threshold::Real) -> ReportSection

The subsection of the section "Dependency exposure" for one pair: one table for each member, from [`dependency_table`](@ref).

# Related

  - [`report_dependency`](@ref)
"""
function dependency_section(pred::AbstractPredictionResult, title::AbstractString, spec,
                            threshold::Real)
    m = prediction_members(pred)
    tabs = AbstractReportBlock[dependency_table(pred, report_exposure_of(pred, spec, j),
                                                threshold, m > 1 ? "Member $(j). " : "")
                               for j in 1:m]
    return ReportSection(title, tabs)
end
"""
    dependency_shown(res::DependencyExposureResult, threshold::Real) -> BitVector

Mark the nodes that a table of the section "Dependency exposure" shows: a node whose largest absolute `total` or largest `gross` over the rows is not less than `threshold`. A `NaN` is not less than `threshold`, so a node of a row with no graph shows.

# Related

  - [`dependency_table`](@ref)
"""
function dependency_shown(res::DependencyExposureResult, threshold::Real)
    net = dropdims(sum(res.L .+ res.S; dims = 3); dims = 3)
    gross = dropdims(sum(res.L .- res.S; dims = 3); dims = 3)
    return [!(maximum(abs, view(net, :, k)) < threshold &&
              maximum(view(gross, :, k)) < threshold) for k in axes(net, 2)]
end
"""
    dependency_caption(prefix::AbstractString, lead::AbstractString, hidden::Integer, threshold::Real) -> String

The caption of a table of the section "Dependency exposure", with the count of the nodes that the table does not show.

# Related

  - [`dependency_table`](@ref)
"""
function dependency_caption(prefix::AbstractString, lead::AbstractString, hidden::Integer,
                            threshold::Real)
    return string(prefix, lead,
                  if hidden > 0
                      " $(hidden) node(s) with no absolute total or gross exposure of at least $(threshold) are not shown."
                  else
                      ""
                  end)
end
"""
    dependency_table(pred::AbstractPredictionResult, res::DependencyExposureResult, threshold::Real, prefix::AbstractString) -> ReportTable

The table of one member of a prediction for one pair of the keyword `dependency`.

  - **One row of weights**: one row for each node, with the columns of [`DataFrames.DataFrame(::DependencyExposureResult)`](@ref), sorted by the absolute value of `total`.
  - **Many rows of weights**: one row for each node with `last`, the net `total` of the last row, which is the exposure of the portfolio to hold now, then the count of the finite values, the minimum, the 25 % quantile, the median, the 75 % quantile and the maximum of the net `total` over the rows, sorted by the absolute value of `last`. A population holds no single portfolio, so it has no `last`, and its rows are sorted by the absolute value of the median.

A node whose largest absolute `total` and largest `gross` are less than `threshold` is not shown, and the caption gives the count.

# Related

  - [`dependency_section`](@ref)
  - [`distribution_table`](@ref)
"""
function dependency_table(pred::AbstractPredictionResult, res::DependencyExposureResult,
                          threshold::Real, prefix::AbstractString)
    keep = dependency_shown(res, threshold)
    hidden = count(!, keep)
    if isone(length(res.rows))
        df = DataFrames.DataFrame(res)[keep, :]
        return ReportTable(df[sortperm(abs.(df.total); rev = true), :];
                           caption = dependency_caption(prefix,
                                                        "One row for each node, sorted by the absolute value of `total`.",
                                                        hidden, threshold), sortable = true)
    end
    net = dropdims(sum(res.L .+ res.S; dims = 3); dims = 3)
    df = dependency_distribution(res.nodes, net)
    lead = "The distribution of the net `total` of each node over the $(length(res.rows)) rows of weights."
    df, lead = dependency_last(pred, df, net, lead)
    df = df[keep, :]
    by = "last" in names(df) ? df.last : df.median
    return ReportTable(df[sortperm(abs.(coalesce.(by, 0.0)); rev = true), :];
                       caption = dependency_caption(prefix, lead, hidden, threshold),
                       sortable = true)
end
"""
    dependency_distribution(nodes::AbstractVector, net::MatNum) -> DataFrames.DataFrame

The distribution of the net `total` of each node over the rows of weights: the count of the finite values, the minimum, the 25 % quantile, the median, the 75 % quantile and the maximum. A node with no finite value shows empty cells, as in [`distribution_table`](@ref).

# Related

  - [`dependency_table`](@ref)
"""
function dependency_distribution(nodes::AbstractVector, net::MatNum)
    q = [0, 0.25, 0.5, 0.75, 1]
    stats = map(axes(net, 2)) do k
        v = filter(isfinite, view(net, :, k))
        return isempty(v) ? ntuple(_ -> missing, 5) : Tuple(Statistics.quantile(v, q))
    end
    return DataFrames.DataFrame(; node = collect(nodes),
                                count = [count(isfinite, view(net, :, k))
                                         for k in axes(net, 2)],
                                min = [s[1] for s in stats], q25 = [s[2] for s in stats],
                                median = [s[3] for s in stats], q75 = [s[4] for s in stats],
                                max = [s[5] for s in stats])
end
"""
    dependency_last(pred::PredRes_MultiPredRes, df::DataFrames.DataFrame, net::MatNum, lead::AbstractString) -> Tuple
    dependency_last(pred::PopulationPredictionResult, df::DataFrames.DataFrame, net::MatNum, lead::AbstractString) -> Tuple

Add the column `last`, the net `total` of the last row, to the distribution table of a fold or a path. A population holds no single portfolio to hold now, so its table has no `last`.

# Algorithm

The method that Julia selects is the algorithm.

# Related

  - [`dependency_table`](@ref)
"""
function dependency_last(::PredRes_MultiPredRes, df::DataFrames.DataFrame, net::MatNum,
                         lead::AbstractString)
    # `insertcols!` copies the column, so it reads a view of the last row.
    DataFrames.insertcols!(df, 2, :last => view(net, size(net, 1), :))
    return df,
           string(lead,
                  " The column `last` is the last row, the exposure of the portfolio to hold now, and the rows are sorted by its absolute value.")
end
function dependency_last(::PopulationPredictionResult, df::DataFrames.DataFrame, ::MatNum,
                         lead::AbstractString)
    return df,
           string(lead,
                  " A population holds no single portfolio to hold now, so the rows are sorted by the absolute value of the median.")
end
