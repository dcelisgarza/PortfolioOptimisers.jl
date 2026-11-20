"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of the blocks of a Portfolio Report: a text, a table, a figure or a section.

A block holds its content and knows no format. A writer asks each block to show itself as the format of the writer, so a format is a set of `show` methods, one for each block. [`Report`](@ref) holds the blocks of a report, and [`write_report`](@ref) writes it. The two formats of the library are Markdown and one self-contained HTML file. A new format adds a method of `show` for its MIME to every block, and a new block adds a method for every format.

# Interfaces

A concrete block subtypes `AbstractReportBlock` and implements one method of `show` for each format.

## `Base.show`

  - `Base.show(io::IO, mime::MIME"text/markdown", b::MyBlock) -> Nothing`: Write the block as Markdown.
  - `Base.show(io::IO, mime::MIME"text/html", b::MyBlock) -> Nothing`: Write the block as an HTML fragment, with every string of the caller escaped.

### Arguments

  - `io`: The stream. The writer of a report sets two keys of its `IOContext`: `:report_level`, the heading level of the block, and `:report_figure_mime`, the image MIME of a figure.
  - `mime`: The format.
  - `b`: The block.

### Returns

  - `nothing`.

# Related

  - [`ReportText`](@ref)
  - [`ReportTable`](@ref)
  - [`ReportFigure`](@ref)
  - [`ReportSection`](@ref)
  - [`Report`](@ref)
  - [`write_report`](@ref)
"""
abstract type AbstractReportBlock <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

A block of plain text in a Portfolio Report.

The text is not Markdown and not HTML. Every writer escapes it, so a string of the caller cannot add markup to the report. A blank line splits the text into paragraphs.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ReportText(text::AbstractString) -> ReportText

# Related

  - [`AbstractReportBlock`](@ref)
  - [`ReportSection`](@ref)
"""
@concrete struct ReportText <: AbstractReportBlock
    """
    The text of the block.
    """
    text
    function ReportText(text::AbstractString)
        return new{String}(String(text))
    end
end
"""
$(DocStringExtensions.TYPEDEF)

A table of a Portfolio Report, from a `DataFrame`.

A number shows with `digits` significant digits. An HTML cell keeps the full value, so a sort reads the value and not its rounded text. `missing`, `nothing` and `NaN` show as `—`, and a sort puts them last. A table with a `key` names a column whose value is the name of an entry of the report, and a click on a row of the HTML table selects that entry.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ReportTable(df::DataFrames.AbstractDataFrame; caption::AbstractString = "",
                sortable::Bool = false, digits::Integer = 4,
                key::Option{<:Symbol} = nothing) -> ReportTable

## Validation

  - `digits > 0`. Otherwise the constructor raises a `DomainError`.
  - `key` is `nothing` or a column of `df`. Otherwise the constructor raises an `ArgumentError`.

# Related

  - [`AbstractReportBlock`](@ref)
  - [`portfolio_report`](@ref)
"""
@concrete struct ReportTable <: AbstractReportBlock
    """
    The table.
    """
    df
    """
    The caption of the table, or an empty string for none.
    """
    caption
    """
    Whether the reader of the HTML file can sort the table by a click on a header.
    """
    sortable
    """
    The number of significant digits of a number.
    """
    digits
    """
    The column whose value names the entry of a row, or `nothing`.
    """
    key
    # One method for each type of `key`, so that no method takes a `Union` for JET to split
    # apart from the type parameter it sets.
    function ReportTable(df::DataFrames.AbstractDataFrame, caption::AbstractString,
                         sortable::Bool, digits::Integer, key::Nothing)
        @argcheck(digits > 0, DomainError(digits, "digits must be positive"))
        return new{typeof(df), String, Bool, typeof(digits), Nothing}(df, String(caption),
                                                                      sortable, digits, key)
    end
    function ReportTable(df::DataFrames.AbstractDataFrame, caption::AbstractString,
                         sortable::Bool, digits::Integer, key::Symbol)
        @argcheck(digits > 0, DomainError(digits, "digits must be positive"))
        @argcheck(String(key) in names(df),
                  ArgumentError("the key `$(key)` is not a column of the table"))
        return new{typeof(df), String, Bool, typeof(digits), Symbol}(df, String(caption),
                                                                     sortable, digits, key)
    end
end
function ReportTable(df::DataFrames.AbstractDataFrame; caption::AbstractString = "",
                     sortable::Bool = false, digits::Integer = 4,
                     key::Option{<:Symbol} = nothing)::ReportTable
    return ReportTable(df, caption, sortable, digits, key)
end
"""
$(DocStringExtensions.TYPEDEF)

A figure of a Portfolio Report.

The figure is any object that shows itself as an SVG or a PNG image, `showable(MIME"image/svg+xml"(), fig)` or `showable(MIME"image/png"(), fig)`. The block does not depend on a plotting package, so a figure of any package that implements `show` for an image MIME works. A writer embeds the image in the report as a data URI, so the report needs no file beside it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ReportFigure(fig; caption::AbstractString = "") -> ReportFigure

## Validation

  - `fig` shows itself as an SVG or a PNG image. Otherwise the constructor raises an `ArgumentError` that names the type of `fig`.

# Related

  - [`AbstractReportBlock`](@ref)
  - [`default_report_figures`](@ref)
"""
@concrete struct ReportFigure <: AbstractReportBlock
    """
    The figure.
    """
    fig
    """
    The caption of the figure, or an empty string for none.
    """
    caption
    function ReportFigure(fig::Any, caption::AbstractString)
        @argcheck(showable(MIME"image/svg+xml"(), fig) || showable(MIME"image/png"(), fig),
                  ArgumentError("a figure of a report must show itself as an SVG or a PNG image, and an object of type `$(typeof(fig))` shows itself as neither"))
        return new{typeof(fig), String}(fig, String(caption))
    end
end
function ReportFigure(fig; caption::AbstractString = "")::ReportFigure
    return ReportFigure(fig, caption)
end
"""
$(DocStringExtensions.TYPEDEF)

A titled list of blocks of a Portfolio Report.

A section can hold another section. Its heading is one level below the heading of the block that holds it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ReportSection(title::AbstractString,
                  blocks::AbstractVector{<:AbstractReportBlock}) -> ReportSection

# Related

  - [`AbstractReportBlock`](@ref)
  - [`ReportEntry`](@ref)
"""
@concrete struct ReportSection <: AbstractReportBlock
    """
    The title of the section.
    """
    title
    """
    The blocks of the section, in order.
    """
    blocks
    function ReportSection(title::AbstractString,
                           blocks::AbstractVector{<:AbstractReportBlock})
        return new{String, typeof(blocks)}(String(title), blocks)
    end
end
"""
    report_meta(meta::Nothing)
    report_meta(meta::AbstractVector{<:Pair})

The labels of a report or of an entry, as pairs of strings.

The library gives a label no meaning. A writer shows the labels as the caller gave them, in order.

# Arguments

  - `meta`: Pairs `label => value`, or `nothing` for none.

# Returns

  - `meta::Vector{Pair{String, String}}`: The labels, each side converted with `string`.

# Related

  - [`ReportEntry`](@ref)
  - [`Report`](@ref)
"""
function report_meta(::Nothing)
    return Pair{String, String}[]
end
function report_meta(meta::AbstractVector{<:Pair})
    return Pair{String, String}[string(first(p)) => string(last(p)) for p in meta]
end
"""
$(DocStringExtensions.TYPEDEF)

One portfolio of a Portfolio Report, a candidate of the comparison.

`group` and `meta` are labels of the caller. The library gives them no meaning, and a writer shows them as given. The HTML file lists the entries of a group under the name of the group in its selector, and can hide a group.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ReportEntry(name::AbstractString, sections::AbstractVector{<:ReportSection};
                group::Option{<:AbstractString} = nothing,
                meta::Option{<:AbstractVector{<:Pair}} = nothing) -> ReportEntry

# Related

  - [`Report`](@ref)
  - [`ReportSection`](@ref)
  - [`portfolio_report`](@ref)
"""
@concrete struct ReportEntry <: AbstractResult
    """
    The name of the entry, unique in its report.
    """
    name
    """
    The sections of the entry, in order.
    """
    sections
    """
    The group of the entry, or `nothing`.
    """
    group
    """
    The labels of the entry, as pairs of strings.
    """
    meta
    function ReportEntry(name::AbstractString, sections::AbstractVector{<:ReportSection},
                         group::Option{<:AbstractString},
                         meta::Option{<:AbstractVector{<:Pair}})
        return new{String, typeof(sections), typeof(group), Vector{Pair{String, String}}}(String(name),
                                                                                          sections,
                                                                                          group,
                                                                                          report_meta(meta))
    end
end
function ReportEntry(name::AbstractString, sections::AbstractVector{<:ReportSection};
                     group::Option{<:AbstractString} = nothing,
                     meta::Option{<:AbstractVector{<:Pair}} = nothing)::ReportEntry
    return ReportEntry(name, sections, group, meta)
end
"""
$(DocStringExtensions.TYPEDEF)

A Portfolio Report: a document that compares portfolios, in a form that no format decides.

The overview compares the entries, and each entry shows one portfolio. [`portfolio_report`](@ref) builds a report from prediction results, and a caller can build one by hand from blocks. [`write_report`](@ref) writes a report as Markdown or as one self-contained HTML file, and `show` with the MIME `text/markdown` or `text/html` writes it to a stream.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Report(title::AbstractString, entries::AbstractVector{<:ReportEntry};
           overview::AbstractVector{<:AbstractReportBlock} = AbstractReportBlock[],
           meta::Option{<:AbstractVector{<:Pair}} = nothing) -> Report

## Validation

  - The names of the entries are unique, because the selector and the links of the HTML file read them. Otherwise the constructor raises an `ArgumentError` that names each repeated name.

# Related

  - [`ReportEntry`](@ref)
  - [`AbstractReportBlock`](@ref)
  - [`portfolio_report`](@ref)
  - [`write_report`](@ref)
"""
@concrete struct Report <: AbstractResult
    """
    The title of the report.
    """
    title
    """
    The entries of the report, one for each portfolio.
    """
    entries
    """
    The blocks above the entries, which compare them.
    """
    overview
    """
    The labels of the report, as pairs of strings.
    """
    meta
    function Report(title::AbstractString, entries::AbstractVector{<:ReportEntry},
                    overview::AbstractVector{<:AbstractReportBlock},
                    meta::Option{<:AbstractVector{<:Pair}})
        nms = [e.name for e in entries]
        rep = unique(filter(n -> count(==(n), nms) > 1, nms))
        @argcheck(isempty(rep),
                  ArgumentError("the names of the entries of a report must be unique, and these names repeat: $(join(repr.(rep), ", "))"))
        return new{String, typeof(entries), typeof(overview), Vector{Pair{String, String}}}(String(title),
                                                                                            entries,
                                                                                            overview,
                                                                                            report_meta(meta))
    end
end
function Report(title::AbstractString, entries::AbstractVector{<:ReportEntry};
                overview::AbstractVector{<:AbstractReportBlock} = AbstractReportBlock[],
                meta::Option{<:AbstractVector{<:Pair}} = nothing)::Report
    return Report(title, entries, overview, meta)
end
"""
    Base.show(io::IO, b::Union{AbstractReportBlock, ReportEntry, Report})

Show a block, an entry or a report on one line, with its type, its name and the size of its content.

A report holds figures and tables, which can be large. The one line keeps the REPL readable. `show` with the MIME `text/markdown` or `text/html` writes the content.

# Arguments

  - `io`: The stream.
  - `b`: The block, the entry or the report.

# Returns

  - `nothing`.

# Related

  - [`Report`](@ref)
  - [`write_report`](@ref)
"""
function Base.show(io::IO, b::ReportText)
    return print(io, "ReportText(", repr(first(b.text, 40)), length(b.text) > 40 ? "…" : "",
                 ")")
end
function Base.show(io::IO, b::ReportTable)
    return print(io, "ReportTable(", DataFrames.nrow(b.df), "×", DataFrames.ncol(b.df),
                 " table)")
end
function Base.show(io::IO, b::ReportFigure)
    return print(io, "ReportFigure(", typeof(b.fig), ")")
end
function Base.show(io::IO, b::ReportSection)
    return print(io, "ReportSection(", repr(b.title), ", ", length(b.blocks), " blocks)")
end
function Base.show(io::IO, e::ReportEntry)
    return print(io, "ReportEntry(", repr(e.name), ", ", length(e.sections), " sections)")
end
function Base.show(io::IO, r::Report)
    return print(io, "Report(", repr(r.title), ", ", length(r.entries), " entries)")
end

export ReportText, ReportTable, ReportFigure, ReportSection, ReportEntry, Report
public AbstractReportBlock
