include_dependency(joinpath(@__DIR__, "report.css"))
include_dependency(joinpath(@__DIR__, "report.js"))
"""
    REPORT_CSS

The style sheet of the HTML file of a Portfolio Report, read at load time from `report.css` beside this file.

The HTML writer puts it in an inline `<style>` element, so the file loads no style sheet. The colours are variables, with a dark set for `prefers-color-scheme: dark` and for `data-theme="dark"` on the root element.

# Related

  - [`REPORT_JS`](@ref)
  - [`write_report`](@ref)
"""
const REPORT_CSS = read(joinpath(@__DIR__, "report.css"), String)
"""
    REPORT_JS

The script of the HTML file of a Portfolio Report, read at load time from `report.js` beside this file.

The HTML writer puts it in an inline `<script>` element, so the file loads no script. The script shows one entry at a time, keeps the selection in `location.hash`, moves between the entries with the arrow keys, sorts a table by a click on a header, hides a group, and toggles the theme. With no script, the file shows every entry, one after the other.

# Related

  - [`REPORT_CSS`](@ref)
  - [`write_report`](@ref)
"""
const REPORT_JS = read(joinpath(@__DIR__, "report.js"), String)
"""
    REPORT_CSP

The Content Security Policy of the HTML file of a Portfolio Report.

The policy refuses every load from the network. It admits only images from data URIs, and the inline style and script of the file. A string of the caller is escaped, and the policy also holds for it.

# Related

  - [`write_report`](@ref)
"""
const REPORT_CSP = "default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'"
"""
    html_escape(s::AbstractString) -> String

Escape a string for HTML text and for the value of a quoted HTML attribute.

# Arguments

  - `s`: The string.

# Returns

  - `t::String`: `s` with `&`, `<`, `>`, `"` and `'` written as character references.

# Related

  - [`md_escape`](@ref)
"""
function html_escape(s::AbstractString)
    return replace(s, '&' => "&amp;", '<' => "&lt;", '>' => "&gt;", '"' => "&quot;",
                   '\'' => "&#39;")
end
"""
    md_escape(s::AbstractString) -> String

Escape a string for Markdown, so that it shows as plain text.

A Markdown renderer reads raw HTML and the markup characters. The function writes `&`, `<`, `>` and `"` as character references, puts a backslash before each other markup character of the set ``\\`*_{}[]()#+-.!|~``, and writes a line break as a space, so the string stays in its paragraph or in its table cell.

# Arguments

  - `s`: The string.

# Returns

  - `t::String`: The escaped string.

# Related

  - [`html_escape`](@ref)
"""
function md_escape(s::AbstractString)
    io = IOBuffer()
    for c in s
        if c == '&'
            print(io, "&amp;")
        elseif c == '<'
            print(io, "&lt;")
        elseif c == '>'
            print(io, "&gt;")
        elseif c == '"'
            print(io, "&quot;")
        elseif c == '\n' || c == '\r'
            print(io, ' ')
        elseif c in "\\`*_{}[]()#+-.!|~"
            print(io, '\\', c)
        else
            print(io, c)
        end
    end
    return String(take!(io))
end
"""
    report_cell(x::Union{Nothing, Missing}, digits::Integer)
    report_cell(x::Integer, digits::Integer)
    report_cell(x::Bool, digits::Integer)
    report_cell(x::AbstractFloat, digits::Integer)
    report_cell(x::Real, digits::Integer)
    report_cell(x::Any, digits::Integer)

The text, the sort value and the kind of one cell of a [`ReportTable`](@ref).

A float shows with `digits` significant digits, and its sort value is the full number. An integer sorts as a number. `nothing`, `missing` and `NaN` are empty cells, which show as `—` and have no sort value. Any other value, a `Bool` included, shows as its `string`, and sorts as that text.

# Arguments

  - `x`: The value of the cell.
  - `digits`: The number of significant digits of a float.

# Returns

  - `cell::NamedTuple`: `text`, the text to show; `v`, the sort value as a string, or `nothing` for an empty cell; `num`, whether the sort value is a number.

# Related

  - [`ReportTable`](@ref)
"""
function report_cell(::Union{Nothing, Missing}, ::Integer)
    return (; text = "—", v = nothing, num = false)
end
function report_cell(x::Integer, ::Integer)
    return (; text = string(x), v = string(x), num = true)
end
function report_cell(x::Bool, ::Integer)
    return (; text = string(x), v = string(x), num = false)
end
function report_cell(x::AbstractFloat, digits::Integer)
    if isnan(x)
        return (; text = "—", v = nothing, num = false)
    end
    # JavaScript's `Number` reads `Infinity`, and not Julia's `Inf`.
    v = isinf(x) ? (x > zero(x) ? "Infinity" : "-Infinity") : string(x)
    return (; text = string(round(x; sigdigits = digits)), v = v, num = true)
end
function report_cell(x::Real, ::Integer)
    return (; text = string(x), v = string(x), num = false)
end
function report_cell(x::Any, ::Integer)
    s = string(x)
    return (; text = s, v = s, num = false)
end
"""
    report_cell_html(io::IO, c::NamedTuple)

Write one cell of a [`ReportTable`](@ref) as an HTML `<td>` element, from its [`report_cell`](@ref).

A number carries the class `po-num`, and a cell with a sort value carries it in `data-v`. The script of the file sorts by `data-v`, as a number when both cells are numbers, and puts a cell with no `data-v` last.

# Arguments

  - `io`: The stream.
  - `c`: The cell, from [`report_cell`](@ref).

# Returns

  - `nothing`.

# Related

  - [`report_cell`](@ref)
  - [`REPORT_JS`](@ref)
"""
function report_cell_html(io::IO, c::NamedTuple)
    print(io, "<td", c.num ? " class=\"po-num\"" : "")
    if !(isnothing(c.v))
        print(io, " data-v=\"", html_escape(c.v), "\"")
    end
    print(io, ">", html_escape(c.text), "</td>")
    return nothing
end
"""
    report_level(io::IO) -> Int

The heading level of the block that a writer writes to `io`, from the key `:report_level` of its `IOContext`, at most 6.

A section writes its blocks one level below its own heading. A block written with no context starts at level 2.

# Related

  - [`AbstractReportBlock`](@ref)
"""
function report_level(io::IO)
    return min(get(io, :report_level, 2)::Int, 6)
end
"""
    figure_mime(fig, preferred::MIME"image/svg+xml")
    figure_mime(fig, preferred::MIME"image/png")

The image MIME in which a writer embeds a figure: the preferred MIME when the figure shows itself in it, and the other image MIME otherwise.

# Arguments

  - `fig`: The figure.
  - `preferred`: The image MIME of the writer, from the key `:report_figure_mime` of its `IOContext`.

# Returns

  - `mime::MIME`: The image MIME.

# Related

  - [`ReportFigure`](@ref)
  - [`figure_data_uri`](@ref)
"""
function figure_mime(fig, preferred::MIME"image/svg+xml")
    return showable(preferred, fig) ? preferred : MIME"image/png"()
end
function figure_mime(fig, preferred::MIME"image/png")
    return showable(preferred, fig) ? preferred : MIME"image/svg+xml"()
end
"""
    figure_data_uri(io::IO, fig) -> String

The data URI of a figure, in the image MIME that [`figure_mime`](@ref) chooses for the writer of `io`.

An image tag that reads a data URI isolates its figure. An inline SVG element shares the identifiers and the styles of its page, so two figures of one plotting backend can clip each other. The data URI holds the base64 code of the image, so it adds a third to the size of the image.

# Arguments

  - `io`: The stream of the writer.
  - `fig`: The figure.

# Returns

  - `uri::String`: `data:<mime>;base64,<code>`.

# Related

  - [`ReportFigure`](@ref)
  - [`write_report`](@ref)
"""
function figure_data_uri(io::IO, fig)
    m = figure_mime(fig, get(io, :report_figure_mime, MIME"image/svg+xml"()))
    return string("data:", m, ";base64,", Base64.base64encode(show, m, fig))
end
"""
    report_paragraphs(text::AbstractString) -> Vector{String}

The paragraphs of a text: the parts between blank lines, with their edges stripped, and with no empty part.

# Related

  - [`ReportText`](@ref)
"""
function report_paragraphs(text::AbstractString)
    return filter(!isempty, [strip(p) for p in split(text, r"\n[ \t]*\n")])
end
"""
    Base.show(io::IO, mime::MIME"text/markdown", b::AbstractReportBlock)
    Base.show(io::IO, mime::MIME"text/markdown", e::ReportEntry)
    Base.show(io::IO, mime::MIME"text/markdown", r::Report)

Write a block, an entry or a Portfolio Report as Markdown.

A section, an entry and a report write a heading, and each writes its blocks one level below it. A table is a pipe table with its caption in italics above it, and a figure is an image with a data URI. A Markdown viewer that does not read a data URI shows the caption of the figure. Every string of the caller passes through [`md_escape`](@ref). The text of a number in a table does not, because the writer makes that text and it holds no markup.

# Arguments

  - `io`: The stream.
  - `mime`: The MIME `text/markdown`.
  - `b`, `e`, `r`: The block, the entry or the report.

# Returns

  - `nothing`.

# Related

  - [`write_report`](@ref)
  - [`AbstractReportBlock`](@ref)
"""
function Base.show(io::IO, ::MIME"text/markdown", b::ReportText)
    print(io, join(md_escape.(report_paragraphs(b.text)), "\n\n"), "\n\n")
    return nothing
end
function Base.show(io::IO, ::MIME"text/markdown", b::ReportTable)
    df = b.df
    if !(isempty(b.caption))
        print(io, "*", md_escape(b.caption), "*\n\n")
    end
    print(io, "| ", join(md_escape.(names(df)), " | "), " |\n")
    print(io, "|", repeat(" --- |", DataFrames.ncol(df)), "\n")
    for row in eachrow(df)
        print(io, "|")
        for x in row
            c = report_cell(x, b.digits)
            # The text of a number is the library's, and holds no markup to escape.
            print(io, " ", c.num ? c.text : md_escape(c.text), " |")
        end
        print(io, "\n")
    end
    print(io, "\n")
    return nothing
end
function Base.show(io::IO, ::MIME"text/markdown", b::ReportFigure)
    print(io, "![", md_escape(b.caption), "](", figure_data_uri(io, b.fig), ")\n\n")
    return nothing
end
function Base.show(io::IO, mime::MIME"text/markdown", b::ReportSection)
    level = report_level(io)
    print(io, repeat("#", level), " ", md_escape(b.title), "\n\n")
    inner = IOContext(io, :report_level => level + 1)
    for x in b.blocks
        show(inner, mime, x)
    end
    return nothing
end
function Base.show(io::IO, mime::MIME"text/markdown", e::ReportEntry)
    print(io, "## ", md_escape(e.name), "\n\n")
    if !(isnothing(e.group))
        print(io, "Group: ", md_escape(e.group), "\n\n")
    end
    report_meta_markdown(io, e.meta)
    inner = IOContext(io, :report_level => 3)
    for s in e.sections
        show(inner, mime, s)
    end
    return nothing
end
function Base.show(io::IO, mime::MIME"text/markdown", r::Report)
    print(io, "# ", md_escape(r.title), "\n\n")
    report_meta_markdown(io, r.meta)
    inner = IOContext(io, :report_level => 3)
    if !isempty(r.overview)
        print(io, "## Overview\n\n")
        for b in r.overview
            show(inner, mime, b)
        end
    end
    for e in r.entries
        show(io, mime, e)
    end
    return nothing
end
"""
    report_meta_markdown(io::IO, meta::AbstractVector{<:Pair})

Write the labels of a report or of an entry as a Markdown list. An empty list writes nothing.

# Arguments

  - `io`: The stream.
  - `meta`: The labels, from [`report_meta`](@ref).

# Returns

  - `nothing`.

# Related

  - [`report_meta`](@ref)
"""
function report_meta_markdown(io::IO, meta::AbstractVector{<:Pair})
    for p in meta
        print(io, "- **", md_escape(first(p)), "**: ", md_escape(last(p)), "\n")
    end
    if !(isempty(meta))
        print(io, "\n")
    end
    return nothing
end
"""
    report_meta_html(io::IO, meta::AbstractVector{<:Pair})

Write the labels of a report or of an entry as an HTML list. An empty list writes nothing.

# Arguments

  - `io`: The stream.
  - `meta`: The labels, from [`report_meta`](@ref).

# Returns

  - `nothing`.

# Related

  - [`report_meta`](@ref)
  - [`report_meta_markdown`](@ref)
"""
function report_meta_html(io::IO, meta::AbstractVector{<:Pair})
    if isempty(meta)
        return nothing
    end
    print(io, "<ul class=\"po-meta\">")
    for p in meta
        print(io, "<li><b>", html_escape(first(p)), "</b>: ", html_escape(last(p)), "</li>")
    end
    print(io, "</ul>\n")
    return nothing
end
"""
    Base.show(io::IO, mime::MIME"text/html", b::AbstractReportBlock)
    Base.show(io::IO, mime::MIME"text/html", e::ReportEntry)
    Base.show(io::IO, mime::MIME"text/html", r::Report)

Write a block or an entry as an HTML fragment, or a Portfolio Report as one self-contained HTML file.

The file of a report holds its style, its script and its figures inline, and loads nothing. A Content Security Policy, [`REPORT_CSP`](@ref), makes the browser refuse any load from the network. Every string of the caller passes through [`html_escape`](@ref). The file shows one entry at a time when its script runs, with a selector when the report holds more than one entry, a filter when an entry has a group, and a toggle of the theme. With no script, and in print, the file shows every entry.

# Arguments

  - `io`: The stream.
  - `mime`: The MIME `text/html`.
  - `b`, `e`, `r`: The block, the entry or the report.

# Returns

  - `nothing`.

# Related

  - [`write_report`](@ref)
  - [`REPORT_CSS`](@ref)
  - [`REPORT_JS`](@ref)
"""
function Base.show(io::IO, ::MIME"text/html", b::ReportText)
    for p in report_paragraphs(b.text)
        print(io, "<p>", html_escape(p), "</p>\n")
    end
    return nothing
end
function Base.show(io::IO, ::MIME"text/html", b::ReportTable)
    df = b.df
    ks = isnothing(b.key) ? nothing : df[!, b.key]
    print(io, "<figure class=\"po-table\">")
    if !(isempty(b.caption))
        print(io, "<figcaption>", html_escape(b.caption), "</figcaption>")
    end
    print(io, "<div class=\"po-scroll\"><table", b.sortable ? " class=\"po-sortable\"" : "",
          "><thead><tr>")
    for n in names(df)
        print(io, "<th scope=\"col\">", html_escape(n), "</th>")
    end
    print(io, "</tr></thead><tbody>\n")
    for (i, row) in enumerate(eachrow(df))
        print(io, "<tr")
        if !(isnothing(ks))
            print(io, " data-entry=\"", html_escape(string(ks[i])), "\"")
        end
        print(io, ">")
        for x in row
            report_cell_html(io, report_cell(x, b.digits))
        end
        print(io, "</tr>\n")
    end
    print(io, "</tbody></table></div></figure>\n")
    return nothing
end
function Base.show(io::IO, ::MIME"text/html", b::ReportFigure)
    print(io, "<figure class=\"po-figure\"><img alt=\"", html_escape(b.caption),
          "\" src=\"", figure_data_uri(io, b.fig), "\">")
    if !(isempty(b.caption))
        print(io, "<figcaption>", html_escape(b.caption), "</figcaption>")
    end
    print(io, "</figure>\n")
    return nothing
end
function Base.show(io::IO, mime::MIME"text/html", b::ReportSection)
    level = report_level(io)
    print(io, "<section class=\"po-section\"><h", level, ">", html_escape(b.title), "</h",
          level, ">\n")
    inner = IOContext(io, :report_level => level + 1)
    for x in b.blocks
        show(inner, mime, x)
    end
    print(io, "</section>\n")
    return nothing
end
function Base.show(io::IO, mime::MIME"text/html", e::ReportEntry)
    k = get(io, :report_entry, 1)::Int
    print(io, "<article class=\"po-entry\" id=\"entry-", k, "\" data-name=\"",
          html_escape(e.name), "\"")
    if !(isnothing(e.group))
        print(io, " data-group=\"", html_escape(e.group), "\"")
    end
    print(io, "><h2>", html_escape(e.name), "</h2>\n")
    if !(isnothing(e.group))
        print(io, "<p class=\"po-group\">", html_escape(e.group), "</p>\n")
    end
    report_meta_html(io, e.meta)
    inner = IOContext(io, :report_level => 3)
    for s in e.sections
        show(inner, mime, s)
    end
    print(io, "</article>\n")
    return nothing
end
function Base.show(io::IO, mime::MIME"text/html", r::Report)
    print(io, "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n",
          "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n",
          "<meta http-equiv=\"Content-Security-Policy\" content=\"", REPORT_CSP, "\">\n",
          "<title>", html_escape(r.title), "</title>\n<style>\n", REPORT_CSS,
          "</style>\n</head>\n<body>\n<header>\n<h1>", html_escape(r.title), "</h1>\n")
    report_meta_html(io, r.meta)
    report_controls_html(io, r.entries)
    print(io, "</header>\n")
    inner = IOContext(io, :report_level => 3)
    if !isempty(r.overview)
        print(io, "<section id=\"overview\"><h2>Overview</h2>\n")
        for b in r.overview
            show(inner, mime, b)
        end
        print(io, "</section>\n")
    end
    print(io, "<main>\n")
    for (k, e) in enumerate(r.entries)
        show(IOContext(io, :report_entry => k), mime, e)
    end
    print(io, "</main>\n<script>\n", REPORT_JS, "</script>\n</body>\n</html>\n")
    return nothing
end
"""
    report_controls_html(io::IO, entries::AbstractVector{<:ReportEntry})

Write the controls of the HTML file of a Portfolio Report: the entry selector, the group filter and the toggle of the theme.

The selector shows only when the report holds more than one entry. It lists the entries with no group first, then one `<optgroup>` for each group, in the order in which the groups first appear. The filter shows only when an entry has a group, with one checkbox for each group. The style sheet hides the controls when the script does not run.

# Arguments

  - `io`: The stream.
  - `entries`: The entries of the report.

# Returns

  - `nothing`.

# Related

  - [`REPORT_JS`](@ref)
"""
function report_controls_html(io::IO, entries::AbstractVector{<:ReportEntry})
    groups = unique(e.group for e in entries if !isnothing(e.group))
    print(io, "<nav class=\"po-controls\">")
    if length(entries) > 1
        report_select_html(io, entries, groups)
    end
    if !isempty(groups)
        print(io, "<fieldset class=\"po-groups\"><legend>Groups</legend>")
        for g in groups
            print(io, "<label><input type=\"checkbox\" checked data-group=\"",
                  html_escape(g), "\"> ", html_escape(g), "</label>")
        end
        print(io, "</fieldset>")
    end
    print(io, "<button type=\"button\" id=\"po-theme\">Theme</button></nav>\n")
    return nothing
end
"""
    report_select_html(io::IO, entries::AbstractVector{<:ReportEntry},
                       groups::AbstractVector)

Write the entry selector of the HTML file of a Portfolio Report: the entries with no group first, then one `<optgroup>` for each group, in the order of `groups`.

# Arguments

  - `io`: The stream.
  - `entries`: The entries of the report.
  - `groups`: The groups of the entries, in the order in which they first appear.

# Returns

  - `nothing`.

# Related

  - [`report_controls_html`](@ref)
"""
function report_select_html(io::IO, entries::AbstractVector{<:ReportEntry},
                            groups::AbstractVector)
    option(k) = string("<option value=\"entry-", k, "\">", html_escape(entries[k].name),
                       "</option>")
    print(io, "<label>Portfolio <select id=\"po-select\">")
    for k in eachindex(entries)
        isnothing(entries[k].group) && print(io, option(k))
    end
    for g in groups
        print(io, "<optgroup label=\"", html_escape(g), "\">")
        for k in eachindex(entries)
            isequal(entries[k].group, g) && print(io, option(k))
        end
        print(io, "</optgroup>")
    end
    print(io, "</select></label>")
    return nothing
end
"""
    report_format(format::Nothing, path::AbstractString)
    report_format(format::MIME"text/markdown", path::AbstractString)
    report_format(format::MIME"text/html", path::AbstractString)
    report_format(format::MIME, path::AbstractString)

The format in which [`write_report`](@ref) writes a report: the format of the caller, or the format that the extension of `path` names.

# Arguments

  - `format`: The MIME of the format, or `nothing` to read the extension of `path`.
  - `path`: The path of the file.

# Validation

  - With no format, the extension of `path` is `.md`, `.html` or `.htm`. Otherwise the function raises an `ArgumentError`.
  - A given format is `text/markdown` or `text/html`. Otherwise the function raises an `ArgumentError`.

# Returns

  - `mime::MIME`: The format.

# Related

  - [`write_report`](@ref)
"""
function report_format(::Nothing, path::AbstractString)
    ext = lowercase(last(splitext(path)))
    if ext == ".md"
        return MIME"text/markdown"()
    elseif ext in (".html", ".htm")
        return MIME"text/html"()
    end
    return throw(ArgumentError("the extension of `$(path)` names no format of a report: give a path that ends in `.md`, `.html` or `.htm`, or pass `format = MIME\"text/markdown\"()` or `format = MIME\"text/html\"()`"))
end
function report_format(format::Union{MIME"text/markdown", MIME"text/html"},
                       ::AbstractString)
    return format
end
function report_format(format::MIME, ::AbstractString)
    return throw(ArgumentError("a report is written as `text/markdown` or `text/html`, and not as `$(format)`"))
end
"""
    write_report(path::AbstractString, r::Report; format::Option{<:MIME} = nothing,
                 figure_mime::MIME = MIME"image/svg+xml"()) -> String

Write a Portfolio Report to a file, as Markdown or as one self-contained HTML file.

The format comes from the extension of `path` when `format` is `nothing`: `.md` for Markdown, `.html` or `.htm` for HTML. The figures are SVG images by default. An SVG image holds every point of its series, so a report of many entries with daily series over many years can be large. `figure_mime = MIME"image/png"()` gives raster images, which have a size that does not grow with the series. A figure that cannot show itself in the chosen MIME takes the other image MIME. `show(io, MIME"text/html"(), r)` writes the same bytes as this function with the default `figure_mime`.

# Arguments

  - `path`: The path of the file.
  - `r`: The report.
  - `format`: The MIME of the format, `MIME"text/markdown"()` or `MIME"text/html"()`, or `nothing`.
  - `figure_mime`: The image MIME of the figures, `MIME"image/svg+xml"()` or `MIME"image/png"()`.

# Validation

  - The rules of [`report_format`](@ref).
  - `figure_mime` is `image/svg+xml` or `image/png`. Otherwise the function raises an `ArgumentError`.

# Returns

  - `path::String`: The path of the file.

# Related

  - [`Report`](@ref)
  - [`portfolio_report`](@ref)
  - [`report_format`](@ref)
"""
function write_report(path::AbstractString, r::Report; format::Option{<:MIME} = nothing,
                      figure_mime::MIME = MIME"image/svg+xml"())
    @argcheck(figure_mime in (MIME"image/svg+xml"(), MIME"image/png"()),
              ArgumentError("the figures of a report are written as `image/svg+xml` or `image/png`, and not as `$(figure_mime)`"))
    mime = report_format(format, path)
    open(path, "w") do io
        return show(IOContext(io, :report_figure_mime => figure_mime), mime, r)
    end
    return String(path)
end

export write_report
