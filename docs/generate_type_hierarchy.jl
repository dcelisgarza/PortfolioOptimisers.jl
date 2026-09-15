# Auto-generates the type-hierarchy API page (`<NN>_TypeHierarchy.md`). The
# numeric prefix is chosen dynamically so the page always sorts last in
# `docs/src/api/`, even after new numbered files are added before it.
#
# Walks the subtype tree of each root abstract type and renders it as an ASCII
# tree (same box-drawing style as the ADRs), with each type name a Documenter
# `@ref` link to its docstring.
#
# Rendering notes (Documenter's HTML writer):
#   * `@ref` links cannot live inside a ```` ``` ```` code fence, so the tree is
#     plain markdown, not a fenced block.
#   * Each type name is a backticked code span, because that is what makes the
#     `@ref` a docstring reference (see `_node`). The theme paints a code span
#     with its own background and padding, which breaks the rows of the tree, so
#     `.type-tree code` in `docs/src/assets/generated-pages.css` flattens it.
#   * A bare `<div>`/`<br>` written into markdown text gets HTML-escaped
#     (`&lt;div&gt;`), so the wrapper `<div class="type-tree">` is emitted via a
#     `@raw html` block, which passes through verbatim. The markdown tree lives
#     *between* the open/close raw blocks so its `@ref` links still resolve.
#   * The whole tree is ONE paragraph, and the entries are separated by a
#     Markdown hard line break: a line that ends with `\` becomes a `<br>`. One
#     paragraph per entry (blank line between) is the obvious alternative, but
#     the theme then puts a `margin-bottom` on every entry and the tree renders
#     with a blank line between its rows. A `<br>` cannot pick up a margin, so
#     the rows stay together whatever the theme does.
#   * Indentation uses the NO-BREAK SPACE character U+00A0, NOT the HTML entity
#     `&nbsp;`. Documenter escapes markdown text, so the entity would reach the
#     page as the literal string `&nbsp;`. The character survives escaping, and
#     HTML does not collapse a run of it. A literal ASCII space cannot be used:
#     four of them start an indented code block.

using PortfolioOptimisers, StatsBase, InteractiveUtils

const _NBSP = "\u00a0"  # NO-BREAK SPACE, U+00A0. See the header note.

# A type is linkable iff it has a docstring registered in PortfolioOptimisers
# (every such docstring is rendered via an `@docs` block, so the `@ref`
# resolves). Foreign / undocumented types fall back to plain text.
function is_linkable(T::Type)
    return haskey(Base.Docs.meta(PortfolioOptimisers),
                  Base.Docs.Binding(parentmodule(T), nameof(T)))
end

function _node(T::Type)
    # The name is backticked. Documenter reads the text of a link to decide what
    # the `@ref` points at: backticked text is a docstring reference, and plain
    # text is a heading reference and nothing else. Every node here must reach a
    # docstring, so every node name carries the backticks, linked or not.
    name = string("`", nameof(T), "`")
    if !(is_linkable(T))
        return name
    end
    return string("[", name, "](@ref)")
end

function _type_tree(lines::Vector{String}, T::Type; prefix::String = "",
                    is_last::Bool = true, is_root::Bool = true)
    if is_root
        push!(lines, _node(T))
    else
        branch = is_last ? "└──$(_NBSP)" : "├──$(_NBSP)"
        push!(lines, string(prefix, branch, _node(T)))
    end
    subs = sort!(subtypes(T); by = x -> string(nameof(x)))
    child_prefix = is_root ? "" : prefix * (is_last ? _NBSP^4 : "│$(_NBSP^3)")
    for (i, S) in enumerate(subs)
        _type_tree(lines, S; prefix = child_prefix, is_last = i == length(subs),
                   is_root = false)
    end
    return lines
end

function type_tree(T::Type)
    # A trailing backslash before the newline is the Markdown hard line break.
    # The last entry carries none. The blank line after it closes the paragraph,
    # so the `@raw html` fence that follows is not read as part of it.
    return string(join(_type_tree(String[], T), "\\\n"), "\n\n")
end

const _PAGE_SUFFIX = "_TypeHierarchy.md"

# Highest `NN_` prefix among the entries in `dir` (files and directories alike,
# since both are numbered in `src/api`), or -1 if none.
function max_api_index(dir::String)
    maxn = -1
    for entry in readdir(dir)
        m = match(r"^(\d+)_", entry)
        if m === nothing
            continue
        end
        maxn = max(maxn, parse(Int, m.captures[1]))
    end
    return maxn
end

function generate_type_hierarchy(dir::String = joinpath(@__DIR__, "src", "api"))
    # Drop any previously generated page first, so its own prefix never inflates
    # the index and no stale duplicate is left behind when the number changes.
    for entry in readdir(dir)
        if endswith(entry, _PAGE_SUFFIX)
            rm(joinpath(dir, entry))
        end
    end
    idx = max_api_index(dir) + 1
    path = joinpath(dir, string(lpad(idx, 2, '0'), _PAGE_SUFFIX))
    roots = ["AbstractResult" => PortfolioOptimisers.AbstractResult,
             "AbstractEstimator" => PortfolioOptimisers.AbstractEstimator,
             "AbstractAlgorithm" => PortfolioOptimisers.AbstractAlgorithm,
             "AbstractCovarianceEstimator" =>
                 PortfolioOptimisers.AbstractCovarianceEstimator]
    open(path, "w") do io
        print(io,
              """
              # Type hierarchy

              The trees below are generated automatically from the live type hierarchy
              every time the documentation is built (see [docs/generate_type_hierarchy.jl](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/docs/generate_type_hierarchy.jl)),
              so they always reflect the current state of the package. Each type links to
              its docstring.
              """)
        for (name, T) in roots
            println(io, "\n## [", name, "](@id type-hierarchy-", name, ")\n")
            println(io, "```@raw html")
            println(io, "<div class=\"type-tree\">")
            println(io, "```\n")
            print(io, type_tree(T))
            println(io, "```@raw html")
            println(io, "</div>")
            println(io, "```")
        end
    end
    return path
end
