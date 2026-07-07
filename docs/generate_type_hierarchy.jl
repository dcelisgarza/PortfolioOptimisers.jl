# Auto-generates the type-hierarchy API page (`<NN>_TypeHierarchy.md`). The
# numeric prefix is chosen dynamically so the page always sorts last in
# `docs/src/api/`, even after new numbered files are added before it.
#
# Walks the subtype tree of each root abstract type and renders it as an ASCII
# tree (same box-drawing style as the ADRs), with each type name a Documenter
# `@ref` link to its docstring.
#
# Rendering notes (Documenter + DocumenterVitepress):
#   * `@ref` links cannot live inside a ```` ``` ```` code fence, so the tree is
#     plain markdown, not a fenced block.
#   * A bare `<div>`/`<br>` written into markdown text gets HTML-escaped
#     (`&lt;div&gt;`), so the wrapper `<div class="type-tree">` is emitted via a
#     `@raw html` block, which passes through verbatim. The markdown tree lives
#     *between* the open/close raw blocks so its `@ref` links still resolve.
#   * Line breaks come from one paragraph per entry (blank line between), styled
#     `margin: 0` in `theme/style.css` — no `<br>` needed.
#   * Indentation uses `&nbsp;` (survives Documenter intact) rather than literal
#     spaces, which would collapse / be parsed as an indented code block.

using PortfolioOptimisers, StatsBase, InteractiveUtils

const _NBSP = "&nbsp;"

# A type is linkable iff it has a docstring registered in PortfolioOptimisers
# (every such docstring is rendered via an `@docs` block, so the `@ref`
# resolves). Foreign / undocumented types fall back to plain text.
function is_linkable(T::Type)
    return haskey(Base.Docs.meta(PortfolioOptimisers),
                  Base.Docs.Binding(parentmodule(T), nameof(T)))
end

function _node(T::Type)#; qualified::Bool = false)
    name = string(nameof(T))
    if !(is_linkable(T))
        return name
    end
    # The root repeats the section heading, so a bare `@ref` would resolve to the
    # heading anchor rather than the docstring; qualifying it (`@ref Mod.Name`)
    # targets the docstring directly.
    # target = qualified ? string(" ", parentmodule(T), ".", name) : ""
    return string("[", name, "](@ref)")
end

function _type_tree(io::IO, T::Type; prefix::String = "", is_last::Bool = true,
                    is_root::Bool = true)
    if is_root
        println(io, _node(T))
    else
        branch = is_last ? "└──$(_NBSP)" : "├──$(_NBSP)"
        println(io, prefix, branch, _node(T))
    end
    println(io)  # blank line => each entry is its own (zero-margin) paragraph
    subs = sort!(subtypes(T); by = x -> string(nameof(x)))
    child_prefix = is_root ? "" : prefix * (is_last ? _NBSP^4 : "│$(_NBSP^3)")
    for (i, S) in enumerate(subs)
        _type_tree(io, S; prefix = child_prefix, is_last = i == length(subs),
                   is_root = false)
    end
    return io
end

function type_tree(T::Type)
    return String(take!(_type_tree(IOBuffer(), T)))
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
        println(io,
                """
                # Type hierarchy

                The trees below are generated automatically from the live type hierarchy
                every time the documentation is built (see [docs/generate_type_hierarchy.jl](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/docs/generate_type_hierarchy.jl)),
                so they always reflect the current state of the package. Each type links to
                its docstring.
                """)
        for (name, T) in roots
            println(io, "## [", name, "](@id type-hierarchy-", name, ")\n")
            println(io, "```@raw html")
            println(io, "<div class=\"type-tree\">")
            println(io, "```\n")
            print(io, type_tree(T))
            println(io, "```@raw html")
            println(io, "</div>")
            println(io, "```\n")
        end
    end
    return path
end
