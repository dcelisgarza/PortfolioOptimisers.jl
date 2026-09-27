# Writes the type hierarchy page, `docs/src/TypeHierarchy.md`. The page sits beside `00_API.md`
# and `capability_catalogue.md`, and not inside a mirror tree, because its trees link the types of
# both mirror trees (ADR 0128).
#
# For each root abstract type, the script walks the subtype tree and writes it as a text tree in
# the box-drawing style of the ADRs. Each type name is a Documenter `@ref` link to its docstring.
#
# Notes on Documenter's HTML writer:
#   * The tree is plain markdown and not a fenced block, because an `@ref` link does not
#     resolve inside a ```` ``` ```` code fence.
#   * Each type name is a code span, because Documenter reads a code span in a link as a docstring
#     reference (see `_node`). The theme gives a code span its own background and padding, which
#     break the rows of the tree, so `.type-tree code` in `docs/src/assets/generated-pages.css`
#     removes them.
#   * Documenter escapes a bare `<div>` or `<br>` in markdown text to `&lt;div&gt;`. So the script
#     writes the wrapper `<div class="type-tree">` in a `@raw html` block, which Documenter copies
#     to the page unchanged. The markdown tree goes between the opening and the closing raw
#     blocks, so its `@ref` links still resolve.
#   * The whole tree is one paragraph. The script ends each entry but the last with `\`, which
#     Markdown reads as a hard line break and renders as a `<br>`. With one paragraph per entry,
#     the theme puts a `margin-bottom` on each entry, and the tree shows a blank line between its
#     rows. A `<br>` takes no margin, so the rows stay together under any theme.
#   * The indentation uses the NO-BREAK SPACE character U+00A0, not the HTML entity `&nbsp;`.
#     Documenter escapes markdown text, so the entity reaches the page as the literal string
#     `&nbsp;`. The character passes through the escape, and HTML does not collapse a run of it.
#     An ASCII space does not work, because four of them start an indented code block.

using PortfolioOptimisers, StatsBase, InteractiveUtils

const _NBSP = "\u00a0"  # NO-BREAK SPACE, U+00A0. See the header note.

# A type is linkable when PortfolioOptimisers registers a docstring for it. An `@docs` block
# renders each such docstring, so the `@ref` resolves. The name of a foreign or an undocumented
# type is a code span with no link.
function is_linkable(T::Type)
    return haskey(Base.Docs.meta(PortfolioOptimisers),
                  Base.Docs.Binding(parentmodule(T), nameof(T)))
end

function _node(T::Type)
    # The name is a code span. Documenter reads the text of a link to find the target of the
    # `@ref`. It reads a code span as a docstring reference, and plain text as a heading reference
    # only. Every link here must reach a docstring, so every name is a code span, with a link or
    # without one.
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
    # A backslash at the end of a line is the Markdown hard line break, and the last entry has
    # none. The blank line after the last entry ends the paragraph, so Markdown does not read the
    # `@raw html` fence that follows as a part of it.
    return string(join(_type_tree(String[], T), "\\\n"), "\n\n")
end

function generate_type_hierarchy(path::String = joinpath(@__DIR__, "src",
                                                         "TypeHierarchy.md"))
    roots = ["AbstractResult" => PortfolioOptimisers.AbstractResult,
             "AbstractEstimator" => PortfolioOptimisers.AbstractEstimator,
             "AbstractAlgorithm" => PortfolioOptimisers.AbstractAlgorithm,
             "AbstractCovarianceEstimator" =>
                 PortfolioOptimisers.AbstractCovarianceEstimator]
    open(path, "w") do io
        print(io,
              """
              ```@meta
              Description = "Every result, estimator, algorithm and covariance estimator type of PortfolioOptimisers.jl as a subtype tree, with links to the docstrings."
              ```

              # Type hierarchy

              Each tree below starts at one abstract type of `PortfolioOptimisers.jl` and
              lists every subtype under it. The documentation build writes the trees from
              the types of the loaded package with
              [docs/generate_type_hierarchy.jl](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/docs/generate_type_hierarchy.jl),
              so a new type appears here at the next build. Each documented type links to
              its docstring, on a public API page or on a private API page.

              The [capability catalogue](@ref capability-catalogue) lists the same types
              by the job each one does.
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
