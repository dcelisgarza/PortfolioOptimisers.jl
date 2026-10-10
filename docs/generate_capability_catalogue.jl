# Writes the Capability Catalogue page, `docs/src/capability_catalogue.md`, from
# `capability_catalogue.jl` (ADR 0040).
#
# The catalogue file holds the grouping alone. For each `Cap`, this file takes the one-line
# description at build time from the docstring of the first name. The description is the first
# sentence of the paragraph after `$(DocStringExtensions.TYPEDEF)`. So the page has a
# description for each type with no second copy of it. `docs/src/contribute/2-developer.md`
# states the rule for a contributor, and `summary_paragraph` fails the build when a docstring
# does not follow it.
#
# This file reads the docstring with `Base.Docs.doc` and not from the source text. `Base.Docs.doc`
# gives the docstring with the `$(TYPEDEF)` and `$(FIELDS)` interpolations expanded, so the first
# `Markdown.Paragraph` is the summary. The source text gives the `$(...)` expressions unexpanded.
#
# Notes on Documenter's HTML writer. `generate_type_hierarchy.jl` has the same constraints.
#   * A group that a reader can collapse is a raw HTML `<details>` block, not a `!!! details`
#     admonition. The Markdown parser accepts only a quoted title, `!!! details "Title"`, and
#     renders it as plain text, so the head of the group would lose its `@ref` links. The
#     unquoted form `- !!! details <head>`, which the page used before, renders as literal text
#     and not as a block that collapses. So the script writes the `<details>` and `<summary>`
#     tags in `@raw html` blocks, and the head and the body stay ordinary markdown between them.
#     Every link then resolves.
#   * A `@raw html` block is a fence, and a fence cannot go inside a list item. Each group therefore
#     ends the list around it, and a `margin-left` gives back the indent of the list. The value
#     is `(indent + 2)em`, because the theme gives `.content ul` a margin of 2em per level of
#     nesting, and `indent` counts two spaces per level. The children of the group start again
#     at indent 0 inside the `<details>`, so the margins add up. See
#     `docs/src/assets/generated-pages.css`.
#   * The rest of the page is plain markdown, because an `@ref` link does not resolve inside
#     a code fence.

using PortfolioOptimisers, Markdown, InteractiveUtils

include(joinpath(@__DIR__, "capability_catalogue.jl"))

const _PAGE_TITLE = "Capability catalogue"

"""
    summary_sentence(name) -> String

The first sentence of the summary paragraph of the docstring of `name`.

Throws an error when the docstring has no summary, and writes no placeholder. A capability with
no summary is a defect of its docstring, and an empty bullet on the page would hide that defect.
"""
function summary_sentence(name::Union{Symbol, String})
    para = summary_paragraph(name)
    return first_sentence(para)
end

function summary_paragraph(name::Union{Symbol, String})
    sym = name isa Symbol ? name : Symbol(name)
    if !isdefined(PortfolioOptimisers, sym)
        error("capability_catalogue: `$name` is not defined in PortfolioOptimisers.")
    end
    md = Base.Docs.doc(Base.Docs.Binding(PortfolioOptimisers, sym))
    para = first_paragraph(md)
    if isnothing(para)
        error("""capability_catalogue: `$name` has no summary paragraph.

               Every type docstring must start with `\$(DocStringExtensions.TYPEDEF)`,
               a blank line and a one-line summary. Add the summary, or give the `Cap`
               of `$name` a `label` in `capability_catalogue.jl` if one bullet cannot
               summarise `$name`.""")
    end
    return para
end

"""
    first_paragraph(md) -> Union{String, Nothing}

The summary paragraph, which is the first paragraph of a docstring after the code block that
`\$(TYPEDEF)` expands to.

Stops at the first heading and returns `nothing`, and does not search past it. A docstring with
no summary still has paragraphs under a later heading, such as `# Constructors` or `# Related`.
If this function returned one of them, the page would show a sentence about something else as
the description of the type. The `nothing` makes `summary_paragraph` throw an error that names
the type, and the build fails.
"""
function first_paragraph(md)
    if !(md isa Markdown.MD)
        return nothing
    end
    for block in md.content
        if block isa Markdown.Header
            # A heading before any paragraph means that the docstring has no summary.
            return nothing
        elseif block isa Markdown.MD
            # A function with several documented methods nests one `MD` per
            # method, so the summary is one level down.
            para = first_paragraph(block)
            isnothing(para) || return para
        elseif block isa Markdown.Paragraph
            para = strip(sprint(Markdown.plain, block))
            isempty(para) || return para
        end
    end
    return nothing
end

# Abbreviations whose full stop does not end a sentence. Every other `.`, `!` or `?` before
# whitespace ends one.
const _ABBREV = r"(?:e\.g|i\.e|cf|et al|vs|approx|Fig|Eq|Ref|Sec|Dr|Mr|Ms|St)$"

"""
    first_sentence(text) -> String

Cut a summary paragraph to its first sentence, and keep the mark that ends it.

The summaries are short statements. The one risk is the full stop of an abbreviation, which
`_ABBREV` catches. A `.` inside inline code or maths, such as the one in `x.y`, has no
whitespace after it, so the split does not cut there.
"""
function first_sentence(text::AbstractString)
    text = replace(strip(text), r"\s*\n\s*" => " ")
    for m in eachmatch(r"[.!?](?=\s|$)", text)
        head = text[1:prevind(text, m.offset)]
        if occursin(_ABBREV, head)
            continue
        end
        return text[1:m.offset]
    end
    return text
end

ref(name::Symbol) = string("[`", name, "`](@ref)")
ref(name::String) = string("[`", name, "`](@ref)")

# Joins two links with " and ", and three or more with ", " and a last ", and ". This is
# the style of the hand-written section that this page replaced.
function join_refs(names)
    strs = ref.(names)
    n = length(strs)
    return if n == 0
        ""
    elseif n == 1
        strs[1]
    elseif n == 2
        string(strs[1], " and ", strs[2])
    else
        string(join(strs[1:(end - 1)], ", "), ", and ", strs[end])
    end
end

"""
    cap_text(c::Cap) -> String

The text of one capability bullet, which is its description and then the `@ref` links to its
docstrings.

A label that already contains links goes to the page unchanged. The docstring of `Cap` says why
some entries write their own links.
"""
function cap_text(c::Cap)
    if !isnothing(c.label) && occursin("](@ref)", c.label)
        return c.label
    end
    desc = isnothing(c.label) ? summary_sentence(first(c.names)) : c.label
    links = join_refs(c.names)
    return isempty(links) ? desc : isempty(desc) ? links : string(desc, " ", links)
end

head_text(h::Cap) = cap_text(h)
head_text(h::String) = h

# `depth` counts the nesting of Sections, and `indent` counts the nesting of lists. Each one
# changes without the other. A Section starts the list again at indent 0. A Group does too,
# because a `<details>` starts a new list inside it, and the `margin-left` of the group keeps
# the level of the list that it ended.
function render(io::IO, node::Section, depth::Int, indent::Int)
    println(io, "\n", "#"^depth, " ", node.title, "\n")
    for child in node.children
        render(io, child, depth + 1, 0)
    end
    return io
end
function render(io::IO, node::Prose, depth::Int, indent::Int)
    # A blank line goes on each side, because markdown joins a paragraph and a list next to it
    # into one block when no blank line separates them. `render_catalogue` removes the extra
    # blank lines that this makes.
    #
    # Inside a list, the paragraph must also start at the content column of the item around it.
    # If it does not, it ends the list, and the bullets after it are a new block indented four
    # spaces, which markdown reads as a code block. Documenter does not resolve an `@ref` inside
    # code, so each link below such a paragraph stays as the literal text `(@ref)`, and the site
    # builder reports a dead `./@ref` link.
    pad = " "^indent
    body = join((string(pad, line) for line in split(node.text, '\n')), "\n")
    println(io, "\n", body, "\n")
    return io
end
function render(io::IO, node::Note, depth::Int, indent::Int)
    println(io, " "^indent, "- ", node.text)
    for child in node.children
        render(io, child, depth, indent + 2)
    end
    return io
end
function render(io::IO, node::Cap, depth::Int, indent::Int)
    println(io, " "^indent, "- ", cap_text(node))
    return io
end

"""
    raw_html(io, html)

Write `html` unchanged in a `@raw html` block.

A blank line goes on each side. Markdown reads a fence with no blank line after a paragraph as a
part of that paragraph, and a fence with no blank line after a list as a lazy continuation of the
last item.
"""
function raw_html(io::IO, html::AbstractString)
    println(io, "\n```@raw html\n", html, "\n```\n")
    return io
end

function render(io::IO, node::Group, depth::Int, indent::Int)
    # The head and the body stay markdown between the raw blocks, so their `@ref`
    # links resolve. The header note explains the value of `margin-left`.
    raw_html(io,
             string("<details class=\"cap-group\" style=\"margin-left: ", indent + 2,
                    "em\">\n<summary>"))
    println(io, "\n", head_text(node.head), "\n")
    raw_html(io, "</summary>")
    # A new list starts inside the `<details>`, so the indent of the children starts at 0.
    for child in node.children
        render(io, child, depth, 0)
    end
    raw_html(io, "</details>")
    return io
end

"""
    render_catalogue(io; base_level = 2)

Render `CATALOGUE` as markdown. `base_level` is the heading level of a top-level `Section`.
It lets a caller compare the output with the section of `00_API.md` that this page came from,
which was one level deeper.
"""
function render_catalogue(io::IO = IOBuffer(); base_level::Int = 2)
    buf = IOBuffer()
    for node in CATALOGUE
        render(buf, node, base_level, 0)
    end
    print(io, collapse_blank_runs(String(take!(buf))))
    return io
end

"""
    collapse_blank_runs(md) -> String

Replace every run of blank lines in `md` with one blank line.

Each block writes its own blank lines and does not know what comes before it. Two blocks in
a row often leave a run. Apply this function to the text that the generator writes to the file.
When it ran on the rendered body alone, the join of the preamble and the body kept a double
blank line, and `markdownlint` removed it again on every commit.
"""
function collapse_blank_runs(md::AbstractString)::String
    return replace(md, r"\n{3,}" => "\n\n")
end

const _PREAMBLE = """
```@meta
Description = "Every estimator, risk measure, constraint and optimiser in PortfolioOptimisers.jl, grouped by the job it does and linked to its docstring."
```

# [$(_PAGE_TITLE)](@id capability-catalogue)

This page lists what `PortfolioOptimisers.jl` can do, grouped by the job that each
type or function does. Each entry links to its docstring, on a public API page, or
on a private API page for a name outside the public API.

The documentation build writes this page with
[docs/generate_capability_catalogue.jl](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/docs/generate_capability_catalogue.jl).
The grouping comes from `docs/capability_catalogue.jl`. The description of an entry
is the first sentence of its docstring, unless the catalogue gives the entry its own
label. The build fails when a type that you can choose, such as an estimator or an
algorithm, has no entry here.

The [type hierarchy](@ref type-hierarchy-AbstractEstimator) lists the same types by
their supertypes.
"""

"""
    declared_leaves(T, acc = Set{Type}()) -> Set{Type}

Every leaf under `T` that `PortfolioOptimisers` declares.

A leaf is a type that is not abstract and has no subtypes. The test is `!isabstracttype` and
not `isconcretetype`. Almost every struct here is `@concrete`, so its bare name is a
`UnionAll`, and `isconcretetype` is false for each one.

The `parentmodule` filter matters in `test/test_26_docs.jl`, which uses this rule too. The
runner gives each test file its own module but not its own process, so an estimator that
another test file declares stays in the worker, and `subtypes` finds it here. The catalogue
lists the types of the package alone, so a leaf from another module is not a member. The set of
files that share a worker changes from run to run. Without this filter, the test fails on
some runs and passes on others.
"""
function declared_leaves(T, acc = Set{Type}())
    subs = subtypes(T)
    if isempty(subs)
        if !isabstracttype(T) && parentmodule(T) === PortfolioOptimisers
            push!(acc, T)
        end
    else
        foreach(S -> declared_leaves(S, acc), subs)
    end
    return acc
end

"""
    declared_descendants(T, acc = Set{Type}()) -> Set{Type}

Every type under `T`, at any depth, that is not abstract and that `PortfolioOptimisers`
declares.

Unlike [`declared_leaves`](@ref), this also keeps a type that is not abstract and has subtypes,
so `choice_surface_names` removes a whole family.
"""
function declared_descendants(T, acc = Set{Type}())
    if !isabstracttype(T) && parentmodule(T) === PortfolioOptimisers
        push!(acc, T)
    end
    foreach(S -> declared_descendants(S, acc), subtypes(T))
    return acc
end

"""
    exported_concrete_types() -> Set{Type}

Every type that is not abstract and that `PortfolioOptimisers` exports under its own name.

The function skips an alias, because the `nameof` of an alias is the name of the type that it
points at. `HRP` and `HierarchicalRiskParity` are one capability, and the catalogue lists the
long name. The docstring of `HRP` names the long name for a reader.
"""
function exported_concrete_types()
    acc = Set{Type}()
    for n in names(PortfolioOptimisers)
        if !Base.isexported(PortfolioOptimisers, n) || contains(string(n), "#")
            continue
        end
        v = getfield(PortfolioOptimisers, n)
        if !(v isa Type)
            continue
        end
        T = v isa UnionAll ? Base.unwrap_unionall(v) : v
        if !(T isa DataType) || isabstracttype(T)
            continue
        end
        if parentmodule(T) !== PortfolioOptimisers || nameof(T) != n
            continue
        end
        push!(acc, T)
    end
    return acc
end

"""
    choice_surface_names() -> Set{Symbol}

The names that the catalogue must list, before the removal of the names in `NOT_A_CHOICE`.

This function is the Choice Surface of `GLOSSARY.md` § 1 in code, and the one statement of the
coverage rule. `test/test_26_docs.jl` calls it too. A type that `PortfolioOptimisers` declares,
and that is not abstract, is on the surface when one of these rules holds:

  - it is a leaf subtype of `AbstractEstimator`, of `AbstractAlgorithm` or of
    `AbstractCovarianceEstimator`, exported or not;
  - the package exports it under its own name.

The function then removes two families, because a caller receives them and never chooses
them. They are the `AbstractResult` family, which includes `OptimisationReturnCode`, and the
`PortfolioOptimisersError` family.

The roots alone left a gap, issue #636. No rule required an entry for a family outside every
root, so the catalogue listed such a family only when an author remembered it.
`AbstractCovarianceEstimator` is a root for that reason. It descends from
`StatsBase.CovarianceEstimator` and not from `AbstractEstimator`, so the two roots before it
reached none of its members, exported or not.
"""
function choice_surface_names()
    required = union(declared_leaves(PortfolioOptimisers.AbstractEstimator),
                     declared_leaves(PortfolioOptimisers.AbstractAlgorithm),
                     declared_leaves(PortfolioOptimisers.AbstractCovarianceEstimator),
                     exported_concrete_types())
    setdiff!(required, declared_descendants(PortfolioOptimisers.AbstractResult),
             declared_descendants(PortfolioOptimisers.PortfolioOptimisersError))
    return Set(nameof.(collect(required)))
end

"""
    assert_complete()

Throw an error, and write no page, when a choice has no entry in the catalogue.

A type in `NOT_A_CHOICE` is exempt, because the library constructs it for itself and a reader
never chooses it.

`test/test_26_docs.jl` makes the same check earlier, on the pull request that adds the type.
This function checks again because a page with a missing capability shows no sign of the gap,
and a reader takes it as complete.
"""
function assert_complete()
    catalogued = Set{Symbol}()
    scan_text(t::AbstractString) =
        for m in eachmatch(r"\[`([^`]+)`\]\(@ref\)", t)
            push!(catalogued, Symbol(m.captures[1]))
        end
    scan(c::Cap) = (foreach(n -> push!(catalogued, n isa Symbol ? n : Symbol(n)), c.names);
                    isnothing(c.label) || scan_text(c.label))
    scan(n::Prose) = scan_text(n.text)
    scan(n::Note) = (scan_text(n.text); foreach(scan, n.children))
    scan(n::Section) = foreach(scan, n.children)
    scan(n::Group) = (n.head isa Cap ? scan(n.head) : scan_text(n.head);
                      foreach(scan, n.children))
    foreach(scan, CATALOGUE)

    required = choice_surface_names()
    setdiff!(required, keys(NOT_A_CHOICE))
    missed = sort(collect(setdiff(required, catalogued)))
    if !isempty(missed)
        error("""capability_catalogue: $(length(missed)) choice(s) have no entry, so the page
                 would be incomplete. Add a `Cap` for each to
                 `docs/capability_catalogue.jl`, or give it an entry in `NOT_A_CHOICE`
                 with a reason:\n  $(join(missed, "\n  "))""")
    end
    return nothing
end

"""
    assert_refs_survive(md)

Check that every `@ref` that the generator writes is still a link after markdown parses the
page.

The text around an `[`X`](@ref)` decides whether it stays a link. Two defects broke links on
the page before this check existed:

  - A bare `_` in a description pairs with the `_` in a `snake_case` link next to it.
    Markdown reads the two as emphasis and drops the brackets. `(f_μ vector)` next to
    ``[`plot_factor_mu`](@ref)`` rendered as `(fμ vector … [`plotfactor_`.
  - A paragraph at column 0 inside a list ends the list, so markdown parses the bullets after
    it, indented four spaces, as a code block.

Documenter resolves `@ref` only in a real link. In both cases the text reaches the site builder
unchanged. The builder then reports one dead `./@ref` link for any number of lost links, and it
does not say where on the page they are. This check names each lost link.

The check uses the `Markdown` stdlib, and not the CommonMark parser that Documenter uses. The
two parsers differ in rare cases. The check finds the common defects and does not replace
the docs build.
"""
function assert_refs_survive(md::AbstractString)
    written = [m.captures[1] for m in eachmatch(r"\[`([^`]+)`\]\(@ref\)", md)]
    survived = String[]
    # Walk each field that holds children, and not a list of node types. `Markdown` names the
    # children `content`, `items` or `text` by the node, and a list of node types would skip a
    # type that a later version of the stdlib adds.
    function walk(node)
        if node isa AbstractVector
            foreach(walk, node)
            return nothing
        end
        if node isa Markdown.Link && node.url == "@ref"
            # The text of a catalogue link is one code span, so take its literal
            # contents. A render of the node gives back the whole `[`X`](@ref)`,
            # which matches no written name.
            buf = IOBuffer()
            for t in node.text
                if t isa Markdown.Code
                    print(buf, t.code)
                elseif t isa AbstractString
                    print(buf, t)
                else
                    nothing
                end
            end
            push!(survived, strip(String(take!(buf))))
        end
        for f in (:content, :items, :text)
            if hasproperty(node, f)
                child = getproperty(node, f)
                child isa AbstractString || walk(child)
            end
        end
        return nothing
    end
    walk(Markdown.parse(md))

    lost = copy(written)
    for s in survived
        i = findfirst(==(strip(s, '`')), strip.(lost, '`'))
        isnothing(i) || deleteat!(lost, i)
    end
    if !isempty(lost)
        error("""capability_catalogue: $(length(lost)) `@ref` link(s) are not links after
                 markdown parses the page, and the site builder would report them as dead
                 `./@ref` links:
                 \n  $(join(unique(lost), "\n  "))\n
                 One usual cause is a bare `_` in a description that pairs with the `_` of a
                 `snake_case` link next to it. Put the bare `_` in backticks. The other is a
                 `Prose` node that ends a list and turns the bullets below it into a code
                 block.""")
    end
    return nothing
end

function generate_capability_catalogue(path::String = joinpath(@__DIR__, "src",
                                                               "capability_catalogue.md"))
    assert_complete()
    body = collapse_blank_runs(string(_PREAMBLE, "\n",
                                      String(take!(render_catalogue(IOBuffer())))))
    assert_refs_survive(body)
    write(path, body)
    return path
end
