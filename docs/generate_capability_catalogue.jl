# Renders the Capability Catalogue page from `capability_catalogue.jl` (ADR 0040).
#
# The catalogue file curates only the *grouping*. Each `Cap`'s one-line
# description is resolved here, at build time, from the first sentence of the
# first paragraph of its leading name's docstring -- the paragraph that follows
# `$(DocStringExtensions.TYPEDEF)`. That convention holds for every documented
# type in the package and is what lets the page carry prose without keeping a
# second copy of it; see `docs/src/contribute/` for the contributor-facing rule.
#
# Resolving via `Base.Docs.doc` rather than by reading source text means the
# `$(TYPEDEF)` / `$(FIELDS)` interpolations are already expanded, so the first
# `Markdown.Paragraph` in the rendered docstring is the summary. Reading the raw
# string would instead hand back the uninterpolated `$(...)` expressions.
#
# Rendering notes (Documenter + DocumenterVitepress) -- see also
# `generate_type_hierarchy.jl`, which shares this page's constraints:
#   * `::: details` containers work *inside* a list item: the bullet body becomes
#     the `<summary>` and the nested sub-list becomes the disclosure body. They
#     are never explicitly closed; the list nesting closes them.
#   * `@ref` links must not sit inside a code fence, so everything here is plain
#     markdown.

using PortfolioOptimisers, Markdown, InteractiveUtils

include(joinpath(@__DIR__, "capability_catalogue.jl"))

const _PAGE_TITLE = "Capability catalogue"

"""
    summary_sentence(name) -> String

The first sentence of `name`'s docstring summary paragraph.

Errors rather than falling back to a placeholder: a capability with no usable
summary is a docstring bug, and silently rendering an empty bullet would hide
exactly the kind of gap this page exists to prevent.
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

               Every type docstring must open with `\$(DocStringExtensions.TYPEDEF)`
               followed by a blank line and a one-line summary. Add one, or pass an
               explicit `label` in `capability_catalogue.jl` if `$name` genuinely
               cannot be summarised in a bullet.""")
    end
    return para
end

"""
    first_paragraph(md) -> Union{String, Nothing}

The summary paragraph: the first prose block of a docstring, skipping the code
block that `\$(TYPEDEF)` expands to.

Stops at the first heading and returns `nothing` rather than searching past it.
That matters more than it looks: a docstring whose summary has been deleted
still contains paragraphs further down, under `# Constructors` or `# Related`,
and happily returning one of those would put an unrelated sentence on the
catalogue page and call it a description -- silent drift, which is the whole
thing this page exists to prevent. Refusing here turns it into a build error
naming the type.
"""
function first_paragraph(md)
    if !(md isa Markdown.MD)
        return nothing
    end
    for block in md.content
        if block isa Markdown.Header
            # A heading before any prose means the summary is gone.
            return nothing
        elseif block isa Markdown.MD
            # A function with several documented methods nests one `MD` per
            # method, so the summary lives one level down.
            para = first_paragraph(block)
            isnothing(para) || return para
        elseif block isa Markdown.Paragraph
            para = strip(sprint(Markdown.plain, block))
            isempty(para) || return para
        end
    end
    return nothing
end

# Abbreviations whose full stop does not end a sentence. Anything else followed
# by whitespace does.
const _ABBREV = r"(?:e\.g|i\.e|cf|et al|vs|approx|Fig|Eq|Ref|Sec|Dr|Mr|Ms|St)$"

"""
    first_sentence(text) -> String

Trim a summary paragraph to its first sentence, keeping the terminator.

Kept deliberately simple: the summaries are short declaratives, so the only real
hazard is an abbreviation's full stop, which `_ABBREV` guards. Inline code and
maths are left untouched because the sentence split never looks inside them --
the scan is over the rendered text, and a `.` inside `x.y` is preceded by a
word character with no following whitespace.
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

# ", " between all but the last pair, " and " before the last -- the dominant
# style in the hand-written section this page replaces.
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

The rendered body of one capability bullet: its description followed by the
`@ref` links a reader follows to reach it.

A label that already contains links is used as-is -- see `Cap`'s docstring for
why some entries have to spell their own.
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

# `depth` counts Section nesting; `indent` counts list nesting. They advance
# independently: a Section resets the list, a Group extends it.
function render(io::IO, node::Section, depth::Int, indent::Int)
    println(io, "\n", "#"^depth, " ", node.title, "\n")
    for child in node.children
        render(io, child, depth + 1, 0)
    end
    return io
end
function render(io::IO, node::Prose, depth::Int, indent::Int)
    # Blank on both sides: markdown needs a paragraph separated from an adjacent
    # list in either direction, or the two run together into one block.
    # `render_catalogue` collapses whatever doubling this causes.
    #
    # Inside a list the paragraph must also be indented to the enclosing item's
    # content column, or it ends the list: the bullets that follow are then a
    # fresh block indented four spaces, which markdown reads as a *code block*.
    # Documenter does not resolve `@ref` inside code, so every link below such a
    # paragraph silently survives as literal `(@ref)` text and reaches the site
    # builder as a dead `./@ref` link.
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
function render(io::IO, node::Group, depth::Int, indent::Int)
    println(io, " "^indent, "- ::: details ", head_text(node.head))
    for child in node.children
        render(io, child, depth, indent + 2)
    end
    return io
end

"""
    render_catalogue(io; base_level = 2)

Render `CATALOGUE` as markdown. `base_level` is the heading level of a top-level
`Section`; it exists so the output can be compared against the section this page
was extracted from, which sat one level deeper inside `00_API.md`.
"""
function render_catalogue(io::IO = IOBuffer(); base_level::Int = 2)
    buf = IOBuffer()
    for node in CATALOGUE
        render(buf, node, base_level, 0)
    end
    # Blocks emit their own separators without knowing what precedes them, so
    # collapse the resulting runs to a single blank line.
    print(io, replace(String(take!(buf)), r"\n{3,}" => "\n\n"))
    return io
end

const _PREAMBLE = """
# [$(_PAGE_TITLE)](@id capability-catalogue)

Everything `PortfolioOptimisers.jl` can do, grouped by the job it does rather
than by the file it lives in. Each entry links to its docstring.

This page is generated (see
[docs/generate_capability_catalogue.jl](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/docs/generate_capability_catalogue.jl)):
the grouping is curated in `docs/capability_catalogue.jl`, and every description
is the first sentence of the corresponding docstring, so the two can never
disagree. A test asserts that every estimator and algorithm in the package
appears here, so the page cannot fall behind the code.

For the same types arranged by subtyping rather than by capability, see the
[type hierarchy](@ref type-hierarchy-AbstractEstimator).
"""

"""
    assert_complete()

Refuse to render a catalogue that is missing an estimator or algorithm.

`test/test_26_docs.jl` is the authoritative check and fires far sooner, on the
PR that added the type. This one exists because the generator's failure mode is
worse than a red test: a page that quietly omits a capability *looks* complete,
which is precisely the defect this page was built to end.
"""
function assert_complete()
    function leaf_types(T, acc = Set{Type}())
        subs = subtypes(T)
        if isempty(subs)
            !isabstracttype(T) && push!(acc, T)
        else
            foreach(S -> leaf_types(S, acc), subs)
        end
        return acc
    end
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

    required = Set(nameof.(collect(union(leaf_types(PortfolioOptimisers.AbstractEstimator),
                                         leaf_types(PortfolioOptimisers.AbstractAlgorithm)))))
    missed = sort(collect(setdiff(required, catalogued)))
    if !isempty(missed)
        error("""capability_catalogue: $(length(missed)) estimator(s)/algorithm(s) are not
                 catalogued, so the page would be rendered incomplete. Add each to
                 `docs/capability_catalogue.jl`:\n  $(join(missed, "\n  "))""")
    end
    return nothing
end

"""
    assert_refs_survive(md)

Check that every `@ref` written into the page is still a *link* once the page is
parsed as markdown.

Writing `[`X`](@ref)` is not enough: the surrounding text decides whether it
survives. Two ways it does not, both of which shipped before this check existed:

  - **A bare `_` in a description** pairs with the `_` inside a neighbouring
    `snake_case` link and the two are read as emphasis, eating the
    brackets. `(f_μ vector)` next to ``[`plot_factor_mu`](@ref)`` rendered as
    `(fμ vector … [`plotfactor_`.
  - **A paragraph at column 0 inside a list** ends the list, so the four-space
    indented bullets that follow parse as an indented *code block*.

Documenter resolves `@ref` only in real links, so in both cases the text passes
through verbatim and reaches the site builder as a dead `./@ref` link -- one
error, no matter how many links were lost, and no indication of which page
region is at fault. Catching it here names them.

Uses the `Markdown` stdlib rather than the CommonMark parser Documenter itself
uses. The two differ in corner cases, so this is a net for the common failures,
not a substitute for the docs build.
"""
function assert_refs_survive(md::AbstractString)
    written = [m.captures[1] for m in eachmatch(r"\[`([^`]+)`\]\(@ref\)", md)]
    survived = String[]
    # Walk structurally rather than by node type: `Markdown` spells its children
    # `content`, `items` or `text` depending on the node, and enumerating those
    # types would silently skip any this version of the stdlib adds.
    function walk(node)
        if node isa AbstractVector
            foreach(walk, node)
            return nothing
        end
        if node isa Markdown.Link && node.url == "@ref"
            # The link text of a catalogue entry is a single code span, so take
            # its literal contents; rendering the node would give back the whole
            # `[`X`](@ref)` and compare against nothing.
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
        error("""capability_catalogue: $(length(lost)) `@ref` link(s) do not survive markdown
                 parsing and would reach the site builder as dead `./@ref` links:
                 \n  $(join(unique(lost), "\n  "))\n
                 Usual causes: a bare `_` in a description pairing with the one inside a
                 neighbouring `snake_case` link (wrap it in backticks), or a `Prose` node
                 that breaks out of a list and turns the bullets below it into a code block.""")
    end
    return nothing
end

function generate_capability_catalogue(path::String = joinpath(@__DIR__, "src",
                                                               "capability_catalogue.md"))
    assert_complete()
    body = string(_PREAMBLE, "\n", String(take!(render_catalogue(IOBuffer()))))
    assert_refs_survive(body)
    write(path, body)
    return path
end
