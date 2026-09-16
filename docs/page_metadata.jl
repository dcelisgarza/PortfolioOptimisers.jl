# The title and the description every docs page carries (issue #561, ADR 0128 amendment).
#
# A search engine ranks a page by its `<title>` and its `<meta name="description">`, and until
# #561 every page of the site carried Documenter's one default description. Two classes of page
# now carry their own:
#
#   * A HAND-WRITTEN page — the landing page, the capability catalogue, the API introduction,
#     the migration guide, the references page, the contribute pages, and every Literate
#     source under `user_guide/` and `examples/` — carries a `Description = "…"` in a
#     `@meta` block of its source. The census in `test/test_64_docs_page_metadata_census.jl`
#     checks that the line exists, is not the default, is unique across the site, and fits
#     the length a search engine shows.
#
#   * A MIRROR page — one of the pages under `docs/src/public_api/` and
#     `docs/src/private_api/` that ADR 0128's migration generates, one pair per source file —
#     carries a description DERIVED from the page itself: its H1 and the names of its own
#     `@docs` blocks. The derivation is written once, here, so the generator that writes the
#     line and the census that re-derives it cannot drift. Everything the derivation reads is
#     in the `.md`, so the census needs no docs build and no live module.
#
# This file is plain data and string functions with no dependencies, because it is `include`d
# from two environments: the docs build (the mirror generator, once the migration lands) and
# the test suite (the census). It must stay loadable with nothing but `Base`.

# The site name the derived line names. It is the `sitename` `docs/make.jl` passes to
# `makedocs`, restated here so the derivation does not depend on the docs environment.
const SITE_NAME = "PortfolioOptimisers.jl"

# The description Documenter writes when a page states none. A page that carries this line
# has slipped the gate, so the census refuses it.
const DOCUMENTER_DEFAULT_DESCRIPTION = "Documentation for $(SITE_NAME)."

# The band a hand-written description must fit. A search engine shows about 155 characters
# of a description and truncates the rest; under 50 the line is a fragment, not a summary.
const DESCRIPTION_MIN_LENGTH = 50
const DESCRIPTION_MAX_LENGTH = 160

# A derived line is cut on a name boundary once it passes this length, and `, …` marks the
# cut. The whole line then fits `DESCRIPTION_MAX_LENGTH`.
const DERIVED_DESCRIPTION_CUT = 155

# The suffix the private mirror's H1 carries (#561 § 3): `# Asset turnover: private API`.
# The public mirror keeps the source page's H1 verbatim.
const PRIVATE_API_SUFFIX = ": private API"

# The directory of each mirror tree under `docs/src`, keyed by the side it holds.
const MIRROR_TREES = (public = "public_api", private = "private_api")

"""
    page_h1(text) -> Union{Nothing, String}

The text of the first level-one heading of a markdown page, with a `[Title](@id anchor)`
wrapper unwrapped. `nothing` when the page has none.
"""
function page_h1(text::AbstractString)
    m = match(r"^# (.+?)\s*$"m, text)
    if isnothing(m)
        return nothing
    end
    title = m.captures[1]
    anchored = match(r"^\[(.+)\]\(@id [^)]+\)$", title)
    return String(isnothing(anchored) ? title : anchored.captures[1])
end

"""
    opening_paragraph(text, h1) -> Union{Nothing, String}

The first paragraph under the level-one heading whose text is `h1`: the lines from the first
non-blank line after the heading to the next blank line, joined with newlines. `nothing` when
the page has no such heading, or nothing but blank lines follows it. The README opening and
the landing-page opening are one text kept in two places (#562 § 5), and this is how the
census reads both.
"""
function opening_paragraph(text::AbstractString, h1::AbstractString)
    lines = split(text, '\n')
    start = findfirst(==("# $h1"), lines)
    if isnothing(start)
        return nothing
    end
    i = start + 1
    while i <= length(lines) && isempty(strip(lines[i]))
        i += 1
    end
    if i > length(lines)
        return nothing
    end
    j = i
    while j <= length(lines) && !isempty(strip(lines[j]))
        j += 1
    end
    return join(lines[i:(j - 1)], '\n')
end

"""
    meta_description(text) -> Union{Nothing, String}

The `Description = "…"` line of the page's `@meta` blocks, or `nothing` when no block
states one. A Literate source holds the block inside a `#= … =#` comment, and a generator
holds it inside an indented string literal that Julia dedents, so the fence is found
verbatim in a `.jl` source as well as in a `.md` page, with any leading indentation.
"""
function meta_description(text::AbstractString)
    for block in eachmatch(r"^[ \t]*```@meta[ \t]*\n(.*?)^[ \t]*```"ms, text)
        m = match(r"^\s*Description\s*=\s*\"(.*)\"\s*$"m, block.captures[1])
        isnothing(m) || return String(m.captures[1])
    end
    return nothing
end

"""
    docs_block_names(text) -> Vector{String}

The binding names of the page's own `@docs` blocks, in page order, each written once: the
signature is stripped (`f(x, y)` → `f`, `T{S}` → `T`), and a `PortfolioOptimisers.`
qualification is dropped because the derived line already names the site. A qualification
by another module (`Base.iterate`, `StatsAPI.fit`) is kept, because the module is part of
how a reader knows the name.
"""
function docs_block_names(text::AbstractString)
    names = String[]
    for block in eachmatch(r"^```@docs\s*\n(.*?)^```"ms, text)
        for line in eachline(IOBuffer(block.captures[1]))
            entry = strip(line)
            if isempty(entry)
                continue
            end
            name = String(first(split(entry, r"[({]"; limit = 2)))
            name = replace(name, r"^PortfolioOptimisers\." => "")
            name in names || push!(names, name)
        end
    end
    return names
end

"""
    derived_description(subject, side, names) -> String

The description of a mirror page (#561 § 4), from its subject — the H1 with the private
suffix stripped, case untouched — the side it sits on (`:public` or `:private`), and the
names of its own `@docs` blocks. The names are joined in page order and cut on a name
boundary once the line passes `DERIVED_DESCRIPTION_CUT` characters, with `, …` marking the
cut. An empty mirror — a source file with nothing on that side — reads as a sentence that
points at the other side.
"""
function derived_description(subject::AbstractString, side::Symbol,
                             names::AbstractVector{<:AbstractString})
    if !(side in (:public, :private))
        throw(ArgumentError("side must be :public or :private, got $(repr(side))"))
    end
    other = side === :public ? :private : :public
    if isempty(names)
        return "$subject has no $side API in $SITE_NAME; its names are in the $other API."
    end
    line = "$subject, $side API of $SITE_NAME: $(names[1])"
    cut = false
    for name in names[2:end]
        candidate = "$line, $name"
        if length(candidate) > DERIVED_DESCRIPTION_CUT
            cut = true
            break
        end
        line = candidate
    end
    return cut ? "$line, …" : "$line."
end

"""
    mirror_side(path) -> Union{Nothing, Symbol}

Which mirror tree a page path sits in, by the tree directory it contains: `:public`,
`:private`, or `nothing` for a page outside both trees.
"""
function mirror_side(path::AbstractString)
    parts = splitpath(path)
    if MIRROR_TREES.public in parts
        return :public
    end
    if MIRROR_TREES.private in parts
        return :private
    end
    return nothing
end

"""
    mirror_subject(h1, side) -> String

The subject of a mirror page: its H1 with `PRIVATE_API_SUFFIX` stripped on the private side.
"""
function mirror_subject(h1::AbstractString, side::Symbol)
    if side === :private && endswith(h1, PRIVATE_API_SUFFIX)
        return String(h1[1:(end - length(PRIVATE_API_SUFFIX))])
    end
    return String(h1)
end

"""
    mirror_description(text, side) -> String

The description a mirror page must carry, re-derived from the page's own text: its H1 and
its `@docs` names. This is the one function the mirror generator and the census both call.
"""
function mirror_description(text::AbstractString, side::Symbol)
    h1 = page_h1(text)
    if isnothing(h1)
        throw(ArgumentError("a mirror page must open with a level-one heading"))
    end
    return derived_description(mirror_subject(h1, side), side, docs_block_names(text))
end
