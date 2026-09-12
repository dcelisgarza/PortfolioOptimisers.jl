# Prototype for issue #557: how an API page shows the boundary between the public API and the
# internals. The script builds six candidate layouts of two real API pages, each one in its own
# temporary Documenter site, and reports what each one renders.
#
# Run it from a REPL that has the `docs` environment active:
#
#     julia --project=docs
#     julia> include("research/prototypes/api_boundary_557.jl")
#     julia> main()
#
# `main` writes each candidate to `OUT/<tag>/build/*.html` and returns the measurement table.
# Nothing under `docs/` is touched. The full docs build stays the maintainer's to run.
#
# The candidates:
#
#   c0   the page as it is today, for comparison
#   c1   two blocks and two headings, `## Public API` and `## Internals`
#   c1b  the same two headings, but `@autodocs` derives the split from the source declarations
#   c2   an admonition before every internal entry, the order of the page unchanged
#   c2b  the same marker, stamped into the built HTML after the build, with no page edit
#   c3   the internals behind a disclosure element
#   c4   a sibling page, one for the public API and one for the internals

using Logging, Pkg, PortfolioOptimisers, Documenter

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const APISRC = joinpath(REPO, "docs", "src", "api")
const OUT = joinpath(tempdir(), "api_boundary_557")
const PAGES = ["15_Turnover.md", "20_Optimisation/01_Base_Optimisation.md"]
const SRCFILES = Dict("15_Turnover.md" => "15_Turnover.jl",
                      "20_Optimisation/01_Base_Optimisation.md" => "01_Base_Optimisation.jl")

# ------------------------------------------------------------------ read one API page
"""
    docs_blocks(lines)

Return the line range of the body of every fence that opens with @docs in `lines`.
"""
function docs_blocks(lines)
    blocks = Vector{UnitRange{Int}}()
    i = 1
    while i <= length(lines)
        if startswith(strip(lines[i]), "```@docs")
            j = i + 1
            while j <= length(lines) && !startswith(strip(lines[j]), "```")
                j += 1
            end
            push!(blocks, (i + 1):(j - 1))
            i = j + 1
        else
            i += 1
        end
    end
    return blocks
end

"""
    block_entries(body)

Split the body of a `@docs` block into entries. A signature that runs over more than one line is
one entry, so the split parses the body rather than reading a line as an entry.
"""
function block_entries(body::Vector{String})
    ex = Meta.parseall(join(body, "\n"))
    starts = Int[]
    for (k, a) in enumerate(ex.args)
        if a isa LineNumberNode &&
           k < length(ex.args) &&
           !(ex.args[k + 1] isa LineNumberNode)
            push!(starts, a.line)
        end
    end
    entries = String[]
    for (k, s) in enumerate(starts)
        stop = k == length(starts) ? length(body) : starts[k + 1] - 1
        push!(entries, join(body[s:stop], "\n"))
    end
    return entries
end

"""
    headname(ex)

Return the head name of a `@docs` entry, and whether the entry names it in the qualified form.
"""
function headname(ex)
    if ex isa QuoteNode
        return headname(ex.value)
    end
    if ex isa Symbol
        return (ex, false)
    end
    if ex isa Expr
        if ex.head === :quote
            return headname(ex.args[1])
        end
        if ex.head === :call
            return headname(ex.args[1])
        end
        if ex.head === :macrocall
            return headname(ex.args[1])
        end
        if ex.head === :curly
            return headname(ex.args[1])
        end
        if ex.head === :where
            return headname(ex.args[1])
        end
        ex.head === :(.) && return (Symbol(string(ex)), true)
    end
    return (Symbol(string(ex)), true)
end

"""
    classify(entry)

Classify one `@docs` entry as `:exported`, `:public`, `:internal` or `:foreign`. The source
declaration is the authority, so the answer comes from `Base.isexported` and `Base.ispublic`.
"""
function classify(entry::String)
    (name, qualified) = headname(Meta.parse(entry))
    if qualified
        return :foreign
    end
    if Base.isexported(PortfolioOptimisers, name)
        return :exported
    end
    if Base.ispublic(PortfolioOptimisers, name)
        return :public
    end
    return :internal
end

struct PageModel
    path::String
    title::String
    prose::String
    entries::Vector{String}
    classes::Vector{Symbol}
end

function read_page(rel)
    lines = readlines(joinpath(APISRC, rel))
    blocks = docs_blocks(lines)
    entries = String[]
    for b in blocks
        append!(entries, block_entries(lines[b]))
    end
    prose = ""
    for l in lines[2:(first(first(blocks)) - 2)]
        if isempty(strip(l))
            continue
        end
        prose = l
        break
    end
    return PageModel(rel, strip(replace(lines[1], "#" => "")), prose, entries,
                     map(classify, entries))
end

ispub(c) = c === :exported || c === :public

# ------------------------------------------------------------------- the candidate pages
const META = "```@meta\nCurrentModule = PortfolioOptimisers\n```\n"
const INDEX = "# Prototypes for issue 557\n\nThe candidate pages are in the navigation.\n"
const INTERNAL_NOTE = """
!!! warning "Internal"
    The names in this section are internal. They are not exported and they are not declared
    `public`. Their behaviour can change in a patch release.
"""

docsblock(entries) = "```@docs\n" * join(entries, "\n") * "\n```\n"
header(m::PageModel) = "# $(m.title)\n\n$(m.prose)\n\n"

c0(m::PageModel) = META * header(m) * docsblock(m.entries)

function c1(m::PageModel)
    pub = m.entries[findall(ispub, m.classes)]
    int = m.entries[findall(==(:internal), m.classes)]
    frn = m.entries[findall(==(:foreign), m.classes)]
    s = META * header(m)
    if !(isempty(pub))
        (s *= "## Public API\n\n" * docsblock(pub) * "\n")
    end
    if !(isempty(frn))
        (s *= "## Methods added to other modules\n\n" * docsblock(frn) * "\n")
    end
    if !(isempty(int))
        (s *= "## Internals\n\n" * INTERNAL_NOTE * "\n" * docsblock(int))
    end
    return s
end

function c1b(m::PageModel)
    src = SRCFILES[m.path]
    return META * header(m) * """
    ## Public API

    ```@autodocs
    Modules = [PortfolioOptimisers]
    Pages = ["$(src)"]
    Public = true
    Private = false
    ```

    ## Internals

    $(INTERNAL_NOTE)

    ```@autodocs
    Modules = [PortfolioOptimisers]
    Pages = ["$(src)"]
    Public = false
    Private = true
    ```
    """
end

function c2(m::PageModel)
    s = META * header(m)
    for (e, c) in zip(m.entries, m.classes)
        if c === :internal
            s *= "!!! warning \"Internal\"\n    Not exported, not `public`.\n\n"
        elseif c === :public
            s *= "!!! info \"Public\"\n    Declared `public`, not exported.\n\n"
        elseif c === :foreign
            s *= "!!! info \"Method on another module's function\"\n" *
                 "    The name belongs to another module.\n\n"
        end
        s *= docsblock([e]) * "\n"
    end
    return s
end

# Two disclosure mechanisms are built side by side, because only a build says which one keeps the
# docstrings. Mechanism a is the Documenter `details` admonition. Mechanism b is raw HTML.
function c3(m::PageModel)
    pub = m.entries[findall(ispub, m.classes)]
    int = [m.entries[findall(==(:internal), m.classes)];
           m.entries[findall(==(:foreign), m.classes)]]
    s = META * header(m)
    if !(isempty(pub))
        (s *= "## Public API\n\n" * docsblock(pub) * "\n")
    end
    s *= "## Internals\n\n"
    half = max(1, div(length(int), 2))
    a, b = int[1:half], int[(half + 1):end]
    s *= "!!! details \"Internals, mechanism a: $(length(a)) names inside a details admonition\"\n\n"
    for l in split(docsblock(a), "\n")
        s *= isempty(l) ? "\n" : "    " * l * "\n"
    end
    s *= "\n```@raw html\n<details><summary>Internals, mechanism b: $(length(b)) names inside " *
         "raw HTML</summary>\n```\n\n" *
         docsblock(b) *
         "\n```@raw html\n</details>\n```\n"
    return s
end

function c4_public(m::PageModel, sibling::String)
    pub = m.entries[findall(ispub, m.classes)]
    n = count(!ispub, m.classes)
    return META *
           header(m) *
           docsblock(pub) *
           "\n" *
           "The $(n) internal names of this file are documented in [Internals]($(sibling)).\n"
end

function c4_internal(m::PageModel, sibling::String)
    int = m.entries[findall(!ispub, m.classes)]
    return META *
           "# $(m.title): internals\n\n" *
           INTERNAL_NOTE *
           "\n" *
           "The public API of this file is documented in [$(m.title)]($(sibling)).\n\n" *
           docsblock(int)
end

# ------------------------------------------------------- candidate 2b, stamped after the build
"""
    anchor_symbol(id)

Take the head symbol out of a docstring anchor, for example
`PortfolioOptimisers.needs_previous_weights-Tuple{…}`. Documenter writes the anchor into every
`<summary id="…">`, so the built page needs no marker of its own.
"""
function anchor_symbol(id::AbstractString)
    s = id
    i = findfirst('-', s)
    if !(i === nothing)
        (s = s[1:(i - 1)])
    end
    parts = split(s, '.')
    if length(parts) == 1
        return (Symbol(parts[1]), false)
    end
    if parts[1] == "PortfolioOptimisers"
        return (Symbol(parts[end]), false)
    end
    return (Symbol(parts[end]), true)
end

function badge_class(id)
    (name, foreign) = anchor_symbol(id)
    if foreign
        return ("foreign", "other module")
    end
    if Base.isexported(PortfolioOptimisers, name)
        return ("exported", "exported")
    end
    if Base.ispublic(PortfolioOptimisers, name)
        return ("public", "public")
    end
    return ("internal", "internal")
end

const BADGE_CSS = """
<style>
.api-badge { border-radius: 4px; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.04em;
             margin-left: 0.5rem; padding: 0.1rem 0.4rem; text-transform: uppercase; }
.api-exported { background: #e6f4ea; color: #137333; }
.api-public   { background: #e8f0fe; color: #1967d2; }
.api-internal { background: #fce8e6; color: #a50e0e; }
.api-foreign  { background: #f1f3f4; color: #3c4043; }
</style>
"""

"""
Stamp a badge into every docstring header of one built page. Return the count per class.
"""
function stamp!(path)
    counts = Dict{String, Int}()
    h = read(path, String)
    out = replace(h,
                  r"<summary id=\"([^\"]+)\">(.*?)</summary>"s => function (m)
                      mm = match(r"<summary id=\"([^\"]+)\">(.*?)</summary>"s, m)
                      id = mm.captures[1]
                      (cls, label) = badge_class(id)
                      counts[cls] = get(counts, cls, 0) + 1
                      return "<summary id=\"$(id)\">" *
                             mm.captures[2] *
                             "<span class=\"api-badge api-$(cls)\">$(label)</span></summary>"
                  end)
    write(path, replace(out, "</head>" => BADGE_CSS * "</head>"; count = 1))
    return counts
end

function stamp_build(root)
    res = Dict{String, Any}()
    for (dir, _, files) in walkdir(joinpath(root, "build"))
        for f in files
            if !(endswith(f, ".html"))
                continue
            end
            res[f] = stamp!(joinpath(dir, f))
        end
    end
    return res
end

# ------------------------------------------------------------------------------ the builder
function build(tag, files, pages)
    root = joinpath(OUT, tag)
    rm(root; force = true, recursive = true)
    mkpath(joinpath(root, "src"))
    for (name, text) in files
        mkpath(dirname(joinpath(root, "src", name)))
        write(joinpath(root, "src", name), text)
    end
    open(joinpath(OUT, "$(tag).log"), "w") do io
        Logging.with_logger(Logging.SimpleLogger(io, Logging.Warn)) do
            return makedocs(; root = root, sitename = "candidate $(tag)",
                            modules = [PortfolioOptimisers], warnonly = true,
                            remotes = nothing, checkdocs = :none, doctest = false,
                            format = Documenter.HTML(; prettyurls = false,
                                                     edit_link = nothing,
                                                     repolink = nothing,
                                                     size_threshold = nothing,
                                                     size_threshold_warn = nothing),
                            pages = pages)
        end
    end
    return root
end

# ---------------------------------------------------------------------------- the measurement
countpat(s, p) = length(collect(eachmatch(p, s)))

function measure(root, page)
    f = joinpath(root, "build", page)
    if !(isfile(f))
        return (page = page, exists = false)
    end
    h = read(f, String)
    return (page = page, exists = true, bytes = length(h),
            docstrings = countpat(h, r"<article class=\"docstring\"|<article>"),
            admonitions = countpat(h, r"class=\"admonition"),
            details = countpat(h, r"<details"),
            source_lines = length(readlines(joinpath(root, "src",
                                                     replace(page, ".html" => ".md")))))
end

function anchor_ids(root, page)
    return [m.captures[1]
            for m in eachmatch(r"<summary id=\"([^\"]+)\"",
                               read(joinpath(root, "build", page), String))]
end

function main()
    models = Dict(rel => read_page(rel) for rel in PAGES)
    tn = models["15_Turnover.md"]
    bo = models["20_Optimisation/01_Base_Optimisation.md"]
    nav = ["Home" => "index.md", "Turnover" => "turnover.md",
           "Base optimisation" => "base_optimisation.md"]
    roots = Dict{String, String}()
    for (tag, f) in (("c0", c0), ("c1", c1), ("c1b", c1b), ("c2", c2), ("c3", c3))
        roots[tag] = build(tag,
                           ["index.md" => INDEX, "turnover.md" => f(tn),
                            "base_optimisation.md" => f(bo)], nav)
    end
    roots["c4"] = build("c4",
                        ["index.md" => INDEX,
                         "turnover.md" => c4_public(tn, "turnover_internals.html"),
                         "turnover_internals.md" => c4_internal(tn, "turnover.html"),
                         "base_optimisation.md" =>
                             c4_public(bo, "base_optimisation_internals.html"),
                         "base_optimisation_internals.md" =>
                             c4_internal(bo, "base_optimisation.html")],
                        ["Home" => "index.md",
                         "Turnover" => ["Public API" => "turnover.md",
                                        "Internals" => "turnover_internals.md"],
                         "Base optimisation" => ["Public API" => "base_optimisation.md",
                                                 "Internals" => "base_optimisation_internals.md"]])
    roots["c2b"] = joinpath(OUT, "c2b")
    rm(roots["c2b"]; force = true, recursive = true)
    cp(roots["c0"], roots["c2b"])
    stamps = stamp_build(roots["c2b"])
    pages = ["turnover.html", "base_optimisation.html", "turnover_internals.html",
             "base_optimisation_internals.html"]
    table = Dict(tag =>
                     [measure(root, p) for p in pages if isfile(joinpath(root, "build", p))]
                 for (tag, root) in roots)
    return (roots = roots, table = table, stamps = stamps,
            split = Dict(rel => (exported = count(==(:exported), m.classes),
                                 public = count(==(:public), m.classes),
                                 internal = count(==(:internal), m.classes),
                                 foreign = count(==(:foreign), m.classes))
                         for (rel, m) in models))
end
