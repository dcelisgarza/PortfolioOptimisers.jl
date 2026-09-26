#=
The title and the description every docs page carries (issue #561, ADR 0128 § Amendment).

Until #561 every page of the site carried Documenter's one default description,
`Documentation for PortfolioOptimisers.jl.`, and the landing page was titled `Home`. A search
engine ranks a page on its `<title>` and its `<meta name="description">`, so one description
for the whole site meant no page could be found by what it says.

Two classes of page, two rules, one file that states them (`docs/page_metadata.jl`):

  * A HAND-WRITTEN page states a `Description = "…"` in a `@meta` block of its source: the
    `.md` of a hand-written page, the Literate `.jl` of a generated guide or example, and the
    generator of the capability catalogue and the type hierarchy. The line must exist, must
    not be the default, must be unique across the site, and must fit the 50–160 characters a
    search engine shows.

  * A MIRROR page under `docs/src/public_api/` or `docs/src/private_api/` (ADR 0128) carries
    a description DERIVED from its own H1 and `@docs` names, and the private mirror's H1 ends
    in `: private API`. The mirror testsets below pass vacuously until the migration creates
    those trees; creating the trees turns them on.

The census reads sources, not built pages, for the reason `test_50_docs_sitemap.jl` gives:
a docs build is slow and run by hand. Everything a description is checked against is in the
source itself, so no build and no live module is needed.
=#
using Test

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "docs", "page_metadata.jl"))

#=
The source of every page that states its own description, keyed by the path the page
renders at under `docs/src`. The generated guide and example pages are read at their Literate
`.jl` sources, and the two generated root-level pages at their generators, because the
generated `.md` is untracked and never edited (ADR 0151).

The legacy `docs/src/api/` pages are not in this set: their descriptions are the derived
lines the migration writes into the mirror pages, and the mirror testsets take over from
them. The introduction moved out of that tree to `docs/src/00_API.md` (ADR 0128), so the
top-level walk below finds it like any other hand-written page.
=#
function described_page_sources(repo::AbstractString)
    docs_src = joinpath(repo, "docs", "src")
    sources = Dict{String, String}()
    for file in readdir(docs_src)
        path = joinpath(docs_src, file)
        if endswith(file, ".md") &&
           file != "capability_catalogue.md" &&
           file != "TypeHierarchy.md"
            sources[file] = path
        end
    end
    for file in readdir(joinpath(docs_src, "contribute"))
        if !(endswith(file, ".md"))
            continue
        end
        sources[joinpath("contribute", file)] = joinpath(docs_src, "contribute", file)
    end
    sources["capability_catalogue.md"] = joinpath(repo, "docs",
                                                  "generate_capability_catalogue.jl")
    sources["TypeHierarchy.md"] = joinpath(repo, "docs", "generate_type_hierarchy.jl")
    for section in ("user_guide", "examples")
        for (dir, _, files) in walkdir(joinpath(repo, section))
            for file in files
                if !(endswith(file, ".jl"))
                    continue
                end
                rel = relpath(joinpath(dir, file), repo)
                sources[replace(rel, r"\.jl$" => ".md")] = joinpath(dir, file)
            end
        end
    end
    return sources
end

# The pages of a mirror tree, keyed the same way; empty until the migration creates the tree.
function mirror_pages(repo::AbstractString, side::Symbol)
    tree = joinpath(repo, "docs", "src", MIRROR_TREES[side])
    pages = Dict{String, String}()
    if !(isdir(tree))
        return pages
    end
    for (dir, _, files) in walkdir(tree)
        for file in files
            if !(endswith(file, ".md"))
                continue
            end
            pages[relpath(joinpath(dir, file), joinpath(repo, "docs", "src"))] = joinpath(dir,
                                                                                          file)
        end
    end
    return pages
end

#=
Every link from a page to another page by a relative `.md` path, as `(target)` in
`[text](target)`. A path breaks when either page moves, and the file split 0ed224263c broke
one so (#1353). The target's H1 carries a label, `# [Title](@id label)`, and the link reads
`[text](@ref label)`, which survives a move. A Literate source puts the label on the H1 of its
first block comment. A colon excludes a URL.
=#
function relative_md_links(text::AbstractString)
    return [m.captures[1] for m in eachmatch(r"\]\(([^():\s]*\.md(?:#[^)]*)?)\)", text)]
end

@testset "Docs page metadata census" begin
    @testset "The derivation" begin
        # A page title is the H1's text, with an `@id` wrapper unwrapped.
        @test page_h1("# Asset turnover\n\ntext") == "Asset turnover"
        @test page_h1("intro\n\n# [Migration guide](@id migration)\n") == "Migration guide"
        @test page_h1("# Asset turnover: private API\n") == "Asset turnover: private API"
        @test isnothing(page_h1("## Only a subheading\n"))

        # The description is read off a `@meta` block, in a `.md` and in a Literate `.jl`.
        md = "```@meta\nCurrentModule = PortfolioOptimisers\nDescription = \"A line.\"\n```\n"
        @test meta_description(md) == "A line."
        jl = "#=\n```@meta\nDescription = \"A line.\"\n```\n\n# Title\n=#\n\nusing Test\n"
        @test meta_description(jl) == "A line."
        @test isnothing(meta_description("```@meta\nCurrentModule = X\n```\n# Title\n"))
        @test isnothing(meta_description("# Title\n"))

        # The opening paragraph under a named H1: blank lines skipped, cut at the next blank
        # line, line breaks kept so a rewrap on one side shows as a difference.
        page = "badge\n\n# Title\n\nFirst line.\nSecond line.\n\nNext paragraph.\n"
        @test opening_paragraph(page, "Title") == "First line.\nSecond line."
        @test isnothing(opening_paragraph(page, "Other"))
        @test isnothing(opening_paragraph("# Title\n\n\n", "Title"))

        # `@docs` names: signatures and parameters stripped, the package qualification
        # dropped, a foreign qualification kept, duplicates dropped, page order kept.
        page = """
               # Asset turnover

               ```@docs
               Turnover
               PortfolioOptimisers.TnE_Tn
               needs_previous_weights(::Turnover)
               Turnover{T}
               Base.iterate(::Turnover)
               ```

               ```@docs
               needs_previous_weights
               ```
               """
        @test docs_block_names(page) ==
              ["Turnover", "TnE_Tn", "needs_previous_weights", "Base.iterate"]

        # The derived line, on each side, and on an empty mirror.
        @test derived_description("Asset turnover", :public, ["Turnover", "TnE_Tn"]) ==
              "Asset turnover, public API of PortfolioOptimisers.jl: Turnover, TnE_Tn."
        @test derived_description("Asset turnover", :private, ["needs_previous_weights"]) ==
              "Asset turnover, private API of PortfolioOptimisers.jl: needs_previous_weights."
        @test derived_description("Asset turnover", :public, String[]) ==
              "Asset turnover has no public API in PortfolioOptimisers.jl; its names are in the private API."
        @test derived_description("Asset turnover", :private, String[]) ==
              "Asset turnover has no private API in PortfolioOptimisers.jl; its names are in the public API."
        @test_throws ArgumentError derived_description("X", :internal, ["a"])

        # The cut: on a name boundary once the line passes the cut length, marked `, …`,
        # and the whole line fits the band a hand-written description must fit.
        many = ["Name$(i)" for i in 1:40]
        long = derived_description("Subject", :public, many)
        @test endswith(long, ", …")
        @test length(long) <= DESCRIPTION_MAX_LENGTH
        @test !occursin("Name40", long)
        kept = split(long[(length("Subject, public API of PortfolioOptimisers.jl: ") + 1):end],
                     ", ")
        # Every kept name is whole, and the next name would not have fit.
        @test all(n -> n in many, kept[1:(end - 1)])
        @test length(long) - length(", …") + length(", $(many[length(kept)])") >
              DERIVED_DESCRIPTION_CUT
        # A subject's case is untouched: `OWA` stays `OWA`.
        @test startswith(derived_description("OWA weights", :public, ["OWA"]),
                         "OWA weights,")

        # A mirror page re-derives from its own text; the private suffix is stripped.
        private = "# Asset turnover: private API\n\n```@docs\nTnE_Tn\n```\n"
        @test mirror_description(private, :private) ==
              "Asset turnover, private API of PortfolioOptimisers.jl: TnE_Tn."
        public = "# Asset turnover\n\n```@docs\nTurnover\n```\n"
        @test mirror_description(public, :public) ==
              "Asset turnover, public API of PortfolioOptimisers.jl: Turnover."
        @test_throws ArgumentError mirror_description("no heading\n", :public)
        @test mirror_side(joinpath("docs", "src", "public_api", "12_Turnover.md")) ==
              :public
        @test mirror_side(joinpath("docs", "src", "private_api", "x", "y.md")) == :private
        @test isnothing(mirror_side(joinpath("docs", "src", "api", "12_Turnover.md")))
    end

    sources = described_page_sources(REPO_ROOT)
    descriptions = Dict(page => meta_description(read(path, String))
                        for (page, path) in sources)
    landing = descriptions["index.md"]

    @testset "the set of described pages is the site" begin
        # A guard on the census itself: the walk found the pages it is meant to find.
        @test haskey(sources, "index.md")
        @test haskey(sources, "migration.md")
        @test haskey(sources, "99_references.md")
        @test haskey(sources, "capability_catalogue.md")
        @test haskey(sources, "00_API.md")
        @test haskey(sources, "TypeHierarchy.md")
        @test count(p -> startswith(p, "contribute"), keys(sources)) >= 3
        @test count(p -> startswith(p, "user_guide"), keys(sources)) >= 10
        @test count(p -> startswith(p, "examples"), keys(sources)) >= 60
    end

    @testset "every page states a description" begin
        missing_desc = sort([p for (p, d) in descriptions if isnothing(d)])
        if !isempty(missing_desc)
            @warn "Pages whose source states no `Description` in a `@meta` block: $(join(missing_desc, ", "))"
        end
        @test missing_desc == String[]
    end

    stated = Dict(p => d for (p, d) in descriptions if !isnothing(d))

    @testset "no page carries the default" begin
        defaulted = sort([p for (p, d) in stated if d == DOCUMENTER_DEFAULT_DESCRIPTION])
        @test defaulted == String[]
    end

    @testset "a description fits the band a search engine shows" begin
        short = sort([p => length(d)
                      for (p, d) in stated if length(d) < DESCRIPTION_MIN_LENGTH])
        long = sort([p => length(d)
                     for (p, d) in stated if length(d) > DESCRIPTION_MAX_LENGTH])
        @test short == Pair{String, Int}[]
        @test long == Pair{String, Int}[]
    end

    @testset "a description is unique across the site" begin
        # The generated search page always takes the site-wide fallback, which is the landing
        # line, and it is the one permitted duplicate. It is not a source here, so the check
        # is over every stated line.
        by_line = Dict{String, Vector{String}}()
        for (p, d) in stated
            push!(get!(by_line, d, String[]), p)
        end
        duplicated = sort([sort(ps) for (d, ps) in by_line if length(ps) > 1])
        @test duplicated == Vector{String}[]
    end

    @testset "spelling: the landing line carries the American form once, no other line" begin
        # #561 § 7 and #562 § 7: British in every title and description; the landing line
        # may carry `optimization` once, as the parenthetical the README opening shares.
        american = r"optimiz"i
        @test count(american, landing) == 1
        elsewhere = sort([p
                          for (p, d) in stated if p != "index.md" && occursin(american, d)])
        @test elsewhere == String[]
    end

    @testset "the README opening and the landing opening are one text" begin
        # #562 § 5: the two-sentence opening under the README's H1 and under the landing
        # page's H1 are one text maintained in two places, identical by rule, character for
        # character. Whichever file is edited, the other must follow.
        readme = opening_paragraph(read(joinpath(REPO_ROOT, "README.md"), String),
                                   "PortfolioOptimisers.jl")
        index = opening_paragraph(read(sources["index.md"], String),
                                  "PortfolioOptimisers.jl")
        @test !isnothing(readme)
        @test !isnothing(index)
        @test readme == index
        # The opening is the two sentences #562 fixed: the library, Julia, composable
        # immutable estimators, and the one parenthetical American spelling.
        @test occursin("library for Julia", readme)
        @test occursin("(portfolio optimization)", readme)
        @test count(r"optimiz"i, readme) == 1
        @test occursin("immutable estimator", readme)
    end

    @testset "make.jl: the home title and the site-wide fallback" begin
        make = read(joinpath(REPO_ROOT, "docs", "make.jl"), String)
        # The landing page's `<title>` is its `pages` label (#555), so the label carries the
        # words the site ranks on.
        @test occursin("\"Portfolio optimisation library in Julia\" => HOME_PAGE", make)
        @test !occursin("\"Home\" => HOME_PAGE", make)
        # The fallback is the landing line, read off the landing page so the two cannot
        # drift, and it is what `Documenter.HTML` receives.
        @test occursin("include(joinpath(@__DIR__, \"page_metadata.jl\"))", make)
        # The formatter may wrap the call, so the match spans whitespace.
        @test occursin(r"const SITE_DESCRIPTION = meta_description\(read\(joinpath\(@__DIR__,\s*\"src\",\s*HOME_PAGE\),\s*String\)\)",
                       make)
        @test occursin("description = SITE_DESCRIPTION", make)
        @test !isnothing(landing)
    end

    #=
    The mirror pages (ADR 0128 § Amendment). Both trees are absent until the migration
    creates them, and then every page of each tree is held to the derivation above and to
    the H1 rule. An absent tree holds no pages, so the testsets pass vacuously and need no
    edit when the trees appear.
    =#
    for side in (:public, :private)
        pages = mirror_pages(REPO_ROOT, side)
        @testset "$(MIRROR_TREES[side]): the description is the derived line" begin
            drifted = String[]
            for (page, path) in sort(collect(pages); by = first)
                text = read(path, String)
                meta_description(text) == mirror_description(text, side) ||
                    push!(drifted, page)
            end
            if !isempty(drifted)
                @warn "Mirror pages whose `Description` is not the line `docs/page_metadata.jl` derives: $(join(drifted, ", "))"
            end
            @test drifted == String[]
        end
        @testset "$(MIRROR_TREES[side]): the H1 carries the suffix exactly on the private side" begin
            wrong = String[]
            for (page, path) in sort(collect(pages); by = first)
                h1 = page_h1(read(path, String))
                suffixed = !isnothing(h1) && endswith(h1, PRIVATE_API_SUFFIX)
                suffixed == (side === :private) || push!(wrong, page)
            end
            @test wrong == String[]
        end
    end

    @testset "a page links a page by its label, never by a relative path" begin
        @test relative_md_links("[a](../b/C.md) [d](E.md#f) [g](./H.md)") ==
              ["../b/C.md", "E.md#f", "./H.md"]
        @test isempty(relative_md_links("[a](@ref label) [`B`](@ref)"))
        @test isempty(relative_md_links("[c](https://x.org/README.md)"))
        # Every described page, the Literate sources and the generators with them, and both
        # mirror trees.
        pages = copy(sources)
        for side in (:public, :private)
            merge!(pages, mirror_pages(REPO_ROOT, side))
        end
        @test haskey(pages, "index.md")
        @test count(p -> startswith(p, "examples"), keys(pages)) >= 60
        @test count(p -> startswith(p, "private_api"), keys(pages)) >= 100
        linked = String[]
        for (page, path) in sort(collect(pages); by = first)
            for target in relative_md_links(read(path, String))
                push!(linked, "$(relpath(path, REPO_ROOT)) -> $target")
            end
        end
        if !isempty(linked)
            @warn "Pages that link a page by a relative `.md` path. Give the target's H1 an `(@id label)` and link `(@ref label)`:\n  $(join(linked, "\n  "))"
        end
        @test linked == String[]
    end
end
