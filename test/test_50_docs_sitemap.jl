#=
The docs sitemap (`docs/generate_sitemap.jl`, issue #568). Documenter writes no
`sitemap.xml` at any version, so `docs/make.jl` writes one between `makedocs` and
`deploydocs`, from the pages `makedocs` actually built.

The tests live here rather than in the docs build for the reason the capability-catalogue
tests give in `test_26_docs.jl`: a full docs build is slow and run by hand, so a defect
found only there is a defect found months late. The generator takes the build directory as
an argument precisely so it can be driven from a fixture, with no docs environment and no
build.
=#
using Test

include(joinpath(@__DIR__, "..", "docs", "generate_sitemap.jl"))

# A stand-in for `docs/build` after a `prettyurls = true` build: one page per directory
# that holds an `index.html`, the generated search page, and files that are not pages.
function make_fixture_build(dir)
    for page in ["", "capability_catalogue", "search", joinpath("api", "00_API"),
                 joinpath("api", "01_Moments"), joinpath("examples", "1_foundations", "01_first")]
        mkpath(joinpath(dir, page))
        write(joinpath(dir, page, "index.html"), "<html></html>")
    end
    mkpath(joinpath(dir, "assets"))
    write(joinpath(dir, "assets", "logo.svg"), "<svg/>")
    write(joinpath(dir, "search_index.js"), "[]")
    write(joinpath(dir, "objects.inv"), "")
    return dir
end

@testset "Docs sitemap" begin
    base = "https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable"

    @testset "URL set" begin
        mktempdir() do dir
            make_fixture_build(dir)
            urls = sitemap_page_urls(dir, base)

            # The build root is the home page, and every URL ends with a slash.
            @test urls[1] == "$base/"
            @test all(x -> startswith(x, "$base/"), urls)
            @test all(x -> endswith(x, "/"), urls)

            # Every page of the fixture is present, and only those pages.
            @test urls == ["$base/", "$base/api/00_API/", "$base/api/01_Moments/",
                           "$base/capability_catalogue/", "$base/examples/1_foundations/01_first/"]

            # The generated search page is excluded: its content is built in the browser.
            @test !("$base/search/" in urls)

            # A directory with no `index.html` is not a page.
            @test !any(x -> occursin("assets", x), urls)

            # The list is sorted and free of duplicates.
            @test issorted(urls)
            @test allunique(urls)

            # A trailing slash on the base changes nothing.
            @test sitemap_page_urls(dir, base * "/") == urls
        end
    end

    @testset "A new page needs no edit" begin
        mktempdir() do dir
            make_fixture_build(dir)
            before = sitemap_page_urls(dir, base)
            mkpath(joinpath(dir, "api", "99_New"))
            write(joinpath(dir, "api", "99_New", "index.html"), "<html></html>")
            after = sitemap_page_urls(dir, base)
            @test setdiff(after, before) == ["$base/api/99_New/"]
        end
    end

    @testset "XML document" begin
        urls = ["$base/", "$base/api/00_API/"]
        xml = sitemap_xml(urls)
        @test startswith(xml, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
        @test occursin("<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">",
                       xml)
        @test endswith(xml, "</urlset>\n")
        @test count("<url>", xml) == length(urls)
        @test count("</url>", xml) == length(urls)
        for url in urls
            @test occursin("<loc>$url</loc>", xml)
        end
        # No `lastmod`, `changefreq` or `priority`: see the header of the generator.
        @test !occursin("lastmod", xml)
        @test !occursin("changefreq", xml)
        @test !occursin("priority", xml)
        # An empty build writes a well-formed, empty document rather than failing.
        @test sitemap_xml(String[]) ==
              "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">\n</urlset>\n"
    end

    @testset "XML escaping" begin
        @test xml_escape("a&b") == "a&amp;b"
        @test xml_escape("<a>") == "&lt;a&gt;"
        @test xml_escape("\"a\"") == "&quot;a&quot;"
        @test xml_escape("'a'") == "&apos;a&apos;"
        @test xml_escape("plain/path/") == "plain/path/"
        @test occursin("<loc>a&amp;b</loc>", sitemap_xml(["a&b"]))
    end

    @testset "Written file" begin
        mktempdir() do dir
            make_fixture_build(dir)
            out = generate_sitemap(dir, base)
            @test out == joinpath(dir, "sitemap.xml")
            @test isfile(out)
            @test read(out, String) == sitemap_xml(sitemap_page_urls(dir, base))
            # The writer only adds a file; it does not disturb the built pages.
            @test isfile(joinpath(dir, "index.html"))
            @test isfile(joinpath(dir, "search", "index.html"))
        end
    end
end

#=
The ADR index (`docs/adr/README.md`, the P10 review of PR #625). The index is hand-written,
and the page states its own rule: "Add a row here when you add an ADR." ADR 0097 shipped
with no row and nothing turned red, because no test read the index.

The census lives here rather than in the docs build for the reason the sitemap tests give
above: a full docs build is slow and run by hand, and Documenter reports a missing row
never, because a row that was never written breaks no link.

The row's title is not checked against the ADR's own heading. Many rows shorten the heading
on purpose, and that is the index working as intended.
=#
@testset "ADR index" begin
    adr_dir = joinpath(@__DIR__, "..", "docs", "adr")

    # `NNNN-slug.md`, the name every numbered ADR carries. `README.md` and
    # `examples-coverage.md` are not ADRs and match nothing.
    files = Dict{String, String}()
    for file in readdir(adr_dir)
        m = match(r"^(\d{4})-.+\.md$", file)
        isnothing(m) || (files[m.captures[1]] = file)
    end

    # One row of the decisions table: the number, the linked title, the `Am.` column.
    rows = Tuple{String, String}[]
    for line in eachline(joinpath(adr_dir, "README.md"))
        m = match(r"^\| (\d{4}) \| \[.+?\]\((.+?)\) \|", line)
        isnothing(m) || push!(rows, (m.captures[1], m.captures[2]))
    end
    numbers = first.(rows)

    @testset "every ADR carries a row" begin
        unindexed = sort(collect(setdiff(keys(files), numbers)))
        if !isempty(unindexed)
            @warn "ADRs with no row in `docs/adr/README.md`: $(join(unindexed, ", "))"
        end
        @test isempty(unindexed)
    end

    @testset "every row carries an ADR" begin
        # The other direction, as `NOT_A_FEATURE` is checked in `test_26_docs.jl`: a row
        # for a file that a rename or a deletion took away is a dead link on the page.
        orphaned = sort(collect(setdiff(numbers, keys(files))))
        if !isempty(orphaned)
            @warn "Rows of `docs/adr/README.md` with no ADR file: $(join(orphaned, ", "))"
        end
        @test isempty(orphaned)
    end

    @testset "a number is written once, and the rows read in order" begin
        @test allunique(numbers)
        @test issorted(numbers)
    end

    @testset "a row links the file that carries its number" begin
        # A slug changes when a decision is reworded, and the link is the only place the
        # index states the file name.
        stale = sort([n for (n, link) in rows if get(files, n, nothing) != link])
        if !isempty(stale)
            @warn "Rows of `docs/adr/README.md` whose link is not their ADR: $(join(stale, ", "))"
        end
        @test isempty(stale)
    end
end
