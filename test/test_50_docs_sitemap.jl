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
