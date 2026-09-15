# Auto-generates `sitemap.xml` for the built documentation (issue #568).
#
# Documenter writes no sitemap at any version, and the upstream request is open and
# unimplemented (issue #555). So `docs/make.jl` writes one itself, between `makedocs` and
# `deploydocs`, and `deploydocs` carries it into the deployed tree with the rest of
# `docs/build`.
#
# The page set comes from the build directory, not from a hand-kept list. A page added
# under `docs/src/api/`, a new Literate example, or a new generated page therefore appears
# in the sitemap with no further edit, and a page that no longer builds disappears from it.
#
# Design notes:
#
#   * `prettyurls` is on, so `makedocs` writes every page as `<path>/index.html`. The URL
#     of a page is the directory that holds its `index.html`, with a trailing slash. This
#     is the same form that Documenter puts in each page's `<link rel="canonical">`.
#
#   * Every entry uses the `canonical` base that `make.jl` already holds, so the sitemap
#     lists exactly the URLs the canonical tags point at. Each version of the deployment
#     therefore ships a sitemap of `stable` URLs. Only `.../stable/sitemap.xml` is
#     submitted to Search Console (issue #566).
#
#   * The generated search page is excluded. Its content is built in the browser from
#     `search_index.js`, so the served HTML holds nothing to index.
#
#   * There is no `lastmod`. Most pages here are generated at build time -- the Literate
#     examples and user guide, the type hierarchy, the capability catalogue -- and the API
#     pages render docstrings that live in `src/`. So a build-directory timestamp is the
#     build time for every page, and the git date of a page's source file says nothing
#     about the text a reader sees. A wrong `lastmod` is worse than none, and the sitemap
#     specification makes the field optional.
#
#   * There is no `changefreq` and no `priority`. The major search engines ignore both.
#
#   * The writer only adds a file under `docs/build`, so a local build and a `LiveServer`
#     run are unaffected. `deploydocs` is inert outside CI.

# Escape the five XML entities. A page path is ASCII here, but a `&` in a file name would
# otherwise make the document ill-formed.
function xml_escape(s::AbstractString)
    return replace(s, "&" => "&amp;", "<" => "&lt;", ">" => "&gt;", "\"" => "&quot;",
                   "'" => "&apos;")
end

# Built pages the sitemap omits, each named by its path relative to the build root, in URL
# form.
const SITEMAP_EXCLUDED_PATHS = Set(["search"])

"""
    sitemap_page_urls(build_dir, base) -> Vector{String}

The sorted, unique URLs of every page under `build_dir`, rooted at `base`.

A page is a directory that holds an `index.html`, which is the form `prettyurls = true`
writes. The build root itself is the home page. `base` may carry a trailing slash or not.
"""
function sitemap_page_urls(build_dir::AbstractString, base::AbstractString)
    root_url = rstrip(base, '/')
    urls = String[]
    for (dir, _, files) in walkdir(build_dir)
        if !("index.html" in files)
            continue
        end
        rel = relpath(dir, build_dir)
        if rel == "."
            push!(urls, root_url * "/")
            continue
        end
        path = join(splitpath(rel), "/")
        if path in SITEMAP_EXCLUDED_PATHS
            continue
        end
        push!(urls, root_url * "/" * path * "/")
    end
    return sort!(unique!(urls))
end

"""
    sitemap_xml(urls) -> String

The `urlset` document for `urls`, as a sitemap-protocol 0.9 XML string.
"""
function sitemap_xml(urls::AbstractVector{<:AbstractString})
    io = IOBuffer()
    println(io, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
    println(io, "<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">")
    for url in urls
        println(io, "  <url>")
        println(io, "    <loc>", xml_escape(url), "</loc>")
        println(io, "  </url>")
    end
    println(io, "</urlset>")
    return String(take!(io))
end

"""
    generate_sitemap(build_dir, base) -> String

Write `sitemap.xml` into `build_dir` from the pages that `makedocs` built, and return the
path of the file written. Call it after `makedocs` and before `deploydocs`.
"""
function generate_sitemap(build_dir::AbstractString, base::AbstractString)
    urls = sitemap_page_urls(build_dir, base)
    out = joinpath(build_dir, "sitemap.xml")
    write(out, sitemap_xml(urls))
    @info "docs/generate_sitemap.jl: wrote $(length(urls)) URLs to $out."
    return out
end
